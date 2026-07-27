#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np

from _bootstrap import ROOT, load_contract, write_json

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description=(
        "Replay a saved standing MPPI trajectory, then identify the local "
        "front-normal response to bounded single-joint position perturbations."
    )
)
parser.add_argument(
    "--task",
    default="RobotLab-Isaac-BeyondMimic-Flat-pcbC-v1-lateral-bipedal-stand-v0",
)
parser.add_argument(
    "--reference-config",
    default="configs/low_load_lateral/train_001/reference.yaml",
)
parser.add_argument(
    "--episode",
    type=Path,
    required=True,
)
parser.add_argument(
    "--action-candidate-index",
    type=int,
    default=None,
    help=(
        "Select one candidate from a diagnostic trace whose "
        "executed_action16 array has shape [steps,candidates,16]. "
        "Leave unset for ordinary [steps,16] episode archives."
    ),
)
parser.add_argument("--seed", type=int, default=5208)
parser.add_argument("--ref-id", type=int, default=8)
parser.add_argument("--replay-steps", type=int, default=30)
parser.add_argument("--probe-steps", type=int, default=10)
parser.add_argument("--perturbation-rad", type=float, default=0.02)
parser.add_argument("--ramp-steps", type=int, default=2)
parser.add_argument(
    "--physical-target-rate-limit-rad-s",
    type=float,
    default=2.25,
    help=(
        "Apply the same physical joint-target rate projection used by the "
        "formal MPPI expert."
    ),
)
parser.add_argument(
    "--joint-scope",
    choices=("front", "all"),
    default="front",
    help="Probe only front joints (legacy default) or all 12 leg joints.",
)
parser.add_argument(
    "--report",
    type=Path,
    default=ROOT
    / "reports/low_load_lateral/train_001/diagnostics/front_force_response.json",
)
parser.add_argument("--disable-fabric", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


FRONT_JOINTS = (
    (0, "FL_hip"),
    (1, "FR_hip"),
    (4, "FL_thigh"),
    (5, "FR_thigh"),
    (8, "FL_calf"),
    (9, "FR_calf"),
)
ALL_LEG_JOINTS = (
    (0, "FL_hip"),
    (1, "FR_hip"),
    (2, "RL_hip"),
    (3, "RR_hip"),
    (4, "FL_thigh"),
    (5, "FR_thigh"),
    (6, "RL_thigh"),
    (7, "RR_thigh"),
    (8, "FL_calf"),
    (9, "FR_calf"),
    (10, "RL_calf"),
    (11, "RR_calf"),
)


def main() -> dict:
    import gymnasium as gym
    import robot_lab.tasks  # noqa: F401
    import torch
    from isaaclab_tasks.utils import parse_env_cfg

    from lateral_mppi_dagger.contract.action16 import ActionContract
    from lateral_mppi_dagger.env.isaac_adapter import IsaacLateralAdapter
    from lateral_mppi_dagger.env.isaac_mppi_rollout import (
        IsaacMPPIRolloutCloner,
        IsaacRolloutCostWeights,
        _quat_rotation_vector,
    )
    from lateral_mppi_dagger.env.scenarios import (
        configure_env_for_scenario,
        load_scenario_profile,
    )
    from lateral_mppi_dagger.reference.loader import ReferenceSet

    if args_cli.replay_steps < 1 or args_cli.probe_steps < 1:
        raise ValueError("Replay and probe steps must be positive.")
    if args_cli.perturbation_rad <= 0.0:
        raise ValueError("--perturbation-rad must be positive.")
    if args_cli.ramp_steps < 1:
        raise ValueError("--ramp-steps must be positive.")
    if (
        not np.isfinite(args_cli.physical_target_rate_limit_rad_s)
        or args_cli.physical_target_rate_limit_rad_s <= 0.0
    ):
        raise ValueError(
            "--physical-target-rate-limit-rad-s must be finite and positive."
        )

    episode_path = args_cli.episode.resolve()
    with np.load(episode_path, allow_pickle=False) as archive:
        recorded_action = np.asarray(
            archive["executed_action16"],
            dtype=np.float32,
        )
    if recorded_action.ndim == 3:
        if args_cli.action_candidate_index is None:
            raise ValueError(
                "A three-dimensional executed_action16 trace requires "
                "--action-candidate-index."
            )
        if not 0 <= args_cli.action_candidate_index < recorded_action.shape[1]:
            raise ValueError(
                "--action-candidate-index is outside the trace candidate "
                f"range [0,{recorded_action.shape[1] - 1}]."
            )
        recorded_action = recorded_action[
            :,
            args_cli.action_candidate_index,
        ]
    elif recorded_action.ndim == 2:
        if args_cli.action_candidate_index is not None:
            raise ValueError(
                "--action-candidate-index is only valid for a "
                "[steps,candidates,16] trace."
            )
    else:
        raise ValueError(
            "executed_action16 must have shape [steps,16] or "
            "[steps,candidates,16]."
        )
    if recorded_action.shape[1] != 16:
        raise ValueError(
            "executed_action16 must have a final dimension of 16."
        )
    required_steps = args_cli.replay_steps + args_cli.probe_steps
    if recorded_action.shape[0] < required_steps:
        raise ValueError(
            f"Episode has {recorded_action.shape[0]} actions, "
            f"but {required_steps} are required."
        )
    if not np.array_equal(
        recorded_action[:, 12:],
        np.zeros_like(recorded_action[:, 12:]),
    ):
        raise ValueError("Recorded wheel actions are not exactly zero.")

    candidates: list[dict[str, object]] = [
        {
            "name": "baseline",
            "joint_index": None,
            "joint_name": None,
            "physical_offset_rad": 0.0,
        }
    ]
    for replica_index in range(1, 4):
        candidates.append(
            {
                "name": f"baseline_replica_{replica_index}",
                "joint_index": None,
                "joint_name": None,
                "physical_offset_rad": 0.0,
            }
        )
    probe_joints = (
        FRONT_JOINTS
        if args_cli.joint_scope == "front"
        else ALL_LEG_JOINTS
    )
    for joint_index, joint_name in probe_joints:
        for sign in (-1.0, 1.0):
            candidates.append(
                {
                    "name": f"{joint_name}_{'plus' if sign > 0.0 else 'minus'}",
                    "joint_index": joint_index,
                    "joint_name": joint_name,
                    "physical_offset_rad": (
                        sign * args_cli.perturbation_rad
                    ),
                }
            )
    if args_cli.joint_scope == "all":
        combination_offset = 0.5 * args_cli.perturbation_rad
        candidates.extend(
            (
                {
                    "name": "rear_thigh_pair_minus",
                    "joint_index": None,
                    "joint_name": "rear_thigh_pair",
                    "physical_offset_rad": 0.0,
                    "physical_offsets_rad": {
                        6: -combination_offset,
                        7: -combination_offset,
                    },
                },
                {
                    "name": "front_thigh_pair_plus",
                    "joint_index": None,
                    "joint_name": "front_thigh_pair",
                    "physical_offset_rad": 0.0,
                    "physical_offsets_rad": {
                        4: combination_offset,
                        5: combination_offset,
                    },
                },
                {
                    "name": "front_calf_pair_plus",
                    "joint_index": None,
                    "joint_name": "front_calf_pair",
                    "physical_offset_rad": 0.0,
                    "physical_offsets_rad": {
                        8: combination_offset,
                        9: combination_offset,
                    },
                },
                {
                    "name": "front_press_pair_plus",
                    "joint_index": None,
                    "joint_name": "front_thigh_calf_pair",
                    "physical_offset_rad": 0.0,
                    "physical_offsets_rad": {
                        4: combination_offset,
                        5: combination_offset,
                        8: combination_offset,
                        9: combination_offset,
                    },
                },
                {
                    "name": "front_press_pair_full_plus",
                    "joint_index": None,
                    "joint_name": "front_thigh_calf_pair_full",
                    "physical_offset_rad": 0.0,
                    "physical_offsets_rad": {
                        4: args_cli.perturbation_rad,
                        5: args_cli.perturbation_rad,
                        8: args_cli.perturbation_rad,
                        9: args_cli.perturbation_rad,
                    },
                },
                {
                    "name": "front_press_plus_rear_thigh_minus",
                    "joint_index": None,
                    "joint_name": "front_press_rear_load_shift",
                    "physical_offset_rad": 0.0,
                    "physical_offsets_rad": {
                        4: combination_offset,
                        5: combination_offset,
                        6: -combination_offset,
                        7: -combination_offset,
                        8: combination_offset,
                        9: combination_offset,
                    },
                },
                {
                    "name": "front_press_full_plus_rear_thigh_minus",
                    "joint_index": None,
                    "joint_name": "front_press_full_rear_load_shift",
                    "physical_offset_rad": 0.0,
                    "physical_offsets_rad": {
                        4: args_cli.perturbation_rad,
                        5: args_cli.perturbation_rad,
                        6: -args_cli.perturbation_rad,
                        7: -args_cli.perturbation_rad,
                        8: args_cli.perturbation_rad,
                        9: args_cli.perturbation_rad,
                    },
                },
                {
                    "name": "rr_unload_lp_l1_040",
                    "joint_index": None,
                    "joint_name": "rr_unload_lp",
                    "physical_offset_rad": 0.0,
                    "physical_offsets_rad": {
                        0: -0.037,
                        1: -0.120,
                        7: -0.150,
                        11: 0.093,
                    },
                },
                {
                    "name": "rr_unload_lp_l1_070",
                    "joint_index": None,
                    "joint_name": "rr_unload_lp",
                    "physical_offset_rad": 0.0,
                    "physical_offsets_rad": {
                        0: -0.120,
                        1: -0.120,
                        3: 0.121,
                        7: -0.150,
                        11: 0.189,
                    },
                },
                {
                    "name": "rr_unload_lp_l1_100",
                    "joint_index": None,
                    "joint_name": "rr_unload_lp",
                    "physical_offset_rad": 0.0,
                    "physical_offsets_rad": {
                        0: -0.120,
                        1: -0.120,
                        3: 0.300,
                        4: 0.009,
                        7: -0.150,
                        10: -0.101,
                        11: 0.200,
                    },
                },
                {
                    "name": "rr_unload_front_support_lp_l1_120",
                    "joint_index": None,
                    "joint_name": "rr_unload_front_support_lp",
                    "physical_offset_rad": 0.0,
                    "physical_offsets_rad": {
                        0: -0.285,
                        2: 0.251,
                        4: 0.201,
                        5: 0.300,
                        7: -0.129,
                        10: -0.034,
                    },
                },
                {
                    "name": "rr_unload_front_support_small_lp_l1_050",
                    "joint_index": None,
                    "joint_name": "rr_unload_front_support_small_lp",
                    "physical_offset_rad": 0.0,
                    "physical_offsets_rad": {
                        2: 0.017,
                        3: 0.120,
                        4: 0.120,
                        5: 0.120,
                        7: -0.032,
                        8: 0.037,
                        10: -0.054,
                    },
                },
                {
                    "name": "rr_unload_rear_total_small_lp_l1_050",
                    "joint_index": None,
                    "joint_name": "rr_unload_rear_total_small_lp",
                    "physical_offset_rad": 0.0,
                    "physical_offsets_rad": {
                        0: -0.041,
                        3: 0.067,
                        4: 0.120,
                        5: 0.120,
                        7: -0.063,
                        8: 0.063,
                        11: 0.026,
                    },
                },
                {
                    "name": "rr_unload_front_support_medium_lp_l1_080",
                    "joint_index": None,
                    "joint_name": "rr_unload_front_support_medium_lp",
                    "physical_offset_rad": 0.0,
                    "physical_offsets_rad": {
                        2: 0.059,
                        3: 0.180,
                        4: 0.180,
                        5: 0.180,
                        7: -0.055,
                        8: 0.057,
                        10: -0.089,
                    },
                },
                {
                    "name": "rr_unload_yaw_positive_small_lp_l1_0475",
                    "joint_index": None,
                    "joint_name": "rr_unload_yaw_positive_small_lp",
                    "physical_offset_rad": 0.0,
                    "physical_offsets_rad": {
                        0: -0.059,
                        2: -0.120,
                        4: 0.055,
                        6: 0.039,
                        7: -0.107,
                        9: 0.092,
                        11: -0.002,
                    },
                },
                {
                    "name": "rr_unload_yaw_positive_medium_lp_l1_060",
                    "joint_index": None,
                    "joint_name": "rr_unload_yaw_positive_medium_lp",
                    "physical_offset_rad": 0.0,
                    "physical_offsets_rad": {
                        0: -0.080,
                        2: -0.117,
                        4: 0.091,
                        5: 0.029,
                        6: 0.032,
                        7: -0.139,
                        9: 0.112,
                        11: -0.001,
                    },
                },
                {
                    "name": "orientation_reduce_min_lp_l1_0304",
                    "joint_index": None,
                    "joint_name": "orientation_reduce_min_lp",
                    "physical_offset_rad": 0.0,
                    "physical_offsets_rad": {
                        2: -0.060,
                        4: 0.043,
                        7: -0.097,
                        9: 0.099,
                        11: -0.005,
                    },
                },
                {
                    "name": "orientation_reduce_strong_lp_l1_0350",
                    "joint_index": None,
                    "joint_name": "orientation_reduce_strong_lp",
                    "physical_offset_rad": 0.0,
                    "physical_offsets_rad": {
                        0: 0.012,
                        2: -0.042,
                        4: 0.052,
                        7: -0.117,
                        9: 0.128,
                    },
                },
            )
        )
        for tracking_mode in ("selected_target", "measured_error"):
            for tracking_gain in (0.25, 0.5, 1.0):
                candidates.append(
                    {
                        "name": (
                            "rr_swing_hip_reference_"
                            f"{tracking_mode}_g{tracking_gain:g}"
                        ),
                        "joint_index": None,
                        "joint_name": "RR_swing_hip_reference",
                        "physical_offset_rad": 0.0,
                        "reference_tracking_rear_index": 1,
                        "reference_tracking_joint_indices": [3],
                        "reference_tracking_gain": tracking_gain,
                        "reference_tracking_lead_steps": 8,
                        "reference_tracking_mode": tracking_mode,
                    }
                )
        for tracking_gain in (0.5, 1.0):
            candidates.append(
                {
                    "name": (
                        "rr_swing_all_reference_selected_target_"
                        f"g{tracking_gain:g}"
                    ),
                    "joint_index": None,
                    "joint_name": "RR_swing_all_reference",
                    "physical_offset_rad": 0.0,
                    "reference_tracking_rear_index": 1,
                    "reference_tracking_joint_indices": [3, 7, 11],
                    "reference_tracking_gain": tracking_gain,
                    "reference_tracking_lead_steps": 8,
                    "reference_tracking_mode": "selected_target",
                }
            )
        for tracking_mode in ("selected_target", "measured_error"):
            candidates.append(
                {
                    "name": (
                        "rr_swing_hip_reference_"
                        f"{tracking_mode}_g0.5_preload_small"
                    ),
                    "joint_index": None,
                    "joint_name": "RR_swing_hip_reference_preload",
                    "physical_offset_rad": 0.0,
                    "physical_offsets_rad": {
                        2: 0.017,
                        3: 0.120,
                        4: 0.120,
                        5: 0.120,
                        7: -0.032,
                        8: 0.037,
                        10: -0.054,
                    },
                    "reference_tracking_rear_index": 1,
                    "reference_tracking_joint_indices": [3],
                    "reference_tracking_gain": 0.5,
                    "reference_tracking_lead_steps": 8,
                    "reference_tracking_mode": tracking_mode,
                }
            )
        for preload_name, preload_offsets in (
            (
                "small",
                {
                    2: 0.017,
                    3: 0.120,
                    4: 0.120,
                    5: 0.120,
                    7: -0.032,
                    8: 0.037,
                    10: -0.054,
                },
            ),
            (
                "medium",
                {
                    2: 0.059,
                    3: 0.180,
                    4: 0.180,
                    5: 0.180,
                    7: -0.055,
                    8: 0.057,
                    10: -0.089,
                },
            ),
            (
                "large",
                {
                    0: -0.285,
                    2: 0.251,
                    4: 0.201,
                    5: 0.300,
                    7: -0.129,
                    10: -0.034,
                },
            ),
        ):
            candidates.append(
                {
                    "name": (
                        "rr_swing_all_reference_selected_target_g1_"
                        f"preload_{preload_name}"
                    ),
                    "joint_index": None,
                    "joint_name": "RR_swing_all_reference_preload",
                    "physical_offset_rad": 0.0,
                    "physical_offsets_rad": preload_offsets,
                    "reference_tracking_rear_index": 1,
                    "reference_tracking_joint_indices": [3, 7, 11],
                    "reference_tracking_gain": 1.0,
                    "reference_tracking_lead_steps": 8,
                    "reference_tracking_mode": "selected_target",
                }
            )
            candidates.append(
                {
                    "name": (
                        "rr_staged_preload_"
                        f"{preload_name}_then_all_reference_swing_end_g1"
                    ),
                    "joint_index": None,
                    "joint_name": "RR_staged_preload_all_reference",
                    "physical_offset_rad": 0.0,
                    "pre_swing_physical_offsets_rad": preload_offsets,
                    "pre_swing_lead_steps": 9,
                    "reference_tracking_rear_index": 1,
                    "reference_tracking_joint_indices": [3, 7, 11],
                    "reference_tracking_gain": 1.0,
                    "reference_tracking_lead_steps": 0,
                    "reference_tracking_mode": "selected_target",
                    "reference_tracking_gate_mode": "formal_swing",
                }
            )
        for gate_mode, lead_steps in (
            ("formal_swing", 0),
            ("preview", 8),
        ):
            for axis_mode, tracking_gains in (
                ("y", (0.25, 0.5, 1.0)),
                ("yz", (0.5, 1.0)),
            ):
                for tracking_gain in tracking_gains:
                    candidates.append(
                        {
                            "name": (
                                "rr_cartesian_"
                                f"{axis_mode}_{gate_mode}_swing_end_"
                                f"g{tracking_gain:g}"
                            ),
                            "joint_index": None,
                            "joint_name": "RR_cartesian_swing_reference",
                            "physical_offset_rad": 0.0,
                            "cartesian_tracking_rear_index": 1,
                            "cartesian_tracking_axis_mode": axis_mode,
                            "cartesian_tracking_gate_mode": gate_mode,
                            "cartesian_tracking_gain": tracking_gain,
                            "cartesian_tracking_lead_steps": lead_steps,
                            "cartesian_tracking_max_abs_rad": 0.12,
                        }
                    )
        for preload_name, preload_offsets in (
            (
                "small",
                {
                    2: 0.017,
                    3: 0.120,
                    4: 0.120,
                    5: 0.120,
                    7: -0.032,
                    8: 0.037,
                    10: -0.054,
                },
            ),
            (
                "medium",
                {
                    2: 0.059,
                    3: 0.180,
                    4: 0.180,
                    5: 0.180,
                    7: -0.055,
                    8: 0.057,
                    10: -0.089,
                },
            ),
        ):
            for tracking_gain in (0.25, 0.5):
                candidates.append(
                    {
                        "name": (
                            "rr_cartesian_y_preview_swing_end_"
                            f"g{tracking_gain:g}_preload_{preload_name}"
                        ),
                        "joint_index": None,
                        "joint_name": (
                            "RR_cartesian_swing_reference_preload"
                        ),
                        "physical_offset_rad": 0.0,
                        "physical_offsets_rad": preload_offsets,
                        "cartesian_tracking_rear_index": 1,
                        "cartesian_tracking_axis_mode": "y",
                        "cartesian_tracking_gate_mode": "preview",
                        "cartesian_tracking_gain": tracking_gain,
                        "cartesian_tracking_lead_steps": 8,
                        "cartesian_tracking_max_abs_rad": 0.12,
                    }
                )
        candidates.append(
            {
                "name": (
                    "rr_staged_preload_medium_lead20_then_"
                    "cartesian_yz_formal_swing_end_g1"
                ),
                "joint_index": None,
                "joint_name": (
                    "RR_staged_preload_cartesian_swing_reference"
                ),
                "physical_offset_rad": 0.0,
                "pre_swing_physical_offsets_rad": {
                    2: 0.059,
                    3: 0.180,
                    4: 0.180,
                    5: 0.180,
                    7: -0.055,
                    8: 0.057,
                    10: -0.089,
                },
                "pre_swing_lead_steps": 20,
                "cartesian_tracking_rear_index": 1,
                "cartesian_tracking_axis_mode": "yz",
                "cartesian_tracking_gate_mode": "formal_swing",
                "cartesian_tracking_gain": 1.0,
                "cartesian_tracking_lead_steps": 0,
                "cartesian_tracking_max_abs_rad": 0.12,
            }
        )

    contract_dict = load_contract()
    contract = ActionContract.from_dict(contract_dict)
    references = ReferenceSet.from_config(args_cli.reference_config)
    scenario = load_scenario_profile("nominal")
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=len(candidates),
        use_fabric=not args_cli.disable_fabric,
    )
    configure_env_for_scenario(
        env_cfg,
        scenario,
        num_envs=len(candidates),
    )
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device
    env = gym.make(args_cli.task, cfg=env_cfg)
    adapter = IsaacLateralAdapter(
        env,
        references,
        contract_dict,
        scenario_profile=scenario,
    )
    rollout_cloner = IsaacMPPIRolloutCloner(
        adapter,
        references,
        contract,
        horizon=1,
        cost_weights=IsaacRolloutCostWeights(),
    )

    try:
        adapter.reset(args_cli.seed, args_cli.ref_id)
        rollout_cloner.reset_episode_alignment()
        device = adapter.base.device
        candidate_count = len(candidates)
        done_any = torch.zeros(
            candidate_count,
            dtype=torch.bool,
            device=device,
        )

        for step in range(args_cli.replay_steps):
            action = torch.as_tensor(
                recorded_action[step],
                dtype=torch.float32,
                device=device,
            ).repeat(candidate_count, 1)
            _, _, terminated, truncated, _ = env.step(action)
            done = (
                torch.as_tensor(terminated, device=device).reshape(-1)
                | torch.as_tensor(truncated, device=device).reshape(-1)
            )
            done_any |= done
        if bool(done_any.any()):
            raise RuntimeError(
                "A clone terminated during the unperturbed replay prefix."
            )

        snapshot = rollout_cloner.capture()
        rollout_cloner.restore(snapshot)
        state_after_copy = rollout_cloner.state_vector()
        state_copy_pre_probe_spread_max_abs = float(
            torch.max(
                torch.abs(state_after_copy - state_after_copy[0:1]),
            ).item()
        )
        force_at_probe_start = (
            adapter.contact_sensor.data.net_forces_w[
                :,
                adapter.contact_body_ids,
            ]
            .detach()
            .cpu()
            .numpy()
        )
        wheel_position_at_probe_start = (
            adapter.robot.data.body_pos_w[
                :,
                adapter.wheel_body_ids,
            ]
            - adapter.base.scene.env_origins.unsqueeze(1)
        ).detach().cpu().numpy()
        joint_position_at_probe_start = (
            adapter.robot.data.joint_pos[
                :,
                adapter.joint_ids[:12],
            ]
            .detach()
            .cpu()
            .numpy()
        )
        previous_action = torch.as_tensor(
            recorded_action[args_cli.replay_steps - 1],
            dtype=torch.float32,
            device=device,
        ).repeat(candidate_count, 1)
        action_before_probe = previous_action.clone()
        perturbation_physical = torch.zeros(
            (candidate_count, 12),
            dtype=torch.float32,
            device=device,
        )
        pre_swing_perturbation_physical = torch.zeros(
            (candidate_count, 12),
            dtype=torch.float32,
            device=device,
        )
        for candidate_index, candidate in enumerate(candidates):
            joint_index = candidate["joint_index"]
            if joint_index is not None:
                perturbation_physical[
                    candidate_index,
                    int(joint_index),
                ] = float(candidate["physical_offset_rad"])
            for offset_index, offset_value in dict(
                candidate.get("physical_offsets_rad", {})
            ).items():
                perturbation_physical[
                    candidate_index,
                    int(offset_index),
                ] = float(offset_value)
            for offset_index, offset_value in dict(
                candidate.get("pre_swing_physical_offsets_rad", {})
            ).items():
                pre_swing_perturbation_physical[
                    candidate_index,
                    int(offset_index),
                ] = float(offset_value)
        scale = torch.as_tensor(
            contract.scale[:12],
            dtype=torch.float32,
            device=device,
        )
        perturbation_raw = perturbation_physical / scale
        pre_swing_perturbation_raw = (
            pre_swing_perturbation_physical / scale
        )
        raw_min = torch.as_tensor(
            contract.raw_min,
            dtype=torch.float32,
            device=device,
        )
        raw_max = torch.as_tensor(
            contract.raw_max,
            dtype=torch.float32,
            device=device,
        )
        max_raw_delta = torch.as_tensor(
            contract.max_raw_delta_per_step,
            dtype=torch.float32,
            device=device,
        )
        maximum_physical_delta = (
            args_cli.physical_target_rate_limit_rad_s
            * adapter.control_dt
        )
        max_raw_delta = torch.minimum(
            max_raw_delta,
            torch.as_tensor(
                maximum_physical_delta / contract.scale,
                dtype=torch.float32,
                device=device,
            ),
        )
        force_samples: list[np.ndarray] = []
        wheel_position_samples: list[np.ndarray] = []
        base_position_samples: list[np.ndarray] = []
        orientation_error_samples: list[np.ndarray] = []
        orientation_rotation_vector_samples: list[np.ndarray] = []
        action_samples: list[np.ndarray] = []
        joint_position_samples: list[np.ndarray] = []
        joint_velocity_samples: list[np.ndarray] = []
        applied_torque_samples: list[np.ndarray] = []
        effort_limit_samples: list[np.ndarray] = []

        for probe_step in range(args_cli.probe_steps):
            baseline = torch.as_tensor(
                recorded_action[args_cli.replay_steps + probe_step],
                dtype=torch.float32,
                device=device,
            ).repeat(candidate_count, 1)
            ramp = min(
                (probe_step + 1) / args_cli.ramp_steps,
                1.0,
            )
            desired = baseline.clone()
            reference_frame = min(
                args_cli.replay_steps + probe_step,
                references[args_cli.ref_id].frames - 1,
            )
            for candidate_index, candidate in enumerate(candidates):
                if "pre_swing_physical_offsets_rad" in candidate:
                    rear_index = int(
                        candidate.get(
                            "reference_tracking_rear_index",
                            candidate.get(
                                "cartesian_tracking_rear_index",
                            ),
                        )
                    )
                    pre_swing_lead_steps = int(
                        candidate["pre_swing_lead_steps"]
                    )
                    pre_swing_target_frame = min(
                        reference_frame + pre_swing_lead_steps,
                        references[args_cli.ref_id].frames - 1,
                    )
                    rear_schedule = adapter.contact_schedules[
                        args_cli.ref_id
                    ][
                        reference_frame : pre_swing_target_frame + 1,
                        2 + rear_index,
                    ]
                    currently_scheduled_support = bool(rear_schedule[0])
                    upcoming_swing = bool(np.any(~rear_schedule))
                    if currently_scheduled_support and upcoming_swing:
                        desired[candidate_index, :12] += (
                            ramp
                            * pre_swing_perturbation_raw[candidate_index]
                        )
                if "cartesian_tracking_rear_index" in candidate:
                    rear_index = int(
                        candidate["cartesian_tracking_rear_index"]
                    )
                    gate_mode = str(
                        candidate["cartesian_tracking_gate_mode"]
                    )
                    lead_steps = int(
                        candidate["cartesian_tracking_lead_steps"]
                    )
                    schedule = adapter.contact_schedules[
                        args_cli.ref_id
                    ]
                    preview_stop = min(
                        reference_frame + lead_steps,
                        references[args_cli.ref_id].frames - 1,
                    )
                    if gate_mode == "formal_swing":
                        planned_swing = not bool(
                            schedule[
                                reference_frame,
                                2 + rear_index,
                            ]
                        )
                        first_swing_frame = reference_frame
                    elif gate_mode == "preview":
                        preview = np.asarray(
                            schedule[
                                reference_frame : preview_stop + 1,
                                2 + rear_index,
                            ],
                            dtype=bool,
                        )
                        swing_offsets = np.flatnonzero(~preview)
                        planned_swing = swing_offsets.size > 0
                        first_swing_frame = (
                            reference_frame
                            if not planned_swing
                            else reference_frame
                            + int(swing_offsets[0])
                        )
                    else:
                        raise ValueError(
                            "Unknown cartesian_tracking_gate_mode "
                            f"{gate_mode!r}."
                        )
                    if planned_swing:
                        target_frame = first_swing_frame
                        while (
                            target_frame + 1
                            < references[args_cli.ref_id].frames
                            and not bool(
                                schedule[
                                    target_frame + 1,
                                    2 + rear_index,
                                ]
                            )
                        ):
                            target_frame += 1
                        target = rollout_cloner._aligned_reference(
                            snapshot,
                            target_frame,
                        )
                        wheel_index = 2 + rear_index
                        actual_wheel_position = (
                            adapter.robot.data.body_pos_w[
                                candidate_index,
                                adapter.wheel_body_ids[wheel_index],
                            ]
                            - adapter.base.scene.env_origins[
                                candidate_index
                            ]
                        )
                        target_wheel_position = target[
                            "body_pos_local"
                        ][
                            rollout_cloner.ref_wheel_body_ids[wheel_index]
                        ]
                        cartesian_error = (
                            target_wheel_position
                            - actual_wheel_position
                        )
                        axis_mode = str(
                            candidate["cartesian_tracking_axis_mode"]
                        )
                        desired_cartesian_delta = torch.zeros(
                            3,
                            dtype=torch.float32,
                            device=device,
                        )
                        tracking_gain = float(
                            candidate["cartesian_tracking_gain"]
                        )
                        desired_cartesian_delta[1] = (
                            tracking_gain * cartesian_error[1]
                        )
                        if axis_mode == "y":
                            axis_weights = torch.tensor(
                                (0.0, 1.0, 0.0),
                                dtype=torch.float32,
                                device=device,
                            )
                        elif axis_mode == "yz":
                            desired_cartesian_delta[2] = (
                                tracking_gain * cartesian_error[2]
                            )
                            axis_weights = torch.tensor(
                                (0.0, 1.0, 1.0),
                                dtype=torch.float32,
                                device=device,
                            )
                        else:
                            raise ValueError(
                                "Unknown cartesian_tracking_axis_mode "
                                f"{axis_mode!r}."
                            )
                        joint_indices = (3, 7, 11)
                        robot = adapter.robot
                        jacobians = (
                            robot.root_physx_view.get_jacobians()
                        )
                        is_fixed_base = bool(
                            getattr(robot, "is_fixed_base", False)
                        )
                        joint_column_offset = (
                            0 if is_fixed_base else 6
                        )
                        body_id = adapter.wheel_body_ids[wheel_index]
                        jacobian_body_id = (
                            body_id - 1 if is_fixed_base else body_id
                        )
                        joint_columns = torch.as_tensor(
                            [
                                adapter.joint_ids[index]
                                + joint_column_offset
                                for index in joint_indices
                            ],
                            dtype=torch.long,
                            device=jacobians.device,
                        )
                        wheel_jacobian = jacobians[
                            candidate_index,
                            jacobian_body_id,
                            :3,
                        ].index_select(-1, joint_columns)
                        weighted_jacobian = (
                            axis_weights.unsqueeze(-1)
                            * wheel_jacobian
                        )
                        damping = torch.as_tensor(
                            1.0e-3,
                            dtype=jacobians.dtype,
                            device=jacobians.device,
                        )
                        normal_matrix = (
                            weighted_jacobian.transpose(0, 1)
                            @ weighted_jacobian
                            + damping
                            * torch.eye(
                                3,
                                dtype=jacobians.dtype,
                                device=jacobians.device,
                            )
                        )
                        joint_delta = torch.linalg.solve(
                            normal_matrix,
                            weighted_jacobian.transpose(0, 1)
                            @ (
                                axis_weights
                                * desired_cartesian_delta
                            ),
                        )
                        maximum_joint_delta = float(
                            candidate[
                                "cartesian_tracking_max_abs_rad"
                            ]
                        )
                        joint_delta = torch.clamp(
                            joint_delta,
                            min=-maximum_joint_delta,
                            max=maximum_joint_delta,
                        )
                        joint_index_tensor = torch.as_tensor(
                            joint_indices,
                            dtype=torch.long,
                            device=device,
                        )
                        measured_q = adapter.robot.data.joint_pos[
                            candidate_index,
                            adapter.joint_ids[:12],
                        ]
                        selected_q = (
                            torch.as_tensor(
                                contract.q_action_offset_runtime[:12],
                                dtype=torch.float32,
                                device=device,
                            )
                            + scale * desired[candidate_index, :12]
                        )
                        requested_q = measured_q[
                            joint_index_tensor
                        ] + joint_delta.to(
                            dtype=torch.float32,
                            device=device,
                        )
                        requested_q = torch.where(
                            joint_delta > 0.0,
                            torch.maximum(
                                selected_q[joint_index_tensor],
                                requested_q,
                            ),
                            requested_q,
                        )
                        requested_q = torch.where(
                            joint_delta < 0.0,
                            torch.minimum(
                                selected_q[joint_index_tensor],
                                requested_q,
                            ),
                            requested_q,
                        )
                        desired[
                            candidate_index,
                            joint_index_tensor,
                        ] = (
                            requested_q
                            - torch.as_tensor(
                                contract.q_action_offset_runtime[:12],
                                dtype=torch.float32,
                                device=device,
                            )[joint_index_tensor]
                        ) / scale[joint_index_tensor]
                if "reference_tracking_rear_index" not in candidate:
                    continue
                rear_index = int(
                    candidate["reference_tracking_rear_index"]
                )
                lead_steps = int(
                    candidate["reference_tracking_lead_steps"]
                )
                gate_mode = str(
                    candidate.get("reference_tracking_gate_mode", "lead")
                )
                target_frame = min(
                    reference_frame + lead_steps,
                    references[args_cli.ref_id].frames - 1,
                )
                if gate_mode == "lead":
                    scheduled_swing = bool(
                        np.any(
                            ~adapter.contact_schedules[args_cli.ref_id][
                                reference_frame : target_frame + 1,
                                2 + rear_index,
                            ]
                        )
                    )
                elif gate_mode == "formal_swing":
                    scheduled_swing = bool(
                        ~adapter.contact_schedules[args_cli.ref_id][
                            reference_frame,
                            2 + rear_index,
                        ]
                    )
                    if scheduled_swing:
                        target_frame = reference_frame
                        while (
                            target_frame + 1
                            < references[args_cli.ref_id].frames
                            and not bool(
                                adapter.contact_schedules[args_cli.ref_id][
                                    target_frame + 1,
                                    2 + rear_index,
                                ]
                            )
                        ):
                            target_frame += 1
                else:
                    raise ValueError(
                        "Unknown reference_tracking_gate_mode "
                        f"{gate_mode!r}."
                    )
                if not scheduled_swing:
                    continue
                joint_indices = [
                    int(index)
                    for index in candidate[
                        "reference_tracking_joint_indices"
                    ]
                ]
                reference_q = torch.as_tensor(
                    references[args_cli.ref_id].joint_pos[
                        target_frame,
                        :12,
                    ],
                    dtype=torch.float32,
                    device=device,
                )
                selected_q = (
                    torch.as_tensor(
                        contract.q_action_offset_runtime[:12],
                        dtype=torch.float32,
                        device=device,
                    )
                    + scale * desired[candidate_index, :12]
                )
                tracking_gain = float(
                    candidate["reference_tracking_gain"]
                )
                tracking_mode = str(
                    candidate["reference_tracking_mode"]
                )
                if tracking_mode == "selected_target":
                    requested_q = selected_q + tracking_gain * (
                        reference_q - selected_q
                    )
                elif tracking_mode == "measured_error":
                    measured_q = adapter.robot.data.joint_pos[
                        candidate_index,
                        adapter.joint_ids[:12],
                    ]
                    requested_q = selected_q + tracking_gain * (
                        reference_q - measured_q
                    )
                else:
                    raise ValueError(
                        "Unknown reference_tracking_mode "
                        f"{tracking_mode!r}."
                    )
                desired[
                    candidate_index,
                    joint_indices,
                ] = (
                    requested_q[joint_indices]
                    - torch.as_tensor(
                        contract.q_action_offset_runtime[:12],
                        dtype=torch.float32,
                        device=device,
                    )[joint_indices]
                ) / scale[joint_indices]
            desired[:, :12] += ramp * perturbation_raw
            desired[:, 12:].zero_()
            desired = torch.maximum(
                torch.minimum(desired, previous_action + max_raw_delta),
                previous_action - max_raw_delta,
            )
            desired = torch.maximum(
                torch.minimum(desired, raw_max),
                raw_min,
            )
            desired[:, 12:].zero_()
            _, _, terminated, truncated, _ = env.step(desired)
            done = (
                torch.as_tensor(terminated, device=device).reshape(-1)
                | torch.as_tensor(truncated, device=device).reshape(-1)
            )
            done_any |= done
            previous_action = desired

            force_samples.append(
                adapter.contact_sensor.data.net_forces_w[
                    :,
                    adapter.contact_body_ids,
                ]
                .detach()
                .cpu()
                .numpy()
            )
            wheel_position_samples.append(
                (
                    adapter.robot.data.body_pos_w[
                        :,
                        adapter.wheel_body_ids,
                    ]
                    - adapter.base.scene.env_origins.unsqueeze(1)
                )
                .detach()
                .cpu()
                .numpy()
            )
            base_position_samples.append(
                (
                    adapter.command.robot_anchor_pos_w
                    - adapter.base.scene.env_origins
                )
                .detach()
                .cpu()
                .numpy()
            )
            frame = min(
                int(adapter.command.time_steps[0].item()),
                references[args_cli.ref_id].frames - 1,
            )
            target_quaternion = torch.as_tensor(
                references[args_cli.ref_id].body_quat_w[frame, 0],
                dtype=torch.float32,
                device=device,
            )
            quaternion_dot = torch.abs(
                torch.sum(
                    adapter.command.robot_anchor_quat_w
                    * target_quaternion.unsqueeze(0),
                    dim=-1,
                )
            ).clamp(max=1.0)
            orientation_error_samples.append(
                (2.0 * torch.acos(quaternion_dot)).detach().cpu().numpy()
            )
            orientation_rotation_vector_samples.append(
                _quat_rotation_vector(
                    adapter.command.robot_anchor_quat_w,
                    target_quaternion.unsqueeze(0),
                )
                .detach()
                .cpu()
                .numpy()
            )
            action_samples.append(desired.detach().cpu().numpy())
            joint_position_samples.append(
                adapter.robot.data.joint_pos[
                    :,
                    adapter.joint_ids[:12],
                ]
                .detach()
                .cpu()
                .numpy()
            )
            joint_velocity_samples.append(
                adapter.robot.data.joint_vel[
                    :,
                    adapter.joint_ids[:12],
                ]
                .detach()
                .cpu()
                .numpy()
            )
            applied_torque_samples.append(
                adapter.robot.data.applied_torque[
                    :,
                    adapter.joint_ids[:12],
                ]
                .detach()
                .cpu()
                .numpy()
            )
            effort_limit_samples.append(
                adapter.robot.data.joint_effort_limits[
                    :,
                    adapter.joint_ids[:12],
                ]
                .detach()
                .cpu()
                .numpy()
            )

        force = np.stack(force_samples)
        wheel_position = np.stack(wheel_position_samples)
        front_normal = np.abs(force[..., :2, 0])
        front_vertical = force[..., :2, 2]
        rear_normal = np.abs(force[..., 2:, 2])
        contact_force_norm = np.linalg.norm(force, axis=-1)
        base_position = np.stack(base_position_samples)
        orientation_error = np.stack(orientation_error_samples)
        orientation_rotation_vector = np.stack(
            orientation_rotation_vector_samples
        )
        actions = np.stack(action_samples)
        joint_position = np.stack(joint_position_samples)
        joint_velocity = np.stack(joint_velocity_samples)
        applied_torque = np.stack(applied_torque_samples)
        effort_limit = np.stack(effort_limit_samples)
        physical_joint_target = (
            np.asarray(
                contract.q_action_offset_runtime[:12],
                dtype=np.float32,
            )[None, None, :]
            + actions[..., :12]
            * np.asarray(contract.scale[:12], dtype=np.float32)[
                None,
                None,
                :,
            ]
        )
        joint_tracking_error = physical_joint_target - joint_position
        torque_utilization = np.abs(applied_torque) / np.maximum(
            effort_limit,
            1.0e-6,
        )
        baseline_indices = tuple(
            index
            for index, candidate in enumerate(candidates)
            if str(candidate["name"]).startswith("baseline")
        )
        baseline_force = np.mean(
            front_normal[:, baseline_indices],
            axis=1,
        )
        baseline_rear_force = np.mean(
            rear_normal[:, baseline_indices],
            axis=1,
        )
        baseline_front_vertical = np.mean(
            front_vertical[:, baseline_indices],
            axis=1,
        )
        baseline_wheel_position = np.mean(
            wheel_position[:, baseline_indices],
            axis=1,
        )
        baseline_contact_force_norm = np.mean(
            contact_force_norm[:, baseline_indices],
            axis=1,
        )
        baseline_base_position = np.mean(
            base_position[:, baseline_indices],
            axis=1,
        )
        baseline_orientation_rotation_vector = np.mean(
            orientation_rotation_vector[:, baseline_indices],
            axis=1,
        )
        baseline_joint_position = np.mean(
            joint_position[:, baseline_indices],
            axis=1,
        )
        baseline_joint_velocity = np.mean(
            joint_velocity[:, baseline_indices],
            axis=1,
        )
        baseline_applied_torque = np.mean(
            applied_torque[:, baseline_indices],
            axis=1,
        )
        baseline_torque_utilization = np.mean(
            torque_utilization[:, baseline_indices],
            axis=1,
        )
        records = []
        for candidate_index, candidate in enumerate(candidates):
            joint_name = candidate["joint_name"]
            affected_wheel_index = None
            if joint_name is not None:
                for prefix, wheel_index in (
                    ("FL", 0),
                    ("FR", 1),
                    ("RL", 2),
                    ("RR", 3),
                ):
                    if str(joint_name).startswith(prefix):
                        affected_wheel_index = wheel_index
                        break
            affected_front_wheel_index = (
                affected_wheel_index
                if affected_wheel_index in (0, 1)
                else None
            )
            candidate_force = front_normal[:, candidate_index]
            candidate_wheel_position = wheel_position[:, candidate_index]
            wheel_position_delta = (
                candidate_wheel_position - baseline_wheel_position
            )
            candidate_contact_force_norm = (
                contact_force_norm[:, candidate_index]
            )
            candidate_joint_position = joint_position[:, candidate_index]
            candidate_joint_velocity = joint_velocity[:, candidate_index]
            candidate_applied_torque = applied_torque[:, candidate_index]
            candidate_effort_limit = effort_limit[:, candidate_index]
            candidate_torque_utilization = torque_utilization[
                :,
                candidate_index,
            ]
            candidate_joint_tracking_error = joint_tracking_error[
                :,
                candidate_index,
            ]
            physical_step = (
                np.diff(
                    np.concatenate(
                        (
                            action_before_probe[
                                candidate_index,
                                :12,
                            ]
                            .detach()
                            .cpu()
                            .numpy()[None],
                            actions[:, candidate_index, :12],
                        ),
                        axis=0,
                    ),
                    axis=0,
                )
                * contract.scale[:12]
            )
            records.append(
                {
                    **candidate,
                    "affected_wheel_index": affected_wheel_index,
                    "affected_front_wheel_index": (
                        affected_front_wheel_index
                    ),
                    "front_normal_start_n": np.abs(
                        force_at_probe_start[candidate_index, :2, 0]
                    ).tolist(),
                    "front_normal_mean_n": np.mean(
                        candidate_force,
                        axis=0,
                    ).tolist(),
                    "front_normal_last5_mean_n": np.mean(
                        candidate_force[-min(5, args_cli.probe_steps) :],
                        axis=0,
                    ).tolist(),
                    "front_normal_delta_vs_baseline_mean_n": np.mean(
                        candidate_force - baseline_force,
                        axis=0,
                    ).tolist(),
                    "front_vertical_support_mean_n": np.mean(
                        front_vertical[:, candidate_index],
                        axis=0,
                    ).tolist(),
                    "front_vertical_support_delta_vs_baseline_mean_n": (
                        np.mean(
                            front_vertical[:, candidate_index]
                            - baseline_front_vertical,
                            axis=0,
                        ).tolist()
                    ),
                    "rear_normal_start_n": np.abs(
                        force_at_probe_start[candidate_index, 2:, 2]
                    ).tolist(),
                    "rear_normal_mean_n": np.mean(
                        rear_normal[:, candidate_index],
                        axis=0,
                    ).tolist(),
                    "rear_normal_last5_mean_n": np.mean(
                        rear_normal[
                            -min(5, args_cli.probe_steps) :,
                            candidate_index,
                        ],
                        axis=0,
                    ).tolist(),
                    "rear_normal_delta_vs_baseline_mean_n": np.mean(
                        rear_normal[:, candidate_index]
                        - baseline_rear_force,
                        axis=0,
                    ).tolist(),
                    "wheel_position_delta_vs_baseline_end_m": (
                        wheel_position_delta[-1].tolist()
                    ),
                    "wheel_position_delta_vs_baseline_max_abs_m": np.max(
                        np.abs(wheel_position_delta),
                        axis=0,
                    ).tolist(),
                    "wheel_vertical_delta_vs_baseline_max_m": np.max(
                        wheel_position_delta[..., 2],
                        axis=0,
                    ).tolist(),
                    "wheel_vertical_delta_vs_baseline_min_m": np.min(
                        wheel_position_delta[..., 2],
                        axis=0,
                    ).tolist(),
                    "wheel_vertical_displacement_from_probe_start_max_m": (
                        np.max(
                            candidate_wheel_position[..., 2]
                            - wheel_position_at_probe_start[
                                candidate_index,
                                :,
                                2,
                            ],
                            axis=0,
                        ).tolist()
                    ),
                    "contact_force_norm_mean_n": np.mean(
                        candidate_contact_force_norm,
                        axis=0,
                    ).tolist(),
                    "contact_force_norm_delta_vs_baseline_mean_n": np.mean(
                        candidate_contact_force_norm
                        - baseline_contact_force_norm,
                        axis=0,
                    ).tolist(),
                    "measured_contact_fraction_at_8n": np.mean(
                        candidate_contact_force_norm >= 8.0,
                        axis=0,
                    ).tolist(),
                    "affected_wheel_delta_vs_baseline_mean_n": (
                        None
                        if affected_front_wheel_index is None
                        else float(
                            np.mean(
                                candidate_force[
                                    :,
                                    affected_front_wheel_index,
                                ]
                                - baseline_force[
                                    :,
                                    affected_front_wheel_index,
                                ]
                            )
                        )
                    ),
                    "base_position_delta_vs_baseline_max_abs_m": np.max(
                        np.abs(
                            base_position[:, candidate_index]
                            - baseline_base_position
                        ),
                        axis=0,
                    ).tolist(),
                    "base_position_delta_vs_baseline_end_m": (
                        base_position[-1, candidate_index]
                        - baseline_base_position[-1]
                    ).tolist(),
                    "base_orientation_error_rmse_rad": float(
                        np.sqrt(
                            np.mean(
                                orientation_error[:, candidate_index] ** 2
                            )
                        )
                    ),
                    "orientation_rotation_vector_mean_rad": np.mean(
                        orientation_rotation_vector[:, candidate_index],
                        axis=0,
                    ).tolist(),
                    "orientation_rotation_vector_rmse_rad": np.sqrt(
                        np.mean(
                            np.square(
                                orientation_rotation_vector[
                                    :,
                                    candidate_index,
                                ]
                            ),
                            axis=0,
                        )
                    ).tolist(),
                    "orientation_rotation_vector_delta_vs_baseline_mean_rad": (
                        np.mean(
                            orientation_rotation_vector[
                                :,
                                candidate_index,
                            ]
                            - baseline_orientation_rotation_vector,
                            axis=0,
                        ).tolist()
                    ),
                    "joint_position_start_rad": (
                        joint_position_at_probe_start[candidate_index].tolist()
                    ),
                    "joint_position_delta_vs_baseline_end_rad": (
                        candidate_joint_position[-1]
                        - baseline_joint_position[-1]
                    ).tolist(),
                    "joint_position_delta_vs_baseline_max_abs_rad": np.max(
                        np.abs(
                            candidate_joint_position
                            - baseline_joint_position
                        ),
                        axis=0,
                    ).tolist(),
                    "joint_velocity_abs_mean_rad_s": np.mean(
                        np.abs(candidate_joint_velocity),
                        axis=0,
                    ).tolist(),
                    "joint_velocity_delta_vs_baseline_mean_rad_s": np.mean(
                        candidate_joint_velocity - baseline_joint_velocity,
                        axis=0,
                    ).tolist(),
                    "physical_joint_target_mean_rad": np.mean(
                        physical_joint_target[:, candidate_index],
                        axis=0,
                    ).tolist(),
                    "joint_tracking_error_mean_rad": np.mean(
                        candidate_joint_tracking_error,
                        axis=0,
                    ).tolist(),
                    "joint_tracking_error_abs_mean_rad": np.mean(
                        np.abs(candidate_joint_tracking_error),
                        axis=0,
                    ).tolist(),
                    "joint_tracking_error_max_abs_rad": np.max(
                        np.abs(candidate_joint_tracking_error),
                        axis=0,
                    ).tolist(),
                    "applied_torque_mean_nm": np.mean(
                        candidate_applied_torque,
                        axis=0,
                    ).tolist(),
                    "applied_torque_abs_mean_nm": np.mean(
                        np.abs(candidate_applied_torque),
                        axis=0,
                    ).tolist(),
                    "applied_torque_max_abs_nm": np.max(
                        np.abs(candidate_applied_torque),
                        axis=0,
                    ).tolist(),
                    "applied_torque_delta_vs_baseline_mean_nm": np.mean(
                        candidate_applied_torque
                        - baseline_applied_torque,
                        axis=0,
                    ).tolist(),
                    "effort_limit_min_nm": np.min(
                        candidate_effort_limit,
                        axis=0,
                    ).tolist(),
                    "torque_utilization_mean": np.mean(
                        candidate_torque_utilization,
                        axis=0,
                    ).tolist(),
                    "torque_utilization_max": np.max(
                        candidate_torque_utilization,
                        axis=0,
                    ).tolist(),
                    "torque_utilization_delta_vs_baseline_mean": np.mean(
                        candidate_torque_utilization
                        - baseline_torque_utilization,
                        axis=0,
                    ).tolist(),
                    "torque_saturation_fraction_at_95pct": np.mean(
                        candidate_torque_utilization >= 0.95,
                        axis=0,
                    ).tolist(),
                    "torque_saturation_fraction_at_99pct": np.mean(
                        candidate_torque_utilization >= 0.99,
                        axis=0,
                    ).tolist(),
                    "physical_action_step_max_abs_rad": float(
                        np.max(np.abs(physical_step))
                        if physical_step.size
                        else 0.0
                    ),
                    "terminated": bool(done_any[candidate_index].item()),
                    "wheel_action_exact_zero": bool(
                        np.array_equal(
                            actions[:, candidate_index, 12:],
                            np.zeros_like(
                                actions[:, candidate_index, 12:]
                            ),
                        )
                    ),
                }
            )

        report = {
            "schema_version": "pcbc-front-force-local-response-v9",
            "status": "diagnostic_not_training_data",
            "task": args_cli.task,
            "reference_config": args_cli.reference_config,
            "source_episode": str(episode_path),
            "action_candidate_index": args_cli.action_candidate_index,
            "seed": args_cli.seed,
            "ref_id": args_cli.ref_id,
            "replay_steps": args_cli.replay_steps,
            "probe_steps": args_cli.probe_steps,
            "perturbation_rad": args_cli.perturbation_rad,
            "ramp_steps": args_cli.ramp_steps,
            "physical_target_rate_limit_rad_s": (
                args_cli.physical_target_rate_limit_rad_s
            ),
            "physical_target_step_limit_rad": maximum_physical_delta,
            "joint_scope": args_cli.joint_scope,
            "candidate_count": candidate_count,
            "baseline_candidate_count": len(baseline_indices),
            "state_copy_applied_before_probe": True,
            "state_copy_pre_probe_spread_max_abs": (
                state_copy_pre_probe_spread_max_abs
            ),
            "baseline_replica_spread": {
                "wheel_position_max_abs_m": float(
                    np.max(
                        np.abs(
                            wheel_position[:, baseline_indices]
                            - baseline_wheel_position[:, None]
                        )
                    )
                ),
                "contact_force_norm_max_abs_n": float(
                    np.max(
                        np.abs(
                            contact_force_norm[:, baseline_indices]
                            - baseline_contact_force_norm[:, None]
                        )
                    )
                ),
                "base_position_max_abs_m": float(
                    np.max(
                        np.abs(
                            base_position[:, baseline_indices]
                            - baseline_base_position[:, None]
                        )
                    )
                ),
                "orientation_rotation_vector_max_abs_rad": float(
                    np.max(
                        np.abs(
                            orientation_rotation_vector[
                                :,
                                baseline_indices,
                            ]
                            - baseline_orientation_rotation_vector[:, None]
                        )
                    )
                ),
                "joint_position_max_abs_rad": float(
                    np.max(
                        np.abs(
                            joint_position[:, baseline_indices]
                            - baseline_joint_position[:, None]
                        )
                    )
                ),
                "joint_velocity_max_abs_rad_s": float(
                    np.max(
                        np.abs(
                            joint_velocity[:, baseline_indices]
                            - baseline_joint_velocity[:, None]
                        )
                    )
                ),
                "applied_torque_max_abs_nm": float(
                    np.max(
                        np.abs(
                            applied_torque[:, baseline_indices]
                            - baseline_applied_torque[:, None]
                        )
                    )
                ),
                "torque_utilization_max_abs": float(
                    np.max(
                        np.abs(
                            torque_utilization[:, baseline_indices]
                            - baseline_torque_utilization[:, None]
                        )
                    )
                ),
            },
            "joint_order": [name for _, name in ALL_LEG_JOINTS],
            "torque_saturation_thresholds": [0.95, 0.99],
            "contact_force_threshold_n": 8.0,
            "records": records,
        }
        write_json(args_cli.report, report)
        for record in records:
            print(record, flush=True)
        return report
    finally:
        adapter.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        write_json(
            args_cli.report.with_suffix(
                args_cli.report.suffix + ".failure.json"
            ),
            {
                "error_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
                "arguments": vars(args_cli),
            },
        )
        traceback.print_exc()
        sys.stderr.flush()
        raise
    finally:
        simulation_app.close()
