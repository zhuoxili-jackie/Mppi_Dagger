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
        candidates.extend(
            (
                {
                    "name": "rear_thigh_pair_minus",
                    "joint_index": None,
                    "joint_name": "rear_thigh_pair",
                    "physical_offset_rad": 0.0,
                    "physical_offsets_rad": {6: -0.02, 7: -0.02},
                },
                {
                    "name": "front_thigh_pair_plus",
                    "joint_index": None,
                    "joint_name": "front_thigh_pair",
                    "physical_offset_rad": 0.0,
                    "physical_offsets_rad": {4: 0.02, 5: 0.02},
                },
                {
                    "name": "front_calf_pair_plus",
                    "joint_index": None,
                    "joint_name": "front_calf_pair",
                    "physical_offset_rad": 0.0,
                    "physical_offsets_rad": {8: 0.02, 9: 0.02},
                },
                {
                    "name": "front_press_pair_plus",
                    "joint_index": None,
                    "joint_name": "front_thigh_calf_pair",
                    "physical_offset_rad": 0.0,
                    "physical_offsets_rad": {
                        4: 0.02,
                        5: 0.02,
                        8: 0.02,
                        9: 0.02,
                    },
                },
                {
                    "name": "front_press_plus_rear_thigh_minus",
                    "joint_index": None,
                    "joint_name": "front_press_rear_load_shift",
                    "physical_offset_rad": 0.0,
                    "physical_offsets_rad": {
                        4: 0.02,
                        5: 0.02,
                        6: -0.02,
                        7: -0.02,
                        8: 0.02,
                        9: 0.02,
                    },
                },
            )
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

    try:
        adapter.reset(args_cli.seed, args_cli.ref_id)
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

        force_at_probe_start = (
            adapter.contact_sensor.data.net_forces_w[
                :,
                adapter.contact_body_ids[:2],
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
        perturbation_physical = torch.zeros(
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
        scale = torch.as_tensor(
            contract.scale[:12],
            dtype=torch.float32,
            device=device,
        )
        perturbation_raw = perturbation_physical / scale
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
        force_samples: list[np.ndarray] = []
        base_position_samples: list[np.ndarray] = []
        orientation_error_samples: list[np.ndarray] = []
        action_samples: list[np.ndarray] = []

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
                    adapter.contact_body_ids[:2],
                ]
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
            action_samples.append(desired.detach().cpu().numpy())

        force = np.stack(force_samples)
        front_normal = np.abs(force[..., 0])
        base_position = np.stack(base_position_samples)
        orientation_error = np.stack(orientation_error_samples)
        actions = np.stack(action_samples)
        baseline_force = front_normal[:, 0]
        records = []
        for candidate_index, candidate in enumerate(candidates):
            joint_name = candidate["joint_name"]
            affected_wheel = (
                None
                if joint_name is None
                or not str(joint_name).startswith(("FL", "FR"))
                else (0 if str(joint_name).startswith("FL") else 1)
            )
            candidate_force = front_normal[:, candidate_index]
            physical_step = (
                np.diff(actions[:, candidate_index, :12], axis=0)
                * contract.scale[:12]
            )
            records.append(
                {
                    **candidate,
                    "affected_front_wheel_index": affected_wheel,
                    "front_normal_start_n": np.abs(
                        force_at_probe_start[candidate_index, :, 0]
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
                    "affected_wheel_delta_vs_baseline_mean_n": (
                        None
                        if affected_wheel is None
                        else float(
                            np.mean(
                                candidate_force[:, affected_wheel]
                                - baseline_force[:, affected_wheel]
                            )
                        )
                    ),
                    "base_position_delta_vs_baseline_max_abs_m": np.max(
                        np.abs(
                            base_position[:, candidate_index]
                            - base_position[:, 0]
                        ),
                        axis=0,
                    ).tolist(),
                    "base_position_delta_vs_baseline_end_m": (
                        base_position[-1, candidate_index]
                        - base_position[-1, 0]
                    ).tolist(),
                    "base_orientation_error_rmse_rad": float(
                        np.sqrt(
                            np.mean(
                                orientation_error[:, candidate_index] ** 2
                            )
                        )
                    ),
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
            "schema_version": "pcbc-front-force-local-response-v1",
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
            "joint_scope": args_cli.joint_scope,
            "candidate_count": candidate_count,
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
