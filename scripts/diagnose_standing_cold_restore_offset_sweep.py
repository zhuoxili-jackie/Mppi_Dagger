#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np

from _bootstrap import ROOT, load_contract, write_json

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description=(
        "Sequentially replay bounded standing-action offsets with the cold "
        "MPPI restore before every real control step."
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
    "--candidate-config",
    type=Path,
    required=True,
)
parser.add_argument(
    "--episode",
    type=Path,
    required=True,
)
parser.add_argument(
    "--action-reference",
    type=Path,
    default=ROOT
    / "assets/low_load_lateral/train_001/nominal_actions/ref_08_standing.npz",
)
parser.add_argument("--seed", type=int, default=5208)
parser.add_argument("--ref-id", type=int, default=8)
parser.add_argument("--steps", type=int, default=100)
parser.add_argument(
    "--physical-target-rate-limit-rad-s",
    type=float,
    default=2.25,
)
parser.add_argument(
    "--report",
    type=Path,
    default=ROOT
    / "reports/low_load_lateral/train_001/diagnostics/cold_restore_offset_sweep.json",
)
parser.add_argument("--disable-fabric", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def _quaternion_angle(
    current,
    initial,
):
    import torch

    dot = torch.abs(torch.sum(current * initial, dim=-1)).clamp(max=1.0)
    return 2.0 * torch.acos(dot)


def main() -> dict:
    import gymnasium as gym
    import robot_lab.tasks  # noqa: F401
    import torch
    from isaaclab_tasks.utils import parse_env_cfg

    from lateral_mppi_dagger.config import load_yaml, sha256_file
    from lateral_mppi_dagger.contract.action16 import ActionContract
    from lateral_mppi_dagger.env.isaac_adapter import IsaacLateralAdapter
    from lateral_mppi_dagger.env.isaac_mppi_rollout import (
        IsaacMPPIRolloutCloner,
        IsaacRolloutCostWeights,
    )
    from lateral_mppi_dagger.env.scenarios import (
        configure_env_for_scenario,
        load_scenario_profile,
    )
    from lateral_mppi_dagger.reference.loader import ReferenceSet

    if args_cli.steps < 1:
        raise ValueError("--steps must be positive.")
    if args_cli.physical_target_rate_limit_rad_s <= 0.0:
        raise ValueError(
            "--physical-target-rate-limit-rad-s must be positive."
        )

    candidate_config_path = args_cli.candidate_config.expanduser().resolve()
    candidate_config = load_yaml(candidate_config_path)
    candidate_entries = candidate_config.get("candidates")
    if not isinstance(candidate_entries, list) or not candidate_entries:
        raise ValueError("candidate config must contain a non-empty candidates list.")
    candidates: list[
        tuple[str, np.ndarray, int | None, int, float, float]
    ] = []
    names: set[str] = set()
    for entry in candidate_entries:
        if not isinstance(entry, dict):
            raise TypeError("Each candidate entry must be a mapping.")
        name = str(entry["name"])
        if name in names:
            raise ValueError(f"Duplicate candidate name {name!r}.")
        offset = np.asarray(entry["offset_leg_rad"], dtype=np.float32)
        if offset.shape != (12,) or not np.isfinite(offset).all():
            raise ValueError(
                f"Candidate {name!r} offset_leg_rad must contain 12 finite values."
            )
        action_frame = (
            None
            if entry.get("hold_action_frame") is None
            else int(entry["hold_action_frame"])
        )
        if action_frame is not None and action_frame < 0:
            raise ValueError(
                f"Candidate {name!r} hold_action_frame must be non-negative."
            )
        action_frame_offset = int(entry.get("action_frame_offset", 0))
        if action_frame is not None and action_frame_offset != 0:
            raise ValueError(
                f"Candidate {name!r} cannot combine hold_action_frame and "
                "action_frame_offset."
            )
        raw_action_scale = float(entry.get("raw_action_scale", 1.0))
        action_time_scale = float(entry.get("action_time_scale", 1.0))
        if (
            not np.isfinite(raw_action_scale)
            or raw_action_scale <= 0.0
            or not np.isfinite(action_time_scale)
            or action_time_scale <= 0.0
        ):
            raise ValueError(
                f"Candidate {name!r} action scales must be positive and finite."
            )
        names.add(name)
        candidates.append(
            (
                name,
                offset,
                action_frame,
                action_frame_offset,
                raw_action_scale,
                action_time_scale,
            )
        )

    episode_path = args_cli.episode.expanduser().resolve()
    with np.load(episode_path, allow_pickle=False) as archive:
        recorded_ref_frame = np.asarray(archive["ref_frame"], dtype=np.int64)
        recorded_desired_contact = np.asarray(
            archive["desired_contact"],
            dtype=bool,
        )
    action_reference_path = args_cli.action_reference.expanduser().resolve()
    with np.load(action_reference_path, allow_pickle=False) as archive:
        raw_action_leg = np.asarray(
            archive["raw_action_leg"],
            dtype=np.float32,
        )
        action_reference_ref_id = int(
            np.asarray(archive["ref_id"]).reshape(-1)[0]
        )
    if raw_action_leg.ndim != 2 or raw_action_leg.shape[1] != 12:
        raise ValueError("raw_action_leg must have shape [frames,12].")
    if action_reference_ref_id != args_cli.ref_id:
        raise ValueError("Action-reference ref_id differs from --ref-id.")
    if recorded_ref_frame.shape[0] < args_cli.steps:
        raise ValueError("Source episode is shorter than --steps.")
    action_frames = np.minimum(
        recorded_ref_frame[: args_cli.steps] + 1,
        raw_action_leg.shape[0] - 1,
    )

    contract_dict = load_contract()
    contract = ActionContract.from_dict(contract_dict)
    references = ReferenceSet.from_config(args_cli.reference_config)
    scenario = load_scenario_profile("nominal")
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )
    configure_env_for_scenario(env_cfg, scenario, num_envs=1)
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device
    env = gym.make(args_cli.task, cfg=env_cfg)
    adapter = IsaacLateralAdapter(
        env,
        references,
        contract_dict,
        scenario_profile=scenario,
    )
    restorer = IsaacMPPIRolloutCloner(
        adapter,
        references,
        contract,
        horizon=1,
        cost_weights=IsaacRolloutCostWeights(),
    )

    device = adapter.base.device
    scale = torch.as_tensor(
        contract.scale,
        dtype=torch.float32,
        device=device,
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
    maximum_physical_delta = (
        float(args_cli.physical_target_rate_limit_rad_s)
        * float(adapter.base.step_dt)
    )
    maximum_raw_delta = maximum_physical_delta / scale
    records: list[dict] = []

    try:
        for candidate_index, (
            name,
            offset,
            hold_action_frame,
            action_frame_offset,
            raw_action_scale,
            action_time_scale,
        ) in enumerate(candidates):
            adapter.reset(args_cli.seed, args_cli.ref_id)
            offset_t = torch.as_tensor(
                offset,
                dtype=torch.float32,
                device=device,
            )
            previous_action = torch.zeros(
                (1, 16),
                dtype=torch.float32,
                device=device,
            )
            initial_anchor_position = (
                adapter.command.robot_anchor_pos_w
                - adapter.base.scene.env_origins
            ).detach().clone()
            initial_anchor_quaternion = (
                adapter.command.robot_anchor_quat_w.detach().clone()
            )
            force_samples: list[np.ndarray] = []
            base_delta_samples: list[np.ndarray] = []
            orientation_error_samples: list[np.ndarray] = []
            lateral_velocity_samples: list[np.ndarray] = []
            action_samples: list[np.ndarray] = []
            done_any = False

            for step in range(args_cli.steps):
                force_samples.append(
                    adapter.contact_sensor.data.net_forces_w[
                        0,
                        adapter.contact_body_ids,
                    ]
                    .detach()
                    .cpu()
                    .numpy()
                )
                restorer.restore(restorer.capture())
                desired = torch.zeros(
                    (1, 16),
                    dtype=torch.float32,
                    device=device,
                )
                desired[0, :12] = torch.as_tensor(
                    raw_action_leg[
                        min(
                            hold_action_frame,
                            raw_action_leg.shape[0] - 1,
                        )
                        if hold_action_frame is not None
                        else int(
                            np.clip(
                                round(
                                    float(action_frames[step])
                                    * action_time_scale
                                )
                                + action_frame_offset,
                                0,
                                raw_action_leg.shape[0] - 1,
                            )
                        )
                    ]
                    * raw_action_scale,
                    dtype=torch.float32,
                    device=device,
                )
                desired[0, :12] += offset_t / scale[:12]
                desired = torch.maximum(
                    torch.minimum(
                        desired,
                        previous_action + maximum_raw_delta,
                    ),
                    previous_action - maximum_raw_delta,
                )
                desired = torch.maximum(
                    torch.minimum(desired, raw_max),
                    raw_min,
                )
                desired[:, 12:].zero_()
                _, _, terminated, truncated, _ = env.step(desired)
                done_any = bool(
                    done_any
                    or torch.as_tensor(terminated).reshape(-1)[0].item()
                    or torch.as_tensor(truncated).reshape(-1)[0].item()
                )
                previous_action = desired
                action_samples.append(desired[0].detach().cpu().numpy())

                anchor_position = (
                    adapter.command.robot_anchor_pos_w
                    - adapter.base.scene.env_origins
                )
                base_delta_samples.append(
                    (anchor_position - initial_anchor_position)[0]
                    .detach()
                    .cpu()
                    .numpy()
                )
                orientation_error_samples.append(
                    _quaternion_angle(
                        adapter.command.robot_anchor_quat_w,
                        initial_anchor_quaternion,
                    )[0]
                    .detach()
                    .cpu()
                    .numpy()
                )
                lateral_velocity_samples.append(
                    adapter.command.robot_anchor_lin_vel_w[0, 1]
                    .detach()
                    .cpu()
                    .numpy()
                )

            force = np.stack(force_samples)
            front_normal_force = np.abs(force[:, :2, 0])
            rear_normal_force = np.abs(force[:, 2:, 2])
            desired_contact = recorded_desired_contact[: args_cli.steps]
            desired_front = desired_contact[:, :2]
            front_below = (
                (front_normal_force < 6.0) & desired_front
            )
            desired_rear = desired_contact[:, 2:]
            rear_measured = (
                rear_normal_force >= 8.0
            ) & desired_rear
            rear_count = np.sum(rear_measured, axis=1)
            base_delta = np.stack(base_delta_samples)
            orientation_error = np.asarray(orientation_error_samples)
            lateral_velocity = np.asarray(lateral_velocity_samples)
            actions = np.stack(action_samples)
            physical_step = (
                np.diff(
                    actions[:, :12],
                    axis=0,
                    prepend=np.zeros_like(actions[:1, :12]),
                )
                * np.asarray(contract.scale[:12], dtype=np.float32)
            )
            record = {
                "candidate_index": candidate_index,
                "name": name,
                "offset_leg_rad": offset.tolist(),
                "hold_action_frame": hold_action_frame,
                "action_frame_offset": action_frame_offset,
                "raw_action_scale": raw_action_scale,
                "action_time_scale": action_time_scale,
                "front_normal_below_6n_fraction_when_desired": float(
                    np.sum(front_below)
                    / max(np.sum(desired_front), 1)
                ),
                "front_normal_force_mean_n": [
                    float(
                        np.mean(
                            front_normal_force[:, wheel][
                                desired_front[:, wheel]
                            ]
                        )
                    )
                    for wheel in range(2)
                ],
                "base_position_delta_max_abs_m": np.max(
                    np.abs(base_delta),
                    axis=0,
                ).tolist(),
                "base_orientation_rmse_rad": float(
                    np.sqrt(np.mean(orientation_error**2))
                ),
                "lateral_velocity_mae_m_s": float(
                    np.mean(np.abs(lateral_velocity))
                ),
                "rear_normal_force_p95_n": [
                    float(
                        np.quantile(
                            rear_normal_force[:, wheel][
                                desired_rear[:, wheel]
                            ],
                            0.95,
                        )
                    )
                    for wheel in range(2)
                ],
                "rear_single_support_fraction": float(
                    np.mean(rear_count == 1)
                ),
                "rear_no_support_fraction": float(
                    np.mean(rear_count == 0)
                ),
                "physical_leg_target_step_max_rad": float(
                    np.max(np.abs(physical_step))
                ),
                "terminated_any": done_any,
                "wheel_action_exact_zero": bool(
                    np.array_equal(
                        actions[:, 12:],
                        np.zeros_like(actions[:, 12:]),
                    )
                ),
            }
            record["standing_gate_pass"] = bool(
                not record["terminated_any"]
                and record[
                    "front_normal_below_6n_fraction_when_desired"
                ]
                <= 0.20
                and np.all(
                    np.asarray(
                        record["base_position_delta_max_abs_m"]
                    )
                    <= np.asarray([0.12, 0.10, 0.10])
                )
                and record["base_orientation_rmse_rad"] <= 0.20
                and record["lateral_velocity_mae_m_s"] <= 0.025
                and max(record["rear_normal_force_p95_n"]) <= 135.0
                and record["rear_single_support_fraction"] <= 0.45
                and record["rear_no_support_fraction"] <= 0.01
                and record["physical_leg_target_step_max_rad"] <= 0.055
                and record["wheel_action_exact_zero"]
            )
            records.append(record)
            print(
                json.dumps(
                    {
                        "progress": {
                            "completed": candidate_index + 1,
                            "total": len(candidates),
                            "name": name,
                            "front_low_fraction": record[
                                "front_normal_below_6n_fraction_when_desired"
                            ],
                            "gate_pass": record["standing_gate_pass"],
                        }
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        adapter.close()

    ranked = sorted(
        records,
        key=lambda record: (
            not record["standing_gate_pass"],
            record["terminated_any"],
            record[
                "front_normal_below_6n_fraction_when_desired"
            ],
            max(record["base_position_delta_max_abs_m"]),
        ),
    )
    report = {
        "schema_version": "pcbc-standing-cold-restore-offset-sweep-v1",
        "status": "diagnostic_not_training_data",
        "task": args_cli.task,
        "reference_config": args_cli.reference_config,
        "ref_id": args_cli.ref_id,
        "seed": args_cli.seed,
        "steps": args_cli.steps,
        "cold_restore_before_each_real_step": True,
        "candidate_config": str(candidate_config_path),
        "candidate_config_sha256": sha256_file(candidate_config_path),
        "source_episode": str(episode_path),
        "action_reference": str(action_reference_path),
        "action_reference_sha256": sha256_file(action_reference_path),
        "physical_target_rate_limit_rad_s": (
            args_cli.physical_target_rate_limit_rad_s
        ),
        "candidate_count": len(candidates),
        "passing_candidate_count": sum(
            int(record["standing_gate_pass"]) for record in records
        ),
        "records": records,
        "ranked": ranked,
    }
    write_json(args_cli.report, report)
    print(json.dumps(ranked[:5], sort_keys=True))
    return report


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        write_json(
            str(args_cli.report) + ".failure.json",
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
