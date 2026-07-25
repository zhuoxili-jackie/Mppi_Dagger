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
        "Record a deterministic Isaac replay of one hashed low-load action "
        "proposal. This is a diagnostic system-identification trace, not an "
        "expert episode or training dataset."
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
parser.add_argument("--ref-id", type=int, required=True)
parser.add_argument("--action-reference", type=Path, required=True)
parser.add_argument(
    "--blend-base-action-reference",
    type=Path,
    default=None,
    help=(
        "Optional raw-action NPZ used as the zero endpoint for a parallel "
        "feedforward gait-amplitude sweep."
    ),
)
parser.add_argument(
    "--action-blend-values",
    type=float,
    nargs="+",
    default=(1.0,),
)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--replicates", type=int, default=2)
parser.add_argument("--seed", type=int, default=5900)
parser.add_argument(
    "--physical-target-rate-limit-rad-s",
    type=float,
    default=2.25,
)
parser.add_argument(
    "--physical-offset-leg-rad",
    type=float,
    nargs=12,
    default=(0.0,) * 12,
    help=(
        "Optional physical joint-target offset applied after action blending; "
        "joint order is the frozen type-grouped leg order."
    ),
)
parser.add_argument(
    "--physical-offset-start-frame",
    type=int,
    default=0,
)
parser.add_argument(
    "--physical-offset-ramp-frames",
    type=int,
    default=0,
    help=(
        "Ramp the physical offset over this many replay frames. Zero applies "
        "the full offset at --physical-offset-start-frame."
    ),
)
parser.add_argument("--trace", type=Path, required=True)
parser.add_argument("--report", type=Path, required=True)
parser.add_argument("--disable-fabric", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def _require_inside_root(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(ROOT)
    except ValueError as exc:
        raise ValueError(
            f"{label} must remain inside the project root {ROOT}: {resolved}"
        ) from exc
    return resolved


def main() -> dict[str, object]:
    import gymnasium as gym
    import robot_lab.tasks  # noqa: F401
    import torch
    from isaaclab_tasks.utils import parse_env_cfg

    from lateral_mppi_dagger.config import sha256_file
    from lateral_mppi_dagger.contract.action16 import ActionContract
    from lateral_mppi_dagger.env.isaac_adapter import IsaacLateralAdapter
    from lateral_mppi_dagger.env.scenarios import (
        configure_env_for_scenario,
        load_scenario_profile,
    )
    from lateral_mppi_dagger.reference.loader import ReferenceSet

    if args_cli.steps < 1:
        raise ValueError("--steps must be positive.")
    if args_cli.replicates < 1:
        raise ValueError("--replicates must be positive.")
    if (
        not np.isfinite(args_cli.physical_target_rate_limit_rad_s)
        or args_cli.physical_target_rate_limit_rad_s <= 0.0
    ):
        raise ValueError(
            "--physical-target-rate-limit-rad-s must be finite and positive."
        )
    physical_offset_leg = np.asarray(
        args_cli.physical_offset_leg_rad,
        dtype=np.float32,
    )
    if physical_offset_leg.shape != (12,) or not np.isfinite(
        physical_offset_leg
    ).all():
        raise ValueError(
            "--physical-offset-leg-rad must contain 12 finite values."
        )
    if args_cli.physical_offset_start_frame < 0:
        raise ValueError(
            "--physical-offset-start-frame must be non-negative."
        )
    if args_cli.physical_offset_ramp_frames < 0:
        raise ValueError(
            "--physical-offset-ramp-frames must be non-negative."
        )

    action_path = args_cli.action_reference.expanduser().resolve()
    with np.load(action_path, allow_pickle=False) as archive:
        raw_action_leg = np.asarray(
            archive["raw_action_leg"],
            dtype=np.float32,
        )
        action_ref_id = int(np.asarray(archive["ref_id"]).reshape(-1)[0])
    if (
        raw_action_leg.ndim != 2
        or raw_action_leg.shape[1] != 12
        or not np.isfinite(raw_action_leg).all()
    ):
        raise ValueError(
            "The action reference must contain finite raw_action_leg "
            "with shape [frames,12]."
        )
    if action_ref_id != args_cli.ref_id:
        raise ValueError(
            "The action-reference ref_id differs from --ref-id."
        )
    blend_values = np.asarray(
        args_cli.action_blend_values,
        dtype=np.float32,
    )
    if (
        blend_values.ndim != 1
        or blend_values.size < 1
        or not np.isfinite(blend_values).all()
        or np.any((blend_values < 0.0) | (blend_values > 1.5))
    ):
        raise ValueError(
            "--action-blend-values must be finite values in [0,1.5]."
        )
    blend_base_path: Path | None = None
    blend_base_raw_action_leg: np.ndarray | None = None
    if args_cli.blend_base_action_reference is not None:
        blend_base_path = (
            args_cli.blend_base_action_reference.expanduser().resolve()
        )
        with np.load(blend_base_path, allow_pickle=False) as archive:
            blend_base_raw_action_leg = np.asarray(
                archive["raw_action_leg"],
                dtype=np.float32,
            )
        if (
            blend_base_raw_action_leg.shape != raw_action_leg.shape
            or not np.isfinite(blend_base_raw_action_leg).all()
        ):
            raise ValueError(
                "The blend-base raw_action_leg must be finite and have the "
                "same shape as the active action reference."
            )
    elif not np.array_equal(
        blend_values,
        np.ones_like(blend_values),
    ):
        raise ValueError(
            "Non-unit --action-blend-values require "
            "--blend-base-action-reference."
        )

    references = ReferenceSet.from_config(args_cli.reference_config)
    if not 0 <= args_cli.ref_id < len(references):
        raise ValueError("--ref-id is outside the reference bank.")
    reference = references[args_cli.ref_id]
    if args_cli.steps >= reference.frames:
        raise ValueError(
            "--steps must leave at least one reference frame for frame+1 "
            f"action lookahead; got steps={args_cli.steps}, "
            f"frames={reference.frames}."
        )
    if raw_action_leg.shape[0] != reference.frames:
        raise ValueError(
            "Action/reference frame count mismatch: "
            f"{raw_action_leg.shape[0]} vs {reference.frames}."
        )

    trace_path = _require_inside_root(args_cli.trace, "--trace")
    report_path = _require_inside_root(args_cli.report, "--report")
    if trace_path.exists():
        raise FileExistsError(f"Refusing to overwrite trace: {trace_path}")
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite report: {report_path}")

    contract_dict = load_contract()
    contract = ActionContract.from_dict(contract_dict)
    scale = torch.as_tensor(
        contract.scale,
        dtype=torch.float32,
        device=args_cli.device,
    )
    raw_min = torch.as_tensor(
        contract.raw_min,
        dtype=torch.float32,
        device=args_cli.device,
    )
    raw_max = torch.as_tensor(
        contract.raw_max,
        dtype=torch.float32,
        device=args_cli.device,
    )
    maximum_physical_delta = (
        args_cli.physical_target_rate_limit_rad_s / reference.fps
    )
    maximum_raw_delta = maximum_physical_delta / scale

    candidate_count = int(blend_values.size)
    num_envs = candidate_count * args_cli.replicates
    blend_by_env = np.repeat(
        blend_values,
        args_cli.replicates,
    )
    scenario = load_scenario_profile("nominal")
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    configure_env_for_scenario(
        env_cfg,
        scenario,
        num_envs=num_envs,
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

    arrays: dict[str, list[np.ndarray]] = {
        "ref_frame": [],
        "desired_contact": [],
        "base_pose_local_w": [],
        "base_twist_w": [],
        "q": [],
        "dq": [],
        "wheel_body_pose_local_w": [],
        "wheel_body_twist_w": [],
        "contact_force_w": [],
        "executed_action16": [],
        "terminated": [],
        "truncated": [],
    }
    try:
        adapter.reset(args_cli.seed, args_cli.ref_id)
        device = adapter.base.device
        previous_action = torch.zeros(
            (num_envs, 16),
            dtype=torch.float32,
            device=device,
        )
        done_any = torch.zeros(
            num_envs,
            dtype=torch.bool,
            device=device,
        )
        for step in range(args_cli.steps):
            expected_frame = step
            if not bool(
                torch.all(adapter.command.time_steps == expected_frame)
            ):
                raise RuntimeError(
                    "Isaac reference frame diverged before replay step "
                    f"{step}: {adapter.command.time_steps.detach().cpu().tolist()}"
                )
            env_origins = adapter.base.scene.env_origins
            anchor_position_local = (
                adapter.command.robot_anchor_pos_w - env_origins
            )
            anchor_pose_local = torch.cat(
                (
                    anchor_position_local,
                    adapter.command.robot_anchor_quat_w,
                ),
                dim=-1,
            )
            anchor_twist = torch.cat(
                (
                    adapter.command.robot_anchor_lin_vel_w,
                    adapter.command.robot_anchor_ang_vel_w,
                ),
                dim=-1,
            )
            wheel_position_local = (
                adapter.robot.data.body_pos_w[
                    :,
                    adapter.wheel_body_ids,
                ]
                - env_origins[:, None, :]
            )
            wheel_pose_local = torch.cat(
                (
                    wheel_position_local,
                    adapter.robot.data.body_quat_w[
                        :,
                        adapter.wheel_body_ids,
                    ],
                ),
                dim=-1,
            )
            wheel_twist = torch.cat(
                (
                    adapter.robot.data.body_lin_vel_w[
                        :,
                        adapter.wheel_body_ids,
                    ],
                    adapter.robot.data.body_ang_vel_w[
                        :,
                        adapter.wheel_body_ids,
                    ],
                ),
                dim=-1,
            )

            action_frame = min(step + 1, raw_action_leg.shape[0] - 1)
            desired = torch.zeros(
                (num_envs, 16),
                dtype=torch.float32,
                device=device,
            )
            active_action = raw_action_leg[action_frame]
            if blend_base_raw_action_leg is None:
                blended_action = np.broadcast_to(
                    active_action,
                    (num_envs, 12),
                )
            else:
                base_action = blend_base_raw_action_leg[action_frame]
                blended_action = (
                    base_action[None]
                    + blend_by_env[:, None]
                    * (active_action - base_action)[None]
                )
            desired[:, :12] = torch.as_tensor(
                blended_action,
                dtype=torch.float32,
                device=device,
            )
            if step >= args_cli.physical_offset_start_frame:
                if args_cli.physical_offset_ramp_frames == 0:
                    offset_factor = 1.0
                else:
                    offset_factor = min(
                        (
                            step
                            - args_cli.physical_offset_start_frame
                            + 1
                        )
                        / args_cli.physical_offset_ramp_frames,
                        1.0,
                    )
                desired[:, :12] += (
                    torch.as_tensor(
                        physical_offset_leg,
                        dtype=torch.float32,
                        device=device,
                    )
                    * offset_factor
                    / scale[:12]
                )
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

            arrays["ref_frame"].append(
                np.full(
                    num_envs,
                    expected_frame,
                    dtype=np.int32,
                )
            )
            arrays["desired_contact"].append(
                np.broadcast_to(
                    adapter.contact_schedules[
                        args_cli.ref_id
                    ][expected_frame],
                    (num_envs, 4),
                ).copy()
            )
            arrays["base_pose_local_w"].append(
                anchor_pose_local.detach().cpu().numpy()
            )
            arrays["base_twist_w"].append(
                anchor_twist.detach().cpu().numpy()
            )
            arrays["q"].append(
                adapter.robot.data.joint_pos[
                    :,
                    adapter.joint_ids,
                ]
                .detach()
                .cpu()
                .numpy()
            )
            arrays["dq"].append(
                adapter.robot.data.joint_vel[
                    :,
                    adapter.joint_ids,
                ]
                .detach()
                .cpu()
                .numpy()
            )
            arrays["wheel_body_pose_local_w"].append(
                wheel_pose_local.detach().cpu().numpy()
            )
            arrays["wheel_body_twist_w"].append(
                wheel_twist.detach().cpu().numpy()
            )
            arrays["contact_force_w"].append(
                adapter.contact_sensor.data.net_forces_w[
                    :,
                    adapter.contact_body_ids,
                ]
                .detach()
                .cpu()
                .numpy()
            )
            arrays["executed_action16"].append(
                desired.detach().cpu().numpy()
            )

            _, _, terminated, truncated, _ = env.step(desired)
            terminated_t = torch.as_tensor(
                terminated,
                dtype=torch.bool,
                device=device,
            ).reshape(-1)
            truncated_t = torch.as_tensor(
                truncated,
                dtype=torch.bool,
                device=device,
            ).reshape(-1)
            arrays["terminated"].append(
                terminated_t.detach().cpu().numpy()
            )
            arrays["truncated"].append(
                truncated_t.detach().cpu().numpy()
            )
            done_any |= terminated_t | truncated_t
            previous_action = desired
            if bool(done_any.any()):
                break

        stacked = {
            name: np.stack(values, axis=0)
            for name, values in arrays.items()
        }
        wheel_zero = np.array_equal(
            stacked["executed_action16"][..., 12:],
            np.zeros_like(stacked["executed_action16"][..., 12:]),
        )
        physical_action = (
            stacked["executed_action16"][..., :12]
            * np.asarray(contract.scale[:12], dtype=np.float32)
        )
        physical_step = np.diff(
            physical_action,
            axis=0,
            prepend=np.zeros_like(physical_action[:1]),
        )
        recorded_steps = int(stacked["ref_frame"].shape[0])
        metric_frames = stacked["ref_frame"][:, 0].astype(np.int64)
        target_lateral_velocity = np.asarray(
            reference.body_lin_vel_w[metric_frames, 0, 1],
            dtype=np.float32,
        )
        target_lateral_position = np.asarray(
            reference.body_pos_w[metric_frames, 0, 1],
            dtype=np.float32,
        )
        front_normal_force = np.abs(
            stacked["contact_force_w"][..., :2, 0]
        )
        desired_front = stacked["desired_contact"][..., :2].astype(bool)
        candidate_records: list[dict[str, object]] = []
        for candidate_index, blend in enumerate(blend_values):
            start = candidate_index * args_cli.replicates
            stop = start + args_cli.replicates
            checkpoints: dict[str, object] = {}
            checkpoint_values = sorted(
                {
                    min(100, recorded_steps),
                    recorded_steps,
                }
            )
            for checkpoint in checkpoint_values:
                if checkpoint < 1:
                    continue
                actual_position = stacked[
                    "base_pose_local_w"
                ][:checkpoint, start:stop, 1]
                actual_displacement = (
                    actual_position[-1] - actual_position[0]
                )
                target_displacement = float(
                    target_lateral_position[checkpoint - 1]
                    - target_lateral_position[0]
                )
                if abs(target_displacement) > 1.0e-4:
                    progress = (
                        actual_displacement / target_displacement
                    )
                else:
                    progress = np.ones(
                        args_cli.replicates,
                        dtype=np.float32,
                    )
                velocity = stacked[
                    "base_twist_w"
                ][:checkpoint, start:stop, 1]
                desired = desired_front[:checkpoint, start:stop]
                below = (
                    front_normal_force[:checkpoint, start:stop] < 6.0
                ) & desired
                desired_count = int(np.sum(desired))
                checkpoints[f"steps_{checkpoint}"] = {
                    "actual_lateral_displacement_m_per_replicate": (
                        actual_displacement.tolist()
                    ),
                    "target_lateral_displacement_m": target_displacement,
                    "signed_lateral_progress_ratio_per_replicate": (
                        progress.tolist()
                    ),
                    "signed_lateral_progress_ratio_mean": float(
                        np.mean(progress)
                    ),
                    "lateral_velocity_mae_m_s_per_replicate": (
                        np.mean(
                            np.abs(
                                velocity
                                - target_lateral_velocity[
                                    :checkpoint,
                                    None,
                                ]
                            ),
                            axis=0,
                        ).tolist()
                    ),
                    "front_normal_below_6n_fraction_when_desired": (
                        float(np.sum(below) / max(desired_count, 1))
                    ),
                    "front_normal_below_6n_fraction_per_replicate": (
                        np.sum(below, axis=(0, 2))
                        / np.maximum(
                            np.sum(desired, axis=(0, 2)),
                            1,
                        )
                    ).tolist(),
                }
            candidate_records.append(
                {
                    "candidate_index": candidate_index,
                    "action_blend": float(blend),
                    "checkpoints": checkpoints,
                    "physical_leg_target_step_max_rad": float(
                        np.max(
                            np.abs(
                                physical_step[:, start:stop]
                            )
                        )
                    ),
                    "terminated_any": bool(
                        np.any(stacked["terminated"][:, start:stop])
                    ),
                    "premature_truncation_any": bool(
                        np.any(
                            stacked["truncated"][
                                :-1,
                                start:stop,
                            ]
                        )
                    ),
                    "wheel_action_exact_zero": bool(
                        np.array_equal(
                            stacked[
                                "executed_action16"
                            ][:, start:stop, 12:],
                            np.zeros_like(
                                stacked[
                                    "executed_action16"
                                ][:, start:stop, 12:]
                            ),
                        )
                    ),
                }
            )
        premature_done = bool(
            np.any(stacked["terminated"][:-1])
            or np.any(stacked["truncated"][:-1])
        )
        terminated_any = bool(np.any(stacked["terminated"]))
        final_step_truncated = bool(
            recorded_steps == args_cli.steps
            and np.any(stacked["truncated"][-1])
        )
        complete = (
            recorded_steps == args_cli.steps
            and not premature_done
            and not terminated_any
            and wheel_zero
        )

        trace_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            trace_path,
            schema_version=np.asarray(
                ["pcbc-low-load-action-replay-trace-v1"]
            ),
            ref_id=np.asarray([args_cli.ref_id], dtype=np.int64),
            seed=np.asarray([args_cli.seed], dtype=np.int64),
            fps=np.asarray([reference.fps], dtype=np.int64),
            action_reference_sha256=np.asarray(
                [sha256_file(action_path)]
            ),
            action_blend_values=blend_values,
            action_blend_by_env=blend_by_env,
            physical_offset_leg_rad=physical_offset_leg,
            physical_offset_start_frame=np.asarray(
                [args_cli.physical_offset_start_frame],
                dtype=np.int64,
            ),
            physical_offset_ramp_frames=np.asarray(
                [args_cli.physical_offset_ramp_frames],
                dtype=np.int64,
            ),
            **stacked,
        )
        report = {
            "schema_version": "pcbc-low-load-action-replay-report-v1",
            "status": (
                "pass_diagnostic_trace_not_training_data"
                if complete
                else "fail_incomplete_or_terminated"
            ),
            "purpose": (
                "system_identification_only_not_expert_episode_or_training_data"
            ),
            "reference_config": args_cli.reference_config,
            "ref_id": args_cli.ref_id,
            "target_vy": float(reference.target_vy),
            "action_reference": str(action_path),
            "action_reference_sha256": sha256_file(action_path),
            "blend_base_action_reference": (
                str(blend_base_path)
                if blend_base_path is not None
                else None
            ),
            "blend_base_action_reference_sha256": (
                sha256_file(blend_base_path)
                if blend_base_path is not None
                else None
            ),
            "action_blend_values": blend_values.tolist(),
            "candidate_count": candidate_count,
            "candidate_records": candidate_records,
            "seed": args_cli.seed,
            "requested_steps": args_cli.steps,
            "recorded_steps": recorded_steps,
            "replicates": args_cli.replicates,
            "physical_target_rate_limit_rad_s": (
                args_cli.physical_target_rate_limit_rad_s
            ),
            "physical_offset_leg_rad": physical_offset_leg.tolist(),
            "physical_offset_start_frame": (
                args_cli.physical_offset_start_frame
            ),
            "physical_offset_ramp_frames": (
                args_cli.physical_offset_ramp_frames
            ),
            "physical_leg_target_step_max_rad": float(
                np.max(np.abs(physical_step))
            ),
            "terminated_any": terminated_any,
            "truncated_any": bool(
                np.any(stacked["truncated"])
            ),
            "premature_done_any": premature_done,
            "final_step_time_limit_truncation": final_step_truncated,
            "wheel_action_exact_zero": bool(wheel_zero),
            "trace": str(trace_path),
            "trace_sha256": sha256_file(trace_path),
        }
        write_json(report_path, report)
        print(report)
        return report
    finally:
        env.close()


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except Exception:
        traceback.print_exc()
        exit_code = 1
    finally:
        simulation_app.close()
    sys.exit(exit_code)
