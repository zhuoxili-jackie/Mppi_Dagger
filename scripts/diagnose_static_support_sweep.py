#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from _bootstrap import ROOT, load_contract, write_json

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="Sweep constant low-load support preloads in parallel Isaac environments."
)
parser.add_argument(
    "--task",
    default="RobotLab-Isaac-BeyondMimic-Flat-pcbC-v1-lateral-bipedal-stand-v0",
)
parser.add_argument(
    "--reference-config",
    default="configs/low_load_lateral/train_001/reference.yaml",
)
parser.add_argument("--steps", type=int, default=100)
parser.add_argument("--seed", type=int, default=5100)
parser.add_argument("--ref-id", type=int, default=8)
parser.add_argument(
    "--root-shift-mm",
    type=float,
    nargs="+",
    default=(0.0, 4.0, 8.0, 12.0, 16.0),
)
parser.add_argument(
    "--front-penetration-mm",
    type=float,
    nargs="+",
    default=(0.0, 4.0, 8.0, 12.0),
)
parser.add_argument(
    "--front-drop-mm",
    type=float,
    nargs="+",
    default=(0.0,),
    help="Downward world-Z offset of both front wheel centers.",
)
parser.add_argument(
    "--report",
    type=Path,
    default=ROOT
    / "reports/low_load_lateral/train_001/diagnostics/static_support_sweep.json",
)
parser.add_argument("--disable-fabric", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


BODY_ORDER = (
    "Base_link",
    "FL_hip_link",
    "FR_hip_link",
    "RL_hip_link",
    "RR_hip_link",
    "FL_thigh_link",
    "FR_thigh_link",
    "RL_thigh_link",
    "RR_thigh_link",
    "FL_calf_link",
    "FR_calf_link",
    "RL_calf_link",
    "RR_calf_link",
    "FL_foot_link",
    "FR_foot_link",
    "RL_foot_link",
    "RR_foot_link",
)
JOINT_ORDER = (
    "FL_hip_joint",
    "FR_hip_joint",
    "RL_hip_joint",
    "RR_hip_joint",
    "FL_thigh_joint",
    "FR_thigh_joint",
    "RL_thigh_joint",
    "RR_thigh_joint",
    "FL_calf_joint",
    "FR_calf_joint",
    "RL_calf_joint",
    "RR_calf_joint",
    "FL_foot_joint",
    "FR_foot_joint",
    "RL_foot_joint",
    "RR_foot_joint",
)
LEGS = ("FL", "FR", "RL", "RR")
LEG_INDICES = {
    leg: (LEGS.index(leg), 4 + LEGS.index(leg), 8 + LEGS.index(leg))
    for leg in LEGS
}


def _joint_map(joint_position: np.ndarray) -> dict[str, float]:
    return {
        name: float(value)
        for name, value in zip(JOINT_ORDER, joint_position, strict=True)
    }


def _solve_support_target(
    tree,
    reference,
    root_shift_m: float,
    front_penetration_m: float,
    front_drop_m: float,
) -> np.ndarray:
    from lateral_mppi_dagger.reference.urdf_kinematics import (
        quaternion_wxyz_to_matrix,
    )

    q0 = np.asarray(reference.joint_pos[0], dtype=np.float64)
    root0 = np.asarray(reference.body_pos_w[0, 0], dtype=np.float64)
    root_quat = np.asarray(reference.body_quat_w[0, 0], dtype=np.float64)
    root_rotation_inverse = quaternion_wxyz_to_matrix(root_quat).T
    root_target = root0.copy()
    root_target[0] += root_shift_m
    q_target = q0.copy()

    for leg in LEGS:
        indices = LEG_INDICES[leg]
        joint_names = tuple(JOINT_ORDER[index] for index in indices)
        foot_target = np.asarray(
            reference.body_pos_w[
                0,
                BODY_ORDER.index(f"{leg}_foot_link"),
            ],
            dtype=np.float64,
        ).copy()
        if leg.startswith("F"):
            foot_target[0] += front_penetration_m
            foot_target[2] -= front_drop_m
        target_base = root_rotation_inverse @ (foot_target - root_target)
        lower = np.asarray(
            [tree.joint_by_name[name].lower for name in joint_names]
        )
        upper = np.asarray(
            [tree.joint_by_name[name].upper for name in joint_names]
        )

        def residual(candidate: np.ndarray) -> np.ndarray:
            trial = q_target.copy()
            trial[list(indices)] = candidate
            return (
                tree.link_transform_base(
                    f"{leg}_foot_link",
                    _joint_map(trial),
                )[:3, 3]
                - target_base
            )

        result = least_squares(
            residual,
            q_target[list(indices)],
            bounds=(lower + 1.0e-6, upper - 1.0e-6),
            xtol=1.0e-11,
            ftol=1.0e-11,
            gtol=1.0e-11,
            max_nfev=80,
        )
        if not result.success or np.max(np.abs(result.fun)) > 1.0e-5:
            raise RuntimeError(
                f"Support IK failed for {leg}: {result.message}; "
                f"max_abs={np.max(np.abs(result.fun)):.6g}"
            )
        q_target[list(indices)] = result.x
    return q_target[:12].astype(np.float32)


def main() -> dict:
    import gymnasium as gym
    import robot_lab.tasks  # noqa: F401
    import torch
    from isaaclab_tasks.utils import parse_env_cfg

    from lateral_mppi_dagger.contract.action16 import ActionContract
    from lateral_mppi_dagger.env.isaac_adapter import IsaacLateralAdapter
    from lateral_mppi_dagger.env.isaac_mppi_rollout import (
        _quat_rotation_vector,
    )
    from lateral_mppi_dagger.env.scenarios import (
        configure_env_for_scenario,
        load_scenario_profile,
    )
    from lateral_mppi_dagger.reference.loader import ReferenceSet
    from lateral_mppi_dagger.reference.urdf_kinematics import URDFKinematicTree

    combinations = [
        (float(root_mm), float(front_mm), float(front_drop_mm))
        for root_mm in args_cli.root_shift_mm
        for front_mm in args_cli.front_penetration_mm
        for front_drop_mm in args_cli.front_drop_mm
    ]
    if args_cli.steps <= 0 or not combinations:
        raise ValueError("The sweep requires positive steps and at least one candidate.")

    contract_dict = load_contract()
    action_contract = ActionContract.from_dict(contract_dict)
    references = ReferenceSet.from_config(args_cli.reference_config)
    scenario = load_scenario_profile("nominal")
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=len(combinations),
        use_fabric=not args_cli.disable_fabric,
    )
    configure_env_for_scenario(env_cfg, scenario, num_envs=len(combinations))
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device
    env = gym.make(args_cli.task, cfg=env_cfg)
    adapter = IsaacLateralAdapter(
        env,
        references,
        contract_dict,
        scenario_profile=scenario,
    )

    urdf_path = ROOT / contract_dict["assets"]["robot_urdf"]["path"]
    tree = URDFKinematicTree(urdf_path)
    reference = references[args_cli.ref_id]
    q_targets = np.stack(
        [
            _solve_support_target(
                tree,
                reference,
                root_mm / 1000.0,
                front_mm / 1000.0,
                front_drop_mm / 1000.0,
            )
            for root_mm, front_mm, front_drop_mm in combinations
        ],
        axis=0,
    )
    raw_targets = (
        q_targets - action_contract.q_action_offset_runtime[None, :12]
    ) / action_contract.scale[None, :12]
    raw_targets = np.clip(
        raw_targets,
        action_contract.raw_min[None, :12],
        action_contract.raw_max[None, :12],
    ).astype(np.float32)
    applied_q_targets = (
        raw_targets * action_contract.scale[None, :12]
        + action_contract.q_action_offset_runtime[None, :12]
    ).astype(np.float32)

    try:
        adapter.reset(args_cli.seed, args_cli.ref_id)
        device = adapter.base.device
        raw_target_t = torch.as_tensor(raw_targets, device=device)
        action_t = torch.zeros(
            (len(combinations), 16),
            dtype=torch.float32,
            device=device,
        )
        max_physical_step = 2.25 * adapter.control_dt
        raw_delta = torch.as_tensor(
            max_physical_step / action_contract.scale[:12],
            dtype=torch.float32,
            device=device,
        )
        alive = torch.ones(len(combinations), dtype=torch.bool, device=device)
        termination_step = torch.full(
            (len(combinations),),
            args_cli.steps,
            dtype=torch.int64,
            device=device,
        )
        base0 = (
            adapter.command.robot_anchor_pos_w
            - adapter.base.scene.env_origins
        ).clone()
        box_root0 = (
            adapter.base.scene["box"].data.root_pos_w
            - adapter.base.scene.env_origins
        ).clone()
        base_samples: list[np.ndarray] = []
        orientation_samples: list[np.ndarray] = []
        wheel_samples: list[np.ndarray] = []
        front_force_samples: list[np.ndarray] = []
        rear_samples: list[np.ndarray] = []
        alive_samples: list[np.ndarray] = []

        for step in range(args_cli.steps):
            base_local = (
                adapter.command.robot_anchor_pos_w
                - adapter.base.scene.env_origins
            )
            force = adapter.contact_sensor.data.net_forces_w[
                :,
                adapter.contact_body_ids,
            ]
            base_samples.append(base_local.detach().cpu().numpy())
            orientation_samples.append(
                _quat_rotation_vector(
                    adapter.command.robot_anchor_quat_w,
                    adapter.command.anchor_quat_w,
                )
                .detach()
                .cpu()
                .numpy()
            )
            wheel_samples.append(
                (
                    adapter.robot.data.body_pos_w[:, adapter.wheel_body_ids]
                    - adapter.base.scene.env_origins.unsqueeze(1)
                )
                .detach()
                .cpu()
                .numpy()
            )
            front_force_samples.append(force[:, :2].detach().cpu().numpy())
            rear_samples.append(
                torch.abs(force[:, 2:, 2]).detach().cpu().numpy()
            )
            alive_samples.append(alive.detach().cpu().numpy())

            action_t[:, :12] += torch.clamp(
                raw_target_t - action_t[:, :12],
                -raw_delta,
                raw_delta,
            )
            action_t[:, 12:].zero_()
            _, _, terminated, truncated, _ = env.step(action_t)
            done = (
                torch.as_tensor(terminated, device=device).reshape(-1)
                | torch.as_tensor(truncated, device=device).reshape(-1)
            )
            newly_done = alive & done
            termination_step[newly_done] = step + 1
            alive &= ~done

        base = np.stack(base_samples)
        orientation = np.stack(orientation_samples)
        wheel = np.stack(wheel_samples)
        front_force = np.stack(front_force_samples)
        front_normal = np.abs(front_force[..., 0])
        rear = np.stack(rear_samples)
        alive_mask = np.stack(alive_samples)
        base_initial = base0.detach().cpu().numpy()
        records = []
        for index, (root_mm, front_mm, front_drop_mm) in enumerate(combinations):
            valid = alive_mask[:, index]
            valid_indices = np.flatnonzero(valid)
            tail_indices = valid_indices[-min(20, len(valid_indices)) :]
            base_delta = base[:, index] - base_initial[index]
            wheel_delta = wheel[:, index] - wheel[0, index]
            records.append(
                {
                    "index": index,
                    "root_shift_mm": root_mm,
                    "front_penetration_mm": front_mm,
                    "front_relative_to_root_mm": front_mm - root_mm,
                    "front_drop_mm": front_drop_mm,
                    "q_target_leg": q_targets[index].tolist(),
                    "q_bias_leg": (
                        q_targets[index] - reference.joint_pos[0, :12]
                    ).tolist(),
                    "q_applied_leg": applied_q_targets[index].tolist(),
                    "q_bias_applied_leg": (
                        applied_q_targets[index]
                        - reference.joint_pos[0, :12]
                    ).tolist(),
                    "survival_steps": int(termination_step[index].item()),
                    "survived": bool(termination_step[index].item() >= args_cli.steps),
                    "base_position_max_abs_m": np.max(
                        np.abs(base_delta[valid]),
                        axis=0,
                    ).tolist(),
                    "base_position_min_delta_m": np.min(
                        base_delta[valid],
                        axis=0,
                    ).tolist(),
                    "base_position_max_delta_m": np.max(
                        base_delta[valid],
                        axis=0,
                    ).tolist(),
                    "base_position_final_delta_m": base_delta[
                        max(int(np.sum(valid)) - 1, 0)
                    ].tolist(),
                    "base_orientation_rmse_rad": float(
                        np.sqrt(
                            np.mean(
                                np.sum(
                                    orientation[valid, index] ** 2,
                                    axis=-1,
                                )
                            )
                        )
                    ),
                    "base_orientation_rotation_vector_mean_rad": np.mean(
                        orientation[valid, index],
                        axis=0,
                    ).tolist(),
                    "front_wheel_position_min_delta_m": np.min(
                        wheel_delta[valid, :2],
                        axis=0,
                    ).tolist(),
                    "front_wheel_position_max_delta_m": np.max(
                        wheel_delta[valid, :2],
                        axis=0,
                    ).tolist(),
                    "front_normal_below_6n_fraction": float(
                        np.mean(front_normal[valid, index] < 6.0)
                    ),
                    "front_normal_mean_n": np.mean(
                        front_normal[valid, index],
                        axis=0,
                    ).tolist(),
                    "front_normal_tail20_below_6n_fraction": float(
                        np.mean(front_normal[tail_indices, index] < 6.0)
                    ),
                    "front_normal_tail20_mean_n": np.mean(
                        front_normal[tail_indices, index],
                        axis=0,
                    ).tolist(),
                    "front_force_mean_w_n": np.mean(
                        front_force[valid, index],
                        axis=0,
                    ).tolist(),
                    "rear_normal_p95_n": np.percentile(
                        rear[valid, index],
                        95.0,
                        axis=0,
                    ).tolist(),
                    "rear_normal_mean_n": np.mean(
                        rear[valid, index],
                        axis=0,
                    ).tolist(),
                    "rear_normal_tail20_mean_n": np.mean(
                        rear[tail_indices, index],
                        axis=0,
                    ).tolist(),
                    "wheel_action_exact_zero": True,
                }
            )
        report = {
            "schema_version": "pcbc-low-load-static-support-sweep-v2",
            "task": args_cli.task,
            "reference_config": args_cli.reference_config,
            "ref_id": args_cli.ref_id,
            "seed": args_cli.seed,
            "steps": args_cli.steps,
            "physical_target_rate_limit_rad_s": 2.25,
            "box_root_position_local_m": box_root0.detach().cpu().numpy().tolist(),
            "initial_front_wheel_position_local_m": wheel[0, :, :2].tolist(),
            "candidates": records,
        }
        write_json(args_cli.report, report)
        for record in sorted(
            records,
            key=lambda item: (
                -item["survival_steps"],
                item["front_normal_below_6n_fraction"],
            ),
        )[:10]:
            print(record, flush=True)
        return report
    finally:
        adapter.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        write_json(
            args_cli.report.with_suffix(args_cli.report.suffix + ".failure.json"),
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
