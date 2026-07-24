from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from lateral_mppi_dagger.contract.obs93 import (
    MotionPrefixSemantics,
    Obs93Builder,
    Obs93Input,
    normalize_quat_wxyz,
    quat_conjugate_wxyz,
    quat_multiply_wxyz,
)


def _yaw_only_wxyz(quaternion: torch.Tensor) -> torch.Tensor:
    quaternion = normalize_quat_wxyz(quaternion)
    w, x, y, z = quaternion.unbind(dim=-1)
    yaw = torch.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )
    zeros = torch.zeros_like(yaw)
    return torch.stack(
        (torch.cos(0.5 * yaw), zeros, zeros, torch.sin(0.5 * yaw)),
        dim=-1,
    )


def align_reference_yaw_like_deployment(
    robot_quaternion_wxyz: torch.Tensor,
    reference_quaternion_wxyz: torch.Tensor,
) -> torch.Tensor:
    robot = normalize_quat_wxyz(robot_quaternion_wxyz)
    reference = normalize_quat_wxyz(reference_quaternion_wxyz)
    world_to_initial = quat_multiply_wxyz(
        _yaw_only_wxyz(robot),
        quat_conjugate_wxyz(_yaw_only_wxyz(reference)),
    )
    return normalize_quat_wxyz(
        quat_multiply_wxyz(world_to_initial, reference)
    )


@dataclass(frozen=True)
class Key7HandoffObservation:
    observation: np.ndarray
    aligned_reference_quaternion_wxyz: np.ndarray


def build_key7_handoff_observation(
    contract: dict,
    golden: dict,
) -> Key7HandoffObservation:
    handoff = golden["nominal_handoff"]
    fixed_pos = torch.as_tensor(
        contract["motion_prefix"]["first_frame_joint_pos"],
        dtype=torch.float32,
    )
    fixed_vel = torch.as_tensor(
        contract["motion_prefix"]["first_frame_joint_vel"],
        dtype=torch.float32,
    )
    reference_quaternion = torch.as_tensor(
        contract["motion_prefix"]["first_frame_anchor_quat_wxyz"],
        dtype=torch.float32,
    )
    robot_quaternion = torch.as_tensor(
        handoff["robot_quaternion_wxyz"],
        dtype=torch.float32,
    )
    aligned_reference = align_reference_yaw_like_deployment(
        robot_quaternion,
        reference_quaternion,
    )
    builder = Obs93Builder(
        fixed_pos,
        fixed_vel,
        aligned_reference,
        MotionPrefixSemantics.FIXED_FIRST_FRAME,
    )
    observation = builder.build(
        Obs93Input(
            robot_anchor_quat_wxyz=robot_quaternion,
            reference_anchor_quat_wxyz=aligned_reference,
            base_ang_vel_b=torch.as_tensor(
                handoff["base_angular_velocity_body"],
                dtype=torch.float32,
            ),
            joint_pos=torch.as_tensor(
                handoff["robot_joint_position_policy"],
                dtype=torch.float32,
            ),
            joint_vel=torch.as_tensor(
                handoff["robot_joint_velocity_policy"],
                dtype=torch.float32,
            ),
            default_joint_pos=fixed_pos,
            default_joint_vel=torch.zeros_like(fixed_vel),
            previous_executed_raw_action=torch.as_tensor(
                handoff["previous_raw_action"],
                dtype=torch.float32,
            ),
            velocity_command=torch.as_tensor(
                handoff["velocity_command"],
                dtype=torch.float32,
            ),
            reference_joint_pos=fixed_pos,
            reference_joint_vel=fixed_vel,
        )
    )
    return Key7HandoffObservation(
        observation=(
            observation.detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=True)
            .reshape(1, 93)
        ),
        aligned_reference_quaternion_wxyz=(
            aligned_reference.detach().cpu().numpy().astype(np.float32, copy=True)
        ),
    )


def replay_key7_dry_inference(
    observation: np.ndarray,
    policy: Callable[[np.ndarray], np.ndarray],
    dry_cycles: int,
    action_scale: np.ndarray,
) -> dict:
    if dry_cycles < 0:
        raise ValueError("dry_cycles must be non-negative.")
    current = np.asarray(observation, dtype=np.float32).reshape(1, 93).copy()
    scale = np.asarray(action_scale, dtype=np.float32)
    if scale.shape != (16,):
        raise ValueError(f"action_scale must have shape (16,), got {scale.shape}.")
    records = []
    for cycle in range(1, dry_cycles + 2):
        action = np.asarray(policy(current), dtype=np.float32).reshape(16)
        if not np.isfinite(action).all():
            raise ValueError(f"Policy produced non-finite action at cycle {cycle}.")
        records.append(
            {
                "cycle": cycle,
                "applied": cycle > dry_cycles,
                "raw_action": action.tolist(),
                "raw_leg_max_abs": float(np.max(np.abs(action[:12]))),
                "physical_leg_delta_max_abs_rad": float(
                    np.max(np.abs(action[:12] * scale[:12]))
                ),
                "wheel_action_max_abs": float(np.max(np.abs(action[12:]))),
            }
        )
        current[0, 73:89] = action
    return {
        "dry_cycles": dry_cycles,
        "cycles": records,
        "first_applied": records[-1],
    }
