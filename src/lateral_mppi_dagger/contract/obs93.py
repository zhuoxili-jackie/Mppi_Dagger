from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch


class MotionPrefixSemantics(str, Enum):
    FIXED_FIRST_FRAME = "fixed_first_frame"
    DYNAMIC_REFERENCE = "dynamic_reference"


def _as_float32(value: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not torch.is_floating_point(value):
        raise TypeError(f"{name} must be floating point, got {value.dtype}")
    return value.to(dtype=torch.float32)


def _check_last_dim(value: torch.Tensor, size: int, name: str) -> None:
    if value.ndim < 1 or value.shape[-1] != size:
        raise ValueError(f"{name} must end in dimension {size}, got {tuple(value.shape)}")


def normalize_quat_wxyz(quat: torch.Tensor) -> torch.Tensor:
    _check_last_dim(quat, 4, "quaternion")
    norm = torch.linalg.vector_norm(quat, dim=-1, keepdim=True)
    if torch.any(norm < 1.0e-8):
        raise ValueError("Quaternion norm is too close to zero.")
    return quat / norm


def quat_conjugate_wxyz(quat: torch.Tensor) -> torch.Tensor:
    result = quat.clone()
    result[..., 1:] = -result[..., 1:]
    return result


def quat_multiply_wxyz(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    lw, lx, ly, lz = lhs.unbind(dim=-1)
    rw, rx, ry, rz = rhs.unbind(dim=-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def quat_to_matrix_wxyz(quat: torch.Tensor) -> torch.Tensor:
    quat = normalize_quat_wxyz(quat)
    w, x, y, z = quat.unbind(dim=-1)
    two = 2.0
    return torch.stack(
        (
            1.0 - two * (y * y + z * z),
            two * (x * y - z * w),
            two * (x * z + y * w),
            two * (x * y + z * w),
            1.0 - two * (x * x + z * z),
            two * (y * z - x * w),
            two * (x * z - y * w),
            two * (y * z + x * w),
            1.0 - two * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(quat.shape[:-1] + (3, 3))


def relative_rotation_6d_columns(
    robot_anchor_quat_wxyz: torch.Tensor,
    reference_anchor_quat_wxyz: torch.Tensor,
) -> torch.Tensor:
    """Match Isaac/部署端 ``R[..., :2].reshape(...)`` exactly.

    For a tensor with shape ``[..., 3, 3]``, PyTorch's ``R[..., :2]`` selects
    the first two *columns* (the slice applies to the final dimension).  The
    C-order result is ``[R00, R01, R10, R11, R20, R21]``.  Spelling both
    matrix dimensions out here prevents the row/column ambiguity that caused
    the v1 deployment mismatch.
    """
    robot = normalize_quat_wxyz(robot_anchor_quat_wxyz)
    reference = normalize_quat_wxyz(reference_anchor_quat_wxyz)
    relative = quat_multiply_wxyz(quat_conjugate_wxyz(robot), reference)
    matrix = quat_to_matrix_wxyz(relative)
    return matrix[..., :, :2].reshape(matrix.shape[:-2] + (6,))


@dataclass(frozen=True)
class Obs93Input:
    robot_anchor_quat_wxyz: torch.Tensor
    reference_anchor_quat_wxyz: torch.Tensor
    base_ang_vel_b: torch.Tensor
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    default_joint_pos: torch.Tensor
    default_joint_vel: torch.Tensor
    previous_executed_raw_action: torch.Tensor
    velocity_command: torch.Tensor
    reference_joint_pos: torch.Tensor
    reference_joint_vel: torch.Tensor


class Obs93Builder:
    """Single authoritative construction of the raw, noise-free 93D deployment observation."""

    OBS_DIM = 93
    WHEEL_POLICY_SLICE = slice(12, 16)
    WHEEL_POSITION_OBS_SLICE = slice(53, 57)

    def __init__(
        self,
        fixed_joint_pos: torch.Tensor,
        fixed_joint_vel: torch.Tensor,
        fixed_anchor_quat_wxyz: torch.Tensor,
        semantics: MotionPrefixSemantics | str = MotionPrefixSemantics.FIXED_FIRST_FRAME,
    ) -> None:
        self.semantics = MotionPrefixSemantics(semantics)
        self.fixed_joint_pos = _as_float32(fixed_joint_pos, "fixed_joint_pos")
        self.fixed_joint_vel = _as_float32(fixed_joint_vel, "fixed_joint_vel")
        self.fixed_anchor_quat_wxyz = _as_float32(fixed_anchor_quat_wxyz, "fixed_anchor_quat_wxyz")
        _check_last_dim(self.fixed_joint_pos, 16, "fixed_joint_pos")
        _check_last_dim(self.fixed_joint_vel, 16, "fixed_joint_vel")
        _check_last_dim(self.fixed_anchor_quat_wxyz, 4, "fixed_anchor_quat_wxyz")

    @staticmethod
    def _expand_to_batch(value: torch.Tensor, batch_shape: torch.Size, size: int, name: str) -> torch.Tensor:
        value = _as_float32(value, name)
        _check_last_dim(value, size, name)
        target = tuple(batch_shape) + (size,)
        try:
            return torch.broadcast_to(value, target)
        except RuntimeError as exc:
            raise ValueError(f"{name} with shape {tuple(value.shape)} cannot broadcast to {target}") from exc

    def build(self, values: Obs93Input) -> torch.Tensor:
        joint_pos = _as_float32(values.joint_pos, "joint_pos")
        _check_last_dim(joint_pos, 16, "joint_pos")
        batch_shape = joint_pos.shape[:-1]

        joint_vel = self._expand_to_batch(values.joint_vel, batch_shape, 16, "joint_vel")
        default_joint_pos = self._expand_to_batch(
            values.default_joint_pos, batch_shape, 16, "default_joint_pos"
        )
        default_joint_vel = self._expand_to_batch(
            values.default_joint_vel, batch_shape, 16, "default_joint_vel"
        )
        previous = self._expand_to_batch(
            values.previous_executed_raw_action, batch_shape, 16, "previous_executed_raw_action"
        )
        velocity = self._expand_to_batch(values.velocity_command, batch_shape, 3, "velocity_command")
        robot_quat = self._expand_to_batch(
            values.robot_anchor_quat_wxyz, batch_shape, 4, "robot_anchor_quat_wxyz"
        )

        if self.semantics is MotionPrefixSemantics.FIXED_FIRST_FRAME:
            prefix_pos = self._expand_to_batch(self.fixed_joint_pos, batch_shape, 16, "fixed_joint_pos")
            prefix_vel = self._expand_to_batch(self.fixed_joint_vel, batch_shape, 16, "fixed_joint_vel")
            reference_quat = self._expand_to_batch(
                self.fixed_anchor_quat_wxyz, batch_shape, 4, "fixed_anchor_quat_wxyz"
            )
        else:
            prefix_pos = self._expand_to_batch(
                values.reference_joint_pos, batch_shape, 16, "reference_joint_pos"
            )
            prefix_vel = self._expand_to_batch(
                values.reference_joint_vel, batch_shape, 16, "reference_joint_vel"
            )
            reference_quat = self._expand_to_batch(
                values.reference_anchor_quat_wxyz, batch_shape, 4, "reference_anchor_quat_wxyz"
            )

        joint_pos_rel = joint_pos - default_joint_pos
        joint_pos_rel = joint_pos_rel.clone()
        joint_pos_rel[..., self.WHEEL_POLICY_SLICE] = 0.0
        joint_vel_rel = joint_vel - default_joint_vel
        rotation_6d = relative_rotation_6d_columns(robot_quat, reference_quat)
        legacy_zero = torch.zeros(batch_shape + (1,), dtype=torch.float32, device=joint_pos.device)

        observation = torch.cat(
            (
                prefix_pos,
                prefix_vel,
                rotation_6d,
                self._expand_to_batch(values.base_ang_vel_b, batch_shape, 3, "base_ang_vel_b"),
                joint_pos_rel,
                joint_vel_rel,
                previous,
                velocity,
                legacy_zero,
            ),
            dim=-1,
        )
        if observation.shape[-1] != self.OBS_DIM:
            raise AssertionError(f"Internal Obs93 assembly error: got {tuple(observation.shape)}")
        if observation.dtype != torch.float32:
            raise AssertionError(f"Obs93 must be float32, got {observation.dtype}")
        if not torch.isfinite(observation).all():
            raise ValueError("Obs93 contains NaN or Inf.")
        return observation
