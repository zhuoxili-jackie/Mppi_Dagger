from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch


POLICY_JOINT_ORDER = (
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

RUNTIME_JOINT_ORDER = (
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "FL_foot_joint",
    "FR_foot_joint",
    "RL_foot_joint",
    "RR_foot_joint",
)

# Definition: runtime[POLICY_TO_RUNTIME[p]] = policy[p].
POLICY_TO_RUNTIME = (0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11, 12, 13, 14, 15)
RUNTIME_GATHER_FROM_POLICY = (0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11, 12, 13, 14, 15)


def _check_last_dim(values: np.ndarray | torch.Tensor, label: str) -> None:
    if values.shape[-1] != 16:
        raise ValueError(f"{label} must have last dimension 16, got {tuple(values.shape)}")


def policy_to_runtime(values: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Scatter policy/type-grouped values into per-leg runtime order."""
    _check_last_dim(values, "policy values")
    if isinstance(values, torch.Tensor):
        result = torch.empty_like(values)
        index = torch.as_tensor(POLICY_TO_RUNTIME, dtype=torch.long, device=values.device)
        result.index_copy_(-1, index, values)
        return result
    result = np.empty_like(values)
    result[..., np.asarray(POLICY_TO_RUNTIME)] = values
    return result


def runtime_to_policy(values: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Gather per-leg runtime values into policy/type-grouped order."""
    _check_last_dim(values, "runtime values")
    if isinstance(values, torch.Tensor):
        index = torch.as_tensor(POLICY_TO_RUNTIME, dtype=torch.long, device=values.device)
        return values.index_select(-1, index)
    return values[..., np.asarray(POLICY_TO_RUNTIME)]


def index_map(source_order: Sequence[str], target_order: Sequence[str]) -> tuple[int, ...]:
    if len(source_order) != len(set(source_order)) or len(target_order) != len(set(target_order)):
        raise ValueError("Joint orders must not contain duplicate names.")
    if set(source_order) != set(target_order):
        missing = sorted(set(source_order) - set(target_order))
        extra = sorted(set(target_order) - set(source_order))
        raise ValueError(f"Joint order mismatch: missing={missing}, extra={extra}")
    return tuple(target_order.index(name) for name in source_order)

