from __future__ import annotations

import time
from dataclasses import dataclass, fields
from typing import Any

import numpy as np
import torch

from lateral_mppi_dagger.contract.action16 import Action16Adapter, ActionContract, WheelActionMode
from lateral_mppi_dagger.expert.base import (
    MPPI_COST_COMPONENT_NAMES,
    ExpertReply,
    ExpertRequest,
    FailureCode,
)
from lateral_mppi_dagger.expert.mppi_expert import MPPIConfig, ReferenceCenteredMPPI
from lateral_mppi_dagger.reference.action_reference import (
    normalize_nominal_solver_overrides,
    resolve_nominal_solver_overrides,
)
from lateral_mppi_dagger.reference.loader import ReferenceSet
from lateral_mppi_dagger.env.action_delay import advance_action_delay


def _quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat((q[..., :1], -q[..., 1:]), dim=-1)


def _quat_multiply(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
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


def _quat_rotate(q: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    q_vector = q[..., 1:]
    uv = torch.cross(q_vector, vector, dim=-1)
    uuv = torch.cross(q_vector, uv, dim=-1)
    return vector + 2.0 * (q[..., :1] * uv + uuv)


def _quat_angle(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    dot = torch.sum(lhs * rhs, dim=-1).abs().clamp(max=1.0)
    return 2.0 * torch.acos(dot)


def _quat_rotation_vector(
    actual_quat_w: torch.Tensor,
    target_quat_w: torch.Tensor,
) -> torch.Tensor:
    """Return the target-frame shortest-rotation vector in radians.

    Both inputs use the frozen Isaac/deployment ``wxyz`` convention.  The
    relative rotation maps the target orientation to the actual orientation.
    The vector direction is the target-frame rotation axis and its magnitude
    is the same shortest quaternion angle used by the formal tracking gate.
    """

    actual = actual_quat_w / torch.linalg.vector_norm(
        actual_quat_w,
        dim=-1,
        keepdim=True,
    ).clamp_min(torch.finfo(actual_quat_w.dtype).eps)
    target = target_quat_w / torch.linalg.vector_norm(
        target_quat_w,
        dim=-1,
        keepdim=True,
    ).clamp_min(torch.finfo(target_quat_w.dtype).eps)
    relative = _quat_multiply(_quat_conjugate(target), actual)
    relative = torch.where(
        relative[..., :1] < 0.0,
        -relative,
        relative,
    )
    vector = relative[..., 1:]
    vector_norm = torch.linalg.vector_norm(
        vector,
        dim=-1,
        keepdim=True,
    )
    angle = 2.0 * torch.atan2(
        vector_norm,
        relative[..., :1].clamp_min(
            torch.finfo(relative.dtype).eps
        ),
    )
    rotation_vector = torch.where(
        vector_norm > torch.finfo(relative.dtype).eps,
        vector * (angle / vector_norm.clamp_min(
            torch.finfo(relative.dtype).eps
        )),
        2.0 * vector,
    )
    return rotation_vector


def base_orientation_tracking_cost(
    actual_quat_w: torch.Tensor,
    target_quat_w: torch.Tensor,
    axis_multipliers: torch.Tensor,
) -> torch.Tensor:
    """Return squared orientation error with optional target-axis emphasis.

    Exact unit multipliers retain the original quaternion-angle computation.
    This makes the new solver option a no-op for every existing config while
    allowing evidence-backed emphasis of one target-frame rotation axis.
    """

    if axis_multipliers.shape != (3,):
        raise ValueError(
            "base orientation axis multipliers must have shape [3]."
        )
    if (
        not torch.isfinite(axis_multipliers).all()
        or torch.any(axis_multipliers < 1.0)
    ):
        raise ValueError(
            "base orientation axis multipliers must be finite and at least "
            "1.0."
        )
    if bool(torch.equal(
        axis_multipliers,
        torch.ones_like(axis_multipliers),
    )):
        return _quat_angle(actual_quat_w, target_quat_w).square()
    rotation_vector = _quat_rotation_vector(
        actual_quat_w,
        target_quat_w,
    )
    return (
        rotation_vector.square()
        * axis_multipliers.to(
            device=rotation_vector.device,
            dtype=rotation_vector.dtype,
        )
    ).sum(dim=-1)


def _signed_orientation_axis_error_rad(
    actual_quat_w: torch.Tensor,
    target_quat_w: torch.Tensor,
    axis_index: int,
) -> torch.Tensor:
    """Return one target-frame component of the shortest rotation."""

    if axis_index not in (0, 1, 2):
        raise ValueError("orientation feedback axis index must be 0, 1, or 2.")

    return _quat_rotation_vector(
        actual_quat_w,
        target_quat_w,
    )[..., axis_index]


def _signed_pitch_error_rad(
    actual_quat_w: torch.Tensor,
    target_quat_w: torch.Tensor,
) -> torch.Tensor:
    """Return the target-frame pitch component of the shortest rotation."""

    return _signed_orientation_axis_error_rad(
        actual_quat_w,
        target_quat_w,
        1,
    )


def _tree_env0_clone(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value[0:1].detach().clone()
    if isinstance(value, dict):
        return {key: _tree_env0_clone(item) for key, item in value.items()}
    raise TypeError(f"Unsupported Isaac scene-state value {type(value)!r}")


def _tree_repeat(value: Any, count: int) -> Any:
    if isinstance(value, torch.Tensor):
        repeats = (count,) + (1,) * (value.ndim - 1)
        return value.repeat(repeats)
    if isinstance(value, dict):
        return {key: _tree_repeat(item, count) for key, item in value.items()}
    raise TypeError(f"Unsupported Isaac scene-state value {type(value)!r}")


@dataclass(frozen=True)
class IsaacRolloutCostWeights:
    base_position: float = 35.0
    base_height_drop: float = 0.0
    base_orientation: float = 18.0
    joint_position: float = 10.0
    rear_leg_position: float = 0.0
    joint_velocity: float = 0.15
    wheel_position: float = 45.0
    rear_swing_lateral_position: float = 0.0
    rear_swing_height_deficit: float = 0.0
    lateral_velocity: float = 4.0
    lateral_position: float = 0.0
    box_x_drift: float = 60.0
    wheel_slip: float = 0.30
    contact_mismatch: float = 2.0
    front_normal_support: float = 8.0
    rear_force_overload: float = 18.0
    rear_force_imbalance: float = 3.0
    rear_support_loss: float = 30.0
    rear_swing_force: float = 0.0
    edge_drop: float = 120.0
    action_residual: float = 0.04
    action_rate: float = 0.025
    joint_acceleration: float = 2.0e-6
    torque_limit: float = 0.20
    joint_limit: float = 150.0
    termination: float = 2500.0
    terminal: float = 8.0
    terminal_base_orientation: float = 0.0

    @classmethod
    def from_dict(cls, values: dict[str, Any] | None) -> "IsaacRolloutCostWeights":
        if not values:
            return cls()
        known = {field.name for field in fields(cls)}
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(f"Unknown MPPI cost weights: {unknown}")
        result = cls(**{name: float(value) for name, value in values.items()})
        if any(
            not np.isfinite(getattr(result, field.name))
            or getattr(result, field.name) < 0.0
            for field in fields(cls)
        ):
            raise ValueError(
                "MPPI cost weights must be non-negative and finite."
            )
        return result


@dataclass(frozen=True)
class IsaacRolloutLoadLimits:
    base_height_drop_margin_m: float = 0.08
    base_height_drop_stop_frame: float = 0.0
    lateral_position_start_frame: float = 0.0
    rear_swing_lateral_position_start_frame: float = 0.0
    rear_swing_height_deficit_start_frame: float = 0.0
    rear_swing_height_scale_m: float = 0.012
    lateral_velocity_absolute_scale_m_s: float = 0.0
    wheel_position_worst_fraction: float = 0.0
    front_normal_min_n: float = 6.0
    front_normal_deficit_power: float = 2.0
    front_normal_low_force_threshold_n: float = 0.0
    front_normal_low_force_count_penalty: float = 0.0
    front_support_worst_fraction: float = 0.0
    front_force_balance_scale_n: float = 0.0
    front_contact_position_margin_m: float = 0.0
    front_contact_position_scale_m: float = 0.0
    front_contact_position_max_normalized: float = 0.0
    rear_normal_overload_n: float = 105.0
    rear_normal_scale_n: float = 35.0
    rear_overload_worst_fraction: float = 0.0
    rear_balance_scale_n: float = 70.0
    rear_swing_action_residual_multiplier: float = 1.0

    @classmethod
    def from_dict(
        cls,
        values: dict[str, Any] | None,
    ) -> "IsaacRolloutLoadLimits":
        if not values:
            return cls()
        known = {field.name for field in fields(cls)}
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(f"Unknown MPPI load limits: {unknown}")
        result = cls(**{name: float(value) for name, value in values.items()})
        scheduled_nonnegative = (
            result.base_height_drop_margin_m,
            result.base_height_drop_stop_frame,
            result.lateral_position_start_frame,
            result.rear_swing_lateral_position_start_frame,
            result.rear_swing_height_deficit_start_frame,
        )
        if (
            any(
                not np.isfinite(value) or value < 0.0
                for value in scheduled_nonnegative
            )
            or not np.isfinite(
                result.lateral_velocity_absolute_scale_m_s
            )
            or result.lateral_velocity_absolute_scale_m_s < 0.0
            or result.front_normal_min_n < 0.0
            or not np.isfinite(result.rear_swing_height_scale_m)
            or result.rear_swing_height_scale_m <= 0.0
        ):
            raise ValueError(
                "Scheduled tracking frames, base-height margin, lateral "
                "velocity absolute scale, and front-normal minimum must be "
                "finite and non-negative, and rear-swing height scale must "
                "be finite and positive."
            )
        if not 1.0 <= result.front_normal_deficit_power <= 2.0:
            raise ValueError(
                "front_normal_deficit_power must be in [1, 2]."
            )
        if (
            not np.isfinite(result.front_normal_low_force_threshold_n)
            or not np.isfinite(
                result.front_normal_low_force_count_penalty
            )
            or result.front_normal_low_force_threshold_n < 0.0
            or result.front_normal_low_force_count_penalty < 0.0
        ):
            raise ValueError(
                "Front low-force threshold and count penalty must be finite "
                "and non-negative."
            )
        if (
            result.front_normal_low_force_threshold_n == 0.0
        ) != (
            result.front_normal_low_force_count_penalty == 0.0
        ):
            raise ValueError(
                "Front low-force threshold and count penalty must either "
                "both be zero or both be positive."
            )
        if not 0.0 <= result.front_support_worst_fraction <= 1.0:
            raise ValueError(
                "front_support_worst_fraction must be in [0, 1]."
            )
        if not 0.0 <= result.wheel_position_worst_fraction <= 1.0:
            raise ValueError(
                "wheel_position_worst_fraction must be in [0, 1]."
            )
        if (
            result.front_force_balance_scale_n < 0.0
            or result.front_contact_position_margin_m < 0.0
            or result.front_contact_position_scale_m < 0.0
            or result.front_contact_position_max_normalized < 0.0
        ):
            raise ValueError(
                "Front force-balance scale and contact-position margin, "
                "scale, and cap must be non-negative."
            )
        if (
            result.rear_normal_overload_n <= 0.0
            or result.rear_normal_scale_n <= 0.0
            or result.rear_balance_scale_n <= 0.0
        ):
            raise ValueError("Rear-force load limits and scales must be positive.")
        if not 0.0 <= result.rear_overload_worst_fraction <= 1.0:
            raise ValueError(
                "rear_overload_worst_fraction must be in [0, 1]."
            )
        if not 0.0 <= result.rear_swing_action_residual_multiplier <= 1.0:
            raise ValueError(
                "rear_swing_action_residual_multiplier must be in [0, 1]."
            )
        return result


def lateral_velocity_tracking_cost(
    velocity_error_m_s: torch.Tensor,
    absolute_scale_m_s: float = 0.0,
) -> torch.Tensor:
    """Blend the squared tracking objective with a continuous MAE term."""

    scale = float(absolute_scale_m_s)
    if not np.isfinite(scale) or scale < 0.0:
        raise ValueError(
            "absolute_scale_m_s must be finite and non-negative."
        )
    squared = velocity_error_m_s.square()
    if scale == 0.0:
        return squared
    return squared + scale * torch.abs(velocity_error_m_s)


def base_height_drop_cost(
    base_position_error: torch.Tensor,
    margin_m: float,
) -> torch.Tensor:
    """Penalize only downward base error beyond a protected height margin."""

    if base_position_error.ndim != 2 or base_position_error.shape[1] != 3:
        raise ValueError("base_position_error must have shape [batch,3].")
    if margin_m < 0.0:
        raise ValueError("margin_m must be non-negative.")
    return torch.relu(
        -base_position_error[:, 2] - margin_m
    ).square()


def rear_leg_position_cost(joint_position_error: torch.Tensor) -> torch.Tensor:
    """Track the rear legs in type-grouped policy joint order."""

    if (
        joint_position_error.ndim != 2
        or joint_position_error.shape[1] != 12
    ):
        raise ValueError(
            "joint_position_error must have shape [batch,12]."
        )
    rear_leg_indices = (2, 3, 6, 7, 10, 11)
    return joint_position_error[:, rear_leg_indices].square().mean(dim=-1)


def scheduled_rear_wheel_lateral_position_cost(
    wheel_position_error_w: torch.Tensor,
    desired_contact: torch.Tensor,
) -> torch.Tensor:
    """Track a rear wheel laterally only while it is scheduled to swing."""

    if wheel_position_error_w.ndim != 3 or (
        wheel_position_error_w.shape[1:] != (4, 3)
    ):
        raise ValueError(
            "wheel_position_error_w must have shape [batch,4,3]."
        )
    if desired_contact.shape != wheel_position_error_w.shape[:2]:
        raise ValueError("desired_contact must have shape [batch,4].")

    rear_swing = ~desired_contact[:, 2:].to(dtype=torch.bool)
    rear_swing_float = rear_swing.to(dtype=wheel_position_error_w.dtype)
    rear_lateral_error = wheel_position_error_w[:, 2:, 1]
    return (
        rear_lateral_error.square() * rear_swing_float
    ).sum(dim=-1) / rear_swing_float.sum(dim=-1).clamp_min(1.0)


def scheduled_rear_wheel_height_deficit_cost(
    wheel_position_error_w: torch.Tensor,
    desired_contact: torch.Tensor,
    height_scale_m: float,
) -> torch.Tensor:
    """Penalize missing height only for a scheduled rear-wheel swing.

    The target remains the frozen reference wheel height.  Positive vertical
    error (more clearance than requested) is not rewarded or penalized here;
    this term only exposes a below-target height deficit to the optimizer.
    """

    if wheel_position_error_w.ndim != 3 or (
        wheel_position_error_w.shape[1:] != (4, 3)
    ):
        raise ValueError(
            "wheel_position_error_w must have shape [batch,4,3]."
        )
    if desired_contact.shape != wheel_position_error_w.shape[:2]:
        raise ValueError("desired_contact must have shape [batch,4].")
    scale = float(height_scale_m)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("height_scale_m must be finite and positive.")

    rear_swing = ~desired_contact[:, 2:].to(dtype=torch.bool)
    rear_swing_float = rear_swing.to(dtype=wheel_position_error_w.dtype)
    rear_height_deficit = torch.relu(
        -wheel_position_error_w[:, 2:, 2]
    ) / scale
    return (
        rear_height_deficit.square() * rear_swing_float
    ).sum(dim=-1) / rear_swing_float.sum(dim=-1).clamp_min(1.0)


def scheduled_action_residual_cost(
    action_residual: torch.Tensor,
    desired_contact: torch.Tensor,
    rear_swing_multiplier: float,
    rear_swing_active: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reduce nominal regularization only for a scheduled rear swing limb."""

    if (
        action_residual.ndim != 2
        or action_residual.shape[1] != 12
    ):
        raise ValueError("action_residual must have shape [batch,12].")
    if desired_contact.shape != (action_residual.shape[0], 4):
        raise ValueError("desired_contact must have shape [batch,4].")
    multiplier = float(rear_swing_multiplier)
    if not np.isfinite(multiplier) or not 0.0 <= multiplier <= 1.0:
        raise ValueError("rear_swing_multiplier must be in [0, 1].")

    per_joint_multiplier = torch.ones_like(action_residual)
    desired = desired_contact.to(dtype=torch.bool)
    if rear_swing_active is None:
        active = ~desired[:, 2:4]
    else:
        if rear_swing_active.shape != (action_residual.shape[0], 2):
            raise ValueError(
                "rear_swing_active must have shape [batch,2]."
            )
        active = rear_swing_active.to(dtype=torch.bool)
    for rear_index, joint_indices in (
        (0, (2, 6, 10)),
        (1, (3, 7, 11)),
    ):
        swing = active[:, rear_index].to(
            dtype=action_residual.dtype
        )
        per_joint_multiplier[:, list(joint_indices)] = (
            1.0 - (1.0 - multiplier) * swing.unsqueeze(-1)
        )
    return (
        action_residual.square() * per_joint_multiplier
    ).mean(dim=-1)


def wheel_position_tracking_cost(
    wheel_position_error_w: torch.Tensor,
    worst_fraction: float,
) -> torch.Tensor:
    """Blend mean-wheel and worst-wheel position error.

    Each wheel first contributes its three-axis mean-square error.  This keeps
    the legacy scale exactly when ``worst_fraction`` is zero while allowing a
    candidate config to align the optimizer with the gate's per-wheel maximum.
    """

    if (
        wheel_position_error_w.ndim != 3
        or wheel_position_error_w.shape[1:] != (4, 3)
    ):
        raise ValueError(
            "wheel_position_error_w must have shape [batch,4,3]."
        )
    if not 0.0 <= worst_fraction <= 1.0:
        raise ValueError("worst_fraction must be in [0, 1].")
    per_wheel_cost = wheel_position_error_w.square().mean(dim=-1)
    mean_cost = per_wheel_cost.mean(dim=-1)
    worst_cost = per_wheel_cost.max(dim=-1).values
    return (
        (1.0 - worst_fraction) * mean_cost
        + worst_fraction * worst_cost
    )


def select_global_best_cost_components(
    costs: torch.Tensor,
    components: dict[str, torch.Tensor],
    previous_best_cost: float = float("inf"),
    previous_best_components: dict[str, float] | None = None,
) -> tuple[float, dict[str, float]]:
    """Retain the component vector for the best sample across MPPI calls."""

    if costs.ndim != 1 or costs.numel() == 0:
        raise ValueError("costs must be a non-empty one-dimensional tensor.")
    if not components:
        raise ValueError("components must not be empty.")
    for name, values in components.items():
        if values.shape != costs.shape:
            raise ValueError(
                f"Cost component {name!r} must have shape {tuple(costs.shape)}, "
                f"got {tuple(values.shape)}."
            )
    best_index = int(torch.argmin(costs).item())
    candidate_cost = float(costs[best_index].item())
    if candidate_cost >= previous_best_cost:
        return (
            float(previous_best_cost),
            dict(previous_best_components or {}),
        )
    return candidate_cost, {
        name: float(values[best_index].item())
        for name, values in components.items()
    }


def structured_candidate_cost_diagnostics(
    costs: torch.Tensor,
    components: dict[str, torch.Tensor],
    proposal_count: int,
) -> dict[str, Any]:
    """Return component-resolved costs for the tail proposal population."""

    if costs.ndim != 1 or costs.numel() < 2:
        raise ValueError("costs must contain at least two samples.")
    count = int(proposal_count)
    if count < 1 or count >= costs.numel():
        raise ValueError(
            "proposal_count must leave at least one stochastic sample."
        )
    if not torch.isfinite(costs).all():
        raise ValueError("costs contains NaN or Inf.")
    for name, values in components.items():
        if values.shape != costs.shape:
            raise ValueError(
                f"Cost component {name!r} must have shape "
                f"{tuple(costs.shape)}, got {tuple(values.shape)}."
            )
        if not torch.isfinite(values).all():
            raise ValueError(
                f"Cost component {name!r} contains NaN or Inf."
            )

    proposal_start = int(costs.numel()) - count
    proposal_costs = costs[proposal_start:]
    iteration_minimum = torch.min(costs)
    return {
        "proposal_count": count,
        "total_costs": proposal_costs.detach().cpu().tolist(),
        "cost_gap_from_iteration_best": (
            proposal_costs - iteration_minimum
        ).detach().cpu().tolist(),
        "iteration_minimum_total_cost": float(
            iteration_minimum.detach().cpu().item()
        ),
        "stochastic_minimum_total_cost": float(
            torch.min(costs[:proposal_start]).detach().cpu().item()
        ),
        "structured_minimum_total_cost": float(
            torch.min(proposal_costs).detach().cpu().item()
        ),
        "cost_components": {
            name: values[proposal_start:].detach().cpu().tolist()
            for name, values in components.items()
        },
    }


def load_support_cost_terms(
    contact_force_w: torch.Tensor,
    desired_contact: torch.Tensor,
    contact_force_threshold_n: float,
    limits: IsaacRolloutLoadLimits,
    wheel_position_error_w: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Return orientation-aware trunk/ground support penalties.

    The robot is nearly vertical in this task.  Front wheels press against the
    trunk mainly along world x, while rear wheels carry gravity mainly along
    world z.  Force-vector magnitude alone cannot distinguish those roles.
    """

    if contact_force_w.ndim != 3 or contact_force_w.shape[1:] != (4, 3):
        raise ValueError("contact_force_w must have shape [batch,4,3].")
    if desired_contact.shape != contact_force_w.shape[:2]:
        raise ValueError("desired_contact must have shape [batch,4].")
    if contact_force_threshold_n <= 0.0:
        raise ValueError("contact_force_threshold_n must be positive.")

    front_normal = torch.abs(contact_force_w[:, :2, 0])
    rear_normal = torch.abs(contact_force_w[:, 2:, 2])
    desired_front = desired_contact[:, :2].float()
    front_denominator = max(limits.front_normal_min_n, 1.0)
    front_deficit = torch.relu(
        limits.front_normal_min_n - front_normal
    ) / front_denominator
    front_deficit_cost = (
        front_deficit.pow(limits.front_normal_deficit_power)
        * desired_front
    )
    front_normal_support_mean = (
        front_deficit_cost.sum(dim=-1)
        / desired_front.sum(dim=-1).clamp_min(1.0)
    )
    front_normal_support_worst = front_deficit_cost.max(dim=-1).values
    front_normal_support = (
        (1.0 - limits.front_support_worst_fraction)
        * front_normal_support_mean
        + limits.front_support_worst_fraction
        * front_normal_support_worst
    )
    if limits.front_normal_low_force_count_penalty > 0.0:
        # The formal gate counts every desired front-wheel sample below its
        # force threshold.  The ordinary deficit term measures total force
        # shortfall instead, so a candidate can otherwise trade several
        # near-threshold misses for one large transient.  This optional,
        # bounded per-step count uses the identical world-X normal and desired
        # contact semantics while keeping the legacy path bit-exact at zero.
        low_force_count = (
            (
                front_normal
                < limits.front_normal_low_force_threshold_n
            ).float()
            * desired_front
        ).sum(dim=-1)
        front_normal_support = front_normal_support + (
            limits.front_normal_low_force_count_penalty
            * low_force_count
        )
    if limits.front_force_balance_scale_n > 0.0:
        # A force deficit already asks each desired front contact to carry the
        # minimum load, but a low-ESS MPPI solve can still settle on one strong
        # and one weak front wheel.  Add a bounded balance proxy only while
        # both front wheels are scheduled as support; single-front swing
        # phases must remain free to unload the moving limb.
        front_imbalance = torch.clamp(
            (front_normal[:, 0] - front_normal[:, 1])
            / limits.front_force_balance_scale_n,
            min=-1.0,
            max=1.0,
        ).square()
        both_front_desired = torch.all(desired_contact[:, :2], dim=-1).float()
        front_normal_support = (
            front_normal_support + front_imbalance * both_front_desired
        )
    if limits.front_contact_position_scale_m > 0.0:
        if (
            wheel_position_error_w is None
            or wheel_position_error_w.shape != contact_force_w.shape
        ):
            raise ValueError(
                "wheel_position_error_w must have shape [batch,4,3] when "
                "front contact-position support is enabled."
            )
        # The trunk is on the +world-X side of the robot.  A negative front
        # wheel X error is therefore detachment, not harmless tangential slip.
        # Gate the smooth geometric proxy by the kinematic contact schedule so
        # the deliberate 8 mm front swing remains unpenalized.
        front_detachment = torch.relu(
            -wheel_position_error_w[:, :2, 0]
            - limits.front_contact_position_margin_m
        ) / limits.front_contact_position_scale_m
        if limits.front_contact_position_max_normalized > 0.0:
            front_detachment = torch.clamp(
                front_detachment,
                max=limits.front_contact_position_max_normalized,
            )
        front_normal_support = front_normal_support + (
            front_detachment.square() * desired_front
        ).sum(dim=-1) / desired_front.sum(dim=-1).clamp_min(1.0)

    rear_overload = torch.relu(
        rear_normal - limits.rear_normal_overload_n
    ) / limits.rear_normal_scale_n
    rear_overload_squared = rear_overload.square()
    rear_force_overload = (
        (1.0 - limits.rear_overload_worst_fraction)
        * rear_overload_squared.mean(dim=-1)
        + limits.rear_overload_worst_fraction
        * rear_overload_squared.max(dim=-1).values
    )

    rear_measured = rear_normal >= contact_force_threshold_n
    desired_rear = desired_contact[:, 2:].bool()
    both_rear = torch.all(rear_measured, dim=-1)
    both_rear_desired = torch.all(desired_rear, dim=-1)
    rear_force_imbalance = (
        (rear_normal[:, 0] - rear_normal[:, 1])
        / limits.rear_balance_scale_n
    ).square() * (both_rear & both_rear_desired).float()
    missing_desired_rear = desired_rear & ~rear_measured
    desired_rear_count = desired_rear.float().sum(dim=-1)
    rear_support_loss = (
        missing_desired_rear.float().sum(dim=-1)
        / desired_rear_count.clamp_min(1.0)
    ).square()
    undesired_rear = ~desired_rear
    undesired_rear_count = undesired_rear.float().sum(dim=-1)
    rear_swing_force = (
        (
            torch.relu(rear_normal - contact_force_threshold_n)
            / limits.rear_balance_scale_n
        ).square()
        * undesired_rear.float()
    ).sum(dim=-1) / undesired_rear_count.clamp_min(1.0)
    return {
        "front_normal_support": front_normal_support,
        "rear_force_overload": rear_force_overload,
        "rear_force_imbalance": rear_force_imbalance,
        "rear_support_loss": rear_support_loss,
        "rear_swing_force": rear_swing_force,
    }


@dataclass
class IsaacRolloutSnapshot:
    scene_state_relative: dict[str, Any]
    action_manager: dict[str, torch.Tensor]
    action_terms: dict[str, dict[str, torch.Tensor]]
    command_buffers: dict[str, torch.Tensor]
    sensor_buffers: dict[str, torch.Tensor]
    sensor_clock: dict[str, torch.Tensor]
    previous_executed_action: torch.Tensor
    previous_commanded_action: torch.Tensor
    action_delay_queue: torch.Tensor
    sim_step_counter: int
    ref_id: int
    ref_frame: int
    seed_anchor_pos_local: torch.Tensor
    seed_anchor_quat_w: torch.Tensor


class IsaacMPPIRolloutCloner:
    """Parallel, restore-before-each-batch Isaac rollout engine.

    The public environment still advances only once after the expert returns.
    Candidate simulation bypasses reward/termination/command managers and
    advances physics plus action terms directly.  All mutable buffers touched
    by this path are restored from env zero before every candidate batch and
    again before returning the selected action.
    """

    _COMMAND_BUFFER_NAMES = (
        "motion_ids",
        "_target_lateral_velocities",
        "time_steps",
        "stage",
        "stable_counter",
        "post_stable_counter",
        "hold_elapsed_counter",
        "ready_anchor_pos_w",
        "ready_anchor_quat_w",
        "body_pos_relative_w",
        "body_quat_relative_w",
    )
    _ACTION_TERM_BUFFER_NAMES = (
        "_raw_actions",
        "_processed_actions",
        "_prev_applied_actions",
    )
    _SENSOR_BUFFER_NAMES = (
        "net_forces_w",
        "net_forces_w_history",
        "force_matrix_w",
        "force_matrix_w_history",
        "last_air_time",
        "current_air_time",
        "last_contact_time",
        "current_contact_time",
        "friction_forces_w",
    )

    def __init__(
        self,
        adapter: Any,
        references: ReferenceSet,
        action_contract: ActionContract,
        horizon: int,
        cost_weights: IsaacRolloutCostWeights,
        contact_force_threshold: float = 8.0,
        load_limits: IsaacRolloutLoadLimits | None = None,
    ):
        self.adapter = adapter
        self.base = adapter.base
        self.robot = adapter.robot
        self.command = adapter.command
        self.references = references
        self.action_contract = action_contract
        self.horizon = int(horizon)
        self.cost_weights = cost_weights
        self.contact_force_threshold = float(contact_force_threshold)
        self.load_limits = load_limits or IsaacRolloutLoadLimits()
        self.device = torch.device(self.base.device)
        self.num_envs = int(self.base.num_envs)
        self.env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        if self.horizon < 1:
            raise ValueError("MPPI rollout horizon must be positive.")
        if action_contract.wheel_action_mode is not WheelActionMode.HARD_ZERO:
            raise NotImplementedError(
                "The current Isaac MPPI rollout provider is intentionally restricted to hard_zero."
            )

        self.anchor_body_id = int(self.command.robot_anchor_body_index)
        self.ref_anchor_body_id = int(self.command.motion_anchor_body_index)
        self.wheel_body_ids = list(adapter.wheel_body_ids)
        self.ref_wheel_body_ids = [
            references.body_order.index(name) for name in ("FL_foot_link", "FR_foot_link", "RL_foot_link", "RR_foot_link")
        ]
        self.joint_ids = list(adapter.joint_ids)
        self.contact_body_ids = list(adapter.contact_body_ids)

        self._joint_pos_refs = torch.as_tensor(
            np.stack([motion.joint_pos for motion in references.motions]),
            dtype=torch.float32,
            device=self.device,
        )
        self._joint_vel_refs = torch.as_tensor(
            np.stack([motion.joint_vel for motion in references.motions]),
            dtype=torch.float32,
            device=self.device,
        )
        self._body_pos_refs = torch.as_tensor(
            np.stack([motion.body_pos_w for motion in references.motions]),
            dtype=torch.float32,
            device=self.device,
        )
        self._body_quat_refs = torch.as_tensor(
            np.stack([motion.body_quat_w for motion in references.motions]),
            dtype=torch.float32,
            device=self.device,
        )
        self._body_lin_vel_refs = torch.as_tensor(
            np.stack([motion.body_lin_vel_w for motion in references.motions]),
            dtype=torch.float32,
            device=self.device,
        )
        self._desired_contacts = torch.as_tensor(
            np.stack(adapter.contact_schedules),
            dtype=torch.bool,
            device=self.device,
        )
        self.last_components: dict[str, torch.Tensor] = {}
        self.last_best_components: dict[str, float] = {}
        self.last_termination_rate = 0.0
        self._episode_ref_id: int | None = None
        self._episode_ref_frame: int | None = None
        self._episode_anchor_pos_local: torch.Tensor | None = None
        self._episode_anchor_quat_w: torch.Tensor | None = None

    def reset_episode_alignment(self) -> None:
        """Freeze the episode-global reference/world alignment.

        Re-anchoring every receding-horizon solve to the current robot pose
        would erase accumulated x/y drift from the cost.  The alignment is
        therefore captured exactly once, after the real environment reset.
        """
        self._episode_ref_id = int(self.command.motion_ids[0].item())
        self._episode_ref_frame = int(self.command.time_steps[0].item())
        self._episode_anchor_pos_local = (
            self.command.robot_anchor_pos_w[0] - self.base.scene.env_origins[0]
        ).detach().clone()
        self._episode_anchor_quat_w = self.command.robot_anchor_quat_w[0].detach().clone()

    @staticmethod
    def _copy_env0_tensor(value: torch.Tensor, num_envs: int) -> torch.Tensor:
        if value.ndim == 0 or value.shape[0] != num_envs:
            raise ValueError("Expected a tensor whose leading dimension is num_envs.")
        return value[0:1].detach().clone()

    def capture(self) -> IsaacRolloutSnapshot:
        scene_state = _tree_env0_clone(self.base.scene.get_state(is_relative=True))
        manager = self.base.action_manager
        action_manager = {
            "_action": self._copy_env0_tensor(manager._action, self.num_envs),
            "_prev_action": self._copy_env0_tensor(manager._prev_action, self.num_envs),
        }
        action_terms: dict[str, dict[str, torch.Tensor]] = {}
        for name in manager.active_terms:
            term = manager.get_term(name)
            buffers = {}
            for buffer_name in self._ACTION_TERM_BUFFER_NAMES:
                value = getattr(term, buffer_name, None)
                if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == self.num_envs:
                    buffers[buffer_name] = value[0:1].detach().clone()
            action_terms[name] = buffers

        command_buffers = {}
        for name in self._COMMAND_BUFFER_NAMES:
            value = getattr(self.command, name, None)
            if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == self.num_envs:
                command_buffers[name] = value[0:1].detach().clone()

        # Access once to make lazy contact buffers current before taking a copy.
        sensor = self.adapter.contact_sensor
        sensor_data = sensor.data
        sensor_buffers = {}
        for name in self._SENSOR_BUFFER_NAMES:
            value = getattr(sensor_data, name, None)
            if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == self.num_envs:
                sensor_buffers[name] = value[0:1].detach().clone()
        sensor_clock = {}
        for name in ("_timestamp", "_timestamp_last_update", "_is_outdated"):
            value = getattr(sensor, name, None)
            if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == self.num_envs:
                sensor_clock[name] = value[0:1].detach().clone()

        anchor_pos_local = (
            self.command.robot_anchor_pos_w[0] - self.base.scene.env_origins[0]
        ).detach().clone()
        anchor_quat = self.command.robot_anchor_quat_w[0].detach().clone()
        return IsaacRolloutSnapshot(
            scene_state_relative=scene_state,
            action_manager=action_manager,
            action_terms=action_terms,
            command_buffers=command_buffers,
            sensor_buffers=sensor_buffers,
            sensor_clock=sensor_clock,
            previous_executed_action=self.adapter.previous_executed_action[0:1].detach().clone(),
            previous_commanded_action=self.adapter.previous_commanded_action[0:1].detach().clone(),
            action_delay_queue=self.adapter.action_delay_queue[:, 0:1].detach().clone(),
            sim_step_counter=int(self.base._sim_step_counter),
            ref_id=int(self.command.motion_ids[0].item()),
            ref_frame=int(self.command.time_steps[0].item()),
            seed_anchor_pos_local=anchor_pos_local,
            seed_anchor_quat_w=anchor_quat,
        )

    def _clear_contact_warm_start(self, snapshot: IsaacRolloutSnapshot) -> None:
        """Destroy candidate-specific PhysX contact pairs before a restore.

        Tensor state writes restore poses and velocities but do not restore the
        opaque PhysX contact solver cache.  Consequently, a passive wheel can
        receive a different first-step impulse depending on which MPPI
        candidate previously occupied the same clone.  Move each dynamic asset
        to a distinct collision-free height for one physics substep, then let
        ``restore`` write the exact snapshot back.  The next candidate and the
        eventual real environment step both start with freshly-created contact
        pairs instead of candidate-history-dependent warm starts.
        """
        separated_state = _tree_repeat(snapshot.scene_state_relative, self.num_envs)
        separation_index = 1
        for category in ("articulation", "rigid_object"):
            for asset_state in separated_state.get(category, {}).values():
                root_pose = asset_state.get("root_pose")
                if isinstance(root_pose, torch.Tensor):
                    root_pose[:, 2] += 10.0 * separation_index
                    separation_index += 1
                root_velocity = asset_state.get("root_velocity")
                if isinstance(root_velocity, torch.Tensor):
                    root_velocity.zero_()
        self.base.scene.reset_to(
            separated_state,
            env_ids=self.env_ids,
            is_relative=True,
        )
        self.base.sim.step(render=False)
        self.base.scene.update(dt=self.base.physics_dt)

    def _restore_snapshot_state_and_buffers(
        self,
        snapshot: IsaacRolloutSnapshot,
        *,
        forward_after_state_write: bool = True,
    ) -> None:
        repeated_state = _tree_repeat(snapshot.scene_state_relative, self.num_envs)
        self.base.scene.reset_to(repeated_state, env_ids=self.env_ids, is_relative=True)
        if forward_after_state_write:
            self.base.sim.forward()
        # Refresh lazy articulation/body buffers after the direct tensor write.
        self.base.scene.update(dt=0.0)

        manager = self.base.action_manager
        for name, value in snapshot.action_manager.items():
            getattr(manager, name).copy_(_tree_repeat(value, self.num_envs))
        for term_name, buffers in snapshot.action_terms.items():
            term = manager.get_term(term_name)
            for name, value in buffers.items():
                getattr(term, name).copy_(_tree_repeat(value, self.num_envs))
        for name, value in snapshot.command_buffers.items():
            getattr(self.command, name).copy_(_tree_repeat(value, self.num_envs))

        sensor = self.adapter.contact_sensor
        for name, value in snapshot.sensor_buffers.items():
            target = getattr(sensor._data, name, None)
            if isinstance(target, torch.Tensor):
                target.copy_(_tree_repeat(value, self.num_envs))
        for name, value in snapshot.sensor_clock.items():
            getattr(sensor, name).copy_(_tree_repeat(value, self.num_envs))

        self.adapter.previous_executed_action.copy_(
            _tree_repeat(snapshot.previous_executed_action, self.num_envs)
        )
        self.adapter.previous_commanded_action.copy_(
            _tree_repeat(snapshot.previous_commanded_action, self.num_envs)
        )
        delay_queue = snapshot.action_delay_queue.repeat(1, self.num_envs, 1)
        self.adapter.action_delay_queue.copy_(delay_queue)
        self.base._sim_step_counter = snapshot.sim_step_counter

    def restore(
        self,
        snapshot: IsaacRolloutSnapshot,
        *,
        clear_contact_warm_start: bool = True,
        forward_after_state_write: bool = True,
        contact_prime_substeps: int = 0,
    ) -> None:
        """Restore explicit state and optionally prime deterministic contacts.

        Direct PhysX tensor writes cannot restore opaque solver impulses.  A
        cold contact pair measurably changes the front normal load on the next
        real control step.  Priming advances the already-processed snapshot
        action for a small number of physics substeps, then restores every
        explicit state and manager/sensor buffer once more without destroying
        the newly-built contact pairs.
        """

        contact_prime_substeps = int(contact_prime_substeps)
        if contact_prime_substeps < 0:
            raise ValueError("contact_prime_substeps must be non-negative.")
        if contact_prime_substeps and not clear_contact_warm_start:
            raise ValueError(
                "contact priming requires a cold contact restore."
            )
        if clear_contact_warm_start:
            self._clear_contact_warm_start(snapshot)
        self._restore_snapshot_state_and_buffers(
            snapshot,
            forward_after_state_write=forward_after_state_write,
        )
        for _ in range(contact_prime_substeps):
            self.base._sim_step_counter += 1
            self.base.action_manager.apply_action()
            self.base.scene.write_data_to_sim()
            self.base.sim.step(render=False)
            self.base.scene.update(dt=self.base.physics_dt)
        if contact_prime_substeps:
            self._restore_snapshot_state_and_buffers(
                snapshot,
                forward_after_state_write=forward_after_state_write,
            )

    def _physics_step(self, action16: torch.Tensor) -> None:
        if action16.shape != (self.num_envs, 16):
            raise ValueError(
                f"Isaac rollout action must have shape {(self.num_envs, 16)}, got {tuple(action16.shape)}"
            )
        if not torch.equal(action16[:, 12:], torch.zeros_like(action16[:, 12:])):
            raise ValueError("Isaac hard-zero MPPI received non-zero wheel action.")
        self.base.action_manager.process_action(action16)
        for _ in range(self.base.cfg.decimation):
            self.base._sim_step_counter += 1
            self.base.action_manager.apply_action()
            self.base.scene.write_data_to_sim()
            self.base.sim.step(render=False)
            self.base.scene.update(dt=self.base.physics_dt)

    def state_vector(self) -> torch.Tensor:
        root_pose = self.robot.data.root_pose_w.clone()
        root_pose[:, :3] -= self.base.scene.env_origins
        return torch.cat(
            (
                root_pose,
                self.robot.data.root_vel_w,
                self.robot.data.joint_pos[:, self.joint_ids],
                self.robot.data.joint_vel[:, self.joint_ids],
            ),
            dim=-1,
        )

    def _aligned_reference(
        self,
        snapshot: IsaacRolloutSnapshot,
        frame: int,
    ) -> dict[str, torch.Tensor]:
        ref_id = snapshot.ref_id
        if (
            self._episode_ref_id == ref_id
            and self._episode_ref_frame is not None
            and self._episode_anchor_pos_local is not None
            and self._episode_anchor_quat_w is not None
        ):
            alignment_frame = self._episode_ref_frame
            alignment_anchor_pos_local = self._episode_anchor_pos_local
            alignment_anchor_quat_w = self._episode_anchor_quat_w
        else:
            # Determinism/state-copy probes may call the cloner before an
            # expert episode has been reset.  Use a solve-local alignment only
            # for that diagnostic path.
            alignment_frame = snapshot.ref_frame
            alignment_anchor_pos_local = snapshot.seed_anchor_pos_local
            alignment_anchor_quat_w = snapshot.seed_anchor_quat_w
        current_frame = min(alignment_frame, self._joint_pos_refs.shape[1] - 1)
        frame = min(frame, self._joint_pos_refs.shape[1] - 1)
        ref_anchor_pos_now = self._body_pos_refs[ref_id, current_frame, self.ref_anchor_body_id]
        ref_anchor_quat_now = self._body_quat_refs[ref_id, current_frame, self.ref_anchor_body_id]
        alignment_quat = _quat_multiply(
            alignment_anchor_quat_w,
            _quat_conjugate(ref_anchor_quat_now),
        )

        ref_body_pos = self._body_pos_refs[ref_id, frame]
        target_body_pos_local = alignment_anchor_pos_local + _quat_rotate(
            alignment_quat.unsqueeze(0).expand(ref_body_pos.shape[0], -1),
            ref_body_pos - ref_anchor_pos_now,
        )
        ref_body_quat = self._body_quat_refs[ref_id, frame]
        target_body_quat = _quat_multiply(
            alignment_quat.unsqueeze(0).expand(ref_body_quat.shape[0], -1),
            ref_body_quat,
        )
        ref_body_lin_vel = self._body_lin_vel_refs[ref_id, frame]
        target_body_lin_vel = _quat_rotate(
            alignment_quat.unsqueeze(0).expand(ref_body_lin_vel.shape[0], -1),
            ref_body_lin_vel,
        )
        return {
            "joint_pos": self._joint_pos_refs[ref_id, frame],
            "joint_vel": self._joint_vel_refs[ref_id, frame],
            "body_pos_local": target_body_pos_local,
            "body_quat": target_body_quat,
            "body_lin_vel": target_body_lin_vel,
            "desired_contact": self._desired_contacts[ref_id, frame],
        }

    def evaluate(
        self,
        candidates_leg: torch.Tensor,
        snapshot: IsaacRolloutSnapshot,
        nominal_leg: torch.Tensor,
        *,
        action_residual_weight: float | None = None,
        rear_swing_action_residual_lead_steps: int = 0,
        base_orientation_cost_multiplier: float = 1.0,
        lateral_velocity_cost_multiplier: float = 1.0,
        rear_support_loss_cost_multiplier: float = 1.0,
        base_orientation_axis_multipliers: (
            tuple[float, float, float]
            | list[float]
            | torch.Tensor
        ) = (1.0, 1.0, 1.0),
    ) -> torch.Tensor:
        if candidates_leg.shape != (self.num_envs, self.horizon, 12):
            raise ValueError(
                "Candidate batch must match rollout clones and horizon: "
                f"expected {(self.num_envs, self.horizon, 12)}, got {tuple(candidates_leg.shape)}"
            )
        if nominal_leg.shape != (self.horizon, 12):
            raise ValueError(f"nominal_leg must have shape {(self.horizon, 12)}")
        if not torch.isfinite(candidates_leg).all():
            raise ValueError("Candidate action batch contains NaN or Inf.")

        self.restore(snapshot)
        weights = self.cost_weights
        effective_action_residual_weight = (
            weights.action_residual
            if action_residual_weight is None
            else float(action_residual_weight)
        )
        if (
            not np.isfinite(effective_action_residual_weight)
            or effective_action_residual_weight < 0.0
        ):
            raise ValueError(
                "action_residual_weight must be finite and non-negative."
            )
        if (
            isinstance(rear_swing_action_residual_lead_steps, bool)
            or not isinstance(
                rear_swing_action_residual_lead_steps,
                (int, np.integer),
            )
            or not 0
            <= int(rear_swing_action_residual_lead_steps)
            < self.horizon
        ):
            raise ValueError(
                "rear_swing_action_residual_lead_steps must be an integer "
                "in [0, horizon)."
            )
        residual_lead_steps = int(
            rear_swing_action_residual_lead_steps
        )
        effective_base_orientation_cost_multiplier = float(
            base_orientation_cost_multiplier
        )
        if (
            not np.isfinite(effective_base_orientation_cost_multiplier)
            or effective_base_orientation_cost_multiplier < 1.0
        ):
            raise ValueError(
                "base_orientation_cost_multiplier must be finite and at "
                "least 1.0."
            )
        effective_lateral_velocity_cost_multiplier = float(
            lateral_velocity_cost_multiplier
        )
        if (
            not np.isfinite(effective_lateral_velocity_cost_multiplier)
            or effective_lateral_velocity_cost_multiplier <= 0.0
        ):
            raise ValueError(
                "lateral_velocity_cost_multiplier must be finite and "
                "positive."
            )
        effective_rear_support_loss_cost_multiplier = float(
            rear_support_loss_cost_multiplier
        )
        if (
            not np.isfinite(
                effective_rear_support_loss_cost_multiplier
            )
            or effective_rear_support_loss_cost_multiplier <= 0.0
        ):
            raise ValueError(
                "rear_support_loss_cost_multiplier must be finite and "
                "positive."
            )
        effective_base_orientation_axis_multipliers = torch.as_tensor(
            base_orientation_axis_multipliers,
            dtype=torch.float32,
            device=self.device,
        )
        if (
            effective_base_orientation_axis_multipliers.shape != (3,)
            or not torch.isfinite(
                effective_base_orientation_axis_multipliers
            ).all()
            or torch.any(
                effective_base_orientation_axis_multipliers < 1.0
            )
        ):
            raise ValueError(
                "base_orientation_axis_multipliers must contain three finite "
                "values of at least 1.0."
            )
        totals = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        alive = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        previous_action = snapshot.previous_commanded_action[:, :12].expand(
            self.num_envs,
            -1,
        )
        delay_queue = snapshot.action_delay_queue.repeat(1, self.num_envs, 1)
        previous_dq = self.robot.data.joint_vel[:, self.joint_ids].clone()
        component_totals: dict[str, torch.Tensor] = {}

        def add(name: str, values: torch.Tensor, weight: float, active: torch.Tensor) -> None:
            weighted = weight * values * active
            totals.add_(weighted)
            if name not in component_totals:
                component_totals[name] = torch.zeros_like(totals)
            component_totals[name].add_(weighted)

        for horizon_step in range(self.horizon):
            leg_action = candidates_leg[:, horizon_step]
            commanded_action16 = torch.cat(
                (leg_action, torch.zeros(self.num_envs, 4, device=self.device)),
                dim=-1,
            )
            applied_action16, delay_queue = advance_action_delay(
                commanded_action16,
                delay_queue,
            )
            self._physics_step(applied_action16)
            frame = snapshot.ref_frame + horizon_step + 1
            target = self._aligned_reference(snapshot, frame)
            active = alive.float()

            actual_anchor_pos = (
                self.command.robot_anchor_pos_w - self.base.scene.env_origins
            )
            actual_anchor_quat = self.command.robot_anchor_quat_w
            actual_anchor_lin_vel = self.command.robot_anchor_lin_vel_w
            target_anchor_pos = target["body_pos_local"][self.ref_anchor_body_id]
            target_anchor_quat = target["body_quat"][self.ref_anchor_body_id]
            target_anchor_lin_vel = target["body_lin_vel"][self.ref_anchor_body_id]

            base_pos_error = actual_anchor_pos - target_anchor_pos
            base_pos_cost = (
                base_pos_error[:, 0].square()
                + base_pos_error[:, 1].square()
                + 2.0 * base_pos_error[:, 2].square()
            )
            add("base_position", base_pos_cost, weights.base_position, active)
            add(
                "base_height_drop",
                base_height_drop_cost(
                    base_pos_error,
                    self.load_limits.base_height_drop_margin_m,
                ),
                weights.base_height_drop,
                (
                    active
                    if (
                        self.load_limits.base_height_drop_stop_frame <= 0.0
                        or frame
                        < self.load_limits.base_height_drop_stop_frame
                    )
                    else torch.zeros_like(active)
                ),
            )
            add(
                "base_orientation",
                base_orientation_tracking_cost(
                    actual_anchor_quat,
                    target_anchor_quat,
                    effective_base_orientation_axis_multipliers,
                ),
                (
                    weights.base_orientation
                    * effective_base_orientation_cost_multiplier
                ),
                active,
            )

            q = self.robot.data.joint_pos[:, self.joint_ids]
            dq = self.robot.data.joint_vel[:, self.joint_ids]
            q_error = q[:, :12] - target["joint_pos"][:12]
            dq_error = dq[:, :12] - target["joint_vel"][:12]
            add("joint_position", q_error.square().mean(dim=-1), weights.joint_position, active)
            add(
                "rear_leg_position",
                rear_leg_position_cost(q_error),
                weights.rear_leg_position,
                active,
            )
            add("joint_velocity", dq_error.square().mean(dim=-1), weights.joint_velocity, active)

            wheel_pos_local = (
                self.robot.data.body_pos_w[:, self.wheel_body_ids]
                - self.base.scene.env_origins.unsqueeze(1)
            )
            wheel_lin_vel = self.robot.data.body_lin_vel_w[:, self.wheel_body_ids]
            target_wheel_pos = target["body_pos_local"][self.ref_wheel_body_ids]
            target_wheel_lin_vel = target["body_lin_vel"][self.ref_wheel_body_ids]
            wheel_error = wheel_pos_local - target_wheel_pos.unsqueeze(0)
            desired_contact = (
                target["desired_contact"].unsqueeze(0).expand(
                    self.num_envs,
                    -1,
                )
            )
            add(
                "wheel_position",
                wheel_position_tracking_cost(
                    wheel_error,
                    self.load_limits.wheel_position_worst_fraction,
                ),
                weights.wheel_position,
                active,
            )
            add(
                "rear_swing_lateral_position",
                scheduled_rear_wheel_lateral_position_cost(
                    wheel_error,
                    desired_contact,
                ),
                weights.rear_swing_lateral_position,
                (
                    active
                    if frame
                    >= self.load_limits.rear_swing_lateral_position_start_frame
                    else torch.zeros_like(active)
                ),
            )
            add(
                "rear_swing_height_deficit",
                scheduled_rear_wheel_height_deficit_cost(
                    wheel_error,
                    desired_contact,
                    self.load_limits.rear_swing_height_scale_m,
                ),
                weights.rear_swing_height_deficit,
                (
                    active
                    if frame
                    >= self.load_limits.rear_swing_height_deficit_start_frame
                    else torch.zeros_like(active)
                ),
            )
            add(
                "lateral_velocity",
                lateral_velocity_tracking_cost(
                    actual_anchor_lin_vel[:, 1]
                    - target_anchor_lin_vel[1],
                    self.load_limits.lateral_velocity_absolute_scale_m_s,
                ),
                (
                    weights.lateral_velocity
                    * effective_lateral_velocity_cost_multiplier
                ),
                active,
            )
            add(
                "lateral_position",
                base_pos_error[:, 1].square(),
                weights.lateral_position,
                (
                    active
                    if frame >= self.load_limits.lateral_position_start_frame
                    else torch.zeros_like(active)
                ),
            )
            add("box_x_drift", base_pos_error[:, 0].square(), weights.box_x_drift, active)

            contact_force = self.adapter.contact_sensor.data.net_forces_w[:, self.contact_body_ids]
            measured_contact = torch.linalg.vector_norm(contact_force, dim=-1) >= self.contact_force_threshold
            contact_mismatch = torch.logical_xor(measured_contact, desired_contact).float().mean(dim=-1)
            support_costs = load_support_cost_terms(
                contact_force,
                desired_contact,
                self.contact_force_threshold,
                self.load_limits,
                wheel_position_error_w=wheel_error,
            )
            slip_error = (wheel_lin_vel[..., :2] - target_wheel_lin_vel[None, :, :2]).square().sum(dim=-1)
            slip_cost = (slip_error * measured_contact.float()).mean(dim=-1)
            add("wheel_slip", slip_cost, weights.wheel_slip, active)
            add("contact_mismatch", contact_mismatch, weights.contact_mismatch, active)
            add(
                "front_normal_support",
                support_costs["front_normal_support"],
                weights.front_normal_support,
                active,
            )
            add(
                "rear_force_overload",
                support_costs["rear_force_overload"],
                weights.rear_force_overload,
                active,
            )
            add(
                "rear_force_imbalance",
                support_costs["rear_force_imbalance"],
                weights.rear_force_imbalance,
                active,
            )
            add(
                "rear_support_loss",
                support_costs["rear_support_loss"],
                (
                    weights.rear_support_loss
                    * effective_rear_support_loss_cost_multiplier
                ),
                active,
            )
            add(
                "rear_swing_force",
                support_costs["rear_swing_force"],
                weights.rear_swing_force,
                active,
            )

            edge_excess = torch.relu(torch.abs(wheel_error[..., 0]) - 0.08)
            drop_excess = torch.relu(target_wheel_pos[None, :, 2] - wheel_pos_local[..., 2] - 0.06)
            edge_drop_cost = edge_excess.square().mean(dim=-1) + drop_excess.square().mean(dim=-1)
            add("edge_drop", edge_drop_cost, weights.edge_drop, active)

            residual = leg_action - nominal_leg[horizon_step]
            action_rate = leg_action - previous_action
            joint_acceleration = (dq - previous_dq) / float(self.base.step_dt)
            residual_swing_active: torch.Tensor | None = None
            if residual_lead_steps:
                maximum_reference_frame = (
                    int(self._desired_contacts.shape[1]) - 1
                )
                residual_start_frame = min(
                    frame,
                    maximum_reference_frame,
                )
                residual_stop_frame = min(
                    frame + residual_lead_steps,
                    maximum_reference_frame,
                )
                future_desired_rear = self._desired_contacts[
                    snapshot.ref_id,
                    residual_start_frame : residual_stop_frame + 1,
                    2:4,
                ]
                residual_swing_active = torch.any(
                    ~future_desired_rear,
                    dim=0,
                ).unsqueeze(0).expand(self.num_envs, -1)
            add(
                "action_residual",
                scheduled_action_residual_cost(
                    residual,
                    desired_contact,
                    self.load_limits.rear_swing_action_residual_multiplier,
                    rear_swing_active=residual_swing_active,
                ),
                effective_action_residual_weight,
                active,
            )
            add("action_rate", action_rate.square().mean(dim=-1), weights.action_rate, active)
            add(
                "joint_acceleration",
                joint_acceleration[:, :12].square().mean(dim=-1),
                weights.joint_acceleration,
                active,
            )
            previous_action = leg_action
            previous_dq = dq.clone()

            joint_limits = self.robot.data.soft_joint_pos_limits[:, self.joint_ids[:12]]
            below = torch.relu(joint_limits[..., 0] - q[:, :12])
            above = torch.relu(q[:, :12] - joint_limits[..., 1])
            add("joint_limit", (below.square() + above.square()).mean(dim=-1), weights.joint_limit, active)

            applied_torque = self.robot.data.applied_torque[:, self.joint_ids[:12]]
            effort_limits = self.robot.data.joint_effort_limits[:, self.joint_ids[:12]].clamp_min(1.0e-6)
            torque_excess = torch.relu(torch.abs(applied_torque) / effort_limits - 1.0)
            add("torque_limit", torque_excess.square().mean(dim=-1), weights.torque_limit, active)

            # Match the actual lateral task termination functions.  The task
            # deliberately gates position termination on z only and measures
            # orientation through projected gravity, so lateral progress/yaw
            # error must remain a tracking cost rather than falsely killing a
            # rollout candidate.
            gravity_w = torch.zeros(
                (self.num_envs, 3),
                dtype=torch.float32,
                device=self.device,
            )
            gravity_w[:, 2] = -1.0
            target_gravity_b = _quat_rotate(
                _quat_conjugate(
                    target_anchor_quat.unsqueeze(0).expand(self.num_envs, -1)
                ),
                gravity_w,
            )
            actual_gravity_b = _quat_rotate(
                _quat_conjugate(actual_anchor_quat),
                gravity_w,
            )
            projected_gravity_z_error = torch.abs(
                target_gravity_b[:, 2] - actual_gravity_b[:, 2]
            )
            terminated = (
                (torch.abs(base_pos_error[:, 2]) > 0.25)
                | (projected_gravity_z_error > 0.80)
                | ~torch.isfinite(self.state_vector()).all(dim=-1)
            )
            newly_terminated = alive & terminated
            add(
                "termination",
                newly_terminated.float(),
                weights.termination,
                torch.ones_like(active),
            )
            alive &= ~terminated

        add(
            "terminal",
            base_orientation_tracking_cost(
                actual_anchor_quat,
                target_anchor_quat,
                effective_base_orientation_axis_multipliers,
            ),
            (
                weights.terminal_base_orientation
                * effective_base_orientation_cost_multiplier
            ),
            alive.float(),
        )
        final_tracking = (
            component_totals["base_position"]
            / max(weights.base_position, 1.0e-12)
        )
        final_tracking += (
            component_totals["joint_position"]
            / max(weights.joint_position, 1.0e-12)
        )
        add(
            "terminal",
            final_tracking / float(self.horizon),
            weights.terminal,
            alive.float(),
        )
        self.last_components = {name: values.detach().clone() for name, values in component_totals.items()}
        best_index = int(torch.argmin(totals).item())
        self.last_best_components = {
            name: float(values[best_index].item()) for name, values in component_totals.items()
        }
        self.last_termination_rate = float((~alive).float().mean().item())
        return totals

    def probe_determinism(
        self,
        candidates_leg: torch.Tensor,
        nominal_leg: torch.Tensor,
        *,
        action_residual_weight: float | None = None,
        base_orientation_cost_multiplier: float = 1.0,
        lateral_velocity_cost_multiplier: float = 1.0,
        rear_support_loss_cost_multiplier: float = 1.0,
        base_orientation_axis_multipliers: (
            tuple[float, float, float]
            | list[float]
            | torch.Tensor
        ) = (1.0, 1.0, 1.0),
    ) -> dict[str, Any]:
        snapshot = self.capture()
        try:
            first_cost = self.evaluate(
                candidates_leg,
                snapshot,
                nominal_leg,
                action_residual_weight=action_residual_weight,
                base_orientation_cost_multiplier=(
                    base_orientation_cost_multiplier
                ),
                lateral_velocity_cost_multiplier=(
                    lateral_velocity_cost_multiplier
                ),
                rear_support_loss_cost_multiplier=(
                    rear_support_loss_cost_multiplier
                ),
                base_orientation_axis_multipliers=(
                    base_orientation_axis_multipliers
                ),
            ).clone()
            first_state = self.state_vector().clone()
            second_cost = self.evaluate(
                candidates_leg,
                snapshot,
                nominal_leg,
                action_residual_weight=action_residual_weight,
                base_orientation_cost_multiplier=(
                    base_orientation_cost_multiplier
                ),
                lateral_velocity_cost_multiplier=(
                    lateral_velocity_cost_multiplier
                ),
                rear_support_loss_cost_multiplier=(
                    rear_support_loss_cost_multiplier
                ),
                base_orientation_axis_multipliers=(
                    base_orientation_axis_multipliers
                ),
            ).clone()
            second_state = self.state_vector().clone()
            cost_abs = torch.abs(first_cost - second_cost)
            state_abs = torch.abs(first_state - second_state)
            cost_max_abs = float(torch.max(cost_abs).item())
            state_max_abs = float(torch.max(state_abs).item())
            cost_max_index = int(torch.argmax(cost_abs).item())
            flat_state_max_index = int(torch.argmax(state_abs).item())
            state_max_env = flat_state_max_index // state_abs.shape[1]
            state_max_dimension = flat_state_max_index % state_abs.shape[1]
            state_slices = {
                "root_pose": (0, 7),
                "root_velocity": (7, 13),
                "joint_position": (13, 29),
                "joint_velocity": (29, 45),
            }
            state_group_max_abs = {
                name: float(state_abs[:, start:end].max().item())
                for name, (start, end) in state_slices.items()
            }
            self.restore(snapshot)
        except BaseException:
            self.restore(snapshot)
            raise
        return {
            "cost_max_abs": cost_max_abs,
            "cost_max_candidate": cost_max_index,
            "state_max_abs": state_max_abs,
            "state_max_candidate": state_max_env,
            "state_max_dimension": state_max_dimension,
            "state_group_max_abs": state_group_max_abs,
            "pass": bool(cost_max_abs <= 1.0e-5 and state_max_abs <= 1.0e-5),
        }


class IsaacWholeBodyMPPIProvider:
    """Reference-centred hard-zero MPPI label provider backed by Isaac clones."""

    def __init__(
        self,
        adapter: Any,
        references: ReferenceSet,
        action_adapter: Action16Adapter,
        config: MPPIConfig,
        noise_std: torch.Tensor,
        cost_weights: IsaacRolloutCostWeights,
        load_limits: IsaacRolloutLoadLimits | None = None,
        contact_force_threshold_n: float = 8.0,
        physical_target_rate_limit_rad_s: float | None = None,
        nominal_action_reference_q_des_by_ref: dict[
            int,
            np.ndarray | torch.Tensor,
        ]
        | None = None,
        nominal_action_reference_raw_by_ref: dict[
            int,
            np.ndarray | torch.Tensor,
        ]
        | None = None,
        nominal_action_reference_overrides_by_ref: dict[
            int,
            dict[str, Any],
        ]
        | None = None,
        nominal_joint_position_bias_leg: list[float] | tuple[float, ...] | torch.Tensor | None = None,
        nominal_joint_position_bias_start_frame: int = 0,
        nominal_joint_position_bias_ramp_frames: int = 0,
        nominal_front_force_feedback_target_n: float = 0.0,
        nominal_front_force_feedback_gain_leg: list[float] | tuple[float, ...] | torch.Tensor | None = None,
        rear_swing_reference_proposal_ref_ids: list[int] | tuple[int, ...] | None = None,
        rear_swing_reference_proposal_scales: list[float] | tuple[float, ...] | None = None,
        rear_swing_reference_proposal_joint_mask_leg: list[int] | tuple[int, ...] | torch.Tensor | None = None,
        rear_swing_reference_proposal_lead_steps: int = 0,
        rear_swing_action_residual_lead_steps: int | None = None,
        rear_swing_tracking_error_proposal_scales: list[float] | tuple[float, ...] | None = None,
        rear_swing_tracking_error_proposal_joint_mask_leg: list[int] | tuple[int, ...] | torch.Tensor | None = None,
        rear_swing_tracking_error_proposal_start_frame: int = 0,
        rear_swing_load_transfer_proposal_ref_ids: list[int] | tuple[int, ...] | None = None,
        rear_swing_load_transfer_proposal_scales: list[float] | tuple[float, ...] | None = None,
        rear_swing_load_transfer_proposal_gain_by_wheel_leg_rad: list[list[float]] | tuple[tuple[float, ...], ...] | torch.Tensor | None = None,
        rear_swing_load_transfer_proposal_start_frame: int = 0,
        rear_swing_load_transfer_proposal_start_frame_by_wheel: list[int] | tuple[int, int] | None = None,
        rear_swing_load_transfer_proposal_gate_mode: str = "swing_schedule",
        rear_swing_load_transfer_proposal_imbalance_threshold_n: float = 0.0,
        front_support_proposal_ref_ids: list[int] | tuple[int, ...] | None = None,
        front_support_proposal_scales: list[float] | tuple[float, ...] | None = None,
        front_support_proposal_gain_leg_rad: list[float] | tuple[float, ...] | torch.Tensor | None = None,
        front_support_proposal_start_frame: int = 0,
        combine_rear_swing_front_support_proposals: bool = False,
        combine_rear_swing_load_transfer_front_support_proposals: bool = False,
        combine_rear_swing_reference_load_transfer_front_support_proposals: bool = False,
        include_rear_support_reference_in_coordinated_proposals: bool = False,
        rear_support_reference_proposal_start_frame: int = 0,
        output_front_force_feedback_target_n: float = 0.0,
        output_front_force_feedback_min_contact_n: float = 0.0,
        output_front_force_feedback_lookahead_steps: int | None = None,
        output_front_force_feedback_gain_leg: list[float] | tuple[float, ...] | torch.Tensor | None = None,
        output_rear_swing_force_feedback_target_n: float = 0.0,
        output_rear_swing_force_feedback_scale_n: float = 1.0,
        output_rear_swing_force_feedback_lookahead_steps: int | None = None,
        output_rear_swing_force_feedback_start_frame: int = 0,
        output_rear_swing_force_feedback_gain_leg: list[float] | tuple[float, ...] | torch.Tensor | None = None,
        output_rear_swing_height_feedback_ref_ids: list[int] | tuple[int, ...] | None = None,
        output_rear_swing_height_feedback_gain: float = 0.0,
        output_rear_swing_height_feedback_max_abs_rad: float = 0.0,
        output_rear_swing_height_feedback_lookahead_steps: int | None = None,
        output_rear_swing_height_feedback_start_frame: int = 0,
        output_rear_support_tracking_feedback_ref_ids: list[int] | tuple[int, ...] | None = None,
        output_rear_support_tracking_feedback_gain: float = 0.0,
        output_rear_support_tracking_feedback_max_abs_rad: float = 0.0,
        output_rear_support_tracking_feedback_lookahead_steps: int | None = None,
        output_rear_support_tracking_feedback_start_frame: int = 0,
        output_pitch_feedback_ref_ids: list[int] | tuple[int, ...] | None = None,
        output_pitch_feedback_gain_leg: list[float] | tuple[float, ...] | torch.Tensor | None = None,
        output_pitch_feedback_axis: str = "y",
        output_pitch_feedback_start_frame: int = 0,
        output_pitch_feedback_max_abs_rad: float = 0.0,
        output_contact_orientation_feedback_ref_ids: list[int] | tuple[int, ...] | None = None,
        output_contact_orientation_feedback_gain_xyz: list[float] | tuple[float, ...] | torch.Tensor | None = None,
        output_contact_orientation_feedback_start_frame: int = 0,
        output_contact_orientation_feedback_max_endpoint_delta_m: float = 0.0,
        output_contact_orientation_feedback_max_abs_rad: float = 0.0,
        output_joint_position_offset_leg: list[float] | tuple[float, ...] | torch.Tensor | None = None,
    ):
        if adapter.num_envs != config.samples:
            raise ValueError(
                f"MPPI requires one Isaac clone per sample: envs={adapter.num_envs}, samples={config.samples}"
            )
        self.adapter = adapter
        self.references = references
        self.action_adapter = action_adapter
        self.config = config
        self.optimizer = ReferenceCenteredMPPI(config, noise_std, device=str(adapter.base.device))
        self.rollout = IsaacMPPIRolloutCloner(
            adapter,
            references,
            action_adapter.contract,
            config.horizon,
            cost_weights,
            contact_force_threshold=contact_force_threshold_n,
            load_limits=load_limits,
        )
        contract = action_adapter.contract
        self.raw_min = torch.as_tensor(contract.raw_min[:12], device=adapter.base.device)
        self.raw_max = torch.as_tensor(contract.raw_max[:12], device=adapter.base.device)
        self.offset = torch.as_tensor(
            contract.q_action_offset_runtime[:12], device=adapter.base.device
        )
        self.scale = torch.as_tensor(contract.scale[:12], device=adapter.base.device)
        self.nominal_action_reference_q_des_by_ref: dict[
            int,
            torch.Tensor,
        ] = {}
        for ref_id, values in (
            nominal_action_reference_q_des_by_ref or {}
        ).items():
            ref_id = int(ref_id)
            if not 0 <= ref_id < len(references):
                raise ValueError(
                    "Nominal action reference ID is outside the active "
                    f"reference bank: {ref_id}."
                )
            q_des = torch.as_tensor(
                values,
                dtype=torch.float32,
                device=adapter.base.device,
            )
            expected_shape = (references[ref_id].frames, 12)
            if tuple(q_des.shape) != expected_shape:
                raise ValueError(
                    "Nominal action reference q_des shape mismatch for "
                    f"ref {ref_id}: expected {expected_shape}, got "
                    f"{tuple(q_des.shape)}."
                )
            if not torch.isfinite(q_des).all():
                raise ValueError(
                    f"Nominal action reference for ref {ref_id} contains "
                    "NaN or Inf."
                )
            self.nominal_action_reference_q_des_by_ref[ref_id] = q_des
        self.nominal_action_reference_raw_by_ref: dict[
            int,
            torch.Tensor,
        ] = {}
        for ref_id, values in (
            nominal_action_reference_raw_by_ref or {}
        ).items():
            ref_id = int(ref_id)
            if ref_id in self.nominal_action_reference_q_des_by_ref:
                raise ValueError(
                    "A reference ID cannot have both q_des and raw nominal "
                    f"action references: ref {ref_id}."
                )
            if not 0 <= ref_id < len(references):
                raise ValueError(
                    "Nominal action reference ID is outside the active "
                    f"reference bank: {ref_id}."
                )
            raw_action = torch.as_tensor(
                values,
                dtype=torch.float32,
                device=adapter.base.device,
            )
            expected_shape = (references[ref_id].frames, 12)
            if tuple(raw_action.shape) != expected_shape:
                raise ValueError(
                    "Nominal raw action reference shape mismatch for "
                    f"ref {ref_id}: expected {expected_shape}, got "
                    f"{tuple(raw_action.shape)}."
                )
            if not torch.isfinite(raw_action).all():
                raise ValueError(
                    f"Nominal raw action reference for ref {ref_id} contains "
                    "NaN or Inf."
                )
            if bool(
                torch.any(raw_action < self.raw_min).item()
                or torch.any(raw_action > self.raw_max).item()
            ):
                raise ValueError(
                    f"Nominal raw action reference for ref {ref_id} exceeds "
                    "the frozen raw-action bounds."
                )
            self.nominal_action_reference_raw_by_ref[ref_id] = raw_action
        self.nominal_action_reference_overrides_by_ref: dict[
            int,
            dict[str, Any],
        ] = {}
        for ref_id, override in (
            nominal_action_reference_overrides_by_ref or {}
        ).items():
            ref_id = int(ref_id)
            if (
                ref_id not in self.nominal_action_reference_q_des_by_ref
                and ref_id not in self.nominal_action_reference_raw_by_ref
            ):
                raise ValueError(
                    "Nominal action solver overrides require an action "
                    f"reference for ref {ref_id}."
                )
            self.nominal_action_reference_overrides_by_ref[ref_id] = (
                normalize_nominal_solver_overrides(override)
            )
        if nominal_joint_position_bias_leg is None:
            self.nominal_joint_position_bias_leg = torch.zeros(
                12,
                dtype=torch.float32,
                device=adapter.base.device,
            )
        else:
            self.nominal_joint_position_bias_leg = torch.as_tensor(
                nominal_joint_position_bias_leg,
                dtype=torch.float32,
                device=adapter.base.device,
            )
            if self.nominal_joint_position_bias_leg.shape != (12,):
                raise ValueError(
                    "nominal_joint_position_bias_leg must contain 12 physical "
                    "joint-position offsets."
                )
        if not torch.isfinite(self.nominal_joint_position_bias_leg).all():
            raise ValueError(
                "nominal_joint_position_bias_leg contains NaN or Inf."
            )
        self.nominal_joint_position_bias_start_frame = int(
            nominal_joint_position_bias_start_frame
        )
        self.nominal_joint_position_bias_ramp_frames = int(
            nominal_joint_position_bias_ramp_frames
        )
        if self.nominal_joint_position_bias_start_frame < 0:
            raise ValueError(
                "nominal_joint_position_bias_start_frame must be non-negative."
            )
        if self.nominal_joint_position_bias_ramp_frames < 0:
            raise ValueError(
                "nominal_joint_position_bias_ramp_frames must be non-negative."
            )
        self.nominal_front_force_feedback_target_n = float(
            nominal_front_force_feedback_target_n
        )
        if (
            not np.isfinite(self.nominal_front_force_feedback_target_n)
            or self.nominal_front_force_feedback_target_n < 0.0
        ):
            raise ValueError(
                "nominal_front_force_feedback_target_n must be finite and "
                "non-negative."
            )
        if self.nominal_action_reference_raw_by_ref and (
            bool(torch.any(self.nominal_joint_position_bias_leg != 0.0).item())
            or self.nominal_front_force_feedback_target_n != 0.0
        ):
            raise ValueError(
                "Raw nominal action references require zero nominal physical "
                "bias and zero nominal front-force feedback so their stored "
                "float32 actions remain exact."
            )
        if nominal_front_force_feedback_gain_leg is None:
            self.nominal_front_force_feedback_gain_leg = torch.zeros(
                12,
                dtype=torch.float32,
                device=adapter.base.device,
            )
        else:
            self.nominal_front_force_feedback_gain_leg = torch.as_tensor(
                nominal_front_force_feedback_gain_leg,
                dtype=torch.float32,
                device=adapter.base.device,
            )
            if self.nominal_front_force_feedback_gain_leg.shape != (12,):
                raise ValueError(
                    "nominal_front_force_feedback_gain_leg must contain 12 "
                    "physical joint-position offsets."
                )
            if not torch.isfinite(
                self.nominal_front_force_feedback_gain_leg
            ).all():
                raise ValueError(
                    "nominal_front_force_feedback_gain_leg contains NaN or "
                    "Inf."
                )
        rear_joint_indices = torch.as_tensor(
            (2, 3, 6, 7, 10, 11),
            device=adapter.base.device,
        )
        if torch.any(
            self.nominal_front_force_feedback_gain_leg[
                rear_joint_indices
            ] != 0.0
        ):
            raise ValueError(
                "nominal_front_force_feedback_gain_leg must be zero for all "
                "rear-leg joints."
            )
        if (
            self.nominal_front_force_feedback_target_n == 0.0
            and torch.any(self.nominal_front_force_feedback_gain_leg != 0.0)
        ):
            raise ValueError(
                "A positive nominal_front_force_feedback_target_n is required "
                "when force-feedback gains are non-zero."
            )
        raw_proposal_ref_ids = tuple(
            rear_swing_reference_proposal_ref_ids or ()
        )
        if any(
            isinstance(ref_id, bool)
            or not isinstance(ref_id, (int, np.integer))
            for ref_id in raw_proposal_ref_ids
        ):
            raise ValueError(
                "rear_swing_reference_proposal_ref_ids must contain only "
                "integer reference IDs."
            )
        proposal_ref_ids = tuple(
            int(ref_id) for ref_id in raw_proposal_ref_ids
        )
        if len(set(proposal_ref_ids)) != len(proposal_ref_ids):
            raise ValueError(
                "rear_swing_reference_proposal_ref_ids must not contain "
                "duplicates."
            )
        if any(
            not 0 <= ref_id < len(references)
            for ref_id in proposal_ref_ids
        ):
            raise ValueError(
                "rear_swing_reference_proposal_ref_ids contains an ID "
                "outside the active reference bank."
            )
        self.rear_swing_reference_proposal_ref_ids = frozenset(
            proposal_ref_ids
        )
        try:
            proposal_scales = tuple(
                float(value)
                for value in (
                    rear_swing_reference_proposal_scales or ()
                )
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "rear_swing_reference_proposal_scales must contain finite "
                "positive values no greater than 1."
            ) from exc
        if (
            any(
                not np.isfinite(value)
                or value <= 0.0
                or value > 1.0
                for value in proposal_scales
            )
            or len(set(proposal_scales)) != len(proposal_scales)
        ):
            raise ValueError(
                "rear_swing_reference_proposal_scales must contain unique "
                "finite positive values no greater than 1."
            )
        if len(proposal_scales) >= config.samples:
            raise ValueError(
                "rear_swing_reference_proposal_scales must leave at least "
                "one MPPI sample for the stochastic population."
            )
        self.rear_swing_reference_proposal_scales = proposal_scales
        if (
            isinstance(rear_swing_reference_proposal_lead_steps, bool)
            or not isinstance(
                rear_swing_reference_proposal_lead_steps,
                (int, np.integer),
            )
            or not 0
            <= int(rear_swing_reference_proposal_lead_steps)
            < config.horizon
        ):
            raise ValueError(
                "rear_swing_reference_proposal_lead_steps must be an "
                "integer in [0, horizon)."
            )
        self.rear_swing_reference_proposal_lead_steps = int(
            rear_swing_reference_proposal_lead_steps
        )
        if rear_swing_action_residual_lead_steps is None:
            residual_lead_steps = (
                self.rear_swing_reference_proposal_lead_steps
            )
        else:
            if (
                isinstance(rear_swing_action_residual_lead_steps, bool)
                or not isinstance(
                    rear_swing_action_residual_lead_steps,
                    (int, np.integer),
                )
                or not 0
                <= int(rear_swing_action_residual_lead_steps)
                < config.horizon
            ):
                raise ValueError(
                    "rear_swing_action_residual_lead_steps must be an "
                    "integer in [0, horizon) or None."
                )
            residual_lead_steps = int(
                rear_swing_action_residual_lead_steps
            )
        self.rear_swing_action_residual_lead_steps = (
            residual_lead_steps
        )
        if rear_swing_reference_proposal_joint_mask_leg is None:
            proposal_joint_mask = torch.zeros(
                12,
                dtype=torch.float32,
                device=adapter.base.device,
            )
        else:
            proposal_joint_mask = torch.as_tensor(
                rear_swing_reference_proposal_joint_mask_leg,
                dtype=torch.float32,
                device=adapter.base.device,
            )
            if proposal_joint_mask.shape != (12,):
                raise ValueError(
                    "rear_swing_reference_proposal_joint_mask_leg must "
                    "contain 12 binary values."
                )
            if (
                not torch.isfinite(proposal_joint_mask).all()
                or torch.any(
                    (proposal_joint_mask != 0.0)
                    & (proposal_joint_mask != 1.0)
                )
            ):
                raise ValueError(
                    "rear_swing_reference_proposal_joint_mask_leg must "
                    "contain only zeros and ones."
                )
        front_joint_indices = torch.as_tensor(
            (0, 1, 4, 5, 8, 9),
            device=adapter.base.device,
        )
        if torch.any(
            proposal_joint_mask[front_joint_indices] != 0.0
        ):
            raise ValueError(
                "rear_swing_reference_proposal_joint_mask_leg must be zero "
                "for all front-leg joints."
            )
        proposal_configured = bool(
            self.rear_swing_reference_proposal_ref_ids
            or self.rear_swing_reference_proposal_scales
            or torch.any(proposal_joint_mask != 0.0).item()
            or self.rear_swing_reference_proposal_lead_steps
        )
        proposal_complete = bool(
            self.rear_swing_reference_proposal_ref_ids
            and self.rear_swing_reference_proposal_scales
            and torch.any(proposal_joint_mask != 0.0).item()
        )
        if proposal_configured and not proposal_complete:
            raise ValueError(
                "Rear-swing reference proposals require non-empty ref IDs, "
                "scales, and at least one enabled rear joint."
            )
        self.rear_swing_reference_proposal_joint_mask_leg = (
            proposal_joint_mask
        )
        try:
            tracking_error_scales = tuple(
                float(value)
                for value in (
                    rear_swing_tracking_error_proposal_scales or ()
                )
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "rear_swing_tracking_error_proposal_scales must contain "
                "finite positive values no greater than 1."
            ) from exc
        if (
            any(
                not np.isfinite(value)
                or value <= 0.0
                or value > 1.0
                for value in tracking_error_scales
            )
            or len(set(tracking_error_scales))
            != len(tracking_error_scales)
        ):
            raise ValueError(
                "rear_swing_tracking_error_proposal_scales must contain "
                "unique finite positive values no greater than 1."
            )
        if tracking_error_scales and not proposal_complete:
            raise ValueError(
                "Rear-swing tracking-error proposals require complete "
                "rear-swing reference proposal settings."
            )
        if (
            isinstance(
                rear_swing_tracking_error_proposal_start_frame,
                bool,
            )
            or not isinstance(
                rear_swing_tracking_error_proposal_start_frame,
                (int, np.integer),
            )
            or int(
                rear_swing_tracking_error_proposal_start_frame
            )
            < 0
        ):
            raise ValueError(
                "rear_swing_tracking_error_proposal_start_frame must be a "
                "non-negative integer."
            )
        if (
            int(rear_swing_tracking_error_proposal_start_frame) != 0
            and not tracking_error_scales
        ):
            raise ValueError(
                "A delayed rear-swing tracking-error proposal requires "
                "non-empty tracking-error proposal scales."
            )
        self.rear_swing_tracking_error_proposal_scales = (
            tracking_error_scales
        )
        if rear_swing_tracking_error_proposal_joint_mask_leg is None:
            tracking_error_joint_mask = proposal_joint_mask.clone()
        else:
            tracking_error_joint_mask = torch.as_tensor(
                rear_swing_tracking_error_proposal_joint_mask_leg,
                dtype=torch.float32,
                device=adapter.base.device,
            )
            if tracking_error_joint_mask.shape != (12,):
                raise ValueError(
                    "rear_swing_tracking_error_proposal_joint_mask_leg must "
                    "contain 12 binary values."
                )
            if (
                not torch.isfinite(tracking_error_joint_mask).all()
                or torch.any(
                    (tracking_error_joint_mask != 0.0)
                    & (tracking_error_joint_mask != 1.0)
                )
            ):
                raise ValueError(
                    "rear_swing_tracking_error_proposal_joint_mask_leg must "
                    "contain only zeros and ones."
                )
            if torch.any(
                tracking_error_joint_mask[front_joint_indices] != 0.0
            ):
                raise ValueError(
                    "rear_swing_tracking_error_proposal_joint_mask_leg must "
                    "be zero for all front-leg joints."
                )
        if tracking_error_scales and not torch.any(
            tracking_error_joint_mask != 0.0
        ):
            raise ValueError(
                "Rear-swing tracking-error proposals require at least one "
                "enabled rear joint."
            )
        if (
            not tracking_error_scales
            and rear_swing_tracking_error_proposal_joint_mask_leg is not None
        ):
            raise ValueError(
                "A rear-swing tracking-error joint mask requires non-empty "
                "tracking-error proposal scales."
            )
        self.rear_swing_tracking_error_proposal_joint_mask_leg = (
            tracking_error_joint_mask
        )
        self.rear_swing_tracking_error_proposal_start_frame = int(
            rear_swing_tracking_error_proposal_start_frame
        )
        raw_load_transfer_ref_ids = tuple(
            rear_swing_load_transfer_proposal_ref_ids or ()
        )
        if any(
            isinstance(ref_id, bool)
            or not isinstance(ref_id, (int, np.integer))
            for ref_id in raw_load_transfer_ref_ids
        ):
            raise ValueError(
                "rear_swing_load_transfer_proposal_ref_ids must contain "
                "only integer reference IDs."
            )
        load_transfer_ref_ids = tuple(
            int(ref_id) for ref_id in raw_load_transfer_ref_ids
        )
        if len(set(load_transfer_ref_ids)) != len(
            load_transfer_ref_ids
        ):
            raise ValueError(
                "rear_swing_load_transfer_proposal_ref_ids must not contain "
                "duplicates."
            )
        if any(
            not 0 <= ref_id < len(references)
            for ref_id in load_transfer_ref_ids
        ):
            raise ValueError(
                "rear_swing_load_transfer_proposal_ref_ids contains an ID "
                "outside the active reference bank."
            )
        try:
            load_transfer_scales = tuple(
                float(value)
                for value in (
                    rear_swing_load_transfer_proposal_scales or ()
                )
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "rear_swing_load_transfer_proposal_scales must contain "
                "finite positive values no greater than 1."
            ) from exc
        if (
            any(
                not np.isfinite(value)
                or value <= 0.0
                or value > 1.0
                for value in load_transfer_scales
            )
            or len(set(load_transfer_scales))
            != len(load_transfer_scales)
        ):
            raise ValueError(
                "rear_swing_load_transfer_proposal_scales must contain "
                "unique finite positive values no greater than 1."
            )
        if rear_swing_load_transfer_proposal_gain_by_wheel_leg_rad is None:
            load_transfer_gain = torch.zeros(
                (2, 12),
                dtype=torch.float32,
                device=adapter.base.device,
            )
        else:
            load_transfer_gain = torch.as_tensor(
                rear_swing_load_transfer_proposal_gain_by_wheel_leg_rad,
                dtype=torch.float32,
                device=adapter.base.device,
            )
            if load_transfer_gain.shape != (2, 12):
                raise ValueError(
                    "rear_swing_load_transfer_proposal_gain_by_wheel_leg_rad "
                    "must contain two rows of 12 physical joint-position "
                    "offsets ordered by RL then RR swing."
                )
            if not torch.isfinite(load_transfer_gain).all():
                raise ValueError(
                    "rear_swing_load_transfer_proposal_gain_by_wheel_leg_rad "
                    "contains NaN or Inf."
                )
        if (
            isinstance(
                rear_swing_load_transfer_proposal_start_frame,
                bool,
            )
            or not isinstance(
                rear_swing_load_transfer_proposal_start_frame,
                (int, np.integer),
            )
            or int(
                rear_swing_load_transfer_proposal_start_frame
            )
            < 0
        ):
            raise ValueError(
                "rear_swing_load_transfer_proposal_start_frame must be a "
                "non-negative integer."
            )
        if (
            rear_swing_load_transfer_proposal_start_frame_by_wheel
            is None
        ):
            load_transfer_start_frames_by_wheel = (
                int(rear_swing_load_transfer_proposal_start_frame),
                int(rear_swing_load_transfer_proposal_start_frame),
            )
        else:
            raw_start_frames_by_wheel = (
                rear_swing_load_transfer_proposal_start_frame_by_wheel
            )
            if (
                not isinstance(
                    raw_start_frames_by_wheel,
                    (list, tuple),
                )
                or len(raw_start_frames_by_wheel) != 2
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, np.integer))
                    or int(value) < 0
                    for value in raw_start_frames_by_wheel
                )
            ):
                raise ValueError(
                    "rear_swing_load_transfer_proposal_start_frame_by_wheel "
                    "must contain exactly two non-negative integers ordered "
                    "by RL then RR swing."
                )
            load_transfer_start_frames_by_wheel = tuple(
                int(value)
                for value in raw_start_frames_by_wheel
            )
        load_transfer_gate_mode = str(
            rear_swing_load_transfer_proposal_gate_mode
        )
        if load_transfer_gate_mode not in (
            "swing_schedule",
            "rear_force_imbalance",
        ):
            raise ValueError(
                "rear_swing_load_transfer_proposal_gate_mode must be "
                "'swing_schedule' or 'rear_force_imbalance'."
            )
        load_transfer_imbalance_threshold_n = float(
            rear_swing_load_transfer_proposal_imbalance_threshold_n
        )
        if (
            not np.isfinite(load_transfer_imbalance_threshold_n)
            or load_transfer_imbalance_threshold_n < 0.0
        ):
            raise ValueError(
                "rear_swing_load_transfer_proposal_imbalance_threshold_n "
                "must be finite and non-negative."
            )
        if (
            load_transfer_gate_mode == "rear_force_imbalance"
            and load_transfer_imbalance_threshold_n <= 0.0
        ):
            raise ValueError(
                "rear_force_imbalance load-transfer gating requires a "
                "positive imbalance threshold."
            )
        if (
            load_transfer_gate_mode == "swing_schedule"
            and load_transfer_imbalance_threshold_n != 0.0
        ):
            raise ValueError(
                "A load-transfer imbalance threshold requires "
                "rear_force_imbalance gate mode."
            )
        load_transfer_configured = bool(
            load_transfer_ref_ids
            or load_transfer_scales
            or torch.any(load_transfer_gain != 0.0).item()
            or int(rear_swing_load_transfer_proposal_start_frame) != 0
            or any(
                start_frame != 0
                for start_frame
                in load_transfer_start_frames_by_wheel
            )
            or load_transfer_gate_mode != "swing_schedule"
            or load_transfer_imbalance_threshold_n != 0.0
        )
        load_transfer_complete = bool(
            load_transfer_ref_ids
            and load_transfer_scales
            and torch.any(load_transfer_gain != 0.0).item()
        )
        if load_transfer_configured and not load_transfer_complete:
            raise ValueError(
                "Rear-swing load-transfer proposals require non-empty ref "
                "IDs, scales, and at least one non-zero wheel-specific "
                "leg gain."
            )
        self.rear_swing_load_transfer_proposal_ref_ids = frozenset(
            load_transfer_ref_ids
        )
        self.rear_swing_load_transfer_proposal_scales = (
            load_transfer_scales
        )
        self.rear_swing_load_transfer_proposal_gain_by_wheel_leg_rad = (
            load_transfer_gain
        )
        self.rear_swing_load_transfer_proposal_start_frame = int(
            rear_swing_load_transfer_proposal_start_frame
        )
        self.rear_swing_load_transfer_proposal_start_frame_by_wheel = (
            load_transfer_start_frames_by_wheel
        )
        self.rear_swing_load_transfer_proposal_gate_mode = (
            load_transfer_gate_mode
        )
        self.rear_swing_load_transfer_proposal_imbalance_threshold_n = (
            load_transfer_imbalance_threshold_n
        )
        raw_front_support_ref_ids = tuple(
            front_support_proposal_ref_ids or ()
        )
        if any(
            isinstance(ref_id, bool)
            or not isinstance(ref_id, (int, np.integer))
            for ref_id in raw_front_support_ref_ids
        ):
            raise ValueError(
                "front_support_proposal_ref_ids must contain only integer "
                "reference IDs."
            )
        front_support_ref_ids = tuple(
            int(ref_id) for ref_id in raw_front_support_ref_ids
        )
        if len(set(front_support_ref_ids)) != len(front_support_ref_ids):
            raise ValueError(
                "front_support_proposal_ref_ids must not contain duplicates."
            )
        if any(
            not 0 <= ref_id < len(references)
            for ref_id in front_support_ref_ids
        ):
            raise ValueError(
                "front_support_proposal_ref_ids contains an ID outside the "
                "active reference bank."
            )
        self.front_support_proposal_ref_ids = frozenset(
            front_support_ref_ids
        )
        try:
            front_support_scales = tuple(
                float(value)
                for value in (front_support_proposal_scales or ())
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "front_support_proposal_scales must contain finite positive "
                "values no greater than 1."
            ) from exc
        if (
            any(
                not np.isfinite(value)
                or value <= 0.0
                or value > 1.0
                for value in front_support_scales
            )
            or len(set(front_support_scales))
            != len(front_support_scales)
        ):
            raise ValueError(
                "front_support_proposal_scales must contain unique finite "
                "positive values no greater than 1."
            )
        self.front_support_proposal_scales = front_support_scales
        if front_support_proposal_gain_leg_rad is None:
            front_support_gain = torch.zeros(
                12,
                dtype=torch.float32,
                device=adapter.base.device,
            )
        else:
            front_support_gain = torch.as_tensor(
                front_support_proposal_gain_leg_rad,
                dtype=torch.float32,
                device=adapter.base.device,
            )
            if front_support_gain.shape != (12,):
                raise ValueError(
                    "front_support_proposal_gain_leg_rad must contain 12 "
                    "physical joint-position offsets."
                )
            if not torch.isfinite(front_support_gain).all():
                raise ValueError(
                    "front_support_proposal_gain_leg_rad contains NaN or "
                    "Inf."
                )
        if torch.any(front_support_gain[rear_joint_indices] != 0.0):
            raise ValueError(
                "front_support_proposal_gain_leg_rad must be zero for all "
                "rear-leg joints."
            )
        front_support_configured = bool(
            self.front_support_proposal_ref_ids
            or self.front_support_proposal_scales
            or torch.any(front_support_gain != 0.0).item()
        )
        front_support_complete = bool(
            self.front_support_proposal_ref_ids
            and self.front_support_proposal_scales
            and torch.any(front_support_gain != 0.0).item()
        )
        if front_support_configured and not front_support_complete:
            raise ValueError(
                "Front-support proposals require non-empty ref IDs, scales, "
                "and at least one non-zero front-leg gain."
            )
        if (
            isinstance(front_support_proposal_start_frame, bool)
            or not isinstance(
                front_support_proposal_start_frame,
                (int, np.integer),
            )
            or int(front_support_proposal_start_frame) < 0
        ):
            raise ValueError(
                "front_support_proposal_start_frame must be a non-negative "
                "integer."
            )
        self.front_support_proposal_start_frame = int(
            front_support_proposal_start_frame
        )
        if (
            self.front_support_proposal_start_frame != 0
            and not front_support_complete
        ):
            raise ValueError(
                "Front-support proposals require non-empty ref IDs, scales, "
                "and at least one non-zero front-leg gain."
            )
        if not isinstance(
            combine_rear_swing_front_support_proposals,
            bool,
        ):
            raise ValueError(
                "combine_rear_swing_front_support_proposals must be a "
                "boolean."
            )
        if (
            combine_rear_swing_front_support_proposals
            and not (proposal_complete and front_support_complete)
        ):
            raise ValueError(
                "Combined rear-swing/front-support proposals require "
                "complete rear-swing and front-support proposal settings."
            )
        if not isinstance(
            combine_rear_swing_load_transfer_front_support_proposals,
            bool,
        ):
            raise ValueError(
                "combine_rear_swing_load_transfer_front_support_proposals "
                "must be a boolean."
            )
        if (
            combine_rear_swing_load_transfer_front_support_proposals
            and not (load_transfer_complete and front_support_complete)
        ):
            raise ValueError(
                "Combined rear-swing load-transfer/front-support proposals "
                "require complete load-transfer and front-support proposal "
                "settings."
            )
        if (
            combine_rear_swing_load_transfer_front_support_proposals
            and load_transfer_gate_mode != "swing_schedule"
        ):
            raise ValueError(
                "Combined rear-swing load-transfer/front-support proposals "
                "require swing_schedule load-transfer gating."
            )
        if not isinstance(
            combine_rear_swing_reference_load_transfer_front_support_proposals,
            bool,
        ):
            raise ValueError(
                "combine_rear_swing_reference_load_transfer_front_support_"
                "proposals must be a boolean."
            )
        if (
            combine_rear_swing_reference_load_transfer_front_support_proposals
            and not (
                proposal_complete
                and load_transfer_complete
                and front_support_complete
            )
        ):
            raise ValueError(
                "Combined rear-swing reference/load-transfer/front-support "
                "proposals require complete rear-swing, load-transfer, and "
                "front-support proposal settings."
            )
        if (
            combine_rear_swing_reference_load_transfer_front_support_proposals
            and load_transfer_gate_mode != "swing_schedule"
        ):
            raise ValueError(
                "Combined rear-swing reference/load-transfer/front-support "
                "proposals require swing_schedule load-transfer gating."
            )
        if not isinstance(
            include_rear_support_reference_in_coordinated_proposals,
            bool,
        ):
            raise ValueError(
                "include_rear_support_reference_in_coordinated_proposals "
                "must be a boolean."
            )
        if (
            include_rear_support_reference_in_coordinated_proposals
            and not combine_rear_swing_front_support_proposals
        ):
            raise ValueError(
                "Rear-support reference coordination requires combined "
                "rear-swing/front-support proposals."
            )
        if (
            isinstance(rear_support_reference_proposal_start_frame, bool)
            or not isinstance(
                rear_support_reference_proposal_start_frame,
                (int, np.integer),
            )
            or int(rear_support_reference_proposal_start_frame) < 0
        ):
            raise ValueError(
                "rear_support_reference_proposal_start_frame must be a "
                "non-negative integer."
            )
        if (
            int(rear_support_reference_proposal_start_frame) != 0
            and not include_rear_support_reference_in_coordinated_proposals
        ):
            raise ValueError(
                "A delayed rear-support reference proposal requires "
                "rear-support reference coordination."
            )
        coordinated_proposal_count = (
            len(self.rear_swing_reference_proposal_scales)
            * len(self.front_support_proposal_scales)
            if combine_rear_swing_front_support_proposals
            else 0
        )
        coordinated_load_transfer_proposal_count = (
            len(self.rear_swing_load_transfer_proposal_scales)
            * len(self.front_support_proposal_scales)
            if combine_rear_swing_load_transfer_front_support_proposals
            else 0
        )
        coordinated_reference_load_transfer_proposal_count = (
            len(self.rear_swing_reference_proposal_scales)
            * len(self.rear_swing_load_transfer_proposal_scales)
            * len(self.front_support_proposal_scales)
            if (
                combine_rear_swing_reference_load_transfer_front_support_proposals
            )
            else 0
        )
        rear_support_variant_count = (
            len(self.rear_swing_reference_proposal_scales)
            if include_rear_support_reference_in_coordinated_proposals
            else 0
        )
        if (
            len(self.rear_swing_reference_proposal_scales)
            + len(self.rear_swing_tracking_error_proposal_scales)
            + len(self.rear_swing_load_transfer_proposal_scales)
            + len(self.front_support_proposal_scales)
            + coordinated_proposal_count
            + coordinated_load_transfer_proposal_count
            + coordinated_reference_load_transfer_proposal_count
            + rear_support_variant_count
            >= config.samples
        ):
            raise ValueError(
                "Structured proposals must leave at least one MPPI sample "
                "for the stochastic population."
            )
        self.front_support_proposal_gain_leg_rad = front_support_gain
        self.combine_rear_swing_front_support_proposals = (
            combine_rear_swing_front_support_proposals
        )
        self.combine_rear_swing_load_transfer_front_support_proposals = (
            combine_rear_swing_load_transfer_front_support_proposals
        )
        self.combine_rear_swing_reference_load_transfer_front_support_proposals = (
            combine_rear_swing_reference_load_transfer_front_support_proposals
        )
        self.include_rear_support_reference_in_coordinated_proposals = (
            include_rear_support_reference_in_coordinated_proposals
        )
        self.rear_support_reference_proposal_start_frame = int(
            rear_support_reference_proposal_start_frame
        )
        self.output_front_force_feedback_target_n = float(
            output_front_force_feedback_target_n
        )
        if (
            not np.isfinite(self.output_front_force_feedback_target_n)
            or self.output_front_force_feedback_target_n < 0.0
        ):
            raise ValueError(
                "output_front_force_feedback_target_n must be finite and "
                "non-negative."
            )
        self.output_front_force_feedback_min_contact_n = float(
            output_front_force_feedback_min_contact_n
        )
        if (
            not np.isfinite(
                self.output_front_force_feedback_min_contact_n
            )
            or self.output_front_force_feedback_min_contact_n < 0.0
        ):
            raise ValueError(
                "output_front_force_feedback_min_contact_n must be finite "
                "and non-negative."
            )
        feedback_lookahead = (
            config.reference_action_lookahead_steps
            if output_front_force_feedback_lookahead_steps is None
            else output_front_force_feedback_lookahead_steps
        )
        if (
            isinstance(feedback_lookahead, bool)
            or not isinstance(feedback_lookahead, (int, np.integer))
            or feedback_lookahead < 0
        ):
            raise ValueError(
                "output_front_force_feedback_lookahead_steps must be a "
                "non-negative integer."
            )
        self.output_front_force_feedback_lookahead_steps = int(
            feedback_lookahead
        )
        if output_front_force_feedback_gain_leg is None:
            self.output_front_force_feedback_gain_leg = torch.zeros(
                12,
                dtype=torch.float32,
                device=adapter.base.device,
            )
        else:
            self.output_front_force_feedback_gain_leg = torch.as_tensor(
                output_front_force_feedback_gain_leg,
                dtype=torch.float32,
                device=adapter.base.device,
            )
            if self.output_front_force_feedback_gain_leg.shape != (12,):
                raise ValueError(
                    "output_front_force_feedback_gain_leg must contain 12 "
                    "physical joint-position offsets."
                )
            if not torch.isfinite(
                self.output_front_force_feedback_gain_leg
            ).all():
                raise ValueError(
                    "output_front_force_feedback_gain_leg contains NaN or "
                    "Inf."
                )
        if torch.any(
            self.output_front_force_feedback_gain_leg[
                rear_joint_indices
            ] != 0.0
        ):
            raise ValueError(
                "output_front_force_feedback_gain_leg must be zero for all "
                "rear-leg joints."
            )
        if (
            self.output_front_force_feedback_target_n == 0.0
            and torch.any(self.output_front_force_feedback_gain_leg != 0.0)
        ):
            raise ValueError(
                "A positive output_front_force_feedback_target_n is required "
                "when output force-feedback gains are non-zero."
            )
        self.output_rear_swing_force_feedback_target_n = float(
            output_rear_swing_force_feedback_target_n
        )
        if (
            not np.isfinite(
                self.output_rear_swing_force_feedback_target_n
            )
            or self.output_rear_swing_force_feedback_target_n < 0.0
        ):
            raise ValueError(
                "output_rear_swing_force_feedback_target_n must be finite "
                "and non-negative."
            )
        self.output_rear_swing_force_feedback_scale_n = float(
            output_rear_swing_force_feedback_scale_n
        )
        if (
            not np.isfinite(
                self.output_rear_swing_force_feedback_scale_n
            )
            or self.output_rear_swing_force_feedback_scale_n <= 0.0
        ):
            raise ValueError(
                "output_rear_swing_force_feedback_scale_n must be finite "
                "and positive."
            )
        rear_feedback_lookahead = (
            config.reference_action_lookahead_steps
            if output_rear_swing_force_feedback_lookahead_steps is None
            else output_rear_swing_force_feedback_lookahead_steps
        )
        if (
            isinstance(rear_feedback_lookahead, bool)
            or not isinstance(
                rear_feedback_lookahead,
                (int, np.integer),
            )
            or rear_feedback_lookahead < 0
        ):
            raise ValueError(
                "output_rear_swing_force_feedback_lookahead_steps must be "
                "a non-negative integer."
            )
        self.output_rear_swing_force_feedback_lookahead_steps = int(
            rear_feedback_lookahead
        )
        if (
            isinstance(output_rear_swing_force_feedback_start_frame, bool)
            or not isinstance(
                output_rear_swing_force_feedback_start_frame,
                (int, np.integer),
            )
            or output_rear_swing_force_feedback_start_frame < 0
        ):
            raise ValueError(
                "output_rear_swing_force_feedback_start_frame must be a "
                "non-negative integer."
            )
        self.output_rear_swing_force_feedback_start_frame = int(
            output_rear_swing_force_feedback_start_frame
        )
        if output_rear_swing_force_feedback_gain_leg is None:
            self.output_rear_swing_force_feedback_gain_leg = torch.zeros(
                12,
                dtype=torch.float32,
                device=adapter.base.device,
            )
        else:
            self.output_rear_swing_force_feedback_gain_leg = torch.as_tensor(
                output_rear_swing_force_feedback_gain_leg,
                dtype=torch.float32,
                device=adapter.base.device,
            )
            if (
                self.output_rear_swing_force_feedback_gain_leg.shape
                != (12,)
            ):
                raise ValueError(
                    "output_rear_swing_force_feedback_gain_leg must contain "
                    "12 physical joint-position offsets."
                )
            if not torch.isfinite(
                self.output_rear_swing_force_feedback_gain_leg
            ).all():
                raise ValueError(
                    "output_rear_swing_force_feedback_gain_leg contains NaN "
                    "or Inf."
                )
        if torch.any(
            self.output_rear_swing_force_feedback_gain_leg[
                front_joint_indices
            ] != 0.0
        ):
            raise ValueError(
                "output_rear_swing_force_feedback_gain_leg must be zero for "
                "all front-leg joints."
            )
        if (
            self.output_rear_swing_force_feedback_target_n == 0.0
            and torch.any(
                self.output_rear_swing_force_feedback_gain_leg != 0.0
            )
        ):
            raise ValueError(
                "A positive output_rear_swing_force_feedback_target_n is "
                "required when rear swing force-feedback gains are non-zero."
            )
        raw_rear_swing_height_feedback_ref_ids = tuple(
            output_rear_swing_height_feedback_ref_ids or ()
        )
        if any(
            isinstance(ref_id, bool)
            or not isinstance(ref_id, (int, np.integer))
            for ref_id in raw_rear_swing_height_feedback_ref_ids
        ):
            raise ValueError(
                "output_rear_swing_height_feedback_ref_ids must contain "
                "only integer reference IDs."
            )
        rear_swing_height_feedback_ref_ids = tuple(
            int(ref_id)
            for ref_id in raw_rear_swing_height_feedback_ref_ids
        )
        if len(set(rear_swing_height_feedback_ref_ids)) != len(
            rear_swing_height_feedback_ref_ids
        ):
            raise ValueError(
                "output_rear_swing_height_feedback_ref_ids must not "
                "contain duplicates."
            )
        if any(
            not 0 <= ref_id < len(references)
            for ref_id in rear_swing_height_feedback_ref_ids
        ):
            raise ValueError(
                "output_rear_swing_height_feedback_ref_ids contains an ID "
                "outside the active reference bank."
            )
        self.output_rear_swing_height_feedback_ref_ids = frozenset(
            rear_swing_height_feedback_ref_ids
        )
        self.output_rear_swing_height_feedback_gain = float(
            output_rear_swing_height_feedback_gain
        )
        if (
            not np.isfinite(
                self.output_rear_swing_height_feedback_gain
            )
            or not 0.0
            <= self.output_rear_swing_height_feedback_gain
            <= 1.0
        ):
            raise ValueError(
                "output_rear_swing_height_feedback_gain must be finite and "
                "lie in [0,1]."
            )
        self.output_rear_swing_height_feedback_max_abs_rad = float(
            output_rear_swing_height_feedback_max_abs_rad
        )
        if (
            not np.isfinite(
                self.output_rear_swing_height_feedback_max_abs_rad
            )
            or not 0.0
            <= self.output_rear_swing_height_feedback_max_abs_rad
            <= 0.12
        ):
            raise ValueError(
                "output_rear_swing_height_feedback_max_abs_rad must be "
                "finite and lie in [0,0.12]."
            )
        rear_swing_height_feedback_lookahead = (
            config.reference_action_lookahead_steps
            if output_rear_swing_height_feedback_lookahead_steps is None
            else output_rear_swing_height_feedback_lookahead_steps
        )
        if (
            isinstance(rear_swing_height_feedback_lookahead, bool)
            or not isinstance(
                rear_swing_height_feedback_lookahead,
                (int, np.integer),
            )
            or rear_swing_height_feedback_lookahead < 0
        ):
            raise ValueError(
                "output_rear_swing_height_feedback_lookahead_steps must be "
                "a non-negative integer."
            )
        self.output_rear_swing_height_feedback_lookahead_steps = int(
            rear_swing_height_feedback_lookahead
        )
        if (
            isinstance(output_rear_swing_height_feedback_start_frame, bool)
            or not isinstance(
                output_rear_swing_height_feedback_start_frame,
                (int, np.integer),
            )
            or output_rear_swing_height_feedback_start_frame < 0
        ):
            raise ValueError(
                "output_rear_swing_height_feedback_start_frame must be a "
                "non-negative integer."
            )
        self.output_rear_swing_height_feedback_start_frame = int(
            output_rear_swing_height_feedback_start_frame
        )
        rear_swing_height_feedback_configured = (
            self.output_rear_swing_height_feedback_gain > 0.0
            or self.output_rear_swing_height_feedback_max_abs_rad > 0.0
            or bool(self.output_rear_swing_height_feedback_ref_ids)
        )
        if rear_swing_height_feedback_configured and (
            self.output_rear_swing_height_feedback_gain == 0.0
            or self.output_rear_swing_height_feedback_max_abs_rad == 0.0
            or not self.output_rear_swing_height_feedback_ref_ids
        ):
            raise ValueError(
                "Rear-swing height feedback requires non-empty reference "
                "IDs, a positive gain, and a positive maximum correction."
            )
        raw_rear_support_feedback_ref_ids = tuple(
            output_rear_support_tracking_feedback_ref_ids or ()
        )
        if any(
            isinstance(ref_id, bool)
            or not isinstance(ref_id, (int, np.integer))
            for ref_id in raw_rear_support_feedback_ref_ids
        ):
            raise ValueError(
                "output_rear_support_tracking_feedback_ref_ids must contain "
                "only integer reference IDs."
            )
        rear_support_feedback_ref_ids = tuple(
            int(ref_id) for ref_id in raw_rear_support_feedback_ref_ids
        )
        if len(set(rear_support_feedback_ref_ids)) != len(
            rear_support_feedback_ref_ids
        ):
            raise ValueError(
                "output_rear_support_tracking_feedback_ref_ids must not "
                "contain duplicates."
            )
        if any(
            not 0 <= ref_id < len(references)
            for ref_id in rear_support_feedback_ref_ids
        ):
            raise ValueError(
                "output_rear_support_tracking_feedback_ref_ids contains an "
                "ID outside the active reference bank."
            )
        self.output_rear_support_tracking_feedback_ref_ids = frozenset(
            rear_support_feedback_ref_ids
        )
        self.output_rear_support_tracking_feedback_gain = float(
            output_rear_support_tracking_feedback_gain
        )
        if (
            not np.isfinite(
                self.output_rear_support_tracking_feedback_gain
            )
            or self.output_rear_support_tracking_feedback_gain < 0.0
        ):
            raise ValueError(
                "output_rear_support_tracking_feedback_gain must be finite "
                "and non-negative."
            )
        self.output_rear_support_tracking_feedback_max_abs_rad = float(
            output_rear_support_tracking_feedback_max_abs_rad
        )
        if (
            not np.isfinite(
                self.output_rear_support_tracking_feedback_max_abs_rad
            )
            or self.output_rear_support_tracking_feedback_max_abs_rad < 0.0
        ):
            raise ValueError(
                "output_rear_support_tracking_feedback_max_abs_rad must be "
                "finite and non-negative."
            )
        rear_support_feedback_lookahead = (
            config.reference_action_lookahead_steps
            if output_rear_support_tracking_feedback_lookahead_steps is None
            else output_rear_support_tracking_feedback_lookahead_steps
        )
        if (
            isinstance(rear_support_feedback_lookahead, bool)
            or not isinstance(
                rear_support_feedback_lookahead,
                (int, np.integer),
            )
            or rear_support_feedback_lookahead < 0
        ):
            raise ValueError(
                "output_rear_support_tracking_feedback_lookahead_steps must "
                "be a non-negative integer."
            )
        self.output_rear_support_tracking_feedback_lookahead_steps = int(
            rear_support_feedback_lookahead
        )
        if (
            isinstance(
                output_rear_support_tracking_feedback_start_frame,
                bool,
            )
            or not isinstance(
                output_rear_support_tracking_feedback_start_frame,
                (int, np.integer),
            )
            or output_rear_support_tracking_feedback_start_frame < 0
        ):
            raise ValueError(
                "output_rear_support_tracking_feedback_start_frame must be "
                "a non-negative integer."
            )
        self.output_rear_support_tracking_feedback_start_frame = int(
            output_rear_support_tracking_feedback_start_frame
        )
        rear_support_feedback_configured = (
            self.output_rear_support_tracking_feedback_gain > 0.0
            or self.output_rear_support_tracking_feedback_max_abs_rad > 0.0
            or bool(self.output_rear_support_tracking_feedback_ref_ids)
        )
        if rear_support_feedback_configured and (
            self.output_rear_support_tracking_feedback_gain == 0.0
            or self.output_rear_support_tracking_feedback_max_abs_rad == 0.0
            or not self.output_rear_support_tracking_feedback_ref_ids
        ):
            raise ValueError(
                "Rear-support tracking feedback requires non-empty reference "
                "IDs, a positive gain, and a positive maximum correction."
            )
        if output_pitch_feedback_axis not in ("x", "y", "z"):
            raise ValueError(
                "output_pitch_feedback_axis must be one of 'x', 'y', or 'z'."
            )
        self.output_pitch_feedback_axis = output_pitch_feedback_axis
        self.output_pitch_feedback_axis_index = {
            "x": 0,
            "y": 1,
            "z": 2,
        }[output_pitch_feedback_axis]
        if (
            isinstance(output_pitch_feedback_start_frame, bool)
            or not isinstance(
                output_pitch_feedback_start_frame,
                (int, np.integer),
            )
            or int(output_pitch_feedback_start_frame) < 0
        ):
            raise ValueError(
                "output_pitch_feedback_start_frame must be a non-negative "
                "integer."
            )
        self.output_pitch_feedback_start_frame = int(
            output_pitch_feedback_start_frame
        )
        raw_pitch_feedback_ref_ids = tuple(
            output_pitch_feedback_ref_ids or ()
        )
        if any(
            isinstance(ref_id, bool)
            or not isinstance(ref_id, (int, np.integer))
            for ref_id in raw_pitch_feedback_ref_ids
        ):
            raise ValueError(
                "output_pitch_feedback_ref_ids must contain only integer "
                "reference IDs."
            )
        pitch_feedback_ref_ids = tuple(
            int(ref_id) for ref_id in raw_pitch_feedback_ref_ids
        )
        if len(set(pitch_feedback_ref_ids)) != len(
            pitch_feedback_ref_ids
        ):
            raise ValueError(
                "output_pitch_feedback_ref_ids must not contain duplicates."
            )
        if any(
            not 0 <= ref_id < len(references)
            for ref_id in pitch_feedback_ref_ids
        ):
            raise ValueError(
                "output_pitch_feedback_ref_ids contains an ID outside the "
                "active reference bank."
            )
        self.output_pitch_feedback_ref_ids = frozenset(
            pitch_feedback_ref_ids
        )
        if output_pitch_feedback_gain_leg is None:
            self.output_pitch_feedback_gain_leg = torch.zeros(
                12,
                dtype=torch.float32,
                device=adapter.base.device,
            )
        else:
            self.output_pitch_feedback_gain_leg = torch.as_tensor(
                output_pitch_feedback_gain_leg,
                dtype=torch.float32,
                device=adapter.base.device,
            )
            if self.output_pitch_feedback_gain_leg.shape != (12,):
                raise ValueError(
                    "output_pitch_feedback_gain_leg must contain 12 "
                    "physical joint-position gains."
                )
            if not torch.isfinite(
                self.output_pitch_feedback_gain_leg
            ).all():
                raise ValueError(
                    "output_pitch_feedback_gain_leg contains NaN or Inf."
                )
        self.output_pitch_feedback_max_abs_rad = float(
            output_pitch_feedback_max_abs_rad
        )
        if (
            not np.isfinite(self.output_pitch_feedback_max_abs_rad)
            or self.output_pitch_feedback_max_abs_rad < 0.0
        ):
            raise ValueError(
                "output_pitch_feedback_max_abs_rad must be finite and "
                "non-negative."
            )
        if torch.any(self.output_pitch_feedback_gain_leg != 0.0):
            if not self.output_pitch_feedback_ref_ids:
                raise ValueError(
                    "Non-zero output pitch-feedback gains require at least "
                    "one output_pitch_feedback_ref_id."
                )
            if self.output_pitch_feedback_max_abs_rad == 0.0:
                raise ValueError(
                    "Non-zero output pitch-feedback gains require a positive "
                    "output_pitch_feedback_max_abs_rad."
                )
        raw_contact_orientation_ref_ids = tuple(
            output_contact_orientation_feedback_ref_ids or ()
        )
        if any(
            isinstance(ref_id, bool)
            or not isinstance(ref_id, (int, np.integer))
            for ref_id in raw_contact_orientation_ref_ids
        ):
            raise ValueError(
                "output_contact_orientation_feedback_ref_ids must contain "
                "only integer reference IDs."
            )
        contact_orientation_ref_ids = tuple(
            int(ref_id) for ref_id in raw_contact_orientation_ref_ids
        )
        if len(set(contact_orientation_ref_ids)) != len(
            contact_orientation_ref_ids
        ):
            raise ValueError(
                "output_contact_orientation_feedback_ref_ids must not "
                "contain duplicates."
            )
        if any(
            not 0 <= ref_id < len(references)
            for ref_id in contact_orientation_ref_ids
        ):
            raise ValueError(
                "output_contact_orientation_feedback_ref_ids contains an "
                "ID outside the active reference bank."
            )
        self.output_contact_orientation_feedback_ref_ids = frozenset(
            contact_orientation_ref_ids
        )
        if output_contact_orientation_feedback_gain_xyz is None:
            self.output_contact_orientation_feedback_gain_xyz = torch.zeros(
                3,
                dtype=torch.float32,
                device=adapter.base.device,
            )
        else:
            self.output_contact_orientation_feedback_gain_xyz = (
                torch.as_tensor(
                    output_contact_orientation_feedback_gain_xyz,
                    dtype=torch.float32,
                    device=adapter.base.device,
                )
            )
            if self.output_contact_orientation_feedback_gain_xyz.shape != (
                3,
            ):
                raise ValueError(
                    "output_contact_orientation_feedback_gain_xyz must "
                    "contain three target-frame gains."
                )
            if (
                not torch.isfinite(
                    self.output_contact_orientation_feedback_gain_xyz
                ).all()
                or torch.any(
                    self.output_contact_orientation_feedback_gain_xyz < 0.0
                )
                or torch.any(
                    self.output_contact_orientation_feedback_gain_xyz > 1.0
                )
            ):
                raise ValueError(
                    "output_contact_orientation_feedback_gain_xyz must be "
                    "finite and lie in [0,1]."
                )
        if (
            isinstance(output_contact_orientation_feedback_start_frame, bool)
            or not isinstance(
                output_contact_orientation_feedback_start_frame,
                (int, np.integer),
            )
            or int(output_contact_orientation_feedback_start_frame) < 0
        ):
            raise ValueError(
                "output_contact_orientation_feedback_start_frame must be a "
                "non-negative integer."
            )
        self.output_contact_orientation_feedback_start_frame = int(
            output_contact_orientation_feedback_start_frame
        )
        self.output_contact_orientation_feedback_max_endpoint_delta_m = float(
            output_contact_orientation_feedback_max_endpoint_delta_m
        )
        if (
            not np.isfinite(
                self.output_contact_orientation_feedback_max_endpoint_delta_m
            )
            or not 0.0
            <= self.output_contact_orientation_feedback_max_endpoint_delta_m
            <= 0.02
        ):
            raise ValueError(
                "output_contact_orientation_feedback_max_endpoint_delta_m "
                "must be finite and lie in [0,0.02]."
            )
        self.output_contact_orientation_feedback_max_abs_rad = float(
            output_contact_orientation_feedback_max_abs_rad
        )
        if (
            not np.isfinite(
                self.output_contact_orientation_feedback_max_abs_rad
            )
            or not 0.0
            <= self.output_contact_orientation_feedback_max_abs_rad
            <= 0.05
        ):
            raise ValueError(
                "output_contact_orientation_feedback_max_abs_rad must be "
                "finite and lie in [0,0.05]."
            )
        contact_orientation_configured = bool(
            self.output_contact_orientation_feedback_ref_ids
            or torch.any(
                self.output_contact_orientation_feedback_gain_xyz != 0.0
            ).item()
            or (
                self.output_contact_orientation_feedback_max_endpoint_delta_m
                > 0.0
            )
            or self.output_contact_orientation_feedback_max_abs_rad > 0.0
        )
        if contact_orientation_configured and (
            not self.output_contact_orientation_feedback_ref_ids
            or not torch.any(
                self.output_contact_orientation_feedback_gain_xyz > 0.0
            ).item()
            or (
                self.output_contact_orientation_feedback_max_endpoint_delta_m
                == 0.0
            )
            or self.output_contact_orientation_feedback_max_abs_rad == 0.0
        ):
            raise ValueError(
                "Contact-orientation feedback requires non-empty reference "
                "IDs, at least one positive axis gain, a positive endpoint "
                "cap, and a positive joint cap."
            )
        if output_joint_position_offset_leg is None:
            self.output_joint_position_offset_leg = torch.zeros(
                12,
                dtype=torch.float32,
                device=adapter.base.device,
            )
        else:
            self.output_joint_position_offset_leg = torch.as_tensor(
                output_joint_position_offset_leg,
                dtype=torch.float32,
                device=adapter.base.device,
            )
            if self.output_joint_position_offset_leg.shape != (12,):
                raise ValueError(
                    "output_joint_position_offset_leg must contain 12 "
                    "physical joint-position offsets."
                )
            if not torch.isfinite(
                self.output_joint_position_offset_leg
            ).all():
                raise ValueError(
                    "output_joint_position_offset_leg contains NaN or Inf."
                )
        if physical_target_rate_limit_rad_s is None:
            self.max_delta = torch.as_tensor(
                contract.max_raw_delta_per_step[:12],
                device=adapter.base.device,
            )
        else:
            if physical_target_rate_limit_rad_s <= 0.0:
                raise ValueError(
                    "physical_target_rate_limit_rad_s must be positive."
                )
            maximum_physical_delta = (
                float(physical_target_rate_limit_rad_s)
                * float(adapter.base.step_dt)
            )
            self.max_delta = maximum_physical_delta / self.scale
        self._warm_residual: torch.Tensor | None = None
        self._active_solver_schedule_phase_by_ref: dict[int, int] = {}
        self.last_diagnostics: dict[str, Any] = {}

    def reset(self, episode_metadata: dict[str, Any] | None = None) -> None:
        self._warm_residual = None
        self._active_solver_schedule_phase_by_ref = {}
        self.last_diagnostics = {}
        episode_seed = int((episode_metadata or {}).get("seed", 0))
        self.optimizer.reset_seed(self.config.seed + episode_seed)
        self.rollout.reset_episode_alignment()

    def _nominal_frames(self, request: ExpertRequest) -> np.ndarray:
        reference = self.references[request.ref_id]
        return np.minimum(
            request.ref_frame
            + self.config.reference_action_lookahead_steps
            + self.adapter.action_delay_steps
            + np.arange(self.config.horizon),
            reference.frames - 1,
        )

    def _nominal(self, request: ExpertRequest) -> torch.Tensor:
        reference = self.references[request.ref_id]
        frames = self._nominal_frames(request)
        raw_action_reference = (
            self.nominal_action_reference_raw_by_ref.get(request.ref_id)
        )
        if raw_action_reference is not None:
            return raw_action_reference[
                torch.as_tensor(
                    frames,
                    dtype=torch.long,
                    device=self.adapter.base.device,
                )
            ]
        action_reference = (
            self.nominal_action_reference_q_des_by_ref.get(request.ref_id)
        )
        if action_reference is None:
            q_ref = torch.as_tensor(
                reference.joint_pos[frames, :12],
                dtype=torch.float32,
                device=self.adapter.base.device,
            )
        else:
            q_ref = action_reference[
                torch.as_tensor(
                    frames,
                    dtype=torch.long,
                    device=self.adapter.base.device,
                )
            ]
        if self.nominal_joint_position_bias_ramp_frames == 0:
            bias_factor = torch.as_tensor(
                frames >= self.nominal_joint_position_bias_start_frame,
                dtype=torch.float32,
                device=self.adapter.base.device,
            )
        else:
            bias_factor = torch.as_tensor(
                (
                    frames - self.nominal_joint_position_bias_start_frame
                )
                / self.nominal_joint_position_bias_ramp_frames,
                dtype=torch.float32,
                device=self.adapter.base.device,
            ).clamp_(0.0, 1.0)
        q_ref = q_ref + (
            bias_factor.unsqueeze(-1) * self.nominal_joint_position_bias_leg
        )
        if self.nominal_front_force_feedback_target_n > 0.0:
            front_normal = torch.abs(
                self.adapter.contact_sensor.data.net_forces_w[
                    0,
                    self.adapter.contact_body_ids[:2],
                    0,
                ]
            )
            force_deficit = torch.clamp(
                (
                    self.nominal_front_force_feedback_target_n
                    - front_normal
                )
                / self.nominal_front_force_feedback_target_n,
                min=0.0,
                max=1.0,
            )
            desired_front = torch.as_tensor(
                self.adapter.contact_schedules[request.ref_id][frames, :2],
                dtype=torch.float32,
                device=self.adapter.base.device,
            )
            feedback_factor = torch.zeros(
                (self.config.horizon, 12),
                dtype=torch.float32,
                device=self.adapter.base.device,
            )
            for front_index, joint_indices in enumerate(
                ((0, 4, 8), (1, 5, 9))
            ):
                feedback_factor[:, joint_indices] = (
                    desired_front[:, front_index : front_index + 1]
                    * force_deficit[front_index]
                )
            q_ref = q_ref + (
                feedback_factor
                * self.nominal_front_force_feedback_gain_leg
            )
        return (q_ref - self.offset) / self.scale

    def _rear_swing_reference_proposals(
        self,
        request: ExpertRequest,
        nominal: torch.Tensor,
    ) -> tuple[torch.Tensor | None, dict[str, Any]]:
        """Build coherent rear-swing offsets from the stored nominal.

        Each offset points the complete horizon toward the frozen reference
        joint trajectory only while the corresponding rear wheel is scheduled
        to swing. MPPI applies it to the current iteration centre, then the
        ordinary projection and Isaac rollout cost remain authoritative.
        """

        configured = bool(self.rear_swing_reference_proposal_scales)
        active_for_ref = (
            request.ref_id
            in self.rear_swing_reference_proposal_ref_ids
        )
        disabled_diagnostics = {
            "enabled": False,
            "configured": configured,
            "active_for_ref": active_for_ref,
            "proposal_count": 0,
            "scales": [],
            "reference_frames": [],
            "reference_target_frames": [],
            "lead_steps": self.rear_swing_reference_proposal_lead_steps,
            "rear_swing_step_count": [0, 0],
            "rear_swing_active_step_count": [0, 0],
            "maximum_unscaled_physical_correction_rad": 0.0,
            "maximum_requested_physical_correction_rad": [],
        }
        if not configured or not active_for_ref:
            return None, disabled_diagnostics
        if nominal.shape != (self.config.horizon, 12):
            raise ValueError(
                "Rear-swing reference proposal nominal shape mismatch."
            )

        reference = self.references[request.ref_id]
        frames = self._nominal_frames(request)
        target_frames = np.minimum(
            frames + self.rear_swing_reference_proposal_lead_steps,
            reference.frames - 1,
        )
        reference_q = torch.as_tensor(
            reference.joint_pos[target_frames, :12],
            dtype=torch.float32,
            device=self.adapter.base.device,
        )
        nominal_q = self.offset + self.scale * nominal
        physical_direction = (
            (reference_q - nominal_q)
            * self.rear_swing_reference_proposal_joint_mask_leg
        )
        contact_schedule = self.adapter.contact_schedules[request.ref_id]
        desired_rear_np = contact_schedule[frames, 2:4]
        rear_swing_active_np = np.stack(
            tuple(
                np.any(
                    ~contact_schedule[
                        int(frame) : int(target_frame) + 1,
                        2:4,
                    ],
                    axis=0,
                )
                for frame, target_frame in zip(
                    frames,
                    target_frames,
                    strict=True,
                )
            ),
            axis=0,
        )
        desired_rear = torch.as_tensor(
            desired_rear_np,
            dtype=torch.bool,
            device=self.adapter.base.device,
        )
        rear_swing_active = torch.as_tensor(
            rear_swing_active_np,
            dtype=torch.bool,
            device=self.adapter.base.device,
        )
        for rear_index, joint_indices in enumerate(
            ((2, 6, 10), (3, 7, 11))
        ):
            swing = rear_swing_active[:, rear_index].to(
                dtype=torch.float32
            )
            physical_direction[:, list(joint_indices)] *= (
                swing.unsqueeze(-1)
            )

        proposal_offsets = torch.stack(
            tuple(
                float(proposal_scale)
                * physical_direction
                / self.scale
                for proposal_scale
                in self.rear_swing_reference_proposal_scales
            )
        )
        maximum_direction = float(
            torch.max(torch.abs(physical_direction)).item()
        )
        return proposal_offsets, {
            "enabled": True,
            "configured": True,
            "active_for_ref": True,
            "proposal_count": len(
                self.rear_swing_reference_proposal_scales
            ),
            "scales": list(
                self.rear_swing_reference_proposal_scales
            ),
            "reference_frames": [
                int(frames[0]),
                int(frames[-1]),
            ],
            "reference_target_frames": [
                int(target_frames[0]),
                int(target_frames[-1]),
            ],
            "lead_steps": self.rear_swing_reference_proposal_lead_steps,
            "rear_swing_step_count": (
                (~desired_rear).sum(dim=0).detach().cpu().tolist()
            ),
            "rear_swing_active_step_count": (
                rear_swing_active.sum(dim=0).detach().cpu().tolist()
            ),
            "maximum_unscaled_physical_correction_rad": (
                maximum_direction
            ),
            "maximum_requested_physical_correction_rad": [
                maximum_direction * float(proposal_scale)
                for proposal_scale
                in self.rear_swing_reference_proposal_scales
            ],
        }

    def _front_support_proposals(
        self,
        request: ExpertRequest,
        nominal: torch.Tensor,
    ) -> tuple[torch.Tensor | None, dict[str, Any]]:
        """Build schedule-gated coherent front-leg pressure offsets.

        These are ordinary MPPI candidates, not post-solve corrections.  Each
        physical joint-position direction is enabled only for a front wheel
        whose frozen contact schedule requests support.  The optimizer still
        projects every candidate through the unchanged raw and sequential
        target-rate limits before evaluating the full Isaac rollout cost.
        """

        configured = bool(self.front_support_proposal_scales)
        active_for_ref = request.ref_id in self.front_support_proposal_ref_ids
        active_for_frame = (
            request.ref_frame >= self.front_support_proposal_start_frame
        )
        disabled_diagnostics = {
            "enabled": False,
            "configured": configured,
            "active_for_ref": active_for_ref,
            "active_for_frame": active_for_frame,
            "start_frame": self.front_support_proposal_start_frame,
            "proposal_count": 0,
            "scales": [],
            "reference_frames": [],
            "front_support_step_count": [0, 0],
            "maximum_unscaled_physical_correction_rad": 0.0,
            "maximum_requested_physical_correction_rad": [],
        }
        if not configured or not active_for_ref or not active_for_frame:
            return None, disabled_diagnostics
        if nominal.shape != (self.config.horizon, 12):
            raise ValueError(
                "Front-support proposal nominal shape mismatch."
            )

        frames = self._nominal_frames(request)
        desired_front = torch.as_tensor(
            self.adapter.contact_schedules[request.ref_id][frames, :2],
            dtype=torch.float32,
            device=self.adapter.base.device,
        )
        physical_direction = self.front_support_proposal_gain_leg_rad.repeat(
            self.config.horizon,
            1,
        )
        for front_index, joint_indices in enumerate(
            ((0, 4, 8), (1, 5, 9))
        ):
            physical_direction[:, list(joint_indices)] *= (
                desired_front[:, front_index : front_index + 1]
            )

        proposal_offsets = torch.stack(
            tuple(
                float(proposal_scale)
                * physical_direction
                / self.scale
                for proposal_scale in self.front_support_proposal_scales
            )
        )
        maximum_direction = float(
            torch.max(torch.abs(physical_direction)).item()
        )
        return proposal_offsets, {
            "enabled": True,
            "configured": True,
            "active_for_ref": True,
            "active_for_frame": True,
            "start_frame": self.front_support_proposal_start_frame,
            "proposal_count": len(self.front_support_proposal_scales),
            "scales": list(self.front_support_proposal_scales),
            "reference_frames": [
                int(frames[0]),
                int(frames[-1]),
            ],
            "front_support_step_count": (
                desired_front.sum(dim=0).detach().cpu().tolist()
            ),
            "maximum_unscaled_physical_correction_rad": maximum_direction,
            "maximum_requested_physical_correction_rad": [
                maximum_direction * float(proposal_scale)
                for proposal_scale in self.front_support_proposal_scales
            ],
        }

    def _rear_swing_tracking_error_proposals(
        self,
        request: ExpertRequest,
        nominal: torch.Tensor,
    ) -> tuple[torch.Tensor | None, dict[str, Any]]:
        """Add bounded reference-tracking error to rear swing commands.

        The frozen reference remains the target.  These ordinary MPPI
        candidates compensate only the measured joint tracking lag under
        load, remain gated by the frozen rear contact schedule, and still
        pass through the unchanged raw/rate projection and full Isaac cost.
        """

        configured = bool(
            self.rear_swing_tracking_error_proposal_scales
        )
        active_for_ref = (
            request.ref_id
            in self.rear_swing_reference_proposal_ref_ids
        )
        active_for_frame = (
            request.ref_frame
            >= self.rear_swing_tracking_error_proposal_start_frame
        )
        disabled_diagnostics = {
            "enabled": False,
            "configured": configured,
            "active_for_ref": active_for_ref,
            "active_for_frame": active_for_frame,
            "start_frame": (
                self.rear_swing_tracking_error_proposal_start_frame
            ),
            "proposal_count": 0,
            "scales": [],
            "reference_frames": [],
            "reference_target_frames": [],
            "lead_steps": self.rear_swing_reference_proposal_lead_steps,
            "rear_swing_active_step_count": [0, 0],
            "maximum_joint_tracking_error_rad": 0.0,
            "maximum_requested_physical_correction_rad": [],
        }
        if not configured or not active_for_ref or not active_for_frame:
            return None, disabled_diagnostics
        if nominal.shape != (self.config.horizon, 12):
            raise ValueError(
                "Rear-swing tracking-error proposal nominal shape mismatch."
            )

        reference = self.references[request.ref_id]
        frames = self._nominal_frames(request)
        target_frames = np.minimum(
            frames + self.rear_swing_reference_proposal_lead_steps,
            reference.frames - 1,
        )
        reference_q = torch.as_tensor(
            reference.joint_pos[target_frames, :12],
            dtype=torch.float32,
            device=self.adapter.base.device,
        )
        actual_q = torch.as_tensor(
            np.asarray(request.q, dtype=np.float32)[:12],
            dtype=torch.float32,
            device=self.adapter.base.device,
        ).view(1, 12)
        physical_direction = (
            (reference_q - actual_q)
            * self.rear_swing_tracking_error_proposal_joint_mask_leg
        )
        contact_schedule = self.adapter.contact_schedules[request.ref_id]
        rear_swing_active_np = np.stack(
            tuple(
                np.any(
                    ~contact_schedule[
                        int(frame) : int(target_frame) + 1,
                        2:4,
                    ],
                    axis=0,
                )
                for frame, target_frame in zip(
                    frames,
                    target_frames,
                    strict=True,
                )
            ),
            axis=0,
        )
        rear_swing_active = torch.as_tensor(
            rear_swing_active_np,
            dtype=torch.float32,
            device=self.adapter.base.device,
        )
        for rear_index, joint_indices in enumerate(
            ((2, 6, 10), (3, 7, 11))
        ):
            physical_direction[:, list(joint_indices)] *= (
                rear_swing_active[
                    :,
                    rear_index : rear_index + 1,
                ]
            )

        proposal_offsets = torch.stack(
            tuple(
                float(proposal_scale)
                * physical_direction
                / self.scale
                for proposal_scale
                in self.rear_swing_tracking_error_proposal_scales
            )
        )
        maximum_direction = float(
            torch.max(torch.abs(physical_direction)).item()
        )
        return proposal_offsets, {
            "enabled": True,
            "configured": True,
            "active_for_ref": True,
            "active_for_frame": True,
            "start_frame": (
                self.rear_swing_tracking_error_proposal_start_frame
            ),
            "proposal_count": len(
                self.rear_swing_tracking_error_proposal_scales
            ),
            "scales": list(
                self.rear_swing_tracking_error_proposal_scales
            ),
            "reference_frames": [
                int(frames[0]),
                int(frames[-1]),
            ],
            "reference_target_frames": [
                int(target_frames[0]),
                int(target_frames[-1]),
            ],
            "lead_steps": self.rear_swing_reference_proposal_lead_steps,
            "rear_swing_active_step_count": (
                rear_swing_active_np.sum(axis=0).tolist()
            ),
            "maximum_joint_tracking_error_rad": maximum_direction,
            "maximum_requested_physical_correction_rad": [
                maximum_direction * float(proposal_scale)
                for proposal_scale
                in self.rear_swing_tracking_error_proposal_scales
            ],
        }

    def _rear_swing_load_transfer_proposals(
        self,
        request: ExpertRequest,
        nominal: torch.Tensor,
    ) -> tuple[torch.Tensor | None, dict[str, Any]]:
        """Build causal, schedule-gated rear-swing unloading candidates.

        Each configured row is a physical 12-leg-joint direction for an
        imminent RL or RR swing.  The frozen contact schedule and the same
        lead window used by rear-swing reference proposals gate the direction
        per rollout step.  These remain ordinary MPPI candidates: unchanged
        raw/rate projection and the complete Isaac rollout cost select or
        reject them.
        """

        configured = bool(
            self.rear_swing_load_transfer_proposal_scales
        )
        active_for_ref = (
            request.ref_id
            in self.rear_swing_load_transfer_proposal_ref_ids
        )
        active_for_frame_by_wheel = tuple(
            request.ref_frame >= start_frame
            for start_frame
            in self.rear_swing_load_transfer_proposal_start_frame_by_wheel
        )
        active_for_frame = any(active_for_frame_by_wheel)
        disabled_diagnostics = {
            "enabled": False,
            "configured": configured,
            "active_for_ref": active_for_ref,
            "active_for_frame": active_for_frame,
            "active_for_state": False,
            "start_frame": (
                self.rear_swing_load_transfer_proposal_start_frame
            ),
            "start_frame_by_wheel": list(
                self.rear_swing_load_transfer_proposal_start_frame_by_wheel
            ),
            "active_for_frame_by_wheel": list(
                active_for_frame_by_wheel
            ),
            "gate_mode": (
                self.rear_swing_load_transfer_proposal_gate_mode
            ),
            "imbalance_threshold_n": (
                self.rear_swing_load_transfer_proposal_imbalance_threshold_n
            ),
            "rear_normal_n": [0.0, 0.0],
            "rear_force_imbalance_by_wheel_n": [0.0, 0.0],
            "proposal_count": 0,
            "scales": [],
            "reference_frames": [],
            "reference_target_frames": [],
            "lead_steps": self.rear_swing_reference_proposal_lead_steps,
            "rear_swing_active_step_count": [0, 0],
            "maximum_unscaled_physical_correction_rad": 0.0,
            "maximum_requested_physical_correction_rad": [],
        }
        if not configured or not active_for_ref or not active_for_frame:
            return None, disabled_diagnostics
        if nominal.shape != (self.config.horizon, 12):
            raise ValueError(
                "Rear-swing load-transfer proposal nominal shape mismatch."
            )

        reference = self.references[request.ref_id]
        frames = self._nominal_frames(request)
        target_frames = np.minimum(
            frames + self.rear_swing_reference_proposal_lead_steps,
            reference.frames - 1,
        )
        contact_schedule = self.adapter.contact_schedules[request.ref_id]
        rear_normal = torch.zeros(
            2,
            dtype=torch.float32,
            device=self.adapter.base.device,
        )
        rear_force_imbalance = torch.zeros_like(rear_normal)
        if (
            self.rear_swing_load_transfer_proposal_gate_mode
            == "swing_schedule"
        ):
            rear_swing_active_np = np.stack(
                tuple(
                    np.any(
                        ~contact_schedule[
                            int(frame) : int(target_frame) + 1,
                            2:4,
                        ],
                        axis=0,
                    )
                    for frame, target_frame in zip(
                        frames,
                        target_frames,
                        strict=True,
                    )
                ),
                axis=0,
            )
            rear_swing_active_np &= np.asarray(
                active_for_frame_by_wheel,
                dtype=bool,
            )[None, :]
        else:
            rear_normal = torch.abs(
                self.adapter.contact_sensor.data.net_forces_w[
                    0,
                    self.adapter.contact_body_ids[2:4],
                    2,
                ]
            )
            rear_force_imbalance = torch.stack(
                (
                    rear_normal[0] - rear_normal[1],
                    rear_normal[1] - rear_normal[0],
                )
            )
            active_by_wheel = (
                rear_force_imbalance
                >= self.rear_swing_load_transfer_proposal_imbalance_threshold_n
            )
            active_by_wheel &= torch.as_tensor(
                active_for_frame_by_wheel,
                dtype=torch.bool,
                device=self.adapter.base.device,
            )
            if not torch.any(active_by_wheel):
                disabled_diagnostics.update(
                    {
                        "active_for_state": False,
                        "rear_normal_n": (
                            rear_normal.detach().cpu().tolist()
                        ),
                        "rear_force_imbalance_by_wheel_n": (
                            rear_force_imbalance.detach().cpu().tolist()
                        ),
                    }
                )
                return None, disabled_diagnostics
            rear_swing_active_np = np.broadcast_to(
                active_by_wheel.detach().cpu().numpy(),
                (self.config.horizon, 2),
            ).copy()
        rear_swing_active = torch.as_tensor(
            rear_swing_active_np,
            dtype=torch.float32,
            device=self.adapter.base.device,
        )
        physical_direction = (
            rear_swing_active
            @ self.rear_swing_load_transfer_proposal_gain_by_wheel_leg_rad
        )
        proposal_offsets = torch.stack(
            tuple(
                float(proposal_scale)
                * physical_direction
                / self.scale
                for proposal_scale
                in self.rear_swing_load_transfer_proposal_scales
            )
        )
        maximum_direction = float(
            torch.max(torch.abs(physical_direction)).item()
        )
        return proposal_offsets, {
            "enabled": True,
            "configured": True,
            "active_for_ref": True,
            "active_for_frame": True,
            "active_for_state": True,
            "start_frame": (
                self.rear_swing_load_transfer_proposal_start_frame
            ),
            "start_frame_by_wheel": list(
                self.rear_swing_load_transfer_proposal_start_frame_by_wheel
            ),
            "active_for_frame_by_wheel": list(
                active_for_frame_by_wheel
            ),
            "gate_mode": (
                self.rear_swing_load_transfer_proposal_gate_mode
            ),
            "imbalance_threshold_n": (
                self.rear_swing_load_transfer_proposal_imbalance_threshold_n
            ),
            "rear_normal_n": rear_normal.detach().cpu().tolist(),
            "rear_force_imbalance_by_wheel_n": (
                rear_force_imbalance.detach().cpu().tolist()
            ),
            "proposal_count": len(
                self.rear_swing_load_transfer_proposal_scales
            ),
            "scales": list(
                self.rear_swing_load_transfer_proposal_scales
            ),
            "reference_frames": [
                int(frames[0]),
                int(frames[-1]),
            ],
            "reference_target_frames": [
                int(target_frames[0]),
                int(target_frames[-1]),
            ],
            "lead_steps": self.rear_swing_reference_proposal_lead_steps,
            "rear_swing_active_step_count": (
                rear_swing_active_np.sum(axis=0).tolist()
            ),
            "maximum_unscaled_physical_correction_rad": (
                maximum_direction
            ),
            "maximum_requested_physical_correction_rad": [
                maximum_direction * float(proposal_scale)
                for proposal_scale
                in self.rear_swing_load_transfer_proposal_scales
            ],
        }

    def _coordinated_rear_swing_front_support_proposals(
        self,
        request: ExpertRequest,
        nominal: torch.Tensor,
        rear_swing_offsets: torch.Tensor | None,
        front_support_offsets: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, dict[str, Any]]:
        """Pair rear-swing motion with front-support preload candidates.

        These candidates remain ordinary MPPI samples.  The front-support
        part, and the optional opposite rear support-leg reference part, are
        active only during the same frozen-schedule lead window as a rear
        swing.  Every combined sequence still passes through the unchanged
        raw/rate projection and full Isaac rollout cost.
        """

        configured = self.combine_rear_swing_front_support_proposals
        rear_support_configured = (
            self.include_rear_support_reference_in_coordinated_proposals
        )
        rear_support_active_for_frame = (
            rear_support_configured
            and request.ref_frame
            >= self.rear_support_reference_proposal_start_frame
        )
        disabled_diagnostics = {
            "enabled": False,
            "configured": configured,
            "include_rear_support_reference": rear_support_configured,
            "rear_support_reference_start_frame": (
                self.rear_support_reference_proposal_start_frame
            ),
            "rear_support_reference_active_for_frame": (
                rear_support_active_for_frame
            ),
            "proposal_count": 0,
            "scale_pairs": [],
            "rear_swing_active_step_count": 0,
            "rear_support_reference_active_step_count": [0, 0],
            "maximum_rear_support_reference_correction_rad": 0.0,
            "maximum_requested_physical_correction_rad": 0.0,
        }
        if (
            not configured
            or rear_swing_offsets is None
            or front_support_offsets is None
        ):
            return None, disabled_diagnostics
        expected_tail = (self.config.horizon, 12)
        if (
            tuple(nominal.shape) != expected_tail
            or tuple(rear_swing_offsets.shape[1:]) != expected_tail
            or tuple(front_support_offsets.shape[1:]) != expected_tail
        ):
            raise ValueError(
                "Coordinated rear-swing/front-support proposal shape "
                "mismatch."
            )

        reference = self.references[request.ref_id]
        frames = self._nominal_frames(request)
        target_frames = np.minimum(
            frames + self.rear_swing_reference_proposal_lead_steps,
            reference.frames - 1,
        )
        contact_schedule = self.adapter.contact_schedules[request.ref_id]
        rear_swing_active_by_wheel_np = np.asarray(
            [
                np.any(
                    ~contact_schedule[
                        int(frame) : int(target_frame) + 1,
                        2:4,
                    ],
                    axis=0,
                )
                for frame, target_frame in zip(
                    frames,
                    target_frames,
                    strict=True,
                )
            ],
            dtype=bool,
        )
        rear_swing_active_np = np.any(
            rear_swing_active_by_wheel_np,
            axis=1,
        ).astype(np.float32)
        rear_swing_active = torch.as_tensor(
            rear_swing_active_np,
            dtype=torch.float32,
            device=self.adapter.base.device,
        ).view(1, self.config.horizon, 1)
        gated_front_offsets = (
            front_support_offsets * rear_swing_active
        )
        base_coordinated_offsets = (
            rear_swing_offsets[:, None, :, :]
            + gated_front_offsets[None, :, :, :]
        ).reshape(-1, self.config.horizon, 12)
        coordinated_offset_groups = [base_coordinated_offsets]
        scale_pairs = [
            {
                "rear_scale": float(rear_scale),
                "front_scale": float(front_scale),
                "include_rear_support_reference": False,
            }
            for rear_scale in self.rear_swing_reference_proposal_scales
            for front_scale in self.front_support_proposal_scales
        ]
        rear_support_active_np = np.zeros_like(
            rear_swing_active_by_wheel_np,
        )
        maximum_rear_support_correction = 0.0
        if rear_support_active_for_frame:
            reference_q = torch.as_tensor(
                reference.joint_pos[target_frames, :12],
                dtype=torch.float32,
                device=self.adapter.base.device,
            )
            nominal_q = self.offset + self.scale * nominal
            rear_support_physical_direction = (
                (reference_q - nominal_q)
                * self.rear_swing_reference_proposal_joint_mask_leg
            )
            rear_support_active_np[:, 0] = (
                rear_swing_active_by_wheel_np[:, 1]
            )
            rear_support_active_np[:, 1] = (
                rear_swing_active_by_wheel_np[:, 0]
            )
            rear_support_active = torch.as_tensor(
                rear_support_active_np,
                dtype=torch.float32,
                device=self.adapter.base.device,
            )
            for rear_index, joint_indices in enumerate(
                ((2, 6, 10), (3, 7, 11))
            ):
                rear_support_physical_direction[
                    :,
                    list(joint_indices),
                ] *= rear_support_active[
                    :,
                    rear_index : rear_index + 1,
                ]
            rear_support_offsets = torch.stack(
                tuple(
                    float(proposal_scale)
                    * rear_support_physical_direction
                    / self.scale
                    for proposal_scale
                    in self.rear_swing_reference_proposal_scales
                )
            )
            support_coordinated_offsets = (
                rear_swing_offsets + rear_support_offsets
                + gated_front_offsets[0:1]
            )
            coordinated_offset_groups.append(
                support_coordinated_offsets
            )
            scale_pairs.extend(
                {
                    "rear_scale": float(rear_scale),
                    "front_scale": float(
                        self.front_support_proposal_scales[0]
                    ),
                    "include_rear_support_reference": True,
                }
                for rear_scale
                in self.rear_swing_reference_proposal_scales
            )
            maximum_rear_support_correction = float(
                torch.max(
                    torch.abs(
                        rear_support_offsets * self.scale
                    )
                ).item()
            )
        coordinated_offsets = torch.cat(
            coordinated_offset_groups,
            dim=0,
        )
        physical_offsets = coordinated_offsets * self.scale
        maximum_correction = float(
            torch.max(torch.abs(physical_offsets)).item()
        )
        return coordinated_offsets, {
            "enabled": True,
            "configured": True,
            "include_rear_support_reference": rear_support_configured,
            "rear_support_reference_start_frame": (
                self.rear_support_reference_proposal_start_frame
            ),
            "rear_support_reference_active_for_frame": (
                rear_support_active_for_frame
            ),
            "proposal_count": len(scale_pairs),
            "scale_pairs": scale_pairs,
            "rear_swing_active_step_count": int(
                rear_swing_active_np.sum()
            ),
            "rear_support_reference_active_step_count": (
                rear_support_active_np.sum(axis=0).tolist()
            ),
            "maximum_rear_support_reference_correction_rad": (
                maximum_rear_support_correction
            ),
            "maximum_requested_physical_correction_rad": (
                maximum_correction
            ),
        }

    def _coordinated_rear_swing_load_transfer_front_support_proposals(
        self,
        request: ExpertRequest,
        nominal: torch.Tensor,
        load_transfer_offsets: torch.Tensor | None,
        front_support_offsets: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, dict[str, Any]]:
        """Pair scheduled rear unloading with scheduled front preload.

        The load-transfer and front-support parts share the same frozen
        rear-swing lead mask.  Their Cartesian product remains a set of
        ordinary MPPI candidates that passes through the unchanged raw/rate
        projection and complete Isaac rollout cost.
        """

        configured = (
            self.combine_rear_swing_load_transfer_front_support_proposals
        )
        disabled_diagnostics = {
            "enabled": False,
            "configured": configured,
            "proposal_count": 0,
            "scale_pairs": [],
            "reference_frames": [],
            "reference_target_frames": [],
            "lead_steps": self.rear_swing_reference_proposal_lead_steps,
            "rear_swing_active_step_count": [0, 0],
            "maximum_requested_physical_correction_rad": 0.0,
        }
        if (
            not configured
            or load_transfer_offsets is None
            or front_support_offsets is None
        ):
            return None, disabled_diagnostics
        if (
            self.rear_swing_load_transfer_proposal_gate_mode
            != "swing_schedule"
        ):
            raise RuntimeError(
                "Coordinated rear-swing load-transfer/front-support "
                "proposals require swing_schedule load-transfer gating."
            )
        expected_tail = (self.config.horizon, 12)
        if (
            tuple(nominal.shape) != expected_tail
            or tuple(load_transfer_offsets.shape[1:]) != expected_tail
            or tuple(front_support_offsets.shape[1:]) != expected_tail
        ):
            raise ValueError(
                "Coordinated rear-swing load-transfer/front-support "
                "proposal shape mismatch."
            )

        reference = self.references[request.ref_id]
        frames = self._nominal_frames(request)
        target_frames = np.minimum(
            frames + self.rear_swing_reference_proposal_lead_steps,
            reference.frames - 1,
        )
        contact_schedule = self.adapter.contact_schedules[request.ref_id]
        rear_swing_active_by_wheel_np = np.asarray(
            [
                np.any(
                    ~contact_schedule[
                        int(frame) : int(target_frame) + 1,
                        2:4,
                    ],
                    axis=0,
                )
                for frame, target_frame in zip(
                    frames,
                    target_frames,
                    strict=True,
                )
            ],
            dtype=bool,
        )
        rear_swing_active = torch.as_tensor(
            np.any(
                rear_swing_active_by_wheel_np,
                axis=1,
            ).astype(np.float32),
            dtype=torch.float32,
            device=self.adapter.base.device,
        ).view(1, self.config.horizon, 1)
        gated_front_offsets = (
            front_support_offsets * rear_swing_active
        )
        coordinated_offsets = (
            load_transfer_offsets[:, None, :, :]
            + gated_front_offsets[None, :, :, :]
        ).reshape(-1, self.config.horizon, 12)
        scale_pairs = [
            {
                "load_transfer_scale": float(load_transfer_scale),
                "front_scale": float(front_scale),
            }
            for load_transfer_scale
            in self.rear_swing_load_transfer_proposal_scales
            for front_scale in self.front_support_proposal_scales
        ]
        maximum_correction = float(
            torch.max(
                torch.abs(coordinated_offsets * self.scale)
            ).item()
        )
        return coordinated_offsets, {
            "enabled": True,
            "configured": True,
            "proposal_count": len(scale_pairs),
            "scale_pairs": scale_pairs,
            "reference_frames": [
                int(frames[0]),
                int(frames[-1]),
            ],
            "reference_target_frames": [
                int(target_frames[0]),
                int(target_frames[-1]),
            ],
            "lead_steps": self.rear_swing_reference_proposal_lead_steps,
            "rear_swing_active_step_count": (
                rear_swing_active_by_wheel_np.sum(axis=0).tolist()
            ),
            "maximum_requested_physical_correction_rad": (
                maximum_correction
            ),
        }

    def _coordinated_rear_swing_reference_load_transfer_front_support_proposals(
        self,
        request: ExpertRequest,
        nominal: torch.Tensor,
        rear_swing_offsets: torch.Tensor | None,
        load_transfer_offsets: torch.Tensor | None,
        front_support_offsets: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, dict[str, Any]]:
        """Pair rear motion, scheduled unloading, and front preload.

        All three components use the frozen rear-swing lead window.  Their
        Cartesian product remains an ordinary MPPI population under the same
        projection and complete Isaac rollout cost as every other sample.
        """

        configured = (
            self.combine_rear_swing_reference_load_transfer_front_support_proposals
        )
        disabled_diagnostics = {
            "enabled": False,
            "configured": configured,
            "proposal_count": 0,
            "scale_triples": [],
            "reference_frames": [],
            "reference_target_frames": [],
            "lead_steps": self.rear_swing_reference_proposal_lead_steps,
            "rear_swing_active_step_count": [0, 0],
            "maximum_requested_physical_correction_rad": 0.0,
        }
        if (
            not configured
            or rear_swing_offsets is None
            or load_transfer_offsets is None
            or front_support_offsets is None
        ):
            return None, disabled_diagnostics
        if (
            self.rear_swing_load_transfer_proposal_gate_mode
            != "swing_schedule"
        ):
            raise RuntimeError(
                "Coordinated rear-swing reference/load-transfer/front-support "
                "proposals require swing_schedule load-transfer gating."
            )
        expected_tail = (self.config.horizon, 12)
        if (
            tuple(nominal.shape) != expected_tail
            or tuple(rear_swing_offsets.shape[1:]) != expected_tail
            or tuple(load_transfer_offsets.shape[1:]) != expected_tail
            or tuple(front_support_offsets.shape[1:]) != expected_tail
        ):
            raise ValueError(
                "Coordinated rear-swing reference/load-transfer/front-support "
                "proposal shape mismatch."
            )

        reference = self.references[request.ref_id]
        frames = self._nominal_frames(request)
        target_frames = np.minimum(
            frames + self.rear_swing_reference_proposal_lead_steps,
            reference.frames - 1,
        )
        contact_schedule = self.adapter.contact_schedules[request.ref_id]
        rear_swing_active_by_wheel_np = np.asarray(
            [
                np.any(
                    ~contact_schedule[
                        int(frame) : int(target_frame) + 1,
                        2:4,
                    ],
                    axis=0,
                )
                for frame, target_frame in zip(
                    frames,
                    target_frames,
                    strict=True,
                )
            ],
            dtype=bool,
        )
        rear_swing_active = torch.as_tensor(
            np.any(
                rear_swing_active_by_wheel_np,
                axis=1,
            ).astype(np.float32),
            dtype=torch.float32,
            device=self.adapter.base.device,
        ).view(1, self.config.horizon, 1)
        gated_front_offsets = front_support_offsets * rear_swing_active
        coordinated_offsets = (
            rear_swing_offsets[:, None, None, :, :]
            + load_transfer_offsets[None, :, None, :, :]
            + gated_front_offsets[None, None, :, :, :]
        ).reshape(-1, self.config.horizon, 12)
        scale_triples = [
            {
                "rear_scale": float(rear_scale),
                "load_transfer_scale": float(load_transfer_scale),
                "front_scale": float(front_scale),
            }
            for rear_scale in self.rear_swing_reference_proposal_scales
            for load_transfer_scale
            in self.rear_swing_load_transfer_proposal_scales
            for front_scale in self.front_support_proposal_scales
        ]
        maximum_correction = float(
            torch.max(
                torch.abs(coordinated_offsets * self.scale)
            ).item()
        )
        return coordinated_offsets, {
            "enabled": True,
            "configured": True,
            "proposal_count": len(scale_triples),
            "scale_triples": scale_triples,
            "reference_frames": [
                int(frames[0]),
                int(frames[-1]),
            ],
            "reference_target_frames": [
                int(target_frames[0]),
                int(target_frames[-1]),
            ],
            "lead_steps": self.rear_swing_reference_proposal_lead_steps,
            "rear_swing_active_step_count": (
                rear_swing_active_by_wheel_np.sum(axis=0).tolist()
            ),
            "maximum_requested_physical_correction_rad": (
                maximum_correction
            ),
        }

    def _apply_output_front_force_feedback(
        self,
        selected_leg: torch.Tensor,
        nominal_leg: torch.Tensor,
        request: ExpertRequest,
        previous_action: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Apply measured front-load correction to the selected raw action.

        Proposal centring alone does not guarantee that the finite-sample MPPI
        solve executes the desired correction.  This final correction is
        therefore deterministic, contact-schedule gated, and projected back
        through the same raw bounds and per-step target-rate limits used by the
        optimizer.  The caller still appends four exact-zero wheel actions.
        """

        if self.output_front_force_feedback_target_n == 0.0:
            return selected_leg, {
                "enabled": False,
                "front_normal_n": [0.0, 0.0],
                "desired_front_contact": [False, False],
                "measured_front_contact": [False, False],
                "force_deficit_fraction": [0.0, 0.0],
                "requested_correction_rad": [0.0] * 12,
                "absolute_feedback_limit_rad": [0.0] * 12,
                "applied_correction_rad": [0.0] * 12,
            }

        front_normal = torch.abs(
            self.adapter.contact_sensor.data.net_forces_w[
                0,
                self.adapter.contact_body_ids[:2],
                0,
            ]
        )
        force_deficit = torch.clamp(
            (
                self.output_front_force_feedback_target_n
                - front_normal
            )
            / self.output_front_force_feedback_target_n,
            min=0.0,
            max=1.0,
        )
        measured_front_contact = (
            front_normal
            >= self.output_front_force_feedback_min_contact_n
        )
        schedule = self.adapter.contact_schedules[request.ref_id]
        schedule_frame = min(
            request.ref_frame
            + self.output_front_force_feedback_lookahead_steps
            + self.adapter.action_delay_steps,
            len(schedule) - 1,
        )
        desired_front = torch.as_tensor(
            schedule[schedule_frame, :2],
            dtype=torch.float32,
            device=self.adapter.base.device,
        )
        feedback_factor = torch.zeros(
            12,
            dtype=torch.float32,
            device=self.adapter.base.device,
        )
        for front_index, joint_indices in enumerate(
            ((0, 4, 8), (1, 5, 9))
        ):
            feedback_factor[list(joint_indices)] = (
                desired_front[front_index]
                * measured_front_contact[front_index].float()
                * force_deficit[front_index]
            )
        requested_correction = (
            feedback_factor * self.output_front_force_feedback_gain_leg
        )
        proposed_leg = selected_leg + requested_correction / self.scale
        # A raw additive correction can otherwise integrate through the
        # previous-action constraint after contact is lost: every new solve is
        # allowed to add the same positive offset again.  Bound each corrected
        # dimension by one full configured feedback offset from this solve's
        # reference-centred nominal action.  This retains the measured local
        # response direction without creating an unbounded position ramp.
        full_offset_raw = (
            self.output_front_force_feedback_gain_leg / self.scale
        )
        feedback_limit = nominal_leg + full_offset_raw
        positive_gain = self.output_front_force_feedback_gain_leg > 0.0
        negative_gain = self.output_front_force_feedback_gain_leg < 0.0
        proposed_leg = torch.where(
            positive_gain,
            torch.maximum(
                selected_leg,
                torch.minimum(proposed_leg, feedback_limit),
            ),
            proposed_leg,
        )
        proposed_leg = torch.where(
            negative_gain,
            torch.minimum(
                selected_leg,
                torch.maximum(proposed_leg, feedback_limit),
            ),
            proposed_leg,
        )
        lower = torch.maximum(
            self.raw_min,
            previous_action - self.max_delta,
        )
        upper = torch.minimum(
            self.raw_max,
            previous_action + self.max_delta,
        )
        corrected_leg = torch.minimum(
            torch.maximum(proposed_leg, lower),
            upper,
        )
        applied_correction = self.scale * (corrected_leg - selected_leg)
        diagnostics = {
            "enabled": True,
            "schedule_frame": schedule_frame,
            "schedule_lookahead_steps": (
                self.output_front_force_feedback_lookahead_steps
            ),
            "front_normal_n": front_normal.detach().cpu().tolist(),
            "desired_front_contact": (
                desired_front.to(dtype=torch.bool).detach().cpu().tolist()
            ),
            "measured_front_contact": (
                measured_front_contact.detach().cpu().tolist()
            ),
            "force_deficit_fraction": force_deficit.detach().cpu().tolist(),
            "requested_correction_rad": (
                requested_correction.detach().cpu().tolist()
            ),
            "absolute_feedback_limit_rad": (
                (self.offset + self.scale * feedback_limit)
                .detach()
                .cpu()
                .tolist()
            ),
            "applied_correction_rad": (
                applied_correction.detach().cpu().tolist()
            ),
        }
        return corrected_leg, diagnostics

    def _apply_output_rear_swing_force_feedback(
        self,
        selected_leg: torch.Tensor,
        nominal_leg: torch.Tensor,
        request: ExpertRequest,
        previous_action: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Unload each scheduled rear swing wheel with bounded force feedback.

        The feedback is independently gated by the left/right rear contact
        schedule.  It responds only to measured load above the configured
        swing target and is capped at one configured physical offset from the
        current nominal action, so repeated solves cannot integrate the
        correction into an unbounded position ramp.
        """

        if (
            self.output_rear_swing_force_feedback_target_n == 0.0
            or request.ref_frame
            < self.output_rear_swing_force_feedback_start_frame
        ):
            return selected_leg, {
                "enabled": False,
                "configured": (
                    self.output_rear_swing_force_feedback_target_n > 0.0
                ),
                "start_frame": (
                    self.output_rear_swing_force_feedback_start_frame
                ),
                "rear_normal_n": [0.0, 0.0],
                "desired_rear_contact": [False, False],
                "force_excess_fraction": [0.0, 0.0],
                "requested_correction_rad": [0.0] * 12,
                "absolute_feedback_limit_rad": [0.0] * 12,
                "applied_correction_rad": [0.0] * 12,
            }

        rear_normal = torch.abs(
            self.adapter.contact_sensor.data.net_forces_w[
                0,
                self.adapter.contact_body_ids[2:4],
                2,
            ]
        )
        force_excess = torch.clamp(
            (
                rear_normal
                - self.output_rear_swing_force_feedback_target_n
            )
            / self.output_rear_swing_force_feedback_scale_n,
            min=0.0,
            max=1.0,
        )
        schedule = self.adapter.contact_schedules[request.ref_id]
        schedule_frame = min(
            request.ref_frame
            + self.output_rear_swing_force_feedback_lookahead_steps
            + self.adapter.action_delay_steps,
            len(schedule) - 1,
        )
        desired_rear = torch.as_tensor(
            schedule[schedule_frame, 2:4],
            dtype=torch.float32,
            device=self.adapter.base.device,
        )
        feedback_factor = torch.zeros(
            12,
            dtype=torch.float32,
            device=self.adapter.base.device,
        )
        for rear_index, joint_indices in enumerate(
            ((2, 6, 10), (3, 7, 11))
        ):
            feedback_factor[list(joint_indices)] = (
                (1.0 - desired_rear[rear_index])
                * force_excess[rear_index]
            )
        requested_correction = (
            feedback_factor
            * self.output_rear_swing_force_feedback_gain_leg
        )
        proposed_leg = selected_leg + requested_correction / self.scale
        full_offset_raw = (
            self.output_rear_swing_force_feedback_gain_leg / self.scale
        )
        feedback_limit = nominal_leg + full_offset_raw
        positive_gain = (
            self.output_rear_swing_force_feedback_gain_leg > 0.0
        )
        negative_gain = (
            self.output_rear_swing_force_feedback_gain_leg < 0.0
        )
        proposed_leg = torch.where(
            positive_gain,
            torch.maximum(
                selected_leg,
                torch.minimum(proposed_leg, feedback_limit),
            ),
            proposed_leg,
        )
        proposed_leg = torch.where(
            negative_gain,
            torch.minimum(
                selected_leg,
                torch.maximum(proposed_leg, feedback_limit),
            ),
            proposed_leg,
        )
        lower = torch.maximum(
            self.raw_min,
            previous_action - self.max_delta,
        )
        upper = torch.minimum(
            self.raw_max,
            previous_action + self.max_delta,
        )
        corrected_leg = torch.minimum(
            torch.maximum(proposed_leg, lower),
            upper,
        )
        applied_correction = self.scale * (
            corrected_leg - selected_leg
        )
        return corrected_leg, {
            "enabled": True,
            "configured": True,
            "start_frame": (
                self.output_rear_swing_force_feedback_start_frame
            ),
            "schedule_frame": schedule_frame,
            "schedule_lookahead_steps": (
                self.output_rear_swing_force_feedback_lookahead_steps
            ),
            "rear_normal_n": rear_normal.detach().cpu().tolist(),
            "desired_rear_contact": (
                desired_rear.to(dtype=torch.bool).detach().cpu().tolist()
            ),
            "force_excess_fraction": force_excess.detach().cpu().tolist(),
            "requested_correction_rad": (
                requested_correction.detach().cpu().tolist()
            ),
            "absolute_feedback_limit_rad": (
                (self.offset + self.scale * feedback_limit)
                .detach()
                .cpu()
                .tolist()
            ),
            "applied_correction_rad": (
                applied_correction.detach().cpu().tolist()
            ),
        }

    def _apply_output_rear_swing_height_feedback(
        self,
        selected_leg: torch.Tensor,
        nominal_leg: torch.Tensor,
        request: ExpertRequest,
        previous_action: torch.Tensor,
        snapshot: IsaacRolloutSnapshot,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Lift a stuck scheduled rear swing only to the frozen reference.

        A measured rear contact during a scheduled swing is corrected with a
        damped least-squares endpoint Jacobian.  The Cartesian target is the
        aligned frozen-reference wheel height, never an added clearance.  The
        resulting joint target leads the measured, load-deflected joint state
        by at most the configured DLS correction and is then projected through
        the unchanged raw-action and physical target-rate limits.
        """

        configured = (
            self.output_rear_swing_height_feedback_gain > 0.0
        )
        active_for_ref = (
            request.ref_id
            in self.output_rear_swing_height_feedback_ref_ids
        )
        active_for_frame = (
            request.ref_frame
            >= self.output_rear_swing_height_feedback_start_frame
        )
        disabled_diagnostics = {
            "enabled": False,
            "configured": configured,
            "active_for_ref": active_for_ref,
            "active_for_frame": active_for_frame,
            "start_frame": (
                self.output_rear_swing_height_feedback_start_frame
            ),
            "schedule_frame": None,
            "preview_start_frame": None,
            "target_frame_by_rear": [None, None],
            "schedule_lookahead_steps": (
                self.output_rear_swing_height_feedback_lookahead_steps
            ),
            "contact_force_threshold_n": (
                self.rollout.contact_force_threshold
            ),
            "damping": 1.0e-3,
            "cartesian_axis_weights": [1.0, 1.0, 4.0],
            "rear_normal_n": [0.0, 0.0],
            "desired_rear_contact": [False, False],
            "measured_rear_contact": [False, False],
            "stuck_rear_swing": [False, False],
            "actual_rear_height_m": [0.0, 0.0],
            "frozen_target_rear_height_m": [0.0, 0.0],
            "height_deficit_m": [0.0, 0.0],
            "jacobian_joint_delta_rad": [0.0] * 12,
            "predicted_cartesian_delta_m_by_rear": [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            "requested_correction_rad": [0.0] * 12,
            "bounded_joint_target_rad": [0.0] * 12,
            "applied_correction_rad": [0.0] * 12,
        }
        if not configured or not active_for_ref or not active_for_frame:
            return selected_leg, disabled_diagnostics

        rear_normal = torch.abs(
            self.adapter.contact_sensor.data.net_forces_w[
                0,
                self.adapter.contact_body_ids[2:4],
                2,
            ]
        )
        measured_rear = (
            rear_normal >= self.rollout.contact_force_threshold
        )
        schedule = self.adapter.contact_schedules[request.ref_id]
        preview_start_frame = min(
            request.ref_frame + self.adapter.action_delay_steps,
            len(schedule) - 1,
        )
        schedule_frame = min(
            request.ref_frame
            + self.output_rear_swing_height_feedback_lookahead_steps
            + self.adapter.action_delay_steps,
            len(schedule) - 1,
        )
        preview_schedule = np.asarray(
            schedule[
                preview_start_frame : schedule_frame + 1,
                2:4,
            ],
            dtype=bool,
        )
        planned_rear_swing = torch.as_tensor(
            np.any(~preview_schedule, axis=0),
            dtype=torch.bool,
            device=self.adapter.base.device,
        )
        desired_rear = torch.as_tensor(
            ~np.any(~preview_schedule, axis=0),
            dtype=torch.bool,
            device=self.adapter.base.device,
        )
        stuck_rear_swing = planned_rear_swing & measured_rear
        actual_rear_height = (
            self.rollout.robot.data.body_pos_w[
                0,
                self.rollout.wheel_body_ids[2:4],
                2,
            ]
            - self.rollout.base.scene.env_origins[0, 2]
        )
        target_rear_height = actual_rear_height.clone()
        target_frame_by_rear: list[int | None] = [None, None]
        aligned_target_cache: dict[int, dict[str, torch.Tensor]] = {}
        for rear_index in range(2):
            preview_swing_offsets = np.flatnonzero(
                ~preview_schedule[:, rear_index]
            )
            if preview_swing_offsets.size == 0:
                continue
            first_swing_frame = (
                preview_start_frame + int(preview_swing_offsets[0])
            )
            swing_start = first_swing_frame
            while (
                swing_start > 0
                and not bool(schedule[swing_start - 1, 2 + rear_index])
            ):
                swing_start -= 1
            swing_stop = first_swing_frame
            while (
                swing_stop + 1 < len(schedule)
                and not bool(schedule[swing_stop + 1, 2 + rear_index])
            ):
                swing_stop += 1
            candidate_frames = tuple(
                range(swing_start, swing_stop + 1)
            )
            candidate_heights = []
            for candidate_frame in candidate_frames:
                if candidate_frame not in aligned_target_cache:
                    aligned_target_cache[candidate_frame] = (
                        self.rollout._aligned_reference(
                            snapshot,
                            candidate_frame,
                        )
                    )
                candidate_heights.append(
                    aligned_target_cache[candidate_frame][
                        "body_pos_local"
                    ][
                        self.rollout.ref_wheel_body_ids[2 + rear_index],
                        2,
                    ]
                )
            candidate_height_tensor = torch.stack(candidate_heights)
            maximum_index = int(
                torch.argmax(candidate_height_tensor).item()
            )
            target_frame_by_rear[rear_index] = candidate_frames[
                maximum_index
            ]
            target_rear_height[rear_index] = (
                candidate_height_tensor[maximum_index]
            )
        height_deficit = torch.clamp(
            target_rear_height - actual_rear_height,
            min=0.0,
        )
        active_height_correction = (
            stuck_rear_swing & (height_deficit > 0.0)
        )

        actual_q = torch.as_tensor(
            np.asarray(request.q, dtype=np.float32)[:12],
            dtype=torch.float32,
            device=self.adapter.base.device,
        )
        selected_q = self.offset + self.scale * selected_leg
        bounded_joint_target = selected_q.clone()
        jacobian_joint_delta = torch.zeros_like(selected_leg)
        predicted_cartesian_delta = torch.zeros(
            (2, 3),
            dtype=torch.float32,
            device=self.adapter.base.device,
        )

        if torch.any(active_height_correction):
            robot = self.rollout.robot
            jacobians = robot.root_physx_view.get_jacobians()
            is_fixed_base = bool(
                getattr(robot, "is_fixed_base", False)
            )
            joint_column_offset = 0 if is_fixed_base else 6
            axis_weights = torch.tensor(
                (1.0, 1.0, 4.0),
                dtype=jacobians.dtype,
                device=jacobians.device,
            )
            damping = torch.as_tensor(
                1.0e-3,
                dtype=jacobians.dtype,
                device=jacobians.device,
            )
            for rear_index, joint_indices in enumerate(
                ((2, 6, 10), (3, 7, 11))
            ):
                if not bool(active_height_correction[rear_index].item()):
                    continue
                body_id = self.rollout.wheel_body_ids[2 + rear_index]
                jacobian_body_id = (
                    body_id - 1 if is_fixed_base else body_id
                )
                joint_columns = torch.as_tensor(
                    [
                        self.rollout.joint_ids[index]
                        + joint_column_offset
                        for index in joint_indices
                    ],
                    dtype=torch.long,
                    device=jacobians.device,
                )
                wheel_jacobian = jacobians[
                    0,
                    jacobian_body_id,
                    :3,
                ].index_select(-1, joint_columns)
                weighted_jacobian = (
                    axis_weights.unsqueeze(-1) * wheel_jacobian
                )
                desired_cartesian_delta = torch.zeros(
                    3,
                    dtype=jacobians.dtype,
                    device=jacobians.device,
                )
                desired_cartesian_delta[2] = (
                    self.output_rear_swing_height_feedback_gain
                    * height_deficit[rear_index]
                )
                weighted_target = (
                    axis_weights * desired_cartesian_delta
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
                    @ weighted_target,
                )
                joint_delta = torch.clamp(
                    joint_delta,
                    min=(
                        -self.output_rear_swing_height_feedback_max_abs_rad
                    ),
                    max=(
                        self.output_rear_swing_height_feedback_max_abs_rad
                    ),
                )
                predicted_delta = wheel_jacobian @ joint_delta
                if (
                    predicted_delta[2] > desired_cartesian_delta[2]
                    and predicted_delta[2] > 0.0
                ):
                    joint_delta = joint_delta * (
                        desired_cartesian_delta[2]
                        / predicted_delta[2]
                    )
                    predicted_delta = wheel_jacobian @ joint_delta

                joint_index_tensor = torch.as_tensor(
                    joint_indices,
                    dtype=torch.long,
                    device=self.adapter.base.device,
                )
                joint_delta = joint_delta.to(
                    dtype=torch.float32,
                    device=self.adapter.base.device,
                )
                jacobian_joint_delta[joint_index_tensor] = joint_delta
                predicted_cartesian_delta[rear_index] = (
                    predicted_delta.to(
                        dtype=torch.float32,
                        device=self.adapter.base.device,
                    )
                )
                unconstrained_target = (
                    actual_q[joint_index_tensor] + joint_delta
                )
                physical_bound_a = (
                    self.offset[joint_index_tensor]
                    + self.scale[joint_index_tensor]
                    * self.raw_min[joint_index_tensor]
                )
                physical_bound_b = (
                    self.offset[joint_index_tensor]
                    + self.scale[joint_index_tensor]
                    * self.raw_max[joint_index_tensor]
                )
                bounded_target = torch.clamp(
                    unconstrained_target,
                    min=torch.minimum(
                        physical_bound_a,
                        physical_bound_b,
                    ),
                    max=torch.maximum(
                        physical_bound_a,
                        physical_bound_b,
                    ),
                )
                bounded_target = torch.where(
                    joint_delta > 0.0,
                    torch.maximum(
                        selected_q[joint_index_tensor],
                        bounded_target,
                    ),
                    bounded_target,
                )
                bounded_target = torch.where(
                    joint_delta < 0.0,
                    torch.minimum(
                        selected_q[joint_index_tensor],
                        bounded_target,
                    ),
                    bounded_target,
                )
                bounded_joint_target[joint_index_tensor] = bounded_target

        requested_correction = bounded_joint_target - selected_q
        proposed_leg = selected_leg.clone()
        correction_mask = requested_correction != 0.0
        proposed_leg[correction_mask] = (
            (
                bounded_joint_target[correction_mask]
                - self.offset[correction_mask]
            )
            / self.scale[correction_mask]
        )
        lower = torch.maximum(
            self.raw_min,
            previous_action - self.max_delta,
        )
        upper = torch.minimum(
            self.raw_max,
            previous_action + self.max_delta,
        )
        corrected_leg = torch.minimum(
            torch.maximum(proposed_leg, lower),
            upper,
        )
        applied_correction = self.scale * (
            corrected_leg - selected_leg
        )
        return corrected_leg, {
            "enabled": True,
            "configured": True,
            "active_for_ref": True,
            "active_for_frame": True,
            "start_frame": (
                self.output_rear_swing_height_feedback_start_frame
            ),
            "schedule_frame": schedule_frame,
            "preview_start_frame": preview_start_frame,
            "target_frame_by_rear": target_frame_by_rear,
            "schedule_lookahead_steps": (
                self.output_rear_swing_height_feedback_lookahead_steps
            ),
            "contact_force_threshold_n": (
                self.rollout.contact_force_threshold
            ),
            "damping": 1.0e-3,
            "cartesian_axis_weights": [1.0, 1.0, 4.0],
            "rear_normal_n": rear_normal.detach().cpu().tolist(),
            "desired_rear_contact": (
                desired_rear.detach().cpu().tolist()
            ),
            "measured_rear_contact": (
                measured_rear.detach().cpu().tolist()
            ),
            "stuck_rear_swing": (
                stuck_rear_swing.detach().cpu().tolist()
            ),
            "actual_rear_height_m": (
                actual_rear_height.detach().cpu().tolist()
            ),
            "frozen_target_rear_height_m": (
                target_rear_height.detach().cpu().tolist()
            ),
            "height_deficit_m": (
                height_deficit.detach().cpu().tolist()
            ),
            "jacobian_joint_delta_rad": (
                jacobian_joint_delta.detach().cpu().tolist()
            ),
            "predicted_cartesian_delta_m_by_rear": (
                predicted_cartesian_delta.detach().cpu().tolist()
            ),
            "requested_correction_rad": (
                requested_correction.detach().cpu().tolist()
            ),
            "bounded_joint_target_rad": (
                bounded_joint_target.detach().cpu().tolist()
            ),
            "applied_correction_rad": (
                applied_correction.detach().cpu().tolist()
            ),
        }

    def _apply_output_rear_support_tracking_feedback(
        self,
        selected_leg: torch.Tensor,
        nominal_leg: torch.Tensor,
        request: ExpertRequest,
        previous_action: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Recover scheduled rear support using bounded reference tracking.

        This correction is active only when a frozen reference requires rear
        support while the corresponding measured rear normal force is below
        the unchanged contact threshold.  It points the affected leg toward
        the frozen reference joint position, is capped relative to the
        reference-centred nominal action, and remains subject to the ordinary
        raw-action and physical target-rate bounds.
        """

        configured = (
            self.output_rear_support_tracking_feedback_gain > 0.0
        )
        active_for_ref = (
            request.ref_id
            in self.output_rear_support_tracking_feedback_ref_ids
        )
        active_for_frame = (
            request.ref_frame
            >= self.output_rear_support_tracking_feedback_start_frame
        )
        disabled_diagnostics = {
            "enabled": False,
            "configured": configured,
            "active_for_ref": active_for_ref,
            "active_for_frame": active_for_frame,
            "start_frame": (
                self.output_rear_support_tracking_feedback_start_frame
            ),
            "schedule_frame": None,
            "schedule_lookahead_steps": (
                self.output_rear_support_tracking_feedback_lookahead_steps
            ),
            "contact_force_threshold_n": (
                self.rollout.contact_force_threshold
            ),
            "rear_normal_n": [0.0, 0.0],
            "desired_rear_contact": [False, False],
            "measured_rear_contact": [False, False],
            "missing_rear_support": [False, False],
            "reference_tracking_error_rad": [0.0] * 12,
            "requested_correction_rad": [0.0] * 12,
            "absolute_feedback_limit_rad": [0.0] * 12,
            "applied_correction_rad": [0.0] * 12,
        }
        if not configured or not active_for_ref or not active_for_frame:
            return selected_leg, disabled_diagnostics

        rear_normal = torch.abs(
            self.adapter.contact_sensor.data.net_forces_w[
                0,
                self.adapter.contact_body_ids[2:4],
                2,
            ]
        )
        measured_rear = (
            rear_normal >= self.rollout.contact_force_threshold
        )
        schedule = self.adapter.contact_schedules[request.ref_id]
        schedule_frame = min(
            request.ref_frame
            + self.output_rear_support_tracking_feedback_lookahead_steps
            + self.adapter.action_delay_steps,
            len(schedule) - 1,
        )
        desired_rear = torch.as_tensor(
            schedule[schedule_frame, 2:4],
            dtype=torch.bool,
            device=self.adapter.base.device,
        )
        missing_rear_support = desired_rear & ~measured_rear

        reference_q = torch.as_tensor(
            self.references[request.ref_id].joint_pos[
                schedule_frame,
                :12,
            ],
            dtype=torch.float32,
            device=self.adapter.base.device,
        )
        actual_q = torch.as_tensor(
            np.asarray(request.q, dtype=np.float32)[:12],
            dtype=torch.float32,
            device=self.adapter.base.device,
        )
        tracking_error = reference_q - actual_q
        requested_correction = torch.clamp(
            self.output_rear_support_tracking_feedback_gain
            * tracking_error,
            min=-self.output_rear_support_tracking_feedback_max_abs_rad,
            max=self.output_rear_support_tracking_feedback_max_abs_rad,
        )
        feedback_mask = torch.zeros(
            12,
            dtype=torch.float32,
            device=self.adapter.base.device,
        )
        for rear_index, joint_indices in enumerate(
            ((2, 6, 10), (3, 7, 11))
        ):
            feedback_mask[list(joint_indices)] = missing_rear_support[
                rear_index
            ].float()
        requested_correction *= feedback_mask

        proposed_leg = selected_leg + requested_correction / self.scale
        feedback_limit = (
            nominal_leg + requested_correction / self.scale
        )
        positive_correction = requested_correction > 0.0
        negative_correction = requested_correction < 0.0
        proposed_leg = torch.where(
            positive_correction,
            torch.maximum(
                selected_leg,
                torch.minimum(proposed_leg, feedback_limit),
            ),
            proposed_leg,
        )
        proposed_leg = torch.where(
            negative_correction,
            torch.minimum(
                selected_leg,
                torch.maximum(proposed_leg, feedback_limit),
            ),
            proposed_leg,
        )
        lower = torch.maximum(
            self.raw_min,
            previous_action - self.max_delta,
        )
        upper = torch.minimum(
            self.raw_max,
            previous_action + self.max_delta,
        )
        corrected_leg = torch.minimum(
            torch.maximum(proposed_leg, lower),
            upper,
        )
        applied_correction = self.scale * (
            corrected_leg - selected_leg
        )
        return corrected_leg, {
            "enabled": True,
            "configured": True,
            "active_for_ref": True,
            "active_for_frame": True,
            "start_frame": (
                self.output_rear_support_tracking_feedback_start_frame
            ),
            "schedule_frame": schedule_frame,
            "schedule_lookahead_steps": (
                self.output_rear_support_tracking_feedback_lookahead_steps
            ),
            "contact_force_threshold_n": (
                self.rollout.contact_force_threshold
            ),
            "rear_normal_n": rear_normal.detach().cpu().tolist(),
            "desired_rear_contact": (
                desired_rear.detach().cpu().tolist()
            ),
            "measured_rear_contact": (
                measured_rear.detach().cpu().tolist()
            ),
            "missing_rear_support": (
                missing_rear_support.detach().cpu().tolist()
            ),
            "reference_tracking_error_rad": (
                tracking_error.detach().cpu().tolist()
            ),
            "requested_correction_rad": (
                requested_correction.detach().cpu().tolist()
            ),
            "absolute_feedback_limit_rad": (
                (self.offset + self.scale * feedback_limit)
                .detach()
                .cpu()
                .tolist()
            ),
            "applied_correction_rad": (
                applied_correction.detach().cpu().tolist()
            ),
        }

    def _apply_output_joint_position_offset(
        self,
        selected_leg: torch.Tensor,
        previous_action: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Apply a fixed physical target offset under the frozen action limits."""

        if not torch.any(self.output_joint_position_offset_leg != 0.0):
            return selected_leg, {
                "enabled": False,
                "requested_correction_rad": [0.0] * 12,
                "applied_correction_rad": [0.0] * 12,
            }
        proposed_leg = (
            selected_leg
            + self.output_joint_position_offset_leg / self.scale
        )
        lower = torch.maximum(
            self.raw_min,
            previous_action - self.max_delta,
        )
        upper = torch.minimum(
            self.raw_max,
            previous_action + self.max_delta,
        )
        corrected_leg = torch.minimum(
            torch.maximum(proposed_leg, lower),
            upper,
        )
        applied_correction = self.scale * (
            corrected_leg - selected_leg
        )
        return corrected_leg, {
            "enabled": True,
            "requested_correction_rad": (
                self.output_joint_position_offset_leg.detach().cpu().tolist()
            ),
            "applied_correction_rad": (
                applied_correction.detach().cpu().tolist()
            ),
        }

    def _apply_output_contact_orientation_feedback(
        self,
        selected_leg: torch.Tensor,
        nominal_leg: torch.Tensor,
        request: ExpertRequest,
        actual_quat_w: torch.Tensor,
        target_quat_w: torch.Tensor,
        previous_action: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Coordinate scheduled support endpoints against orientation drift.

        For a small desired base correction ``-e``, a pinned support endpoint
        needs the joint-induced world displacement ``e x r``.  Each scheduled
        support leg receives that Cartesian target through its live
        translational Jacobian.  The result remains capped relative to the
        frozen nominal action and is projected through the existing raw and
        physical target-rate limits.
        """

        configured = bool(
            torch.any(
                self.output_contact_orientation_feedback_gain_xyz != 0.0
            ).item()
        )
        active_for_ref = (
            request.ref_id
            in self.output_contact_orientation_feedback_ref_ids
        )
        started = (
            request.ref_frame
            >= self.output_contact_orientation_feedback_start_frame
        )
        disabled_diagnostics = {
            "enabled": False,
            "configured": configured,
            "active_for_ref": active_for_ref,
            "started": started,
            "start_frame": (
                self.output_contact_orientation_feedback_start_frame
            ),
            "schedule_frame": int(request.ref_frame),
            "orientation_error_target_rad": [0.0] * 3,
            "weighted_orientation_error_world_rad": [0.0] * 3,
            "desired_support": [False] * 4,
            "measured_contact": [False] * 4,
            "normal_force_n": [0.0] * 4,
            "lever_arm_world_m": [[0.0] * 3 for _ in range(4)],
            "desired_endpoint_delta_world_m": [
                [0.0] * 3 for _ in range(4)
            ],
            "predicted_endpoint_delta_world_m": [
                [0.0] * 3 for _ in range(4)
            ],
            "jacobian_joint_delta_rad": [0.0] * 12,
            "requested_correction_rad": [0.0] * 12,
            "absolute_feedback_limit_rad": [0.0] * 12,
            "applied_correction_rad": [0.0] * 12,
        }
        if not configured or not active_for_ref or not started:
            return selected_leg, disabled_diagnostics

        orientation_error_target = _quat_rotation_vector(
            actual_quat_w,
            target_quat_w,
        )
        weighted_orientation_error_world = _quat_rotate(
            target_quat_w,
            orientation_error_target
            * self.output_contact_orientation_feedback_gain_xyz,
        )
        schedule = self.adapter.contact_schedules[request.ref_id]
        schedule_frame = min(
            request.ref_frame + self.adapter.action_delay_steps,
            len(schedule) - 1,
        )
        desired_support = torch.as_tensor(
            np.asarray(schedule[schedule_frame], dtype=bool),
            dtype=torch.bool,
            device=self.adapter.base.device,
        )
        normal_force = torch.abs(
            self.adapter.contact_sensor.data.net_forces_w[
                0,
                self.adapter.contact_body_ids,
                2,
            ]
        )
        measured_contact = (
            normal_force >= self.rollout.contact_force_threshold
        )

        robot = self.rollout.robot
        wheel_positions = robot.data.body_pos_w[
            0,
            self.rollout.wheel_body_ids,
        ]
        anchor_position = self.rollout.command.robot_anchor_pos_w[0]
        lever_arms = wheel_positions - anchor_position
        desired_endpoint_delta = torch.cross(
            weighted_orientation_error_world.expand_as(lever_arms),
            lever_arms,
            dim=-1,
        )
        desired_endpoint_delta = torch.where(
            desired_support.unsqueeze(-1),
            desired_endpoint_delta,
            torch.zeros_like(desired_endpoint_delta),
        )
        endpoint_norm = torch.linalg.vector_norm(
            desired_endpoint_delta,
            dim=-1,
            keepdim=True,
        )
        endpoint_scale = torch.clamp(
            self.output_contact_orientation_feedback_max_endpoint_delta_m
            / endpoint_norm.clamp_min(
                torch.finfo(endpoint_norm.dtype).eps
            ),
            max=1.0,
        )
        desired_endpoint_delta = (
            desired_endpoint_delta * endpoint_scale
        )

        jacobians = robot.root_physx_view.get_jacobians()
        is_fixed_base = bool(getattr(robot, "is_fixed_base", False))
        joint_column_offset = 0 if is_fixed_base else 6
        damping = torch.as_tensor(
            1.0e-3,
            dtype=jacobians.dtype,
            device=jacobians.device,
        )
        joint_delta = torch.zeros(
            12,
            dtype=torch.float32,
            device=self.adapter.base.device,
        )
        predicted_endpoint_delta = torch.zeros(
            (4, 3),
            dtype=torch.float32,
            device=self.adapter.base.device,
        )
        joint_indices_by_wheel = (
            (0, 4, 8),
            (1, 5, 9),
            (2, 6, 10),
            (3, 7, 11),
        )
        for wheel_index, joint_indices in enumerate(
            joint_indices_by_wheel
        ):
            if not bool(desired_support[wheel_index].item()):
                continue
            body_id = self.rollout.wheel_body_ids[wheel_index]
            jacobian_body_id = (
                body_id - 1 if is_fixed_base else body_id
            )
            joint_columns = torch.as_tensor(
                [
                    self.rollout.joint_ids[index]
                    + joint_column_offset
                    for index in joint_indices
                ],
                dtype=torch.long,
                device=jacobians.device,
            )
            wheel_jacobian = jacobians[
                0,
                jacobian_body_id,
                :3,
            ].index_select(-1, joint_columns)
            desired_delta = desired_endpoint_delta[
                wheel_index
            ].to(
                dtype=jacobians.dtype,
                device=jacobians.device,
            )
            normal_matrix = (
                wheel_jacobian.transpose(0, 1) @ wheel_jacobian
                + damping
                * torch.eye(
                    3,
                    dtype=jacobians.dtype,
                    device=jacobians.device,
                )
            )
            wheel_joint_delta = torch.linalg.solve(
                normal_matrix,
                wheel_jacobian.transpose(0, 1) @ desired_delta,
            )
            wheel_joint_delta = torch.clamp(
                wheel_joint_delta,
                min=(
                    -self.output_contact_orientation_feedback_max_abs_rad
                ),
                max=(
                    self.output_contact_orientation_feedback_max_abs_rad
                ),
            )
            predicted_delta = wheel_jacobian @ wheel_joint_delta
            desired_norm = torch.linalg.vector_norm(desired_delta)
            predicted_norm = torch.linalg.vector_norm(predicted_delta)
            if (
                predicted_norm > desired_norm
                and predicted_norm > 0.0
            ):
                wheel_joint_delta = wheel_joint_delta * (
                    desired_norm / predicted_norm
                )
                predicted_delta = wheel_jacobian @ wheel_joint_delta
            joint_index_tensor = torch.as_tensor(
                joint_indices,
                dtype=torch.long,
                device=self.adapter.base.device,
            )
            joint_delta[joint_index_tensor] = wheel_joint_delta.to(
                dtype=torch.float32,
                device=self.adapter.base.device,
            )
            predicted_endpoint_delta[wheel_index] = (
                predicted_delta.to(
                    dtype=torch.float32,
                    device=self.adapter.base.device,
                )
            )

        proposed_leg = selected_leg + joint_delta / self.scale
        feedback_limit = nominal_leg + joint_delta / self.scale
        positive_correction = joint_delta > 0.0
        negative_correction = joint_delta < 0.0
        proposed_leg = torch.where(
            positive_correction,
            torch.maximum(
                selected_leg,
                torch.minimum(proposed_leg, feedback_limit),
            ),
            proposed_leg,
        )
        proposed_leg = torch.where(
            negative_correction,
            torch.minimum(
                selected_leg,
                torch.maximum(proposed_leg, feedback_limit),
            ),
            proposed_leg,
        )
        lower = torch.maximum(
            self.raw_min,
            previous_action - self.max_delta,
        )
        upper = torch.minimum(
            self.raw_max,
            previous_action + self.max_delta,
        )
        corrected_leg = torch.minimum(
            torch.maximum(proposed_leg, lower),
            upper,
        )
        applied_correction = self.scale * (
            corrected_leg - selected_leg
        )
        return corrected_leg, {
            "enabled": True,
            "configured": True,
            "active_for_ref": True,
            "started": True,
            "start_frame": (
                self.output_contact_orientation_feedback_start_frame
            ),
            "schedule_frame": schedule_frame,
            "orientation_error_target_rad": (
                orientation_error_target.detach().cpu().tolist()
            ),
            "weighted_orientation_error_world_rad": (
                weighted_orientation_error_world.detach().cpu().tolist()
            ),
            "desired_support": (
                desired_support.detach().cpu().tolist()
            ),
            "measured_contact": (
                measured_contact.detach().cpu().tolist()
            ),
            "normal_force_n": normal_force.detach().cpu().tolist(),
            "lever_arm_world_m": lever_arms.detach().cpu().tolist(),
            "desired_endpoint_delta_world_m": (
                desired_endpoint_delta.detach().cpu().tolist()
            ),
            "predicted_endpoint_delta_world_m": (
                predicted_endpoint_delta.detach().cpu().tolist()
            ),
            "jacobian_joint_delta_rad": (
                joint_delta.detach().cpu().tolist()
            ),
            "requested_correction_rad": (
                joint_delta.detach().cpu().tolist()
            ),
            "absolute_feedback_limit_rad": (
                (self.offset + self.scale * feedback_limit)
                .detach()
                .cpu()
                .tolist()
            ),
            "applied_correction_rad": (
                applied_correction.detach().cpu().tolist()
            ),
        }

    def _apply_output_pitch_feedback(
        self,
        selected_leg: torch.Tensor,
        nominal_leg: torch.Tensor,
        ref_id: int,
        actual_quat_w: torch.Tensor,
        target_quat_w: torch.Tensor,
        previous_action: torch.Tensor,
        ref_frame: int = 0,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Apply bounded state feedback for one signed orientation axis.

        The correction is expressed in physical joint-position radians and is
        capped relative to this solve's nominal action.  It therefore cannot
        integrate through the previous-action rate constraint.  Raw bounds and
        the same per-step physical target-rate limit used by MPPI are applied
        again before returning.
        """

        configured = bool(
            torch.any(self.output_pitch_feedback_gain_leg != 0.0).item()
        )
        active_for_ref = ref_id in self.output_pitch_feedback_ref_ids
        started = ref_frame >= self.output_pitch_feedback_start_frame
        if not configured or not active_for_ref or not started:
            return selected_leg, {
                "enabled": False,
                "configured": configured,
                "active_for_ref": active_for_ref,
                "started": started,
                "feedback_axis": self.output_pitch_feedback_axis,
                "signed_orientation_axis_error_rad": 0.0,
                "signed_pitch_error_rad": 0.0,
                "requested_correction_rad": [0.0] * 12,
                "absolute_feedback_limit_rad": [0.0] * 12,
                "applied_correction_rad": [0.0] * 12,
            }

        signed_orientation_axis_error = (
            _signed_orientation_axis_error_rad(
                actual_quat_w,
                target_quat_w,
                self.output_pitch_feedback_axis_index,
            )
        )
        signed_pitch_error = (
            signed_orientation_axis_error
            if self.output_pitch_feedback_axis == "y"
            else torch.zeros_like(signed_orientation_axis_error)
        )
        requested_correction = torch.clamp(
            signed_orientation_axis_error
            * self.output_pitch_feedback_gain_leg,
            min=-self.output_pitch_feedback_max_abs_rad,
            max=self.output_pitch_feedback_max_abs_rad,
        )
        proposed_leg = selected_leg + requested_correction / self.scale
        feedback_limit = nominal_leg + requested_correction / self.scale
        positive_correction = requested_correction > 0.0
        negative_correction = requested_correction < 0.0
        proposed_leg = torch.where(
            positive_correction,
            torch.maximum(
                selected_leg,
                torch.minimum(proposed_leg, feedback_limit),
            ),
            proposed_leg,
        )
        proposed_leg = torch.where(
            negative_correction,
            torch.minimum(
                selected_leg,
                torch.maximum(proposed_leg, feedback_limit),
            ),
            proposed_leg,
        )
        lower = torch.maximum(
            self.raw_min,
            previous_action - self.max_delta,
        )
        upper = torch.minimum(
            self.raw_max,
            previous_action + self.max_delta,
        )
        corrected_leg = torch.minimum(
            torch.maximum(proposed_leg, lower),
            upper,
        )
        applied_correction = self.scale * (
            corrected_leg - selected_leg
        )
        return corrected_leg, {
            "enabled": True,
            "configured": True,
            "active_for_ref": True,
            "started": True,
            "feedback_axis": self.output_pitch_feedback_axis,
            "signed_orientation_axis_error_rad": float(
                signed_orientation_axis_error.detach().cpu().item()
            ),
            "signed_pitch_error_rad": float(
                signed_pitch_error.detach().cpu().item()
            ),
            "max_abs_correction_rad": (
                self.output_pitch_feedback_max_abs_rad
            ),
            "requested_correction_rad": (
                requested_correction.detach().cpu().tolist()
            ),
            "absolute_feedback_limit_rad": (
                (self.offset + self.scale * feedback_limit)
                .detach()
                .cpu()
                .tolist()
            ),
            "applied_correction_rad": (
                applied_correction.detach().cpu().tolist()
            ),
        }

    def __call__(self, request: ExpertRequest) -> ExpertReply:
        start = time.perf_counter()
        snapshot: IsaacRolloutSnapshot | None = None
        try:
            request.validate()
            nominal = self._nominal(request)
            (
                rear_swing_reference_proposal_offsets,
                rear_swing_reference_proposal_diagnostics,
            ) = self._rear_swing_reference_proposals(
                request,
                nominal,
            )
            (
                rear_swing_tracking_error_proposal_offsets,
                rear_swing_tracking_error_proposal_diagnostics,
            ) = self._rear_swing_tracking_error_proposals(
                request,
                nominal,
            )
            (
                rear_swing_load_transfer_proposal_offsets,
                rear_swing_load_transfer_proposal_diagnostics,
            ) = self._rear_swing_load_transfer_proposals(
                request,
                nominal,
            )
            (
                front_support_proposal_offsets,
                front_support_proposal_diagnostics,
            ) = self._front_support_proposals(
                request,
                nominal,
            )
            (
                coordinated_proposal_offsets,
                coordinated_proposal_diagnostics,
            ) = self._coordinated_rear_swing_front_support_proposals(
                request,
                nominal,
                rear_swing_reference_proposal_offsets,
                front_support_proposal_offsets,
            )
            (
                coordinated_load_transfer_proposal_offsets,
                coordinated_load_transfer_proposal_diagnostics,
            ) = (
                self._coordinated_rear_swing_load_transfer_front_support_proposals(
                    request,
                    nominal,
                    rear_swing_load_transfer_proposal_offsets,
                    front_support_proposal_offsets,
                )
            )
            (
                coordinated_reference_load_transfer_proposal_offsets,
                coordinated_reference_load_transfer_proposal_diagnostics,
            ) = (
                self._coordinated_rear_swing_reference_load_transfer_front_support_proposals(
                    request,
                    nominal,
                    rear_swing_reference_proposal_offsets,
                    rear_swing_load_transfer_proposal_offsets,
                    front_support_proposal_offsets,
                )
            )
            structured_offset_groups = [
                offsets
                for offsets in (
                    rear_swing_reference_proposal_offsets,
                    rear_swing_tracking_error_proposal_offsets,
                    rear_swing_load_transfer_proposal_offsets,
                    front_support_proposal_offsets,
                    coordinated_proposal_offsets,
                    coordinated_load_transfer_proposal_offsets,
                    coordinated_reference_load_transfer_proposal_offsets,
                )
                if offsets is not None
            ]
            structured_proposal_offsets = (
                torch.cat(structured_offset_groups, dim=0)
                if structured_offset_groups
                else None
            )
            structured_proposal_descriptors = [
                {
                    "kind": "rear_swing_reference",
                    "scale": float(scale),
                }
                for scale in (
                    rear_swing_reference_proposal_diagnostics.get(
                        "scales",
                        [],
                    )
                )
            ] + [
                {
                    "kind": "rear_swing_tracking_error",
                    "scale": float(scale),
                }
                for scale in (
                    rear_swing_tracking_error_proposal_diagnostics.get(
                        "scales",
                        [],
                    )
                )
            ] + [
                {
                    "kind": "rear_swing_load_transfer",
                    "scale": float(scale),
                }
                for scale in (
                    rear_swing_load_transfer_proposal_diagnostics.get(
                        "scales",
                        [],
                    )
                )
            ] + [
                {
                    "kind": "front_support",
                    "scale": float(scale),
                }
                for scale in front_support_proposal_diagnostics.get(
                    "scales",
                    [],
                )
            ] + [
                {
                    "kind": "coordinated_rear_swing_front_support",
                    **scale_pair,
                }
                for scale_pair in coordinated_proposal_diagnostics.get(
                    "scale_pairs",
                    [],
                )
            ] + [
                {
                    "kind": (
                        "coordinated_rear_swing_load_transfer_front_support"
                    ),
                    **scale_pair,
                }
                for scale_pair
                in coordinated_load_transfer_proposal_diagnostics.get(
                    "scale_pairs",
                    [],
                )
            ] + [
                {
                    "kind": (
                        "coordinated_rear_swing_reference_load_transfer_"
                        "front_support"
                    ),
                    **scale_triple,
                }
                for scale_triple
                in coordinated_reference_load_transfer_proposal_diagnostics.get(
                    "scale_triples",
                    [],
                )
            ]
            previous_action = self.adapter.previous_commanded_action[0, :12].clone()
            initial = nominal
            configured_reference_overrides = (
                self.nominal_action_reference_overrides_by_ref.get(
                    request.ref_id,
                    {},
                )
            )
            reference_overrides, solver_schedule_phase = (
                resolve_nominal_solver_overrides(
                    configured_reference_overrides,
                    request.ref_frame,
                )
            )
            solver_schedule_start_frame: int | None = None
            solver_schedule_reset_warm_start = False
            if solver_schedule_phase is not None:
                phase = configured_reference_overrides[
                    "solver_schedule"
                ][solver_schedule_phase]
                solver_schedule_start_frame = int(phase["start_frame"])
                previous_phase = (
                    self._active_solver_schedule_phase_by_ref.get(
                        request.ref_id
                    )
                )
                phase_changed = previous_phase != solver_schedule_phase
                solver_schedule_reset_warm_start = bool(
                    phase_changed
                    and phase.get("reset_warm_start", False)
                )
                if solver_schedule_reset_warm_start:
                    self._warm_residual = None
                self._active_solver_schedule_phase_by_ref[
                    request.ref_id
                ] = solver_schedule_phase
            effective_warm_start = bool(
                reference_overrides.get(
                    "warm_start",
                    self.config.warm_start,
                )
            )
            effective_selection_mode = str(
                reference_overrides.get(
                    "selection_mode",
                    self.config.selection_mode,
                )
            )
            effective_temperature = float(
                reference_overrides.get(
                    "temperature",
                    self.config.temperature,
                )
            )
            action_residual_weight = reference_overrides.get(
                "action_residual_weight"
            )
            base_orientation_cost_multiplier = float(
                reference_overrides.get(
                    "base_orientation_cost_multiplier",
                    1.0,
                )
            )
            lateral_velocity_cost_multiplier = float(
                reference_overrides.get(
                    "lateral_velocity_cost_multiplier",
                    1.0,
                )
            )
            rear_support_loss_cost_multiplier = float(
                reference_overrides.get(
                    "rear_support_loss_cost_multiplier",
                    1.0,
                )
            )
            base_orientation_axis_multipliers = tuple(
                float(value)
                for value in reference_overrides.get(
                    "base_orientation_axis_multipliers",
                    (1.0, 1.0, 1.0),
                )
            )
            if effective_warm_start and self._warm_residual is not None:
                shifted = torch.cat(
                    (self._warm_residual[1:], self._warm_residual[-1:]),
                    dim=0,
                )
                initial = nominal + shifted

            snapshot = self.rollout.capture()
            best_component_cost = float("inf")
            best_cost_components: dict[str, float] = {}
            structured_proposal_cost_iterations: list[
                dict[str, Any]
            ] = []

            def rollout_cost(candidates: torch.Tensor) -> torch.Tensor:
                nonlocal best_component_cost, best_cost_components
                costs = self.rollout.evaluate(
                    candidates,
                    snapshot,
                    nominal,
                    action_residual_weight=action_residual_weight,
                    rear_swing_action_residual_lead_steps=(
                        self.rear_swing_action_residual_lead_steps
                    ),
                    base_orientation_cost_multiplier=(
                        base_orientation_cost_multiplier
                    ),
                    lateral_velocity_cost_multiplier=(
                        lateral_velocity_cost_multiplier
                    ),
                    rear_support_loss_cost_multiplier=(
                        rear_support_loss_cost_multiplier
                    ),
                    base_orientation_axis_multipliers=(
                        base_orientation_axis_multipliers
                    ),
                )
                if structured_proposal_offsets is not None:
                    proposal_cost_record = (
                        structured_candidate_cost_diagnostics(
                            costs,
                            self.rollout.last_components,
                            int(
                                structured_proposal_offsets.shape[
                                    0
                                ]
                            ),
                        )
                    )
                    proposal_cost_record["iteration"] = len(
                        structured_proposal_cost_iterations
                    )
                    structured_proposal_cost_iterations.append(
                        proposal_cost_record
                    )
                best_component_cost, best_cost_components = (
                    select_global_best_cost_components(
                        costs,
                        self.rollout.last_components,
                        best_component_cost,
                        best_cost_components,
                    )
                )
                return costs

            sequence, optimizer_diagnostics = self.optimizer.optimize(
                nominal,
                rollout_cost,
                self.raw_min,
                self.raw_max,
                previous_action=previous_action,
                max_delta=self.max_delta,
                initial_sequence=initial,
                proposal_offsets=structured_proposal_offsets,
                selection_mode=effective_selection_mode,
                temperature=effective_temperature,
            )
            self.rollout.restore(snapshot)
            missing_cost_components = sorted(
                set(MPPI_COST_COMPONENT_NAMES) - set(best_cost_components)
            )
            if missing_cost_components:
                raise RuntimeError(
                    "MPPI best-candidate diagnostics are missing cost "
                    f"components: {missing_cost_components}"
                )
            self.rollout.last_best_components = dict(
                best_cost_components
            )
            self._warm_residual = (sequence - nominal).detach().clone()
            selected_leg = sequence[0]
            selected_leg, output_force_feedback = (
                self._apply_output_front_force_feedback(
                    selected_leg,
                    nominal[0],
                    request,
                    previous_action,
                )
            )
            selected_leg, output_rear_swing_force_feedback = (
                self._apply_output_rear_swing_force_feedback(
                    selected_leg,
                    nominal[0],
                    request,
                    previous_action,
                )
            )
            selected_leg, output_rear_swing_height_feedback = (
                self._apply_output_rear_swing_height_feedback(
                    selected_leg,
                    nominal[0],
                    request,
                    previous_action,
                    snapshot,
                )
            )
            selected_leg, output_rear_support_tracking_feedback = (
                self._apply_output_rear_support_tracking_feedback(
                    selected_leg,
                    nominal[0],
                    request,
                    previous_action,
                )
            )
            aligned_target = self.rollout._aligned_reference(
                snapshot,
                request.ref_frame,
            )
            selected_leg, output_contact_orientation_feedback = (
                self._apply_output_contact_orientation_feedback(
                    selected_leg,
                    nominal[0],
                    request,
                    self.rollout.command.robot_anchor_quat_w[0],
                    aligned_target["body_quat"][
                        self.rollout.ref_anchor_body_id
                    ],
                    previous_action,
                )
            )
            selected_leg, output_pitch_feedback = (
                self._apply_output_pitch_feedback(
                    selected_leg,
                    nominal[0],
                    request.ref_id,
                    self.rollout.command.robot_anchor_quat_w[0],
                    aligned_target["body_quat"][
                        self.rollout.ref_anchor_body_id
                    ],
                    previous_action,
                    ref_frame=request.ref_frame,
                )
            )
            selected_leg, output_joint_offset = (
                self._apply_output_joint_position_offset(
                    selected_leg,
                    previous_action,
                )
            )
            action16_t = torch.cat(
                (selected_leg, torch.zeros(4, device=selected_leg.device)),
                dim=0,
            )
            if not torch.isfinite(action16_t).all():
                raise FloatingPointError("MPPI selected a non-finite action.")
            action16 = action16_t.detach().cpu().numpy().astype(np.float32)
            q_des_leg = (
                self.offset + self.scale * selected_leg
            ).detach().cpu().numpy().astype(np.float32)
            lower_margin = selected_leg - self.raw_min
            upper_margin = self.raw_max - selected_leg
            safety_margin = float(torch.minimum(lower_margin, upper_margin).min().item())
            solve_ms = (time.perf_counter() - start) * 1000.0
            self.last_diagnostics = {
                **optimizer_diagnostics,
                "solve_ms": solve_ms,
                "best_cost_components": dict(self.rollout.last_best_components),
                "rollout_termination_rate": self.rollout.last_termination_rate,
                "ref_id": request.ref_id,
                "ref_frame": request.ref_frame,
                "output_front_force_feedback": output_force_feedback,
                "output_rear_swing_force_feedback": (
                    output_rear_swing_force_feedback
                ),
                "output_rear_swing_height_feedback": (
                    output_rear_swing_height_feedback
                ),
                "output_rear_support_tracking_feedback": (
                    output_rear_support_tracking_feedback
                ),
                "rear_swing_reference_proposals": (
                    rear_swing_reference_proposal_diagnostics
                ),
                "rear_swing_tracking_error_proposals": (
                    rear_swing_tracking_error_proposal_diagnostics
                ),
                "rear_swing_load_transfer_proposals": (
                    rear_swing_load_transfer_proposal_diagnostics
                ),
                "front_support_proposals": (
                    front_support_proposal_diagnostics
                ),
                "coordinated_rear_swing_front_support_proposals": (
                    coordinated_proposal_diagnostics
                ),
                "coordinated_rear_swing_load_transfer_front_support_proposals": (
                    coordinated_load_transfer_proposal_diagnostics
                ),
                "coordinated_rear_swing_reference_load_transfer_front_support_proposals": (
                    coordinated_reference_load_transfer_proposal_diagnostics
                ),
                "structured_proposal_descriptors": (
                    structured_proposal_descriptors
                ),
                "structured_proposal_cost_iterations": (
                    structured_proposal_cost_iterations
                ),
                "output_pitch_feedback": output_pitch_feedback,
                "output_contact_orientation_feedback": (
                    output_contact_orientation_feedback
                ),
                "output_joint_position_offset": output_joint_offset,
                "effective_warm_start": effective_warm_start,
                "effective_selection_mode": effective_selection_mode,
                "effective_temperature": effective_temperature,
                "effective_action_residual_weight": (
                    self.rollout.cost_weights.action_residual
                    if action_residual_weight is None
                    else float(action_residual_weight)
                ),
                "rear_swing_action_residual_lead_steps": (
                    self.rear_swing_action_residual_lead_steps
                ),
                "effective_base_orientation_cost_multiplier": (
                    base_orientation_cost_multiplier
                ),
                "effective_lateral_velocity_cost_multiplier": (
                    lateral_velocity_cost_multiplier
                ),
                "effective_rear_support_loss_cost_multiplier": (
                    rear_support_loss_cost_multiplier
                ),
                "effective_base_orientation_axis_multipliers": list(
                    base_orientation_axis_multipliers
                ),
                "solver_schedule_phase": solver_schedule_phase,
                "solver_schedule_start_frame": (
                    solver_schedule_start_frame
                ),
                "solver_schedule_reset_warm_start": (
                    solver_schedule_reset_warm_start
                ),
            }
            status = "MPPI_VALID"
            reply_diagnostics = {
                "minimum_total_cost": optimizer_diagnostics["minimum_total_cost"],
                "mean_total_cost": optimizer_diagnostics["mean_total_cost"],
                "effective_sample_size": optimizer_diagnostics["effective_sample_size"],
                "rollout_termination_rate": self.rollout.last_termination_rate,
                "cost_components": [
                    float(self.rollout.last_best_components.get(name, np.nan))
                    for name in MPPI_COST_COMPONENT_NAMES
                ],
            }
            return ExpertReply(
                valid=True,
                q_des_leg=q_des_leg,
                wheel_vel_des=np.zeros(4, dtype=np.float32),
                action16=action16,
                tau_ff_leg=None,
                predicted_grf=None,
                solve_ms=solve_ms,
                solver_status=status,
                safety_margin=safety_margin,
                source="mppi",
                failure_code=FailureCode.NONE,
                diagnostics=reply_diagnostics,
            )
        except BaseException as exc:
            if snapshot is not None:
                try:
                    self.rollout.restore(snapshot)
                except BaseException as restore_exc:
                    self.last_diagnostics = {
                        "error": repr(exc),
                        "restore_error": repr(restore_exc),
                    }
                    failure_code = FailureCode.STATE_COPY_ERROR
                else:
                    self.last_diagnostics = {"error": repr(exc)}
                    failure_code = FailureCode.TEACHER_INFEASIBLE
            else:
                self.last_diagnostics = {"error": repr(exc)}
                failure_code = FailureCode.TEACHER_INFEASIBLE
            return ExpertReply(
                valid=False,
                q_des_leg=np.full(12, np.nan, dtype=np.float32),
                wheel_vel_des=np.zeros(4, dtype=np.float32),
                action16=np.full(16, np.nan, dtype=np.float32),
                tau_ff_leg=None,
                predicted_grf=None,
                solve_ms=(time.perf_counter() - start) * 1000.0,
                solver_status=f"MPPI_ERROR:{type(exc).__name__}:{exc}",
                safety_margin=float("nan"),
                source="mppi",
                failure_code=failure_code,
                diagnostics={"error": repr(exc)},
            )
