from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np


class FailureCode(IntEnum):
    NONE = 0
    TEACHER_INFEASIBLE = 1
    TEACHER_TIMEOUT = 2
    CONTACT_LOSS = 3
    WHEEL_SLIP = 4
    EDGE_MARGIN = 5
    BASE_ORIENTATION = 6
    JOINT_LIMIT = 7
    TORQUE_SATURATION = 8
    STUDENT_OOD = 9
    NAN_INF = 10
    STATE_COPY_ERROR = 11
    REFERENCE_TIMEBASE_ERROR = 12
    ASSET_CONTRACT_ERROR = 13


class LabelSource(IntEnum):
    INVALID = 0
    REFERENCE_WBC = 1
    DWMPC = 2
    MPPI = 3
    SAFETY_FALLBACK = 4


LEGACY_MPPI_COST_COMPONENT_NAMES = (
    "base_position",
    "base_orientation",
    "joint_position",
    "joint_velocity",
    "wheel_position",
    "lateral_velocity",
    "box_x_drift",
    "wheel_slip",
    "contact_mismatch",
    "edge_drop",
    "action_residual",
    "action_rate",
    "joint_acceleration",
    "torque_limit",
    "joint_limit",
    "termination",
    "terminal",
)

LOAD_SUPPORT_MPPI_COST_COMPONENT_NAMES = (
    *LEGACY_MPPI_COST_COMPONENT_NAMES[:-2],
    "front_normal_support",
    "rear_force_overload",
    "rear_force_imbalance",
    "rear_support_loss",
    *LEGACY_MPPI_COST_COMPONENT_NAMES[-2:],
)

OBSERVABILITY_MPPI_COST_COMPONENT_NAMES = (
    LOAD_SUPPORT_MPPI_COST_COMPONENT_NAMES[0],
    "base_height_drop",
    *LOAD_SUPPORT_MPPI_COST_COMPONENT_NAMES[1:3],
    "rear_leg_position",
    *LOAD_SUPPORT_MPPI_COST_COMPONENT_NAMES[3:6],
    "lateral_position",
    *LOAD_SUPPORT_MPPI_COST_COMPONENT_NAMES[6:],
)

REAR_SWING_MPPI_COST_COMPONENT_NAMES = (
    *OBSERVABILITY_MPPI_COST_COMPONENT_NAMES[:7],
    "rear_swing_lateral_position",
    *OBSERVABILITY_MPPI_COST_COMPONENT_NAMES[7:-2],
    "rear_swing_force",
    *OBSERVABILITY_MPPI_COST_COMPONENT_NAMES[-2:],
)

MPPI_COST_COMPONENT_NAMES = (
    *REAR_SWING_MPPI_COST_COMPONENT_NAMES[:8],
    "rear_swing_height_deficit",
    *REAR_SWING_MPPI_COST_COMPONENT_NAMES[8:],
)


@dataclass(frozen=True)
class ExpertRequest:
    dt: float
    base_pose_w: np.ndarray
    base_twist_w: np.ndarray
    q: np.ndarray
    dq: np.ndarray
    wheel_body_pose_w: np.ndarray
    wheel_body_twist_w: np.ndarray
    contact_force_w: np.ndarray
    ref_id: int
    ref_frame: int
    ref_window: dict[str, Any]
    target_vy: float
    desired_contact: np.ndarray
    platform_geometry: Any

    def validate(self) -> None:
        shapes = {
            "base_pose_w": (7,),
            "base_twist_w": (6,),
            "q": (16,),
            "dq": (16,),
            "wheel_body_pose_w": (4, 7),
            "wheel_body_twist_w": (4, 6),
            "contact_force_w": (4, 3),
            "desired_contact": (4,),
        }
        for name, shape in shapes.items():
            value = np.asarray(getattr(self, name))
            if value.shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
            if name != "desired_contact" and not np.isfinite(value).all():
                raise ValueError(f"{name} contains NaN or Inf")
        if self.dt <= 0.0 or not np.isfinite(self.dt):
            raise ValueError(f"dt must be finite and positive, got {self.dt}")


@dataclass(frozen=True)
class ExpertReply:
    valid: bool
    q_des_leg: np.ndarray
    wheel_vel_des: np.ndarray | None
    action16: np.ndarray
    tau_ff_leg: np.ndarray | None
    predicted_grf: np.ndarray | None
    solve_ms: float
    solver_status: str
    safety_margin: float
    source: str
    failure_code: FailureCode
    diagnostics: dict[str, Any] | None = None

    def validate(self) -> None:
        if np.asarray(self.q_des_leg).shape != (12,):
            raise ValueError(f"q_des_leg must have shape (12,), got {np.asarray(self.q_des_leg).shape}")
        if np.asarray(self.action16).shape != (16,):
            raise ValueError(f"action16 must have shape (16,), got {np.asarray(self.action16).shape}")
        if self.wheel_vel_des is not None and np.asarray(self.wheel_vel_des).shape != (4,):
            raise ValueError("wheel_vel_des must have shape (4,) or be None")
        if self.valid and (
            not np.isfinite(self.q_des_leg).all()
            or not np.isfinite(self.action16).all()
            or not np.isfinite(self.solve_ms)
            or not np.isfinite(self.safety_margin)
        ):
            raise ValueError("A valid ExpertReply contains NaN or Inf")


class Expert(ABC):
    @abstractmethod
    def reset(self, episode_metadata: dict[str, Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    def act(self, request: ExpertRequest) -> ExpertReply:
        raise NotImplementedError


class BackendUnavailable(RuntimeError):
    pass
