from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SafetyMetrics:
    finite: bool
    base_tilt_rad: float
    minimum_joint_margin_rad: float
    wheel_slip_speed_mps: float
    minimum_edge_margin_m: float
    safe: bool
    failure_code: str


def _quat_to_roll_pitch_wxyz(quat: np.ndarray) -> tuple[float, float]:
    w, x, y, z = quat / np.linalg.norm(quat)
    sin_roll = 2.0 * (w * x + y * z)
    cos_roll = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sin_roll, cos_roll)
    sin_pitch = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(sin_pitch)
    return float(roll), float(pitch)


def compute_safety_metrics(
    base_pose_w: np.ndarray,
    q_leg: np.ndarray,
    joint_lower: np.ndarray,
    joint_upper: np.ndarray,
    wheel_linear_velocity_w: np.ndarray,
    measured_contact: np.ndarray,
    minimum_edge_margin_m: float,
    maximum_tilt_rad: float = 0.8,
    maximum_wheel_slip_mps: float = 1.0,
) -> SafetyMetrics:
    values = [
        np.asarray(base_pose_w),
        np.asarray(q_leg),
        np.asarray(joint_lower),
        np.asarray(joint_upper),
        np.asarray(wheel_linear_velocity_w),
    ]
    finite = all(np.isfinite(value).all() for value in values)
    if not finite:
        return SafetyMetrics(False, np.nan, np.nan, np.nan, minimum_edge_margin_m, False, "NAN_INF")
    roll, pitch = _quat_to_roll_pitch_wxyz(np.asarray(base_pose_w)[3:7])
    tilt = float(max(abs(roll), abs(pitch)))
    margin = float(
        np.min(
            np.minimum(
                np.asarray(q_leg) - np.asarray(joint_lower),
                np.asarray(joint_upper) - np.asarray(q_leg),
            )
        )
    )
    contact = np.asarray(measured_contact, dtype=bool)
    planar_speed = np.linalg.norm(np.asarray(wheel_linear_velocity_w)[..., :2], axis=-1)
    slip = float(np.max(np.where(contact, planar_speed, 0.0)))
    if margin < 0.0:
        failure = "JOINT_LIMIT"
    elif minimum_edge_margin_m < 0.0:
        failure = "EDGE_MARGIN"
    elif tilt > maximum_tilt_rad:
        failure = "BASE_ORIENTATION"
    elif slip > maximum_wheel_slip_mps:
        failure = "WHEEL_SLIP"
    else:
        failure = ""
    return SafetyMetrics(finite, tilt, margin, slip, minimum_edge_margin_m, failure == "", failure)
