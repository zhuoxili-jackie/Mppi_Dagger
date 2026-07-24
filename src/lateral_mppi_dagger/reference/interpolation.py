from __future__ import annotations

import numpy as np

from .loader import ReferenceMotion


def _slerp_wxyz(first: np.ndarray, second: np.ndarray, alpha: float) -> np.ndarray:
    first = first / np.linalg.norm(first, axis=-1, keepdims=True)
    second = second / np.linalg.norm(second, axis=-1, keepdims=True)
    dot = np.sum(first * second, axis=-1, keepdims=True)
    second = np.where(dot < 0.0, -second, second)
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    near = dot > 0.9995
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    safe_denominator = np.where(np.abs(sin_theta) < 1.0e-8, 1.0, sin_theta)
    slerped = (
        np.sin((1.0 - alpha) * theta) / safe_denominator * first
        + np.sin(alpha * theta) / safe_denominator * second
    )
    linear = (1.0 - alpha) * first + alpha * second
    result = np.where(near, linear, slerped)
    return (result / np.linalg.norm(result, axis=-1, keepdims=True)).astype(np.float32)


def interpolate_reference(motion: ReferenceMotion, time_seconds: float) -> dict[str, np.ndarray | float | int]:
    frame_position = np.clip(time_seconds * motion.fps, 0.0, motion.frames - 1)
    lower = int(np.floor(frame_position))
    upper = min(lower + 1, motion.frames - 1)
    alpha = float(frame_position - lower)

    def linear(field: str) -> np.ndarray:
        values = getattr(motion, field)
        return ((1.0 - alpha) * values[lower] + alpha * values[upper]).astype(np.float32)

    return {
        "ref_id": motion.index,
        "ref_frame": lower,
        "frame_position": float(frame_position),
        "phase": float(frame_position / max(motion.frames - 1, 1)),
        "target_vy": motion.target_vy,
        "joint_pos": linear("joint_pos"),
        "joint_vel": linear("joint_vel"),
        "body_pos_w": linear("body_pos_w"),
        "body_quat_w": _slerp_wxyz(motion.body_quat_w[lower], motion.body_quat_w[upper], alpha),
        "body_lin_vel_w": linear("body_lin_vel_w"),
        "body_ang_vel_w": linear("body_ang_vel_w"),
    }


def assert_compatible_timebase(control_dt: float, reference_fps: int, tolerance: float = 1.0e-9) -> None:
    reference_dt = 1.0 / reference_fps
    if abs(control_dt - reference_dt) > tolerance:
        raise ValueError(
            f"REFERENCE_TIMEBASE_ERROR: control dt={control_dt:.12f}s does not match "
            f"reference dt={reference_dt:.12f}s; configure an explicit tested resampler."
        )

