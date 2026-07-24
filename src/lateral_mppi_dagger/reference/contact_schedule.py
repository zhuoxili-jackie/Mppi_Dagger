from __future__ import annotations

import numpy as np

from .loader import ReferenceMotion


def _remove_short_runs(mask: np.ndarray, minimum_run: int) -> np.ndarray:
    result = mask.copy()
    for wheel in range(result.shape[1]):
        values = result[:, wheel]
        transitions = np.diff(np.pad(values.astype(np.int8), (1, 1)))
        starts = np.flatnonzero(transitions == 1)
        stops = np.flatnonzero(transitions == -1)
        for start, stop in zip(starts, stops, strict=True):
            if stop - start < minimum_run:
                result[start:stop, wheel] = False
    return result


def infer_contact_schedule(
    motion: ReferenceMotion,
    wheel_body_indices: tuple[int, int, int, int] = (13, 14, 15, 16),
    per_wheel_height_quantile: float = 0.35,
    height_margin_m: float | tuple[float, float, float, float] = 0.018,
    speed_threshold_mps: float = 0.30,
    minimum_contact_run_frames: int = 2,
    contact_axis_indices: tuple[int, int, int, int] = (0, 0, 2, 2),
    contact_surface_sides: tuple[str, str, str, str] = (
        "max",
        "max",
        "min",
        "min",
    ),
) -> np.ndarray:
    """Infer a candidate schedule along each wheel's actual contact normal.

    In the lateral trunk task the front wheels press against a near-vertical
    surface along world x, while the rear wheels rest on the ground along
    world z.  Treating all four wheels as world-z contacts hides front lift.
    The result remains a candidate until compared with Isaac force traces.
    """
    positions = motion.body_pos_w[:, wheel_body_indices, :]
    velocities = motion.body_lin_vel_w[:, wheel_body_indices, :]
    axes = np.asarray(contact_axis_indices, dtype=np.int64)
    if axes.shape != (4,) or np.any((axes < 0) | (axes > 2)):
        raise ValueError("contact_axis_indices must contain four xyz indices.")
    if len(contact_surface_sides) != 4 or any(
        side not in {"min", "max"} for side in contact_surface_sides
    ):
        raise ValueError("contact_surface_sides must contain four min/max values.")
    margins = np.broadcast_to(
        np.asarray(height_margin_m, dtype=np.float64),
        (4,),
    )
    if np.any(margins < 0.0):
        raise ValueError("height_margin_m must be non-negative.")
    coordinate = np.take_along_axis(
        positions,
        axes[None, :, None],
        axis=-1,
    )[..., 0]
    contact_coordinate_ok = np.zeros_like(coordinate, dtype=bool)
    for wheel, side in enumerate(contact_surface_sides):
        quantile = (
            per_wheel_height_quantile
            if side == "min"
            else 1.0 - per_wheel_height_quantile
        )
        support_coordinate = np.quantile(coordinate[:, wheel], quantile)
        if side == "min":
            contact_coordinate_ok[:, wheel] = (
                coordinate[:, wheel] <= support_coordinate + margins[wheel]
            )
        else:
            contact_coordinate_ok[:, wheel] = (
                coordinate[:, wheel] >= support_coordinate - margins[wheel]
            )
    speed_ok = np.linalg.norm(velocities, axis=-1) <= speed_threshold_mps
    candidate = contact_coordinate_ok & speed_ok
    return _remove_short_runs(candidate, minimum_contact_run_frames).astype(np.uint8)


def validate_contact_schedule(
    desired_contact: np.ndarray,
    contact_force_w: np.ndarray,
    force_threshold_n: float = 8.0,
) -> dict[str, float]:
    desired = np.asarray(desired_contact, dtype=bool)
    measured = np.linalg.norm(np.asarray(contact_force_w), axis=-1) >= force_threshold_n
    if desired.shape != measured.shape:
        raise ValueError(f"Contact shape mismatch: desired={desired.shape}, measured={measured.shape}")
    mismatch = desired != measured
    return {
        "mismatch_rate": float(mismatch.mean()),
        "false_support_rate": float((desired & ~measured).mean()),
        "missed_support_rate": float((~desired & measured).mean()),
    }
