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
    method: str = "per_wheel_contact_normal_and_speed",
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
    stride_m: float = 0.04,
    duty_factor: float = 0.80,
    acceleration_seconds: float = 0.60,
    support_preload_seconds: float = 0.40,
    phase_offsets: tuple[float, float, float, float] = (
        0.75,
        0.25,
        0.50,
        0.00,
    ),
    negative_direction_phase_mirrored: bool = True,
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
    geometric_candidate = contact_coordinate_ok & speed_ok
    if method == "per_wheel_contact_normal_and_speed":
        candidate = geometric_candidate
    elif method == "generated_crawl_phase_with_geometric_preload":
        if (
            not np.isfinite(stride_m)
            or stride_m <= 0.0
            or not np.isfinite(duty_factor)
            or not 0.0 < duty_factor < 1.0
            or not np.isfinite(acceleration_seconds)
            or acceleration_seconds <= 0.0
            or not np.isfinite(support_preload_seconds)
            or support_preload_seconds < 0.0
        ):
            raise ValueError(
                "Generated crawl contact parameters must be finite with "
                "positive stride/acceleration, duty in (0,1), and "
                "non-negative preload duration."
            )
        offsets = np.asarray(phase_offsets, dtype=np.float64)
        if (
            offsets.shape != (4,)
            or not np.isfinite(offsets).all()
            or np.any((offsets < 0.0) | (offsets >= 1.0))
        ):
            raise ValueError(
                "phase_offsets must contain four finite values in [0,1)."
            )
        if motion.target_vy < 0.0 and negative_direction_phase_mirrored:
            offsets = offsets[[1, 0, 3, 2]]
        time_s = np.arange(motion.frames, dtype=np.float64) / motion.fps
        ramp = np.clip(time_s / acceleration_seconds, 0.0, 1.0)
        smooth_ramp = ramp * ramp * (3.0 - 2.0 * ramp)
        speed = abs(motion.target_vy) * smooth_ramp
        displacement = np.zeros(motion.frames, dtype=np.float64)
        displacement[1:] = np.cumsum(
            0.5 * (speed[1:] + speed[:-1]) / motion.fps
        )
        phase = (
            displacement[:, None] / stride_m + offsets[None]
        ) % 1.0
        candidate = phase < duty_factor

        # The fixed first frame deliberately starts away from the trunk and
        # ramps into the support pose. During that preload only geometry can
        # establish whether a wheel has reached its surface. Once preload is
        # complete, the generator's explicit stance/swing phase is the source
        # of truth, including low-clearance swing endpoints.
        preload_frames = min(
            int(round(support_preload_seconds * motion.fps)),
            motion.frames,
        )
        candidate[:preload_frames] = geometric_candidate[:preload_frames]
    else:
        raise ValueError(f"Unsupported contact inference method {method!r}.")
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
