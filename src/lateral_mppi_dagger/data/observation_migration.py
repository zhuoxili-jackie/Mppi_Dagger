from __future__ import annotations

import numpy as np


OBS_DIM = 93
ROTATION_SLICE = slice(32, 38)
_NOISE_REMAP_ROWS_TO_COLUMNS = np.asarray([0, 1, 3, 4, 2, 5], dtype=np.int64)


def _as_observation_array(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim < 1 or array.shape[-1] != OBS_DIM:
        raise ValueError(f"{name} must end in dimension {OBS_DIM}, got {array.shape}.")
    if array.dtype != np.float32:
        raise TypeError(f"{name} must be float32, got {array.dtype}.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf.")
    return array


def rotation_rows_to_deployment_columns(
    rows6: np.ndarray,
    *,
    orthonormal_tolerance: float = 2.0e-4,
) -> np.ndarray:
    """Convert ``[R00,R01,R02,R10,R11,R12]`` to deployment columns.

    The third rotation row is reconstructed as ``row0 × row1``.  Clean
    observations came from a quaternion rotation matrix, so this is exact up
    to float32 rounding and requires no simulator/reference replay.
    """

    rows = np.asarray(rows6)
    if rows.ndim < 1 or rows.shape[-1] != 6:
        raise ValueError(f"rows6 must end in dimension 6, got {rows.shape}.")
    if rows.dtype != np.float32:
        raise TypeError(f"rows6 must be float32, got {rows.dtype}.")
    row0 = rows[..., 0:3]
    row1 = rows[..., 3:6]
    norm0_error = np.abs(np.linalg.norm(row0, axis=-1) - 1.0)
    norm1_error = np.abs(np.linalg.norm(row1, axis=-1) - 1.0)
    orthogonality_error = np.abs(np.sum(row0 * row1, axis=-1))
    maximum_error = float(
        max(
            np.max(norm0_error, initial=0.0),
            np.max(norm1_error, initial=0.0),
            np.max(orthogonality_error, initial=0.0),
        )
    )
    if maximum_error > orthonormal_tolerance:
        raise ValueError(
            "Clean legacy rotation rows are not orthonormal: "
            f"max_error={maximum_error:.6g} > {orthonormal_tolerance:.6g}."
        )
    row2 = np.cross(row0, row1)
    return np.stack(
        (
            row0[..., 0],
            row0[..., 1],
            row1[..., 0],
            row1[..., 1],
            row2[..., 0],
            row2[..., 1],
        ),
        axis=-1,
    ).astype(np.float32, copy=False)


def migrate_clean_observation_rows_to_columns(
    legacy_observation: np.ndarray,
) -> np.ndarray:
    legacy = _as_observation_array(legacy_observation, "legacy_observation")
    migrated = legacy.copy()
    migrated[..., ROTATION_SLICE] = rotation_rows_to_deployment_columns(
        legacy[..., ROTATION_SLICE]
    )
    return migrated


def migrate_noisy_training_observation_rows_to_columns(
    legacy_clean: np.ndarray,
    legacy_train: np.ndarray,
) -> np.ndarray:
    """Migrate a noisy training observation without changing noise statistics.

    Clean rotation values use the exact geometric conversion.  The six
    independent Gaussian perturbations are then reassigned to the six new
    slots; discarded legacy ``R02/R12`` noises become the new ``R20/R21``
    noises.  Non-rotation channels remain bit-identical.
    """

    clean = _as_observation_array(legacy_clean, "legacy_clean")
    train = _as_observation_array(legacy_train, "legacy_train")
    if clean.shape != train.shape:
        raise ValueError(
            f"legacy_clean and legacy_train shapes differ: {clean.shape} != {train.shape}."
        )
    migrated = train.copy()
    migrated_clean = migrate_clean_observation_rows_to_columns(clean)
    legacy_noise = train[..., ROTATION_SLICE] - clean[..., ROTATION_SLICE]
    migrated[..., ROTATION_SLICE] = (
        migrated_clean[..., ROTATION_SLICE]
        + legacy_noise[..., _NOISE_REMAP_ROWS_TO_COLUMNS]
    )
    return migrated.astype(np.float32, copy=False)
