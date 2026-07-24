from __future__ import annotations

from pathlib import Path

import numpy as np


def save_tracking_csv(path: str | Path, metrics: dict[str, np.ndarray]) -> None:
    """Dependency-light plot input; plotting can be done later without changing metrics."""
    names = list(metrics)
    arrays = [np.asarray(metrics[name]).reshape(-1) for name in names]
    if len({array.shape for array in arrays}) != 1:
        raise ValueError("All tracking arrays must have the same flattened length.")
    matrix = np.column_stack(arrays)
    np.savetxt(path, matrix, delimiter=",", header=",".join(names), comments="")

