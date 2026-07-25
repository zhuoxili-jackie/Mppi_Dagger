from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_low_load_action_window_offset.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location(
    "build_low_load_action_window_offset",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_window_offset_applies_physical_vector_by_blend() -> None:
    offset = np.zeros(12, dtype=np.float32)
    offset[[5, 9]] = (0.02, 0.04)
    blend = np.asarray([0.0, 0.5, 1.0], dtype=np.float32)

    correction = MODULE.build_window_offset(
        physical_offset_leg=offset,
        blend=blend,
    )

    assert correction.shape == (3, 12)
    assert correction[:, 5].tolist() == pytest.approx([0.0, 0.01, 0.02])
    assert correction[:, 9].tolist() == pytest.approx([0.0, 0.02, 0.04])
    assert not np.count_nonzero(
        correction[:, [0, 1, 2, 3, 4, 6, 7, 8, 10, 11]]
    )


@pytest.mark.parametrize(
    "offset, blend",
    (
        (np.zeros(11, dtype=np.float32), np.zeros(3, dtype=np.float32)),
        (
            np.full(12, np.nan, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
        ),
        (np.zeros(12, dtype=np.float32), np.zeros((3, 1), dtype=np.float32)),
        (
            np.zeros(12, dtype=np.float32),
            np.asarray([0.0, 1.1], dtype=np.float32),
        ),
    ),
)
def test_window_offset_rejects_invalid_arrays(
    offset: np.ndarray,
    blend: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        MODULE.build_window_offset(
            physical_offset_leg=offset,
            blend=blend,
        )
