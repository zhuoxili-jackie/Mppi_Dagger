from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_periodic_action_reference.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location(
    "build_periodic_action_reference",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_periodic_source_frames_preserve_prefix_and_repeat_cycle() -> None:
    result = MODULE._periodic_source_frames(
        frames=12,
        cycle_start_frame=2,
        period_frames=4,
    )

    np.testing.assert_array_equal(
        result,
        np.asarray([0, 1, 2, 3, 4, 5, 2, 3, 4, 5, 2, 3]),
    )


@pytest.mark.parametrize(
    ("cycle_start", "period", "message"),
    [
        (0, 4, "frame-zero"),
        (2, 1, "at least two"),
        (5, 4, "complete source cycle"),
    ],
)
def test_periodic_source_frames_reject_invalid_windows(
    cycle_start: int,
    period: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        MODULE._periodic_source_frames(
            frames=8,
            cycle_start_frame=cycle_start,
            period_frames=period,
        )


def test_rate_projection_enforces_physical_delta() -> None:
    desired = np.asarray(
        [
            [0.0] * 12,
            [1.0] * 12,
            [-1.0] * 12,
        ],
        dtype=np.float32,
    )
    scale = np.asarray([0.125] * 4 + [0.25] * 8, dtype=np.float32)

    projected, correction = MODULE._rate_project_raw_actions(
        desired,
        scale=scale,
        raw_min=np.full(12, -10.0, dtype=np.float32),
        raw_max=np.full(12, 10.0, dtype=np.float32),
        maximum_physical_delta=0.045,
    )

    physical = projected * scale
    steps = np.diff(
        physical,
        axis=0,
        prepend=np.zeros((1, 12), dtype=np.float32),
    )
    assert float(np.max(np.abs(steps))) <= 0.04500001
    assert np.any(correction > 0.0)
