from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_low_load_action_window_correction.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location(
    "build_low_load_action_window_correction",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_parse_window_uses_exclusive_end() -> None:
    assert MODULE.parse_window("3:9", steps=10) == (3, 9)


@pytest.mark.parametrize("value", ("3", "x:4", "-1:4", "4:4", "3:11"))
def test_parse_window_rejects_invalid_values(value: str) -> None:
    with pytest.raises(ValueError, match="Window"):
        MODULE.parse_window(value, steps=10)


def test_window_blend_unions_windows_and_ramps_both_edges() -> None:
    blend = MODULE.build_window_blend(
        steps=12,
        windows=((1, 6), (5, 10)),
        ramp_in_frames=2,
        ramp_out_frames=2,
    )

    assert blend.tolist() == pytest.approx(
        [0.0, 0.0, 0.5, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 0.5, 0.0, 0.0]
    )


def test_window_correction_respects_policy_mask_and_scale() -> None:
    base = np.zeros((6, 12), dtype=np.float32)
    support = np.ones((6, 12), dtype=np.float32)
    blend = np.asarray([0.0, 0.5, 1.0, 0.5, 0.0], dtype=np.float32)
    joint_mask = np.zeros(12, dtype=bool)
    joint_mask[[2, 10]] = True
    joint_scales = np.ones(12, dtype=np.float32)
    joint_scales[10] = 2.0

    correction = MODULE.build_window_correction(
        base,
        support,
        scale_leg=np.ones(12, dtype=np.float32),
        blend=blend,
        joint_mask=joint_mask,
        joint_scales=joint_scales,
    )

    assert not np.count_nonzero(
        correction[:, [0, 1, 3, 4, 5, 6, 7, 8, 9, 11]]
    )
    assert correction[:, 2].tolist() == pytest.approx(blend)
    assert correction[:, 10].tolist() == pytest.approx(2.0 * blend)
