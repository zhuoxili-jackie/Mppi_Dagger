from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_low_load_action_blend_correction.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location(
    "build_low_load_action_blend_correction",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_front_scope_uses_policy_type_grouped_joint_order() -> None:
    mask = MODULE._joint_scope_mask("front")

    assert np.flatnonzero(mask).tolist() == [0, 1, 4, 5, 8, 9]


@pytest.mark.parametrize(
    ("scope", "expected"),
    [
        ("hips", [0, 1, 2, 3]),
        ("front_thighs", [4, 5]),
        ("rear_calves", [10, 11]),
        ("fr_hips", [1]),
        ("rl_hips", [2]),
        ("fl_calves", [8]),
        ("fr_thighs", [5]),
        ("rl_calves", [10]),
        ("rr_calves", [11]),
    ],
)
def test_joint_type_scopes_use_policy_order(
    scope: str,
    expected: list[int],
) -> None:
    assert np.flatnonzero(
        MODULE._joint_scope_mask(scope)
    ).tolist() == expected


def test_tail_blend_masks_rear_joints() -> None:
    base = np.zeros((5, 12), dtype=np.float32)
    support = np.ones((5, 12), dtype=np.float32)
    scale = np.ones(12, dtype=np.float32)

    correction, blend = MODULE.build_tail_blend(
        base,
        support,
        scale_leg=scale,
        steps=4,
        tail_start=0,
        ramp_frames=1,
        joint_mask=MODULE._joint_scope_mask("front"),
    )

    assert blend.tolist() == pytest.approx([0.0, 1.0, 1.0, 1.0])
    assert not np.count_nonzero(
        correction[:, [2, 3, 6, 7, 10, 11]]
    )
    assert np.all(
        correction[1:, [0, 1, 4, 5, 8, 9]] == 1.0
    )


def test_joint_scope_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="Unknown joint scope"):
        MODULE._joint_scope_mask("middle")


def test_tail_blend_applies_policy_order_joint_scales() -> None:
    base = np.zeros((4, 12), dtype=np.float32)
    support = np.ones((4, 12), dtype=np.float32)
    joint_scales = np.arange(12, dtype=np.float32)

    correction, _ = MODULE.build_tail_blend(
        base,
        support,
        scale_leg=np.ones(12, dtype=np.float32),
        steps=3,
        tail_start=0,
        ramp_frames=1,
        joint_scales=joint_scales,
    )

    assert np.array_equal(correction[1], joint_scales)
