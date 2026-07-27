from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
SCRIPT = SCRIPTS_DIR / "build_low_load_moving_action_references.py"
sys.path.insert(0, str(SCRIPTS_DIR))
SPEC = importlib.util.spec_from_file_location(
    "build_low_load_moving_action_references",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_project_raw_sequence_enforces_rate_and_absolute_bounds() -> None:
    proposed = np.asarray(
        [
            [0.0] * 12,
            [0.7] * 12,
            [-0.7] * 12,
        ],
        dtype=np.float32,
    )
    projected = MODULE._project_raw_sequence(
        proposed,
        np.full(12, -0.5, dtype=np.float32),
        np.full(12, 0.5, dtype=np.float32),
        np.full(12, 0.2, dtype=np.float32),
    )

    assert projected.dtype == np.float32
    assert np.max(np.abs(projected)) <= 0.5
    assert np.max(
        np.abs(
            np.diff(
                projected,
                axis=0,
                prepend=np.zeros((1, 12), dtype=np.float32),
            )
        )
    ) <= 0.2 + 1.0e-7
    np.testing.assert_allclose(projected[:, 0], [0.0, 0.2, 0.0])


def test_standing_reference_baseline_reproduces_moving_reference() -> None:
    standing_reference = np.zeros((3, 12), dtype=np.float32)
    moving_reference = np.arange(36, dtype=np.float32).reshape(3, 12)
    standing_action = np.full((3, 12), 100.0, dtype=np.float32)

    proposal = MODULE._moving_proposal_q_des(
        standing_action_q_des=standing_action,
        standing_reference_q=standing_reference,
        moving_reference_q=moving_reference,
        gait_delta_scale_leg=np.ones(12, dtype=np.float32),
        balance_baseline="standing_reference",
    )

    np.testing.assert_array_equal(proposal, moving_reference)


def test_standing_action_baseline_preserves_balance_offset() -> None:
    standing_reference = np.zeros((2, 12), dtype=np.float32)
    moving_reference = np.ones((2, 12), dtype=np.float32)
    standing_action = np.full((2, 12), 0.25, dtype=np.float32)

    proposal = MODULE._moving_proposal_q_des(
        standing_action_q_des=standing_action,
        standing_reference_q=standing_reference,
        moving_reference_q=moving_reference,
        gait_delta_scale_leg=np.full(12, 0.5, dtype=np.float32),
        balance_baseline="standing_action",
    )

    np.testing.assert_allclose(proposal, 0.75)
