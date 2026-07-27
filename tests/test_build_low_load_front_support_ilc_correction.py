from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_low_load_front_support_ilc_correction.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location(
    "build_low_load_front_support_ilc_correction",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_load_support_trace_accepts_collected_episode(tmp_path: Path) -> None:
    path = tmp_path / "episode.npz"
    force = np.zeros((3, 4, 3), dtype=np.float32)
    force[:, 0, 0] = (-1.0, 2.0, -3.0)
    force[:, 1, 0] = (4.0, -5.0, 6.0)
    desired = np.ones((3, 4), dtype=np.uint8)
    action = np.zeros((3, 16), dtype=np.float32)
    np.savez_compressed(
        path,
        contact_force_w=force,
        desired_contact=desired,
        ref_frame=np.arange(3, dtype=np.int32),
        ref_id=np.full(3, 5, dtype=np.int32),
        executed_action16=action,
    )

    loaded_force, loaded_desired, frames, ref_id, loaded_action = (
        MODULE.load_support_trace(path)
    )

    assert loaded_force.shape == (3, 1, 2)
    np.testing.assert_allclose(
        loaded_force[:, 0],
        np.asarray(
            [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]],
            dtype=np.float32,
        ),
    )
    assert loaded_desired.shape == (3, 2)
    assert frames.tolist() == [0, 1, 2]
    assert ref_id == 5
    assert loaded_action.shape == (3, 1, 16)


def test_load_support_trace_rejects_mixed_reference_ids(
    tmp_path: Path,
) -> None:
    path = tmp_path / "mixed.npz"
    np.savez_compressed(
        path,
        contact_force_w=np.zeros((2, 4, 3), dtype=np.float32),
        desired_contact=np.ones((2, 4), dtype=np.uint8),
        ref_frame=np.arange(2, dtype=np.int32),
        ref_id=np.asarray([5, 6], dtype=np.int32),
        executed_action16=np.zeros((2, 16), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="consistent ref_id"):
        MODULE.load_support_trace(path)


@pytest.mark.parametrize(
    ("mode", "expected"),
    (
        ("independent", [[0.2, 0.8], [0.4, 0.1]]),
        ("mean", [[0.5, 0.5], [0.25, 0.25]]),
        ("max", [[0.8, 0.8], [0.4, 0.4]]),
    ),
)
def test_couple_front_deficit(
    mode: str,
    expected: list[list[float]],
) -> None:
    deficit = np.asarray(
        [[0.2, 0.8], [0.4, 0.1]],
        dtype=np.float32,
    )

    coupled = MODULE.couple_front_deficit(deficit, mode)

    np.testing.assert_allclose(
        coupled,
        np.asarray(expected, dtype=np.float32),
    )
    assert coupled is not deficit


def test_couple_front_deficit_rejects_invalid_mode() -> None:
    with pytest.raises(ValueError, match="independent, mean, or max"):
        MODULE.couple_front_deficit(
            np.zeros((3, 2), dtype=np.float32),
            "invalid",
        )
