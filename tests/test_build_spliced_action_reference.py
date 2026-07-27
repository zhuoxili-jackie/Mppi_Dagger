from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_spliced_action_reference.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location(
    "build_spliced_action_reference",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_load_action_stream_accepts_raw_leg_asset(
    tmp_path: Path,
) -> None:
    path = tmp_path / "raw_action_asset.npz"
    raw = np.arange(36, dtype=np.float32).reshape(3, 12) / 100.0
    np.savez_compressed(
        path,
        raw_action_leg=raw,
        ref_id=np.asarray([4], dtype=np.int64),
    )

    actions, provenance = MODULE._load_action_stream(
        path,
        array_key="raw_action_leg",
        env_index=0,
        ref_id=4,
    )

    assert actions.shape == (3, 16)
    np.testing.assert_allclose(actions[:, :12], raw)
    np.testing.assert_array_equal(
        actions[:, 12:],
        np.zeros((3, 4), dtype=np.float32),
    )
    assert provenance["representation"] == "raw_action_leg"


def test_rear_joint_scope_selects_type_grouped_columns() -> None:
    np.testing.assert_array_equal(
        np.flatnonzero(MODULE._joint_scope_mask("rear")),
        np.asarray([2, 3, 6, 7, 10, 11]),
    )


def test_splice_action_streams_replaces_only_selected_joints() -> None:
    prefix = np.zeros((4, 16), dtype=np.float32)
    tail = np.ones((4, 16), dtype=np.float32)
    result, tail_steps = MODULE._splice_action_streams(
        prefix,
        tail,
        steps=4,
        tail_intervals=[(1, 3)],
        tail_joint_mask=MODULE._joint_scope_mask("rear"),
    )

    expected = np.zeros((4, 16), dtype=np.float32)
    expected[1:3, [2, 3, 6, 7, 10, 11]] = 1.0
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(
        tail_steps,
        np.asarray([False, True, True, False]),
    )
