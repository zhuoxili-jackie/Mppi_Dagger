from __future__ import annotations

import importlib.util
import numpy as np
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/analyze_expert_episode.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location(
    "analyze_expert_episode",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _axis_quaternion(axis: int, angle: float) -> np.ndarray:
    result = np.zeros(4, dtype=np.float64)
    result[0] = np.cos(angle / 2.0)
    result[axis + 1] = np.sin(angle / 2.0)
    return result


def test_quat_rotation_vector_uses_target_axes() -> None:
    target = _axis_quaternion(2, 0.4)
    relative = _axis_quaternion(0, -0.2)
    actual = MODULE.quat_multiply(target, relative)

    result = MODULE.quat_rotation_vector(actual[None], target[None])[0]

    np.testing.assert_allclose(result, [-0.2, 0.0, 0.0], atol=1.0e-12)


def test_quat_rotation_vector_chooses_shortest_sign() -> None:
    target = np.asarray([1.0, 0.0, 0.0, 0.0])
    actual = -_axis_quaternion(1, 0.3)

    result = MODULE.quat_rotation_vector(actual[None], target[None])[0]

    np.testing.assert_allclose(result, [0.0, 0.3, 0.0], atol=1.0e-12)


def test_safe_correlation_handles_constant_input() -> None:
    assert MODULE._safe_correlation(np.ones(3), np.arange(3)) is None
    assert MODULE._safe_correlation(np.arange(3), np.arange(3)) == 1.0


def test_contiguous_true_intervals_are_half_open() -> None:
    result = MODULE._contiguous_true_intervals(
        np.asarray([False, True, True, False, True])
    )

    assert result == [(1, 3), (4, 5)]


def test_mppi_diagnostics_use_stored_component_order() -> None:
    result = MODULE._mppi_diagnostics(
        {
            "mppi_cost_components": np.asarray(
                [[1.0, 2.0], [3.0, 4.0]],
                dtype=np.float32,
            ),
            "mppi_minimum_total_cost": np.asarray([3.0, 7.0]),
            "mppi_mean_total_cost": np.asarray([4.0, 8.0]),
            "mppi_effective_sample_size": np.asarray([1.5, 4.5]),
            "mppi_rollout_termination_rate": np.asarray([0.0, 0.25]),
        },
        {"mppi_cost_component_order": ["first", "second"]},
    )

    assert result["cost_components"]["first"]["mean"] == 2.0
    assert result["cost_components"]["second"]["maximum"] == 4.0
    assert result["effective_sample_size"]["fraction_below_2"] == 0.5
