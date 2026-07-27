from __future__ import annotations

import hashlib

import numpy as np
import pytest

from lateral_mppi_dagger.reference.action_reference import (
    load_nominal_action_references,
    normalize_nominal_solver_overrides,
    resolve_nominal_solver_overrides,
)


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_load_raw_nominal_action_reference(tmp_path) -> None:
    path = tmp_path / "actions.npz"
    values = np.linspace(
        -0.5,
        0.5,
        36,
        dtype=np.float32,
    ).reshape(3, 12)
    np.savez(path, raw_action_leg=values, ref_id=np.asarray([8]))

    q_des, raw, overrides, record = load_nominal_action_references(
        {
            "entries": [
                {
                    "ref_id": 8,
                    "path": str(path),
                    "sha256": _sha256(path),
                    "raw_action_key": "raw_action_leg",
                }
            ]
        }
    )

    assert q_des == {}
    assert np.array_equal(raw[8], values)
    assert overrides == {}
    assert record["entries"][0]["representation"] == "raw_action"
    assert record["entries"][0]["array_key"] == "raw_action_leg"


def test_nominal_action_reference_rejects_ambiguous_representation(
    tmp_path,
) -> None:
    path = tmp_path / "actions.npz"
    values = np.zeros((3, 12), dtype=np.float32)
    np.savez(
        path,
        q_des_leg=values,
        raw_action_leg=values,
        ref_id=np.asarray([8]),
    )

    with pytest.raises(ValueError, match="either q_des_key or raw_action_key"):
        load_nominal_action_references(
            {
                "entries": [
                    {
                        "ref_id": 8,
                        "path": str(path),
                        "sha256": _sha256(path),
                        "q_des_key": "q_des_leg",
                        "raw_action_key": "raw_action_leg",
                    }
                ]
            }
        )


def test_nominal_action_reference_rejects_hash_mismatch(tmp_path) -> None:
    path = tmp_path / "actions.npz"
    np.savez(
        path,
        raw_action_leg=np.zeros((3, 12), dtype=np.float32),
        ref_id=np.asarray([8]),
    )

    with pytest.raises(ValueError, match="hash mismatch"):
        load_nominal_action_references(
            {
                "entries": [
                    {
                        "ref_id": 8,
                        "path": str(path),
                        "sha256": "0" * 64,
                        "raw_action_key": "raw_action_leg",
                    }
                ]
            }
        )


def test_load_per_reference_solver_overrides(tmp_path) -> None:
    path = tmp_path / "actions.npz"
    values = np.zeros((3, 12), dtype=np.float32)
    np.savez(path, raw_action_leg=values, ref_id=np.asarray([8]))

    _, _, overrides, record = load_nominal_action_references(
        {
            "entries": [
                {
                    "ref_id": 8,
                    "path": str(path),
                    "sha256": _sha256(path),
                    "raw_action_key": "raw_action_leg",
                    "action_residual_weight": 500.0,
                    "selection_mode": "best_sample",
                    "temperature": 75.0,
                    "warm_start": False,
                }
            ]
        }
    )

    assert overrides == {
        8: {
            "action_residual_weight": 500.0,
            "selection_mode": "best_sample",
            "temperature": 75.0,
            "warm_start": False,
        }
    }
    assert record["entries"][0]["solver_overrides"] == overrides[8]


def test_load_frame_scheduled_solver_overrides(tmp_path) -> None:
    path = tmp_path / "actions.npz"
    values = np.zeros((120, 12), dtype=np.float32)
    np.savez(path, raw_action_leg=values, ref_id=np.asarray([8]))
    configured_schedule = [
        {
            "start_frame": 0,
            "action_residual_weight": 500.0,
            "base_orientation_cost_multiplier": 1.0,
            "lateral_velocity_cost_multiplier": 1.0,
            "rear_support_loss_cost_multiplier": 1.0,
            "selection_mode": "best_sample",
            "temperature": 50.0,
            "warm_start": False,
        },
        {
            "start_frame": 100,
            "action_residual_weight": 0.06,
            "base_orientation_cost_multiplier": 5.0,
            "base_orientation_axis_multipliers": [1.0, 1.0, 8.0],
            "lateral_velocity_cost_multiplier": 0.5,
            "rear_support_loss_cost_multiplier": 10.0,
            "selection_mode": "weighted",
            "temperature": 200.0,
            "warm_start": True,
            "reset_warm_start": True,
        },
    ]

    _, _, overrides, record = load_nominal_action_references(
        {
            "entries": [
                {
                    "ref_id": 8,
                    "path": str(path),
                    "sha256": _sha256(path),
                    "raw_action_key": "raw_action_leg",
                    "solver_schedule": configured_schedule,
                }
            ]
        }
    )

    assert overrides == {8: {"solver_schedule": configured_schedule}}
    assert record["entries"][0]["solver_overrides"] == overrides[8]
    first, first_index = resolve_nominal_solver_overrides(
        overrides[8],
        99,
    )
    second, second_index = resolve_nominal_solver_overrides(
        overrides[8],
        100,
    )
    assert first_index == 0
    assert first == {
        "action_residual_weight": 500.0,
        "base_orientation_cost_multiplier": 1.0,
        "lateral_velocity_cost_multiplier": 1.0,
        "rear_support_loss_cost_multiplier": 1.0,
        "selection_mode": "best_sample",
        "temperature": 50.0,
        "warm_start": False,
    }
    assert second_index == 1
    assert second == {
        "action_residual_weight": 0.06,
        "base_orientation_cost_multiplier": 5.0,
        "base_orientation_axis_multipliers": [1.0, 1.0, 8.0],
        "lateral_velocity_cost_multiplier": 0.5,
        "rear_support_loss_cost_multiplier": 10.0,
        "selection_mode": "weighted",
        "temperature": 200.0,
        "warm_start": True,
    }


@pytest.mark.parametrize(
    ("overrides", "error_type", "message"),
    [
        (
            {
                "base_orientation_cost_multiplier": 0.99,
            },
            ValueError,
            "at least 1.0",
        ),
        (
            {
                "base_orientation_cost_multiplier": float("inf"),
            },
            ValueError,
            "finite",
        ),
        (
            {
                "base_orientation_axis_multipliers": [1.0, 1.0],
            },
            ValueError,
            "three finite",
        ),
        (
            {
                "base_orientation_axis_multipliers": [1.0, 0.5, 1.0],
            },
            ValueError,
            "at least 1.0",
        ),
        (
            {
                "base_orientation_axis_multipliers": [
                    1.0,
                    1.0,
                    float("nan"),
                ],
            },
            ValueError,
            "three finite",
        ),
        (
            {
                "lateral_velocity_cost_multiplier": 0.0,
            },
            ValueError,
            "finite and positive",
        ),
        (
            {
                "lateral_velocity_cost_multiplier": float("nan"),
            },
            ValueError,
            "finite and positive",
        ),
        (
            {
                "rear_support_loss_cost_multiplier": 0.0,
            },
            ValueError,
            "rear_support_loss_cost_multiplier must be finite and positive",
        ),
        (
            {
                "rear_support_loss_cost_multiplier": float("nan"),
            },
            ValueError,
            "rear_support_loss_cost_multiplier must be finite and positive",
        ),
        (
            {
                "temperature": 0.0,
            },
            ValueError,
            "temperature override must be finite and positive",
        ),
        (
            {
                "temperature": float("nan"),
            },
            ValueError,
            "temperature override must be finite and positive",
        ),
        (
            {"solver_schedule": []},
            ValueError,
            "non-empty",
        ),
        (
            {
                "solver_schedule": [
                    {
                        "start_frame": 1,
                        "warm_start": False,
                    }
                ]
            },
            ValueError,
            "start at frame 0",
        ),
        (
            {
                "solver_schedule": [
                    {
                        "start_frame": 0,
                        "warm_start": False,
                    },
                    {
                        "start_frame": 0,
                        "warm_start": True,
                    },
                ]
            },
            ValueError,
            "strictly increasing",
        ),
        (
            {
                "solver_schedule": [
                    {
                        "start_frame": 0,
                        "warm_start": False,
                        "unknown": 1,
                    }
                ]
            },
            ValueError,
            "Unknown solver_schedule",
        ),
        (
            {
                "solver_schedule": [
                    {
                        "warm_start": False,
                    }
                ]
            },
            ValueError,
            "missing start_frame",
        ),
        (
            {
                "solver_schedule": [
                    {
                        "start_frame": 0,
                        "reset_warm_start": True,
                    }
                ]
            },
            ValueError,
            "at least one solver setting",
        ),
        (
            {
                "solver_schedule": [
                    {
                        "start_frame": 0,
                        "warm_start": False,
                        "reset_warm_start": 1,
                    }
                ]
            },
            TypeError,
            "reset_warm_start must be boolean",
        ),
        (
            {
                "warm_start": False,
                "solver_schedule": [
                    {
                        "start_frame": 0,
                        "warm_start": False,
                    }
                ],
            },
            ValueError,
            "cannot be combined",
        ),
    ],
)
def test_solver_schedule_rejects_invalid_configuration(
    overrides,
    error_type,
    message,
) -> None:
    with pytest.raises(error_type, match=message):
        normalize_nominal_solver_overrides(overrides)
