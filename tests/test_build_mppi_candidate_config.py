from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_mppi_candidate_config.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location(
    "build_mppi_candidate_config",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_replace_nominal_action_entries_preserves_solver_fields(
    tmp_path: Path,
) -> None:
    asset = tmp_path / "asset.npz"
    asset.write_bytes(b"candidate")
    config = {
        "status": "base",
        "nominal_action_reference": {
            "entries": [
                {
                    "ref_id": 0,
                    "path": "old.npz",
                    "sha256": "old",
                    "action_residual_weight": 5000.0,
                },
                {
                    "ref_id": 1,
                    "path": "keep.npz",
                    "sha256": "keep",
                },
            ]
        },
    }

    result, records = MODULE.replace_nominal_action_entries(
        config,
        {0: asset},
        root=tmp_path,
    )

    entries = result["nominal_action_reference"]["entries"]
    assert entries[0]["path"] == "asset.npz"
    assert entries[0]["sha256"] != "old"
    assert entries[0]["action_residual_weight"] == 5000.0
    assert entries[1] == config["nominal_action_reference"]["entries"][1]
    assert records[0]["previous_path"] == "old.npz"
    assert config["nominal_action_reference"]["entries"][0]["path"] == "old.npz"


def test_replace_nominal_solver_overrides_removes_static_fields() -> None:
    config = {
        "nominal_action_reference": {
            "entries": [
                {
                    "ref_id": 0,
                    "path": "asset.npz",
                    "sha256": "hash",
                    "action_residual_weight": 5000.0,
                    "selection_mode": "best_sample",
                    "temperature": 50.0,
                    "warm_start": False,
                },
                {
                    "ref_id": 1,
                    "path": "keep.npz",
                    "sha256": "keep",
                    "action_residual_weight": 5000.0,
                },
            ]
        },
    }
    schedule = {
        "solver_schedule": [
            {
                "start_frame": 0,
                "action_residual_weight": 5000.0,
                "base_orientation_cost_multiplier": 1.0,
                "rear_support_loss_cost_multiplier": 1.0,
                "selection_mode": "best_sample",
                "temperature": 50.0,
                "warm_start": False,
            },
            {
                "start_frame": 80,
                "action_residual_weight": 1000.0,
                "base_orientation_cost_multiplier": 10.0,
                "base_orientation_axis_multipliers": [1.0, 1.0, 8.0],
                "rear_support_loss_cost_multiplier": 10.0,
                "selection_mode": "best_sample",
                "temperature": 200.0,
                "warm_start": True,
                "reset_warm_start": True,
            },
        ]
    }

    result, records = MODULE.replace_nominal_solver_overrides(
        config,
        {0: schedule},
    )

    entries = result["nominal_action_reference"]["entries"]
    assert entries[0]["path"] == "asset.npz"
    assert entries[0]["sha256"] == "hash"
    assert entries[0]["solver_schedule"] == schedule["solver_schedule"]
    assert "action_residual_weight" not in entries[0]
    assert "selection_mode" not in entries[0]
    assert "temperature" not in entries[0]
    assert "warm_start" not in entries[0]
    assert entries[1] == config["nominal_action_reference"]["entries"][1]
    assert records == [
        {
            "ref_id": 0,
            "previous": {
                "action_residual_weight": 5000.0,
                "selection_mode": "best_sample",
                "temperature": 50.0,
                "warm_start": False,
            },
            "replacement": schedule,
        }
    ]
    assert "solver_schedule" not in (
        config["nominal_action_reference"]["entries"][0]
    )


def test_replace_nominal_solver_overrides_validates_schedule() -> None:
    config = {
        "nominal_action_reference": {
            "entries": [{"ref_id": 0}]
        },
    }

    with pytest.raises(ValueError, match="start at frame 0"):
        MODULE.replace_nominal_solver_overrides(
            config,
            {
                0: {
                    "solver_schedule": [
                        {
                            "start_frame": 80,
                            "warm_start": True,
                        }
                    ]
                }
            },
        )


def test_replace_cost_weights_preserves_unselected_values() -> None:
    config = {
        "cost_weights": {
            "base_position": 400.0,
            "wheel_position": 60.0,
            "rear_force_overload": 26.0,
        },
    }

    result, records = MODULE.replace_cost_weights(
        config,
        {
            "wheel_position": 600.0,
            "rear_force_overload": 104.0,
        },
    )

    assert result["cost_weights"] == {
        "base_position": 400.0,
        "wheel_position": 600.0,
        "rear_force_overload": 104.0,
    }
    assert records == [
        {
            "name": "rear_force_overload",
            "previous": 26.0,
            "replacement": 104.0,
        },
        {
            "name": "wheel_position",
            "previous": 60.0,
            "replacement": 600.0,
        },
    ]
    assert config["cost_weights"]["wheel_position"] == 60.0


def test_replace_load_limits_validates_and_preserves_base() -> None:
    config = {
        "load_limits": {
            "front_normal_min_n": 8.0,
            "wheel_position_worst_fraction": 0.0,
        },
    }

    result, records = MODULE.replace_load_limits(
        config,
        {
            "wheel_position_worst_fraction": 1.0,
            "rear_overload_worst_fraction": 1.0,
        },
    )

    assert result["load_limits"] == {
        "front_normal_min_n": 8.0,
        "wheel_position_worst_fraction": 1.0,
        "rear_overload_worst_fraction": 1.0,
    }
    assert records == [
        {
            "name": "rear_overload_worst_fraction",
            "previous": None,
            "replacement": 1.0,
        },
        {
            "name": "wheel_position_worst_fraction",
            "previous": 0.0,
            "replacement": 1.0,
        },
    ]
    assert config["load_limits"]["wheel_position_worst_fraction"] == 0.0


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"unknown": 1.0}, "Unknown MPPI load limits"),
        ({"wheel_position_worst_fraction": -0.01}, "must be in"),
        ({"rear_overload_worst_fraction": 1.01}, "must be in"),
    ],
)
def test_replace_load_limits_rejects_invalid_values(
    overrides,
    message,
) -> None:
    config = {"load_limits": {"front_normal_min_n": 8.0}}

    with pytest.raises(ValueError, match=message):
        MODULE.replace_load_limits(config, overrides)


def test_replace_mppi_settings_validates_and_preserves_base() -> None:
    config = {
        "temperature": 5.0,
        "temporal_smoothing": 0.8,
        "samples": 256,
        "reference_action_lookahead_steps": 1,
    }

    result, records = MODULE.replace_mppi_settings(
        config,
        {
            "reference_action_lookahead_steps": 2,
            "temperature": 50,
            "temporal_smoothing": 0.5,
        },
    )

    assert result == {
        "temperature": 50.0,
        "temporal_smoothing": 0.5,
        "samples": 256,
        "reference_action_lookahead_steps": 2,
    }
    assert records == [
        {
            "name": "reference_action_lookahead_steps",
            "previous": 1,
            "replacement": 2,
        },
        {
            "name": "temperature",
            "previous": 5.0,
            "replacement": 50.0,
        },
        {
            "name": "temporal_smoothing",
            "previous": 0.8,
            "replacement": 0.5,
        },
    ]
    assert config["temperature"] == 5.0
    assert config["reference_action_lookahead_steps"] == 1


def test_replace_mppi_sample_count_validates_capacity_and_preserves_base() -> None:
    config = {
        "samples": 256,
        "temperature": 50.0,
        "temporal_smoothing": 0.8,
        "rear_swing_reference_proposal_scales": [
            0.2,
            0.35,
            0.5,
            0.75,
            1.0,
        ],
    }

    result, records = MODULE.replace_mppi_settings(
        config,
        {"samples": 512},
    )

    assert result["samples"] == 512
    assert records == [
        {
            "name": "samples",
            "previous": 256,
            "replacement": 512,
        }
    ]
    assert config["samples"] == 256


def test_replace_mppi_horizon_and_iterations() -> None:
    config = {
        "horizon": 40,
        "samples": 256,
        "optimization_iterations": 2,
        "temperature": 5.0,
        "temporal_smoothing": 0.8,
    }

    result, records = MODULE.replace_mppi_settings(
        config,
        {
            "horizon": 60,
            "optimization_iterations": 3,
        },
    )

    assert result["horizon"] == 60
    assert result["optimization_iterations"] == 3
    assert records == [
        {
            "name": "horizon",
            "previous": 40,
            "replacement": 60,
        },
        {
            "name": "optimization_iterations",
            "previous": 2,
            "replacement": 3,
        },
    ]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"samples": 1}, "greater than or equal to 2"),
        ({"samples": 512.0}, "must be an integer"),
        ({"samples": True}, "must be an integer"),
        ({"horizon": 0}, "positive integer"),
        ({"horizon": 60.0}, "positive integer"),
        ({"optimization_iterations": True}, "positive integer"),
        ({"temperature": 0.0}, "temperature must be positive"),
        ({"temperature": float("nan")}, "must be finite"),
        ({"temporal_smoothing": 1.0}, "must lie in"),
        (
            {"reference_action_lookahead_steps": -1},
            "non-negative integer",
        ),
        (
            {"reference_action_lookahead_steps": 1.5},
            "non-negative integer",
        ),
        (
            {"reference_action_lookahead_steps": True},
            "non-negative integer",
        ),
    ],
)
def test_replace_mppi_settings_rejects_invalid_values(
    overrides,
    message,
) -> None:
    config = {
        "horizon": 40,
        "samples": 256,
        "optimization_iterations": 2,
        "temperature": 5.0,
        "temporal_smoothing": 0.8,
        "reference_action_lookahead_steps": 1,
    }
    with pytest.raises(ValueError, match=message):
        MODULE.replace_mppi_settings(config, overrides)


def test_replace_mppi_sample_count_rejects_full_structured_population() -> None:
    config = {
        "samples": 16,
        "temperature": 5.0,
        "temporal_smoothing": 0.8,
        "rear_swing_reference_proposal_scales": [0.1, 0.2],
        "front_support_proposal_scales": [0.25],
        "combine_rear_swing_front_support_proposals": True,
    }

    with pytest.raises(ValueError, match="one stochastic sample"):
        MODULE.replace_mppi_settings(config, {"samples": 5})


def test_replace_mppi_sample_count_counts_coordinated_load_transfer() -> None:
    config = {
        "samples": 16,
        "temperature": 5.0,
        "temporal_smoothing": 0.8,
        "rear_swing_load_transfer_proposal_scales": [0.5, 1.0],
        "front_support_proposal_scales": [0.25, 1.0],
        "combine_rear_swing_load_transfer_front_support_proposals": True,
    }

    with pytest.raises(ValueError, match="one stochastic sample"):
        MODULE.replace_mppi_settings(config, {"samples": 8})


def test_replace_mppi_sample_count_counts_reference_load_transfer_triples() -> (
    None
):
    config = {
        "samples": 16,
        "temperature": 5.0,
        "temporal_smoothing": 0.8,
        "rear_swing_reference_proposal_scales": [0.1, 0.2],
        "rear_swing_load_transfer_proposal_scales": [0.5, 1.0],
        "front_support_proposal_scales": [0.25, 1.0],
        "combine_rear_swing_reference_load_transfer_front_support_proposals": (
            True
        ),
    }

    with pytest.raises(ValueError, match="one stochastic sample"):
        MODULE.replace_mppi_settings(config, {"samples": 14})


def test_replace_output_feedback_settings_validates_and_preserves_base() -> None:
    config = {
        "temperature": 5.0,
        "output_rear_swing_force_feedback_target_n": 0.0,
    }
    gains = [
        0.0,
        0.0,
        0.0025,
        -0.0025,
        0.0,
        0.0,
        0.0025,
        0.0025,
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    result, records = MODULE.replace_output_feedback_settings(
        config,
        {
            "output_rear_swing_force_feedback_target_n": 8,
            "output_rear_swing_force_feedback_scale_n": 127,
            "output_rear_swing_force_feedback_lookahead_steps": 8,
            "output_rear_swing_force_feedback_start_frame": 100,
            "output_rear_swing_force_feedback_gain_leg": gains,
        },
    )

    assert result["temperature"] == 5.0
    assert (
        result["output_rear_swing_force_feedback_target_n"]
        == 8.0
    )
    assert result["output_rear_swing_force_feedback_start_frame"] == 100
    assert result["output_rear_swing_force_feedback_gain_leg"] == gains
    assert len(records) == 5
    assert config == {
        "temperature": 5.0,
        "output_rear_swing_force_feedback_target_n": 0.0,
    }


def test_replace_front_output_feedback_settings_validates_and_preserves_base() -> None:
    config = {
        "temperature": 5.0,
        "output_front_force_feedback_target_n": 0.0,
    }
    gains = [
        0.0,
        0.0,
        0.0,
        0.0,
        0.02,
        0.02,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    result, records = MODULE.replace_output_feedback_settings(
        config,
        {
            "output_front_force_feedback_target_n": 8,
            "output_front_force_feedback_min_contact_n": 1,
            "output_front_force_feedback_lookahead_steps": 1,
            "output_front_force_feedback_gain_leg": gains,
        },
    )

    assert result["temperature"] == 5.0
    assert result["output_front_force_feedback_target_n"] == 8.0
    assert result["output_front_force_feedback_min_contact_n"] == 1.0
    assert result["output_front_force_feedback_gain_leg"] == gains
    assert len(records) == 4
    assert config == {
        "temperature": 5.0,
        "output_front_force_feedback_target_n": 0.0,
    }


def test_replace_rear_support_tracking_feedback_settings() -> None:
    config = {"temperature": 5.0}

    result, records = MODULE.replace_output_feedback_settings(
        config,
        {
            "output_rear_support_tracking_feedback_ref_ids": [0],
            "output_rear_support_tracking_feedback_gain": 0.1,
            "output_rear_support_tracking_feedback_max_abs_rad": 0.02,
            "output_rear_support_tracking_feedback_lookahead_steps": 1,
            "output_rear_support_tracking_feedback_start_frame": 220,
        },
    )

    assert result["temperature"] == 5.0
    assert result["output_rear_support_tracking_feedback_ref_ids"] == [0]
    assert result["output_rear_support_tracking_feedback_gain"] == 0.1
    assert (
        result["output_rear_support_tracking_feedback_max_abs_rad"]
        == 0.02
    )
    assert (
        result["output_rear_support_tracking_feedback_start_frame"]
        == 220
    )
    assert len(records) == 5
    assert config == {"temperature": 5.0}


def test_replace_rear_swing_height_feedback_settings() -> None:
    config = {"temperature": 5.0}

    result, records = MODULE.replace_output_feedback_settings(
        config,
        {
            "output_rear_swing_height_feedback_ref_ids": [0],
            "output_rear_swing_height_feedback_gain": 1.0,
            "output_rear_swing_height_feedback_max_abs_rad": 0.12,
            "output_rear_swing_height_feedback_lookahead_steps": 8,
            "output_rear_swing_height_feedback_start_frame": 0,
        },
    )

    assert result["temperature"] == 5.0
    assert result["output_rear_swing_height_feedback_ref_ids"] == [0]
    assert result["output_rear_swing_height_feedback_gain"] == 1.0
    assert (
        result["output_rear_swing_height_feedback_max_abs_rad"] == 0.12
    )
    assert (
        result["output_rear_swing_height_feedback_lookahead_steps"] == 8
    )
    assert len(records) == 5
    assert config == {"temperature": 5.0}


def test_replace_orientation_axis_feedback_settings() -> None:
    gains = [0.0, -0.1] + [0.0] * 5 + [-0.1] + [0.0] * 4

    result, records = MODULE.replace_output_feedback_settings(
        {"temperature": 5.0},
        {
            "output_pitch_feedback_ref_ids": [0],
            "output_pitch_feedback_gain_leg": gains,
            "output_pitch_feedback_axis": "x",
            "output_pitch_feedback_start_frame": 100,
            "output_pitch_feedback_max_abs_rad": 0.025,
        },
    )

    assert result["temperature"] == 5.0
    assert result["output_pitch_feedback_ref_ids"] == [0]
    assert result["output_pitch_feedback_gain_leg"] == gains
    assert result["output_pitch_feedback_axis"] == "x"
    assert result["output_pitch_feedback_start_frame"] == 100
    assert result["output_pitch_feedback_max_abs_rad"] == 0.025
    assert len(records) == 5


def test_replace_contact_orientation_feedback_settings() -> None:
    result, records = MODULE.replace_output_feedback_settings(
        {"temperature": 5.0},
        {
            "output_contact_orientation_feedback_ref_ids": [0],
            "output_contact_orientation_feedback_gain_xyz": [0.0, 0.0, 0.05],
            "output_contact_orientation_feedback_start_frame": 120,
            "output_contact_orientation_feedback_max_endpoint_delta_m": 0.008,
            "output_contact_orientation_feedback_max_abs_rad": 0.025,
        },
    )

    assert result["temperature"] == 5.0
    assert result["output_contact_orientation_feedback_ref_ids"] == [0]
    assert result["output_contact_orientation_feedback_gain_xyz"] == [
        0.0,
        0.0,
        0.05,
    ]
    assert (
        result["output_contact_orientation_feedback_start_frame"] == 120
    )
    assert (
        result[
            "output_contact_orientation_feedback_max_endpoint_delta_m"
        ]
        == 0.008
    )
    assert (
        result["output_contact_orientation_feedback_max_abs_rad"] == 0.025
    )
    assert len(records) == 5


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"unknown": 1.0}, "Unknown output feedback setting"),
        (
            {"output_rear_swing_force_feedback_scale_n": 0.0},
            "must be positive",
        ),
        (
            {"output_rear_swing_force_feedback_lookahead_steps": 1.5},
            "non-negative integer",
        ),
        (
            {"output_rear_swing_force_feedback_start_frame": True},
            "non-negative integer",
        ),
        (
            {
                "output_rear_swing_force_feedback_gain_leg": [0.1]
                + [0.0] * 11,
                "output_rear_swing_force_feedback_target_n": 8.0,
            },
            "zero for all front-leg",
        ),
        (
            {
                "output_rear_swing_force_feedback_gain_leg": [0.0, 0.0]
                + [0.1]
                + [0.0] * 9,
            },
            "positive target",
        ),
        (
            {
                "output_front_force_feedback_gain_leg": [0.0, 0.0, 0.1]
                + [0.0] * 9,
                "output_front_force_feedback_target_n": 8.0,
            },
            "zero for all rear-leg",
        ),
        (
            {
                "output_front_force_feedback_gain_leg": [0.0] * 4
                + [0.1]
                + [0.0] * 7,
            },
            "positive target",
        ),
        (
            {"output_front_force_feedback_min_contact_n": -1.0},
            "must be non-negative",
        ),
        (
            {"output_front_force_feedback_lookahead_steps": True},
            "non-negative integer",
        ),
        (
            {
                "output_rear_support_tracking_feedback_ref_ids": [0],
                "output_rear_support_tracking_feedback_gain": 0.1,
            },
            "positive maximum correction",
        ),
        (
            {
                "output_rear_support_tracking_feedback_ref_ids": [0],
                "output_rear_support_tracking_feedback_gain": -0.1,
                "output_rear_support_tracking_feedback_max_abs_rad": 0.02,
            },
            "must be non-negative",
        ),
        (
            {
                "output_rear_support_tracking_feedback_ref_ids": [0, 0],
                "output_rear_support_tracking_feedback_gain": 0.1,
                "output_rear_support_tracking_feedback_max_abs_rad": 0.02,
            },
            "must not contain duplicates",
        ),
        (
            {
                "output_rear_support_tracking_feedback_ref_ids": [0],
                "output_rear_support_tracking_feedback_gain": 0.1,
                "output_rear_support_tracking_feedback_max_abs_rad": 0.02,
                "output_rear_support_tracking_feedback_start_frame": True,
            },
            "non-negative integer",
        ),
        (
            {
                "output_rear_swing_height_feedback_ref_ids": [0],
                "output_rear_swing_height_feedback_gain": 1.0,
            },
            "positive maximum correction",
        ),
        (
            {
                "output_rear_swing_height_feedback_ref_ids": [0],
                "output_rear_swing_height_feedback_gain": 1.01,
                "output_rear_swing_height_feedback_max_abs_rad": 0.12,
            },
            r"must lie in \[0,1\]",
        ),
        (
            {
                "output_rear_swing_height_feedback_ref_ids": [0],
                "output_rear_swing_height_feedback_gain": 1.0,
                "output_rear_swing_height_feedback_max_abs_rad": 0.121,
            },
            r"must lie in \[0,0\.12\]",
        ),
        (
            {
                "output_rear_swing_height_feedback_ref_ids": [0, 0],
                "output_rear_swing_height_feedback_gain": 1.0,
                "output_rear_swing_height_feedback_max_abs_rad": 0.12,
            },
            "must not contain duplicates",
        ),
        (
            {
                "output_rear_swing_height_feedback_ref_ids": [0],
                "output_rear_swing_height_feedback_gain": 1.0,
                "output_rear_swing_height_feedback_max_abs_rad": 0.12,
                "output_rear_swing_height_feedback_lookahead_steps": True,
            },
            "non-negative integer",
        ),
        (
            {"output_pitch_feedback_axis": "roll"},
            "must be one of",
        ),
        (
            {"output_pitch_feedback_start_frame": True},
            "non-negative integer",
        ),
        (
            {"output_pitch_feedback_max_abs_rad": -0.01},
            "must be non-negative",
        ),
        (
            {"output_pitch_feedback_ref_ids": [0, 0]},
            "must not contain duplicates",
        ),
        (
            {
                "output_pitch_feedback_gain_leg": [0.1] + [0.0] * 11,
                "output_pitch_feedback_max_abs_rad": 0.025,
            },
            "at least one",
        ),
        (
            {
                "output_contact_orientation_feedback_ref_ids": [0],
                "output_contact_orientation_feedback_gain_xyz": [
                    0.0,
                    0.0,
                    0.05,
                ],
            },
            "positive endpoint cap",
        ),
        (
            {
                "output_contact_orientation_feedback_ref_ids": [0],
                "output_contact_orientation_feedback_gain_xyz": [
                    0.0,
                    0.0,
                    1.01,
                ],
                "output_contact_orientation_feedback_max_endpoint_delta_m": 0.008,
                "output_contact_orientation_feedback_max_abs_rad": 0.025,
            },
            r"lie in \[0,1\]",
        ),
        (
            {
                "output_contact_orientation_feedback_ref_ids": [0],
                "output_contact_orientation_feedback_gain_xyz": [
                    0.0,
                    0.0,
                    0.05,
                ],
                "output_contact_orientation_feedback_max_endpoint_delta_m": 0.021,
                "output_contact_orientation_feedback_max_abs_rad": 0.025,
            },
            r"lie in \[0,0\.02\]",
        ),
        (
            {
                "output_contact_orientation_feedback_ref_ids": [0],
                "output_contact_orientation_feedback_gain_xyz": [
                    0.0,
                    0.0,
                    0.05,
                ],
                "output_contact_orientation_feedback_max_endpoint_delta_m": 0.008,
                "output_contact_orientation_feedback_max_abs_rad": 0.051,
            },
            r"lie in \[0,0\.05\]",
        ),
    ],
)
def test_replace_output_feedback_settings_rejects_invalid_values(
    overrides,
    message,
) -> None:
    with pytest.raises(ValueError, match=message):
        MODULE.replace_output_feedback_settings({}, overrides)


def test_replace_rear_swing_reference_proposals_validates_and_preserves_base() -> None:
    config = {
        "samples": 16,
        "nominal_action_reference": {
            "entries": [{"ref_id": 0}, {"ref_id": 1}]
        },
        "temperature": 5.0,
    }
    replacements = {
        "rear_swing_reference_proposal_ref_ids": [0],
        "rear_swing_reference_proposal_scales": [0.025, 0.05, 0.1],
        "rear_swing_reference_proposal_joint_mask_leg": [
            0,
            0,
            1,
            1,
            0,
            0,
            1,
            1,
            0,
            0,
            1,
            1,
        ],
        "rear_swing_reference_proposal_lead_steps": 4,
        "rear_swing_tracking_error_proposal_scales": [0.25, 0.5, 1.0],
        "rear_swing_tracking_error_proposal_joint_mask_leg": [
            0,
            0,
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],
        "rear_swing_tracking_error_proposal_start_frame": 100,
    }

    result, records = (
        MODULE.replace_rear_swing_reference_proposal_settings(
            config,
            replacements,
        )
    )

    assert result["temperature"] == 5.0
    assert result["rear_swing_reference_proposal_ref_ids"] == [0]
    assert result["rear_swing_reference_proposal_scales"] == [
        0.025,
        0.05,
        0.1,
    ]
    assert result["rear_swing_reference_proposal_lead_steps"] == 4
    assert result["rear_swing_tracking_error_proposal_scales"] == [
        0.25,
        0.5,
        1.0,
    ]
    assert result[
        "rear_swing_tracking_error_proposal_joint_mask_leg"
    ] == [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    assert result["rear_swing_tracking_error_proposal_start_frame"] == 100
    assert len(records) == 7
    assert "rear_swing_reference_proposal_ref_ids" not in config


def test_replace_rear_swing_load_transfer_proposals_validates_and_preserves_base() -> None:
    config = {
        "samples": 16,
        "nominal_action_reference": {
            "entries": [{"ref_id": 0}, {"ref_id": 1}]
        },
        "temperature": 5.0,
    }
    gains = [
        [0.0] * 12,
        [0.0, -0.02, 0.0, 0.0, 0.0, 0.0, 0.0, -0.02]
        + [0.0] * 4,
    ]

    result, records = (
        MODULE.replace_rear_swing_reference_proposal_settings(
            config,
            {
                "rear_swing_load_transfer_proposal_ref_ids": [0],
                "rear_swing_load_transfer_proposal_scales": [0.5, 1.0],
                "rear_swing_load_transfer_proposal_gain_by_wheel_leg_rad": (
                    gains
                ),
                "rear_swing_load_transfer_proposal_start_frame": 100,
                "rear_swing_load_transfer_proposal_start_frame_by_wheel": [
                    134,
                    100,
                ],
                "rear_swing_load_transfer_proposal_gate_mode": (
                    "rear_force_imbalance"
                ),
                "rear_swing_load_transfer_proposal_imbalance_threshold_n": (
                    20.0
                ),
            },
        )
    )

    assert result["temperature"] == 5.0
    assert result["rear_swing_load_transfer_proposal_ref_ids"] == [0]
    assert result["rear_swing_load_transfer_proposal_scales"] == [0.5, 1.0]
    assert (
        result[
            "rear_swing_load_transfer_proposal_gain_by_wheel_leg_rad"
        ]
        == gains
    )
    assert result["rear_swing_load_transfer_proposal_start_frame"] == 100
    assert result[
        "rear_swing_load_transfer_proposal_start_frame_by_wheel"
    ] == [134, 100]
    assert (
        result["rear_swing_load_transfer_proposal_gate_mode"]
        == "rear_force_imbalance"
    )
    assert (
        result[
            "rear_swing_load_transfer_proposal_imbalance_threshold_n"
        ]
        == 20.0
    )
    assert len(records) == 7
    assert "rear_swing_load_transfer_proposal_ref_ids" not in config


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {
                "rear_swing_load_transfer_proposal_ref_ids": [2],
                "rear_swing_load_transfer_proposal_scales": [0.5],
                "rear_swing_load_transfer_proposal_gain_by_wheel_leg_rad": (
                    [[0.01] + [0.0] * 11, [0.0] * 12]
                ),
            },
            "active nominal-action references",
        ),
        (
            {
                "rear_swing_load_transfer_proposal_ref_ids": [0],
                "rear_swing_load_transfer_proposal_scales": [0.0],
                "rear_swing_load_transfer_proposal_gain_by_wheel_leg_rad": (
                    [[0.01] + [0.0] * 11, [0.0] * 12]
                ),
            },
            "finite positive",
        ),
        (
            {
                "rear_swing_load_transfer_proposal_ref_ids": [0],
                "rear_swing_load_transfer_proposal_scales": [0.5],
                "rear_swing_load_transfer_proposal_gain_by_wheel_leg_rad": (
                    [[0.01] + [0.0] * 11]
                ),
            },
            "exactly two rows",
        ),
        (
            {
                "rear_swing_load_transfer_proposal_start_frame": 100,
            },
            "require non-empty ref IDs",
        ),
        (
            {
                "rear_swing_load_transfer_proposal_ref_ids": [0],
                "rear_swing_load_transfer_proposal_scales": [0.5],
                "rear_swing_load_transfer_proposal_gain_by_wheel_leg_rad": (
                    [[0.01] + [0.0] * 11, [0.0] * 12]
                ),
                "rear_swing_load_transfer_proposal_start_frame": -1,
            },
            "must be a non-negative integer",
        ),
        (
            {
                "rear_swing_load_transfer_proposal_ref_ids": [0],
                "rear_swing_load_transfer_proposal_scales": [0.5],
                "rear_swing_load_transfer_proposal_gain_by_wheel_leg_rad": (
                    [[0.01] + [0.0] * 11, [0.0] * 12]
                ),
                "rear_swing_load_transfer_proposal_start_frame_by_wheel": [
                    100
                ],
            },
            "must contain exactly two non-negative integers",
        ),
        (
            {
                "rear_swing_load_transfer_proposal_ref_ids": [0],
                "rear_swing_load_transfer_proposal_scales": [0.5],
                "rear_swing_load_transfer_proposal_gain_by_wheel_leg_rad": (
                    [[0.01] + [0.0] * 11, [0.0] * 12]
                ),
                "rear_swing_load_transfer_proposal_gate_mode": (
                    "rear_force_imbalance"
                ),
            },
            "positive imbalance threshold",
        ),
        (
            {
                "rear_swing_load_transfer_proposal_ref_ids": [0],
                "rear_swing_load_transfer_proposal_scales": [0.5],
                "rear_swing_load_transfer_proposal_gain_by_wheel_leg_rad": (
                    [[0.01] + [0.0] * 11, [0.0] * 12]
                ),
                "rear_swing_load_transfer_proposal_imbalance_threshold_n": (
                    20.0
                ),
            },
            "requires rear_force_imbalance",
        ),
        (
            {
                "rear_swing_load_transfer_proposal_ref_ids": [0],
                "rear_swing_load_transfer_proposal_scales": [0.5],
                "rear_swing_load_transfer_proposal_gain_by_wheel_leg_rad": (
                    [[0.01] + [0.0] * 11, [0.0] * 12]
                ),
                "rear_swing_load_transfer_proposal_gate_mode": "unknown",
            },
            "must be 'swing_schedule' or 'rear_force_imbalance'",
        ),
    ],
)
def test_replace_rear_swing_load_transfer_proposals_rejects_invalid_values(
    overrides,
    message,
) -> None:
    config = {
        "samples": 16,
        "nominal_action_reference": {
            "entries": [{"ref_id": 0}, {"ref_id": 1}]
        },
    }

    with pytest.raises(ValueError, match=message):
        MODULE.replace_rear_swing_reference_proposal_settings(
            config,
            overrides,
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"unknown": []}, "Unknown rear-swing"),
        (
            {
                "rear_swing_reference_proposal_ref_ids": [2],
                "rear_swing_reference_proposal_scales": [0.1],
                "rear_swing_reference_proposal_joint_mask_leg": (
                    [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
                ),
            },
            "active nominal-action references",
        ),
        (
            {
                "rear_swing_reference_proposal_ref_ids": [0],
                "rear_swing_reference_proposal_scales": [0.0],
                "rear_swing_reference_proposal_joint_mask_leg": (
                    [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
                ),
            },
            "finite positive",
        ),
        (
            {
                "rear_swing_reference_proposal_ref_ids": [0],
                "rear_swing_reference_proposal_scales": [0.1],
                "rear_swing_reference_proposal_joint_mask_leg": (
                    [1] + [0] * 11
                ),
            },
            "zero for all front-leg",
        ),
        (
            {
                "rear_swing_reference_proposal_ref_ids": [0],
            },
            "require non-empty",
        ),
        (
            {
                "rear_swing_reference_proposal_ref_ids": [0],
                "rear_swing_reference_proposal_scales": [0.1],
                "rear_swing_reference_proposal_joint_mask_leg": (
                    [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
                ),
                "rear_swing_reference_proposal_lead_steps": -1,
            },
            "non-negative integer",
        ),
        (
            {
                "rear_swing_reference_proposal_ref_ids": [0],
                "rear_swing_reference_proposal_scales": [0.1],
                "rear_swing_reference_proposal_joint_mask_leg": (
                    [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
                ),
                "rear_swing_action_residual_lead_steps": -1,
            },
            "action_residual_lead_steps",
        ),
        (
            {
                "rear_swing_tracking_error_proposal_scales": [0.0],
            },
            "finite positive",
        ),
        (
            {
                "rear_swing_tracking_error_proposal_scales": [0.5],
            },
            "require complete rear-swing",
        ),
        (
            {
                "rear_swing_tracking_error_proposal_joint_mask_leg": (
                    [0, 0, 1] + [0] * 9
                ),
            },
            "requires non-empty tracking-error",
        ),
        (
            {
                "rear_swing_tracking_error_proposal_start_frame": -1,
            },
            "must be a non-negative integer",
        ),
        (
            {
                "rear_swing_tracking_error_proposal_start_frame": 100,
            },
            "requires non-empty tracking-error",
        ),
    ],
)
def test_replace_rear_swing_reference_proposals_rejects_invalid_values(
    overrides,
    message,
) -> None:
    config = {
        "samples": 16,
        "nominal_action_reference": {
            "entries": [{"ref_id": 0}, {"ref_id": 1}]
        },
    }
    with pytest.raises(ValueError, match=message):
        MODULE.replace_rear_swing_reference_proposal_settings(
            config,
            overrides,
        )


def test_replace_rear_swing_reference_proposals_can_decouple_residual_lead() -> None:
    config = {
        "samples": 16,
        "horizon": 8,
        "nominal_action_reference": {
            "entries": [{"ref_id": 0}]
        },
        "rear_swing_reference_proposal_ref_ids": [0],
        "rear_swing_reference_proposal_scales": [0.1],
        "rear_swing_reference_proposal_joint_mask_leg": [
            0,
            0,
            1,
            1,
            0,
            0,
            1,
            1,
            0,
            0,
            1,
            1,
        ],
        "rear_swing_reference_proposal_lead_steps": 4,
    }

    result, records = (
        MODULE.replace_rear_swing_reference_proposal_settings(
            config,
            {"rear_swing_action_residual_lead_steps": 0},
        )
    )

    assert result["rear_swing_reference_proposal_lead_steps"] == 4
    assert result["rear_swing_action_residual_lead_steps"] == 0
    assert records == [
        {
            "name": "rear_swing_action_residual_lead_steps",
            "previous": None,
            "replacement": 0,
        }
    ]


def test_replace_front_support_proposals_validates_and_preserves_base() -> None:
    config = {
        "samples": 16,
        "nominal_action_reference": {
            "entries": [{"ref_id": 0}, {"ref_id": 1}]
        },
        "rear_swing_reference_proposal_ref_ids": [0],
        "rear_swing_reference_proposal_scales": [0.25, 0.5],
        "rear_swing_reference_proposal_joint_mask_leg": [
            0,
            0,
            1,
            1,
            0,
            0,
            1,
            1,
            0,
            0,
            1,
            1,
        ],
    }
    gains = [
        0.0,
        0.0,
        0.0,
        0.0,
        0.02,
        0.02,
        0.0,
        0.0,
        0.02,
        0.02,
        0.0,
        0.0,
    ]

    result, records = MODULE.replace_front_support_proposal_settings(
        config,
        {
            "front_support_proposal_ref_ids": [0],
            "front_support_proposal_scales": [0.25, 0.5, 1.0],
            "front_support_proposal_gain_leg_rad": gains,
            "front_support_proposal_start_frame": 40,
            "combine_rear_swing_front_support_proposals": True,
            "include_rear_support_reference_in_coordinated_proposals": True,
            "rear_support_reference_proposal_start_frame": 100,
        },
    )

    assert result["front_support_proposal_ref_ids"] == [0]
    assert result["front_support_proposal_scales"] == [0.25, 0.5, 1.0]
    assert result["front_support_proposal_gain_leg_rad"] == gains
    assert result["front_support_proposal_start_frame"] == 40
    assert result["combine_rear_swing_front_support_proposals"] is True
    assert result[
        "include_rear_support_reference_in_coordinated_proposals"
    ] is True
    assert result["rear_support_reference_proposal_start_frame"] == 100
    assert len(records) == 7
    assert "front_support_proposal_ref_ids" not in config


def test_replace_front_support_proposals_can_coordinate_load_transfer() -> None:
    load_transfer_gains = [
        [0.0, -0.02] + [0.0] * 10,
        [0.0, 0.0, -0.02] + [0.0] * 9,
    ]
    config = {
        "samples": 16,
        "nominal_action_reference": {
            "entries": [{"ref_id": 0}]
        },
        "rear_swing_load_transfer_proposal_ref_ids": [0],
        "rear_swing_load_transfer_proposal_scales": [0.5, 1.0],
        "rear_swing_load_transfer_proposal_gain_by_wheel_leg_rad": (
            load_transfer_gains
        ),
        "rear_swing_load_transfer_proposal_gate_mode": "swing_schedule",
    }
    front_gains = [0.01, 0.01] + [0.0] * 10

    result, records = MODULE.replace_front_support_proposal_settings(
        config,
        {
            "front_support_proposal_ref_ids": [0],
            "front_support_proposal_scales": [0.25, 1.0],
            "front_support_proposal_gain_leg_rad": front_gains,
            "combine_rear_swing_load_transfer_front_support_proposals": True,
        },
    )

    assert (
        result[
            "combine_rear_swing_load_transfer_front_support_proposals"
        ]
        is True
    )
    assert len(records) == 4
    assert (
        "combine_rear_swing_load_transfer_front_support_proposals"
        not in config
    )


def test_replace_front_support_proposals_can_coordinate_reference_and_load() -> (
    None
):
    config = {
        "samples": 32,
        "nominal_action_reference": {
            "entries": [{"ref_id": 0}]
        },
        "rear_swing_reference_proposal_ref_ids": [0],
        "rear_swing_reference_proposal_scales": [0.1, 0.2],
        "rear_swing_reference_proposal_joint_mask_leg": [0, 0, 1, 1]
        + [0, 0, 1, 1]
        + [0, 0, 1, 1],
        "rear_swing_load_transfer_proposal_ref_ids": [0],
        "rear_swing_load_transfer_proposal_scales": [0.5, 1.0],
        "rear_swing_load_transfer_proposal_gain_by_wheel_leg_rad": [
            [0.0, -0.02] + [0.0] * 10,
            [0.0, 0.0, -0.02] + [0.0] * 9,
        ],
        "rear_swing_load_transfer_proposal_gate_mode": "swing_schedule",
    }

    result, records = MODULE.replace_front_support_proposal_settings(
        config,
        {
            "front_support_proposal_ref_ids": [0],
            "front_support_proposal_scales": [0.25, 1.0],
            "front_support_proposal_gain_leg_rad": [0.01, 0.01]
            + [0.0] * 10,
            "combine_rear_swing_reference_load_transfer_front_support_proposals": (
                True
            ),
        },
    )

    assert (
        result[
            "combine_rear_swing_reference_load_transfer_front_support_"
            "proposals"
        ]
        is True
    )
    assert len(records) == 4
    assert (
        "combine_rear_swing_reference_load_transfer_front_support_proposals"
        not in config
    )


def test_coordinated_load_transfer_rejects_force_imbalance_gate() -> None:
    config = {
        "samples": 16,
        "nominal_action_reference": {
            "entries": [{"ref_id": 0}]
        },
        "rear_swing_load_transfer_proposal_ref_ids": [0],
        "rear_swing_load_transfer_proposal_scales": [0.5],
        "rear_swing_load_transfer_proposal_gain_by_wheel_leg_rad": [
            [0.01] + [0.0] * 11,
            [0.0] * 12,
        ],
        "rear_swing_load_transfer_proposal_gate_mode": (
            "rear_force_imbalance"
        ),
        "rear_swing_load_transfer_proposal_imbalance_threshold_n": 20.0,
    }

    with pytest.raises(ValueError, match="require swing_schedule"):
        MODULE.replace_front_support_proposal_settings(
            config,
            {
                "front_support_proposal_ref_ids": [0],
                "front_support_proposal_scales": [0.5],
                "front_support_proposal_gain_leg_rad": [0.01]
                + [0.0] * 11,
                "combine_rear_swing_load_transfer_front_support_proposals": (
                    True
                ),
            },
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"unknown": []}, "Unknown front-support"),
        (
            {
                "front_support_proposal_ref_ids": [2],
                "front_support_proposal_scales": [0.5],
                "front_support_proposal_gain_leg_rad": [0.01]
                + [0.0] * 11,
            },
            "active nominal-action references",
        ),
        (
            {
                "front_support_proposal_ref_ids": [0],
                "front_support_proposal_scales": [0.0],
                "front_support_proposal_gain_leg_rad": [0.01]
                + [0.0] * 11,
            },
            "finite positive",
        ),
        (
            {
                "front_support_proposal_ref_ids": [0],
                "front_support_proposal_scales": [0.5],
                "front_support_proposal_gain_leg_rad": [0.0, 0.0, 0.01]
                + [0.0] * 9,
            },
            "zero for all rear-leg",
        ),
        (
            {
                "front_support_proposal_ref_ids": [0],
            },
            "require non-empty",
        ),
        (
            {
                "front_support_proposal_ref_ids": [0],
                "front_support_proposal_scales": [0.5],
                "front_support_proposal_gain_leg_rad": [0.01]
                + [0.0] * 11,
                "front_support_proposal_start_frame": -1,
            },
            "non-negative integer",
        ),
        (
            {
                "front_support_proposal_ref_ids": [0],
                "front_support_proposal_scales": [0.5],
                "front_support_proposal_gain_leg_rad": [0.01]
                + [0.0] * 11,
                "combine_rear_swing_front_support_proposals": "yes",
            },
            "must be a boolean",
        ),
        (
            {
                "combine_rear_swing_load_transfer_front_support_proposals": (
                    "yes"
                ),
            },
            "must be a boolean",
        ),
        (
            {
                "front_support_proposal_ref_ids": [0],
                "front_support_proposal_scales": [0.5],
                "front_support_proposal_gain_leg_rad": [0.01]
                + [0.0] * 11,
                "combine_rear_swing_load_transfer_front_support_proposals": (
                    True
                ),
            },
            "require complete load-transfer",
        ),
        (
            {
                "front_support_proposal_ref_ids": [0],
                "front_support_proposal_scales": [0.5],
                "front_support_proposal_gain_leg_rad": [0.01]
                + [0.0] * 11,
                "combine_rear_swing_front_support_proposals": True,
            },
            "require complete rear-swing",
        ),
        (
            {
                "front_support_proposal_ref_ids": [0],
                "front_support_proposal_scales": [0.5],
                "front_support_proposal_gain_leg_rad": [0.01]
                + [0.0] * 11,
                "include_rear_support_reference_in_coordinated_proposals": (
                    "yes"
                ),
            },
            "must be a boolean",
        ),
        (
            {
                "front_support_proposal_ref_ids": [0],
                "front_support_proposal_scales": [0.5],
                "front_support_proposal_gain_leg_rad": [0.01]
                + [0.0] * 11,
                "include_rear_support_reference_in_coordinated_proposals": (
                    True
                ),
            },
            "requires combined",
        ),
        (
            {
                "rear_support_reference_proposal_start_frame": -1,
            },
            "must be a non-negative integer",
        ),
        (
            {
                "rear_support_reference_proposal_start_frame": 100,
            },
            "requires rear-support reference coordination",
        ),
    ],
)
def test_replace_front_support_proposals_rejects_invalid_values(
    overrides,
    message,
) -> None:
    config = {
        "samples": 16,
        "nominal_action_reference": {
            "entries": [{"ref_id": 0}, {"ref_id": 1}]
        },
    }
    with pytest.raises(ValueError, match=message):
        MODULE.replace_front_support_proposal_settings(
            config,
            overrides,
        )


def test_scale_noise_std_validates_and_preserves_base() -> None:
    configured = [0.25] * 4 + [0.35] * 8
    config = {
        "noise_std_leg": configured,
        "samples": 256,
    }

    result, record = MODULE.scale_noise_std(config, 0.5)

    assert result == {
        "noise_std_leg": [0.125] * 4 + [0.175] * 8,
        "samples": 256,
    }
    assert record == {
        "scale": 0.5,
        "previous": configured,
        "replacement": [0.125] * 4 + [0.175] * 8,
    }
    assert config["noise_std_leg"] == configured


@pytest.mark.parametrize(
    ("config", "scale", "message"),
    [
        ({"noise_std_leg": [0.25] * 11}, 0.5, "exactly 12"),
        ({"noise_std_leg": [0.0] * 12}, 0.5, "at least one positive"),
        (
            {"noise_std_leg": [0.25] * 11 + [float("nan")]},
            0.5,
            "finite non-negative",
        ),
        ({"noise_std_leg": [0.25] * 12}, 0.0, "finite and positive"),
        ({"noise_std_leg": [0.25] * 12}, float("inf"), "finite and positive"),
    ],
)
def test_scale_noise_std_rejects_invalid_values(
    config,
    scale,
    message,
) -> None:
    with pytest.raises(ValueError, match=message):
        MODULE.scale_noise_std(config, scale)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"unknown": 1.0}, "Unknown MPPI cost weights"),
        ({"wheel_position": -1.0}, "must be non-negative"),
    ],
)
def test_replace_cost_weights_rejects_invalid_values(
    overrides,
    message,
) -> None:
    config = {"cost_weights": {"wheel_position": 60.0}}

    with pytest.raises(ValueError, match=message):
        MODULE.replace_cost_weights(config, overrides)
