from __future__ import annotations

import pytest
import torch

from lateral_mppi_dagger.env.isaac_mppi_rollout import (
    IsaacRolloutCostWeights,
    IsaacRolloutLoadLimits,
    base_height_drop_cost,
    base_orientation_tracking_cost,
    lateral_velocity_tracking_cost,
    load_support_cost_terms,
    rear_leg_position_cost,
    scheduled_rear_wheel_height_deficit_cost,
    scheduled_rear_wheel_lateral_position_cost,
    select_global_best_cost_components,
    structured_candidate_cost_diagnostics,
    wheel_position_tracking_cost,
)


def test_terminal_orientation_cost_weight_is_default_exact_noop() -> None:
    weights = IsaacRolloutCostWeights()

    assert weights.terminal_base_orientation == 0.0
    assert weights.rear_swing_height_deficit == 0.0


def test_terminal_orientation_cost_weight_parses_positive_value() -> None:
    weights = IsaacRolloutCostWeights.from_dict(
        {"terminal_base_orientation": 800.0}
    )

    assert weights.terminal_base_orientation == 800.0


@pytest.mark.parametrize(
    "value",
    (-0.01, float("nan"), float("inf")),
)
def test_cost_weights_fail_closed_on_invalid_values(value: float) -> None:
    with pytest.raises(ValueError, match="non-negative and finite"):
        IsaacRolloutCostWeights.from_dict(
            {"terminal_base_orientation": value}
        )


def test_lateral_velocity_absolute_scale_is_default_exact_noop() -> None:
    error = torch.tensor([-0.05, 0.0, 0.02], dtype=torch.float32)

    actual = lateral_velocity_tracking_cost(error)

    assert torch.equal(actual, error.square())


def test_lateral_velocity_absolute_scale_adds_continuous_mae_cost() -> None:
    error = torch.tensor([-0.05, 0.0, 0.02], dtype=torch.float32)

    actual = lateral_velocity_tracking_cost(
        error,
        absolute_scale_m_s=0.01,
    )

    torch.testing.assert_close(
        actual,
        error.square() + 0.01 * torch.abs(error),
    )


@pytest.mark.parametrize("value", (-0.01, float("nan"), float("inf")))
def test_lateral_velocity_absolute_scale_fails_closed(value: float) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        lateral_velocity_tracking_cost(torch.zeros(1), value)
    with pytest.raises(ValueError, match="finite and non-negative"):
        IsaacRolloutLoadLimits.from_dict(
            {"lateral_velocity_absolute_scale_m_s": value}
        )


def test_base_height_drop_cost_has_a_downward_only_margin() -> None:
    errors = torch.tensor(
        [
            [0.0, 0.0, -0.10],
            [0.0, 0.0, -0.07],
            [0.0, 0.0, 0.10],
        ]
    )

    cost = base_height_drop_cost(errors, 0.08)

    torch.testing.assert_close(
        cost,
        torch.tensor([0.0004, 0.0, 0.0]),
    )


def test_base_orientation_axis_multipliers_emphasize_target_axes() -> None:
    angle = 0.2
    half_angle = torch.tensor(angle / 2.0)
    cosine = torch.cos(half_angle)
    sine = torch.sin(half_angle)
    actual = torch.tensor(
        [
            [cosine, sine, 0.0, 0.0],
            [cosine, 0.0, sine, 0.0],
            [cosine, 0.0, 0.0, sine],
        ],
        dtype=torch.float32,
    )
    target = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0]] * 3,
        dtype=torch.float32,
    )

    cost = base_orientation_tracking_cost(
        actual,
        target,
        torch.tensor([1.0, 2.0, 5.0]),
    )

    torch.testing.assert_close(
        cost,
        torch.tensor([1.0, 2.0, 5.0]) * angle**2,
        rtol=1.0e-5,
        atol=1.0e-6,
    )


def test_unit_orientation_axis_multipliers_preserve_quaternion_angle() -> None:
    actual = torch.tensor(
        [[0.9, 0.1, 0.2, 0.3]],
        dtype=torch.float32,
    )
    actual /= torch.linalg.vector_norm(actual, dim=-1, keepdim=True)
    target = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0]],
        dtype=torch.float32,
    )

    cost = base_orientation_tracking_cost(
        actual,
        target,
        torch.ones(3),
    )
    expected = (
        2.0
        * torch.acos(torch.abs(torch.sum(actual * target, dim=-1)))
    ).square()

    assert torch.equal(cost, expected)


@pytest.mark.parametrize(
    "multipliers",
    (
        torch.ones(2),
        torch.tensor([1.0, 0.99, 1.0]),
        torch.tensor([1.0, float("inf"), 1.0]),
    ),
)
def test_base_orientation_axis_multipliers_fail_closed(
    multipliers: torch.Tensor,
) -> None:
    quaternion = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0]],
        dtype=torch.float32,
    )

    with pytest.raises(ValueError, match="axis multipliers"):
        base_orientation_tracking_cost(
            quaternion,
            quaternion,
            multipliers,
        )


def test_load_limits_accept_nonnegative_height_guard_stop_frame() -> None:
    limits = IsaacRolloutLoadLimits.from_dict(
        {
            "base_height_drop_stop_frame": 50,
            "lateral_position_start_frame": 50,
            "rear_swing_lateral_position_start_frame": 100,
            "rear_swing_height_deficit_start_frame": 110,
            "rear_swing_height_scale_m": 0.012,
        }
    )

    assert limits.base_height_drop_stop_frame == 50.0
    assert limits.lateral_position_start_frame == 50.0
    assert limits.rear_swing_lateral_position_start_frame == 100.0
    assert limits.rear_swing_height_deficit_start_frame == 110.0
    assert limits.rear_swing_height_scale_m == 0.012


@pytest.mark.parametrize(
    "name",
    (
        "lateral_position_start_frame",
        "rear_swing_lateral_position_start_frame",
        "rear_swing_height_deficit_start_frame",
    ),
)
def test_load_limits_reject_negative_scheduled_tracking_frame(
    name: str,
) -> None:
    with pytest.raises(ValueError, match="Scheduled tracking frames"):
        IsaacRolloutLoadLimits.from_dict({name: -1})


@pytest.mark.parametrize("value", (0.0, -0.01, float("nan"), float("inf")))
def test_rear_swing_height_scale_rejects_nonpositive_values(
    value: float,
) -> None:
    with pytest.raises(ValueError, match="height scale"):
        IsaacRolloutLoadLimits.from_dict(
            {"rear_swing_height_scale_m": value}
        )


@pytest.mark.parametrize(
    "values",
    (
        {
            "front_normal_low_force_threshold_n": 6.0,
            "front_normal_low_force_count_penalty": 0.0,
        },
        {
            "front_normal_low_force_threshold_n": 0.0,
            "front_normal_low_force_count_penalty": 1.0,
        },
        {
            "front_normal_low_force_threshold_n": float("nan"),
            "front_normal_low_force_count_penalty": 1.0,
        },
        {
            "front_normal_low_force_threshold_n": 6.0,
            "front_normal_low_force_count_penalty": -1.0,
        },
    ),
)
def test_load_limits_reject_invalid_front_low_force_count(
    values: dict[str, float],
) -> None:
    with pytest.raises(ValueError, match="Front low-force"):
        IsaacRolloutLoadLimits.from_dict(values)


@pytest.mark.parametrize(
    "name",
    (
        "wheel_position_worst_fraction",
        "rear_overload_worst_fraction",
    ),
)
@pytest.mark.parametrize("value", (-0.01, 1.01, float("nan")))
def test_load_limits_reject_invalid_worst_fraction(
    name: str,
    value: float,
) -> None:
    with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
        IsaacRolloutLoadLimits.from_dict({name: value})


def test_wheel_position_worst_mix_preserves_legacy_mean_scale() -> None:
    errors = torch.zeros((2, 4, 3), dtype=torch.float32)
    errors[0, 0] = 2.0
    errors[1] = 2.0

    mean_cost = wheel_position_tracking_cost(errors, 0.0)
    worst_cost = wheel_position_tracking_cost(errors, 1.0)

    torch.testing.assert_close(mean_cost, torch.tensor([1.0, 4.0]))
    torch.testing.assert_close(worst_cost, torch.tensor([4.0, 4.0]))


def test_wheel_position_cost_rejects_bad_shape_and_fraction() -> None:
    with pytest.raises(ValueError, match=r"shape \[batch,4,3\]"):
        wheel_position_tracking_cost(torch.zeros((1, 4, 2)), 0.0)
    with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
        wheel_position_tracking_cost(torch.zeros((1, 4, 3)), 1.1)


def test_rear_leg_position_cost_uses_type_grouped_rear_joints_only() -> None:
    errors = torch.zeros((2, 12), dtype=torch.float32)
    errors[0, [2, 3, 6, 7, 10, 11]] = 2.0
    errors[1, [0, 1, 4, 5, 8, 9]] = 5.0

    cost = rear_leg_position_cost(errors)

    torch.testing.assert_close(cost, torch.tensor([4.0, 0.0]))


def test_rear_leg_position_cost_rejects_non_policy_shape() -> None:
    with pytest.raises(
        ValueError,
        match=r"joint_position_error must have shape \[batch,12\]",
    ):
        rear_leg_position_cost(torch.zeros((2, 16)))


def test_rear_swing_lateral_position_tracks_only_scheduled_rear_wheel() -> None:
    errors = torch.zeros((3, 4, 3), dtype=torch.float32)
    errors[0, 0, 1] = 20.0
    errors[0, 2, 1] = 2.0
    errors[0, 3, 1] = 5.0
    errors[1, 2, 1] = 2.0
    errors[1, 3, 1] = 4.0
    desired = torch.ones((3, 4), dtype=torch.bool)
    desired[0, 2] = False
    desired[1, 2:] = False

    cost = scheduled_rear_wheel_lateral_position_cost(errors, desired)

    torch.testing.assert_close(cost, torch.tensor([4.0, 10.0, 0.0]))


def test_rear_swing_lateral_position_rejects_bad_shapes() -> None:
    with pytest.raises(
        ValueError,
        match=r"wheel_position_error_w must have shape \[batch,4,3\]",
    ):
        scheduled_rear_wheel_lateral_position_cost(
            torch.zeros((2, 4, 2)),
            torch.ones((2, 4), dtype=torch.bool),
        )
    with pytest.raises(
        ValueError,
        match=r"desired_contact must have shape \[batch,4\]",
    ):
        scheduled_rear_wheel_lateral_position_cost(
            torch.zeros((2, 4, 3)),
            torch.ones((2, 3), dtype=torch.bool),
        )


def test_rear_swing_height_deficit_uses_only_below_target_swing_wheel() -> None:
    errors = torch.zeros((3, 4, 3), dtype=torch.float32)
    errors[0, 0, 2] = -2.0
    errors[0, 2, 2] = -0.012
    errors[0, 3, 2] = -0.024
    errors[1, 2, 2] = -0.006
    errors[1, 3, 2] = 0.004
    errors[2, 2, 2] = -0.012
    desired = torch.ones((3, 4), dtype=torch.bool)
    desired[0, 2] = False
    desired[1, 2:] = False

    cost = scheduled_rear_wheel_height_deficit_cost(
        errors,
        desired,
        0.012,
    )

    torch.testing.assert_close(cost, torch.tensor([1.0, 0.125, 0.0]))


def test_rear_swing_height_deficit_rejects_bad_inputs() -> None:
    with pytest.raises(
        ValueError,
        match=r"wheel_position_error_w must have shape \[batch,4,3\]",
    ):
        scheduled_rear_wheel_height_deficit_cost(
            torch.zeros((2, 4, 2)),
            torch.ones((2, 4), dtype=torch.bool),
            0.012,
        )
    with pytest.raises(
        ValueError,
        match=r"desired_contact must have shape \[batch,4\]",
    ):
        scheduled_rear_wheel_height_deficit_cost(
            torch.zeros((2, 4, 3)),
            torch.ones((2, 3), dtype=torch.bool),
            0.012,
        )
    with pytest.raises(ValueError, match="finite and positive"):
        scheduled_rear_wheel_height_deficit_cost(
            torch.zeros((2, 4, 3)),
            torch.ones((2, 4), dtype=torch.bool),
            0.0,
        )


def test_global_best_cost_components_survive_a_worse_later_iteration() -> None:
    first_cost, first_components = select_global_best_cost_components(
        torch.tensor([8.0, 3.0, 5.0]),
        {
            "tracking": torch.tensor([7.0, 2.0, 4.0]),
            "regularization": torch.tensor([1.0, 1.0, 1.0]),
        },
    )
    final_cost, final_components = select_global_best_cost_components(
        torch.tensor([9.0, 4.0, 6.0]),
        {
            "tracking": torch.tensor([8.0, 3.0, 5.0]),
            "regularization": torch.tensor([1.0, 1.0, 1.0]),
        },
        first_cost,
        first_components,
    )

    assert final_cost == 3.0
    assert final_components == {
        "tracking": 2.0,
        "regularization": 1.0,
    }


def test_global_best_cost_components_accept_a_better_later_iteration() -> None:
    final_cost, final_components = select_global_best_cost_components(
        torch.tensor([2.0, 4.0]),
        {
            "tracking": torch.tensor([1.25, 3.0]),
            "regularization": torch.tensor([0.75, 1.0]),
        },
        3.0,
        {"tracking": 2.0, "regularization": 1.0},
    )

    assert final_cost == 2.0
    assert final_components == {
        "tracking": 1.25,
        "regularization": 0.75,
    }


def test_structured_candidate_cost_diagnostics_resolves_tail_components() -> None:
    result = structured_candidate_cost_diagnostics(
        torch.tensor([3.0, 2.0, 5.0, 4.0]),
        {
            "tracking": torch.tensor([2.0, 1.0, 3.0, 2.5]),
            "regularization": torch.tensor([1.0, 1.0, 2.0, 1.5]),
        },
        proposal_count=2,
    )

    assert result == {
        "proposal_count": 2,
        "total_costs": [5.0, 4.0],
        "cost_gap_from_iteration_best": [3.0, 2.0],
        "iteration_minimum_total_cost": 2.0,
        "stochastic_minimum_total_cost": 2.0,
        "structured_minimum_total_cost": 4.0,
        "cost_components": {
            "tracking": [3.0, 2.5],
            "regularization": [2.0, 1.5],
        },
    }


def test_load_support_cost_detects_front_loss_and_rear_overload() -> None:
    limits = IsaacRolloutLoadLimits()
    safe = torch.tensor(
        [[[8.0, 0.0, 0.0], [8.0, 0.0, 0.0], [0.0, 0.0, 70.0], [0.0, 0.0, 70.0]]]
    )
    unsafe = torch.tensor(
        [[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 145.0], [0.0, 0.0, 0.0]]]
    )
    desired = torch.ones((1, 4), dtype=torch.bool)
    safe_cost = load_support_cost_terms(safe, desired, 8.0, limits)
    unsafe_cost = load_support_cost_terms(unsafe, desired, 8.0, limits)
    assert safe_cost["front_normal_support"].item() == 0.0
    assert safe_cost["rear_force_overload"].item() == 0.0
    assert safe_cost["rear_support_loss"].item() == 0.0
    assert unsafe_cost["front_normal_support"].item() > 0.0
    assert unsafe_cost["rear_force_overload"].item() > 0.0
    assert unsafe_cost["rear_support_loss"].item() > 0.0


def test_rear_support_loss_follows_the_desired_contact_schedule() -> None:
    limits = IsaacRolloutLoadLimits()
    forces = torch.tensor(
        [[[8.0, 0.0, 0.0], [8.0, 0.0, 0.0], [0.0, 0.0, 70.0], [0.0, 0.0, 0.0]]]
    )
    one_rear_swing = torch.ones((1, 4), dtype=torch.bool)
    one_rear_swing[0, 3] = False
    both_rear_desired = torch.ones((1, 4), dtype=torch.bool)

    scheduled_swing_cost = load_support_cost_terms(
        forces,
        one_rear_swing,
        8.0,
        limits,
    )
    missing_support_cost = load_support_cost_terms(
        forces,
        both_rear_desired,
        8.0,
        limits,
    )

    assert scheduled_swing_cost["rear_support_loss"].item() == 0.0
    assert missing_support_cost["rear_support_loss"].item() == 0.25


def test_rear_force_imbalance_is_disabled_during_scheduled_swing() -> None:
    limits = IsaacRolloutLoadLimits(rear_balance_scale_n=70.0)
    forces = torch.tensor(
        [[[8.0, 0.0, 0.0], [8.0, 0.0, 0.0], [0.0, 0.0, 140.0], [0.0, 0.0, 70.0]]]
    )
    both_rear_desired = torch.ones((1, 4), dtype=torch.bool)
    one_rear_swing = both_rear_desired.clone()
    one_rear_swing[0, 3] = False

    balanced_phase_cost = load_support_cost_terms(
        forces,
        both_rear_desired,
        8.0,
        limits,
    )
    swing_phase_cost = load_support_cost_terms(
        forces,
        one_rear_swing,
        8.0,
        limits,
    )

    assert balanced_phase_cost["rear_force_imbalance"].item() == 1.0
    assert swing_phase_cost["rear_force_imbalance"].item() == 0.0


def test_rear_swing_force_penalizes_only_unscheduled_contact() -> None:
    limits = IsaacRolloutLoadLimits(rear_balance_scale_n=70.0)
    forces = torch.tensor(
        [[[8.0, 0.0, 0.0], [8.0, 0.0, 0.0], [0.0, 0.0, 70.0], [0.0, 0.0, 148.0]]]
    )
    both_rear_desired = torch.ones((1, 4), dtype=torch.bool)
    right_rear_swing = both_rear_desired.clone()
    right_rear_swing[0, 3] = False

    support_phase_cost = load_support_cost_terms(
        forces,
        both_rear_desired,
        8.0,
        limits,
    )
    swing_phase_cost = load_support_cost_terms(
        forces,
        right_rear_swing,
        8.0,
        limits,
    )

    assert support_phase_cost["rear_swing_force"].item() == 0.0
    # Only the undesired RR normal force above the existing 8 N contact
    # threshold contributes: ((148 - 8) / 70)^2 == 4.
    assert swing_phase_cost["rear_swing_force"].item() == 4.0


def test_rear_overload_worst_mix_targets_the_more_loaded_wheel() -> None:
    mean_limits = IsaacRolloutLoadLimits(
        rear_normal_overload_n=100.0,
        rear_normal_scale_n=10.0,
        rear_overload_worst_fraction=0.0,
    )
    worst_limits = IsaacRolloutLoadLimits(
        rear_normal_overload_n=100.0,
        rear_normal_scale_n=10.0,
        rear_overload_worst_fraction=1.0,
    )
    forces = torch.tensor(
        [[[8.0, 0.0, 0.0], [8.0, 0.0, 0.0], [0.0, 0.0, 110.0], [0.0, 0.0, 120.0]]]
    )
    desired = torch.ones((1, 4), dtype=torch.bool)

    mean_cost = load_support_cost_terms(
        forces,
        desired,
        8.0,
        mean_limits,
    )["rear_force_overload"]
    worst_cost = load_support_cost_terms(
        forces,
        desired,
        8.0,
        worst_limits,
    )["rear_force_overload"]

    torch.testing.assert_close(mean_cost, torch.tensor([2.5]))
    torch.testing.assert_close(worst_cost, torch.tensor([4.0]))


def test_load_support_cost_penalizes_only_desired_front_x_detachment() -> None:
    limits = IsaacRolloutLoadLimits(
        front_contact_position_margin_m=0.002,
        front_contact_position_scale_m=0.010,
    )
    forces = torch.tensor(
        [[[8.0, 0.0, 0.0], [8.0, 0.0, 0.0], [0.0, 0.0, 70.0], [0.0, 0.0, 70.0]]]
    )
    position_error = torch.zeros((1, 4, 3))
    position_error[0, 0, 0] = -0.012
    desired = torch.ones((1, 4), dtype=torch.bool)

    desired_cost = load_support_cost_terms(
        forces,
        desired,
        8.0,
        limits,
        wheel_position_error_w=position_error,
    )
    desired[0, 0] = False
    swing_cost = load_support_cost_terms(
        forces,
        desired,
        8.0,
        limits,
        wheel_position_error_w=position_error,
    )

    assert desired_cost["front_normal_support"].item() == 0.5
    assert swing_cost["front_normal_support"].item() == 0.0


def test_load_support_contact_position_proxy_is_bounded() -> None:
    limits = IsaacRolloutLoadLimits(
        front_contact_position_scale_m=0.040,
        front_contact_position_max_normalized=1.0,
    )
    forces = torch.tensor(
        [[[8.0, 0.0, 0.0], [8.0, 0.0, 0.0], [0.0, 0.0, 70.0], [0.0, 0.0, 70.0]]]
    )
    position_error = torch.zeros((1, 4, 3))
    position_error[0, :2, 0] = -0.200
    desired = torch.ones((1, 4), dtype=torch.bool)

    cost = load_support_cost_terms(
        forces,
        desired,
        8.0,
        limits,
        wheel_position_error_w=position_error,
    )

    assert cost["front_normal_support"].item() == 1.0


def test_front_force_balance_is_bounded_and_requires_both_desired() -> None:
    limits = IsaacRolloutLoadLimits(front_force_balance_scale_n=6.0)
    forces = torch.tensor(
        [[[12.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 70.0], [0.0, 0.0, 70.0]]]
    )
    both_desired = torch.ones((1, 4), dtype=torch.bool)
    one_front_swing = both_desired.clone()
    one_front_swing[0, 1] = False

    balanced_cost = load_support_cost_terms(
        forces,
        both_desired,
        8.0,
        limits,
    )
    swing_cost = load_support_cost_terms(
        forces,
        one_front_swing,
        8.0,
        limits,
    )

    # The ordinary two-front deficit is 0.5 and the capped imbalance adds 1.
    assert balanced_cost["front_normal_support"].item() == 1.5
    # The remaining desired front is above the minimum; swing is unpenalized.
    assert swing_cost["front_normal_support"].item() == 0.0


def test_front_support_worst_mix_targets_the_weaker_desired_wheel() -> None:
    mean_limits = IsaacRolloutLoadLimits(
        front_normal_min_n=10.0,
        front_support_worst_fraction=0.0,
    )
    worst_limits = IsaacRolloutLoadLimits(
        front_normal_min_n=10.0,
        front_support_worst_fraction=1.0,
    )
    forces = torch.tensor(
        [[[5.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 0.0, 70.0], [0.0, 0.0, 70.0]]]
    )
    desired = torch.ones((1, 4), dtype=torch.bool)

    mean_cost = load_support_cost_terms(
        forces,
        desired,
        8.0,
        mean_limits,
    )["front_normal_support"]
    worst_cost = load_support_cost_terms(
        forces,
        desired,
        8.0,
        worst_limits,
    )["front_normal_support"]

    torch.testing.assert_close(mean_cost, torch.tensor([0.125]))
    torch.testing.assert_close(worst_cost, torch.tensor([0.25]))


def test_front_low_force_count_matches_desired_world_x_samples() -> None:
    limits = IsaacRolloutLoadLimits(
        front_normal_min_n=0.0,
        front_normal_low_force_threshold_n=6.0,
        front_normal_low_force_count_penalty=0.75,
    )
    forces = torch.tensor(
        [
            [
                [5.99, 100.0, 100.0],
                [6.0, 0.0, 0.0],
                [0.0, 0.0, 70.0],
                [0.0, 0.0, 70.0],
            ],
            [
                [0.0, 0.0, 0.0],
                [5.0, 0.0, 0.0],
                [0.0, 0.0, 70.0],
                [0.0, 0.0, 70.0],
            ],
        ]
    )
    desired = torch.ones((2, 4), dtype=torch.bool)
    desired[1, 0] = False

    cost = load_support_cost_terms(
        forces,
        desired,
        8.0,
        limits,
    )["front_normal_support"]

    # Exact-threshold force is not below the gate, non-X force is ignored,
    # and an undesired front contact never contributes.
    torch.testing.assert_close(cost, torch.tensor([0.75, 0.75]))


def test_linear_front_deficit_keeps_the_same_bound_and_more_near_gate_cost() -> None:
    squared_limits = IsaacRolloutLoadLimits(
        front_normal_min_n=10.0,
        front_normal_deficit_power=2.0,
    )
    linear_limits = IsaacRolloutLoadLimits(
        front_normal_min_n=10.0,
        front_normal_deficit_power=1.0,
    )
    forces = torch.tensor(
        [[[6.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 70.0], [0.0, 0.0, 70.0]]]
    )
    desired = torch.ones((1, 4), dtype=torch.bool)

    squared = load_support_cost_terms(
        forces,
        desired,
        8.0,
        squared_limits,
    )["front_normal_support"]
    linear = load_support_cost_terms(
        forces,
        desired,
        8.0,
        linear_limits,
    )["front_normal_support"]

    torch.testing.assert_close(squared, torch.tensor([0.58]))
    torch.testing.assert_close(linear, torch.tensor([0.7]))
