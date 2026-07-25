from __future__ import annotations

import pytest
import torch

from lateral_mppi_dagger.env.isaac_mppi_rollout import (
    IsaacRolloutLoadLimits,
    base_height_drop_cost,
    load_support_cost_terms,
    rear_leg_position_cost,
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


def test_load_limits_accept_nonnegative_height_guard_stop_frame() -> None:
    limits = IsaacRolloutLoadLimits.from_dict(
        {
            "base_height_drop_stop_frame": 50,
            "lateral_position_start_frame": 50,
        }
    )

    assert limits.base_height_drop_stop_frame == 50.0
    assert limits.lateral_position_start_frame == 50.0


def test_load_limits_reject_negative_scheduled_tracking_frame() -> None:
    with pytest.raises(ValueError, match="Scheduled tracking frames"):
        IsaacRolloutLoadLimits.from_dict(
            {"lateral_position_start_frame": -1}
        )


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
