from __future__ import annotations

import pytest
import torch

from lateral_mppi_dagger.env.isaac_mppi_rollout import (
    IsaacRolloutLoadLimits,
    scheduled_action_residual_cost,
)


def test_scheduled_action_residual_default_is_exact_legacy_mean() -> None:
    residual = torch.arange(24, dtype=torch.float32).reshape(2, 12) / 10.0
    desired_contact = torch.tensor(
        (
            (True, True, False, True),
            (True, True, True, False),
        )
    )

    actual = scheduled_action_residual_cost(
        residual,
        desired_contact,
        rear_swing_multiplier=1.0,
    )

    torch.testing.assert_close(
        actual,
        residual.square().mean(dim=-1),
        rtol=0.0,
        atol=0.0,
    )


def test_scheduled_action_residual_scales_only_the_swing_rear_leg() -> None:
    residual = torch.ones((2, 12), dtype=torch.float32)
    residual[0, (2, 6, 10)] = 2.0
    residual[0, (3, 7, 11)] = 3.0
    residual[1, (2, 6, 10)] = 4.0
    residual[1, (3, 7, 11)] = 5.0
    desired_contact = torch.tensor(
        (
            (True, True, False, True),
            (True, True, True, False),
        )
    )

    actual = scheduled_action_residual_cost(
        residual,
        desired_contact,
        rear_swing_multiplier=0.1,
    )
    expected = torch.tensor(
        (
            (6.0 + 3.0 * 4.0 * 0.1 + 3.0 * 9.0) / 12.0,
            (6.0 + 3.0 * 16.0 + 3.0 * 25.0 * 0.1) / 12.0,
        )
    )

    torch.testing.assert_close(actual, expected)


def test_scheduled_action_residual_accepts_a_leading_swing_mask() -> None:
    residual = torch.ones((1, 12), dtype=torch.float32)
    desired_contact = torch.ones((1, 4), dtype=torch.bool)

    actual = scheduled_action_residual_cost(
        residual,
        desired_contact,
        rear_swing_multiplier=0.0,
        rear_swing_active=torch.tensor(((True, False),)),
    )

    torch.testing.assert_close(actual, torch.tensor((0.75,)))


@pytest.mark.parametrize("value", (-0.1, 1.1, float("nan")))
def test_scheduled_action_residual_rejects_invalid_multiplier(
    value: float,
) -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        scheduled_action_residual_cost(
            torch.zeros((1, 12)),
            torch.ones((1, 4), dtype=torch.bool),
            rear_swing_multiplier=value,
        )
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        IsaacRolloutLoadLimits.from_dict(
            {"rear_swing_action_residual_multiplier": value}
        )
