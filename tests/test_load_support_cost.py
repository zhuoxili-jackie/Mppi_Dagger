from __future__ import annotations

import torch

from lateral_mppi_dagger.env.isaac_mppi_rollout import (
    IsaacRolloutLoadLimits,
    load_support_cost_terms,
)


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
