from __future__ import annotations

import torch

from lateral_mppi_dagger.env.state_snapshot import StateSnapshot


def test_state_snapshot_clone_round_trip() -> None:
    snapshot = StateSnapshot(
        root_state_w=torch.randn(2, 13),
        # Start the mutated field at an exactly representable value so the
        # asserted delta is not sensitive to float32 rounding of x + 1.
        joint_pos=torch.zeros(2, 16),
        joint_vel=torch.randn(2, 16),
        previous_executed_action=torch.randn(2, 16),
        reference_frame=torch.tensor([1, 2]),
        reference_id=torch.tensor([0, 1]),
        contact_history=torch.zeros(2, 3, 4, dtype=torch.bool),
    )
    cloned = snapshot.clone()
    assert snapshot.max_abs_difference(cloned) == 0.0
    cloned.joint_pos[0, 0] += 1.0
    assert snapshot.max_abs_difference(cloned) == 1.0
