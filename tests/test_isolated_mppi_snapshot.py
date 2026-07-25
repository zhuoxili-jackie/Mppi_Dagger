from __future__ import annotations

import torch

from lateral_mppi_dagger.env.isaac_mppi_rollout import (
    IsaacRolloutSnapshot,
)
from lateral_mppi_dagger.env.isolated_mppi import (
    snapshot_from_payload,
    snapshot_to_payload,
)


def _snapshot() -> IsaacRolloutSnapshot:
    tensor = torch.arange(6, dtype=torch.float32).reshape(1, 6)
    return IsaacRolloutSnapshot(
        scene_state_relative={
            "articulation": {
                "robot": {
                    "root_pose": tensor,
                }
            }
        },
        action_manager={"_action": tensor},
        action_terms={"joint_pos": {"_raw_actions": tensor}},
        command_buffers={"time_steps": torch.asarray([4])},
        sensor_buffers={"net_forces_w": tensor},
        sensor_clock={"_timestamp": torch.asarray([0.2])},
        previous_executed_action=tensor,
        previous_commanded_action=tensor,
        action_delay_queue=tensor.unsqueeze(0),
        sim_step_counter=17,
        ref_id=8,
        ref_frame=4,
        seed_anchor_pos_local=torch.asarray([1.0, 2.0, 3.0]),
        seed_anchor_quat_w=torch.asarray([1.0, 0.0, 0.0, 0.0]),
    )


def test_isolated_snapshot_payload_round_trip_is_exact() -> None:
    original = _snapshot()

    restored = snapshot_from_payload(
        snapshot_to_payload(original),
        "cpu",
    )

    assert restored.sim_step_counter == 17
    assert restored.ref_id == 8
    assert restored.ref_frame == 4
    assert torch.equal(
        restored.scene_state_relative["articulation"]["robot"]["root_pose"],
        original.scene_state_relative["articulation"]["robot"]["root_pose"],
    )
    assert torch.equal(
        restored.action_delay_queue,
        original.action_delay_queue,
    )
    assert torch.equal(
        restored.seed_anchor_quat_w,
        original.seed_anchor_quat_w,
    )
