from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lateral_mppi_dagger.env.isaac_adapter import (
    synchronize_motion_command_reference_bank,
)
from lateral_mppi_dagger.reference.loader import ReferenceSet


def _fake_command(references: ReferenceSet) -> SimpleNamespace:
    frames = references.fixed_motion.frames
    bodies = len(references.body_order)
    return SimpleNamespace(
        cfg=SimpleNamespace(
            body_names=list(references.body_order),
            standing_probability=0.20,
        ),
        motion=SimpleNamespace(time_step_total=frames),
        motion_ids=torch.full((3,), 6, dtype=torch.long),
        time_steps=torch.full((3,), 17, dtype=torch.long),
        _target_lateral_velocities=torch.ones(3),
        _joint_pos_refs=torch.zeros((7, frames, 16)),
        _joint_vel_refs=torch.zeros((7, frames, 16)),
        _body_pos_refs=torch.zeros((7, frames, bodies, 3)),
        _body_quat_refs=torch.zeros((7, frames, bodies, 4)),
        _body_lin_vel_refs=torch.zeros((7, frames, bodies, 3)),
        _body_ang_vel_refs=torch.zeros((7, frames, bodies, 3)),
        _reference_lateral_velocities=torch.zeros(7),
        _moving_motion_count=6,
        motion_anchor_body_index=0,
        num_envs=3,
    )


def test_low_load_reference_bank_replaces_legacy_ids_without_modulo() -> None:
    references = ReferenceSet.from_config(
        "configs/low_load_lateral/train_001/reference.yaml"
    )
    command = _fake_command(references)

    bridge = synchronize_motion_command_reference_bank(command, references)

    assert bridge == {
        "semantics": "standalone_reference_set_replaces_isaac_runtime_bank",
        "legacy_reference_count": 7,
        "active_reference_count": 9,
        "moving_reference_count": 8,
        "standing_reference_id": 8,
        "id_mapping": "identity_after_runtime_bank_replacement",
    }
    assert command._moving_motion_count == 8
    assert command._joint_pos_refs.shape == (9, 332, 16)
    assert command._body_pos_refs.shape == (9, 332, 17, 3)
    torch.testing.assert_close(
        command._reference_lateral_velocities,
        torch.tensor(
            [-0.060, -0.030, -0.024, -0.012, 0.012, 0.024, 0.030, 0.060, 0.0]
        ),
    )
    for ref_id in range(9):
        np.testing.assert_array_equal(
            command._joint_pos_refs[ref_id].cpu().numpy(),
            references[ref_id].joint_pos,
        )
    torch.testing.assert_close(
        command._actor_joint_pos[0],
        command._joint_pos_refs[0, 0],
    )
    assert torch.equal(command.motion_ids, torch.zeros_like(command.motion_ids))
    assert torch.equal(command.time_steps, torch.zeros_like(command.time_steps))
    assert torch.equal(
        command._target_lateral_velocities,
        torch.zeros_like(command._target_lateral_velocities),
    )


def test_explicit_standing_reference_keeps_identity_id_semantics() -> None:
    references = ReferenceSet.from_config(
        "configs/low_load_lateral/train_001/reference.yaml"
    )
    command = _fake_command(references)

    bridge = synchronize_motion_command_reference_bank(command, references)

    assert bridge["active_reference_count"] == 9
    assert bridge["moving_reference_count"] == 8
    assert bridge["standing_reference_id"] == 8
    assert command._moving_motion_count == 8
    torch.testing.assert_close(
        command._joint_pos_refs[8],
        torch.as_tensor(references[8].joint_pos),
    )


def test_reference_bridge_rejects_body_order_mismatch() -> None:
    references = ReferenceSet.from_config(
        "configs/low_load_lateral/train_001/reference.yaml"
    )
    command = _fake_command(references)
    command.cfg.body_names = list(reversed(command.cfg.body_names))

    with pytest.raises(RuntimeError, match="body order"):
        synchronize_motion_command_reference_bank(command, references)
