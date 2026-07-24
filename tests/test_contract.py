from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from lateral_mppi_dagger.config import load_yaml
from lateral_mppi_dagger.contract.action16 import Action16Adapter, ActionContract, SafetyShield
from lateral_mppi_dagger.contract.joint_mapping import policy_to_runtime, runtime_to_policy
from lateral_mppi_dagger.contract.obs93 import (
    MotionPrefixSemantics,
    Obs93Builder,
    Obs93Input,
    relative_rotation_6d_columns,
)


def contract() -> dict:
    return load_yaml("configs/deployment_contract.yaml")


def observation_input(previous: torch.Tensor | None = None) -> Obs93Input:
    quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    return Obs93Input(
        robot_anchor_quat_wxyz=quaternion,
        reference_anchor_quat_wxyz=quaternion,
        base_ang_vel_b=torch.tensor([[0.1, 0.2, 0.3]]),
        joint_pos=torch.arange(16, dtype=torch.float32).unsqueeze(0),
        joint_vel=torch.arange(16, dtype=torch.float32).unsqueeze(0) * 0.1,
        default_joint_pos=torch.zeros((1, 16)),
        default_joint_vel=torch.zeros((1, 16)),
        previous_executed_raw_action=torch.zeros((1, 16)) if previous is None else previous,
        velocity_command=torch.tensor([[0.0, -0.15, 0.0]]),
        reference_joint_pos=torch.ones((1, 16)),
        reference_joint_vel=torch.ones((1, 16)) * 2.0,
    )


def test_obs93_shape_order_wheel_slots_and_previous_action() -> None:
    fixed_pos = torch.arange(16, dtype=torch.float32)
    previous = torch.arange(16, dtype=torch.float32).unsqueeze(0) / 10.0
    builder = Obs93Builder(
        fixed_pos,
        torch.zeros(16),
        torch.tensor([1.0, 0.0, 0.0, 0.0]),
        MotionPrefixSemantics.FIXED_FIRST_FRAME,
    )
    observation = builder.build(observation_input(previous))
    assert observation.shape == (1, 93)
    assert observation.dtype == torch.float32
    torch.testing.assert_close(observation[0, :16], fixed_pos)
    torch.testing.assert_close(observation[0, 53:57], torch.zeros(4))
    torch.testing.assert_close(observation[0, 73:89], previous[0])
    torch.testing.assert_close(observation[0, 89:92], torch.tensor([0.0, -0.15, 0.0]))
    assert observation[0, 92].item() == 0.0


def test_rotation_6d_uses_deployment_matrix_columns() -> None:
    robot = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    reference = torch.tensor([[math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)]])
    actual = relative_rotation_6d_columns(robot, reference)
    expected = torch.tensor([[0.0, -1.0, 1.0, 0.0, 0.0, 0.0]])
    torch.testing.assert_close(actual, expected, atol=1.0e-6, rtol=0.0)
    row_major_rows = torch.tensor([[0.0, -1.0, 0.0, 1.0, 0.0, 0.0]])
    assert not torch.allclose(actual, row_major_rows)


def test_rotation_6d_identity_matches_key7_deployment_layout() -> None:
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    actual = relative_rotation_6d_columns(identity, identity)
    torch.testing.assert_close(
        actual,
        torch.tensor([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]]),
        atol=0.0,
        rtol=0.0,
    )


def test_dynamic_prefix_changes_original_32_values_without_changing_shape() -> None:
    fixed = Obs93Builder(
        torch.zeros(16),
        torch.zeros(16),
        torch.tensor([1.0, 0.0, 0.0, 0.0]),
        "fixed_first_frame",
    ).build(observation_input())
    dynamic = Obs93Builder(
        torch.zeros(16),
        torch.zeros(16),
        torch.tensor([1.0, 0.0, 0.0, 0.0]),
        "dynamic_reference",
    ).build(observation_input())
    assert fixed.shape == dynamic.shape == (1, 93)
    assert not torch.equal(fixed[:, :32], dynamic[:, :32])
    torch.testing.assert_close(fixed[:, 38:], dynamic[:, 38:])


def test_joint_mapping_scatter_and_round_trip_with_unique_sentinel() -> None:
    policy = np.arange(16, dtype=np.int64)
    runtime = policy_to_runtime(policy)
    np.testing.assert_array_equal(
        runtime,
        np.asarray([0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11, 12, 13, 14, 15]),
    )
    np.testing.assert_array_equal(runtime_to_policy(runtime), policy)


def test_hard_zero_action_adapter_and_safety_history() -> None:
    values = contract()
    action_contract = ActionContract.from_dict(values)
    adapter = Action16Adapter(action_contract)
    q_des = np.asarray(values["action"]["q_action_offset_runtime"], dtype=np.float32)[:12]
    action = adapter.physical_to_raw(q_des, np.full(4, 123.0, dtype=np.float32))
    np.testing.assert_array_equal(action, np.zeros(16, dtype=np.float32))
    shield = SafetyShield(action_contract)
    proposed = np.arange(16, dtype=np.float32)
    executed, info = shield.apply(proposed)
    np.testing.assert_array_equal(executed[12:], np.zeros(4, dtype=np.float32))
    assert info.intervened
    np.testing.assert_array_equal(shield.last_action, executed)
    shield.reset()
    np.testing.assert_array_equal(shield.last_action, np.zeros(16, dtype=np.float32))


def test_reset_and_action_offsets_are_separately_named_and_equal_by_evidence() -> None:
    values = contract()
    np.testing.assert_allclose(
        values["reset"]["q_reset_ref0"],
        values["action"]["q_action_offset_runtime"],
        atol=0.0,
        rtol=0.0,
    )


def test_wrong_reference_hash_is_refused() -> None:
    values = contract()
    assert len(values["motion_prefix"]["reference_sha256"]) == 64
    assert values["source_evidence"]["legacy_graphs_present"] is False
