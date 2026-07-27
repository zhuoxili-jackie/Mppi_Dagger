from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lateral_mppi_dagger.env.isaac_mppi_rollout import (
    IsaacWholeBodyMPPIProvider,
)


def _provider(
    bias: list[float],
    *,
    start_frame: int = 0,
    ramp_frames: int = 0,
) -> IsaacWholeBodyMPPIProvider:
    provider = object.__new__(IsaacWholeBodyMPPIProvider)
    provider.references = [
        SimpleNamespace(
            frames=3,
            joint_pos=np.zeros((3, 16), dtype=np.float32),
        )
    ]
    provider.config = SimpleNamespace(
        horizon=2,
        reference_action_lookahead_steps=1,
    )
    provider.adapter = SimpleNamespace(
        action_delay_steps=0,
        base=SimpleNamespace(device=torch.device("cpu")),
    )
    provider.rollout = SimpleNamespace(contact_force_threshold=8.0)
    provider.offset = torch.zeros(12)
    provider.scale = torch.ones(12)
    provider.nominal_action_reference_q_des_by_ref = {}
    provider.nominal_action_reference_raw_by_ref = {}
    provider.nominal_joint_position_bias_leg = torch.tensor(bias)
    provider.nominal_joint_position_bias_start_frame = start_frame
    provider.nominal_joint_position_bias_ramp_frames = ramp_frames
    provider.nominal_front_force_feedback_target_n = 0.0
    provider.nominal_front_force_feedback_gain_leg = torch.zeros(12)
    provider.rear_swing_reference_proposal_ref_ids = frozenset()
    provider.rear_swing_reference_proposal_scales = ()
    provider.rear_swing_reference_proposal_joint_mask_leg = torch.zeros(12)
    provider.rear_swing_reference_proposal_lead_steps = 0
    provider.rear_swing_action_residual_lead_steps = 0
    provider.rear_swing_tracking_error_proposal_scales = ()
    provider.rear_swing_tracking_error_proposal_joint_mask_leg = (
        provider.rear_swing_reference_proposal_joint_mask_leg
    )
    provider.rear_swing_tracking_error_proposal_start_frame = 0
    provider.rear_swing_load_transfer_proposal_ref_ids = frozenset()
    provider.rear_swing_load_transfer_proposal_scales = ()
    provider.rear_swing_load_transfer_proposal_gain_by_wheel_leg_rad = (
        torch.zeros((2, 12))
    )
    provider.rear_swing_load_transfer_proposal_start_frame = 0
    provider.rear_swing_load_transfer_proposal_start_frame_by_wheel = (
        0,
        0,
    )
    provider.rear_swing_load_transfer_proposal_gate_mode = (
        "swing_schedule"
    )
    provider.rear_swing_load_transfer_proposal_imbalance_threshold_n = 0.0
    provider.front_support_proposal_ref_ids = frozenset()
    provider.front_support_proposal_scales = ()
    provider.front_support_proposal_gain_leg_rad = torch.zeros(12)
    provider.front_support_proposal_start_frame = 0
    provider.combine_rear_swing_front_support_proposals = False
    provider.combine_rear_swing_load_transfer_front_support_proposals = False
    provider.combine_rear_swing_reference_load_transfer_front_support_proposals = (
        False
    )
    provider.include_rear_support_reference_in_coordinated_proposals = False
    provider.rear_support_reference_proposal_start_frame = 0
    provider.output_front_force_feedback_target_n = 0.0
    provider.output_front_force_feedback_min_contact_n = 0.0
    provider.output_front_force_feedback_lookahead_steps = 1
    provider.output_front_force_feedback_gain_leg = torch.zeros(12)
    provider.output_rear_swing_force_feedback_target_n = 0.0
    provider.output_rear_swing_force_feedback_scale_n = 1.0
    provider.output_rear_swing_force_feedback_lookahead_steps = 1
    provider.output_rear_swing_force_feedback_start_frame = 0
    provider.output_rear_swing_force_feedback_gain_leg = torch.zeros(12)
    provider.output_rear_swing_height_feedback_ref_ids = frozenset()
    provider.output_rear_swing_height_feedback_gain = 0.0
    provider.output_rear_swing_height_feedback_max_abs_rad = 0.0
    provider.output_rear_swing_height_feedback_lookahead_steps = 1
    provider.output_rear_swing_height_feedback_start_frame = 0
    provider.output_rear_support_tracking_feedback_ref_ids = frozenset()
    provider.output_rear_support_tracking_feedback_gain = 0.0
    provider.output_rear_support_tracking_feedback_max_abs_rad = 0.0
    provider.output_rear_support_tracking_feedback_lookahead_steps = 1
    provider.output_rear_support_tracking_feedback_start_frame = 0
    provider.output_pitch_feedback_ref_ids = frozenset()
    provider.output_pitch_feedback_gain_leg = torch.zeros(12)
    provider.output_pitch_feedback_axis = "y"
    provider.output_pitch_feedback_axis_index = 1
    provider.output_pitch_feedback_start_frame = 0
    provider.output_pitch_feedback_max_abs_rad = 0.0
    provider.output_contact_orientation_feedback_ref_ids = frozenset()
    provider.output_contact_orientation_feedback_gain_xyz = torch.zeros(3)
    provider.output_contact_orientation_feedback_start_frame = 0
    provider.output_contact_orientation_feedback_max_endpoint_delta_m = 0.0
    provider.output_contact_orientation_feedback_max_abs_rad = 0.0
    provider.output_joint_position_offset_leg = torch.zeros(12)
    provider.raw_min = torch.full((12,), -1.0)
    provider.raw_max = torch.full((12,), 1.0)
    provider.max_delta = torch.full((12,), 0.2)
    return provider


def test_nominal_action_reference_overrides_only_the_proposal_centre() -> None:
    provider = _provider([0.0] * 12)
    action_reference = torch.arange(
        3 * 12,
        dtype=torch.float32,
    ).reshape(3, 12)
    provider.nominal_action_reference_q_des_by_ref = {
        0: action_reference,
    }

    nominal = provider._nominal(SimpleNamespace(ref_id=0, ref_frame=0))

    torch.testing.assert_close(nominal, action_reference[1:3])
    assert np.array_equal(
        provider.references[0].joint_pos,
        np.zeros((3, 16), dtype=np.float32),
    )


def test_nominal_raw_action_reference_is_returned_bit_exact() -> None:
    provider = _provider([0.0] * 12)
    provider.offset = torch.linspace(-0.7, 0.3, 12)
    provider.scale = torch.linspace(0.1, 0.4, 12)
    raw_reference = torch.linspace(
        -0.8,
        0.8,
        3 * 12,
        dtype=torch.float32,
    ).reshape(3, 12)
    provider.nominal_action_reference_raw_by_ref = {
        0: raw_reference,
    }

    nominal = provider._nominal(SimpleNamespace(ref_id=0, ref_frame=0))

    assert torch.equal(nominal, raw_reference[1:3])


def test_nominal_joint_position_bias_is_applied_in_physical_space() -> None:
    bias = [0.01 * index for index in range(12)]
    provider = _provider(bias)

    nominal = provider._nominal(SimpleNamespace(ref_id=0, ref_frame=0))

    assert nominal.shape == (2, 12)
    torch.testing.assert_close(
        nominal,
        torch.tensor(bias, dtype=torch.float32).repeat(2, 1),
    )


def test_nominal_joint_position_bias_can_ramp_by_reference_frame() -> None:
    bias = [0.1] * 12
    provider = _provider(bias, start_frame=1, ramp_frames=2)

    nominal = provider._nominal(SimpleNamespace(ref_id=0, ref_frame=0))

    assert nominal.shape == (2, 12)
    torch.testing.assert_close(
        nominal,
        torch.tensor([[0.0] * 12, [0.05] * 12], dtype=torch.float32),
    )


def test_nominal_front_force_feedback_is_per_wheel_and_schedule_gated() -> None:
    provider = _provider([0.0] * 12)
    provider.nominal_front_force_feedback_target_n = 8.0
    provider.nominal_front_force_feedback_gain_leg = torch.tensor(
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.04,
            0.04,
            0.0,
            0.0,
            0.02,
            0.02,
            0.0,
            0.0,
        ]
    )
    force = torch.zeros((1, 4, 3), dtype=torch.float32)
    force[0, 0, 0] = -4.0
    force[0, 1, 0] = -8.0
    provider.adapter.contact_sensor = SimpleNamespace(
        data=SimpleNamespace(net_forces_w=force)
    )
    provider.adapter.contact_body_ids = [0, 1, 2, 3]
    provider.adapter.contact_schedules = (
        np.asarray(
            [
                [False, False, True, True],
                [True, True, True, True],
                [False, True, True, True],
            ],
            dtype=bool,
        ),
    )

    nominal = provider._nominal(SimpleNamespace(ref_id=0, ref_frame=0))

    expected = torch.zeros((2, 12), dtype=torch.float32)
    expected[0, 4] = 0.02
    expected[0, 8] = 0.01
    torch.testing.assert_close(nominal, expected)


def test_rear_swing_reference_proposals_are_coherent_and_schedule_gated() -> None:
    provider = _provider([0.0] * 12)
    provider.references[0].joint_pos[:, :12] = 1.0
    provider.rear_swing_reference_proposal_ref_ids = frozenset({0})
    provider.rear_swing_reference_proposal_scales = (0.25, 0.5)
    provider.rear_swing_reference_proposal_joint_mask_leg = torch.tensor(
        [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
        dtype=torch.float32,
    )
    provider.adapter.contact_schedules = (
        np.asarray(
            [
                [True, True, True, True],
                [True, True, False, True],
                [True, True, True, False],
            ],
            dtype=bool,
        ),
    )

    proposals, diagnostics = provider._rear_swing_reference_proposals(
        SimpleNamespace(ref_id=0, ref_frame=0),
        torch.full((2, 12), 0.2),
    )

    assert proposals is not None
    expected_direction = torch.zeros((2, 12))
    expected_direction[0, [2, 6, 10]] = 0.8
    expected_direction[1, [3, 7, 11]] = 0.8
    torch.testing.assert_close(proposals[0], 0.25 * expected_direction)
    torch.testing.assert_close(proposals[1], 0.5 * expected_direction)
    assert diagnostics["rear_swing_step_count"] == [1, 1]
    assert diagnostics["reference_frames"] == [1, 2]
    np.testing.assert_allclose(
        diagnostics["maximum_requested_physical_correction_rad"],
        [0.2, 0.4],
        rtol=0.0,
        atol=1.0e-7,
    )


def test_rear_swing_reference_proposals_disabled_is_exact_noop() -> None:
    provider = _provider([0.0] * 12)

    proposals, diagnostics = provider._rear_swing_reference_proposals(
        SimpleNamespace(ref_id=0, ref_frame=0),
        torch.zeros((2, 12)),
    )

    assert proposals is None
    assert diagnostics["enabled"] is False
    assert diagnostics["configured"] is False


def test_rear_swing_reference_proposals_can_lead_the_schedule() -> None:
    provider = _provider([0.0] * 12)
    provider.references[0] = SimpleNamespace(
        frames=6,
        joint_pos=np.repeat(
            np.arange(6, dtype=np.float32)[:, None],
            16,
            axis=1,
        ),
    )
    provider.rear_swing_reference_proposal_ref_ids = frozenset({0})
    provider.rear_swing_reference_proposal_scales = (0.5,)
    provider.rear_swing_reference_proposal_joint_mask_leg = torch.tensor(
        [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
        dtype=torch.float32,
    )
    provider.rear_swing_reference_proposal_lead_steps = 2
    provider.adapter.contact_schedules = (
        np.asarray(
            [
                [True, True, True, True],
                [True, True, True, True],
                [True, True, True, True],
                [True, True, False, True],
                [True, True, True, False],
                [True, True, True, True],
            ],
            dtype=bool,
        ),
    )

    proposals, diagnostics = provider._rear_swing_reference_proposals(
        SimpleNamespace(ref_id=0, ref_frame=0),
        torch.zeros((2, 12)),
    )

    assert proposals is not None
    expected = torch.zeros((2, 12))
    expected[0, [2, 6, 10]] = 1.5
    expected[1, [2, 6, 10]] = 2.0
    expected[1, [3, 7, 11]] = 2.0
    torch.testing.assert_close(proposals[0], expected)
    assert diagnostics["reference_frames"] == [1, 2]
    assert diagnostics["reference_target_frames"] == [3, 4]
    assert diagnostics["rear_swing_step_count"] == [0, 0]
    assert diagnostics["rear_swing_active_step_count"] == [2, 1]
    assert diagnostics["lead_steps"] == 2


def test_front_support_proposals_are_physical_and_schedule_gated() -> None:
    provider = _provider([0.0] * 12)
    provider.front_support_proposal_ref_ids = frozenset({0})
    provider.front_support_proposal_scales = (0.25, 1.0)
    provider.front_support_proposal_gain_leg_rad = torch.tensor(
        [
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
        ],
        dtype=torch.float32,
    )
    provider.scale = torch.tensor([0.25] * 12)
    provider.adapter.contact_schedules = (
        np.asarray(
            [
                [False, False, True, True],
                [True, False, True, True],
                [True, True, True, True],
            ],
            dtype=bool,
        ),
    )

    proposals, diagnostics = provider._front_support_proposals(
        SimpleNamespace(ref_id=0, ref_frame=0),
        torch.zeros((2, 12)),
    )

    assert proposals is not None
    expected_direction_raw = torch.zeros((2, 12))
    expected_direction_raw[0, [4, 8]] = 0.08
    expected_direction_raw[1, [4, 5, 8, 9]] = 0.08
    torch.testing.assert_close(
        proposals[0],
        0.25 * expected_direction_raw,
    )
    torch.testing.assert_close(proposals[1], expected_direction_raw)
    assert diagnostics["front_support_step_count"] == [2.0, 1.0]
    assert diagnostics["reference_frames"] == [1, 2]
    np.testing.assert_allclose(
        diagnostics["maximum_requested_physical_correction_rad"],
        [0.005, 0.02],
        rtol=0.0,
        atol=1.0e-7,
    )


def test_front_support_proposals_disabled_is_exact_noop() -> None:
    provider = _provider([0.0] * 12)

    proposals, diagnostics = provider._front_support_proposals(
        SimpleNamespace(ref_id=0, ref_frame=0),
        torch.zeros((2, 12)),
    )

    assert proposals is None
    assert diagnostics["enabled"] is False
    assert diagnostics["configured"] is False


def test_front_support_proposals_do_not_change_pre_start_population() -> None:
    provider = _provider([0.0] * 12)
    provider.front_support_proposal_ref_ids = frozenset({0})
    provider.front_support_proposal_scales = (1.0,)
    provider.front_support_proposal_gain_leg_rad[4] = 0.02
    provider.front_support_proposal_start_frame = 40

    proposals, diagnostics = provider._front_support_proposals(
        SimpleNamespace(ref_id=0, ref_frame=39),
        torch.zeros((2, 12)),
    )

    assert proposals is None
    assert diagnostics["configured"] is True
    assert diagnostics["active_for_ref"] is True
    assert diagnostics["active_for_frame"] is False
    assert diagnostics["start_frame"] == 40


def test_coordinated_proposals_pair_rear_swing_and_front_preload() -> None:
    provider = _provider([0.0] * 12)
    provider.references[0].joint_pos[:, :12] = 1.0
    provider.rear_swing_reference_proposal_ref_ids = frozenset({0})
    provider.rear_swing_reference_proposal_scales = (0.5,)
    provider.rear_swing_reference_proposal_joint_mask_leg = torch.tensor(
        [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
        dtype=torch.float32,
    )
    provider.front_support_proposal_ref_ids = frozenset({0})
    provider.front_support_proposal_scales = (0.25, 1.0)
    provider.front_support_proposal_gain_leg_rad = torch.tensor(
        [
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
        ],
        dtype=torch.float32,
    )
    provider.combine_rear_swing_front_support_proposals = True
    provider.scale = torch.full((12,), 0.25)
    provider.adapter.contact_schedules = (
        np.asarray(
            [
                [True, True, True, True],
                [True, False, False, True],
                [True, True, True, True],
            ],
            dtype=bool,
        ),
    )
    request = SimpleNamespace(ref_id=0, ref_frame=0)
    nominal = torch.full((2, 12), 0.8)
    rear_offsets, _ = provider._rear_swing_reference_proposals(
        request,
        nominal,
    )
    front_offsets, _ = provider._front_support_proposals(
        request,
        nominal,
    )

    proposals, diagnostics = (
        provider._coordinated_rear_swing_front_support_proposals(
            request,
            nominal,
            rear_offsets,
            front_offsets,
        )
    )

    assert proposals is not None
    assert proposals.shape == (2, 2, 12)
    expected_rear = torch.zeros((2, 12))
    expected_rear[0, [2, 6, 10]] = 1.6
    expected_front = torch.zeros((2, 12))
    expected_front[0, [4, 8]] = 0.08
    torch.testing.assert_close(
        proposals[0],
        expected_rear + 0.25 * expected_front,
    )
    torch.testing.assert_close(
        proposals[1],
        expected_rear + expected_front,
    )
    assert diagnostics["proposal_count"] == 2
    assert diagnostics["rear_swing_active_step_count"] == 1
    assert diagnostics["scale_pairs"] == [
        {
            "rear_scale": 0.5,
            "front_scale": 0.25,
            "include_rear_support_reference": False,
        },
        {
            "rear_scale": 0.5,
            "front_scale": 1.0,
            "include_rear_support_reference": False,
        },
    ]


def test_rear_swing_tracking_error_proposals_compensate_measured_lag() -> None:
    provider = _provider([0.0] * 12)
    provider.references[0].joint_pos[:, :12] = 1.0
    provider.rear_swing_reference_proposal_ref_ids = frozenset({0})
    provider.rear_swing_reference_proposal_joint_mask_leg = torch.tensor(
        [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
        dtype=torch.float32,
    )
    provider.rear_swing_tracking_error_proposal_scales = (0.25, 1.0)
    provider.rear_swing_tracking_error_proposal_joint_mask_leg = torch.tensor(
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        dtype=torch.float32,
    )
    provider.rear_swing_tracking_error_proposal_start_frame = 1
    provider.scale = torch.full((12,), 0.25)
    provider.adapter.contact_schedules = (
        np.asarray(
            [
                [True, True, True, True],
                [True, True, True, True],
                [True, True, False, True],
            ],
            dtype=bool,
        ),
    )
    request_q = np.full(16, 0.2, dtype=np.float32)
    nominal = torch.zeros((2, 12))

    before, before_diagnostics = (
        provider._rear_swing_tracking_error_proposals(
            SimpleNamespace(ref_id=0, ref_frame=0, q=request_q),
            nominal,
        )
    )
    after, after_diagnostics = (
        provider._rear_swing_tracking_error_proposals(
            SimpleNamespace(ref_id=0, ref_frame=1, q=request_q),
            nominal,
        )
    )

    assert before is None
    assert before_diagnostics["active_for_frame"] is False
    assert after is not None
    assert after.shape == (2, 2, 12)
    expected = torch.zeros((2, 12))
    expected[:, 2] = 3.2
    torch.testing.assert_close(after[1], expected)
    assert after_diagnostics["proposal_count"] == 2
    assert after_diagnostics["rear_swing_active_step_count"] == [2, 0]
    assert (
        after_diagnostics["maximum_joint_tracking_error_rad"]
        == 0.800000011920929
    )


def test_rear_swing_load_transfer_proposals_are_wheel_and_schedule_gated() -> None:
    provider = _provider([0.0] * 12)
    provider.rear_swing_load_transfer_proposal_ref_ids = frozenset({0})
    provider.rear_swing_load_transfer_proposal_scales = (0.5, 1.0)
    provider.rear_swing_load_transfer_proposal_gain_by_wheel_leg_rad = (
        torch.tensor(
            [
                [0.02] + [0.0] * 11,
                [0.0, -0.02, 0.0, 0.0, 0.0, 0.0, 0.0, -0.02]
                + [0.0] * 4,
            ],
            dtype=torch.float32,
        )
    )
    provider.rear_swing_load_transfer_proposal_start_frame = 1
    provider.rear_swing_load_transfer_proposal_start_frame_by_wheel = (
        1,
        1,
    )
    provider.scale = torch.full((12,), 0.25)
    provider.adapter.contact_schedules = (
        np.asarray(
            [
                [True, True, True, True],
                [True, True, False, True],
                [True, True, True, False],
            ],
            dtype=bool,
        ),
    )
    nominal = torch.zeros((2, 12))

    before, before_diagnostics = (
        provider._rear_swing_load_transfer_proposals(
            SimpleNamespace(ref_id=0, ref_frame=0),
            nominal,
        )
    )
    after, after_diagnostics = (
        provider._rear_swing_load_transfer_proposals(
            SimpleNamespace(ref_id=0, ref_frame=1),
            nominal,
        )
    )

    assert before is None
    assert before_diagnostics["active_for_frame"] is False
    assert after is not None
    assert after.shape == (2, 2, 12)
    expected = torch.zeros((2, 12))
    expected[:, [1, 7]] = -0.08
    torch.testing.assert_close(after[1], expected)
    assert after_diagnostics["rear_swing_active_step_count"] == [0, 2]
    assert after_diagnostics["lead_steps"] == 0
    np.testing.assert_allclose(
        after_diagnostics[
            "maximum_requested_physical_correction_rad"
        ],
        [0.01, 0.02],
        rtol=0.0,
        atol=1.0e-7,
    )


def test_rear_swing_load_transfer_proposals_support_wheel_specific_start_frames() -> (
    None
):
    provider = _provider([0.0] * 12)
    provider.rear_swing_load_transfer_proposal_ref_ids = frozenset({0})
    provider.rear_swing_load_transfer_proposal_scales = (1.0,)
    provider.rear_swing_load_transfer_proposal_gain_by_wheel_leg_rad = (
        torch.tensor(
            [
                [0.02] + [0.0] * 11,
                [0.0, -0.02] + [0.0] * 10,
            ],
            dtype=torch.float32,
        )
    )
    provider.rear_swing_load_transfer_proposal_start_frame_by_wheel = (
        2,
        0,
    )
    provider.scale = torch.full((12,), 0.25)
    provider.adapter.contact_schedules = (
        np.asarray(
            [
                [True, True, True, True],
                [True, True, False, False],
                [True, True, False, False],
            ],
            dtype=bool,
        ),
    )
    nominal = torch.zeros((2, 12))

    before, before_diagnostics = (
        provider._rear_swing_load_transfer_proposals(
            SimpleNamespace(ref_id=0, ref_frame=1),
            nominal,
        )
    )
    after, after_diagnostics = (
        provider._rear_swing_load_transfer_proposals(
            SimpleNamespace(ref_id=0, ref_frame=2),
            nominal,
        )
    )

    assert before is not None
    expected_before = torch.zeros((2, 12))
    expected_before[:, 1] = -0.08
    torch.testing.assert_close(before[0], expected_before)
    assert before_diagnostics["active_for_frame_by_wheel"] == [
        False,
        True,
    ]
    assert before_diagnostics["rear_swing_active_step_count"] == [0, 2]

    assert after is not None
    expected_after = torch.zeros((2, 12))
    expected_after[:, 0] = 0.08
    expected_after[:, 1] = -0.08
    torch.testing.assert_close(after[0], expected_after)
    assert after_diagnostics["active_for_frame_by_wheel"] == [
        True,
        True,
    ]
    assert after_diagnostics["rear_swing_active_step_count"] == [2, 2]


def test_rear_swing_load_transfer_proposals_disabled_is_exact_noop() -> None:
    provider = _provider([0.0] * 12)

    proposals, diagnostics = (
        provider._rear_swing_load_transfer_proposals(
            SimpleNamespace(ref_id=0, ref_frame=0),
            torch.zeros((2, 12)),
        )
    )

    assert proposals is None
    assert diagnostics["enabled"] is False
    assert diagnostics["configured"] is False


def test_rear_swing_load_transfer_proposals_can_use_rear_force_imbalance() -> None:
    provider = _provider([0.0] * 12)
    provider.rear_swing_load_transfer_proposal_ref_ids = frozenset({0})
    provider.rear_swing_load_transfer_proposal_scales = (0.5, 1.0)
    provider.rear_swing_load_transfer_proposal_gain_by_wheel_leg_rad = (
        torch.tensor(
            [
                [0.0] * 12,
                [0.0, -0.02, 0.0, 0.0, 0.0, 0.0, 0.0, -0.02]
                + [0.0] * 4,
            ],
            dtype=torch.float32,
        )
    )
    provider.rear_swing_load_transfer_proposal_gate_mode = (
        "rear_force_imbalance"
    )
    provider.rear_swing_load_transfer_proposal_imbalance_threshold_n = 20.0
    provider.scale = torch.full((12,), 0.25)
    provider.adapter.contact_schedules = (
        np.ones((3, 4), dtype=bool),
    )
    provider.adapter.contact_body_ids = [0, 1, 2, 3]
    contact_force = torch.zeros((1, 4, 3), dtype=torch.float32)
    contact_force[0, 2, 2] = 40.0
    contact_force[0, 3, 2] = 100.0
    provider.adapter.contact_sensor = SimpleNamespace(
        data=SimpleNamespace(net_forces_w=contact_force)
    )

    proposals, diagnostics = (
        provider._rear_swing_load_transfer_proposals(
            SimpleNamespace(ref_id=0, ref_frame=0),
            torch.zeros((2, 12)),
        )
    )

    assert proposals is not None
    expected = torch.zeros((2, 12))
    expected[:, [1, 7]] = -0.08
    torch.testing.assert_close(proposals[1], expected)
    assert diagnostics["gate_mode"] == "rear_force_imbalance"
    assert diagnostics["active_for_state"] is True
    assert diagnostics["rear_swing_active_step_count"] == [0, 2]
    assert diagnostics["rear_normal_n"] == [40.0, 100.0]
    assert diagnostics["rear_force_imbalance_by_wheel_n"] == [
        -60.0,
        60.0,
    ]

    contact_force[0, 2, 2] = 90.0
    contact_force[0, 3, 2] = 100.0
    inactive, inactive_diagnostics = (
        provider._rear_swing_load_transfer_proposals(
            SimpleNamespace(ref_id=0, ref_frame=0),
            torch.zeros((2, 12)),
        )
    )
    assert inactive is None
    assert inactive_diagnostics["active_for_state"] is False


def test_coordinated_proposals_disabled_is_exact_noop() -> None:
    provider = _provider([0.0] * 12)

    proposals, diagnostics = (
        provider._coordinated_rear_swing_front_support_proposals(
            SimpleNamespace(ref_id=0, ref_frame=0),
            torch.zeros((2, 12)),
            torch.zeros((1, 2, 12)),
            torch.zeros((1, 2, 12)),
        )
    )

    assert proposals is None
    assert diagnostics["enabled"] is False
    assert diagnostics["configured"] is False


def test_coordinated_load_transfer_proposals_pair_with_front_preload() -> None:
    provider = _provider([0.0] * 12)
    provider.rear_swing_load_transfer_proposal_ref_ids = frozenset({0})
    provider.rear_swing_load_transfer_proposal_scales = (0.5, 1.0)
    provider.rear_swing_load_transfer_proposal_gain_by_wheel_leg_rad = (
        torch.zeros((2, 12), dtype=torch.float32)
    )
    provider.rear_swing_load_transfer_proposal_gain_by_wheel_leg_rad[
        0,
        2,
    ] = -0.02
    provider.front_support_proposal_ref_ids = frozenset({0})
    provider.front_support_proposal_scales = (0.25, 1.0)
    provider.front_support_proposal_gain_leg_rad = torch.zeros(
        12,
        dtype=torch.float32,
    )
    provider.front_support_proposal_gain_leg_rad[4] = 0.02
    provider.combine_rear_swing_load_transfer_front_support_proposals = True
    provider.scale = torch.full((12,), 0.25)
    provider.adapter.contact_schedules = (
        np.asarray(
            [
                [True, True, True, True],
                [True, True, False, True],
                [True, True, True, True],
            ],
            dtype=bool,
        ),
    )
    request = SimpleNamespace(ref_id=0, ref_frame=0)
    nominal = torch.zeros((2, 12))
    load_transfer_offsets, _ = (
        provider._rear_swing_load_transfer_proposals(
            request,
            nominal,
        )
    )
    front_offsets, _ = provider._front_support_proposals(
        request,
        nominal,
    )

    proposals, diagnostics = (
        provider._coordinated_rear_swing_load_transfer_front_support_proposals(
            request,
            nominal,
            load_transfer_offsets,
            front_offsets,
        )
    )

    assert proposals is not None
    assert proposals.shape == (4, 2, 12)
    expected = torch.zeros((4, 2, 12))
    expected[0, 0, 2] = -0.04
    expected[1, 0, 2] = -0.04
    expected[2, 0, 2] = -0.08
    expected[3, 0, 2] = -0.08
    expected[0, 0, 4] = 0.02
    expected[1, 0, 4] = 0.08
    expected[2, 0, 4] = 0.02
    expected[3, 0, 4] = 0.08
    torch.testing.assert_close(proposals, expected)
    assert diagnostics["proposal_count"] == 4
    assert diagnostics["rear_swing_active_step_count"] == [1, 0]
    assert diagnostics["scale_pairs"] == [
        {"load_transfer_scale": 0.5, "front_scale": 0.25},
        {"load_transfer_scale": 0.5, "front_scale": 1.0},
        {"load_transfer_scale": 1.0, "front_scale": 0.25},
        {"load_transfer_scale": 1.0, "front_scale": 1.0},
    ]


def test_coordinated_load_transfer_proposals_disabled_is_exact_noop() -> None:
    provider = _provider([0.0] * 12)

    proposals, diagnostics = (
        provider._coordinated_rear_swing_load_transfer_front_support_proposals(
            SimpleNamespace(ref_id=0, ref_frame=0),
            torch.zeros((2, 12)),
            torch.zeros((1, 2, 12)),
            torch.zeros((1, 2, 12)),
        )
    )

    assert proposals is None
    assert diagnostics["enabled"] is False
    assert diagnostics["configured"] is False


def test_coordinated_reference_load_transfer_front_support_cartesian_product() -> (
    None
):
    provider = _provider([0.0] * 12)
    provider.rear_swing_reference_proposal_scales = (0.1, 0.2)
    provider.rear_swing_load_transfer_proposal_scales = (0.5, 1.0)
    provider.front_support_proposal_scales = (0.25, 1.0)
    provider.combine_rear_swing_reference_load_transfer_front_support_proposals = (
        True
    )
    provider.scale = torch.full((12,), 0.25)
    provider.adapter.contact_schedules = (
        np.asarray(
            [
                [True, True, True, True],
                [True, True, False, True],
                [True, True, True, True],
            ],
            dtype=bool,
        ),
    )
    rear_offsets = torch.zeros((2, 2, 12))
    rear_offsets[0, 0, 6] = 0.4
    rear_offsets[1, 0, 6] = 0.8
    load_transfer_offsets = torch.zeros((2, 2, 12))
    load_transfer_offsets[0, 0, 2] = -0.04
    load_transfer_offsets[1, 0, 2] = -0.08
    front_offsets = torch.zeros((2, 2, 12))
    front_offsets[0, :, 4] = 0.02
    front_offsets[1, :, 4] = 0.08

    proposals, diagnostics = (
        provider._coordinated_rear_swing_reference_load_transfer_front_support_proposals(
            SimpleNamespace(ref_id=0, ref_frame=0),
            torch.zeros((2, 12)),
            rear_offsets,
            load_transfer_offsets,
            front_offsets,
        )
    )

    assert proposals is not None
    assert proposals.shape == (8, 2, 12)
    expected = torch.zeros((8, 2, 12))
    expected[:, 0, 6] = torch.tensor(
        [0.4, 0.4, 0.4, 0.4, 0.8, 0.8, 0.8, 0.8]
    )
    expected[:, 0, 2] = torch.tensor(
        [-0.04, -0.04, -0.08, -0.08] * 2
    )
    expected[:, 0, 4] = torch.tensor(
        [0.02, 0.08, 0.02, 0.08] * 2
    )
    torch.testing.assert_close(proposals, expected)
    assert diagnostics["proposal_count"] == 8
    assert diagnostics["rear_swing_active_step_count"] == [1, 0]
    assert diagnostics["scale_triples"][0] == {
        "rear_scale": 0.1,
        "load_transfer_scale": 0.5,
        "front_scale": 0.25,
    }
    assert diagnostics["scale_triples"][-1] == {
        "rear_scale": 0.2,
        "load_transfer_scale": 1.0,
        "front_scale": 1.0,
    }


def test_coordinated_proposals_can_move_the_supporting_rear_leg() -> None:
    provider = _provider([0.0] * 12)
    provider.references[0].joint_pos[:, :12] = 1.0
    provider.rear_swing_reference_proposal_scales = (0.5,)
    provider.rear_swing_reference_proposal_joint_mask_leg = torch.tensor(
        [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
        dtype=torch.float32,
    )
    provider.front_support_proposal_scales = (0.25,)
    provider.combine_rear_swing_front_support_proposals = True
    provider.include_rear_support_reference_in_coordinated_proposals = True
    provider.scale = torch.full((12,), 0.25)
    provider.adapter.contact_schedules = (
        np.asarray(
            [
                [True, True, True, True],
                [True, True, False, True],
                [True, True, True, True],
            ],
            dtype=bool,
        ),
    )
    nominal = torch.full((2, 12), 0.8)
    rear_offsets = torch.zeros((1, 2, 12))
    rear_offsets[0, 0, [2, 6, 10]] = 1.6

    proposals, diagnostics = (
        provider._coordinated_rear_swing_front_support_proposals(
            SimpleNamespace(ref_id=0, ref_frame=0),
            nominal,
            rear_offsets,
            torch.zeros((1, 2, 12)),
        )
    )

    assert proposals is not None
    assert proposals.shape == (2, 2, 12)
    expected = torch.zeros((2, 12))
    expected[0, [2, 3, 6, 7, 10, 11]] = 1.6
    torch.testing.assert_close(proposals[1], expected)
    assert diagnostics["rear_support_reference_active_step_count"] == [
        0,
        1,
    ]
    np.testing.assert_allclose(
        diagnostics[
            "maximum_rear_support_reference_correction_rad"
        ],
        0.4,
        rtol=0.0,
        atol=1.0e-7,
    )


def test_rear_support_reference_proposals_can_start_after_a_prefix() -> None:
    provider = _provider([0.0] * 12)
    provider.references[0].joint_pos[:, :12] = 1.0
    provider.rear_swing_reference_proposal_scales = (0.5,)
    provider.rear_swing_reference_proposal_joint_mask_leg = torch.tensor(
        [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
        dtype=torch.float32,
    )
    provider.front_support_proposal_scales = (0.25,)
    provider.combine_rear_swing_front_support_proposals = True
    provider.include_rear_support_reference_in_coordinated_proposals = True
    provider.rear_support_reference_proposal_start_frame = 100
    provider.scale = torch.full((12,), 0.25)
    provider.adapter.contact_schedules = (
        np.asarray(
            [
                [True, True, True, True],
                [True, True, False, True],
                [True, True, True, True],
            ],
            dtype=bool,
        ),
    )
    nominal = torch.full((2, 12), 0.8)
    rear_offsets = torch.zeros((1, 2, 12))
    front_offsets = torch.zeros((1, 2, 12))

    before, before_diagnostics = (
        provider._coordinated_rear_swing_front_support_proposals(
            SimpleNamespace(ref_id=0, ref_frame=99),
            nominal,
            rear_offsets,
            front_offsets,
        )
    )
    after, after_diagnostics = (
        provider._coordinated_rear_swing_front_support_proposals(
            SimpleNamespace(ref_id=0, ref_frame=100),
            nominal,
            rear_offsets,
            front_offsets,
        )
    )

    assert before is not None
    assert before.shape == (1, 2, 12)
    assert before_diagnostics["proposal_count"] == 1
    assert (
        before_diagnostics["rear_support_reference_active_for_frame"]
        is False
    )
    assert after is not None
    assert after.shape == (2, 2, 12)
    assert after_diagnostics["proposal_count"] == 2
    assert (
        after_diagnostics["rear_support_reference_active_for_frame"]
        is True
    )


def test_output_front_force_feedback_is_schedule_gated_and_rate_limited() -> None:
    provider = _provider([0.0] * 12)
    provider.output_front_force_feedback_target_n = 8.0
    provider.output_front_force_feedback_gain_leg = torch.tensor(
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.08,
            0.08,
            0.0,
            0.0,
            0.08,
            0.08,
            0.0,
            0.0,
        ]
    )
    force = torch.zeros((1, 4, 3), dtype=torch.float32)
    force[0, 0, 0] = -4.0
    force[0, 1, 0] = -2.0
    provider.adapter.contact_sensor = SimpleNamespace(
        data=SimpleNamespace(net_forces_w=force)
    )
    provider.adapter.contact_body_ids = [0, 1, 2, 3]
    provider.adapter.contact_schedules = (
        np.asarray(
            [
                [True, True, True, True],
                [True, False, True, True],
            ],
            dtype=bool,
        ),
    )
    selected = torch.zeros(12)
    previous = torch.zeros(12)
    provider.max_delta[4] = 0.02

    corrected, diagnostics = provider._apply_output_front_force_feedback(
        selected,
        torch.zeros(12),
        SimpleNamespace(ref_id=0, ref_frame=0),
        previous,
    )

    expected = torch.zeros(12)
    expected[4] = 0.02
    expected[8] = 0.04
    torch.testing.assert_close(corrected, expected)
    assert diagnostics["desired_front_contact"] == [True, False]
    assert diagnostics["force_deficit_fraction"] == [0.5, 0.75]
    assert abs(diagnostics["requested_correction_rad"][4] - 0.04) < 1.0e-7
    assert diagnostics["requested_correction_rad"][5] == 0.0
    assert abs(diagnostics["applied_correction_rad"][4] - 0.02) < 1.0e-7
    assert diagnostics["schedule_frame"] == 1
    assert diagnostics["schedule_lookahead_steps"] == 1


def test_output_front_force_feedback_uses_independent_lookahead() -> None:
    provider = _provider([0.0] * 12)
    provider.output_front_force_feedback_target_n = 8.0
    provider.output_front_force_feedback_lookahead_steps = 2
    provider.output_front_force_feedback_gain_leg = torch.tensor(
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.04,
            0.04,
            0.0,
            0.0,
            0.02,
            0.02,
            0.0,
            0.0,
        ]
    )
    force = torch.zeros((1, 4, 3), dtype=torch.float32)
    provider.adapter.contact_sensor = SimpleNamespace(
        data=SimpleNamespace(net_forces_w=force)
    )
    provider.adapter.contact_body_ids = [0, 1, 2, 3]
    provider.adapter.contact_schedules = (
        np.asarray(
            [
                [True, False, True, True],
                [True, False, True, True],
                [False, True, True, True],
            ],
            dtype=bool,
        ),
    )

    corrected, diagnostics = provider._apply_output_front_force_feedback(
        torch.zeros(12),
        torch.zeros(12),
        SimpleNamespace(ref_id=0, ref_frame=0),
        torch.zeros(12),
    )

    expected = torch.zeros(12)
    expected[5] = 0.04
    expected[9] = 0.02
    torch.testing.assert_close(corrected, expected)
    assert diagnostics["desired_front_contact"] == [False, True]
    assert diagnostics["schedule_frame"] == 2


def test_output_front_force_feedback_disabled_is_exact_noop() -> None:
    provider = _provider([0.0] * 12)
    selected = torch.linspace(-0.4, 0.4, 12)

    corrected, diagnostics = provider._apply_output_front_force_feedback(
        selected,
        torch.zeros(12),
        SimpleNamespace(ref_id=0, ref_frame=0),
        torch.zeros(12),
    )

    assert torch.equal(corrected, selected)
    assert diagnostics["enabled"] is False


def test_output_front_force_feedback_cannot_accumulate_past_nominal_cap() -> None:
    provider = _provider([0.0] * 12)
    provider.output_front_force_feedback_target_n = 8.0
    provider.output_front_force_feedback_gain_leg = torch.tensor(
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.08,
            0.08,
            0.0,
            0.0,
            0.08,
            0.08,
            0.0,
            0.0,
        ]
    )
    force = torch.zeros((1, 4, 3), dtype=torch.float32)
    provider.adapter.contact_sensor = SimpleNamespace(
        data=SimpleNamespace(net_forces_w=force)
    )
    provider.adapter.contact_body_ids = [0, 1, 2, 3]
    provider.adapter.contact_schedules = (
        np.ones((3, 4), dtype=bool),
    )
    selected = torch.zeros(12)
    selected[[4, 5, 8, 9]] = 0.07

    corrected, diagnostics = provider._apply_output_front_force_feedback(
        selected,
        torch.zeros(12),
        SimpleNamespace(ref_id=0, ref_frame=0),
        torch.zeros(12),
    )

    expected = selected.clone()
    expected[[4, 5, 8, 9]] = 0.08
    torch.testing.assert_close(corrected, expected)
    assert max(diagnostics["applied_correction_rad"]) < 0.011


def test_output_front_force_feedback_releases_a_lost_contact() -> None:
    provider = _provider([0.0] * 12)
    provider.output_front_force_feedback_target_n = 8.0
    provider.output_front_force_feedback_min_contact_n = 1.0
    provider.output_front_force_feedback_gain_leg = torch.tensor(
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.04,
            0.04,
            0.0,
            0.0,
            0.02,
            0.02,
            0.0,
            0.0,
        ]
    )
    force = torch.zeros((1, 4, 3), dtype=torch.float32)
    force[0, 1, 0] = -4.0
    provider.adapter.contact_sensor = SimpleNamespace(
        data=SimpleNamespace(net_forces_w=force)
    )
    provider.adapter.contact_body_ids = [0, 1, 2, 3]
    provider.adapter.contact_schedules = (
        np.ones((3, 4), dtype=bool),
    )
    selected = torch.zeros(12)

    corrected, diagnostics = provider._apply_output_front_force_feedback(
        selected,
        torch.zeros(12),
        SimpleNamespace(ref_id=0, ref_frame=0),
        torch.zeros(12),
    )

    expected = torch.zeros(12)
    expected[5] = 0.02
    expected[9] = 0.01
    torch.testing.assert_close(corrected, expected)
    assert diagnostics["measured_front_contact"] == [False, True]


def test_output_front_force_feedback_never_reverses_selected_direction() -> None:
    provider = _provider([0.0] * 12)
    provider.output_front_force_feedback_target_n = 8.0
    provider.output_front_force_feedback_gain_leg = torch.tensor(
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.04,
            0.04,
            0.0,
            0.0,
            0.02,
            0.02,
            0.0,
            0.0,
        ]
    )
    force = torch.zeros((1, 4, 3), dtype=torch.float32)
    provider.adapter.contact_sensor = SimpleNamespace(
        data=SimpleNamespace(net_forces_w=force)
    )
    provider.adapter.contact_body_ids = [0, 1, 2, 3]
    provider.adapter.contact_schedules = (
        np.ones((3, 4), dtype=bool),
    )
    selected = torch.zeros(12)
    selected[[4, 5, 8, 9]] = 0.10

    corrected, _ = provider._apply_output_front_force_feedback(
        selected,
        torch.zeros(12),
        SimpleNamespace(ref_id=0, ref_frame=0),
        torch.zeros(12),
    )

    torch.testing.assert_close(corrected, selected)


def test_output_rear_swing_force_feedback_is_per_wheel_and_rate_limited() -> None:
    provider = _provider([0.0] * 12)
    provider.output_rear_swing_force_feedback_target_n = 8.0
    provider.output_rear_swing_force_feedback_scale_n = 64.0
    provider.output_rear_swing_force_feedback_gain_leg = torch.tensor(
        [
            0.0,
            0.0,
            -0.04,
            -0.04,
            0.0,
            0.0,
            0.08,
            0.08,
            0.0,
            0.0,
            0.02,
            0.02,
        ]
    )
    force = torch.zeros((1, 4, 3), dtype=torch.float32)
    force[0, 2, 2] = -72.0
    force[0, 3, 2] = -22.0
    provider.adapter.contact_sensor = SimpleNamespace(
        data=SimpleNamespace(net_forces_w=force)
    )
    provider.adapter.contact_body_ids = [0, 1, 2, 3]
    provider.adapter.contact_schedules = (
        np.asarray(
            [
                [True, True, True, True],
                [True, True, False, True],
            ],
            dtype=bool,
        ),
    )
    provider.max_delta[6] = 0.03

    corrected, diagnostics = (
        provider._apply_output_rear_swing_force_feedback(
            torch.zeros(12),
            torch.zeros(12),
            SimpleNamespace(ref_id=0, ref_frame=0),
            torch.zeros(12),
        )
    )

    expected = torch.zeros(12)
    expected[2] = -0.04
    expected[6] = 0.03
    expected[10] = 0.02
    torch.testing.assert_close(corrected, expected)
    assert diagnostics["desired_rear_contact"] == [False, True]
    assert diagnostics["force_excess_fraction"] == [1.0, 0.21875]
    assert diagnostics["schedule_frame"] == 1
    assert diagnostics["schedule_lookahead_steps"] == 1


def test_output_rear_swing_force_feedback_uses_independent_lookahead() -> None:
    provider = _provider([0.0] * 12)
    provider.output_rear_swing_force_feedback_target_n = 8.0
    provider.output_rear_swing_force_feedback_scale_n = 32.0
    provider.output_rear_swing_force_feedback_lookahead_steps = 2
    provider.output_rear_swing_force_feedback_gain_leg = torch.tensor(
        [
            0.0,
            0.0,
            -0.04,
            -0.04,
            0.0,
            0.0,
            0.04,
            0.04,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )
    force = torch.zeros((1, 4, 3), dtype=torch.float32)
    force[0, 2:4, 2] = -40.0
    provider.adapter.contact_sensor = SimpleNamespace(
        data=SimpleNamespace(net_forces_w=force)
    )
    provider.adapter.contact_body_ids = [0, 1, 2, 3]
    provider.adapter.contact_schedules = (
        np.asarray(
            [
                [True, True, True, True],
                [True, True, True, True],
                [True, True, True, False],
            ],
            dtype=bool,
        ),
    )

    corrected, diagnostics = (
        provider._apply_output_rear_swing_force_feedback(
            torch.zeros(12),
            torch.zeros(12),
            SimpleNamespace(ref_id=0, ref_frame=0),
            torch.zeros(12),
        )
    )

    expected = torch.zeros(12)
    expected[3] = -0.04
    expected[7] = 0.04
    torch.testing.assert_close(corrected, expected)
    assert diagnostics["desired_rear_contact"] == [True, False]
    assert diagnostics["schedule_frame"] == 2


def test_output_rear_swing_force_feedback_disabled_is_exact_noop() -> None:
    provider = _provider([0.0] * 12)
    selected = torch.linspace(-0.4, 0.4, 12)

    corrected, diagnostics = (
        provider._apply_output_rear_swing_force_feedback(
            selected,
            torch.zeros(12),
            SimpleNamespace(ref_id=0, ref_frame=0),
            torch.zeros(12),
        )
    )

    assert torch.equal(corrected, selected)
    assert diagnostics["enabled"] is False


def test_output_rear_swing_force_feedback_is_exact_noop_before_start() -> None:
    provider = _provider([0.0] * 12)
    provider.output_rear_swing_force_feedback_target_n = 8.0
    provider.output_rear_swing_force_feedback_start_frame = 100
    provider.output_rear_swing_force_feedback_gain_leg[3] = -0.04
    selected = torch.linspace(-0.4, 0.4, 12)

    corrected, diagnostics = (
        provider._apply_output_rear_swing_force_feedback(
            selected,
            torch.zeros(12),
            SimpleNamespace(ref_id=0, ref_frame=99),
            torch.zeros(12),
        )
    )

    assert torch.equal(corrected, selected)
    assert diagnostics["enabled"] is False
    assert diagnostics["configured"] is True
    assert diagnostics["start_frame"] == 100


def test_output_rear_swing_force_feedback_uses_rear_z_normal_axis() -> None:
    provider = _provider([0.0] * 12)
    provider.output_rear_swing_force_feedback_target_n = 8.0
    provider.output_rear_swing_force_feedback_scale_n = 32.0
    provider.output_rear_swing_force_feedback_gain_leg[3] = -0.04
    force = torch.zeros((1, 4, 3), dtype=torch.float32)
    force[0, 3, 0] = -100.0
    force[0, 3, 2] = -24.0
    provider.adapter.contact_sensor = SimpleNamespace(
        data=SimpleNamespace(net_forces_w=force)
    )
    provider.adapter.contact_body_ids = [0, 1, 2, 3]
    provider.adapter.contact_schedules = (
        np.asarray(
            [
                [True, True, True, False],
                [True, True, True, False],
            ],
            dtype=bool,
        ),
    )

    corrected, diagnostics = (
        provider._apply_output_rear_swing_force_feedback(
            torch.zeros(12),
            torch.zeros(12),
            SimpleNamespace(ref_id=0, ref_frame=0),
            torch.zeros(12),
        )
    )

    assert abs(float(corrected[3]) + 0.02) < 1.0e-7
    assert diagnostics["rear_normal_n"] == [0.0, 24.0]
    assert diagnostics["force_excess_fraction"] == [0.0, 0.5]


def test_output_rear_swing_force_feedback_cannot_accumulate_past_nominal_cap() -> None:
    provider = _provider([0.0] * 12)
    provider.output_rear_swing_force_feedback_target_n = 8.0
    provider.output_rear_swing_force_feedback_scale_n = 1.0
    provider.output_rear_swing_force_feedback_gain_leg = torch.tensor(
        [
            0.0,
            0.0,
            -0.08,
            -0.08,
            0.0,
            0.0,
            0.08,
            0.08,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )
    force = torch.zeros((1, 4, 3), dtype=torch.float32)
    force[0, 2:4, 2] = -40.0
    provider.adapter.contact_sensor = SimpleNamespace(
        data=SimpleNamespace(net_forces_w=force)
    )
    provider.adapter.contact_body_ids = [0, 1, 2, 3]
    provider.adapter.contact_schedules = (
        np.asarray(
            [
                [True, True, False, False],
                [True, True, False, False],
            ],
            dtype=bool,
        ),
    )
    selected = torch.zeros(12)
    selected[[2, 3]] = -0.07
    selected[[6, 7]] = 0.07

    corrected, diagnostics = (
        provider._apply_output_rear_swing_force_feedback(
            selected,
            torch.zeros(12),
            SimpleNamespace(ref_id=0, ref_frame=0),
            torch.zeros(12),
        )
    )

    expected = selected.clone()
    expected[[2, 3]] = -0.08
    expected[[6, 7]] = 0.08
    torch.testing.assert_close(corrected, expected)
    assert max(abs(value) for value in diagnostics["applied_correction_rad"]) < 0.011


def test_output_rear_swing_height_feedback_uses_live_jacobian_and_frozen_caps() -> None:
    provider = _provider([0.0] * 12)
    provider.output_rear_swing_height_feedback_ref_ids = frozenset({0})
    provider.output_rear_swing_height_feedback_gain = 1.0
    provider.output_rear_swing_height_feedback_max_abs_rad = 0.12
    force = torch.zeros((1, 4, 3), dtype=torch.float32)
    force[0, 2:4, 2] = -20.0
    provider.adapter.contact_sensor = SimpleNamespace(
        data=SimpleNamespace(net_forces_w=force)
    )
    provider.adapter.contact_body_ids = [0, 1, 2, 3]
    provider.adapter.contact_schedules = (
        np.asarray(
            [
                [True, True, True, True],
                [True, True, False, True],
            ],
            dtype=bool,
        ),
    )

    body_pos_w = torch.zeros((1, 4, 3), dtype=torch.float32)
    body_pos_w[0, 2:4, 2] = 0.10
    target_body_pos = torch.zeros((4, 3), dtype=torch.float32)
    target_body_pos[2:4, 2] = 0.112
    jacobians = torch.zeros((1, 4, 6, 22), dtype=torch.float32)
    jacobians[0, 2, 0, 8] = 1.0
    jacobians[0, 2, 1, 12] = 1.0
    jacobians[0, 2, 2, 16] = 1.0
    robot = SimpleNamespace(
        is_fixed_base=False,
        data=SimpleNamespace(body_pos_w=body_pos_w),
        root_physx_view=SimpleNamespace(
            get_jacobians=lambda: jacobians
        ),
    )
    rollout_base = SimpleNamespace(
        scene=SimpleNamespace(env_origins=torch.zeros((1, 3)))
    )
    provider.rollout = SimpleNamespace(
        contact_force_threshold=8.0,
        ref_wheel_body_ids=[0, 1, 2, 3],
        wheel_body_ids=[0, 1, 2, 3],
        joint_ids=list(range(16)),
        robot=robot,
        base=rollout_base,
        _aligned_reference=(
            lambda _snapshot, _frame: {
                "body_pos_local": target_body_pos,
            }
        ),
    )
    selected = torch.zeros(12)
    selected[10] = 0.10
    previous = selected.clone()
    provider.max_delta[10] = 0.005
    actual_q = np.zeros(16, dtype=np.float32)
    actual_q[10] = 0.20

    corrected, diagnostics = (
        provider._apply_output_rear_swing_height_feedback(
            selected,
            torch.zeros(12),
            SimpleNamespace(
                ref_id=0,
                ref_frame=0,
                q=actual_q,
            ),
            previous,
            object(),
        )
    )

    expected = selected.clone()
    expected[10] = 0.105
    torch.testing.assert_close(corrected, expected)
    assert diagnostics["desired_rear_contact"] == [False, True]
    assert diagnostics["measured_rear_contact"] == [True, True]
    assert diagnostics["stuck_rear_swing"] == [True, False]
    assert diagnostics["height_deficit_m"][0] == pytest.approx(0.012)
    assert diagnostics["jacobian_joint_delta_rad"][10] == pytest.approx(
        0.012,
        abs=1.0e-5,
    )
    assert diagnostics["bounded_joint_target_rad"][10] == pytest.approx(
        0.212,
        abs=1.0e-5,
    )
    assert diagnostics["requested_correction_rad"][10] == pytest.approx(
        0.112,
        abs=1.0e-5,
    )
    assert diagnostics["applied_correction_rad"][10] == pytest.approx(
        0.005
    )
    assert diagnostics["predicted_cartesian_delta_m_by_rear"][0][
        2
    ] == pytest.approx(0.012, abs=1.0e-5)


def test_output_rear_swing_height_feedback_stops_at_frozen_height() -> None:
    provider = _provider([0.0] * 12)
    provider.output_rear_swing_height_feedback_ref_ids = frozenset({0})
    provider.output_rear_swing_height_feedback_gain = 1.0
    provider.output_rear_swing_height_feedback_max_abs_rad = 0.12
    force = torch.zeros((1, 4, 3), dtype=torch.float32)
    force[0, 2, 2] = -20.0
    provider.adapter.contact_sensor = SimpleNamespace(
        data=SimpleNamespace(net_forces_w=force)
    )
    provider.adapter.contact_body_ids = [0, 1, 2, 3]
    provider.adapter.contact_schedules = (
        np.asarray(
            [
                [True, True, True, True],
                [True, True, False, True],
            ],
            dtype=bool,
        ),
    )
    body_pos_w = torch.zeros((1, 4, 3), dtype=torch.float32)
    body_pos_w[0, 2, 2] = 0.112
    target_body_pos = torch.zeros((4, 3), dtype=torch.float32)
    target_body_pos[2, 2] = 0.112
    provider.rollout = SimpleNamespace(
        contact_force_threshold=8.0,
        ref_wheel_body_ids=[0, 1, 2, 3],
        wheel_body_ids=[0, 1, 2, 3],
        robot=SimpleNamespace(
            data=SimpleNamespace(body_pos_w=body_pos_w)
        ),
        base=SimpleNamespace(
            scene=SimpleNamespace(env_origins=torch.zeros((1, 3)))
        ),
        _aligned_reference=(
            lambda _snapshot, _frame: {
                "body_pos_local": target_body_pos,
            }
        ),
    )
    selected = torch.linspace(-0.4, 0.4, 12)

    corrected, diagnostics = (
        provider._apply_output_rear_swing_height_feedback(
            selected,
            torch.zeros(12),
            SimpleNamespace(
                ref_id=0,
                ref_frame=0,
                q=np.zeros(16, dtype=np.float32),
            ),
            selected,
            object(),
        )
    )

    assert torch.equal(corrected, selected)
    assert diagnostics["enabled"] is True
    assert diagnostics["stuck_rear_swing"] == [True, False]
    assert diagnostics["height_deficit_m"][0] == pytest.approx(0.0)
    assert diagnostics["applied_correction_rad"] == [0.0] * 12


def test_output_rear_swing_height_feedback_previews_and_holds_through_swing() -> None:
    provider = _provider([0.0] * 12)
    provider.output_rear_swing_height_feedback_ref_ids = frozenset({0})
    provider.output_rear_swing_height_feedback_gain = 1.0
    provider.output_rear_swing_height_feedback_max_abs_rad = 0.12
    provider.output_rear_swing_height_feedback_lookahead_steps = 2
    force = torch.zeros((1, 4, 3), dtype=torch.float32)
    force[0, 2, 2] = -20.0
    provider.adapter.contact_sensor = SimpleNamespace(
        data=SimpleNamespace(net_forces_w=force)
    )
    provider.adapter.contact_body_ids = [0, 1, 2, 3]
    provider.adapter.contact_schedules = (
        np.asarray(
            [
                [True, True, True, True],
                [True, True, True, True],
                [True, True, False, True],
                [True, True, False, True],
                [True, True, True, True],
            ],
            dtype=bool,
        ),
    )
    body_pos_w = torch.zeros((1, 4, 3), dtype=torch.float32)
    body_pos_w[0, 2, 2] = 0.112
    target_body_pos = torch.zeros((4, 3), dtype=torch.float32)
    target_body_pos[2, 2] = 0.112
    provider.rollout = SimpleNamespace(
        contact_force_threshold=8.0,
        ref_wheel_body_ids=[0, 1, 2, 3],
        wheel_body_ids=[0, 1, 2, 3],
        robot=SimpleNamespace(
            data=SimpleNamespace(body_pos_w=body_pos_w)
        ),
        base=SimpleNamespace(
            scene=SimpleNamespace(env_origins=torch.zeros((1, 3)))
        ),
        _aligned_reference=(
            lambda _snapshot, _frame: {
                "body_pos_local": target_body_pos,
            }
        ),
    )
    selected = torch.linspace(-0.4, 0.4, 12)

    corrected, diagnostics = (
        provider._apply_output_rear_swing_height_feedback(
            selected,
            torch.zeros(12),
            SimpleNamespace(
                ref_id=0,
                ref_frame=2,
                q=np.zeros(16, dtype=np.float32),
            ),
            selected,
            object(),
        )
    )

    assert torch.equal(corrected, selected)
    assert diagnostics["schedule_frame"] == 4
    assert diagnostics["preview_start_frame"] == 2
    assert diagnostics["desired_rear_contact"] == [False, True]
    assert diagnostics["stuck_rear_swing"] == [True, False]
    assert diagnostics["target_frame_by_rear"] == [2, None]


def test_output_rear_swing_height_feedback_disabled_is_exact_noop() -> None:
    provider = _provider([0.0] * 12)
    selected = torch.linspace(-0.4, 0.4, 12)

    corrected, diagnostics = (
        provider._apply_output_rear_swing_height_feedback(
            selected,
            torch.zeros(12),
            SimpleNamespace(
                ref_id=0,
                ref_frame=0,
                q=np.zeros(16, dtype=np.float32),
            ),
            torch.zeros(12),
            object(),
        )
    )

    assert torch.equal(corrected, selected)
    assert diagnostics["enabled"] is False
    assert diagnostics["configured"] is False


def test_output_rear_support_tracking_feedback_is_per_wheel_bounded_and_rate_limited() -> None:
    provider = _provider([0.0] * 12)
    provider.output_rear_support_tracking_feedback_ref_ids = frozenset({0})
    provider.output_rear_support_tracking_feedback_gain = 0.5
    provider.output_rear_support_tracking_feedback_max_abs_rad = 0.04
    provider.references[0].joint_pos[1, [2, 6, 10]] = [
        0.20,
        -0.20,
        0.04,
    ]
    force = torch.zeros((1, 4, 3), dtype=torch.float32)
    force[0, 3, 2] = -20.0
    provider.adapter.contact_sensor = SimpleNamespace(
        data=SimpleNamespace(net_forces_w=force)
    )
    provider.adapter.contact_body_ids = [0, 1, 2, 3]
    provider.adapter.contact_schedules = (
        np.ones((3, 4), dtype=bool),
    )
    provider.max_delta[6] = 0.03

    corrected, diagnostics = (
        provider._apply_output_rear_support_tracking_feedback(
            torch.zeros(12),
            torch.zeros(12),
            SimpleNamespace(
                ref_id=0,
                ref_frame=0,
                q=np.zeros(16, dtype=np.float32),
            ),
            torch.zeros(12),
        )
    )

    expected = torch.zeros(12)
    expected[2] = 0.04
    expected[6] = -0.03
    expected[10] = 0.02
    torch.testing.assert_close(corrected, expected)
    assert diagnostics["desired_rear_contact"] == [True, True]
    assert diagnostics["measured_rear_contact"] == [False, True]
    assert diagnostics["missing_rear_support"] == [True, False]
    assert diagnostics["schedule_frame"] == 1
    assert diagnostics["requested_correction_rad"][2] == pytest.approx(
        0.04
    )


def test_output_rear_support_tracking_feedback_ignores_scheduled_swing() -> None:
    provider = _provider([0.0] * 12)
    provider.output_rear_support_tracking_feedback_ref_ids = frozenset({0})
    provider.output_rear_support_tracking_feedback_gain = 0.5
    provider.output_rear_support_tracking_feedback_max_abs_rad = 0.04
    provider.references[0].joint_pos[1, [2, 6, 10]] = 0.20
    provider.adapter.contact_sensor = SimpleNamespace(
        data=SimpleNamespace(
            net_forces_w=torch.zeros((1, 4, 3), dtype=torch.float32)
        )
    )
    provider.adapter.contact_body_ids = [0, 1, 2, 3]
    provider.adapter.contact_schedules = (
        np.asarray(
            [
                [True, True, True, True],
                [True, True, False, True],
                [True, True, True, True],
            ],
            dtype=bool,
        ),
    )
    selected = torch.linspace(-0.4, 0.4, 12)

    corrected, diagnostics = (
        provider._apply_output_rear_support_tracking_feedback(
            selected,
            torch.zeros(12),
            SimpleNamespace(
                ref_id=0,
                ref_frame=0,
                q=np.zeros(16, dtype=np.float32),
            ),
            selected,
        )
    )

    assert torch.equal(corrected, selected)
    assert diagnostics["missing_rear_support"] == [False, True]
    assert diagnostics["applied_correction_rad"] == [0.0] * 12


def test_output_rear_support_tracking_feedback_disabled_is_exact_noop() -> None:
    provider = _provider([0.0] * 12)
    selected = torch.linspace(-0.4, 0.4, 12)

    corrected, diagnostics = (
        provider._apply_output_rear_support_tracking_feedback(
            selected,
            torch.zeros(12),
            SimpleNamespace(
                ref_id=0,
                ref_frame=0,
                q=np.zeros(16, dtype=np.float32),
            ),
            torch.zeros(12),
        )
    )

    assert torch.equal(corrected, selected)
    assert diagnostics["enabled"] is False
    assert diagnostics["configured"] is False


def test_output_contact_orientation_feedback_uses_scheduled_support_jacobians() -> None:
    provider = _provider([0.0] * 12)
    provider.output_contact_orientation_feedback_ref_ids = frozenset({0})
    provider.output_contact_orientation_feedback_gain_xyz = torch.tensor(
        [0.0, 0.0, 0.5]
    )
    provider.output_contact_orientation_feedback_max_endpoint_delta_m = 0.02
    provider.output_contact_orientation_feedback_max_abs_rad = 0.05
    provider.adapter.contact_schedules = (
        np.asarray(
            [[True, True, False, True]],
            dtype=bool,
        ),
    )
    force = torch.zeros((1, 4, 3), dtype=torch.float32)
    force[0, :, 2] = -20.0
    provider.adapter.contact_sensor = SimpleNamespace(
        data=SimpleNamespace(net_forces_w=force)
    )
    provider.adapter.contact_body_ids = [0, 1, 2, 3]
    body_pos_w = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    jacobians = torch.zeros((1, 4, 6, 22), dtype=torch.float32)
    for wheel_index, joint_indices in enumerate(
        ((0, 4, 8), (1, 5, 9), (2, 6, 10), (3, 7, 11))
    ):
        for axis, joint_index in enumerate(joint_indices):
            jacobians[
                0,
                wheel_index,
                axis,
                joint_index + 6,
            ] = 1.0
    provider.rollout = SimpleNamespace(
        contact_force_threshold=8.0,
        wheel_body_ids=[0, 1, 2, 3],
        joint_ids=list(range(16)),
        robot=SimpleNamespace(
            is_fixed_base=False,
            data=SimpleNamespace(body_pos_w=body_pos_w),
            root_physx_view=SimpleNamespace(
                get_jacobians=lambda: jacobians
            ),
        ),
        command=SimpleNamespace(
            robot_anchor_pos_w=torch.zeros((1, 3))
        ),
    )
    provider.max_delta[4] = 0.01
    angle = 0.10
    actual_quat = torch.tensor(
        [
            np.cos(angle / 2.0),
            0.0,
            0.0,
            np.sin(angle / 2.0),
        ],
        dtype=torch.float32,
    )

    corrected, diagnostics = (
        provider._apply_output_contact_orientation_feedback(
            torch.zeros(12),
            torch.zeros(12),
            SimpleNamespace(ref_id=0, ref_frame=0),
            actual_quat,
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
            torch.zeros(12),
        )
    )

    expected_delta = 0.02 / 1.001
    assert corrected[1].item() == pytest.approx(-expected_delta)
    assert corrected[3].item() == pytest.approx(expected_delta)
    assert corrected[4].item() == pytest.approx(0.01)
    assert corrected[2].item() == pytest.approx(0.0)
    assert corrected[6].item() == pytest.approx(0.0)
    assert corrected[10].item() == pytest.approx(0.0)
    assert diagnostics["desired_support"] == [True, True, False, True]
    assert diagnostics["measured_contact"] == [True] * 4
    assert diagnostics["desired_endpoint_delta_world_m"][0] == pytest.approx(
        [0.0, 0.02, 0.0]
    )
    assert diagnostics["desired_endpoint_delta_world_m"][1] == pytest.approx(
        [-0.02, 0.0, 0.0]
    )
    assert diagnostics["desired_endpoint_delta_world_m"][2] == [
        0.0,
        0.0,
        0.0,
    ]
    assert diagnostics["requested_correction_rad"][4] == pytest.approx(
        expected_delta
    )
    assert diagnostics["applied_correction_rad"][4] == pytest.approx(0.01)


def test_output_contact_orientation_feedback_disabled_is_exact_noop() -> None:
    provider = _provider([0.0] * 12)
    selected = torch.linspace(-0.4, 0.4, 12)

    corrected, diagnostics = (
        provider._apply_output_contact_orientation_feedback(
            selected,
            torch.zeros(12),
            SimpleNamespace(ref_id=0, ref_frame=0),
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
            torch.zeros(12),
        )
    )

    assert torch.equal(corrected, selected)
    assert diagnostics["enabled"] is False
    assert diagnostics["configured"] is False


def test_output_pitch_feedback_is_signed_bounded_and_rate_limited() -> None:
    provider = _provider([0.0] * 12)
    provider.output_pitch_feedback_ref_ids = frozenset({0})
    provider.output_pitch_feedback_gain_leg[7] = 0.5
    provider.output_pitch_feedback_max_abs_rad = 0.05
    provider.max_delta[7] = 0.03
    angle = -0.20
    actual_quat = torch.tensor(
        [
            np.cos(angle / 2.0),
            0.0,
            np.sin(angle / 2.0),
            0.0,
        ],
        dtype=torch.float32,
    )

    corrected, diagnostics = provider._apply_output_pitch_feedback(
        torch.zeros(12),
        torch.zeros(12),
        0,
        actual_quat,
        torch.tensor([1.0, 0.0, 0.0, 0.0]),
        torch.zeros(12),
    )

    expected = torch.zeros(12)
    expected[7] = -0.03
    torch.testing.assert_close(corrected, expected)
    assert abs(diagnostics["signed_pitch_error_rad"] - angle) < 1.0e-6
    assert (
        abs(diagnostics["requested_correction_rad"][7] + 0.05)
        < 1.0e-7
    )
    assert (
        abs(diagnostics["applied_correction_rad"][7] + 0.03)
        < 1.0e-7
    )


def test_output_pitch_feedback_supports_bounded_roll_axis_after_start() -> None:
    provider = _provider([0.0] * 12)
    provider.output_pitch_feedback_ref_ids = frozenset({0})
    provider.output_pitch_feedback_gain_leg[1] = -0.1
    provider.output_pitch_feedback_axis = "x"
    provider.output_pitch_feedback_axis_index = 0
    provider.output_pitch_feedback_start_frame = 100
    provider.output_pitch_feedback_max_abs_rad = 0.025
    angle = 0.20
    actual_quat = torch.tensor(
        [
            np.cos(angle / 2.0),
            np.sin(angle / 2.0),
            0.0,
            0.0,
        ],
        dtype=torch.float32,
    )
    selected = torch.zeros(12)

    before_start, before_diagnostics = (
        provider._apply_output_pitch_feedback(
            selected,
            torch.zeros(12),
            0,
            actual_quat,
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
            torch.zeros(12),
            ref_frame=99,
        )
    )
    corrected, diagnostics = provider._apply_output_pitch_feedback(
        selected,
        torch.zeros(12),
        0,
        actual_quat,
        torch.tensor([1.0, 0.0, 0.0, 0.0]),
        torch.zeros(12),
        ref_frame=100,
    )

    assert torch.equal(before_start, selected)
    assert before_diagnostics["started"] is False
    expected = torch.zeros(12)
    expected[1] = -0.02
    torch.testing.assert_close(corrected, expected)
    assert diagnostics["feedback_axis"] == "x"
    assert abs(
        diagnostics["signed_orientation_axis_error_rad"] - angle
    ) < 1.0e-6
    assert diagnostics["signed_pitch_error_rad"] == 0.0


def test_output_pitch_feedback_is_reference_gated_exact_noop() -> None:
    provider = _provider([0.0] * 12)
    provider.output_pitch_feedback_ref_ids = frozenset({0})
    provider.output_pitch_feedback_gain_leg[7] = 0.25
    provider.output_pitch_feedback_max_abs_rad = 0.05
    selected = torch.linspace(-0.2, 0.2, 12)

    corrected, diagnostics = provider._apply_output_pitch_feedback(
        selected,
        torch.zeros(12),
        1,
        torch.tensor([1.0, 0.0, 0.0, 0.0]),
        torch.tensor([1.0, 0.0, 0.0, 0.0]),
        torch.zeros(12),
    )

    assert torch.equal(corrected, selected)
    assert diagnostics["configured"] is True
    assert diagnostics["active_for_ref"] is False
    assert diagnostics["enabled"] is False


def test_output_pitch_feedback_cannot_accumulate_past_nominal_cap() -> None:
    provider = _provider([0.0] * 12)
    provider.output_pitch_feedback_ref_ids = frozenset({0})
    provider.output_pitch_feedback_gain_leg[7] = 0.25
    provider.output_pitch_feedback_max_abs_rad = 0.05
    angle = -0.20
    actual_quat = torch.tensor(
        [
            np.cos(angle / 2.0),
            0.0,
            np.sin(angle / 2.0),
            0.0,
        ],
        dtype=torch.float32,
    )
    selected = torch.zeros(12)
    selected[7] = -0.06

    corrected, diagnostics = provider._apply_output_pitch_feedback(
        selected,
        torch.zeros(12),
        0,
        actual_quat,
        torch.tensor([1.0, 0.0, 0.0, 0.0]),
        selected,
    )

    assert torch.equal(corrected, selected)
    assert diagnostics["applied_correction_rad"][7] == 0.0
    assert abs(
        diagnostics["absolute_feedback_limit_rad"][7] + 0.05
    ) < 1.0e-7


def test_output_joint_position_offset_is_physical_and_rate_limited() -> None:
    provider = _provider([0.0] * 12)
    provider.output_joint_position_offset_leg = torch.tensor(
        [
            0.0,
            0.0,
            0.0,
            0.0,
            -0.025,
            -0.005,
            0.0,
            0.0,
            0.020,
            0.015,
            0.0,
            0.0,
        ]
    )
    provider.max_delta[4] = 0.01

    corrected, diagnostics = provider._apply_output_joint_position_offset(
        torch.zeros(12),
        torch.zeros(12),
    )

    expected = provider.output_joint_position_offset_leg.clone()
    expected[4] = -0.01
    torch.testing.assert_close(corrected, expected)
    assert diagnostics["enabled"] is True
    assert (
        abs(diagnostics["applied_correction_rad"][4] + 0.01)
        < 1.0e-7
    )


def test_output_joint_position_offset_disabled_is_exact_noop() -> None:
    provider = _provider([0.0] * 12)
    selected = torch.linspace(-0.2, 0.2, 12)

    corrected, diagnostics = provider._apply_output_joint_position_offset(
        selected,
        torch.zeros(12),
    )

    assert torch.equal(corrected, selected)
    assert diagnostics["enabled"] is False
