from __future__ import annotations

from types import SimpleNamespace

import numpy as np
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
    provider.offset = torch.zeros(12)
    provider.scale = torch.ones(12)
    provider.nominal_action_reference_q_des_by_ref = {}
    provider.nominal_action_reference_raw_by_ref = {}
    provider.nominal_joint_position_bias_leg = torch.tensor(bias)
    provider.nominal_joint_position_bias_start_frame = start_frame
    provider.nominal_joint_position_bias_ramp_frames = ramp_frames
    provider.nominal_front_force_feedback_target_n = 0.0
    provider.nominal_front_force_feedback_gain_leg = torch.zeros(12)
    provider.output_front_force_feedback_target_n = 0.0
    provider.output_front_force_feedback_min_contact_n = 0.0
    provider.output_front_force_feedback_lookahead_steps = 1
    provider.output_front_force_feedback_gain_leg = torch.zeros(12)
    provider.output_pitch_feedback_ref_ids = frozenset()
    provider.output_pitch_feedback_gain_leg = torch.zeros(12)
    provider.output_pitch_feedback_max_abs_rad = 0.0
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
