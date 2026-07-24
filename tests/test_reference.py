from __future__ import annotations

import numpy as np
import pytest

from lateral_mppi_dagger.config import load_yaml
from lateral_mppi_dagger.reference.contact_schedule import infer_contact_schedule
from lateral_mppi_dagger.reference.interpolation import assert_compatible_timebase, interpolate_reference
from lateral_mppi_dagger.reference.loader import ReferenceSet


def test_all_708_references_validate_and_share_first_frame() -> None:
    references = ReferenceSet.from_config()
    assert len(references) == 7
    assert all(motion.frames == 332 and motion.fps == 50 for motion in references.motions)
    assert all(abs(motion.duration_seconds - 6.62) < 1.0e-12 for motion in references.motions)
    for motion in references.motions[1:]:
        np.testing.assert_array_equal(motion.joint_pos[0], references.fixed_motion.joint_pos[0])


def test_derived_standing_reference_matches_isaac_command_semantics() -> None:
    references = ReferenceSet.from_config()
    moving = references[0]
    standing = references[6]
    assert standing.source_kind == "derived_standing_first_frame"
    assert standing.target_vy == 0.0
    assert len(standing.sha256) == 64
    np.testing.assert_array_equal(
        standing.joint_pos,
        np.broadcast_to(moving.joint_pos[0], standing.joint_pos.shape),
    )
    np.testing.assert_array_equal(
        standing.body_quat_w,
        np.broadcast_to(moving.body_quat_w[0], standing.body_quat_w.shape),
    )
    np.testing.assert_array_equal(standing.joint_vel, np.zeros_like(standing.joint_vel))
    np.testing.assert_array_equal(
        standing.body_lin_vel_w,
        np.zeros_like(standing.body_lin_vel_w),
    )
    np.testing.assert_array_equal(
        standing.body_ang_vel_w,
        np.zeros_like(standing.body_ang_vel_w),
    )


def test_low_load_reference_is_small_stride_frequency_scaled_crawl() -> None:
    references = ReferenceSet.from_config(
        "configs/reference_low_load_v1.yaml"
    )
    assert len(references) == 9
    assert references[8].target_vy == 0.0
    negative_slow = references[1]
    positive_slow = references[6]
    assert negative_slow.target_vy == -0.03
    assert positive_slow.target_vy == 0.03
    assert {
        motion.target_vy for motion in references.motions[:8]
    } == {-0.06, -0.03, -0.024, -0.012, 0.012, 0.024, 0.03, 0.06}
    for motion in references.motions[:8]:
        front_detachment = np.max(
            motion.body_pos_w[0, 13:15, 0]
            - motion.body_pos_w[:, 13:15, 0]
        )
        rear_clearance = np.max(
            motion.body_pos_w[:, 15:17, 2]
            - motion.body_pos_w[0, 15:17, 2]
        )
        assert front_detachment <= 0.0081
        assert rear_clearance <= 0.0121


def test_low_load_reference_fits_student_joint_specific_envelope() -> None:
    references = ReferenceSet.from_config(
        "configs/reference_low_load_v1.yaml"
    )
    student = load_yaml("configs/student.yaml")
    limits = np.asarray(
        student["physical_target_abs_limit_rad_by_joint"],
        dtype=np.float32,
    )
    initial = references[0].joint_pos[0, :12]
    for motion in references.motions[:8]:
        deviation = np.max(
            np.abs(motion.joint_pos[:, :12] - initial),
            axis=0,
        )
        assert np.all(deviation <= limits + 1.0e-5)


def test_timebase_requires_one_reference_frame_per_control_step() -> None:
    assert_compatible_timebase(0.02, 50)
    with pytest.raises(ValueError, match="REFERENCE_TIMEBASE_ERROR"):
        assert_compatible_timebase(0.01, 50)


def test_interpolation_and_per_wheel_contact_schedule() -> None:
    motion = ReferenceSet.from_config()[0]
    sample = interpolate_reference(motion, 0.015)
    assert sample["joint_pos"].shape == (16,)
    quaternion_norm = np.linalg.norm(sample["body_quat_w"], axis=-1)
    np.testing.assert_allclose(quaternion_norm, np.ones(17), atol=1.0e-5)
    schedule = infer_contact_schedule(motion)
    assert schedule.shape == (332, 4)
    assert schedule.dtype == np.uint8
