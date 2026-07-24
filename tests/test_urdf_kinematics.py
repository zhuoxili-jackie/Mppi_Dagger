from __future__ import annotations

import numpy as np

from lateral_mppi_dagger.reference.urdf_kinematics import URDFKinematicTree


BODY_ORDER = (
    "Base_link",
    "FL_hip_link",
    "FR_hip_link",
    "RL_hip_link",
    "RR_hip_link",
    "FL_thigh_link",
    "FR_thigh_link",
    "RL_thigh_link",
    "RR_thigh_link",
    "FL_calf_link",
    "FR_calf_link",
    "RL_calf_link",
    "RR_calf_link",
    "FL_foot_link",
    "FR_foot_link",
    "RL_foot_link",
    "RR_foot_link",
)
JOINT_ORDER = (
    "FL_hip_joint",
    "FR_hip_joint",
    "RL_hip_joint",
    "RR_hip_joint",
    "FL_thigh_joint",
    "FR_thigh_joint",
    "RL_thigh_joint",
    "RR_thigh_joint",
    "FL_calf_joint",
    "FR_calf_joint",
    "RL_calf_joint",
    "RR_calf_joint",
    "FL_foot_joint",
    "FR_foot_joint",
    "RL_foot_joint",
    "RR_foot_joint",
)


def test_urdf_fk_reproduces_reference_first_frame() -> None:
    tree = URDFKinematicTree(
        "vendor/robot_lab/data/Robots/pcbC/pcb_v2_description_0.88/urdf/pcb_v88.urdf"
    )
    with np.load(
        "vendor/robot_lab/data/Motions/pcbc_lateral_708/trajectory_trotting_acc_005.npz",
        allow_pickle=False,
    ) as archive:
        q = np.asarray(archive["joint_pos"][0], dtype=np.float64)
        expected_pos = np.asarray(archive["body_pos_w"][0], dtype=np.float64)
        expected_quat = np.asarray(archive["body_quat_w"][0], dtype=np.float64)
    transforms = tree.forward(
        dict(zip(JOINT_ORDER, q, strict=True)),
        expected_pos[0],
        expected_quat[0],
    )
    actual_pos = np.stack([transforms[name][:3, 3] for name in BODY_ORDER])
    assert np.max(np.abs(actual_pos - expected_pos)) < 1.0e-5
