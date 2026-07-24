# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Configuration for pcbB robots."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from robot_lab.assets import ISAACLAB_ASSETS_DATA_DIR


pcbB_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=True,
        replace_cylinders_with_capsules=False,
        asset_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/pcbB/pcbv2_description/urdf/pcbv2.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.45),
        joint_pos={
            ".*L_hip_joint": 0.0,
            ".*R_hip_joint": 0.0,
            "F.*_thigh_joint": 0.81,
            "R.*_thigh_joint": -0.81,
            "F.*_calf_joint": -1.535,
            "R.*_calf_joint": 1.535,
            ".*_wheel_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs_hip": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_joint"],
            effort_limit_sim=48.0,
            velocity_limit_sim=28.051,
            stiffness=35.0,
            damping=0.8,
            armature=0.01594,
            friction=0.0,
        ),
        "legs_thigh": ImplicitActuatorCfg(
            joint_names_expr=[".*_thigh_joint"],
            effort_limit_sim=75.0,
            velocity_limit_sim=15.7,
            stiffness=35.0,
            damping=0.8,
            armature=0.01594,
            friction=0.0,
        ),
        "legs_calf": ImplicitActuatorCfg(
            joint_names_expr=[".*_calf_joint"],
            effort_limit_sim=75.0,
            velocity_limit_sim=15.7,
            stiffness=35.0,
            damping=0.8,
            armature=0.01594,
            friction=0.0,
        ),
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=[".*_wheel_joint"],
            effort_limit_sim=30.0,
            velocity_limit_sim=31.0,
            stiffness=0.0,
            damping=0.6,
            armature=0.0005103,
            friction=0.0,
        ),
    },
)
