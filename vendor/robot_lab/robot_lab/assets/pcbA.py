# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Configuration for Unitree robots.
Reference: https://github.com/unitreerobotics/unitree_ros
"""

import isaaclab.sim as sim_utils
# DelayedPDActuatorCfg, DCMotorCfg, ImplicitActuatorCfg 延迟非理想，执行器基础电机-扭矩，理想-隐式PD
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from robot_lab.assets import ISAACLAB_ASSETS_DATA_DIR

##
# Configuration
##


"""Configuration of Unitree Go2 using DC motor.
"""

pcbA_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=True,
        replace_cylinders_with_capsules=False,
        asset_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/pcbA/pcb_description/urdf/pcb.urdf",
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
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=1
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Original init base pos:
        pos=(0.0, 0.0, 0.45),
        # v3.8 first-frame base pos from csv:
        # pos=(0.0063504949, 0.0, 0.8259446621),
        # Original init base quat is implicit identity (wxyz = [1, 0, 0, 0]).
        # v3.8 first-frame base quat from csv is in xyzw:
        # rot=(0.8518133759, -6.558868e-09, -0.5238453746, -1.410186e-08),
        joint_pos={
            ".*L_hip_joint": 0.0,  # 0.55246
            ".*R_hip_joint": 0.0,
            "F.*_thigh_joint": 0.81,
            "R.*_thigh_joint": -0.81,
            "F.*_calf_joint": -1.535,
            "R.*_calf_joint": 1.535,
            ".*_foot_joint": 0.0,

        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs_hip": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_joint"],
            effort_limit_sim=30,
            velocity_limit_sim=13.2,
            stiffness=45.0, #刚度系数
            damping=0.9,
            armature=0.01594,
            friction=0.0, #关节摩擦系数
        ),
         "legs_thigh": ImplicitActuatorCfg(
            joint_names_expr=[".*_thigh_joint"],
            effort_limit_sim=45,
            velocity_limit_sim=15.7,
            stiffness=45.0,
            damping=0.9,
            armature=0.01594,
            friction=0.0,
        ),
        "legs_calf": ImplicitActuatorCfg(
            joint_names_expr=[".*_calf_joint"],
            effort_limit_sim=45,
            velocity_limit_sim=15.7,
            stiffness=45.0,
            damping=0.9,
            armature=0.01594,
            friction=0.0,
        ),
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=[".*_foot_joint"],
            effort_limit_sim=25,
            velocity_limit_sim=31,
            stiffness=0,
            damping=0.7,
            armature=0.0005103,
            friction=0.0,
        ),
    },
)
"""Configuration of Unitree Go2W using DC motor.
"""
