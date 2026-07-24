# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Configuration for pcbC robots."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from robot_lab.assets import ISAACLAB_ASSETS_DATA_DIR


pcbC_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=True,
        replace_cylinders_with_capsules=False,
        asset_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/pcbC/pcb_v2_description_0.88/urdf/pcb_v88.urdf",
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
        # joint_pos={
        #     ".*L_hip_joint": 0.0,
        #     ".*R_hip_joint": 0.0,
        #     "F.*_thigh_joint": 0.81,
        #     "R.*_thigh_joint": -0.81,
        #     "F.*_calf_joint": -1.535,
        #     "R.*_calf_joint": 1.535,
        #     ".*_foot_joint": 0.0,
        # },
        joint_pos={
            "FL_hip_joint": -0.00827642,
            "FL_thigh_joint": 1.02468,
            "FL_calf_joint": -1.79535,
            "FR_hip_joint": 0.00822189,
            "FR_thigh_joint": 1.02249,
            "FR_calf_joint": -1.79259,
            "RL_hip_joint": -0.00813608,
            "RL_thigh_joint": -1.04512,
            "RL_calf_joint": 1.78846,
            "RR_hip_joint": 0.00835223,
            "RR_thigh_joint": -1.04912,
            "RR_calf_joint": 1.79229,
            "FL_foot_joint": 0.00016969,
            "FR_foot_joint": 0.0000237746,
            "RL_foot_joint": -0.0000115743,
            "RR_foot_joint": -0.000181753,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=1.0, #0.9-1.0
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
            joint_names_expr=[".*_foot_joint"],
            effort_limit_sim=30.0,
            velocity_limit_sim=31.0,
            stiffness=0.0,
            damping=0.6,
            armature=0.0005103,
            friction=0.0,
        ),
    },
)

# Backward-compatible alias in case any local code still imports the old symbol.
pcbB_CFG = pcbC_CFG
