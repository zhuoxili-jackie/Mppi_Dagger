# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import os

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import robot_lab.tasks.manager_based.beyondmimic.mdp as mdp
from robot_lab.assets.pcbB import pcbB_CFG
from robot_lab.tasks.manager_based.beyondmimic.tracking_env_cfg_go2w import BeyondMimicEnvCfg


@configclass
class PcbBBeyondMimicFlatEnvCfg(BeyondMimicEnvCfg):
    base_link_name = "Base_link"
    foot_link_name = ".*_wheel_link"

    leg_joint_names = [
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
    ]
    wheel_joint_names = [
        "FL_wheel_joint",
        "FR_wheel_joint",
        "RL_wheel_joint",
        "RR_wheel_joint",
    ]
    joint_names = leg_joint_names + wheel_joint_names

    body_names = [
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
        "FL_wheel_link",
        "FR_wheel_link",
        "RL_wheel_link",
        "RR_wheel_link",
    ]

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = pcbB_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.commands.motion.motion_file = f"{os.path.dirname(__file__)}/../go2w/motion/pcb_80cm_60hz.npz"
        self.commands.motion.anchor_body_name = self.base_link_name
        self.commands.motion.body_names = self.body_names

        self.actions.joint_pos.scale = {".*_hip_joint": 0.125, "^(?!.*_hip_joint).*": 0.25}
        self.actions.joint_vel.scale = 5.0
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_vel.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_pos.joint_names = self.leg_joint_names
        self.actions.joint_vel.joint_names = self.wheel_joint_names

        self.observations.policy.joint_pos.func = mdp.joint_pos_rel_without_wheel
        self.observations.policy.joint_pos.params["wheel_asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.wheel_joint_names
        )

        self.observations.critic.joint_pos.func = mdp.joint_pos_rel_without_wheel
        self.observations.critic.joint_pos.params["wheel_asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.wheel_joint_names
        )

        self.events.randomize_com_positions.params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names=self.base_link_name
        )

        self.rewards.motion_global_anchor_pos.weight = 1.0
        self.rewards.motion_global_anchor_ori.weight = 1.0
        self.rewards.motion_body_pos.weight = 1.0
        self.rewards.motion_body_ori.weight = 0.8
        self.rewards.motion_body_lin_vel.weight = 0.5
        self.rewards.motion_body_ang_vel.weight = 0.5
        self.rewards.action_rate_l2.func = mdp.action_rate_l2_clamped
        self.rewards.action_rate_l2.params = {"clip": 1.0, "max_value": 64.0}
        self.rewards.action_rate_l2.weight = -1.0e-2
        self.rewards.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces",
            body_names=[r"^(?!.*_wheel_link$).+"],
        )

        self.terminations.ee_body_pos.params["body_names"] = [
            "FL_wheel_link",
            "FR_wheel_link",
            "RL_wheel_link",
            "RR_wheel_link",
        ]
        self.terminations.illegal_contact = None

        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None

        self.episode_length_s = 30.0
