# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import os

from isaaclab.utils import configclass


from robot_lab.tasks.manager_based.beyondmimic.tracking_env_cfg_go2w import BeyondMimicEnvCfg
from robot_lab.assets.pcbA  import pcbA_CFG
from robot_lab.assets.unitree import UNITREE_GO2W_CFG
import robot_lab.tasks.manager_based.beyondmimic.mdp as mdp
from isaaclab.managers import SceneEntityCfg
@configclass


class go2wBeyondMimicFlatEnvCfg(BeyondMimicEnvCfg):
    base_link_name = "base"
    foot_link_name = ".*_foot"

    # fmt: off
    leg_joint_names = [
        "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
        "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
        "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
     
    ]
    wheel_joint_names = [
        "FL_foot_joint", "FR_foot_joint", "RL_foot_joint", "RR_foot_joint",
    ]
    joint_names = leg_joint_names + wheel_joint_names
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = UNITREE_GO2W_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.commands.motion.motion_file = f"{os.path.dirname(__file__)}/motion/go2w_80cm_60hz.npz"

        self.commands.motion.anchor_body_name = "base"
        self.commands.motion.body_names = [
            "base",
                "FL_hip", "FR_hip",           # 髋关节
                "Head_upper", "Head_lower",   # 头部（添加这两个）
                "RL_hip", "RR_hip",
                "FL_thigh", "FR_thigh",
                "RL_thigh", "RR_thigh",
                "FL_calf", "FR_calf",
                "RL_calf", "RR_calf",
                "FL_foot", "FR_foot", 
                "RL_foot", "RR_foot"
        ]
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

        # self.terminations.illegal_contact.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        self.terminations.illegal_contact = None

        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None

        self.episode_length_s = 30.0
