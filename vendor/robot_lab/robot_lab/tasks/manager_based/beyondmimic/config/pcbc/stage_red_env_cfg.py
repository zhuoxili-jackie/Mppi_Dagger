# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Differential RED stage-boarding imitation task for pcbC."""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

import robot_lab.tasks.manager_based.beyondmimic.mdp as mdp

from .pure_imitation import (
    PCBC_MOTION_FILE,
    PcbCBeyondMimicFlatV1StandEnvCfg,
    pcbc_default_joint_pos,
)


@configclass
class StageRedPolicyObsCfg(ObsGroup):
    style = ObsTerm(func=mdp.red_policy_style, params={"command_name": "motion"})

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class StageRedDemoObsCfg(ObsGroup):
    style = ObsTerm(func=mdp.red_demo_style, params={"command_name": "motion"})

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class PcbCStageRedEnvCfg(PcbCBeyondMimicFlatV1StandEnvCfg):
    """531 boarding imitation with RED as the core learned imitation reward."""

    def __post_init__(self):
        super().__post_init__()

        self.commands.motion = mdp.StageRedMotionCommandCfg(
            asset_name="robot",
            motion_file=PCBC_MOTION_FILE,
            anchor_body_name=self.base_link_name,
            body_names=self.body_names,
            style_joint_names=self.joint_names,
            style_body_names=[name for name in self.body_names if name != self.base_link_name],
            red_history_length=8,
            red_demo_feature_noise_std=0.01,
            red_feature_scales={
                "pose": 2.2,
                "velocity": 0.35,
                "temporal": 0.40,
                "joint_pos": 4.0,
                "anchor_pos": 4.0,
                "anchor_ori": 3.2,
                "projected_gravity": 2.4,
                "body_pos": 2.6,
                "body_ori": 2.0,
            },
            reset_at_first_frame=False,
            resampling_time_range=(1.0e9, 1.0e9),
            debug_vis=False,
            pose_range={
                "x": (-0.01, 0.01),
                "y": (-0.01, 0.01),
                "z": (-0.01, 0.01),
                "roll": (-0.03, 0.03),
                "pitch": (-0.03, 0.03),
                "yaw": (-0.05, 0.05),
            },
            velocity_range={
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (-0.03, 0.03),
                "roll": (-0.05, 0.05),
                "pitch": (-0.05, 0.05),
                "yaw": (-0.05, 0.05),
            },
            joint_position_range=(-0.03, 0.03),
        )

        self.scene.robot.init_state.joint_pos = pcbc_default_joint_pos()
        self.scene.box.init_state.pos = (0.97, 0.0, 0.4)
        self.scene.box.spawn.size = (1.05, 8.80, 0.8)

        self.observations.red_policy = StageRedPolicyObsCfg()
        self.observations.red_demo = StageRedDemoObsCfg()

        # Differential RED is the imitation scorer in this task. Keep only
        # physical regularizers here; frame-level reference tracking is provided
        # by the RED delta between policy state and current reference state.
        self.rewards.motion_global_anchor_pos = None
        self.rewards.motion_global_anchor_ori = None
        self.rewards.motion_body_pos = None
        self.rewards.motion_body_ori = None
        self.rewards.motion_body_lin_vel = None
        self.rewards.motion_body_ang_vel = None
        self.rewards.motion_joint_pos = None
        self.rewards.motion_joint_vel = None
        self.rewards.motion_wheel_joint_vel = None

        self.rewards.undesired_contacts = None
        self.rewards.action_rate_l2.weight = -1.0e-2
        self.rewards.joint_acc_l2.weight = -2.5e-7
        self.rewards.joint_torques_l2.weight = -1.0e-5

        self.terminations.anchor_pos.params["threshold"] = 0.45
        self.terminations.anchor_ori.params["threshold"] = 0.90
        self.terminations.ee_body_pos.params["threshold"] = 0.35
        self.terminations.illegal_contact = None

        self.episode_length_s = 18.0
