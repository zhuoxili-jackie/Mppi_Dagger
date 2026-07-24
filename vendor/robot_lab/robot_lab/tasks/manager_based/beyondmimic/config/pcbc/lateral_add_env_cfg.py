# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""ADD-guided lateral locomotion task for pcbC."""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

import robot_lab.tasks.manager_based.beyondmimic.mdp as mdp

from .lateral_guided_env_cfg import PCBC_LATERAL_MOTIONS, PcbCLateralGuidedEnvCfg


@configclass
class LateralAddPolicyObsCfg(ObsGroup):
    style = ObsTerm(func=mdp.add_policy_style, params={"command_name": "motion"})

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class PcbCLateralAddEnvCfg(PcbCLateralGuidedEnvCfg):
    """Continuous y-command policy trained with ADD over differential style features."""

    def __post_init__(self):
        super().__post_init__()

        motion_files = [str(path) for path, _ in PCBC_LATERAL_MOTIONS]
        lateral_velocities = [velocity for _, velocity in PCBC_LATERAL_MOTIONS]
        self.commands.motion = mdp.LateralRedMotionCommandCfg(
            asset_name="robot",
            motion_file=motion_files[0],
            motion_files=motion_files,
            lateral_velocities=lateral_velocities,
            standing_probability=0.15,
            target_velocity_range=(-0.20, 0.20),
            min_abs_target_velocity=0.03,
            reset_at_first_frame=True,
            anchor_body_name=self.base_link_name,
            body_names=self.body_names,
            style_joint_names=self.leg_joint_names,
            style_body_names=[name for name in self.body_names if name != self.base_link_name],
            red_history_length=8,
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

        self.observations.add_policy = LateralAddPolicyObsCfg()

        # ADD receives posture, heading, temporal, and y-speed objective terms
        # inside the differential vector, so the lateral task reward is not
        # duplicated here. Keep only small physical regularizers from the
        # successful guided setup.
        self.rewards.track_lateral_velocity = None
        self.rewards.motion_body_ori = None
        self.rewards.motion_joint_pos = None
        self.rewards.motion_joint_vel = None

        # The reference is an expert support set, not a global trajectory player.
        self.terminations.anchor_pos = None
        self.terminations.ee_body_pos = None
        self.episode_length_s = 12.0
