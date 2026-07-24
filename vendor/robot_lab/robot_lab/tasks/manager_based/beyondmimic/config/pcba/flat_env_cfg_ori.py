# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import os

from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import robot_lab.tasks.manager_based.beyondmimic.mdp as mdp
from robot_lab.assets.pcbA import pcbA_CFG
from robot_lab.tasks.manager_based.beyondmimic.tracking_env_cfg_go2w import BeyondMimicEnvCfg


@configclass
class PcbABeyondMimicFlatBaseEnvCfg(BeyondMimicEnvCfg):
    """Shared pcbA BeyondMimic flat env base config.

    Subclasses should only override profile-dependent parts:
    - motion_file
    - init pose (base/joints)
    - reward weights / randomization strength
    """

    base_link_name = "base"
    foot_link_name = ".*_foot"

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
        "FL_foot_joint",
        "FR_foot_joint",
        "RL_foot_joint",
        "RR_foot_joint",
    ]
    joint_names = leg_joint_names + wheel_joint_names

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = pcbA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.commands.motion.anchor_body_name = "base"
        self.commands.motion.body_names = [
            "base",
            "FL_hip",
            "FR_hip",
            "RL_hip",
            "RR_hip",
            "FL_thigh",
            "FR_thigh",
            "RL_thigh",
            "RR_thigh",
            "FL_calf",
            "FR_calf",
            "RL_calf",
            "RR_calf",
            "FL_foot",
            "FR_foot",
            "RL_foot",
            "RR_foot",
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

        self.events.randomize_rigid_body_material.params.update(
            {
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "static_friction_range": (0.3, 1.0),
                "dynamic_friction_range": (0.3, 1.0),
                "restitution_range": (0.0, 0.4),
                "num_buckets": 64,
            }
        )
        self.events.randomize_box_material = None
        self.events.randomize_com_positions.params.update(
            {
                "asset_cfg": SceneEntityCfg("robot", body_names=self.base_link_name),
                "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
            }
        )
        self.events.randomize_push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(4.0, 8.0),
            params={
                "velocity_range": {
                    "x": (-0.05, 0.05),
                    "y": (-0.05, 0.05),
                    "z": (-0.05, 0.05),
                    "roll": (-0.05, 0.05),
                    "pitch": (-0.05, 0.05),
                    "yaw": (-0.05, 0.05),
                }
            },
        )
        self.events.randomize_leg_actuator_gains = None
        self.events.randomize_wheel_actuator_gains = None
        self.events.randomize_actuator_gains = EventTerm(
            func=mdp.randomize_actuator_gains,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
                "stiffness_distribution_params": (0.8, 1.2),
                "damping_distribution_params": (0.8, 2.0),
                "operation": "scale",
            },
        )

        self.events.randomize_joint_friction = EventTerm(
            func=mdp.randomize_joint_parameters,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
                "friction_distribution_params": (0.0, 0.005),
                "operation": "abs",
                "distribution": "uniform",
            },
        )
        self.events.randomize_base_mass = None
        self.events.randomize_other_bodies_mass = None
        self.events.randomize_bodies_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "mass_distribution_params": (0.95, 1.05),
                "operation": "scale",
            },
        )

        self.rewards.motion_global_anchor_pos.weight = 1.3
        self.rewards.motion_global_anchor_ori.weight = 1.2
        self.rewards.motion_body_pos.weight = 0.8
        self.rewards.motion_body_ori.weight = 0.8
        self.rewards.motion_body_lin_vel.weight = 0.6
        self.rewards.motion_body_ang_vel.weight = 0.5
        # legs: pos+vel tracking
        self.rewards.motion_joint_pos.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.leg_joint_names
        )
        self.rewards.motion_joint_vel.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.leg_joint_names
        )
        self.rewards.motion_joint_pos.weight = 0.3
        self.rewards.motion_joint_vel.weight = 0.2
        # wheels: separate velocity tracking term
        self.rewards.motion_wheel_joint_vel = RewTerm(
            func=mdp.motion_joint_velocity_error_exp,
            weight=0.02,
            params={
                "command_name": "motion",
                "std": 1.0,
                "asset_cfg": SceneEntityCfg("robot", joint_names=self.wheel_joint_names),
            },
        )
        self.rewards.action_rate_l2.func = mdp.action_rate_l2_clamped
        self.rewards.action_rate_l2.params = {"clip": 1.0, "max_value": 64.0}
        self.rewards.action_rate_l2.weight = -1.0e-2

        self.terminations.illegal_contact = None
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None
        self.episode_length_s = 30.0
        # self.episode_length_s = 120.0
@configclass
class PcbABeyondMimicFlatV1StandEnvCfg(PcbABeyondMimicFlatBaseEnvCfg):
    """v0 profile: standing start motion."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.motion_file = f"{os.path.dirname(__file__)}/../go2w/motion/pcb_80cm_60hz_v1.npz"

        # Use standing init (aligned with v1 first frame trend).
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.45)
        self.scene.robot.init_state.rot = (1.0, 0.0, 0.0, 0.0)
        self.scene.robot.init_state.joint_pos = {
            ".*L_hip_joint": 0.0,
            ".*R_hip_joint": 0.0,
            "F.*_thigh_joint": 0.81,
            "R.*_thigh_joint": -0.81,
            "F.*_calf_joint": -1.535,
            "R.*_calf_joint": 1.535,
            ".*_foot_joint": 0.0,
        }

        # Keep a little wheel-vel weight for v1 profile.
        self.rewards.motion_wheel_joint_vel.weight = 0.05

@configclass
class PcbABeyondMimicFlatV1StandDelayEnvCfg(PcbABeyondMimicFlatBaseEnvCfg):
    """v1 profile: standing start motion."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.motion_file = f"{os.path.dirname(__file__)}/../go2w/motion/pcb_80cm_60hz_v1.npz"

        # Use standing init (aligned with v1 first frame trend).
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.45)
        self.scene.robot.init_state.rot = (1.0, 0.0, 0.0, 0.0)
        self.scene.robot.init_state.joint_pos = {
            ".*L_hip_joint": 0.0,
            ".*R_hip_joint": 0.0,
            "F.*_thigh_joint": 0.79,
            "R.*_thigh_joint": -0.79,
            "F.*_calf_joint": -1.48,
            "R.*_calf_joint": 1.48,
            ".*_foot_joint": 0.0,
        }

        # Task-local actuator override:
        # use DelayedPDActuator only for v1-stand task, keep other tasks unchanged.
        self.scene.robot.actuators = {
            "legs_hip": DelayedPDActuatorCfg(
                joint_names_expr=[".*_hip_joint"],
                effort_limit=60.0,
                velocity_limit=14.13,
                stiffness=35.0,
                damping=0.8,
                # armature=0.01594,
                friction=0.0,
                min_delay=0,
                max_delay=4,
            ),
            "legs_thigh": DelayedPDActuatorCfg(
                joint_names_expr=[".*_thigh_joint"],
                effort_limit=75.0,
                velocity_limit=32.46,
                stiffness=35.0,
                damping=0.8,
                # armature=0.01594,
                friction=0.0,
                min_delay=0,
                max_delay=4,
            ),
            "legs_calf": DelayedPDActuatorCfg(
                joint_names_expr=[".*_calf_joint"],
                effort_limit=75.0,
                velocity_limit=18.22,
                stiffness=35.0,
                damping=0.8,
                # armature=0.01594,
                friction=0.0,
                min_delay=0,
                max_delay=4,
            ),
            "wheels": DelayedPDActuatorCfg(
                joint_names_expr=[".*_foot_joint"],
                effort_limit=35.0,
                velocity_limit=18,
                stiffness=0.0,
                damping=0.6,
                armature=0.0005103,
                friction=0.0,
                min_delay=0,
                max_delay=4,
            ),
        }


@configclass
class PcbABeyondMimicFlatV1StandCommandEnvCfg(PcbABeyondMimicFlatV1StandEnvCfg):
    """v1 stand command task (stage C3): aggressive velocity tracking."""

    def __post_init__(self):
        super().__post_init__()
        # Use one task id with two phases:
        #   PCBA_COMMAND_PHASE=baseline  -> pure mimic foundation
        #   PCBA_COMMAND_PHASE=stage2    -> enable velocity command finetune
        command_phase = os.getenv("PCBA_COMMAND_PHASE", "stage2").strip().lower()

        # Stage2 C2-lite:
        # Smaller command range + stronger posture preservation for stable fine-tuning.
        self.commands.base_velocity = mdp.UniformVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(2.0, 4.0),
            rel_standing_envs=0.2,
            rel_heading_envs=0.0,
            heading_command=False,
            debug_vis=True,
            ranges=mdp.UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-0.08, 0.10),
                lin_vel_y=(-0.02, 0.02),
                ang_vel_z=(-0.12, 0.12),
                heading=(-3.14, 3.14),
            ),
        )
        # Stage A:
        # 0 trajectory tracking -> 1 hold final frame until stable -> 2 enable velocity command tracking
        self.commands.motion.enable_stage_command = True
        self.commands.motion.hold_anchor_rot_threshold = 0.12
        self.commands.motion.hold_body_pos_threshold = 0.06
        self.commands.motion.hold_joint_pos_threshold = 0.12
        self.commands.motion.hold_base_lin_vel_threshold = 0.12
        self.commands.motion.hold_base_ang_vel_threshold = 0.35
        self.commands.motion.hold_stable_steps = 60
        self.commands.motion.extra_hold_steps_after_stable = 80
        self.commands.motion.max_hold_steps_before_force_command = 1300
        if command_phase == "baseline":
            # Pure mimic baseline: keep v1-stand behavior intact.
            # Do not use stage command logic or stage-weighted wrappers.
            self.commands.motion.enable_stage_command = False
            self.commands.motion.max_hold_steps_before_force_command = 0
            # Add command observation term but keep value fixed at zero.
            self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
            self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
            self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
            self.observations.policy.base_velocity_command = ObsTerm(
                func=mdp.generated_commands,
                params={"command_name": "base_velocity"},
            )
            self.observations.critic.base_velocity_command = ObsTerm(
                func=mdp.generated_commands,
                params={"command_name": "base_velocity"},
            )
            # Keep stage observation dimension aligned; force constant zero in baseline.
            self.observations.policy.motion_stage = ObsTerm(
                func=mdp.constant_zero_scalar,
            )
            self.observations.critic.motion_stage = ObsTerm(
                func=mdp.constant_zero_scalar,
            )
            return

        # Stage2 finetune keeps longer episode for command tracking.
        self.episode_length_s = 40.0

        self.observations.policy.base_velocity_command = ObsTerm(
            func=mdp.stage_gated_generated_commands,
            params={"command_name": "base_velocity", "stage_command_name": "motion", "enabled_stage": 2},
        )
        self.observations.critic.base_velocity_command = ObsTerm(
            func=mdp.stage_gated_generated_commands,
            params={"command_name": "base_velocity", "stage_command_name": "motion", "enabled_stage": 2},
        )
        self.observations.policy.motion_stage = ObsTerm(
            func=mdp.motion_stage,
            params={"command_name": "motion", "normalize": True},
        )
        self.observations.critic.motion_stage = ObsTerm(
            func=mdp.motion_stage,
            params={"command_name": "motion", "normalize": True},
        )

        # Stage-weighted mimic shaping:
        # stage0 strong imitation, stage1 hold-pose imitation, stage2 reduce global constraints.
        self.rewards.motion_global_anchor_pos = RewTerm(
            func=mdp.stage_weighted_motion_global_anchor_position_error_exp,
            weight=1.30,
            params={
                "command_name": "motion",
                "std": 0.3,
                "stage_command_name": "motion",
                "stage0_weight": 1.0,
                "stage1_weight": 0.8,
                "stage2_weight": 0.0,
            },
        )
        self.rewards.motion_global_anchor_ori = RewTerm(
            func=mdp.stage_weighted_motion_global_anchor_orientation_error_exp,
            weight=1.20,
            params={
                "command_name": "motion",
                "std": 0.4,
                "stage_command_name": "motion",
                "stage0_weight": 1.0,
                "stage1_weight": 0.8,
                "stage2_weight": 0.2,
            },
        )
        self.rewards.motion_body_pos = RewTerm(
            func=mdp.stage_weighted_motion_relative_body_position_error_exp,
            weight=0.90,
            params={
                "command_name": "motion",
                "std": 0.3,
                "stage_command_name": "motion",
                "stage0_weight": 1.0,
                "stage1_weight": 1.15,
                "stage2_weight": 0.8,
            },
        )
        self.rewards.motion_body_ori = RewTerm(
            func=mdp.stage_weighted_motion_relative_body_orientation_error_exp,
            weight=0.95,
            params={
                "command_name": "motion",
                "std": 0.4,
                "stage_command_name": "motion",
                "stage0_weight": 1.0,
                "stage1_weight": 1.15,
                "stage2_weight": 0.8,
            },
        )
        self.rewards.motion_body_lin_vel = RewTerm(
            func=mdp.stage_weighted_motion_global_body_linear_velocity_error_exp,
            weight=0.60,
            params={
                "command_name": "motion",
                "std": 1.0,
                "stage_command_name": "motion",
                "stage0_weight": 1.0,
                "stage1_weight": 0.3,
                "stage2_weight": 0.0,
            },
        )
        self.rewards.motion_body_ang_vel = RewTerm(
            func=mdp.stage_weighted_motion_global_body_angular_velocity_error_exp,
            weight=0.50,
            params={
                "command_name": "motion",
                "std": 3.14,
                "stage_command_name": "motion",
                "stage0_weight": 1.0,
                "stage1_weight": 0.3,
                "stage2_weight": 0.0,
            },
        )
        self.rewards.motion_joint_pos = RewTerm(
            func=mdp.stage_weighted_motion_joint_position_error_exp,
            weight=0.40,
            params={
                "command_name": "motion",
                "std": 0.5,
                "asset_cfg": SceneEntityCfg("robot", joint_names=self.leg_joint_names),
                "stage_command_name": "motion",
                "stage0_weight": 1.0,
                "stage1_weight": 1.2,
                "stage2_weight": 0.7,
            },
        )
        self.rewards.motion_joint_vel = RewTerm(
            func=mdp.stage_weighted_motion_joint_velocity_error_exp,
            weight=0.18,
            params={
                "command_name": "motion",
                "std": 1.0,
                "asset_cfg": SceneEntityCfg("robot", joint_names=self.leg_joint_names),
                "stage_command_name": "motion",
                "stage0_weight": 1.0,
                "stage1_weight": 0.5,
                "stage2_weight": 0.35,
            },
        )
        self.rewards.motion_wheel_joint_vel = RewTerm(
            func=mdp.stage_weighted_motion_joint_velocity_error_exp,
            weight=0.05,
            params={
                "command_name": "motion",
                "std": 1.0,
                "asset_cfg": SceneEntityCfg("robot", joint_names=self.wheel_joint_names),
                "stage_command_name": "motion",
                "stage0_weight": 1.0,
                "stage1_weight": 0.4,
                "stage2_weight": 0.0,
            },
        )

        # Allow more action change for velocity response.
        self.rewards.action_rate_l2.weight = -1.0e-2

        # Strong velocity tracking reward.
        self.rewards.track_lin_vel_xy = RewTerm(
            func=mdp.stage_gated_track_lin_vel_xy_exp,
            weight=0.25,
            params={
                "command_name": "base_velocity",
                "std": 0.12,
                "stage_command_name": "motion",
                "enabled_stage": 2,
            },
        )
        #惩罚动作变化太大
        self.rewards.track_yaw_rate = RewTerm(
            func=mdp.stage_gated_track_ang_vel_z_exp,
            weight=0.15,
            params={
                "command_name": "base_velocity",
                "std": 0.1,
                "stage_command_name": "motion",
                "enabled_stage": 2,
            },
        )
        # No curriculum. Directly expose full C3 command range.
        self.curriculum.base_velocity_lin = None
        self.curriculum.base_velocity_yaw = None

        # Keep relaxed termination.
        self.terminations.anchor_pos = None
        self.terminations.anchor_ori = None
        self.terminations.ee_body_pos = None
        self.terminations.illegal_contact = None

        # episode_length_s is phase-dependent and set above.

@configclass
class PcbABeyondMimicFlatV38RootEnvCfg(PcbABeyondMimicFlatBaseEnvCfg):
    """v3.8 profile: root-tilted start motion."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.motion_file = (
            f"{os.path.dirname(__file__)}/../go2w/motion/pcbA_80cm_60hz_lateral_move_v3.8.npz"
        )

        # v3.8 first-frame base pose:
        # base pos xyz = [0.0063504949, 0.0, 0.8259446621]
        # base quat xyzw = [-6.558868e-09, -0.52384537, -1.410186e-08, 0.85181338]
        # converted for InitialStateCfg rot(wxyz):
        # [0.85181338, -6.558868e-09, -0.52384537, -1.410186e-08]
        self.scene.robot.init_state.pos = (0.0063504949, 0.0, 0.8259446621)
        self.scene.robot.init_state.rot = (0.8518133759, -6.558868e-09, -0.5238453746, -1.410186e-08)
        self.scene.robot.init_state.joint_pos = {
            ".*L_hip_joint": 0.0,
            ".*R_hip_joint": 0.0,
            "F.*_thigh_joint": 0.09836920,
            "R.*_thigh_joint": 0.65720934,
            "F.*_calf_joint": -0.29722634,
            "R.*_calf_joint": 0.81193393,
            ".*_foot_joint": 0.0,
        }

        # v3.8 wheel velocity target is near zero, keep small wheel-vel reward.
        self.rewards.motion_wheel_joint_vel.weight = 0.02


@configclass
class PcbABeyondMimicFlatEnvCfg(PcbABeyondMimicFlatV38RootEnvCfg):
    """Backward-compatible alias for legacy references."""
