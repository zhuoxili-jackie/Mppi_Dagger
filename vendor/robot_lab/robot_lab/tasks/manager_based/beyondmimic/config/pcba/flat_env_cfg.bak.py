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

# 2026-05-20_14-52-29基线版本训练baseline,然后stage2,stage2速度响应不是很好
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
            resampling_time_range=(2.0, 4.0),  #速度指令采样周期是 2.0 ~ 4.0 秒随机重采样一次
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
        # Extra-long platform along y direction for stage2 command adaptation/play.
        # size = (x_length, y_width, z_height)
        self.scene.box.spawn.size = (1.05, 8.80, 0.8)

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

        # Decoupled x/y velocity tracking in stage2:
        # keep x moderate, boost y to encourage lateral response.
        self.rewards.track_lin_vel_xy = RewTerm(
            func=mdp.stage_gated_track_lin_vel_axis_exp,
            weight=0.16,
            params={
                "command_name": "base_velocity",
                "axis": 0,
                "std": 0.10,
                "stage_command_name": "motion",
                "enabled_stage": 2,
            },
        )
        self.rewards.track_lin_vel_y = RewTerm(
            func=mdp.stage_gated_track_lin_vel_axis_exp,
            weight=0.24,
            params={
                "command_name": "base_velocity",
                "axis": 1,
                "std": 0.08,
                "stage_command_name": "motion",
                "enabled_stage": 2,
            },
        )
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

        # Weak y-command gated leg-lift shaping.
        self.rewards.y_cmd_leg_lift = RewTerm(
            func=mdp.stage_gated_y_command_leg_lift_posture_reward,
            weight=0.04,
            params={
                "command_name": "base_velocity",
                "joint_names": ["FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint"],
                "y_cmd_threshold": 0.010,
                "target_delta": 0.10,
                "std": 0.18,
                "stage_command_name": "motion",
                "enabled_stage": 2,
            },
        )

        # If lateral response is still weak, this very small term nudges rear feet airtime.
        self.rewards.rear_foot_air_time = RewTerm(
            func=mdp.stage_gated_rear_feet_air_time_reward,
            weight=0.01,
            params={
                "command_name": "base_velocity",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["RL_foot", "RR_foot"]),
                "y_cmd_threshold": 0.012,
                "threshold": 0.22,
                "stage_command_name": "motion",
                "enabled_stage": 2,
            },
        )

        # Safety/style constraints in stage2.
        self.rewards.hip_abduction_penalty = RewTerm(
            func=mdp.stage_gated_joint_deviation_l1_penalty,
            weight=-0.04,
            params={
                "joint_names": ["FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint"],
                "command_name": "motion",
                "enabled_stage": 2,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.base_tilt_penalty = RewTerm(
            func=mdp.stage_gated_base_tilt_l2_penalty,
            weight=-0.06,
            params={
                "command_name": "motion",
                "enabled_stage": 2,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.multi_foot_air_penalty = RewTerm(
            func=mdp.stage_gated_multi_foot_air_penalty,
            weight=-0.05,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
                "max_air_feet": 2,
                "command_name": "motion",
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
# 2026-05-21_18-07-18，29999模型play效果，这个版本y有响应，但是有速度之后开始横移，悬空后翻
@configclass
class Stage2CommandEnvCfg(PcbABeyondMimicFlatV1StandCommandEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        command_phase = os.getenv("PCBA_COMMAND_PHASE", "stage2").strip().lower()
        if command_phase == "baseline":
            return

        self.commands.base_velocity.ranges = mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.04, 0.04),
            lin_vel_y=(-0.2, 0.2),
            ang_vel_z=(-0.08, 0.08),
            heading=(-3.14, 3.14),
        )

        self.rewards.track_lin_vel_xy = RewTerm(
            func=mdp.stage_gated_track_lin_vel_axis_projected_body_exp,
            weight=0.10,
            params={
                "command_name": "base_velocity",
                "axis": 0,
                "std": 0.12,
                "stage_command_name": "motion",
                "enabled_stage": 2,
            },
        )

        self.rewards.track_lin_vel_y = RewTerm(
            func=mdp.stage_gated_track_lin_vel_axis_projected_body_exp,
            weight=0.55,
            params={
                "command_name": "base_velocity",
                "axis": 1,
                "std": 0.10,
                "stage_command_name": "motion",
                "enabled_stage": 2,
            },
        )

        self.rewards.track_yaw_rate = RewTerm(
            func=mdp.stage_gated_track_ang_vel_z_projected_exp,
            weight=0.05,
            params={
                "command_name": "base_velocity",
                "std": 0.12,
                "stage_command_name": "motion",
                "enabled_stage": 2,
            },
        )

        self.rewards.y_cmd_leg_lift = RewTerm(
            func=mdp.stage_gated_y_command_leg_lift_posture_reward,
            weight=0.06,
            params={
                "command_name": "base_velocity",
                "joint_names": ["FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint"],
                "y_cmd_threshold": 0.004,
                "target_delta": 0.10,
                "std": 0.20,
                "stage_command_name": "motion",
                "enabled_stage": 2,
            },
        )

        self.rewards.rear_foot_air_time = RewTerm(
            func=mdp.stage_gated_rear_feet_air_time_reward,
            weight=0.005,
            params={
                "command_name": "base_velocity",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["RL_foot", "RR_foot"]),
                "y_cmd_threshold": 0.004,
                "threshold": 0.05,
                "stage_command_name": "motion",
                "enabled_stage": 2,
            },
        )
# 2026-05-22_10-13-57，7500的play效果，不翻到，横移不明显，呈外八，保留 y 速度主目标， roll/pitch 不要太大；hip 外摆不要太大；不要多脚同时离地。
@configclass
class Stage2CommandEnvCfgV1(PcbABeyondMimicFlatV1StandCommandEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        command_phase = os.getenv("PCBA_COMMAND_PHASE", "stage2").strip().lower()
        if command_phase == "baseline":
            return

        # Focus on lateral y motion, keep x/yaw weak.
        self.commands.base_velocity.ranges = mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.02, 0.03),
            lin_vel_y=(-0.20, 0.20),
            ang_vel_z=(-0.08, 0.08),
            heading=(-3.14, 3.14),
        )

        self.rewards.track_lin_vel_xy = RewTerm(
            func=mdp.stage_gated_track_lin_vel_axis_projected_body_exp,
            weight=0.06,
            params={
                "command_name": "base_velocity",
                "axis": 0,
                "std": 0.14,
                "stage_command_name": "motion",
                "enabled_stage": 2,
            },
        )

        self.rewards.track_lin_vel_y = RewTerm(
            func=mdp.stage_gated_track_lin_vel_axis_projected_body_exp,
            weight=0.60,
            params={
                "command_name": "base_velocity",
                "axis": 1,
                "std": 0.12,
                "stage_command_name": "motion",
                "enabled_stage": 2,
            },
        )

        self.rewards.track_yaw_rate = RewTerm(
            func=mdp.stage_gated_track_ang_vel_z_projected_exp,
            weight=0.04,
            params={
                "command_name": "base_velocity",
                "std": 0.14,
                "stage_command_name": "motion",
                "enabled_stage": 2,
            },
        )

        # 然后鼓励指定 thigh joints 产生一个轻微抬腿姿态
        self.rewards.y_cmd_leg_lift = RewTerm(
            func=mdp.stage_gated_y_command_leg_lift_posture_reward,
            weight=0.06,
            params={
                "command_name": "base_velocity",
                "joint_names": ["FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint"],
                "y_cmd_threshold": 0.004,
                "target_delta": 0.10,
                "std": 0.22,
                "stage_command_name": "motion",
                "enabled_stage": 2,
            },
        )

        # 如果后脚短暂离地达到一定条件，就给一点奖励
        self.rewards.rear_foot_air_time = RewTerm(
            func=mdp.stage_gated_rear_feet_air_time_reward,
            weight=0.004,
            params={
                "command_name": "base_velocity",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["RL_foot", "RR_foot"]),
                "y_cmd_threshold": 0.004,
                "threshold": 0.04,
                "stage_command_name": "motion",
                "enabled_stage": 2,
            },
        )

        # stage2 时鼓励前脚/前轮保持接触。
        self.rewards.front_feet_contact = RewTerm(
            func=mdp.stage_gated_feet_contact_reward,
            weight=0.08,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["FL_foot", "FR_foot"]),
                "force_threshold": 1.0,
                "stage_command_name": "motion",
                "enabled_stage": 2,
            },
        )
        # stage2 时，如果 base pitch 超过 0.35 rad，就惩罚。
        self.rewards.base_pitch_excess = RewTerm(
            func=mdp.stage_gated_base_pitch_excess_l2_penalty,
            weight=-0.12,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "pitch_limit": 0.35,
                "stage_command_name": "motion",
                "enabled_stage": 2,
            },
        )

        # Strengthen existing safety penalties.
        self.rewards.base_tilt_penalty.weight = -0.10
        self.rewards.hip_abduction_penalty.weight = -0.06
        self.rewards.multi_foot_air_penalty.weight = -0.08

# 继承 V1 的稳定性；关闭一直为 0 的 leg_lift / airtime；新增 y_cmd_leg_motion 连续腿部参与奖励；稍微提高 y tracking；稍微压制 hip 外摆；保持 front contact 和 pitch 稳定约束。
# 2026-05-22_13-58-45，8000的play效果：
@configclass
class Stage2CommandEnvCfgV2(Stage2CommandEnvCfgV1):
    """Stage2 y-lateral stable v2.

    Goal:
    - keep V1 stability: no fall, no backflip, front feet contact
    - recover stronger y response
    - replace sparse leg-lift / airtime rewards with continuous leg-motion reward
    - reduce excessive hip abduction / out-toeing
    """

    def __post_init__(self):
        super().__post_init__()

        command_phase = os.getenv("PCBA_COMMAND_PHASE", "stage2").strip().lower()
        if command_phase == "baseline":
            return

        # Keep y as the main command, but do not jump to +-0.5 yet.
        self.commands.base_velocity.ranges = mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.02, 0.03),
            lin_vel_y=(-0.20, 0.20),
            ang_vel_z=(-0.06, 0.06),
            heading=(-3.14, 3.14),
        )

        # Keep x weak.
        self.rewards.track_lin_vel_xy = RewTerm(
            func=mdp.stage_gated_track_lin_vel_axis_projected_body_exp,
            weight=0.05,
            params={
                "command_name": "base_velocity",
                "axis": 0,
                "std": 0.14,
                "stage_command_name": "motion",
                "enabled_stage": 2,
            },
        )

        # Recover stronger y tracking.
        self.rewards.track_lin_vel_y = RewTerm(
            func=mdp.stage_gated_track_lin_vel_axis_projected_body_exp,
            weight=0.70,
            params={
                "command_name": "base_velocity",
                "axis": 1,
                "std": 0.12,
                "stage_command_name": "motion",
                "enabled_stage": 2,
            },
        )

        # Keep yaw weak, avoid using yaw to fake lateral motion.
        self.rewards.track_yaw_rate = RewTerm(
            func=mdp.stage_gated_track_ang_vel_z_projected_exp,
            weight=0.04,
            params={
                "command_name": "base_velocity",
                "std": 0.14,
                "stage_command_name": "motion",
                "enabled_stage": 2,
            },
        )

        # These two were always zero in previous runs, so disable them.
        self.rewards.y_cmd_leg_lift = None
        self.rewards.rear_foot_air_time = None

        # 只要 y 命令来了，腿部稍微动就有连续信号
        self.rewards.y_cmd_leg_motion = RewTerm(
            func=mdp.stage_gated_y_command_joint_motion_reward,
            weight=0.025,
            params={
                "command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "FL_hip_joint", "FR_hip_joint",
                        "RL_hip_joint", "RR_hip_joint",
                        "FL_thigh_joint", "FR_thigh_joint",
                        "RL_thigh_joint", "RR_thigh_joint",
                        "FL_calf_joint", "FR_calf_joint",
                        "RL_calf_joint", "RR_calf_joint",
                    ],
                ),
                "y_cmd_threshold": 0.004,
                "max_value": 2.0,
                "stage_command_name": "motion",
                "enabled_stage": 2,
            },
        )

        # Keep V1 stability, but slightly relax if y is too weak.
        self.rewards.front_feet_contact.weight = 0.07
        self.rewards.base_pitch_excess.weight = -0.10
        self.rewards.base_tilt_penalty.weight = -0.09

        # Stronger hip abduction penalty to reduce out-toeing.
        self.rewards.hip_abduction_penalty.weight = -0.08

        # Keep multi-foot airborne penalty.
        self.rewards.multi_foot_air_penalty.weight = -0.08

# V3：对角协同步态引导，对角共同运动奖励，和对角不平衡惩罚
# 保留 V2 稳定性；
# y tracking 小幅增强；
# y_cmd_leg_motion 降为辅助；
# 新增对角腿共同运动 reward；
# 新增对角运动不平衡 penalty；
# hip 外摆小幅加强；
# 暂时不加 airtime。2026-05-22_17-40-40，29999 横向移动小步子响应，旋转和前进无响应，reset有轨迹模仿摔倒的风险
@configclass
class Stage2CommandEnvCfgV3(Stage2CommandEnvCfgV2):
    """Stage2 y-lateral v3: diagonal coordination for lateral stepping."""

    def __post_init__(self):
        super().__post_init__()

        command_phase = os.getenv("PCBA_COMMAND_PHASE", "stage2").strip().lower()
        if command_phase == "baseline":
            return

        # Keep current safe command range.
        self.commands.base_velocity.ranges = mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.02, 0.03),
            lin_vel_y=(-0.20, 0.20),
            ang_vel_z=(-0.06, 0.06),
            heading=(-3.14, 3.14),
        )

        # Slightly stronger y, but not aggressive.
        self.rewards.track_lin_vel_y.weight = 0.80
        self.rewards.track_lin_vel_xy.weight = 0.05
        self.rewards.track_yaw_rate.weight = 0.04

        # The old all-leg motion reward can stay but weaker.
        # It provides dense signal, but should not dominate diagonal coordination.
        self.rewards.y_cmd_leg_motion.weight = 0.015

        # New: require both diagonal pairs to participate.
        self.rewards.y_cmd_diagonal_leg_motion = RewTerm(
            func=mdp.stage_gated_y_command_diagonal_leg_motion_reward,
            weight=0.035,
            params={
                "command_name": "base_velocity",
                "diag_a_joint_names": [
                    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
                    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
                ],
                "diag_b_joint_names": [
                    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
                    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
                ],
                "y_cmd_threshold": 0.004,
                "max_value": 2.0,
                "stage_command_name": "motion",
                "enabled_stage": 2,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        # New: penalize one diagonal pair moving while the other does not.
        self.rewards.y_cmd_diagonal_motion_balance = RewTerm(
            func=mdp.stage_gated_y_command_diagonal_motion_balance_penalty,
            weight=-0.02,
            params={
                "command_name": "base_velocity",
                "diag_a_joint_names": [
                    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
                    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
                ],
                "diag_b_joint_names": [
                    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
                    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
                ],
                "y_cmd_threshold": 0.004,
                "max_value": 2.0,
                "stage_command_name": "motion",
                "enabled_stage": 2,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        # Slightly reduce out-toeing.
        self.rewards.hip_abduction_penalty.weight = -0.09

        # Keep safety constraints.
        self.rewards.front_feet_contact.weight = 0.07
        self.rewards.base_pitch_excess.weight = -0.10
        self.rewards.base_tilt_penalty.weight = -0.09
        self.rewards.multi_foot_air_penalty.weight = -0.08

# V3.1: 推迟进入 stage2，不继续增强 y 速度，保持0.8，3. 保留对角步态奖励，4. 小幅加强 hip 外摆惩罚，5. 强化 stage0/1 imitation
# 2026-05-23_11-08-37,16500播放效果，左移可以，但是右移不响应，reset不后翻
@configclass
class Stage2CommandEnvCfgV31(Stage2CommandEnvCfgV3):
    """Stage2 y-lateral v3.1.

    Goal:
    - keep V3 lateral small-step behavior
    - reduce reset / stage0-1 backflip risk
    - reduce front-leg out-toeing during y direction switching
    - do NOT further increase y aggression
    - do NOT restore x/yaw yet
    """

    def __post_init__(self):
        super().__post_init__()

        command_phase = os.getenv("PCBA_COMMAND_PHASE", "stage2").strip().lower()
        if command_phase == "baseline":
            return

        # ------------------------------------------------------------
        # 1. Protect stage0/stage1: delay stage2 entry.
        # ------------------------------------------------------------
        # V3 已经能横向小步，V3.1 不继续加速，而是让 policy 多复习
        # 上台轨迹和末帧 hold，减少 stage2 长训导致的 stage0 遗忘。
        self.commands.motion.hold_stable_steps = 80
        self.commands.motion.extra_hold_steps_after_stable = 120
        self.commands.motion.max_hold_steps_before_force_command = 1800

        # ------------------------------------------------------------
        # 2. Keep current command range.
        # ------------------------------------------------------------
        # 先不要恢复 x/yaw。当前目标是修 reset 后翻和外八，
        # 不是训练完整 x/y/yaw command。
        self.commands.base_velocity.ranges = mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.02, 0.03),
            lin_vel_y=(-0.20, 0.20),
            ang_vel_z=(-0.06, 0.06),
            heading=(-3.14, 3.14),
        )

        # ------------------------------------------------------------
        # 3. Keep V3 velocity objective.
        # ------------------------------------------------------------
        # 不继续提高 y，避免进一步破坏 stage0/1。
        self.rewards.track_lin_vel_y.weight = 0.80
        self.rewards.track_lin_vel_xy.weight = 0.05
        self.rewards.track_yaw_rate.weight = 0.04

        # ------------------------------------------------------------
        # 4. Keep diagonal stepping behavior.
        # ------------------------------------------------------------
        # 普通 leg motion 只当 dense signal，不主导。
        self.rewards.y_cmd_leg_motion.weight = 0.015

        # 对角腿协同保持 V3，不继续加。
        self.rewards.y_cmd_diagonal_leg_motion.weight = 0.035
        self.rewards.y_cmd_diagonal_motion_balance.weight = -0.02

        # ------------------------------------------------------------
        # 5. Slightly reduce out-toeing.
        # ------------------------------------------------------------
        # V3 play 中右/左切换时前腿会先外八，
        # 这里小幅加强 hip 外摆惩罚。
        self.rewards.hip_abduction_penalty.weight = -0.10

        # ------------------------------------------------------------
        # 6. Keep stability constraints.
        # ------------------------------------------------------------
        # 不要放松这些项。你之前已经验证：
        # front_feet_contact / base_pitch_excess / base_tilt 是防后翻和前轮悬空的关键。
        self.rewards.front_feet_contact.weight = 0.07
        self.rewards.base_pitch_excess.weight = -0.10
        self.rewards.base_tilt_penalty.weight = -0.09
        self.rewards.multi_foot_air_penalty.weight = -0.08

        # ------------------------------------------------------------
        # 7. Re-balance imitation weights to protect stage0/stage1.
        # ------------------------------------------------------------
        # 这些 reward 在父类里已经是 stage-weighted。
        # 这里不改结构，只略微强化 stage0/stage1，
        # 同时 stage2 不再进一步压得太死。
        self.rewards.motion_global_anchor_pos.params["stage0_weight"] = 1.10
        self.rewards.motion_global_anchor_pos.params["stage1_weight"] = 0.90
        self.rewards.motion_global_anchor_pos.params["stage2_weight"] = 0.00

        self.rewards.motion_global_anchor_ori.params["stage0_weight"] = 1.10
        self.rewards.motion_global_anchor_ori.params["stage1_weight"] = 0.90
        self.rewards.motion_global_anchor_ori.params["stage2_weight"] = 0.15

        self.rewards.motion_body_pos.params["stage0_weight"] = 1.10
        self.rewards.motion_body_pos.params["stage1_weight"] = 1.25
        self.rewards.motion_body_pos.params["stage2_weight"] = 0.80

        self.rewards.motion_body_ori.params["stage0_weight"] = 1.10
        self.rewards.motion_body_ori.params["stage1_weight"] = 1.25
        self.rewards.motion_body_ori.params["stage2_weight"] = 0.80

        self.rewards.motion_body_lin_vel.params["stage0_weight"] = 1.10
        self.rewards.motion_body_lin_vel.params["stage1_weight"] = 0.35
        self.rewards.motion_body_lin_vel.params["stage2_weight"] = 0.00

        self.rewards.motion_body_ang_vel.params["stage0_weight"] = 1.10
        self.rewards.motion_body_ang_vel.params["stage1_weight"] = 0.35
        self.rewards.motion_body_ang_vel.params["stage2_weight"] = 0.00

        self.rewards.motion_joint_pos.params["stage0_weight"] = 1.10
        self.rewards.motion_joint_pos.params["stage1_weight"] = 1.30
        self.rewards.motion_joint_pos.params["stage2_weight"] = 0.65

        self.rewards.motion_joint_vel.params["stage0_weight"] = 1.00
        self.rewards.motion_joint_vel.params["stage1_weight"] = 0.55
        self.rewards.motion_joint_vel.params["stage2_weight"] = 0.30

        self.rewards.motion_wheel_joint_vel.params["stage0_weight"] = 1.00
        self.rewards.motion_wheel_joint_vel.params["stage1_weight"] = 0.40
        self.rewards.motion_wheel_joint_vel.params["stage2_weight"] = 0.00

#V31 stage2 占比太低；横向速度范围和 reward 压力不足；正负 y 没有被重新巩固。
#play了2026-05-23_17-55-19 29999模型，效果是左右两边都能响应，但是身体会贴近高台，reset也有翻倒的风险
@configclass
class Stage2CommandEnvCfgV32(Stage2CommandEnvCfgV31):
    """Stage2 y-lateral v3.2.

    Goal:
    - keep V3.1 reset/stage0 stability
    - recover more stage2 exposure
    - improve negative-y and small-y response
    - gently expand lateral command range
    - keep x/yaw weak for now
    """

    def __post_init__(self):
        super().__post_init__()

        command_phase = os.getenv("PCBA_COMMAND_PHASE", "stage2").strip().lower()
        if command_phase == "baseline":
            return

        # ------------------------------------------------------------
        # 1. Recover stage2 exposure.
        # ------------------------------------------------------------
        # V31 的 1800 太保守，stage2 占比过低。
        # 这里收回到 1500，让 stage2 重新有训练量。
        self.commands.motion.hold_stable_steps = 70
        self.commands.motion.extra_hold_steps_after_stable = 100
        self.commands.motion.max_hold_steps_before_force_command = 1500

        # ------------------------------------------------------------
        # 2. Expand y command mildly, not directly to +-0.5.
        # ------------------------------------------------------------
        # 下一阶段先到 +-0.30。等 +-0.30 两边稳定后再上 +-0.40 / +-0.50。
        self.commands.base_velocity.ranges = mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.02, 0.03),
            lin_vel_y=(-0.30, 0.30),
            ang_vel_z=(-0.06, 0.06),
            heading=(-3.14, 3.14),
        )

        # ------------------------------------------------------------
        # 3. Restore y tracking pressure.
        # ------------------------------------------------------------
        # y 是主任务。std 稍微放宽，避免为了追大速度导致动作太激进。
        self.rewards.track_lin_vel_y.weight = 0.95
        self.rewards.track_lin_vel_y.params["std"] = 0.16

        # x/yaw 仍保持弱，不在这一版恢复前进/旋转。
        self.rewards.track_lin_vel_xy.weight = 0.05
        self.rewards.track_yaw_rate.weight = 0.04

        # ------------------------------------------------------------
        # 4. Strengthen small-step leg participation.
        # ------------------------------------------------------------
        # V31 里这些几乎为 0，说明 stage2 练得太少。
        # V32 小幅增强，让 vy=0.1 也更容易触发腿部响应。
        self.rewards.y_cmd_leg_motion.weight = 0.025
        self.rewards.y_cmd_diagonal_leg_motion.weight = 0.050
        self.rewards.y_cmd_diagonal_motion_balance.weight = -0.025

        # ------------------------------------------------------------
        # 5. Keep anti-out-toe and stability.
        # ------------------------------------------------------------
        self.rewards.hip_abduction_penalty.weight = -0.10

        self.rewards.front_feet_contact.weight = 0.07
        self.rewards.base_pitch_excess.weight = -0.10
        self.rewards.base_tilt_penalty.weight = -0.09
        self.rewards.multi_foot_air_penalty.weight = -0.08

        # ------------------------------------------------------------
        # 6. Keep V31's stage0/1 protection, but slightly less conservative.
        # ------------------------------------------------------------
        # V31 已经证明能修 reset 后 stage0/1 翻倒。
        # V32 稍微放回一点 stage2 空间。
        self.rewards.motion_global_anchor_pos.params["stage0_weight"] = 1.05
        self.rewards.motion_global_anchor_pos.params["stage1_weight"] = 0.85
        self.rewards.motion_global_anchor_pos.params["stage2_weight"] = 0.00

        self.rewards.motion_global_anchor_ori.params["stage0_weight"] = 1.05
        self.rewards.motion_global_anchor_ori.params["stage1_weight"] = 0.85
        self.rewards.motion_global_anchor_ori.params["stage2_weight"] = 0.15

        self.rewards.motion_body_pos.params["stage0_weight"] = 1.05
        self.rewards.motion_body_pos.params["stage1_weight"] = 1.20
        self.rewards.motion_body_pos.params["stage2_weight"] = 0.80

        self.rewards.motion_body_ori.params["stage0_weight"] = 1.05
        self.rewards.motion_body_ori.params["stage1_weight"] = 1.20
        self.rewards.motion_body_ori.params["stage2_weight"] = 0.80

        self.rewards.motion_body_lin_vel.params["stage0_weight"] = 1.05
        self.rewards.motion_body_lin_vel.params["stage1_weight"] = 0.35
        self.rewards.motion_body_lin_vel.params["stage2_weight"] = 0.00

        self.rewards.motion_body_ang_vel.params["stage0_weight"] = 1.05
        self.rewards.motion_body_ang_vel.params["stage1_weight"] = 0.35
        self.rewards.motion_body_ang_vel.params["stage2_weight"] = 0.00

        self.rewards.motion_joint_pos.params["stage0_weight"] = 1.05
        self.rewards.motion_joint_pos.params["stage1_weight"] = 1.25
        self.rewards.motion_joint_pos.params["stage2_weight"] = 0.65

        self.rewards.motion_joint_vel.params["stage0_weight"] = 1.00
        self.rewards.motion_joint_vel.params["stage1_weight"] = 0.55
        self.rewards.motion_joint_vel.params["stage2_weight"] = 0.30

        self.rewards.motion_wheel_joint_vel.params["stage0_weight"] = 1.00
        self.rewards.motion_wheel_joint_vel.params["stage1_weight"] = 0.40
        self.rewards.motion_wheel_joint_vel.params["stage2_weight"] = 0.00

# 主任务仍然是 y 横向；允许小 x 速度修正离高台距离；允许小 yaw 修正身体朝向；不要再加 body clearance penalty；保留 front contact / pitch / tilt 稳定约束；
# 稍微加强 stage0/1 防 reset 后翻。
@configclass
class Stage2CommandEnvCfgV32Safe(Stage2CommandEnvCfgV32):
    """Stage2 y-lateral v32-safe.

    Goal:
    - keep V32 bidirectional y response
    - improve reset/stage0 robustness
    - avoid failed V33/V33Fix clearance penalties
    - add small x/yaw correction ability
    - do not increase y range yet
    """

    def __post_init__(self):
        super().__post_init__()

        command_phase = os.getenv("PCBA_COMMAND_PHASE", "stage2").strip().lower()
        if command_phase == "baseline":
            return

        # ------------------------------------------------------------
        # 1. Stage timing: between V31 and V32.
        # ------------------------------------------------------------
        # V31: 80 / 120 / 1800 -> stable but stage2 too weak
        # V32: 70 / 100 / 1500 -> y response good but reset can flip
        # V32Safe: slightly safer than V32, not as conservative as V31
        self.commands.motion.hold_stable_steps = 75
        self.commands.motion.extra_hold_steps_after_stable = 110
        self.commands.motion.max_hold_steps_before_force_command = 1600

        # ------------------------------------------------------------
        # 2. Command range: keep y = +-0.30, add small x/yaw correction.
        # ------------------------------------------------------------
        # Do not jump to +-0.5 yet.
        self.commands.base_velocity.ranges = mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.05, 0.05),
            lin_vel_y=(-0.30, 0.30),
            ang_vel_z=(-0.10, 0.10),
            heading=(-3.14, 3.14),
        )

        # ------------------------------------------------------------
        # 3. Velocity tracking.
        # ------------------------------------------------------------
        # Keep y strong, but not stronger than V32.
        self.rewards.track_lin_vel_y.weight = 0.90
        self.rewards.track_lin_vel_y.params["std"] = 0.16

        # Add mild x/yaw correction.
        # This is not full locomotion yet, just enough to self-align
        # and avoid drifting into the platform.
        self.rewards.track_lin_vel_xy.weight = 0.10
        self.rewards.track_lin_vel_xy.params["std"] = 0.14

        self.rewards.track_yaw_rate.weight = 0.08
        self.rewards.track_yaw_rate.params["std"] = 0.14

        # ------------------------------------------------------------
        # 4. Keep V32 diagonal y stepping.
        # ------------------------------------------------------------
        self.rewards.y_cmd_leg_motion.weight = 0.025
        self.rewards.y_cmd_diagonal_leg_motion.weight = 0.050
        self.rewards.y_cmd_diagonal_motion_balance.weight = -0.025

        # ------------------------------------------------------------
        # 5. Remove failed clearance penalties.
        # ------------------------------------------------------------
        if hasattr(self.rewards, "stage2_base_box_x_clearance_penalty"):
            self.rewards.stage2_base_box_x_clearance_penalty = None
        if hasattr(self.rewards, "stage2_body_box_x_clearance_penalty"):
            self.rewards.stage2_body_box_x_clearance_penalty = None
        if hasattr(self.rewards, "stage2_base_lateral_edge_penalty"):
            self.rewards.stage2_base_lateral_edge_penalty = None
        if hasattr(self.rewards, "stage2_feet_lateral_edge_penalty"):
            self.rewards.stage2_feet_lateral_edge_penalty = None

        # ------------------------------------------------------------
        # 6. Stability constraints.
        # ------------------------------------------------------------
        self.rewards.hip_abduction_penalty.weight = -0.10
        self.rewards.front_feet_contact.weight = 0.075
        self.rewards.base_pitch_excess.weight = -0.11
        self.rewards.base_tilt_penalty.weight = -0.10
        self.rewards.multi_foot_air_penalty.weight = -0.08

        # ------------------------------------------------------------
        # 7. Restore some V31-like stage0/stage1 protection.
        # ------------------------------------------------------------
        self.rewards.motion_global_anchor_pos.params["stage0_weight"] = 1.08
        self.rewards.motion_global_anchor_pos.params["stage1_weight"] = 0.90
        self.rewards.motion_global_anchor_pos.params["stage2_weight"] = 0.00

        self.rewards.motion_global_anchor_ori.params["stage0_weight"] = 1.08
        self.rewards.motion_global_anchor_ori.params["stage1_weight"] = 0.90
        self.rewards.motion_global_anchor_ori.params["stage2_weight"] = 0.15

        self.rewards.motion_body_pos.params["stage0_weight"] = 1.08
        self.rewards.motion_body_pos.params["stage1_weight"] = 1.23
        self.rewards.motion_body_pos.params["stage2_weight"] = 0.75

        self.rewards.motion_body_ori.params["stage0_weight"] = 1.08
        self.rewards.motion_body_ori.params["stage1_weight"] = 1.23
        self.rewards.motion_body_ori.params["stage2_weight"] = 0.75

        self.rewards.motion_body_lin_vel.params["stage0_weight"] = 1.05
        self.rewards.motion_body_lin_vel.params["stage1_weight"] = 0.35
        self.rewards.motion_body_lin_vel.params["stage2_weight"] = 0.00

        self.rewards.motion_body_ang_vel.params["stage0_weight"] = 1.05
        self.rewards.motion_body_ang_vel.params["stage1_weight"] = 0.35
        self.rewards.motion_body_ang_vel.params["stage2_weight"] = 0.00

        self.rewards.motion_joint_pos.params["stage0_weight"] = 1.08
        self.rewards.motion_joint_pos.params["stage1_weight"] = 1.28
        self.rewards.motion_joint_pos.params["stage2_weight"] = 0.60

        self.rewards.motion_joint_vel.params["stage0_weight"] = 1.00
        self.rewards.motion_joint_vel.params["stage1_weight"] = 0.55
        self.rewards.motion_joint_vel.params["stage2_weight"] = 0.30

        self.rewards.motion_wheel_joint_vel.params["stage0_weight"] = 1.00
        self.rewards.motion_wheel_joint_vel.params["stage1_weight"] = 0.40
        self.rewards.motion_wheel_joint_vel.params["stage2_weight"] = 0.00

# 2026-05-25_10-21-41/model_9000.pt reset不会翻倒，0.3的速度才响应，横向移动走的挺直的，不贴高台，前进后退旋转基本没响应
#delay 抑制了无 delay 版本里那种“快速贴台蹭着走”的投机解，同时 V32Safe 的轻微 x/yaw 修正和稳定项让横移更直
@configclass
class Stage2CommandEnvCfgV32delay(Stage2CommandEnvCfgV32Safe):
    """V32Safe with delayed PD actuators for sim-to-real finetuning.

    Goal:
    - inherit all Stage2CommandEnvCfgV32Safe stage/reward/command settings
    - use delayed actuator model aligned with PcbABeyondMimicFlatV1StandDelayEnvCfg
    - prepare policy for real robot deployment
    """

    def __post_init__(self):
        super().__post_init__()

        # Use DelayedPDActuator for sim-to-real finetuning.
        # Keep this active for both baseline and stage2 phases.
        self.scene.robot.actuators = {
            "legs_hip": DelayedPDActuatorCfg(
                joint_names_expr=[".*_hip_joint"],
                effort_limit=60.0,
                velocity_limit=14.13,
                stiffness=35.0,
                damping=0.8,
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

# 2026-05-25_16-54-40/model_29999.pt横向移动0.2/0.3响应，左边会越来越贴近高台，右边会越来越远离高台，直到掉下来。前后速度只晃动，旋转完全不响应
@configclass
class Stage2CommandEnvCfgV4(Stage2CommandEnvCfgV32delay):
    """V4: delayed-actuator command fine-tune.

    Goal:
    - keep delayed actuator sim-to-real setting
    - keep stable stage0/reset behavior from V32delay
    - keep straight lateral movement without sticking to the platform
    - improve 0.15~0.2 lateral response
    - keep 0.3 lateral response
    - mildly recover x forward/backward and yaw response
    """

    def __post_init__(self):
        super().__post_init__()

        command_phase = os.getenv("PCBA_COMMAND_PHASE", "stage2").strip().lower()
        if command_phase == "baseline":
            return

        # ------------------------------------------------------------
        # 1. Keep V32delay stable stage timing.
        # ------------------------------------------------------------
        # 9000 play 已经验证：
        # - reset 不翻
        # - stage0 能正常模仿轨迹
        # - stage2 横移不贴高台
        #
        # 所以这里不再激进提前 stage2。
        self.commands.motion.hold_stable_steps = 75
        self.commands.motion.extra_hold_steps_after_stable = 110
        self.commands.motion.max_hold_steps_before_force_command = 1600

        # ------------------------------------------------------------
        # 2. Command range.
        # ------------------------------------------------------------
        # y 仍然是主任务，保持 +-0.30。
        # x/yaw 稍微加大，让 policy 有姿态修正和轻微前后/旋转能力。
        #
        # 不建议现在 y 直接上 +-0.50。
        self.commands.base_velocity.ranges = mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.12, 0.12),
            lin_vel_y=(-0.30, 0.30),
            ang_vel_z=(-0.20, 0.20),
            heading=(-3.14, 3.14),
        )

        # ------------------------------------------------------------
        # 3. Main y velocity tracking.
        # ------------------------------------------------------------
        # V32delay 中 vy=0.3 可以，vy=0.2 弱。
        # 所以这里 y tracking 保持主导，但不继续过分加大。
        self.rewards.track_lin_vel_y.weight = 0.95
        self.rewards.track_lin_vel_y.params["std"] = 0.16

        # ------------------------------------------------------------
        # 4. Fine y tracking for 0.15~0.2 response.
        # ------------------------------------------------------------
        # 这个是额外的小 std y 跟踪项。
        # 目的不是扩大最大速度，而是让中小 y command 也有更明显梯度。
        self.rewards.track_lin_vel_y_fine = RewTerm(
            func=mdp.stage_gated_track_lin_vel_axis_projected_body_exp,
            weight=0.14,
            params={
                "command_name": "base_velocity",
                "axis": 1,
                "std": 0.08,
                "stage_command_name": "motion",
                "enabled_stage": 2,
            },
        )

        # ------------------------------------------------------------
        # 5. Recover x forward/backward tracking.
        # ------------------------------------------------------------
        # 之前 x 基本没有响应，原因是：
        # - x range 小；
        # - x reward 权重小；
        # - y 任务和 imitation/stability 把它压住了。
        #
        # 这里给 x 一个真实但仍然温和的学习信号。
        self.rewards.track_lin_vel_xy.weight = 0.22
        self.rewards.track_lin_vel_xy.params["std"] = 0.14

        # ------------------------------------------------------------
        # 6. Recover yaw tracking.
        # ------------------------------------------------------------
        # yaw 也不能太大，否则会在高台上诱发扭身、外八、前轮支撑不稳。
        # 这里先给到 0.18。
        self.rewards.track_yaw_rate.weight = 0.18
        self.rewards.track_yaw_rate.params["std"] = 0.14

        # ------------------------------------------------------------
        # 7. Keep diagonal lateral stepping.
        # ------------------------------------------------------------
        # 这些是 V3/V32 系列里横向小步能出来的关键项。
        # V4 不继续加大，避免腿抖/外八加剧。
        self.rewards.y_cmd_leg_motion.weight = 0.025
        self.rewards.y_cmd_diagonal_leg_motion.weight = 0.050
        self.rewards.y_cmd_diagonal_motion_balance.weight = -0.025

        # ------------------------------------------------------------
        # 8. Keep stability constraints.
        # ------------------------------------------------------------
        # delay 版本已经不贴高台、reset 不翻；
        # 这些约束先保持，不要放松。
        self.rewards.hip_abduction_penalty.weight = -0.10
        self.rewards.front_feet_contact.weight = 0.075
        self.rewards.base_pitch_excess.weight = -0.11
        self.rewards.base_tilt_penalty.weight = -0.10
        self.rewards.multi_foot_air_penalty.weight = -0.08

        # ------------------------------------------------------------
        # 9. Do NOT use failed clearance penalties.
        # ------------------------------------------------------------
        # V33 / V33Fix 证明 clearance penalty 这条线效果不好。世界 x 方向。如果机身 yaw 有偏、或者高台/轨迹参考方向和世界坐标有偏
        # V4 继续禁用它们。
        if hasattr(self.rewards, "stage2_base_box_x_clearance_penalty"):
            self.rewards.stage2_base_box_x_clearance_penalty = None
        if hasattr(self.rewards, "stage2_body_box_x_clearance_penalty"):
            self.rewards.stage2_body_box_x_clearance_penalty = None
        if hasattr(self.rewards, "stage2_base_lateral_edge_penalty"):
            self.rewards.stage2_base_lateral_edge_penalty = None
        if hasattr(self.rewards, "stage2_feet_lateral_edge_penalty"):
            self.rewards.stage2_feet_lateral_edge_penalty = None

        # ------------------------------------------------------------
        # 10. Keep stage0/stage1 imitation protection.
        # ------------------------------------------------------------
        # 这部分保证 reset 后还能正常模仿轨迹，不因为 stage2 command finetune 忘掉上台过程。
        self.rewards.motion_global_anchor_pos.params["stage0_weight"] = 1.08
        self.rewards.motion_global_anchor_pos.params["stage1_weight"] = 0.90
        self.rewards.motion_global_anchor_pos.params["stage2_weight"] = 0.00

        self.rewards.motion_global_anchor_ori.params["stage0_weight"] = 1.08
        self.rewards.motion_global_anchor_ori.params["stage1_weight"] = 0.90
        self.rewards.motion_global_anchor_ori.params["stage2_weight"] = 0.15

        self.rewards.motion_body_pos.params["stage0_weight"] = 1.08
        self.rewards.motion_body_pos.params["stage1_weight"] = 1.23
        self.rewards.motion_body_pos.params["stage2_weight"] = 0.75

        self.rewards.motion_body_ori.params["stage0_weight"] = 1.08
        self.rewards.motion_body_ori.params["stage1_weight"] = 1.23
        self.rewards.motion_body_ori.params["stage2_weight"] = 0.75

        self.rewards.motion_body_lin_vel.params["stage0_weight"] = 1.05
        self.rewards.motion_body_lin_vel.params["stage1_weight"] = 0.35
        self.rewards.motion_body_lin_vel.params["stage2_weight"] = 0.00

        self.rewards.motion_body_ang_vel.params["stage0_weight"] = 1.05
        self.rewards.motion_body_ang_vel.params["stage1_weight"] = 0.35
        self.rewards.motion_body_ang_vel.params["stage2_weight"] = 0.00

        self.rewards.motion_joint_pos.params["stage0_weight"] = 1.08
        self.rewards.motion_joint_pos.params["stage1_weight"] = 1.28
        self.rewards.motion_joint_pos.params["stage2_weight"] = 0.60

        self.rewards.motion_joint_vel.params["stage0_weight"] = 1.00
        self.rewards.motion_joint_vel.params["stage1_weight"] = 0.55
        self.rewards.motion_joint_vel.params["stage2_weight"] = 0.30

        self.rewards.motion_wheel_joint_vel.params["stage0_weight"] = 1.00
        self.rewards.motion_wheel_joint_vel.params["stage1_weight"] = 0.40
        self.rewards.motion_wheel_joint_vel.params["stage2_weight"] = 0.00

# 加 anchor-local x drift 约束，解决纯 vy 横移时左贴右远离；增大x速度为0.2并提高权重 保留 V4 的横向移动能力；
# 2026-05-26_10-26-22/model_6000.pt 0.2/0.3都能响应，左边不贴/右边贴
@configclass
class Stage2CommandEnvCfgV41(Stage2CommandEnvCfgV4):
    """V4.1: delayed-actuator command fine-tune with anchor-local x drift control.

    Goal:
    - keep V4 delayed actuator behavior
    - keep V4 lateral walking ability
    - improve vx / yaw response with 0.2-level commands
    - prevent pure-y walking from drifting into / away from platform
    - use anchor yaw local frame, not raw world x
    - do not reintroduce V33/V33Fix clearance penalties
    """

    def __post_init__(self):
        super().__post_init__()

        command_phase = os.getenv("PCBA_COMMAND_PHASE", "stage2").strip().lower()
        if command_phase == "baseline":
            return

        # ------------------------------------------------------------
        # 1. Keep V4 / V32delay stage timing.
        # ------------------------------------------------------------
        self.commands.motion.hold_stable_steps = 75
        self.commands.motion.extra_hold_steps_after_stable = 110
        self.commands.motion.max_hold_steps_before_force_command = 1600

        # ------------------------------------------------------------
        # 2. Command range.
        # ------------------------------------------------------------
        # 你前面多版 play 都说明：0.2 以下速度基本不明显响应。
        # 所以 V41 里 vx / yaw 都给到 0.2 级别。
        #
        # 注意：
        # - lin_vel_x 是前进/后退；
        # - lin_vel_y 是横向；
        # - ang_vel_z 是绕 z 轴 yaw 旋转；
        # - 不是给竖直 z 速度。
        self.commands.base_velocity.ranges = mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.20, 0.20),
            lin_vel_y=(-0.30, 0.30),
            ang_vel_z=(-0.20, 0.20),
            heading=(-3.14, 3.14),
        )

        # ------------------------------------------------------------
        # 3. Main y velocity tracking.
        # ------------------------------------------------------------
        # y 仍然是主任务，但不要继续加大。
        # 继续加 y 只会让横移更快，不能解决左贴/右远离问题。
        self.rewards.track_lin_vel_y.weight = 0.90
        self.rewards.track_lin_vel_y.params["std"] = 0.16

        # ------------------------------------------------------------
        # 4. Fine y tracking.
        # ------------------------------------------------------------
        # 用于改善 vy=0.15~0.2 时只动一下的问题。
        # 这个项保留，但权重略低于 V4，给 x/yaw 留奖励空间。
        if not hasattr(self.rewards, "track_lin_vel_y_fine") or self.rewards.track_lin_vel_y_fine is None:
            self.rewards.track_lin_vel_y_fine = RewTerm(
                func=mdp.stage_gated_track_lin_vel_axis_projected_body_exp,
                weight=0.12,
                params={
                    "command_name": "base_velocity",
                    "axis": 1,
                    "std": 0.08,
                    "stage_command_name": "motion",
                    "enabled_stage": 2,
                },
            )
        else:
            self.rewards.track_lin_vel_y_fine.weight = 0.12
            self.rewards.track_lin_vel_y_fine.params["std"] = 0.08

        # ------------------------------------------------------------
        # 5. Strengthen x / yaw response.
        # ------------------------------------------------------------
        # V4 中 vx/yaw 仍然基本不响应；
        # V41 里把 x/yaw 指令和 reward 都提到有真实学习压力的级别。
        self.rewards.track_lin_vel_xy.weight = 0.32
        self.rewards.track_lin_vel_xy.params["std"] = 0.16

        self.rewards.track_yaw_rate.weight = 0.28
        self.rewards.track_yaw_rate.params["std"] = 0.16

        # ------------------------------------------------------------
        # 6. Anchor-local x drift penalty.
        # ------------------------------------------------------------
        # 这是 V41 的核心修复项。
        #
        # 解决的问题：
        # - 左向横移越来越贴高台；
        # - 右向横移越来越远离高台；
        # - 纯 y 横移久了掉下高台。
        #
        # 它不是 world-x clearance penalty。
        # 它把 robot anchor 相对 motion anchor 的位置误差投影到
        # motion anchor yaw frame 里，只约束 local x。
        #
        # axis=0:
        #   local forward/backward drift
        #
        # command_axis=0 + command_deadzone=0.05:
        #   只有 vx command 接近 0 的时候启用；
        #   如果你主动给 vx 前进/后退，它不会和 vx tracking 冲突。
        self.rewards.stage2_anchor_local_x_drift = RewTerm(
            func=mdp.stage_gated_anchor_local_axis_drift_l2_penalty,
            weight=-0.06,
            params={
                "command_name": "base_velocity",
                "motion_command_name": "motion",
                "asset_cfg": SceneEntityCfg("robot"),
                "axis": 0,
                "deadband": 0.10,
                "command_axis": 0,
                "command_deadzone": 0.05,
                "max_penalty": 1.0,
                "enabled_stage": 2,
            },
        )

        # ------------------------------------------------------------
        # 7. Keep diagonal lateral stepping.
        # ------------------------------------------------------------
        # 这几个项是横向小步能出来的关键。
        # 不继续加大，避免重新外八、腿抖、步态过激。
        self.rewards.y_cmd_leg_motion.weight = 0.025
        self.rewards.y_cmd_diagonal_leg_motion.weight = 0.050
        self.rewards.y_cmd_diagonal_motion_balance.weight = -0.025

        # ------------------------------------------------------------
        # 8. Stability constraints.
        # ------------------------------------------------------------
        # 因为 V41 提高了 x/yaw tracking，这里略微加强稳定项。
        self.rewards.hip_abduction_penalty.weight = -0.11
        self.rewards.front_feet_contact.weight = 0.08
        self.rewards.base_pitch_excess.weight = -0.12
        self.rewards.base_tilt_penalty.weight = -0.11
        self.rewards.multi_foot_air_penalty.weight = -0.08

        # ------------------------------------------------------------
        # 9. Relax stage2 yaw/orientation locking.
        # ------------------------------------------------------------
        # V4 yaw 完全不响应，一个原因是 stage2 仍有 global/body ori 约束。
        # 这里轻微放松 stage2 orientation，不影响 stage0/stage1。
        self.rewards.motion_global_anchor_ori.params["stage2_weight"] = 0.08
        self.rewards.motion_body_ori.params["stage2_weight"] = 0.70
        self.rewards.motion_joint_pos.params["stage2_weight"] = 0.58

        # ------------------------------------------------------------
        # 10. Disable failed clearance / edge penalties.
        # ------------------------------------------------------------
        # V33 / V33Fix 的 world-x clearance 已经验证效果不好。
        # V41 用 anchor-local x drift 替代它们。
        if hasattr(self.rewards, "stage2_base_box_x_clearance_penalty"):
            self.rewards.stage2_base_box_x_clearance_penalty = None
        if hasattr(self.rewards, "stage2_body_box_x_clearance_penalty"):
            self.rewards.stage2_body_box_x_clearance_penalty = None
        if hasattr(self.rewards, "stage2_base_lateral_edge_penalty"):
            self.rewards.stage2_base_lateral_edge_penalty = None
        if hasattr(self.rewards, "stage2_feet_lateral_edge_penalty"):
            self.rewards.stage2_feet_lateral_edge_penalty = None

        # ------------------------------------------------------------
        # 11. Keep stage0/stage1 imitation protection.
        # ------------------------------------------------------------
        # 不破坏 reset / 上台轨迹模仿。
        self.rewards.motion_global_anchor_pos.params["stage0_weight"] = 1.08
        self.rewards.motion_global_anchor_pos.params["stage1_weight"] = 0.90
        self.rewards.motion_global_anchor_pos.params["stage2_weight"] = 0.00

        self.rewards.motion_global_anchor_ori.params["stage0_weight"] = 1.08
        self.rewards.motion_global_anchor_ori.params["stage1_weight"] = 0.90

        self.rewards.motion_body_pos.params["stage0_weight"] = 1.08
        self.rewards.motion_body_pos.params["stage1_weight"] = 1.23
        self.rewards.motion_body_pos.params["stage2_weight"] = 0.75

        self.rewards.motion_body_ori.params["stage0_weight"] = 1.08
        self.rewards.motion_body_ori.params["stage1_weight"] = 1.23

        self.rewards.motion_body_lin_vel.params["stage0_weight"] = 1.05
        self.rewards.motion_body_lin_vel.params["stage1_weight"] = 0.35
        self.rewards.motion_body_lin_vel.params["stage2_weight"] = 0.00

        self.rewards.motion_body_ang_vel.params["stage0_weight"] = 1.05
        self.rewards.motion_body_ang_vel.params["stage1_weight"] = 0.35
        self.rewards.motion_body_ang_vel.params["stage2_weight"] = 0.00

        self.rewards.motion_joint_pos.params["stage0_weight"] = 1.08
        self.rewards.motion_joint_pos.params["stage1_weight"] = 1.28

        self.rewards.motion_joint_vel.params["stage0_weight"] = 1.00
        self.rewards.motion_joint_vel.params["stage1_weight"] = 0.55
        self.rewards.motion_joint_vel.params["stage2_weight"] = 0.30

        self.rewards.motion_wheel_joint_vel.params["stage0_weight"] = 1.00
        self.rewards.motion_wheel_joint_vel.params["stage1_weight"] = 0.40
        self.rewards.motion_wheel_joint_vel.params["stage2_weight"] = 0.00

@configclass
class Stage2CommandEnvCfgV42(Stage2CommandEnvCfgV41):
    """V4.2: wheel-guided vx/yaw with rear-wheel participation.

    Goal:
    - keep V41 delayed-actuator lateral stability
    - keep improved small-y response
    - make vx use all wheels, not only front-leg/front-wheel reaching
    - make yaw use wheel differential, with rear wheels participating
    - suppress front-leg over-extension under vx/yaw
    - avoid returning to old body-twist / leg-splay solution
    """

    def __post_init__(self):
        super().__post_init__()

        command_phase = os.getenv("PCBA_COMMAND_PHASE", "stage2").strip().lower()
        if command_phase == "baseline":
            return

        # ------------------------------------------------------------
        # 1. Stage2 exposure.
        # ------------------------------------------------------------
        # Keep old V42's stronger stage2 exposure.
        # Do not make it more aggressive; current issue is mechanism, not lack of stage2 samples.
        self.commands.motion.hold_stable_steps = 60
        self.commands.motion.extra_hold_steps_after_stable = 80
        self.commands.motion.max_hold_steps_before_force_command = 1100

        # ------------------------------------------------------------
        # 2. Command range.
        # ------------------------------------------------------------
        # Keep 0.2-level vx/yaw because smaller commands were repeatedly ignored.
        # y remains the main useful command.
        self.commands.base_velocity.ranges = mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.20, 0.20),
            lin_vel_y=(-0.30, 0.30),
            ang_vel_z=(-0.25, 0.25),
            heading=(-3.14, 3.14),
        )

        if hasattr(self.commands.base_velocity, "rel_standing_envs"):
            self.commands.base_velocity.rel_standing_envs = 0.05

        # ------------------------------------------------------------
        # 3. Keep y lateral response.
        # ------------------------------------------------------------
        # model_3000 shows vy=0.1 already responds, so do not weaken y too much.
        self.rewards.track_lin_vel_y.weight = 0.78
        self.rewards.track_lin_vel_y.params["std"] = 0.16

        if not hasattr(self.rewards, "track_lin_vel_y_fine") or self.rewards.track_lin_vel_y_fine is None:
            self.rewards.track_lin_vel_y_fine = RewTerm(
                func=mdp.stage_gated_track_lin_vel_axis_projected_body_exp,
                weight=0.06,
                params={
                    "command_name": "base_velocity",
                    "axis": 1,
                    "std": 0.08,
                    "stage_command_name": "motion",
                    "enabled_stage": 2,
                },
            )
        else:
            self.rewards.track_lin_vel_y_fine.weight = 0.06
            self.rewards.track_lin_vel_y_fine.params["std"] = 0.08

        # ------------------------------------------------------------
        # 4. Direct vx / yaw tracking.
        # ------------------------------------------------------------
        # Keep these moderate. Do not increase direct body tracking,
        # otherwise the policy returns to body-twist / front-leg reaching.
        self.rewards.track_lin_vel_xy.weight = 0.36
        self.rewards.track_lin_vel_xy.params["std"] = 0.16

        self.rewards.track_lin_vel_x_fine = RewTerm(
            func=mdp.stage_gated_track_lin_vel_axis_projected_body_exp,
            weight=0.06,
            params={
                "command_name": "base_velocity",
                "axis": 0,
                "std": 0.08,
                "stage_command_name": "motion",
                "enabled_stage": 2,
            },
        )

        self.rewards.track_yaw_rate.weight = 0.36
        self.rewards.track_yaw_rate.params["std"] = 0.18

        self.rewards.track_yaw_rate_fine = RewTerm(
            func=mdp.stage_gated_track_ang_vel_z_projected_exp,
            weight=0.06,
            params={
                "command_name": "base_velocity",
                "std": 0.10,
                "stage_command_name": "motion",
                "enabled_stage": 2,
            },
        )

        # ------------------------------------------------------------
        # 5. Wheel rolling for vx.
        # ------------------------------------------------------------
        # Slightly stronger than previous V42.
        # If vx direction is opposite in play, flip direction to -1.0.
        self.rewards.x_cmd_wheel_vel = RewTerm(
            func=mdp.stage_gated_x_command_wheel_velocity_reward,
            weight=0.18,
            params={
                "command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg("robot", joint_names=self.wheel_joint_names),
                "x_cmd_threshold": 0.05,
                "target_scale": 18.0,
                "std": 5.0,
                "stage_command_name": "motion",
                "enabled_stage": 2,
                "direction": 1.0,
            },
        )

        # ------------------------------------------------------------
        # 6. Wheel differential for yaw.
        # ------------------------------------------------------------
        # Slightly stronger than previous V42.
        # If yaw direction is opposite in play, flip direction to -1.0.
        self.rewards.yaw_cmd_wheel_diff = RewTerm(
            func=mdp.stage_gated_yaw_command_wheel_diff_velocity_reward,
            weight=0.14,
            params={
                "command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg("robot"),
                "left_wheel_joint_names": ["FL_foot_joint", "RL_foot_joint"],
                "right_wheel_joint_names": ["FR_foot_joint", "RR_foot_joint"],
                "yaw_cmd_threshold": 0.05,
                "target_scale": 10.0,
                "std": 5.0,
                "stage_command_name": "motion",
                "enabled_stage": 2,
                "direction": 1.0,
            },
        )

        # ------------------------------------------------------------
        # 7. New: rear wheel participation.
        # ------------------------------------------------------------
        # model_3000 shows front legs/wheels respond, rear wheels mostly do not.
        # This term prevents rear wheels from staying passive under vx/yaw commands.
        self.rewards.rear_wheel_motion = RewTerm(
            func=mdp.stage_gated_rear_wheel_motion_reward,
            weight=0.08,
            params={
                "command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg("robot"),
                "rear_wheel_joint_names": ["RL_foot_joint", "RR_foot_joint"],
                "cmd_threshold": 0.05,
                "max_value": 18.0,
                "stage_command_name": "motion",
                "enabled_stage": 2,
            },
        )

        # ------------------------------------------------------------
        # 8. New: front/rear wheel balance.
        # ------------------------------------------------------------
        # Prevents the policy from using only front wheels/front legs.
        self.rewards.wheel_front_rear_balance = RewTerm(
            func=mdp.stage_gated_wheel_front_rear_balance_penalty,
            weight=-0.08,
            params={
                "command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg("robot"),
                "front_wheel_joint_names": ["FL_foot_joint", "FR_foot_joint"],
                "rear_wheel_joint_names": ["RL_foot_joint", "RR_foot_joint"],
                "cmd_threshold": 0.05,
                "max_value": 20.0,
                "stage_command_name": "motion",
                "enabled_stage": 2,
            },
        )

        # ------------------------------------------------------------
        # 9. New: front leg over-extension penalty.
        # ------------------------------------------------------------
        # model_3000 shows vx command causes front legs to stretch forward.
        # This term suppresses that local optimum.
        self.rewards.front_leg_extension_penalty = RewTerm(
            func=mdp.stage_gated_command_front_leg_extension_penalty,
            weight=-0.10,
            params={
                "command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg("robot"),
                "joint_names": [
                    "FL_thigh_joint",
                    "FR_thigh_joint",
                    "FL_calf_joint",
                    "FR_calf_joint",
                ],
                "cmd_threshold": 0.05,
                "deadband": 0.25,
                "stage_command_name": "motion",
                "enabled_stage": 2,
            },
        )

        # ------------------------------------------------------------
        # 10. Weak local-x drift helper only.
        # ------------------------------------------------------------
        # Keep this weak. Operator should be able to correct x/yaw manually.
        # This should not suppress vx learning.
        if hasattr(self.rewards, "stage2_anchor_local_x_drift"):
            self.rewards.stage2_anchor_local_x_drift.weight = -0.02
            self.rewards.stage2_anchor_local_x_drift.params["deadband"] = 0.10
            self.rewards.stage2_anchor_local_x_drift.params["command_deadzone"] = 0.05
            self.rewards.stage2_anchor_local_x_drift.params["max_penalty"] = 1.0

        # ------------------------------------------------------------
        # 11. Stage2 imitation balance.
        # ------------------------------------------------------------
        # Keep enough structure, but allow wheel-driven vx/yaw.
        self.rewards.motion_global_anchor_ori.params["stage2_weight"] = 0.00
        self.rewards.motion_body_ori.params["stage2_weight"] = 0.50
        self.rewards.motion_body_pos.params["stage2_weight"] = 0.68
        self.rewards.motion_joint_pos.params["stage2_weight"] = 0.45
        self.rewards.motion_joint_vel.params["stage2_weight"] = 0.30

        # ------------------------------------------------------------
        # 12. Keep y stepping, but weaker than V32.
        # ------------------------------------------------------------
        # Do not let y-specific leg motion dominate vx/yaw.
        self.rewards.y_cmd_leg_motion.weight = 0.012
        self.rewards.y_cmd_diagonal_leg_motion.weight = 0.030
        self.rewards.y_cmd_diagonal_motion_balance.weight = -0.018

        # ------------------------------------------------------------
        # 13. Stability and anti-leg-splay constraints.
        # ------------------------------------------------------------
        self.rewards.hip_abduction_penalty.weight = -0.13
        self.rewards.front_feet_contact.weight = 0.08
        self.rewards.base_pitch_excess.weight = -0.13
        self.rewards.base_tilt_penalty.weight = -0.12
        self.rewards.multi_foot_air_penalty.weight = -0.08

        # ------------------------------------------------------------
        # 14. Disable failed clearance / edge penalties.
        # ------------------------------------------------------------
        if hasattr(self.rewards, "stage2_base_box_x_clearance_penalty"):
            self.rewards.stage2_base_box_x_clearance_penalty = None
        if hasattr(self.rewards, "stage2_body_box_x_clearance_penalty"):
            self.rewards.stage2_body_box_x_clearance_penalty = None
        if hasattr(self.rewards, "stage2_base_lateral_edge_penalty"):
            self.rewards.stage2_base_lateral_edge_penalty = None
        if hasattr(self.rewards, "stage2_feet_lateral_edge_penalty"):
            self.rewards.stage2_feet_lateral_edge_penalty = None

        # ------------------------------------------------------------
        # 15. Keep stage0/stage1 imitation protection.
        # ------------------------------------------------------------
        self.rewards.motion_global_anchor_pos.params["stage0_weight"] = 1.08
        self.rewards.motion_global_anchor_pos.params["stage1_weight"] = 0.90
        self.rewards.motion_global_anchor_pos.params["stage2_weight"] = 0.00

        self.rewards.motion_global_anchor_ori.params["stage0_weight"] = 1.08
        self.rewards.motion_global_anchor_ori.params["stage1_weight"] = 0.90
        # stage2_weight remains 0.00

        self.rewards.motion_body_pos.params["stage0_weight"] = 1.08
        self.rewards.motion_body_pos.params["stage1_weight"] = 1.23
        # stage2_weight remains 0.68

        self.rewards.motion_body_ori.params["stage0_weight"] = 1.08
        self.rewards.motion_body_ori.params["stage1_weight"] = 1.23
        # stage2_weight remains 0.50

        self.rewards.motion_body_lin_vel.params["stage0_weight"] = 1.05
        self.rewards.motion_body_lin_vel.params["stage1_weight"] = 0.35
        self.rewards.motion_body_lin_vel.params["stage2_weight"] = 0.00

        self.rewards.motion_body_ang_vel.params["stage0_weight"] = 1.05
        self.rewards.motion_body_ang_vel.params["stage1_weight"] = 0.35
        self.rewards.motion_body_ang_vel.params["stage2_weight"] = 0.00

        self.rewards.motion_joint_pos.params["stage0_weight"] = 1.08
        self.rewards.motion_joint_pos.params["stage1_weight"] = 1.28
        # stage2_weight remains 0.45

        self.rewards.motion_joint_vel.params["stage0_weight"] = 1.00
        self.rewards.motion_joint_vel.params["stage1_weight"] = 0.55
        # stage2_weight remains 0.30

        self.rewards.motion_wheel_joint_vel.params["stage0_weight"] = 1.00
        self.rewards.motion_wheel_joint_vel.params["stage1_weight"] = 0.40
        self.rewards.motion_wheel_joint_vel.params["stage2_weight"] = 0.00

@configclass
class PcbABeyondMimicFlatV38RootEnvCfg(PcbABeyondMimicFlatBaseEnvCfg):
    """v3.8 profile: root-tilted start motion."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.motion_file = (
            f"{os.path.dirname(__file__)}/../go2w/motion/pcbA_80cm_60hz_lateral_move_v3.8.npz"
        )
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

