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
from robot_lab.assets.pcbC import pcbC_CFG
from robot_lab.tasks.manager_based.beyondmimic.tracking_env_cfg_go2w import BeyondMimicEnvCfg


PCBC_MOTION_FILE = f"{os.path.dirname(__file__)}/../go2w/motion/pcbv2_x6u_60hz_stage_up.npz"


def pcbc_default_joint_pos() -> dict[str, float]:
    """pcbC default standing posture.

    NPZ joint order is type-grouped:
    4 hips, 4 thighs, 4 calves, 4 wheels.
    """
    return {
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
    }


def apply_pcbc_delayed_actuators(env_cfg) -> None:
    """Use delayed PD actuators for pcbC delay fine-tuning tasks."""
    env_cfg.scene.robot.actuators = {
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
            velocity_limit=18.0,
            stiffness=0.0,
            damping=0.6,
            armature=0.0005103,
            friction=0.0,
            min_delay=0,
            max_delay=4,
        ),
    }


@configclass
class PcbCBeyondMimicFlatBaseEnvCfg(BeyondMimicEnvCfg):
    """pcbC base config.

    This base config is for high-precision non-delay mimic first.
    It does NOT force delayed actuators. Use delay classes only after baseline is stable.
    """

    base_link_name = "Base_link"
    foot_link_name = ".*_foot_link"

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

    # pcbC NPZ body order is type-grouped:
    # Base, 4 hips, 4 thighs, 4 calves, 4 feet.
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
        "FL_foot_link",
        "FR_foot_link",
        "RL_foot_link",
        "RR_foot_link",
    ]

    def __post_init__(self):
        super().__post_init__()

        # ------------------------------------------------------------
        # Robot and platform
        # ------------------------------------------------------------
        self.scene.robot = pcbC_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Keep replay/train aligned at x=0.97.
        self.scene.box.init_state.pos = (0.97, 0.0, 0.4)
        self.scene.box.spawn.size = (1.05, 8.80, 0.8)

        # ------------------------------------------------------------
        # Motion command mapping
        # ------------------------------------------------------------
        self.commands.motion.anchor_body_name = self.base_link_name
        self.commands.motion.body_names = self.body_names

        # ------------------------------------------------------------
        # Actions
        # Keep non-delay action scale same as current successful baseline.
        # ------------------------------------------------------------
        self.actions.joint_pos.scale = {
            ".*_hip_joint": 0.125,
            "^(?!.*_hip_joint).*": 0.25,
        }
        self.actions.joint_vel.scale = 5.0

        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_vel.clip = {".*": (-100.0, 100.0)}

        self.actions.joint_pos.joint_names = self.leg_joint_names
        self.actions.joint_vel.joint_names = self.wheel_joint_names

        # ------------------------------------------------------------
        # Observations
        # ------------------------------------------------------------
        self.observations.policy.joint_pos.func = mdp.joint_pos_rel_without_wheel
        self.observations.policy.joint_pos.params["wheel_asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.wheel_joint_names
        )

        self.observations.critic.joint_pos.func = mdp.joint_pos_rel_without_wheel
        self.observations.critic.joint_pos.params["wheel_asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.wheel_joint_names
        )

        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None

        # ------------------------------------------------------------
        # Domain randomization
        # Precision baseline from scratch: reduce perturbations first.
        # Add stronger randomization back only after mimic is stable.
        # ------------------------------------------------------------
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
                "com_range": {
                    "x": (-0.02, 0.02),
                    "y": (-0.02, 0.02),
                    "z": (-0.02, 0.02),
                },
            }
        )

        # Disable push in precision mimic baseline.
        self.events.randomize_push_robot = None

        self.events.randomize_leg_actuator_gains = None
        self.events.randomize_wheel_actuator_gains = None

        self.events.randomize_actuator_gains = EventTerm(
            func=mdp.randomize_actuator_gains,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
                "stiffness_distribution_params": (0.90, 1.10),
                "damping_distribution_params": (0.90, 1.30),
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
                "mass_distribution_params": (0.98, 1.02),
                "operation": "scale",
            },
        )

        # ------------------------------------------------------------
        # Pure mimic rewards: pcbC precision-from-scratch version.
        # Compared to pcbA-style weights:
        # - slightly reduce anchor dominance
        # - increase body_ori / joint_pos / joint_vel
        # - use wider std at from-scratch stage to keep gradients useful
        # ------------------------------------------------------------
        self.rewards.motion_global_anchor_pos.weight = 1.15
        self.rewards.motion_global_anchor_ori.weight = 1.10

        self.rewards.motion_body_pos.weight = 0.90
        self.rewards.motion_body_ori.weight = 1.05
        self.rewards.motion_body_lin_vel.weight = 0.55
        self.rewards.motion_body_ang_vel.weight = 0.45

        self.rewards.motion_body_ori.params["std"] = 0.50

        self.rewards.motion_joint_pos.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.leg_joint_names
        )
        self.rewards.motion_joint_vel.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.leg_joint_names
        )

        self.rewards.motion_joint_pos.weight = 0.50
        self.rewards.motion_joint_vel.weight = 0.30

        self.rewards.motion_joint_pos.params["std"] = 0.60
        self.rewards.motion_joint_vel.params["std"] = 1.00

        self.rewards.motion_wheel_joint_vel = RewTerm(
            func=mdp.motion_joint_velocity_error_exp,
            weight=0.06,
            params={
                "command_name": "motion",
                "std": 1.0,
                "asset_cfg": SceneEntityCfg("robot", joint_names=self.wheel_joint_names),
            },
        )

        self.rewards.action_rate_l2.func = mdp.action_rate_l2_clamped
        self.rewards.action_rate_l2.params = {
            "clip": 1.0,
            "max_value": 64.0,
        }
        self.rewards.action_rate_l2.weight = -1.0e-2

        # Same as pcbA: do not terminate on complex platform contacts by default.
        self.terminations.illegal_contact = None

        # pcbC trajectory is 14.94s. Use 16s for pure baseline.
        self.episode_length_s = 18


@configclass
class PcbCBeyondMimicFlatV1StandEnvCfg(PcbCBeyondMimicFlatBaseEnvCfg):
    """pcbC pure mimic task, non-delay version."""

    def __post_init__(self):
        super().__post_init__()

        self.commands.motion.motion_file = PCBC_MOTION_FILE

        self.scene.robot.init_state.pos = (0.0, 0.0, 0.45)
        self.scene.robot.init_state.rot = (1.0, 0.0, 0.0, 0.0)
        self.scene.robot.init_state.joint_pos = pcbc_default_joint_pos()

        self.rewards.motion_wheel_joint_vel.weight = 0.06


@configclass
class PcbCBeyondMimicFlatV1StandDelayEnvCfg(PcbCBeyondMimicFlatV1StandEnvCfg):
    """pcbC pure mimic task, delayed actuator version."""

    def __post_init__(self):
        super().__post_init__()
        apply_pcbc_delayed_actuators(self)


@configclass
class PcbCBeyondMimicFlatV1StandCommandEnvCfg(PcbCBeyondMimicFlatV1StandEnvCfg):
    """pcbC command task, non-delay version.

    PCBC_COMMAND_PHASE=baseline:
        pure high-precision motion imitation only.

    PCBC_COMMAND_PHASE=stage2:
        stage-gated velocity command finetune.
    """

    def __post_init__(self):
        super().__post_init__()

        command_phase = os.getenv("PCBC_COMMAND_PHASE", "stage2").strip().lower()

        self.commands.base_velocity = mdp.UniformVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(2.0, 4.0),
            rel_standing_envs=0.2,
            rel_heading_envs=0.0,
            heading_command=False,
            debug_vis=True,
            ranges=mdp.UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-0.02, 0.02),
                lin_vel_y=(-0.30, 0.30),
                ang_vel_z=(-0.06, 0.06),
                heading=(-3.14, 3.14),
            ),
        )

        # Stage logic: used only in stage2 branch.
        self.commands.motion.enable_stage_command = True
        self.commands.motion.hold_anchor_rot_threshold = 0.12
        self.commands.motion.hold_body_pos_threshold = 0.06
        self.commands.motion.hold_joint_pos_threshold = 0.12
        self.commands.motion.hold_base_lin_vel_threshold = 0.12
        self.commands.motion.hold_base_ang_vel_threshold = 0.35
        self.commands.motion.hold_stable_steps = 80
        self.commands.motion.extra_hold_steps_after_stable = 120
        self.commands.motion.max_hold_steps_before_force_command = 1600

        if command_phase == "baseline":
            self.commands.motion.enable_stage_command = False
            self.commands.motion.max_hold_steps_before_force_command = 0

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

            self.observations.policy.motion_stage = ObsTerm(
                func=mdp.constant_zero_scalar,
            )
            self.observations.critic.motion_stage = ObsTerm(
                func=mdp.constant_zero_scalar,
            )
            self.terminations.anchor_pos.params["threshold"] = 0.40

            return

        # ------------------------------------------------------------
        # Stage2 command phase
        # ------------------------------------------------------------
        self.episode_length_s = 40.0

        self.scene.box.init_state.pos = (0.97, 0.0, 0.4)
        self.scene.box.spawn.size = (1.05, 8.80, 0.8)

        self.observations.policy.base_velocity_command = ObsTerm(
            func=mdp.stage_gated_generated_commands,
            params={
                "command_name": "base_velocity",
                "stage_command_name": "motion",
                "enabled_stage": 2,
            },
        )
        self.observations.critic.base_velocity_command = ObsTerm(
            func=mdp.stage_gated_generated_commands,
            params={
                "command_name": "base_velocity",
                "stage_command_name": "motion",
                "enabled_stage": 2,
            },
        )

        self.observations.policy.motion_stage = ObsTerm(
            func=mdp.motion_stage,
            params={
                "command_name": "motion",
                "normalize": True,
            },
        )
        self.observations.critic.motion_stage = ObsTerm(
            func=mdp.motion_stage,
            params={
                "command_name": "motion",
                "normalize": True,
            },
        )

        # ------------------------------------------------------------
        # Stage-weighted mimic rewards
        # ------------------------------------------------------------
        self.rewards.motion_global_anchor_pos = RewTerm(
            func=mdp.stage_weighted_motion_global_anchor_position_error_exp,
            weight=1.30,
            params={
                "command_name": "motion",
                "std": 0.30,
                "stage_command_name": "motion",
                "stage0_weight": 1.00,
                "stage1_weight": 0.85,
                "stage2_weight": 0.00,
            },
        )

        self.rewards.motion_global_anchor_ori = RewTerm(
            func=mdp.stage_weighted_motion_global_anchor_orientation_error_exp,
            weight=1.20,
            params={
                "command_name": "motion",
                "std": 0.40,
                "stage_command_name": "motion",
                "stage0_weight": 1.00,
                "stage1_weight": 0.85,
                "stage2_weight": 0.15,
            },
        )

        self.rewards.motion_body_pos = RewTerm(
            func=mdp.stage_weighted_motion_relative_body_position_error_exp,
            weight=0.90,
            params={
                "command_name": "motion",
                "std": 0.30,
                "stage_command_name": "motion",
                "stage0_weight": 1.00,
                "stage1_weight": 1.20,
                "stage2_weight": 0.80,
            },
        )

        self.rewards.motion_body_ori = RewTerm(
            func=mdp.stage_weighted_motion_relative_body_orientation_error_exp,
            weight=0.95,
            params={
                "command_name": "motion",
                "std": 0.40,
                "stage_command_name": "motion",
                "stage0_weight": 1.00,
                "stage1_weight": 1.20,
                "stage2_weight": 0.75,
            },
        )

        self.rewards.motion_body_lin_vel = RewTerm(
            func=mdp.stage_weighted_motion_global_body_linear_velocity_error_exp,
            weight=0.60,
            params={
                "command_name": "motion",
                "std": 1.0,
                "stage_command_name": "motion",
                "stage0_weight": 1.00,
                "stage1_weight": 0.35,
                "stage2_weight": 0.00,
            },
        )

        self.rewards.motion_body_ang_vel = RewTerm(
            func=mdp.stage_weighted_motion_global_body_angular_velocity_error_exp,
            weight=0.50,
            params={
                "command_name": "motion",
                "std": 3.14,
                "stage_command_name": "motion",
                "stage0_weight": 1.00,
                "stage1_weight": 0.35,
                "stage2_weight": 0.00,
            },
        )

        self.rewards.motion_joint_pos = RewTerm(
            func=mdp.stage_weighted_motion_joint_position_error_exp,
            weight=0.40,
            params={
                "command_name": "motion",
                "std": 0.50,
                "asset_cfg": SceneEntityCfg("robot", joint_names=self.leg_joint_names),
                "stage_command_name": "motion",
                "stage0_weight": 1.00,
                "stage1_weight": 1.20,
                "stage2_weight": 0.65,
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
                "stage0_weight": 1.00,
                "stage1_weight": 0.55,
                "stage2_weight": 0.30,
            },
        )

        self.rewards.motion_wheel_joint_vel = RewTerm(
            func=mdp.stage_weighted_motion_joint_velocity_error_exp,
            weight=0.04,
            params={
                "command_name": "motion",
                "std": 1.0,
                "asset_cfg": SceneEntityCfg("robot", joint_names=self.wheel_joint_names),
                "stage_command_name": "motion",
                "stage0_weight": 1.00,
                "stage1_weight": 0.40,
                "stage2_weight": 0.00,
            },
        )

        # ------------------------------------------------------------
        # Stage2 velocity tracking: lateral y first.
        # ------------------------------------------------------------
        self.rewards.track_lin_vel_xy = RewTerm(
            func=mdp.stage_gated_track_lin_vel_axis_projected_body_exp,
            weight=0.04,
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
            weight=0.90,
            params={
                "command_name": "base_velocity",
                "axis": 1,
                "std": 0.16,
                "stage_command_name": "motion",
                "enabled_stage": 2,
            },
        )

        self.rewards.track_lin_vel_y_fine = RewTerm(
            func=mdp.stage_gated_track_lin_vel_axis_projected_body_exp,
            weight=0.10,
            params={
                "command_name": "base_velocity",
                "axis": 1,
                "std": 0.08,
                "stage_command_name": "motion",
                "enabled_stage": 2,
            },
        )

        self.rewards.track_yaw_rate = RewTerm(
            func=mdp.stage_gated_track_ang_vel_z_projected_exp,
            weight=0.03,
            params={
                "command_name": "base_velocity",
                "std": 0.16,
                "stage_command_name": "motion",
                "enabled_stage": 2,
            },
        )

        self.rewards.y_cmd_leg_lift = None
        self.rewards.rear_foot_air_time = None

        self.rewards.y_cmd_leg_motion = RewTerm(
            func=mdp.stage_gated_y_command_joint_motion_reward,
            weight=0.025,
            params={
                "command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg("robot", joint_names=self.leg_joint_names),
                "y_cmd_threshold": 0.004,
                "max_value": 2.0,
                "stage_command_name": "motion",
                "enabled_stage": 2,
            },
        )

        self.rewards.y_cmd_diagonal_leg_motion = RewTerm(
            func=mdp.stage_gated_y_command_diagonal_leg_motion_reward,
            weight=0.050,
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

        self.rewards.y_cmd_diagonal_motion_balance = RewTerm(
            func=mdp.stage_gated_y_command_diagonal_motion_balance_penalty,
            weight=-0.025,
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

        self.rewards.stage2_anchor_local_x_drift = RewTerm(
            func=mdp.stage_gated_anchor_local_axis_drift_l2_penalty,
            weight=-0.045,
            params={
                "command_name": "base_velocity",
                "motion_command_name": "motion",
                "asset_cfg": SceneEntityCfg("robot"),
                "axis": 0,
                "deadband": 0.10,
                "command_axis": 0,
                "command_deadzone": 0.08,
                "max_penalty": 1.0,
                "enabled_stage": 2,
            },
        )

        # ------------------------------------------------------------
        # Stage2 stability constraints
        # ------------------------------------------------------------
        self.rewards.hip_abduction_penalty = RewTerm(
            func=mdp.stage_gated_joint_deviation_l1_penalty,
            weight=-0.10,
            params={
                "joint_names": [
                    "FL_hip_joint",
                    "FR_hip_joint",
                    "RL_hip_joint",
                    "RR_hip_joint",
                ],
                "command_name": "motion",
                "enabled_stage": 2,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        self.rewards.base_tilt_penalty = RewTerm(
            func=mdp.stage_gated_base_tilt_l2_penalty,
            weight=-0.11,
            params={
                "command_name": "motion",
                "enabled_stage": 2,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        self.rewards.multi_foot_air_penalty = RewTerm(
            func=mdp.stage_gated_multi_foot_air_penalty,
            weight=-0.08,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot_link"),
                "max_air_feet": 2,
                "command_name": "motion",
                "enabled_stage": 2,
            },
        )

        self.rewards.front_feet_contact = RewTerm(
            func=mdp.stage_gated_feet_contact_reward,
            weight=0.08,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=[
                        "FL_foot_link",
                        "FR_foot_link",
                    ],
                ),
                "force_threshold": 1.0,
                "stage_command_name": "motion",
                "enabled_stage": 2,
            },
        )

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

        self.curriculum.base_velocity_lin = None
        self.curriculum.base_velocity_yaw = None

        self.terminations.anchor_pos = None
        self.terminations.anchor_ori = None
        self.terminations.ee_body_pos = None
        self.terminations.illegal_contact = None


@configclass
class PcbCBeyondMimicFlatV1StandCommandDelayEnvCfg(PcbCBeyondMimicFlatV1StandCommandEnvCfg):
    """pcbC command task, delayed actuator version.

    Use this after non-delay baseline is stable.
    """

    def __post_init__(self):
        super().__post_init__()
        apply_pcbc_delayed_actuators(self)

@configclass
class PcbCBeyondMimicFlatV1StandCommandStage2HoldWarmupEnvCfg(PcbCBeyondMimicFlatV1StandCommandEnvCfg):
    """pcbC stage2 warmup with built-in hold.

    PCBC_COMMAND_PHASE=baseline:
        Use parent baseline branch:
        - pure trajectory imitation
        - no stage command
        - zero velocity command
        - constant-zero motion_stage observation

    PCBC_COMMAND_PHASE=stage2:
        pcba-style stage flow:
        - stage0: trajectory imitation
        - stage1: final-frame hold
        - stage2: very small lateral command warmup

    这一版的目标不是立刻训练大速度，而是：
    1. 从 baseline model_12000 接住上台能力；
    2. 让 stage1 hold 不炸、不顶限位；
    3. 只给极弱 stage2 小 y 速度，让模型慢慢适应 command 分布。
    """

    def __post_init__(self):
        super().__post_init__()

        command_phase = os.getenv("PCBC_COMMAND_PHASE", "stage2").strip().lower()
        if command_phase == "baseline":
            return

        # ------------------------------------------------------------
        # Episode length
        # pcbC motion is about 12.44s.
        # 40s = stage0 trajectory + stage1 hold + optional weak stage2.
        # ------------------------------------------------------------
        self.episode_length_s = 40.0

        self.scene.box.init_state.pos = (0.97, 0.0, 0.4)
        self.scene.box.spawn.size = (1.05, 8.80, 0.8)

        # ------------------------------------------------------------
        # Stage timing
        # 当前日志 stage 过早到 1.5~1.7，所以先推迟 stage2。
        # max_hold_steps_before_force_command > 2000，意味着 40s 内不会强制进 stage2；
        # 只有真的 stable 后，经过 extra_hold_steps_after_stable 才进入 stage2。
        # ------------------------------------------------------------
        self.commands.motion.enable_stage_command = True
        self.commands.motion.hold_anchor_rot_threshold = 0.12
        self.commands.motion.hold_body_pos_threshold = 0.06
        self.commands.motion.hold_joint_pos_threshold = 0.12
        self.commands.motion.hold_base_lin_vel_threshold = 0.12
        self.commands.motion.hold_base_ang_vel_threshold = 0.35

        self.commands.motion.hold_stable_steps = 120
        self.commands.motion.extra_hold_steps_after_stable = 300
        self.commands.motion.max_hold_steps_before_force_command = 2200

        # ------------------------------------------------------------
        # Very small stage2 command
        # 先只保留极小 y，关闭 x/yaw。
        # 修正你当前代码中缺少逗号的语法错误。
        # ------------------------------------------------------------
        self.commands.base_velocity.ranges = mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0),
            lin_vel_y=(-0.05, 0.05),
            ang_vel_z=(0.0, 0.0),
            heading=(-3.14, 3.14),
        )
        self.commands.base_velocity.rel_standing_envs = 0.20
        self.commands.base_velocity.rel_heading_envs = 0.0

        # ------------------------------------------------------------
        # Unified observations
        # 保持 stage2 观测结构统一。
        # stage0/stage1 通过 stage_gated_generated_commands 看到 0 command，
        # stage2 才看到真实小速度 command。
        # ------------------------------------------------------------
        self.observations.policy.base_velocity_command = ObsTerm(
            func=mdp.stage_gated_generated_commands,
            params={
                "command_name": "base_velocity",
                "stage_command_name": "motion",
                "enabled_stage": 2,
            },
        )
        self.observations.critic.base_velocity_command = ObsTerm(
            func=mdp.stage_gated_generated_commands,
            params={
                "command_name": "base_velocity",
                "stage_command_name": "motion",
                "enabled_stage": 2,
            },
        )

        self.observations.policy.motion_stage = ObsTerm(
            func=mdp.motion_stage,
            params={
                "command_name": "motion",
                "normalize": True,
            },
        )
        self.observations.critic.motion_stage = ObsTerm(
            func=mdp.motion_stage,
            params={
                "command_name": "motion",
                "normalize": True,
            },
        )

        # ------------------------------------------------------------
        # Stage-weighted mimic rewards
        #
        # 当前日志：
        # - joint_pos_limits 已经从 -1.x 降到 -0.03~-0.08，说明不能再强压 joint；
        # - error_joint_pos 仍大，说明先稳 hold，再逐步收紧 joint；
        # - body_ori 仍弱，所以 body_ori 保持略高。
        # ------------------------------------------------------------
        self.rewards.motion_global_anchor_pos = RewTerm(
            func=mdp.stage_weighted_motion_global_anchor_position_error_exp,
            weight=1.20,
            params={
                "command_name": "motion",
                "std": 0.30,
                "stage_command_name": "motion",
                "stage0_weight": 1.05,
                "stage1_weight": 0.85,
                "stage2_weight": 0.00,
            },
        )

        self.rewards.motion_global_anchor_ori = RewTerm(
            func=mdp.stage_weighted_motion_global_anchor_orientation_error_exp,
            weight=1.15,
            params={
                "command_name": "motion",
                "std": 0.40,
                "stage_command_name": "motion",
                "stage0_weight": 1.05,
                "stage1_weight": 0.90,
                "stage2_weight": 0.10,
            },
        )

        self.rewards.motion_body_pos = RewTerm(
            func=mdp.stage_weighted_motion_relative_body_position_error_exp,
            weight=0.95,
            params={
                "command_name": "motion",
                "std": 0.30,
                "stage_command_name": "motion",
                "stage0_weight": 1.05,
                "stage1_weight": 1.05,
                "stage2_weight": 0.80,
            },
        )

        self.rewards.motion_body_ori = RewTerm(
            func=mdp.stage_weighted_motion_relative_body_orientation_error_exp,
            weight=1.05,
            params={
                "command_name": "motion",
                "std": 0.45,
                "stage_command_name": "motion",
                "stage0_weight": 1.05,
                "stage1_weight": 1.10,
                "stage2_weight": 0.80,
            },
        )

        # pcbC 末帧参考速度可能不是 0；
        # stage1 速度模仿必须弱，否则会在接触高台时继续推。
        self.rewards.motion_body_lin_vel = RewTerm(
            func=mdp.stage_weighted_motion_global_body_linear_velocity_error_exp,
            weight=0.35,
            params={
                "command_name": "motion",
                "std": 1.0,
                "stage_command_name": "motion",
                "stage0_weight": 1.00,
                "stage1_weight": 0.02,
                "stage2_weight": 0.00,
            },
        )

        self.rewards.motion_body_ang_vel = RewTerm(
            func=mdp.stage_weighted_motion_global_body_angular_velocity_error_exp,
            weight=0.30,
            params={
                "command_name": "motion",
                "std": 3.14,
                "stage_command_name": "motion",
                "stage0_weight": 1.00,
                "stage1_weight": 0.02,
                "stage2_weight": 0.00,
            },
        )

        self.rewards.motion_joint_pos = RewTerm(
            func=mdp.stage_weighted_motion_joint_position_error_exp,
            weight=0.38,
            params={
                "command_name": "motion",
                "std": 0.65,
                "asset_cfg": SceneEntityCfg("robot", joint_names=self.leg_joint_names),
                "stage_command_name": "motion",
                "stage0_weight": 1.05,
                "stage1_weight": 0.90,
                "stage2_weight": 0.55,
            },
        )

        self.rewards.motion_joint_vel = RewTerm(
            func=mdp.stage_weighted_motion_joint_velocity_error_exp,
            weight=0.15,
            params={
                "command_name": "motion",
                "std": 1.0,
                "asset_cfg": SceneEntityCfg("robot", joint_names=self.leg_joint_names),
                "stage_command_name": "motion",
                "stage0_weight": 1.00,
                "stage1_weight": 0.03,
                "stage2_weight": 0.15,
            },
        )

        self.rewards.motion_wheel_joint_vel = RewTerm(
            func=mdp.stage_weighted_motion_joint_velocity_error_exp,
            weight=0.03,
            params={
                "command_name": "motion",
                "std": 1.0,
                "asset_cfg": SceneEntityCfg("robot", joint_names=self.wheel_joint_names),
                "stage_command_name": "motion",
                "stage0_weight": 1.00,
                "stage1_weight": 0.03,
                "stage2_weight": 0.00,
            },
        )

        # ------------------------------------------------------------
        # Stage1 hold stability rewards
        # 当前保持，不再加大，避免为了接触奖励强压前脚。
        # ------------------------------------------------------------
        self.rewards.front_feet_contact_hold = RewTerm(
            func=mdp.stage_gated_feet_contact_reward,
            weight=0.08,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=[
                        "FL_foot_link",
                        "FR_foot_link",
                    ],
                ),
                "force_threshold": 1.0,
                "stage_command_name": "motion",
                "enabled_stage": 1,
            },
        )

        self.rewards.base_pitch_excess_hold = RewTerm(
            func=mdp.stage_gated_base_pitch_excess_l2_penalty,
            weight=-0.10,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "pitch_limit": 0.35,
                "stage_command_name": "motion",
                "enabled_stage": 1,
            },
        )

        self.rewards.base_tilt_penalty_hold = RewTerm(
            func=mdp.stage_gated_base_tilt_l2_penalty,
            weight=-0.08,
            params={
                "command_name": "motion",
                "enabled_stage": 1,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        self.rewards.multi_foot_air_penalty_hold = RewTerm(
            func=mdp.stage_gated_multi_foot_air_penalty,
            weight=-0.04,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot_link"),
                "max_air_feet": 2,
                "command_name": "motion",
                "enabled_stage": 1,
            },
        )

        # ------------------------------------------------------------
        # Stage2 velocity warmup
        # 第一版先极弱速度，不训练 gait pattern。
        # 等 hold 稳住，再逐步打开。
        # ------------------------------------------------------------
        self.rewards.track_lin_vel_xy = RewTerm(
            func=mdp.stage_gated_track_lin_vel_axis_projected_body_exp,
            weight=0.00,
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
            weight=0.15,
            params={
                "command_name": "base_velocity",
                "axis": 1,
                "std": 0.14,
                "stage_command_name": "motion",
                "enabled_stage": 2,
            },
        )

        self.rewards.track_lin_vel_y_fine = RewTerm(
            func=mdp.stage_gated_track_lin_vel_axis_projected_body_exp,
            weight=0.00,
            params={
                "command_name": "base_velocity",
                "axis": 1,
                "std": 0.08,
                "stage_command_name": "motion",
                "enabled_stage": 2,
            },
        )

        self.rewards.track_yaw_rate = RewTerm(
            func=mdp.stage_gated_track_ang_vel_z_projected_exp,
            weight=0.00,
            params={
                "command_name": "base_velocity",
                "std": 0.16,
                "stage_command_name": "motion",
                "enabled_stage": 2,
            },
        )

        # 第一版 warmup 先关闭 gait shaping。
        # 否则还没稳定 hold，就开始诱导腿部摆动。
        self.rewards.y_cmd_leg_lift = None
        self.rewards.rear_foot_air_time = None
        self.rewards.y_cmd_leg_motion = None
        self.rewards.y_cmd_diagonal_leg_motion = None
        self.rewards.y_cmd_diagonal_motion_balance = None

        # ------------------------------------------------------------
        # Stage2 stability constraints
        # 这些只在 stage2 生效，保留但不要太强。
        # ------------------------------------------------------------
        self.rewards.hip_abduction_penalty = RewTerm(
            func=mdp.stage_gated_joint_deviation_l1_penalty,
            weight=-0.07,
            params={
                "joint_names": [
                    "FL_hip_joint",
                    "FR_hip_joint",
                    "RL_hip_joint",
                    "RR_hip_joint",
                ],
                "command_name": "motion",
                "enabled_stage": 2,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        self.rewards.front_feet_contact = RewTerm(
            func=mdp.stage_gated_feet_contact_reward,
            weight=0.06,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=[
                        "FL_foot_link",
                        "FR_foot_link",
                    ],
                ),
                "force_threshold": 1.0,
                "stage_command_name": "motion",
                "enabled_stage": 2,
            },
        )

        self.rewards.base_pitch_excess = RewTerm(
            func=mdp.stage_gated_base_pitch_excess_l2_penalty,
            weight=-0.08,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "pitch_limit": 0.35,
                "stage_command_name": "motion",
                "enabled_stage": 2,
            },
        )

        self.rewards.base_tilt_penalty = RewTerm(
            func=mdp.stage_gated_base_tilt_l2_penalty,
            weight=-0.07,
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
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot_link"),
                "max_air_feet": 2,
                "command_name": "motion",
                "enabled_stage": 2,
            },
        )

        self.rewards.stage2_anchor_local_x_drift = RewTerm(
            func=mdp.stage_gated_anchor_local_axis_drift_l2_penalty,
            weight=-0.025,
            params={
                "command_name": "base_velocity",
                "motion_command_name": "motion",
                "asset_cfg": SceneEntityCfg("robot"),
                "axis": 0,
                "deadband": 0.10,
                "command_axis": 0,
                "command_deadzone": 0.06,
                "max_penalty": 1.0,
                "enabled_stage": 2,
            },
        )

        self.curriculum.base_velocity_lin = None
        self.curriculum.base_velocity_yaw = None

        self.terminations.anchor_pos = None
        self.terminations.anchor_ori = None
        self.terminations.ee_body_pos = None
        self.terminations.illegal_contact = None


@configclass
class PcbCBeyondMimicFlatV1StandCommandHoldPolishDelayEnvCfg(
    PcbCBeyondMimicFlatV1StandCommandStage2HoldWarmupEnvCfg
):
    """Delayed actuator version of stage2-hold-warmup.

    这个名字保留是为了兼容你之前的注册习惯。
    实际已经不是单独 hold-polish，而是 stage2-hold-warmup + delayed actuator。
    """

    def __post_init__(self):
        super().__post_init__()
        apply_pcbc_delayed_actuators(self)
