# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import os

from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import robot_lab.tasks.manager_based.beyondmimic.mdp as mdp
from robot_lab.assets.pcbC import pcbC_CFG
from robot_lab.tasks.manager_based.beyondmimic.tracking_env_cfg_go2w import BeyondMimicEnvCfg


_PCBC_MOTION_FILES = {
    "stage_up_hold2p5s": f"{os.path.dirname(__file__)}/../go2w/motion/pcbv2_x6u_60hz_stage_up.npz",
    "train23_hold2s": f"{os.path.dirname(__file__)}/../go2w/motion/pcbv2_x6u_50hz_hold2s.npz",
    "train_5.31": f"{os.path.dirname(__file__)}/../go2w/motion/pcbv2_x6u_60hz_stage_up_531.npz",
}
PCBC_MOTION_VARIANT = os.getenv("PCBC_MOTION_VARIANT", "train23_hold2s")
PCBC_MOTION_FILE = _PCBC_MOTION_FILES.get(PCBC_MOTION_VARIANT, _PCBC_MOTION_FILES["train23_hold2s"])
# The frozen baseline at PCBC_BASELINE_POLICY_PATH was trained with train_5.31.
# Follow an explicit PCBC_MOTION_VARIANT override, otherwise preserve that exact motion input.
PCBC_RESIDUAL_MOTION_VARIANT = os.getenv(
    "PCBC_RESIDUAL_MOTION_VARIANT",
    os.getenv("PCBC_MOTION_VARIANT", "train_5.31"),
)
PCBC_RESIDUAL_MOTION_FILE = _PCBC_MOTION_FILES.get(
    PCBC_RESIDUAL_MOTION_VARIANT,
    _PCBC_MOTION_FILES["train_5.31"],
)
PCBC_BASELINE_POLICY_PATH = os.getenv(
    "PCBC_BASELINE_POLICY_PATH",
    "logs/rsl_rl/pcbc_v1_command_stage/2026-06-06_16-26-16/exported/policy_model_8000_frozen.pt",
)


def pcbc_default_joint_pos(motion_variant: str | None = None) -> dict[str, float]:
    """pcbC default standing posture.

    NPZ joint order is type-grouped:
    4 hips, 4 thighs, 4 calves, 4 wheels.
    """
    motion_variant = PCBC_MOTION_VARIANT if motion_variant is None else motion_variant

    if motion_variant == "stage_up_hold2p5s":
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

    if motion_variant == "train_5.31":
        return {
            "FL_hip_joint": -0.00634721,
            "FL_thigh_joint": 0.726606,
            "FL_calf_joint": -1.30170,
            "FR_hip_joint": 0.00638027,
            "FR_thigh_joint": 0.724651,
            "FR_calf_joint": -1.29884,
            "RL_hip_joint": -0.00635801,
            "RL_thigh_joint": -0.743733,
            "RL_calf_joint": 1.29775,
            "RR_hip_joint": 0.00634459,
            "RR_thigh_joint": -0.747285,
            "RR_calf_joint": 1.30355,
            "FL_foot_joint": 0.000125342,
            "FR_foot_joint": 0.0000198542,
            "RL_foot_joint": -0.0000104716,
            "RR_foot_joint": -0.000134248,
        }

    return {
        "FL_hip_joint": -0.00634721,
        "FL_thigh_joint": 0.726606,
        "FL_calf_joint": -1.30170,
        "FR_hip_joint": 0.00638027,
        "FR_thigh_joint": 0.724651,
        "FR_calf_joint": -1.29884,
        "RL_hip_joint": 0.00634459,
        "RL_thigh_joint": -0.747285,
        "RL_calf_joint": 1.30355,
        "RR_hip_joint": -0.00635801,
        "RR_thigh_joint": -0.743733,
        "RR_calf_joint": 1.29775,
        "FL_foot_joint": 0.000125342,
        "FR_foot_joint": 0.0000198542,
        "RL_foot_joint": -0.000134248,
        "RR_foot_joint": -0.0000104716,
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

#baseline 
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
class PcbCResidualMoveEnvCfg(PcbCBeyondMimicFlatV1StandEnvCfg):
    """Baseline-preserving residual controller for on-platform motion.

    This task does not continue the old command-stage policy training. Instead:
    - the exported baseline policy is frozen inside the action term;
    - residual output is forced ineffective until the automatic completion gate opens;
    - after ready, the residual policy learns full 16-D corrections.
    """

    def __post_init__(self):
        super().__post_init__()

        self.episode_length_s = 80.0
        self.commands.motion.motion_file = PCBC_RESIDUAL_MOTION_FILE
        self.scene.robot.init_state.joint_pos = pcbc_default_joint_pos(PCBC_RESIDUAL_MOTION_VARIANT)
        self.scene.box.init_state.pos = (0.97, 0.0, 0.4)
        self.scene.box.spawn.size = (1.05, 8.80, 0.8)

        # Keep the baseline trajectory command alive after the final frame so the
        # residual can train on top of the final on-platform posture.
        self.commands.motion.enable_completion_gate = True
        self.commands.motion.hold_joint_names = self.leg_joint_names
        self.commands.motion.hold_anchor_rot_threshold = 0.30
        self.commands.motion.hold_body_pos_threshold = 0.15
        # The verified baseline settles in a valid posture that differs from the
        # reference final-frame joint angles. Body pose and low velocity decide ready.
        self.commands.motion.hold_joint_pos_threshold = None
        self.commands.motion.hold_base_lin_vel_threshold = 0.35
        self.commands.motion.hold_base_ang_vel_threshold = 0.90
        # Open the residual gate aggressively so most samples train the yaw skill.
        self.commands.motion.hold_stable_steps = 3
        self.commands.motion.extra_hold_steps_after_stable = 0
        self.commands.motion.max_hold_steps_before_force_command = 50
        # Concentrate resets near the fixed final posture while retaining a small
        # fraction of the full boarding/transition trajectory.
        self.commands.motion.late_sampling_ratio = 0.95
        self.commands.motion.late_sampling_start_fraction = 0.97

        self.commands.base_velocity = mdp.DecoupledPlanarVelocityCommandCfg(
            asset_name="robot",
            motion_command_name="motion",
            resampling_time_range=(3.0, 5.0),
            rel_standing_envs=0.15,
            rel_heading_envs=0.0,
            # Effective mix after the 15% standing mask:
            # about 55% yaw-only, 30% y-only, and 15% standing.
            rel_yaw_envs=0.6471,
            rel_combined_envs=0.0,
            min_abs_lin_vel_y=0.10,
            min_abs_ang_vel_z=0.22,
            zero_command_warmup_steps=0,
            # Sample standing, y, or yaw immediately when the residual gate opens.
            post_ready_zero_command_steps=0,
            heading_command=False,
            debug_vis=True,
            ranges=mdp.DecoupledPlanarVelocityCommandCfg.Ranges(
                lin_vel_x=(0.0, 0.0),
                lin_vel_y=(-0.30, 0.30),
                ang_vel_z=(-0.45, 0.45),
                heading=None,
            ),
        )

        self.actions.joint_pos = None
        self.actions.joint_vel = None
        self.actions.residual_joint = mdp.BaselineResidualJointActionCfg(
            asset_name="robot",
            baseline_policy_path=PCBC_BASELINE_POLICY_PATH,
            motion_command_name="motion",
            leg_joint_names=self.leg_joint_names,
            wheel_joint_names=self.wheel_joint_names,
            leg_residual_scale=1.00,
            wheel_residual_scale=0.45,
            final_action_clip=100.0,
        )
        # The residual action reads asset.data.default_joint_pos directly and
        # therefore has no cached JointPositionAction offset to synchronize.
        self.events.randomize_joint_default_pos.params["action_name"] = None

        self.observations.policy.base_velocity_command = ObsTerm(
            func=mdp.ready_gated_generated_commands,
            params={
                "command_name": "base_velocity",
                "motion_command_name": "motion",
            },
        )
        self.observations.critic.base_velocity_command = ObsTerm(
            func=mdp.ready_gated_generated_commands,
            params={
                "command_name": "base_velocity",
                "motion_command_name": "motion",
            },
        )

        # The frozen baseline owns the boarding phase. Residual rewards only become
        # meaningful after ready, so they cannot rewrite the boarding behavior.
        self.rewards.joint_acc_l2.func = mdp.ready_gated_joint_acc_l2
        self.rewards.joint_acc_l2.params = {"motion_command_name": "motion"}
        self.rewards.joint_torques_l2.func = mdp.ready_gated_joint_torques_l2
        self.rewards.joint_torques_l2.params = {"motion_command_name": "motion"}
        self.rewards.action_rate_l2 = None
        self.rewards.joint_pos_limits.func = mdp.ready_gated_joint_pos_limits
        self.rewards.joint_pos_limits.params = {
            "motion_command_name": "motion",
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
        }
        # The inherited humanoid-style regex matches pcbC's normal foot contacts,
        # so it conflicts with both boarding and stepping.
        self.rewards.undesired_contacts = None

        self.rewards.motion_global_anchor_pos = None
        self.rewards.motion_global_anchor_ori = None
        # Track one fixed final-frame relative posture. Global y translation is
        # free, while the hitch body layout remains a consistent target.
        self.rewards.motion_body_pos = RewTerm(
            func=mdp.ready_gated_motion_relative_body_position_error_exp,
            weight=1.00,
            params={
                "command_name": "motion",
                "std": 0.25,
                "motion_command_name": "motion",
            },
        )
        self.rewards.motion_body_ori = RewTerm(
            func=mdp.ready_gated_motion_relative_body_orientation_error_exp,
            weight=0.90,
            params={
                "command_name": "motion",
                "std": 0.36,
                "motion_command_name": "motion",
            },
        )
        self.rewards.motion_body_lin_vel = None
        self.rewards.motion_body_ang_vel = None
        # A weak, wide final-joint target prevents persistent malformed leg
        # postures while still allowing the transient deviations needed to step.
        self.rewards.motion_joint_pos = RewTerm(
            func=mdp.ready_gated_motion_joint_position_error_exp,
            weight=0.16,
            params={
                "command_name": "motion",
                "std": 0.65,
                "asset_cfg": SceneEntityCfg("robot", joint_names=self.leg_joint_names),
                "motion_command_name": "motion",
            },
        )
        self.rewards.motion_joint_vel = None
        self.rewards.motion_wheel_joint_vel = None

        self.rewards.track_lin_vel_x = RewTerm(
            func=mdp.ready_gated_track_lin_vel_axis_projected_body_exp,
            weight=0.20,
            params={
                "command_name": "base_velocity",
                "axis": 0,
                "std": 0.10,
            },
        )
        self.rewards.track_lin_vel_y = RewTerm(
            func=mdp.ready_gated_track_lin_vel_axis_projected_body_exp,
            weight=2.00,
            params={
                "command_name": "base_velocity",
                "axis": 1,
                "std": 0.10,
                "motion_command_name": "motion",
            },
        )
        self.rewards.track_yaw_rate = RewTerm(
            func=mdp.ready_gated_track_ang_vel_z_projected_exp,
            weight=2.20,
            params={
                "command_name": "base_velocity",
                "std": 0.15,
            },
        )
        self.rewards.track_lin_vel_y_directional = RewTerm(
            func=mdp.ready_gated_normalized_lin_vel_axis_tracking,
            weight=1.60,
            params={
                "command_name": "base_velocity",
                "axis": 1,
                "command_threshold": 0.07,
                "motion_command_name": "motion",
            },
        )
        # The exponential yaw reward still gives some credit for standing under
        # a non-zero command. This clean directional term makes standing score
        # exactly zero and rewards only progress in the commanded yaw direction.
        self.rewards.track_yaw_rate_directional = RewTerm(
            func=mdp.ready_gated_normalized_yaw_rate_tracking,
            weight=2.00,
            params={
                "command_name": "base_velocity",
                "command_threshold": 0.07,
                "motion_command_name": "motion",
            },
        )
        # Disable inherited gait-shaping patches. Fixed posture and yaw-rate
        # tracking define the task without prescribing a particular stepping style.
        self.rewards.y_cmd_leg_motion = None
        self.rewards.y_cmd_diagonal_leg_motion = None
        self.rewards.y_cmd_diagonal_motion_balance = None
        self.rewards.y_cmd_diagonal_feet_air_time = None
        self.rewards.y_cmd_foot_up_velocity = None
        self.rewards.y_cmd_airborne_foot_clearance = None
        self.rewards.front_feet_contact = None
        # Relative-pose constraints preserve the hitch pitch/roll while still
        # allowing commanded global yaw.
        self.rewards.base_tilt_penalty = None
        self.rewards.base_pitch_excess = None
        self.rewards.hip_abduction_penalty = None
        self.rewards.multi_foot_air_penalty = None
        # The verified hitch posture intentionally leaves the base and rear legs
        # outside the platform. Platform-center margin penalties would pull the
        # robot forward and destroy that valid landing pose.
        self.rewards.base_platform_x_margin = None
        self.rewards.feet_platform_x_margin = None
        self.rewards.residual_pre_ready_l2 = RewTerm(
            func=mdp.residual_action_l2_penalty,
            weight=-0.05,
            params={
                "action_name": "residual_joint",
                "motion_command_name": "motion",
                "only_ready": False,
            },
        )
        self.rewards.residual_ready_l2 = RewTerm(
            func=mdp.residual_action_l2_penalty,
            weight=-0.0005,
            params={
                "action_name": "residual_joint",
                "motion_command_name": "motion",
                "only_ready": True,
            },
        )
        self.rewards.residual_rate_l2 = RewTerm(
            func=mdp.residual_action_rate_l2_penalty,
            weight=-0.006,
            params={"action_name": "residual_joint", "motion_command_name": "motion"},
        )

        self.curriculum.base_velocity_lin = None
        self.curriculum.base_velocity_yaw = None

        # Keep the baseline fall/failure checks. They constrain only height and
        # roll/pitch, so they remain valid while allowing lateral motion and yaw.
        self.terminations.ready_heading_yaw_drift = DoneTerm(
            func=mdp.ready_gated_heading_yaw_drift_out_of_range,
            params={
                "command_name": "base_velocity",
                "yaw_command_deadzone": 0.02,
                "yaw_error_threshold": 0.90,
                "motion_command_name": "motion",
            },
        )
        self.terminations.ready_roll_drift = DoneTerm(
            func=mdp.ready_gated_roll_error_out_of_range,
            params={
                "roll_error_threshold": 0.70,
                "max_duration_s": 1.20,
                "motion_command_name": "motion",
            },
        )
        self.terminations.ready_all_feet_ground_contact = DoneTerm(
            func=mdp.ready_gated_low_foot_contact_out_of_range,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot_link"),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot_link"),
                "height_threshold": 0.45,
                "force_threshold": 1.0,
                "max_duration_s": 0.15,
                "require_all": True,
                "motion_command_name": "motion",
            },
        )
        self.terminations.anchor_pos = None
        self.terminations.anchor_ori = None
        self.terminations.ee_body_pos = None


@configclass
class Stage2CommandEnvCfgX1(PcbCBeyondMimicFlatV1StandCommandEnvCfg):
    """pcbC low-speed 3-axis command warmup.

    用途：
    - 从已经训好的 baseline checkpoint 接。
    - baseline 使用 train23_hold2s 轨迹，已经学会上台 + 轨迹自带 2.5s hold。
    - 本类进入 stage2 后，不再只训 y，而是低速接入 x / y / yaw 三轴 command。

    PCBC_COMMAND_PHASE=baseline:
        直接沿用父类 baseline 分支，只做纯模仿。

    PCBC_COMMAND_PHASE=stage2:
        stage0: 继续模仿上台轨迹
        stage1: 末端稳定确认
        stage2: 低速三轴 command，x/y/yaw 都打开但幅度很小
    """

    def __post_init__(self):
        super().__post_init__()

        command_phase = os.getenv("PCBC_COMMAND_PHASE", "stage2").strip().lower()
        if command_phase == "baseline":
            return

        # ------------------------------------------------------------
        # Episode
        # train23_hold2s 轨迹自身已经包含末端 hold。
        # stage1 不需要再拖太久，否则 stage2 样本太少。
        # ------------------------------------------------------------
        self.episode_length_s = 40.0

        self.scene.box.init_state.pos = (0.97, 0.0, 0.4)
        self.scene.box.spawn.size = (1.05, 8.80, 0.8)

        # ------------------------------------------------------------
        # Stage timing
        #
        # 50Hz 下：
        # hold_stable_steps=80 约 1.6s 稳定确认
        # extra_hold_steps_after_stable=120 约 2.4s 额外 hold
        # max_hold_steps_before_force_command=900 约 18s
        #
        # 相比 1600/1800/2200，这版会更早给 stage2 command 样本；
        # 但仍然先经过一段 hold，避免直接破坏上台稳定性。
        # ------------------------------------------------------------
        self.commands.motion.enable_stage_command = True
        self.commands.motion.hold_anchor_rot_threshold = 0.12
        self.commands.motion.hold_body_pos_threshold = 0.06
        self.commands.motion.hold_joint_pos_threshold = 0.12
        self.commands.motion.hold_base_lin_vel_threshold = 0.12
        self.commands.motion.hold_base_ang_vel_threshold = 0.35

        self.commands.motion.hold_stable_steps = 80
        self.commands.motion.extra_hold_steps_after_stable = 120
        self.commands.motion.max_hold_steps_before_force_command = 900

        # ------------------------------------------------------------
        # Low-speed 3-axis command
        #
        # 不再只训 y。
        # 一开始就让 policy 看到 x / y / yaw 三轴 command，
        # 但范围很小，避免破坏 baseline 的上台和 hold。
        # ------------------------------------------------------------
        self.commands.base_velocity.ranges = mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.04, 0.04),
            lin_vel_y=(-0.06, 0.06),
            ang_vel_z=(-0.05, 0.05),
            heading=(-3.14, 3.14),
        )
        self.commands.base_velocity.rel_standing_envs = 0.30
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.heading_command = False

        # ------------------------------------------------------------
        # Unified observations
        #
        # stage0/stage1: stage_gated_generated_commands 输出 0 command
        # stage2: 输出真实低速 command
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
                "stage2_weight": 0.12,
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
                "stage2_weight": 0.75,
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
                "stage2_weight": 0.78,
            },
        )

        # stage1 不强追末端速度，避免前脚接触高台后继续推。
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

        self.rewards.front_feet_contact_hold = RewTerm(
            func=mdp.stage_gated_feet_contact_reward,
            weight=0,
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

        self.rewards.track_lin_vel_xy = RewTerm(
            func=mdp.stage_gated_track_lin_vel_axis_projected_body_exp,
            weight=0.08,
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
            weight=0.12,
            params={
                "command_name": "base_velocity",
                "axis": 1,
                "std": 0.14,
                "stage_command_name": "motion",
                "enabled_stage": 2,
            },
        )

        # 第一版先不要 fine y，避免过早强压横向速度。
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
            weight=0.06,
            params={
                "command_name": "base_velocity",
                "std": 0.16,
                "stage_command_name": "motion",
                "enabled_stage": 2,
            },
        )


        self.rewards.y_cmd_leg_lift = None
        self.rewards.rear_foot_air_time = None
        self.rewards.y_cmd_leg_motion = None
        self.rewards.y_cmd_diagonal_leg_motion = None
        self.rewards.y_cmd_diagonal_motion_balance = None

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
            weight=0,
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

        # x 方向漂移约束：低速前进/后退时不能过度贴近或远离高台边缘。
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
