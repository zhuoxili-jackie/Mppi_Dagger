# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
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

_PCBC_REPO_ROOT = Path(__file__).resolve().parents[8]
PCBC_LATERAL_MOTION_DIR = _PCBC_REPO_ROOT / "pcbB样侧向移动数据608"
PCBC_LATERAL_MOTIONS = [
    (PCBC_LATERAL_MOTION_DIR / "trajectory_trotting_acc_f02.npz", -0.20),
    (PCBC_LATERAL_MOTION_DIR / "trajectory_trotting_acc_f015.npz", -0.15),
    (PCBC_LATERAL_MOTION_DIR / "trajectory_trotting_acc_f01.npz", -0.10),
    (PCBC_LATERAL_MOTION_DIR / "trajectory_trotting_acc_f005.npz", -0.05),
    (PCBC_LATERAL_MOTION_DIR / "trajectory_trotting_acc_005.npz", 0.05),
    (PCBC_LATERAL_MOTION_DIR / "trajectory_trotting_acc_01.npz", 0.10),
    (PCBC_LATERAL_MOTION_DIR / "trajectory_trotting_acc_015.npz", 0.15),
    (PCBC_LATERAL_MOTION_DIR / "trajectory_trotting_acc_02.npz", 0.20),
]


def pcbc_default_joint_pos() -> dict[str, float]:
    if PCBC_MOTION_VARIANT == "stage_up_hold2p5s":
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

    if PCBC_MOTION_VARIANT == "train_5.31":
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


@configclass
class PcbCBeyondMimicFlatV1Stage2LateralGuidedEnvCfg(BeyondMimicEnvCfg):
    """pcbC boarding imitation baseline plus stage2 lateral-reference walking."""

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
    wheel_joint_names = ["FL_foot_joint", "FR_foot_joint", "RL_foot_joint", "RR_foot_joint"]
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
        "FL_foot_link",
        "FR_foot_link",
        "RL_foot_link",
        "RR_foot_link",
    ]

    def __post_init__(self):
        super().__post_init__()
        command_phase = os.getenv("PCBC_COMMAND_PHASE", "stage2").strip().lower()

        self._configure_robot_and_scene()
        self._configure_actions()
        self._configure_observations_common()
        self._configure_randomization()
        self._configure_baseline_mimic_rewards()
        self._configure_motion_command_common()

        if command_phase == "baseline":
            self._configure_baseline_phase()
            return

        self._configure_stage2_lateral_guided_phase()

    def _configure_robot_and_scene(self):
        self.scene.robot = pcbC_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.45)
        self.scene.robot.init_state.rot = (1.0, 0.0, 0.0, 0.0)
        self.scene.robot.init_state.joint_pos = pcbc_default_joint_pos()
        self.scene.box.init_state.pos = (0.97, 0.0, 0.4)
        self.scene.box.spawn.size = (1.05, 8.80, 0.8)

    def _configure_actions(self):
        self.actions.joint_pos.scale = {".*_hip_joint": 0.125, "^(?!.*_hip_joint).*": 0.25}
        self.actions.joint_vel.scale = 5.0
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_vel.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_pos.joint_names = self.leg_joint_names
        self.actions.joint_vel.joint_names = self.wheel_joint_names

    def _configure_observations_common(self):
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

    def _configure_randomization(self):
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
                "com_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (-0.02, 0.02)},
            }
        )
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

    def _configure_baseline_mimic_rewards(self):
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
        self.rewards.action_rate_l2.params = {"clip": 1.0, "max_value": 64.0}
        self.rewards.action_rate_l2.weight = -1.0e-2
        self.terminations.illegal_contact = None
        self.episode_length_s = 18

    def _configure_motion_command_common(self):
        self.commands.motion.motion_file = PCBC_MOTION_FILE
        self.commands.motion.anchor_body_name = self.base_link_name
        self.commands.motion.body_names = self.body_names
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
        self.commands.motion.enable_stage_command = True
        self.commands.motion.hold_anchor_rot_threshold = 0.12
        self.commands.motion.hold_body_pos_threshold = 0.06
        self.commands.motion.hold_joint_pos_threshold = 0.12
        self.commands.motion.hold_base_lin_vel_threshold = 0.12
        self.commands.motion.hold_base_ang_vel_threshold = 0.35
        self.commands.motion.hold_stable_steps = 80
        self.commands.motion.extra_hold_steps_after_stable = 120
        self.commands.motion.max_hold_steps_before_force_command = 1600

    def _configure_baseline_phase(self):
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
        self.observations.policy.motion_stage = ObsTerm(func=mdp.constant_zero_scalar)
        self.observations.critic.motion_stage = ObsTerm(func=mdp.constant_zero_scalar)
        self.terminations.anchor_pos.params["threshold"] = 0.40

    def _configure_stage2_lateral_guided_phase(self):
        self.episode_length_s = 40.0
        self._configure_stage_weighted_boarding_rewards()
        self._configure_stage2_safety_rewards()
        self._configure_stage2_lateral_reference()
        self.curriculum.base_velocity_lin = None
        self.curriculum.base_velocity_yaw = None
        self.terminations.anchor_pos = None
        self.terminations.anchor_ori = None
        self.terminations.ee_body_pos = None
        self.terminations.illegal_contact = None

    def _configure_stage_weighted_boarding_rewards(self):
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
                "stage2_weight": 0.00,
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
                "stage2_weight": 0.15,
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
                "stage2_weight": 0.30,
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
                "stage2_weight": 0.45,
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
                "stage2_weight": 0.15,
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

    def _configure_stage2_safety_rewards(self):
        self.rewards.stage2_anchor_local_x_drift = RewTerm(
            func=mdp.stage_gated_anchor_local_axis_drift_l2_penalty,
            weight=-0.060,
            params={
                "command_name": "stage2_lateral",
                "motion_command_name": "motion",
                "asset_cfg": SceneEntityCfg("robot"),
                "axis": 0,
                "deadband": 0.08,
                "command_axis": 0,
                "command_deadzone": 0.05,
                "max_penalty": 1.0,
                "enabled_stage": 2,
            },
        )
        self.rewards.stage2_base_box_x_clearance_penalty = RewTerm(
            func=mdp.stage_gated_base_box_x_clearance_penalty,
            weight=-0.040,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=self.base_link_name),
                "box_name": "box",
                "box_half_x": 0.525,
                "min_clearance": 0.18,
                "max_penalty": 1.0,
                "stage_command_name": "motion",
                "enabled_stage": 2,
            },
        )
        self.rewards.stage2_front_feet_x_clearance_penalty = RewTerm(
            func=mdp.stage_gated_feet_lateral_edge_penalty,
            weight=-0.100,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=["FL_foot_link", "FR_foot_link"]),
                "edge_abs": 0.525,
                "soft_margin": 0.08,
                "axis": 0,
                "box_name": "box",
                "stage_command_name": "motion",
                "enabled_stage": 2,
            },
        )
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
            params={"command_name": "motion", "enabled_stage": 2, "asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.multi_foot_air_penalty = RewTerm(
            func=mdp.stage_gated_multi_foot_air_penalty,
            weight=-0.06,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot_link"),
                "max_air_feet": 2,
                "command_name": "motion",
                "enabled_stage": 2,
            },
        )
        self.rewards.front_feet_lateral_separation_penalty = RewTerm(
            func=mdp.stage_gated_front_feet_lateral_separation_penalty,
            weight=-0.35,
            params={
                "command_name": "stage2_lateral",
                "stage_command_name": "motion",
                "enabled_stage": 2,
                "asset_cfg": SceneEntityCfg("robot"),
                "left_body_name": "FL_foot_link",
                "right_body_name": "FR_foot_link",
                "min_separation": 0.16,
                "command_axis": 1,
                "command_deadband": 0.02,
            },
        )
        self.rewards.front_feet_contact = RewTerm(
            func=mdp.stage_gated_feet_contact_reward,
            weight=0.04,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["FL_foot_link", "FR_foot_link"]),
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

    def _configure_stage2_lateral_reference(self):
        lateral_files = [str(path) for path, _ in PCBC_LATERAL_MOTIONS]
        lateral_velocities = [velocity for _, velocity in PCBC_LATERAL_MOTIONS]

        self.commands.stage2_lateral = mdp.StageGatedLateralReferenceMotionCommandCfg(
            asset_name="robot",
            motion_file=lateral_files[0],
            motion_files=lateral_files,
            lateral_velocities=lateral_velocities,
            standing_probability=0.25,
            target_velocity_range=(-0.10, 0.10),
            min_abs_target_velocity=0.02,
            reset_at_first_frame=True,
            stage_command_name="motion",
            enabled_stage=2,
            anchor_body_name=self.base_link_name,
            body_names=self.body_names,
            resampling_time_range=(1.0e9, 1.0e9),
            debug_vis=False,
            pose_range={"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
            velocity_range={"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
            joint_position_range=(0.0, 0.0),
        )
        self.commands.motion.hold_stable_steps = 60
        self.commands.motion.extra_hold_steps_after_stable = 40
        self.commands.motion.max_hold_steps_before_force_command = 800
        self.commands.motion.hold_joint_pos_threshold = 0.12
        self.commands.base_velocity.ranges = mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0), lin_vel_y=(-0.10, 0.10), ang_vel_z=(0.0, 0.0), heading=(-3.14, 3.14)
        )
        if hasattr(self.commands.base_velocity, "rel_standing_envs"):
            self.commands.base_velocity.rel_standing_envs = 0.0
        if hasattr(self.commands.base_velocity, "rel_heading_envs"):
            self.commands.base_velocity.rel_heading_envs = 0.0

        self.observations.policy.base_velocity_command = ObsTerm(
            func=mdp.stage_gated_lateral_velocity_command,
            params={"command_name": "stage2_lateral", "stage_command_name": "motion", "enabled_stage": 2},
        )
        self.observations.critic.base_velocity_command = ObsTerm(
            func=mdp.stage_gated_lateral_velocity_command,
            params={"command_name": "stage2_lateral", "stage_command_name": "motion", "enabled_stage": 2},
        )
        self.observations.policy.motion_stage = ObsTerm(
            func=mdp.motion_stage,
            params={"command_name": "motion", "normalize": True},
        )
        self.observations.critic.motion_stage = ObsTerm(
            func=mdp.motion_stage,
            params={"command_name": "motion", "normalize": True},
        )

        self.rewards.stage2_lateral_anchor_ori = RewTerm(
            func=mdp.stage_gated_reference_anchor_orientation_error_l2,
            weight=-0.35,
            params={"command_name": "stage2_lateral", "stage_command_name": "motion", "enabled_stage": 2},
        )
        self.rewards.stage2_lateral_joint_pos = RewTerm(
            func=mdp.stage_weighted_motion_joint_position_error_exp,
            weight=0.20,
            params={
                "command_name": "stage2_lateral",
                "std": 0.55,
                "asset_cfg": SceneEntityCfg("robot", joint_names=self.leg_joint_names),
                "stage_command_name": "motion",
                "stage0_weight": 0.00,
                "stage1_weight": 0.00,
                "stage2_weight": 1.00,
            },
        )
        self.rewards.stage2_lateral_joint_vel = RewTerm(
            func=mdp.stage_weighted_motion_joint_velocity_error_exp,
            weight=0.05,
            params={
                "command_name": "stage2_lateral",
                "std": 1.50,
                "asset_cfg": SceneEntityCfg("robot", joint_names=self.leg_joint_names),
                "stage_command_name": "motion",
                "stage0_weight": 0.00,
                "stage1_weight": 0.00,
                "stage2_weight": 1.00,
            },
        )
        self.rewards.stage2_lateral_velocity = RewTerm(
            func=mdp.stage_gated_track_lateral_reference_velocity_exp,
            weight=1.30,
            params={
                "command_name": "stage2_lateral",
                "std": 0.10,
                "stage_command_name": "motion",
                "enabled_stage": 2,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
