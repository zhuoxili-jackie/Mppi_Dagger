# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Bipedal-stand lateral locomotion task for pcbC on the high platform.

This file is intentionally a reduced copy for the 708 bipedal-stand lateral
references.  It flattens the previous v17 inheritance chain into one class so
that future tuning is easy to reason about.
"""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import robot_lab.tasks.manager_based.beyondmimic.mdp as mdp

from .lateral_guided_env_cfg import PcbCLateralGuidedCarTrunkRobustEnv17Cfg
from .pure_imitation import PcbCBeyondMimicFlatV1StandEnvCfg

_ROBOT_LAB_EXTENSION_ROOT = Path(__file__).resolve().parents[6]
PCBC_BIPEDAL_STAND_MOTION_DIR = (
    _ROBOT_LAB_EXTENSION_ROOT / "data/Motions/pcbc_lateral_708"
)
PCBC_CAR_TRUNK_USD = (
    _ROBOT_LAB_EXTENSION_ROOT
    / "data/Robots/pcbC/pcb_v2_description_0.88/mesh/530X6U_simple.usd"
)
PCBC_CAR_TRUNK_POS = (4.851, 0.0, 0.0)
PCBC_CAR_TRUNK_ROT = (0.0, 0.0, 0.0, 1.0)
PCBC_CAR_TRUNK_Y_SCALE = 6.0


def pcbc_car_trunk_cfg(name: str) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(PCBC_CAR_TRUNK_USD),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1000.0),
            scale=(1.0, PCBC_CAR_TRUNK_Y_SCALE, 1.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=PCBC_CAR_TRUNK_POS,
            rot=PCBC_CAR_TRUNK_ROT,
        ),
    )


def apply_lateral_car_trunk_scene(env_cfg) -> None:
    env_cfg.scene.box = pcbc_car_trunk_cfg("CarTrunk")


# 708 bipedal-stand lateral references, converted to 50 Hz NPZ.
PCBC_BIPEDAL_STAND_MOTIONS = [
    (PCBC_BIPEDAL_STAND_MOTION_DIR / "trajectory_trotting_acc_f015.npz", -0.15),
    (PCBC_BIPEDAL_STAND_MOTION_DIR / "trajectory_trotting_acc_f01.npz", -0.10),
    (PCBC_BIPEDAL_STAND_MOTION_DIR / "trajectory_trotting_acc_f005.npz", -0.05),
    (PCBC_BIPEDAL_STAND_MOTION_DIR / "trajectory_trotting_acc_005.npz", 0.05),
    (PCBC_BIPEDAL_STAND_MOTION_DIR / "trajectory_trotting_acc_01.npz", 0.10),
    (PCBC_BIPEDAL_STAND_MOTION_DIR / "trajectory_trotting_acc_015.npz", 0.15),
]


# First frame of the 708 bipedal-stand lateral references.
PCBC_BIPEDAL_STAND_INIT_JOINT_POS = {
    "FL_hip_joint": 0.0,
    "FR_hip_joint": 0.0,
    "RL_hip_joint": 0.0,
    "RR_hip_joint": 0.0,
    "FL_thigh_joint": 0.610812,
    "FR_thigh_joint": 0.610812,
    "RL_thigh_joint": 0.698123,
    "RR_thigh_joint": 0.698123,
    "FL_calf_joint": -0.785317,
    "FR_calf_joint": -0.785317,
    "RL_calf_joint": 1.0472,
    "RR_calf_joint": 1.0472,
    "FL_foot_joint": 0.0,
    "FR_foot_joint": 0.0,
    "RL_foot_joint": 0.0,
    "RR_foot_joint": 0.0,
}


@configclass
class PcbCLateralBipedalStandEnvCfg(PcbCBeyondMimicFlatV1StandEnvCfg):
    """Flattened v17 safety setup bound to the 708 bipedal-stand references."""

    def __post_init__(self):
        super().__post_init__()

        apply_lateral_car_trunk_scene(self)

        # ------------------------------------------------------------------
        # Reset pose and command references
        # ------------------------------------------------------------------
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.741806)
        self.scene.robot.init_state.rot = (0.7514101294, -0.0067587112, -0.6597561136, -0.0076816513)
        self.scene.robot.init_state.joint_pos = dict(PCBC_BIPEDAL_STAND_INIT_JOINT_POS)

        motion_files = [str(path) for path, _ in PCBC_BIPEDAL_STAND_MOTIONS]
        lateral_velocities = [velocity for _, velocity in PCBC_BIPEDAL_STAND_MOTIONS]
        self.commands.motion = mdp.LateralReferenceMotionCommandCfg(
            asset_name="robot",
            motion_file=motion_files[0],
            motion_files=motion_files,
            lateral_velocities=lateral_velocities,
            standing_probability=0.20,
            target_velocity_range=(-0.15, 0.15),
            min_abs_target_velocity=0.03,
            reset_at_first_frame=True,
            anchor_body_name=self.base_link_name,
            body_names=self.body_names,
            resampling_time_range=(1.0e9, 1.0e9),
            debug_vis=True,
            pose_range={
                "x": (0.0, 0.1),
                "y": (-0.01, 0.01),
                "z": (-0.01, 0.01),
                "roll": (-0.03, 0.03),
                "pitch": (-0.03, 0.03),
                "yaw": (-0.04, 0.04),
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

        # ------------------------------------------------------------------
        # Observations: keep the existing lateral actor layout.
        # ------------------------------------------------------------------
        self.observations.policy.command = ObsTerm(
            func=mdp.lateral_actor_motion_command,
            params={"command_name": "motion"},
        )
        self.observations.policy.motion_anchor_ori_b = ObsTerm(
            func=mdp.lateral_actor_anchor_ori_b,
            params={"command_name": "motion"},
        )
        self.observations.policy.base_velocity_command = ObsTerm(
            func=mdp.lateral_velocity_command,
            params={"command_name": "motion"},
        )
        self.observations.policy.motion_stage = ObsTerm(func=mdp.constant_zero_scalar)

        self.observations.critic.base_velocity_command = ObsTerm(
            func=mdp.lateral_velocity_command,
            params={"command_name": "motion"},
        )
        self.observations.critic.motion_stage = ObsTerm(func=mdp.constant_zero_scalar)

        # ------------------------------------------------------------------
        # Randomization and high-platform material model from the v17 chain.
        # ------------------------------------------------------------------
        non_wheel_bodies = [body_name for body_name in self.body_names if not body_name.endswith("_foot_link")]
        front_wheel_bodies = ["FL_foot_link", "FR_foot_link"]
        rear_wheel_bodies = ["RL_foot_link", "RR_foot_link"]

        self.scene.box.spawn.physics_material = sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.2,
            dynamic_friction=1.0,
        )
        self.events.randomize_rigid_body_material.params.update(
            {
                "asset_cfg": SceneEntityCfg("robot", body_names=non_wheel_bodies),
                "static_friction_range": (0.35, 0.95),
                "dynamic_friction_range": (0.25, 0.75),
                "restitution_range": (0.0, 0.04),
                "num_buckets": 64,
            }
        )
        self.events.randomize_box_material = EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("box"),
                "static_friction_range": (0.45, 1.50),
                "dynamic_friction_range": (0.30, 1.10),
                "restitution_range": (0.0, 0.04),
                "num_buckets": 64,
            },
        )
        self.events.randomize_front_wheel_material = EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=front_wheel_bodies),
                "static_friction_range": (0.45, 1.40),
                "dynamic_friction_range": (0.30, 1.00),
                "restitution_range": (0.0, 0.04),
                "num_buckets": 64,
            },
        )
        self.events.randomize_rear_wheel_material = EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=rear_wheel_bodies),
                "static_friction_range": (0.35, 1.65),
                "dynamic_friction_range": (0.25, 1.20),
                "restitution_range": (0.0, 0.05),
                "num_buckets": 64,
            },
        )

        # ------------------------------------------------------------------
        # Dynamic 708 imitation plus lateral velocity tracking.
        # ------------------------------------------------------------------
        self.rewards.motion_global_anchor_pos = None
        self.rewards.motion_global_anchor_ori = None
        self.rewards.motion_body_lin_vel = None
        self.rewards.motion_body_ang_vel = None
        self.rewards.motion_wheel_joint_vel = None
        self.rewards.undesired_contacts = None

        self.rewards.motion_body_ori = RewTerm(
            func=mdp.reference_anchor_orientation_error_l2,
            weight=-0.60,
            params={"command_name": "motion"},
        )
        # Follow the selected 708 frame so the rear legs retain the demonstrated
        # lateral gait instead of converging to a wheel-only rigid stance.
        self.rewards.motion_joint_pos = RewTerm(
            func=mdp.motion_joint_position_error_exp,
            weight=1.65,
            params={
                "command_name": "motion",
                "std": 0.40,
                "asset_cfg": SceneEntityCfg("robot", joint_names=self.leg_joint_names),
            },
        )
        self.rewards.motion_joint_vel = RewTerm(
            func=mdp.motion_joint_velocity_error_exp,
            weight=0.65,
            params={
                "command_name": "motion",
                "std": 1.50,
                "asset_cfg": SceneEntityCfg("robot", joint_names=self.leg_joint_names),
            },
        )

        # Track all four wheel centers from the current reference frame.  This
        # preserves the demonstrated opening/closing while rejecting extra
        # policy-created splay or fore-aft staggering.
        self.rewards.motion_body_pos = RewTerm(
            func=mdp.motion_relative_body_position_error_exp,
            weight=1.00,
            params={
                "command_name": "motion",
                "std": 0.080,
                "body_names": ["FL_foot_link", "FR_foot_link", "RL_foot_link", "RR_foot_link"],
            },
        )

        self.rewards.track_lateral_velocity = RewTerm(
            func=mdp.track_lateral_reference_velocity_with_heading_stability_exp,
            weight=1.90,
            params={
                "command_name": "motion",
                "std": 0.10,
                "command_threshold": 0.03,
                "yaw_tolerance": 0.05,
                "yaw_std": 0.12,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.base_lin_vel_z_l2 = RewTerm(
            func=mdp.lin_vel_z_l2,
            weight=-0.30,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.base_ang_vel_xy_l2 = RewTerm(
            func=mdp.ang_vel_xy_l2,
            weight=-0.08,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        # ------------------------------------------------------------------
        # Flattened v17 high-platform safety terms.
        # ------------------------------------------------------------------
        self.rewards.car_trunk_base_x_drift = RewTerm(
            func=mdp.box_local_axis_position_deviation_reset_l2_penalty,
            weight=-1.15,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="Base_link"),
                "command_name": "motion",
                "box_name": "box",
                "axis": 0,
                "deadband": 0.085,
                "reset_steps": 2,
                "max_penalty": 1.0,
                "cache_key": "bipedal_base",
            },
        )
        self.rewards.car_trunk_left_front_wheel_x_drift = RewTerm(
            func=mdp.box_local_axis_position_deviation_reset_l2_penalty,
            weight=-1.55,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="FL_foot_link"),
                "command_name": "motion",
                "box_name": "box",
                "axis": 0,
                "deadband": 0.040,
                "reset_steps": 2,
                "max_penalty": 1.0,
                "cache_key": "bipedal_front_left_wheel",
            },
        )
        self.rewards.car_trunk_right_front_wheel_x_drift = RewTerm(
            func=mdp.box_local_axis_position_deviation_reset_l2_penalty,
            weight=-1.55,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="FR_foot_link"),
                "command_name": "motion",
                "box_name": "box",
                "axis": 0,
                "deadband": 0.040,
                "reset_steps": 2,
                "max_penalty": 1.0,
                "cache_key": "bipedal_front_right_wheel",
            },
        )
        self.rewards.car_trunk_front_wheel_x_pair_gap = RewTerm(
            func=mdp.box_local_axis_pair_difference_l2_penalty,
            weight=-1.35,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=["FL_foot_link", "FR_foot_link"]),
                "box_name": "box",
                "axis": 0,
                "deadband": 0.040,
                "max_penalty": 1.0,
            },
        )
        self.rewards.car_trunk_lateral_x_velocity = RewTerm(
            func=mdp.local_x_velocity_l2_penalty,
            weight=-1.10,
            params={
                "command_name": "motion",
                "x_command_deadzone": 0.03,
                "velocity_deadband": 0.040,
                "velocity_scale": 0.140,
                "max_penalty": 2.0,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.car_trunk_heading_drift = RewTerm(
            func=mdp.y_command_yaw_drift_l2_penalty,
            weight=-0.45,
            params={
                "command_name": "motion",
                "command_threshold": 0.03,
                "yaw_command_deadzone": 0.03,
                "deadband": 0.04,
                "max_penalty": 1.0,
            },
        )
        self.rewards.car_trunk_roll_error = RewTerm(
            func=mdp.motion_tilt_error_l2_penalty,
            weight=-0.45,
            params={
                "command_name": "motion",
                "deadband": 0.04,
                "max_penalty": 1.0,
            },
        )
        self.rewards.car_trunk_front_wheel_contact = RewTerm(
            func=mdp.y_command_feet_contact_reward,
            weight=0.10,
            params={
                "command_name": "motion",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["FL_foot_link", "FR_foot_link"]),
                "command_threshold": 0.03,
                "force_threshold": 8.0,
            },
        )
        self.rewards.car_trunk_front_wheel_force_balance = RewTerm(
            func=mdp.y_command_feet_force_balance_l1_penalty,
            weight=-0.03,
            params={
                "command_name": "motion",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["FL_foot_link", "FR_foot_link"]),
                "command_threshold": 0.03,
                "force_threshold": 8.0,
                "force_scale": 110.0,
                "max_penalty": 1.0,
            },
        )
        self.rewards.car_trunk_front_wheel_backward_margin = RewTerm(
            func=mdp.box_local_axis_backward_margin_reset_max_penalty,
            weight=-3.20,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=["FL_foot_link", "FR_foot_link"]),
                "command_name": "motion",
                "box_name": "box",
                "axis": 0,
                "backward_margin": 0.026,
                "unsafe_direction": -1.0,
                "reset_steps": 2,
                "max_penalty": 1.0,
                "cache_key": "bipedal_front_wheels",
            },
        )
        self.rewards.car_trunk_front_wheel_z_drop_margin = RewTerm(
            func=mdp.box_local_axis_drop_margin_reset_max_penalty,
            weight=-3.00,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=["FL_foot_link", "FR_foot_link"]),
                "command_name": "motion",
                "box_name": "box",
                "axis": 2,
                "drop_margin": 0.018,
                "reset_steps": 2,
                "max_penalty": 1.0,
                "cache_key": "bipedal_front_wheels",
            },
        )

        self.rewards.car_trunk_zero_cmd_wheel_motion = RewTerm(
            func=mdp.zero_command_wheel_motion_l2_penalty,
            weight=-1.15,
            params={
                "command_name": "motion",
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=["FL_foot_joint", "FR_foot_joint", "RL_foot_joint", "RR_foot_joint"],
                ),
                "command_deadband": 0.025,
                "wheel_velocity_deadband": 0.045,
                "max_penalty": 2.0,
            },
        )
        self.rewards.car_trunk_zero_cmd_front_wheel_contact = RewTerm(
            func=mdp.zero_command_feet_contact_reward,
            weight=0.20,
            params={
                "command_name": "motion",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["FL_foot_link", "FR_foot_link"]),
                "command_deadband": 0.025,
                "force_threshold": 8.0,
            },
        )
        self.rewards.car_trunk_zero_cmd_base_planar_velocity = RewTerm(
            func=mdp.zero_command_base_planar_velocity_l2_penalty,
            weight=-0.36,
            params={
                "command_name": "motion",
                "command_deadband": 0.025,
                "velocity_deadband": 0.008,
                "velocity_scale": 0.060,
                "max_penalty": 2.0,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.car_trunk_zero_cmd_front_wheel_backward_margin = RewTerm(
            func=mdp.zero_command_box_local_axis_backward_margin_reset_max_penalty,
            weight=-2.80,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=["FL_foot_link", "FR_foot_link"]),
                "command_name": "motion",
                "command_deadband": 0.025,
                "box_name": "box",
                "axis": 0,
                "backward_margin": 0.018,
                "unsafe_direction": -1.0,
                "max_penalty": 0.25,
                "cache_key": "bipedal_zero_backward",
            },
        )
        self.rewards.car_trunk_zero_cmd_front_wheel_z_drop_margin = RewTerm(
            func=mdp.zero_command_box_local_axis_drop_margin_reset_max_penalty,
            weight=-2.40,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=["FL_foot_link", "FR_foot_link"]),
                "command_name": "motion",
                "command_deadband": 0.025,
                "box_name": "box",
                "axis": 2,
                "drop_margin": 0.012,
                "max_penalty": 0.25,
                "cache_key": "bipedal_zero_drop",
            },
        )
        self.rewards.car_trunk_zero_cmd_base_x_drift = RewTerm(
            func=mdp.zero_command_box_local_axis_position_deviation_reset_l2_penalty,
            weight=-1.20,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="Base_link"),
                "command_name": "motion",
                "box_name": "box",
                "axis": 0,
                "command_deadband": 0.030,
                "deadband": 0.035,
                "reset_steps": 2,
                "max_penalty": 1.0,
                "cache_key": "bipedal_zero_base_x",
            },
        )
        self.rewards.car_trunk_zero_cmd_base_z_drop = RewTerm(
            func=mdp.zero_command_box_local_axis_drop_margin_reset_max_penalty,
            weight=-1.20,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="Base_link"),
                "command_name": "motion",
                "box_name": "box",
                "axis": 2,
                "command_deadband": 0.030,
                "drop_margin": 0.020,
                "reset_steps": 2,
                "max_penalty": 1.0,
                "cache_key": "bipedal_zero_base_z",
            },
        )
        self.rewards.car_trunk_zero_cmd_joint_motion = RewTerm(
            func=mdp.zero_command_joint_motion_l2_penalty,
            weight=-0.18,
            params={
                "command_name": "motion",
                "asset_cfg": SceneEntityCfg("robot", joint_names=self.leg_joint_names),
                "command_deadband": 0.025,
                "joint_velocity_deadband": 0.060,
                "max_penalty": 2.0,
            },
        )

        # Disable hand-authored gait shaping; the selected 708 sequence supplies
        # the demonstrated bipedal lateral pattern directly.
        self.rewards.car_trunk_diagonal_leg_motion = None
        self.rewards.car_trunk_diagonal_motion_balance = None
        self.rewards.car_trunk_rear_sync_without_front = None

        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.joint_acc_l2.weight = -2.5e-7
        self.rewards.joint_torques_l2.weight = -1.0e-5

        # ------------------------------------------------------------------
        # Terminations / episode length
        # ------------------------------------------------------------------
        self.terminations.anchor_pos.params["threshold"] = 0.25
        self.terminations.anchor_ori.params["threshold"] = 0.80
        self.terminations.ee_body_pos.params["threshold"] = 0.25
        self.terminations.illegal_contact = None

        # 708 references have 332 frames at 50 Hz.  Ending at 6 s avoids an
        # in-episode reference resample/state rewrite near the tail frame.
        self.episode_length_s = 6.0


@configclass
class PcbCLateralBipedalStandStableEnvCfg(PcbCLateralBipedalStandEnvCfg):
    """v0 with low-speed response and reset-relative anti-inward-drift shaping.

    The 708 motions, reset pose randomization, observations, and terminations
    stay identical to v0. Only soft rewards are adjusted so this remains a
    clean from-scratch comparison against the existing bipedal-stand task.
    """

    def __post_init__(self):
        super().__post_init__()

        # At 0.1 m/s the old std=0.10 reward still paid 36.8% when the robot did
        # not move. Tighten the main tracker and add a narrow 0.1 m/s band term
        # without changing the response objective at 0.05 or 0.15 m/s.
        self.rewards.track_lateral_velocity.weight = 2.10
        self.rewards.track_lateral_velocity.params["std"] = 0.075
        self.rewards.track_lateral_velocity_010 = RewTerm(
            func=mdp.track_lateral_reference_velocity_band_exp,
            weight=0.35,
            params={
                "command_name": "motion",
                "std": 0.050,
                "min_abs_command": 0.075,
                "max_abs_command": 0.125,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        # The 708 right-moving sequence has larger rear-leg x staggering and
        # orientation variation than the left sequence. This direction-gated
        # term keeps the policy close to that demonstrated geometry instead of
        # amplifying it into a progressively twisted support polygon.
        self.rewards.car_trunk_right_cmd_wheel_reference = RewTerm(
            func=mdp.y_command_motion_relative_body_position_error_exp,
            weight=0.35,
            params={
                "command_name": "motion",
                "std": 0.055,
                "body_names": ["FL_foot_link", "FR_foot_link", "RL_foot_link", "RR_foot_link"],
                "direction": -1.0,
                "command_threshold": 0.03,
            },
        )

        # v0's position penalties used raw metre-squared errors. Their logged
        # values therefore looked nearly zero even after visible centimetres of
        # drift. Express the same reset-relative constraints on useful scales;
        # each episode still holds around its own randomized landing position.
        self.rewards.car_trunk_base_x_drift.weight = -0.55
        self.rewards.car_trunk_base_x_drift.params.update(
            {"deadband": 0.050, "position_scale": 0.050}
        )
        self.rewards.car_trunk_left_front_wheel_x_drift.weight = -0.65
        self.rewards.car_trunk_left_front_wheel_x_drift.params.update(
            {"deadband": 0.030, "position_scale": 0.045}
        )
        self.rewards.car_trunk_right_front_wheel_x_drift.weight = -0.65
        self.rewards.car_trunk_right_front_wheel_x_drift.params.update(
            {"deadband": 0.030, "position_scale": 0.045}
        )

        # Positive box-local x is the inward/top direction. Allow normal gait
        # motion, but make additional inward travel beyond 4.5 cm progressively
        # expensive. This is a soft guard, not a termination or fixed pose.
        self.rewards.car_trunk_inward_margin = RewTerm(
            func=mdp.box_local_axis_backward_margin_reset_max_penalty,
            weight=-0.85,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", body_names=["Base_link", "FL_foot_link", "FR_foot_link"]
                ),
                "command_name": "motion",
                "box_name": "box",
                "axis": 0,
                "backward_margin": 0.045,
                "unsafe_direction": 1.0,
                "reset_steps": 2,
                "position_scale": 0.040,
                "max_penalty": 1.0,
                "cache_key": "bipedal_stable_inward",
            },
        )

        self.rewards.car_trunk_lateral_x_velocity.weight = -0.75
        self.rewards.car_trunk_lateral_x_velocity.params.update(
            {"velocity_deadband": 0.030, "velocity_scale": 0.120}
        )
        self.rewards.car_trunk_heading_drift.weight = -0.25
        self.rewards.car_trunk_heading_drift.params["deadband"] = 0.025

        # Keep front support but reduce the incentive to solve lateral tracking
        # by continuously pressing the body farther into the trunk.
        self.rewards.car_trunk_front_wheel_contact.weight = 0.06
        self.rewards.car_trunk_front_wheel_force_balance.weight = -0.04

        # When command returns to zero, hold the base near the episode's own
        # stop/reset x region on a centimetre-scale gradient.
        self.rewards.car_trunk_zero_cmd_base_x_drift.weight = -0.80
        self.rewards.car_trunk_zero_cmd_base_x_drift.params.update(
            {"deadband": 0.020, "position_scale": 0.050}
        )

        # Match the successful v17 smoothness level more closely. These terms
        # remain weak enough for the 708 legs to execute their reference gait.
        self.rewards.base_ang_vel_xy_l2.weight = -0.12
        self.rewards.action_rate_l2.weight = -0.014
        self.rewards.joint_acc_l2.weight = -3.5e-7

# 存在掉坡面问题，特别是0.1速度的时候，移动动作类似amp
@configclass
class PcbCLateralBipedalStandStableV1EnvCfg(PcbCLateralBipedalStandStableEnvCfg):
    """Stable-v0 with symmetric low-speed response and non-returning zero hold."""

    def __post_init__(self):
        super().__post_init__()

        # The negative-y 708 motions contain more yaw and leg lift than their
        # positive-y counterparts. Do not add a second imitation objective only
        # on that side; the shared joint/body imitation terms remain active.
        self.rewards.car_trunk_right_cmd_wheel_reference = None

        # Decouple velocity reward from the asymmetric reference yaw. The
        # dedicated reset-heading term below keeps both directions straight.
        self.rewards.track_lateral_velocity.params.update(
            {"yaw_tolerance": 0.30, "yaw_std": 0.30}
        )
        self.rewards.track_right_lateral_velocity_010 = RewTerm(
            func=mdp.track_lateral_reference_velocity_direction_band_exp,
            weight=0.30,
            params={
                "command_name": "motion",
                "std": 0.045,
                "min_abs_command": 0.075,
                "max_abs_command": 0.125,
                "direction": -1.0,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        # Preserve the demonstrated pitch/roll but do not imitate the 708 yaw
        # excursion. Heading is held around each randomized reset orientation.
        self.rewards.motion_body_ori = RewTerm(
            func=mdp.motion_tilt_error_l2_penalty,
            weight=-0.45,
            params={
                "command_name": "motion",
                "deadband": 0.035,
                "max_penalty": 1.0,
            },
        )
        self.rewards.car_trunk_heading_drift = None
        self.rewards.car_trunk_reset_heading = RewTerm(
            func=mdp.y_command_heading_deviation_reset_l2_penalty,
            weight=-0.35,
            params={
                "command_name": "motion",
                "command_threshold": 0.03,
                "deadband": 0.035,
                "angle_scale": 0.12,
                "reset_steps": 2,
                "max_penalty": 1.0,
                "cache_key": "bipedal_stable_v1",
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        # Do not pull a stopped robot back toward its reset x position. Static
        # damping is tightened instead, so zero command suppresses motion
        # without creating a position-servo/contact fight after right motion.
        self.rewards.car_trunk_zero_cmd_base_x_drift = None
        self.rewards.car_trunk_zero_cmd_base_planar_velocity.weight = -0.50
        self.rewards.car_trunk_zero_cmd_base_planar_velocity.params.update(
            {"velocity_deadband": 0.006, "velocity_scale": 0.055}
        )
        self.rewards.car_trunk_zero_cmd_wheel_motion.params["wheel_velocity_deadband"] = 0.040
        self.rewards.car_trunk_zero_cmd_joint_motion.weight = -0.22
        self.rewards.car_trunk_zero_cmd_joint_motion.params["joint_velocity_deadband"] = 0.050

        # Stable-v0's stronger all-command smoothing raised the activation cost
        # of the more dynamic negative-y gait. Restore v0 levels and let the
        # zero-command-specific terms above handle stopping.
        self.rewards.action_rate_l2.weight = -0.010
        self.rewards.joint_acc_l2.weight = -2.5e-7


@configclass
class PcbCLateralBipedalStandStableV3EnvCfg(PcbCLateralBipedalStandStableV1EnvCfg):
    """Stable-v1 single-variable trunk-distance comparison at x=4.776 m."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.box.init_state.pos = (4.776, 0.0, 0.0)
        self.commands.motion.pose_range["x"] = (0.0, 0.0)
        self.commands.motion.pose_range["z"] = (0.0, 0.0)
