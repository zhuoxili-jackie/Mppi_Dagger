# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Data-guided y-lateral locomotion task for pcbC.

This task is intentionally independent from the boarding baseline, completion
gate, and residual controller. Reference motions guide PPO toward a useful
lateral gait while the actor learns to respond to a y-velocity command.
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

from .pure_imitation import PcbCBeyondMimicFlatV1StandEnvCfg

_REPO_ROOT = Path(__file__).resolve().parents[8]
PCBC_LATERAL_MOTION_DIR = _REPO_ROOT / "pcbB样侧向移动数据608"
PCBC_LATERAL_703_MOTION_DIR = _REPO_ROOT / "pcbc侧向数据703"
PCBC_LATERAL_0706_MOTION_DIR = _REPO_ROOT / "pcb_0706"
PCBC_LATERAL_FANQU_713_MOTION_DIR = _REPO_ROOT / "fanqu_713"
PCBC_CAR_TRUNK_USD = Path(__file__).resolve().parents[6] / "data/Robots/pcbC/pcb_v2_description_0.88/mesh/530X6U_simple.usd"
PCBC_CAR_TRUNK_POS = (4.951, 0.0, 0.0)
PCBC_CAR_TRUNK_V6_POS = (4.7585847092, 0.0, 0.0038228869)
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



# The source CSV files are 100 Hz. Convert them to 50 Hz NPZ files with the
# existing pcbC csv_to_npz tool before starting training.
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

PCBC_LATERAL_703_MOTIONS = [
    (PCBC_LATERAL_703_MOTION_DIR / "trajectory_walking_sideways_sc_v0.20.npz", -0.20),
    (PCBC_LATERAL_703_MOTION_DIR / "trajectory_walking_sideways_sc_v0.15.npz", -0.15),
    (PCBC_LATERAL_703_MOTION_DIR / "trajectory_walking_sideways_sc_v0.10.npz", -0.10),
    (PCBC_LATERAL_703_MOTION_DIR / "trajectory_walking_sideways_sc_v0.05.npz", -0.05),
    (PCBC_LATERAL_703_MOTION_DIR / "trajectory_walking_sideways_sc_left_v0.05.npz", 0.05),
    (PCBC_LATERAL_703_MOTION_DIR / "trajectory_walking_sideways_sc_left_v0.10.npz", 0.10),
    (PCBC_LATERAL_703_MOTION_DIR / "trajectory_walking_sideways_sc_left_v0.15.npz", 0.15),
    (PCBC_LATERAL_703_MOTION_DIR / "trajectory_walking_sideways_sc_left_v0.20.npz", 0.20),
]

PCBC_LATERAL_0706_MOTIONS = [
    (PCBC_LATERAL_0706_MOTION_DIR / "tripod_tilt_v020.npz", -0.20),
    (PCBC_LATERAL_0706_MOTION_DIR / "tripod_tilt_v015.npz", -0.15),
    (PCBC_LATERAL_0706_MOTION_DIR / "tripod_tilt_v010.npz", -0.10),
    (PCBC_LATERAL_0706_MOTION_DIR / "tripod_tilt_v005.npz", -0.05),
    (PCBC_LATERAL_0706_MOTION_DIR / "tripod_left_v005.npz", 0.05),
    (PCBC_LATERAL_0706_MOTION_DIR / "tripod_left_v010.npz", 0.10),
    (PCBC_LATERAL_0706_MOTION_DIR / "tripod_left_v015.npz", 0.15),
    (PCBC_LATERAL_0706_MOTION_DIR / "tripod_left_v020.npz", 0.20),
]

PCBC_LATERAL_FANQU_713_MOTIONS = [
    (PCBC_LATERAL_FANQU_713_MOTION_DIR / "trajectory_trot_diag_h03_symramp_sc_right_v0.20.npz", -0.20),
    (PCBC_LATERAL_FANQU_713_MOTION_DIR / "trajectory_trot_diag_h03_symramp_sc_right_v0.15.npz", -0.15),
    (PCBC_LATERAL_FANQU_713_MOTION_DIR / "trajectory_trot_diag_h03_symramp_sc_right_v0.10.npz", -0.10),
    (PCBC_LATERAL_FANQU_713_MOTION_DIR / "trajectory_trot_diag_h03_symramp_sc_right_v0.05.npz", -0.05),
    (PCBC_LATERAL_FANQU_713_MOTION_DIR / "trajectory_trot_diag_h03_symramp_sc_left_v0.05.npz", 0.05),
    (PCBC_LATERAL_FANQU_713_MOTION_DIR / "trajectory_trot_diag_h03_symramp_sc_left_v0.10.npz", 0.10),
    (PCBC_LATERAL_FANQU_713_MOTION_DIR / "trajectory_trot_diag_h03_symramp_sc_left_v0.15.npz", 0.15),
    (PCBC_LATERAL_FANQU_713_MOTION_DIR / "trajectory_trot_diag_h03_symramp_sc_left_v0.20.npz", 0.20),
]

# Shared first-frame pose of all lateral references. Making this the articulation
# default ensures that a zero policy action holds the valid hitched reset pose.
PCBC_LATERAL_INIT_JOINT_POS = {
    "FL_hip_joint": 0.0,
    "FR_hip_joint": 0.0,
    "RL_hip_joint": 0.0,
    "RR_hip_joint": 0.0,
    "FL_thigh_joint": -0.81,
    "FR_thigh_joint": -0.81,
    "RL_thigh_joint": 0.65,
    "RR_thigh_joint": 0.65,
    "FL_calf_joint": 0.872,
    "FR_calf_joint": 0.872,
    "RL_calf_joint": 1.0,
    "RR_calf_joint": 1.0,
    "FL_foot_joint": 0.0,
    "FR_foot_joint": 0.0,
    "RL_foot_joint": 0.0,
    "RR_foot_joint": 0.0,
}


@configclass
class PcbCLateralGuidedEnvCfg(PcbCBeyondMimicFlatV1StandEnvCfg):
    """From-scratch y-command policy trained with weak lateral motion guidance."""

    def __post_init__(self):
        super().__post_init__()
        apply_lateral_car_trunk_scene(self)

        # The inherited baseline action uses the articulation default as its
        # offset. Align that offset with the lateral reset pose so action=0 does
        # not immediately pull the robot toward an incompatible standing pose.
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.72963)
        self.scene.robot.init_state.rot = (0.8522562647, 0.0, -0.5231245161, 0.0)
        self.scene.robot.init_state.joint_pos = dict(PCBC_LATERAL_INIT_JOINT_POS)

        motion_files = [str(path) for path, _ in PCBC_LATERAL_MOTIONS]
        lateral_velocities = [velocity for _, velocity in PCBC_LATERAL_MOTIONS]
        self.commands.motion = mdp.LateralReferenceMotionCommandCfg(
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
            resampling_time_range=(1.0e9, 1.0e9),
            debug_vis=True,
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

        # Match the baseline actor's 93-D observation layout. Dynamic reference
        # frames are hidden from the actor and remain available to the critic.
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

        # The reference is a gait guide, not a global trajectory target.
        self.rewards.motion_global_anchor_pos = None
        self.rewards.motion_global_anchor_ori = None
        self.rewards.motion_body_pos = None
        self.rewards.motion_body_lin_vel = None
        self.rewards.motion_body_ang_vel = None
        self.rewards.motion_wheel_joint_vel = None
        self.rewards.undesired_contacts = None

        # Track the current data-frame base orientation directly. Unlike the
        # standard relative-body reward, this target never follows robot yaw.
        self.rewards.motion_body_ori = RewTerm(
            func=mdp.reference_anchor_orientation_error_l2,
            weight=-0.35,
            params={"command_name": "motion"},
        )
        self.rewards.motion_joint_pos.weight = 0.80
        self.rewards.motion_joint_pos.params["std"] = 0.55
        self.rewards.motion_joint_vel.weight = 0.35
        self.rewards.motion_joint_vel.params["std"] = 1.50

        self.rewards.track_lateral_velocity = RewTerm(
            func=mdp.track_lateral_reference_velocity_exp,
            weight=2.5,
            params={
                "command_name": "motion",
                "std": 0.10,
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

        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.joint_acc_l2.weight = -2.5e-7
        self.rewards.joint_torques_l2.weight = -1.0e-5

        # Every reset is written from a valid hitched lateral-reference state,
        # with only the small pose/velocity/joint perturbations configured above.
        self.terminations.anchor_pos.params["threshold"] = 0.25
        self.terminations.anchor_ori.params["threshold"] = 0.80
        self.terminations.ee_body_pos.params["threshold"] = 0.25
        self.terminations.illegal_contact = None

        # The converted references contain 332 frames at 50 Hz (6.64 s).
        # Ending at 6 s avoids an in-episode reference resample/state rewrite.
        self.episode_length_s = 6.0

# v0 会有默认在斜坡上运动的趋势，平地也会在斜坡上，不给速度有微向前的移动，policy_lateral_623_12000.onnx
@configclass
class PcbCLateralGuidedCarTrunkRobustEnvCfg(PcbCLateralGuidedEnvCfg):
    """Car-trunk lateral task with surface material randomization and x-drift guards."""

    def __post_init__(self):
        super().__post_init__()

        non_wheel_bodies = [body_name for body_name in self.body_names if not body_name.endswith("_foot_link")]
        front_wheel_bodies = ["FL_foot_link", "FR_foot_link"]
        rear_wheel_bodies = ["RL_foot_link", "RR_foot_link"]

        # Car-trunk training deliberately sees a wider reset envelope than the
        # base high-platform task, matching the less repeatable sloped contact.
        self.commands.motion.pose_range = {
            "x": (-0.01, 0.01),
            "y": (-0.015, 0.015),
            "z": (-0.015, 0.015),
            "roll": (-0.05, 0.05),
            "pitch": (-0.05, 0.05),
            "yaw": (-0.08, 0.08),
        }
        self.commands.motion.joint_position_range = (-0.05, 0.05)

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

        self.rewards.track_lateral_velocity = RewTerm(
            func=mdp.track_lateral_reference_velocity_with_heading_stability_exp,
            weight=2.5,
            params={
                "command_name": "motion",
                "std": 0.10,
                "command_threshold": 0.03,
                "yaw_tolerance": 0.12,
                "yaw_std": 0.20,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.car_trunk_base_x_drift = RewTerm(
            func=mdp.box_local_axis_position_deviation_l2_penalty,
            weight=-0.80,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="Base_link"),
                "box_name": "box",
                "axis": 0,
                "deadband": 0.16,
                "max_penalty": 1.0,
                "cache_key": "base",
            },
        )
        self.rewards.car_trunk_left_front_wheel_x_drift = RewTerm(
            func=mdp.box_local_axis_position_deviation_l2_penalty,
            weight=-1.00,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="FL_foot_link"),
                "box_name": "box",
                "axis": 0,
                "deadband": 0.06,
                "max_penalty": 1.0,
                "cache_key": "front_left_wheel",
            },
        )
        self.rewards.car_trunk_right_front_wheel_x_drift = RewTerm(
            func=mdp.box_local_axis_position_deviation_l2_penalty,
            weight=-1.00,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="FR_foot_link"),
                "box_name": "box",
                "axis": 0,
                "deadband": 0.06,
                "max_penalty": 1.0,
                "cache_key": "front_right_wheel",
            },
        )
        self.rewards.car_trunk_front_wheel_x_pair_gap = RewTerm(
            func=mdp.box_local_axis_pair_difference_l2_penalty,
            weight=-1.20,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=["FL_foot_link", "FR_foot_link"]),
                "box_name": "box",
                "axis": 0,
                "deadband": 0.05,
                "max_penalty": 1.0,
            },
        )
        self.rewards.car_trunk_lateral_x_velocity = RewTerm(
            func=mdp.y_command_local_x_velocity_l2_penalty,
            weight=-0.45,
            params={
                "command_name": "motion",
                "command_threshold": 0.03,
                "x_command_deadzone": 0.03,
                "velocity_deadband": 0.08,
                "velocity_scale": 0.20,
                "max_penalty": 2.0,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.car_trunk_heading_drift = RewTerm(
            func=mdp.y_command_yaw_drift_l2_penalty,
            weight=-0.35,
            params={
                "command_name": "motion",
                "command_threshold": 0.03,
                "yaw_command_deadzone": 0.03,
                "deadband": 0.10,
                "max_penalty": 1.0,
            },
        )
        self.rewards.car_trunk_roll_error = RewTerm(
            func=mdp.motion_tilt_error_l2_penalty,
            weight=-0.35,
            params={
                "command_name": "motion",
                "deadband": 0.10,
                "max_penalty": 1.0,
            },
        )
        self.rewards.car_trunk_zero_cmd_wheel_motion = RewTerm(
            func=mdp.zero_command_wheel_motion_l2_penalty,
            weight=-0.30,
            params={
                "command_name": "motion",
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=["FL_foot_joint", "FR_foot_joint", "RL_foot_joint", "RR_foot_joint"],
                ),
                "command_deadband": 0.025,
                "wheel_velocity_deadband": 0.20,
                "max_penalty": 2.0,
            },
        )

# V1 对角步态，左边移动会往后退，以及到斜坡都还有后退的趋势，0速度也后退，右边表现良好，0速度能保持静止，但是前腿习惯性往前伸。policy_lateral_624_v1.onnx
@configclass
class PcbCLateralGuidedCarTrunkRobustEnv1Cfg(PcbCLateralGuidedCarTrunkRobustEnvCfg):
    """Car-trunk lateral task that preserves the reset contact x-region."""

    def __post_init__(self):
        super().__post_init__()

        # Env1 starts from the current valid trunk contact region and only
        # samples forward along the trunk local x axis. Backward resets put the
        # front wheels near the ramp edge and encourage the policy to seek the
        # slope/transition area instead of staying where it landed.
        self.commands.motion.pose_range = {
            "x": (0.0, 0.1),
            "y": (-0.01, 0.01),
            "z": (-0.01, 0.01),
            "roll": (-0.03, 0.03),
            "pitch": (-0.03, 0.03),
            "yaw": (-0.04, 0.04),
        }
        self.commands.motion.joint_position_range = (-0.03, 0.03)

        self.rewards.car_trunk_base_x_drift = RewTerm(
            func=mdp.box_local_axis_position_deviation_reset_l2_penalty,
            weight=-1.00,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="Base_link"),
                "command_name": "motion",
                "box_name": "box",
                "axis": 0,
                "deadband": 0.10,
                "reset_steps": 2,
                "max_penalty": 1.0,
                "cache_key": "base_env1",
            },
        )
        self.rewards.car_trunk_left_front_wheel_x_drift = RewTerm(
            func=mdp.box_local_axis_position_deviation_reset_l2_penalty,
            weight=-1.40,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="FL_foot_link"),
                "command_name": "motion",
                "box_name": "box",
                "axis": 0,
                "deadband": 0.045,
                "reset_steps": 2,
                "max_penalty": 1.0,
                "cache_key": "front_left_wheel_env1",
            },
        )
        self.rewards.car_trunk_right_front_wheel_x_drift = RewTerm(
            func=mdp.box_local_axis_position_deviation_reset_l2_penalty,
            weight=-1.40,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="FR_foot_link"),
                "command_name": "motion",
                "box_name": "box",
                "axis": 0,
                "deadband": 0.045,
                "reset_steps": 2,
                "max_penalty": 1.0,
                "cache_key": "front_right_wheel_env1",
            },
        )
        self.rewards.car_trunk_lateral_x_velocity = RewTerm(
            func=mdp.local_x_velocity_l2_penalty,
            weight=-0.70,
            params={
                "command_name": "motion",
                "x_command_deadzone": 0.03,
                "velocity_deadband": 0.06,
                "velocity_scale": 0.18,
                "max_penalty": 2.0,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.car_trunk_zero_cmd_wheel_motion = RewTerm(
            func=mdp.zero_command_wheel_motion_l2_penalty,
            weight=-0.45,
            params={
                "command_name": "motion",
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=["FL_foot_joint", "FR_foot_joint", "RL_foot_joint", "RR_foot_joint"],
                ),
                "command_deadband": 0.025,
                "wheel_velocity_deadband": 0.16,
                "max_penalty": 2.0,
            },
        )

# V2 左边找坡面，坡面上移动良好，0速度整体也会向前打滑，步态三足蹦policy_624_V2.onnx
@configclass
class PcbCLateralGuidedCarTrunkRobustEnv2Cfg(PcbCLateralGuidedCarTrunkRobustEnv1Cfg):
    """Car-trunk lateral task with stronger left-motion x-drift suppression."""

    def __post_init__(self):
        super().__post_init__()

        # V1 with x=(0, 0.1) covers the flat top, but it also makes the policy
        # see enough transition states to rediscover the ramp as a left-motion
        # support. V2 keeps resets on the safe flat-side band and lets the
        # reset-relative x rewards preserve that region.
        self.commands.motion.pose_range = {
            "x": (0.02, 0.08),
            "y": (-0.008, 0.008),
            "z": (-0.008, 0.008),
            "roll": (-0.025, 0.025),
            "pitch": (-0.025, 0.025),
            "yaw": (-0.035, 0.035),
        }
        self.commands.motion.joint_position_range = (-0.025, 0.025)

        self.commands.motion.standing_probability = 0.25

        self.rewards.car_trunk_base_x_drift.weight = -1.20
        self.rewards.car_trunk_base_x_drift.params["deadband"] = 0.08

        self.rewards.car_trunk_left_front_wheel_x_drift.weight = -1.90
        self.rewards.car_trunk_left_front_wheel_x_drift.params["deadband"] = 0.035
        self.rewards.car_trunk_right_front_wheel_x_drift.weight = -1.90
        self.rewards.car_trunk_right_front_wheel_x_drift.params["deadband"] = 0.035

        self.rewards.car_trunk_front_wheel_x_pair_gap.weight = -1.80
        self.rewards.car_trunk_front_wheel_x_pair_gap.params["deadband"] = 0.035

        self.rewards.car_trunk_lateral_x_velocity.weight = -1.00
        self.rewards.car_trunk_lateral_x_velocity.params["velocity_deadband"] = 0.045
        self.rewards.car_trunk_lateral_x_velocity.params["velocity_scale"] = 0.14

        self.rewards.car_trunk_left_lateral_x_velocity = RewTerm(
            func=mdp.y_direction_local_x_velocity_l2_penalty,
            weight=-0.70,
            params={
                "command_name": "motion",
                "direction": -1.0,
                "command_threshold": 0.03,
                "x_command_deadzone": 0.03,
                "velocity_deadband": 0.035,
                "velocity_scale": 0.12,
                "max_penalty": 2.0,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.car_trunk_left_front_wheel_contact = RewTerm(
            func=mdp.y_direction_feet_contact_reward,
            weight=0.35,
            params={
                "command_name": "motion",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["FL_foot_link", "FR_foot_link"]),
                "direction": -1.0,
                "command_threshold": 0.03,
                "force_threshold": 12.0,
            },
        )
        self.rewards.car_trunk_left_front_wheel_force_balance = RewTerm(
            func=mdp.y_direction_feet_force_balance_l1_penalty,
            weight=-0.18,
            params={
                "command_name": "motion",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["FL_foot_link", "FR_foot_link"]),
                "direction": -1.0,
                "command_threshold": 0.03,
                "force_threshold": 12.0,
                "force_scale": 90.0,
                "max_penalty": 1.0,
            },
        )

        self.rewards.car_trunk_zero_cmd_wheel_motion.weight = -0.75
        self.rewards.car_trunk_zero_cmd_wheel_motion.params["wheel_velocity_deadband"] = 0.10

# v3:找坡面，而且坡面会在reset在下来一点那格，0速度静止，对角步态，轮子一前一后也有，但不明显
@configclass
class PcbCLateralGuidedCarTrunkRobustEnv3Cfg(PcbCLateralGuidedCarTrunkRobustEnv1Cfg):
    """v1 gait base with symmetric x-hold and stronger zero-speed stability."""

    def __post_init__(self):
        super().__post_init__()

        # Preserve v1's reset envelope because its gait is closest to the
        # desired alternating lateral step. The reset-relative x rewards below
        # are responsible for keeping each episode near its own landing region
        # instead of seeking the ramp.
        self.commands.motion.pose_range = {
            "x": (0.0, 0.1),
            "y": (-0.01, 0.01),
            "z": (-0.01, 0.01),
            "roll": (-0.03, 0.03),
            "pitch": (-0.03, 0.03),
            "yaw": (-0.04, 0.04),
        }
        self.commands.motion.joint_position_range = (-0.03, 0.03)
        self.commands.motion.standing_probability = 0.25

        # Light, symmetric front-wheel contact shaping. This only prevents
        # front wheel unloading; it does not force a particular gait phase.
        self.rewards.car_trunk_front_wheel_contact = RewTerm(
            func=mdp.y_command_feet_contact_reward,
            weight=0.14,
            params={
                "command_name": "motion",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["FL_foot_link", "FR_foot_link"]),
                "command_threshold": 0.03,
                "force_threshold": 10.0,
            },
        )
        self.rewards.car_trunk_front_wheel_force_balance = RewTerm(
            func=mdp.y_command_feet_force_balance_l1_penalty,
            weight=-0.07,
            params={
                "command_name": "motion",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["FL_foot_link", "FR_foot_link"]),
                "command_threshold": 0.03,
                "force_threshold": 10.0,
                "force_scale": 110.0,
                "max_penalty": 1.0,
            },
        )

        # Symmetric x-hold: reduce the "move backward to find the slope" trick
        # while still allowing the v1 alternating gait to emerge.
        self.rewards.car_trunk_base_x_drift.weight = -1.15
        self.rewards.car_trunk_base_x_drift.params["deadband"] = 0.085
        self.rewards.car_trunk_left_front_wheel_x_drift.weight = -1.55
        self.rewards.car_trunk_left_front_wheel_x_drift.params["deadband"] = 0.040
        self.rewards.car_trunk_right_front_wheel_x_drift.weight = -1.55
        self.rewards.car_trunk_right_front_wheel_x_drift.params["deadband"] = 0.040
        self.rewards.car_trunk_front_wheel_x_pair_gap.weight = -1.35
        self.rewards.car_trunk_front_wheel_x_pair_gap.params["deadband"] = 0.040
        self.rewards.car_trunk_lateral_x_velocity.weight = -0.95
        self.rewards.car_trunk_lateral_x_velocity.params["velocity_deadband"] = 0.045
        self.rewards.car_trunk_lateral_x_velocity.params["velocity_scale"] = 0.14

        # v2's useful part: stop wheel creep when command is zero.
        self.rewards.car_trunk_zero_cmd_wheel_motion.weight = -0.85
        self.rewards.car_trunk_zero_cmd_wheel_motion.params["wheel_velocity_deadband"] = 0.10
        self.rewards.base_ang_vel_xy_l2.weight = -0.12
        self.rewards.action_rate_l2.weight = -0.016
        self.rewards.joint_acc_l2.weight = -3.5e-7

# v4，不找坡面，0速度静止，姿态一前一后，0速度斜坡有时会一抽一抽的，有时候静止会在坡面2
@configclass
class PcbCLateralGuidedCarTrunkRobustEnv4Cfg(PcbCLateralGuidedCarTrunkRobustEnv3Cfg):
    """v3 with light diagonal gait participation shaping."""

    def __post_init__(self):
        super().__post_init__()

        # Do not raise min_abs_target_velocity here: v4's speed-only experiment
        # made both directions scrape. Instead add a soft diagonal participation
        # prior that discourages "one leg moves, three legs drag" without forcing
        # front wheels to leave the trunk surface.
        diagonal_a_joints = [
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
        ]
        diagonal_b_joints = [
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
        ]
        self.rewards.car_trunk_diagonal_leg_motion = RewTerm(
            func=mdp.y_command_diagonal_leg_motion_reward,
            weight=0.16,
            params={
                "command_name": "motion",
                "asset_cfg": SceneEntityCfg("robot"),
                "diag_a_joint_names": diagonal_a_joints,
                "diag_b_joint_names": diagonal_b_joints,
                "y_cmd_threshold": 0.03,
                "max_value": 2.0,
            },
        )
        self.rewards.car_trunk_diagonal_motion_balance = RewTerm(
            func=mdp.y_command_diagonal_motion_balance_penalty,
            weight=-0.10,
            params={
                "command_name": "motion",
                "asset_cfg": SceneEntityCfg("robot"),
                "diag_a_joint_names": diagonal_a_joints,
                "diag_b_joint_names": diagonal_b_joints,
                "y_cmd_threshold": 0.03,
                "max_value": 2.0,
            },
        )

# v5 找坡面，坡面位置临界，更靠后，姿势一前一后
@configclass
class PcbCLateralGuidedCarTrunkRobustEnv5Cfg(PcbCLateralGuidedCarTrunkRobustEnv4Cfg):
    """v4 plus four-wheel local-x geometry constraints for ramp stability."""

    def __post_init__(self):
        super().__post_init__()

        # V4 can still settle with the rear wheels one-ahead/one-behind. On the
        # ramp that twisted support polygon looks like a wheel is about to drop,
        # then the zero-speed stabilizer pulls it back. Keep the gait shaping,
        # but add direct local-x geometry constraints on the rear pair and the
        # left/right front-rear skew.
        self.rewards.car_trunk_front_wheel_x_pair_gap.params["deadband"] = 0.025
        self.rewards.car_trunk_rear_wheel_x_pair_gap = RewTerm(
            func=mdp.box_local_axis_pair_difference_l2_penalty,
            weight=-2.00,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=["RL_foot_link", "RR_foot_link"]),
                "box_name": "box",
                "axis": 0,
                "deadband": 0.018,
                "max_penalty": 1.0,
            },
        )
        self.rewards.car_trunk_four_wheel_x_skew = RewTerm(
            func=mdp.box_local_axis_four_body_skew_l2_penalty,
            weight=-2.00,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=["FL_foot_link", "FR_foot_link", "RL_foot_link", "RR_foot_link"],
                ),
                "box_name": "box",
                "axis": 0,
                "deadband": 0.020,
                "max_penalty": 1.0,
            },
        )
        self.rewards.car_trunk_backward_wheel_margin = RewTerm(
            func=mdp.box_local_axis_backward_margin_reset_l2_penalty,
            weight=-2.20,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=["FL_foot_link", "FR_foot_link", "RL_foot_link", "RR_foot_link"],
                ),
                "command_name": "motion",
                "box_name": "box",
                "axis": 0,
                "backward_margin": 0.025,
                "reset_steps": 2,
                "max_penalty": 1.0,
                "cache_key": "v5_all_wheels",
            },
        )
        self.rewards.car_trunk_diagonal_leg_motion.weight = 0.22
        self.rewards.car_trunk_diagonal_motion_balance.weight = -0.16
        self.rewards.base_ang_vel_xy_l2.weight = -0.16

# v6 高台训 打滑严重
@configclass
class PcbCLateralGuidedCarTrunkRobustEnv6Cfg(PcbCLateralGuidedCarTrunkRobustEnv4Cfg):
    """v4 with the trunk pulled closer and stronger no-backward lateral motion."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.box.init_state.pos = PCBC_CAR_TRUNK_V6_POS
        self.scene.box.spawn.collision_props = sim_utils.CollisionPropertiesCfg(contact_offset=0.02, rest_offset=0.0)

        # Train from the valid local contact region with limited perturbation.
        # The policy should either hold still or move laterally where it lands,
        # not first drift in local x to search for the ramp.
        self.commands.motion.pose_range = {
            "x": (0.02, 0.08),
            "y": (-0.008, 0.008),
            "z": (-0.008, 0.008),
            "roll": (-0.025, 0.025),
            "pitch": (-0.025, 0.025),
            "yaw": (-0.035, 0.035),
        }
        self.commands.motion.joint_position_range = (-0.025, 0.025)
        self.commands.motion.standing_probability = 0.25

        # Keep the useful zero-speed behavior from v3/v4.
        self.rewards.car_trunk_zero_cmd_wheel_motion.weight = -0.90
        self.rewards.car_trunk_zero_cmd_wheel_motion.params["wheel_velocity_deadband"] = 0.09

        # Make local-x drift expensive during lateral command. This is the direct
        # anti "search the ramp first" term for v6.
        self.rewards.car_trunk_base_x_drift.weight = -1.60
        self.rewards.car_trunk_base_x_drift.params["deadband"] = 0.060
        self.rewards.car_trunk_left_front_wheel_x_drift.weight = -2.10
        self.rewards.car_trunk_left_front_wheel_x_drift.params["deadband"] = 0.030
        self.rewards.car_trunk_right_front_wheel_x_drift.weight = -2.10
        self.rewards.car_trunk_right_front_wheel_x_drift.params["deadband"] = 0.030
        self.rewards.car_trunk_front_wheel_x_pair_gap.weight = -1.60
        self.rewards.car_trunk_front_wheel_x_pair_gap.params["deadband"] = 0.030
        self.rewards.car_trunk_lateral_x_velocity.weight = -1.45
        self.rewards.car_trunk_lateral_x_velocity.params["velocity_deadband"] = 0.030
        self.rewards.car_trunk_lateral_x_velocity.params["velocity_scale"] = 0.12

        # Keep reference velocity imitation strong enough to preserve the hitched
        # support posture. Reducing it made the policy chase lateral speed by
        # collapsing the rear legs and lying onto the trunk.
        self.rewards.motion_joint_vel.weight = 0.35
        self.rewards.track_lateral_velocity.weight = 2.50
        self.rewards.action_rate_l2.weight = -0.016
        self.rewards.joint_acc_l2.weight = -3.5e-7

        # Leave only a weak diagonal hint; the main job of v6 is trunk placement
        # plus anti-ramp-seeking, not hard gait phase shaping.
        self.rewards.car_trunk_diagonal_leg_motion.weight = 0.12
        self.rewards.car_trunk_diagonal_motion_balance.weight = -0.08
        self.rewards.base_ang_vel_xy_l2.weight = -0.14

# v7 前轮不再明显退到坡面 2；前轮没有明显掉到低层；0 速度稳定；前轮接触比之前更可靠。步态是后腿同时动，右移往里
@configclass
class PcbCLateralGuidedCarTrunkRobustEnv7Cfg(PcbCLateralGuidedCarTrunkRobustEnv4Cfg):
    """v4 with a one-sided front-wheel no-backward margin from reset."""

    def __post_init__(self):
        super().__post_init__()

        # Keep v4 as the sim2sim baseline. Only add the missing rule: after
        # reset, either front wheel may move toward the flat top, but neither
        # front wheel should move backward toward slope-2 / the drop edge.
        # Use max aggregation so one dangerous wheel cannot be averaged away.
        self.rewards.car_trunk_front_wheel_backward_margin = RewTerm(
            func=mdp.box_local_axis_backward_margin_reset_max_penalty,
            weight=-2.40,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=["FL_foot_link", "FR_foot_link"]),
                "command_name": "motion",
                "box_name": "box",
                "axis": 0,
                "backward_margin": 0.020,
                "unsafe_direction": -1.0,
                "reset_steps": 2,
                "max_penalty": 1.0,
                "cache_key": "v7_front_wheels",
            },
        )
        self.rewards.car_trunk_front_wheel_z_drop_margin = RewTerm(
            func=mdp.box_local_axis_drop_margin_reset_max_penalty,
            weight=-2.20,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=["FL_foot_link", "FR_foot_link"]),
                "command_name": "motion",
                "box_name": "box",
                "axis": 2,
                "drop_margin": 0.015,
                "reset_steps": 2,
                "max_penalty": 1.0,
                "cache_key": "v7_front_wheels",
            },
        )
        self.rewards.car_trunk_front_wheel_contact.weight = 0.22
        self.rewards.car_trunk_front_wheel_contact.params["force_threshold"] = 15.0
        self.rewards.car_trunk_lateral_x_velocity.weight = -1.10
        self.rewards.car_trunk_lateral_x_velocity.params["velocity_deadband"] = 0.040
        self.rewards.car_trunk_zero_cmd_wheel_motion.weight = -0.90
        self.rewards.car_trunk_zero_cmd_wheel_motion.params["wheel_velocity_deadband"] = 0.09

# v8 ，0速度静止打滑，左右两边同步步态
@configclass
class PcbCLateralGuidedCarTrunkRobustEnv8Cfg(PcbCLateralGuidedCarTrunkRobustEnv7Cfg):
    """v7 safety with diagonal gait shaping and front-wheel height guards."""

    def __post_init__(self):
        super().__post_init__()

        # V7 fixed the support region and zero-speed behavior. V8 keeps that
        # safety core, then blocks two remaining low-quality optima: both rear
        # legs moving together, or front wheels lifting to fake diagonal motion.
        self.rewards.car_trunk_heading_drift.weight = -0.55
        self.rewards.car_trunk_roll_error.weight = -0.50
        self.rewards.base_ang_vel_xy_l2.weight = -0.16

        self.rewards.car_trunk_diagonal_leg_motion.weight = 0.24
        self.rewards.car_trunk_diagonal_motion_balance.weight = -0.18
        self.rewards.car_trunk_rear_sync_without_front = RewTerm(
            func=mdp.y_command_rear_sync_without_front_penalty,
            weight=-0.36,
            params={
                "command_name": "motion",
                "asset_cfg": SceneEntityCfg("robot"),
                "rear_left_joint_names": ["RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"],
                "rear_right_joint_names": ["RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"],
                "front_left_joint_names": ["FL_hip_joint", "FL_thigh_joint", "FL_calf_joint"],
                "front_right_joint_names": ["FR_hip_joint", "FR_thigh_joint", "FR_calf_joint"],
                "y_cmd_threshold": 0.03,
                "min_rear_sync": 0.10,
                "front_relief": 0.70,
                "max_value": 2.0,
                "max_penalty": 1.0,
            },
        )
        self.rewards.car_trunk_front_wheel_z_rise_margin = RewTerm(
            func=mdp.box_local_axis_rise_margin_reset_max_penalty,
            weight=-1.80,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=["FL_foot_link", "FR_foot_link"]),
                "command_name": "motion",
                "box_name": "box",
                "axis": 2,
                "rise_margin": 0.020,
                "reset_steps": 2,
                "max_penalty": 1.0,
                "cache_key": "v8_front_wheels",
            },
        )
        self.rewards.car_trunk_front_wheel_backward_margin.weight = -2.70
        self.rewards.car_trunk_front_wheel_backward_margin.params["backward_margin"] = 0.018
        self.rewards.car_trunk_front_wheel_z_drop_margin.weight = -2.60
        self.rewards.car_trunk_front_wheel_z_drop_margin.params["drop_margin"] = 0.012
        self.rewards.car_trunk_front_wheel_contact.weight = 0.30
        self.rewards.car_trunk_front_wheel_contact.params["force_threshold"] = 18.0
        self.rewards.car_trunk_front_wheel_force_balance.weight = -0.12
        self.rewards.car_trunk_front_wheel_force_balance.params["force_threshold"] = 14.0
        self.rewards.car_trunk_front_wheel_force_balance.params["force_scale"] = 100.0
        self.rewards.action_rate_l2.weight = -0.018
        self.rewards.joint_acc_l2.weight = -4.0e-7


# v9 基于 v7：参考lt v6，去掉手工步态/前轮接触塑形，让对角步态从参考 imitation 中自然出现。
# 左边对角步态，右边后轮同时动，不会掉坡面2,右边偶尔有左前轮在坡面1边缘的情况，这时候有微微往前拉一抽一抽的感觉，往左给一下速度可以中和掉
@configclass
class PcbCLateralGuidedCarTrunkRobustEnv9Cfg(PcbCLateralGuidedCarTrunkRobustEnv7Cfg):
    """v7 safety with cleaner imitation-driven gait emergence."""

    def __post_init__(self):
        super().__post_init__()

        # Keep v7's one-sided safety rules for slope-2/drop avoidance, but do
        # not prescribe the gait phase directly. The older v6 result suggests
        # the diagonal gait emerges more reliably from reference imitation than
        # from joint-velocity gait rewards.
        self.commands.motion.standing_probability = 0.30
        self.rewards.motion_body_ori.weight = -0.60
        self.rewards.motion_joint_pos.weight = 1.05
        self.rewards.motion_joint_vel.weight = 0.50
        self.rewards.track_lateral_velocity.weight = 3.00

        self.rewards.car_trunk_front_wheel_contact = None
        self.rewards.car_trunk_front_wheel_force_balance = None
        self.rewards.car_trunk_front_wheel_x_pair_gap = None
        self.rewards.car_trunk_diagonal_leg_motion = None
        self.rewards.car_trunk_diagonal_motion_balance = None

        self.rewards.car_trunk_heading_drift.weight = -0.25
        self.rewards.car_trunk_heading_drift.params["deadband"] = 0.05
        self.rewards.car_trunk_roll_error.weight = -0.4
        self.rewards.car_trunk_roll_error.params["deadband"] = 0.05
        self.rewards.base_ang_vel_xy_l2.weight = -0.20

        # This is a safety guard rather than gait shaping: v7 already prevents
        # front wheels from backing onto slope-2 or dropping low; this closes
        # the remaining escape route where a front wheel rises and stays light.
        self.rewards.car_trunk_front_wheel_z_rise_margin = RewTerm(
            func=mdp.box_local_axis_rise_margin_reset_max_penalty,
            weight=-1.20,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=["FL_foot_link", "FR_foot_link"]),
                "command_name": "motion",
                "box_name": "box",
                "axis": 2,
                "rise_margin": 0.020,
                "reset_steps": 2,
                "max_penalty": 1.0,
                "cache_key": "v9_front_wheels",
            },
        )
        self.rewards.car_trunk_left_cmd_right_front_wheel_contact = RewTerm(
            func=mdp.y_direction_feet_contact_reward,
            weight=0.16,
            params={
                "command_name": "motion",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["FR_foot_link"]),
                "direction": -1.0,
                "command_threshold": 0.03,
                "force_threshold": 8.0,
            },
        )
        self.rewards.car_trunk_left_cmd_front_wheel_force_balance = RewTerm(
            func=mdp.y_direction_feet_force_balance_l1_penalty,
            weight=-0.08,
            params={
                "command_name": "motion",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["FL_foot_link", "FR_foot_link"]),
                "direction": -1.0,
                "command_threshold": 0.03,
                "force_threshold": 8.0,
                "force_scale": 120.0,
                "max_penalty": 1.0,
            },
        )
        self.rewards.car_trunk_front_wheel_backward_margin.weight = -2.40
        self.rewards.car_trunk_front_wheel_backward_margin.params["backward_margin"] = 0.020
        self.rewards.car_trunk_front_wheel_z_drop_margin.weight = -2.20
        self.rewards.car_trunk_front_wheel_z_drop_margin.params["drop_margin"] = 0.015
        self.rewards.car_trunk_lateral_x_velocity.weight = -1.10
        self.rewards.car_trunk_lateral_x_velocity.params["velocity_deadband"] = 0.040
        self.rewards.car_trunk_zero_cmd_wheel_motion.weight = -0.90
        self.rewards.car_trunk_zero_cmd_wheel_motion.params["wheel_velocity_deadband"] = 0.09
        self.rewards.action_rate_l2.weight = -0.016
        self.rewards.joint_acc_l2.weight = -3.5e-7


# v10 19999左移动0速度之后静止不下来，仍然左边横向移动，13000，左右横向都能走，小速度0.1也能响应，0速度也能静止， 10的表现都是前腿叉开，
@configclass
class PcbCLateralGuidedCarTrunkRobustEnv10Cfg(PcbCLateralGuidedCarTrunkRobustEnv9Cfg):
    """v9 with yaw-frame front alignment and full-box edge stability."""

    def __post_init__(self):
        super().__post_init__()

        # Keep v7's yaw-only box-local slope-2/drop guards untouched. The
        # visible one-ahead/one-behind front posture follows the robot/anchor
        # yaw red axis more than the yaw-only box frame, so constrain that here.
        self.rewards.car_trunk_front_wheel_x_pair_gap = None
        self.rewards.car_trunk_front_wheel_robot_yaw_x_pair_gap = RewTerm(
            func=mdp.motion_yaw_axis_pair_difference_l2_penalty,
            weight=-1.20,
            params={
                "command_name": "motion",
                "asset_cfg": SceneEntityCfg("robot", body_names=["FL_foot_link", "FR_foot_link"]),
                "axis": 0,
                "frame": "robot",
                "deadband": 0.022,
                "max_penalty": 1.0,
            },
        )
        self.rewards.car_trunk_zero_cmd_front_wheel_robot_yaw_x_pair_gap = RewTerm(
            func=mdp.motion_yaw_axis_pair_difference_l2_penalty,
            weight=-2.80,
            params={
                "command_name": "motion",
                "asset_cfg": SceneEntityCfg("robot", body_names=["FL_foot_link", "FR_foot_link"]),
                "axis": 0,
                "frame": "robot",
                "command_deadband": 0.050,
                "deadband": 0.014,
                "max_penalty": 1.0,
            },
        )
        self.rewards.car_trunk_front_wheel_full_box_x_pair_gap = RewTerm(
            func=mdp.full_box_local_axis_pair_difference_l2_penalty,
            weight=-0.50,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=["FL_foot_link", "FR_foot_link"]),
                "box_name": "box",
                "axis": 0,
                "deadband": 0.025,
                "max_penalty": 1.0,
            },
        )
        self.rewards.car_trunk_front_wheel_full_box_z_drop_margin = RewTerm(
            func=mdp.full_box_local_axis_drop_margin_reset_max_penalty,
            weight=-1.20,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=["FL_foot_link", "FR_foot_link"]),
                "command_name": "motion",
                "box_name": "box",
                "axis": 2,
                "drop_margin": 0.012,
                "reset_steps": 2,
                "max_penalty": 1.0,
                "cache_key": "v10_front_wheels_full_box",
            },
        )
        self.rewards.car_trunk_right_cmd_left_front_wheel_contact = RewTerm(
            func=mdp.y_direction_feet_contact_reward,
            weight=0.12,
            params={
                "command_name": "motion",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["FL_foot_link"]),
                "direction": 1.0,
                "command_threshold": 0.03,
                "force_threshold": 8.0,
            },
        )
        self.rewards.car_trunk_right_cmd_front_wheel_force_balance = RewTerm(
            func=mdp.y_direction_feet_force_balance_l1_penalty,
            weight=-0.06,
            params={
                "command_name": "motion",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["FL_foot_link", "FR_foot_link"]),
                "direction": 1.0,
                "command_threshold": 0.03,
                "force_threshold": 8.0,
                "force_scale": 120.0,
                "max_penalty": 1.0,
            },
        )
        self.rewards.car_trunk_zero_cmd_wheel_motion.weight = -1.10
        self.rewards.car_trunk_zero_cmd_wheel_motion.params["wheel_velocity_deadband"] = 0.07


# v11 基于 v10：姿态保持版。 0.3 速度响应最好，小速度响应一般。真机版本
@configclass
class PcbCLateralGuidedCarTrunkRobustEnv11Cfg(PcbCLateralGuidedCarTrunkRobustEnv10Cfg):
    """v10 posture-preserving variant for high-platform lateral motion."""

    def __post_init__(self):
        super().__post_init__()


        self.rewards.car_trunk_diagonal_leg_motion = None
        self.rewards.car_trunk_diagonal_motion_balance = None
        self.rewards.car_trunk_rear_sync_without_front = None


        self.rewards.track_lateral_velocity.weight = 2.70
        self.rewards.motion_body_ori.weight = -0.75
        self.rewards.motion_joint_pos.weight = 1.20
        self.rewards.motion_joint_vel.weight = 0.55


        self.rewards.car_trunk_front_wheel_robot_yaw_x_pair_gap.weight = -1.15
        self.rewards.car_trunk_front_wheel_robot_yaw_x_pair_gap.params["deadband"] = 0.020


        self.rewards.car_trunk_front_wheel_robot_yaw_y_spread = RewTerm(
            func=mdp.y_command_motion_yaw_pair_spread_error_l2_penalty,
            weight=-0.90,
            params={
                "command_name": "motion",
                "body_names": ["FL_foot_link", "FR_foot_link"],
                "axis": 1,
                "direction": None,
                "command_threshold": 0.03,
                "deadband": 0.025,
                "scale": 0.10,
                "max_penalty": 1.0,
            },
        )
        self.rewards.car_trunk_front_wheel_reference_pos = RewTerm(
            func=mdp.y_command_motion_relative_body_position_error_exp,
            weight=0.16,
            params={
                "command_name": "motion",
                "std": 0.09,
                "body_names": ["FL_foot_link", "FR_foot_link"],
                "direction": None,
                "command_threshold": 0.03,
            },
        )


        self.rewards.car_trunk_front_wheel_full_box_x_pair_gap.weight = -0.45
        self.rewards.car_trunk_front_wheel_full_box_z_drop_margin.weight = -1.10

        self.rewards.car_trunk_zero_cmd_front_wheel_robot_yaw_x_pair_gap.weight = -3.10
        self.rewards.car_trunk_zero_cmd_front_wheel_robot_yaw_x_pair_gap.params["command_deadband"] = 0.060
        self.rewards.car_trunk_zero_cmd_front_wheel_robot_yaw_x_pair_gap.params["deadband"] = 0.012
        self.rewards.car_trunk_zero_cmd_wheel_motion.weight = -1.35
        self.rewards.car_trunk_zero_cmd_wheel_motion.params["command_deadband"] = 0.060
        self.rewards.car_trunk_zero_cmd_wheel_motion.params["wheel_velocity_deadband"] = 0.050
        self.rewards.car_trunk_zero_cmd_base_lateral_velocity = RewTerm(
            func=mdp.zero_command_base_lateral_velocity_l2_penalty,
            weight=-0.45,
            params={
                "command_name": "motion",
                "command_deadband": 0.060,
                "velocity_deadband": 0.012,
                "velocity_scale": 0.06,
                "max_penalty": 2.0,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )


# v12 基于 v7 安全底座：703轨迹，12000效果最好 三足支撑/单腿小步轨迹训练。小速单侧抬腿，大速前抬后齐；左移时左前后微动，右前独抬，右后左后并步。
@configclass
class PcbCLateralGuidedCarTrunkRobustEnv12Cfg(PcbCLateralGuidedCarTrunkRobustEnv7Cfg):
    """v7 safety with 703 single-leg stepping references.

    Keep the validated v7 anti-slope2/drop/zero-speed stability terms, but let
    the new tri-support references decide the gait instead of forcing a
    hand-written diagonal pattern.
    """

    def __post_init__(self):
        super().__post_init__()

        motion_files = [str(path) for path, _ in PCBC_LATERAL_703_MOTIONS]
        lateral_velocities = [velocity for _, velocity in PCBC_LATERAL_703_MOTIONS]
        self.commands.motion.motion_file = motion_files[0]
        self.commands.motion.motion_files = motion_files
        self.commands.motion.lateral_velocities = lateral_velocities
        self.commands.motion.target_velocity_range = (-0.20, 0.20)
        self.commands.motion.min_abs_target_velocity = 0.03
        self.commands.motion.standing_probability = 0.20

        self.rewards.car_trunk_diagonal_leg_motion = None
        self.rewards.car_trunk_diagonal_motion_balance = None

        # Gait-debug priority: first make the 703 single-leg lift sequence
        # visible. Speed and hard front contact are kept secondary so the policy
        # cannot solve the task by scraping along the slope.
        self.rewards.car_trunk_front_wheel_contact.weight = 0.04
        self.rewards.car_trunk_front_wheel_contact.params["force_threshold"] = 6.0
        self.rewards.car_trunk_front_wheel_force_balance.weight = -0.01
        self.rewards.car_trunk_front_wheel_force_balance.params["force_threshold"] = 6.0

        self.rewards.motion_joint_pos.weight = 1.50
        self.rewards.motion_joint_vel.weight = 0.75
        self.rewards.track_lateral_velocity.weight = 1.40


# v15 基于 v12：小速度不响应，0.3响应，与v11效果类似
@configclass
class PcbCLateralGuidedCarTrunkRobustEnv15Cfg(PcbCLateralGuidedCarTrunkRobustEnv12Cfg):
    """v12 with zero-command hold and stronger front-wheel safe-region guards."""

    def __post_init__(self):
        super().__post_init__()

        # Keep v12's gait/reference setup, but recover enough front-wheel support
        # so zero-command does not stabilize an unsafe light-contact posture.
        self.commands.motion.standing_probability = 0.25

        self.rewards.car_trunk_front_wheel_contact.weight = 0.10
        self.rewards.car_trunk_front_wheel_contact.params["force_threshold"] = 8.0
        self.rewards.car_trunk_front_wheel_force_balance.weight = -0.03
        self.rewards.car_trunk_front_wheel_force_balance.params["force_threshold"] = 8.0

        # Strengthen the v7 safe-region rules: a wheel must not retreat toward
        # slope-2 or drop low, including after motion transitions into zero cmd.
        self.rewards.car_trunk_front_wheel_backward_margin.weight = -3.20
        self.rewards.car_trunk_front_wheel_backward_margin.params["backward_margin"] = 0.026
        self.rewards.car_trunk_front_wheel_z_drop_margin.weight = -3.00
        self.rewards.car_trunk_front_wheel_z_drop_margin.params["drop_margin"] = 0.018

        self.rewards.car_trunk_zero_cmd_wheel_motion.weight = -1.15
        self.rewards.car_trunk_zero_cmd_wheel_motion.params["command_deadband"] = 0.060
        self.rewards.car_trunk_zero_cmd_wheel_motion.params["wheel_velocity_deadband"] = 0.060
        self.rewards.car_trunk_zero_cmd_front_wheel_contact = RewTerm(
            func=mdp.zero_command_feet_contact_reward,
            weight=0.20,
            params={
                "command_name": "motion",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["FL_foot_link", "FR_foot_link"]),
                "command_deadband": 0.060,
                "force_threshold": 8.0,
            },
        )
        self.rewards.car_trunk_zero_cmd_base_planar_velocity = RewTerm(
            func=mdp.zero_command_base_planar_velocity_l2_penalty,
            weight=-0.36,
            params={
                "command_name": "motion",
                "command_deadband": 0.060,
                "velocity_deadband": 0.012,
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
                "command_deadband": 0.060,
                "box_name": "box",
                "axis": 0,
                "backward_margin": 0.018,
                "unsafe_direction": -1.0,
                "max_penalty": 0.25,
                "cache_key": "v15_zero_backward",
            },
        )
        self.rewards.car_trunk_zero_cmd_front_wheel_z_drop_margin = RewTerm(
            func=mdp.zero_command_box_local_axis_drop_margin_reset_max_penalty,
            weight=-2.40,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=["FL_foot_link", "FR_foot_link"]),
                "command_name": "motion",
                "command_deadband": 0.060,
                "box_name": "box",
                "axis": 2,
                "drop_margin": 0.012,
                "max_penalty": 0.25,
                "cache_key": "v15_zero_drop",
            },
        )
        self.rewards.car_trunk_zero_cmd_joint_motion = None


# v16 基于 v15：小速度响应，但是移动之后打滑
@configclass
class PcbCLateralGuidedCarTrunkRobustEnv16Cfg(PcbCLateralGuidedCarTrunkRobustEnv15Cfg):
    """v15 safety base with a narrow zero-command gate for low-speed response."""

    def __post_init__(self):
        super().__post_init__()

        # v15's 0.060 deadband made 0.05-ish reference commands and deployed
        # small joystick commands look like "stand still". Keep the same safety
        # terms, but only enable them for truly near-zero commands.
        zero_cmd_deadband = 0.025
        self.commands.motion.standing_probability = 0.20
        self.rewards.track_lateral_velocity.weight = 1.90

        self.rewards.car_trunk_zero_cmd_wheel_motion.params["command_deadband"] = zero_cmd_deadband
        self.rewards.car_trunk_zero_cmd_front_wheel_contact.params["command_deadband"] = zero_cmd_deadband
        self.rewards.car_trunk_zero_cmd_base_planar_velocity.params["command_deadband"] = zero_cmd_deadband
        self.rewards.car_trunk_zero_cmd_front_wheel_backward_margin.params["command_deadband"] = zero_cmd_deadband
        self.rewards.car_trunk_zero_cmd_front_wheel_backward_margin.params["cache_key"] = "v16_zero_backward"
        self.rewards.car_trunk_zero_cmd_front_wheel_z_drop_margin.params["command_deadband"] = zero_cmd_deadband
        self.rewards.car_trunk_zero_cmd_front_wheel_z_drop_margin.params["cache_key"] = "v16_zero_drop"


# v17 基于 v16：12000响应最好，真机效果响应挺好的，左边移动后静止有轻微的往后滑现象，落到坡面2能滑回来
@configclass
class PcbCLateralGuidedCarTrunkRobustEnv17Cfg(PcbCLateralGuidedCarTrunkRobustEnv16Cfg):
    """v16 with zero-command base-position guards for post-move slow slip."""

    def __post_init__(self):
        super().__post_init__()

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
                "cache_key": "v17_zero_base_x",
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
                "cache_key": "v17_zero_base_z",
            },
        )


# v19 基于 v17：保持速度响应、零速稳定和防滑底座，收紧前轮导轨间距与机身航向。静止后抖动
@configclass
class PcbCLateralGuidedCarTrunkRobustEnv19Cfg(PcbCLateralGuidedCarTrunkRobustEnv17Cfg):
    """v17 with rail-width front-wheel spacing and straighter body heading."""

    def __post_init__(self):
        super().__post_init__()

        front_wheels = SceneEntityCfg(
            "robot", body_names=["FL_foot_link", "FR_foot_link"]
        )

        # During lateral motion, keep the wheel centers at the 38 cm rail
        # spacing.  Measure in the robot yaw frame so body roll/pitch and world
        # orientation do not corrupt the lateral distance.
        self.rewards.car_trunk_moving_front_wheel_rail_spacing = RewTerm(
            func=mdp.motion_yaw_pair_target_distance_l2_penalty,
            weight=-1.25,
            params={
                "command_name": "motion",
                "asset_cfg": front_wheels,
                "axis": 1,
                "target_distance": 0.38,
                "frame": "robot",
                "command_mode": "moving",
                "command_threshold": 0.03,
                "deadband": 0.008,
                "scale": 0.040,
                "max_penalty": 1.0,
            },
        )

        # At zero command use a gentler band.  This prevents the final stance
        # from staying visibly pinched, without making the controller fight
        # static friction and twitch around an exact 38 cm equality.
        self.rewards.car_trunk_zero_cmd_front_wheel_rail_spacing = RewTerm(
            func=mdp.motion_yaw_pair_target_distance_l2_penalty,
            weight=-0.35,
            params={
                "command_name": "motion",
                "asset_cfg": front_wheels,
                "axis": 1,
                "target_distance": 0.38,
                "frame": "robot",
                "command_mode": "zero",
                "command_threshold": 0.025,
                "deadband": 0.012,
                "scale": 0.050,
                "max_penalty": 1.0,
            },
        )

        # A correct y spacing alone can still miss two parallel rails if one
        # wheel is ahead of the other.  Keep their robot-yaw x coordinates
        # aligned while moving, with a scale appropriate for centimeter errors.
        self.rewards.car_trunk_moving_front_wheel_x_alignment = RewTerm(
            func=mdp.motion_yaw_pair_target_distance_l2_penalty,
            weight=-0.60,
            params={
                "command_name": "motion",
                "asset_cfg": front_wheels,
                "axis": 0,
                "target_distance": 0.0,
                "frame": "robot",
                "command_mode": "moving",
                "command_threshold": 0.03,
                "deadband": 0.012,
                "scale": 0.040,
                "max_penalty": 1.0,
            },
        )

        # v17 allows about 5.7 degrees of yaw error before its heading penalty
        # starts.  Tighten that softly to about 2.9 degrees, while retaining
        # the original velocity reward and all zero-command/slope safeguards.
        self.rewards.car_trunk_heading_drift.weight = -0.55
        self.rewards.car_trunk_heading_drift.params["deadband"] = 0.05
        self.rewards.motion_body_ori.weight = -0.50
        self.rewards.base_ang_vel_xy_l2.weight = -0.12


# v20 基于 v19：完整保留 v19 的速度、姿态和移动轮距，仅取消零速 38 cm 绝对轮距纠形。
@configclass
class PcbCLateralGuidedCarTrunkRobustEnv20Cfg(PcbCLateralGuidedCarTrunkRobustEnv19Cfg):
    """Strict V19 ablation that removes only the zero-command rail-spacing servo."""

    def __post_init__(self):
        super().__post_init__()

        # Keep every validated V19 movement and posture term unchanged.  The
        # only ablation is the zero-command 38 cm servo, which conflicts with
        # the roughly 40.5 cm standing reference on high-friction contacts.
        self.rewards.car_trunk_zero_cmd_front_wheel_rail_spacing = None


