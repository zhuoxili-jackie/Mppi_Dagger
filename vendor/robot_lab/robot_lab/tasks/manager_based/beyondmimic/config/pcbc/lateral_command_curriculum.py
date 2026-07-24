# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

from .lateral_guided_env_cfg import PcbCLateralGuidedCarTrunkRobustEnv17Cfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


_ROBOT_LAB_EXTENSION_ROOT = Path(__file__).resolve().parents[6]
PCBC_BIPEDAL_STAND_MOTION_DIR = (
    _ROBOT_LAB_EXTENSION_ROOT / "data/Motions/pcbc_lateral_708"
)
PCBC_BIPEDAL_STAND_MOTIONS = (
    (PCBC_BIPEDAL_STAND_MOTION_DIR / "trajectory_trotting_acc_f015.npz", -0.15),
    (PCBC_BIPEDAL_STAND_MOTION_DIR / "trajectory_trotting_acc_f01.npz", -0.10),
    (PCBC_BIPEDAL_STAND_MOTION_DIR / "trajectory_trotting_acc_f005.npz", -0.05),
    (PCBC_BIPEDAL_STAND_MOTION_DIR / "trajectory_trotting_acc_005.npz", 0.05),
    (PCBC_BIPEDAL_STAND_MOTION_DIR / "trajectory_trotting_acc_01.npz", 0.10),
    (PCBC_BIPEDAL_STAND_MOTION_DIR / "trajectory_trotting_acc_015.npz", 0.15),
)
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
PCBC_BIPEDAL_STAND_INIT_ROOT_POS = (0.0, 0.0, 0.741806)
PCBC_BIPEDAL_STAND_INIT_ROOT_ROT = (0.7514101294, -0.0067587112, -0.6597561136, -0.0076816513)
PCBC_CURRICULUM_CAR_TRUNK_POS = (4.851, 0.0, 0.0)


# Earlier stages simplify command sampling while all stages retain v17 safety,
# observations and rewards with the 708 bipedal-stand references.
_COMMAND_STAGES = (
    {"min_speed": 0.10, "max_speed": 0.10, "standing_probability": 0.10},
    {"min_speed": 0.05, "max_speed": 0.10, "standing_probability": 0.15},
    {"min_speed": 0.05, "max_speed": 0.15, "standing_probability": 0.20},
    {"min_speed": 0.03, "max_speed": 0.15, "standing_probability": 0.20},
)


def _as_env_ids(env: ManagerBasedRLEnv, env_ids: Sequence[int] | slice) -> torch.Tensor:
    if isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)[env_ids]
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    return ids.reshape(-1)


def _reset_evaluation_window(env: ManagerBasedRLEnv) -> None:
    for direction in ("negative", "positive"):
        setattr(env, f"_lateral_curriculum_{direction}_count", 0)
        setattr(env, f"_lateral_curriculum_{direction}_tracking_sum", 0.0)
        setattr(env, f"_lateral_curriculum_{direction}_timeout_sum", 0.0)
    env._lateral_curriculum_standing_count = 0
    env._lateral_curriculum_standing_timeout_sum = 0.0
    env._lateral_curriculum_standing_planar_penalty_sum = 0.0


def _initialize_curriculum_state(env: ManagerBasedRLEnv) -> None:
    env._lateral_curriculum_level = 0
    env._lateral_curriculum_last_negative_tracking = 0.0
    env._lateral_curriculum_last_positive_tracking = 0.0
    env._lateral_curriculum_last_negative_timeout = 0.0
    env._lateral_curriculum_last_positive_timeout = 0.0
    env._lateral_curriculum_last_standing_timeout = 0.0
    env._lateral_curriculum_last_standing_planar_penalty = 0.0
    _reset_evaluation_window(env)


def _apply_command_stage(env: ManagerBasedRLEnv, command_name: str) -> dict[str, float]:
    level = int(env._lateral_curriculum_level)
    stage = _COMMAND_STAGES[level]
    command_cfg = env.command_manager.get_term(command_name).cfg
    command_cfg.min_abs_target_velocity = stage["min_speed"]
    command_cfg.target_velocity_range = (-stage["max_speed"], stage["max_speed"])
    command_cfg.standing_probability = stage["standing_probability"]
    return stage


def _curriculum_metrics(env: ManagerBasedRLEnv, stage: dict[str, float]) -> dict[str, float]:
    return {
        "level": float(env._lateral_curriculum_level),
        "min_speed": stage["min_speed"],
        "max_speed": stage["max_speed"],
        "standing_probability": stage["standing_probability"],
        "negative_y_tracking": env._lateral_curriculum_last_negative_tracking,
        "positive_y_tracking": env._lateral_curriculum_last_positive_tracking,
        "negative_y_timeout": env._lateral_curriculum_last_negative_timeout,
        "positive_y_timeout": env._lateral_curriculum_last_positive_timeout,
        "standing_timeout": env._lateral_curriculum_last_standing_timeout,
        "standing_planar_penalty": env._lateral_curriculum_last_standing_planar_penalty,
    }


def lateral_command_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | slice,
    command_name: str,
    reward_term_name: str,
    standing_penalty_term_name: str,
    min_direction_episodes: int,
    min_standing_episodes: int,
    tracking_threshold: float,
    timeout_threshold: float,
    standing_penalty_threshold: float,
) -> dict[str, float]:
    """Promote command difficulty after both lateral directions are reliable.

    Moving episodes are evaluated separately for negative and positive y
    commands. This prevents a strong direction from hiding a weak direction in
    the aggregate reward. Promotion is monotonic and changes command sampling
    only; all v17 MDP terms remain untouched.
    """
    if not hasattr(env, "_lateral_curriculum_level"):
        _initialize_curriculum_state(env)

    stage = _apply_command_stage(env, command_name)
    ids = _as_env_ids(env, env_ids)
    if ids.numel() == 0 or env.common_step_counter == 0:
        return _curriculum_metrics(env, stage)

    command_y = env.command_manager.get_term(command_name).velocity_command[ids, 1]
    timeouts = env.termination_manager.time_outs[ids].float()
    reward_cfg = env.reward_manager.get_term_cfg(reward_term_name)
    reward_scale = max(float(reward_cfg.weight) * float(env.max_episode_length_s), 1.0e-6)
    tracking = torch.clamp(env.reward_manager._episode_sums[reward_term_name][ids] / reward_scale, 0.0, 1.0)
    standing_planar_penalty = torch.clamp(
        -env.reward_manager._episode_sums[standing_penalty_term_name][ids]
        / float(env.max_episode_length_s),
        min=0.0,
    )

    direction_masks = {
        "negative": command_y < -1.0e-6,
        "positive": command_y > 1.0e-6,
    }
    for direction, mask in direction_masks.items():
        count = int(mask.sum().item())
        if count == 0:
            continue
        setattr(
            env,
            f"_lateral_curriculum_{direction}_count",
            getattr(env, f"_lateral_curriculum_{direction}_count") + count,
        )
        setattr(
            env,
            f"_lateral_curriculum_{direction}_tracking_sum",
            getattr(env, f"_lateral_curriculum_{direction}_tracking_sum") + float(tracking[mask].sum().item()),
        )
        setattr(
            env,
            f"_lateral_curriculum_{direction}_timeout_sum",
            getattr(env, f"_lateral_curriculum_{direction}_timeout_sum") + float(timeouts[mask].sum().item()),
        )

    standing_mask = torch.abs(command_y) <= 1.0e-6
    standing_count = int(standing_mask.sum().item())
    if standing_count > 0:
        env._lateral_curriculum_standing_count += standing_count
        env._lateral_curriculum_standing_timeout_sum += float(timeouts[standing_mask].sum().item())
        env._lateral_curriculum_standing_planar_penalty_sum += float(
            standing_planar_penalty[standing_mask].sum().item()
        )

    enough_data = (
        env._lateral_curriculum_negative_count >= min_direction_episodes
        and env._lateral_curriculum_positive_count >= min_direction_episodes
        and env._lateral_curriculum_standing_count >= min_standing_episodes
    )
    if not enough_data:
        return _curriculum_metrics(env, stage)

    negative_tracking = (
        env._lateral_curriculum_negative_tracking_sum / env._lateral_curriculum_negative_count
    )
    positive_tracking = (
        env._lateral_curriculum_positive_tracking_sum / env._lateral_curriculum_positive_count
    )
    negative_timeout = env._lateral_curriculum_negative_timeout_sum / env._lateral_curriculum_negative_count
    positive_timeout = env._lateral_curriculum_positive_timeout_sum / env._lateral_curriculum_positive_count
    standing_timeout = env._lateral_curriculum_standing_timeout_sum / env._lateral_curriculum_standing_count
    standing_planar_penalty = (
        env._lateral_curriculum_standing_planar_penalty_sum / env._lateral_curriculum_standing_count
    )

    env._lateral_curriculum_last_negative_tracking = negative_tracking
    env._lateral_curriculum_last_positive_tracking = positive_tracking
    env._lateral_curriculum_last_negative_timeout = negative_timeout
    env._lateral_curriculum_last_positive_timeout = positive_timeout
    env._lateral_curriculum_last_standing_timeout = standing_timeout
    env._lateral_curriculum_last_standing_planar_penalty = standing_planar_penalty

    passed = (
        min(negative_tracking, positive_tracking) >= tracking_threshold
        and min(negative_timeout, positive_timeout, standing_timeout) >= timeout_threshold
        and standing_planar_penalty <= standing_penalty_threshold
    )
    if passed and env._lateral_curriculum_level < len(_COMMAND_STAGES) - 1:
        env._lateral_curriculum_level += 1

    # Evaluate independent windows so poor early samples cannot permanently
    # prevent promotion after the policy improves.
    _reset_evaluation_window(env)
    stage = _apply_command_stage(env, command_name)
    return _curriculum_metrics(env, stage)


@configclass
class PcbCLateralCommandCurriculumCfg:
    """Performance-gated lateral command curriculum."""

    command_levels = CurrTerm(
        func=lateral_command_curriculum,
        params={
            "command_name": "motion",
            "reward_term_name": "track_lateral_velocity",
            "standing_penalty_term_name": "car_trunk_zero_cmd_base_planar_velocity",
            "min_direction_episodes": 256,
            "min_standing_episodes": 64,
            "tracking_threshold": 0.65,
            "timeout_threshold": 0.85,
            "standing_penalty_threshold": 0.08,
        },
    )


@configclass
class PcbCLateralGuidedCarTrunkCommandCurriculumEnvCfg(PcbCLateralGuidedCarTrunkRobustEnv17Cfg):
    """v17 safety with 708 bipedal-stand command curriculum learning."""

    curriculum: PcbCLateralCommandCurriculumCfg = PcbCLateralCommandCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()

        motion_files = [str(path) for path, _ in PCBC_BIPEDAL_STAND_MOTIONS]
        lateral_velocities = [velocity for _, velocity in PCBC_BIPEDAL_STAND_MOTIONS]
        self.commands.motion.motion_file = motion_files[0]
        self.commands.motion.motion_files = motion_files
        self.commands.motion.lateral_velocities = lateral_velocities

        self.scene.box.init_state.pos = PCBC_CURRICULUM_CAR_TRUNK_POS
        self.scene.robot.init_state.pos = PCBC_BIPEDAL_STAND_INIT_ROOT_POS
        self.scene.robot.init_state.rot = PCBC_BIPEDAL_STAND_INIT_ROOT_ROT
        self.scene.robot.init_state.joint_pos = dict(PCBC_BIPEDAL_STAND_INIT_JOINT_POS)

        # Match stage zero before the first command-manager reset. The
        # curriculum function owns all later changes to these three fields.
        self.commands.motion.min_abs_target_velocity = _COMMAND_STAGES[0]["min_speed"]
        self.commands.motion.target_velocity_range = (
            -_COMMAND_STAGES[0]["max_speed"],
            _COMMAND_STAGES[0]["max_speed"],
        )
        self.commands.motion.standing_probability = _COMMAND_STAGES[0]["standing_probability"]
