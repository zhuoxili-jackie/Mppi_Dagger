# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs import mdp as il_mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, quat_error_magnitude, quat_inv, quat_mul, yaw_quat

from robot_lab.tasks.manager_based.beyondmimic.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]


def _get_joint_ids(asset_cfg: SceneEntityCfg):
    if asset_cfg.joint_ids == slice(None):
        return slice(None)
    return asset_cfg.joint_ids


def action_rate_l2_clamped(
    env: ManagerBasedRLEnv,
    clip: float = 1.0,
    max_value: float | None = None,
) -> torch.Tensor:
    """Penalize action changes after bounding raw actions to a safe range.

    This keeps the action-rate term finite even if the policy briefly produces very
    large values before action clipping is applied elsewhere in the pipeline.
    """
    action = torch.nan_to_num(env.action_manager.action, nan=0.0, posinf=clip, neginf=-clip)
    prev_action = torch.nan_to_num(env.action_manager.prev_action, nan=0.0, posinf=clip, neginf=-clip)
    delta = torch.clamp(action, -clip, clip) - torch.clamp(prev_action, -clip, clip)
    penalty = torch.sum(torch.square(delta), dim=1)
    if max_value is not None:
        penalty = penalty.clamp(max=max_value)
    return penalty


def motion_global_anchor_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(command.body_quat_relative_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes])
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_joint_position_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    joint_ids = _get_joint_ids(asset_cfg)
    error = torch.square(command.joint_pos[:, joint_ids] - command.robot_joint_pos[:, joint_ids])
    return torch.exp(-error.mean(dim=-1) / std**2)


def motion_joint_velocity_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    joint_ids = _get_joint_ids(asset_cfg)
    error = torch.square(command.joint_vel[:, joint_ids] - command.robot_joint_vel[:, joint_ids])
    return torch.exp(-error.mean(dim=-1) / std**2)


def feet_contact_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_air = contact_sensor.compute_first_air(env.step_dt, env.physics_dt)[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_contact_time < threshold) * first_air, dim=-1)
    return reward


def stage_gated_track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    stage_command_name: str = "motion",
    enabled_stage: int = 2,
) -> torch.Tensor:
    """Enable lin-vel tracking reward only after motion stage reaches enabled_stage."""
    base_reward = il_mdp.track_lin_vel_xy_exp(env, command_name=command_name, std=std)
    motion_cmd: MotionCommand = env.command_manager.get_term(stage_command_name)
    gate = (motion_cmd.stage >= enabled_stage).to(base_reward.dtype)
    return base_reward * gate


def stage_gated_track_ang_vel_z_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    stage_command_name: str = "motion",
    enabled_stage: int = 2,
) -> torch.Tensor:
    """Enable yaw-rate tracking reward only after motion stage reaches enabled_stage."""
    base_reward = il_mdp.track_ang_vel_z_exp(env, command_name=command_name, std=std)
    motion_cmd: MotionCommand = env.command_manager.get_term(stage_command_name)
    gate = (motion_cmd.stage >= enabled_stage).to(base_reward.dtype)
    return base_reward * gate


def stage_capped_motion_global_anchor_position_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    stage_command_name: str = "motion",
    disable_from_stage: int = 2,
) -> torch.Tensor:
    """Enable anchor-pos mimic reward before command stage, disable in/after command stage."""
    base_reward = motion_global_anchor_position_error_exp(env, command_name=command_name, std=std)
    motion_cmd: MotionCommand = env.command_manager.get_term(stage_command_name)
    gate = (motion_cmd.stage < disable_from_stage).to(base_reward.dtype)
    return base_reward * gate


def stage_capped_motion_global_anchor_orientation_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    stage_command_name: str = "motion",
    disable_from_stage: int = 2,
) -> torch.Tensor:
    """Enable anchor-ori mimic reward before command stage, disable in/after command stage."""
    base_reward = motion_global_anchor_orientation_error_exp(env, command_name=command_name, std=std)
    motion_cmd: MotionCommand = env.command_manager.get_term(stage_command_name)
    gate = (motion_cmd.stage < disable_from_stage).to(base_reward.dtype)
    return base_reward * gate


def _stage_weight(
    motion_cmd: MotionCommand,
    dtype: torch.dtype,
    stage0_weight: float,
    stage1_weight: float,
    stage2_weight: float,
) -> torch.Tensor:
    stage = motion_cmd.stage
    return (
        (stage == 0).to(dtype) * stage0_weight
        + (stage == 1).to(dtype) * stage1_weight
        + (stage >= 2).to(dtype) * stage2_weight
    )


def stage_weighted_motion_global_anchor_position_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    stage_command_name: str = "motion",
    stage0_weight: float = 1.0,
    stage1_weight: float = 1.0,
    stage2_weight: float = 0.0,
) -> torch.Tensor:
    base_reward = motion_global_anchor_position_error_exp(env, command_name=command_name, std=std)
    motion_cmd: MotionCommand = env.command_manager.get_term(stage_command_name)
    return base_reward * _stage_weight(motion_cmd, base_reward.dtype, stage0_weight, stage1_weight, stage2_weight)


def stage_weighted_motion_global_anchor_orientation_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    stage_command_name: str = "motion",
    stage0_weight: float = 1.0,
    stage1_weight: float = 1.0,
    stage2_weight: float = 0.0,
) -> torch.Tensor:
    base_reward = motion_global_anchor_orientation_error_exp(env, command_name=command_name, std=std)
    motion_cmd: MotionCommand = env.command_manager.get_term(stage_command_name)
    return base_reward * _stage_weight(motion_cmd, base_reward.dtype, stage0_weight, stage1_weight, stage2_weight)


def stage_weighted_motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    body_names: list[str] | None = None,
    stage_command_name: str = "motion",
    stage0_weight: float = 1.0,
    stage1_weight: float = 1.0,
    stage2_weight: float = 1.0,
) -> torch.Tensor:
    base_reward = motion_relative_body_position_error_exp(env, command_name=command_name, std=std, body_names=body_names)
    motion_cmd: MotionCommand = env.command_manager.get_term(stage_command_name)
    return base_reward * _stage_weight(motion_cmd, base_reward.dtype, stage0_weight, stage1_weight, stage2_weight)


def stage_weighted_motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    body_names: list[str] | None = None,
    stage_command_name: str = "motion",
    stage0_weight: float = 1.0,
    stage1_weight: float = 1.0,
    stage2_weight: float = 1.0,
) -> torch.Tensor:
    base_reward = motion_relative_body_orientation_error_exp(
        env, command_name=command_name, std=std, body_names=body_names
    )
    motion_cmd: MotionCommand = env.command_manager.get_term(stage_command_name)
    return base_reward * _stage_weight(motion_cmd, base_reward.dtype, stage0_weight, stage1_weight, stage2_weight)


def stage_weighted_motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    body_names: list[str] | None = None,
    stage_command_name: str = "motion",
    stage0_weight: float = 1.0,
    stage1_weight: float = 0.0,
    stage2_weight: float = 0.0,
) -> torch.Tensor:
    base_reward = motion_global_body_linear_velocity_error_exp(
        env, command_name=command_name, std=std, body_names=body_names
    )
    motion_cmd: MotionCommand = env.command_manager.get_term(stage_command_name)
    return base_reward * _stage_weight(motion_cmd, base_reward.dtype, stage0_weight, stage1_weight, stage2_weight)


def stage_weighted_motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    body_names: list[str] | None = None,
    stage_command_name: str = "motion",
    stage0_weight: float = 1.0,
    stage1_weight: float = 0.0,
    stage2_weight: float = 0.0,
) -> torch.Tensor:
    base_reward = motion_global_body_angular_velocity_error_exp(
        env, command_name=command_name, std=std, body_names=body_names
    )
    motion_cmd: MotionCommand = env.command_manager.get_term(stage_command_name)
    return base_reward * _stage_weight(motion_cmd, base_reward.dtype, stage0_weight, stage1_weight, stage2_weight)


def stage_weighted_motion_joint_position_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    stage_command_name: str = "motion",
    stage0_weight: float = 1.0,
    stage1_weight: float = 1.0,
    stage2_weight: float = 1.0,
) -> torch.Tensor:
    base_reward = motion_joint_position_error_exp(env, command_name=command_name, std=std, asset_cfg=asset_cfg)
    motion_cmd: MotionCommand = env.command_manager.get_term(stage_command_name)
    return base_reward * _stage_weight(motion_cmd, base_reward.dtype, stage0_weight, stage1_weight, stage2_weight)


def stage_weighted_motion_joint_velocity_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    stage_command_name: str = "motion",
    stage0_weight: float = 1.0,
    stage1_weight: float = 1.0,
    stage2_weight: float = 0.0,
) -> torch.Tensor:
    base_reward = motion_joint_velocity_error_exp(env, command_name=command_name, std=std, asset_cfg=asset_cfg)
    motion_cmd: MotionCommand = env.command_manager.get_term(stage_command_name)
    return base_reward * _stage_weight(motion_cmd, base_reward.dtype, stage0_weight, stage1_weight, stage2_weight)


def stage_gated_track_lin_vel_axis_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    axis: int,
    std: float,
    stage_command_name: str = "motion",
    enabled_stage: int = 2,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Track a single linear velocity axis in body frame (0:x, 1:y), gated by stage."""
    motion_cmd: MotionCommand = env.command_manager.get_term(stage_command_name)
    gate = (motion_cmd.stage >= enabled_stage)
    cmd = env.command_manager.get_command(command_name)
    asset = env.scene[asset_cfg.name]
    vel = asset.data.root_lin_vel_b[:, axis]
    err = torch.square(cmd[:, axis] - vel)
    reward = torch.exp(-err / std**2)
    reward *= gate.to(reward.dtype)
    return reward

def stage_gated_track_lin_vel_axis_projected_body_exp(
    env,
    command_name: str,
    axis: int,
    std: float,
    stage_command_name: str = "motion",
    enabled_stage: int = 2,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Track one linear velocity axis in yaw-projected body frame, gated by motion stage.

    axis=0: horizontal forward direction after removing roll/pitch
    axis=1: horizontal lateral direction after removing roll/pitch
    """
    motion_cmd: MotionCommand = env.command_manager.get_term(stage_command_name)
    gate = motion_cmd.stage >= enabled_stage

    cmd = env.command_manager.get_command(command_name)
    asset = env.scene[asset_cfg.name]

    yaw_only_quat = yaw_quat(asset.data.root_quat_w)
    lin_vel_proj_b = quat_apply_inverse(yaw_only_quat, asset.data.root_lin_vel_w)

    err = torch.square(cmd[:, axis] - lin_vel_proj_b[:, axis])
    reward = torch.exp(-err / std**2)
    reward *= gate.to(reward.dtype)
    return reward

def stage_gated_y_command_joint_motion_reward(
    env,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    y_cmd_threshold: float = 0.004,
    max_value: float = 2.0,
    stage_command_name: str = "motion",
    enabled_stage: int = 2,
) -> torch.Tensor:
    """Reward moderate leg joint motion under lateral y command in stage2.

    This is a dense shaping reward:
    - active only in stage2
    - active only when |vy_cmd| is large enough
    - encourages selected leg joints to participate
    - does not require foot airtime
    """
    motion_cmd = env.command_manager.get_term(stage_command_name)
    stage_gate = (motion_cmd.stage >= enabled_stage).float()

    cmd = env.command_manager.get_command(command_name)
    y_cmd = cmd[:, 1]
    y_gate = (torch.abs(y_cmd) > y_cmd_threshold).float()

    asset = env.scene[asset_cfg.name]
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]

    # Mean absolute selected joint velocity.
    motion = torch.mean(torch.abs(joint_vel), dim=1)

    # Clamp to avoid rewarding violent motion.
    motion = torch.clamp(motion, max=max_value) / max_value

    return motion * stage_gate * y_gate

def stage_gated_track_ang_vel_z_projected_exp(
    env,
    command_name: str,
    std: float,
    stage_command_name: str = "motion",
    enabled_stage: int = 2,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Track yaw angular velocity around world z axis, gated by motion stage."""
    motion_cmd: MotionCommand = env.command_manager.get_term(stage_command_name)
    gate = motion_cmd.stage >= enabled_stage

    cmd = env.command_manager.get_command(command_name)
    asset = env.scene[asset_cfg.name]

    err = torch.square(cmd[:, 2] - asset.data.root_ang_vel_w[:, 2])
    reward = torch.exp(-err / std**2)
    reward *= gate.to(reward.dtype)
    return reward

def stage_gated_y_command_leg_lift_posture_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    joint_names: list[str],
    y_cmd_threshold: float,
    target_delta: float,
    std: float,
    stage_command_name: str = "motion",
    enabled_stage: int = 2,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Weakly shape lateral command into slight leg lift by thigh/calf offsets from default."""
    motion_cmd: MotionCommand = env.command_manager.get_term(stage_command_name)
    gate = (motion_cmd.stage >= enabled_stage)
    cmd = env.command_manager.get_command(command_name)
    y_active = (torch.abs(cmd[:, 1]) > y_cmd_threshold)
    asset = env.scene[asset_cfg.name]
    joint_ids, _ = asset.find_joints(joint_names)
    delta = asset.data.joint_pos[:, joint_ids] - asset.data.default_joint_pos[:, joint_ids]
    target = torch.full_like(delta, target_delta)
    err = torch.mean(torch.square(torch.abs(delta) - target), dim=1)
    reward = torch.exp(-err / std**2)
    reward *= (gate & y_active).to(reward.dtype)
    return reward


def stage_gated_rear_feet_air_time_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    y_cmd_threshold: float,
    threshold: float = 0.25,
    stage_command_name: str = "motion",
    enabled_stage: int = 2,
) -> torch.Tensor:
    """Very weak rear-foot air-time encouragement when lateral command is active."""
    motion_cmd: MotionCommand = env.command_manager.get_term(stage_command_name)
    gate = (motion_cmd.stage >= enabled_stage)
    cmd = env.command_manager.get_command(command_name)
    y_active = (torch.abs(cmd[:, 1]) > y_cmd_threshold)
    reward = feet_contact_time(env, sensor_cfg=sensor_cfg, threshold=threshold)
    reward *= (gate & y_active).to(reward.dtype)
    return reward


def stage_gated_joint_deviation_l1_penalty(
    env: ManagerBasedRLEnv,
    joint_names: list[str],
    command_name: str = "motion",
    enabled_stage: int = 2,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Stage-gated joint deviation penalty, useful for hip abduction constraint."""
    motion_cmd: MotionCommand = env.command_manager.get_term(command_name)
    gate = (motion_cmd.stage >= enabled_stage)
    asset = env.scene[asset_cfg.name]
    joint_ids, _ = asset.find_joints(joint_names)
    err = torch.mean(
        torch.abs(asset.data.joint_pos[:, joint_ids] - asset.data.default_joint_pos[:, joint_ids]),
        dim=1,
    )
    err *= gate.to(err.dtype)
    return err


def stage_gated_base_tilt_l2_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    enabled_stage: int = 2,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Stage-gated roll/pitch penalty using projected gravity x/y components."""
    motion_cmd: MotionCommand = env.command_manager.get_term(command_name)
    gate = (motion_cmd.stage >= enabled_stage)
    asset = env.scene[asset_cfg.name]
    tilt = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    tilt *= gate.to(tilt.dtype)
    return tilt


def stage_gated_multi_foot_air_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    max_air_feet: int = 2,
    command_name: str = "motion",
    enabled_stage: int = 2,
) -> torch.Tensor:
    """Penalize having too many feet in the air simultaneously during command stage."""
    motion_cmd: MotionCommand = env.command_manager.get_term(command_name)
    gate = (motion_cmd.stage >= enabled_stage)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    in_air = contact_sensor.compute_first_air(env.step_dt, env.physics_dt)[:, sensor_cfg.body_ids]
    num_air = torch.sum(in_air.to(torch.int32), dim=1)
    excess = torch.clamp(num_air - max_air_feet, min=0).to(torch.float32)
    excess *= gate.to(excess.dtype)
    return excess


def stage_gated_feet_contact_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    force_threshold: float = 1.0,
    stage_command_name: str = "motion",
    enabled_stage: int = 2,
) -> torch.Tensor:
    """Reward contact presence on selected feet, enabled only in/after a given stage."""
    motion_cmd: MotionCommand = env.command_manager.get_term(stage_command_name)
    gate = (motion_cmd.stage >= enabled_stage)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # Robust contact detection from force history.
    net_contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    is_contact = net_contact_forces.norm(dim=-1).max(dim=1)[0] > force_threshold
    reward = is_contact.to(torch.float32).mean(dim=1)
    reward *= gate.to(reward.dtype)
    return reward


def stage_gated_base_pitch_excess_l2_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    pitch_limit: float = 0.35,
    stage_command_name: str = "motion",
    enabled_stage: int = 2,
) -> torch.Tensor:
    """Penalize pitch only when |pitch| exceeds pitch_limit, enabled only in/after a given stage."""
    motion_cmd: MotionCommand = env.command_manager.get_term(stage_command_name)
    gate = (motion_cmd.stage >= enabled_stage)
    asset = env.scene[asset_cfg.name]
    # projected_gravity_b[:, 0] ~= -sin(pitch)
    pitch_abs = torch.abs(torch.asin(torch.clamp(-asset.data.projected_gravity_b[:, 0], -1.0, 1.0)))
    excess = torch.clamp(pitch_abs - pitch_limit, min=0.0)
    penalty = torch.square(excess)
    penalty *= gate.to(penalty.dtype)
    return penalty

def stage_gated_y_command_diagonal_leg_motion_reward(
    env,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    diag_a_joint_names: list[str],
    diag_b_joint_names: list[str],
    y_cmd_threshold: float = 0.004,
    max_value: float = 2.0,
    stage_command_name: str = "motion",
    enabled_stage: int = 2,
) -> torch.Tensor:
    """Reward both diagonal leg pairs participating under lateral y command.

    Uses min(diag_a_motion, diag_b_motion), so one-sided motion gets little reward.
    """
    motion_cmd = env.command_manager.get_term(stage_command_name)
    stage_gate = (motion_cmd.stage >= enabled_stage).float()

    cmd = env.command_manager.get_command(command_name)
    y_cmd = cmd[:, 1]
    y_gate = (torch.abs(y_cmd) > y_cmd_threshold).float()

    asset = env.scene[asset_cfg.name]

    diag_a_ids, _ = asset.find_joints(diag_a_joint_names)
    diag_b_ids, _ = asset.find_joints(diag_b_joint_names)

    vel_a = asset.data.joint_vel[:, diag_a_ids]
    vel_b = asset.data.joint_vel[:, diag_b_ids]

    motion_a = torch.mean(torch.abs(vel_a), dim=1)
    motion_b = torch.mean(torch.abs(vel_b), dim=1)

    motion_a = torch.clamp(motion_a, max=max_value) / max_value
    motion_b = torch.clamp(motion_b, max=max_value) / max_value

    reward = torch.minimum(motion_a, motion_b)

    return reward * stage_gate * y_gate

def stage_gated_y_command_diagonal_motion_balance_penalty(
    env,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    diag_a_joint_names: list[str],
    diag_b_joint_names: list[str],
    y_cmd_threshold: float = 0.004,
    max_value: float = 2.0,
    stage_command_name: str = "motion",
    enabled_stage: int = 2,
) -> torch.Tensor:
    """Penalize imbalance between two diagonal leg-pair motions under y command."""
    motion_cmd = env.command_manager.get_term(stage_command_name)
    stage_gate = (motion_cmd.stage >= enabled_stage).float()

    cmd = env.command_manager.get_command(command_name)
    y_cmd = cmd[:, 1]
    y_gate = (torch.abs(y_cmd) > y_cmd_threshold).float()

    asset = env.scene[asset_cfg.name]

    diag_a_ids, _ = asset.find_joints(diag_a_joint_names)
    diag_b_ids, _ = asset.find_joints(diag_b_joint_names)

    vel_a = asset.data.joint_vel[:, diag_a_ids]
    vel_b = asset.data.joint_vel[:, diag_b_ids]

    motion_a = torch.mean(torch.abs(vel_a), dim=1)
    motion_b = torch.mean(torch.abs(vel_b), dim=1)

    motion_a = torch.clamp(motion_a, max=max_value) / max_value
    motion_b = torch.clamp(motion_b, max=max_value) / max_value

    penalty = torch.abs(motion_a - motion_b)

    return penalty * stage_gate * y_gate

def stage_gated_base_lateral_edge_penalty(
    env,
    asset_cfg: SceneEntityCfg,
    edge_abs: float,
    soft_margin: float = 0.10,
    axis: int = 0,
    box_name: str = "box",
    stage_command_name: str = "motion",
    enabled_stage: int = 2,
) -> torch.Tensor:
    """Penalize base getting too close to platform edge in stage2.

    Uses position relative to the box center instead of absolute world position.

    axis:
      0 -> x relative to box
      1 -> y relative to box
    """
    motion_cmd = env.command_manager.get_term(stage_command_name)
    stage_gate = (motion_cmd.stage >= enabled_stage).float()

    robot = env.scene[asset_cfg.name]
    box = env.scene[box_name]

    base_rel = robot.data.root_pos_w[:, axis] - box.data.root_pos_w[:, axis]

    excess = torch.clamp(torch.abs(base_rel) - (edge_abs - soft_margin), min=0.0)
    penalty = excess * excess

    return penalty * stage_gate

def stage_gated_feet_lateral_edge_penalty(
    env,
    asset_cfg: SceneEntityCfg,
    edge_abs: float,
    soft_margin: float = 0.10,
    axis: int = 0,
    box_name: str = "box",
    stage_command_name: str = "motion",
    enabled_stage: int = 2,
) -> torch.Tensor:
    """Penalize selected feet/wheels getting too close to platform edge in stage2.

    Uses foot/body position relative to the box center.
    """
    motion_cmd = env.command_manager.get_term(stage_command_name)
    stage_gate = (motion_cmd.stage >= enabled_stage).float()

    robot = env.scene[asset_cfg.name]
    box = env.scene[box_name]

    body_rel = robot.data.body_pos_w[:, asset_cfg.body_ids, axis] - box.data.root_pos_w[:, axis].unsqueeze(-1)

    excess = torch.clamp(torch.abs(body_rel) - (edge_abs - soft_margin), min=0.0)
    penalty = torch.mean(excess * excess, dim=1)

    return penalty * stage_gate

def stage_gated_base_box_x_clearance_penalty(
    env,
    asset_cfg: SceneEntityCfg,
    box_name: str = "box",
    box_half_x: float = 0.525,
    min_clearance: float = 0.18,
    max_penalty: float = 1.0,
    stage_command_name: str = "motion",
    enabled_stage: int = 2,
) -> torch.Tensor:
    """Penalize base getting too close to the platform side in stage2.

    The platform is long along y. Lateral walking uses y command.
    Getting closer/farther to the platform is mainly x-direction motion.

    clearance = abs(base_x - box_x) - box_half_x

    If clearance < min_clearance, penalize.
    """
    motion_cmd = env.command_manager.get_term(stage_command_name)
    stage_gate = (motion_cmd.stage >= enabled_stage).float()

    robot = env.scene[asset_cfg.name]
    box = env.scene[box_name]

    base_x_rel = robot.data.root_pos_w[:, 0] - box.data.root_pos_w[:, 0]
    clearance = torch.abs(base_x_rel) - box_half_x

    excess = torch.clamp(min_clearance - clearance, min=0.0)
    penalty = torch.square(excess)

    if max_penalty is not None:
        penalty = torch.clamp(penalty, max=max_penalty)

    return penalty * stage_gate

def stage_gated_body_box_x_clearance_penalty(
    env,
    asset_cfg: SceneEntityCfg,
    box_name: str = "box",
    box_half_x: float = 0.525,
    min_clearance: float = 0.22,
    max_penalty: float = 1.0,
    stage_command_name: str = "motion",
    enabled_stage: int = 2,
) -> torch.Tensor:
    """Penalize selected robot bodies getting too close to the platform side in stage2.

    The platform is long along y. Lateral walking uses y command.
    Getting closer/farther to the platform is mainly x-direction motion.

    clearance = abs(body_x - box_x) - box_half_x

    If selected bodies are too close to the side wall, penalize.
    """
    motion_cmd = env.command_manager.get_term(stage_command_name)
    stage_gate = (motion_cmd.stage >= enabled_stage).float()

    robot = env.scene[asset_cfg.name]
    box = env.scene[box_name]

    body_x_rel = robot.data.body_pos_w[:, asset_cfg.body_ids, 0] - box.data.root_pos_w[:, 0].unsqueeze(-1)
    clearance = torch.abs(body_x_rel) - box_half_x

    excess = torch.clamp(min_clearance - clearance, min=0.0)
    penalty = torch.mean(torch.square(excess), dim=1)

    if max_penalty is not None:
        penalty = torch.clamp(penalty, max=max_penalty)

    return penalty * stage_gate
# 使用 motion anchor 的 yaw 局部坐标系。只约束局部 x 漂移，不约束局部 y，所以不会阻碍沿高台方向横移。
# 同时它只在 vx_cmd 很小时生效；如果你主动给 x 指令，它不会强行把机器人拉回去
def stage_gated_anchor_local_axis_drift_l2_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    motion_command_name: str = "motion",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    axis: int = 0,
    deadband: float = 0.10,
    command_axis: int = 0,
    command_deadzone: float = 0.025,
    max_penalty: float = 1.0,
    enabled_stage: int = 2,
) -> torch.Tensor:
    """Penalize robot anchor drift along one local axis of the motion anchor frame.

    This is different from world-x clearance penalty.

    It compares:
        robot anchor position - reference motion anchor position

    Then projects that difference into the yaw-only frame of the motion anchor.

    axis:
        0 -> local forward/backward drift relative to anchor yaw
        1 -> local lateral drift relative to anchor yaw

    For your stage2 lateral walking:
        use axis=0

    That means:
        - allow walking along local y
        - suppress drifting into / away from the platform along local x
        - avoid using raw world-x, which is wrong when the body/anchor has yaw rotation

    The penalty is disabled when the corresponding velocity command axis is active,
    so explicit x commands can still be learned.
    """
    motion_cmd: MotionCommand = env.command_manager.get_term(motion_command_name)
    stage_gate = motion_cmd.stage >= enabled_stage

    cmd = env.command_manager.get_command(command_name)
    command_inactive = torch.abs(cmd[:, command_axis]) < command_deadzone

    # Preserve the frozen baseline's actual pose at the instant ready opens.
    # This avoids pulling a valid real landing toward the reference trajectory.
    rel_pos_w = motion_cmd.robot_anchor_pos_w - motion_cmd.ready_anchor_pos_w

    # Use only the yaw of the ready snapshot.
    # This removes roll/pitch mixing and avoids raw world-frame x/y assumptions.
    anchor_yaw_quat = yaw_quat(motion_cmd.ready_anchor_quat_w)
    rel_pos_anchor_yaw = quat_apply_inverse(anchor_yaw_quat, rel_pos_w)

    drift = torch.abs(rel_pos_anchor_yaw[:, axis])
    excess = torch.clamp(drift - deadband, min=0.0)

    penalty = torch.square(excess)

    if max_penalty is not None:
        penalty = torch.clamp(penalty, max=max_penalty)

    penalty *= (stage_gate & command_inactive).to(penalty.dtype)
    return penalty

#vx 指令轮速奖励
def stage_gated_x_command_wheel_velocity_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    x_cmd_threshold: float = 0.05,
    target_scale: float = 18.0,
    std: float = 5.0,
    stage_command_name: str = "motion",
    enabled_stage: int = 2,
    direction: float = 1.0,
) -> torch.Tensor:
    """Reward wheel velocity participation under vx command.

    Purpose:
    - vx command should be realized by wheel rolling, not body twisting or leg splay.
    - Active only in stage2.
    - Active only when |vx_cmd| > x_cmd_threshold.

    Args:
        direction:
            Use 1.0 by default. If wheels spin opposite to expected direction in play,
            change to -1.0 in the env config.
    """
    motion_cmd: MotionCommand = env.command_manager.get_term(stage_command_name)
    stage_gate = motion_cmd.stage >= enabled_stage

    cmd = env.command_manager.get_command(command_name)
    vx_cmd = cmd[:, 0]
    x_gate = torch.abs(vx_cmd) > x_cmd_threshold

    asset = env.scene[asset_cfg.name]
    joint_ids = _get_joint_ids(asset_cfg)
    wheel_vel = asset.data.joint_vel[:, joint_ids]

    target = direction * vx_cmd.unsqueeze(-1) * target_scale

    err = torch.mean(torch.square(wheel_vel - target), dim=1)
    reward = torch.exp(-err / (std * std))

    reward *= (stage_gate & x_gate).to(reward.dtype)
    return reward

#yaw 指令左右轮差速奖励
def stage_gated_yaw_command_wheel_diff_velocity_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    left_wheel_joint_names: list[str],
    right_wheel_joint_names: list[str],
    yaw_cmd_threshold: float = 0.05,
    target_scale: float = 10.0,
    std: float = 5.0,
    stage_command_name: str = "motion",
    enabled_stage: int = 2,
    direction: float = 1.0,
) -> torch.Tensor:
    """Reward left-right wheel differential velocity under yaw command.

    Purpose:
    - yaw command should be realized by left-right wheel differential motion,
      not by body twisting or leg splay.
    - Active only in stage2.
    - Active only when |yaw_cmd| > yaw_cmd_threshold.

    Args:
        direction:
            Use 1.0 by default. If yaw response direction is opposite in play,
            change to -1.0 in the env config.
    """
    motion_cmd: MotionCommand = env.command_manager.get_term(stage_command_name)
    stage_gate = motion_cmd.stage >= enabled_stage

    cmd = env.command_manager.get_command(command_name)
    yaw_cmd = cmd[:, 2]
    yaw_gate = torch.abs(yaw_cmd) > yaw_cmd_threshold

    asset = env.scene[asset_cfg.name]

    left_ids, _ = asset.find_joints(left_wheel_joint_names)
    right_ids, _ = asset.find_joints(right_wheel_joint_names)

    left_vel = torch.mean(asset.data.joint_vel[:, left_ids], dim=1)
    right_vel = torch.mean(asset.data.joint_vel[:, right_ids], dim=1)

    actual_diff = left_vel - right_vel
    target_diff = direction * yaw_cmd * target_scale

    err = torch.square(actual_diff - target_diff)
    reward = torch.exp(-err / (std * std))

    reward *= (stage_gate & yaw_gate).to(reward.dtype)
    return reward

def stage_gated_wheel_front_rear_balance_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    front_wheel_joint_names: list[str],
    rear_wheel_joint_names: list[str],
    cmd_threshold: float = 0.05,
    max_value: float = 20.0,
    stage_command_name: str = "motion",
    enabled_stage: int = 2,
) -> torch.Tensor:
    """Penalize front/rear wheel velocity imbalance under vx or yaw command.

    This prevents the policy from moving only front legs/wheels while rear wheels remain passive.
    """
    motion_cmd: MotionCommand = env.command_manager.get_term(stage_command_name)
    stage_gate = motion_cmd.stage >= enabled_stage

    cmd = env.command_manager.get_command(command_name)
    cmd_active = (torch.abs(cmd[:, 0]) > cmd_threshold) | (torch.abs(cmd[:, 2]) > cmd_threshold)

    asset = env.scene[asset_cfg.name]

    front_ids, _ = asset.find_joints(front_wheel_joint_names)
    rear_ids, _ = asset.find_joints(rear_wheel_joint_names)

    front_motion = torch.mean(torch.abs(asset.data.joint_vel[:, front_ids]), dim=1)
    rear_motion = torch.mean(torch.abs(asset.data.joint_vel[:, rear_ids]), dim=1)

    penalty = torch.abs(front_motion - rear_motion)
    penalty = torch.clamp(penalty, max=max_value) / max_value

    return penalty * (stage_gate & cmd_active).to(penalty.dtype)

def stage_gated_rear_wheel_motion_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    rear_wheel_joint_names: list[str],
    cmd_threshold: float = 0.05,
    max_value: float = 18.0,
    stage_command_name: str = "motion",
    enabled_stage: int = 2,
) -> torch.Tensor:
    """Reward rear wheel participation under vx or yaw command."""
    motion_cmd: MotionCommand = env.command_manager.get_term(stage_command_name)
    stage_gate = motion_cmd.stage >= enabled_stage

    cmd = env.command_manager.get_command(command_name)
    cmd_active = (torch.abs(cmd[:, 0]) > cmd_threshold) | (torch.abs(cmd[:, 2]) > cmd_threshold)

    asset = env.scene[asset_cfg.name]
    rear_ids, _ = asset.find_joints(rear_wheel_joint_names)

    rear_motion = torch.mean(torch.abs(asset.data.joint_vel[:, rear_ids]), dim=1)
    reward = torch.clamp(rear_motion, max=max_value) / max_value

    return reward * (stage_gate & cmd_active).to(reward.dtype)


#world-frame x 速度跟踪
def stage_gated_track_lin_vel_axis_world_exp(
    env,
    command_name: str,
    axis: int,
    std: float,
    stage_command_name: str = "motion",
    enabled_stage: int = 2,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Track one linear velocity axis in world frame, gated by motion stage.

    axis=0: world x velocity
    axis=1: world y velocity
    axis=2: world z velocity, normally not used here.
    """
    motion_cmd: MotionCommand = env.command_manager.get_term(stage_command_name)
    gate = motion_cmd.stage >= enabled_stage

    cmd = env.command_manager.get_command(command_name)
    asset = env.scene[asset_cfg.name]

    vel_w = asset.data.root_lin_vel_w[:, axis]
    err = torch.square(cmd[:, axis] - vel_w)

    reward = torch.exp(-err / (std * std))
    reward *= gate.to(reward.dtype)
    return reward
# 四轮 vx/yaw 目标轮速模式奖励
def stage_gated_command_wheel_velocity_pattern_reward(
    env,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    wheel_joint_names: list[str],
    left_wheel_joint_names: list[str],
    right_wheel_joint_names: list[str],
    x_cmd_threshold: float = 0.05,
    yaw_cmd_threshold: float = 0.05,
    x_target_scale: float = 18.0,
    yaw_target_scale: float = 10.0,
    std: float = 5.0,
    x_direction: float = 1.0,
    yaw_direction: float = 1.0,
    max_target_abs: float = 18.0,
    stage_command_name: str = "motion",
    enabled_stage: int = 2,
) -> torch.Tensor:
    """Reward a commanded wheel velocity pattern for vx and yaw.

    vx:
      all wheels should roll in the same direction.

    yaw:
      left and right wheels should roll in opposite directions.

    This avoids the local optimum where only front legs/front wheels respond.
    """
    motion_cmd: MotionCommand = env.command_manager.get_term(stage_command_name)
    stage_gate = motion_cmd.stage >= enabled_stage

    cmd = env.command_manager.get_command(command_name)
    vx_cmd = cmd[:, 0]
    yaw_cmd = cmd[:, 2]

    x_active = torch.abs(vx_cmd) > x_cmd_threshold
    yaw_active = torch.abs(yaw_cmd) > yaw_cmd_threshold
    active = x_active | yaw_active

    asset = env.scene[asset_cfg.name]

    wheel_ids, wheel_names = asset.find_joints(wheel_joint_names)
    left_ids, _ = asset.find_joints(left_wheel_joint_names)
    right_ids, _ = asset.find_joints(right_wheel_joint_names)

    # Build side signs for the wheel order returned by wheel_ids.
    side_sign = torch.zeros((len(wheel_ids),), device=asset.data.joint_vel.device, dtype=asset.data.joint_vel.dtype)

    left_set = set(left_ids)
    right_set = set(right_ids)
    for i, jid in enumerate(wheel_ids):
        if jid in left_set:
            side_sign[i] = 1.0
        elif jid in right_set:
            side_sign[i] = -1.0
        else:
            side_sign[i] = 0.0

    vx_target = x_direction * x_target_scale * vx_cmd.unsqueeze(-1)
    yaw_target = yaw_direction * yaw_target_scale * yaw_cmd.unsqueeze(-1) * side_sign.unsqueeze(0)

    target = vx_target + yaw_target
    target = torch.clamp(target, min=-max_target_abs, max=max_target_abs)

    wheel_vel = asset.data.joint_vel[:, wheel_ids]

    err = torch.mean(torch.square(wheel_vel - target), dim=1)
    reward = torch.exp(-err / (std * std))

    reward *= (stage_gate & active).to(reward.dtype)
    return reward
# 弱前腿过伸惩罚
def stage_gated_command_front_leg_extension_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    joint_names: list[str],
    cmd_threshold: float = 0.12,
    deadband: float = 0.40,
    stage_command_name: str = "motion",
    enabled_stage: int = 2,
) -> torch.Tensor:
    """Weakly penalize excessive front leg extension under vx/yaw command.

    This should not suppress normal lateral stepping.
    """
    motion_cmd: MotionCommand = env.command_manager.get_term(stage_command_name)
    stage_gate = motion_cmd.stage >= enabled_stage

    cmd = env.command_manager.get_command(command_name)
    cmd_active = (torch.abs(cmd[:, 0]) > cmd_threshold) | (torch.abs(cmd[:, 2]) > cmd_threshold)

    asset = env.scene[asset_cfg.name]
    joint_ids, _ = asset.find_joints(joint_names)

    deviation = torch.abs(asset.data.joint_pos[:, joint_ids] - asset.data.default_joint_pos[:, joint_ids])
    excess = torch.clamp(deviation - deadband, min=0.0)

    penalty = torch.mean(torch.square(excess), dim=1)
    penalty *= (stage_gate & cmd_active).to(penalty.dtype)
    return penalty
# 前轮不要靠近高台边缘太多
def stage_gated_front_wheel_platform_x_margin_penalty(
    env,
    asset_cfg: SceneEntityCfg,
    box_name: str = "box",
    box_half_x: float = 0.525,
    soft_margin: float = 0.08,
    command_name: str = "base_velocity",
    command_axis: int = 0,
    command_deadzone: float = 0.08,
    stage_command_name: str = "motion",
    enabled_stage: int = 2,
) -> torch.Tensor:
    """Penalize front wheels getting too close to platform x edge.

    Active only when |vx_cmd| is small, so the user can still command vx
    to correct distance from the platform edge.
    """
    motion_cmd: MotionCommand = env.command_manager.get_term(stage_command_name)
    stage_gate = motion_cmd.stage >= enabled_stage

    cmd = env.command_manager.get_command(command_name)
    no_x_cmd = torch.abs(cmd[:, command_axis]) < command_deadzone

    robot = env.scene[asset_cfg.name]
    box = env.scene[box_name]

    front_x_rel = robot.data.body_pos_w[:, asset_cfg.body_ids, 0] - box.data.root_pos_w[:, 0].unsqueeze(-1)

    limit = box_half_x - soft_margin
    excess = torch.clamp(torch.abs(front_x_rel) - limit, min=0.0)
    penalty = torch.mean(torch.square(excess), dim=1)

    penalty *= (stage_gate & no_x_cmd).to(penalty.dtype)
    return penalty


def residual_action_l2_penalty(
    env: ManagerBasedRLEnv,
    action_name: str,
    motion_command_name: str = "motion",
    only_ready: bool | None = None,
) -> torch.Tensor:
    """Penalize residual action magnitude before or after the automatic ready gate."""
    action_term = env.action_manager.get_term(action_name)
    residual = action_term.residual_raw_actions
    penalty = torch.mean(torch.square(residual), dim=1)

    motion_cmd: MotionCommand = env.command_manager.get_term(motion_command_name)
    ready = motion_cmd.ready
    if only_ready is True:
        penalty *= ready.to(penalty.dtype)
    elif only_ready is False:
        penalty *= (~ready).to(penalty.dtype)
    return penalty


def residual_action_rate_l2_penalty(
    env: ManagerBasedRLEnv,
    action_name: str,
    motion_command_name: str = "motion",
) -> torch.Tensor:
    """Penalize residual action jumps after the automatic ready gate opens."""
    action_term = env.action_manager.get_term(action_name)
    penalty = torch.mean(torch.square(action_term.residual_raw_actions - action_term.prev_residual_raw_actions), dim=1)
    motion_cmd: MotionCommand = env.command_manager.get_term(motion_command_name)
    penalty *= motion_cmd.ready.to(penalty.dtype)
    return penalty


def ready_gated_joint_acc_l2(
    env: ManagerBasedRLEnv,
    motion_command_name: str = "motion",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    penalty = il_mdp.joint_acc_l2(env, asset_cfg=asset_cfg)
    motion_cmd: MotionCommand = env.command_manager.get_term(motion_command_name)
    return penalty * motion_cmd.ready.to(penalty.dtype)


def ready_gated_joint_torques_l2(
    env: ManagerBasedRLEnv,
    motion_command_name: str = "motion",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    penalty = il_mdp.joint_torques_l2(env, asset_cfg=asset_cfg)
    motion_cmd: MotionCommand = env.command_manager.get_term(motion_command_name)
    return penalty * motion_cmd.ready.to(penalty.dtype)


def ready_gated_joint_pos_limits(
    env: ManagerBasedRLEnv,
    motion_command_name: str = "motion",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    penalty = il_mdp.joint_pos_limits(env, asset_cfg=asset_cfg)
    motion_cmd: MotionCommand = env.command_manager.get_term(motion_command_name)
    return penalty * motion_cmd.ready.to(penalty.dtype)


def ready_gated_motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    body_names: list[str] | None = None,
    motion_command_name: str = "motion",
) -> torch.Tensor:
    reward = motion_relative_body_position_error_exp(env, command_name=command_name, std=std, body_names=body_names)
    motion_cmd: MotionCommand = env.command_manager.get_term(motion_command_name)
    return reward * motion_cmd.ready.to(reward.dtype)


def ready_gated_motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    body_names: list[str] | None = None,
    motion_command_name: str = "motion",
) -> torch.Tensor:
    reward = motion_relative_body_orientation_error_exp(
        env, command_name=command_name, std=std, body_names=body_names
    )
    motion_cmd: MotionCommand = env.command_manager.get_term(motion_command_name)
    return reward * motion_cmd.ready.to(reward.dtype)


def ready_gated_motion_joint_position_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    motion_command_name: str = "motion",
) -> torch.Tensor:
    reward = motion_joint_position_error_exp(env, command_name=command_name, std=std, asset_cfg=asset_cfg)
    motion_cmd: MotionCommand = env.command_manager.get_term(motion_command_name)
    return reward * motion_cmd.ready.to(reward.dtype)


def ready_gated_motion_joint_position_deviation_l2_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    deadband: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_penalty: float | None = None,
    motion_command_name: str = "motion",
) -> torch.Tensor:
    """Penalize large joint deviation from the final hitch posture after ready."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    joint_ids = _get_joint_ids(asset_cfg)
    error = torch.abs(command.joint_pos[:, joint_ids] - command.robot_joint_pos[:, joint_ids])
    penalty = torch.mean(torch.square(torch.clamp(error - deadband, min=0.0)), dim=-1)
    if max_penalty is not None:
        penalty = torch.clamp(penalty, max=max_penalty)
    motion_cmd: MotionCommand = env.command_manager.get_term(motion_command_name)
    return penalty * motion_cmd.ready.to(penalty.dtype)


def ready_gated_motion_joint_velocity_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    motion_command_name: str = "motion",
) -> torch.Tensor:
    reward = motion_joint_velocity_error_exp(env, command_name=command_name, std=std, asset_cfg=asset_cfg)
    motion_cmd: MotionCommand = env.command_manager.get_term(motion_command_name)
    return reward * motion_cmd.ready.to(reward.dtype)


def ready_gated_leg_pair_deviation_balance_l2_penalty(
    env: ManagerBasedRLEnv,
    left_joint_names: list[str],
    right_joint_names: list[str],
    command_name: str = "motion",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    deadband: float = 0.25,
    max_penalty: float | None = 1.0,
) -> torch.Tensor:
    """Weakly penalize extreme left/right imbalance while allowing alternating steps."""
    if len(left_joint_names) != len(right_joint_names):
        raise ValueError("left_joint_names and right_joint_names must have the same length.")

    motion_cmd: MotionCommand = env.command_manager.get_term(command_name)
    asset = env.scene[asset_cfg.name]
    left_ids, _ = asset.find_joints(left_joint_names, preserve_order=True)
    right_ids, _ = asset.find_joints(right_joint_names, preserve_order=True)

    left_deviation = torch.abs(asset.data.joint_pos[:, left_ids] - motion_cmd.joint_pos[:, left_ids])
    right_deviation = torch.abs(asset.data.joint_pos[:, right_ids] - motion_cmd.joint_pos[:, right_ids])
    imbalance = torch.abs(left_deviation - right_deviation)
    penalty = torch.mean(torch.square(torch.clamp(imbalance - deadband, min=0.0)), dim=-1)
    if max_penalty is not None:
        penalty = torch.clamp(penalty, max=max_penalty)
    return penalty * motion_cmd.ready.to(penalty.dtype)


def ready_gated_track_lin_vel_axis_projected_body_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    axis: int,
    std: float,
    motion_command_name: str = "motion",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Track horizontal body-frame velocity only after boarding is ready."""
    motion_cmd: MotionCommand = env.command_manager.get_term(motion_command_name)
    cmd = env.command_manager.get_command(command_name)
    asset = env.scene[asset_cfg.name]
    yaw_only_quat = yaw_quat(asset.data.root_quat_w)
    lin_vel_proj_b = quat_apply_inverse(yaw_only_quat, asset.data.root_lin_vel_w)
    reward = torch.exp(-torch.square(cmd[:, axis] - lin_vel_proj_b[:, axis]) / std**2)
    return reward * motion_cmd.ready.to(reward.dtype)


def ready_gated_track_lin_vel_y_exp_with_heading_stability(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    command_threshold: float,
    yaw_tolerance: float,
    yaw_std: float,
    motion_command_name: str = "motion",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Track lateral velocity while preserving the ready-time heading."""
    motion_cmd: MotionCommand = env.command_manager.get_term(motion_command_name)
    command_y = env.command_manager.get_command(command_name)[:, 1]
    asset = env.scene[asset_cfg.name]
    yaw_only_quat = yaw_quat(asset.data.root_quat_w)
    velocity_y = quat_apply_inverse(yaw_only_quat, asset.data.root_lin_vel_w)[:, 1]

    reward = torch.exp(-torch.square(command_y - velocity_y) / std**2)
    yaw_error = quat_error_magnitude(
        yaw_quat(motion_cmd.ready_anchor_quat_w),
        yaw_quat(motion_cmd.robot_anchor_quat_w),
    )
    yaw_excess = torch.clamp(yaw_error - yaw_tolerance, min=0.0)
    reward *= torch.exp(-torch.square(yaw_excess) / yaw_std**2)

    active = motion_cmd.ready & (torch.abs(command_y) >= command_threshold)
    return reward * active.to(reward.dtype)


def ready_gated_track_ang_vel_z_projected_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    motion_command_name: str = "motion",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Track world-z yaw rate only after boarding is ready."""
    motion_cmd: MotionCommand = env.command_manager.get_term(motion_command_name)
    cmd = env.command_manager.get_command(command_name)
    asset = env.scene[asset_cfg.name]
    reward = torch.exp(-torch.square(cmd[:, 2] - asset.data.root_ang_vel_w[:, 2]) / std**2)
    return reward * motion_cmd.ready.to(reward.dtype)


def ready_gated_anchor_local_axis_drift_l2_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    motion_command_name: str = "motion",
    axis: int = 0,
    deadband: float = 0.04,
    scale: float | None = None,
    command_axis: int = 0,
    command_deadzone: float = 0.02,
    max_penalty: float | None = 1.0,
) -> torch.Tensor:
    """Preserve ready-time local x while allowing lateral translation and yaw."""
    motion_cmd: MotionCommand = env.command_manager.get_term(motion_command_name)
    command = env.command_manager.get_command(command_name)
    command_inactive = torch.abs(command[:, command_axis]) < command_deadzone

    rel_pos_w = motion_cmd.robot_anchor_pos_w - motion_cmd.ready_anchor_pos_w
    ready_yaw_quat = yaw_quat(motion_cmd.ready_anchor_quat_w)
    rel_pos_ready_yaw = quat_apply_inverse(ready_yaw_quat, rel_pos_w)
    excess = torch.clamp(torch.abs(rel_pos_ready_yaw[:, axis]) - deadband, min=0.0)
    if scale is not None:
        excess = excess / max(scale, 1.0e-6)
    penalty = torch.square(excess)
    if max_penalty is not None:
        penalty = torch.clamp(penalty, max=max_penalty)
    gate = motion_cmd.ready & command_inactive
    return penalty * gate.to(penalty.dtype)


def ready_gated_y_command_local_x_velocity_l2_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float,
    x_command_deadzone: float,
    velocity_deadband: float,
    velocity_scale: float,
    max_penalty: float | None = 4.0,
    motion_command_name: str = "motion",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize x velocity in the ready yaw frame while executing lateral y commands."""
    motion_cmd: MotionCommand = env.command_manager.get_term(motion_command_name)
    command = env.command_manager.get_command(command_name)
    y_active = torch.abs(command[:, 1]) > command_threshold
    x_inactive = torch.abs(command[:, 0]) < x_command_deadzone

    asset = env.scene[asset_cfg.name]
    ready_yaw_quat = yaw_quat(motion_cmd.ready_anchor_quat_w)
    lin_vel_ready_yaw = quat_apply_inverse(ready_yaw_quat, asset.data.root_lin_vel_w)
    excess = torch.clamp(torch.abs(lin_vel_ready_yaw[:, 0]) - velocity_deadband, min=0.0)
    penalty = torch.square(excess / max(velocity_scale, 1.0e-6))
    if max_penalty is not None:
        penalty = torch.clamp(penalty, max=max_penalty)

    gate = motion_cmd.ready & y_active & x_inactive
    return penalty * gate.to(penalty.dtype)


def ready_gated_multi_foot_air_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    max_air_feet: int = 2,
    motion_command_name: str = "motion",
) -> torch.Tensor:
    """Penalize lifting more than the allowed number of feet after ready."""
    motion_cmd: MotionCommand = env.command_manager.get_term(motion_command_name)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    in_air = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids] > 0.0
    num_air = torch.sum(in_air.to(torch.int32), dim=1)
    penalty = torch.clamp(num_air - max_air_feet, min=0).to(torch.float32)
    return penalty * motion_cmd.ready.to(penalty.dtype)


def ready_gated_feet_contact_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    force_threshold: float = 1.0,
    motion_command_name: str = "motion",
) -> torch.Tensor:
    """Reward contact presence on selected feet only after boarding is ready."""
    motion_cmd: MotionCommand = env.command_manager.get_term(motion_command_name)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    is_contact = net_contact_forces.norm(dim=-1).max(dim=1)[0] > force_threshold
    reward = is_contact.to(torch.float32).mean(dim=1)
    return reward * motion_cmd.ready.to(reward.dtype)


def ready_gated_y_command_yaw_drift_l2_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float,
    yaw_command_deadzone: float,
    deadband: float,
    max_penalty: float | None = 1.0,
    motion_command_name: str = "motion",
) -> torch.Tensor:
    """Preserve ready-time heading during lateral motion without blocking commanded yaw."""
    motion_cmd: MotionCommand = env.command_manager.get_term(motion_command_name)
    command = env.command_manager.get_command(command_name)
    y_active = torch.abs(command[:, 1]) > command_threshold
    yaw_inactive = torch.abs(command[:, 2]) < yaw_command_deadzone

    ready_yaw_quat = yaw_quat(motion_cmd.ready_anchor_quat_w)
    current_yaw_quat = yaw_quat(motion_cmd.robot_anchor_quat_w)
    yaw_error = quat_error_magnitude(ready_yaw_quat, current_yaw_quat)
    penalty = torch.square(torch.clamp(yaw_error - deadband, min=0.0))
    if max_penalty is not None:
        penalty = torch.clamp(penalty, max=max_penalty)
    gate = motion_cmd.ready & y_active & yaw_inactive
    return penalty * gate.to(penalty.dtype)


def ready_gated_roll_error_l2_penalty(
    env: ManagerBasedRLEnv,
    deadband: float,
    max_penalty: float | None = 1.0,
    motion_command_name: str = "motion",
) -> torch.Tensor:
    """Penalize side tilt relative to the posture captured when ready begins."""
    motion_cmd: MotionCommand = env.command_manager.get_term(motion_command_name)
    relative_quat = quat_mul(quat_inv(motion_cmd.ready_anchor_quat_w), motion_cmd.robot_anchor_quat_w)
    w, x, y, z = relative_quat.unbind(dim=-1)
    roll_error = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    penalty = torch.square(torch.clamp(torch.abs(roll_error) - deadband, min=0.0))
    if max_penalty is not None:
        penalty = torch.clamp(penalty, max=max_penalty)
    return penalty * motion_cmd.ready.to(penalty.dtype)


def ready_gated_y_command_diagonal_feet_air_time_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    command_threshold: float,
    threshold: float,
    max_air_time: float,
    stance_air_time_std: float,
    single_swing_scale: float,
    motion_command_name: str = "motion",
) -> torch.Tensor:
    """Encourage swing initiation, with a larger score for diagonal foot pairs."""
    motion_cmd: MotionCommand = env.command_manager.get_term(motion_command_name)
    command = env.command_manager.get_command(command_name)
    y_active = torch.abs(command[:, 1]) > command_threshold

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    current_air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]

    cache_attr = f"_diagonal_foot_indices_{sensor_cfg.name}"
    foot_indices = getattr(env, cache_attr, None)
    if foot_indices is None:
        asset = env.scene["robot"]
        sensor_body_ids = torch.as_tensor(sensor_cfg.body_ids, device=env.device, dtype=torch.long)
        foot_indices = {}
        for key, pattern in {
            "fl": "FL_foot.*",
            "fr": "FR_foot.*",
            "rl": "RL_foot.*",
            "rr": "RR_foot.*",
        }.items():
            body_ids, _ = asset.find_bodies(pattern)
            body_ids = torch.as_tensor(body_ids, device=env.device, dtype=torch.long)
            match = torch.nonzero(torch.isin(sensor_body_ids, body_ids), as_tuple=False).flatten()
            foot_indices[key] = int(match[0].item()) if len(match) > 0 else None
        setattr(env, cache_attr, foot_indices)

    if any(index is None for index in foot_indices.values()):
        return torch.zeros(env.num_envs, device=env.device, dtype=current_air_time.dtype)

    stance_std = max(stance_air_time_std, 1.0e-6)
    useful_air_time = {
        key: (
            torch.clamp(
                current_air_time[:, index] - threshold,
                min=0.0,
                max=max_air_time - threshold,
            )
            * torch.exp(
                -torch.square(torch.clamp(current_air_time[:, index] - max_air_time, min=0.0))
                / stance_std**2
            )
        )
        for key, index in foot_indices.items()
    }
    fl_rr_score = torch.minimum(useful_air_time["fl"], useful_air_time["rr"]) * torch.exp(
        -(torch.square(useful_air_time["fr"]) + torch.square(useful_air_time["rl"])) / stance_std**2
    )
    fr_rl_score = torch.minimum(useful_air_time["fr"], useful_air_time["rl"]) * torch.exp(
        -(torch.square(useful_air_time["fl"]) + torch.square(useful_air_time["rr"])) / stance_std**2
    )
    single_swing_score = single_swing_scale * torch.maximum(
        torch.maximum(useful_air_time["fl"], useful_air_time["fr"]),
        torch.maximum(useful_air_time["rl"], useful_air_time["rr"]),
    )
    reward = torch.maximum(torch.maximum(fl_rr_score, fr_rl_score), single_swing_score)
    gate = motion_cmd.ready & y_active
    return reward * gate.to(reward.dtype)


def ready_gated_y_command_airborne_foot_clearance_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    command_threshold: float,
    min_air_time: float,
    min_clearance: float,
    target_clearance: float,
    motion_command_name: str = "motion",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward airborne feet clearing the stance-foot height during lateral y commands."""
    motion_cmd: MotionCommand = env.command_manager.get_term(motion_command_name)
    command = env.command_manager.get_command(command_name)
    y_active = torch.abs(command[:, 1]) > command_threshold

    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    current_air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]

    cache_attr = f"_airborne_clearance_indices_{sensor_cfg.name}"
    indices = getattr(env, cache_attr, None)
    if indices is None:
        sensor_body_ids = torch.as_tensor(sensor_cfg.body_ids, device=env.device, dtype=torch.long)
        sensor_indices = []
        asset_body_indices = []
        for pattern in ("FL_foot.*", "FR_foot.*", "RL_foot.*", "RR_foot.*"):
            body_ids, _ = asset.find_bodies(pattern)
            body_ids = torch.as_tensor(body_ids, device=env.device, dtype=torch.long)
            match = torch.nonzero(torch.isin(sensor_body_ids, body_ids), as_tuple=False).flatten()
            if len(body_ids) == 0 or len(match) == 0:
                continue
            sensor_indices.append(int(match[0].item()))
            asset_body_indices.append(int(body_ids[0].item()))
        if len(sensor_indices) == 0:
            indices = None
        else:
            indices = (
                torch.as_tensor(sensor_indices, device=env.device, dtype=torch.long),
                torch.as_tensor(asset_body_indices, device=env.device, dtype=torch.long),
            )
        setattr(env, cache_attr, indices)

    if indices is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=current_air_time.dtype)

    sensor_indices, asset_body_indices = indices
    air_time = current_air_time[:, sensor_indices]
    foot_z = asset.data.body_pos_w[:, asset_body_indices, 2]
    in_air = air_time > min_air_time

    large_height = torch.full_like(foot_z, 1.0e6)
    stance_candidates = torch.where(in_air, large_height, foot_z)
    stance_z = torch.min(stance_candidates, dim=1).values
    all_air = torch.all(in_air, dim=1)
    stance_z = torch.where(all_air, torch.min(foot_z, dim=1).values, stance_z)

    clearance = torch.clamp(foot_z - stance_z.unsqueeze(1) - min_clearance, min=0.0)
    clearance_scale = max(target_clearance - min_clearance, 1.0e-6)
    clearance_score = torch.clamp(clearance / clearance_scale, max=1.0)
    reward = torch.clamp(torch.sum(clearance_score * in_air.to(clearance_score.dtype), dim=1) / 2.0, max=1.0)

    gate = motion_cmd.ready & y_active
    return reward * gate.to(reward.dtype)


def ready_gated_y_command_foot_up_velocity_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    command_threshold: float,
    min_up_velocity: float,
    target_up_velocity: float,
    motion_command_name: str = "motion",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward upward foot velocity under lateral y commands before airtime appears."""
    motion_cmd: MotionCommand = env.command_manager.get_term(motion_command_name)
    command = env.command_manager.get_command(command_name)
    y_active = torch.abs(command[:, 1]) > command_threshold

    asset = env.scene[asset_cfg.name]
    cache_attr = f"_foot_up_velocity_body_indices_{sensor_cfg.name}"
    body_indices = getattr(env, cache_attr, None)
    if body_indices is None:
        sensor_body_ids = torch.as_tensor(sensor_cfg.body_ids, device=env.device, dtype=torch.long)
        body_index_list = []
        for pattern in ("FL_foot.*", "FR_foot.*", "RL_foot.*", "RR_foot.*"):
            body_ids, _ = asset.find_bodies(pattern)
            body_ids = torch.as_tensor(body_ids, device=env.device, dtype=torch.long)
            match = torch.nonzero(torch.isin(sensor_body_ids, body_ids), as_tuple=False).flatten()
            if len(body_ids) == 0 or len(match) == 0:
                continue
            body_index_list.append(int(body_ids[0].item()))
        body_indices = (
            torch.as_tensor(body_index_list, device=env.device, dtype=torch.long)
            if len(body_index_list) > 0
            else None
        )
        setattr(env, cache_attr, body_indices)

    if body_indices is None:
        return torch.zeros(env.num_envs, device=env.device)

    up_velocity = torch.clamp(asset.data.body_lin_vel_w[:, body_indices, 2] - min_up_velocity, min=0.0)
    velocity_scale = max(target_up_velocity - min_up_velocity, 1.0e-6)
    up_score = torch.clamp(up_velocity / velocity_scale, max=1.0)
    top2_score = torch.topk(up_score, k=min(2, up_score.shape[1]), dim=1).values.mean(dim=1)

    gate = motion_cmd.ready & y_active
    return top2_score * gate.to(top2_score.dtype)


def ready_gated_normalized_lin_vel_axis_tracking(
    env: ManagerBasedRLEnv,
    command_name: str,
    axis: int,
    command_threshold: float,
    motion_command_name: str = "motion",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward signed command progress, with zero score for remaining stationary."""
    motion_cmd: MotionCommand = env.command_manager.get_term(motion_command_name)
    command = env.command_manager.get_command(command_name)[:, axis]
    asset = env.scene[asset_cfg.name]
    yaw_only_quat = yaw_quat(asset.data.root_quat_w)
    velocity = quat_apply_inverse(yaw_only_quat, asset.data.root_lin_vel_w)[:, axis]
    return _ready_gated_normalized_tracking_score(
        command,
        velocity,
        motion_cmd.ready,
        command_threshold,
    )


def ready_gated_normalized_yaw_rate_tracking(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float,
    motion_command_name: str = "motion",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward signed yaw-rate progress, with zero score for remaining stationary."""
    motion_cmd: MotionCommand = env.command_manager.get_term(motion_command_name)
    command = env.command_manager.get_command(command_name)[:, 2]
    asset = env.scene[asset_cfg.name]
    velocity = asset.data.root_ang_vel_w[:, 2]
    return _ready_gated_normalized_tracking_score(
        command,
        velocity,
        motion_cmd.ready,
        command_threshold,
    )


def _ready_gated_normalized_tracking_score(
    command: torch.Tensor,
    velocity: torch.Tensor,
    ready: torch.Tensor,
    command_threshold: float,
) -> torch.Tensor:
    """Return -1..1 tracking progress relative to doing nothing.

    Exact tracking scores 1, standing still under a non-zero command scores 0,
    and moving in the wrong direction scores below 0.
    """
    active = ready & (torch.abs(command) >= command_threshold)
    normalized_error = torch.abs(command - velocity) / torch.abs(command).clamp_min(command_threshold)
    score = torch.clamp(1.0 - normalized_error, min=-1.0, max=1.0)
    return score * active.to(score.dtype)
