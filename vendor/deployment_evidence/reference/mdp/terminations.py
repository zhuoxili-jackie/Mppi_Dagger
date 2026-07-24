# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.utils.math import quat_error_magnitude, quat_inv, quat_mul, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

from robot_lab.tasks.manager_based.beyondmimic.mdp.commands import MotionCommand
from robot_lab.tasks.manager_based.beyondmimic.mdp.rewards import _get_body_indexes


def bad_anchor_pos(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=1) > threshold


def bad_anchor_pos_z_only(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1]) > threshold


def bad_anchor_ori(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float
) -> torch.Tensor:
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    command: MotionCommand = env.command_manager.get_term(command_name)
    motion_projected_gravity_b = math_utils.quat_apply_inverse(command.anchor_quat_w, asset.data.GRAVITY_VEC_W)

    robot_projected_gravity_b = math_utils.quat_apply_inverse(command.robot_anchor_quat_w, asset.data.GRAVITY_VEC_W)

    return (motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]).abs() > threshold


def ready_gated_heading_yaw_drift_out_of_range(
    env: ManagerBasedRLEnv,
    command_name: str,
    yaw_command_deadzone: float,
    yaw_error_threshold: float,
    motion_command_name: str = "motion",
) -> torch.Tensor:
    """Terminate excessive ready-time heading drift when yaw is not commanded."""
    motion_cmd: MotionCommand = env.command_manager.get_term(motion_command_name)
    command = env.command_manager.get_command(command_name)
    yaw_inactive = torch.abs(command[:, 2]) < yaw_command_deadzone
    yaw_error = quat_error_magnitude(
        yaw_quat(motion_cmd.ready_anchor_quat_w),
        yaw_quat(motion_cmd.robot_anchor_quat_w),
    )
    return motion_cmd.ready & yaw_inactive & (yaw_error > yaw_error_threshold)


def ready_gated_roll_error_out_of_range(
    env: ManagerBasedRLEnv,
    roll_error_threshold: float,
    max_duration_s: float,
    motion_command_name: str = "motion",
) -> torch.Tensor:
    """Terminate side tilt that remains excessive relative to the ready posture."""
    motion_cmd: MotionCommand = env.command_manager.get_term(motion_command_name)
    relative_quat = quat_mul(quat_inv(motion_cmd.ready_anchor_quat_w), motion_cmd.robot_anchor_quat_w)
    w, x, y, z = relative_quat.unbind(dim=-1)
    roll_error = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    excessive_roll = motion_cmd.ready & (torch.abs(roll_error) > roll_error_threshold)

    duration_attr = "_ready_roll_excess_duration"
    excessive_duration = getattr(env, duration_attr, None)
    if excessive_duration is None or excessive_duration.shape != roll_error.shape:
        excessive_duration = torch.zeros_like(roll_error)
    excessive_duration = torch.where(
        excessive_roll,
        excessive_duration + env.step_dt,
        torch.zeros_like(excessive_duration),
    )
    setattr(env, duration_attr, excessive_duration.detach())
    return excessive_duration >= max_duration_s


def ready_gated_low_foot_contact_out_of_range(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    height_threshold: float,
    force_threshold: float = 1.0,
    max_duration_s: float = 0.15,
    require_all: bool = True,
    motion_command_name: str = "motion",
) -> torch.Tensor:
    """Terminate ready-stage ground crawling without affecting pre-ready boarding.

    The check is intentionally gated by ready and by contact. A foot that is
    briefly low during a swing is not enough; selected feet must be low and in
    contact for ``max_duration_s``.
    """
    motion_cmd: MotionCommand = env.command_manager.get_term(motion_command_name)
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor = env.scene.sensors[sensor_cfg.name]

    foot_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - env.scene.env_origins[:, 2].unsqueeze(-1)
    low_foot = foot_height < height_threshold

    net_contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    in_contact = net_contact_forces.norm(dim=-1).max(dim=1)[0] > force_threshold
    low_contact = low_foot & in_contact
    if require_all:
        invalid_contact = torch.all(low_contact, dim=1)
    else:
        invalid_contact = torch.any(low_contact, dim=1)

    active = motion_cmd.ready & invalid_contact
    duration_attr = "_ready_low_foot_contact_duration"
    duration = getattr(env, duration_attr, None)
    if duration is None or duration.shape[0] != env.num_envs:
        duration = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    duration = torch.where(active, duration + env.step_dt, torch.zeros_like(duration))
    setattr(env, duration_attr, duration.detach())
    return duration >= max_duration_s


def bad_motion_body_pos(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.norm(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes], dim=-1)
    return torch.any(error > threshold, dim=-1)


def bad_motion_body_pos_z_only(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.abs(command.body_pos_relative_w[:, body_indexes, -1] - command.robot_body_pos_w[:, body_indexes, -1])
    return torch.any(error > threshold, dim=-1)
