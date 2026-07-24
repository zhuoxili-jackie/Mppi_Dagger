# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

from robot_lab.tasks.manager_based.beyondmimic.mdp.commands import MotionCommand
from robot_lab.tasks.manager_based.beyondmimic.mdp.rewards import (
    _fixed_motion_tilt_error,
    _get_body_indexes,
    _ready_segment_heading_yaw_error,
)


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
    """Terminate heading drift from the end of the latest yaw-command segment."""
    yaw_error, gate = _ready_segment_heading_yaw_error(
        env,
        command_name=command_name,
        yaw_command_deadzone=yaw_command_deadzone,
        motion_command_name=motion_command_name,
    )
    return gate & (yaw_error > yaw_error_threshold)


def ready_gated_tilt_error_out_of_range(
    env: ManagerBasedRLEnv,
    tilt_error_threshold: float,
    motion_command_name: str = "motion",
) -> torch.Tensor:
    """Terminate yaw-invariant tilt deviation from the fixed final motion pose."""
    tilt_error, motion_cmd = _fixed_motion_tilt_error(env, motion_command_name)
    return motion_cmd.ready & (tilt_error > tilt_error_threshold)


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


def selected_bodies_airborne_too_long(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    max_air_time_s: float = 0.25,
    require_all: bool = True,
    command_name: str = "motion",
    reset_steps: int = 10,
) -> torch.Tensor:
    """Terminate prolonged loss of support without rejecting a normal one-foot swing.

    With ``require_all=True`` the term fires only when every selected body has
    remained airborne longer than ``max_air_time_s``.  This is useful for a
    front-wheel pair: either front wheel may swing, but lifting both wheels to
    evade a contact-gated surface termination is not a valid support state.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    too_long = air_time > max_air_time_s
    violation = torch.all(too_long, dim=1) if require_all else torch.any(too_long, dim=1)
    return (command.time_steps > reset_steps) & violation


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


def box_local_any_body_backward_or_drop_from_reset(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str = "motion",
    box_name: str = "box",
    sensor_cfg: SceneEntityCfg | None = None,
    force_threshold: float = 5.0,
    backward_margin: float = 0.035,
    drop_margin: float = 0.030,
    require_both: bool = False,
    unsafe_direction: float = -1.0,
    reset_steps: int = 3,
    cache_key: str = "default",
) -> torch.Tensor:
    """Terminate when any selected body retreats or drops toward an unsafe box region.

    The first few control steps establish a per-episode safe reference. This is
    intended for hard support-surface constraints where a small reward penalty
    is insufficient to prevent the policy from exploiting a lower ramp.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    box: RigidObject = env.scene[box_name]
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    rel_w = body_pos_w - box.data.root_pos_w[:, None, :]
    num_envs, num_bodies, _ = rel_w.shape
    box_yaw = math_utils.yaw_quat(box.data.root_quat_w).unsqueeze(1).expand(-1, num_bodies, -1)
    rel_local = math_utils.quat_apply_inverse(
        box_yaw.reshape(-1, 4), rel_w.reshape(-1, 3)
    ).reshape(num_envs, num_bodies, 3)
    current_xz = rel_local[..., [0, 2]]

    cache_attr = f"_box_local_unsafe_reset_target_{box_name}_{asset_cfg.name}_{cache_key}"
    target_xz = getattr(env, cache_attr, None)
    if target_xz is None or target_xz.shape != current_xz.shape:
        target_xz = current_xz.detach().clone()

    refresh = command.time_steps <= reset_steps
    if torch.any(refresh):
        target_xz = target_xz.clone()
        target_xz[refresh] = current_xz.detach()[refresh]
    setattr(env, cache_attr, target_xz.detach())

    direction = 1.0 if unsafe_direction >= 0.0 else -1.0
    backward = direction * (current_xz[..., 0] - target_xz[..., 0]) > backward_margin
    dropped = target_xz[..., 1] - current_xz[..., 1] > drop_margin
    violation = (backward & dropped) if require_both else (backward | dropped)
    if sensor_cfg is not None:
        contact_sensor = env.scene.sensors[sensor_cfg.name]
        net_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        in_contact = torch.linalg.norm(net_forces, dim=-1).amax(dim=1) > force_threshold
        violation = violation & in_contact

    active = command.time_steps > reset_steps
    return active & torch.any(violation, dim=1)


def box_local_any_body_axis_deviation_from_reset(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str = "motion",
    box_name: str = "box",
    axis: int = 0,
    max_deviation: float = 0.05,
    max_duration_s: float = 0.0,
    reset_steps: int = 10,
    cache_key: str = "default",
) -> torch.Tensor:
    """Terminate when any selected body leaves a symmetric reset-relative corridor."""
    asset: Articulation = env.scene[asset_cfg.name]
    box: RigidObject = env.scene[box_name]
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    rel_w = body_pos_w - box.data.root_pos_w[:, None, :]
    num_envs, num_bodies, _ = rel_w.shape
    box_yaw = math_utils.yaw_quat(box.data.root_quat_w).unsqueeze(1).expand(-1, num_bodies, -1)
    rel_local = math_utils.quat_apply_inverse(
        box_yaw.reshape(-1, 4), rel_w.reshape(-1, 3)
    ).reshape(num_envs, num_bodies, 3)
    current = rel_local[..., axis]

    cache_attr = f"_box_local_axis_termination_target_{box_name}_{asset_cfg.name}_{axis}_{cache_key}"
    target = getattr(env, cache_attr, None)
    if target is None or target.shape != current.shape:
        target = current.detach().clone()

    refresh = command.time_steps <= reset_steps
    if torch.any(refresh):
        target = target.clone()
        target[refresh] = current.detach()[refresh]
    setattr(env, cache_attr, target.detach())

    active = command.time_steps > reset_steps
    outside = active & torch.any(torch.abs(current - target) > max_deviation, dim=1)
    if max_duration_s <= 0.0:
        return outside

    duration_attr = f"_box_local_axis_termination_duration_{box_name}_{asset_cfg.name}_{axis}_{cache_key}"
    duration = getattr(env, duration_attr, None)
    if duration is None or duration.shape[0] != env.num_envs:
        duration = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    duration = torch.where(outside, duration + env.step_dt, torch.zeros_like(duration))
    setattr(env, duration_attr, duration.detach())
    return duration >= max_duration_s
