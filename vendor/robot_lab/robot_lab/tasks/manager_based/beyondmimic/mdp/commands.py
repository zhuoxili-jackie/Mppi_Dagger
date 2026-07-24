# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import numpy as np
import torch

from isaaclab.assets import Articulation
from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    matrix_from_quat,
    quat_apply,
    quat_apply_inverse,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class MotionLoader:
    def __init__(self, motion_file: str, body_indexes: Sequence[int], device: str = "cpu"):
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        data = np.load(motion_file)


        self.fps = data["fps"]
        self.joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
        self.joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
        self._body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
        self._body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
        self._body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
        self._body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)
        self._body_indexes = body_indexes
        self.time_step_total = self.joint_pos.shape[0]
        print("\n" + "=" * 50)
        print("Tensor shapes after conversion:")
        print(f"  joint_pos: {self.joint_pos.shape} (expected: [num_frames, num_joints])")
        print(f"  joint_vel: {self.joint_vel.shape}")
        print(f"  _body_pos_w: {self._body_pos_w.shape} (expected: [num_frames, num_bodies, 3])")
        print(f"  _body_quat_w: {self._body_quat_w.shape} (expected: [num_frames, num_bodies, 4])")
        print(f"  _body_lin_vel_w: {self._body_lin_vel_w.shape}")
        print(f"  _body_ang_vel_w: {self._body_ang_vel_w.shape}")
        print(f"  _body_indexes : {self._body_indexes}")
    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]


class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )

        self.motion = MotionLoader(self.cfg.motion_file, self.body_indexes, device=self.device)
        if self.cfg.hold_joint_names:
            self._hold_joint_ids, _ = self.robot.find_joints(self.cfg.hold_joint_names, preserve_order=True)
        else:
            self._hold_joint_ids = slice(None)
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        self.bin_count = int(self.motion.time_step_total // (1 / (env.cfg.decimation * env.cfg.sim.dt))) + 1
        self.bin_failed_count = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self._current_bin_failed = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self.kernel = torch.tensor(
            [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)], device=self.device
        )
        self.kernel = self.kernel / self.kernel.sum()

        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_late_fraction"] = torch.zeros(self.num_envs, device=self.device)
        # Internal motion phase: 0=track trajectory, 1=hold last frame, 2=ready for a follow-up skill.
        self.stage = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.stable_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.post_stable_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.hold_elapsed_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.ready_anchor_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.ready_anchor_quat_w = torch.zeros(self.num_envs, 4, device=self.device)
        self.ready_anchor_quat_w[:, 0] = 1.0
        self.metrics["stage"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["ready_gate"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:  # TODO Consider again if this is the best observation
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def ready(self) -> torch.Tensor:
        """Whether the reference motion is complete and the robot is stable enough for a follow-up skill."""
        return self.stage >= 2

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motion.joint_pos[self.time_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion.joint_vel[self.time_steps]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps, self.motion_anchor_body_index] + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]

    def _update_metrics(self):
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)

        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(
            dim=-1
        )

        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(
            dim=-1
        )

        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        episode_failed = self._env.termination_manager.terminated[env_ids]
        if torch.any(episode_failed):
            current_bin_index = torch.clamp(
                (self.time_steps * self.bin_count) // max(self.motion.time_step_total, 1), 0, self.bin_count - 1
            )
            fail_bins = current_bin_index[env_ids][episode_failed]
            self._current_bin_failed[:] = torch.bincount(fail_bins, minlength=self.bin_count)

        # Sample
        sampling_probabilities = self.bin_failed_count + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
        sampling_probabilities = torch.nn.functional.pad(
            sampling_probabilities.unsqueeze(0).unsqueeze(0),
            (0, self.cfg.adaptive_kernel_size - 1),  # Non-causal kernel
            mode="replicate",
        )
        sampling_probabilities = torch.nn.functional.conv1d(sampling_probabilities, self.kernel.view(1, 1, -1)).view(-1)

        sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

        sampled_bins = torch.multinomial(sampling_probabilities, len(env_ids), replacement=True)
        late_mask = torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)
        if self.cfg.late_sampling_ratio > 0.0:
            late_mask = torch.rand(len(env_ids), device=self.device) < self.cfg.late_sampling_ratio
            late_count = int(late_mask.sum().item())
            if late_count > 0:
                late_bin_start = min(
                    max(int(self.cfg.late_sampling_start_fraction * self.bin_count), 0),
                    self.bin_count - 1,
                )
                sampled_bins[late_mask] = torch.randint(
                    late_bin_start,
                    self.bin_count,
                    (late_count,),
                    device=self.device,
                )

        self.time_steps[env_ids] = (
            (sampled_bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device))
            / self.bin_count
            * (self.motion.time_step_total - 1)
        ).long()

        # Metrics
        H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
        H_norm = H / math.log(self.bin_count)
        pmax, imax = sampling_probabilities.max(dim=0)
        self.metrics["sampling_entropy"][:] = H_norm
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count
        self.metrics["sampling_late_fraction"][env_ids] = late_mask.float()

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        self._adaptive_sampling(env_ids)
        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self.stage[env_ids_t] = 0
        self.stable_counter[env_ids_t] = 0
        self.post_stable_counter[env_ids_t] = 0
        self.hold_elapsed_counter[env_ids_t] = 0
        self.ready_anchor_pos_w[env_ids_t] = 0.0
        self.ready_anchor_quat_w[env_ids_t] = 0.0
        self.ready_anchor_quat_w[env_ids_t, 0] = 1.0

        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos[env_ids] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel[env_ids] += rand_samples[:, :3]
        root_ang_vel[env_ids] += rand_samples[:, 3:]

        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()

        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
        )
        self.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1),
            env_ids=env_ids,
        )

    def _update_command(self):
        completion_gate_enabled = self.cfg.enable_stage_command or self.cfg.enable_completion_gate
        if not completion_gate_enabled:
            self.time_steps += 1
            env_ids = torch.where(self.time_steps >= self.motion.time_step_total)[0]
            self._resample_command(env_ids)
        else:
            last_frame = self.motion.time_step_total - 1
            track_mask = self.stage == 0
            self.time_steps[track_mask] += 1
            reached_end = track_mask & (self.time_steps >= last_frame)
            self.time_steps[reached_end] = last_frame
            self.stage[reached_end] = 1
            self.stable_counter[reached_end] = 0
            self.post_stable_counter[reached_end] = 0
            self.hold_elapsed_counter[reached_end] = 0
            # Keep hold/cmd stages pinned to the final reference frame.
            self.time_steps[self.stage >= 1] = last_frame

        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)

        self.bin_failed_count = (
            self.cfg.adaptive_alpha * self._current_bin_failed + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
        )
        self._current_bin_failed.zero_()
        self.metrics["stage"] = self.stage.float()
        self.metrics["ready_gate"] = self.ready.float()

        if completion_gate_enabled:
            hold_env_ids = torch.where(self.stage == 1)[0]
            if hold_env_ids.numel() > 0:
                self.hold_elapsed_counter[hold_env_ids] += 1
                err_anchor_rot = quat_error_magnitude(
                    self.anchor_quat_w[hold_env_ids], self.robot_anchor_quat_w[hold_env_ids]
                )
                err_body_pos = torch.norm(
                    self.body_pos_relative_w[hold_env_ids] - self.robot_body_pos_w[hold_env_ids], dim=-1
                ).mean(dim=-1)
                base_lin_vel = torch.norm(self.robot_anchor_lin_vel_w[hold_env_ids], dim=-1)
                base_ang_vel = torch.norm(self.robot_anchor_ang_vel_w[hold_env_ids], dim=-1)

                stable_now = (
                    (err_anchor_rot < self.cfg.hold_anchor_rot_threshold)
                    & (err_body_pos < self.cfg.hold_body_pos_threshold)
                    & (base_lin_vel < self.cfg.hold_base_lin_vel_threshold)
                    & (base_ang_vel < self.cfg.hold_base_ang_vel_threshold)
                )
                if self.cfg.hold_joint_pos_threshold is not None:
                    err_joint_pos = torch.norm(
                        self.joint_pos[hold_env_ids][:, self._hold_joint_ids]
                        - self.robot_joint_pos[hold_env_ids][:, self._hold_joint_ids],
                        dim=-1,
                    )
                    stable_now &= err_joint_pos < self.cfg.hold_joint_pos_threshold
                self.stable_counter[hold_env_ids] = torch.where(
                    stable_now,
                    self.stable_counter[hold_env_ids] + 1,
                    torch.zeros_like(self.stable_counter[hold_env_ids]),
                )

                stable_reached = self.stable_counter[hold_env_ids] >= self.cfg.hold_stable_steps
                self.post_stable_counter[hold_env_ids] = torch.where(
                    stable_reached,
                    self.post_stable_counter[hold_env_ids] + 1,
                    torch.zeros_like(self.post_stable_counter[hold_env_ids]),
                )

                promote = stable_reached & (
                    self.post_stable_counter[hold_env_ids] >= self.cfg.extra_hold_steps_after_stable
                )
                if self.cfg.max_hold_steps_before_force_command > 0:
                    promote = promote | (
                        self.hold_elapsed_counter[hold_env_ids] >= self.cfg.max_hold_steps_before_force_command
                    )
                if torch.any(promote):
                    promote_ids = hold_env_ids[promote]
                    self.ready_anchor_pos_w[promote_ids] = self.robot_anchor_pos_w[promote_ids]
                    self.ready_anchor_quat_w[promote_ids] = self.robot_anchor_quat_w[promote_ids]
                    self.stage[promote_ids] = 2

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/current/anchor")
                )
                self.goal_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/anchor")
                )

                self.current_body_visualizers = []
                self.goal_body_visualizers = []
                for name in self.cfg.body_names:
                    self.current_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/current/" + name)
                        )
                    )
                    self.goal_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/" + name)
                        )
                    )

            self.current_anchor_visualizer.set_visibility(True)
            self.goal_anchor_visualizer.set_visibility(True)
            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].set_visibility(True)
                self.goal_body_visualizers[i].set_visibility(True)

        else:
            if hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer.set_visibility(False)
                self.goal_anchor_visualizer.set_visibility(False)
                for i in range(len(self.cfg.body_names)):
                    self.current_body_visualizers[i].set_visibility(False)
                    self.goal_body_visualizers[i].set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        self.current_anchor_visualizer.visualize(self.robot_anchor_pos_w, self.robot_anchor_quat_w)
        self.goal_anchor_visualizer.visualize(self.anchor_pos_w, self.anchor_quat_w)

        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i])
            self.goal_body_visualizers[i].visualize(self.body_pos_relative_w[:, i], self.body_quat_relative_w[:, i])


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    class_type: type = MotionCommand

    asset_name: str = MISSING

    motion_file: str = MISSING
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    joint_position_range: tuple[float, float] = (-0.52, 0.52)

    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001
    # Optional reset bias for follow-up skills. Defaults preserve the original
    # full-trajectory adaptive sampler.
    late_sampling_ratio: float = 0.0
    late_sampling_start_fraction: float = 0.8

    # Internal motion-completion phases:
    # 0(track trajectory) -> 1(hold last frame) -> 2(ready for a follow-up skill)
    enable_stage_command: bool = False
    # Generic completion/ready gate for follow-up skills such as a frozen-baseline residual controller.
    # This reuses the final-frame stability detector without enabling the old stage2 training route.
    enable_completion_gate: bool = False
    hold_joint_names: list[str] | None = None
    hold_anchor_rot_threshold: float = 0.12
    hold_body_pos_threshold: float = 0.06
    # Set to None when a stable follow-up skill may legitimately use a different
    # joint configuration from the final reference frame.
    hold_joint_pos_threshold: float | None = 0.12
    hold_base_lin_vel_threshold: float = 0.12
    hold_base_ang_vel_threshold: float = 0.35
    hold_stable_steps: int = 60
    extra_hold_steps_after_stable: int = 60
    max_hold_steps_before_force_command: int = 400

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)


class StageRedMotionCommand(MotionCommand):
    """Reference-aligned differential RED features for staged boarding imitation.

    The motion command itself still owns the normal BeyondMimic reference timing,
    reset, visualization, and metric behavior. This extension only exposes
    train-side RED observations:
    - policy history minus the current reference history;
    - zero-centered expert differentials with a small tolerance noise.

    That keeps the deployable actor path unchanged while making RED the core
    learned imitation scorer.
    """

    cfg: StageRedMotionCommandCfg

    def __init__(self, cfg: StageRedMotionCommandCfg, env: ManagerBasedRLEnv):
        if cfg.red_history_length < 2:
            raise ValueError("red_history_length must be at least 2.")
        if cfg.red_demo_feature_noise_std < 0.0:
            raise ValueError("red_demo_feature_noise_std must be non-negative.")
        for name, value in (cfg.red_feature_scales or {}).items():
            if value < 0.0:
                raise ValueError(f"red_feature_scales['{name}'] must be non-negative.")
        if not cfg.style_joint_names:
            raise ValueError("style_joint_names must contain at least one joint.")
        if not cfg.style_body_names:
            raise ValueError("style_body_names must contain at least one body.")

        super().__init__(cfg, env)

        self._stage_red_style_joint_ids, _ = self.robot.find_joints(cfg.style_joint_names, preserve_order=True)
        self._stage_red_style_robot_body_ids, _ = self.robot.find_bodies(
            cfg.style_body_names,
            preserve_order=True,
        )
        self._stage_red_style_motion_body_ids = torch.tensor(
            [cfg.body_names.index(name) for name in cfg.style_body_names],
            dtype=torch.long,
            device=self.device,
        )
        self._stage_red_joint_default = self.motion.joint_pos[0, self._stage_red_style_joint_ids].clone()
        self._stage_red_reference_frame_features = self._build_stage_red_reference_frame_features()
        self._stage_red_frame_dim = self._stage_red_reference_frame_features.shape[-1]
        self._stage_red_frame_feature_weights = self._build_stage_red_frame_feature_weights()
        zero_window = torch.zeros(
            1,
            cfg.red_history_length,
            self._stage_red_frame_dim,
            dtype=torch.float32,
            device=self.device,
        )
        self._stage_red_feature_dim = self._stage_red_window_differential_features(zero_window, zero_window).shape[-1]
        self._stage_red_policy_history = torch.zeros(
            self.num_envs,
            cfg.red_history_length,
            self._stage_red_frame_dim,
            dtype=torch.float32,
            device=self.device,
        )
        self._reset_stage_red_policy_history(torch.arange(self.num_envs, device=self.device))
        self._sync_stage_red_action_offsets(torch.arange(self.num_envs, device=self.device))

    @property
    def red_policy_features(self) -> torch.Tensor:
        """Return policy-vs-current-reference differential features for RED."""
        reference_windows = self._stage_red_reference_windows(self.time_steps)
        return self._stage_red_window_differential_features(self._stage_red_policy_history, reference_windows)

    @property
    def red_demo_features(self) -> torch.Tensor:
        """Return the expert differential target: exact reference tracking."""
        demo_features = torch.zeros(
            self.num_envs,
            self._stage_red_feature_dim,
            dtype=torch.float32,
            device=self.device,
        )
        if self.cfg.red_demo_feature_noise_std > 0.0:
            demo_features = demo_features + torch.randn_like(demo_features) * self.cfg.red_demo_feature_noise_std
        return demo_features

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        if hasattr(self, "_stage_red_policy_history"):
            env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            self._reset_stage_red_policy_history(env_ids_t)
            self._sync_stage_red_action_offsets(env_ids_t)

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        if not self.cfg.reset_at_first_frame:
            super()._adaptive_sampling(env_ids)
            return

        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self.time_steps[env_ids_t] = 0
        self.metrics["sampling_entropy"][:] = 0.0
        self.metrics["sampling_top1_prob"][:] = 1.0
        self.metrics["sampling_top1_bin"][:] = 0.0
        self.metrics["sampling_late_fraction"][env_ids_t] = 0.0

    def _update_command(self):
        super()._update_command()
        self._append_stage_red_policy_history()

    def _stage_red_reference_windows(self, end_steps: torch.Tensor) -> torch.Tensor:
        offsets = torch.arange(
            -self.cfg.red_history_length + 1,
            1,
            dtype=torch.long,
            device=self.device,
        )
        frame_ids = torch.clamp(end_steps.unsqueeze(1) + offsets.unsqueeze(0), 0, self.motion.time_step_total - 1)
        return self._stage_red_reference_frame_features[frame_ids]

    def _stage_red_window_differential_features(
        self,
        candidate: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        frame_error = (candidate - reference) * self._stage_red_frame_feature_weights
        candidate_delta = candidate[:, 1:] - candidate[:, :-1]
        reference_delta = reference[:, 1:] - reference[:, :-1]
        temporal_scale = float((self.cfg.red_feature_scales or {}).get("temporal", 1.0))
        temporal_error = (candidate_delta - reference_delta) * self._stage_red_frame_feature_weights * temporal_scale
        return torch.cat([frame_error.flatten(start_dim=1), temporal_error.flatten(start_dim=1)], dim=-1)

    def _build_stage_red_frame_feature_weights(self) -> torch.Tensor:
        scales = self.cfg.red_feature_scales or {}

        def scale(name: str, group: str) -> float:
            return float(scales.get(name, scales.get(group, 1.0)))

        num_joints = len(self._stage_red_style_joint_ids)
        num_bodies = len(self._stage_red_style_motion_body_ids)
        weights = torch.cat(
            [
                torch.full((num_joints,), scale("joint_pos", "pose"), dtype=torch.float32, device=self.device),
                torch.full((num_joints,), scale("joint_vel", "velocity"), dtype=torch.float32, device=self.device),
                torch.full((3,), scale("anchor_pos", "pose"), dtype=torch.float32, device=self.device),
                torch.full((6,), scale("anchor_ori", "pose"), dtype=torch.float32, device=self.device),
                torch.full((3,), scale("projected_gravity", "pose"), dtype=torch.float32, device=self.device),
                torch.full((3 * num_bodies,), scale("body_pos", "pose"), dtype=torch.float32, device=self.device),
                torch.full((6 * num_bodies,), scale("body_ori", "pose"), dtype=torch.float32, device=self.device),
                torch.full((3 * num_bodies,), scale("body_lin_vel", "velocity"), dtype=torch.float32, device=self.device),
                torch.full((3 * num_bodies,), scale("body_ang_vel", "velocity"), dtype=torch.float32, device=self.device),
                torch.full((3,), scale("anchor_lin_vel", "velocity"), dtype=torch.float32, device=self.device),
                torch.full((3,), scale("anchor_ang_vel", "velocity"), dtype=torch.float32, device=self.device),
            ]
        )
        if weights.numel() != self._stage_red_frame_dim:
            raise RuntimeError(
                f"Stage RED feature weights have dim {weights.numel()}, expected {self._stage_red_frame_dim}."
            )
        return weights

    def _build_stage_red_reference_frame_features(self) -> torch.Tensor:
        anchor_pos = self.motion.body_pos_w[:, self.motion_anchor_body_index]
        anchor_quat = self.motion.body_quat_w[:, self.motion_anchor_body_index]
        anchor_lin_vel = self.motion.body_lin_vel_w[:, self.motion_anchor_body_index]
        anchor_ang_vel = self.motion.body_ang_vel_w[:, self.motion_anchor_body_index]

        anchor_ori_w = matrix_from_quat(anchor_quat)[..., :2].flatten(start_dim=1)

        body_pos = self.motion.body_pos_w[:, self._stage_red_style_motion_body_ids]
        body_quat = self.motion.body_quat_w[:, self._stage_red_style_motion_body_ids]
        body_lin_vel = self.motion.body_lin_vel_w[:, self._stage_red_style_motion_body_ids]
        body_ang_vel = self.motion.body_ang_vel_w[:, self._stage_red_style_motion_body_ids]
        expanded_anchor_quat = anchor_quat.unsqueeze(1).expand(-1, body_pos.shape[1], -1)
        body_pos_b = quat_apply_inverse(expanded_anchor_quat, body_pos - anchor_pos.unsqueeze(1))
        body_quat_b = quat_mul(quat_inv(expanded_anchor_quat), body_quat)
        body_ori_b = matrix_from_quat(body_quat_b)[..., :2].flatten(start_dim=1)
        body_lin_vel_b = quat_apply_inverse(expanded_anchor_quat, body_lin_vel - anchor_lin_vel.unsqueeze(1))
        body_ang_vel_b = quat_apply_inverse(expanded_anchor_quat, body_ang_vel - anchor_ang_vel.unsqueeze(1))

        gravity_w = torch.zeros_like(anchor_pos)
        gravity_w[..., 2] = -1.0
        projected_gravity = quat_apply_inverse(anchor_quat, gravity_w)
        anchor_lin_vel_b = quat_apply_inverse(anchor_quat, anchor_lin_vel)
        anchor_ang_vel_b = quat_apply_inverse(anchor_quat, anchor_ang_vel)

        joint_pos = self.motion.joint_pos[:, self._stage_red_style_joint_ids] - self._stage_red_joint_default
        joint_vel = self.motion.joint_vel[:, self._stage_red_style_joint_ids]

        return torch.cat(
            [
                joint_pos,
                joint_vel,
                anchor_pos,
                anchor_ori_w,
                projected_gravity,
                body_pos_b.flatten(start_dim=1),
                body_ori_b,
                body_lin_vel_b.flatten(start_dim=1),
                body_ang_vel_b.flatten(start_dim=1),
                anchor_lin_vel_b,
                anchor_ang_vel_b,
            ],
            dim=-1,
        )

    def _current_stage_red_frame_features(self) -> torch.Tensor:
        env_origins = self._env.scene.env_origins
        anchor_pos = self.robot_anchor_pos_w
        anchor_quat = self.robot_anchor_quat_w
        anchor_lin_vel = self.robot_anchor_lin_vel_w
        anchor_ang_vel = self.robot_anchor_ang_vel_w

        anchor_pos_local = anchor_pos - env_origins
        anchor_ori_w = matrix_from_quat(anchor_quat)[..., :2].flatten(start_dim=1)

        body_pos = self.robot.data.body_pos_w[:, self._stage_red_style_robot_body_ids]
        body_quat = self.robot.data.body_quat_w[:, self._stage_red_style_robot_body_ids]
        body_lin_vel = self.robot.data.body_lin_vel_w[:, self._stage_red_style_robot_body_ids]
        body_ang_vel = self.robot.data.body_ang_vel_w[:, self._stage_red_style_robot_body_ids]
        expanded_anchor_quat = anchor_quat.unsqueeze(1).expand(-1, body_pos.shape[1], -1)
        body_pos_b = quat_apply_inverse(expanded_anchor_quat, body_pos - anchor_pos.unsqueeze(1))
        body_quat_b = quat_mul(quat_inv(expanded_anchor_quat), body_quat)
        body_ori_b = matrix_from_quat(body_quat_b)[..., :2].flatten(start_dim=1)
        body_lin_vel_b = quat_apply_inverse(expanded_anchor_quat, body_lin_vel - anchor_lin_vel.unsqueeze(1))
        body_ang_vel_b = quat_apply_inverse(expanded_anchor_quat, body_ang_vel - anchor_ang_vel.unsqueeze(1))

        projected_gravity = quat_apply_inverse(anchor_quat, self.robot.data.GRAVITY_VEC_W)
        anchor_lin_vel_b = quat_apply_inverse(anchor_quat, anchor_lin_vel)
        anchor_ang_vel_b = quat_apply_inverse(anchor_quat, anchor_ang_vel)

        joint_pos = self.robot.data.joint_pos[:, self._stage_red_style_joint_ids] - self._stage_red_joint_default
        joint_vel = self.robot.data.joint_vel[:, self._stage_red_style_joint_ids]

        return torch.cat(
            [
                joint_pos,
                joint_vel,
                anchor_pos_local,
                anchor_ori_w,
                projected_gravity,
                body_pos_b.flatten(start_dim=1),
                body_ori_b,
                body_lin_vel_b.flatten(start_dim=1),
                body_ang_vel_b.flatten(start_dim=1),
                anchor_lin_vel_b,
                anchor_ang_vel_b,
            ],
            dim=-1,
        )

    def _append_stage_red_policy_history(self):
        self._stage_red_policy_history[:, :-1] = self._stage_red_policy_history[:, 1:].clone()
        self._stage_red_policy_history[:, -1] = self._current_stage_red_frame_features()

    def _reset_stage_red_policy_history(self, env_ids: torch.Tensor):
        self._stage_red_policy_history[env_ids] = self._stage_red_reference_windows(self.time_steps[env_ids]).clone()

    def _sync_stage_red_action_offsets(self, env_ids: torch.Tensor):
        if not self.cfg.sync_action_offsets_to_reference:
            return
        self._sync_stage_red_action_offset(
            action_name=self.cfg.joint_position_action_name,
            env_ids=env_ids,
            reference=self.joint_pos,
        )
        self._sync_stage_red_action_offset(
            action_name=self.cfg.joint_velocity_action_name,
            env_ids=env_ids,
            reference=self.joint_vel,
        )

    def _sync_stage_red_action_offset(self, action_name: str | None, env_ids: torch.Tensor, reference: torch.Tensor):
        if action_name is None:
            return
        if not hasattr(self._env, "action_manager"):
            return
        try:
            action_term = self._env.action_manager.get_term(action_name)
        except (AttributeError, KeyError, ValueError):
            return
        if not hasattr(action_term, "_offset") or not isinstance(action_term._offset, torch.Tensor):
            return
        if not hasattr(action_term, "_joint_ids"):
            return

        joint_ids = action_term._joint_ids
        action_term._offset[env_ids] = reference[env_ids][:, joint_ids]
        if hasattr(action_term, "_raw_actions"):
            action_term._raw_actions[env_ids] = 0.0
        if hasattr(action_term, "_processed_actions"):
            action_term._processed_actions[env_ids] = action_term._offset[env_ids]


@configclass
class StageRedMotionCommandCfg(MotionCommandCfg):
    """Configuration for reference-aligned differential RED motion imitation."""

    class_type: type = StageRedMotionCommand

    style_joint_names: list[str] = MISSING
    style_body_names: list[str] = MISSING
    red_history_length: int = 8
    red_demo_feature_noise_std: float = 0.0
    red_feature_scales: dict[str, float] | None = None
    sync_action_offsets_to_reference: bool = True
    joint_position_action_name: str | None = "joint_pos"
    joint_velocity_action_name: str | None = "joint_vel"
    reset_at_first_frame: bool = False


class LateralReferenceMotionCommand(MotionCommand):
    """Sample a continuous y command and select its nearest reference motion."""

    cfg: LateralReferenceMotionCommandCfg

    def __init__(self, cfg: LateralReferenceMotionCommandCfg, env: ManagerBasedRLEnv):
        if len(cfg.motion_files) == 0:
            raise ValueError("motion_files must contain at least one lateral reference motion.")
        if len(cfg.motion_files) != len(cfg.lateral_velocities):
            raise ValueError("motion_files and lateral_velocities must have the same length.")
        if not 0.0 <= cfg.standing_probability < 1.0:
            raise ValueError("standing_probability must be in [0, 1).")
        if cfg.target_velocity_range[0] >= 0.0 or cfg.target_velocity_range[1] <= 0.0:
            raise ValueError("target_velocity_range must contain both negative and positive velocities.")
        max_target_velocity = min(abs(cfg.target_velocity_range[0]), abs(cfg.target_velocity_range[1]))
        if not 0.0 < cfg.min_abs_target_velocity <= max_target_velocity:
            raise ValueError("min_abs_target_velocity must be positive and within target_velocity_range.")

        # MotionCommand owns reset, metrics, visualization, and reference-frame updates.
        # Use the first file for that shared machinery, then replace reference access
        # with a per-environment selection across all lateral motions.
        cfg.motion_file = cfg.motion_files[0]
        super().__init__(cfg, env)

        motions = [self.motion]
        motions.extend(MotionLoader(path, self.body_indexes, device=self.device) for path in cfg.motion_files[1:])
        frame_counts = {motion.time_step_total for motion in motions}
        if len(frame_counts) != 1:
            raise ValueError(f"All lateral reference motions must have equal frame counts, got {sorted(frame_counts)}.")

        self._joint_pos_refs = torch.stack([motion.joint_pos for motion in motions], dim=0)
        self._joint_vel_refs = torch.stack([motion.joint_vel for motion in motions], dim=0)
        self._body_pos_refs = torch.stack([motion.body_pos_w for motion in motions], dim=0)
        self._body_quat_refs = torch.stack([motion.body_quat_w for motion in motions], dim=0)
        self._body_lin_vel_refs = torch.stack([motion.body_lin_vel_w for motion in motions], dim=0)
        self._body_ang_vel_refs = torch.stack([motion.body_ang_vel_w for motion in motions], dim=0)

        self._moving_motion_count = len(motions)
        velocity_values = list(cfg.lateral_velocities)
        if cfg.standing_probability > 0.0:
            # A stationary reference is the first valid hitched pose held in place.
            self._joint_pos_refs = torch.cat(
                [self._joint_pos_refs, self._joint_pos_refs[0:1, 0:1].expand(1, self.motion.time_step_total, -1)],
                dim=0,
            )
            self._joint_vel_refs = torch.cat([self._joint_vel_refs, torch.zeros_like(self._joint_vel_refs[0:1])], dim=0)
            self._body_pos_refs = torch.cat(
                [self._body_pos_refs, self._body_pos_refs[0:1, 0:1].expand(1, self.motion.time_step_total, -1, -1)],
                dim=0,
            )
            self._body_quat_refs = torch.cat(
                [self._body_quat_refs, self._body_quat_refs[0:1, 0:1].expand(1, self.motion.time_step_total, -1, -1)],
                dim=0,
            )
            self._body_lin_vel_refs = torch.cat(
                [self._body_lin_vel_refs, torch.zeros_like(self._body_lin_vel_refs[0:1])], dim=0
            )
            self._body_ang_vel_refs = torch.cat(
                [self._body_ang_vel_refs, torch.zeros_like(self._body_ang_vel_refs[0:1])], dim=0
            )
            velocity_values.append(0.0)

        self._reference_lateral_velocities = torch.tensor(velocity_values, dtype=torch.float32, device=self.device)
        self._target_lateral_velocities = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # The actor receives a fixed, deployment-friendly motion prefix. Dynamic
        # references remain available to the critic and imitation rewards.
        self._actor_joint_pos = self._joint_pos_refs[0, 0].unsqueeze(0).expand(self.num_envs, -1)
        self._actor_joint_vel = torch.zeros_like(self._actor_joint_pos)
        self._actor_anchor_quat_w = self._body_quat_refs[0, 0, self.motion_anchor_body_index].unsqueeze(0).expand(
            self.num_envs, -1
        )
        self.metrics["target_lateral_velocity"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["target_lateral_speed"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["reference_velocity_gap"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self._joint_pos_refs[self.motion_ids, self.time_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self._joint_vel_refs[self.motion_ids, self.time_steps]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_refs[self.motion_ids, self.time_steps] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_refs[self.motion_ids, self.time_steps]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_refs[self.motion_ids, self.time_steps]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_refs[self.motion_ids, self.time_steps]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self.body_pos_w[:, self.motion_anchor_body_index]

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.body_quat_w[:, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.body_lin_vel_w[:, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.body_ang_vel_w[:, self.motion_anchor_body_index]

    @property
    def actor_command(self) -> torch.Tensor:
        return torch.cat([self._actor_joint_pos, self._actor_joint_vel], dim=1)

    @property
    def actor_anchor_quat_w(self) -> torch.Tensor:
        return self._actor_anchor_quat_w

    @property
    def velocity_command(self) -> torch.Tensor:
        command = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        command[:, 1] = self._target_lateral_velocities
        return command

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        magnitude = torch.empty(len(env_ids), device=self.device).uniform_(
            self.cfg.min_abs_target_velocity,
            min(abs(self.cfg.target_velocity_range[0]), abs(self.cfg.target_velocity_range[1])),
        )
        sign = torch.where(
            torch.rand(len(env_ids), device=self.device) < 0.5,
            -torch.ones_like(magnitude),
            torch.ones_like(magnitude),
        )
        target_velocity = magnitude * sign

        moving_reference_velocities = self._reference_lateral_velocities[: self._moving_motion_count]
        selected = torch.argmin(
            torch.abs(target_velocity.unsqueeze(1) - moving_reference_velocities.unsqueeze(0)),
            dim=1,
        )
        if self.cfg.standing_probability > 0.0:
            standing = torch.rand(len(env_ids), device=self.device) < self.cfg.standing_probability
            selected[standing] = self._moving_motion_count
            target_velocity[standing] = 0.0

        self.motion_ids[env_ids_t] = selected
        self._target_lateral_velocities[env_ids_t] = target_velocity
        super()._resample_command(env_ids)

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        if not self.cfg.reset_at_first_frame:
            super()._adaptive_sampling(env_ids)
            return

        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self.time_steps[env_ids_t] = 0
        self.metrics["sampling_entropy"][:] = 0.0
        self.metrics["sampling_top1_prob"][:] = 1.0
        self.metrics["sampling_top1_bin"][:] = 0.0
        self.metrics["sampling_late_fraction"][env_ids_t] = 0.0

    def _update_metrics(self):
        super()._update_metrics()
        target_velocity = self.velocity_command[:, 1]
        reference_velocity = self._reference_lateral_velocities[self.motion_ids]
        self.metrics["target_lateral_velocity"] = target_velocity
        self.metrics["target_lateral_speed"] = torch.abs(target_velocity)
        self.metrics["reference_velocity_gap"] = torch.abs(target_velocity - reference_velocity)


@configclass
class LateralReferenceMotionCommandCfg(MotionCommandCfg):
    """Configuration for command-conditioned lateral reference imitation."""

    class_type: type = LateralReferenceMotionCommand

    motion_files: list[str] = MISSING
    lateral_velocities: list[float] = MISSING
    standing_probability: float = 0.15
    target_velocity_range: tuple[float, float] = (-0.20, 0.20)
    min_abs_target_velocity: float = 0.03
    reset_at_first_frame: bool = True


class LateralForwardMicroAdjustMotionCommand(LateralReferenceMotionCommand):
    """Lateral command with sparse x-axis micro-adjust samples.

    Most environments keep the proven lateral v17 sampling.  A small fraction
    instead receives a forward/backward-only command while holding the first
    valid hitched reference pose, teaching deployment-time position nudges
    without diluting the lateral gait.
    """

    cfg: LateralForwardMicroAdjustMotionCommandCfg

    def __init__(self, cfg: LateralForwardMicroAdjustMotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._target_forward_velocities = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.metrics["target_forward_velocity"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["target_forward_speed"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def velocity_command(self) -> torch.Tensor:
        command = super().velocity_command
        if hasattr(self, "_target_forward_velocities"):
            command[:, 0] = self._target_forward_velocities
        return command

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        if len(env_ids) == 0:
            return

        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        forward_sample = torch.rand(len(env_ids), device=self.device) < self.cfg.forward_probability
        forward_ids = env_ids_t[forward_sample]

        self._target_forward_velocities[env_ids_t] = 0.0
        if forward_ids.numel() == 0:
            return

        max_forward_velocity = min(abs(self.cfg.forward_velocity_range[0]), abs(self.cfg.forward_velocity_range[1]))
        magnitude = torch.empty(forward_ids.numel(), device=self.device).uniform_(
            self.cfg.min_abs_forward_velocity,
            max_forward_velocity,
        )
        sign = torch.where(
            torch.rand(forward_ids.numel(), device=self.device) < 0.5,
            -torch.ones_like(magnitude),
            torch.ones_like(magnitude),
        )
        self._target_forward_velocities[forward_ids] = magnitude * sign
        self._target_lateral_velocities[forward_ids] = 0.0

        # Use the standing reference for x-only nudges so the policy is not
        # asked to imitate a lateral gait while moving along the platform depth.
        if self.cfg.standing_probability > 0.0:
            self.motion_ids[forward_ids] = self._moving_motion_count

    def _update_metrics(self):
        super()._update_metrics()
        forward_velocity = self.velocity_command[:, 0]
        self.metrics["target_forward_velocity"] = forward_velocity
        self.metrics["target_forward_speed"] = torch.abs(forward_velocity)


@configclass
class LateralForwardMicroAdjustMotionCommandCfg(LateralReferenceMotionCommandCfg):
    """Lateral reference command plus sparse x-axis micro-adjust commands."""

    class_type: type = LateralForwardMicroAdjustMotionCommand
    forward_probability: float = 0.25
    forward_velocity_range: tuple[float, float] = (-0.10, 0.10)
    min_abs_forward_velocity: float = 0.04


class StopHoldLateralReferenceMotionCommand(LateralReferenceMotionCommand):
    """Lateral reference command whose late hold phase commands zero speed."""

    cfg: StopHoldLateralReferenceMotionCommandCfg

    @property
    def velocity_command(self) -> torch.Tensor:
        command = super().velocity_command
        hold_phase = self.time_steps >= self.cfg.hold_start_step
        command[hold_phase, 1] = 0.0
        return command

    def _update_metrics(self):
        super()._update_metrics()
        target_velocity = self.velocity_command[:, 1]
        reference_velocity = self._reference_lateral_velocities[self.motion_ids]
        reference_velocity = torch.where(self.time_steps >= self.cfg.hold_start_step, 0.0, reference_velocity)
        self.metrics["target_lateral_velocity"] = target_velocity
        self.metrics["target_lateral_speed"] = torch.abs(target_velocity)
        self.metrics["reference_velocity_gap"] = torch.abs(target_velocity - reference_velocity)


@configclass
class StopHoldLateralReferenceMotionCommandCfg(LateralReferenceMotionCommandCfg):
    """Lateral reference command with a fixed zero-speed hold phase."""

    class_type: type = StopHoldLateralReferenceMotionCommand
    hold_start_step: int = 300


class SmoothStopHoldLateralReferenceMotionCommand(LateralReferenceMotionCommand):
    """Lateral command with a smooth stop phase before the final hold."""

    cfg: SmoothStopHoldLateralReferenceMotionCommandCfg

    @property
    def velocity_command(self) -> torch.Tensor:
        command = super().velocity_command
        time_steps = self.time_steps.to(dtype=torch.float32)
        settle = max(float(self.cfg.settle_steps), 1.0)
        t = torch.clamp((time_steps - float(self.cfg.stop_start_step)) / settle, 0.0, 1.0)
        alpha = t * t * (3.0 - 2.0 * t)
        command[:, 1] *= 1.0 - alpha
        return command

    def _update_metrics(self):
        super()._update_metrics()
        target_velocity = self.velocity_command[:, 1]
        reference_velocity = self._reference_lateral_velocities[self.motion_ids]
        time_steps = self.time_steps.to(dtype=torch.float32)
        settle = max(float(self.cfg.settle_steps), 1.0)
        t = torch.clamp((time_steps - float(self.cfg.stop_start_step)) / settle, 0.0, 1.0)
        alpha = t * t * (3.0 - 2.0 * t)
        reference_velocity = reference_velocity * (1.0 - alpha)
        self.metrics["target_lateral_velocity"] = target_velocity
        self.metrics["target_lateral_speed"] = torch.abs(target_velocity)
        self.metrics["reference_velocity_gap"] = torch.abs(target_velocity - reference_velocity)


@configclass
class SmoothStopHoldLateralReferenceMotionCommandCfg(LateralReferenceMotionCommandCfg):
    """Lateral reference command that decelerates smoothly before holding still."""

    class_type: type = SmoothStopHoldLateralReferenceMotionCommand
    stop_start_step: int = 300
    settle_steps: int = 25


class TransitionLateralReferenceMotionCommand(LateralReferenceMotionCommand):
    """Lateral command that trains move-to-zero transitions without teleporting.

    The first resample of an episode still uses ``MotionCommand`` reset logic to
    place the robot on the valid reference pose. Later command-manager
    resamples only switch the target/reference speed, so the policy experiences
    the real physical state after moving and must brake or hold from there.
    """

    def _resample_reference_only(self, env_ids_t: torch.Tensor):
        count = env_ids_t.numel()
        if count == 0:
            return

        magnitude = torch.empty(count, device=self.device).uniform_(
            self.cfg.min_abs_target_velocity,
            min(abs(self.cfg.target_velocity_range[0]), abs(self.cfg.target_velocity_range[1])),
        )
        sign = torch.where(
            torch.rand(count, device=self.device) < 0.5,
            -torch.ones_like(magnitude),
            torch.ones_like(magnitude),
        )
        target_velocity = magnitude * sign

        moving_reference_velocities = self._reference_lateral_velocities[: self._moving_motion_count]
        selected = torch.argmin(
            torch.abs(target_velocity.unsqueeze(1) - moving_reference_velocities.unsqueeze(0)),
            dim=1,
        )
        if self.cfg.standing_probability > 0.0:
            standing = torch.rand(count, device=self.device) < self.cfg.standing_probability
            selected[standing] = self._moving_motion_count
            target_velocity[standing] = 0.0

        self.motion_ids[env_ids_t] = selected
        self._target_lateral_velocities[env_ids_t] = target_velocity
        self.time_steps[env_ids_t] = 0
        self.stage[env_ids_t] = 0
        self.stable_counter[env_ids_t] = 0
        self.post_stable_counter[env_ids_t] = 0
        self.hold_elapsed_counter[env_ids_t] = 0
        self.metrics["sampling_entropy"][:] = 0.0
        self.metrics["sampling_top1_prob"][:] = 1.0
        self.metrics["sampling_top1_bin"][:] = 0.0
        self.metrics["sampling_late_fraction"][env_ids_t] = 0.0

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return

        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        first_episode_sample = self.command_counter[env_ids_t] == 0

        if torch.any(first_episode_sample):
            initial_env_ids = env_ids_t[first_episode_sample]
            LateralReferenceMotionCommand._resample_command(self, initial_env_ids)

        if torch.any(~first_episode_sample):
            transition_env_ids = env_ids_t[~first_episode_sample]
            self._resample_reference_only(transition_env_ids)


@configclass
class TransitionLateralReferenceMotionCommandCfg(LateralReferenceMotionCommandCfg):
    """Lateral reference command with in-episode speed transitions."""

    class_type: type = TransitionLateralReferenceMotionCommand


class PhaseLatchedStopLateralReferenceMotionCommand(TransitionLateralReferenceMotionCommand):
    """Train occasional move-to-zero stops while holding the current gait phase.

    A stop does not select the global standing reference or reset the robot.  It
    freezes the current reference pose, zeros its velocities, and commands zero
    lateral speed.  This gives the policy a physically reachable stop target
    from whichever gait phase was active when the command was released.
    """

    _STOP_PROBABILITY = 0.35

    def __init__(self, cfg: LateralReferenceMotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._holding_zero = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    @property
    def joint_vel(self) -> torch.Tensor:
        velocity = super().joint_vel.clone()
        velocity[self._holding_zero] = 0.0
        return velocity

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        velocity = super().body_lin_vel_w.clone()
        velocity[self._holding_zero] = 0.0
        return velocity

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        velocity = super().body_ang_vel_w.clone()
        velocity[self._holding_zero] = 0.0
        return velocity

    def _sample_moving_reference(self, env_ids_t: torch.Tensor):
        count = env_ids_t.numel()
        if count == 0:
            return

        magnitude = torch.empty(count, device=self.device).uniform_(
            self.cfg.min_abs_target_velocity,
            min(abs(self.cfg.target_velocity_range[0]), abs(self.cfg.target_velocity_range[1])),
        )
        sign = torch.where(
            torch.rand(count, device=self.device) < 0.5,
            -torch.ones_like(magnitude),
            torch.ones_like(magnitude),
        )
        target_velocity = magnitude * sign
        moving_reference_velocities = self._reference_lateral_velocities[: self._moving_motion_count]
        selected = torch.argmin(
            torch.abs(target_velocity.unsqueeze(1) - moving_reference_velocities.unsqueeze(0)),
            dim=1,
        )
        self.motion_ids[env_ids_t] = selected
        self._target_lateral_velocities[env_ids_t] = target_velocity
        self.time_steps[env_ids_t] = 0
        self._holding_zero[env_ids_t] = False

    def _resample_reference_only(self, env_ids_t: torch.Tensor):
        if env_ids_t.numel() == 0:
            return

        was_zero = self._holding_zero[env_ids_t]
        moving_ids = env_ids_t[~was_zero]
        restart_ids = env_ids_t[was_zero]

        # Every zero segment is followed by movement.  Moving segments stop
        # occasionally, keeping the zero-command share close to v17's 20%.
        if moving_ids.numel() > 0:
            stop = torch.rand(moving_ids.numel(), device=self.device) < self._STOP_PROBABILITY
            stop_ids = moving_ids[stop]
            continue_ids = moving_ids[~stop]
            self._target_lateral_velocities[stop_ids] = 0.0
            self._holding_zero[stop_ids] = True
            self._sample_moving_reference(continue_ids)
        self._sample_moving_reference(restart_ids)

        self.stage[env_ids_t] = 0
        self.stable_counter[env_ids_t] = 0
        self.post_stable_counter[env_ids_t] = 0
        self.hold_elapsed_counter[env_ids_t] = 0
        self.metrics["sampling_entropy"][:] = 0.0
        self.metrics["sampling_top1_prob"][:] = 1.0
        self.metrics["sampling_top1_bin"][:] = 0.0
        self.metrics["sampling_late_fraction"][env_ids_t] = 0.0

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        super()._resample_command(env_ids)
        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self._holding_zero[env_ids_t] = torch.abs(self._target_lateral_velocities[env_ids_t]) < 1.0e-6

    def _sync_relative_reference(self):
        anchor_pos = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos = robot_anchor_pos
        delta_pos[..., 2] = anchor_pos[..., 2]
        delta_ori = yaw_quat(quat_mul(robot_anchor_quat, quat_inv(anchor_quat)))
        self.body_quat_relative_w = quat_mul(delta_ori, self.body_quat_w)
        self.body_pos_relative_w = delta_pos + quat_apply(delta_ori, self.body_pos_w - anchor_pos)

    def _update_command(self):
        hold_mask = self._holding_zero.clone()
        hold_steps = self.time_steps.clone()

        # Prevent the base class from reaching the trajectory end and resampling
        # a frozen hold through its normal looping path.
        if torch.any(hold_mask):
            safe_last = max(self.motion.time_step_total - 2, 0)
            self.time_steps[hold_mask] = torch.clamp(self.time_steps[hold_mask], max=safe_last)
        super()._update_command()

        if torch.any(hold_mask):
            self.time_steps[hold_mask] = hold_steps[hold_mask]
            self._sync_relative_reference()


@configclass
class PhaseLatchedStopLateralReferenceMotionCommandCfg(LateralReferenceMotionCommandCfg):
    """Lateral command with occasional phase-latched zero-speed segments."""

    class_type: type = PhaseLatchedStopLateralReferenceMotionCommand


class StageGatedLateralReferenceMotionCommand(LateralReferenceMotionCommand):
    """Lateral reference command for stage2 without resetting robot state.

    The main ``motion`` command still owns the boarding reset and stage gate. This
    command only supplies a looping lateral reference, joint targets, and y-speed
    command after the boarding command reaches stage2.
    """

    cfg: StageGatedLateralReferenceMotionCommandCfg

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        magnitude = torch.empty(len(env_ids), device=self.device).uniform_(
            self.cfg.min_abs_target_velocity,
            min(abs(self.cfg.target_velocity_range[0]), abs(self.cfg.target_velocity_range[1])),
        )
        sign = torch.where(
            torch.rand(len(env_ids), device=self.device) < 0.5,
            -torch.ones_like(magnitude),
            torch.ones_like(magnitude),
        )
        target_velocity = magnitude * sign

        moving_reference_velocities = self._reference_lateral_velocities[: self._moving_motion_count]
        selected = torch.argmin(
            torch.abs(target_velocity.unsqueeze(1) - moving_reference_velocities.unsqueeze(0)),
            dim=1,
        )
        if self.cfg.standing_probability > 0.0:
            standing = torch.rand(len(env_ids), device=self.device) < self.cfg.standing_probability
            selected[standing] = self._moving_motion_count
            target_velocity[standing] = 0.0

        self.motion_ids[env_ids_t] = selected
        self._target_lateral_velocities[env_ids_t] = target_velocity
        self.time_steps[env_ids_t] = 0
        self.metrics["sampling_entropy"][:] = 0.0
        self.metrics["sampling_top1_prob"][:] = 1.0
        self.metrics["sampling_top1_bin"][:] = 0.0
        self.metrics["sampling_late_fraction"][env_ids_t] = 0.0

    def _update_command(self):
        stage_cmd: MotionCommand = self._env.command_manager.get_term(self.cfg.stage_command_name)
        enabled = stage_cmd.stage >= self.cfg.enabled_stage
        moving = enabled & (torch.abs(self._target_lateral_velocities) > 0.0)
        self.time_steps[moving] = (self.time_steps[moving] + 1) % self.motion.time_step_total
        self.time_steps[~enabled] = 0

        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)
        self.stage[:] = torch.where(enabled, self.cfg.enabled_stage, 0)
        self.metrics["stage"] = self.stage.float()
        self.metrics["ready_gate"] = enabled.float()


@configclass
class StageGatedLateralReferenceMotionCommandCfg(LateralReferenceMotionCommandCfg):
    """Stage2-only lateral reference that never writes reset state."""

    class_type: type = StageGatedLateralReferenceMotionCommand
    stage_command_name: str = "motion"
    enabled_stage: int = 2


class LateralRedMotionCommand(LateralReferenceMotionCommand):
    """Continuous lateral command with differential expert windows for RED.

    Reference motions are used for two purposes only:
    - reset every episode from their shared valid first frame;
    - sample short expert windows for the RED differential style prior.

    The robot is never assigned a nearest reference motion during the episode.
    RED receives only train-side differential descriptors:
    policy-window minus expert-window, and expert-window minus expert-window
    as the support distribution. The deployable actor observation is unchanged.
    """

    cfg: LateralRedMotionCommandCfg

    def __init__(self, cfg: LateralRedMotionCommandCfg, env: ManagerBasedRLEnv):
        if cfg.red_history_length < 2:
            raise ValueError("red_history_length must be at least 2.")
        if not cfg.style_joint_names:
            raise ValueError("style_joint_names must contain at least one joint.")
        if not cfg.style_body_names:
            raise ValueError("style_body_names must contain at least one body.")

        super().__init__(cfg, env)

        self._style_joint_ids, _ = self.robot.find_joints(cfg.style_joint_names, preserve_order=True)
        self._style_robot_body_ids, _ = self.robot.find_bodies(cfg.style_body_names, preserve_order=True)
        self._style_motion_body_ids = torch.tensor(
            [cfg.body_names.index(name) for name in cfg.style_body_names],
            dtype=torch.long,
            device=self.device,
        )
        self._style_joint_default = self._joint_pos_refs[0, 0, self._style_joint_ids].clone()
        self._red_demo_frame_features = self._build_red_demo_frame_features()
        self._red_frame_dim = self._red_demo_frame_features.shape[-1]
        self._red_policy_history = torch.zeros(
            self.num_envs,
            cfg.red_history_length,
            self._red_frame_dim,
            dtype=torch.float32,
            device=self.device,
        )
        self._reset_red_policy_history(torch.arange(self.num_envs, device=self.device))

    @property
    def red_policy_features(self) -> torch.Tensor:
        """Return policy-vs-expert differential features for RED."""
        mode = torch.sign(self._target_lateral_velocities)
        expert_ids = self._sample_red_motion_ids(mode)
        expert_windows = self._sample_red_windows(expert_ids)
        return self._red_window_differential_features(self._red_policy_history, expert_windows)

    @property
    def red_demo_features(self) -> torch.Tensor:
        """Return expert-vs-expert differential features for RED positives."""
        mode = torch.sign(self._target_lateral_velocities)
        candidate_ids = self._sample_red_motion_ids(mode)
        reference_ids = self._sample_red_motion_ids(mode)
        candidate_windows = self._sample_red_windows(candidate_ids)
        reference_windows = self._sample_red_windows(reference_ids)
        return self._red_window_differential_features(candidate_windows, reference_windows)

    def _sample_red_motion_ids(self, mode: torch.Tensor) -> torch.Tensor:
        motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        for mode_value in (-1.0, 0.0, 1.0):
            env_ids = torch.where(mode == mode_value)[0]
            if env_ids.numel() == 0:
                continue
            candidates = torch.where(torch.sign(self._reference_lateral_velocities) == mode_value)[0]
            if candidates.numel() == 0:
                raise RuntimeError(f"No RED expert motion is available for command mode {mode_value}.")
            selection = torch.randint(candidates.numel(), (env_ids.numel(),), device=self.device)
            motion_ids[env_ids] = candidates[selection]
        return motion_ids

    def _sample_red_windows(self, motion_ids: torch.Tensor) -> torch.Tensor:
        max_start = self.motion.time_step_total - self.cfg.red_history_length
        start = torch.randint(max_start + 1, (self.num_envs,), device=self.device)
        offsets = torch.arange(self.cfg.red_history_length, device=self.device)
        frame_ids = start.unsqueeze(1) + offsets.unsqueeze(0)
        return self._red_demo_frame_features[motion_ids.unsqueeze(1), frame_ids]

    def _red_window_differential_features(self, candidate: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """Encode both pose error and short-horizon motion error.

        The RED predictor is trained on expert-expert differences, so the reward
        describes an expert support region instead of a hand-weighted sum of
        posture, speed, heading, and smoothness terms.
        """
        frame_error = candidate - reference
        candidate_delta = candidate[:, 1:] - candidate[:, :-1]
        reference_delta = reference[:, 1:] - reference[:, :-1]
        temporal_error = candidate_delta - reference_delta
        return torch.cat([frame_error.flatten(start_dim=1), temporal_error.flatten(start_dim=1)], dim=-1)

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        max_target_velocity = min(abs(self.cfg.target_velocity_range[0]), abs(self.cfg.target_velocity_range[1]))
        magnitude = torch.empty(len(env_ids), device=self.device).uniform_(
            self.cfg.min_abs_target_velocity,
            max_target_velocity,
        )
        sign = torch.where(
            torch.rand(len(env_ids), device=self.device) < 0.5,
            -torch.ones_like(magnitude),
            torch.ones_like(magnitude),
        )
        target_velocity = magnitude * sign
        if self.cfg.standing_probability > 0.0:
            standing = torch.rand(len(env_ids), device=self.device) < self.cfg.standing_probability
            target_velocity[standing] = 0.0

        # All lateral references share the same valid first frame. Keep reset
        # independent of command speed and leave expert selection to RED.
        self.motion_ids[env_ids_t] = 0
        self._target_lateral_velocities[env_ids_t] = target_velocity
        MotionCommand._resample_command(self, env_ids)
        if hasattr(self, "_red_policy_history"):
            self._reset_red_policy_history(env_ids_t)

    def _update_command(self):
        # RED is deliberately not a trajectory player. The reference stays at
        # the reset frame while PPO controls continuous lateral speed.
        self.time_steps.zero_()
        self.stage.zero_()
        self._update_first_frame_relative_reference()
        self._append_red_policy_history()
        self.metrics["stage"].zero_()
        self.metrics["ready_gate"].zero_()

    def _update_metrics(self):
        MotionCommand._update_metrics(self)
        target_velocity = self.velocity_command[:, 1]
        self.metrics["target_lateral_velocity"] = target_velocity
        self.metrics["target_lateral_speed"] = torch.abs(target_velocity)
        # RED samples an expert support set rather than assigning a nearest
        # speed reference, so a per-environment reference gap is undefined.
        self.metrics["reference_velocity_gap"].zero_()

    def _update_first_frame_relative_reference(self):
        num_bodies = len(self.cfg.body_names)
        anchor_pos_w = self.anchor_pos_w[:, None, :].repeat(1, num_bodies, 1)
        anchor_quat_w = self.anchor_quat_w[:, None, :].repeat(1, num_bodies, 1)
        robot_anchor_pos_w = self.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1)
        robot_anchor_quat_w = self.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1)

        aligned_anchor_pos_w = robot_anchor_pos_w
        aligned_anchor_pos_w[..., 2] = anchor_pos_w[..., 2]
        yaw_delta_w = yaw_quat(quat_mul(robot_anchor_quat_w, quat_inv(anchor_quat_w)))
        self.body_quat_relative_w = quat_mul(yaw_delta_w, self.body_quat_w)
        self.body_pos_relative_w = aligned_anchor_pos_w + quat_apply(
            yaw_delta_w,
            self.body_pos_w - anchor_pos_w,
        )

    def _build_red_demo_frame_features(self) -> torch.Tensor:
        anchor_pos = self._body_pos_refs[:, :, self.motion_anchor_body_index]
        anchor_quat = self._body_quat_refs[:, :, self.motion_anchor_body_index]
        anchor_lin_vel = self._body_lin_vel_refs[:, :, self.motion_anchor_body_index]
        anchor_ang_vel = self._body_ang_vel_refs[:, :, self.motion_anchor_body_index]
        # Express every expert and policy orientation in one canonical frame
        # defined by the shared data first frame. This keeps RED sensitive to
        # heading drift without adding a separate heading reward.
        canonical_anchor_quat = self._body_quat_refs[
            0:1, 0:1, self.motion_anchor_body_index
        ].expand_as(anchor_quat)
        anchor_quat_relative = quat_mul(quat_inv(canonical_anchor_quat), anchor_quat)
        anchor_ori_relative = matrix_from_quat(anchor_quat_relative)[..., :2].flatten(start_dim=2)

        body_pos = self._body_pos_refs[:, :, self._style_motion_body_ids]
        body_quat = self._body_quat_refs[:, :, self._style_motion_body_ids]
        body_lin_vel = self._body_lin_vel_refs[:, :, self._style_motion_body_ids]
        body_ang_vel = self._body_ang_vel_refs[:, :, self._style_motion_body_ids]
        expanded_anchor_quat = anchor_quat.unsqueeze(2).expand(-1, -1, body_pos.shape[2], -1)
        body_pos_b = quat_apply_inverse(expanded_anchor_quat, body_pos - anchor_pos.unsqueeze(2))
        body_quat_b = quat_mul(quat_inv(expanded_anchor_quat), body_quat)
        body_ori_b = matrix_from_quat(body_quat_b)[..., :2].flatten(start_dim=2)
        body_lin_vel_b = quat_apply_inverse(expanded_anchor_quat, body_lin_vel - anchor_lin_vel.unsqueeze(2))
        body_ang_vel_b = quat_apply_inverse(expanded_anchor_quat, body_ang_vel - anchor_ang_vel.unsqueeze(2))

        gravity_w = torch.zeros_like(anchor_pos)
        gravity_w[..., 2] = -1.0
        projected_gravity = quat_apply_inverse(anchor_quat, gravity_w)
        base_lin_vel_b = quat_apply_inverse(anchor_quat, anchor_lin_vel)
        base_ang_vel_b = quat_apply_inverse(anchor_quat, anchor_ang_vel)
        base_lin_vel_xz_b = torch.cat([base_lin_vel_b[..., 0:1], base_lin_vel_b[..., 2:3]], dim=-1)
        speed_error = base_lin_vel_b[..., 1:2] - self._reference_lateral_velocities.view(-1, 1, 1)

        joint_pos = self._joint_pos_refs[:, :, self._style_joint_ids] - self._style_joint_default
        joint_vel = self._joint_vel_refs[:, :, self._style_joint_ids]
        command_mode = torch.sign(self._reference_lateral_velocities).view(-1, 1, 1)
        command_mode = command_mode.expand(-1, self.motion.time_step_total, -1)

        return torch.cat(
            [
                joint_pos,
                joint_vel,
                anchor_pos[..., 2:3],
                anchor_ori_relative,
                projected_gravity,
                body_pos_b.flatten(start_dim=2),
                body_ori_b,
                body_lin_vel_b.flatten(start_dim=2),
                body_ang_vel_b.flatten(start_dim=2),
                base_lin_vel_xz_b,
                base_ang_vel_b,
                speed_error,
                command_mode,
            ],
            dim=-1,
        )

    def _current_red_frame_features(self) -> torch.Tensor:
        anchor_pos = self.robot_anchor_pos_w
        anchor_quat = self.robot_anchor_quat_w
        actor_anchor_quat = self.actor_anchor_quat_w
        anchor_quat_relative = quat_mul(quat_inv(actor_anchor_quat), anchor_quat)
        anchor_ori_relative = matrix_from_quat(anchor_quat_relative)[..., :2].flatten(start_dim=1)
        body_pos = self.robot.data.body_pos_w[:, self._style_robot_body_ids]
        body_quat = self.robot.data.body_quat_w[:, self._style_robot_body_ids]
        body_lin_vel = self.robot.data.body_lin_vel_w[:, self._style_robot_body_ids]
        body_ang_vel = self.robot.data.body_ang_vel_w[:, self._style_robot_body_ids]
        expanded_anchor_quat = anchor_quat.unsqueeze(1).expand(-1, body_pos.shape[1], -1)
        body_pos_b = quat_apply_inverse(expanded_anchor_quat, body_pos - anchor_pos.unsqueeze(1))
        body_quat_b = quat_mul(quat_inv(expanded_anchor_quat), body_quat)
        body_ori_b = matrix_from_quat(body_quat_b)[..., :2].flatten(start_dim=1)
        body_lin_vel_b = quat_apply_inverse(
            expanded_anchor_quat,
            body_lin_vel - self.robot_anchor_lin_vel_w.unsqueeze(1),
        )
        body_ang_vel_b = quat_apply_inverse(
            expanded_anchor_quat,
            body_ang_vel - self.robot_anchor_ang_vel_w.unsqueeze(1),
        )

        joint_pos = self.robot.data.joint_pos[:, self._style_joint_ids] - self._style_joint_default
        joint_vel = self.robot.data.joint_vel[:, self._style_joint_ids]
        projected_gravity = quat_apply_inverse(anchor_quat, self.robot.data.GRAVITY_VEC_W)
        base_lin_vel_b = quat_apply_inverse(anchor_quat, self.robot_anchor_lin_vel_w)
        base_ang_vel_b = quat_apply_inverse(anchor_quat, self.robot_anchor_ang_vel_w)
        base_lin_vel_xz_b = torch.cat([base_lin_vel_b[:, 0:1], base_lin_vel_b[:, 2:3]], dim=-1)
        speed_error = base_lin_vel_b[:, 1:2] - self._target_lateral_velocities.unsqueeze(-1)
        command_mode = torch.sign(self._target_lateral_velocities).unsqueeze(-1)

        return torch.cat(
            [
                joint_pos,
                joint_vel,
                anchor_pos[:, 2:3],
                anchor_ori_relative,
                projected_gravity,
                body_pos_b.flatten(start_dim=1),
                body_ori_b,
                body_lin_vel_b.flatten(start_dim=1),
                body_ang_vel_b.flatten(start_dim=1),
                base_lin_vel_xz_b,
                base_ang_vel_b,
                speed_error,
                command_mode,
            ],
            dim=-1,
        )

    def _append_red_policy_history(self):
        self._red_policy_history[:, :-1] = self._red_policy_history[:, 1:].clone()
        self._red_policy_history[:, -1] = self._current_red_frame_features()

    def _reset_red_policy_history(self, env_ids: torch.Tensor):
        reset_frame = self._current_red_frame_features()[env_ids].clone()
        self._red_policy_history[env_ids] = reset_frame.unsqueeze(1).expand(-1, self.cfg.red_history_length, -1)


@configclass
class LateralRedMotionCommandCfg(LateralReferenceMotionCommandCfg):
    """Configuration for differential RED-guided lateral locomotion."""

    class_type: type = LateralRedMotionCommand

    style_joint_names: list[str] = MISSING
    style_body_names: list[str] = MISSING
    red_history_length: int = 4


class DecoupledPlanarVelocityCommand(UniformVelocityCommand):
    """Sample focused lateral or yaw commands for the residual movement skill."""

    cfg: DecoupledPlanarVelocityCommandCfg

    def __init__(self, cfg: DecoupledPlanarVelocityCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        if cfg.rel_yaw_envs < 0.0 or cfg.rel_combined_envs < 0.0 or cfg.rel_yaw_envs + cfg.rel_combined_envs > 1.0:
            raise ValueError("rel_yaw_envs and rel_combined_envs must be non-negative and sum to at most 1.0.")
        if cfg.post_ready_zero_command_steps < 0:
            raise ValueError("post_ready_zero_command_steps must be non-negative.")
        if cfg.yaw_only_inactive_axis_noise_range[0] > cfg.yaw_only_inactive_axis_noise_range[1]:
            raise ValueError(
                "yaw_only_inactive_axis_noise_range lower bound must not exceed upper bound, "
                f"got {cfg.yaw_only_inactive_axis_noise_range}."
            )
        self._warmup_active = self._env.common_step_counter < self.cfg.zero_command_warmup_steps
        self._post_ready_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._ready_command_started = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.metrics["ready_fraction"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["ready_error_vel_y"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["ready_error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["ready_error_heading_yaw"] = torch.zeros(self.num_envs, device=self.device)

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        if len(env_ids) == 0:
            return

        env_ids_t = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        mode = torch.rand(len(env_ids_t), device=self.device)
        yaw_only = mode < self.cfg.rel_yaw_envs
        combined = (mode >= self.cfg.rel_yaw_envs) & (
            mode < self.cfg.rel_yaw_envs + self.cfg.rel_combined_envs
        )
        lateral_only = ~(yaw_only | combined)

        self.vel_command_b[env_ids_t[lateral_only], 2] = 0.0
        yaw_only_ids = env_ids_t[yaw_only]
        self.vel_command_b[yaw_only_ids, 0] = torch.empty(yaw_only_ids.numel(), device=self.device).uniform_(
            *self.cfg.yaw_only_inactive_axis_noise_range
        )
        self.vel_command_b[yaw_only_ids, 1] = torch.empty(yaw_only_ids.numel(), device=self.device).uniform_(
            *self.cfg.yaw_only_inactive_axis_noise_range
        )

        lateral_ids = env_ids_t[lateral_only | combined]
        yaw_ids = env_ids_t[yaw_only | combined]
        self._sample_nonzero_component(
            self.vel_command_b[:, 1],
            lateral_ids,
            self.cfg.ranges.lin_vel_y,
            self.cfg.min_abs_lin_vel_y,
        )
        self._sample_nonzero_component(
            self.vel_command_b[:, 2],
            yaw_ids,
            self.cfg.ranges.ang_vel_z,
            self.cfg.min_abs_ang_vel_z,
        )
        if self._env.common_step_counter < self.cfg.zero_command_warmup_steps:
            self.vel_command_b[env_ids_t] = 0.0

    def _update_command(self):
        warmup_active = self._env.common_step_counter < self.cfg.zero_command_warmup_steps
        if self._warmup_active and not warmup_active:
            self._resample(torch.arange(self.num_envs, device=self.device))
        super()._update_command()

        motion_cmd = self._env.command_manager.get_term(self.cfg.motion_command_name)
        ready = motion_cmd.ready
        self._post_ready_counter[~ready] = 0
        self._ready_command_started[~ready] = False

        waiting_for_command = ready & ~self._ready_command_started
        self._post_ready_counter[waiting_for_command] += 1
        start_command = waiting_for_command & (
            self._post_ready_counter >= self.cfg.post_ready_zero_command_steps
        )
        start_ids = torch.where(start_command)[0]
        if start_ids.numel() > 0:
            self._resample(start_ids)
            self._ready_command_started[start_ids] = True
            # Apply standing/heading mode masks to the freshly sampled command
            # immediately, avoiding a one-step command spike.
            super()._update_command()

        # Before ready and during the post-ready settling window, the residual
        # controller sees and tracks a true zero command.
        self.vel_command_b[~self._ready_command_started] = 0.0
        if warmup_active:
            self.vel_command_b.zero_()
        self._warmup_active = warmup_active

    def _update_metrics(self):
        motion_cmd = self._env.command_manager.get_term(self.cfg.motion_command_name)
        ready = motion_cmd.ready.to(self.vel_command_b.dtype)
        episode_steps = max(float(self._env.max_episode_length), 1.0)

        yaw_only_quat = yaw_quat(self.robot.data.root_quat_w)
        lin_vel_proj_b = quat_apply_inverse(yaw_only_quat, self.robot.data.root_lin_vel_w)
        error_y = torch.abs(self.vel_command_b[:, 1] - lin_vel_proj_b[:, 1])
        error_yaw = torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_ang_vel_w[:, 2])
        error_heading_yaw = quat_error_magnitude(
            yaw_quat(motion_cmd.ready_anchor_quat_w),
            yaw_quat(motion_cmd.robot_anchor_quat_w),
        )

        # Keep the inherited metric names, but make them describe only the phase
        # in which the residual controller can actually affect the robot.
        self.metrics["error_vel_xy"] += error_y * ready / episode_steps
        self.metrics["error_vel_yaw"] += error_yaw * ready / episode_steps
        self.metrics["ready_fraction"] += ready / episode_steps
        self.metrics["ready_error_vel_y"] += error_y * ready / episode_steps
        self.metrics["ready_error_vel_yaw"] += error_yaw * ready / episode_steps
        self.metrics["ready_error_heading_yaw"] += error_heading_yaw * ready / episode_steps

    def _sample_nonzero_component(
        self,
        buffer: torch.Tensor,
        env_ids: torch.Tensor,
        bounds: tuple[float, float],
        min_abs: float,
    ):
        if env_ids.numel() == 0:
            return
        low, high = bounds
        can_sample_negative = low <= -min_abs
        can_sample_positive = high >= min_abs
        if not can_sample_negative and not can_sample_positive:
            raise ValueError(f"Velocity bounds {bounds} cannot satisfy minimum absolute command {min_abs}.")

        if can_sample_negative and can_sample_positive:
            sample_positive = torch.rand(env_ids.numel(), device=self.device) >= 0.5
        else:
            sample_positive = torch.full(
                (env_ids.numel(),),
                can_sample_positive,
                dtype=torch.bool,
                device=self.device,
            )

        positive_values = min_abs + torch.rand(env_ids.numel(), device=self.device) * (high - min_abs)
        negative_values = -(min_abs + torch.rand(env_ids.numel(), device=self.device) * (-low - min_abs))
        values = torch.where(sample_positive, positive_values, negative_values)
        buffer[env_ids] = values


@configclass
class DecoupledPlanarVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for focused lateral/yaw residual-skill commands."""

    class_type: type = DecoupledPlanarVelocityCommand

    motion_command_name: str = "motion"
    rel_yaw_envs: float = 0.45
    rel_combined_envs: float = 0.0
    min_abs_lin_vel_y: float = 0.06
    min_abs_ang_vel_z: float = 0.08
    zero_command_warmup_steps: int = 0
    post_ready_zero_command_steps: int = 0
    yaw_only_inactive_axis_noise_range: tuple[float, float] = (0.0, 0.0)
