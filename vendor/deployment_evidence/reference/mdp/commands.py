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


class DecoupledPlanarVelocityCommand(UniformVelocityCommand):
    """Sample focused lateral or yaw commands for the residual movement skill."""

    cfg: DecoupledPlanarVelocityCommandCfg

    def __init__(self, cfg: DecoupledPlanarVelocityCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        if cfg.rel_yaw_envs < 0.0 or cfg.rel_combined_envs < 0.0 or cfg.rel_yaw_envs + cfg.rel_combined_envs > 1.0:
            raise ValueError("rel_yaw_envs and rel_combined_envs must be non-negative and sum to at most 1.0.")
        if cfg.post_ready_zero_command_steps < 0:
            raise ValueError("post_ready_zero_command_steps must be non-negative.")
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
        self.vel_command_b[env_ids_t[yaw_only], 1] = 0.0

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
