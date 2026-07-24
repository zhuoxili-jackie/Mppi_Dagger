# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

import torch

from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from robot_lab.tasks.manager_based.beyondmimic.mdp.commands import MotionCommand
from robot_lab.tasks.manager_based.beyondmimic.mdp.observations import motion_anchor_ori_b


class BaselineResidualJointAction(ActionTerm):
    cfg: BaselineResidualJointActionCfg
    _asset: Articulation

    def __init__(self, cfg: BaselineResidualJointActionCfg, env):
        super().__init__(cfg, env)

        self._leg_joint_ids, self._leg_joint_names = self._asset.find_joints(cfg.leg_joint_names, preserve_order=True)
        self._wheel_joint_ids, self._wheel_joint_names = self._asset.find_joints(
            cfg.wheel_joint_names, preserve_order=True
        )
        self._wheel_joint_ids_all_order, _ = self._asset.find_joints(cfg.wheel_joint_names, preserve_order=False)

        self._num_baseline_actions = len(self._leg_joint_ids) + len(self._wheel_joint_ids)
        if self._num_baseline_actions != 16:
            raise RuntimeError(f"Expected 16 baseline actions, got {self._num_baseline_actions}.")

        baseline_policy_path = os.path.expandvars(os.path.expanduser(cfg.baseline_policy_path))
        if not os.path.isabs(baseline_policy_path):
            baseline_policy_path = os.path.abspath(baseline_policy_path)
        if not os.path.isfile(baseline_policy_path):
            raise FileNotFoundError(
                f"Frozen baseline JIT policy not found: {baseline_policy_path}. "
                "Set PCBC_BASELINE_POLICY_PATH to the exported baseline policy.pt."
            )

        self._baseline_policy = torch.jit.load(baseline_policy_path, map_location=self.device)
        self._baseline_policy.eval()
        for param in self._baseline_policy.parameters():
            param.requires_grad_(False)

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._prev_raw_actions = torch.zeros_like(self._raw_actions)

        self._baseline_raw_actions = torch.zeros(self.num_envs, self._num_baseline_actions, device=self.device)
        self._latched_baseline_raw_actions = torch.zeros_like(self._baseline_raw_actions)
        self._baseline_latched = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.bool)
        self._residual_raw_actions = torch.zeros_like(self._baseline_raw_actions)
        self._prev_residual_raw_actions = torch.zeros_like(self._baseline_raw_actions)
        self._final_raw_actions = torch.zeros_like(self._baseline_raw_actions)
        self._prev_final_raw_actions = torch.zeros_like(self._baseline_raw_actions)
        self._ready = torch.zeros(self.num_envs, 1, device=self.device)

        self._leg_scale = torch.empty(1, len(self._leg_joint_ids), device=self.device)
        for index, joint_name in enumerate(self._leg_joint_names):
            self._leg_scale[:, index] = 0.125 if "hip_joint" in joint_name else 0.25
        self._wheel_scale = torch.full((1, len(self._wheel_joint_ids)), 5.0, device=self.device)

    @property
    def action_dim(self) -> int:
        return 16

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._final_raw_actions

    @property
    def baseline_raw_actions(self) -> torch.Tensor:
        return self._baseline_raw_actions

    @property
    def residual_raw_actions(self) -> torch.Tensor:
        return self._residual_raw_actions

    @property
    def prev_residual_raw_actions(self) -> torch.Tensor:
        return self._prev_residual_raw_actions

    @property
    def final_raw_actions(self) -> torch.Tensor:
        return self._final_raw_actions

    @property
    def ready(self) -> torch.Tensor:
        return self._ready

    def process_actions(self, actions: torch.Tensor):
        self._prev_raw_actions[:] = self._raw_actions
        self._prev_residual_raw_actions[:] = self._residual_raw_actions
        self._prev_final_raw_actions[:] = self._final_raw_actions
        self._raw_actions[:] = actions

        baseline_obs = self._compute_baseline_observations()
        with torch.no_grad():
            baseline_actions = self._baseline_policy(baseline_obs)
        if baseline_actions.shape[-1] != self._num_baseline_actions:
            raise RuntimeError(
                f"Frozen baseline policy returned {baseline_actions.shape[-1]} actions, "
                f"expected {self._num_baseline_actions}."
            )
        self._baseline_raw_actions[:] = baseline_actions

        self._residual_raw_actions[:] = self._raw_actions
        ready_bool = self._compute_ready_mask() > 0.0
        just_ready = ready_bool & ~self._baseline_latched
        self._latched_baseline_raw_actions[:] = torch.where(
            just_ready,
            self._baseline_raw_actions,
            self._latched_baseline_raw_actions,
        )
        self._baseline_latched[:] = self._baseline_latched | ready_bool
        residual_active = self._baseline_latched
        self._ready[:] = residual_active.to(dtype=self._ready.dtype)

        residual_scale = torch.cat(
            [
                torch.full((1, len(self._leg_joint_ids)), self.cfg.leg_residual_scale, device=self.device),
                torch.full((1, len(self._wheel_joint_ids)), self.cfg.wheel_residual_scale, device=self.device),
            ],
            dim=1,
        )
        baseline_for_final = torch.where(
            residual_active,
            self._latched_baseline_raw_actions,
            self._baseline_raw_actions,
        )
        final_actions = baseline_for_final + self._ready * residual_scale * self._residual_raw_actions
        self._final_raw_actions[:] = torch.clamp(
            final_actions,
            min=-self.cfg.final_action_clip,
            max=self.cfg.final_action_clip,
        )

    def apply_actions(self):
        leg_raw = self._final_raw_actions[:, : len(self._leg_joint_ids)]
        wheel_raw = self._final_raw_actions[:, len(self._leg_joint_ids) :]

        leg_targets = self._asset.data.default_joint_pos[:, self._leg_joint_ids] + leg_raw * self._leg_scale
        wheel_targets = self._asset.data.default_joint_vel[:, self._wheel_joint_ids] + wheel_raw * self._wheel_scale

        self._asset.set_joint_position_target(leg_targets, joint_ids=self._leg_joint_ids)
        self._asset.set_joint_velocity_target(wheel_targets, joint_ids=self._wheel_joint_ids)

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            self._raw_actions.zero_()
            self._prev_raw_actions.zero_()
            self._baseline_raw_actions.zero_()
            self._latched_baseline_raw_actions.zero_()
            self._baseline_latched.zero_()
            self._residual_raw_actions.zero_()
            self._prev_residual_raw_actions.zero_()
            self._final_raw_actions.zero_()
            self._prev_final_raw_actions.zero_()
            self._ready.zero_()
            return

        self._raw_actions[env_ids] = 0.0
        self._prev_raw_actions[env_ids] = 0.0
        self._baseline_raw_actions[env_ids] = 0.0
        self._latched_baseline_raw_actions[env_ids] = 0.0
        self._baseline_latched[env_ids] = False
        self._residual_raw_actions[env_ids] = 0.0
        self._prev_residual_raw_actions[env_ids] = 0.0
        self._final_raw_actions[env_ids] = 0.0
        self._prev_final_raw_actions[env_ids] = 0.0
        self._ready[env_ids] = 0.0

    def _compute_baseline_observations(self) -> torch.Tensor:
        command = self._env.command_manager.get_command(self.cfg.motion_command_name)
        anchor_ori = motion_anchor_ori_b(self._env, command_name=self.cfg.motion_command_name)
        base_ang_vel = self._asset.data.root_ang_vel_b

        joint_pos_rel = self._asset.data.joint_pos - self._asset.data.default_joint_pos
        joint_pos_rel = joint_pos_rel.clone()
        joint_pos_rel[:, self._wheel_joint_ids_all_order] = 0.0
        joint_vel_rel = self._asset.data.joint_vel - self._asset.data.default_joint_vel

        zero_velocity_command = torch.zeros(self.num_envs, 3, device=self.device)
        # The exported baseline was trained with one trailing constant-zero scalar.
        # Keep it only inside the frozen baseline branch to preserve the 93-D input contract.
        zero_legacy_stage = torch.zeros(self.num_envs, 1, device=self.device)

        obs = torch.cat(
            [
                command,
                anchor_ori,
                base_ang_vel,
                joint_pos_rel,
                joint_vel_rel,
                self._prev_final_raw_actions,
                zero_velocity_command,
                zero_legacy_stage,
            ],
            dim=1,
        )
        if obs.shape[-1] != 93:
            raise RuntimeError(f"Expected frozen baseline observation dim 93, got {obs.shape[-1]}.")
        return obs

    def _compute_ready_mask(self) -> torch.Tensor:
        motion_cmd: MotionCommand = self._env.command_manager.get_term(self.cfg.motion_command_name)
        return motion_cmd.ready.to(torch.float32).unsqueeze(-1)


@configclass
class BaselineResidualJointActionCfg(ActionTermCfg):
    """Frozen-baseline action plus a learned residual delta.

    The policy action is 16-D:
    - first 12 values: leg joint-position residuals in the baseline raw-action space
    - next 4 values: wheel joint-velocity residuals in the baseline raw-action space

    The action sent to the robot keeps the original baseline layout:
    12 leg position targets and 4 wheel velocity targets.
    """

    class_type: type[ActionTerm] = BaselineResidualJointAction

    asset_name: str = ""
    baseline_policy_path: str = ""
    motion_command_name: str = "motion"

    leg_joint_names: list[str] = ()
    wheel_joint_names: list[str] = ()

    leg_residual_scale: float = 0.35
    wheel_residual_scale: float = 0.35
    final_action_clip: float = 100.0
