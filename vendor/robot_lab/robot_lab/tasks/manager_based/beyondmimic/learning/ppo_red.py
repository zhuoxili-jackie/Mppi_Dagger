# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from rsl_rl.algorithms import PPO
from tensordict import TensorDict

from .red import RandomExpertDistillation


class PPORED(PPO):
    """PPO augmented with a Random Expert Distillation prior."""

    def __init__(
        self,
        policy,
        storage,
        red_cfg: dict,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(policy, storage, device=device, **kwargs)

        red_cfg = dict(red_cfg)
        self.red_policy_group = red_cfg.pop("policy_group", "red_policy")
        self.red_demo_group = red_cfg.pop("demo_group", "red_demo")
        self.red_reward_weight = float(red_cfg.pop("reward_weight", 1.0))
        self.red_reward_warmup_steps = int(red_cfg.pop("reward_warmup_steps", 0))
        self.red_learning_rate = float(red_cfg.pop("learning_rate", 1.0e-3))
        self.red_num_learning_epochs = int(red_cfg.pop("num_learning_epochs", 2))
        self.red_num_mini_batches = int(red_cfg.pop("num_mini_batches", 4))
        self.red_contrastive_margin = float(red_cfg.pop("contrastive_margin", 0.0))
        self.red_contrastive_coef = float(red_cfg.pop("contrastive_coef", 0.0))
        self.red_contrastive_min_delta_rms = float(red_cfg.pop("contrastive_min_delta_rms", 0.0))
        self.red_normalization_source = red_cfg.pop("normalization_source", "demo")
        if self.red_contrastive_margin < 0.0:
            raise ValueError("contrastive_margin must be non-negative.")
        if self.red_contrastive_coef < 0.0:
            raise ValueError("contrastive_coef must be non-negative.")
        if self.red_contrastive_min_delta_rms < 0.0:
            raise ValueError("contrastive_min_delta_rms must be non-negative.")
        if self.red_normalization_source not in {"demo", "policy", "demo_policy"}:
            raise ValueError("normalization_source must be 'demo', 'policy', or 'demo_policy'.")

        num_states = storage.observations[self.red_policy_group].shape[-1]
        if storage.observations[self.red_demo_group].shape[-1] != num_states:
            raise ValueError("RED policy and expert observations must have identical dimensions.")

        self.red = RandomExpertDistillation(num_states=num_states, device=device, **red_cfg)
        self.red_optimizer = optim.Adam(self.red.predictor.parameters(), lr=self.red_learning_rate)
        self.red_step_counter = 0
        self.red_style_rewards = torch.zeros(storage.num_envs, device=device)
        self.red_policy_errors = torch.zeros(storage.num_envs, device=device)

    @property
    def current_red_weight(self) -> float:
        if self.red_reward_warmup_steps <= 0:
            return self.red_reward_weight
        progress = min(self.red_step_counter / self.red_reward_warmup_steps, 1.0)
        return self.red_reward_weight * progress

    def process_env_step(
        self,
        obs: TensorDict,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict[str, torch.Tensor],
    ) -> None:
        self.red_step_counter += 1
        style_reward, policy_error = self.red.style_reward(obs[self.red_policy_group])
        self.red_style_rewards = style_reward
        self.red_policy_errors = policy_error
        super().process_env_step(obs, rewards + self.current_red_weight * style_reward, dones, extras)

    def update(self) -> dict[str, float]:
        demo_states = self.storage.observations[self.red_demo_group].flatten(0, 1)
        policy_states = self.storage.observations[self.red_policy_group].flatten(0, 1)
        if self.red_normalization_source == "demo_policy":
            self.red.update_normalization(torch.cat([demo_states, policy_states.detach()], dim=0))
        elif self.red_normalization_source == "policy":
            self.red.update_normalization(policy_states.detach())
        else:
            self.red.update_normalization(demo_states)

        batch_size = demo_states.shape[0]
        mini_batch_size = max(batch_size // self.red_num_mini_batches, 1)
        mean_red_loss = 0.0
        mean_red_demo_loss = 0.0
        mean_red_contrastive_loss = 0.0
        red_updates = 0
        for _ in range(self.red_num_learning_epochs):
            indices = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, mini_batch_size):
                batch_indices = indices[start : start + mini_batch_size]
                demo_batch = demo_states[batch_indices]
                policy_batch = policy_states[batch_indices]
                demo_loss = self.red.prediction_error(demo_batch).mean()
                contrastive_loss = torch.zeros((), device=self.device)

                if self.red_contrastive_coef > 0.0 and self.red_contrastive_margin > 0.0:
                    policy_error = self.red.prediction_error(policy_batch)
                    if self.red_contrastive_min_delta_rms > 0.0:
                        policy_delta_rms = torch.sqrt(torch.mean(torch.square(policy_batch.detach()), dim=-1))
                        policy_error = policy_error[policy_delta_rms > self.red_contrastive_min_delta_rms]
                    if policy_error.numel() > 0:
                        contrastive_loss = torch.relu(self.red_contrastive_margin - policy_error).square().mean()

                red_loss = demo_loss + self.red_contrastive_coef * contrastive_loss

                self.red_optimizer.zero_grad()
                red_loss.backward()
                if self.is_multi_gpu:
                    for parameter in self.red.predictor.parameters():
                        if parameter.grad is not None:
                            torch.distributed.all_reduce(parameter.grad, op=torch.distributed.ReduceOp.SUM)
                            parameter.grad /= self.gpu_world_size
                nn.utils.clip_grad_norm_(self.red.predictor.parameters(), self.max_grad_norm)
                self.red_optimizer.step()

                mean_red_loss += red_loss.item()
                mean_red_demo_loss += demo_loss.item()
                mean_red_contrastive_loss += contrastive_loss.item()
                red_updates += 1

        with torch.no_grad():
            demo_error = self.red.prediction_error(demo_states).mean().item()
            policy_error = self.red.prediction_error(policy_states).mean().item()
            style_reward = self.red.style_reward(policy_states)[0].mean().item()

        loss_dict = super().update()
        loss_dict.update(
            {
                "red_predictor": mean_red_loss / max(red_updates, 1),
                "red_expert": mean_red_demo_loss / max(red_updates, 1),
                "red_contrastive": mean_red_contrastive_loss / max(red_updates, 1),
                "red_demo_error": demo_error,
                "red_policy_error": policy_error,
                "red_style_reward": style_reward,
                "red_reward_weight": self.current_red_weight,
            }
        )
        return loss_dict

    def broadcast_parameters(self) -> None:
        super().broadcast_parameters()
        if self.is_multi_gpu:
            state = [
                self.red.predictor.state_dict(),
                self.red.target.state_dict(),
                self.red.state_normalizer.state_dict(),
            ]
            torch.distributed.broadcast_object_list(state, src=0)
            self.red.predictor.load_state_dict(state[0])
            self.red.target.load_state_dict(state[1])
            self.red.state_normalizer.load_state_dict(state[2])
