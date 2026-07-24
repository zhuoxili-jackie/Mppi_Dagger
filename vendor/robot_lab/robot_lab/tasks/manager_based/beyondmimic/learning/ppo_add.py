# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from rsl_rl.algorithms import PPO
from tensordict import TensorDict

from .add import DifferentialDiscriminator, DifferentialReplayBuffer


class PPOADD(PPO):
    """PPO augmented with an Adversarial Differential Discriminator."""

    def __init__(
        self,
        policy,
        storage,
        add_cfg: dict,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(policy, storage, device=device, **kwargs)

        add_cfg = dict(add_cfg)
        self.add_policy_group = add_cfg.pop("policy_group", "add_policy")
        self.add_reward_weight = float(add_cfg.pop("reward_weight", 1.0))
        self.add_reward_warmup_steps = int(add_cfg.pop("reward_warmup_steps", 0))
        self.add_reward_scale = float(add_cfg.pop("reward_scale", 1.0))
        self.add_learning_rate = float(add_cfg.pop("learning_rate", 2.5e-4))
        self.add_num_learning_epochs = int(add_cfg.pop("num_learning_epochs", 2))
        self.add_num_mini_batches = int(add_cfg.pop("num_mini_batches", 4))
        self.add_replay_capacity = int(add_cfg.pop("replay_capacity", 200_000))
        self.add_replay_samples = int(add_cfg.pop("replay_samples", 50_000))
        self.add_grad_penalty = float(add_cfg.pop("grad_penalty", 2.0))
        self.add_logit_reg = float(add_cfg.pop("logit_reg", 1.0e-2))
        self.add_positive_noise_std = float(add_cfg.pop("positive_noise_std", 0.0))

        if self.add_reward_scale <= 0.0:
            raise ValueError("reward_scale must be positive.")
        if self.add_replay_capacity < 0:
            raise ValueError("replay_capacity must be non-negative.")
        if self.add_replay_samples < 0:
            raise ValueError("replay_samples must be non-negative.")
        if self.add_grad_penalty < 0.0:
            raise ValueError("grad_penalty must be non-negative.")
        if self.add_logit_reg < 0.0:
            raise ValueError("logit_reg must be non-negative.")
        if self.add_positive_noise_std < 0.0:
            raise ValueError("positive_noise_std must be non-negative.")

        num_states = storage.observations[self.add_policy_group].shape[-1]
        self.add = DifferentialDiscriminator(num_states=num_states, device=device, **add_cfg)
        self.add_optimizer = optim.Adam(self.add.parameters(), lr=self.add_learning_rate)
        self.add_replay = DifferentialReplayBuffer(self.add_replay_capacity, num_states, device)
        self.add_step_counter = 0
        self.add_rewards = torch.zeros(storage.num_envs, device=device)
        self.add_logits = torch.zeros(storage.num_envs, device=device)

    @property
    def current_add_weight(self) -> float:
        if self.add_reward_warmup_steps <= 0:
            return self.add_reward_weight
        progress = min(self.add_step_counter / self.add_reward_warmup_steps, 1.0)
        return self.add_reward_weight * progress

    def process_env_step(
        self,
        obs: TensorDict,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict[str, torch.Tensor],
    ) -> None:
        self.add_step_counter += 1
        add_reward, add_logits = self.add.imitation_reward(obs[self.add_policy_group], self.add_reward_scale)
        self.add_rewards = add_reward
        self.add_logits = add_logits
        super().process_env_step(obs, rewards + self.current_add_weight * add_reward, dones, extras)

    def update(self) -> dict[str, float]:
        policy_deltas = self.storage.observations[self.add_policy_group].flatten(0, 1).detach()
        self.add.update_normalization(policy_deltas)
        self.add_replay.insert(policy_deltas)

        mean_add_loss = 0.0
        mean_add_pos_loss = 0.0
        mean_add_neg_loss = 0.0
        mean_add_grad_penalty = 0.0
        mean_add_logit_reg = 0.0
        mean_add_pos_acc = 0.0
        mean_add_neg_acc = 0.0
        add_updates = 0

        batch_size = policy_deltas.shape[0]
        mini_batch_size = max(batch_size // self.add_num_mini_batches, 1)
        for _ in range(self.add_num_learning_epochs):
            indices = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, mini_batch_size):
                batch_indices = indices[start : start + mini_batch_size]
                negative_batch = policy_deltas[batch_indices]
                replay_batch = self.add_replay.sample(min(self.add_replay_samples, negative_batch.shape[0]))
                if replay_batch is not None:
                    negative_batch = torch.cat([negative_batch, replay_batch], dim=0)

                positive_batch = torch.zeros_like(negative_batch)
                if self.add_positive_noise_std > 0.0:
                    positive_batch = positive_batch + torch.randn_like(positive_batch) * self.add_positive_noise_std

                positive_norm = self.add.normalize(positive_batch).detach().requires_grad_(True)
                negative_norm = self.add.normalize(negative_batch).detach().requires_grad_(True)

                positive_logits = self.add.forward_normalized(positive_norm)
                negative_logits = self.add.forward_normalized(negative_norm)

                positive_loss = nn.functional.binary_cross_entropy_with_logits(
                    positive_logits,
                    torch.ones_like(positive_logits),
                )
                negative_loss = nn.functional.binary_cross_entropy_with_logits(
                    negative_logits,
                    torch.zeros_like(negative_logits),
                )
                add_loss = 0.5 * (positive_loss + negative_loss)

                grad_penalty = torch.zeros((), device=self.device)
                if self.add_grad_penalty > 0.0:
                    positive_grad = torch.autograd.grad(
                        positive_logits,
                        positive_norm,
                        grad_outputs=torch.ones_like(positive_logits),
                        create_graph=True,
                        retain_graph=True,
                        only_inputs=True,
                    )[0]
                    negative_grad = torch.autograd.grad(
                        negative_logits,
                        negative_norm,
                        grad_outputs=torch.ones_like(negative_logits),
                        create_graph=True,
                        retain_graph=True,
                        only_inputs=True,
                    )[0]
                    grad_penalty = 0.5 * (
                        torch.square(positive_grad).sum(dim=-1).mean()
                        + torch.square(negative_grad).sum(dim=-1).mean()
                    )
                    add_loss = add_loss + self.add_grad_penalty * grad_penalty

                logit_reg = torch.square(self.add.logit_weights()).sum()
                if self.add_logit_reg > 0.0:
                    add_loss = add_loss + self.add_logit_reg * logit_reg

                self.add_optimizer.zero_grad()
                add_loss.backward()
                if self.is_multi_gpu:
                    for parameter in self.add.parameters():
                        if parameter.grad is not None:
                            torch.distributed.all_reduce(parameter.grad, op=torch.distributed.ReduceOp.SUM)
                            parameter.grad /= self.gpu_world_size
                nn.utils.clip_grad_norm_(self.add.parameters(), self.max_grad_norm)
                self.add_optimizer.step()

                with torch.no_grad():
                    positive_acc = (positive_logits > 0.0).float().mean()
                    negative_acc = (negative_logits < 0.0).float().mean()

                mean_add_loss += add_loss.item()
                mean_add_pos_loss += positive_loss.item()
                mean_add_neg_loss += negative_loss.item()
                mean_add_grad_penalty += grad_penalty.item()
                mean_add_logit_reg += logit_reg.item()
                mean_add_pos_acc += positive_acc.item()
                mean_add_neg_acc += negative_acc.item()
                add_updates += 1

        with torch.no_grad():
            add_reward, add_logits = self.add.imitation_reward(policy_deltas, self.add_reward_scale)

        loss_dict = super().update()
        loss_dict.update(
            {
                "add_disc": mean_add_loss / max(add_updates, 1),
                "add_pos_loss": mean_add_pos_loss / max(add_updates, 1),
                "add_neg_loss": mean_add_neg_loss / max(add_updates, 1),
                "add_grad_penalty": mean_add_grad_penalty / max(add_updates, 1),
                "add_logit_reg": mean_add_logit_reg / max(add_updates, 1),
                "add_pos_acc": mean_add_pos_acc / max(add_updates, 1),
                "add_neg_acc": mean_add_neg_acc / max(add_updates, 1),
                "add_reward": add_reward.mean().item(),
                "add_logit": add_logits.mean().item(),
                "add_reward_weight": self.current_add_weight,
            }
        )
        return loss_dict

    def broadcast_parameters(self) -> None:
        super().broadcast_parameters()
        if self.is_multi_gpu:
            state = [
                self.add.state_dict(),
                self.add_replay.state_dict(),
                self.add_step_counter,
            ]
            torch.distributed.broadcast_object_list(state, src=0)
            self.add.load_state_dict(state[0])
            self.add_replay.load_state_dict(state[1])
            self.add_step_counter = state[2]
