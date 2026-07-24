# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import torch.nn as nn
from rsl_rl.networks import MLP, EmpiricalNormalization


class DifferentialNormalization(nn.Module):
    """Normalize differential observations by running mean absolute magnitude.

    Differential imitation features should keep zero as the expert center.
    This normalizer scales by absolute magnitude without subtracting a mean.
    """

    def __init__(
        self,
        shape: list[int] | tuple[int, ...],
        min_scale: float = 1.0e-4,
        momentum: float = 0.01,
        clip: float | None = None,
        device: str = "cpu",
    ):
        super().__init__()
        if min_scale <= 0.0:
            raise ValueError("min_scale must be positive.")
        if not 0.0 < momentum <= 1.0:
            raise ValueError("momentum must be in (0, 1].")
        if clip is not None and clip <= 0.0:
            raise ValueError("clip must be positive when set.")

        self.min_scale = min_scale
        self.momentum = momentum
        self.clip = clip
        self.register_buffer("mean_abs", torch.ones(shape, dtype=torch.float32, device=device))
        self.register_buffer("initialized", torch.zeros((), dtype=torch.bool, device=device))

    @torch.no_grad()
    def update(self, states: torch.Tensor) -> None:
        batch_mean_abs = torch.mean(torch.abs(states.detach()), dim=0)
        if not bool(self.initialized.item()):
            self.mean_abs.copy_(torch.clamp_min(batch_mean_abs, self.min_scale))
            self.initialized.fill_(True)
        else:
            self.mean_abs.lerp_(torch.clamp_min(batch_mean_abs, self.min_scale), self.momentum)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        normalized = states / torch.clamp_min(self.mean_abs, self.min_scale)
        if self.clip is not None:
            normalized = torch.clamp(normalized, -self.clip, self.clip)
        return normalized


class RandomExpertDistillation(nn.Module):
    """An expert support estimator based on random distillation."""

    def __init__(
        self,
        num_states: int,
        num_outputs: int = 64,
        predictor_hidden_dims: list[int] | tuple[int, ...] = (256, 128),
        target_hidden_dims: list[int] | tuple[int, ...] = (256, 128),
        activation: str = "elu",
        state_normalization: bool = True,
        normalization_type: str = "empirical",
        state_clip: float | None = None,
        diff_min_scale: float = 1.0e-4,
        diff_momentum: float = 0.01,
        reward_temperature: float = 0.5,
        use_delta_kernel: bool = False,
        delta_kernel_temperature: float = 1.0,
        delta_kernel_use_raw_states: bool = False,
        device: str = "cpu",
    ):
        super().__init__()
        if reward_temperature <= 0.0:
            raise ValueError("reward_temperature must be positive.")
        if state_clip is not None and state_clip <= 0.0:
            raise ValueError("state_clip must be positive when set.")
        if normalization_type not in {"empirical", "differential", "none"}:
            raise ValueError("normalization_type must be 'empirical', 'differential', or 'none'.")
        if delta_kernel_temperature <= 0.0:
            raise ValueError("delta_kernel_temperature must be positive.")

        self.num_states = num_states
        self.reward_temperature = reward_temperature
        self.state_normalization = state_normalization
        self.state_clip = state_clip
        self.normalization_type = normalization_type
        self.use_delta_kernel = use_delta_kernel
        self.delta_kernel_temperature = delta_kernel_temperature
        self.delta_kernel_use_raw_states = delta_kernel_use_raw_states

        if state_normalization and normalization_type == "empirical":
            self.state_normalizer = EmpiricalNormalization(shape=[num_states], until=1.0e8).to(device)
        elif state_normalization and normalization_type == "differential":
            self.state_normalizer = DifferentialNormalization(
                shape=[num_states],
                min_scale=diff_min_scale,
                momentum=diff_momentum,
                clip=state_clip,
                device=device,
            )
        else:
            self.state_normalizer = nn.Identity()

        self.predictor = MLP(num_states, num_outputs, predictor_hidden_dims, activation).to(device)
        self.target = MLP(num_states, num_outputs, target_hidden_dims, activation).to(device)
        self.target.eval()
        for parameter in self.target.parameters():
            parameter.requires_grad_(False)

    def _normalize_states(self, states: torch.Tensor) -> torch.Tensor:
        normalized_states = self.state_normalizer(states)
        if self.state_clip is not None and not isinstance(self.state_normalizer, DifferentialNormalization):
            normalized_states = torch.clamp(normalized_states, -self.state_clip, self.state_clip)
        return normalized_states

    def prediction_error(self, states: torch.Tensor) -> torch.Tensor:
        normalized_states = self._normalize_states(states)
        prediction = self.predictor(normalized_states)
        with torch.no_grad():
            target = self.target(normalized_states)
        return torch.square(prediction - target).mean(dim=-1)

    def delta_kernel_reward(self, states: torch.Tensor) -> torch.Tensor:
        kernel_states = states if self.delta_kernel_use_raw_states else self._normalize_states(states)
        if self.state_clip is not None:
            kernel_states = torch.clamp(kernel_states, -self.state_clip, self.state_clip)
        delta_rms = torch.sqrt(torch.mean(torch.square(kernel_states), dim=-1))
        return torch.exp(-torch.square(delta_rms) / self.delta_kernel_temperature)

    @torch.no_grad()
    def style_reward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        error = self.prediction_error(states)
        reward = torch.exp(-error / self.reward_temperature)
        if self.use_delta_kernel:
            # Dense in-module shaping for differential RED: the predictor still
            # scores expert support, while this term prevents all bad deltas from
            # collapsing to the same near-zero reward early in training.
            reward = 0.5 * (reward + self.delta_kernel_reward(states))
        return reward, error

    def update_normalization(self, expert_states: torch.Tensor) -> None:
        if self.state_normalization and hasattr(self.state_normalizer, "update"):
            self.state_normalizer.update(expert_states)

    def train(self, mode: bool = True):
        self.predictor.train(mode)
        if self.state_normalization:
            self.state_normalizer.train(mode)
        self.target.eval()
        return self

    def eval(self):
        return self.train(False)
