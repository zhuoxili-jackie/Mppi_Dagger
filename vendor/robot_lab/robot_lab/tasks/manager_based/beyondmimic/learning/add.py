# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import torch.nn as nn

from .red import DifferentialNormalization


def _activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "elu":
        return nn.ELU()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "leaky_relu":
        return nn.LeakyReLU()
    raise ValueError(f"Unsupported activation: {name}")


class DifferentialDiscriminator(nn.Module):
    """ADD-style discriminator over reference-policy differential features."""

    def __init__(
        self,
        num_states: int,
        hidden_dims: list[int] | tuple[int, ...] = (1024, 512),
        activation: str = "elu",
        state_clip: float | None = 10.0,
        diff_min_scale: float = 1.0e-4,
        diff_momentum: float = 0.01,
        device: str = "cpu",
    ):
        super().__init__()
        if state_clip is not None and state_clip <= 0.0:
            raise ValueError("state_clip must be positive when set.")

        self.num_states = num_states
        self.state_normalizer = DifferentialNormalization(
            shape=[num_states],
            min_scale=diff_min_scale,
            momentum=diff_momentum,
            clip=state_clip,
            device=device,
        )

        layers: list[nn.Module] = []
        last_dim = num_states
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(_activation(activation))
            last_dim = hidden_dim
        self.trunk = nn.Sequential(*layers).to(device)
        self.logits = nn.Linear(last_dim, 1).to(device)
        nn.init.uniform_(self.logits.weight, -1.0, 1.0)
        nn.init.zeros_(self.logits.bias)

    def normalize(self, states: torch.Tensor) -> torch.Tensor:
        return self.state_normalizer(states)

    def forward_normalized(self, normalized_states: torch.Tensor) -> torch.Tensor:
        return self.logits(self.trunk(normalized_states)).squeeze(-1)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.forward_normalized(self.normalize(states))

    def logit_weights(self) -> torch.Tensor:
        return self.logits.weight.flatten()

    @torch.no_grad()
    def imitation_reward(self, states: torch.Tensor, reward_scale: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ADD reward -log(1 - D(delta)) and discriminator logits."""
        logits = self.forward(states)
        probability = torch.sigmoid(logits)
        reward = -torch.log(torch.clamp(1.0 - probability, min=1.0e-4))
        return reward_scale * reward, logits

    @torch.no_grad()
    def update_normalization(self, states: torch.Tensor) -> None:
        self.state_normalizer.update(states)


class DifferentialReplayBuffer:
    """Small ring buffer for discriminator negatives, matching ADD/AMP practice."""

    def __init__(self, capacity: int, num_states: int, device: str):
        self.capacity = int(capacity)
        self.num_states = int(num_states)
        self.device = device
        self.storage = torch.zeros(self.capacity, self.num_states, dtype=torch.float32, device=device)
        self.index = 0
        self.count = 0

    def insert(self, states: torch.Tensor) -> None:
        if self.capacity <= 0:
            return
        states = states.detach()
        if states.shape[0] >= self.capacity:
            self.storage[:] = states[-self.capacity :]
            self.index = 0
            self.count = self.capacity
            return

        n = states.shape[0]
        end = self.index + n
        if end <= self.capacity:
            self.storage[self.index : end] = states
        else:
            first = self.capacity - self.index
            self.storage[self.index :] = states[:first]
            self.storage[: end - self.capacity] = states[first:]
        self.index = end % self.capacity
        self.count = min(self.count + n, self.capacity)

    def sample(self, num_samples: int) -> torch.Tensor | None:
        if self.count == 0 or num_samples <= 0:
            return None
        indices = torch.randint(self.count, (num_samples,), device=self.storage.device)
        return self.storage[indices]

    def state_dict(self) -> dict:
        return {
            "storage": self.storage,
            "index": self.index,
            "count": self.count,
            "capacity": self.capacity,
            "num_states": self.num_states,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        if state_dict["storage"].shape != self.storage.shape:
            raise ValueError("Cannot load ADD replay buffer with a different shape.")
        self.storage.copy_(state_dict["storage"])
        self.index = int(state_dict["index"])
        self.count = int(state_dict["count"])
