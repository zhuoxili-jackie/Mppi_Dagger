# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from rsl_rl.runners import OnPolicyRunner


class REDOnPolicyRunner(OnPolicyRunner):
    """On-policy runner that checkpoints the RED training module."""

    def _construct_algorithm(self, obs):
        red_cfg = self.alg_cfg.get("red_cfg")
        if red_cfg is not None:
            red_cfg["reward_weight"] *= self.env.unwrapped.step_dt
        return super()._construct_algorithm(obs)

    def save(self, path: str, infos: dict | None = None) -> None:
        saved_dict = {
            "model_state_dict": self.alg.policy.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "red_state_dict": self.alg.red.state_dict(),
            "red_optimizer_state_dict": self.alg.red_optimizer.state_dict(),
            "red_step_counter": self.alg.red_step_counter,
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        torch.save(saved_dict, path)
        self.logger.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None) -> dict | None:
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        resumed_training = self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
        self.alg.red.load_state_dict(loaded_dict["red_state_dict"])
        self.alg.red_step_counter = loaded_dict.get("red_step_counter", 0)
        if load_optimizer and resumed_training:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            self.alg.red_optimizer.load_state_dict(loaded_dict["red_optimizer_state_dict"])
        if resumed_training:
            self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict.get("infos")

    def train_mode(self) -> None:
        super().train_mode()
        self.alg.red.train()

    def eval_mode(self) -> None:
        super().eval_mode()
        self.alg.red.eval()
