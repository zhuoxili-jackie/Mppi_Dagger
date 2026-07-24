# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from rsl_rl.runners import OnPolicyRunner


class ADDOnPolicyRunner(OnPolicyRunner):
    """On-policy runner that checkpoints the ADD discriminator state."""

    def _construct_algorithm(self, obs):
        add_cfg = self.alg_cfg.get("add_cfg")
        if add_cfg is not None:
            add_cfg["reward_weight"] *= self.env.unwrapped.step_dt
        return super()._construct_algorithm(obs)

    def save(self, path: str, infos: dict | None = None) -> None:
        saved_dict = {
            "model_state_dict": self.alg.policy.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "add_state_dict": self.alg.add.state_dict(),
            "add_optimizer_state_dict": self.alg.add_optimizer.state_dict(),
            "add_replay_state_dict": self.alg.add_replay.state_dict(),
            "add_step_counter": self.alg.add_step_counter,
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        torch.save(saved_dict, path)
        self.logger.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None) -> dict | None:
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        resumed_training = self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
        self.alg.add.load_state_dict(loaded_dict["add_state_dict"])
        if "add_replay_state_dict" in loaded_dict:
            self.alg.add_replay.load_state_dict(loaded_dict["add_replay_state_dict"])
        self.alg.add_step_counter = loaded_dict.get("add_step_counter", 0)
        if load_optimizer and resumed_training:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            self.alg.add_optimizer.load_state_dict(loaded_dict["add_optimizer_state_dict"])
        if resumed_training:
            self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict.get("infos")

    def train_mode(self) -> None:
        super().train_mode()
        self.alg.add.train()

    def eval_mode(self) -> None:
        super().eval_mode()
        self.alg.add.eval()
