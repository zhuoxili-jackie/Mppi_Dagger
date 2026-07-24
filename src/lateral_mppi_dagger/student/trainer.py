from __future__ import annotations

import json
import hashlib
import os
import random
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from lateral_mppi_dagger.config import canonical_hash
from lateral_mppi_dagger.data.dataset import (
    EpisodeWindowDataset,
    compute_observation_normalizer,
    load_manifest,
)

from .losses import imitation_loss
from .model import StudentPolicy


@dataclass(frozen=True)
class TrainerConfig:
    seed: int = 42
    hidden_dims: tuple[int, ...] = (256, 256, 128)
    activation: str = "elu"
    wheel_action_mode: str = "hard_zero"
    zero_command_previous_action_deadband: float = 0.03
    lateral_command_activation_start_m_s: float = 0.0
    lateral_command_activation_full_m_s: float = 0.0
    lateral_command_abs_limit_m_s: float = 0.0
    physical_target_rate_limit_rad_s: float = 0.0
    physical_target_abs_limit_rad: float = 0.0
    physical_target_abs_limit_rad_by_joint: tuple[float, ...] | None = None
    control_dt_s: float = 0.02
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-5
    batch_size: int = 1024
    epochs: int = 100
    window_length: int = 3
    window_stride: int = 1
    huber_delta: float = 0.5
    first_difference_weight: float = 0.01
    second_difference_weight: float = 0.002
    gradient_clip_norm: float = 5.0
    checkpoint_every_epochs: int = 5
    num_workers: int = 0
    normalization_std_floor: float = 1.0e-4
    max_batches_per_epoch: int | None = None
    sampling_policy: str = "uniform"
    latest_dagger_round: int | None = None
    sampling_mix_initial: float = 0.30
    sampling_mix_latest: float = 0.40
    sampling_mix_historical: float = 0.30
    sampling_mix_latest_student: float = 0.50
    sampling_mix_latest_teacher: float = 0.10

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "TrainerConfig":
        supported = {field.name for field in cls.__dataclass_fields__.values()}
        selected = {key: value for key, value in values.items() if key in supported}
        if "hidden_dims" in selected:
            selected["hidden_dims"] = tuple(selected["hidden_dims"])
        if selected.get("physical_target_abs_limit_rad_by_joint") is not None:
            selected["physical_target_abs_limit_rad_by_joint"] = tuple(
                selected["physical_target_abs_limit_rad_by_joint"]
            )
        return cls(**selected)


class BCTrainer:
    def __init__(
        self,
        dataset_root: str | Path,
        output_dir: str | Path,
        config: TrainerConfig,
        raw_min: np.ndarray,
        raw_max: np.ndarray,
        action_scale: np.ndarray,
        device: str = "cuda",
    ):
        self.dataset_root = Path(dataset_root).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
        self.raw_min = torch.as_tensor(raw_min, dtype=torch.float32)
        self.raw_max = torch.as_tensor(raw_max, dtype=torch.float32)
        self.action_scale = torch.as_tensor(action_scale, dtype=torch.float32)
        self.dataset_manifest_hash = canonical_hash(load_manifest(self.dataset_root))
        self.initialized_from: dict[str, str] | None = None
        self._seed_everything(config.seed)

        mean, std = compute_observation_normalizer(
            self.dataset_root, split="train", std_floor=config.normalization_std_floor
        )
        self.model = StudentPolicy(
            hidden_dims=config.hidden_dims,
            activation=config.activation,
            wheel_action_mode=config.wheel_action_mode,
            zero_command_previous_action_deadband=(
                config.zero_command_previous_action_deadband
            ),
            lateral_command_activation_start_m_s=(
                config.lateral_command_activation_start_m_s
            ),
            lateral_command_activation_full_m_s=(
                config.lateral_command_activation_full_m_s
            ),
            lateral_command_abs_limit_m_s=(
                config.lateral_command_abs_limit_m_s
            ),
            physical_target_rate_limit_rad_s=(
                config.physical_target_rate_limit_rad_s
            ),
            physical_target_abs_limit_rad=(
                config.physical_target_abs_limit_rad
            ),
            physical_target_abs_limit_rad_by_joint=(
                config.physical_target_abs_limit_rad_by_joint
            ),
            control_dt_s=config.control_dt_s,
            observation_mean=torch.from_numpy(mean),
            observation_std=torch.from_numpy(std),
            raw_min=self.raw_min,
            raw_max=self.raw_max,
            action_scale=self.action_scale,
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.train_dataset = EpisodeWindowDataset(
            self.dataset_root, "train", config.window_length, config.window_stride
        )
        self.validation_dataset = EpisodeWindowDataset(
            self.dataset_root, "validation", config.window_length, config.window_stride
        )
        generator = torch.Generator()
        generator.manual_seed(config.seed)
        self.train_generator = generator
        if config.sampling_policy == "uniform":
            sampler = None
            shuffle = True
        elif config.sampling_policy == "dagger_mix":
            if config.latest_dagger_round is None:
                raise ValueError(
                    "sampling_policy='dagger_mix' requires latest_dagger_round."
                )
            weights = self.train_dataset.dagger_sampling_weights(
                config.latest_dagger_round,
                initial_mass=config.sampling_mix_initial,
                latest_mass=config.sampling_mix_latest,
                historical_mass=config.sampling_mix_historical,
            )
            sampler = WeightedRandomSampler(
                torch.from_numpy(weights),
                num_samples=len(weights),
                replacement=True,
                generator=generator,
            )
            shuffle = False
        elif config.sampling_policy == "dagger_recovery":
            if config.latest_dagger_round is None:
                raise ValueError(
                    "sampling_policy='dagger_recovery' requires latest_dagger_round."
                )
            weights = self.train_dataset.dagger_recovery_sampling_weights(
                config.latest_dagger_round,
                initial_mass=config.sampling_mix_initial,
                latest_student_mass=config.sampling_mix_latest_student,
                latest_teacher_mass=config.sampling_mix_latest_teacher,
                historical_mass=config.sampling_mix_historical,
            )
            sampler = WeightedRandomSampler(
                torch.from_numpy(weights),
                num_samples=len(weights),
                replacement=True,
                generator=generator,
            )
            shuffle = False
        else:
            raise ValueError(f"Unsupported sampling_policy {config.sampling_policy!r}.")
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=config.num_workers,
            generator=generator,
        )
        self.validation_loader = DataLoader(
            self.validation_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        )
        self.start_epoch = 0
        self.best_validation = float("inf")
        self.metrics_path = self.output_dir / "metrics.jsonl"

    @staticmethod
    def _seed_everything(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def resume(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        expected_hash = canonical_hash(asdict(self.config))
        if checkpoint.get("trainer_config_hash") != expected_hash:
            raise ValueError(
                "Resume trainer config does not match checkpoint. Use the exact saved config or start a new run."
            )
        if checkpoint.get("dataset_manifest_hash") != self.dataset_manifest_hash:
            raise ValueError(
                "Resume dataset manifest differs from the checkpoint. Start a new DAgger "
                "round with --initialize-from instead of resuming."
            )
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.start_epoch = int(checkpoint["epoch"]) + 1
        self.best_validation = float(checkpoint["best_validation"])
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if "data_loader_rng_state" not in checkpoint:
            raise ValueError(
                "Resume checkpoint predates deterministic DataLoader RNG capture."
            )
        self.train_generator.set_state(checkpoint["data_loader_rng_state"].cpu())
        np.random.set_state(checkpoint["numpy_rng_state"])
        random.setstate(checkpoint["python_rng_state"])

    def initialize_from(self, path: str | Path) -> None:
        """Initialize network weights while retaining this round's normalization."""
        source = Path(path).resolve()
        checkpoint = torch.load(source, map_location=self.device, weights_only=False)
        if checkpoint.get("model_spec") != self.model.specification():
            raise ValueError(
                "Initialization checkpoint model architecture/ABI differs from the current student."
            )
        self.model.load_network_preserving_function(
            checkpoint["model_state"],
            checkpoint["observation_mean"],
            checkpoint["observation_std"],
        )
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        self.initialized_from = {"path": str(source), "sha256": digest}

    def _run_loader(self, loader: DataLoader, training: bool) -> dict[str, float]:
        self.model.train(training)
        totals = {"total": 0.0, "action": 0.0, "first": 0.0, "second": 0.0, "rmse_sum": 0.0}
        batches = 0
        valid_values = 0
        for batch_index, batch in enumerate(loader):
            if self.config.max_batches_per_epoch is not None and batch_index >= self.config.max_batches_per_epoch:
                break
            obs = batch["obs"].to(self.device)
            target = batch["target"].to(self.device)
            valid = batch["valid"].to(self.device)
            prediction = self.model(obs)
            loss = imitation_loss(
                prediction,
                target,
                valid,
                self.config.huber_delta,
                self.config.first_difference_weight,
                self.config.second_difference_weight,
            )
            if training:
                self.optimizer.zero_grad(set_to_none=True)
                loss.total.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
                self.optimizer.step()
            finite_mask = valid & torch.isfinite(target).all(dim=-1)
            squared = torch.square(prediction - torch.nan_to_num(target)).mean(dim=-1)
            totals["rmse_sum"] += float(squared.masked_select(finite_mask).sum().item())
            valid_values += int(finite_mask.sum().item())
            totals["total"] += float(loss.total.detach().item())
            totals["action"] += float(loss.action.detach().item())
            totals["first"] += float(loss.first_difference.detach().item())
            totals["second"] += float(loss.second_difference.detach().item())
            batches += 1
        if batches == 0 or valid_values == 0:
            raise RuntimeError("A training/evaluation loader produced no valid labels.")
        return {
            "loss": totals["total"] / batches,
            "action_loss": totals["action"] / batches,
            "first_difference_loss": totals["first"] / batches,
            "second_difference_loss": totals["second"] / batches,
            "action_rmse": float(np.sqrt(totals["rmse_sum"] / valid_values)),
            "batches": batches,
            "valid_windows": valid_values,
        }

    def _checkpoint_payload(self, epoch: int) -> dict[str, Any]:
        return {
            "format": "lateral_mppi_dagger_student_checkpoint_v1",
            "epoch": epoch,
            "best_validation": self.best_validation,
            "model_spec": self.model.specification(),
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "observation_mean": self.model.observation_mean.detach().cpu(),
            "observation_std": self.model.observation_std.detach().cpu(),
            "raw_min": self.model.raw_min.detach().cpu(),
            "raw_max": self.model.raw_max.detach().cpu(),
            "action_scale": self.model.action_scale.detach().cpu(),
            "trainer_config": asdict(self.config),
            "trainer_config_hash": canonical_hash(asdict(self.config)),
            "dataset_root": str(self.dataset_root),
            "dataset_manifest_hash": self.dataset_manifest_hash,
            "initialized_from": self.initialized_from,
            "torch_rng_state": torch.get_rng_state(),
            "data_loader_rng_state": self.train_generator.get_state(),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
        }

    def _save_checkpoint(self, filename: str, epoch: int) -> Path:
        path = self.output_dir / filename
        with tempfile.NamedTemporaryFile(prefix=f".{filename}.", suffix=".tmp", dir=self.output_dir, delete=False) as stream:
            temporary = Path(stream.name)
        try:
            torch.save(self._checkpoint_payload(epoch), temporary)
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
        return path

    def train(self) -> Path:
        best_path = self.output_dir / "student_best_checkpoint.pt"
        for epoch in range(self.start_epoch, self.config.epochs):
            started = time.perf_counter()
            train_metrics = self._run_loader(self.train_loader, training=True)
            with torch.inference_mode():
                validation_metrics = self._run_loader(self.validation_loader, training=False)
            improved = validation_metrics["loss"] < self.best_validation
            if improved:
                self.best_validation = validation_metrics["loss"]
                self._save_checkpoint(best_path.name, epoch)
            if (epoch + 1) % self.config.checkpoint_every_epochs == 0 or epoch + 1 == self.config.epochs:
                self._save_checkpoint(f"student_epoch_{epoch:04d}.pt", epoch)
            record = {
                "epoch": epoch,
                "elapsed_seconds": time.perf_counter() - started,
                "train": train_metrics,
                "validation": validation_metrics,
                "best_validation": self.best_validation,
                "improved": improved,
                "device": str(self.device),
            }
            with self.metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
            print(json.dumps(record, sort_keys=True))
        if not best_path.is_file():
            raise RuntimeError("Training ended without a best checkpoint.")
        return best_path
