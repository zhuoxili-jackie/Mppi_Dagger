from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .schema import ENUMS, EpisodeShard, read_episode_shard


def load_manifest(dataset_root: str | Path) -> list[dict[str, Any]]:
    root = Path(dataset_root).resolve()
    path = root / "manifest.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Dataset manifest does not exist: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            records.append(record)
    episode_ids = [record["episode_id"] for record in records]
    if len(episode_ids) != len(set(episode_ids)):
        raise ValueError("Dataset manifest contains duplicate episode_id entries.")
    return records


def _load_selected_shards(dataset_root: Path, split: str) -> list[tuple[dict[str, Any], EpisodeShard]]:
    selected = []
    for record in load_manifest(dataset_root):
        if record["split"] != split:
            continue
        shard = read_episode_shard(dataset_root / record["path"])
        selected.append((record, shard))
    if not selected:
        raise ValueError(f"No episodes found for split={split!r} in {dataset_root}")
    return selected


def compute_observation_normalizer(
    dataset_root: str | Path,
    split: str = "train",
    std_floor: float = 1.0e-4,
) -> tuple[np.ndarray, np.ndarray]:
    root = Path(dataset_root).resolve()
    observations = []
    for _, shard in _load_selected_shards(root, split):
        valid = shard.arrays["teacher_valid"].astype(bool)
        observations.append(shard.arrays["obs93_clean"][valid])
    concatenated = np.concatenate(observations, axis=0).astype(np.float64)
    if concatenated.shape[0] < 2:
        raise ValueError("At least two valid training observations are required for normalization.")
    mean = concatenated.mean(axis=0)
    std = concatenated.std(axis=0)
    std = np.maximum(std, std_floor)
    return mean.astype(np.float32), std.astype(np.float32)


class EpisodeWindowDataset(Dataset[dict[str, torch.Tensor]]):
    """Windows never cross episode/reset boundaries, so temporal losses are well defined."""

    def __init__(
        self,
        dataset_root: str | Path,
        split: str,
        window_length: int = 3,
        stride: int = 1,
    ):
        self.root = Path(dataset_root).resolve()
        self.window_length = int(window_length)
        self.stride = int(stride)
        if self.window_length < 1 or self.stride < 1:
            raise ValueError("window_length and stride must be positive.")
        self.shards = _load_selected_shards(self.root, split)
        self.index: list[tuple[int, int]] = []
        for shard_index, (_, shard) in enumerate(self.shards):
            steps = shard.arrays["step_id"].shape[0]
            for start in range(0, steps - self.window_length + 1, self.stride):
                stop = start + self.window_length
                step_ids = shard.arrays["step_id"][start:stop]
                if not np.array_equal(np.diff(step_ids), np.ones(self.window_length - 1, dtype=np.int32)):
                    continue
                if np.any(shard.arrays["terminal"][start : stop - 1]):
                    continue
                self.index.append((shard_index, start))
        if not self.index:
            raise ValueError(f"No valid windows of length {self.window_length} for split={split!r}")

    def dagger_sampling_weights(
        self,
        latest_round: int,
        initial_mass: float = 0.30,
        latest_mass: float = 0.40,
        historical_mass: float = 0.30,
    ) -> np.ndarray:
        """Allocate probability mass by DAgger source without crossing episodes."""
        if latest_round < 1:
            raise ValueError("latest_round must be at least 1 for DAgger mixing.")
        requested_mass = {
            "initial": float(initial_mass),
            "latest": float(latest_mass),
            "historical": float(historical_mass),
        }
        if any(value < 0.0 for value in requested_mass.values()):
            raise ValueError("DAgger sampling masses must be non-negative.")
        if sum(requested_mass.values()) <= 0.0:
            raise ValueError("At least one DAgger sampling mass must be positive.")

        groups: dict[str, list[int]] = {
            "initial": [],
            "latest": [],
            "historical": [],
        }
        for window_index, (shard_index, _) in enumerate(self.index):
            record, shard = self.shards[shard_index]
            round_value = int(
                record["dagger_round"]
                if "dagger_round" in record
                else shard.metadata.get("dagger_round", 0)
            )
            if round_value == 0:
                groups["initial"].append(window_index)
            elif round_value == latest_round:
                groups["latest"].append(window_index)
            elif 0 < round_value < latest_round:
                groups["historical"].append(window_index)
            else:
                raise ValueError(
                    f"Dataset contains DAgger round {round_value}, beyond requested latest round "
                    f"{latest_round}."
                )

        active_mass = sum(
            requested_mass[name] for name, indices in groups.items() if indices
        )
        if active_mass <= 0.0:
            raise ValueError("No dataset windows belong to a non-zero DAgger sampling group.")
        weights = np.zeros(len(self.index), dtype=np.float64)
        for name, indices in groups.items():
            if not indices:
                continue
            group_mass = requested_mass[name] / active_mass
            weights[np.asarray(indices, dtype=np.int64)] = group_mass / len(indices)
        if not np.isclose(weights.sum(), 1.0, atol=1.0e-12):
            raise RuntimeError("DAgger sampling weights do not sum to one.")
        return weights

    def dagger_recovery_sampling_weights(
        self,
        latest_round: int,
        initial_mass: float = 0.20,
        latest_student_mass: float = 0.50,
        latest_teacher_mass: float = 0.10,
        historical_mass: float = 0.20,
    ) -> np.ndarray:
        """Prioritize MPPI labels from states actually visited by the latest student.

        Standard round-level mixing can hide the small but important set of
        student-executed failure trajectories behind many full-horizon teacher
        trajectories.  This recovery policy preserves R0 and historical data
        while assigning an explicit probability mass to latest-round windows
        whose behavior policy was the student.
        """
        if latest_round < 1:
            raise ValueError("latest_round must be at least 1 for DAgger recovery.")
        requested_mass = {
            "initial": float(initial_mass),
            "latest_student": float(latest_student_mass),
            "latest_teacher": float(latest_teacher_mass),
            "historical": float(historical_mass),
        }
        if any(value < 0.0 for value in requested_mass.values()):
            raise ValueError("DAgger recovery sampling masses must be non-negative.")
        if sum(requested_mass.values()) <= 0.0:
            raise ValueError("At least one DAgger recovery sampling mass must be positive.")

        groups: dict[str, list[int]] = {
            "initial": [],
            "latest_student": [],
            "latest_teacher": [],
            "historical": [],
        }
        student_code = ENUMS["behavior_policy"]["STUDENT"]
        teacher_code = ENUMS["behavior_policy"]["TEACHER"]
        for window_index, (shard_index, start) in enumerate(self.index):
            record, shard = self.shards[shard_index]
            round_value = int(
                record["dagger_round"]
                if "dagger_round" in record
                else shard.metadata.get("dagger_round", 0)
            )
            if round_value == 0:
                groups["initial"].append(window_index)
            elif 0 < round_value < latest_round:
                groups["historical"].append(window_index)
            elif round_value == latest_round:
                behavior = np.asarray(
                    shard.arrays["behavior_policy"][
                        start : start + self.window_length
                    ]
                )
                if behavior.shape != (self.window_length,) or not np.all(
                    behavior == behavior[0]
                ):
                    raise ValueError(
                        "DAgger recovery requires episode-level behavior selection; "
                        "a training window contains mixed behavior policies."
                    )
                if int(behavior[0]) == student_code:
                    groups["latest_student"].append(window_index)
                elif int(behavior[0]) == teacher_code:
                    groups["latest_teacher"].append(window_index)
                else:
                    raise ValueError(
                        "DAgger recovery encountered a latest-round fallback-only "
                        "training window."
                    )
            else:
                raise ValueError(
                    f"Dataset contains DAgger round {round_value}, beyond requested "
                    f"latest round {latest_round}."
                )

        active_mass = sum(
            requested_mass[name] for name, indices in groups.items() if indices
        )
        if active_mass <= 0.0:
            raise ValueError(
                "No dataset windows belong to a non-zero DAgger recovery group."
            )
        weights = np.zeros(len(self.index), dtype=np.float64)
        for name, indices in groups.items():
            if not indices:
                continue
            group_mass = requested_mass[name] / active_mass
            weights[np.asarray(indices, dtype=np.int64)] = group_mass / len(indices)
        if not np.isclose(weights.sum(), 1.0, atol=1.0e-12):
            raise RuntimeError("DAgger recovery sampling weights do not sum to one.")
        return weights

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        shard_index, start = self.index[index]
        shard = self.shards[shard_index][1]
        stop = start + self.window_length
        arrays = shard.arrays
        return {
            "obs": torch.from_numpy(arrays["obs93_train"][start:stop]),
            "target": torch.from_numpy(arrays["teacher_action16"][start:stop]),
            "valid": torch.from_numpy(arrays["teacher_valid"][start:stop].astype(np.bool_)),
            "terminal": torch.from_numpy(arrays["terminal"][start:stop].astype(np.bool_)),
            "step_id": torch.from_numpy(arrays["step_id"][start:stop].astype(np.int64)),
        }


class EpisodeFrameDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, dataset_root: str | Path, split: str):
        root = Path(dataset_root).resolve()
        self.frames: list[tuple[np.ndarray, np.ndarray, bool]] = []
        for _, shard in _load_selected_shards(root, split):
            for observation, target, valid in zip(
                shard.arrays["obs93_clean"],
                shard.arrays["teacher_action16"],
                shard.arrays["teacher_valid"],
                strict=True,
            ):
                self.frames.append((observation, target, bool(valid)))

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        observation, target, valid = self.frames[index]
        return {
            "obs": torch.from_numpy(observation),
            "target": torch.from_numpy(target),
            "valid": torch.tensor(valid, dtype=torch.bool),
        }
