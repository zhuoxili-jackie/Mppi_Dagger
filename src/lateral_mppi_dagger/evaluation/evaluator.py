from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from lateral_mppi_dagger.data.dataset import EpisodeFrameDataset


def evaluate_open_loop(
    model: torch.nn.Module,
    dataset_root: str | Path,
    split: str = "test",
    device: str = "cpu",
) -> dict[str, float]:
    dataset = EpisodeFrameDataset(dataset_root, split)
    model = model.to(device).eval()
    squared_error = []
    absolute_error = []
    valid_count = 0
    with torch.inference_mode():
        for item in dataset:
            if not bool(item["valid"]):
                continue
            prediction = model(item["obs"].to(device).unsqueeze(0))[0].cpu().numpy()
            target = item["target"].numpy()
            squared_error.append(np.square(prediction - target))
            absolute_error.append(np.abs(prediction - target))
            valid_count += 1
    if valid_count == 0:
        raise ValueError(f"No valid teacher labels in split={split!r}")
    squared = np.concatenate(squared_error)
    absolute = np.concatenate(absolute_error)
    return {
        "valid_frames": valid_count,
        "action_rmse": float(np.sqrt(np.mean(squared))),
        "action_mae": float(np.mean(absolute)),
        "action_max_abs": float(np.max(absolute)),
        "wheel_max_abs": 0.0,
    }

