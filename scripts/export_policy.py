#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from _bootstrap import ROOT, load_contract

from lateral_mppi_dagger.data.dataset import load_manifest
from lateral_mppi_dagger.data.schema import read_episode_shard
from lateral_mppi_dagger.export.exporter import export_student_policy


def golden_observations(dataset: Path, maximum: int) -> np.ndarray:
    values = []
    for record in load_manifest(dataset):
        shard = read_episode_shard(dataset / record["path"])
        valid = shard.arrays["teacher_valid"].astype(bool)
        values.append(shard.arrays["obs93_clean"][valid])
    observations = np.concatenate(values, axis=0)
    if observations.shape[0] <= maximum:
        return observations
    indices = np.linspace(0, observations.shape[0] - 1, maximum, dtype=np.int64)
    return observations[indices]


def main() -> None:
    parser = argparse.ArgumentParser(description="Pure-PyTorch export; Isaac Sim is not launched.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "exported")
    parser.add_argument("--golden-samples", type=int, default=256)
    args = parser.parse_args()
    manifest = export_student_policy(
        args.checkpoint,
        args.output,
        load_contract(),
        golden_observations(args.dataset.resolve(), args.golden_samples),
    )
    print(json.dumps({"output": str(args.output.resolve()), "model_hash": manifest["model_hash"]}, sort_keys=True))


if __name__ == "__main__":
    main()

