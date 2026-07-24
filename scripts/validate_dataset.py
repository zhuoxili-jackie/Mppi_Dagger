#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from _bootstrap import ROOT, write_json

from lateral_mppi_dagger.data.dataset import load_manifest
from lateral_mppi_dagger.data.schema import read_episode_shard


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate every dataset shard and episode-level split.")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/dataset_validation.json")
    args = parser.parse_args()
    root = args.dataset.resolve()
    records = load_manifest(root)
    episode_ids_by_split: dict[str, set[str]] = {}
    reports = []
    for record in records:
        shard = read_episode_shard(root / record["path"])
        episode_ids_by_split.setdefault(record["split"], set()).add(record["episode_id"])
        reports.append(
            {
                "episode_id": record["episode_id"],
                "split": record["split"],
                "steps": shard.arrays["step_id"].shape[0],
                "teacher_valid_rate": float(shard.arrays["teacher_valid"].mean()),
                "student_valid_rate": float(shard.arrays["student_valid"].mean()),
                "wheel_action_max_abs": float(np.max(np.abs(shard.arrays["executed_action16"][:, 12:]))),
            }
        )
    split_names = sorted(episode_ids_by_split)
    for index, first in enumerate(split_names):
        for second in split_names[index + 1 :]:
            overlap = episode_ids_by_split[first] & episode_ids_by_split[second]
            if overlap:
                raise ValueError(f"Episode leakage between {first} and {second}: {sorted(overlap)}")
    result = {
        "schema_version": "pcbc-dataset-validation-v1",
        "dataset": str(root),
        "episodes": reports,
        "split_counts": {name: len(values) for name, values in episode_ids_by_split.items()},
        "episode_level_split_leakage": False,
    }
    write_json(args.output, result)
    print(json.dumps({"episodes": len(reports), "split_counts": result["split_counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()

