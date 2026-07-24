#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from _bootstrap import ROOT, write_json

from lateral_mppi_dagger.config import sha256_file
from lateral_mppi_dagger.data.dataset import load_manifest
from lateral_mppi_dagger.data.schema import ENUMS, read_episode_shard
from lateral_mppi_dagger.student.model import build_student_from_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select a recovery checkpoint on latest-round validation frames "
            "actually visited by the student and labeled by MPPI."
        )
    )
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        type=Path,
        action="append",
        default=[],
        help="Also compare an external baseline checkpoint; may be repeated.",
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--latest-dagger-round", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/recovery_checkpoint_selection.json",
    )
    args = parser.parse_args()

    dataset_root = args.dataset.resolve()
    observations: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    episode_ids: list[str] = []
    student_code = ENUMS["behavior_policy"]["STUDENT"]
    mppi_code = ENUMS["label_source"]["MPPI"]
    for record in load_manifest(dataset_root):
        if (
            record["split"] != "validation"
            or int(record.get("dagger_round", 0)) != args.latest_dagger_round
        ):
            continue
        shard = read_episode_shard(dataset_root / record["path"])
        behavior = shard.arrays["behavior_policy"]
        if not np.all(behavior == student_code):
            continue
        valid = shard.arrays["teacher_valid"].astype(bool)
        if not np.all(shard.arrays["label_source"][valid] == mppi_code):
            raise ValueError(
                f"Student validation episode {record['episode_id']} has non-MPPI labels."
            )
        observations.append(shard.arrays["obs93_train"][valid])
        targets.append(shard.arrays["teacher_action16"][valid])
        episode_ids.append(str(record["episode_id"]))
    if not observations:
        raise ValueError(
            "No latest-round student-executed MPPI validation frames were found."
        )
    obs = np.concatenate(observations, axis=0).astype(np.float32)
    target = np.concatenate(targets, axis=0).astype(np.float32)

    checkpoint_dir = args.checkpoint_dir.resolve()
    candidates = sorted(checkpoint_dir.glob("student_epoch_*.pt"))
    best_checkpoint = checkpoint_dir / "student_best_checkpoint.pt"
    if best_checkpoint.is_file():
        candidates.insert(0, best_checkpoint)
    candidates.extend(path.resolve() for path in args.candidate)
    candidates = list(dict.fromkeys(candidates))
    if not candidates:
        raise FileNotFoundError(f"No student checkpoints found below {checkpoint_dir}.")

    rows = []
    input_tensor = torch.from_numpy(obs).to(args.device)
    target_tensor = torch.from_numpy(target).to(args.device)
    for checkpoint_path in candidates:
        model, payload = build_student_from_checkpoint(
            str(checkpoint_path), map_location=args.device
        )
        model.to(args.device).eval()
        with torch.inference_mode():
            prediction = model(input_tensor)
        difference = prediction - target_tensor
        row = {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "epoch": int(payload["epoch"]),
            "action_rmse": float(torch.sqrt(torch.mean(torch.square(difference))).item()),
            "action_mae": float(torch.mean(torch.abs(difference)).item()),
            "action_max_abs": float(torch.max(torch.abs(difference)).item()),
            "wheel_max_abs": float(torch.max(torch.abs(prediction[:, 12:])).item()),
        }
        rows.append(row)
    rows.sort(key=lambda row: (row["action_rmse"], row["epoch"]))
    result = {
        "schema_version": "pcbc-recovery-checkpoint-selection-v1",
        "dataset": str(dataset_root),
        "latest_dagger_round": args.latest_dagger_round,
        "selection_split": "validation",
        "selection_behavior": "student",
        "selection_label_source": "MPPI",
        "episodes": episode_ids,
        "frames": int(obs.shape[0]),
        "selected": rows[0],
        "candidates": rows,
    }
    write_json(args.output, result)
    print(json.dumps(result["selected"], sort_keys=True))


if __name__ == "__main__":
    main()
