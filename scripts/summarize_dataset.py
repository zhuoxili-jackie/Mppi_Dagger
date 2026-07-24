#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from _bootstrap import ROOT, write_json

from lateral_mppi_dagger.data.dataset import load_manifest
from lateral_mppi_dagger.data.schema import ENUMS, read_episode_shard


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize frames, rounds, scenarios, behavior, validity, and provenance."
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/dataset_summary.json",
    )
    args = parser.parse_args()
    root = args.dataset.resolve()
    records = load_manifest(root)
    split_episodes: Counter[str] = Counter()
    split_frames: Counter[str] = Counter()
    round_episodes: Counter[int] = Counter()
    round_frames: Counter[int] = Counter()
    round_student_frames: Counter[int] = Counter()
    scenario_episodes: Counter[str] = Counter()
    reference_episodes: Counter[int] = Counter()
    successes: Counter[str] = Counter()
    expert_hashes: set[str] = set()
    student_hashes: set[str] = set()
    total_teacher_valid = 0
    total_student_valid = 0
    total_shield = 0
    total_frames = 0
    solve_values = []
    scenario_frames: defaultdict[str, int] = defaultdict(int)
    exact_zero = True

    for record in records:
        shard = read_episode_shard(root / record["path"])
        arrays = shard.arrays
        metadata = shard.metadata
        steps = int(arrays["step_id"].shape[0])
        round_number = int(metadata.get("dagger_round", record.get("dagger_round", 0)))
        scenario = str(metadata.get("scenario", record.get("scenario", "UNKNOWN")))
        split = str(record["split"])
        total_frames += steps
        split_episodes[split] += 1
        split_frames[split] += steps
        round_episodes[round_number] += 1
        round_frames[round_number] += steps
        scenario_episodes[scenario] += 1
        scenario_frames[scenario] += steps
        reference_episodes[int(record["ref_id"])] += 1
        successes["success" if record["success"] else "failure"] += 1
        total_teacher_valid += int(np.sum(arrays["teacher_valid"]))
        total_student_valid += int(np.sum(arrays["student_valid"]))
        total_shield += int(np.sum(arrays["shield_intervened"]))
        round_student_frames[round_number] += int(
            np.sum(
                arrays["behavior_policy"]
                == ENUMS["behavior_policy"]["STUDENT"]
            )
        )
        finite_solve = arrays["solve_ms"][np.isfinite(arrays["solve_ms"])]
        if finite_solve.size:
            solve_values.append(finite_solve)
        expert_hash = metadata.get("expert_config_hash")
        if expert_hash:
            expert_hashes.add(str(expert_hash))
        student = metadata.get("student_checkpoint")
        if isinstance(student, dict) and student.get("sha256"):
            student_hashes.add(str(student["sha256"]))
        exact_zero &= bool(
            np.array_equal(
                arrays["executed_action16"][:, 12:],
                np.zeros_like(arrays["executed_action16"][:, 12:]),
            )
        )

    solve = np.concatenate(solve_values) if solve_values else np.asarray([], dtype=np.float32)
    result = {
        "schema_version": "pcbc-dataset-summary-v1",
        "dataset": str(root),
        "episodes": len(records),
        "frames": total_frames,
        "compressed_bytes": sum(
            (root / record["path"]).stat().st_size for record in records
        ),
        "split_episodes": dict(sorted(split_episodes.items())),
        "split_frames": dict(sorted(split_frames.items())),
        "round_episodes": {
            str(key): value for key, value in sorted(round_episodes.items())
        },
        "round_frames": {
            str(key): value for key, value in sorted(round_frames.items())
        },
        "round_student_behavior_frames": {
            str(key): value for key, value in sorted(round_student_frames.items())
        },
        "scenario_episodes": dict(sorted(scenario_episodes.items())),
        "scenario_frames": dict(sorted(scenario_frames.items())),
        "reference_episodes": {
            str(key): value for key, value in sorted(reference_episodes.items())
        },
        "success_counts": dict(successes),
        "teacher_valid_rate": (
            total_teacher_valid / total_frames if total_frames else 0.0
        ),
        "student_valid_rate": (
            total_student_valid / total_frames if total_frames else 0.0
        ),
        "shield_intervention_rate": (
            total_shield / total_frames if total_frames else 0.0
        ),
        "mean_teacher_solve_ms": float(np.mean(solve)) if solve.size else None,
        "p95_teacher_solve_ms": float(np.percentile(solve, 95)) if solve.size else None,
        "wheel_action_exact_zero": exact_zero,
        "expert_config_hashes": sorted(expert_hashes),
        "student_checkpoint_hashes": sorted(student_hashes),
    }
    write_json(args.output, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
