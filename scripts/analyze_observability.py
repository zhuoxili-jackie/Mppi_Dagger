#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from _bootstrap import ROOT, write_json

from lateral_mppi_dagger.data.dataset import load_manifest
from lateral_mppi_dagger.data.schema import read_episode_shard
from lateral_mppi_dagger.evaluation.observability import analyze_observability


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare fixed-prefix and dynamic-reference observation aliasing.")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--max-samples", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/observability_gate.json")
    args = parser.parse_args()
    fixed, dynamic, action, contact, phase = [], [], [], [], []
    root = args.dataset.resolve()
    for record in load_manifest(root):
        shard = read_episode_shard(root / record["path"])
        if "obs93_dynamic" not in shard.arrays:
            raise ValueError(f"{record['episode_id']} has no obs93_dynamic diagnostic field.")
        valid = shard.arrays["teacher_valid"].astype(bool)
        fixed.append(shard.arrays["obs93_clean"][valid])
        dynamic.append(shard.arrays["obs93_dynamic"][valid])
        action.append(shard.arrays["teacher_action16"][valid])
        contact.append(shard.arrays["desired_contact"][valid])
        phase.append(shard.arrays["phase"][valid])
    arrays = [np.concatenate(values, axis=0) for values in (fixed, dynamic, action, contact, phase)]
    if arrays[0].shape[0] > args.max_samples:
        rng = np.random.default_rng(args.seed)
        selected = np.sort(rng.choice(arrays[0].shape[0], size=args.max_samples, replace=False))
        arrays = [value[selected] for value in arrays]
    fixed_result = analyze_observability(arrays[0], arrays[2], arrays[3], arrays[4])
    dynamic_result = analyze_observability(arrays[1], arrays[2], arrays[3], arrays[4])
    systematic = (
        fixed_result.conflict_rate >= 0.05
        and dynamic_result.conflict_rate <= 0.5 * fixed_result.conflict_rate
    )
    result = {
        "schema_version": "pcbc-observability-gate-v1",
        "dataset": str(root),
        "fixed_prefix": asdict(fixed_result),
        "dynamic_reference": asdict(dynamic_result),
        "fixed_prefix_systematically_aliased": systematic,
        "gate": "BLOCK_FORMAL_BC_DAGGER" if systematic else "PASS",
    }
    write_json(args.output, result)
    print(json.dumps({"gate": result["gate"], "fixed": fixed_result.conflict_rate, "dynamic": dynamic_result.conflict_rate}))
    if systematic:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

