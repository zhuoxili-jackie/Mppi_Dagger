#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import ROOT, write_json

from lateral_mppi_dagger.evaluation.evaluator import evaluate_open_loop
from lateral_mppi_dagger.student.model import build_student_from_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate checkpoint action error on an episode-level split.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/bc_open_loop_metrics.json")
    args = parser.parse_args()
    model, _ = build_student_from_checkpoint(str(args.checkpoint), map_location=args.device)
    metrics = evaluate_open_loop(model, args.dataset, args.split, args.device)
    result = {
        "schema_version": "pcbc-open-loop-evaluation-v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "dataset": str(args.dataset.resolve()),
        "split": args.split,
        "metrics": metrics,
    }
    write_json(args.output, result)
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()

