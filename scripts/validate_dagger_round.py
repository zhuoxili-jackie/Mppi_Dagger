#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import ROOT, write_json

from lateral_mppi_dagger.config import load_yaml
from lateral_mppi_dagger.env.scenarios import load_scenario_profile
from lateral_mppi_dagger.evaluation.dagger_gate import (
    DaggerCollectionGateConfig,
    evaluate_dagger_collection_gate,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate that one DAgger round contains real MPPI-labeled student states."
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--expected-episodes", type=int, required=True)
    parser.add_argument(
        "--expected-beta",
        type=float,
        default=None,
        help="Override the configured beta for an explicitly named corrective collection.",
    )
    parser.add_argument(
        "--expected-scenario",
        default=None,
        help="Override the configured scenario for an explicitly named corrective collection.",
    )
    parser.add_argument(
        "--minimum-student-behavior-episodes",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/dagger_collection_gate.json",
    )
    args = parser.parse_args()
    dagger = load_yaml("configs/dagger.yaml")
    selected = [
        item for item in dagger["rounds"] if int(item["round"]) == args.round
    ]
    if len(selected) != 1:
        raise ValueError(f"No unique DAgger round {args.round}.")
    item = selected[0]
    expected_scenario = str(args.expected_scenario or item["scenario"])
    expected_beta = (
        float(item["beta"])
        if args.expected_beta is None
        else float(args.expected_beta)
    )
    scenario = load_scenario_profile(expected_scenario)
    result = evaluate_dagger_collection_gate(
        args.dataset,
        DaggerCollectionGateConfig(
            round_number=args.round,
            expected_scenario=expected_scenario,
            expected_beta=expected_beta,
            expected_episodes=args.expected_episodes,
            minimum_student_behavior_episodes=args.minimum_student_behavior_episodes,
        ),
        scenario,
    )
    write_json(args.output, result)
    print(json.dumps({"ok": result["ok"], **result["summary"]}, sort_keys=True))
    if not result["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
