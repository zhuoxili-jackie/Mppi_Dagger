#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from _bootstrap import ROOT


def _run(command: list[str]) -> None:
    print(json.dumps({"command": command}, ensure_ascii=False), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run fast beta=0 student-only Isaac episodes and validate the closed-loop gate."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--scenario", default="nominal")
    parser.add_argument(
        "--reference-config",
        type=str,
        default="configs/reference_708.yaml",
    )
    parser.add_argument(
        "--tracking-config",
        type=str,
        default="configs/expert_mppi.yaml",
    )
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--success-rate-min", type=float, default=0.80)
    parser.add_argument("--per-reference-success-rate-min", type=float, default=0.50)
    parser.add_argument("--shield-intervention-rate-max", type=float, default=0.01)
    parser.add_argument("--expert-gate", type=Path, default=None)
    parser.add_argument("--max-expert-success-gap", type=float, default=0.05)
    parser.add_argument(
        "--gate-purpose",
        choices=("performance", "dagger_admission"),
        default="performance",
    )
    parser.add_argument(
        "--dagger-admission-full-horizon-success-rate-min",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--dagger-admission-per-reference-mean-horizon-fraction-min",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--dagger-admission-minimum-episode-horizon-fraction",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--dagger-admission-signed-progress-ratio-min",
        type=float,
        default=0.20,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/student_closed_loop_gate.json",
    )
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    dataset = args.dataset.resolve()
    if not checkpoint.is_file():
        parser.error(f"Checkpoint/model does not exist: {checkpoint}")
    collection_report = args.output.with_name(
        f"{args.output.stem}_collection.json"
    ).resolve()
    _run(
        [
            sys.executable,
            str(ROOT / "scripts/collect_expert.py"),
            "--headless",
            "--device",
            args.device,
            "--expert-backend",
            "disabled",
            "--dataset",
            str(dataset),
            "--episodes",
            str(args.episodes),
            "--steps",
            str(args.steps),
            "--seed",
            str(args.seed),
            "--run-name",
            args.run_name,
            "--scenario",
            args.scenario,
            "--reference-config",
            args.reference_config,
            "--split",
            "test",
            "--rotate-references",
            "--beta",
            "0.0",
            "--student-checkpoint",
            str(checkpoint),
            "--resume",
            "--report",
            str(collection_report),
        ]
    )
    command = [
        sys.executable,
        str(ROOT / "scripts/validate_student_gate.py"),
        str(dataset),
        "--seed-start",
        str(args.seed),
        "--episodes",
        str(args.episodes),
        "--steps",
        str(args.steps),
        "--scenario",
        args.scenario,
        "--reference-config",
        args.reference_config,
        "--tracking-config",
        args.tracking_config,
        "--success-rate-min",
        str(args.success_rate_min),
        "--per-reference-success-rate-min",
        str(args.per_reference_success_rate_min),
        "--shield-intervention-rate-max",
        str(args.shield_intervention_rate_max),
        "--output",
        str(args.output.resolve()),
        "--gate-purpose",
        args.gate_purpose,
    ]
    admission_values = (
        args.dagger_admission_full_horizon_success_rate_min,
        args.dagger_admission_per_reference_mean_horizon_fraction_min,
        args.dagger_admission_minimum_episode_horizon_fraction,
    )
    if any(value is not None for value in admission_values):
        if not all(value is not None for value in admission_values):
            parser.error(
                "All three --dagger-admission-* thresholds are required together."
            )
        command.extend(
            [
                "--dagger-admission-full-horizon-success-rate-min",
                str(args.dagger_admission_full_horizon_success_rate_min),
                "--dagger-admission-per-reference-mean-horizon-fraction-min",
                str(
                    args.dagger_admission_per_reference_mean_horizon_fraction_min
                ),
                "--dagger-admission-minimum-episode-horizon-fraction",
                str(args.dagger_admission_minimum_episode_horizon_fraction),
                "--dagger-admission-signed-progress-ratio-min",
                str(args.dagger_admission_signed_progress_ratio_min),
            ]
        )
    if args.expert_gate is not None:
        command.extend(
            [
                "--expert-gate",
                str(args.expert_gate.resolve()),
                "--max-expert-success-gap",
                str(args.max_expert_success_gap),
            ]
        )
    _run(command)


if __name__ == "__main__":
    main()
