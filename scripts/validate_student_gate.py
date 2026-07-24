#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import ROOT, load_contract, write_json

from lateral_mppi_dagger.contract.action16 import ActionContract
from lateral_mppi_dagger.config import load_yaml
from lateral_mppi_dagger.env.scenarios import load_scenario_profile
from lateral_mppi_dagger.evaluation.closed_loop_gate import (
    StudentClosedLoopGateConfig,
    evaluate_student_closed_loop_gate,
)
from lateral_mppi_dagger.reference.loader import ReferenceSet


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail-closed validation of beta=0 real-Isaac student episodes."
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--steps", type=int, default=300)
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
    parser.add_argument("--success-rate-min", type=float, default=0.80)
    parser.add_argument(
        "--per-reference-success-rate-min",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--teacher-valid-rate-min",
        type=float,
        default=0.0,
        help="Use 0 for the fast disabled-label evaluation backend.",
    )
    parser.add_argument("--shield-intervention-rate-max", type=float, default=0.01)
    parser.add_argument(
        "--expert-gate",
        type=Path,
        default=None,
        help="Optional passing expert gate on the identical seed set.",
    )
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
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/student_closed_loop_gate.json",
    )
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive.")
    references = ReferenceSet.from_config(args.reference_config)
    config = StudentClosedLoopGateConfig(
        expected_seeds=tuple(
            range(args.seed_start, args.seed_start + args.episodes)
        ),
        full_episode_steps=args.steps,
        success_rate_min=args.success_rate_min,
        per_reference_success_rate_min=args.per_reference_success_rate_min,
        teacher_valid_rate_min=args.teacher_valid_rate_min,
        shield_intervention_rate_max=args.shield_intervention_rate_max,
        required_ref_ids=tuple(range(len(references))),
        required_scenario_resolved_name=load_scenario_profile(
            args.scenario
        ).resolved_name,
        require_reset_reference_identity=True,
        tracking_thresholds=dict(
            load_yaml(args.tracking_config)["closed_loop_gate"][
                "tracking_thresholds"
            ]
        ),
        gate_purpose=args.gate_purpose,
        dagger_admission_full_horizon_success_rate_min=(
            args.dagger_admission_full_horizon_success_rate_min
        ),
        dagger_admission_per_reference_mean_horizon_fraction_min=(
            args.dagger_admission_per_reference_mean_horizon_fraction_min
        ),
        dagger_admission_minimum_episode_horizon_fraction=(
            args.dagger_admission_minimum_episode_horizon_fraction
        ),
        dagger_admission_signed_progress_ratio_min=(
            args.dagger_admission_signed_progress_ratio_min
        ),
    )
    result = evaluate_student_closed_loop_gate(
        args.dataset,
        config,
        references,
        ActionContract.from_dict(load_contract()),
    )
    if args.expert_gate is not None:
        expert_gate = json.loads(
            args.expert_gate.resolve().read_text(encoding="utf-8")
        )
        if expert_gate.get("ok") is not True:
            raise ValueError("The comparison expert gate is not passing.")
        expert_seeds = expert_gate.get("config", {}).get("expected_seeds")
        expected_seeds = list(config.expected_seeds)
        if expert_seeds != expected_seeds:
            raise ValueError(
                "Student and expert gate seed sets differ; a success-gap comparison "
                "would not be valid."
            )
        expert_success_rate = float(expert_gate["summary"]["success_rate"])
        success_gap = expert_success_rate - float(result["summary"]["success_rate"])
        gap_pass = success_gap <= args.max_expert_success_gap
        result["expert_comparison"] = {
            "expert_gate": str(args.expert_gate.resolve()),
            "expert_success_rate": expert_success_rate,
            "student_success_rate": result["summary"]["success_rate"],
            "expert_minus_student_success_gap": success_gap,
            "maximum_allowed_gap": args.max_expert_success_gap,
            "pass": gap_pass,
        }
        result["checks"]["expert_success_gap"] = gap_pass
        result["ok"] = bool(all(result["checks"].values()))
    write_json(args.output, result)
    print(
        json.dumps(
            {
                "ok": result["ok"],
                "gate_purpose": result["gate_purpose"],
                "performance_ok": result["performance_ok"],
                "dagger_admission_ok": result["dagger_admission_ok"],
                **result["summary"],
            },
            sort_keys=True,
        )
    )
    if not result["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
