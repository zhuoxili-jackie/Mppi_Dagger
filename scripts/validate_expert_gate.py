#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import ROOT, load_contract, write_json

from lateral_mppi_dagger.config import load_yaml, resolve_project_path, sha256_file
from lateral_mppi_dagger.evaluation.expert_gate import (
    ExpertGateConfig,
    evaluate_expert_gate,
    load_gate_seeds,
)
from lateral_mppi_dagger.contract.action16 import ActionContract
from lateral_mppi_dagger.reference.loader import ReferenceSet


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail-closed MPPI closed-loop expert gate.")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/expert_mppi.yaml")
    parser.add_argument(
        "--reference-config",
        type=str,
        default=None,
    )
    parser.add_argument("--output", type=Path, default=ROOT / "reports/05_expert_gate.json")
    parser.add_argument("--episode-count", type=int, default=None)
    parser.add_argument("--required-successes", type=int, default=None)
    args = parser.parse_args()

    expert_config = load_yaml(args.config)
    gate = expert_config["closed_loop_gate"]
    episode_count = int(args.episode_count or gate["fixed_seed_count"])
    required_successes = int(
        args.required_successes
        if args.required_successes is not None
        else gate["required_successes"]
    )
    seed_path = resolve_project_path(gate["seed_file"])
    implementation_paths = (
        "scripts/_isaac_workflow.py",
        "src/lateral_mppi_dagger/expert/mppi_expert.py",
        "src/lateral_mppi_dagger/env/isaac_mppi_rollout.py",
        "src/lateral_mppi_dagger/env/isaac_adapter.py",
        "src/lateral_mppi_dagger/env/action_delay.py",
        "src/lateral_mppi_dagger/env/scenarios.py",
        "src/lateral_mppi_dagger/data/collector.py",
        "src/lateral_mppi_dagger/contract/action16.py",
        "src/lateral_mppi_dagger/reference/loader.py",
    )
    expected_implementation_sha256 = {
        path: sha256_file(ROOT / path) for path in implementation_paths
    }
    references = ReferenceSet.from_config(
        args.reference_config
        or expert_config.get(
            "reference_config",
            "configs/reference_708.yaml",
        )
    )
    result = evaluate_expert_gate(
        args.dataset,
        ExpertGateConfig(
            expected_seeds=load_gate_seeds(seed_path, episode_count),
            required_successes=required_successes,
            full_episode_steps=int(gate["full_episode_steps"]),
            required_ref_ids=tuple(range(len(references))),
            required_teacher_valid_rate=float(gate["required_teacher_valid_rate"]),
            required_scenario_resolved_name=gate.get(
                "required_scenario_resolved_name"
            ),
            require_requested_reference_reset=bool(
                gate.get("require_requested_reference_reset", False)
            ),
            reset_reference_tolerance=float(
                gate.get("reset_reference_tolerance", 1.0e-5)
            ),
            require_implementation_hashes=bool(
                gate.get("require_implementation_hashes", False)
            ),
            expected_implementation_sha256=expected_implementation_sha256,
            tracking_thresholds=dict(gate["tracking_thresholds"]),
        ),
        references=references,
        action_contract=ActionContract.from_dict(load_contract()),
    )
    write_json(args.output, result)
    print(json.dumps({"ok": result["ok"], **result["summary"]}, sort_keys=True))
    if not result["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
