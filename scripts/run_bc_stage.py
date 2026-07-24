#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from _bootstrap import ROOT, load_contract, write_json

from lateral_mppi_dagger.config import load_yaml, sha256_file
from lateral_mppi_dagger.contract.action16 import ActionContract
from lateral_mppi_dagger.data.dataset import load_manifest
from lateral_mppi_dagger.data.schema import read_episode_shard
from lateral_mppi_dagger.evaluation.expert_gate import (
    ExpertGateConfig,
    evaluate_expert_gate,
)
from lateral_mppi_dagger.reference.loader import ReferenceSet


def _run(command: list[str]) -> None:
    print(json.dumps({"command": command}, ensure_ascii=False), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def _latest_epoch_checkpoint(output: Path) -> Path | None:
    candidates = sorted(output.glob("student_epoch_*.pt"))
    return candidates[-1] if candidates else None


def _last_epoch(output: Path) -> int | None:
    metrics = output / "metrics.jsonl"
    if not metrics.is_file():
        return None
    records = [
        json.loads(line)
        for line in metrics.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return int(records[-1]["epoch"]) if records else None


def _seed_dataset_from_base(base: Path, target: Path) -> dict[str, object]:
    """Copy a validated R0 base into the aggregate without creating dependencies."""
    base = base.resolve()
    target = target.resolve()
    if base == target:
        raise ValueError("R0 base_dataset and dataset must be different directories.")
    if not base.is_dir():
        raise FileNotFoundError(f"R0 base dataset does not exist: {base}")

    base_records = load_manifest(base)
    if not base_records:
        raise ValueError(f"R0 base dataset is empty: {base}")
    for record in base_records:
        read_episode_shard(base / record["path"])

    copied = False
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=f".{target.name}.seed.",
            dir=target.parent,
        ) as temporary:
            staged = Path(temporary) / target.name
            shutil.copytree(base, staged)
            staged.replace(target)
        copied = True
    if not target.is_dir():
        raise NotADirectoryError(f"R0 aggregate path is not a directory: {target}")

    target_records = load_manifest(target)
    target_by_id = {
        str(record["episode_id"]): record for record in target_records
    }
    for base_record in base_records:
        episode_id = str(base_record["episode_id"])
        target_record = target_by_id.get(episode_id)
        if target_record != base_record:
            raise ValueError(
                f"R0 aggregate does not contain an identical base record: {episode_id}"
            )
        relative_path = Path(str(base_record["path"]))
        source_shard = base / relative_path
        target_shard = target / relative_path
        if not target_shard.is_file():
            raise FileNotFoundError(
                f"R0 aggregate is missing base shard: {target_shard}"
            )
        if sha256_file(source_shard) != sha256_file(target_shard):
            raise ValueError(
                f"R0 aggregate base shard differs byte-for-byte: {episode_id}"
            )
        read_episode_shard(target_shard)

    return {
        "base_dataset": str(base),
        "base_manifest_sha256": sha256_file(base / "manifest.jsonl"),
        "base_episodes": len(base_records),
        "base_frames": sum(int(record["steps"]) for record in base_records),
        "copied": copied,
    }


def _current_implementation_hashes(
    expected: dict[str, str],
) -> dict[str, str]:
    current: dict[str, str] = {}
    for relative_path in expected:
        path = ROOT / relative_path
        if not path.is_file():
            raise FileNotFoundError(
                f"Expert implementation source is missing: {path}"
            )
        current[relative_path] = sha256_file(path)
    return current


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect the formal R0 dataset, pass observability, and train BC."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/formal_pipeline.yaml",
    )
    parser.add_argument("--expert-gate", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = load_yaml(args.config)
    expert_gate_path = args.expert_gate.resolve()
    expert_gate = json.loads(expert_gate_path.read_text(encoding="utf-8"))
    if expert_gate.get("ok") is not True:
        raise ValueError("Formal R0 collection requires a passing expert gate.")

    r0 = config["r0"]
    dataset = (ROOT / r0["dataset"]).resolve()
    base_dataset = (ROOT / r0["base_dataset"]).resolve()
    gate_dataset = Path(str(expert_gate.get("dataset", ""))).resolve()
    if gate_dataset != base_dataset:
        raise ValueError(
            "The passing expert gate must refer to r0.base_dataset exactly: "
            f"{gate_dataset} != {base_dataset}"
        )
    gate_config = expert_gate.get("config", {})
    expected_implementation_sha256 = gate_config.get(
        "expected_implementation_sha256"
    )
    if not isinstance(expected_implementation_sha256, dict) or not (
        expected_implementation_sha256
    ):
        raise ValueError(
            "The passing expert gate has no expected implementation hashes."
        )
    current_implementation_sha256 = _current_implementation_hashes(
        expected_implementation_sha256
    )
    if current_implementation_sha256 != expected_implementation_sha256:
        raise RuntimeError(
            "Expert implementation changed after the formal gate; collect and "
            "validate a new base dataset before continuing."
        )
    base_seed_info = _seed_dataset_from_base(base_dataset, dataset)
    if int(base_seed_info["base_frames"]) != int(r0["base_frames"]):
        raise ValueError(
            "Configured r0.base_frames does not match the passing base dataset: "
            f"{r0['base_frames']} != {base_seed_info['base_frames']}"
        )

    steps = int(config["episode_steps"])
    scenario = str(r0["scenario"])
    for split in ("train", "validation", "test"):
        split_cfg = r0[split]
        report = ROOT / f"reports/r0_{split}_collection.json"
        _run(
            [
                sys.executable,
                str(ROOT / "scripts/collect_expert.py"),
                "--headless",
                "--device",
                args.device,
                "--expert-backend",
                "mppi",
                "--dataset",
                str(dataset),
                "--episodes",
                str(split_cfg["episodes"]),
                "--steps",
                str(steps),
                "--seed",
                str(split_cfg["seed"]),
                "--run-name",
                f"r0_{split}",
                "--scenario",
                scenario,
                "--dagger-round",
                "0",
                "--split",
                split,
                "--rotate-references",
                "--resume",
                "--report",
                str(report),
            ]
        )

    dataset_validation = ROOT / "reports/r0_dataset_validation.json"
    observability = ROOT / "reports/r0_observability_gate.json"
    _run(
        [
            sys.executable,
            str(ROOT / "scripts/validate_dataset.py"),
            str(dataset),
            "--output",
            str(dataset_validation),
        ]
    )
    added_seeds = tuple(
        seed
        for split in ("train", "validation", "test")
        for seed in range(
            int(r0[split]["seed"]),
            int(r0[split]["seed"]) + int(r0[split]["episodes"]),
        )
    )
    base_seeds = tuple(int(seed) for seed in gate_config["expected_seeds"])
    expected_seeds = base_seeds + added_seeds
    if len(set(expected_seeds)) != len(expected_seeds):
        raise ValueError("R0 base and added seed ranges overlap.")

    records = load_manifest(dataset)
    total_frames = sum(int(record["steps"]) for record in records)
    train_frames = sum(
        int(record["steps"])
        for record in records
        if record["split"] == "train"
    )
    if total_frames != int(r0["total_frames"]):
        raise RuntimeError(
            f"R0 frame count mismatch: {total_frames} != {r0['total_frames']}"
        )
    if train_frames != int(r0["train_frames"]):
        raise RuntimeError(
            f"R0 train frame count mismatch: {train_frames} != {r0['train_frames']}"
        )

    mppi_gate_config = load_yaml("configs/expert_mppi.yaml")["closed_loop_gate"]
    r0_gate = evaluate_expert_gate(
        dataset,
        ExpertGateConfig(
            expected_seeds=expected_seeds,
            required_successes=math.ceil(0.95 * len(expected_seeds)),
            full_episode_steps=steps,
            required_teacher_valid_rate=float(
                mppi_gate_config["required_teacher_valid_rate"]
            ),
            required_scenario_resolved_name=mppi_gate_config.get(
                "required_scenario_resolved_name"
            ),
            require_requested_reference_reset=bool(
                mppi_gate_config.get(
                    "require_requested_reference_reset",
                    False,
                )
            ),
            reset_reference_tolerance=float(
                mppi_gate_config.get("reset_reference_tolerance", 1.0e-5)
            ),
            require_implementation_hashes=bool(
                mppi_gate_config.get("require_implementation_hashes", False)
            ),
            expected_implementation_sha256=expected_implementation_sha256,
            tracking_thresholds=dict(
                mppi_gate_config["tracking_thresholds"]
            ),
        ),
        references=ReferenceSet.from_config(),
        action_contract=ActionContract.from_dict(load_contract()),
    )
    r0_gate_output = ROOT / "reports/r0_expert_dataset_gate.json"
    write_json(r0_gate_output, r0_gate)
    if not r0_gate["ok"]:
        raise RuntimeError(
            f"Formal R0 expert dataset gate failed; see {r0_gate_output}"
        )
    _run(
        [
            sys.executable,
            str(ROOT / "scripts/analyze_observability.py"),
            str(dataset),
            "--output",
            str(observability),
        ]
    )

    bc = config["bc"]
    output = (ROOT / bc["output"]).resolve()
    epochs = int(bc["epochs"])
    if _last_epoch(output) is None or int(_last_epoch(output)) < epochs - 1:
        command = [
            sys.executable,
            str(ROOT / "scripts/train_bc.py"),
            "--dataset",
            str(dataset),
            "--output",
            str(output),
            "--device",
            args.device,
            "--epochs",
            str(epochs),
            "--batch-size",
            str(bc["batch_size"]),
        ]
        resume_checkpoint = _latest_epoch_checkpoint(output)
        if resume_checkpoint is not None:
            command.extend(["--resume", str(resume_checkpoint)])
        _run(command)
    best = output / "student_best_checkpoint.pt"
    if not best.is_file():
        raise FileNotFoundError(f"BC did not produce {best}")
    open_loop = ROOT / "reports/bc_formal_open_loop_test.json"
    _run(
        [
            sys.executable,
            str(ROOT / "scripts/evaluate.py"),
            "--checkpoint",
            str(best),
            "--dataset",
            str(dataset),
            "--split",
            "test",
            "--device",
            args.device,
            "--output",
            str(open_loop),
        ]
    )
    result = {
        "schema_version": "pcbc-bc-stage-result-v1",
        "expert_gate": str(expert_gate_path),
        "dataset": str(dataset),
        "base_seed_info": base_seed_info,
        "configured_total_frames": int(r0["total_frames"]),
        "configured_train_frames": int(r0["train_frames"]),
        "dataset_validation": str(dataset_validation),
        "r0_expert_dataset_gate": str(r0_gate_output),
        "observability_gate": str(observability),
        "best_checkpoint": str(best),
        "best_checkpoint_sha256": sha256_file(best),
        "open_loop_report": str(open_loop),
    }
    output_report = ROOT / "reports/bc_stage_result.json"
    write_json(output_report, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
