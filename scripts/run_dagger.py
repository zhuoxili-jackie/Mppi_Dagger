#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

from _bootstrap import ROOT

from lateral_mppi_dagger.config import load_yaml, sha256_file
from lateral_mppi_dagger.dagger.runner import DaggerRound, DaggerState


def _run(command: list[str]) -> None:
    print(json.dumps({"command": command}, ensure_ascii=False), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def _load_round(round_number: int) -> DaggerRound:
    config = load_yaml("configs/dagger.yaml")
    selected = [
        item for item in config["rounds"] if int(item["round"]) == round_number
    ]
    if len(selected) != 1:
        raise ValueError(f"DAgger config has no unique round {round_number}.")
    item = selected[0]
    result = DaggerRound(
        round=int(item["round"]),
        beta=float(item["beta"]),
        scenario=str(item["scenario"]),
    )
    result.validate()
    if result.round not in set(config["minimum_required_real_student_state_rounds"]):
        raise ValueError(
            "This runner is fail-closed to the required R1-R3 rounds; "
            f"round {result.round} is not enabled."
        )
    return result


def _require_gate(path: Path, checkpoint: Path) -> dict[str, Any]:
    values = json.loads(path.read_text(encoding="utf-8"))
    if values.get("ok") is not True:
        raise ValueError(f"Previous closed-loop gate is not passing: {path}")
    purpose = values.get("gate_purpose", "performance")
    if purpose == "dagger_admission":
        if values.get("dagger_admission_ok") is not True:
            raise ValueError(
                f"Previous DAgger admission gate is malformed or failed: {path}"
            )
    elif purpose != "performance":
        raise ValueError(f"Unsupported previous gate purpose {purpose!r}: {path}")
    hashes = values.get("summary", {}).get("student_checkpoint_hashes", [])
    checkpoint_hash = sha256_file(checkpoint)
    if hashes and hashes != [checkpoint_hash]:
        raise ValueError(
            "Previous gate was not produced by the requested input checkpoint: "
            f"gate={hashes}, checkpoint={checkpoint_hash}"
        )
    return values


def _last_training_epoch(output: Path) -> int | None:
    metrics = output / "metrics.jsonl"
    if not metrics.is_file():
        return None
    records = [
        json.loads(line)
        for line in metrics.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return int(records[-1]["epoch"]) if records else None


def _latest_epoch_checkpoint(output: Path) -> Path | None:
    candidates = sorted(output.glob("student_epoch_*.pt"))
    return candidates[-1] if candidates else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run one fail-closed R1/R2/R3 DAgger collection and training round. "
            "A passing beta=0 gate from the input checkpoint is required."
        )
    )
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--input-checkpoint", type=Path, required=True)
    parser.add_argument("--previous-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state", type=Path, default=ROOT / "reports/dagger_state.json")
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=ROOT / "reports",
        help="Run-scoped report directory; keeps separate training IDs isolated.",
    )
    parser.add_argument("--train-episodes", type=int, default=20)
    parser.add_argument("--validation-episodes", type=int, default=6)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed-base", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--reference-config",
        type=str,
        default="configs/reference_708.yaml",
    )
    parser.add_argument(
        "--mppi-config",
        type=str,
        default="configs/expert_mppi.yaml",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    round_config = _load_round(args.round)
    dataset = args.dataset.resolve()
    checkpoint = args.input_checkpoint.resolve()
    previous_gate = args.previous_gate.resolve()
    output = args.output.resolve()
    state_path = args.state.resolve()
    report_dir = args.report_dir.resolve()
    if report_dir != ROOT and ROOT not in report_dir.parents:
        raise ValueError("--report-dir must remain inside the project root.")
    report_dir.mkdir(parents=True, exist_ok=True)
    if not checkpoint.is_file():
        parser.error(f"Input checkpoint does not exist: {checkpoint}")
    if not previous_gate.is_file():
        parser.error(f"Previous gate report does not exist: {previous_gate}")
    if args.train_episodes <= 0 or args.validation_episodes <= 0:
        parser.error("Train and validation episode counts must be positive.")
    gate = _require_gate(previous_gate, checkpoint)

    if state_path.is_file():
        state = DaggerState.load(state_path)
    else:
        state = DaggerState()
    if round_config.round in state.completed_rounds:
        outputs = state.round_outputs[str(round_config.round)]
        print(json.dumps({"already_complete": outputs}, sort_keys=True))
        return
    if state.active_round is None:
        state.begin_round(round_config)
        state.save(state_path)
    elif state.active_round != round_config.round:
        raise RuntimeError(
            f"DAgger state has active round {state.active_round}, requested {round_config.round}."
        )
    elif not args.resume:
        raise RuntimeError(
            f"Round {round_config.round} is already active; rerun with --resume."
        )

    train_seed = args.seed_base + 1000 * round_config.round
    validation_seed = train_seed + 500
    common = [
        sys.executable,
        str(ROOT / "scripts/collect_expert.py"),
        "--headless",
        "--device",
        args.device,
        "--expert-backend",
        "mppi",
        "--dataset",
        str(dataset),
        "--steps",
        str(args.steps),
        "--rotate-references",
        "--scenario",
        round_config.scenario,
        "--reference-config",
        args.reference_config,
        "--mppi-config",
        args.mppi_config,
        "--dagger-round",
        str(round_config.round),
        "--beta",
        str(round_config.beta),
        "--student-checkpoint",
        str(checkpoint),
        "--resume",
    ]
    train_report = report_dir / f"dagger_r{round_config.round}_train_collection.json"
    validation_report = (
        report_dir / f"dagger_r{round_config.round}_validation_collection.json"
    )
    try:
        _run(
            common
            + [
                "--episodes",
                str(args.train_episodes),
                "--seed",
                str(train_seed),
                "--run-name",
                f"dagger_r{round_config.round}_train",
                "--split",
                "train",
                "--report",
                str(train_report),
            ]
        )
        _run(
            common
            + [
                "--episodes",
                str(args.validation_episodes),
                "--seed",
                str(validation_seed),
                "--run-name",
                f"dagger_r{round_config.round}_validation",
                "--split",
                "validation",
                "--report",
                str(validation_report),
            ]
        )
        validation_output = (
            report_dir / f"dagger_r{round_config.round}_dataset_validation.json"
        )
        _run(
            [
                sys.executable,
                str(ROOT / "scripts/validate_dataset.py"),
                str(dataset),
                "--output",
                str(validation_output),
            ]
        )
        round_gate_output = (
            report_dir / f"dagger_r{round_config.round}_collection_gate.json"
        )
        _run(
            [
                sys.executable,
                str(ROOT / "scripts/validate_dagger_round.py"),
                str(dataset),
                "--round",
                str(round_config.round),
                "--expected-episodes",
                str(args.train_episodes + args.validation_episodes),
                "--output",
                str(round_gate_output),
            ]
        )

        last_epoch = _last_training_epoch(output)
        best_checkpoint = output / "student_best_checkpoint.pt"
        if last_epoch is None or last_epoch < args.epochs - 1:
            train_command = [
                sys.executable,
                str(ROOT / "scripts/train_bc.py"),
                "--dataset",
                str(dataset),
                "--output",
                str(output),
                "--device",
                args.device,
                "--epochs",
                str(args.epochs),
                "--latest-dagger-round",
                str(round_config.round),
            ]
            if args.batch_size is not None:
                train_command.extend(["--batch-size", str(args.batch_size)])
            resume_checkpoint = _latest_epoch_checkpoint(output)
            if resume_checkpoint is not None:
                train_command.extend(["--resume", str(resume_checkpoint)])
            else:
                train_command.extend(["--initialize-from", str(checkpoint)])
            _run(train_command)
        if not best_checkpoint.is_file():
            raise FileNotFoundError(
                f"DAgger training did not produce {best_checkpoint}"
            )
        open_loop_report = (
            report_dir / f"dagger_r{round_config.round}_open_loop.json"
        )
        _run(
            [
                sys.executable,
                str(ROOT / "scripts/evaluate.py"),
                "--checkpoint",
                str(best_checkpoint),
                "--dataset",
                str(dataset),
                "--split",
                "validation",
                "--device",
                args.device,
                "--output",
                str(open_loop_report),
            ]
        )
        outputs = {
            "round": round_config.round,
            "beta": round_config.beta,
            "scenario": round_config.scenario,
            "dataset": str(dataset),
            "input_checkpoint": {
                "path": str(checkpoint),
                "sha256": sha256_file(checkpoint),
            },
            "previous_gate": str(previous_gate),
            "previous_gate_success_rate": gate["summary"]["success_rate"],
            "train_collection_report": str(train_report),
            "validation_collection_report": str(validation_report),
            "dataset_validation_report": str(validation_output),
            "collection_gate_report": str(round_gate_output),
            "open_loop_report": str(open_loop_report),
            "best_checkpoint": str(best_checkpoint),
            "best_checkpoint_sha256": sha256_file(best_checkpoint),
        }
        state.complete_round(round_config, outputs)
        state.save(state_path)
        print(json.dumps(outputs, sort_keys=True))
    except BaseException as exc:
        state.fail_round(
            type(exc).__name__,
            str(exc),
            {
                "traceback": traceback.format_exc(),
                "round": round_config.round,
                "dataset": str(dataset),
                "output": str(output),
            },
        )
        state.save(state_path)
        raise


if __name__ == "__main__":
    main()
