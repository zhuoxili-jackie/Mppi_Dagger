#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import traceback
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from _bootstrap import ROOT, load_contract, write_json

from lateral_mppi_dagger.config import load_yaml
from lateral_mppi_dagger.student.trainer import BCTrainer, TrainerConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the standalone 93D-to-16D BC student.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--max-batches-per-epoch", type=int, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--initialize-from", type=Path, default=None)
    parser.add_argument(
        "--latest-dagger-round",
        type=int,
        default=None,
        help="Enable 30/40/30 R0/latest/historical sampling for this DAgger round.",
    )
    parser.add_argument(
        "--sampling-policy",
        choices=("uniform", "dagger_mix", "dagger_recovery"),
        default=None,
        help=(
            "Override dataset sampling. dagger_recovery explicitly upweights "
            "latest-round student-executed MPPI labels."
        ),
    )
    parser.add_argument("--sampling-mix-initial", type=float, default=None)
    parser.add_argument("--sampling-mix-latest", type=float, default=None)
    parser.add_argument("--sampling-mix-historical", type=float, default=None)
    parser.add_argument("--sampling-mix-latest-student", type=float, default=None)
    parser.add_argument("--sampling-mix-latest-teacher", type=float, default=None)
    args = parser.parse_args()
    contract = load_contract()
    config = TrainerConfig.from_dict(load_yaml("configs/student.yaml"))
    overrides = {}
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        if args.learning_rate <= 0.0:
            parser.error("--learning-rate must be positive.")
        overrides["learning_rate"] = args.learning_rate
    if args.max_batches_per_epoch is not None:
        overrides["max_batches_per_epoch"] = args.max_batches_per_epoch
    if args.latest_dagger_round is not None:
        overrides["sampling_policy"] = "dagger_mix"
        overrides["latest_dagger_round"] = args.latest_dagger_round
    if args.sampling_policy is not None:
        overrides["sampling_policy"] = args.sampling_policy
    for argument, field_name in (
        (args.sampling_mix_initial, "sampling_mix_initial"),
        (args.sampling_mix_latest, "sampling_mix_latest"),
        (args.sampling_mix_historical, "sampling_mix_historical"),
        (args.sampling_mix_latest_student, "sampling_mix_latest_student"),
        (args.sampling_mix_latest_teacher, "sampling_mix_latest_teacher"),
    ):
        if argument is not None:
            overrides[field_name] = argument
    config = replace(config, **overrides)
    if args.resume is not None and args.initialize_from is not None:
        parser.error("--resume and --initialize-from are mutually exclusive.")
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output / "resolved_train_config.json",
        {
            "trainer": asdict(config),
            "dataset": str(args.dataset.resolve()),
            "contract_schema": contract["schema_version"],
        },
    )
    try:
        trainer = BCTrainer(
            dataset_root=args.dataset,
            output_dir=args.output,
            config=config,
            raw_min=np.asarray(contract["action"]["raw_min"], dtype=np.float32),
            raw_max=np.asarray(contract["action"]["raw_max"], dtype=np.float32),
            action_scale=np.asarray(contract["action"]["scale"], dtype=np.float32),
            device=args.device,
        )
        if args.resume is not None:
            trainer.resume(args.resume)
        elif args.initialize_from is not None:
            trainer.initialize_from(args.initialize_from)
        best = trainer.train()
    except Exception as exc:
        write_json(
            args.output / "failure_state.json",
            {
                "error_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
                "resume_requested": str(args.resume) if args.resume else None,
                "config": asdict(config),
            },
        )
        raise
    print(json.dumps({"best_checkpoint": str(best)}, sort_keys=True))


if __name__ == "__main__":
    main()
