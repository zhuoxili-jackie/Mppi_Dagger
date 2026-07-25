#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _bootstrap import ROOT, write_json

from lateral_mppi_dagger.config import sha256_file
from lateral_mppi_dagger.reference.loader import ReferenceSet


def _inside_root(path: Path) -> bool:
    resolved = path.expanduser().resolve()
    return resolved != ROOT and ROOT in resolved.parents


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a provenance-locked physical q_des ILC correction from "
            "the gait-tracking error between paired moving and standing "
            "Isaac replays. This creates a diagnostic correction trajectory, "
            "not training data or a deployment policy."
        )
    )
    parser.add_argument(
        "--reference-config",
        default="configs/low_load_lateral/train_001/reference.yaml",
    )
    parser.add_argument("--ref-id", type=int, required=True)
    parser.add_argument("--standing-ref-id", type=int, default=8)
    parser.add_argument("--moving-episode", type=Path, required=True)
    parser.add_argument("--standing-episode", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    output_path = args.output.expanduser().resolve()
    report_path = args.report.expanduser().resolve()
    if not _inside_root(output_path) or not _inside_root(report_path):
        parser.error("Output and report must be inside the standalone root.")
    if output_path.exists():
        parser.error(f"Refusing to overwrite existing output {output_path}.")

    references = ReferenceSet.from_config(args.reference_config)
    if not 0 <= args.ref_id < len(references):
        parser.error("--ref-id is outside the reference bank.")
    if not 0 <= args.standing_ref_id < len(references):
        parser.error("--standing-ref-id is outside the reference bank.")
    if args.ref_id == args.standing_ref_id:
        parser.error("--ref-id must be a moving reference.")

    moving_path = args.moving_episode.expanduser().resolve()
    standing_path = args.standing_episode.expanduser().resolve()
    with np.load(moving_path, allow_pickle=False) as archive:
        moving_q = np.asarray(archive["q"][:, :12], dtype=np.float32)
        moving_frames = np.asarray(archive["ref_frame"], dtype=np.int64)
        moving_ref_ids = np.asarray(archive["ref_id"], dtype=np.int64)
        moving_wheel_action = np.asarray(
            archive["executed_action16"][:, 12:],
            dtype=np.float32,
        )
    with np.load(standing_path, allow_pickle=False) as archive:
        standing_q = np.asarray(archive["q"][:, :12], dtype=np.float32)
        standing_frames = np.asarray(
            archive["ref_frame"],
            dtype=np.int64,
        )
        standing_ref_ids = np.asarray(
            archive["ref_id"],
            dtype=np.int64,
        )
        standing_wheel_action = np.asarray(
            archive["executed_action16"][:, 12:],
            dtype=np.float32,
        )

    steps = min(moving_q.shape[0], standing_q.shape[0])
    if steps < 1:
        raise ValueError("Paired episodes contain no transitions.")
    if not np.all(moving_ref_ids[:steps] == args.ref_id):
        raise ValueError("Moving episode ref_id does not match --ref-id.")
    if not np.all(standing_ref_ids[:steps] == args.standing_ref_id):
        raise ValueError(
            "Standing episode ref_id does not match --standing-ref-id."
        )
    if not np.array_equal(
        moving_wheel_action[:steps],
        np.zeros_like(moving_wheel_action[:steps]),
    ) or not np.array_equal(
        standing_wheel_action[:steps],
        np.zeros_like(standing_wheel_action[:steps]),
    ):
        raise ValueError("Paired episodes must have exact-zero wheel actions.")
    if not np.array_equal(
        moving_frames[:steps],
        standing_frames[:steps],
    ):
        raise ValueError("Paired episode reference frames do not align.")

    moving_reference = references[args.ref_id]
    standing_reference = references[args.standing_ref_id]
    frames = np.minimum(
        moving_frames[:steps],
        min(moving_reference.frames, standing_reference.frames) - 1,
    )
    target_gait_delta = (
        moving_reference.joint_pos[frames, :12]
        - standing_reference.joint_pos[frames, :12]
    ).astype(np.float32)
    actual_gait_delta = (
        moving_q[:steps] - standing_q[:steps]
    ).astype(np.float32)
    physical_correction = (
        target_gait_delta - actual_gait_delta
    ).astype(np.float32)
    if not np.isfinite(physical_correction).all():
        raise ValueError("Computed ILC correction contains NaN or Inf.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        schema_version=np.asarray(
            ["pcbc-low-load-ilc-correction-v1"],
            dtype="U40",
        ),
        ref_id=np.asarray([args.ref_id], dtype=np.int32),
        standing_ref_id=np.asarray(
            [args.standing_ref_id],
            dtype=np.int32,
        ),
        ref_frame=frames.astype(np.int32),
        physical_correction_leg=physical_correction,
        target_gait_delta_leg=target_gait_delta,
        actual_gait_delta_leg=actual_gait_delta,
    )
    report = {
        "schema_version": "pcbc-low-load-ilc-correction-report-v1",
        "status": "diagnostic_not_training_data",
        "reference_config": args.reference_config,
        "ref_id": args.ref_id,
        "standing_ref_id": args.standing_ref_id,
        "moving_episode": str(moving_path),
        "moving_episode_sha256": sha256_file(moving_path),
        "standing_episode": str(standing_path),
        "standing_episode_sha256": sha256_file(standing_path),
        "steps": steps,
        "target_gait_delta_rmse_rad": float(
            np.sqrt(np.mean(np.square(target_gait_delta)))
        ),
        "actual_gait_delta_rmse_rad": float(
            np.sqrt(np.mean(np.square(actual_gait_delta)))
        ),
        "correction_rmse_rad": float(
            np.sqrt(np.mean(np.square(physical_correction)))
        ),
        "correction_max_abs_rad": float(
            np.max(np.abs(physical_correction))
        ),
        "correction_rmse_rad_by_joint": np.sqrt(
            np.mean(np.square(physical_correction), axis=0)
        ).tolist(),
        "wheel_action_exact_zero": True,
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
    }
    write_json(report_path, report)
    print(report)


if __name__ == "__main__":
    main()
