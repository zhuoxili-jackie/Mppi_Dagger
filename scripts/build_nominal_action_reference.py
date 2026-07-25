#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _bootstrap import ROOT, load_contract, write_json

from lateral_mppi_dagger.config import sha256_file
from lateral_mppi_dagger.contract.action16 import ActionContract
from lateral_mppi_dagger.reference.loader import ReferenceSet


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a hashed physical q_des proposal trajectory from a "
            "successful bounded Isaac action replay. This asset centres MPPI "
            "proposals; it is not a state reference or training dataset."
        )
    )
    parser.add_argument(
        "--episode",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--reference-config",
        default="configs/low_load_lateral/train_001/reference.yaml",
    )
    parser.add_argument("--ref-id", type=int, default=8)
    parser.add_argument("--lookahead-steps", type=int, default=1)
    parser.add_argument(
        "--physical-offset-leg-rad",
        type=float,
        nargs=12,
        default=(
            0.025,
            0.015,
            0.0,
            0.0,
            -0.025,
            -0.005,
            -0.010,
            0.020,
            0.020,
            0.015,
            0.015,
            0.015,
        ),
    )
    parser.add_argument(
        "--physical-target-rate-limit-rad-s",
        type=float,
        default=2.25,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT
        / "reports/low_load_lateral/train_001/diagnostics/nominal_action_reference.json",
    )
    args = parser.parse_args()

    if args.lookahead_steps < 0:
        parser.error("--lookahead-steps must be non-negative.")
    if args.physical_target_rate_limit_rad_s <= 0.0:
        parser.error(
            "--physical-target-rate-limit-rad-s must be positive."
        )
    physical_offset = np.asarray(
        args.physical_offset_leg_rad,
        dtype=np.float32,
    )
    if physical_offset.shape != (12,) or not np.isfinite(
        physical_offset
    ).all():
        parser.error(
            "--physical-offset-leg-rad must contain 12 finite values."
        )

    references = ReferenceSet.from_config(args.reference_config)
    if not 0 <= args.ref_id < len(references):
        parser.error("--ref-id is outside the selected reference bank.")
    frames = references[args.ref_id].frames
    contract = ActionContract.from_dict(load_contract())
    scale = np.asarray(contract.scale[:12], dtype=np.float32)
    q_offset = np.asarray(
        contract.q_action_offset_runtime[:12],
        dtype=np.float32,
    )
    raw_min = np.asarray(contract.raw_min[:12], dtype=np.float32)
    raw_max = np.asarray(contract.raw_max[:12], dtype=np.float32)
    maximum_physical_delta = (
        args.physical_target_rate_limit_rad_s
        / references[args.ref_id].fps
    )
    maximum_raw_delta = maximum_physical_delta / scale

    episode_path = args.episode.expanduser().resolve()
    with np.load(episode_path, allow_pickle=False) as archive:
        source_action = np.asarray(
            archive["executed_action16"],
            dtype=np.float32,
        )
        ref_frame = np.asarray(archive["ref_frame"], dtype=np.int64)
        source_ref_id = np.asarray(archive["ref_id"], dtype=np.int64)
    if source_action.ndim != 2 or source_action.shape[1] != 16:
        raise ValueError(
            f"executed_action16 must have shape [steps,16], got "
            f"{source_action.shape}."
        )
    if not np.array_equal(
        source_action[:, 12:],
        np.zeros_like(source_action[:, 12:]),
    ):
        raise ValueError("Source episode contains non-zero wheel actions.")
    if not np.all(source_ref_id == args.ref_id):
        raise ValueError(
            "Source episode ref_id does not match the requested ref_id."
        )
    if not np.array_equal(
        ref_frame,
        np.arange(source_action.shape[0], dtype=np.int64),
    ):
        raise ValueError(
            "Source reference frames must be contiguous from zero."
        )

    projected_raw = np.zeros(
        (source_action.shape[0], 12),
        dtype=np.float32,
    )
    previous = np.zeros(12, dtype=np.float32)
    for step, source in enumerate(source_action[:, :12]):
        proposed = source + physical_offset / scale
        projected = np.maximum(
            np.minimum(proposed, previous + maximum_raw_delta),
            previous - maximum_raw_delta,
        )
        projected = np.maximum(
            np.minimum(projected, raw_max),
            raw_min,
        ).astype(np.float32)
        projected_raw[step] = projected
        previous = projected

    q_des_leg = np.repeat(q_offset[None], frames, axis=0)
    raw_action_leg = np.zeros((frames, 12), dtype=np.float32)
    source_step_id = np.full(frames, -1, dtype=np.int32)
    last_assigned_frame = 0
    for step, source_frame in enumerate(ref_frame):
        target_frame = min(
            int(source_frame) + args.lookahead_steps,
            frames - 1,
        )
        raw_action_leg[target_frame] = projected_raw[step]
        q_des_leg[target_frame] = (
            q_offset + scale * projected_raw[step]
        )
        source_step_id[target_frame] = step
        last_assigned_frame = max(last_assigned_frame, target_frame)
    for frame in range(1, frames):
        if source_step_id[frame] < 0:
            raw_action_leg[frame] = raw_action_leg[frame - 1]
            q_des_leg[frame] = q_des_leg[frame - 1]
    if last_assigned_frame + 1 < frames:
        raw_action_leg[last_assigned_frame + 1 :] = raw_action_leg[
            last_assigned_frame
        ]
        q_des_leg[last_assigned_frame + 1 :] = q_des_leg[
            last_assigned_frame
        ]

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        ref_id=np.asarray([args.ref_id], dtype=np.int64),
        q_des_leg=q_des_leg.astype(np.float32),
        raw_action_leg=raw_action_leg.astype(np.float32),
        source_step_id=source_step_id,
    )
    report = {
        "schema_version": "pcbc-nominal-action-reference-v1",
        "status": "diagnostic_feedforward_requires_closed_loop_mppi_gate",
        "purpose": "mppi_proposal_centre_not_state_reference_or_training_data",
        "source_episode": str(episode_path),
        "source_episode_sha256": sha256_file(episode_path),
        "reference_config": args.reference_config,
        "ref_id": args.ref_id,
        "frames": frames,
        "lookahead_steps": args.lookahead_steps,
        "physical_offset_leg_rad": physical_offset.tolist(),
        "physical_target_rate_limit_rad_s": (
            args.physical_target_rate_limit_rad_s
        ),
        "physical_target_step_max_rad": float(
            np.max(
                np.abs(
                    np.diff(
                        q_des_leg,
                        axis=0,
                        prepend=q_offset[None],
                    )
                )
            )
        ),
        "source_steps": int(source_action.shape[0]),
        "held_after_frame": int(last_assigned_frame),
        "output": str(output),
        "output_sha256": sha256_file(output),
        "q_des_shape": list(q_des_leg.shape),
        "wheel_action_exact_zero": True,
    }
    write_json(args.report, report)
    print(report)


if __name__ == "__main__":
    main()
