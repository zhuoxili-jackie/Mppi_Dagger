#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _bootstrap import ROOT, load_contract, write_json

from lateral_mppi_dagger.config import sha256_file
from lateral_mppi_dagger.contract.action16 import ActionContract
from lateral_mppi_dagger.reference.loader import ReferenceSet


def _require_inside_root(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if resolved == ROOT or ROOT not in resolved.parents:
        raise ValueError(f"{label} must remain inside project root {ROOT}.")
    return resolved


def _periodic_source_frames(
    *,
    frames: int,
    cycle_start_frame: int,
    period_frames: int,
) -> np.ndarray:
    if frames < 1:
        raise ValueError("frames must be positive.")
    if cycle_start_frame < 1:
        raise ValueError(
            "cycle_start_frame must preserve the fixed frame-zero handoff."
        )
    if period_frames < 2:
        raise ValueError("period_frames must be at least two.")
    repeat_start = cycle_start_frame + period_frames
    if repeat_start >= frames:
        raise ValueError(
            "The first complete source cycle must fit inside the asset."
        )
    source = np.arange(frames, dtype=np.int32)
    tail = np.arange(repeat_start, frames, dtype=np.int32)
    source[repeat_start:] = (
        cycle_start_frame
        + (tail - cycle_start_frame) % period_frames
    )
    return source


def _rate_project_raw_actions(
    desired: np.ndarray,
    *,
    scale: np.ndarray,
    raw_min: np.ndarray,
    raw_max: np.ndarray,
    maximum_physical_delta: float,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(desired, dtype=np.float32)
    scale = np.asarray(scale, dtype=np.float32)
    raw_min = np.asarray(raw_min, dtype=np.float32)
    raw_max = np.asarray(raw_max, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 12:
        raise ValueError("desired must have shape [frames,12].")
    if any(item.shape != (12,) for item in (scale, raw_min, raw_max)):
        raise ValueError("Action contract arrays must have shape [12].")
    if (
        not np.isfinite(values).all()
        or not np.isfinite(scale).all()
        or not np.isfinite(raw_min).all()
        or not np.isfinite(raw_max).all()
        or np.any(scale <= 0.0)
    ):
        raise ValueError("Action arrays must be finite with positive scale.")
    if (
        not np.isfinite(maximum_physical_delta)
        or maximum_physical_delta <= 0.0
    ):
        raise ValueError("maximum_physical_delta must be finite and positive.")

    maximum_raw_delta = maximum_physical_delta / scale
    projected = np.empty_like(values)
    correction = np.empty_like(values)
    previous = np.zeros(12, dtype=np.float32)
    for frame, target in enumerate(values):
        bounded = np.maximum(
            np.minimum(target, previous + maximum_raw_delta),
            previous - maximum_raw_delta,
        )
        bounded = np.maximum(
            np.minimum(bounded, raw_max),
            raw_min,
        ).astype(np.float32)
        projected[frame] = bounded
        correction[frame] = np.abs((bounded - target) * scale)
        previous = bounded
    return projected, correction


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a rate-feasible periodic MPPI proposal centre from a "
            "hashed raw-action asset. This never changes the state reference."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--array-key", default="raw_action_leg")
    parser.add_argument(
        "--reference-config",
        default="configs/low_load_lateral/train_001/reference.yaml",
    )
    parser.add_argument("--ref-id", type=int, required=True)
    parser.add_argument("--cycle-start-frame", type=int, required=True)
    parser.add_argument("--period-frames", type=int, required=True)
    parser.add_argument(
        "--reference-periodicity-tolerance",
        type=float,
        default=1.0e-6,
    )
    parser.add_argument(
        "--physical-target-rate-limit-rad-s",
        type=float,
        default=2.25,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    input_path = _require_inside_root(args.input, "--input")
    output_path = _require_inside_root(args.output, "--output")
    report_path = _require_inside_root(args.report, "--report")
    if not input_path.is_file():
        parser.error(f"--input does not exist: {input_path}")
    if output_path.exists() or report_path.exists():
        parser.error("Refusing to overwrite an existing output or report.")
    if (
        not np.isfinite(args.reference_periodicity_tolerance)
        or args.reference_periodicity_tolerance < 0.0
    ):
        parser.error(
            "--reference-periodicity-tolerance must be finite and non-negative."
        )

    references = ReferenceSet.from_config(args.reference_config)
    if not 0 <= args.ref_id < len(references):
        parser.error("--ref-id is outside the selected reference bank.")
    reference = references[args.ref_id]
    frames = int(reference.frames)
    source_frames = _periodic_source_frames(
        frames=frames,
        cycle_start_frame=args.cycle_start_frame,
        period_frames=args.period_frames,
    )
    repeat_start = args.cycle_start_frame + args.period_frames

    with np.load(input_path, allow_pickle=False) as archive:
        if args.array_key not in archive.files:
            parser.error(
                f"{input_path} does not contain {args.array_key!r}."
            )
        source_action = np.asarray(
            archive[args.array_key],
            dtype=np.float32,
        )
        if "ref_id" in archive.files:
            stored_ref_id = int(
                np.asarray(archive["ref_id"]).reshape(-1)[0]
            )
            if stored_ref_id != args.ref_id:
                parser.error(
                    "Input ref_id does not match the requested reference."
                )
    if source_action.shape != (frames, 12):
        parser.error(
            "Input raw action must match the full reference shape "
            f"{(frames, 12)}, got {source_action.shape}."
        )
    if not np.isfinite(source_action).all():
        parser.error("Input raw action contains NaN or Inf.")

    period = args.period_frames
    periodic_stop = frames - period
    if repeat_start >= periodic_stop:
        parser.error(
            "Reference does not contain a repeated cycle after repeat_start."
        )
    reference_joint_period_error = float(
        np.max(
            np.abs(
                reference.joint_pos[repeat_start:periodic_stop]
                - reference.joint_pos[
                    repeat_start + period : periodic_stop + period
                ]
            )
        )
    )
    reference_relative_body_position = (
        reference.body_pos_w - reference.body_pos_w[:, :1]
    )
    reference_body_period_error = float(
        np.max(
            np.abs(
                reference_relative_body_position[
                    repeat_start:periodic_stop
                ]
                - reference_relative_body_position[
                    repeat_start + period : periodic_stop + period
                ]
            )
        )
    )
    periodicity_error = max(
        reference_joint_period_error,
        reference_body_period_error,
    )
    if periodicity_error > args.reference_periodicity_tolerance:
        parser.error(
            "Frozen reference is not periodic at the requested period: "
            f"{periodicity_error} > "
            f"{args.reference_periodicity_tolerance}."
        )

    desired_action = source_action[source_frames]
    contract = ActionContract.from_dict(load_contract())
    scale = np.asarray(contract.scale[:12], dtype=np.float32)
    raw_min = np.asarray(contract.raw_min[:12], dtype=np.float32)
    raw_max = np.asarray(contract.raw_max[:12], dtype=np.float32)
    offset = np.asarray(
        contract.q_action_offset_runtime[:12],
        dtype=np.float32,
    )
    maximum_physical_delta = (
        args.physical_target_rate_limit_rad_s / reference.fps
    )
    projected_action, projection_correction = _rate_project_raw_actions(
        desired_action,
        scale=scale,
        raw_min=raw_min,
        raw_max=raw_max,
        maximum_physical_delta=maximum_physical_delta,
    )
    q_des = offset + scale * projected_action
    physical_steps = np.diff(
        q_des,
        axis=0,
        prepend=offset[None],
    )
    changed = np.any(projection_correction > 1.0e-7, axis=1)
    source_cycle_index = np.full(frames, -1, dtype=np.int16)
    source_cycle_index[repeat_start:] = (
        np.arange(repeat_start, frames) - repeat_start
    ) // period

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        ref_id=np.asarray([args.ref_id], dtype=np.int64),
        q_des_leg=q_des.astype(np.float32),
        raw_action_leg=projected_action.astype(np.float32),
        source_frame_id=source_frames,
        source_cycle_index=source_cycle_index,
    )
    report = {
        "schema_version": "pcbc-periodic-action-reference-v1",
        "status": "diagnostic_feedforward_requires_closed_loop_mppi_gate",
        "purpose": "mppi_proposal_centre_not_state_reference_or_training_data",
        "input": str(input_path),
        "input_sha256": sha256_file(input_path),
        "input_array_key": args.array_key,
        "reference_config": args.reference_config,
        "ref_id": args.ref_id,
        "frames": frames,
        "cycle_start_frame": args.cycle_start_frame,
        "period_frames": period,
        "repeat_start_frame": repeat_start,
        "reference_periodicity_tolerance": (
            args.reference_periodicity_tolerance
        ),
        "reference_joint_periodicity_max_abs_rad": (
            reference_joint_period_error
        ),
        "reference_relative_body_periodicity_max_abs_m": (
            reference_body_period_error
        ),
        "physical_target_rate_limit_rad_s": (
            args.physical_target_rate_limit_rad_s
        ),
        "physical_target_step_limit_rad": maximum_physical_delta,
        "physical_target_step_max_rad": float(
            np.max(np.abs(physical_steps))
        ),
        "rate_projection": {
            "changed_frame_count": int(np.count_nonzero(changed)),
            "first_changed_frame": (
                int(np.flatnonzero(changed)[0])
                if np.any(changed)
                else None
            ),
            "last_changed_frame": (
                int(np.flatnonzero(changed)[-1])
                if np.any(changed)
                else None
            ),
            "maximum_correction_rad": float(
                np.max(projection_correction)
            ),
        },
        "wheel_action_exact_zero": True,
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
    }
    write_json(report_path, report)
    print(report)


if __name__ == "__main__":
    main()
