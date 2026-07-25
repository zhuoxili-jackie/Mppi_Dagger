#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _bootstrap import write_json
from build_low_load_action_blend_correction import (
    _inside_root,
    _load_raw_action,
)
from build_low_load_action_window_correction import (
    build_window_blend,
    parse_window,
)

from lateral_mppi_dagger.config import sha256_file
from lateral_mppi_dagger.contract.joint_mapping import POLICY_JOINT_ORDER


def build_window_offset(
    *,
    physical_offset_leg: np.ndarray,
    blend: np.ndarray,
) -> np.ndarray:
    offset = np.asarray(physical_offset_leg, dtype=np.float32)
    weight = np.asarray(blend, dtype=np.float32)
    if offset.shape != (12,) or not np.isfinite(offset).all():
        raise ValueError(
            "physical_offset_leg must be finite with shape [12]."
        )
    if (
        weight.ndim != 1
        or not np.isfinite(weight).all()
        or np.any(weight < 0.0)
        or np.any(weight > 1.0)
    ):
        raise ValueError("blend must be a finite one-dimensional [0,1] array.")
    return (weight[:, None] * offset[None, :]).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a provenance-locked, windowed physical leg-target offset "
            "for bounded low-load diagnostics."
        )
    )
    parser.add_argument("--base-action-reference", type=Path, required=True)
    parser.add_argument("--ref-id", type=int, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument(
        "--windows",
        nargs="+",
        required=True,
        metavar="START:END",
        help="Exclusive frame windows; overlapping windows are unioned.",
    )
    parser.add_argument("--ramp-in-frames", type=int, default=3)
    parser.add_argument("--ramp-out-frames", type=int, default=3)
    parser.add_argument(
        "--physical-offset-leg",
        type=float,
        nargs=12,
        required=True,
        metavar=tuple(f"OFFSET_{index}" for index in range(12)),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    output = _inside_root(args.output, "--output")
    report_path = _inside_root(args.report, "--report")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output}")
    if report_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite report: {report_path}"
        )
    base_path = args.base_action_reference.expanduser().resolve()
    base_raw = _load_raw_action(base_path, args.ref_id)
    if base_raw.shape[0] < args.steps + 1:
        raise ValueError(
            "The base action reference needs at least steps+1 frames."
        )
    windows = tuple(
        parse_window(value, steps=args.steps)
        for value in args.windows
    )
    blend = build_window_blend(
        steps=args.steps,
        windows=windows,
        ramp_in_frames=args.ramp_in_frames,
        ramp_out_frames=args.ramp_out_frames,
    )
    physical_offset = np.asarray(
        args.physical_offset_leg,
        dtype=np.float32,
    )
    if np.max(np.abs(physical_offset)) > 0.25:
        raise ValueError(
            "Every physical leg offset must be bounded to 0.25 rad."
        )
    correction = build_window_offset(
        physical_offset_leg=physical_offset,
        blend=blend,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        schema_version=np.asarray(
            ["pcbc-low-load-action-window-offset-v1"],
            dtype="U48",
        ),
        physical_correction_leg=correction,
        blend=blend,
        ref_id=np.asarray([args.ref_id], dtype=np.int64),
    )
    report = {
        "schema_version": "pcbc-low-load-action-window-offset-report-v1",
        "status": "diagnostic_not_training_data",
        "purpose": "windowed_physical_leg_target_offset",
        "ref_id": args.ref_id,
        "steps": args.steps,
        "windows_start_end_exclusive": [
            [start, end] for start, end in windows
        ],
        "ramp_in_frames": args.ramp_in_frames,
        "ramp_out_frames": args.ramp_out_frames,
        "physical_offset_policy_order_rad": dict(
            zip(
                POLICY_JOINT_ORDER[:12],
                physical_offset.tolist(),
                strict=True,
            )
        ),
        "base_action_reference": str(base_path),
        "base_action_reference_sha256": sha256_file(base_path),
        "maximum_blend": float(np.max(blend)),
        "nonzero_frames": np.flatnonzero(blend).astype(int).tolist(),
        "maximum_applied_correction_rad": float(
            np.max(np.abs(correction))
        ),
        "wheel_action_exact_zero_by_construction": True,
        "output": str(output),
    }
    report["output_sha256"] = sha256_file(output)
    write_json(report_path, report)
    print(report)


if __name__ == "__main__":
    main()
