#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _bootstrap import ROOT, load_contract, write_json
from build_low_load_action_blend_correction import (
    _inside_root,
    _joint_scope_mask,
    _load_raw_action,
    _smoothstep,
)

from lateral_mppi_dagger.config import sha256_file
from lateral_mppi_dagger.contract.action16 import ActionContract
from lateral_mppi_dagger.contract.joint_mapping import POLICY_JOINT_ORDER


def parse_window(value: str, *, steps: int) -> tuple[int, int]:
    parts = value.split(":")
    if len(parts) != 2:
        raise ValueError(
            f"Window {value!r} must use the exclusive START:END form."
        )
    try:
        start, end = (int(part) for part in parts)
    except ValueError as exc:
        raise ValueError(
            f"Window {value!r} must contain integer frame indices."
        ) from exc
    if not 0 <= start < end <= steps:
        raise ValueError(
            f"Window {value!r} must satisfy 0 <= START < END <= {steps}."
        )
    return start, end


def build_window_blend(
    *,
    steps: int,
    windows: tuple[tuple[int, int], ...],
    ramp_in_frames: int,
    ramp_out_frames: int,
) -> np.ndarray:
    if steps < 1:
        raise ValueError("steps must be positive.")
    if not windows:
        raise ValueError("At least one window is required.")
    if ramp_in_frames < 1 or ramp_out_frames < 1:
        raise ValueError("Ramp lengths must be positive.")
    frame = np.arange(steps, dtype=np.float32)
    blend = np.zeros(steps, dtype=np.float32)
    for start, end in windows:
        if not 0 <= start < end <= steps:
            raise ValueError(
                "Every window must satisfy 0 <= start < end <= steps."
            )
        rise = _smoothstep(
            (frame - float(start)) / float(ramp_in_frames)
        )
        fall = _smoothstep(
            (float(end) - frame) / float(ramp_out_frames)
        )
        active = (frame >= start) & (frame < end)
        window_blend = np.where(active, rise * fall, 0.0)
        blend = np.maximum(blend, window_blend.astype(np.float32))
    return blend


def build_window_correction(
    base_raw: np.ndarray,
    support_raw: np.ndarray,
    *,
    scale_leg: np.ndarray,
    blend: np.ndarray,
    joint_mask: np.ndarray,
    joint_scales: np.ndarray,
) -> np.ndarray:
    steps = int(blend.shape[0])
    if min(base_raw.shape[0], support_raw.shape[0]) < steps + 1:
        raise ValueError(
            "Both action references need at least steps+1 frames."
        )
    if scale_leg.shape != (12,):
        raise ValueError("scale_leg must have shape [12].")
    if joint_mask.shape != (12,):
        raise ValueError("joint_mask must have shape [12].")
    if (
        joint_scales.shape != (12,)
        or not np.isfinite(joint_scales).all()
    ):
        raise ValueError("joint_scales must be finite with shape [12].")
    physical_delta = (
        support_raw[1 : steps + 1] - base_raw[1 : steps + 1]
    ) * scale_leg[None]
    physical_delta = np.where(
        joint_mask[None],
        physical_delta * joint_scales[None],
        0.0,
    )
    return (physical_delta * blend[:, None]).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a provenance-locked physical action correction that "
            "blends toward a support proposal only inside explicit frame "
            "windows."
        )
    )
    parser.add_argument("--base-action-reference", type=Path, required=True)
    parser.add_argument(
        "--support-action-reference",
        type=Path,
        required=True,
    )
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
    parser.add_argument("--joint-scope", default="all")
    parser.add_argument(
        "--joint-scales",
        type=float,
        nargs=12,
        default=(1.0,) * 12,
        metavar=tuple(f"SCALE_{index}" for index in range(12)),
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
    base_path = args.base_action_reference.expanduser().resolve()
    support_path = args.support_action_reference.expanduser().resolve()
    base_raw = _load_raw_action(base_path, args.ref_id)
    support_raw = _load_raw_action(support_path, args.ref_id)
    contract = ActionContract.from_dict(load_contract())
    joint_mask = _joint_scope_mask(args.joint_scope)
    joint_scales = np.asarray(args.joint_scales, dtype=np.float32)
    correction = build_window_correction(
        base_raw,
        support_raw,
        scale_leg=np.asarray(contract.scale[:12], dtype=np.float32),
        blend=blend,
        joint_mask=joint_mask,
        joint_scales=joint_scales,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        physical_correction_leg=correction,
        blend=blend,
        ref_id=np.asarray([args.ref_id], dtype=np.int64),
    )
    report = {
        "schema_version": "pcbc-low-load-action-window-correction-v1",
        "status": "diagnostic_not_training_data",
        "purpose": "windowed_support_proposal_blend",
        "ref_id": args.ref_id,
        "steps": args.steps,
        "windows_start_end_exclusive": [
            [start, end] for start, end in windows
        ],
        "ramp_in_frames": args.ramp_in_frames,
        "ramp_out_frames": args.ramp_out_frames,
        "joint_scope": args.joint_scope,
        "selected_joint_names": [
            name
            for name, selected in zip(
                POLICY_JOINT_ORDER[:12],
                joint_mask,
                strict=True,
            )
            if selected
        ],
        "joint_scales_policy_order": dict(
            zip(
                POLICY_JOINT_ORDER[:12],
                joint_scales.tolist(),
                strict=True,
            )
        ),
        "base_action_reference": str(base_path),
        "base_action_reference_sha256": sha256_file(base_path),
        "support_action_reference": str(support_path),
        "support_action_reference_sha256": sha256_file(support_path),
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
