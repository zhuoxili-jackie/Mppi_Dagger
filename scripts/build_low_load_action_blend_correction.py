#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _bootstrap import ROOT, load_contract, write_json

from lateral_mppi_dagger.config import sha256_file
from lateral_mppi_dagger.contract.action16 import ActionContract
from lateral_mppi_dagger.contract.joint_mapping import POLICY_JOINT_ORDER


def _inside_root(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(ROOT)
    except ValueError as exc:
        raise ValueError(
            f"{label} must stay inside project root {ROOT}: {resolved}"
        ) from exc
    return resolved


def _load_raw_action(path: Path, ref_id: int) -> np.ndarray:
    resolved = path.expanduser().resolve()
    with np.load(resolved, allow_pickle=False) as archive:
        raw = np.asarray(archive["raw_action_leg"], dtype=np.float32)
        stored_ref_id = int(np.asarray(archive["ref_id"]).reshape(-1)[0])
    if (
        raw.ndim != 2
        or raw.shape[1] != 12
        or not np.isfinite(raw).all()
    ):
        raise ValueError(
            f"{resolved} must contain finite raw_action_leg [frames,12]."
        )
    if stored_ref_id != ref_id:
        raise ValueError(
            f"{resolved} has ref_id={stored_ref_id}, expected {ref_id}."
        )
    return raw


def _smoothstep(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def build_tail_blend(
    base_raw: np.ndarray,
    support_raw: np.ndarray,
    *,
    scale_leg: np.ndarray,
    steps: int,
    tail_start: int,
    ramp_frames: int,
    joint_mask: np.ndarray | None = None,
    joint_scales: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if steps < 1:
        raise ValueError("--steps must be positive.")
    if not 0 <= tail_start < steps:
        raise ValueError("--tail-start must lie inside [0, steps).")
    if ramp_frames < 1:
        raise ValueError("--ramp-frames must be positive.")
    if min(base_raw.shape[0], support_raw.shape[0]) < steps + 1:
        raise ValueError(
            "Both action references need at least steps+1 frames."
        )
    step_ids = np.arange(steps, dtype=np.float32)
    blend = _smoothstep(
        (step_ids - float(tail_start)) / float(ramp_frames)
    ).astype(np.float32)
    base_step_raw = base_raw[1 : steps + 1]
    support_step_raw = support_raw[1 : steps + 1]
    physical_delta = (
        (support_step_raw - base_step_raw) * scale_leg[None]
    )
    if joint_mask is not None:
        mask = np.asarray(joint_mask, dtype=bool)
        if mask.shape != (12,):
            raise ValueError("joint_mask must have shape [12].")
        physical_delta = np.where(mask[None], physical_delta, 0.0)
    if joint_scales is not None:
        scales = np.asarray(joint_scales, dtype=np.float32)
        if (
            scales.shape != (12,)
            or not np.isfinite(scales).all()
        ):
            raise ValueError(
                "joint_scales must be finite with shape [12]."
            )
        physical_delta = physical_delta * scales[None]
    correction = physical_delta * blend[:, None]
    return correction.astype(np.float32), blend


def _joint_scope_mask(scope: str) -> np.ndarray:
    leg_names = POLICY_JOINT_ORDER[:12]
    scope_parts = scope.split("_", maxsplit=1)
    if len(scope_parts) == 1:
        limb_scope = (
            scope_parts[0]
            if scope_parts[0]
            in {"all", "front", "rear", "fl", "fr", "rl", "rr"}
            else "all"
        )
        joint_scope = (
            "all"
            if scope_parts[0]
            in {"all", "front", "rear", "fl", "fr", "rl", "rr"}
            else scope_parts[0]
        )
    else:
        limb_scope, joint_scope = scope_parts
    if limb_scope not in {
        "all",
        "front",
        "rear",
        "fl",
        "fr",
        "rl",
        "rr",
    }:
        raise ValueError(f"Unknown joint scope: {scope!r}.")
    if joint_scope not in {"all", "hips", "thighs", "calves"}:
        raise ValueError(f"Unknown joint scope: {scope!r}.")

    limb_prefixes = {
        "all": ("FL_", "FR_", "RL_", "RR_"),
        "front": ("FL_", "FR_"),
        "rear": ("RL_", "RR_"),
        "fl": ("FL_",),
        "fr": ("FR_",),
        "rl": ("RL_",),
        "rr": ("RR_",),
    }[limb_scope]
    joint_suffix = {
        "all": None,
        "hips": "_hip_joint",
        "thighs": "_thigh_joint",
        "calves": "_calf_joint",
    }[joint_scope]
    return np.asarray(
        [
            name.startswith(limb_prefixes)
            and (
                joint_suffix is None
                or name.endswith(joint_suffix)
            )
            for name in leg_names
        ],
        dtype=bool,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a provenance-locked physical action correction that "
            "smoothly blends from a moving proposal to a support proposal."
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
    parser.add_argument("--tail-start", type=int, required=True)
    parser.add_argument("--ramp-frames", type=int, default=10)
    parser.add_argument(
        "--joint-scope",
        choices=(
            "all",
            "front",
            "rear",
            "hips",
            "thighs",
            "calves",
            "front_hips",
            "front_thighs",
            "front_calves",
            "rear_hips",
            "rear_thighs",
            "rear_calves",
            "fl_hips",
            "fl_thighs",
            "fl_calves",
            "fr_hips",
            "fr_thighs",
            "fr_calves",
            "rl_hips",
            "rl_thighs",
            "rl_calves",
            "rr_hips",
            "rr_thighs",
            "rr_calves",
        ),
        default="all",
        help=(
            "Apply the support-minus-base correction to all leg joints, "
            "only FL/FR joints, or only RL/RR joints in policy order."
        ),
    )
    parser.add_argument(
        "--joint-scales",
        type=float,
        nargs=12,
        default=(1.0,) * 12,
        metavar=tuple(f"SCALE_{index}" for index in range(12)),
        help=(
            "Per-joint multipliers in frozen policy/type-grouped order, "
            "applied after --joint-scope."
        ),
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
    support_path = args.support_action_reference.expanduser().resolve()
    base_raw = _load_raw_action(base_path, args.ref_id)
    support_raw = _load_raw_action(support_path, args.ref_id)
    contract = ActionContract.from_dict(load_contract())
    scale_leg = np.asarray(contract.scale[:12], dtype=np.float32)
    correction, blend = build_tail_blend(
        base_raw,
        support_raw,
        scale_leg=scale_leg,
        steps=args.steps,
        tail_start=args.tail_start,
        ramp_frames=args.ramp_frames,
        joint_mask=_joint_scope_mask(args.joint_scope),
        joint_scales=np.asarray(args.joint_scales, dtype=np.float32),
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        physical_correction_leg=correction,
        blend=blend,
        ref_id=np.asarray([args.ref_id], dtype=np.int64),
    )
    report = {
        "schema_version": "pcbc-low-load-action-blend-correction-v1",
        "status": "diagnostic_not_training_data",
        "purpose": "smooth_support_proposal_tail_blend",
        "ref_id": args.ref_id,
        "steps": args.steps,
        "tail_start": args.tail_start,
        "ramp_frames": args.ramp_frames,
        "joint_scope": args.joint_scope,
        "selected_joint_names": [
            name
            for name, selected in zip(
                POLICY_JOINT_ORDER[:12],
                _joint_scope_mask(args.joint_scope),
                strict=True,
            )
            if selected
        ],
        "joint_scales_policy_order": {
            name: float(value)
            for name, value in zip(
                POLICY_JOINT_ORDER[:12],
                args.joint_scales,
                strict=True,
            )
        },
        "base_action_reference": str(base_path),
        "base_action_reference_sha256": sha256_file(base_path),
        "support_action_reference": str(support_path),
        "support_action_reference_sha256": sha256_file(support_path),
        "maximum_full_physical_delta_rad": float(
            np.max(
                np.abs(
                    (support_raw[1 : args.steps + 1]
                    - base_raw[1 : args.steps + 1])
                    * scale_leg[None]
                )
            )
        ),
        "maximum_applied_correction_rad": float(
            np.max(np.abs(correction))
        ),
        "first_nonzero_step": (
            int(np.flatnonzero(blend > 0.0)[0])
            if np.any(blend > 0.0)
            else None
        ),
        "first_full_blend_step": (
            int(np.flatnonzero(blend >= 1.0)[0])
            if np.any(blend >= 1.0)
            else None
        ),
        "wheel_action_exact_zero_by_construction": True,
        "output": str(output),
    }
    report["output_sha256"] = sha256_file(output)
    write_json(report_path, report)
    print(report)


if __name__ == "__main__":
    main()
