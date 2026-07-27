#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _bootstrap import ROOT, load_contract, write_json

from lateral_mppi_dagger.config import sha256_file
from lateral_mppi_dagger.contract.action16 import ActionContract
from lateral_mppi_dagger.reference.contact_schedule import (
    infer_contact_schedule,
)
from lateral_mppi_dagger.reference.loader import ReferenceSet


def _project_raw_sequence(
    proposed_raw: np.ndarray,
    raw_min: np.ndarray,
    raw_max: np.ndarray,
    maximum_raw_delta: np.ndarray,
) -> np.ndarray:
    if proposed_raw.ndim != 2 or proposed_raw.shape[1] != 12:
        raise ValueError("proposed_raw must have shape [frames,12].")
    projected = np.empty_like(proposed_raw, dtype=np.float32)
    previous = np.zeros(12, dtype=np.float32)
    for frame, proposed in enumerate(proposed_raw):
        current = np.maximum(
            np.minimum(proposed, previous + maximum_raw_delta),
            previous - maximum_raw_delta,
        )
        current = np.maximum(
            np.minimum(current, raw_max),
            raw_min,
        ).astype(np.float32)
        projected[frame] = current
        previous = current
    return projected


def _moving_proposal_q_des(
    *,
    standing_action_q_des: np.ndarray,
    standing_reference_q: np.ndarray,
    moving_reference_q: np.ndarray,
    gait_delta_scale_leg: np.ndarray,
    balance_baseline: str,
) -> np.ndarray:
    if balance_baseline == "standing_action":
        baseline = standing_action_q_des
    elif balance_baseline == "standing_reference":
        baseline = standing_reference_q
    else:
        raise ValueError(
            f"Unknown balance baseline {balance_baseline!r}."
        )
    expected_shape = standing_reference_q.shape
    if (
        expected_shape != moving_reference_q.shape
        or expected_shape != standing_action_q_des.shape
        or len(expected_shape) != 2
        or expected_shape[1] != 12
        or gait_delta_scale_leg.shape != (12,)
    ):
        raise ValueError(
            "Proposal inputs must use matching [frames,12] trajectories "
            "and a 12-element scale."
        )
    gait_delta = moving_reference_q - standing_reference_q
    return (
        baseline + gait_delta_scale_leg[None] * gait_delta
    ).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build moving MPPI proposal centres by adding each low-load "
            "reference's joint-space gait delta to a validated standing "
            "balance action trajectory. The result remains a proposal asset, "
            "not a state reference or training dataset."
        )
    )
    parser.add_argument(
        "--reference-config",
        default="configs/low_load_lateral/train_001/reference.yaml",
    )
    parser.add_argument(
        "--standing-action",
        type=Path,
        default=ROOT
        / "assets/low_load_lateral/train_001/nominal_actions/ref_08_standing.npz",
    )
    parser.add_argument("--standing-ref-id", type=int, default=8)
    parser.add_argument(
        "--balance-baseline",
        choices=("standing_action", "standing_reference"),
        default="standing_action",
        help=(
            "Use the validated standing balance action (default) or the "
            "frozen standing reference itself as the gait-delta baseline. "
            "The latter is intended for deterministic reachability "
            "diagnostics."
        ),
    )
    parser.add_argument(
        "--moving-ref-ids",
        type=int,
        nargs="+",
        default=tuple(range(8)),
    )
    parser.add_argument("--gait-delta-scale", type=float, default=1.0)
    parser.add_argument(
        "--gait-delta-scale-leg",
        type=float,
        nargs=12,
        default=None,
        help=(
            "Optional per-joint physical gait-delta scales in frozen "
            "type-grouped leg order. This overrides --gait-delta-scale."
        ),
    )
    parser.add_argument(
        "--gait-delta-frame-shift",
        type=int,
        default=0,
        help=(
            "Integer frame shift applied to the gait delta. Positive values "
            "advance the action; indices are edge-clamped, never wrapped."
        ),
    )
    parser.add_argument(
        "--gait-delta-frame-shift-mode",
        choices=(
            "always",
            "both_front_desired",
            "any_front_swing",
            "fl_desired",
            "fr_desired",
        ),
        default="always",
        help=(
            "Schedule mask selecting when the shifted gait delta is used; "
            "outside the mask the unshifted delta is retained."
        ),
    )
    parser.add_argument(
        "--physical-target-rate-limit-rad-s",
        type=float,
        default=2.25,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT
        / "reports/low_load_lateral/train_001/diagnostics/moving_action_references.json",
    )
    args = parser.parse_args()

    if args.gait_delta_scale_leg is None:
        if (
            not np.isfinite(args.gait_delta_scale)
            or args.gait_delta_scale <= 0.0
        ):
            parser.error("--gait-delta-scale must be finite and positive.")
        gait_delta_scale_leg = np.full(
            12,
            args.gait_delta_scale,
            dtype=np.float32,
        )
    else:
        gait_delta_scale_leg = np.asarray(
            args.gait_delta_scale_leg,
            dtype=np.float32,
        )
        if (
            gait_delta_scale_leg.shape != (12,)
            or not np.isfinite(gait_delta_scale_leg).all()
            or np.any(gait_delta_scale_leg < 0.0)
            or not np.any(gait_delta_scale_leg > 0.0)
        ):
            parser.error(
                "--gait-delta-scale-leg must contain 12 finite "
                "non-negative values and at least one positive value."
            )
    if args.physical_target_rate_limit_rad_s <= 0.0:
        parser.error(
            "--physical-target-rate-limit-rad-s must be positive."
        )
    if len(set(args.moving_ref_ids)) != len(args.moving_ref_ids):
        parser.error("--moving-ref-ids must not contain duplicates.")

    references = ReferenceSet.from_config(args.reference_config)
    if not 0 <= args.standing_ref_id < len(references):
        parser.error("--standing-ref-id is outside the reference bank.")
    for ref_id in args.moving_ref_ids:
        if not 0 <= ref_id < len(references):
            parser.error(
                f"Moving ref ID {ref_id} is outside the reference bank."
            )
        if ref_id == args.standing_ref_id:
            parser.error(
                "--moving-ref-ids must not include --standing-ref-id."
            )

    standing_action_path = args.standing_action.expanduser().resolve()
    with np.load(standing_action_path, allow_pickle=False) as archive:
        if "q_des_leg" not in archive.files:
            raise KeyError(
                f"{standing_action_path} has no 'q_des_leg' array."
            )
        standing_q_des = np.asarray(
            archive["q_des_leg"],
            dtype=np.float32,
        )
        standing_raw_action = (
            np.asarray(
                archive["raw_action_leg"],
                dtype=np.float32,
            )
            if "raw_action_leg" in archive.files
            else None
        )
        if "ref_id" in archive.files:
            source_ref_id = int(
                np.asarray(archive["ref_id"]).reshape(-1)[0]
            )
            if source_ref_id != args.standing_ref_id:
                raise ValueError(
                    "Standing action ref_id does not match "
                    "--standing-ref-id."
                )

    standing_reference = references[args.standing_ref_id]
    expected_shape = (standing_reference.frames, 12)
    if standing_q_des.shape != expected_shape:
        raise ValueError(
            "Standing q_des shape mismatch: expected "
            f"{expected_shape}, got {standing_q_des.shape}."
        )
    if not np.isfinite(standing_q_des).all():
        raise ValueError("Standing q_des contains NaN or Inf.")

    contract = ActionContract.from_dict(load_contract())
    scale = np.asarray(contract.scale[:12], dtype=np.float32)
    q_offset = np.asarray(
        contract.q_action_offset_runtime[:12],
        dtype=np.float32,
    )
    raw_min = np.asarray(contract.raw_min[:12], dtype=np.float32)
    raw_max = np.asarray(contract.raw_max[:12], dtype=np.float32)
    if standing_raw_action is None:
        standing_raw_action = (
            (standing_q_des - q_offset) / scale
        ).astype(np.float32)
    if (
        standing_raw_action.shape != expected_shape
        or not np.isfinite(standing_raw_action).all()
    ):
        raise ValueError(
            "Standing raw_action_leg must be finite with shape "
            f"{expected_shape}."
        )
    maximum_physical_delta = (
        args.physical_target_rate_limit_rad_s
        / standing_reference.fps
    )
    maximum_raw_delta = maximum_physical_delta / scale
    standing_joint_reference = np.asarray(
        standing_reference.joint_pos[:, :12],
        dtype=np.float32,
    )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, object]] = []
    for ref_id in args.moving_ref_ids:
        reference = references[ref_id]
        if reference.frames != standing_reference.frames:
            raise ValueError(
                f"Reference {ref_id} frame count does not match standing."
            )
        gait_delta = (
            np.asarray(reference.joint_pos[:, :12], dtype=np.float32)
            - standing_joint_reference
        )
        inferred_contact = infer_contact_schedule(
            reference,
            **references.contact_inference_kwargs(),
        )
        # The expert consumes proposal frame ref_frame + 1.  Store the
        # schedule mask one frame later so proposal frame i is conditioned on
        # the desired contact at public reference frame i - 1.
        proposal_schedule_frames = np.clip(
            np.arange(reference.frames) - 1,
            0,
            reference.frames - 1,
        )
        desired_front = inferred_contact[
            proposal_schedule_frames,
            :2,
        ]
        if args.gait_delta_frame_shift_mode == "always":
            shift_mask = np.ones(reference.frames, dtype=bool)
        elif args.gait_delta_frame_shift_mode == "both_front_desired":
            shift_mask = np.all(desired_front, axis=1)
        elif args.gait_delta_frame_shift_mode == "any_front_swing":
            shift_mask = ~np.all(desired_front, axis=1)
        elif args.gait_delta_frame_shift_mode == "fl_desired":
            shift_mask = desired_front[:, 0]
        elif args.gait_delta_frame_shift_mode == "fr_desired":
            shift_mask = desired_front[:, 1]
        else:
            raise AssertionError(
                "Unhandled gait-delta frame-shift mode "
                f"{args.gait_delta_frame_shift_mode!r}."
            )
        proposed_q_des = _moving_proposal_q_des(
            standing_action_q_des=standing_q_des,
            standing_reference_q=standing_joint_reference,
            moving_reference_q=np.asarray(
                reference.joint_pos[:, :12],
                dtype=np.float32,
            ),
            gait_delta_scale_leg=gait_delta_scale_leg,
            balance_baseline=args.balance_baseline,
        )
        proposed_raw = (proposed_q_des - q_offset) / scale
        unshifted_projected_raw = _project_raw_sequence(
            proposed_raw,
            raw_min,
            raw_max,
            maximum_raw_delta,
        )
        unshifted_action_delta = (
            unshifted_projected_raw - standing_raw_action
        )
        shifted_action_delta = unshifted_action_delta[
            np.clip(
                np.arange(reference.frames)
                + args.gait_delta_frame_shift,
                0,
                reference.frames - 1,
            )
        ]
        scheduled_action_delta = np.where(
            shift_mask[:, None],
            shifted_action_delta,
            unshifted_action_delta,
        )
        scheduled_raw = standing_raw_action + scheduled_action_delta
        projected_raw = _project_raw_sequence(
            scheduled_raw,
            raw_min,
            raw_max,
            maximum_raw_delta,
        )
        projected_q_des = (
            q_offset + scale * projected_raw
        ).astype(np.float32)
        output = output_dir / f"low_load_ref{ref_id}_proposal.npz"
        np.savez_compressed(
            output,
            ref_id=np.asarray([ref_id], dtype=np.int64),
            q_des_leg=projected_q_des,
            raw_action_leg=projected_raw,
            standing_q_des_leg=standing_q_des,
            gait_delta_leg=gait_delta.astype(np.float32),
        )
        entries.append(
            {
                "ref_id": ref_id,
                "target_vy": float(reference.target_vy),
                "output": str(output),
                "output_sha256": sha256_file(output),
                "frames": reference.frames,
                "gait_delta_max_abs_rad": float(
                    np.max(np.abs(gait_delta))
                ),
                "gait_delta_frame_shift": int(
                    args.gait_delta_frame_shift
                ),
                "gait_delta_frame_shift_mode": (
                    args.gait_delta_frame_shift_mode
                ),
                "gait_delta_shift_mask_fraction": float(
                    np.mean(shift_mask)
                ),
                "physical_target_step_max_rad": float(
                    np.max(
                        np.abs(
                            np.diff(
                                projected_q_des,
                                axis=0,
                                prepend=q_offset[None],
                            )
                        )
                    )
                ),
                "raw_bound_margin_min": float(
                    np.minimum(
                        projected_raw - raw_min,
                        raw_max - projected_raw,
                    ).min()
                ),
            }
        )

    report = {
        "schema_version": (
            "pcbc-low-load-moving-action-reference-build-v1"
        ),
        "status": (
            "diagnostic_feedforward_requires_closed_loop_mppi_gate"
        ),
        "purpose": (
            "mppi_proposal_centres_not_state_references_or_training_data"
        ),
        "reference_config": args.reference_config,
        "standing_ref_id": args.standing_ref_id,
        "balance_baseline": args.balance_baseline,
        "standing_action": str(standing_action_path),
        "standing_action_sha256": sha256_file(standing_action_path),
        "gait_delta_scale": args.gait_delta_scale,
        "gait_delta_scale_leg": gait_delta_scale_leg.tolist(),
        "gait_delta_frame_shift": int(args.gait_delta_frame_shift),
        "gait_delta_frame_shift_mode": (
            args.gait_delta_frame_shift_mode
        ),
        "physical_target_rate_limit_rad_s": (
            args.physical_target_rate_limit_rad_s
        ),
        "output_dir": str(output_dir),
        "entries": entries,
        "wheel_action_exact_zero": True,
    }
    write_json(args.report, report)
    print(report)


if __name__ == "__main__":
    main()
