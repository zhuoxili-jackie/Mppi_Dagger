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
    try:
        resolved.relative_to(ROOT)
    except ValueError as exc:
        raise ValueError(
            f"{label} must remain inside the project root {ROOT}: {resolved}"
        ) from exc
    return resolved


def _load_action_stream(
    path: Path,
    *,
    array_key: str,
    env_index: int,
    ref_id: int,
) -> tuple[np.ndarray, dict[str, object]]:
    resolved = path.expanduser().resolve()
    with np.load(resolved, allow_pickle=False) as archive:
        if array_key not in archive:
            raise KeyError(
                f"{resolved} does not contain action array {array_key!r}."
            )
        stored = np.asarray(archive[array_key], dtype=np.float32)
        if stored.ndim == 2:
            if env_index != 0:
                raise ValueError(
                    f"{resolved} has no environment axis; --env-index must "
                    "be zero."
                )
            actions = stored
        elif stored.ndim == 3:
            if not 0 <= env_index < stored.shape[1]:
                raise ValueError(
                    f"--env-index {env_index} is outside {resolved}'s "
                    f"environment axis of length {stored.shape[1]}."
                )
            actions = stored[:, env_index]
        else:
            raise ValueError(
                f"{resolved}:{array_key} must have shape [steps,16] or "
                f"[steps,envs,16], got {stored.shape}."
            )
        if (
            actions.shape[1:] != (16,)
            or not np.isfinite(actions).all()
        ):
            raise ValueError(
                f"{resolved}:{array_key} must resolve to finite "
                f"[steps,16], got {actions.shape}."
            )
        if not np.array_equal(
            actions[:, 12:],
            np.zeros_like(actions[:, 12:]),
        ):
            raise ValueError(
                f"{resolved}:{array_key} contains non-zero wheel actions."
            )

        if "ref_id" not in archive:
            raise KeyError(f"{resolved} does not contain ref_id.")
        source_ref_id = np.asarray(archive["ref_id"], dtype=np.int64)
        if not np.all(source_ref_id == ref_id):
            raise ValueError(
                f"{resolved} contains a ref_id other than {ref_id}."
            )

        if "ref_frame" not in archive:
            raise KeyError(f"{resolved} does not contain ref_frame.")
        source_ref_frame = np.asarray(
            archive["ref_frame"],
            dtype=np.int64,
        )
        if source_ref_frame.ndim == 2:
            if not 0 <= env_index < source_ref_frame.shape[1]:
                raise ValueError(
                    f"--env-index {env_index} is outside {resolved}'s "
                    "ref_frame environment axis."
                )
            source_ref_frame = source_ref_frame[:, env_index]
        else:
            source_ref_frame = source_ref_frame.reshape(-1)
        expected_frames = np.arange(actions.shape[0], dtype=np.int64)
        if not np.array_equal(source_ref_frame, expected_frames):
            raise ValueError(
                f"{resolved} reference frames must be contiguous from zero."
            )

    return actions, {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "array_key": array_key,
        "env_index": env_index,
        "steps": int(actions.shape[0]),
    }


def _rate_project_leg_actions(
    desired_raw_leg: np.ndarray,
    *,
    scale_leg: np.ndarray,
    raw_min_leg: np.ndarray,
    raw_max_leg: np.ndarray,
    maximum_physical_delta: float,
) -> tuple[np.ndarray, np.ndarray]:
    maximum_raw_delta = maximum_physical_delta / np.abs(scale_leg)
    projected = np.empty_like(desired_raw_leg, dtype=np.float32)
    projection_physical = np.empty_like(
        desired_raw_leg,
        dtype=np.float32,
    )
    previous = np.zeros(12, dtype=np.float32)
    for step, desired in enumerate(desired_raw_leg):
        bounded = np.maximum(
            np.minimum(desired, previous + maximum_raw_delta),
            previous - maximum_raw_delta,
        )
        bounded = np.maximum(
            np.minimum(bounded, raw_max_leg),
            raw_min_leg,
        ).astype(np.float32)
        projected[step] = bounded
        projection_physical[step] = np.abs(
            (bounded - desired) * scale_leg
        )
        previous = bounded
    return projected, projection_physical


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a rate-feasible diagnostic action proposal by splicing "
            "two hashed deterministic replay streams. The result is an "
            "MPPI proposal centre, not a state reference or training label."
        )
    )
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--tail", type=Path, required=True)
    parser.add_argument(
        "--prefix-array-key",
        default="executed_action16",
    )
    parser.add_argument(
        "--tail-array-key",
        default="executed_action16",
    )
    parser.add_argument("--prefix-env-index", type=int, default=0)
    parser.add_argument("--tail-env-index", type=int, default=0)
    parser.add_argument(
        "--switch-frame",
        type=int,
        default=None,
        help=(
            "Use the tail stream from this frame through --steps. Mutually "
            "exclusive with repeatable --tail-interval."
        ),
    )
    parser.add_argument(
        "--tail-interval",
        type=int,
        nargs=2,
        action="append",
        metavar=("START", "STOP"),
        default=None,
        help=(
            "Use the tail stream on half-open [START,STOP). May be repeated "
            "for disjoint intervals."
        ),
    )
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument(
        "--reference-config",
        default="configs/low_load_lateral/train_001/reference.yaml",
    )
    parser.add_argument("--ref-id", type=int, default=8)
    parser.add_argument(
        "--physical-target-rate-limit-rad-s",
        type=float,
        default=2.25,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    if args.steps < 1:
        parser.error("--steps must be positive.")
    if (args.switch_frame is None) == (args.tail_interval is None):
        parser.error(
            "Specify exactly one of --switch-frame or --tail-interval."
        )
    if (
        args.switch_frame is not None
        and not 0 < args.switch_frame < args.steps
    ):
        parser.error("--switch-frame must lie strictly inside --steps.")
    tail_intervals: list[tuple[int, int]]
    if args.tail_interval is None:
        tail_intervals = [(args.switch_frame, args.steps)]
    else:
        tail_intervals = [
            (int(start), int(stop))
            for start, stop in args.tail_interval
        ]
        previous_stop = 0
        for start, stop in tail_intervals:
            if not 0 < start < stop <= args.steps:
                parser.error(
                    "Each --tail-interval must satisfy "
                    "0 < START < STOP <= --steps."
                )
            if start < previous_stop:
                parser.error(
                    "--tail-interval values must be ordered and disjoint."
                )
            previous_stop = stop
    if args.prefix_env_index < 0 or args.tail_env_index < 0:
        parser.error("Environment indices must be non-negative.")
    if (
        not np.isfinite(args.physical_target_rate_limit_rad_s)
        or args.physical_target_rate_limit_rad_s <= 0.0
    ):
        parser.error(
            "--physical-target-rate-limit-rad-s must be finite and positive."
        )

    references = ReferenceSet.from_config(args.reference_config)
    if not 0 <= args.ref_id < len(references):
        parser.error("--ref-id is outside the selected reference bank.")
    reference = references[args.ref_id]
    if args.steps >= reference.frames:
        parser.error(
            "--steps must leave a frame for one-step action lookahead."
        )

    prefix, prefix_provenance = _load_action_stream(
        args.prefix,
        array_key=args.prefix_array_key,
        env_index=args.prefix_env_index,
        ref_id=args.ref_id,
    )
    tail, tail_provenance = _load_action_stream(
        args.tail,
        array_key=args.tail_array_key,
        env_index=args.tail_env_index,
        ref_id=args.ref_id,
    )
    if prefix.shape[0] < args.steps or tail.shape[0] < args.steps:
        raise ValueError(
            "Both action streams must contain at least --steps rows."
        )

    desired_action16 = prefix[: args.steps].copy()
    tail_step_mask = np.zeros(args.steps, dtype=bool)
    for start, stop in tail_intervals:
        desired_action16[start:stop] = tail[start:stop]
        tail_step_mask[start:stop] = True
    contract = ActionContract.from_dict(load_contract())
    scale_leg = np.asarray(contract.scale[:12], dtype=np.float32)
    q_offset_leg = np.asarray(
        contract.q_action_offset_runtime[:12],
        dtype=np.float32,
    )
    raw_min_leg = np.asarray(contract.raw_min[:12], dtype=np.float32)
    raw_max_leg = np.asarray(contract.raw_max[:12], dtype=np.float32)
    maximum_physical_delta = (
        args.physical_target_rate_limit_rad_s / reference.fps
    )
    projected_raw_leg, projection_physical = (
        _rate_project_leg_actions(
            desired_action16[:, :12],
            scale_leg=scale_leg,
            raw_min_leg=raw_min_leg,
            raw_max_leg=raw_max_leg,
            maximum_physical_delta=maximum_physical_delta,
        )
    )

    raw_action_leg = np.zeros(
        (reference.frames, 12),
        dtype=np.float32,
    )
    q_des_leg = np.repeat(
        q_offset_leg[None],
        reference.frames,
        axis=0,
    )
    source_stream_id = np.full(
        reference.frames,
        -1,
        dtype=np.int8,
    )
    source_step_id = np.full(
        reference.frames,
        -1,
        dtype=np.int32,
    )
    for step in range(args.steps):
        target_frame = step + 1
        raw_action_leg[target_frame] = projected_raw_leg[step]
        q_des_leg[target_frame] = (
            q_offset_leg + scale_leg * projected_raw_leg[step]
        )
        source_stream_id[target_frame] = int(tail_step_mask[step])
        source_step_id[target_frame] = step
    raw_action_leg[args.steps + 1 :] = projected_raw_leg[-1]
    q_des_leg[args.steps + 1 :] = (
        q_offset_leg + scale_leg * projected_raw_leg[-1]
    )
    source_stream_id[args.steps + 1 :] = int(tail_step_mask[-1])
    source_step_id[args.steps + 1 :] = args.steps - 1

    output = _require_inside_root(args.output, "--output")
    report_path = _require_inside_root(args.report, "--report")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output}")
    if report_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite report: {report_path}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        ref_id=np.asarray([args.ref_id], dtype=np.int64),
        q_des_leg=q_des_leg,
        raw_action_leg=raw_action_leg,
        source_stream_id=source_stream_id,
        source_step_id=source_step_id,
    )

    physical_steps = np.diff(
        q_des_leg,
        axis=0,
        prepend=q_offset_leg[None],
    )
    changed_rows = np.any(projection_physical > 1.0e-7, axis=1)
    desired_boundary_steps: list[dict[str, object]] = []
    for start, stop in tail_intervals:
        entry_step = np.abs(
            (tail[start, :12] - prefix[start - 1, :12])
            * scale_leg
        )
        record: dict[str, object] = {
            "frame": start,
            "direction": "prefix_to_tail",
            "maximum_physical_step_rad": float(np.max(entry_step)),
        }
        desired_boundary_steps.append(record)
        if stop < args.steps:
            exit_step = np.abs(
                (prefix[stop, :12] - tail[stop - 1, :12])
                * scale_leg
            )
            desired_boundary_steps.append(
                {
                    "frame": stop,
                    "direction": "tail_to_prefix",
                    "maximum_physical_step_rad": float(
                        np.max(exit_step)
                    ),
                }
            )
    report = {
        "schema_version": "pcbc-spliced-nominal-action-reference-v1",
        "status": "diagnostic_feedforward_requires_closed_loop_mppi_gate",
        "purpose": "mppi_proposal_centre_not_training_data",
        "reference_config": args.reference_config,
        "ref_id": args.ref_id,
        "fps": reference.fps,
        "frames": reference.frames,
        "steps": args.steps,
        "switch_frame": args.switch_frame,
        "tail_intervals": [
            {"start_frame": start, "stop_frame": stop}
            for start, stop in tail_intervals
        ],
        "prefix": prefix_provenance,
        "tail": tail_provenance,
        "physical_target_rate_limit_rad_s": (
            args.physical_target_rate_limit_rad_s
        ),
        "physical_target_step_limit_rad": maximum_physical_delta,
        "desired_splice_boundary_step_max_rad": float(
            max(
                item["maximum_physical_step_rad"]
                for item in desired_boundary_steps
            )
        ),
        "desired_splice_boundaries": desired_boundary_steps,
        "rate_projection": {
            "changed_step_count": int(np.count_nonzero(changed_rows)),
            "first_changed_step": (
                int(np.flatnonzero(changed_rows)[0])
                if np.any(changed_rows)
                else None
            ),
            "last_changed_step": (
                int(np.flatnonzero(changed_rows)[-1])
                if np.any(changed_rows)
                else None
            ),
            "maximum_correction_rad": float(
                np.max(projection_physical)
            ),
        },
        "physical_target_step_max_rad": float(
            np.max(np.abs(physical_steps))
        ),
        "wheel_action_exact_zero": bool(
            np.array_equal(
                desired_action16[:, 12:],
                np.zeros_like(desired_action16[:, 12:]),
            )
        ),
        "output": str(output),
        "output_sha256": sha256_file(output),
    }
    write_json(report_path, report)
    print(report)


if __name__ == "__main__":
    main()
