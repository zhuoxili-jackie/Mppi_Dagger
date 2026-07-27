#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _bootstrap import ROOT, write_json

from lateral_mppi_dagger.config import sha256_file


FRONT_JOINTS = ((0, 4, 8), (1, 5, 9))


def couple_front_deficit(
    deficit: np.ndarray,
    mode: str,
) -> np.ndarray:
    """Optionally couple the two front-wheel deficits.

    Coupling is useful when a physical correction must be applied as a
    left/right pair because the local contact response is not separable.
    """

    value = np.asarray(deficit, dtype=np.float32)
    if value.ndim != 2 or value.shape[1] != 2:
        raise ValueError("Front deficit must have shape [steps,2].")
    if mode == "independent":
        return value.copy()
    if mode == "mean":
        shared = np.mean(value, axis=1, keepdims=True)
    elif mode == "max":
        shared = np.max(value, axis=1, keepdims=True)
    else:
        raise ValueError(
            "Deficit coupling must be independent, mean, or max."
        )
    return np.repeat(shared, 2, axis=1).astype(np.float32)


def _inside_root(path: Path) -> bool:
    resolved = path.expanduser().resolve()
    return resolved != ROOT and ROOT in resolved.parents


def load_support_trace(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray]:
    """Load either a diagnostic replay trace or a collected episode shard."""

    with np.load(path, allow_pickle=False) as archive:
        desired_contact = np.asarray(
            archive["desired_contact"][:, :2],
            dtype=bool,
        )
        ref_frame = np.asarray(archive["ref_frame"], dtype=np.int32)
        ref_ids = np.asarray(archive["ref_id"], dtype=np.int32).reshape(-1)
        executed_action = np.asarray(
            archive["executed_action16"],
            dtype=np.float32,
        )
        if "front_normal_force_n" in archive:
            force = np.asarray(
                archive["front_normal_force_n"],
                dtype=np.float32,
            )
        elif "contact_force_w" in archive:
            contact_force = np.asarray(
                archive["contact_force_w"],
                dtype=np.float32,
            )
            if (
                contact_force.ndim != 3
                or contact_force.shape[1:] != (4, 3)
            ):
                raise ValueError(
                    "Episode contact_force_w must have shape [steps,4,3]."
                )
            force = np.abs(contact_force[:, None, :2, 0])
            if executed_action.ndim == 2:
                executed_action = executed_action[:, None, :]
        else:
            raise ValueError(
                "Trace must contain front_normal_force_n or contact_force_w."
            )

    if not ref_ids.size or np.any(ref_ids != ref_ids[0]):
        raise ValueError("Trace must contain exactly one consistent ref_id.")
    return (
        force,
        desired_contact,
        ref_frame,
        int(ref_ids[0]),
        executed_action,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a bounded phase-aligned front-support ILC correction from "
            "an exact Isaac diagnostic trace. This is diagnostic only."
        )
    )
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--target-force-n", type=float, default=6.0)
    parser.add_argument("--lead-steps", type=int, default=4)
    parser.add_argument("--smoothing-radius", type=int, default=2)
    parser.add_argument(
        "--deficit-coupling",
        choices=("independent", "mean", "max"),
        default="independent",
        help=(
            "Use each front-wheel deficit independently, or apply the mean/"
            "maximum deficit symmetrically to both front legs."
        ),
    )
    parser.add_argument(
        "--physical-gain-leg",
        type=float,
        nargs=12,
        required=True,
        metavar=tuple(f"J{index}" for index in range(12)),
        help=(
            "Per-joint physical correction in radians at full force deficit. "
            "Only front-leg indices 0/1, 4/5, and 8/9 may be non-zero."
        ),
    )
    args = parser.parse_args()

    trace_path = args.trace.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    report_path = args.report.expanduser().resolve()
    if not _inside_root(output_path) or not _inside_root(report_path):
        parser.error("Output and report must be inside the standalone root.")
    if output_path.exists():
        parser.error(f"Refusing to overwrite existing output {output_path}.")
    if args.target_force_n <= 0.0 or not np.isfinite(args.target_force_n):
        parser.error("--target-force-n must be positive and finite.")
    if args.lead_steps < 0 or args.smoothing_radius < 0:
        parser.error("--lead-steps and --smoothing-radius must be non-negative.")

    gain = np.asarray(args.physical_gain_leg, dtype=np.float32)
    allowed = np.zeros(12, dtype=bool)
    allowed[[0, 1, 4, 5, 8, 9]] = True
    if (
        gain.shape != (12,)
        or not np.isfinite(gain).all()
        or np.any(gain[~allowed] != 0.0)
        or np.max(np.abs(gain)) > 0.25
    ):
        parser.error(
            "--physical-gain-leg must be finite, zero outside the six front "
            "joints, and bounded to 0.25 rad."
        )

    (
        force,
        desired_contact,
        ref_frame,
        ref_id,
        executed_action,
    ) = load_support_trace(trace_path)
    if force.ndim != 3 or force.shape[2] != 2:
        raise ValueError(
            "Trace front_normal_force_n must have shape [steps,candidates,2]."
        )
    if force.shape[1] != 1:
        raise ValueError(
            "Support ILC builder requires a single-candidate trace."
        )
    force = force[:, 0]
    steps = force.shape[0]
    if (
        desired_contact.shape != (steps, 2)
        or ref_frame.shape != (steps,)
        or executed_action.shape != (steps, 1, 16)
    ):
        raise ValueError("Trace arrays are not aligned to one candidate.")
    if not np.array_equal(
        executed_action[:, 0, 12:],
        np.zeros_like(executed_action[:, 0, 12:]),
    ):
        raise ValueError("Trace wheel actions are not exact zero.")

    deficit = np.clip(
        (args.target_force_n - force) / args.target_force_n,
        0.0,
        1.0,
    )
    deficit *= desired_contact.astype(np.float32)
    future_index = np.minimum(
        np.arange(steps) + args.lead_steps,
        steps - 1,
    )
    anticipated = deficit[future_index]
    if args.smoothing_radius:
        radius = args.smoothing_radius
        padded = np.pad(
            anticipated,
            ((radius, radius), (0, 0)),
            mode="edge",
        )
        kernel = np.full(
            2 * radius + 1,
            1.0 / (2 * radius + 1),
            dtype=np.float32,
        )
        anticipated = np.stack(
            [
                np.convolve(
                    padded[:, wheel],
                    kernel,
                    mode="valid",
                )
                for wheel in range(2)
            ],
            axis=-1,
        ).astype(np.float32)
    anticipated = couple_front_deficit(
        anticipated,
        args.deficit_coupling,
    )

    physical_correction = np.zeros((steps, 12), dtype=np.float32)
    for wheel, joint_indices in enumerate(FRONT_JOINTS):
        physical_correction[:, list(joint_indices)] = (
            anticipated[:, wheel : wheel + 1]
            * gain[np.asarray(joint_indices)][None, :]
        )
    if not np.isfinite(physical_correction).all():
        raise ValueError("Computed correction contains NaN or Inf.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        schema_version=np.asarray(
            ["pcbc-low-load-front-support-ilc-v1"],
            dtype="U48",
        ),
        ref_id=np.asarray([ref_id], dtype=np.int32),
        ref_frame=ref_frame,
        target_force_n=np.asarray([args.target_force_n], dtype=np.float32),
        lead_steps=np.asarray([args.lead_steps], dtype=np.int32),
        smoothing_radius=np.asarray(
            [args.smoothing_radius],
            dtype=np.int32,
        ),
        deficit_coupling=np.asarray(
            [args.deficit_coupling],
            dtype="U16",
        ),
        physical_gain_leg=gain,
        measured_front_normal_force_n=force,
        desired_front_contact=desired_contact,
        anticipated_normalized_deficit=anticipated,
        physical_correction_leg=physical_correction,
    )
    report = {
        "schema_version": "pcbc-low-load-front-support-ilc-report-v1",
        "status": "diagnostic_not_training_data",
        "trace": str(trace_path),
        "trace_sha256": sha256_file(trace_path),
        "ref_id": ref_id,
        "steps": steps,
        "target_force_n": args.target_force_n,
        "lead_steps": args.lead_steps,
        "smoothing_radius": args.smoothing_radius,
        "deficit_coupling": args.deficit_coupling,
        "physical_gain_leg": gain.tolist(),
        "source_below_target_count": np.sum(
            (force < args.target_force_n) & desired_contact,
            axis=0,
        ).tolist(),
        "source_desired_count": np.sum(
            desired_contact,
            axis=0,
        ).tolist(),
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
