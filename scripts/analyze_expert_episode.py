#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from _bootstrap import ROOT, load_contract, write_json

from lateral_mppi_dagger.contract.action16 import ActionContract
from lateral_mppi_dagger.data.dataset import load_manifest
from lateral_mppi_dagger.data.schema import read_episode_shard
from lateral_mppi_dagger.evaluation.closed_loop_gate import (
    compute_tracking_metrics,
)
from lateral_mppi_dagger.reference.loader import ReferenceSet


WHEEL_NAMES = ("FL", "FR", "RL", "RR")


def quat_multiply(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.moveaxis(lhs, -1, 0)
    rw, rx, ry, rz = np.moveaxis(rhs, -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )


def quat_conjugate(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value).copy()
    result[..., 1:] *= -1.0
    return result


def quat_rotate(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    zeros = np.zeros(vector.shape[:-1] + (1,), dtype=vector.dtype)
    pure = np.concatenate((zeros, vector), axis=-1)
    return quat_multiply(
        quat_multiply(quaternion, pure),
        quat_conjugate(quaternion),
    )[..., 1:]


def quat_rotation_vector(
    actual_quat_w: np.ndarray,
    target_quat_w: np.ndarray,
) -> np.ndarray:
    """Return the target-frame shortest rotation vector for wxyz quaternions."""

    actual = np.asarray(actual_quat_w, dtype=np.float64)
    target = np.asarray(target_quat_w, dtype=np.float64)
    epsilon = np.finfo(np.float64).eps
    actual /= np.maximum(
        np.linalg.norm(actual, axis=-1, keepdims=True),
        epsilon,
    )
    target /= np.maximum(
        np.linalg.norm(target, axis=-1, keepdims=True),
        epsilon,
    )
    relative = quat_multiply(quat_conjugate(target), actual)
    relative = np.where(relative[..., :1] < 0.0, -relative, relative)
    vector = relative[..., 1:]
    vector_norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(
        vector_norm,
        np.maximum(relative[..., :1], epsilon),
    )
    return np.where(
        vector_norm > epsilon,
        vector * angle / np.maximum(vector_norm, epsilon),
        2.0 * vector,
    )


def _rmse(value: np.ndarray, axis: int | tuple[int, ...] = 0) -> np.ndarray:
    return np.sqrt(np.mean(np.square(value), axis=axis))


def _safe_correlation(lhs: np.ndarray, rhs: np.ndarray) -> float | None:
    left = np.asarray(lhs, dtype=np.float64).reshape(-1)
    right = np.asarray(rhs, dtype=np.float64).reshape(-1)
    if (
        left.size != right.size
        or left.size < 2
        or np.std(left) <= 1.0e-12
        or np.std(right) <= 1.0e-12
    ):
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _distribution_summary(value: np.ndarray) -> dict[str, float]:
    flattened = np.asarray(value, dtype=np.float64).reshape(-1)
    if flattened.size == 0 or not np.isfinite(flattened).all():
        raise ValueError(
            "Diagnostic distribution must contain finite values."
        )
    return {
        "mean": float(np.mean(flattened)),
        "median": float(np.median(flattened)),
        "p10": float(np.quantile(flattened, 0.10)),
        "p95": float(np.quantile(flattened, 0.95)),
        "minimum": float(np.min(flattened)),
        "maximum": float(np.max(flattened)),
    }


def _contiguous_true_intervals(mask: np.ndarray) -> list[tuple[int, int]]:
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 1:
        raise ValueError("Interval mask must be one-dimensional.")
    padded = np.concatenate(
        (
            np.asarray([False]),
            values,
            np.asarray([False]),
        )
    )
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    stops = np.flatnonzero(padded[:-1] & ~padded[1:])
    return [
        (int(start), int(stop))
        for start, stop in zip(starts, stops, strict=True)
    ]


def _mppi_diagnostics(
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> dict[str, Any] | None:
    if "mppi_cost_components" not in arrays:
        return None
    components = np.asarray(
        arrays["mppi_cost_components"],
        dtype=np.float64,
    )
    order = metadata.get("mppi_cost_component_order")
    if not isinstance(order, list) or len(order) != components.shape[1]:
        raise ValueError(
            "Stored MPPI component order is absent or inconsistent."
        )
    effective_sample_size = np.asarray(
        arrays["mppi_effective_sample_size"],
        dtype=np.float64,
    )
    return {
        "cost_component_order": list(order),
        "cost_components": {
            str(name): _distribution_summary(components[:, index])
            for index, name in enumerate(order)
        },
        "minimum_total_cost": _distribution_summary(
            arrays["mppi_minimum_total_cost"]
        ),
        "mean_total_cost": _distribution_summary(
            arrays["mppi_mean_total_cost"]
        ),
        "effective_sample_size": {
            **_distribution_summary(effective_sample_size),
            "fraction_below_2": float(
                np.mean(effective_sample_size < 2.0)
            ),
            "fraction_below_4": float(
                np.mean(effective_sample_size < 4.0)
            ),
        },
        "rollout_termination_rate": _distribution_summary(
            arrays["mppi_rollout_termination_rate"]
        ),
    }


def _rear_swing_records(
    *,
    arrays: dict[str, np.ndarray],
    frames: np.ndarray,
    reference_joint_position: np.ndarray,
    target_wheel_position: np.ndarray,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    rear_legs = (
        (2, "RL", (2, 6, 10)),
        (3, "RR", (3, 7, 11)),
    )
    for wheel_index, wheel_name, joint_indices in rear_legs:
        swing_mask = ~np.asarray(
            arrays["desired_contact"][:, wheel_index],
            dtype=bool,
        )
        for start, stop in _contiguous_true_intervals(swing_mask):
            if stop - start < 2:
                continue
            last = stop - 1
            actual_delta = (
                arrays["wheel_body_pose_w"][last, wheel_index, :3]
                - arrays["wheel_body_pose_w"][start, wheel_index, :3]
            )
            target_delta = (
                target_wheel_position[last, wheel_index]
                - target_wheel_position[start, wheel_index]
            )
            target_lateral_delta = float(target_delta[1])
            actual_lateral_delta = float(actual_delta[1])
            records.append(
                {
                    "wheel": wheel_name,
                    "start_frame": start,
                    "stop_frame": stop,
                    "frames": stop - start,
                    "target_wheel_displacement_m": target_delta.tolist(),
                    "actual_wheel_displacement_m": actual_delta.tolist(),
                    "lateral_displacement_completion_ratio": (
                        actual_lateral_delta / target_lateral_delta
                        if abs(target_lateral_delta) > 1.0e-12
                        else None
                    ),
                    "measured_contact_fraction": float(
                        np.mean(
                            arrays["measured_contact"][
                                start:stop,
                                wheel_index,
                            ]
                        )
                    ),
                    "joint_indices_policy_order": list(joint_indices),
                    "reference_joint_first_rad": (
                        reference_joint_position[
                            frames[start],
                            list(joint_indices),
                        ].tolist()
                    ),
                    "reference_joint_last_rad": (
                        reference_joint_position[
                            frames[last],
                            list(joint_indices),
                        ].tolist()
                    ),
                    "actual_joint_first_rad": (
                        arrays["q"][start, list(joint_indices)].tolist()
                    ),
                    "actual_joint_last_rad": (
                        arrays["q"][last, list(joint_indices)].tolist()
                    ),
                    "teacher_q_des_first_rad": (
                        arrays["teacher_q_des_leg"][
                            start,
                            list(joint_indices),
                        ].tolist()
                    ),
                    "teacher_q_des_last_rad": (
                        arrays["teacher_q_des_leg"][
                            last,
                            list(joint_indices),
                        ].tolist()
                    ),
                }
            )
    return records


def _window_record(
    *,
    start: int,
    stop: int,
    base_position_error: np.ndarray,
    rotation_vector: np.ndarray,
    wheel_error_vector: np.ndarray,
    desired_contact: np.ndarray,
    measured_contact: np.ndarray,
    contact_force: np.ndarray,
    lateral_velocity_error: np.ndarray,
) -> dict[str, Any]:
    window = slice(start, stop)
    wheel_error_norm = np.linalg.norm(
        wheel_error_vector[window],
        axis=-1,
    )
    mismatch = np.not_equal(
        desired_contact[window],
        measured_contact[window],
    )
    front_normal = np.abs(contact_force[window, :2, 0])
    rear_normal = np.abs(contact_force[window, 2:, 2])
    desired_front = desired_contact[window, :2].astype(bool)
    desired_count = np.sum(desired_front, axis=0)
    below_count = np.sum(
        (front_normal < 6.0) & desired_front,
        axis=0,
    )
    below_fraction = np.divide(
        below_count,
        np.maximum(desired_count, 1),
    )
    return {
        "start_frame": start,
        "stop_frame": stop,
        "frames": stop - start,
        "base_position_rmse_m": _rmse(
            base_position_error[window],
        ).tolist(),
        "base_position_max_abs_m": np.max(
            np.abs(base_position_error[window]),
            axis=0,
        ).tolist(),
        "orientation_rotation_vector_rmse_rad": _rmse(
            rotation_vector[window],
        ).tolist(),
        "orientation_rotation_vector_mean_rad": np.mean(
            rotation_vector[window],
            axis=0,
        ).tolist(),
        "base_orientation_rmse_rad": float(
            _rmse(
                np.linalg.norm(rotation_vector[window], axis=-1),
            )
        ),
        "wheel_position_rmse_m": _rmse(
            wheel_error_norm,
        ).tolist(),
        "wheel_position_vector_rmse_m": _rmse(
            wheel_error_vector[window],
        ).tolist(),
        "contact_mismatch_rate": float(np.mean(mismatch)),
        "contact_mismatch_rate_by_wheel": np.mean(
            mismatch,
            axis=0,
        ).tolist(),
        "lateral_velocity_mae_m_s": float(
            np.mean(np.abs(lateral_velocity_error[window]))
        ),
        "lateral_velocity_mean_error_m_s": float(
            np.mean(lateral_velocity_error[window])
        ),
        "front_normal_force_mean_n": np.mean(
            front_normal,
            axis=0,
        ).tolist(),
        "front_normal_force_p10_n": np.quantile(
            front_normal,
            0.10,
            axis=0,
        ).tolist(),
        "front_normal_below_6n_count_when_desired": (
            below_count.tolist()
        ),
        "front_normal_desired_count": desired_count.tolist(),
        "front_normal_below_6n_fraction_when_desired_by_wheel": (
            below_fraction.tolist()
        ),
        "front_normal_below_6n_fraction_when_desired": float(
            np.sum(below_count) / max(int(np.sum(desired_count)), 1)
        ),
        "rear_normal_force_mean_n": np.mean(
            rear_normal,
            axis=0,
        ).tolist(),
        "rear_normal_force_p95_n": np.quantile(
            rear_normal,
            0.95,
            axis=0,
        ).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Write a structured, offline attribution report for one expert "
            "episode. This never changes a gate or a training dataset."
        )
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument(
        "--reference-config",
        default="configs/low_load_lateral/train_001/reference.yaml",
    )
    parser.add_argument(
        "--window-boundaries",
        type=int,
        nargs="*",
        default=(40, 100, 200),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dataset = args.dataset.expanduser().resolve()
    records = load_manifest(dataset)
    if len(records) != 1:
        parser.error(
            f"Expected exactly one episode in {dataset}, found {len(records)}."
        )
    record = records[0]
    shard_path = dataset / record["path"]
    shard = read_episode_shard(shard_path)
    arrays = shard.arrays
    steps = int(arrays["ref_frame"].shape[0])
    boundaries = [0, *args.window_boundaries, steps]
    if (
        any(boundary <= 0 or boundary >= steps for boundary in boundaries[1:-1])
        or boundaries != sorted(set(boundaries))
    ):
        parser.error(
            "--window-boundaries must be unique, increasing, and inside "
            "the episode."
        )

    references = ReferenceSet.from_config(args.reference_config)
    ref_id = int(record["ref_id"])
    reference = references[ref_id]
    frames = np.minimum(arrays["ref_frame"], reference.frames - 1)
    actual_base_pos = arrays["base_pose_w"][:, :3]
    actual_base_quat = arrays["base_pose_w"][:, 3:7]
    ref_base_pos = reference.body_pos_w[frames, 0]
    ref_base_quat = reference.body_quat_w[frames, 0]
    alignment_quat = quat_multiply(
        actual_base_quat[0],
        quat_conjugate(ref_base_quat[0]),
    )
    alignment_batch = np.broadcast_to(
        alignment_quat,
        (steps, 4),
    )
    target_base_pos = actual_base_pos[0] + quat_rotate(
        alignment_batch,
        ref_base_pos - ref_base_pos[0],
    )
    target_base_quat = quat_multiply(
        alignment_batch,
        ref_base_quat,
    )
    base_position_error = actual_base_pos - target_base_pos
    rotation_vector = quat_rotation_vector(
        actual_base_quat,
        target_base_quat,
    )

    wheel_ids = [
        references.body_order.index(name)
        for name in (
            "FL_foot_link",
            "FR_foot_link",
            "RL_foot_link",
            "RR_foot_link",
        )
    ]
    reference_wheel_position = reference.body_pos_w[frames][:, wheel_ids]
    target_wheel_position = actual_base_pos[0, None, :] + quat_rotate(
        np.broadcast_to(alignment_quat, (steps, 4, 4)),
        reference_wheel_position - ref_base_pos[0, None, :],
    )
    wheel_error_vector = (
        arrays["wheel_body_pose_w"][..., :3] - target_wheel_position
    )
    target_linear_velocity = quat_rotate(
        alignment_batch,
        reference.body_lin_vel_w[frames, 0],
    )
    lateral_velocity_error = (
        arrays["base_twist_w"][:, 1] - target_linear_velocity[:, 1]
    )

    orientation_angle = np.linalg.norm(rotation_vector, axis=-1)
    formal_orientation_angle = 2.0 * np.arccos(
        np.clip(
            np.abs(
                np.sum(actual_base_quat * target_base_quat, axis=-1)
            ),
            0.0,
            1.0,
        )
    )
    contact_force = np.asarray(
        arrays["contact_force_w"],
        dtype=np.float64,
    )
    correlations: dict[str, Any] = {}
    for axis, axis_name in enumerate(("x", "y", "z")):
        correlations[axis_name] = {
            "base_position_error": [
                _safe_correlation(
                    rotation_vector[:, axis],
                    base_position_error[:, component],
                )
                for component in range(3)
            ],
            "wheel_position_error": {
                wheel_name: [
                    _safe_correlation(
                        rotation_vector[:, axis],
                        wheel_error_vector[:, wheel, component],
                    )
                    for component in range(3)
                ]
                for wheel, wheel_name in enumerate(WHEEL_NAMES)
            },
            "front_normal_force": [
                _safe_correlation(
                    rotation_vector[:, axis],
                    np.abs(contact_force[:, wheel, 0]),
                )
                for wheel in range(2)
            ],
            "rear_normal_force": [
                _safe_correlation(
                    rotation_vector[:, axis],
                    np.abs(contact_force[:, wheel, 2]),
                )
                for wheel in range(2, 4)
            ],
        }

    action = arrays.get(
        "scheduled_action16",
        arrays["executed_action16"],
    )
    physical_action = action[:, :12] * np.asarray(
        ActionContract.from_dict(load_contract()).scale[:12],
        dtype=np.float32,
    )
    report = {
        "schema_version": "pcbc-expert-episode-attribution-v2",
        "status": "diagnostic_not_gate_or_training_data",
        "dataset": str(dataset),
        "episode_id": record["episode_id"],
        "episode_shard": str(shard_path),
        "ref_id": ref_id,
        "seed": int(record["seed"]),
        "steps": steps,
        "reference_config": args.reference_config,
        "formal_tracking_metrics": compute_tracking_metrics(
            arrays,
            reference,
            references,
            ActionContract.from_dict(load_contract()),
        ),
        "orientation_rotation_vector_rmse_rad": _rmse(
            rotation_vector,
        ).tolist(),
        "orientation_rotation_vector_mean_rad": np.mean(
            rotation_vector,
            axis=0,
        ).tolist(),
        "orientation_angle_consistency_max_abs_rad": float(
            np.max(np.abs(orientation_angle - formal_orientation_angle))
        ),
        "window_records": [
            _window_record(
                start=start,
                stop=stop,
                base_position_error=base_position_error,
                rotation_vector=rotation_vector,
                wheel_error_vector=wheel_error_vector,
                desired_contact=arrays["desired_contact"],
                measured_contact=arrays["measured_contact"],
                contact_force=contact_force,
                lateral_velocity_error=lateral_velocity_error,
            )
            for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True)
        ],
        "orientation_axis_correlations": correlations,
        "mppi_diagnostics": _mppi_diagnostics(
            arrays,
            shard.metadata,
        ),
        "rear_swing_records": _rear_swing_records(
            arrays=arrays,
            frames=frames,
            reference_joint_position=reference.joint_pos,
            target_wheel_position=target_wheel_position,
        ),
        "physical_leg_target_step_max_rad": float(
            np.max(
                np.abs(
                    np.diff(
                        physical_action,
                        axis=0,
                        prepend=np.zeros_like(physical_action[:1]),
                    )
                )
            )
        ),
        "wheel_action_exact_zero": bool(
            np.array_equal(
                action[:, 12:],
                np.zeros_like(action[:, 12:]),
            )
        ),
    }
    output = args.output.expanduser().resolve()
    try:
        output.relative_to(ROOT)
    except ValueError:
        parser.error(f"--output must remain inside project root {ROOT}.")
    if output.exists():
        parser.error(f"Refusing to overwrite existing report {output}.")
    write_json(output, report)
    print(report)


if __name__ == "__main__":
    main()
