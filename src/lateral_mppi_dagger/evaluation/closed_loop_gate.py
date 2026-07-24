from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from lateral_mppi_dagger.contract.action16 import ActionContract
from lateral_mppi_dagger.data.dataset import load_manifest
from lateral_mppi_dagger.data.schema import ENUMS, read_episode_shard
from lateral_mppi_dagger.reference.loader import ReferenceSet


def _quat_multiply(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
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


def _quat_conjugate(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value).copy()
    result[..., 1:] *= -1.0
    return result


def _quat_rotate(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    zeros = np.zeros(vector.shape[:-1] + (1,), dtype=vector.dtype)
    pure = np.concatenate((zeros, vector), axis=-1)
    return _quat_multiply(
        _quat_multiply(quaternion, pure),
        _quat_conjugate(quaternion),
    )[..., 1:]


@dataclass(frozen=True)
class StudentClosedLoopGateConfig:
    expected_seeds: tuple[int, ...]
    full_episode_steps: int
    success_rate_min: float
    per_reference_success_rate_min: float
    teacher_valid_rate_min: float = 0.99
    shield_intervention_rate_max: float = 0.01
    required_ref_ids: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6)
    required_scenario_resolved_name: str | None = None
    require_reset_reference_identity: bool = False
    nominal_reset_reference_tolerance: float = 1.0e-5
    tracking_thresholds: dict[str, Any] | None = None
    gate_purpose: str = "performance"
    dagger_admission_full_horizon_success_rate_min: float | None = None
    dagger_admission_per_reference_mean_horizon_fraction_min: float | None = None
    dagger_admission_minimum_episode_horizon_fraction: float | None = None
    dagger_admission_signed_progress_ratio_min: float = 0.20

    def validate(self) -> None:
        if not self.expected_seeds or len(set(self.expected_seeds)) != len(
            self.expected_seeds
        ):
            raise ValueError("Student gate expected seeds must be non-empty and unique.")
        if self.full_episode_steps <= 0:
            raise ValueError("Student gate full_episode_steps must be positive.")
        for name in (
            "success_rate_min",
            "per_reference_success_rate_min",
            "teacher_valid_rate_min",
            "shield_intervention_rate_max",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0,1].")
        if self.nominal_reset_reference_tolerance < 0.0:
            raise ValueError(
                "nominal_reset_reference_tolerance must be non-negative."
            )
        if self.gate_purpose not in {"performance", "dagger_admission"}:
            raise ValueError(
                "gate_purpose must be 'performance' or 'dagger_admission'."
            )
        admission_values = (
            self.dagger_admission_full_horizon_success_rate_min,
            self.dagger_admission_per_reference_mean_horizon_fraction_min,
            self.dagger_admission_minimum_episode_horizon_fraction,
        )
        if any(value is not None for value in admission_values):
            if not all(value is not None for value in admission_values):
                raise ValueError(
                    "All DAgger admission thresholds must be provided together."
                )
            if not all(0.0 <= float(value) <= 1.0 for value in admission_values):
                raise ValueError(
                    "DAgger admission thresholds must lie in [0,1]."
                )
        elif self.gate_purpose == "dagger_admission":
            raise ValueError(
                "gate_purpose='dagger_admission' requires all admission thresholds."
            )
        if not 0.0 <= self.dagger_admission_signed_progress_ratio_min <= 1.0:
            raise ValueError(
                "dagger_admission_signed_progress_ratio_min must lie in [0,1]."
            )


def compute_tracking_metrics(
    arrays: dict[str, np.ndarray],
    reference: Any,
    references: ReferenceSet,
    action_contract: ActionContract,
) -> dict[str, float | list[float]]:
    frames = np.minimum(arrays["ref_frame"], reference.frames - 1)
    actual_base_pos = arrays["base_pose_w"][:, :3]
    actual_base_quat = arrays["base_pose_w"][:, 3:7]
    ref_base_pos = reference.body_pos_w[frames, 0]
    ref_base_quat = reference.body_quat_w[frames, 0]
    alignment_quat = _quat_multiply(
        actual_base_quat[0],
        _quat_conjugate(ref_base_quat[0]),
    )
    alignment_batch = np.broadcast_to(alignment_quat, (frames.shape[0], 4))
    target_base_pos = actual_base_pos[0] + _quat_rotate(
        alignment_batch,
        ref_base_pos - ref_base_pos[0],
    )
    target_base_quat = _quat_multiply(alignment_batch, ref_base_quat)
    base_position_error = actual_base_pos - target_base_pos
    orientation_error = 2.0 * np.arccos(
        np.clip(
            np.abs(np.sum(actual_base_quat * target_base_quat, axis=-1)),
            0.0,
            1.0,
        )
    )

    target_linear_velocity = _quat_rotate(
        alignment_batch,
        reference.body_lin_vel_w[frames, 0],
    )
    joint_position_error = arrays["q"] - reference.joint_pos[frames]
    joint_velocity_error = arrays["dq"] - reference.joint_vel[frames]
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
    target_wheel_position = actual_base_pos[0, None, :] + _quat_rotate(
        np.broadcast_to(alignment_quat, (frames.shape[0], 4, 4)),
        reference_wheel_position - ref_base_pos[0, None, :],
    )
    wheel_position_error = np.linalg.norm(
        arrays["wheel_body_pose_w"][..., :3] - target_wheel_position,
        axis=-1,
    )
    contact_mismatch = np.not_equal(
        arrays["desired_contact"],
        arrays["measured_contact"],
    )
    action = arrays.get("scheduled_action16", arrays["executed_action16"])
    previous = np.vstack((np.zeros((1, 16), dtype=np.float32), action[:-1]))
    action_rate = np.abs(action - previous)
    lower = action_contract.raw_min
    upper = action_contract.raw_max
    leg_saturated = np.logical_or(
        np.isclose(action[:, :12], lower[:12], atol=1.0e-6),
        np.isclose(action[:, :12], upper[:12], atol=1.0e-6),
    )
    measured_contact = arrays["measured_contact"].astype(bool)
    wheel_planar_speed = np.linalg.norm(
        arrays["wheel_body_twist_w"][..., :2],
        axis=-1,
    )
    contact_wheel_speed = np.where(measured_contact, wheel_planar_speed, 0.0)
    actual_lateral_displacement = float(
        actual_base_pos[-1, 1] - actual_base_pos[0, 1]
    )
    target_lateral_displacement = float(
        target_base_pos[-1, 1] - target_base_pos[0, 1]
    )
    if abs(target_lateral_displacement) > 1.0e-4:
        signed_progress_ratio = (
            actual_lateral_displacement / target_lateral_displacement
        )
    else:
        signed_progress_ratio = 1.0
    contact_force = np.asarray(arrays["contact_force_w"], dtype=np.float64)
    front_normal_force = np.abs(contact_force[:, :2, 0])
    rear_normal_force = np.abs(contact_force[:, 2:, 2])
    desired_front = arrays["desired_contact"][:, :2].astype(bool)
    front_desired_count = int(np.sum(desired_front))
    front_normal_below_6_fraction = (
        float(np.sum((front_normal_force < 6.0) & desired_front))
        / front_desired_count
        if front_desired_count
        else 0.0
    )
    rear_contact_count = np.sum(measured_contact[:, 2:], axis=-1)
    physical_action = action[:, :12] * action_contract.scale[:12]
    previous_physical_action = np.vstack(
        (
            np.zeros((1, 12), dtype=np.float32),
            physical_action[:-1],
        )
    )
    return {
        "base_position_rmse_m": np.sqrt(
            np.mean(np.square(base_position_error), axis=0)
        ).tolist(),
        "base_position_max_abs_m": np.max(
            np.abs(base_position_error),
            axis=0,
        ).tolist(),
        "base_orientation_rmse_rad": float(
            np.sqrt(np.mean(np.square(orientation_error)))
        ),
        "joint_position_rmse_rad": float(
            np.sqrt(np.mean(np.square(joint_position_error)))
        ),
        "joint_velocity_rmse_rad_s": float(
            np.sqrt(np.mean(np.square(joint_velocity_error)))
        ),
        "lateral_displacement_error_m": float(base_position_error[-1, 1]),
        "actual_lateral_displacement_m": actual_lateral_displacement,
        "target_lateral_displacement_m": target_lateral_displacement,
        "signed_lateral_progress_ratio": float(signed_progress_ratio),
        "actual_lateral_velocity_mean_m_s": float(
            np.mean(arrays["base_twist_w"][:, 1])
        ),
        "target_lateral_velocity_mean_m_s": float(
            np.mean(target_linear_velocity[:, 1])
        ),
        "lateral_velocity_mae_m_s": float(
            np.mean(np.abs(arrays["base_twist_w"][:, 1] - target_linear_velocity[:, 1]))
        ),
        "box_local_x_drift_max_abs_m": float(
            np.max(np.abs(actual_base_pos[:, 0] - actual_base_pos[0, 0]))
        ),
        "wheel_center_rmse_m": float(
            np.sqrt(np.mean(np.square(wheel_position_error)))
        ),
        "wheel_position_rmse_m": np.sqrt(
            np.mean(np.square(wheel_position_error), axis=0)
        ).tolist(),
        "contact_mismatch_rate": float(np.mean(contact_mismatch)),
        "contact_wheel_planar_speed_max_m_s": float(
            np.max(contact_wheel_speed)
        ),
        "leg_action_saturation_rate": float(np.mean(leg_saturated)),
        "action_rate_mean_abs": float(np.mean(action_rate)),
        "action_rate_max_abs": float(np.max(action_rate)),
        "physical_leg_target_step_max_rad": float(
            np.max(
                np.abs(physical_action - previous_physical_action)
            )
        ),
        "front_normal_force_mean_n": np.mean(
            front_normal_force,
            axis=0,
        ).tolist(),
        "front_normal_below_6n_fraction_when_desired": (
            front_normal_below_6_fraction
        ),
        "rear_normal_force_mean_n": np.mean(
            rear_normal_force,
            axis=0,
        ).tolist(),
        "rear_normal_force_p95_n": np.quantile(
            rear_normal_force,
            0.95,
            axis=0,
        ).tolist(),
        "rear_single_support_fraction": float(
            np.mean(rear_contact_count == 1)
        ),
        "rear_no_support_fraction": float(
            np.mean(rear_contact_count == 0)
        ),
    }


def tracking_threshold_failures(
    metrics: dict[str, Any],
    thresholds: dict[str, Any],
) -> list[str]:
    """Apply every configured motion and load threshold fail-closed."""

    failures: list[str] = []
    if np.any(
        np.asarray(metrics["base_position_max_abs_m"])
        > np.asarray(thresholds["base_position_max_abs_m"])
    ):
        failures.append("base_position")
    if (
        float(metrics["base_orientation_rmse_rad"])
        > float(thresholds["base_orientation_rmse_rad"])
    ):
        failures.append("base_orientation")
    if (
        max(metrics["wheel_position_rmse_m"])
        > float(thresholds["wheel_position_rmse_m"])
    ):
        failures.append("wheel_position")
    if (
        float(metrics["contact_mismatch_rate"])
        > float(thresholds["contact_mismatch_rate"])
    ):
        failures.append("contact_mismatch")
    if (
        float(metrics["box_local_x_drift_max_abs_m"])
        > float(thresholds["box_local_x_drift_max_abs_m"])
    ):
        failures.append("box_local_x_drift")
    scalar_upper_bounds = {
        "lateral_velocity_mae_max_m_s": "lateral_velocity_mae_m_s",
        "lateral_displacement_error_max_abs_m": (
            "lateral_displacement_error_m"
        ),
        "rear_single_support_fraction_max": "rear_single_support_fraction",
        "rear_no_support_fraction_max": "rear_no_support_fraction",
        "front_normal_below_6n_fraction_max": (
            "front_normal_below_6n_fraction_when_desired"
        ),
        "physical_leg_target_step_max_rad": (
            "physical_leg_target_step_max_rad"
        ),
    }
    for threshold_name, metric_name in scalar_upper_bounds.items():
        if threshold_name not in thresholds:
            continue
        value = abs(float(metrics[metric_name])) if (
            threshold_name == "lateral_displacement_error_max_abs_m"
        ) else float(metrics[metric_name])
        if value > float(thresholds[threshold_name]):
            failures.append(metric_name)
    if (
        "signed_lateral_progress_ratio_min" in thresholds
        and abs(float(metrics["target_lateral_displacement_m"])) > 1.0e-4
        and float(metrics["signed_lateral_progress_ratio"])
        < float(thresholds["signed_lateral_progress_ratio_min"])
    ):
        failures.append("signed_lateral_progress_ratio")
    if (
        "rear_normal_force_p95_max_n" in thresholds
        and max(metrics["rear_normal_force_p95_n"])
        > float(thresholds["rear_normal_force_p95_max_n"])
    ):
        failures.append("rear_normal_force_p95")
    return failures


def evaluate_student_closed_loop_gate(
    dataset_root: str | Path,
    config: StudentClosedLoopGateConfig,
    references: ReferenceSet,
    action_contract: ActionContract,
) -> dict[str, Any]:
    config.validate()
    root = Path(dataset_root).resolve()
    records = load_manifest(root)
    expected = set(config.expected_seeds)
    by_seed: dict[int, dict[str, Any]] = {}
    duplicate: list[int] = []
    unexpected: list[int] = []
    for record in records:
        seed = int(record["seed"])
        if seed not in expected:
            unexpected.append(seed)
        elif seed in by_seed:
            duplicate.append(seed)
        else:
            by_seed[seed] = record
    missing = sorted(expected - set(by_seed))

    episode_reports: list[dict[str, Any]] = []
    total_steps = 0
    total_teacher_valid = 0
    total_shield = 0
    successes = 0
    full_horizon_successes = 0
    per_ref_successes: Counter[int] = Counter()
    per_ref_full_horizon_successes: Counter[int] = Counter()
    per_ref_totals: Counter[int] = Counter()
    per_ref_horizon_fraction_sum: Counter[int] = Counter()
    per_ref_signed_progress_sum: Counter[int] = Counter()
    per_ref_moving_totals: Counter[int] = Counter()
    horizon_fractions: list[float] = []
    termination_reasons: Counter[int] = Counter()
    student_checkpoint_hashes: set[str] = set()
    structural_failures: list[str] = []
    hard_invariant_failures: list[str] = []
    if missing:
        structural_failures.append(f"missing_seeds:{missing}")
    if duplicate:
        structural_failures.append(f"duplicate_seeds:{sorted(set(duplicate))}")
    if unexpected:
        structural_failures.append(f"unexpected_seeds:{sorted(set(unexpected))}")

    for seed in config.expected_seeds:
        record = by_seed.get(seed)
        if record is None:
            continue
        shard = read_episode_shard(root / record["path"])
        arrays = shard.arrays
        checkpoint_record = shard.metadata.get("student_checkpoint")
        if not isinstance(checkpoint_record, dict) or not checkpoint_record.get("sha256"):
            failures = ["missing_student_checkpoint_provenance"]
        else:
            student_checkpoint_hashes.add(str(checkpoint_record["sha256"]))
            failures = []
        steps = int(arrays["step_id"].shape[0])
        ref_values = np.unique(arrays["ref_id"])
        ref_id = int(ref_values[0]) if ref_values.size == 1 else -1
        if ref_values.size != 1:
            failures.append("reference_id_changes_within_episode")
        scenario_profile = shard.metadata.get("scenario_profile", {})
        if config.required_scenario_resolved_name is not None and (
            not isinstance(scenario_profile, dict)
            or scenario_profile.get("resolved_name")
            != config.required_scenario_resolved_name
        ):
            failures.append("required_scenario_mismatch")
        if config.require_reset_reference_identity:
            reset_state = shard.metadata.get("reset_state", {})
            reset_identity_match = bool(
                isinstance(reset_state, dict)
                and int(reset_state.get("reference_motion_id", -1)) == ref_id
                and int(reset_state.get("reference_frame", -1)) == 0
            )
            if not reset_identity_match:
                failures.append("requested_reference_reset_identity_mismatch")
            if (
                config.required_scenario_resolved_name == "nominal"
                and reset_identity_match
            ):
                reset_values = [
                    np.asarray(
                        reset_state.get("anchor_position_minus_reference_m", []),
                        dtype=np.float64,
                    ),
                    np.asarray(
                        reset_state.get("anchor_twist_minus_reference", []),
                        dtype=np.float64,
                    ),
                    np.asarray(
                        reset_state.get("joint_position_minus_reference", []),
                        dtype=np.float64,
                    ),
                    np.asarray(
                        reset_state.get("joint_velocity_minus_reference", []),
                        dtype=np.float64,
                    ),
                ]
                reset_shapes_match = [value.shape for value in reset_values] == [
                    (3,),
                    (6,),
                    (16,),
                    (16,),
                ]
                reset_error_max = (
                    max(
                        abs(
                            float(
                                reset_state.get(
                                    "anchor_orientation_error_rad",
                                    np.nan,
                                )
                            )
                        ),
                        *(
                            float(np.max(np.abs(value)))
                            for value in reset_values
                        ),
                    )
                    if reset_shapes_match
                    and all(np.isfinite(value).all() for value in reset_values)
                    else float("inf")
                )
                if (
                    not np.isfinite(reset_error_max)
                    or reset_error_max
                    > config.nominal_reset_reference_tolerance
                ):
                    failures.append("nominal_reset_reference_mismatch")
        per_ref_totals[ref_id] += 1
        teacher_valid = arrays["teacher_valid"].astype(bool)
        student_valid = arrays["student_valid"].astype(bool)
        total_steps += steps
        total_teacher_valid += int(np.sum(teacher_valid))
        total_shield += int(np.sum(arrays["shield_intervened"]))
        behavior_student = bool(
            np.all(
                arrays["behavior_policy"]
                == ENUMS["behavior_policy"]["STUDENT"]
            )
        )
        if not behavior_student:
            failures.append("behavior_is_not_all_student")
        if not np.all(student_valid):
            failures.append("student_invalid")
        finite_fields = (
            "obs93_clean",
            "student_action16",
            "executed_action16",
            "base_pose_w",
            "q",
            "dq",
            "wheel_body_pose_w",
        )
        if not all(np.isfinite(arrays[name]).all() for name in finite_fields):
            failures.append("nan_or_inf")
        zero_fields = ["student_action16", "executed_action16"]
        if "scheduled_action16" in arrays:
            zero_fields.append("scheduled_action16")
        if not all(
            np.array_equal(
                arrays[name][:, 12:],
                np.zeros_like(arrays[name][:, 12:]),
            )
            for name in zero_fields
        ):
            failures.append("wheel_action_not_exact_zero")
        full_horizon = bool(
            steps == config.full_episode_steps
            and bool(record.get("success", False))
            and bool(shard.metadata.get("success", False))
            and int(arrays["termination_reason"][-1])
            == ENUMS["termination_reason"]["TIME_LIMIT"]
        )
        if not full_horizon:
            failures.append("did_not_complete_full_horizon")
        horizon_fraction = min(
            float(steps) / float(config.full_episode_steps),
            1.0,
        )
        horizon_fractions.append(horizon_fraction)
        per_ref_horizon_fraction_sum[ref_id] += horizon_fraction
        full_horizon_successes += int(full_horizon)
        per_ref_full_horizon_successes[ref_id] += int(full_horizon)
        metrics = compute_tracking_metrics(
            arrays,
            references[ref_id],
            references,
            action_contract,
        ) if ref_id >= 0 else {}
        if (
            metrics
            and abs(float(metrics["target_lateral_displacement_m"]))
            > 1.0e-4
        ):
            per_ref_signed_progress_sum[ref_id] += float(
                metrics["signed_lateral_progress_ratio"]
            )
            per_ref_moving_totals[ref_id] += 1
        if config.tracking_thresholds is not None and metrics:
            failures.extend(
                f"tracking:{failure}"
                for failure in tracking_threshold_failures(
                    metrics,
                    config.tracking_thresholds,
                )
            )
        episode_success = bool(full_horizon and not failures)
        performance_failures = {
            failure
            for failure in failures
            if failure == "did_not_complete_full_horizon"
            or failure.startswith("tracking:")
        }
        hard_invariant_failures.extend(
            f"seed_{seed}:{failure}"
            for failure in failures
            if failure not in performance_failures
        )
        successes += int(episode_success)
        per_ref_successes[ref_id] += int(episode_success)
        termination_reason = int(arrays["termination_reason"][-1])
        termination_reasons[termination_reason] += 1
        episode_reports.append(
            {
                "episode_id": record["episode_id"],
                "seed": seed,
                "ref_id": ref_id,
                "steps": steps,
                "success": episode_success,
                "teacher_valid_rate": float(np.mean(teacher_valid)),
                "student_valid_rate": float(np.mean(student_valid)),
                "shield_intervention_rate": float(
                    np.mean(arrays["shield_intervened"])
                ),
                "termination_reason": termination_reason,
                "horizon_fraction": horizon_fraction,
                "failures": failures,
                "tracking": metrics,
            }
        )

    overall_success = successes / len(config.expected_seeds)
    per_reference_success_rate = {
        str(ref_id): (
            per_ref_successes[ref_id] / per_ref_totals[ref_id]
            if per_ref_totals[ref_id]
            else 0.0
        )
        for ref_id in config.required_ref_ids
    }
    full_horizon_success_rate = (
        full_horizon_successes / len(config.expected_seeds)
    )
    per_reference_full_horizon_success_rate = {
        str(ref_id): (
            per_ref_full_horizon_successes[ref_id] / per_ref_totals[ref_id]
            if per_ref_totals[ref_id]
            else 0.0
        )
        for ref_id in config.required_ref_ids
    }
    per_reference_mean_horizon_fraction = {
        str(ref_id): (
            per_ref_horizon_fraction_sum[ref_id] / per_ref_totals[ref_id]
            if per_ref_totals[ref_id]
            else 0.0
        )
        for ref_id in config.required_ref_ids
    }
    per_reference_mean_signed_progress_ratio = {
        str(ref_id): (
            per_ref_signed_progress_sum[ref_id]
            / per_ref_moving_totals[ref_id]
            if per_ref_moving_totals[ref_id]
            else None
        )
        for ref_id in config.required_ref_ids
    }
    mean_horizon_fraction = (
        float(np.mean(horizon_fractions)) if horizon_fractions else 0.0
    )
    minimum_horizon_fraction = (
        float(np.min(horizon_fractions)) if horizon_fractions else 0.0
    )
    teacher_valid_rate = total_teacher_valid / total_steps if total_steps else 0.0
    shield_rate = total_shield / total_steps if total_steps else 1.0
    observed_refs = set(per_ref_totals)
    if len(student_checkpoint_hashes) != 1:
        structural_failures.append(
            f"student_checkpoint_hash_count:{len(student_checkpoint_hashes)}"
        )
    performance_checks = {
        "seed_set_exact": not (missing or duplicate or unexpected),
        "required_reference_coverage": set(config.required_ref_ids).issubset(
            observed_refs
        ),
        "overall_success_rate": overall_success >= config.success_rate_min,
        "per_reference_success_rate": all(
            value >= config.per_reference_success_rate_min
            for value in per_reference_success_rate.values()
        ),
        "teacher_valid_rate": teacher_valid_rate >= config.teacher_valid_rate_min,
        "shield_intervention_rate": shield_rate
        <= config.shield_intervention_rate_max,
        # Completion/tracking failures consume the configured overall and
        # per-reference success budgets.  Contract/provenance/finite-value
        # violations remain fail-closed even when the rate budget has room.
        "hard_invariant_failures_empty": not hard_invariant_failures,
        "structural_failures_empty": not structural_failures,
        "single_student_checkpoint": len(student_checkpoint_hashes) == 1,
    }
    admission_requested = (
        config.dagger_admission_full_horizon_success_rate_min is not None
    )
    admission_checks = {
        "seed_set_exact": not (missing or duplicate or unexpected),
        "required_reference_coverage": set(config.required_ref_ids).issubset(
            observed_refs
        ),
        "full_horizon_success_rate": bool(
            admission_requested
            and full_horizon_success_rate
            >= float(
                config.dagger_admission_full_horizon_success_rate_min
            )
        ),
        "per_reference_mean_horizon_fraction": bool(
            admission_requested
            and all(
                value
                >= float(
                    config.dagger_admission_per_reference_mean_horizon_fraction_min
                )
                for value in per_reference_mean_horizon_fraction.values()
            )
        ),
        "minimum_episode_horizon_fraction": bool(
            admission_requested
            and minimum_horizon_fraction
            >= float(
                config.dagger_admission_minimum_episode_horizon_fraction
            )
        ),
        "nontrivial_signed_lateral_progress": bool(
            admission_requested
            and all(
                value is None
                or value
                >= config.dagger_admission_signed_progress_ratio_min
                for value in per_reference_mean_signed_progress_ratio.values()
            )
            and any(
                value is not None
                for value in per_reference_mean_signed_progress_ratio.values()
            )
        ),
        "teacher_valid_rate": teacher_valid_rate
        >= config.teacher_valid_rate_min,
        "shield_intervention_rate": shield_rate
        <= config.shield_intervention_rate_max,
        "hard_invariant_failures_empty": not hard_invariant_failures,
        "structural_failures_empty": not structural_failures,
        "single_student_checkpoint": len(student_checkpoint_hashes) == 1,
    }
    performance_ok = bool(all(performance_checks.values()))
    dagger_admission_ok = bool(
        admission_requested and all(admission_checks.values())
    )
    selected_checks = (
        admission_checks
        if config.gate_purpose == "dagger_admission"
        else performance_checks
    )
    return {
        "schema_version": "pcbc-student-closed-loop-gate-v1",
        "dataset": str(root),
        "gate_purpose": config.gate_purpose,
        "ok": bool(all(selected_checks.values())),
        "performance_ok": performance_ok,
        "dagger_admission_ok": dagger_admission_ok,
        "config": {
            "expected_seeds": list(config.expected_seeds),
            "full_episode_steps": config.full_episode_steps,
            "success_rate_min": config.success_rate_min,
            "per_reference_success_rate_min": config.per_reference_success_rate_min,
            "teacher_valid_rate_min": config.teacher_valid_rate_min,
            "shield_intervention_rate_max": config.shield_intervention_rate_max,
            "required_ref_ids": list(config.required_ref_ids),
            "required_scenario_resolved_name": config.required_scenario_resolved_name,
            "require_reset_reference_identity": config.require_reset_reference_identity,
            "nominal_reset_reference_tolerance": config.nominal_reset_reference_tolerance,
            "tracking_thresholds": config.tracking_thresholds,
            "gate_purpose": config.gate_purpose,
            "dagger_admission_full_horizon_success_rate_min": (
                config.dagger_admission_full_horizon_success_rate_min
            ),
            "dagger_admission_per_reference_mean_horizon_fraction_min": (
                config.dagger_admission_per_reference_mean_horizon_fraction_min
            ),
            "dagger_admission_minimum_episode_horizon_fraction": (
                config.dagger_admission_minimum_episode_horizon_fraction
            ),
            "dagger_admission_signed_progress_ratio_min": (
                config.dagger_admission_signed_progress_ratio_min
            ),
        },
        "summary": {
            "episodes": len(episode_reports),
            "successes": successes,
            "success_rate": overall_success,
            "per_reference_success_rate": per_reference_success_rate,
            "full_horizon_successes": full_horizon_successes,
            "full_horizon_success_rate": full_horizon_success_rate,
            "per_reference_full_horizon_success_rate": (
                per_reference_full_horizon_success_rate
            ),
            "mean_horizon_fraction": mean_horizon_fraction,
            "minimum_horizon_fraction": minimum_horizon_fraction,
            "per_reference_mean_horizon_fraction": (
                per_reference_mean_horizon_fraction
            ),
            "per_reference_mean_signed_progress_ratio": (
                per_reference_mean_signed_progress_ratio
            ),
            "teacher_valid_rate": teacher_valid_rate,
            "shield_intervention_rate": shield_rate,
            "termination_reason_counts": {
                str(key): value for key, value in sorted(termination_reasons.items())
            },
            "student_checkpoint_hashes": sorted(student_checkpoint_hashes),
        },
        "checks": selected_checks,
        "performance_checks": performance_checks,
        "dagger_admission_checks": admission_checks,
        "structural_failures": structural_failures,
        "hard_invariant_failures": hard_invariant_failures,
        "episodes": episode_reports,
    }
