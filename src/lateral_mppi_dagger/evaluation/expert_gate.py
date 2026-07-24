from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from lateral_mppi_dagger.data.dataset import load_manifest
from lateral_mppi_dagger.data.schema import ENUMS, read_episode_shard
from lateral_mppi_dagger.expert.base import FailureCode, LabelSource
from lateral_mppi_dagger.evaluation.closed_loop_gate import (
    compute_tracking_metrics,
    tracking_threshold_failures,
)


@dataclass(frozen=True)
class ExpertGateConfig:
    expected_seeds: tuple[int, ...]
    required_successes: int
    full_episode_steps: int
    required_teacher_valid_rate: float = 0.99
    required_ref_ids: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6)
    require_zero_shield_interventions: bool = True
    required_scenario_resolved_name: str | None = None
    require_requested_reference_reset: bool = False
    reset_reference_tolerance: float = 1.0e-5
    require_implementation_hashes: bool = False
    expected_implementation_sha256: dict[str, str] | None = None
    tracking_thresholds: dict[str, Any] | None = None

    def validate(self) -> None:
        if not self.expected_seeds:
            raise ValueError("Expert gate requires at least one expected seed.")
        if len(set(self.expected_seeds)) != len(self.expected_seeds):
            raise ValueError("Expert gate seed list contains duplicates.")
        if not 0 < self.required_successes <= len(self.expected_seeds):
            raise ValueError("required_successes must lie in [1, number of expected seeds].")
        if self.full_episode_steps <= 0:
            raise ValueError("full_episode_steps must be positive.")
        if not 0.0 <= self.required_teacher_valid_rate <= 1.0:
            raise ValueError("required_teacher_valid_rate must lie in [0,1].")
        if self.reset_reference_tolerance < 0.0:
            raise ValueError("reset_reference_tolerance must be non-negative.")
        if self.expected_implementation_sha256 is not None and not all(
            isinstance(path, str)
            and path
            and isinstance(digest, str)
            and len(digest) == 64
            for path, digest in self.expected_implementation_sha256.items()
        ):
            raise ValueError("expected_implementation_sha256 is malformed.")


def load_gate_seeds(path: str | Path, count: int) -> tuple[int, ...]:
    values = tuple(
        int(line.strip())
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if len(values) < count:
        raise ValueError(f"Seed file contains {len(values)} entries, but {count} are required.")
    selected = values[:count]
    if len(set(selected)) != len(selected):
        raise ValueError("Selected gate seeds are not unique.")
    return selected


def evaluate_expert_gate(
    dataset_root: str | Path,
    config: ExpertGateConfig,
    references: Any | None = None,
    action_contract: Any | None = None,
) -> dict[str, Any]:
    """Fail-closed evaluation of completed MPPI expert episode shards."""
    config.validate()
    root = Path(dataset_root).resolve()
    records = load_manifest(root)
    expected_seed_set = set(config.expected_seeds)
    records_by_seed: dict[int, dict[str, Any]] = {}
    duplicate_seeds: list[int] = []
    unexpected_seeds: list[int] = []
    for record in records:
        seed = int(record["seed"])
        if seed not in expected_seed_set:
            unexpected_seeds.append(seed)
            continue
        if seed in records_by_seed:
            duplicate_seeds.append(seed)
            continue
        records_by_seed[seed] = record

    missing_seeds = sorted(expected_seed_set - set(records_by_seed))
    structural_failures: list[str] = []
    if duplicate_seeds:
        structural_failures.append(f"duplicate_seeds:{sorted(set(duplicate_seeds))}")
    if unexpected_seeds:
        structural_failures.append(f"unexpected_seeds:{sorted(set(unexpected_seeds))}")
    if missing_seeds:
        structural_failures.append(f"missing_seeds:{missing_seeds}")

    episode_reports: list[dict[str, Any]] = []
    config_hashes: set[str] = set()
    observed_ref_ids: set[int] = set()
    total_valid = 0
    total_steps = 0
    successes = 0
    exact_zero = True
    finite = True
    failure_code_none = True
    label_source_mppi = True
    shield_interventions = 0
    tracking_successes = 0
    requested_reference_resets_match = True
    required_scenarios_match = True
    implementation_hashes_present = True
    implementation_hashes_match_current = True
    if config.tracking_thresholds is not None and (
        references is None or action_contract is None
    ):
        raise ValueError(
            "Tracking thresholds require ReferenceSet and ActionContract inputs."
        )

    for seed in config.expected_seeds:
        record = records_by_seed.get(seed)
        if record is None:
            continue
        episode_failures: list[str] = []
        try:
            shard = read_episode_shard(root / record["path"])
        except BaseException as exc:
            structural_failures.append(f"unreadable_seed_{seed}:{type(exc).__name__}:{exc}")
            exact_zero = False
            finite = False
            failure_code_none = False
            label_source_mppi = False
            continue

        arrays = shard.arrays
        metadata = shard.metadata
        steps = int(arrays["step_id"].shape[0])
        ref_values = np.unique(arrays["ref_id"])
        ref_id = int(ref_values[0]) if ref_values.size == 1 else -1
        if ref_values.size != 1:
            episode_failures.append("reference_id_changes_within_episode")
        observed_ref_ids.add(ref_id)
        if int(record["ref_id"]) != ref_id or int(metadata.get("ref_id", -1)) != ref_id:
            episode_failures.append("reference_id_metadata_mismatch")
        if int(metadata.get("seed", -1)) != seed:
            episode_failures.append("seed_metadata_mismatch")
        if metadata.get("expert_backend") != "mppi":
            episode_failures.append("expert_backend_is_not_mppi")
        if metadata.get("wheel_action_mode") != "hard_zero":
            episode_failures.append("wheel_action_mode_is_not_hard_zero")
        if config.require_implementation_hashes:
            implementation_hashes = metadata.get(
                "expert_implementation_sha256"
            )
            hashes_valid = bool(
                isinstance(implementation_hashes, dict)
                and implementation_hashes
                and all(
                    isinstance(path, str)
                    and path
                    and isinstance(digest, str)
                    and len(digest) == 64
                    for path, digest in implementation_hashes.items()
                )
            )
            implementation_hashes_present &= hashes_valid
            if not hashes_valid:
                episode_failures.append("missing_expert_implementation_hashes")
            if config.expected_implementation_sha256 is not None:
                hashes_match = bool(
                    hashes_valid
                    and implementation_hashes
                    == config.expected_implementation_sha256
                )
                implementation_hashes_match_current &= hashes_match
                if not hashes_match:
                    episode_failures.append(
                        "expert_implementation_hash_mismatch"
                    )
        if config.required_scenario_resolved_name is not None:
            resolved_scenario = (
                metadata.get("scenario_profile", {}).get("resolved_name")
                if isinstance(metadata.get("scenario_profile"), dict)
                else None
            )
            scenario_match = (
                resolved_scenario == config.required_scenario_resolved_name
            )
            required_scenarios_match &= scenario_match
            if not scenario_match:
                episode_failures.append("required_scenario_mismatch")
        if config.require_requested_reference_reset:
            reset_state = metadata.get("reset_state", {})
            reset_vectors = {
                "anchor_position": np.asarray(
                    reset_state.get("anchor_position_minus_reference_m", []),
                    dtype=np.float64,
                ),
                "anchor_twist": np.asarray(
                    reset_state.get("anchor_twist_minus_reference", []),
                    dtype=np.float64,
                ),
                "joint_position": np.asarray(
                    reset_state.get("joint_position_minus_reference", []),
                    dtype=np.float64,
                ),
                "joint_velocity": np.asarray(
                    reset_state.get("joint_velocity_minus_reference", []),
                    dtype=np.float64,
                ),
            }
            reset_evidence_complete = bool(
                int(reset_state.get("reference_motion_id", -1)) == ref_id
                and int(reset_state.get("reference_frame", -1)) == 0
                and reset_vectors["anchor_position"].shape == (3,)
                and reset_vectors["anchor_twist"].shape == (6,)
                and reset_vectors["joint_position"].shape == (16,)
                and reset_vectors["joint_velocity"].shape == (16,)
                and np.isfinite(
                    float(reset_state.get("anchor_orientation_error_rad", np.nan))
                )
                and all(np.isfinite(value).all() for value in reset_vectors.values())
            )
            reset_error_max = (
                max(
                    float(
                        reset_state["anchor_orientation_error_rad"]
                    ),
                    *(
                        float(np.max(np.abs(value)))
                        for value in reset_vectors.values()
                    ),
                )
                if reset_evidence_complete
                else float("inf")
            )
            reset_match = bool(
                reset_evidence_complete
                and reset_error_max <= config.reset_reference_tolerance
            )
            requested_reference_resets_match &= reset_match
            if not reset_match:
                episode_failures.append("requested_reference_reset_mismatch")
        config_hash = str(metadata.get("expert_config_hash", ""))
        if not config_hash:
            episode_failures.append("missing_expert_config_hash")
        else:
            config_hashes.add(config_hash)

        teacher_valid = arrays["teacher_valid"].astype(bool)
        valid_rate = float(np.mean(teacher_valid))
        total_valid += int(np.sum(teacher_valid))
        total_steps += steps
        wheel_zero_episode = bool(
            np.array_equal(
                arrays["executed_action16"][:, 12:],
                np.zeros_like(arrays["executed_action16"][:, 12:]),
            )
            and np.array_equal(
                arrays["teacher_action16"][teacher_valid, 12:],
                np.zeros_like(arrays["teacher_action16"][teacher_valid, 12:]),
            )
        )
        exact_zero &= wheel_zero_episode
        if not wheel_zero_episode:
            episode_failures.append("wheel_action_not_exact_zero")

        required_finite_fields = (
            "obs93_clean",
            "next_obs93_clean",
            "teacher_action16",
            "executed_action16",
            "base_pose_w",
            "q",
            "dq",
            "wheel_body_pose_w",
        )
        finite_episode = all(
            np.isfinite(arrays[name][teacher_valid] if name == "teacher_action16" else arrays[name]).all()
            for name in required_finite_fields
        )
        finite &= bool(finite_episode)
        if not finite_episode:
            episode_failures.append("nan_or_inf")

        failure_none_episode = bool(
            np.all(arrays["failure_code"] == int(FailureCode.NONE))
        )
        failure_code_none &= failure_none_episode
        if not failure_none_episode:
            episode_failures.append("nonzero_failure_code")

        source_episode = bool(
            np.all(arrays["label_source"][teacher_valid] == int(LabelSource.MPPI))
        )
        label_source_mppi &= source_episode
        if not source_episode:
            episode_failures.append("valid_label_source_is_not_mppi")

        episode_shield_interventions = int(np.sum(arrays["shield_intervened"]))
        shield_interventions += episode_shield_interventions
        if config.require_zero_shield_interventions and episode_shield_interventions:
            episode_failures.append("shield_intervention")

        full_horizon = bool(
            steps == config.full_episode_steps
            and bool(metadata.get("success", False))
            and bool(record.get("success", False))
            and int(arrays["termination_reason"][-1])
            == ENUMS["termination_reason"]["TIME_LIMIT"]
        )
        if not full_horizon:
            episode_failures.append("did_not_complete_full_horizon")
        if valid_rate < config.required_teacher_valid_rate:
            episode_failures.append("teacher_valid_rate_below_threshold")
        tracking: dict[str, Any] = {}
        episode_tracking_pass = config.tracking_thresholds is None
        if config.tracking_thresholds is not None and ref_id >= 0:
            episode_tracking_pass = True
            tracking = compute_tracking_metrics(
                arrays,
                references[ref_id],
                references,
                action_contract,
            )
            tracking_failures = tracking_threshold_failures(
                tracking,
                config.tracking_thresholds,
            )
            if tracking_failures:
                episode_failures.extend(
                    f"tracking:{name}" for name in tracking_failures
                )
                episode_tracking_pass = False
        tracking_successes += int(episode_tracking_pass)
        episode_success = bool(full_horizon and not episode_failures)
        successes += int(episode_success)
        episode_reports.append(
            {
                "episode_id": record["episode_id"],
                "seed": seed,
                "ref_id": ref_id,
                "steps": steps,
                "success": episode_success,
                "collector_success": bool(metadata.get("success", False)),
                "teacher_valid_rate": valid_rate,
                "wheel_action_exact_zero": wheel_zero_episode,
                "shield_interventions": episode_shield_interventions,
                "termination_reason": int(arrays["termination_reason"][-1]),
                "failures": episode_failures,
                "tracking": tracking,
            }
        )

    if len(config_hashes) > 1:
        structural_failures.append(f"mixed_expert_config_hashes:{sorted(config_hashes)}")
    missing_ref_ids = sorted(set(config.required_ref_ids) - observed_ref_ids)
    if missing_ref_ids:
        structural_failures.append(f"missing_reference_ids:{missing_ref_ids}")

    aggregate_valid_rate = float(total_valid / total_steps) if total_steps else 0.0
    checks = {
        "expected_episode_count": len(records_by_seed) == len(config.expected_seeds),
        "required_successes": successes >= config.required_successes,
        "teacher_valid_rate": aggregate_valid_rate >= config.required_teacher_valid_rate,
        "finite_required_fields": finite,
        "failure_codes_all_none": failure_code_none,
        "valid_label_sources_all_mppi": label_source_mppi,
        "wheel_action_exact_zero": exact_zero,
        "shield_interventions_zero": (
            shield_interventions == 0 if config.require_zero_shield_interventions else True
        ),
        "required_scenario_matches": required_scenarios_match,
        "requested_reference_resets_match": requested_reference_resets_match,
        "expert_implementation_hashes_present": implementation_hashes_present,
        "expert_implementation_hashes_match_current": (
            implementation_hashes_match_current
        ),
        "single_expert_config_hash": len(config_hashes) == 1,
        "required_reference_coverage": not missing_ref_ids,
        "seed_set_exact": not (missing_seeds or duplicate_seeds or unexpected_seeds),
        "structural_failures_empty": not structural_failures,
        # Tracking failures are episode failures and therefore consume the
        # same explicit failure budget as every other expert-gate failure.
        # Requiring this check to be true only when *all* episodes track would
        # silently turn a configured 48/50 gate into a contradictory 50/50
        # gate.
        "tracking_thresholds": tracking_successes >= config.required_successes,
    }
    return {
        "schema_version": "pcbc-expert-gate-v1",
        "dataset": str(root),
        "ok": bool(all(checks.values())),
        "config": {
            "expected_seeds": list(config.expected_seeds),
            "required_successes": config.required_successes,
            "full_episode_steps": config.full_episode_steps,
            "required_teacher_valid_rate": config.required_teacher_valid_rate,
            "required_ref_ids": list(config.required_ref_ids),
            "require_zero_shield_interventions": config.require_zero_shield_interventions,
            "required_scenario_resolved_name": config.required_scenario_resolved_name,
            "require_requested_reference_reset": config.require_requested_reference_reset,
            "reset_reference_tolerance": config.reset_reference_tolerance,
            "require_implementation_hashes": config.require_implementation_hashes,
            "expected_implementation_sha256": config.expected_implementation_sha256,
            "tracking_thresholds": config.tracking_thresholds,
        },
        "summary": {
            "episodes_found": len(records_by_seed),
            "successes": successes,
            "success_rate": float(successes / len(config.expected_seeds)),
            "tracking_successes": tracking_successes,
            "teacher_valid_rate": aggregate_valid_rate,
            "wheel_action_exact_zero": exact_zero,
            "shield_interventions": shield_interventions,
            "requested_reference_resets_match": requested_reference_resets_match,
            "observed_ref_ids": sorted(observed_ref_ids),
            "expert_config_hashes": sorted(config_hashes),
        },
        "checks": checks,
        "structural_failures": structural_failures,
        "episodes": episode_reports,
    }
