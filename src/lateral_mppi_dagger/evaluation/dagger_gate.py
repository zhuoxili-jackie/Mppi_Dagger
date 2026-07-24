from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from lateral_mppi_dagger.data.dataset import load_manifest
from lateral_mppi_dagger.data.schema import ENUMS, read_episode_shard
from lateral_mppi_dagger.expert.base import LabelSource
from lateral_mppi_dagger.env.scenarios import ScenarioProfile


@dataclass(frozen=True)
class DaggerCollectionGateConfig:
    round_number: int
    expected_scenario: str
    expected_beta: float
    expected_episodes: int
    minimum_student_behavior_episodes: int = 1
    teacher_valid_rate_min: float = 0.99


def evaluate_dagger_collection_gate(
    dataset_root: str | Path,
    config: DaggerCollectionGateConfig,
    scenario_profile: ScenarioProfile,
) -> dict[str, Any]:
    root = Path(dataset_root).resolve()
    selected = [
        record
        for record in load_manifest(root)
        if int(record.get("dagger_round", 0)) == config.round_number
    ]
    episode_reports = []
    total_steps = 0
    teacher_valid_steps = 0
    student_behavior_steps = 0
    student_behavior_episodes = 0
    checkpoint_hashes: set[str] = set()
    all_labels_mppi = True
    all_student_actions_valid = True
    all_scenario_runtime_matches = True
    observation_noise_exercised = (
        scenario_profile.observation_noise_std == 0.0
    )
    initial_state_ranges = scenario_profile.values["initial_state"]
    initial_state_perturbation_requested = any(
        abs(float(value)) > 0.0
        for group in ("pose", "velocity")
        for bounds in initial_state_ranges[group].values()
        for value in bounds
    ) or any(
        abs(float(value)) > 0.0
        for value in initial_state_ranges["joint_position"]
    )
    initial_state_perturbation_exercised = not initial_state_perturbation_requested

    for record in selected:
        shard = read_episode_shard(root / record["path"])
        arrays = shard.arrays
        metadata = shard.metadata
        failures: list[str] = []
        steps = int(arrays["step_id"].shape[0])
        total_steps += steps
        teacher_valid = arrays["teacher_valid"].astype(bool)
        student_valid = arrays["student_valid"].astype(bool)
        teacher_valid_steps += int(np.sum(teacher_valid))
        all_student_actions_valid &= bool(np.all(student_valid))
        labels_mppi = bool(
            np.all(
                arrays["label_source"][teacher_valid]
                == int(LabelSource.MPPI)
            )
        )
        all_labels_mppi &= labels_mppi
        behavior_values = np.unique(arrays["behavior_policy"])
        episode_student = bool(
            behavior_values.size == 1
            and int(behavior_values[0])
            == ENUMS["behavior_policy"]["STUDENT"]
        )
        episode_teacher = bool(
            behavior_values.size == 1
            and int(behavior_values[0])
            == ENUMS["behavior_policy"]["TEACHER"]
        )
        if not (episode_student or episode_teacher):
            failures.append("behavior_selection_not_episode_level")
        if episode_student:
            student_behavior_episodes += 1
            student_behavior_steps += steps
        checkpoint = metadata.get("student_checkpoint")
        if not isinstance(checkpoint, dict) or not checkpoint.get("sha256"):
            failures.append("missing_student_checkpoint")
        else:
            checkpoint_hashes.add(str(checkpoint["sha256"]))
        if metadata.get("scenario") != config.expected_scenario:
            failures.append("scenario_name_mismatch")
        if int(metadata.get("dagger_round", -1)) != config.round_number:
            failures.append("round_metadata_mismatch")
        if not np.isclose(
            float(metadata.get("beta", np.nan)),
            config.expected_beta,
            atol=0.0,
            rtol=0.0,
        ):
            failures.append("beta_mismatch")
        scenario_record = metadata.get("scenario_profile", {})
        runtime_matches = bool(
            scenario_record == scenario_profile.metadata()
            and int(metadata.get("action_delay_steps", -1))
            == scenario_profile.action_delay_steps
            and np.isclose(
                float(metadata.get("observation_noise_std", np.nan)),
                scenario_profile.observation_noise_std,
            )
        )
        all_scenario_runtime_matches &= runtime_matches
        if not runtime_matches:
            failures.append("scenario_runtime_metadata_mismatch")
        if scenario_profile.observation_noise_std > 0.0:
            changed = np.any(arrays["obs93_train"] != arrays["obs93_clean"])
            structural_zeros = np.array_equal(
                arrays["obs93_train"][:, 53:57],
                np.zeros((steps, 4), dtype=np.float32),
            ) and np.array_equal(
                arrays["obs93_train"][:, 85:89],
                np.zeros((steps, 4), dtype=np.float32),
            ) and np.array_equal(
                arrays["obs93_train"][:, 92],
                np.zeros(steps, dtype=np.float32),
            )
            observation_noise_exercised |= bool(changed and structural_zeros)
            if not changed or not structural_zeros:
                failures.append("observation_noise_not_exercised_correctly")
        sampled_jitter = np.asarray(
            metadata.get("platform_position_jitter_m_sampled", []),
            dtype=np.float64,
        )
        maximum_jitter = np.asarray(
            scenario_profile.platform_position_jitter_m,
            dtype=np.float64,
        )
        if sampled_jitter.shape != (3,) or np.any(
            np.abs(sampled_jitter) > maximum_jitter + 1.0e-9
        ):
            failures.append("platform_jitter_out_of_contract")
        reset_state = metadata.get("reset_state", {})
        joint_offset = np.asarray(
            reset_state.get("joint_position_minus_reference", []),
            dtype=np.float64,
        )
        anchor_twist_offset = np.asarray(
            reset_state.get("anchor_twist_minus_reference", []),
            dtype=np.float64,
        )
        if joint_offset.shape != (16,) or anchor_twist_offset.shape != (6,):
            failures.append("missing_reset_state_evidence")
        else:
            initial_state_perturbation_exercised |= bool(
                np.max(np.abs(joint_offset)) > 1.0e-6
                or np.max(np.abs(anchor_twist_offset)) > 1.0e-6
            )
        if not labels_mppi:
            failures.append("valid_teacher_label_is_not_mppi")
        if not np.all(student_valid):
            failures.append("student_action_invalid")
        episode_reports.append(
            {
                "episode_id": record["episode_id"],
                "split": record["split"],
                "steps": steps,
                "student_behavior": episode_student,
                "teacher_valid_rate": float(np.mean(teacher_valid)),
                "student_valid_rate": float(np.mean(student_valid)),
                "success": bool(record["success"]),
                "failures": failures,
            }
        )

    teacher_valid_rate = (
        teacher_valid_steps / total_steps if total_steps else 0.0
    )
    checks = {
        "expected_episode_count": len(selected) == config.expected_episodes,
        "minimum_student_behavior_episodes": student_behavior_episodes
        >= config.minimum_student_behavior_episodes,
        "real_student_visited_states_labeled": student_behavior_steps > 0,
        "teacher_valid_rate": teacher_valid_rate
        >= config.teacher_valid_rate_min,
        "all_valid_labels_mppi": all_labels_mppi,
        "all_student_actions_valid": all_student_actions_valid,
        "single_student_checkpoint": len(checkpoint_hashes) == 1,
        "scenario_runtime_matches": all_scenario_runtime_matches,
        "observation_noise_exercised": observation_noise_exercised,
        "initial_state_perturbation_exercised": initial_state_perturbation_exercised,
        "episode_failures_empty": all(
            not episode["failures"] for episode in episode_reports
        ),
    }
    return {
        "schema_version": "pcbc-dagger-collection-gate-v1",
        "dataset": str(root),
        "ok": bool(all(checks.values())),
        "config": {
            "round_number": config.round_number,
            "expected_scenario": config.expected_scenario,
            "expected_beta": config.expected_beta,
            "expected_episodes": config.expected_episodes,
            "minimum_student_behavior_episodes": config.minimum_student_behavior_episodes,
            "teacher_valid_rate_min": config.teacher_valid_rate_min,
        },
        "summary": {
            "episodes": len(selected),
            "steps": total_steps,
            "student_behavior_episodes": student_behavior_episodes,
            "student_behavior_steps": student_behavior_steps,
            "teacher_valid_rate": teacher_valid_rate,
            "student_checkpoint_hashes": sorted(checkpoint_hashes),
            "scenario_profile": scenario_profile.metadata(),
            "initial_state_perturbation_requested": initial_state_perturbation_requested,
            "initial_state_perturbation_exercised": initial_state_perturbation_exercised,
        },
        "checks": checks,
        "episodes": episode_reports,
    }
