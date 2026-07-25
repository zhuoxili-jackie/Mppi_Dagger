#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from _bootstrap import ROOT, load_contract, write_json

from lateral_mppi_dagger.config import load_yaml
from lateral_mppi_dagger.evaluation.key7_handoff import (
    build_key7_handoff_observation,
)
from lateral_mppi_dagger.evaluation.lateral_stability import (
    contact_force_metrics,
    reference_gait_metrics,
    replay_fixed_state_velocity_ramp,
)
from lateral_mppi_dagger.reference.contact_schedule import (
    infer_contact_schedule,
)
from lateral_mppi_dagger.reference.loader import ReferenceSet


def _onnx_policy(path: Path):
    import onnxruntime as ort

    session = ort.InferenceSession(
        str(path.resolve()),
        providers=["CPUExecutionProvider"],
    )

    def infer(observation: np.ndarray) -> np.ndarray:
        return session.run(
            ["actions"],
            {"obs": np.asarray(observation, dtype=np.float32).reshape(1, 93)},
        )[0][0]

    return infer


def _reference_metrics(config_path: str | Path) -> list[dict]:
    config = load_yaml(config_path)
    root = ROOT / config["reference_directory"]
    references = ReferenceSet.from_config(config_path)
    reports = []
    for item, motion in zip(
        config["references"],
        references.motions,
        strict=False,
    ):
        path = root / item["file"]
        with np.load(path, allow_pickle=False) as data:
            metrics = reference_gait_metrics(
                joint_position=data["joint_pos"],
                joint_velocity=data["joint_vel"],
                body_position_w=data["body_pos_w"],
                body_linear_velocity_w=data["body_lin_vel_w"],
                fps=float(np.asarray(data["fps"]).reshape(-1)[0]),
            )
        reports.append(
            {
                "ref_id": int(item["index"]),
                "file": item["file"],
                "target_vy_m_s": float(item["target_vy"]),
                "desired_contact_fraction": np.mean(
                    infer_contact_schedule(
                        motion,
                        **references.contact_inference_kwargs(),
                    ),
                    axis=0,
                ).tolist(),
                **metrics,
            }
        )
    return reports


def _dataset_metrics(root: Path) -> dict | None:
    manifest = root / "manifest.jsonl"
    if not manifest.is_file():
        return None
    grouped_force: dict[int, list[np.ndarray]] = defaultdict(list)
    grouped_contact: dict[int, list[np.ndarray]] = defaultdict(list)
    grouped_vy: dict[int, list[np.ndarray]] = defaultdict(list)
    command_values = set()
    episodes = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        with np.load(root / record["path"], allow_pickle=False) as data:
            ref_id = int(record["ref_id"])
            grouped_force[ref_id].append(data["contact_force_w"])
            grouped_contact[ref_id].append(data["measured_contact"])
            grouped_vy[ref_id].append(data["base_twist_w"][:, 1])
            command_values.update(
                float(value) for value in np.unique(data["target_vy"])
            )
        episodes += 1
    per_reference = {}
    for ref_id in sorted(grouped_force):
        force = np.concatenate(grouped_force[ref_id], axis=0)
        contact = np.concatenate(grouped_contact[ref_id], axis=0)
        velocity = np.concatenate(grouped_vy[ref_id], axis=0)
        per_reference[str(ref_id)] = {
            **contact_force_metrics(force, contact),
            "base_lateral_velocity_mean_m_s": float(np.mean(velocity)),
            "base_lateral_velocity_mean_abs_m_s": float(
                np.mean(np.abs(velocity))
            ),
        }
    return {
        "path": str(root.resolve()),
        "episodes": episodes,
        "target_velocity_atoms_m_s": sorted(command_values),
        "per_reference": per_reference,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Offline diagnosis of 708 gait clearance/load and exact key7 "
            "velocity-command activation. Never launches the deployment runtime."
        )
    )
    parser.add_argument(
        "--reference-config",
        type=str,
        default="configs/reference_708.yaml",
    )
    parser.add_argument(
        "--candidate-reference-config",
        type=str,
        default="configs/low_load_lateral/train_001/reference.yaml",
    )
    parser.add_argument(
        "--onnx",
        type=Path,
        default=(
            ROOT
            / "exported/lateral_policy_v2_key7_aligned_r1/policy.onnx"
        ),
    )
    parser.add_argument(
        "--expert-dataset",
        type=Path,
        default=ROOT / "datasets/mppi_formal_gate50_v3",
    )
    parser.add_argument(
        "--student-dataset",
        type=Path,
        default=ROOT / "datasets/onnx_gate_r3_conservative_admission",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/28_lateral_stability_diagnosis.json",
    )
    args = parser.parse_args()

    contract = load_contract()
    golden = load_yaml("configs/key7_handoff_golden.yaml")
    handoff = build_key7_handoff_observation(contract, golden)
    policy = _onnx_policy(args.onnx)
    ramps = {
        direction: replay_fixed_state_velocity_ramp(
            handoff.observation,
            policy,
            np.asarray(contract["action"]["scale"], dtype=np.float32),
            target_lateral_velocity_m_s=target,
            acceleration_m_s2=0.60,
            control_dt_s=0.02,
            settle_steps=10,
            command_steps=20,
        )
        for direction, target in (("left_first_key", 0.03), ("right_first_key", -0.03))
    }
    references = _reference_metrics(args.reference_config)
    candidate_references = _reference_metrics(
        args.candidate_reference_config
    )
    expert = _dataset_metrics(args.expert_dataset.resolve())
    student = _dataset_metrics(args.student_dataset.resolve())
    report = {
        "schema_version": "pcbc-lateral-stability-diagnosis-v1",
        "scope": (
            "offline MPPI-repository analysis; deployment source/runtime is "
            "not modified or launched"
        ),
        "control_contract": {
            "operator_command": "A/D select signed lateral base velocity",
            "first_key_target_velocity_m_s": 0.03,
            "command_acceleration_m_s2": 0.60,
            "policy_output": (
                "12 leg joint-position residuals plus four hard-zero wheel "
                "velocity outputs"
            ),
            "leg_execution": "PD position control",
            "wheel_execution": "velocity control",
        },
        "references": references,
        "candidate_references": candidate_references,
        "expert_dataset": expert,
        "historical_student_dataset": student,
        "candidate_fixed_state_activation": ramps,
        "findings": {
            "first_key_command_present_in_training": bool(
                expert is not None
                and any(
                    np.isclose(value, 0.03)
                    for value in expert["target_velocity_atoms_m_s"]
                )
            ),
            "all_references_share_fixed_cadence": bool(
                np.ptp(
                    [
                        item["dominant_leg_frequency_hz"]
                        for item in references
                    ]
                )
                < 0.05
            ),
            "rear_single_support_overload_observed": bool(
                expert is not None
                and max(
                    entry["support_force_mean_n"] or 0.0
                    for metrics in expert["per_reference"].values()
                    for entry in metrics["rear_single_support"].values()
                )
                > 120.0
            ),
            "historical_student_survives_without_lateral_tracking": bool(
                student is not None
                and max(
                    abs(metrics["base_lateral_velocity_mean_m_s"])
                    for metrics in student["per_reference"].values()
                )
                < 0.01
            ),
            "candidate_first_key_feedback_exceeds_0p20_rad": bool(
                max(
                    ramp[
                        "maximum_physical_leg_target_delta_from_default_rad"
                    ]
                    for ramp in ramps.values()
                )
                > 0.20
            ),
            "low_load_candidate_covers_key7_ramp_atoms": bool(
                {
                    -0.024,
                    -0.012,
                    0.012,
                    0.024,
                    -0.03,
                    0.03,
                }.issubset(
                    {
                        float(item["target_vy_m_s"])
                        for item in candidate_references
                    }
                )
            ),
            "low_load_candidate_clearance_within_limits": bool(
                max(
                    max(item["front_trunk_detachment_max_m"])
                    for item in candidate_references
                )
                <= 0.0081
                and max(
                    max(item["rear_ground_clearance_max_m"])
                    for item in candidate_references
                )
                <= 0.0121
            ),
        },
    }
    write_json(args.output, report)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "findings": report["findings"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
