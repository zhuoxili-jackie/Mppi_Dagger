from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lateral_mppi_dagger.config import load_yaml
from lateral_mppi_dagger.contract.action16 import (
    Action16Adapter,
    ActionContract,
    SafetyShield,
)
from lateral_mppi_dagger.data.collector import CollectorConfig, collect_episode
from lateral_mppi_dagger.data.schema import SCHEMA_VERSION, write_episode_shard
from lateral_mppi_dagger.env.replay_env import ReplayContractEnv
from lateral_mppi_dagger.evaluation.closed_loop_gate import (
    StudentClosedLoopGateConfig,
    evaluate_student_closed_loop_gate,
)
from lateral_mppi_dagger.expert.reference_wbc import ReferenceWBCExpert
from lateral_mppi_dagger.reference.loader import ReferenceSet


def _write_student_episode(root: Path, seed: int, ref_id: int) -> None:
    contract_values = load_yaml("configs/deployment_contract.yaml")
    references = ReferenceSet.from_config()
    adapter = Action16Adapter(ActionContract.from_dict(contract_values))
    shard = collect_episode(
        ReplayContractEnv(references, contract_values, adapter),
        ReferenceWBCExpert(adapter),
        SafetyShield(adapter.contract),
        CollectorConfig(
            seed=seed,
            ref_id=ref_id,
            max_steps=8,
            beta=0.0,
            scenario="student_gate_test",
        ),
        {
            "schema_version": SCHEMA_VERSION,
            "wheel_action_mode": "hard_zero",
            "student_checkpoint": {
                "path": "/diagnostic/student.pt",
                "sha256": "test-student-checkpoint",
            },
        },
        student_policy=lambda _: np.zeros(16, dtype=np.float32),
    )
    write_episode_shard(
        root,
        f"student_gate_ref{ref_id}_seed{seed}",
        shard,
        split="test",
    )


def test_student_gate_applies_configured_performance_failure_budget(
    tmp_path: Path,
) -> None:
    references = ReferenceSet.from_config()
    contract = ActionContract.from_dict(load_yaml("configs/deployment_contract.yaml"))
    for ref_id, seed in enumerate((10, 11, 12)):
        _write_student_episode(tmp_path, seed, ref_id)

    manifest_path = tmp_path / "manifest.jsonl"
    records = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
    ]
    records[-1]["success"] = False
    manifest_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )

    result = evaluate_student_closed_loop_gate(
        tmp_path,
        StudentClosedLoopGateConfig(
            expected_seeds=(10, 11, 12),
            full_episode_steps=8,
            success_rate_min=2.0 / 3.0,
            per_reference_success_rate_min=0.0,
            required_ref_ids=(0, 1, 2),
        ),
        references,
        contract,
    )
    assert result["ok"] is True
    assert result["summary"]["successes"] == 2
    assert result["checks"]["hard_invariant_failures_empty"] is True
    assert result["episodes"][-1]["failures"] == [
        "did_not_complete_full_horizon"
    ]


def test_student_gate_keeps_performance_and_dagger_admission_separate(
    tmp_path: Path,
) -> None:
    references = ReferenceSet.from_config()
    contract = ActionContract.from_dict(load_yaml("configs/deployment_contract.yaml"))
    for ref_id, seed in enumerate((20, 21, 22)):
        _write_student_episode(tmp_path, seed, ref_id)

    manifest_path = tmp_path / "manifest.jsonl"
    records = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
    ]
    records[-1]["success"] = False
    manifest_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )

    result = evaluate_student_closed_loop_gate(
        tmp_path,
        StudentClosedLoopGateConfig(
            expected_seeds=(20, 21, 22),
            full_episode_steps=8,
            success_rate_min=1.0,
            per_reference_success_rate_min=1.0,
            required_ref_ids=(0, 1, 2),
            gate_purpose="dagger_admission",
            dagger_admission_full_horizon_success_rate_min=2.0 / 3.0,
            dagger_admission_per_reference_mean_horizon_fraction_min=1.0,
            dagger_admission_minimum_episode_horizon_fraction=1.0,
        ),
        references,
        contract,
    )
    assert result["performance_ok"] is False
    assert result["dagger_admission_ok"] is True
    assert result["ok"] is True
    assert result["summary"]["full_horizon_success_rate"] == 2.0 / 3.0
