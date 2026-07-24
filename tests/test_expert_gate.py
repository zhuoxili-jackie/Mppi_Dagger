from __future__ import annotations

from pathlib import Path

import numpy as np

from lateral_mppi_dagger.config import load_yaml
from lateral_mppi_dagger.contract.action16 import Action16Adapter, ActionContract, SafetyShield
from lateral_mppi_dagger.data.collector import CollectorConfig, collect_episode
from lateral_mppi_dagger.data.schema import SCHEMA_VERSION, write_episode_shard
from lateral_mppi_dagger.env.replay_env import ReplayContractEnv
from lateral_mppi_dagger.evaluation.expert_gate import ExpertGateConfig, evaluate_expert_gate
from lateral_mppi_dagger.expert.reference_wbc import ReferenceWBCExpert
from lateral_mppi_dagger.reference.loader import ReferenceSet


def _write_gate_episode(root: Path, seed: int, ref_id: int, source: str = "mppi") -> None:
    contract = load_yaml("configs/deployment_contract.yaml")
    references = ReferenceSet.from_config()
    adapter = Action16Adapter(ActionContract.from_dict(contract))
    shard = collect_episode(
        ReplayContractEnv(references, contract, adapter),
        ReferenceWBCExpert(adapter),
        SafetyShield(adapter.contract),
        CollectorConfig(seed=seed, ref_id=ref_id, max_steps=8, scenario="gate_test"),
        {
            "schema_version": SCHEMA_VERSION,
            "wheel_action_mode": "hard_zero",
            "expert_backend": "mppi",
            "expert_config_hash": "test-config",
        },
    )
    shard.arrays["label_source"][:] = 3
    for name in (
        "mppi_cost_components",
        "mppi_minimum_total_cost",
        "mppi_mean_total_cost",
        "mppi_effective_sample_size",
        "mppi_rollout_termination_rate",
    ):
        shard.arrays[name][:] = 0.0
    shard.metadata["expert_backend"] = source
    write_episode_shard(root, f"gate_ref{ref_id}_seed{seed}", shard, split="test")


def test_expert_gate_passes_complete_exact_zero_set(tmp_path: Path) -> None:
    for index, seed in enumerate((10, 11, 12)):
        _write_gate_episode(tmp_path, seed, index)
    result = evaluate_expert_gate(
        tmp_path,
        ExpertGateConfig(
            expected_seeds=(10, 11, 12),
            required_successes=3,
            full_episode_steps=8,
            required_ref_ids=(0, 1, 2),
        ),
    )
    assert result["ok"] is True
    assert result["summary"]["successes"] == 3


def test_expert_gate_fails_nonzero_wheel_action(tmp_path: Path) -> None:
    _write_gate_episode(tmp_path, 10, 0)
    episode = next((tmp_path / "episodes").glob("*.npz"))
    with np.load(episode, allow_pickle=False) as archive:
        payload = {name: np.asarray(archive[name]) for name in archive.files}
    payload["executed_action16"] = payload["executed_action16"].copy()
    payload["executed_action16"][0, 12] = np.float32(1.0e-7)
    np.savez_compressed(episode, **payload)
    result = evaluate_expert_gate(
        tmp_path,
        ExpertGateConfig(
            expected_seeds=(10,),
            required_successes=1,
            full_episode_steps=8,
            required_ref_ids=(0,),
        ),
    )
    assert result["ok"] is False
    assert result["checks"]["wheel_action_exact_zero"] is False


def test_expert_gate_tracking_uses_configured_episode_failure_budget(
    tmp_path: Path,
) -> None:
    references = ReferenceSet.from_config()
    contract = ActionContract.from_dict(load_yaml("configs/deployment_contract.yaml"))
    for index, seed in enumerate((10, 11, 12)):
        _write_gate_episode(tmp_path, seed, index)

    bad_episode = next(
        (tmp_path / "episodes").glob("*seed12.npz")
    )
    with np.load(bad_episode, allow_pickle=False) as archive:
        payload = {name: np.asarray(archive[name]) for name in archive.files}
    payload["base_pose_w"] = payload["base_pose_w"].copy()
    payload["base_pose_w"][-1, 0] += np.float32(1.0)
    np.savez_compressed(bad_episode, **payload)

    result = evaluate_expert_gate(
        tmp_path,
        ExpertGateConfig(
            expected_seeds=(10, 11, 12),
            required_successes=2,
            full_episode_steps=8,
            required_ref_ids=(0, 1, 2),
            tracking_thresholds={
                "base_position_max_abs_m": [0.2, 100.0, 100.0],
                "box_local_x_drift_max_abs_m": 0.2,
                "base_orientation_rmse_rad": 100.0,
                "wheel_position_rmse_m": 100.0,
                "contact_mismatch_rate": 1.0,
            },
        ),
        references=references,
        action_contract=contract,
    )
    assert result["ok"] is True
    assert result["summary"]["successes"] == 2
    assert result["summary"]["tracking_successes"] == 2
    assert result["checks"]["tracking_thresholds"] is True
