from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from lateral_mppi_dagger.config import load_yaml
from lateral_mppi_dagger.contract.action16 import Action16Adapter, ActionContract, SafetyShield
from lateral_mppi_dagger.data.collector import CollectorConfig, collect_episode
from lateral_mppi_dagger.data.schema import SCHEMA_VERSION, read_episode_shard, write_episode_shard
from lateral_mppi_dagger.data.dataset import EpisodeWindowDataset
from lateral_mppi_dagger.env.replay_env import ReplayContractEnv
from lateral_mppi_dagger.expert.reference_wbc import ReferenceWBCExpert
from lateral_mppi_dagger.reference.loader import ReferenceSet


def test_episode_schema_preserves_transition_and_masks(tmp_path: Path) -> None:
    contract = load_yaml("configs/deployment_contract.yaml")
    references = ReferenceSet.from_config()
    action_contract = ActionContract.from_dict(contract)
    adapter = Action16Adapter(action_contract)
    student_inputs: list[np.ndarray] = []

    def student_policy(observation: np.ndarray) -> np.ndarray:
        student_inputs.append(observation[0].copy())
        return np.zeros(16, dtype=np.float32)

    shard = collect_episode(
        ReplayContractEnv(references, contract, adapter),
        ReferenceWBCExpert(adapter),
        SafetyShield(action_contract),
        CollectorConfig(
            seed=7,
            ref_id=0,
            max_steps=8,
            scenario="unit_test",
            observation_noise_std=0.01,
        ),
        {"schema_version": SCHEMA_VERSION, "wheel_action_mode": "hard_zero"},
        student_policy=student_policy,
    )
    assert shard.validate() == 8
    record = write_episode_shard(tmp_path, "episode_7", shard, split="train")
    loaded = read_episode_shard(tmp_path / record["path"])
    np.testing.assert_array_equal(loaded.arrays["step_id"], np.arange(8, dtype=np.int32))
    np.testing.assert_array_equal(
        loaded.arrays["executed_action16"][:, 12:],
        np.zeros((8, 4), dtype=np.float32),
    )
    np.testing.assert_array_equal(
        loaded.arrays["scheduled_action16"],
        loaded.arrays["executed_action16"],
    )
    assert loaded.arrays["terminal"][-1] == 1
    assert loaded.metadata["success"] is True
    assert "obs93_dynamic" in loaded.arrays
    assert np.any(loaded.arrays["obs93_train"] != loaded.arrays["obs93_clean"])
    np.testing.assert_array_equal(
        loaded.arrays["obs93_train"][:, 53:57],
        np.zeros((8, 4), dtype=np.float32),
    )
    np.testing.assert_array_equal(
        loaded.arrays["obs93_train"][:, 85:89],
        np.zeros((8, 4), dtype=np.float32),
    )
    np.testing.assert_array_equal(
        loaded.arrays["obs93_train"][:, 92],
        np.zeros(8, dtype=np.float32),
    )
    np.testing.assert_array_equal(
        np.asarray(student_inputs, dtype=np.float32),
        loaded.arrays["obs93_train"],
    )


def test_dagger_sampling_allocates_mass_by_round() -> None:
    dataset = object.__new__(EpisodeWindowDataset)
    dataset.index = [
        (0, 0),
        (0, 1),
        (1, 0),
        (2, 0),
        (2, 1),
        (2, 2),
    ]
    dataset.shards = [
        ({"dagger_round": 0}, None),
        ({"dagger_round": 1}, None),
        ({"dagger_round": 2}, None),
    ]
    weights = dataset.dagger_sampling_weights(latest_round=2)
    assert np.isclose(weights[:2].sum(), 0.30)
    assert np.isclose(weights[2], 0.30)
    assert np.isclose(weights[3:].sum(), 0.40)


def test_dagger_recovery_sampling_prioritizes_latest_student_states() -> None:
    dataset = object.__new__(EpisodeWindowDataset)
    dataset.window_length = 1
    dataset.index = [
        (0, 0),
        (0, 1),
        (1, 0),
        (2, 0),
        (2, 1),
        (3, 0),
        (3, 1),
        (3, 2),
    ]
    dataset.shards = [
        (
            {"dagger_round": 0},
            SimpleNamespace(
                arrays={"behavior_policy": np.asarray([0, 0], dtype=np.uint8)}
            ),
        ),
        (
            {"dagger_round": 1},
            SimpleNamespace(
                arrays={"behavior_policy": np.asarray([1], dtype=np.uint8)}
            ),
        ),
        (
            {"dagger_round": 2},
            SimpleNamespace(
                arrays={"behavior_policy": np.asarray([0, 0], dtype=np.uint8)}
            ),
        ),
        (
            {"dagger_round": 2},
            SimpleNamespace(
                arrays={"behavior_policy": np.asarray([1, 1, 1], dtype=np.uint8)}
            ),
        ),
    ]
    weights = dataset.dagger_recovery_sampling_weights(latest_round=2)
    assert np.isclose(weights[:2].sum(), 0.20)
    assert np.isclose(weights[2], 0.20)
    assert np.isclose(weights[3:5].sum(), 0.10)
    assert np.isclose(weights[5:].sum(), 0.50)
