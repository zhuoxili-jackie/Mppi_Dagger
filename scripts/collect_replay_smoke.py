#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from _bootstrap import ROOT, load_contract

from lateral_mppi_dagger.contract.action16 import Action16Adapter, ActionContract, SafetyShield
from lateral_mppi_dagger.data.collector import CollectorConfig, collect_episode
from lateral_mppi_dagger.data.schema import write_episode_shard
from lateral_mppi_dagger.env.replay_env import ReplayContractEnv
from lateral_mppi_dagger.expert.reference_wbc import ReferenceWBCExpert
from lateral_mppi_dagger.reference.loader import ReferenceSet


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a kinematic contract-smoke dataset. This is not an expert/Isaac gate."
    )
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets/replay_smoke")
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    contract = load_contract()
    references = ReferenceSet.from_config()
    action_contract = ActionContract.from_dict(contract)
    adapter = Action16Adapter(action_contract)
    expert = ReferenceWBCExpert(adapter)
    records = []
    splits = ("train",) * 8 + ("validation",) * 2 + ("test",) * 2
    for episode in range(args.episodes):
        ref_id = episode % len(references)
        seed = args.seed + episode
        environment = ReplayContractEnv(references, contract, adapter)
        shard = collect_episode(
            environment,
            expert,
            SafetyShield(action_contract),
            CollectorConfig(
                seed=seed,
                ref_id=ref_id,
                max_steps=args.steps,
                beta=1.0,
                scenario="offline_contract_smoke_only",
            ),
            {
                "expert_backend": "reference_wbc",
                "environment_backend": "offline_contract_smoke_only",
                "wheel_action_mode": action_contract.wheel_action_mode.value,
                "joint_order": contract["joint_order_policy"],
                "action_scale": contract["action"]["scale"],
                "q_reset_ref0": contract["reset"]["q_reset_ref0"],
                "q_action_offset_runtime": contract["action"]["q_action_offset_runtime"],
                "observation_schema": contract["observation"],
                "control_frequency": contract["timebase"]["control_hz"],
                "reference_sha256": references[ref_id].sha256,
            },
        )
        split = splits[episode % len(splits)]
        record = write_episode_shard(
            args.dataset,
            f"replay_ref{ref_id}_seed{seed}",
            shard,
            split=split,
        )
        records.append(record)
    print(json.dumps({"dataset": str(args.dataset.resolve()), "episodes": len(records)}, sort_keys=True))


if __name__ == "__main__":
    main()

