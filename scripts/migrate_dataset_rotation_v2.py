#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from _bootstrap import ROOT, load_contract, write_json

from lateral_mppi_dagger.config import canonical_hash, sha256_file
from lateral_mppi_dagger.data.dataset import load_manifest
from lateral_mppi_dagger.data.observation_migration import (
    ROTATION_SLICE,
    migrate_clean_observation_rows_to_columns,
    migrate_noisy_training_observation_rows_to_columns,
)
from lateral_mppi_dagger.data.schema import (
    EpisodeShard,
    read_episode_shard,
    write_episode_shard,
)


_CLEAN_FIELDS = (
    "obs93_clean",
    "next_obs93_clean",
    "obs93_dynamic",
    "next_obs93_dynamic",
)


def _assert_legacy_rows_schema(metadata: dict) -> dict:
    schema = metadata.get("observation_schema")
    if not isinstance(schema, dict):
        raise ValueError("Episode has no observation_schema metadata.")
    rotation = schema.get("rotation_6d", {})
    slices = schema.get("slices", [])
    names = {str(item.get("name")) for item in slices if isinstance(item, dict)}
    if (
        "motion_anchor_ori_b_rows01" not in names
        or rotation.get("flatten_order") != "C"
    ):
        raise ValueError(
            "Source dataset is not the known v1 row-layout dataset; refusing "
            "an ambiguous or repeated migration."
        )
    return schema


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a new immutable dataset with key7/Isaac-compatible rotation "
            "columns. The source dataset is never modified."
        )
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/20_rotation_dataset_migration.json",
    )
    args = parser.parse_args()
    source = args.source.resolve()
    destination = args.destination.resolve()
    if source == destination:
        parser.error("--source and --destination must differ.")
    if destination.exists():
        parser.error(
            f"Destination already exists; refusing to overwrite or append: {destination}"
        )

    contract = load_contract()
    target_schema = copy.deepcopy(contract["observation"])
    source_records = load_manifest(source)
    episode_reports = []
    total_steps = 0
    maximum_rotation_delta = 0.0
    for record in source_records:
        source_path = source / record["path"]
        shard = read_episode_shard(source_path)
        source_schema = _assert_legacy_rows_schema(shard.metadata)
        arrays = {name: value.copy() for name, value in shard.arrays.items()}
        legacy_clean = arrays["obs93_clean"].copy()
        arrays["obs93_train"] = migrate_noisy_training_observation_rows_to_columns(
            legacy_clean,
            arrays["obs93_train"],
        )
        for field in _CLEAN_FIELDS:
            if field in arrays:
                before = arrays[field]
                after = migrate_clean_observation_rows_to_columns(before)
                maximum_rotation_delta = max(
                    maximum_rotation_delta,
                    float(
                        np.max(
                            np.abs(
                                after[..., ROTATION_SLICE]
                                - before[..., ROTATION_SLICE]
                            )
                        )
                    ),
                )
                arrays[field] = after

        metadata = copy.deepcopy(shard.metadata)
        metadata["observation_schema"] = target_schema
        metadata["deployment_contract_schema_version"] = contract["schema_version"]
        metadata["observation_contract_migration"] = {
            "schema_version": "pcbc-observation-layout-migration-v1",
            "source_episode": str(source_path),
            "source_episode_sha256": sha256_file(source_path),
            "source_observation_schema": source_schema,
            "target_observation_schema": target_schema,
            "operation": (
                "legacy rows [R00,R01,R02,R10,R11,R12] -> deployment columns "
                "[R00,R01,R10,R11,R20,R21]"
            ),
            "non_rotation_channels": "bit_identical",
            "noisy_rotation_channels": "clean geometric conversion plus iid noise remap",
        }
        migrated = EpisodeShard(arrays=arrays, metadata=metadata)
        migrated.validate()
        target_record = write_episode_shard(
            destination,
            str(record["episode_id"]),
            migrated,
            split=str(record["split"]),
        )
        steps = int(target_record["steps"])
        total_steps += steps
        episode_reports.append(
            {
                "episode_id": record["episode_id"],
                "steps": steps,
                "source_sha256": sha256_file(source_path),
                "destination_sha256": sha256_file(
                    destination / target_record["path"]
                ),
            }
        )

    result = {
        "schema_version": "pcbc-rotation-dataset-migration-report-v1",
        "source": str(source),
        "destination": str(destination),
        "source_manifest_sha256": sha256_file(source / "manifest.jsonl"),
        "destination_manifest_sha256": sha256_file(
            destination / "manifest.jsonl"
        ),
        "deployment_contract_schema_version": contract["schema_version"],
        "deployment_contract_hash": canonical_hash(contract),
        "episodes": len(episode_reports),
        "steps": total_steps,
        "maximum_rotation_slot_delta": maximum_rotation_delta,
        "source_unchanged": True,
        "episode_reports": episode_reports,
    }
    write_json(args.output, result)
    print(
        json.dumps(
            {
                "episodes": result["episodes"],
                "steps": result["steps"],
                "maximum_rotation_slot_delta": maximum_rotation_delta,
                "destination": str(destination),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
