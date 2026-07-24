#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from _bootstrap import ROOT, write_json

from lateral_mppi_dagger.config import sha256_file
from lateral_mppi_dagger.data.dataset import load_manifest
from lateral_mppi_dagger.data.schema import read_episode_shard


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256_file(source) != sha256_file(destination):
            raise FileExistsError(
                f"Destination episode exists with different bytes: {destination}"
            )
        return
    with tempfile.NamedTemporaryFile(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        with source.open("rb") as source_stream:
            shutil.copyfileobj(source_stream, stream)
        stream.flush()
        os.fsync(stream.fileno())
    if sha256_file(source) != sha256_file(temporary):
        temporary.unlink(missing_ok=True)
        raise IOError(f"Copied episode failed byte-parity validation: {source}")
    os.replace(temporary, destination)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Transactionally merge immutable episode shards into an aggregate "
            "dataset while preserving byte-for-byte provenance."
        )
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/dataset_merge.json",
    )
    args = parser.parse_args()

    source_root = args.source.resolve()
    destination_root = args.destination.resolve()
    source_records = load_manifest(source_root)
    destination_records = (
        load_manifest(destination_root)
        if (destination_root / "manifest.jsonl").is_file()
        else []
    )
    destination_by_id = {
        str(record["episode_id"]): record for record in destination_records
    }
    source_ids = [str(record["episode_id"]) for record in source_records]
    collisions = sorted(set(source_ids) & set(destination_by_id))
    if collisions:
        raise ValueError(
            "Refusing to merge duplicate episode IDs: "
            + ", ".join(collisions[:10])
        )

    copied = []
    for record in source_records:
        source_path = source_root / record["path"]
        destination_path = destination_root / record["path"]
        shard = read_episode_shard(source_path)
        if int(record["steps"]) != shard.validate():
            raise ValueError(f"Manifest step count differs for {record['episode_id']}.")
        digest = sha256_file(source_path)
        _atomic_copy(source_path, destination_path)
        if sha256_file(destination_path) != digest:
            raise IOError(
                f"Destination byte parity failed for {record['episode_id']}."
            )
        copied.append(
            {
                "episode_id": record["episode_id"],
                "source": str(source_path),
                "destination": str(destination_path),
                "sha256": digest,
                "bytes": source_path.stat().st_size,
                "steps": int(record["steps"]),
            }
        )

    destination_root.mkdir(parents=True, exist_ok=True)
    manifest_path = destination_root / "manifest.jsonl"
    combined_records = destination_records + source_records
    encoded = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in combined_records
    )
    with tempfile.NamedTemporaryFile(
        prefix=".manifest.",
        suffix=".jsonl.tmp",
        dir=destination_root,
        mode="w",
        encoding="utf-8",
        delete=False,
    ) as stream:
        temporary_manifest = Path(stream.name)
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary_manifest, manifest_path)
    reloaded = load_manifest(destination_root)
    if len(reloaded) != len(combined_records):
        raise RuntimeError("Merged manifest failed record-count parity.")

    result = {
        "schema_version": "pcbc-dataset-merge-v1",
        "source": str(source_root),
        "destination": str(destination_root),
        "source_episodes": len(source_records),
        "destination_episodes_before": len(destination_records),
        "destination_episodes_after": len(reloaded),
        "copied_steps": sum(item["steps"] for item in copied),
        "copied_bytes": sum(item["bytes"] for item in copied),
        "byte_parity": True,
        "episodes": copied,
    }
    write_json(args.output, result)
    print(
        json.dumps(
            {
                "episodes": result["source_episodes"],
                "steps": result["copied_steps"],
                "byte_parity": result["byte_parity"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
