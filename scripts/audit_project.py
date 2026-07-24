#!/usr/bin/env python3
"""Audit the standalone tree without reading either reference repository."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from _bootstrap import ISAACLAB_ROOT, ISAACLAB_SOURCES, ROBOT_LAB_SOURCE, ROOT, load_contract, write_json

from lateral_mppi_dagger.config import load_yaml, resolve_project_path, sha256_file


ASSET_SUFFIXES = {".csv", ".npz", ".obj", ".stl", ".urdf", ".usd"}
REQUIRED_STANDALONE_PATHS = (
    "configs/deployment_contract.yaml",
    "configs/reference_708.yaml",
    "configs/reference_low_load_v1.yaml",
    "assets/references/low_load_v1/generation_report.json",
    "src/lateral_mppi_dagger/env/__init__.py",
    "src/lateral_mppi_dagger/env/action_delay.py",
    "src/lateral_mppi_dagger/env/isaac_adapter.py",
    "src/lateral_mppi_dagger/env/isaac_mppi_rollout.py",
    "src/lateral_mppi_dagger/env/replay_env.py",
    "src/lateral_mppi_dagger/env/safety_metrics.py",
    "src/lateral_mppi_dagger/env/scenarios.py",
    "src/lateral_mppi_dagger/env/state_snapshot.py",
    "vendor/robot_lab/SNAPSHOT.json",
    "vendor/robot_lab/LICENSE",
    "vendor/robot_lab/config/extension.toml",
    "vendor/robot_lab/robot_lab/__init__.py",
    "vendor/robot_lab/data/Robots/pcbC/pcb_v2_description_0.88/urdf/pcb_v88.urdf",
    "vendor/robot_lab/data/Robots/pcbC/pcb_v2_description_0.88/mesh/530X6U_simple.usd",
    "vendor/robot_lab/data/Robots/pcbC/pcb_v2_description_0.88/mesh/530X6U_simple.stl",
    "vendor/deployment_evidence/deployment_manifest.yaml",
    "vendor/IsaacLab/SNAPSHOT.json",
    "vendor/IsaacLab/LICENSE",
    "vendor/IsaacLab/apps/isaaclab.python.headless.kit",
    "vendor/IsaacLab/source/isaaclab/isaaclab/__init__.py",
    "vendor/IsaacLab/source/isaaclab_assets/isaaclab_assets/__init__.py",
    "vendor/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/__init__.py",
    "vendor/IsaacLab/source/isaaclab_rl/isaaclab_rl/__init__.py",
    "vendor/IsaacLab/source/isaaclab_mimic/isaaclab_mimic/__init__.py",
    "reference_only/README.md",
    "reference_only/SNAPSHOT.json",
)
FORBIDDEN_RUNTIME_FRAGMENTS = (
    "/robot_move_lt/",
    "/robot_move_it/",
    "/transfer/robot_move_lt/",
    "/transfer/RL_augmented_MPC/",
    "../source/robot_lab",
    "../双足搭车侧向移动数据708",
    "reference_only/",
    "import whole_body_mppi",
    "from whole_body_mppi",
    "legged_mppi @",
)


def run(command: list[str]) -> str:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        return f"ERROR({result.returncode}): {result.stderr.strip()}"
    return result.stdout.strip()


def package_versions() -> dict[str, str]:
    result = {}
    for name in (
        "torch",
        "onnx",
        "onnxruntime",
        "rsl-rl-lib",
        "isaacsim",
        "isaaclab",
        "isaaclab_assets",
        "isaaclab_tasks",
        "robot_lab",
    ):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "NOT_INSTALLED"
    return result


def is_below_root(path: Path) -> bool:
    try:
        path.resolve().relative_to(ROOT.resolve())
    except ValueError:
        return False
    return True


def relevant_assets() -> list[Path]:
    data_root = ROOT / "vendor/robot_lab/data"
    evidence_root = ROOT / "vendor/deployment_evidence"
    selected = {
        path
    for directory in (data_root, evidence_root, ROOT / "assets")
        if directory.is_dir()
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in ASSET_SUFFIXES | {".yaml", ".yml"}
    }
    return sorted(selected)


def validate_contract_paths(contract: dict) -> list[dict]:
    declared = [
        (
            "motion_prefix.reference_file",
            contract["motion_prefix"]["reference_file"],
            contract["motion_prefix"]["reference_sha256"],
        ),
        *[
            (f"assets.{name}", item["path"], item["sha256"])
            for name, item in contract["assets"].items()
            if isinstance(item, dict) and "path" in item and "sha256" in item
        ],
        (
            "source_evidence.legacy_bundle_manifest",
            contract["source_evidence"]["legacy_bundle_manifest"],
            contract["source_evidence"]["legacy_bundle_manifest_sha256"],
        ),
    ]
    checks = []
    for name, configured_path, expected_hash in declared:
        path = resolve_project_path(configured_path)
        actual_hash = sha256_file(path) if path.is_file() else None
        checks.append(
            {
                "name": name,
                "configured_path": configured_path,
                "resolved_path": str(path),
                "inside_standalone_root": is_below_root(path),
                "present": path.is_file(),
                "expected_sha256": expected_hash,
                "actual_sha256": actual_hash,
                "hash_matches": actual_hash == expected_hash,
            }
        )
    return checks


def validate_reference_files() -> list[dict]:
    checks = []
    for config_path in (
        "configs/reference_708.yaml",
        "configs/reference_low_load_v1.yaml",
    ):
        config = load_yaml(config_path)
        reference_root = resolve_project_path(config["reference_directory"])
        for item in config["references"]:
            path = reference_root / item["file"]
            actual_hash = sha256_file(path) if path.is_file() else None
            checks.append(
                {
                    "config": config_path,
                    "index": item["index"],
                    "path": str(path),
                    "inside_standalone_root": is_below_root(path),
                    "present": path.is_file(),
                    "expected_sha256": item["sha256"],
                    "actual_sha256": actual_hash,
                    "hash_matches": actual_hash == item["sha256"],
                }
            )
    return checks


def validate_urdf_meshes() -> list[dict]:
    urdf_root = ROOT / "vendor/robot_lab/data/Robots/pcbC/pcb_v2_description_0.88/urdf"
    checks = []
    # pcb_v88.urdf is the frozen runtime asset.  The snapshot also retains an
    # unused historical URDF whose upstream mesh directory is named
    # inconsistently ("meshes" versus "mesh"); it is evidence, not a runtime
    # dependency, so it is deliberately outside this executable-asset gate.
    for urdf in (urdf_root / "pcb_v88.urdf",):
        tree = ET.parse(urdf)
        for mesh in tree.findall(".//mesh"):
            filename = mesh.attrib.get("filename", "")
            if not filename or "://" in filename:
                continue
            path = (urdf.parent / filename).resolve()
            checks.append(
                {
                    "urdf": str(urdf.relative_to(ROOT)),
                    "mesh_reference": filename,
                    "resolved_path": str(path),
                    "inside_standalone_root": is_below_root(path),
                    "present": path.is_file(),
                }
            )
    return checks


def scan_runtime_boundaries() -> list[dict[str, str | int]]:
    """Reject code/config links back into either one-time reference checkout."""
    selected = [
        *sorted((ROOT / "configs").rglob("*.yaml")),
        *sorted((ROOT / "scripts").glob("*.py")),
        *sorted((ROOT / "scripts").glob("*.sh")),
        *sorted((ROOT / "src").rglob("*.py")),
        ROOT / "pyproject.toml",
        ROOT
        / "vendor/robot_lab/robot_lab/tasks/manager_based/beyondmimic/config/pcbc"
        / "lateral_bipedal_stand.py",
        ROOT
        / "vendor/robot_lab/robot_lab/tasks/manager_based/beyondmimic/config/pcbc"
        / "lateral_command_curriculum.py",
    ]
    matches = []
    for path in selected:
        if not path.is_file():
            continue
        if path.resolve() == Path(__file__).resolve():
            # This auditor necessarily declares the forbidden fragments that
            # it searches for; those literals are not runtime dependencies.
            continue
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            for fragment in FORBIDDEN_RUNTIME_FRAGMENTS:
                if fragment in line:
                    matches.append(
                        {
                            "path": str(path.relative_to(ROOT)),
                            "line": line_number,
                            "fragment": fragment,
                        }
                    )
    return matches


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit only the movable lateral_mppi_dagger directory."
    )
    parser.add_argument("--output", type=Path, default=ROOT / "reports/00_audit_state.json")
    args = parser.parse_args()

    contract = load_contract()
    assets = relevant_assets()
    required = [
        {
            "path": relative,
            "present": (ROOT / relative).is_file() or (ROOT / relative).is_dir(),
        }
        for relative in REQUIRED_STANDALONE_PATHS
    ]
    contract_checks = validate_contract_paths(contract)
    reference_checks = validate_reference_files()
    urdf_mesh_checks = validate_urdf_meshes()
    forbidden_runtime_references = scan_runtime_boundaries()
    external_symlinks = [
        {
            "path": str(path.relative_to(ROOT)),
            "target": str(path.resolve()),
        }
        for path in ROOT.rglob("*")
        if path.is_symlink() and not is_below_root(path)
    ]
    robot_snapshot_path = ROOT / "vendor/robot_lab/SNAPSHOT.json"
    robot_snapshot = (
        json.loads(robot_snapshot_path.read_text(encoding="utf-8"))
        if robot_snapshot_path.is_file()
        else None
    )
    isaaclab_snapshot_path = ROOT / "vendor/IsaacLab/SNAPSHOT.json"
    isaaclab_snapshot = (
        json.loads(isaaclab_snapshot_path.read_text(encoding="utf-8"))
        if isaaclab_snapshot_path.is_file()
        else None
    )
    failures = [
        *(f"missing:{item['path']}" for item in required if not item["present"]),
        *(
            f"contract:{item['name']}"
            for item in contract_checks
            if not (item["inside_standalone_root"] and item["present"] and item["hash_matches"])
        ),
        *(
            f"reference:{item['config']}:{item['index']}"
            for item in reference_checks
            if not (item["inside_standalone_root"] and item["present"] and item["hash_matches"])
        ),
        *(
            f"urdf_mesh:{item['mesh_reference']}"
            for item in urdf_mesh_checks
            if not (item["inside_standalone_root"] and item["present"])
        ),
        *(f"forbidden_runtime_reference:{item['path']}:{item['line']}" for item in forbidden_runtime_references),
        *(f"external_symlink:{item['path']}" for item in external_symlinks),
    ]
    audit = {
        "schema_version": "pcbc-standalone-audit-v2",
        "ok": not failures,
        "failures": failures,
        "timestamp_timezone": os.environ.get("TZ", "system"),
        "standalone_root": str(ROOT.resolve()),
        "original_repository_accessed": False,
        "vendored_robot_lab_source": str(ROBOT_LAB_SOURCE.resolve()),
        "vendored_isaaclab_root": str(ISAACLAB_ROOT.resolve()),
        "vendored_isaaclab_sources": [str(path.resolve()) for path in ISAACLAB_SOURCES],
        "robot_lab_snapshot": robot_snapshot,
        "isaaclab_snapshot": isaaclab_snapshot,
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "gpu": run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,memory.total",
                    "--format=csv,noheader",
                ]
            ).splitlines(),
            "packages": package_versions(),
        },
        "required_paths": required,
        "contract_paths": contract_checks,
        "reference_files": reference_checks,
        "urdf_mesh_references": urdf_mesh_checks,
        "forbidden_runtime_references": forbidden_runtime_references,
        "external_symlinks": external_symlinks,
        "asset_summary": {
            "file_count": len(assets),
            "bytes": sum(path.stat().st_size for path in assets),
            "stl_count": sum(path.suffix.lower() == ".stl" for path in assets),
            "usd_count": sum(path.suffix.lower() == ".usd" for path in assets),
            "urdf_count": sum(path.suffix.lower() == ".urdf" for path in assets),
            "npz_count": sum(path.suffix.lower() == ".npz" for path in assets),
        },
        "assets": [
            {
                "path": str(path.relative_to(ROOT)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in assets
        ],
        "contract_evidence": {
            "legacy_baseline_onnx_present": (
                ROOT / "vendor/deployment_evidence/runtime/models/baseline.onnx"
            ).is_file(),
            "legacy_residual_onnx_present": (
                ROOT / "vendor/deployment_evidence/runtime/models/residual_candidate.onnx"
            ).is_file(),
        },
    }
    write_json(args.output, audit)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "ok": audit["ok"],
                "asset_count": len(assets),
                "failures": failures,
            },
            sort_keys=True,
        )
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
