from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT
PACKAGE_SOURCE = ROOT / "src"
VENDOR_ROOT = ROOT / "vendor"
ROBOT_LAB_SOURCE = VENDOR_ROOT / "robot_lab"
ISAACLAB_ROOT = VENDOR_ROOT / "IsaacLab"
ISAACLAB_SOURCES = tuple(
    ISAACLAB_ROOT / "source" / name
    for name in (
        "isaaclab",
        "isaaclab_assets",
        "isaaclab_tasks",
        "isaaclab_rl",
        "isaaclab_mimic",
    )
)
IMPORT_ROOTS = (PACKAGE_SOURCE, ROBOT_LAB_SOURCE, *ISAACLAB_SOURCES)
for path in reversed(tuple(str(item) for item in IMPORT_ROOTS)):
    if path not in sys.path:
        sys.path.insert(0, path)


def load_contract() -> dict[str, Any]:
    from lateral_mppi_dagger.config import load_yaml

    return load_yaml("configs/deployment_contract.yaml")


def load_reference_config() -> dict[str, Any]:
    from lateral_mppi_dagger.config import load_yaml

    return load_yaml("configs/reference_708.yaml")


def verify_current_robot_lab_import() -> Path:
    import robot_lab

    loaded = Path(robot_lab.__file__).resolve()
    expected = ROBOT_LAB_SOURCE.resolve()
    if expected not in loaded.parents:
        raise RuntimeError(
            f"Wrong robot_lab import: loaded {loaded}, expected a module below {expected}. "
            "Do not train until PYTHONPATH/import precedence is fixed."
        )
    return loaded


def verify_current_isaaclab_import() -> Path:
    import isaaclab

    loaded = Path(isaaclab.__file__).resolve()
    expected = ISAACLAB_ROOT.resolve()
    if expected not in loaded.parents:
        raise RuntimeError(
            f"Wrong isaaclab import: loaded {loaded}, expected a module below {expected}. "
            "The standalone runtime must not fall back to an editable install in the old project."
        )
    return loaded


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
