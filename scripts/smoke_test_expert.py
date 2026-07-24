#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from _bootstrap import ROOT, write_json

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Run a bounded expert smoke test in the real 708 trunk task.")
parser.add_argument(
    "--task",
    default="RobotLab-Isaac-BeyondMimic-Flat-pcbC-v1-lateral-bipedal-stand-v0",
)
parser.add_argument("--dataset", type=Path, default=ROOT / "datasets/isaac_expert_smoke")
parser.add_argument("--episodes", type=int, default=1)
parser.add_argument("--steps", type=int, default=32)
parser.add_argument("--seed", type=int, default=1000)
parser.add_argument("--ref-id", type=int, default=0)
parser.add_argument(
    "--reference-config",
    type=str,
    default=None,
)
parser.add_argument("--rotate-references", action="store_true")
parser.add_argument("--run-name", default="expert_smoke")
parser.add_argument("--scenario", default="nominal_smoke")
parser.add_argument("--dagger-round", type=int, default=0)
parser.add_argument("--split", choices=("auto", "train", "validation", "test"), default="train")
parser.add_argument("--beta", type=float, default=1.0)
parser.add_argument(
    "--observation-noise-std",
    type=float,
    default=None,
    help="Override the selected scenario profile; omitted means use the profile value.",
)
parser.add_argument("--student-checkpoint", type=Path, default=None)
parser.add_argument(
    "--expert-backend",
    choices=("mppi", "reference_wbc", "disabled"),
    default="mppi",
)
parser.add_argument(
    "--mppi-config",
    type=Path,
    default=ROOT / "configs/expert_mppi.yaml",
)
parser.add_argument("--mppi-samples", type=int, default=16)
parser.add_argument("--mppi-horizon", type=int, default=8)
parser.add_argument("--mppi-iterations", type=int, default=1)
parser.add_argument("--mppi-temperature", type=float, default=None)
parser.add_argument(
    "--mppi-selection-mode",
    choices=("weighted", "best_sample"),
    default=None,
)
parser.add_argument("--mppi-noise-scale", type=float, default=1.0)
parser.add_argument("--disable-fabric", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from _isaac_workflow import run_isaac_collection


if __name__ == "__main__":
    try:
        run_isaac_collection(args_cli, ROOT / "reports/02_expert_smoke.json")
    except BaseException as exc:
        write_json(
            ROOT / "reports/02_expert_smoke.failure.json",
            {
                "error_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
                "arguments": vars(args_cli),
            },
        )
        traceback.print_exc()
        sys.stderr.flush()
        raise
    finally:
        simulation_app.close()
