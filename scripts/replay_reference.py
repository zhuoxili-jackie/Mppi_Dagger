#!/usr/bin/env python3
from __future__ import annotations

# Gate-1 replay uses the same physical target/action path as expert collection.
# Keeping one implementation prevents a second hand-written observation/action ABI.
from smoke_test_expert import args_cli, run_isaac_collection, simulation_app
from _bootstrap import ROOT


if __name__ == "__main__":
    try:
        run_isaac_collection(args_cli, ROOT / "reports/01b_reference_replay.json")
    finally:
        simulation_app.close()

