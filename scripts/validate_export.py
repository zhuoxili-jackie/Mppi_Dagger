#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import ROOT, write_json

from lateral_mppi_dagger.export.validator import validate_export_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate fixed ABI, parity, exact wheel zero, and CPU latency.")
    parser.add_argument("--exported", type=Path, default=ROOT / "exported")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/export_validation.json")
    args = parser.parse_args()
    metrics = validate_export_bundle(args.exported)
    write_json(args.output, metrics)
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()

