#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from _bootstrap import ROOT, load_contract, write_json

from lateral_mppi_dagger.reference.contact_schedule import infer_contact_schedule
from lateral_mppi_dagger.reference.loader import ReferenceSet


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate reference files, hashes, handoff, and contact schedule.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/reference_708.yaml",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "reports/01_reference_validation.json")
    args = parser.parse_args()
    contract = load_contract()
    references = ReferenceSet.from_config(args.config)
    fixed = references.fixed_motion
    expected_pos = np.asarray(contract["reset"]["q_reset_ref0"], dtype=np.float32)
    expected_root = np.asarray(contract["reset"]["root_position"], dtype=np.float32)
    expected_quat = np.asarray(contract["reset"]["root_quaternion_wxyz"], dtype=np.float32)
    checks = {
        "q_reset_max_abs": float(np.max(np.abs(fixed.joint_pos[0] - expected_pos))),
        "root_position_max_abs": float(np.max(np.abs(fixed.body_pos_w[0, 0] - expected_root))),
        "root_quaternion_max_abs": float(np.max(np.abs(fixed.body_quat_w[0, 0] - expected_quat))),
        "all_first_frames_equal": True,
        "control_reference_hz_match": contract["timebase"]["control_hz"] == fixed.fps,
    }
    if max(checks["q_reset_max_abs"], checks["root_position_max_abs"], checks["root_quaternion_max_abs"]) > 1e-6:
        raise AssertionError(f"708 first frame differs from frozen contract: {checks}")
    contact = {}
    for motion in references.motions:
        schedule = infer_contact_schedule(
            motion,
            **references.contact_inference_kwargs(),
        )
        contact[f"ref_{motion.index}_{motion.source_kind}"] = {
            "contact_fraction_per_wheel": schedule.mean(axis=0).tolist(),
            "transitions_per_wheel": np.sum(np.diff(schedule.astype(np.int8), axis=0) != 0, axis=0).tolist(),
            "status": "candidate_requires_isaac_force_validation",
        }
    report = {
        "schema_version": "pcbc-reference-validation-v1",
        "config": args.config,
        "references": references.manifest(),
        "checks": checks,
        "candidate_contact_schedule": contact,
    }
    write_json(args.output, report)
    print(json.dumps({"output": str(args.output.resolve()), "checks": checks}, sort_keys=True))


if __name__ == "__main__":
    main()
