#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import numpy as np
import torch

from _bootstrap import ROOT, load_contract, write_json

from lateral_mppi_dagger.config import load_yaml, sha256_file
from lateral_mppi_dagger.evaluation.key7_handoff import (
    build_key7_handoff_observation,
    replay_key7_dry_inference,
)
from lateral_mppi_dagger.student.model import build_student_from_checkpoint


def _checkpoint_policy(path: Path):
    model, _ = build_student_from_checkpoint(str(path), map_location="cpu")
    model.eval()

    def infer(observation: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            return model(torch.from_numpy(observation)).numpy()[0]

    return infer, model


def _onnx_policy(path: Path):
    import onnxruntime as ort

    session = ort.InferenceSession(
        str(path),
        providers=["CPUExecutionProvider"],
    )

    def infer(observation: np.ndarray) -> np.ndarray:
        return session.run(
            ["actions"],
            {"obs": np.asarray(observation, dtype=np.float32).reshape(1, 93)},
        )[0][0]

    return infer


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Offline key7 handoff replay. It never imports, launches, or writes "
            "the deployment repository."
        )
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--onnx", type=Path)
    parser.add_argument(
        "--golden",
        type=Path,
        default=ROOT / "configs/key7_handoff_golden.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/21_key7_handoff_validation.json",
    )
    args = parser.parse_args()
    if args.checkpoint is None and args.onnx is None:
        parser.error("At least one of --checkpoint or --onnx is required.")

    contract = load_contract()
    golden = load_yaml(args.golden)
    gates = golden["gates"]
    handoff = build_key7_handoff_observation(contract, golden)
    expected_quaternion = np.asarray(
        golden["nominal_handoff"][
            "expected_aligned_reference_quaternion_wxyz"
        ],
        dtype=np.float32,
    )
    expected_rotation = np.asarray(
        golden["nominal_handoff"]["expected_rotation_columns6"],
        dtype=np.float32,
    )
    quaternion_error = float(
        np.max(
            np.abs(
                handoff.aligned_reference_quaternion_wxyz
                - expected_quaternion
            )
        )
    )
    rotation_error = float(
        np.max(np.abs(handoff.observation[0, 32:38] - expected_rotation))
    )
    default_joint = np.asarray(
        contract["action"]["q_action_offset_runtime"],
        dtype=np.float32,
    )
    handoff_joint = np.asarray(
        golden["nominal_handoff"]["robot_joint_position_policy"],
        dtype=np.float32,
    )
    leg_handoff_error = float(
        np.max(np.abs(handoff_joint[:12] - default_joint[:12]))
    )
    checks = {
        "aligned_reference_quaternion": (
            quaternion_error <= float(gates["rotation_atol"])
        ),
        "rotation_columns": rotation_error <= float(gates["rotation_atol"]),
        "handoff_leg_pose": (
            leg_handoff_error
            <= float(gates["handoff_leg_position_error_max_rad"])
        ),
    }
    backends = {}
    if args.checkpoint is not None:
        checkpoint = args.checkpoint.resolve()
        policy, model = _checkpoint_policy(checkpoint)
        replay = replay_key7_dry_inference(
            handoff.observation,
            policy,
            int(
                golden["key7"][
                    "dry_inference_cycles_before_first_applied_action"
                ]
            ),
            np.asarray(contract["action"]["scale"], dtype=np.float32),
        )
        normalized_rotation = (
            torch.from_numpy(handoff.observation)[0, 32:38]
            - model.observation_mean[32:38]
        ) / model.observation_std[32:38]
        normalized_max = float(torch.max(torch.abs(normalized_rotation)).item())
        replay["artifact"] = str(checkpoint)
        replay["sha256"] = sha256_file(checkpoint)
        replay["normalized_rotation"] = normalized_rotation.tolist()
        replay["normalized_rotation_max_abs"] = normalized_max
        replay["checks"] = {
            "normalized_rotation_in_distribution": (
                normalized_max
                <= float(gates["normalized_rotation_max_abs"])
            ),
            "first_applied_physical_delta": (
                replay["first_applied"][
                    "physical_leg_delta_max_abs_rad"
                ]
                <= float(
                    gates["first_applied_action_physical_delta_max_rad"]
                )
            ),
            "hard_zero_wheels": (
                replay["first_applied"]["wheel_action_max_abs"] == 0.0
            ),
        }
        checks["checkpoint"] = all(replay["checks"].values())
        backends["checkpoint"] = replay
    if args.onnx is not None:
        onnx = args.onnx.resolve()
        replay = replay_key7_dry_inference(
            handoff.observation,
            _onnx_policy(onnx),
            int(
                golden["key7"][
                    "dry_inference_cycles_before_first_applied_action"
                ]
            ),
            np.asarray(contract["action"]["scale"], dtype=np.float32),
        )
        replay["artifact"] = str(onnx)
        replay["sha256"] = sha256_file(onnx)
        replay["checks"] = {
            "first_applied_physical_delta": (
                replay["first_applied"][
                    "physical_leg_delta_max_abs_rad"
                ]
                <= float(
                    gates["first_applied_action_physical_delta_max_rad"]
                )
            ),
            "hard_zero_wheels": (
                replay["first_applied"]["wheel_action_max_abs"] == 0.0
            ),
        }
        checks["onnx"] = all(replay["checks"].values())
        backends["onnx"] = replay

    result = {
        "schema_version": "pcbc-key7-handoff-validation-v1",
        "ok": all(checks.values()),
        "golden": str(args.golden.resolve()),
        "golden_sha256": sha256_file(args.golden.resolve()),
        "deployment_repository_accessed": False,
        "deployment_repository_modified": False,
        "observation": {
            "aligned_reference_quaternion_wxyz": (
                handoff.aligned_reference_quaternion_wxyz.tolist()
            ),
            "rotation_columns6": handoff.observation[0, 32:38].tolist(),
            "aligned_reference_quaternion_max_abs_error": quaternion_error,
            "rotation_columns_max_abs_error": rotation_error,
            "handoff_leg_pose_max_abs_error_rad": leg_handoff_error,
        },
        "backends": backends,
        "checks": checks,
    }
    write_json(args.output, result)
    print(
        json.dumps(
            {
                "ok": result["ok"],
                "checks": checks,
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
        )
    )
    if not result["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
