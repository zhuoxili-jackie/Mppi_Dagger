from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import torch

from lateral_mppi_dagger.config import canonical_hash, sha256_file
from lateral_mppi_dagger.student.model import build_student_from_checkpoint


def _tensor_hash(*tensors: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        value = tensor.detach().cpu().contiguous().numpy()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(value.shape).encode("ascii"))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _state_dict_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(value.shape).encode("ascii"))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _add_onnx_metadata(path: Path, metadata: dict[str, Any]) -> None:
    model = onnx.load(path)
    del model.metadata_props[:]
    for key, value in sorted(metadata.items()):
        item = model.metadata_props.add()
        item.key = key
        item.value = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
    onnx.checker.check_model(model)
    onnx.save(model, path)


def export_student_policy(
    checkpoint_path: str | Path,
    output_dir: str | Path,
    contract: dict[str, Any],
    golden_observations: np.ndarray,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    observations = np.asarray(golden_observations, dtype=np.float32)
    if observations.ndim != 2 or observations.shape[1] != 93 or observations.shape[0] == 0:
        raise ValueError(f"golden_observations must have shape [K,93], got {observations.shape}")
    if not np.isfinite(observations).all():
        raise ValueError("golden_observations contain NaN or Inf")

    model, checkpoint = build_student_from_checkpoint(str(checkpoint_path), map_location="cpu")
    checkpoint_hash = sha256_file(checkpoint_path)
    model_hash = _state_dict_hash(model.state_dict())
    contract_hash = canonical_hash(contract)
    model.eval()
    with torch.inference_mode():
        eager_actions = model(torch.from_numpy(observations)).numpy()
    if not np.array_equal(eager_actions[:, 12:], np.zeros_like(eager_actions[:, 12:])):
        raise RuntimeError("Hard-zero wheel invariant failed before export.")

    torchscript_path = output / "policy.pt"
    scripted = torch.jit.script(model)
    scripted.save(str(torchscript_path))

    onnx_path = output / "policy.onnx"
    example = torch.from_numpy(observations[:1])
    torch.onnx.export(
        model,
        example,
        str(onnx_path),
        export_params=True,
        opset_version=int(contract["export"]["opset"]),
        do_constant_folding=True,
        input_names=[contract["export"]["input_name"]],
        output_names=[contract["export"]["output_name"]],
        dynamic_axes=None,
        dynamo=False,
    )
    metadata = {
        "schema_version": contract["schema_version"],
        "policy_family": contract["policy_family"],
        "wheel_action_mode": contract["action"]["wheel_action_mode"],
        "motion_prefix_semantics": contract["motion_prefix"]["semantics"],
        "reference_file": contract["motion_prefix"]["reference_file"],
        "reference_sha256": contract["motion_prefix"]["reference_sha256"],
        "reference_index": contract["motion_prefix"]["reference_index"],
        "first_frame_joint_pos": contract["motion_prefix"]["first_frame_joint_pos"],
        "first_frame_joint_vel": contract["motion_prefix"]["first_frame_joint_vel"],
        "previous_action_semantics": contract["observation"]["previous_action_semantics"],
        "zero_command_previous_action_deadband": (
            model.zero_command_previous_action_deadband
        ),
        "lateral_command_activation_start_m_s": (
            model.lateral_command_activation_start_m_s
        ),
        "lateral_command_activation_full_m_s": (
            model.lateral_command_activation_full_m_s
        ),
        "lateral_command_abs_limit_m_s": (
            model.lateral_command_abs_limit_m_s
        ),
        "physical_target_rate_limit_rad_s": (
            model.physical_target_rate_limit_rad_s
        ),
        "physical_target_abs_limit_rad": (
            model.physical_target_abs_limit_rad
        ),
        "physical_target_abs_limit_rad_by_joint": (
            model.physical_target_abs_limit_by_joint.detach().cpu().tolist()
        ),
        "control_dt_s": model.control_dt_s,
        "joint_order_policy": contract["joint_order_policy"],
        "runtime_index_map_from_policy": contract["runtime_index_map_from_policy"],
        "model_hash": model_hash,
        "contract_hash": contract_hash,
    }
    _add_onnx_metadata(onnx_path, metadata)

    golden_path = output / "golden_io.npz"
    np.savez_compressed(
        golden_path,
        obs=observations,
        actions=eager_actions.astype(np.float32),
    )
    normalization_hash = _tensor_hash(model.observation_mean, model.observation_std)
    manifest = {
        "schema_version": "pcbc-student-export-v1",
        "deployment_contract_schema_version": contract["schema_version"],
        "policy_family": contract["policy_family"],
        "model_hash": model_hash,
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "trainer_config_hash": checkpoint.get("trainer_config_hash"),
        "dataset_root_at_training": checkpoint.get("dataset_root"),
        "dataset_manifest_hash": checkpoint.get("dataset_manifest_hash"),
        "initialized_from": checkpoint.get("initialized_from"),
        "wheel_action_mode": contract["action"]["wheel_action_mode"],
        "motion_prefix_semantics": contract["motion_prefix"]["semantics"],
        "reference_file": contract["motion_prefix"]["reference_file"],
        "reference_sha256": contract["motion_prefix"]["reference_sha256"],
        "reference_index": contract["motion_prefix"]["reference_index"],
        "first_frame_joint_pos": contract["motion_prefix"]["first_frame_joint_pos"],
        "first_frame_joint_vel": contract["motion_prefix"]["first_frame_joint_vel"],
        "input": {
            "name": contract["export"]["input_name"],
            "shape": contract["export"]["input_shape"],
            "dtype": contract["export"]["dtype"],
        },
        "output": {
            "name": contract["export"]["output_name"],
            "shape": contract["export"]["output_shape"],
            "dtype": contract["export"]["dtype"],
        },
        "opset": contract["export"]["opset"],
        "dynamic_axes": contract["export"]["dynamic_axes"],
        "observation_schema": contract["observation"],
        "previous_action_semantics": contract["observation"]["previous_action_semantics"],
        "runtime_handoff": contract["runtime_handoff"],
        "joint_order_policy": contract["joint_order_policy"],
        "runtime_index_map_from_policy": contract["runtime_index_map_from_policy"],
        "action_scale": contract["action"]["scale"],
        "q_action_offset_runtime": contract["action"]["q_action_offset_runtime"],
        "normalization": {
            "embedded": True,
            "mean_std_sha256": normalization_hash,
        },
        "model_spec": checkpoint["model_spec"],
        "contract_hash": contract_hash,
        "files": {},
    }
    manifest_path = output / "export_manifest.json"
    artifact_paths = (torchscript_path, onnx_path, golden_path)
    manifest["files"] = {path.name: sha256_file(path) for path in artifact_paths}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    sums_path = output / "SHA256SUMS"
    with sums_path.open("w", encoding="utf-8") as stream:
        for path in (*artifact_paths, manifest_path):
            stream.write(f"{sha256_file(path)}  {path.name}\n")
    return manifest
