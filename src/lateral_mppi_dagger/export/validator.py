from __future__ import annotations

import json
import hashlib
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch


def _shape_from_onnx(value_info: onnx.ValueInfoProto) -> list[int | str | None]:
    shape: list[int | str | None] = []
    for dimension in value_info.type.tensor_type.shape.dim:
        if dimension.HasField("dim_value"):
            shape.append(int(dimension.dim_value))
        elif dimension.HasField("dim_param"):
            shape.append(dimension.dim_param)
        else:
            shape.append(None)
    return shape


def validate_export_bundle(
    output_dir: str | Path,
    max_abs_threshold: float = 1.0e-5,
    mean_abs_threshold: float = 1.0e-6,
    latency_p95_threshold_ms: float = 20.0,
) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    required_files = {
        "policy.pt",
        "policy.onnx",
        "golden_io.npz",
        "export_manifest.json",
        "SHA256SUMS",
    }
    missing = sorted(name for name in required_files if not (output / name).is_file())
    if missing:
        raise FileNotFoundError(f"Export bundle is missing files: {missing}")
    manifest = json.loads((output / "export_manifest.json").read_text(encoding="utf-8"))
    verified_hashes: dict[str, str] = {}
    for filename, expected_hash in manifest["files"].items():
        path = output / filename
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(
                f"Export artifact hash mismatch for {filename}: {actual_hash} != {expected_hash}"
            )
        verified_hashes[filename] = actual_hash
    sums = {}
    for line in (output / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, filename = line.split(maxsplit=1)
        sums[filename.strip()] = digest
    for filename in ("policy.pt", "policy.onnx", "golden_io.npz", "export_manifest.json"):
        actual_hash = hashlib.sha256((output / filename).read_bytes()).hexdigest()
        if sums.get(filename) != actual_hash:
            raise ValueError(f"SHA256SUMS mismatch or missing entry for {filename}.")
    with np.load(output / "golden_io.npz", allow_pickle=False) as archive:
        observations = np.asarray(archive["obs"], dtype=np.float32)
        expected = np.asarray(archive["actions"], dtype=np.float32)

    onnx_model = onnx.load(output / "policy.onnx")
    onnx.checker.check_model(onnx_model)
    if len(onnx_model.graph.input) != 1 or len(onnx_model.graph.output) != 1:
        raise ValueError("ONNX deployment graph must have exactly one input and one output.")
    graph_input = onnx_model.graph.input[0]
    graph_output = onnx_model.graph.output[0]
    input_shape = _shape_from_onnx(graph_input)
    output_shape = _shape_from_onnx(graph_output)
    if graph_input.name != "obs" or input_shape != [1, 93]:
        raise ValueError(f"ONNX input ABI mismatch: name={graph_input.name}, shape={input_shape}")
    if graph_output.name != "actions" or output_shape != [1, 16]:
        raise ValueError(f"ONNX output ABI mismatch: name={graph_output.name}, shape={output_shape}")
    opsets = {item.domain: item.version for item in onnx_model.opset_import}
    if opsets.get("", opsets.get("ai.onnx")) != int(manifest["opset"]):
        raise ValueError(f"ONNX opset mismatch: graph={opsets}, manifest={manifest['opset']}")
    onnx_metadata = {item.key: item.value for item in onnx_model.metadata_props}
    required_metadata = {
        "schema_version": manifest["deployment_contract_schema_version"],
        "policy_family": manifest["policy_family"],
        "wheel_action_mode": manifest["wheel_action_mode"],
        "motion_prefix_semantics": manifest["motion_prefix_semantics"],
        "reference_file": manifest["reference_file"],
        "reference_sha256": manifest["reference_sha256"],
        "model_hash": manifest["model_hash"],
        "contract_hash": manifest["contract_hash"],
        "zero_command_previous_action_deadband": str(
            manifest["model_spec"]["zero_command_previous_action_deadband"]
        ),
    }
    optional_safety_metadata = (
        "lateral_command_activation_start_m_s",
        "lateral_command_activation_full_m_s",
        "lateral_command_abs_limit_m_s",
        "physical_target_rate_limit_rad_s",
        "physical_target_abs_limit_rad",
        "physical_target_abs_limit_rad_by_joint",
        "control_dt_s",
    )
    for key in optional_safety_metadata:
        if key not in manifest["model_spec"]:
            continue
        value = manifest["model_spec"][key]
        required_metadata[key] = (
            json.dumps(value) if isinstance(value, list) else str(value)
        )
    for key, expected_value in required_metadata.items():
        if key not in onnx_metadata or onnx_metadata[key] != str(expected_value):
            raise ValueError(
                f"ONNX metadata mismatch for {key}: "
                f"{onnx_metadata.get(key)!r} != {str(expected_value)!r}"
            )

    scripted = torch.jit.load(str(output / "policy.pt"), map_location="cpu").eval()
    with torch.inference_mode():
        torchscript_actions = scripted(torch.from_numpy(observations)).numpy()

    session = ort.InferenceSession(str(output / "policy.onnx"), providers=["CPUExecutionProvider"])
    onnx_actions = []
    for observation in observations[: min(10, observations.shape[0])]:
        session.run(["actions"], {"obs": observation.reshape(1, 93)})
    latencies_ms = []
    started = time.perf_counter()
    for observation in observations:
        sample_started = time.perf_counter()
        result = session.run(["actions"], {"obs": observation.reshape(1, 93)})[0]
        latencies_ms.append((time.perf_counter() - sample_started) * 1000.0)
        onnx_actions.append(result[0])
    total_seconds = time.perf_counter() - started
    onnx_actions_array = np.asarray(onnx_actions, dtype=np.float32)

    differences = {
        "torchscript_vs_eager": np.abs(torchscript_actions - expected),
        "onnx_vs_eager": np.abs(onnx_actions_array - expected),
    }
    metrics: dict[str, Any] = {
        "samples": int(observations.shape[0]),
        "onnx_input_name": graph_input.name,
        "onnx_input_shape": input_shape,
        "onnx_output_name": graph_output.name,
        "onnx_output_shape": output_shape,
        "opset": int(manifest["opset"]),
        "onnx_cpu_total_ms": total_seconds * 1000.0,
        "onnx_cpu_mean_ms": total_seconds * 1000.0 / observations.shape[0],
        "onnx_cpu_p50_ms": float(np.percentile(latencies_ms, 50)),
        "onnx_cpu_p95_ms": float(np.percentile(latencies_ms, 95)),
        "onnx_cpu_p99_ms": float(np.percentile(latencies_ms, 99)),
        "onnx_cpu_max_ms": float(np.max(latencies_ms)),
        "latency_p95_threshold_ms": float(latency_p95_threshold_ms),
        "file_hashes_verified": verified_hashes,
        "sha256sums_verified": True,
        "onnx_metadata_verified": True,
    }
    for name, difference in differences.items():
        metrics[name] = {
            "max_abs_error": float(difference.max()),
            "mean_abs_error": float(difference.mean()),
        }
        if difference.max() > max_abs_threshold or difference.mean() > mean_abs_threshold:
            raise AssertionError(f"{name} parity failed: {metrics[name]}")

    for backend_name, actions in (
        ("eager", expected),
        ("torchscript", torchscript_actions),
        ("onnx", onnx_actions_array),
    ):
        wheels = actions[:, 12:16]
        if not np.array_equal(wheels, np.zeros_like(wheels)):
            raise AssertionError(f"{backend_name} does not produce exact hard-zero wheel actions.")
    metrics["hard_zero_all_backends"] = True
    model_spec = manifest["model_spec"]
    safety_rate = float(
        model_spec.get("physical_target_rate_limit_rad_s", 0.0)
    )
    safety_abs = float(
        model_spec.get("physical_target_abs_limit_rad", 0.0)
    )
    safety_abs_by_joint = np.asarray(
        model_spec.get(
            "physical_target_abs_limit_rad_by_joint",
            [safety_abs] * 12,
        ),
        dtype=np.float32,
    )
    if safety_rate > 0.0:
        control_dt = float(model_spec["control_dt_s"])
        action_scale = np.asarray(manifest["action_scale"], dtype=np.float32)
        base_observation = observations[0:1].copy()
        base_observation[:, 73:89] = 0.0
        base_observation[:, 89:92] = 0.0

        def run_safety_probe(name: str, infer) -> dict[str, float | bool]:
            current = base_observation.copy()
            previous_physical = np.zeros(12, dtype=np.float32)
            maximum_step = 0.0
            maximum_target = 0.0
            absolute_limit_pass = True
            zero_exact = True
            commands = (0.0, 0.0, 0.0, 0.012, 0.024) + (0.030,) * 15
            for command in commands:
                current[0, 90] = command
                action = np.asarray(infer(current), dtype=np.float32).reshape(16)
                physical = action[:12] * action_scale[:12]
                if command == 0.0:
                    zero_exact &= bool(
                        np.array_equal(action, np.zeros(16, dtype=np.float32))
                    )
                maximum_step = max(
                    maximum_step,
                    float(np.max(np.abs(physical - previous_physical))),
                )
                maximum_target = max(
                    maximum_target,
                    float(np.max(np.abs(physical))),
                )
                if np.any(safety_abs_by_joint > 0.0):
                    absolute_limit_pass &= bool(
                        np.all(
                            np.abs(physical)
                            <= safety_abs_by_joint + 1.0e-6
                        )
                    )
                current[0, 73:89] = action
                previous_physical = physical
            return {
                "backend": name,
                "zero_handoff_exact": zero_exact,
                "maximum_physical_leg_step_rad": maximum_step,
                "maximum_physical_leg_target_rad": maximum_target,
                "pass": bool(
                    zero_exact
                    and maximum_step <= safety_rate * control_dt + 1.0e-6
                    and absolute_limit_pass
                ),
            }

        safety_probes = {
            "torchscript": run_safety_probe(
                "torchscript",
                lambda value: scripted(torch.from_numpy(value)).detach().numpy()[0],
            ),
            "onnx": run_safety_probe(
                "onnx",
                lambda value: session.run(
                    ["actions"],
                    {"obs": value.reshape(1, 93)},
                )[0][0],
            ),
        }
        if not all(bool(probe["pass"]) for probe in safety_probes.values()):
            raise AssertionError(
                f"Embedded activation/rate/absolute safety probe failed: {safety_probes}"
            )
        metrics["embedded_safety_probes"] = safety_probes
        metrics["embedded_safety_gate_pass"] = True
    else:
        metrics["embedded_safety_gate_pass"] = None
    if metrics["onnx_cpu_p95_ms"] > latency_p95_threshold_ms:
        raise AssertionError(
            "ONNX CPU p95 latency exceeds the 50 Hz deployment budget: "
            f"{metrics['onnx_cpu_p95_ms']:.3f} ms > {latency_p95_threshold_ms:.3f} ms"
        )
    metrics["latency_50hz_pass"] = True
    return metrics
