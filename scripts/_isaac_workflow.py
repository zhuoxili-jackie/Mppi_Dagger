from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch

from _bootstrap import (
    ROOT,
    load_contract,
    verify_current_isaaclab_import,
    verify_current_robot_lab_import,
    write_json,
)

from lateral_mppi_dagger.contract.action16 import (
    Action16Adapter,
    ActionContract,
    SafetyShield,
    WheelActionMode,
)
from lateral_mppi_dagger.config import (
    canonical_hash,
    load_yaml,
    resolve_project_path,
    sha256_file,
)
from lateral_mppi_dagger.data.collector import CollectorConfig, collect_episode
from lateral_mppi_dagger.data.dataset import load_manifest
from lateral_mppi_dagger.data.schema import read_episode_shard, write_episode_shard
from lateral_mppi_dagger.env.isaac_adapter import IsaacLateralAdapter
from lateral_mppi_dagger.env.isaac_mppi_rollout import (
    IsaacRolloutCostWeights,
    IsaacRolloutLoadLimits,
    IsaacWholeBodyMPPIProvider,
)
from lateral_mppi_dagger.env.isolated_mppi import (
    IsolatedIsaacMPPIProvider,
)
from lateral_mppi_dagger.env.scenarios import (
    configure_env_for_scenario,
    load_scenario_profile,
)
from lateral_mppi_dagger.expert.base import MPPI_COST_COMPONENT_NAMES
from lateral_mppi_dagger.expert.disabled import DisabledLabelExpert
from lateral_mppi_dagger.expert.mppi_expert import MPPIConfig, WholeBodyMPPIExpert
from lateral_mppi_dagger.expert.reference_wbc import ReferenceWBCExpert
from lateral_mppi_dagger.reference.loader import ReferenceSet
from lateral_mppi_dagger.reference.action_reference import (
    load_nominal_action_references,
)
from lateral_mppi_dagger.student.model import build_student_from_checkpoint


DEFAULT_TASK = "RobotLab-Isaac-BeyondMimic-Flat-pcbC-v1-lateral-bipedal-stand-v0"


def _git(command: list[str]) -> str | None:
    if not (ROOT / ".git").exists():
        return None
    result = subprocess.run(
        ["git", *command],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _provenance() -> dict[str, Any]:
    """Describe this tree without consulting the original MoveIt repository."""
    standalone_commit = _git(["rev-parse", "HEAD"])
    standalone_status = _git(["status", "--short"])
    snapshot_path = ROOT / "vendor/robot_lab/SNAPSHOT.json"
    snapshot = (
        json.loads(snapshot_path.read_text(encoding="utf-8"))
        if snapshot_path.is_file()
        else {}
    )
    source = snapshot.get("source_repository", {})
    return {
        "scope": "standalone_git" if standalone_commit else "vendored_source_snapshot",
        "commit": standalone_commit or source.get("commit", "UNKNOWN"),
        "dirty": bool(standalone_status) if standalone_status is not None else None,
        "dirty_status": standalone_status.splitlines() if standalone_status else [],
        "robot_lab_snapshot": str(snapshot_path.relative_to(ROOT)),
        "robot_lab_source_dirty_at_copy_time": source.get("dirty_at_copy_time"),
        "original_repository_accessed": False,
    }


def _student_callable(
    path: Path | None,
    device: str,
    action_contract: ActionContract,
):
    if path is None:
        return None
    path = path.expanduser().resolve()
    if path.suffix.lower() == ".onnx":
        import onnxruntime as ort

        providers = ["CPUExecutionProvider"]
        session = ort.InferenceSession(str(path), providers=providers)
        input_meta = session.get_inputs()
        output_meta = session.get_outputs()
        if len(input_meta) != 1 or input_meta[0].name != "obs":
            raise ValueError(f"Expected one ONNX input named 'obs', got {[item.name for item in input_meta]}")
        if len(output_meta) != 1 or output_meta[0].name != "actions":
            raise ValueError(
                f"Expected one ONNX output named 'actions', got {[item.name for item in output_meta]}"
            )
        if input_meta[0].shape != [1, 93] or output_meta[0].shape != [1, 16]:
            raise ValueError(
                "Legacy diagnostic ONNX must have fixed ABI [1,93] -> [1,16], got "
                f"{input_meta[0].shape} -> {output_meta[0].shape}"
            )

        def infer(observation: np.ndarray) -> np.ndarray:
            batch = np.asarray(observation, dtype=np.float32).reshape(1, 93)
            result = np.asarray(session.run(["actions"], {"obs": batch})[0][0], dtype=np.float32)
            if action_contract.wheel_action_mode is WheelActionMode.HARD_ZERO:
                result[12:] = 0.0
            return result

        return infer

    model, _ = build_student_from_checkpoint(str(path), map_location=device)
    model.to(device).eval()

    def infer(observation: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            result = model(torch.from_numpy(observation).to(device))
        result_np = np.asarray(result.detach().cpu().numpy(), dtype=np.float32).reshape(16)
        if action_contract.wheel_action_mode is WheelActionMode.HARD_ZERO:
            result_np[12:] = 0.0
        return result_np

    return infer


def run_isaac_collection(
    args: Any,
    report_path: Path,
) -> dict[str, Any]:
    import gymnasium as gym
    import robot_lab.tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    loaded_robot_lab = verify_current_robot_lab_import()
    loaded_isaaclab = verify_current_isaaclab_import()
    contract = load_contract()
    mppi_yaml = load_yaml(
        getattr(
            args,
            "mppi_config",
            ROOT / "configs/expert_mppi.yaml",
        )
    )
    reference_config = getattr(args, "reference_config", None) or mppi_yaml.get(
        "reference_config",
        "configs/reference_708.yaml",
    )
    references = ReferenceSet.from_config(
        reference_config
    )
    action_contract = ActionContract.from_dict(contract)
    action_adapter = Action16Adapter(action_contract)
    scenario_profile = load_scenario_profile(str(args.scenario))
    scenario_record = scenario_profile.metadata()
    scenario_config_hash = canonical_hash(scenario_record)
    observation_noise_arg = getattr(args, "observation_noise_std", None)
    observation_noise_std = (
        scenario_profile.observation_noise_std
        if observation_noise_arg is None
        else float(observation_noise_arg)
    )
    if observation_noise_std < 0.0:
        raise ValueError("--observation-noise-std must be non-negative.")
    expert_backend = getattr(args, "expert_backend", "reference_wbc")
    mppi_server_socket = getattr(args, "mppi_server_socket", None)
    if mppi_server_socket is not None and expert_backend != "mppi":
        raise ValueError(
            "--mppi-server-socket requires --expert-backend mppi."
        )
    if expert_backend == "disabled" and (
        args.student_checkpoint is None or float(args.beta) != 0.0
    ):
        raise ValueError(
            "The disabled label backend is restricted to beta=0 student-only "
            "evaluation with --student-checkpoint."
        )
    if expert_backend == "mppi":
        requested_mppi_samples = int(
            args.mppi_samples or mppi_yaml["samples"]
        )
        num_envs = 1 if mppi_server_socket is not None else requested_mppi_samples
    elif expert_backend in {"reference_wbc", "disabled"}:
        num_envs = 1
    else:
        raise ValueError(f"Unsupported expert backend {expert_backend!r}")

    env_cfg = parse_env_cfg(
        args.task,
        device=args.device,
        num_envs=num_envs,
        use_fabric=not args.disable_fabric,
    )
    configure_env_for_scenario(env_cfg, scenario_profile, num_envs=num_envs)
    env_cfg.seed = args.seed
    env_cfg.sim.device = args.device
    video_dir = getattr(args, "video_dir", None)
    if video_dir is not None and not bool(getattr(args, "enable_cameras", False)):
        raise ValueError("--video-dir requires AppLauncher --enable_cameras.")
    env = gym.make(
        args.task,
        cfg=env_cfg,
        render_mode="rgb_array" if video_dir is not None else None,
    )
    if video_dir is not None:
        video_dir = Path(video_dir).resolve()
        video_dir.mkdir(parents=True, exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(video_dir),
            step_trigger=lambda step: step == 0,
            video_length=int(args.steps),
            name_prefix=str(args.run_name),
            fps=int(contract["timebase"]["control_hz"]),
            disable_logger=True,
        )
    adapter = IsaacLateralAdapter(
        env,
        references,
        contract,
        scenario_profile=scenario_profile,
    )
    provider = None
    expert_implementation_sha256: dict[str, str] = {}
    if expert_backend == "mppi":
        (
            nominal_action_reference_q_des_by_ref,
            nominal_action_reference_raw_by_ref,
            nominal_action_reference_overrides_by_ref,
            nominal_action_reference_record,
        ) = load_nominal_action_references(
            mppi_yaml.get("nominal_action_reference")
        )
        mppi_config = MPPIConfig(
            horizon=int(args.mppi_horizon or mppi_yaml["horizon"]),
            samples=requested_mppi_samples,
            iterations=int(args.mppi_iterations or mppi_yaml["optimization_iterations"]),
            temperature=float(
                args.mppi_temperature
                if args.mppi_temperature is not None
                else mppi_yaml["temperature"]
            ),
            temporal_smoothing=float(mppi_yaml["temporal_smoothing"]),
            warm_start=bool(mppi_yaml["warm_start"]),
            selection_mode=str(
                args.mppi_selection_mode
                if getattr(args, "mppi_selection_mode", None) is not None
                else mppi_yaml.get("selection_mode", "weighted")
            ),
            reference_action_lookahead_steps=int(
                mppi_yaml.get("reference_action_lookahead_steps", 1)
            ),
            seed=int(mppi_yaml["seed"]),
        )
        noise_scale = float(getattr(args, "mppi_noise_scale", 1.0))
        if noise_scale <= 0.0:
            raise ValueError("--mppi-noise-scale must be positive.")
        noise_std = (
            torch.as_tensor(
                mppi_yaml["noise_std_leg"],
                dtype=torch.float32,
                device=args.device,
            )
            * noise_scale
        )
        mppi_config_path = resolve_project_path(
            getattr(
                args,
                "mppi_config",
                ROOT / "configs/expert_mppi.yaml",
            )
        )
        isolated_server_identity = {
            "schema_version": "pcbc-isolated-mppi-server-v1",
            "mppi_config_sha256": sha256_file(mppi_config_path),
            "reference_config": str(reference_config),
            "samples": mppi_config.samples,
            "horizon": mppi_config.horizon,
            "optimization_iterations": mppi_config.iterations,
        }
        if mppi_server_socket is None:
            provider = IsaacWholeBodyMPPIProvider(
                adapter,
                references,
                action_adapter,
                mppi_config,
                noise_std,
                IsaacRolloutCostWeights.from_dict(
                    mppi_yaml.get("cost_weights")
                ),
                load_limits=IsaacRolloutLoadLimits.from_dict(
                    mppi_yaml.get("load_limits")
                ),
                contact_force_threshold_n=float(
                    mppi_yaml["contact_force_threshold_n"]
                ),
                physical_target_rate_limit_rad_s=(
                    float(
                        mppi_yaml["physical_target_rate_limit_rad_s"]
                    )
                    if mppi_yaml.get("physical_target_rate_limit_rad_s")
                    is not None
                    else None
                ),
                nominal_action_reference_q_des_by_ref=(
                    nominal_action_reference_q_des_by_ref
                ),
                nominal_action_reference_raw_by_ref=(
                    nominal_action_reference_raw_by_ref
                ),
                nominal_action_reference_overrides_by_ref=(
                    nominal_action_reference_overrides_by_ref
                ),
                nominal_joint_position_bias_leg=mppi_yaml.get(
                    "nominal_joint_position_bias_leg"
                ),
                nominal_joint_position_bias_start_frame=int(
                    mppi_yaml.get(
                        "nominal_joint_position_bias_start_frame",
                        0,
                    )
                ),
                nominal_joint_position_bias_ramp_frames=int(
                    mppi_yaml.get(
                        "nominal_joint_position_bias_ramp_frames",
                        0,
                    )
                ),
                nominal_front_force_feedback_target_n=float(
                    mppi_yaml.get(
                        "nominal_front_force_feedback_target_n",
                        0.0,
                    )
                ),
                nominal_front_force_feedback_gain_leg=mppi_yaml.get(
                    "nominal_front_force_feedback_gain_leg"
                ),
                rear_swing_reference_proposal_ref_ids=mppi_yaml.get(
                    "rear_swing_reference_proposal_ref_ids"
                ),
                rear_swing_reference_proposal_scales=mppi_yaml.get(
                    "rear_swing_reference_proposal_scales"
                ),
                rear_swing_reference_proposal_joint_mask_leg=mppi_yaml.get(
                    "rear_swing_reference_proposal_joint_mask_leg"
                ),
                rear_swing_reference_proposal_lead_steps=int(
                    mppi_yaml.get(
                        "rear_swing_reference_proposal_lead_steps",
                        0,
                    )
                ),
                rear_swing_action_residual_lead_steps=(
                    int(
                        mppi_yaml[
                            "rear_swing_action_residual_lead_steps"
                        ]
                    )
                    if mppi_yaml.get(
                        "rear_swing_action_residual_lead_steps"
                    )
                    is not None
                    else None
                ),
                rear_swing_tracking_error_proposal_scales=mppi_yaml.get(
                    "rear_swing_tracking_error_proposal_scales"
                ),
                rear_swing_tracking_error_proposal_joint_mask_leg=(
                    mppi_yaml.get(
                        "rear_swing_tracking_error_proposal_joint_mask_leg"
                    )
                ),
                rear_swing_tracking_error_proposal_start_frame=int(
                    mppi_yaml.get(
                        "rear_swing_tracking_error_proposal_start_frame",
                        0,
                    )
                ),
                rear_swing_load_transfer_proposal_ref_ids=mppi_yaml.get(
                    "rear_swing_load_transfer_proposal_ref_ids"
                ),
                rear_swing_load_transfer_proposal_scales=mppi_yaml.get(
                    "rear_swing_load_transfer_proposal_scales"
                ),
                rear_swing_load_transfer_proposal_gain_by_wheel_leg_rad=(
                    mppi_yaml.get(
                        "rear_swing_load_transfer_proposal_gain_by_wheel_leg_rad"
                    )
                ),
                rear_swing_load_transfer_proposal_start_frame=int(
                    mppi_yaml.get(
                        "rear_swing_load_transfer_proposal_start_frame",
                        0,
                    )
                ),
                rear_swing_load_transfer_proposal_start_frame_by_wheel=(
                    mppi_yaml.get(
                        "rear_swing_load_transfer_proposal_start_frame_by_wheel"
                    )
                ),
                rear_swing_load_transfer_proposal_gate_mode=str(
                    mppi_yaml.get(
                        "rear_swing_load_transfer_proposal_gate_mode",
                        "swing_schedule",
                    )
                ),
                rear_swing_load_transfer_proposal_imbalance_threshold_n=float(
                    mppi_yaml.get(
                        "rear_swing_load_transfer_proposal_imbalance_threshold_n",
                        0.0,
                    )
                ),
                front_support_proposal_ref_ids=mppi_yaml.get(
                    "front_support_proposal_ref_ids"
                ),
                front_support_proposal_scales=mppi_yaml.get(
                    "front_support_proposal_scales"
                ),
                front_support_proposal_gain_leg_rad=mppi_yaml.get(
                    "front_support_proposal_gain_leg_rad"
                ),
                front_support_proposal_start_frame=int(
                    mppi_yaml.get(
                        "front_support_proposal_start_frame",
                        0,
                    )
                ),
                combine_rear_swing_front_support_proposals=mppi_yaml.get(
                    "combine_rear_swing_front_support_proposals",
                    False,
                ),
                combine_rear_swing_load_transfer_front_support_proposals=(
                    mppi_yaml.get(
                        "combine_rear_swing_load_transfer_front_support_proposals",
                        False,
                    )
                ),
                combine_rear_swing_reference_load_transfer_front_support_proposals=(
                    mppi_yaml.get(
                        "combine_rear_swing_reference_load_transfer_front_"
                        "support_proposals",
                        False,
                    )
                ),
                include_rear_support_reference_in_coordinated_proposals=(
                    mppi_yaml.get(
                        "include_rear_support_reference_in_coordinated_proposals",
                        False,
                    )
                ),
                rear_support_reference_proposal_start_frame=int(
                    mppi_yaml.get(
                        "rear_support_reference_proposal_start_frame",
                        0,
                    )
                ),
                output_front_force_feedback_target_n=float(
                    mppi_yaml.get(
                        "output_front_force_feedback_target_n",
                        0.0,
                    )
                ),
                output_front_force_feedback_min_contact_n=float(
                    mppi_yaml.get(
                        "output_front_force_feedback_min_contact_n",
                        0.0,
                    )
                ),
                output_front_force_feedback_lookahead_steps=mppi_yaml.get(
                    "output_front_force_feedback_lookahead_steps"
                ),
                output_front_force_feedback_gain_leg=mppi_yaml.get(
                    "output_front_force_feedback_gain_leg"
                ),
                output_rear_swing_force_feedback_target_n=float(
                    mppi_yaml.get(
                        "output_rear_swing_force_feedback_target_n",
                        0.0,
                    )
                ),
                output_rear_swing_force_feedback_scale_n=float(
                    mppi_yaml.get(
                        "output_rear_swing_force_feedback_scale_n",
                        1.0,
                    )
                ),
                output_rear_swing_force_feedback_lookahead_steps=(
                    mppi_yaml.get(
                        "output_rear_swing_force_feedback_lookahead_steps"
                    )
                ),
                output_rear_swing_force_feedback_start_frame=int(
                    mppi_yaml.get(
                        "output_rear_swing_force_feedback_start_frame",
                        0,
                    )
                ),
                output_rear_swing_force_feedback_gain_leg=mppi_yaml.get(
                    "output_rear_swing_force_feedback_gain_leg"
                ),
                output_rear_swing_height_feedback_ref_ids=mppi_yaml.get(
                    "output_rear_swing_height_feedback_ref_ids"
                ),
                output_rear_swing_height_feedback_gain=float(
                    mppi_yaml.get(
                        "output_rear_swing_height_feedback_gain",
                        0.0,
                    )
                ),
                output_rear_swing_height_feedback_max_abs_rad=float(
                    mppi_yaml.get(
                        "output_rear_swing_height_feedback_max_abs_rad",
                        0.0,
                    )
                ),
                output_rear_swing_height_feedback_lookahead_steps=(
                    mppi_yaml.get(
                        "output_rear_swing_height_feedback_lookahead_steps"
                    )
                ),
                output_rear_swing_height_feedback_start_frame=int(
                    mppi_yaml.get(
                        "output_rear_swing_height_feedback_start_frame",
                        0,
                    )
                ),
                output_rear_support_tracking_feedback_ref_ids=mppi_yaml.get(
                    "output_rear_support_tracking_feedback_ref_ids"
                ),
                output_rear_support_tracking_feedback_gain=float(
                    mppi_yaml.get(
                        "output_rear_support_tracking_feedback_gain",
                        0.0,
                    )
                ),
                output_rear_support_tracking_feedback_max_abs_rad=float(
                    mppi_yaml.get(
                        "output_rear_support_tracking_feedback_max_abs_rad",
                        0.0,
                    )
                ),
                output_rear_support_tracking_feedback_lookahead_steps=(
                    mppi_yaml.get(
                        "output_rear_support_tracking_feedback_lookahead_steps"
                    )
                ),
                output_rear_support_tracking_feedback_start_frame=int(
                    mppi_yaml.get(
                        "output_rear_support_tracking_feedback_start_frame",
                        0,
                    )
                ),
                output_pitch_feedback_ref_ids=mppi_yaml.get(
                    "output_pitch_feedback_ref_ids"
                ),
                output_pitch_feedback_gain_leg=mppi_yaml.get(
                    "output_pitch_feedback_gain_leg"
                ),
                output_pitch_feedback_axis=str(
                    mppi_yaml.get("output_pitch_feedback_axis", "y")
                ),
                output_pitch_feedback_start_frame=int(
                    mppi_yaml.get(
                        "output_pitch_feedback_start_frame",
                        0,
                    )
                ),
                output_pitch_feedback_max_abs_rad=float(
                    mppi_yaml.get(
                        "output_pitch_feedback_max_abs_rad",
                        0.0,
                    )
                ),
                output_joint_position_offset_leg=mppi_yaml.get(
                    "output_joint_position_offset_leg"
                ),
            )
            execution_isolation = {
                "mode": "shared_isaac_scene",
                "public_env_count": num_envs,
            }
        else:
            provider = IsolatedIsaacMPPIProvider(
                adapter,
                references,
                action_contract,
                mppi_server_socket,
                expected_server=isolated_server_identity,
            )
            execution_isolation = {
                "mode": "separate_persistent_isaac_process",
                "public_env_count": 1,
                "server": isolated_server_identity,
            }
        expert = WholeBodyMPPIExpert(provider)
        expert_implementation_sha256 = {
            path: sha256_file(ROOT / path)
            for path in (
                "scripts/_isaac_workflow.py",
                "src/lateral_mppi_dagger/expert/mppi_expert.py",
                "src/lateral_mppi_dagger/env/isaac_mppi_rollout.py",
                "src/lateral_mppi_dagger/env/isaac_adapter.py",
                "src/lateral_mppi_dagger/env/action_delay.py",
                "src/lateral_mppi_dagger/env/scenarios.py",
                "src/lateral_mppi_dagger/data/collector.py",
                "src/lateral_mppi_dagger/contract/action16.py",
                "src/lateral_mppi_dagger/reference/loader.py",
                "src/lateral_mppi_dagger/reference/action_reference.py",
                "src/lateral_mppi_dagger/env/isolated_mppi.py",
                "scripts/run_isolated_mppi_server.py",
            )
        }
        expert_runtime_config = {
            "backend": "whole_body_mppi",
            "control_hz": int(mppi_yaml["control_hz"]),
            "horizon": mppi_config.horizon,
            "samples": mppi_config.samples,
            "optimization_iterations": mppi_config.iterations,
            "temperature": mppi_config.temperature,
            "temporal_smoothing": mppi_config.temporal_smoothing,
            "warm_start": mppi_config.warm_start,
            "selection_mode": mppi_config.selection_mode,
            "reference_action_lookahead_steps": mppi_config.reference_action_lookahead_steps,
            "action_delay_steps": scenario_profile.action_delay_steps,
            "mppi_seed": mppi_config.seed,
            "episode_rng_seed_semantics": "mppi_config_seed_plus_episode_seed",
            "noise_std_leg": noise_std.detach().cpu().tolist(),
            "contact_force_threshold_n": float(mppi_yaml["contact_force_threshold_n"]),
            "cost_weights": dict(mppi_yaml["cost_weights"]),
            "load_limits": dict(mppi_yaml["load_limits"]),
            "physical_target_rate_limit_rad_s": mppi_yaml.get(
                "physical_target_rate_limit_rad_s"
            ),
            "nominal_action_reference": nominal_action_reference_record,
            "execution_isolation": execution_isolation,
            "nominal_joint_position_bias_leg": list(
                mppi_yaml.get("nominal_joint_position_bias_leg", [0.0] * 12)
            ),
            "nominal_joint_position_bias_start_frame": int(
                mppi_yaml.get("nominal_joint_position_bias_start_frame", 0)
            ),
            "nominal_joint_position_bias_ramp_frames": int(
                mppi_yaml.get("nominal_joint_position_bias_ramp_frames", 0)
            ),
            "nominal_front_force_feedback_target_n": float(
                mppi_yaml.get("nominal_front_force_feedback_target_n", 0.0)
            ),
            "nominal_front_force_feedback_gain_leg": list(
                mppi_yaml.get(
                    "nominal_front_force_feedback_gain_leg",
                    [0.0] * 12,
                )
            ),
            "rear_swing_reference_proposal_ref_ids": list(
                mppi_yaml.get(
                    "rear_swing_reference_proposal_ref_ids",
                    [],
                )
            ),
            "rear_swing_reference_proposal_scales": list(
                mppi_yaml.get(
                    "rear_swing_reference_proposal_scales",
                    [],
                )
            ),
            "rear_swing_reference_proposal_joint_mask_leg": list(
                mppi_yaml.get(
                    "rear_swing_reference_proposal_joint_mask_leg",
                    [0] * 12,
                )
            ),
            "rear_swing_reference_proposal_lead_steps": int(
                mppi_yaml.get(
                    "rear_swing_reference_proposal_lead_steps",
                    0,
                )
            ),
            "rear_swing_action_residual_lead_steps": int(
                mppi_yaml.get(
                    "rear_swing_action_residual_lead_steps",
                    mppi_yaml.get(
                        "rear_swing_reference_proposal_lead_steps",
                        0,
                    ),
                )
            ),
            "rear_swing_tracking_error_proposal_scales": list(
                mppi_yaml.get(
                    "rear_swing_tracking_error_proposal_scales",
                    [],
                )
            ),
            "rear_swing_tracking_error_proposal_joint_mask_leg": list(
                mppi_yaml.get(
                    "rear_swing_tracking_error_proposal_joint_mask_leg",
                    mppi_yaml.get(
                        "rear_swing_reference_proposal_joint_mask_leg",
                        [0] * 12,
                    ),
                )
            ),
            "rear_swing_tracking_error_proposal_start_frame": int(
                mppi_yaml.get(
                    "rear_swing_tracking_error_proposal_start_frame",
                    0,
                )
            ),
            "rear_swing_load_transfer_proposal_ref_ids": list(
                mppi_yaml.get(
                    "rear_swing_load_transfer_proposal_ref_ids",
                    [],
                )
            ),
            "rear_swing_load_transfer_proposal_scales": list(
                mppi_yaml.get(
                    "rear_swing_load_transfer_proposal_scales",
                    [],
                )
            ),
            "rear_swing_load_transfer_proposal_gain_by_wheel_leg_rad": [
                list(row)
                for row in mppi_yaml.get(
                    "rear_swing_load_transfer_proposal_gain_by_wheel_leg_rad",
                    [[0.0] * 12, [0.0] * 12],
                )
            ],
            "rear_swing_load_transfer_proposal_start_frame": int(
                mppi_yaml.get(
                    "rear_swing_load_transfer_proposal_start_frame",
                    0,
                )
            ),
            "rear_swing_load_transfer_proposal_start_frame_by_wheel": list(
                mppi_yaml.get(
                    "rear_swing_load_transfer_proposal_start_frame_by_wheel"
                )
                or [
                    int(
                        mppi_yaml.get(
                            "rear_swing_load_transfer_proposal_start_frame",
                            0,
                        )
                    )
                ]
                * 2
            ),
            "rear_swing_load_transfer_proposal_gate_mode": str(
                mppi_yaml.get(
                    "rear_swing_load_transfer_proposal_gate_mode",
                    "swing_schedule",
                )
            ),
            "rear_swing_load_transfer_proposal_imbalance_threshold_n": float(
                mppi_yaml.get(
                    "rear_swing_load_transfer_proposal_imbalance_threshold_n",
                    0.0,
                )
            ),
            "front_support_proposal_ref_ids": list(
                mppi_yaml.get(
                    "front_support_proposal_ref_ids",
                    [],
                )
            ),
            "front_support_proposal_scales": list(
                mppi_yaml.get(
                    "front_support_proposal_scales",
                    [],
                )
            ),
            "front_support_proposal_gain_leg_rad": list(
                mppi_yaml.get(
                    "front_support_proposal_gain_leg_rad",
                    [0.0] * 12,
                )
            ),
            "front_support_proposal_start_frame": int(
                mppi_yaml.get(
                    "front_support_proposal_start_frame",
                    0,
                )
            ),
            "combine_rear_swing_front_support_proposals": bool(
                mppi_yaml.get(
                    "combine_rear_swing_front_support_proposals",
                    False,
                )
            ),
            "combine_rear_swing_load_transfer_front_support_proposals": bool(
                mppi_yaml.get(
                    "combine_rear_swing_load_transfer_front_support_proposals",
                    False,
                )
            ),
            "combine_rear_swing_reference_load_transfer_front_support_proposals": bool(
                mppi_yaml.get(
                    "combine_rear_swing_reference_load_transfer_front_"
                    "support_proposals",
                    False,
                )
            ),
            "include_rear_support_reference_in_coordinated_proposals": bool(
                mppi_yaml.get(
                    "include_rear_support_reference_in_coordinated_proposals",
                    False,
                )
            ),
            "rear_support_reference_proposal_start_frame": int(
                mppi_yaml.get(
                    "rear_support_reference_proposal_start_frame",
                    0,
                )
            ),
            "output_front_force_feedback_target_n": float(
                mppi_yaml.get("output_front_force_feedback_target_n", 0.0)
            ),
            "output_front_force_feedback_min_contact_n": float(
                mppi_yaml.get(
                    "output_front_force_feedback_min_contact_n",
                    0.0,
                )
            ),
            "output_front_force_feedback_lookahead_steps": int(
                mppi_yaml.get(
                    "output_front_force_feedback_lookahead_steps",
                    mppi_yaml.get(
                        "reference_action_lookahead_steps",
                        1,
                    ),
                )
            ),
            "output_front_force_feedback_gain_leg": list(
                mppi_yaml.get(
                    "output_front_force_feedback_gain_leg",
                    [0.0] * 12,
                )
            ),
            "output_rear_swing_force_feedback_target_n": float(
                mppi_yaml.get(
                    "output_rear_swing_force_feedback_target_n",
                    0.0,
                )
            ),
            "output_rear_swing_force_feedback_scale_n": float(
                mppi_yaml.get(
                    "output_rear_swing_force_feedback_scale_n",
                    1.0,
                )
            ),
            "output_rear_swing_force_feedback_lookahead_steps": int(
                mppi_yaml.get(
                    "output_rear_swing_force_feedback_lookahead_steps",
                    mppi_yaml.get(
                        "reference_action_lookahead_steps",
                        1,
                    ),
                )
            ),
            "output_rear_swing_force_feedback_start_frame": int(
                mppi_yaml.get(
                    "output_rear_swing_force_feedback_start_frame",
                    0,
                )
            ),
            "output_rear_swing_force_feedback_gain_leg": list(
                mppi_yaml.get(
                    "output_rear_swing_force_feedback_gain_leg",
                    [0.0] * 12,
                )
            ),
            "output_rear_swing_height_feedback_ref_ids": list(
                mppi_yaml.get(
                    "output_rear_swing_height_feedback_ref_ids",
                    [],
                )
            ),
            "output_rear_swing_height_feedback_gain": float(
                mppi_yaml.get(
                    "output_rear_swing_height_feedback_gain",
                    0.0,
                )
            ),
            "output_rear_swing_height_feedback_max_abs_rad": float(
                mppi_yaml.get(
                    "output_rear_swing_height_feedback_max_abs_rad",
                    0.0,
                )
            ),
            "output_rear_swing_height_feedback_lookahead_steps": int(
                mppi_yaml.get(
                    "output_rear_swing_height_feedback_lookahead_steps",
                    mppi_yaml.get(
                        "reference_action_lookahead_steps",
                        1,
                    ),
                )
            ),
            "output_rear_swing_height_feedback_start_frame": int(
                mppi_yaml.get(
                    "output_rear_swing_height_feedback_start_frame",
                    0,
                )
            ),
            "output_rear_support_tracking_feedback_ref_ids": list(
                mppi_yaml.get(
                    "output_rear_support_tracking_feedback_ref_ids",
                    [],
                )
            ),
            "output_rear_support_tracking_feedback_gain": float(
                mppi_yaml.get(
                    "output_rear_support_tracking_feedback_gain",
                    0.0,
                )
            ),
            "output_rear_support_tracking_feedback_max_abs_rad": float(
                mppi_yaml.get(
                    "output_rear_support_tracking_feedback_max_abs_rad",
                    0.0,
                )
            ),
            "output_rear_support_tracking_feedback_lookahead_steps": int(
                mppi_yaml.get(
                    "output_rear_support_tracking_feedback_lookahead_steps",
                    mppi_yaml.get(
                        "reference_action_lookahead_steps",
                        1,
                    ),
                )
            ),
            "output_rear_support_tracking_feedback_start_frame": int(
                mppi_yaml.get(
                    "output_rear_support_tracking_feedback_start_frame",
                    0,
                )
            ),
            "output_pitch_feedback_ref_ids": list(
                mppi_yaml.get("output_pitch_feedback_ref_ids", [])
            ),
            "output_pitch_feedback_gain_leg": list(
                mppi_yaml.get(
                    "output_pitch_feedback_gain_leg",
                    [0.0] * 12,
                )
            ),
            "output_pitch_feedback_axis": str(
                mppi_yaml.get("output_pitch_feedback_axis", "y")
            ),
            "output_pitch_feedback_start_frame": int(
                mppi_yaml.get(
                    "output_pitch_feedback_start_frame",
                    0,
                )
            ),
            "output_pitch_feedback_max_abs_rad": float(
                mppi_yaml.get(
                    "output_pitch_feedback_max_abs_rad",
                    0.0,
                )
            ),
            "output_joint_position_offset_leg": list(
                mppi_yaml.get(
                    "output_joint_position_offset_leg",
                    [0.0] * 12,
                )
            ),
            "wheel_action_mode": action_contract.wheel_action_mode.value,
            "implementation_sha256": expert_implementation_sha256,
        }
        expert_config_hash = canonical_hash(expert_runtime_config)
        expert_config_record = {
            **mppi_yaml,
            "horizon": mppi_config.horizon,
            "samples": mppi_config.samples,
            "optimization_iterations": mppi_config.iterations,
            "temperature": mppi_config.temperature,
            "selection_mode": mppi_config.selection_mode,
            "seed": mppi_config.seed,
            "reference_action_lookahead_steps": mppi_config.reference_action_lookahead_steps,
            "noise_scale": noise_scale,
            "noise_std_leg": noise_std.detach().cpu().tolist(),
            "episode_rng_seed_semantics": "mppi_config_seed_plus_episode_seed",
            "runtime_config": expert_runtime_config,
            "runtime_config_hash": expert_config_hash,
        }
    elif expert_backend == "reference_wbc":
        expert = ReferenceWBCExpert(action_adapter)
        expert_config_record = {"backend": "reference_wbc"}
        expert_config_hash = canonical_hash(expert_config_record)
    else:
        expert = DisabledLabelExpert()
        expert_config_record = {
            "backend": "disabled",
            "scope": "student_only_closed_loop_evaluation",
            "valid_label_provider": False,
        }
        expert_config_hash = canonical_hash(expert_config_record)
    student_checkpoint_path = (
        Path(args.student_checkpoint).expanduser().resolve()
        if args.student_checkpoint is not None
        else None
    )
    student_checkpoint_record = (
        {
            "path": str(student_checkpoint_path),
            "sha256": sha256_file(student_checkpoint_path),
        }
        if student_checkpoint_path is not None
        else None
    )
    student_policy = _student_callable(
        student_checkpoint_path,
        args.device,
        action_contract,
    )

    provenance = _provenance()
    records = []
    episode_metrics = []
    existing_by_episode_id = {
        record["episode_id"]: record
        for record in load_manifest(args.dataset)
    } if bool(getattr(args, "resume", False)) and Path(args.dataset).exists() else {}
    try:
        for episode in range(args.episodes):
            seed = args.seed + episode
            ref_id = (args.ref_id + episode) % len(references) if args.rotate_references else args.ref_id
            metadata = {
                "seed": seed,
                "ref_id": ref_id,
                "scenario": args.scenario,
                "dagger_round": int(getattr(args, "dagger_round", 0)),
                "scenario_profile": scenario_record,
                "scenario_config_hash": scenario_config_hash,
                "observation_noise_std": observation_noise_std,
                "git_commit": provenance["commit"],
                "git_dirty": provenance["dirty"],
                "git_dirty_status": provenance["dirty_status"],
                "provenance": provenance,
                "expert_backend": expert_backend,
                "expert_config": expert_config_record,
                "expert_config_hash": expert_config_hash,
                "expert_implementation_sha256": expert_implementation_sha256,
                "environment_backend": "isaac_lab",
                "task_id": args.task,
                "robot_lab_import": str(loaded_robot_lab),
                "isaaclab_import": str(loaded_isaaclab),
                "wheel_action_mode": action_contract.wheel_action_mode.value,
                "robot_urdf_hash": contract["assets"]["robot_urdf"]["sha256"],
                "trunk_asset_hash": contract["assets"]["trunk_usd"]["sha256"],
                "trajectory_hash": references[ref_id].sha256,
                "joint_order": contract["joint_order_policy"],
                "action_scale": contract["action"]["scale"],
                "q_reset_ref0": contract["reset"]["q_reset_ref0"],
                "q_action_offset_runtime": contract["action"]["q_action_offset_runtime"],
                "observation_schema": contract["observation"],
                "pose_layout": contract["pose_layout"],
                "control_frequency": contract["timebase"]["control_hz"],
                "direction_speed": references[ref_id].target_vy,
                "video_recording_enabled": video_dir is not None and episode == 0,
                "student_checkpoint": student_checkpoint_record,
            }
            if expert_backend == "mppi":
                metadata["expert_episode_rng_seed"] = mppi_config.seed + seed
            episode_id = f"{args.run_name}_ref{ref_id}_seed{seed}"
            split = args.split if args.split != "auto" else None
            existing_record = existing_by_episode_id.get(episode_id)
            if existing_record is not None:
                shard = read_episode_shard(Path(args.dataset) / existing_record["path"])
                expected_metadata = {
                    "seed": seed,
                    "ref_id": ref_id,
                    "scenario": args.scenario,
                    "dagger_round": int(getattr(args, "dagger_round", 0)),
                    "scenario_config_hash": scenario_config_hash,
                    "expert_backend": expert_backend,
                    "expert_config_hash": metadata["expert_config_hash"],
                    "wheel_action_mode": action_contract.wheel_action_mode.value,
                    "student_checkpoint": student_checkpoint_record,
                }
                mismatches = {
                    key: {"expected": value, "actual": shard.metadata.get(key)}
                    for key, value in expected_metadata.items()
                    if shard.metadata.get(key) != value
                }
                if split is not None and existing_record["split"] != split:
                    mismatches["split"] = {
                        "expected": split,
                        "actual": existing_record["split"],
                    }
                if int(shard.arrays["step_id"].shape[0]) != int(args.steps):
                    mismatches["steps"] = {
                        "expected": int(args.steps),
                        "actual": int(shard.arrays["step_id"].shape[0]),
                    }
                if mismatches:
                    raise ValueError(
                        f"Cannot resume episode {episode_id}: metadata differs: {mismatches}"
                    )
                record = existing_record
            else:
                shard = collect_episode(
                    adapter,
                    expert,
                    SafetyShield(action_contract),
                    CollectorConfig(
                        seed=seed,
                        ref_id=ref_id,
                        max_steps=args.steps,
                        beta=args.beta,
                        observation_noise_std=observation_noise_std,
                        scenario=args.scenario,
                    ),
                    metadata,
                    student_policy=student_policy,
                )
                record = write_episode_shard(args.dataset, episode_id, shard, split=split)
            records.append(record)
            mppi_mask = shard.arrays["label_source"] == 3
            if np.any(mppi_mask):
                component_order = shard.metadata.get(
                    "mppi_cost_component_order",
                    MPPI_COST_COMPONENT_NAMES,
                )
                cost_component_mean = {
                    name: float(np.mean(shard.arrays["mppi_cost_components"][mppi_mask, index]))
                    for index, name in enumerate(component_order)
                }
                mppi_minimum_cost_mean = float(
                    np.mean(shard.arrays["mppi_minimum_total_cost"][mppi_mask])
                )
                mppi_ess_mean = float(
                    np.mean(shard.arrays["mppi_effective_sample_size"][mppi_mask])
                )
                mppi_rollout_termination_rate = float(
                    np.mean(shard.arrays["mppi_rollout_termination_rate"][mppi_mask])
                )
            else:
                cost_component_mean = {}
                mppi_minimum_cost_mean = None
                mppi_ess_mean = None
                mppi_rollout_termination_rate = None
            episode_metrics.append(
                {
                    "episode_id": episode_id,
                    "success": record["success"],
                    "steps": record["steps"],
                    "teacher_valid_rate": record["teacher_valid_rate"],
                    "shield_intervention_rate": record["shield_intervention_rate"],
                    "max_abs_wheel_action": float(np.max(np.abs(shard.arrays["executed_action16"][:, 12:]))),
                    "mean_solve_ms": (
                        float(np.mean(shard.arrays["solve_ms"][np.isfinite(shard.arrays["solve_ms"])]))
                        if np.any(np.isfinite(shard.arrays["solve_ms"]))
                        else 0.0
                    ),
                    "termination_reason": int(shard.arrays["termination_reason"][-1]),
                    "mppi_minimum_cost_mean": mppi_minimum_cost_mean,
                    "mppi_effective_sample_size_mean": mppi_ess_mean,
                    "mppi_rollout_termination_rate_mean": mppi_rollout_termination_rate,
                    "mppi_cost_component_mean": cost_component_mean,
                }
            )
            print(
                json.dumps(
                    {
                        "progress": {
                            "completed": episode + 1,
                            "total": args.episodes,
                            "episode_id": episode_id,
                            "ref_id": ref_id,
                            "seed": seed,
                            "success": record["success"],
                            "steps": record["steps"],
                            "resumed": existing_record is not None,
                        }
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        if provider is not None and hasattr(provider, "close"):
            provider.close()
        adapter.close()

    report = {
        "schema_version": "pcbc-isaac-collection-report-v1",
        "task_id": args.task,
        "expert_backend": expert_backend,
        "expert_config": expert_config_record,
        "expert_config_hash": expert_config_hash,
        "expert_implementation_sha256": expert_implementation_sha256,
        "scenario_profile": scenario_record,
        "scenario_config_hash": scenario_config_hash,
        "observation_noise_std": observation_noise_std,
        "student_checkpoint": student_checkpoint_record,
        "standalone_root": str(ROOT.resolve()),
        "robot_lab_import": str(loaded_robot_lab),
        "isaaclab_import": str(loaded_isaaclab),
        "provenance": provenance,
        "dataset": str(Path(args.dataset).resolve()),
        "video_dir": str(video_dir) if video_dir is not None else None,
        "episodes": episode_metrics,
        "summary": {
            "episodes": len(episode_metrics),
            "success_rate": float(np.mean([item["success"] for item in episode_metrics])),
            "teacher_valid_rate": float(np.mean([item["teacher_valid_rate"] for item in episode_metrics])),
            "shield_intervention_rate": float(
                np.mean([item["shield_intervention_rate"] for item in episode_metrics])
            ),
            "wheel_action_max_abs": float(max(item["max_abs_wheel_action"] for item in episode_metrics)),
            "mean_solve_ms": float(np.mean([item["mean_solve_ms"] for item in episode_metrics])),
        },
    }
    write_json(report_path, report)
    print(json.dumps(report["summary"], sort_keys=True))
    return report
