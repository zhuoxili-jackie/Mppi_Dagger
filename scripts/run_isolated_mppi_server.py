#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import traceback
from multiprocessing.connection import Listener
from pathlib import Path
from typing import Any

import torch

from _bootstrap import ROOT, load_contract, write_json

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description=(
        "Run MPPI rollouts in a persistent Isaac process isolated from the "
        "single-environment collector."
    )
)
parser.add_argument(
    "--task",
    default="RobotLab-Isaac-BeyondMimic-Flat-pcbC-v1-lateral-bipedal-stand-v0",
)
parser.add_argument(
    "--mppi-config",
    type=Path,
    required=True,
)
parser.add_argument("--reference-config", type=str, default=None)
parser.add_argument("--scenario", default="nominal")
parser.add_argument("--seed", type=int, default=5208)
parser.add_argument("--socket", type=Path, required=True)
parser.add_argument("--disable-fabric", action="store_true")
parser.add_argument(
    "--report",
    type=Path,
    default=ROOT
    / "reports/low_load_lateral/train_001/diagnostics/isolated_mppi_server.json",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def _server_identity(
    mppi_config_path: Path,
    reference_config: str,
    mppi_yaml: dict[str, Any],
) -> dict[str, Any]:
    from lateral_mppi_dagger.config import sha256_file

    return {
        "schema_version": "pcbc-isolated-mppi-server-v1",
        "mppi_config_sha256": sha256_file(mppi_config_path),
        "reference_config": reference_config,
        "samples": int(mppi_yaml["samples"]),
        "horizon": int(mppi_yaml["horizon"]),
        "optimization_iterations": int(
            mppi_yaml["optimization_iterations"]
        ),
    }


def main() -> dict[str, Any]:
    import gymnasium as gym
    import robot_lab.tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    from lateral_mppi_dagger.config import load_yaml
    from lateral_mppi_dagger.contract.action16 import (
        Action16Adapter,
        ActionContract,
    )
    from lateral_mppi_dagger.env.isaac_adapter import IsaacLateralAdapter
    from lateral_mppi_dagger.env.isaac_mppi_rollout import (
        IsaacRolloutCostWeights,
        IsaacRolloutLoadLimits,
        IsaacWholeBodyMPPIProvider,
    )
    from lateral_mppi_dagger.env.isolated_mppi import (
        snapshot_from_payload,
    )
    from lateral_mppi_dagger.env.scenarios import (
        configure_env_for_scenario,
        load_scenario_profile,
    )
    from lateral_mppi_dagger.expert.base import ExpertRequest
    from lateral_mppi_dagger.expert.mppi_expert import MPPIConfig
    from lateral_mppi_dagger.reference.action_reference import (
        load_nominal_action_references,
    )
    from lateral_mppi_dagger.reference.loader import ReferenceSet

    socket_path = args_cli.socket.expanduser().resolve()
    if socket_path == ROOT or ROOT not in socket_path.parents:
        raise ValueError(
            "The isolated MPPI socket must be inside the standalone root."
        )
    if socket_path.exists():
        raise FileExistsError(
            f"Refusing to replace existing socket path {socket_path}."
        )
    config_path = args_cli.mppi_config.expanduser().resolve()
    mppi_yaml = load_yaml(config_path)
    reference_config = (
        args_cli.reference_config
        or mppi_yaml.get(
            "reference_config",
            "configs/reference_708.yaml",
        )
    )
    identity = _server_identity(
        config_path,
        reference_config,
        mppi_yaml,
    )
    references = ReferenceSet.from_config(reference_config)
    contract_dict = load_contract()
    action_contract = ActionContract.from_dict(contract_dict)
    action_adapter = Action16Adapter(action_contract)
    samples = int(mppi_yaml["samples"])
    scenario = load_scenario_profile(args_cli.scenario)
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=samples,
        use_fabric=not args_cli.disable_fabric,
    )
    configure_env_for_scenario(
        env_cfg,
        scenario,
        num_envs=samples,
    )
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device
    env = gym.make(args_cli.task, cfg=env_cfg)
    adapter = IsaacLateralAdapter(
        env,
        references,
        contract_dict,
        scenario_profile=scenario,
    )
    (
        nominal_q_des,
        nominal_raw,
        nominal_overrides,
        nominal_record,
    ) = load_nominal_action_references(
        mppi_yaml.get("nominal_action_reference")
    )
    mppi_config = MPPIConfig(
        horizon=int(mppi_yaml["horizon"]),
        samples=samples,
        iterations=int(mppi_yaml["optimization_iterations"]),
        temperature=float(mppi_yaml["temperature"]),
        temporal_smoothing=float(mppi_yaml["temporal_smoothing"]),
        warm_start=bool(mppi_yaml["warm_start"]),
        selection_mode=str(mppi_yaml.get("selection_mode", "weighted")),
        reference_action_lookahead_steps=int(
            mppi_yaml.get("reference_action_lookahead_steps", 1)
        ),
        seed=int(mppi_yaml["seed"]),
    )
    provider = IsaacWholeBodyMPPIProvider(
        adapter,
        references,
        action_adapter,
        mppi_config,
        torch.as_tensor(
            mppi_yaml["noise_std_leg"],
            dtype=torch.float32,
            device=args_cli.device,
        ),
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
            float(mppi_yaml["physical_target_rate_limit_rad_s"])
            if mppi_yaml.get("physical_target_rate_limit_rad_s")
            is not None
            else None
        ),
        nominal_action_reference_q_des_by_ref=nominal_q_des,
        nominal_action_reference_raw_by_ref=nominal_raw,
        nominal_action_reference_overrides_by_ref=nominal_overrides,
        nominal_joint_position_bias_leg=mppi_yaml.get(
            "nominal_joint_position_bias_leg"
        ),
        nominal_joint_position_bias_start_frame=int(
            mppi_yaml.get("nominal_joint_position_bias_start_frame", 0)
        ),
        nominal_joint_position_bias_ramp_frames=int(
            mppi_yaml.get("nominal_joint_position_bias_ramp_frames", 0)
        ),
        nominal_front_force_feedback_target_n=float(
            mppi_yaml.get("nominal_front_force_feedback_target_n", 0.0)
        ),
        nominal_front_force_feedback_gain_leg=mppi_yaml.get(
            "nominal_front_force_feedback_gain_leg"
        ),
        output_front_force_feedback_target_n=float(
            mppi_yaml.get("output_front_force_feedback_target_n", 0.0)
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
        output_pitch_feedback_ref_ids=mppi_yaml.get(
            "output_pitch_feedback_ref_ids"
        ),
        output_pitch_feedback_gain_leg=mppi_yaml.get(
            "output_pitch_feedback_gain_leg"
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

    listener = None
    connection = None
    request_count = 0
    reset_count = 0
    solver_schedule_phase_counts: dict[str, int] = {}
    solver_schedule_reset_events: list[dict[str, int]] = []
    status = "starting"
    error_record = None
    try:
        listener = Listener(str(socket_path), family="AF_UNIX")
        status = "ready"
        print(
            {
                "status": status,
                "socket": str(socket_path),
                "server": identity,
            },
            flush=True,
        )
        connection = listener.accept()
        while True:
            try:
                message = connection.recv()
            except EOFError:
                status = "client_disconnected"
                break
            if not isinstance(message, dict):
                raise TypeError("Server message must be a mapping.")
            operation = message.get("op")
            try:
                if operation == "hello":
                    connection.send(
                        {
                            "ok": True,
                            "server": identity,
                        }
                    )
                elif operation == "reset":
                    snapshot = snapshot_from_payload(
                        message["snapshot"],
                        adapter.base.device,
                    )
                    provider.rollout.restore(snapshot)
                    provider.reset(message.get("episode_metadata"))
                    reset_count += 1
                    connection.send({"ok": True})
                elif operation == "act":
                    snapshot = snapshot_from_payload(
                        message["snapshot"],
                        adapter.base.device,
                    )
                    request = message.get("request")
                    if not isinstance(request, ExpertRequest):
                        raise TypeError(
                            "act request must contain an ExpertRequest."
                        )
                    provider.rollout.restore(snapshot)
                    reply = provider(request)
                    request_count += 1
                    diagnostics = dict(provider.last_diagnostics)
                    schedule_phase = diagnostics.get(
                        "solver_schedule_phase"
                    )
                    if schedule_phase is not None:
                        phase_key = str(int(schedule_phase))
                        solver_schedule_phase_counts[phase_key] = (
                            solver_schedule_phase_counts.get(phase_key, 0)
                            + 1
                        )
                    if diagnostics.get(
                        "solver_schedule_reset_warm_start",
                        False,
                    ):
                        solver_schedule_reset_events.append(
                            {
                                "request_count": request_count,
                                "ref_id": int(request.ref_id),
                                "ref_frame": int(request.ref_frame),
                                "solver_schedule_phase": int(
                                    schedule_phase
                                ),
                            }
                        )
                    connection.send(
                        {
                            "ok": True,
                            "reply": reply,
                            "last_diagnostics": diagnostics,
                        }
                    )
                elif operation == "close":
                    status = "closed_by_client"
                    connection.send({"ok": True})
                    break
                else:
                    raise ValueError(
                        f"Unsupported isolated MPPI operation {operation!r}."
                    )
            except BaseException as exc:
                error_record = {
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                    "operation": operation,
                }
                connection.send({"ok": False, **error_record})
                status = "request_failed"
                break
    finally:
        if connection is not None:
            connection.close()
        if listener is not None:
            listener.close()
        adapter.close()
        if socket_path.exists():
            socket_path.unlink()

    report = {
        "schema_version": "pcbc-isolated-mppi-server-report-v1",
        "status": status,
        "server": identity,
        "socket": str(socket_path),
        "request_count": request_count,
        "reset_count": reset_count,
        "solver_schedule_phase_counts": (
            solver_schedule_phase_counts
        ),
        "solver_schedule_reset_events": (
            solver_schedule_reset_events
        ),
        "nominal_action_reference": nominal_record,
        "error": error_record,
    }
    write_json(args_cli.report, report)
    print(report, flush=True)
    if error_record is not None:
        raise RuntimeError(
            "Isolated MPPI server stopped after a request failure."
        )
    return report


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        write_json(
            str(args_cli.report) + ".failure.json",
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
