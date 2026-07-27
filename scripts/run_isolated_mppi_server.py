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
            int(mppi_yaml["rear_swing_action_residual_lead_steps"])
            if mppi_yaml.get(
                "rear_swing_action_residual_lead_steps"
            )
            is not None
            else None
        ),
        rear_swing_tracking_error_proposal_scales=mppi_yaml.get(
            "rear_swing_tracking_error_proposal_scales"
        ),
        rear_swing_tracking_error_proposal_joint_mask_leg=mppi_yaml.get(
            "rear_swing_tracking_error_proposal_joint_mask_leg"
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
                "combine_rear_swing_reference_load_transfer_front_support_"
                "proposals",
                False,
            )
        ),
        include_rear_support_reference_in_coordinated_proposals=mppi_yaml.get(
            "include_rear_support_reference_in_coordinated_proposals",
            False,
        ),
        rear_support_reference_proposal_start_frame=int(
            mppi_yaml.get(
                "rear_support_reference_proposal_start_frame",
                0,
            )
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
        output_rear_swing_force_feedback_lookahead_steps=mppi_yaml.get(
            "output_rear_swing_force_feedback_lookahead_steps"
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
            mppi_yaml.get("output_pitch_feedback_start_frame", 0)
        ),
        output_pitch_feedback_max_abs_rad=float(
            mppi_yaml.get(
                "output_pitch_feedback_max_abs_rad",
                0.0,
            )
        ),
        output_contact_orientation_feedback_ref_ids=mppi_yaml.get(
            "output_contact_orientation_feedback_ref_ids"
        ),
        output_contact_orientation_feedback_gain_xyz=mppi_yaml.get(
            "output_contact_orientation_feedback_gain_xyz"
        ),
        output_contact_orientation_feedback_start_frame=int(
            mppi_yaml.get(
                "output_contact_orientation_feedback_start_frame",
                0,
            )
        ),
        output_contact_orientation_feedback_max_endpoint_delta_m=float(
            mppi_yaml.get(
                "output_contact_orientation_feedback_max_endpoint_delta_m",
                0.0,
            )
        ),
        output_contact_orientation_feedback_max_abs_rad=float(
            mppi_yaml.get(
                "output_contact_orientation_feedback_max_abs_rad",
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
    selected_best_sample_source_counts: dict[str, int] = {}
    structured_proposal_selection_events: list[dict[str, object]] = []
    structured_proposal_cost_events: list[dict[str, object]] = []
    rear_swing_height_feedback_events: list[dict[str, object]] = []
    rear_swing_height_feedback_stuck_counts = [0, 0]
    rear_swing_height_feedback_nonzero_applied_count = 0
    rear_swing_height_feedback_max_deficit_m = 0.0
    rear_swing_height_feedback_max_requested_abs_rad = 0.0
    rear_swing_height_feedback_max_applied_abs_rad = 0.0
    rear_support_tracking_feedback_events: list[dict[str, object]] = []
    rear_support_tracking_feedback_missing_counts = [0, 0]
    rear_support_tracking_feedback_nonzero_applied_count = 0
    rear_support_tracking_feedback_max_requested_abs_rad = 0.0
    rear_support_tracking_feedback_max_applied_abs_rad = 0.0
    orientation_feedback_events: list[dict[str, object]] = []
    orientation_feedback_nonzero_applied_count = 0
    orientation_feedback_max_error_abs_rad = 0.0
    orientation_feedback_max_requested_abs_rad = 0.0
    orientation_feedback_max_applied_abs_rad = 0.0
    contact_orientation_feedback_events: list[dict[str, object]] = []
    contact_orientation_feedback_nonzero_applied_count = 0
    contact_orientation_feedback_max_error_abs_rad = 0.0
    contact_orientation_feedback_max_endpoint_delta_m = 0.0
    contact_orientation_feedback_max_requested_abs_rad = 0.0
    contact_orientation_feedback_max_applied_abs_rad = 0.0
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
                    selected_source = str(
                        diagnostics.get(
                            "selected_best_sample_source",
                            "missing",
                        )
                    )
                    selected_best_sample_source_counts[selected_source] = (
                        selected_best_sample_source_counts.get(
                            selected_source,
                            0,
                        )
                        + 1
                    )
                    if selected_source.startswith(
                        "structured_proposal_"
                    ):
                        proposal_index = int(
                            selected_source.rsplit("_", 1)[1]
                        )
                        proposal_descriptors = diagnostics.get(
                            "structured_proposal_descriptors",
                            [],
                        )
                        proposal_descriptor = (
                            dict(proposal_descriptors[proposal_index])
                            if proposal_index
                            < len(proposal_descriptors)
                            else {}
                        )
                        structured_proposal_selection_events.append(
                            {
                                "request_count": request_count,
                                "ref_id": int(request.ref_id),
                                "ref_frame": int(request.ref_frame),
                                "proposal_index": proposal_index,
                                "proposal_kind": (
                                    proposal_descriptor.get("kind")
                                ),
                                "proposal_scale": (
                                    proposal_descriptor.get("scale")
                                ),
                                "selected_best_sample_iteration": int(
                                    diagnostics.get(
                                        "selected_best_sample_iteration",
                                        -1,
                                    )
                                ),
                            }
                        )
                    proposal_cost_iterations = diagnostics.get(
                        "structured_proposal_cost_iterations",
                        [],
                    )
                    if proposal_cost_iterations:
                        proposal_descriptors = list(
                            diagnostics.get(
                                "structured_proposal_descriptors",
                                [],
                            )
                        )
                        proposal_diagnostics = diagnostics.get(
                            "rear_swing_reference_proposals",
                            {},
                        )
                        structured_proposal_cost_events.append(
                            {
                                "request_count": request_count,
                                "ref_id": int(request.ref_id),
                                "ref_frame": int(request.ref_frame),
                                "proposal_descriptors": (
                                    proposal_descriptors
                                ),
                                "proposal_scales": [
                                    descriptor.get("scale")
                                    for descriptor in proposal_descriptors
                                ],
                                "lead_steps": int(
                                    proposal_diagnostics.get(
                                        "lead_steps",
                                        0,
                                    )
                                ),
                                "iterations": proposal_cost_iterations,
                            }
                        )
                    rear_swing_height_feedback = diagnostics.get(
                        "output_rear_swing_height_feedback",
                        {},
                    )
                    stuck_rear_swing = list(
                        rear_swing_height_feedback.get(
                            "stuck_rear_swing",
                            [False, False],
                        )
                    )
                    if (
                        rear_swing_height_feedback.get("enabled", False)
                        and any(bool(value) for value in stuck_rear_swing)
                    ):
                        height_deficit = [
                            float(value)
                            for value in rear_swing_height_feedback.get(
                                "height_deficit_m",
                                [0.0, 0.0],
                            )
                        ]
                        requested_correction = [
                            float(value)
                            for value in rear_swing_height_feedback.get(
                                "requested_correction_rad",
                                [0.0] * 12,
                            )
                        ]
                        applied_correction = [
                            float(value)
                            for value in rear_swing_height_feedback.get(
                                "applied_correction_rad",
                                [0.0] * 12,
                            )
                        ]
                        for rear_index in range(2):
                            rear_swing_height_feedback_stuck_counts[
                                rear_index
                            ] += int(bool(stuck_rear_swing[rear_index]))
                        if any(
                            abs(value) > 0.0
                            for value in applied_correction
                        ):
                            rear_swing_height_feedback_nonzero_applied_count += 1
                        rear_swing_height_feedback_max_deficit_m = max(
                            rear_swing_height_feedback_max_deficit_m,
                            max(height_deficit),
                        )
                        rear_swing_height_feedback_max_requested_abs_rad = max(
                            rear_swing_height_feedback_max_requested_abs_rad,
                            max(
                                abs(value)
                                for value in requested_correction
                            ),
                        )
                        rear_swing_height_feedback_max_applied_abs_rad = max(
                            rear_swing_height_feedback_max_applied_abs_rad,
                            max(
                                abs(value)
                                for value in applied_correction
                            ),
                        )
                        rear_swing_height_feedback_events.append(
                            {
                                "request_count": request_count,
                                "ref_id": int(request.ref_id),
                                "ref_frame": int(request.ref_frame),
                                "schedule_frame": int(
                                    rear_swing_height_feedback[
                                        "schedule_frame"
                                    ]
                                ),
                                "preview_start_frame": int(
                                    rear_swing_height_feedback[
                                        "preview_start_frame"
                                    ]
                                ),
                                "target_frame_by_rear": [
                                    (
                                        None
                                        if value is None
                                        else int(value)
                                    )
                                    for value
                                    in rear_swing_height_feedback.get(
                                        "target_frame_by_rear",
                                        [None, None],
                                    )
                                ],
                                "stuck_rear_swing": [
                                    bool(value)
                                    for value in stuck_rear_swing
                                ],
                                "rear_normal_n": [
                                    float(value)
                                    for value
                                    in rear_swing_height_feedback.get(
                                        "rear_normal_n",
                                        [0.0, 0.0],
                                    )
                                ],
                                "height_deficit_m": height_deficit,
                                "jacobian_joint_delta_rad": [
                                    float(value)
                                    for value
                                    in rear_swing_height_feedback.get(
                                        "jacobian_joint_delta_rad",
                                        [0.0] * 12,
                                    )
                                ],
                                "predicted_cartesian_delta_m_by_rear": [
                                    [float(value) for value in vector]
                                    for vector
                                    in rear_swing_height_feedback.get(
                                        "predicted_cartesian_delta_m_by_rear",
                                        [
                                            [0.0, 0.0, 0.0],
                                            [0.0, 0.0, 0.0],
                                        ],
                                    )
                                ],
                                "requested_correction_rad": (
                                    requested_correction
                                ),
                                "applied_correction_rad": (
                                    applied_correction
                                ),
                            }
                        )
                    rear_support_feedback = diagnostics.get(
                        "output_rear_support_tracking_feedback",
                        {},
                    )
                    missing_rear_support = list(
                        rear_support_feedback.get(
                            "missing_rear_support",
                            [False, False],
                        )
                    )
                    if (
                        rear_support_feedback.get("enabled", False)
                        and any(bool(value) for value in missing_rear_support)
                    ):
                        requested_correction = [
                            float(value)
                            for value in rear_support_feedback.get(
                                "requested_correction_rad",
                                [0.0] * 12,
                            )
                        ]
                        applied_correction = [
                            float(value)
                            for value in rear_support_feedback.get(
                                "applied_correction_rad",
                                [0.0] * 12,
                            )
                        ]
                        for rear_index in range(2):
                            rear_support_tracking_feedback_missing_counts[
                                rear_index
                            ] += int(
                                bool(
                                    missing_rear_support[rear_index]
                                )
                            )
                        if any(
                            abs(value) > 0.0
                            for value in applied_correction
                        ):
                            rear_support_tracking_feedback_nonzero_applied_count += 1
                        rear_support_tracking_feedback_max_requested_abs_rad = max(
                            rear_support_tracking_feedback_max_requested_abs_rad,
                            max(
                                abs(value)
                                for value in requested_correction
                            ),
                        )
                        rear_support_tracking_feedback_max_applied_abs_rad = max(
                            rear_support_tracking_feedback_max_applied_abs_rad,
                            max(
                                abs(value)
                                for value in applied_correction
                            ),
                        )
                        rear_support_tracking_feedback_events.append(
                            {
                                "request_count": request_count,
                                "ref_id": int(request.ref_id),
                                "ref_frame": int(request.ref_frame),
                                "schedule_frame": int(
                                    rear_support_feedback[
                                        "schedule_frame"
                                    ]
                                ),
                                "missing_rear_support": [
                                    bool(value)
                                    for value in missing_rear_support
                                ],
                                "rear_normal_n": [
                                    float(value)
                                    for value in rear_support_feedback.get(
                                        "rear_normal_n",
                                        [0.0, 0.0],
                                    )
                                ],
                                "requested_correction_rad": (
                                    requested_correction
                                ),
                                "applied_correction_rad": (
                                    applied_correction
                                ),
                            }
                        )
                    contact_orientation_feedback = diagnostics.get(
                        "output_contact_orientation_feedback",
                        {},
                    )
                    if contact_orientation_feedback.get(
                        "enabled",
                        False,
                    ):
                        orientation_error = [
                            float(value)
                            for value
                            in contact_orientation_feedback.get(
                                "orientation_error_target_rad",
                                [0.0] * 3,
                            )
                        ]
                        endpoint_delta = [
                            [float(value) for value in vector]
                            for vector
                            in contact_orientation_feedback.get(
                                "desired_endpoint_delta_world_m",
                                [[0.0] * 3 for _ in range(4)],
                            )
                        ]
                        requested_correction = [
                            float(value)
                            for value
                            in contact_orientation_feedback.get(
                                "requested_correction_rad",
                                [0.0] * 12,
                            )
                        ]
                        applied_correction = [
                            float(value)
                            for value
                            in contact_orientation_feedback.get(
                                "applied_correction_rad",
                                [0.0] * 12,
                            )
                        ]
                        if any(
                            abs(value) > 0.0
                            for value in applied_correction
                        ):
                            contact_orientation_feedback_nonzero_applied_count += 1
                        contact_orientation_feedback_max_error_abs_rad = max(
                            contact_orientation_feedback_max_error_abs_rad,
                            sum(
                                value * value
                                for value in orientation_error
                            )
                            ** 0.5,
                        )
                        contact_orientation_feedback_max_endpoint_delta_m = max(
                            contact_orientation_feedback_max_endpoint_delta_m,
                            max(
                                (
                                    sum(
                                        value * value
                                        for value in vector
                                    )
                                    ** 0.5
                                )
                                for vector in endpoint_delta
                            ),
                        )
                        contact_orientation_feedback_max_requested_abs_rad = max(
                            contact_orientation_feedback_max_requested_abs_rad,
                            max(
                                abs(value)
                                for value in requested_correction
                            ),
                        )
                        contact_orientation_feedback_max_applied_abs_rad = max(
                            contact_orientation_feedback_max_applied_abs_rad,
                            max(
                                abs(value)
                                for value in applied_correction
                            ),
                        )
                        contact_orientation_feedback_events.append(
                            {
                                "request_count": request_count,
                                "ref_id": int(request.ref_id),
                                "ref_frame": int(request.ref_frame),
                                "schedule_frame": int(
                                    contact_orientation_feedback.get(
                                        "schedule_frame",
                                        request.ref_frame,
                                    )
                                ),
                                "orientation_error_target_rad": (
                                    orientation_error
                                ),
                                "weighted_orientation_error_world_rad": [
                                    float(value)
                                    for value
                                    in contact_orientation_feedback.get(
                                        "weighted_orientation_error_world_rad",
                                        [0.0] * 3,
                                    )
                                ],
                                "desired_support": [
                                    bool(value)
                                    for value
                                    in contact_orientation_feedback.get(
                                        "desired_support",
                                        [False] * 4,
                                    )
                                ],
                                "measured_contact": [
                                    bool(value)
                                    for value
                                    in contact_orientation_feedback.get(
                                        "measured_contact",
                                        [False] * 4,
                                    )
                                ],
                                "normal_force_n": [
                                    float(value)
                                    for value
                                    in contact_orientation_feedback.get(
                                        "normal_force_n",
                                        [0.0] * 4,
                                    )
                                ],
                                "desired_endpoint_delta_world_m": (
                                    endpoint_delta
                                ),
                                "jacobian_joint_delta_rad": [
                                    float(value)
                                    for value
                                    in contact_orientation_feedback.get(
                                        "jacobian_joint_delta_rad",
                                        [0.0] * 12,
                                    )
                                ],
                                "requested_correction_rad": (
                                    requested_correction
                                ),
                                "applied_correction_rad": (
                                    applied_correction
                                ),
                            }
                        )
                    orientation_feedback = diagnostics.get(
                        "output_pitch_feedback",
                        {},
                    )
                    if orientation_feedback.get("enabled", False):
                        requested_correction = [
                            float(value)
                            for value in orientation_feedback.get(
                                "requested_correction_rad",
                                [0.0] * 12,
                            )
                        ]
                        applied_correction = [
                            float(value)
                            for value in orientation_feedback.get(
                                "applied_correction_rad",
                                [0.0] * 12,
                            )
                        ]
                        signed_error = float(
                            orientation_feedback.get(
                                "signed_orientation_axis_error_rad",
                                0.0,
                            )
                        )
                        if any(
                            abs(value) > 0.0
                            for value in applied_correction
                        ):
                            orientation_feedback_nonzero_applied_count += 1
                        orientation_feedback_max_error_abs_rad = max(
                            orientation_feedback_max_error_abs_rad,
                            abs(signed_error),
                        )
                        orientation_feedback_max_requested_abs_rad = max(
                            orientation_feedback_max_requested_abs_rad,
                            max(
                                abs(value)
                                for value in requested_correction
                            ),
                        )
                        orientation_feedback_max_applied_abs_rad = max(
                            orientation_feedback_max_applied_abs_rad,
                            max(
                                abs(value)
                                for value in applied_correction
                            ),
                        )
                        orientation_feedback_events.append(
                            {
                                "request_count": request_count,
                                "ref_id": int(request.ref_id),
                                "ref_frame": int(request.ref_frame),
                                "feedback_axis": str(
                                    orientation_feedback.get(
                                        "feedback_axis",
                                        "missing",
                                    )
                                ),
                                "signed_orientation_axis_error_rad": (
                                    signed_error
                                ),
                                "requested_correction_rad": (
                                    requested_correction
                                ),
                                "applied_correction_rad": (
                                    applied_correction
                                ),
                            }
                        )
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
        "schema_version": "pcbc-isolated-mppi-server-report-v3",
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
        "selected_best_sample_source_counts": (
            selected_best_sample_source_counts
        ),
        "structured_proposal_selection_events": (
            structured_proposal_selection_events
        ),
        "structured_proposal_cost_events": (
            structured_proposal_cost_events
        ),
        "rear_swing_height_feedback": {
            "event_count": len(rear_swing_height_feedback_events),
            "stuck_swing_counts": (
                rear_swing_height_feedback_stuck_counts
            ),
            "nonzero_applied_count": (
                rear_swing_height_feedback_nonzero_applied_count
            ),
            "max_height_deficit_m": (
                rear_swing_height_feedback_max_deficit_m
            ),
            "max_requested_abs_rad": (
                rear_swing_height_feedback_max_requested_abs_rad
            ),
            "max_applied_abs_rad": (
                rear_swing_height_feedback_max_applied_abs_rad
            ),
            "events": rear_swing_height_feedback_events,
        },
        "rear_support_tracking_feedback": {
            "event_count": len(
                rear_support_tracking_feedback_events
            ),
            "missing_support_counts": (
                rear_support_tracking_feedback_missing_counts
            ),
            "nonzero_applied_count": (
                rear_support_tracking_feedback_nonzero_applied_count
            ),
            "max_requested_abs_rad": (
                rear_support_tracking_feedback_max_requested_abs_rad
            ),
            "max_applied_abs_rad": (
                rear_support_tracking_feedback_max_applied_abs_rad
            ),
            "events": rear_support_tracking_feedback_events,
        },
        "orientation_feedback": {
            "event_count": len(orientation_feedback_events),
            "nonzero_applied_count": (
                orientation_feedback_nonzero_applied_count
            ),
            "max_error_abs_rad": (
                orientation_feedback_max_error_abs_rad
            ),
            "max_requested_abs_rad": (
                orientation_feedback_max_requested_abs_rad
            ),
            "max_applied_abs_rad": (
                orientation_feedback_max_applied_abs_rad
            ),
            "events": orientation_feedback_events,
        },
        "contact_orientation_feedback": {
            "event_count": len(
                contact_orientation_feedback_events
            ),
            "nonzero_applied_count": (
                contact_orientation_feedback_nonzero_applied_count
            ),
            "max_error_abs_rad": (
                contact_orientation_feedback_max_error_abs_rad
            ),
            "max_endpoint_delta_m": (
                contact_orientation_feedback_max_endpoint_delta_m
            ),
            "max_requested_abs_rad": (
                contact_orientation_feedback_max_requested_abs_rad
            ),
            "max_applied_abs_rad": (
                contact_orientation_feedback_max_applied_abs_rad
            ),
            "events": contact_orientation_feedback_events,
        },
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
