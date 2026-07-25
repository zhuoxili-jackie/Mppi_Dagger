#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import traceback

import torch

from _bootstrap import ROOT, load_contract, write_json

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="Validate Isaac MPPI clone copy/restore and repeated-rollout determinism."
)
parser.add_argument(
    "--task",
    default="RobotLab-Isaac-BeyondMimic-Flat-pcbC-v1-lateral-bipedal-stand-v0",
)
parser.add_argument("--samples", type=int, default=16)
parser.add_argument("--horizon", type=int, default=8)
parser.add_argument("--seed", type=int, default=1200)
parser.add_argument("--ref-id", type=int, default=0)
parser.add_argument(
    "--reference-config",
    type=str,
    default=None,
)
parser.add_argument(
    "--mppi-config",
    type=str,
    default="configs/expert_mppi.yaml",
)
parser.add_argument("--scenario", default="nominal")
parser.add_argument("--disable-fabric", action="store_true")
parser.add_argument(
    "--report",
    type=str,
    default=str(ROOT / "reports/02_mppi_state_copy_gate.json"),
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def main() -> dict:
    import gymnasium as gym
    import robot_lab.tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    from lateral_mppi_dagger.config import load_yaml
    from lateral_mppi_dagger.contract.action16 import Action16Adapter, ActionContract
    from lateral_mppi_dagger.env.isaac_adapter import IsaacLateralAdapter
    from lateral_mppi_dagger.env.isaac_mppi_rollout import (
        IsaacRolloutCostWeights,
        IsaacRolloutLoadLimits,
        IsaacWholeBodyMPPIProvider,
    )
    from lateral_mppi_dagger.env.scenarios import (
        configure_env_for_scenario,
        load_scenario_profile,
    )
    from lateral_mppi_dagger.expert.mppi_expert import MPPIConfig, ReferenceCenteredMPPI
    from lateral_mppi_dagger.reference.loader import ReferenceSet
    from lateral_mppi_dagger.reference.action_reference import (
        load_nominal_action_references,
        resolve_nominal_solver_overrides,
    )

    mppi_yaml = load_yaml(args_cli.mppi_config)
    contract = load_contract()
    references = ReferenceSet.from_config(
        args_cli.reference_config
        or mppi_yaml.get("reference_config", "configs/reference_708.yaml")
    )
    action_contract = ActionContract.from_dict(contract)
    action_adapter = Action16Adapter(action_contract)
    (
        nominal_action_reference_q_des_by_ref,
        nominal_action_reference_raw_by_ref,
        nominal_action_reference_overrides_by_ref,
        nominal_action_reference_record,
    ) = load_nominal_action_references(
        mppi_yaml.get("nominal_action_reference")
    )
    mppi_config = MPPIConfig(
        horizon=args_cli.horizon,
        samples=args_cli.samples,
        iterations=1,
        temperature=float(mppi_yaml["temperature"]),
        temporal_smoothing=float(mppi_yaml["temporal_smoothing"]),
        warm_start=bool(mppi_yaml["warm_start"]),
        selection_mode=str(mppi_yaml.get("selection_mode", "weighted")),
        reference_action_lookahead_steps=int(
            mppi_yaml.get("reference_action_lookahead_steps", 1)
        ),
        seed=args_cli.seed,
    )

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.samples,
        use_fabric=not args_cli.disable_fabric,
    )
    scenario_profile = load_scenario_profile(args_cli.scenario)
    configure_env_for_scenario(
        env_cfg,
        scenario_profile,
        num_envs=args_cli.samples,
    )
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device
    env = gym.make(args_cli.task, cfg=env_cfg)
    adapter = IsaacLateralAdapter(
        env,
        references,
        contract,
        scenario_profile=scenario_profile,
    )
    provider = IsaacWholeBodyMPPIProvider(
        adapter,
        references,
        action_adapter,
        mppi_config,
        torch.as_tensor(mppi_yaml["noise_std_leg"], dtype=torch.float32, device=args_cli.device),
        IsaacRolloutCostWeights.from_dict(mppi_yaml.get("cost_weights")),
        load_limits=IsaacRolloutLoadLimits.from_dict(
            mppi_yaml.get("load_limits")
        ),
        contact_force_threshold_n=float(
            mppi_yaml["contact_force_threshold_n"]
        ),
        physical_target_rate_limit_rad_s=(
            float(mppi_yaml["physical_target_rate_limit_rad_s"])
            if mppi_yaml.get("physical_target_rate_limit_rad_s") is not None
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

    try:
        adapter.reset(args_cli.seed, args_cli.ref_id)
        reset_state = adapter.episode_metadata()["reset_state"]
        reset_reference_max_abs = {
            "anchor_position_m": max(
                abs(float(value))
                for value in reset_state["anchor_position_minus_reference_m"]
            ),
            "anchor_orientation_rad": abs(
                float(reset_state["anchor_orientation_error_rad"])
            ),
            "anchor_twist": max(
                abs(float(value))
                for value in reset_state["anchor_twist_minus_reference"]
            ),
            "joint_position_rad": max(
                abs(float(value))
                for value in reset_state["joint_position_minus_reference"]
            ),
            "joint_velocity_rad_s": max(
                abs(float(value))
                for value in reset_state["joint_velocity_minus_reference"]
            ),
        }
        nominal_reset_match = bool(
            scenario_profile.resolved_name != "nominal"
            or max(reset_reference_max_abs.values()) <= 1.0e-5
        )
        request = adapter.expert_request()
        nominal = provider._nominal(request)
        previous = adapter.previous_commanded_action[0, :12]
        candidates = nominal.unsqueeze(0).repeat(args_cli.samples, 1, 1)
        generator = torch.Generator(device=args_cli.device)
        generator.manual_seed(args_cli.seed + 17)
        perturbation = 0.03 * torch.randn(
            candidates.shape,
            generator=generator,
            dtype=torch.float32,
            device=args_cli.device,
        )
        perturbation[0].zero_()
        candidates = ReferenceCenteredMPPI.project_sequence(
            candidates + perturbation,
            provider.raw_min,
            provider.raw_max,
            previous.expand(args_cli.samples, -1),
            provider.max_delta,
        )

        snapshot = provider.rollout.capture()
        configured_reference_overrides = (
            nominal_action_reference_overrides_by_ref.get(
                args_cli.ref_id,
                {},
            )
        )
        reference_overrides, solver_schedule_phase = (
            resolve_nominal_solver_overrides(
                configured_reference_overrides,
                request.ref_frame,
            )
        )
        solver_schedule_start_frame = None
        if solver_schedule_phase is not None:
            solver_schedule_start_frame = int(
                configured_reference_overrides["solver_schedule"][
                    solver_schedule_phase
                ]["start_frame"]
            )
        action_residual_weight = reference_overrides.get(
            "action_residual_weight"
        )
        base_orientation_cost_multiplier = float(
            reference_overrides.get(
                "base_orientation_cost_multiplier",
                1.0,
            )
        )
        before = provider.rollout.state_vector()[0].clone()
        before_delay_queue = adapter.action_delay_queue.clone()
        before_previous_command = adapter.previous_commanded_action.clone()
        before_platform_pose = adapter.platform.data.root_pose_w.clone()
        costs = provider.rollout.evaluate(
            candidates,
            snapshot,
            nominal,
            action_residual_weight=action_residual_weight,
            base_orientation_cost_multiplier=(
                base_orientation_cost_multiplier
            ),
        )
        provider.rollout.restore(snapshot)
        after = provider.rollout.state_vector()[0].clone()
        delay_queue_round_trip = float(
            torch.max(
                torch.abs(adapter.action_delay_queue - before_delay_queue),
            ).item()
        ) if adapter.action_delay_queue.numel() else 0.0
        previous_command_round_trip = float(
            torch.max(
                torch.abs(
                    adapter.previous_commanded_action - before_previous_command
                ),
            ).item()
        )
        platform_pose_round_trip = float(
            torch.max(
                torch.abs(adapter.platform.data.root_pose_w - before_platform_pose),
            ).item()
        )
        round_trip_max_abs = float(torch.max(torch.abs(before - after)).item())
        clone_state = provider.rollout.state_vector()
        clone_spread_max_abs = float(torch.max(torch.abs(clone_state - clone_state[0:1])).item())
        determinism = provider.rollout.probe_determinism(
            candidates,
            nominal,
            action_residual_weight=action_residual_weight,
            base_orientation_cost_multiplier=(
                base_orientation_cost_multiplier
            ),
        )
        rollout_action16 = torch.cat(
            (
                candidates,
                torch.zeros(
                    args_cli.samples,
                    args_cli.horizon,
                    4,
                    dtype=candidates.dtype,
                    device=candidates.device,
                ),
            ),
            dim=-1,
        )
        hard_zero = bool(
            torch.equal(
                rollout_action16[..., 12:],
                torch.zeros_like(rollout_action16[..., 12:]),
            )
        )
        thresholds = mppi_yaml["state_copy_gate"]
        passed = bool(
            torch.isfinite(costs).all().item()
            and round_trip_max_abs <= float(thresholds["state_max_abs"])
            and clone_spread_max_abs <= float(thresholds["state_max_abs"])
            and determinism["cost_max_abs"] <= float(thresholds["cost_max_abs"])
            and determinism["state_max_abs"] <= float(thresholds["state_max_abs"])
            and delay_queue_round_trip == 0.0
            and previous_command_round_trip == 0.0
            and platform_pose_round_trip <= float(thresholds["state_max_abs"])
            and hard_zero
            and nominal_reset_match
        )
        report = {
            "schema_version": "pcbc-mppi-state-copy-gate-v1",
            "pass": passed,
            "task": args_cli.task,
            "device": args_cli.device,
            "samples": args_cli.samples,
            "horizon": args_cli.horizon,
            "seed": args_cli.seed,
            "ref_id": args_cli.ref_id,
            "scenario": scenario_profile.metadata(),
            "reference_bridge": dict(adapter.reference_bridge),
            "nominal_action_reference": nominal_action_reference_record,
            "effective_action_residual_weight": (
                provider.rollout.cost_weights.action_residual
                if action_residual_weight is None
                else float(action_residual_weight)
            ),
            "effective_base_orientation_cost_multiplier": (
                base_orientation_cost_multiplier
            ),
            "effective_selection_mode": str(
                reference_overrides.get(
                    "selection_mode",
                    mppi_config.selection_mode,
                )
            ),
            "effective_warm_start": bool(
                reference_overrides.get(
                    "warm_start",
                    mppi_config.warm_start,
                )
            ),
            "solver_schedule_phase": solver_schedule_phase,
            "solver_schedule_start_frame": solver_schedule_start_frame,
            "validated_reference_frame": int(request.ref_frame),
            "action_delay_steps": adapter.action_delay_steps,
            "platform_position_jitter_m_sampled": list(adapter._platform_jitter),
            "reset_reference_max_abs": reset_reference_max_abs,
            "nominal_reset_matches_requested_reference": nominal_reset_match,
            "round_trip_state_max_abs": round_trip_max_abs,
            "round_trip_delay_queue_max_abs": delay_queue_round_trip,
            "round_trip_previous_command_max_abs": previous_command_round_trip,
            "round_trip_platform_pose_max_abs": platform_pose_round_trip,
            "clone_spread_max_abs": clone_spread_max_abs,
            "determinism": determinism,
            "finite_costs": bool(torch.isfinite(costs).all().item()),
            "minimum_cost": float(costs.min().item()),
            "maximum_cost": float(costs.max().item()),
            "hard_zero_rollout_actions": hard_zero,
            "thresholds": thresholds,
            "cost_components_for_best_candidate": provider.rollout.last_best_components,
        }
        write_json(args_cli.report, report)
        print(report)
        if not passed:
            raise RuntimeError(f"MPPI state-copy gate failed; see {args_cli.report}")
        return report
    finally:
        adapter.close()


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
