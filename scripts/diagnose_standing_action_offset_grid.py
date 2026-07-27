#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import sys
import traceback
from pathlib import Path

import numpy as np

from _bootstrap import ROOT, load_contract, write_json

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description=(
        "Replay a stable standing expert trajectory across an Isaac clone grid "
        "of bounded front-joint target offsets."
    )
)
parser.add_argument(
    "--task",
    default="RobotLab-Isaac-BeyondMimic-Flat-pcbC-v1-lateral-bipedal-stand-v0",
)
parser.add_argument(
    "--reference-config",
    default="configs/low_load_lateral/train_001/reference.yaml",
)
parser.add_argument(
    "--tracking-config",
    type=Path,
    default=ROOT / "configs/low_load_lateral/train_001/expert.yaml",
    help=(
        "Expert YAML supplying the unchanged closed-loop tracking thresholds "
        "used to score each diagnostic replay."
    ),
)
parser.add_argument(
    "--episode",
    type=Path,
    required=True,
)
parser.add_argument(
    "--precondition-episodes",
    type=Path,
    nargs="*",
    default=(),
    help=(
        "Optional collected episode shards replayed in order, with their "
        "recorded reset seed/reference, before the target reset. This "
        "diagnoses cross-episode reset-state dependence."
    ),
)
parser.add_argument(
    "--action-reference",
    type=Path,
    default=None,
    help=(
        "Optional nominal-action NPZ containing raw_action_leg and ref_id. "
        "When set, it replaces the episode's actions but keeps the episode's "
        "reference frames and desired-contact labels."
    ),
)
parser.add_argument(
    "--blend-base-action-reference",
    type=Path,
    default=None,
    help=(
        "Optional raw-action NPZ used as the zero endpoint for "
        "--action-blend-values. The active action reference remains the "
        "one endpoint; this is useful for bounded moving-from-standing grids."
    ),
)
parser.add_argument(
    "--action-blend-values",
    type=float,
    nargs="+",
    default=(1.0,),
    help=(
        "Per-candidate blend factors from the base action (0) to the active "
        "action reference (1); bounded diagnostic extrapolation up to 1.5 "
        "is allowed and remains subject to raw and physical rate limits."
    ),
)
parser.add_argument(
    "--gait-smoothing-window-values",
    type=int,
    nargs="+",
    default=(1,),
    help=(
        "Odd centered boxcar windows applied to the moving-minus-base raw "
        "action trajectory before blending. One preserves the input exactly."
    ),
)
parser.add_argument(
    "--action-frame-shift-values",
    type=int,
    nargs="+",
    default=(0,),
    help=(
        "Integer frame shifts applied to the moving-minus-base action "
        "trajectory after smoothing. Positive values advance the gait "
        "action; indices are edge-clamped rather than wrapped."
    ),
)
parser.add_argument(
    "--action-frame-shift-mode-values",
    choices=(
        "always",
        "both_front_desired",
        "any_front_swing",
        "fl_desired",
        "fr_desired",
    ),
    nargs="+",
    default=("always",),
    help=(
        "Schedule mask selecting when the shifted moving-action delta is "
        "used; outside the mask the unshifted delta is used."
    ),
)
parser.add_argument(
    "--lateral-velocity-feedback-gain-values",
    type=float,
    nargs="+",
    default=(0.0,),
    help=(
        "Physical joint-target correction gain in rad/(m/s), applied to "
        "(reference base vy - measured base vy) along the configured vector."
    ),
)
parser.add_argument(
    "--lateral-velocity-feedback-vector-leg",
    type=float,
    nargs="+",
    default=(0.0,) * 12,
    help=(
        "One or more concatenated 12-element physical joint vectors. Each "
        "vector is crossed with every configured feedback gain."
    ),
)
parser.add_argument(
    "--lateral-velocity-feedback-max-rad",
    type=float,
    default=0.06,
)
parser.add_argument(
    "--pitch-feedback-gain-values",
    type=float,
    nargs="+",
    default=(0.0,),
    help=(
        "Signed-pitch feedback gains in rad/rad, crossed with every "
        "configured 12-element physical joint vector."
    ),
)
parser.add_argument(
    "--pitch-feedback-vector-leg",
    type=float,
    nargs="+",
    default=(0.0,) * 12,
    help=(
        "One or more concatenated 12-element physical joint vectors. Each "
        "vector is crossed with every signed-pitch feedback gain."
    ),
)
parser.add_argument(
    "--pitch-feedback-max-abs-rad",
    type=float,
    default=0.05,
)
parser.add_argument(
    "--pitch-feedback-axis",
    choices=("x", "y", "z"),
    default="y",
    help=(
        "Target-frame rotation-vector component driving the bounded "
        "orientation feedback. The default y axis preserves the historical "
        "signed-pitch diagnostic exactly."
    ),
)
parser.add_argument(
    "--pitch-feedback-start-frame",
    type=int,
    default=0,
    help=(
        "Do not apply the bounded orientation feedback before this replay "
        "frame. Zero preserves the historical diagnostic behavior."
    ),
)
parser.add_argument(
    "--front-force-feedback-scale-values",
    type=float,
    nargs="+",
    default=(0.0,),
)
parser.add_argument(
    "--front-force-feedback-gain-leg",
    type=float,
    nargs=12,
    default=(0.0,) * 12,
)
parser.add_argument(
    "--front-force-feedback-target-n",
    type=float,
    default=8.0,
)
parser.add_argument(
    "--front-force-feedback-min-contact-n",
    type=float,
    default=1.0,
)
parser.add_argument(
    "--trajectory-correction",
    type=Path,
    default=None,
    help=(
        "Optional NPZ containing physical_correction_leg [steps,12] and "
        "ref_id for deterministic ILC-style action correction."
    ),
)
parser.add_argument(
    "--trajectory-correction-scale-values",
    type=float,
    nargs="+",
    default=(0.0,),
)
parser.add_argument("--seed", type=int, default=5208)
parser.add_argument("--ref-id", type=int, default=8)
parser.add_argument("--steps", type=int, default=100)
parser.add_argument("--replicates", type=int, default=3)
parser.add_argument(
    "--offset-values-rad",
    type=float,
    nargs="+",
    default=(-0.02, 0.0, 0.02),
)
parser.add_argument(
    "--front-offset-schedule-mode-values",
    choices=(
        "always",
        "both_front_desired",
        "any_front_swing",
        "each_front_desired",
        "each_front_swing",
    ),
    nargs="+",
    default=("always",),
    help=(
        "Schedule the configured front hip/thigh/calf offsets using the "
        "one-step-lookahead contact labels. Rear offsets remain continuous."
    ),
)
parser.add_argument(
    "--front-offset-start-frame",
    type=int,
    default=0,
    help=(
        "Do not apply front hip/thigh/calf offsets before this replay frame."
    ),
)
parser.add_argument(
    "--front-offset-ramp-frames",
    type=int,
    default=0,
    help=(
        "Smoothly ramp front offsets after --front-offset-start-frame. "
        "Zero applies them immediately."
    ),
)
parser.add_argument(
    "--rate-limits-rad-s",
    type=float,
    nargs="+",
    default=(2.25,),
    help=(
        "Positive physical leg-target rate limits replayed in parallel. "
        "The default preserves the original diagnostic behavior."
    ),
)
parser.add_argument(
    "--center-offsets-rad",
    type=float,
    nargs=4,
    default=(0.0, 0.0, 0.0, 0.0),
    metavar=("FL_THIGH", "FR_THIGH", "FL_CALF", "FR_CALF"),
    help="Four physical offsets around which the Cartesian delta grid is built.",
)
parser.add_argument(
    "--rear-thigh-offset-values-rad",
    type=float,
    nargs="+",
    default=(0.0,),
    help=(
        "Cartesian deltas applied independently to RL/RR thigh targets. "
        "The default keeps both rear-thigh offsets at zero."
    ),
)
parser.add_argument(
    "--rear-thigh-center-offsets-rad",
    type=float,
    nargs=2,
    default=(0.0, 0.0),
    metavar=("RL_THIGH", "RR_THIGH"),
)
parser.add_argument(
    "--front-hip-offset-values-rad",
    type=float,
    nargs="+",
    default=(0.0,),
)
parser.add_argument(
    "--front-hip-center-offsets-rad",
    type=float,
    nargs=2,
    default=(0.0, 0.0),
    metavar=("FL_HIP", "FR_HIP"),
)
parser.add_argument(
    "--rear-hip-offset-values-rad",
    type=float,
    nargs="+",
    default=(0.0,),
)
parser.add_argument(
    "--rear-hip-center-offsets-rad",
    type=float,
    nargs=2,
    default=(0.0, 0.0),
    metavar=("RL_HIP", "RR_HIP"),
)
parser.add_argument(
    "--rear-calf-offset-values-rad",
    type=float,
    nargs="+",
    default=(0.0,),
)
parser.add_argument(
    "--rear-calf-center-offsets-rad",
    type=float,
    nargs=2,
    default=(0.0, 0.0),
    metavar=("RL_CALF", "RR_CALF"),
)
parser.add_argument(
    "--report",
    type=Path,
    default=ROOT
    / "reports/low_load_lateral/train_001/diagnostics/standing_action_offset_grid.json",
)
parser.add_argument(
    "--trace-output",
    type=Path,
    default=None,
    help=(
        "Optional NPZ for the exact rate-limited candidate actions, stored "
        "as [steps,candidates*replicates,16] for proposal reconstruction."
    ),
)
parser.add_argument(
    "--clear-contact-warm-start-before-step",
    action="store_true",
    help=(
        "Before each real replay step, run the same snapshot restore used "
        "after MPPI candidate evaluation. This isolates restore-induced "
        "PhysX contact-cache effects from action differences."
    ),
)
parser.add_argument(
    "--clear-contact-warm-start-after-reset",
    action="store_true",
    help=(
        "Cold-recreate contact pairs once after the target reset and clear "
        "the contact-sensor buffers. This diagnoses cross-episode PhysX "
        "warm-start leakage without altering every replay step."
    ),
)
parser.add_argument(
    "--clone-env0-after-reset",
    action="store_true",
    help=(
        "Copy env0's complete explicit reset state and manager/sensor buffers "
        "to every diagnostic clone before replay. This removes independent "
        "vectorized-reset perturbations without cold-recreating contacts."
    ),
)
parser.add_argument(
    "--contact-prime-substeps",
    type=int,
    default=0,
    help=(
        "After the diagnostic cold restore, prime PhysX contacts for this "
        "many physics substeps before restoring explicit state again."
    ),
)
parser.add_argument(
    "--contact-restore-mode",
    choices=("cold", "in_place", "in_place_no_forward"),
    default="cold",
    help=(
        "cold preserves the MPPI restore used by the current provider; "
        "in_place skips the 10 m separation; in_place_no_forward also skips "
        "the post-write PhysX forward call."
    ),
)
parser.add_argument("--disable-fabric", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


FRONT_CONTROL_INDICES = (4, 5, 8, 9)
FRONT_LEG_INDICES = (0, 1, 4, 5, 8, 9)


def _quat_multiply_np(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.moveaxis(lhs, -1, 0)
    rw, rx, ry, rz = np.moveaxis(rhs, -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )


def _quat_conjugate_np(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).copy()
    result[..., 1:] *= -1.0
    return result


def _quat_rotation_vector_np(
    actual_quat_w: np.ndarray,
    target_quat_w: np.ndarray,
) -> np.ndarray:
    """Return target-frame shortest rotation vectors for wxyz quaternions."""

    actual = np.asarray(actual_quat_w, dtype=np.float64)
    target = np.asarray(target_quat_w, dtype=np.float64)
    epsilon = np.finfo(np.float64).eps
    actual /= np.maximum(
        np.linalg.norm(actual, axis=-1, keepdims=True),
        epsilon,
    )
    target /= np.maximum(
        np.linalg.norm(target, axis=-1, keepdims=True),
        epsilon,
    )
    relative = _quat_multiply_np(
        _quat_conjugate_np(target),
        actual,
    )
    relative = np.where(relative[..., :1] < 0.0, -relative, relative)
    vector = relative[..., 1:]
    vector_norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(
        vector_norm,
        np.maximum(relative[..., :1], epsilon),
    )
    return np.where(
        vector_norm > epsilon,
        vector * angle / np.maximum(vector_norm, epsilon),
        2.0 * vector,
    )


def main() -> dict:
    import gymnasium as gym
    import robot_lab.tasks  # noqa: F401
    import torch
    from isaaclab_tasks.utils import parse_env_cfg

    from lateral_mppi_dagger.contract.action16 import ActionContract
    from lateral_mppi_dagger.config import load_yaml, sha256_file
    from lateral_mppi_dagger.env.isaac_adapter import IsaacLateralAdapter
    from lateral_mppi_dagger.env.isaac_mppi_rollout import (
        IsaacMPPIRolloutCloner,
        IsaacRolloutCostWeights,
        _quat_conjugate,
        _quat_multiply,
        _quat_rotation_vector,
    )
    from lateral_mppi_dagger.env.scenarios import (
        configure_env_for_scenario,
        load_scenario_profile,
    )
    from lateral_mppi_dagger.evaluation.closed_loop_gate import (
        compute_tracking_metrics,
        tracking_threshold_failures,
    )
    from lateral_mppi_dagger.reference.loader import ReferenceSet

    if args_cli.steps < 1 or args_cli.replicates < 1:
        raise ValueError("--steps and --replicates must be positive.")
    if args_cli.pitch_feedback_start_frame < 0:
        raise ValueError("--pitch-feedback-start-frame must be non-negative.")
    if args_cli.contact_prime_substeps < 0:
        raise ValueError("--contact-prime-substeps must be non-negative.")
    if (
        args_cli.contact_prime_substeps
        and not (
            args_cli.clear_contact_warm_start_before_step
            or args_cli.clear_contact_warm_start_after_reset
        )
    ):
        raise ValueError(
            "--contact-prime-substeps requires a contact warm-start clear."
        )
    if (
        args_cli.contact_prime_substeps
        and args_cli.contact_restore_mode != "cold"
    ):
        raise ValueError(
            "--contact-prime-substeps requires --contact-restore-mode cold."
        )
    values = tuple(float(value) for value in args_cli.offset_values_rad)
    if not values or not np.isfinite(values).all():
        raise ValueError("--offset-values-rad must contain finite values.")
    front_offset_schedule_modes = tuple(
        args_cli.front_offset_schedule_mode_values
    )
    if not front_offset_schedule_modes:
        raise ValueError(
            "--front-offset-schedule-mode-values must contain at least one "
            "mode."
        )
    if (
        args_cli.front_offset_start_frame < 0
        or args_cli.front_offset_start_frame >= args_cli.steps
    ):
        raise ValueError(
            "--front-offset-start-frame must lie inside [0, steps)."
        )
    if args_cli.front_offset_ramp_frames < 0:
        raise ValueError(
            "--front-offset-ramp-frames must be non-negative."
        )
    blend_values = tuple(
        float(value) for value in args_cli.action_blend_values
    )
    if (
        not blend_values
        or not np.isfinite(blend_values).all()
        or any(value < 0.0 or value > 1.5 for value in blend_values)
    ):
        raise ValueError(
            "--action-blend-values must be finite values in [0, 1.5]."
        )
    smoothing_windows = tuple(
        int(value) for value in args_cli.gait_smoothing_window_values
    )
    if (
        not smoothing_windows
        or any(value < 1 or value % 2 == 0 for value in smoothing_windows)
    ):
        raise ValueError(
            "--gait-smoothing-window-values must contain positive odd "
            "integers."
        )
    action_frame_shifts = tuple(
        int(value) for value in args_cli.action_frame_shift_values
    )
    if not action_frame_shifts:
        raise ValueError(
            "--action-frame-shift-values must contain at least one integer."
        )
    action_frame_shift_modes = tuple(
        args_cli.action_frame_shift_mode_values
    )
    if not action_frame_shift_modes:
        raise ValueError(
            "--action-frame-shift-mode-values must contain at least one mode."
        )
    velocity_feedback_gains = tuple(
        float(value)
        for value in args_cli.lateral_velocity_feedback_gain_values
    )
    velocity_feedback_vector_values = np.asarray(
        args_cli.lateral_velocity_feedback_vector_leg,
        dtype=np.float32,
    )
    if velocity_feedback_vector_values.size % 12 != 0:
        raise ValueError(
            "Velocity-feedback vectors must contain a multiple of 12 values."
        )
    velocity_feedback_vectors = velocity_feedback_vector_values.reshape(
        -1,
        12,
    )
    velocity_feedback_max_rad = float(
        args_cli.lateral_velocity_feedback_max_rad
    )
    if (
        not velocity_feedback_gains
        or not np.isfinite(velocity_feedback_gains).all()
        or velocity_feedback_vectors.shape[0] < 1
        or not np.isfinite(velocity_feedback_vectors).all()
        or np.max(np.abs(velocity_feedback_vectors)) > 1.0
        or not np.isfinite(velocity_feedback_max_rad)
        or velocity_feedback_max_rad < 0.0
    ):
        raise ValueError(
            "Velocity-feedback gains/vector/cap must be finite, the vector "
            "must be within [-1,1], and the cap must be non-negative."
        )
    pitch_feedback_gains = tuple(
        float(value) for value in args_cli.pitch_feedback_gain_values
    )
    pitch_feedback_vector_values = np.asarray(
        args_cli.pitch_feedback_vector_leg,
        dtype=np.float32,
    )
    if pitch_feedback_vector_values.size % 12 != 0:
        raise ValueError(
            "Pitch-feedback vectors must contain a multiple of 12 values."
        )
    pitch_feedback_vectors = pitch_feedback_vector_values.reshape(
        -1,
        12,
    )
    pitch_feedback_max_abs_rad = float(
        args_cli.pitch_feedback_max_abs_rad
    )
    pitch_feedback_axis_index = {
        "x": 0,
        "y": 1,
        "z": 2,
    }[args_cli.pitch_feedback_axis]
    if (
        not pitch_feedback_gains
        or not np.isfinite(pitch_feedback_gains).all()
        or pitch_feedback_vectors.shape[0] < 1
        or not np.isfinite(pitch_feedback_vectors).all()
        or np.max(np.abs(pitch_feedback_vectors)) > 1.0
        or not np.isfinite(pitch_feedback_max_abs_rad)
        or pitch_feedback_max_abs_rad < 0.0
    ):
        raise ValueError(
            "Pitch-feedback gains/vector/cap must be finite, the vector "
            "must be within [-1,1], and the cap must be non-negative."
        )
    front_force_feedback_scales = tuple(
        float(value)
        for value in args_cli.front_force_feedback_scale_values
    )
    front_force_feedback_gain = np.asarray(
        args_cli.front_force_feedback_gain_leg,
        dtype=np.float32,
    )
    front_force_feedback_target_n = float(
        args_cli.front_force_feedback_target_n
    )
    front_force_feedback_min_contact_n = float(
        args_cli.front_force_feedback_min_contact_n
    )
    if (
        not front_force_feedback_scales
        or not np.isfinite(front_force_feedback_scales).all()
        or any(value < 0.0 for value in front_force_feedback_scales)
        or front_force_feedback_gain.shape != (12,)
        or not np.isfinite(front_force_feedback_gain).all()
        or not np.isfinite(front_force_feedback_target_n)
        or front_force_feedback_target_n <= 0.0
        or not np.isfinite(front_force_feedback_min_contact_n)
        or front_force_feedback_min_contact_n < 0.0
    ):
        raise ValueError(
            "Front-force feedback scale/gain/threshold values are invalid."
        )
    trajectory_correction_scales = tuple(
        float(value)
        for value in args_cli.trajectory_correction_scale_values
    )
    if (
        not trajectory_correction_scales
        or not np.isfinite(trajectory_correction_scales).all()
        or any(abs(value) > 5.0 for value in trajectory_correction_scales)
    ):
        raise ValueError(
            "Trajectory-correction scales must be finite and within [-5,5]."
        )
    rate_limits = tuple(
        float(value) for value in args_cli.rate_limits_rad_s
    )
    if (
        not rate_limits
        or not np.isfinite(rate_limits).all()
        or any(value <= 0.0 for value in rate_limits)
    ):
        raise ValueError(
            "--rate-limits-rad-s must contain positive finite values."
        )
    center = np.asarray(
        args_cli.center_offsets_rad,
        dtype=np.float32,
    )
    if center.shape != (4,) or not np.isfinite(center).all():
        raise ValueError("--center-offsets-rad must contain four finite values.")
    rear_values = tuple(
        float(value)
        for value in args_cli.rear_thigh_offset_values_rad
    )
    if not rear_values or not np.isfinite(rear_values).all():
        raise ValueError(
            "--rear-thigh-offset-values-rad must contain finite values."
        )
    rear_center = np.asarray(
        args_cli.rear_thigh_center_offsets_rad,
        dtype=np.float32,
    )
    if rear_center.shape != (2,) or not np.isfinite(rear_center).all():
        raise ValueError(
            "--rear-thigh-center-offsets-rad must contain two finite values."
        )
    front_hip_values = tuple(
        float(value) for value in args_cli.front_hip_offset_values_rad
    )
    front_hip_center = np.asarray(
        args_cli.front_hip_center_offsets_rad,
        dtype=np.float32,
    )
    if (
        not front_hip_values
        or not np.isfinite(front_hip_values).all()
        or front_hip_center.shape != (2,)
        or not np.isfinite(front_hip_center).all()
    ):
        raise ValueError(
            "Front-hip offset values and two-element center must be finite."
        )
    rear_hip_values = tuple(
        float(value) for value in args_cli.rear_hip_offset_values_rad
    )
    rear_hip_center = np.asarray(
        args_cli.rear_hip_center_offsets_rad,
        dtype=np.float32,
    )
    if (
        not rear_hip_values
        or not np.isfinite(rear_hip_values).all()
        or rear_hip_center.shape != (2,)
        or not np.isfinite(rear_hip_center).all()
    ):
        raise ValueError(
            "Rear-hip offset values and two-element center must be finite."
        )
    rear_calf_values = tuple(
        float(value) for value in args_cli.rear_calf_offset_values_rad
    )
    rear_calf_center = np.asarray(
        args_cli.rear_calf_center_offsets_rad,
        dtype=np.float32,
    )
    if (
        not rear_calf_values
        or not np.isfinite(rear_calf_values).all()
        or rear_calf_center.shape != (2,)
        or not np.isfinite(rear_calf_center).all()
    ):
        raise ValueError(
            "Rear-calf offset values and two-element center must be finite."
        )

    episode_path = args_cli.episode.resolve()
    with np.load(episode_path, allow_pickle=False) as archive:
        recorded_action = np.asarray(
            archive["executed_action16"],
            dtype=np.float32,
        )
        recorded_ref_frame = np.asarray(
            archive["ref_frame"],
            dtype=np.int64,
        )
        recorded_desired_contact = np.asarray(
            archive["desired_contact"],
            dtype=bool,
        )
    precondition_replays: list[
        tuple[Path, int, int, np.ndarray]
    ] = []
    precondition_records: list[dict[str, object]] = []
    for value in args_cli.precondition_episodes:
        precondition_path = value.expanduser().resolve()
        with np.load(precondition_path, allow_pickle=False) as archive:
            precondition_action = np.asarray(
                archive["executed_action16"],
                dtype=np.float32,
            )
            precondition_ref_ids = np.asarray(
                archive["ref_id"],
                dtype=np.int64,
            ).reshape(-1)
            metadata = json.loads(
                str(np.asarray(archive["metadata_json"]).reshape(-1)[0])
            )
        if (
            precondition_action.ndim != 2
            or precondition_action.shape[1] != 16
            or precondition_action.shape[0] < 1
            or not np.isfinite(precondition_action).all()
            or not precondition_ref_ids.size
            or np.any(precondition_ref_ids != precondition_ref_ids[0])
            or not np.array_equal(
                precondition_action[:, 12:],
                np.zeros_like(precondition_action[:, 12:]),
            )
        ):
            raise ValueError(
                "Each precondition episode must contain finite "
                "[steps,16] executed actions, one consistent ref_id, and "
                "exact-zero wheel actions."
            )
        precondition_seed = int(metadata["seed"])
        precondition_ref_id = int(precondition_ref_ids[0])
        if int(metadata["ref_id"]) != precondition_ref_id:
            raise ValueError(
                "Precondition episode metadata ref_id is inconsistent."
            )
        precondition_replays.append(
            (
                precondition_path,
                precondition_seed,
                precondition_ref_id,
                precondition_action,
            )
        )
        precondition_records.append(
            {
                "path": str(precondition_path),
                "sha256": sha256_file(precondition_path),
                "seed": precondition_seed,
                "ref_id": precondition_ref_id,
                "steps": int(precondition_action.shape[0]),
            }
        )
    action_reference_path = None
    blend_base_action_reference_path = None
    blend_base_recorded_action = None
    trajectory_correction_path = None
    physical_trajectory_correction = None
    if args_cli.action_reference is not None:
        action_reference_path = (
            args_cli.action_reference.expanduser().resolve()
        )
        with np.load(
            action_reference_path,
            allow_pickle=False,
        ) as archive:
            raw_action_leg = np.asarray(
                archive["raw_action_leg"],
                dtype=np.float32,
            )
            action_reference_ref_id = int(
                np.asarray(archive["ref_id"]).reshape(-1)[0]
            )
        if (
            raw_action_leg.ndim != 2
            or raw_action_leg.shape[1] != 12
        ):
            raise ValueError(
                "action-reference raw_action_leg must have shape [frames,12]."
            )
        if action_reference_ref_id != args_cli.ref_id:
            raise ValueError(
                "action-reference ref_id differs from --ref-id."
            )
        action_frames = np.minimum(
            recorded_ref_frame + 1,
            raw_action_leg.shape[0] - 1,
        )
        recorded_action = np.zeros_like(recorded_action)
        recorded_action[:, :12] = raw_action_leg[action_frames]
    if args_cli.blend_base_action_reference is not None:
        if action_reference_path is None:
            raise ValueError(
                "--blend-base-action-reference requires --action-reference."
            )
        blend_base_action_reference_path = (
            args_cli.blend_base_action_reference.expanduser().resolve()
        )
        with np.load(
            blend_base_action_reference_path,
            allow_pickle=False,
        ) as archive:
            blend_base_raw_action_leg = np.asarray(
                archive["raw_action_leg"],
                dtype=np.float32,
            )
        if (
            blend_base_raw_action_leg.ndim != 2
            or blend_base_raw_action_leg.shape[1] != 12
        ):
            raise ValueError(
                "blend-base raw_action_leg must have shape [frames,12]."
            )
        blend_base_frames = np.minimum(
            recorded_ref_frame + 1,
            blend_base_raw_action_leg.shape[0] - 1,
        )
        blend_base_recorded_action = np.zeros_like(recorded_action)
        blend_base_recorded_action[:, :12] = (
            blend_base_raw_action_leg[blend_base_frames]
        )
    elif (
        any(value != 1.0 for value in blend_values)
        or any(value != 1 for value in smoothing_windows)
        or any(value != 0 for value in action_frame_shifts)
    ):
        raise ValueError(
            "Non-unit action blends, smoothing windows, or frame shifts require "
            "--blend-base-action-reference."
        )
    if args_cli.trajectory_correction is not None:
        trajectory_correction_path = (
            args_cli.trajectory_correction.expanduser().resolve()
        )
        with np.load(
            trajectory_correction_path,
            allow_pickle=False,
        ) as archive:
            physical_trajectory_correction = np.asarray(
                archive["physical_correction_leg"],
                dtype=np.float32,
            )
            correction_ref_id = int(
                np.asarray(archive["ref_id"]).reshape(-1)[0]
            )
        if (
            physical_trajectory_correction.ndim != 2
            or physical_trajectory_correction.shape[1] != 12
            or physical_trajectory_correction.shape[0] < args_cli.steps
            or not np.isfinite(physical_trajectory_correction).all()
            or correction_ref_id != args_cli.ref_id
        ):
            raise ValueError(
                "Trajectory correction must be finite [steps,12] and match "
                "--ref-id."
            )
    elif any(value != 0.0 for value in trajectory_correction_scales):
        raise ValueError(
            "Non-zero trajectory-correction scales require "
            "--trajectory-correction."
        )
    if recorded_action.shape[0] < args_cli.steps:
        raise ValueError("The source episode is shorter than --steps.")
    if not np.array_equal(
        recorded_action[:, 12:],
        np.zeros_like(recorded_action[:, 12:]),
    ):
        raise ValueError("The source episode has non-zero wheel actions.")

    candidate_specs = [
        (
            blend,
            smoothing_window,
            action_frame_shift,
            action_frame_shift_mode,
            front_offset_schedule_mode,
            velocity_feedback_gain,
            velocity_feedback_vector_index,
            pitch_feedback_gain,
            pitch_feedback_vector_index,
            front_force_feedback_scale,
            trajectory_correction_scale,
            rate_limit,
            tuple(
                float(center[index] + delta)
                for index, delta in enumerate(deltas)
            ),
            tuple(
                float(rear_center[index] + delta)
                for index, delta in enumerate(rear_deltas)
            ),
            tuple(
                float(front_hip_center[index] + delta)
                for index, delta in enumerate(front_hip_deltas)
            ),
            tuple(
                float(rear_hip_center[index] + delta)
                for index, delta in enumerate(rear_hip_deltas)
            ),
            tuple(
                float(rear_calf_center[index] + delta)
                for index, delta in enumerate(rear_calf_deltas)
            ),
        )
        for blend in blend_values
        for smoothing_window in smoothing_windows
        for action_frame_shift in action_frame_shifts
        for action_frame_shift_mode in action_frame_shift_modes
        for front_offset_schedule_mode in front_offset_schedule_modes
        for velocity_feedback_gain in velocity_feedback_gains
        for velocity_feedback_vector_index in range(
            velocity_feedback_vectors.shape[0]
        )
        for pitch_feedback_gain in pitch_feedback_gains
        for pitch_feedback_vector_index in range(
            pitch_feedback_vectors.shape[0]
        )
        for front_force_feedback_scale in front_force_feedback_scales
        for trajectory_correction_scale in trajectory_correction_scales
        for rate_limit in rate_limits
        for deltas in itertools.product(values, repeat=4)
        for rear_deltas in itertools.product(rear_values, repeat=2)
        for front_hip_deltas in itertools.product(
            front_hip_values, repeat=2
        )
        for rear_hip_deltas in itertools.product(
            rear_hip_values, repeat=2
        )
        for rear_calf_deltas in itertools.product(
            rear_calf_values, repeat=2
        )
    ]
    candidate_count = len(candidate_specs)
    num_envs = candidate_count * args_cli.replicates
    physical_offsets = np.zeros((num_envs, 12), dtype=np.float32)
    physical_rate_limits = np.zeros(num_envs, dtype=np.float32)
    candidate_velocity_feedback_gains = np.zeros(
        num_envs,
        dtype=np.float32,
    )
    candidate_velocity_feedback_vectors = np.zeros(
        (num_envs, 12),
        dtype=np.float32,
    )
    candidate_pitch_feedback_gains = np.zeros(
        num_envs,
        dtype=np.float32,
    )
    candidate_pitch_feedback_vectors = np.zeros(
        (num_envs, 12),
        dtype=np.float32,
    )
    candidate_front_force_feedback_scales = np.zeros(
        num_envs,
        dtype=np.float32,
    )
    front_offset_schedule_code = {
        "always": 0,
        "both_front_desired": 1,
        "any_front_swing": 2,
        "each_front_desired": 3,
        "each_front_swing": 4,
    }
    candidate_front_offset_schedule_codes = np.zeros(
        num_envs,
        dtype=np.int64,
    )
    candidate_trajectory_correction_scales = np.zeros(
        num_envs,
        dtype=np.float32,
    )
    candidate_recorded_action = np.zeros(
        (recorded_action.shape[0], num_envs, 16),
        dtype=np.float32,
    )
    smoothed_action_delta: dict[int, np.ndarray] = {}
    if blend_base_recorded_action is not None:
        action_delta = recorded_action - blend_base_recorded_action
        for smoothing_window in smoothing_windows:
            if smoothing_window == 1:
                smoothed_action_delta[smoothing_window] = action_delta
                continue
            radius = smoothing_window // 2
            padded = np.pad(
                action_delta[:, :12],
                ((radius, radius), (0, 0)),
                mode="edge",
            )
            kernel = np.full(
                smoothing_window,
                1.0 / smoothing_window,
                dtype=np.float32,
            )
            filtered = np.stack(
                [
                    np.convolve(
                        padded[:, joint_index],
                        kernel,
                        mode="valid",
                    )
                    for joint_index in range(12)
                ],
                axis=-1,
            ).astype(np.float32)
            value = np.zeros_like(action_delta)
            value[:, :12] = filtered
            smoothed_action_delta[smoothing_window] = value
    for candidate_index, (
        blend,
        smoothing_window,
        action_frame_shift,
        action_frame_shift_mode,
        front_offset_schedule_mode,
        velocity_feedback_gain,
        velocity_feedback_vector_index,
        pitch_feedback_gain,
        pitch_feedback_vector_index,
        front_force_feedback_scale,
        trajectory_correction_scale,
        rate_limit,
        offsets,
        rear_thigh_offsets,
        front_hip_offsets,
        rear_hip_offsets,
        rear_calf_offsets,
    ) in enumerate(
        candidate_specs
    ):
        for replicate_index in range(args_cli.replicates):
            env_index = (
                candidate_index * args_cli.replicates + replicate_index
            )
            if blend_base_recorded_action is None:
                candidate_recorded_action[:, env_index] = recorded_action
            else:
                unshifted_delta = smoothed_action_delta[smoothing_window]
                shifted_delta = unshifted_delta[
                    np.clip(
                        np.arange(recorded_action.shape[0])
                        + action_frame_shift,
                        0,
                        recorded_action.shape[0] - 1,
                    )
                ]
                if action_frame_shift_mode == "always":
                    shift_mask = np.ones(
                        recorded_action.shape[0],
                        dtype=bool,
                    )
                elif action_frame_shift_mode == "both_front_desired":
                    shift_mask = np.all(
                        recorded_desired_contact[:, :2],
                        axis=1,
                    )
                elif action_frame_shift_mode == "any_front_swing":
                    shift_mask = ~np.all(
                        recorded_desired_contact[:, :2],
                        axis=1,
                    )
                elif action_frame_shift_mode == "fl_desired":
                    shift_mask = recorded_desired_contact[:, 0]
                elif action_frame_shift_mode == "fr_desired":
                    shift_mask = recorded_desired_contact[:, 1]
                else:
                    raise AssertionError(
                        f"Unhandled shift mode {action_frame_shift_mode!r}."
                    )
                scheduled_delta = np.where(
                    shift_mask[:, None],
                    shifted_delta,
                    unshifted_delta,
                )
                candidate_recorded_action[:, env_index] = (
                    blend_base_recorded_action
                    + blend * scheduled_delta
                )
            candidate_velocity_feedback_gains[env_index] = (
                velocity_feedback_gain
            )
            candidate_velocity_feedback_vectors[env_index] = (
                velocity_feedback_vectors[
                    velocity_feedback_vector_index
                ]
            )
            candidate_pitch_feedback_gains[env_index] = (
                pitch_feedback_gain
            )
            candidate_pitch_feedback_vectors[env_index] = (
                pitch_feedback_vectors[
                    pitch_feedback_vector_index
                ]
            )
            candidate_front_force_feedback_scales[env_index] = (
                front_force_feedback_scale
            )
            candidate_front_offset_schedule_codes[env_index] = (
                front_offset_schedule_code[front_offset_schedule_mode]
            )
            candidate_trajectory_correction_scales[env_index] = (
                trajectory_correction_scale
            )
            physical_rate_limits[env_index] = rate_limit
            physical_offsets[
                env_index,
                list(FRONT_CONTROL_INDICES),
            ] = offsets
            physical_offsets[
                env_index,
                [6, 7],
            ] = rear_thigh_offsets
            physical_offsets[
                env_index,
                [0, 1],
            ] = front_hip_offsets
            physical_offsets[
                env_index,
                [2, 3],
            ] = rear_hip_offsets
            physical_offsets[
                env_index,
                [10, 11],
            ] = rear_calf_offsets

    contract_dict = load_contract()
    contract = ActionContract.from_dict(contract_dict)
    references = ReferenceSet.from_config(args_cli.reference_config)
    reference = references[args_cli.ref_id]
    tracking_config_path = args_cli.tracking_config.expanduser().resolve()
    tracking_config = load_yaml(tracking_config_path)
    tracking_thresholds = dict(
        tracking_config["closed_loop_gate"]["tracking_thresholds"]
    )
    metric_frames = np.minimum(
        recorded_ref_frame[: args_cli.steps],
        reference.frames - 1,
    )
    target_lateral_velocity = np.asarray(
        reference.body_lin_vel_w[metric_frames, 0, 1],
        dtype=np.float32,
    )
    target_lateral_displacement = float(
        reference.body_pos_w[metric_frames[-1], 0, 1]
        - reference.body_pos_w[metric_frames[0], 0, 1]
    )
    scenario = load_scenario_profile("nominal")
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    configure_env_for_scenario(env_cfg, scenario, num_envs=num_envs)
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device
    env = gym.make(args_cli.task, cfg=env_cfg)
    adapter = IsaacLateralAdapter(
        env,
        references,
        contract_dict,
        scenario_profile=scenario,
    )
    contact_restorer = (
        IsaacMPPIRolloutCloner(
            adapter,
            references,
            contract,
            horizon=1,
            cost_weights=IsaacRolloutCostWeights(),
        )
        if (
            args_cli.clone_env0_after_reset
            or
            args_cli.clear_contact_warm_start_before_step
            or args_cli.clear_contact_warm_start_after_reset
        )
        else None
    )

    try:
        for (
            _,
            precondition_seed,
            precondition_ref_id,
            precondition_action,
        ) in precondition_replays:
            adapter.reset(
                precondition_seed,
                precondition_ref_id,
            )
            action_t = torch.as_tensor(
                precondition_action,
                dtype=torch.float32,
                device=adapter.base.device,
            )
            for step_action in action_t:
                env.step(
                    step_action.unsqueeze(0).expand(
                        num_envs,
                        -1,
                    )
                )
        adapter.reset(args_cli.seed, args_cli.ref_id)
        if args_cli.clone_env0_after_reset:
            assert contact_restorer is not None
            clone_snapshot = contact_restorer.capture()
            contact_restorer.restore(
                clone_snapshot,
                clear_contact_warm_start=False,
                forward_after_state_write=(
                    args_cli.contact_restore_mode
                    != "in_place_no_forward"
                ),
            )
        if args_cli.clear_contact_warm_start_after_reset:
            assert contact_restorer is not None
            contact_restorer.restore(
                contact_restorer.capture(),
                clear_contact_warm_start=True,
                contact_prime_substeps=args_cli.contact_prime_substeps,
            )
            adapter.contact_sensor.reset()
        if action_reference_path is not None:
            schedule_frames = np.minimum(
                recorded_ref_frame,
                adapter.contact_schedules[args_cli.ref_id].shape[0] - 1,
            )
            recorded_desired_contact = np.asarray(
                adapter.contact_schedules[args_cli.ref_id][schedule_frames],
                dtype=bool,
            )
        device = adapter.base.device
        offsets_t = torch.as_tensor(
            physical_offsets,
            dtype=torch.float32,
            device=device,
        )
        velocity_feedback_gains_t = torch.as_tensor(
            candidate_velocity_feedback_gains,
            dtype=torch.float32,
            device=device,
        )
        velocity_feedback_vector_t = torch.as_tensor(
            candidate_velocity_feedback_vectors,
            dtype=torch.float32,
            device=device,
        )
        pitch_feedback_gains_t = torch.as_tensor(
            candidate_pitch_feedback_gains,
            dtype=torch.float32,
            device=device,
        )
        pitch_feedback_vector_t = torch.as_tensor(
            candidate_pitch_feedback_vectors,
            dtype=torch.float32,
            device=device,
        )
        front_force_feedback_scales_t = torch.as_tensor(
            candidate_front_force_feedback_scales,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(-1)
        front_force_feedback_gain_t = torch.as_tensor(
            front_force_feedback_gain,
            dtype=torch.float32,
            device=device,
        )
        front_offset_schedule_codes_t = torch.as_tensor(
            candidate_front_offset_schedule_codes,
            dtype=torch.int64,
            device=device,
        )
        trajectory_correction_scales_t = torch.as_tensor(
            candidate_trajectory_correction_scales,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(-1)
        scale = torch.as_tensor(
            contract.scale,
            dtype=torch.float32,
            device=device,
        )
        raw_min = torch.as_tensor(
            contract.raw_min,
            dtype=torch.float32,
            device=device,
        )
        raw_max = torch.as_tensor(
            contract.raw_max,
            dtype=torch.float32,
            device=device,
        )
        maximum_physical_delta = torch.as_tensor(
            physical_rate_limits / 50.0,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(-1)
        maximum_raw_delta = maximum_physical_delta / scale.unsqueeze(0)
        previous_action = torch.zeros(
            (num_envs, 16),
            dtype=torch.float32,
            device=device,
        )
        initial_anchor_position = (
            adapter.command.robot_anchor_pos_w
            - adapter.base.scene.env_origins
        ).detach().clone()
        initial_anchor_quaternion = (
            adapter.command.robot_anchor_quat_w.detach().clone()
        )
        reference_anchor_quaternion = torch.as_tensor(
            reference.body_quat_w[
                metric_frames,
                int(adapter.command.motion_anchor_body_index),
            ],
            dtype=torch.float32,
            device=device,
        )
        alignment_quaternion = _quat_multiply(
            initial_anchor_quaternion[0],
            _quat_conjugate(reference_anchor_quaternion[0]),
        )
        target_anchor_quaternion = _quat_multiply(
            alignment_quaternion.unsqueeze(0).expand(
                args_cli.steps,
                -1,
            ),
            reference_anchor_quaternion,
        )
        done_any = torch.zeros(
            num_envs,
            dtype=torch.bool,
            device=device,
        )
        force_samples: list[np.ndarray] = []
        pre_step_force_samples: list[np.ndarray] = []
        base_delta_samples: list[np.ndarray] = []
        orientation_error_samples: list[np.ndarray] = []
        lateral_velocity_samples: list[np.ndarray] = []
        action_samples: list[np.ndarray] = []
        velocity_feedback_samples: list[np.ndarray] = []
        pitch_feedback_samples: list[np.ndarray] = []
        signed_pitch_error_samples: list[np.ndarray] = []
        front_force_feedback_samples: list[np.ndarray] = []
        trajectory_correction_samples: list[np.ndarray] = []
        base_pose_samples: list[np.ndarray] = []
        base_twist_samples: list[np.ndarray] = []
        joint_position_samples: list[np.ndarray] = []
        joint_velocity_samples: list[np.ndarray] = []
        wheel_pose_samples: list[np.ndarray] = []
        wheel_twist_samples: list[np.ndarray] = []
        contact_force_samples: list[np.ndarray] = []
        measured_contact_samples: list[np.ndarray] = []

        for step in range(args_cli.steps):
            anchor_position_w = adapter.command.robot_anchor_pos_w
            base_pose_samples.append(
                torch.cat(
                    (
                        anchor_position_w,
                        adapter.command.robot_anchor_quat_w,
                    ),
                    dim=-1,
                )
                .detach()
                .cpu()
                .numpy()
            )
            base_twist_samples.append(
                torch.cat(
                    (
                        adapter.command.robot_anchor_lin_vel_w,
                        adapter.command.robot_anchor_ang_vel_w,
                    ),
                    dim=-1,
                )
                .detach()
                .cpu()
                .numpy()
            )
            joint_position_samples.append(
                adapter.robot.data.joint_pos[:, adapter.joint_ids]
                .detach()
                .cpu()
                .numpy()
            )
            joint_velocity_samples.append(
                adapter.robot.data.joint_vel[:, adapter.joint_ids]
                .detach()
                .cpu()
                .numpy()
            )
            wheel_pose_samples.append(
                torch.cat(
                    (
                        adapter.robot.data.body_pos_w[
                            :,
                            adapter.wheel_body_ids,
                        ],
                        adapter.robot.data.body_quat_w[
                            :,
                            adapter.wheel_body_ids,
                        ],
                    ),
                    dim=-1,
                )
                .detach()
                .cpu()
                .numpy()
            )
            wheel_twist_samples.append(
                torch.cat(
                    (
                        adapter.robot.data.body_lin_vel_w[
                            :,
                            adapter.wheel_body_ids,
                        ],
                        adapter.robot.data.body_ang_vel_w[
                            :,
                            adapter.wheel_body_ids,
                        ],
                    ),
                    dim=-1,
                )
                .detach()
                .cpu()
                .numpy()
            )
            pre_step_contact_force = (
                adapter.contact_sensor.data.net_forces_w[
                    :,
                    adapter.contact_body_ids,
                ]
                .detach()
                .cpu()
                .numpy()
            )
            contact_force_samples.append(pre_step_contact_force)
            measured_contact_samples.append(
                np.linalg.norm(pre_step_contact_force, axis=-1) >= 8.0
            )
            pre_step_force_samples.append(
                adapter.contact_sensor.data.net_forces_w[
                    :,
                    adapter.contact_body_ids[:2],
                ]
                .detach()
                .cpu()
                .numpy()
            )
            if contact_restorer is not None:
                contact_restorer.restore(
                    contact_restorer.capture(),
                    clear_contact_warm_start=(
                        args_cli.contact_restore_mode == "cold"
                    ),
                    forward_after_state_write=(
                        args_cli.contact_restore_mode
                        != "in_place_no_forward"
                    ),
                    contact_prime_substeps=args_cli.contact_prime_substeps,
                )
            desired = torch.as_tensor(
                candidate_recorded_action[step],
                dtype=torch.float32,
                device=device,
            ).clone()
            signed_pitch_error = _quat_rotation_vector(
                adapter.command.robot_anchor_quat_w,
                target_anchor_quaternion[step].unsqueeze(0).expand(
                    num_envs,
                    -1,
                ),
            )[..., pitch_feedback_axis_index]
            active_pitch_feedback_error = (
                signed_pitch_error
                if step >= args_cli.pitch_feedback_start_frame
                else torch.zeros_like(signed_pitch_error)
            )
            pitch_feedback_physical = torch.clamp(
                (
                    active_pitch_feedback_error
                    * pitch_feedback_gains_t
                ).unsqueeze(-1)
                * pitch_feedback_vector_t,
                min=-pitch_feedback_max_abs_rad,
                max=pitch_feedback_max_abs_rad,
            )
            desired[:, :12] += pitch_feedback_physical / scale[:12]
            velocity_error = (
                float(target_lateral_velocity[step])
                - adapter.command.robot_anchor_lin_vel_w[:, 1]
            )
            velocity_feedback_scalar = torch.clamp(
                velocity_feedback_gains_t * velocity_error,
                min=-velocity_feedback_max_rad,
                max=velocity_feedback_max_rad,
            )
            velocity_feedback_physical = (
                velocity_feedback_scalar.unsqueeze(-1)
                * velocity_feedback_vector_t
            )
            desired[:, :12] += velocity_feedback_physical / scale[:12]
            front_normal = torch.abs(
                adapter.contact_sensor.data.net_forces_w[
                    :,
                    adapter.contact_body_ids[:2],
                    0,
                ]
            )
            front_force_deficit = torch.clamp(
                (
                    front_force_feedback_target_n
                    - front_normal
                )
                / front_force_feedback_target_n,
                min=0.0,
                max=1.0,
            )
            measured_front_contact = (
                front_normal >= front_force_feedback_min_contact_n
            )
            desired_front_lookahead = torch.as_tensor(
                recorded_desired_contact[
                    min(step + 1, args_cli.steps - 1),
                    :2,
                ],
                dtype=torch.float32,
                device=device,
            )
            front_feedback_factor = torch.zeros(
                (num_envs, 12),
                dtype=torch.float32,
                device=device,
            )
            for front_index, joint_indices in enumerate(
                ((0, 4, 8), (1, 5, 9))
            ):
                front_feedback_factor[:, list(joint_indices)] = (
                    desired_front_lookahead[front_index]
                    * measured_front_contact[:, front_index].float()
                    * front_force_deficit[:, front_index]
                ).unsqueeze(-1)
            front_force_feedback_physical = (
                front_feedback_factor
                * front_force_feedback_gain_t.unsqueeze(0)
                * front_force_feedback_scales_t
            )
            desired[:, :12] += (
                front_force_feedback_physical / scale[:12]
            )
            if physical_trajectory_correction is None:
                trajectory_correction_physical = torch.zeros(
                    (num_envs, 12),
                    dtype=torch.float32,
                    device=device,
                )
            else:
                trajectory_correction_physical = (
                    torch.as_tensor(
                        physical_trajectory_correction[step],
                        dtype=torch.float32,
                        device=device,
                    ).unsqueeze(0)
                    * trajectory_correction_scales_t
                )
            desired[:, :12] += (
                trajectory_correction_physical / scale[:12]
            )
            scheduled_offsets = offsets_t[:, :12].clone()
            both_front_desired = torch.all(
                desired_front_lookahead.bool()
            ).float()
            any_front_swing = 1.0 - both_front_desired
            front_offset_factor = torch.ones(
                (num_envs, 2),
                dtype=torch.float32,
                device=device,
            )
            both_mask = front_offset_schedule_codes_t == 1
            swing_mask = front_offset_schedule_codes_t == 2
            each_desired_mask = front_offset_schedule_codes_t == 3
            each_swing_mask = front_offset_schedule_codes_t == 4
            front_offset_factor[both_mask] = both_front_desired
            front_offset_factor[swing_mask] = any_front_swing
            front_offset_factor[each_desired_mask] = (
                desired_front_lookahead
            )
            front_offset_factor[each_swing_mask] = (
                1.0 - desired_front_lookahead
            )
            scheduled_offsets[:, [0, 4, 8]] *= (
                front_offset_factor[:, 0:1]
            )
            scheduled_offsets[:, [1, 5, 9]] *= (
                front_offset_factor[:, 1:2]
            )
            if step < args_cli.front_offset_start_frame:
                front_offset_activation = 0.0
            elif args_cli.front_offset_ramp_frames == 0:
                front_offset_activation = 1.0
            else:
                ramp_phase = min(
                    (
                        step
                        - args_cli.front_offset_start_frame
                        + 1
                    )
                    / args_cli.front_offset_ramp_frames,
                    1.0,
                )
                front_offset_activation = (
                    ramp_phase * ramp_phase * (3.0 - 2.0 * ramp_phase)
                )
            scheduled_offsets[:, list(FRONT_LEG_INDICES)] *= (
                front_offset_activation
            )
            desired[:, :12] += scheduled_offsets / scale[:12]
            desired[:, 12:].zero_()
            desired = torch.maximum(
                torch.minimum(
                    desired,
                    previous_action + maximum_raw_delta,
                ),
                previous_action - maximum_raw_delta,
            )
            desired = torch.maximum(
                torch.minimum(desired, raw_max),
                raw_min,
            )
            desired[:, 12:].zero_()
            _, _, terminated, truncated, _ = env.step(desired)
            done_any |= (
                torch.as_tensor(terminated, device=device).reshape(-1)
                | torch.as_tensor(truncated, device=device).reshape(-1)
            )
            previous_action = desired

            force_samples.append(
                adapter.contact_sensor.data.net_forces_w[
                    :,
                    adapter.contact_body_ids[:2],
                ]
                .detach()
                .cpu()
                .numpy()
            )
            anchor_position = (
                adapter.command.robot_anchor_pos_w
                - adapter.base.scene.env_origins
            )
            base_delta_samples.append(
                (anchor_position - initial_anchor_position)
                .detach()
                .cpu()
                .numpy()
            )
            quaternion_dot = torch.abs(
                torch.sum(
                    adapter.command.robot_anchor_quat_w
                    * initial_anchor_quaternion,
                    dim=-1,
                )
            ).clamp(max=1.0)
            orientation_error_samples.append(
                (2.0 * torch.acos(quaternion_dot)).detach().cpu().numpy()
            )
            lateral_velocity_samples.append(
                adapter.command.robot_anchor_lin_vel_w[:, 1]
                .detach()
                .cpu()
                .numpy()
            )
            action_samples.append(desired.detach().cpu().numpy())
            velocity_feedback_samples.append(
                velocity_feedback_physical.detach().cpu().numpy()
            )
            pitch_feedback_samples.append(
                pitch_feedback_physical.detach().cpu().numpy()
            )
            signed_pitch_error_samples.append(
                signed_pitch_error.detach().cpu().numpy()
            )
            front_force_feedback_samples.append(
                front_force_feedback_physical.detach().cpu().numpy()
            )
            trajectory_correction_samples.append(
                trajectory_correction_physical.detach().cpu().numpy()
            )

        force = np.abs(np.stack(force_samples)[..., 0])
        pre_step_force = np.abs(
            np.stack(pre_step_force_samples)[..., 0]
        )
        base_delta = np.stack(base_delta_samples)
        orientation_error = np.stack(orientation_error_samples)
        lateral_velocity = np.stack(lateral_velocity_samples)
        actions = np.stack(action_samples)
        velocity_feedback = np.stack(velocity_feedback_samples)
        pitch_feedback = np.stack(pitch_feedback_samples)
        signed_pitch_error = np.stack(signed_pitch_error_samples)
        front_force_feedback = np.stack(front_force_feedback_samples)
        trajectory_correction = np.stack(
            trajectory_correction_samples
        )
        base_pose = np.stack(base_pose_samples)
        base_twist = np.stack(base_twist_samples)
        joint_position = np.stack(joint_position_samples)
        joint_velocity = np.stack(joint_velocity_samples)
        wheel_pose = np.stack(wheel_pose_samples)
        wheel_twist = np.stack(wheel_twist_samples)
        contact_force = np.stack(contact_force_samples)
        measured_contact = np.stack(measured_contact_samples)
        desired_front = recorded_desired_contact[
            : args_cli.steps,
            :2,
        ]
        records = []
        for candidate_index, (
            blend,
            smoothing_window,
            action_frame_shift,
            action_frame_shift_mode,
            front_offset_schedule_mode,
            velocity_feedback_gain,
            velocity_feedback_vector_index,
            pitch_feedback_gain,
            pitch_feedback_vector_index,
            front_force_feedback_scale,
            trajectory_correction_scale,
            rate_limit,
            offsets,
            rear_thigh_offsets,
            front_hip_offsets,
            rear_hip_offsets,
            rear_calf_offsets,
        ) in enumerate(
            candidate_specs
        ):
            start = candidate_index * args_cli.replicates
            stop = start + args_cli.replicates
            candidate_force = force[:, start:stop]
            candidate_pre_step_force = pre_step_force[:, start:stop]
            desired = np.broadcast_to(
                desired_front[:, None, :],
                candidate_force.shape,
            )
            below = (candidate_force < 6.0) & desired
            pre_step_below = (
                candidate_pre_step_force < 6.0
            ) & desired
            per_wheel_below = np.sum(below, axis=(0, 1))
            per_wheel_desired = np.sum(desired, axis=(0, 1))
            per_replicate_below = np.sum(below, axis=(0, 2))
            per_replicate_desired = np.sum(desired, axis=(0, 2))
            desired_values = [
                candidate_force[..., wheel][desired[..., wheel]]
                for wheel in range(2)
            ]
            candidate_actions = actions[:, start:stop]
            candidate_velocity_feedback = velocity_feedback[:, start:stop]
            candidate_pitch_feedback = pitch_feedback[:, start:stop]
            candidate_signed_pitch_error = signed_pitch_error[:, start:stop]
            candidate_front_force_feedback = (
                front_force_feedback[:, start:stop]
            )
            candidate_trajectory_correction = (
                trajectory_correction[:, start:stop]
            )
            candidate_lateral_velocity = lateral_velocity[:, start:stop]
            candidate_actual_displacement = (
                base_delta[-1, start:stop, 1]
                - base_delta[0, start:stop, 1]
            )
            if abs(target_lateral_displacement) > 1.0e-4:
                signed_progress_per_replicate = (
                    candidate_actual_displacement
                    / target_lateral_displacement
                )
            else:
                signed_progress_per_replicate = np.ones(
                    args_cli.replicates,
                    dtype=np.float32,
                )
            physical_step = (
                np.diff(
                    candidate_actions[..., :12],
                    axis=0,
                    prepend=np.zeros_like(candidate_actions[:1, ..., :12]),
                )
                * contract.scale[:12]
            )
            formal_metrics_per_replicate = []
            formal_failures_per_replicate = []
            orientation_rotation_vectors = []
            for replicate_index in range(args_cli.replicates):
                env_index = start + replicate_index
                actual_quaternion = np.asarray(
                    base_pose[:, env_index, 3:7],
                    dtype=np.float64,
                )
                reference_quaternion = np.asarray(
                    reference.body_quat_w[
                        recorded_ref_frame[: args_cli.steps],
                        0,
                    ],
                    dtype=np.float64,
                )
                alignment_quaternion_np = _quat_multiply_np(
                    actual_quaternion[0],
                    _quat_conjugate_np(reference_quaternion[0]),
                )
                target_quaternion = _quat_multiply_np(
                    np.broadcast_to(
                        alignment_quaternion_np,
                        reference_quaternion.shape,
                    ),
                    reference_quaternion,
                )
                orientation_rotation_vectors.append(
                    _quat_rotation_vector_np(
                        actual_quaternion,
                        target_quaternion,
                    )
                )
                formal_metrics = compute_tracking_metrics(
                    {
                        "ref_frame": recorded_ref_frame[
                            : args_cli.steps
                        ],
                        "base_pose_w": base_pose[:, env_index],
                        "base_twist_w": base_twist[:, env_index],
                        "q": joint_position[:, env_index],
                        "dq": joint_velocity[:, env_index],
                        "wheel_body_pose_w": wheel_pose[:, env_index],
                        "wheel_body_twist_w": wheel_twist[:, env_index],
                        "contact_force_w": contact_force[:, env_index],
                        "desired_contact": recorded_desired_contact[
                            : args_cli.steps
                        ],
                        "measured_contact": measured_contact[:, env_index],
                        "scheduled_action16": candidate_actions[
                            :,
                            replicate_index,
                        ],
                        "executed_action16": candidate_actions[
                            :,
                            replicate_index,
                        ],
                    },
                    reference,
                    references,
                    contract,
                )
                formal_metrics_per_replicate.append(formal_metrics)
                formal_failures_per_replicate.append(
                    tracking_threshold_failures(
                        formal_metrics,
                        tracking_thresholds,
                    )
                )
            formal_metric_means = {
                key: np.mean(
                    np.asarray(
                        [
                            metrics[key]
                            for metrics in formal_metrics_per_replicate
                        ],
                        dtype=np.float64,
                    ),
                    axis=0,
                ).tolist()
                for key in formal_metrics_per_replicate[0]
            }
            formal_metric_means = {
                key: (
                    float(value)
                    if not isinstance(value, list)
                    else value
                )
                for key, value in formal_metric_means.items()
            }
            orientation_rotation_vectors_np = np.stack(
                orientation_rotation_vectors,
                axis=1,
            )
            formal_failure_union = sorted(
                {
                    failure
                    for failures in formal_failures_per_replicate
                    for failure in failures
                }
            )
            records.append(
                {
                    "action_blend": blend,
                    "gait_smoothing_window": smoothing_window,
                    "action_frame_shift_frames": action_frame_shift,
                    "action_frame_shift_mode": action_frame_shift_mode,
                    "front_offset_schedule_mode": (
                        front_offset_schedule_mode
                    ),
                    "front_offset_start_frame": (
                        args_cli.front_offset_start_frame
                    ),
                    "front_offset_ramp_frames": (
                        args_cli.front_offset_ramp_frames
                    ),
                    "lateral_velocity_feedback_gain": (
                        velocity_feedback_gain
                    ),
                    "lateral_velocity_feedback_vector_index": (
                        velocity_feedback_vector_index
                    ),
                    "lateral_velocity_feedback_vector_leg": (
                        velocity_feedback_vectors[
                            velocity_feedback_vector_index
                        ].tolist()
                    ),
                    "lateral_velocity_feedback_max_abs_rad": float(
                        np.max(np.abs(candidate_velocity_feedback))
                    ),
                    "pitch_feedback_gain": pitch_feedback_gain,
                    "pitch_feedback_axis": args_cli.pitch_feedback_axis,
                    "pitch_feedback_start_frame": (
                        args_cli.pitch_feedback_start_frame
                    ),
                    "pitch_feedback_vector_index": (
                        pitch_feedback_vector_index
                    ),
                    "pitch_feedback_vector_leg": (
                        pitch_feedback_vectors[
                            pitch_feedback_vector_index
                        ].tolist()
                    ),
                    "pitch_feedback_max_abs_rad": float(
                        np.max(np.abs(candidate_pitch_feedback))
                    ),
                    "signed_pitch_error_rmse_rad": float(
                        np.sqrt(
                            np.mean(candidate_signed_pitch_error ** 2)
                        )
                    ),
                    "front_force_feedback_scale": (
                        front_force_feedback_scale
                    ),
                    "front_force_feedback_max_abs_rad": float(
                        np.max(np.abs(candidate_front_force_feedback))
                    ),
                    "trajectory_correction_scale": (
                        trajectory_correction_scale
                    ),
                    "trajectory_correction_max_abs_rad": float(
                        np.max(np.abs(candidate_trajectory_correction))
                    ),
                    "physical_target_rate_limit_rad_s": rate_limit,
                    "offsets_rad": {
                        "FL_thigh": offsets[0],
                        "FR_thigh": offsets[1],
                        "FL_calf": offsets[2],
                        "FR_calf": offsets[3],
                        "RL_thigh": rear_thigh_offsets[0],
                        "RR_thigh": rear_thigh_offsets[1],
                        "FL_hip": front_hip_offsets[0],
                        "FR_hip": front_hip_offsets[1],
                        "RL_hip": rear_hip_offsets[0],
                        "RR_hip": rear_hip_offsets[1],
                        "RL_calf": rear_calf_offsets[0],
                        "RR_calf": rear_calf_offsets[1],
                    },
                    "front_normal_below_6n_fraction_when_desired": float(
                        np.sum(below) / max(np.sum(desired), 1)
                    ),
                    "front_normal_below_6n_fraction_when_desired_pre_step": float(
                        np.sum(pre_step_below)
                        / max(np.sum(desired), 1)
                    ),
                    "front_normal_below_6n_count": (
                        per_wheel_below.astype(int).tolist()
                    ),
                    "front_normal_desired_count": (
                        per_wheel_desired.astype(int).tolist()
                    ),
                    "front_normal_below_6n_fraction_per_replicate": (
                        per_replicate_below
                        / np.maximum(per_replicate_desired, 1)
                    ).tolist(),
                    "front_normal_below_6n_count_per_replicate": (
                        per_replicate_below.astype(int).tolist()
                    ),
                    "front_normal_desired_count_per_replicate": (
                        per_replicate_desired.astype(int).tolist()
                    ),
                    "front_normal_force_mean_n": [
                        float(np.mean(value)) for value in desired_values
                    ],
                    "front_normal_force_p10_n": [
                        float(np.quantile(value, 0.10))
                        for value in desired_values
                    ],
                    "base_position_delta_max_abs_m": np.max(
                        np.abs(base_delta[:, start:stop]),
                        axis=(0, 1),
                    ).tolist(),
                    "base_position_delta_final_mean_m": np.mean(
                        base_delta[-1, start:stop],
                        axis=0,
                    ).tolist(),
                    "base_orientation_rmse_rad": float(
                        np.sqrt(
                            np.mean(
                                orientation_error[:, start:stop] ** 2
                            )
                        )
                    ),
                    "orientation_rotation_vector_rmse_rad": np.sqrt(
                        np.mean(
                            orientation_rotation_vectors_np ** 2,
                            axis=(0, 1),
                        )
                    ).tolist(),
                    "orientation_rotation_vector_mean_rad": np.mean(
                        orientation_rotation_vectors_np,
                        axis=(0, 1),
                    ).tolist(),
                    "lateral_velocity_mae_m_s": float(
                        np.mean(
                            np.abs(
                                candidate_lateral_velocity
                                - target_lateral_velocity[:, None]
                            )
                        )
                    ),
                    "lateral_velocity_mean_m_s": float(
                        np.mean(candidate_lateral_velocity)
                    ),
                    "target_lateral_velocity_mean_m_s": float(
                        np.mean(target_lateral_velocity)
                    ),
                    "actual_lateral_displacement_m": float(
                        np.mean(candidate_actual_displacement)
                    ),
                    "target_lateral_displacement_m": (
                        target_lateral_displacement
                    ),
                    "signed_lateral_progress_ratio": float(
                        np.mean(signed_progress_per_replicate)
                    ),
                    "signed_lateral_progress_ratio_per_replicate": (
                        signed_progress_per_replicate.tolist()
                    ),
                    "physical_leg_target_step_max_rad": float(
                        np.max(np.abs(physical_step))
                    ),
                    "formal_tracking_metrics_mean": formal_metric_means,
                    "formal_tracking_metrics_per_replicate": (
                        formal_metrics_per_replicate
                    ),
                    "formal_tracking_failures_union": (
                        formal_failure_union
                    ),
                    "formal_tracking_failures_per_replicate": (
                        formal_failures_per_replicate
                    ),
                    "formal_tracking_pass_all": bool(
                        not any(formal_failures_per_replicate)
                    ),
                    "terminated_any": bool(
                        done_any[start:stop].any().item()
                    ),
                    "wheel_action_exact_zero": bool(
                        np.array_equal(
                            candidate_actions[..., 12:],
                            np.zeros_like(
                                candidate_actions[..., 12:]
                            ),
                        )
                    ),
                }
            )

        ranked = sorted(
            records,
            key=lambda record: (
                record["terminated_any"],
                not record["formal_tracking_pass_all"],
                len(record["formal_tracking_failures_union"]),
                record["formal_tracking_metrics_mean"][
                    "base_orientation_rmse_rad"
                ],
                max(
                    record["formal_tracking_metrics_mean"][
                        "wheel_position_rmse_m"
                    ]
                ),
                max(
                    record["formal_tracking_metrics_mean"][
                        "rear_normal_force_p95_n"
                    ]
                ),
                record["formal_tracking_metrics_mean"][
                    "front_normal_below_6n_fraction_when_desired"
                ],
            ),
        )
        trace_path = None
        trace_sha256 = None
        if args_cli.trace_output is not None:
            trace_path = args_cli.trace_output.expanduser().resolve()
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                trace_path,
                executed_action16=actions,
                front_normal_force_n=force,
                pre_step_front_normal_force_n=pre_step_force,
                base_position_delta_m=base_delta,
                base_orientation_error_rad=orientation_error,
                base_lateral_velocity_m_s=lateral_velocity,
                signed_pitch_error_rad=signed_pitch_error,
                pitch_feedback_physical_rad=pitch_feedback,
                ref_frame=recorded_ref_frame[: args_cli.steps],
                desired_contact=recorded_desired_contact[
                    : args_cli.steps
                ],
                ref_id=np.asarray([args_cli.ref_id], dtype=np.int64),
                seed=np.asarray([args_cli.seed], dtype=np.int64),
                candidate_count=np.asarray(
                    [candidate_count],
                    dtype=np.int64,
                ),
                replicates=np.asarray(
                    [args_cli.replicates],
                    dtype=np.int64,
                ),
            )
            trace_sha256 = sha256_file(trace_path)
        report = {
            "schema_version": "pcbc-standing-action-offset-grid-v2",
            "status": "diagnostic_not_training_data",
            "task": args_cli.task,
            "reference_config": args_cli.reference_config,
            "tracking_config": str(tracking_config_path),
            "tracking_config_sha256": sha256_file(
                tracking_config_path
            ),
            "tracking_thresholds": tracking_thresholds,
            "source_episode": str(episode_path),
        "precondition_episodes": precondition_records,
        "action_reference": (
            str(action_reference_path)
            if action_reference_path is not None
            else None
        ),
        "blend_base_action_reference": (
            str(blend_base_action_reference_path)
            if blend_base_action_reference_path is not None
            else None
        ),
        "action_blend_values": list(blend_values),
        "gait_smoothing_window_values": list(smoothing_windows),
        "action_frame_shift_values": list(action_frame_shifts),
        "action_frame_shift_mode_values": list(
            action_frame_shift_modes
        ),
        "front_offset_schedule_mode_values": list(
            front_offset_schedule_modes
        ),
        "front_offset_start_frame": args_cli.front_offset_start_frame,
        "front_offset_ramp_frames": args_cli.front_offset_ramp_frames,
        "lateral_velocity_feedback_gain_values": list(
            velocity_feedback_gains
        ),
        "lateral_velocity_feedback_vectors_leg": (
            velocity_feedback_vectors.tolist()
        ),
        "lateral_velocity_feedback_max_rad": velocity_feedback_max_rad,
        "pitch_feedback_gain_values": list(
            pitch_feedback_gains
        ),
        "pitch_feedback_vectors_leg": (
            pitch_feedback_vectors.tolist()
        ),
        "pitch_feedback_max_abs_rad": pitch_feedback_max_abs_rad,
        "pitch_feedback_axis": args_cli.pitch_feedback_axis,
        "pitch_feedback_start_frame": (
            args_cli.pitch_feedback_start_frame
        ),
        "front_force_feedback_scale_values": list(
            front_force_feedback_scales
        ),
        "front_force_feedback_gain_leg": (
            front_force_feedback_gain.tolist()
        ),
        "front_force_feedback_target_n": front_force_feedback_target_n,
        "front_force_feedback_min_contact_n": (
            front_force_feedback_min_contact_n
        ),
        "trajectory_correction": (
            str(trajectory_correction_path)
            if trajectory_correction_path is not None
            else None
        ),
        "trajectory_correction_scale_values": list(
            trajectory_correction_scales
        ),
        "clear_contact_warm_start_before_step": bool(
            args_cli.clear_contact_warm_start_before_step
        ),
        "clone_env0_after_reset": bool(
            args_cli.clone_env0_after_reset
        ),
        "clear_contact_warm_start_after_reset": bool(
            args_cli.clear_contact_warm_start_after_reset
        ),
        "contact_prime_substeps": int(args_cli.contact_prime_substeps),
        "contact_restore_mode": args_cli.contact_restore_mode,
            "seed": args_cli.seed,
            "ref_id": args_cli.ref_id,
            "steps": args_cli.steps,
            "replicates": args_cli.replicates,
            "candidate_count": candidate_count,
            "num_envs": num_envs,
            "trace_output": (
                str(trace_path) if trace_path is not None else None
            ),
            "trace_sha256": trace_sha256,
            "offset_values_rad": list(values),
            "center_offsets_rad": center.tolist(),
            "rear_thigh_offset_values_rad": list(rear_values),
            "rear_thigh_center_offsets_rad": rear_center.tolist(),
            "front_hip_offset_values_rad": list(front_hip_values),
            "front_hip_center_offsets_rad": front_hip_center.tolist(),
            "rear_hip_offset_values_rad": list(rear_hip_values),
            "rear_hip_center_offsets_rad": rear_hip_center.tolist(),
            "rear_calf_offset_values_rad": list(rear_calf_values),
            "rear_calf_center_offsets_rad": rear_calf_center.tolist(),
            "physical_target_rate_limits_rad_s": list(rate_limits),
            "records": records,
            "top_candidates": ranked[:20],
        }
        write_json(args_cli.report, report)
        for rank, record in enumerate(ranked[:20], start=1):
            print({"rank": rank, **record}, flush=True)
        return report
    finally:
        adapter.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        write_json(
            args_cli.report.with_suffix(
                args_cli.report.suffix + ".failure.json"
            ),
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
