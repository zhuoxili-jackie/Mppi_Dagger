#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from _bootstrap import ROOT

from lateral_mppi_dagger.reference.urdf_kinematics import (
    URDFKinematicTree,
    matrix_to_quaternion_wxyz,
    quaternion_wxyz_to_matrix,
)


BODY_ORDER = (
    "Base_link",
    "FL_hip_link",
    "FR_hip_link",
    "RL_hip_link",
    "RR_hip_link",
    "FL_thigh_link",
    "FR_thigh_link",
    "RL_thigh_link",
    "RR_thigh_link",
    "FL_calf_link",
    "FR_calf_link",
    "RL_calf_link",
    "RR_calf_link",
    "FL_foot_link",
    "FR_foot_link",
    "RL_foot_link",
    "RR_foot_link",
)
JOINT_ORDER = (
    "FL_hip_joint",
    "FR_hip_joint",
    "RL_hip_joint",
    "RR_hip_joint",
    "FL_thigh_joint",
    "FR_thigh_joint",
    "RL_thigh_joint",
    "RR_thigh_joint",
    "FL_calf_joint",
    "FR_calf_joint",
    "RL_calf_joint",
    "RR_calf_joint",
    "FL_foot_joint",
    "FR_foot_joint",
    "RL_foot_joint",
    "RR_foot_joint",
)
LEGS = ("FL", "FR", "RL", "RR")
LEG_INDICES = {
    leg: (LEGS.index(leg), 4 + LEGS.index(leg), 8 + LEGS.index(leg))
    for leg in LEGS
}
# One limb swings at a time: FL -> RL -> FR -> RR.
PHASE_OFFSET = {"FL": 0.75, "RL": 0.50, "FR": 0.25, "RR": 0.00}


def _smoothstep(value: np.ndarray | float) -> np.ndarray | float:
    return value * value * (3.0 - 2.0 * value)


def _portable_project_path(path: Path) -> str:
    """Keep in-project provenance valid when the standalone tree is moved."""
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _phase_relative_y(
    phase: float,
    stride_m: float,
    duty_factor: float,
) -> float:
    half_stance_travel = 0.5 * duty_factor * stride_m
    if phase < duty_factor:
        return half_stance_travel - stride_m * phase
    swing = (phase - duty_factor) / (1.0 - duty_factor)
    start = -half_stance_travel
    stop = half_stance_travel
    endpoint_derivative = -stride_m * (1.0 - duty_factor)
    h00 = 2.0 * swing**3 - 3.0 * swing**2 + 1.0
    h10 = swing**3 - 2.0 * swing**2 + swing
    h01 = -2.0 * swing**3 + 3.0 * swing**2
    h11 = swing**3 - swing**2
    return (
        h00 * start
        + h10 * endpoint_derivative
        + h01 * stop
        + h11 * endpoint_derivative
    )


def _swing_clearance(phase: float, duty_factor: float) -> float:
    if phase < duty_factor:
        return 0.0
    swing = (phase - duty_factor) / (1.0 - duty_factor)
    return float(np.sin(np.pi * swing) ** 2)


def _rear_preload(phase: float, duty_factor: float, lead_fraction: float) -> float:
    start = duty_factor - lead_fraction
    if phase < start:
        return 0.0
    window = 1.0 - start
    progress = (phase - start) / window
    return float(np.sin(np.pi * progress) ** 2)


def _integrated_velocity_profile(
    frames: int,
    fps: int,
    target_speed: float,
    acceleration_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    time = np.arange(frames, dtype=np.float64) / fps
    ramp = np.clip(time / acceleration_seconds, 0.0, 1.0)
    velocity = target_speed * _smoothstep(ramp)
    displacement = np.zeros(frames, dtype=np.float64)
    displacement[1:] = np.cumsum(
        0.5 * (velocity[1:] + velocity[:-1]) / fps
    )
    return displacement, velocity


def _angular_velocity(quaternions: np.ndarray, dt: float) -> np.ndarray:
    values = np.asarray(quaternions, dtype=np.float64).copy()
    for index in range(1, values.shape[0]):
        if np.sum(values[index - 1] * values[index]) < 0.0:
            values[index] *= -1.0
    result = np.zeros(values.shape[:-1] + (3,), dtype=np.float64)
    for frame in range(1, values.shape[0] - 1):
        previous = values[frame - 1]
        following = values[frame + 1]
        pw, px, py, pz = np.moveaxis(previous, -1, 0)
        inverse_previous = np.stack((pw, -px, -py, -pz), axis=-1)
        fw, fx, fy, fz = np.moveaxis(following, -1, 0)
        iw, ix, iy, iz = np.moveaxis(inverse_previous, -1, 0)
        delta = np.stack(
            (
                fw * iw - fx * ix - fy * iy - fz * iz,
                fw * ix + fx * iw + fy * iz - fz * iy,
                fw * iy - fx * iz + fy * iw + fz * ix,
                fw * iz + fx * iy - fy * ix + fz * iw,
            ),
            axis=-1,
        )
        delta = np.where(delta[..., :1] < 0.0, -delta, delta)
        vector_norm = np.linalg.norm(delta[..., 1:], axis=-1)
        active = vector_norm > 1.0e-12
        angle = 2.0 * np.arctan2(
            vector_norm,
            np.clip(delta[..., 0], -1.0, 1.0),
        )
        result[frame, active] = (
            delta[active, 1:]
            / vector_norm[active, None]
            * angle[active, None]
            / (2.0 * dt)
        )
    result[0] = 0.0
    result[-1] = result[-2]
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _joint_map(joint_position: np.ndarray) -> dict[str, float]:
    return {
        name: float(value)
        for name, value in zip(JOINT_ORDER, joint_position, strict=True)
    }


def _generate_motion(
    tree: URDFKinematicTree,
    source: dict[str, np.ndarray],
    target_vy: float,
    fps: int,
    stride_m: float,
    duty_factor: float,
    front_clearance_m: float,
    rear_clearance_m: float,
    rear_preload_x_m: float,
    rear_load_shift_y_m: float,
    acceleration_seconds: float,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    frames = int(source["joint_pos"].shape[0])
    q0 = np.asarray(source["joint_pos"][0], dtype=np.float64)
    root_position0 = np.asarray(source["body_pos_w"][0, 0], dtype=np.float64)
    root_quaternion = np.asarray(source["body_quat_w"][0, 0], dtype=np.float64)
    root_rotation = quaternion_wxyz_to_matrix(root_quaternion)
    initial_feet = {
        leg: np.asarray(
            source["body_pos_w"][0, BODY_ORDER.index(f"{leg}_foot_link")],
            dtype=np.float64,
        )
        for leg in LEGS
    }
    displacement, commanded_velocity = _integrated_velocity_profile(
        frames,
        fps,
        abs(target_vy),
        acceleration_seconds,
    )
    direction = float(np.sign(target_vy))

    joint_pos = np.zeros((frames, 16), dtype=np.float64)
    body_pos = np.zeros((frames, len(BODY_ORDER), 3), dtype=np.float64)
    body_quat = np.zeros((frames, len(BODY_ORDER), 4), dtype=np.float64)
    root_positions = np.zeros((frames, 3), dtype=np.float64)
    foot_targets = np.zeros((frames, 4, 3), dtype=np.float64)
    phases = np.zeros((frames, 4), dtype=np.float64)
    ik_residual_max = 0.0
    previous_q = q0.copy()

    for frame in range(frames):
        progress = displacement[frame]
        base_phase = (progress / stride_m) % 1.0
        leg_phase = {
            leg: (base_phase + PHASE_OFFSET[leg]) % 1.0
            for leg in LEGS
        }
        rear_pulses = {
            leg: _rear_preload(leg_phase[leg], duty_factor, 0.12)
            for leg in ("RL", "RR")
        }
        root_position = root_position0.copy()
        root_position[0] += rear_preload_x_m * max(rear_pulses.values())
        root_position[1] += direction * progress
        load_shift_gain = _smoothstep(
            min(progress / stride_m, 1.0)
        )
        # Shift toward the remaining rear support over the entire gait cycle,
        # rather than making a fast pulse only after rear-leg lift-off.
        root_position[1] += (
            rear_load_shift_y_m
            * load_shift_gain
            * np.cos(2.0 * np.pi * (base_phase + 0.10))
        )
        root_positions[frame] = root_position

        desired_feet: dict[str, np.ndarray] = {}
        for leg_index, leg in enumerate(LEGS):
            phase = leg_phase[leg]
            phases[frame, leg_index] = phase
            relative_now = _phase_relative_y(phase, stride_m, duty_factor)
            relative_initial = _phase_relative_y(
                PHASE_OFFSET[leg],
                stride_m,
                duty_factor,
            )
            target = initial_feet[leg].copy()
            target[1] += direction * (
                progress + relative_now - relative_initial
            )
            clearance = _swing_clearance(phase, duty_factor)
            if leg.startswith("F"):
                target[0] -= front_clearance_m * clearance
            else:
                target[2] += rear_clearance_m * clearance
            desired_feet[leg] = target
            foot_targets[frame, leg_index] = target

        current_q = previous_q.copy()
        base_inverse_rotation = root_rotation.T
        for leg in LEGS:
            indices = LEG_INDICES[leg]
            joint_names = tuple(
                JOINT_ORDER[index] for index in indices
            )
            target_base = base_inverse_rotation @ (
                desired_feet[leg] - root_position
            )
            lower = np.asarray(
                [tree.joint_by_name[name].lower for name in joint_names]
            )
            upper = np.asarray(
                [tree.joint_by_name[name].upper for name in joint_names]
            )

            def residual(candidate: np.ndarray) -> np.ndarray:
                trial = current_q.copy()
                trial[list(indices)] = candidate
                transform = tree.link_transform_base(
                    f"{leg}_foot_link",
                    _joint_map(trial),
                )
                return transform[:3, 3] - target_base

            if frame == 0:
                solution = q0[list(indices)]
                error = residual(solution)
            else:
                result = least_squares(
                    residual,
                    current_q[list(indices)],
                    bounds=(lower + 1.0e-6, upper - 1.0e-6),
                    xtol=1.0e-11,
                    ftol=1.0e-11,
                    gtol=1.0e-11,
                    max_nfev=60,
                )
                solution = result.x
                error = result.fun
            current_q[list(indices)] = solution
            ik_residual_max = max(
                ik_residual_max,
                float(np.max(np.abs(error))),
            )
        current_q[12:] = 0.0
        joint_pos[frame] = current_q
        previous_q = current_q

        transforms = tree.forward(
            _joint_map(current_q),
            root_position,
            root_quaternion,
        )
        for body_index, name in enumerate(BODY_ORDER):
            transform = transforms[name]
            body_pos[frame, body_index] = transform[:3, 3]
            body_quat[frame, body_index] = matrix_to_quaternion_wxyz(
                transform[:3, :3]
            )

    # Preserve the deployment handoff frame bit-for-bit.
    joint_pos[0] = source["joint_pos"][0]
    body_pos[0] = source["body_pos_w"][0]
    body_quat[0] = source["body_quat_w"][0]
    dt = 1.0 / fps
    joint_vel = np.gradient(joint_pos, dt, axis=0, edge_order=2)
    body_lin_vel = np.gradient(body_pos, dt, axis=0, edge_order=2)
    body_ang_vel = _angular_velocity(body_quat, dt)
    joint_vel[0] = 0.0
    body_lin_vel[0] = 0.0
    body_ang_vel[0] = 0.0

    front_indices = (0, 1)
    rear_indices = (2, 3)
    initial_targets = foot_targets[0]
    front_detachment = np.maximum(
        initial_targets[None, front_indices, 0]
        - foot_targets[:, front_indices, 0],
        0.0,
    )
    rear_clearance = np.maximum(
        foot_targets[:, rear_indices, 2]
        - initial_targets[None, rear_indices, 2],
        0.0,
    )
    outputs = {
        "fps": np.asarray([fps], dtype=np.int64),
        "joint_pos": joint_pos.astype(np.float32),
        "joint_vel": joint_vel.astype(np.float32),
        "body_pos_w": body_pos.astype(np.float32),
        "body_quat_w": body_quat.astype(np.float32),
        "body_lin_vel_w": body_lin_vel.astype(np.float32),
        "body_ang_vel_w": body_ang_vel.astype(np.float32),
    }
    metrics = {
        "target_vy_m_s": float(target_vy),
        "stride_m": float(stride_m),
        "nominal_cadence_hz": float(abs(target_vy) / stride_m),
        "duty_factor": float(duty_factor),
        "front_detachment_max_m": float(np.max(front_detachment)),
        "rear_clearance_max_m": float(np.max(rear_clearance)),
        "joint_velocity_max_abs_rad_s": float(np.max(np.abs(joint_vel[:, :12]))),
        "joint_step_max_abs_rad": float(np.max(np.abs(np.diff(joint_pos[:, :12], axis=0)))),
        "ik_residual_max_abs_m": float(ik_residual_max),
        "root_displacement_y_m": float(root_positions[-1, 1] - root_positions[0, 1]),
        "commanded_velocity_final_m_s": float(direction * commanded_velocity[-1]),
        "single_swing_fraction": float(
            np.mean(np.sum(phases >= duty_factor, axis=1) == 1)
        ),
        "multi_swing_fraction": float(
            np.mean(np.sum(phases >= duty_factor, axis=1) > 1)
        ),
    }
    return outputs, metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a kinematically coherent low-clearance, small-stride "
            "crawl reference. The original 708 assets are read-only seeds."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT
        / "vendor/robot_lab/data/Motions/pcbc_lateral_708/trajectory_trotting_acc_005.npz",
    )
    parser.add_argument(
        "--urdf",
        type=Path,
        default=ROOT
        / "vendor/robot_lab/data/Robots/pcbC/pcb_v2_description_0.88/urdf/pcb_v88.urdf",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "assets/references/low_load_v1",
    )
    parser.add_argument(
        "--target-speeds",
        type=float,
        nargs="+",
        default=(-0.06, -0.03, -0.024, -0.012, 0.012, 0.024, 0.03, 0.06),
    )
    parser.add_argument("--stride-m", type=float, default=0.04)
    parser.add_argument("--duty-factor", type=float, default=0.80)
    parser.add_argument("--front-clearance-m", type=float, default=0.008)
    parser.add_argument("--rear-clearance-m", type=float, default=0.012)
    parser.add_argument("--rear-preload-x-m", type=float, default=0.008)
    parser.add_argument("--rear-load-shift-y-m", type=float, default=0.008)
    parser.add_argument("--acceleration-seconds", type=float, default=0.60)
    args = parser.parse_args()
    if not 0.75 < args.duty_factor < 1.0:
        parser.error("--duty-factor must be greater than 0.75 and below 1.")
    if args.stride_m <= 0.0:
        parser.error("--stride-m must be positive.")
    if any(speed == 0.0 for speed in args.target_speeds):
        parser.error("Moving target speeds must be non-zero.")

    with np.load(args.source.resolve(), allow_pickle=False) as archive:
        source = {name: np.asarray(archive[name]) for name in archive.files}
    fps = int(source["fps"][0])
    tree = URDFKinematicTree(args.urdf.resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for target_vy in args.target_speeds:
        sign = "p" if target_vy > 0.0 else "n"
        millimeters = int(round(abs(target_vy) * 1000.0))
        filename = f"low_load_{sign}{millimeters:03d}.npz"
        path = args.output / filename
        arrays, metrics = _generate_motion(
            tree,
            source,
            target_vy,
            fps,
            args.stride_m,
            args.duty_factor,
            args.front_clearance_m,
            args.rear_clearance_m,
            args.rear_preload_x_m,
            args.rear_load_shift_y_m,
            args.acceleration_seconds,
        )
        np.savez_compressed(path, **arrays)
        records.append(
            {
                "file": filename,
                "sha256": _sha256(path),
                "frames": int(arrays["joint_pos"].shape[0]),
                "fps": fps,
                **metrics,
            }
        )
    report = {
        "schema_version": "pcbc-low-load-reference-generation-v1",
        "status": "kinematic_seed_requires_isaac_force_validation",
        "source": _portable_project_path(args.source),
        "source_sha256": _sha256(args.source.resolve()),
        "urdf": _portable_project_path(args.urdf),
        "urdf_sha256": _sha256(args.urdf.resolve()),
        "parameters": {
            "stride_m": args.stride_m,
            "duty_factor": args.duty_factor,
            "front_clearance_m": args.front_clearance_m,
            "rear_clearance_m": args.rear_clearance_m,
            "rear_preload_x_m": args.rear_preload_x_m,
            "rear_load_shift_y_m": args.rear_load_shift_y_m,
            "acceleration_seconds": args.acceleration_seconds,
        },
        "references": records,
    }
    report_path = args.output / "generation_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
