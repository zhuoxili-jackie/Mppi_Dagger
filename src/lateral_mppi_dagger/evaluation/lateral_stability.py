from __future__ import annotations

from collections.abc import Callable

import numpy as np


FOOT_NAMES = ("FL", "FR", "RL", "RR")


def dominant_leg_frequency_hz(
    joint_position: np.ndarray,
    fps: float,
    *,
    start_frame: int = 50,
    stop_frame: int = 300,
) -> float:
    """Return the strongest non-DC frequency across the twelve leg joints."""

    values = np.asarray(joint_position, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 12:
        raise ValueError("joint_position must have shape [frames, >=12].")
    stop = min(int(stop_frame), values.shape[0])
    start = min(max(int(start_frame), 0), stop - 2)
    window = values[start:stop, :12]
    window = window - np.mean(window, axis=0, keepdims=True)
    spectrum = np.abs(np.fft.rfft(window, axis=0))
    spectrum[0] = 0.0
    frequencies = np.fft.rfftfreq(window.shape[0], d=1.0 / float(fps))
    frequency_index = np.unravel_index(
        int(np.argmax(spectrum)),
        spectrum.shape,
    )[0]
    return float(frequencies[frequency_index])


def reference_gait_metrics(
    *,
    joint_position: np.ndarray,
    joint_velocity: np.ndarray,
    body_position_w: np.ndarray,
    body_linear_velocity_w: np.ndarray,
    fps: float,
) -> dict:
    """Measure the 708 gait in contact-relevant world directions.

    The robot is almost vertical: front wheels contact the trunk approximately
    along world x, while rear wheels support the robot against gravity along
    world z.  A generic relative-z range would therefore hide the important
    rear-wheel ground clearance.
    """

    joint_position = np.asarray(joint_position, dtype=np.float64)
    joint_velocity = np.asarray(joint_velocity, dtype=np.float64)
    body_position_w = np.asarray(body_position_w, dtype=np.float64)
    body_linear_velocity_w = np.asarray(
        body_linear_velocity_w,
        dtype=np.float64,
    )
    if body_position_w.ndim != 3 or body_position_w.shape[1:] != (17, 3):
        raise ValueError("body_position_w must have shape [frames,17,3].")
    if body_linear_velocity_w.shape != body_position_w.shape:
        raise ValueError("body_linear_velocity_w shape must match body_position_w.")
    if joint_position.shape != joint_velocity.shape:
        raise ValueError("joint position/velocity shapes must match.")

    base = body_position_w[:, 0]
    feet = body_position_w[:, 13:17]
    relative = feet - base[:, None, :]
    front_x_floor = np.quantile(feet[:, :2, 0], 0.90, axis=0)
    front_detachment = np.maximum(
        front_x_floor[None, :] - feet[:, :2, 0],
        0.0,
    )
    rear_z_floor = np.quantile(feet[:, 2:, 2], 0.10, axis=0)
    rear_lift = np.maximum(
        feet[:, 2:, 2] - rear_z_floor[None, :],
        0.0,
    )
    front_clearance = np.max(front_detachment, axis=0)
    rear_clearance = np.max(rear_lift, axis=0)
    front_air_fraction = np.mean(
        front_detachment > 0.010,
        axis=0,
    )
    rear_air_fraction = np.mean(
        rear_lift > 0.015,
        axis=0,
    )
    base_vy = body_linear_velocity_w[:, 0, 1]
    return {
        "frames": int(body_position_w.shape[0]),
        "fps": float(fps),
        "duration_s": float(body_position_w.shape[0] / fps),
        "dominant_leg_frequency_hz": dominant_leg_frequency_hz(
            joint_position,
            fps,
        ),
        "base_lateral_displacement_m": float(base[-1, 1] - base[0, 1]),
        "base_lateral_velocity_mean_m_s": float(np.mean(base_vy)),
        "base_lateral_velocity_max_abs_m_s": float(np.max(np.abs(base_vy))),
        "front_trunk_detachment_max_m": front_clearance.tolist(),
        "rear_ground_clearance_max_m": rear_clearance.tolist(),
        "front_detachment_fraction_over_10mm": front_air_fraction.tolist(),
        "rear_air_fraction_over_15mm": rear_air_fraction.tolist(),
        "foot_lateral_range_relative_base_m": np.ptp(
            relative[:, :, 1],
            axis=0,
        ).tolist(),
        "leg_joint_range_rad": np.ptp(
            joint_position[:, :12],
            axis=0,
        ).tolist(),
        "leg_joint_velocity_max_abs_rad_s": float(
            np.max(np.abs(joint_velocity[:, :12]))
        ),
    }


def contact_force_metrics(
    contact_force_w: np.ndarray,
    measured_contact: np.ndarray,
) -> dict:
    force = np.linalg.norm(
        np.asarray(contact_force_w, dtype=np.float64),
        axis=-1,
    )
    contact = np.asarray(measured_contact, dtype=bool)
    if force.ndim != 2 or force.shape[1] != 4:
        raise ValueError("contact_force_w must have shape [steps,4,3].")
    if contact.shape != force.shape:
        raise ValueError("measured_contact must have shape [steps,4].")

    rear_single_support = {}
    for index in (2, 3):
        opposite = 5 - index
        selected = contact[:, index] & ~contact[:, opposite]
        rear_single_support[FOOT_NAMES[index]] = {
            "fraction": float(np.mean(selected)),
            "support_force_mean_n": (
                float(np.mean(force[selected, index]))
                if np.any(selected)
                else None
            ),
            "support_force_p95_n": (
                float(np.quantile(force[selected, index], 0.95))
                if np.any(selected)
                else None
            ),
        }
    return {
        "force_mean_n": np.mean(force, axis=0).tolist(),
        "force_median_n": np.median(force, axis=0).tolist(),
        "force_p95_n": np.quantile(force, 0.95, axis=0).tolist(),
        "contact_fraction": np.mean(contact, axis=0).tolist(),
        "rear_single_support": rear_single_support,
    }


def replay_fixed_state_velocity_ramp(
    observation: np.ndarray,
    policy: Callable[[np.ndarray], np.ndarray],
    action_scale: np.ndarray,
    *,
    target_lateral_velocity_m_s: float,
    acceleration_m_s2: float,
    control_dt_s: float,
    settle_steps: int = 10,
    command_steps: int = 20,
) -> dict:
    """Replay deployment command acceleration with the measured state frozen.

    This is deliberately not a dynamics simulation.  It isolates whether the
    one-step previous-action feedback can amplify while proprioception has not
    yet responded, which is exactly the dangerous regime during activation.
    """

    current = np.asarray(observation, dtype=np.float32).reshape(1, 93).copy()
    scale = np.asarray(action_scale, dtype=np.float32)
    if scale.shape != (16,):
        raise ValueError("action_scale must have shape (16,).")
    if acceleration_m_s2 <= 0.0 or control_dt_s <= 0.0:
        raise ValueError("acceleration and control_dt must be positive.")
    if settle_steps < 0 or command_steps < 1:
        raise ValueError("settle_steps must be non-negative and command_steps positive.")

    previous_action = np.zeros(16, dtype=np.float32)
    for _ in range(settle_steps):
        previous_action = np.asarray(
            policy(current),
            dtype=np.float32,
        ).reshape(16)
        current[0, 73:89] = previous_action

    records = []
    command = 0.0
    maximum_increment = acceleration_m_s2 * control_dt_s
    for step in range(1, command_steps + 1):
        difference = target_lateral_velocity_m_s - command
        command += float(
            np.clip(difference, -maximum_increment, maximum_increment)
        )
        current[0, 89:92] = (0.0, command, 0.0)
        action = np.asarray(policy(current), dtype=np.float32).reshape(16)
        physical = action * scale
        previous_physical = previous_action * scale
        records.append(
            {
                "step": step,
                "command_vy_m_s": command,
                "raw_leg_max_abs": float(np.max(np.abs(action[:12]))),
                "physical_leg_target_delta_from_default_max_abs_rad": float(
                    np.max(np.abs(physical[:12]))
                ),
                "physical_leg_target_step_change_max_abs_rad": float(
                    np.max(np.abs(physical[:12] - previous_physical[:12]))
                ),
                "wheel_action_max_abs": float(np.max(np.abs(action[12:]))),
            }
        )
        current[0, 73:89] = action
        previous_action = action
    return {
        "kind": "fixed_measured_state_previous_action_feedback_probe",
        "target_lateral_velocity_m_s": float(target_lateral_velocity_m_s),
        "acceleration_m_s2": float(acceleration_m_s2),
        "control_dt_s": float(control_dt_s),
        "settled_zero_action": previous_action.tolist()
        if not records
        else None,
        "records": records,
        "maximum_physical_leg_target_delta_from_default_rad": float(
            max(
                record[
                    "physical_leg_target_delta_from_default_max_abs_rad"
                ]
                for record in records
            )
        ),
        "maximum_physical_leg_target_step_change_rad": float(
            max(
                record["physical_leg_target_step_change_max_abs_rad"]
                for record in records
            )
        ),
    }
