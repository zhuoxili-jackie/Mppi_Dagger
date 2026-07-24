from __future__ import annotations

import numpy as np

from lateral_mppi_dagger.evaluation.lateral_stability import (
    contact_force_metrics,
    dominant_leg_frequency_hz,
    replay_fixed_state_velocity_ramp,
)


def test_dominant_leg_frequency_detects_known_signal() -> None:
    fps = 50.0
    time = np.arange(500, dtype=np.float64) / fps
    joint = np.zeros((500, 16), dtype=np.float64)
    joint[:, 3] = np.sin(2.0 * np.pi * 2.0 * time)
    assert abs(
        dominant_leg_frequency_hz(
            joint,
            fps,
            start_frame=0,
            stop_frame=500,
        )
        - 2.0
    ) < 1.0e-12


def test_contact_force_metrics_detects_single_rear_support() -> None:
    force = np.zeros((4, 4, 3), dtype=np.float32)
    force[:, 2, 2] = 100.0
    force[2:, 3, 2] = 50.0
    contact = np.zeros((4, 4), dtype=np.uint8)
    contact[:, 2] = 1
    contact[2:, 3] = 1
    metrics = contact_force_metrics(force, contact)
    assert metrics["rear_single_support"]["RL"]["fraction"] == 0.5
    assert metrics["rear_single_support"]["RL"]["support_force_mean_n"] == 100.0


def test_velocity_ramp_replays_previous_raw_action() -> None:
    observation = np.zeros((1, 93), dtype=np.float32)

    def policy(current: np.ndarray) -> np.ndarray:
        result = np.zeros(16, dtype=np.float32)
        result[0] = current[0, 73] + current[0, 90]
        return result

    result = replay_fixed_state_velocity_ramp(
        observation,
        policy,
        np.ones(16, dtype=np.float32),
        target_lateral_velocity_m_s=0.03,
        acceleration_m_s2=0.60,
        control_dt_s=0.02,
        settle_steps=0,
        command_steps=3,
    )
    np.testing.assert_allclose(
        [record["command_vy_m_s"] for record in result["records"]],
        [0.012, 0.024, 0.03],
        atol=1.0e-8,
    )
    np.testing.assert_allclose(
        [record["raw_leg_max_abs"] for record in result["records"]],
        [0.012, 0.036, 0.066],
        atol=1.0e-7,
    )
