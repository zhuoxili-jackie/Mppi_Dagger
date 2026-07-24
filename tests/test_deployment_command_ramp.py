from __future__ import annotations

from lateral_mppi_dagger.env.isaac_adapter import (
    deployment_lateral_command_ramp_value,
)


def test_deployment_command_ramp_matches_first_key_both_directions() -> None:
    assert [
        deployment_lateral_command_ramp_value(0.03, frame, 0.012)
        for frame in range(4)
    ] == [0.0, 0.012, 0.024, 0.03]
    assert [
        deployment_lateral_command_ramp_value(-0.03, frame, 0.012)
        for frame in range(4)
    ] == [0.0, -0.012, -0.024, -0.03]
