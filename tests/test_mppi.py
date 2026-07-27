from __future__ import annotations

import pytest
import torch

from lateral_mppi_dagger.expert.mppi_expert import MPPIConfig, ReferenceCenteredMPPI


def test_sequence_projection_enforces_absolute_and_rate_bounds() -> None:
    sequence = torch.tensor(
        [
            [4.0] * 12,
            [-4.0] * 12,
            [0.5] * 12,
        ],
        dtype=torch.float32,
    )
    projected = ReferenceCenteredMPPI.project_sequence(
        sequence,
        raw_min=torch.full((12,), -1.0),
        raw_max=torch.full((12,), 1.0),
        previous_action=torch.zeros(12),
        max_delta=torch.full((12,), 0.25),
    )
    expected = torch.tensor([0.25, 0.0, 0.25], dtype=torch.float32)
    torch.testing.assert_close(projected[:, 0], expected)
    assert torch.max(torch.abs(projected[1:] - projected[:-1])).item() <= 0.25


def test_mppi_is_deterministic_and_improves_quadratic_cost() -> None:
    config = MPPIConfig(
        horizon=4,
        samples=128,
        iterations=3,
        temperature=0.2,
        temporal_smoothing=0.3,
        seed=7,
    )
    target = torch.full((config.horizon, 12), 0.6)

    def run_once() -> tuple[torch.Tensor, dict[str, float]]:
        optimizer = ReferenceCenteredMPPI(config, torch.full((12,), 0.5), device="cpu")
        return optimizer.optimize(
            torch.zeros_like(target),
            lambda candidates: torch.sum((candidates - target) ** 2, dim=(1, 2)),
            raw_min=torch.full((12,), -1.0),
            raw_max=torch.full((12,), 1.0),
            previous_action=torch.zeros(12),
            max_delta=torch.full((12,), 1.0),
        )

    first, first_diag = run_once()
    second, second_diag = run_once()
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
    assert first_diag == second_diag
    assert torch.sum((first - target) ** 2) < torch.sum(target**2)
    assert first_diag["effective_sample_size"] > 0.0


def test_episode_seed_reset_reproduces_sampling_stream() -> None:
    config = MPPIConfig(horizon=3, samples=16, iterations=1, seed=7)
    optimizer = ReferenceCenteredMPPI(config, torch.ones(12), device="cpu")
    optimizer.reset_seed(123)
    first = optimizer._correlated_noise()
    optimizer.reset_seed(123)
    second = optimizer._correlated_noise()
    optimizer.reset_seed(124)
    third = optimizer._correlated_noise()
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
    assert not torch.equal(first, third)


def test_reserved_proposal_slots_leave_balanced_stochastic_noise() -> None:
    config = MPPIConfig(horizon=3, samples=16, iterations=1, seed=7)
    optimizer = ReferenceCenteredMPPI(config, torch.ones(12), device="cpu")

    noise = optimizer._correlated_noise(sample_count=11)

    assert noise.shape == (11, 3, 12)
    assert torch.equal(noise[0], torch.zeros((3, 12)))
    torch.testing.assert_close(
        noise[1:6],
        -noise[6:11],
        rtol=0.0,
        atol=0.0,
    )


def test_best_sample_mode_returns_an_actually_evaluated_sequence() -> None:
    config = MPPIConfig(
        horizon=3,
        samples=32,
        iterations=2,
        temperature=0.5,
        selection_mode="best_sample",
        seed=11,
    )
    optimizer = ReferenceCenteredMPPI(
        config,
        torch.full((12,), 0.4),
        device="cpu",
    )
    target = torch.full((3, 12), 0.25)

    def cost(candidates: torch.Tensor) -> torch.Tensor:
        return torch.sum(torch.square(candidates - target), dim=(1, 2))

    sequence, diagnostics = optimizer.optimize(
        torch.zeros_like(target),
        cost,
        raw_min=torch.full((12,), -1.0),
        raw_max=torch.full((12,), 1.0),
    )
    assert torch.isclose(
        cost(sequence.unsqueeze(0))[0],
        torch.tensor(diagnostics["minimum_total_cost"]),
    )


def test_selection_mode_can_be_overridden_per_solve() -> None:
    config = MPPIConfig(
        horizon=3,
        samples=32,
        iterations=1,
        temperature=0.5,
        selection_mode="weighted",
        seed=19,
    )
    optimizer = ReferenceCenteredMPPI(
        config,
        torch.full((12,), 0.4),
        device="cpu",
    )
    target = torch.full((3, 12), 0.25)

    def cost(candidates: torch.Tensor) -> torch.Tensor:
        return torch.sum(torch.square(candidates - target), dim=(1, 2))

    sequence, diagnostics = optimizer.optimize(
        torch.zeros_like(target),
        cost,
        raw_min=torch.full((12,), -1.0),
        raw_max=torch.full((12,), 1.0),
        selection_mode="best_sample",
    )

    assert diagnostics["selection_mode"] == "best_sample"
    assert torch.isclose(
        cost(sequence.unsqueeze(0))[0],
        torch.tensor(diagnostics["minimum_total_cost"]),
    )


def test_temperature_can_be_overridden_per_solve() -> None:
    config = MPPIConfig(
        horizon=2,
        samples=4,
        iterations=1,
        temperature=0.5,
        seed=29,
    )
    optimizer = ReferenceCenteredMPPI(
        config,
        torch.zeros(12),
        device="cpu",
    )

    _, diagnostics = optimizer.optimize(
        torch.zeros((2, 12)),
        lambda candidates: torch.arange(
            candidates.shape[0],
            dtype=torch.float32,
        ),
        raw_min=torch.full((12,), -1.0),
        raw_max=torch.full((12,), 1.0),
        temperature=2.0,
    )

    assert diagnostics["temperature"] == 2.0


@pytest.mark.parametrize(
    "temperature",
    (0.0, -1.0, float("nan"), float("inf")),
)
def test_temperature_override_fails_closed(temperature: float) -> None:
    config = MPPIConfig(horizon=2, samples=4, iterations=1)
    optimizer = ReferenceCenteredMPPI(
        config,
        torch.zeros(12),
        device="cpu",
    )

    with pytest.raises(ValueError, match="finite and positive"):
        optimizer.optimize(
            torch.zeros((2, 12)),
            lambda candidates: torch.zeros(candidates.shape[0]),
            raw_min=torch.full((12,), -1.0),
            raw_max=torch.full((12,), 1.0),
            temperature=temperature,
        )


def test_structured_proposal_competes_under_the_same_projection_and_cost() -> None:
    config = MPPIConfig(
        horizon=3,
        samples=16,
        iterations=1,
        temperature=0.5,
        selection_mode="best_sample",
        seed=23,
    )
    optimizer = ReferenceCenteredMPPI(
        config,
        torch.zeros(12),
        device="cpu",
    )
    target = torch.full((3, 12), 0.4)
    proposals = torch.stack(
        (
            torch.full((3, 12), -0.4),
            target,
        )
    )

    sequence, diagnostics = optimizer.optimize(
        torch.zeros_like(target),
        lambda candidates: torch.sum(
            torch.square(candidates - target),
            dim=(1, 2),
        ),
        raw_min=torch.full((12,), -1.0),
        raw_max=torch.full((12,), 1.0),
        previous_action=torch.zeros(12),
        max_delta=torch.full((12,), 0.25),
        proposal_offsets=proposals,
    )

    expected = torch.tensor([0.25, 0.4, 0.4]).view(3, 1).repeat(1, 12)
    torch.testing.assert_close(sequence, expected)
    assert diagnostics["selected_best_sample_source"] == (
        "structured_proposal_1"
    )
    assert diagnostics["structured_proposal_count"] == 2


def test_structured_proposals_fail_closed_on_shape_count_and_finiteness() -> None:
    config = MPPIConfig(horizon=3, samples=4, iterations=1)
    optimizer = ReferenceCenteredMPPI(
        config,
        torch.ones(12),
        device="cpu",
    )
    arguments = {
        "nominal": torch.zeros((3, 12)),
        "rollout_cost": lambda candidates: torch.sum(
            candidates.square(),
            dim=(1, 2),
        ),
        "raw_min": torch.full((12,), -1.0),
        "raw_max": torch.full((12,), 1.0),
    }

    for proposals, message in (
        (torch.zeros((2, 2, 12)), "shape"),
        (torch.zeros((4, 3, 12)), "leave at least one"),
        (
            torch.full((1, 3, 12), float("nan")),
            "NaN or Inf",
        ),
    ):
        try:
            optimizer.optimize(
                **arguments,
                proposal_offsets=proposals,
            )
        except ValueError as exc:
            assert message in str(exc)
        else:
            raise AssertionError("Invalid structured proposals were accepted.")
