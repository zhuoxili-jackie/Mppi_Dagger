from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch

from .base import BackendUnavailable, Expert, ExpertReply, ExpertRequest


@dataclass(frozen=True)
class MPPIConfig:
    horizon: int = 20
    samples: int = 256
    iterations: int = 2
    temperature: float = 1.0
    temporal_smoothing: float = 0.70
    warm_start: bool = True
    reference_action_lookahead_steps: int = 1
    selection_mode: str = "weighted"
    seed: int = 42


class ReferenceCenteredMPPI:
    """Backend-agnostic MPPI optimizer around a future reference raw-action sequence."""

    def __init__(self, config: MPPIConfig, noise_std: torch.Tensor, device: str = "cuda"):
        self.config = config
        self.device = torch.device(device)
        self.noise_std = noise_std.to(self.device, dtype=torch.float32)
        if self.noise_std.shape != (12,):
            raise ValueError("MPPI noise_std must have shape (12,)")
        if config.horizon < 1 or config.samples < 2 or config.iterations < 1:
            raise ValueError("MPPI horizon/iterations must be positive and samples must be at least two.")
        if config.temperature <= 0.0:
            raise ValueError("MPPI temperature must be positive.")
        if not 0.0 <= config.temporal_smoothing < 1.0:
            raise ValueError("MPPI temporal_smoothing must be in [0,1).")
        if config.reference_action_lookahead_steps < 0:
            raise ValueError("MPPI reference_action_lookahead_steps must be non-negative.")
        if config.selection_mode not in {"weighted", "best_sample"}:
            raise ValueError(
                "MPPI selection_mode must be 'weighted' or 'best_sample'."
            )
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(config.seed)

    def reset_seed(self, seed: int) -> None:
        """Start an episode-specific, reproducible sampling stream."""
        self.generator.manual_seed(int(seed))

    def _correlated_noise(self) -> torch.Tensor:
        # Use antithetic pairs so a small rollout batch has exactly zero
        # perturbation mean instead of injecting a random control bias.
        pair_count = (self.config.samples - 1) // 2
        paired = torch.randn(
            pair_count,
            self.config.horizon,
            12,
            generator=self.generator,
            device=self.device,
        )
        paired *= self.noise_std.view(1, 1, 12)
        alpha = self.config.temporal_smoothing
        for step in range(1, self.config.horizon):
            paired[:, step] = alpha * paired[:, step - 1] + (1.0 - alpha) * paired[:, step]
        zero_count = self.config.samples - 2 * pair_count
        zeros = torch.zeros(
            zero_count,
            self.config.horizon,
            12,
            device=self.device,
        )
        return torch.cat((zeros, paired, -paired), dim=0)

    @staticmethod
    def project_sequence(
        sequence: torch.Tensor,
        raw_min: torch.Tensor,
        raw_max: torch.Tensor,
        previous_action: torch.Tensor | None = None,
        max_delta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Project [...,H,12] actions onto absolute and sequential rate bounds."""
        if sequence.shape[-1] != 12:
            raise ValueError("MPPI sequence must end in 12 leg actions.")
        lower = raw_min.to(sequence).reshape((1,) * (sequence.ndim - 2) + (12,))
        upper = raw_max.to(sequence).reshape((1,) * (sequence.ndim - 2) + (12,))
        projected = torch.clamp(sequence, lower, upper)
        if (previous_action is None) != (max_delta is None):
            raise ValueError("previous_action and max_delta must either both be supplied or both omitted.")
        if previous_action is None:
            return projected
        previous = previous_action.to(sequence).reshape(sequence.shape[:-2] + (12,))
        delta_bound = max_delta.to(sequence).reshape((1,) * (sequence.ndim - 2) + (12,))
        steps = []
        for step in range(sequence.shape[-2]):
            current = torch.maximum(
                torch.minimum(projected[..., step, :], previous + delta_bound),
                previous - delta_bound,
            )
            current = torch.clamp(current, lower, upper)
            steps.append(current)
            previous = current
        return torch.stack(steps, dim=-2)

    def optimize(
        self,
        nominal: torch.Tensor,
        rollout_cost: Callable[[torch.Tensor], torch.Tensor],
        raw_min: torch.Tensor,
        raw_max: torch.Tensor,
        *,
        previous_action: torch.Tensor | None = None,
        max_delta: torch.Tensor | None = None,
        initial_sequence: torch.Tensor | None = None,
        selection_mode: str | None = None,
    ) -> tuple[torch.Tensor, dict[str, float | str]]:
        if nominal.shape != (self.config.horizon, 12):
            raise ValueError(f"nominal must have shape {(self.config.horizon, 12)}, got {tuple(nominal.shape)}")
        raw_min = raw_min.to(self.device, dtype=torch.float32)
        raw_max = raw_max.to(self.device, dtype=torch.float32)
        previous_action = (
            None
            if previous_action is None
            else previous_action.to(self.device, dtype=torch.float32)
        )
        max_delta = (
            None if max_delta is None else max_delta.to(self.device, dtype=torch.float32)
        )
        start = nominal if initial_sequence is None else initial_sequence
        if start.shape != nominal.shape:
            raise ValueError("initial_sequence must have the same shape as nominal.")
        effective_selection_mode = (
            self.config.selection_mode
            if selection_mode is None
            else str(selection_mode)
        )
        if effective_selection_mode not in {"weighted", "best_sample"}:
            raise ValueError(
                "selection_mode must be 'weighted' or 'best_sample'."
            )
        sequence = self.project_sequence(
            start.to(self.device, dtype=torch.float32).clone(),
            raw_min,
            raw_max,
            previous_action,
            max_delta,
        )
        minimum_cost = float("inf")
        mean_cost = float("inf")
        effective_sample_size = 0.0
        best_candidate: torch.Tensor | None = None
        best_candidate_cost = float("inf")
        for _ in range(self.config.iterations):
            noise = self._correlated_noise()
            candidates = self.project_sequence(
                sequence.unsqueeze(0) + noise,
                raw_min,
                raw_max,
                None if previous_action is None else previous_action.expand(self.config.samples, -1),
                max_delta,
            )
            costs = rollout_cost(candidates)
            if costs.shape != (self.config.samples,) or not torch.isfinite(costs).all():
                raise RuntimeError("MPPI rollout_cost must return one finite cost per sample.")
            shifted = costs - costs.min()
            weights = torch.softmax(-shifted / self.config.temperature, dim=0)
            iteration_best_index = int(torch.argmin(costs).item())
            iteration_best_cost = float(costs[iteration_best_index].item())
            if iteration_best_cost < best_candidate_cost:
                best_candidate_cost = iteration_best_cost
                best_candidate = candidates[iteration_best_index].detach().clone()
            weighted = torch.sum(weights.view(-1, 1, 1) * candidates, dim=0)
            sequence = self.project_sequence(
                weighted,
                raw_min,
                raw_max,
                previous_action,
                max_delta,
            )
            minimum_cost = min(minimum_cost, float(costs.min().item()))
            mean_cost = float(costs.mean().item())
            effective_sample_size = float((1.0 / torch.sum(weights.square())).item())
        if effective_selection_mode == "best_sample":
            if best_candidate is None:
                raise RuntimeError("MPPI did not produce a best sampled candidate.")
            sequence = best_candidate
        return sequence, {
            "minimum_total_cost": minimum_cost,
            "mean_total_cost": mean_cost,
            "effective_sample_size": effective_sample_size,
            "selected_best_sample_cost": best_candidate_cost,
            "selection_mode": effective_selection_mode,
        }


class WholeBodyMPPIExpert(Expert):
    """Interface holder until an Isaac rollout-clone provider passes state-copy tests."""

    def __init__(self, provider: Callable[[ExpertRequest], ExpertReply] | None = None):
        self.provider = provider

    def reset(self, episode_metadata: dict[str, Any]) -> None:
        if self.provider is not None and hasattr(self.provider, "reset"):
            self.provider.reset(episode_metadata)

    def act(self, request: ExpertRequest) -> ExpertReply:
        if self.provider is None:
            raise BackendUnavailable(
                "Whole-Body MPPI is not an active label provider. Supply a validated Isaac rollout-clone "
                "provider after deterministic state-copy/restore tests pass."
            )
        reply = self.provider(request)
        reply.validate()
        if reply.source != "mppi":
            raise ValueError(f"MPPI provider returned unexpected source={reply.source!r}")
        return reply
