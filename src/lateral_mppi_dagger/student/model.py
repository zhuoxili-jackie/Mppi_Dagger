from __future__ import annotations

from typing import Any

import torch
from torch import nn

from lateral_mppi_dagger.contract.action16 import WheelActionMode


DEFAULT_ACTION_SCALE = (
    0.125,
    0.125,
    0.125,
    0.125,
    0.25,
    0.25,
    0.25,
    0.25,
    0.25,
    0.25,
    0.25,
    0.25,
    5.0,
    5.0,
    5.0,
    5.0,
)


def _activation(name: str) -> nn.Module:
    if name.lower() == "elu":
        return nn.ELU()
    if name.lower() == "relu":
        return nn.ReLU()
    if name.lower() == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation {name!r}")


class StudentPolicy(nn.Module):
    """Stateless deployment policy with embedded input normalization and wheel ABI."""

    def __init__(
        self,
        hidden_dims: tuple[int, ...] = (256, 256, 128),
        activation: str = "elu",
        wheel_action_mode: str = "hard_zero",
        zero_command_previous_action_deadband: float = 0.03,
        lateral_command_activation_start_m_s: float = 0.0,
        lateral_command_activation_full_m_s: float = 0.0,
        lateral_command_abs_limit_m_s: float = 0.0,
        physical_target_rate_limit_rad_s: float = 0.0,
        physical_target_abs_limit_rad: float = 0.0,
        physical_target_abs_limit_rad_by_joint: tuple[float, ...] | None = None,
        control_dt_s: float = 0.02,
        observation_mean: torch.Tensor | None = None,
        observation_std: torch.Tensor | None = None,
        raw_min: torch.Tensor | None = None,
        raw_max: torch.Tensor | None = None,
        action_scale: torch.Tensor | None = None,
    ):
        super().__init__()
        self.obs_dim = 93
        self.action_dim = 16
        self.wheel_action_mode = WheelActionMode(wheel_action_mode).value
        self.zero_command_previous_action_deadband = float(
            zero_command_previous_action_deadband
        )
        if self.zero_command_previous_action_deadband < 0.0:
            raise ValueError(
                "zero_command_previous_action_deadband must be non-negative."
            )
        self.lateral_command_activation_start_m_s = float(
            lateral_command_activation_start_m_s
        )
        self.lateral_command_activation_full_m_s = float(
            lateral_command_activation_full_m_s
        )
        self.lateral_command_abs_limit_m_s = float(
            lateral_command_abs_limit_m_s
        )
        self.physical_target_rate_limit_rad_s = float(
            physical_target_rate_limit_rad_s
        )
        self.physical_target_abs_limit_rad = float(
            physical_target_abs_limit_rad
        )
        self.control_dt_s = float(control_dt_s)
        scalar_parameters = {
            "lateral_command_activation_start_m_s": (
                self.lateral_command_activation_start_m_s
            ),
            "lateral_command_activation_full_m_s": (
                self.lateral_command_activation_full_m_s
            ),
            "lateral_command_abs_limit_m_s": (
                self.lateral_command_abs_limit_m_s
            ),
            "physical_target_rate_limit_rad_s": (
                self.physical_target_rate_limit_rad_s
            ),
            "physical_target_abs_limit_rad": (
                self.physical_target_abs_limit_rad
            ),
        }
        if any(value < 0.0 for value in scalar_parameters.values()):
            raise ValueError(
                "Command activation and physical safety parameters must be "
                "non-negative."
            )
        if self.control_dt_s <= 0.0:
            raise ValueError("control_dt_s must be strictly positive.")
        activation_enabled = self.lateral_command_activation_full_m_s > 0.0
        if activation_enabled and (
            self.lateral_command_activation_full_m_s
            <= self.lateral_command_activation_start_m_s
        ):
            raise ValueError(
                "lateral_command_activation_full_m_s must exceed the start "
                "threshold when continuous activation is enabled."
            )
        if (
            self.lateral_command_abs_limit_m_s > 0.0
            and activation_enabled
            and self.lateral_command_abs_limit_m_s
            < self.lateral_command_activation_full_m_s
        ):
            raise ValueError(
                "lateral_command_abs_limit_m_s must not be below the full "
                "activation threshold."
            )
        output_dim = 12 if self.wheel_action_mode == WheelActionMode.HARD_ZERO.value else 16
        dimensions = (self.obs_dim,) + tuple(int(value) for value in hidden_dims) + (output_dim,)
        layers: list[nn.Module] = []
        for input_dim, output_dim_layer in zip(dimensions[:-2], dimensions[1:-1], strict=True):
            layers.extend((nn.Linear(input_dim, output_dim_layer), _activation(activation)))
        layers.append(nn.Linear(dimensions[-2], dimensions[-1]))
        self.network = nn.Sequential(*layers)

        mean = torch.zeros(self.obs_dim, dtype=torch.float32) if observation_mean is None else observation_mean.float()
        std = torch.ones(self.obs_dim, dtype=torch.float32) if observation_std is None else observation_std.float()
        if mean.shape != (self.obs_dim,) or std.shape != (self.obs_dim,):
            raise ValueError("observation_mean and observation_std must both have shape (93,)")
        if torch.any(std <= 0.0):
            raise ValueError("observation_std must be strictly positive.")
        self.register_buffer("observation_mean", mean)
        self.register_buffer("observation_std", std)

        default_min = torch.full((16,), -100.0, dtype=torch.float32)
        default_max = torch.full((16,), 100.0, dtype=torch.float32)
        minimum = default_min if raw_min is None else raw_min.float()
        maximum = default_max if raw_max is None else raw_max.float()
        if minimum.shape != (16,) or maximum.shape != (16,):
            raise ValueError("raw_min and raw_max must both have shape (16,)")
        self.register_buffer("raw_min", minimum)
        self.register_buffer("raw_max", maximum)
        scale = (
            torch.tensor(DEFAULT_ACTION_SCALE, dtype=torch.float32)
            if action_scale is None
            else action_scale.float()
        )
        if scale.shape != (16,) or torch.any(scale <= 0.0):
            raise ValueError("action_scale must have shape (16,) and be positive.")
        self.register_buffer("action_scale", scale)
        if physical_target_abs_limit_rad_by_joint is None:
            absolute_limit = torch.full(
                (12,),
                self.physical_target_abs_limit_rad,
                dtype=torch.float32,
            )
        else:
            absolute_limit = torch.as_tensor(
                physical_target_abs_limit_rad_by_joint,
                dtype=torch.float32,
            )
        if absolute_limit.shape != (12,) or torch.any(absolute_limit < 0.0):
            raise ValueError(
                "physical_target_abs_limit_rad_by_joint must contain twelve "
                "non-negative values."
            )
        if torch.any(absolute_limit > 0.0) and torch.any(absolute_limit <= 0.0):
            raise ValueError(
                "Per-joint physical target limits must be either all positive "
                "or all disabled."
            )
        self.register_buffer(
            "physical_target_abs_limit_by_joint",
            absolute_limit,
        )
        self.physical_target_abs_limit_enabled = bool(
            torch.any(absolute_limit > 0.0).item()
        )

        self.hidden_dims = tuple(int(value) for value in hidden_dims)
        self.activation_name = activation

    def set_normalization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        if mean.shape != (93,) or std.shape != (93,):
            raise ValueError("Normalization tensors must both have shape (93,)")
        if torch.any(std <= 0.0):
            raise ValueError("Normalization std must be strictly positive.")
        self.observation_mean.copy_(mean.to(self.observation_mean))
        self.observation_std.copy_(std.to(self.observation_std))

    def load_network_preserving_function(
        self,
        source_state: dict[str, torch.Tensor],
        source_observation_mean: torch.Tensor,
        source_observation_std: torch.Tensor,
    ) -> None:
        """Load actor weights and analytically transfer them to this normalizer."""
        network_state = {
            name: value for name, value in source_state.items() if name.startswith("network.")
        }
        result = self.load_state_dict(network_state, strict=False)
        expected_missing = {
            "observation_mean",
            "observation_std",
            "raw_min",
            "raw_max",
            "action_scale",
            "physical_target_abs_limit_by_joint",
        }
        if set(result.missing_keys) != expected_missing or result.unexpected_keys:
            raise ValueError(
                "Initialization checkpoint network state is incomplete: "
                f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
            )
        first = self.network[0]
        if not isinstance(first, nn.Linear):
            raise TypeError("Student network must start with a Linear layer.")
        old_mean = source_observation_mean.to(
            device=first.weight.device,
            dtype=first.weight.dtype,
        )
        old_std = source_observation_std.to(
            device=first.weight.device,
            dtype=first.weight.dtype,
        )
        new_mean = self.observation_mean.to(first.weight)
        new_std = self.observation_std.to(first.weight)
        if old_mean.shape != (93,) or old_std.shape != (93,):
            raise ValueError("Source observation normalization must have shape (93,).")
        if torch.any(old_std <= 0.0):
            raise ValueError("Source observation std must be strictly positive.")
        with torch.no_grad():
            old_weight = first.weight.detach().clone()
            old_bias = first.bias.detach().clone()
            first.weight.copy_(old_weight * (new_std / old_std).unsqueeze(0))
            first.bias.copy_(
                old_bias + old_weight @ ((new_mean - old_mean) / old_std)
            )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        raw_observation = observation.to(dtype=torch.float32)
        lateral_command = raw_observation[..., 90:91]
        if self.lateral_command_abs_limit_m_s > 0.0:
            lateral_command = torch.clamp(
                lateral_command,
                -self.lateral_command_abs_limit_m_s,
                self.lateral_command_abs_limit_m_s,
            )
            raw_observation = torch.cat(
                (
                    raw_observation[..., :90],
                    lateral_command,
                    raw_observation[..., 91:],
                ),
                dim=-1,
            )

        continuous_activation = (
            self.lateral_command_activation_full_m_s > 0.0
        )
        movement_gate = torch.ones_like(lateral_command)
        if continuous_activation:
            activation_range = (
                self.lateral_command_activation_full_m_s
                - self.lateral_command_activation_start_m_s
            )
            activation = torch.clamp(
                (
                    torch.abs(lateral_command)
                    - self.lateral_command_activation_start_m_s
                )
                / activation_range,
                0.0,
                1.0,
            )
            # Cubic smoothstep has zero slope at both ends, avoiding the
            # previous hard transition exactly at the first key7 target.
            movement_gate = activation * activation * (3.0 - 2.0 * activation)
            previous = (
                raw_observation[..., 73:89] * movement_gate
            )
            raw_observation = torch.cat(
                (
                    raw_observation[..., :73],
                    previous,
                    raw_observation[..., 89:],
                ),
                dim=-1,
            )
        elif self.zero_command_previous_action_deadband > 0.0:
            # The key7 runtime performs three dry inference cycles before the
            # first action is applied, feeding each raw output back through
            # prev_action while the robot state is unchanged.  No such loop
            # exists in the collected transition data.  Below the task's
            # minimum moving command (0.03 m/s), mask that synthetic feedback
            # so the dry cycles are idempotent and standing remains driven by
            # measured state rather than an unobserved recurrent loop.
            command_max_abs = torch.amax(
                torch.abs(raw_observation[..., 89:92]),
                dim=-1,
                keepdim=True,
            )
            standing = (
                command_max_abs
                < self.zero_command_previous_action_deadband
            )
            previous = torch.where(
                standing,
                torch.zeros_like(raw_observation[..., 73:89]),
                raw_observation[..., 73:89],
            )
            raw_observation = torch.cat(
                (
                    raw_observation[..., :73],
                    previous,
                    raw_observation[..., 89:],
                ),
                dim=-1,
            )
        normalized = (
            raw_observation - self.observation_mean
        ) / self.observation_std
        predicted = self.network(normalized)
        if self.wheel_action_mode == "hard_zero":
            leg = torch.maximum(torch.minimum(predicted, self.raw_max[:12]), self.raw_min[:12])
            if continuous_activation:
                leg = leg * movement_gate
            physical_leg = leg * self.action_scale[:12]
            if self.physical_target_abs_limit_enabled:
                physical_leg = torch.maximum(
                    torch.minimum(
                        physical_leg,
                        self.physical_target_abs_limit_by_joint,
                    ),
                    -self.physical_target_abs_limit_by_joint,
                )
            if self.physical_target_rate_limit_rad_s > 0.0:
                previous_physical_leg = (
                    observation.to(dtype=torch.float32)[..., 73:85]
                    * self.action_scale[:12]
                )
                maximum_delta = (
                    self.physical_target_rate_limit_rad_s * self.control_dt_s
                )
                physical_leg = torch.maximum(
                    torch.minimum(
                        physical_leg,
                        previous_physical_leg + maximum_delta,
                    ),
                    previous_physical_leg - maximum_delta,
                )
                if self.physical_target_abs_limit_enabled:
                    physical_leg = torch.maximum(
                        torch.minimum(
                            physical_leg,
                            self.physical_target_abs_limit_by_joint,
                        ),
                        -self.physical_target_abs_limit_by_joint,
                    )
            leg = physical_leg / self.action_scale[:12]
            wheels = torch.zeros_like(leg[..., :4])
            return torch.cat((leg, wheels), dim=-1)
        return torch.maximum(torch.minimum(predicted, self.raw_max), self.raw_min)

    def specification(self) -> dict[str, Any]:
        return {
            "hidden_dims": list(self.hidden_dims),
            "activation": self.activation_name,
            "wheel_action_mode": self.wheel_action_mode,
            "zero_command_previous_action_deadband": (
                self.zero_command_previous_action_deadband
            ),
            "lateral_command_activation_start_m_s": (
                self.lateral_command_activation_start_m_s
            ),
            "lateral_command_activation_full_m_s": (
                self.lateral_command_activation_full_m_s
            ),
            "lateral_command_abs_limit_m_s": (
                self.lateral_command_abs_limit_m_s
            ),
            "physical_target_rate_limit_rad_s": (
                self.physical_target_rate_limit_rad_s
            ),
            "physical_target_abs_limit_rad": (
                self.physical_target_abs_limit_rad
            ),
            "physical_target_abs_limit_rad_by_joint": (
                self.physical_target_abs_limit_by_joint.detach()
                .cpu()
                .tolist()
            ),
            "control_dt_s": self.control_dt_s,
            "action_scale": self.action_scale.detach().cpu().tolist(),
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
        }


def build_student_from_checkpoint(
    checkpoint_path: str,
    map_location: str | torch.device = "cpu",
) -> tuple[StudentPolicy, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    specification = checkpoint["model_spec"]
    model = StudentPolicy(
        hidden_dims=tuple(specification["hidden_dims"]),
        activation=specification["activation"],
        wheel_action_mode=specification["wheel_action_mode"],
        zero_command_previous_action_deadband=float(
            specification.get(
                "zero_command_previous_action_deadband",
                0.03,
            )
        ),
        lateral_command_activation_start_m_s=float(
            specification.get("lateral_command_activation_start_m_s", 0.0)
        ),
        lateral_command_activation_full_m_s=float(
            specification.get("lateral_command_activation_full_m_s", 0.0)
        ),
        lateral_command_abs_limit_m_s=float(
            specification.get("lateral_command_abs_limit_m_s", 0.0)
        ),
        physical_target_rate_limit_rad_s=float(
            specification.get("physical_target_rate_limit_rad_s", 0.0)
        ),
        physical_target_abs_limit_rad=float(
            specification.get("physical_target_abs_limit_rad", 0.0)
        ),
        physical_target_abs_limit_rad_by_joint=tuple(
            specification["physical_target_abs_limit_rad_by_joint"]
        )
        if "physical_target_abs_limit_rad_by_joint" in specification
        else None,
        control_dt_s=float(specification.get("control_dt_s", 0.02)),
        observation_mean=checkpoint["observation_mean"],
        observation_std=checkpoint["observation_std"],
        raw_min=checkpoint["raw_min"],
        raw_max=checkpoint["raw_max"],
        action_scale=checkpoint.get(
            "action_scale",
            torch.as_tensor(
                specification.get("action_scale", DEFAULT_ACTION_SCALE),
                dtype=torch.float32,
            ),
        ),
    )
    load_result = model.load_state_dict(checkpoint["model_state"], strict=False)
    allowed_missing = {
        "action_scale",
        "physical_target_abs_limit_by_joint",
    }
    if (
        set(load_result.missing_keys) - allowed_missing
        or load_result.unexpected_keys
    ):
        raise ValueError(
            "Student checkpoint state is incomplete or incompatible: "
            f"missing={load_result.missing_keys}, "
            f"unexpected={load_result.unexpected_keys}"
        )
    model.eval()
    return model, checkpoint
