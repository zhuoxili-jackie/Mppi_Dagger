from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import torch


class WheelActionMode(str, Enum):
    HARD_ZERO = "hard_zero"
    LEARNED_VELOCITY = "learned_velocity"


@dataclass(frozen=True)
class ActionContract:
    wheel_action_mode: WheelActionMode
    q_action_offset_runtime: np.ndarray
    qd_action_offset_runtime: np.ndarray
    scale: np.ndarray
    raw_min: np.ndarray
    raw_max: np.ndarray
    max_raw_delta_per_step: np.ndarray

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "ActionContract":
        action = config["action"] if "action" in config else config
        instance = cls(
            wheel_action_mode=WheelActionMode(action["wheel_action_mode"]),
            q_action_offset_runtime=np.asarray(action["q_action_offset_runtime"], dtype=np.float32),
            qd_action_offset_runtime=np.asarray(action["qd_action_offset_runtime"], dtype=np.float32),
            scale=np.asarray(action["scale"], dtype=np.float32),
            raw_min=np.asarray(action["raw_min"], dtype=np.float32),
            raw_max=np.asarray(action["raw_max"], dtype=np.float32),
            max_raw_delta_per_step=np.asarray(action["max_raw_delta_per_step"], dtype=np.float32),
        )
        instance.validate()
        return instance

    def validate(self) -> None:
        for name in (
            "q_action_offset_runtime",
            "scale",
            "raw_min",
            "raw_max",
            "max_raw_delta_per_step",
        ):
            value = getattr(self, name)
            if value.shape != (16,):
                raise ValueError(f"{name} must have shape (16,), got {value.shape}")
            if not np.isfinite(value).all():
                raise ValueError(f"{name} contains NaN or Inf")
        if self.qd_action_offset_runtime.shape != (4,):
            raise ValueError("qd_action_offset_runtime must have shape (4,)")
        if np.any(self.scale <= 0.0):
            raise ValueError("All action scales must be positive.")
        if np.any(self.raw_min > self.raw_max):
            raise ValueError("raw_min exceeds raw_max.")
        if self.wheel_action_mode is WheelActionMode.HARD_ZERO:
            if not np.array_equal(self.raw_min[12:], np.zeros(4, dtype=np.float32)):
                raise ValueError("hard_zero contract requires wheel raw_min == 0")
            if not np.array_equal(self.raw_max[12:], np.zeros(4, dtype=np.float32)):
                raise ValueError("hard_zero contract requires wheel raw_max == 0")


@dataclass(frozen=True)
class ShieldInfo:
    intervened: bool
    clip_delta16: np.ndarray
    failure_code: str


class Action16Adapter:
    """Only physical-target to raw-action conversion used by experts and validation."""

    def __init__(self, contract: ActionContract):
        self.contract = contract

    def physical_to_raw(
        self,
        q_des_leg: np.ndarray,
        wheel_vel_des: np.ndarray | None = None,
    ) -> np.ndarray:
        q_des = np.asarray(q_des_leg, dtype=np.float32)
        if q_des.shape[-1] != 12:
            raise ValueError(f"q_des_leg must end in 12 values, got {q_des.shape}")
        raw_leg = (
            q_des - self.contract.q_action_offset_runtime[:12]
        ) / self.contract.scale[:12]
        if self.contract.wheel_action_mode is WheelActionMode.HARD_ZERO:
            raw_wheel = np.zeros(q_des.shape[:-1] + (4,), dtype=np.float32)
        else:
            if wheel_vel_des is None:
                raise ValueError("learned_velocity mode requires wheel_vel_des")
            wheel = np.asarray(wheel_vel_des, dtype=np.float32)
            if wheel.shape != q_des.shape[:-1] + (4,):
                raise ValueError(f"wheel_vel_des shape mismatch: expected {q_des.shape[:-1] + (4,)}, got {wheel.shape}")
            raw_wheel = (
                wheel - self.contract.qd_action_offset_runtime
            ) / self.contract.scale[12:]
        result = np.concatenate((raw_leg, raw_wheel), axis=-1).astype(np.float32, copy=False)
        if not np.isfinite(result).all():
            raise ValueError("Converted raw action contains NaN or Inf.")
        return result

    def raw_to_physical(self, raw_action16: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raw = np.asarray(raw_action16, dtype=np.float32)
        if raw.shape[-1] != 16:
            raise ValueError(f"raw_action16 must end in 16 values, got {raw.shape}")
        if self.contract.wheel_action_mode is WheelActionMode.HARD_ZERO and not np.array_equal(
            raw[..., 12:], np.zeros_like(raw[..., 12:])
        ):
            raise ValueError("Non-zero wheel action supplied to a hard_zero contract.")
        q_des = self.contract.q_action_offset_runtime[:12] + self.contract.scale[:12] * raw[..., :12]
        wheel_vel = self.contract.qd_action_offset_runtime + self.contract.scale[12:] * raw[..., 12:]
        return q_des.astype(np.float32), wheel_vel.astype(np.float32)


class SafetyShield:
    def __init__(self, contract: ActionContract):
        self.contract = contract
        self._last_action = np.zeros(16, dtype=np.float32)

    @property
    def last_action(self) -> np.ndarray:
        return self._last_action.copy()

    def reset(self) -> None:
        self._last_action.fill(0.0)

    def apply(self, proposed_action16: np.ndarray) -> tuple[np.ndarray, ShieldInfo]:
        proposed = np.asarray(proposed_action16, dtype=np.float32)
        if proposed.shape != (16,):
            raise ValueError(f"SafetyShield expects one action with shape (16,), got {proposed.shape}")
        failure_code = ""
        if not np.isfinite(proposed).all():
            proposed = self._last_action.copy()
            proposed[12:] = 0.0
            failure_code = "NAN_INF"
        bounded = np.clip(proposed, self.contract.raw_min, self.contract.raw_max)
        delta = bounded - self._last_action
        rate_limited = self._last_action + np.clip(
            delta,
            -self.contract.max_raw_delta_per_step,
            self.contract.max_raw_delta_per_step,
        )
        if self.contract.wheel_action_mode is WheelActionMode.HARD_ZERO:
            rate_limited[12:] = 0.0
        rate_limited = rate_limited.astype(np.float32, copy=False)
        clip_delta = rate_limited - np.asarray(proposed_action16, dtype=np.float32)
        # NumPy/Torch round-trips can move an otherwise in-range float32 value
        # by one ULP.  Record material clipping/rate limiting, not numerical
        # noise at ~1e-8.
        intervened = failure_code != "" or bool(np.any(np.abs(clip_delta) > 1.0e-6))
        self._last_action = rate_limited.copy()
        return rate_limited, ShieldInfo(intervened, clip_delta.astype(np.float32), failure_code)


def hard_zero_torch(action: torch.Tensor) -> torch.Tensor:
    if action.shape[-1] == 12:
        return torch.cat((action, torch.zeros_like(action[..., :4])), dim=-1)
    if action.shape[-1] != 16:
        raise ValueError(f"Expected 12D or 16D tensor, got {tuple(action.shape)}")
    return torch.cat((action[..., :12], torch.zeros_like(action[..., 12:])), dim=-1)
