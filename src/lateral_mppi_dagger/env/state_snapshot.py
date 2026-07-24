from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class StateSnapshot:
    root_state_w: torch.Tensor
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    previous_executed_action: torch.Tensor
    reference_frame: torch.Tensor
    reference_id: torch.Tensor
    contact_history: torch.Tensor

    def clone(self) -> "StateSnapshot":
        return StateSnapshot(**{name: value.detach().clone() for name, value in self.__dict__.items()})

    def validate(self) -> None:
        if self.root_state_w.shape[-1] != 13:
            raise ValueError("root_state_w must end in 13 values")
        if self.joint_pos.shape != self.joint_vel.shape or self.joint_pos.shape[-1] != 16:
            raise ValueError("joint_pos and joint_vel must have identical [...,16] shape")
        if self.previous_executed_action.shape[-1] != 16:
            raise ValueError("previous_executed_action must end in 16 values")
        batch_shape = self.joint_pos.shape[:-1]
        if self.reference_frame.shape != batch_shape or self.reference_id.shape != batch_shape:
            raise ValueError("reference_frame/reference_id batch shape mismatch")
        for value in (self.root_state_w, self.joint_pos, self.joint_vel, self.previous_executed_action):
            if not torch.isfinite(value).all():
                raise ValueError("StateSnapshot contains NaN or Inf")

    def max_abs_difference(self, other: "StateSnapshot") -> float:
        self.validate()
        other.validate()
        maxima = []
        for name, value in self.__dict__.items():
            candidate = getattr(other, name)
            if value.shape != candidate.shape:
                return float("inf")
            if value.dtype == torch.bool or not torch.is_floating_point(value):
                maxima.append(0.0 if torch.equal(value, candidate) else float("inf"))
            elif value.numel() > 0:
                maxima.append(float(torch.max(torch.abs(value - candidate)).item()))
        return max(maxima, default=0.0)
