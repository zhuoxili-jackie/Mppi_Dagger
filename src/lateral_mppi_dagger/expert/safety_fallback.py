from __future__ import annotations

import time
from typing import Any

import numpy as np

from lateral_mppi_dagger.contract.action16 import Action16Adapter

from .base import Expert, ExpertReply, ExpertRequest, FailureCode


class SafetyFallbackExpert(Expert):
    def __init__(self, action_adapter: Action16Adapter, stable_q_leg: np.ndarray, blend: float = 0.15):
        self.action_adapter = action_adapter
        self.stable_q_leg = np.asarray(stable_q_leg, dtype=np.float32)
        if self.stable_q_leg.shape != (12,):
            raise ValueError("stable_q_leg must have shape (12,)")
        self.blend = float(blend)
        self._last_q_des = self.stable_q_leg.copy()

    def reset(self, episode_metadata: dict[str, Any]) -> None:
        self._last_q_des = self.stable_q_leg.copy()

    def act(self, request: ExpertRequest) -> ExpertReply:
        start = time.perf_counter()
        current_q = np.asarray(request.q[:12], dtype=np.float32)
        q_des = (1.0 - self.blend) * current_q + self.blend * self.stable_q_leg
        q_des = 0.5 * self._last_q_des + 0.5 * q_des
        self._last_q_des = q_des.copy()
        wheel = np.zeros(4, dtype=np.float32)
        action = self.action_adapter.physical_to_raw(q_des, wheel)
        return ExpertReply(
            valid=True,
            q_des_leg=q_des,
            wheel_vel_des=wheel,
            action16=action,
            tau_ff_leg=None,
            predicted_grf=None,
            solve_ms=(time.perf_counter() - start) * 1000.0,
            solver_status="SAFETY_LOWER_TO_STABLE_SUPPORT",
            safety_margin=0.0,
            source="fallback",
            failure_code=FailureCode.NONE,
        )

