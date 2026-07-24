from __future__ import annotations

import time
from typing import Any

import numpy as np

from lateral_mppi_dagger.contract.action16 import Action16Adapter, WheelActionMode

from .base import Expert, ExpertReply, ExpertRequest, FailureCode


class ReferenceWBCExpert(Expert):
    """Reference-target nominal expert used to validate every adapter and the real closed loop.

    This class deliberately does not claim to solve whole-body dynamics. It maps the
    current validated 708 physical targets through the sole Action16Adapter. It is the
    first feasibility label provider; DWMPC/MPPI may replace it only after their gates.
    """

    def __init__(self, action_adapter: Action16Adapter):
        self.action_adapter = action_adapter
        self._episode_metadata: dict[str, Any] = {}

    def reset(self, episode_metadata: dict[str, Any]) -> None:
        self._episode_metadata = dict(episode_metadata)

    def act(self, request: ExpertRequest) -> ExpertReply:
        start = time.perf_counter()
        try:
            request.validate()
            q_reference = np.asarray(request.ref_window["joint_pos"], dtype=np.float32)
            dq_reference = np.asarray(request.ref_window["joint_vel"], dtype=np.float32)
            if q_reference.shape != (16,) or dq_reference.shape != (16,):
                raise ValueError(
                    f"ref_window joint shapes must both be (16,), got {q_reference.shape} and {dq_reference.shape}"
                )
            q_des_leg = q_reference[:12].copy()
            if self.action_adapter.contract.wheel_action_mode is WheelActionMode.HARD_ZERO:
                wheel_vel_des = np.zeros(4, dtype=np.float32)
            else:
                wheel_vel_des = dq_reference[12:].copy()
            action = self.action_adapter.physical_to_raw(q_des_leg, wheel_vel_des)
            lower_margin = action - self.action_adapter.contract.raw_min
            upper_margin = self.action_adapter.contract.raw_max - action
            safety_margin = float(np.min(np.minimum(lower_margin[:12], upper_margin[:12])))
            valid = bool(np.isfinite(action).all() and safety_margin >= 0.0)
            failure = FailureCode.NONE if valid else FailureCode.JOINT_LIMIT
            status = "REFERENCE_TARGET_VALID" if valid else "REFERENCE_TARGET_OUTSIDE_LIMITS"
        except (KeyError, TypeError, ValueError) as exc:
            q_des_leg = np.full(12, np.nan, dtype=np.float32)
            wheel_vel_des = np.zeros(4, dtype=np.float32)
            action = np.full(16, np.nan, dtype=np.float32)
            safety_margin = float("nan")
            valid = False
            failure = FailureCode.NAN_INF if "finite" in str(exc).lower() else FailureCode.TEACHER_INFEASIBLE
            status = f"REFERENCE_WBC_ERROR:{type(exc).__name__}:{exc}"

        reply = ExpertReply(
            valid=valid,
            q_des_leg=q_des_leg,
            wheel_vel_des=wheel_vel_des,
            action16=action,
            tau_ff_leg=None,
            predicted_grf=None,
            solve_ms=(time.perf_counter() - start) * 1000.0,
            solver_status=status,
            safety_margin=safety_margin,
            source="reference_wbc",
            failure_code=failure,
        )
        reply.validate()
        return reply

