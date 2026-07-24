from __future__ import annotations

from typing import Any

import numpy as np

from .base import Expert, ExpertReply, ExpertRequest, FailureCode


class DisabledLabelExpert(Expert):
    """Zero-cost placeholder for student-only closed-loop evaluation.

    It is never a label provider.  DAgger data collection must use the MPPI
    backend; this class only lets the common transition schema record a
    student-controlled evaluation without launching rollout clones.
    """

    def reset(self, episode_metadata: dict[str, Any]) -> None:
        del episode_metadata

    def act(self, request: ExpertRequest) -> ExpertReply:
        request.validate()
        return ExpertReply(
            valid=False,
            q_des_leg=np.full(12, np.nan, dtype=np.float32),
            wheel_vel_des=np.zeros(4, dtype=np.float32),
            action16=np.full(16, np.nan, dtype=np.float32),
            tau_ff_leg=None,
            predicted_grf=None,
            solve_ms=0.0,
            solver_status="LABEL_EXPERT_DISABLED_FOR_STUDENT_EVALUATION",
            safety_margin=float("nan"),
            source="disabled",
            failure_code=FailureCode.TEACHER_INFEASIBLE,
            diagnostics=None,
        )
