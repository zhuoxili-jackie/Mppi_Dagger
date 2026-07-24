from __future__ import annotations

from typing import Any, Protocol

from .base import BackendUnavailable, Expert, ExpertReply, ExpertRequest


class DWMPCSolver(Protocol):
    def reset(self, metadata: dict[str, Any]) -> None: ...

    def solve(self, request: ExpertRequest) -> ExpertReply: ...


class DWMPCExpert(Expert):
    """Strict integration boundary; no unverified third-party implementation is bundled."""

    def __init__(self, solver: DWMPCSolver | None = None):
        self.solver = solver

    def reset(self, episode_metadata: dict[str, Any]) -> None:
        if self.solver is None:
            return
        self.solver.reset(episode_metadata)

    def act(self, request: ExpertRequest) -> ExpertReply:
        if self.solver is None:
            raise BackendUnavailable(
                "DWMPC is intentionally not selected for this migration. MPPI is the only "
                "formal R0-R3 label provider, and no DWMPC checkout should be added."
            )
        reply = self.solver.solve(request)
        reply.validate()
        if reply.source != "dwmpc":
            raise ValueError(f"DWMPC solver returned unexpected source={reply.source!r}")
        return reply
