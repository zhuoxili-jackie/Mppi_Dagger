from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DaggerRound:
    round: int
    beta: float
    scenario: str

    def validate(self) -> None:
        if self.round < 0 or not 0.0 <= self.beta <= 1.0:
            raise ValueError(f"Invalid DAgger round: {self}")


@dataclass
class DaggerState:
    schema_version: str = "pcbc-dagger-run-state-v1"
    completed_rounds: list[int] = field(default_factory=list)
    active_round: int | None = None
    best_checkpoint: str | None = None
    round_outputs: dict[str, dict[str, Any]] = field(default_factory=dict)
    failure: dict[str, Any] | None = None

    @classmethod
    def load(cls, path: str | Path) -> "DaggerState":
        with Path(path).open("r", encoding="utf-8") as stream:
            values = json.load(stream)
        state = cls(**values)
        if state.schema_version != "pcbc-dagger-run-state-v1":
            raise ValueError(f"Unsupported DAgger state schema {state.schema_version!r}")
        return state

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent, mode="w", encoding="utf-8", delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(asdict(self), stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)

    def begin_round(self, round_config: DaggerRound) -> None:
        round_config.validate()
        if self.active_round is not None:
            raise RuntimeError(f"Cannot begin round {round_config.round}; round {self.active_round} is active.")
        if round_config.round in self.completed_rounds:
            raise RuntimeError(f"DAgger round {round_config.round} is already complete.")
        self.active_round = round_config.round
        self.failure = None

    def complete_round(self, round_config: DaggerRound, outputs: dict[str, Any]) -> None:
        if self.active_round != round_config.round:
            raise RuntimeError(f"Round completion mismatch: active={self.active_round}, got={round_config.round}")
        self.round_outputs[str(round_config.round)] = outputs
        self.completed_rounds.append(round_config.round)
        self.completed_rounds.sort()
        self.best_checkpoint = outputs.get("best_checkpoint", self.best_checkpoint)
        self.active_round = None

    def fail_round(self, error_type: str, message: str, context: dict[str, Any]) -> None:
        self.failure = {
            "round": self.active_round,
            "error_type": error_type,
            "message": message,
            "context": context,
        }
        self.active_round = None

