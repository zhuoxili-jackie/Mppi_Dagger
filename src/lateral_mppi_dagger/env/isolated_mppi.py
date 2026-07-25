from __future__ import annotations

from dataclasses import fields
from multiprocessing.connection import Client
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lateral_mppi_dagger.contract.action16 import ActionContract
from lateral_mppi_dagger.env.isaac_mppi_rollout import (
    IsaacMPPIRolloutCloner,
    IsaacRolloutCostWeights,
    IsaacRolloutSnapshot,
)
from lateral_mppi_dagger.expert.base import ExpertReply, ExpertRequest
from lateral_mppi_dagger.reference.loader import ReferenceSet


def _tree_to_numpy(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, dict):
        return {
            key: _tree_to_numpy(child)
            for key, child in value.items()
        }
    raise TypeError(
        "Isolated MPPI snapshot trees may contain only tensors and mappings, "
        f"got {type(value).__name__}."
    )


def _tree_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, np.ndarray):
        return torch.as_tensor(value, device=device)
    if isinstance(value, dict):
        return {
            key: _tree_to_device(child, device)
            for key, child in value.items()
        }
    raise TypeError(
        "Serialized isolated MPPI snapshot trees may contain only arrays and "
        f"mappings, got {type(value).__name__}."
    )


def snapshot_to_payload(snapshot: IsaacRolloutSnapshot) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for field in fields(snapshot):
        value = getattr(snapshot, field.name)
        if isinstance(value, torch.Tensor) or isinstance(value, dict):
            payload[field.name] = _tree_to_numpy(value)
        elif isinstance(value, int):
            payload[field.name] = int(value)
        else:
            raise TypeError(
                f"Unsupported IsaacRolloutSnapshot field {field.name}: "
                f"{type(value).__name__}."
            )
    return payload


def snapshot_from_payload(
    payload: dict[str, Any],
    device: str | torch.device,
) -> IsaacRolloutSnapshot:
    target_device = torch.device(device)
    expected = {field.name for field in fields(IsaacRolloutSnapshot)}
    if set(payload) != expected:
        raise ValueError(
            "Serialized rollout snapshot fields differ from the current "
            f"schema: expected={sorted(expected)}, got={sorted(payload)}."
        )
    converted: dict[str, Any] = {}
    for field in fields(IsaacRolloutSnapshot):
        value = payload[field.name]
        if isinstance(value, (np.ndarray, dict)):
            converted[field.name] = _tree_to_device(
                value,
                target_device,
            )
        elif isinstance(value, (int, np.integer)):
            converted[field.name] = int(value)
        else:
            raise TypeError(
                f"Unsupported serialized field {field.name}: "
                f"{type(value).__name__}."
            )
    return IsaacRolloutSnapshot(**converted)


class IsolatedIsaacMPPIProvider:
    """MPPI client that never advances or restores the public Isaac scene."""

    def __init__(
        self,
        adapter: Any,
        references: ReferenceSet,
        action_contract: ActionContract,
        socket_path: str | Path,
        *,
        expected_server: dict[str, Any],
    ):
        if adapter.num_envs != 1:
            raise ValueError(
                "The isolated MPPI client requires exactly one public Isaac "
                f"environment, got {adapter.num_envs}."
            )
        self.adapter = adapter
        self.socket_path = Path(socket_path).expanduser().resolve()
        self.capture_cloner = IsaacMPPIRolloutCloner(
            adapter,
            references,
            action_contract,
            horizon=1,
            cost_weights=IsaacRolloutCostWeights(),
        )
        self.connection = Client(
            str(self.socket_path),
            family="AF_UNIX",
        )
        self.connection.send(
            {
                "op": "hello",
                "expected_server": expected_server,
            }
        )
        response = self.connection.recv()
        self._raise_remote_error(response)
        actual = response.get("server")
        if actual != expected_server:
            raise RuntimeError(
                "Isolated MPPI server identity mismatch: "
                f"expected={expected_server}, actual={actual}."
            )
        self.last_diagnostics: dict[str, Any] = {}
        self.closed = False

    @staticmethod
    def _raise_remote_error(response: Any) -> None:
        if not isinstance(response, dict):
            raise TypeError(
                "Isolated MPPI server returned a non-mapping response."
            )
        if response.get("ok") is False:
            raise RuntimeError(
                "Isolated MPPI server error "
                f"{response.get('error_type', 'Unknown')}: "
                f"{response.get('message', '')}\n"
                f"{response.get('traceback', '')}"
            )

    def reset(self, episode_metadata: dict[str, Any] | None = None) -> None:
        snapshot = self.capture_cloner.capture()
        self.connection.send(
            {
                "op": "reset",
                "snapshot": snapshot_to_payload(snapshot),
                "episode_metadata": dict(episode_metadata or {}),
            }
        )
        response = self.connection.recv()
        self._raise_remote_error(response)
        if response.get("ok") is not True:
            raise RuntimeError("Isolated MPPI reset was not acknowledged.")
        self.last_diagnostics = {}

    def __call__(self, request: ExpertRequest) -> ExpertReply:
        request.validate()
        snapshot = self.capture_cloner.capture()
        self.connection.send(
            {
                "op": "act",
                "snapshot": snapshot_to_payload(snapshot),
                "request": request,
            }
        )
        response = self.connection.recv()
        self._raise_remote_error(response)
        reply = response.get("reply")
        if not isinstance(reply, ExpertReply):
            raise TypeError(
                "Isolated MPPI server did not return an ExpertReply."
            )
        diagnostics = response.get("last_diagnostics", {})
        if not isinstance(diagnostics, dict):
            raise TypeError(
                "Isolated MPPI server diagnostics must be a mapping."
            )
        self.last_diagnostics = diagnostics
        return reply

    def close(self) -> None:
        if self.closed:
            return
        try:
            self.connection.send({"op": "close"})
            response = self.connection.recv()
            self._raise_remote_error(response)
        finally:
            self.connection.close()
            self.closed = True
