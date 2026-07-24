from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ObservabilityResult:
    samples: int
    conflict_rate: float
    close_pair_rate: float
    median_neighbor_distance: float
    cases: list[dict[str, float | int]]


def analyze_observability(
    observations: np.ndarray,
    expert_actions: np.ndarray,
    contacts: np.ndarray,
    phases: np.ndarray,
    distance_threshold: float = 0.10,
    action_conflict_threshold: float = 0.50,
    maximum_cases: int = 50,
    chunk_size: int = 1024,
) -> ObservabilityResult:
    obs = np.asarray(observations, dtype=np.float32)
    actions = np.asarray(expert_actions, dtype=np.float32)
    contact = np.asarray(contacts, dtype=np.uint8)
    phase = np.asarray(phases, dtype=np.float32)
    samples = obs.shape[0]
    if obs.shape != (samples, 93) or actions.shape != (samples, 16) or contact.shape != (samples, 4):
        raise ValueError("Expected observations [N,93], actions [N,16], contacts [N,4].")
    if phase.shape != (samples,) or samples < 2:
        raise ValueError("phases must have shape [N] and at least two samples are required.")
    if not (np.isfinite(obs).all() and np.isfinite(actions).all() and np.isfinite(phase).all()):
        raise ValueError("Observability inputs contain NaN or Inf.")

    mean = obs.mean(axis=0)
    std = np.maximum(obs.std(axis=0), 1.0e-4)
    normalized = (obs - mean) / std
    neighbor_index = np.empty(samples, dtype=np.int64)
    neighbor_distance = np.empty(samples, dtype=np.float32)
    normalized_sq_norm = np.sum(np.square(normalized), axis=1)
    for start in range(0, samples, chunk_size):
        stop = min(start + chunk_size, samples)
        squared_distance = (
            normalized_sq_norm[start:stop, None]
            + normalized_sq_norm[None, :]
            - 2.0 * normalized[start:stop] @ normalized.T
        ) / normalized.shape[1]
        distance = np.sqrt(np.maximum(squared_distance, 0.0))
        row = np.arange(start, stop)
        distance[np.arange(stop - start), row] = np.inf
        # Gate requires a different phase/contact mode, not an adjacent duplicate.
        same_phase = np.abs(phase[start:stop, None] - phase[None, :]) < 0.03
        same_contact = np.all(contact[start:stop, None, :] == contact[None, :, :], axis=-1)
        distance[same_phase & same_contact] = np.inf
        nearest = np.argmin(distance, axis=1)
        neighbor_index[start:stop] = nearest
        neighbor_distance[start:stop] = distance[np.arange(stop - start), nearest]

    neighbor_action_delta = np.max(np.abs(actions - actions[neighbor_index]), axis=-1)
    contact_conflict = np.any(contact != contact[neighbor_index], axis=-1)
    close = neighbor_distance <= distance_threshold
    conflict = close & ((neighbor_action_delta >= action_conflict_threshold) | contact_conflict)
    conflict_indices = np.flatnonzero(conflict)
    ordered = conflict_indices[np.argsort(neighbor_distance[conflict_indices])] if conflict_indices.size else []
    cases = [
        {
            "index": int(index),
            "neighbor_index": int(neighbor_index[index]),
            "distance": float(neighbor_distance[index]),
            "max_action_delta": float(neighbor_action_delta[index]),
            "phase": float(phase[index]),
            "neighbor_phase": float(phase[neighbor_index[index]]),
            "contact_conflict": int(contact_conflict[index]),
        }
        for index in ordered[:maximum_cases]
    ]
    return ObservabilityResult(
        samples=samples,
        conflict_rate=float(conflict.mean()),
        close_pair_rate=float(close.mean()),
        median_neighbor_distance=float(np.median(neighbor_distance[np.isfinite(neighbor_distance)])),
        cases=cases,
    )
