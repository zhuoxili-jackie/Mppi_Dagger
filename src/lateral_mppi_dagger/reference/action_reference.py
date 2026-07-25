from __future__ import annotations

from typing import Any

import numpy as np

from lateral_mppi_dagger.config import resolve_project_path, sha256_file


_SOLVER_OVERRIDE_KEYS = {
    "action_residual_weight",
    "base_orientation_cost_multiplier",
    "selection_mode",
    "warm_start",
}
_SOLVER_SCHEDULE_PHASE_KEYS = {
    "start_frame",
    "reset_warm_start",
    *_SOLVER_OVERRIDE_KEYS,
}


def normalize_nominal_solver_overrides(
    overrides: dict[str, Any],
) -> dict[str, Any]:
    """Validate and normalize static or frame-scheduled solver overrides."""

    if not isinstance(overrides, dict):
        raise TypeError("Nominal action solver override must be a mapping.")
    unknown = set(overrides) - {
        *_SOLVER_OVERRIDE_KEYS,
        "solver_schedule",
    }
    if unknown:
        raise ValueError(
            "Unknown nominal action solver override keys: "
            f"{sorted(unknown)}."
        )
    if "solver_schedule" in overrides and (
        set(overrides) & _SOLVER_OVERRIDE_KEYS
    ):
        raise ValueError(
            "solver_schedule cannot be combined with static nominal action "
            "solver overrides."
        )

    if "solver_schedule" not in overrides:
        normalized: dict[str, Any] = {}
        if "action_residual_weight" in overrides:
            action_residual_weight = float(
                overrides["action_residual_weight"]
            )
            if (
                not np.isfinite(action_residual_weight)
                or action_residual_weight < 0.0
            ):
                raise ValueError(
                    "action_residual_weight must be finite and non-negative."
                )
            normalized["action_residual_weight"] = action_residual_weight
        if "base_orientation_cost_multiplier" in overrides:
            base_orientation_cost_multiplier = float(
                overrides["base_orientation_cost_multiplier"]
            )
            if (
                not np.isfinite(base_orientation_cost_multiplier)
                or base_orientation_cost_multiplier < 1.0
            ):
                raise ValueError(
                    "base_orientation_cost_multiplier must be finite and "
                    "at least 1.0."
                )
            normalized["base_orientation_cost_multiplier"] = (
                base_orientation_cost_multiplier
            )
        if "selection_mode" in overrides:
            selection_mode = str(overrides["selection_mode"])
            if selection_mode not in {"weighted", "best_sample"}:
                raise ValueError(
                    "selection_mode override must be 'weighted' or "
                    "'best_sample'."
                )
            normalized["selection_mode"] = selection_mode
        if "warm_start" in overrides:
            warm_start = overrides["warm_start"]
            if not isinstance(warm_start, bool):
                raise TypeError("warm_start override must be boolean.")
            normalized["warm_start"] = warm_start
        return normalized

    schedule = overrides["solver_schedule"]
    if not isinstance(schedule, list) or not schedule:
        raise ValueError("solver_schedule must be a non-empty list.")
    normalized_schedule: list[dict[str, Any]] = []
    previous_start = -1
    for index, phase in enumerate(schedule):
        if not isinstance(phase, dict):
            raise TypeError(
                "Each solver_schedule phase must be a mapping."
            )
        unknown_phase = set(phase) - _SOLVER_SCHEDULE_PHASE_KEYS
        if unknown_phase:
            raise ValueError(
                "Unknown solver_schedule phase keys: "
                f"{sorted(unknown_phase)}."
            )
        if "start_frame" not in phase:
            raise ValueError(
                f"solver_schedule phase {index} is missing start_frame."
            )
        start_frame = phase["start_frame"]
        if isinstance(start_frame, bool) or not isinstance(
            start_frame,
            (int, np.integer),
        ):
            raise TypeError(
                "solver_schedule start_frame must be an integer."
            )
        start_frame = int(start_frame)
        if start_frame < 0:
            raise ValueError(
                "solver_schedule start_frame must be non-negative."
            )
        if index == 0 and start_frame != 0:
            raise ValueError(
                "solver_schedule must start at frame 0."
            )
        if start_frame <= previous_start:
            raise ValueError(
                "solver_schedule start_frame values must be strictly "
                "increasing."
            )
        solver_values = {
            key: phase[key]
            for key in _SOLVER_OVERRIDE_KEYS
            if key in phase
        }
        if not solver_values:
            raise ValueError(
                "Each solver_schedule phase must override at least one "
                "solver setting."
            )
        normalized_phase = {
            "start_frame": start_frame,
            **normalize_nominal_solver_overrides(solver_values),
        }
        if "reset_warm_start" in phase:
            reset_warm_start = phase["reset_warm_start"]
            if not isinstance(reset_warm_start, bool):
                raise TypeError(
                    "solver_schedule reset_warm_start must be boolean."
                )
            normalized_phase["reset_warm_start"] = reset_warm_start
        normalized_schedule.append(normalized_phase)
        previous_start = start_frame
    return {"solver_schedule": normalized_schedule}


def resolve_nominal_solver_overrides(
    overrides: dict[str, Any],
    ref_frame: int,
) -> tuple[dict[str, Any], int | None]:
    """Resolve normalized overrides for one reference frame."""

    schedule = overrides.get("solver_schedule")
    if schedule is None:
        return dict(overrides), None
    frame = int(ref_frame)
    if frame < 0:
        raise ValueError("ref_frame must be non-negative.")
    active_index = 0
    for index, phase in enumerate(schedule):
        if int(phase["start_frame"]) > frame:
            break
        active_index = index
    active_phase = schedule[active_index]
    return (
        {
            key: active_phase[key]
            for key in _SOLVER_OVERRIDE_KEYS
            if key in active_phase
        },
        active_index,
    )


def load_nominal_action_references(
    config: dict[str, Any] | None,
) -> tuple[
    dict[int, np.ndarray],
    dict[int, np.ndarray],
    dict[int, dict[str, Any]],
    dict[str, Any],
]:
    """Load hashed physical-q or raw-action proposal centres by reference ID."""

    if config is None:
        return {}, {}, {}, {"enabled": False, "entries": []}
    entries = config.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError(
            "nominal_action_reference.entries must be a non-empty list."
        )

    q_des_by_ref: dict[int, np.ndarray] = {}
    raw_action_by_ref: dict[int, np.ndarray] = {}
    overrides_by_ref: dict[int, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise TypeError(
                "Each nominal action reference entry must be a mapping."
            )
        ref_id = int(entry["ref_id"])
        if ref_id in q_des_by_ref or ref_id in raw_action_by_ref:
            raise ValueError(
                f"Duplicate nominal action reference for ref {ref_id}."
            )

        has_q_des_key = "q_des_key" in entry
        has_raw_action_key = "raw_action_key" in entry
        if has_q_des_key and has_raw_action_key:
            raise ValueError(
                "A nominal action reference entry must select either "
                "q_des_key or raw_action_key, not both."
            )
        representation = (
            "raw_action" if has_raw_action_key else "q_des"
        )
        array_key = str(
            entry.get(
                "raw_action_key"
                if representation == "raw_action"
                else "q_des_key",
                "raw_action_leg"
                if representation == "raw_action"
                else "q_des_leg",
            )
        )

        configured_path = str(entry["path"])
        path = resolve_project_path(configured_path)
        expected_hash = str(entry["sha256"])
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(
                "Nominal action reference hash mismatch for "
                f"{configured_path}: expected {expected_hash}, got "
                f"{actual_hash}."
            )
        with np.load(path, allow_pickle=False) as archive:
            if array_key not in archive.files:
                raise KeyError(
                    f"{configured_path} has no array {array_key!r}."
                )
            values = np.asarray(archive[array_key], dtype=np.float32)
            if "ref_id" in archive.files:
                stored_ref_id = int(
                    np.asarray(archive["ref_id"]).reshape(-1)[0]
                )
                if stored_ref_id != ref_id:
                    raise ValueError(
                        "Nominal action reference ref_id mismatch: "
                        f"config={ref_id}, asset={stored_ref_id}."
                    )
        if values.ndim != 2 or values.shape[1] != 12:
            raise ValueError(
                "Nominal action reference array must have shape "
                f"[frames,12], got {values.shape}."
            )
        if not np.isfinite(values).all():
            raise ValueError(
                f"Nominal action reference for ref {ref_id} contains NaN/Inf."
            )

        if representation == "raw_action":
            raw_action_by_ref[ref_id] = values
        else:
            q_des_by_ref[ref_id] = values
        configured_overrides = {
            key: entry[key]
            for key in {
                *_SOLVER_OVERRIDE_KEYS,
                "solver_schedule",
            }
            if key in entry
        }
        overrides = normalize_nominal_solver_overrides(
            configured_overrides
        )
        if overrides:
            overrides_by_ref[ref_id] = overrides
        record = {
            "ref_id": ref_id,
            "path": configured_path,
            "sha256": actual_hash,
            "representation": representation,
            "array_key": array_key,
            "shape": list(values.shape),
        }
        if overrides:
            record["solver_overrides"] = dict(overrides)
        records.append(record)
    return (
        q_des_by_ref,
        raw_action_by_ref,
        overrides_by_ref,
        {"enabled": True, "entries": records},
    )
