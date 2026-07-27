#!/usr/bin/env python3
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any

import yaml

from _bootstrap import ROOT, write_json

from lateral_mppi_dagger.config import sha256_file
from lateral_mppi_dagger.env.isaac_mppi_rollout import (
    IsaacRolloutCostWeights,
    IsaacRolloutLoadLimits,
)
from lateral_mppi_dagger.reference.action_reference import (
    normalize_nominal_solver_overrides,
)


_SOLVER_OVERRIDE_KEYS = (
    "action_residual_weight",
    "base_orientation_cost_multiplier",
    "base_orientation_axis_multipliers",
    "lateral_velocity_cost_multiplier",
    "rear_support_loss_cost_multiplier",
    "selection_mode",
    "temperature",
    "warm_start",
    "solver_schedule",
)

_MPPI_SETTING_KEYS = (
    "horizon",
    "samples",
    "optimization_iterations",
    "temperature",
    "temporal_smoothing",
    "reference_action_lookahead_steps",
)

_OUTPUT_FEEDBACK_SETTING_KEYS = (
    "output_front_force_feedback_target_n",
    "output_front_force_feedback_min_contact_n",
    "output_front_force_feedback_lookahead_steps",
    "output_front_force_feedback_gain_leg",
    "output_rear_swing_force_feedback_target_n",
    "output_rear_swing_force_feedback_scale_n",
    "output_rear_swing_force_feedback_lookahead_steps",
    "output_rear_swing_force_feedback_start_frame",
    "output_rear_swing_force_feedback_gain_leg",
    "output_rear_swing_height_feedback_ref_ids",
    "output_rear_swing_height_feedback_gain",
    "output_rear_swing_height_feedback_max_abs_rad",
    "output_rear_swing_height_feedback_lookahead_steps",
    "output_rear_swing_height_feedback_start_frame",
    "output_rear_support_tracking_feedback_ref_ids",
    "output_rear_support_tracking_feedback_gain",
    "output_rear_support_tracking_feedback_max_abs_rad",
    "output_rear_support_tracking_feedback_lookahead_steps",
    "output_rear_support_tracking_feedback_start_frame",
    "output_pitch_feedback_ref_ids",
    "output_pitch_feedback_gain_leg",
    "output_pitch_feedback_axis",
    "output_pitch_feedback_start_frame",
    "output_pitch_feedback_max_abs_rad",
    "output_contact_orientation_feedback_ref_ids",
    "output_contact_orientation_feedback_gain_xyz",
    "output_contact_orientation_feedback_start_frame",
    "output_contact_orientation_feedback_max_endpoint_delta_m",
    "output_contact_orientation_feedback_max_abs_rad",
)

_REAR_SWING_REFERENCE_PROPOSAL_SETTING_KEYS = (
    "rear_swing_reference_proposal_ref_ids",
    "rear_swing_reference_proposal_scales",
    "rear_swing_reference_proposal_joint_mask_leg",
    "rear_swing_reference_proposal_lead_steps",
    "rear_swing_action_residual_lead_steps",
    "rear_swing_tracking_error_proposal_scales",
    "rear_swing_tracking_error_proposal_joint_mask_leg",
    "rear_swing_tracking_error_proposal_start_frame",
    "rear_swing_load_transfer_proposal_ref_ids",
    "rear_swing_load_transfer_proposal_scales",
    "rear_swing_load_transfer_proposal_gain_by_wheel_leg_rad",
    "rear_swing_load_transfer_proposal_start_frame",
    "rear_swing_load_transfer_proposal_start_frame_by_wheel",
    "rear_swing_load_transfer_proposal_gate_mode",
    "rear_swing_load_transfer_proposal_imbalance_threshold_n",
)

_FRONT_SUPPORT_PROPOSAL_SETTING_KEYS = (
    "front_support_proposal_ref_ids",
    "front_support_proposal_scales",
    "front_support_proposal_gain_leg_rad",
    "front_support_proposal_start_frame",
    "combine_rear_swing_front_support_proposals",
    "combine_rear_swing_load_transfer_front_support_proposals",
    "combine_rear_swing_reference_load_transfer_front_support_proposals",
    "include_rear_support_reference_in_coordinated_proposals",
    "rear_support_reference_proposal_start_frame",
)


def _structured_proposal_count(
    config: dict[str, Any],
    *,
    rear_scales: list[float] | tuple[float, ...] | None = None,
    tracking_error_scales: list[float] | tuple[float, ...] | None = None,
    load_transfer_scales: list[float] | tuple[float, ...] | None = None,
    front_scales: list[float] | tuple[float, ...] | None = None,
    combine: bool | None = None,
    combine_load_transfer_front: bool | None = None,
    combine_reference_load_transfer_front: bool | None = None,
    include_rear_support: bool | None = None,
) -> int:
    rear = (
        config.get("rear_swing_reference_proposal_scales", [])
        if rear_scales is None
        else rear_scales
    )
    front = (
        config.get("front_support_proposal_scales", [])
        if front_scales is None
        else front_scales
    )
    tracking_error = (
        config.get(
            "rear_swing_tracking_error_proposal_scales",
            [],
        )
        if tracking_error_scales is None
        else tracking_error_scales
    )
    load_transfer = (
        config.get(
            "rear_swing_load_transfer_proposal_scales",
            [],
        )
        if load_transfer_scales is None
        else load_transfer_scales
    )
    coordinated = (
        bool(
            config.get(
                "combine_rear_swing_front_support_proposals",
                False,
            )
        )
        if combine is None
        else combine
    )
    coordinated_load_transfer = (
        bool(
            config.get(
                "combine_rear_swing_load_transfer_front_support_proposals",
                False,
            )
        )
        if combine_load_transfer_front is None
        else combine_load_transfer_front
    )
    coordinated_reference_load_transfer = (
        bool(
            config.get(
                "combine_rear_swing_reference_load_transfer_front_support_"
                "proposals",
                False,
            )
        )
        if combine_reference_load_transfer_front is None
        else combine_reference_load_transfer_front
    )
    rear_support = (
        bool(
            config.get(
                "include_rear_support_reference_in_coordinated_proposals",
                False,
            )
        )
        if include_rear_support is None
        else include_rear_support
    )
    return (
        len(rear)
        + len(tracking_error)
        + len(load_transfer)
        + len(front)
        + (len(rear) * len(front) if coordinated else 0)
        + (
            len(load_transfer) * len(front)
            if coordinated_load_transfer
            else 0
        )
        + (
            len(rear) * len(load_transfer) * len(front)
            if coordinated_reference_load_transfer
            else 0
        )
        + (len(rear) if rear_support else 0)
    )


def _inside_root(path: Path, root: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if resolved == root or root not in resolved.parents:
        raise ValueError(f"{label} must be inside {root}: {resolved}")
    return resolved


def replace_nominal_action_entries(
    config: dict,
    replacements: dict[int, Path],
    *,
    root: Path,
) -> tuple[dict, list[dict[str, object]]]:
    """Return a config copy with selected nominal assets hash-locked."""

    result = deepcopy(config)
    block = result.get("nominal_action_reference")
    entries = block.get("entries") if isinstance(block, dict) else None
    if not isinstance(entries, list):
        raise ValueError(
            "Config has no nominal_action_reference.entries list."
        )
    by_ref = {
        int(entry["ref_id"]): entry
        for entry in entries
        if isinstance(entry, dict) and "ref_id" in entry
    }
    missing = sorted(set(replacements) - set(by_ref))
    if missing:
        raise ValueError(
            f"Replacement ref IDs are absent from the config: {missing}."
        )

    records: list[dict[str, object]] = []
    for ref_id, value in sorted(replacements.items()):
        resolved = _inside_root(value, root, "Nominal action asset")
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        relative = resolved.relative_to(root).as_posix()
        digest = sha256_file(resolved)
        entry = by_ref[ref_id]
        previous_path = str(entry.get("path"))
        previous_sha256 = str(entry.get("sha256"))
        entry["path"] = relative
        entry["sha256"] = digest
        records.append(
            {
                "ref_id": ref_id,
                "previous_path": previous_path,
                "previous_sha256": previous_sha256,
                "path": relative,
                "sha256": digest,
            }
        )
    return result, records


def replace_nominal_solver_overrides(
    config: dict,
    replacements: dict[int, dict[str, Any]],
) -> tuple[dict, list[dict[str, object]]]:
    """Return a config copy with selected solver overrides replaced."""

    result = deepcopy(config)
    block = result.get("nominal_action_reference")
    entries = block.get("entries") if isinstance(block, dict) else None
    if not isinstance(entries, list):
        raise ValueError(
            "Config has no nominal_action_reference.entries list."
        )
    by_ref = {
        int(entry["ref_id"]): entry
        for entry in entries
        if isinstance(entry, dict) and "ref_id" in entry
    }
    missing = sorted(set(replacements) - set(by_ref))
    if missing:
        raise ValueError(
            f"Solver override ref IDs are absent from the config: {missing}."
        )

    records: list[dict[str, object]] = []
    for ref_id, configured in sorted(replacements.items()):
        normalized = normalize_nominal_solver_overrides(configured)
        entry = by_ref[ref_id]
        previous = {
            key: deepcopy(entry[key])
            for key in _SOLVER_OVERRIDE_KEYS
            if key in entry
        }
        for key in _SOLVER_OVERRIDE_KEYS:
            entry.pop(key, None)
        entry.update(deepcopy(normalized))
        records.append(
            {
                "ref_id": ref_id,
                "previous": previous,
                "replacement": deepcopy(normalized),
            }
        )
    return result, records


def replace_cost_weights(
    config: dict,
    replacements: dict[str, float],
) -> tuple[dict, list[dict[str, object]]]:
    """Return a config copy with selected MPPI cost weights replaced."""

    result = deepcopy(config)
    configured = result.get("cost_weights")
    if not isinstance(configured, dict):
        raise ValueError("Config has no cost_weights mapping.")
    merged = {**configured, **replacements}
    # Reuse the runtime parser so unknown, negative, or non-numeric values
    # fail before a candidate config is written.
    normalized = IsaacRolloutCostWeights.from_dict(merged)
    normalized_values = {
        name: float(getattr(normalized, name))
        for name in merged
    }
    records: list[dict[str, object]] = []
    for name in sorted(replacements):
        records.append(
            {
                "name": name,
                "previous": configured.get(name),
                "replacement": normalized_values[name],
            }
        )
        configured[name] = normalized_values[name]
    return result, records


def replace_load_limits(
    config: dict,
    replacements: dict[str, float],
) -> tuple[dict, list[dict[str, object]]]:
    """Return a config copy with selected rollout load limits replaced."""

    result = deepcopy(config)
    configured = result.get("load_limits")
    if not isinstance(configured, dict):
        raise ValueError("Config has no load_limits mapping.")
    merged = {**configured, **replacements}
    # Reuse the runtime parser so unknown, non-numeric, or out-of-range
    # values fail before a candidate config is written.
    normalized = IsaacRolloutLoadLimits.from_dict(merged)
    normalized_values = {
        name: float(getattr(normalized, name))
        for name in merged
    }
    records: list[dict[str, object]] = []
    for name in sorted(replacements):
        records.append(
            {
                "name": name,
                "previous": configured.get(name),
                "replacement": normalized_values[name],
            }
        )
        configured[name] = normalized_values[name]
    return result, records


def replace_mppi_settings(
    config: dict,
    replacements: dict[str, Any],
) -> tuple[dict, list[dict[str, object]]]:
    """Return a config copy with validated optimizer scalars replaced."""

    result = deepcopy(config)
    unknown = sorted(set(replacements) - set(_MPPI_SETTING_KEYS))
    if unknown:
        raise ValueError(f"Unknown MPPI setting overrides: {unknown}.")
    records: list[dict[str, object]] = []
    for name in sorted(replacements):
        if name not in result:
            raise ValueError(
                f"Base config has no top-level MPPI setting {name!r}."
            )
        if name == "reference_action_lookahead_steps":
            raw_value = replacements[name]
            if (
                isinstance(raw_value, bool)
                or not isinstance(raw_value, int)
                or raw_value < 0
            ):
                raise ValueError(
                    "MPPI reference_action_lookahead_steps must be a "
                    "non-negative integer."
                )
            records.append(
                {
                    "name": name,
                    "previous": result[name],
                    "replacement": int(raw_value),
                }
            )
            result[name] = int(raw_value)
            continue
        if name in ("horizon", "optimization_iterations"):
            raw_value = replacements[name]
            if (
                isinstance(raw_value, bool)
                or not isinstance(raw_value, int)
                or raw_value < 1
            ):
                raise ValueError(
                    f"MPPI {name} must be a positive integer."
                )
            records.append(
                {
                    "name": name,
                    "previous": result[name],
                    "replacement": int(raw_value),
                }
            )
            result[name] = int(raw_value)
            continue
        if name == "samples":
            raw_value = replacements[name]
            if (
                isinstance(raw_value, bool)
                or not isinstance(raw_value, int)
                or raw_value < 2
            ):
                raise ValueError(
                    "MPPI samples must be an integer greater than or equal "
                    "to 2."
                )
            value = int(raw_value)
            structured_proposal_count = _structured_proposal_count(
                result
            )
            if structured_proposal_count >= value:
                raise ValueError(
                    "MPPI samples must leave at least one stochastic sample "
                    "after structured proposals."
                )
            records.append(
                {
                    "name": name,
                    "previous": result[name],
                    "replacement": value,
                }
            )
            result[name] = value
            continue
        try:
            value = float(replacements[name])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"MPPI setting {name!r} must be numeric."
            ) from exc
        if not math.isfinite(value):
            raise ValueError(
                f"MPPI setting {name!r} must be finite."
            )
        if name == "temperature" and value <= 0.0:
            raise ValueError("MPPI temperature must be positive.")
        if name == "temporal_smoothing" and not 0.0 <= value < 1.0:
            raise ValueError(
                "MPPI temporal_smoothing must lie in [0,1)."
            )
        records.append(
            {
                "name": name,
                "previous": result[name],
                "replacement": value,
            }
        )
        result[name] = value
    return result, records


def replace_output_feedback_settings(
    config: dict,
    replacements: dict[str, Any],
) -> tuple[dict, list[dict[str, object]]]:
    """Return a config copy with validated bounded output feedback settings."""

    result = deepcopy(config)
    unknown = sorted(
        set(replacements) - set(_OUTPUT_FEEDBACK_SETTING_KEYS)
    )
    if unknown:
        raise ValueError(
            f"Unknown output feedback setting overrides: {unknown}."
        )
    normalized: dict[str, object] = {}
    for name, raw_value in replacements.items():
        if name == "output_contact_orientation_feedback_gain_xyz":
            if (
                not isinstance(raw_value, (list, tuple))
                or len(raw_value) != 3
            ):
                raise ValueError(
                    f"{name} must contain exactly three numeric values."
                )
            try:
                value = [float(item) for item in raw_value]
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{name} must contain exactly three numeric values."
                ) from exc
            if (
                not all(math.isfinite(item) for item in value)
                or any(item < 0.0 or item > 1.0 for item in value)
            ):
                raise ValueError(
                    f"{name} must be finite and lie in [0,1]."
                )
            normalized[name] = value
        elif name in (
            "output_front_force_feedback_gain_leg",
            "output_rear_swing_force_feedback_gain_leg",
            "output_pitch_feedback_gain_leg",
        ):
            if (
                not isinstance(raw_value, (list, tuple))
                or len(raw_value) != 12
            ):
                raise ValueError(
                    f"{name} must contain exactly 12 numeric values."
                )
            try:
                value = [float(item) for item in raw_value]
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{name} must contain exactly 12 numeric values."
                ) from exc
            if not all(math.isfinite(item) for item in value):
                raise ValueError(f"{name} must contain only finite values.")
            if name != "output_pitch_feedback_gain_leg":
                forbidden_indices = (
                    (2, 3, 6, 7, 10, 11)
                    if name == "output_front_force_feedback_gain_leg"
                    else (0, 1, 4, 5, 8, 9)
                )
                forbidden_label = (
                    "rear-leg"
                    if name == "output_front_force_feedback_gain_leg"
                    else "front-leg"
                )
                if any(
                    value[index] != 0.0
                    for index in forbidden_indices
                ):
                    raise ValueError(
                        f"{name} must be zero for all "
                        f"{forbidden_label} joints."
                    )
            normalized[name] = value
        elif name in (
            "output_pitch_feedback_ref_ids",
            "output_rear_swing_height_feedback_ref_ids",
            "output_rear_support_tracking_feedback_ref_ids",
            "output_contact_orientation_feedback_ref_ids",
        ):
            if not isinstance(raw_value, (list, tuple)):
                raise ValueError(
                    f"{name} must be a list of non-negative integer "
                    "reference IDs."
                )
            if any(
                isinstance(ref_id, bool)
                or not isinstance(ref_id, int)
                or ref_id < 0
                for ref_id in raw_value
            ):
                raise ValueError(
                    f"{name} must contain only non-negative integer "
                    "reference IDs."
                )
            if len(set(raw_value)) != len(raw_value):
                raise ValueError(
                    f"{name} must not contain duplicates."
                )
            normalized[name] = list(raw_value)
        elif name == "output_pitch_feedback_axis":
            if raw_value not in ("x", "y", "z"):
                raise ValueError(
                    "output_pitch_feedback_axis must be one of 'x', 'y', "
                    "or 'z'."
                )
            normalized[name] = raw_value
        elif name in (
            "output_front_force_feedback_lookahead_steps",
            "output_rear_swing_force_feedback_lookahead_steps",
            "output_rear_swing_force_feedback_start_frame",
            "output_rear_swing_height_feedback_lookahead_steps",
            "output_rear_swing_height_feedback_start_frame",
            "output_rear_support_tracking_feedback_lookahead_steps",
            "output_rear_support_tracking_feedback_start_frame",
            "output_pitch_feedback_start_frame",
            "output_contact_orientation_feedback_start_frame",
        ):
            if (
                isinstance(raw_value, bool)
                or not isinstance(raw_value, int)
                or raw_value < 0
            ):
                raise ValueError(
                    f"{name} must be a non-negative integer."
                )
            normalized[name] = int(raw_value)
        else:
            try:
                value = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be numeric.") from exc
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite.")
            if (
                name.endswith("_target_n")
                or name.endswith("_min_contact_n")
                or name.endswith("_max_abs_rad")
                or name.endswith("_feedback_gain")
            ) and value < 0.0:
                raise ValueError(f"{name} must be non-negative.")
            if name.endswith("_scale_n") and value <= 0.0:
                raise ValueError(f"{name} must be positive.")
            if (
                name == "output_rear_swing_height_feedback_gain"
                and value > 1.0
            ):
                raise ValueError(f"{name} must lie in [0,1].")
            if (
                name
                == "output_rear_swing_height_feedback_max_abs_rad"
                and value > 0.12
            ):
                raise ValueError(f"{name} must lie in [0,0.12].")
            if (
                name
                == (
                    "output_contact_orientation_feedback_"
                    "max_endpoint_delta_m"
                )
                and not 0.0 <= value <= 0.02
            ):
                raise ValueError(f"{name} must lie in [0,0.02].")
            if (
                name
                == "output_contact_orientation_feedback_max_abs_rad"
                and value > 0.05
            ):
                raise ValueError(f"{name} must lie in [0,0.05].")
            normalized[name] = value

    for prefix, label in (
        ("output_front_force_feedback", "front"),
        ("output_rear_swing_force_feedback", "rear-swing"),
    ):
        target_name = f"{prefix}_target_n"
        gain_name = f"{prefix}_gain_leg"
        target = float(
            normalized.get(
                target_name,
                result.get(target_name, 0.0),
            )
        )
        gains = normalized.get(
            gain_name,
            result.get(gain_name, [0.0] * 12),
        )
        if target == 0.0 and any(float(value) != 0.0 for value in gains):
            raise ValueError(
                f"Non-zero {label} feedback gains require a positive target."
            )

    rear_swing_height_ref_ids = normalized.get(
        "output_rear_swing_height_feedback_ref_ids",
        result.get(
            "output_rear_swing_height_feedback_ref_ids",
            [],
        ),
    )
    rear_swing_height_gain = float(
        normalized.get(
            "output_rear_swing_height_feedback_gain",
            result.get(
                "output_rear_swing_height_feedback_gain",
                0.0,
            ),
        )
    )
    rear_swing_height_cap = float(
        normalized.get(
            "output_rear_swing_height_feedback_max_abs_rad",
            result.get(
                "output_rear_swing_height_feedback_max_abs_rad",
                0.0,
            ),
        )
    )
    rear_swing_height_configured = bool(
        rear_swing_height_ref_ids
        or rear_swing_height_gain > 0.0
        or rear_swing_height_cap > 0.0
    )
    if rear_swing_height_configured and (
        not rear_swing_height_ref_ids
        or rear_swing_height_gain <= 0.0
        or rear_swing_height_cap <= 0.0
    ):
        raise ValueError(
            "Rear-swing height feedback requires non-empty reference IDs, "
            "a positive gain, and a positive maximum correction."
        )

    rear_support_ref_ids = normalized.get(
        "output_rear_support_tracking_feedback_ref_ids",
        result.get(
            "output_rear_support_tracking_feedback_ref_ids",
            [],
        ),
    )
    rear_support_gain = float(
        normalized.get(
            "output_rear_support_tracking_feedback_gain",
            result.get(
                "output_rear_support_tracking_feedback_gain",
                0.0,
            ),
        )
    )
    rear_support_cap = float(
        normalized.get(
            "output_rear_support_tracking_feedback_max_abs_rad",
            result.get(
                "output_rear_support_tracking_feedback_max_abs_rad",
                0.0,
            ),
        )
    )
    rear_support_configured = bool(
        rear_support_ref_ids
        or rear_support_gain > 0.0
        or rear_support_cap > 0.0
    )
    if rear_support_configured and (
        not rear_support_ref_ids
        or rear_support_gain <= 0.0
        or rear_support_cap <= 0.0
    ):
        raise ValueError(
            "Rear-support tracking feedback requires non-empty reference "
            "IDs, a positive gain, and a positive maximum correction."
        )

    pitch_gains = normalized.get(
        "output_pitch_feedback_gain_leg",
        result.get("output_pitch_feedback_gain_leg", [0.0] * 12),
    )
    if any(float(value) != 0.0 for value in pitch_gains):
        pitch_ref_ids = normalized.get(
            "output_pitch_feedback_ref_ids",
            result.get("output_pitch_feedback_ref_ids", []),
        )
        if not pitch_ref_ids:
            raise ValueError(
                "Non-zero output pitch-feedback gains require at least one "
                "output_pitch_feedback_ref_id."
            )
        pitch_cap = float(
            normalized.get(
                "output_pitch_feedback_max_abs_rad",
                result.get("output_pitch_feedback_max_abs_rad", 0.0),
            )
        )
        if pitch_cap <= 0.0:
            raise ValueError(
                "Non-zero output pitch-feedback gains require a positive "
                "output_pitch_feedback_max_abs_rad."
            )

    contact_orientation_ref_ids = normalized.get(
        "output_contact_orientation_feedback_ref_ids",
        result.get(
            "output_contact_orientation_feedback_ref_ids",
            [],
        ),
    )
    contact_orientation_gains = normalized.get(
        "output_contact_orientation_feedback_gain_xyz",
        result.get(
            "output_contact_orientation_feedback_gain_xyz",
            [0.0] * 3,
        ),
    )
    contact_orientation_endpoint_cap = float(
        normalized.get(
            "output_contact_orientation_feedback_max_endpoint_delta_m",
            result.get(
                "output_contact_orientation_feedback_"
                "max_endpoint_delta_m",
                0.0,
            ),
        )
    )
    contact_orientation_joint_cap = float(
        normalized.get(
            "output_contact_orientation_feedback_max_abs_rad",
            result.get(
                "output_contact_orientation_feedback_max_abs_rad",
                0.0,
            ),
        )
    )
    contact_orientation_configured = bool(
        contact_orientation_ref_ids
        or any(
            float(value) != 0.0
            for value in contact_orientation_gains
        )
        or contact_orientation_endpoint_cap > 0.0
        or contact_orientation_joint_cap > 0.0
    )
    if contact_orientation_configured and (
        not contact_orientation_ref_ids
        or not any(
            float(value) > 0.0
            for value in contact_orientation_gains
        )
        or contact_orientation_endpoint_cap <= 0.0
        or contact_orientation_joint_cap <= 0.0
    ):
        raise ValueError(
            "Contact-orientation feedback requires non-empty reference IDs, "
            "at least one positive axis gain, a positive endpoint cap, and "
            "a positive joint cap."
        )

    records: list[dict[str, object]] = []
    for name in sorted(normalized):
        records.append(
            {
                "name": name,
                "previous": deepcopy(result.get(name)),
                "replacement": deepcopy(normalized[name]),
            }
        )
        result[name] = deepcopy(normalized[name])
    return result, records


def replace_rear_swing_reference_proposal_settings(
    config: dict,
    replacements: dict[str, Any],
) -> tuple[dict, list[dict[str, object]]]:
    """Return a config copy with validated coherent rear-swing proposals."""

    result = deepcopy(config)
    unknown = sorted(
        set(replacements)
        - set(_REAR_SWING_REFERENCE_PROPOSAL_SETTING_KEYS)
    )
    if unknown:
        raise ValueError(
            "Unknown rear-swing reference proposal setting overrides: "
            f"{unknown}."
        )
    merged = {
        "rear_swing_reference_proposal_ref_ids": deepcopy(
            result.get("rear_swing_reference_proposal_ref_ids", [])
        ),
        "rear_swing_reference_proposal_scales": deepcopy(
            result.get("rear_swing_reference_proposal_scales", [])
        ),
        "rear_swing_reference_proposal_joint_mask_leg": deepcopy(
            result.get(
                "rear_swing_reference_proposal_joint_mask_leg",
                [0] * 12,
            )
        ),
        "rear_swing_reference_proposal_lead_steps": deepcopy(
            result.get(
                "rear_swing_reference_proposal_lead_steps",
                0,
            )
        ),
        "rear_swing_action_residual_lead_steps": deepcopy(
            result.get("rear_swing_action_residual_lead_steps")
        ),
        "rear_swing_tracking_error_proposal_scales": deepcopy(
            result.get(
                "rear_swing_tracking_error_proposal_scales",
                [],
            )
        ),
        "rear_swing_tracking_error_proposal_joint_mask_leg": deepcopy(
            result.get(
                "rear_swing_tracking_error_proposal_joint_mask_leg",
                result.get(
                    "rear_swing_reference_proposal_joint_mask_leg",
                    [0] * 12,
                ),
            )
        ),
        "rear_swing_tracking_error_proposal_start_frame": deepcopy(
            result.get(
                "rear_swing_tracking_error_proposal_start_frame",
                0,
            )
        ),
        "rear_swing_load_transfer_proposal_ref_ids": deepcopy(
            result.get(
                "rear_swing_load_transfer_proposal_ref_ids",
                [],
            )
        ),
        "rear_swing_load_transfer_proposal_scales": deepcopy(
            result.get(
                "rear_swing_load_transfer_proposal_scales",
                [],
            )
        ),
        "rear_swing_load_transfer_proposal_gain_by_wheel_leg_rad": deepcopy(
            result.get(
                "rear_swing_load_transfer_proposal_gain_by_wheel_leg_rad",
                [[0.0] * 12, [0.0] * 12],
            )
        ),
        "rear_swing_load_transfer_proposal_start_frame": deepcopy(
            result.get(
                "rear_swing_load_transfer_proposal_start_frame",
                0,
            )
        ),
        "rear_swing_load_transfer_proposal_start_frame_by_wheel": deepcopy(
            result.get(
                "rear_swing_load_transfer_proposal_start_frame_by_wheel"
            )
        ),
        "rear_swing_load_transfer_proposal_gate_mode": deepcopy(
            result.get(
                "rear_swing_load_transfer_proposal_gate_mode",
                "swing_schedule",
            )
        ),
        "rear_swing_load_transfer_proposal_imbalance_threshold_n": deepcopy(
            result.get(
                "rear_swing_load_transfer_proposal_imbalance_threshold_n",
                0.0,
            )
        ),
        **deepcopy(replacements),
    }
    ref_ids = merged["rear_swing_reference_proposal_ref_ids"]
    if (
        not isinstance(ref_ids, (list, tuple))
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in ref_ids
        )
        or len(set(ref_ids)) != len(ref_ids)
    ):
        raise ValueError(
            "rear_swing_reference_proposal_ref_ids must contain unique "
            "integer values."
        )
    configured_entries = (
        result.get("nominal_action_reference", {}).get("entries", [])
    )
    valid_ref_ids = {
        int(entry["ref_id"])
        for entry in configured_entries
        if isinstance(entry, dict) and "ref_id" in entry
    }
    if any(ref_id not in valid_ref_ids for ref_id in ref_ids):
        raise ValueError(
            "rear_swing_reference_proposal_ref_ids must identify active "
            "nominal-action references."
        )

    scales = merged["rear_swing_reference_proposal_scales"]
    if not isinstance(scales, (list, tuple)):
        raise ValueError(
            "rear_swing_reference_proposal_scales must be a list."
        )
    try:
        scales = [float(value) for value in scales]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "rear_swing_reference_proposal_scales must contain finite "
            "positive values no greater than 1."
        ) from exc
    if (
        any(
            not math.isfinite(value)
            or value <= 0.0
            or value > 1.0
            for value in scales
        )
        or len(set(scales)) != len(scales)
    ):
        raise ValueError(
            "rear_swing_reference_proposal_scales must contain unique "
            "finite positive values no greater than 1."
        )
    mask = merged["rear_swing_reference_proposal_joint_mask_leg"]
    if not isinstance(mask, (list, tuple)) or len(mask) != 12:
        raise ValueError(
            "rear_swing_reference_proposal_joint_mask_leg must contain "
            "exactly 12 binary values."
        )
    if any(
        isinstance(value, bool) or value not in (0, 1)
        for value in mask
    ):
        raise ValueError(
            "rear_swing_reference_proposal_joint_mask_leg must contain only "
            "zeros and ones."
        )
    mask = [int(value) for value in mask]
    if any(mask[index] for index in (0, 1, 4, 5, 8, 9)):
        raise ValueError(
            "rear_swing_reference_proposal_joint_mask_leg must be zero for "
            "all front-leg joints."
        )
    lead_steps = merged["rear_swing_reference_proposal_lead_steps"]
    if (
        isinstance(lead_steps, bool)
        or not isinstance(lead_steps, int)
        or lead_steps < 0
        or (
            "horizon" in result
            and lead_steps >= int(result["horizon"])
        )
    ):
        raise ValueError(
            "rear_swing_reference_proposal_lead_steps must be a "
            "non-negative integer smaller than horizon."
        )
    residual_lead_steps = merged[
        "rear_swing_action_residual_lead_steps"
    ]
    if residual_lead_steps is None:
        residual_lead_steps = lead_steps
    if (
        isinstance(residual_lead_steps, bool)
        or not isinstance(residual_lead_steps, int)
        or residual_lead_steps < 0
        or (
            "horizon" in result
            and residual_lead_steps >= int(result["horizon"])
        )
    ):
        raise ValueError(
            "rear_swing_action_residual_lead_steps must be a "
            "non-negative integer smaller than horizon."
        )
    configured = bool(ref_ids or scales or any(mask) or lead_steps)
    complete = bool(ref_ids and scales and any(mask))
    if configured and not complete:
        raise ValueError(
            "Rear-swing reference proposals require non-empty ref IDs, "
            "scales, and at least one enabled rear joint."
        )
    tracking_error_scales = merged[
        "rear_swing_tracking_error_proposal_scales"
    ]
    if not isinstance(tracking_error_scales, (list, tuple)):
        raise ValueError(
            "rear_swing_tracking_error_proposal_scales must be a list."
        )
    try:
        tracking_error_scales = [
            float(value) for value in tracking_error_scales
        ]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "rear_swing_tracking_error_proposal_scales must contain finite "
            "positive values no greater than 1."
        ) from exc
    if (
        any(
            not math.isfinite(value)
            or value <= 0.0
            or value > 1.0
            for value in tracking_error_scales
        )
        or len(set(tracking_error_scales))
        != len(tracking_error_scales)
    ):
        raise ValueError(
            "rear_swing_tracking_error_proposal_scales must contain unique "
            "finite positive values no greater than 1."
        )
    if tracking_error_scales and not complete:
        raise ValueError(
            "Rear-swing tracking-error proposals require complete "
            "rear-swing reference proposal settings."
        )
    tracking_error_mask = merged[
        "rear_swing_tracking_error_proposal_joint_mask_leg"
    ]
    if (
        not isinstance(tracking_error_mask, (list, tuple))
        or len(tracking_error_mask) != 12
    ):
        raise ValueError(
            "rear_swing_tracking_error_proposal_joint_mask_leg must contain "
            "exactly 12 binary values."
        )
    if any(
        isinstance(value, bool) or value not in (0, 1)
        for value in tracking_error_mask
    ):
        raise ValueError(
            "rear_swing_tracking_error_proposal_joint_mask_leg must contain "
            "only zeros and ones."
        )
    tracking_error_mask = [
        int(value) for value in tracking_error_mask
    ]
    if any(
        tracking_error_mask[index]
        for index in (0, 1, 4, 5, 8, 9)
    ):
        raise ValueError(
            "rear_swing_tracking_error_proposal_joint_mask_leg must be zero "
            "for all front-leg joints."
        )
    if tracking_error_scales and not any(tracking_error_mask):
        raise ValueError(
            "Rear-swing tracking-error proposals require at least one "
            "enabled rear joint."
        )
    if (
        not tracking_error_scales
        and "rear_swing_tracking_error_proposal_joint_mask_leg"
        in replacements
    ):
        raise ValueError(
            "A rear-swing tracking-error joint mask requires non-empty "
            "tracking-error proposal scales."
        )
    tracking_error_start_frame = merged[
        "rear_swing_tracking_error_proposal_start_frame"
    ]
    if (
        isinstance(tracking_error_start_frame, bool)
        or not isinstance(tracking_error_start_frame, int)
        or tracking_error_start_frame < 0
    ):
        raise ValueError(
            "rear_swing_tracking_error_proposal_start_frame must be a "
            "non-negative integer."
        )
    if tracking_error_start_frame and not tracking_error_scales:
        raise ValueError(
            "A delayed rear-swing tracking-error proposal requires "
            "non-empty tracking-error proposal scales."
        )
    load_transfer_ref_ids = merged[
        "rear_swing_load_transfer_proposal_ref_ids"
    ]
    if (
        not isinstance(load_transfer_ref_ids, (list, tuple))
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in load_transfer_ref_ids
        )
        or len(set(load_transfer_ref_ids))
        != len(load_transfer_ref_ids)
    ):
        raise ValueError(
            "rear_swing_load_transfer_proposal_ref_ids must contain unique "
            "integer values."
        )
    if any(
        ref_id not in valid_ref_ids
        for ref_id in load_transfer_ref_ids
    ):
        raise ValueError(
            "rear_swing_load_transfer_proposal_ref_ids must identify active "
            "nominal-action references."
        )
    load_transfer_scales = merged[
        "rear_swing_load_transfer_proposal_scales"
    ]
    if not isinstance(load_transfer_scales, (list, tuple)):
        raise ValueError(
            "rear_swing_load_transfer_proposal_scales must be a list."
        )
    try:
        load_transfer_scales = [
            float(value) for value in load_transfer_scales
        ]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "rear_swing_load_transfer_proposal_scales must contain finite "
            "positive values no greater than 1."
        ) from exc
    if (
        any(
            not math.isfinite(value)
            or value <= 0.0
            or value > 1.0
            for value in load_transfer_scales
        )
        or len(set(load_transfer_scales))
        != len(load_transfer_scales)
    ):
        raise ValueError(
            "rear_swing_load_transfer_proposal_scales must contain unique "
            "finite positive values no greater than 1."
        )
    load_transfer_gain = merged[
        "rear_swing_load_transfer_proposal_gain_by_wheel_leg_rad"
    ]
    if (
        not isinstance(load_transfer_gain, (list, tuple))
        or len(load_transfer_gain) != 2
        or any(
            not isinstance(row, (list, tuple)) or len(row) != 12
            for row in load_transfer_gain
        )
    ):
        raise ValueError(
            "rear_swing_load_transfer_proposal_gain_by_wheel_leg_rad must "
            "contain exactly two rows of 12 values."
        )
    if any(
        isinstance(value, bool)
        for row in load_transfer_gain
        for value in row
    ):
        raise ValueError(
            "rear_swing_load_transfer_proposal_gain_by_wheel_leg_rad must "
            "contain only finite numeric values."
        )
    try:
        load_transfer_gain = [
            [float(value) for value in row]
            for row in load_transfer_gain
        ]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "rear_swing_load_transfer_proposal_gain_by_wheel_leg_rad must "
            "contain only finite numeric values."
        ) from exc
    if any(
        not math.isfinite(value)
        for row in load_transfer_gain
        for value in row
    ):
        raise ValueError(
            "rear_swing_load_transfer_proposal_gain_by_wheel_leg_rad must "
            "contain only finite numeric values."
        )
    load_transfer_start_frame = merged[
        "rear_swing_load_transfer_proposal_start_frame"
    ]
    if (
        isinstance(load_transfer_start_frame, bool)
        or not isinstance(load_transfer_start_frame, int)
        or load_transfer_start_frame < 0
    ):
        raise ValueError(
            "rear_swing_load_transfer_proposal_start_frame must be a "
            "non-negative integer."
        )
    load_transfer_start_frame_by_wheel = merged[
        "rear_swing_load_transfer_proposal_start_frame_by_wheel"
    ]
    if load_transfer_start_frame_by_wheel is not None:
        if (
            not isinstance(
                load_transfer_start_frame_by_wheel,
                (list, tuple),
            )
            or len(load_transfer_start_frame_by_wheel) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in load_transfer_start_frame_by_wheel
            )
        ):
            raise ValueError(
                "rear_swing_load_transfer_proposal_start_frame_by_wheel "
                "must contain exactly two non-negative integers ordered "
                "by RL then RR swing."
            )
        load_transfer_start_frame_by_wheel = list(
            load_transfer_start_frame_by_wheel
        )
    load_transfer_gate_mode = merged[
        "rear_swing_load_transfer_proposal_gate_mode"
    ]
    if load_transfer_gate_mode not in (
        "swing_schedule",
        "rear_force_imbalance",
    ):
        raise ValueError(
            "rear_swing_load_transfer_proposal_gate_mode must be "
            "'swing_schedule' or 'rear_force_imbalance'."
        )
    load_transfer_imbalance_threshold_n = merged[
        "rear_swing_load_transfer_proposal_imbalance_threshold_n"
    ]
    if isinstance(load_transfer_imbalance_threshold_n, bool):
        raise ValueError(
            "rear_swing_load_transfer_proposal_imbalance_threshold_n must "
            "be finite and non-negative."
        )
    try:
        load_transfer_imbalance_threshold_n = float(
            load_transfer_imbalance_threshold_n
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "rear_swing_load_transfer_proposal_imbalance_threshold_n must "
            "be finite and non-negative."
        ) from exc
    if (
        not math.isfinite(load_transfer_imbalance_threshold_n)
        or load_transfer_imbalance_threshold_n < 0.0
    ):
        raise ValueError(
            "rear_swing_load_transfer_proposal_imbalance_threshold_n must "
            "be finite and non-negative."
        )
    if (
        load_transfer_gate_mode == "rear_force_imbalance"
        and load_transfer_imbalance_threshold_n <= 0.0
    ):
        raise ValueError(
            "rear_force_imbalance load-transfer gating requires a positive "
            "imbalance threshold."
        )
    if (
        load_transfer_gate_mode == "swing_schedule"
        and load_transfer_imbalance_threshold_n != 0.0
    ):
        raise ValueError(
            "A load-transfer imbalance threshold requires "
            "rear_force_imbalance gate mode."
        )
    load_transfer_configured = bool(
        load_transfer_ref_ids
        or load_transfer_scales
        or any(value != 0.0 for row in load_transfer_gain for value in row)
        or load_transfer_start_frame
        or (
            load_transfer_start_frame_by_wheel is not None
            and any(load_transfer_start_frame_by_wheel)
        )
        or load_transfer_gate_mode != "swing_schedule"
        or load_transfer_imbalance_threshold_n != 0.0
    )
    load_transfer_complete = bool(
        load_transfer_ref_ids
        and load_transfer_scales
        and any(
            value != 0.0
            for row in load_transfer_gain
            for value in row
        )
    )
    if load_transfer_configured and not load_transfer_complete:
        raise ValueError(
            "Rear-swing load-transfer proposals require non-empty ref IDs, "
            "scales, and at least one non-zero wheel-specific leg gain."
        )
    combine_load_transfer_front = result.get(
        "combine_rear_swing_load_transfer_front_support_proposals",
        False,
    )
    if combine_load_transfer_front and not load_transfer_complete:
        raise ValueError(
            "Combined rear-swing load-transfer/front-support proposals "
            "require complete load-transfer and front-support proposal "
            "settings."
        )
    if (
        combine_load_transfer_front
        and load_transfer_gate_mode != "swing_schedule"
    ):
        raise ValueError(
            "Combined rear-swing load-transfer/front-support proposals "
            "require swing_schedule load-transfer gating."
        )
    combine_reference_load_transfer_front = result.get(
        "combine_rear_swing_reference_load_transfer_front_support_proposals",
        False,
    )
    front_complete = bool(
        result.get("front_support_proposal_ref_ids", [])
        and result.get("front_support_proposal_scales", [])
        and any(
            result.get(
                "front_support_proposal_gain_leg_rad",
                [0.0] * 12,
            )
        )
    )
    if (
        combine_reference_load_transfer_front
        and not (complete and load_transfer_complete and front_complete)
    ):
        raise ValueError(
            "Combined rear-swing reference/load-transfer/front-support "
            "proposals require complete rear-swing, load-transfer, and "
            "front-support proposal settings."
        )
    if (
        combine_reference_load_transfer_front
        and load_transfer_gate_mode != "swing_schedule"
    ):
        raise ValueError(
            "Combined rear-swing reference/load-transfer/front-support "
            "proposals require swing_schedule load-transfer gating."
        )
    samples = int(result["samples"])
    if _structured_proposal_count(
        result,
        rear_scales=scales,
        tracking_error_scales=tracking_error_scales,
        load_transfer_scales=load_transfer_scales,
    ) >= samples:
        raise ValueError(
            "Structured proposals must leave at least one MPPI sample for "
            "the stochastic population."
        )

    normalized = {
        "rear_swing_reference_proposal_ref_ids": list(ref_ids),
        "rear_swing_reference_proposal_scales": scales,
        "rear_swing_reference_proposal_joint_mask_leg": mask,
        "rear_swing_reference_proposal_lead_steps": lead_steps,
        "rear_swing_action_residual_lead_steps": (
            residual_lead_steps
        ),
        "rear_swing_tracking_error_proposal_scales": (
            tracking_error_scales
        ),
        "rear_swing_tracking_error_proposal_joint_mask_leg": (
            tracking_error_mask
        ),
        "rear_swing_tracking_error_proposal_start_frame": (
            tracking_error_start_frame
        ),
        "rear_swing_load_transfer_proposal_ref_ids": list(
            load_transfer_ref_ids
        ),
        "rear_swing_load_transfer_proposal_scales": (
            load_transfer_scales
        ),
        "rear_swing_load_transfer_proposal_gain_by_wheel_leg_rad": (
            load_transfer_gain
        ),
        "rear_swing_load_transfer_proposal_start_frame": (
            load_transfer_start_frame
        ),
        "rear_swing_load_transfer_proposal_start_frame_by_wheel": (
            load_transfer_start_frame_by_wheel
        ),
        "rear_swing_load_transfer_proposal_gate_mode": (
            load_transfer_gate_mode
        ),
        "rear_swing_load_transfer_proposal_imbalance_threshold_n": (
            load_transfer_imbalance_threshold_n
        ),
    }
    records: list[dict[str, object]] = []
    for name in _REAR_SWING_REFERENCE_PROPOSAL_SETTING_KEYS:
        if name not in replacements:
            continue
        records.append(
            {
                "name": name,
                "previous": deepcopy(result.get(name)),
                "replacement": deepcopy(normalized[name]),
            }
        )
        result[name] = deepcopy(normalized[name])
    return result, records


def replace_front_support_proposal_settings(
    config: dict,
    replacements: dict[str, Any],
) -> tuple[dict, list[dict[str, object]]]:
    """Return a config copy with validated front-support proposals."""

    result = deepcopy(config)
    unknown = sorted(
        set(replacements) - set(_FRONT_SUPPORT_PROPOSAL_SETTING_KEYS)
    )
    if unknown:
        raise ValueError(
            f"Unknown front-support proposal setting overrides: {unknown}."
        )
    merged = {
        "front_support_proposal_ref_ids": deepcopy(
            result.get("front_support_proposal_ref_ids", [])
        ),
        "front_support_proposal_scales": deepcopy(
            result.get("front_support_proposal_scales", [])
        ),
        "front_support_proposal_gain_leg_rad": deepcopy(
            result.get("front_support_proposal_gain_leg_rad", [0.0] * 12)
        ),
        "front_support_proposal_start_frame": deepcopy(
            result.get("front_support_proposal_start_frame", 0)
        ),
        "combine_rear_swing_front_support_proposals": deepcopy(
            result.get(
                "combine_rear_swing_front_support_proposals",
                False,
            )
        ),
        "combine_rear_swing_load_transfer_front_support_proposals": deepcopy(
            result.get(
                "combine_rear_swing_load_transfer_front_support_proposals",
                False,
            )
        ),
        "combine_rear_swing_reference_load_transfer_front_support_proposals": deepcopy(
            result.get(
                "combine_rear_swing_reference_load_transfer_front_support_"
                "proposals",
                False,
            )
        ),
        "include_rear_support_reference_in_coordinated_proposals": deepcopy(
            result.get(
                "include_rear_support_reference_in_coordinated_proposals",
                False,
            )
        ),
        "rear_support_reference_proposal_start_frame": deepcopy(
            result.get(
                "rear_support_reference_proposal_start_frame",
                0,
            )
        ),
        **deepcopy(replacements),
    }

    ref_ids = merged["front_support_proposal_ref_ids"]
    if (
        not isinstance(ref_ids, (list, tuple))
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in ref_ids
        )
        or len(set(ref_ids)) != len(ref_ids)
    ):
        raise ValueError(
            "front_support_proposal_ref_ids must contain unique integer "
            "values."
        )
    configured_entries = (
        result.get("nominal_action_reference", {}).get("entries", [])
    )
    valid_ref_ids = {
        int(entry["ref_id"])
        for entry in configured_entries
        if isinstance(entry, dict) and "ref_id" in entry
    }
    if any(ref_id not in valid_ref_ids for ref_id in ref_ids):
        raise ValueError(
            "front_support_proposal_ref_ids must identify active "
            "nominal-action references."
        )

    scales = merged["front_support_proposal_scales"]
    if not isinstance(scales, (list, tuple)):
        raise ValueError("front_support_proposal_scales must be a list.")
    try:
        scales = [float(value) for value in scales]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "front_support_proposal_scales must contain finite positive "
            "values no greater than 1."
        ) from exc
    if (
        any(
            not math.isfinite(value)
            or value <= 0.0
            or value > 1.0
            for value in scales
        )
        or len(set(scales)) != len(scales)
    ):
        raise ValueError(
            "front_support_proposal_scales must contain unique finite "
            "positive values no greater than 1."
        )

    gains = merged["front_support_proposal_gain_leg_rad"]
    if not isinstance(gains, (list, tuple)) or len(gains) != 12:
        raise ValueError(
            "front_support_proposal_gain_leg_rad must contain exactly 12 "
            "numeric values."
        )
    try:
        gains = [float(value) for value in gains]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "front_support_proposal_gain_leg_rad must contain exactly 12 "
            "numeric values."
        ) from exc
    if not all(math.isfinite(value) for value in gains):
        raise ValueError(
            "front_support_proposal_gain_leg_rad must contain only finite "
            "values."
        )
    if any(gains[index] != 0.0 for index in (2, 3, 6, 7, 10, 11)):
        raise ValueError(
            "front_support_proposal_gain_leg_rad must be zero for all "
            "rear-leg joints."
        )

    start_frame = merged["front_support_proposal_start_frame"]
    if (
        isinstance(start_frame, bool)
        or not isinstance(start_frame, int)
        or start_frame < 0
    ):
        raise ValueError(
            "front_support_proposal_start_frame must be a non-negative "
            "integer."
        )
    configured = bool(ref_ids or scales or any(gains) or start_frame)
    complete = bool(ref_ids and scales and any(gains))
    if configured and not complete:
        raise ValueError(
            "Front-support proposals require non-empty ref IDs, scales, and "
            "at least one non-zero front-leg gain."
        )
    combine = merged[
        "combine_rear_swing_front_support_proposals"
    ]
    if not isinstance(combine, bool):
        raise ValueError(
            "combine_rear_swing_front_support_proposals must be a boolean."
        )
    rear_complete = bool(
        result.get("rear_swing_reference_proposal_ref_ids", [])
        and result.get("rear_swing_reference_proposal_scales", [])
        and any(
            result.get(
                "rear_swing_reference_proposal_joint_mask_leg",
                [0] * 12,
            )
        )
    )
    if combine and not (rear_complete and complete):
        raise ValueError(
            "Combined rear-swing/front-support proposals require complete "
            "rear-swing and front-support proposal settings."
        )
    combine_load_transfer_front = merged[
        "combine_rear_swing_load_transfer_front_support_proposals"
    ]
    if not isinstance(combine_load_transfer_front, bool):
        raise ValueError(
            "combine_rear_swing_load_transfer_front_support_proposals must "
            "be a boolean."
        )
    load_transfer_complete = bool(
        result.get("rear_swing_load_transfer_proposal_ref_ids", [])
        and result.get("rear_swing_load_transfer_proposal_scales", [])
        and any(
            value != 0.0
            for row in result.get(
                "rear_swing_load_transfer_proposal_gain_by_wheel_leg_rad",
                [[0.0] * 12, [0.0] * 12],
            )
            for value in row
        )
    )
    if (
        combine_load_transfer_front
        and not (load_transfer_complete and complete)
    ):
        raise ValueError(
            "Combined rear-swing load-transfer/front-support proposals "
            "require complete load-transfer and front-support proposal "
            "settings."
        )
    if (
        combine_load_transfer_front
        and result.get(
            "rear_swing_load_transfer_proposal_gate_mode",
            "swing_schedule",
        )
        != "swing_schedule"
    ):
        raise ValueError(
            "Combined rear-swing load-transfer/front-support proposals "
            "require swing_schedule load-transfer gating."
        )
    combine_reference_load_transfer_front = merged[
        "combine_rear_swing_reference_load_transfer_front_support_proposals"
    ]
    if not isinstance(combine_reference_load_transfer_front, bool):
        raise ValueError(
            "combine_rear_swing_reference_load_transfer_front_support_"
            "proposals must be a boolean."
        )
    if (
        combine_reference_load_transfer_front
        and not (rear_complete and load_transfer_complete and complete)
    ):
        raise ValueError(
            "Combined rear-swing reference/load-transfer/front-support "
            "proposals require complete rear-swing, load-transfer, and "
            "front-support proposal settings."
        )
    if (
        combine_reference_load_transfer_front
        and result.get(
            "rear_swing_load_transfer_proposal_gate_mode",
            "swing_schedule",
        )
        != "swing_schedule"
    ):
        raise ValueError(
            "Combined rear-swing reference/load-transfer/front-support "
            "proposals require swing_schedule load-transfer gating."
        )
    include_rear_support_reference = merged[
        "include_rear_support_reference_in_coordinated_proposals"
    ]
    if not isinstance(include_rear_support_reference, bool):
        raise ValueError(
            "include_rear_support_reference_in_coordinated_proposals must "
            "be a boolean."
        )
    if include_rear_support_reference and not combine:
        raise ValueError(
            "Rear-support reference coordination requires combined "
            "rear-swing/front-support proposals."
        )
    rear_support_start_frame = merged[
        "rear_support_reference_proposal_start_frame"
    ]
    if (
        isinstance(rear_support_start_frame, bool)
        or not isinstance(rear_support_start_frame, int)
        or rear_support_start_frame < 0
    ):
        raise ValueError(
            "rear_support_reference_proposal_start_frame must be a "
            "non-negative integer."
        )
    if rear_support_start_frame and not include_rear_support_reference:
        raise ValueError(
            "A delayed rear-support reference proposal requires "
            "rear-support reference coordination."
        )
    structured_proposal_count = _structured_proposal_count(
        result,
        front_scales=scales,
        combine=combine,
        combine_load_transfer_front=combine_load_transfer_front,
        combine_reference_load_transfer_front=(
            combine_reference_load_transfer_front
        ),
        include_rear_support=include_rear_support_reference,
    )
    if structured_proposal_count >= int(result["samples"]):
        raise ValueError(
            "Structured proposals must leave at least one MPPI sample for "
            "the stochastic population."
        )

    normalized = {
        "front_support_proposal_ref_ids": list(ref_ids),
        "front_support_proposal_scales": scales,
        "front_support_proposal_gain_leg_rad": gains,
        "front_support_proposal_start_frame": start_frame,
        "combine_rear_swing_front_support_proposals": combine,
        "combine_rear_swing_load_transfer_front_support_proposals": (
            combine_load_transfer_front
        ),
        "combine_rear_swing_reference_load_transfer_front_support_proposals": (
            combine_reference_load_transfer_front
        ),
        "include_rear_support_reference_in_coordinated_proposals": (
            include_rear_support_reference
        ),
        "rear_support_reference_proposal_start_frame": (
            rear_support_start_frame
        ),
    }
    records: list[dict[str, object]] = []
    for name in _FRONT_SUPPORT_PROPOSAL_SETTING_KEYS:
        if name not in replacements:
            continue
        records.append(
            {
                "name": name,
                "previous": deepcopy(result.get(name)),
                "replacement": deepcopy(normalized[name]),
            }
        )
        result[name] = deepcopy(normalized[name])
    return result, records


def scale_noise_std(
    config: dict,
    scale: float,
) -> tuple[dict, dict[str, object]]:
    """Return a config copy with its 12-D MPPI noise vector scaled."""

    result = deepcopy(config)
    configured = result.get("noise_std_leg")
    if not isinstance(configured, (list, tuple)) or len(configured) != 12:
        raise ValueError(
            "Config noise_std_leg must contain exactly 12 values."
        )
    try:
        normalized = [float(value) for value in configured]
        normalized_scale = float(scale)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "MPPI noise values and --noise-scale must be numeric."
        ) from exc
    if (
        not math.isfinite(normalized_scale)
        or normalized_scale <= 0.0
    ):
        raise ValueError("--noise-scale must be finite and positive.")
    if (
        not all(math.isfinite(value) for value in normalized)
        or any(value < 0.0 for value in normalized)
        or not any(value > 0.0 for value in normalized)
    ):
        raise ValueError(
            "Config noise_std_leg must contain finite non-negative values "
            "and at least one positive value."
        )
    replacement = [value * normalized_scale for value in normalized]
    if not all(math.isfinite(value) for value in replacement):
        raise ValueError("Scaled MPPI noise values must remain finite.")
    result["noise_std_leg"] = replacement
    return result, {
        "scale": normalized_scale,
        "previous": normalized,
        "replacement": replacement,
    }


def _parse_replacement(value: str) -> tuple[int, Path]:
    ref_text, separator, path_text = value.partition("=")
    if not separator or not ref_text or not path_text:
        raise argparse.ArgumentTypeError(
            "Replacement must use REF_ID=PATH syntax."
        )
    try:
        ref_id = int(ref_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid replacement ref ID {ref_text!r}."
        ) from exc
    return ref_id, Path(path_text)


def _parse_solver_overrides(value: str) -> tuple[int, dict[str, Any]]:
    ref_text, separator, json_text = value.partition("=")
    if not separator or not ref_text or not json_text:
        raise argparse.ArgumentTypeError(
            "Solver overrides must use REF_ID=JSON_MAPPING syntax."
        )
    try:
        ref_id = int(ref_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid solver override ref ID {ref_text!r}."
        ) from exc
    try:
        configured = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid solver override JSON: {exc}."
        ) from exc
    if not isinstance(configured, dict):
        raise argparse.ArgumentTypeError(
            "Solver override JSON must decode to a mapping."
        )
    return ref_id, configured


def _parse_json_mapping(value: str) -> dict[str, Any]:
    try:
        configured = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid JSON mapping: {exc}."
        ) from exc
    if not isinstance(configured, dict):
        raise argparse.ArgumentTypeError(
            "JSON value must decode to a mapping."
        )
    return configured


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a fail-closed MPPI candidate config by replacing selected "
            "nominal-action paths and recomputing their SHA256 values."
        )
    )
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--note", required=True)
    parser.add_argument(
        "--replace",
        type=_parse_replacement,
        action="append",
        metavar="REF_ID=PATH",
    )
    parser.add_argument(
        "--solver-overrides",
        type=_parse_solver_overrides,
        action="append",
        metavar="REF_ID=JSON_MAPPING",
    )
    parser.add_argument(
        "--cost-overrides",
        type=_parse_json_mapping,
        metavar="JSON_MAPPING",
    )
    parser.add_argument(
        "--load-limit-overrides",
        type=_parse_json_mapping,
        metavar="JSON_MAPPING",
    )
    parser.add_argument(
        "--mppi-overrides",
        type=_parse_json_mapping,
        metavar="JSON_MAPPING",
    )
    parser.add_argument(
        "--output-feedback-overrides",
        type=_parse_json_mapping,
        metavar="JSON_MAPPING",
    )
    parser.add_argument(
        "--reference-proposal-overrides",
        type=_parse_json_mapping,
        metavar="JSON_MAPPING",
    )
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=None,
        help=(
            "Multiply all 12 configured MPPI noise standard deviations by "
            "one finite positive scalar."
        ),
    )
    args = parser.parse_args()

    base_path = args.base_config.expanduser().resolve()
    output_path = _inside_root(args.output, ROOT, "--output")
    report_path = _inside_root(args.report, ROOT, "--report")
    if output_path.exists() or report_path.exists():
        raise FileExistsError(
            "Refusing to overwrite an existing candidate config or report."
        )
    with base_path.open("r", encoding="utf-8") as stream:
        base = yaml.safe_load(stream)
    if not isinstance(base, dict):
        raise TypeError("Base config must be a YAML mapping.")
    action_arguments = args.replace or []
    solver_arguments = args.solver_overrides or []
    cost_replacements = args.cost_overrides or {}
    load_limit_replacements = args.load_limit_overrides or {}
    mppi_replacements = args.mppi_overrides or {}
    output_feedback_replacements = (
        args.output_feedback_overrides or {}
    )
    reference_proposal_replacements = (
        args.reference_proposal_overrides or {}
    )
    if (
        not action_arguments
        and not solver_arguments
        and not cost_replacements
        and not load_limit_replacements
        and not mppi_replacements
        and not output_feedback_replacements
        and not reference_proposal_replacements
        and args.noise_scale is None
    ):
        raise ValueError(
            "At least one --replace, --solver-overrides, or "
            "--cost-overrides, --load-limit-overrides, --mppi-overrides, "
            "--output-feedback-overrides, "
            "--reference-proposal-overrides, or --noise-scale is required."
        )
    replacements = dict(action_arguments)
    if len(replacements) != len(action_arguments):
        raise ValueError("Each replacement ref ID may appear only once.")
    solver_replacements = dict(solver_arguments)
    if len(solver_replacements) != len(solver_arguments):
        raise ValueError(
            "Each solver override ref ID may appear only once."
        )

    result = base
    records: list[dict[str, object]] = []
    if replacements:
        result, records = replace_nominal_action_entries(
            result,
            replacements,
            root=ROOT,
        )
    solver_records: list[dict[str, object]] = []
    if solver_replacements:
        result, solver_records = replace_nominal_solver_overrides(
            result,
            solver_replacements,
        )
    cost_records: list[dict[str, object]] = []
    if cost_replacements:
        result, cost_records = replace_cost_weights(
            result,
            cost_replacements,
        )
    load_limit_records: list[dict[str, object]] = []
    if load_limit_replacements:
        result, load_limit_records = replace_load_limits(
            result,
            load_limit_replacements,
        )
    mppi_records: list[dict[str, object]] = []
    if mppi_replacements:
        result, mppi_records = replace_mppi_settings(
            result,
            mppi_replacements,
        )
    output_feedback_records: list[dict[str, object]] = []
    if output_feedback_replacements:
        result, output_feedback_records = (
            replace_output_feedback_settings(
                result,
                output_feedback_replacements,
            )
        )
    reference_proposal_records: list[dict[str, object]] = []
    if reference_proposal_replacements:
        unknown_reference_proposal_keys = sorted(
            set(reference_proposal_replacements)
            - set(_REAR_SWING_REFERENCE_PROPOSAL_SETTING_KEYS)
            - set(_FRONT_SUPPORT_PROPOSAL_SETTING_KEYS)
        )
        if unknown_reference_proposal_keys:
            raise ValueError(
                "Unknown reference proposal setting overrides: "
                f"{unknown_reference_proposal_keys}."
            )
        rear_proposal_replacements = {
            key: value
            for key, value in reference_proposal_replacements.items()
            if key in _REAR_SWING_REFERENCE_PROPOSAL_SETTING_KEYS
        }
        front_proposal_replacements = {
            key: value
            for key, value in reference_proposal_replacements.items()
            if key in _FRONT_SUPPORT_PROPOSAL_SETTING_KEYS
        }
        if rear_proposal_replacements:
            result, rear_proposal_records = (
                replace_rear_swing_reference_proposal_settings(
                    result,
                    rear_proposal_replacements,
                )
            )
            reference_proposal_records.extend(rear_proposal_records)
        if front_proposal_replacements:
            result, front_proposal_records = (
                replace_front_support_proposal_settings(
                    result,
                    front_proposal_replacements,
                )
            )
            reference_proposal_records.extend(front_proposal_records)
    noise_record: dict[str, object] | None = None
    if args.noise_scale is not None:
        result, noise_record = scale_noise_std(
            result,
            args.noise_scale,
        )
    result["status"] = args.status
    result["note"] = args.note
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as stream:
        yaml.safe_dump(
            result,
            stream,
            sort_keys=False,
            allow_unicode=True,
            width=120,
        )
    with output_path.open("r", encoding="utf-8") as stream:
        reloaded = yaml.safe_load(stream)
    if reloaded != result:
        raise RuntimeError("Written YAML does not round-trip exactly.")

    report = {
        "schema_version": "pcbc-mppi-candidate-config-build-v11",
        "status": "unqualified_debug_requires_full_requalification",
        "base_config": str(base_path),
        "base_config_sha256": sha256_file(base_path),
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "candidate_status": args.status,
        "note": args.note,
        "replacements": records,
        "solver_override_replacements": solver_records,
        "cost_weight_replacements": cost_records,
        "load_limit_replacements": load_limit_records,
        "mppi_setting_replacements": mppi_records,
        "output_feedback_replacements": output_feedback_records,
        "reference_proposal_replacements": (
            reference_proposal_records
        ),
        "noise_std_scale": noise_record,
    }
    write_json(report_path, report)
    print(report)


if __name__ == "__main__":
    main()
