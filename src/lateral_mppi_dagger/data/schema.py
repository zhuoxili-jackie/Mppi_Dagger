from __future__ import annotations

import fcntl
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from lateral_mppi_dagger.expert.base import (
    LEGACY_MPPI_COST_COMPONENT_NAMES,
    MPPI_COST_COMPONENT_NAMES,
    FailureCode,
    LabelSource,
)


SCHEMA_VERSION = "pcbc-dagger-episode-v1"

FIELD_SPECS: dict[str, tuple[np.dtype, tuple[int, ...]]] = {
    "step_id": (np.dtype(np.int32), ()),
    "sim_time": (np.dtype(np.float64), ()),
    "obs93_clean": (np.dtype(np.float32), (93,)),
    "obs93_train": (np.dtype(np.float32), (93,)),
    "next_obs93_clean": (np.dtype(np.float32), (93,)),
    "teacher_action16": (np.dtype(np.float32), (16,)),
    "student_action16": (np.dtype(np.float32), (16,)),
    "pre_shield_action16": (np.dtype(np.float32), (16,)),
    "executed_action16": (np.dtype(np.float32), (16,)),
    "action_clip_delta16": (np.dtype(np.float32), (16,)),
    "teacher_q_des_leg": (np.dtype(np.float32), (12,)),
    "teacher_valid": (np.dtype(np.uint8), ()),
    "student_valid": (np.dtype(np.uint8), ()),
    "label_source": (np.dtype(np.uint8), ()),
    "behavior_policy": (np.dtype(np.uint8), ()),
    "teacher_takeover": (np.dtype(np.uint8), ()),
    "shield_intervened": (np.dtype(np.uint8), ()),
    "ref_id": (np.dtype(np.int32), ()),
    "ref_frame": (np.dtype(np.int32), ()),
    "phase": (np.dtype(np.float32), ()),
    "target_vy": (np.dtype(np.float32), ()),
    "desired_contact": (np.dtype(np.uint8), (4,)),
    "measured_contact": (np.dtype(np.uint8), (4,)),
    "base_pose_w": (np.dtype(np.float32), (7,)),
    "base_twist_w": (np.dtype(np.float32), (6,)),
    "q": (np.dtype(np.float32), (16,)),
    "dq": (np.dtype(np.float32), (16,)),
    "wheel_body_pose_w": (np.dtype(np.float32), (4, 7)),
    "contact_force_w": (np.dtype(np.float32), (4, 3)),
    "solver_status": (np.dtype(np.int16), ()),
    "solve_ms": (np.dtype(np.float32), ()),
    "safety_margin": (np.dtype(np.float32), ()),
    "failure_code": (np.dtype(np.int16), ()),
    "terminal": (np.dtype(np.uint8), ()),
    "termination_reason": (np.dtype(np.int16), ()),
}

OPTIONAL_FIELD_SPECS: dict[str, tuple[np.dtype, tuple[int, ...]]] = {
    "scheduled_action16": (np.dtype(np.float32), (16,)),
    "obs93_dynamic": (np.dtype(np.float32), (93,)),
    "next_obs93_dynamic": (np.dtype(np.float32), (93,)),
    "wheel_body_twist_w": (np.dtype(np.float32), (4, 6)),
    "mppi_cost_components": (
        np.dtype(np.float32),
        (len(MPPI_COST_COMPONENT_NAMES),),
    ),
    "mppi_minimum_total_cost": (np.dtype(np.float32), ()),
    "mppi_mean_total_cost": (np.dtype(np.float32), ()),
    "mppi_effective_sample_size": (np.dtype(np.float32), ()),
    "mppi_rollout_termination_rate": (np.dtype(np.float32), ()),
}

ENUMS = {
    "label_source": {item.name: int(item) for item in LabelSource},
    "failure_code": {item.name: int(item) for item in FailureCode},
    "behavior_policy": {"TEACHER": 0, "STUDENT": 1, "FALLBACK": 2},
    "termination_reason": {
        "NONE": 0,
        "TIME_LIMIT": 1,
        "ENV_TERMINATED": 2,
        "EXPERT_INVALID": 3,
        "NAN_INF": 4,
        "SAFETY": 5,
    },
}


@dataclass(frozen=True)
class EpisodeShard:
    arrays: dict[str, np.ndarray]
    metadata: dict[str, Any]

    def validate(self) -> int:
        missing = sorted(set(FIELD_SPECS) - set(self.arrays))
        if missing:
            raise ValueError(f"Episode shard is missing fields: {missing}")
        first = np.asarray(self.arrays["step_id"])
        if first.ndim != 1 or first.shape[0] == 0:
            raise ValueError("Episode shard must contain at least one step.")
        steps = first.shape[0]
        for name, (expected_dtype, tail_shape) in {**FIELD_SPECS, **OPTIONAL_FIELD_SPECS}.items():
            if name not in self.arrays:
                continue
            value = np.asarray(self.arrays[name])
            if name == "mppi_cost_components":
                allowed_widths = {
                    len(LEGACY_MPPI_COST_COMPONENT_NAMES),
                    len(MPPI_COST_COMPONENT_NAMES),
                }
                if (
                    value.ndim != 2
                    or value.shape[0] != steps
                    or value.shape[1] not in allowed_widths
                ):
                    raise ValueError(
                        "mppi_cost_components expected shape "
                        f"[{steps}, one of {sorted(allowed_widths)}], got "
                        f"{value.shape}"
                    )
                if value.dtype != expected_dtype:
                    raise TypeError(
                        "mppi_cost_components expected dtype "
                        f"{expected_dtype}, got {value.dtype}"
                    )
                component_order = self.metadata.get(
                    "mppi_cost_component_order"
                )
                if (
                    component_order is not None
                    and len(component_order) != value.shape[1]
                ):
                    raise ValueError(
                        "mppi_cost_component_order length does not match the "
                        "stored diagnostic vector."
                    )
                continue
            expected_shape = (steps,) + tail_shape
            if value.shape != expected_shape:
                raise ValueError(f"{name} expected shape {expected_shape}, got {value.shape}")
            if value.dtype != expected_dtype:
                raise TypeError(f"{name} expected dtype {expected_dtype}, got {value.dtype}")

        if not np.array_equal(first, np.arange(steps, dtype=np.int32)):
            raise ValueError("step_id must be contiguous and start at zero.")
        if not np.all(np.diff(self.arrays["sim_time"]) > 0.0):
            raise ValueError("sim_time must be strictly increasing.")
        for name in (
            "obs93_clean",
            "obs93_train",
            "next_obs93_clean",
            "pre_shield_action16",
            "scheduled_action16",
            "executed_action16",
            "action_clip_delta16",
            "base_pose_w",
            "base_twist_w",
            "q",
            "dq",
            "wheel_body_pose_w",
            "wheel_body_twist_w",
            "contact_force_w",
        ):
            if name in self.arrays and not np.isfinite(self.arrays[name]).all():
                raise ValueError(f"{name} contains NaN or Inf in a required-valid field.")

        if "mppi_cost_components" in self.arrays:
            mppi_rows = self.arrays["label_source"] == int(LabelSource.MPPI)
            for name in (
                "mppi_cost_components",
                "mppi_minimum_total_cost",
                "mppi_mean_total_cost",
                "mppi_effective_sample_size",
                "mppi_rollout_termination_rate",
            ):
                if name not in self.arrays:
                    raise ValueError(f"MPPI diagnostics are incomplete: missing {name}")
                if not np.isfinite(self.arrays[name][mppi_rows]).all():
                    raise ValueError(f"{name} contains NaN or Inf for an MPPI label.")

        teacher_valid = self.arrays["teacher_valid"].astype(bool)
        for name in ("teacher_action16", "teacher_q_des_leg", "solve_ms", "safety_margin"):
            if not np.isfinite(self.arrays[name][teacher_valid]).all():
                raise ValueError(f"{name} contains NaN or Inf where teacher_valid=1.")
        student_valid = self.arrays["student_valid"].astype(bool)
        if not np.isfinite(self.arrays["student_action16"][student_valid]).all():
            raise ValueError("student_action16 contains NaN or Inf where student_valid=1.")
        if np.any(self.arrays["terminal"][:-1]):
            raise ValueError("terminal may only be true on the last saved transition.")
        if steps > 1:
            if not np.array_equal(
                self.arrays["next_obs93_clean"][:-1],
                self.arrays["obs93_clean"][1:],
            ):
                raise ValueError(
                    "next_obs93_clean[t] must equal obs93_clean[t+1] before the terminal row."
                )
            if not np.array_equal(
                self.arrays["obs93_clean"][1:, 73:89],
                self.arrays["executed_action16"][:-1],
            ):
                raise ValueError(
                    "Observation previous-action slice does not equal the prior physically executed action."
                )
        if not np.array_equal(
            self.arrays["obs93_clean"][0, 73:89],
            np.zeros(16, dtype=np.float32),
        ):
            raise ValueError("Reset observation previous-action slice must be exact zero.")
        if "scheduled_action16" in self.arrays:
            scheduled = self.arrays["scheduled_action16"]
            if not np.array_equal(
                self.arrays["action_clip_delta16"],
                scheduled - self.arrays["pre_shield_action16"],
            ):
                raise ValueError(
                    "action_clip_delta16 must describe pre-shield to scheduled-command clipping."
                )
            delay = int(self.metadata.get("action_delay_steps", 0))
            if delay < 0:
                raise ValueError("metadata action_delay_steps must be non-negative.")
            expected_applied = np.zeros_like(scheduled)
            if delay == 0:
                expected_applied[:] = scheduled
            elif delay < steps:
                expected_applied[delay:] = scheduled[:-delay]
            if not np.array_equal(
                self.arrays["executed_action16"],
                expected_applied,
            ):
                raise ValueError(
                    "executed_action16 does not match scheduled_action16 and the recorded FIFO delay."
                )
        if self.metadata.get("wheel_action_mode") == "hard_zero":
            zero_fields = ["pre_shield_action16", "executed_action16"]
            if "scheduled_action16" in self.arrays:
                zero_fields.append("scheduled_action16")
            for name in zero_fields:
                if not np.array_equal(
                    self.arrays[name][:, 12:],
                    np.zeros_like(self.arrays[name][:, 12:]),
                ):
                    raise ValueError(f"{name} violates hard-zero wheel action mode.")
            teacher_valid = self.arrays["teacher_valid"].astype(bool)
            if not np.array_equal(
                self.arrays["teacher_action16"][teacher_valid, 12:],
                np.zeros_like(self.arrays["teacher_action16"][teacher_valid, 12:]),
            ):
                raise ValueError("Valid teacher labels violate hard-zero wheel action mode.")
            student_valid = self.arrays["student_valid"].astype(bool)
            if not np.array_equal(
                self.arrays["student_action16"][student_valid, 12:],
                np.zeros_like(self.arrays["student_action16"][student_valid, 12:]),
            ):
                raise ValueError("Valid student actions violate hard-zero wheel action mode.")
        if self.metadata.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"metadata schema_version must be {SCHEMA_VERSION!r}, got {self.metadata.get('schema_version')!r}"
            )
        return steps


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot JSON encode {type(value).__name__}")


def assign_episode_split(episode_id: str) -> str:
    import hashlib

    bucket = int(hashlib.sha256(episode_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "test"


def write_episode_shard(
    dataset_root: str | Path,
    episode_id: str,
    shard: EpisodeShard,
    split: str | None = None,
) -> dict[str, Any]:
    steps = shard.validate()
    root = Path(dataset_root).resolve()
    episodes_dir = root / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    root.mkdir(parents=True, exist_ok=True)
    path = episodes_dir / f"{episode_id}.npz"
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing episode shard: {path}")

    metadata_json = json.dumps(shard.metadata, ensure_ascii=False, sort_keys=True, default=_json_default)
    payload = dict(shard.arrays)
    payload["metadata_json"] = np.asarray(metadata_json)
    with tempfile.NamedTemporaryFile(prefix=f".{episode_id}.", suffix=".npz", dir=episodes_dir, delete=False) as stream:
        temporary_path = Path(stream.name)
        np.savez_compressed(stream, **payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary_path, path)

    record = {
        "schema_version": SCHEMA_VERSION,
        "episode_id": episode_id,
        "path": str(path.relative_to(root)),
        "split": split or assign_episode_split(episode_id),
        "steps": steps,
        "seed": shard.metadata.get("seed"),
        "scenario": shard.metadata.get("scenario"),
        "dagger_round": int(shard.metadata.get("dagger_round", 0)),
        "ref_id": shard.metadata.get("ref_id"),
        "success": bool(shard.metadata.get("success", False)),
        "teacher_valid_rate": float(np.mean(shard.arrays["teacher_valid"])),
        "shield_intervention_rate": float(np.mean(shard.arrays["shield_intervened"])),
    }
    manifest_path = root / "manifest.jsonl"
    encoded = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
    with manifest_path.open("a", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    return record


def read_episode_shard(path: str | Path) -> EpisodeShard:
    with np.load(path, allow_pickle=False) as archive:
        if "metadata_json" not in archive:
            raise ValueError(f"{path} has no metadata_json")
        metadata = json.loads(str(archive["metadata_json"].item()))
        arrays = {key: np.asarray(archive[key]) for key in archive.files if key != "metadata_json"}
    shard = EpisodeShard(arrays=arrays, metadata=metadata)
    shard.validate()
    return shard
