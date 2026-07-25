from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from lateral_mppi_dagger.config import PACKAGE_ROOT, load_yaml, sha256_file


REQUIRED_KEYS = (
    "fps",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)


@dataclass(frozen=True)
class ReferenceMotion:
    index: int
    path: Path
    sha256: str
    target_vy: float
    fps: int
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    body_pos_w: np.ndarray
    body_quat_w: np.ndarray
    body_lin_vel_w: np.ndarray
    body_ang_vel_w: np.ndarray
    source_kind: str = "npz"

    @property
    def frames(self) -> int:
        return int(self.joint_pos.shape[0])

    @property
    def duration_seconds(self) -> float:
        return (self.frames - 1) / self.fps

    def frame(self, frame_index: int) -> dict[str, np.ndarray | int | float]:
        index = int(np.clip(frame_index, 0, self.frames - 1))
        return {
            "ref_id": self.index,
            "ref_frame": index,
            "phase": index / max(self.frames - 1, 1),
            "target_vy": self.target_vy,
            "joint_pos": self.joint_pos[index],
            "joint_vel": self.joint_vel[index],
            "body_pos_w": self.body_pos_w[index],
            "body_quat_w": self.body_quat_w[index],
            "body_lin_vel_w": self.body_lin_vel_w[index],
            "body_ang_vel_w": self.body_ang_vel_w[index],
        }


def _load_motion(
    index: int,
    path: Path,
    expected_hash: str,
    target_vy: float,
    expected_fps: int,
    expected_frames: int,
) -> ReferenceMotion:
    if not path.is_file():
        raise FileNotFoundError(f"Reference file is missing: {path}")
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        raise ValueError(f"Reference hash mismatch for {path}: expected {expected_hash}, got {actual_hash}")

    with np.load(path, allow_pickle=False) as archive:
        missing = [key for key in REQUIRED_KEYS if key not in archive]
        if missing:
            raise ValueError(f"{path} is missing required keys: {missing}")
        arrays = {key: np.asarray(archive[key]) for key in REQUIRED_KEYS}

    fps_values = arrays.pop("fps")
    if fps_values.shape != (1,):
        raise ValueError(f"{path}: fps must have shape (1,), got {fps_values.shape}")
    fps = int(fps_values[0])
    if fps != expected_fps:
        raise ValueError(f"{path}: expected {expected_fps} Hz, got {fps} Hz")

    expected_shapes = {
        "joint_pos": (expected_frames, 16),
        "joint_vel": (expected_frames, 16),
        "body_pos_w": (expected_frames, 17, 3),
        "body_quat_w": (expected_frames, 17, 4),
        "body_lin_vel_w": (expected_frames, 17, 3),
        "body_ang_vel_w": (expected_frames, 17, 3),
    }
    for key, shape in expected_shapes.items():
        value = arrays[key]
        if value.shape != shape:
            raise ValueError(f"{path}: {key} expected shape {shape}, got {value.shape}")
        if not np.issubdtype(value.dtype, np.floating):
            raise TypeError(f"{path}: {key} must be floating point, got {value.dtype}")
        if not np.isfinite(value).all():
            bad = np.argwhere(~np.isfinite(value))[0].tolist()
            raise ValueError(f"{path}: {key} contains non-finite data at {bad}")

    quat_norm = np.linalg.norm(arrays["body_quat_w"], axis=-1)
    maximum_quat_error = float(np.max(np.abs(quat_norm - 1.0)))
    if maximum_quat_error > 1.0e-4:
        raise ValueError(f"{path}: quaternion norm maximum error {maximum_quat_error:.3e} exceeds 1e-4")

    float_arrays = {key: np.asarray(value, dtype=np.float32) for key, value in arrays.items()}
    return ReferenceMotion(
        index=index,
        path=path.resolve(),
        sha256=actual_hash,
        target_vy=float(target_vy),
        fps=fps,
        **float_arrays,
    )


def _derive_standing_reference(
    index: int,
    source: ReferenceMotion,
) -> ReferenceMotion:
    """Mirror Isaac's built-in zero-command reference without another asset."""

    frames = source.frames
    arrays = {
        "joint_pos": np.broadcast_to(
            source.joint_pos[0:1],
            (frames, 16),
        ).copy(),
        "joint_vel": np.zeros_like(source.joint_vel),
        "body_pos_w": np.broadcast_to(
            source.body_pos_w[0:1],
            source.body_pos_w.shape,
        ).copy(),
        "body_quat_w": np.broadcast_to(
            source.body_quat_w[0:1],
            source.body_quat_w.shape,
        ).copy(),
        "body_lin_vel_w": np.zeros_like(source.body_lin_vel_w),
        "body_ang_vel_w": np.zeros_like(source.body_ang_vel_w),
    }
    digest = hashlib.sha256()
    digest.update(b"pcbc-derived-standing-first-frame-v1")
    digest.update(source.sha256.encode("ascii"))
    digest.update(str(source.fps).encode("ascii"))
    for name, value in arrays.items():
        digest.update(name.encode("ascii"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(str(value.shape).encode("ascii"))
        digest.update(value.tobytes(order="C"))
    return ReferenceMotion(
        index=index,
        path=source.path,
        sha256=digest.hexdigest(),
        target_vy=0.0,
        fps=source.fps,
        source_kind="derived_standing_first_frame",
        **arrays,
    )


class ReferenceSet:
    def __init__(
        self,
        motions: list[ReferenceMotion],
        body_order: tuple[str, ...],
        contact_inference: dict[str, Any] | None = None,
    ):
        if not motions:
            raise ValueError("ReferenceSet requires at least one motion.")
        ordered = sorted(motions, key=lambda item: item.index)
        if [motion.index for motion in ordered] != list(range(len(ordered))):
            raise ValueError("Reference indices must be contiguous and start at zero.")
        self.motions = tuple(ordered)
        self.body_order = body_order
        self.contact_inference = dict(contact_inference or {})
        self._validate_common_first_frame()

    @classmethod
    def from_config(cls, path: str | Path = "configs/reference_708.yaml") -> "ReferenceSet":
        config_path = Path(path)
        if not config_path.is_absolute():
            config_path = PACKAGE_ROOT / config_path
        config = load_yaml(config_path)
        reference_dir = (PACKAGE_ROOT / config["reference_directory"]).resolve()
        motions = [
            _load_motion(
                index=int(item["index"]),
                path=reference_dir / item["file"],
                expected_hash=str(item["sha256"]),
                target_vy=float(item["target_vy"]),
                expected_fps=int(config["fps"]),
                expected_frames=int(config["frames"]),
            )
            for item in config["references"]
        ]
        standing = config.get("standing_reference", {})
        if bool(standing.get("enabled", False)):
            index = int(standing["index"])
            if index != len(motions):
                raise ValueError(
                    "The standing reference must immediately follow the "
                    f"moving references; expected index {len(motions)}, got {index}."
                )
            if "file" in standing:
                explicit = _load_motion(
                    index=index,
                    path=reference_dir / str(standing["file"]),
                    expected_hash=str(standing["sha256"]),
                    target_vy=0.0,
                    expected_fps=int(config["fps"]),
                    expected_frames=int(config["frames"]),
                )
                motions.append(
                    replace(explicit, source_kind="explicit_standing_npz")
                )
            else:
                source_index = int(standing["source_reference_index"])
                if source_index < 0 or source_index >= len(motions):
                    raise ValueError(
                        "standing_reference.source_reference_index is outside "
                        "the moving-reference range."
                    )
                motions.append(
                    _derive_standing_reference(index, motions[source_index])
                )
        return cls(
            motions,
            tuple(config["body_order"]),
            contact_inference=config.get("contact_inference"),
        )

    def contact_inference_kwargs(self) -> dict[str, Any]:
        supported = {
            "method",
            "wheel_body_indices",
            "per_wheel_height_quantile",
            "height_margin_m",
            "speed_threshold_mps",
            "minimum_contact_run_frames",
            "contact_axis_indices",
            "contact_surface_sides",
            "stride_m",
            "duty_factor",
            "acceleration_seconds",
            "support_preload_seconds",
            "phase_offsets",
            "negative_direction_phase_mirrored",
        }
        return {
            name: value
            for name, value in self.contact_inference.items()
            if name in supported
        }

    def _validate_common_first_frame(self) -> None:
        base = self.motions[0]
        fields = ("joint_pos", "body_pos_w", "body_quat_w")
        for motion in self.motions[1:]:
            for field in fields:
                lhs = getattr(base, field)[0]
                rhs = getattr(motion, field)[0]
                if not np.array_equal(lhs, rhs):
                    delta = float(np.max(np.abs(lhs - rhs)))
                    raise ValueError(
                        f"Reference first-frame mismatch: {base.path.name} vs {motion.path.name}, "
                        f"field={field}, max_abs={delta}"
                    )

    def __len__(self) -> int:
        return len(self.motions)

    def __getitem__(self, index: int) -> ReferenceMotion:
        return self.motions[index]

    @property
    def fixed_motion(self) -> ReferenceMotion:
        return self.motions[0]

    def manifest(self) -> list[dict[str, Any]]:
        return [
            {
                "index": motion.index,
                "path": str(motion.path),
                "file": (
                    motion.path.name
                    if motion.source_kind in {"npz", "explicit_standing_npz"}
                    else f"derived_standing_from_{motion.path.name}"
                ),
                "sha256": motion.sha256,
                "target_vy": motion.target_vy,
                "fps": motion.fps,
                "frames": motion.frames,
                "duration_seconds": motion.duration_seconds,
                "source_kind": motion.source_kind,
            }
            for motion in self.motions
        ]
