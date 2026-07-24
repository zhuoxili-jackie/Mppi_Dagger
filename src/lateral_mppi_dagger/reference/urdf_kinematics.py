from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np


def _vector(text: str | None, default: tuple[float, float, float]) -> np.ndarray:
    if text is None:
        return np.asarray(default, dtype=np.float64)
    value = np.fromstring(text, sep=" ", dtype=np.float64)
    if value.shape != (3,):
        raise ValueError(f"Expected a three-vector, got {text!r}.")
    return value


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.asarray(
        (
            (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
            (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
            (-sp, cp * sr, cp * cr),
        ),
        dtype=np.float64,
    )


def quaternion_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    value = np.asarray(quaternion, dtype=np.float64)
    value = value / np.linalg.norm(value)
    w, x, y, z = value
    return np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def matrix_to_quaternion_wxyz(matrix: np.ndarray) -> np.ndarray:
    rotation = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quaternion = np.asarray(
            (
                0.25 * scale,
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
            )
        )
    else:
        diagonal = np.diag(rotation)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            quaternion = np.asarray(
                (
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                    0.25 * scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                )
            )
        elif index == 1:
            scale = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            quaternion = np.asarray(
                (
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    0.25 * scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                )
            )
        else:
            scale = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            quaternion = np.asarray(
                (
                    (rotation[1, 0] - rotation[0, 1]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    0.25 * scale,
                )
            )
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[0] < 0.0:
        quaternion = -quaternion
    return quaternion


def _axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    unit = axis / np.linalg.norm(axis)
    x, y, z = unit
    cosine = np.cos(angle)
    sine = np.sin(angle)
    complement = 1.0 - cosine
    return np.asarray(
        (
            (
                cosine + x * x * complement,
                x * y * complement - z * sine,
                x * z * complement + y * sine,
            ),
            (
                y * x * complement + z * sine,
                cosine + y * y * complement,
                y * z * complement - x * sine,
            ),
            (
                z * x * complement - y * sine,
                z * y * complement + x * sine,
                cosine + z * z * complement,
            ),
        ),
        dtype=np.float64,
    )


def pose_matrix(position: np.ndarray, quaternion_wxyz: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = quaternion_wxyz_to_matrix(quaternion_wxyz)
    transform[:3, 3] = np.asarray(position, dtype=np.float64)
    return transform


@dataclass(frozen=True)
class JointKinematics:
    name: str
    kind: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray
    lower: float
    upper: float

    def transform(self, position: float) -> np.ndarray:
        result = self.origin.copy()
        if self.kind in {"revolute", "continuous"}:
            result[:3, :3] @= _axis_angle_matrix(self.axis, float(position))
        elif self.kind != "fixed":
            raise NotImplementedError(f"Unsupported URDF joint type {self.kind!r}.")
        return result


class URDFKinematicTree:
    """Small, deterministic URDF FK helper for generating reference assets."""

    def __init__(self, path: str | Path, root_link: str = "Base_link"):
        self.path = Path(path).resolve()
        self.root_link = root_link
        root = ET.parse(self.path).getroot()
        joints: list[JointKinematics] = []
        for element in root.findall("joint"):
            parent_element = element.find("parent")
            child_element = element.find("child")
            if parent_element is None or child_element is None:
                raise ValueError("URDF joint is missing parent or child.")
            origin_element = element.find("origin")
            xyz = _vector(
                None if origin_element is None else origin_element.get("xyz"),
                (0.0, 0.0, 0.0),
            )
            rpy = _vector(
                None if origin_element is None else origin_element.get("rpy"),
                (0.0, 0.0, 0.0),
            )
            origin = np.eye(4, dtype=np.float64)
            origin[:3, :3] = _rpy_matrix(rpy)
            origin[:3, 3] = xyz
            axis_element = element.find("axis")
            axis = _vector(
                None if axis_element is None else axis_element.get("xyz"),
                (1.0, 0.0, 0.0),
            )
            limit_element = element.find("limit")
            lower = -np.inf
            upper = np.inf
            if limit_element is not None:
                if limit_element.get("lower") is not None:
                    lower = float(limit_element.get("lower", "-inf"))
                if limit_element.get("upper") is not None:
                    upper = float(limit_element.get("upper", "inf"))
            joints.append(
                JointKinematics(
                    name=str(element.get("name")),
                    kind=str(element.get("type")),
                    parent=str(parent_element.get("link")),
                    child=str(child_element.get("link")),
                    origin=origin,
                    axis=axis,
                    lower=lower,
                    upper=upper,
                )
            )
        self.joints = tuple(joints)
        self.joint_by_name = {joint.name: joint for joint in joints}
        self.joint_by_child = {joint.child: joint for joint in joints}
        self.children: dict[str, list[JointKinematics]] = {}
        for joint in joints:
            self.children.setdefault(joint.parent, []).append(joint)

    def path_to_link(self, link: str) -> tuple[JointKinematics, ...]:
        reverse: list[JointKinematics] = []
        current = link
        while current != self.root_link:
            if current not in self.joint_by_child:
                raise KeyError(f"{link!r} is not connected to {self.root_link!r}.")
            joint = self.joint_by_child[current]
            reverse.append(joint)
            current = joint.parent
        return tuple(reversed(reverse))

    def link_transform_base(
        self,
        link: str,
        joint_positions: Mapping[str, float],
    ) -> np.ndarray:
        transform = np.eye(4, dtype=np.float64)
        for joint in self.path_to_link(link):
            transform @= joint.transform(float(joint_positions.get(joint.name, 0.0)))
        return transform

    def forward(
        self,
        joint_positions: Mapping[str, float],
        base_position: np.ndarray,
        base_quaternion_wxyz: np.ndarray,
    ) -> dict[str, np.ndarray]:
        transforms = {
            self.root_link: pose_matrix(base_position, base_quaternion_wxyz)
        }
        pending = [self.root_link]
        while pending:
            parent = pending.pop()
            for joint in self.children.get(parent, ()):
                transforms[joint.child] = transforms[parent] @ joint.transform(
                    float(joint_positions.get(joint.name, 0.0))
                )
                pending.append(joint.child)
        return transforms
