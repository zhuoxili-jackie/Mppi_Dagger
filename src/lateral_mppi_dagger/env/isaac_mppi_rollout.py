from __future__ import annotations

import time
from dataclasses import dataclass, fields
from typing import Any

import numpy as np
import torch

from lateral_mppi_dagger.contract.action16 import Action16Adapter, ActionContract, WheelActionMode
from lateral_mppi_dagger.expert.base import (
    MPPI_COST_COMPONENT_NAMES,
    ExpertReply,
    ExpertRequest,
    FailureCode,
)
from lateral_mppi_dagger.expert.mppi_expert import MPPIConfig, ReferenceCenteredMPPI
from lateral_mppi_dagger.reference.loader import ReferenceSet
from lateral_mppi_dagger.env.action_delay import advance_action_delay


def _quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat((q[..., :1], -q[..., 1:]), dim=-1)


def _quat_multiply(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    lw, lx, ly, lz = lhs.unbind(dim=-1)
    rw, rx, ry, rz = rhs.unbind(dim=-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def _quat_rotate(q: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    q_vector = q[..., 1:]
    uv = torch.cross(q_vector, vector, dim=-1)
    uuv = torch.cross(q_vector, uv, dim=-1)
    return vector + 2.0 * (q[..., :1] * uv + uuv)


def _quat_angle(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    dot = torch.sum(lhs * rhs, dim=-1).abs().clamp(max=1.0)
    return 2.0 * torch.acos(dot)


def _tree_env0_clone(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value[0:1].detach().clone()
    if isinstance(value, dict):
        return {key: _tree_env0_clone(item) for key, item in value.items()}
    raise TypeError(f"Unsupported Isaac scene-state value {type(value)!r}")


def _tree_repeat(value: Any, count: int) -> Any:
    if isinstance(value, torch.Tensor):
        repeats = (count,) + (1,) * (value.ndim - 1)
        return value.repeat(repeats)
    if isinstance(value, dict):
        return {key: _tree_repeat(item, count) for key, item in value.items()}
    raise TypeError(f"Unsupported Isaac scene-state value {type(value)!r}")


@dataclass(frozen=True)
class IsaacRolloutCostWeights:
    base_position: float = 35.0
    base_orientation: float = 18.0
    joint_position: float = 10.0
    joint_velocity: float = 0.15
    wheel_position: float = 45.0
    lateral_velocity: float = 4.0
    box_x_drift: float = 60.0
    wheel_slip: float = 0.30
    contact_mismatch: float = 2.0
    front_normal_support: float = 8.0
    rear_force_overload: float = 18.0
    rear_force_imbalance: float = 3.0
    rear_support_loss: float = 30.0
    edge_drop: float = 120.0
    action_residual: float = 0.04
    action_rate: float = 0.025
    joint_acceleration: float = 2.0e-6
    torque_limit: float = 0.20
    joint_limit: float = 150.0
    termination: float = 2500.0
    terminal: float = 8.0

    @classmethod
    def from_dict(cls, values: dict[str, Any] | None) -> "IsaacRolloutCostWeights":
        if not values:
            return cls()
        known = {field.name for field in fields(cls)}
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(f"Unknown MPPI cost weights: {unknown}")
        result = cls(**{name: float(value) for name, value in values.items()})
        if any(getattr(result, field.name) < 0.0 for field in fields(cls)):
            raise ValueError("MPPI cost weights must be non-negative.")
        return result


@dataclass(frozen=True)
class IsaacRolloutLoadLimits:
    front_normal_min_n: float = 6.0
    rear_normal_overload_n: float = 105.0
    rear_normal_scale_n: float = 35.0
    rear_balance_scale_n: float = 70.0

    @classmethod
    def from_dict(
        cls,
        values: dict[str, Any] | None,
    ) -> "IsaacRolloutLoadLimits":
        if not values:
            return cls()
        known = {field.name for field in fields(cls)}
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(f"Unknown MPPI load limits: {unknown}")
        result = cls(**{name: float(value) for name, value in values.items()})
        if result.front_normal_min_n < 0.0:
            raise ValueError("front_normal_min_n must be non-negative.")
        if (
            result.rear_normal_overload_n <= 0.0
            or result.rear_normal_scale_n <= 0.0
            or result.rear_balance_scale_n <= 0.0
        ):
            raise ValueError("Rear-force load limits and scales must be positive.")
        return result


def load_support_cost_terms(
    contact_force_w: torch.Tensor,
    desired_contact: torch.Tensor,
    contact_force_threshold_n: float,
    limits: IsaacRolloutLoadLimits,
) -> dict[str, torch.Tensor]:
    """Return orientation-aware trunk/ground support penalties.

    The robot is nearly vertical in this task.  Front wheels press against the
    trunk mainly along world x, while rear wheels carry gravity mainly along
    world z.  Force-vector magnitude alone cannot distinguish those roles.
    """

    if contact_force_w.ndim != 3 or contact_force_w.shape[1:] != (4, 3):
        raise ValueError("contact_force_w must have shape [batch,4,3].")
    if desired_contact.shape != contact_force_w.shape[:2]:
        raise ValueError("desired_contact must have shape [batch,4].")
    if contact_force_threshold_n <= 0.0:
        raise ValueError("contact_force_threshold_n must be positive.")

    front_normal = torch.abs(contact_force_w[:, :2, 0])
    rear_normal = torch.abs(contact_force_w[:, 2:, 2])
    desired_front = desired_contact[:, :2].float()
    front_denominator = max(limits.front_normal_min_n, 1.0)
    front_deficit = torch.relu(
        limits.front_normal_min_n - front_normal
    ) / front_denominator
    front_normal_support = (
        front_deficit.square() * desired_front
    ).sum(dim=-1) / desired_front.sum(dim=-1).clamp_min(1.0)

    rear_overload = torch.relu(
        rear_normal - limits.rear_normal_overload_n
    ) / limits.rear_normal_scale_n
    rear_force_overload = rear_overload.square().mean(dim=-1)

    rear_measured = rear_normal >= contact_force_threshold_n
    both_rear = torch.all(rear_measured, dim=-1)
    rear_force_imbalance = (
        (rear_normal[:, 0] - rear_normal[:, 1])
        / limits.rear_balance_scale_n
    ).square() * both_rear.float()
    rear_contact_count = rear_measured.float().sum(dim=-1)
    rear_support_loss = ((2.0 - rear_contact_count) / 2.0).square()
    return {
        "front_normal_support": front_normal_support,
        "rear_force_overload": rear_force_overload,
        "rear_force_imbalance": rear_force_imbalance,
        "rear_support_loss": rear_support_loss,
    }


@dataclass
class IsaacRolloutSnapshot:
    scene_state_relative: dict[str, Any]
    action_manager: dict[str, torch.Tensor]
    action_terms: dict[str, dict[str, torch.Tensor]]
    command_buffers: dict[str, torch.Tensor]
    sensor_buffers: dict[str, torch.Tensor]
    sensor_clock: dict[str, torch.Tensor]
    previous_executed_action: torch.Tensor
    previous_commanded_action: torch.Tensor
    action_delay_queue: torch.Tensor
    sim_step_counter: int
    ref_id: int
    ref_frame: int
    seed_anchor_pos_local: torch.Tensor
    seed_anchor_quat_w: torch.Tensor


class IsaacMPPIRolloutCloner:
    """Parallel, restore-before-each-batch Isaac rollout engine.

    The public environment still advances only once after the expert returns.
    Candidate simulation bypasses reward/termination/command managers and
    advances physics plus action terms directly.  All mutable buffers touched
    by this path are restored from env zero before every candidate batch and
    again before returning the selected action.
    """

    _COMMAND_BUFFER_NAMES = (
        "motion_ids",
        "_target_lateral_velocities",
        "time_steps",
        "stage",
        "stable_counter",
        "post_stable_counter",
        "hold_elapsed_counter",
        "ready_anchor_pos_w",
        "ready_anchor_quat_w",
        "body_pos_relative_w",
        "body_quat_relative_w",
    )
    _ACTION_TERM_BUFFER_NAMES = (
        "_raw_actions",
        "_processed_actions",
        "_prev_applied_actions",
    )
    _SENSOR_BUFFER_NAMES = (
        "net_forces_w",
        "net_forces_w_history",
        "force_matrix_w",
        "force_matrix_w_history",
        "last_air_time",
        "current_air_time",
        "last_contact_time",
        "current_contact_time",
        "friction_forces_w",
    )

    def __init__(
        self,
        adapter: Any,
        references: ReferenceSet,
        action_contract: ActionContract,
        horizon: int,
        cost_weights: IsaacRolloutCostWeights,
        contact_force_threshold: float = 8.0,
        load_limits: IsaacRolloutLoadLimits | None = None,
    ):
        self.adapter = adapter
        self.base = adapter.base
        self.robot = adapter.robot
        self.command = adapter.command
        self.references = references
        self.action_contract = action_contract
        self.horizon = int(horizon)
        self.cost_weights = cost_weights
        self.contact_force_threshold = float(contact_force_threshold)
        self.load_limits = load_limits or IsaacRolloutLoadLimits()
        self.device = torch.device(self.base.device)
        self.num_envs = int(self.base.num_envs)
        self.env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        if self.horizon < 1:
            raise ValueError("MPPI rollout horizon must be positive.")
        if action_contract.wheel_action_mode is not WheelActionMode.HARD_ZERO:
            raise NotImplementedError(
                "The current Isaac MPPI rollout provider is intentionally restricted to hard_zero."
            )

        self.anchor_body_id = int(self.command.robot_anchor_body_index)
        self.ref_anchor_body_id = int(self.command.motion_anchor_body_index)
        self.wheel_body_ids = list(adapter.wheel_body_ids)
        self.ref_wheel_body_ids = [
            references.body_order.index(name) for name in ("FL_foot_link", "FR_foot_link", "RL_foot_link", "RR_foot_link")
        ]
        self.joint_ids = list(adapter.joint_ids)
        self.contact_body_ids = list(adapter.contact_body_ids)

        self._joint_pos_refs = torch.as_tensor(
            np.stack([motion.joint_pos for motion in references.motions]),
            dtype=torch.float32,
            device=self.device,
        )
        self._joint_vel_refs = torch.as_tensor(
            np.stack([motion.joint_vel for motion in references.motions]),
            dtype=torch.float32,
            device=self.device,
        )
        self._body_pos_refs = torch.as_tensor(
            np.stack([motion.body_pos_w for motion in references.motions]),
            dtype=torch.float32,
            device=self.device,
        )
        self._body_quat_refs = torch.as_tensor(
            np.stack([motion.body_quat_w for motion in references.motions]),
            dtype=torch.float32,
            device=self.device,
        )
        self._body_lin_vel_refs = torch.as_tensor(
            np.stack([motion.body_lin_vel_w for motion in references.motions]),
            dtype=torch.float32,
            device=self.device,
        )
        self._desired_contacts = torch.as_tensor(
            np.stack(adapter.contact_schedules),
            dtype=torch.bool,
            device=self.device,
        )
        self.last_components: dict[str, torch.Tensor] = {}
        self.last_best_components: dict[str, float] = {}
        self.last_termination_rate = 0.0
        self._episode_ref_id: int | None = None
        self._episode_ref_frame: int | None = None
        self._episode_anchor_pos_local: torch.Tensor | None = None
        self._episode_anchor_quat_w: torch.Tensor | None = None

    def reset_episode_alignment(self) -> None:
        """Freeze the episode-global reference/world alignment.

        Re-anchoring every receding-horizon solve to the current robot pose
        would erase accumulated x/y drift from the cost.  The alignment is
        therefore captured exactly once, after the real environment reset.
        """
        self._episode_ref_id = int(self.command.motion_ids[0].item())
        self._episode_ref_frame = int(self.command.time_steps[0].item())
        self._episode_anchor_pos_local = (
            self.command.robot_anchor_pos_w[0] - self.base.scene.env_origins[0]
        ).detach().clone()
        self._episode_anchor_quat_w = self.command.robot_anchor_quat_w[0].detach().clone()

    @staticmethod
    def _copy_env0_tensor(value: torch.Tensor, num_envs: int) -> torch.Tensor:
        if value.ndim == 0 or value.shape[0] != num_envs:
            raise ValueError("Expected a tensor whose leading dimension is num_envs.")
        return value[0:1].detach().clone()

    def capture(self) -> IsaacRolloutSnapshot:
        scene_state = _tree_env0_clone(self.base.scene.get_state(is_relative=True))
        manager = self.base.action_manager
        action_manager = {
            "_action": self._copy_env0_tensor(manager._action, self.num_envs),
            "_prev_action": self._copy_env0_tensor(manager._prev_action, self.num_envs),
        }
        action_terms: dict[str, dict[str, torch.Tensor]] = {}
        for name in manager.active_terms:
            term = manager.get_term(name)
            buffers = {}
            for buffer_name in self._ACTION_TERM_BUFFER_NAMES:
                value = getattr(term, buffer_name, None)
                if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == self.num_envs:
                    buffers[buffer_name] = value[0:1].detach().clone()
            action_terms[name] = buffers

        command_buffers = {}
        for name in self._COMMAND_BUFFER_NAMES:
            value = getattr(self.command, name, None)
            if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == self.num_envs:
                command_buffers[name] = value[0:1].detach().clone()

        # Access once to make lazy contact buffers current before taking a copy.
        sensor = self.adapter.contact_sensor
        sensor_data = sensor.data
        sensor_buffers = {}
        for name in self._SENSOR_BUFFER_NAMES:
            value = getattr(sensor_data, name, None)
            if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == self.num_envs:
                sensor_buffers[name] = value[0:1].detach().clone()
        sensor_clock = {}
        for name in ("_timestamp", "_timestamp_last_update", "_is_outdated"):
            value = getattr(sensor, name, None)
            if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == self.num_envs:
                sensor_clock[name] = value[0:1].detach().clone()

        anchor_pos_local = (
            self.command.robot_anchor_pos_w[0] - self.base.scene.env_origins[0]
        ).detach().clone()
        anchor_quat = self.command.robot_anchor_quat_w[0].detach().clone()
        return IsaacRolloutSnapshot(
            scene_state_relative=scene_state,
            action_manager=action_manager,
            action_terms=action_terms,
            command_buffers=command_buffers,
            sensor_buffers=sensor_buffers,
            sensor_clock=sensor_clock,
            previous_executed_action=self.adapter.previous_executed_action[0:1].detach().clone(),
            previous_commanded_action=self.adapter.previous_commanded_action[0:1].detach().clone(),
            action_delay_queue=self.adapter.action_delay_queue[:, 0:1].detach().clone(),
            sim_step_counter=int(self.base._sim_step_counter),
            ref_id=int(self.command.motion_ids[0].item()),
            ref_frame=int(self.command.time_steps[0].item()),
            seed_anchor_pos_local=anchor_pos_local,
            seed_anchor_quat_w=anchor_quat,
        )

    def _clear_contact_warm_start(self, snapshot: IsaacRolloutSnapshot) -> None:
        """Destroy candidate-specific PhysX contact pairs before a restore.

        Tensor state writes restore poses and velocities but do not restore the
        opaque PhysX contact solver cache.  Consequently, a passive wheel can
        receive a different first-step impulse depending on which MPPI
        candidate previously occupied the same clone.  Move each dynamic asset
        to a distinct collision-free height for one physics substep, then let
        ``restore`` write the exact snapshot back.  The next candidate and the
        eventual real environment step both start with freshly-created contact
        pairs instead of candidate-history-dependent warm starts.
        """
        separated_state = _tree_repeat(snapshot.scene_state_relative, self.num_envs)
        separation_index = 1
        for category in ("articulation", "rigid_object"):
            for asset_state in separated_state.get(category, {}).values():
                root_pose = asset_state.get("root_pose")
                if isinstance(root_pose, torch.Tensor):
                    root_pose[:, 2] += 10.0 * separation_index
                    separation_index += 1
                root_velocity = asset_state.get("root_velocity")
                if isinstance(root_velocity, torch.Tensor):
                    root_velocity.zero_()
        self.base.scene.reset_to(
            separated_state,
            env_ids=self.env_ids,
            is_relative=True,
        )
        self.base.sim.step(render=False)
        self.base.scene.update(dt=self.base.physics_dt)

    def restore(self, snapshot: IsaacRolloutSnapshot) -> None:
        self._clear_contact_warm_start(snapshot)
        repeated_state = _tree_repeat(snapshot.scene_state_relative, self.num_envs)
        self.base.scene.reset_to(repeated_state, env_ids=self.env_ids, is_relative=True)
        self.base.sim.forward()
        # Refresh lazy articulation/body buffers after the direct tensor write.
        self.base.scene.update(dt=0.0)

        manager = self.base.action_manager
        for name, value in snapshot.action_manager.items():
            getattr(manager, name).copy_(_tree_repeat(value, self.num_envs))
        for term_name, buffers in snapshot.action_terms.items():
            term = manager.get_term(term_name)
            for name, value in buffers.items():
                getattr(term, name).copy_(_tree_repeat(value, self.num_envs))
        for name, value in snapshot.command_buffers.items():
            getattr(self.command, name).copy_(_tree_repeat(value, self.num_envs))

        sensor = self.adapter.contact_sensor
        for name, value in snapshot.sensor_buffers.items():
            target = getattr(sensor._data, name, None)
            if isinstance(target, torch.Tensor):
                target.copy_(_tree_repeat(value, self.num_envs))
        for name, value in snapshot.sensor_clock.items():
            getattr(sensor, name).copy_(_tree_repeat(value, self.num_envs))

        self.adapter.previous_executed_action.copy_(
            _tree_repeat(snapshot.previous_executed_action, self.num_envs)
        )
        self.adapter.previous_commanded_action.copy_(
            _tree_repeat(snapshot.previous_commanded_action, self.num_envs)
        )
        delay_queue = snapshot.action_delay_queue.repeat(1, self.num_envs, 1)
        self.adapter.action_delay_queue.copy_(delay_queue)
        self.base._sim_step_counter = snapshot.sim_step_counter

    def _physics_step(self, action16: torch.Tensor) -> None:
        if action16.shape != (self.num_envs, 16):
            raise ValueError(
                f"Isaac rollout action must have shape {(self.num_envs, 16)}, got {tuple(action16.shape)}"
            )
        if not torch.equal(action16[:, 12:], torch.zeros_like(action16[:, 12:])):
            raise ValueError("Isaac hard-zero MPPI received non-zero wheel action.")
        self.base.action_manager.process_action(action16)
        for _ in range(self.base.cfg.decimation):
            self.base._sim_step_counter += 1
            self.base.action_manager.apply_action()
            self.base.scene.write_data_to_sim()
            self.base.sim.step(render=False)
            self.base.scene.update(dt=self.base.physics_dt)

    def state_vector(self) -> torch.Tensor:
        root_pose = self.robot.data.root_pose_w.clone()
        root_pose[:, :3] -= self.base.scene.env_origins
        return torch.cat(
            (
                root_pose,
                self.robot.data.root_vel_w,
                self.robot.data.joint_pos[:, self.joint_ids],
                self.robot.data.joint_vel[:, self.joint_ids],
            ),
            dim=-1,
        )

    def _aligned_reference(
        self,
        snapshot: IsaacRolloutSnapshot,
        frame: int,
    ) -> dict[str, torch.Tensor]:
        ref_id = snapshot.ref_id
        if (
            self._episode_ref_id == ref_id
            and self._episode_ref_frame is not None
            and self._episode_anchor_pos_local is not None
            and self._episode_anchor_quat_w is not None
        ):
            alignment_frame = self._episode_ref_frame
            alignment_anchor_pos_local = self._episode_anchor_pos_local
            alignment_anchor_quat_w = self._episode_anchor_quat_w
        else:
            # Determinism/state-copy probes may call the cloner before an
            # expert episode has been reset.  Use a solve-local alignment only
            # for that diagnostic path.
            alignment_frame = snapshot.ref_frame
            alignment_anchor_pos_local = snapshot.seed_anchor_pos_local
            alignment_anchor_quat_w = snapshot.seed_anchor_quat_w
        current_frame = min(alignment_frame, self._joint_pos_refs.shape[1] - 1)
        frame = min(frame, self._joint_pos_refs.shape[1] - 1)
        ref_anchor_pos_now = self._body_pos_refs[ref_id, current_frame, self.ref_anchor_body_id]
        ref_anchor_quat_now = self._body_quat_refs[ref_id, current_frame, self.ref_anchor_body_id]
        alignment_quat = _quat_multiply(
            alignment_anchor_quat_w,
            _quat_conjugate(ref_anchor_quat_now),
        )

        ref_body_pos = self._body_pos_refs[ref_id, frame]
        target_body_pos_local = alignment_anchor_pos_local + _quat_rotate(
            alignment_quat.unsqueeze(0).expand(ref_body_pos.shape[0], -1),
            ref_body_pos - ref_anchor_pos_now,
        )
        ref_body_quat = self._body_quat_refs[ref_id, frame]
        target_body_quat = _quat_multiply(
            alignment_quat.unsqueeze(0).expand(ref_body_quat.shape[0], -1),
            ref_body_quat,
        )
        ref_body_lin_vel = self._body_lin_vel_refs[ref_id, frame]
        target_body_lin_vel = _quat_rotate(
            alignment_quat.unsqueeze(0).expand(ref_body_lin_vel.shape[0], -1),
            ref_body_lin_vel,
        )
        return {
            "joint_pos": self._joint_pos_refs[ref_id, frame],
            "joint_vel": self._joint_vel_refs[ref_id, frame],
            "body_pos_local": target_body_pos_local,
            "body_quat": target_body_quat,
            "body_lin_vel": target_body_lin_vel,
            "desired_contact": self._desired_contacts[ref_id, frame],
        }

    def evaluate(
        self,
        candidates_leg: torch.Tensor,
        snapshot: IsaacRolloutSnapshot,
        nominal_leg: torch.Tensor,
    ) -> torch.Tensor:
        if candidates_leg.shape != (self.num_envs, self.horizon, 12):
            raise ValueError(
                "Candidate batch must match rollout clones and horizon: "
                f"expected {(self.num_envs, self.horizon, 12)}, got {tuple(candidates_leg.shape)}"
            )
        if nominal_leg.shape != (self.horizon, 12):
            raise ValueError(f"nominal_leg must have shape {(self.horizon, 12)}")
        if not torch.isfinite(candidates_leg).all():
            raise ValueError("Candidate action batch contains NaN or Inf.")

        self.restore(snapshot)
        weights = self.cost_weights
        totals = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        alive = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        previous_action = snapshot.previous_commanded_action[:, :12].expand(
            self.num_envs,
            -1,
        )
        delay_queue = snapshot.action_delay_queue.repeat(1, self.num_envs, 1)
        previous_dq = self.robot.data.joint_vel[:, self.joint_ids].clone()
        component_totals: dict[str, torch.Tensor] = {}

        def add(name: str, values: torch.Tensor, weight: float, active: torch.Tensor) -> None:
            weighted = weight * values * active
            totals.add_(weighted)
            if name not in component_totals:
                component_totals[name] = torch.zeros_like(totals)
            component_totals[name].add_(weighted)

        for horizon_step in range(self.horizon):
            leg_action = candidates_leg[:, horizon_step]
            commanded_action16 = torch.cat(
                (leg_action, torch.zeros(self.num_envs, 4, device=self.device)),
                dim=-1,
            )
            applied_action16, delay_queue = advance_action_delay(
                commanded_action16,
                delay_queue,
            )
            self._physics_step(applied_action16)
            frame = snapshot.ref_frame + horizon_step + 1
            target = self._aligned_reference(snapshot, frame)
            active = alive.float()

            actual_anchor_pos = (
                self.command.robot_anchor_pos_w - self.base.scene.env_origins
            )
            actual_anchor_quat = self.command.robot_anchor_quat_w
            actual_anchor_lin_vel = self.command.robot_anchor_lin_vel_w
            target_anchor_pos = target["body_pos_local"][self.ref_anchor_body_id]
            target_anchor_quat = target["body_quat"][self.ref_anchor_body_id]
            target_anchor_lin_vel = target["body_lin_vel"][self.ref_anchor_body_id]

            base_pos_error = actual_anchor_pos - target_anchor_pos
            base_pos_cost = (
                base_pos_error[:, 0].square()
                + base_pos_error[:, 1].square()
                + 2.0 * base_pos_error[:, 2].square()
            )
            orientation_error = _quat_angle(actual_anchor_quat, target_anchor_quat)
            add("base_position", base_pos_cost, weights.base_position, active)
            add("base_orientation", orientation_error.square(), weights.base_orientation, active)

            q = self.robot.data.joint_pos[:, self.joint_ids]
            dq = self.robot.data.joint_vel[:, self.joint_ids]
            q_error = q[:, :12] - target["joint_pos"][:12]
            dq_error = dq[:, :12] - target["joint_vel"][:12]
            add("joint_position", q_error.square().mean(dim=-1), weights.joint_position, active)
            add("joint_velocity", dq_error.square().mean(dim=-1), weights.joint_velocity, active)

            wheel_pos_local = (
                self.robot.data.body_pos_w[:, self.wheel_body_ids]
                - self.base.scene.env_origins.unsqueeze(1)
            )
            wheel_lin_vel = self.robot.data.body_lin_vel_w[:, self.wheel_body_ids]
            target_wheel_pos = target["body_pos_local"][self.ref_wheel_body_ids]
            target_wheel_lin_vel = target["body_lin_vel"][self.ref_wheel_body_ids]
            wheel_error = wheel_pos_local - target_wheel_pos.unsqueeze(0)
            add("wheel_position", wheel_error.square().mean(dim=(1, 2)), weights.wheel_position, active)
            add(
                "lateral_velocity",
                (actual_anchor_lin_vel[:, 1] - target_anchor_lin_vel[1]).square(),
                weights.lateral_velocity,
                active,
            )
            add("box_x_drift", base_pos_error[:, 0].square(), weights.box_x_drift, active)

            contact_force = self.adapter.contact_sensor.data.net_forces_w[:, self.contact_body_ids]
            measured_contact = torch.linalg.vector_norm(contact_force, dim=-1) >= self.contact_force_threshold
            desired_contact = target["desired_contact"].unsqueeze(0).expand_as(measured_contact)
            contact_mismatch = torch.logical_xor(measured_contact, desired_contact).float().mean(dim=-1)
            support_costs = load_support_cost_terms(
                contact_force,
                desired_contact,
                self.contact_force_threshold,
                self.load_limits,
            )
            slip_error = (wheel_lin_vel[..., :2] - target_wheel_lin_vel[None, :, :2]).square().sum(dim=-1)
            slip_cost = (slip_error * measured_contact.float()).mean(dim=-1)
            add("wheel_slip", slip_cost, weights.wheel_slip, active)
            add("contact_mismatch", contact_mismatch, weights.contact_mismatch, active)
            add(
                "front_normal_support",
                support_costs["front_normal_support"],
                weights.front_normal_support,
                active,
            )
            add(
                "rear_force_overload",
                support_costs["rear_force_overload"],
                weights.rear_force_overload,
                active,
            )
            add(
                "rear_force_imbalance",
                support_costs["rear_force_imbalance"],
                weights.rear_force_imbalance,
                active,
            )
            add(
                "rear_support_loss",
                support_costs["rear_support_loss"],
                weights.rear_support_loss,
                active,
            )

            edge_excess = torch.relu(torch.abs(wheel_error[..., 0]) - 0.08)
            drop_excess = torch.relu(target_wheel_pos[None, :, 2] - wheel_pos_local[..., 2] - 0.06)
            edge_drop_cost = edge_excess.square().mean(dim=-1) + drop_excess.square().mean(dim=-1)
            add("edge_drop", edge_drop_cost, weights.edge_drop, active)

            residual = leg_action - nominal_leg[horizon_step]
            action_rate = leg_action - previous_action
            joint_acceleration = (dq - previous_dq) / float(self.base.step_dt)
            add("action_residual", residual.square().mean(dim=-1), weights.action_residual, active)
            add("action_rate", action_rate.square().mean(dim=-1), weights.action_rate, active)
            add(
                "joint_acceleration",
                joint_acceleration[:, :12].square().mean(dim=-1),
                weights.joint_acceleration,
                active,
            )
            previous_action = leg_action
            previous_dq = dq.clone()

            joint_limits = self.robot.data.soft_joint_pos_limits[:, self.joint_ids[:12]]
            below = torch.relu(joint_limits[..., 0] - q[:, :12])
            above = torch.relu(q[:, :12] - joint_limits[..., 1])
            add("joint_limit", (below.square() + above.square()).mean(dim=-1), weights.joint_limit, active)

            applied_torque = self.robot.data.applied_torque[:, self.joint_ids[:12]]
            effort_limits = self.robot.data.joint_effort_limits[:, self.joint_ids[:12]].clamp_min(1.0e-6)
            torque_excess = torch.relu(torch.abs(applied_torque) / effort_limits - 1.0)
            add("torque_limit", torque_excess.square().mean(dim=-1), weights.torque_limit, active)

            # Match the actual lateral task termination functions.  The task
            # deliberately gates position termination on z only and measures
            # orientation through projected gravity, so lateral progress/yaw
            # error must remain a tracking cost rather than falsely killing a
            # rollout candidate.
            gravity_w = torch.zeros(
                (self.num_envs, 3),
                dtype=torch.float32,
                device=self.device,
            )
            gravity_w[:, 2] = -1.0
            target_gravity_b = _quat_rotate(
                _quat_conjugate(
                    target_anchor_quat.unsqueeze(0).expand(self.num_envs, -1)
                ),
                gravity_w,
            )
            actual_gravity_b = _quat_rotate(
                _quat_conjugate(actual_anchor_quat),
                gravity_w,
            )
            projected_gravity_z_error = torch.abs(
                target_gravity_b[:, 2] - actual_gravity_b[:, 2]
            )
            terminated = (
                (torch.abs(base_pos_error[:, 2]) > 0.25)
                | (projected_gravity_z_error > 0.80)
                | ~torch.isfinite(self.state_vector()).all(dim=-1)
            )
            newly_terminated = alive & terminated
            add(
                "termination",
                newly_terminated.float(),
                weights.termination,
                torch.ones_like(active),
            )
            alive &= ~terminated

        final_tracking = component_totals["base_position"] / max(weights.base_position, 1.0e-12)
        final_tracking += component_totals["joint_position"] / max(weights.joint_position, 1.0e-12)
        add(
            "terminal",
            final_tracking / float(self.horizon),
            weights.terminal,
            alive.float(),
        )
        self.last_components = {name: values.detach().clone() for name, values in component_totals.items()}
        best_index = int(torch.argmin(totals).item())
        self.last_best_components = {
            name: float(values[best_index].item()) for name, values in component_totals.items()
        }
        self.last_termination_rate = float((~alive).float().mean().item())
        return totals

    def probe_determinism(
        self,
        candidates_leg: torch.Tensor,
        nominal_leg: torch.Tensor,
    ) -> dict[str, Any]:
        snapshot = self.capture()
        try:
            first_cost = self.evaluate(candidates_leg, snapshot, nominal_leg).clone()
            first_state = self.state_vector().clone()
            second_cost = self.evaluate(candidates_leg, snapshot, nominal_leg).clone()
            second_state = self.state_vector().clone()
            cost_abs = torch.abs(first_cost - second_cost)
            state_abs = torch.abs(first_state - second_state)
            cost_max_abs = float(torch.max(cost_abs).item())
            state_max_abs = float(torch.max(state_abs).item())
            cost_max_index = int(torch.argmax(cost_abs).item())
            flat_state_max_index = int(torch.argmax(state_abs).item())
            state_max_env = flat_state_max_index // state_abs.shape[1]
            state_max_dimension = flat_state_max_index % state_abs.shape[1]
            state_slices = {
                "root_pose": (0, 7),
                "root_velocity": (7, 13),
                "joint_position": (13, 29),
                "joint_velocity": (29, 45),
            }
            state_group_max_abs = {
                name: float(state_abs[:, start:end].max().item())
                for name, (start, end) in state_slices.items()
            }
            self.restore(snapshot)
        except BaseException:
            self.restore(snapshot)
            raise
        return {
            "cost_max_abs": cost_max_abs,
            "cost_max_candidate": cost_max_index,
            "state_max_abs": state_max_abs,
            "state_max_candidate": state_max_env,
            "state_max_dimension": state_max_dimension,
            "state_group_max_abs": state_group_max_abs,
            "pass": bool(cost_max_abs <= 1.0e-5 and state_max_abs <= 1.0e-5),
        }


class IsaacWholeBodyMPPIProvider:
    """Reference-centred hard-zero MPPI label provider backed by Isaac clones."""

    def __init__(
        self,
        adapter: Any,
        references: ReferenceSet,
        action_adapter: Action16Adapter,
        config: MPPIConfig,
        noise_std: torch.Tensor,
        cost_weights: IsaacRolloutCostWeights,
        load_limits: IsaacRolloutLoadLimits | None = None,
        contact_force_threshold_n: float = 8.0,
        physical_target_rate_limit_rad_s: float | None = None,
    ):
        if adapter.num_envs != config.samples:
            raise ValueError(
                f"MPPI requires one Isaac clone per sample: envs={adapter.num_envs}, samples={config.samples}"
            )
        self.adapter = adapter
        self.references = references
        self.action_adapter = action_adapter
        self.config = config
        self.optimizer = ReferenceCenteredMPPI(config, noise_std, device=str(adapter.base.device))
        self.rollout = IsaacMPPIRolloutCloner(
            adapter,
            references,
            action_adapter.contract,
            config.horizon,
            cost_weights,
            contact_force_threshold=contact_force_threshold_n,
            load_limits=load_limits,
        )
        contract = action_adapter.contract
        self.raw_min = torch.as_tensor(contract.raw_min[:12], device=adapter.base.device)
        self.raw_max = torch.as_tensor(contract.raw_max[:12], device=adapter.base.device)
        self.offset = torch.as_tensor(
            contract.q_action_offset_runtime[:12], device=adapter.base.device
        )
        self.scale = torch.as_tensor(contract.scale[:12], device=adapter.base.device)
        if physical_target_rate_limit_rad_s is None:
            self.max_delta = torch.as_tensor(
                contract.max_raw_delta_per_step[:12],
                device=adapter.base.device,
            )
        else:
            if physical_target_rate_limit_rad_s <= 0.0:
                raise ValueError(
                    "physical_target_rate_limit_rad_s must be positive."
                )
            maximum_physical_delta = (
                float(physical_target_rate_limit_rad_s)
                * float(adapter.base.step_dt)
            )
            self.max_delta = maximum_physical_delta / self.scale
        self._warm_residual: torch.Tensor | None = None
        self.last_diagnostics: dict[str, Any] = {}

    def reset(self, episode_metadata: dict[str, Any] | None = None) -> None:
        self._warm_residual = None
        self.last_diagnostics = {}
        episode_seed = int((episode_metadata or {}).get("seed", 0))
        self.optimizer.reset_seed(self.config.seed + episode_seed)
        self.rollout.reset_episode_alignment()

    def _nominal(self, request: ExpertRequest) -> torch.Tensor:
        reference = self.references[request.ref_id]
        frames = np.minimum(
            request.ref_frame
            + self.config.reference_action_lookahead_steps
            + self.adapter.action_delay_steps
            + np.arange(self.config.horizon),
            reference.frames - 1,
        )
        q_ref = torch.as_tensor(
            reference.joint_pos[frames, :12],
            dtype=torch.float32,
            device=self.adapter.base.device,
        )
        return (q_ref - self.offset) / self.scale

    def __call__(self, request: ExpertRequest) -> ExpertReply:
        start = time.perf_counter()
        snapshot: IsaacRolloutSnapshot | None = None
        try:
            request.validate()
            nominal = self._nominal(request)
            previous_action = self.adapter.previous_commanded_action[0, :12].clone()
            initial = nominal
            if self.config.warm_start and self._warm_residual is not None:
                shifted = torch.cat(
                    (self._warm_residual[1:], self._warm_residual[-1:]),
                    dim=0,
                )
                initial = nominal + shifted

            snapshot = self.rollout.capture()
            sequence, optimizer_diagnostics = self.optimizer.optimize(
                nominal,
                lambda candidates: self.rollout.evaluate(candidates, snapshot, nominal),
                self.raw_min,
                self.raw_max,
                previous_action=previous_action,
                max_delta=self.max_delta,
                initial_sequence=initial,
            )
            self.rollout.restore(snapshot)
            self._warm_residual = (sequence - nominal).detach().clone()
            selected_leg = sequence[0]
            action16_t = torch.cat(
                (selected_leg, torch.zeros(4, device=selected_leg.device)),
                dim=0,
            )
            if not torch.isfinite(action16_t).all():
                raise FloatingPointError("MPPI selected a non-finite action.")
            action16 = action16_t.detach().cpu().numpy().astype(np.float32)
            q_des_leg = (
                self.offset + self.scale * selected_leg
            ).detach().cpu().numpy().astype(np.float32)
            lower_margin = selected_leg - self.raw_min
            upper_margin = self.raw_max - selected_leg
            safety_margin = float(torch.minimum(lower_margin, upper_margin).min().item())
            solve_ms = (time.perf_counter() - start) * 1000.0
            self.last_diagnostics = {
                **optimizer_diagnostics,
                "solve_ms": solve_ms,
                "best_cost_components": dict(self.rollout.last_best_components),
                "rollout_termination_rate": self.rollout.last_termination_rate,
                "ref_id": request.ref_id,
                "ref_frame": request.ref_frame,
            }
            status = "MPPI_VALID"
            reply_diagnostics = {
                "minimum_total_cost": optimizer_diagnostics["minimum_total_cost"],
                "mean_total_cost": optimizer_diagnostics["mean_total_cost"],
                "effective_sample_size": optimizer_diagnostics["effective_sample_size"],
                "rollout_termination_rate": self.rollout.last_termination_rate,
                "cost_components": [
                    float(self.rollout.last_best_components.get(name, np.nan))
                    for name in MPPI_COST_COMPONENT_NAMES
                ],
            }
            return ExpertReply(
                valid=True,
                q_des_leg=q_des_leg,
                wheel_vel_des=np.zeros(4, dtype=np.float32),
                action16=action16,
                tau_ff_leg=None,
                predicted_grf=None,
                solve_ms=solve_ms,
                solver_status=status,
                safety_margin=safety_margin,
                source="mppi",
                failure_code=FailureCode.NONE,
                diagnostics=reply_diagnostics,
            )
        except BaseException as exc:
            if snapshot is not None:
                try:
                    self.rollout.restore(snapshot)
                except BaseException as restore_exc:
                    self.last_diagnostics = {
                        "error": repr(exc),
                        "restore_error": repr(restore_exc),
                    }
                    failure_code = FailureCode.STATE_COPY_ERROR
                else:
                    self.last_diagnostics = {"error": repr(exc)}
                    failure_code = FailureCode.TEACHER_INFEASIBLE
            else:
                self.last_diagnostics = {"error": repr(exc)}
                failure_code = FailureCode.TEACHER_INFEASIBLE
            return ExpertReply(
                valid=False,
                q_des_leg=np.full(12, np.nan, dtype=np.float32),
                wheel_vel_des=np.zeros(4, dtype=np.float32),
                action16=np.full(16, np.nan, dtype=np.float32),
                tau_ff_leg=None,
                predicted_grf=None,
                solve_ms=(time.perf_counter() - start) * 1000.0,
                solver_status=f"MPPI_ERROR:{type(exc).__name__}:{exc}",
                safety_margin=float("nan"),
                source="mppi",
                failure_code=failure_code,
                diagnostics={"error": repr(exc)},
            )
