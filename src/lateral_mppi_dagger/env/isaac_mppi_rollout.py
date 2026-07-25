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
from lateral_mppi_dagger.reference.action_reference import (
    normalize_nominal_solver_overrides,
    resolve_nominal_solver_overrides,
)
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


def _signed_pitch_error_rad(
    actual_quat_w: torch.Tensor,
    target_quat_w: torch.Tensor,
) -> torch.Tensor:
    """Return the target-frame pitch component of the shortest rotation.

    Both inputs use the frozen Isaac/deployment ``wxyz`` convention.  The
    relative rotation maps the target orientation to the actual orientation,
    so a negative value means that the robot is pitched negatively relative
    to the aligned reference.
    """

    actual = actual_quat_w / torch.linalg.vector_norm(
        actual_quat_w,
        dim=-1,
        keepdim=True,
    ).clamp_min(torch.finfo(actual_quat_w.dtype).eps)
    target = target_quat_w / torch.linalg.vector_norm(
        target_quat_w,
        dim=-1,
        keepdim=True,
    ).clamp_min(torch.finfo(target_quat_w.dtype).eps)
    relative = _quat_multiply(_quat_conjugate(target), actual)
    relative = torch.where(
        relative[..., :1] < 0.0,
        -relative,
        relative,
    )
    vector = relative[..., 1:]
    vector_norm = torch.linalg.vector_norm(
        vector,
        dim=-1,
        keepdim=True,
    )
    angle = 2.0 * torch.atan2(
        vector_norm,
        relative[..., :1].clamp_min(
            torch.finfo(relative.dtype).eps
        ),
    )
    rotation_vector = torch.where(
        vector_norm > torch.finfo(relative.dtype).eps,
        vector * (angle / vector_norm.clamp_min(
            torch.finfo(relative.dtype).eps
        )),
        2.0 * vector,
    )
    return rotation_vector[..., 1]


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
    base_height_drop: float = 0.0
    base_orientation: float = 18.0
    joint_position: float = 10.0
    rear_leg_position: float = 0.0
    joint_velocity: float = 0.15
    wheel_position: float = 45.0
    lateral_velocity: float = 4.0
    lateral_position: float = 0.0
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
    base_height_drop_margin_m: float = 0.08
    base_height_drop_stop_frame: float = 0.0
    lateral_position_start_frame: float = 0.0
    front_normal_min_n: float = 6.0
    front_normal_deficit_power: float = 2.0
    front_support_worst_fraction: float = 0.0
    front_force_balance_scale_n: float = 0.0
    front_contact_position_margin_m: float = 0.0
    front_contact_position_scale_m: float = 0.0
    front_contact_position_max_normalized: float = 0.0
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
        if (
            result.base_height_drop_margin_m < 0.0
            or result.base_height_drop_stop_frame < 0.0
            or result.lateral_position_start_frame < 0.0
            or result.front_normal_min_n < 0.0
        ):
            raise ValueError(
                "Scheduled tracking frames, base-height margin, and "
                "front-normal minimum must be non-negative."
            )
        if not 1.0 <= result.front_normal_deficit_power <= 2.0:
            raise ValueError(
                "front_normal_deficit_power must be in [1, 2]."
            )
        if not 0.0 <= result.front_support_worst_fraction <= 1.0:
            raise ValueError(
                "front_support_worst_fraction must be in [0, 1]."
            )
        if (
            result.front_force_balance_scale_n < 0.0
            or result.front_contact_position_margin_m < 0.0
            or result.front_contact_position_scale_m < 0.0
            or result.front_contact_position_max_normalized < 0.0
        ):
            raise ValueError(
                "Front force-balance scale and contact-position margin, "
                "scale, and cap must be non-negative."
            )
        if (
            result.rear_normal_overload_n <= 0.0
            or result.rear_normal_scale_n <= 0.0
            or result.rear_balance_scale_n <= 0.0
        ):
            raise ValueError("Rear-force load limits and scales must be positive.")
        return result


def base_height_drop_cost(
    base_position_error: torch.Tensor,
    margin_m: float,
) -> torch.Tensor:
    """Penalize only downward base error beyond a protected height margin."""

    if base_position_error.ndim != 2 or base_position_error.shape[1] != 3:
        raise ValueError("base_position_error must have shape [batch,3].")
    if margin_m < 0.0:
        raise ValueError("margin_m must be non-negative.")
    return torch.relu(
        -base_position_error[:, 2] - margin_m
    ).square()


def rear_leg_position_cost(joint_position_error: torch.Tensor) -> torch.Tensor:
    """Track the rear legs in type-grouped policy joint order."""

    if (
        joint_position_error.ndim != 2
        or joint_position_error.shape[1] != 12
    ):
        raise ValueError(
            "joint_position_error must have shape [batch,12]."
        )
    rear_leg_indices = (2, 3, 6, 7, 10, 11)
    return joint_position_error[:, rear_leg_indices].square().mean(dim=-1)


def load_support_cost_terms(
    contact_force_w: torch.Tensor,
    desired_contact: torch.Tensor,
    contact_force_threshold_n: float,
    limits: IsaacRolloutLoadLimits,
    wheel_position_error_w: torch.Tensor | None = None,
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
    front_deficit_cost = (
        front_deficit.pow(limits.front_normal_deficit_power)
        * desired_front
    )
    front_normal_support_mean = (
        front_deficit_cost.sum(dim=-1)
        / desired_front.sum(dim=-1).clamp_min(1.0)
    )
    front_normal_support_worst = front_deficit_cost.max(dim=-1).values
    front_normal_support = (
        (1.0 - limits.front_support_worst_fraction)
        * front_normal_support_mean
        + limits.front_support_worst_fraction
        * front_normal_support_worst
    )
    if limits.front_force_balance_scale_n > 0.0:
        # A force deficit already asks each desired front contact to carry the
        # minimum load, but a low-ESS MPPI solve can still settle on one strong
        # and one weak front wheel.  Add a bounded balance proxy only while
        # both front wheels are scheduled as support; single-front swing
        # phases must remain free to unload the moving limb.
        front_imbalance = torch.clamp(
            (front_normal[:, 0] - front_normal[:, 1])
            / limits.front_force_balance_scale_n,
            min=-1.0,
            max=1.0,
        ).square()
        both_front_desired = torch.all(desired_contact[:, :2], dim=-1).float()
        front_normal_support = (
            front_normal_support + front_imbalance * both_front_desired
        )
    if limits.front_contact_position_scale_m > 0.0:
        if (
            wheel_position_error_w is None
            or wheel_position_error_w.shape != contact_force_w.shape
        ):
            raise ValueError(
                "wheel_position_error_w must have shape [batch,4,3] when "
                "front contact-position support is enabled."
            )
        # The trunk is on the +world-X side of the robot.  A negative front
        # wheel X error is therefore detachment, not harmless tangential slip.
        # Gate the smooth geometric proxy by the kinematic contact schedule so
        # the deliberate 8 mm front swing remains unpenalized.
        front_detachment = torch.relu(
            -wheel_position_error_w[:, :2, 0]
            - limits.front_contact_position_margin_m
        ) / limits.front_contact_position_scale_m
        if limits.front_contact_position_max_normalized > 0.0:
            front_detachment = torch.clamp(
                front_detachment,
                max=limits.front_contact_position_max_normalized,
            )
        front_normal_support = front_normal_support + (
            front_detachment.square() * desired_front
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

    def _restore_snapshot_state_and_buffers(
        self,
        snapshot: IsaacRolloutSnapshot,
        *,
        forward_after_state_write: bool = True,
    ) -> None:
        repeated_state = _tree_repeat(snapshot.scene_state_relative, self.num_envs)
        self.base.scene.reset_to(repeated_state, env_ids=self.env_ids, is_relative=True)
        if forward_after_state_write:
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

    def restore(
        self,
        snapshot: IsaacRolloutSnapshot,
        *,
        clear_contact_warm_start: bool = True,
        forward_after_state_write: bool = True,
        contact_prime_substeps: int = 0,
    ) -> None:
        """Restore explicit state and optionally prime deterministic contacts.

        Direct PhysX tensor writes cannot restore opaque solver impulses.  A
        cold contact pair measurably changes the front normal load on the next
        real control step.  Priming advances the already-processed snapshot
        action for a small number of physics substeps, then restores every
        explicit state and manager/sensor buffer once more without destroying
        the newly-built contact pairs.
        """

        contact_prime_substeps = int(contact_prime_substeps)
        if contact_prime_substeps < 0:
            raise ValueError("contact_prime_substeps must be non-negative.")
        if contact_prime_substeps and not clear_contact_warm_start:
            raise ValueError(
                "contact priming requires a cold contact restore."
            )
        if clear_contact_warm_start:
            self._clear_contact_warm_start(snapshot)
        self._restore_snapshot_state_and_buffers(
            snapshot,
            forward_after_state_write=forward_after_state_write,
        )
        for _ in range(contact_prime_substeps):
            self.base._sim_step_counter += 1
            self.base.action_manager.apply_action()
            self.base.scene.write_data_to_sim()
            self.base.sim.step(render=False)
            self.base.scene.update(dt=self.base.physics_dt)
        if contact_prime_substeps:
            self._restore_snapshot_state_and_buffers(
                snapshot,
                forward_after_state_write=forward_after_state_write,
            )

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
        *,
        action_residual_weight: float | None = None,
        base_orientation_cost_multiplier: float = 1.0,
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
        effective_action_residual_weight = (
            weights.action_residual
            if action_residual_weight is None
            else float(action_residual_weight)
        )
        if (
            not np.isfinite(effective_action_residual_weight)
            or effective_action_residual_weight < 0.0
        ):
            raise ValueError(
                "action_residual_weight must be finite and non-negative."
            )
        effective_base_orientation_cost_multiplier = float(
            base_orientation_cost_multiplier
        )
        if (
            not np.isfinite(effective_base_orientation_cost_multiplier)
            or effective_base_orientation_cost_multiplier < 1.0
        ):
            raise ValueError(
                "base_orientation_cost_multiplier must be finite and at "
                "least 1.0."
            )
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
            add(
                "base_position",
                base_height_drop_cost(
                    base_pos_error,
                    self.load_limits.base_height_drop_margin_m,
                ),
                weights.base_height_drop,
                (
                    active
                    if (
                        self.load_limits.base_height_drop_stop_frame <= 0.0
                        or frame
                        < self.load_limits.base_height_drop_stop_frame
                    )
                    else torch.zeros_like(active)
                ),
            )
            add(
                "base_orientation",
                orientation_error.square(),
                (
                    weights.base_orientation
                    * effective_base_orientation_cost_multiplier
                ),
                active,
            )

            q = self.robot.data.joint_pos[:, self.joint_ids]
            dq = self.robot.data.joint_vel[:, self.joint_ids]
            q_error = q[:, :12] - target["joint_pos"][:12]
            dq_error = dq[:, :12] - target["joint_vel"][:12]
            add("joint_position", q_error.square().mean(dim=-1), weights.joint_position, active)
            add(
                "rear_leg_position",
                rear_leg_position_cost(q_error),
                weights.rear_leg_position,
                active,
            )
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
            add(
                "lateral_position",
                base_pos_error[:, 1].square(),
                weights.lateral_position,
                (
                    active
                    if frame >= self.load_limits.lateral_position_start_frame
                    else torch.zeros_like(active)
                ),
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
                wheel_position_error_w=wheel_error,
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
            add(
                "action_residual",
                residual.square().mean(dim=-1),
                effective_action_residual_weight,
                active,
            )
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
        *,
        action_residual_weight: float | None = None,
        base_orientation_cost_multiplier: float = 1.0,
    ) -> dict[str, Any]:
        snapshot = self.capture()
        try:
            first_cost = self.evaluate(
                candidates_leg,
                snapshot,
                nominal_leg,
                action_residual_weight=action_residual_weight,
                base_orientation_cost_multiplier=(
                    base_orientation_cost_multiplier
                ),
            ).clone()
            first_state = self.state_vector().clone()
            second_cost = self.evaluate(
                candidates_leg,
                snapshot,
                nominal_leg,
                action_residual_weight=action_residual_weight,
                base_orientation_cost_multiplier=(
                    base_orientation_cost_multiplier
                ),
            ).clone()
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
        nominal_action_reference_q_des_by_ref: dict[
            int,
            np.ndarray | torch.Tensor,
        ]
        | None = None,
        nominal_action_reference_raw_by_ref: dict[
            int,
            np.ndarray | torch.Tensor,
        ]
        | None = None,
        nominal_action_reference_overrides_by_ref: dict[
            int,
            dict[str, Any],
        ]
        | None = None,
        nominal_joint_position_bias_leg: list[float] | tuple[float, ...] | torch.Tensor | None = None,
        nominal_joint_position_bias_start_frame: int = 0,
        nominal_joint_position_bias_ramp_frames: int = 0,
        nominal_front_force_feedback_target_n: float = 0.0,
        nominal_front_force_feedback_gain_leg: list[float] | tuple[float, ...] | torch.Tensor | None = None,
        output_front_force_feedback_target_n: float = 0.0,
        output_front_force_feedback_min_contact_n: float = 0.0,
        output_front_force_feedback_lookahead_steps: int | None = None,
        output_front_force_feedback_gain_leg: list[float] | tuple[float, ...] | torch.Tensor | None = None,
        output_pitch_feedback_ref_ids: list[int] | tuple[int, ...] | None = None,
        output_pitch_feedback_gain_leg: list[float] | tuple[float, ...] | torch.Tensor | None = None,
        output_pitch_feedback_max_abs_rad: float = 0.0,
        output_joint_position_offset_leg: list[float] | tuple[float, ...] | torch.Tensor | None = None,
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
        self.nominal_action_reference_q_des_by_ref: dict[
            int,
            torch.Tensor,
        ] = {}
        for ref_id, values in (
            nominal_action_reference_q_des_by_ref or {}
        ).items():
            ref_id = int(ref_id)
            if not 0 <= ref_id < len(references):
                raise ValueError(
                    "Nominal action reference ID is outside the active "
                    f"reference bank: {ref_id}."
                )
            q_des = torch.as_tensor(
                values,
                dtype=torch.float32,
                device=adapter.base.device,
            )
            expected_shape = (references[ref_id].frames, 12)
            if tuple(q_des.shape) != expected_shape:
                raise ValueError(
                    "Nominal action reference q_des shape mismatch for "
                    f"ref {ref_id}: expected {expected_shape}, got "
                    f"{tuple(q_des.shape)}."
                )
            if not torch.isfinite(q_des).all():
                raise ValueError(
                    f"Nominal action reference for ref {ref_id} contains "
                    "NaN or Inf."
                )
            self.nominal_action_reference_q_des_by_ref[ref_id] = q_des
        self.nominal_action_reference_raw_by_ref: dict[
            int,
            torch.Tensor,
        ] = {}
        for ref_id, values in (
            nominal_action_reference_raw_by_ref or {}
        ).items():
            ref_id = int(ref_id)
            if ref_id in self.nominal_action_reference_q_des_by_ref:
                raise ValueError(
                    "A reference ID cannot have both q_des and raw nominal "
                    f"action references: ref {ref_id}."
                )
            if not 0 <= ref_id < len(references):
                raise ValueError(
                    "Nominal action reference ID is outside the active "
                    f"reference bank: {ref_id}."
                )
            raw_action = torch.as_tensor(
                values,
                dtype=torch.float32,
                device=adapter.base.device,
            )
            expected_shape = (references[ref_id].frames, 12)
            if tuple(raw_action.shape) != expected_shape:
                raise ValueError(
                    "Nominal raw action reference shape mismatch for "
                    f"ref {ref_id}: expected {expected_shape}, got "
                    f"{tuple(raw_action.shape)}."
                )
            if not torch.isfinite(raw_action).all():
                raise ValueError(
                    f"Nominal raw action reference for ref {ref_id} contains "
                    "NaN or Inf."
                )
            if bool(
                torch.any(raw_action < self.raw_min).item()
                or torch.any(raw_action > self.raw_max).item()
            ):
                raise ValueError(
                    f"Nominal raw action reference for ref {ref_id} exceeds "
                    "the frozen raw-action bounds."
                )
            self.nominal_action_reference_raw_by_ref[ref_id] = raw_action
        self.nominal_action_reference_overrides_by_ref: dict[
            int,
            dict[str, Any],
        ] = {}
        for ref_id, override in (
            nominal_action_reference_overrides_by_ref or {}
        ).items():
            ref_id = int(ref_id)
            if (
                ref_id not in self.nominal_action_reference_q_des_by_ref
                and ref_id not in self.nominal_action_reference_raw_by_ref
            ):
                raise ValueError(
                    "Nominal action solver overrides require an action "
                    f"reference for ref {ref_id}."
                )
            self.nominal_action_reference_overrides_by_ref[ref_id] = (
                normalize_nominal_solver_overrides(override)
            )
        if nominal_joint_position_bias_leg is None:
            self.nominal_joint_position_bias_leg = torch.zeros(
                12,
                dtype=torch.float32,
                device=adapter.base.device,
            )
        else:
            self.nominal_joint_position_bias_leg = torch.as_tensor(
                nominal_joint_position_bias_leg,
                dtype=torch.float32,
                device=adapter.base.device,
            )
            if self.nominal_joint_position_bias_leg.shape != (12,):
                raise ValueError(
                    "nominal_joint_position_bias_leg must contain 12 physical "
                    "joint-position offsets."
                )
        if not torch.isfinite(self.nominal_joint_position_bias_leg).all():
            raise ValueError(
                "nominal_joint_position_bias_leg contains NaN or Inf."
            )
        self.nominal_joint_position_bias_start_frame = int(
            nominal_joint_position_bias_start_frame
        )
        self.nominal_joint_position_bias_ramp_frames = int(
            nominal_joint_position_bias_ramp_frames
        )
        if self.nominal_joint_position_bias_start_frame < 0:
            raise ValueError(
                "nominal_joint_position_bias_start_frame must be non-negative."
            )
        if self.nominal_joint_position_bias_ramp_frames < 0:
            raise ValueError(
                "nominal_joint_position_bias_ramp_frames must be non-negative."
            )
        self.nominal_front_force_feedback_target_n = float(
            nominal_front_force_feedback_target_n
        )
        if (
            not np.isfinite(self.nominal_front_force_feedback_target_n)
            or self.nominal_front_force_feedback_target_n < 0.0
        ):
            raise ValueError(
                "nominal_front_force_feedback_target_n must be finite and "
                "non-negative."
            )
        if self.nominal_action_reference_raw_by_ref and (
            bool(torch.any(self.nominal_joint_position_bias_leg != 0.0).item())
            or self.nominal_front_force_feedback_target_n != 0.0
        ):
            raise ValueError(
                "Raw nominal action references require zero nominal physical "
                "bias and zero nominal front-force feedback so their stored "
                "float32 actions remain exact."
            )
        if nominal_front_force_feedback_gain_leg is None:
            self.nominal_front_force_feedback_gain_leg = torch.zeros(
                12,
                dtype=torch.float32,
                device=adapter.base.device,
            )
        else:
            self.nominal_front_force_feedback_gain_leg = torch.as_tensor(
                nominal_front_force_feedback_gain_leg,
                dtype=torch.float32,
                device=adapter.base.device,
            )
            if self.nominal_front_force_feedback_gain_leg.shape != (12,):
                raise ValueError(
                    "nominal_front_force_feedback_gain_leg must contain 12 "
                    "physical joint-position offsets."
                )
            if not torch.isfinite(
                self.nominal_front_force_feedback_gain_leg
            ).all():
                raise ValueError(
                    "nominal_front_force_feedback_gain_leg contains NaN or "
                    "Inf."
                )
        rear_joint_indices = torch.as_tensor(
            (2, 3, 6, 7, 10, 11),
            device=adapter.base.device,
        )
        if torch.any(
            self.nominal_front_force_feedback_gain_leg[
                rear_joint_indices
            ] != 0.0
        ):
            raise ValueError(
                "nominal_front_force_feedback_gain_leg must be zero for all "
                "rear-leg joints."
            )
        if (
            self.nominal_front_force_feedback_target_n == 0.0
            and torch.any(self.nominal_front_force_feedback_gain_leg != 0.0)
        ):
            raise ValueError(
                "A positive nominal_front_force_feedback_target_n is required "
                "when force-feedback gains are non-zero."
            )
        self.output_front_force_feedback_target_n = float(
            output_front_force_feedback_target_n
        )
        if (
            not np.isfinite(self.output_front_force_feedback_target_n)
            or self.output_front_force_feedback_target_n < 0.0
        ):
            raise ValueError(
                "output_front_force_feedback_target_n must be finite and "
                "non-negative."
            )
        self.output_front_force_feedback_min_contact_n = float(
            output_front_force_feedback_min_contact_n
        )
        if (
            not np.isfinite(
                self.output_front_force_feedback_min_contact_n
            )
            or self.output_front_force_feedback_min_contact_n < 0.0
        ):
            raise ValueError(
                "output_front_force_feedback_min_contact_n must be finite "
                "and non-negative."
            )
        feedback_lookahead = (
            config.reference_action_lookahead_steps
            if output_front_force_feedback_lookahead_steps is None
            else output_front_force_feedback_lookahead_steps
        )
        if (
            isinstance(feedback_lookahead, bool)
            or not isinstance(feedback_lookahead, (int, np.integer))
            or feedback_lookahead < 0
        ):
            raise ValueError(
                "output_front_force_feedback_lookahead_steps must be a "
                "non-negative integer."
            )
        self.output_front_force_feedback_lookahead_steps = int(
            feedback_lookahead
        )
        if output_front_force_feedback_gain_leg is None:
            self.output_front_force_feedback_gain_leg = torch.zeros(
                12,
                dtype=torch.float32,
                device=adapter.base.device,
            )
        else:
            self.output_front_force_feedback_gain_leg = torch.as_tensor(
                output_front_force_feedback_gain_leg,
                dtype=torch.float32,
                device=adapter.base.device,
            )
            if self.output_front_force_feedback_gain_leg.shape != (12,):
                raise ValueError(
                    "output_front_force_feedback_gain_leg must contain 12 "
                    "physical joint-position offsets."
                )
            if not torch.isfinite(
                self.output_front_force_feedback_gain_leg
            ).all():
                raise ValueError(
                    "output_front_force_feedback_gain_leg contains NaN or "
                    "Inf."
                )
        if torch.any(
            self.output_front_force_feedback_gain_leg[
                rear_joint_indices
            ] != 0.0
        ):
            raise ValueError(
                "output_front_force_feedback_gain_leg must be zero for all "
                "rear-leg joints."
            )
        if (
            self.output_front_force_feedback_target_n == 0.0
            and torch.any(self.output_front_force_feedback_gain_leg != 0.0)
        ):
            raise ValueError(
                "A positive output_front_force_feedback_target_n is required "
                "when output force-feedback gains are non-zero."
            )
        raw_pitch_feedback_ref_ids = tuple(
            output_pitch_feedback_ref_ids or ()
        )
        if any(
            isinstance(ref_id, bool)
            or not isinstance(ref_id, (int, np.integer))
            for ref_id in raw_pitch_feedback_ref_ids
        ):
            raise ValueError(
                "output_pitch_feedback_ref_ids must contain only integer "
                "reference IDs."
            )
        pitch_feedback_ref_ids = tuple(
            int(ref_id) for ref_id in raw_pitch_feedback_ref_ids
        )
        if len(set(pitch_feedback_ref_ids)) != len(
            pitch_feedback_ref_ids
        ):
            raise ValueError(
                "output_pitch_feedback_ref_ids must not contain duplicates."
            )
        if any(
            not 0 <= ref_id < len(references)
            for ref_id in pitch_feedback_ref_ids
        ):
            raise ValueError(
                "output_pitch_feedback_ref_ids contains an ID outside the "
                "active reference bank."
            )
        self.output_pitch_feedback_ref_ids = frozenset(
            pitch_feedback_ref_ids
        )
        if output_pitch_feedback_gain_leg is None:
            self.output_pitch_feedback_gain_leg = torch.zeros(
                12,
                dtype=torch.float32,
                device=adapter.base.device,
            )
        else:
            self.output_pitch_feedback_gain_leg = torch.as_tensor(
                output_pitch_feedback_gain_leg,
                dtype=torch.float32,
                device=adapter.base.device,
            )
            if self.output_pitch_feedback_gain_leg.shape != (12,):
                raise ValueError(
                    "output_pitch_feedback_gain_leg must contain 12 "
                    "physical joint-position gains."
                )
            if not torch.isfinite(
                self.output_pitch_feedback_gain_leg
            ).all():
                raise ValueError(
                    "output_pitch_feedback_gain_leg contains NaN or Inf."
                )
        self.output_pitch_feedback_max_abs_rad = float(
            output_pitch_feedback_max_abs_rad
        )
        if (
            not np.isfinite(self.output_pitch_feedback_max_abs_rad)
            or self.output_pitch_feedback_max_abs_rad < 0.0
        ):
            raise ValueError(
                "output_pitch_feedback_max_abs_rad must be finite and "
                "non-negative."
            )
        if torch.any(self.output_pitch_feedback_gain_leg != 0.0):
            if not self.output_pitch_feedback_ref_ids:
                raise ValueError(
                    "Non-zero output pitch-feedback gains require at least "
                    "one output_pitch_feedback_ref_id."
                )
            if self.output_pitch_feedback_max_abs_rad == 0.0:
                raise ValueError(
                    "Non-zero output pitch-feedback gains require a positive "
                    "output_pitch_feedback_max_abs_rad."
                )
        if output_joint_position_offset_leg is None:
            self.output_joint_position_offset_leg = torch.zeros(
                12,
                dtype=torch.float32,
                device=adapter.base.device,
            )
        else:
            self.output_joint_position_offset_leg = torch.as_tensor(
                output_joint_position_offset_leg,
                dtype=torch.float32,
                device=adapter.base.device,
            )
            if self.output_joint_position_offset_leg.shape != (12,):
                raise ValueError(
                    "output_joint_position_offset_leg must contain 12 "
                    "physical joint-position offsets."
                )
            if not torch.isfinite(
                self.output_joint_position_offset_leg
            ).all():
                raise ValueError(
                    "output_joint_position_offset_leg contains NaN or Inf."
                )
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
        self._active_solver_schedule_phase_by_ref: dict[int, int] = {}
        self.last_diagnostics: dict[str, Any] = {}

    def reset(self, episode_metadata: dict[str, Any] | None = None) -> None:
        self._warm_residual = None
        self._active_solver_schedule_phase_by_ref = {}
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
        raw_action_reference = (
            self.nominal_action_reference_raw_by_ref.get(request.ref_id)
        )
        if raw_action_reference is not None:
            return raw_action_reference[
                torch.as_tensor(
                    frames,
                    dtype=torch.long,
                    device=self.adapter.base.device,
                )
            ]
        action_reference = (
            self.nominal_action_reference_q_des_by_ref.get(request.ref_id)
        )
        if action_reference is None:
            q_ref = torch.as_tensor(
                reference.joint_pos[frames, :12],
                dtype=torch.float32,
                device=self.adapter.base.device,
            )
        else:
            q_ref = action_reference[
                torch.as_tensor(
                    frames,
                    dtype=torch.long,
                    device=self.adapter.base.device,
                )
            ]
        if self.nominal_joint_position_bias_ramp_frames == 0:
            bias_factor = torch.as_tensor(
                frames >= self.nominal_joint_position_bias_start_frame,
                dtype=torch.float32,
                device=self.adapter.base.device,
            )
        else:
            bias_factor = torch.as_tensor(
                (
                    frames - self.nominal_joint_position_bias_start_frame
                )
                / self.nominal_joint_position_bias_ramp_frames,
                dtype=torch.float32,
                device=self.adapter.base.device,
            ).clamp_(0.0, 1.0)
        q_ref = q_ref + (
            bias_factor.unsqueeze(-1) * self.nominal_joint_position_bias_leg
        )
        if self.nominal_front_force_feedback_target_n > 0.0:
            front_normal = torch.abs(
                self.adapter.contact_sensor.data.net_forces_w[
                    0,
                    self.adapter.contact_body_ids[:2],
                    0,
                ]
            )
            force_deficit = torch.clamp(
                (
                    self.nominal_front_force_feedback_target_n
                    - front_normal
                )
                / self.nominal_front_force_feedback_target_n,
                min=0.0,
                max=1.0,
            )
            desired_front = torch.as_tensor(
                self.adapter.contact_schedules[request.ref_id][frames, :2],
                dtype=torch.float32,
                device=self.adapter.base.device,
            )
            feedback_factor = torch.zeros(
                (self.config.horizon, 12),
                dtype=torch.float32,
                device=self.adapter.base.device,
            )
            for front_index, joint_indices in enumerate(
                ((0, 4, 8), (1, 5, 9))
            ):
                feedback_factor[:, joint_indices] = (
                    desired_front[:, front_index : front_index + 1]
                    * force_deficit[front_index]
                )
            q_ref = q_ref + (
                feedback_factor
                * self.nominal_front_force_feedback_gain_leg
            )
        return (q_ref - self.offset) / self.scale

    def _apply_output_front_force_feedback(
        self,
        selected_leg: torch.Tensor,
        nominal_leg: torch.Tensor,
        request: ExpertRequest,
        previous_action: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Apply measured front-load correction to the selected raw action.

        Proposal centring alone does not guarantee that the finite-sample MPPI
        solve executes the desired correction.  This final correction is
        therefore deterministic, contact-schedule gated, and projected back
        through the same raw bounds and per-step target-rate limits used by the
        optimizer.  The caller still appends four exact-zero wheel actions.
        """

        if self.output_front_force_feedback_target_n == 0.0:
            return selected_leg, {
                "enabled": False,
                "front_normal_n": [0.0, 0.0],
                "desired_front_contact": [False, False],
                "measured_front_contact": [False, False],
                "force_deficit_fraction": [0.0, 0.0],
                "requested_correction_rad": [0.0] * 12,
                "absolute_feedback_limit_rad": [0.0] * 12,
                "applied_correction_rad": [0.0] * 12,
            }

        front_normal = torch.abs(
            self.adapter.contact_sensor.data.net_forces_w[
                0,
                self.adapter.contact_body_ids[:2],
                0,
            ]
        )
        force_deficit = torch.clamp(
            (
                self.output_front_force_feedback_target_n
                - front_normal
            )
            / self.output_front_force_feedback_target_n,
            min=0.0,
            max=1.0,
        )
        measured_front_contact = (
            front_normal
            >= self.output_front_force_feedback_min_contact_n
        )
        schedule = self.adapter.contact_schedules[request.ref_id]
        schedule_frame = min(
            request.ref_frame
            + self.output_front_force_feedback_lookahead_steps
            + self.adapter.action_delay_steps,
            len(schedule) - 1,
        )
        desired_front = torch.as_tensor(
            schedule[schedule_frame, :2],
            dtype=torch.float32,
            device=self.adapter.base.device,
        )
        feedback_factor = torch.zeros(
            12,
            dtype=torch.float32,
            device=self.adapter.base.device,
        )
        for front_index, joint_indices in enumerate(
            ((0, 4, 8), (1, 5, 9))
        ):
            feedback_factor[list(joint_indices)] = (
                desired_front[front_index]
                * measured_front_contact[front_index].float()
                * force_deficit[front_index]
            )
        requested_correction = (
            feedback_factor * self.output_front_force_feedback_gain_leg
        )
        proposed_leg = selected_leg + requested_correction / self.scale
        # A raw additive correction can otherwise integrate through the
        # previous-action constraint after contact is lost: every new solve is
        # allowed to add the same positive offset again.  Bound each corrected
        # dimension by one full configured feedback offset from this solve's
        # reference-centred nominal action.  This retains the measured local
        # response direction without creating an unbounded position ramp.
        full_offset_raw = (
            self.output_front_force_feedback_gain_leg / self.scale
        )
        feedback_limit = nominal_leg + full_offset_raw
        positive_gain = self.output_front_force_feedback_gain_leg > 0.0
        negative_gain = self.output_front_force_feedback_gain_leg < 0.0
        proposed_leg = torch.where(
            positive_gain,
            torch.maximum(
                selected_leg,
                torch.minimum(proposed_leg, feedback_limit),
            ),
            proposed_leg,
        )
        proposed_leg = torch.where(
            negative_gain,
            torch.minimum(
                selected_leg,
                torch.maximum(proposed_leg, feedback_limit),
            ),
            proposed_leg,
        )
        lower = torch.maximum(
            self.raw_min,
            previous_action - self.max_delta,
        )
        upper = torch.minimum(
            self.raw_max,
            previous_action + self.max_delta,
        )
        corrected_leg = torch.minimum(
            torch.maximum(proposed_leg, lower),
            upper,
        )
        applied_correction = self.scale * (corrected_leg - selected_leg)
        diagnostics = {
            "enabled": True,
            "schedule_frame": schedule_frame,
            "schedule_lookahead_steps": (
                self.output_front_force_feedback_lookahead_steps
            ),
            "front_normal_n": front_normal.detach().cpu().tolist(),
            "desired_front_contact": (
                desired_front.to(dtype=torch.bool).detach().cpu().tolist()
            ),
            "measured_front_contact": (
                measured_front_contact.detach().cpu().tolist()
            ),
            "force_deficit_fraction": force_deficit.detach().cpu().tolist(),
            "requested_correction_rad": (
                requested_correction.detach().cpu().tolist()
            ),
            "absolute_feedback_limit_rad": (
                (self.offset + self.scale * feedback_limit)
                .detach()
                .cpu()
                .tolist()
            ),
            "applied_correction_rad": (
                applied_correction.detach().cpu().tolist()
            ),
        }
        return corrected_leg, diagnostics

    def _apply_output_joint_position_offset(
        self,
        selected_leg: torch.Tensor,
        previous_action: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Apply a fixed physical target offset under the frozen action limits."""

        if not torch.any(self.output_joint_position_offset_leg != 0.0):
            return selected_leg, {
                "enabled": False,
                "requested_correction_rad": [0.0] * 12,
                "applied_correction_rad": [0.0] * 12,
            }
        proposed_leg = (
            selected_leg
            + self.output_joint_position_offset_leg / self.scale
        )
        lower = torch.maximum(
            self.raw_min,
            previous_action - self.max_delta,
        )
        upper = torch.minimum(
            self.raw_max,
            previous_action + self.max_delta,
        )
        corrected_leg = torch.minimum(
            torch.maximum(proposed_leg, lower),
            upper,
        )
        applied_correction = self.scale * (
            corrected_leg - selected_leg
        )
        return corrected_leg, {
            "enabled": True,
            "requested_correction_rad": (
                self.output_joint_position_offset_leg.detach().cpu().tolist()
            ),
            "applied_correction_rad": (
                applied_correction.detach().cpu().tolist()
            ),
        }

    def _apply_output_pitch_feedback(
        self,
        selected_leg: torch.Tensor,
        nominal_leg: torch.Tensor,
        ref_id: int,
        actual_quat_w: torch.Tensor,
        target_quat_w: torch.Tensor,
        previous_action: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Apply bounded state feedback for signed base-pitch error.

        The correction is expressed in physical joint-position radians and is
        capped relative to this solve's nominal action.  It therefore cannot
        integrate through the previous-action rate constraint.  Raw bounds and
        the same per-step physical target-rate limit used by MPPI are applied
        again before returning.
        """

        configured = bool(
            torch.any(self.output_pitch_feedback_gain_leg != 0.0).item()
        )
        active_for_ref = ref_id in self.output_pitch_feedback_ref_ids
        if not configured or not active_for_ref:
            return selected_leg, {
                "enabled": False,
                "configured": configured,
                "active_for_ref": active_for_ref,
                "signed_pitch_error_rad": 0.0,
                "requested_correction_rad": [0.0] * 12,
                "absolute_feedback_limit_rad": [0.0] * 12,
                "applied_correction_rad": [0.0] * 12,
            }

        signed_pitch_error = _signed_pitch_error_rad(
            actual_quat_w,
            target_quat_w,
        )
        requested_correction = torch.clamp(
            signed_pitch_error
            * self.output_pitch_feedback_gain_leg,
            min=-self.output_pitch_feedback_max_abs_rad,
            max=self.output_pitch_feedback_max_abs_rad,
        )
        proposed_leg = selected_leg + requested_correction / self.scale
        feedback_limit = nominal_leg + requested_correction / self.scale
        positive_correction = requested_correction > 0.0
        negative_correction = requested_correction < 0.0
        proposed_leg = torch.where(
            positive_correction,
            torch.maximum(
                selected_leg,
                torch.minimum(proposed_leg, feedback_limit),
            ),
            proposed_leg,
        )
        proposed_leg = torch.where(
            negative_correction,
            torch.minimum(
                selected_leg,
                torch.maximum(proposed_leg, feedback_limit),
            ),
            proposed_leg,
        )
        lower = torch.maximum(
            self.raw_min,
            previous_action - self.max_delta,
        )
        upper = torch.minimum(
            self.raw_max,
            previous_action + self.max_delta,
        )
        corrected_leg = torch.minimum(
            torch.maximum(proposed_leg, lower),
            upper,
        )
        applied_correction = self.scale * (
            corrected_leg - selected_leg
        )
        return corrected_leg, {
            "enabled": True,
            "configured": True,
            "active_for_ref": True,
            "signed_pitch_error_rad": float(
                signed_pitch_error.detach().cpu().item()
            ),
            "max_abs_correction_rad": (
                self.output_pitch_feedback_max_abs_rad
            ),
            "requested_correction_rad": (
                requested_correction.detach().cpu().tolist()
            ),
            "absolute_feedback_limit_rad": (
                (self.offset + self.scale * feedback_limit)
                .detach()
                .cpu()
                .tolist()
            ),
            "applied_correction_rad": (
                applied_correction.detach().cpu().tolist()
            ),
        }

    def __call__(self, request: ExpertRequest) -> ExpertReply:
        start = time.perf_counter()
        snapshot: IsaacRolloutSnapshot | None = None
        try:
            request.validate()
            nominal = self._nominal(request)
            previous_action = self.adapter.previous_commanded_action[0, :12].clone()
            initial = nominal
            configured_reference_overrides = (
                self.nominal_action_reference_overrides_by_ref.get(
                    request.ref_id,
                    {},
                )
            )
            reference_overrides, solver_schedule_phase = (
                resolve_nominal_solver_overrides(
                    configured_reference_overrides,
                    request.ref_frame,
                )
            )
            solver_schedule_start_frame: int | None = None
            solver_schedule_reset_warm_start = False
            if solver_schedule_phase is not None:
                phase = configured_reference_overrides[
                    "solver_schedule"
                ][solver_schedule_phase]
                solver_schedule_start_frame = int(phase["start_frame"])
                previous_phase = (
                    self._active_solver_schedule_phase_by_ref.get(
                        request.ref_id
                    )
                )
                phase_changed = previous_phase != solver_schedule_phase
                solver_schedule_reset_warm_start = bool(
                    phase_changed
                    and phase.get("reset_warm_start", False)
                )
                if solver_schedule_reset_warm_start:
                    self._warm_residual = None
                self._active_solver_schedule_phase_by_ref[
                    request.ref_id
                ] = solver_schedule_phase
            effective_warm_start = bool(
                reference_overrides.get(
                    "warm_start",
                    self.config.warm_start,
                )
            )
            effective_selection_mode = str(
                reference_overrides.get(
                    "selection_mode",
                    self.config.selection_mode,
                )
            )
            action_residual_weight = reference_overrides.get(
                "action_residual_weight"
            )
            base_orientation_cost_multiplier = float(
                reference_overrides.get(
                    "base_orientation_cost_multiplier",
                    1.0,
                )
            )
            if effective_warm_start and self._warm_residual is not None:
                shifted = torch.cat(
                    (self._warm_residual[1:], self._warm_residual[-1:]),
                    dim=0,
                )
                initial = nominal + shifted

            snapshot = self.rollout.capture()
            sequence, optimizer_diagnostics = self.optimizer.optimize(
                nominal,
                lambda candidates: self.rollout.evaluate(
                    candidates,
                    snapshot,
                    nominal,
                    action_residual_weight=action_residual_weight,
                    base_orientation_cost_multiplier=(
                        base_orientation_cost_multiplier
                    ),
                ),
                self.raw_min,
                self.raw_max,
                previous_action=previous_action,
                max_delta=self.max_delta,
                initial_sequence=initial,
                selection_mode=effective_selection_mode,
            )
            self.rollout.restore(snapshot)
            self._warm_residual = (sequence - nominal).detach().clone()
            selected_leg = sequence[0]
            selected_leg, output_force_feedback = (
                self._apply_output_front_force_feedback(
                    selected_leg,
                    nominal[0],
                    request,
                    previous_action,
                )
            )
            aligned_target = self.rollout._aligned_reference(
                snapshot,
                request.ref_frame,
            )
            selected_leg, output_pitch_feedback = (
                self._apply_output_pitch_feedback(
                    selected_leg,
                    nominal[0],
                    request.ref_id,
                    self.rollout.command.robot_anchor_quat_w[0],
                    aligned_target["body_quat"][
                        self.rollout.ref_anchor_body_id
                    ],
                    previous_action,
                )
            )
            selected_leg, output_joint_offset = (
                self._apply_output_joint_position_offset(
                    selected_leg,
                    previous_action,
                )
            )
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
                "output_front_force_feedback": output_force_feedback,
                "output_pitch_feedback": output_pitch_feedback,
                "output_joint_position_offset": output_joint_offset,
                "effective_warm_start": effective_warm_start,
                "effective_selection_mode": effective_selection_mode,
                "effective_action_residual_weight": (
                    self.rollout.cost_weights.action_residual
                    if action_residual_weight is None
                    else float(action_residual_weight)
                ),
                "effective_base_orientation_cost_multiplier": (
                    base_orientation_cost_multiplier
                ),
                "solver_schedule_phase": solver_schedule_phase,
                "solver_schedule_start_frame": (
                    solver_schedule_start_frame
                ),
                "solver_schedule_reset_warm_start": (
                    solver_schedule_reset_warm_start
                ),
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
