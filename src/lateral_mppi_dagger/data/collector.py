from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

import numpy as np

from lateral_mppi_dagger.contract.action16 import SafetyShield
from lateral_mppi_dagger.expert.base import (
    MPPI_COST_COMPONENT_NAMES,
    Expert,
    ExpertRequest,
    FailureCode,
    LabelSource,
)

from .schema import ENUMS, SCHEMA_VERSION, EpisodeShard


@dataclass(frozen=True)
class EnvironmentStep:
    next_obs93_clean: np.ndarray
    next_obs93_dynamic: np.ndarray
    applied_action16: np.ndarray
    terminated: bool
    truncated: bool
    termination_reason: int
    info: dict[str, Any]


class RolloutEnvironment(Protocol):
    control_dt: float

    def reset(self, seed: int, ref_id: int) -> tuple[np.ndarray, np.ndarray]: ...

    def expert_request(self) -> ExpertRequest: ...

    def step(self, executed_action16: np.ndarray) -> EnvironmentStep: ...


@dataclass(frozen=True)
class CollectorConfig:
    seed: int
    ref_id: int
    max_steps: int
    beta: float = 1.0
    observation_noise_std: float = 0.0
    behavior_selection: str = "episode"
    scenario: str = "nominal"

    def validate(self) -> None:
        if not 0.0 <= self.beta <= 1.0:
            raise ValueError("beta must lie in [0, 1]")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.behavior_selection != "episode":
            raise NotImplementedError("Only episode-level behavior selection is currently supported.")


StudentPolicy = Callable[[np.ndarray], np.ndarray]


def _label_source(source: str, valid: bool) -> LabelSource:
    if not valid:
        return LabelSource.INVALID
    mapping = {
        "reference_wbc": LabelSource.REFERENCE_WBC,
        "dwmpc": LabelSource.DWMPC,
        "mppi": LabelSource.MPPI,
        "fallback": LabelSource.SAFETY_FALLBACK,
    }
    return mapping.get(source, LabelSource.INVALID)


def collect_episode(
    environment: RolloutEnvironment,
    expert: Expert,
    shield: SafetyShield,
    config: CollectorConfig,
    episode_metadata: dict[str, Any],
    student_policy: StudentPolicy | None = None,
) -> EpisodeShard:
    config.validate()
    rng = np.random.default_rng(config.seed)
    shield.reset()
    observation, dynamic_observation = environment.reset(config.seed, config.ref_id)
    effective_episode_metadata = dict(episode_metadata)
    metadata_provider = getattr(environment, "episode_metadata", None)
    if callable(metadata_provider):
        runtime_metadata = metadata_provider()
        if not isinstance(runtime_metadata, dict):
            raise TypeError("environment.episode_metadata() must return a dictionary.")
        effective_episode_metadata.update(runtime_metadata)
    expert.reset(effective_episode_metadata)
    use_teacher_episode = bool(rng.random() < config.beta)
    fields: dict[str, list[Any]] = {}
    solver_status_codes: dict[str, int] = {}

    def append(name: str, value: Any) -> None:
        fields.setdefault(name, []).append(value)

    success = False
    for step_id in range(config.max_steps):
        train_observation = observation.copy()
        if config.observation_noise_std > 0.0:
            train_observation += rng.normal(
                0.0, config.observation_noise_std, size=train_observation.shape
            ).astype(np.float32)
            # Preserve graph/ABI structural zeros while perturbing measured
            # channels: wheel position slots, hard-zero previous wheel action,
            # and the legacy scalar.
            train_observation[53:57] = 0.0
            train_observation[85:89] = 0.0
            train_observation[92] = 0.0
        request = environment.expert_request()
        reply = expert.act(request)
        teacher_valid = bool(reply.valid)
        teacher_action = (
            np.asarray(reply.action16, dtype=np.float32)
            if teacher_valid
            else np.full(16, np.nan, dtype=np.float32)
        )
        teacher_q_des = (
            np.asarray(reply.q_des_leg, dtype=np.float32)
            if teacher_valid
            else np.full(12, np.nan, dtype=np.float32)
        )

        if student_policy is None:
            student_action = np.full(16, np.nan, dtype=np.float32)
            student_valid = False
        else:
            try:
                student_action = np.asarray(
                    student_policy(train_observation[None, :]),
                    dtype=np.float32,
                ).reshape(-1)
                student_valid = student_action.shape == (16,) and bool(np.isfinite(student_action).all())
            except Exception:
                student_action = np.full(16, np.nan, dtype=np.float32)
                student_valid = False

        choose_teacher = use_teacher_episode or not student_valid
        if choose_teacher and teacher_valid:
            pre_shield = teacher_action
            behavior_policy = ENUMS["behavior_policy"]["TEACHER"]
        elif student_valid:
            pre_shield = student_action
            behavior_policy = ENUMS["behavior_policy"]["STUDENT"]
        else:
            pre_shield = shield.last_action
            pre_shield[12:] = 0.0
            behavior_policy = ENUMS["behavior_policy"]["FALLBACK"]

        scheduled_action, shield_info = shield.apply(pre_shield)
        step_result = environment.step(scheduled_action)
        applied_action = np.asarray(step_result.applied_action16, dtype=np.float32).reshape(-1)
        if applied_action.shape != (16,) or not np.isfinite(applied_action).all():
            raise ValueError("Environment returned an invalid applied_action16.")

        status_code = solver_status_codes.setdefault(reply.solver_status, len(solver_status_codes) + 1)
        measured_contact = (
            np.linalg.norm(np.asarray(request.contact_force_w), axis=-1) >= 8.0
        ).astype(np.uint8)
        reached_requested_horizon = step_id == config.max_steps - 1
        terminal = bool(step_result.terminated or step_result.truncated or reached_requested_horizon)
        if step_result.terminated:
            reason = step_result.termination_reason or ENUMS["termination_reason"]["ENV_TERMINATED"]
        elif step_result.truncated or step_id == config.max_steps - 1:
            reason = ENUMS["termination_reason"]["TIME_LIMIT"]
        else:
            reason = ENUMS["termination_reason"]["NONE"]

        append("step_id", step_id)
        append("sim_time", (step_id + 1) * environment.control_dt)
        append("obs93_clean", observation)
        append("obs93_train", train_observation)
        append("next_obs93_clean", step_result.next_obs93_clean)
        append("obs93_dynamic", dynamic_observation)
        append("next_obs93_dynamic", step_result.next_obs93_dynamic)
        append("teacher_action16", teacher_action)
        append("student_action16", student_action)
        append("pre_shield_action16", pre_shield)
        append("scheduled_action16", scheduled_action)
        append("executed_action16", applied_action)
        append("action_clip_delta16", scheduled_action - pre_shield)
        append("teacher_q_des_leg", teacher_q_des)
        append("teacher_valid", teacher_valid)
        append("student_valid", student_valid)
        append("label_source", int(_label_source(reply.source, teacher_valid)))
        append("behavior_policy", behavior_policy)
        append("teacher_takeover", choose_teacher and teacher_valid)
        append("shield_intervened", shield_info.intervened)
        append("ref_id", request.ref_id)
        append("ref_frame", request.ref_frame)
        append("phase", float(request.ref_window.get("phase", 0.0)))
        append("target_vy", request.target_vy)
        append("desired_contact", np.asarray(request.desired_contact, dtype=np.uint8))
        append("measured_contact", measured_contact)
        append("base_pose_w", np.asarray(request.base_pose_w, dtype=np.float32))
        append("base_twist_w", np.asarray(request.base_twist_w, dtype=np.float32))
        append("q", np.asarray(request.q, dtype=np.float32))
        append("dq", np.asarray(request.dq, dtype=np.float32))
        append("wheel_body_pose_w", np.asarray(request.wheel_body_pose_w, dtype=np.float32))
        append("wheel_body_twist_w", np.asarray(request.wheel_body_twist_w, dtype=np.float32))
        append("contact_force_w", np.asarray(request.contact_force_w, dtype=np.float32))
        diagnostics = reply.diagnostics or {}
        cost_components = np.asarray(
            diagnostics.get(
                "cost_components",
                np.full(len(MPPI_COST_COMPONENT_NAMES), np.nan, dtype=np.float32),
            ),
            dtype=np.float32,
        )
        if cost_components.shape != (len(MPPI_COST_COMPONENT_NAMES),):
            cost_components = np.full(
                len(MPPI_COST_COMPONENT_NAMES), np.nan, dtype=np.float32
            )
        append("mppi_cost_components", cost_components)
        append(
            "mppi_minimum_total_cost",
            float(diagnostics.get("minimum_total_cost", np.nan)),
        )
        append(
            "mppi_mean_total_cost",
            float(diagnostics.get("mean_total_cost", np.nan)),
        )
        append(
            "mppi_effective_sample_size",
            float(diagnostics.get("effective_sample_size", np.nan)),
        )
        append(
            "mppi_rollout_termination_rate",
            float(diagnostics.get("rollout_termination_rate", np.nan)),
        )
        append("solver_status", status_code)
        append("solve_ms", reply.solve_ms if teacher_valid else np.nan)
        append("safety_margin", reply.safety_margin if teacher_valid else np.nan)
        failure_code = reply.failure_code
        if shield_info.failure_code == "NAN_INF":
            failure_code = FailureCode.NAN_INF
        append("failure_code", int(failure_code))
        append("terminal", terminal)
        append("termination_reason", reason)

        observation = step_result.next_obs93_clean
        dynamic_observation = step_result.next_obs93_dynamic
        if terminal:
            # A collector horizon is an evaluation boundary, not a simulated
            # failure.  Reaching it without an environment termination means
            # the controller survived the entire requested task interval.
            success = bool(
                not step_result.terminated
                and (step_result.truncated or reached_requested_horizon)
            )
            break

    arrays: dict[str, np.ndarray] = {}
    uint8_fields = {
        "teacher_valid",
        "student_valid",
        "label_source",
        "behavior_policy",
        "teacher_takeover",
        "shield_intervened",
        "desired_contact",
        "measured_contact",
        "terminal",
    }
    int16_fields = {"solver_status", "failure_code", "termination_reason"}
    int32_fields = {"step_id", "ref_id", "ref_frame"}
    float64_fields = {"sim_time"}
    for name, values in fields.items():
        if name in uint8_fields:
            dtype = np.uint8
        elif name in int16_fields:
            dtype = np.int16
        elif name in int32_fields:
            dtype = np.int32
        elif name in float64_fields:
            dtype = np.float64
        else:
            dtype = np.float32
        arrays[name] = np.asarray(values, dtype=dtype)

    metadata = dict(effective_episode_metadata)
    metadata.update(
        {
            "schema_version": SCHEMA_VERSION,
            "seed": config.seed,
            "ref_id": config.ref_id,
            "scenario": config.scenario,
            "beta": config.beta,
            "behavior_selection": config.behavior_selection,
            "success": success,
            "success_semantics": (
                "completed_requested_horizon_or_clean_env_truncation_without_env_termination"
            ),
            "enums": ENUMS,
            "solver_status_codes": solver_status_codes,
            "mppi_cost_component_order": list(MPPI_COST_COMPONENT_NAMES),
            "timing_semantics": (
                "obs_t -> teacher/student -> behavior -> pre_shield -> shield -> "
                "scheduled_command -> actuation_delay -> executed_action -> "
                "simulator -> next_obs"
            ),
        }
    )
    return EpisodeShard(arrays=arrays, metadata=metadata)
