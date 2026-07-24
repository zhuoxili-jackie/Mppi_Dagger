from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from lateral_mppi_dagger.config import load_yaml


@dataclass(frozen=True)
class ScenarioProfile:
    requested_name: str
    resolved_name: str
    values: dict[str, Any]

    @property
    def observation_noise_std(self) -> float:
        return float(self.values["observation_noise_std"])

    @property
    def action_delay_steps(self) -> int:
        return int(self.values["action_delay_steps"])

    @property
    def platform_position_jitter_m(self) -> tuple[float, float, float]:
        values = tuple(float(value) for value in self.values["platform_position_jitter_m"])
        if len(values) != 3:
            raise ValueError("platform_position_jitter_m must contain three values.")
        return values

    def metadata(self) -> dict[str, Any]:
        return {
            "requested_name": self.requested_name,
            "resolved_name": self.resolved_name,
            **deepcopy(self.values),
        }


def load_scenario_profile(
    name: str,
    config_path: str = "configs/scenarios.yaml",
) -> ScenarioProfile:
    config = load_yaml(config_path)
    aliases = config.get("aliases", {})
    resolved = str(aliases.get(name, name))
    profiles = config["profiles"]
    if resolved not in profiles:
        available = sorted(set(profiles) | set(aliases))
        raise ValueError(
            f"Unknown scenario profile {name!r}; choose one of {available}."
        )
    values = deepcopy(profiles[resolved])
    profile = ScenarioProfile(str(name), resolved, values)
    if profile.observation_noise_std < 0.0:
        raise ValueError("Scenario observation noise must be non-negative.")
    if profile.action_delay_steps < 0:
        raise ValueError("Scenario action delay must be non-negative.")
    profile.platform_position_jitter_m
    return profile


def _set_range_mapping(target: Any, values: dict[str, list[float]]) -> None:
    target.clear()
    target.update(
        {
            name: (float(bounds[0]), float(bounds[1]))
            for name, bounds in values.items()
        }
    )


def _fixed_range(value: float) -> tuple[float, float]:
    scalar = float(value)
    return (scalar, scalar)


def configure_env_for_scenario(
    env_cfg: Any,
    scenario: ScenarioProfile,
    num_envs: int,
) -> None:
    """Apply a recorded DAgger scenario while preserving clone comparability."""
    env_cfg.scene.num_envs = int(num_envs)
    env_cfg.observations.policy.enable_corruption = False
    if hasattr(env_cfg.observations, "critic"):
        env_cfg.observations.critic.enable_corruption = False
    motion = env_cfg.commands.motion
    motion.debug_vis = False
    initial = scenario.values["initial_state"]
    _set_range_mapping(motion.pose_range, initial["pose"])
    _set_range_mapping(motion.velocity_range, initial["velocity"])
    motion.joint_position_range = tuple(
        float(value) for value in initial["joint_position"]
    )

    enabled_events: set[str] = set()
    physics_profile = str(scenario.values.get("physics_profile", "nominal"))
    if physics_profile == "fixed_light_shift":
        enabled_events = {
            "randomize_rigid_body_material",
            "randomize_box_material",
            "randomize_front_wheel_material",
            "randomize_rear_wheel_material",
            "randomize_bodies_mass",
        }
    elif physics_profile != "nominal":
        raise ValueError(f"Unsupported physics profile {physics_profile!r}.")

    events = getattr(env_cfg, "events", None)
    if events is not None:
        for event_name in dir(events):
            if event_name.startswith("randomize_") and event_name not in enabled_events:
                setattr(events, event_name, None)

    if physics_profile == "fixed_light_shift":
        physics = scenario.values["fixed_physics"]
        required = {name: getattr(events, name, None) for name in enabled_events}
        missing = sorted(name for name, event in required.items() if event is None)
        if missing:
            raise RuntimeError(
                f"Scenario requires unavailable Isaac randomization terms: {missing}"
            )

        material_values = {
            "randomize_rigid_body_material": (
                physics["robot_non_wheel_static_friction"],
                physics["robot_non_wheel_dynamic_friction"],
            ),
            "randomize_box_material": (
                physics["box_static_friction"],
                physics["box_dynamic_friction"],
            ),
            "randomize_front_wheel_material": (
                physics["front_wheel_static_friction"],
                physics["front_wheel_dynamic_friction"],
            ),
            "randomize_rear_wheel_material": (
                physics["rear_wheel_static_friction"],
                physics["rear_wheel_dynamic_friction"],
            ),
        }
        for name, (static, dynamic) in material_values.items():
            event = required[name]
            event.params["static_friction_range"] = _fixed_range(static)
            event.params["dynamic_friction_range"] = _fixed_range(dynamic)
            event.params["restitution_range"] = (0.0, 0.0)
            event.params["num_buckets"] = 1
        mass_event = required["randomize_bodies_mass"]
        mass_event.params["mass_distribution_params"] = _fixed_range(
            physics["robot_mass_scale"]
        )
        mass_event.params["operation"] = "scale"

    if hasattr(env_cfg.scene, "contact_forces"):
        env_cfg.scene.contact_forces.debug_vis = False
