from __future__ import annotations

from types import SimpleNamespace

import pytest

from lateral_mppi_dagger.env.scenarios import (
    configure_env_for_scenario,
    load_scenario_profile,
)


def _event() -> SimpleNamespace:
    return SimpleNamespace(params={})


def _fake_env_cfg() -> SimpleNamespace:
    events = SimpleNamespace(
        randomize_rigid_body_material=_event(),
        randomize_box_material=_event(),
        randomize_front_wheel_material=_event(),
        randomize_rear_wheel_material=_event(),
        randomize_bodies_mass=_event(),
        randomize_push_robot=_event(),
    )
    return SimpleNamespace(
        scene=SimpleNamespace(
            num_envs=1,
            contact_forces=SimpleNamespace(debug_vis=True),
        ),
        observations=SimpleNamespace(
            policy=SimpleNamespace(enable_corruption=True),
            critic=SimpleNamespace(enable_corruption=True),
        ),
        commands=SimpleNamespace(
            motion=SimpleNamespace(
                debug_vis=True,
                pose_range={"x": (-1.0, 1.0)},
                velocity_range={"x": (-1.0, 1.0)},
                joint_position_range=(-1.0, 1.0),
            )
        ),
        events=events,
    )


def test_nominal_scenario_disables_randomization() -> None:
    cfg = _fake_env_cfg()
    profile = load_scenario_profile("nominal_fixed_gate")
    configure_env_for_scenario(cfg, profile, num_envs=16)
    assert profile.resolved_name == "nominal"
    assert cfg.scene.num_envs == 16
    assert cfg.commands.motion.pose_range["x"] == (0.0, 0.0)
    assert cfg.events.randomize_rigid_body_material is None
    assert cfg.events.randomize_push_robot is None


def test_fixed_physics_scenario_uses_clone_identical_ranges() -> None:
    cfg = _fake_env_cfg()
    profile = load_scenario_profile("light_friction_mass_observation_noise")
    configure_env_for_scenario(cfg, profile, num_envs=256)
    assert cfg.events.randomize_push_robot is None
    material = cfg.events.randomize_rigid_body_material.params
    assert material["static_friction_range"][0] == material["static_friction_range"][1]
    mass = cfg.events.randomize_bodies_mass.params["mass_distribution_params"]
    assert mass == (1.02, 1.02)
    assert profile.observation_noise_std == 0.005


def test_r3_profile_requests_real_delay_and_bounded_platform_jitter() -> None:
    profile = load_scenario_profile("light_delay_platform_pose")
    assert profile.action_delay_steps == 1
    assert profile.platform_position_jitter_m == (0.005, 0.005, 0.002)
    assert profile.observation_noise_std == 0.003


def test_unknown_scenario_fails_closed() -> None:
    with pytest.raises(ValueError, match="Unknown scenario profile"):
        load_scenario_profile("dagger_r3_typo")
