from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from lateral_mppi_dagger.config import load_yaml
from lateral_mppi_dagger.export.exporter import export_student_policy
from lateral_mppi_dagger.export.validator import validate_export_bundle
from lateral_mppi_dagger.student.losses import imitation_loss
from lateral_mppi_dagger.student.model import StudentPolicy


def make_model() -> StudentPolicy:
    contract = load_yaml("configs/deployment_contract.yaml")
    return StudentPolicy(
        hidden_dims=(32, 16),
        wheel_action_mode="hard_zero",
        raw_min=torch.tensor(contract["action"]["raw_min"]),
        raw_max=torch.tensor(contract["action"]["raw_max"]),
    )


def make_safety_model() -> StudentPolicy:
    contract = load_yaml("configs/deployment_contract.yaml")
    return StudentPolicy(
        hidden_dims=(32, 16),
        wheel_action_mode="hard_zero",
        zero_command_previous_action_deadband=0.0,
        lateral_command_activation_start_m_s=0.0,
        lateral_command_activation_full_m_s=0.012,
        lateral_command_abs_limit_m_s=0.060,
        physical_target_rate_limit_rad_s=1.5,
        physical_target_abs_limit_rad=0.18,
        control_dt_s=0.02,
        raw_min=torch.tensor(contract["action"]["raw_min"]),
        raw_max=torch.tensor(contract["action"]["raw_max"]),
        action_scale=torch.tensor(contract["action"]["scale"]),
    )


def test_student_wheels_are_exact_zero_in_eager_and_torchscript() -> None:
    model = make_model().eval()
    observation = torch.randn(11, 93)
    eager = model(observation)
    scripted = torch.jit.script(model)(observation)
    assert eager.shape == scripted.shape == (11, 16)
    assert torch.equal(eager[:, 12:], torch.zeros_like(eager[:, 12:]))
    assert torch.equal(scripted[:, 12:], torch.zeros_like(scripted[:, 12:]))


def test_zero_command_dry_inference_ignores_prev_action_feedback() -> None:
    model = make_model().eval()
    first = torch.randn(1, 93)
    first[:, 89:92] = 0.0
    second = first.clone()
    second[:, 73:89] = torch.randn(1, 16) * 10.0
    scripted = torch.jit.script(model)
    with torch.no_grad():
        eager_first = model(first)
        eager_second = model(second)
        scripted_first = scripted(first)
        scripted_second = scripted(second)
    torch.testing.assert_close(eager_first, eager_second, atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        scripted_first,
        scripted_second,
        atol=0.0,
        rtol=0.0,
    )


def test_continuous_key7_ramp_has_bounded_physical_target_steps() -> None:
    model = make_safety_model().eval()
    contract = load_yaml("configs/deployment_contract.yaml")
    scale = torch.tensor(contract["action"]["scale"])
    observation = torch.randn(1, 93)
    observation[:, 73:89] = 0.0
    previous_physical = torch.zeros(12)
    commands = (0.0, 0.0, 0.0, 0.012, 0.024, 0.030)
    with torch.no_grad():
        for command in commands:
            observation[:, 90] = command
            action = model(observation)
            physical = action[0, :12] * scale[:12]
            assert torch.max(torch.abs(physical - previous_physical)) <= 0.030001
            assert torch.max(torch.abs(physical)) <= 0.180001
            assert torch.equal(action[:, 12:], torch.zeros_like(action[:, 12:]))
            observation[:, 73:89] = action
            previous_physical = physical


def test_continuous_policy_zero_handoff_and_command_cap() -> None:
    model = make_safety_model().eval()
    observation = torch.randn(1, 93)
    observation[:, 73:89] = 0.0
    observation[:, 89:92] = 0.0
    positive_limit = observation.clone()
    positive_limit[:, 90] = 0.060
    positive_large = observation.clone()
    positive_large[:, 90] = 0.300
    negative_limit = observation.clone()
    negative_limit[:, 90] = -0.060
    negative_large = observation.clone()
    negative_large[:, 90] = -0.300
    scripted = torch.jit.script(model)
    with torch.no_grad():
        zero = model(observation)
        scripted_zero = scripted(observation)
        positive = model(positive_limit)
        positive_clamped = model(positive_large)
        negative = model(negative_limit)
        negative_clamped = model(negative_large)
    torch.testing.assert_close(zero, torch.zeros_like(zero), atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        scripted_zero,
        torch.zeros_like(scripted_zero),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(positive, positive_clamped, atol=0.0, rtol=0.0)
    torch.testing.assert_close(negative, negative_clamped, atol=0.0, rtol=0.0)


def test_normalization_transfer_preserves_policy_function() -> None:
    old_mean = torch.linspace(-1.0, 1.0, 93)
    old_std = torch.linspace(0.5, 1.5, 93)
    source = StudentPolicy(
        hidden_dims=(32, 16),
        wheel_action_mode="hard_zero",
        observation_mean=old_mean,
        observation_std=old_std,
    ).eval()
    target = StudentPolicy(
        hidden_dims=(32, 16),
        wheel_action_mode="hard_zero",
        observation_mean=torch.linspace(0.25, -0.25, 93),
        observation_std=torch.linspace(1.7, 0.7, 93),
    ).eval()
    target.load_network_preserving_function(
        source.state_dict(),
        source.observation_mean,
        source.observation_std,
    )
    observations = torch.randn(19, 93)
    torch.testing.assert_close(
        target(observations),
        source(observations),
        rtol=1.0e-5,
        atol=2.0e-6,
    )


def test_temporal_loss_masks_invalid_labels() -> None:
    prediction = torch.zeros(2, 3, 16, requires_grad=True)
    target = torch.zeros(2, 3, 16)
    target[0, 1] = torch.nan
    valid = torch.ones(2, 3, dtype=torch.bool)
    loss = imitation_loss(prediction, target, valid, 0.5, 0.01, 0.002)
    assert torch.isfinite(loss.total)
    assert loss.valid_samples == 5
    loss.total.backward()


def test_fixed_batch_onnx_export_and_parity(tmp_path: Path) -> None:
    model = make_model().eval()
    checkpoint = {
        "format": "lateral_mppi_dagger_student_checkpoint_v1",
        "epoch": 0,
        "best_validation": 0.0,
        "model_spec": model.specification(),
        "model_state": model.state_dict(),
        "observation_mean": model.observation_mean,
        "observation_std": model.observation_std,
        "raw_min": model.raw_min,
        "raw_max": model.raw_max,
    }
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)
    observations = np.random.default_rng(5).normal(size=(7, 93)).astype(np.float32)
    contract = load_yaml("configs/deployment_contract.yaml")
    export_student_policy(checkpoint_path, tmp_path / "exported", contract, observations)
    metrics = validate_export_bundle(tmp_path / "exported")
    assert metrics["onnx_input_shape"] == [1, 93]
    assert metrics["onnx_output_shape"] == [1, 16]
    assert metrics["hard_zero_all_backends"] is True


def test_safety_policy_export_replays_exact_key7_ramp(tmp_path: Path) -> None:
    model = make_safety_model().eval()
    checkpoint = {
        "format": "lateral_mppi_dagger_student_checkpoint_v1",
        "epoch": 0,
        "best_validation": 0.0,
        "model_spec": model.specification(),
        "model_state": model.state_dict(),
        "observation_mean": model.observation_mean,
        "observation_std": model.observation_std,
        "raw_min": model.raw_min,
        "raw_max": model.raw_max,
        "action_scale": model.action_scale,
    }
    checkpoint_path = tmp_path / "safety_checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)
    observations = np.random.default_rng(6).normal(
        size=(7, 93)
    ).astype(np.float32)
    contract = load_yaml("configs/deployment_contract.yaml")
    export_student_policy(
        checkpoint_path,
        tmp_path / "safety_exported",
        contract,
        observations,
    )
    metrics = validate_export_bundle(tmp_path / "safety_exported")
    assert metrics["embedded_safety_gate_pass"] is True
    assert all(
        probe["zero_handoff_exact"]
        for probe in metrics["embedded_safety_probes"].values()
    )
