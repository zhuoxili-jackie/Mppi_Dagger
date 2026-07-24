from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from lateral_mppi_dagger.contract.joint_mapping import POLICY_JOINT_ORDER
from lateral_mppi_dagger.contract.obs93 import MotionPrefixSemantics, Obs93Builder, Obs93Input
from lateral_mppi_dagger.config import resolve_project_path
from lateral_mppi_dagger.data.collector import EnvironmentStep
from lateral_mppi_dagger.expert.base import ExpertRequest
from lateral_mppi_dagger.reference.contact_schedule import infer_contact_schedule
from lateral_mppi_dagger.reference.interpolation import assert_compatible_timebase
from lateral_mppi_dagger.reference.loader import ReferenceSet
from lateral_mppi_dagger.env.scenarios import (
    ScenarioProfile,
    configure_env_for_scenario,
    load_scenario_profile,
)
from lateral_mppi_dagger.env.action_delay import advance_action_delay


WHEEL_BODY_NAMES = ("FL_foot_link", "FR_foot_link", "RL_foot_link", "RR_foot_link")


def deployment_lateral_command_ramp_value(
    goal_m_s: float,
    frame: int,
    increment_per_step_m_s: float,
) -> float:
    if frame < 0 or increment_per_step_m_s <= 0.0:
        raise ValueError("frame must be non-negative and increment must be positive.")
    maximum_magnitude = increment_per_step_m_s * frame
    return float(
        np.clip(goal_m_s, -maximum_magnitude, maximum_magnitude)
    )


@dataclass(frozen=True)
class PlatformGeometry:
    asset_name: str
    position_w: tuple[float, float, float]
    scale: tuple[float, float, float]
    usd_path: str


def configure_nominal_env(env_cfg: Any, num_envs: int = 1) -> None:
    """Disable stochastic terms for contract/reference/expert gates."""
    configure_env_for_scenario(
        env_cfg,
        load_scenario_profile("nominal"),
        num_envs,
    )


class IsaacLateralAdapter:
    """Isaac bridge exposing environment zero through the fixed-batch company ABI.

    Additional environments may be present as MPPI rollout clones.  Collection,
    observations, terminal flags, and episode accounting always use env zero.
    """

    def __init__(
        self,
        gym_env: Any,
        references: ReferenceSet,
        contract: dict[str, Any],
        scenario_profile: ScenarioProfile | None = None,
    ):
        self.env = gym_env
        self.base = gym_env.unwrapped
        self.references = references
        self.contract = contract
        self.scenario_profile = scenario_profile or load_scenario_profile("nominal")
        self.num_envs = int(self.base.num_envs)
        if self.num_envs < 1:
            raise ValueError("IsaacLateralAdapter requires at least one environment.")
        self.control_dt = float(self.base.step_dt)
        assert_compatible_timebase(self.control_dt, references.fixed_motion.fps)
        self.robot = self.base.scene["robot"]
        self.command = self.base.command_manager.get_term("motion")
        self.joint_ids, resolved_joint_names = self.robot.find_joints(
            list(POLICY_JOINT_ORDER), preserve_order=True
        )
        if tuple(resolved_joint_names) != POLICY_JOINT_ORDER:
            raise RuntimeError(
                f"Isaac joint mapping mismatch: expected {POLICY_JOINT_ORDER}, got {tuple(resolved_joint_names)}"
            )
        self.wheel_body_ids, resolved_body_names = self.robot.find_bodies(
            list(WHEEL_BODY_NAMES), preserve_order=True
        )
        if tuple(resolved_body_names) != WHEEL_BODY_NAMES:
            raise RuntimeError(f"Isaac wheel-body mapping mismatch: {resolved_body_names}")
        self.contact_sensor = self.base.scene["contact_forces"]
        self.contact_body_ids, contact_names = self.contact_sensor.find_bodies(
            list(WHEEL_BODY_NAMES), preserve_order=True
        )
        if tuple(contact_names) != WHEEL_BODY_NAMES:
            raise RuntimeError(f"Contact sensor wheel-body mapping mismatch: {contact_names}")

        fixed = references.fixed_motion
        device = self.base.device
        self.fixed_builder = Obs93Builder(
            torch.as_tensor(fixed.joint_pos[0], device=device),
            torch.zeros(16, dtype=torch.float32, device=device),
            torch.as_tensor(fixed.body_quat_w[0, 0], device=device),
            MotionPrefixSemantics.FIXED_FIRST_FRAME,
        )
        self.dynamic_builder = Obs93Builder(
            torch.as_tensor(fixed.joint_pos[0], device=device),
            torch.zeros(16, dtype=torch.float32, device=device),
            torch.as_tensor(fixed.body_quat_w[0, 0], device=device),
            MotionPrefixSemantics.DYNAMIC_REFERENCE,
        )
        contact_inference_kwargs = references.contact_inference_kwargs()
        self.contact_schedules = tuple(
            infer_contact_schedule(motion, **contact_inference_kwargs)
            for motion in references.motions
        )
        runtime_command = contract["runtime_lateral_command"]
        self._lateral_command_increment_m_s = (
            float(runtime_command["acceleration_max_m_s2"])
            * self.control_dt
        )
        self._lateral_command_goal_m_s = 0.0
        self.previous_executed_action = torch.zeros(
            (self.num_envs, 16), dtype=torch.float32, device=device
        )
        self.previous_commanded_action = torch.zeros(
            (self.num_envs, 16), dtype=torch.float32, device=device
        )
        self.action_delay_steps = self.scenario_profile.action_delay_steps
        self.action_delay_queue = torch.zeros(
            (self.action_delay_steps, self.num_envs, 16),
            dtype=torch.float32,
            device=device,
        )
        assets = contract["assets"]
        self._nominal_platform_position = tuple(
            float(value) for value in assets["trunk_position"]
        )
        self._platform_jitter = (0.0, 0.0, 0.0)
        self.platform_geometry = PlatformGeometry(
            asset_name="CarTrunk",
            position_w=self._nominal_platform_position,
            scale=tuple(float(value) for value in assets["trunk_scale"]),
            usd_path=str(resolve_project_path(assets["trunk_usd"]["path"])),
        )
        self.platform = self.base.scene["box"]
        action_dim = int(self.base.action_manager.total_action_dim)
        if action_dim != 16:
            raise RuntimeError(f"Expected Isaac action dimension 16, got {action_dim}")
        expected_default = torch.as_tensor(
            contract["action"]["q_action_offset_runtime"],
            dtype=torch.float32,
            device=device,
        )
        actual_default = self.robot.data.default_joint_pos[:, self.joint_ids]
        max_default_error = float(torch.max(torch.abs(actual_default - expected_default)).item())
        if max_default_error > 1.0e-5:
            raise RuntimeError(
                "q_action_offset_runtime differs from Isaac default_joint_pos after nominal configuration: "
                f"max_abs={max_default_error:.6g}"
            )

    def reset(self, seed: int, ref_id: int) -> tuple[np.ndarray, np.ndarray]:
        if not 0 <= ref_id < len(self.references):
            raise ValueError(f"ref_id {ref_id} is outside [0,{len(self.references)})")
        torch.manual_seed(seed)
        self.env.reset(seed=seed)

        # LateralReferenceMotionCommand samples its motion id during env.reset
        # and immediately writes that sampled reference state to the robot.
        # Merely replacing motion_ids afterwards changes the future command
        # while leaving q/dq/root velocity from a different (or standing)
        # reference in the simulator.  That made a nominal fixed-seed gate
        # depend on an unrelated hidden reset sample and produced apparent
        # cross-episode degradation.  Re-run only MotionCommand's common
        # state-reset implementation after pinning the requested reference;
        # this preserves the configured pose/velocity/joint perturbations
        # without invoking the lateral selector a second time.
        from robot_lab.tasks.manager_based.beyondmimic.mdp.commands import (
            MotionCommand,
        )

        self.command.motion_ids[:] = ref_id
        self.command._target_lateral_velocities[:] = self.references[ref_id].target_vy
        self.command.time_steps.zero_()
        if not isinstance(self.command, MotionCommand):
            raise TypeError(
                "Expected the lateral command to derive from MotionCommand, got "
                f"{type(self.command)!r}."
            )
        MotionCommand._resample_command(
            self.command,
            torch.arange(
                self.num_envs,
                dtype=torch.long,
                device=self.base.device,
            ),
        )
        if not bool(torch.all(self.command.motion_ids == ref_id)):
            raise RuntimeError("Pinned lateral reference changed during deterministic reset.")
        if not bool(torch.all(self.command.time_steps == 0)):
            raise RuntimeError("Lateral command did not reset at reference frame zero.")
        self._lateral_command_goal_m_s = float(
            self.references[ref_id].target_vy
        )
        self.command._target_lateral_velocities.zero_()
        self.base.scene.write_data_to_sim()
        self.base.sim.forward()
        self.base.scene.update(dt=0.0)
        self.previous_executed_action.zero_()
        self.previous_commanded_action.zero_()
        self.action_delay_queue.zero_()
        self._apply_platform_jitter(seed)
        return self.observe(dynamic=False), self.observe(dynamic=True)

    def _apply_deployment_lateral_command_ramp(self) -> None:
        """Mirror key7's 0.6 m/s² reset-to-command acceleration ramp."""

        frame = int(self.command.time_steps[0].item())
        value = deployment_lateral_command_ramp_value(
            self._lateral_command_goal_m_s,
            frame,
            self._lateral_command_increment_m_s,
        )
        self.command._target_lateral_velocities[:] = value

    def _apply_platform_jitter(self, seed: int) -> None:
        maximum = np.asarray(
            self.scenario_profile.platform_position_jitter_m,
            dtype=np.float32,
        )
        if np.any(maximum < 0.0):
            raise ValueError("Platform jitter limits must be non-negative.")
        rng = np.random.default_rng(int(seed) + 0x708)
        jitter = rng.uniform(-maximum, maximum).astype(np.float32)
        self._platform_jitter = tuple(float(value) for value in jitter)
        nominal = torch.as_tensor(
            self._nominal_platform_position,
            dtype=torch.float32,
            device=self.base.device,
        )
        offset = torch.as_tensor(jitter, dtype=torch.float32, device=self.base.device)
        root_pose = self.platform.data.root_pose_w.clone()
        root_pose[:, :3] = self.base.scene.env_origins + nominal + offset
        root_velocity = torch.zeros_like(self.platform.data.root_vel_w)
        self.platform.write_root_pose_to_sim(root_pose)
        self.platform.write_root_velocity_to_sim(root_velocity)
        self.base.scene.write_data_to_sim()
        self.base.sim.forward()
        self.base.scene.update(dt=0.0)
        actual_position = tuple(
            float(value) for value in (nominal + offset).detach().cpu().tolist()
        )
        self.platform_geometry = PlatformGeometry(
            asset_name=self.platform_geometry.asset_name,
            position_w=actual_position,
            scale=self.platform_geometry.scale,
            usd_path=self.platform_geometry.usd_path,
        )

    def episode_metadata(self) -> dict[str, Any]:
        ref_id = int(self.command.motion_ids[0].item())
        ref_frame = min(
            int(self.command.time_steps[0].item()),
            self.references[ref_id].frames - 1,
        )
        reference = self.references[ref_id]
        reference_q = torch.as_tensor(
            reference.joint_pos[ref_frame],
            dtype=torch.float32,
            device=self.base.device,
        )
        reference_dq = torch.as_tensor(
            reference.joint_vel[ref_frame],
            dtype=torch.float32,
            device=self.base.device,
        )
        actual_q = self.robot.data.joint_pos[0, self.joint_ids]
        actual_dq = self.robot.data.joint_vel[0, self.joint_ids]
        anchor_position_local = (
            self.command.robot_anchor_pos_w[0] - self.base.scene.env_origins[0]
        )
        reference_anchor_position_local = torch.as_tensor(
            reference.body_pos_w[
                ref_frame,
                self.command.motion_anchor_body_index,
            ],
            dtype=torch.float32,
            device=self.base.device,
        )
        actual_anchor_quat = self.command.robot_anchor_quat_w[0]
        reference_anchor_quat = torch.as_tensor(
            reference.body_quat_w[
                ref_frame,
                self.command.motion_anchor_body_index,
            ],
            dtype=torch.float32,
            device=self.base.device,
        )
        orientation_error = 2.0 * torch.acos(
            torch.abs(torch.dot(actual_anchor_quat, reference_anchor_quat)).clamp(
                max=1.0
            )
        )
        actual_anchor_twist = torch.cat(
            (
                self.command.robot_anchor_lin_vel_w[0],
                self.command.robot_anchor_ang_vel_w[0],
            )
        )
        reference_anchor_twist = torch.as_tensor(
            np.concatenate(
                (
                    reference.body_lin_vel_w[
                        ref_frame,
                        self.command.motion_anchor_body_index,
                    ],
                    reference.body_ang_vel_w[
                        ref_frame,
                        self.command.motion_anchor_body_index,
                    ],
                )
            ),
            dtype=torch.float32,
            device=self.base.device,
        )
        return {
            "action_delay_steps": self.action_delay_steps,
            "lateral_command_profile": {
                "semantics": "deployment_acceleration_ramp_from_zero",
                "goal_m_s": self._lateral_command_goal_m_s,
                "increment_per_control_step_m_s": (
                    self._lateral_command_increment_m_s
                ),
                "frame0_m_s": 0.0,
            },
            "platform_position_jitter_m_sampled": list(self._platform_jitter),
            "platform_position_local_m": list(self.platform_geometry.position_w),
            "reset_state": {
                "reference_motion_id": ref_id,
                "reference_frame": ref_frame,
                "anchor_position_local_m": anchor_position_local.detach().cpu().tolist(),
                "anchor_position_minus_reference_m": (
                    anchor_position_local - reference_anchor_position_local
                )
                .detach()
                .cpu()
                .tolist(),
                "anchor_quaternion_wxyz": actual_anchor_quat.detach().cpu().tolist(),
                "anchor_orientation_error_rad": float(orientation_error.item()),
                "anchor_twist_w": actual_anchor_twist.detach().cpu().tolist(),
                "anchor_twist_minus_reference": (
                    actual_anchor_twist - reference_anchor_twist
                )
                .detach()
                .cpu()
                .tolist(),
                "joint_position": actual_q.detach().cpu().tolist(),
                "joint_position_minus_reference": (actual_q - reference_q)
                .detach()
                .cpu()
                .tolist(),
                "joint_velocity_minus_reference": (actual_dq - reference_dq)
                .detach()
                .cpu()
                .tolist(),
            },
        }

    def _observation_values(self) -> Obs93Input:
        self._apply_deployment_lateral_command_ramp()
        ref_id = int(self.command.motion_ids[0].item())
        frame = int(self.command.time_steps[0].item())
        reference = self.references[ref_id]
        frame = min(frame, reference.frames - 1)
        device = self.base.device
        env0 = slice(0, 1)
        q = self.robot.data.joint_pos[env0, self.joint_ids]
        dq = self.robot.data.joint_vel[env0, self.joint_ids]
        reference_joint_pos = torch.as_tensor(reference.joint_pos[frame], device=device).unsqueeze(0)
        reference_joint_vel = torch.as_tensor(reference.joint_vel[frame], device=device).unsqueeze(0)
        reference_quat = torch.as_tensor(reference.body_quat_w[frame, 0], device=device).unsqueeze(0)
        velocity_command = torch.zeros((1, 3), dtype=torch.float32, device=device)
        velocity_command[:, 1] = float(self.command._target_lateral_velocities[0].item())
        return Obs93Input(
            robot_anchor_quat_wxyz=self.command.robot_anchor_quat_w[env0],
            reference_anchor_quat_wxyz=reference_quat,
            base_ang_vel_b=self.robot.data.root_ang_vel_b[env0],
            joint_pos=q,
            joint_vel=dq,
            default_joint_pos=self.robot.data.default_joint_pos[env0, self.joint_ids],
            default_joint_vel=self.robot.data.default_joint_vel[env0, self.joint_ids],
            previous_executed_raw_action=self.previous_executed_action[env0],
            velocity_command=velocity_command,
            reference_joint_pos=reference_joint_pos,
            reference_joint_vel=reference_joint_vel,
        )

    def observe(self, dynamic: bool) -> np.ndarray:
        builder = self.dynamic_builder if dynamic else self.fixed_builder
        observation = builder.build(self._observation_values())
        return observation[0].detach().cpu().numpy().astype(np.float32, copy=True)

    def expert_request(self) -> ExpertRequest:
        self._apply_deployment_lateral_command_ramp()
        ref_id = int(self.command.motion_ids[0].item())
        ref_frame = min(
            int(self.command.time_steps[0].item()),
            self.references[ref_id].frames - 1,
        )
        reference = self.references[ref_id]
        ref_window = reference.frame(ref_frame)
        end = min(ref_frame + 20, reference.frames)
        ref_window["future_joint_pos"] = reference.joint_pos[ref_frame:end]
        ref_window["future_joint_vel"] = reference.joint_vel[ref_frame:end]

        body_pos = self.robot.data.body_pos_w[0, self.wheel_body_ids]
        body_quat = self.robot.data.body_quat_w[0, self.wheel_body_ids]
        body_lin = self.robot.data.body_lin_vel_w[0, self.wheel_body_ids]
        body_ang = self.robot.data.body_ang_vel_w[0, self.wheel_body_ids]
        wheel_pose = torch.cat((body_pos, body_quat), dim=-1)
        wheel_twist = torch.cat((body_lin, body_ang), dim=-1)
        contact_force = self.contact_sensor.data.net_forces_w[0, self.contact_body_ids]
        base_pose = torch.cat(
            (self.command.robot_anchor_pos_w[0], self.command.robot_anchor_quat_w[0]), dim=-1
        )
        base_twist = torch.cat(
            (self.command.robot_anchor_lin_vel_w[0], self.command.robot_anchor_ang_vel_w[0]), dim=-1
        )
        return ExpertRequest(
            dt=self.control_dt,
            base_pose_w=base_pose.detach().cpu().numpy().astype(np.float32),
            base_twist_w=base_twist.detach().cpu().numpy().astype(np.float32),
            q=self.robot.data.joint_pos[0, self.joint_ids].detach().cpu().numpy().astype(np.float32),
            dq=self.robot.data.joint_vel[0, self.joint_ids].detach().cpu().numpy().astype(np.float32),
            wheel_body_pose_w=wheel_pose.detach().cpu().numpy().astype(np.float32),
            wheel_body_twist_w=wheel_twist.detach().cpu().numpy().astype(np.float32),
            contact_force_w=contact_force.detach().cpu().numpy().astype(np.float32),
            ref_id=ref_id,
            ref_frame=ref_frame,
            ref_window=ref_window,
            target_vy=float(self.command._target_lateral_velocities[0].item()),
            desired_contact=self.contact_schedules[ref_id][ref_frame].astype(bool),
            platform_geometry=self.platform_geometry,
        )

    def step(self, executed_action16: np.ndarray) -> EnvironmentStep:
        # ManagerBasedRLEnv auto-resets done environments inside env.step().
        # Preserve the last valid transition-side observations before that
        # reset so a newly sampled standing command (internal motion id 6)
        # cannot be mistaken for a different ReferenceSet entry.
        terminal_fallback_clean = self.observe(dynamic=False)
        terminal_fallback_dynamic = self.observe(dynamic=True)
        commanded_action = torch.as_tensor(
            executed_action16,
            dtype=torch.float32,
            device=self.base.device,
        ).reshape(1, 16)
        commanded_batch = commanded_action.expand(self.num_envs, -1).contiguous()
        applied_batch, updated_queue = advance_action_delay(
            commanded_batch,
            self.action_delay_queue,
        )
        self.action_delay_queue.copy_(updated_queue)
        self.previous_commanded_action.copy_(commanded_batch)
        result = self.env.step(applied_batch)
        if len(result) != 5:
            raise RuntimeError(f"Expected Gymnasium five-value step result, got {len(result)}")
        _, _, terminated, truncated, extras = result
        is_terminated = bool(torch.as_tensor(terminated).reshape(-1)[0].item())
        is_truncated = bool(torch.as_tensor(truncated).reshape(-1)[0].item())
        if is_terminated or is_truncated:
            self.previous_executed_action.zero_()
            self.previous_commanded_action.zero_()
            self.action_delay_queue.zero_()
            next_clean = terminal_fallback_clean
            next_dynamic = terminal_fallback_dynamic
        else:
            self.previous_executed_action.copy_(applied_batch)
            next_clean = self.observe(dynamic=False)
            next_dynamic = self.observe(dynamic=True)
        info = dict(extras) if isinstance(extras, dict) else {"extras": extras}
        if is_terminated or is_truncated:
            info["terminal_next_observation_semantics"] = (
                "last_pre_step_observation_placeholder_due_to_isaac_auto_reset"
            )
        return EnvironmentStep(
            next_obs93_clean=next_clean,
            next_obs93_dynamic=next_dynamic,
            applied_action16=applied_batch[0].detach().cpu().numpy().astype(
                np.float32,
                copy=True,
            ),
            terminated=is_terminated,
            truncated=is_truncated,
            termination_reason=2 if is_terminated else (1 if is_truncated else 0),
            info=info,
        )

    def close(self) -> None:
        self.env.close()
