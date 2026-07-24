from __future__ import annotations

import numpy as np
import torch

from lateral_mppi_dagger.contract.action16 import Action16Adapter
from lateral_mppi_dagger.contract.obs93 import MotionPrefixSemantics, Obs93Builder, Obs93Input
from lateral_mppi_dagger.data.collector import EnvironmentStep
from lateral_mppi_dagger.expert.base import ExpertRequest
from lateral_mppi_dagger.reference.contact_schedule import infer_contact_schedule
from lateral_mppi_dagger.reference.loader import ReferenceSet

from .isaac_adapter import PlatformGeometry


class ReplayContractEnv:
    """Deterministic kinematic harness for contract/data/BC smoke tests only."""

    def __init__(
        self,
        references: ReferenceSet,
        contract: dict,
        action_adapter: Action16Adapter,
    ):
        self.references = references
        self.contract = contract
        self.action_adapter = action_adapter
        self.control_dt = 1.0 / references.fixed_motion.fps
        fixed = references.fixed_motion
        self.fixed_builder = Obs93Builder(
            torch.from_numpy(fixed.joint_pos[0]),
            torch.zeros(16),
            torch.from_numpy(fixed.body_quat_w[0, 0]),
            MotionPrefixSemantics.FIXED_FIRST_FRAME,
        )
        self.dynamic_builder = Obs93Builder(
            torch.from_numpy(fixed.joint_pos[0]),
            torch.zeros(16),
            torch.from_numpy(fixed.body_quat_w[0, 0]),
            MotionPrefixSemantics.DYNAMIC_REFERENCE,
        )
        self.default_q = np.asarray(contract["action"]["q_action_offset_runtime"], dtype=np.float32)
        self.default_dq = np.zeros(16, dtype=np.float32)
        self.contact_schedules = tuple(infer_contact_schedule(motion) for motion in references.motions)
        assets = contract["assets"]
        self.platform = PlatformGeometry(
            "CarTrunk",
            tuple(assets["trunk_position"]),
            tuple(assets["trunk_scale"]),
            assets["trunk_usd"]["path"],
        )
        self.ref_id = 0
        self.frame = 0
        self.q = self.default_q.copy()
        self.dq = self.default_dq.copy()
        self.previous_action = np.zeros(16, dtype=np.float32)

    def reset(self, seed: int, ref_id: int) -> tuple[np.ndarray, np.ndarray]:
        del seed
        self.ref_id = ref_id
        self.frame = 0
        self.q = self.references[ref_id].joint_pos[0].copy()
        self.dq = self.references[ref_id].joint_vel[0].copy()
        self.previous_action.fill(0.0)
        return self._observe(False), self._observe(True)

    def _values(self) -> Obs93Input:
        reference = self.references[self.ref_id]
        return Obs93Input(
            robot_anchor_quat_wxyz=torch.from_numpy(reference.body_quat_w[self.frame, 0]).unsqueeze(0),
            reference_anchor_quat_wxyz=torch.from_numpy(reference.body_quat_w[self.frame, 0]).unsqueeze(0),
            base_ang_vel_b=torch.from_numpy(reference.body_ang_vel_w[self.frame, 0]).unsqueeze(0),
            joint_pos=torch.from_numpy(self.q).unsqueeze(0),
            joint_vel=torch.from_numpy(self.dq).unsqueeze(0),
            default_joint_pos=torch.from_numpy(self.default_q).unsqueeze(0),
            default_joint_vel=torch.from_numpy(self.default_dq).unsqueeze(0),
            previous_executed_raw_action=torch.from_numpy(self.previous_action).unsqueeze(0),
            velocity_command=torch.tensor([[0.0, reference.target_vy, 0.0]], dtype=torch.float32),
            reference_joint_pos=torch.from_numpy(reference.joint_pos[self.frame]).unsqueeze(0),
            reference_joint_vel=torch.from_numpy(reference.joint_vel[self.frame]).unsqueeze(0),
        )

    def _observe(self, dynamic: bool) -> np.ndarray:
        builder = self.dynamic_builder if dynamic else self.fixed_builder
        return builder.build(self._values())[0].numpy().astype(np.float32, copy=True)

    def expert_request(self) -> ExpertRequest:
        reference = self.references[self.ref_id]
        body_pose = np.concatenate(
            (
                reference.body_pos_w[self.frame, 13:17],
                reference.body_quat_w[self.frame, 13:17],
            ),
            axis=-1,
        )
        body_twist = np.concatenate(
            (
                reference.body_lin_vel_w[self.frame, 13:17],
                reference.body_ang_vel_w[self.frame, 13:17],
            ),
            axis=-1,
        )
        desired = self.contact_schedules[self.ref_id][self.frame].astype(bool)
        forces = np.zeros((4, 3), dtype=np.float32)
        forces[:, 2] = desired.astype(np.float32) * 100.0
        ref_window = reference.frame(self.frame)
        end = min(self.frame + 20, reference.frames)
        ref_window["future_joint_pos"] = reference.joint_pos[self.frame:end]
        ref_window["future_joint_vel"] = reference.joint_vel[self.frame:end]
        return ExpertRequest(
            dt=self.control_dt,
            base_pose_w=np.concatenate(
                (reference.body_pos_w[self.frame, 0], reference.body_quat_w[self.frame, 0])
            ).astype(np.float32),
            base_twist_w=np.concatenate(
                (reference.body_lin_vel_w[self.frame, 0], reference.body_ang_vel_w[self.frame, 0])
            ).astype(np.float32),
            q=self.q.copy(),
            dq=self.dq.copy(),
            wheel_body_pose_w=body_pose.astype(np.float32),
            wheel_body_twist_w=body_twist.astype(np.float32),
            contact_force_w=forces,
            ref_id=self.ref_id,
            ref_frame=self.frame,
            ref_window=ref_window,
            target_vy=reference.target_vy,
            desired_contact=desired,
            platform_geometry=self.platform,
        )

    def step(self, executed_action16: np.ndarray) -> EnvironmentStep:
        old_q = self.q.copy()
        q_des, wheel_vel = self.action_adapter.raw_to_physical(executed_action16)
        self.q[:12] = q_des
        self.dq[:12] = (self.q[:12] - old_q[:12]) / self.control_dt
        self.dq[12:] = wheel_vel
        self.q[12:] = 0.0
        self.previous_action = np.asarray(executed_action16, dtype=np.float32).copy()
        self.frame = min(self.frame + 1, self.references[self.ref_id].frames - 1)
        truncated = self.frame >= self.references[self.ref_id].frames - 1
        return EnvironmentStep(
            next_obs93_clean=self._observe(False),
            next_obs93_dynamic=self._observe(True),
            applied_action16=np.asarray(executed_action16, dtype=np.float32).copy(),
            terminated=False,
            truncated=truncated,
            termination_reason=1 if truncated else 0,
            info={"harness": "offline_contract_smoke_only"},
        )
