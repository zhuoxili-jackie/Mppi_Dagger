# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PcbABeyondMimicFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 30000
    # clip_actions = 1.0
    save_interval = 500
    experiment_name = "pcba_beyondmimic_flat"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=0.5,
    )



# @configclass
# class PcbABeyondMimicFlatCommandPPORunnerCfg(PcbABeyondMimicFlatPPORunnerCfg):
#     """Task-local stable PPO config for the command-finetune variant.

#     This config is intentionally scoped to the command task only to avoid affecting
#     other BeyondMimic tasks.
#     """

#     def __post_init__(self):
#         super().__post_init__()
#         self.experiment_name = "pcba_v1_command"

        # # Keep scalar std format for checkpoint compatibility with existing runs.
        # self.policy.init_noise_std = 0.3

        # # Slightly more conservative optimizer settings for finetuning stability.
        # self.algorithm.learning_rate = 1.5e-4
        # self.algorithm.entropy_coef = 0.003
        # self.algorithm.max_grad_norm = 0.3
        # self.algorithm.desired_kl = 0.008
# @configclass
# class PcbABeyondMimicFlatCommandPPORunnerCfg(PcbABeyondMimicFlatPPORunnerCfg):
#     def __post_init__(self):
#         super().__post_init__()

#         self.experiment_name = "pcba_v1_command"

#         self.policy.init_noise_std = 0.3

#         self.algorithm.learning_rate = 1.5e-4
#         self.algorithm.entropy_coef = 0.003
#         self.algorithm.max_grad_norm = 0.3
#         self.algorithm.desired_kl = 0.008

#         self.algorithm.value_loss_coef = 0.5
#         self.algorithm.use_clipped_value_loss = True


# @configclass
# class PcbABeyondMimicFlatCommandPPORunnerCfg(PcbABeyondMimicFlatPPORunnerCfg):
#     """Task-local PPO config for command Stage C1."""

#     def __post_init__(self):
#         super().__post_init__()

#         self.experiment_name = "pcba_v1_command"

#         # Fine-tuning from Stage B, so exploration should be moderate.
#         self.policy.init_noise_std = 0.3

#         # Conservative but not too slow.
#         self.algorithm.learning_rate = 1.5e-4
#         self.algorithm.entropy_coef = 0.003
#         self.algorithm.max_grad_norm = 0.3
#         self.algorithm.desired_kl = 0.008

#         # Keep value learning normal; your Stage B is stable now.
#         self.algorithm.value_loss_coef = 0.5
#         self.algorithm.use_clipped_value_loss = True

#         self.algorithm.num_learning_epochs = 4
#         self.algorithm.num_mini_batches = 4

# @configclass
# class PcbABeyondMimicFlatCommandPPORunnerCfg(PcbABeyondMimicFlatPPORunnerCfg):
#     """Task-local PPO config for command Stage C2."""

#     def __post_init__(self):
#         super().__post_init__()

#         self.experiment_name = "pcba_v1_command"

#         # Continue from C1, moderate exploration.
#         self.policy.init_noise_std = 0.25

#         # Slightly conservative because C2 reward shift is significant.
#         self.algorithm.learning_rate = 1.0e-4
#         self.algorithm.entropy_coef = 0.002
#         self.algorithm.max_grad_norm = 0.25
#         self.algorithm.desired_kl = 0.006

#         self.algorithm.value_loss_coef = 0.35
#         self.algorithm.use_clipped_value_loss = True

#         self.algorithm.num_learning_epochs = 4
#         self.algorithm.num_mini_batches = 4
# @configclass
# class PcbABeyondMimicFlatCommandPPORunnerCfg(PcbABeyondMimicFlatPPORunnerCfg):
#     """Task-local PPO config for command Stage C3."""

#     def __post_init__(self):
#         super().__post_init__()

#         self.experiment_name = "pcba_v1_command_stage"

#         self.policy.init_noise_std = 0.25

#         self.algorithm.learning_rate = 1.0e-4
#         self.algorithm.entropy_coef = 0.002
#         self.algorithm.max_grad_norm = 0.25
#         self.algorithm.desired_kl = 0.006

#         self.algorithm.value_loss_coef = 0.35
#         self.algorithm.use_clipped_value_loss = True

#         self.algorithm.num_learning_epochs = 4
#         self.algorithm.num_mini_batches = 4

@configclass
class PcbABeyondMimicFlatCommandPPORunnerCfg(PcbABeyondMimicFlatPPORunnerCfg):
    """Task-local PPO config for command stage2 stable finetuning."""

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "pcba_v1_command_stage"
        self.policy.init_noise_std = 0.20

        # Conservative optimizer settings for stable stage2 finetuning.
        self.algorithm.learning_rate = 5.0e-5
        self.algorithm.entropy_coef = 0.002
        self.algorithm.max_grad_norm = 0.20
        self.algorithm.desired_kl = 0.004

        self.algorithm.value_loss_coef = 0.35
        self.algorithm.use_clipped_value_loss = True

        self.algorithm.num_learning_epochs = 4
        self.algorithm.num_mini_batches = 4
