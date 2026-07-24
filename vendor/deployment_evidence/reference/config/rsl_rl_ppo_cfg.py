# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

#baseline改了域值可以训满
@configclass
class PcbCBeyondMimicFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """From-scratch pure mimic baseline runner."""

    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 1000
    experiment_name = "pcbc_v1_command_stage"

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.4,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.004,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=2.5e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=0.4,
    )

@configclass
class PcbCBeyondMimicFlatCommandPPORunnerCfg(PcbCBeyondMimicFlatPPORunnerCfg):
    """Stage2 hold-warmup / command fine-tuning runner.

    用途：
    - 从 baseline model_12000 接。
    - 先稳住 stage1 hold，再慢慢进入 stage2 小速度。
    - 必须配合 --load_model_only。
    """

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "pcbc_v1_command_stage"

        # 这个值只影响新建策略时的 init std。
        # 从 checkpoint load_model_only 时，旧模型里的 std 也会被加载进来，
        # 所以后面主要靠低 entropy/lr 慢慢收敛。
        self.policy.init_noise_std = 0.20

        # 比之前更保守：
        # 目标是保护 model_12000 已学会的上台动作，不让 stage1/stage2 把策略洗坏。
        self.algorithm.learning_rate = 3.0e-5
        self.algorithm.entropy_coef = 8.0e-4
        self.algorithm.max_grad_norm = 0.15
        self.algorithm.desired_kl = 0.0025

        self.algorithm.value_loss_coef = 0.30
        self.algorithm.use_clipped_value_loss = True

        self.algorithm.num_learning_epochs = 4
        self.algorithm.num_mini_batches = 4


@configclass
class PcbCBeyondMimicFlatHoldPolishPPORunnerCfg(PcbCBeyondMimicFlatCommandPPORunnerCfg):
    """兼容旧注册名。

    现在不建议再单独跑 PCBC_COMMAND_PHASE=hold。
    如果你暂时有旧注册依赖这个类，它会等价使用更保守的 command fine-tune runner。
    """

    pass


@configclass
class PcbCResidualMovePPORunnerCfg(PcbCBeyondMimicFlatPPORunnerCfg):
    """PPO config for baseline-frozen residual movement on the platform."""

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "pcbc_v1_residual_move"
        self.max_iterations = 20000
        self.save_interval = 500

        self.policy.init_noise_std = 0.55
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]

        self.algorithm.learning_rate = 7.5e-5
        self.algorithm.entropy_coef = 1.0e-3
        self.algorithm.max_grad_norm = 0.25
        self.algorithm.desired_kl = 0.008
        self.algorithm.value_loss_coef = 0.35
