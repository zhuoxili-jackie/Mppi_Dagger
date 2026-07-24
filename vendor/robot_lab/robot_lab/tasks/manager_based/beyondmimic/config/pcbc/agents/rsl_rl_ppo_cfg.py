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
# stage2 lateral
@configclass
class PcbCStage2LateralGuidedPPORunnerCfg(PcbCBeyondMimicFlatPPORunnerCfg):
    """Conservative PPO runner for stage2 lateral-reference fine-tuning."""

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "pcbc_v1_command_stage"
        # Stage2 fine-tuning starts from a good baseline. Keep exploration tiny and
        # clip raw actions before they enter both the simulator and last_action obs.
        self.clip_actions = 1.0
        self.policy.init_noise_std = 0.03

        self.algorithm.learning_rate = 5.0e-6
        self.algorithm.entropy_coef = 0.0
        self.algorithm.max_grad_norm = 0.05
        self.algorithm.desired_kl = 5.0e-4
        self.algorithm.value_loss_coef = 0.03
        self.algorithm.use_clipped_value_loss = True
        self.algorithm.num_learning_epochs = 2
        self.algorithm.num_mini_batches = 8



@configclass
class PcbCResidualMovePPORunnerCfg(PcbCBeyondMimicFlatPPORunnerCfg):
    """PPO config for baseline-frozen residual movement on the platform."""

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "pcbc_v1_residual_move"
        self.max_iterations = 20000
        self.save_interval = 500

        # The frozen baseline already supplies a valid posture and motion prior.
        # Unit-scale residual noise destroys that posture as soon as ready opens,
        # so start with moderate residual exploration instead.
        self.policy.init_noise_std = 0.7
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]

        self.algorithm.learning_rate = 7.5e-5
        # Movement has already emerged; retain moderate exploration while allowing
        # the policy to polish a stable, repeatable residual correction.
        self.algorithm.entropy_coef = 2.0e-3
        self.algorithm.max_grad_norm = 0.25
        self.algorithm.desired_kl = 0.008
        self.algorithm.value_loss_coef = 0.35


@configclass
class PcbCLateralGuidedPPORunnerCfg(PcbCBeyondMimicFlatPPORunnerCfg):
    """From-scratch PPO runner for the data-guided y-lateral experiment."""

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "pcbc_v1_lateral_guided"
        self.max_iterations = 20000
        self.save_interval = 1000
        #1
        self.policy.init_noise_std = 1.0
        self.algorithm.learning_rate = 2.5e-4
        self.algorithm.entropy_coef = 6.0e-3
        self.algorithm.max_grad_norm = 0.4
        self.algorithm.desired_kl = 0.008


@configclass
class PcbCLateralGuidedFanqu713PPORunnerCfg(PcbCLateralGuidedPPORunnerCfg):
    """Lower-noise PPO for the short periodic fanqu_713 gait."""

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "pcbc_v1_lateral_guided_fanqu713"
        self.policy.init_noise_std = 0.80
        self.algorithm.entropy_coef = 3.0e-3


@configclass
class PcbCLateralRedAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """PPO with a simultaneously trained differential Random Expert Distillation prior."""

    class_name: str = "robot_lab.tasks.manager_based.beyondmimic.learning.ppo_red:PPORED"
    red_cfg: dict = {
        "policy_group": "red_policy",
        "demo_group": "red_demo",
        "reward_weight": 1.2,
        "reward_warmup_steps": 2000,
        "learning_rate": 1.0e-3,
        "num_learning_epochs": 2,
        "num_mini_batches": 4,
        "num_outputs": 128,
        "predictor_hidden_dims": [768, 384],
        "target_hidden_dims": [768, 384],
        "activation": "elu",
        "state_normalization": True,
        "reward_temperature": 0.25,
    }


@configclass
class PcbCLateralRedPPORunnerCfg(PcbCBeyondMimicFlatPPORunnerCfg):
    """PPO runner for differential RED-guided lateral locomotion."""

    class_name: str = "REDOnPolicyRunner"
    obs_groups: dict[str, list[str]] = {
        "policy": ["policy"],
        "critic": ["critic"],
        "red_policy": ["red_policy"],
        "red_demo": ["red_demo"],
    }

    algorithm = PcbCLateralRedAlgorithmCfg(
        value_loss_coef=0.4,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=2.5e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=0.4,
    )

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "pcbc_v1_lateral_red"
        self.max_iterations = 20000
        self.save_interval = 500
        self.policy.init_noise_std = 1.0


@configclass
class PcbCLateralAddAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """PPO with an ADD-style adversarial differential discriminator."""

    class_name: str = "robot_lab.tasks.manager_based.beyondmimic.learning.ppo_add:PPOADD"
    add_cfg: dict = {
        "policy_group": "add_policy",
        "reward_weight": 25.0,
        "reward_warmup_steps": 0,
        "reward_scale": 1.0,
        "learning_rate": 2.5e-4,
        "num_learning_epochs": 2,
        "num_mini_batches": 16,
        "hidden_dims": [1024, 512],
        "activation": "elu",
        "state_clip": 10.0,
        "diff_min_scale": 1.0e-4,
        "diff_momentum": 0.01,
        "replay_capacity": 200000,
        "replay_samples": 8192,
        "grad_penalty": 2.0,
        "logit_reg": 1.0e-2,
        "positive_noise_std": 0.0,
    }


@configclass
class PcbCLateralAddPPORunnerCfg(PcbCBeyondMimicFlatPPORunnerCfg):
    """PPO runner for ADD-guided lateral locomotion."""

    class_name: str = "ADDOnPolicyRunner"
    obs_groups: dict[str, list[str]] = {
        "policy": ["policy"],
        "critic": ["critic"],
        "add_policy": ["add_policy"],
    }

    algorithm = PcbCLateralAddAlgorithmCfg(
        value_loss_coef=0.4,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=2.5e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=0.4,
    )

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "pcbc_v1_lateral_add"
        self.max_iterations = 20000
        self.save_interval = 500
        self.policy.init_noise_std = 1.0


@configclass
class PcbCStageRedAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """PPO with reference-aligned differential RED for stage boarding."""

    class_name: str = "robot_lab.tasks.manager_based.beyondmimic.learning.ppo_red:PPORED"
    red_cfg: dict = {
        "policy_group": "red_policy",
        "demo_group": "red_demo",
        "reward_weight": 25.0,
        "reward_warmup_steps": 0,
        "learning_rate": 1.0e-3,
        "num_learning_epochs": 2,
        "num_mini_batches": 4,
        "num_outputs": 128,
        "predictor_hidden_dims": [1024, 512],
        "target_hidden_dims": [1024, 512],
        "activation": "elu",
        "state_normalization": True,
        "normalization_type": "differential",
        "normalization_source": "policy",
        "state_clip": 10.0,
        "diff_min_scale": 1.0e-4,
        "diff_momentum": 0.01,
        "reward_temperature": 0.3,
        "use_delta_kernel": True,
        "delta_kernel_temperature": 1.0,
        "delta_kernel_use_raw_states": True,
        "contrastive_margin": 0.25,
        "contrastive_coef": 1.0,
        "contrastive_min_delta_rms": 0.05,
    }


@configclass
class PcbCStageRedPPORunnerCfg(PcbCBeyondMimicFlatPPORunnerCfg):
    """PPO runner for 531 stage boarding with RED as the core imitation scorer."""

    class_name: str = "REDOnPolicyRunner"
    obs_groups: dict[str, list[str]] = {
        "policy": ["policy"],
        "critic": ["critic"],
        "red_policy": ["red_policy"],
        "red_demo": ["red_demo"],
    }

    algorithm = PcbCStageRedAlgorithmCfg(
        value_loss_coef=0.4,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=2.5e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=0.4,
    )

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "pcbc_v1_stage_red"
        self.max_iterations = 30000
        self.save_interval = 500
        self.policy.init_noise_std = 1.0
