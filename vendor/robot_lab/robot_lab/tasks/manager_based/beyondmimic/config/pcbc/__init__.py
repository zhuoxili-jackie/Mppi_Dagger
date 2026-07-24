# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym

from . import (
    agents,
    flat_env_cfg,
    lateral_add_env_cfg,
    lateral_bipedal_stand,
    lateral_command_curriculum,
    lateral_guided_env_cfg,
    lateral_red_env_cfg,
    pure_imitation,
    stage_red_env_cfg,
)


gym.register(
    id="RobotLab-Isaac-BeyondMimic-Flat-pcbC-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{pure_imitation.__name__}:PcbCBeyondMimicFlatV1StandEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PcbCBeyondMimicFlatPPORunnerCfg",
    },
)


gym.register(
    id="RobotLab-Isaac-BeyondMimic-Flat-pcbC-v1-stand-command-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{pure_imitation.__name__}:PcbCBeyondMimicFlatV1StandCommandEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PcbCBeyondMimicFlatPPORunnerCfg",
    },
)


gym.register(
    id="RobotLab-Isaac-BeyondMimic-Flat-pcbC-v1-lateral-guided-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{lateral_guided_env_cfg.__name__}:PcbCLateralGuidedEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PcbCLateralGuidedPPORunnerCfg",
    },
)


gym.register(
    id="RobotLab-Isaac-BeyondMimic-Flat-pcbC-v1-lateral-bipedal-stand-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{lateral_bipedal_stand.__name__}:PcbCLateralBipedalStandEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PcbCLateralGuidedPPORunnerCfg",
    },
)


gym.register(
    id="RobotLab-Isaac-BeyondMimic-Flat-pcbC-v1-lateral-bipedal-stand-stable-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{lateral_bipedal_stand.__name__}:PcbCLateralBipedalStandStableEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PcbCLateralGuidedPPORunnerCfg",
    },
)


gym.register(
    id="RobotLab-Isaac-BeyondMimic-Flat-pcbC-v1-lateral-bipedal-stand-stable-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{lateral_bipedal_stand.__name__}:PcbCLateralBipedalStandStableV1EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PcbCLateralGuidedPPORunnerCfg",
    },
)


gym.register(
    id="RobotLab-Isaac-BeyondMimic-Flat-pcbC-v1-lateral-bipedal-stand-stable-v2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{lateral_bipedal_stand.__name__}:PcbCLateralBipedalStandStableV2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PcbCLateralGuidedPPORunnerCfg",
    },
)


gym.register(
    id="RobotLab-Isaac-BeyondMimic-Flat-pcbC-v1-lateral-bipedal-stand-stable-v3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{lateral_bipedal_stand.__name__}:PcbCLateralBipedalStandStableV3EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PcbCLateralGuidedPPORunnerCfg",
    },
)



gym.register(
    id="RobotLab-Isaac-BeyondMimic-Flat-pcbC-v1-lateral-bipedal-stand-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{lateral_bipedal_stand.__name__}:PcbCLateralBipedalStandDeployObsEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PcbCLateralGuidedPPORunnerCfg",
    },
)


gym.register(
    id="RobotLab-Isaac-BeyondMimic-Flat-pcbC-v1-lateral-guided-cartrunk-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{lateral_guided_env_cfg.__name__}:PcbCLateralGuidedCarTrunkRobustEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PcbCLateralGuidedPPORunnerCfg",
    },
)


gym.register(
    id="RobotLab-Isaac-BeyondMimic-Flat-pcbC-v1-lateral-guided-cartrunk-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{lateral_guided_env_cfg.__name__}:PcbCLateralGuidedCarTrunkRobustEnv1Cfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PcbCLateralGuidedPPORunnerCfg",
    },
)


gym.register(
    id="RobotLab-Isaac-BeyondMimic-Flat-pcbC-v1-lateral-guided-cartrunk-v2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{lateral_guided_env_cfg.__name__}:PcbCLateralGuidedCarTrunkRobustEnv2Cfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PcbCLateralGuidedPPORunnerCfg",
    },
)


gym.register(
    id="RobotLab-Isaac-BeyondMimic-Flat-pcbC-v1-lateral-guided-cartrunk-v3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{lateral_guided_env_cfg.__name__}:PcbCLateralGuidedCarTrunkRobustEnv3Cfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PcbCLateralGuidedPPORunnerCfg",
    },
)


gym.register(
    id="RobotLab-Isaac-BeyondMimic-Flat-pcbC-v1-lateral-guided-cartrunk-v4",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{lateral_guided_env_cfg.__name__}:PcbCLateralGuidedCarTrunkRobustEnv4Cfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PcbCLateralGuidedPPORunnerCfg",
    },
)


gym.register(
    id="RobotLab-Isaac-BeyondMimic-Flat-pcbC-v1-lateral-guided-cartrunk-v5",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{lateral_guided_env_cfg.__name__}:PcbCLateralGuidedCarTrunkRobustEnv5Cfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PcbCLateralGuidedPPORunnerCfg",
    },
)


gym.register(
    id="RobotLab-Isaac-BeyondMimic-Flat-pcbC-v1-lateral-guided-cartrunk-v6",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{lateral_guided_env_cfg.__name__}:PcbCLateralGuidedCarTrunkRobustEnv6Cfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PcbCLateralGuidedPPORunnerCfg",
    },
)


gym.register(
    id="RobotLab-Isaac-BeyondMimic-Flat-pcbC-v1-lateral-guided-cartrunk-v7",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{lateral_guided_env_cfg.__name__}:PcbCLateralGuidedCarTrunkRobustEnv7Cfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PcbCLateralGuidedPPORunnerCfg",
    },
)


gym.register(
    id="RobotLab-Isaac-BeyondMimic-Flat-pcbC-v1-lateral-guided-cartrunk-v8",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{lateral_guided_env_cfg.__name__}:PcbCLateralGuidedCarTrunkRobustEnv8Cfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PcbCLateralGuidedPPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-BeyondMimic-Flat-pcbC-v1-lateral-guided-cartrunk-v9",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{lateral_guided_env_cfg.__name__}:PcbCLateralGuidedCarTrunkRobustEnv9Cfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PcbCLateralGuidedPPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-BeyondMimic-Flat-pcbC-v1-lateral-guided-cartrunk-v10",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{lateral_guided_env_cfg.__name__}:PcbCLateralGuidedCarTrunkRobustEnv10Cfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PcbCLateralGuidedPPORunnerCfg",
    },
)


gym.register(
    id="RobotLab-Isaac-BeyondMimic-Flat-pcbC-v1-lateral-guided-cartrunk-v11",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{lateral_guided_env_cfg.__name__}:PcbCLateralGuidedCarTrunkRobustEnv11Cfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PcbCLateralGuidedPPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-BeyondMimic-Flat-pcbC-v1-lateral-guided-cartrunk-v12",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{lateral_guided_env_cfg.__name__}:PcbCLateralGuidedCarTrunkRobustEnv12Cfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PcbCLateralGuidedPPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-BeyondMimic-Flat-pcbC-v1-lateral-guided-cartrunk-v13",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{lateral_guided_env_cfg.__name__}:PcbCLateralGuidedCarTrunkRobustEnv13Cfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PcbCLateralGuidedPPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-BeyondMimic-Flat-pcbC-v1-lateral-guided-cartrunk-v14",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{lateral_guided_env_cfg.__name__}:PcbCLateralGuidedCarTrunkRobustEnv14Cfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PcbCLateralGuidedPPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-BeyondMimic-Flat-pcbC-v1-lateral-guided-cartrunk-v15",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{lateral_guided_env_cfg.__name__}:PcbCLateralGuidedCarTrunkRobustEnv15Cfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PcbCLateralGuidedPPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-BeyondMimic-Flat-pcbC-v1-lateral-guided-cartrunk-v16",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{lateral_guided_env_cfg.__name__}:PcbCLateralGuidedCarTrunkRobustEnv16Cfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PcbCLateralGuidedPPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-BeyondMimic-Flat-pcbC-v1-lateral-guided-cartrunk-h1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{lateral_guided_env_cfg.__name__}:PcbCLateralGuidedCarTrunkRobustEnvH1Cfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PcbCLateralGuidedPPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-BeyondMimic-Flat-pcbC-v1-lateral-guided-cartrunk-h2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{lateral_guided_env_cfg.__name__}:PcbCLateralGuidedCarTrunkRobustEnvH2Cfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PcbCLateralGuidedPPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-BeyondMimic-Flat-pcbC-v1-lateral-guided-cartrunk-v17",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{lateral_guided_env_cfg.__name__}:PcbCLateralGuidedCarTrunkRobustEnv17Cfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PcbCLateralGuidedPPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-BeyondMimic-Flat-pcbC-v1-lateral-guided-cartrunk-command-curriculum-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{lateral_command_curriculum.__name__}:"
            "PcbCLateralGuidedCarTrunkCommandCurriculumEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PcbCLateralGuidedPPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-BeyondMimic-Flat-pcbC-v1-lateral-guided-cartrunk-v18",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{lateral_guided_env_cfg.__name__}:PcbCLateralGuidedCarTrunkRobustEnv18Cfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PcbCLateralGuidedPPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-BeyondMimic-Flat-pcbC-v1-lateral-guided-cartrunk-v19",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{lateral_guided_env_cfg.__name__}:PcbCLateralGuidedCarTrunkRobustEnv19Cfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PcbCLateralGuidedPPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-BeyondMimic-Flat-pcbC-v1-lateral-guided-cartrunk-v20",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{lateral_guided_env_cfg.__name__}:PcbCLateralGuidedCarTrunkRobustEnv20Cfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PcbCLateralGuidedPPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-BeyondMimic-Flat-pcbC-v1-lateral-guided-cartrunk-v17-fanqu713-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{lateral_guided_env_cfg.__name__}:PcbCLateralGuidedCarTrunkRobustEnv17Fanqu713Cfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:PcbCLateralGuidedFanqu713PPORunnerCfg"
        ),
    },
)

gym.register(
    id="RobotLab-Isaac-BeyondMimic-Flat-pcbC-v1-lateral-guided-cartrunk-vX0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{lateral_guided_env_cfg.__name__}:PcbCLateralGuidedCarTrunkRobustEnvX0Cfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PcbCLateralGuidedPPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-BeyondMimic-Flat-pcbC-v1-lateral-guided-cartrunk-vX1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{lateral_guided_env_cfg.__name__}:PcbCLateralGuidedCarTrunkRobustEnvX1Cfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PcbCLateralGuidedPPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-BeyondMimic-Flat-pcbC-v1-lateral-guided-cartrunk-vX2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{lateral_guided_env_cfg.__name__}:PcbCLateralGuidedCarTrunkRobustEnvX2Cfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PcbCLateralGuidedPPORunnerCfg",
    },
)



gym.register(
    id="RobotLab-Isaac-BeyondMimic-Flat-pcbC-v1-lateral-red-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{lateral_red_env_cfg.__name__}:PcbCLateralRedEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PcbCLateralRedPPORunnerCfg",
    },
)


gym.register(
    id="RobotLab-Isaac-BeyondMimic-Flat-pcbC-v1-lateral-add-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{lateral_add_env_cfg.__name__}:PcbCLateralAddEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PcbCLateralAddPPORunnerCfg",
    },
)


gym.register(
    id="RobotLab-Isaac-BeyondMimic-Flat-pcbC-v1-stage-red-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{stage_red_env_cfg.__name__}:PcbCStageRedEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PcbCStageRedPPORunnerCfg",
    },
)


gym.register(
    id="RobotLab-Isaac-BeyondMimic-Flat-pcbC-v1-stage2-lateral-guided-baseline-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{flat_env_cfg.__name__}:PcbCBeyondMimicFlatV1Stage2LateralGuidedEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PcbCBeyondMimicFlatPPORunnerCfg",
    },
)


gym.register(
    id="RobotLab-Isaac-BeyondMimic-Flat-pcbC-v1-stage2-lateral-guided-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{flat_env_cfg.__name__}:PcbCBeyondMimicFlatV1Stage2LateralGuidedEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PcbCStage2LateralGuidedPPORunnerCfg",
    },
)
