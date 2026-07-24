# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym

from . import agents, flat_env_cfg


gym.register(
    id="RobotLab-Isaac-BeyondMimic-Flat-pcbA-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv", #`ManagerBasedRLEnv` 是 Isaac Lab 中的强化学习环境基类，负责把 `actions/observations/rewards/terminations/events` 这些 manager 串起来。
    disable_env_checker=True,
    kwargs={
    
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:PcbABeyondMimicFlatV1StandEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PcbABeyondMimicFlatPPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-BeyondMimic-Flat-pcbA-v1-stand-delay",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:PcbABeyondMimicFlatV1StandDelayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PcbABeyondMimicFlatPPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-BeyondMimic-Flat-pcbA-v1-stand-command-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:PcbABeyondMimicFlatV1StandCommandEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PcbABeyondMimicFlatPPORunnerCfg",
    },
)

# gym.register(
#     id="RobotLab-Isaac-BeyondMimic-Flat-pcbA-v1-stand-command-v0",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.flat_env_cfg:Stage2CommandEnvCfgV41",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PcbABeyondMimicFlatCommandPPORunnerCfg",
#     },
# )


