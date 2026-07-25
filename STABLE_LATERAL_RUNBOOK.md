# 稳定横移训练手册

本手册只描述当前可继续执行的 low-load 主线。所有阶段 fail-closed；前一门
没有通过，不得开始后一阶段，也不得导出 release ONNX。

## 1. 任务与训练编号

当前唯一主线：

```text
task_id       low_load_lateral
training_id   train_001
reference     configs/low_load_lateral/train_001/reference.yaml
expert        configs/low_load_lateral/train_001/expert.yaml
assets        assets/low_load_lateral/train_001/
reports       reports/low_load_lateral/train_001/
datasets      datasets/low_load_lateral/train_001/
checkpoints   checkpoints/low_load_lateral/train_001/
export        exported/low_load_lateral/train_001/
```

`train_001` 表示第一条将从 R0 进入 BC、R1、R2、R3 的正式训练链，不表示
第 001 次参数试错。若以后因部署问题修改 reference、cost、proposal 或
solver schedule，创建 `train_002` 并重新通过全部门；不得在 `train_001`
目录内静默替换资产，也不要恢复 `v147` 一类编号。

阶段目录统一使用：

```text
qualification/   state-copy、9×100 和 50×300 expert 门
r0/              新 low-load aggregate 与 BC
r1/ r2/ r3/      真实 student-state DAgger
performance/     100×300 student gate
release/         export、ONNX 和 Key7 handoff 验证
```

## 2. 不可变合同

- 仅在本项目根目录写入；`loco_rl_deploy`、`robot_move_lt`、旧工程和
  `reference_only/RTWholeBodyMPPI` 永久只读。
- 不控制真机，不复制或覆盖部署 ONNX，不修改部署端距离模式。
- ABI 固定为 `float32 [1,93] -> [1,16]`、opset 18。
- rotation 6D 是矩阵前两列
  `[R00,R01,R10,R11,R20,R21]`；obs 92 恒为 0。
- motion prefix 固定第一帧；joint order 为 policy type-grouped。
- student 前 12 维是腿关节位置 residual，后 4 个轮动作必须逐位精确为 0。
- Key7 三次 dry inference，第四次才 apply。
- 部署指令 ramp 为 `0 -> ±0.012 -> ±0.024 -> ±0.030 m/s`。

## 3. 每次启动先做的检查

```bash
pwd -P
git status --short
nvidia-smi
/home/zzc/miniconda3/envs/isaaclab/bin/python -c \
  "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0)); assert torch.cuda.is_available()"

/home/zzc/miniconda3/envs/isaaclab/bin/python scripts/audit_project.py \
  --output reports/low_load_lateral/train_001/qualification/audit.json

/home/zzc/miniconda3/envs/isaaclab/bin/python scripts/validate_reference.py \
  --config configs/low_load_lateral/train_001/reference.yaml \
  --output reports/low_load_lateral/train_001/qualification/reference.json

/home/zzc/miniconda3/envs/isaaclab/bin/python -m pytest -q
```

根目录必须是实际 `lateral_mppi_dagger`，CUDA 必须可用，audit、reference
和 pytest 必须全部通过。

## 4. state-copy 门

```bash
/home/zzc/miniconda3/envs/isaaclab/bin/python \
  scripts/validate_mppi_rollout.py \
  --headless --device cuda:0 \
  --mppi-config configs/low_load_lateral/train_001/expert.yaml \
  --samples 16 --horizon 12 --ref-id 1 \
  --report reports/low_load_lateral/train_001/qualification/state_copy_ref01.json
```

要求 `pass=true`，state/cost 误差不超过 `1e-5`，delay queue、buffers、
platform pose 和 previous command 完整恢复，wheel action exact zero。

## 5. 9 references × 100 steps 门

在项目根目录准备短 socket 路径：

```bash
socket_path="$PWD/artifacts/ll_t001_q.sock"
```

先启动 persistent isolated MPPI server：

```bash
/home/zzc/miniconda3/envs/isaaclab/bin/python \
  scripts/run_isolated_mppi_server.py \
  --headless --device cuda:0 \
  --mppi-config configs/low_load_lateral/train_001/expert.yaml \
  --reference-config configs/low_load_lateral/train_001/reference.yaml \
  --scenario nominal --seed 1000 \
  --socket "$socket_path" \
  --report reports/low_load_lateral/train_001/qualification/server_9x100.json
```

server ready 后运行：

```bash
/home/zzc/miniconda3/envs/isaaclab/bin/python \
  scripts/smoke_test_expert.py \
  --headless --device cuda:0 \
  --expert-backend mppi \
  --mppi-config configs/low_load_lateral/train_001/expert.yaml \
  --mppi-samples 256 --mppi-horizon 40 --mppi-iterations 2 \
  --mppi-server-socket "$socket_path" \
  --reference-config configs/low_load_lateral/train_001/reference.yaml \
  --dataset datasets/low_load_lateral/train_001/qualification_9x100 \
  --episodes 9 --steps 100 --seed 1000 \
  --ref-id 0 --rotate-references \
  --scenario nominal --dagger-round 0 --split train \
  --run-name low_load_lateral_train_001_qualification_9x100 \
  --report reports/low_load_lateral/train_001/qualification/collection_9x100.json
```

最后运行门：

```bash
/home/zzc/miniconda3/envs/isaaclab/bin/python \
  scripts/validate_expert_gate.py \
  datasets/low_load_lateral/train_001/qualification_9x100 \
  --config configs/low_load_lateral/train_001/expert.yaml \
  --episode-count 9 --required-successes 9 \
  --seed-start 1000 --full-episode-steps 100 \
  --output reports/low_load_lateral/train_001/qualification/gate_9x100.json
```

必须同时满足：

- `9/9`、teacher valid `1.0`、ref ID `0..8` 全覆盖；
- requested/reset reference 完全一致，无取模；
- 双向 signed progress、速度/位移、姿态、base drift 全通过；
- rear p95、rear single/no support、front X-normal support 全通过；
- physical target step、finite、survival 和 termination 全通过；
- 四轮 action exact zero。

## 6. 下一阶段：50 × 300 expert 门

只有第 5 节通过后才能启动。该阶段预计运行数小时；中断时保留已完成
episode，使用 `--resume` 和新 server report，不覆盖旧报告。

```text
dataset:
  datasets/low_load_lateral/train_001/expert_50x300
reports:
  reports/low_load_lateral/train_001/qualification/server_50x300.json
  reports/low_load_lateral/train_001/qualification/collection_50x300.json
  reports/low_load_lateral/train_001/qualification/gate_50x300.json
```

门要求至少 `48/50`，teacher valid 不低于 `0.99`，9 references 全覆盖，
tracking/load/action/wheel/reset/hash checks 全通过。随后才可运行
observability；若返回 `BLOCK_FORMAL_BC_DAGGER`，立即停止，不修改 93D。

## 7. R0、BC、R1–R3 与 release

observability 通过后：

1. 只从 `train_001/expert_50x300` 创建新的 low-load aggregate；
2. 不混入旧 43,651 帧或任何已删除历史 dataset；
3. BC 训练 100 epoch；
4. 50×300 student DAgger admission 的 moving signed progress 必须至少
   `0.20`；
5. R1、R2、R3 每轮约 7,800 个真实 student-state transitions，每轮
   100 epoch，总新数据至少 38,400；
6. 每轮失败即保留报告并停止，不放宽门；
7. 最终 100×300 student performance gate 要求总成功率至少 `0.95`、
   每 reference 至少 `0.90`，并通过双向 progress、速度/位移、
   force/load、standing/start/stop、action-step、wheel-zero、
   finite/survival。

只有全部通过后，才允许导出：

```text
exported/low_load_lateral/train_001/release/
```

并运行 `validate_export.py` 与 `validate_key7_handoff.py`。未通过全部门的
模型只能放在 `unqualified_debug/`，绝不能覆盖部署模型。

## 8. 保留与清理规则

- 永久保留每个正式 training ID 的最终配置、哈希资产、门报告、最终
  aggregate、选中的 best checkpoint 和 release 验证。
- 候选网格、失败 replay、临时 socket/log 和中间 epoch checkpoint 在结论
  写入结构化报告后可删除。
- 新问题使用新的 `train_NNN`，不要修改已完成 training ID 的资产。
- 所有 Git commit/push 必须得到用户在当前对话中的明确授权。
