# MPPI + DAgger 横向运动策略训练

本项目用于训练 pcbC 机器人在尾箱接触场景下的横向运动策略。系统在 Isaac Lab 中使用 GPU 并行 MPPI 作为专家，先通过行为克隆（BC）训练初始学生策略，再通过 DAgger 在学生实际访问的状态上持续补充专家标签，最终导出 TorchScript 和 ONNX 模型。

策略以 50 Hz 运行，固定接口为：

```text
float32 obs [1, 93] -> policy -> float32 actions [1, 16]
```

前 12 个输出为腿关节位置残差，后 4 个输出为轮关节速度槽。当前控制合同要求四个轮输出在模型图中始终精确为零。

## 训练流程

```text
参考轨迹与控制合同
        ↓
Isaac Lab 并行 MPPI 专家
        ↓
R0 专家数据采集
        ↓
行为克隆（BC）
        ↓
学生闭环准入门禁
        ↓
DAgger R1 → R2 → R3
        ↓
最终闭环门禁
        ↓
TorchScript / ONNX 导出
```

各阶段采用 fail-closed 方式运行：前一阶段的验证或门禁没有通过时，不应继续下一阶段。

## 项目结构

```text
assets/                 项目生成的参考轨迹、动作参考等哈希资产
configs/                控制合同、参考、MPPI、学生、DAgger 和门禁配置
  low_load_lateral/     按 train_NNN 隔离的稳定横移训练配置
scripts/                数据采集、训练、评估、导出与审计入口
src/lateral_mppi_dagger/
  contract/             93D observation、16D action 和 joint mapping
  data/                 episode schema、collector 和 dataset
  env/                  Isaac adapter、MPPI rollout clone 和场景扰动
  expert/               MPPI、reference WBC 和 safety fallback
  student/              学生网络、loss 和 BC trainer
  evaluation/           expert/student/DAgger 闭环门禁
  export/               TorchScript/ONNX 导出与验证
tests/                  单元测试和合同回归测试
vendor/                 Isaac Lab、RobotLab、机器人资产和参考轨迹快照
reference_only/         上游算法来源说明，不参与运行
reports/                项目内已有的固定分析与验证报告
training_records/       本地训练记录和报告，整个目录不提交 Git
datasets/               生成的数据集，不提交 Git
checkpoints/            训练 checkpoint，不提交 Git
exported/               导出模型，不提交 Git
```

## Low-load 训练编号

稳定横移使用 `task_id/training_id/stage` 三层命名，而不是用每次参数试错的
`v1...v146` 作为训练版本。当前第一条正式训练链是：

```text
task_id       low_load_lateral
training_id   train_001
reference     configs/low_load_lateral/train_001/reference.yaml
expert        configs/low_load_lateral/train_001/expert.yaml
assets        assets/low_load_lateral/train_001/
```

`qualification`、`r0`、`r1`、`r2`、`r3`、`performance` 和 `release`
表示阶段。以后 reference、cost、proposal 或 solver schedule 发生实质变化
时创建 `train_002`，并从 state-copy 和 9-reference 门重新验证。完整命令和
保留规则见 `STABLE_LATERAL_RUNBOOK.md`。

## Clone 后从零运行

### 1. 获取仓库

```bash
git clone https://github.com/zhuoxili-jackie/Mppi_Dagger.git
cd Mppi_Dagger
```

仓库已经包含运行所需的项目源码、Isaac Lab/RobotLab 源码快照、机器人与尾箱资产、参考轨迹和配置。`reference_only/RTWholeBodyMPPI` 不在 Git 中，也不是运行依赖。

### 2. 只运行离线测试、BC 和模型导出

离线流程不需要 Isaac Sim 或 NVIDIA GPU：

```bash
conda create -n mppi-dagger python=3.11 -y
conda activate mppi-dagger

python -m pip install --upgrade pip
python -m pip install -e ".[test,export]"

python scripts/audit_project.py
python scripts/validate_reference.py \
  --config configs/reference_708.yaml
python -m pytest -q
```

如果以上命令通过，可以继续运行下文的[快速离线 Smoke](#快速离线-smoke)。

### 3. 运行 Isaac GPU 采集与 MPPI

真实 Isaac 仿真建议使用 Linux x86-64、Python 3.11、GLIBC 2.35+、支持 CUDA 的 NVIDIA GPU 和可用的 NVIDIA 驱动。下面给出本项目固定的已验证软件基线：

```text
Python       3.11
Isaac Sim    5.1.0
PyTorch      2.7.0
CUDA wheel   12.8
```

创建独立环境：

```bash
conda create -n isaaclab python=3.11 -y
conda activate isaaclab

python -m pip install --upgrade pip

python -m pip install \
  torch==2.7.0 torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cu128

python -m pip install \
  "isaacsim[all,extscache]==5.1.0" \
  --extra-index-url https://pypi.nvidia.com

python -m pip install -e ".[test,export]"
```

项目脚本会优先加载仓库 `vendor/` 中固定的 Isaac Lab 和 RobotLab 源码，不需要另外 clone 这两个源码仓库。

安装完成后先检查系统和 Python：

```bash
nvidia-smi

python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0)); assert torch.cuda.is_available()"

python -c "import isaacsim; print('Isaac Sim import OK')"
```

如果是新机器，建议先运行 Isaac Sim 自带的兼容性检查：

```bash
isaacsim isaacsim.exp.compatibility_check
```

然后在项目根目录运行：

```bash
python scripts/audit_project.py

python scripts/validate_reference.py \
  --config configs/reference_708.yaml

python -m pytest -q

python scripts/validate_mppi_rollout.py \
  --headless \
  --device cuda:0 \
  --samples 4 \
  --horizon 2 \
  --ref-id 1 \
  --mppi-config configs/expert_mppi.yaml \
  --report training_records/first_clone_mppi_smoke.json
```

最后一个命令是最小 Isaac/MPPI 验收。报告中的 `"pass"` 必须为 `true`，并且启动日志中的 experience 路径应指向当前 clone 内的：

```text
vendor/IsaacLab/apps/isaaclab.python.headless.kit
```

### 4. 基础检查的含义

- `audit_project.py` 检查运行路径、资产、SHA256、URDF mesh 引用和外部符号链接；
- `validate_reference.py` 检查参考轨迹长度、频率、首帧和接触计划；
- 测试覆盖 observation/action 合同、MPPI、数据集、学生网络、闭环门禁和模型导出；
- `validate_mppi_rollout.py` 检查 Isaac clone state、restore、重复 rollout、有限代价和 wheel hard-zero。

## 训练方法

### MPPI 专家

正式专家是本项目实现的 Isaac-native MPPI。每个控制周期：

1. 从参考轨迹构造未来动作序列；
2. 对 12 个腿部动作采样时间相关扰动；
3. 将候选序列投影到动作绝对边界和变化率边界；
4. 在 Isaac GPU 并行环境中执行 rollout；
5. 根据轨迹代价加权候选序列；
6. 执行第一拍动作，并将优化结果用于下一拍 warm start。

代价函数包括 reference tracking、横向速度、姿态、尾箱局部漂移、轮端位置、接触匹配、轮端载荷、动作变化率、关节限制和仿真终止。

主要配置：

- `configs/expert_mppi.yaml`
- `configs/reference_708.yaml`
- `configs/low_load_lateral/train_001/expert.yaml`
- `configs/low_load_lateral/train_001/reference.yaml`

参考配置与 Isaac task 内部的 motion/reset 定义必须保持一致。更换参考集合后，应先逐条运行 reset 和短时 MPPI smoke，再启用 rotating-reference 正式采集。

### 学生策略

默认学生网络为：

```text
93 → 256 → 256 → 128 → 12
          ELU
```

模型内部包含 observation normalization、横向指令限制、连续指令激活、腿部目标包络、目标变化率限制，以及四个图级 exact-zero wheel outputs。

默认训练参数位于 `configs/student.yaml`：

- AdamW；
- batch size 1024；
- 100 epochs；
- 长度为 3 的 episode 内时间窗；
- masked Huber imitation loss；
- 一阶和二阶动作差分正则；
- gradient norm clipping。

### DAgger

DAgger 在 episode 级选择教师或学生作为执行者，但每个访问状态都会查询并保存 MPPI 标签。默认 round、β 和场景扰动定义在 `configs/dagger.yaml`。

进入下一轮前，当前学生必须先通过独立的 `beta=0` 闭环准入门禁。新一轮训练从上一轮 checkpoint 初始化，同时使用新数据重新计算 normalization。

## 快速离线 Smoke

下面的流程不启动 Isaac，可用于检查数据、训练和 checkpoint 流程：

```bash
python scripts/collect_replay_smoke.py \
  --dataset datasets/replay_smoke_fresh \
  --episodes 12 \
  --steps 64

python scripts/validate_dataset.py datasets/replay_smoke_fresh

python scripts/train_bc.py \
  --dataset datasets/replay_smoke_fresh \
  --output checkpoints/replay_smoke_fresh \
  --device cpu \
  --epochs 5 \
  --batch-size 64 \
  --max-batches-per-epoch 4

python scripts/evaluate.py \
  --checkpoint checkpoints/replay_smoke_fresh/student_best_checkpoint.pt \
  --dataset datasets/replay_smoke_fresh \
  --split validation \
  --device cpu \
  --output training_records/replay_smoke_evaluation.json
```

## 完整训练示例

以下命令以 `reference_708.yaml` 和 `expert_mppi.yaml` 为例。使用其他参考或 MPPI 配置时，应在所有采集、门禁和 DAgger 命令中保持配置一致。

### 1. MPPI 状态复制与专家 Smoke

```bash
python scripts/validate_mppi_rollout.py \
  --headless \
  --device cuda:0 \
  --mppi-config configs/expert_mppi.yaml \
  --samples 16 \
  --horizon 8 \
  --ref-id 0 \
  --report training_records/mppi_state_copy.json

python scripts/smoke_test_expert.py \
  --headless \
  --device cuda:0 \
  --expert-backend mppi \
  --mppi-config configs/expert_mppi.yaml \
  --dataset datasets/expert_smoke \
  --episodes 1 \
  --steps 32 \
  --run-name expert_smoke
```

### 2. 采集 R0 数据

显式指定 episode split，避免时间相邻样本跨 split 泄漏：

```bash
python scripts/collect_expert.py \
  --headless --device cuda:0 \
  --expert-backend mppi \
  --reference-config configs/reference_708.yaml \
  --mppi-config configs/expert_mppi.yaml \
  --dataset datasets/dagger_aggregate \
  --episodes 40 --steps 300 --seed 1000 \
  --ref-id 0 --rotate-references \
  --scenario nominal --dagger-round 0 \
  --split train --run-name r0_train \
  --report training_records/r0_train_collection.json

python scripts/collect_expert.py \
  --headless --device cuda:0 \
  --expert-backend mppi \
  --reference-config configs/reference_708.yaml \
  --mppi-config configs/expert_mppi.yaml \
  --dataset datasets/dagger_aggregate \
  --episodes 10 --steps 300 --seed 2000 \
  --ref-id 0 --rotate-references \
  --scenario nominal --dagger-round 0 \
  --split validation --run-name r0_validation \
  --report training_records/r0_validation_collection.json

python scripts/validate_dataset.py datasets/dagger_aggregate \
  --output training_records/r0_dataset_validation.json

python scripts/analyze_observability.py datasets/dagger_aggregate \
  --output training_records/r0_observability.json
```

### 3. 训练 BC

```bash
python scripts/train_bc.py \
  --dataset datasets/dagger_aggregate \
  --output checkpoints/bc \
  --device cuda:0 \
  --epochs 100 \
  --batch-size 1024

python scripts/evaluate.py \
  --checkpoint checkpoints/bc/student_best_checkpoint.pt \
  --dataset datasets/dagger_aggregate \
  --split validation \
  --device cuda:0 \
  --output training_records/bc_open_loop.json
```

同一次训练中断后使用 `--resume`。数据 manifest 发生变化或进入新一轮 DAgger 时，应使用 `--initialize-from`，不要把它当作同一次训练续跑。

### 4. 学生闭环准入

```bash
python scripts/run_student_gate.py \
  --checkpoint checkpoints/bc/student_best_checkpoint.pt \
  --dataset datasets/bc_admission \
  --run-name bc_admission \
  --reference-config configs/reference_708.yaml \
  --tracking-config configs/expert_mppi.yaml \
  --device cuda:0 \
  --episodes 50 \
  --steps 300 \
  --seed 4000 \
  --gate-purpose dagger_admission \
  --dagger-admission-full-horizon-success-rate-min 0.80 \
  --dagger-admission-per-reference-mean-horizon-fraction-min 0.95 \
  --dagger-admission-minimum-episode-horizon-fraction 0.85 \
  --dagger-admission-signed-progress-ratio-min 0.20 \
  --output training_records/bc_admission.json
```

### 5. DAgger Round

```bash
python scripts/run_dagger.py \
  --round 1 \
  --dataset datasets/dagger_aggregate \
  --input-checkpoint checkpoints/bc/student_best_checkpoint.pt \
  --previous-gate training_records/bc_admission.json \
  --output checkpoints/dagger_r1 \
  --reference-config configs/reference_708.yaml \
  --mppi-config configs/expert_mppi.yaml \
  --device cuda:0 \
  --epochs 100
```

R2 和 R3 使用上一轮通过准入门禁的 best checkpoint，并修改 `--round`、`--previous-gate` 和 `--output`。

### 6. 导出与验证

```bash
python scripts/export_policy.py \
  --checkpoint checkpoints/dagger_r3/student_best_checkpoint.pt \
  --dataset datasets/dagger_aggregate \
  --output exported/lateral_policy

python scripts/validate_export.py \
  --exported exported/lateral_policy \
  --output training_records/export_validation.json

python scripts/validate_key7_handoff.py \
  --checkpoint checkpoints/dagger_r3/student_best_checkpoint.pt \
  --onnx exported/lateral_policy/policy.onnx \
  --output training_records/key7_handoff.json
```

导出目录包含：

- `policy.pt`
- `policy.onnx`
- `golden_io.npz`
- `export_manifest.json`
- `SHA256SUMS`

## 训练记录

每次实验都应在根目录的 `training_records/` 下建立独立目录。该目录已被 Git 忽略，不应把实验结果持续追加到 README。

推荐命名方式：

```text
training_records/YYYYMMDD_HHMM_<run_name>/
```

建议保存：

- 完整命令；
- 使用的配置副本；
- Git commit 和工作树状态；
- 控制台日志；
- collection、validation、gate 和 evaluation JSON；
- checkpoint 与导出文件的 SHA256；
- 实验结论和下一步计划。

示例：

```bash
RUN_DIR="training_records/$(date +%Y%m%d_%H%M)_bc_r0"
mkdir -p "$RUN_DIR"

git rev-parse HEAD > "$RUN_DIR/git_commit.txt"
git status --short > "$RUN_DIR/git_status.txt"
cp configs/student.yaml "$RUN_DIR/"

python scripts/train_bc.py \
  --dataset datasets/dagger_aggregate \
  --output checkpoints/bc \
  --device cuda:0 \
  2>&1 | tee "$RUN_DIR/console.log"

cp checkpoints/bc/resolved_train_config.json "$RUN_DIR/"
cp checkpoints/bc/metrics.jsonl "$RUN_DIR/"
sha256sum checkpoints/bc/student_best_checkpoint.pt \
  > "$RUN_DIR/checkpoint_sha256.txt"
```

`training_records/` 整个目录仅保存在本地，不会被 Git 跟踪。

## Observation 与 Action 合同

权威定义位于：

- `configs/deployment_contract.yaml`
- `src/lateral_mppi_dagger/contract/obs93.py`
- `src/lateral_mppi_dagger/contract/action16.py`
- `src/lateral_mppi_dagger/contract/joint_mapping.py`

| Observation slice | 维度 | 含义 |
|---|---:|---|
| `0:16` | 16 | motion joint position prefix |
| `16:32` | 16 | motion joint velocity prefix |
| `32:38` | 6 | 相对旋转矩阵前两列 |
| `38:41` | 3 | body-frame base angular velocity |
| `41:57` | 16 | joint position residual |
| `57:73` | 16 | joint velocity residual |
| `73:89` | 16 | 上一拍实际执行的 raw action |
| `89:92` | 3 | velocity command |
| `92:93` | 1 | constant zero |

Action 的 policy joint order 与 runtime joint order 不同。训练、导出和部署必须统一使用合同中定义的映射，不能在其他位置重新维护一份映射。

## License

本项目采用 Apache License 2.0，见 `LICENSE`。`vendor/IsaacLab` 与 `vendor/robot_lab` 保留各自的许可证文件。
