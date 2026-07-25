# low_load_lateral / train_001 清理与资格复核

日期：2026-07-25

## 清理

- `git status --short --untracked-files=all`：2,461 条降至清理后的 80 条；
- 删除 2,392 个未跟踪试错文件和 11 个已被当前主线替代的旧 v1
  reference/config 文件；
- `datasets/`：约 443 MB 清为空壳，避免混入旧 43,651 帧；
- `checkpoints/`：约 217 MB 降至约 1.5 MB；
- `assets/`：约 102 MB 降至约 1.7 MB；
- `reports/`：约 63 MB 降至只包含 Git 固定历史报告和本次结构化报告。

删除的未跟踪文件不受 Git 管理，无法从本仓库恢复。其最终合格参数、哈希
reference 和 nominal-action 资产已迁入：

```text
configs/low_load_lateral/train_001/
assets/low_load_lateral/train_001/
```

唯一保留的旧模型用于 Key7 首帧/ABI 回归，不是横移候选：

```text
checkpoints/deployment_v2_columns_handoff/student_best_checkpoint.pt
sha256 5f03be871d173f2cad29cd8777a8906db1ad61ffa077768c92172c5bae998604

exported/lateral_policy_v2_key7_aligned_r1/policy.onnx
sha256 1995c8166d92e0973178084ddf10b290ad1604009b215453e6e85b14fa9a2109
```

外部部署仓库、存档工程、`reference_only/RTWholeBodyMPPI` 和真机均未触碰。

## 命名

```text
task_id       low_load_lateral
training_id   train_001
stage         qualification / r0 / r1 / r2 / r3 / performance / release
```

以后实质修改使用 `train_002`，不得在已合格 `train_001` 内静默换资产。

## 清理后复核

```text
audit                         PASS
reference validation          PASS
pytest                        129 passed
state-copy                    PASS
9 references × 100 MPPI       PASS
```

权威输出：

```text
reports/low_load_lateral/train_001/qualification/audit.json
reports/low_load_lateral/train_001/qualification/reference.json
reports/low_load_lateral/train_001/qualification/state_copy_ref01.json
reports/low_load_lateral/train_001/qualification/server_9x100.json
reports/low_load_lateral/train_001/qualification/collection_9x100.json
reports/low_load_lateral/train_001/qualification/gate_9x100.json
datasets/low_load_lateral/train_001/qualification_9x100
```

state-copy 的 state/cost 最大误差均为 0，buffers、delay queue、platform
pose 和 previous command 完整恢复，rollout wheel action exact zero。

9×100 server 为 900 requests / 9 resets，solver phase counts 为
`0:122, 1:38, 2:40`，无错误。expert gate 为 9/9、tracking 9/9、
teacher valid 1.0、ref IDs 0..8 全覆盖、reset 全匹配、shield 0、wheel
action exact zero。

当前已按用户要求停在 50×300/R0 前。没有启动 R0、BC、R1–R3，也没有生成
或复制新的 ONNX。
