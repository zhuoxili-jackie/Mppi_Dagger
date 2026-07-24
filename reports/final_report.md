# MPPI + DAgger 迁移最终工程报告

## 结论

独立工程迁移、真实 Isaac Lab MPPI 教师、超过两万帧数据采集、BC、R1–R3
DAgger、TorchScript/ONNX 导出及 ONNX 闭环准入均已完成。当前产物是
**工程发布候选（engineering release candidate）**，不是生产任务验收版本。

- 工程发布检查：`True`
- 严格生产 tracking 验收：`False`
- 最终 student checkpoint SHA256：`c91ef211b25731da80348aa21b90598a22731c0b31e02f7634655f708dd0a8b6`
- 最终 ONNX SHA256：`45719ba3e1afa3a52db49212f0baa61b36885860edb09b5ac9d8f129f239961b`

没有把 `RTWholeBodyMPPI` 直接接入运行时。它只保存在只读参考归档中供算法参考；
正式教师是在本工程中重新实现、适配 Isaac Lab/708/尾箱/93→16 合同的 GPU MPPI。
旧 MoveIt 工程同样只作为迁移来源，新工程运行时不依赖它。

## 合同与资产

- 输入：float32 `obs`，固定 `[1,93]`
- 输出：float32 `actions`，固定 `[1,16]`
- 控制频率：50 Hz
- 93D 字段保持已冻结 legacy 语义，包括 fixed-first-frame motion prefix、
  shield 后上一拍 raw action 和末尾 constant-zero。
- wheel position observation 槽置零；模型图中 `actions[12:16]` 对任意输入精确为零。
- 708 reference：
  `vendor/robot_lab/data/Motions/pcbc_lateral_708/trajectory_trotting_acc_f015.npz`
- 尾箱 USD/STL、机器人 URDF 与全部 mesh 都在本目录 `vendor/robot_lab/` 内。
- 独立性审计：`reports/00_audit_state.json`（通过）。

## Expert

选择 MPPI 作为唯一正式专家，DWMPC 未接入。正式未见 seed 门禁为
50/50，teacher-valid
rate=1.000000，wheel hard-zero=
True。完整证据见
`reports/13_mppi_formal_gate50_v3_gate.json`。

## 数据与训练

聚合数据共 166 个 episode、43651 帧；
episode split 为 `{"test": 7, "train": 121, "validation": 38}`。
表中的 solve time 是 shard 中 MPPI 求解计时，不是端到端墙钟时间。

| 阶段 | episodes | frames | student 执行帧 | MPPI 有效标签帧 | 平均 solve ms |
|---|---:|---:|---:|---:|---:|
| R0 | 70 | 21000 | 0 | 21000 | 562.168 |
| R1 | 26 | 7654 | 3454 | 7654 | 566.480 |
| R2 | 44 | 7197 | 2397 | 7197 | 567.199 |
| R3 | 26 | 7800 | 4800 | 7800 | 594.208 |

BC 与 R1、R2、R3 均真实训练和闭环运行。标准 R3 checkpoint 在固定闭环 seed 上
退化为 0/50 全时域；因此没有按离线 RMSE 盲选它。最终选择从稳定 R2 checkpoint
以 `1e-6` 学习率吸收 R3 数据的一轮保守微调：
`checkpoints/dagger_r3_conservative/student_best_checkpoint.pt`。其 PyTorch 闭环准入
为 50/50 全时域，
shield intervention rate=0.000000。
选择证据见 `reports/deployment_checkpoint_selection.json`。

## 导出与闭环复评

- `exported/policy.pt`
- `exported/policy.onnx`
- `exported/export_manifest.json`
- `exported/golden_io.npz`
- `exported/SHA256SUMS`

256 个 golden observations 上 TorchScript/eager 最大误差
`0.000e+00`，
ONNX/eager 最大误差 `7.153e-07`、
平均误差 `6.705e-08`。
ONNX CPU p95 为 `0.011918 ms`，满足 50 Hz；
三种后端 wheel 均 exact zero。

最终根目录 ONNX 与闭环验证的文件 hash 完全相同。ONNX 在 R3
`light_delay_platform_pose` 场景为
50/50 全时域，
shield intervention rate=0.000000。

## 尚未通过的生产指标

当前 ONNX 的严格 reference-tracking success 为
0/50。虽然没有提前终止，但
base position、wheel center、contact schedule 与 box-local x drift 仍超阈值。
因此没有继续运行每条 reference 100 回合的最终大门禁，也没有声称达到 95%
overall / 90% per-reference 的生产标准。详见 `reports/failure_analysis.md`。

## 部署集成

公司运行时只需 ONNX Runtime 和本报告冻结的 observation/action adapter，不需要
Isaac Lab、MPPI、Python、MoveIt 或 `RTWholeBodyMPPI`。每个 20 ms 控制周期构造
一个 `[1,93]` float32 `obs`，调用 `policy.onnx`，读取 `[1,16]` float32
`actions`，再按 manifest 的 joint mapping、offset 与 scale 送入现有控制层。
上一拍 observation action 必须是 safety/shield 后真正执行的 raw action。
