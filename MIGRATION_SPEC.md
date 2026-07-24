# 给远程 4090 电脑上 Codex 的完整执行提示词

> **运行时勘误（2026-07-24）：** 本文后面把
> `mat[..., :2]` 描述为“取矩阵前两行”，这是错误的。PyTorch/NumPy
> 的这个切片作用在最后一维，部署端与 Isaac 任务实际取的是前两列，
> C-order 展平后为 `[R00, R01, R10, R11, R20, R21]`。本仓库以
> `configs/deployment_contract.yaml` 和部署端只读源码证据为准，旧的
> two-row 数据与模型必须经过 v2 columns 迁移，不能直接用于 key 7。

你是这项迁移工作的主执行 Codex。请直接检查代码、实现、运行测试、生成数据、训练、评估并导出模型，不要只给我一份方案后停止。除非出现无法通过现有文件和只读检查解决的真实阻塞，否则请自行做合理判断并持续推进。

## 一、机器上已有内容

远程工作区中会有：

1. 一个完整的 `robot_move_it` 项目文件夹（实际名称也可能是 `robot_move_lt`，请自动查找，不要硬编码目录名）。它包含：
   - 当前 Isaac Lab 训练工程；
   - pcbC 轮式四足机器人 URDF/USD/mesh；
   - 汽车尾箱/高台模型；
   - 横移参考轨迹 NPZ；
   - 当前任务、观测拼接、动作缩放、训练和 PT/ONNX 导出代码；
   - 可能还包含未提交到 Git 的文件，因此必须以磁盘上的完整文件夹为准，而不能只看 `git ls-files`。
2. 我已经克隆的 `RL_augmented_MPC`：
   - `https://github.com/DRCL-USC/RL_augmented_MPC.git`
   - 这个仓库用于参考“MPC 保证平衡、学习模块只承担小范围修正”的结构，不要假设它能不经适配直接运行 pcbC。
   - 如需核对本文所指的两个 expert 后端，只使用作者官方仓库：
     - DWMPC：`https://github.com/iit-DLSLab/DWMPC`
     - Whole-Body MPPI：`https://github.com/jrapudg/RTWholeBodyMPPI`
   - 不要为了“代码看起来相似”使用未核实的第三方复现。
3. NVIDIA RTX 4090。
4. 已经能运行原工程的 conda + Isaac Lab 环境。

先自动定位这两个目录并记录绝对路径。先读取所有适用的 `AGENTS.md`、README、部署 manifest 和环境说明。不要重新创建 conda 环境，不要主动升级 CUDA、PyTorch、Isaac Sim、Isaac Lab、rsl_rl 或 ONNX Runtime；先使用现有可运行环境，并把版本写进审计报告。

## 二、最终目标

在机器人项目根目录中新建独立文件夹：

```text
lateral_mppi_dagger/
```

在其中实现新的训练链路：

```text
708 横移参考轨迹
        ↓
DWMPC/WBC 或 Whole-Body MPPI 专家
        ↓
专家闭环轨迹数据
        ↓
Behavior Cloning 预训练
        ↓
DAgger：student 访问状态上持续向 expert 查询
        ↓
93→16 的部署 student
        ↓
训练 checkpoint + TorchScript policy.pt + policy.onnx
```

最终真正交付的是 student 神经网络。MPC/MPPI 只在专家数据生成和 DAgger 查询阶段使用，最终公司运行时不依赖 MPC、acados、Pinocchio、Isaac Lab 或 Python。

不要重新做一套端到端 PPO/SAC。除非 BC+DAgger 已完整工作、且有对照实验说明必要性，否则不要加入 RL fine-tuning。

## 三、工作和 Git 约束

1. 开始前执行只读审计：
   - `git status --short`
   - 当前分支与 commit；
   - GPU、驱动、Python、PyTorch、Isaac Sim、Isaac Lab、rsl_rl、onnx、onnxruntime 版本；
   - 所有轨迹、URDF/USD/STL 和部署 manifest 的 SHA256。
2. 现有工作树可能很脏，所有已有修改和未跟踪文件都属于用户，禁止覆盖、删除、reset 或 checkout。
3. 新代码优先全部放入 `lateral_mppi_dagger/`。只有确实需要注册环境或暴露现有接口时，才对旧工程做最小改动，并逐项记录原因。
4. 原始 NPZ、URDF、USD、STL、旧模型不得改写。
5. 大型 dataset、checkpoint、视频和导出模型默认加入 `.gitignore`；代码、配置、测试、报告和小型 golden data 可以提交。
6. 建议创建 `codex/mppi-dagger-lateral` 分支，并按阶段做小提交，但不得处理或提交用户原有的无关修改。
7. 所有长任务必须：
   - 支持 seed；
   - 支持 checkpoint/resume；
   - 支持限定 episode 数的 smoke test；
   - 输出结构化日志；
   - 失败时保存配置与失败状态；
   - 不允许训练数小时后才发现 shape、joint order 或资产路径错误。
8. 外部论文仓库只能作为参考或显式 third-party dependency。先检查 LICENSE；没有明确许可或与公司使用冲突时，不要复制源码进新目录，只实现接口和必要算法。
9. 权限范围仅包括本机仿真、离线数据生成、训练、评估和模型导出。未经我另行明确授权，
   不得连接、下发命令或控制真实机器人，也不得自动上传公司资产、轨迹、模型或日志。
10. 在 `configs/resource_budget.yaml` 固化资源上限。默认：
    - 单个 smoke job 最多 20 分钟；
    - 单次正式采集或训练 job 最多 6 GPU-hours；
    - 同一失败配置最多自动重试 2 次；
    - DAgger 最多到 R5；
    - 新数据/模型总磁盘预算默认 100 GB。
    可以基于实测吞吐调整，但必须先记录理由。预算耗尽不等于完成；应保存 resume 状态并
    报告当前结果和下一条恢复命令。

## 四、第一阶段：先冻结公司部署合同

在写 teacher 或训练代码前，创建：

```text
lateral_mppi_dagger/reports/00_contract_audit.md
lateral_mppi_dagger/configs/deployment_contract.yaml
lateral_mppi_dagger/tests/test_contract.py
```

合同的权威顺序是：

1. 公司实际部署代码及现有可用 ONNX 的真实 graph metadata；
2. 当前工程的 observation builder、action manager 和 exporter；
3. deployment manifest；
4. 本提示词中的已知值。

如四者冲突，不得猜测或静默修改；应保留公司运行时 ABI，增加显式 adapter，并在报告中列出冲突、采用的证据和最终选择。

### 4.1 已知的 93 维 observation ABI

默认应验证并保持以下 `float32`、无推理噪声、无外部归一化的顺序：

| Slice | 大小 | 当前合同含义 |
|---|---:|---|
| `0:16` | 16 | `motion_command` 的 joint position 部分；具体是固定首帧还是动态当前帧由 policy family 决定 |
| `16:32` | 16 | `motion_command` 的 joint velocity 部分；具体语义同上 |
| `32:38` | 6 | 将 desired/fixed reference anchor 表达到当前 robot anchor frame 的相对旋转 6D；严格按当前代码取矩阵前两行 |
| `38:41` | 3 | body frame base angular velocity |
| `41:57` | 16 | `joint_pos - default_joint_pos`；四个轮位置槽保留但置零 |
| `57:73` | 16 | `joint_vel - default_joint_vel`，单位 rad/s；包括四个轮子的真实速度 |
| `73:89` | 16 | 上一拍经过 wrapper 和 safety shield 后真正送入 action manager 的 executed raw action |
| `89:92` | 3 | velocity command；横移时为 `[0, target_vy, 0]` |
| `92:93` | 1 | legacy constant zero |

本次新建的 standalone student 目标维度必须严格为 93。最后一个标量继续为 0，不要为了加入 phase 而改写它。reference phase、接触计划、平台几何可以作为 expert 的 privileged input 和数据字段，但不能偷偷改变最终 student 的输入合同。任何真实 graph 为 92D 的旧 residual 模型都不得冒充 93D 模型，必须单独适配或重导出。

在 schema 和 export manifest 中必须同时保存：

```text
policy_family
policy/model hash
motion_prefix_semantics: fixed_first_frame | dynamic_reference
reference_file
reference SHA256
reference_index
first_frame_joint_pos
first_frame_joint_vel
```

两个模型即使都是 93D，只要 `motion_prefix_semantics` 不同，也视为 ABI 语义不兼容并拒绝混用。

相对旋转必须按当前实现精确复现：等价于
`subtract_frame_transforms(robot_anchor, reference_anchor)`，再执行
`R[..., :2].reshape(...)`。这里 `:2` 选的是矩阵前两行，不是前两列；C-order
flatten 为 `[R00,R01,R02,R10,R11,R12]`。写一个非单位旋转的 golden test；只测
identity 无法发现变换方向或行列写反。

当前横移实现很可能把 `0:32` 固定为“训练配置中 `motion_files[0]` 的第 0 帧 joint position + 全零 joint velocity”，动态 reference 只进入 critic/reward。这会让 student 面临部分可观测问题。固定前缀必须绑定到该文件的路径、SHA256、reference index 和首帧数值；hash 不匹配时拒绝运行，不能笼统地把所有模型都绑定到“708”。请做两件事：

1. 最终公司兼容模型必须先按真实 legacy builder 训练和验收；
2. 另外做一个小型诊断实验，比较：
   - legacy fixed-prefix 93D；
   - 在不改变 93 维 shape 的情况下，把原有 `0:32` 填为当前 `q_ref,dq_ref`。

第二种只能作为诊断/候选，除非确认公司运行时本来就逐帧填充当前 reference，或者我明确允许改变字段语义，否则不得把它冒充“合同不变”的最终模型。如果 legacy 版本失败而 dynamic-reference 版本明显成功，要用数据证明这是观测混叠造成的，并报告所需的最小部署改动。

在正式 expert 数据采集前增加“可观测性 Gate”：

1. 从不同 reference phase/contact mode 收集 legacy `obs93` 和 expert label；
2. 对归一化 observation 做近邻检索；
3. 找出 observation 几乎相同、但 expert action/contact mode 明显冲突的样本；
4. 分别统计 fixed-prefix 与 dynamic-reference 的冲突率并保存案例；
5. 如果 fixed-prefix 存在系统性的不可辨识映射，而 dynamic-reference 明显消除冲突，
   则把它判定为部署 ABI/语义阻塞，暂停大规模 BC/DAgger，不允许靠增加网络和训练
   时长掩盖。完成其余不受阻工作后，向我报告最小部署改动。

所有训练、仿真、导出验证必须调用同一个 `Obs93Builder` 或由同一份 schema 生成；禁止在 environment、dataset collector、ONNX 验证器中分别手写三套拼接逻辑。

`previous_action` 的 standalone student 语义冻结为
`prev_executed_raw_action16_after_wrapper_and_safety`，reset 首帧全 0。它不是
student proposal、未执行的 teacher label 或 residual raw。若读取旧 baseline+residual
模型，必须另按其 manifest 区分 previous final raw 与 previous residual raw，不能共用一
个历史 buffer。

训练时可以给观测加噪声，但 dataset 中必须同时记录无噪声 `obs93_clean`；公司推理端绝不能复现随机噪声。

### 4.2 已知 joint/action 顺序

policy 顺序是按关节类型分组，不是逐腿排列：

```text
0  FL_hip_joint
1  FR_hip_joint
2  RL_hip_joint
3  RR_hip_joint
4  FL_thigh_joint
5  FR_thigh_joint
6  RL_thigh_joint
7  RR_thigh_joint
8  FL_calf_joint
9  FR_calf_joint
10 RL_calf_joint
11 RR_calf_joint
12 FL_foot_joint
13 FR_foot_joint
14 RL_foot_joint
15 RR_foot_joint
```

如果公司内部使用逐腿顺序，先从现有 manifest 核对映射。已知 policy→runtime scatter map 候选为：

```text
[0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11, 12, 13, 14, 15]
```

其运算定义必须明确写成：

```text
runtime[map[p]] = policy[p]
```

若实现成 gather，等价式是：

```text
runtime = policy[[0,4,8,1,5,9,2,6,10,3,7,11,12,13,14,15]]
```

必须用 `0..15` 唯一哨兵值写 round-trip 单元测试，禁止靠视觉判断 joint mapping。

### 4.3 action ABI

最终 ONNX/TorchScript 输出必须是 16 维 raw action：

```text
0:4   四个 hip position raw action
4:8   四个 thigh position raw action
8:12  四个 calf position raw action
12:16 四个 wheel velocity raw action
```

物理映射应从当前工程重新核实，已知候选是：

```text
hip target   = q_action_offset_runtime + 0.125 * raw
thigh target = q_action_offset_runtime + 0.25  * raw
calf target  = q_action_offset_runtime + 0.25  * raw
wheel qd target = qd_action_offset_runtime + 5.0 * raw
```

用户描述了“轮子置 0”，但现有代码中至少存在两种不同含义：

```text
wheel_position_observation_zero:
  轮关节位置 observation 的四个槽为 0；现有横移代码已这样做。

wheel_action_mode:
  learned_velocity:
    actor 仍预测四个 wheel velocity raw action。
  hard_zero:
    actor 不预测轮动作，部署图内拼接四个常数 0。
```

Gate 0 必须通过公司控制器、当前有效模型 graph/manifest 和用户要求，把
`wheel_action_mode` 明确冻结进 `deployment_contract.yaml`，不能把“轮位置观测置 0”
自动推导成“轮动作置 0”。如果现有材料仍无法裁决，只针对这一点向我提出一个简短问题。
若最终选择 `hard_zero`，报告中要明确写成“有意新增/冻结的 wheel action 语义”，不能
声称这是旧 lateral actor 原本就具备的行为。

若最终确认 `wheel_action_mode=hard_zero`，student 内部只预测 12 个腿动作，部署
wrapper 在图内拼接四个常数 0：

```python
leg_action = actor(obs93)            # [..., 12]
wheel_action = zeros(..., 4)
action16 = concat(leg_action, wheel_action)
```

若最终确认 `wheel_action_mode=learned_velocity`，student 输出完整 16 维，四个轮输出按
现有速度 action scale 执行，禁止再拼 0。

在 `hard_zero` 模式下，必须保证对任意输入，Torch eager、TorchScript 和 ONNX 的
`action[...,12:16]` 都精确为 0，而不是依靠 loss 逼近 0；previous executed action
的四个轮槽也必须为 0。在两种模式下，observation 中的实际四轮 joint velocity 都保
留，用于感知被动滑转。

`Action16Adapter` 和 student wrapper 应支持这两个显式模式，但一个导出模型只能绑定
其中一个模式，必须写入 ONNX metadata、manifest、文件 hash 和 golden tests。

### 4.4 导出 ABI

最终公司文件默认保持现有 exporter 约定：

```text
policy.onnx
  input name:  obs
  input shape: [1, 93]
  output name: actions
  output shape: [1, 16]
  dtype: float32
  opset: 18（若现有部署只支持其他 opset，以实际运行时为准）

policy.pt
  自包含 TorchScript
  forward(x: Tensor) -> Tensor
  [N,93] -> [N,16]
```

不要把训练 checkpoint 当成部署 `policy.pt`。最终同时输出：

```text
checkpoints/student_best_checkpoint.pt
exported/policy.pt
exported/policy.onnx
exported/export_manifest.json
exported/golden_io.npz
exported/SHA256SUMS
```

如果为了训练方便额外导出 dynamic-batch ONNX，文件名必须不同；公司主文件仍按真实 ABI 导出。

如果公司主 ONNX 冻结为 `[1,93]→[1,16]`、`dynamic_axes={}`，golden parity 必须逐条
以 `[1,93]` 调用，不能直接喂 `[K,93]`。主 ONNX 闭环使用 `num_envs=1`，或由 adapter
对每个 env 显式逐条调用。不得为了让多 env 测试方便而偷偷把公司主文件改为动态 batch。

旧 `residual_deploy_bundle` 可能存在 92/93 维冲突，且 manifest 与真实 candidate ONNX 未必一致。必须在远程机器上用 ONNX Runtime 读取真实 graph 输入输出，不能只相信 YAML。新 standalone student 的目标是 93D；真实输入为 92D 的旧模型必须单独命名、单独 adapter 或重导出，禁止自动补一个 0 后假装兼容。

## 五、708 reference 和初始姿态约束

“708”是数据目录/数据版本名，不是第 708 帧。不要使用 `motion_start_frame=708`。已知参考每条只有 332 帧。

先自动查找类似：

```text
双足搭车侧向移动数据708/
```

NPZ 预期 schema：

```text
fps                 int64   [1]        预期 50
joint_pos           float32 [332,16]
joint_vel           float32 [332,16]
body_pos_w          float32 [332,17,3]
body_quat_w         float32 [332,17,4]
body_lin_vel_w      float32 [332,17,3]
body_ang_vel_w      float32 [332,17,3]
```

不要只检查 shape，还要检查：

- keys 完整；
- 所有值 finite；
- quaternion 归一化；
- `fps == 1 / (sim.dt × decimation)`，或存在经过测试的显式重采样；
- NPZ joint/body 顺序与机器人模型一致；
- 多条轨迹的第 0 帧是否一致；
- 参考时长按 `(frames-1)/fps` 计算；
- 当前代码是否每个 control step 简单 `frame += 1`。如果是，则必须保持 50 Hz，否则会静默错相。

已知 708 第 0 帧候选值如下，必须从远程 NPZ 重新计算并用单元测试核对，不能只复制常量：

```text
root position:
[0, 0, 0.74180591]

root quaternion（NPZ/Isaac 边界预期 wxyz，Gate 0 必须实测冻结）:
[0.7514101, -0.006758712, -0.6597562, -0.007681652]

joint position，按 policy type-grouped 顺序:
[0, 0, 0, 0,
 0.610812, 0.610812, 0.698123, 0.698123,
 -0.785317, -0.785317, 1.0472, 1.0472,
 0, 0, 0, 0]
```

把下面两个量分开审计、命名和保存：

```text
q_reset_ref0:
  reference 第 0 帧，用于 episode reset。

q_action_offset_runtime:
  公司 action 解码公式中的 position offset。
```

当前 708 Isaac task 里它们可能相等，但旧 deployment manifest 或公司 runtime 可能不同。
只有 hash/数值审计证明相等时才能合并；不相等时 `Action16Adapter` 必须使用
`q_action_offset_runtime` 计算 raw label，同时 reset 仍使用 `q_reset_ref0`。否则全部
teacher 标签会带系统性偏置。

Gate 0 后还必须把所有 7D pose 的布局冻结进 schema，例如：

```text
base_pose_w:       [x,y,z,qw,qx,qy,qz]
wheel_body_pose_w: [x,y,z,qw,qx,qy,qz]
quaternion_order:  wxyz
```

每个使用 xyzw 的外部 backend（例如部分 Pinocchio/MuJoCo 接口）都要通过显式
`wxyz_to_xyzw/xyzw_to_wxyz` adapter，写 identity、90° rotation 和 round-trip test。

如果只有一个 NPZ，就只实现和评估这条轨迹，不得假装支持双向、多速度。如果存在多条：

```text
trajectory_trotting_acc_f015.npz -> -0.15 m/s
trajectory_trotting_acc_f01.npz  -> -0.10 m/s
trajectory_trotting_acc_f005.npz -> -0.05 m/s
trajectory_trotting_acc_005.npz  -> +0.05 m/s
trajectory_trotting_acc_01.npz   -> +0.10 m/s
trajectory_trotting_acc_015.npz  -> +0.15 m/s
```

必须从当前任务代码核实正负号，不得只按文件名猜。

当前 708 轨迹可能是未跟踪文件，车尾箱 USD 也可能被 `.gitignore` 忽略。因此远程完整文件夹是必要输入。启动时明确检查以下资产实际存在且可加载：

```text
pcbC URDF 候选:
source/robot_lab/data/Robots/pcbC/pcb_v2_description_0.88/urdf/pcb_v88.urdf

车尾箱 USD 候选:
source/robot_lab/data/Robots/pcbC/pcb_v2_description_0.88/mesh/530X6U_simple.usd
```

已知 708 场景候选参数为车尾箱 position `(4.851,0,0)`、scale `(1,6,1)`；必须从实际环境类再次核对。不要用普通平地或简单 box 替换后声称任务成功。

不要盲用旧 run 保存的 `env.yaml`，其中可能保存另一台机器的绝对 NPZ 路径。路径应由 CLI/config 注入并在 run manifest 中保存相对路径和 hash。

## 六、新文件夹建议结构

至少实现：

```text
lateral_mppi_dagger/
├── README.md
├── configs/
│   ├── deployment_contract.yaml
│   ├── reference_708.yaml
│   ├── expert_dwmpc.yaml
│   ├── expert_mppi.yaml
│   ├── student.yaml
│   └── dagger.yaml
├── src/lateral_mppi_dagger/
│   ├── contract/
│   │   ├── obs93.py
│   │   ├── action16.py
│   │   └── joint_mapping.py
│   ├── reference/
│   │   ├── loader.py
│   │   ├── interpolation.py
│   │   └── contact_schedule.py
│   ├── env/
│   │   ├── isaac_adapter.py
│   │   ├── state_snapshot.py
│   │   └── safety_metrics.py
│   ├── expert/
│   │   ├── base.py
│   │   ├── dwmpc_expert.py
│   │   ├── mppi_expert.py
│   │   ├── reference_wbc.py
│   │   └── safety_fallback.py
│   ├── data/
│   │   ├── schema.py
│   │   ├── collector.py
│   │   └── dataset.py
│   ├── student/
│   │   ├── model.py
│   │   ├── losses.py
│   │   └── trainer.py
│   ├── dagger/
│   │   └── runner.py
│   ├── export/
│   │   ├── exporter.py
│   │   └── validator.py
│   └── evaluation/
│       ├── evaluator.py
│       └── plots.py
├── scripts/
│   ├── audit_project.py
│   ├── validate_reference.py
│   ├── replay_reference.py
│   ├── smoke_test_expert.py
│   ├── collect_expert.py
│   ├── train_bc.py
│   ├── run_dagger.py
│   ├── evaluate.py
│   ├── export_policy.py
│   └── validate_export.py
├── tests/
├── reports/
├── datasets/        # gitignored
├── checkpoints/     # gitignored
└── exported/        # gitignored，最终产物另行归档
```

可以根据实际工程适当调整，但不能把所有逻辑塞进一个脚本。
`dwmpc_expert.py` 与 `mppi_expert.py` 不要求同时成为完整生产实现：feasibility 后未被选中
的后端可以只保留带明确 `NotImplemented` 的接口、实验脚本和调查报告，不能留下会被误
认为可用的半成品。

## 七、Expert 统一接口

先实现与算法无关的接口：

```python
@dataclass
class ExpertRequest:
    dt: float
    base_pose_w: Tensor       # [7]
    base_twist_w: Tensor      # [6]
    q: Tensor                 # [16]
    dq: Tensor                # [16]
    wheel_body_pose_w: Tensor # [4,7]
    wheel_body_twist_w: Tensor
    contact_force_w: Tensor   # [4,3]
    ref_id: int
    ref_frame: int
    ref_window: dict
    target_vy: float
    desired_contact: Tensor   # [4] bool
    platform_geometry: object

@dataclass
class ExpertReply:
    valid: bool
    q_des_leg: Tensor         # [12]，policy type-grouped 顺序
    wheel_vel_des: Tensor | None  # [4]；hard_zero 时为全 0
    action16: Tensor          # [16]，遵循已冻结的 wheel_action_mode
    tau_ff_leg: Tensor | None
    predicted_grf: Tensor | None
    solve_ms: float
    solver_status: str
    safety_margin: float
    source: str               # dwmpc / mppi / reference_wbc / fallback
    failure_code: str
```

接口形式：

```python
expert.reset(episode_metadata)
reply = expert.act(request)
```

所有 expert 先输出物理量 `q_des_leg` 和已冻结 wheel mode 所需的
`wheel_vel_des`，再经过唯一的 `Action16Adapter` 转成公司 raw action：

```text
a_hip   = (q_des_hip   - q_action_offset_runtime_hip)   / 0.125
a_thigh = (q_des_thigh - q_action_offset_runtime_thigh) / 0.25
a_calf  = (q_des_calf  - q_action_offset_runtime_calf)  / 0.25
a_wheel = 0                                      # hard_zero
a_wheel = (qd_des_wheel - qd_action_offset_runtime) / 5 # learned_velocity
```

统一执行 joint limit、速度限制、action rate 限制和 finite 检查。不要直接用 torque teacher 监督 joint-position student；如果 DWMPC 输出 torque/GRF，必须经过同一个 WBC/IK/PD target 层得到 `q_des_leg`。

## 八、Expert 实施策略

不要一开始同时重写两套完整控制器。目标是尽快获得一个可靠、可重复、能在真实任务环境闭环运行的 expert。

### 8.1 推荐顺序

1. 先实现 `ReferenceWBCExpert`：
   - 用 708 当前/未来 reference 作为 nominal；
   - 正确完成 q、dq、base、wheel-body 和 contact 数据映射；
   - 输出 12 维腿 `q_des`，轮通道遵循已冻结的 `wheel_action_mode`；
   - 目的不是最终方案，而是验证 action/obs/joint mapping 和参考回放。
2. 对 DWMPC 做小型 feasibility spike：
   - 如果可以在当前 URDF、接触几何和依赖环境中稳定求解，把 DWMPC/WBC 作为 nominal expert；
   - 接触时序来自 reference 或经过验证的接触提取器；
   - 不得假设四个轮足在同一高度。
3. 如果 DWMPC 集成代价过高或 nominal 解频繁不可行，切换为 reference-centred Whole-Body MPPI：
   - 直接在现有 Isaac Lab 动作空间优化未来 12 维腿 raw-action residual；
   - `hard_zero` 模式下四个轮 action 在采样、rollout 和执行中始终硬置零；
   - `learned_velocity` 模式下按现有四轮速度 action 共同优化或由 reference/WBC 确定；
   - 使用 4090 上的并行 rollout；
   - 不要求可微分。
4. 如果两者都可用：
   - DWMPC/WBC 负责 nominal；
   - MPPI 只处理困难扰动/恢复状态；
   - 必须实现确定性仲裁，例如“DWMPC valid 且安全时使用 DWMPC，否则才允许 MPPI”；
   - 只有仲裁后的唯一动作可以进入 imitation label；
   - 其他 expert 候选、两者 action disagreement 和失败原因只作诊断记录。

两个 feasibility spike 各自默认最多运行 25 个闭环 smoke episode 或消耗 2 GPU-hours
（不含正常代码实现时间）；随后必须根据成功率、valid rate、solve time 和集成复杂度选
定一个正式 label provider。不要把“完整实现两套 expert”当作完成条件。若调整这个预算，
在配置和报告中写明理由。

`RL_augmented_MPC` 主要借鉴：

- MPC 与学习模块边界；
- stance balance 与 swing/foot-placement residual 的组织；
- 状态和 action adapter；
- C++/Python 运行接口。

不要把 A1 的 joint order、质量、默认姿态、足端名、接触计划或 PyBullet 环境直接带进 pcbC。

### 8.2 DWMPC/WBC 必须建模的任务约束

- 保持搭车基座 pitch/roll/height 和相对尾箱 x 位置；
- 跟踪 reference 的横向 y 进度；
- 轮足接触面可能分属不同高度；
- 支撑轮足的接触、摩擦锥和法向力；
- 横移不能依赖不符合轮轴方向的四轮侧向滚动；
- `hard_zero` 只表示 wheel velocity servo 的目标为 0，不等于机械锁死，也不保证实际轮速为 0；expert 必须显式建模轮关节/速度伺服，或使用经 Isaac 回放验证的等效约束；
- `learned_velocity` 模式必须满足轮轴方向和非完整约束；
- 摆动轮足按 reference 抬起、横向重定位、落下；
- 关节位置/速度/力矩限制；
- 前轮距尾箱边缘的安全 margin；
- action/target 连续性；
- 失败时可解释的 solver status。

接触计划优先使用 reference 中已有信息。如果 NPZ 没有 contact label，可依据轮心位置、速度、与各接触面的几何距离生成候选，再用 Isaac 接触力回放验证并生成诊断图。禁止只用统一 z 阈值，因为前后轮足可能位于不同高度。

### 8.3 MPPI 实现要求

MPPI 应围绕 reference nominal 采样，而不是从随机动作搜索整段运动：

```text
nominal action sequence
  = future q_ref 经 Action16Adapter 得到的 12D raw leg action

candidate
  = nominal + temporally-correlated residual noise
```

配置化参数至少包括：

- control frequency；
- horizon；
- samples；
- optimization iterations；
- temperature/lambda；
- per-joint noise std；
- temporal smoothing；
- action bounds/rate bounds；
- warm start；
- rollout batch grouping；
- seed。

先用很小参数完成 smoke test，再 profile 4090，最后逐步提高。不得在代码里写死一个看似论文默认、但没有在当前机器人上测过的参数。

cost 至少包含：

- base relative pose/reference pose；
- base pitch/roll/yaw/height；
- joint q/dq reference tracking；
- 四个 wheel body 的 reference 相对位置；
- lateral y progress/velocity；
- box-local x drift；
- wheel slip；
- desired/measured contact mismatch；
- edge/drop margin；
- action magnitude、action rate、joint acceleration；
- torque/limit violation；
- termination/fall/illegal contact 的大终端代价；
- horizon terminal cost。

每个 cost 分量单独记录，不允许只有一个总 cost，避免再次陷入“只能盲调权重”。

如果使用 Isaac Lab 并行环境做 MPPI rollout：

- 明确实现 seed-state 到 rollout clones 的 state copy；
- 处理 env origin 坐标偏移；
- 每轮 rollout 后恢复/覆盖所有 simulator state 和必要 buffer；
- 不把 rollout clone 的 previous action、reference frame 或 contact history串到别的 seed；
- 写 state-copy round-trip test；
- 首先验证相同 state + 相同 action 序列产生相同结果。

### 8.4 Expert 门槛

不允许在 expert 本身还不可靠时开始大规模 BC。最低门槛：

- 单轨迹 nominal 环境完整闭环成功；
- action 和状态均无 NaN/Inf；
- wheel 输出严格符合已冻结 mode；`hard_zero` 时精确为 0；
- 无 joint mapping 错位；
- 无掉落和非法接触；
- solver valid rate、solve time、失败码有统计；
- 至少在未见 seed 上重复评估；
- reference、base pose、wheel center 和接触曲线已画图；
- 生成视频确认不是通过异常侧滑或卡模型“成功”。

使用保存到配置中的固定 50 个未见 nominal seeds 做 expert Gate。至少 48/50 个完整
episode 成功、无 NaN/非法接触、solver valid rate 达到配置阈值后，才能开始正式数据采
集；达不到时先修 expert，不要指望 DAgger 修复错误标签。

## 九、数据采集格式

使用每 episode 一个压缩 NPZ shard，加一个全局 `manifest.jsonl`。按 episode/seed/scenario 划分 train/validation/test，禁止先打散帧再随机切分造成泄漏。

每个 episode 至少保存：

```text
step_id                  int32   [T]
sim_time                 float64 [T]
obs93_clean              float32 [T,93]
obs93_train              float32 [T,93]
next_obs93_clean         float32 [T,93]
teacher_action16         float32 [T,16]
student_action16         float32 [T,16]
pre_shield_action16      float32 [T,16]
executed_action16        float32 [T,16]
action_clip_delta16      float32 [T,16]
teacher_q_des_leg        float32 [T,12]
teacher_valid            uint8   [T]
student_valid            uint8   [T]
label_source             uint8   [T]
behavior_policy          uint8   [T]
teacher_takeover         uint8   [T]
shield_intervened        uint8   [T]
ref_id                   int32   [T]
ref_frame                int32   [T]
phase                    float32 [T]       # 仅数据/teacher，不自动进入最终 obs
target_vy                float32 [T]
desired_contact          uint8   [T,4]
measured_contact         uint8   [T,4]
base_pose_w              float32 [T,7]
base_twist_w             float32 [T,6]
q                        float32 [T,16]
dq                       float32 [T,16]
wheel_body_pose_w        float32 [T,4,7]
contact_force_w          float32 [T,4,3]
solver_status            int16   [T]
solve_ms                 float32 [T]
safety_margin            float32 [T]
failure_code             int16   [T]
terminal                 uint8   [T]
termination_reason       int16   [T]
```

每一行的时序语义固定为：

```text
obs_t
  → teacher_action_t / student_action_t
  → behavior policy selection
  → pre_shield_action_t
  → safety shield / clipping
  → executed_action_t
  → simulator step
  → next_obs_{t+1}
```

所有 enum 的数字编码和 schema version 必须写入 manifest。无效的连续字段使用 NaN 并配
套 valid mask，禁止用数值 0 代表“无效”，因为 0 本身可能是合法动作。

episode metadata 至少记录：

```text
schema_version
git commit / dirty diff hash
expert backend + config hash
robot URDF/USD hash
trunk asset hash
trajectory hash
joint order
action scale
q_reset_ref0
q_action_offset_runtime
observation schema
wheel mode
pose/quaternion layout
control frequency
seed
friction/mass/delay/noise/platform parameters
direction/speed
success
```

invalid/infeasible expert reply、负 edge margin、严重 safety shield 冲突、NaN、经过大幅裁剪才可执行的动作，不得当作普通 imitation label。保留记录用于失败分析。

先做：

1. 1 episode smoke dataset；
2. 10–20 episode 小数据集，验证可加载、可训练、可复现；
3. 只有 end-to-end 流程通过后才生成正式数据。

正式数据量按实际 expert 速度决定，不要一开始盲目跑满。可以从数百个 6 秒 episode 起步，并提高接触切换、near-failure、恢复状态的采样权重。

## 十、Student 和 BC

默认网络保持简单、可部署、无循环状态：

```text
hard_zero:
  93 → 256 → 256 → 128 → 12
  wrapper: concat 4 exact zeros → 16

learned_velocity:
  93 → 256 → 256 → 128 → 16

activation: ELU
```

如果当前部署 ABI 是单帧 stateless，就不要换成 RNN/Transformer。已有 `previous_action` 提供一拍历史。

归一化规则：

- 只从 train episodes 统计；
- std 设置下限；
- 归一化作为模型 buffer 嵌入 TorchScript/ONNX 内部；
- 对公司外部仍接受原始 93D；
- 导出 manifest 保存 mean/std hash；
- 如果为严格复用旧模型而选择不归一化，也必须用实验说明，并保持三种推理后端一致。

推荐损失从简单开始：

```text
Huber(predicted raw leg action, teacher raw leg action)
+ 小权重的一阶 action smoothness
+ 可选的二阶 smoothness
```

不要同时监督互相不一致的 torque 和 position target。接触切换、摆腿起落、near-failure 样本可以提高权重。`teacher_valid=0` 不进入普通 imitation loss。

一阶/二阶 smoothness 只能在同一 episode 内、时间连续且 teacher/student valid 的窗口上
计算；遇到 reset、terminal、缺帧、reference 切换或 invalid label 必须 mask。禁止对随机
打散的 batch 相邻行直接计算时序 loss。

进入 DAgger 前至少满足：

- validation action RMSE 和 q_des RMSE 已报告；
- 固定 50 个未见 nominal seeds 中至少 40/50 个完整 episode 成功；
- 无 NaN、joint limit 错误和巨大动作跳变；
- wheel 输出符合已冻结 mode；`hard_zero` 时任意输入最后四维精确为 0；
- dataset split 无帧级泄漏。

如果 BC 开环 loss 很低但闭环立即失败，优先检查：

1. joint order；
2. q_reset_ref0 与 q_action_offset_runtime；
3. action scale；
4. previous action 语义；
5. reset/reference frame；
6. 50 Hz 时基；
7. legacy actor 看不到动态 phase 导致的 observation aliasing；
8. teacher label 是否经过了另一个未记录的 safety/controller 层。

不要第一反应扩大网络或增加训练 epoch。

## 十一、DAgger

DAgger 必须在 student 真正访问的状态上查询 expert，而不是只对原 reference 加高斯噪声。

每一步保存 teacher、student 和实际执行动作。不要默认逐元素线性插值 teacher 和
student；跨接触模式的中间关节目标可能两边都不可行。`beta` 表示按 episode 或至少按
完整接触 phase 选择 behavior policy 的概率，或者采用 student 执行、触发安全条件时
teacher 整体接管。必须记录选择和接管：

```python
use_teacher = sample_behavior_policy(beta, episode_or_contact_phase)
a_pre_shield = a_teacher if use_teacher else a_student
a_exec, shield_info = safety_shield(a_pre_shield)
if wheel_action_mode == "hard_zero":
    a_exec[..., 12:16] = 0
```

建议从以下 beta 计划开始，但只有当前轮安全指标通过后才能降低：

```text
R0: 1.00  纯 expert，BC
R1: 0.75  nominal + 小初始状态扰动
R2: 0.50  加轻度摩擦/质量/观测噪声
R3: 0.25  加轻度延迟和平台/初始姿态偏差
R4: 0.10  student 主导，重点收 near-failure
R5: 0.00  student 完全执行，expert 只打标签
```

不要让最初 nominal 数据淹没新 hard states。训练采样可从：

```text
30% 初始 expert
40% 最新一轮
30% 历史 hard/near-failure
```

开始，再按数据调整。

每轮：

1. 采集；
2. 数据完整性检查；
3. 聚合；
4. 从上一 best student 微调；
5. 固定验证集评估；
6. 闭环评估；
7. 保存 metrics、失败分类、视频和 best checkpoint；
8. 达到门槛才进入下一轮。

连续两轮性能不再提升且验收条件已满足时停止。若没有满足验收条件，不能只因预算耗尽就声称完成。

至少完成 R1、R2、R3 三轮真实 student-state 查询；最多运行到 R5。每轮在固定验证 seed
集上至少评估 50 个 episode。任一轮发生未分类的 NaN、state-copy error 或资产/contract
错误时不得继续降 beta。

## 十二、安全与失败回退

至少定义：

```text
TEACHER_INFEASIBLE
TEACHER_TIMEOUT
CONTACT_LOSS
WHEEL_SLIP
EDGE_MARGIN
BASE_ORIENTATION
JOINT_LIMIT
TORQUE_SATURATION
STUDENT_OOD
NAN_INF
STATE_COPY_ERROR
REFERENCE_TIMEBASE_ERROR
```

训练/采集中的回退顺序：

1. 四轮 raw action 立即归零；
2. 冻结横移 reference frame；
3. 若摆动轮足处于空中，优先安全下放到最近有效支撑面；
4. 最多短暂复用最后一个 feasible q_des；
5. 平滑进入最近稳定支撑姿态；
6. 无法恢复则终止；
7. 保存失败前至少 0.5 秒窗口、solver diagnostics 和视频。

安全回退动作和正常 expert 标签必须用 `label_source/failure_code` 区分。

## 十三、评估与验收

先从现有环境的 termination、edge margin 和安全阈值生成 `evaluation_protocol.yaml`。以下是最低要求，不得只看 imitation loss。

### 13.1 合同测试

- `obs.shape == [1,93]`；
- `action.shape == [1,16]`；
- dtype 为 float32；
- joint mapping round trip 正确；
- `obs[53:57]`（按最终核实索引）轮位置槽为 0；
- 任意输入 `action[12:16] == 0`；
- previous action 精确等于上一拍 shield 后 executed raw action；`hard_zero` 时其四轮槽为 0；
- 50 Hz 时 reference 一拍一帧，改变 dt 时有显式报错或重采样；
- 所有资产、reference 和 schema hash 进入 manifest。

### 13.2 导出数值一致性

在至少包含 nominal、扰动、near-failure 的 golden observations 上比较：

```text
PyTorch eager
TorchScript policy.pt
ONNX Runtime policy.onnx
```

公司主 ONNX 若为固定 batch 1，以上测试逐个样本 loop 执行；性能统计同时报告单样本
延迟和整个 golden set 总耗时。

目标：

```text
max_abs_error <= 1e-5
mean_abs_error <= 1e-6
```

如果实际算子/opset导致做不到，报告误差来源并设有证据的新阈值。还要测公司目标 CPU 上 ONNX 延迟，确认满足 50 Hz。

### 13.3 闭环任务指标

每个实际存在的 reference、未见 seed、nominal 和分层随机化分别报告：

- 完整 episode success rate；
- 相对 reference 的 base pose/orientation RMSE；
- joint q/dq tracking；
- 实际 y 位移与 reference y 位移；
- lateral velocity MAE；
- box-local x drift；
- wheel center reference error；
- 支撑轮足 slip；
- desired/measured contact mismatch；
- 最小 edge/drop margin；
- torque/joint/action saturation；
- action rate；
- solver valid rate 和 teacher solve time；
- termination reason 分布。

冻结最终评估 seeds 后，nominal student 的硬门槛为：

- 每个实际存在的 reference 至少评估 100 个 episode；
- overall success 至少 95%；
- 单个 reference/command bin 成功率至少 90%；
- student success 不得比同一组 seeds 上 expert 低超过 5 个百分点；
- 无掉落、非法接触和 NaN；
- 姿态、x drift、edge margin 满足现有任务安全阈值；
- wheel 命令始终符合已冻结 mode；`hard_zero` 时始终为 0；
- 行为视频确认是通过换接触/摆腿横移，而不是依靠异常侧滑、穿模或尾箱碰撞卡住。

鲁棒性必须分层加入：

1. nominal；
2. 轻度质量、摩擦、初始姿态和 0–1 拍延迟；
3. 更宽但仍符合公司部署分布的随机化。

不要一开始开启所有宽随机化，否则无法判断 expert、student、contact 或 mapping 哪一部分出错。

### 13.4 对照实验

至少保留：

- reference-only/WBC；
- expert；
- BC；
- 每轮 DAgger best；
- 最终 PT；
- 最终 ONNX。

在相同 seeds/config 下比较。若 MPPI 与 DWMPC 都尝试过，报告成功率、solve time、实现复杂度和失败类型，不要只凭主观选择。

## 十四、必须提供的可运行命令

README 中给出从空数据到最终模型的真实命令，不得使用伪命令。至少包含：

```text
1. 环境/资产/合同审计
2. 708 NPZ 验证
3. reference replay
4. 单 episode expert smoke test
5. 小数据采集
6. BC smoke train
7. BC 完整训练与 resume
8. 单轮 DAgger smoke
9. 多轮 DAgger 与 resume
10. 闭环 evaluation
11. 导出 policy.pt / policy.onnx
12. ONNX Runtime golden parity
13. 生成最终报告
```

不要沿用文档里未经核实的 task id。当前工程可能注册了不存在的 `stable-v2`、
`deploy-obs` 或 `fixed-slope` 类。必须先用 `list_envs` 和 Python import 检查，再通过
现有 train/play CLI 的 `--num_envs=1` 做 smoke test；若程序内创建，则先用本工程的
`parse_env_cfg(..., num_envs=1)` 设置 `env_cfg.scene.num_envs`，再执行
`gym.make(task_id, cfg=env_cfg)`，不要调用本工程不支持的
`gym.make(..., num_envs=1)`。708 的候选可运行入口可能包括：

```text
RobotLab-Isaac-BeyondMimic-Flat-pcbC-v1-lateral-bipedal-stand-v0
RobotLab-Isaac-BeyondMimic-Flat-pcbC-v1-lateral-bipedal-stand-stable-v0
RobotLab-Isaac-BeyondMimic-Flat-pcbC-v1-lateral-bipedal-stand-stable-v1
RobotLab-Isaac-BeyondMimic-Flat-pcbC-v1-lateral-bipedal-stand-stable-v3
```

但必须以远程文件实际能实例化为准。

导出旧 RSL-RL checkpoint 时，现有 `play.py` 可能先创建完整环境再导出，因此导出机也需要 NPZ、USD 和 Isaac 环境。新 student exporter 应尽量支持不启动 Isaac Sim 的纯 PyTorch 离线导出和 ONNX 验证。

## 十五、阶段性执行顺序

严格按以下 gate 推进：

### Gate 0：资产和合同

- 完成 `00_contract_audit.md`；
- 所有资产存在；
- 93/16 schema、joint order、action scale、ONNX I/O 已用测试冻结；
- 708 第 0 帧和 50 Hz 时基已验证。

### Gate 1：reference 回放

- 在真实尾箱环境正确 reset；
- reference 第 0 帧无跳变；
- 逐帧回放的 joint/body/contact 曲线合理；
- 没有 joint/body 顺序错位。

### Gate 2：expert

- 单轨迹、单环境闭环；
- 再做多 seed；
- nominal expert 通过安全门槛；
- 记录 solver/cost/失败数据。

### Gate 3：BC

- 小数据端到端通过；
- 正式 expert dataset；
- BC 闭环评估；
- 不通过先查合同和 covariate shift。

### Gate 4：DAgger

- 至少完成 R1–R3 三轮真实 student-state 查询，最多到 R5；
- 每轮有固定验证与闭环结果；
- 失败样本进入下一轮而不是被丢弃。

### Gate 5：导出

- 训练 checkpoint、TorchScript、ONNX 分开；
- shape/name/opset 符合公司 ABI；
- golden parity、wheel hard-zero、CPU latency 全部通过；
- 用 ONNX 在相同闭环环境复评，不只做单步数值测试。

### Gate 6：最终交付

生成：

```text
lateral_mppi_dagger/exported/policy.pt
lateral_mppi_dagger/exported/policy.onnx
lateral_mppi_dagger/exported/export_manifest.json
lateral_mppi_dagger/exported/golden_io.npz
lateral_mppi_dagger/exported/SHA256SUMS
lateral_mppi_dagger/reports/final_report.md
lateral_mppi_dagger/reports/final_metrics.json
lateral_mppi_dagger/reports/failure_analysis.md
```

`final_report.md` 必须说明：

- 最终选择 DWMPC、MPPI 或组合的原因；
- 是否完全保持 legacy 93D 字段语义；
- wheel “观测置零”和“动作置零”的最终处理；
- 708 reference 和初始姿态来源；
- expert/BC/DAgger 各阶段数据量和时间；
- 所有成功率、安全和 tracking 指标；
- PT/ONNX parity 和延迟；
- 未解决问题；
- 一条从模型文件到公司部署的明确集成说明。

## 十六、完成标准

只有同时满足以下条件，才可以报告任务完成：

1. 新方法代码位于独立文件夹，旧项目核心接口未被破坏；
2. 真实 708 reference 和真实汽车尾箱环境已运行；
3. expert 在闭环中可靠，而不只是生成离线动作；
4. BC 和 DAgger 均真实执行过；
5. 最终 student 使用公司要求的 93D 输入和 16D 输出；
6. wheel 输出与已冻结的公司 mode 一致；`hard_zero` 时在模型图中恒为 0；
7. `policy.pt` 和 `policy.onnx` 均已实际生成；
8. ONNX 输入输出名称、shape、opset 与公司运行时一致；
9. eager/PT/ONNX golden parity 和 ONNX 闭环评估通过；
10. 有可复现命令、配置、hash、指标、失败分析和最终报告。

如果因为外部依赖、资产缺失或合同冲突确实无法完成某一步，不要伪造结果。先完成所有不受阻部分，保存可复现状态，然后给出：

- 精确阻塞点；
- 已执行的证据；
- 最小缺失输入；
- 恢复后应运行的下一条命令。

现在开始执行。第一条进度更新先告诉我：你找到的两个仓库路径、当前 GPU/环境版本、708 NPZ 和尾箱资产是否齐全、真实 obs/action/ONNX 合同，以及准备采用的 expert feasibility 顺序。随后不要停在计划阶段，继续完成 Gate 0。
