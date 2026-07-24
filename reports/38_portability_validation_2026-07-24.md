# `lateral_mppi_dagger` 独立搬运验收（2026-07-24）

## 结论

状态：`PORTABLE_WITH_PROJECT_ENTRYPOINTS`

可以把整个 `lateral_mppi_dagger/` 目录移动到
`/home/zzc/Desktop/zhuoxili-jackie/lateral_mppi_dagger` 或其他位置。运行时需要的机器人、尾箱、
STL、URDF、USD、NPZ、RobotLab 源码和 IsaacLab 源码均已在目录内；不存在指向目录外的符号链接。
正式目录没有在本次验收中被移动，部署目录也没有被修改。

必须从项目根目录使用 `python scripts/...` 入口运行。当前 Conda 环境中的裸 editable 安装仍记录旧
`robot_move_lt` 路径；项目入口会先执行 `scripts/_bootstrap.py`，把当前项目内的 `src/`、
`vendor/robot_lab/` 和 `vendor/IsaacLab/source/*` 放到导入路径最前面，并在 Isaac workflow 中校验
实际导入位置。不要从任意目录用裸 `python -c "import robot_lab"` 来判断本项目是否独立。

## 资源盘点

- 整个项目约 1.0 GiB；加入本次最终报告后当前为 3673 个普通文件（验收副本复制时为 3670 个）。
- 整棵项目树：35 STL、888 NPZ、4 USD、6 URDF、1 OBJ。
- 当前执行资产集合（`assets/`、`vendor/robot_lab/data/`、`vendor/deployment_evidence/`）：
  30 STL、15 NPZ、4 USD、2 URDF、1 OBJ，另有 8 CSV，共 66 个审计文件。
- `scripts/audit_project.py` 对冻结合同、参考轨迹 SHA256、URDF mesh 引用、禁止旧仓库引用和外部
  symlink 的最终检查全部通过：`reports/40_portability_audit_final.json`，`failures=[]`。
- 全树符号链接数量为 0。
- USD 二进制字符串检查没有发现 `/home/...` 外部引用。
- `assets/references/low_load_v1/generation_report.json` 原有两个仅用于 provenance 的绝对路径，
  本次已改成项目相对路径；生成脚本以后也会持续写相对路径。

888 个 NPZ 中包含训练数据集、历史数据和 `reference_only/` 的只读论文仓库内容；它们都物理位于项目
目录内。运行时关键 NPZ 是配置文件声明并有 SHA256 门禁的 6 条 legacy 参考和 9 条 low-load 参考。

全树文本中仍能看到旧绝对路径，但它们只位于历史 `reports/`（137 个文件）、10 个
`resolved_train_config.json`、4 个 `export_manifest.json` 以及本文交接说明中，用于记录当时的
dataset/artifact provenance。`configs/`、`scripts/`、`src/`、`tests/`、`assets/` 和运行用
`vendor/` 已无旧项目/旧根目录绝对引用。训练 resume 由命令行传入当前 dataset/checkpoint，并按
manifest hash 校验；导出包也已在临时新根目录通过校验，所以这些历史字段不是运行时路径依赖。

## 真正的“改根目录”验收

使用独立临时根目录：

```text
/tmp/lateral_mppi_portability_OwHabq/lateral_mppi_dagger
```

完整复制原项目后，在临时副本内得到：

- 独立审计：`ok=true`，66 个资产，0 failure；
- low-load reference validation：首帧一致、`q_reset_max_abs=0`、50 Hz 匹配；
- `pytest -q`：50 passed；
- 原目录与临时目录的合同、low-load 9 个 NPZ、冻结 URDF/STL 的 SHA256 完全一致；
- Key 7 离线交接：checkpoint、ONNX、旋转 6D、首帧腿姿态全部通过；
- 导出包验证：固定 `[1,93] -> [1,16]`、opset 18、SHA256、TorchScript/ONNX parity、
  wheel hard-zero、20 ms CPU p95 门全部通过；
- Isaac/MPPI smoke（`ref_id=1`、4 samples、horizon 2、2 steps）成功，报告中的
  `isaaclab_import`、`robot_lab_import` 和 `standalone_root` 全部指向临时副本。启动日志使用：

```text
/tmp/lateral_mppi_portability_OwHabq/lateral_mppi_dagger/vendor/IsaacLab/apps/isaaclab.python.headless.kit
```

这项真实 Isaac 启动证明项目入口覆盖了 Conda 中旧 editable 路径；不是只做了静态文件检查。
验收完成后，上述约 1 GiB 临时副本已移动到桌面环境的回收站，正式项目原位保留且未被移动。

## Conda 环境结论

`isaaclab` 环境的 PEP 660 元数据仍指向旧目录：

```text
isaaclab* -> file:///home/zzc/Desktop/zhuoxili-jackie/robot_move_lt/IsaacLab/source/...
robot_lab -> file:///home/zzc/Desktop/zhuoxili-jackie/robot_move_lt/source/robot_lab
```

这不会影响本项目的官方脚本入口，因为根目录由 `__file__` 动态计算并优先插入 vendored 源码。当前不要
为了“修路径”重建环境或在旧工程执行 `pip install -e`。搬家后的推荐方式：

```bash
conda activate isaaclab
cd /home/zzc/Desktop/zhuoxili-jackie/lateral_mppi_dagger
python scripts/audit_project.py --output reports/post_move_audit.json
python scripts/validate_reference.py \
  --config configs/reference_low_load_v1.yaml \
  --output reports/post_move_reference_validation.json
python -m pytest -q
```

之后先做一个很小的 Isaac smoke，并确认输出中的 experience、`isaaclab_import`、`robot_lab_import`、
`standalone_root` 都在新根目录内。只有需要从项目外直接 import 包时，才考虑把 Conda editable 安装
重定向到新目录；当前训练/采集/导出流程不需要这样做。

## 与搬运无关但必须先修的训练阻塞

搬运副本第一次使用 `ref_id=8` 启动 Isaac 时，发现
`src/lateral_mppi_dagger/env/isaac_adapter.py::IsaacLateralAdapter.reset()` 将 low-load reference ID
直接写入 vendored `MotionCommand.motion_ids`。low-load 配置有 9 条参考（0..8），而当前 Isaac task
内部只有 6 条 legacy motion（0..5），所以 6..8 会在 `MotionCommand._resample_command()` 中触发
CUDA 越界断言。

`ref_id=1` 的搬运 smoke 已成功，因此这不是缺 STL/NPZ、Conda 指错或搬家失败。正式采集前必须：

1. 将 low-load reference ID 与 legacy Isaac command ID 解耦；
2. 正确把选中 low-load reference 的 frame-zero 状态写入机器人，同时保留预期 reset 扰动；
3. 验证 episode metadata、动态 reference、接触计划和命令速度仍使用同一 low-load ID；
4. 分别 smoke `ref_id=0..8`；
5. 通过后才允许 `--rotate-references` 和正式 MPPI 数据采集。

不能简单用 `% 6` 取模；那会让机器人 reset 状态、标签参考和 metadata 语义错位。

## 推荐移动步骤

先复制并验证，确认无误后再处理旧副本：

```bash
cp -a /home/zzc/Desktop/zhuoxili-jackie/transfer/lateral_mppi_dagger \
  /home/zzc/Desktop/zhuoxili-jackie/lateral_mppi_dagger
cd /home/zzc/Desktop/zhuoxili-jackie/lateral_mppi_dagger
conda activate isaaclab
python scripts/audit_project.py --output reports/post_move_audit.json
python scripts/validate_reference.py \
  --config configs/reference_low_load_v1.yaml \
  --output reports/post_move_reference_validation.json
python -m pytest -q
```

不要先删除或覆盖旧副本。新副本通过上述门和一项小型 Isaac smoke 后，再由用户决定是否保留旧副本。
