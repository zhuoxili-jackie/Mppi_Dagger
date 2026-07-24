#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any

import numpy as np

from _bootstrap import ROOT, write_json

from lateral_mppi_dagger.config import sha256_file
from lateral_mppi_dagger.data.dataset import load_manifest
from lateral_mppi_dagger.data.schema import ENUMS, read_episode_shard


def _load_json(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Required evidence does not exist: {resolved}")
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {resolved}.")
    return value


def _tracking_summary(report: dict[str, Any]) -> dict[str, Any]:
    values: dict[str, list[Any]] = collections.defaultdict(list)
    failures: collections.Counter[str] = collections.Counter()
    for episode in report.get("episodes", []):
        failures.update(str(item) for item in episode.get("failures", []))
        for name, value in episode.get("tracking", {}).items():
            values[str(name)].append(value)
    metrics: dict[str, Any] = {}
    for name, items in sorted(values.items()):
        array = np.asarray(items, dtype=np.float64)
        metrics[name] = {
            "mean": np.mean(array, axis=0).tolist(),
            "minimum": np.min(array, axis=0).tolist(),
            "maximum": np.max(array, axis=0).tolist(),
        }
    return {
        "metrics": metrics,
        "failure_counts": dict(sorted(failures.items())),
    }


def _dataset_summary(dataset_root: Path) -> dict[str, Any]:
    records = load_manifest(dataset_root)
    splits: collections.Counter[str] = collections.Counter()
    rounds: collections.Counter[int] = collections.Counter()
    round_frames: collections.Counter[int] = collections.Counter()
    student_behavior_frames: collections.Counter[int] = collections.Counter()
    teacher_valid_frames: collections.Counter[int] = collections.Counter()
    solve_values: dict[int, list[np.ndarray]] = collections.defaultdict(list)
    student_code = ENUMS["behavior_policy"]["STUDENT"]
    for record in records:
        shard = read_episode_shard(dataset_root / record["path"])
        steps = int(shard.arrays["step_id"].shape[0])
        round_number = int(
            record.get("dagger_round", shard.metadata.get("dagger_round", 0))
        )
        splits[str(record["split"])] += 1
        rounds[round_number] += 1
        round_frames[round_number] += steps
        behavior = np.asarray(shard.arrays["behavior_policy"])
        student_behavior_frames[round_number] += int(
            np.count_nonzero(behavior == student_code)
        )
        valid = np.asarray(shard.arrays["teacher_valid"], dtype=bool)
        teacher_valid_frames[round_number] += int(np.count_nonzero(valid))
        solve_ms = np.asarray(shard.arrays["solve_ms"], dtype=np.float64)
        finite = valid & np.isfinite(solve_ms)
        if np.any(finite):
            solve_values[round_number].append(solve_ms[finite])
    per_round: dict[str, Any] = {}
    for round_number in sorted(rounds):
        solve = (
            np.concatenate(solve_values[round_number])
            if solve_values[round_number]
            else np.empty(0, dtype=np.float64)
        )
        per_round[str(round_number)] = {
            "episodes": int(rounds[round_number]),
            "frames": int(round_frames[round_number]),
            "student_behavior_frames": int(student_behavior_frames[round_number]),
            "teacher_valid_frames": int(teacher_valid_frames[round_number]),
            "mean_teacher_solve_ms": (
                float(np.mean(solve)) if solve.size else None
            ),
            "cumulative_teacher_solve_hours": (
                float(np.sum(solve) / 3_600_000.0) if solve.size else None
            ),
        }
    return {
        "root": str(dataset_root),
        "episodes": len(records),
        "frames": int(sum(round_frames.values())),
        "split_episode_counts": dict(sorted(splits.items())),
        "per_round": per_round,
    }


def _report_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build fail-closed release evidence from completed MPPI, BC, "
            "DAgger, export-parity, and ONNX closed-loop reports."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT
        / "checkpoints/dagger_r3_conservative/student_best_checkpoint.pt",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "datasets/dagger_aggregate",
    )
    parser.add_argument("--exported", type=Path, default=ROOT / "exported")
    parser.add_argument(
        "--audit",
        type=Path,
        default=ROOT / "reports/00_audit_state.json",
    )
    parser.add_argument(
        "--reference-validation",
        type=Path,
        default=ROOT / "reports/01_reference_validation.json",
    )
    parser.add_argument(
        "--expert-gate",
        type=Path,
        default=ROOT / "reports/13_mppi_formal_gate50_v3_gate.json",
    )
    parser.add_argument(
        "--torch-gate",
        type=Path,
        default=ROOT / "reports/r3_conservative_dagger_admission_gate.json",
    )
    parser.add_argument(
        "--onnx-gate",
        type=Path,
        default=ROOT / "reports/onnx_r3_conservative_dagger_admission_gate.json",
    )
    parser.add_argument(
        "--export-validation",
        type=Path,
        default=ROOT / "reports/export_validation.json",
    )
    parser.add_argument(
        "--selection-output",
        type=Path,
        default=ROOT / "reports/deployment_checkpoint_selection.json",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=ROOT / "reports/final_metrics.json",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=ROOT / "reports/final_report.md",
    )
    parser.add_argument(
        "--failure-output",
        type=Path,
        default=ROOT / "reports/failure_analysis.md",
    )
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    dataset = args.dataset.resolve()
    exported = args.exported.resolve()
    audit = _load_json(args.audit)
    reference_validation = _load_json(args.reference_validation)
    expert_gate = _load_json(args.expert_gate)
    torch_gate = _load_json(args.torch_gate)
    onnx_gate = _load_json(args.onnx_gate)
    export_validation = _load_json(args.export_validation)
    round_gates = {
        round_number: _load_json(
            ROOT / f"reports/dagger_r{round_number}_collection_gate.json"
        )
        for round_number in (1, 2, 3)
    }
    rejected_r3_gate = _load_json(ROOT / "reports/r3_dagger_admission_gate.json")
    r2_recovery_gate = _load_json(
        ROOT / "reports/r2_recovery1_dagger_admission_gate.json"
    )
    checkpoint_hash = sha256_file(checkpoint)
    export_manifest = _load_json(exported / "export_manifest.json")
    onnx_hash = sha256_file(exported / "policy.onnx")
    torchscript_hash = sha256_file(exported / "policy.pt")
    required_export_files = (
        "policy.pt",
        "policy.onnx",
        "export_manifest.json",
        "golden_io.npz",
        "SHA256SUMS",
    )

    checks = {
        "standalone_audit": audit.get("ok") is True,
        "reference_validation": (
            reference_validation.get("checks", {}).get("all_first_frames_equal")
            is True
            and reference_validation.get("checks", {}).get(
                "control_reference_hz_match"
            )
            is True
            and float(
                reference_validation.get("checks", {}).get(
                    "q_reset_max_abs", float("inf")
                )
            )
            <= 1.0e-6
            and float(
                reference_validation.get("checks", {}).get(
                    "root_position_max_abs", float("inf")
                )
            )
            <= 1.0e-6
            and float(
                reference_validation.get("checks", {}).get(
                    "root_quaternion_max_abs", float("inf")
                )
            )
            <= 1.0e-6
        ),
        "formal_mppi_expert_gate": expert_gate.get("ok") is True,
        "aggregate_at_least_20000_frames": _dataset_summary(dataset)["frames"]
        >= 20_000,
        "dagger_r1_r3_collection_gates": all(
            gate.get("ok") is True for gate in round_gates.values()
        ),
        "selected_torch_closed_loop_admission": (
            torch_gate.get("dagger_admission_ok") is True
            and torch_gate.get("ok") is True
        ),
        "onnx_closed_loop_admission": (
            onnx_gate.get("dagger_admission_ok") is True
            and onnx_gate.get("ok") is True
        ),
        "checkpoint_matches_export_manifest": (
            export_manifest.get("checkpoint_sha256") == checkpoint_hash
        ),
        "onnx_closed_loop_hash_matches_final_file": (
            onnx_gate.get("summary", {}).get("student_checkpoint_hashes")
            == [onnx_hash]
        ),
        "fixed_93_to_16_abi": (
            export_validation.get("onnx_input_name") == "obs"
            and export_validation.get("onnx_input_shape") == [1, 93]
            and export_validation.get("onnx_output_name") == "actions"
            and export_validation.get("onnx_output_shape") == [1, 16]
        ),
        "golden_parity": (
            export_validation.get("torchscript_vs_eager", {}).get(
                "max_abs_error", float("inf")
            )
            <= 1.0e-5
            and export_validation.get("onnx_vs_eager", {}).get(
                "max_abs_error", float("inf")
            )
            <= 1.0e-5
            and export_validation.get("onnx_vs_eager", {}).get(
                "mean_abs_error", float("inf")
            )
            <= 1.0e-6
        ),
        "hard_zero_all_backends": (
            export_validation.get("hard_zero_all_backends") is True
        ),
        "onnx_cpu_50hz_latency": (
            export_validation.get("latency_50hz_pass") is True
        ),
        "required_export_files": all(
            (exported / name).is_file() for name in required_export_files
        ),
    }
    engineering_release_ready = all(checks.values())
    production_acceptance = bool(
        onnx_gate.get("performance_ok") is True
        and onnx_gate.get("summary", {}).get("success_rate", 0.0) >= 0.95
        and all(
            value >= 0.90
            for value in onnx_gate.get("summary", {})
            .get("per_reference_success_rate", {})
            .values()
        )
    )
    if not engineering_release_ready:
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Engineering release checks failed: {failed}")

    dataset_summary = _dataset_summary(dataset)
    selection = {
        "schema_version": "pcbc-deployment-checkpoint-selection-v1",
        "selection_policy": (
            "closed_loop_safety_first_then_export_parity; "
            "open_loop_action_rmse_is_not_a_deployment_gate"
        ),
        "selected": {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_hash,
            "torch_closed_loop_gate": _report_path(args.torch_gate),
            "full_horizon_success_rate": torch_gate["summary"][
                "full_horizon_success_rate"
            ],
            "shield_intervention_rate": torch_gate["summary"][
                "shield_intervention_rate"
            ],
            "incorporates_dagger_round": 3,
        },
        "rejected_standard_r3": {
            "checkpoint": str(
                (ROOT / "checkpoints/dagger_r3/student_best_checkpoint.pt").resolve()
            ),
            "checkpoint_sha256": sha256_file(
                ROOT / "checkpoints/dagger_r3/student_best_checkpoint.pt"
            ),
            "gate": "reports/r3_dagger_admission_gate.json",
            "dagger_admission_ok": rejected_r3_gate.get("dagger_admission_ok"),
            "full_horizon_success_rate": rejected_r3_gate["summary"][
                "full_horizon_success_rate"
            ],
            "reason": "closed-loop regression despite a better offline selection metric",
        },
        "stable_r2_source": {
            "checkpoint": str(
                (
                    ROOT
                    / "checkpoints/dagger_r2_recovery1/student_best_checkpoint.pt"
                ).resolve()
            ),
            "checkpoint_sha256": sha256_file(
                ROOT
                / "checkpoints/dagger_r2_recovery1/student_best_checkpoint.pt"
            ),
            "gate": "reports/r2_recovery1_dagger_admission_gate.json",
            "dagger_admission_ok": r2_recovery_gate.get("dagger_admission_ok"),
        },
        "production_accepted": production_acceptance,
    }
    write_json(args.selection_output, selection)

    final_metrics = {
        "schema_version": "pcbc-final-metrics-v1",
        "release_status": (
            "engineering_release_candidate_not_production_accepted"
            if not production_acceptance
            else "production_acceptance_passed"
        ),
        "engineering_release_ready": engineering_release_ready,
        "production_acceptance_passed": production_acceptance,
        "checks": checks,
        "expert": {
            "backend": "whole_body_mppi",
            "reference_repository_policy": "RTWholeBodyMPPI_reference_only",
            "gate": _report_path(args.expert_gate),
            "summary": expert_gate.get("summary", {}),
            "tracking": _tracking_summary(expert_gate),
        },
        "dataset": dataset_summary,
        "dagger": {
            "rounds_completed": [1, 2, 3],
            "collection_gates": {
                str(round_number): {
                    "path": f"reports/dagger_r{round_number}_collection_gate.json",
                    "ok": gate.get("ok"),
                    "summary": gate.get("summary", {}),
                }
                for round_number, gate in round_gates.items()
            },
        },
        "selected_student": {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_hash,
            "torch_closed_loop_gate": _report_path(args.torch_gate),
            "torch_summary": torch_gate.get("summary", {}),
            "torch_tracking": _tracking_summary(torch_gate),
            "onnx_closed_loop_gate": _report_path(args.onnx_gate),
            "onnx_summary": onnx_gate.get("summary", {}),
            "onnx_tracking": _tracking_summary(onnx_gate),
        },
        "export": {
            "directory": str(exported),
            "checkpoint_sha256": checkpoint_hash,
            "torchscript_sha256": torchscript_hash,
            "onnx_sha256": onnx_hash,
            "manifest_model_hash": export_manifest.get("model_hash"),
            "validation": export_validation,
        },
        "unresolved": [
            "Strict reference-tracking success is 0/50 on the current ONNX gate.",
            "The required final 100 episodes per reference production gate was not run because the 50-episode precursor already failed tracking thresholds.",
            "Legacy baseline/residual ONNX graphs were absent from the supplied deployment evidence.",
            "No final behavior video has been accepted by a human reviewer.",
        ],
    }
    write_json(args.metrics_output, final_metrics)

    round_lines = []
    for round_number, values in dataset_summary["per_round"].items():
        round_lines.append(
            f"| R{round_number} | {values['episodes']} | {values['frames']} | "
            f"{values['student_behavior_frames']} | "
            f"{values['teacher_valid_frames']} | "
            f"{values['mean_teacher_solve_ms']:.3f} |"
        )
    expert_summary = expert_gate["summary"]
    torch_summary = torch_gate["summary"]
    onnx_summary = onnx_gate["summary"]
    report = f"""# MPPI + DAgger 迁移最终工程报告

## 结论

独立工程迁移、真实 Isaac Lab MPPI 教师、超过两万帧数据采集、BC、R1–R3
DAgger、TorchScript/ONNX 导出及 ONNX 闭环准入均已完成。当前产物是
**工程发布候选（engineering release candidate）**，不是生产任务验收版本。

- 工程发布检查：`{engineering_release_ready}`
- 严格生产 tracking 验收：`{production_acceptance}`
- 最终 student checkpoint SHA256：`{checkpoint_hash}`
- 最终 ONNX SHA256：`{onnx_hash}`

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
{expert_summary['successes']}/{expert_summary['episodes_found']}，teacher-valid
rate={expert_summary['teacher_valid_rate']:.6f}，wheel hard-zero=
{expert_summary['wheel_action_exact_zero']}。完整证据见
`reports/13_mppi_formal_gate50_v3_gate.json`。

## 数据与训练

聚合数据共 {dataset_summary['episodes']} 个 episode、{dataset_summary['frames']} 帧；
episode split 为 `{json.dumps(dataset_summary['split_episode_counts'], sort_keys=True)}`。
表中的 solve time 是 shard 中 MPPI 求解计时，不是端到端墙钟时间。

| 阶段 | episodes | frames | student 执行帧 | MPPI 有效标签帧 | 平均 solve ms |
|---|---:|---:|---:|---:|---:|
{chr(10).join(round_lines)}

BC 与 R1、R2、R3 均真实训练和闭环运行。标准 R3 checkpoint 在固定闭环 seed 上
退化为 0/50 全时域；因此没有按离线 RMSE 盲选它。最终选择从稳定 R2 checkpoint
以 `1e-6` 学习率吸收 R3 数据的一轮保守微调：
`checkpoints/dagger_r3_conservative/student_best_checkpoint.pt`。其 PyTorch 闭环准入
为 {torch_summary['full_horizon_successes']}/{torch_summary['episodes']} 全时域，
shield intervention rate={torch_summary['shield_intervention_rate']:.6f}。
选择证据见 `reports/deployment_checkpoint_selection.json`。

## 导出与闭环复评

- `exported/policy.pt`
- `exported/policy.onnx`
- `exported/export_manifest.json`
- `exported/golden_io.npz`
- `exported/SHA256SUMS`

256 个 golden observations 上 TorchScript/eager 最大误差
`{export_validation['torchscript_vs_eager']['max_abs_error']:.3e}`，
ONNX/eager 最大误差 `{export_validation['onnx_vs_eager']['max_abs_error']:.3e}`、
平均误差 `{export_validation['onnx_vs_eager']['mean_abs_error']:.3e}`。
ONNX CPU p95 为 `{export_validation['onnx_cpu_p95_ms']:.6f} ms`，满足 50 Hz；
三种后端 wheel 均 exact zero。

最终根目录 ONNX 与闭环验证的文件 hash 完全相同。ONNX 在 R3
`light_delay_platform_pose` 场景为
{onnx_summary['full_horizon_successes']}/{onnx_summary['episodes']} 全时域，
shield intervention rate={onnx_summary['shield_intervention_rate']:.6f}。

## 尚未通过的生产指标

当前 ONNX 的严格 reference-tracking success 为
{onnx_summary['successes']}/{onnx_summary['episodes']}。虽然没有提前终止，但
base position、wheel center、contact schedule 与 box-local x drift 仍超阈值。
因此没有继续运行每条 reference 100 回合的最终大门禁，也没有声称达到 95%
overall / 90% per-reference 的生产标准。详见 `reports/failure_analysis.md`。

## 部署集成

公司运行时只需 ONNX Runtime 和本报告冻结的 observation/action adapter，不需要
Isaac Lab、MPPI、Python、MoveIt 或 `RTWholeBodyMPPI`。每个 20 ms 控制周期构造
一个 `[1,93]` float32 `obs`，调用 `policy.onnx`，读取 `[1,16]` float32
`actions`，再按 manifest 的 joint mapping、offset 与 scale 送入现有控制层。
上一拍 observation action 必须是 safety/shield 后真正执行的 raw action。
"""
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(report, encoding="utf-8")

    rejected_summary = rejected_r3_gate["summary"]
    failure_report = f"""# Failure analysis

## Closed-loop regressions

- Standard R3 checkpoint: {rejected_summary['full_horizon_successes']}/
  {rejected_summary['episodes']} full-horizon episodes, mean horizon fraction
  {rejected_summary['mean_horizon_fraction']:.6f}. It was rejected.
- Conservative R3 checkpoint: {torch_summary['full_horizon_successes']}/
  {torch_summary['episodes']} full-horizon episodes. It was selected for the
  engineering bundle.
- Final ONNX: {onnx_summary['full_horizon_successes']}/
  {onnx_summary['episodes']} full-horizon episodes, but strict tracking success
  {onnx_summary['successes']}/{onnx_summary['episodes']}.

## Evidence-backed interpretation

The export is not the source of the tracking failure: eager, TorchScript and
ONNX satisfy golden parity, and the exact same ONNX SHA256 passed the same
full-horizon gate as the PyTorch checkpoint. The remaining error belongs to the
learned closed-loop policy.

The frozen legacy 93D contract exposes a fixed-first-frame motion prefix and a
constant final scalar; it does not expose dynamic reference phase. The student
must infer phase from robot state and the previous executed action. Offline
imitation metrics therefore did not reliably predict autoregressive closed-loop
behavior. This is consistent with the standard R3 checkpoint having a better
offline selection score while catastrophically regressing in closed loop.

## Required next work for production acceptance

1. Continue only with safety-gated R4/R5 student-state collection, retaining
   MPPI as the sole label source.
2. Evaluate action-history exposure/scheduled self-conditioning while keeping
   the external 93D ABI unchanged.
3. Select every candidate by fixed-seed closed-loop tracking, not by validation
   action RMSE alone.
4. When the 50-episode precursor passes strict tracking, run at least 100
   episodes per actual reference and compare against MPPI on identical seeds.
5. Record and manually review behavior video for contact switching, slipping,
   penetration, and trunk-assisted motion.

No threshold was relaxed and no failed metric was relabeled as success.
"""
    args.failure_output.parent.mkdir(parents=True, exist_ok=True)
    args.failure_output.write_text(failure_report, encoding="utf-8")
    print(
        json.dumps(
            {
                "engineering_release_ready": engineering_release_ready,
                "production_acceptance_passed": production_acceptance,
                "checkpoint_sha256": checkpoint_hash,
                "onnx_sha256": onnx_hash,
                "metrics": str(args.metrics_output.resolve()),
                "report": str(args.report_output.resolve()),
                "failure_analysis": str(args.failure_output.resolve()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
