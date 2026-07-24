# Stable lateral MPPI → DAgger → ONNX runbook

This runbook is fail-closed: a later stage must not start until the preceding
report has `"ok": true`. It modifies only this standalone repository. It never
writes to `loco_rl_deploy`.

## 0. Reboot and verify CUDA

The current host has NVIDIA kernel module `580.159.03` but user-space library
`580.173`. Reboot before attempting Isaac:

```bash
sudo reboot
```

After login:

```bash
conda activate isaaclab
cd /实际路径/lateral_mppi_dagger
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Do not continue unless both commands succeed.

## 1. Offline integrity and low-load reference

```bash
python scripts/audit_project.py
python scripts/validate_reference.py \
  --config configs/reference_low_load_v1.yaml \
  --output reports/31_low_load_reference_validation.json
pytest -q
```

The low-load reference has:

- fixed 4 cm stride;
- cadence proportional to requested speed;
- one limb in swing at a time;
- front trunk detachment at most 8 mm;
- rear ground clearance at most 12 mm;
- exact command atoms `0, ±0.012, ±0.024, ±0.03, ±0.06 m/s`;
- a common deployment-compatible first frame.

## 2. Isaac clone-state and bounded MPPI smoke gates

> **2026-07-24 fail-closed blocker:** `reference_low_load_v1.yaml` has 9
> references, but the vendored Isaac task's `MotionCommand` currently owns only
> the 6 legacy references. `IsaacLateralAdapter.reset()` writes the low-load
> `ref_id` directly into that legacy command, so IDs 6, 7, and 8 currently cause
> a CUDA index assertion. Do not run the 9-episode rotating-reference command
> below, and do not begin formal collection, until the adapter decouples the
> low-load reference ID from the legacy command ID and all 9 reset IDs pass an
> Isaac regression smoke. This is a reference-index bridge bug, not a missing
> asset or directory-relocation failure.

```bash
python scripts/validate_mppi_rollout.py \
  --headless --device cuda:0 \
  --mppi-config configs/expert_mppi_low_load_v1.yaml \
  --samples 16 --horizon 12 --ref-id 1 \
  --report reports/32_low_load_mppi_state_copy.json

python scripts/smoke_test_expert.py \
  --headless --device cuda:0 \
  --expert-backend mppi \
  --mppi-config configs/expert_mppi_low_load_v1.yaml \
  --mppi-samples 16 --mppi-horizon 12 --mppi-iterations 1 \
  --dataset datasets/low_load_smoke_v1 \
  --episodes 9 --steps 100 --seed 3000 \
  --ref-id 0 --rotate-references \
  --run-name low_load_smoke_v1
```

Inspect the smoke dataset before formal collection. Reject the candidate if:

- either rear wheel has a normal-force p95 above 135 N;
- both rear contacts are absent in any sustained interval;
- front desired-contact force is below 6 N for more than 20% of stance;
- signed lateral progress is below 60%;
- the base falls, yaws away, or drifts more than 10 cm in box-local x.

Tune only the new reference and `configs/expert_mppi_low_load_v1.yaml`; do not
change the deployment repository.

## 3. Formal MPPI expert gate and R0 data

This produces 15,000 real Isaac/MPPI transitions:

```bash
python scripts/collect_expert.py \
  --headless --device cuda:0 \
  --expert-backend mppi \
  --mppi-config configs/expert_mppi_low_load_v1.yaml \
  --dataset datasets/low_load_r0_v1 \
  --episodes 50 --steps 300 --seed 1000 \
  --ref-id 0 --rotate-references \
  --scenario nominal --dagger-round 0 \
  --run-name low_load_r0_v1 \
  --report reports/33_low_load_r0_collection.json

python scripts/validate_expert_gate.py datasets/low_load_r0_v1 \
  --config configs/expert_mppi_low_load_v1.yaml \
  --output reports/34_low_load_r0_expert_gate.json

python scripts/analyze_observability.py datasets/low_load_r0_v1 \
  --output reports/35_low_load_r0_observability.json

python scripts/merge_dataset.py \
  --source datasets/low_load_r0_v1 \
  --destination datasets/low_load_dagger_aggregate_v1 \
  --output reports/36_low_load_r0_merge.json
```

## 4. Initial BC student

```bash
python scripts/train_bc.py \
  --dataset datasets/low_load_dagger_aggregate_v1 \
  --output checkpoints/low_load_bc_v1 \
  --device cuda:0 --epochs 100 --batch-size 1024

python scripts/evaluate.py \
  --checkpoint checkpoints/low_load_bc_v1/student_best_checkpoint.pt \
  --dataset datasets/low_load_dagger_aggregate_v1 \
  --split test --device cuda:0 \
  --output reports/37_low_load_bc_open_loop.json
```

The student graph embeds the exact-zero handoff, continuous previous-action
activation, ±0.06 m/s safe command cap, joint-specific 0.12–0.30 rad target
envelopes, 2.25 rad/s target-rate limit, and exact-zero wheel outputs.

## 5. DAgger admission and three real student-state rounds

First obtain an admission gate. Survival alone is insufficient; moving
references must reach at least 20% signed progress:

```bash
python scripts/run_student_gate.py \
  --checkpoint checkpoints/low_load_bc_v1/student_best_checkpoint.pt \
  --dataset datasets/low_load_bc_admission_v1 \
  --run-name low_load_bc_admission_v1 \
  --reference-config configs/reference_low_load_v1.yaml \
  --tracking-config configs/expert_mppi_low_load_v1.yaml \
  --device cuda:0 --episodes 50 --steps 300 --seed 4000 \
  --gate-purpose dagger_admission \
  --dagger-admission-full-horizon-success-rate-min 0.80 \
  --dagger-admission-per-reference-mean-horizon-fraction-min 0.95 \
  --dagger-admission-minimum-episode-horizon-fraction 0.85 \
  --dagger-admission-signed-progress-ratio-min 0.20 \
  --output reports/38_low_load_bc_admission.json
```

Then run R1. Repeat the same pattern for R2 and R3, using the preceding
round's best checkpoint and gate:

```bash
python scripts/run_dagger.py \
  --round 1 \
  --dataset datasets/low_load_dagger_aggregate_v1 \
  --input-checkpoint checkpoints/low_load_bc_v1/student_best_checkpoint.pt \
  --previous-gate reports/38_low_load_bc_admission.json \
  --output checkpoints/low_load_dagger_r1_v1 \
  --reference-config configs/reference_low_load_v1.yaml \
  --mppi-config configs/expert_mppi_low_load_v1.yaml \
  --device cuda:0 --epochs 100
```

R0 plus R1–R3 contains at least 38,400 Isaac transitions
(`15,000 + 3 × 7,800`) before temporal-window sampling and 100 training
epochs.

## 6. Final performance gate and export

The final checkpoint must pass both directions, all command atoms, standing,
force/load gates, signed tracking, start/stop behavior, and exact wheel zero:

```bash
python scripts/run_student_gate.py \
  --checkpoint checkpoints/low_load_dagger_r3_v1/student_best_checkpoint.pt \
  --dataset datasets/low_load_final_gate_v1 \
  --run-name low_load_final_gate_v1 \
  --reference-config configs/reference_low_load_v1.yaml \
  --tracking-config configs/expert_mppi_low_load_v1.yaml \
  --device cuda:0 --episodes 100 --steps 300 --seed 5000 \
  --gate-purpose performance \
  --success-rate-min 0.95 \
  --per-reference-success-rate-min 0.90 \
  --output reports/45_low_load_final_gate.json

python scripts/export_policy.py \
  --checkpoint checkpoints/low_load_dagger_r3_v1/student_best_checkpoint.pt \
  --dataset datasets/low_load_dagger_aggregate_v1 \
  --output exported/low_load_lateral_release_v1

python scripts/validate_export.py \
  --exported exported/low_load_lateral_release_v1 \
  --output reports/46_low_load_export_validation.json

python scripts/validate_key7_handoff.py \
  --checkpoint checkpoints/low_load_dagger_r3_v1/student_best_checkpoint.pt \
  --onnx exported/low_load_lateral_release_v1/policy.onnx \
  --output reports/47_low_load_key7_handoff.json
```

Only `exported/low_load_lateral_release_v1/policy.onnx` is a deployment
candidate, and only after all reports above pass. Copying it into a deployment
machine is a separate, explicitly authorized action.
