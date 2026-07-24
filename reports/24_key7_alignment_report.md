# Key-7 alignment correction report

Date: 2026-07-24

## Scope boundary

`/home/zzc/Desktop/zhuoxili-jackie/loco_rl_deploy` was used only as read-only
ABI evidence.  It was not edited, built, launched, or used to control a robot.
Every code/data/model change described here is contained in
`lateral_mppi_dagger`.

## Root cause

The deployment observation builder and the vendored Isaac task both evaluate
`matrix[..., :2]`.  Since the slice is on the final dimension, this selects
the first two **columns**, flattened in C order as:

```text
[R00, R01, R10, R11, R20, R21]
```

The migrated MPPI observation adapter incorrectly selected the first two rows:

```text
[R00, R01, R02, R10, R11, R12]
```

The identity rotation is enough to expose the mismatch:

```text
deployment/Isaac: [1, 0, 0, 1, 0, 0]
old MPPI:         [1, 0, 0, 0, 1, 0]
```

The old checkpoint's key-4-to-key-7 handoff observation put two rotation
features about 11.7 and 92.8 standard deviations outside its training
distribution.  The deployment performs three inference-only warm-up cycles
while feeding each raw output back as `previous_action`; the fourth output is
the first applied command.  Offline reproduction gave a maximum physical leg
target jump of 0.6230159 rad.

Evidence: `reports/19_pre_fix_key7_handoff_failure.json`.

## MPPI-repository corrections

- The canonical 93D builder now uses rotation columns.
- The deployment contract explicitly lists the six flattened elements.
- Existing immutable episode shards were migrated into
  `datasets/dagger_aggregate_v2_columns`; the source dataset was preserved.
- The reference set now exposes a seventh, deterministic zero-velocity
  standing reference derived from reference 0's first frame.
- The student masks the previous-action slice internally when the deployment
  velocity command is inside the configured zero-command deadband.  This
  makes the three key-7 warm-up cycles stable without changing deployment
  code.
- An offline golden replay verifies yaw alignment, the key-4 final posture,
  three warm-up cycles, the first applied target delta and hard-zero wheel
  outputs.

The corrected dataset contains 166 episodes and 43,651 frames.  All 39 unit
tests pass.

## Corrected engineering artifact

Checkpoint:

```text
checkpoints/deployment_v2_columns_handoff/student_best_checkpoint.pt
sha256 5f03be871d173f2cad29cd8777a8906db1ad61ffa077768c92172c5bae998604
```

ONNX:

```text
exported/lateral_policy_v2_key7_aligned_r1/policy.onnx
sha256 1995c8166d92e0973178084ddf10b290ad1604009b215453e6e85b14fa9a2109
```

Offline export results:

- fixed input/output shapes: `[1,93] -> [1,16]`;
- ONNX/eager maximum absolute error: `9.5367431640625e-07`;
- exact graph-level zero wheel outputs;
- CPU inference p95: `0.0141177443 ms`;
- key-7 normalized rotation maximum: `0.44979665` standard deviations;
- first applied physical leg target delta: `0.10625207 rad`.

Evidence:

- `reports/24_v2_r1_export_validation.json`
- `reports/24_key7_handoff_v2_r1_export.json`

## Remaining acceptance blocker

This artifact is not approved for the real robot.  The corrected model has not
yet completed an Isaac closed-loop gate because the machine currently loads
NVIDIA kernel module `580.159.03` while the installed module/userspace version
is `580.173.02`.  Isaac fails before environment creation with CUDA error 804.
No system or deployment files were changed.

After reboot, verify `nvidia-smi` and `torch.cuda.is_available()`, then run the
post-reboot sequence in `README.md`: standing-reference smoke, seven-reference
student gate, genuine MPPI standing-label collection, retraining, export,
handoff replay, and a fresh seven-reference gate.  Only a model that passes
those simulator gates should be considered for a later, separately authorized
hardware test.
