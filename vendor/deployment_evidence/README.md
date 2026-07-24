# pcbC Baseline + Residual Sim2Sim Bundle

This directory contains the runtime inputs and source references needed to add
the residual policy to the existing MuJoCo baseline deployment.

## Important

`runtime/models/residual_candidate.onnx` is only an integration candidate. It
was exported before the latest y/yaw sampling and reward changes. Replace it
with a newly exported final residual policy before evaluating final behavior.

## Runtime architecture

Run two policies and one deterministic state machine:

1. Before ready, execute the realtime baseline output.
2. When ready first opens, latch the current baseline raw output.
3. After ready, execute:

   `final_raw = latched_baseline_raw + residual_scale * residual_raw`

4. Apply normal action scaling and joint mapping to `final_raw`.

Do not replace the baseline model with the residual model. The residual output
is a correction, not a complete motor command.

## Observation history

The baseline and residual models both receive 93-D observations, but their
previous-action fields have different meanings:

- Baseline previous action: previous `final_raw`.
- Residual previous action: previous `residual_raw`.

Maintain these as separate 16-D buffers.

## Contents

- `deployment_manifest.yaml`: authoritative deployment contract.
- `runtime/`: files needed by the MuJoCo controller.
- `reference/`: training-side source snapshots and candidate-run parameters.
- `validation/`: hashes and a location for future golden traces.

## Replacing the candidate residual

After exporting the final residual policy:

```bash
cp /path/to/final/exported/policy.onnx \
  runtime/models/residual_candidate.onnx
sha256sum runtime/models/residual_candidate.onnx
```

Then update `validation/SHA256SUMS`.

## Golden trace

Before final integration, export a short IsaacLab play trace containing:

- baseline observation
- residual observation
- baseline raw action
- residual raw action
- final raw action
- final joint targets
- ready flag
- velocity command

Use this trace to compare the C++ implementation frame by frame.

