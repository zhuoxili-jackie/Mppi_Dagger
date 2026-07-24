# Gate 0 contract and asset audit

Status: **passed for the standalone contract and vendored runtime tree**.

## Runtime evidence

- Frozen student ABI: raw `float32 [1,93]` input named `obs`; raw
  `float32 [1,16]` output named `actions`.
- Policy observation semantics: legacy fixed first-frame motion prefix,
  reference index 0, SHA256
  `b0d19be2c44418455996fc707fbd625afab467018657bf10009a03d9189335ae`.
- Control/reference timebase: 50 Hz (`sim_dt=0.005`, decimation 4), one
  reference frame per control step.
- Policy joint order and the policy/runtime permutation are frozen in
  `configs/deployment_contract.yaml` and covered by a sentinel round-trip
  test.
- Action mode: `hard_zero`.  Twelve leg position raw actions are learned;
  outputs 12:16 are exact zeros.  This is an intentional migration semantic:
  the legacy lateral actor used learned wheel velocity, while the migration
  acceptance contract requires hard-zero wheel output.
- Previous action means the action actually executed after wrapper and safety
  shield; reset value is sixteen zeros.

## Evidence conflict

The copied legacy deployment manifest describes a 93D-to-16D bundle, but its
named baseline and residual ONNX files were absent from the source checkout.
Therefore graph metadata could not be independently read and the manifest is
not treated as stronger evidence than executable environment code.  No 92D
graph is silently padded or presented as compatible.

## Frozen assets

- Runtime URDF SHA256:
  `8db81376ab48c2f78c9f17af47704ea726fddc64c93e9a7db9651436c76a22fe`
- Car-trunk USD SHA256:
  `0e52ce79c269707ea1a7702cef1faa9ab10900e6ada24b4a9d8e21b7989f8ac8`
- Car-trunk STL SHA256:
  `a2156a278ab9847c5ce586b1f152178813f094e772c6445f1dd3b78e846fd378`
- Six 708 NPZ files are finite, each has 332 frames, 16 joints, 17 bodies and
  normalized wxyz quaternions.  All share the verified reset frame.
- The standalone snapshot contains all 30 STL files in the pcbC source asset
  tree, plus USD, OBJ, URDF and CSV files.

The machine-readable audit at `reports/00_audit_state.json` contains the full
per-file SHA256 inventory.

## Standalone boundary

The migration result is outside the original MoveIt repository.  Runtime
configuration paths resolve only beneath `lateral_mppi_dagger`; the active
source/config scan contains no links to the original MoveIt or
`RL_augmented_MPC` checkouts, and no external symlink exists.

An isolation test copied the tree to a fresh `/tmp` location while excluding
all generated outputs.  Asset audit, reference validation, 15 unit tests and
real Isaac environment creation succeeded.  The environment recorded its
`robot_lab` import from the temporary copy, and provenance recorded
`original_repository_accessed=false`.

Isaac Lab 2.3.0 app files and source extensions are also vendored and import
checked.  External installed binaries/framework packages (Isaac Sim,
CUDA/PyTorch and normal Python dependencies) remain required; no source or
asset is loaded from the original MoveIt project.

## Gate limitation

Passing Gate 0 does not certify a deployable policy.  The bounded
ReferenceWBC smoke validates mapping and environment integration only.
Formal expert reliability, observability, BC closed-loop, at least R1-R3
DAgger rounds, final ONNX closed-loop evaluation and final acceptance metrics
remain required.
