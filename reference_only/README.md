# Reference-only sources

Nothing below this directory is a runtime dependency of
`lateral_mppi_dagger`.

`RTWholeBodyMPPI` is retained only so that its MPPI algorithm can be inspected
while implementing the Isaac Lab teacher.  Do not:

- install `legged_mppi` into the Isaac Lab environment;
- add this directory to `PYTHONPATH`;
- import `whole_body_mppi` from project code;
- load its MuJoCo models, Go1 actions, or task definitions at runtime.

The upstream implementation uses MuJoCo rollouts driven from NumPy/SciPy and
CPU workers, and its action space is the twelve Go1 leg joints.  It does not
implement this project's Isaac Lab environment, 708-frame references, car
trunk interaction, four locked wheel joints, fixed `93 -> 16` policy ABI,
DAgger pipeline, or PT/ONNX export.  Those integrations are reimplemented in
`src/lateral_mppi_dagger`.

The useful reference concepts are limited to receding-horizon warm starts,
sampled action trajectories, trajectory-cost weighting, temporal smoothing,
and rollout diagnostics.  The production implementation uses Isaac Lab
parallel environments and PyTorch tensors on the GPU.  The first formal
teacher is MPPI only; DWMPC is intentionally outside the first migration.

Archived checkout:

- upstream: `https://github.com/jrapudg/RTWholeBodyMPPI.git`
- commit: `55e6dc35d042ed5f175ad6ae312c886282f32529`
- policy: reference only, never installed or imported

