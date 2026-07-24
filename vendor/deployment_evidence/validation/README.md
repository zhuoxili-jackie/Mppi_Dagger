# Validation

Add a future `golden_trace.npz` here. It should contain several seconds of
IsaacLab play data for standing, y-only, and yaw-only commands.

At minimum, record:

- `baseline_observation`
- `residual_observation`
- `baseline_raw`
- `residual_raw`
- `final_raw`
- `joint_targets`
- `ready`
- `velocity_command`

The MuJoCo controller should reproduce the policy observations and raw actions
within normal floating-point tolerance before physics behavior is evaluated.

