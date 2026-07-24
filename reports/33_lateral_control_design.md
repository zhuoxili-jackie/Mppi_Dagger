# Lateral control diagnosis and selected design

## Control semantics

Key 7 is a velocity-command interface:

- A increments the requested `+y` velocity and D decrements it.
- The first key target is `±0.03 m/s`.
- The runtime acceleration limiter presents `±0.012`, `±0.024`, then
  `±0.030 m/s` to the network at 50 Hz.

The actuator contract is mixed:

- network outputs 0–11 are leg joint-position residuals;
- those targets are executed by joint PD (`Kp=35`, `Kd=0.8`);
- outputs 12–15 are wheel velocity slots, but this policy hard-zeros them.

Therefore the correct description is **velocity-command-conditioned leg
position policy**, not a motor-velocity controller.

## Measured causes of the current fall

1. The deployed student's training commands were only
   `±0.05, ±0.10, ±0.15 m/s`; key 7's `0.012/0.024/0.030` startup was absent.
2. A hard previous-action switch was placed at exactly `0.03 m/s`. In a
   frozen measured state, feedback grew to 0.368 rad left and 0.416 rad right.
3. All six original references used approximately 1.2 Hz cadence. Speed was
   increased mainly by step amplitude.
4. Original clearance was approximately 35–43 mm at the front trunk contact
   and 72–83 mm at the rear ground contact.
5. Recorded MPPI episodes showed rear single-support means of 131–161 N and
   p95 values up to approximately 214 N, while front mean force was only
   approximately 10–14 N.
6. The former MPPI objective used binary contact mismatch but no oriented
   front/rear normal-force or rear-overload cost.
7. A historical student completed the horizon while averaging only about
   0.002 m/s lateral speed. The old admission gate rewarded survival without
   requiring signed movement.

## Selected approach

Keep the existing deployment ABI and velocity command. A distance controller
would require a new high-level planner or persistent displacement state in the
deployment runtime. It would still need a stable low-level gait, so it does not
remove the reference problem.

Use a load-aware crawl seed:

- fixed 40 mm stride;
- speed changes cadence (`0.75 Hz` at 0.03 m/s, `1.5 Hz` at 0.06 m/s);
- one limb swings at a time;
- front detachment limited to 8 mm;
- rear clearance limited to 12 mm;
- lateral load shift toward the remaining rear support;
- small forward preload while a rear limb swings;
- exact deployment command atoms, including startup values;
- standing reference with the identical first frame.

Use MPPI to adapt that seed in Isaac, not to copy joint actions directly from
the upstream MuJoCo project. The selected design follows Whole-Body MPPI's
joint-target/smooth-control principle while adding this robot's oriented
contact-force costs.

## Implemented safety and acceptance

- Continuous command/previous-action activation; no threshold at 0.03 m/s.
- Graph-embedded 2.25 rad/s leg-target rate limit.
- Joint-specific physical residual limits: 0.12 rad front hips, 0.30 rad rear
  hips, 0.15 rad thighs, and 0.20 rad calves.
- Exact-zero wheels and exact-zero key-7 handoff.
- Front normal-support cost in world x.
- Rear overload/balance/support costs in world z.
- Correct front-x/rear-z contact-schedule inference.
- Signed progress, velocity/displacement, force p95, single/no rear support,
  front-force, physical action-step, start/stop, and parity gates.
- DAgger admission now requires nontrivial signed progress.

## Research basis

- Whole-Body MPPI: https://whole-body-mppi.github.io/
- Whole-Body MPPI paper: https://ar5iv.labs.arxiv.org/html/2409.10469
- Contact-conditioned locomotion policy: https://www.arxiv.org/pdf/2408.00776
- Trajectory optimization through contact: https://arxiv.org/abs/1607.04537
- Reaction-force-aware whole-body locomotion control:
  https://arxiv.org/abs/1909.06586

## Remaining acceptance blocker

Isaac cannot currently start because the loaded NVIDIA kernel module is
`580.159.03` while user-space NVML is `580.173`. Reboot, then execute
`STABLE_LATERAL_RUNBOOK.md`. The new trajectory remains a kinematic candidate
until the Isaac force gate passes; it is not yet a release ONNX.
