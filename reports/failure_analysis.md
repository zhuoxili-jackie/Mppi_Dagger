# Failure analysis

## Closed-loop regressions

- Standard R3 checkpoint: 0/
  50 full-horizon episodes, mean horizon fraction
  0.257133. It was rejected.
- Conservative R3 checkpoint: 50/
  50 full-horizon episodes. It was selected for the
  engineering bundle.
- Final ONNX: 50/
  50 full-horizon episodes, but strict tracking success
  0/50.

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
