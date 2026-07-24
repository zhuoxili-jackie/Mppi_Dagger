# Vendored runtime snapshot

This directory is the runtime boundary for source and data copied from the
reference MoveIt/Isaac project.  Training and evaluation must not read that
original repository.

Included:

- `IsaacLab`: the Isaac Lab 2.3.0 app files plus the `isaaclab`,
  `isaaclab_assets`, `isaaclab_tasks`, `isaaclab_rl` and `isaaclab_mimic`
  extension source snapshots;
- `robot_lab/robot_lab`: the Python package snapshot used to register and run
  the pcbC lateral task;
- `robot_lab/config`: extension metadata required by `robot_lab.assets`;
- `robot_lab/data/Robots/pcbC`: the complete pcbC asset tree, including every
  STL, USD, OBJ and URDF present at migration time;
- `robot_lab/data/Motions/pcbc_lateral_708`: all six 708 NPZ references and
  their CSV sources;
- `deployment_evidence`: the legacy deployment manifest and reference evidence
  used only for contract auditing.

`robot_lab/LICENSE` contains the upstream Apache-2.0 license.  No source file
from `RL_augmented_MPC` is copied: that repository was used as an algorithmic
reference only, and the standalone MPPI/DAgger implementation lives under
`src/lateral_mppi_dagger`.

The only expected dependencies outside this directory are installed binary
runtime software such as Isaac Sim, CUDA/PyTorch and ordinary Python packages.
Local scripts prepend the vendored Isaac Lab and robot_lab sources to
`sys.path` and reject either package if it is imported from another location.
