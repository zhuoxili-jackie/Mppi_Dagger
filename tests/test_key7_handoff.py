from __future__ import annotations

import numpy as np

from lateral_mppi_dagger.config import load_yaml
from lateral_mppi_dagger.evaluation.key7_handoff import (
    build_key7_handoff_observation,
)


def test_key7_handoff_golden_matches_deployment_columns_and_pose() -> None:
    contract = load_yaml("configs/deployment_contract.yaml")
    golden = load_yaml("configs/key7_handoff_golden.yaml")
    handoff = build_key7_handoff_observation(contract, golden)
    np.testing.assert_allclose(
        handoff.aligned_reference_quaternion_wxyz,
        np.asarray(
            golden["nominal_handoff"][
                "expected_aligned_reference_quaternion_wxyz"
            ],
            dtype=np.float32,
        ),
        atol=float(golden["gates"]["rotation_atol"]),
        rtol=0.0,
    )
    np.testing.assert_allclose(
        handoff.observation[0, 32:38],
        np.asarray(
            golden["nominal_handoff"]["expected_rotation_columns6"],
            dtype=np.float32,
        ),
        atol=float(golden["gates"]["rotation_atol"]),
        rtol=0.0,
    )
    np.testing.assert_array_equal(
        handoff.observation[0, 53:57],
        np.zeros(4, dtype=np.float32),
    )
    np.testing.assert_array_equal(
        handoff.observation[0, 73:93],
        np.zeros(20, dtype=np.float32),
    )
