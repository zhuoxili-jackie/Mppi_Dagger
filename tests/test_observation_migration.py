from __future__ import annotations

import numpy as np

from lateral_mppi_dagger.data.observation_migration import (
    migrate_clean_observation_rows_to_columns,
    migrate_noisy_training_observation_rows_to_columns,
    rotation_rows_to_deployment_columns,
)


def test_identity_rows_migrate_to_key7_columns() -> None:
    rows = np.asarray([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    expected = np.asarray([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    np.testing.assert_array_equal(rotation_rows_to_deployment_columns(rows), expected)


def test_rotation_migration_preserves_non_rotation_channels_and_noise() -> None:
    clean = np.zeros((2, 93), dtype=np.float32)
    clean[:, 32:38] = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    clean[:, :32] = np.arange(32, dtype=np.float32)
    noise = np.arange(186, dtype=np.float32).reshape(2, 93) * 1.0e-5
    train = clean + noise
    migrated_clean = migrate_clean_observation_rows_to_columns(clean)
    migrated_train = migrate_noisy_training_observation_rows_to_columns(
        clean,
        train,
    )
    mask = np.ones(93, dtype=bool)
    mask[32:38] = False
    np.testing.assert_array_equal(migrated_clean[:, mask], clean[:, mask])
    np.testing.assert_array_equal(migrated_train[:, mask], train[:, mask])
    expected_noise = noise[:, 32:38][:, [0, 1, 3, 4, 2, 5]]
    np.testing.assert_allclose(
        migrated_train[:, 32:38] - migrated_clean[:, 32:38],
        expected_noise,
        atol=1.0e-7,
        rtol=0.0,
    )
