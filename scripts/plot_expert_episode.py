#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _bootstrap import ROOT, write_json

from lateral_mppi_dagger.data.dataset import load_manifest
from lateral_mppi_dagger.data.schema import read_episode_shard
from lateral_mppi_dagger.reference.loader import ReferenceSet


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        axis=-1,
    )


def quat_conjugate(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value).copy()
    result[..., 1:] *= -1.0
    return result


def quat_rotate(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    zeros = np.zeros(vector.shape[:-1] + (1,), dtype=vector.dtype)
    pure = np.concatenate((zeros, vector), axis=-1)
    return quat_multiply(
        quat_multiply(quaternion, pure),
        quat_conjugate(quaternion),
    )[..., 1:]


def select_record(dataset: Path, episode_id: str | None) -> dict:
    records = load_manifest(dataset)
    if episode_id is None:
        if len(records) != 1:
            raise ValueError(
                f"Dataset has {len(records)} episodes; pass --episode-id to select one."
            )
        return records[0]
    selected = [record for record in records if record["episode_id"] == episode_id]
    if len(selected) != 1:
        raise ValueError(f"Expected one episode_id={episode_id!r}, found {len(selected)}.")
    return selected[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot MPPI expert tracking/contact diagnostics.")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--episode-id", default=None)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/expert_plots")
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    record = select_record(dataset, args.episode_id)
    shard = read_episode_shard(dataset / record["path"])
    arrays = shard.arrays
    references = ReferenceSet.from_config()
    ref_id = int(record["ref_id"])
    reference = references[ref_id]
    frames = arrays["ref_frame"]
    actual_base_pos = arrays["base_pose_w"][:, :3]
    actual_base_quat = arrays["base_pose_w"][:, 3:7]

    ref_base_pos = reference.body_pos_w[frames, 0]
    ref_base_quat = reference.body_quat_w[frames, 0]
    alignment_quat = quat_multiply(
        actual_base_quat[0],
        quat_conjugate(ref_base_quat[0]),
    )
    alignment_batch = np.broadcast_to(alignment_quat, (frames.shape[0], 4))
    target_base_pos = actual_base_pos[0] + quat_rotate(
        alignment_batch,
        ref_base_pos - ref_base_pos[0],
    )
    target_base_quat = quat_multiply(alignment_batch, ref_base_quat)
    base_position_error = actual_base_pos - target_base_pos
    base_orientation_error = 2.0 * np.arccos(
        np.clip(
            np.abs(np.sum(actual_base_quat * target_base_quat, axis=-1)),
            0.0,
            1.0,
        )
    )

    wheel_ids = [references.body_order.index(name) for name in (
        "FL_foot_link",
        "FR_foot_link",
        "RL_foot_link",
        "RR_foot_link",
    )]
    ref_wheel_pos = reference.body_pos_w[frames][:, wheel_ids]
    target_wheel_pos = actual_base_pos[0, None, :] + quat_rotate(
        np.broadcast_to(alignment_quat, (frames.shape[0], 4, 4)),
        ref_wheel_pos - ref_base_pos[0, None, :],
    )
    actual_wheel_pos = arrays["wheel_body_pose_w"][..., :3]
    wheel_error = np.linalg.norm(actual_wheel_pos - target_wheel_pos, axis=-1)
    contact_mismatch = np.not_equal(
        arrays["desired_contact"],
        arrays["measured_contact"],
    ).astype(np.float32)
    time_s = arrays["sim_time"]

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = record["episode_id"]
    csv_path = output_dir / f"{stem}.csv"
    png_path = output_dir / f"{stem}.png"
    json_path = output_dir / f"{stem}.json"
    csv_values = np.column_stack(
        (
            time_s,
            base_position_error,
            base_orientation_error,
            wheel_error,
            contact_mismatch,
            arrays["solve_ms"],
            arrays["mppi_rollout_termination_rate"],
        )
    )
    csv_header = ",".join(
        (
            "time_s",
            "base_error_x_m",
            "base_error_y_m",
            "base_error_z_m",
            "base_orientation_error_rad",
            "wheel_error_fl_m",
            "wheel_error_fr_m",
            "wheel_error_rl_m",
            "wheel_error_rr_m",
            "contact_mismatch_fl",
            "contact_mismatch_fr",
            "contact_mismatch_rl",
            "contact_mismatch_rr",
            "solve_ms",
            "rollout_termination_rate",
        )
    )
    np.savetxt(csv_path, csv_values, delimiter=",", header=csv_header, comments="")

    figure, axes = plt.subplots(5, 1, figsize=(12, 16), sharex=True)
    for axis, label in zip(range(3), ("x", "y", "z"), strict=True):
        axes[0].plot(time_s, base_position_error[:, axis], label=label)
    axes[0].set_ylabel("base error [m]")
    axes[0].legend(ncol=3)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time_s, base_orientation_error)
    axes[1].set_ylabel("orientation [rad]")
    axes[1].grid(True, alpha=0.3)

    for wheel, label in enumerate(("FL", "FR", "RL", "RR")):
        axes[2].plot(time_s, wheel_error[:, wheel], label=label)
    axes[2].set_ylabel("wheel pos err [m]")
    axes[2].legend(ncol=4)
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(time_s, contact_mismatch.mean(axis=1), label="mismatch fraction")
    axes[3].plot(
        time_s,
        arrays["mppi_rollout_termination_rate"],
        label="rollout termination rate",
    )
    axes[3].set_ylabel("fraction")
    axes[3].set_ylim(-0.05, 1.05)
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    axes[4].plot(time_s, arrays["solve_ms"], label="solve")
    axes[4].set_ylabel("solve [ms]")
    axes[4].set_xlabel("simulation time [s]")
    axes[4].grid(True, alpha=0.3)
    figure.suptitle(
        f"{record['episode_id']} | ref={ref_id} | target_vy={reference.target_vy:+.2f} m/s"
    )
    figure.tight_layout()
    figure.savefig(png_path, dpi=150)
    plt.close(figure)

    summary = {
        "schema_version": "pcbc-expert-tracking-plot-v1",
        "episode_id": record["episode_id"],
        "ref_id": ref_id,
        "target_vy": reference.target_vy,
        "steps": int(frames.shape[0]),
        "base_position_rmse_m": np.sqrt(np.mean(np.square(base_position_error), axis=0)).tolist(),
        "base_position_max_abs_m": np.max(np.abs(base_position_error), axis=0).tolist(),
        "base_orientation_rmse_rad": float(np.sqrt(np.mean(np.square(base_orientation_error)))),
        "wheel_position_rmse_m": np.sqrt(np.mean(np.square(wheel_error), axis=0)).tolist(),
        "contact_mismatch_rate": float(contact_mismatch.mean()),
        "mean_solve_ms": float(np.mean(arrays["solve_ms"])),
        "max_rollout_termination_rate": float(
            np.max(arrays["mppi_rollout_termination_rate"])
        ),
        "wheel_action_exact_zero": bool(
            np.array_equal(
                arrays["executed_action16"][:, 12:],
                np.zeros_like(arrays["executed_action16"][:, 12:]),
            )
        ),
        "csv": str(csv_path),
        "plot": str(png_path),
    }
    write_json(json_path, summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()

