"""Build the side-by-side HTML visualization.

One self-contained file: both videos are embedded as data URIs and every trace
is drawn from an inlined JSON payload, so the result can be opened from disk,
emailed, or committed without breaking.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np

from ..config import Embodiment, load_embodiment
from ..render.robot import render_robot_video
from ..types import RobotTrajectory, SafetyReport, WristTrajectory
from ..video import probe_video

TEMPLATE_PATH = Path(__file__).with_name("template.html")


def _data_uri(path: Path) -> str:
    return "data:video/mp4;base64," + base64.b64encode(Path(path).read_bytes()).decode("ascii")


def build_payload(
    trajectory: RobotTrajectory,
    embodiment: Embodiment,
    report: SafetyReport | None = None,
    wrist: WristTrajectory | None = None,
    ground_truth_positions: np.ndarray | None = None,
    task: str = "",
) -> dict:
    n_frames = len(trajectory)
    report = report or SafetyReport(n_frames=n_frames)

    flags = {
        "held": sorted(set(report.held_frames)),
        "velocity": sorted(set(report.velocity_clamped_frames)),
        "limit": sorted(set(report.limit_clamped_frames)),
        "collision": sorted(set(report.collision_frames)),
        "unobserved": sorted(wrist.missing_frames) if wrist is not None else [],
    }

    dt = 1.0 / trajectory.fps
    velocity = np.abs(np.diff(trajectory.joints, axis=0) / dt)
    peak_velocity = velocity.max(axis=0) if len(velocity) else np.zeros(embodiment.n_joints)

    accuracy = None
    if ground_truth_positions is not None:
        error = np.linalg.norm(trajectory.ee_positions - ground_truth_positions, axis=1)
        accuracy = {
            "rmse_mm": float(np.sqrt(np.mean(error**2)) * 1000),
            "max_mm": float(error.max() * 1000),
            "per_frame_mm": np.round(error * 1000, 3).tolist(),
        }

    return {
        "task": task,
        "embodiment": {
            "name": embodiment.name,
            "robotType": embodiment.robot_type,
            "joints": list(embodiment.arm_joints),
            "limits": embodiment.joint_limits.tolist(),
            "velocityLimits": embodiment.velocity_limits.tolist(),
            "gripperLimits": list(embodiment.gripper.limits),
        },
        "fps": float(trajectory.fps),
        "nFrames": n_frames,
        "joints": np.round(trajectory.joints, 5).tolist(),
        "gripper": np.round(trajectory.gripper, 5).tolist(),
        "eePositions": np.round(trajectory.ee_positions, 5).tolist(),
        "peakVelocity": np.round(peak_velocity, 4).tolist(),
        "flags": flags,
        "clampMagnitude": [round(v, 4) for v in report.velocity_clamp_magnitude]
        or [0.0] * n_frames,
        "observedFraction": report.observed_fraction,
        "accuracy": accuracy,
    }


def build_visualization(
    dataset_dir: str | Path,
    source_video: str | Path,
    out_path: str | Path,
    embodiment: Embodiment | str,
    trajectory: RobotTrajectory | None = None,
    report: SafetyReport | None = None,
    wrist: WristTrajectory | None = None,
    robot_video: str | Path | None = None,
    ground_truth_positions: np.ndarray | None = None,
    task: str = "",
) -> Path:
    """Render the visualization for an exported dataset.

    `trajectory` may be passed directly; otherwise it is read back from the
    dataset on disk, which also proves the export round-trips.
    """
    embodiment = load_embodiment(embodiment)
    dataset_dir = Path(dataset_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if trajectory is None:
        trajectory = read_trajectory_from_dataset(dataset_dir, embodiment)

    probe = probe_video(source_video)
    if robot_video is None:
        # Match the source resolution so the two panels line up exactly.
        robot_video = out_path.with_name(f"{out_path.stem}_robot.mp4")
        render_robot_video(
            trajectory, embodiment, robot_video, width=probe["width"], height=probe["height"]
        )

    payload = build_payload(
        trajectory,
        embodiment,
        report=report,
        wrist=wrist,
        ground_truth_positions=ground_truth_positions,
        task=task,
    )
    payload["source"] = {"width": probe["width"], "height": probe["height"]}

    html = TEMPLATE_PATH.read_text()
    html = html.replace("/*__PAYLOAD__*/null", json.dumps(payload))
    html = html.replace("__SOURCE_VIDEO__", _data_uri(Path(source_video)))
    html = html.replace("__ROBOT_VIDEO__", _data_uri(Path(robot_video)))
    out_path.write_text(html)
    return out_path


def read_trajectory_from_dataset(dataset_dir: Path, embodiment: Embodiment) -> RobotTrajectory:
    """Reconstruct a trajectory from an exported dataset, using FK for EE poses."""
    import pyarrow.parquet as pq

    from ..retarget.kinematics import Kinematics

    dataset_dir = Path(dataset_dir)
    info = json.loads((dataset_dir / "meta" / "info.json").read_text())
    parquet_files = sorted(dataset_dir.glob("data/**/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"no parquet data under {dataset_dir}")

    table = pq.read_table(parquet_files[0])
    state = np.stack(table.column("observation.state").to_pylist()).astype(float)
    joints, gripper = state[:, : embodiment.n_joints], state[:, embodiment.n_joints]

    kinematics = Kinematics(embodiment)
    positions = np.zeros((len(joints), 3))
    quats = np.zeros((len(joints), 4))
    for i, q in enumerate(joints):
        positions[i], quats[i] = kinematics.fk(q)

    return RobotTrajectory(
        joints=joints,
        gripper=gripper,
        ee_positions=positions,
        ee_quats=quats,
        fps=float(info["fps"]),
    )
