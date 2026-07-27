"""No velocity or acceleration discontinuities above the embodiment's limits."""

from __future__ import annotations

import numpy as np
import pytest

from vid2traj import run_pipeline


@pytest.fixture(scope="module")
def emitted(synth_clip, panda, tmp_path_factory):
    return run_pipeline(
        video=synth_clip.video_path,
        embodiment=panda,
        out_dir=tmp_path_factory.mktemp("smooth") / "ds",
        frontend="marker",
        camera=synth_clip.camera,
        fps=synth_clip.fps,
    )


def test_joint_velocity_within_limits(emitted, panda):
    dt = 1.0 / emitted.robot_trajectory.fps
    vel = np.diff(emitted.robot_trajectory.joints, axis=0) / dt
    worst = np.abs(vel).max(axis=0)
    limit = panda.velocity_limits
    assert np.all(worst <= limit + 1e-6), (
        f"joint velocity exceeds limit: worst={np.round(worst, 3)} limit={np.round(limit, 3)}"
    )


def test_joint_acceleration_within_limits(emitted, panda):
    dt = 1.0 / emitted.robot_trajectory.fps
    vel = np.diff(emitted.robot_trajectory.joints, axis=0) / dt
    acc = np.diff(vel, axis=0) / dt
    worst = np.abs(acc).max(axis=0)
    limit = panda.acceleration_limits
    assert np.all(worst <= limit + 1e-6), (
        f"joint acceleration exceeds limit: worst={np.round(worst, 2)} limit={np.round(limit, 2)}"
    )


def test_no_ee_teleports(emitted):
    """Cartesian jump check — catches IK branch flips that joint limits miss."""
    step = np.linalg.norm(np.diff(emitted.robot_trajectory.ee_positions, axis=0), axis=1)
    assert step.max() < 0.05, f"end-effector jumps {step.max() * 1000:.0f} mm in one frame"


def test_smoothing_reduces_jitter_versus_raw(synth_clip, panda, tmp_path):
    """The smoothing stage must actually do something measurable."""
    raw = run_pipeline(
        video=synth_clip.video_path,
        embodiment=panda,
        out_dir=tmp_path / "raw",
        frontend="marker",
        camera=synth_clip.camera,
        fps=synth_clip.fps,
        smoothing=False,
    )
    smoothed = run_pipeline(
        video=synth_clip.video_path,
        embodiment=panda,
        out_dir=tmp_path / "smoothed",
        frontend="marker",
        camera=synth_clip.camera,
        fps=synth_clip.fps,
        smoothing=True,
    )

    def jerk(traj):
        return float(np.abs(np.diff(traj.joints, n=3, axis=0)).mean())

    assert jerk(smoothed.robot_trajectory) <= jerk(raw.robot_trajectory)
