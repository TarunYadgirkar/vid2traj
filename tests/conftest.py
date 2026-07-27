"""Shared fixtures for the vid2traj acceptance suite.

Conventions used throughout the tests:
  * quaternions are (w, x, y, z)
  * the world frame is the robot base frame
  * a camera extrinsic `T_cam_world` maps world points into the camera frame
"""

from __future__ import annotations

import numpy as np
import pytest

from vid2traj import load_embodiment
from vid2traj.render.synth import make_demo_joint_trajectory, render_marker_clip

FPS = 30
N_FRAMES = 90


@pytest.fixture(scope="session")
def panda():
    return load_embodiment("franka_panda")


@pytest.fixture(scope="session")
def so101():
    return load_embodiment("so101")


@pytest.fixture(scope="session")
def synth_clip(panda, tmp_path_factory):
    """A deterministic synthetic clip plus its exact ground-truth EE poses."""
    out = tmp_path_factory.mktemp("synth")
    joints = make_demo_joint_trajectory(panda, n_frames=N_FRAMES, seed=0)
    return render_marker_clip(panda, joints, out / "demo.mp4", fps=FPS)


def quat_angle_deg(q_a: np.ndarray, q_b: np.ndarray) -> np.ndarray:
    """Geodesic angle (degrees) between two arrays of (w,x,y,z) quaternions."""
    q_a = q_a / np.linalg.norm(q_a, axis=-1, keepdims=True)
    q_b = q_b / np.linalg.norm(q_b, axis=-1, keepdims=True)
    dot = np.abs(np.sum(q_a * q_b, axis=-1))
    return np.degrees(2.0 * np.arccos(np.clip(dot, -1.0, 1.0)))


def rmse(err: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(err))))
