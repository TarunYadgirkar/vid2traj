"""Lift per-frame camera-space observations into a fixed world frame."""

from __future__ import annotations

import numpy as np

from ..camera import CameraModel
from ..math3d import make_transform, split_transform
from ..types import WristObservation, WristTrajectory


def observations_to_world(
    observations: list[WristObservation | None],
    camera: CameraModel,
    fps: float,
) -> WristTrajectory:
    """Apply the camera extrinsic; unobserved frames are marked, not invented."""
    n_frames = len(observations)
    positions = np.zeros((n_frames, 3))
    quats = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n_frames, 1))
    apertures = np.ones(n_frames)
    visible = np.zeros(n_frames, dtype=bool)

    T_world_cam = camera.T_world_cam
    for i, obs in enumerate(observations):
        if obs is None:
            continue
        T_world_wrist = T_world_cam @ make_transform(obs.position, obs.quat)
        positions[i], quats[i] = split_transform(T_world_wrist)
        apertures[i] = obs.aperture
        visible[i] = True

    return WristTrajectory(
        positions=positions,
        quats=quats,
        visible=visible,
        apertures=apertures,
        fps=fps,
    )
