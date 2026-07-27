"""Deterministic synthetic clips with exact ground truth.

Renders a known-size ArUco marker rigidly attached to the robot's wrist frame,
following a known joint trajectory, through a virtual pinhole camera. No GL
context is involved (see DECISIONS D2), so this runs headless anywhere and is
reproducible frame-for-frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from ..camera import CameraModel
from ..config import Embodiment
from ..math3d import invert_transform, make_transform, split_transform
from ..perception.marker import DEFAULT_DICT, DEFAULT_MARKER_SIZE, marker_object_points
from ..retarget.kinematics import Kinematics
from ..video import write_video

BACKGROUND_GRAY = 205
MARKER_PIXELS = 400
QUIET_ZONE_FRACTION = 0.4
TARGET_MARKER_ID = 0
DISTRACTOR_MARKER_ID = 5
CAMERA_DISTANCE_M = 0.85


@dataclass
class SyntheticClip:
    video_path: Path
    camera: CameraModel
    fps: float
    joints: np.ndarray
    ee_positions: np.ndarray
    ee_quats: np.ndarray
    wrist_positions: np.ndarray
    wrist_quats: np.ndarray
    marker_size: float


def make_demo_joint_trajectory(
    embodiment: Embodiment, n_frames: int = 90, seed: int = 0
) -> np.ndarray:
    """A smooth, in-limits, collision-free reach-and-return around the home pose.

    Built from a fixed set of seeded sinusoids: no integration, no randomness at
    playback, so the same seed always yields the same array.
    """
    rng = np.random.default_rng(seed)
    n_joints = embodiment.n_joints
    home = embodiment.home_joints

    amplitudes = rng.uniform(0.10, 0.22, size=n_joints)
    periods = rng.uniform(1.6, 3.2, size=n_joints)
    phases = rng.uniform(0.0, 2 * np.pi, size=n_joints)

    t = np.linspace(0.0, 1.0, n_frames)[:, None]
    # A raised-cosine envelope starts and ends at rest, so the clip has no
    # velocity step at its boundaries.
    envelope = 0.5 * (1.0 - np.cos(2 * np.pi * np.clip(t, 0.0, 1.0)))
    wave = np.sin(2 * np.pi * t / periods[None, :] * 3.0 + phases[None, :])

    joints = home[None, :] + envelope * amplitudes[None, :] * wave
    margin = 0.05
    low = embodiment.joint_limits[:, 0] + margin
    high = embodiment.joint_limits[:, 1] - margin
    return np.clip(joints, low, high)


def _padded_marker(marker_id: int, dictionary: int = DEFAULT_DICT) -> tuple[np.ndarray, int]:
    aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary)
    marker = cv2.aruco.generateImageMarker(aruco_dict, marker_id, MARKER_PIXELS)
    pad = int(MARKER_PIXELS * QUIET_ZONE_FRACTION)
    canvas = np.full((MARKER_PIXELS + 2 * pad, MARKER_PIXELS + 2 * pad), 255, np.uint8)
    canvas[pad : pad + MARKER_PIXELS, pad : pad + MARKER_PIXELS] = marker
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR), pad


def _draw_marker(
    scene: np.ndarray,
    sprite: np.ndarray,
    pad: int,
    camera: CameraModel,
    T_cam_marker: np.ndarray,
    marker_size: float,
) -> bool:
    """Paste the marker into the scene under the given camera-frame pose."""
    if T_cam_marker[2, 3] <= 1e-3:
        return False  # behind the camera

    rvec, _ = cv2.Rodrigues(T_cam_marker[:3, :3])
    image_points, _ = cv2.projectPoints(
        marker_object_points(marker_size),
        rvec,
        T_cam_marker[:3, 3],
        camera.intrinsics,
        camera.distortion,
    )
    image_points = image_points.reshape(-1, 2).astype(np.float32)
    if not np.all(np.isfinite(image_points)):
        return False

    height, width = scene.shape[:2]
    inside = (
        (image_points[:, 0] > -width)
        & (image_points[:, 0] < 2 * width)
        & (image_points[:, 1] > -height)
        & (image_points[:, 1] < 2 * height)
    )
    if not inside.all():
        return False

    side = MARKER_PIXELS
    source = np.array(
        [[pad, pad], [pad + side, pad], [pad + side, pad + side], [pad, pad + side]],
        np.float32,
    )
    homography = cv2.getPerspectiveTransform(source, image_points)
    warped = cv2.warpPerspective(sprite, homography, (width, height), borderValue=(0, 0, 0))
    mask = cv2.warpPerspective(
        np.full(sprite.shape[:2], 255, np.uint8), homography, (width, height)
    )
    scene[mask > 0] = warped[mask > 0]
    return True


def _camera_for(wrist_pose: np.ndarray, width: int, height: int) -> CameraModel:
    """Put the camera in front of the marker face so it is visible at rest."""
    position = wrist_pose[:3, 3]
    normal = wrist_pose[:3, :3] @ np.array([0.0, 0.0, 1.0])
    eye = position + normal * CAMERA_DISTANCE_M
    up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(normal / np.linalg.norm(normal), up)) > 0.95:
        up = np.array([0.0, 1.0, 0.0])
    return CameraModel.looking_at(eye, position, width=width, height=height, up=up)


def render_marker_clip(
    embodiment: Embodiment,
    joints: np.ndarray,
    out_path: str | Path,
    fps: float = 30.0,
    marker_size: float = DEFAULT_MARKER_SIZE,
    width: int = 960,
    height: int = 720,
    occlude_frames: Iterable[int] | None = None,
    offscreen_frames: Iterable[int] | None = None,
    distractor: bool = False,
    camera: CameraModel | None = None,
) -> SyntheticClip:
    """Render `joints` as a marker-on-the-wrist video plus its ground truth."""
    joints = np.asarray(joints, dtype=float)
    occluded = set(occlude_frames or ())
    offscreen = set(offscreen_frames or ())

    kinematics = Kinematics(embodiment)
    wrist_from_ee = invert_transform(embodiment.wrist_to_ee)

    ee_positions = np.zeros((len(joints), 3))
    ee_quats = np.zeros((len(joints), 4))
    wrist_poses = []
    for i, q in enumerate(joints):
        position, quat = kinematics.fk(q)
        ee_positions[i], ee_quats[i] = position, quat
        wrist_poses.append(make_transform(position, quat) @ wrist_from_ee)

    camera = camera or _camera_for(wrist_poses[0], width, height)
    sprite, pad = _padded_marker(TARGET_MARKER_ID)
    distractor_sprite, distractor_pad = _padded_marker(DISTRACTOR_MARKER_ID)

    frames = []
    wrist_positions = np.zeros((len(joints), 3))
    wrist_quats = np.zeros((len(joints), 4))

    for i, pose in enumerate(wrist_poses):
        wrist_positions[i], wrist_quats[i] = split_transform(pose)
        scene = np.full((height, width, 3), BACKGROUND_GRAY, np.uint8)

        if distractor:
            # A second subject, moving on its own path, never the tracked one.
            offset = make_transform(
                [0.28 + 0.05 * np.sin(i * 0.15), -0.10, 0.12 * np.cos(i * 0.11)],
                [1.0, 0.0, 0.0, 0.0],
            )
            _draw_marker(
                scene,
                distractor_sprite,
                distractor_pad,
                camera,
                camera.T_cam_world @ (pose @ offset),
                marker_size,
            )

        target_pose = pose
        if i in offscreen:
            leave = make_transform([1.6, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
            target_pose = pose @ leave

        T_cam_marker = camera.T_cam_world @ target_pose
        drawn = _draw_marker(scene, sprite, pad, camera, T_cam_marker, marker_size)

        if drawn and i in occluded:
            _occlude(scene, camera, T_cam_marker, marker_size)

        frames.append(scene)

    video_path = write_video(frames, out_path, fps=fps)
    return SyntheticClip(
        video_path=video_path,
        camera=camera,
        fps=fps,
        joints=joints,
        ee_positions=ee_positions,
        ee_quats=ee_quats,
        wrist_positions=wrist_positions,
        wrist_quats=wrist_quats,
        marker_size=marker_size,
    )


def _occlude(
    scene: np.ndarray, camera: CameraModel, T_cam_marker: np.ndarray, marker_size: float
) -> None:
    """Cover the marker with an opaque blob, as a passing hand or object would."""
    rvec, _ = cv2.Rodrigues(T_cam_marker[:3, :3])
    corners, _ = cv2.projectPoints(
        marker_object_points(marker_size * 1.6),
        rvec,
        T_cam_marker[:3, 3],
        camera.intrinsics,
        camera.distortion,
    )
    cv2.fillConvexPoly(scene, corners.reshape(-1, 2).astype(np.int32), (90, 90, 90))
