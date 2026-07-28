"""Pinhole camera model plus the world<-camera extrinsic."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .math3d import invert_transform


@dataclass(frozen=True)
class CameraModel:
    width: int
    height: int
    intrinsics: np.ndarray  # (3, 3)
    distortion: np.ndarray  # (5,)
    T_cam_world: np.ndarray  # (4, 4) world point -> camera frame

    @classmethod
    def from_fov(
        cls,
        width: int,
        height: int,
        fov_deg: float = 46.0,
        T_cam_world: np.ndarray | None = None,
    ) -> CameraModel:
        focal = 0.5 * width / np.tan(np.radians(fov_deg) / 2.0)
        intrinsics = np.array(
            [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]]
        )
        return cls(
            width=width,
            height=height,
            intrinsics=intrinsics,
            distortion=np.zeros(5),
            T_cam_world=np.eye(4) if T_cam_world is None else np.asarray(T_cam_world, dtype=float),
        )

    @classmethod
    def looking_at(
        cls,
        eye,
        target,
        width: int = 960,
        height: int = 720,
        fov_deg: float = 46.0,
        up=(0.0, 0.0, 1.0),
    ) -> CameraModel:
        """Place a camera at `eye` aimed at `target`, in OpenCV axes (x right, y down, z fwd)."""
        eye = np.asarray(eye, dtype=float)
        target = np.asarray(target, dtype=float)
        forward = target - eye
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, np.asarray(up, dtype=float))
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)

        rot_world_cam = np.stack([right, down, forward], axis=1)  # columns are camera axes
        T_world_cam = np.eye(4)
        T_world_cam[:3, :3] = rot_world_cam
        T_world_cam[:3, 3] = eye
        return cls.from_fov(width, height, fov_deg, T_cam_world=invert_transform(T_world_cam))

    @property
    def T_world_cam(self) -> np.ndarray:
        return invert_transform(self.T_cam_world)

    def to_dict(self) -> dict:
        return {
            "width": int(self.width),
            "height": int(self.height),
            "intrinsics": self.intrinsics.tolist(),
            "distortion": self.distortion.tolist(),
            "T_cam_world": self.T_cam_world.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> CameraModel:
        return cls(
            width=int(data["width"]),
            height=int(data["height"]),
            intrinsics=np.asarray(data["intrinsics"], dtype=float),
            distortion=np.asarray(data.get("distortion", np.zeros(5)), dtype=float),
            T_cam_world=np.asarray(data.get("T_cam_world", np.eye(4)), dtype=float),
        )

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: Path) -> CameraModel:
        return cls.from_dict(json.loads(Path(path).read_text()))
