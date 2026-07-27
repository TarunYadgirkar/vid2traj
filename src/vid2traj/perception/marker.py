"""ArUco fiducial frontend: metric, deterministic, no learned weights.

Marker frame convention (OpenCV's, see DECISIONS D7): corners are detected in
TL, TR, BR, BL order and pair with object points

    (-s/2, +s/2, 0), (+s/2, +s/2, 0), (+s/2, -s/2, 0), (-s/2, -s/2, 0)

so the marker frame is x-right, y-up, z out of the printed face.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..camera import CameraModel
from ..math3d import quat_from_matrix
from ..types import WristObservation
from .base import HandFrontend

DEFAULT_DICT = cv2.aruco.DICT_4X4_50

# The printed side length the frontend assumes when none is configured. Depth
# from a fiducial scales linearly with this number, so a mismatch between the
# assumed and the actual size is a pure range error: the synthetic renderer
# imports this same constant rather than keeping its own default.
DEFAULT_MARKER_SIZE = 0.08


def marker_object_points(size_m: float) -> np.ndarray:
    half = size_m / 2.0
    return np.array(
        [[-half, half, 0.0], [half, half, 0.0], [half, -half, 0.0], [-half, -half, 0.0]],
        dtype=np.float64,
    )


@dataclass
class MarkerConfig:
    marker_id: int = 0
    marker_size: float = DEFAULT_MARKER_SIZE  # metres, the printed side length
    dictionary: int = DEFAULT_DICT
    aperture_marker_id: int | None = None
    aperture_span: tuple[float, float] = (0.04, 0.12)


class MarkerFrontend(HandFrontend):
    """Track one specific marker id, ignoring every other subject in the frame."""

    name = "marker"

    def __init__(self, camera: CameraModel, config: MarkerConfig | None = None) -> None:
        self.camera = camera
        self.config = config or MarkerConfig()
        self._dictionary = cv2.aruco.getPredefinedDictionary(self.config.dictionary)
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self._detector = cv2.aruco.ArucoDetector(self._dictionary, params)
        self._object_points = marker_object_points(self.config.marker_size)

    def process(self, frame: np.ndarray, frame_index: int) -> WristObservation | None:
        corners, ids, _ = self._detector.detectMarkers(frame)
        if ids is None:
            return None

        found = {int(marker_id): corner.reshape(-1, 2) for marker_id, corner in zip(ids.ravel(), corners)}
        target = found.get(self.config.marker_id)
        if target is None:
            return None

        pose = self._solve(target)
        if pose is None:
            return None
        position, quat = pose

        return WristObservation(
            frame_index=frame_index,
            position=position,
            quat=quat,
            aperture=self._aperture(found, position),
            track_id=self.config.marker_id,
        )

    def _solve(self, image_points: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        ok, rvec, tvec = cv2.solvePnP(
            self._object_points,
            image_points.astype(np.float64),
            self.camera.intrinsics,
            self.camera.distortion,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not ok:
            return None
        rot, _ = cv2.Rodrigues(rvec)
        return tvec.reshape(3), quat_from_matrix(rot)

    def _aperture(self, found: dict[int, np.ndarray], position: np.ndarray) -> float:
        """Optional second marker whose separation encodes hand opening."""
        other_id = self.config.aperture_marker_id
        if other_id is None or other_id not in found:
            return 1.0
        other = self._solve(found[other_id])
        if other is None:
            return 1.0
        distance = float(np.linalg.norm(other[0] - position))
        low, high = self.config.aperture_span
        return float(np.clip((distance - low) / max(high - low, 1e-9), 0.0, 1.0))
