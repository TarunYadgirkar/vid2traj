"""MediaPipe Hands frontend for un-instrumented real video.

Monocular RGB gives no absolute scale, so depth is recovered by comparing the
apparent size of the hand against a nominal physical hand span (config). This
is an approximation and is documented as such in SPEC section 6 — the marker
frontend is the metric one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..camera import CameraModel
from ..math3d import quat_from_matrix
from ..types import WristObservation
from .base import HandFrontend

WRIST = 0
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_TIP = 8
PINKY_MCP = 17


@dataclass
class MediaPipeConfig:
    handedness: str = "any"  # "Left", "Right", or "any"
    hand_span_m: float = 0.09  # wrist-to-index-MCP distance of a nominal adult hand
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    max_num_hands: int = 2
    aperture_span: tuple[float, float] = (0.02, 0.10)


class MediaPipeFrontend(HandFrontend):
    name = "mediapipe"

    def __init__(self, camera: CameraModel, config: MediaPipeConfig | None = None) -> None:
        try:
            import mediapipe as mp
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "the mediapipe frontend needs the optional dependency: "
                "pip install 'vid2traj[mediapipe]'"
            ) from exc

        self.camera = camera
        self.config = config or MediaPipeConfig()
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=self.config.max_num_hands,
            min_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
        )
        self._last_position: np.ndarray | None = None

    def process(self, frame: np.ndarray, frame_index: int) -> WristObservation | None:
        import cv2

        result = self._hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not result.multi_hand_landmarks:
            return None

        candidates = list(
            zip(result.multi_hand_landmarks, result.multi_handedness or [None] * len(result.multi_hand_landmarks))
        )
        chosen = self._select(candidates, frame.shape)
        if chosen is None:
            return None

        landmarks, confidence = chosen
        points = self._to_pixels(landmarks, frame.shape)
        position = self._estimate_position(points)
        quat = self._estimate_orientation(points, position)
        self._last_position = position

        return WristObservation(
            frame_index=frame_index,
            position=position,
            quat=quat,
            aperture=self._aperture(points, position),
            track_id=0,
            confidence=confidence,
        )

    def _select(self, candidates, shape):
        """Pick one subject and stay with it, so two people cannot cause flip-flop."""
        wanted = self.config.handedness.lower()
        scored = []
        for landmarks, handedness in candidates:
            label = handedness.classification[0].label if handedness else "Unknown"
            score = handedness.classification[0].score if handedness else 0.5
            if wanted != "any" and label.lower() != wanted:
                continue
            points = self._to_pixels(landmarks, shape)
            if self._last_position is not None:
                position = self._estimate_position(points)
                distance = float(np.linalg.norm(position - self._last_position))
            else:
                distance = -self._span_px(points)  # first frame: prefer the nearest/largest hand
            scored.append((distance, float(score), landmarks))
        if not scored:
            return None
        scored.sort(key=lambda item: (item[0], -item[1]))
        distance, score, landmarks = scored[0]
        return landmarks, score

    def _to_pixels(self, landmarks, shape) -> np.ndarray:
        height, width = shape[:2]
        return np.array([[lm.x * width, lm.y * height, lm.z] for lm in landmarks.landmark])

    def _span_px(self, points: np.ndarray) -> float:
        return float(np.linalg.norm(points[INDEX_MCP, :2] - points[WRIST, :2]))

    def _estimate_position(self, points: np.ndarray) -> np.ndarray:
        """Back-project the wrist pixel to the depth implied by apparent hand size."""
        focal = float(self.camera.intrinsics[0, 0])
        span_px = max(self._span_px(points), 1e-6)
        depth = focal * self.config.hand_span_m / span_px
        pixel = np.array([points[WRIST, 0], points[WRIST, 1], 1.0])
        ray = np.linalg.inv(self.camera.intrinsics) @ pixel
        return ray / ray[2] * depth

    def _estimate_orientation(self, points: np.ndarray, position: np.ndarray) -> np.ndarray:
        """Build a frame from the palm triangle: wrist -> index MCP -> pinky MCP."""
        scale = np.linalg.norm(position) / max(self.camera.intrinsics[0, 0], 1e-9)
        wrist = np.array([*points[WRIST, :2], 0.0]) * scale
        index = np.array([*points[INDEX_MCP, :2], 0.0]) * scale
        pinky = np.array([*points[PINKY_MCP, :2], 0.0]) * scale

        x_axis = index - wrist
        if np.linalg.norm(x_axis) < 1e-9:
            return np.array([1.0, 0.0, 0.0, 0.0])
        x_axis /= np.linalg.norm(x_axis)

        across = pinky - wrist
        z_axis = np.cross(x_axis, across)
        if np.linalg.norm(z_axis) < 1e-9:
            return np.array([1.0, 0.0, 0.0, 0.0])
        z_axis /= np.linalg.norm(z_axis)
        y_axis = np.cross(z_axis, x_axis)
        return quat_from_matrix(np.stack([x_axis, y_axis, z_axis], axis=1))

    def _aperture(self, points: np.ndarray, position: np.ndarray) -> float:
        focal = float(self.camera.intrinsics[0, 0])
        depth = float(position[2])
        pinch_px = float(np.linalg.norm(points[THUMB_TIP, :2] - points[INDEX_TIP, :2]))
        pinch_m = pinch_px * depth / max(focal, 1e-9)
        low, high = self.config.aperture_span
        return float(np.clip((pinch_m - low) / max(high - low, 1e-9), 0.0, 1.0))

    def close(self) -> None:
        if getattr(self, "_hands", None) is not None:
            self._hands.close()
            self._hands = None
