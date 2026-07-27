"""End-to-end: video in, LeRobot dataset out."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .camera import CameraModel
from .config import Embodiment, load_embodiment
from .export.lerobot import DEFAULT_TASK, export_lerobot
from .perception.base import HandFrontend
from .perception.marker import MarkerConfig, MarkerFrontend
from .retarget.retargeter import Retargeter
from .safety.checker import SafetyChecker
from .trajectory.smoothing import fill_gaps, smooth_trajectory
from .trajectory.world import observations_to_world
from .types import RobotTrajectory, SafetyReport, WristTrajectory
from .video import probe_video, read_frames


@dataclass
class PipelineResult:
    dataset_dir: Path
    robot_trajectory: RobotTrajectory
    wrist_trajectory: WristTrajectory
    safety_report: SafetyReport
    camera: CameraModel
    video_path: Path
    embodiment: Embodiment


def build_frontend(
    name: str, camera: CameraModel, marker_config: MarkerConfig | None = None, **kwargs
) -> HandFrontend:
    if name == "marker":
        return MarkerFrontend(camera, marker_config)
    if name == "mediapipe":
        from .perception.mediapipe_frontend import MediaPipeFrontend

        return MediaPipeFrontend(camera, **kwargs)
    raise ValueError(f"unknown frontend {name!r}; expected 'marker' or 'mediapipe'")


def run_pipeline(
    video: str | Path,
    embodiment: str | Path | Embodiment,
    out_dir: str | Path,
    frontend: str | HandFrontend = "marker",
    camera: CameraModel | None = None,
    fps: float | None = None,
    task: str = DEFAULT_TASK,
    smoothing: bool = True,
    marker_config: MarkerConfig | None = None,
    export: bool = True,
) -> PipelineResult:
    video = Path(video)
    if not video.exists():
        raise FileNotFoundError(f"input video not found: {video}")

    probe = probe_video(video)
    fps = float(fps if fps is not None else probe["fps"])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"fps must be positive and finite, got {fps}")

    embodiment = load_embodiment(embodiment)
    camera = camera or CameraModel.from_fov(probe["width"], probe["height"])

    estimator = (
        frontend
        if isinstance(frontend, HandFrontend)
        else build_frontend(frontend, camera, marker_config)
    )

    try:
        observations = [
            estimator.process(frame, index) for index, frame in enumerate(read_frames(video))
        ]
    finally:
        estimator.close()

    if not observations:
        raise ValueError(f"no frames decoded from {video}")

    wrist = observations_to_world(observations, camera, fps)
    wrist = smooth_trajectory(wrist) if smoothing else fill_gaps(wrist)

    retargeter = Retargeter(embodiment)
    candidate, retarget_report = retargeter.retarget(wrist)

    checker = SafetyChecker(embodiment)
    trajectory, report = checker.filter_trajectory(candidate, kinematics=retargeter.kinematics)
    report.observed_fraction = wrist.observed_fraction
    report.unreachable_frames = retarget_report.unreachable_frames

    out_dir = Path(out_dir)
    if export:
        export_lerobot(trajectory, embodiment, video, out_dir, task=task)

    return PipelineResult(
        dataset_dir=out_dir,
        robot_trajectory=trajectory,
        wrist_trajectory=wrist,
        safety_report=report,
        camera=camera,
        video_path=video,
        embodiment=embodiment,
    )
