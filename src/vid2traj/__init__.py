"""vid2traj — RGB video of a human manipulation task -> robot trajectory dataset."""

from .camera import CameraModel
from .config import Embodiment, GripperSpec, IKParams, list_embodiments, load_embodiment
from .perception.base import HandFrontend
from .perception.marker import MarkerConfig, MarkerFrontend
from .pipeline import PipelineResult, build_frontend, run_pipeline
from .retarget.retargeter import Retargeter
from .safety.checker import SafetyChecker
from .trajectory.smoothing import fill_gaps, smooth_trajectory
from .trajectory.world import observations_to_world
from .types import RobotTrajectory, SafetyReport, WristObservation, WristTrajectory

__version__ = "0.1.0"

__all__ = [
    "CameraModel",
    "Embodiment",
    "GripperSpec",
    "HandFrontend",
    "IKParams",
    "MarkerConfig",
    "MarkerFrontend",
    "PipelineResult",
    "Retargeter",
    "RobotTrajectory",
    "SafetyChecker",
    "SafetyReport",
    "WristObservation",
    "WristTrajectory",
    "build_frontend",
    "fill_gaps",
    "list_embodiments",
    "load_embodiment",
    "observations_to_world",
    "run_pipeline",
    "smooth_trajectory",
]
