"""Data carried between pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .math3d import canonical_quat


@dataclass(frozen=True)
class WristObservation:
    """One frame's worth of perceived wrist state, in the camera frame."""

    frame_index: int
    position: np.ndarray  # (3,) metres
    quat: np.ndarray  # (4,) w,x,y,z
    aperture: float = 1.0  # normalized hand opening, 0 closed .. 1 open
    track_id: int = 0
    confidence: float = 1.0


@dataclass
class WristTrajectory:
    """Per-frame wrist pose in the world frame, with an observability mask."""

    positions: np.ndarray  # (T, 3)
    quats: np.ndarray  # (T, 4)
    visible: np.ndarray  # (T,) bool — True where an actual detection backed the sample
    apertures: np.ndarray  # (T,)
    fps: float
    # True where the sample may be tracked: detected, or inside a short enough gap
    # that interpolation is honest. Long gaps stay False and are held downstream.
    usable: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.positions = np.asarray(self.positions, dtype=float).reshape(-1, 3)
        self.quats = canonical_quat(np.asarray(self.quats, dtype=float).reshape(-1, 4))
        self.visible = np.asarray(self.visible, dtype=bool).reshape(-1)
        self.apertures = np.asarray(self.apertures, dtype=float).reshape(-1)
        if self.usable is None:
            self.usable = self.visible.copy()
        else:
            self.usable = np.asarray(self.usable, dtype=bool).reshape(-1)
        lengths = {len(self.positions), len(self.quats), len(self.visible), len(self.apertures)}
        if len(lengths) != 1:
            raise ValueError(f"WristTrajectory field lengths disagree: {lengths}")

    def __len__(self) -> int:
        return len(self.positions)

    @property
    def missing_frames(self) -> list[int]:
        return np.flatnonzero(~self.visible).tolist()

    @property
    def observed_fraction(self) -> float:
        return float(self.visible.mean()) if len(self) else 0.0


@dataclass
class RobotTrajectory:
    """Joint-space trajectory for one embodiment, plus the EE poses it realizes."""

    joints: np.ndarray  # (T, n_arm_joints)
    gripper: np.ndarray  # (T,)
    ee_positions: np.ndarray  # (T, 3)
    ee_quats: np.ndarray  # (T, 4)
    fps: float

    def __post_init__(self) -> None:
        self.joints = np.asarray(self.joints, dtype=float)
        self.gripper = np.asarray(self.gripper, dtype=float).reshape(-1)
        self.ee_positions = np.asarray(self.ee_positions, dtype=float).reshape(-1, 3)
        self.ee_quats = canonical_quat(np.asarray(self.ee_quats, dtype=float).reshape(-1, 4))

    def __len__(self) -> int:
        return len(self.joints)

    @property
    def n_dof(self) -> int:
        return self.joints.shape[1] + 1  # arm joints plus the gripper command


@dataclass
class SafetyReport:
    """What the safety pass had to do. Kept alongside the data, not logged away."""

    n_frames: int
    held_frames: list[int] = field(default_factory=list)
    limit_clamped_frames: list[int] = field(default_factory=list)
    velocity_clamped_frames: list[int] = field(default_factory=list)
    collision_frames: list[int] = field(default_factory=list)
    unreachable_frames: list[int] = field(default_factory=list)
    observed_fraction: float = 1.0

    def summary(self) -> str:
        return (
            f"{self.n_frames} frames | observed {self.observed_fraction:.0%} | "
            f"held {len(self.held_frames)} | limit-clamped {len(self.limit_clamped_frames)} | "
            f"vel-clamped {len(self.velocity_clamped_frames)} | "
            f"collisions {len(self.collision_frames)} | unreachable {len(self.unreachable_frames)}"
        )
