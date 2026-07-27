"""Map a human wrist trajectory onto a robot embodiment's joint space."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import Embodiment
from ..math3d import make_transform, split_transform
from ..types import RobotTrajectory, WristTrajectory
from .ik import solve_ik
from .kinematics import Kinematics


@dataclass
class RetargetReport:
    unreachable_frames: list[int]
    frozen_frames: list[int]
    max_position_error: float


class Retargeter:
    """Wrist pose -> end-effector target -> joint vector, one frame at a time.

    Nothing here knows which robot it is driving; everything embodiment-specific
    arrives through `Embodiment` (see SPEC section 4).
    """

    def __init__(self, embodiment: Embodiment) -> None:
        self.embodiment = embodiment
        self.kinematics = Kinematics(embodiment)

    def ee_targets(self, trajectory: WristTrajectory) -> tuple[np.ndarray, np.ndarray]:
        """Apply the fixed wrist->end-effector offset from the embodiment config."""
        positions = np.zeros_like(trajectory.positions)
        quats = np.zeros_like(trajectory.quats)
        for i in range(len(trajectory)):
            world_wrist = make_transform(trajectory.positions[i], trajectory.quats[i])
            positions[i], quats[i] = split_transform(world_wrist @ self.embodiment.wrist_to_ee)
        return positions, quats

    def retarget(self, trajectory: WristTrajectory) -> tuple[RobotTrajectory, RetargetReport]:
        target_positions, target_quats = self.ee_targets(trajectory)
        n_frames = len(trajectory)

        joints = np.zeros((n_frames, self.embodiment.n_joints))
        seed = self.embodiment.home_joints.copy()
        unreachable: list[int] = []
        frozen: list[int] = []
        worst_error = 0.0

        for i in range(n_frames):
            if not trajectory.usable[i]:
                # No trustworthy observation: freeze rather than chase noise.
                joints[i] = seed
                frozen.append(i)
                continue

            result = solve_ik(
                self.kinematics,
                self.embodiment,
                target_positions[i],
                target_quats[i],
                seed=seed,
            )
            joints[i] = result.joints
            seed = result.joints
            worst_error = max(worst_error, result.position_error)
            if not result.converged:
                unreachable.append(i)

        gripper = self.embodiment.gripper.from_hand_aperture(trajectory.apertures)

        ee_positions = np.zeros((n_frames, 3))
        ee_quats = np.zeros((n_frames, 4))
        for i in range(n_frames):
            ee_positions[i], ee_quats[i] = self.kinematics.fk(joints[i])

        robot_trajectory = RobotTrajectory(
            joints=joints,
            gripper=gripper,
            ee_positions=ee_positions,
            ee_quats=ee_quats,
            fps=trajectory.fps,
        )
        report = RetargetReport(
            unreachable_frames=unreachable,
            frozen_frames=frozen,
            max_position_error=worst_error,
        )
        return robot_trajectory, report
