"""Safety pass: joint limits, velocity/acceleration limits, self-collision.

Policy, in order, per frame:
  1. clamp the candidate into the joint position limits;
  2. limit the step so neither the velocity nor the acceleration limit is broken;
  3. if the resulting pose self-collides, brake toward a stop, and if that still
     collides, hold the previous valid pose.

An unsafe sample is never emitted. Where the pipeline had to intervene it is
recorded in the `SafetyReport` rather than silently smoothed over.
"""

from __future__ import annotations

import mujoco
import numpy as np

from ..config import Embodiment, load_mujoco_model
from ..types import RobotTrajectory, SafetyReport


class SafetyChecker:
    def __init__(self, embodiment: Embodiment) -> None:
        self.embodiment = embodiment
        if embodiment.collision_margin > 0:
            # A margin needs a private model: the cached one is shared.
            self.model = mujoco.MjModel.from_xml_path(str(embodiment.model_path))
            self.model.geom_margin[:] = np.maximum(
                self.model.geom_margin, embodiment.collision_margin
            )
        else:
            self.model = load_mujoco_model(embodiment)
        self.data = mujoco.MjData(self.model)

        self.qpos_adr = np.array(
            [self._qpos_adr(name) for name in embodiment.arm_joints], dtype=int
        )
        gripper_adr = [self._qpos_adr(name) for name in embodiment.gripper.joints]

        self._home_qpos = self.model.qpos0.copy()
        # Park the fingers open: a gripper left closed rests in permanent
        # finger-on-finger contact, which is not a fault of the arm's pose.
        for adr in gripper_adr:
            self._home_qpos[adr] = embodiment.gripper.open_aperture

        # Contacts internal to the gripper are what a gripper is for, so they
        # are excluded from self-collision. Derived from the config's gripper
        # joints, so this holds for any embodiment without naming its parts.
        self._gripper_bodies = {
            int(self.model.jnt_bodyid[self._joint_id(name)]) for name in embodiment.gripper.joints
        }

    def _joint_id(self, name: str) -> int:
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"joint {name!r} not found in {self.embodiment.model_path}")
        return jid

    def _qpos_adr(self, name: str) -> int:
        return int(self.model.jnt_qposadr[self._joint_id(name)])

    def _is_gripper_internal(self, contact) -> bool:
        body1 = int(self.model.geom_bodyid[contact.geom1])
        body2 = int(self.model.geom_bodyid[contact.geom2])
        return body1 in self._gripper_bodies and body2 in self._gripper_bodies

    def in_self_collision(self, joints: np.ndarray) -> bool:
        self.data.qpos[:] = self._home_qpos
        self.data.qpos[self.qpos_adr] = joints
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        if self.data.ncon == 0:
            return False
        margin = self.embodiment.collision_margin
        return any(
            self.data.contact[i].dist < margin and not self._is_gripper_internal(self.data.contact[i])
            for i in range(self.data.ncon)
        )

    def within_joint_limits(self, joints: np.ndarray) -> bool:
        low, high = self.embodiment.joint_limits[:, 0], self.embodiment.joint_limits[:, 1]
        return bool(np.all(joints >= low - 1e-12) and np.all(joints <= high + 1e-12))

    def filter(self, candidates: np.ndarray, dt: float) -> tuple[np.ndarray, SafetyReport]:
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")

        candidates = np.asarray(candidates, dtype=float)
        n_frames = len(candidates)
        low, high = self.embodiment.joint_limits[:, 0], self.embodiment.joint_limits[:, 1]
        v_max = self.embodiment.velocity_limits
        a_max = self.embodiment.acceleration_limits

        out = np.zeros_like(candidates)
        report = SafetyReport(n_frames=n_frames)
        report.velocity_clamp_magnitude = [0.0] * n_frames

        previous = None
        velocity = np.zeros(candidates.shape[1])

        for i, raw in enumerate(candidates):
            clamped = np.clip(raw, low, high)
            if not np.allclose(clamped, raw, atol=1e-12):
                report.limit_clamped_frames.append(i)

            if previous is None:
                accepted = clamped
                if self.in_self_collision(accepted):
                    report.collision_frames.append(i)
                    report.held_frames.append(i)
                    accepted = np.clip(self.embodiment.home_joints, low, high)
                out[i] = accepted
                previous = accepted
                velocity = np.zeros_like(accepted)
                continue

            v_low = np.maximum(-v_max, velocity - a_max * dt)
            v_high = np.minimum(v_max, velocity + a_max * dt)
            desired = (clamped - previous) / dt
            limited = np.clip(desired, v_low, v_high)
            if not np.allclose(limited, desired, atol=1e-12):
                report.velocity_clamped_frames.append(i)
                report.velocity_clamp_magnitude[i] = float(
                    np.max(np.abs(limited - desired) / np.maximum(v_max, 1e-9))
                )

            accepted = np.clip(previous + limited * dt, low, high)

            if self.in_self_collision(accepted):
                report.collision_frames.append(i)
                braking = np.clip(np.zeros_like(velocity), v_low, v_high)
                braked = np.clip(previous + braking * dt, low, high)
                if self.in_self_collision(braked):
                    accepted = previous  # hold the last known-good pose
                else:
                    accepted = braked
                report.held_frames.append(i)

            out[i] = accepted
            velocity = (accepted - previous) / dt
            previous = accepted

        return out, report

    def filter_trajectory(
        self, trajectory: RobotTrajectory, kinematics=None
    ) -> tuple[RobotTrajectory, SafetyReport]:
        """Filter joints and recompute the end-effector poses that actually result."""
        from ..retarget.kinematics import Kinematics

        dt = 1.0 / trajectory.fps
        joints, report = self.filter(trajectory.joints, dt)

        kinematics = kinematics or Kinematics(self.embodiment)
        ee_positions = np.zeros((len(joints), 3))
        ee_quats = np.zeros((len(joints), 4))
        for i, q in enumerate(joints):
            ee_positions[i], ee_quats[i] = kinematics.fk(q)

        gripper = np.clip(
            trajectory.gripper,
            self.embodiment.gripper.limits[0],
            self.embodiment.gripper.limits[1],
        )

        filtered = RobotTrajectory(
            joints=joints,
            gripper=gripper,
            ee_positions=ee_positions,
            ee_quats=ee_quats,
            fps=trajectory.fps,
        )
        return filtered, report
