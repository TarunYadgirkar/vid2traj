"""Thin MuJoCo wrapper: forward kinematics and analytic Jacobians."""

from __future__ import annotations

import mujoco
import numpy as np

from ..config import Embodiment, load_mujoco_model
from ..math3d import quat_from_matrix


class Kinematics:
    """Evaluates the embodiment's end-effector pose and Jacobian for a joint vector.

    Owns its own `MjData`, so instances are independent but cheap; the compiled
    `MjModel` is shared through the config-level cache.
    """

    def __init__(self, embodiment: Embodiment) -> None:
        self.embodiment = embodiment
        self.model = load_mujoco_model(embodiment)
        self.data = mujoco.MjData(self.model)

        self.site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, embodiment.ee_site)
        if self.site_id < 0:
            raise ValueError(f"site {embodiment.ee_site!r} not found in {embodiment.model_path}")

        self.qpos_adr = np.array(
            [self._joint_qpos_adr(name) for name in embodiment.arm_joints], dtype=int
        )
        self.dof_adr = np.array(
            [self._joint_dof_adr(name) for name in embodiment.arm_joints], dtype=int
        )
        self.gripper_qpos_adr = np.array(
            [self._joint_qpos_adr(name) for name in embodiment.gripper.joints], dtype=int
        )
        self._home_qpos = self.model.qpos0.copy()

    def _joint_id(self, name: str) -> int:
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"joint {name!r} not found in {self.embodiment.model_path}")
        return jid

    def _joint_qpos_adr(self, name: str) -> int:
        return int(self.model.jnt_qposadr[self._joint_id(name)])

    def _joint_dof_adr(self, name: str) -> int:
        return int(self.model.jnt_dofadr[self._joint_id(name)])

    def set_configuration(self, joints: np.ndarray, gripper: float | None = None) -> None:
        self.data.qpos[:] = self._home_qpos
        self.data.qpos[self.qpos_adr] = joints
        if gripper is not None and self.gripper_qpos_adr.size:
            self.data.qpos[self.gripper_qpos_adr] = gripper
        self.data.qvel[:] = 0.0
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

    def fk(self, joints: np.ndarray, gripper: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        self.set_configuration(joints, gripper)
        position = self.data.site_xpos[self.site_id].copy()
        rotation = self.data.site_xmat[self.site_id].reshape(3, 3).copy()
        return position, quat_from_matrix(rotation)

    def fk_matrix(self, joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.set_configuration(joints)
        return (
            self.data.site_xpos[self.site_id].copy(),
            self.data.site_xmat[self.site_id].reshape(3, 3).copy(),
        )

    def jacobian(self, joints: np.ndarray) -> np.ndarray:
        """(6, n_joints) site Jacobian: translation rows then rotation rows."""
        self.set_configuration(joints)
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site_id)
        return np.vstack([jacp[:, self.dof_adr], jacr[:, self.dof_adr]])
