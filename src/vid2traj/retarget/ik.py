"""Damped least-squares inverse kinematics on MuJoCo's analytic Jacobian.

Deliberately dependency-free (see DECISIONS D3). Redundant DOF are resolved by
a null-space pull toward the embodiment's home pose, which keeps the arm in a
consistent, human-looking branch instead of drifting between elbow solutions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import Embodiment
from ..math3d import matrix_from_quat, orientation_error
from .kinematics import Kinematics


@dataclass
class IKResult:
    joints: np.ndarray
    position_error: float
    orientation_error: float
    converged: bool


def solve_ik(
    kinematics: Kinematics,
    embodiment: Embodiment,
    target_position: np.ndarray,
    target_quat: np.ndarray,
    seed: np.ndarray,
) -> IKResult:
    params = embodiment.ik
    limits = embodiment.joint_limits
    joints = np.clip(np.asarray(seed, dtype=float).copy(), limits[:, 0], limits[:, 1])
    target_rotation = matrix_from_quat(target_quat)

    position_error = np.inf
    rotation_error = np.inf
    converged = False

    for _ in range(params.max_iters):
        position, rotation = kinematics.fk_matrix(joints)
        err_position = target_position - position
        err_rotation = orientation_error(rotation, target_rotation)

        position_error = float(np.linalg.norm(err_position))
        rotation_error = float(np.linalg.norm(err_rotation))
        if position_error < params.tol_position and (
            embodiment.position_only or rotation_error < params.tol_orientation
        ):
            converged = True
            break

        jac = kinematics.jacobian(joints)
        if embodiment.position_only:
            jac = jac[:3]
            error = err_position
        else:
            jac = np.vstack([jac[:3], jac[3:] * embodiment.orientation_weight])
            error = np.concatenate([err_position, err_rotation * embodiment.orientation_weight])

        lam_sq = params.damping**2
        jjt = jac @ jac.T + lam_sq * np.eye(jac.shape[0])
        delta = jac.T @ np.linalg.solve(jjt, error)

        if params.null_space_gain and jac.shape[1] > jac.shape[0]:
            pinv = jac.T @ np.linalg.solve(jjt, np.eye(jac.shape[0]))
            null_projector = np.eye(jac.shape[1]) - pinv @ jac
            delta += null_projector @ (params.null_space_gain * (embodiment.home_joints - joints))

        norm = np.linalg.norm(delta)
        if norm > params.step_limit:
            delta *= params.step_limit / norm

        joints = np.clip(joints + delta, limits[:, 0], limits[:, 1])

    else:
        position, rotation = kinematics.fk_matrix(joints)
        position_error = float(np.linalg.norm(target_position - position))
        rotation_error = float(np.linalg.norm(orientation_error(rotation, target_rotation)))
        converged = position_error < params.tol_position and (
            embodiment.position_only or rotation_error < params.tol_orientation
        )

    return IKResult(
        joints=joints,
        position_error=position_error,
        orientation_error=rotation_error,
        converged=converged,
    )
