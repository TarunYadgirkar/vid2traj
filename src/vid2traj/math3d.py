"""Small rigid-transform helpers.

Quaternions are (w, x, y, z) everywhere in vid2traj. SciPy uses (x, y, z, w),
so every crossing of that boundary goes through the helpers here rather than
being open-coded at call sites.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def quat_to_scipy(q_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(q_wxyz, dtype=float)
    return np.roll(q, -1, axis=-1)


def quat_from_scipy(q_xyzw: np.ndarray) -> np.ndarray:
    q = np.asarray(q_xyzw, dtype=float)
    return np.roll(q, 1, axis=-1)


def rotation_from_quat(q_wxyz: np.ndarray) -> Rotation:
    return Rotation.from_quat(quat_to_scipy(q_wxyz))


def quat_from_rotation(rot: Rotation) -> np.ndarray:
    return canonical_quat(quat_from_scipy(rot.as_quat()))


def matrix_from_quat(q_wxyz: np.ndarray) -> np.ndarray:
    return rotation_from_quat(q_wxyz).as_matrix()


def quat_from_matrix(mat: np.ndarray) -> np.ndarray:
    return quat_from_rotation(Rotation.from_matrix(np.asarray(mat, dtype=float)))


def canonical_quat(q_wxyz: np.ndarray) -> np.ndarray:
    """Normalize and fix the sign so q and -q have one representation.

    Without this, IK targets and exported orientations can flip sign between
    frames and read as a 360-degree discontinuity downstream.
    """
    q = np.asarray(q_wxyz, dtype=float)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    sign = np.where(q[..., 0:1] < 0, -1.0, 1.0)
    return q * sign


def make_transform(position: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    mat = np.eye(4)
    mat[:3, :3] = matrix_from_quat(quat_wxyz)
    mat[:3, 3] = np.asarray(position, dtype=float)
    return mat


def split_transform(transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return transform[:3, 3].copy(), quat_from_matrix(transform[:3, :3])


def invert_transform(transform: np.ndarray) -> np.ndarray:
    rot = transform[:3, :3]
    out = np.eye(4)
    out[:3, :3] = rot.T
    out[:3, 3] = -rot.T @ transform[:3, 3]
    return out


def quat_slerp_fill(quats: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Fill invalid orientations by SLERP between the nearest valid neighbours.

    Leading and trailing gaps are held at the nearest valid sample, since there
    is nothing to interpolate towards.
    """
    quats = np.asarray(quats, dtype=float).copy()
    valid = np.asarray(valid, dtype=bool)
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        return quats
    if idx.size == 1:
        quats[:] = quats[idx[0]]
        return quats

    rot = Rotation.from_quat(quat_to_scipy(quats[idx]))
    slerp = Slerp(idx.astype(float), rot)
    targets = np.flatnonzero(~valid)
    inside = targets[(targets > idx[0]) & (targets < idx[-1])]
    if inside.size:
        quats[inside] = quat_from_scipy(slerp(inside.astype(float)).as_quat())
    quats[targets[targets < idx[0]]] = quats[idx[0]]
    quats[targets[targets > idx[-1]]] = quats[idx[-1]]
    return canonical_quat(quats)


def unwrap_quaternions(quats: np.ndarray) -> np.ndarray:
    """Flip signs so consecutive quaternions lie on the same hemisphere."""
    out = np.asarray(quats, dtype=float).copy()
    for i in range(1, len(out)):
        if np.dot(out[i], out[i - 1]) < 0:
            out[i] = -out[i]
    return out


def orientation_error(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Rotation vector taking `current` onto `target`, both 3x3 matrices."""
    return Rotation.from_matrix(target @ current.T).as_rotvec()
