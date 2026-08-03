"""The analytic Jacobian and the rigid-transform helpers, checked against first principles.

Frame-convention and DOF-indexing mistakes do not fail loudly: they produce an IK
loop that still converges, just to the wrong pose. So the Jacobian is differenced
numerically here rather than trusted, and every quaternion helper is pinned to an
algebraic identity it must satisfy for any input.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation, Slerp

from vid2traj.camera import CameraModel
from vid2traj.math3d import (
    canonical_quat,
    invert_transform,
    make_transform,
    matrix_from_quat,
    orientation_error,
    quat_from_matrix,
    quat_from_scipy,
    quat_slerp_fill,
    quat_to_scipy,
    split_transform,
    unwrap_quaternions,
)
from vid2traj.retarget.kinematics import Kinematics

N_SAMPLES = 25
FD_STEP = 1e-5
FD_TOL = 1e-6


def random_quats(rng: np.random.Generator, n: int) -> np.ndarray:
    return canonical_quat(rng.normal(size=(n, 4)))


def interior_joint_samples(embodiment, rng: np.random.Generator, n: int) -> np.ndarray:
    """Draw strictly inside the limits; the margin keeps ±h differencing legal too."""
    low, high = embodiment.joint_limits[:, 0], embodiment.joint_limits[:, 1]
    span = high - low
    return rng.uniform(low + 0.05 * span, high - 0.05 * span, size=(n, embodiment.n_joints))


@pytest.fixture(params=["panda", "so101"])
def embodiment(request):
    return request.getfixturevalue(request.param)


def test_translation_jacobian_matches_central_differences(embodiment):
    kin = Kinematics(embodiment)
    rng = np.random.default_rng(0)

    worst = 0.0
    for q in interior_joint_samples(embodiment, rng, N_SAMPLES):
        analytic = kin.jacobian(q)[:3]
        for k in range(embodiment.n_joints):
            step = np.zeros(embodiment.n_joints)
            step[k] = FD_STEP
            forward, _ = kin.fk_matrix(q + step)
            backward, _ = kin.fk_matrix(q - step)
            numeric = (forward - backward) / (2.0 * FD_STEP)
            worst = max(worst, float(np.abs(analytic[:, k] - numeric).max()))

    assert worst < FD_TOL, f"translation Jacobian disagrees with finite differences by {worst:.3e}"


def test_rotation_jacobian_matches_central_differences(embodiment):
    """The rotation rows are the world-frame angular velocity, not a d/dq of the quaternion."""
    kin = Kinematics(embodiment)
    rng = np.random.default_rng(1)

    worst = 0.0
    for q in interior_joint_samples(embodiment, rng, N_SAMPLES):
        analytic = kin.jacobian(q)[3:]
        for k in range(embodiment.n_joints):
            step = np.zeros(embodiment.n_joints)
            step[k] = FD_STEP
            _, forward = kin.fk_matrix(q + step)
            _, backward = kin.fk_matrix(q - step)
            numeric = orientation_error(backward, forward) / (2.0 * FD_STEP)
            worst = max(worst, float(np.abs(analytic[:, k] - numeric).max()))

    assert worst < FD_TOL, f"rotation Jacobian disagrees with finite differences by {worst:.3e}"


def test_jacobian_is_the_first_order_map_for_a_generic_step(embodiment):
    """Column-wise agreement is necessary but not sufficient: check the whole 6xN map at once."""
    kin = Kinematics(embodiment)
    rng = np.random.default_rng(2)
    q = interior_joint_samples(embodiment, rng, 1)[0]

    jac = kin.jacobian(q)
    assert jac.shape == (6, embodiment.n_joints)

    dq = rng.normal(scale=1e-5, size=embodiment.n_joints)
    forward_pos, forward_rot = kin.fk_matrix(q + dq)
    backward_pos, backward_rot = kin.fk_matrix(q - dq)
    twist = np.concatenate(
        [(forward_pos - backward_pos) / 2.0, orientation_error(backward_rot, forward_rot) / 2.0]
    )
    np.testing.assert_allclose(jac @ dq, twist, atol=FD_TOL)


def test_invert_transform_is_a_true_inverse():
    rng = np.random.default_rng(3)
    for quat in random_quats(rng, N_SAMPLES):
        transform = make_transform(rng.normal(scale=2.0, size=3), quat)
        np.testing.assert_allclose(invert_transform(transform) @ transform, np.eye(4), atol=1e-12)
        np.testing.assert_allclose(transform @ invert_transform(transform), np.eye(4), atol=1e-12)


def test_make_and_split_transform_round_trip():
    rng = np.random.default_rng(4)
    for quat in random_quats(rng, N_SAMPLES):
        position = rng.normal(scale=2.0, size=3)
        recovered_position, recovered_quat = split_transform(make_transform(position, quat))
        np.testing.assert_allclose(recovered_position, position, atol=1e-12)
        np.testing.assert_allclose(recovered_quat, quat, atol=1e-12)


def test_quaternion_and_matrix_round_trip():
    rng = np.random.default_rng(5)
    for quat in random_quats(rng, N_SAMPLES):
        np.testing.assert_allclose(quat_from_matrix(matrix_from_quat(quat)), quat, atol=1e-12)
        assert np.linalg.det(matrix_from_quat(quat)) == pytest.approx(1.0, abs=1e-12)


def test_scipy_quaternion_order_conversion_is_an_involution():
    rng = np.random.default_rng(6)
    quats = random_quats(rng, N_SAMPLES)
    np.testing.assert_allclose(quat_from_scipy(quat_to_scipy(quats)), quats, atol=1e-15)
    np.testing.assert_allclose(quat_to_scipy(quats)[:, 3], quats[:, 0], atol=1e-15)


def test_canonical_quat_is_unit_norm_positive_w_and_idempotent():
    rng = np.random.default_rng(7)
    raw = rng.normal(scale=3.0, size=(N_SAMPLES, 4))
    canonical = canonical_quat(raw)

    np.testing.assert_allclose(np.linalg.norm(canonical, axis=-1), 1.0, atol=1e-12)
    assert np.all(canonical[:, 0] >= 0.0)
    np.testing.assert_allclose(canonical_quat(canonical), canonical, atol=1e-15)
    np.testing.assert_allclose(canonical_quat(-raw), canonical, atol=1e-12)

    np.testing.assert_allclose(
        matrix_from_quat(canonical), matrix_from_quat(canonical_quat(raw)), atol=1e-12
    )


def test_unwrap_quaternions_keeps_the_rotation_and_fixes_the_hemisphere():
    rng = np.random.default_rng(8)
    quats = random_quats(rng, N_SAMPLES)
    flipped = quats * np.where(rng.random((N_SAMPLES, 1)) < 0.5, -1.0, 1.0)

    unwrapped = unwrap_quaternions(flipped)
    dots = np.sum(unwrapped[1:] * unwrapped[:-1], axis=-1)
    assert np.all(dots >= 0.0)

    for original, out in zip(quats, unwrapped):
        np.testing.assert_allclose(matrix_from_quat(out), matrix_from_quat(original), atol=1e-12)


def test_quat_slerp_fill_matches_scipy_across_an_interior_gap():
    rng = np.random.default_rng(9)
    quats = random_quats(rng, 12)
    valid = np.ones(12, dtype=bool)
    valid[4:8] = False

    filled = quat_slerp_fill(quats.copy(), valid)
    assert np.allclose(np.linalg.norm(filled, axis=-1), 1.0)

    idx = np.flatnonzero(valid)
    expected = Slerp(idx.astype(float), Rotation.from_quat(quat_to_scipy(quats[idx])))
    gap = np.flatnonzero(~valid)
    np.testing.assert_allclose(
        filled[gap],
        canonical_quat(quat_from_scipy(expected(gap.astype(float)).as_quat())),
        atol=1e-12,
    )
    np.testing.assert_allclose(filled[valid], canonical_quat(quats[valid]), atol=1e-12)


def test_quat_slerp_fill_holds_leading_and_trailing_gaps():
    rng = np.random.default_rng(10)
    quats = random_quats(rng, 10)
    valid = np.zeros(10, dtype=bool)
    valid[3:7] = True

    filled = quat_slerp_fill(quats.copy(), valid)
    expected_first = canonical_quat(quats[3])
    expected_last = canonical_quat(quats[6])
    for i in range(3):
        np.testing.assert_allclose(filled[i], expected_first, atol=1e-12)
    for i in range(7, 10):
        np.testing.assert_allclose(filled[i], expected_last, atol=1e-12)


def test_orientation_error_rotates_current_onto_target():
    rng = np.random.default_rng(11)
    for quat_a, quat_b in zip(random_quats(rng, N_SAMPLES), random_quats(rng, N_SAMPLES)):
        current = matrix_from_quat(quat_a)
        target = matrix_from_quat(quat_b)
        rotvec = orientation_error(current, target)
        np.testing.assert_allclose(
            Rotation.from_rotvec(rotvec).as_matrix() @ current, target, atol=1e-12
        )
        assert np.linalg.norm(rotvec) <= np.pi + 1e-9


def test_looking_at_puts_the_target_on_the_optical_axis():
    rng = np.random.default_rng(12)
    for _ in range(N_SAMPLES):
        eye = rng.normal(scale=1.5, size=3)
        target = rng.normal(scale=1.5, size=3)
        if np.linalg.norm(target - eye) < 0.2:
            continue

        camera = CameraModel.looking_at(eye, target)
        in_camera = camera.T_cam_world @ np.append(target, 1.0)
        np.testing.assert_allclose(in_camera[:2], 0.0, atol=1e-9)
        assert in_camera[2] > 0.0
        np.testing.assert_allclose(in_camera[2], np.linalg.norm(target - eye), atol=1e-9)

        np.testing.assert_allclose(
            camera.T_world_cam @ np.array([0.0, 0.0, 0.0, 1.0]), np.append(eye, 1.0), atol=1e-9
        )
