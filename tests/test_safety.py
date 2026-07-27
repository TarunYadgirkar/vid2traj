"""Every emitted frame must be physically admissible under MuJoCo."""

from __future__ import annotations

import numpy as np
import pytest

from vid2traj import run_pipeline
from vid2traj.safety import SafetyChecker


@pytest.fixture(scope="module")
def emitted(synth_clip, panda, tmp_path_factory):
    return run_pipeline(
        video=synth_clip.video_path,
        embodiment=panda,
        out_dir=tmp_path_factory.mktemp("safe") / "ds",
        frontend="marker",
        camera=synth_clip.camera,
        fps=synth_clip.fps,
    )


def test_every_frame_within_joint_limits(emitted, panda):
    joints = emitted.robot_trajectory.joints
    low, high = panda.joint_limits[:, 0], panda.joint_limits[:, 1]
    assert np.all(joints >= low - 1e-9), "joint below lower limit"
    assert np.all(joints <= high + 1e-9), "joint above upper limit"


def test_every_frame_is_self_collision_free(emitted, panda):
    checker = SafetyChecker(panda)
    colliding = [
        i for i, q in enumerate(emitted.robot_trajectory.joints) if checker.in_self_collision(q)
    ]
    assert colliding == [], f"self-collision on frames {colliding[:10]}"


def test_gripper_command_within_range(emitted, panda):
    grip = emitted.robot_trajectory.gripper
    assert np.all(grip >= panda.gripper.limits[0] - 1e-9)
    assert np.all(grip <= panda.gripper.limits[1] + 1e-9)


def test_checker_actually_rejects_a_known_bad_pose(panda):
    """A safety check that never fires is not a safety check.

    Drive the arm into a pose that folds the wrist back into the base and
    assert the checker flags it, so the passing results above mean something.
    """
    checker = SafetyChecker(panda)
    home = panda.home_joints.copy()
    assert not checker.in_self_collision(home), "home pose must be collision free"

    folded = home.copy()
    folded[1] = panda.joint_limits[1, 1]  # shoulder fully up
    folded[3] = panda.joint_limits[3, 1]  # elbow fully folded
    folded[5] = panda.joint_limits[5, 0]
    assert checker.in_self_collision(folded), "folded pose should self-collide"


def _collision_ramp(panda, n_frames=200):
    """Candidates that walk from home steadily into a self-colliding pose.

    A single-frame jump to a distant bad pose is not a useful probe: the
    velocity limiter correctly refuses to execute it in one step, so the arm
    never actually reaches the collision. Ramping in is what a real bad
    retargeting looks like, and it forces the collision path to be taken.
    """
    home = panda.home_joints.copy()
    bad = home.copy()
    bad[1] = panda.joint_limits[1, 1]
    bad[3] = panda.joint_limits[3, 1]
    bad[5] = panda.joint_limits[5, 0]
    alpha = np.linspace(0.0, 1.0, n_frames)[:, None]
    return home[None, :] * (1 - alpha) + bad[None, :] * alpha


def test_violations_never_emit_an_unsafe_pose(panda):
    checker = SafetyChecker(panda)
    candidates = _collision_ramp(panda)
    assert any(checker.in_self_collision(q) for q in candidates), "ramp must reach a collision"

    out, report = checker.filter(candidates, dt=1.0 / 30.0)

    assert out.shape == candidates.shape
    emitted_bad = [i for i, q in enumerate(out) if checker.in_self_collision(q)]
    assert emitted_bad == [], f"unsafe poses emitted on frames {emitted_bad[:10]}"
    assert report.held_frames, "report must record which frames were held"


def test_violations_hold_the_previous_valid_pose(panda):
    """Once blocked, the output must stop advancing rather than push through."""
    checker = SafetyChecker(panda)
    out, report = checker.filter(_collision_ramp(panda), dt=1.0 / 30.0)

    first_hold = report.held_frames[0]
    steps = np.linalg.norm(np.diff(out[first_hold:], axis=0), axis=1)
    assert steps.max() <= 1e-9, "trajectory kept moving after the safety pass blocked it"

    # It comes to rest at the last admissible pose, within one braking step of it.
    assert np.linalg.norm(out[-1] - out[first_hold - 1]) < 0.05
