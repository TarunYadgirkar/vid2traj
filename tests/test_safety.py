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


def test_violations_hold_the_previous_valid_pose(panda):
    """On rejection the checker must repeat the last good sample, not emit junk."""
    checker = SafetyChecker(panda)
    home = panda.home_joints.copy()

    good = np.stack([home, home])
    bad = home.copy()
    bad[1] = panda.joint_limits[1, 1]
    bad[3] = panda.joint_limits[3, 1]
    bad[5] = panda.joint_limits[5, 0]

    candidates = np.stack([home, home, bad, home])
    out, report = checker.filter(candidates, dt=1.0 / 30.0)

    assert out.shape == candidates.shape
    np.testing.assert_allclose(out[2], out[1], atol=1e-12), "rejected frame must hold previous pose"
    assert report.held_frames, "report must record which frames were held"
    assert 2 in report.held_frames
    del good
