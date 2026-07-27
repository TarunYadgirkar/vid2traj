"""Graceful degradation: occlusion, hands leaving frame, two people.

The bar is not "produces a good trajectory" — with no observation there is no
good trajectory. The bar is: never crash, never emit an unsafe or discontinuous
sample, and report honestly which frames were not observed.
"""

from __future__ import annotations

import numpy as np
import pytest

from vid2traj import run_pipeline
from vid2traj.render.synth import make_demo_joint_trajectory, render_marker_clip
from vid2traj.safety import SafetyChecker

FPS = 30
N_FRAMES = 90


def _clip(panda, tmp_path, name, **kwargs):
    joints = make_demo_joint_trajectory(panda, n_frames=N_FRAMES, seed=0)
    return render_marker_clip(panda, joints, tmp_path / f"{name}.mp4", fps=FPS, **kwargs)


def _run(clip, panda, out_dir):
    return run_pipeline(
        video=clip.video_path,
        embodiment=panda,
        out_dir=out_dir,
        frontend="marker",
        camera=clip.camera,
        fps=clip.fps,
    )


def _assert_output_is_sane(result, panda):
    traj = result.robot_trajectory
    assert len(traj) == N_FRAMES, "every input frame must produce an output frame"
    assert np.all(np.isfinite(traj.joints)), "non-finite joints emitted"

    low, high = panda.joint_limits[:, 0], panda.joint_limits[:, 1]
    assert np.all(traj.joints >= low - 1e-9) and np.all(traj.joints <= high + 1e-9)

    checker = SafetyChecker(panda)
    assert not any(checker.in_self_collision(q) for q in traj.joints)

    dt = 1.0 / traj.fps
    vel = np.abs(np.diff(traj.joints, axis=0) / dt).max(axis=0)
    assert np.all(vel <= panda.velocity_limits + 1e-6), "degraded input produced a velocity spike"


def test_occluded_hand_midway(panda, tmp_path):
    """Marker hidden for 15 frames in the middle of the clip."""
    clip = _clip(panda, tmp_path, "occluded", occlude_frames=range(35, 50))
    result = _run(clip, panda, tmp_path / "ds")
    _assert_output_is_sane(result, panda)

    unobserved = set(result.wrist_trajectory.missing_frames)
    assert unobserved >= set(range(35, 50)), "occluded frames must be reported as unobserved"


def test_hand_leaves_frame(panda, tmp_path):
    """Marker translated out of the image for the last third of the clip."""
    clip = _clip(panda, tmp_path, "offscreen", offscreen_frames=range(60, N_FRAMES))
    result = _run(clip, panda, tmp_path / "ds")
    _assert_output_is_sane(result, panda)
    assert set(result.wrist_trajectory.missing_frames) >= set(range(60, N_FRAMES))


def test_two_people_in_frame(panda, tmp_path):
    """A second, distractor marker is present for the whole clip.

    The pipeline must lock onto the configured target and stay locked, rather
    than flip-flopping between subjects.
    """
    clip = _clip(panda, tmp_path, "two_people", distractor=True)
    result = _run(clip, panda, tmp_path / "ds")
    _assert_output_is_sane(result, panda)

    step = np.linalg.norm(np.diff(result.robot_trajectory.ee_positions, axis=0), axis=1)
    assert step.max() < 0.05, "tracker jumped between subjects"


def test_no_detection_at_all_does_not_crash(panda, tmp_path):
    """Every frame blank: the run must fail loudly or emit a safe held pose."""
    clip = _clip(panda, tmp_path, "blank", occlude_frames=range(0, N_FRAMES))
    result = _run(clip, panda, tmp_path / "ds")
    _assert_output_is_sane(result, panda)
    assert len(result.wrist_trajectory.missing_frames) == N_FRAMES
    np.testing.assert_allclose(
        result.robot_trajectory.joints,
        np.tile(panda.home_joints, (N_FRAMES, 1)),
        atol=1e-9,
        err_msg="with no observations the output should hold the home pose",
    )


def test_degraded_frames_are_flagged_in_the_export(panda, tmp_path):
    clip = _clip(panda, tmp_path, "flagged", occlude_frames=range(35, 50))
    result = _run(clip, panda, tmp_path / "ds")
    assert result.safety_report.n_frames == N_FRAMES
    assert result.safety_report.observed_fraction < 1.0


@pytest.mark.parametrize("bad_fps", [0, -1])
def test_invalid_fps_is_rejected(panda, tmp_path, synth_clip, bad_fps):
    with pytest.raises(ValueError):
        run_pipeline(
            video=synth_clip.video_path,
            embodiment=panda,
            out_dir=tmp_path / "ds",
            frontend="marker",
            camera=synth_clip.camera,
            fps=bad_fps,
        )
