"""Same input video + config -> byte-identical output."""

from __future__ import annotations

import hashlib

import numpy as np

from vid2traj import run_pipeline


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _files(root, pattern):
    return sorted(p for p in root.rglob(pattern) if p.is_file())


def _run(synth_clip, panda, out_dir):
    return run_pipeline(
        video=synth_clip.video_path,
        embodiment=panda,
        out_dir=out_dir,
        frontend="marker",
        camera=synth_clip.camera,
        fps=synth_clip.fps,
        task="synthetic reach",
    )


def test_parquet_is_byte_identical_across_runs(synth_clip, panda, tmp_path):
    a = _run(synth_clip, panda, tmp_path / "a").dataset_dir
    b = _run(synth_clip, panda, tmp_path / "b").dataset_dir

    files_a = [p.relative_to(a) for p in _files(a, "*.parquet")]
    files_b = [p.relative_to(b) for p in _files(b, "*.parquet")]
    assert files_a == files_b and files_a, "parquet file sets differ or are empty"

    for rel in files_a:
        assert _digest(a / rel) == _digest(b / rel), f"parquet {rel} differs between runs"


def test_video_is_byte_identical_across_runs(synth_clip, panda, tmp_path):
    a = _run(synth_clip, panda, tmp_path / "a").dataset_dir
    b = _run(synth_clip, panda, tmp_path / "b").dataset_dir

    vids_a = [p.relative_to(a) for p in _files(a, "*.mp4")]
    vids_b = [p.relative_to(b) for p in _files(b, "*.mp4")]
    assert vids_a == vids_b and vids_a, "video file sets differ or are empty"

    for rel in vids_a:
        assert _digest(a / rel) == _digest(b / rel), f"video {rel} differs between runs"


def test_numeric_output_is_exactly_reproducible(synth_clip, panda, tmp_path):
    a = _run(synth_clip, panda, tmp_path / "a").robot_trajectory
    b = _run(synth_clip, panda, tmp_path / "b").robot_trajectory
    np.testing.assert_array_equal(a.joints, b.joints)
    np.testing.assert_array_equal(a.gripper, b.gripper)


def test_synthetic_renderer_is_deterministic(panda, tmp_path):
    """The fixture generator itself must be reproducible, or nothing else can be."""
    from vid2traj.render.synth import make_demo_joint_trajectory, render_marker_clip

    joints = make_demo_joint_trajectory(panda, n_frames=30, seed=0)
    again = make_demo_joint_trajectory(panda, n_frames=30, seed=0)
    np.testing.assert_array_equal(joints, again)

    one = render_marker_clip(panda, joints, tmp_path / "one.mp4", fps=30)
    two = render_marker_clip(panda, joints, tmp_path / "two.mp4", fps=30)
    assert _digest(one.video_path) == _digest(two.video_path)
