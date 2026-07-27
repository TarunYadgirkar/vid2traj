"""Core regression: known joint trajectory -> render -> full pipeline -> recover.

This is the test that says "the pipeline works". Everything else guards a
property; this one guards the actual claim.

Tolerances are stated as absolutes rather than derived, and are set well above
the measured perception floor (~3 mm / ~3 deg for a single marker observation)
but tight enough that a real regression in smoothing, IK, or frame conventions
fails the test.
"""

from __future__ import annotations

import numpy as np

from vid2traj import run_pipeline
from tests.conftest import quat_angle_deg, rmse

POSITION_RMSE_TOL_M = 0.02  # 2 cm
ORIENTATION_RMSE_TOL_DEG = 5.0
POSITION_MAX_TOL_M = 0.05  # no single frame may be wildly off


def test_pipeline_recovers_ground_truth_ee_pose(synth_clip, panda, tmp_path):
    result = run_pipeline(
        video=synth_clip.video_path,
        embodiment=panda,
        out_dir=tmp_path / "ds",
        frontend="marker",
        camera=synth_clip.camera,
        fps=synth_clip.fps,
        task="synthetic reach",
    )

    got = result.robot_trajectory
    assert len(got) == len(synth_clip.ee_positions), "frame count must be preserved"

    pos_err = np.linalg.norm(got.ee_positions - synth_clip.ee_positions, axis=1)
    ori_err = quat_angle_deg(got.ee_quats, synth_clip.ee_quats)

    assert rmse(pos_err) <= POSITION_RMSE_TOL_M, (
        f"EE position RMSE {rmse(pos_err) * 1000:.1f} mm exceeds "
        f"{POSITION_RMSE_TOL_M * 1000:.0f} mm"
    )
    assert rmse(ori_err) <= ORIENTATION_RMSE_TOL_DEG, (
        f"EE orientation RMSE {rmse(ori_err):.2f} deg exceeds {ORIENTATION_RMSE_TOL_DEG} deg"
    )
    assert pos_err.max() <= POSITION_MAX_TOL_M, (
        f"worst-frame EE position error {pos_err.max() * 1000:.1f} mm "
        f"exceeds {POSITION_MAX_TOL_M * 1000:.0f} mm"
    )


def test_pipeline_tracks_motion_not_just_the_mean(synth_clip, panda, tmp_path):
    """Guard against a degenerate 'output the home pose forever' solution.

    A trajectory frozen at the mean would still pass a loose RMSE bound if the
    motion were small, so assert the recovered path actually moves, and that it
    correlates with ground truth.
    """
    result = run_pipeline(
        video=synth_clip.video_path,
        embodiment=panda,
        out_dir=tmp_path / "ds",
        frontend="marker",
        camera=synth_clip.camera,
        fps=synth_clip.fps,
    )
    got = result.robot_trajectory.ee_positions
    truth = synth_clip.ee_positions

    travelled = np.linalg.norm(np.diff(got, axis=0), axis=1).sum()
    assert travelled > 0.10, "recovered EE barely moves; pipeline is degenerate"

    for axis in range(3):
        if truth[:, axis].std() < 1e-3:
            continue
        corr = np.corrcoef(got[:, axis], truth[:, axis])[0, 1]
        assert corr > 0.9, f"axis {axis} correlation with ground truth is only {corr:.3f}"
