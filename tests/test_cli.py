"""The CLI is the only surface most people touch, so its failures are part of the contract.

Mistakes here are cheap and common — a typo'd robot name, a clip whose intrinsics came
from a different resolution. Each must produce one readable line and exit 2, not a
traceback, and none may be silently absorbed: `--traceback` re-raises everything.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vid2traj.camera import CameraModel
from vid2traj.cli import _check_camera_matches_video, main

FIXTURES = Path(__file__).parent / "fixtures"


def run_cli(capsys, argv):
    code = main(argv)
    captured = capsys.readouterr()
    return code, captured.err


def test_unknown_embodiment_is_a_curated_error(capsys, tmp_path):
    code, err = run_cli(
        capsys,
        [
            "convert",
            str(FIXTURES / "clean_reach.mp4"),
            "--embodiment",
            "pandaa",
            "--out",
            str(tmp_path / "ds"),
        ],
    )
    assert code == 2
    assert err.startswith("vid2traj: error: unknown embodiment 'pandaa'")
    assert "franka_panda" in err and "so101" in err
    assert '"' not in err.splitlines()[0], "KeyError repr quoting leaked into the message"
    assert "vid2traj embodiments" in err


def test_missing_video_is_a_curated_error(capsys, tmp_path):
    missing = tmp_path / "nope.mp4"
    code, err = run_cli(capsys, ["convert", str(missing), "--out", str(tmp_path / "ds")])
    assert code == 2
    assert "input video not found" in err
    assert "vid2traj synth" in err


def test_viz_on_a_non_dataset_directory_is_a_curated_error(capsys, tmp_path):
    code, err = run_cli(
        capsys,
        [
            "viz",
            str(tmp_path),
            "--video",
            str(FIXTURES / "clean_reach.mp4"),
            "--out",
            str(tmp_path / "review.html"),
        ],
    )
    assert code == 2
    assert "not a LeRobot dataset" in err


def test_camera_resolution_mismatch_is_caught_before_any_work(capsys, tmp_path):
    """Mismatched intrinsics do not crash the pipeline; they rescale it. Catch it up front."""
    camera = json.loads((FIXTURES / "clean_reach.camera.json").read_text())
    camera["width"] = 1920
    camera["height"] = 1080
    bad = tmp_path / "camera.json"
    bad.write_text(json.dumps(camera))

    out = tmp_path / "ds"
    code, err = run_cli(
        capsys,
        [
            "convert",
            str(FIXTURES / "clean_reach.mp4"),
            "--camera",
            str(bad),
            "--out",
            str(out),
        ],
    )
    assert code == 2
    assert "1920x1080" in err and "960x720" in err
    assert "intrinsics scale with resolution" in err
    assert not out.exists(), "the pipeline ran anyway"


def test_traceback_flag_re_raises_instead_of_swallowing(tmp_path):
    with pytest.raises(KeyError):
        main(
            [
                "--traceback",
                "convert",
                str(FIXTURES / "clean_reach.mp4"),
                "--embodiment",
                "pandaa",
                "--out",
                str(tmp_path / "ds"),
            ]
        )


def test_embodiments_command_still_lists_both_robots(capsys):
    assert main(["embodiments"]) == 0
    out = capsys.readouterr().out
    assert "franka_panda" in out and "so101" in out


def test_matching_camera_and_video_pass_the_preflight():
    camera = CameraModel.load(FIXTURES / "clean_reach.camera.json")
    _check_camera_matches_video(camera, FIXTURES / "clean_reach.mp4")
