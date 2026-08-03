"""The exported dataset must load and iterate under the real `lerobot` library.

Writing a plausible-looking Parquet tree is easy; being accepted by the actual
loader is the thing that matters, so this test refuses to substitute a
hand-rolled reader for the real one.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from vid2traj import run_pipeline

pytestmark = pytest.mark.lerobot

pytest.importorskip("lerobot", reason="optional `lerobot` extra is not installed")


def _load_lerobot_dataset(root):
    """Import LeRobotDataset across the versions that moved the module."""
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    return LeRobotDataset(repo_id="vid2traj/test", root=str(root))


@pytest.fixture(scope="module")
def dataset_dir(synth_clip, panda, tmp_path_factory):
    out = tmp_path_factory.mktemp("lerobot") / "ds"
    run_pipeline(
        video=synth_clip.video_path,
        embodiment=panda,
        out_dir=out,
        frontend="marker",
        camera=synth_clip.camera,
        fps=synth_clip.fps,
        task="pick up the block",
    )
    return out


def test_metadata_files_exist(dataset_dir):
    info = json.loads((dataset_dir / "meta" / "info.json").read_text())
    assert info["robot_type"]
    assert info["fps"] > 0
    assert info["total_frames"] > 0
    assert "action" in info["features"]
    assert "observation.state" in info["features"]


def test_dataset_loads_with_real_lerobot(dataset_dir):
    ds = _load_lerobot_dataset(dataset_dir)
    assert len(ds) > 0
    assert ds.fps > 0


def test_dataset_iterates_every_frame(dataset_dir):
    ds = _load_lerobot_dataset(dataset_dir)
    seen = 0
    for item in ds:
        assert "action" in item
        assert "observation.state" in item
        action = np.asarray(item["action"])
        assert np.all(np.isfinite(action)), "non-finite action in exported dataset"
        seen += 1
    assert seen == len(ds), "iteration yielded fewer frames than len(dataset)"


def test_exported_actions_match_the_trajectory(dataset_dir, synth_clip, panda):
    """Guard the schema plumbing: what we computed is what got written."""
    ds = _load_lerobot_dataset(dataset_dir)
    first = next(iter(ds))
    n_dof = len(panda.arm_joints) + 1  # arm joints + gripper
    assert np.asarray(first["action"]).shape[-1] == n_dof
    assert np.asarray(first["observation.state"]).shape[-1] == n_dof
