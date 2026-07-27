"""A second embodiment must be a config file, not a code change."""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from vid2traj import list_embodiments, load_embodiment, run_pipeline
from vid2traj.safety import SafetyChecker


def test_both_shipped_embodiments_are_available():
    names = set(list_embodiments())
    assert {"franka_panda", "so101"} <= names


def test_embodiment_config_is_self_describing(panda, so101):
    for emb in (panda, so101):
        n = len(emb.arm_joints)
        assert emb.joint_limits.shape == (n, 2)
        assert emb.home_joints.shape == (n,)
        assert emb.velocity_limits.shape == (n,)
        assert emb.acceleration_limits.shape == (n,)
        assert np.all(emb.joint_limits[:, 0] < emb.joint_limits[:, 1])
        assert np.all(emb.home_joints >= emb.joint_limits[:, 0])
        assert np.all(emb.home_joints <= emb.joint_limits[:, 1])


def test_so101_has_fewer_joints_than_panda(panda, so101):
    """Proves the two configs are genuinely different kinematics, not a copy."""
    assert len(so101.arm_joints) < len(panda.arm_joints)


def test_second_embodiment_runs_the_same_pipeline(so101, synth_clip, tmp_path):
    """Same code path, different YAML. Nothing embodiment-specific in the caller."""
    result = run_pipeline(
        video=synth_clip.video_path,
        embodiment=so101,
        out_dir=tmp_path / "ds",
        frontend="marker",
        camera=synth_clip.camera,
        fps=synth_clip.fps,
    )
    traj = result.robot_trajectory
    assert traj.joints.shape == (len(synth_clip.ee_positions), len(so101.arm_joints))
    assert np.all(np.isfinite(traj.joints))

    low, high = so101.joint_limits[:, 0], so101.joint_limits[:, 1]
    assert np.all(traj.joints >= low - 1e-9) and np.all(traj.joints <= high + 1e-9)

    checker = SafetyChecker(so101)
    assert not any(checker.in_self_collision(q) for q in traj.joints)

    dt = 1.0 / traj.fps
    vel = np.abs(np.diff(traj.joints, axis=0) / dt).max(axis=0)
    assert np.all(vel <= so101.velocity_limits + 1e-6)


def test_a_brand_new_embodiment_needs_only_yaml(panda, tmp_path, synth_clip):
    """Define a third embodiment on the fly by editing config alone."""
    src = yaml.safe_load(panda.source_path.read_text())
    src["name"] = "panda_slow"
    src["limits"]["velocity_scale"] = 0.5
    custom = tmp_path / "panda_slow.yaml"
    custom.write_text(yaml.safe_dump(src, sort_keys=False))

    emb = load_embodiment(custom)
    assert emb.name == "panda_slow"
    np.testing.assert_allclose(emb.velocity_limits, panda.velocity_limits * 0.5, rtol=1e-9)

    result = run_pipeline(
        video=synth_clip.video_path,
        embodiment=emb,
        out_dir=tmp_path / "ds",
        frontend="marker",
        camera=synth_clip.camera,
        fps=synth_clip.fps,
    )
    dt = 1.0 / result.robot_trajectory.fps
    vel = np.abs(np.diff(result.robot_trajectory.joints, axis=0) / dt).max(axis=0)
    assert np.all(vel <= emb.velocity_limits + 1e-6), "config-only velocity change was ignored"


def test_unknown_embodiment_raises(tmp_path):
    with pytest.raises((KeyError, FileNotFoundError, ValueError)):
        load_embodiment("definitely_not_a_robot")
