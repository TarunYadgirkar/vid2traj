"""Generate the committed fixture clips and their datasets.

Three short clips standing in for self-recorded demonstrations: a clean take, a
take where the hand is occluded mid-motion, and a take with a second person in
frame. Regenerating them is deterministic — same bytes every run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from vid2traj import load_embodiment, run_pipeline
from vid2traj.render.synth import make_demo_joint_trajectory, render_marker_clip

FIXTURES = [
    ("clean_reach", dict(), "reach and return, hand visible throughout"),
    ("occluded_grasp", dict(occlude_frames=range(34, 52)), "hand occluded mid-motion"),
    ("two_people", dict(distractor=True), "a second person in frame throughout"),
]
N_FRAMES = 90
FPS = 30


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("tests/fixtures"))
    parser.add_argument("--embodiment", default="franka_panda")
    parser.add_argument("--viz", type=Path, default=None, help="also build a viz for clip 1")
    args = parser.parse_args()

    embodiment = load_embodiment(args.embodiment)
    args.out.mkdir(parents=True, exist_ok=True)
    joints = make_demo_joint_trajectory(embodiment, n_frames=N_FRAMES, seed=0)

    for name, kwargs, description in FIXTURES:
        clip = render_marker_clip(
            embodiment, joints, args.out / f"{name}.mp4", fps=FPS, **kwargs
        )
        clip.camera.save(args.out / f"{name}.camera.json")
        np.save(args.out / f"{name}.ee_positions.npy", clip.ee_positions)
        result = run_pipeline(
            video=clip.video_path,
            embodiment=embodiment,
            out_dir=args.out / f"{name}.dataset",
            camera=clip.camera,
            fps=FPS,
            task=description,
        )
        error = np.linalg.norm(
            result.robot_trajectory.ee_positions - clip.ee_positions, axis=1
        )
        print(
            f"{name:16s} {clip.video_path.stat().st_size // 1024:4d} KB  "
            f"rmse {np.sqrt(np.mean(error**2)) * 1000:6.1f} mm  |  {result.safety_report.summary()}"
        )

        if args.viz and name == "clean_reach":
            from vid2traj.viz.html import build_visualization

            out = build_visualization(
                result.dataset_dir,
                clip.video_path,
                args.viz,
                embodiment,
                trajectory=result.robot_trajectory,
                report=result.safety_report,
                wrist=result.wrist_trajectory,
                ground_truth_positions=clip.ee_positions,
                task=description,
            )
            print(f"viz: {out} ({out.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
