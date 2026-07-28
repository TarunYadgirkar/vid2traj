"""Command line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import __version__
from .camera import CameraModel
from .config import list_embodiments, load_embodiment
from .pipeline import run_pipeline
from .render.synth import make_demo_joint_trajectory, render_marker_clip


def _add_convert(subparsers) -> None:
    parser = subparsers.add_parser("convert", help="video -> LeRobot dataset")
    parser.add_argument("video", type=Path)
    parser.add_argument("--embodiment", default="franka_panda")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frontend", default="marker", choices=["marker", "mediapipe"])
    parser.add_argument("--camera", type=Path, help="camera intrinsics/extrinsics JSON")
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--task", default="manipulation demonstration")
    parser.add_argument("--no-smoothing", action="store_true")


def _add_synth(subparsers) -> None:
    parser = subparsers.add_parser("synth", help="render a synthetic clip with ground truth")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--embodiment", default="franka_panda")
    parser.add_argument("--frames", type=int, default=90)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--occlude", action="store_true", help="hide the hand mid-clip")
    parser.add_argument("--distractor", action="store_true", help="add a second person")


def _add_viz(subparsers) -> None:
    parser = subparsers.add_parser("viz", help="build the side-by-side HTML visualization")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--embodiment", default=None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vid2traj", description=__doc__)
    parser.add_argument("--version", action="version", version=f"vid2traj {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    _add_convert(subparsers)
    _add_synth(subparsers)
    _add_viz(subparsers)
    subparsers.add_parser("embodiments", help="list available embodiment configs")

    args = parser.parse_args(argv)

    if args.command == "embodiments":
        for name in list_embodiments():
            embodiment = load_embodiment(name)
            print(f"{name:16s} {embodiment.n_joints} joints  model={embodiment.model_path.name}")
        return 0

    if args.command == "synth":
        return _run_synth(args)

    if args.command == "convert":
        return _run_convert(args)

    if args.command == "viz":
        return _run_viz(args)

    raise AssertionError(f"unhandled command {args.command}")


def _run_synth(args) -> int:
    embodiment = load_embodiment(args.embodiment)
    joints = make_demo_joint_trajectory(embodiment, n_frames=args.frames, seed=args.seed)
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    occlude = range(args.frames // 3, args.frames // 3 + args.frames // 8) if args.occlude else None
    clip = render_marker_clip(
        embodiment,
        joints,
        out_dir / "clip.mp4",
        fps=args.fps,
        occlude_frames=occlude,
        distractor=args.distractor,
    )
    clip.camera.save(out_dir / "camera.json")
    np.save(out_dir / "ground_truth_joints.npy", clip.joints)
    np.save(out_dir / "ground_truth_ee_positions.npy", clip.ee_positions)
    np.save(out_dir / "ground_truth_ee_quats.npy", clip.ee_quats)

    print(f"wrote {clip.video_path} ({len(joints)} frames @ {args.fps} fps)")
    print(f"wrote {out_dir / 'camera.json'} and ground-truth arrays")
    return 0


def _run_convert(args) -> int:
    camera = CameraModel.load(args.camera) if args.camera else None
    result = run_pipeline(
        video=args.video,
        embodiment=args.embodiment,
        out_dir=args.out,
        frontend=args.frontend,
        camera=camera,
        fps=args.fps,
        task=args.task,
        smoothing=not args.no_smoothing,
    )
    print(f"dataset: {result.dataset_dir}")
    print(f"safety : {result.safety_report.summary()}")
    if result.safety_report.observed_fraction < 0.9:
        print(
            f"warning: only {result.safety_report.observed_fraction:.0%} of frames had a "
            "confident detection; the rest hold the previous pose",
            file=sys.stderr,
        )
    return 0


def _run_viz(args) -> int:
    from .viz.html import build_visualization

    meta = json.loads((args.dataset / "meta" / "info.json").read_text())
    embodiment = load_embodiment(args.embodiment or meta["robot_type"])
    out = build_visualization(args.dataset, args.video, args.out, embodiment)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
