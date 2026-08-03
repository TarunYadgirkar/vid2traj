"""Command line interface."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

from . import __version__
from .camera import CameraModel
from .config import list_embodiments, load_embodiment
from .pipeline import run_pipeline
from .render.synth import make_demo_joint_trajectory, render_marker_clip
from .video import probe_video

EXIT_ERROR = 2

_HINTS = {
    "unknown embodiment": "run `vid2traj embodiments` to list the configs that ship with vid2traj",
    "ffmpeg": "install ffmpeg (macOS: `brew install ffmpeg`, Debian: `apt install ffmpeg`)",
    "camera resolution": "re-export the intrinsics at the video's resolution, or pass the "
    "camera.json that `vid2traj synth` wrote next to the clip",
    "video": "`vid2traj synth --out demo/` writes a clip and its camera.json to start from",
    "info.json": "point at the dataset directory `vid2traj convert --out` produced, "
    "not at the video or the parquet",
}


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
    parser.add_argument(
        "--traceback", action="store_true", help="re-raise instead of printing a short error"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    _add_convert(subparsers)
    _add_synth(subparsers)
    _add_viz(subparsers)
    subparsers.add_parser("embodiments", help="list available embodiment configs")

    args = parser.parse_args(argv)

    handlers = {
        "embodiments": _run_embodiments,
        "synth": _run_synth,
        "convert": _run_convert,
        "viz": _run_viz,
    }
    handler = handlers.get(args.command)
    if handler is None:
        raise AssertionError(f"unhandled command {args.command}")

    try:
        return handler(args)
    except (FileNotFoundError, KeyError, ValueError, RuntimeError) as exc:
        if args.traceback:
            raise
        return _fail(exc)


def _fail(exc: Exception) -> int:
    # KeyError stringifies as repr, which would wrap an already-readable message in quotes.
    message = str(exc.args[0]) if isinstance(exc, KeyError) and exc.args else str(exc)
    print(f"vid2traj: error: {message}", file=sys.stderr)
    hint = _hint(message)
    if hint:
        print(f"vid2traj: hint: {hint}", file=sys.stderr)
    print("vid2traj: rerun as `vid2traj --traceback <command>` for the full stack", file=sys.stderr)
    return EXIT_ERROR


def _hint(message: str) -> str | None:
    lowered = message.lower()
    for needle, hint in _HINTS.items():
        if needle in lowered:
            return hint
    return None


def _run_embodiments(args) -> int:
    for name in list_embodiments():
        embodiment = load_embodiment(name)
        print(f"{name:16s} {embodiment.n_joints} joints  model={embodiment.model_path.name}")
    return 0


def _require_ffmpeg() -> None:
    """Fail before the work, not after: video.write_video only notices once the frames exist."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to encode clips but was not found on PATH")


def _check_camera_matches_video(camera: CameraModel, video: Path) -> None:
    """Intrinsics scale with resolution, so a mismatch silently rescales the whole trajectory."""
    probe = probe_video(video)
    if (probe["width"], probe["height"]) != (camera.width, camera.height):
        raise ValueError(
            f"camera resolution {camera.width}x{camera.height} does not match "
            f"{video.name} at {probe['width']}x{probe['height']}; "
            "intrinsics scale with resolution, so the recovered metric scale would be wrong"
        )


def _run_synth(args) -> int:
    _require_ffmpeg()
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
    _require_ffmpeg()
    if not args.video.exists():
        raise FileNotFoundError(f"input video not found: {args.video}")

    camera = CameraModel.load(args.camera) if args.camera else None
    if camera is not None:
        _check_camera_matches_video(camera, args.video)

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

    _require_ffmpeg()
    info = args.dataset / "meta" / "info.json"
    if not info.exists():
        raise FileNotFoundError(f"not a LeRobot dataset, no {info}")

    meta = json.loads(info.read_text())
    embodiment = load_embodiment(args.embodiment or meta["robot_type"])
    out = build_visualization(args.dataset, args.video, args.out, embodiment)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
