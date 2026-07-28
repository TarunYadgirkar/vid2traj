"""Assemble the static explainer site from real pipeline artifacts.

Every image and clip on the site is produced by running the pipeline, not drawn
by hand, so the page cannot drift from what the code actually does.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from vid2traj import CameraModel, load_embodiment, run_pipeline
from vid2traj.perception.marker import MarkerFrontend, marker_object_points
from vid2traj.render.robot import RobotRenderer
from vid2traj.video import read_frames, write_video

VELLUM = (231, 237, 239)  # BGR
AMBER = (41, 182, 255)
INK = (30, 23, 20)


def detection_overlay(frame: np.ndarray, camera: CameraModel, marker_size: float) -> np.ndarray:
    """Draw what the marker frontend actually found: corners and the pose axes."""
    out = frame.copy()
    detector = MarkerFrontend(camera)
    corners, ids, _ = detector._detector.detectMarkers(frame)
    if ids is None:
        return out
    quad = corners[0].reshape(-1, 2)
    cv2.polylines(out, [quad.astype(np.int32)], True, AMBER, 2, cv2.LINE_AA)
    for point in quad:
        cv2.circle(out, tuple(point.astype(int)), 5, AMBER, -1, cv2.LINE_AA)

    ok, rvec, tvec = cv2.solvePnP(
        marker_object_points(marker_size), quad.astype(np.float64),
        camera.intrinsics, camera.distortion, flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if ok:
        axis = np.float32([[0, 0, 0], [marker_size, 0, 0], [0, marker_size, 0], [0, 0, marker_size]])
        pts, _ = cv2.projectPoints(axis, rvec, tvec, camera.intrinsics, camera.distortion)
        pts = pts.reshape(-1, 2).astype(int)
        for end, colour in zip(pts[1:], [(60, 60, 240), (90, 200, 90), (240, 150, 60)]):
            cv2.arrowedLine(out, tuple(pts[0]), tuple(end), colour, 3, cv2.LINE_AA, tipLength=.18)
    return out


def hero_crop(frames: list, camera: CameraModel) -> list:
    """Crop every frame to one fixed 4:3 box containing the whole marker path.

    The box is fixed for the clip rather than per-frame, so the crop reads as a
    tighter camera rather than a wobbling auto-follow.
    """
    detector = MarkerFrontend(camera)
    centres = []
    for frame in frames:
        corners, ids, _ = detector._detector.detectMarkers(frame)
        if ids is not None:
            centres.append(corners[0].reshape(-1, 2))
    if not centres:
        return frames
    allpts = np.concatenate(centres, axis=0)
    lo, hi = allpts.min(axis=0), allpts.max(axis=0)
    h, w = frames[0].shape[:2]
    cx, cy = (lo + hi) / 2
    need_w = (hi[0] - lo[0]) * 1.35
    need_h = (hi[1] - lo[1]) * 1.35
    box_w = int(min(w, max(need_w, need_h * 4 / 3)))
    box_w -= box_w % 2
    box_h = int(min(h, box_w * 3 / 4)); box_h -= box_h % 2
    box_w = int(min(w, box_h * 4 / 3)); box_w -= box_w % 2
    left = int(np.clip(cx - box_w / 2, 0, w - box_w))
    top = int(np.clip(cy - box_h / 2, 0, h - box_h))
    return [f[top:top + box_h, left:left + box_w] for f in frames]


def crop_box(frame: np.ndarray, camera: CameraModel, scale: float = 3.4):
    """Square crop centred on the marker. Computed from the clean frame — the
    overlay's own annotations cover the border and defeat re-detection."""
    detector = MarkerFrontend(camera)
    corners, ids, _ = detector._detector.detectMarkers(frame)
    h, w = frame.shape[:2]
    if ids is None:
        return 0, 0, h, w
    quad = corners[0].reshape(-1, 2)
    cx, cy = quad.mean(axis=0)
    side = max(quad.max(axis=0) - quad.min(axis=0)) * scale
    half = int(min(side, min(h, w)) / 2)
    cx = int(np.clip(cx, half, w - half)); cy = int(np.clip(cy, half, h - half))
    return cy - half, cx - half, 2 * half, 2 * half


def apply_crop(frame: np.ndarray, box) -> np.ndarray:
    top, left, height, width = box
    return frame[top:top + height, left:left + width]


def path_plot(positions: np.ndarray, size=(720, 540)) -> np.ndarray:
    """Top-down plot of the recovered end-effector path in the world frame."""
    w, h = size
    canvas = np.full((h, w, 3), 16, np.uint8)
    for gx in range(0, w, 60):
        cv2.line(canvas, (gx, 0), (gx, h), (34, 38, 46), 1)
    for gy in range(0, h, 60):
        cv2.line(canvas, (0, gy), (w, gy), (34, 38, 46), 1)

    xy = positions[:, [0, 2]].copy()  # x forward, z up
    lo, hi = xy.min(0), xy.max(0)
    span = np.maximum(hi - lo, 1e-6)
    pad = 70
    pts = ((xy - lo) / span) * np.array([w - 2 * pad, -(h - 2 * pad)]) + np.array([pad, h - pad])
    pts = pts.astype(np.int32)
    for i in range(1, len(pts)):
        shade = int(90 + 130 * i / len(pts))
        cv2.line(canvas, tuple(pts[i - 1]), tuple(pts[i]), (shade, shade, shade), 2, cv2.LINE_AA)
    cv2.circle(canvas, tuple(pts[0]), 6, (150, 150, 150), -1, cv2.LINE_AA)
    cv2.circle(canvas, tuple(pts[-1]), 6, AMBER, -1, cv2.LINE_AA)
    cv2.putText(canvas, "start", tuple(pts[0] + [12, 4]), cv2.FONT_HERSHEY_SIMPLEX, .45, (150, 150, 150), 1, cv2.LINE_AA)
    cv2.putText(canvas, "end", tuple(pts[-1] + [12, 22]), cv2.FONT_HERSHEY_SIMPLEX, .45, AMBER, 1, cv2.LINE_AA)
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", type=Path, default=Path("tests/fixtures"))
    parser.add_argument("--out", type=Path, default=Path("site"))
    args = parser.parse_args()

    media = args.out / "media"
    media.mkdir(parents=True, exist_ok=True)
    embodiment = load_embodiment("franka_panda")

    source = args.fixtures / "clean_reach.mp4"
    camera = CameraModel.load(args.fixtures / "clean_reach.camera.json")
    truth = np.load(args.fixtures / "clean_reach.ee_positions.npy")

    result = run_pipeline(
        video=source, embodiment=embodiment, out_dir=args.out / "_dataset",
        camera=camera, fps=30, task="reach and return", export=False,
    )
    traj = result.robot_trajectory

    frames = list(read_frames(source))
    key = 45
    box = crop_box(frames[key], camera)
    cv2.imwrite(str(media / "stage-source.jpg"), apply_crop(frames[key], box),
                [cv2.IMWRITE_JPEG_QUALITY, 90])
    cv2.imwrite(str(media / "stage-detect.jpg"),
                apply_crop(detection_overlay(frames[key], camera, 0.08), box),
                [cv2.IMWRITE_JPEG_QUALITY, 90])
    cv2.imwrite(str(media / "stage-world.jpg"), path_plot(traj.ee_positions),
                [cv2.IMWRITE_JPEG_QUALITY, 88])

    with RobotRenderer(embodiment, width=960, height=720) as renderer:
        shot = renderer.render_frame(traj.joints[key], float(traj.gripper[key]))
    cv2.imwrite(str(media / "stage-robot.jpg"), shot[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])

    shutil.copy(source, media / "source.mp4")
    write_video(hero_crop(frames, camera), media / "hero_source.mp4", fps=30)
    from vid2traj.render.robot import render_robot_video
    render_robot_video(traj, embodiment, media / "robot.mp4", width=960, height=720)

    error = np.linalg.norm(traj.ee_positions - truth, axis=1)
    stats = {
        "rmseMm": round(float(np.sqrt(np.mean(error ** 2))) * 1000, 2),
        "maxMm": round(float(error.max()) * 1000, 2),
        "frames": len(traj),
        "fps": traj.fps,
        "held": len(result.safety_report.held_frames),
        "collisions": len(result.safety_report.collision_frames),
        "observedPct": round(result.safety_report.observed_fraction * 100),
        "joints": embodiment.n_joints,
    }
    (args.out / "stats.json").write_text(json.dumps(stats, indent=2))
    shutil.rmtree(args.out / "_dataset", ignore_errors=True)
    print(json.dumps(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
