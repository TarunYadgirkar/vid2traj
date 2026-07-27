"""Video IO.

Reading goes through OpenCV. Writing goes through ffmpeg with bit-exact flags
so the same frames always produce the same bytes (see DECISIONS D5).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


def read_frames(path: str | Path) -> Iterator[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                return
            yield frame
    finally:
        capture.release()


def probe_video(path: str | Path) -> dict:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    try:
        return {
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": float(capture.get(cv2.CAP_PROP_FPS)),
            "n_frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
    finally:
        capture.release()


def write_video(frames: list[np.ndarray], path: str | Path, fps: float, crf: int = 12) -> Path:
    """Encode BGR frames to H.264 deterministically."""
    if not frames:
        raise ValueError("refusing to write an empty video")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to write videos but was not found on PATH")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        _rational_fps(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        # determinism: no wall-clock metadata, no encoder version string, fixed threading
        "-fflags",
        "+bitexact",
        "-flags:v",
        "+bitexact",
        "-threads",
        "1",
        "-x264-params",
        "threads=1:sliced-threads=0:deterministic=1",
        "-map_metadata",
        "-1",
        str(path),
    ]

    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for frame in frames:
        if frame.shape[:2] != (height, width):
            process.stdin.close()
            process.wait()
            raise ValueError("all frames must share the same resolution")
        process.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
    process.stdin.close()
    stderr = process.stderr.read()
    process.stderr.close()
    if process.wait() != 0:
        raise RuntimeError(f"ffmpeg failed: {stderr.decode(errors='replace')[:500]}")
    return path


def _rational_fps(fps: float) -> str:
    """Pass fps to ffmpeg exactly, so 29.97 does not silently become 30."""
    if abs(fps - round(fps)) < 1e-9:
        return str(int(round(fps)))
    return f"{round(fps * 1000)}/1000"
