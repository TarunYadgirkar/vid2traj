"""Write a LeRobotDataset in the v3.0 layout.

Layout produced (matching `lerobot.datasets.utils` path templates):
    meta/info.json
    meta/tasks.parquet                        index = task string, col task_index
    meta/episodes/chunk-000/file-000.parquet  per-episode metadata + flattened stats
    meta/stats.json
    data/chunk-000/file-000.parquet
    videos/observation.images.side/chunk-000/file-000.mp4

vid2traj writes this format directly rather than importing lerobot, so the core
library stays free of torch and its dependency tree. The acceptance suite then
loads the result with the real `lerobot` package, which is what actually proves
the schema is right (see tests/test_lerobot_export.py).

Determinism: fixed column order and dtypes, timestamps derived from the frame
index rather than a clock, and bit-exact video encoding (DECISIONS D5).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ..config import Embodiment
from ..types import RobotTrajectory
from ..video import read_frames, write_video

CODEBASE_VERSION = "v3.0"
CHUNK_SIZE = 1000
DATA_FILE_SIZE_MB = 100
VIDEO_FILE_SIZE_MB = 200
VIDEO_KEY = "observation.images.side"
DEFAULT_TASK = "manipulation demonstration"

DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
EPISODES_PATH = "meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"


def _feature_names(embodiment: Embodiment) -> list[str]:
    return list(embodiment.arm_joints) + ["gripper"]


def _stats(values: np.ndarray) -> dict:
    flat = np.asarray(values, dtype=np.float64).reshape(len(values), -1)
    return {
        "min": flat.min(axis=0).tolist(),
        "max": flat.max(axis=0).tolist(),
        "mean": flat.mean(axis=0).tolist(),
        "std": flat.std(axis=0).tolist(),
        "count": [int(len(flat))],
    }


def _video_stats(frames: list[np.ndarray], max_samples: int = 16) -> dict:
    """Per-channel statistics in lerobot's (C, 1, 1) layout, normalized to 0..1."""
    step = max(len(frames) // max_samples, 1)
    sampled = np.stack(frames[::step])[..., ::-1] / 255.0  # BGR -> RGB
    per_channel = sampled.reshape(-1, 3)
    return {
        "min": per_channel.min(axis=0).reshape(3, 1, 1).tolist(),
        "max": per_channel.max(axis=0).reshape(3, 1, 1).tolist(),
        "mean": per_channel.mean(axis=0).reshape(3, 1, 1).tolist(),
        "std": per_channel.std(axis=0).reshape(3, 1, 1).tolist(),
        "count": [int(len(sampled))],
    }


def _flatten_stats(stats: dict) -> dict:
    return {
        f"stats/{feature}/{stat}": [value]
        for feature, entries in stats.items()
        for stat, value in entries.items()
    }


def export_lerobot(
    trajectory: RobotTrajectory,
    embodiment: Embodiment,
    video_path: str | Path,
    out_dir: str | Path,
    task: str = DEFAULT_TASK,
    episode_index: int = 0,
) -> Path:
    out_dir = Path(out_dir)
    n_frames = len(trajectory)
    if n_frames == 0:
        raise ValueError("refusing to export an empty trajectory")

    names = _feature_names(embodiment)
    n_dof = len(names)
    state = np.concatenate([trajectory.joints, trajectory.gripper[:, None]], axis=1).astype(
        np.float32
    )

    frames = list(read_frames(video_path))
    if len(frames) != n_frames:
        raise ValueError(
            f"video has {len(frames)} frames but the trajectory has {n_frames}; "
            "they must correspond one to one"
        )
    height, width = frames[0].shape[:2]

    out_video = out_dir / VIDEO_PATH.format(video_key=VIDEO_KEY, chunk_index=0, file_index=0)
    write_video(frames, out_video, fps=trajectory.fps)

    # ---- data ------------------------------------------------------------
    frame_index = np.arange(n_frames, dtype=np.int64)
    timestamp = (frame_index / trajectory.fps).astype(np.float32)
    table = pa.table(
        {
            "action": pa.array(state.tolist(), type=pa.list_(pa.float32(), n_dof)),
            "observation.state": pa.array(state.tolist(), type=pa.list_(pa.float32(), n_dof)),
            "timestamp": pa.array(timestamp),
            "frame_index": pa.array(frame_index),
            "episode_index": pa.array(np.full(n_frames, episode_index, dtype=np.int64)),
            "index": pa.array(frame_index),
            "task_index": pa.array(np.zeros(n_frames, dtype=np.int64)),
        }
    )
    data_path = out_dir / DATA_PATH.format(chunk_index=0, file_index=0)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, data_path, compression="snappy", version="2.6")

    # ---- meta/info.json --------------------------------------------------
    motor_feature = {"dtype": "float32", "shape": [n_dof], "names": names}
    scalar = {"dtype": None, "shape": [1], "names": None}
    info = {
        "codebase_version": CODEBASE_VERSION,
        "robot_type": embodiment.robot_type,
        "total_episodes": 1,
        "total_frames": n_frames,
        "total_tasks": 1,
        "chunks_size": CHUNK_SIZE,
        "data_files_size_in_mb": DATA_FILE_SIZE_MB,
        "video_files_size_in_mb": VIDEO_FILE_SIZE_MB,
        "fps": float(trajectory.fps),
        "splits": {"train": "0:1"},
        "data_path": DATA_PATH,
        "video_path": VIDEO_PATH,
        "features": {
            "action": motor_feature,
            "observation.state": motor_feature,
            VIDEO_KEY: {
                "dtype": "video",
                "shape": [height, width, 3],
                "names": ["height", "width", "channel"],
                "info": {
                    "video.height": height,
                    "video.width": width,
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "video.fps": float(trajectory.fps),
                    "video.channels": 3,
                    "has_audio": False,
                },
            },
            "timestamp": {**scalar, "dtype": "float32"},
            "frame_index": {**scalar, "dtype": "int64"},
            "episode_index": {**scalar, "dtype": "int64"},
            "index": {**scalar, "dtype": "int64"},
            "task_index": {**scalar, "dtype": "int64"},
        },
    }
    (out_dir / "meta").mkdir(parents=True, exist_ok=True)
    (out_dir / "meta" / "info.json").write_text(json.dumps(info, indent=4))

    # ---- meta/tasks.parquet ---------------------------------------------
    tasks = pd.DataFrame({"task_index": [0]}, index=pd.Index([task], name="task"))
    tasks.to_parquet(out_dir / "meta" / "tasks.parquet")

    # ---- stats -----------------------------------------------------------
    stats = {
        "action": _stats(state),
        "observation.state": _stats(state),
        "timestamp": _stats(timestamp[:, None]),
        "frame_index": _stats(frame_index[:, None]),
        "episode_index": _stats(np.full((n_frames, 1), episode_index)),
        "index": _stats(frame_index[:, None]),
        "task_index": _stats(np.zeros((n_frames, 1))),
        VIDEO_KEY: _video_stats(frames),
    }
    (out_dir / "meta" / "stats.json").write_text(json.dumps(stats, indent=4))

    # ---- meta/episodes ---------------------------------------------------
    episode = {
        "episode_index": [episode_index],
        "tasks": [[task]],
        "length": [n_frames],
        "data/chunk_index": [0],
        "data/file_index": [0],
        f"videos/{VIDEO_KEY}/chunk_index": [0],
        f"videos/{VIDEO_KEY}/file_index": [0],
        # Where this episode sits inside its video file. v3.0 packs several
        # episodes per file; vid2traj writes one, so the span is the whole clip.
        f"videos/{VIDEO_KEY}/from_timestamp": [0.0],
        f"videos/{VIDEO_KEY}/to_timestamp": [float(n_frames / trajectory.fps)],
        "dataset_from_index": [0],
        "dataset_to_index": [n_frames],
        "meta/episodes/chunk_index": [0],
        "meta/episodes/file_index": [0],
        **_flatten_stats(stats),
    }
    episodes_path = out_dir / EPISODES_PATH.format(chunk_index=0, file_index=0)
    episodes_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(episode), episodes_path, compression="snappy", version="2.6")

    return out_dir
