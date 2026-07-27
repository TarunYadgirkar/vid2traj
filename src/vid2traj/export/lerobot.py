"""Write a LeRobotDataset (v2.1 layout) from a retargeted trajectory.

Layout produced:
    meta/info.json, meta/tasks.jsonl, meta/episodes.jsonl,
    meta/episodes_stats.jsonl, meta/stats.json
    data/chunk-000/episode_000000.parquet
    videos/chunk-000/observation.images.side/episode_000000.mp4

Determinism: column order and dtypes are fixed, timestamps are derived from the
frame index rather than any clock, and the video is re-encoded with the
bit-exact ffmpeg flags in `vid2traj.video` (DECISIONS D5).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..config import Embodiment
from ..types import RobotTrajectory
from ..video import read_frames, write_video

CODEBASE_VERSION = "v2.1"
CHUNK_SIZE = 1000
VIDEO_KEY = "observation.images.side"
DEFAULT_TASK = "manipulation demonstration"


def _feature_names(embodiment: Embodiment) -> list[str]:
    return list(embodiment.arm_joints) + ["gripper"]


def _stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    flat = values.reshape(len(values), -1)
    return {
        "min": flat.min(axis=0).tolist(),
        "max": flat.max(axis=0).tolist(),
        "mean": flat.mean(axis=0).tolist(),
        "std": flat.std(axis=0).tolist(),
        "count": [int(len(flat))],
    }


def _video_stats(video_path: Path, max_samples: int = 16) -> dict:
    """Per-channel statistics in lerobot's (C, 1, 1) layout, normalized to 0..1."""
    frames = list(read_frames(video_path))
    if not frames:
        raise ValueError(f"no frames decoded from {video_path}")
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


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


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

    chunk = episode_index // CHUNK_SIZE
    data_path = out_dir / f"data/chunk-{chunk:03d}/episode_{episode_index:06d}.parquet"
    out_video = out_dir / f"videos/chunk-{chunk:03d}/{VIDEO_KEY}/episode_{episode_index:06d}.mp4"

    frames = list(read_frames(video_path))
    if len(frames) != n_frames:
        raise ValueError(
            f"video has {len(frames)} frames but the trajectory has {n_frames}; "
            "they must correspond one to one"
        )
    write_video(frames, out_video, fps=trajectory.fps)
    height, width = frames[0].shape[:2]

    frame_index = np.arange(n_frames, dtype=np.int64)
    table = pa.table(
        {
            "action": pa.array(state.tolist(), type=pa.list_(pa.float32(), n_dof)),
            "observation.state": pa.array(state.tolist(), type=pa.list_(pa.float32(), n_dof)),
            "timestamp": pa.array((frame_index / trajectory.fps).astype(np.float32)),
            "frame_index": pa.array(frame_index),
            "episode_index": pa.array(np.full(n_frames, episode_index, dtype=np.int64)),
            "index": pa.array(frame_index),
            "task_index": pa.array(np.zeros(n_frames, dtype=np.int64)),
        }
    )
    data_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, data_path, compression="snappy", version="2.6", store_schema=True)

    motor_feature = {"dtype": "float32", "shape": [n_dof], "names": names}
    scalar = lambda dtype: {"dtype": dtype, "shape": [1], "names": None}  # noqa: E731
    info = {
        "codebase_version": CODEBASE_VERSION,
        "robot_type": embodiment.robot_type,
        "total_episodes": 1,
        "total_frames": n_frames,
        "total_tasks": 1,
        "total_videos": 1,
        "total_chunks": 1,
        "chunks_size": CHUNK_SIZE,
        "fps": float(trajectory.fps),
        "splits": {"train": "0:1"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
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
            "timestamp": scalar("float32"),
            "frame_index": scalar("int64"),
            "episode_index": scalar("int64"),
            "index": scalar("int64"),
            "task_index": scalar("int64"),
        },
    }
    (out_dir / "meta").mkdir(parents=True, exist_ok=True)
    (out_dir / "meta" / "info.json").write_text(json.dumps(info, indent=4, sort_keys=False))

    _write_jsonl(out_dir / "meta" / "tasks.jsonl", [{"task_index": 0, "task": task}])
    _write_jsonl(
        out_dir / "meta" / "episodes.jsonl",
        [{"episode_index": episode_index, "tasks": [task], "length": n_frames}],
    )

    stats = {
        "action": _stats(state),
        "observation.state": _stats(state),
        "timestamp": _stats((frame_index / trajectory.fps).astype(np.float32)[:, None]),
        "frame_index": _stats(frame_index[:, None]),
        "episode_index": _stats(np.full((n_frames, 1), episode_index)),
        "index": _stats(frame_index[:, None]),
        "task_index": _stats(np.zeros((n_frames, 1))),
        VIDEO_KEY: _video_stats(out_video),
    }
    _write_jsonl(
        out_dir / "meta" / "episodes_stats.jsonl",
        [{"episode_index": episode_index, "stats": stats}],
    )
    (out_dir / "meta" / "stats.json").write_text(json.dumps(stats, indent=4, sort_keys=False))

    return out_dir
