"""Gap filling and temporal smoothing of the world-frame wrist trajectory.

Two separate jobs, deliberately kept apart:
  * `fill_gaps`  — make short dropouts continuous; leave long ones flagged.
  * `smooth_trajectory` — attenuate per-frame estimator jitter.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

from ..math3d import canonical_quat, quat_slerp_fill, unwrap_quaternions
from ..types import WristTrajectory

DEFAULT_MAX_GAP_S = 0.5
DEFAULT_WINDOW_S = 0.25
DEFAULT_POLYORDER = 2


def _gap_runs(visible: np.ndarray) -> list[tuple[int, int]]:
    """Half-open [start, stop) runs of consecutive invisible frames."""
    runs = []
    start = None
    for i, ok in enumerate(visible):
        if not ok and start is None:
            start = i
        elif ok and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(visible)))
    return runs


def fill_gaps(
    trajectory: WristTrajectory, max_gap_s: float = DEFAULT_MAX_GAP_S
) -> WristTrajectory:
    """Interpolate across short dropouts; hold (and keep flagged) across long ones."""
    visible = trajectory.visible
    usable = visible.copy()
    if not visible.any():
        return WristTrajectory(
            positions=trajectory.positions,
            quats=trajectory.quats,
            visible=visible,
            apertures=trajectory.apertures,
            fps=trajectory.fps,
            usable=usable,
        )

    max_gap_frames = max(int(round(max_gap_s * trajectory.fps)), 1)
    idx = np.flatnonzero(visible)
    first, last = idx[0], idx[-1]

    interpolate = visible.copy()
    for start, stop in _gap_runs(visible):
        interior = start > 0 and stop < len(visible)
        if interior and (stop - start) <= max_gap_frames:
            interpolate[start:stop] = True
            usable[start:stop] = True

    positions = trajectory.positions.copy()
    apertures = trajectory.apertures.copy()
    targets = np.flatnonzero(interpolate & ~visible)
    if targets.size:
        for axis in range(3):
            positions[targets, axis] = np.interp(targets, idx, trajectory.positions[idx, axis])
        apertures[targets] = np.interp(targets, idx, trajectory.apertures[idx])

    quats = quat_slerp_fill(unwrap_quaternions(trajectory.quats.copy()), visible)

    # Frames outside any fillable region hold the nearest observed sample so the
    # arrays stay finite; `usable` remains False so retargeting will freeze there.
    hold = ~interpolate
    if hold.any():
        source = np.clip(np.searchsorted(idx, np.flatnonzero(hold), side="right") - 1, 0, len(idx) - 1)
        nearest = idx[source]
        nearest = np.where(np.flatnonzero(hold) < first, first, nearest)
        nearest = np.where(np.flatnonzero(hold) > last, last, nearest)
        positions[hold] = positions[nearest]
        quats[hold] = quats[nearest]
        apertures[hold] = apertures[nearest]

    return WristTrajectory(
        positions=positions,
        quats=canonical_quat(quats),
        visible=visible,
        apertures=apertures,
        fps=trajectory.fps,
        usable=usable,
    )


def smooth_trajectory(
    trajectory: WristTrajectory,
    window_s: float = DEFAULT_WINDOW_S,
    polyorder: int = DEFAULT_POLYORDER,
    max_gap_s: float = DEFAULT_MAX_GAP_S,
) -> WristTrajectory:
    """Fill short gaps, then low-pass position, orientation, and aperture."""
    filled = fill_gaps(trajectory, max_gap_s=max_gap_s)
    n_frames = len(filled)
    window = _odd_window(window_s, filled.fps, n_frames)
    if window is None:
        return filled

    order = min(polyorder, window - 1)
    positions = savgol_filter(filled.positions, window, order, axis=0, mode="nearest")
    apertures = np.clip(
        savgol_filter(filled.apertures, window, order, mode="nearest"), 0.0, 1.0
    )

    quats = unwrap_quaternions(filled.quats)
    quats = savgol_filter(quats, window, order, axis=0, mode="nearest")
    norms = np.linalg.norm(quats, axis=1, keepdims=True)
    quats = np.where(norms > 1e-9, quats / np.maximum(norms, 1e-9), filled.quats)

    return WristTrajectory(
        positions=positions,
        quats=canonical_quat(quats),
        visible=filled.visible,
        apertures=apertures,
        fps=filled.fps,
        usable=filled.usable,
    )


def _odd_window(window_s: float, fps: float, n_frames: int) -> int | None:
    window = int(round(window_s * fps))
    if window % 2 == 0:
        window += 1
    window = min(window, n_frames if n_frames % 2 == 1 else n_frames - 1)
    if window < 3:
        return None
    return window
