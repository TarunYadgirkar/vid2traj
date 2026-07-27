# vid2traj — Specification

Convert ordinary RGB video of a human performing a manipulation task into a
robot-executable trajectory dataset, exported in the LeRobot v2 format.

## 1. Goal

Input: a monocular RGB video (phone / webcam, no depth).
Output: a `LeRobotDataset` (Parquet + MP4 + metadata) whose `action` /
`observation.state` streams are a kinematically valid, safety-checked joint
trajectory for a chosen robot embodiment, plus a side-by-side HTML visualization
for eyeballing quality.

## 2. Pipeline stages

```
video ─▶ perception ─▶ world-frame wrist ─▶ smoothing ─▶ retarget (IK) ─▶ safety ─▶ export
        (hand pose)    trajectory                        to embodiment   pass       (LeRobot)
```

1. **Perception (hand/wrist pose per frame).** Pluggable frontends behind one
   interface. Each frontend maps a frame to an optional `WristObservation`
   (6-DOF pose in the camera frame + a visibility flag + handedness/track id).
   - `marker` — ArUco fiducial via OpenCV `solvePnP`. Metric, deterministic.
     Used for synthetic ground-truth and for marker-assisted real capture.
   - `mediapipe` — MediaPipe Hands landmarks → wrist pose. Optional heavy dep;
     the primary frontend for un-instrumented real hand video.
   A frontend returning `None` for a frame = "no confident detection".

2. **World-frame wrist trajectory.** Apply a known camera→world extrinsic
   (config; identity by default) to lift per-frame camera-frame poses into a
   fixed world frame. Emit a `WristTrajectory` (T×(position, quaternion,
   visible)).

3. **Temporal smoothing.** Fill short gaps (hold/interpolate), then smooth
   position (Savitzky–Golay / one-euro-style low-pass) and orientation (SLERP
   window). Never invents motion across long gaps — long gaps stay flagged.

4. **Kinematic retargeting (IK).** Map the smoothed wrist pose to the target
   robot end-effector pose (via a fixed wrist→EE offset from the embodiment
   config) and solve damped-least-squares IK against the embodiment's MuJoCo
   model. Gripper opening maps from a normalized hand-aperture signal.
   **Embodiment is data, not code**: URDF/MJCF path, EE site, joint set,
   limits, home pose, wrist→EE offset, gripper mapping, and IK gains all live
   in a YAML config.

5. **Safety pass.** For every candidate joint vector, in order:
   - clamp to joint position limits,
   - reject self-collision (MuJoCo contact query) and out-of-limit velocity
     (finite difference vs. previous accepted sample against a per-joint cap).
   On any rejection, **hold the previous valid pose** rather than emitting a bad
   sample. First-frame rejection falls back to the home pose.

6. **Export.** Write a `LeRobotDataset` (v2 layout): per-episode Parquet with
   `action`, `observation.state`, `timestamp`, `frame_index`, `episode_index`,
   `index`, `task_index`; an encoded MP4 video stream keyed
   `observation.images.side`; `meta/info.json`, `episodes`, `tasks`, `stats`.

## 3. Public interfaces (library)

```python
from vid2traj import (
    Embodiment,           # load_embodiment(path|name) -> Embodiment
    HandFrontend,         # ABC: process(frame, t) -> WristObservation | None
    MarkerFrontend, MediaPipeFrontend,
    WristObservation, WristTrajectory, RobotTrajectory,
    smooth_trajectory,    # WristTrajectory -> WristTrajectory
    Retargeter,           # (Embodiment) ; retarget(WristTrajectory) -> RobotTrajectory
    SafetyChecker,        # (Embodiment) ; filter(RobotTrajectory) -> RobotTrajectory + report
    export_lerobot,       # (RobotTrajectory, video, out_dir, ...) -> path
    run_pipeline,         # video path -> dataset dir (end to end)
)
```

### CLI

```
vid2traj convert INPUT.mp4 --embodiment franka_panda --out DIR [--frontend marker|mediapipe]
vid2traj synth --out DIR            # render a synthetic GT clip + its known trajectory
vid2traj viz DATASET_DIR --video INPUT.mp4 --out viz.html
vid2traj embodiments                # list available embodiment configs
```

## 4. Target embodiments

- `franka_panda` — Franka Emika Panda (7 revolute arm joints, full 6-DOF task
  space with 1 DOF redundancy) + parallel gripper. MuJoCo Menagerie MJCF.
- `so101` — SO-101 (5 arm joints) + gripper. Added purely as a second YAML to
  prove the retargeting layer is embodiment-agnostic.

## 5. Acceptance criteria (see `tests/`)

- **Synthetic round-trip (core regression).** Generate a known EE-pose
  trajectory, render a marker video of it, run the full pipeline, assert the
  retargeted robot EE pose tracks the original within tolerance
  (position ≤ 2 cm RMSE, orientation ≤ 5° RMSE on the deterministic marker path).
- **Every emitted frame is safe.** All exported joint vectors satisfy the
  embodiment joint limits and are self-collision-free under MuJoCo.
- **Smoothness.** No inter-frame joint velocity/acceleration exceeds configured
  thresholds anywhere in the exported trajectory.
- **LeRobot compatibility.** The exported dataset loads with the real `lerobot`
  package and iterates over every frame without error.
- **Determinism.** Same input video + config → byte-identical Parquet and MP4
  across runs on the same machine.
- **Graceful degradation.** Occluded / out-of-frame hands and multi-person
  frames never crash and never emit an unsafe or discontinuous sample.

## 6. Out of scope

- Metric absolute scale from un-instrumented monocular video (documented
  limitation of the `mediapipe` frontend; marker frontend is metric).
- Dynamics / force / torque; contact-rich in-hand manipulation.
- Real-time / on-robot execution; learning a policy from the dataset.
- Multi-camera fusion, depth sensors, full-body pose.
- GPU requirement — everything runs CPU-only.
