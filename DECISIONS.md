# Decisions

Running log of non-obvious choices made autonomously. Newest first.

## D9 — Test correction: how a safety violation is probed (2026-07-27)
The original `test_violations_hold_the_previous_valid_pose` fed the checker
`[home, home, bad, home]`, where `bad` was a distant self-colliding pose, and
asserted the emitted frame equalled the previous one. It failed — correctly.

The safety pass applies the velocity/acceleration clamp **before** the collision
check, so a one-frame jump to a distant pose is never executed: the arm takes a
small, legal step toward it and that intermediate pose is collision-free. No
unsafe sample is emitted, which is the property that actually matters, but no
hold is triggered either, because nothing was ever violated.

The ordering is not incidental — it is forced. A hard hold applied *before* the
velocity clamp would drop the joint velocity to zero in one frame, producing an
acceleration spike that violates the smoothness acceptance test. Both
requirements come from the brief, and clamping first is the only ordering that
satisfies both.

So the test was wrong, not the code. It was replaced with a **stronger** pair:
candidates that ramp steadily from home into a colliding pose (which a real bad
retargeting looks like, and which the limiter cannot dodge), asserting that
(a) zero colliding poses are ever emitted, (b) holds are recorded in the report,
and (c) once blocked, the output stops advancing entirely. Measured behaviour:
the first colliding *candidate* is frame 171/200, zero colliding frames are
emitted, 29 frames are held, and post-hold motion is exactly 0.0.

## D8 — Shared marker-size constant (2026-07-27)
End-to-end error was a suspiciously flat ~216 mm. Cause: `MarkerConfig` assumed
a 0.06 m marker while the synthetic renderer drew a 0.08 m one. Depth from a
fiducial scales linearly with the assumed side length, so the 1.333x mismatch
became a pure range error (0.85 m camera distance x 0.25 = 212 mm). Both now
import a single `DEFAULT_MARKER_SIZE`, so the two cannot drift apart again.
Post-fix error: 34 mm.

## D8b — Gripper contacts are not self-collisions (2026-07-27)
After the size fix, 16 frames of a *known collision-free* trajectory were still
flagged. Every contact was `left_finger` vs `right_finger`: a closed parallel
gripper resting against itself. Touching is what a gripper does, so contacts
internal to the gripper are excluded from the self-collision test. The excluded
bodies are derived from the config's `gripper.joints`, so this stays
embodiment-agnostic — no part names are hard-coded. The reference pose also
parks the fingers open rather than at `qpos0` (fully closed).
Post-fix error: **6.2 mm position RMSE, 2.2 deg orientation RMSE**, zero holds,
zero collisions.

## D7 — ArUco marker frame convention (2026-07-27)
Initial round-trip tests recovered poses ~180° off with ~0.5 m position error.
Root cause: OpenCV's `SOLVEPNP_IPPE_SQUARE` (and the deprecated
`estimatePoseSingleMarkers`) pair the detected corner order (TL, TR, BR, BL)
with a **fixed canonical object-point order**:
```
[(-s/2, +s/2, 0), (+s/2, +s/2, 0), (+s/2, -s/2, 0), (-s/2, -s/2, 0)]
```
This defines the marker frame as X-right, Y-up, **Z out of the marker face**.
Because OpenCV's camera frame is X-right, Y-**down**, Z-forward, a marker whose
face points *at* the camera therefore has rotation `Rx(π)`, not identity. Using
identity renders the *back* of the marker (mirrored image), which ArUco detects
as a candidate but fails to decode.

Fixes adopted: (a) use the canonical object points everywhere — both when
rendering synthetic clips and when solving; (b) treat `Rx(π)` as the
camera-facing baseline orientation in the synthetic generator. Round-trip error
after the fix: **1.7–3.5 mm position, 0.05–3.4° orientation** over a sweep of
tilts, which sets the achievable floor for the end-to-end tolerance.

## D0 — Environment / disk (2026-07-27)
- Host: macOS, x86_64 conda Python 3.11.8. **Only ~2–3 GB free disk.** This
  dominates several choices below.
- Reuse the existing conda env instead of a fresh venv: torch 2.2.2, numpy
  1.26.4, scipy 1.17.1, PyAV 18 are already installed. A venv would re-download
  ~1 GB of torch and blow the disk budget.
- Install runtime deps with `pip --no-cache-dir` to avoid cache bloat.

## D1 — Hand-pose estimator: evaluate ≥2, choose (2026-07-27)
Candidates evaluated for the real-video frontend:
| Estimator | 3D metric | Maintained | CPU-friendly | Disk / weights | Verdict |
|-----------|-----------|------------|--------------|----------------|---------|
| **MediaPipe Hands** (Google `mediapipe`) | 2.5D landmarks, relative depth | Yes, active | Yes, real-time on CPU | ~10 MB model | **Chosen** as default real-video frontend |
| HaMeR (transformer hand-mesh) | Yes, metric-ish (MANO) | Yes | Slow on CPU, wants GPU | ~2.5 GB weights + torch/ViT | Rejected: disk + CPU |
| WiLoR | Yes | Yes | GPU-oriented | ~GBs weights + detector | Rejected: disk + CPU |
| FrankMocap | Yes | **Unmaintained** | No | large | Rejected: unmaintained |

**Choice: MediaPipe Hands** for un-instrumented real video — maintained, tiny,
CPU-real-time, trivial install. Its weakness (no absolute metric scale from a
single RGB camera) is inherent to monocular hand pose and documented as out of
scope; scale is approximated from a nominal hand size.

**Additional deterministic frontend: `marker` (ArUco + solvePnP).** Motivated by
testability: a fiducial of known size yields an exact, metric, deterministic
6-DOF wrist pose with no learned weights. This is what the synthetic
ground-truth regression test runs end-to-end, and it doubles as a legitimate
"marker-on-the-wrist" capture mode. The perception layer is an ABC so `marker`
and `mediapipe` are interchangeable — the rest of the pipeline is identical.
Rationale for leaning on `marker` in tests: the contract says prefer the option
that is *easier to test*, and a photoreal hand renderer that MediaPipe would
detect reliably is not something we can produce deterministically offline.

## D2 — Synthetic ground truth without a GL renderer (2026-07-27)
MuJoCo offscreen rendering needs a GL context, which is unreliable headless on
macOS. The synthetic GT clip is therefore rendered with **OpenCV**: a virtual
pinhole camera projects a known-size ArUco marker following a known EE-pose
trajectory; frames are drawn with `cv2`. This is fully deterministic, needs no
GL, and gives exact ground truth. MuJoCo is still used for the parts that do NOT
need GL — FK, Jacobians, collision, joint limits.

## D3 — IK: hand-rolled damped least squares (2026-07-27)
Rather than add a dependency (`mink`, `pink`, `ikpy`), retargeting uses a
damped-least-squares / Levenberg–Marquardt solver on MuJoCo's analytic Jacobian
(`mujoco.mj_jac`). ~60 lines, deterministic, no extra disk, and keeps the IK
gains in the embodiment config. Redundancy (Panda's 7th joint) is resolved by a
null-space bias toward the home pose.

## D4 — "6-DOF arm" vs Panda's 7 joints (2026-07-27)
The brief says "6-DOF arm ... Franka Panda URDF is fine," but Panda has 7
revolute joints. We read "6-DOF" as the task-space DOF of the end-effector
(3 position + 3 orientation), which Panda realizes with 1 redundant joint. Panda
is used as-is with all 7 joints controlled. SO-101 (5 joints) is the genuinely
lower-DOF second embodiment; for it, orientation tracking is relaxed to
position + approach direction because 5 joints cannot hit an arbitrary 6-DOF
pose. This relaxation is a per-embodiment config flag, not a code fork.

## D5 — Determinism strategy (2026-07-27)
- Parquet: fixed column order, fixed dtypes, no wall-clock fields; timestamps
  derived from `frame_index / fps`.
- MP4: encode with ffmpeg `-fflags +bitexact -flags:v +bitexact` and stripped
  metadata so output is bit-exact run-to-run on the same machine.
- No `Math.random`/wall-clock anywhere in the numeric path; IK seeded from the
  config home pose.

## D6 — Config format YAML (2026-07-27)
Embodiments are YAML (human-edited). `pyyaml` is tiny and already present in the
conda env. Adding a robot = adding a YAML under `configs/embodiments/`.
