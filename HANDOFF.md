# Handoff

State of vid2traj as of 2026-07-27.

- Repo: https://github.com/TarunYadgirkar/vid2traj
- Site: https://vid2traj.vercel.app
- `pytest` → **52 passed, 1 skipped** on a plain checkout; the skip is the whole
  `lerobot` compatibility module (4 tests), which runs and passes once the
  optional package is installed — **56 passed**

---

## What works

**The whole pipeline, end to end.** `run_pipeline()` takes a video path and
writes a LeRobotDataset. Every acceptance test in the brief passes.

| Area | Status | Evidence |
|---|---|---|
| Perception — ArUco marker frontend | Works, metric, deterministic | 1.7 mm EE RMSE end to end |
| Perception — MediaPipe frontend | **Implemented, not validated** | see gaps below |
| World lift + gap filling + smoothing | Works | short gaps interpolated, long gaps flagged not invented |
| IK retargeting (damped least squares on MuJoCo Jacobians) | Works | 0 unreachable frames on the fixtures |
| Safety (limits, rate limiting, self-collision) | Works | zero unsafe frames emitted, verified by re-checking in MuJoCo |
| LeRobot v3.0 export | Works | loads + fully iterates under real `lerobot` 0.4.4 |
| Determinism | Works | byte-identical Parquet and MP4 across runs |
| Graceful degradation | Works | occlusion / off-frame / two-people / blank all covered |
| Second embodiment (SO-101) | Works as a config-only addition | runs the same code path; see caveat |
| MuJoCo robot rendering | Works headless on this machine | used by the review page and the site |
| Review page + explainer site | Built, reviewed in-browser, deployed | |

**Measured on the committed fixtures (Franka Panda, 90 frames @ 30 fps):**

| Fixture | EE RMSE | Observed | Held | Collisions |
|---|---|---|---|---|
| `clean_reach` | 1.7 mm | 100% | 0 | 0 |
| `two_people` | 1.7 mm | 100% | 0 | 0 |
| `occluded_grasp` | 41.9 mm | 80% | 0 | 0 |

`occluded_grasp` is the honest number for interpolating across an 18-frame
blackout — the gap is reported, not hidden.

## What is stubbed, unvalidated, or narrower than it looks

1. **The MediaPipe frontend has never been run on real hand video.** It is
   written and import-guarded, but there is no real footage in this environment
   and no test covers it. Treat it as unverified code. The entire tested path
   uses the ArUco frontend. This is the single biggest gap between "vid2traj
   converts video of a hand" and what is actually demonstrated.

2. **All fixtures are synthetic.** The brief allowed generating clips in sim if
   real ones could not be sourced, and that is what happened. The synthetic
   renderer is a virtual camera projecting a marker — it does not model motion
   blur, rolling shutter, lighting, or a real hand.

3. **SO-101's retargeting quality is unvalidated.** It is proven to run the same
   code path from config alone, and its outputs are safe and in-limits. But the
   only trajectories fed to it were generated for the Panda's workspace, which
   is far outside SO-101's ~0.5 m reach, so the IK saturates. Nothing measures
   how well it tracks a trajectory it *could* reach.

4. **One episode per dataset.** The exporter writes a single episode into
   `chunk-000/file-000`. The v3.0 format packs many episodes per file; that
   path is not implemented.

5. **Metric scale for un-instrumented video is approximated,** by assuming a
   nominal 9 cm hand span. This is inherent to monocular hand pose and is
   declared out of scope in SPEC §6, but it means the MediaPipe path yields
   trajectories whose absolute scale is a guess.

6. **`lerobot` is installed with `--no-deps`** because its `torchvision>=0.21`
   requirement cannot be satisfied on x86_64 macOS. The compatibility test does
   run against the real library. A machine with normal wheels should just
   `pip install lerobot`.

7. **Gripper aperture is constant on the marker path.** A fiducial encodes no
   hand opening, so the marker frontend emits a fixed aperture unless a second
   marker is configured. The gripper column exists and is range-checked, but it
   carries no signal in the fixtures.

## The three highest-value next steps

### 1. Validate the MediaPipe frontend against real video — and put a number on it
This is what turns the project from "marker-based capture" into the thing the
brief describes. Record (or source) a clip of a hand doing a reach, with an
ArUco marker attached to the same wrist. Run both frontends on the *same*
footage and report the MediaPipe wrist pose error against the marker as
reference. That single experiment converts the biggest unknown into a measured
error bar, and the harness for it already exists — the frontends are
interchangeable behind `HandFrontend` and the pipeline takes either.

### 2. Workspace-aware retargeting so a smaller robot is genuinely usable
Right now a human trajectory is mapped into robot coordinates one-to-one, which
works for a Panda and saturates a SO-101. Add a workspace transform to the
embodiment config — an affine scale/offset fitted so the demonstration's
bounding box maps into the robot's reachable envelope — plus a reachability
report when targets fall outside it. Then re-run the synthetic regression on
SO-101 natively and give it real accuracy numbers, closing gap 3.

### 3. Multi-episode datasets and a batch CLI
Real training sets are hundreds of takes. Extend the exporter to append
episodes (the per-episode metadata rows and `dataset_from_index` /
`dataset_to_index` bookkeeping already exist for it) and add
`vid2traj convert DIR/*.mp4 --out dataset` with per-episode task strings and an
aggregate safety report. This is mechanical rather than research-y, and it is
what stands between this and being usable on a real data-collection run.

## Things worth knowing before you touch the code

- **The safety pass order is load-bearing.** Rate limiting runs *before* the
  collision check. Reversing it makes the smoothness test fail, because a hard
  hold from full speed is an acceleration spike. See DECISIONS D9.
- **Marker size is a single shared constant** (`DEFAULT_MARKER_SIZE`) imported
  by both the frontend and the synthetic renderer. Depth scales linearly with
  it; when they drifted apart it produced a flat ~216 mm error. See D8.
- **Gripper-internal contacts are excluded from self-collision** — a closed
  parallel gripper touches itself by design. Derived from `gripper.joints`, so
  it stays embodiment-agnostic. See D8b.
- **Keep torch and lerobot out of the core import path.** The library writes the
  LeRobot format with pyarrow/pandas/ffmpeg only; just the tests import
  `lerobot`. `import torch` costs ~30 s under Rosetta on this machine.
- **Quaternions are (w, x, y, z) everywhere**, converted at the SciPy boundary
  in `math3d.py`.
- Regenerate fixtures with `python scripts/make_fixtures.py`; regenerate site
  assets with `python scripts/build_site.py`. Both are deterministic.
