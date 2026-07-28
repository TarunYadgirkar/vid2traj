# External dependencies pulled in

Every external thing added to the machine for this project, what it is, why.

## Python packages (installed into the existing conda env, `pip --no-cache-dir`)

| Package | Why | Notes |
|---------|-----|-------|
| `mujoco` | FK, analytic Jacobians for IK, collision + joint-limit checks, robot render | CPU physics needs no GL |
| `opencv-contrib-python-headless` | ArUco marker detection + `solvePnP`, synthetic clip rendering | `-contrib` for `cv2.aruco`; `-headless` to save disk (no Qt) |
| `pyarrow` | Parquet writing for the LeRobot dataset | also a lerobot dep |
| `pandas` | tabular assembly before Parquet | also a lerobot dep |
| `pyyaml` | embodiment config loading | usually already present |
| `lerobot` 0.4.4 | acceptance test: dataset must load + iterate under the real library | **installed with `--no-deps`** — see note below |
| `datasets`, `huggingface_hub`, `accelerate`, `jsonlines`, `deepdiff` | lerobot's actual runtime imports for dataset loading | pulled explicitly since `--no-deps` skipped them |
| `mediapipe` | optional real-video hand-pose frontend (see DECISIONS D1) | import-guarded; not required by the core test suite |

Already present (not installed by us): `torch` 2.2.2, `numpy` 1.26.4, `scipy`
1.17.1, `av` (PyAV) 18.

## Cloned repositories (into `../` next to this repo, git-ignored)

| Repo | Why | Fetch method |
|------|-----|--------------|
| `google-deepmind/mujoco_menagerie` | Franka Panda MJCF + meshes | sparse checkout of `franka_emika_panda` only |
| SO-101 / SO-ARM100 model | second embodiment MJCF/URDF | sparse checkout / vendored minimal model |

Model files actually needed at runtime are **vendored** into
`src/vid2traj/assets/models/` so the package is self-contained and the clones
can be discarded.

## System tools (pre-existing)

`ffmpeg` 8.1.2 (bit-exact MP4 encode), `git`, `gh` (authenticated).


## Note: why `lerobot` is installed with `--no-deps`

`pip install lerobot` on this machine resolves to the abandoned 0.1.0 placeholder,
because every 0.3+ release requires `torchvision>=0.21`, which requires
`torch>=2.6` — and PyTorch stopped publishing **x86_64 macOS** wheels after
2.2.2 (max torchvision here is 0.17.2). The dependency is genuinely
unsatisfiable on Intel Macs, not a resolver quirk.

Since only the *dataset reader* is needed (not training, not `torchvision`'s
video decoding), lerobot 0.4.4 was installed with `--no-deps` plus the modules
its dataset path actually imports. The acceptance test then loads and iterates
the exported dataset with that real library, video decoding included (it falls
back from `torchcodec` to `pyav`, which is installed). This is a host
limitation, not a compromise in what the test verifies.
