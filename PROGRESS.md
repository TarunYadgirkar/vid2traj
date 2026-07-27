# Progress

Updated every ~30 min. Assume reader is catching up cold.

## Now
- Environment validated; Panda MJCF vendored with a TCP site; ArUco
  render→detect→solvePnP round-trip de-risked (detection working).
- Interrupted for a machine restart — committing + pushing state.
- Next after restart: resolve the IPPE_SQUARE planar pose ambiguity, then write
  the acceptance test suite (must fail), then implement to green.

## Done
- [x] Env recon (Python 3.11 conda x86_64 under Rosetta; torch/numpy/scipy/av present)
- [x] Deps installed: mujoco 3.10, opencv-contrib-headless 4.10, pyarrow 17,
      pandas 2.2.3, pyyaml. Pinned numpy==1.26.4 (torch 2.2.2 needs <2).
- [x] SPEC / DECISIONS / DEPENDENCIES written
- [x] Vendored Franka Panda MJCF (+meshes) into src/vid2traj/assets/models,
      added `tcp` site in hand body (TCP @home = [0.5545, 0, 0.5211])
- [x] Spiked ArUco pipeline: marker generation, quiet-zone padding,
      perspective render, detection. **Detection confirmed.**

## Marker-render recipe (learned in spike — use in impl)
- `DICT_4X4_50`, `generateImageMarker(dict,id,sidePixels)`.
- Pad marker with a white quiet zone (~40% of side) before warping, else
  `detectMarkers` finds candidates but rejects them (no quiet zone).
- Render: `warpPerspective(padded, H)` where `H` maps the *inner* marker pixel
  corners to `projectPoints(obj,...)`.
- **Object corners MUST be image-convention (Y-down), same winding as the source
  pixel corners** `[TL,TR,BR,BL] = [[-s/2,-s/2,0],[s/2,-s/2,0],[s/2,s/2,0],[-s/2,s/2,0]]`.
  Y-up ordering mirrors the marker and decode fails.
- TODO(impl): `SOLVEPNP_IPPE_SQUARE` returns two planar solutions (~180° apart);
  disambiguate by lowest reprojection error / marker facing camera, OR derive
  synthetic GT by round-tripping the pose through the same solvePnP so the test
  compares like-for-like.

## Next (ordered)
1. Resolve pose ambiguity; finalize marker frontend + synthetic renderer
2. Write acceptance tests (failing)
3. Implement: types → config → perception(marker) → world/smoothing →
   retarget(IK) → safety → export
4. SO-101 second embodiment (config only)
5. HTML side-by-side viz; commit synthetic fixtures
6. README + HANDOFF; install lerobot and make the load test pass for real

## Blocked
- none

## Notes
- torch import ~30s under Rosetta → keep torch/lerobot OUT of the core library
  import path (library writes LeRobot format via pyarrow/pandas/ffmpeg only;
  only the lerobot-load *test* imports lerobot).
- Disk was tight (~2 GB) then freed to ~10 GB; still watch `df`.
