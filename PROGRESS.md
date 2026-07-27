# Progress

Updated every ~30 min. Assume reader is catching up cold.

## Now
- Scaffolding + contract docs written (SPEC / DECISIONS / DEPENDENCIES).
- Next: install runtime deps, vendor Panda MJCF, write the acceptance test suite
  (must fail), then implement to green.

## Done
- [x] Env recon (Python 3.11 conda x86_64; torch/numpy/scipy/av present; ~3.7 GB free)
- [x] Repo skeleton + git init
- [x] SPEC.md, DECISIONS.md, DEPENDENCIES.md

## Next (ordered)
1. Install deps (mujoco, opencv-contrib-headless, pyarrow, pandas, pyyaml)
2. Vendor Franka Panda MJCF from mujoco_menagerie
3. Write acceptance tests (failing): synthetic round-trip, safety, smoothness,
   lerobot load, determinism, graceful degradation
4. Implement: types → config → perception(marker) → world/smoothing →
   retarget(IK) → safety → export
5. Second embodiment SO-101 (config only)
6. HTML side-by-side viz
7. Commit fixtures (synthetic clips)
8. README + HANDOFF

## Blocked
- none yet

## Notes
- Disk is the tight constraint; installing incrementally and watching `df`.
