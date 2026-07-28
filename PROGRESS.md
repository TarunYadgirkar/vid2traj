# Progress

Updated every ~30 min. Assume reader is catching up cold.

## Now
- Pipeline complete and green. Visualization built and reviewed in-browser.
- Next: explainer/landing site, README + HANDOFF.

## Done
- [x] Env recon, deps, disk triage
- [x] SPEC / DECISIONS / DEPENDENCIES
- [x] Vendored Franka Panda + SO-ARM100 (SO-101 family) MJCF, TCP sites added
- [x] Acceptance suite written first, from the spec, all failing
- [x] Full pipeline implemented; **33/33 acceptance tests green**
- [x] LeRobot **v3.0** export, verified by loading + iterating under the real
      `lerobot` 0.4.4 library (not a hand-rolled reader)
- [x] SO-101 second embodiment — config file only, zero code change
- [x] MuJoCo offscreen robot rendering (works headless here, contrary to the
      initial assumption in D2)
- [x] Side-by-side HTML visualization, reviewed and revised twice in-browser
- [x] 3 committed fixture clips + their exported datasets

## Numbers (clean_reach fixture, Franka Panda)
- EE tracking error **1.7 mm RMSE**, 7.7 mm worst frame
- orientation error ~2 deg RMSE
- 100% frames observed, 0 held, 0 collisions, 0 unreachable
- occluded_grasp: 41.9 mm RMSE across an 18-frame blackout (expected — that is
  interpolation across a gap, and the gap is reported honestly)
- two_people: 1.7 mm — the tracker stays locked on the target subject

## Next (ordered)
1. Explainer / landing page, deploy to Vercel
2. README + HANDOFF
3. Stretch: mediapipe frontend on real video, multi-episode datasets

## Blocked
- none. (`lerobot` needed a `--no-deps` install because its `torchvision>=0.21`
  requirement is unsatisfiable on x86_64 macOS — see DEPENDENCIES. The
  acceptance test still runs against the real library.)

## Notes
- Keep torch/lerobot out of the core import path; only tests import lerobot.
- Regenerate fixtures with `python scripts/make_fixtures.py`.
