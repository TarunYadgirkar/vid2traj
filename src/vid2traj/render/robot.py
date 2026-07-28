"""Render a retargeted trajectory as robot video, for eyeballing quality.

Uses MuJoCo's offscreen renderer against a visualization-only scene (floor,
lights, backdrop). Verified working headless on this machine, contrary to the
initial assumption recorded in DECISIONS D2 — the synthetic ground-truth
generator still deliberately avoids GL, since ground truth must be reproducible
anywhere.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from ..config import Embodiment
from ..types import RobotTrajectory
from ..video import write_video

DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480


def scene_path_for(embodiment: Embodiment) -> Path:
    """The `*_scene.xml` beside the model, falling back to the bare model."""
    model_path = Path(embodiment.model_path)
    for candidate in (
        model_path.with_name(f"{model_path.stem}_scene.xml"),
        model_path.with_name(f"{embodiment.name}_scene.xml"),
    ):
        if candidate.exists():
            return candidate
    return model_path


class RobotRenderer:
    def __init__(
        self,
        embodiment: Embodiment,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        azimuth: float = 135.0,
        elevation: float = -20.0,
        distance: float | None = None,
        lookat: np.ndarray | None = None,
    ) -> None:
        self.embodiment = embodiment
        self.model = mujoco.MjModel.from_xml_path(str(scene_path_for(embodiment)))
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)

        self.qpos_adr = np.array(
            [self._qpos_adr(name) for name in embodiment.arm_joints], dtype=int
        )
        self.gripper_adr = np.array(
            [self._qpos_adr(name) for name in embodiment.gripper.joints], dtype=int
        )

        self.camera = mujoco.MjvCamera()
        self.camera.azimuth = azimuth
        self.camera.elevation = elevation
        self.camera.lookat[:] = (
            lookat if lookat is not None else self._default_lookat(embodiment)
        )
        self.camera.distance = distance if distance is not None else self._default_distance()

    def _qpos_adr(self, name: str) -> int:
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"joint {name!r} not found in the visualization scene")
        return int(self.model.jnt_qposadr[jid])

    def _default_lookat(self, embodiment: Embodiment) -> np.ndarray:
        """Aim at the middle of the arm's own workspace, whatever robot it is."""
        self.data.qpos[:] = self.model.qpos0
        self.data.qpos[self.qpos_adr] = embodiment.home_joints
        mujoco.mj_kinematics(self.model, self.data)
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, embodiment.ee_site)
        tip = self.data.site_xpos[site_id].copy()
        return np.array([tip[0] * 0.5, tip[1] * 0.5, max(tip[2] * 0.7, 0.12)])

    def _default_distance(self) -> float:
        reach = float(np.linalg.norm(self.camera.lookat)) + 0.35
        return max(reach * 2.2, 0.9)

    def render_frame(self, joints: np.ndarray, gripper: float | None = None) -> np.ndarray:
        self.data.qpos[:] = self.model.qpos0
        self.data.qpos[self.qpos_adr] = joints
        if gripper is not None and self.gripper_adr.size:
            self.data.qpos[self.gripper_adr] = gripper
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera=self.camera)
        return self.renderer.render()  # RGB

    def render_trajectory(self, trajectory: RobotTrajectory) -> list[np.ndarray]:
        return [
            self.render_frame(trajectory.joints[i], float(trajectory.gripper[i]))
            for i in range(len(trajectory))
        ]

    def close(self) -> None:
        if getattr(self, "renderer", None) is not None:
            self.renderer.close()
            self.renderer = None

    def __enter__(self) -> RobotRenderer:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def render_robot_video(
    trajectory: RobotTrajectory,
    embodiment: Embodiment,
    out_path: str | Path,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    **camera_kwargs,
) -> Path:
    with RobotRenderer(embodiment, width=width, height=height, **camera_kwargs) as renderer:
        frames = renderer.render_trajectory(trajectory)
    bgr = [frame[..., ::-1] for frame in frames]
    return write_video(bgr, out_path, fps=trajectory.fps)
