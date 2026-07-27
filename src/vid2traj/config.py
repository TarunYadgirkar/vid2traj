"""Embodiment configuration.

Adding a robot means adding a YAML file under `configs/embodiments/`. Nothing
in the retargeting, safety, or export layers may branch on the robot's name.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from .math3d import make_transform

PACKAGE_ROOT = Path(__file__).resolve().parent
CONFIG_DIR = PACKAGE_ROOT.parent.parent / "configs" / "embodiments"


@dataclass(frozen=True)
class GripperSpec:
    joints: list[str]
    limits: tuple[float, float]
    open_aperture: float
    closed_aperture: float

    def from_hand_aperture(self, normalized: np.ndarray) -> np.ndarray:
        """Map a normalized hand opening (0 closed .. 1 open) to a joint command."""
        frac = np.clip(np.asarray(normalized, dtype=float), 0.0, 1.0)
        return self.closed_aperture + frac * (self.open_aperture - self.closed_aperture)


@dataclass(frozen=True)
class IKParams:
    damping: float = 0.05
    max_iters: int = 60
    tol_position: float = 1e-3
    tol_orientation: float = 1e-2
    null_space_gain: float = 0.02
    step_limit: float = 0.2


@dataclass(frozen=True)
class Embodiment:
    name: str
    robot_type: str
    model_path: Path
    ee_site: str
    arm_joints: list[str]
    home_joints: np.ndarray
    joint_limits: np.ndarray
    velocity_limits: np.ndarray
    acceleration_limits: np.ndarray
    gripper: GripperSpec
    wrist_to_ee: np.ndarray
    position_only: bool = False
    orientation_weight: float = 1.0
    ik: IKParams = field(default_factory=IKParams)
    collision_margin: float = 0.0
    source_path: Path | None = None

    @property
    def n_joints(self) -> int:
        return len(self.arm_joints)


@functools.lru_cache(maxsize=8)
def _load_model(path_str: str):
    import mujoco

    return mujoco.MjModel.from_xml_path(path_str)


def load_mujoco_model(embodiment: Embodiment):
    """Shared, cached MjModel. MjData is always created fresh by callers."""
    return _load_model(str(embodiment.model_path))


def list_embodiments() -> list[str]:
    return sorted(p.stem for p in CONFIG_DIR.glob("*.yaml"))


def _resolve_config(name_or_path: str | Path) -> Path:
    path = Path(name_or_path)
    if path.suffix in {".yaml", ".yml"}:
        if not path.exists():
            raise FileNotFoundError(f"embodiment config not found: {path}")
        return path
    candidate = CONFIG_DIR / f"{name_or_path}.yaml"
    if not candidate.exists():
        raise KeyError(
            f"unknown embodiment {name_or_path!r}; available: {', '.join(list_embodiments())}"
        )
    return candidate


def load_embodiment(name_or_path: str | Path | Embodiment) -> Embodiment:
    if isinstance(name_or_path, Embodiment):
        return name_or_path

    config_path = _resolve_config(name_or_path)
    raw = yaml.safe_load(config_path.read_text())

    model_path = (PACKAGE_ROOT / raw["model"]["path"]).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"model file missing for {raw['name']}: {model_path}")

    arm_joints = list(raw["arm_joints"])
    limits_cfg = raw.get("limits", {})
    joint_limits = _joint_limits_from_model(
        str(model_path), arm_joints, float(limits_cfg.get("joint_margin", 0.0))
    )

    home = np.asarray(raw["home"], dtype=float)
    if home.shape != (len(arm_joints),):
        raise ValueError(f"{raw['name']}: home has {home.shape} entries, expected {len(arm_joints)}")
    home = np.clip(home, joint_limits[:, 0], joint_limits[:, 1])

    scale = float(limits_cfg.get("velocity_scale", 1.0))
    velocity = np.asarray(limits_cfg["velocity"], dtype=float) * scale
    acceleration = np.asarray(limits_cfg["acceleration"], dtype=float) * scale

    grip_cfg = raw.get("gripper", {})
    gripper = GripperSpec(
        joints=list(grip_cfg.get("joints", [])),
        limits=tuple(float(v) for v in grip_cfg.get("limits", (0.0, 1.0))),
        open_aperture=float(grip_cfg.get("open", grip_cfg.get("limits", (0.0, 1.0))[1])),
        closed_aperture=float(grip_cfg.get("closed", grip_cfg.get("limits", (0.0, 1.0))[0])),
    )

    retarget_cfg = raw.get("retarget", {})
    offset_cfg = retarget_cfg.get("wrist_to_ee", {})
    wrist_to_ee = make_transform(
        offset_cfg.get("position", [0.0, 0.0, 0.0]),
        offset_cfg.get("quaternion", [1.0, 0.0, 0.0, 0.0]),
    )

    return Embodiment(
        name=raw["name"],
        robot_type=raw.get("robot_type", raw["name"]),
        model_path=model_path,
        ee_site=raw["model"]["ee_site"],
        arm_joints=arm_joints,
        home_joints=home,
        joint_limits=joint_limits,
        velocity_limits=velocity,
        acceleration_limits=acceleration,
        gripper=gripper,
        wrist_to_ee=wrist_to_ee,
        position_only=bool(retarget_cfg.get("position_only", False)),
        orientation_weight=float(retarget_cfg.get("orientation_weight", 1.0)),
        ik=IKParams(**raw.get("ik", {})),
        collision_margin=float(raw.get("safety", {}).get("collision_margin", 0.0)),
        source_path=config_path,
    )


def _joint_limits_from_model(model_path: str, joints: list[str], margin: float) -> np.ndarray:
    """Take limits from the model itself so config and physics cannot disagree."""
    import mujoco

    model = _load_model(model_path)
    limits = np.zeros((len(joints), 2))
    for i, name in enumerate(joints):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"joint {name!r} not found in {model_path}")
        if not model.jnt_limited[jid]:
            limits[i] = (-np.pi, np.pi)
        else:
            limits[i] = model.jnt_range[jid]
    limits[:, 0] += margin
    limits[:, 1] -= margin
    if np.any(limits[:, 0] >= limits[:, 1]):
        raise ValueError(f"joint_margin {margin} collapses a joint range in {model_path}")
    return limits
