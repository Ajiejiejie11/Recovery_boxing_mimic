"""Evaluate the joint recovery/tracking Z1 policy in MuJoCo.

This is an evaluation script rather than a reference-motion player.  It mirrors
the deployment state machine described in
``docs/boxing_recovery_training_and_rb_deployment.md``:

* RECOVERY sends frame 64 of the get-ready clip, freezes its joint velocity,
  and aligns the target yaw with the robot's current torso yaw.
* TRACKING sends a normal boxing reference and advances it at the source FPS.
* ``combined`` starts from a captured fallen pose, switches to tracking without
  resetting MuJoCo after a successful get-up, and reports both sets of metrics.

The actor observation is exactly 127 values:

    command(46) | motion_anchor_ori_b(6) | base_ang_vel(3) |
    joint_pos_rel(23) | joint_vel(23) | last_action(23) |
    projected_gravity(3)

Unlike the older playback script, all NPZ joints and bodies are mapped by name.
This matters for the current files, where ``torso_link`` is not body index 3.

Examples:

    # Visual combined test using the latest policy in this repository.
    python scripts/evaluate_recovery_tracking_mujoco.py --mode combined

    # Headless recovery statistics over 20 captured fallen poses.
    python scripts/evaluate_recovery_tracking_mujoco.py \
        --mode recovery --trials 20 --no-render --no-real-time

    # Tracking-only test for one boxing clip, with a velocity impulse at 2 s.
    python scripts/evaluate_recovery_tracking_mujoco.py \
        --mode tracking --motion path/to/boxing_motion.npz \
        --push-time 2.0 --push-linear 0.8,0,0
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import select
import signal
import sys
import termios
import time
import tty
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ModuleNotFoundError as import_error:  # Keep --help and the error message usable.
    _IMPORT_ERROR: ModuleNotFoundError | None = import_error
else:
    try:
        import mujoco
        import mujoco.viewer
        import onnxruntime as ort
    except ModuleNotFoundError as import_error:
        _IMPORT_ERROR = import_error
    else:
        _IMPORT_ERROR = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LATEST_RUN = PROJECT_ROOT / (
    "logs/rsl_rl/z1_flat/"
    "2026-07-31_20-55-24_boxing_recovery_height35_refbridge_feet01_resume_6000"
)
DEFAULT_POLICY = LATEST_RUN / (
    "exported/2026-07-31_20-55-24_boxing_recovery_height35_refbridge_feet01_resume_6000.onnx"
)
DEFAULT_XML = PROJECT_ROOT / (
    "source/whole_body_tracking/whole_body_tracking/assets/"
    "magicbot-z1_description/mjcf/MagicBotZ1_23dof.xml"
)
DEFAULT_MOTIONS = PROJECT_ROOT / (
    "source/whole_body_tracking/whole_body_tracking/datasets/"
    "boxing-dataset-magiclab-z1/train_npz"
)
DEFAULT_RECOVERY_TARGET = PROJECT_ROOT / (
    "source/whole_body_tracking/whole_body_tracking/datasets/"
    "recovery_targets/boxing_walk_001_get_ready_370_530.npz"
)
DEFAULT_FALLS = PROJECT_ROOT / (
    "source/whole_body_tracking/whole_body_tracking/datasets/fall_recovety/"
    "prepare_stand_slice_06/train_npz"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs/mujoco_recovery_tracking"

# Stable keyboard mapping for the eight clipped boxing references. Patterns are
# used instead of complete filenames so regenerated clips may change their
# numeric frame suffix without changing the operator controls.
REFERENCE_KEY_PRESETS: dict[int, tuple[str, str]] = {
    1: ("嘲讽", "boxing_chaofeng_005_*_clip.npz"),
    2: ("上勾拳", "boxing_shanggouquan_001_*_clip.npz"),
    3: ("膝踢", "boxing_xiti_001_*_clip.npz"),
    4: ("叶问盾", "boxing_yewendun_002_*_clip.npz"),
    5: ("右侧踢", "boxing_youceti_001_*_clip.npz"),
    6: ("直拳刺探", "boxing_zhiquancitan_*_clip.npz"),
    7: ("组合拳", "boxing_zuhequan_003_[0-9]*_clip.npz"),
    8: ("组合拳 A", "boxing_zuhequan_003_A_*_clip.npz"),
}


def csv_values(value: str, dtype: type = str) -> list[Any]:
    return [dtype(item.strip()) for item in value.split(",") if item.strip()]


def vec3(value: str) -> tuple[float, float, float]:
    values = csv_values(value, float)
    if len(values) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated values, for example 0.8,0,0")
    return float(values[0]), float(values[1]), float(values[2])


def normalize_quat(q: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(q))
    if norm < 1.0e-8:
        raise ValueError("zero-length quaternion")
    return np.asarray(q, dtype=np.float64) / norm


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product for scalar-first quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_inverse(q: np.ndarray) -> np.ndarray:
    q = normalize_quat(q)
    return quat_conjugate(q)


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = normalize_quat(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def quat_rotate(q: np.ndarray, vector: np.ndarray) -> np.ndarray:
    return quat_to_rotmat(q) @ vector


def quat_rotate_inverse(q: np.ndarray, vector: np.ndarray) -> np.ndarray:
    return quat_to_rotmat(q).T @ vector


def quat_distance(q1: np.ndarray, q2: np.ndarray) -> float:
    relative = normalize_quat(quat_mul(quat_inverse(q1), normalize_quat(q2)))
    return float(2.0 * math.atan2(np.linalg.norm(relative[1:]), abs(relative[0])))


def yaw_from_quat(q: np.ndarray) -> float:
    w, x, y, z = normalize_quat(q)
    return float(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def yaw_quat(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)], dtype=np.float64)


def yaw_component(q: np.ndarray) -> np.ndarray:
    return yaw_quat(yaw_from_quat(q))


def rotmat_to_6d(rotation: np.ndarray) -> np.ndarray:
    # Matches Isaac Lab: matrix_from_quat(q)[..., :2].reshape(...).
    return rotation[:, :2].reshape(-1).astype(np.float32)


class NamedMotion:
    """A copied, validated NPZ motion with name-based indexing."""

    REQUIRED = (
        "fps",
        "joint_names",
        "body_names",
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
    )

    def __init__(self, path: Path):
        self.path = path.resolve()
        with np.load(self.path, allow_pickle=False) as data:
            missing = [key for key in self.REQUIRED if key not in data]
            if missing:
                raise ValueError(f"{self.path} is missing required fields: {missing}")
            self.fps = int(np.asarray(data["fps"]).item())
            self.joint_names = np.asarray(data["joint_names"]).astype(str).tolist()
            self.body_names = np.asarray(data["body_names"]).astype(str).tolist()
            self.joint_pos = np.asarray(data["joint_pos"], dtype=np.float32).copy()
            self.joint_vel = np.asarray(data["joint_vel"], dtype=np.float32).copy()
            self.body_pos_w = np.asarray(data["body_pos_w"], dtype=np.float32).copy()
            self.body_quat_w = np.asarray(data["body_quat_w"], dtype=np.float32).copy()
            self.body_lin_vel_w = (
                np.asarray(data["body_lin_vel_w"], dtype=np.float32).copy()
                if "body_lin_vel_w" in data
                else None
            )
            self.body_ang_vel_w = (
                np.asarray(data["body_ang_vel_w"], dtype=np.float32).copy()
                if "body_ang_vel_w" in data
                else None
            )

        if self.fps <= 0 or self.joint_pos.ndim != 2:
            raise ValueError(f"invalid motion data in {self.path}")
        self.frames = int(self.joint_pos.shape[0])
        if self.joint_vel.shape != self.joint_pos.shape:
            raise ValueError(f"joint_pos/joint_vel shape mismatch in {self.path}")
        if self.body_pos_w.shape != (self.frames, len(self.body_names), 3):
            raise ValueError(f"body_pos_w shape does not match body_names in {self.path}")
        if self.body_quat_w.shape != (self.frames, len(self.body_names), 4):
            raise ValueError(f"body_quat_w shape does not match body_names in {self.path}")

    def joint_indices(self, names: list[str]) -> np.ndarray:
        missing = [name for name in names if name not in self.joint_names]
        if missing:
            raise ValueError(f"{self.path.name} lacks policy joints: {missing}")
        return np.asarray([self.joint_names.index(name) for name in names], dtype=np.int32)

    def body_index(self, name: str) -> int:
        if name not in self.body_names:
            raise ValueError(f"{self.path.name} lacks body {name!r}")
        return self.body_names.index(name)

    def frame(self, elapsed_s: float, start_frame: int, loop: bool) -> int | None:
        frame = start_frame + int(math.floor(elapsed_s * self.fps + 1.0e-9))
        if loop:
            return frame % self.frames
        return frame if frame < self.frames else None


@dataclass(frozen=True)
class FallPose:
    motion: NamedMotion
    frame: int

    @property
    def label(self) -> str:
        return f"{self.motion.path.name}:{self.frame}"


class FallCatalog:
    """The same height-filtered captured fall distribution used during training."""

    def __init__(self, source: Path, torso_name: str, max_height: float, max_upright: float):
        files = sorted(source.glob("*.npz")) if source.is_dir() else [source]
        if not files or not all(path.is_file() for path in files):
            raise FileNotFoundError(f"no recovery reset NPZ files found at {source}")
        self.motions = [NamedMotion(path) for path in files]
        self.valid: list[tuple[int, int]] = []
        self.valid_by_motion: list[list[int]] = [[] for _ in self.motions]
        for motion_id, motion in enumerate(self.motions):
            torso_idx = motion.body_index(torso_name)
            ground_z = motion.body_pos_w[:, :, 2].min(axis=1)
            height = motion.body_pos_w[:, torso_idx, 2] - ground_z
            quat = motion.body_quat_w[:, torso_idx]
            upright = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
            for frame in np.flatnonzero((height <= max_height) & (upright <= max_upright)):
                self.valid.append((motion_id, int(frame)))
                self.valid_by_motion[motion_id].append(int(frame))
        if not self.valid:
            raise ValueError("no captured fall frame passed the configured recovery reset filter")

    def sample(self, rng: np.random.Generator) -> FallPose:
        motion_id, frame = self.valid[int(rng.integers(len(self.valid)))]
        return FallPose(self.motions[motion_id], frame)

    def sample_from_motion(self, rng: np.random.Generator, motion_id: int) -> FallPose:
        motion_id %= len(self.motions)
        frames = self.valid_by_motion[motion_id]
        if not frames:
            raise ValueError(f"{self.motions[motion_id].path.name} has no eligible recovery reset frames")
        frame = frames[int(rng.integers(len(frames)))]
        return FallPose(self.motions[motion_id], frame)

    def exact(self, file_name: str, frame: int) -> FallPose:
        matches = [motion for motion in self.motions if motion.path.name == Path(file_name).name]
        if len(matches) != 1:
            raise ValueError(f"fall file {file_name!r} was not found uniquely in the fall source")
        if not 0 <= frame < matches[0].frames:
            raise ValueError(f"fall frame {frame} is outside [0, {matches[0].frames})")
        return FallPose(matches[0], frame)


def expand_motion_paths(source: Path, pattern: str) -> list[Path]:
    paths = sorted(source.glob(pattern)) if source.is_dir() else [source]
    if not paths or not all(path.is_file() for path in paths):
        raise FileNotFoundError(
            f"no tracking motion files matching {pattern!r} were found at {source}"
        )
    return paths


def resolve_reference_key_map(source: Path) -> dict[int, Path]:
    if not source.is_dir():
        raise ValueError("keyboard reference selection requires --motion to be a directory")
    result: dict[int, Path] = {}
    for key, (label, pattern) in REFERENCE_KEY_PRESETS.items():
        matches = sorted(source.glob(pattern))
        if len(matches) != 1:
            names = [path.name for path in matches]
            raise ValueError(
                f"reference key {key} ({label}) pattern {pattern!r} matched {len(matches)} files: {names}"
            )
        result[key] = matches[0]
    return result


def print_reference_key_map(reference_paths: dict[int, Path]) -> None:
    print("[REFERENCE KEYS] 终端直接按 1-8（无需回车），或在 MuJoCo viewer 按数字键：")
    for key, path in reference_paths.items():
        label = REFERENCE_KEY_PRESETS[key][0]
        print(f"  {key}: {label:<8} -> {path.name}")


def effort_limit_for_joint(name: str) -> float:
    """Isaac Lab effort_limit_sim from the saved training configuration."""
    if "ankle_roll" in name:
        return 20.0
    if "ankle_pitch" in name:
        return 80.0
    if any(part in name for part in ("shoulder", "elbow", "wrist")):
        return 50.0
    return 120.0


class TerminalDigitReader:
    """Non-blocking single-key input while the MuJoCo loop keeps stepping.

    A real terminal is temporarily placed in cbreak mode, so digits become
    available immediately without Enter.  The original terminal settings are
    restored even when the trial raises or is interrupted.
    """

    def __init__(self, enabled: bool):
        self.enabled = enabled and sys.stdin.isatty()
        self.fd: int | None = None
        self.original_settings: list[Any] | None = None

    def __enter__(self) -> "TerminalDigitReader":
        if not self.enabled:
            return self
        try:
            self.fd = sys.stdin.fileno()
            self.original_settings = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        except (OSError, termios.error):
            self.enabled = False
            self.fd = None
            self.original_settings = None
        return self

    def poll(self) -> list[int]:
        if not self.enabled or self.fd is None:
            return []
        digits: list[int] = []
        while select.select([self.fd], [], [], 0.0)[0]:
            chunk = os.read(self.fd, 64)
            if not chunk:
                break
            digits.extend(byte - ord("0") for byte in chunk if ord("1") <= byte <= ord("8"))
        return digits

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if self.fd is not None and self.original_settings is not None:
            try:
                termios.tcsetattr(self.fd, termios.TCSADRAIN, self.original_settings)
            except (OSError, termios.error):
                pass


class RecoveryTrackingEvaluator:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.model = mujoco.MjModel.from_xml_path(str(args.xml))
        self.data = mujoco.MjData(self.model)
        if args.sim_dt is not None:
            self.model.opt.timestep = args.sim_dt
        self.sim_dt = float(self.model.opt.timestep)
        self.decimation = max(1, round(args.policy_dt / self.sim_dt))
        effective_dt = self.decimation * self.sim_dt
        if not math.isclose(effective_dt, args.policy_dt, abs_tol=1.0e-9):
            raise ValueError(
                f"policy_dt={args.policy_dt} is not an integer multiple of sim_dt={self.sim_dt}; "
                f"nearest effective dt is {effective_dt}"
            )
        self.policy_dt = effective_dt

        session_options = ort.SessionOptions()
        session_options.log_severity_level = 3
        self.session = ort.InferenceSession(
            str(args.policy), sess_options=session_options, providers=["CPUExecutionProvider"]
        )
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) < 1:
            raise ValueError("expected a single-input ONNX actor")
        self.input_name = inputs[0].name
        self.output_name = outputs[0].name
        input_shape = inputs[0].shape
        if input_shape[-1] != 127:
            raise ValueError(f"this evaluator requires a 127-D actor, ONNX input is {input_shape}")
        metadata = self.session.get_modelmeta().custom_metadata_map
        required_metadata = (
            "joint_names",
            "joint_stiffness",
            "joint_damping",
            "default_joint_pos",
            "action_scale",
            "anchor_body_name",
        )
        missing = [key for key in required_metadata if key not in metadata]
        if missing:
            raise ValueError(f"ONNX is missing deployment metadata: {missing}")

        self.policy_joint_names = csv_values(metadata["joint_names"])
        self.kp = np.asarray(csv_values(metadata["joint_stiffness"], float), dtype=np.float32)
        self.kd = np.asarray(csv_values(metadata["joint_damping"], float), dtype=np.float32)
        self.default_q = np.asarray(csv_values(metadata["default_joint_pos"], float), dtype=np.float32)
        self.action_scale = np.asarray(csv_values(metadata["action_scale"], float), dtype=np.float32)
        self.anchor_name = metadata["anchor_body_name"].strip()
        self.metric_body_names = csv_values(metadata.get("body_names", ""))
        lengths = {
            len(self.policy_joint_names),
            len(self.kp),
            len(self.kd),
            len(self.default_q),
            len(self.action_scale),
        }
        if lengths != {23}:
            raise ValueError(f"policy metadata must describe 23 joints, got lengths {sorted(lengths)}")

        self.joint_ids = self._ids(mujoco.mjtObj.mjOBJ_JOINT, self.policy_joint_names)
        self.qpos_addrs = np.asarray([self.model.jnt_qposadr[jid] for jid in self.joint_ids], dtype=np.int32)
        self.qvel_addrs = np.asarray([self.model.jnt_dofadr[jid] for jid in self.joint_ids], dtype=np.int32)
        self.actuator_ids = np.asarray([self._actuator_for_joint(jid) for jid in self.joint_ids], dtype=np.int32)
        self.anchor_bid = self._id(mujoco.mjtObj.mjOBJ_BODY, self.anchor_name)
        self.pelvis_bid = self._id(mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.left_foot_bid = self._id(mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
        self.right_foot_bid = self._id(mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")
        self.floor_gid = self._id(mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.effort_limits = np.asarray(
            [effort_limit_for_joint(name) for name in self.policy_joint_names], dtype=np.float32
        )
        self.last_action = np.zeros(23, dtype=np.float32)
        self.target_q = self.default_q.copy()
        self.viewer = None
        self.reset_height_correction = 0.0
        self._viewer_phase = "IDLE"
        self._pending_reference_key: int | None = None

    def _queue_reference_key(self, reference_key: int, source: str) -> None:
        """Accept a viewer/terminal digit only after recovery is complete."""
        if self._viewer_phase != "WAITING":
            print(
                f"[KEY:{source}] ignored reference {reference_key}: current phase is "
                f"{self._viewer_phase}; wait for the WAITING prompt after stable recovery."
            )
            return
        self._pending_reference_key = reference_key
        label = REFERENCE_KEY_PRESETS[reference_key][0]
        print(f"[KEY:{source}] selected reference {reference_key}: {label}")

    def _viewer_key_callback(self, keycode: int) -> None:
        """Translate GLFW main/keypad digits into a reference selection."""
        reference_key: int | None = None
        if ord("1") <= keycode <= ord("8"):
            reference_key = keycode - ord("0")
        # GLFW keypad keys KP_1 ... KP_8 have integer values 321 ... 328.
        elif 321 <= keycode <= 328:
            reference_key = keycode - 320
        if reference_key is None:
            return
        self._queue_reference_key(reference_key, "viewer")

    def _id(self, object_type: Any, name: str) -> int:
        object_id = mujoco.mj_name2id(self.model, object_type, name)
        if object_id < 0:
            raise ValueError(f"MuJoCo model does not contain {name!r}")
        return int(object_id)

    def _ids(self, object_type: Any, names: list[str]) -> np.ndarray:
        return np.asarray([self._id(object_type, name) for name in names], dtype=np.int32)

    def _actuator_for_joint(self, joint_id: int) -> int:
        matches = np.flatnonzero(self.model.actuator_trnid[:, 0] == joint_id)
        if len(matches) != 1:
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            raise ValueError(f"expected exactly one actuator for joint {name!r}, found {len(matches)}")
        return int(matches[0])

    def _set_policy_joints(self, motion: NamedMotion, frame: int) -> None:
        indexes = motion.joint_indices(self.policy_joint_names)
        self.data.qpos[self.qpos_addrs] = motion.joint_pos[frame, indexes]

    def _resolve_reset_floor_penetration(self) -> float:
        """Lift the free base just enough to remove MJCF mesh/floor overlap.

        Training reset height is reconstructed from NPZ body origins, while
        MuJoCo contact is generated from the actual mesh surfaces.  Their lowest
        points need not coincide.  Leaving a deep initial overlap lets the
        contact solver inject a large artificial recovery impulse.
        """
        mujoco.mj_forward(self.model, self.data)
        deepest = 0.0
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            if self.floor_gid in (int(contact.geom1), int(contact.geom2)) and contact.dist < 0.0:
                deepest = max(deepest, float(-contact.dist))
        if deepest > 0.0:
            correction = deepest + self.args.reset_contact_clearance
            self.data.qpos[2] += correction
            mujoco.mj_forward(self.model, self.data)
            return correction
        return 0.0

    def _reset_tracking(self, motion: NamedMotion, frame: int) -> None:
        if not 0 <= frame < motion.frames:
            raise ValueError(f"tracking start frame {frame} is outside [0, {motion.frames})")
        mujoco.mj_resetData(self.model, self.data)
        self._set_policy_joints(motion, frame)
        pelvis_idx = motion.body_index("pelvis")
        self.data.qpos[:3] = motion.body_pos_w[frame, pelvis_idx]
        root_quat = normalize_quat(motion.body_quat_w[frame, pelvis_idx])
        self.data.qpos[3:7] = root_quat
        self.data.qvel[:] = 0.0
        joint_indexes = motion.joint_indices(self.policy_joint_names)
        self.data.qvel[self.qvel_addrs] = motion.joint_vel[frame, joint_indexes]
        if motion.body_lin_vel_w is not None:
            self.data.qvel[:3] = motion.body_lin_vel_w[frame, pelvis_idx]
        if motion.body_ang_vel_w is not None:
            # NPZ stores world angular velocity; MuJoCo free-joint rotational
            # qvel is expressed in the floating body's local frame.
            self.data.qvel[3:6] = quat_rotate_inverse(
                root_quat, motion.body_ang_vel_w[frame, pelvis_idx]
            )
        mujoco.mj_forward(self.model, self.data)
        self.reset_height_correction = self._resolve_reset_floor_penetration()
        self.last_action.fill(0.0)
        self.target_q = self.default_q.copy()

    def _reset_fallen(self, pose: FallPose) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self._set_policy_joints(pose.motion, pose.frame)
        pelvis_idx = pose.motion.body_index("pelvis")
        root_pos = pose.motion.body_pos_w[pose.frame, pelvis_idx].astype(np.float64).copy()
        source_ground_z = float(pose.motion.body_pos_w[pose.frame, :, 2].min())
        root_pos[:2] = 0.0
        root_pos[2] = root_pos[2] - source_ground_z + self.args.ground_clearance
        self.data.qpos[:3] = root_pos
        self.data.qpos[3:7] = normalize_quat(pose.motion.body_quat_w[pose.frame, pelvis_idx])
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.reset_height_correction = self._resolve_reset_floor_penetration()
        self.last_action.fill(0.0)
        self.target_q = self.default_q.copy()

    def _reference(
        self,
        phase: str,
        tracking_motion: NamedMotion,
        tracking_frame: int,
        recovery_target: NamedMotion,
        tracking_yaw_offset: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if phase in ("RECOVERY", "RETURN_TO_STAND", "WAITING"):
            indexes = recovery_target.joint_indices(self.policy_joint_names)
            joint_pos = recovery_target.joint_pos[self.args.recovery_target_frame, indexes]
            joint_vel = np.zeros(23, dtype=np.float32)
            robot_anchor_q = self.data.xquat[self.anchor_bid]
            anchor_quat = yaw_quat(yaw_from_quat(robot_anchor_q))
        else:
            indexes = tracking_motion.joint_indices(self.policy_joint_names)
            joint_pos = tracking_motion.joint_pos[tracking_frame, indexes]
            joint_vel = tracking_motion.joint_vel[tracking_frame, indexes]
            anchor_idx = tracking_motion.body_index(self.anchor_name)
            anchor_quat = quat_mul(
                tracking_yaw_offset,
                tracking_motion.body_quat_w[tracking_frame, anchor_idx],
            )
        return joint_pos.astype(np.float32), joint_vel.astype(np.float32), normalize_quat(anchor_quat)

    def build_obs(
        self,
        phase: str,
        tracking_motion: NamedMotion,
        tracking_frame: int,
        recovery_target: NamedMotion,
        tracking_yaw_offset: np.ndarray,
    ) -> np.ndarray:
        reference_q, reference_qd, reference_anchor_q = self._reference(
            phase, tracking_motion, tracking_frame, recovery_target, tracking_yaw_offset
        )
        q = self.data.qpos[self.qpos_addrs].astype(np.float32)
        qd = self.data.qvel[self.qvel_addrs].astype(np.float32)
        robot_anchor_q = normalize_quat(self.data.xquat[self.anchor_bid])
        relative_q = normalize_quat(quat_mul(quat_inverse(robot_anchor_q), reference_anchor_q))
        anchor_orientation_6d = rotmat_to_6d(quat_to_rotmat(relative_q))

        # MuJoCo free-joint rotational qvel is expressed in the child body frame,
        # matching Isaac Lab's base_ang_vel observation.
        base_ang_vel_b = self.data.qvel[3:6].astype(np.float32)
        base_quat = normalize_quat(self.data.xquat[self.pelvis_bid])
        projected_gravity = quat_rotate_inverse(base_quat, np.array([0.0, 0.0, -1.0])).astype(np.float32)
        command = np.concatenate((reference_q, reference_qd))
        obs = np.concatenate(
            (
                command,
                anchor_orientation_6d,
                base_ang_vel_b,
                q - self.default_q,
                qd,
                self.last_action,
                projected_gravity,
            )
        ).astype(np.float32)
        if obs.shape != (127,) or not np.isfinite(obs).all():
            raise RuntimeError(f"invalid actor observation: shape={obs.shape}, finite={np.isfinite(obs).all()}")
        return obs

    def infer(self, obs: np.ndarray) -> np.ndarray:
        output = self.session.run([self.output_name], {self.input_name: obs.reshape(1, 127)})[0]
        action = np.asarray(output, dtype=np.float32).reshape(-1)
        if action.shape != (23,) or not np.isfinite(action).all():
            raise RuntimeError(f"invalid policy action: shape={action.shape}")
        if self.args.action_clip is not None:
            action = np.clip(action, -self.args.action_clip, self.args.action_clip)
        self.last_action = action
        self.target_q = self.default_q + self.action_scale * action
        return action

    def apply_pd(self) -> np.ndarray:
        q = self.data.qpos[self.qpos_addrs].astype(np.float32)
        qd = self.data.qvel[self.qvel_addrs].astype(np.float32)
        torque = self.kp * (self.target_q - q) - self.kd * qd
        torque *= self.args.torque_scale
        if not self.args.no_torque_limits:
            torque = np.clip(torque, -self.effort_limits, self.effort_limits)
        if self.args.torque_clip is not None:
            torque = np.clip(torque, -self.args.torque_clip, self.args.torque_clip)
        self.data.ctrl[:] = 0.0
        self.data.ctrl[self.actuator_ids] = torque
        return torque

    def step_physics(self) -> tuple[np.ndarray, float]:
        max_penetration = 0.0
        torque = np.zeros(23, dtype=np.float32)
        for _ in range(self.decimation):
            torque = self.apply_pd()
            mujoco.mj_step(self.model, self.data)
            for contact_index in range(self.data.ncon):
                contact = self.data.contact[contact_index]
                if self.floor_gid in (int(contact.geom1), int(contact.geom2)) and contact.dist < 0.0:
                    max_penetration = max(max_penetration, float(-contact.dist))
            if self.viewer is not None:
                self.viewer.sync()
        return torque, max_penetration

    def _foot_contact(self, body_id: int) -> bool:
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            if self.floor_gid not in (geom1, geom2):
                continue
            robot_geom = geom2 if geom1 == self.floor_gid else geom1
            if int(self.model.geom_bodyid[robot_geom]) == body_id:
                return True
        return False

    def _body_planar_speed(self, body_id: int) -> float:
        velocity = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, body_id, velocity, 0
        )
        return float(np.linalg.norm(velocity[3:5]))

    def physical_state(self) -> dict[str, float | bool]:
        anchor_q = normalize_quat(self.data.xquat[self.anchor_bid])
        upright = float(quat_to_rotmat(anchor_q)[2, 2])
        return {
            "torso_height": float(self.data.xpos[self.anchor_bid, 2]),
            "uprightness": upright,
            "left_contact": self._foot_contact(self.left_foot_bid),
            "right_contact": self._foot_contact(self.right_foot_bid),
            "left_foot_speed": self._body_planar_speed(self.left_foot_bid),
            "right_foot_speed": self._body_planar_speed(self.right_foot_bid),
        }

    def tracking_metrics(
        self, motion: NamedMotion, frame: int, tracking_yaw_offset: np.ndarray
    ) -> dict[str, float]:
        joint_indexes = motion.joint_indices(self.policy_joint_names)
        q = self.data.qpos[self.qpos_addrs]
        qd = self.data.qvel[self.qvel_addrs]
        reference_q = motion.joint_pos[frame, joint_indexes]
        reference_qd = motion.joint_vel[frame, joint_indexes]

        anchor_idx = motion.body_index(self.anchor_name)
        robot_anchor_pos = self.data.xpos[self.anchor_bid].copy()
        robot_anchor_quat = normalize_quat(self.data.xquat[self.anchor_bid])
        reference_anchor_pos = motion.body_pos_w[frame, anchor_idx].astype(np.float64)
        reference_anchor_quat = normalize_quat(
            quat_mul(tracking_yaw_offset, motion.body_quat_w[frame, anchor_idx])
        )

        # Same horizontal/yaw alignment used by MotionCommand._update_reference_alignment.
        delta_position = robot_anchor_pos.copy()
        delta_position[2] = reference_anchor_pos[2]
        delta_yaw = yaw_component(quat_mul(robot_anchor_quat, quat_inverse(reference_anchor_quat)))

        position_errors: list[float] = []
        orientation_errors: list[float] = []
        for body_name in self.metric_body_names:
            if body_name not in motion.body_names:
                continue
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id < 0:
                continue
            body_index = motion.body_names.index(body_name)
            relative_position = quat_rotate(
                tracking_yaw_offset,
                motion.body_pos_w[frame, body_index] - reference_anchor_pos,
            )
            target_position = delta_position + quat_rotate(delta_yaw, relative_position)
            aligned_body_quat = quat_mul(
                tracking_yaw_offset, motion.body_quat_w[frame, body_index]
            )
            target_quat = normalize_quat(quat_mul(delta_yaw, aligned_body_quat))
            position_errors.append(float(np.linalg.norm(target_position - self.data.xpos[body_id])))
            orientation_errors.append(quat_distance(target_quat, self.data.xquat[body_id]))

        return {
            "joint_pos_rmse": float(np.sqrt(np.mean((q - reference_q) ** 2))),
            "joint_vel_rmse": float(np.sqrt(np.mean((qd - reference_qd) ** 2))),
            "anchor_height_error": abs(float(robot_anchor_pos[2] - reference_anchor_pos[2])),
            "anchor_ori_error": quat_distance(reference_anchor_quat, robot_anchor_quat),
            "body_pos_error": float(np.mean(position_errors)) if position_errors else math.nan,
            "body_ori_error": float(np.mean(orientation_errors)) if orientation_errors else math.nan,
        }

    def tracking_yaw_offset(self, motion: NamedMotion, frame: int) -> np.ndarray:
        """Rotate a fresh tracking reference so its starting yaw matches the robot."""
        anchor_idx = motion.body_index(self.anchor_name)
        source_yaw = yaw_from_quat(motion.body_quat_w[frame, anchor_idx])
        robot_yaw = yaw_from_quat(self.data.xquat[self.anchor_bid])
        return yaw_quat(robot_yaw - source_yaw)

    def _apply_push(self) -> None:
        self.data.qvel[:3] += np.asarray(self.args.push_linear, dtype=np.float64)
        self.data.qvel[3:6] += np.asarray(self.args.push_angular, dtype=np.float64)
        mujoco.mj_forward(self.model, self.data)

    def run_trial(
        self,
        trial_index: int,
        tracking_motion: NamedMotion,
        recovery_target: NamedMotion,
        fall_pose: FallPose | None,
        render: bool,
        reference_motions: dict[int, NamedMotion] | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        if self.args.recovery_target_frame >= recovery_target.frames:
            raise ValueError(
                f"recovery target frame {self.args.recovery_target_frame} is outside "
                f"[0, {recovery_target.frames})"
            )
        recovery_joint_indexes = recovery_target.joint_indices(self.policy_joint_names)
        standard_stand_q = recovery_target.joint_pos[
            self.args.recovery_target_frame, recovery_joint_indexes
        ]
        if self.args.mode == "tracking":
            self._reset_tracking(tracking_motion, self.args.tracking_start_frame)
            phase = "TRACKING"
        else:
            if fall_pose is None:
                raise RuntimeError("a fallen pose is required for recovery evaluation")
            self._reset_fallen(fall_pose)
            phase = "RECOVERY"

        if render:
            key_callback = self._viewer_key_callback if self.args.mode == "interactive" else None
            self.viewer = mujoco.viewer.launch_passive(
                self.model, self.data, key_callback=key_callback
            )
        else:
            self.viewer = None

        rows: list[dict[str, Any]] = []
        recovery_attempts = 1 if phase == "RECOVERY" else 0
        recovery_successes = 0
        recovery_durations: list[float] = []
        recovery_elapsed = 0.0
        tracking_elapsed = 0.0
        return_elapsed = 0.0
        waiting_elapsed = 0.0
        total_elapsed = 0.0
        left_contact_time = 0.0
        right_contact_time = 0.0
        tracking_falls = 0
        tracking_frame = self.args.tracking_start_frame
        tracking_yaw_offset = yaw_quat(0.0)
        tracking_yaw_alignments: list[float] = []
        selected_reference_key: int | None = None
        selected_reference_keys: list[int] = []
        active_reference_key: int | None = None
        completed_reference_count = 0
        interrupted_reference_count = 0
        return_durations: list[float] = []
        self._pending_reference_key = None
        self._viewer_phase = phase
        push_applied = False
        outcome = "running"
        max_penetration = 0.0
        max_abs_action = 0.0
        max_abs_torque = 0.0
        next_print_time = 0.0
        real_time_target = time.monotonic()
        interrupt_requested = False

        def request_clean_interrupt(signum: int, frame: Any) -> None:
            del signum, frame
            nonlocal interrupt_requested
            interrupt_requested = True

        terminal_reader = TerminalDigitReader(self.args.mode == "interactive")
        terminal_reader.__enter__()
        previous_sigint_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, request_clean_interrupt)
        try:
            while outcome == "running":
                if interrupt_requested:
                    outcome = "operator_interrupt"
                    print(f"\n[trial {trial_index:03d}] interrupted by operator; closing viewer cleanly")
                    break
                self._viewer_phase = phase
                if self.viewer is not None and not self.viewer.is_running():
                    outcome = "viewer_closed"
                    break

                for reference_key in terminal_reader.poll():
                    self._queue_reference_key(reference_key, "terminal")

                if phase == "WAITING" and self._pending_reference_key is not None:
                    reference_key = self._pending_reference_key
                    self._pending_reference_key = None
                    if reference_motions is None or reference_key not in reference_motions:
                        print(f"[KEY] reference {reference_key} is unavailable; still WAITING")
                    else:
                        tracking_motion = reference_motions[reference_key]
                        selected_reference_key = reference_key
                        if self.args.tracking_start_frame >= tracking_motion.frames:
                            print(
                                f"[KEY] reference {reference_key} has only {tracking_motion.frames} frames, "
                                f"but --tracking-start-frame is {self.args.tracking_start_frame}; still WAITING"
                            )
                            selected_reference_key = None
                        else:
                            selected_reference_keys.append(reference_key)
                            active_reference_key = reference_key
                            if self.args.align_tracking_yaw_after_recovery:
                                tracking_yaw_offset = self.tracking_yaw_offset(
                                    tracking_motion, self.args.tracking_start_frame
                                )
                            else:
                                tracking_yaw_offset = yaw_quat(0.0)
                            yaw_offset_rad = yaw_from_quat(tracking_yaw_offset)
                            tracking_yaw_alignments.append(yaw_offset_rad)
                            tracking_elapsed = 0.0
                            left_contact_time = 0.0
                            right_contact_time = 0.0
                            push_applied = False
                            phase = "TRACKING"
                            self._viewer_phase = phase
                            label = REFERENCE_KEY_PRESETS[reference_key][0]
                            print(
                                f"[trial {trial_index:03d}] START TRACKING key={reference_key} ({label}), "
                                f"motion={tracking_motion.path.name}, yaw_offset="
                                f"{math.degrees(yaw_offset_rad):+.1f} deg"
                            )

                if phase == "TRACKING":
                    frame_or_none = tracking_motion.frame(
                        tracking_elapsed, self.args.tracking_start_frame, self.args.tracking_loop
                    )
                    if frame_or_none is None:
                        if self.args.mode == "interactive":
                            phase = "RETURN_TO_STAND"
                            self._viewer_phase = phase
                            return_elapsed = 0.0
                            left_contact_time = 0.0
                            right_contact_time = 0.0
                            print(
                                f"[trial {trial_index:03d}] reference {active_reference_key} finished; "
                                "RETURN_TO_STAND using the standard standing target"
                            )
                        else:
                            outcome = "tracking_complete"
                            break
                    else:
                        tracking_frame = frame_or_none
                    if phase == "TRACKING" and (
                        self.args.push_time is not None
                        and not push_applied
                        and tracking_elapsed >= self.args.push_time
                    ):
                        self._apply_push()
                        push_applied = True
                        print(
                            f"[trial {trial_index:03d}] applied push at tracking t={tracking_elapsed:.2f}s: "
                            f"linear={self.args.push_linear}, angular={self.args.push_angular}"
                        )

                obs = self.build_obs(
                    phase,
                    tracking_motion,
                    tracking_frame,
                    recovery_target,
                    tracking_yaw_offset,
                )
                action = self.infer(obs)
                torque, penetration = self.step_physics()
                max_penetration = max(max_penetration, penetration)
                max_abs_action = max(max_abs_action, float(np.abs(action).max()))
                max_abs_torque = max(max_abs_torque, float(np.abs(torque).max()))
                total_elapsed += self.policy_dt
                phase_before_transition = phase
                if phase == "RECOVERY":
                    recovery_elapsed += self.policy_dt
                elif phase == "TRACKING":
                    tracking_elapsed += self.policy_dt
                elif phase == "RETURN_TO_STAND":
                    return_elapsed += self.policy_dt
                elif phase == "WAITING":
                    waiting_elapsed += self.policy_dt

                state = self.physical_state()
                left_contact_time = (
                    left_contact_time + self.policy_dt if state["left_contact"] else 0.0
                )
                right_contact_time = (
                    right_contact_time + self.policy_dt if state["right_contact"] else 0.0
                )
                feet_stable = (
                    left_contact_time >= self.args.min_contact_time
                    and right_contact_time >= self.args.min_contact_time
                    and state["left_foot_speed"] <= self.args.max_foot_speed
                    and state["right_foot_speed"] <= self.args.max_foot_speed
                )
                fallen = (
                    state["torso_height"] < self.args.fall_height
                    or state["uprightness"] < self.args.fall_uprightness
                )
                standing = (
                    state["torso_height"] >= self.args.stand_height
                    and state["uprightness"] >= self.args.stand_uprightness
                    and feet_stable
                )
                standard_stand_joint_rmse = float(
                    np.sqrt(
                        np.mean(
                            (self.data.qpos[self.qpos_addrs] - standard_stand_q) ** 2
                        )
                    )
                )

                metrics = {
                    "joint_pos_rmse": math.nan,
                    "joint_vel_rmse": math.nan,
                    "anchor_height_error": math.nan,
                    "anchor_ori_error": math.nan,
                    "body_pos_error": math.nan,
                    "body_ori_error": math.nan,
                }
                if phase_before_transition == "TRACKING":
                    metrics = self.tracking_metrics(
                        tracking_motion, tracking_frame, tracking_yaw_offset
                    )

                rows.append(
                    {
                        "trial": trial_index,
                        "sim_time_s": total_elapsed,
                        "phase": phase_before_transition,
                        "reference_frame": (
                            self.args.recovery_target_frame
                            if phase_before_transition in ("RECOVERY", "RETURN_TO_STAND", "WAITING")
                            else tracking_frame
                        ),
                        "reference_key": active_reference_key,
                        "tracking_yaw_offset_rad": yaw_from_quat(tracking_yaw_offset),
                        **state,
                        "left_contact_time_s": left_contact_time,
                        "right_contact_time_s": right_contact_time,
                        "feet_stable": feet_stable,
                        "fallen": fallen,
                        "standing": standing,
                        "standard_stand_joint_rmse_rad": standard_stand_joint_rmse,
                        "max_abs_action": float(np.abs(action).max()),
                        "max_abs_torque": float(np.abs(torque).max()),
                        **metrics,
                    }
                )

                if phase == "RECOVERY":
                    if standing:
                        recovery_successes += 1
                        recovery_durations.append(recovery_elapsed)
                        print(
                            f"[trial {trial_index:03d}] RECOVERY success in {recovery_elapsed:.2f}s "
                            f"(height={state['torso_height']:.3f}, upright={state['uprightness']:.3f})"
                        )
                        if self.args.mode == "recovery":
                            outcome = "recovery_success"
                        elif self.args.mode == "interactive":
                            phase = "WAITING"
                            self._viewer_phase = phase
                            recovery_elapsed = 0.0
                            waiting_elapsed = 0.0
                            left_contact_time = 0.0
                            right_contact_time = 0.0
                            print(
                                f"[trial {trial_index:03d}] WAITING: robot is stable. "
                                "Press 1-8 in this terminal (no Enter), or in the MuJoCo viewer, "
                                "to start a reference."
                            )
                        else:
                            if self.args.align_tracking_yaw_after_recovery:
                                tracking_yaw_offset = self.tracking_yaw_offset(
                                    tracking_motion, self.args.tracking_start_frame
                                )
                            else:
                                tracking_yaw_offset = yaw_quat(0.0)
                            yaw_offset_rad = yaw_from_quat(tracking_yaw_offset)
                            tracking_yaw_alignments.append(yaw_offset_rad)
                            print(
                                f"[trial {trial_index:03d}] TRACKING reference yaw offset "
                                f"{math.degrees(yaw_offset_rad):+.1f} deg"
                            )
                            phase = "TRACKING"
                            recovery_elapsed = 0.0
                            tracking_elapsed = 0.0
                            left_contact_time = 0.0
                            right_contact_time = 0.0
                            push_applied = False
                    elif recovery_elapsed >= self.args.recovery_timeout:
                        recovery_durations.append(recovery_elapsed)
                        outcome = "recovery_timeout"
                elif phase == "WAITING":
                    if fallen:
                        recovery_attempts += 1
                        if recovery_attempts > self.args.max_recovery_attempts:
                            outcome = "max_recovery_attempts"
                        else:
                            phase = "RECOVERY"
                            self._viewer_phase = phase
                            recovery_elapsed = 0.0
                            left_contact_time = 0.0
                            right_contact_time = 0.0
                            print(
                                f"[trial {trial_index:03d}] robot fell while WAITING; "
                                "re-entering RECOVERY"
                            )
                elif phase == "RETURN_TO_STAND":
                    if fallen:
                        recovery_attempts += 1
                        if recovery_attempts > self.args.max_recovery_attempts:
                            outcome = "max_recovery_attempts"
                        else:
                            phase = "RECOVERY"
                            self._viewer_phase = phase
                            recovery_elapsed = 0.0
                            left_contact_time = 0.0
                            right_contact_time = 0.0
                            active_reference_key = None
                            print(
                                f"[trial {trial_index:03d}] robot fell while returning to stand; "
                                "re-entering RECOVERY"
                            )
                    elif (
                        standing
                        and return_elapsed >= self.args.return_min_time
                        and standard_stand_joint_rmse <= self.args.return_joint_rmse
                    ):
                        completed_reference_count += 1
                        return_durations.append(return_elapsed)
                        print(
                            f"[trial {trial_index:03d}] READY for next command after "
                            f"{return_elapsed:.2f}s return; press 1-8"
                        )
                        phase = "WAITING"
                        self._viewer_phase = phase
                        return_elapsed = 0.0
                        waiting_elapsed = 0.0
                        left_contact_time = 0.0
                        right_contact_time = 0.0
                        active_reference_key = None
                elif phase == "TRACKING":
                    if fallen:
                        tracking_falls += 1
                        print(
                            f"[trial {trial_index:03d}] TRACKING fall at t={tracking_elapsed:.2f}s "
                            f"(height={state['torso_height']:.3f}, upright={state['uprightness']:.3f})"
                        )
                        should_recover = self.args.mode == "interactive" or (
                            self.args.mode == "combined" and self.args.recover_after_tracking_fall
                        )
                        if should_recover:
                            interrupted_reference_count += int(self.args.mode == "interactive")
                            recovery_attempts += 1
                            if recovery_attempts > self.args.max_recovery_attempts:
                                outcome = "max_recovery_attempts"
                            else:
                                phase = "RECOVERY"
                                self._viewer_phase = phase
                                recovery_elapsed = 0.0
                                left_contact_time = 0.0
                                right_contact_time = 0.0
                                active_reference_key = None
                        else:
                            outcome = "tracking_fall"
                    elif (
                        self.args.mode != "interactive"
                        and tracking_elapsed >= self.args.tracking_seconds
                    ):
                        outcome = "tracking_complete"

                if total_elapsed >= next_print_time:
                    print(
                        f"[trial {trial_index:03d}] {phase_before_transition:<8} "
                        f"sim={total_elapsed:6.2f}s height={state['torso_height']:.3f} "
                        f"upright={state['uprightness']:+.3f} feet={int(feet_stable)} "
                        f"|action|max={np.abs(action).max():.2f}"
                    )
                    next_print_time += self.args.print_interval

                if self.args.real_time:
                    real_time_target += self.policy_dt
                    remaining = real_time_target - time.monotonic()
                    if remaining > 0.0:
                        time.sleep(remaining)
        finally:
            terminal_reader.__exit__(*sys.exc_info())
            self._viewer_phase = "IDLE"
            self._pending_reference_key = None
            try:
                if self.viewer is not None:
                    self.viewer.close()
                    self.viewer = None
            finally:
                signal.signal(signal.SIGINT, previous_sigint_handler)

        tracking_rows = [row for row in rows if row["phase"] == "TRACKING"]
        recovery_rows = [row for row in rows if row["phase"] == "RECOVERY"]
        return_rows = [row for row in rows if row["phase"] == "RETURN_TO_STAND"]
        waiting_rows = [row for row in rows if row["phase"] == "WAITING"]

        def mean_metric(key: str) -> float | None:
            values = np.asarray([row[key] for row in tracking_rows], dtype=np.float64)
            values = values[np.isfinite(values)]
            return float(values.mean()) if len(values) else None

        mean_anchor_height = mean_metric("anchor_height_error")
        mean_anchor_ori = mean_metric("anchor_ori_error")
        mean_body_pos = mean_metric("body_pos_error")
        mean_body_ori = mean_metric("body_ori_error")
        tracking_pass = bool(
            tracking_rows
            and tracking_falls == 0
            and mean_anchor_height is not None
            and mean_anchor_height < 0.20
            and mean_anchor_ori is not None
            and mean_anchor_ori < 0.60
            and mean_body_pos is not None
            and mean_body_pos < 0.18
            and mean_body_ori is not None
            and mean_body_ori < 0.60
        )
        expected_recovery = self.args.mode in ("recovery", "combined", "interactive")
        recovery_pass = bool(recovery_successes >= 1) if expected_recovery else None
        summary: dict[str, Any] = {
            "trial": trial_index,
            "mode": self.args.mode,
            "outcome": outcome,
            "policy": str(self.args.policy),
            "tracking_motion": str(tracking_motion.path),
            "selected_reference_key": selected_reference_key,
            "selected_reference_keys": selected_reference_keys,
            "completed_reference_count": completed_reference_count,
            "interrupted_reference_count": interrupted_reference_count,
            "fall_pose": fall_pose.label if fall_pose else None,
            "sim_time_s": total_elapsed,
            "policy_steps": len(rows),
            "recovery_attempts": recovery_attempts,
            "recovery_successes": recovery_successes,
            "recovery_success": recovery_pass,
            "recovery_durations_s": recovery_durations,
            "tracking_duration_s": len(tracking_rows) * self.policy_dt,
            "return_to_stand_duration_s": len(return_rows) * self.policy_dt,
            "return_to_stand_durations_s": return_durations,
            "waiting_duration_s": len(waiting_rows) * self.policy_dt,
            "tracking_falls": tracking_falls,
            "tracking_pass": tracking_pass if self.args.mode != "recovery" else None,
            "tracking_yaw_alignments_rad": tracking_yaw_alignments,
            "mean_joint_pos_rmse_rad": mean_metric("joint_pos_rmse"),
            "mean_joint_vel_rmse_rad_s": mean_metric("joint_vel_rmse"),
            "mean_anchor_height_error_m": mean_anchor_height,
            "mean_anchor_ori_error_rad": mean_anchor_ori,
            "mean_body_pos_error_m": mean_body_pos,
            "mean_body_ori_error_rad": mean_body_ori,
            "max_abs_action": max_abs_action,
            "max_abs_torque_nm": max_abs_torque,
            "max_floor_penetration_m": max_penetration,
            "reset_height_correction_m": self.reset_height_correction,
            "recovery_initial_height_m": recovery_rows[0]["torso_height"] if recovery_rows else None,
            "recovery_initial_uprightness": recovery_rows[0]["uprightness"] if recovery_rows else None,
            "recovery_max_height_m": (
                max(row["torso_height"] for row in recovery_rows) if recovery_rows else None
            ),
            "recovery_max_uprightness": (
                max(row["uprightness"] for row in recovery_rows) if recovery_rows else None
            ),
        }
        return summary, rows


def write_trial_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    recovery_trials = [
        item for item in summaries if item["mode"] in ("recovery", "combined", "interactive")
    ]
    tracking_trials = [
        item for item in summaries if item["mode"] in ("tracking", "combined", "interactive")
    ]
    successful_durations = [
        duration
        for item in recovery_trials
        if item["recovery_success"]
        for duration in item["recovery_durations_s"][:1]
    ]
    return {
        "trial_count": len(summaries),
        "recovery_trial_count": len(recovery_trials),
        "recovery_success_rate": (
            sum(bool(item["recovery_success"]) for item in recovery_trials) / len(recovery_trials)
            if recovery_trials
            else None
        ),
        "mean_successful_recovery_time_s": (
            float(np.mean(successful_durations)) if successful_durations else None
        ),
        "tracking_trial_count": len(tracking_trials),
        "tracking_pass_rate": (
            sum(bool(item["tracking_pass"]) for item in tracking_trials) / len(tracking_trials)
            if tracking_trials
            else None
        ),
        "tracking_fall_rate": (
            sum(item["tracking_falls"] > 0 for item in tracking_trials) / len(tracking_trials)
            if tracking_trials
            else None
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Quantitative MuJoCo sim-to-sim test for the joint Z1 recovery/tracking policy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=("combined", "recovery", "tracking", "interactive"),
        default="combined",
        help=(
            "interactive repeatedly waits for key 1-8, plays one reference, and returns to "
            "the standard standing target."
        ),
    )
    policy_source = parser.add_mutually_exclusive_group()
    policy_source.add_argument(
        "--policy",
        type=Path,
        default=None,
        help="Exact exported ONNX path. Uses the repository's default policy when omitted.",
    )
    policy_source.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Training run directory; automatically load its exported/<run-name>.onnx.",
    )
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument(
        "--motion",
        type=Path,
        default=DEFAULT_MOTIONS,
        help="Tracking NPZ file or directory. Directory entries are cycled across trials.",
    )
    parser.add_argument(
        "--motion-glob",
        type=str,
        default="*_clip.npz",
        help=(
            "Glob used only when --motion is a directory. The default selects the eight clipped boxing "
            "references and excludes boxing_walk_*.npz files."
        ),
    )
    parser.add_argument("--recovery-target", type=Path, default=DEFAULT_RECOVERY_TARGET)
    parser.add_argument("--recovery-target-frame", type=int, default=64)
    parser.add_argument("--fall-source", type=Path, default=DEFAULT_FALLS)
    parser.add_argument("--fall-file", type=str, default=None, help="Optional exact filename within --fall-source.")
    parser.add_argument("--fall-frame", type=int, default=None, help="Requires --fall-file.")
    parser.add_argument(
        "--cycle-fall-files",
        action="store_true",
        help=(
            "Cycle through recovery NPZ files across trials and randomly sample one eligible frame per file. "
            "Useful for visually pairing different recovery clips with different tracking references."
        ),
    )
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tracking-start-frame", type=int, default=0)
    parser.add_argument(
        "--tracking-seconds",
        type=float,
        default=8.0,
        help="Tracking horizon for non-interactive modes; interactive plays each selected clip once.",
    )
    parser.add_argument(
        "--tracking-loop",
        action="store_true",
        help="Loop the tracking clip in non-interactive modes; ignored by interactive mode.",
    )
    parser.add_argument("--recovery-timeout", type=float, default=6.0)
    parser.add_argument(
        "--return-min-time",
        type=float,
        default=0.5,
        help="Minimum standard-stand settling time after each interactive reference.",
    )
    parser.add_argument(
        "--return-joint-rmse",
        type=float,
        default=0.35,
        help="Maximum joint RMSE to the standard stand target before accepting another key.",
    )
    parser.add_argument("--max-recovery-attempts", type=int, default=2)
    parser.add_argument(
        "--recover-after-tracking-fall",
        action="store_true",
        help="In combined mode, reopen recovery after a fall during tracking instead of stopping.",
    )
    parser.add_argument(
        "--no-align-tracking-yaw-after-recovery",
        dest="align_tracking_yaw_after_recovery",
        action="store_false",
        help=(
            "Keep the source clip's global yaw after recovery. By default a fresh reference is rotated "
            "to the recovered robot's current yaw, avoiding a discontinuous 180-degree turn command."
        ),
    )

    parser.add_argument("--fall-height", type=float, default=0.50)
    parser.add_argument("--fall-uprightness", type=float, default=0.342)
    parser.add_argument("--stand-height", type=float, default=0.75)
    parser.add_argument("--stand-uprightness", type=float, default=0.85)
    parser.add_argument("--min-contact-time", type=float, default=0.15)
    parser.add_argument("--max-foot-speed", type=float, default=0.20)
    parser.add_argument("--fall-reset-max-height", type=float, default=0.62)
    parser.add_argument("--fall-reset-max-uprightness", type=float, default=1.0)
    parser.add_argument("--ground-clearance", type=float, default=0.02)
    parser.add_argument(
        "--reset-contact-clearance",
        type=float,
        default=0.001,
        help="Extra clearance after lifting the base out of initial MuJoCo mesh/floor penetration.",
    )

    parser.add_argument("--policy-dt", type=float, default=0.02)
    parser.add_argument("--sim-dt", type=float, default=None)
    parser.add_argument("--action-clip", type=float, default=None)
    parser.add_argument("--torque-clip", type=float, default=None)
    parser.add_argument("--torque-scale", type=float, default=1.0)
    parser.add_argument(
        "--no-torque-limits",
        action="store_true",
        help="Disable the per-joint Isaac effort_limit_sim clipping (not recommended).",
    )
    parser.add_argument("--push-time", type=float, default=None, help="Tracking-phase time at which to add qvel.")
    parser.add_argument("--push-linear", type=vec3, default=(0.0, 0.0, 0.0))
    parser.add_argument("--push-angular", type=vec3, default=(0.0, 0.0, 0.0))

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Exact result directory. When omitted, a timestamped <time>_<mode> directory is created under "
            "outputs/mujoco_recovery_tracking so different runs cannot overwrite one another."
        ),
    )
    parser.add_argument("--print-interval", type=float, default=0.5)
    parser.add_argument("--no-render", dest="render", action="store_false")
    parser.add_argument(
        "--render-all-trials",
        action="store_true",
        help="Open the MuJoCo viewer for every trial, not only the first one.",
    )
    parser.add_argument("--no-real-time", dest="real_time", action="store_false")
    parser.set_defaults(render=True, real_time=True, align_tracking_yaw_after_recovery=True)
    return parser


def resolve_policy_path(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.run_dir is None:
        args.policy = args.policy or DEFAULT_POLICY
        return
    if not args.run_dir.is_dir():
        parser.error(f"--run-dir is not a directory: {args.run_dir}")
    exported_dir = args.run_dir / "exported"
    preferred = exported_dir / f"{args.run_dir.name}.onnx"
    if preferred.is_file():
        args.policy = preferred
        return
    candidates = sorted(exported_dir.glob("*.onnx")) if exported_dir.is_dir() else []
    if not candidates:
        parser.error(f"--run-dir contains no exported/*.onnx: {args.run_dir}")
    # If a legacy run has several exports, the newest file is the least
    # surprising automatic choice and is always printed before evaluation.
    args.policy = max(candidates, key=lambda path: path.stat().st_mtime_ns)


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.trials <= 0:
        parser.error("--trials must be positive")
    if args.policy_dt <= 0.0 or (args.sim_dt is not None and args.sim_dt <= 0.0):
        parser.error("control timesteps must be positive")
    if args.tracking_seconds <= 0.0 or args.recovery_timeout <= 0.0:
        parser.error("phase timeouts must be positive")
    if args.return_min_time < 0.0 or args.return_joint_rmse <= 0.0:
        parser.error("--return-min-time must be non-negative and --return-joint-rmse must be positive")
    if args.max_recovery_attempts <= 0 or args.torque_scale <= 0.0:
        parser.error("--max-recovery-attempts and --torque-scale must be positive")
    if args.action_clip is not None and args.action_clip <= 0.0:
        parser.error("--action-clip must be positive")
    if args.torque_clip is not None and args.torque_clip <= 0.0:
        parser.error("--torque-clip must be positive")
    if args.fall_frame is not None and args.fall_file is None:
        parser.error("--fall-frame requires --fall-file")
    if args.fall_file is not None and args.fall_frame is None:
        parser.error("--fall-file requires --fall-frame")
    if args.cycle_fall_files and args.fall_file is not None:
        parser.error("--cycle-fall-files cannot be combined with --fall-file/--fall-frame")
    if args.print_interval <= 0.0:
        parser.error("--print-interval must be positive")
    if args.ground_clearance < 0.0 or args.reset_contact_clearance < 0.0:
        parser.error("reset clearances must be non-negative")
    for path_name in ("policy", "xml", "motion", "recovery_target"):
        path = getattr(args, path_name)
        if not path.exists():
            parser.error(f"--{path_name.replace('_', '-')} does not exist: {path}")
    if args.mode != "tracking" and not args.fall_source.exists():
        parser.error(f"--fall-source does not exist: {args.fall_source}")
    if args.output_dir is not None and args.output_dir.exists() and not args.output_dir.is_dir():
        parser.error(f"--output-dir exists but is not a directory: {args.output_dir}")
    if args.mode == "interactive" and not args.render:
        parser.error("--mode interactive requires the MuJoCo viewer; do not pass --no-render")
    if args.mode == "interactive" and not args.real_time:
        parser.error("--mode interactive requires real-time stepping; do not pass --no-real-time")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    resolve_policy_path(args, parser)
    validate_args(args, parser)
    if args.mode == "interactive" and args.tracking_loop:
        print(
            "[WARN] --tracking-loop is ignored in interactive mode: each selected reference "
            "plays once, returns to standard stand, then waits for the next key."
        )
        args.tracking_loop = False
    if _IMPORT_ERROR is not None:
        parser.error(
            f"missing Python package {_IMPORT_ERROR.name!r}. Install numpy, mujoco and onnxruntime "
            "in the Python environment used to run this script."
        )
    if args.output_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.output_dir = DEFAULT_OUTPUT_ROOT / f"{timestamp}_{args.mode}"
    elif args.output_dir.exists() and any(args.output_dir.iterdir()):
        print(
            f"[WARN] explicit output directory is not empty: {args.output_dir}. "
            "Files with matching names will be overwritten; unrelated old trial CSVs are not removed."
        )

    tracking_paths = expand_motion_paths(args.motion, args.motion_glob)
    tracking_motions = [NamedMotion(path) for path in tracking_paths]
    reference_motions: dict[int, NamedMotion] | None = None
    if args.mode == "interactive":
        try:
            reference_paths = resolve_reference_key_map(args.motion)
        except ValueError as error:
            parser.error(str(error))
        print_reference_key_map(reference_paths)
        reference_motions = {
            key: NamedMotion(path) for key, path in reference_paths.items()
        }
    recovery_target = NamedMotion(args.recovery_target)
    rng = np.random.default_rng(args.seed)
    fall_catalog = (
        FallCatalog(
            args.fall_source,
            "torso_link",
            args.fall_reset_max_height,
            args.fall_reset_max_uprightness,
        )
        if args.mode != "tracking"
        else None
    )
    exact_fall = (
        fall_catalog.exact(args.fall_file, args.fall_frame)
        if fall_catalog is not None and args.fall_file is not None
        else None
    )

    print(f"[INFO] policy: {args.policy}")
    print(f"[INFO] mode={args.mode}, trials={args.trials}, tracking motions={len(tracking_motions)}")
    if fall_catalog is not None:
        print(
            f"[INFO] recovery reset distribution: {len(fall_catalog.motions)} files, "
            f"{len(fall_catalog.valid)} eligible frames"
        )

    evaluator = RecoveryTrackingEvaluator(args)
    print(
        f"[INFO] ONNX input=({evaluator.input_name}, 127), policy_dt={evaluator.policy_dt:.4f}s, "
        f"sim_dt={evaluator.sim_dt:.4f}s, decimation={evaluator.decimation}"
    )
    summaries: list[dict[str, Any]] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for trial in range(args.trials):
        motion = tracking_motions[trial % len(tracking_motions)]
        if exact_fall is not None:
            fall_pose = exact_fall
        elif fall_catalog is not None and args.cycle_fall_files:
            fall_pose = fall_catalog.sample_from_motion(rng, trial)
        else:
            fall_pose = fall_catalog.sample(rng) if fall_catalog is not None else None
        print(
            f"\n[trial {trial:03d}] motion={motion.path.name}, "
            f"fall={fall_pose.label if fall_pose else 'n/a'}"
        )
        # By default only the first batch trial is visualized; explicitly opt in
        # to watching every sampled fall/reference pair sequentially.
        render_this_trial = bool(
            args.render
            and (trial == 0 or args.render_all_trials or args.mode == "interactive")
        )
        summary, rows = evaluator.run_trial(
            trial,
            motion,
            recovery_target,
            fall_pose,
            render=render_this_trial,
            reference_motions=reference_motions,
        )
        summaries.append(summary)
        write_trial_csv(args.output_dir / f"trial_{trial:03d}.csv", rows)
        print("[RESULT] " + json.dumps(summary, ensure_ascii=False, indent=2))
        if summary["outcome"] == "operator_interrupt":
            break

    report = {
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "aggregate": aggregate_summaries(summaries),
        "trials": summaries,
    }
    report_path = args.output_dir / "summary.json"
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2, allow_nan=False)
    print("\n[SUMMARY] " + json.dumps(report["aggregate"], ensure_ascii=False, indent=2))
    print(f"[SUMMARY] JSON/CSV written to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
