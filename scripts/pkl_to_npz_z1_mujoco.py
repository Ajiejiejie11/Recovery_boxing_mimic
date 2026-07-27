"""Create a trainable Z1 motion ``.npz`` from a retargeted motion pickle.

This is a CPU-only counterpart to ``csv_to_npz_z1.py``.  It uses the repository's
Z1 MJCF model for forward kinematics, so it is suitable on machines where Isaac
Sim cannot start (for example, no CUDA device is available).  Output keys match
``whole_body_tracking.tasks.tracking.mdp.commands.MotionLoader``.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import mujoco
import numpy as np


EXPECTED_DOF = 23


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product for wxyz quaternions."""
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack(
        (aw * bw - ax * bx - ay * by - az * bz,
         aw * bx + ax * bw + ay * bz - az * by,
         aw * by - ax * bz + ay * bw + az * bx,
         aw * bz + ax * by - ay * bx + az * bw),
        axis=-1,
    )


def quat_conjugate(quat: np.ndarray) -> np.ndarray:
    result = quat.copy()
    result[..., 1:] *= -1
    return result


def slerp(a: np.ndarray, b: np.ndarray, blend: np.ndarray) -> np.ndarray:
    """Vectorized shortest-arc interpolation for wxyz quaternions."""
    dot = np.sum(a * b, axis=-1, keepdims=True)
    b = np.where(dot < 0.0, -b, b)
    dot = np.clip(np.abs(dot), -1.0, 1.0)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    safe_sin_theta = np.where(np.abs(sin_theta) < 1e-7, 1.0, sin_theta)
    linear = (1.0 - blend[:, None]) * a + blend[:, None] * b
    spherical = (np.sin((1.0 - blend[:, None]) * theta) / safe_sin_theta) * a + (
        np.sin(blend[:, None] * theta) / safe_sin_theta
    ) * b
    result = np.where(sin_theta < 1e-7, linear, spherical)
    return result / np.linalg.norm(result, axis=-1, keepdims=True)


def angular_velocity(quat_wxyz: np.ndarray, dt: float) -> np.ndarray:
    """Central SO(3) derivative, matching the Isaac CSV conversion script."""
    if len(quat_wxyz) < 3:
        return np.zeros((len(quat_wxyz), 3), dtype=np.float64)
    relative = quat_mul(quat_wxyz[2:], quat_conjugate(quat_wxyz[:-2]))
    relative *= np.where(relative[:, :1] < 0.0, -1.0, 1.0)
    xyz_norm = np.linalg.norm(relative[:, 1:], axis=-1)
    angle = 2.0 * np.arctan2(xyz_norm, np.clip(relative[:, 0], -1.0, 1.0))
    axis = np.divide(relative[:, 1:], xyz_norm[:, None], out=np.zeros_like(relative[:, 1:]), where=xyz_norm[:, None] > 1e-8)
    omega = axis * (angle / (2.0 * dt))[:, None]
    return np.concatenate((omega[:1], omega, omega[-1:]), axis=0)


def load_and_resample(path: Path, output_fps: int, quat_order: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open("rb") as file:
        source = pickle.load(file)
    root_pos = np.asarray(source["root_pos"], dtype=np.float64)
    root_quat = np.asarray(source["root_rot"], dtype=np.float64)
    joint_pos = np.asarray(source["dof_pos"], dtype=np.float64)
    fps = float(source["fps"])
    if root_pos.ndim != 2 or root_pos.shape[1] != 3 or root_quat.shape != (len(root_pos), 4):
        raise ValueError("Expected root_pos (T, 3) and root_rot (T, 4).")
    if joint_pos.shape != (len(root_pos), EXPECTED_DOF):
        raise ValueError(f"Expected dof_pos (T, {EXPECTED_DOF}), got {joint_pos.shape}.")
    if not all(np.isfinite(value).all() for value in (root_pos, root_quat, joint_pos)):
        raise ValueError("Input contains NaN or Inf.")
    if quat_order == "xyzw":
        root_quat = root_quat[:, [3, 0, 1, 2]]
    root_quat /= np.linalg.norm(root_quat, axis=1, keepdims=True)

    duration = (len(root_pos) - 1) / fps
    times = np.arange(0.0, duration, 1.0 / output_fps)
    phase = times * fps
    index0 = np.floor(phase).astype(np.int64)
    index1 = np.minimum(index0 + 1, len(root_pos) - 1)
    blend = phase - index0
    root_pos = (1.0 - blend[:, None]) * root_pos[index0] + blend[:, None] * root_pos[index1]
    joint_pos = (1.0 - blend[:, None]) * joint_pos[index0] + blend[:, None] * joint_pos[index1]
    root_quat = slerp(root_quat[index0], root_quat[index1], blend)
    return root_pos, root_quat, joint_pos


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a Z1 pkl motion to a trainable NPZ using MuJoCo FK.")
    parser.add_argument("--input_file", type=Path, required=True)
    parser.add_argument("--output_file", type=Path, required=True)
    parser.add_argument("--output_fps", type=int, default=50)
    parser.add_argument("--quat_order", choices=("xyzw", "wxyz"), default="xyzw")
    parser.add_argument(
        "--xml",
        type=Path,
        default=Path("source/whole_body_tracking/whole_body_tracking/assets/magicbot-z1_description/mjcf/MagicBotZ1_23dof.xml"),
    )
    args = parser.parse_args()

    root_pos, root_quat, joint_pos = load_and_resample(args.input_file, args.output_fps, args.quat_order)
    dt = 1.0 / args.output_fps
    joint_vel = np.gradient(joint_pos, dt, axis=0)
    root_lin_vel = np.gradient(root_pos, dt, axis=0)
    root_ang_vel = angular_velocity(root_quat, dt)

    model = mujoco.MjModel.from_xml_path(str(args.xml))
    if model.nq < 7 + EXPECTED_DOF:
        raise ValueError(f"MJCF has only {model.nq} qpos values; expected a floating base plus {EXPECTED_DOF} joints.")
    data = mujoco.MjData(model)
    joint_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        for joint_id in range(model.njnt)
        if 7 <= model.jnt_qposadr[joint_id] < 7 + EXPECTED_DOF
    ]
    if len(joint_names) != EXPECTED_DOF or any(name is None for name in joint_names):
        raise ValueError(f"Could not resolve the {EXPECTED_DOF} actuated joint names from the MJCF model.")
    body_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) for body_id in range(1, model.nbody)
    ]
    if any(name is None for name in body_names):
        raise ValueError("Could not resolve all body names from the MJCF model.")
    body_count = model.nbody - 1  # Isaac's articulation data excludes MuJoCo's world body.
    body_pos = np.empty((len(root_pos), body_count, 3), dtype=np.float32)
    body_quat = np.empty((len(root_pos), body_count, 4), dtype=np.float32)

    for frame in range(len(root_pos)):
        data.qpos[:] = model.qpos0
        data.qpos[:3] = root_pos[frame]
        data.qpos[3:7] = root_quat[frame]
        data.qpos[7 : 7 + EXPECTED_DOF] = joint_pos[frame]
        mujoco.mj_forward(model, data)
        body_pos[frame] = data.xpos[1:]
        body_quat[frame] = data.xquat[1:]

    body_lin_vel = np.gradient(body_pos, dt, axis=0).astype(np.float32)
    body_ang_vel = np.empty((len(root_pos), body_count, 3), dtype=np.float32)
    for body_index in range(body_count):
        body_ang_vel[:, body_index] = angular_velocity(body_quat[:, body_index], dt)

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output_file,
        schema_version=np.array(2, dtype=np.int64),
        fps=np.array(args.output_fps, dtype=np.int64),
        joint_names=np.asarray(joint_names),
        body_names=np.asarray(body_names),
        joint_pos=joint_pos.astype(np.float32),
        joint_vel=joint_vel.astype(np.float32),
        body_pos_w=body_pos,
        body_quat_w=body_quat,
        body_lin_vel_w=body_lin_vel,
        body_ang_vel_w=body_ang_vel,
    )
    print(f"Wrote {len(root_pos)} frames at {args.output_fps} FPS to {args.output_file} (bodies: {body_count}).")


if __name__ == "__main__":
    main()
