"""Convert a retargeted MagicBot Z1 motion pickle to the CSV format used here.

The generated CSV has no header and one frame per row:
``root_x, root_y, root_z, root_qx, root_qy, root_qz, root_qw, joint_0, ..., joint_22``.
It can be consumed directly by ``scripts/csv_to_npz_z1.py``.  The latter expects
the quaternion columns in xyzw order and changes them internally to wxyz.

Example:
    python scripts/pkl_to_csv_z1.py \
        --input_file source/whole_body_tracking/whole_body_tracking/datasets/boxing-dataset-magiclab-z1/quanji_real_mix_pkl/boxing_yewendun_002.pkl \
        --output_file /tmp/boxing_yewendun_002.csv
    python scripts/csv_to_npz_z1.py --input_file /tmp/boxing_yewendun_002.csv \
        --input_fps 120 --output_name boxing_yewendun_002 --output_fps 50 --headless
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np


EXPECTED_DOF = 23


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a MagicBot Z1 pkl motion to CSV.")
    parser.add_argument("--input_file", type=Path, required=True, help="Input .pkl motion file.")
    parser.add_argument("--output_file", type=Path, required=True, help="Output headerless CSV path.")
    parser.add_argument(
        "--frame_range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="Optional half-open frame range [START, END). Defaults to every frame.",
    )
    parser.add_argument(
        "--quat_order",
        choices=("xyzw", "wxyz"),
        default="xyzw",
        help="Quaternion order in the pickle (default: xyzw). Output is always xyzw.",
    )
    parser.add_argument("--precision", type=int, default=9, help="Decimal places in the output CSV.")
    return parser.parse_args()


def load_motion(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, int | None]:
    with path.open("rb") as file:
        data = pickle.load(file)

    if not isinstance(data, dict):
        raise ValueError(f"Expected a dictionary pickle, got {type(data).__name__}.")
    required = ("root_pos", "root_rot", "dof_pos")
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}.")

    root_pos = np.asarray(data["root_pos"], dtype=np.float64)
    root_rot = np.asarray(data["root_rot"], dtype=np.float64)
    dof_pos = np.asarray(data["dof_pos"], dtype=np.float64)
    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(f"root_pos must have shape (T, 3), got {root_pos.shape}.")
    if root_rot.ndim != 2 or root_rot.shape[1] != 4:
        raise ValueError(f"root_rot must have shape (T, 4), got {root_rot.shape}.")
    if dof_pos.ndim != 2 or dof_pos.shape[1] != EXPECTED_DOF:
        raise ValueError(f"dof_pos must have shape (T, {EXPECTED_DOF}), got {dof_pos.shape}.")
    if not (len(root_pos) == len(root_rot) == len(dof_pos)):
        raise ValueError("root_pos, root_rot, and dof_pos must contain the same number of frames.")
    if not all(np.isfinite(array).all() for array in (root_pos, root_rot, dof_pos)):
        raise ValueError("Motion contains NaN or Inf values.")

    quat_norm = np.linalg.norm(root_rot, axis=1)
    if np.any(quat_norm < 1e-8):
        raise ValueError("Motion contains a zero-length root quaternion.")
    root_rot = root_rot / quat_norm[:, None]
    return root_pos, root_rot, dof_pos, data.get("fps")


def main() -> None:
    args = parse_args()
    root_pos, root_rot, dof_pos, fps = load_motion(args.input_file)

    if args.quat_order == "wxyz":
        root_rot = root_rot[:, [1, 2, 3, 0]]

    start, end = (0, len(root_pos)) if args.frame_range is None else args.frame_range
    if not 0 <= start < end <= len(root_pos):
        raise ValueError(f"Invalid frame range [{start}, {end}) for {len(root_pos)} frames.")

    motion = np.concatenate((root_pos[start:end], root_rot[start:end], dof_pos[start:end]), axis=1)
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.output_file, motion, delimiter=",", fmt=f"%.{args.precision}f")
    fps_text = f", source fps: {fps}" if fps is not None else ""
    print(f"Wrote {len(motion)} frames x {motion.shape[1]} columns to {args.output_file}{fps_text}.")


if __name__ == "__main__":
    main()
