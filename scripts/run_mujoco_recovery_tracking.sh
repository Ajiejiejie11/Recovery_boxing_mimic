#!/usr/bin/env bash
set -euo pipefail

# Always use the environment that contains MuJoCo and ONNX Runtime.  This keeps
# the launcher working even when the interactive shell currently shows (base).
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname -- "${SCRIPT_DIR}")"
DANCE_STOP_PYTHON="${Z1_MUJOCO_PYTHON:-/home/user/miniconda3/envs/dance_stop/bin/python}"

if [[ ! -x "${DANCE_STOP_PYTHON}" ]]; then
    echo "[ERROR] MuJoCo Python is not executable: ${DANCE_STOP_PYTHON}" >&2
    echo "Set Z1_MUJOCO_PYTHON to the Python executable containing mujoco and onnxruntime." >&2
    exit 2
fi

exec "${DANCE_STOP_PYTHON}" \
    "${PROJECT_ROOT}/scripts/evaluate_recovery_tracking_mujoco.py" \
    "$@"
