#!/usr/bin/env bash
# Source this file to enter the project's Isaac Lab training environment.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "Please source this script so it can activate the current shell:"
    echo "  source ./activate_lj_mimic_z1.sh"
    exit 1
fi

_magicbot_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
_magicbot_conda_root="${_magicbot_repo_root}/.miniforge3"
_magicbot_env="/data0/alan/conda/envs/lj-mimic-z1"
_magicbot_isaac_sim="/data0/isaacsim"
_magicbot_isaac_lab="/data0/alan/IsaacLab"

if [[ ! -f "${_magicbot_conda_root}/etc/profile.d/conda.sh" ]]; then
    echo "Miniforge was not found at: ${_magicbot_conda_root}" >&2
    return 1
fi

if [[ ! -x "${_magicbot_env}/bin/python" ]]; then
    echo "Training environment was not found at: ${_magicbot_env}" >&2
    return 1
fi

source "${_magicbot_conda_root}/etc/profile.d/conda.sh"
if ! conda activate "${_magicbot_env}"; then
    echo "Failed to activate training environment: ${_magicbot_env}" >&2
    return 1
fi

# A tmux shell may retain an alias or Bash's cached location for `python` from
# another Conda installation. Make the selected interpreter unambiguous.
unalias python pip 2>/dev/null || true
export PATH="${_magicbot_env}/bin:${PATH}"
hash -r

if [[ ! -f "${_magicbot_isaac_sim}/setup_conda_env.sh" ]]; then
    echo "Isaac Sim environment script was not found at: ${_magicbot_isaac_sim}" >&2
    return 1
fi

# A fresh shell (including a new tmux pane) does not inherit the Isaac Sim
# runtime paths merely by activating Conda.
source "${_magicbot_isaac_sim}/setup_conda_env.sh"

# Keep this explicit so the source checkout is importable even if its editable
# installs or Conda activation hooks are missing in a newly-created shell.
for _magicbot_python_path in \
    "${_magicbot_repo_root}/source/whole_body_tracking" \
    "${_magicbot_isaac_lab}/source/isaaclab" \
    "${_magicbot_isaac_lab}/source/isaaclab_assets" \
    "${_magicbot_isaac_lab}/source/isaaclab_mimic" \
    "${_magicbot_isaac_lab}/source/isaaclab_rl" \
    "${_magicbot_isaac_lab}/source/isaaclab_tasks"; do
    case ":${PYTHONPATH:-}:" in
        *":${_magicbot_python_path}:"*) ;;
        *) export PYTHONPATH="${_magicbot_python_path}${PYTHONPATH:+:${PYTHONPATH}}" ;;
    esac
done

# Keep Z1 URDF-to-USD conversion artifacts off shared /tmp.
export WBT_USD_DIR="/data0/alan/isaac-cache/z1_urdf"
export PIP_CACHE_DIR="/data0/alan/pip-cache"
mkdir -p "${WBT_USD_DIR}" "${PIP_CACHE_DIR}"

_magicbot_actual_python="$(python -c 'import os, sys; print(os.path.realpath(sys.executable))')"
_magicbot_expected_python="$(realpath "${_magicbot_env}/bin/python")"
if [[ "${_magicbot_actual_python}" != "${_magicbot_expected_python}" ]]; then
    echo "Wrong Python interpreter after activation:" >&2
    echo "  expected: ${_magicbot_expected_python}" >&2
    echo "  actual:   ${_magicbot_actual_python}" >&2
    return 1
fi

if ! python -c 'import flatdict, importlib.util; assert all(importlib.util.find_spec(name) is not None for name in ("isaaclab", "isaaclab_rl", "isaaclab_tasks", "whole_body_tracking"))' 2>/dev/null; then
    echo "The training environment is missing one or more required Python packages." >&2
    echo "Python: ${_magicbot_actual_python}" >&2
    return 1
fi

echo "Activated lj-mimic-z1"
echo "  Python: ${_magicbot_actual_python}"
echo "  USD cache: ${WBT_USD_DIR}"
echo "  Isaac Sim: ${ISAAC_PATH}"

unset _magicbot_repo_root _magicbot_conda_root _magicbot_env
unset _magicbot_isaac_sim _magicbot_isaac_lab _magicbot_python_path
unset _magicbot_actual_python _magicbot_expected_python
