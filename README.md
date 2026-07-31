# magicbot-z1-mimic

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.1.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)

[[Website]](https://beyondmimic.github.io/)
[[Arxiv]](https://arxiv.org/abs/2508.08241)
[[Video]](https://youtu.be/RS_MtKVIAzY)

## Overview

This project inherits the **BeyondMimic** motion-tracking stack: highly dynamic tracking, local **`.npz`** motions and
**WandB registry** sources, CSV preprocessing through training, playback, and **MuJoCo sim-to-sim**
(`scripts/sim2sim_mujoco.py`). Upstream references: [BeyondMimic site](https://beyondmimic.github.io/), paper and video
in the badges above.

For sim-to-real deployment, refer to
the [motion_tracking_controller](https://github.com/HybridRobotics/motion_tracking_controller).

## Supported robots / 当前支持机型

| Robot | Notes |
|--------|--------|
| **MagicBot Z1** | 23 actuated DoF; URDF/MJCF under `whole_body_tracking/assets/magicbot-z1_description/`, env/agents under `tasks/tracking/config/z1/`. |

**Only Z1 is supported at the moment.** Other platforms that may exist in upstream forks are not wired up in this repo.

## Installation

1. Install Isaac Lab v2.1.0 by following the
   [official installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
   The conda installation is recommended.

2. Clone this repository (outside the `IsaacLab` directory), for example:

```bash
git clone <your-remote-url>/magicbot-z1-mimic.git
cd magicbot-z1-mimic
```

3. The Z1 robot description (URDF + meshes) lives at
   `source/whole_body_tracking/whole_body_tracking/assets/magicbot-z1_description/`:

```
magicbot-z1_description/
├── urdf/MagicBotZ1_23dof.urdf   # main URDF, referenced by robots/z1.py
└── meshes/*.STL                 # collision/visual meshes (relative ../meshes paths)
```

   If you don't already have it locally, place your MagicBot Z1 URDF and STL meshes under that path.

4. Install the package into your Isaac Lab Python environment:

```bash
python -m pip install -e source/whole_body_tracking
```

## Motion Tracking

The pipeline has three stages: **(1) CSV → NPZ preprocessing**, **(2) replay to verify the motion**,
**(3) train + play the policy**. Every stage accepts either a local `.npz` file or a WandB registry.

### 1. Motion Preprocessing

#### 1a. Retargeting human motion to Z1 (produces the pkl)

`csv_to_npz_z1.py` consumes a CSV that **already** stores per-frame Z1 joint angles + a root pose. To produce this CSV
from a generic human motion (e.g. SMPL-X, AMASS, LAFAN1), use [**GMR — General Motion Retargeting**](https://github.com/YanjieZe/GMR)
which supports the MagicBot Z1 skeleton and outputs the exact column layout we expect.


#### 1b. CSV → NPZ via forward kinematics on Z1

```bash
python scripts/csv_to_npz_z1.py \
  --input_file path/to/motion.csv \
  --input_fps 30 \
  --output_name motion_name \
  --headless
```

This produces an `.npz` (and uploads it to the WandB registry if `WANDB_ENTITY` is set).

Supported pre-retargeted source datasets (please respect their original licenses):
- Unitree-retargeted LAFAN1 — [HuggingFace](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset)
- Sidekicks — [KungfuBot](https://kungfu-bot.github.io/)
- Cristiano Ronaldo celebration — [ASAP](https://github.com/LeCAR-Lab/ASAP)
- Balance motions — [HuB](https://hub-robot.github.io/)

### 2. Replay Reference Motion in Isaac Sim

Verify a motion file before training. The replay loop drives the robot kinematically from the npz.

```bash
# Local file
python scripts/replay_npz_z1.py \
  --motion_file source/whole_body_tracking/whole_body_tracking/datasets/simple_dance.npz

# Or from WandB registry
python scripts/replay_npz_z1.py --registry_name your-org/wandb-registry-motions/dance
```

### 3. Policy Training

```bash
# Local motion file
python scripts/rsl_rl/train.py \
  --task=Tracking-Flat-Z1-v0 \
  --motion_file source/whole_body_tracking/whole_body_tracking/datasets/simple_dance.npz \
  --num_envs 4096 --headless

# Or WandB registry (set --logger wandb to also log to WandB)
python scripts/rsl_rl/train.py \
  --task=Tracking-Flat-Z1-v0 \
  --registry_name your-org/wandb-registry-motions/dance \
  --headless --logger wandb --log_project_name beyondmimic --run_name z1_dance --num_envs 4096
```

Available task variants:
- `Tracking-Flat-Z1-v0` — default

Logs are written to `logs/rsl_rl/z1_flat/{timestamp}_{run_name}/`. Checkpoints are saved every 500 iterations (see
`Z1FlatPPORunnerCfg.save_interval`).

#### Multi-motion boxing training

Pass a directory to train one policy on every `.npz` file in it. Motions remain separate; each motion is divided into
approximately one-second bins for sampling and difficulty scoring; a final segment shorter than half a second is
merged into the preceding bin. Rollouts continue across adjacent bins in the same motion and reset only on a hard
failure, source-motion end, or episode timeout, so inter-bin transitions are trained.
The Z1 task reserves a fixed 40% of slots for delayed-fall eligibility and fallen-pose resets. Eligibility is separate
from the dynamic recovery phase: after a stable get-up the same environment samples a fresh boxing reference and
returns to tracking without reset; if it falls again, a new six-second recovery window opens. The remaining 60% start
as pure reference tracking (37.5% global coverage, 7.5% hard replay, 15% soft-error replay globally). Tracking and
recovery rewards are mutually exclusive, with only regularizers shared; there is no success/failure terminal bonus.
The Actor has no task flag or privileged height and appends only deployable projected gravity to the legacy input.
A single privileged Critic receives phase/progress/torso-height/stable-feet state. See
[the complete training and RB deployment design](docs/boxing_recovery_training_and_rb_deployment.md) for thresholds,
reward scales, data, from-scratch training commands, and the real-robot reference switch.
Motion files must include `joint_names` and `body_names`. The loader maps data by name into the active Isaac robot
order and deliberately rejects legacy index-only files, preventing MuJoCo/Isaac articulation-order mismatches.

```bash
python scripts/rsl_rl/train.py \
  --task=Tracking-Flat-Z1-v0 \
  --motion_dir source/whole_body_tracking/whole_body_tracking/datasets/boxing-dataset-magiclab-z1/train_npz \
  --num_envs 256 --max_iterations 30000 --headless \
  --logger tensorboard --run_name boxing_multi_v1
```

TensorBoard event files are written into the run directory. When training finishes normally, the numerically latest
`model_*.pt` is reloaded and exported to `<run>/exported/<run-name>.onnx`; its checkpoint path is embedded in the
ONNX metadata as `source_checkpoint`.

### 4. Policy Evaluation

Play a trained checkpoint and (optionally) export it to ONNX:

```bash
# From a local log directory
python scripts/rsl_rl/play.py \
  --task=Tracking-Flat-Z1-v0 \
  --load_run 2000iters \
  --checkpoint model_2000.pt \
  --motion_file source/whole_body_tracking/whole_body_tracking/datasets/simple_dance.npz \
  --num_envs 2

# Or directly from a WandB run
python scripts/rsl_rl/play.py \
  --task=Tracking-Flat-Z1-v0 \
  --wandb_path your-org/beyondmimic/{8-char-id} \
  --num_envs 2
```

`play.py` automatically exports the policy to ONNX under `logs/rsl_rl/z1_flat/{run}/exported/policy.onnx` for
sim-to-real deployment.

### 5. Sim-to-Sim Playback in MuJoCo

Run an exported ONNX policy in MuJoCo against the same Z1 model used for training — useful for sim-to-sim checks,
debugging observation/action conventions, and offline rollouts without launching Isaac Sim.

```bash
python scripts/sim2sim_mujoco.py \
  --xml source/whole_body_tracking/whole_body_tracking/assets/magicbot-z1_description/mjcf/MagicBotZ1_23dof.xml \
  --policy logs/rsl_rl/z1_flat/2000iters/exported/policy.onnx \
  --motion source/whole_body_tracking/whole_body_tracking/datasets/simple_dance.npz
```

The MJCF model (`MagicBotZ1_23dof.xml`) is tuned to match Isaac Lab's `Z1_CYLINDER_CFG`:

- Per-joint **armatures** identical to the `ImplicitActuator` config (legs/waist `0.0573`,
  ankles `0.0301`, arms `0.0225`).
- Contact `solref="0.005 1.0"` (5ms time constant) approximating PhysX's stiff contact response.
- Floor `friction="1.0 0.05 0.01"` with `condim="6"` so feet don't slip or pivot.

The script reproduces Isaac Lab's exact observation layout (`obs ∈ R^124`):

```
command (46) | motion_anchor_ori_b (6) | base_ang_vel (3) |
joint_pos_rel (23) | joint_vel (23) | actions (23)
```

PD torques are computed at every MuJoCo sim step (not just at policy rate) to mirror PhysX's
`ImplicitActuator`, using `joint_stiffness` / `joint_damping` / `action_scale` / `default_joint_pos`
pulled directly from the ONNX metadata.

#### Useful flags

| flag | behaviour |
|---|---|
| `--init_at_motion` | initialise robot at motion frame 0 (pelvis pose + joints) instead of the default pose |
| `--zero_motion` | replace the motion command with default-pose + zero velocity (PD stand-in-place test) |
| `--hold_default` | bypass the policy entirely and PD-hold the default pose (pure-physics sanity check) |
| `--debug_steps N` | print the full obs / action / torque vectors for the first `N` policy steps |
| `--action_clip K` | clip policy outputs to ±K (safety) |
| `--torque_clip K` | clip computed PD torque to ±K (safety) |
| `--no_render` / `--no_real_time` | run as fast as possible, no viewer (used for batch evaluation) |

By default the script:

- inherits the **yaw** from the motion's first frame so the robot starts facing the same direction as the reference,
- runs MuJoCo at `sim_dt = 0.002s` and the policy at `policy_dt = 0.02s` (50 Hz, decimation 10),
- reports the maximum floor-penetration depth and final pelvis pose at exit.

### Debugging Notes

- `export WANDB_ENTITY=your_org` before any WandB-backed script.
- If `/tmp` is not writable, edit the temp folder paths in `scripts/csv_to_npz_z1.py`.
- The 23-DoF Z1 URDF expects motion `.npz` files with `joint_pos`/`joint_vel` of shape `(T, 23)` and
  `body_*_w` of shape `(T, 24, ...)`.

## Code Structure

- **`scripts/`** — Utility entry points
    - `csv_to_npz_z1.py` — Retargeted CSV → motion `.npz` (with WandB upload)
    - `replay_npz_z1.py` — Replay an `.npz` in Isaac Sim
    - `rsl_rl/train.py` — PPO training
    - `rsl_rl/play.py` — Policy playback + ONNX export
    - `sim2sim_mujoco.py` — Run an exported `policy.onnx` in MuJoCo (sim-to-sim)
- **`source/whole_body_tracking/whole_body_tracking/robots/z1.py`** — Z1 articulation configuration
  (asset path, actuators, joint limits, action scale).
- **`source/whole_body_tracking/whole_body_tracking/assets/magicbot-z1_description/`** — MagicBot Z1 description.
    - `urdf/MagicBotZ1_23dof.urdf` — URDF used by Isaac Lab.
    - `mjcf/MagicBotZ1_23dof.xml` — MuJoCo model used by `sim2sim_mujoco.py` (armatures/contacts tuned to match Isaac).
    - `meshes/*.STL` — shared collision/visual meshes.
- **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/`**
    - `tracking_env_cfg.py` — Tracking environment (MDP) configuration.
    - `mdp/`
        - `commands.py` — Reference-motion command and adaptive sampling.
        - `rewards.py` — DeepMimic reward terms and smoothing.
        - `events.py` — Domain randomization.
        - `observations.py` — Motion-tracking observations.
        - `terminations.py` — Early terminations and timeouts.
    - `config/z1/`
        - `flat_env_cfg.py` — `Z1FlatEnvCfg` and its variants.
        - `agents/rsl_rl_ppo_cfg.py` — `Z1FlatPPORunnerCfg` (PPO hyperparameters).
        - `__init__.py` — Gym task registrations (`Tracking-Flat-Z1-*`).
- **`source/whole_body_tracking/whole_body_tracking/datasets/`** — Local motion `.npz` files.
- **`logs/`** and **`outputs/`** — Training artifacts (ignored by git).
