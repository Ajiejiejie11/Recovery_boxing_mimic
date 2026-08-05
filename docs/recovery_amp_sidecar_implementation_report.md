# Recovery-only AMP 奖励旁路：实现报告

## 1. 最终奖励形式

现有单 Actor、共享 Critic、标准 RSL-RL PPO、tracking/recovery 状态机、reset、termination 和
RewardManager 奖励项保持不变。AMP 只在 recovery 有效 transition 上使用与 `AMP_mjlab` 相同的
discriminator reward 和线性插值形式，但比例按本任务要求设为 task:AMP = 0.85:0.15：

```text
r_amp_raw = clamp(1 - 0.25 * (D(s_t, s_t+1) - 1)^2, min=0)
r_disc    = 0.1 * r_amp_raw
r_final   = 0.85 * r_task + 0.15 * r_disc
```

因此：

```text
0 <= r_amp_raw <= 1
0 <= r_disc <= 0.1
0 <= 0.15 * r_disc <= 0.015
```

`0.15` 是固定插值系数，AMP 分量每个有效控制步的绝对最大值固定为 `0.015`。这里没有 warm-up、ramp、
EMA 奖励尺度标定或动态 reward weight。

该插值只替换有效 recovery transition 的 reward。tracking、tracking→fall 入口和 done/reset transition 继续
使用环境原本的完整 task reward，不乘 0.85：

```text
valid_amp = recovery_active_before_step AND NOT done_after_step

valid_amp:     r = 0.85 * r_task + 0.15 * r_disc
not valid_amp: r = r_task
```

原任务 `Tracking-Flat-Z1-v0` 保留为无 AMP 基线；新增任务为
`Tracking-Flat-Z1-Recovery-AMP-v0`。

## 2. 模块划分

新增 5 个 AMP 核心模块，外加 runner 和配置接线：

| 模块 | 职责 |
|---|---|
| `amp/features.py` | 在线 rollout 与离线 expert 共用的 AMP state 构造 |
| `amp/dataset.py` | 校验 NPZ、按文件构造 expert transition，禁止跨文件连接 |
| `amp/replay_buffer.py` | 保存 policy 在 recovery 中产生的有效 transition |
| `amp/discriminator.py` | running normalization、LSGAN discriminator、AMP raw reward |
| `amp/sidecar.py` | 独立 optimizer、固定 reward 插值、checkpoint 和日志状态 |

集成点：

- `mdp/observations.py`：生成单独的 `amp` observation，不进入 Actor/critic。
- `tracking_env_cfg.py`、`config/z1/flat_env_cfg.py`：只在 AMP 任务启用该 observation group。
- `utils/my_on_policy_runner.py`：在标准 PPO rollout 外围采集 transition、混合奖励并更新判别器。
- `config/z1/agents/rsl_rl_ppo_cfg.py`：AMP 超参数及专家目录。
- `config/z1/__init__.py`、`scripts/rsl_rl/train.py`：独立 Gym task 和 runner 选择。

## 3. 特征和专家数据

AMP state 使用 motion command 中除 `torso_link` anchor 外的 13 个刚体。每个刚体 15 维：

```text
anchor-relative position       3
anchor-relative rotation 6D    6
body-local linear velocity     3
body-local angular velocity    3
--------------------------------
per body                       15
state = 13 * 15               195
transition input = 195 * 2    390
```

位置和姿态相对 torso anchor 表示，速度转换到各刚体局部坐标。在线和离线严格调用同一个纯 PyTorch
特征函数，避免 quaternion、6D rotation 或坐标系不一致。

专家目录：

```text
motion_data/data_npz/npz
```

目录中的 3 个文件均为 schema v2、50 Hz、8410 帧，必需的 27-body 位姿/速度数组没有 NaN/Inf。当前控制
周期 0.02 s 对应 stride=1。每个文件产生 8409 个 transition，共 25227 个；文件末帧不会连接到下一文件
首帧。完整 fall-and-get-up 片段全部保留，不做阶段裁剪。

## 4. 数据流和 recovery 边界

```text
expert NPZ ──> shared AMP feature ──> expert transitions ────────┐
                                                                 ├─> discriminator update
recovery rollout ──> valid transitions ──> policy replay ────────┘
                                      │
                                      └─> raw AMP reward ─> fixed 0.1 scale
                                                               │
recovery task reward ───────────> 0.85 / 0.15 interpolation ──> PPO storage
```

runner 在 `env.step()` 前 clone `recovery_active`，step 后与 `~done` 相交：

- tracking 全阶段：不进入 policy replay，不计算混合 reward；
- tracking→fall/recovery 入口步：step 前仍为 tracking，排除；
- recovery 普通步：加入 replay，并使用 0.85/0.15 混合 reward；
- recovery 成功步：step 前是 recovery 且不 reset，保留；
- recovery timeout/其他 reset 步：`done=True`，排除 reset 伪 transition。

判别器只优化自身参数；Actor、critic 和 PPO optimizer 中没有 discriminator 参数。PPO 的网络、storage、
GAE 和 update 流程保持标准实现。

## 5. 固定参数和后续手动调整

默认参数：

| 参数 | 默认值 |
|---|---:|
| `amp_reward_coef` | 0.1 |
| `amp_task_reward_lerp` | 0.85 |
| AMP 插值系数 | 0.15 |
| AMP 每步最大分量 | 0.015 |
| discriminator hidden dims | 512, 256, 128 |
| discriminator LR | 1e-4 |
| batch / updates per iteration | 2048 / 4 |
| policy replay capacity | 50,000 transitions |
| gradient penalty | 10.0 |

后续需要改为 task:AMP = 0.70:0.30 时，必须手动设置：

```bash
agent.recovery_amp.amp_task_reward_lerp=0.70
```

此时 AMP 最大分量自然变为：

```text
(1 - 0.70) * 0.1 = 0.03
```

checkpoint 恢复 discriminator、normalizer 和 AMP optimizer，但不会覆盖当前配置的插值比例。因此使用
0.70 配置恢复 0.85 checkpoint 时，下一 rollout 会直接使用 0.70/0.30，无自动比例变化或重标定。

注意：0.85/0.15 是公式中的固定混合系数，不保证 AMP 数值在最终 reward 中恰好占 15%。实际数值占比还
取决于 task reward 的尺度，TensorBoard 使用 `AMP/observed_amp_fraction` 单独记录观察值。

## 6. 训练、日志与部署

```bash
python scripts/rsl_rl/train.py \
  --task=Tracking-Flat-Z1-Recovery-AMP-v0 \
  --motion_dir source/whole_body_tracking/whole_body_tracking/datasets/boxing-dataset-magiclab-z1/train_npz \
  --num_envs 4096 --max_iterations 30000 --headless \
  --logger tensorboard --run_name recovery_amp_15
```

对应基线只需把 task 换回 `Tracking-Flat-Z1-v0`。AMP 日志包括：

```text
AMP/amp_reward_coef
AMP/task_reward_lerp
AMP/amp_reward_lerp
AMP/max_amp_component
AMP/mean_raw_reward
AMP/mean_scaled_discriminator_reward
AMP/mean_task_component_abs
AMP/mean_amp_component
AMP/observed_amp_fraction
AMP/discriminator_loss
AMP/expert_score
AMP/policy_score
```

checkpoint 同时保存 policy/PPO、motion curriculum、discriminator、AMP normalizer 和 AMP optimizer。policy
ONNX 导出流程不包含 AMP，部署逻辑不变。

## 7. 验证结果和运行限制

已验证：

- 新增/修改 Python 文件通过 `compileall`、AST parse 和 `git diff --check`；
- AMP 特征为 195 维，全局刚体变换不变性最大误差约 `9.5e-7`；
- 三份真实 NPZ 正确产生 25227 个 transition；
- tracking/无效 transition reward 完全不变；
- recovery 有效 transition 严格满足 `0.85 * task + 0.15 * (0.1 * raw_amp)`；
- raw AMP 范围为 `[0, 1]`，AMP 分量不超过 `0.015`；
- replay、判别器反向更新、checkpoint round-trip 和真实 RSL-RL PPO 集成测试通过。

当前执行环境没有可用的 Isaac Sim GPU/Vulkan device，因此未在此处启动完整 PhysX 并行环境。首次正式
训练建议先用较小 `--num_envs` 跑 2--5 个 iteration，确认 AMP 日志后再启动完整训练。
