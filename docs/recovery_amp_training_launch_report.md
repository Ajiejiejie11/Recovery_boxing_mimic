# Recovery AMP 训练启动报告

## 1. 本次启动结果

已在宿主机 RTX 4090 上完成两组真实 Isaac Sim smoke test：

1. 128 environments × 2 iterations：验证环境、PPO、checkpoint 和 ONNX 完整链路；
2. 512 environments × 10 iterations：越过 replay 门槛，验证 AMP 判别器实际更新。

两组任务和配置均正确解析为：

```text
Z1FlatRecoveryAmpEnvCfg
Z1FlatRecoveryAmpPPORunnerCfg
physics_dt = 0.005
step_dt = 0.02
```

最终 512 × 10 轮的运行目录为：

```text
logs/rsl_rl/z1_flat/2026-08-04_18-36-06_recovery_amp_85_15_smoke_512x10
```

已生成 `model_18508.pt` 和 ONNX。最后一轮 AMP 指标：

| 指标 | 实测值 |
|---|---:|
| `AMP/discriminator_loss` | 0.105575 |
| `AMP/expert_score` | 0.794758 |
| `AMP/policy_score` | -0.943539 |
| `AMP/gradient_penalty` | 0.782121 |
| `AMP/updates` | 4 |
| `AMP/replay_size` | 46821 |
| `AMP/valid_transition_fraction` | 0.363037 |
| `AMP/mean_raw_reward` | 0.093677 |
| `AMP/mean_scaled_discriminator_reward` | 0.009368 |
| `AMP/mean_amp_component` | 0.001405 |
| `AMP/observed_amp_fraction` | 0.018209 |

checkpoint 中 AMP normalizer count 为 327680，判别器 optimizer 的 8 个参数组均已有 state，证明不是只初始化而是确实发生了训练。

同时确认当前文件系统只剩约 1.5 GB，已有 `logs/` 约占 864 MB。以下正式命令将 checkpoint 保存间隔从
500 调整为 2000，预计 6000 个新增 iteration 只保存约 4 个 checkpoint，避免不必要地占满磁盘。没有删除
任何现有训练结果。

## 2. 推荐正式训练：从当前 recovery 策略继续

已核对最新 recovery checkpoint：

```text
logs/rsl_rl/z1_flat/
2026-07-31_20-55-24_boxing_recovery_height35_refbridge_feet01_resume_6000/
model_18499.pt
```

它包含 127 维 Actor、260 维 Critic 和 motion curriculum，与当前 AMP 任务兼容；checkpoint 没有 AMP state，
加载后 Actor/PPO 从 iteration 18499 继续，discriminator 从头初始化。旧 checkpoint 内部记录了 `cuda:5`
优化器张量，为了能安全隔离到任意一张物理 GPU，已生成数值相同的 CPU-mapped
`model_18499_portable.pt`。

下面命令只向 Isaac Sim 暴露物理 GPU 0，隔离后程序内部仍使用 `cuda:0`。它训练额外
6000 iterations，最终 iteration 约为 24498：

```bash
source /data0/Software/anaconda3/etc/profile.d/conda.sh
conda activate /data0/alan/conda/envs/lj-mimic-z1

export PYTHONPATH="/home/alan/Project/walking-boxing-amp（recovery）/magicbot-mimic/source/whole_body_tracking:${PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0

python scripts/rsl_rl/train.py \
  --task=Tracking-Flat-Z1-Recovery-AMP-v0 \
  --motion_dir source/whole_body_tracking/whole_body_tracking/datasets/boxing-dataset-magiclab-z1/train_npz \
  --resume True \
  --load_run 2026-07-31_20-55-24_boxing_recovery_height35_refbridge_feet01_resume_6000 \
  --checkpoint model_18499_portable.pt \
  --num_envs 2048 \
  --max_iterations 6000 \
  --device cuda:0 \
  --headless \
  --logger tensorboard \
  --run_name recovery_amp_85_15_resume_18499_6000 \
  agent.experiment_name=z1_flat \
  agent.save_interval=2000
```

`CUDA_VISIBLE_DEVICES=0` 中的 `0` 是宿主机物理卡号；换卡时只改这个数字，`--device cuda:0`
保持不变。这样可防止 Isaac Sim/Vulkan 枚举其他已占满的 GPU 后启动 OOM。
`agent.experiment_name=z1_flat` 用于让 resume loader 在原 `z1_flat` 日志根目录找到 checkpoint；任务和 runner
仍然是 AMP 版本。

## 3. 已通过的 10-iteration 烟雾测试

将正式命令中的参数临时改为：

```text
--num_envs 512
--max_iterations 10
--run_name recovery_amp_85_15_smoke
agent.save_interval=10
```

512 environments 实测已越过默认的 4096 transition 更新门槛，同时覆盖了环境、PPO、AMP
replay、discriminator update、checkpoint 和 ONNX 导出。

## 4. 训练开始后首先检查的硬约束

以下值应保持固定或有界：

| TensorBoard tag | 预期 |
|---|---:|
| `AMP/amp_reward_coef` | 0.1 |
| `AMP/task_reward_lerp` | 0.85 |
| `AMP/amp_reward_lerp` | 0.15 |
| `AMP/max_amp_component` | 0.015 |
| `AMP/mean_raw_reward` | `[0, 1]` |
| `AMP/mean_scaled_discriminator_reward` | `[0, 0.1]` |
| `AMP/mean_amp_component` | `[0, 0.015]` |
| `AMP/valid_transition_fraction` | 大于 0 且不超过 recovery 活跃比例 |
| `AMP/replay_size` | 上升后在 50000 饱和 |

任何 NaN、raw reward 超出 `[0,1]`、AMP component 超过 `0.015`，都应立即停止训练排查。

## 5. 判别器指标

重点观察：

- `AMP/discriminator_loss`：应保持有限，不能持续爆炸或 NaN；
- `AMP/gradient_penalty`：应有限，不能单调爆炸；
- `AMP/expert_score`：总体应高于 policy score，并朝 expert target `+1` 移动；
- `AMP/policy_score`：初期通常朝 policy target `-1` 移动；
- `AMP/mean_raw_reward`：不能长期全部为 0，也不能在判别器完全没有区分度时无解释地固定不变。

不要要求 expert/policy score 必须精确等于 `+1/-1`：gradient penalty、不断变化的 policy 数据和归一化都会使
它们在目标附近动态平衡。

## 6. Recovery 任务指标

最重要的是 AMP 加入后不能破坏已有起身能力：

1. `Metrics/motion/recovery_success_rate`：应至少保持基线，理想情况下上升；
2. `Metrics/motion/recovery_failure_rate`：不能持续上升；
3. `Metrics/motion/recovery_duration_s`：不能明显变长；
4. `Metrics/motion/recovery_timeout_torso_height`：失败时的 torso height 不应下降；
5. `Metrics/motion/recovery_active_torso_height`、`torso_uprightness`：应逐渐改善；
6. `Metrics/motion/recovery_active_feet_stable`：接近站立阶段应上升；
7. `Episode_Reward_Group/recovery`：应稳定，不能因 0.85 插值突然长期崩塌；
8. `Episode_Reward_Group/shared_regularization`：绝对值不能显著恶化，否则可能用更剧烈动作换取 AMP 风格。

同时检查 tracking：recovery 成功后同一 Actor 会回到 reference tracking，不能只看起身画面而忽略 tracking
误差、hard failure 和 episode mean reward。

## 7. PPO 稳定性

至少关注：

- `Loss/value_function`：不能出现持续数量级增长；
- `Loss/surrogate`：应保持有限，不能突然发散；
- `Policy/mean_noise_std`：不能快速坍缩到接近 0 或持续暴涨；
- `Train/mean_reward`、`Train/mean_episode_length`：加入 AMP 后允许短暂变化，但不应持续恶化。

建议在 iteration 50、200、500、1000 分别保存或录制同一组倒地初态，对比动作是否更接近专家，同时核对
成功率和恢复时长。风格改善必须建立在 recovery 完成率没有明显下降的前提上。

## 8. TensorBoard

```bash
source /data0/Software/anaconda3/etc/profile.d/conda.sh
conda activate /data0/alan/conda/envs/lj-mimic-z1
tensorboard --logdir logs/rsl_rl/z1_flat --port 6006
```

浏览器访问 `http://localhost:6006`。正式训练 run 名称为
`recovery_amp_85_15_resume_18499_6000`。
