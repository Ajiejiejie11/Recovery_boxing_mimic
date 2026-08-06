# Recovery-only AMP V2 实施与验证报告

日期：2026-08-05

## 1. 最终行为

- 保留现有 Actor、Critic、PPO、recovery 状态机、RewardManager 和 tracking 逻辑。
- AMP 只使用 `recovery_active_before_step & ~done` 的 transition；tracking、tracking 进入 recovery 的边界步、done 步都不写 replay，也不混合 AMP reward。
- recovery 步的最终 reward 为：

  ```text
  r_style = min(softplus(D(s, s')) - softplus(-1), 1.0)
  r_recovery = 0.75 * r_task + 0.25 * (0.1 * r_style)
  ```

  因此 AMP 分量的上界是 `+0.025`，下界是约 `-0.0078315`。`D=-1` 时 reward 为 0，`D<-1` 时仍有平滑的负值排序和非零导数，不再存在旧 clamp 的硬死区。
- 这个 0.75/0.25 是对 RewardManager 已经乘过 `dt=0.02` 的 task reward 与 runner 侧 AMP reward 做混合，与 AMP_mjlab 的 reward 尺度形式一致。

## 2. Expert 与 Policy 数据流

Expert 目录共读取 25293 条 transition：

- `boxing_walk_001_get_ready_370_530.npz`：66 条，作为每个有效 expert batch 的强制配额。
- 3 个 fall-and-get-up 文件：各 8409 条，共 25227 条随机池。
- 每次 24576 条 expert 采样为 `66 条完整目标 transition + 24510 条从其他文件有放回随机采样`，然后打乱。
- 66 条配额在每个 batch 中的占比约为 0.2686%。它保证覆盖，但不等价于高权重过采样。

Policy 侧使用一个容量 100000 的环形 replay buffer，仅写入 recovery 有效 transition；采样使用 `torch.randint`，因此是有放回随机采样。

4096 环境下由 PPO 配置自动推导：

```text
rollout transitions       = 4096 * 24 = 98304
effective AMP batch       = 98304 / 4 = 24576
discriminator updates     = 5 * 4 = 20
policy samples/iteration  = 24576 * 20 = 491520
expert samples/iteration  = 24576 * 20 = 491520
```

24576 是一次 optimizer step 的有效 batch；为控制显存，内部按 2048 分成 12 个 micro-batch 累积梯度，最后只调用一次 optimizer step。

## 3. Discriminator 与 GP

- 网络：`390 -> 256 -> 128 -> 1`，ELU。
- LSGAN target：expert `+1`，policy `-1`。
- Adam learning rate：`3e-5`。
- 每个 micro-batch 对完整、已归一化的 transition pair 做插值：

  ```text
  x_exp = concat(s_exp, s_exp_next)
  x_pol = concat(s_pol, s_pol_next)
  x_mix = alpha * x_exp + (1 - alpha) * x_pol
  GP = 5 * (||grad D(x_mix)||_2 - 1)^2
  ```

- GP 样本不作为 expert/policy label，不直接生成 Actor reward，只约束两个分布之间的判别几何。
- discriminator 参数梯度的 global norm 最后被 clip 到 1.0。

## 4. Checkpoint 行为

- 首次从 baseline `model_16000` 启动时，恢复 Actor、Critic、observation normalizer、iteration 和 motion curriculum。
- baseline checkpoint 没有 Recovery AMP 状态，所以 discriminator、AMP normalizer 和 AMP optimizer 从头初始化。
- 因 reward 定义改变，默认不恢复 baseline PPO optimizer state。
- 原 `model_16000.pt` 是在逻辑 `cuda:5` 保存的。已生成等价的 CPU-stored `model_16000_portable.pt`，便于在任意单卡映射下加载。
- 新 V2 checkpoint 保存 discriminator、AMP optimizer、normalizer 和 formulation 版本。旧版 AMP reward/GP checkpoint 不会被静默加载。
- 若中断后从新 V2 checkpoint 续训，应将 `agent.recovery_amp.resume_ppo_optimizer=True`，以同时恢复 PPO 和 AMP optimizer。

## 5. 验证结果

### 纯数值测试

- Python compile、纯数值测试和 `git diff --check` 通过。
- 数据集实测：25293 条、195 维 state、66 条强制目标 transition、25227 条随机池。
- 平滑 reward 数值实测：`D=[-10,-2,-1,0,1,2]` 对应约 `[-0.3132,-0.1863,0,0.3799,1,1]`，且 `D<-1` 仍可反传。
- 小规模 sidecar 测试完成 replay、分层 expert sampling、micro-batch 梯度累积、插值 GP 和 optimizer step。

### 真实 Isaac Lab GPU 短跑

128 环境短跑完成一轮，AMP Adam 所有参数状态均到 step 20，checkpoint 和 ONNX 正常导出。

4096 环境完整规模短跑完成一轮：

```text
effective_batch_size                  24576
micro_batch_size                       2048
updates                                  20
replay_size after first rollout       39312
required_expert_transitions_per_batch    66
valid_transition_fraction            0.3999
collection time                       3.301 s
PPO + AMP learning time               1.366 s
total iteration time                  4.67 s
```

首轮判别指标：

```text
expert score mean          -0.0306
policy score mean          -0.0890
required clip score mean    0.5595
GP gradient norm mean       0.8493
weighted GP loss            0.1253
style reward p10/p50/p90   -0.0246 / 0.2188 / 0.6382
negative style fraction     0.1249
```

这只证明第一轮数值健康、未瞬间饱和，不代表动作质量已经改善。

## 6. 正式训练指令

```bash
cd '/home/alan/Project/walking-boxing-amp（recovery）/magicbot-mimic'
source /data0/Software/anaconda3/etc/profile.d/conda.sh
conda activate /data0/alan/conda/envs/lj-mimic-z1
export PYTHONPATH="$PWD/source/whole_body_tracking:${PYTHONPATH}"

CUDA_VISIBLE_DEVICES=3 python scripts/rsl_rl/train.py \
  --task Tracking-Flat-Z1-Recovery-AMP-v0 \
  --motion_dir "$PWD/source/whole_body_tracking/whole_body_tracking/datasets/boxing-dataset-magiclab-z1/train_npz" \
  --num_envs 4096 \
  --max_iterations 6000 \
  --headless \
  --device cuda:0 \
  --resume True \
  --experiment_name z1_flat \
  --load_run 2026-07-31_20-55-24_boxing_recovery_height35_refbridge_feet01_resume_6000 \
  --checkpoint model_16000_portable.pt \
  --run_name recovery_amp_v2_75_25_from16000_6000
```

`CUDA_VISIBLE_DEVICES=3` 中的 3 是本次实测时的空闲物理 GPU；正式开训前应按当时占用情况替换。

## 7. 优先监控项

1. `AMP/expert_score` 与 `AMP/policy_score`：不应很快固定在 `+1/-1` 两端。
2. `AMP/style_reward_p10/p50/p90` 和 `AMP/negative_style_fraction`：用来看 reward 是否仍有分辨率，以及是否整体滑向过度负值。
3. `AMP/gradient_norm` 和 `AMP/gradient_penalty`：前者应在 1 附近，后者不应持续爆炸。
4. `AMP/required_expert_score`：单独确认 boxing get-ready 片段没有被其他 fall/get-up 数据淹没。
5. `AMP/replay_size`、`AMP/valid_transition_fraction` 和 `AMP/updates`：预期 replay 很快到 100000，recovery 比例约 0.4，每轮更新 20 次。
6. `AMP/observed_amp_fraction`：这是按实际绝对分量计算的影响占比，不会机械等于配置中的 25%。
7. recovery 任务指标：继续优先看 recovery success/completion rate、duration、torso height/uprightness 和实际动作视频，确保风格改善没有牺牲起身成功率。

可使用：

```bash
tensorboard --logdir logs/rsl_rl/z1_flat --port 6006 --bind_all
```

查看全部指标。
