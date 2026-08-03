# Z1 Recovery + Tracking MuJoCo 简明手册

## 1. 用途

该工具用于在 MuJoCo 中验证同一个 ONNX 策略的：

- 倒地起身（recovery）；
- 拳击动作跟踪（tracking）；
- 起身后连续执行多轮键盘指定动作。

相关文件：

- `scripts/evaluate_recovery_tracking_mujoco.py`：仿真与评估主脚本；
- `scripts/run_mujoco_recovery_tracking.sh`：固定使用 `dance_stop` 环境的启动器；
- `docs/boxing_recovery_training_and_rb_deployment.md`：训练和部署设计。

## 2. 启动方式

在仓库根目录运行：

```bash
./scripts/run_mujoco_recovery_tracking.sh --help
```

启动器固定调用：

```text
/home/user/miniconda3/envs/dance_stop/bin/python
```

因此即使当前终端显示 `(base)`，也建议使用该启动器，不要直接使用裸 `python`。

模型有两种指定方式，不能同时使用：

```bash
# 自动加载训练目录 exported/ 下的 ONNX
--run-dir logs/rsl_rl/z1_flat/YOUR_RUN_NAME

# 精确指定 ONNX
--policy path/to/policy.onnx
```

脚本不能直接加载任意 `.pt`；需要先将对应 checkpoint 导出为 ONNX。

## 3. 推荐：键盘连续选择动作

```bash
./scripts/run_mujoco_recovery_tracking.sh \
  --run-dir logs/rsl_rl/z1_flat/2026-08-01_20-36-36_boxing_recovery_fullbody_resume19000_remaining5499 \
  --mode interactive \
  --fall-file fallAndGetUp1_subject1_f2834_f2982.npz \
  --fall-frame 168
```

状态机：

```text
RECOVERY
  → WAITING
  → 按 1-8
  → TRACKING（动作播放一次）
  → RETURN_TO_STAND（返回标准站姿）
  → WAITING
  → 可继续输入 1-8
```

出现以下提示后才接受数字：

```text
WAITING: robot is stable. Press 1-8 ...
```

可以在终端直接按键（无需回车），也可以让 MuJoCo viewer 获得焦点后按键。

| 按键 | 动作 | Reference 文件 |
|---:|---|---|
| 1 | 嘲讽 | `boxing_chaofeng_005_*_clip.npz` |
| 2 | 上勾拳 | `boxing_shanggouquan_001_*_clip.npz` |
| 3 | 膝踢 | `boxing_xiti_001_*_clip.npz` |
| 4 | 叶问盾 | `boxing_yewendun_002_*_clip.npz` |
| 5 | 右侧踢 | `boxing_youceti_001_*_clip.npz` |
| 6 | 直拳刺探 | `boxing_zhiquancitan_*_clip.npz` |
| 7 | 组合拳 | `boxing_zuhequan_003_[0-9]*_clip.npz` |
| 8 | 组合拳 A | `boxing_zuhequan_003_A_*_clip.npz` |

动作结束后会看到：

```text
reference 2 finished; RETURN_TO_STAND using the standard standing target
READY for next command after 0.50s return; press 1-8
```

看到 `READY` 后可以输入下一条指令。退出使用 `Ctrl+C`。

交互模式中：

- 每个 reference 按文件长度完整播放一次；
- `--tracking-seconds` 不会截断动作；
- `--tracking-loop` 会被忽略；
- tracking 中跌倒会重新进入 recovery；
- 只有返回标准站姿并稳定后才接受下一次按键。

返回站姿参数：

```bash
--return-min-time 0.5       # 最短稳定时间
--return-joint-rmse 0.35   # 相对标准站姿的最大关节 RMSE，单位 rad
```

例如希望标准站姿多保持一会：

```bash
./scripts/run_mujoco_recovery_tracking.sh \
  --run-dir logs/rsl_rl/z1_flat/2026-08-01_20-36-36_boxing_recovery_fullbody_resume19000_remaining5499 \
  --mode interactive \
  --fall-file fallAndGetUp1_subject1_f2834_f2982.npz \
  --fall-frame 168 \
  --return-min-time 1.5
```

## 4. 其他测试模式

### Combined：起身后自动跟踪

```bash
./scripts/run_mujoco_recovery_tracking.sh \
  --run-dir logs/rsl_rl/z1_flat/YOUR_RUN_NAME \
  --mode combined
```

Recovery 成功后不重置 MuJoCo，直接切换到 tracking。

### Recovery：批量测试起身

```bash
./scripts/run_mujoco_recovery_tracking.sh \
  --run-dir logs/rsl_rl/z1_flat/YOUR_RUN_NAME \
  --mode recovery \
  --trials 20 \
  --no-render \
  --no-real-time
```

重点查看 `recovery_success_rate` 和 `mean_successful_recovery_time_s`。

### Tracking：批量测试拳击动作

```bash
./scripts/run_mujoco_recovery_tracking.sh \
  --run-dir logs/rsl_rl/z1_flat/YOUR_RUN_NAME \
  --mode tracking \
  --trials 8 \
  --tracking-seconds 8 \
  --no-render \
  --no-real-time
```

默认 `--motion-glob '*_clip.npz'`，会覆盖当前目录中的 8 个拳击 reference。

## 5. 精确指定数据

固定倒地文件、倒地帧和 tracking reference：

```bash
./scripts/run_mujoco_recovery_tracking.sh \
  --run-dir logs/rsl_rl/z1_flat/YOUR_RUN_NAME \
  --mode combined \
  --fall-file fallAndGetUp1_subject1_f2834_f2982.npz \
  --fall-frame 168 \
  --motion source/whole_body_tracking/whole_body_tracking/datasets/boxing-dataset-magiclab-z1/train_npz/boxing_yewendun_002_500_1150_clip.npz
```

注意：

- `--fall-file` 只需要文件名；
- `--fall-file` 和 `--fall-frame` 必须同时提供；
- `--tracking-start-frame` 可指定 reference 起始帧；
- `--cycle-fall-files` 可在多 trial 中轮换倒地文件；
- 默认会对齐 recovery 后机器人与 tracking reference 的 yaw。

## 6. 扰动测试

```bash
./scripts/run_mujoco_recovery_tracking.sh \
  --run-dir logs/rsl_rl/z1_flat/YOUR_RUN_NAME \
  --mode combined \
  --trials 20 \
  --tracking-seconds 12 \
  --push-time 2.0 \
  --push-linear 0.6,0,0 \
  --recover-after-tracking-fall \
  --no-render \
  --no-real-time
```

`--push-angular 0,1.0,0` 可增加角速度扰动。

## 7. 输出

结果默认写入独立时间戳目录：

```text
outputs/mujoco_recovery_tracking/<timestamp>_<mode>/
├── summary.json
└── trial_000.csv
```

常用汇总指标：

- `recovery_success_rate`；
- `mean_successful_recovery_time_s`；
- `tracking_pass_rate`；
- `tracking_fall_rate`；
- `selected_reference_keys`；
- `completed_reference_count`；
- `return_to_stand_durations_s`。

Tracking 通过要求无跌倒，并满足预设的 anchor/body 平均误差阈值。没有跌倒不代表 tracking 一定通过。

## 8. 实现要点

- Actor 输入：127 维；
- policy 周期：`0.02 s`；
- MuJoCo 步长：`0.002 s`；
- recovery target：默认 frame 64；
- recovery→tracking 默认进行 yaw 对齐；
- 关节和 body 按 ONNX/NPZ 名称映射。

## 9. 常见问题

### 缺少 `mujoco` 或 `onnxruntime`

使用封装启动器，不要使用 `(base)` 环境中的裸 `python`：

```bash
./scripts/run_mujoco_recovery_tracking.sh --help
```

### 按数字没有反应

必须等待 `WAITING` 或 `READY`。终端和 viewer 只有当前获得焦点的一方能收到按键。

### Viewer 没出现

确认没有传入 `--no-render`，并检查当前会话是否有图形显示权限。

### Viewer 播放太快

可视化时不要使用 `--no-real-time`。

### Interactive 一直不结束

这是预期行为。它会持续等待下一次 `1-8`，使用 `Ctrl+C` 退出。

## 10. 安全说明

MuJoCo 通过只表示 Sim2Sim 控制链路能够工作，不等同于真机安全验证。真机测试仍需吊架、限位、力矩限制、
急停和逐级增加扰动。
