# 拳击 Reference Tracking + 倒地起身：训练与 RB 部署说明

## 1. 最终目标

当前任务使用一套共享 Actor，同时学习两种行为：

1. 正常阶段跟踪拳击 reference。
2. 倒地阶段从真实跌倒姿态恢复到稳定站立。

Actor 不接收 `recovery flag`、task token、躯干高度或接触真值。它只能根据实机可获得的输入（reference、关节状态、角速度、历史动作以及投影重力）隐式判断自身状态。训练环境内部显式维护 tracking/recovery 阶段，并据此切换奖励。

实现不使用 AMP，也没有额外 gating 网络或第二套 Actor。

## 2. 环境分配

并行环境初始化时固定划分：

- 40%：`delay_env_mask=True`，具备延迟终止资格，并在每次 reset 时从跌倒数据初始化，首先训练 recovery。
- 60%：`delay_env_mask=False`，从拳击 reference 初始化，只训练正常 tracking；物理跌倒后立即终止。

40% 只是固定的“延迟资格”，不等于环境始终处于 recovery。动态阶段由另一个变量表示：

```text
is_recovering = recovery_active
is_tracking   = not recovery_active
```

具备延迟资格的环境成功站起后不会 reset，而是关闭 `recovery_active`，采样一条新的拳击 reference，并在下一控制步进入正常 tracking。它以后若再次物理跌倒，会再次打开 recovery 窗口。

60% tracking 环境内部按 `0.625 / 0.125 / 0.25` 分配，对全部环境的全局占比为：

- 37.5% 全局覆盖采样；
- 7.5% hard-failure replay；
- 15% tracking-error/soft-failure replay。

Recovery 成功后 delayed 环境会动态转入 tracking，因此运行时实际 tracking 比例通常高于固定的 60% reset 比例。

异步 reset 时不能对当前小批次数量分别四舍五入，否则 `count=1` 会永远落到 coverage。当前实现维护跨 reset 的长期通道配额，使连续单环境 reset 也收敛到上述比例。该调度器只选择 coverage/hard/soft 采样通道，不判定失败类型：hard/soft 标签和 EMA score 仍完全来自实际 tracking rollout。冷启动尚无对应失败证据时，hard/soft 配额临时回退到 coverage，观察到真实失败后才按 score replay。

## 3. 状态机和执行时序

```text
reset
├── 普通 60% ──> TRACKING
└── 延迟 40% ──> FALLEN RESET ──> RECOVERY

TRACKING
├── 普通环境发生物理跌倒 ──> terminate/reset
├── 延迟环境发生物理跌倒 ──> RECOVERY（不 reset）
└── reference 结束/其他 tracking 终止 ──> reset

RECOVERY
├── 满足稳定站立 ──> 不 reset，采样新 boxing reference ──> TRACKING
└── 连续 6 s 未站稳 ──> terminate/reset
```

Tracking 和 recovery 使用独立计时：`tracking_steps` 只在 tracking 增加，`recovery_steps` 只在 recovery 增加。成功后新 boxing reference 获得新的 tracking 时间预算。因此即使机器人在原 episode 很晚时再次跌倒，也不会被原来的 20 秒全局 timeout 抢先截断，仍然拥有完整 300 控制步的 recovery 窗口。

状态更新放在 termination manager 的第一个非 timeout term 中，并早于 reward manager 执行。因此，跌倒被检测到的同一个控制步就会关闭 tracking 奖励、打开 recovery 奖励，不会出现同一步两组任务奖励同时生效。

恢复成功先标记为 `success_pending`。该步仍按 recovery 计奖；reward 计算结束后 command manager 才关闭 recovery、采样新 reference。这样不会在成功边界突然用一条刚采样的 reference 计算 tracking 误差。

## 4. 进入、退出和超时条件

### 进入 recovery

训练环境使用 `torso_link` 的物理状态：

```text
fallen = torso_height < 0.50 m
         OR torso_uprightness < 0.342
```

`0.342 ≈ cos(70°)`，即倾斜超过约 70 度也认为跌倒。这里的高度是 `torso_link` 原点相对当前环境地面的高度，不是 pelvis 高度。

低高度入口采用 `0.50 m` 而不是 `0.55 m`：现有拳击 reference 中合法深蹲姿态的最低 `torso_link` 高度约为 `0.546 m`，并且 reset 还会叠加高度噪声。`0.50 m` 能避免把合法低姿态误判为跌倒；明显侧倒、仰倒仍会由倾角条件立即触发。

Delayed 环境向下塌落时，reference-relative anchor 高度误差不会提前终止；它必须由上述绝对高度/倾角判据进入 recovery。普通 60% tracking 环境以及 delayed 环境向上偏差仍保留原 anchor 高度终止。这避免高位 reference 在机器人尚未降到 `0.50 m` 前先把 delayed 环境 reset，同时不放松普通 tracking 的失败条件。

### 退出 recovery

必须同时满足：

```text
torso_link height >= 0.75 m
torso uprightness  >= 0.85
左右脚连续接触     >= 0.15 s
左右脚平面速度     <= 0.20 m/s
```

进入阈值比退出阈值宽松，形成迟滞区，避免机器人临界晃动时在 tracking/recovery 之间频繁抖动。

站立 reference 的第 64 帧中，`torso_link` 原点高度约为 `0.7934 m`，pelvis 原点高度约为 `0.7315 m`，因此当前 `0.75 m` 审核阈值与所选站姿匹配，并留有约 4.3 cm 余量。

### 六秒上限

六秒计数从每次进入 recovery 时重新开始，不使用整个 episode 的长度。达到六秒仍未满足稳定站立条件就终止；没有额外 `-10` 惩罚。成功也没有 `+10`，且没有“越早站起奖励”。失败只通过 episode 终止和丢失后续正回报体现。

## 5. Reference 在两个阶段的处理

### Tracking

按原拳击数据正常输出 reference，并正常推进时间帧。

### Recovery

环境自动把 reference 替换为独立保存的交叉双手默认站姿：

```text
source/whole_body_tracking/whole_body_tracking/datasets/recovery_targets/
boxing_walk_001_get_ready_370_530.npz
```

使用第 64 帧，关节速度和身体目标速度设为零；目标 yaw 与机器人当前 yaw 对齐，只要求恢复直立，不强迫倒地机器人先旋转到采集数据的世界朝向。Recovery 期间 reference 不推进。

恢复成功后不继续旧 reference，而是采样一条新的拳击 reference。这与 RB 部署时“检测到跌倒就停止当前 reference、发送默认站姿”相匹配，也避免恢复后追赶已经过期的动作相位。

## 6. 奖励分组和尺度

最终每一步只有下面两种形式之一：

```text
TRACKING: r = r_tracking + r_shared
RECOVERY: r = r_recovery + r_shared
```

### Tracking-only 组

- anchor position tracking：最大权重 `0.5`
- anchor orientation tracking：最大权重 `0.5`
- body position tracking：最大权重 `1.0`
- body orientation tracking：最大权重 `1.0`
- body linear velocity tracking：最大权重 `1.0`
- body angular velocity tracking：最大权重 `1.0`
- `undesired_contacts`
- `hand_slip`

六项正向 tracking 奖励的理论最大和为 `5.0`。`undesired_contacts` 和 `hand_slip` 只在 tracking 开启，因为起身时手掌、前臂、膝盖或身体接触地面可能是必要动作。

### Recovery-only 组

- `recovery_upright`：权重 `1.5`；使用 `clamp(uprightness, -1, 1)`。正确竖直为 `+1`，水平为 `0`，头朝下的反向竖直为 `-1`，从而避免错误方向进入零梯度区。
- `recovery_height`：权重 `3.5`，在 `0.75 m` 饱和。
- `recovery_feet_stable`：权重 `0.10`。它仍使用“左右脚接触持续至少 `0.15 s` 且平面速度不超过 `0.20 m/s`”的判定，但额外乘以晚期 recovery 门控，不能在躺倒时刷取。
- `recovery_lower_body_reference`：权重 `0.15`，使用 `exp(-MSE/0.5²)` 比较髋、膝、踝及腰部与默认双手交叉站姿 reference 的关节角；手臂不参与，避免妨碍撑地起身。
- `recovery_torso_reference`：权重 `0.15`，使用 `exp(-orientation_error²/0.4²)` 对齐默认站姿的 torso roll/pitch。目标 yaw 始终跟随机器人当前 yaw，不约束全局朝向、位置或速度。

`recovery_feet_stable` 和两个 reference bridge 共用平滑晚期门控：torso 高度从 `0.55 m` 到 `0.70 m` 由 0 增至 1，uprightness 从 `0.30` 到 `0.80` 由 0 增至 1，最终门值是两者乘积。这样起身早期主要由高度和方向引导，接近站稳后才整理双脚、下肢和 torso 姿态。

Recovery 正向奖励的理论最大和为 `3.5 + 1.5 + 0.10 + 0.15 + 0.15 = 5.40/s`。其中后 `0.40/s` 只在 late recovery 生效，而且满足成功条件后 recovery 会立即退出，不能长期停留刷取。`both_feet_stable` 同时继续作为成功退出条件、Critic 特权信息和日志指标。这里没有成功 bonus、失败 penalty 或提前成功 bonus。

### 两个任务共享的正则项

- action rate
- joint position limit
- joint velocity limit
- torque limit
- foot contact force
- `foot_slip`，权重 `-0.3`
- 有界 `self_collision`，权重 `-0.5`

`foot_slip` 在 recovery 仍保留，用来抑制双脚打滑；`self_collision` 也共享，但只监控经过筛选的非相邻危险碰撞对，并把最强碰撞归一化到 `[0, 1]`，避免碰撞对数量改变奖励尺度。交叉双手站姿需要的手-手/手-躯干接触没有被加入该惩罚。

运行时每 100 个控制步验证 tracking/recovery mask 互斥且完备。TensorBoard 额外输出 `Episode_Reward_Group/tracking`、`recovery` 和 `shared_regularization`；它们只汇总 RewardManager 已计算的分项，不会再次加入总 reward。

## 7. Actor 与 Critic

### Actor：单一网络，无任务标志

旧 Actor 输入为 124 维，新输入为 127 维：

```text
command                    46
motion_anchor_ori_b         6
base_ang_vel                3
joint_pos                  23
joint_vel                  23
last_action                23
projected_gravity           3
-----------------------------
total                     127
```

投影重力追加在旧 124 维之后，来自 base/pelvis IMU 姿态：世界重力单位向量转换到机体坐标系。Actor 输入中没有：

- `recovery_active`
- `delay_env_mask`
- 躯干/基座绝对高度
- 脚接触真值
- 任务 token

因此训练策略可直接迁移到无法获得全局高度的真机。

### Critic：一套共享的 privileged critic

没有采用两套 Critic。当前共享 Critic 包含原有 256 维状态和四个训练特权量，总计 260 维：

```text
recovery_active
recovery_progress
torso_link height
both_feet_stable
```

两套 Critic 会带来样本路由、GAE/value target 切换、恢复边界 bootstrap 和 checkpoint 兼容方面的额外复杂度。当前两个任务的正奖励尺度已经接近，并且共享 Critic 能看到精确阶段及物理完成度，足以解决“同样是站立状态，但 recovery 与 tracking 的价值不同”的 value aliasing。建议先以训练曲线验证该方案；只有在两阶段 value loss 长期明显分叉时再考虑双 value head，而不是直接上两个完全独立 Critic。

Recovery 日志使用专属累计计数，普通 tracking reset 不进入成功率、失败率和平均时长的分母。关键字段包括 entry/success/failure count、completion/success/failure rate、真实 recovery mean duration、timeout torso height，以及 delayed/active ratio。采样日志同时给出调度目标比例和包含冷启动 fallback 的实际使用比例。

## 8. 数据

### Tracking 数据

```text
source/whole_body_tracking/whole_body_tracking/datasets/
boxing-dataset-magiclab-z1/train_npz
```

当前目录只保留 8 个拳击动作文件，已按任务要求删除走路片段。

### Recovery reset 数据

```text
source/whole_body_tracking/whole_body_tracking/datasets/
fall_recovety/prepare_stand_slice_06/train_npz
```

目录包含由截断后 pkl 数据重新转换得到的 19 个 NPZ，共 4071 帧。转换后的 `torso_link` 相对地面高度范围约为 `0.0168–0.6194 m`；reset 筛选上限设为 `0.62 m`，与当前数据切片一致。

## 9. 域随机化

保留摩擦、质量、COM、执行器 Kp/Kd、初始状态和观测噪声随机化。外力扰动已缩小为 root velocity perturbation：

```text
x/y:          [-0.15, 0.15] m/s
z:            [-0.05, 0.05] m/s
roll/pitch:   [-0.15, 0.15] rad/s
yaw:          [-0.20, 0.20] rad/s
```

这比原配置温和，避免训练初期过大的 push 同时妨碍拳击 tracking 和起身动作形成。

## 10. 从头训练命令

本方案使用随机初始化，从头联合学习拳击 tracking 和 recovery；不加载任何旧 checkpoint，也不继承旧 optimizer 或 curriculum 状态。

在已配置 Isaac Lab Python 环境的仓库根目录执行：

```bash
python scripts/rsl_rl/train.py \
  --task Tracking-Flat-Z1-v0 \
  --motion_dir source/whole_body_tracking/whole_body_tracking/datasets/boxing-dataset-magiclab-z1/train_npz \
  --num_envs 8192 \
  --max_iterations 3000 \
  --headless \
  --logger tensorboard \
  --run_name boxing_dynamic_recovery_3000
```

如果指定 GPU，再增加例如 `--device cuda:0`。开始完整训练前，建议先跑 1–5 轮并确认：Actor/critic 维度为 `127/260`、40% reset 为 recovery、普通环境跌倒立即 reset、延迟环境跌倒不 reset、成功环境切回 tracking、两组 reward 不同时非零。

## 11. RB 真机部署建议

RB 侧也维护一个简单的 `TRACKING/RECOVERY` 状态，但不把这个状态输入 Actor；它只负责选择发送哪一种 reference。

### 进入 recovery

1. 把 IMU 四元数转换为归一化投影重力 `g_b`，确保坐标系和训练一致。
2. 计算 `uprightness = -g_b.z`。
3. 当倾斜超过阈值并持续约 `0.10–0.20 s` 后进入 recovery，避免单帧 IMU 尖峰误触发。可先以训练阈值 `uprightness < 0.342` 为基线，再根据真机日志调整。
4. 立即停止推进/发送当前拳击 reference，改发第 64 帧交叉双手默认站姿：目标关节速度为零，目标 roll/pitch 直立，目标 yaw 与机器人当前 yaw 对齐。

真机不需要也不应伪造训练中的 `torso_height`。高度只属于环境和 Critic 的训练特权信息。

### 退出 recovery

真机没有全局高度时，建议用可测信号替代训练审核条件：

- `uprightness >= 0.85`；
- base angular velocity 足够小；
- 关节速度整体足够小；
- 若有脚底力/触地传感器，要求双脚稳定接触；若没有，可用足端运动学、关节力矩或电流构造保守接触估计；
- 上述条件连续保持约 `0.30–0.50 s` 后才退出，形成时间迟滞。

退出时建议先继续保持默认站姿短暂稳定，再从 `get_ready` 等安全拳击帧重新同步 reference；不要恢复到跌倒前已经过期的拳击相位。

### 安全保护

- RB 侧保留独立的动作、速度、关节限位和力矩裁剪，不依赖策略自觉满足安全边界。
- recovery 超过 6 秒仍未稳定时，不要无限重复大动作；切到阻尼/安全姿态并由上层决定重试或人工介入。
- 实机首次测试应降低动作 scale/力矩上限、使用吊架或保护绳，并逐步放开。
- 记录每次进入/退出时间、投影重力、关节状态、动作和 reference 模式，重点排查状态抖动以及训练/实机坐标系符号不一致。

## 12. 核心代码位置

- 动态资格、recovery 状态机、reset 数据和成功后新 reference：`tasks/tracking/mdp/commands.py`
- 进入/退出物理判据和延迟终止：`tasks/tracking/mdp/terminations.py`
- tracking/recovery 奖励门控和接触正则：`tasks/tracking/mdp/rewards.py`
- Critic 特权状态：`tasks/tracking/mdp/observations.py`
- recovery 条件统计与三组 reward 只读汇总：`tasks/tracking/tracking_env.py`
- Actor/Critic 观测布局与通用 reward/termination 项：`tasks/tracking/tracking_env_cfg.py`
- Z1 比例、阈值、数据路径、奖励尺度、push 和碰撞传感器：`tasks/tracking/config/z1/flat_env_cfg.py`
- MuJoCo 127 维部署观测：`scripts/sim2sim_mujoco.py`
