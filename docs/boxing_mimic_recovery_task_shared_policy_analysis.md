# Boxing：Mimic、Recovery 与 Task 速度跟踪的统一训练分析

本文根据项目代码、Z1 环境配置以及训练日志 logs/rsl_rl/z1_flat/2026-08-18_22-06-52_boxing_task_recovery_8192x15000 整理。只做结构和结果说明，不改变训练代码或模型文件。

## 结论

这次训练使用的是一个共享的 PPO Actor-Critic 策略，处理三种行为阶段：

1. Mimic / reference tracking：跟踪 boxing 动作参考。
2. Recovery：从跌倒姿态恢复到站立稳定状态。
3. Task velocity tracking：根据速度指令跟踪平面线速度和偏航角速度，同时保持直立和 guard 姿态。

这里的“合到一个网络”应理解为一个策略网络根据观测和阶段命令输出动作，而不是三个独立策略拼接。Actor 和 Critic 仍是两个功能不同的网络；Critic 还用了特权状态帮助训练，部署时不需要这些输入。

## 三类行为如何组织

### 环境槽位分配

Z1 配置里，Recovery 占 40%；非 Recovery 槽位里 Task 占一半、Mimic 占一半，因此全局约为 40% Recovery + 30% Task + 30% Mimic。Recovery 成功后，环境不会立即 reset，而是切回 Mimic 或 Task；切回 Task 的概率是 0.5。

### Mimic 的困难采样

参考动作跟踪槽位使用分桶和重放采样：

- coverage sampling：0.625
- hard-failure replay：0.125
- tracking-error replay：0.25

动作文件按约 1 秒切成 bin，采样器会根据跟踪误差和失败统计调整难度。

## 网络和观测结构

训练使用标准 RSL-RL PPO：

- Actor：512-256-128，ELU
- Critic：512-256-128，ELU
- PPO：5 个 epoch、4 个 mini-batch、gamma=0.99、lambda=0.95

Actor 的策略观测包含参考命令/姿态信息、角速度、关节相对位置和速度、上一动作、重力方向、基座线速度以及 Task 速度命令。Task 阶段的命令为 [vx, vy, yaw_rate]；非 Task 阶段该命令置零。Recovery 不依赖单独的 Recovery 网络，而是由当前身体状态、参考/目标信息和阶段奖励共同驱动同一个 Actor。

Critic 额外看到特权信息，例如完整参考身体状态、阶段状态、进度、躯干高度和脚部稳定性。这属于 asymmetric actor-critic：训练时帮助价值估计，部署时不要求这些仿真特权量。

## 奖励并不是三套策略的简单相加

代码通过 phase mask 让主要奖励按阶段互斥：

- Mimic 阶段激活 reference tracking 奖励；
- Recovery 阶段激活 upright、height、feet stability 和 recovery reference 奖励；
- Task 阶段激活线速度、角速度、躯干直立和上肢姿态奖励；
- 动作变化、关节/力矩限制、接触和脚滑等 regularization 在阶段间共享。

因此总奖励可以理解为“当前阶段的目标奖励 + 跨阶段的安全/平滑约束”。

## 本次 15000 iteration 训练结果

从 TensorBoard 日志看，训练总体稳定并取得了明显的站立恢复能力：

| 指标 | 训练初期 | 训练末期/后段均值 | 解读 |
|---|---:|---:|---|
| 平均 reward | 约 -1.7 | 约 25--27 | 总体学习有效，后段趋于平台 |
| 平均 episode 长度 | 16.7 | 约 510 | 早期快速失败显著减少 |
| 躯干直立度 | 0.35 | 约 0.985 | 站立姿态已经较稳定 |
| 躯干高度 | 0.30 m | 约 0.50 m | Recovery 的起身能力明显改善 |
| Recovery success rate | 0 | 约 11.2% | 仍是主要瓶颈 |
| bin success | 0 | 约 19% | 只有部分困难片段真正完成 |
| bin hard failure | 约 54% | 约 0.2--0.4% | 硬崩溃大幅下降 |
| body position error | 0.19 | 约 1.2 | 全身动作跟踪仍偏粗 |
| joint position error | 1.39 | 约 2.7 | 关节模仿尚未达到理想水平 |
| policy noise std | 0.99 | 约 0.22 | 已从探索转向较确定的策略 |

value_function loss 后段约为 0.01，说明 PPO 的价值学习没有明显发散；平均奖励在约 10000 iteration 后进入缓慢增长/平台阶段。需要注意，recovery_completion_rate 接近 1 并不等同于 Recovery 成功率；前者是内部流程完成统计，真正判断恢复效果应看 recovery_success_rate、躯干高度、直立度和脚部稳定性。

## 对项目整体思想的理解

项目的核心不是把三个独立任务串联成三个模型，而是训练一个能够进行状态驱动阶段切换的通用运动策略：

参考动作 Mimic -> 跌倒检测 -> Recovery -> 成功后回到 Mimic 或 Task

这种设计的优点是部署时只需要一个 Actor/ONNX policy，能在动作模仿、摔倒自恢复和速度控制之间连续切换；缺点是不同阶段的奖励和数据分布竞争同一个策略容量，训练难度明显高于单一 Mimic 或单一 locomotion 任务。当前日志体现出：站稳和直立已经学会，但精细动作跟踪以及从各种跌倒姿态稳定完成 Recovery 仍需继续优化和评估。

## 建议的评估方式

不要只用总 reward 选择 checkpoint。建议分别统计：

1. Recovery：按跌倒姿态分桶的成功率、恢复时间、最终躯干高度/直立度、脚部稳定率。
2. Mimic：anchor/body/joint position 和 orientation error，以及动作片段完成率。
3. Task：线速度和偏航速度 tracking error、上肢 guard 保持率、摔倒率。
4. 统一策略：在 Isaac Sim 和 MuJoCo 中分别测试，并确认 ONNX 的观测顺序与训练 Actor 一致。

当前 model_14999.pt 和导出的 ONNX 可以作为本轮训练候选模型，但是否最佳应由上述分任务指标和实际视频/仿真回放共同决定。
