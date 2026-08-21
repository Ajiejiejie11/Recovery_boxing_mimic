from __future__ import annotations

import torch
from collections.abc import Sequence

from isaaclab.envs import ManagerBasedRLEnv


class TrackingRLEnv(ManagerBasedRLEnv):
    """Tracking environment with task-conditioned diagnostic aggregation.

    Isaac Lab's default manager logs per-environment tensors only for the IDs
    reset on a given step. That dilutes a single six-second recovery outcome by
    unrelated tracking resets. This subclass snapshots reward groups before
    their episode buffers are cleared, then reads persistent recovery/sampling
    counters after command reset. It does not alter observations, rewards,
    terminations, or reset behavior.
    """

    _TRACKING_REWARD_TERMS = (
        "motion_global_anchor_pos",
        "motion_global_anchor_ori",
        "motion_body_pos",
        "motion_body_ori",
        "motion_body_lin_vel",
        "motion_body_ang_vel",
        "undesired_contacts",
        "hand_slip",
    )
    _RECOVERY_REWARD_TERMS = (
        "recovery_upright",
        "recovery_height",
        "recovery_early_success",
        "recovery_feet_stable",
        "recovery_full_body_reference",
        "recovery_torso_reference",
    )
    _TASK_REWARD_TERMS = (
        "task_lin_vel_xy_track",
        "task_ang_vel_z_track",
        "task_upright",
        "task_upper_body_pose",
    )
    _SHARED_REWARD_TERMS = (
        "action_rate_l2",
        "joint_limit",
        "joint_vel_limit",
        "joint_torque_limit",
        "contact_forces",
        "foot_slip",
        "self_collision",
    )

    def _reward_group_episode_mean(self, env_ids: torch.Tensor, names: tuple[str, ...]) -> torch.Tensor:
        total = torch.zeros(len(env_ids), device=self.device)
        for name in names:
            episode_sum = self.reward_manager._episode_sums.get(name)
            if episode_sum is not None:
                total += episode_sum[env_ids]
        return total.mean() / self.max_episode_length_s

    def _reset_idx(self, env_ids: Sequence[int]):
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        command = self.command_manager.get_term("motion")

        reward_group_logs = {
            "Episode_Reward_Group/tracking": self._reward_group_episode_mean(
                ids, self._TRACKING_REWARD_TERMS
            ),
            "Episode_Reward_Group/recovery": self._reward_group_episode_mean(
                ids, self._RECOVERY_REWARD_TERMS
            ),
            "Episode_Reward_Group/task": self._reward_group_episode_mean(
                ids, self._TASK_REWARD_TERMS
            ),
            "Episode_Reward_Group/shared_regularization": self._reward_group_episode_mean(
                ids, self._SHARED_REWARD_TERMS
            ),
        }

        # RewardManager clears its episode accumulators during reset, so the
        # reward-group snapshot above must be taken first.  Conversely, task
        # counters and sampling-channel totals must be read *after* the command
        # reset: this includes the newly assigned tracking references and also
        # prevents the initial all-zero snapshot from diluting ratio logs for an
        # entire rollout iteration.
        super()._reset_idx(env_ids)

        completed = command.total_recovery_successes + command.total_recovery_failures
        completed_denominator = completed.clamp_min(1).float()
        entry_denominator = command.total_recovery_entries.clamp_min(1).float()
        active = command.recovery_active
        if torch.any(active):
            active_height = command.recovery_torso_height[active].mean()
            active_uprightness = command.recovery_torso_uprightness[active].mean()
            active_feet_stable = command.recovery_feet_stable[active].float().mean()
        else:
            active_height = torch.zeros((), device=self.device)
            active_uprightness = torch.zeros((), device=self.device)
            active_feet_stable = torch.zeros((), device=self.device)

        actual_channel_counts = torch.as_tensor(
            command._actual_sampling_channel_counts, dtype=torch.float, device=self.device
        )
        actual_channel_total = actual_channel_counts.sum().clamp_min(1.0)
        scheduled_channel_counts = torch.as_tensor(
            command._sampling_channel_counts, dtype=torch.float, device=self.device
        )
        scheduled_channel_total = scheduled_channel_counts.sum().clamp_min(1.0)

        recovery_logs = {
            "Metrics/motion/delay_env_ratio": command.delay_env_mask.float().mean(),
            "Metrics/motion/recovery_active_ratio": active.float().mean(),
            "Metrics/motion/task_active_ratio": command.task_active.float().mean(),
            "Metrics/motion/recovery_entry_count": command.total_recovery_entries.float(),
            "Metrics/motion/recovery_success_count": command.total_recovery_successes.float(),
            "Metrics/motion/recovery_failure_count": command.total_recovery_failures.float(),
            "Metrics/motion/recovery_completion_rate": completed.float() / entry_denominator,
            "Metrics/motion/recovery_success_rate": command.total_recovery_successes.float()
            / completed_denominator,
            "Metrics/motion/recovery_failure_rate": command.total_recovery_failures.float()
            / completed_denominator,
            "Metrics/motion/recovery_duration_s": command.total_recovery_duration_s
            / completed_denominator,
            "Metrics/motion/recovery_timeout_torso_height": command.total_recovery_timeout_height
            / command.total_recovery_failures.clamp_min(1).float(),
            "Metrics/motion/recovery_active_torso_height": active_height,
            "Metrics/motion/recovery_active_torso_uprightness": active_uprightness,
            "Metrics/motion/recovery_active_feet_stable": active_feet_stable,
            # Actual ratios include cold-start replay slots that correctly fell
            # back to coverage before real failure evidence existed.
            "Metrics/motion/coverage_sampling_ratio": actual_channel_counts[0] / actual_channel_total,
            "Metrics/motion/hard_failure_replay_ratio": actual_channel_counts[1] / actual_channel_total,
            "Metrics/motion/tracking_error_replay_ratio": actual_channel_counts[2] / actual_channel_total,
            "Metrics/motion/difficulty_replay_ratio": actual_channel_counts[1:].sum() / actual_channel_total,
            # Scheduled ratios verify the cross-reset debt allocator itself.
            "Metrics/motion/scheduled_coverage_ratio": scheduled_channel_counts[0] / scheduled_channel_total,
            "Metrics/motion/scheduled_hard_replay_ratio": scheduled_channel_counts[1] / scheduled_channel_total,
            "Metrics/motion/scheduled_soft_replay_ratio": scheduled_channel_counts[2] / scheduled_channel_total,
        }

        self.extras["log"].update(reward_group_logs)
        self.extras["log"].update(recovery_logs)
