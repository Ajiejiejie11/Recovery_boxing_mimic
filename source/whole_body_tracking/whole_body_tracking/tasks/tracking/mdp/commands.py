from __future__ import annotations

import hashlib
import numpy as np
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from pathlib import Path
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class MotionLoader:
    """Load one motion file or a directory of motion files as one indexed dataset."""

    _REQUIRED_KEYS = (
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
    )

    def __init__(
        self,
        motion_source: str,
        joint_names: Sequence[str],
        body_names: Sequence[str],
        device: str = "cpu",
        bin_duration_s: float = 1.0,
        min_bin_duration_s: float = 0.5,
    ):
        source = Path(motion_source).expanduser().resolve()
        if source.is_dir():
            motion_files = sorted(source.glob("*.npz"))
        elif source.is_file():
            motion_files = [source]
        else:
            raise FileNotFoundError(f"Invalid motion source: {source}")
        if not motion_files:
            raise ValueError(f"No .npz motion files found in: {source}")

        arrays: dict[str, list[np.ndarray]] = {key: [] for key in self._REQUIRED_KEYS}
        motion_lengths: list[int] = []
        motion_names: list[str] = []
        motion_hashes: list[str] = []
        fps: int | None = None
        target_joint_names = list(joint_names)
        target_body_names = list(body_names)
        for motion_file in motion_files:
            with np.load(motion_file, allow_pickle=False) as data:
                missing = [
                    key for key in ("fps", "joint_names", "body_names", *self._REQUIRED_KEYS) if key not in data
                ]
                if missing:
                    raise ValueError(
                        f"{motion_file} is missing fields: {missing}. Regenerate legacy motions with a converter "
                        "that writes joint_names/body_names; index-only loading is unsafe."
                    )
                file_fps = int(np.asarray(data["fps"]).item())
                if fps is None:
                    fps = file_fps
                elif file_fps != fps:
                    raise ValueError(f"All motions must have the same FPS; {motion_file} has {file_fps}, expected {fps}")
                frame_count = int(data["joint_pos"].shape[0])
                file_joint_names = np.asarray(data["joint_names"]).astype(str).tolist()
                file_body_names = np.asarray(data["body_names"]).astype(str).tolist()
                if len(file_joint_names) != len(set(file_joint_names)):
                    raise ValueError(f"{motion_file}: joint_names contains duplicates")
                if len(file_body_names) != len(set(file_body_names)):
                    raise ValueError(f"{motion_file}: body_names contains duplicates")
                missing_joints = [name for name in target_joint_names if name not in file_joint_names]
                missing_bodies = [name for name in target_body_names if name not in file_body_names]
                if missing_joints or missing_bodies:
                    raise ValueError(
                        f"{motion_file}: names required by the robot are missing; "
                        f"joints={missing_joints}, bodies={missing_bodies}"
                    )
                joint_indexes = [file_joint_names.index(name) for name in target_joint_names]
                body_indexes = [file_body_names.index(name) for name in target_body_names]
                if data["joint_pos"].shape != (frame_count, len(file_joint_names)) or data["joint_vel"].shape != (
                    frame_count,
                    len(file_joint_names),
                ):
                    raise ValueError(
                        f"{motion_file}: joint arrays do not match its {len(file_joint_names)} joint_names"
                    )
                file_body_count = int(data["body_pos_w"].shape[1])
                if file_body_count != len(file_body_names):
                    raise ValueError(
                        f"{motion_file}: body arrays contain {file_body_count} bodies but body_names has "
                        f"{len(file_body_names)} entries"
                    )
                expected_shapes = {
                    "body_pos_w": (frame_count, file_body_count, 3),
                    "body_quat_w": (frame_count, file_body_count, 4),
                    "body_lin_vel_w": (frame_count, file_body_count, 3),
                    "body_ang_vel_w": (frame_count, file_body_count, 3),
                }
                for key in self._REQUIRED_KEYS:
                    value = np.asarray(data[key], dtype=np.float32)
                    if key in expected_shapes and value.shape != expected_shapes[key]:
                        raise ValueError(f"{motion_file}: {key} has shape {value.shape}, expected {expected_shapes[key]}")
                    if not np.isfinite(value).all():
                        raise ValueError(f"{motion_file}: {key} contains NaN or Inf")
                    if key in ("joint_pos", "joint_vel"):
                        value = value[:, joint_indexes]
                    elif key.startswith("body_"):
                        value = value[:, body_indexes]
                    arrays[key].append(value)
            motion_names.append(motion_file.name)
            motion_lengths.append(frame_count)
            digest = hashlib.sha256()
            with motion_file.open("rb") as file:
                for chunk in iter(lambda: file.read(1024 * 1024), b""):
                    digest.update(chunk)
            motion_hashes.append(digest.hexdigest())

        assert fps is not None
        self.fps = fps
        self.motion_files = [str(path) for path in motion_files]
        self.motion_names = motion_names
        self.motion_hashes = motion_hashes
        self.motion_lengths = torch.tensor(motion_lengths, dtype=torch.long, device=device)
        starts = np.cumsum([0, *motion_lengths[:-1]], dtype=np.int64)
        self.motion_starts = torch.tensor(starts, dtype=torch.long, device=device)
        self.joint_pos = torch.tensor(np.concatenate(arrays["joint_pos"]), dtype=torch.float32, device=device)
        self.joint_vel = torch.tensor(np.concatenate(arrays["joint_vel"]), dtype=torch.float32, device=device)
        self._body_pos_w = torch.tensor(np.concatenate(arrays["body_pos_w"]), dtype=torch.float32, device=device)
        self._body_quat_w = torch.tensor(np.concatenate(arrays["body_quat_w"]), dtype=torch.float32, device=device)
        self._body_lin_vel_w = torch.tensor(
            np.concatenate(arrays["body_lin_vel_w"]), dtype=torch.float32, device=device
        )
        self._body_ang_vel_w = torch.tensor(
            np.concatenate(arrays["body_ang_vel_w"]), dtype=torch.float32, device=device
        )
        self.time_step_total = self.joint_pos.shape[0]

        bin_frames = max(1, round(self.fps * bin_duration_s))
        min_bin_frames = max(1, round(self.fps * min_bin_duration_s))
        bin_motion_ids: list[int] = []
        bin_start_frames: list[int] = []
        bin_end_frames: list[int] = []
        for motion_id, (motion_start, motion_length) in enumerate(zip(starts, motion_lengths)):
            local_starts = list(range(0, motion_length, bin_frames))
            if len(local_starts) > 1 and motion_length - local_starts[-1] < min_bin_frames:
                local_starts.pop()
            for index, local_start in enumerate(local_starts):
                local_end = local_starts[index + 1] if index + 1 < len(local_starts) else motion_length
                bin_motion_ids.append(motion_id)
                bin_start_frames.append(int(motion_start + local_start))
                bin_end_frames.append(int(motion_start + local_end))
        self.bin_motion_ids = torch.tensor(bin_motion_ids, dtype=torch.long, device=device)
        self.bin_start_frames = torch.tensor(bin_start_frames, dtype=torch.long, device=device)
        self.bin_end_frames = torch.tensor(bin_end_frames, dtype=torch.long, device=device)
        self.bin_count = len(bin_start_frames)

    @property
    def signature(self) -> tuple[tuple[str, int, str], ...]:
        return tuple(zip(self.motion_names, self.motion_lengths.cpu().tolist(), self.motion_hashes))

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w


class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )

        self.motion = MotionLoader(
            self.cfg.motion_file,
            self.robot.joint_names,
            self.cfg.body_names,
            device=self.device,
            bin_duration_s=self.cfg.bin_duration_s,
            min_bin_duration_s=self.cfg.min_bin_duration_s,
        )
        if not 0.0 <= self.cfg.recovery_fraction <= 1.0:
            raise ValueError(f"recovery_fraction must be in [0, 1], got {self.cfg.recovery_fraction}")
        if not 0.0 <= self.cfg.task_fraction <= 1.0:
            raise ValueError(f"task_fraction must be in [0, 1], got {self.cfg.task_fraction}")
        if self.cfg.recovery_fraction + self.cfg.task_fraction > 1.0:
            raise ValueError(
                f"recovery_fraction + task_fraction must not exceed 1.0, got "
                f"{self.cfg.recovery_fraction} + {self.cfg.task_fraction}."
            )
        if self.cfg.recovery_duration_s <= 0.0:
            raise ValueError(f"recovery_duration_s must be positive, got {self.cfg.recovery_duration_s}")
        recovery_count = int(round(self.num_envs * self.cfg.recovery_fraction))
        self.recovery_count = recovery_count
        # The default stand pose is shared by BOTH the recovery phase and the
        # task (walking) phase, so it is loaded as soon as either is enabled.
        needs_stand_target = recovery_count > 0 or self.cfg.task_fraction > 0.0
        if needs_stand_target:
            if self.cfg.recovery_target_file is None:
                raise ValueError("recovery_target_file is required when recovery or task is enabled.")
            self.recovery_target = MotionLoader(
                self.cfg.recovery_target_file,
                self.robot.joint_names,
                self.cfg.body_names,
                device=self.device,
            )
            if not 0 <= self.cfg.recovery_target_frame < self.recovery_target.time_step_total:
                raise ValueError(
                    f"recovery_target_frame={self.cfg.recovery_target_frame} is outside "
                    f"[0, {self.recovery_target.time_step_total})."
                )
        if recovery_count:
            if self.cfg.recovery_reset_file is None:
                raise ValueError("recovery_reset_file is required when recovery_fraction is non-zero.")
            self.recovery_reset = MotionLoader(
                self.cfg.recovery_reset_file,
                self.robot.joint_names,
                self.robot.body_names,
                device=self.device,
            )
            reset_anchor_index = self.robot.body_names.index(self.cfg.anchor_body_name)
            reset_anchor_pos = self.recovery_reset.body_pos_w[:, reset_anchor_index]
            reset_anchor_quat = self.recovery_reset.body_quat_w[:, reset_anchor_index]
            reset_ground_z = self.recovery_reset.body_pos_w[..., 2].amin(dim=1)
            reset_anchor_height = reset_anchor_pos[:, 2] - reset_ground_z
            reset_uprightness = 1.0 - 2.0 * (reset_anchor_quat[:, 1].square() + reset_anchor_quat[:, 2].square())
            fallen = (reset_anchor_height <= self.cfg.recovery_reset_max_height) & (
                reset_uprightness <= self.cfg.recovery_reset_max_uprightness
            )
            self.recovery_reset_frames = torch.where(fallen)[0]
            if len(self.recovery_reset_frames) == 0:
                raise ValueError(
                    "No fallen frames matched the recovery reset thresholds; "
                    "relax recovery_reset_max_height/recovery_reset_max_uprightness."
                )
        else:
            self.recovery_reset_frames = torch.empty(0, dtype=torch.long, device=self.device)
        sampling_fractions = (
            self.cfg.coverage_sampling_fraction,
            self.cfg.hard_failure_replay_fraction,
            self.cfg.tracking_error_replay_fraction,
        )
        if any(fraction < 0.0 for fraction in sampling_fractions) or not np.isclose(sum(sampling_fractions), 1.0):
            raise ValueError(f"Motion sampling fractions must be non-negative and sum to 1.0, got {sampling_fractions}")
        # Long-run channel allocation is tracked across asynchronous reset
        # batches.  Per-call rounding makes count=1/2 batches almost entirely
        # coverage and silently starves hard/soft replay.
        self._sampling_channel_fractions = np.asarray(sampling_fractions, dtype=np.float64)
        self._sampling_channel_counts = np.zeros(3, dtype=np.int64)
        self._actual_sampling_channel_counts = np.zeros(3, dtype=np.int64)
        self._sampling_channel_total = 0
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.active_bin_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.active_bin_end = torch.ones(self.num_envs, dtype=torch.long, device=self.device)
        self.motion_complete = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Eligibility and phase are deliberately separate.  ``delay_env_mask``
        # is a fixed allocation saying which slots may delay a physical-fall
        # termination.  ``recovery_active`` is the dynamic reward/reference gate.
        # Keeping the allocation fixed preserves the requested global 20/80 reset
        # split under asynchronous resets.
        self.delay_env_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if recovery_count:
            recovery_slots = torch.randperm(self.num_envs, device=self.device)[:recovery_count]
            self.delay_env_mask[recovery_slots] = True
        self.recovery_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.recovery_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.tracking_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # Task (velocity-command walking) phase. Mutually exclusive with both
        # tracking and recovery, exactly like recovery is with tracking.
        self.task_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.task_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.task_command = torch.zeros(self.num_envs, 3, device=self.device)
        self._task_resample_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.recovery_success_pending = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.recovery_failure = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.recovery_torso_height = torch.zeros(self.num_envs, device=self.device)
        self.recovery_torso_uprightness = torch.zeros(self.num_envs, device=self.device)
        self.recovery_feet_stable = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._recovery_termination = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_recovery_state_update_step = -1
        self._bin_needs_outcome = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._has_active_bin = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        self.bin_count = self.motion.bin_count
        # Keep falling risk separate from non-terminal tracking error so rare
        # hard failures cannot be diluted by the more common soft outcomes.
        self.hard_failure_score = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self.tracking_error_score = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self.bin_sample_count = torch.zeros(self.bin_count, dtype=torch.long, device=self.device)
        self.coverage_order = torch.randperm(self.bin_count, device=self.device)
        self.coverage_cursor = 0
        self.coverage_epoch = 0

        self._error_sample_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._anchor_pos_error_sum = torch.zeros(self.num_envs, device=self.device)
        self._anchor_rot_error_sum = torch.zeros(self.num_envs, device=self.device)
        self._body_pos_error_sum = torch.zeros(self.num_envs, device=self.device)
        self._body_rot_error_sum = torch.zeros(self.num_envs, device=self.device)
        self._episode_bin_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._episode_success_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._episode_soft_failure_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._episode_hard_failure_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._episode_recovery_entry_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._episode_recovery_success_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._episode_recovery_failure_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._episode_recovery_duration_sum = torch.zeros(self.num_envs, device=self.device)
        # Global recovery counters are logging-only and deliberately exclude
        # ordinary tracking resets from their denominators.
        self.total_recovery_entries = torch.zeros((), dtype=torch.long, device=self.device)
        self.total_recovery_successes = torch.zeros((), dtype=torch.long, device=self.device)
        self.total_recovery_failures = torch.zeros((), dtype=torch.long, device=self.device)
        self.total_recovery_duration_s = torch.zeros((), device=self.device)
        self.total_recovery_timeout_height = torch.zeros((), device=self.device)

        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["difficulty_score_mean"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["difficulty_score_max"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["difficulty_sampling_entropy"] = torch.ones(self.num_envs, device=self.device)
        self.metrics["difficulty_replay_ratio"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["hard_failure_score_mean"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["hard_failure_score_max"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["hard_failure_sampling_entropy"] = torch.ones(self.num_envs, device=self.device)
        self.metrics["tracking_error_score_mean"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["tracking_error_score_max"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["tracking_error_sampling_entropy"] = torch.ones(self.num_envs, device=self.device)
        self.metrics["coverage_sampling_ratio"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["hard_failure_replay_ratio"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["tracking_error_replay_ratio"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["delay_env_ratio"] = self.delay_env_mask.float().clone()
        self.metrics["recovery_active_ratio"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["task_active_ratio"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["recovery_success_rate"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["recovery_failure_rate"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["recovery_duration_s"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["torso_height"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["torso_uprightness"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["feet_stable"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["coverage_epoch"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["bin_success"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["bin_soft_failure"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["bin_hard_failure"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:  # TODO Consider again if this is the best observation
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        """Classify the final active bin before CommandTerm emits episode metrics and resamples."""
        if env_ids is None:
            ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        elif isinstance(env_ids, slice):
            ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)[env_ids]
        else:
            ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self._classify_reset_bins(ids)
        return super().reset(env_ids=ids)

    @property
    def tracking_env_mask(self) -> torch.Tensor:
        """Dynamic tracking (mimic) phase mask; complementary to task and recovery."""
        return ~self.recovery_active & ~self.task_active

    @property
    def task_env_mask(self) -> torch.Tensor:
        """Dynamic task (velocity-command walking) phase mask."""
        return self.task_active

    @property
    def recovery_env_mask(self) -> torch.Tensor:
        """Backward-compatible name for the dynamic recovery phase mask."""
        return self.recovery_active

    @property
    def recovery_progress(self) -> torch.Tensor:
        """Privileged normalized progress through the current recovery window."""
        max_steps = max(1, round(self.cfg.recovery_duration_s / self._env.step_dt))
        return (self.recovery_steps.float() / max_steps).clamp(0.0, 1.0)

    def _validate_task_reward_masks(self, step_index: int) -> None:
        """Periodically verify the three task-reward gates form an exact partition."""
        interval = self.cfg.task_mask_assert_interval
        if interval <= 0 or step_index % interval:
            return
        tracking_mask = self.tracking_env_mask
        task_mask = self.task_active
        recovery_mask = self.recovery_active
        any_overlap = (tracking_mask & task_mask) | (task_mask & recovery_mask) | (tracking_mask & recovery_mask)
        all_covered = tracking_mask | task_mask | recovery_mask
        if torch.any(any_overlap) or not torch.all(all_covered):
            raise RuntimeError("Tracking, task, and recovery reward masks must be mutually exclusive and exhaustive.")

    def _replace_recovery(self, reference: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Replace reference values while a stand-pose phase (recovery OR task) is open.

        Task walking and recovery share the same default stand reference, so a
        single ``recovery_active | task_active`` gate serves both.  The policy
        disambiguates the two phases from the task command (non-zero only in
        task) and projected gravity (fallen only in recovery).
        """
        if not hasattr(self, "recovery_target"):
            return reference
        result = reference.clone()
        mask = self.recovery_active | self.task_active
        if target.shape == reference.shape:
            result[mask] = target[mask]
        else:
            result[mask] = target
        return result

    @property
    def joint_pos(self) -> torch.Tensor:
        reference = self.motion.joint_pos[self.time_steps]
        if not hasattr(self, "recovery_target"):
            return reference
        target = self.recovery_target.joint_pos[self.cfg.recovery_target_frame]
        return self._replace_recovery(reference, target)

    @property
    def joint_vel(self) -> torch.Tensor:
        reference = self.motion.joint_vel[self.time_steps]
        return self._replace_recovery(reference, torch.zeros_like(reference[0]))

    @property
    def body_pos_w(self) -> torch.Tensor:
        reference = self.motion.body_pos_w[self.time_steps] + self._env.scene.env_origins[:, None, :]
        if not hasattr(self, "recovery_target"):
            return reference
        target = self.recovery_target.body_pos_w[self.cfg.recovery_target_frame].clone()
        target[:, :2] -= target[self.motion_anchor_body_index, :2]
        target = target + self._env.scene.env_origins[:, None, :]
        return self._replace_recovery(reference, target)

    @property
    def body_quat_w(self) -> torch.Tensor:
        reference = self.motion.body_quat_w[self.time_steps]
        if not hasattr(self, "recovery_target"):
            return reference
        target = self.recovery_target.body_quat_w[self.cfg.recovery_target_frame]
        return self._replace_recovery(reference, target)

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        reference = self.motion.body_lin_vel_w[self.time_steps]
        return self._replace_recovery(reference, torch.zeros_like(reference[0]))

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        reference = self.motion.body_ang_vel_w[self.time_steps]
        return self._replace_recovery(reference, torch.zeros_like(reference[0]))

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        reference = self.motion.body_pos_w[self.time_steps, self.motion_anchor_body_index] + self._env.scene.env_origins
        if not hasattr(self, "recovery_target"):
            return reference
        target_height = self.recovery_target.body_pos_w[
            self.cfg.recovery_target_frame, self.motion_anchor_body_index, 2
        ]
        target = self.robot_anchor_pos_w.clone()
        target[:, 2] = self._env.scene.env_origins[:, 2] + target_height
        return self._replace_recovery(reference, target)

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        reference = self.motion.body_quat_w[self.time_steps, self.motion_anchor_body_index]
        # Recovery should stand upright without being forced to turn toward the
        # source clip's global yaw.
        return self._replace_recovery(reference, yaw_quat(self.robot_anchor_quat_w))

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        reference = self.motion.body_lin_vel_w[self.time_steps, self.motion_anchor_body_index]
        return self._replace_recovery(reference, torch.zeros_like(reference[0]))

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        reference = self.motion.body_ang_vel_w[self.time_steps, self.motion_anchor_body_index]
        return self._replace_recovery(reference, torch.zeros_like(reference[0]))

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]

    def _update_reference_alignment(self) -> None:
        """Align the reference horizontal pose to each robot while preserving reference height."""
        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))
        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)

    def _update_metrics(self):
        self._update_reference_alignment()
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)

        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(
            dim=-1
        )

        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(
            dim=-1
        )

        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

        active = self._has_active_bin & self._bin_needs_outcome & ~self.motion_complete
        self._error_sample_count[active] += 1
        self._anchor_pos_error_sum[active] += self.metrics["error_anchor_pos"][active]
        self._anchor_rot_error_sum[active] += self.metrics["error_anchor_rot"][active]
        self._body_pos_error_sum[active] += self.metrics["error_body_pos"][active]
        self._body_rot_error_sum[active] += self.metrics["error_body_rot"][active]

        hard_probabilities = self._score_sampling_probabilities(self.hard_failure_score)
        tracking_probabilities = self._score_sampling_probabilities(self.tracking_error_score)
        combined_score = self.hard_failure_score + self.tracking_error_score
        difficulty_probabilities = self._score_sampling_probabilities(combined_score)
        self.metrics["difficulty_score_mean"][:] = combined_score.mean()
        self.metrics["difficulty_score_max"][:] = combined_score.max()
        self.metrics["hard_failure_score_mean"][:] = self.hard_failure_score.mean()
        self.metrics["hard_failure_score_max"][:] = self.hard_failure_score.max()
        self.metrics["tracking_error_score_mean"][:] = self.tracking_error_score.mean()
        self.metrics["tracking_error_score_max"][:] = self.tracking_error_score.max()
        if self.bin_count > 1:
            entropy_scale = torch.log(torch.tensor(float(self.bin_count), device=self.device))
            difficulty_entropy = -(difficulty_probabilities * (difficulty_probabilities + 1.0e-12).log()).sum()
            hard_entropy = -(hard_probabilities * (hard_probabilities + 1.0e-12).log()).sum()
            tracking_entropy = -(tracking_probabilities * (tracking_probabilities + 1.0e-12).log()).sum()
            self.metrics["difficulty_sampling_entropy"][:] = difficulty_entropy / entropy_scale
            self.metrics["hard_failure_sampling_entropy"][:] = hard_entropy / entropy_scale
            self.metrics["tracking_error_sampling_entropy"][:] = tracking_entropy / entropy_scale
        else:
            self.metrics["difficulty_sampling_entropy"][:] = 1.0
            self.metrics["hard_failure_sampling_entropy"][:] = 1.0
            self.metrics["tracking_error_sampling_entropy"][:] = 1.0
        self.metrics["coverage_epoch"][:] = float(self.coverage_epoch)

    def _update_bin_outcomes(
        self, bins: torch.Tensor, hard: torch.Tensor, soft: torch.Tensor, successful: torch.Tensor
    ) -> None:
        """Update independent hard-failure and tracking-error EMA targets per bin."""
        if bins.numel() == 0:
            return
        unique_bins, inverse = torch.unique(bins, return_inverse=True)
        outcome_count = torch.bincount(inverse, minlength=len(unique_bins)).float()
        hard_count = torch.bincount(inverse, weights=hard.float(), minlength=len(unique_bins))
        soft_count = torch.bincount(inverse, weights=soft.float(), minlength=len(unique_bins))
        denominator = outcome_count.clamp_min(1.0)
        hard_target = hard_count / denominator
        tracking_target = soft_count / denominator
        self.hard_failure_score[unique_bins] = torch.lerp(
            self.hard_failure_score[unique_bins], hard_target, self.cfg.hard_failure_ema_alpha
        ).clamp_(0.0, 1.0)
        self.tracking_error_score[unique_bins] = torch.lerp(
            self.tracking_error_score[unique_bins], tracking_target, self.cfg.tracking_error_ema_alpha
        ).clamp_(0.0, 1.0)

    def _settle_tracking_bins(self, env_ids: torch.Tensor) -> None:
        """Classify completed/timeout bins as successful or soft failures."""
        valid = self._has_active_bin[env_ids] & self._bin_needs_outcome[env_ids]
        if not torch.any(valid):
            return
        ids = env_ids[valid]
        bins = self.active_bin_ids[ids]
        counts = self._error_sample_count[ids].clamp_min(1).float()
        successful = (
            (self._anchor_pos_error_sum[ids] / counts <= self.cfg.success_anchor_pos_error)
            & (self._anchor_rot_error_sum[ids] / counts <= self.cfg.success_anchor_rot_error)
            & (self._body_pos_error_sum[ids] / counts <= self.cfg.success_body_pos_error)
            & (self._body_rot_error_sum[ids] / counts <= self.cfg.success_body_rot_error)
        )
        soft = ~successful
        hard = torch.zeros_like(successful)
        self._episode_bin_count[ids] += 1
        self._episode_success_count[ids] += successful.long()
        self._episode_soft_failure_count[ids] += soft.long()
        denominator = self._episode_bin_count[ids].float()
        self.metrics["bin_success"][ids] = self._episode_success_count[ids] / denominator
        self.metrics["bin_soft_failure"][ids] = self._episode_soft_failure_count[ids] / denominator
        self.metrics["bin_hard_failure"][ids] = self._episode_hard_failure_count[ids] / denominator
        self._update_bin_outcomes(bins, hard, soft, successful)
        self._bin_needs_outcome[ids] = False

    def _settle_tracking_bins_as_hard(self, env_ids: torch.Tensor) -> None:
        """Settle tracking bins that physically fell before entering recovery."""
        valid = self._has_active_bin[env_ids] & self._bin_needs_outcome[env_ids]
        if not torch.any(valid):
            return
        ids = env_ids[valid]
        hard = torch.ones(len(ids), dtype=torch.bool, device=self.device)
        soft = torch.zeros_like(hard)
        successful = torch.zeros_like(hard)
        self._episode_bin_count[ids] += 1
        self._episode_hard_failure_count[ids] += 1
        denominator = self._episode_bin_count[ids].float()
        self.metrics["bin_success"][ids] = self._episode_success_count[ids] / denominator
        self.metrics["bin_soft_failure"][ids] = self._episode_soft_failure_count[ids] / denominator
        self.metrics["bin_hard_failure"][ids] = self._episode_hard_failure_count[ids] / denominator
        self._update_bin_outcomes(self.active_bin_ids[ids], hard, soft, successful)
        self._bin_needs_outcome[ids] = False
        self._has_active_bin[ids] = False

    def _classify_reset_bins(self, env_ids: torch.Tensor) -> None:
        """Settle the current bin before an environment reset, prioritizing hard failure."""
        # A recovery phase has no active reference bin.  Delayed slots that have
        # already recovered and returned to tracking are classified normally.
        valid = self._has_active_bin[env_ids] & self._bin_needs_outcome[env_ids]
        if not torch.any(valid):
            return
        ids = env_ids[valid]
        hard = self._env.termination_manager.terminated[ids]
        successful = torch.zeros_like(hard)
        non_hard = ~hard
        if torch.any(non_hard):
            non_hard_ids = ids[non_hard]
            counts = self._error_sample_count[non_hard_ids].clamp_min(1).float()
            successful[non_hard] = (
                (self._anchor_pos_error_sum[non_hard_ids] / counts <= self.cfg.success_anchor_pos_error)
                & (self._anchor_rot_error_sum[non_hard_ids] / counts <= self.cfg.success_anchor_rot_error)
                & (self._body_pos_error_sum[non_hard_ids] / counts <= self.cfg.success_body_pos_error)
                & (self._body_rot_error_sum[non_hard_ids] / counts <= self.cfg.success_body_rot_error)
            )
        soft = non_hard & ~successful

        self._episode_bin_count[ids] += 1
        self._episode_success_count[ids] += successful.long()
        self._episode_soft_failure_count[ids] += soft.long()
        self._episode_hard_failure_count[ids] += hard.long()
        denominator = self._episode_bin_count[ids].float()
        self.metrics["bin_success"][ids] = self._episode_success_count[ids] / denominator
        self.metrics["bin_soft_failure"][ids] = self._episode_soft_failure_count[ids] / denominator
        self.metrics["bin_hard_failure"][ids] = self._episode_hard_failure_count[ids] / denominator
        self._update_bin_outcomes(self.active_bin_ids[ids], hard, soft, successful)
        self._bin_needs_outcome[ids] = False

    def update_recovery_state(
        self,
        fallen: torch.Tensor,
        standing: torch.Tensor,
        torso_height: torch.Tensor,
        torso_uprightness: torch.Tensor,
        feet_stable: torch.Tensor,
    ) -> torch.Tensor:
        """Advance the delayed-termination state machine once per environment step.

        This method is called by the first non-timeout termination term.  Doing
        the transition before reward evaluation guarantees that tracking and
        recovery rewards are mutually exclusive on the fall-triggering step.
        A successful recovery is committed by :meth:`_update_command` after that
        step's recovery reward has been evaluated, then a fresh boxing reference
        is visible in the next policy observation.
        """
        step_index = int(self._env.common_step_counter)
        if self._last_recovery_state_update_step == step_index:
            return self._recovery_termination
        self._last_recovery_state_update_step = step_index

        fallen = fallen.to(device=self.device, dtype=torch.bool)
        standing = standing.to(device=self.device, dtype=torch.bool)
        self.recovery_torso_height.copy_(torso_height)
        self.recovery_torso_uprightness.copy_(torso_uprightness)
        self.recovery_feet_stable.copy_(feet_stable)
        self.recovery_failure.zero_()
        termination = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Count only time spent in tracking or task. Recovery owns a separate
        # 300-step deadline and therefore cannot be pre-empted by a global
        # episode clock.
        tracking_at_start = ~self.recovery_active & ~self.task_active
        task_at_start = self.task_active & ~self.recovery_success_pending
        self.tracking_steps[tracking_at_start] += 1
        self.task_steps[task_at_start] += 1

        # Ordinary tracking slots never receive a delayed window.
        termination |= fallen & ~self.delay_env_mask

        active_at_start = self.recovery_active & ~self.recovery_success_pending
        if torch.any(active_at_start):
            self.recovery_steps[active_at_start] += 1
            successful = active_at_start & standing
            if torch.any(successful):
                self.recovery_success_pending[successful] = True
                self._episode_recovery_success_count[successful] += 1
                duration = self.recovery_steps[successful].float() * self._env.step_dt
                self._episode_recovery_duration_sum[successful] += duration
                self.total_recovery_successes += successful.sum()
                self.total_recovery_duration_s += duration.sum()

            max_steps = max(1, round(self.cfg.recovery_duration_s / self._env.step_dt))
            failed = active_at_start & ~successful & (self.recovery_steps >= max_steps)
            if torch.any(failed):
                self.recovery_failure[failed] = True
                self._episode_recovery_failure_count[failed] += 1
                duration = self.recovery_steps[failed].float() * self._env.step_dt
                self._episode_recovery_duration_sum[failed] += duration
                self.total_recovery_failures += failed.sum()
                self.total_recovery_duration_s += duration.sum()
                self.total_recovery_timeout_height += torso_height[failed].sum()
                termination |= failed

        # A delayed slot in tracking enters recovery only after a physical fall.
        # Its current reference bin is recorded as a hard failure exactly once.
        entering = self.delay_env_mask & ~self.recovery_active & fallen
        if torch.any(entering):
            entering_ids = torch.where(entering)[0]
            # Settles a tracking bin as a hard failure; a task slot has no
            # active bin, so this is a no-op for it.
            self._settle_tracking_bins_as_hard(entering_ids)
            self.recovery_active[entering_ids] = True
            self.task_active[entering_ids] = False
            self.task_command[entering_ids] = 0.0
            self._task_resample_steps[entering_ids] = 0
            self.recovery_steps[entering_ids] = 1
            self.recovery_success_pending[entering_ids] = False
            self.motion_complete[entering_ids] = False
            self.time_steps[entering_ids] = 0
            self.active_bin_end[entering_ids] = 1
            self._bin_needs_outcome[entering_ids] = False
            self._has_active_bin[entering_ids] = False
            self._episode_recovery_entry_count[entering_ids] += 1
            self.total_recovery_entries += len(entering_ids)

        completed = self._episode_recovery_success_count + self._episode_recovery_failure_count
        entries = self._episode_recovery_entry_count.clamp_min(1).float()
        self.metrics["recovery_active_ratio"][:] = self.recovery_active.float()
        self.metrics["task_active_ratio"][:] = self.task_active.float()
        self.metrics["recovery_success_rate"][:] = self._episode_recovery_success_count / entries
        self.metrics["recovery_failure_rate"][:] = self._episode_recovery_failure_count / entries
        self.metrics["recovery_duration_s"][:] = self._episode_recovery_duration_sum / completed.clamp_min(1)
        self.metrics["torso_height"][:] = self.recovery_torso_height
        self.metrics["torso_uprightness"][:] = self.recovery_torso_uprightness
        self.metrics["feet_stable"][:] = self.recovery_feet_stable.float()
        self._validate_task_reward_masks(step_index)
        self._recovery_termination = termination
        return termination

    def _reset_error_accumulators(self, env_ids: torch.Tensor) -> None:
        self._error_sample_count[env_ids] = 0
        self._anchor_pos_error_sum[env_ids] = 0.0
        self._anchor_rot_error_sum[env_ids] = 0.0
        self._body_pos_error_sum[env_ids] = 0.0
        self._body_rot_error_sum[env_ids] = 0.0

    def _sample_coverage_bins(self, count: int) -> torch.Tensor:
        if count <= 0:
            return torch.empty(0, dtype=torch.long, device=self.device)
        selected: list[torch.Tensor] = []
        remaining = count

        # Finish the currently active shuffled coverage epoch first.
        take = min(remaining, self.bin_count - self.coverage_cursor)
        selected.append(self.coverage_order[self.coverage_cursor : self.coverage_cursor + take])
        self.coverage_cursor += take
        remaining -= take
        if self.coverage_cursor == self.bin_count:
            self.coverage_epoch += 1
            self.coverage_cursor = 0

        # Generate all additional shuffled epochs in one GPU operation. This is
        # important for a short single motion (few bins) with 8192 environments.
        if remaining:
            full_epochs, tail = divmod(remaining, self.bin_count)
            order_count = full_epochs + int(tail > 0)
            orders = torch.rand(order_count, self.bin_count, device=self.device).argsort(dim=1)
            if full_epochs:
                selected.append(orders[:full_epochs].reshape(-1))
                self.coverage_epoch += full_epochs
            if tail:
                self.coverage_order = orders[-1]
                selected.append(self.coverage_order[:tail])
                self.coverage_cursor = tail
            else:
                self.coverage_order = torch.randperm(self.bin_count, device=self.device)
        elif self.coverage_cursor == 0:
            self.coverage_order = torch.randperm(self.bin_count, device=self.device)
        return torch.cat(selected)

    def _score_sampling_probabilities(self, score: torch.Tensor) -> torch.Tensor:
        """Return normalized EMA score weights, uniform before evidence exists."""
        weights = score.clamp_min(0.0) + self.cfg.difficulty_score_floor
        return weights / weights.sum()

    def _allocate_sampling_channels(self, count: int) -> tuple[int, int, int]:
        """Allocate coverage/hard/soft channels with bounded long-run ratio error.

        The running deficit is global to the command term, so even a sequence
        of single-environment asynchronous resets converges to the configured
        fractions. This scheduler never labels an outcome: hard and soft bin
        scores are still learned exclusively from completed tracking rollouts.
        """
        if count < 0:
            raise ValueError(f"Sampling count must be non-negative, got {count}")
        before = self._sampling_channel_counts.copy()
        for _ in range(count):
            next_total = self._sampling_channel_total + 1
            target = self._sampling_channel_fractions * next_total
            deficit = target - self._sampling_channel_counts
            channel = int(np.argmax(deficit))
            self._sampling_channel_counts[channel] += 1
            self._sampling_channel_total = next_total
        allocated = self._sampling_channel_counts - before
        return int(allocated[0]), int(allocated[1]), int(allocated[2])

    def _sample_bins(self, count: int) -> tuple[torch.Tensor, int, int, int]:
        coverage_requested, hard_requested, tracking_requested = self._allocate_sampling_channels(count)

        # Before a replay channel has observed a real failure, route its slots
        # through coverage. Do not manufacture hard/soft labels from a uniform
        # distribution merely to satisfy a quota.
        has_hard_evidence = bool(torch.any(self.hard_failure_score > 0.0).item())
        has_tracking_evidence = bool(torch.any(self.tracking_error_score > 0.0).item())
        hard_count = hard_requested if has_hard_evidence else 0
        tracking_count = tracking_requested if has_tracking_evidence else 0
        coverage_count = coverage_requested + (hard_requested - hard_count) + (tracking_requested - tracking_count)
        self._actual_sampling_channel_counts += np.asarray(
            (coverage_count, hard_count, tracking_count), dtype=np.int64
        )
        if hard_count:
            hard_bins = torch.multinomial(
                self._score_sampling_probabilities(self.hard_failure_score), hard_count, replacement=True
            )
        else:
            hard_bins = torch.empty(0, dtype=torch.long, device=self.device)
        if tracking_count:
            tracking_bins = torch.multinomial(
                self._score_sampling_probabilities(self.tracking_error_score), tracking_count, replacement=True
            )
        else:
            tracking_bins = torch.empty(0, dtype=torch.long, device=self.device)
        coverage_bins = self._sample_coverage_bins(coverage_count)
        bins = torch.cat((coverage_bins, hard_bins, tracking_bins))
        bins = bins[torch.randperm(len(bins), device=self.device)]
        return bins, coverage_count, hard_count, tracking_count

    def _assign_tracking_reference(self, env_ids: torch.Tensor) -> None:
        """Assign fresh boxing bins without changing the robot's physical state."""
        if len(env_ids) == 0:
            return
        sampled_bins, coverage_count, hard_count, tracking_count = self._sample_bins(len(env_ids))
        self.active_bin_ids[env_ids] = sampled_bins
        self.time_steps[env_ids] = self.motion.bin_start_frames[sampled_bins]
        self.active_bin_end[env_ids] = self.motion.bin_end_frames[sampled_bins]
        self.motion_complete[env_ids] = False
        self._bin_needs_outcome[env_ids] = True
        self._has_active_bin[env_ids] = True
        self.bin_sample_count += torch.bincount(sampled_bins, minlength=self.bin_count)
        denominator = len(env_ids)
        self.metrics["coverage_sampling_ratio"][env_ids] = coverage_count / denominator
        self.metrics["hard_failure_replay_ratio"][env_ids] = hard_count / denominator
        self.metrics["tracking_error_replay_ratio"][env_ids] = tracking_count / denominator
        self.metrics["difficulty_replay_ratio"][env_ids] = (hard_count + tracking_count) / denominator
        self._reset_error_accumulators(env_ids)

    def _sample_task_command(self, env_ids: torch.Tensor) -> None:
        """Sample a fresh velocity command and schedule its next resampling."""
        if len(env_ids) == 0:
            return
        xy_range = self.cfg.task_lin_vel_xy_range
        yaw_range = self.cfg.task_ang_vel_yaw_range
        self.task_command[env_ids, 0] = sample_uniform(xy_range[0], xy_range[1], (len(env_ids),), self.device)
        self.task_command[env_ids, 1] = sample_uniform(xy_range[0], xy_range[1], (len(env_ids),), self.device)
        self.task_command[env_ids, 2] = sample_uniform(yaw_range[0], yaw_range[1], (len(env_ids),), self.device)
        resample_range = self.cfg.task_resampling_time_range
        seconds = sample_uniform(resample_range[0], resample_range[1], (len(env_ids),), self.device)
        self._task_resample_steps[env_ids] = (seconds / self._env.step_dt).round().long().clamp_min(1)

    def _assign_task_command(self, env_ids: torch.Tensor) -> None:
        """Enter the task phase with a fresh velocity command and no reference bin."""
        if len(env_ids) == 0:
            return
        self.task_active[env_ids] = True
        self.task_steps[env_ids] = 0
        self.time_steps[env_ids] = 0
        self.motion_complete[env_ids] = False
        self._bin_needs_outcome[env_ids] = False
        self._has_active_bin[env_ids] = False
        self._sample_task_command(env_ids)
        self.metrics["task_active_ratio"][env_ids] = 1.0
        self._reset_error_accumulators(env_ids)

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        # Every delayed slot resets into a fallen recovery pose.  It can later
        # leave this phase without resetting and receive a fresh boxing reference.
        # Non-delayed slots split into velocity-command walking (task) and
        # reference tracking (mimic) by an independent Bernoulli draw, which
        # still converges to the configured ratio under asynchronous resets.
        recovery_ids = env_ids[self.delay_env_mask[env_ids]]
        non_delayed_ids = env_ids[~self.delay_env_mask[env_ids]]
        is_task = torch.rand(len(non_delayed_ids), device=self.device) < self.cfg.task_fraction
        task_ids = non_delayed_ids[is_task]
        mimic_ids = non_delayed_ids[~is_task]
        self.recovery_active[env_ids] = False
        self.recovery_active[recovery_ids] = True
        self.task_active[env_ids] = False
        self.task_command[env_ids] = 0.0
        self._task_resample_steps[env_ids] = 0
        self.recovery_steps[env_ids] = 0
        self.tracking_steps[env_ids] = 0
        self.task_steps[env_ids] = 0
        self.recovery_success_pending[env_ids] = False
        self.recovery_failure[env_ids] = False
        self.recovery_feet_stable[env_ids] = False
        self._recovery_termination[env_ids] = False

        self._assign_tracking_reference(mimic_ids)
        self._assign_task_command(task_ids)
        self.time_steps[recovery_ids] = 0
        self.active_bin_end[recovery_ids] = 1
        self.motion_complete[env_ids] = False
        self._bin_needs_outcome[recovery_ids] = False
        self._has_active_bin[recovery_ids] = False
        stand_ids = torch.cat((recovery_ids, task_ids))
        self.metrics["coverage_sampling_ratio"][stand_ids] = 0.0
        self.metrics["hard_failure_replay_ratio"][stand_ids] = 0.0
        self.metrics["tracking_error_replay_ratio"][stand_ids] = 0.0
        self.metrics["difficulty_replay_ratio"][stand_ids] = 0.0
        self.metrics["recovery_active_ratio"][env_ids] = self.recovery_active[env_ids].float()
        self.metrics["task_active_ratio"][env_ids] = self.task_active[env_ids].float()

        self._reset_error_accumulators(env_ids)
        self._episode_bin_count[env_ids] = 0
        self._episode_success_count[env_ids] = 0
        self._episode_soft_failure_count[env_ids] = 0
        self._episode_hard_failure_count[env_ids] = 0
        self._episode_recovery_entry_count[env_ids] = 0
        self._episode_recovery_entry_count[recovery_ids] = 1
        self.total_recovery_entries += len(recovery_ids)
        self._episode_recovery_success_count[env_ids] = 0
        self._episode_recovery_failure_count[env_ids] = 0
        self._episode_recovery_duration_sum[env_ids] = 0.0

        # Both mimic and task slots reset into an upright, roughly standing pose
        # (the properties above already resolve to the stand pose inside task).
        standing_ids = torch.cat((mimic_ids, task_ids))
        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(standing_ids), 6), device=self.device)
        root_pos[standing_ids] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori[standing_ids] = quat_mul(orientations_delta, root_ori[standing_ids])
        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(standing_ids), 6), device=self.device)
        root_lin_vel[standing_ids] += rand_samples[:, :3]
        root_ang_vel[standing_ids] += rand_samples[:, 3:]

        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()

        joint_pos[standing_ids] += sample_uniform(
            *self.cfg.joint_position_range, (len(standing_ids), joint_pos.shape[1]), self.device
        )

        if len(recovery_ids):
            # Sample real fallen poses from the converted fall/get-up capture.
            reset_indexes = torch.randint(
                len(self.recovery_reset_frames), (len(recovery_ids),), device=self.device
            )
            reset_frames = self.recovery_reset_frames[reset_indexes]
            joint_pos[recovery_ids] = self.recovery_reset.joint_pos[reset_frames]
            joint_vel[recovery_ids] = 0.0
            reset_root_pos = self.recovery_reset.body_pos_w[reset_frames, 0].clone()
            reset_ground_z = self.recovery_reset.body_pos_w[reset_frames, :, 2].amin(dim=1)
            root_pos[recovery_ids, :2] = self._env.scene.env_origins[recovery_ids, :2]
            root_pos[recovery_ids, 2] = (
                self._env.scene.env_origins[recovery_ids, 2]
                + reset_root_pos[:, 2]
                - reset_ground_z
                + self.cfg.recovery_reset_ground_clearance
            )
            root_ori[recovery_ids] = self.recovery_reset.body_quat_w[reset_frames, 0]
            root_lin_vel[recovery_ids] = 0.0
            root_ang_vel[recovery_ids] = 0.0

        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos[env_ids] = torch.clip(joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1])

        self.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1),
            env_ids=env_ids,
        )

    def _update_command(self):
        # The termination manager marks success before rewards are evaluated.
        # Commit it here, after that reward step, so the next observation exposes
        # a fresh boxing reference and tracking rewards resume on the next step.
        recovered_ids = torch.where(self.recovery_success_pending)[0]
        just_recovered = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if len(recovered_ids):
            self.recovery_active[recovered_ids] = False
            self.recovery_steps[recovered_ids] = 0
            self.tracking_steps[recovered_ids] = 0
            self.task_steps[recovered_ids] = 0
            self.recovery_success_pending[recovered_ids] = False
            self.recovery_failure[recovered_ids] = False
            # After a successful get-up, randomly continue with either walking
            # (task) or reference tracking (mimic).
            to_task = torch.rand(len(recovered_ids), device=self.device) < self.cfg.task_after_recovery_prob
            task_ids = recovered_ids[to_task]
            mimic_ids = recovered_ids[~to_task]
            self.task_active[mimic_ids] = False
            self.task_command[mimic_ids] = 0.0
            self._task_resample_steps[mimic_ids] = 0
            self._assign_tracking_reference(mimic_ids)
            self._assign_task_command(task_ids)
            just_recovered[recovered_ids] = True
            self.metrics["recovery_active_ratio"][recovered_ids] = 0.0
            self.metrics["task_active_ratio"][recovered_ids] = self.task_active[recovered_ids].float()

        # Resample the velocity command for ongoing task slots whose timer elapsed.
        ongoing_task_ids = torch.where(
            self.task_active & ~self.recovery_active & ~just_recovered
        )[0]
        if len(ongoing_task_ids):
            self._task_resample_steps[ongoing_task_ids] -= 1
            expired_ids = ongoing_task_ids[self._task_resample_steps[ongoing_task_ids] <= 0]
            if len(expired_ids):
                self._sample_task_command(expired_ids)

        active_ids = torch.where(
            self._has_active_bin & ~self.motion_complete & ~self.recovery_active & ~just_recovered
        )[0]
        if len(active_ids):
            next_frames = self.time_steps[active_ids] + 1
            crossed = next_frames >= self.active_bin_end[active_ids]
            continuing_ids = active_ids[~crossed]
            self.time_steps[continuing_ids] += 1

            crossed_ids = active_ids[crossed]
            if len(crossed_ids):
                self._settle_tracking_bins(crossed_ids)
                current_bins = self.active_bin_ids[crossed_ids]
                candidate_bins = torch.clamp(current_bins + 1, max=self.bin_count - 1)
                has_next_bin = (current_bins + 1 < self.bin_count) & (
                    self.motion.bin_motion_ids[candidate_bins] == self.motion.bin_motion_ids[current_bins]
                )

                next_ids = crossed_ids[has_next_bin]
                next_bins = candidate_bins[has_next_bin]
                if len(next_ids):
                    self.active_bin_ids[next_ids] = next_bins
                    self.time_steps[next_ids] = self.motion.bin_start_frames[next_bins]
                    self.active_bin_end[next_ids] = self.motion.bin_end_frames[next_bins]
                    self._bin_needs_outcome[next_ids] = True
                    self._reset_error_accumulators(next_ids)

                finished_ids = crossed_ids[~has_next_bin]
                if len(finished_ids):
                    self.motion_complete[finished_ids] = True

        self._update_reference_alignment()

    def get_curriculum_state(self) -> dict:
        """Return CPU curriculum state suitable for inclusion in a model checkpoint."""
        return {
            "curriculum_version": 3,
            "motion_signature": self.motion.signature,
            "hard_failure_score": self.hard_failure_score.cpu(),
            "tracking_error_score": self.tracking_error_score.cpu(),
            "bin_sample_count": self.bin_sample_count.cpu(),
            "coverage_order": self.coverage_order.cpu(),
            "coverage_cursor": self.coverage_cursor,
            "coverage_epoch": self.coverage_epoch,
            "sampling_channel_counts": torch.as_tensor(self._sampling_channel_counts.copy()),
            "actual_sampling_channel_counts": torch.as_tensor(self._actual_sampling_channel_counts.copy()),
            "sampling_channel_total": self._sampling_channel_total,
        }

    def load_curriculum_state(self, state: dict) -> bool:
        """Restore curriculum state when the motion file list and lengths still match."""
        if state.get("motion_signature") != self.motion.signature:
            print("[WARN] Motion dataset changed; starting with fresh difficulty scores and coverage order.")
            return False
        common_tensor_names = ("bin_sample_count", "coverage_order")
        if any(name not in state or state[name].numel() != self.bin_count for name in common_tensor_names):
            print("[WARN] Incompatible motion curriculum checkpoint; starting with a fresh EMA curriculum.")
            return False
        for name in common_tensor_names:
            setattr(self, name, state[name].to(self.device))
        new_score_names = ("hard_failure_score", "tracking_error_score")
        if all(name in state and state[name].numel() == self.bin_count for name in new_score_names):
            for name in new_score_names:
                setattr(self, name, state[name].to(self.device))
        elif "bin_difficulty" in state and state["bin_difficulty"].numel() == self.bin_count:
            # Legacy single-channel scores mixed hard failures with 0.4-weighted
            # soft failures. Preserve that useful ordering as the initial
            # tracking-error curriculum; hard-failure evidence is relearned.
            self.hard_failure_score.zero_()
            self.tracking_error_score.copy_(state["bin_difficulty"].to(self.device).clamp(0.0, 1.0))
            print("[INFO] Migrated legacy single-channel EMA into tracking_error_score; hard score starts fresh.")
        else:
            print("[WARN] Incompatible motion curriculum scores; starting with fresh dual-channel EMA scores.")
            self.hard_failure_score.zero_()
            self.tracking_error_score.zero_()
        self.coverage_cursor = int(state.get("coverage_cursor", 0))
        self.coverage_epoch = int(state.get("coverage_epoch", 0))
        channel_counts = state.get("sampling_channel_counts")
        if channel_counts is not None and channel_counts.numel() == 3:
            self._sampling_channel_counts = channel_counts.cpu().numpy().astype(np.int64, copy=True)
            self._sampling_channel_total = int(state.get("sampling_channel_total", self._sampling_channel_counts.sum()))
        actual_channel_counts = state.get("actual_sampling_channel_counts")
        if actual_channel_counts is not None and actual_channel_counts.numel() == 3:
            self._actual_sampling_channel_counts = actual_channel_counts.cpu().numpy().astype(np.int64, copy=True)
        return True

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/current/anchor")
                )
                self.goal_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/anchor")
                )

                self.current_body_visualizers = []
                self.goal_body_visualizers = []
                for name in self.cfg.body_names:
                    self.current_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/current/" + name)
                        )
                    )
                    self.goal_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/" + name)
                        )
                    )

            self.current_anchor_visualizer.set_visibility(True)
            self.goal_anchor_visualizer.set_visibility(True)
            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].set_visibility(True)
                self.goal_body_visualizers[i].set_visibility(True)

        else:
            if hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer.set_visibility(False)
                self.goal_anchor_visualizer.set_visibility(False)
                for i in range(len(self.cfg.body_names)):
                    self.current_body_visualizers[i].set_visibility(False)
                    self.goal_body_visualizers[i].set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        self.current_anchor_visualizer.visualize(self.robot_anchor_pos_w, self.robot_anchor_quat_w)
        self.goal_anchor_visualizer.visualize(self.anchor_pos_w, self.anchor_quat_w)

        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i])
            self.goal_body_visualizers[i].visualize(self.body_pos_relative_w[:, i], self.body_quat_relative_w[:, i])


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    class_type: type = MotionCommand

    asset_name: str = MISSING

    motion_file: str = MISSING
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    joint_position_range: tuple[float, float] = (-0.52, 0.52)

    bin_duration_s: float = 1.0
    min_bin_duration_s: float = 0.5
    coverage_sampling_fraction: float = 0.70
    hard_failure_replay_fraction: float = 0.10
    tracking_error_replay_fraction: float = 0.20
    # This fraction is split off before reference-bin sampling.  The selected
    # slots have delayed-fall eligibility and reset from fallen poses; eligibility
    # remains fixed while their active phase changes dynamically.
    recovery_fraction: float = 0.0
    recovery_target_file: str | None = None
    recovery_target_frame: int = 0
    recovery_reset_file: str | None = None
    recovery_reset_max_height: float = 0.45
    recovery_reset_max_uprightness: float = 0.65
    recovery_reset_ground_clearance: float = 0.02
    recovery_duration_s: float = 6.0
    # Task (velocity-command walking) phase. ``task_fraction`` is the share of
    # NON-delayed slots that reset into task; the remaining non-delayed slots
    # reset into reference tracking. The velocity command is sampled in the
    # robot root body frame as [vx, vy, yaw-rate].
    task_fraction: float = 0.0
    task_after_recovery_prob: float = 0.5
    task_lin_vel_xy_range: tuple[float, float] = (-0.5, 0.5)
    task_ang_vel_yaw_range: tuple[float, float] = (-0.5, 0.5)
    task_resampling_time_range: tuple[float, float] = (2.0, 6.0)
    # Set to zero to disable. The check runs sparsely to avoid synchronizing the
    # GPU on every control step during large-scale training.
    task_mask_assert_interval: int = 100
    hard_failure_ema_alpha: float = 0.02
    tracking_error_ema_alpha: float = 0.02
    difficulty_score_floor: float = 1.0e-3

    success_anchor_pos_error: float = 0.20
    success_anchor_rot_error: float = 0.60
    success_body_pos_error: float = 0.18
    success_body_rot_error: float = 0.60

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
