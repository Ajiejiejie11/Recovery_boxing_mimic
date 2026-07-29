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
        sampling_fractions = (
            self.cfg.coverage_sampling_fraction,
            self.cfg.hard_failure_replay_fraction,
            self.cfg.tracking_error_replay_fraction,
        )
        if any(fraction < 0.0 for fraction in sampling_fractions) or not np.isclose(sum(sampling_fractions), 1.0):
            raise ValueError(f"Motion sampling fractions must be non-negative and sum to 1.0, got {sampling_fractions}")
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.active_bin_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.active_bin_end = torch.ones(self.num_envs, dtype=torch.long, device=self.device)
        self.motion_complete = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
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
    def joint_pos(self) -> torch.Tensor:
        return self.motion.joint_pos[self.time_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion.joint_vel[self.time_steps]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps, self.motion_anchor_body_index] + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps, self.motion_anchor_body_index]

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

    def _classify_reset_bins(self, env_ids: torch.Tensor) -> None:
        """Settle the current bin before an environment reset, prioritizing hard failure."""
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

    def _sample_bins(self, count: int) -> tuple[torch.Tensor, int, int, int]:
        hard_count = int(round(count * self.cfg.hard_failure_replay_fraction))
        tracking_count = int(round(count * self.cfg.tracking_error_replay_fraction))
        coverage_count = count - hard_count - tracking_count
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

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        sampled_bins, coverage_count, hard_count, tracking_count = self._sample_bins(len(env_ids))
        self.active_bin_ids[env_ids] = sampled_bins
        self.time_steps[env_ids] = self.motion.bin_start_frames[sampled_bins]
        self.active_bin_end[env_ids] = self.motion.bin_end_frames[sampled_bins]
        self.motion_complete[env_ids] = False
        self._bin_needs_outcome[env_ids] = True
        self._has_active_bin[env_ids] = True
        self.bin_sample_count += torch.bincount(sampled_bins, minlength=self.bin_count)
        denominator = max(len(env_ids), 1)
        self.metrics["coverage_sampling_ratio"][env_ids] = coverage_count / denominator
        self.metrics["hard_failure_replay_ratio"][env_ids] = hard_count / denominator
        self.metrics["tracking_error_replay_ratio"][env_ids] = tracking_count / denominator
        self.metrics["difficulty_replay_ratio"][env_ids] = (hard_count + tracking_count) / denominator

        self._reset_error_accumulators(env_ids)
        self._episode_bin_count[env_ids] = 0
        self._episode_success_count[env_ids] = 0
        self._episode_soft_failure_count[env_ids] = 0
        self._episode_hard_failure_count[env_ids] = 0

        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos[env_ids] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel[env_ids] += rand_samples[:, :3]
        root_ang_vel[env_ids] += rand_samples[:, 3:]

        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()

        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
        )
        self.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1),
            env_ids=env_ids,
        )

    def _update_command(self):
        active_ids = torch.where(self._has_active_bin & ~self.motion_complete)[0]
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
            "curriculum_version": 2,
            "motion_signature": self.motion.signature,
            "hard_failure_score": self.hard_failure_score.cpu(),
            "tracking_error_score": self.tracking_error_score.cpu(),
            "bin_sample_count": self.bin_sample_count.cpu(),
            "coverage_order": self.coverage_order.cpu(),
            "coverage_cursor": self.coverage_cursor,
            "coverage_epoch": self.coverage_epoch,
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
