import os
import time
from collections import deque

import torch
from rsl_rl.env import VecEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner
from rsl_rl.utils import store_code_state

from isaaclab_rl.rsl_rl import export_policy_as_onnx

import wandb
from whole_body_tracking.amp import RecoveryAmpSidecar
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx


class MyOnPolicyRunner(OnPolicyRunner):
    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        if self.logger_type in ["wandb"]:
            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            export_policy_as_onnx(self.alg.policy, normalizer=self.obs_normalizer, path=policy_path, filename=filename)
            attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))


class MotionOnPolicyRunner(OnPolicyRunner):
    def __init__(
        self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu", registry_name: str = None
    ):
        super().__init__(env, train_cfg, log_dir, device)
        self.registry_name = registry_name

    def save(self, path: str, infos=None):
        """Save the model and training information."""
        checkpoint_infos = dict(infos) if isinstance(infos, dict) else {}
        motion_command = self.env.unwrapped.command_manager.get_term("motion")
        checkpoint_infos["motion_curriculum"] = motion_command.get_curriculum_state()
        super().save(path, checkpoint_infos)
        if self.logger_type in ["wandb"]:
            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            export_motion_policy_as_onnx(
                self.env.unwrapped, self.alg.policy, normalizer=self.obs_normalizer, path=policy_path, filename=filename
            )
            attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))

            # link the artifact registry to this run
            if self.registry_name is not None:
                wandb.run.use_artifact(self.registry_name)
                self.registry_name = None

    def load(self, path: str, load_optimizer: bool = True):
        infos = super().load(path, load_optimizer=load_optimizer)
        if isinstance(infos, dict) and "motion_curriculum" in infos:
            self.env.unwrapped.command_manager.get_term("motion").load_curriculum_state(
                infos["motion_curriculum"]
            )
        return infos


class RecoveryAmpOnPolicyRunner(MotionOnPolicyRunner):
    """Standard PPO runner with an external, recovery-only AMP reward sidecar."""

    def __init__(
        self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu", registry_name: str = None
    ):
        amp_cfg = dict(train_cfg.get("recovery_amp", {}))
        if not amp_cfg.get("enabled", False):
            raise ValueError("RecoveryAmpOnPolicyRunner requires recovery_amp.enabled=True.")
        super().__init__(env, train_cfg, log_dir, device, registry_name)
        if self.training_type != "rl":
            raise ValueError("Recovery AMP reward shaping supports PPO/RL training only.")
        if self.is_distributed:
            raise NotImplementedError(
                "Recovery AMP currently supports a single training process; discriminator synchronization "
                "must be added before multi-GPU training."
            )

        _, extras = self.env.get_observations()
        amp_observation = extras["observations"].get("amp")
        if amp_observation is None:
            raise ValueError("AMP observation group is missing; use the recovery AMP environment config.")
        motion_command = self.env.unwrapped.command_manager.get_term("motion")
        anchor_body_name = motion_command.cfg.anchor_body_name
        amp_body_names = [name for name in motion_command.cfg.body_names if name != anchor_body_name]
        self.recovery_amp = RecoveryAmpSidecar(
            amp_cfg,
            amp_body_names,
            anchor_body_name,
            self.env.unwrapped.step_dt,
            amp_observation.shape[1],
            self.device,
        )
        clip_summary = ", ".join(
            f"{os.path.basename(info.path)}:{info.transitions}@stride{info.stride}"
            for info in self.recovery_amp.expert_dataset.clip_infos
        )
        print(
            f"[INFO] Recovery AMP initialized: state_dim={amp_observation.shape[1]}, "
            f"expert_transitions={len(self.recovery_amp.expert_dataset)} ({clip_summary})"
        )
        print(
            "[INFO] Recovery reward mix: "
            f"{self.recovery_amp.task_reward_lerp:.2f} * task + "
            f"{1.0 - self.recovery_amp.task_reward_lerp:.2f} * "
            f"({self.recovery_amp.amp_reward_coef:.3f} * raw AMP), "
            f"max AMP component="
            f"{(1.0 - self.recovery_amp.task_reward_lerp) * self.recovery_amp.amp_reward_coef:.6f}"
        )

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):  # noqa: C901
        # This follows RSL-RL's OnPolicyRunner loop intentionally.  The only
        # algorithmic additions are the marked AMP transition/reward/update
        # operations; Actor, storage, PPO returns, and PPO update are unchanged.
        if self.log_dir is not None and self.writer is None and not self.disable_logs:
            self.logger_type = self.cfg.get("logger", "tensorboard").lower()
            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise ValueError("Logger type not found. Please choose 'neptune', 'wandb' or 'tensorboard'.")

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs, extras = self.env.get_observations()
        privileged_obs = extras["observations"].get(self.privileged_obs_type, obs)
        amp_obs = extras["observations"]["amp"].to(self.device)
        obs, privileged_obs = obs.to(self.device), privileged_obs.to(self.device)
        self.train_mode()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        if self.alg.rnd:
            erewbuffer = deque(maxlen=100)
            irewbuffer = deque(maxlen=100)
            cur_ereward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
            cur_ireward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        motion_command = self.env.unwrapped.command_manager.get_term("motion")
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()
            amp_rollout_sums = {
                "valid_transitions": 0.0,
                "task_abs_sum": 0.0,
                "raw_reward_sum": 0.0,
                "scaled_discriminator_reward_sum": 0.0,
                "task_component_abs_sum": 0.0,
                "amp_component_sum": 0.0,
            }
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, privileged_obs)

                    # Snapshot the phase before stepping.  This excludes the
                    # tracking->fall entry transition from AMP by construction.
                    recovery_active_before_step = motion_command.recovery_active.clone()
                    obs, rewards, dones, infos = self.env.step(actions.to(self.env.device))
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    next_amp_obs = infos["observations"]["amp"].to(self.device)
                    valid_amp_transition = recovery_active_before_step.to(self.device) & (dones == 0)
                    rewards, amp_step_sums = self.recovery_amp.shape_rewards(
                        rewards, amp_obs, next_amp_obs, valid_amp_transition
                    )
                    for key, value in amp_step_sums.items():
                        amp_rollout_sums[key] += value
                    amp_obs = next_amp_obs

                    obs = self.obs_normalizer(obs)
                    if self.privileged_obs_type is not None:
                        privileged_obs = self.privileged_obs_normalizer(
                            infos["observations"][self.privileged_obs_type].to(self.device)
                        )
                    else:
                        privileged_obs = obs
                    self.alg.process_env_step(rewards, dones, infos)
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.alg.rnd else None

                    if self.log_dir is not None:
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])
                        if self.alg.rnd:
                            cur_ereward_sum += rewards
                            cur_ireward_sum += intrinsic_rewards
                            cur_reward_sum += rewards + intrinsic_rewards
                        else:
                            cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                        if self.alg.rnd:
                            erewbuffer.extend(cur_ereward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            irewbuffer.extend(cur_ireward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            cur_ereward_sum[new_ids] = 0
                            cur_ireward_sum[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                start = stop
                self.alg.compute_returns(privileged_obs)

            loss_dict = self.alg.update()
            amp_metrics = self.recovery_amp.update()
            valid_count = amp_rollout_sums["valid_transitions"]
            denominator = max(valid_count, 1.0)
            task_component_abs = amp_rollout_sums["task_component_abs_sum"]
            amp_component = amp_rollout_sums["amp_component_sum"]
            amp_metrics.update(self.recovery_amp.training_metrics())
            amp_metrics.update(
                {
                    "valid_transition_fraction": valid_count
                    / (self.num_steps_per_env * self.env.num_envs),
                    "mean_recovery_task_abs": amp_rollout_sums["task_abs_sum"] / denominator,
                    "mean_task_component_abs": task_component_abs / denominator,
                    "mean_raw_reward": amp_rollout_sums["raw_reward_sum"] / denominator,
                    "mean_scaled_discriminator_reward": amp_rollout_sums[
                        "scaled_discriminator_reward_sum"
                    ]
                    / denominator,
                    "mean_amp_component": amp_component / denominator,
                    "observed_amp_fraction": amp_component
                    / max(task_component_abs + amp_component, 1.0e-12),
                }
            )

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            if self.log_dir is not None and not self.disable_logs:
                self.log(locals())
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            ep_infos.clear()
            if it == start_iter and not self.disable_logs:
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        super().log(locs, width=width, pad=pad)
        for key, value in locs.get("amp_metrics", {}).items():
            self.writer.add_scalar(f"AMP/{key}", value, locs["it"])

    def save(self, path: str, infos=None):
        checkpoint_infos = dict(infos) if isinstance(infos, dict) else {}
        checkpoint_infos["recovery_amp"] = self.recovery_amp.state_dict()
        super().save(path, checkpoint_infos)

    def load(self, path: str, load_optimizer: bool = True):
        infos = super().load(path, load_optimizer=load_optimizer)
        if isinstance(infos, dict) and "recovery_amp" in infos:
            self.recovery_amp.load_state_dict(infos["recovery_amp"], load_optimizer=load_optimizer)
        else:
            print("[INFO] Checkpoint has no recovery AMP state; discriminator starts from scratch.")
        return infos

    def train_mode(self):
        super().train_mode()
        self.recovery_amp.train()

    def eval_mode(self):
        super().eval_mode()
        self.recovery_amp.eval()
