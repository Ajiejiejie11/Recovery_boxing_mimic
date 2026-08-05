"""Independent AMP optimizer and recovery-only reward shaping sidecar."""

from __future__ import annotations

from typing import Any

import torch

from .dataset import AmpExpertDataset
from .discriminator import AmpDiscriminator, RunningMeanStd
from .replay_buffer import AmpReplayBuffer


class RecoveryAmpSidecar:
    """Own AMP data, replay, discriminator, optimizer, and recovery reward mixing."""

    def __init__(
        self,
        cfg: dict[str, Any],
        body_names: list[str],
        anchor_body_name: str,
        step_dt: float,
        state_dim: int,
        device: str | torch.device,
    ):
        self.cfg = cfg
        self.device = torch.device(device)
        self.amp_reward_coef = float(cfg["amp_reward_coef"])
        self.task_reward_lerp = float(cfg["amp_task_reward_lerp"])
        if self.amp_reward_coef < 0.0:
            raise ValueError("amp_reward_coef must be non-negative.")
        if not 0.0 <= self.task_reward_lerp <= 1.0:
            raise ValueError("amp_task_reward_lerp must be between zero and one.")
        self.batch_size = int(cfg["batch_size"])
        self.updates_per_iteration = int(cfg["updates_per_iteration"])
        self.gradient_penalty = float(cfg["gradient_penalty"])
        self.max_grad_norm = float(cfg["max_grad_norm"])
        self.min_replay_size = max(int(cfg["min_replay_size"]), self.batch_size)

        self.expert_dataset = AmpExpertDataset(
            cfg["expert_motion_path"], body_names, anchor_body_name, step_dt, self.device
        )
        if self.expert_dataset.state_dim != state_dim:
            raise ValueError(
                f"Online AMP state has {state_dim} columns but expert state has "
                f"{self.expert_dataset.state_dim}."
            )
        self.replay = AmpReplayBuffer(int(cfg["replay_capacity"]), state_dim, self.device)
        self.normalizer = RunningMeanStd(state_dim).to(self.device)
        self.discriminator = AmpDiscriminator(
            state_dim, list(cfg["hidden_dims"]), str(cfg["activation"])
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=float(cfg["learning_rate"])
        )

    @torch.no_grad()
    def shape_rewards(
        self,
        task_rewards: torch.Tensor,
        states: torch.Tensor,
        next_states: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Add style reward only on valid recovery transitions and store them."""
        valid_mask = valid_mask.to(device=self.device, dtype=torch.bool).reshape(-1)
        if states.shape != next_states.shape or states.shape[0] != valid_mask.numel():
            raise ValueError("AMP states, next states, and valid mask have incompatible shapes.")
        if task_rewards.numel() != valid_mask.numel():
            raise ValueError("AMP valid mask must have one value per task reward.")

        valid_count = int(valid_mask.sum().item())
        if valid_count == 0:
            return task_rewards, {
                "valid_transitions": 0.0,
                "task_abs_sum": 0.0,
                "raw_reward_sum": 0.0,
                "scaled_discriminator_reward_sum": 0.0,
                "task_component_abs_sum": 0.0,
                "amp_component_sum": 0.0,
            }

        valid_states = states[valid_mask].to(self.device)
        valid_next_states = next_states[valid_mask].to(self.device)
        self.replay.insert(valid_states, valid_next_states)
        normalized_states = self.normalizer.normalize(valid_states)
        normalized_next_states = self.normalizer.normalize(valid_next_states)
        logits = self.discriminator(normalized_states, normalized_next_states)
        raw_rewards = self.discriminator.style_reward(logits)

        task_values = task_rewards.reshape(-1)[valid_mask]
        discriminator_rewards = self.amp_reward_coef * raw_rewards
        amp_lerp = 1.0 - self.task_reward_lerp
        task_component = self.task_reward_lerp * task_values
        amp_component = amp_lerp * discriminator_rewards
        mixed_rewards = task_component + amp_component
        shaped_rewards = task_rewards.clone()
        shaped_rewards.reshape(-1)[valid_mask] = mixed_rewards.to(task_rewards.dtype)
        return shaped_rewards, {
            "valid_transitions": float(valid_count),
            "task_abs_sum": float(task_values.abs().sum().item()),
            "raw_reward_sum": float(raw_rewards.sum().item()),
            "scaled_discriminator_reward_sum": float(discriminator_rewards.sum().item()),
            "task_component_abs_sum": float(task_component.abs().sum().item()),
            "amp_component_sum": float(amp_component.sum().item()),
        }

    def update(self) -> dict[str, float]:
        """Train the discriminator without touching Actor/PPO parameters."""
        metrics = {
            "discriminator_loss": 0.0,
            "expert_score": 0.0,
            "policy_score": 0.0,
            "gradient_penalty": 0.0,
            "updates": 0.0,
        }
        if len(self.replay) < self.min_replay_size or self.updates_per_iteration <= 0:
            return metrics

        self.discriminator.train()
        for _ in range(self.updates_per_iteration):
            policy_states, policy_next_states = self.replay.sample(self.batch_size)
            expert_states, expert_next_states = self.expert_dataset.sample(self.batch_size)
            with torch.no_grad():
                self.normalizer.update(
                    torch.cat(
                        (policy_states, policy_next_states, expert_states, expert_next_states), dim=0
                    )
                )
            policy_states = self.normalizer.normalize(policy_states)
            policy_next_states = self.normalizer.normalize(policy_next_states)
            expert_states = self.normalizer.normalize(expert_states)
            expert_next_states = self.normalizer.normalize(expert_next_states)

            policy_logits = self.discriminator(policy_states, policy_next_states)
            expert_logits = self.discriminator(expert_states, expert_next_states)
            least_squares_loss = 0.5 * (
                (expert_logits - 1.0).square().mean() + (policy_logits + 1.0).square().mean()
            )

            expert_pair = torch.cat((expert_states, expert_next_states), dim=-1).detach()
            expert_pair.requires_grad_(True)
            expert_gp_logits = self.discriminator.network(expert_pair).squeeze(-1)
            gradients = torch.autograd.grad(
                expert_gp_logits.sum(), expert_pair, create_graph=True, only_inputs=True
            )[0]
            gradient_penalty = 0.5 * self.gradient_penalty * gradients.square().sum(dim=-1).mean()
            loss = least_squares_loss + gradient_penalty

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.max_grad_norm)
            self.optimizer.step()

            metrics["discriminator_loss"] += float(least_squares_loss.detach().item())
            metrics["expert_score"] += float(expert_logits.detach().mean().item())
            metrics["policy_score"] += float(policy_logits.detach().mean().item())
            metrics["gradient_penalty"] += float(gradient_penalty.detach().item())
            metrics["updates"] += 1.0

        update_count = metrics["updates"]
        for key in ("discriminator_loss", "expert_score", "policy_score", "gradient_penalty"):
            metrics[key] /= update_count
        return metrics

    def training_metrics(self) -> dict[str, float]:
        return {
            "amp_reward_coef": self.amp_reward_coef,
            "task_reward_lerp": self.task_reward_lerp,
            "amp_reward_lerp": 1.0 - self.task_reward_lerp,
            "max_amp_component": (1.0 - self.task_reward_lerp) * self.amp_reward_coef,
            "replay_size": float(len(self.replay)),
            "expert_transitions": float(len(self.expert_dataset)),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "discriminator": self.discriminator.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "normalizer": self.normalizer.state_dict(),
            "reward_formulation": "amp_mjlab_lerp_v1",
            "amp_reward_coef": self.amp_reward_coef,
            "amp_task_reward_lerp": self.task_reward_lerp,
        }

    def load_state_dict(self, state: dict[str, Any], load_optimizer: bool = True) -> None:
        self.discriminator.load_state_dict(state["discriminator"])
        self.normalizer.load_state_dict(state["normalizer"])
        if load_optimizer and "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])

    def train(self) -> None:
        self.discriminator.train()

    def eval(self) -> None:
        self.discriminator.eval()
