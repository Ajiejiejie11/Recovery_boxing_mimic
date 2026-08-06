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
        self.micro_batch_size = int(cfg["micro_batch_size"])
        self.gradient_penalty = float(cfg["gradient_penalty"])
        self.max_grad_norm = float(cfg["max_grad_norm"])
        if self.batch_size <= 0 or self.updates_per_iteration <= 0:
            raise ValueError(
                "AMP batch size and updates per iteration must be resolved before construction."
            )
        if self.micro_batch_size <= 0:
            raise ValueError("AMP micro batch size must be positive.")
        self.micro_batch_size = min(self.micro_batch_size, self.batch_size)
        self.min_replay_size = max(int(cfg["min_replay_size"]), self.batch_size)

        self.expert_dataset = AmpExpertDataset(
            cfg["expert_motion_path"],
            body_names,
            anchor_body_name,
            step_dt,
            self.device,
            required_clip_name=cfg.get("required_expert_clip_name"),
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
                "style_reward_sum": 0.0,
                "style_reward_abs_sum": 0.0,
                "negative_style_transitions": 0.0,
                "scaled_discriminator_reward_sum": 0.0,
                "task_component_abs_sum": 0.0,
                "amp_component_sum": 0.0,
                "amp_component_abs_sum": 0.0,
            }

        valid_states = states[valid_mask].to(self.device)
        valid_next_states = next_states[valid_mask].to(self.device)
        self.replay.insert(valid_states, valid_next_states)
        normalized_states = self.normalizer.normalize(valid_states)
        normalized_next_states = self.normalizer.normalize(valid_next_states)
        logits = self.discriminator(normalized_states, normalized_next_states)
        style_rewards = self.discriminator.style_reward(logits)

        task_values = task_rewards.reshape(-1)[valid_mask]
        discriminator_rewards = self.amp_reward_coef * style_rewards
        amp_lerp = 1.0 - self.task_reward_lerp
        task_component = self.task_reward_lerp * task_values
        amp_component = amp_lerp * discriminator_rewards
        mixed_rewards = task_component + amp_component
        shaped_rewards = task_rewards.clone()
        shaped_rewards.reshape(-1)[valid_mask] = mixed_rewards.to(task_rewards.dtype)
        return shaped_rewards, {
            "valid_transitions": float(valid_count),
            "task_abs_sum": float(task_values.abs().sum().item()),
            "style_reward_sum": float(style_rewards.sum().item()),
            "style_reward_abs_sum": float(style_rewards.abs().sum().item()),
            "negative_style_transitions": float((style_rewards < 0.0).sum().item()),
            "scaled_discriminator_reward_sum": float(discriminator_rewards.sum().item()),
            "task_component_abs_sum": float(task_component.abs().sum().item()),
            "amp_component_sum": float(amp_component.sum().item()),
            "amp_component_abs_sum": float(amp_component.abs().sum().item()),
        }

    def update(self) -> dict[str, float]:
        """Train the discriminator without touching Actor/PPO parameters."""
        metrics = {
            "discriminator_loss": 0.0,
            "least_squares_loss": 0.0,
            "expert_score": 0.0,
            "policy_score": 0.0,
            "gradient_penalty": 0.0,
            "gradient_norm": 0.0,
            "updates": 0.0,
        }
        if len(self.replay) < self.min_replay_size or self.updates_per_iteration <= 0:
            return metrics

        self.discriminator.train()
        last_expert_scores: list[torch.Tensor] = []
        last_policy_scores: list[torch.Tensor] = []
        for _ in range(self.updates_per_iteration):
            policy_states, policy_next_states = self.replay.sample(self.batch_size)
            expert_states, expert_next_states = self.expert_dataset.sample(self.batch_size)
            with torch.no_grad():
                # Updating the four populations sequentially is equivalent to
                # concatenating them, without allocating a very large tensor.
                self.normalizer.update(policy_states)
                self.normalizer.update(policy_next_states)
                self.normalizer.update(expert_states)
                self.normalizer.update(expert_next_states)

            self.optimizer.zero_grad(set_to_none=True)
            update_least_squares = 0.0
            update_gradient_penalty = 0.0
            update_gradient_norm = 0.0
            update_expert_score = 0.0
            update_policy_score = 0.0
            current_expert_scores: list[torch.Tensor] = []
            current_policy_scores: list[torch.Tensor] = []
            for start in range(0, self.batch_size, self.micro_batch_size):
                stop = min(start + self.micro_batch_size, self.batch_size)
                chunk_size = stop - start
                chunk_weight = chunk_size / self.batch_size

                policy_state = self.normalizer.normalize(policy_states[start:stop])
                policy_next_state = self.normalizer.normalize(policy_next_states[start:stop])
                expert_state = self.normalizer.normalize(expert_states[start:stop])
                expert_next_state = self.normalizer.normalize(expert_next_states[start:stop])

                policy_logits = self.discriminator(policy_state, policy_next_state)
                expert_logits = self.discriminator(expert_state, expert_next_state)
                least_squares_loss = 0.5 * (
                    (expert_logits - 1.0).square().mean() + (policy_logits + 1.0).square().mean()
                )

                expert_pair = torch.cat((expert_state, expert_next_state), dim=-1)
                policy_pair = torch.cat((policy_state, policy_next_state), dim=-1)
                alpha = torch.rand(chunk_size, 1, device=self.device)
                interpolated_pair = (
                    alpha * expert_pair + (1.0 - alpha) * policy_pair
                ).detach()
                interpolated_pair.requires_grad_(True)
                interpolated_logits = self.discriminator.network(interpolated_pair).squeeze(-1)
                gradients = torch.autograd.grad(
                    interpolated_logits.sum(),
                    interpolated_pair,
                    create_graph=True,
                    only_inputs=True,
                )[0]
                gradient_norm = gradients.norm(2, dim=-1)
                gradient_penalty = self.gradient_penalty * (
                    gradient_norm - 1.0
                ).square().mean()
                loss = (least_squares_loss + gradient_penalty) * chunk_weight
                loss.backward()

                update_least_squares += float(least_squares_loss.detach().item()) * chunk_weight
                update_gradient_penalty += float(gradient_penalty.detach().item()) * chunk_weight
                update_gradient_norm += float(gradient_norm.detach().mean().item()) * chunk_weight
                update_expert_score += float(expert_logits.detach().mean().item()) * chunk_weight
                update_policy_score += float(policy_logits.detach().mean().item()) * chunk_weight
                current_expert_scores.append(expert_logits.detach().cpu())
                current_policy_scores.append(policy_logits.detach().cpu())

            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.max_grad_norm)
            self.optimizer.step()

            metrics["discriminator_loss"] += update_least_squares + update_gradient_penalty
            metrics["least_squares_loss"] += update_least_squares
            metrics["expert_score"] += update_expert_score
            metrics["policy_score"] += update_policy_score
            metrics["gradient_penalty"] += update_gradient_penalty
            metrics["gradient_norm"] += update_gradient_norm
            metrics["updates"] += 1.0
            last_expert_scores = current_expert_scores
            last_policy_scores = current_policy_scores

        update_count = metrics["updates"]
        for key in (
            "discriminator_loss",
            "least_squares_loss",
            "expert_score",
            "policy_score",
            "gradient_penalty",
            "gradient_norm",
        ):
            metrics[key] /= update_count
        expert_scores = torch.cat(last_expert_scores)
        policy_scores = torch.cat(last_policy_scores)
        style_rewards = self.discriminator.style_reward(policy_scores)
        quantiles = torch.tensor((0.1, 0.5, 0.9))
        for prefix, values in (
            ("expert_score", expert_scores),
            ("policy_score", policy_scores),
            ("style_reward", style_rewards),
        ):
            values_p10, values_p50, values_p90 = torch.quantile(values.float(), quantiles)
            metrics[f"{prefix}_p10"] = float(values_p10.item())
            metrics[f"{prefix}_p50"] = float(values_p50.item())
            metrics[f"{prefix}_p90"] = float(values_p90.item())
        metrics["negative_style_fraction"] = float((style_rewards < 0.0).float().mean().item())
        metrics["policy_below_minus_one_fraction"] = float(
            (policy_scores <= -1.0).float().mean().item()
        )
        metrics["required_expert_score"] = self._required_expert_score()
        return metrics

    @torch.no_grad()
    def _required_expert_score(self) -> float:
        states = self.expert_dataset.required_states
        next_states = self.expert_dataset.required_next_states
        if states is None or next_states is None:
            return 0.0
        scores: list[torch.Tensor] = []
        for start in range(0, states.shape[0], self.micro_batch_size):
            stop = min(start + self.micro_batch_size, states.shape[0])
            scores.append(
                self.discriminator(
                    self.normalizer.normalize(states[start:stop]),
                    self.normalizer.normalize(next_states[start:stop]),
                )
            )
        return float(torch.cat(scores).mean().item())

    def training_metrics(self) -> dict[str, float]:
        return {
            "amp_reward_coef": self.amp_reward_coef,
            "task_reward_lerp": self.task_reward_lerp,
            "amp_reward_lerp": 1.0 - self.task_reward_lerp,
            "max_amp_component": (1.0 - self.task_reward_lerp) * self.amp_reward_coef,
            "min_amp_component": (
                -(1.0 - self.task_reward_lerp) * self.amp_reward_coef * 0.3132616875
            ),
            "replay_size": float(len(self.replay)),
            "replay_capacity": float(self.replay.capacity),
            "expert_transitions": float(len(self.expert_dataset)),
            "required_expert_transitions_per_batch": float(
                self.expert_dataset.required_transition_count
            ),
            "random_expert_transitions": float(self.expert_dataset.random_transition_count),
            "random_expert_samples_per_batch": float(
                self.batch_size - self.expert_dataset.required_transition_count
            ),
            "required_expert_fraction_per_batch": (
                self.expert_dataset.required_transition_count / self.batch_size
            ),
            "effective_batch_size": float(self.batch_size),
            "micro_batch_size": float(self.micro_batch_size),
            "updates_per_iteration": float(self.updates_per_iteration),
            "policy_samples_per_iteration": float(
                self.batch_size * self.updates_per_iteration
            ),
            "expert_samples_per_iteration": float(
                self.batch_size * self.updates_per_iteration
            ),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "discriminator": self.discriminator.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "normalizer": self.normalizer.state_dict(),
            "reward_formulation": "centered_softplus_interpolation_gp_v2",
            "amp_reward_coef": self.amp_reward_coef,
            "amp_task_reward_lerp": self.task_reward_lerp,
        }

    def load_state_dict(self, state: dict[str, Any], load_optimizer: bool = True) -> None:
        reward_formulation = state.get("reward_formulation")
        if reward_formulation != "centered_softplus_interpolation_gp_v2":
            raise ValueError(
                "Recovery AMP checkpoint uses an incompatible reward/discriminator formulation: "
                f"{reward_formulation!r}. Start with a fresh AMP sidecar."
            )
        self.discriminator.load_state_dict(state["discriminator"])
        self.normalizer.load_state_dict(state["normalizer"])
        if load_optimizer and "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])

    def train(self) -> None:
        self.discriminator.train()

    def eval(self) -> None:
        self.discriminator.eval()
