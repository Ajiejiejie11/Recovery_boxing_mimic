"""AMP transition discriminator and its state normalizer."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class RunningMeanStd(nn.Module):
    """Numerically stable running moments for individual AMP states."""

    def __init__(self, state_dim: int, epsilon: float = 1.0e-4):
        super().__init__()
        self.register_buffer("mean", torch.zeros(state_dim))
        self.register_buffer("variance", torch.ones(state_dim))
        self.register_buffer("count", torch.tensor(float(epsilon), dtype=torch.float64))

    @torch.no_grad()
    def update(self, values: torch.Tensor) -> None:
        if values.numel() == 0:
            return
        values = values.detach().reshape(-1, self.mean.numel()).float()
        batch_mean = values.mean(dim=0)
        batch_variance = values.var(dim=0, unbiased=False)
        batch_count = values.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * (batch_count / total_count)
        old_m2 = self.variance * self.count
        batch_m2 = batch_variance * batch_count
        new_m2 = old_m2 + batch_m2 + delta.square() * self.count * batch_count / total_count
        self.mean.copy_(new_mean)
        self.variance.copy_((new_m2 / total_count).clamp_min(1.0e-6))
        self.count.copy_(total_count)

    def normalize(self, values: torch.Tensor, clip: float = 10.0) -> torch.Tensor:
        normalized = (values - self.mean) * torch.rsqrt(self.variance + 1.0e-6)
        return normalized.clamp(-clip, clip)


class AmpDiscriminator(nn.Module):
    """Least-squares discriminator over ``(state_t, state_t+1)`` pairs."""

    def __init__(self, state_dim: int, hidden_dims: list[int], activation: str = "elu"):
        super().__init__()
        activation_classes = {
            "elu": nn.ELU,
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "silu": nn.SiLU,
        }
        if activation not in activation_classes:
            raise ValueError(f"Unsupported AMP discriminator activation: {activation}")
        layers: list[nn.Module] = []
        input_dim = state_dim * 2
        for hidden_dim in hidden_dims:
            layers.extend((nn.Linear(input_dim, hidden_dim), activation_classes[activation]()))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)
        self._initialize()

    def _initialize(self) -> None:
        linear_layers = [module for module in self.modules() if isinstance(module, nn.Linear)]
        for layer in linear_layers[:-1]:
            nn.init.orthogonal_(layer.weight, gain=2.0**0.5)
            nn.init.zeros_(layer.bias)
        nn.init.uniform_(linear_layers[-1].weight, -0.1, 0.1)
        nn.init.zeros_(linear_layers[-1].bias)

    def forward(self, states: torch.Tensor, next_states: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat((states, next_states), dim=-1)).squeeze(-1)

    @staticmethod
    def style_reward(logits: torch.Tensor) -> torch.Tensor:
        """Smooth style score centered at the least-squares policy target.

        ``softplus(1) - softplus(-1) == 1`` and a policy-target score of
        ``-1`` maps to zero.  Scores below ``-1`` retain a bounded negative
        ordering instead of entering the old hard-clamped zero-reward region.
        """
        policy_target_softplus = F.softplus(logits.new_tensor(-1.0))
        return (F.softplus(logits) - policy_target_softplus).clamp_max(1.0)
