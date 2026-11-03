"""Briques communes aux agents offline (IQL, CQL)."""
from __future__ import annotations

import torch
import torch.nn as nn


def mlp(in_dim: int, hidden: list[int], out_dim: int, dropout: float = 0.0) -> nn.Sequential:
    layers, d = [], in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class TwinQ(nn.Module):
    def __init__(self, in_dim: int, act_dim: int, hidden: list[int]):
        super().__init__()
        self.q1 = mlp(in_dim + act_dim, hidden, 1)
        self.q2 = mlp(in_dim + act_dim, hidden, 1)

    def both(self, x: torch.Tensor, a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        xa = torch.cat([x, a], dim=-1)
        return self.q1(xa).squeeze(-1), self.q2(xa).squeeze(-1)

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return torch.min(*self.both(x, a))


def critic_input(obs: torch.Tensor, timesteps: torch.Tensor | None, time_feature: bool,
                 max_timestep: int) -> torch.Tensor:
    """Entrée des critiques : s, ou [s, t/H] pour un critique à horizon fini."""
    if not time_feature:
        return obs
    if timesteps is None:
        raise ValueError("critique avec time_feature : timesteps requis")
    return torch.cat([obs, (timesteps.float() / max_timestep).unsqueeze(-1)], dim=-1)


@torch.no_grad()
def polyak_update(net: nn.Module, target: nn.Module, tau: float) -> None:
    for p, pt in zip(net.parameters(), target.parameters()):
        pt.mul_(1.0 - tau).add_(p, alpha=tau)
