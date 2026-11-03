"""Implicit Q-Learning (Kostrikov, Nair, Levine, ICLR 2022).

  L_V(ψ) = E[(s,a)~D] L2^τ( min_i Q̄_i(s,a) - V_ψ(s) ),   L2^τ(u) = |τ - 1(u<0)| u²
  L_Q(θ) = E[(s,a,r,s')~D] Σ_i ( r + γ (1-d) V_ψ(s') - Q_θi(s,a) )²
  L_π(φ) = -E[(s,a)~D] exp(β (Q̄(s,a) - V_ψ(s))) log π_φ(a|s)        (AWR)

Seules les actions du dataset sont évaluées par Q : pas d'action hors distribution.
Option `time_feature` : Q et V reçoivent t / H en plus de l'état (critique à horizon fini).
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from src.models.common import TwinQ, critic_input, mlp, polyak_update


class ValueNet(nn.Module):
    def __init__(self, in_dim: int, hidden: list[int]):
        super().__init__()
        self.v = mlp(in_dim, hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.v(x).squeeze(-1)


class GaussianPolicy(nn.Module):
    """π(a|s) = N(tanh(μ(s)), σ), σ indépendant de l'état (implémentation officielle Gym-MuJoCo)."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: list[int], dropout: float,
                 log_std_min: float, log_std_max: float):
        super().__init__()
        self.net = mlp(obs_dim, hidden, act_dim, dropout)
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        self.log_std_min, self.log_std_max = log_std_min, log_std_max

    def dist(self, obs: torch.Tensor) -> Normal:
        mean = torch.tanh(self.net(obs))
        std = self.log_std.clamp(self.log_std_min, self.log_std_max).exp()
        return Normal(mean, std)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        d = self.dist(obs)
        return (d.mean if deterministic else d.sample()).clamp(-1.0, 1.0)


def expectile_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    w = torch.where(u < 0, 1.0 - tau, tau)
    return (w * u.pow(2)).mean()


class IQL(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, cfg_model, cfg_train, max_timestep: int = 1000):
        super().__init__()
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.time_feature = bool(cfg_model.time_feature)
        self.max_timestep = max_timestep
        hidden = list(cfg_model.hidden_dims)
        crit_in = obs_dim + int(self.time_feature)

        self.qf = TwinQ(crit_in, act_dim, hidden)
        self.q_target = copy.deepcopy(self.qf).requires_grad_(False)
        self.vf = ValueNet(crit_in, hidden)
        self.policy = GaussianPolicy(obs_dim, act_dim, hidden, cfg_model.policy_dropout,
                                     cfg_model.log_std_min, cfg_model.log_std_max)

        t = cfg_train
        self.discount, self.tau, self.expectile = t.discount, t.tau, t.expectile
        self.beta, self.exp_adv_max = t.beta, t.exp_adv_max
        self.train_policy = bool(t.train_policy)
        self.q_opt = torch.optim.Adam(self.qf.parameters(), lr=t.critic_lr)
        self.v_opt = torch.optim.Adam(self.vf.parameters(), lr=t.critic_lr)
        self.pi_opt = torch.optim.Adam(self.policy.parameters(), lr=t.policy_lr)
        self.pi_sched = (torch.optim.lr_scheduler.CosineAnnealingLR(self.pi_opt, t.max_steps)
                         if t.policy_cosine_schedule else None)

    # -- entrées du critique ------------------------------------------------------------
    def critic_input(self, obs: torch.Tensor, timesteps: torch.Tensor | None) -> torch.Tensor:
        return critic_input(obs, timesteps, self.time_feature, self.max_timestep)

    @torch.no_grad()
    def q_value(self, obs, act, timesteps=None) -> torch.Tensor:
        return self.q_target(self.critic_input(obs, timesteps), act)

    @torch.no_grad()
    def value(self, obs, timesteps=None) -> torch.Tensor:
        return self.vf(self.critic_input(obs, timesteps))

    # -- une itération ------------------------------------------------------------------
    def update(self, b: dict[str, torch.Tensor]) -> dict[str, float]:
        t = b.get("timesteps")
        x = self.critic_input(b["observations"], t)
        x_next = self.critic_input(b["next_observations"], None if t is None else t + 1)

        # 1) V : régression expectile vers min_i Q̄_i(s, a)
        with torch.no_grad():
            target_q = self.q_target(x, b["actions"])
        v = self.vf(x)
        adv = target_q - v
        v_loss = expectile_loss(adv, self.expectile)
        self.v_opt.zero_grad(set_to_none=True)
        v_loss.backward()
        self.v_opt.step()

        # 2) Q : TD vers r + γ (1 - d) V(s')
        with torch.no_grad():
            td_target = b["rewards"] + self.discount * (1.0 - b["dones"]) * self.vf(x_next)
        q1, q2 = self.qf.both(x, b["actions"])
        q_loss = 0.5 * (F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target))
        self.q_opt.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_opt.step()

        polyak_update(self.qf, self.q_target, self.tau)

        metrics = {"v_loss": v_loss.item(), "q_loss": q_loss.item(), "v_mean": v.mean().item(),
                   "q_mean": q1.mean().item(), "adv_mean": adv.mean().item()}

        # 3) Politique : régression pondérée par l'avantage (AWR)
        if self.train_policy:
            exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=self.exp_adv_max)
            log_prob = self.policy.dist(b["observations"]).log_prob(b["actions"]).sum(-1)
            pi_loss = -(exp_adv * log_prob).mean()
            self.pi_opt.zero_grad(set_to_none=True)
            pi_loss.backward()
            self.pi_opt.step()
            if self.pi_sched:
                self.pi_sched.step()
            metrics["pi_loss"] = pi_loss.item()
        return metrics
