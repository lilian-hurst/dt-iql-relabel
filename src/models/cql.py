"""Conservative Q-Learning, CQL(H) (Kumar et al., NeurIPS 2020) — critique de la baseline QDT.

QDT (Yamagata et al., ICML 2023) relabellise les RTG avec la valeur d'un critique CQL ; on
l'implémente pour comparer, à règle de relabellisation égale, un critique Q-learning
conservateur (CQL) et un critique in-sample (IQL).

  L_Q = Σ_i  ½ E[(Q_i(s,a) - y)²]  +  α_cql · E_s[ logsumexp_a Q_i(s,a) - E_{a~D} Q_i(s,a) ]
  y   = r + γ (1-d) min_j Q̄_j(s', a'),  a' ~ π(·|s')
  L_π = E_s[ α log π(a|s) - min_j Q_j(s,a) ],  a ~ π(·|s)  (SAC, entropie auto-réglée)

Le logsumexp est estimé par échantillonnage préférentiel (actions uniformes, π(·|s), π(·|s')),
comme dans l'implémentation officielle. V(s) = E_{a~π}[min_j Q̄_j(s,a)] (ou max sur les tirages).
"""
from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from src.models.common import TwinQ, critic_input, mlp, polyak_update


class TanhGaussianPolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: list[int], log_std_min: float, log_std_max: float):
        super().__init__()
        self.net = mlp(obs_dim, hidden, 2 * act_dim)
        self.log_std_min, self.log_std_max = log_std_min, log_std_max

    def _params(self, obs):
        mu, log_std = self.net(obs).chunk(2, dim=-1)
        return mu, log_std.clamp(self.log_std_min, self.log_std_max)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Action reparamétrisée a = tanh(u) et log π(a|s) (correction du changement de variable)."""
        mu, log_std = self._params(obs)
        dist = Normal(mu, log_std.exp())
        u = dist.rsample()
        log_prob = dist.log_prob(u).sum(-1) - (2.0 * (math.log(2.0) - u - F.softplus(-2.0 * u))).sum(-1)
        return torch.tanh(u), log_prob

    def log_prob(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        mu, log_std = self._params(obs)
        a = actions.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        u = torch.atanh(a)
        return (Normal(mu, log_std.exp()).log_prob(u) - torch.log1p(-a.pow(2) + 1e-6)).sum(-1)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        if deterministic:
            return torch.tanh(self._params(obs)[0])
        return self.forward(obs)[0]


class CQL(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, cfg_model, cfg_train, max_timestep: int = 1000):
        super().__init__()
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.time_feature = bool(cfg_model.time_feature)
        self.max_timestep = max_timestep
        hidden = list(cfg_model.hidden_dims)
        crit_in = obs_dim + int(self.time_feature)

        self.qf = TwinQ(crit_in, act_dim, hidden)
        self.q_target = copy.deepcopy(self.qf).requires_grad_(False)
        self.policy = TanhGaussianPolicy(obs_dim, act_dim, hidden, cfg_model.log_std_min, cfg_model.log_std_max)
        self.log_alpha = nn.Parameter(torch.zeros(()))

        t = cfg_train
        self.discount, self.tau = t.discount, t.tau
        self.cql_alpha, self.n_actions, self.temp = t.cql_alpha, int(t.cql_n_actions), t.cql_temp
        self.bc_steps = int(t.bc_steps)
        self.target_entropy = -float(act_dim)
        self.value_estimate, self.value_samples = t.value_estimate, int(t.value_samples)
        self.train_policy = bool(t.train_policy)  # ici : évaluer la politique (elle est toujours entraînée)
        self.q_opt = torch.optim.Adam(self.qf.parameters(), lr=t.critic_lr)
        self.pi_opt = torch.optim.Adam(self.policy.parameters(), lr=t.policy_lr)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=t.alpha_lr)
        self.n_updates = 0  # entier Python : un tenseur sur le device forcerait une synchronisation par pas

    def critic_input(self, obs, timesteps):
        return critic_input(obs, timesteps, self.time_feature, self.max_timestep)

    @torch.no_grad()
    def q_value(self, obs, act, timesteps=None) -> torch.Tensor:
        return self.q_target(self.critic_input(obs, timesteps), act)

    @torch.no_grad()
    def value(self, obs, timesteps=None) -> torch.Tensor:
        """V(s) ≈ moyenne (ou max) de min_j Q̄_j(s, a) sur `value_samples` actions a ~ π(·|s)."""
        B, N = obs.shape[0], self.value_samples
        obs_rep = obs.repeat_interleave(N, 0)
        t_rep = None if timesteps is None else timesteps.repeat_interleave(N, 0)
        a, _ = self.policy(obs_rep)
        q = self.q_target(self.critic_input(obs_rep, t_rep), a).view(B, N)
        return q.max(1).values if self.value_estimate == "max" else q.mean(1)

    def update(self, b: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        obs, act, nxt = b["observations"], b["actions"], b["next_observations"]
        t = b.get("timesteps")
        x = self.critic_input(obs, t)
        x_next = self.critic_input(nxt, None if t is None else t + 1)
        B, N = obs.shape[0], self.n_actions

        # 1) politique + température d'entropie
        a_pi, logp_pi = self.policy(obs)
        alpha_loss = -(self.log_alpha * (logp_pi.detach() + self.target_entropy)).mean()
        alpha = self.log_alpha.exp().detach()
        if self.n_updates < self.bc_steps:  # échauffement optionnel par clonage comportemental
            pi_loss = (alpha * logp_pi - self.policy.log_prob(obs, act)).mean()
        else:
            pi_loss = (alpha * logp_pi - self.qf(x, a_pi)).mean()
        self.pi_opt.zero_grad(set_to_none=True)
        pi_loss.backward()
        self.pi_opt.step()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()

        # 2) TD
        with torch.no_grad():
            a_next, _ = self.policy(nxt)
            y = b["rewards"] + self.discount * (1.0 - b["dones"]) * self.q_target(x_next, a_next)
        q1, q2 = self.qf.both(x, act)
        td_loss = 0.5 * (F.mse_loss(q1, y) + F.mse_loss(q2, y))

        # 3) pénalité conservatrice CQL(H), logsumexp par échantillonnage préférentiel
        x_rep = x.repeat_interleave(N, 0)
        with torch.no_grad():
            a_cur, logp_cur = self.policy(obs.repeat_interleave(N, 0))
            a_nxt, logp_nxt = self.policy(nxt.repeat_interleave(N, 0))
        a_rand = torch.empty(B * N, self.act_dim, device=obs.device).uniform_(-1.0, 1.0)
        log_unif = self.act_dim * math.log(0.5)  # densité de U([-1, 1]^d)

        def q_samples(a):
            qa, qb = self.qf.both(x_rep, a)
            return qa.view(B, N), qb.view(B, N)

        (q1_r, q2_r), (q1_c, q2_c), (q1_n, q2_n) = q_samples(a_rand), q_samples(a_cur), q_samples(a_nxt)
        lc, ln = logp_cur.view(B, N), logp_nxt.view(B, N)
        penalties = []
        for qr, qc, qn, qd in ((q1_r, q1_c, q1_n, q1), (q2_r, q2_c, q2_n, q2)):
            cat = torch.cat([qr - log_unif, qn - ln, qc - lc], dim=1)
            lse = torch.logsumexp(cat / self.temp, dim=1) * self.temp
            penalties.append((lse - qd).mean())
        cql_loss = self.cql_alpha * (penalties[0] + penalties[1])

        q_loss = td_loss + cql_loss
        self.q_opt.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_opt.step()
        polyak_update(self.qf, self.q_target, self.tau)
        self.n_updates += 1

        return {"q_loss": td_loss.detach(), "cql_penalty": 0.5 * (penalties[0] + penalties[1]).detach(),
                "q_mean": q1.mean().detach(), "pi_loss": pi_loss.detach(), "alpha": alpha,
                "entropy": -logp_pi.mean().detach()}
