"""Decision Transformer (Chen et al., NeurIPS 2021) — implémentation PyTorch autonome.

Séquence d'entrée (K pas de temps -> 3K tokens) :  R̂_1, s_1, a_1, R̂_2, s_2, a_2, ...
  * chaque modalité a son embedding linéaire ; un embedding de pas de temps (appris,
    indexé par le t absolu dans l'épisode) est ajouté aux trois tokens du même pas ;
  * pas d'embedding positionnel supplémentaire (comme le code officiel, qui passe des
    position_ids nuls à GPT-2) ;
  * Transformer causal type GPT-2 (pré-LayerNorm, MLP 4x) ;
  * l'action a_t est prédite à partir de la sortie du token s_t : elle voit R̂_≤t, s_≤t, a_<t.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

_ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU}


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, dropout: float):
        super().__init__()
        if embed_dim % n_heads:
            raise ValueError("embed_dim doit être divisible par n_heads")
        self.n_heads = n_heads
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = dropout
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (z.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) for z in (q, k, v))
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.attn_dropout if self.training else 0.0
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(y))


class Block(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, dropout: float, activation: str):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, n_heads, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            _ACTIVATIONS[activation](),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), attn_mask)
        return x + self.mlp(self.ln2(x))


class DecisionTransformer(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        context_len: int = 20,
        embed_dim: int = 128,
        n_layers: int = 3,
        n_heads: int = 1,
        dropout: float = 0.1,
        activation: str = "relu",
        max_timestep: int = 4096,
        tanh_action: bool = True,
    ):
        super().__init__()
        self.obs_dim, self.act_dim, self.context_len = obs_dim, act_dim, context_len
        self.max_timestep = max_timestep

        self.embed_timestep = nn.Embedding(max_timestep, embed_dim)
        self.embed_return = nn.Linear(1, embed_dim)
        self.embed_state = nn.Linear(obs_dim, embed_dim)
        self.embed_action = nn.Linear(act_dim, embed_dim)
        self.embed_ln = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(Block(embed_dim, n_heads, dropout, activation) for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(embed_dim)
        head = [nn.Linear(embed_dim, act_dim)]
        if tanh_action:
            head.append(nn.Tanh())
        self.predict_action = nn.Sequential(*head)

        causal = torch.tril(torch.ones(3 * context_len, 3 * context_len, dtype=torch.bool))
        self.register_buffer("causal_mask", causal, persistent=False)
        self.apply(self._init_weights)
        for name, p in self.named_parameters():  # init GPT-2 des projections résiduelles
            if name.endswith("proj.weight") or name.endswith("mlp.2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layers))

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(
        self,
        states: torch.Tensor,      # (B, K, obs_dim), normalisés
        actions: torch.Tensor,     # (B, K, act_dim)
        rtg: torch.Tensor,         # (B, K, 1), divisés par return_scale
        timesteps: torch.Tensor,   # (B, K) long
        mask: torch.Tensor | None = None,  # (B, K) 1 = token réel, 0 = padding
    ) -> torch.Tensor:
        B, K, _ = states.shape
        if K > self.context_len:
            raise ValueError(f"séquence de longueur {K} > context_len {self.context_len}")
        timesteps = timesteps.clamp(0, self.max_timestep - 1)
        t_emb = self.embed_timestep(timesteps)
        r = self.embed_return(rtg) + t_emb
        s = self.embed_state(states) + t_emb
        a = self.embed_action(actions) + t_emb
        h = torch.stack((r, s, a), dim=2).reshape(B, 3 * K, -1)  # R1 s1 a1 R2 s2 a2 ...
        h = self.drop(self.embed_ln(h))

        attn = self.causal_mask[: 3 * K, : 3 * K].unsqueeze(0).unsqueeze(0)  # (1,1,3K,3K)
        if mask is not None:
            keys = mask.bool().repeat_interleave(3, dim=1)                  # (B, 3K)
            attn = attn & keys[:, None, None, :]
            # Chaque token peut s'attendre lui-même : évite les lignes entièrement masquées
            # (NaN) pour les tokens de padding, dont la sortie est ignorée par la loss.
            attn = attn | torch.eye(3 * K, dtype=torch.bool, device=h.device)
        for blk in self.blocks:
            h = blk(h, attn)
        h = self.ln_f(h).reshape(B, K, 3, -1)
        return self.predict_action(h[:, :, 1])  # sortie des tokens d'état -> a_t

    @torch.no_grad()
    def get_action(
        self,
        states: torch.Tensor,     # (N, T, obs_dim)  historique complet (T >= 1)
        actions: torch.Tensor,    # (N, T, act_dim)  l'action courante a_T est un placeholder (0)
        rtg: torch.Tensor,        # (N, T, 1)
        timesteps: torch.Tensor,  # (N, T)
    ) -> torch.Tensor:
        """Action pour le dernier pas de temps, à partir des K derniers pas (padding à gauche)."""
        K = self.context_len
        states, actions, rtg, timesteps = states[:, -K:], actions[:, -K:], rtg[:, -K:], timesteps[:, -K:]
        N, T = states.shape[:2]
        pad = K - T
        mask = torch.cat([torch.zeros(N, pad, device=states.device), torch.ones(N, T, device=states.device)], 1)
        if pad:
            zpad = lambda x: torch.cat([x.new_zeros((N, pad) + x.shape[2:]), x], dim=1)
            states, actions, rtg, timesteps = map(zpad, (states, actions, rtg, timesteps))
        return self.forward(states, actions, rtg, timesteps, mask)[:, -1]


def build_dt(cfg_model, obs_dim: int, act_dim: int) -> DecisionTransformer:
    return DecisionTransformer(
        obs_dim=obs_dim,
        act_dim=act_dim,
        context_len=cfg_model.context_len,
        embed_dim=cfg_model.embed_dim,
        n_layers=cfg_model.n_layers,
        n_heads=cfg_model.n_heads,
        dropout=cfg_model.dropout,
        activation=cfg_model.activation,
        max_timestep=cfg_model.max_timestep,
        tanh_action=cfg_model.tanh_action,
    )
