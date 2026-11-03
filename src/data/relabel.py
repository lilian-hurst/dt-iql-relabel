"""Règles de relabellisation des return-to-go (NumPy pur, testable sans PyTorch).

Voir config/relabel.yaml pour les formules. Toutes les valeurs (rewards, q, v_next) sont
à l'échelle des récompenses brutes du dataset.
"""
from __future__ import annotations

import numpy as np

from src.data.d4rl_loader import discount_cumsum

RULES = ("none", "max_q", "qdt")


def relabel_trajectory(
    rewards: np.ndarray,
    q: np.ndarray | None = None,       # Q(s_t, a_t), forme (T,)
    v_next: np.ndarray | None = None,  # V(s_{t+1}), forme (T,) ; v_next[T-1] ignoré
    rule: str = "max_q",
    alpha: float = 1.0,
    gamma: float = 1.0,
    min_improvement: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Renvoie (RTG relabellisés non actualisés, gain ajouté à chaque pas)."""
    rewards = np.asarray(rewards, dtype=np.float64)
    T = len(rewards)
    R = discount_cumsum(rewards, 1.0)
    if rule == "none":
        return R, np.zeros(T)

    if rule == "max_q":
        G = discount_cumsum(rewards, gamma)
        gap = np.maximum(0.0, np.asarray(q, np.float64) - G)
        gap = np.where(gap > min_improvement * np.abs(G), gap, 0.0) * alpha
        return R + gap, gap

    if rule == "qdt":
        v_next = np.asarray(v_next, np.float64)
        R_hat, G_hat, gaps = np.empty(T), np.empty(T), np.zeros(T)
        R_hat[-1] = G_hat[-1] = rewards[-1]
        for t in range(T - 2, -1, -1):
            gap = max(0.0, v_next[t] - G_hat[t + 1])
            if gap <= min_improvement * abs(G_hat[t + 1]):
                gap = 0.0
            gap *= alpha
            gaps[t] = gap
            G_hat[t] = rewards[t] + gamma * (G_hat[t + 1] + gap)
            R_hat[t] = rewards[t] + R_hat[t + 1] + gap
        return R_hat, gaps

    raise ValueError(f"règle inconnue {rule!r} ; choix : {RULES}")


def relabel_tag(cfg) -> str:
    """Nom du dossier d'une configuration de relabellisation, ex. 'iql_max_q_a1'."""
    tag = f"{cfg.value_source}_{cfg.rule}_a{float(cfg.alpha):g}"
    if cfg.min_improvement:
        tag += f"_th{float(cfg.min_improvement):g}"
    return tag


def relabel_output_dir(cfg, dataset: str, seed: int, runs_dir) -> "Path":
    from pathlib import Path
    return Path(runs_dir) / "relabel" / relabel_tag(cfg) / dataset / f"seed{seed}"


def critic_checkpoint(cfg, dataset: str, seed: int, runs_dir) -> "Path":
    from pathlib import Path
    if cfg.checkpoint:
        return Path(cfg.checkpoint)
    return Path(runs_dir) / cfg.critics[cfg.value_source].method / dataset / f"seed{seed}" / "model.pt"


def relabel_stats(original: list[np.ndarray], relabeled: list[np.ndarray]) -> dict:
    """Statistiques pour le rapport : part des pas modifiés, gain moyen, effet sur R̂_0."""
    o, r = np.concatenate(original), np.concatenate(relabeled)
    changed = r > o + 1e-6
    o0 = np.array([x[0] for x in original])
    r0 = np.array([x[0] for x in relabeled])
    return {
        "frac_steps_relabeled": float(changed.mean()),
        "mean_gain": float((r - o).mean()),
        "mean_gain_when_relabeled": float((r - o)[changed].mean()) if changed.any() else 0.0,
        "initial_rtg_mean_before": float(o0.mean()),
        "initial_rtg_mean_after": float(r0.mean()),
        "initial_rtg_max_before": float(o0.max()),
        "initial_rtg_max_after": float(r0.max()),
    }
