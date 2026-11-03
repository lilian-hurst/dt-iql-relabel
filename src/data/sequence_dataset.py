"""Découpage des trajectoires en sous-séquences de longueur K pour le Decision Transformer.

Reprend le schéma d'échantillonnage du code officiel (gym/experiment.py) :
  * une trajectoire est tirée avec une probabilité proportionnelle à sa longueur,
    donc chaque pas de temps du dataset est (approximativement) équiprobable ;
  * un indice de départ si ~ U{0, ..., T-1} ; la fenêtre [si, si+K) est tronquée en fin
    d'épisode puis complétée À GAUCHE (padding) avec un masque d'attention à 0 ;
  * états normalisés (s - mean) / std, RTG divisés par `return_scale`.

Les RTG sont fournis par trajectoire : RTG Monte-Carlo par défaut, ou RTG relabellisés
(IQL / QDT) à l'étape 3 — le reste du pipeline DT est inchangé.
Ce module ne dépend que de NumPy (testable sans PyTorch).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from src.data.d4rl_loader import OfflineDataset


class SequenceDataset:
    def __init__(
        self,
        dataset: OfflineDataset,
        context_len: int,
        return_scale: float,
        max_timestep: int,
        rtgs: Sequence[np.ndarray] | None = None,
        normalize_states: bool = True,
    ):
        trajs = dataset.trajectories
        if rtgs is None:
            rtgs = [t.returns_to_go() for t in trajs]
        if len(rtgs) != len(trajs) or any(len(r) != len(t) for r, t in zip(rtgs, trajs)):
            raise ValueError("rtgs doit contenir un tableau de même longueur que chaque trajectoire")

        self.K = int(context_len)
        self.return_scale = float(return_scale)
        self.max_timestep = int(max_timestep)
        self.obs_dim, self.act_dim = dataset.obs_dim, dataset.act_dim
        if normalize_states:
            self.state_mean, self.state_std = dataset.state_mean, dataset.state_std
        else:
            self.state_mean = np.zeros(self.obs_dim, np.float32)
            self.state_std = np.ones(self.obs_dim, np.float32)

        self.states = [((t.observations - self.state_mean) / self.state_std).astype(np.float32) for t in trajs]
        self.actions = [t.actions.astype(np.float32) for t in trajs]
        self.rtgs = [(np.asarray(r, np.float64) / self.return_scale).astype(np.float32) for r in rtgs]
        lengths = np.array([len(t) for t in trajs], dtype=np.int64)
        self.lengths = lengths
        self.p_sample = lengths / lengths.sum()

    def __len__(self) -> int:
        return int(self.lengths.sum())

    def get_window(self, i: int, start: int) -> dict[str, np.ndarray]:
        """Fenêtre [start, start+K) de la trajectoire i, complétée à gauche jusqu'à K."""
        K = self.K
        end = min(start + K, self.lengths[i])
        L = end - start
        pad = K - L
        out = {
            "states": np.zeros((K, self.obs_dim), np.float32),
            "actions": np.zeros((K, self.act_dim), np.float32),
            "rtg": np.zeros((K, 1), np.float32),
            "timesteps": np.zeros(K, np.int64),
            "mask": np.zeros(K, np.float32),
        }
        out["states"][pad:] = self.states[i][start:end]
        out["actions"][pad:] = self.actions[i][start:end]
        out["rtg"][pad:, 0] = self.rtgs[i][start:end]
        out["timesteps"][pad:] = np.minimum(np.arange(start, end), self.max_timestep - 1)
        out["mask"][pad:] = 1.0
        return out

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        idx = rng.choice(len(self.lengths), size=batch_size, p=self.p_sample)
        starts = (rng.random(batch_size) * self.lengths[idx]).astype(np.int64)
        windows = [self.get_window(i, s) for i, s in zip(idx, starts)]
        return {k: np.stack([w[k] for w in windows]) for k in windows[0]}
