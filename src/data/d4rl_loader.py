"""Chargement des datasets offline (D4RL HDF5 / Minari) sous une forme commune.

Représentation :
  * `Trajectory` : un épisode (observations, actions, rewards, dones) + RTG Monte-Carlo.
  * `OfflineDataset` : liste de trajectoires, statistiques de normalisation, et vue
    "transitions" (s, a, r, s', done) pour IQL.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.data.registry import DatasetSpec, get_spec


# --------------------------------------------------------------------------------------
# Structures
# --------------------------------------------------------------------------------------
@dataclass
class Trajectory:
    observations: np.ndarray        # (T, obs_dim)
    actions: np.ndarray             # (T, act_dim)
    rewards: np.ndarray             # (T,)
    terminals: np.ndarray           # (T,) bool — vraie fin d'épisode (pas un timeout)
    next_observations: np.ndarray   # (T, obs_dim)
    infos: dict = field(default_factory=dict)  # ex. qpos/qvel pour les tests de rejeu

    def __len__(self) -> int:
        return len(self.rewards)

    @property
    def episode_return(self) -> float:
        return float(self.rewards.sum())

    def returns_to_go(self, gamma: float = 1.0) -> np.ndarray:
        return discount_cumsum(self.rewards, gamma)


def discount_cumsum(x: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """RTG_t = sum_{t'>=t} gamma^{t'-t} x_{t'}  (gamma=1 pour le DT)."""
    out = np.zeros(len(x), dtype=np.float64)
    acc = 0.0
    for t in range(len(x) - 1, -1, -1):
        acc = x[t] + gamma * acc
        out[t] = acc
    return out


@dataclass
class OfflineDataset:
    spec: DatasetSpec
    trajectories: list[Trajectory]
    state_mean: np.ndarray = None
    state_std: np.ndarray = None

    def __post_init__(self):
        if self.state_mean is None:
            self.compute_state_stats()

    # -- statistiques -------------------------------------------------------------
    def compute_state_stats(self, eps: float = 1e-6) -> None:
        obs = np.concatenate([t.observations for t in self.trajectories], axis=0)
        self.state_mean = obs.mean(0).astype(np.float32)
        self.state_std = (obs.std(0) + eps).astype(np.float32)

    def normalize_states(self, obs: np.ndarray) -> np.ndarray:
        return (obs - self.state_mean) / self.state_std

    @property
    def obs_dim(self) -> int:
        return self.trajectories[0].observations.shape[1]

    @property
    def act_dim(self) -> int:
        return self.trajectories[0].actions.shape[1]

    @property
    def num_transitions(self) -> int:
        return sum(len(t) for t in self.trajectories)

    def episode_returns(self) -> np.ndarray:
        return np.array([t.episode_return for t in self.trajectories])

    # -- vue transitions (IQL) ----------------------------------------------------
    def transitions(self) -> dict[str, np.ndarray]:
        """Transitions concaténées dans l'ordre des trajectoires.

        timesteps : indice t du pas dans son épisode (pour les critiques dépendant du temps).
        """
        cat = lambda key: np.concatenate([getattr(t, key) for t in self.trajectories], axis=0)
        return {
            "observations": cat("observations").astype(np.float32),
            "actions": cat("actions").astype(np.float32),
            "rewards": cat("rewards").astype(np.float32),
            "next_observations": cat("next_observations").astype(np.float32),
            "terminals": cat("terminals").astype(np.float32),
            "timesteps": np.concatenate([np.arange(len(t)) for t in self.trajectories]).astype(np.int64),
        }

    def lengths(self) -> np.ndarray:
        return np.array([len(t) for t in self.trajectories], dtype=np.int64)

    # -- filtrage par qualité -----------------------------------------------------
    def filter_by_return_quantile(self, q: float) -> "OfflineDataset":
        """Garde les trajectoires dont le return est <= quantile q (ablation qualité / stitching)."""
        if q >= 1.0:
            return self
        rets = self.episode_returns()
        thr = np.quantile(rets, q)
        kept = [t for t, r in zip(self.trajectories, rets) if r <= thr]
        return OfflineDataset(self.spec, kept)  # stats de normalisation recalculées

    def summary(self) -> dict:
        from src.evaluation.normalization import normalized_score

        rets = self.episode_returns()
        lens = np.array([len(t) for t in self.trajectories])
        return {
            "dataset": self.spec.name,
            "trajectories": len(self.trajectories),
            "transitions": int(lens.sum()),
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "ep_len_mean": float(lens.mean()),
            "return_min": float(rets.min()),
            "return_mean": float(rets.mean()),
            "return_max": float(rets.max()),
            "norm_score_mean": float(normalized_score(rets.mean(), self.spec)),
            "norm_score_max": float(normalized_score(rets.max(), self.spec)),
        }


# --------------------------------------------------------------------------------------
# Téléchargement
# --------------------------------------------------------------------------------------
def dataset_root(root: str | os.PathLike | None = None) -> Path:
    root = os.environ.get("D4RL_DATASET_DIR") or root or "datasets"
    path = Path(root).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def download(url: str, dest: Path) -> Path:
    """Téléchargement atomique (fichier temporaire puis renommage) avec barre de progression."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[data] téléchargement {url} -> {dest}", file=sys.stderr)
    fd, tmp = tempfile.mkstemp(dir=dest.parent, suffix=".part")
    try:
        with urllib.request.urlopen(url) as resp, os.fdopen(fd, "wb") as out:
            total = int(resp.headers.get("Content-Length") or 0)
            done = 0
            while chunk := resp.read(1 << 20):
                out.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {done / 1e6:7.1f} / {total / 1e6:.1f} Mo", end="", file=sys.stderr)
        print(file=sys.stderr)
        shutil.move(tmp, dest)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return dest


# --------------------------------------------------------------------------------------
# Backend HDF5 (D4RL Gym-MuJoCo v2)
# --------------------------------------------------------------------------------------
def load_hdf5(path: str | Path, spec: DatasetSpec, keep_infos: bool = False) -> list[Trajectory]:
    """Découpe le fichier D4RL en épisodes sur terminals | timeouts (comme le code officiel du DT)."""
    import h5py

    with h5py.File(path, "r") as f:
        obs = f["observations"][:].astype(np.float32)
        act = f["actions"][:].astype(np.float32)
        rew = f["rewards"][:].astype(np.float32).reshape(-1)
        term = f["terminals"][:].astype(bool).reshape(-1)
        timeouts = f["timeouts"][:].astype(bool).reshape(-1) if "timeouts" in f else np.zeros_like(term)
        next_obs = f["next_observations"][:].astype(np.float32) if "next_observations" in f else None
        infos = {}
        if keep_infos and "infos" in f:
            for k in ("qpos", "qvel"):
                if k in f["infos"]:
                    infos[k] = f["infos"][k][:]

    n = len(rew)
    if next_obs is None:
        # s'_t = s_{t+1} ; faux uniquement à une frontière d'épisode, où il est ignoré
        # (terminal => pas de bootstrap) ou retiré (timeout, voir plus bas).
        next_obs = np.concatenate([obs[1:], obs[-1:]], axis=0)
        drop_last_of_timeout = True
    else:
        drop_last_of_timeout = False

    ends = np.flatnonzero(term | timeouts)
    if len(ends) == 0 or ends[-1] != n - 1:
        ends = np.append(ends, n - 1)  # dernier épisode incomplet (medium-replay)

    trajs, start = [], 0
    for end in ends:
        sl = slice(start, end + 1)
        if drop_last_of_timeout and timeouts[end] and not term[end] and end > start:
            sl = slice(start, end)  # s' du dernier pas inconnu
        if sl.stop - sl.start > 0:
            trajs.append(Trajectory(
                observations=obs[sl], actions=act[sl], rewards=rew[sl], terminals=term[sl],
                next_observations=next_obs[sl],
                infos={k: v[sl] for k, v in infos.items()},
            ))
        start = end + 1
    return trajs


# --------------------------------------------------------------------------------------
# Backend Minari (Franka Kitchen)
# --------------------------------------------------------------------------------------
def load_minari(spec: DatasetSpec) -> list[Trajectory]:
    try:
        import minari
    except ImportError as e:
        raise ImportError("Franka Kitchen nécessite : pip install -r requirements-kitchen.txt") from e

    ds = minari.load_dataset(spec.source, download=True)
    trajs = []
    for ep in ds.iterate_episodes():
        o = ep.observations
        o = o["observation"] if isinstance(o, dict) else o  # Kitchen : obs = dict, on garde l'état robot+objets
        o = np.asarray(o, dtype=np.float32)
        trajs.append(Trajectory(
            observations=o[:-1], next_observations=o[1:],
            actions=np.asarray(ep.actions, dtype=np.float32),
            rewards=np.asarray(ep.rewards, dtype=np.float32),
            terminals=np.asarray(ep.terminations, dtype=bool),
        ))
    return trajs


# --------------------------------------------------------------------------------------
# Point d'entrée
# --------------------------------------------------------------------------------------
def load_dataset(
    name: str,
    root: str | None = None,
    path: str | Path | None = None,
    max_return_quantile: float = 1.0,
    keep_infos: bool = False,
) -> OfflineDataset:
    """Charge `name` (ex. 'hopper-medium-replay-v2').

    path : fichier HDF5 local à utiliser à la place du téléchargement (tests, miroirs).
    """
    spec = get_spec(name)
    if spec.backend == "hdf5":
        file = Path(path) if path else download(spec.source, dataset_root(root) / spec.filename)
        trajs = load_hdf5(file, spec, keep_infos=keep_infos)
    else:
        trajs = load_minari(spec)
    ds = OfflineDataset(spec, trajs)
    return ds.filter_by_return_quantile(max_return_quantile)
