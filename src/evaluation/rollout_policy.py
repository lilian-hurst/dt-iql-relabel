"""Évaluation d'une politique markovienne (IQL, ...) : N épisodes en parallèle."""
from __future__ import annotations

from typing import Callable

import numpy as np

from src.data.registry import DatasetSpec, get_spec
from src.envs.make_env import make_env
from src.evaluation.normalization import normalized_score


def evaluate_policy(
    act_fn: Callable[[np.ndarray], np.ndarray],  # obs normalisées (N, obs_dim) -> actions (N, act_dim)
    spec: DatasetSpec | str,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    episodes: int = 10,
    seed: int = 0,
    max_ep_len: int | None = None,
) -> dict:
    spec = get_spec(spec) if isinstance(spec, str) else spec
    H = max_ep_len or spec.max_episode_len
    envs = [make_env(spec) for _ in range(episodes)]
    obs = np.stack([env.reset(seed=seed + i)[0] for i, env in enumerate(envs)]).astype(np.float32)
    ep_return, ep_len = np.zeros(episodes), np.zeros(episodes, dtype=np.int64)
    done = np.zeros(episodes, dtype=bool)
    lo, hi = envs[0].action_space.low, envs[0].action_space.high

    for _ in range(H):
        act = np.clip(act_fn((obs - state_mean) / state_std), lo, hi)
        for i in np.flatnonzero(~done):
            o, r, term, trunc, _ = envs[i].step(act[i])
            ep_return[i] += r
            ep_len[i] += 1
            obs[i] = o
            done[i] = term or trunc or ep_len[i] >= H
        if done.all():
            break
    for env in envs:
        env.close()
    scores = normalized_score(ep_return, spec)
    return {
        "return_mean": float(ep_return.mean()), "return_std": float(ep_return.std()),
        "score_mean": float(np.mean(scores)), "score_std": float(np.std(scores)),
        "length_mean": float(ep_len.mean()), "returns": ep_return.tolist(),
    }
