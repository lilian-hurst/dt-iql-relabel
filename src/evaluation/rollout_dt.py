"""Évaluation en environnement d'un Decision Transformer conditionné par un return cible.

Protocole du papier : R̂_0 = return cible ; après chaque pas, R̂_{t+1} = R̂_t - r_t.
Les N épisodes tournent en parallèle (N environnements avancés en lockstep, un seul
appel au modèle par pas de temps) ; un épisode terminé est simplement gelé.
"""
from __future__ import annotations

import numpy as np
import torch

from src.data.registry import DatasetSpec, get_spec
from src.envs.make_env import make_env
from src.evaluation.normalization import normalized_score


@torch.no_grad()
def evaluate_dt(
    model,
    spec: DatasetSpec | str,
    target_return: float,
    return_scale: float,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    episodes: int = 10,
    seed: int = 0,
    device: str | torch.device = "cpu",
    max_ep_len: int | None = None,
) -> dict:
    spec = get_spec(spec) if isinstance(spec, str) else spec
    H = max_ep_len or spec.max_episode_len
    N = episodes
    device = torch.device(device)
    was_training = model.training
    model.eval()

    envs = [make_env(spec) for _ in range(N)]
    mean = torch.as_tensor(state_mean, device=device)
    std = torch.as_tensor(state_std, device=device)
    obs_dim, act_dim = model.obs_dim, model.act_dim

    S = torch.zeros(N, H, obs_dim, device=device)
    A = torch.zeros(N, H, act_dim, device=device)
    R = torch.zeros(N, H, 1, device=device)
    T = torch.arange(H, device=device).unsqueeze(0).expand(N, H)

    obs = np.stack([env.reset(seed=seed + i)[0] for i, env in enumerate(envs)])
    S[:, 0] = (torch.as_tensor(obs, dtype=torch.float32, device=device) - mean) / std
    R[:, 0] = target_return / return_scale

    ep_return = np.zeros(N)
    ep_len = np.zeros(N, dtype=np.int64)
    done = np.zeros(N, dtype=bool)
    lo_act, hi_act = envs[0].action_space.low, envs[0].action_space.high

    for t in range(H):
        lo = max(0, t + 1 - model.context_len)
        act = model.get_action(S[:, lo : t + 1], A[:, lo : t + 1], R[:, lo : t + 1], T[:, lo : t + 1])
        A[:, t] = act
        act_np = np.clip(act.cpu().numpy(), lo_act, hi_act)
        for i in np.flatnonzero(~done):
            o, r, term, trunc, _ = envs[i].step(act_np[i])
            ep_return[i] += r
            ep_len[i] += 1
            if term or trunc or t + 1 >= H:
                done[i] = True
            else:
                S[i, t + 1] = (torch.as_tensor(o, dtype=torch.float32, device=device) - mean) / std
                R[i, t + 1] = R[i, t] - r / return_scale
        if done.all():
            break

    for env in envs:
        env.close()
    model.train(was_training)
    scores = normalized_score(ep_return, spec)
    return {
        "target_return": float(target_return),
        "return_mean": float(ep_return.mean()),
        "return_std": float(ep_return.std()),
        "score_mean": float(np.mean(scores)),
        "score_std": float(np.std(scores)),
        "length_mean": float(ep_len.mean()),
        "returns": ep_return.tolist(),
    }
