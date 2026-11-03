"""Relabellise les return-to-go d'un dataset D4RL avec un critique entraîné (IQL ou CQL).

    # IQL-DT : critique IQL + règle max(R_t, Q) (le critique est entraîné s'il manque)
    python -m src.training.relabel_returns --dataset hopper-medium-replay-v2 --seed 0 --train-critic
    # QDT : critique CQL + règle récursive de QDT
    python -m src.training.relabel_returns --dataset hopper-medium-replay-v2 --train-critic \
        --set value_source=cql rule=qdt
    # DT sur les RTG relabellisés
    python -m src.training.train_dt --dataset hopper-medium-replay-v2 --method iql_dt \
        --rtg-file runs/relabel/iql_max_q_a1/hopper-medium-replay-v2/seed0/rtg.npz

Sortie : runs/relabel/<source>_<rule>_a<alpha>[_th<seuil>]/<dataset>/seed<k>/{rtg.npz, stats.json}
  rtg.npz : 'rtg' (RTG relabellisés concaténés, non actualisés, échelle brute), 'lengths',
            'rtg_original', et pour l'analyse 'q', 'v_next', 'g_mc' (return MC actualisé), 'gamma'.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.data.d4rl_loader import load_dataset
from src.data.relabel import (critic_checkpoint, relabel_output_dir, relabel_stats,
                              relabel_trajectory)
from src.utils.config import load_config, save_config


def critic_values(agent, ck, tr: dict, device, chunk: int = 65536) -> tuple[np.ndarray, np.ndarray]:
    """Q(s_t, a_t) et V(s_{t+1}) pour toutes les transitions, à l'échelle des récompenses brutes."""
    import torch

    mean, std = ck["state_mean"], ck["state_std"]
    qs, vs = [], []
    with torch.no_grad():
        for i in range(0, len(tr["rewards"]), chunk):
            sl = slice(i, i + chunk)
            t = lambda x, dt=torch.float32: torch.as_tensor(x[sl], dtype=dt, device=device)
            ts = t(tr["timesteps"], torch.int64)
            obs = t((tr["observations"] - mean) / std)
            nxt = t((tr["next_observations"] - mean) / std)
            qs.append(agent.q_value(obs, t(tr["actions"]), ts).cpu().numpy())
            vs.append(agent.value(nxt, ts + 1).cpu().numpy())
    scale = ck["reward_scale"]
    return np.concatenate(qs) / scale, np.concatenate(vs) / scale


def relabel(
    dataset: str,
    seed: int = 0,
    overrides: list[str] = (),
    data_overrides: list[str] = (),
    output_dir: str | None = None,
    train_critic: bool = False,
) -> Path:
    cfg = load_config("relabel.yaml", overrides)
    data_cfg = load_config("data.yaml", data_overrides)
    exp_cfg = load_config("experiment.yaml")
    runs = Path(output_dir or exp_cfg.output_dir)

    ds = load_dataset(dataset, root=data_cfg.root, max_return_quantile=data_cfg.max_return_quantile)
    tr = ds.transitions()
    lengths = ds.lengths()
    split = lambda x: np.split(x, np.cumsum(lengths)[:-1])
    original = [t.returns_to_go() for t in ds.trajectories]

    q = v_next = None
    gamma = 1.0
    if cfg.rule != "none":
        from src.training.train_offline import load_agent, train as train_offline
        from src.utils.seed import get_device

        device = get_device(exp_cfg.device)
        critic = cfg.critics[cfg.value_source]
        ckpt = critic_checkpoint(cfg, dataset, seed, runs)
        if not ckpt.exists():
            if not train_critic:
                raise FileNotFoundError(f"critique introuvable : {ckpt} (relancer avec --train-critic)")
            train_offline(cfg.value_source, dataset, seed, list(critic.overrides), critic.method, str(runs),
                          data_overrides)
        agent, ck = load_agent(ckpt, device)
        if ck["algo"] != cfg.value_source:
            raise ValueError(f"{ckpt} est un critique {ck['algo']}, value_source={cfg.value_source}")
        if ck.get("max_return_quantile", 1.0) != data_cfg.max_return_quantile:
            raise ValueError("le critique a été entraîné sur un autre filtrage du dataset (max_return_quantile)")
        gamma = float(ck["discount"])
        q, v_next = critic_values(agent, ck, tr, device)
        print(f"[relabel] critique {ck['algo']} {ckpt} : γ={gamma}, time_feature={ck['time_feature']}")

    new_rtgs = []
    q_s = split(q) if q is not None else [None] * len(lengths)
    v_s = split(v_next) if v_next is not None else [None] * len(lengths)
    for traj, qt, vt in zip(ds.trajectories, q_s, v_s):
        new_rtgs.append(relabel_trajectory(traj.rewards, qt, vt, cfg.rule, cfg.alpha, gamma, cfg.min_improvement)[0])

    stats = relabel_stats(original, new_rtgs)
    stats.update(dataset=dataset, seed=seed, rule=cfg.rule, alpha=cfg.alpha, gamma=gamma,
                 min_improvement=cfg.min_improvement, value_source=cfg.value_source,
                 max_return_quantile=data_cfg.max_return_quantile)
    G = np.zeros(0)
    if q is not None:
        G = np.concatenate([t.returns_to_go(gamma) for t in ds.trajectories])
        stats["q_minus_mc_mean"] = float((q - G).mean())        # biais du critique vs Monte-Carlo
        stats["q_mc_correlation"] = float(np.corrcoef(q, G)[0, 1])

    out = relabel_output_dir(cfg, dataset, seed, runs)
    out.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out / "config.yaml")
    with open(out / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    # rtg.npz écrit en dernier : sa présence marque une relabellisation terminée
    np.savez_compressed(
        out / "rtg.npz", rtg=np.concatenate(new_rtgs).astype(np.float32), lengths=lengths,
        rtg_original=np.concatenate(original).astype(np.float32),
        q=q if q is not None else np.zeros(0), v_next=v_next if v_next is not None else np.zeros(0),
        g_mc=G.astype(np.float32), gamma=np.float32(gamma),  # return Monte-Carlo actualisé (diagnostic)
    )
    print(f"[relabel] {stats['frac_steps_relabeled']:.1%} des pas relabellisés, gain moyen "
          f"{stats['mean_gain']:.1f} ; R̂_0 moyen {stats['initial_rtg_mean_before']:.1f} -> "
          f"{stats['initial_rtg_mean_after']:.1f} -> {out / 'rtg.npz'}")
    return out / "rtg.npz"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--seed", type=int, default=0, help="seed du critique (et du dossier de sortie)")
    p.add_argument("--train-critic", action="store_true", help="entraîne le critique s'il est absent")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--set", nargs="*", default=[], metavar="CLÉ=VALEUR", help="surcharges de config/relabel.yaml")
    p.add_argument("--data-set", nargs="*", default=[], metavar="CLÉ=VALEUR", help="surcharges de config/data.yaml")
    a = p.parse_args()
    relabel(a.dataset, a.seed, a.set, a.data_set, a.output_dir, a.train_critic)


if __name__ == "__main__":
    main()
