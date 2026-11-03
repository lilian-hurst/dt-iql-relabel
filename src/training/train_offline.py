"""Entraînement d'un agent offline RL (IQL ou CQL) : baseline, ou critique de relabellisation.

    python -m src.training.train_offline --algo iql --dataset hopper-medium-replay-v2 --seed 0
    python -m src.training.train_offline --algo cql --dataset hopper-medium-replay-v2 --method cql_critic \
        --set train.discount=0.995 model.time_feature=true train.train_policy=false

Sorties : runs/<method>/<dataset>/seed<k>/{config.yaml, metrics.jsonl, model.pt, results.json}
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.data.d4rl_loader import OfflineDataset, load_dataset
from src.data.registry import get_spec
from src.evaluation.rollout_policy import evaluate_policy
from src.utils.config import Config, load_config, save_config
from src.utils.logger import Logger
from src.utils.seed import get_device, set_seed


def _algos() -> dict:
    from src.models.cql import CQL
    from src.models.iql import IQL
    return {"iql": ("iql.yaml", IQL), "cql": ("cql.yaml", CQL)}


def reward_scale(ds: OfflineDataset, mode: str) -> float:
    """Facteur multiplicatif appliqué aux récompenses pendant l'entraînement."""
    if mode == "none" or (mode == "auto" and ds.spec.domain == "kitchen"):
        return 1.0
    if mode == "auto":  # Gym-MuJoCo, comme l'implémentation officielle d'IQL
        rets = ds.episode_returns()
        return float(1000.0 / (rets.max() - rets.min()))
    raise ValueError(f"reward_normalization inconnu : {mode!r}")


def make_buffer(ds: OfflineDataset, scale: float, time_feature: bool, device) -> dict:
    tr = ds.transitions()
    dones = tr["terminals"].copy()
    if time_feature:
        # horizon fini : aucune valeur au-delà de H (fin par timeout = fin réelle pour V(s, t))
        dones = np.maximum(dones, (tr["timesteps"] + 1 >= ds.spec.max_episode_len).astype(np.float32))
    buf = {"observations": ds.normalize_states(tr["observations"]), "actions": tr["actions"],
           "rewards": tr["rewards"] * scale, "next_observations": ds.normalize_states(tr["next_observations"]),
           "dones": dones, "timesteps": tr["timesteps"]}
    return {k: torch.as_tensor(v, device=device) for k, v in buf.items()}


def load_agent(path: str | Path, device="cpu"):
    """Recharge un agent sauvegardé ; renvoie (agent, checkpoint)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    _, cls = _algos()[ck["algo"]]
    agent = cls(ck["obs_dim"], ck["act_dim"], Config(ck["cfg"]["model"]), Config(ck["cfg"]["train"]),
                ck["max_timestep"]).to(device)
    agent.load_state_dict(ck["model"])
    return agent.eval(), ck


def train(
    algo: str,
    dataset: str,
    seed: int = 0,
    overrides: list[str] = (),
    method: str | None = None,
    output_dir: str | None = None,
    data_overrides: list[str] = (),
) -> dict:
    cfg_file, agent_cls = _algos()[algo]
    method = method or algo
    spec = get_spec(dataset)
    cfg = load_config(cfg_file, overrides, domain=spec.domain)
    data_cfg = load_config("data.yaml", data_overrides)
    exp_cfg = load_config("experiment.yaml")
    set_seed(seed, exp_cfg.deterministic)
    device = get_device(exp_cfg.device)
    gen = torch.Generator(device=device).manual_seed(seed)

    out = Path(output_dir or exp_cfg.output_dir) / method / dataset / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    save_config(Config({algo: cfg, "data": data_cfg, "dataset": dataset, "seed": seed, "method": method}),
                out / "config.yaml")

    ds = load_dataset(dataset, root=data_cfg.root, max_return_quantile=data_cfg.max_return_quantile)
    if not data_cfg.normalize_states:
        ds.state_mean = np.zeros(ds.obs_dim, np.float32)
        ds.state_std = np.ones(ds.obs_dim, np.float32)
    scale = reward_scale(ds, cfg.train.reward_normalization)
    buf = make_buffer(ds, scale, cfg.model.time_feature, device)
    n = buf["rewards"].shape[0]
    print(f"[{algo}] {dataset} : {n} transitions, reward_scale={scale:.4g}, γ={cfg.train.discount}, "
          f"time_feature={cfg.model.time_feature} | device={device} | sortie={out}")

    agent = agent_cls(ds.obs_dim, ds.act_dim, cfg.model, cfg.train, spec.max_episode_len).to(device)
    logger = Logger(out, exp_cfg.logger.backend, exp_cfg.logger.project, f"{method}-{dataset}-s{seed}",
                    cfg.to_dict())

    def act_fn(obs_np: np.ndarray) -> np.ndarray:
        return agent.policy.act(torch.as_tensor(obs_np, dtype=torch.float32, device=device)).cpu().numpy()

    def evaluate(episodes: int, eval_seed: int) -> dict:
        agent.eval()
        r = evaluate_policy(act_fn, spec, ds.state_mean, ds.state_std, episodes=episodes, seed=eval_seed)
        agent.train()
        return r

    def save_ckpt(step: int) -> None:
        torch.save({
            "algo": algo, "model": agent.state_dict(),
            "cfg": {"model": cfg.model.to_dict(), "train": cfg.train.to_dict()},
            "step": step, "obs_dim": ds.obs_dim, "act_dim": ds.act_dim, "dataset": dataset,
            "max_timestep": spec.max_episode_len, "state_mean": ds.state_mean, "state_std": ds.state_std,
            "reward_scale": scale, "discount": cfg.train.discount, "time_feature": cfg.model.time_feature,
            "max_return_quantile": data_cfg.max_return_quantile,
        }, out / "model.pt")

    agent.train()
    t0, acc = time.time(), {}
    for step in range(1, cfg.train.max_steps + 1):
        idx = torch.randint(0, n, (cfg.train.batch_size,), device=device, generator=gen)
        m = agent.update({k: v[idx] for k, v in buf.items()})
        for k, v in m.items():
            acc.setdefault(k, []).append(v)

        if step % cfg.train.log_every == 0:
            row = {f"train/{k}": float(np.mean(v)) for k, v in acc.items()}
            row["train/steps_per_s"] = cfg.train.log_every / (time.time() - t0)
            logger.log(row, step)
            logger.print(row, step)
            acc, t0 = {}, time.time()
        if (cfg.train.train_policy and cfg.train.eval_every and step % cfg.train.eval_every == 0
                and step < cfg.train.max_steps):
            r = evaluate(cfg.eval.episodes, seed)
            logger.log({"eval/score": r["score_mean"]}, step)
            logger.print({"eval/score": r["score_mean"]}, step)
            t0 = time.time()
        if cfg.train.checkpoint_every and step % cfg.train.checkpoint_every == 0:
            save_ckpt(step)
    save_ckpt(cfg.train.max_steps)

    summ = ds.summary()
    results = {"method": method, "algo": algo, "dataset": dataset, "seed": seed,
               "max_return_quantile": data_cfg.max_return_quantile,
               "dataset_max_score": summ["norm_score_max"], "dataset_mean_score": summ["norm_score_mean"]}
    if cfg.train.train_policy:
        r = evaluate(cfg.eval.final_episodes, 10_000 + seed)
        results.update(score=r["score_mean"], score_episode_std=r["score_std"], return_mean=r["return_mean"])
        logger.log({"final/score": r["score_mean"]}, cfg.train.max_steps)
        print(f"[{algo}] score normalisé final = {r['score_mean']:.1f} ± {r['score_std']:.1f}")
    # results.json est écrit en dernier : sa présence marque un run terminé (reprise des expériences)
    with open(out / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.close()
    return results


def main(default_algo: str | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    if default_algo is None:
        p.add_argument("--algo", required=True, choices=["iql", "cql"])
    p.add_argument("--dataset", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--method", default=None, help="nom du dossier de sortie (défaut : nom de l'algo)")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--set", nargs="*", default=[], metavar="CLÉ=VALEUR", help="surcharges de config/<algo>.yaml")
    p.add_argument("--data-set", nargs="*", default=[], metavar="CLÉ=VALEUR", help="surcharges de config/data.yaml")
    a = p.parse_args()
    algo = default_algo or a.algo
    train(algo, a.dataset, a.seed, a.set, a.method, a.output_dir, a.data_set)


if __name__ == "__main__":
    main()
