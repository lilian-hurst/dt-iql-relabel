"""Entraînement d'un Decision Transformer (vanilla, ou sur RTG relabellisés).

Exemples :
    python -m src.training.train_dt --dataset hopper-medium-replay-v2 --seed 0
    python -m src.training.train_dt --dataset hopper-medium-v2 --set model.context_len=5 train.max_steps=20000
    # étape 3 : RTG relabellisés produits par relabel_returns.py
    python -m src.training.train_dt --dataset hopper-medium-replay-v2 --rtg-file runs/relabel/.../rtg.npz --method iql_dt

Sorties dans runs/<method>/<dataset>/seed<k>/ :
    config.yaml, metrics.jsonl, model.pt, results.json (score normalisé final par return cible).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.data.d4rl_loader import load_dataset
from src.data.registry import get_spec
from src.data.sequence_dataset import SequenceDataset
from src.evaluation.rollout_dt import evaluate_dt
from src.models.decision_transformer import build_dt
from src.utils.config import Config, load_config, save_config
from src.utils.logger import Logger
from src.utils.seed import get_device, set_seed


def load_rtg_file(path: str | Path, n_trajs: int) -> list[np.ndarray]:
    """Fichier .npz produit par la relabellisation : clés 'rtg' (concaténés) et 'lengths'."""
    data = np.load(path)
    lengths = data["lengths"]
    if len(lengths) != n_trajs:
        raise ValueError(f"{path} : {len(lengths)} trajectoires, dataset : {n_trajs} "
                         "(même dataset et même filtrage requis)")
    return np.split(data["rtg"], np.cumsum(lengths)[:-1])


def dt_loss(model, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """MSE sur les actions, uniquement sur les pas non-paddés."""
    pred = model(batch["states"], batch["actions"], batch["rtg"], batch["timesteps"], batch["mask"])
    mask = batch["mask"].unsqueeze(-1)
    return (((pred - batch["actions"]) ** 2) * mask).sum() / (mask.sum() * pred.shape[-1])


def evaluate_all_targets(model, spec, cfg: Config, seq: SequenceDataset, episodes: int, seed: int, device):
    results = []
    for target in cfg.target_returns[spec.domain]:
        results.append(evaluate_dt(
            model, spec, target, seq.return_scale, seq.state_mean, seq.state_std,
            episodes=episodes, seed=seed, device=device,
        ))
    return results


def train(
    dataset: str,
    seed: int = 0,
    overrides: list[str] = (),
    method: str = "dt",
    rtg_file: str | None = None,
    output_dir: str | None = None,
    data_overrides: list[str] = (),
) -> dict:
    spec = get_spec(dataset)
    cfg = load_config("dt.yaml", overrides, domain=spec.domain)
    data_cfg = load_config("data.yaml", data_overrides)
    exp_cfg = load_config("experiment.yaml")
    set_seed(seed, exp_cfg.deterministic)
    device = get_device(exp_cfg.device)
    rng = np.random.default_rng(seed)

    out = Path(output_dir or exp_cfg.output_dir) / method / dataset / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    save_config(Config({"dt": cfg, "data": data_cfg, "dataset": dataset, "seed": seed,
                        "method": method, "rtg_file": rtg_file}), out / "config.yaml")

    # --- données ---------------------------------------------------------------------
    ds = load_dataset(dataset, root=data_cfg.root, max_return_quantile=data_cfg.max_return_quantile)
    rtgs = load_rtg_file(rtg_file, len(ds.trajectories)) if rtg_file else None
    seq = SequenceDataset(
        ds, cfg.model.context_len, cfg.return_scale[spec.domain], cfg.model.max_timestep,
        rtgs=rtgs, normalize_states=data_cfg.normalize_states,
    )
    print(f"[dt] {dataset} : {len(ds.trajectories)} trajectoires, {len(seq)} pas | device={device} | sortie={out}")

    # --- modèle / optimisation ------------------------------------------------------
    model = build_dt(cfg.model, seq.obs_dim, seq.act_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    warmup = max(1, int(cfg.train.warmup_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warmup, 1.0))
    logger = Logger(out, exp_cfg.logger.backend, exp_cfg.logger.project,
                    f"{method}-{dataset}-s{seed}", cfg.to_dict())
    print(f"[dt] {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M paramètres")

    def save_ckpt(step: int) -> None:
        torch.save({
            "model": model.state_dict(), "model_cfg": cfg.model.to_dict(), "step": step,
            "obs_dim": seq.obs_dim, "act_dim": seq.act_dim, "dataset": dataset,
            "state_mean": seq.state_mean, "state_std": seq.state_std, "return_scale": seq.return_scale,
        }, out / "model.pt")

    # --- boucle -------------------------------------------------------------------
    model.train()
    t0, losses = time.time(), []
    for step in range(1, cfg.train.max_steps + 1):
        batch = {k: torch.as_tensor(v, device=device) for k, v in seq.sample(cfg.train.batch_size, rng).items()}
        loss = dt_loss(model, batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        opt.step()
        sched.step()
        losses.append(loss.item())

        if step % cfg.train.log_every == 0:
            m = {"train/action_mse": float(np.mean(losses)), "train/lr": sched.get_last_lr()[0],
                 "train/steps_per_s": cfg.train.log_every / (time.time() - t0)}
            logger.log(m, step)
            logger.print(m, step)
            losses, t0 = [], time.time()

        if cfg.train.eval_every and step % cfg.train.eval_every == 0 and step < cfg.train.max_steps:
            for r in evaluate_all_targets(model, spec, cfg, seq, cfg.eval.episodes, seed, device):
                m = {f"eval/score_target{r['target_return']:g}": r["score_mean"]}
                logger.log(m, step)
                logger.print(m, step)
            t0 = time.time()

        if cfg.train.checkpoint_every and step % cfg.train.checkpoint_every == 0:
            save_ckpt(step)

    save_ckpt(cfg.train.max_steps)

    # --- évaluation finale (seed d'évaluation distincte des évaluations intermédiaires) ----
    final = evaluate_all_targets(model, spec, cfg, seq, cfg.eval.final_episodes, 10_000 + seed, device)
    best = max(final, key=lambda r: r["score_mean"])
    for r in final:
        logger.log({f"final/score_target{r['target_return']:g}": r["score_mean"]}, cfg.train.max_steps)
    summ = ds.summary()
    results = {
        "method": method, "dataset": dataset, "seed": seed,
        "context_len": cfg.model.context_len, "max_return_quantile": data_cfg.max_return_quantile,
        "score": best["score_mean"], "score_episode_std": best["score_std"],
        "best_target_return": best["target_return"], "per_target": final,
        # score normalisé de la meilleure / moyenne trajectoire des données : le dépasser = stitching
        "dataset_max_score": summ["norm_score_max"], "dataset_mean_score": summ["norm_score_mean"],
    }
    with open(out / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.close()
    print(f"[dt] score normalisé final = {best['score_mean']:.1f} ± {best['score_std']:.1f} "
          f"(cible {best['target_return']:g}) -> {out / 'results.json'}")
    return results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--method", default="dt", help="nom du dossier de sortie (dt, iql_dt, qdt, ...)")
    p.add_argument("--rtg-file", default=None, help="RTG relabellisés (.npz) ; défaut : RTG Monte-Carlo")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--set", nargs="*", default=[], metavar="CLÉ=VALEUR", help="surcharges de config/dt.yaml")
    p.add_argument("--data-set", nargs="*", default=[], metavar="CLÉ=VALEUR", help="surcharges de config/data.yaml")
    a = p.parse_args()
    train(a.dataset, a.seed, a.set, a.method, a.rtg_file, a.output_dir, a.data_set)


if __name__ == "__main__":
    main()
