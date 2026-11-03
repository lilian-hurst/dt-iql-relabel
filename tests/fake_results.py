"""Résultats SYNTHÉTIQUES (valeurs inventées) pour tester tableaux et figures sans entraînement.

Ne jamais utiliser ces nombres comme résultats : ils ne servent qu'à exercer le code de rendu.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

BASE = {"halfcheetah": 40.0, "hopper": 70.0, "walker2d": 65.0}
BONUS = {"dt": 0.0, "iql": 6.0, "cql": 4.0, "qdt": 2.0, "iql_dt": 5.0, "iql_qdt": 3.0}
DT_FAMILY = {"dt", "qdt", "iql_dt", "iql_qdt"}


def _score(method: str, dataset: str, rng, shift: float = 0.0) -> float:
    return BASE.get(dataset.split("-")[0], 50.0) + BONUS.get(method.split("__")[0], 0.0) + shift + rng.normal(0, 3)


def write_run(run_dir: Path, method: str, dataset: str, seed: int, score: float, dt_like: bool,
              steps: int = 100_000, every: int = 10_000, rng=None, dataset_max: float = 95.0) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    res = {"method": method, "dataset": dataset, "seed": seed, "score": score, "score_episode_std": 5.0,
           "dataset_max_score": dataset_max, "dataset_mean_score": 30.0}
    if dt_like:
        res.update(best_target_return=3600.0, per_target=[])
        rng = rng or np.random.default_rng(seed)
        with open(run_dir / "metrics.jsonl", "w") as f:
            for step in range(every, steps, every):
                v = score * (1 - np.exp(-3 * step / steps)) + rng.normal(0, 4)
                f.write(json.dumps({"step": step, "eval/score_target3600": v}) + "\n")
            f.write(json.dumps({"step": steps, "final/score_target3600": score}) + "\n")
    (run_dir / "results.json").write_text(json.dumps(res))


def make_experiment(runs: Path, methods, datasets, seeds, rng=None) -> None:
    rng = rng or np.random.default_rng(0)
    for m in methods:
        for d in datasets:
            for s in seeds:
                write_run(runs / m / d / f"seed{s}", m, d, s, _score(m, d, rng), m in DT_FAMILY, rng=rng)


def make_relabel_npz(path: Path, n_traj: int = 40, rng=None) -> Path:
    rng = rng or np.random.default_rng(0)
    lengths = rng.integers(50, 300, size=n_traj)
    rewards = [rng.uniform(0, 3, size=L) for L in lengths]
    rtg_o = [np.cumsum(r[::-1])[::-1] for r in rewards]
    gamma = 0.995
    g = [np.array([np.sum(r[t:] * gamma ** np.arange(len(r) - t)) for t in range(len(r))]) for r in rewards]
    q = [x + rng.normal(5, 15, size=len(x)) for x in g]
    rtg_n = [o + np.maximum(0, qq - gg) for o, qq, gg in zip(rtg_o, q, g)]
    path.parent.mkdir(parents=True, exist_ok=True)
    cat = lambda xs: np.concatenate(xs).astype(np.float32)
    np.savez_compressed(path, rtg=cat(rtg_n), rtg_original=cat(rtg_o), lengths=lengths, q=cat(q),
                        v_next=cat(q), g_mc=cat(g), gamma=np.float32(gamma))
    return path


def fill_ablation(root: Path, manifest: dict, rng=None) -> None:
    """Remplit les dossiers d'un manifeste d'ablation avec des scores dépendant de la valeur."""
    rng = rng or np.random.default_rng(1)
    for i, e in enumerate(manifest["entries"]):
        k = manifest["value_keys"].index(e["value_key"])
        for d in manifest["datasets"]:
            for s in manifest["seeds"]:
                run = root / e["dir"] / d / f"seed{s}"
                if not (run / "results.json").exists():
                    dmax = 60.0 + 12.0 * k
                    write_run(run, e["method"], d, s, _score(e["method"], d, rng, shift=4.0 * k),
                              e["method"] in DT_FAMILY, rng=rng, dataset_max=dmax)
    for e in manifest["references"]:
        for d in manifest["datasets"]:
            for s in manifest["seeds"]:
                run = root / e["dir"] / d / f"seed{s}"
                if not (run / "results.json").exists():
                    write_run(run, e["method"], d, s, _score(e["method"], d, rng), e["method"] in DT_FAMILY, rng=rng)
