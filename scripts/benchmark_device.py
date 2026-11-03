"""Mesure la vitesse d'entraînement de DT, IQL et CQL sur chaque device disponible (cpu, mps, cuda),
recommande le plus rapide par algorithme et estime la durée d'un run.

    python scripts/benchmark_device.py                    # mesure et recommandation (~1-3 min)
    python scripts/benchmark_device.py --write            # enregistre dans config/experiment.local.yaml
    python scripts/benchmark_device.py --threads 3        # comme un job parmi 4 en parallèle (12 cœurs)

Pourquoi : les réseaux du projet sont petits (MLP 256x2, Transformer 128x3). Sur un Mac Apple Silicon,
le coût fixe de chaque opération sur le GPU (MPS) peut dépasser le gain de calcul : le CPU est parfois
plus rapide, en particulier pour IQL. Seule une mesure sur la machine permet de trancher.

Données synthétiques aux dimensions de hopper (obs 11, action 3), mêmes tailles de réseaux et de batchs
que config/{dt,iql,cql}.yaml, même boucle que l'entraînement (tirage des batchs compris). Les
évaluations en environnement ne sont pas comptées.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src  # noqa: E402,F401  (PYTORCH_ENABLE_MPS_FALLBACK avant l'import de torch)
from src.utils.config import CONFIG_DIR, Config, deep_merge, load_config, load_yaml, save_config  # noqa: E402

OBS, ACT, N_TRAJ, T_LEN = 11, 3, 60, 1000


def _devices(torch) -> list[str]:
    out = ["cpu"]
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        out.append("mps")
    if torch.cuda.is_available():
        out.append("cuda")
    return out


def _timed(step, device, steps: int, warmup: int) -> float:
    from src.utils.seed import synchronize

    for _ in range(warmup):
        step()
    synchronize(device)
    t0 = time.perf_counter()
    for _ in range(steps):
        step()
    synchronize(device)
    return steps / (time.perf_counter() - t0)


def _synthetic_dataset():
    from src.data.d4rl_loader import OfflineDataset, Trajectory
    from src.data.registry import get_spec

    rng = np.random.default_rng(0)
    trajs = []
    for _ in range(N_TRAJ):
        obs = rng.normal(size=(T_LEN + 1, OBS)).astype(np.float32)
        trajs.append(Trajectory(observations=obs[:-1], next_observations=obs[1:],
                                actions=rng.uniform(-1, 1, (T_LEN, ACT)).astype(np.float32),
                                rewards=rng.uniform(0, 3, T_LEN).astype(np.float32),
                                terminals=np.zeros(T_LEN, dtype=bool)))
    return OfflineDataset(get_spec("hopper-medium-replay-v2"), trajs)


def bench_dt(torch, device, steps: int, warmup: int) -> float:
    from src.data.sequence_dataset import SequenceDataset
    from src.models.decision_transformer import build_dt
    from src.training.train_dt import dt_loss

    cfg = load_config("dt.yaml")
    seq = SequenceDataset(_synthetic_dataset(), cfg.model.context_len, 1000.0, cfg.model.max_timestep)
    model = build_dt(cfg.model, OBS, ACT).to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    rng = np.random.default_rng(0)
    loss_sum = torch.zeros((), device=device)

    def step():
        nonlocal loss_sum
        batch = {k: torch.as_tensor(v, device=device) for k, v in seq.sample(cfg.train.batch_size, rng).items()}
        loss = dt_loss(model, batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        opt.step()
        loss_sum += loss.detach()

    return _timed(step, device, steps, warmup)


def bench_offline(torch, algo: str, device, steps: int, warmup: int) -> float:
    from src.training.train_offline import _algos, make_buffer

    cfg_file, cls = _algos()[algo]
    cfg = load_config(cfg_file)
    ds = _synthetic_dataset()
    buf = make_buffer(ds, 1.0, cfg.model.time_feature, device)
    agent = cls(OBS, ACT, cfg.model, cfg.train, 1000).to(device).train()
    n = buf["rewards"].shape[0]
    gen_device = device if device.type == "cuda" else torch.device("cpu")
    gen = torch.Generator(device=gen_device).manual_seed(0)
    acc = {}

    def step():
        idx = torch.randint(0, n, (cfg.train.batch_size,), device=gen_device, generator=gen).to(device)
        for k, v in agent.update({k: v[idx] for k, v in buf.items()}).items():
            acc[k] = acc[k] + v if k in acc else v.clone()

    return _timed(step, device, steps, warmup)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--algos", nargs="+", default=["dt", "iql", "cql"], choices=["dt", "iql", "cql"])
    p.add_argument("--devices", nargs="+", default=None, help="défaut : tous ceux disponibles")
    p.add_argument("--steps", type=int, default=None, help="pas mesurés (défaut : dt 150, iql 600, cql 200)")
    p.add_argument("--threads", type=int, default=0, help="threads CPU (simule un job parmi plusieurs)")
    p.add_argument("--write", action="store_true", help="écrit devices et débits dans config/experiment.local.yaml")
    a = p.parse_args()

    try:
        import torch
    except ImportError:
        print("PyTorch n'est pas installé : pip install -e \".[dev]\"")
        return 1
    if a.threads:
        torch.set_num_threads(a.threads)
    devices = a.devices or _devices(torch)
    default_steps = {"dt": 150, "iql": 600, "cql": 200}
    max_steps = {"dt": load_config("dt.yaml").train.max_steps, "iql": load_config("iql.yaml").train.max_steps,
                 "cql": load_config("cql.yaml").train.max_steps}
    print(f"PyTorch {torch.__version__} | devices : {', '.join(devices)} | threads CPU : {torch.get_num_threads()}\n")

    results: dict[str, dict[str, float]] = {}
    for algo in a.algos:
        steps = a.steps or default_steps[algo]
        for dev in devices:
            device = torch.device(dev)
            torch.manual_seed(0)
            try:
                sps = bench_dt(torch, device, steps, 20) if algo == "dt" else \
                    bench_offline(torch, algo, device, steps, 20)
            except Exception as e:  # noqa: BLE001  (opération non supportée sur un backend, etc.)
                print(f"  {algo:4s} {dev:5s} ÉCHEC : {type(e).__name__}: {e}")
                continue
            results.setdefault(algo, {})[dev] = sps
            hours = max_steps[algo] / sps / 3600
            print(f"  {algo:4s} {dev:5s} {sps:8.1f} pas/s   -> {max_steps[algo]:,} pas ≈ {hours:5.2f} h".replace(",", " "))

    best = {algo: max(r, key=r.get) for algo, r in results.items() if r}
    print("\nRecommandation (device le plus rapide) : " + ", ".join(f"{k}: {v}" for k, v in best.items()))
    print("Durées hors évaluations en environnement ; plusieurs jobs en parallèle (--jobs) se partagent la machine.")

    if a.write and best:
        path = CONFIG_DIR / "experiment.local.yaml"
        local = load_yaml(path) if path.exists() else Config()
        update = {"devices": dict(best), "throughput": {k: round(results[k][v], 1) for k, v in best.items()}}
        if a.threads:
            update["threads"] = a.threads
        save_config(deep_merge(local, update), path)
        print(f"-> écrit dans {path} (non versionné)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
