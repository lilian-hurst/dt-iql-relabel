"""Planification et exécution des jobs d'une expérience (utilisé par run_experiments.py / run_ablations.py).

Chaque job est un sous-processus `python -m src.training.<...>` dont le fichier de fin
(results.json ou rtg.npz, écrit en dernier) sert de marqueur : un job terminé est sauté,
ce qui permet de relancer une expérience interrompue. Les jobs sont groupés en phases :
  0 : DT vanilla, baselines IQL/CQL, critiques de relabellisation
  1 : relabellisation des RTG (dépend du critique)
  2 : DT sur RTG relabellisés (dépend de la relabellisation)
Ce module n'importe pas PyTorch (planification et --dry-run possibles partout).
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.data.relabel import critic_checkpoint, relabel_output_dir  # noqa: E402
from src.utils.config import Config, load_config, load_yaml  # noqa: E402

EXPERIMENT_DEFAULTS = {
    "name": "default", "datasets": "mujoco", "methods": ["dt"], "seeds": [0], "output_dir": None,
    "dt_overrides": [], "iql_overrides": [], "cql_overrides": [], "critic_overrides": [],
    "relabel_overrides": [], "data_overrides": [],
}


@dataclass
class Job:
    name: str
    cmd: list[str]
    done_file: Path
    phase: int
    deps: list[Path] = field(default_factory=list)


def load_experiment(path: str | Path | None = None, **updates) -> Config:
    cfg = Config(EXPERIMENT_DEFAULTS)
    if path:
        for k, v in load_yaml(Path(path).resolve()).items():
            if k not in EXPERIMENT_DEFAULTS:
                raise KeyError(f"{path} : clé inconnue {k!r} ; clés valides : {list(EXPERIMENT_DEFAULTS)}")
            cfg[k] = v
    for k, v in updates.items():
        if v is not None:
            cfg[k] = v
    return cfg


def resolve_datasets(spec) -> list[str]:
    if isinstance(spec, str):
        suites = load_yaml("data.yaml").suites
        return list(suites[spec]) if spec in suites else [spec]
    out: list[str] = []
    for s in spec:
        out += resolve_datasets(s)
    return out


def runs_dir_of(exp: Config) -> Path:
    base = Path(exp.output_dir or load_yaml("experiment.yaml").output_dir)
    return (base if base.is_absolute() else REPO / base) / exp.name


def _set(flag: str, values) -> list[str]:
    values = [str(v) for v in values]
    return [flag, *values] if values else []


def plan_jobs(exp: Config, python: str = sys.executable, suffix: str = "") -> list[Job]:
    """Jobs de l'expérience. `suffix` est ajouté au nom de sortie des méthodes (pas des critiques ni
    des relabellisations, ainsi partagés entre les valeurs d'une ablation : ex. 'iql_dt__K5')."""
    methods_cfg = load_yaml("methods.yaml")
    runs = runs_dir_of(exp)
    jobs: dict[str, Job] = {}

    def add(job: Job) -> None:
        jobs.setdefault(str(job.done_file), job)  # critiques partagés entre méthodes : dédupliqués

    for ds in resolve_datasets(exp.datasets):
        for seed in exp.seeds:
            common = ["--dataset", ds, "--seed", str(seed), "--output-dir", str(runs),
                      *_set("--data-set", exp.data_overrides)]
            for m in exp.methods:
                if m not in methods_cfg:
                    raise KeyError(f"méthode inconnue {m!r} ; voir config/methods.yaml : {list(methods_cfg)}")
                mc = methods_cfg[m]
                name = f"{m}{suffix}"
                out = runs / name / ds / f"seed{seed}" / "results.json"

                if mc.train in ("iql", "cql"):
                    add(Job(f"{name}/{ds}/s{seed}", [python, "-m", "src.training.train_offline", "--algo", mc.train,
                                                  "--method", name, *common,
                                                  *_set("--set", exp[f"{mc.train}_overrides"])], out, 0))
                elif mc.train == "dt" and "relabel" not in mc:
                    add(Job(f"{name}/{ds}/s{seed}", [python, "-m", "src.training.train_dt", "--method", name, *common,
                                                  *_set("--set", exp.dt_overrides)], out, 0))
                elif mc.train == "dt":
                    r_over = [f"{k}={v}" for k, v in mc.relabel.items()] + list(exp.relabel_overrides)
                    rcfg = load_config("relabel.yaml", r_over)
                    ckpt = critic_checkpoint(rcfg, ds, seed, runs)
                    if rcfg.checkpoint:  # critique fourni : pas de job d'entraînement
                        critic_done = ckpt
                    else:
                        critic = rcfg.critics[rcfg.value_source]
                        critic_done = ckpt.parent / "results.json"
                        add(Job(f"{critic.method}/{ds}/s{seed}",
                                [python, "-m", "src.training.train_offline", "--algo", rcfg.value_source,
                                 "--method", critic.method, *common,
                                 *_set("--set", list(critic.overrides) + list(exp.critic_overrides))],
                                critic_done, 0))
                    rtg = relabel_output_dir(rcfg, ds, seed, runs) / "rtg.npz"
                    add(Job(f"relabel/{rtg.parent.parent.parent.name}/{ds}/s{seed}",
                            [python, "-m", "src.training.relabel_returns", *common, *_set("--set", r_over)],
                            rtg, 1, [critic_done]))
                    add(Job(f"{name}/{ds}/s{seed}",
                            [python, "-m", "src.training.train_dt", "--method", name, "--rtg-file", str(rtg), *common,
                             *_set("--set", exp.dt_overrides)], out, 2, [rtg]))
                else:
                    raise ValueError(f"méthode {m!r} : train={mc.train!r} non pris en charge")
    return sorted(jobs.values(), key=lambda j: j.phase)


def describe_jobs(jobs: list[Job]) -> str:
    """Décompte par type, pour estimer le coût avant de lancer."""
    kinds = {"src.training.train_dt": "entraînements DT", "src.training.train_offline": "entraînements IQL/CQL",
             "src.training.relabel_returns": "relabellisations"}
    counts: dict[str, int] = {}
    for j in jobs:
        k = kinds.get(j.cmd[2], j.cmd[2])
        counts[k] = counts.get(k, 0) + 1
    return ", ".join(f"{n} {k}" for k, n in counts.items())


def run_jobs(jobs: list[Job], runs_dir: Path, workers: int = 1, gpus: list[str] | None = None,
             force: bool = False, dry_run: bool = False) -> dict[str, str]:
    """Exécute les jobs phase par phase ; renvoie {nom du job: statut}."""
    seen: dict[str, int] = {}
    for j in jobs:
        seen[j.name] = seen.get(j.name, 0) + 1
    dup = sorted(n for n, c in seen.items() if c > 1)
    if dup:  # les logs et statuts sont indexés par nom
        raise ValueError(f"noms de jobs en double : {dup[:5]}")
    log_dir = runs_dir / "logs"
    status: dict[str, str] = {}
    by_file: dict[str, str] = {}
    gpu_q: queue.Queue = queue.Queue()
    for i in range(max(workers, 1)):
        gpu_q.put(gpus[i % len(gpus)] if gpus else None)
    total, counter = len(jobs), [0]

    def run_one(job: Job) -> str:
        gpu = gpu_q.get()
        env = dict(os.environ)
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        log = log_dir / (job.name.replace("/", "__") + ".log")
        log.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        try:
            with open(log, "w") as f:
                f.write(" ".join(job.cmd) + "\n\n")
                f.flush()
                rc = subprocess.run(job.cmd, cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
        finally:
            gpu_q.put(gpu)
        ok = rc == 0 and job.done_file.exists()
        counter[0] += 1
        print(f"[{counter[0]:>3}/{total}] {'OK   ' if ok else 'ÉCHEC'} {job.name} "
              f"({(time.time() - t0) / 60:.1f} min){'' if ok else f' -> voir {log}'}", flush=True)
        return "done" if ok else "failed"

    for phase in sorted({j.phase for j in jobs}):
        todo = []
        for j in (j for j in jobs if j.phase == phase):
            if j.done_file.exists() and not force:
                status[j.name] = by_file[str(j.done_file)] = "cached"
                counter[0] += 1
            elif any(by_file.get(str(d)) == "failed" or (not d.exists() and not dry_run
                                                          and by_file.get(str(d)) is None) for d in j.deps):
                status[j.name] = by_file[str(j.done_file)] = "failed"
                counter[0] += 1
                print(f"[{counter[0]:>3}/{total}] SAUTÉ {j.name} (dépendance en échec ou absente)")
            else:
                todo.append(j)
        if dry_run:
            for j in todo:
                print(f"[phase {phase}] {' '.join(j.cmd[2:])}")
                status[j.name] = by_file[str(j.done_file)] = "planned"
            continue
        with ThreadPoolExecutor(max_workers=max(workers, 1)) as ex:
            for j, st in zip(todo, ex.map(run_one, todo)):
                status[j.name] = by_file[str(j.done_file)] = st

    if not dry_run:
        runs_dir.mkdir(parents=True, exist_ok=True)
        with open(runs_dir / "jobs.json", "w") as f:
            json.dump(status, f, indent=2)
    counts = {s: list(status.values()).count(s) for s in sorted(set(status.values()))}
    print(f"[launcher] {len(jobs)} jobs : " + ", ".join(f"{n} {s}" for s, n in counts.items()))
    return status
