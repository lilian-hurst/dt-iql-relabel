"""Lance une expérience complète puis agrège les résultats.

    # aperçu des commandes, sans rien exécuter
    python experiments/run_experiments.py experiments/configs/main.yaml --dry-run
    # comparaison principale, 4 jobs en parallèle répartis sur 2 GPU
    python experiments/run_experiments.py experiments/configs/main.yaml --jobs 4 --gpus 0 1
    # sous-ensemble : une méthode, un dataset, deux seeds
    python experiments/run_experiments.py experiments/configs/main.yaml --methods dt \
        --datasets hopper-medium-v2 --seeds 0 1
    # chaîne complète en quelques minutes
    python experiments/run_experiments.py experiments/configs/smoke.yaml

Les jobs terminés sont sautés (reprise après interruption) ; --force les relance.
Logs : runs/<name>/logs/ ; tableaux : results/<name>/ (aggregate.py) ; figures : figures/<name>/ (plotting.py).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aggregate import aggregate  # noqa: E402
from launcher import describe_jobs, estimate, load_experiment, plan_jobs, run_jobs, runs_dir_of  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", help="experiments/configs/<nom>.yaml")
    p.add_argument("--methods", nargs="+", default=None)
    p.add_argument("--datasets", nargs="+", default=None, help="noms de datasets ou de suites (mujoco, kitchen)")
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--name", default=None, help="remplace le nom (dossier runs/<name>)")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--jobs", type=int, default=1, help="jobs en parallèle")
    p.add_argument("--gpus", nargs="+", default=None, help="ids GPU attribués aux jobs à tour de rôle")
    p.add_argument("--force", action="store_true", help="relance aussi les jobs terminés")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    exp = load_experiment(a.config, methods=a.methods, datasets=a.datasets, seeds=a.seeds,
                          name=a.name, output_dir=a.output_dir)
    runs = runs_dir_of(exp)
    jobs = plan_jobs(exp)
    print(f"[experiment] {exp.name} : {len(jobs)} jobs ({describe_jobs(jobs)}) -> {runs}")
    pending = [j for j in jobs if a.force or not j.done_file.exists()]
    print(f"[experiment] {len(pending)} à exécuter : {estimate(pending, a.jobs)}")
    status = run_jobs(jobs, runs, workers=a.jobs, gpus=a.gpus, force=a.force, dry_run=a.dry_run)
    if not a.dry_run:
        aggregate(runs, methods=list(exp.methods))
        try:
            from plotting import plot_experiment
        except ImportError:
            print("[plots] matplotlib absent : figures non générées (pip install matplotlib)")
        else:
            plot_experiment(runs, methods=list(exp.methods))
    return 1 if "failed" in status.values() else 0


if __name__ == "__main__":
    sys.exit(main())
