"""Génère les figures du rapport (PDF vectoriel + PNG 300 dpi) dans figures/.

    # expérience : comparaison (points ± écart-type, score DT publié en repère), courbes
    # d'apprentissage (moyenne + bande ± écart-type) et diagnostics de relabellisation
    python experiments/plot_results.py experiment runs/main
    python experiments/plot_results.py experiment runs/stitching_q0.5 --methods dt iql_dt qdt
    # ablations déjà lancées (tableaux + figures)
    python experiments/plot_results.py ablations
    python experiments/plot_results.py ablations --only context_len
    # diagnostic d'une relabellisation
    python experiments/plot_results.py relabel runs/main/relabel/iql_max_q_a1/hopper-medium-replay-v2/seed0/rtg.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plotting import plot_experiment, plot_relabel_diagnostics  # noqa: E402
from run_ablations import ablations_root, report  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("experiment", help="figures d'une expérience (runs/<nom>)")
    e.add_argument("runs_dir")
    e.add_argument("--methods", nargs="+", default=None)
    e.add_argument("--out", default=None)
    e.add_argument("--no-relabel", action="store_true")
    ab = sub.add_parser("ablations", help="tableaux et figures des ablations")
    ab.add_argument("--only", nargs="+", default=None)
    ab.add_argument("--output-dir", default=None, help="racine des runs (défaut : runs)")
    ab.add_argument("--name", default="ablations", help="clé `name` du fichier d'ablations")
    r = sub.add_parser("relabel", help="diagnostic d'un ou plusieurs rtg.npz")
    r.add_argument("npz", nargs="+")
    r.add_argument("--out", default="figures/relabel")
    a = p.parse_args()

    if a.cmd == "experiment":
        plot_experiment(a.runs_dir, a.out, a.methods, relabel=not a.no_relabel)
    elif a.cmd == "ablations":
        manifests = sorted((ablations_root(a.output_dir, a.name) / "manifests").glob("*.json"))
        if a.only:
            manifests = [m for m in manifests if m.stem in a.only]
        if not manifests:
            print("aucun manifeste : lancer d'abord experiments/run_ablations.py")
        for m in manifests:
            report(m)
    else:
        for npz in map(Path, a.npz):
            name = "__".join(npz.parts[-4:-1]) if len(npz.parts) >= 4 else npz.stem
            paths = plot_relabel_diagnostics(npz, Path(a.out) / name, title=" — ".join(npz.parts[-4:-1]))
            print(*paths, sep="\n")


if __name__ == "__main__":
    main()
