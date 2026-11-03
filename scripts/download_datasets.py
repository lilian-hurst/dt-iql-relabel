"""Télécharge et vérifie les datasets réels du projet (D4RL v2 Gym-MuJoCo, Franka Kitchen via Minari).

    python scripts/download_datasets.py                                 # suite mujoco (9 datasets)
    python scripts/download_datasets.py --suite mujoco kitchen
    python scripts/download_datasets.py --datasets hopper-medium-replay-v2 walker2d-medium-replay-v2
    python scripts/download_datasets.py --sources hf_mirror             # forcer le miroir Hugging Face

Gym-MuJoCo : fichiers HDF5 officiels D4RL v2, essayés dans l'ordre de `download_sources`
(config/data.yaml) : serveur du RAIL (Berkeley), puis miroir Hugging Face imone/D4RL. Chaque fichier
est vérifié (lecture HDF5, clés et longueurs) avant d'être conservé dans datasets/ (ou $D4RL_DATASET_DIR).
Kitchen : Minari D4RL/kitchen/*-v2 (cache Minari, ~/.minari/datasets), requiert `pip install -e ".[kitchen]"`.

Le résumé affiché (trajectoires, transitions, score normalisé moyen et maximal des trajectoires) permet
de repérer un fichier anormal. Ensuite, `python scripts/check_env.py --dataset <nom>` vérifie que
l'environnement de simulation reproduit bien les transitions du dataset.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.d4rl_loader import dataset_root, fetch_hdf5, load_dataset  # noqa: E402
from src.data.registry import get_spec  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def main() -> int:
    data_cfg = load_config("data.yaml")
    suites = data_cfg.suites
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suite", nargs="+", default=None, choices=list(suites))
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--root", default=None, help="dossier des HDF5 (défaut : config/data.yaml ou $D4RL_DATASET_DIR)")
    p.add_argument("--sources", nargs="+", default=None, help="ordre des sources, ex. hf_mirror official")
    a = p.parse_args()

    names = list(a.datasets or [])
    for s in a.suite or ([] if a.datasets else ["mujoco"]):
        names += [d for d in suites[s] if d not in names]
    root = a.root or data_cfg.root
    sources = a.sources or list(data_cfg.download_sources)
    print(f"[download] {len(names)} datasets -> {dataset_root(root)} (sources : {', '.join(sources)})\n")

    rows, failed = [], 0
    for name in names:
        t0 = time.time()
        try:
            spec = get_spec(name)
            size = ""
            if spec.backend == "hdf5":
                path = fetch_hdf5(spec, root, sources)
                size = f"{path.stat().st_size / 1e6:.0f} Mo"
            s = load_dataset(name, root=root, sources=sources).summary()
            rows.append((name, "OK", f"{s['trajectories']}", f"{s['transitions']}",
                         f"{s['norm_score_mean']:.1f}", f"{s['norm_score_max']:.1f}", size,
                         f"{time.time() - t0:.0f}s"))
        except Exception as e:  # noqa: BLE001
            failed += 1
            rows.append((name, "ÉCHEC", "", "", "", "", "", ""))
            print(f"[download] {name} : {type(e).__name__}: {e}\n", file=sys.stderr)

    head = ("dataset", "statut", "trajectoires", "transitions", "score moyen", "score max", "taille", "durée")
    widths = [max(len(str(r[i])) for r in [head, *rows]) for i in range(len(head))]
    for r in [head, *rows]:
        print("  ".join(str(c).ljust(w) for c, w in zip(r, widths)))
    print("\nScores normalisés D4RL des trajectoires du dataset (moyenne / meilleure trajectoire).")
    if failed:
        print(f"{failed} échec(s) : voir les messages ci-dessus (réseau, miroir indisponible, extra kitchen absent).")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
