"""Vérifie l'installation : dépendances, environnement MuJoCo, chargement d'un dataset D4RL.

Usage :
    python scripts/check_env.py                                    # hopper-medium-replay-v2
    python scripts/check_env.py --dataset walker2d-medium-v2
    python scripts/check_env.py --dataset hopper-medium-v2 --file /chemin/hopper_medium-v2.hdf5
    python scripts/check_env.py --dataset kitchen-mixed-v2         # requiert requirements-kitchen.txt
    python scripts/check_env.py --skip-dataset                     # dépendances + env seulement

Code de sortie non nul si une vérification obligatoire échoue.
"""
from __future__ import annotations

import argparse
import importlib
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.d4rl_loader import load_dataset  # noqa: E402
from src.data.registry import get_spec  # noqa: E402
from src.envs.make_env import make_env  # noqa: E402
from src.evaluation.normalization import normalized_score  # noqa: E402

OK, WARN, FAIL = "\033[32mOK\033[0m", "\033[33mWARN\033[0m", "\033[31mFAIL\033[0m"


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def check_packages(kitchen: bool) -> bool:
    section("Dépendances")
    print(f"python {platform.python_version()} ({platform.platform()})")
    required = ["numpy", "yaml", "h5py", "gymnasium", "mujoco", "torch"]
    optional = ["pandas", "matplotlib", "tensorboard", "wandb"]
    if kitchen:
        required += ["minari", "gymnasium_robotics"]
    else:
        optional += ["minari", "gymnasium_robotics"]
    ok = True
    for name in required + optional:
        try:
            mod = importlib.import_module(name)
            print(f"  [{OK}] {name:20s} {getattr(mod, '__version__', '')}")
        except ImportError:
            is_req = name in required
            ok &= not is_req
            print(f"  [{FAIL if is_req else WARN}] {name:20s} absent{' (requis)' if is_req else ''}")
    try:
        import torch

        cuda = torch.cuda.is_available()
        dev = torch.cuda.get_device_name(0) if cuda else "CPU uniquement"
        print(f"  [{OK if cuda else WARN}] CUDA : {dev}")
    except ImportError:
        pass
    return ok


def check_env(name: str, steps: int = 200) -> tuple[bool, object]:
    section(f"Environnement ({get_spec(name).env_id})")
    try:
        env = make_env(name, seed=0)
    except Exception as e:  # noqa: BLE001
        print(f"  [{FAIL}] création impossible : {type(e).__name__}: {e}")
        return False, None
    obs, _ = env.reset(seed=0)
    total, t0 = 0.0, time.time()
    for _ in range(steps):
        obs, r, term, trunc, _ = env.step(env.action_space.sample())
        total += float(r)
        if term or trunc:
            obs, _ = env.reset()
    fps = steps / (time.time() - t0)
    print(f"  [{OK}] obs_dim={env.observation_space.shape[0]} act_dim={env.action_space.shape[0]} "
          f"action∈[{env.action_space.low.min():.0f},{env.action_space.high.max():.0f}] "
          f"| {steps} pas aléatoires, {fps:.0f} pas/s")
    return True, env


def replay_check(ds, env, n: int = 200) -> bool:
    """Remet l'env dans l'état (qpos, qvel) du dataset, joue l'action logguée, compare s'."""
    trajs = [t for t in ds.trajectories if "qpos" in t.infos and len(t) > 1]
    if not trajs or not hasattr(env.unwrapped, "set_state"):
        print(f"  [{WARN}] rejeu ignoré (pas de qpos/qvel dans le dataset)")
        return True
    rng = np.random.default_rng(0)
    errs = []
    for _ in range(n):
        tr = trajs[rng.integers(len(trajs))]
        t = int(rng.integers(len(tr)))
        env.reset(seed=0)
        env.unwrapped.set_state(tr.infos["qpos"][t], tr.infos["qvel"][t])
        nxt, *_ = env.step(tr.actions[t])
        errs.append(np.abs(nxt - tr.next_observations[t]).max())
    errs = np.array(errs)
    good = np.median(errs) < 1e-2
    print(f"  [{OK if good else WARN}] rejeu de {n} transitions : |s'_env - s'_dataset|∞ "
          f"médiane={np.median(errs):.2e}, max={errs.max():.2e}")
    if not good:
        print("         -> dynamique différente de celle du dataset (version MuJoCo/env ?)")
    return True


def check_dataset(name: str, file: str | None, root: str | None, env) -> bool:
    section(f"Dataset {name}")
    t0 = time.time()
    try:
        ds = load_dataset(name, root=root, path=file, keep_infos=True)
    except Exception as e:  # noqa: BLE001
        print(f"  [{FAIL}] chargement impossible : {type(e).__name__}: {e}")
        if get_spec(name).backend == "hdf5":
            print(f"         URL : {get_spec(name).source}\n"
                  "         Sans accès réseau : téléchargez le fichier ailleurs puis --file <chemin>.")
        return False
    s = ds.summary()
    print(f"  [{OK}] chargé en {time.time() - t0:.1f}s : {s['trajectories']} trajectoires, "
          f"{s['transitions']} transitions, longueur moyenne {s['ep_len_mean']:.0f}")
    print(f"         return  min/moy/max = {s['return_min']:.1f} / {s['return_mean']:.1f} / {s['return_max']:.1f}")
    print(f"         score normalisé moyen = {s['norm_score_mean']:.1f}, max = {s['norm_score_max']:.1f}")

    ok = True
    if env is not None:
        dims = (env.observation_space.shape[0], env.action_space.shape[0])
        match = dims == (ds.obs_dim, ds.act_dim)
        ok &= match
        print(f"  [{OK if match else FAIL}] dimensions dataset {(ds.obs_dim, ds.act_dim)} vs env {dims}")
        amax = max(np.abs(t.actions).max() for t in ds.trajectories)
        print(f"  [{OK if amax <= 1.0 + 1e-6 else WARN}] |action| max dans le dataset = {amax:.3f}")
        replay_check(ds, env)

    rtg = ds.trajectories[0].returns_to_go()
    ok_rtg = np.isclose(rtg[0], ds.trajectories[0].episode_return, rtol=1e-4)
    ok &= ok_rtg
    print(f"  [{OK if ok_rtg else FAIL}] RTG_0 = return de l'épisode ({rtg[0]:.2f})")
    return ok


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="hopper-medium-replay-v2")
    p.add_argument("--file", default=None, help="fichier HDF5 local (évite le téléchargement)")
    p.add_argument("--root", default=None, help="dossier des datasets (défaut : ./datasets ou $D4RL_DATASET_DIR)")
    p.add_argument("--skip-dataset", action="store_true")
    args = p.parse_args()

    kitchen = get_spec(args.dataset).domain == "kitchen"
    ok = check_packages(kitchen)
    env_ok, env = check_env(args.dataset)
    ok &= env_ok
    if not args.skip_dataset:
        ok &= check_dataset(args.dataset, args.file, args.root, env)
    section("Résultat")
    print(f"  [{OK if ok else FAIL}] {'environnement prêt' if ok else 'des vérifications ont échoué'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
