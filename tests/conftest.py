"""Fixtures partagées : faux fichier D4RL v2 généré avec Hopper-v4 (aucun téléchargement)."""
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.envs.make_env import make_env  # noqa: E402

NAME = "hopper-medium-replay-v2"


@pytest.fixture(scope="session")
def fake_hdf5(tmp_path_factory):
    """~6 épisodes aléatoires de Hopper, dont un coupé par timeout (horizon court = 50)."""
    env = make_env(NAME, seed=0)
    rng = np.random.default_rng(0)
    buf = {k: [] for k in ("observations", "actions", "rewards", "terminals", "timeouts",
                           "next_observations", "qpos", "qvel")}
    horizon, n_eps = 50, 6
    for ep in range(n_eps):
        obs, _ = env.reset(seed=ep)
        for t in range(horizon):
            qpos, qvel = env.unwrapped.data.qpos.copy(), env.unwrapped.data.qvel.copy()
            a = rng.uniform(-1, 1, env.action_space.shape).astype(np.float32)
            nxt, r, term, _, _ = env.step(a)
            timeout = (t == horizon - 1) and not term
            for k, v in zip(buf, (obs, a, r, term, timeout, nxt, qpos, qvel)):
                buf[k].append(v)
            obs = nxt
            if term or timeout:
                break
    path = tmp_path_factory.mktemp("d4rl") / "hopper_medium_replay-v2.hdf5"  # nom attendu par le registre
    with h5py.File(path, "w") as f:
        for k in ("observations", "actions", "rewards", "terminals", "timeouts", "next_observations"):
            f.create_dataset(k, data=np.array(buf[k]))
        f.create_dataset("infos/qpos", data=np.array(buf["qpos"]))
        f.create_dataset("infos/qvel", data=np.array(buf["qvel"]))
    return path, n_eps, len(buf["rewards"])
