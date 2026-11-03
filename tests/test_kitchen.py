"""Franka Kitchen : environnement (réel) et chargeur Minari (dataset simulé)."""
import sys
import types

import numpy as np
import pytest

from src.data.registry import get_spec


def test_kitchen_spec():
    s = get_spec("kitchen-mixed-v2")
    assert s.backend == "minari" and s.max_episode_len == 280 and s.ref_max_score == 4.0
    assert "bottom burner" in s.env_kwargs["tasks_to_complete"]


def test_kitchen_env_runs():
    pytest.importorskip("gymnasium_robotics")
    from src.envs.make_env import make_env

    env = make_env("kitchen-partial-v2", seed=0)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (59,) and env.action_space.shape == (9,)
    for _ in range(5):
        obs, r, term, trunc, _ = env.step(env.action_space.sample())
    assert obs.dtype == np.float32 and np.isfinite(obs).all()


class _FakeEpisode:
    def __init__(self, T, rng):
        self.observations = {"observation": rng.normal(size=(T + 1, 59)).astype(np.float32),
                             "achieved_goal": {}, "desired_goal": {}}
        self.actions = rng.uniform(-1, 1, size=(T, 9)).astype(np.float32)
        self.rewards = np.zeros(T)
        self.rewards[[T // 3, T // 2]] = 1.0             # 2 sous-tâches accomplies
        self.terminations = np.zeros(T, dtype=bool)
        self.truncations = np.zeros(T, dtype=bool)
        self.truncations[-1] = True


def test_minari_loader(monkeypatch):
    rng = np.random.default_rng(0)
    episodes = [_FakeEpisode(T, rng) for T in (40, 55, 30)]
    fake = types.SimpleNamespace(load_dataset=lambda name, download=False: types.SimpleNamespace(
        iterate_episodes=lambda: iter(episodes)))
    monkeypatch.setitem(sys.modules, "minari", fake)

    from src.data.d4rl_loader import load_dataset
    ds = load_dataset("kitchen-mixed-v2")
    assert len(ds.trajectories) == 3 and (ds.obs_dim, ds.act_dim) == (59, 9)
    tr = ds.trajectories[0]
    np.testing.assert_array_equal(tr.next_observations[:-1], tr.observations[1:])
    s = ds.summary()
    assert s["return_max"] == 2.0 and s["norm_score_max"] == pytest.approx(50.0)
