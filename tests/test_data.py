"""Tests hors-ligne du pipeline de données, sur un faux fichier D4RL généré avec Hopper-v4.

Le fichier imite le format D4RL v2 (observations, actions, rewards, terminals, timeouts,
next_observations, infos/qpos, infos/qvel). Lancer : pytest -q
"""
from __future__ import annotations

import numpy as np
import pytest

from src.data.d4rl_loader import discount_cumsum, load_dataset
from src.data.registry import get_spec
from src.envs.make_env import make_env
from src.evaluation.normalization import normalized_score
from src.utils.config import load_config

NAME = "hopper-medium-replay-v2"


def test_split_into_trajectories(fake_hdf5):
    path, n_eps, n = fake_hdf5
    ds = load_dataset(NAME, path=path, keep_infos=True)
    assert len(ds.trajectories) == n_eps
    assert ds.num_transitions == n
    assert (ds.obs_dim, ds.act_dim) == (11, 3)
    for tr in ds.trajectories:
        assert tr.terminals[:-1].sum() == 0          # terminal uniquement en fin d'épisode
        np.testing.assert_allclose(tr.observations[1:], tr.next_observations[:-1])


def test_returns_to_go():
    r = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(discount_cumsum(r), [6.0, 5.0, 3.0])
    np.testing.assert_allclose(discount_cumsum(r, 0.5), [1 + 0.5 * 2 + 0.25 * 3, 2 + 0.5 * 3, 3.0])


def test_state_normalization_and_transitions(fake_hdf5):
    path, _, n = fake_hdf5
    ds = load_dataset(NAME, path=path)
    z = ds.normalize_states(np.concatenate([t.observations for t in ds.trajectories]))
    np.testing.assert_allclose(z.mean(0), 0.0, atol=1e-4)
    tr = ds.transitions()
    assert all(v.shape[0] == n for v in tr.values())


def test_quantile_filter(fake_hdf5):
    path, n_eps, _ = fake_hdf5
    ds = load_dataset(NAME, path=path)
    low = ds.filter_by_return_quantile(0.5)
    assert 0 < len(low.trajectories) < n_eps
    assert low.episode_returns().max() <= np.median(ds.episode_returns()) + 1e-6


def test_replay_matches_dynamics(fake_hdf5):
    path, _, _ = fake_hdf5
    ds = load_dataset(NAME, path=path, keep_infos=True)
    env = make_env(NAME)
    tr = ds.trajectories[0]
    env.reset(seed=0)
    env.unwrapped.set_state(tr.infos["qpos"][3], tr.infos["qvel"][3])
    nxt, *_ = env.step(tr.actions[3])
    np.testing.assert_allclose(nxt, tr.next_observations[3], atol=1e-6)


def test_normalized_score_reference_points():
    spec = get_spec("hopper-medium-v2")
    assert normalized_score(spec.ref_min_score, spec) == pytest.approx(0.0)
    assert normalized_score(spec.ref_max_score, spec) == pytest.approx(100.0)
    assert get_spec("halfcheetah-medium-expert-v2").filename == "halfcheetah_medium_expert-v2.hdf5"
    assert get_spec("kitchen-mixed-v2").source == "D4RL/kitchen/mixed-v2"


def test_config_overrides_and_domain():
    cfg = load_config("dt.yaml", overrides=["model.context_len=5", "train.lr=3e-4"])
    assert cfg.model.context_len == 5 and cfg.train.lr == pytest.approx(3e-4)
    kcfg = load_config("iql.yaml", domain="kitchen")
    assert kcfg.train.beta == 0.5 and kcfg.model.policy_dropout == 0.1
    with pytest.raises(KeyError):
        load_config("dt.yaml", overrides=["model.contxt_len=5"])  # faute de frappe détectée
