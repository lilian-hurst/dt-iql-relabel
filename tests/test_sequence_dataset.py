"""Découpage en sous-séquences de longueur K (NumPy uniquement)."""
import numpy as np
import pytest

from src.data.d4rl_loader import load_dataset
from tests.conftest import NAME

K = 20


@pytest.fixture(scope="module")
def ds(fake_hdf5):
    return load_dataset(NAME, path=fake_hdf5[0])


def make_seq(ds, **kw):
    from src.data.sequence_dataset import SequenceDataset
    return SequenceDataset(ds, context_len=K, return_scale=1000.0, max_timestep=4096, **kw)


def test_window_inside_trajectory(ds):
    seq = make_seq(ds)
    tr = ds.trajectories[0]
    w = seq.get_window(0, 2)
    assert w["mask"].all()
    np.testing.assert_allclose(w["states"], (tr.observations[2:2 + K] - ds.state_mean) / ds.state_std, rtol=1e-5)
    np.testing.assert_allclose(w["rtg"][:, 0], tr.returns_to_go()[2:2 + K] / 1000.0, rtol=1e-5)
    np.testing.assert_array_equal(w["timesteps"], np.arange(2, 2 + K))


def test_window_left_padding_at_episode_end(ds):
    seq = make_seq(ds)
    T = seq.lengths[0]
    w = seq.get_window(0, T - 3)  # 3 pas réels, K-3 de padding à gauche
    assert w["mask"].tolist() == [0.0] * (K - 3) + [1.0] * 3
    assert np.all(w["states"][: K - 3] == 0) and np.all(w["actions"][: K - 3] == 0)
    np.testing.assert_array_equal(w["timesteps"][-3:], np.arange(T - 3, T))
    np.testing.assert_allclose(w["rtg"][-1, 0], ds.trajectories[0].rewards[-1] / 1000.0, rtol=1e-5)


def test_sample_shapes_and_coverage(ds):
    seq = make_seq(ds)
    b = seq.sample(256, np.random.default_rng(0))
    assert b["states"].shape == (256, K, ds.obs_dim) and b["actions"].shape == (256, K, ds.act_dim)
    assert b["rtg"].shape == (256, K, 1) and b["timesteps"].dtype == np.int64
    assert (b["mask"].sum(1) >= 1).all()           # au moins un pas réel par fenêtre
    assert (b["mask"][:, -1] == 1).all()           # padding uniquement à gauche


def test_custom_rtgs_are_used(ds):
    rtgs = [np.full(len(t), 7.0) for t in ds.trajectories]
    seq = make_seq(ds, rtgs=rtgs)
    np.testing.assert_allclose(seq.get_window(1, 0)["rtg"][:, 0][seq.get_window(1, 0)["mask"] > 0], 0.007)
    with pytest.raises(ValueError):
        make_seq(ds, rtgs=rtgs[:-1])
