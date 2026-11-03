"""Règles de relabellisation des RTG (NumPy uniquement)."""
import numpy as np
import pytest

from src.data.d4rl_loader import discount_cumsum
from src.data.relabel import relabel_stats, relabel_trajectory

rng = np.random.default_rng(0)
r = rng.uniform(0, 3, size=50)
R = discount_cumsum(r)


def test_none_is_monte_carlo():
    out, gain = relabel_trajectory(r, rule="none")
    np.testing.assert_allclose(out, R)
    assert not gain.any()


def test_max_q_gamma1_is_max_of_rtg_and_q():
    q = R + rng.normal(0, 10, size=len(r))
    out, _ = relabel_trajectory(r, q=q, rule="max_q", alpha=1.0, gamma=1.0)
    np.testing.assert_allclose(out, np.maximum(R, q))


def test_max_q_alpha_and_threshold():
    q = R + 20.0
    np.testing.assert_allclose(relabel_trajectory(r, q=q, rule="max_q", alpha=0.0)[0], R)
    np.testing.assert_allclose(relabel_trajectory(r, q=q, rule="max_q", alpha=0.5)[0], R + 10.0)
    # seuil : gain de 20 ignoré là où 20 <= 0.5 * |R_t|
    out, _ = relabel_trajectory(r, q=q, rule="max_q", min_improvement=0.5)
    np.testing.assert_allclose(out, np.where(20.0 > 0.5 * R, R + 20.0, R))


def test_max_q_discounted_compares_with_discounted_mc():
    gamma = 0.9
    G = discount_cumsum(r, gamma)
    out, gain = relabel_trajectory(r, q=G + 1.0, rule="max_q", gamma=gamma)  # Q = G^γ + 1
    np.testing.assert_allclose(gain, 1.0)
    np.testing.assert_allclose(out, R + 1.0)


def test_qdt_gamma1_matches_reference_recursion():
    v_next = R + rng.normal(0, 15, size=len(r))
    out, _ = relabel_trajectory(r, v_next=v_next, rule="qdt", gamma=1.0)
    ref = np.empty_like(r)
    ref[-1] = r[-1]
    for t in range(len(r) - 2, -1, -1):
        ref[t] = r[t] + max(ref[t + 1], v_next[t])
    np.testing.assert_allclose(out, ref)


def test_qdt_propagates_a_good_future_backwards():
    """Stitching : une valeur élevée en milieu de trajectoire relève tous les RTG antérieurs."""
    v_next = np.full(len(r), -1e9)
    v_next[30] = R[31] + 100.0           # V(s_31) annonce +100 de mieux que la suite observée
    out, gain = relabel_trajectory(r, v_next=v_next, rule="qdt", gamma=1.0)
    np.testing.assert_allclose(out[:31], R[:31] + 100.0)
    np.testing.assert_allclose(out[31:], R[31:])
    assert gain[30] == pytest.approx(100.0) and gain.sum() == pytest.approx(100.0)


def test_relabeled_never_below_original_and_stats():
    q = R + rng.normal(0, 10, size=len(r))
    v = R + rng.normal(0, 10, size=len(r))
    for rule, kw in (("max_q", {"q": q}), ("qdt", {"v_next": v})):
        out, _ = relabel_trajectory(r, rule=rule, gamma=0.99, **kw)
        assert (out >= R - 1e-9).all()
    s = relabel_stats([R], [np.maximum(R, q)])
    assert 0 < s["frac_steps_relabeled"] < 1 and s["mean_gain"] > 0


def test_unknown_rule():
    with pytest.raises(ValueError):
        relabel_trajectory(r, rule="foo")
