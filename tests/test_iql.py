"""IQL et pipeline critique -> relabellisation -> DT. Ignoré si PyTorch est absent."""
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.models.iql import IQL, expectile_loss  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from tests.conftest import NAME  # noqa: E402


def test_expectile_loss_asymmetry():
    u = torch.tensor([1.0, -1.0])
    assert expectile_loss(u[:1], 0.7).item() == pytest.approx(0.7)
    assert expectile_loss(u[1:], 0.7).item() == pytest.approx(0.3)
    assert expectile_loss(u, 0.5).item() == pytest.approx(0.5)


def test_value_is_upper_expectile_of_q():
    """Sur un état unique avec Q fixé, V converge vers l'expectile τ des Q (> moyenne si τ>0.5)."""
    torch.manual_seed(0)
    cfg = load_config("iql.yaml", ["train.train_policy=false", "train.critic_lr=1e-2"])
    agent = IQL(2, 1, cfg.model, cfg.train)
    qs = torch.linspace(0, 10, 101)
    obs = torch.zeros(101, 2)
    for _ in range(1500):
        v = agent.vf(obs)
        loss = expectile_loss(qs - v, 0.7)
        agent.v_opt.zero_grad()
        loss.backward()
        agent.v_opt.step()
    assert agent.vf(obs[:1]).item() > 5.5   # moyenne = 5, expectile 0.7 ≈ 6.1


def test_time_feature_requires_timesteps():
    cfg = load_config("iql.yaml", ["model.time_feature=true"])
    agent = IQL(3, 2, cfg.model, cfg.train, max_timestep=100)
    assert agent.value(torch.zeros(4, 3), torch.arange(4)).shape == (4,)
    with pytest.raises(ValueError):
        agent.value(torch.zeros(4, 3))


def test_iql_update_learns_one_step_rewards():
    """Transitions terminales : Q doit apprendre r(s, a)."""
    torch.manual_seed(0)
    cfg = load_config("iql.yaml", ["train.batch_size=64"])
    agent = IQL(2, 1, cfg.model, cfg.train)
    n = 512
    obs = torch.randn(n, 2)
    act = torch.rand(n, 1) * 2 - 1
    rew = (obs[:, 0] + act[:, 0]).float()
    buf = dict(observations=obs, actions=act, rewards=rew, next_observations=obs,
               dones=torch.ones(n), timesteps=torch.zeros(n, dtype=torch.long))
    for _ in range(2000):
        idx = torch.randint(0, n, (64,))
        m = agent.update({k: v[idx] for k, v in buf.items()})
    assert np.isfinite(list(m.values())).all()
    err = (agent.qf(obs, act) - rew).abs().mean().item()
    assert err < 0.15


def test_full_pipeline(fake_hdf5, tmp_path, monkeypatch):
    """critique IQL -> relabel_returns -> DT sur RTG relabellisés, plus baseline IQL."""
    from src.training import relabel_returns, train_dt, train_iql

    monkeypatch.setenv("D4RL_DATASET_DIR", str(fake_hdf5[0].parent))
    small = ["train.max_steps=50", "train.log_every=25", "train.batch_size=32", "train.eval_every=0"]

    res = train_iql.train(NAME, 0, small + ["eval.final_episodes=2"], "iql", str(tmp_path))
    assert np.isfinite(res["score"])

    # critique entraîné à la volée (--train-critic) puis relabellisation
    monkeypatch.setattr(relabel_returns, "load_config",
                        lambda name, ov=(), **kw: _small_critic(name, ov, small))
    rtg_file = relabel_returns.relabel(NAME, 0, ["rule=qdt", "alpha=1.0"], output_dir=str(tmp_path),
                                       train_critic=True)
    data = np.load(rtg_file)
    assert data["rtg"].shape == data["rtg_original"].shape == data["q"].shape
    assert (data["rtg"] >= data["rtg_original"] - 1e-3).all()
    stats = json.loads((rtg_file.parent / "stats.json").read_text())
    assert stats["gamma"] == pytest.approx(0.995)

    res = train_dt.train(NAME, 0, ["train.max_steps=10", "train.eval_every=0", "train.log_every=5",
                                   "eval.final_episodes=2", "model.context_len=5"],
                         method="iql_dt", rtg_file=str(rtg_file), output_dir=str(tmp_path))
    assert res["method"] == "iql_dt" and (tmp_path / "iql_dt" / NAME / "seed0" / "results.json").exists()


def _small_critic(name, overrides, small):
    cfg = load_config(name, overrides)
    if name == "relabel.yaml":
        for c in cfg.critics.values():
            c.overrides = list(c.overrides) + small
    return cfg
