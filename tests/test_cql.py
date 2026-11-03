"""CQL et baseline QDT (critique CQL -> relabellisation récursive -> DT). Ignoré si PyTorch est absent."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.models.cql import CQL, TanhGaussianPolicy  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from tests.conftest import NAME  # noqa: E402


def test_tanh_policy_log_prob_consistent():
    torch.manual_seed(0)
    pi = TanhGaussianPolicy(3, 2, [32], -20.0, 2.0)
    obs = torch.randn(64, 3)
    a, logp = pi(obs)
    assert a.abs().max() < 1.0
    torch.testing.assert_close(pi.log_prob(obs, a), logp, atol=1e-3, rtol=1e-3)


def test_cql_penalty_keeps_ood_actions_below_data():
    """Données : une seule action (a=0.5) par état. CQL doit sous-estimer les actions hors données."""
    torch.manual_seed(0)
    cfg = load_config("cql.yaml", ["model.hidden_dims=[64,64]", "train.cql_alpha=5.0"])
    agent = CQL(2, 1, cfg.model, cfg.train)
    n = 512
    obs = torch.randn(n, 2)
    act = torch.full((n, 1), 0.5)
    buf = dict(observations=obs, actions=act, rewards=torch.ones(n), next_observations=obs,
               dones=torch.ones(n), timesteps=torch.zeros(n, dtype=torch.long))
    for _ in range(1500):
        idx = torch.randint(0, n, (128,))
        m = agent.update({k: v[idx] for k, v in buf.items()})
    assert np.isfinite(list(m.values())).all()
    q_data = agent.qf(obs, act).mean().item()
    q_ood = agent.qf(obs, torch.full((n, 1), -0.9)).mean().item()
    assert q_ood < q_data - 0.1


def test_cql_value_with_time_feature():
    cfg = load_config("cql.yaml", ["model.time_feature=true", "train.value_samples=4"])
    agent = CQL(3, 2, cfg.model, cfg.train, max_timestep=100)
    v = agent.value(torch.zeros(5, 3), torch.arange(5))
    assert v.shape == (5,) and torch.isfinite(v).all()


def test_qdt_pipeline(fake_hdf5, tmp_path, monkeypatch):
    from src.training import relabel_returns, train_dt

    monkeypatch.setenv("D4RL_DATASET_DIR", str(fake_hdf5[0].parent))
    small = ["train.max_steps=30", "train.log_every=15", "train.batch_size=32", "model.hidden_dims=[32,32]"]
    real_load = relabel_returns.load_config

    def small_cfg(name, ov=(), **kw):
        cfg = real_load(name, ov, **kw)
        if name == "relabel.yaml":
            cfg.critics.cql.overrides = list(cfg.critics.cql.overrides) + small
        return cfg

    monkeypatch.setattr(relabel_returns, "load_config", small_cfg)
    rtg_file = relabel_returns.relabel(NAME, 0, ["value_source=cql", "rule=qdt"], output_dir=str(tmp_path),
                                       train_critic=True)
    assert rtg_file.parent.parent.parent.name == "cql_qdt_a1"
    assert (tmp_path / "cql_critic" / NAME / "seed0" / "results.json").exists()
    res = train_dt.train(NAME, 0, ["train.max_steps=10", "train.eval_every=0", "train.log_every=5",
                                   "eval.final_episodes=2", "model.context_len=5"],
                         method="qdt", rtg_file=str(rtg_file), output_dir=str(tmp_path))
    assert res["method"] == "qdt"
