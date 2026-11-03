"""Decision Transformer : forme, causalité, masque de padding, apprentissage, rollout, entraînement complet.

Ignoré automatiquement si PyTorch n'est pas installé.
"""
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.models.decision_transformer import DecisionTransformer  # noqa: E402
from tests.conftest import NAME  # noqa: E402

OBS, ACT, K = 11, 3, 8


def batch(B=4, k=K, seed=0):
    g = torch.Generator().manual_seed(seed)
    return dict(
        states=torch.randn(B, k, OBS, generator=g),
        actions=torch.rand(B, k, ACT, generator=g) * 2 - 1,
        rtg=torch.randn(B, k, 1, generator=g),
        timesteps=torch.arange(k).repeat(B, 1),
        mask=torch.ones(B, k),
    )


@pytest.fixture
def model():
    torch.manual_seed(0)
    return DecisionTransformer(OBS, ACT, context_len=K, embed_dim=32, n_layers=2, n_heads=2).eval()


def test_forward_shape_and_range(model):
    out = model(**batch())
    assert out.shape == (4, K, ACT) and out.abs().max() <= 1.0


def test_causality(model):
    """a_t ne dépend ni de a_t, ni d'aucun token futur."""
    b = batch()
    ref = model(**b)
    t = 3
    b2 = {k: v.clone() for k, v in b.items()}
    b2["actions"][:, t:] += 5.0
    b2["states"][:, t + 1:] += 5.0
    b2["rtg"][:, t + 1:] += 5.0
    out = model(**b2)
    torch.testing.assert_close(out[:, : t + 1], ref[:, : t + 1])
    assert not torch.allclose(out[:, t + 1:], ref[:, t + 1:])


def test_padding_is_ignored(model):
    """Le contenu des tokens de padding (masque=0) n'influence pas les prédictions réelles."""
    b = batch()
    b["mask"][:, :3] = 0
    ref = model(**b)
    b2 = {k: v.clone() for k, v in b.items()}
    b2["states"][:, :3] = 100.0
    b2["rtg"][:, :3] = -100.0
    out = model(**b2)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out[:, 3:], ref[:, 3:])


def test_get_action_matches_unpadded_forward(model):
    """Historique court (T<K) : padding à gauche == forward direct sur les T pas."""
    b = batch(B=2, k=3)
    a1 = model.get_action(b["states"], b["actions"], b["rtg"], b["timesteps"])
    a2 = model(b["states"], b["actions"], b["rtg"], b["timesteps"])[:, -1]
    torch.testing.assert_close(a1, a2)
    # historique long (T>K) : seuls les K derniers pas comptent
    long = batch(B=2, k=K)
    pre = {k: torch.cat([torch.randn_like(v[:, :5].float()).to(v.dtype), v], 1) for k, v in long.items()}
    pre["timesteps"] = torch.arange(K + 5).repeat(2, 1)
    long["timesteps"] = torch.arange(5, K + 5).repeat(2, 1)
    torch.testing.assert_close(
        model.get_action(pre["states"], pre["actions"], pre["rtg"], pre["timesteps"]),
        model(long["states"], long["actions"], long["rtg"], long["timesteps"])[:, -1],
    )


def test_overfits_small_batch():
    from src.training.train_dt import dt_loss

    torch.manual_seed(0)
    m = DecisionTransformer(OBS, ACT, context_len=K, embed_dim=32, n_layers=2, n_heads=2, dropout=0.0)
    b = batch(B=16)
    b["mask"][:8, :2] = 0
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    first = dt_loss(m, b).item()
    for _ in range(300):
        loss = dt_loss(m, b)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1 * first


def test_rollout_hopper(model, fake_hdf5):
    from src.evaluation.rollout_dt import evaluate_dt

    m = DecisionTransformer(OBS, ACT, context_len=K, embed_dim=32, n_layers=1, n_heads=1)
    r = evaluate_dt(m, NAME, 3600.0, 1000.0, np.zeros(OBS, np.float32), np.ones(OBS, np.float32),
                    episodes=3, seed=0, max_ep_len=50)
    assert len(r["returns"]) == 3 and 1 <= r["length_mean"] <= 50
    assert np.isfinite(r["score_mean"])


def test_train_end_to_end(fake_hdf5, tmp_path, monkeypatch):
    from src.training.train_dt import train

    monkeypatch.setenv("D4RL_DATASET_DIR", str(fake_hdf5[0].parent))  # fichier déjà « téléchargé »
    res = train(NAME, seed=0, output_dir=str(tmp_path), overrides=[
        "train.max_steps=30", "train.log_every=10", "train.eval_every=20", "train.warmup_steps=5",
        "train.batch_size=8", "eval.episodes=2", "eval.final_episodes=2", "model.context_len=5",
    ])
    run = tmp_path / "dt" / NAME / "seed0"
    assert (run / "model.pt").exists() and (run / "metrics.jsonl").exists()
    assert json.loads((run / "results.json").read_text())["score"] == pytest.approx(res["score"])
    assert len(res["per_target"]) == 2

    from src.evaluation.eval_dt import load_dt
    m, ck = load_dt(str(run / "model.pt"), "cpu")
    assert m.context_len == 5 and ck["dataset"] == NAME
