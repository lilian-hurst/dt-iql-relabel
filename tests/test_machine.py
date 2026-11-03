"""Réglages par machine : fichier experiment.local.yaml, choix du device et des threads."""
import shutil

import pytest

from src.utils import config as config_mod


@pytest.fixture
def config_dir(tmp_path, monkeypatch):
    for f in config_mod.CONFIG_DIR.glob("*.yaml"):
        if f.name != "experiment.local.yaml":
            shutil.copy(f, tmp_path / f.name)
    monkeypatch.setattr(config_mod, "CONFIG_DIR", tmp_path)
    return tmp_path


def test_local_overlay(config_dir):
    base = config_mod.load_experiment_config()
    assert base.device == "auto" and base.devices == {}
    (config_dir / "experiment.local.yaml").write_text(
        "devices: {dt: mps, iql: cpu}\nthroughput: {dt: 80.0}\nthreads: 3\n")
    cfg = config_mod.load_experiment_config()
    assert cfg.devices == {"dt": "mps", "iql": "cpu"} and cfg.threads == 3
    assert cfg.throughput["dt"] == 80.0 and cfg.output_dir == "runs"   # le reste vient de experiment.yaml


def test_device_priority(monkeypatch):
    torch = pytest.importorskip("torch")
    from src.utils.seed import device_for

    cfg = config_mod.Config({"device": "auto", "devices": {"iql": "cpu"}})
    assert device_for("iql", cfg) == torch.device("cpu")
    assert device_for("dt", cfg, override="cpu") == torch.device("cpu")
    monkeypatch.setenv("DTIQL_DEVICE", "cpu")
    assert device_for("dt", cfg) == torch.device("cpu")


def test_mps_fallback_env_set_by_package():
    import os

    import src  # noqa: F401
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"
