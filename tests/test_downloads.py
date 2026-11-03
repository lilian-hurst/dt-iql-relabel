"""Téléchargement des HDF5 : repli sur la source suivante, rejet des fichiers invalides (sans réseau).

Les sources sont remplacées par des URL file:// : même code (urllib) que pour HTTP(S).
"""
import h5py
import numpy as np
import pytest

from src.data import registry
from src.data.d4rl_loader import check_hdf5, fetch_hdf5, load_dataset
from tests.conftest import NAME


@pytest.fixture
def mirror(fake_hdf5, monkeypatch, tmp_path):
    """Source 'official' introuvable, 'hf_mirror' = dossier contenant le faux fichier D4RL."""
    monkeypatch.setitem(registry.D4RL_HDF5_SOURCES, "official", f"file://{tmp_path}/absent/{{filename}}")
    monkeypatch.setitem(registry.D4RL_HDF5_SOURCES, "hf_mirror", f"file://{fake_hdf5[0].parent}/{{filename}}")
    monkeypatch.delenv("D4RL_DATASET_DIR", raising=False)
    return tmp_path / "datasets"


def test_spec_urls():
    spec = registry.get_spec("hopper-medium-replay-v2")
    urls = dict(spec.download_urls())
    assert urls["official"].endswith("/gym_mujoco_v2/hopper_medium_replay-v2.hdf5")
    assert urls["hf_mirror"] == "https://huggingface.co/datasets/imone/D4RL/resolve/main/hopper_medium_replay-v2.hdf5"
    assert registry.get_spec("kitchen-mixed-v2").download_urls() == []
    with pytest.raises(KeyError):
        spec.download_urls(("inconnue",))


def test_fallback_to_mirror(mirror, capsys):
    path = fetch_hdf5(registry.get_spec(NAME), root=str(mirror))
    assert path == mirror / "hopper_medium_replay-v2.hdf5" and check_hdf5(path) is None
    assert "source « hf_mirror »" in capsys.readouterr().err
    ds = load_dataset(NAME, root=str(mirror))            # deuxième appel : fichier local réutilisé
    assert len(ds.trajectories) > 0


def test_corrupted_file_is_replaced(mirror):
    mirror.mkdir(parents=True)
    bad = mirror / "hopper_medium_replay-v2.hdf5"
    bad.write_text("<html>404</html>")                  # page d'erreur enregistrée à la place du fichier
    assert check_hdf5(bad) is not None
    path = fetch_hdf5(registry.get_spec(NAME), root=str(mirror))
    assert check_hdf5(path) is None


def test_all_sources_fail(mirror, monkeypatch, tmp_path):
    monkeypatch.setitem(registry.D4RL_HDF5_SOURCES, "hf_mirror", f"file://{tmp_path}/absent2/{{filename}}")
    with pytest.raises(RuntimeError, match="impossible d'obtenir"):
        fetch_hdf5(registry.get_spec(NAME), root=str(mirror))
    assert not list(mirror.glob("*.part"))               # pas de fichier partiel laissé sur le disque


def test_check_hdf5_rejects_inconsistent(tmp_path):
    f = tmp_path / "x.hdf5"
    with h5py.File(f, "w") as h:
        for k, n in (("observations", 10), ("actions", 10), ("rewards", 9), ("terminals", 10)):
            h.create_dataset(k, data=np.zeros(n))
    assert "incohérentes" in check_hdf5(f)
