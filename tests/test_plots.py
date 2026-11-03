"""Figures de l'expérience principale et diagnostic de relabellisation (ignoré sans matplotlib)."""
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("matplotlib")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from plotting import load_curves, method_style, plot_experiment  # noqa: E402
from result_utils import mean_std, seed_average  # noqa: E402
from tests.fake_results import make_experiment, make_relabel_npz  # noqa: E402

DS = ["hopper-medium-v2", "hopper-medium-replay-v2"]


def test_method_styles_are_fixed_and_distinct():
    styles = [method_style(m) for m in ("dt", "iql_dt", "iql", "qdt", "iql_qdt", "cql")]
    assert len({c for c, _ in styles}) == 6 and len({mk for _, mk in styles}) == 6
    assert method_style("iql_dt__alpha0.5") == method_style("iql_dt")   # la couleur suit la méthode


def test_seed_average_requires_all_datasets():
    import pandas as pd
    df = pd.DataFrame([
        {"method": "dt", "dataset": DS[0], "seed": 0, "score": 10.0},
        {"method": "dt", "dataset": DS[1], "seed": 0, "score": 30.0},
        {"method": "dt", "dataset": DS[0], "seed": 1, "score": 50.0},   # seed 1 incomplète : exclue
    ])
    avg = seed_average(df, DS, ["method"])
    assert avg["score"].tolist() == [20.0]
    s = mean_std(df, ["method", "dataset"])
    assert s.set_index("dataset").loc[DS[0], "std"] == pytest.approx(20.0)


def test_experiment_figures(tmp_path):
    runs = tmp_path / "runs" / "exp"
    make_experiment(runs, ["dt", "iql", "iql_dt"], DS, [0, 1, 2])
    make_relabel_npz(runs / "relabel" / "iql_max_q_a1" / DS[1] / "seed0" / "rtg.npz")
    curves, finals = load_curves(runs)
    assert set(curves["method"]) == {"dt", "iql_dt"} and set(finals["method"]) == {"iql"}
    last = curves[curves["step"] == curves["step"].max()]
    assert np.isfinite(last["score"]).all()

    paths = plot_experiment(runs, tmp_path / "fig")
    names = {p.name for p in paths}
    assert {"comparison.pdf", "comparison.png", "learning_curves.pdf", "learning_curves.png"} <= names
    assert f"iql_max_q_a1__{DS[1]}.pdf" in names
    assert all(p.stat().st_size > 1000 for p in paths)
