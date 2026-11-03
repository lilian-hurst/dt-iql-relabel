"""Ablations : planification (réutilisation des runs), tableaux et figures (sans PyTorch)."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from launcher import plan_jobs  # noqa: E402
from run_ablations import load_ablations, plan_ablation, report  # noqa: E402
from tests.fake_results import fill_ablation  # noqa: E402
from src.utils.config import Config  # noqa: E402

DS = ["hopper-medium-replay-v2", "walker2d-medium-replay-v2"]


@pytest.fixture
def plans(tmp_path):
    defaults, abl = load_ablations(datasets=DS, seeds=[0, 1])
    return {name: plan_ablation(name, ab, defaults, str(tmp_path)) for name, ab in abl.items()}, tmp_path


def _dirs(manifest, method):
    return {e["value_key"]: e["dir"] for e in manifest["entries"] if e["method"] == method}


def test_default_values_share_one_run(plans):
    p, _ = plans
    # K=20, α=1, seuil=0, γ=0.995, quantile=1 : la configuration par défaut, un seul run partagé
    assert _dirs(p["context_len"][0], "iql_dt")["20"] == "shared/iql_dt"
    assert _dirs(p["relabel_strength"][0], "iql_dt")["1"] == "shared/iql_dt"
    assert _dirs(p["q_threshold"][0], "iql_dt")["0"] == "shared/iql_dt"
    assert _dirs(p["critic_discount"][0], "iql_dt")["0.995"] == "shared/iql_dt"
    assert _dirs(p["dataset_quality"][0], "dt")["1"] == "shared/dt"


def test_non_default_values(plans):
    p, _ = plans
    assert _dirs(p["context_len"][0], "dt")["5"] == "shared/dt__context_len5"            # même critique
    assert _dirs(p["relabel_strength"][0], "qdt")["0.5"] == "shared/qdt__alpha0.5"
    assert _dirs(p["dataset_quality"][0], "iql")["0.5"] == "dataset_quality/max_return_quantile=0.5/iql"
    assert _dirs(p["critic_discount"][0], "iql_dt")["0.99"] == "critic_discount/discount=0.99/iql_dt"
    m = p["relabel_strength"][0]
    assert m["references"] == [{"method": "dt", "dir": "shared/dt"}, {"method": "iql", "dir": "shared/iql"}]


def test_jobs_reuse_critics_and_relabel(plans):
    p, root = plans
    jobs = {}
    for _, exps in p.values():
        for exp, suffix in exps:
            for j in plan_jobs(exp, suffix=suffix):
                jobs.setdefault(str(j.done_file), j)
    rel = lambda j: j.done_file.relative_to(root / "ablations").as_posix()
    critics = sorted(rel(j) for j in jobs.values() if "critic/" in rel(j))
    # hopper, seed 0 : critiques partagés + un par valeur de γ / de quantile
    h0 = [c for c in critics if "hopper" in c and c.endswith("seed0/results.json")]
    assert "shared/iql_critic/hopper-medium-replay-v2/seed0/results.json" in h0
    assert "shared/cql_critic/hopper-medium-replay-v2/seed0/results.json" in h0
    assert len(h0) == 2 + 3 + 3 * 2   # partagés + γ∈{0.99,0.999,1} + quantiles {0.25,0.5,0.75} x (IQL, CQL)
    # α=0.5 relabellise avec le critique partagé dans son propre dossier
    alpha = next(j for j in jobs.values() if rel(j).startswith("shared/relabel/iql_max_q_a0.5/hopper"))
    assert alpha.deps[0].relative_to(root / "ablations").parts[:2] == ("shared", "iql_critic")
    # K=5 entraîne un DT sur les RTG relabellisés par défaut (α=1), sans nouveau critique
    k5 = next(j for j in jobs.values() if rel(j).startswith("shared/iql_dt__context_len5/hopper"))
    assert "iql_max_q_a1" in str(k5.deps[0]) and "model.context_len=5" in k5.cmd


def test_param_must_affect_method():
    defaults, abl = load_ablations(datasets=DS, seeds=[0])
    bad = Config({"label": "x", "param": "relabel_overrides", "key": "alpha", "values": [0.5],
                  "methods": ["dt"], "reference_methods": []})
    with pytest.raises(ValueError):
        plan_ablation("bad", bad, defaults)
    with pytest.raises(KeyError):
        load_ablations(only=["inexistante"])


def test_report_tables_and_figure(plans):
    p, root = plans
    manifest = p["context_len"][0]
    fill_ablation(root / "ablations", manifest)
    path = root / "ablations" / "manifests" / "context_len.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(manifest))
    plot = True
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        plot = False
    report(path, results_dir=root / "res", figures_dir=root / "fig", plot=plot)
    md = (root / "res" / "table.md").read_text()
    assert "| Moyenne | 5 |" in md and "IQL (réf.)" in md
    assert (root / "res" / "table.tex").read_text().startswith("\\begin{tabular}")
    if plot:
        assert (root / "fig" / "context_len.pdf").exists() and (root / "fig" / "context_len.png").exists()


def test_base_experiment_runs_are_reused(tmp_path):
    from launcher import load_experiment

    repo = Path(__file__).resolve().parents[1]
    exp = load_experiment(repo / "experiments/configs/laptop.yaml", output_dir=str(tmp_path))
    main = {str(j.done_file) for j in plan_jobs(exp)}
    defaults, abl = load_ablations(repo / "experiments/configs/ablations_laptop.yaml")
    assert defaults.name == "laptop" and defaults.seeds == [0, 1, 2]
    assert "train.max_steps=300000" in defaults.overrides.critic_overrides
    manifest, exps = plan_ablation("context_len", abl.context_len, defaults, str(tmp_path))
    assert _dirs(manifest, "iql_dt")["20"] == "iql_dt"                 # dossier de l'expérience laptop
    assert _dirs(manifest, "dt")["5"] == "dt__context_len5"
    abl_files = {str(j.done_file) for e, s in exps for j in plan_jobs(e, suffix=s)}
    default_runs = {f for f in abl_files if "__" not in f}
    assert default_runs and default_runs <= main                       # aucun recalcul des runs par défaut
