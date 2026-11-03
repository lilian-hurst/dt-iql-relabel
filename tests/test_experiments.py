"""Lanceur d'expériences et agrégation (sans PyTorch)."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from aggregate import aggregate, collect  # noqa: E402
from launcher import Job, load_experiment, plan_jobs, run_jobs  # noqa: E402

NAME = "hopper-medium-replay-v2"


def test_plan_smoke(tmp_path):
    exp = load_experiment(Path(__file__).parents[1] / "experiments/configs/smoke.yaml", output_dir=str(tmp_path))
    jobs = plan_jobs(exp)
    names = [j.name for j in jobs]
    assert len(jobs) == len(set(j.done_file for j in jobs)) == 8
    # dt, iql, 2 critiques en phase 0 ; 2 relabellisations ; 2 DT relabellisés
    assert [j.phase for j in jobs].count(0) == 4 and [j.phase for j in jobs].count(2) == 2
    qdt = next(j for j in jobs if j.name == f"qdt/{NAME}/s0")
    assert "--rtg-file" in qdt.cmd and qdt.deps[0].name == "rtg.npz" and "cql_qdt_a1" in str(qdt.deps[0])
    relabel = next(j for j in jobs if j.name.startswith("relabel/cql_qdt"))
    assert relabel.deps[0].parts[-4] == "cql_critic"
    assert f"iql_critic/{NAME}/s0" in names


def test_shared_critic_and_suite_expansion(tmp_path):
    exp = load_experiment(None, datasets="mujoco", methods=["iql_dt", "iql_qdt"], seeds=[0, 1],
                          output_dir=str(tmp_path))
    jobs = plan_jobs(exp)
    critics = [j for j in jobs if j.name.startswith("iql_critic/")]
    assert len(critics) == 9 * 2                     # un critique par (dataset, seed), partagé
    assert len(jobs) == 9 * 2 * (1 + 2 + 2)


def test_unknown_keys_and_methods(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: x\nmethod: [dt]\n")
    with pytest.raises(KeyError):
        load_experiment(bad)
    with pytest.raises(KeyError):
        plan_jobs(load_experiment(None, methods=["foo"], datasets=[NAME]))


def _job(tmp_path, name, code, phase=0, deps=()):
    done = tmp_path / f"{name}.done"
    return Job(name, [sys.executable, "-c", code.format(done=done)], done, phase, list(deps))


def test_run_jobs_resume_failure_and_dependencies(tmp_path):
    ok = _job(tmp_path, "ok", "open(r'{done}', 'w').write('x')")
    bad = _job(tmp_path, "bad", "raise SystemExit(1)")
    child_ok = _job(tmp_path, "child_ok", "open(r'{done}', 'w').write('x')", 1, [ok.done_file])
    child_bad = _job(tmp_path, "child_bad", "open(r'{done}', 'w').write('x')", 1, [bad.done_file])
    status = run_jobs([ok, bad, child_ok, child_bad], tmp_path, workers=2)
    assert status == {"ok": "done", "bad": "failed", "child_ok": "done", "child_bad": "failed"}
    assert not child_bad.done_file.exists()
    assert (tmp_path / "logs" / "bad.log").exists()
    # relance : les jobs terminés sont repris du cache
    assert run_jobs([ok, child_ok], tmp_path)["ok"] == "cached"


def test_aggregate(tmp_path):
    runs = tmp_path / "runs" / "exp"
    for m, s, score in [("dt", 0, 60.0), ("dt", 1, 70.0), ("iql_dt", 0, 80.0), ("iql_dt", 1, 90.0)]:
        p = runs / m / NAME / f"seed{s}"
        p.mkdir(parents=True)
        (p / "results.json").write_text(json.dumps({"method": m, "dataset": NAME, "seed": s, "score": score}))
    crit = runs / "iql_critic" / NAME / "seed0"
    crit.mkdir(parents=True)
    (crit / "results.json").write_text(json.dumps({"method": "iql_critic", "dataset": NAME, "seed": 0}))

    assert len(collect(runs)) == 4                    # le critique (sans score) est ignoré
    table = aggregate(runs, out_dir=tmp_path / "out")
    assert table.at[NAME, "DT"][:2] == (65.0, 5.0)
    assert table.at[NAME, "IQL-DT"][0] == 85.0
    assert table.at[NAME, "DT (papier)"][0] == pytest.approx(82.7)
    md = (tmp_path / "out" / "table.md").read_text()
    assert "**85.0 ± 5.0**" in md
    assert (tmp_path / "out" / "table.tex").read_text().startswith("\\begin{tabular}")


def test_method_datasets(tmp_path):
    exp = load_experiment(None, datasets="mujoco", methods=["dt", "iql_dt"], seeds=[0],
                          method_datasets={"iql_dt": [NAME]}, output_dir=str(tmp_path))
    jobs = plan_jobs(exp)
    dt = [j for j in jobs if j.name.startswith("dt/")]
    iql_dt = [j for j in jobs if j.name.startswith("iql_dt/")]
    assert len(dt) == 9 and [j.name for j in iql_dt] == [f"iql_dt/{NAME}/s0"]
    with pytest.raises(KeyError):
        plan_jobs(load_experiment(None, methods=["dt"], datasets=[NAME], method_datasets={"qdt": [NAME]}))


def test_estimate(tmp_path, monkeypatch):
    import launcher
    from src.utils.config import Config

    exp = load_experiment(Path(__file__).parents[1] / "experiments/configs/smoke.yaml", output_dir=str(tmp_path))
    jobs = plan_jobs(exp)
    monkeypatch.setattr(launcher, "load_experiment_config", lambda: Config({"throughput": {}}))
    assert "benchmark_device" in launcher.estimate(jobs)
    monkeypatch.setattr(launcher, "load_experiment_config",
                        lambda: Config({"throughput": {"dt": 100.0, "iql": 500.0, "cql": 100.0}}))
    dt_job = next(j for j in jobs if j.name.startswith("dt/"))
    assert launcher.job_hours(dt_job, {"dt": 100.0}) == pytest.approx(200 / 100 / 3600 + 0.15)  # smoke : 200 pas
    assert "avec --jobs 4" in launcher.estimate(jobs, workers=4)


def test_parallel_jobs_split_cpu_threads(tmp_path, monkeypatch):
    monkeypatch.delenv("DTIQL_THREADS", raising=False)
    code = "import os; open(r'{done}', 'w').write(os.environ.get('DTIQL_THREADS', ''))"
    jobs = [_job(tmp_path, f"j{i}", code) for i in range(2)]
    run_jobs(jobs, tmp_path, workers=2)
    expected = str(max(1, (__import__("os").cpu_count() or 1) // 2))
    assert all(j.done_file.read_text() == expected for j in jobs)
    single = _job(tmp_path, "solo", code)
    run_jobs([single], tmp_path, workers=1)
    assert single.done_file.read_text() == ""            # un seul job : threads par défaut
