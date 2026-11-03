"""Ablations (Phase 3) : longueur de contexte, force / seuil de relabellisation, γ du critique,
qualité du dataset. Définitions dans experiments/configs/ablations.yaml.

    python experiments/run_ablations.py --dry-run                          # plan de toutes les ablations
    python experiments/run_ablations.py --only context_len --jobs 4 --gpus 0 1
    python experiments/run_ablations.py --only relabel_strength q_threshold --seeds 0 1 2
    python experiments/run_ablations.py --report-only                      # tableaux + figures seuls
    python experiments/run_ablations.py --config experiments/configs/ablations_smoke.yaml  # test rapide

Réutilisation : les runs à la configuration par défaut (ex. K=20, α=1, seuil=0, quantile=1) sont
communs à toutes les ablations (runs/<nom>/shared/), de même que les critiques et
relabellisations quand le paramètre ne les affecte pas.
Sorties (<nom> = clé `name` du fichier, « ablations » par défaut) : runs/<nom>/manifests/<ablation>.json,
results/<nom>/<ablation>/{scores.csv, summary.csv, table.md, table.tex}, figures/<nom>/<ablation>.{pdf,png}.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from launcher import REPO, describe_jobs, load_experiment, plan_jobs, run_jobs  # noqa: E402
from result_utils import dataset_order, mean_std, method_label, seed_average  # noqa: E402
from src.utils.config import Config, load_config, load_yaml  # noqa: E402

DEFAULT_CONFIG = HERE / "configs" / "ablations.yaml"
PARAMS = ("dt_overrides", "iql_overrides", "cql_overrides", "relabel_overrides", "critic_overrides",
          "data_overrides")
SHARED_RELABEL_KEYS = {"alpha", "min_improvement", "rule", "value_source"}  # inclus dans le dossier relabel/
SHARED = "shared"
_MISSING = object()


def fmt_value(v) -> str:
    return f"{v:g}" if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v)


def _same(a, b) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)
    return a == b


def is_shared(param: str, key: str) -> bool:
    """Le paramètre laisse-t-il intacts données, critiques et (sauf alpha/seuil) relabellisations ?"""
    return param in ("dt_overrides", "iql_overrides", "cql_overrides") or (
        param == "relabel_overrides" and key in SHARED_RELABEL_KEYS)


def _check_affected(param: str, method: str, mc) -> None:
    ok = {"dt_overrides": mc.train == "dt", "iql_overrides": mc.train == "iql",
          "cql_overrides": mc.train == "cql", "relabel_overrides": "relabel" in mc,
          "critic_overrides": "relabel" in mc, "data_overrides": True}[param]
    if not ok:
        raise ValueError(f"{param} n'affecte pas la méthode {method!r} : la mettre dans reference_methods")


def default_value(param: str, key: str, method: str, base: Config):
    """Valeur de `key` pour `method` dans la configuration de base (fichiers YAML + surcharges communes)."""
    mc = load_yaml("methods.yaml")[method]
    r_over = [f"{k}={v}" for k, v in (mc.get("relabel") or {}).items()] + list(base.relabel_overrides)
    if param == "relabel_overrides":
        cfg = load_config("relabel.yaml", r_over)
    elif param == "critic_overrides":
        rcfg = load_config("relabel.yaml", r_over)
        critic = rcfg.critics[rcfg.value_source]
        cfg = load_config(f"{rcfg.value_source}.yaml", list(critic.overrides) + list(base.critic_overrides))
    else:
        file = {"dt_overrides": "dt.yaml", "iql_overrides": "iql.yaml", "cql_overrides": "cql.yaml",
                "data_overrides": "data.yaml"}[param]
        cfg = load_config(file, base[param])
    value = cfg.get_path(key, _MISSING)
    if value is _MISSING:
        raise KeyError(f"clé inconnue {key!r} pour {param}")
    return value


def load_ablations(path: str | Path = DEFAULT_CONFIG, only=None, datasets=None, seeds=None):
    cfg = load_yaml(Path(path).resolve())
    defaults = cfg.defaults
    defaults.name = cfg.get("name", "ablations")  # dossier runs/<name>/ propre à ce fichier d'ablations
    if datasets:
        defaults.datasets = datasets
    if seeds:
        defaults.seeds = seeds
    for k in defaults.get("overrides") or {}:
        if k not in PARAMS:
            raise KeyError(f"defaults.overrides : clé inconnue {k!r} ; choix : {PARAMS}")
    ablations = cfg.ablations
    if only:
        missing = set(only) - set(ablations)
        if missing:
            raise KeyError(f"ablations inconnues : {sorted(missing)} ; choix : {list(ablations)}")
        ablations = Config({k: v for k, v in ablations.items() if k in only})
    for name, ab in ablations.items():
        if ab.param not in PARAMS:
            raise KeyError(f"{name} : param {ab.param!r} invalide ; choix : {PARAMS}")
    return defaults, ablations


def _experiment(defaults: Config, root: str, method: str, extra: tuple[str, str] | None, output_dir):
    overrides = {k: list(v) for k, v in (defaults.get("overrides") or {}).items()}
    if extra:
        overrides[extra[0]] = overrides.get(extra[0], []) + [extra[1]]
    return load_experiment(None, name=f"{defaults.name}/{root}", datasets=list(defaults.datasets),
                           seeds=list(defaults.seeds), methods=[method], output_dir=output_dir, **overrides)


def plan_ablation(name: str, ab: Config, defaults: Config, output_dir=None):
    """Renvoie (manifeste, liste de (expérience, suffixe))."""
    methods_cfg = load_yaml("methods.yaml")
    base = _experiment(defaults, SHARED, "dt", None, output_dir)
    short = ab.key.split(".")[-1]
    entries, refs, exps = [], [], []
    for v in ab["values"]:
        for m in ab.methods:
            _check_affected(ab.param, m, methods_cfg[m])
            if _same(v, default_value(ab.param, ab.key, m, base)):
                root, suffix, extra = SHARED, "", None
            elif is_shared(ab.param, ab.key):
                root, suffix, extra = SHARED, f"__{short}{fmt_value(v)}", (ab.param, f"{ab.key}={v}")
            else:
                root, suffix, extra = f"{name}/{short}={fmt_value(v)}", "", (ab.param, f"{ab.key}={v}")
            exps.append((_experiment(defaults, root, m, extra, output_dir), suffix))
            entries.append({"value": v, "value_key": fmt_value(v), "method": m, "dir": f"{root}/{m}{suffix}"})
    for m in ab.get("reference_methods") or []:
        exps.append((_experiment(defaults, SHARED, m, None, output_dir), ""))
        refs.append({"method": m, "dir": f"{SHARED}/{m}"})
    manifest = {
        "ablation": name, "suite": defaults.name, "label": ab.get("label", name), "param": ab.param,
        "key": ab.key,
        "values": list(ab["values"]), "value_keys": [fmt_value(v) for v in ab["values"]],
        "methods": list(ab.methods), "reference_methods": list(ab.get("reference_methods") or []),
        "datasets": list(defaults.datasets), "seeds": list(defaults.seeds),
        "entries": entries, "references": refs,
    }
    return manifest, exps


def ablations_root(output_dir=None, name: str = "ablations") -> Path:
    base = Path(output_dir or load_yaml("experiment.yaml").output_dir)
    return (base if base.is_absolute() else REPO / base) / name


# --------------------------------------------------------------------------------------
# Résultats
# --------------------------------------------------------------------------------------
def collect_ablation(manifest: dict, root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(runs balayés : value_key, method, dataset, seed, score, dataset_max_score ; runs de référence)."""
    def rows(entry, extra):
        out = []
        for f in sorted((root / entry["dir"]).glob("*/seed*/results.json")):
            r = json.loads(f.read_text())
            if "score" in r and r["dataset"] in manifest["datasets"] and r["seed"] in manifest["seeds"]:
                out.append({**extra, "method": entry["method"], "dataset": r["dataset"], "seed": r["seed"],
                            "score": r["score"], "dataset_max_score": r.get("dataset_max_score")})
        return out

    swept = [x for e in manifest["entries"] for x in rows(e, {"value_key": e["value_key"]})]
    refs = [x for e in manifest["references"] for x in rows(e, {})]
    cols = ["method", "dataset", "seed", "score", "dataset_max_score"]
    return pd.DataFrame(swept, columns=["value_key", *cols]), pd.DataFrame(refs, columns=cols)


def _render_table(swept_stats: pd.DataFrame, ref_stats: pd.DataFrame, manifest: dict, latex=False) -> str:
    pm = " $\\pm$ " if latex else " ± "
    fmt = lambda s: "–" if s is None else f"{s['mean']:.1f}{pm}{s['std']:.1f}"
    cols = [method_label(m) for m in manifest["methods"]] + [
        method_label(m) + " (réf.)" for m in manifest["reference_methods"]]
    lines = []
    for ds in dataset_order(manifest["datasets"]) + ["Moyenne"]:
        for vk in manifest["value_keys"]:
            cells = []
            for m in manifest["methods"]:
                s = swept_stats[(swept_stats.method == m) & (swept_stats.dataset == ds) & (swept_stats.value_key == vk)]
                cells.append(fmt(None if s.empty else s.iloc[0]))
            for m in manifest["reference_methods"]:
                s = ref_stats[(ref_stats.method == m) & (ref_stats.dataset == ds)]
                cells.append(fmt(None if s.empty else s.iloc[0]))
            lines.append((ds, vk, cells))
    key = manifest["key"].split(".")[-1]
    if latex:
        head = " & ".join(["Dataset", key.replace("_", r"\_"), *cols]) + r" \\"
        body = "\n".join(" & ".join([d.replace("_", r"\_"), v, *c]) + r" \\" for d, v, c in lines)
        return ("\\begin{tabular}{ll" + "c" * len(cols) + "}\n\\toprule\n" + head + "\n\\midrule\n" + body
                + "\n\\bottomrule\n\\end{tabular}\n")
    head = f"| Dataset | {key} | " + " | ".join(cols) + " |\n|---|---|" + "---|" * len(cols)
    return head + "\n" + "\n".join(f"| {d} | {v} | " + " | ".join(c) + " |" for d, v, c in lines) + "\n"


def report(manifest_path: str | Path, results_dir: Path | None = None, figures_dir: Path | None = None,
           plot: bool = True) -> dict | None:
    """Tableaux (CSV, Markdown, LaTeX) et figure d'une ablation à partir de son manifeste."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    name = manifest["ablation"]
    swept, refs = collect_ablation(manifest, manifest_path.parent.parent)
    if swept.empty:
        print(f"[ablation] {name} : aucun résultat")
        return None
    ds = manifest["datasets"]
    swept_all = pd.concat([swept, seed_average(swept, ds, ["method", "value_key"])])
    swept_stats = mean_std(swept_all, ["method", "value_key", "dataset"])
    ref_stats = mean_std(pd.concat([refs, seed_average(refs, ds, ["method"])]), ["method", "dataset"]) \
        if not refs.empty else pd.DataFrame(columns=["method", "dataset", "mean", "std", "n"])

    suite = manifest.get("suite", "ablations")
    out = results_dir or REPO / "results" / suite / name
    out.mkdir(parents=True, exist_ok=True)
    pd.concat([swept.assign(reference=False), refs.assign(reference=True)]).to_csv(out / "scores.csv", index=False)
    swept_stats.to_csv(out / "summary.csv", index=False)
    md = _render_table(swept_stats, ref_stats, manifest)
    (out / "table.md").write_text(md)
    (out / "table.tex").write_text(_render_table(swept_stats, ref_stats, manifest, latex=True))
    print(f"\n## Ablation {name} — {manifest['label']}\n\n{md}")

    if plot:
        try:
            from plotting import plot_ablation
        except ImportError:
            print("[ablation] matplotlib absent : tableaux seuls (pip install matplotlib)")
            return manifest
        data_max = None
        if manifest["param"] == "data_overrides" and swept["dataset_max_score"].notna().any():
            dm = swept.groupby(["value_key", "dataset"])["dataset_max_score"].mean().reset_index()
            dm = dm.rename(columns={"dataset_max_score": "score"})
            avg = dm[dm.dataset.isin(ds)].groupby("value_key")["score"].mean().reset_index()
            data_max = pd.concat([dm, avg.assign(dataset="Moyenne")])
        fig_dir = figures_dir or REPO / "figures" / suite
        plot_ablation(swept_stats, ref_stats, manifest, fig_dir / name, data_max)
        print(f"[ablation] figure -> {fig_dir / name}.pdf|png")
    return manifest


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--only", nargs="+", default=None, help="sous-ensemble d'ablations")
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--gpus", nargs="+", default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--report-only", action="store_true", help="tableaux et figures sans lancer de job")
    p.add_argument("--no-plot", action="store_true")
    a = p.parse_args()

    defaults, ablations = load_ablations(a.config, a.only, a.datasets, a.seeds)
    root = ablations_root(a.output_dir, defaults.name)
    jobs, manifests = {}, []
    for name, ab in ablations.items():
        manifest, exps = plan_ablation(name, ab, defaults, a.output_dir)
        for exp, suffix in exps:
            for j in plan_jobs(exp, suffix=suffix):
                jobs.setdefault(str(j.done_file), j)   # runs communs entre valeurs et ablations
        path = root / "manifests" / f"{name}.json"
        if not a.dry_run:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        manifests.append(path)
    jobs = sorted(jobs.values(), key=lambda j: j.phase)
    for j in jobs:  # noms uniques entre dossiers (shared/, <ablation>/<valeur>/) : logs distincts
        j.name = j.done_file.parent.relative_to(root).as_posix()
    print(f"[ablations] {', '.join(ablations)} : {len(jobs)} jobs uniques ({describe_jobs(jobs)}) -> {root}")

    status = {}
    if not a.report_only:
        status = run_jobs(jobs, root, workers=a.jobs, gpus=a.gpus, force=a.force, dry_run=a.dry_run)
    if not a.dry_run:
        for path in manifests:
            report(path, plot=not a.no_plot)
    return 1 if "failed" in status.values() else 0


if __name__ == "__main__":
    sys.exit(main())
