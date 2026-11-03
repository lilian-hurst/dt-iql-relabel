"""Agrège les results.json d'une expérience en tableau « moyenne ± écart-type sur les seeds ».

    python experiments/aggregate.py runs/main
    python experiments/aggregate.py runs/main --methods dt iql_dt --no-reference

Sorties dans results/<nom>/ : scores.csv (un score par run), summary.csv, table.md, table.tex.
L'écart-type est calculé sur les seeds (ddof=0), chaque seed étant le score moyen de ses épisodes
d'évaluation finale. La ligne « Moyenne » n'est remplie que pour les méthodes présentes sur tous les datasets.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.utils.config import load_yaml  # noqa: E402

REFERENCE_FILE = Path(__file__).resolve().parent / "reference_scores.yaml"


def collect(runs_dir: str | Path, include_variants: bool = False) -> pd.DataFrame:
    """Un rang par run terminé ayant un score (les critiques de relabellisation n'en ont pas).

    Les variantes d'ablation rangées dans le même dossier (méthode suffixée, ex. 'dt__context_len5')
    sont ignorées sauf include_variants=True : elles relèvent de run_ablations.py.
    """
    rows = []
    for f in sorted(Path(runs_dir).glob("*/*/seed*/results.json")):
        r = json.loads(f.read_text())
        if "score" not in r or ("__" in r["method"] and not include_variants):
            continue
        rows.append({"method": r["method"], "dataset": r["dataset"], "seed": r["seed"], "score": r["score"],
                     "context_len": r.get("context_len"), "max_return_quantile": r.get("max_return_quantile", 1.0)})
    return pd.DataFrame(rows, columns=["method", "dataset", "seed", "score", "context_len", "max_return_quantile"])


def summarize(df: pd.DataFrame, keys=("method", "dataset")) -> pd.DataFrame:
    g = df.groupby(list(keys))["score"]
    return pd.DataFrame({"mean": g.mean(), "std": g.std(ddof=0), "n": g.count()}).reset_index()


def _order(values, preferred) -> list:
    pref = [v for v in preferred if v in set(values)]
    return pref + sorted(set(values) - set(pref))


def build_table(summary: pd.DataFrame, methods: list[str] | None = None, reference: bool = True) -> pd.DataFrame:
    """Tableau (dataset x méthode) de cellules (mean, std) ; colonnes de référence en fin."""
    methods_cfg = load_yaml("methods.yaml")
    suites = load_yaml("data.yaml").suites
    ds_order = _order(summary["dataset"], [d for s in suites.values() for d in s])
    m_order = _order(summary["method"], methods or list(methods_cfg))
    if methods:
        m_order = [m for m in m_order if m in methods]
    labels = {m: methods_cfg[m].get("label", m) if m in methods_cfg else m for m in m_order}

    cols = [labels[m] for m in m_order]
    ref = {}
    if reference and REFERENCE_FILE.exists():
        ref = {col: {d: v for d, v in scores.items() if d in ds_order}
               for col, scores in load_yaml(REFERENCE_FILE).items()}
        ref = {c: v for c, v in ref.items() if v}
    table = pd.DataFrame(None, index=[*ds_order, "Moyenne"], columns=cols + list(ref), dtype=object)
    for _, r in summary.iterrows():
        if r["method"] in labels:
            table.at[r["dataset"], labels[r["method"]]] = (float(r["mean"]), float(r["std"]), int(r["n"]))
    for col, scores in ref.items():
        for d, (mean, std) in scores.items():
            table.at[d, col] = (float(mean), None if std is None else float(std), None)

    for c in table.columns:  # moyenne sur les datasets, seulement si la colonne est complète
        cells = table.loc[ds_order, c]
        if all(isinstance(v, tuple) for v in cells):
            table.at["Moyenne", c] = (float(np.mean([v[0] for v in cells])), None, None)
    return table


def _fmt(cell, bold=False, latex=False) -> str:
    if not isinstance(cell, tuple):
        return "–"
    mean, std, _ = cell
    s = f"{mean:.1f}" + ("" if std is None or np.isnan(std) else (" $\\pm$ " if latex else " ± ") + f"{std:.1f}")
    if bold:
        s = f"\\textbf{{{s}}}" if latex else f"**{s}**"
    return s


def render(table: pd.DataFrame, latex: bool = False) -> str:
    ours = [c for c in table.columns if "(papier)" not in c]
    lines = []
    for ds, row in table.iterrows():
        cells = [row[c] for c in ours]
        means = [c[0] for c in cells if isinstance(c, tuple)]
        best = max(means) if means else None
        out = [_fmt(row[c], bold=isinstance(row[c], tuple) and best is not None and row[c][0] >= best - 1e-9
                    and c in ours, latex=latex) for c in table.columns]
        lines.append((ds, out))
    if latex:
        head = " & ".join(["Dataset", *table.columns]) + r" \\"
        body = "\n".join(" & ".join([ds.replace("_", r"\_"), *cells]) + r" \\" for ds, cells in lines)
        return ("\\begin{tabular}{l" + "c" * len(table.columns) + "}\n\\toprule\n" + head + "\n\\midrule\n"
                + body + "\n\\bottomrule\n\\end{tabular}\n")
    head = "| Dataset | " + " | ".join(table.columns) + " |\n|---|" + "---|" * len(table.columns)
    return head + "\n" + "\n".join(f"| {ds} | " + " | ".join(cells) + " |" for ds, cells in lines) + "\n"


def aggregate(runs_dir: str | Path, methods: list[str] | None = None, out_dir: str | Path | None = None,
              reference: bool = True) -> pd.DataFrame | None:
    runs_dir = Path(runs_dir)
    df = collect(runs_dir)
    if df.empty:
        print(f"[aggregate] aucun résultat dans {runs_dir}")
        return None
    out = Path(out_dir) if out_dir else REPO / "results" / runs_dir.name
    out.mkdir(parents=True, exist_ok=True)
    summary = summarize(df)
    table = build_table(summary, methods, reference)
    df.to_csv(out / "scores.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    md = render(table)
    (out / "table.md").write_text(md)
    (out / "table.tex").write_text(render(table, latex=True))
    seeds = df.groupby(["method", "dataset"]).size()
    print(f"\n{md}\n[aggregate] {len(df)} runs ({seeds.min()}–{seeds.max()} seeds par case) -> {out}")
    return table


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs_dir")
    p.add_argument("--methods", nargs="+", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--no-reference", action="store_true")
    a = p.parse_args()
    aggregate(a.runs_dir, a.methods, a.out, not a.no_reference)


if __name__ == "__main__":
    main()
