"""Figures publication-ready (PDF vectoriel à texte éditable + PNG 300 dpi).

Conventions communes à toutes les figures :
  * une couleur ET une forme de marqueur fixes par méthode, quel que soit le sous-ensemble tracé
    (palette catégorielle validée daltonisme : écart ΔE >= 9 entre voisines) ;
  * moyenne sur les seeds en trait plein, bande ou barre = ± 1 écart-type sur les seeds (ddof=0) ;
  * références (méthodes non affectées par le paramètre étudié) en tirets ;
  * grille en filets clairs, axes discrets, texte en gris/noir (jamais dans la couleur d'une série).
Chaque figure a son tableau équivalent (aggregate.py / run_ablations.py) : les valeurs exactes
ne dépendent jamais de la lecture des couleurs.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from launcher import REPO  # noqa: E402  (ajoute aussi la racine du dépôt au sys.path)
from result_utils import dataset_order, mean_std, method_label, seed_average, short_dataset  # noqa: E402,F401
from src.utils.config import load_yaml  # noqa: E402

# Palette catégorielle (ordre fixe, validé : scripts/validate_palette.js de la skill dataviz)
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "D", "^", "v", "P", "X", "*"]
METHOD_SLOT = {"dt": 0, "iql_dt": 1, "iql": 2, "qdt": 3, "iql_qdt": 4, "cql": 5}
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781",
       "grid": "#e1e0d9", "axis": "#c3c2b7"}
SURFACE = "#ffffff"
SEQ_CMAP = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
REF_DASH = (0, (4, 3))
SCORE_LABEL = "Score normalisé D4RL"
_extra_slots: dict[str, int] = {}


# --------------------------------------------------------------------------------------
# Style
# --------------------------------------------------------------------------------------
def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
        "legend.fontsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "text.color": INK["primary"], "axes.labelcolor": INK["secondary"], "axes.titlecolor": INK["primary"],
        "xtick.color": INK["secondary"], "ytick.color": INK["secondary"],
        "axes.edgecolor": INK["axis"], "axes.linewidth": 0.8, "axes.spines.top": False,
        "axes.spines.right": False, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
        "axes.grid": True, "grid.color": INK["grid"], "grid.linewidth": 0.6, "grid.linestyle": "-",
        "axes.axisbelow": True, "xtick.major.size": 0, "ytick.major.size": 0,
        "lines.linewidth": 1.8, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "legend.frameon": False, "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def method_style(method: str) -> tuple[str, str]:
    """(couleur, marqueur) fixes d'une méthode ; 'iql_dt__alpha0.5' a le style de 'iql_dt'."""
    base = method.split("__")[0]
    if base in METHOD_SLOT:
        i = METHOD_SLOT[base]
    else:
        i = _extra_slots.setdefault(base, len(METHOD_SLOT) + len(_extra_slots))
    if i >= len(SLOTS):  # au-delà de 8 méthodes : gris, jamais de teinte générée
        return INK["muted"], "."
    return SLOTS[i], MARKERS[i]


def save(fig, out: str | Path, formats=("pdf", "png")) -> list[Path]:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in formats:
        p = Path(f"{out}.{ext}")
        fig.savefig(p, facecolor=SURFACE)
        paths.append(p)
    plt.close(fig)
    return paths


def _legend_handle(method: str, label: str | None = None, reference: bool = False) -> Line2D:
    color, marker = method_style(method)
    return Line2D([], [], color=color, lw=1.6 if reference else 1.8, ls=REF_DASH if reference else "-",
                  marker=None if reference else marker, ms=5.5, mec=SURFACE, mew=1.0,
                  label=(label or method_label(method)) + (" (réf.)" if reference else ""))


def _top_legend(fig, handles, max_cols: int = 5) -> None:
    """Légende au-dessus de la figure, lue ligne par ligne (matplotlib remplit par colonnes)."""
    n = len(handles)
    ncol = n if n <= max_cols else math.ceil(n / math.ceil(n / max_cols))
    nrows = math.ceil(n / ncol)
    blank = Line2D([], [], ls="none", label=" ")
    grid = [[handles[r * ncol + c] if r * ncol + c < n else blank for c in range(ncol)] for r in range(nrows)]
    ordered = [grid[r][c] for c in range(ncol) for r in range(nrows)]
    fig.legend(handles=ordered, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=ncol,
               handlelength=2.4, columnspacing=1.4)


# --------------------------------------------------------------------------------------
# 1. Comparaison principale : points + barres d'erreur, un rang par dataset
# --------------------------------------------------------------------------------------
def plot_comparison(df: pd.DataFrame, out: str | Path, methods: list[str] | None = None,
                    datasets: list[str] | None = None, paper: dict[str, float] | None = None) -> list[Path]:
    """df : une ligne par run (method, dataset, seed, score)."""
    setup_style()
    methods = methods or [m for m in METHOD_SLOT if m in set(df["method"])] + sorted(
        set(df["method"]) - set(METHOD_SLOT))
    df = df[df["method"].isin(methods)]
    datasets = dataset_order(datasets or df["dataset"].unique())
    rows = datasets + ["Moyenne"]
    stats = mean_std(pd.concat([df, seed_average(df, datasets, ["method"])]), ["method", "dataset"])
    stats = stats.set_index(["method", "dataset"])

    n = len(methods)
    offsets = np.linspace(-0.27, 0.27, n) if n > 1 else np.zeros(1)
    fig, ax = plt.subplots(figsize=(6.6, 0.55 + 0.36 * len(rows) * max(1.0, n / 4)))
    lo, hi = 0.0, 0.0
    for i, m in enumerate(methods):
        color, marker = method_style(m)
        for j, ds in enumerate(rows):
            if (m, ds) not in stats.index:
                continue
            s = stats.loc[(m, ds)]
            ax.errorbar(s["mean"], j + offsets[i], xerr=s["std"], fmt=marker, color=color, ms=5.5,
                        mec=SURFACE, mew=1.0, elinewidth=1.6, capsize=0, zorder=3)
            lo, hi = min(lo, s["mean"] - s["std"]), max(hi, s["mean"] + s["std"])
    handles = [_legend_handle(m) for m in methods]
    if paper:
        ys = [(paper[d], j) for j, d in enumerate(rows) if paper.get(d) is not None]
        for p, j in ys:  # repère vertical sur toute la hauteur du rang, derrière les points
            ax.plot([p, p], [j - 0.42, j + 0.42], color=INK["primary"], lw=1.3, zorder=2,
                    solid_capstyle="butt")
        if ys:
            hi = max(hi, max(p for p, _ in ys))
            handles.append(Line2D([], [], ls="none", marker="|", ms=10, mew=1.3, color=INK["primary"],
                                  label="DT (papier)"))
    ax.set_yticks(range(len(rows)), [short_dataset(d) for d in rows])
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.axhline(len(datasets) - 0.5, color=INK["axis"], lw=0.8)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(min(0.0, lo * 1.05), hi * 1.05 + 1)
    ax.set_xlabel(f"{SCORE_LABEL} (moyenne ± écart-type sur les seeds)")
    ax.tick_params(axis="y", colors=INK["primary"])
    fig.tight_layout()
    _top_legend(fig, handles, max_cols=4)
    return save(fig, out)


# --------------------------------------------------------------------------------------
# 2. Courbes d'apprentissage (méthodes DT) ; baselines en référence horizontale
# --------------------------------------------------------------------------------------
def load_curves(runs_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(courbes DT : method, dataset, seed, step, score ; scores finaux des autres méthodes)."""
    curves, finals = [], []
    for res in sorted(Path(runs_dir).glob("*/*/seed*/results.json")):
        r = json.loads(res.read_text())
        if "score" not in r:
            continue
        base = {"method": r["method"], "dataset": r["dataset"], "seed": r["seed"]}
        metrics = res.parent / "metrics.jsonl"
        if "best_target_return" in r and metrics.exists():
            key = f"score_target{r['best_target_return']:g}"
            logged = [json.loads(line) for line in metrics.read_text().splitlines() if line.strip()]
            pts = [(row["step"], row[f"eval/{key}"]) for row in logged if f"eval/{key}" in row]
            final_step = next((row["step"] for row in logged if f"final/{key}" in row),
                              max((row["step"] for row in logged), default=0))
            pts.append((final_step, r["score"]))  # dernier point : évaluation finale (100 épisodes)
            curves += [{**base, "step": s, "score": v} for s, v in pts]
        else:
            finals.append({**base, "score": r["score"]})
    return (pd.DataFrame(curves, columns=["method", "dataset", "seed", "step", "score"]),
            pd.DataFrame(finals, columns=["method", "dataset", "seed", "score"]))


def _grid(n_panels: int, ncols: int = 3, panel_w: float = 2.2, panel_h: float = 1.75):
    ncols = min(ncols, n_panels)
    nrows = math.ceil(n_panels / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_w * ncols, panel_h * nrows), squeeze=False)
    for ax in axes.flat[n_panels:]:
        ax.set_visible(False)
    return fig, axes.flat[:n_panels], ncols


def plot_learning_curves(runs_dir: str | Path, out: str | Path, methods: list[str] | None = None,
                         datasets: list[str] | None = None) -> list[Path] | None:
    setup_style()
    curves, finals = load_curves(runs_dir)
    if curves.empty:
        return None
    if methods:
        curves, finals = curves[curves["method"].isin(methods)], finals[finals["method"].isin(methods)]
    datasets = dataset_order(datasets or curves["dataset"].unique())
    dt_methods = [m for m in METHOD_SLOT if m in set(curves["method"])] + sorted(
        set(curves["method"]) - set(METHOD_SLOT))
    ref_methods = [m for m in METHOD_SLOT if m in set(finals["method"])] + sorted(
        set(finals["method"]) - set(METHOD_SLOT))

    fig, axes, ncols = _grid(len(datasets))
    for k, (ax, ds) in enumerate(zip(axes, datasets)):
        for m in dt_methods:
            c = curves[(curves["method"] == m) & (curves["dataset"] == ds)]
            if c.empty:
                continue
            wide = c.pivot_table(index="step", columns="seed", values="score").dropna()  # pas communs
            color, marker = method_style(m)
            mu, sd = wide.mean(axis=1), wide.std(axis=1, ddof=0)
            ax.fill_between(wide.index, mu - sd, mu + sd, color=color, alpha=0.12, lw=0)
            ax.plot(wide.index, mu, color=color, zorder=3)
            ax.plot(wide.index[-1:], mu.iloc[-1:], ls="none", marker=marker, ms=5.5, color=color,
                    mec=SURFACE, mew=1.0, zorder=4)
        for m in ref_methods:
            f = finals[(finals["method"] == m) & (finals["dataset"] == ds)]
            if not f.empty:
                ax.axhline(f["score"].mean(), color=method_style(m)[0], ls=REF_DASH, lw=1.4)
        ax.set_title(short_dataset(ds))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:g}k"))
        if k % ncols == 0:
            ax.set_ylabel(SCORE_LABEL)
    fig.supxlabel("Pas de gradient (dernier point : évaluation finale)", fontsize=8.5, color=INK["secondary"])
    handles = [_legend_handle(m) for m in dt_methods] + [_legend_handle(m, reference=True) for m in ref_methods]
    fig.tight_layout()
    _top_legend(fig, handles)
    return save(fig, out)


# --------------------------------------------------------------------------------------
# 3. Ablation : score en fonction du paramètre, un panneau par dataset + moyenne
# --------------------------------------------------------------------------------------
def plot_ablation(swept: pd.DataFrame, refs: pd.DataFrame, meta: dict, out: str | Path,
                  data_max: pd.DataFrame | None = None) -> list[Path]:
    """swept : (method, value_key, dataset, mean, std) ; refs : (method, dataset, mean, std) ;
    data_max : (value_key, dataset, score) = meilleure trajectoire du dataset (ablation sur les données)."""
    setup_style()
    keys = [str(v) for v in meta["value_keys"]]
    xs = np.arange(len(keys))
    panels = dataset_order(meta["datasets"]) + ["Moyenne"]
    fig, axes, ncols = _grid(len(panels), ncols=min(4, len(panels)), panel_w=2.0, panel_h=1.85)
    for k, (ax, ds) in enumerate(zip(axes, panels)):
        for m in meta["methods"]:
            s = swept[(swept["method"] == m) & (swept["dataset"] == ds)].set_index("value_key").reindex(keys)
            if s["mean"].isna().all():
                continue
            color, marker = method_style(m)
            ax.fill_between(xs, s["mean"] - s["std"], s["mean"] + s["std"], color=color, alpha=0.12, lw=0)
            ax.plot(xs, s["mean"], color=color, marker=marker, ms=5.5, mec=SURFACE, mew=1.0, zorder=3)
        for m in meta.get("reference_methods", []):
            r = refs[(refs["method"] == m) & (refs["dataset"] == ds)]
            if not r.empty:
                ax.axhline(r["mean"].iloc[0], color=method_style(m)[0], ls=REF_DASH, lw=1.4, zorder=2)
        if data_max is not None and not data_max.empty:
            d = data_max[data_max["dataset"] == ds].set_index("value_key").reindex(keys)["score"]
            ax.plot(xs, d, color=INK["muted"], ls=(0, (1.2, 2.0)), lw=1.6, zorder=2)
        ax.set_xticks(xs, keys)
        ax.set_xlim(-0.35, len(keys) - 0.65)
        ax.grid(axis="x", visible=False)
        ax.set_title(short_dataset(ds), fontweight="bold" if ds == "Moyenne" else "normal")
        if k % ncols == 0:
            ax.set_ylabel(SCORE_LABEL)
    fig.supxlabel(meta["label"], fontsize=8.5, color=INK["secondary"])
    handles = [_legend_handle(m) for m in meta["methods"]]
    handles += [_legend_handle(m, reference=True) for m in meta.get("reference_methods", [])]
    if data_max is not None and not data_max.empty:
        handles.append(Line2D([], [], color=INK["muted"], ls=(0, (1.2, 2.0)), lw=1.6,
                              label="Meilleure trajectoire du dataset"))
    fig.tight_layout()
    _top_legend(fig, handles)
    return save(fig, out)


# --------------------------------------------------------------------------------------
# 4. Diagnostic d'une relabellisation (rtg.npz)
# --------------------------------------------------------------------------------------
def plot_relabel_diagnostics(npz_path: str | Path, out: str | Path, title: str | None = None) -> list[Path]:
    setup_style()
    d = np.load(npz_path)
    rtg, orig, lengths = d["rtg"], d["rtg_original"], d["lengths"]
    split = lambda x: np.split(x, np.cumsum(lengths)[:-1])
    gain = rtg - orig
    relabeled = gain > 1e-6
    fig, ((a, b), (c, e)) = plt.subplots(2, 2, figsize=(6.8, 5.0))

    # (a) calibration du critique : Q(s, a) contre le return Monte-Carlo actualisé
    q = d["q"] if "q" in d.files else np.zeros(0)
    g = d["g_mc"] if "g_mc" in d.files else np.zeros(0)
    if q.size and q.size == g.size:
        hb = a.hexbin(g, q, gridsize=55, bins="log", mincnt=1, cmap=SEQ_CMAP, linewidths=0)
        lim = [min(g.min(), q.min()), max(g.max(), q.max())]
        a.plot(lim, lim, color=INK["primary"], lw=0.9)
        a.text(0.04, 0.95, f"corr = {np.corrcoef(g, q)[0, 1]:.2f}\nbiais moyen = {np.mean(q - g):+.1f}",
               transform=a.transAxes, va="top", fontsize=7.5, color=INK["secondary"])
        cb = fig.colorbar(hb, ax=a, fraction=0.046, pad=0.02)
        cb.set_label("transitions (échelle log)", color=INK["secondary"])
        cb.outline.set_visible(False)
        gamma = float(d["gamma"]) if "gamma" in d.files else None
        a.set_xlabel("Return Monte-Carlo actualisé $G^{\\gamma}_t$" + (f" (γ={gamma:g})" if gamma else ""))
        a.set_ylabel("$Q(s_t, a_t)$ du critique")
        a.set_title("Calibration du critique (droite : Q = G)")
    else:
        a.text(0.5, 0.5, "pas de critique (règle « none »)", ha="center", va="center", color=INK["muted"])
        a.set_axis_off()

    # (b) distribution des gains
    if relabeled.any():
        b.hist(gain[relabeled], bins=50, color=SLOTS[0], edgecolor=SURFACE, linewidth=0.6)
        b.set_yscale("log")
    b.set_title(f"Gain ajouté : {relabeled.mean():.1%} des pas relabellisés")
    b.set_xlabel("$\\hat R_t - R_t$ (pas relabellisés)")
    b.set_ylabel("nombre de pas")

    # (c) trajectoire dont le R̂_0 gagne le plus
    o_s, r_s = split(orig), split(rtg)
    i = int(np.argmax([r[0] - o[0] for r, o in zip(r_s, o_s)]))
    t = np.arange(len(o_s[i]))
    c.plot(t, o_s[i], color=INK["muted"], label="RTG d'origine $R_t$")
    c.plot(t, r_s[i], color=SLOTS[0], label="RTG relabellisé $\\hat R_t$")
    c.set_title(f"Trajectoire n°{i} (plus grand gain sur $\\hat R_0$)")
    c.set_xlabel("pas de temps t")
    c.set_ylabel("return-to-go")
    c.legend(loc="upper right")

    # (e) RTG initiaux avant / après
    r0_o, r0_n = np.array([x[0] for x in o_s]), np.array([x[0] for x in r_s])
    bins = np.histogram_bin_edges(np.concatenate([r0_o, r0_n]), bins="auto")
    if len(bins) > 51:  # au plus 50 classes
        bins = np.linspace(bins[0], bins[-1], 51)
    e.hist(r0_o, bins=bins, histtype="step", color=INK["muted"], lw=1.8, label="avant")
    e.hist(r0_n, bins=bins, histtype="step", color=SLOTS[0], lw=1.8, label="après")
    e.set_title("RTG initiaux $\\hat R_0$ des trajectoires")
    e.set_xlabel("$\\hat R_0$")
    e.set_ylabel("trajectoires")
    e.legend(loc="upper right")

    if title:
        fig.suptitle(title, fontsize=9.5)
    fig.tight_layout()
    return save(fig, out)


# --------------------------------------------------------------------------------------
# Figures d'une expérience complète
# --------------------------------------------------------------------------------------
def plot_experiment(runs_dir: str | Path, out_dir: str | Path | None = None, methods: list[str] | None = None,
                    relabel: bool = True) -> list[Path]:
    from aggregate import REFERENCE_FILE, collect

    runs_dir = Path(runs_dir)
    out_dir = Path(out_dir) if out_dir else REPO / "figures" / runs_dir.name
    paths: list[Path] = []
    df = collect(runs_dir)
    if df.empty:
        print(f"[plots] aucun résultat dans {runs_dir}")
        return paths
    paper = load_yaml(REFERENCE_FILE).get("DT (papier)", {}) if REFERENCE_FILE.exists() else {}
    paper = {k: v[0] for k, v in paper.items()} if (methods is None or "dt" in methods) else None
    paths += plot_comparison(df, out_dir / "comparison", methods, paper=paper)
    paths += plot_learning_curves(runs_dir, out_dir / "learning_curves", methods) or []
    if relabel:
        for npz in sorted(runs_dir.glob("relabel/*/*/seed0/rtg.npz")):
            tag, ds = npz.parent.parent.parent.name, npz.parent.parent.name
            paths += plot_relabel_diagnostics(npz, out_dir / "relabel" / f"{tag}__{ds}", f"{ds} — {tag} (seed 0)")
    print(f"[plots] {len(paths)} fichiers -> {out_dir}")
    return paths
