"""Utilitaires communs aux tableaux et aux figures (sans dépendance graphique)."""
from __future__ import annotations

import pandas as pd

from launcher import REPO  # noqa: F401  (ajoute la racine du dépôt au sys.path)
from src.utils.config import load_yaml


def method_label(method: str) -> str:
    """Libellé d'une méthode (config/methods.yaml) ; 'iql_dt__alpha0.5' -> libellé de 'iql_dt'."""
    cfg = load_yaml("methods.yaml")
    base = method.split("__")[0]
    return cfg[base].get("label", base) if base in cfg else base


def short_dataset(ds: str) -> str:
    """'hopper-medium-replay-v2' -> 'hopper · medium-replay'."""
    if ds == "Moyenne":
        return ds
    name = ds.rsplit("-v", 1)[0]
    domain, _, quality = name.partition("-")
    return f"{domain} · {quality}" if quality else domain


def dataset_order(datasets) -> list[str]:
    """Ordre des suites de config/data.yaml, puis les datasets inconnus triés."""
    suites = load_yaml("data.yaml").suites
    known = [d for s in suites.values() for d in s]
    present = list(dict.fromkeys(datasets))
    return [d for d in known if d in present] + sorted(d for d in present if d not in known)


def seed_average(df: pd.DataFrame, datasets: list[str], keys: list[str]) -> pd.DataFrame:
    """Score moyen sur les datasets, par seed (seulement si la seed couvre tous les datasets).

    L'écart-type de la ligne « Moyenne » est ensuite calculé sur ces moyennes par seed.
    """
    rows = []
    for k, g in df[df["dataset"].isin(datasets)].groupby([*keys, "seed"], dropna=False):
        if set(datasets) <= set(g["dataset"]):
            k = k if isinstance(k, tuple) else (k,)
            rows.append({**dict(zip([*keys, "seed"], k)), "dataset": "Moyenne", "score": g["score"].mean()})
    return pd.DataFrame(rows, columns=[*keys, "seed", "dataset", "score"])


def mean_std(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Moyenne, écart-type (ddof=0) et nombre de seeds par groupe."""
    g = df.groupby(keys, dropna=False)["score"]
    return pd.DataFrame({"mean": g.mean(), "std": g.std(ddof=0), "n": g.count()}).reset_index()
