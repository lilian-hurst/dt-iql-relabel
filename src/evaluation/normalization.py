"""Score normalisé D4RL : 0 = politique aléatoire, 100 = expert."""
from __future__ import annotations

import numpy as np

from src.data.registry import DatasetSpec, get_spec


def normalized_score(returns, spec: DatasetSpec | str):
    """Accepte un scalaire ou un tableau de returns bruts ; renvoie le(s) score(s) normalisé(s)."""
    if isinstance(spec, str):
        spec = get_spec(spec)
    returns = np.asarray(returns, dtype=np.float64)
    out = 100.0 * (returns - spec.ref_min_score) / (spec.ref_max_score - spec.ref_min_score)
    return float(out) if out.ndim == 0 else out


def summarize(scores) -> dict:
    """Moyenne et écart-type (ddof=0, convention des papiers D4RL) d'une liste de scores."""
    scores = np.asarray(scores, dtype=np.float64)
    return {"mean": float(scores.mean()), "std": float(scores.std()), "n": int(scores.size)}
