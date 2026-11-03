"""Chargement des configurations YAML avec accès par attribut et surcharges en ligne de commande.

Exemple :
    cfg = load_config("config/dt.yaml", overrides=["model.context_len=5", "train.lr=3e-4"])
    cfg.model.context_len  # -> 5
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


class Config(dict):
    """dict récursif accessible par attribut (cfg.train.lr).

    Attention : une clé homonyme d'une méthode de dict (values, items, keys, get, ...) doit être lue
    par crochets — cfg["values"] — car cfg.values renvoie la méthode dict.values.
    """

    def __init__(self, data: Mapping[str, Any] | None = None):
        super().__init__()
        for k, v in (data or {}).items():
            self[k] = _wrap(v)

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(f"clé de configuration absente : {key!r}") from e

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = _wrap(value)

    def __deepcopy__(self, memo):
        return Config(copy.deepcopy(dict(self), memo))

    def to_dict(self) -> dict:
        return {k: v.to_dict() if isinstance(v, Config) else v for k, v in self.items()}

    # --- accès par chemin pointé -------------------------------------------------
    def get_path(self, path: str, default: Any = None) -> Any:
        node: Any = self
        for part in path.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return default
            node = node[part]
        return node

    def set_path(self, path: str, value: Any, strict: bool = True) -> None:
        parts = path.split(".")
        node: Config = self
        for part in parts[:-1]:
            if part not in node:
                if strict:
                    raise KeyError(f"surcharge invalide, section inconnue : {path!r}")
                node[part] = Config()
            node = node[part]
        if strict and parts[-1] not in node:
            raise KeyError(f"surcharge invalide, clé inconnue : {path!r}")
        node[parts[-1]] = _wrap(value)


def _wrap(v: Any) -> Any:
    if isinstance(v, Config):
        return v
    if isinstance(v, Mapping):
        return Config(v)
    if isinstance(v, list):
        return [_wrap(x) for x in v]
    return v


def deep_merge(base: Mapping, update: Mapping) -> Config:
    out = Config(copy.deepcopy(dict(base)))
    for k, v in update.items():
        if k in out and isinstance(out[k], Mapping) and isinstance(v, Mapping):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = _wrap(copy.deepcopy(v))
    return out


def parse_override(text: str) -> tuple[str, Any]:
    """'train.lr=3e-4' -> ('train.lr', 0.0003). La valeur est interprétée comme du YAML."""
    if "=" not in text:
        raise ValueError(f"surcharge attendue au format clé=valeur, reçu {text!r}")
    key, raw = text.split("=", 1)
    value = yaml.safe_load(raw)
    # YAML 1.1 lit "3e-4" comme une chaîne : on tente une conversion numérique.
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            pass
    return key.strip(), value


def load_yaml(path: str | Path) -> Config:
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = CONFIG_DIR / path
    with open(path) as f:
        return Config(yaml.safe_load(f) or {})


def load_config(
    path: str | Path,
    overrides: Iterable[str] = (),
    domain: str | None = None,
    strict: bool = True,
) -> Config:
    """Charge un YAML, applique la section `per_domain[domain]` puis les surcharges CLI."""
    cfg = load_yaml(path)
    per_domain = cfg.pop("per_domain", None) or {}
    if domain is not None and domain in per_domain:
        for key, value in per_domain[domain].items():
            cfg.set_path(key, value, strict=strict)
    for ov in overrides:
        key, value = parse_override(ov)
        cfg.set_path(key, value, strict=strict)
    return cfg


def save_config(cfg: Mapping, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = cfg.to_dict() if isinstance(cfg, Config) else dict(cfg)
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
