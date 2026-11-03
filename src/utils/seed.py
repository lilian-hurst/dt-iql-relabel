"""Reproductibilité (graines), choix du device et des threads."""
from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:  # PyTorch n'est pas requis pour les étapes données/env
        return
    torch.manual_seed(seed)  # graine aussi les générateurs CUDA et MPS
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False


def get_device(spec: str = "auto"):
    import torch

    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if spec == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("device 'mps' demandé mais indisponible (Mac Apple Silicon + PyTorch >= 2 requis)")
    return torch.device(spec)


def device_for(algo: str, exp_cfg, override: str | None = None):
    """Device d'un algorithme : override > $DTIQL_DEVICE > exp_cfg.devices[algo] > exp_cfg.device."""
    devices = exp_cfg.get("devices") or {}
    spec = override or os.environ.get("DTIQL_DEVICE") or devices.get(algo) or exp_cfg.get("device", "auto")
    return get_device(spec)


def configure_threads(exp_cfg) -> int:
    """Fixe le nombre de threads PyTorch CPU ($DTIQL_THREADS > exp_cfg.threads) ; 0 = défaut."""
    n = int(os.environ.get("DTIQL_THREADS") or exp_cfg.get("threads") or 0)
    if n > 0:
        import torch

        torch.set_num_threads(n)
    return n


def synchronize(device) -> None:
    """Attend la fin des calculs asynchrones (mesures de temps sur GPU/MPS)."""
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
