"""Ré-évalue un checkpoint DT, éventuellement avec d'autres returns cibles.

    python -m src.evaluation.eval_dt runs/dt/hopper-medium-v2/seed0/model.pt --targets 3600 1800 --episodes 100
"""
from __future__ import annotations

import argparse
import json

import torch

from src.evaluation.rollout_dt import evaluate_dt
from src.models.decision_transformer import build_dt
from src.utils.config import Config
from src.utils.seed import get_device


def load_dt(path: str, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_dt(Config(ckpt["model_cfg"]), ckpt["obs_dim"], ckpt["act_dim"]).to(device)
    model.load_state_dict(ckpt["model"])
    return model.eval(), ckpt


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--targets", type=float, nargs="+", required=True)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=10_000)
    p.add_argument("--device", default="auto")
    a = p.parse_args()
    device = get_device(a.device)
    model, ck = load_dt(a.checkpoint, device)
    for target in a.targets:
        r = evaluate_dt(model, ck["dataset"], target, ck["return_scale"], ck["state_mean"], ck["state_std"],
                        episodes=a.episodes, seed=a.seed, device=device)
        r.pop("returns")
        print(json.dumps(r))


if __name__ == "__main__":
    main()
