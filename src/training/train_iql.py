"""Baseline IQL (point d'entrée de src/training/train_offline.py avec --algo iql).

    python -m src.training.train_iql --dataset hopper-medium-replay-v2 --seed 0
"""
from __future__ import annotations

from src.training.train_offline import load_agent, main as _main, train as _train

load_iql = load_agent


def train(dataset: str, seed: int = 0, overrides=(), method: str = "iql", output_dir=None, data_overrides=()):
    return _train("iql", dataset, seed, overrides, method, output_dir, data_overrides)


if __name__ == "__main__":
    _main(default_algo="iql")
