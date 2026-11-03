"""CQL (point d'entrée de src/training/train_offline.py avec --algo cql).

    python -m src.training.train_cql --dataset hopper-medium-replay-v2 --seed 0
"""
from __future__ import annotations

from src.training.train_offline import main as _main, train as _train


def train(dataset: str, seed: int = 0, overrides=(), method: str = "cql", output_dir=None, data_overrides=()):
    return _train("cql", dataset, seed, overrides, method, output_dir, data_overrides)


if __name__ == "__main__":
    _main(default_algo="cql")
