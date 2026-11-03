"""Logger minimal : toujours un fichier metrics.jsonl (lisible avec pandas.read_json(lines=True)),
plus TensorBoard ou Weights & Biases si demandés et installés."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path


class Logger:
    def __init__(self, out_dir: str | Path, backend: str = "tensorboard", project: str = "",
                 run_name: str = "", config: dict | None = None):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._jsonl = open(self.out_dir / "metrics.jsonl", "a")
        self._tb = self._wandb = None
        self._t0 = time.time()
        if backend == "tensorboard":
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._tb = SummaryWriter(str(self.out_dir / "tb"))
            except ImportError:
                print("[logger] tensorboard absent : métriques en JSONL uniquement", file=sys.stderr)
        elif backend == "wandb":
            try:
                import wandb
                self._wandb = wandb.init(project=project, name=run_name, config=config, dir=str(self.out_dir))
            except ImportError:
                print("[logger] wandb absent : métriques en JSONL uniquement", file=sys.stderr)

    def log(self, metrics: dict, step: int) -> None:
        row = {"step": int(step), "time": round(time.time() - self._t0, 1), **metrics}
        self._jsonl.write(json.dumps(row) + "\n")
        self._jsonl.flush()
        scalars = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        if self._tb:
            for k, v in scalars.items():
                self._tb.add_scalar(k, v, step)
        if self._wandb:
            self._wandb.log(scalars, step=step)

    def print(self, metrics: dict, step: int) -> None:
        msg = " | ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items())
        print(f"[{step:>7d}] {msg}", flush=True)

    def close(self) -> None:
        self._jsonl.close()
        if self._tb:
            self._tb.close()
        if self._wandb:
            self._wandb.finish()
