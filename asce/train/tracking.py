"""Experiment tracking. Weights & Biases or MLflow when requested, JSONL always."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ExperimentTracker:
    """Logs metrics to ``experiments/*.jsonl``.

    ``backend="wandb"`` or ``"mlflow"`` also mirrors to that service when the
    package is installed. ``auto`` (the default) uses W&B only when
    ``WANDB_API_KEY`` is set, so a laptop run never blocks on a login prompt.
    """

    def __init__(
        self,
        backend: str = "auto",
        project: str = "asce",
        run_name: str | None = None,
        out_dir: str | Path = "experiments",
        config: dict | None = None,
    ) -> None:
        self.project = project
        self.run_name = run_name or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.out_dir / f"{self.run_name}.jsonl"
        self._wandb = None
        self._mlflow = None
        self.backend = self._open(backend, config or {})

    def _open(self, backend: str, config: dict) -> str:
        chosen = backend
        if backend == "auto":
            chosen = "wandb" if os.environ.get("WANDB_API_KEY") else "jsonl"
        if chosen == "wandb":
            try:
                import wandb

                wandb.init(project=self.project, name=self.run_name, config=config, reinit=True)
                self._wandb = wandb
            except Exception as exc:
                self.path.write_text("")
                self.log({"tracker_warning": f"wandb unavailable: {exc}"})
                return "jsonl"
        elif chosen == "mlflow":
            try:
                import mlflow

                mlflow.set_experiment(self.project)
                mlflow.start_run(run_name=self.run_name)
                if config:
                    mlflow.log_params({k: v for k, v in config.items() if _flat(v)})
                self._mlflow = mlflow
            except Exception as exc:
                self.log({"tracker_warning": f"mlflow unavailable: {exc}"})
                return "jsonl"
        self.log({"event": "start", "backend": chosen, "config": config})
        return chosen

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        row = dict(metrics)
        if step is not None:
            row["step"] = step
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, default=str) + "\n")
        if self._wandb is not None:
            payload = {k: v for k, v in row.items() if isinstance(v, (int, float))}
            if payload:
                self._wandb.log(payload, step=step)
        if self._mlflow is not None:
            numeric = {k: v for k, v in row.items() if isinstance(v, (int, float))}
            if numeric:
                self._mlflow.log_metrics(numeric, step=step or 0)

    def finish(self) -> None:
        self.log({"event": "finish"})
        if self._wandb is not None:
            self._wandb.finish()
            self._wandb = None
        if self._mlflow is not None:
            self._mlflow.end_run()
            self._mlflow = None


def _flat(value: object) -> bool:
    return isinstance(value, (str, int, float, bool))
