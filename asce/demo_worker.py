"""JSON-lines worker for the live demo.

stdin:  {"type":"audio","sample_rate":16000,"pcm_b64":"..."}
stdout: {"type":"predictions","mode":"model"|"detect","text":"...","keys":[...]}

Logs go to stderr so the protocol stream stays clean.
"""

from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path

import numpy as np

from asce.config import load_yaml
from asce.pipeline import ChunkRunner, build_pipeline
from asce.types import Prediction

ROOT = Path(__file__).resolve().parents[1]


class DetectionOnlyModel:
    """Onset events without a trained classifier. Honest placeholder, not a guess."""

    def predict(self, features: object) -> Prediction:
        energy = float(np.mean(np.asarray(features) ** 2))
        return Prediction(label="?", confidence=energy, probs={"?": 1.0})


def _pipeline():
    config_path = Path(os.environ.get("ASCE_CONFIG", ROOT / "configs" / "keytap_baseline.yaml"))
    cfg = load_yaml(config_path)
    checkpoint = os.environ.get("ASCE_CHECKPOINT") or (cfg.get("model") or {}).get("checkpoint")
    mode = "detect"
    if checkpoint and Path(checkpoint).exists():
        import torch

        blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
        labels = blob.get("labels") if isinstance(blob, dict) else None
        if labels and labels != ["unused"]:
            cfg["model"]["labels"] = list(labels)
            cfg["model"].pop("num_classes", None)
        cfg["model"]["checkpoint"] = str(checkpoint)
        mode = "model"
    pipeline = build_pipeline(cfg)
    if mode == "detect":
        pipeline.model = DetectionOnlyModel()
    return pipeline, mode


def _emit(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def main() -> None:
    pipeline, mode = _pipeline()
    runner = ChunkRunner(pipeline, tail_sec=0.5)
    _emit({"type": "ready", "mode": mode})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            _emit({"type": "error", "message": "invalid json"})
            continue
        if message.get("type") == "reset":
            runner = ChunkRunner(pipeline, tail_sec=0.5)
            _emit({"type": "ready", "mode": mode})
            continue
        if message.get("type") != "audio":
            continue
        raw = base64.b64decode(message["pcm_b64"])
        samples = np.frombuffer(raw, dtype=np.float32).copy()
        result = runner.push(samples, int(message.get("sample_rate", 16000)))
        if result is None or not result.predictions:
            continue
        _emit(
            {
                "type": "predictions",
                "mode": mode,
                "text": result.text,
                "keys": [
                    {
                        "key": pred.label,
                        "confidence": pred.confidence,
                        "t_start": pred.t_start,
                        "t_end": pred.t_end,
                    }
                    for pred in result.predictions
                ],
            }
        )


if __name__ == "__main__":
    main()
