"""Linear / MLP head over a fixed vector. Used by band-energy and embedding features."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from asce.registry import register
from asce.types import Prediction


def _as_batch(features: object) -> torch.Tensor:
    tensor = torch.as_tensor(np.asarray(features), dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim != 2:
        raise ValueError(f"vector features must be 1d or 2d, got {tuple(tensor.shape)}")
    return tensor


@register("model", "mlp")
class MLPClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        labels: list[str] | None = None,
        hidden: int = 64,
        checkpoint: str | None = None,
    ) -> None:
        super().__init__()
        if labels is None:
            labels = [str(i) for i in range(num_classes)]
        if len(labels) != num_classes:
            raise ValueError("labels length must equal num_classes")
        self.labels = list(labels)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )
        if checkpoint:
            self.load_checkpoint(checkpoint)

    def forward(self, features: torch.Tensor, return_features: bool = False):
        hidden = self.net[0](features)
        activated = self.net[1](hidden)
        logits = self.net[2](activated)
        if return_features:
            return logits, activated
        return logits

    def predict(self, features: object) -> Prediction:
        was_training = self.training
        self.eval()
        device = next(self.parameters()).device
        batch = _as_batch(features).to(device)
        with torch.no_grad():
            logits = self.forward(batch)
            probabilities = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()
        if was_training:
            self.train()
        index = int(np.argmax(probabilities))
        probs = {self.labels[i]: float(probabilities[i]) for i in range(len(self.labels))}
        return Prediction(label=self.labels[index], confidence=probs[self.labels[index]], probs=probs)

    def load_checkpoint(self, path: str | Path) -> None:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        state = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
        self.load_state_dict(state)
        if isinstance(blob, dict) and blob.get("labels"):
            self.labels = list(blob["labels"])
