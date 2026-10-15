"""Small CNN on a single-channel spectrogram. The from-scratch baseline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from asce.keys import VOCAB
from asce.registry import register
from asce.types import Prediction


def _as_nchw(features: object) -> torch.Tensor:
    tensor = torch.as_tensor(np.asarray(features), dtype=torch.float32)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim != 4:
        raise ValueError(f"spectrogram features must be 2d, 3d, or 4d, got {tuple(tensor.shape)}")
    return tensor


@register("model", "spectrogram_cnn")
class SpectrogramCNN(nn.Module):
    def __init__(
        self,
        num_classes: int | None = None,
        labels: list[str] | None = None,
        dropout: float = 0.0,
        checkpoint: str | None = None,
    ) -> None:
        super().__init__()
        self.labels = list(labels) if labels is not None else list(VOCAB)
        if num_classes is None:
            num_classes = len(self.labels)
        if num_classes != len(self.labels):
            raise ValueError(f"num_classes={num_classes} does not match {len(self.labels)} labels")
        # Pool time, keep frequency bands. A full spatial collapse hides which
        # mel bin rang, and that bin is the cue that separates keys.
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(9, 3), padding=(4, 1)),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=(9, 3), padding=(4, 1)),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 1)),
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(64, num_classes)
        if checkpoint:
            self.load_checkpoint(checkpoint)

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        return self.encoder(features).flatten(1)

    def forward(self, features: torch.Tensor, return_features: bool = False):
        embedding = self.encode(features)
        logits = self.head(self.dropout(embedding))
        if return_features:
            return logits, embedding
        return logits

    def predict(self, features: object) -> Prediction:
        was_training = self.training
        self.eval()
        device = next(self.parameters()).device
        batch = _as_nchw(features).to(device)
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
