"""wav2vec2 and HuBERT backbones with a small classification head.

Weights download on first construct. The class itself imports cleanly without
Hugging Face installed; construction is what requires ``transformers``.
MPS is preferred when it passes a tiny matmul check, otherwise CPU.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from asce.keys import VOCAB
from asce.registry import register
from asce.train.device import resolve_device
from asce.types import Prediction

_SPECS = {
    "wav2vec2": ("Wav2Vec2Model", "facebook/wav2vec2-base"),
    "hubert": ("HubertModel", "facebook/hubert-base-ls960"),
}


class HFEncoderClassifier(nn.Module):
    def __init__(
        self,
        kind: str,
        model_name: str | None = None,
        num_classes: int | None = None,
        labels: list[str] | None = None,
        freeze: bool = True,
        unfreeze_last_n: int = 0,
        device: str = "auto",
        checkpoint: str | None = None,
        encoder: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if kind not in _SPECS:
            raise ValueError(f"unknown backbone '{kind}'")
        class_name, default_name = _SPECS[kind]
        self.kind = kind
        self.labels = list(labels) if labels is not None else list(VOCAB)
        if num_classes is None:
            num_classes = len(self.labels)
        if num_classes != len(self.labels):
            raise ValueError(f"num_classes={num_classes} does not match {len(self.labels)} labels")
        name = model_name or default_name
        if encoder is None:
            try:
                from transformers import AutoModel
            except ImportError as exc:
                raise ImportError("transformers is required for wav2vec2/HuBERT backbones") from exc
            self.encoder = AutoModel.from_pretrained(name)
        else:
            self.encoder = encoder
        # ``class_name`` documents the intended architecture; AutoModel covers both.
        del class_name
        hidden = int(self.encoder.config.hidden_size)
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
        if unfreeze_last_n > 0:
            layers = _encoder_layers(self.encoder)
            for layer in layers[-unfreeze_last_n:]:
                for param in layer.parameters():
                    param.requires_grad = True
        self.head = nn.Linear(hidden, num_classes)
        self.to(resolve_device(device))
        if checkpoint:
            self.load_checkpoint(checkpoint)

    def trainable_parameters(self):
        return (param for param in self.parameters() if param.requires_grad)

    def forward(self, waveforms: torch.Tensor, return_features: bool = False, attention_mask: torch.Tensor | None = None):
        if waveforms.ndim == 3:
            waveforms = waveforms.squeeze(1)
        output = self.encoder(waveforms, attention_mask=attention_mask)
        hidden = output.last_hidden_state
        if attention_mask is None:
            embedding = hidden.mean(dim=1)
        else:
            # Mask is at waveform rate; approximate by interpolating to frames.
            frame_mask = torch.nn.functional.interpolate(
                attention_mask.unsqueeze(1).float(),
                size=hidden.shape[1],
                mode="nearest",
            ).squeeze(1)
            weights = frame_mask.unsqueeze(-1)
            embedding = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
        logits = self.head(embedding)
        if return_features:
            return logits, embedding
        return logits

    def predict(self, features: object) -> Prediction:
        was_training = self.training
        self.eval()
        device = next(self.parameters()).device
        tensor = torch.as_tensor(np.asarray(features), dtype=torch.float32)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        tensor = tensor.to(device)
        with torch.no_grad():
            logits = self.forward(tensor)
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


def _encoder_layers(encoder: nn.Module) -> list[nn.Module]:
    base = getattr(encoder, "encoder", None)
    layers = getattr(base, "layers", None) if base is not None else None
    if layers is None:
        return []
    return list(layers)


@register("model", "wav2vec2")
class Wav2Vec2Backbone(HFEncoderClassifier):
    def __init__(self, **kwargs) -> None:
        super().__init__(kind="wav2vec2", **kwargs)


@register("model", "hubert")
class HuBERTBackbone(HFEncoderClassifier):
    def __init__(self, **kwargs) -> None:
        super().__init__(kind="hubert", **kwargs)
