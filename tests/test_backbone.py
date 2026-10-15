from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from asce.stages.models.hf_backbone import Wav2Vec2Backbone


class _Out:
    def __init__(self, hidden):
        self.last_hidden_state = hidden


class FakeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8)
        self.lin = nn.Linear(16, 8)

    def forward(self, waveforms, attention_mask=None):
        piece = waveforms[:, :16]
        if piece.shape[1] < 16:
            piece = torch.nn.functional.pad(piece, (0, 16 - piece.shape[1]))
        hidden = self.lin(piece).unsqueeze(1)
        return _Out(hidden)


def test_wav2vec_head_predicts_without_downloading_weights():
    model = Wav2Vec2Backbone(
        encoder=FakeEncoder(),
        labels=["a", "b"],
        num_classes=2,
        freeze=True,
        device="cpu",
    )
    prediction = model.predict(np.zeros(1600, dtype=np.float32))
    assert prediction.label in {"a", "b"}
    assert set(prediction.probs) == {"a", "b"}
    assert abs(sum(prediction.probs.values()) - 1.0) < 1e-4
    # The encoder stays frozen; the head is what training would update.
    assert all(not param.requires_grad for param in model.encoder.parameters())
    assert any(param.requires_grad for param in model.head.parameters())
