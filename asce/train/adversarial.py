"""Gradient reversal so a keyboard discriminator cannot be the cue the encoder uses."""

from __future__ import annotations

import torch
from torch import nn


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features: torch.Tensor, lambd: torch.Tensor):
        ctx.save_for_backward(lambd)
        return features.view_as(features)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (lambd,) = ctx.saved_tensors
        return -lambd * grad_output, None


def grad_reverse(features: torch.Tensor, lambd: float) -> torch.Tensor:
    scale = torch.tensor(float(lambd), device=features.device, dtype=features.dtype)
    return _GradReverse.apply(features, scale)


class KeyboardDiscriminator(nn.Module):
    """Predict which keyboard (or mic) a segment came from."""

    def __init__(self, input_dim: int, num_domains: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_domains),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)
