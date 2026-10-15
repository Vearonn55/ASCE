"""Tiny conditional generator over spectrograms.

This exists to augment a training set, not to replace recorded audio. Generator
artifacts are a real failure mode: anything trained here still has to pass the
transfer check against the real validation sessions.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class ConditionalSpecGenerator(nn.Module):
    def __init__(self, num_classes: int, n_mels: int = 64, n_frames: int = 32, z_dim: int = 16) -> None:
        super().__init__()
        self.n_mels = n_mels
        self.n_frames = n_frames
        self.z_dim = z_dim
        self.embed = nn.Embedding(num_classes, 8)
        self.net = nn.Sequential(
            nn.Linear(z_dim + 8, 128),
            nn.ReLU(),
            nn.Linear(128, n_mels * n_frames),
        )

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        hidden = torch.cat([noise, self.embed(labels)], dim=-1)
        return self.net(hidden).view(-1, 1, self.n_mels, self.n_frames)


class SpecDiscriminator(nn.Module):
    def __init__(self, num_classes: int, n_mels: int = 64, n_frames: int = 32) -> None:
        super().__init__()
        self.embed = nn.Embedding(num_classes, 8)
        self.net = nn.Sequential(
            nn.Linear(n_mels * n_frames + 8, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, spec: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        flat = torch.cat([spec.flatten(1), self.embed(labels)], dim=-1)
        return self.net(flat).squeeze(-1)


def train_conditional_generator(
    real_specs: np.ndarray,
    labels: np.ndarray,
    steps: int = 30,
    batch_size: int = 16,
    lr: float = 2e-4,
    seed: int = 0,
    device: torch.device | None = None,
) -> ConditionalSpecGenerator:
    """A few GAN steps. ``real_specs`` is (N, n_mels, frames) or (N, 1, n_mels, frames)."""

    if device is None:
        device = torch.device("cpu")
    torch.manual_seed(seed)
    specs = np.asarray(real_specs, dtype=np.float32)
    if specs.ndim == 3:
        specs = specs[:, None, :, :]
    n_mels, n_frames = int(specs.shape[-2]), int(specs.shape[-1])
    label_arr = np.asarray(labels, dtype=np.int64)
    num_classes = int(label_arr.max()) + 1 if len(label_arr) else 1
    generator = ConditionalSpecGenerator(num_classes, n_mels, n_frames).to(device)
    discriminator = SpecDiscriminator(num_classes, n_mels, n_frames).to(device)
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    loss_fn = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(seed)
    real_t = torch.from_numpy(specs)
    label_t = torch.from_numpy(label_arr)
    count = len(real_t)
    if count == 0:
        raise ValueError("no spectrograms to fit")
    generator.train()
    for _step in range(steps):
        choice = rng.integers(0, count, size=min(batch_size, count))
        real = real_t[choice].to(device)
        y = label_t[choice].to(device)
        noise = torch.randn(len(choice), generator.z_dim, device=device)
        with torch.no_grad():
            fake = generator(noise, y)
        opt_d.zero_grad()
        loss_d = loss_fn(discriminator(real, y), torch.ones(len(choice), device=device))
        loss_d = loss_d + loss_fn(discriminator(fake, y), torch.zeros(len(choice), device=device))
        loss_d.backward()
        opt_d.step()
        opt_g.zero_grad()
        noise = torch.randn(len(choice), generator.z_dim, device=device)
        generated = generator(noise, y)
        loss_g = loss_fn(discriminator(generated, y), torch.ones(len(choice), device=device))
        loss_g.backward()
        opt_g.step()
    generator.eval()
    return generator


@torch.no_grad()
def generate_specs(generator: ConditionalSpecGenerator, labels: list[int], seed: int = 0) -> np.ndarray:
    device = next(generator.parameters()).device
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    noise = torch.randn(len(labels), generator.z_dim, generator=g).to(device)
    y = torch.tensor(labels, dtype=torch.long, device=device)
    specs = generator(noise, y)
    return specs.squeeze(1).detach().cpu().numpy()
