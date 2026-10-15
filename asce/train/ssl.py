"""Contrastive pretraining on unlabeled keystroke clips, then optional k-means."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from asce.audio_io import read_wav
from asce.dsp import log_mel
from asce.stages.models.cnn import SpectrogramCNN
from asce.train.augment import AudioAugmentor


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    z1 = torch.nn.functional.normalize(z1, dim=-1)
    z2 = torch.nn.functional.normalize(z2, dim=-1)
    batch = z1.shape[0]
    features = torch.cat([z1, z2], dim=0)
    similarity = features @ features.T / temperature
    similarity = similarity.masked_fill(torch.eye(2 * batch, device=features.device, dtype=torch.bool), -1e9)
    targets = torch.cat(
        [
            torch.arange(batch, device=features.device) + batch,
            torch.arange(batch, device=features.device),
        ],
        dim=0,
    )
    return torch.nn.functional.cross_entropy(similarity, targets)


def _spec_tensor(samples: np.ndarray, sample_rate: int, n_mels: int) -> torch.Tensor:
    spec = log_mel(samples, sample_rate, n_mels=n_mels)
    return torch.from_numpy(spec).unsqueeze(0)


def contrastive_pretrain(
    waves: list[tuple[np.ndarray, int]],
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 1e-3,
    device: torch.device | None = None,
    n_mels: int = 64,
    temperature: float = 0.2,
    seed: int = 0,
) -> dict:
    """Train the CNN encoder so two augmented views of one click land together."""

    if len(waves) < 2:
        raise ValueError("contrastive pretraining needs at least two clips")
    if device is None:
        device = torch.device("cpu")
    torch.manual_seed(seed)
    model = SpectrogramCNN(labels=["unused"], num_classes=1).to(device)
    # The classification head is unused; the encoder is the pretrained object.
    optimizer = torch.optim.Adam(model.encoder.parameters(), lr=lr)
    augmentor = AudioAugmentor()
    losses = []
    model.train()
    for _epoch in range(epochs):
        order = np.random.default_rng(seed + _epoch).permutation(len(waves))
        running = 0.0
        steps = 0
        for start in range(0, len(order) - batch_size + 1, batch_size):
            batch_ids = order[start : start + batch_size]
            view_a = []
            view_b = []
            for index in batch_ids:
                samples, sample_rate = waves[int(index)]
                view_a.append(_spec_tensor(augmentor(samples, sample_rate), sample_rate, n_mels))
                view_b.append(_spec_tensor(augmentor(samples, sample_rate), sample_rate, n_mels))
            left = torch.stack(view_a).to(device)
            right = torch.stack(view_b).to(device)
            optimizer.zero_grad()
            loss = nt_xent(model.encode(left), model.encode(right), temperature)
            loss.backward()
            optimizer.step()
            running += float(loss.item())
            steps += 1
        losses.append(running / max(steps, 1))
    return {"model": model, "loss": losses[-1] if losses else None, "history": losses}


def kmeans(points: np.ndarray, k: int, iters: int = 20, seed: int = 0) -> np.ndarray:
    if k > len(points):
        raise ValueError("k cannot exceed the number of points")
    rng = np.random.default_rng(seed)
    centroids = points[rng.choice(len(points), size=k, replace=False)].astype(np.float64)
    labels = np.zeros(len(points), dtype=np.int64)
    for _ in range(iters):
        distances = ((points[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=-1)
        labels = distances.argmin(axis=1)
        for index in range(k):
            chosen = points[labels == index]
            if len(chosen):
                centroids[index] = chosen.mean(axis=0)
    return labels


@torch.no_grad()
def embed_waves(model: nn.Module, waves: list[tuple[np.ndarray, int]], device: torch.device, n_mels: int = 64) -> np.ndarray:
    model.eval()
    rows = []
    for samples, sample_rate in waves:
        spec = _spec_tensor(samples, sample_rate, n_mels).unsqueeze(0).to(device)
        rows.append(model.encode(spec).squeeze(0).cpu().numpy())
    return np.stack(rows, axis=0)


def load_unlabeled_dir(root: str | Path) -> list[tuple[np.ndarray, int]]:
    """Load every wav under ``root`` (farmed clips or a flat folder)."""

    root = Path(root)
    waves = []
    for path in sorted(root.rglob("*.wav")):
        samples, sample_rate = read_wav(path)
        if len(samples):
            waves.append((samples, sample_rate))
    return waves
