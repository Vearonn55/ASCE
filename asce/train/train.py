"""Training loop for the keytap classifier, including optional domain-adversarial loss."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from asce.data.dataset import KeystrokeDataset, collate_keystrokes
from asce.train.adversarial import KeyboardDiscriminator, grad_reverse
from asce.train.tracking import ExperimentTracker


def set_seed(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_loader(dataset: KeystrokeDataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_keystrokes,
        num_workers=0,
    )


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    correct = 0
    top5 = 0
    total = 0
    for features, labels, _domains in loader:
        features = features.to(device)
        labels = labels.to(device)
        logits = model(features)
        pred = logits.argmax(dim=-1)
        correct += int((pred == labels).sum().item())
        k = min(5, logits.shape[-1])
        best = logits.topk(k, dim=-1).indices
        top5 += int((best == labels.unsqueeze(1)).any(dim=1).sum().item())
        total += int(labels.numel())
    denom = max(total, 1)
    return {"top1": correct / denom, "top5": top5 / denom, "n": total}


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    device: torch.device,
    epochs: int,
    lr: float,
    tracker: ExperimentTracker | None = None,
    adversarial: bool = False,
    num_domains: int = 0,
    grl_lambda: float = 0.1,
    embed_dim: int | None = None,
) -> dict:
    model.to(device)
    parameters = [param for param in model.parameters() if param.requires_grad]
    if not parameters:
        raise RuntimeError("model has no trainable parameters")
    optimizer = torch.optim.Adam(parameters, lr=lr)
    discriminator = None
    if adversarial:
        if num_domains < 2:
            raise ValueError("domain-adversarial training needs at least two keyboards")
        width = embed_dim or _infer_embed_dim(model, train_loader, device)
        discriminator = KeyboardDiscriminator(width, num_domains).to(device)
        optimizer.add_param_group({"params": list(discriminator.parameters()), "lr": lr})
    history = []
    best_state = None
    best_top1 = float("-inf")
    best_reported = None
    best_train = None
    for epoch in range(epochs):
        model.train()
        if discriminator is not None:
            discriminator.train()
        running = 0.0
        seen = 0
        correct = 0
        for features, labels, domains in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            domains = domains.to(device)
            optimizer.zero_grad()
            if discriminator is not None:
                logits, embedding = model(features, return_features=True)
                loss = torch.nn.functional.cross_entropy(logits, labels)
                loss = loss + torch.nn.functional.cross_entropy(discriminator(grad_reverse(embedding, grl_lambda)), domains)
            else:
                logits = model(features)
                loss = torch.nn.functional.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * int(labels.shape[0])
            correct += int((logits.argmax(dim=-1) == labels).sum().item())
            seen += int(labels.shape[0])
        row = {"epoch": epoch, "train_loss": running / max(seen, 1), "train_top1": correct / max(seen, 1)}
        if val_loader is not None and len(val_loader.dataset) > 0:
            metrics = evaluate(model, val_loader, device)
            row.update(metrics)
            score = metrics["top1"]
        else:
            score = -row["train_loss"]
        history.append(row)
        if tracker is not None:
            tracker.log(row, step=epoch)
        if score > best_top1:
            best_top1 = score
            best_reported = row.get("top1", score)
            best_train = row.get("train_top1")
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"top1": best_reported, "train_top1": best_train, "history": history}


def save_checkpoint(model: nn.Module, path: str | Path, labels: list[str], extra: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"state_dict": model.state_dict(), "labels": list(labels)}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_encoder_weights(model: nn.Module, path: str | Path) -> int:
    """Copy matching ``encoder.*`` tensors from a contrastive checkpoint."""

    blob = torch.load(path, map_location="cpu", weights_only=False)
    state = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
    own = model.state_dict()
    copied = 0
    for key, value in state.items():
        if key.startswith("encoder.") and key in own and tuple(own[key].shape) == tuple(value.shape):
            own[key] = value
            copied += 1
    model.load_state_dict(own)
    return copied


def _infer_embed_dim(model: nn.Module, loader: DataLoader, device: torch.device) -> int:
    features, _labels, _domains = next(iter(loader))
    features = features.to(device)
    model.eval()
    with torch.no_grad():
        _logits, embedding = model(features[:1], return_features=True)
    model.train()
    return int(embedding.shape[-1])
