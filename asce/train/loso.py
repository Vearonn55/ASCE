"""Leave-one-keyboard-out evaluation.

Train on every keyboard but one, test on the held-out keyboard. The gap
between in-distribution validation accuracy and held-out accuracy is the
number that says whether the model generalized.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from asce.data.schema import LabeledExample
from asce.stages.models.cnn import SpectrogramCNN
from asce.train.train import evaluate, fit, make_loader


def group_by_keyboard(examples: list[LabeledExample]) -> dict[str, list[LabeledExample]]:
    groups: dict[str, list[LabeledExample]] = defaultdict(list)
    for example in examples:
        groups[example.keyboard_id].append(example)
    return dict(groups)


def loso_folds(examples: list[LabeledExample]):
    groups = group_by_keyboard(examples)
    if len(groups) < 2:
        raise ValueError("leave-one-keyboard-out needs at least two keyboards")
    for held, test_examples in groups.items():
        train_examples = [item for name, items in groups.items() if name != held for item in items]
        yield held, train_examples, test_examples


def random_split(examples: list[LabeledExample], val_fraction: float, seed: int):
    if len(examples) < 2 or val_fraction <= 0:
        return list(examples), []
    rng = np.random.default_rng(seed)
    order = np.arange(len(examples))
    rng.shuffle(order)
    n_val = max(1, int(round(len(examples) * val_fraction)))
    n_val = min(n_val, len(examples) - 1)
    val_index = set(order[:n_val].tolist())
    train, val = [], []
    for index, example in enumerate(examples):
        (val if index in val_index else train).append(example)
    return train, val


def _model_for(labels: list[str], model_name: str, device_prefer: str):
    if model_name == "spectrogram_cnn":
        return SpectrogramCNN(labels=labels)
    if model_name in ("wav2vec2", "hubert"):
        from asce.stages.models.hf_backbone import HuBERTBackbone, Wav2Vec2Backbone

        cls = Wav2Vec2Backbone if model_name == "wav2vec2" else HuBERTBackbone
        return cls(labels=labels, freeze=True, device=device_prefer)
    raise ValueError(f"unknown model '{model_name}'")


def run_loso(
    examples: list[LabeledExample],
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 1e-2,
    seed: int = 0,
    device: torch.device | None = None,
    model_name: str = "spectrogram_cnn",
    representation: str = "mel",
    val_fraction: float = 0.2,
    augment=None,
    out_path: str | Path | None = None,
) -> dict:
    if device is None:
        from asce.train.device import resolve_device

        device = resolve_device("auto")
    folds = []
    for held, train_examples, test_examples in loso_folds(examples):
        train_part, val_part = random_split(train_examples, val_fraction, seed)
        labels = sorted({item.key for item in train_examples + test_examples})
        keyboards = sorted({item.keyboard_id for item in train_examples + test_examples})
        label_to_index = {label: index for index, label in enumerate(labels)}
        keyboard_to_index = {name: index for index, name in enumerate(keyboards)}
        from asce.data.dataset import KeystrokeDataset

        common = dict(
            representation=representation,
            label_to_index=label_to_index,
            keyboard_to_index=keyboard_to_index,
        )
        train_ds = KeystrokeDataset(train_part, augment=augment, **common)
        val_ds = KeystrokeDataset(val_part, **common) if val_part else None
        test_ds = KeystrokeDataset(test_examples, **common)
        model = _model_for(labels, model_name, str(device))
        train_loader = make_loader(train_ds, batch_size, shuffle=True)
        val_loader = make_loader(val_ds, batch_size, shuffle=False) if val_ds is not None else None
        fit(model, train_loader, val_loader, device, epochs=epochs, lr=lr)
        in_dist = evaluate(model, val_loader, device)["top1"] if val_loader is not None else None
        held_out = evaluate(model, make_loader(test_ds, batch_size, shuffle=False), device)["top1"]
        gap = None if in_dist is None else in_dist - held_out
        folds.append(
            {
                "held_out_keyboard": held,
                "n_train": len(train_part),
                "n_val": len(val_part),
                "n_test": len(test_examples),
                "in_dist_top1": in_dist,
                "held_out_top1": held_out,
                "gap": gap,
            }
        )
    held_scores = [fold["held_out_top1"] for fold in folds]
    dist_scores = [fold["in_dist_top1"] for fold in folds if fold["in_dist_top1"] is not None]
    summary = {
        "model": model_name,
        "folds": folds,
        "mean_held_out_top1": float(np.mean(held_scores)) if held_scores else None,
        "mean_in_dist_top1": float(np.mean(dist_scores)) if dist_scores else None,
    }
    if summary["mean_in_dist_top1"] is not None and summary["mean_held_out_top1"] is not None:
        summary["mean_gap"] = summary["mean_in_dist_top1"] - summary["mean_held_out_top1"]
    else:
        summary["mean_gap"] = None
    if out_path is not None:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def format_loso(summary: dict) -> str:
    lines = [
        f"model: {summary['model']}",
        f"mean in-dist top1: {_pct(summary.get('mean_in_dist_top1'))}",
        f"mean held-out top1: {_pct(summary.get('mean_held_out_top1'))}",
        f"mean generalization gap: {_pct(summary.get('mean_gap'))}",
    ]
    for fold in summary["folds"]:
        lines.append(
            f"  hold out {fold['held_out_keyboard']}: "
            f"in-dist {_pct(fold['in_dist_top1'])} held-out {_pct(fold['held_out_top1'])} "
            f"gap {_pct(fold['gap'])} (n_test={fold['n_test']})"
        )
    return "\n".join(lines)


def _pct(value) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * float(value):.1f}%"
