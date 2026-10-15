"""Train on synthetic sessions and test on a real validation directory.

If accuracy on the real set is not clearly above chance, the synthetic data
is teaching the wrong thing and should not be scaled up.
"""

from __future__ import annotations

import torch

from asce.data.dataset import KeystrokeDataset, examples_from_sessions
from asce.data.sessions import load_sessions
from asce.stages.models.cnn import SpectrogramCNN
from asce.train.train import evaluate, fit, make_loader


def transfer_gate(
    synth_dir: str,
    real_dir: str,
    epochs: int = 4,
    batch_size: int = 16,
    lr: float = 1e-3,
    min_ratio_above_chance: float = 2.0,
    device: torch.device | None = None,
    seed: int = 0,
) -> dict:
    if device is None:
        device = torch.device("cpu")
    torch.manual_seed(seed)
    synth_sessions = load_sessions(synth_dir)
    real_sessions = load_sessions(real_dir)
    if not synth_sessions:
        return {"transfers": False, "reason": "no synthetic sessions", "real_accuracy": None, "chance": None}
    if not real_sessions:
        return {"transfers": False, "reason": "no real validation sessions", "real_accuracy": None, "chance": None}
    synth_examples = examples_from_sessions(synth_sessions)
    real_examples = examples_from_sessions(real_sessions)
    labels = sorted({example.key for example in synth_examples})
    real_examples = [example for example in real_examples if example.key in labels]
    if not synth_examples or not real_examples:
        return {
            "transfers": False,
            "reason": "no overlapping labeled examples",
            "real_accuracy": None,
            "chance": 1.0 / max(len(labels), 1),
        }
    label_to_index = {label: index for index, label in enumerate(labels)}
    keyboards = sorted({ex.keyboard_id for ex in synth_examples + real_examples})
    keyboard_to_index = {name: index for index, name in enumerate(keyboards)}
    train_ds = KeystrokeDataset(
        synth_examples,
        label_to_index=label_to_index,
        keyboard_to_index=keyboard_to_index,
    )
    real_ds = KeystrokeDataset(
        real_examples,
        label_to_index=label_to_index,
        keyboard_to_index=keyboard_to_index,
    )
    model = SpectrogramCNN(labels=labels)
    # The real sessions are the test, not the model-selection set.
    fit(
        model,
        make_loader(train_ds, batch_size, shuffle=True),
        None,
        device,
        epochs=epochs,
        lr=lr,
    )
    accuracy = evaluate(model, make_loader(real_ds, batch_size, shuffle=False), device)["top1"]
    chance = 1.0 / len(labels)
    transfers = accuracy >= chance * min_ratio_above_chance
    return {
        "transfers": bool(transfers),
        "reason": "ok" if transfers else "below transfer threshold",
        "real_accuracy": accuracy,
        "chance": chance,
        "n_synth": len(synth_examples),
        "n_real": len(real_examples),
        "n_labels": len(labels),
    }
