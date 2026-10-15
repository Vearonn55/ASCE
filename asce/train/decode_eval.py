"""Compare raw per-key accuracy with language-model decoding on real session order."""

from __future__ import annotations

import torch

from asce.data.schema import LabeledExample
from asce.decoding import sequence_accuracy
from asce.stages.base import Decoder
from asce.types import Segment


@torch.no_grad()
def score_sessions(
    model,
    examples: list[LabeledExample],
    decoder: Decoder,
    extractor,
    device: torch.device,
) -> dict:
    """Walk each session in time order so the decoder sees a real key sequence."""

    model.eval()
    model.to(device)
    by_session: dict[str, list[LabeledExample]] = {}
    for example in examples:
        by_session.setdefault(example.session_id, []).append(example)
    raw_correct = raw_total = decoded_correct = decoded_total = 0
    per_session = []
    for session_id, rows in by_session.items():
        rows = sorted(rows, key=lambda item: item.t_start)
        predictions = []
        reference = []
        for example in rows:
            segment = Segment(
                example.samples,
                example.sample_rate,
                t_start=example.t_start,
                t_end=example.t_start + len(example.samples) / example.sample_rate,
            )
            features = extractor.extract(segment)
            prediction = model.predict(features)
            predictions.append(prediction)
            reference.append(example.key)
        decoded = decoder.decode(predictions)
        raw_labels = [item.label for item in predictions]
        dec_labels = [item.label for item in decoded.predictions]
        raw_acc = sequence_accuracy(reference, raw_labels)
        dec_acc = sequence_accuracy(reference, dec_labels)
        raw_correct += sum(a == b for a, b in zip(reference, raw_labels))
        decoded_correct += sum(a == b for a, b in zip(reference, dec_labels))
        raw_total += len(reference)
        decoded_total += len(reference)
        per_session.append(
            {
                "session_id": session_id,
                "raw_accuracy": raw_acc,
                "decoded_accuracy": dec_acc,
                "reference": reference,
                "raw": raw_labels,
                "decoded": dec_labels,
            }
        )
    return {
        "raw_accuracy": raw_correct / max(raw_total, 1),
        "decoded_accuracy": decoded_correct / max(decoded_total, 1),
        "lift": (decoded_correct - raw_correct) / max(raw_total, 1),
        "sessions": per_session,
    }
