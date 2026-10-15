"""Shared beam search: acoustic probabilities plus a sequence prior."""

from __future__ import annotations

import math

from asce.types import Prediction


def sequence_accuracy(reference: list[str], hypothesis: list[str]) -> float:
    if not reference and not hypothesis:
        return 1.0
    correct = sum(1 for left, right in zip(reference, hypothesis) if left == right)
    return correct / float(max(len(reference), len(hypothesis), 1))


def top_candidates(prediction: Prediction, k: int, allowed: set[str] | None) -> list[tuple[str, float]]:
    table = prediction.probs or {prediction.label: max(prediction.confidence, 1e-6)}
    rows = [(label, float(prob)) for label, prob in table.items() if allowed is None or label in allowed]
    if not rows:
        rows = [(prediction.label, max(prediction.confidence, 1e-6))]
    rows.sort(key=lambda item: item[1], reverse=True)
    return rows[: max(1, k)]


def beam_search(
    predictions: list[Prediction],
    transition_logprob,
    beam_size: int = 8,
    top_k: int = 5,
    allowed: set[str] | None = None,
    force_above: float = 0.85,
) -> list[str]:
    """Left-to-right beam.

    ``transition_logprob(prev2, prev1, label)`` is the sequence prior in nats.
    A prediction whose argmax sits outside ``allowed`` and is confident is
    copied through unchanged (shift, enter, and other non-linguistic keys).
    """

    beam: list[tuple[tuple[str, ...], float, str, str]] = [(tuple(), 0.0, "<s>", "<s>")]
    for prediction in predictions:
        forced = None
        if allowed is not None and prediction.label not in allowed and prediction.confidence >= force_above:
            forced = prediction.label
        candidates = [(forced, 1.0)] if forced else top_candidates(prediction, top_k, allowed)
        expanded: list[tuple[tuple[str, ...], float, str, str]] = []
        for labels, score, prev2, prev1 in beam:
            for label, probability in candidates:
                acoustic = math.log(max(probability, 1e-8))
                prior = 0.0 if forced else float(transition_logprob(prev2, prev1, label))
                expanded.append((labels + (label,), score + acoustic + prior, prev1, label))
        expanded.sort(key=lambda item: item[1], reverse=True)
        beam = expanded[:beam_size]
    return list(beam[0][0]) if beam else []
