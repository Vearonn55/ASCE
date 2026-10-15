"""Instruction-prior decoder. Same beam pattern as the keytap language model, different prior."""

from __future__ import annotations

import math

from asce.decoding import beam_search
from asce.registry import register
from asce.stages.base import Decoder
from asce.types import DecodeResult, Prediction

INSTRUCTIONS = ["nop", "mov", "add", "load", "store", "cmp", "jmp", "call"]

# Hand-set bigram weights: compare tends to be followed by a jump, mov by add, and so on.
_BIGRAMS: dict[tuple[str, str], float] = {
    ("<s>", "mov"): 5,
    ("<s>", "load"): 4,
    ("mov", "add"): 6,
    ("mov", "load"): 3,
    ("load", "add"): 5,
    ("load", "store"): 3,
    ("add", "store"): 4,
    ("add", "cmp"): 3,
    ("store", "mov"): 3,
    ("cmp", "jmp"): 8,
    ("cmp", "call"): 2,
    ("jmp", "mov"): 3,
    ("call", "mov"): 3,
    ("nop", "mov"): 2,
}


@register("decoder", "isa_prior")
class ISAPriorDecoder(Decoder):
    def __init__(self, prior_weight: float = 0.8, beam_size: int = 4, top_k: int = 4) -> None:
        self.prior_weight = float(prior_weight)
        self.beam_size = int(beam_size)
        self.top_k = int(top_k)
        self._denom: dict[str, float] = {}
        for (left, _right), weight in _BIGRAMS.items():
            self._denom[left] = self._denom.get(left, 0.0) + weight

    def _prior(self, _prev2: str, prev1: str, label: str) -> float:
        total = self._denom.get(prev1, 0.0)
        count = _BIGRAMS.get((prev1, label), 0.0)
        vocab = len(INSTRUCTIONS)
        probability = (count + 0.5) / (total + 0.5 * vocab)
        return self.prior_weight * math.log(probability)

    def decode(self, predictions: list[Prediction]) -> DecodeResult:
        labels = beam_search(
            predictions,
            self._prior,
            beam_size=self.beam_size,
            top_k=self.top_k,
            allowed=set(INSTRUCTIONS),
        )
        revised = [
            Prediction(
                label=label,
                confidence=float(prediction.probs.get(label, prediction.confidence)),
                probs=dict(prediction.probs),
                t_start=prediction.t_start,
                t_end=prediction.t_end,
            )
            for prediction, label in zip(predictions, labels)
        ]
        text = " ".join(item.label for item in revised)
        return DecodeResult(text=text, predictions=revised, raw_predictions=list(predictions))
