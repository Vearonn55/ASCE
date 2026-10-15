"""Language-model beam decoder for keytap predictions."""

from __future__ import annotations

from asce.decoding import beam_search
from asce.keys import LINGUISTIC_KEYS, key_to_char
from asce.registry import register
from asce.stages.base import Decoder
from asce.types import DecodeResult, Prediction
from plugins.keytap.lm import CharLM


def _as_char(token: str) -> str:
    if token in ("<s>", "</s>"):
        return token
    return key_to_char(token)


@register("decoder", "ngram_beam")
class NgramBeamDecoder(Decoder):
    """Beam search over acoustic probabilities plus an English character n-gram.

    This is the keytap half of the engine's decoder seam: raw classifier
    confidence in, a character sequence out. A CPU-hum plugin swaps the prior
    for an instruction distribution without touching the rest of the pipeline.
    """

    def __init__(self, lm_weight: float = 1.5, beam_size: int = 8, top_k: int = 5) -> None:
        self.lm = CharLM()
        self.lm_weight = float(lm_weight)
        self.beam_size = int(beam_size)
        self.top_k = int(top_k)

    def _prior(self, prev2: str, prev1: str, label: str) -> float:
        return self.lm_weight * self.lm.log_prob(_as_char(prev2), _as_char(prev1), _as_char(label))

    def decode(self, predictions: list[Prediction]) -> DecodeResult:
        labels = beam_search(
            predictions,
            self._prior,
            beam_size=self.beam_size,
            top_k=self.top_k,
            allowed=set(LINGUISTIC_KEYS),
        )
        revised: list[Prediction] = []
        for prediction, label in zip(predictions, labels):
            revised.append(
                Prediction(
                    label=label,
                    confidence=float(prediction.probs.get(label, prediction.confidence)),
                    probs=dict(prediction.probs),
                    t_start=prediction.t_start,
                    t_end=prediction.t_end,
                )
            )
        text = "".join(key_to_char(item.label) for item in revised)
        return DecodeResult(text=text, predictions=revised, raw_predictions=list(predictions))
