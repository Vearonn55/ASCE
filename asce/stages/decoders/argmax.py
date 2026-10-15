"""Take the argmax label. No language model."""

from __future__ import annotations

from asce.keys import key_to_char
from asce.registry import register
from asce.stages.base import Decoder
from asce.types import DecodeResult, Prediction


@register("decoder", "argmax")
class ArgmaxDecoder(Decoder):
    def decode(self, predictions: list[Prediction]) -> DecodeResult:
        text = "".join(key_to_char(pred.label) for pred in predictions)
        return DecodeResult(text=text, predictions=list(predictions), raw_predictions=list(predictions))
