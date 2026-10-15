"""Shared data types passed between pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class AudioBuffer:
    """Mono float32 audio with a start time on the session clock."""

    samples: np.ndarray
    sample_rate: int
    t0: float = 0.0

    def __post_init__(self) -> None:
        self.samples = np.asarray(self.samples, dtype=np.float32).reshape(-1)
        self.sample_rate = int(self.sample_rate)

    @property
    def duration(self) -> float:
        if self.sample_rate <= 0:
            return 0.0
        return float(len(self.samples)) / float(self.sample_rate)


@dataclass
class Segment:
    """A discrete slice of audio, usually one keystroke or one analysis window."""

    samples: np.ndarray
    sample_rate: int
    t_start: float
    t_end: float

    def __post_init__(self) -> None:
        self.samples = np.asarray(self.samples, dtype=np.float32).reshape(-1)


@dataclass
class Prediction:
    """One stage-model decision, with a full distribution when the model has one."""

    label: str
    confidence: float
    probs: dict[str, float] = field(default_factory=dict)
    t_start: float = 0.0
    t_end: float = 0.0


@dataclass
class DecodeResult:
    """Decoder output: revised per-event predictions plus a joined string."""

    text: str
    predictions: list[Prediction]
    raw_predictions: list[Prediction]
