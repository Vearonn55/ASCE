"""Log-mel spectrogram features. Primary keytap representation."""

from __future__ import annotations

import numpy as np

from asce.dsp import log_mel
from asce.registry import register
from asce.stages.base import FeatureExtractor
from asce.types import Segment


@register("features", "logmel")
class LogMelExtractor(FeatureExtractor):
    def __init__(
        self,
        n_mels: int = 128,
        n_fft: int = 512,
        hop_length: int = 128,
        n_frames: int = 32,
    ) -> None:
        self.n_mels = int(n_mels)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.n_frames = int(n_frames)

    def extract(self, segment: Segment) -> np.ndarray:
        return log_mel(
            segment.samples,
            segment.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_frames=self.n_frames,
        )
