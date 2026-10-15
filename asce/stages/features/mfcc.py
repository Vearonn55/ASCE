"""MFCC features. Cheaper baseline beside log-mel."""

from __future__ import annotations

import numpy as np

from asce.dsp import mfcc
from asce.registry import register
from asce.stages.base import FeatureExtractor
from asce.types import Segment


@register("features", "mfcc")
class MFCCExtractor(FeatureExtractor):
    def __init__(
        self,
        n_mfcc: int = 20,
        n_mels: int = 64,
        n_fft: int = 512,
        hop_length: int = 128,
        n_frames: int = 32,
    ) -> None:
        self.n_mfcc = int(n_mfcc)
        self.n_mels = int(n_mels)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.n_frames = int(n_frames)

    def extract(self, segment: Segment) -> np.ndarray:
        return mfcc(
            segment.samples,
            segment.sample_rate,
            n_mfcc=self.n_mfcc,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_frames=self.n_frames,
        )
