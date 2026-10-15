"""Log-spaced band energy. A continuous-signal feature for CPU hum, not keytaps."""

from __future__ import annotations

import numpy as np

from asce.registry import register
from asce.stages.base import FeatureExtractor
from asce.types import Segment

N_BANDS = 16


@register("features", "band_energy")
class BandEnergyExtractor(FeatureExtractor):
    def __init__(self, n_bands: int = N_BANDS) -> None:
        self.n_bands = int(n_bands)

    def extract(self, segment: Segment) -> np.ndarray:
        samples = np.asarray(segment.samples, dtype=np.float32).reshape(-1)
        if len(samples) < 8:
            return np.zeros(self.n_bands, dtype=np.float32)
        window = np.hanning(len(samples)).astype(np.float32)
        power = np.abs(np.fft.rfft(samples * window)) ** 2
        # Drop DC and split the rest into log-spaced bands.
        power = power[1:]
        if len(power) < self.n_bands:
            power = np.pad(power, (0, self.n_bands - len(power)))
        edges = np.unique(np.round(np.geomspace(1, len(power), self.n_bands + 1)).astype(int))
        if len(edges) < 2:
            return np.zeros(self.n_bands, dtype=np.float32)
        bands = []
        for left, right in zip(edges[:-1], edges[1:]):
            right = max(right, left + 1)
            bands.append(float(np.log(np.mean(power[left:right]) + 1e-8)))
        while len(bands) < self.n_bands:
            bands.append(bands[-1] if bands else 0.0)
        return np.asarray(bands[: self.n_bands], dtype=np.float32)
