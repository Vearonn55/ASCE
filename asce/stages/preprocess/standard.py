"""Resample, high-pass, and peak-normalize."""

from __future__ import annotations

from asce.dsp import highpass, peak_normalize, resample_linear, to_mono
from asce.registry import register
from asce.stages.base import Preprocessor
from asce.types import AudioBuffer


@register("preprocessor", "standard")
class StandardPreprocessor(Preprocessor):
    def __init__(
        self,
        target_sample_rate: int = 16000,
        highpass_hz: float = 80.0,
        normalize: bool = True,
        noise_gate_db: float | None = None,
    ) -> None:
        self.target_sample_rate = int(target_sample_rate)
        self.highpass_hz = float(highpass_hz)
        self.normalize = bool(normalize)
        self.noise_gate_db = noise_gate_db

    def process(self, audio: AudioBuffer) -> AudioBuffer:
        samples = to_mono(audio.samples)
        samples = samples - float(np_mean(samples))
        if audio.sample_rate != self.target_sample_rate:
            samples = resample_linear(samples, audio.sample_rate, self.target_sample_rate)
        if self.highpass_hz > 0:
            samples = highpass(samples, self.highpass_hz, self.target_sample_rate)
        if self.noise_gate_db is not None and len(samples):
            peak = float(abs(samples).max()) + 1e-8
            threshold = peak * (10 ** (float(self.noise_gate_db) / 20.0))
            quiet = abs(samples) < threshold
            samples = samples.copy()
            samples[quiet] *= 0.05
        if self.normalize:
            samples = peak_normalize(samples)
        return AudioBuffer(samples, self.target_sample_rate, t0=audio.t0)


def np_mean(samples):
    import numpy as np

    if len(samples) == 0:
        return 0.0
    return float(np.mean(samples))
