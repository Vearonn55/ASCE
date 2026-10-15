"""Read an audio file into the pipeline."""

from __future__ import annotations

from pathlib import Path

from asce.audio_io import read_wav
from asce.registry import register
from asce.stages.base import AudioSource
from asce.types import AudioBuffer


@register("source", "file")
class FileAudioSource(AudioSource):
    def __init__(self, path: str, t0: float = 0.0) -> None:
        self.path = Path(path)
        self.t0 = float(t0)

    def read(self) -> AudioBuffer:
        samples, sample_rate = read_wav(self.path)
        return AudioBuffer(samples, sample_rate, t0=self.t0)
