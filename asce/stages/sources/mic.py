"""Live microphone capture via sounddevice."""

from __future__ import annotations

from collections.abc import Iterator

from asce.registry import register
from asce.stages.base import AudioSource
from asce.types import AudioBuffer


@register("source", "mic")
class MicAudioSource(AudioSource):
    def __init__(
        self,
        seconds: float = 3.0,
        sample_rate: int = 16000,
        device: int | None = None,
    ) -> None:
        self.seconds = float(seconds)
        self.sample_rate = int(sample_rate)
        self.device = device

    def read(self) -> AudioBuffer:
        import sounddevice as sd

        frames = int(self.seconds * self.sample_rate)
        audio = sd.rec(
            frames,
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            device=self.device,
        )
        sd.wait()
        return AudioBuffer(audio.reshape(-1), self.sample_rate, t0=0.0)

    def stream(self, chunk_sec: float = 0.25) -> Iterator[AudioBuffer]:
        import sounddevice as sd

        hop = max(1, int(chunk_sec * self.sample_rate))
        cursor = 0.0
        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            device=self.device,
            blocksize=hop,
        ) as stream:
            while True:
                data, _overflow = stream.read(hop)
                yield AudioBuffer(data.reshape(-1), self.sample_rate, t0=cursor)
                cursor += len(data) / float(self.sample_rate)
