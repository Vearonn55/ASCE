"""Stage contracts.

Every acoustic side-channel task uses the same skeleton. A task plugin swaps
implementations; it does not grow a second pipeline.

    AudioSource -> Preprocessor -> EventSegmenter -> FeatureExtractor -> Model -> Decoder
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

from asce.types import AudioBuffer, DecodeResult, Prediction, Segment


class AudioSource(ABC):
    """Capture. Microphone, file, or audio extracted from farmed video."""

    @abstractmethod
    def read(self) -> AudioBuffer:
        """Return the whole capture."""

    def stream(self, chunk_sec: float = 0.25) -> Iterator[AudioBuffer]:
        """Yield successive chunks. The default slices ``read()``."""

        audio = self.read()
        hop = max(1, int(chunk_sec * audio.sample_rate))
        cursor = 0
        while cursor < len(audio.samples):
            piece = audio.samples[cursor : cursor + hop]
            t0 = audio.t0 + cursor / float(audio.sample_rate)
            yield AudioBuffer(piece, audio.sample_rate, t0=t0)
            cursor += hop


class Preprocessor(ABC):
    """Denoise, resample, normalize. Shared by keytap and continuous-hum tasks."""

    @abstractmethod
    def process(self, audio: AudioBuffer) -> AudioBuffer:
        pass


class EventSegmenter(ABC):
    """Isolate discrete events. Optional in spirit: continuous tasks use a window segmenter instead of onsets."""

    @abstractmethod
    def segment(self, audio: AudioBuffer) -> list[Segment]:
        pass


class FeatureExtractor(ABC):
    """Spectrogram, MFCC, or a learned embedding."""

    @abstractmethod
    def extract(self, segment: Segment) -> object:
        pass


class Model(ABC):
    """Classifier or sequence model. ``predict`` is the inference contract; training uses the module forward."""

    @abstractmethod
    def predict(self, features: object) -> Prediction:
        pass


class Decoder(ABC):
    """Raw predictions to task output: keys for keytap, instructions for CPU hum."""

    @abstractmethod
    def decode(self, predictions: list[Prediction]) -> DecodeResult:
        pass
