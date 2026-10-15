"""Assemble a config into a pipeline and run it offline or as a stream."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np

from asce.bootstrap import load_builtins
from asce.config import stage_specs
from asce.registry import build
from asce.stages.base import Decoder, EventSegmenter, FeatureExtractor, Model, Preprocessor
from asce.types import AudioBuffer, DecodeResult, Prediction, Segment


class Pipeline:
    def __init__(
        self,
        source,
        preprocessor: Preprocessor,
        segmenter: EventSegmenter,
        features: FeatureExtractor,
        model: Model,
        decoder: Decoder,
    ) -> None:
        self.source = source
        self.preprocessor = preprocessor
        self.segmenter = segmenter
        self.features = features
        self.model = model
        self.decoder = decoder

    def _predict_segment(self, segment: Segment) -> Prediction:
        features = self.features.extract(segment)
        prediction = self.model.predict(features)
        prediction.t_start = segment.t_start
        prediction.t_end = segment.t_end
        return prediction

    def run_audio(self, audio: AudioBuffer) -> DecodeResult:
        processed = self.preprocessor.process(audio)
        segments = self.segmenter.segment(processed)
        predictions = [self._predict_segment(segment) for segment in segments]
        return self.decoder.decode(predictions)

    def run(self) -> DecodeResult:
        return self.run_audio(self.source.read())

    def run_stream(self, chunk_sec: float = 0.25, tail_sec: float = 0.3) -> Iterator[DecodeResult]:
        """Push chunks, emit a decode each time new events finish.

        A tail is kept so an onset split across two chunks is not dropped.
        Events that start before the previous emit point are not repeated.
        """

        pending = np.zeros(0, dtype=np.float32)
        pending_t0 = 0.0
        sample_rate = None
        emitted_until = -1e9
        for chunk in self.source.stream(chunk_sec=chunk_sec):
            processed = self.preprocessor.process(chunk)
            if sample_rate is None:
                sample_rate = processed.sample_rate
                pending_t0 = processed.t0
            elif processed.sample_rate != sample_rate:
                raise RuntimeError("stream sample rate changed mid-flight")
            if len(pending) == 0:
                pending_t0 = processed.t0
            pending = np.concatenate([pending, processed.samples])
            audio = AudioBuffer(pending, sample_rate, t0=pending_t0)
            segments = self.segmenter.segment(audio)
            ready = [
                seg
                for seg in segments
                if seg.t_start > emitted_until + 0.05 and seg.t_end <= audio.t0 + audio.duration - 0.02
            ]
            if ready:
                predictions = [self._predict_segment(seg) for seg in ready]
                emitted_until = ready[-1].t_start
                yield self.decoder.decode(predictions)
            keep = int(tail_sec * sample_rate)
            if len(pending) > keep:
                drop = len(pending) - keep
                pending = pending[drop:]
                pending_t0 += drop / float(sample_rate)


class ChunkRunner:
    """Streaming counterpart of ``Pipeline.run`` fed one chunk at a time.

    Used by the live demo. A tail of audio is retained so an onset that
    straddles two chunks is still segmented, and already-emitted events
    are not repeated.
    """

    def __init__(self, pipeline: Pipeline, tail_sec: float = 0.5) -> None:
        self.pipeline = pipeline
        self.tail_sec = float(tail_sec)
        self.pending = np.zeros(0, dtype=np.float32)
        self.pending_t0 = 0.0
        self.sample_rate: int | None = None
        self.emitted_until = -1e9
        self._next_t0 = 0.0

    def push(self, samples: np.ndarray, sample_rate: int) -> DecodeResult | None:
        chunk = self.pipeline.preprocessor.process(AudioBuffer(samples, sample_rate, t0=self._next_t0))
        self._next_t0 = chunk.t0 + chunk.duration
        if self.sample_rate is None:
            self.sample_rate = chunk.sample_rate
            self.pending_t0 = chunk.t0
        if len(self.pending) == 0:
            self.pending_t0 = chunk.t0
        self.pending = np.concatenate([self.pending, chunk.samples])
        audio = AudioBuffer(self.pending, self.sample_rate, t0=self.pending_t0)
        segments = self.pipeline.segmenter.segment(audio)
        horizon = audio.t0 + audio.duration - 0.02
        # Onset times jitter by a frame when the buffer slides. A small refractory
        # keeps one tap from being reported again, without swallowing the next key.
        ready = [
            seg
            for seg in segments
            if seg.t_start > self.emitted_until + 0.05 and seg.t_end <= horizon
        ]
        result = None
        if ready:
            predictions = [self.pipeline._predict_segment(seg) for seg in ready]
            self.emitted_until = ready[-1].t_start
            result = self.pipeline.decoder.decode(predictions)
        keep = int(self.tail_sec * self.sample_rate)
        if len(self.pending) > keep:
            drop = len(self.pending) - keep
            self.pending = self.pending[drop:]
            self.pending_t0 += drop / float(self.sample_rate)
        return result


def build_pipeline(cfg: dict[str, Any]) -> Pipeline:
    load_builtins()
    specs = stage_specs(cfg)
    return Pipeline(
        source=build("source", specs["source"]),
        preprocessor=build("preprocessor", specs["preprocessor"]),
        segmenter=build("segmenter", specs["segmenter"]),
        features=build("features", specs["features"]),
        model=build("model", specs["model"]),
        decoder=build("decoder", specs["decoder"]),
    )
