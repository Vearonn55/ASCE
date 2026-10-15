"""Fixed windows for continuous signals such as CPU hum, where onsets are the wrong tool."""

from __future__ import annotations

from asce.registry import register
from asce.stages.base import EventSegmenter
from asce.types import AudioBuffer, Segment


@register("segmenter", "fixed_window")
class FixedWindowSegmenter(EventSegmenter):
    def __init__(self, window_sec: float = 0.5, hop_sec: float = 0.25) -> None:
        self.window_sec = float(window_sec)
        self.hop_sec = float(hop_sec)

    def segment(self, audio: AudioBuffer) -> list[Segment]:
        if self.hop_sec <= 0 or self.window_sec <= 0:
            raise ValueError("window_sec and hop_sec must be positive")
        segments: list[Segment] = []
        t = audio.t0
        end = audio.t0 + audio.duration
        while t + self.window_sec <= end + 1e-9:
            a = int(round((t - audio.t0) * audio.sample_rate))
            b = int(round((t + self.window_sec - audio.t0) * audio.sample_rate))
            b = min(len(audio.samples), b)
            piece = audio.samples[a:b]
            if len(piece):
                segments.append(Segment(piece, audio.sample_rate, t_start=t, t_end=t + self.window_sec))
            t += self.hop_sec
        return segments
