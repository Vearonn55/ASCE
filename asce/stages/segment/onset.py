"""Energy or librosa onset detection, one window per keystroke.

Press and release are a double transient. Peaks closer than ``min_gap_sec``
collapse into a single event so the segment contains both impacts.
"""

from __future__ import annotations

from asce.dsp import energy_onsets, librosa_onsets, slice_audio
from asce.registry import register
from asce.stages.base import EventSegmenter
from asce.types import AudioBuffer, Segment


@register("segmenter", "onset")
class OnsetSegmenter(EventSegmenter):
    def __init__(
        self,
        method: str = "energy",
        hop_length: int = 256,
        threshold_ratio: float = 3.0,
        min_gap_sec: float = 0.08,
        pre_sec: float = 0.02,
        post_sec: float = 0.18,
    ) -> None:
        self.method = method
        self.hop_length = int(hop_length)
        self.threshold_ratio = float(threshold_ratio)
        self.min_gap_sec = float(min_gap_sec)
        self.pre_sec = float(pre_sec)
        self.post_sec = float(post_sec)

    def onsets(self, audio: AudioBuffer) -> list[float]:
        if self.method == "librosa":
            times = librosa_onsets(audio.samples, audio.sample_rate, self.hop_length, self.min_gap_sec)
        elif self.method == "energy":
            times = energy_onsets(
                audio.samples,
                audio.sample_rate,
                hop_length=self.hop_length,
                threshold_ratio=self.threshold_ratio,
                min_gap_sec=self.min_gap_sec,
            )
        else:
            raise ValueError(f"unknown onset method '{self.method}'")
        return [audio.t0 + t for t in times]

    def segment(self, audio: AudioBuffer) -> list[Segment]:
        segments: list[Segment] = []
        for onset in self.onsets(audio):
            t_start = onset - self.pre_sec
            t_end = onset + self.post_sec
            samples = slice_audio(audio, t_start, t_end)
            if len(samples) == 0:
                continue
            segments.append(
                Segment(samples, audio.sample_rate, t_start=max(audio.t0, t_start), t_end=t_end)
            )
        return segments
