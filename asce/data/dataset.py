"""Labeled keystroke windows from recorded sessions."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from asce.data.schema import KeyEvent, LabeledExample, Session
from asce.stages.features.logmel import LogMelExtractor
from asce.stages.segment.onset import OnsetSegmenter
from asce.types import AudioBuffer, Segment


def slice_window(session: Session, t_center: float, pre_sec: float, post_sec: float) -> np.ndarray | None:
    audio_t0 = float(session.meta.get("audio_t0", 0.0))
    sample_rate = session.sample_rate
    start = int(round((t_center - pre_sec - audio_t0) * sample_rate))
    end = int(round((t_center + post_sec - audio_t0) * sample_rate))
    if end <= 0 or start >= len(session.audio):
        return None
    left = max(0, start)
    right = min(len(session.audio), end)
    piece = session.audio[left:right]
    if start < 0:
        piece = np.pad(piece, (-start, 0))
    if end > len(session.audio):
        piece = np.pad(piece, (0, end - len(session.audio)))
    if len(piece) == 0:
        return None
    return piece.astype(np.float32)


def align_onsets_to_events(
    onset_times: list[float],
    events: list[KeyEvent],
    tolerance: float = 0.08,
) -> list[tuple[float, KeyEvent]]:
    """Greedy nearest-neighbor match. Unmatched events are dropped."""

    used: set[int] = set()
    pairs: list[tuple[float, KeyEvent]] = []
    for event in events:
        best_index = None
        best_dt = tolerance
        for index, onset in enumerate(onset_times):
            if index in used:
                continue
            delta = abs(onset - event.t_press)
            if delta <= best_dt:
                best_index = index
                best_dt = delta
        if best_index is not None:
            used.add(best_index)
            pairs.append((onset_times[best_index], event))
    return pairs


def examples_from_sessions(
    sessions: list[Session],
    pre_sec: float = 0.03,
    post_sec: float = 0.17,
    label_mode: str = "event_window",
    onset_tolerance: float = 0.08,
) -> list[LabeledExample]:
    """Build labeled clips.

    ``event_window`` cuts a fixed window around each key-logger timestamp.
    ``onset_align`` detects onsets and keeps only those that match an event,
    which is how farmed audio would be checked against partial labels.
    """

    examples: list[LabeledExample] = []
    segmenter = OnsetSegmenter()
    for session in sessions:
        meta = session.meta
        keyboard_id = str(meta.get("keyboard_id", "unknown"))
        mic_id = str(meta.get("mic_id", "unknown"))
        noise = str(meta.get("noise", "unknown"))
        distance = meta.get("distance_cm")
        if label_mode == "event_window":
            cuts = [(event.t_press, event) for event in session.events]
        elif label_mode == "onset_align":
            audio = AudioBuffer(session.audio, session.sample_rate, t0=float(meta.get("audio_t0", 0.0)))
            cuts = align_onsets_to_events(segmenter.onsets(audio), session.events, onset_tolerance)
        else:
            raise ValueError(f"unknown label_mode '{label_mode}'")
        for t_center, event in cuts:
            samples = slice_window(session, t_center, pre_sec, post_sec)
            if samples is None:
                continue
            examples.append(
                LabeledExample(
                    samples=samples,
                    sample_rate=session.sample_rate,
                    key=event.key,
                    keyboard_id=keyboard_id,
                    mic_id=mic_id,
                    session_id=session.session_id,
                    t_start=t_center - pre_sec,
                    noise=noise,
                    distance_cm=None if distance is None else float(distance),
                )
            )
    return examples


def collate_keystrokes(batch):
    features, labels, domains = zip(*batch)
    if features[0].ndim == 1:
        lengths = [item.shape[0] for item in features]
        width = max(lengths)
        stacked = torch.zeros(len(features), width, dtype=torch.float32)
        for index, item in enumerate(features):
            stacked[index, : item.shape[0]] = item
        features_out = stacked
    else:
        features_out = torch.stack(list(features), dim=0)
    return (
        features_out,
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(domains, dtype=torch.long),
    )


class KeystrokeDataset(Dataset):
    def __init__(
        self,
        examples: list[LabeledExample],
        representation: str = "mel",
        extractor=None,
        augment=None,
        label_to_index: dict[str, int] | None = None,
        keyboard_to_index: dict[str, int] | None = None,
    ) -> None:
        if representation not in ("mel", "mfcc", "wave"):
            raise ValueError(f"unknown representation '{representation}'")
        self.examples = list(examples)
        self.representation = representation
        self.extractor = extractor if extractor is not None else LogMelExtractor()
        self.augment = augment
        labels = sorted({example.key for example in self.examples})
        keyboards = sorted({example.keyboard_id for example in self.examples})
        self.label_to_index = label_to_index or {label: index for index, label in enumerate(labels)}
        self.keyboard_to_index = keyboard_to_index or {name: index for index, name in enumerate(keyboards)}
        unknown = sorted({ex.key for ex in self.examples if ex.key not in self.label_to_index})
        if unknown:
            raise KeyError(f"examples contain labels missing from label_to_index: {unknown}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int):
        example = self.examples[index]
        samples = example.samples
        if self.augment is not None:
            samples = self.augment(samples, example.sample_rate)
        if self.representation == "wave":
            feature = torch.from_numpy(np.asarray(samples, dtype=np.float32))
        else:
            segment = Segment(samples, example.sample_rate, t_start=0.0, t_end=len(samples) / example.sample_rate)
            spec = np.asarray(self.extractor.extract(segment), dtype=np.float32)
            feature = torch.from_numpy(spec).unsqueeze(0)
        return (
            feature,
            self.label_to_index[example.key],
            self.keyboard_to_index[example.keyboard_id],
        )
