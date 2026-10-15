"""Physics-informed keystroke synthesis.

Each key has a stable fundamental. Each keyboard profile changes decay,
brightness, and microphone cutoff. That is augmentation by convolution and
filtering, not a claim that the waveform is a full mechanical model.

A model trained only on this audio must still be checked against real
recordings (see ``asce.synth.validate``) before the data is scaled up.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from asce.data.schema import KeyEvent
from asce.data.sessions import write_session
from asce.keys import VOCAB
from asce.train.augment import convolve_rir, synthetic_rir

KEYBOARD_PROFILES: dict[str, dict] = {
    "mechanical": {"pitch": 1.00, "decay": 16.0, "noise": 0.35, "brightness": 2.1, "cutoff": 7000.0, "mic": "usb"},
    "membrane": {"pitch": 0.92, "decay": 42.0, "noise": 0.08, "brightness": 1.15, "cutoff": 3400.0, "mic": "laptop"},
    "chiclet": {"pitch": 1.08, "decay": 58.0, "noise": 0.05, "brightness": 0.85, "cutoff": 2400.0, "mic": "phone"},
    "laptop": {"pitch": 0.86, "decay": 30.0, "noise": 0.16, "brightness": 1.4, "cutoff": 4200.0, "mic": "laptop"},
}


def key_f0(key: str) -> float:
    """Map key index to a fundamental. Spacing is wide enough that a small CNN can learn it."""

    index = VOCAB.index(key) if key in VOCAB else 0
    return 400.0 + index * 120.0


def _lowpass(samples: np.ndarray, cutoff_hz: float, sample_rate: int) -> np.ndarray:
    if cutoff_hz <= 0 or len(samples) == 0:
        return samples.astype(np.float32)
    dt = 1.0 / float(sample_rate)
    rc = 1.0 / (2.0 * np.pi * float(cutoff_hz))
    alpha = dt / (rc + dt)
    out = np.empty_like(samples, dtype=np.float32)
    acc = float(samples[0])
    out[0] = acc
    for index in range(1, len(samples)):
        acc = acc + alpha * (float(samples[index]) - acc)
        out[index] = acc
    return out


def synth_stroke(
    key: str,
    sample_rate: int,
    profile: dict,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    """Return (waveform, seconds from the start of the clip to the press)."""

    f0 = key_f0(key) * float(profile["pitch"]) * float(rng.uniform(0.98, 1.02))
    pre = int(0.02 * sample_rate)
    body_n = int(0.20 * sample_rate)
    times = np.arange(body_n, dtype=np.float32) / float(sample_rate)
    decay = float(profile["decay"])
    press = np.exp(-times * decay) * np.sin(2.0 * np.pi * f0 * times)
    press = press + 0.25 * np.exp(-times * decay * 1.4) * np.sin(2.0 * np.pi * f0 * float(profile["brightness"]) * times)
    noise = rng.normal(0.0, float(profile["noise"]), body_n).astype(np.float32) * np.exp(-times * 50.0)
    release_at = int(0.08 * sample_rate)
    release = np.zeros(body_n, dtype=np.float32)
    rel_t = np.arange(body_n - release_at, dtype=np.float32) / float(sample_rate)
    release[release_at:] = 0.45 * np.exp(-rel_t * decay * 1.2) * np.sin(2.0 * np.pi * (f0 * 0.5) * rel_t)
    body = _lowpass((press + noise + release).astype(np.float32), float(profile["cutoff"]), sample_rate)
    stroke = np.concatenate([np.zeros(pre, dtype=np.float32), body])
    peak = float(np.max(np.abs(stroke)) + 1e-8)
    stroke = stroke / peak * float(rng.uniform(0.45, 0.9))
    return stroke.astype(np.float32), pre / float(sample_rate)


def _render(keys: list[str], sample_rate: int, profile: dict, rng: np.random.Generator, apply_rir: bool) -> tuple[np.ndarray, list[KeyEvent]]:
    pieces: list[np.ndarray] = []
    events: list[KeyEvent] = []
    clock = 0.0
    for key in keys:
        gap = float(rng.uniform(0.12, 0.25))
        pieces.append(np.zeros(int(gap * sample_rate), dtype=np.float32))
        clock += gap
        stroke, press_offset = synth_stroke(key, sample_rate, profile, rng)
        events.append(KeyEvent(key=key, t_press=clock + press_offset, t_release=clock + press_offset + 0.08))
        pieces.append(stroke)
        clock += len(stroke) / float(sample_rate)
    audio = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
    if apply_rir and len(audio):
        rir = synthetic_rir(sample_rate, rt60=float(rng.uniform(0.04, 0.12)), rng=rng)
        audio = convolve_rir(audio, rir)
        peak = float(np.max(np.abs(audio)) + 1e-8)
        audio = (audio / peak).astype(np.float32)
    return audio, events


def text_to_keys(text: str) -> list[str]:
    keys = []
    for char in text.lower():
        if char == " ":
            keys.append("space")
        elif char.isalpha() or char.isdigit():
            keys.append(char)
        elif char == ",":
            keys.append("comma")
        elif char == ".":
            keys.append("period")
    return keys


def synthesize_corpus(
    out_dir: str | Path,
    keyboards: list[str] | None = None,
    keys: list[str] | None = None,
    examples_per_key: int = 6,
    sample_rate: int = 16000,
    seed: int = 0,
    apply_rir: bool = False,
    phrase: str | None = None,
) -> list[Path]:
    """Write one labeled session per keyboard profile."""

    names = keyboards or list(KEYBOARD_PROFILES)
    unknown = [name for name in names if name not in KEYBOARD_PROFILES]
    if unknown:
        raise KeyError(f"unknown keyboard profiles: {unknown}")
    chosen_keys = keys or list("aeotnshr")
    bad = [key for key in chosen_keys if key not in VOCAB]
    if bad:
        raise KeyError(f"keys outside vocab: {bad}")
    rng = np.random.default_rng(seed)
    written = []
    for index, name in enumerate(names):
        profile = KEYBOARD_PROFILES[name]
        sequence: list[str] = []
        for key in chosen_keys:
            sequence.extend([key] * examples_per_key)
        if phrase:
            sequence.extend(text_to_keys(phrase))
        rng.shuffle(sequence)
        audio, events = _render(sequence, sample_rate, profile, rng, apply_rir)
        meta = {
            "session_id": f"synth-{name}-{index}",
            "keyboard_id": name,
            "mic_id": profile["mic"],
            "distance_cm": 20,
            "noise": "synthetic",
            "audio_t0": 0.0,
            "synthetic": True,
            "profile": name,
        }
        written.append(write_session(out_dir, audio, sample_rate, events, meta))
    return written
