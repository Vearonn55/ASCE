"""Audio augmentation: room impulse responses, noise, pitch and time jitter."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AugmentConfig:
    rir_probability: float = 0.5
    rt60_range: tuple[float, float] = (0.05, 0.3)
    snr_db_range: tuple[float, float] = (5.0, 30.0)
    noise_probability: float = 0.7
    pitch_ratio_range: tuple[float, float] = (0.97, 1.03)
    time_shift_sec: float = 0.01
    seed: int | None = None


def synthetic_rir(sample_rate: int, rt60: float, rng: np.random.Generator) -> np.ndarray:
    """Exponential-decay noise burst. Stand-in for a measured room impulse response."""

    length = max(8, int(rt60 * sample_rate))
    times = np.arange(length, dtype=np.float32) / float(sample_rate)
    decay = np.exp(-times * (6.91 / max(rt60, 1e-3))).astype(np.float32)
    impulse = rng.normal(0.0, 1.0, length).astype(np.float32) * decay
    impulse[0] = 1.0
    energy = float(np.sqrt(np.sum(impulse * impulse)) + 1e-8)
    return impulse / energy


def convolve_rir(samples: np.ndarray, rir: np.ndarray) -> np.ndarray:
    if len(samples) == 0 or len(rir) == 0:
        return np.asarray(samples, dtype=np.float32)
    wet = np.convolve(samples.astype(np.float32), rir.astype(np.float32))
    return wet[: len(samples)].astype(np.float32)


def mix_snr(samples: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    signal = np.asarray(samples, dtype=np.float32)
    if len(noise) < len(signal):
        reps = int(np.ceil(len(signal) / max(len(noise), 1)))
        noise = np.tile(noise, reps)
    noise = np.asarray(noise[: len(signal)], dtype=np.float32)
    power_s = float(np.mean(signal * signal) + 1e-12)
    power_n = float(np.mean(noise * noise) + 1e-12)
    target = power_s / (10 ** (snr_db / 10.0))
    return signal + noise * np.sqrt(target / power_n)


def pitch_jitter(samples: np.ndarray, ratio: float) -> np.ndarray:
    """Resample by ``ratio`` and fit back to the original length."""

    signal = np.asarray(samples, dtype=np.float32).reshape(-1)
    if len(signal) < 2 or abs(ratio - 1.0) < 1e-3:
        return signal
    stretched = max(2, int(round(len(signal) / ratio)))
    old = np.linspace(0.0, 1.0, len(signal), endpoint=False)
    new = np.linspace(0.0, 1.0, stretched, endpoint=False)
    shifted = np.interp(new, old, signal).astype(np.float32)
    if len(shifted) >= len(signal):
        start = (len(shifted) - len(signal)) // 2
        return shifted[start : start + len(signal)]
    pad = len(signal) - len(shifted)
    return np.pad(shifted, (pad // 2, pad - pad // 2))


def time_shift(samples: np.ndarray, sample_rate: int, shift_sec: float) -> np.ndarray:
    shift = int(round(shift_sec * sample_rate))
    if shift == 0 or len(samples) == 0:
        return np.asarray(samples, dtype=np.float32)
    out = np.zeros_like(samples, dtype=np.float32)
    if shift > 0:
        out[shift:] = samples[:-shift]
    else:
        out[:shift] = samples[-shift:]
    return out


class AudioAugmentor:
    def __init__(self, config: AugmentConfig | None = None, rirs: list[np.ndarray] | None = None) -> None:
        self.config = config or AugmentConfig()
        self.rirs = rirs or []
        self.rng = np.random.default_rng(self.config.seed)

    def __call__(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        cfg = self.config
        out = np.asarray(samples, dtype=np.float32).copy()
        ratio = float(self.rng.uniform(*cfg.pitch_ratio_range))
        out = pitch_jitter(out, ratio)
        shift = float(self.rng.uniform(-cfg.time_shift_sec, cfg.time_shift_sec))
        out = time_shift(out, sample_rate, shift)
        if self.rng.random() < cfg.rir_probability:
            if self.rirs:
                rir = self.rirs[int(self.rng.integers(0, len(self.rirs)))]
            else:
                rt60 = float(self.rng.uniform(*cfg.rt60_range))
                rir = synthetic_rir(sample_rate, rt60, self.rng)
            out = convolve_rir(out, rir)
        if self.rng.random() < cfg.noise_probability:
            noise = self.rng.normal(0.0, 1.0, len(out)).astype(np.float32)
            snr = float(self.rng.uniform(*cfg.snr_db_range))
            out = mix_snr(out, noise, snr)
        peak = float(np.max(np.abs(out)) + 1e-8)
        return (out / peak).astype(np.float32)
