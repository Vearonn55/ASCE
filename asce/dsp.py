"""Signal helpers used by feature extractors and the onset segmenter.

librosa is used when it imports cleanly. A NumPy implementation is always
available so the engine still runs if a backend op is missing.
"""

from __future__ import annotations

import numpy as np

from asce.types import AudioBuffer


def to_mono(samples: np.ndarray) -> np.ndarray:
    arr = np.asarray(samples, dtype=np.float32)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        # (frames, channels) or (channels, frames). Channels are the short axis.
        if arr.shape[0] <= 8 and arr.shape[1] > arr.shape[0]:
            return arr.mean(axis=0).astype(np.float32)
        return arr.mean(axis=1).astype(np.float32)
    raise ValueError(f"expected 1d or 2d audio, got shape {arr.shape}")


def resample_linear(samples: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    if orig_sr == target_sr or len(samples) == 0:
        return samples
    try:
        import librosa

        return librosa.resample(samples, orig_sr=orig_sr, target_sr=target_sr).astype(np.float32)
    except Exception:
        duration = len(samples) / float(orig_sr)
        count = max(1, int(round(duration * target_sr)))
        old = np.linspace(0.0, 1.0, len(samples), endpoint=False)
        new = np.linspace(0.0, 1.0, count, endpoint=False)
        return np.interp(new, old, samples).astype(np.float32)


def highpass(samples: np.ndarray, cutoff_hz: float, sample_rate: int) -> np.ndarray:
    """One-pole high-pass. Removes rumble without a SciPy dependency."""

    if cutoff_hz <= 0 or len(samples) == 0:
        return samples.astype(np.float32)
    dt = 1.0 / float(sample_rate)
    rc = 1.0 / (2.0 * np.pi * float(cutoff_hz))
    alpha = rc / (rc + dt)
    x = np.asarray(samples, dtype=np.float32)
    y = np.empty_like(x)
    y[0] = x[0]
    # Recursive filter; sessions are minutes, not hours, so a loop is acceptable.
    prev_y = float(x[0])
    prev_x = float(x[0])
    for index in range(1, len(x)):
        prev_y = alpha * (prev_y + float(x[index]) - prev_x)
        prev_x = float(x[index])
        y[index] = prev_y
    return y


def peak_normalize(samples: np.ndarray, target: float = 0.99) -> np.ndarray:
    x = np.asarray(samples, dtype=np.float32)
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    if peak < 1e-8:
        return x
    return (x * (target / peak)).astype(np.float32)


def _standardize(spec: np.ndarray) -> np.ndarray:
    spec = np.asarray(spec, dtype=np.float32)
    return ((spec - float(spec.mean())) / (float(spec.std()) + 1e-6)).astype(np.float32)


def fix_frames(spec: np.ndarray, n_frames: int) -> np.ndarray:
    """Center-crop or right-pad a (bins, time) array to ``n_frames``."""

    spec = np.asarray(spec, dtype=np.float32)
    if spec.ndim != 2:
        raise ValueError("spectrogram must be 2d")
    time = spec.shape[1]
    if time == n_frames:
        return spec
    if time > n_frames:
        start = (time - n_frames) // 2
        return spec[:, start : start + n_frames]
    pad = n_frames - time
    return np.pad(spec, ((0, 0), (0, pad)))


def _mel_filterbank(sample_rate: int, n_fft: int, n_mels: int) -> np.ndarray:
    def hz_to_mel(hz: np.ndarray) -> np.ndarray:
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel: np.ndarray) -> np.ndarray:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    n_freqs = n_fft // 2 + 1
    mels = np.linspace(hz_to_mel(np.array(0.0)), hz_to_mel(np.array(sample_rate / 2)), n_mels + 2)
    hz = mel_to_hz(mels)
    bins = np.floor((n_fft + 1) * hz / sample_rate).astype(int)
    bank = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for index in range(n_mels):
        left, center, right = int(bins[index]), int(bins[index + 1]), int(bins[index + 2])
        if center <= left:
            center = left + 1
        if right <= center:
            right = center + 1
        for freq in range(left, center):
            if 0 <= freq < n_freqs:
                bank[index, freq] = (freq - left) / float(center - left)
        for freq in range(center, right):
            if 0 <= freq < n_freqs:
                bank[index, freq] = (right - freq) / float(right - center)
    return bank


def stft_power(samples: np.ndarray, n_fft: int, hop_length: int) -> np.ndarray:
    y = np.asarray(samples, dtype=np.float32).reshape(-1)
    if len(y) < n_fft:
        y = np.pad(y, (0, n_fft - len(y)))
    window = np.hanning(n_fft).astype(np.float32)
    frames = []
    for start in range(0, len(y) - n_fft + 1, hop_length):
        spectrum = np.fft.rfft(y[start : start + n_fft] * window)
        frames.append((spectrum.real ** 2 + spectrum.imag ** 2).astype(np.float32))
    if not frames:
        frames.append(np.zeros(n_fft // 2 + 1, dtype=np.float32))
    return np.stack(frames, axis=1)


def log_mel(
    samples: np.ndarray,
    sample_rate: int,
    n_mels: int = 128,
    n_fft: int = 512,
    hop_length: int = 128,
    n_frames: int = 32,
) -> np.ndarray:
    try:
        import librosa

        mel = librosa.feature.melspectrogram(
            y=np.asarray(samples, dtype=np.float32),
            sr=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )
        spec = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    except Exception:
        power = stft_power(samples, n_fft, hop_length)
        bank = _mel_filterbank(sample_rate, n_fft, n_mels)
        mel = bank @ power
        spec = np.log(mel + 1e-6).astype(np.float32)
    return _standardize(fix_frames(spec, n_frames))


def mfcc(
    samples: np.ndarray,
    sample_rate: int,
    n_mfcc: int = 20,
    n_mels: int = 64,
    n_fft: int = 512,
    hop_length: int = 128,
    n_frames: int = 32,
) -> np.ndarray:
    try:
        import librosa

        coeffs = librosa.feature.mfcc(
            y=np.asarray(samples, dtype=np.float32),
            sr=sample_rate,
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        ).astype(np.float32)
        return _standardize(fix_frames(coeffs, n_frames))
    except Exception:
        spec = log_mel(samples, sample_rate, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, n_frames=n_frames)
        n_bins = spec.shape[0]
        basis_n = np.arange(n_bins, dtype=np.float32)
        rows = []
        for k in range(n_mfcc):
            basis = np.cos(np.pi * k * (basis_n + 0.5) / n_bins)
            rows.append(basis @ spec)
        return _standardize(np.stack(rows, axis=0).astype(np.float32))


def energy_onsets(
    samples: np.ndarray,
    sample_rate: int,
    hop_length: int = 256,
    threshold_ratio: float = 3.0,
    min_gap_sec: float = 0.08,
) -> list[float]:
    """Peak-pick positive energy flux. Nearby peaks collapse into one keystroke."""

    y = np.asarray(samples, dtype=np.float32).reshape(-1)
    if len(y) < hop_length * 2:
        return []
    count = len(y) // hop_length
    energy = np.empty(count, dtype=np.float32)
    for index in range(count):
        frame = y[index * hop_length : (index + 1) * hop_length]
        energy[index] = np.sqrt(np.mean(frame * frame) + 1e-12)
    flux = np.maximum(0.0, np.diff(energy, prepend=energy[0]))
    peak = float(np.max(flux))
    if peak <= 1e-8:
        return []
    median = float(np.median(flux))
    mad = float(np.median(np.abs(flux - median))) + 1e-8
    threshold = max(median + threshold_ratio * mad, 0.2 * peak)
    peaks: list[int] = []
    for index in range(1, len(flux) - 1):
        if flux[index] > threshold and flux[index] >= flux[index - 1] and flux[index] > flux[index + 1]:
            peaks.append(index)
    times = [index * hop_length / float(sample_rate) for index in peaks]
    return _merge_times(times, min_gap_sec)


def librosa_onsets(
    samples: np.ndarray,
    sample_rate: int,
    hop_length: int = 256,
    min_gap_sec: float = 0.08,
) -> list[float]:
    import librosa

    times = librosa.onset.onset_detect(
        y=np.asarray(samples, dtype=np.float32),
        sr=sample_rate,
        hop_length=hop_length,
        units="time",
        backtrack=False,
    )
    return _merge_times([float(t) for t in times], min_gap_sec)


def _merge_times(times: list[float], min_gap_sec: float) -> list[float]:
    if not times:
        return []
    merged = [times[0]]
    for t in times[1:]:
        if t - merged[-1] < min_gap_sec:
            continue
        merged.append(t)
    return merged


def slice_audio(audio: AudioBuffer, t_start: float, t_end: float) -> np.ndarray:
    """Slice using times relative to the buffer clock (``audio.t0`` is time zero of sample 0 only if t0 matches)."""

    a = int(round((t_start - audio.t0) * audio.sample_rate))
    b = int(round((t_end - audio.t0) * audio.sample_rate))
    a = max(0, a)
    b = min(len(audio.samples), max(a, b))
    return audio.samples[a:b].astype(np.float32)
