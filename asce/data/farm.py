"""Pull typing audio and cut it into unlabeled keystroke clips.

Farmed clips have timing and no key identity. The intended use is
contrastive pretraining, not supervised classification. Weak visual labels
(OCR of an on-screen keyboard) are deliberately not built here: they are a
larger project than the audio engine, and unlabeled audio is the better
match for a generalization-first goal.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from asce.audio_io import read_wav, write_wav
from asce.stages.preprocess.standard import StandardPreprocessor
from asce.stages.segment.onset import OnsetSegmenter
from asce.types import AudioBuffer


def segment_audio_file(path: str | Path, out_dir: str | Path, method: str = "energy") -> list[dict]:
    samples, sample_rate = read_wav(path)
    audio = StandardPreprocessor().process(AudioBuffer(samples, sample_rate, t0=0.0))
    segments = OnsetSegmenter(method=method).segment(audio)
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    stem = Path(path).stem
    for index, segment in enumerate(segments):
        clip = destination / f"{stem}-{index:05d}.wav"
        write_wav(clip, segment.samples, segment.sample_rate)
        rows.append(
            {
                "path": str(clip),
                "t_start": segment.t_start,
                "t_end": segment.t_end,
                "source": str(path),
                "key": None,
            }
        )
    return rows


def write_manifest(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def download_youtube_audio(url: str, out_dir: str | Path) -> Path:
    if shutil.which("yt-dlp") is None:
        raise SystemExit("yt-dlp is not on PATH")
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is required for yt-dlp audio extraction. Install ffmpeg and retry.")
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    template = str(destination / "%(id)s.%(ext)s")
    subprocess.run(
        ["yt-dlp", "-x", "--audio-format", "wav", "-o", template, url],
        check=True,
    )
    wavs = sorted(destination.glob("*.wav"), key=lambda item: item.stat().st_mtime, reverse=True)
    if not wavs:
        raise RuntimeError(f"yt-dlp finished but no wav appeared in {destination}")
    return wavs[0]


def farm_urls(urls: list[str], out_dir: str | Path, method: str = "energy") -> list[dict]:
    out_dir = Path(out_dir)
    raw = out_dir / "raw"
    clips = out_dir / "clips"
    rows: list[dict] = []
    for url in urls:
        wav = download_youtube_audio(url, raw)
        segmented = segment_audio_file(wav, clips, method=method)
        for row in segmented:
            row["url"] = url
        rows.extend(segmented)
    write_manifest(rows, out_dir / "manifest.jsonl")
    return rows
