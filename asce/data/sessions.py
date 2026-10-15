"""Read and write a recorded session directory.

Layout::

    <session_id>/
        audio.wav
        events.jsonl
        meta.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from asce.audio_io import read_wav, write_wav
from asce.data.schema import KeyEvent, Session


def write_session(
    root: str | Path,
    audio: np.ndarray,
    sample_rate: int,
    events: list[KeyEvent],
    meta: dict,
) -> Path:
    session_id = str(meta["session_id"])
    folder = Path(root) / session_id
    folder.mkdir(parents=True, exist_ok=True)
    write_wav(folder / "audio.wav", audio, sample_rate)
    with (folder / "events.jsonl").open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event.to_json()) + "\n")
    payload = dict(meta)
    payload["sample_rate"] = int(sample_rate)
    with (folder / "meta.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return folder


def load_session(folder: str | Path) -> Session:
    folder = Path(folder)
    audio, sample_rate = read_wav(folder / "audio.wav")
    events: list[KeyEvent] = []
    events_path = folder / "events.jsonl"
    if events_path.exists():
        with events_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    events.append(KeyEvent.from_json(json.loads(line)))
    meta_path = folder / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    meta.setdefault("session_id", folder.name)
    return Session(
        session_id=str(meta.get("session_id", folder.name)),
        audio=audio,
        sample_rate=sample_rate,
        events=events,
        meta=meta,
        path=str(folder),
    )


def load_sessions(root: str | Path) -> list[Session]:
    root = Path(root)
    if not root.exists():
        return []
    folders = [path for path in sorted(root.iterdir()) if path.is_dir() and (path / "audio.wav").exists()]
    return [load_session(folder) for folder in folders]
