"""Session schema: key events, manifests, and labeled windows."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class KeyEvent:
    key: str
    t_press: float
    t_release: float | None = None

    def to_json(self) -> dict:
        return {"key": self.key, "t_press": self.t_press, "t_release": self.t_release}

    @classmethod
    def from_json(cls, payload: dict) -> "KeyEvent":
        return cls(key=str(payload["key"]), t_press=float(payload["t_press"]), t_release=payload.get("t_release"))


@dataclass
class LabeledExample:
    samples: np.ndarray
    sample_rate: int
    key: str
    keyboard_id: str
    mic_id: str
    session_id: str
    t_start: float
    noise: str = "unknown"
    distance_cm: float | None = None

    def __post_init__(self) -> None:
        self.samples = np.asarray(self.samples, dtype=np.float32).reshape(-1)


@dataclass
class Session:
    session_id: str
    audio: np.ndarray
    sample_rate: int
    events: list[KeyEvent]
    meta: dict = field(default_factory=dict)
    path: str = ""

    def __post_init__(self) -> None:
        self.audio = np.asarray(self.audio, dtype=np.float32).reshape(-1)
