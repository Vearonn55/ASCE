"""Pair key-logger events with the microphone on one monotonic clock.

macOS will not deliver key events until the process has Input Monitoring
(and sometimes Accessibility) permission. That grant is per-terminal app:
System Settings -> Privacy & Security -> Input Monitoring. This harness only
records the machine it runs on; it is the ground-truth side of a dataset
you record yourself.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from asce.data.schema import KeyEvent
from asce.data.sessions import write_session
from asce.keys import normalize_key


class EventLog:
    """Press/release pairing, independent of the audio device so it can be tested."""

    def __init__(self, origin: float | None = None) -> None:
        self.origin = time.monotonic() if origin is None else origin
        self.held: dict[str, KeyEvent] = {}
        self.events: list[KeyEvent] = []

    def _now(self, now: float | None) -> float:
        return (time.monotonic() if now is None else now) - self.origin

    def _ident(self, key: object) -> str:
        char = getattr(key, "char", None)
        if isinstance(char, str) and char:
            return f"char:{char}"
        name = getattr(key, "name", None)
        if isinstance(name, str):
            return f"name:{name}"
        return f"id:{id(key)}"

    def press(self, key: object, now: float | None = None) -> None:
        name = normalize_key(key)
        ident = self._ident(key)
        if name is None or ident in self.held:
            return
        self.held[ident] = KeyEvent(key=name, t_press=self._now(now))

    def release(self, key: object, now: float | None = None) -> None:
        ident = self._ident(key)
        event = self.held.pop(ident, None)
        if event is None:
            return
        event.t_release = self._now(now)
        self.events.append(event)

    def flush(self, now: float | None = None) -> None:
        stamp = self._now(now)
        for event in self.held.values():
            event.t_release = stamp
            self.events.append(event)
        self.held.clear()
        self.events.sort(key=lambda event: event.t_press)


class SessionRecorder:
    def __init__(
        self,
        out_dir: str | Path,
        keyboard_id: str,
        mic_id: str,
        distance_cm: float | None = None,
        noise: str = "quiet",
        sample_rate: int = 16000,
        device: int | None = None,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.keyboard_id = keyboard_id
        self.mic_id = mic_id
        self.distance_cm = distance_cm
        self.noise = noise
        self.sample_rate = int(sample_rate)
        self.device = device

    def record(self, seconds: float | None = None) -> Path:
        import sounddevice as sd
        from pynput import keyboard

        log = EventLog()
        chunks: list[np.ndarray] = []
        audio_t0: dict[str, float] = {}

        def callback(indata, frames, time_info, status) -> None:  # noqa: ARG001
            if "t0" not in audio_t0:
                audio_t0["t0"] = time.monotonic() - log.origin
            chunks.append(np.asarray(indata[:, 0], dtype=np.float32).copy())

        stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            device=self.device,
            callback=callback,
        )
        listener = keyboard.Listener(on_press=log.press, on_release=log.release)
        stream.start()
        listener.start()
        try:
            if seconds is None:
                print("Recording. Press Ctrl+C to stop.", flush=True)
                while True:
                    time.sleep(0.1)
            else:
                print(f"Recording for {seconds:.1f}s.", flush=True)
                time.sleep(seconds)
        except KeyboardInterrupt:
            print("Stopping.", flush=True)
        finally:
            listener.stop()
            stream.stop()
            stream.close()
            log.flush()
        audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
        meta = {
            "session_id": session_id,
            "keyboard_id": self.keyboard_id,
            "mic_id": self.mic_id,
            "distance_cm": self.distance_cm,
            "noise": self.noise,
            "sample_rate": self.sample_rate,
            "audio_t0": float(audio_t0.get("t0", 0.0)),
            "created": datetime.now(timezone.utc).isoformat(),
            "channels": 1,
        }
        path = write_session(self.out_dir, audio, self.sample_rate, log.events, meta)
        print(f"Wrote {path} ({len(log.events)} key events, {len(audio) / self.sample_rate:.2f}s audio).", flush=True)
        return path


def list_input_devices() -> list[dict]:
    import sounddevice as sd

    devices = []
    for index, info in enumerate(sd.query_devices()):
        if info.get("max_input_channels", 0) > 0:
            devices.append({"index": index, "name": info["name"], "sample_rate": info.get("default_samplerate")})
    return devices
