from types import SimpleNamespace

from asce.data.recorder import EventLog
from asce.data.schema import KeyEvent
from asce.data.sessions import load_session, write_session
from asce.data.dataset import align_onsets_to_events, examples_from_sessions
from asce.keys import VOCAB, normalize_key
from asce.synth.physics import synthesize_corpus


def test_vocab_size_is_in_the_planned_range():
    assert 40 <= len(VOCAB) <= 60


def test_normalize_key_maps_characters_and_specials():
    assert normalize_key(SimpleNamespace(char="A")) == "a"
    assert normalize_key(SimpleNamespace(char=",")) == "comma"
    assert normalize_key(SimpleNamespace(char=None, name="space")) == "space"
    assert normalize_key(SimpleNamespace(char=None, name="shift_r")) == "rshift"
    assert normalize_key(SimpleNamespace(char=None, name="f5")) is None


def test_event_log_pairs_press_and_release_and_ignores_repeats():
    log = EventLog(origin=0.0)
    key = SimpleNamespace(char="a")
    log.press(key, now=1.0)
    log.press(key, now=1.05)
    log.release(key, now=1.2)
    assert len(log.events) == 1
    assert log.events[0].t_press == 1.0
    assert log.events[0].t_release == 1.2


def test_session_roundtrip_and_windows(tmp_path):
    paths = synthesize_corpus(tmp_path, keyboards=["mechanical"], keys=list("ab"), examples_per_key=2, seed=1)
    session = load_session(paths[0])
    assert session.meta["keyboard_id"] == "mechanical"
    assert session.events
    examples = examples_from_sessions([session])
    assert len(examples) == len(session.events)
    assert examples[0].samples.shape[0] > 100


def test_onset_alignment_matches_nearby_events():
    events = [KeyEvent("a", 0.20), KeyEvent("b", 0.50), KeyEvent("c", 1.50)]
    pairs = align_onsets_to_events([0.21, 0.52, 0.90], events, tolerance=0.08)
    matched = [event.key for _t, event in pairs]
    assert matched == ["a", "b"]


def test_write_session_meta(tmp_path):
    import numpy as np

    folder = write_session(
        tmp_path,
        np.zeros(160, dtype=np.float32),
        16000,
        [KeyEvent("a", 0.01, 0.02)],
        {"session_id": "s1", "keyboard_id": "k", "mic_id": "m", "audio_t0": 0.0},
    )
    loaded = load_session(folder)
    assert loaded.events[0].key == "a"
    assert loaded.sample_rate == 16000
