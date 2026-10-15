import numpy as np

from asce.audio_io import read_wav, write_wav
from asce.bootstrap import load_builtins
from asce.config import load_yaml
from asce.pipeline import ChunkRunner, build_pipeline
from asce.registry import RegistryError, available, build
from asce.types import AudioBuffer


def _clicks(path, times=(0.2, 0.55, 0.9), sr=16000):
    audio = np.zeros(sr, dtype=np.float32)
    for t in times:
        start = int(t * sr)
        n = 180
        burst = np.hanning(n).astype(np.float32) * np.sin(2 * np.pi * 1800 * np.arange(n) / sr)
        audio[start : start + n] += burst
    write_wav(path, audio, sr)


def test_registry_lists_keytap_and_cpuhum_stages():
    load_builtins()
    assert "onset" in available("segmenter")
    assert "fixed_window" in available("segmenter")
    assert "logmel" in available("features")
    assert "band_energy" in available("features")
    assert "ngram_beam" in available("decoder")
    assert "isa_prior" in available("decoder")
    assert "spectrogram_cnn" in available("model")
    assert "wav2vec2" in available("model")
    assert "hubert" in available("model")


def test_unknown_stage_names_the_known_ones():
    load_builtins()
    try:
        build("decoder", {"type": "missing"})
    except RegistryError as exc:
        assert "argmax" in str(exc)
    else:
        raise AssertionError("expected RegistryError")


def test_pipeline_finds_clicks(tmp_path):
    wav = tmp_path / "clicks.wav"
    _clicks(wav)
    cfg = load_yaml("configs/keytap_baseline.yaml")
    cfg["source"] = {"type": "file", "path": str(wav)}
    result = build_pipeline(cfg).run()
    assert len(result.predictions) >= 2
    assert result.raw_predictions[0].label


def test_stream_emits_without_duplicating(tmp_path):
    wav = tmp_path / "clicks.wav"
    _clicks(wav)
    cfg = load_yaml("configs/keytap_baseline.yaml")
    cfg["source"] = {"type": "file", "path": str(wav)}
    pipeline = build_pipeline(cfg)
    runner = ChunkRunner(pipeline, tail_sec=0.4)
    samples, sr = read_wav(wav)
    labels = []
    hop = sr // 5
    for cursor in range(0, len(samples), hop):
        result = runner.push(samples[cursor : cursor + hop], sr)
        if result:
            labels.extend(pred.label for pred in result.predictions)
    assert len(labels) >= 2
    assert len(labels) <= 4


def test_stream_keeps_reporting_a_long_take():
    cfg = load_yaml("configs/keytap_baseline.yaml")
    cfg["source"] = {"type": "file", "path": "configs/keytap_baseline.yaml"}
    runner = ChunkRunner(build_pipeline(cfg), tail_sec=0.5)
    sample_rate = 16000
    duration = 8.0
    audio = np.zeros(int(sample_rate * duration), dtype=np.float32)
    clicks = [0.3 + index * 0.25 for index in range(28)]
    for t in clicks:
        start = int(t * sample_rate)
        audio[start : start + 160] += np.hanning(160).astype(np.float32)
    found = 0
    hop = 2048
    for cursor in range(0, len(audio), hop):
        result = runner.push(audio[cursor : cursor + hop], sample_rate)
        if result:
            found += len(result.predictions)
    assert found >= 24
    assert found <= len(clicks) + 2


def test_logmel_shape():
    load_builtins()
    from asce.stages.features.logmel import LogMelExtractor
    from asce.types import Segment

    extractor = LogMelExtractor(n_mels=64, n_frames=32)
    segment = Segment(np.random.randn(3200).astype(np.float32), 16000, 0.0, 0.2)
    spec = extractor.extract(segment)
    assert spec.shape == (64, 32)


def test_preprocessor_resamples_to_target():
    load_builtins()
    pre = build("preprocessor", {"type": "standard", "target_sample_rate": 8000, "highpass_hz": 0})
    audio = AudioBuffer(np.zeros(16000, dtype=np.float32), 16000)
    out = pre.process(audio)
    assert out.sample_rate == 8000
    assert abs(len(out.samples) - 8000) <= 2
