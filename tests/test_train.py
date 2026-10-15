import numpy as np
import torch

from asce.audio_io import write_wav
from asce.data.dataset import KeystrokeDataset, examples_from_sessions
from asce.data.farm import segment_audio_file, write_manifest
from asce.data.sessions import load_sessions
from asce.stages.models.cnn import SpectrogramCNN
from asce.synth.generative import generate_specs, train_conditional_generator
from asce.synth.physics import synthesize_corpus
from asce.synth.validate import transfer_gate
from asce.train.adversarial import grad_reverse
from asce.train.augment import AudioAugmentor, mix_snr, synthetic_rir
from asce.train.device import resolve_device, sanity_check_device
from asce.train.loso import format_loso, run_loso
from asce.train.ssl import contrastive_pretrain, embed_waves, kmeans
from asce.train.train import evaluate, fit, load_encoder_weights, make_loader


def test_augment_keeps_length_and_rir_decays():
    rng = np.random.default_rng(0)
    signal = rng.normal(0, 0.1, 1600).astype(np.float32)
    out = AudioAugmentor()(signal, 16000)
    assert out.shape == signal.shape
    impulse = synthetic_rir(16000, 0.05, rng)
    assert abs(impulse[0]) > abs(impulse[-1])
    mixed = mix_snr(np.ones(100, dtype=np.float32), rng.normal(0, 1, 100).astype(np.float32), 20.0)
    assert mixed.shape == (100,)


def test_grad_reverse_flips_the_gradient():
    features = torch.tensor([1.0, 2.0], requires_grad=True)
    out = grad_reverse(features, 0.5).sum()
    out.backward()
    assert torch.allclose(features.grad, torch.tensor([-0.5, -0.5]))


def test_single_keyboard_training_beats_chance(tmp_path):
    from asce.train.train import set_seed

    set_seed(0)
    synthesize_corpus(
        tmp_path,
        keyboards=["mechanical"],
        keys=list("abcd"),
        examples_per_key=8,
        seed=0,
    )
    examples = examples_from_sessions(load_sessions(tmp_path))
    labels = sorted({item.key for item in examples})
    ds = KeystrokeDataset(examples, label_to_index={label: i for i, label in enumerate(labels)})
    model = SpectrogramCNN(labels=labels)
    device = torch.device("cpu")
    result = fit(model, make_loader(ds, 8, True), make_loader(ds, 8, False), device, epochs=12, lr=1e-3)
    assert result["top1"] > 0.8


def test_loso_reports_a_gap(tmp_path):
    synthesize_corpus(
        tmp_path,
        keyboards=["mechanical", "membrane"],
        keys=list("abc"),
        examples_per_key=6,
        seed=2,
    )
    examples = examples_from_sessions(load_sessions(tmp_path))
    summary = run_loso(examples, epochs=2, batch_size=8, device=torch.device("cpu"), seed=2, out_path=tmp_path / "loso.json")
    assert len(summary["folds"]) == 2
    assert summary["mean_gap"] is not None
    text = format_loso(summary)
    assert "generalization gap" in text
    assert (tmp_path / "loso.json").exists()


def test_contrastive_pretrain_and_kmeans():
    rng = np.random.default_rng(0)
    waves = [(rng.normal(0, 0.2, 3200).astype(np.float32), 16000) for _ in range(4)]
    device = torch.device("cpu")
    result = contrastive_pretrain(waves, epochs=1, batch_size=2, device=device, n_mels=32)
    assert np.isfinite(result["loss"])
    embeddings = embed_waves(result["model"], waves, device, n_mels=32)
    groups = kmeans(embeddings, k=2, seed=0)
    assert len(groups) == 4
    copied = load_encoder_weights(SpectrogramCNN(labels=["a", "b"]), _save(result["model"], waves))
    assert copied > 0


def _save(model, _waves):
    import tempfile
    from pathlib import Path

    path = Path(tempfile.mkdtemp()) / "ssl.pt"
    torch.save({"state_dict": model.state_dict(), "labels": ["unused"]}, path)
    return path


def test_transfer_gate_rejects_missing_real_data(tmp_path):
    synth = tmp_path / "synth"
    synthesize_corpus(synth, keyboards=["mechanical"], keys=list("ab"), examples_per_key=2, seed=0)
    report = transfer_gate(str(synth), str(tmp_path / "missing"))
    assert report["transfers"] is False
    assert "real" in report["reason"]


def test_transfer_gate_accepts_a_matched_domain(tmp_path):
    synth = tmp_path / "synth"
    real = tmp_path / "real"
    synthesize_corpus(synth, keyboards=["mechanical"], keys=list("abc"), examples_per_key=8, seed=0)
    synthesize_corpus(real, keyboards=["mechanical"], keys=list("abc"), examples_per_key=4, seed=1)
    report = transfer_gate(str(synth), str(real), epochs=12, batch_size=8, device=torch.device("cpu"))
    assert report["real_accuracy"] > report["chance"]
    assert report["transfers"] is True


def test_generator_emits_the_requested_shape():
    rng = np.random.default_rng(0)
    specs = rng.normal(0, 1, (12, 16, 8)).astype(np.float32)
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    generator = train_conditional_generator(specs, labels, steps=2, batch_size=4)
    out = generate_specs(generator, [0, 1, 1])
    assert out.shape == (3, 16, 8)


def test_farm_segments_are_unlabeled(tmp_path):
    sr = 16000
    audio = np.zeros(sr, dtype=np.float32)
    for t in (0.15, 0.45, 0.75):
        i = int(t * sr)
        audio[i : i + 160] += np.hanning(160).astype(np.float32)
    wav = tmp_path / "typing.wav"
    write_wav(wav, audio, sr)
    rows = segment_audio_file(wav, tmp_path / "clips")
    write_manifest(rows, tmp_path / "manifest.jsonl")
    assert len(rows) >= 2
    assert all(row["key"] is None for row in rows)


def test_cpuhum_config_runs(tmp_path):
    from asce.config import load_yaml
    from asce.pipeline import build_pipeline

    sr = 16000
    t = np.arange(sr, dtype=np.float32) / sr
    audio = 0.2 * np.sin(2 * np.pi * 180 * t)
    wav = tmp_path / "hum.wav"
    write_wav(wav, audio, sr)
    cfg = load_yaml("configs/cpuhum.yaml")
    cfg["source"] = {"type": "file", "path": str(wav)}
    result = build_pipeline(cfg).run()
    assert result.predictions
    assert result.predictions[0].label in {"nop", "mov", "add", "load", "store", "cmp", "jmp", "call"}


def test_device_report_has_a_resolved_device():
    report = sanity_check_device()
    assert report["resolved"] in ("cpu", "mps", "cuda")
    assert resolve_device("cpu").type == "cpu"


def test_adversarial_fit_runs():
    from asce.data.schema import LabeledExample

    rng = np.random.default_rng(0)
    examples = []
    for keyboard, shift in (("mechanical", 0.0), ("membrane", 1.0)):
        for key_index, key in enumerate("ab"):
            for _ in range(4):
                samples = rng.normal(shift + key_index, 0.3, 3200).astype(np.float32)
                examples.append(
                    LabeledExample(samples, 16000, key, keyboard, "mic", "s", 0.0)
                )
    labels = ["a", "b"]
    ds = KeystrokeDataset(examples, label_to_index={"a": 0, "b": 1})
    model = SpectrogramCNN(labels=labels)
    fit(
        model,
        make_loader(ds, 4, True),
        None,
        torch.device("cpu"),
        epochs=1,
        lr=1e-3,
        adversarial=True,
        num_domains=2,
        embed_dim=64,
    )
    metrics = evaluate(model, make_loader(ds, 4, False), torch.device("cpu"))
    assert metrics["n"] == len(examples)
