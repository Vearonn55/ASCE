"""Command line for recording, training, evaluation, synthesis, and inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from asce.bootstrap import load_builtins
from asce.config import load_yaml
from asce.registry import build


def _pct(value) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    return f"{100.0 * float(value):.1f}%"


def _train_block(cfg: dict) -> dict:
    block = cfg.get("train") or {}
    return block if isinstance(block, dict) else {}


def _examples(data: str, keyboard: str | None, label_mode: str):
    from asce.data.dataset import examples_from_sessions
    from asce.data.sessions import load_sessions

    sessions = load_sessions(data)
    if keyboard:
        sessions = [session for session in sessions if str(session.meta.get("keyboard_id")) == keyboard]
    if not sessions:
        raise SystemExit(f"no sessions found in {data}" + (f" for keyboard {keyboard}" if keyboard else ""))
    examples = examples_from_sessions(sessions, label_mode=label_mode)
    if not examples:
        raise SystemExit("sessions contain no labeled keystroke windows")
    return sessions, examples


def _spectral_or_wave(model_type: str, requested: str) -> str:
    if model_type in ("wav2vec2", "hubert"):
        return "wave"
    return requested


def cmd_record(args: argparse.Namespace) -> None:
    from asce.data.recorder import SessionRecorder

    recorder = SessionRecorder(
        out_dir=args.out,
        keyboard_id=args.keyboard,
        mic_id=args.mic,
        distance_cm=args.distance_cm,
        noise=args.noise,
        sample_rate=args.sample_rate,
        device=args.device,
    )
    recorder.record(seconds=args.seconds)


def cmd_mics(_args: argparse.Namespace) -> None:
    from asce.data.recorder import list_input_devices

    for info in list_input_devices():
        print(f"{info['index']:>3}  {info['name']}  ({info['sample_rate']} Hz)")


def cmd_device(_args: argparse.Namespace) -> None:
    from asce.train.device import sanity_check_device

    report = sanity_check_device()
    print(json.dumps(report, indent=2))
    if report.get("max_abs_diff") is not None and report["max_abs_diff"] > 1e-2:
        print("MPS disagrees with CPU on a tiny conv. Re-run experiments with --device cpu.")


def cmd_train(args: argparse.Namespace) -> None:
    import torch

    from asce.data.dataset import KeystrokeDataset
    from asce.train.augment import AudioAugmentor
    from asce.train.device import resolve_device
    from asce.train.loso import random_split
    from asce.train.tracking import ExperimentTracker
    from asce.train.train import fit, load_encoder_weights, make_loader, save_checkpoint, set_seed

    cfg = load_yaml(args.config)
    train_cfg = _train_block(cfg)
    set_seed(args.seed)
    _sessions, examples = _examples(args.data, args.keyboard, args.label_mode)
    train_examples, val_examples = random_split(examples, args.val_fraction, args.seed)
    labels = sorted({example.key for example in examples})
    keyboards = sorted({example.keyboard_id for example in examples})
    label_to_index = {label: i for i, label in enumerate(labels)}
    keyboard_to_index = {name: i for i, name in enumerate(keyboards)}
    device = resolve_device(args.device)
    model_spec = dict(cfg["model"])
    model_spec.pop("num_classes", None)
    model_spec["labels"] = labels
    model_type = str(model_spec.get("type"))
    if model_type in ("wav2vec2", "hubert"):
        model_spec["device"] = str(device)
    representation = _spectral_or_wave(model_type, args.representation or train_cfg.get("representation", "mel"))
    load_builtins()
    extractor = build("features", cfg["features"])
    model = build("model", model_spec)
    if args.init:
        copied = load_encoder_weights(model, args.init)
        print(f"loaded {copied} encoder tensors from {args.init}")
    augment = AudioAugmentor() if args.augment else None
    common = dict(
        representation=representation,
        extractor=extractor,
        label_to_index=label_to_index,
        keyboard_to_index=keyboard_to_index,
    )
    train_ds = KeystrokeDataset(train_examples, augment=augment, **common)
    val_ds = KeystrokeDataset(val_examples, **common) if val_examples else None
    epochs = args.epochs if args.epochs is not None else int(train_cfg.get("epochs", 8))
    batch_size = args.batch_size or int(train_cfg.get("batch_size", 16))
    lr = args.lr if args.lr is not None else float(train_cfg.get("lr", 1e-3))
    tracker = ExperimentTracker(
        backend=args.tracker,
        run_name=args.run_name,
        config={"model": model_type, "keyboard": args.keyboard, "epochs": epochs, "labels": labels},
    )
    result = fit(
        model,
        make_loader(train_ds, batch_size, shuffle=True),
        make_loader(val_ds, batch_size, shuffle=False) if val_ds is not None else None,
        device,
        epochs=epochs,
        lr=lr,
        tracker=tracker,
        adversarial=args.adversarial,
        num_domains=len(keyboards),
    )
    tracker.finish()
    out = Path(args.checkpoint)
    save_checkpoint(model, out, labels, extra={"model_type": model_type, "keyboard": args.keyboard})
    summary = {
        "top1": result["top1"],
        "train_top1": result.get("train_top1"),
        "n_train": len(train_examples),
        "n_val": len(val_examples),
        "labels": labels,
        "keyboards": keyboards,
        "checkpoint": str(out),
        "device": str(device),
    }
    report = Path("experiments") / ("m1_single_keyboard.json" if args.keyboard else "train.json")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"validation top-1: {_pct(summary['top1'])}  train top-1: {_pct(summary['train_top1'])}  "
        f"({summary['n_val']} val / {summary['n_train']} train)"
    )
    print(f"checkpoint: {out}")
    print(f"report: {report}")


def cmd_loso(args: argparse.Namespace) -> None:
    from asce.train.augment import AudioAugmentor
    from asce.train.device import resolve_device
    from asce.train.loso import format_loso, run_loso

    _sessions, examples = _examples(args.data, None, args.label_mode)
    cfg = load_yaml(args.config) if args.config else {"model": {"type": "spectrogram_cnn"}}
    model_name = args.model or cfg.get("model", {}).get("type", "spectrogram_cnn")
    representation = _spectral_or_wave(str(model_name), "mel")
    summary = run_loso(
        examples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        device=resolve_device(args.device),
        model_name=str(model_name),
        representation=representation,
        augment=AudioAugmentor() if args.augment else None,
        out_path=args.out,
    )
    print(format_loso(summary))
    print(f"report: {args.out}")


def cmd_pretrain(args: argparse.Namespace) -> None:
    from asce.train.device import resolve_device
    from asce.train.ssl import contrastive_pretrain, embed_waves, kmeans, load_unlabeled_dir
    from asce.train.train import save_checkpoint

    waves = load_unlabeled_dir(args.data)
    if len(waves) < 2:
        raise SystemExit(f"need at least two wav clips in {args.data}")
    device = resolve_device(args.device)
    result = contrastive_pretrain(waves, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=device)
    save_checkpoint(result["model"], args.checkpoint, labels=["unused"], extra={"objective": "nt-xent"})
    embeddings = embed_waves(result["model"], waves, device)
    groups = kmeans(embeddings, k=min(args.clusters, len(waves)), seed=args.seed)
    sizes = {int(index): int((groups == index).sum()) for index in sorted(set(groups.tolist()))}
    print(f"contrastive loss: {result['loss']:.4f}")
    print(f"cluster sizes: {sizes}")
    print(f"checkpoint: {args.checkpoint}")


def cmd_decode_eval(args: argparse.Namespace) -> None:
    from asce.stages.features.logmel import LogMelExtractor
    from asce.train.decode_eval import score_sessions
    from asce.train.device import resolve_device

    cfg = load_yaml(args.config)
    _sessions, examples = _examples(args.data, args.keyboard, "event_window")
    device = resolve_device(args.device)
    model = _model_from_checkpoint(cfg, args.checkpoint, device)
    load_builtins()
    decoder = build("decoder", {"type": "ngram_beam", "lm_weight": args.lm_weight})
    extractor = build("features", cfg["features"]) if "features" in cfg else LogMelExtractor()
    report = score_sessions(model, examples, decoder, extractor, device)
    printable = {key: report[key] for key in ("raw_accuracy", "decoded_accuracy", "lift")}
    print(json.dumps(printable, indent=2))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    slim = dict(printable)
    slim["sessions"] = [{k: v for k, v in row.items() if k != "reference"} for row in report["sessions"]]
    out.write_text(json.dumps(slim, indent=2), encoding="utf-8")


def _model_from_checkpoint(cfg: dict, checkpoint: str, device):
    import torch

    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    labels = list(blob.get("labels") or [])
    if not labels or labels == ["unused"]:
        raise SystemExit(f"{checkpoint} has no classification labels")
    spec = dict(cfg.get("model") or {"type": "spectrogram_cnn"})
    spec.pop("num_classes", None)
    spec["labels"] = labels
    spec["checkpoint"] = checkpoint
    if spec.get("type") in ("wav2vec2", "hubert"):
        spec["device"] = str(device)
    load_builtins()
    model = build("model", spec)
    model.to(device)
    return model


def cmd_infer(args: argparse.Namespace) -> None:
    from asce.pipeline import build_pipeline

    cfg = load_yaml(args.config)
    cfg["source"] = {"type": "file", "path": args.wav}
    if args.checkpoint:
        import torch

        blob = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        labels = blob.get("labels")
        if labels and labels != ["unused"]:
            cfg["model"]["labels"] = list(labels)
            cfg["model"].pop("num_classes", None)
            cfg["model"]["checkpoint"] = args.checkpoint
    result = build_pipeline(cfg).run()
    print(result.text)
    for prediction in result.predictions:
        print(f"{prediction.t_start:7.3f}  {prediction.label:<12}  {prediction.confidence:.3f}")


def cmd_synth(args: argparse.Namespace) -> None:
    from asce.synth.physics import synthesize_corpus

    keys = None if args.keys == "common" else (list("abcdefghijklmnopqrstuvwxyz0123456789") if args.keys == "all" else args.keys.split(","))
    paths = synthesize_corpus(
        args.out,
        keyboards=args.keyboards.split(",") if args.keyboards else None,
        keys=keys,
        examples_per_key=args.per_key,
        seed=args.seed,
        apply_rir=args.rir,
        phrase=args.phrase,
    )
    print(f"wrote {len(paths)} sessions under {args.out}")


def cmd_transfer(args: argparse.Namespace) -> None:
    from asce.train.device import resolve_device
    from asce.synth.validate import transfer_gate

    report = transfer_gate(
        args.synth,
        args.real,
        epochs=args.epochs,
        device=resolve_device(args.device),
        min_ratio_above_chance=args.min_ratio,
    )
    print(json.dumps(report, indent=2))
    if not report["transfers"]:
        raise SystemExit(2)


def cmd_farm(args: argparse.Namespace) -> None:
    from asce.data.farm import farm_urls, segment_audio_file, write_manifest

    if args.wav:
        rows = []
        for wav in args.wav:
            rows.extend(segment_audio_file(wav, Path(args.out) / "clips"))
        write_manifest(rows, Path(args.out) / "manifest.jsonl")
    elif args.urls:
        rows = farm_urls(args.urls, args.out)
    else:
        raise SystemExit("pass YouTube URLs or --wav files")
    labeled = sum(1 for row in rows if row.get("key"))
    print(f"{len(rows)} unlabeled clips, {labeled} with keys (expect 0 unless you add labels)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="asce", description="Acoustic Side-Channel Engine")
    sub = parser.add_subparsers(dest="cmd", required=True)

    record = sub.add_parser("record", help="record mic audio and local key events")
    record.add_argument("--out", default="data/sessions")
    record.add_argument("--keyboard", required=True)
    record.add_argument("--mic", default="laptop")
    record.add_argument("--distance-cm", type=float, default=None)
    record.add_argument("--noise", default="quiet")
    record.add_argument("--seconds", type=float, default=None)
    record.add_argument("--sample-rate", type=int, default=16000)
    record.add_argument("--device", type=int, default=None)
    record.set_defaults(func=cmd_record)

    mics = sub.add_parser("mics", help="list input devices")
    mics.set_defaults(func=cmd_mics)

    device = sub.add_parser("device", help="MPS vs CPU sanity check")
    device.set_defaults(func=cmd_device)

    train = sub.add_parser("train", help="train a keytap classifier")
    train.add_argument("--config", default="configs/keytap_baseline.yaml")
    train.add_argument("--data", default="data/sessions")
    train.add_argument("--keyboard", default=None, help="restrict to one keyboard (milestone 1)")
    train.add_argument("--checkpoint", default="checkpoints/keytap.pt")
    train.add_argument("--epochs", type=int, default=None)
    train.add_argument("--batch-size", type=int, default=None)
    train.add_argument("--lr", type=float, default=None)
    train.add_argument("--val-fraction", type=float, default=0.2)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--device", default="auto")
    train.add_argument("--tracker", default="auto", choices=["auto", "wandb", "mlflow", "jsonl"])
    train.add_argument("--run-name", default=None)
    train.add_argument("--augment", action="store_true")
    train.add_argument("--adversarial", action="store_true")
    train.add_argument("--init", default=None, help="contrastive checkpoint to warm-start the encoder")
    train.add_argument("--representation", default=None)
    train.add_argument("--label-mode", default="event_window", choices=["event_window", "onset_align"])
    train.set_defaults(func=cmd_train)

    loso = sub.add_parser("loso", help="leave-one-keyboard-out evaluation")
    loso.add_argument("--data", default="data/sessions")
    loso.add_argument("--config", default="configs/keytap_baseline.yaml")
    loso.add_argument("--model", default=None)
    loso.add_argument("--epochs", type=int, default=5)
    loso.add_argument("--batch-size", type=int, default=16)
    loso.add_argument("--lr", type=float, default=1e-2)
    loso.add_argument("--seed", type=int, default=0)
    loso.add_argument("--device", default="auto")
    loso.add_argument("--augment", action="store_true")
    loso.add_argument("--label-mode", default="event_window", choices=["event_window", "onset_align"])
    loso.add_argument("--out", default="experiments/loso.json")
    loso.set_defaults(func=cmd_loso)

    pretrain = sub.add_parser("pretrain", help="contrastive pretraining on unlabeled clips")
    pretrain.add_argument("--data", default="data/farmed/clips")
    pretrain.add_argument("--checkpoint", default="checkpoints/ssl.pt")
    pretrain.add_argument("--epochs", type=int, default=5)
    pretrain.add_argument("--batch-size", type=int, default=16)
    pretrain.add_argument("--lr", type=float, default=1e-3)
    pretrain.add_argument("--clusters", type=int, default=8)
    pretrain.add_argument("--seed", type=int, default=0)
    pretrain.add_argument("--device", default="auto")
    pretrain.set_defaults(func=cmd_pretrain)

    decode = sub.add_parser("decode-eval", help="raw accuracy vs n-gram beam decoding")
    decode.add_argument("--config", default="configs/keytap_baseline.yaml")
    decode.add_argument("--data", default="data/sessions")
    decode.add_argument("--checkpoint", required=True)
    decode.add_argument("--keyboard", default=None)
    decode.add_argument("--lm-weight", type=float, default=1.5)
    decode.add_argument("--device", default="auto")
    decode.add_argument("--out", default="experiments/decode_eval.json")
    decode.set_defaults(func=cmd_decode_eval)

    infer = sub.add_parser("infer", help="run a pipeline config on a wav file")
    infer.add_argument("--config", default="configs/keytap_baseline.yaml")
    infer.add_argument("--wav", required=True)
    infer.add_argument("--checkpoint", default=None)
    infer.set_defaults(func=cmd_infer)

    synth = sub.add_parser("synth", help="physics-informed synthetic sessions")
    synth.add_argument("--out", default="data/synthetic")
    synth.add_argument("--keyboards", default=None, help="comma list, default all profiles")
    synth.add_argument("--keys", default="common", help="'common', 'all', or a,b,c")
    synth.add_argument("--per-key", type=int, default=6)
    synth.add_argument("--phrase", default=None)
    synth.add_argument("--rir", action="store_true")
    synth.add_argument("--seed", type=int, default=0)
    synth.set_defaults(func=cmd_synth)

    transfer = sub.add_parser("transfer-check", help="synthetic-only model vs real sessions")
    transfer.add_argument("--synth", required=True)
    transfer.add_argument("--real", required=True)
    transfer.add_argument("--epochs", type=int, default=4)
    transfer.add_argument("--min-ratio", type=float, default=2.0)
    transfer.add_argument("--device", default="cpu")
    transfer.set_defaults(func=cmd_transfer)

    farm = sub.add_parser("farm", help="segment a wav, or download and segment YouTube audio")
    farm.add_argument("urls", nargs="*", help="YouTube URLs")
    farm.add_argument("--wav", nargs="*", default=None, help="local wavs to segment instead of downloading")
    farm.add_argument("--out", default="data/farmed")
    farm.set_defaults(func=cmd_farm)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
