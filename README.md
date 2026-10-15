# Acoustic Side-Channel Engine

ASCE is a pipeline with swappable stages. The keytap analyzer is the first plugin. A CPU-hum stub is the second, and it reuses capture and preprocessing.

```
AudioSource -> Preprocessor -> EventSegmenter -> FeatureExtractor -> Model -> Decoder
```

Stages register by name. A YAML config picks the implementation, so a new task is a new config plus the stages that actually differ (usually the segmenter and the decoder).

## Setup

Python 3.11 or newer. On this Mac, use 3.12 if 3.14 has no PyTorch wheel yet.

```bash
uv python install 3.12
uv venv --python 3.12
uv sync --extra dev --extra tracking
```

`ffmpeg` is only required for YouTube audio extraction (`brew install ffmpeg`). Recording, training, and the demo do not need it.

## Recording your own ground truth

The harness listens to the microphone and to the keyboard of the machine it runs on, and timestamps both from one monotonic clock. macOS will not deliver key events until you grant Input Monitoring (and sometimes Accessibility) to the terminal that launches Python:

System Settings -> Privacy & Security -> Input Monitoring.

```bash
uv run asce mics
uv run asce record --keyboard mechanical --mic laptop --distance-cm 30 --noise quiet --seconds 60 --out data/sessions
```

Each session is a folder with `audio.wav`, `events.jsonl`, and `meta.json` (keyboard, mic, distance, noise, and the audio-to-clock offset).

## Train and measure generalization

Synthetic sessions exist so the code path runs before you have recorded anything. They are not a substitute for the real validation set.

```bash
uv run asce synth --out data/synthetic --per-key 8
uv run asce train --data data/synthetic --keyboard mechanical --epochs 8 --checkpoint checkpoints/keytap.pt
uv run asce loso --data data/synthetic --epochs 5 --out experiments/loso.json
```

`train` restricted to one keyboard writes `experiments/m1_single_keyboard.json` with validation top-1. `loso` trains on every keyboard but one and reports the gap between in-distribution accuracy and held-out accuracy. That gap is the generalization number.

Other training flags:

- `--augment` convolves room impulse responses and mixes noise, with pitch and time jitter.
- `--adversarial` adds a gradient-reversal keyboard discriminator.
- `--init checkpoints/ssl.pt` warm-starts the CNN encoder from contrastive pretraining.
- `--device cpu` forces the CPU path when MPS numerics look wrong (`asce device` prints the comparison).

wav2vec2 and HuBERT configs are `configs/keytap_wav2vec.yaml` and `configs/keytap_hubert.yaml`. They fine-tune a small head on a frozen pretrained encoder. The first run downloads the weights.

```bash
uv run asce train --config configs/keytap_wav2vec.yaml --data data/sessions --device auto
```

## Farmed audio and self-supervised pretraining

Farmed clips have no key labels. Segment a local file, or download a typing video:

```bash
uv run asce farm --wav path/to/typing.wav --out data/farmed
uv run asce farm 'https://www.youtube.com/watch?v=...' --out data/farmed
uv run asce pretrain --data data/farmed/clips --checkpoint checkpoints/ssl.pt
uv run asce train --data data/sessions --init checkpoints/ssl.pt --checkpoint checkpoints/keytap.pt
```

## Language-model decoding

Raw per-key accuracy is only half the keytap problem. `ngram_beam` runs beam search with an English character n-gram.

```bash
uv run asce decode-eval --data data/sessions --checkpoint checkpoints/keytap.pt
```

The report is raw accuracy, decoded accuracy, and the lift.

## Synthetic data, with a gate

Physics-informed synthesis records nothing: it renders a press/release transient per key and can convolve a synthetic room impulse. A conditional spectrogram generator lives in `asce.synth.generative` for augmentation only.

Do not scale synthetic data up unless a model trained only on it beats chance on real sessions:

```bash
uv run asce transfer-check --synth data/synthetic --real data/sessions
```

The command exits non-zero when transfer fails.

## Live demo

```bash
cd demo && npm install && npm start
```

Open `http://127.0.0.1:8787`. The page shows a waveform, a scrolling spectrogram, and the keys the local worker emits. Without a checkpoint it marks onsets as `?` rather than inventing letters. Point it at a trained model with `ASCE_CHECKPOINT=checkpoints/keytap.pt`.

## CPU-hum stub

`configs/cpuhum.yaml` swaps in fixed windows, band energy, a small MLP, and an instruction-frequency decoder. Capture and preprocessing stay the same.

```bash
uv run asce infer --config configs/cpuhum.yaml --wav some.wav
```

## Layout

- `asce/stages` registered implementations
- `asce/pipeline.py` offline run and streaming chunks
- `asce/data` sessions, the recorder, datasets, YouTube farming
- `asce/train` fit, leave-one-keyboard-out, augmentation, contrastive pretraining
- `plugins/keytap` English n-gram decoder
- `plugins/cpuhum` instruction prior and band-energy features
- `docs/roadmap.md` the papers the pipeline is built against

Tracking defaults to a JSONL file in `experiments/`. Set `WANDB_API_KEY` and `--tracker wandb`, or pass `--tracker mlflow`, to mirror the same metrics.
