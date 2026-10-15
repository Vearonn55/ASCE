# Roadmap

The engine is a shared skeleton. Keytap is the first task. CPU hum is the second, and it should not grow a second pipeline.

```
Capture -> Preprocess -> Segment -> Features -> Model -> Decode -> Output
```

Raw classification of isolated keystrokes is half the keytap problem. The other half is language-model decoding: a character n-gram (or a dictionary) turning a noisy key sequence into text. That lives in the decoder stage from the start, because the CPU-hum task will swap the prior (instruction frequencies) without changing the beam-search pattern.

## Phase 0 — literature

Acoustic keystroke inference is an existing research area. These are the papers the design is checked against, not a claim that this repo reproduces their numbers.

- Asonov and Agrawal, 2004, "Keyboard Acoustic Emanations." Established that individual keys have distinguishable acoustic signatures, including a press and a release.
- Zhuang, Zhou, and Tygar, 2009, "Keyboard Acoustic Emanations Revisited." Unsupervised clustering plus a language model. The result that matters here: language-model correction is what moves a mediocre per-key classifier into a usable transcription.
- Compagno, Tsudik, et al., 2017, "Don't Skype & Type!" Keystrokes remain informative after VoIP compression. Relevant if a later plugin targets a call recording instead of a local microphone.
- Harrison, Toreini, and Mehrnezhad, 2023, "A Practical Deep Learning-Based Acoustic Side Channel Attack." CNN on spectrograms, phone and Zoom microphones, high accuracy when the keyboard is known. The open problem this repo treats as primary is the other one: a keyboard the model has not been trained on.

What to carry into the code:

- Log-mel spectrograms are the default representation. MFCCs stay available as the cheap baseline.
- The segmenter should keep the press and the release inside one window. They are two transients of one keystroke.
- Report leave-one-keyboard-out accuracy, not only same-keyboard accuracy. Switch mechanism (mechanical, membrane, chiclet) moves the spectrum more than the microphone does.
- Put the language model in the decoder, and always publish raw accuracy next to decoded accuracy.

## Milestones

1. Engine skeleton, self-recorded sessions (pynput + sounddevice, one clock), CNN on log-mel, single-keyboard top-1.
2. More keyboards. Leave-one-keyboard-out. The generalization gap is the baseline that later milestones have to move.
3. wav2vec2 or HuBERT backbone, frozen or lightly fine-tuned, MPS with a CPU fallback. Re-measure the gap.
4. Farmed unlabeled audio, contrastive pretraining, then fine-tune the head on the small labeled set. Re-measure.
5. Room-impulse and noise augmentation, plus the n-gram beam decoder. Compare raw and decoded accuracy.
6. Physics-informed synthesis (and an optional spectrogram generator) only as gap-filling. A model trained on synthetic data must transfer to the real validation sessions or the synthetic data is discarded.
7. Local demo: waveform, spectrogram, streamed key predictions.

Domain-adversarial training (a keyboard discriminator behind a gradient reversal layer) is implemented and off by default. Turn it on after the leave-one-keyboard-out gap is measured and is still large.

## Hardware note

The intended training machine is a single Mac with unified memory (M-series, on the order of 48 GB). Fine-tuning a ~95M pretrained encoder fits. Training a large transformer from scratch does not, and the labeled set would not support it anyway. `asce device` compares a tiny convolution on MPS and CPU; use `--device cpu` when they disagree.
