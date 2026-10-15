from asce.decoding import sequence_accuracy
from asce.types import Prediction
from plugins.cpuhum.decoder import INSTRUCTIONS, ISAPriorDecoder
from plugins.keytap.decoder import NgramBeamDecoder


def _pred(probs):
    label = max(probs, key=probs.get)
    return Prediction(label=label, confidence=probs[label], probs=probs)


def test_ngram_corrects_teh_toward_the():
    frames = [
        _pred({"t": 0.55, "e": 0.25, "a": 0.20}),
        _pred({"e": 0.46, "h": 0.42, "a": 0.12}),
        _pred({"e": 0.62, "h": 0.20, "o": 0.18}),
    ]
    raw = [frame.label for frame in frames]
    assert raw == ["t", "e", "e"]
    decoded = NgramBeamDecoder(lm_weight=2.0).decode(frames)
    assert [item.label for item in decoded.predictions] == ["t", "h", "e"]
    assert decoded.text == "the"
    assert sequence_accuracy(["t", "h", "e"], raw) < sequence_accuracy(
        ["t", "h", "e"], [item.label for item in decoded.predictions]
    )


def test_forced_nonlinguistic_key_passes_through():
    frames = [
        _pred({"t": 0.9, "a": 0.1}),
        Prediction(label="backspace", confidence=0.95, probs={"backspace": 0.95, "a": 0.05}),
    ]
    decoded = NgramBeamDecoder().decode(frames)
    assert [item.label for item in decoded.predictions] == ["t", "backspace"]


def test_isa_decoder_stays_inside_the_instruction_set():
    frames = [
        _pred({"cmp": 0.4, "mov": 0.35, "nop": 0.25}),
        _pred({"mov": 0.45, "jmp": 0.4, "add": 0.15}),
    ]
    decoded = ISAPriorDecoder(prior_weight=2.0).decode(frames)
    assert all(item.label in INSTRUCTIONS for item in decoded.predictions)
    assert "jmp" in decoded.text or "mov" in decoded.text
