"""Key vocabulary and pynput-to-label normalization.

The label set is a QWERTY-ish slice of roughly fifty keys: letters, digits,
and the punctuation/modifiers a typing session actually produces.
"""

from __future__ import annotations

from typing import Any

LETTERS = list("abcdefghijklmnopqrstuvwxyz")
DIGITS = list("0123456789")
PUNCT = [
    "space",
    "enter",
    "backspace",
    "tab",
    "comma",
    "period",
    "minus",
    "slash",
    "semicolon",
    "apostrophe",
    "equal",
    "lbracket",
    "rbracket",
    "backslash",
    "grave",
    "lshift",
    "rshift",
]
VOCAB: list[str] = LETTERS + DIGITS + PUNCT

CHAR_TO_KEY = {
    ",": "comma",
    ".": "period",
    "-": "minus",
    "/": "slash",
    ";": "semicolon",
    "'": "apostrophe",
    "=": "equal",
    "[": "lbracket",
    "]": "rbracket",
    "\\": "backslash",
    "`": "grave",
    " ": "space",
}

KEY_TO_CHAR = {value: key for key, value in CHAR_TO_KEY.items()}
KEY_TO_CHAR.update({ch: ch for ch in LETTERS + DIGITS})
KEY_TO_CHAR["enter"] = "\n"
KEY_TO_CHAR["tab"] = "\t"
KEY_TO_CHAR["space"] = " "

# pynput Key.name values for non-character keys.
SPECIAL_NAMES = {
    "space": "space",
    "enter": "enter",
    "backspace": "backspace",
    "tab": "tab",
    "shift": "lshift",
    "shift_l": "lshift",
    "shift_r": "rshift",
}

LINGUISTIC_KEYS = set(LETTERS + DIGITS + ["space", "comma", "period", "minus", "apostrophe"])


def key_to_char(key: str) -> str:
    return KEY_TO_CHAR.get(key, "")


def normalize_key(key: Any) -> str | None:
    """Map a pynput key (or any object with ``char`` / ``name``) onto ``VOCAB``.

    Returns None for keys outside the research vocabulary (function keys, arrows).
    """

    char = getattr(key, "char", None)
    if isinstance(char, str) and char:
        lowered = char.lower()
        if lowered in CHAR_TO_KEY:
            return CHAR_TO_KEY[lowered]
        if lowered in LETTERS or lowered in DIGITS:
            return lowered
        return None
    name = getattr(key, "name", None)
    if isinstance(name, str):
        return SPECIAL_NAMES.get(name)
    return None


def label_map(labels: list[str] | None = None) -> dict[str, int]:
    chosen = list(labels) if labels is not None else list(VOCAB)
    return {label: index for index, label in enumerate(chosen)}
