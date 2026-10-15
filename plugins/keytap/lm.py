"""Character n-gram prior estimated from a small built-in English lexicon.

Counts are relative weights, not a published corpus. They exist so beam
search can prefer 'the' over 'teh' the way the acoustic-cryptanalysis
papers used a language model to lift raw key accuracy.
"""

from __future__ import annotations

import math
from collections import Counter

# (word, weight). "th" words are heavy on purpose; words starting with "te" are absent
# so the classic the/teh ambiguity resolves toward English.
COMMON_WORDS: list[tuple[str, int]] = [
    ("the", 12000),
    ("they", 1800),
    ("them", 1400),
    ("then", 900),
    ("there", 1600),
    ("their", 1200),
    ("this", 2200),
    ("that", 3000),
    ("to", 7000),
    ("of", 6500),
    ("and", 6200),
    ("in", 4500),
    ("is", 3000),
    ("it", 2800),
    ("for", 2400),
    ("you", 2600),
    ("was", 1800),
    ("with", 2000),
    ("as", 1600),
    ("on", 1900),
    ("at", 1400),
    ("he", 1700),
    ("his", 1300),
    ("have", 1500),
    ("not", 1400),
    ("but", 1100),
    ("what", 900),
    ("all", 1000),
    ("were", 800),
    ("when", 800),
    ("we", 1200),
    ("your", 900),
    ("can", 800),
    ("said", 700),
    ("which", 800),
    ("she", 700),
    ("do", 900),
    ("how", 700),
    ("if", 800),
    ("will", 700),
    ("other", 500),
    ("about", 600),
    ("out", 700),
    ("many", 400),
    ("so", 900),
    ("some", 600),
    ("her", 600),
    ("would", 700),
    ("make", 500),
    ("like", 600),
    ("him", 500),
    ("into", 500),
    ("time", 600),
    ("has", 700),
    ("look", 400),
    ("more", 600),
    ("write", 200),
    ("go", 400),
    ("see", 500),
    ("no", 600),
    ("way", 400),
    ("could", 400),
    ("people", 400),
    ("my", 500),
    ("than", 400),
    ("first", 300),
    ("been", 400),
    ("call", 250),
    ("who", 400),
    ("now", 400),
    ("find", 250),
    ("long", 250),
    ("down", 300),
    ("day", 300),
    ("did", 300),
    ("get", 400),
    ("come", 300),
    ("made", 250),
    ("may", 300),
    ("part", 200),
    ("cat", 150),
    ("sat", 150),
    ("mat", 120),
    ("on", 1900),
]


class CharLM:
    def __init__(self, entries: list[tuple[str, int]] | None = None, k: float = 0.5) -> None:
        self.k = float(k)
        self.uni: Counter[str] = Counter()
        self.bi: Counter[tuple[str, str]] = Counter()
        self.tri: Counter[tuple[str, str, str]] = Counter()
        for word, weight in entries or COMMON_WORDS:
            tokens = ["<s>"] + list(word) + ["</s>"]
            for char in tokens:
                self.uni[char] += weight
            for left, right in zip(tokens, tokens[1:]):
                self.bi[(left, right)] += weight
            padded = ["<s>"] + tokens
            for a, b, c in zip(padded, padded[1:], padded[2:]):
                self.tri[(a, b, c)] += weight
        alphabet = set("abcdefghijklmnopqrstuvwxyz ")
        alphabet.update(self.uni)
        self.alphabet = sorted(alphabet)
        self.vocab_size = max(len(self.alphabet), 1)
        self._bi_denom: dict[str, int] = {}
        for (left, _right), weight in self.bi.items():
            self._bi_denom[left] = self._bi_denom.get(left, 0) + weight
        self._tri_denom: dict[tuple[str, str], int] = {}
        for (a, b, _c), weight in self.tri.items():
            self._tri_denom[(a, b)] = self._tri_denom.get((a, b), 0) + weight
        self._uni_total = sum(self.uni.values())

    def _smoothed(self, count: float, total: float) -> float:
        return (count + self.k) / (total + self.k * self.vocab_size)

    def log_prob(self, prev2: str, prev1: str, char: str) -> float:
        if not char:
            return -math.log(self.vocab_size)
        uni = self._smoothed(self.uni[char], self._uni_total)
        bi = self._smoothed(self.bi[(prev1, char)], self._bi_denom.get(prev1, 0))
        tri = self._smoothed(self.tri[(prev2, prev1, char)], self._tri_denom.get((prev2, prev1), 0))
        interpolated = 0.5 * tri + 0.3 * bi + 0.2 * uni
        return math.log(max(interpolated, 1e-12))
