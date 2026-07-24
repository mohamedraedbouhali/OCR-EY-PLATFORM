"""OCR quality metrics: Character Error Rate (CER) and Word Error Rate (WER)."""

from __future__ import annotations

import re
import unicodedata


def _levenshtein(a: list | str, b: list | str) -> int:
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, item_a in enumerate(a, start=1):
        current = [i]
        for j, item_b in enumerate(b, start=1):
            cost = 0 if item_a == item_b else 1
            current.append(min(previous[j] + 1,        # deletion
                               current[j - 1] + 1,     # insertion
                               previous[j - 1] + cost))  # substitution
        previous = current
    return previous[-1]


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", " ", text).strip()


def cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate — 0.0 is perfect; can exceed 1.0."""
    ref = _normalize(reference)
    if not ref:
        return 0.0 if not _normalize(hypothesis) else 1.0
    return _levenshtein(ref, _normalize(hypothesis)) / len(ref)


def wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate — 0.0 is perfect; can exceed 1.0."""
    ref_words = _normalize(reference).split()
    if not ref_words:
        return 0.0 if not _normalize(hypothesis).split() else 1.0
    return _levenshtein(ref_words, _normalize(hypothesis).split()) / len(ref_words)


def accuracy_summary(reference: str, hypothesis: str) -> dict:
    c, w = cer(reference, hypothesis), wer(reference, hypothesis)
    return {
        "cer": round(c, 4),
        "wer": round(w, 4),
        "char_accuracy": round(max(0.0, 1 - c) * 100, 2),
        "word_accuracy": round(max(0.0, 1 - w) * 100, 2),
    }
