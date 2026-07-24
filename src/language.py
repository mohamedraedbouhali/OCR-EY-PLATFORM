"""Lightweight language detection and RTL helpers for FR / EN / AR text."""

from __future__ import annotations

import re

_ARABIC_RE = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]")
_LATIN_RE = re.compile(r"[A-Za-zÀ-ſ]")

_FRENCH_MARKERS = {
    "le", "la", "les", "des", "une", "un", "et", "est", "dans", "pour",
    "que", "qui", "avec", "sur", "pas", "par", "vous", "nous", "cette", "être",
}
_ENGLISH_MARKERS = {
    "the", "and", "of", "to", "in", "is", "for", "that", "with", "this",
    "are", "was", "from", "have", "not", "you", "will", "been", "which",
}


def detect_languages(text: str) -> list[str]:
    """Return ISO codes of the languages present, ordered by prevalence."""
    if not text.strip():
        return []

    arabic_chars = len(_ARABIC_RE.findall(text))
    latin_chars = len(_LATIN_RE.findall(text))
    scores: dict[str, float] = {}

    if arabic_chars > 20 or (arabic_chars > 0 and arabic_chars >= latin_chars * 0.2):
        scores["ar"] = arabic_chars

    if latin_chars > 0:
        words = {w.lower() for w in re.findall(r"[A-Za-zÀ-ſ']+", text)}
        fr_hits = len(words & _FRENCH_MARKERS)
        en_hits = len(words & _ENGLISH_MARKERS)
        accented = len(re.findall(r"[àâçéèêëîïôùûüœÀÂÇÉÈÊËÎÏÔÙÛÜŒ]", text))
        fr_score = fr_hits * 2 + accented
        en_score = en_hits * 2
        if fr_score > 0 and fr_score >= en_score * 0.5:
            scores["fr"] = latin_chars * fr_score / max(fr_score + en_score, 1)
        if en_score > 0 and en_score >= fr_score * 0.5:
            scores["en"] = latin_chars * en_score / max(fr_score + en_score, 1)
        if fr_score == 0 and en_score == 0:
            scores["en"] = latin_chars * 0.5  # latin script, undetermined → default en

    return sorted(scores, key=scores.get, reverse=True)


def is_mostly_rtl(text: str) -> bool:
    arabic = len(_ARABIC_RE.findall(text))
    latin = len(_LATIN_RE.findall(text))
    return arabic > latin
