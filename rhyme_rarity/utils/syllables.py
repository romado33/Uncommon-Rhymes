"""Utilities for shared syllable estimation logic."""

from __future__ import annotations

import re


__all__ = ["estimate_syllable_count"]


def estimate_syllable_count(word: str) -> int:
    """Estimate the number of syllables in ``word`` using basic heuristics."""

    normalized = word.lower()
    vowel_groups = re.findall(r"[aeiou]+", normalized)
    syllable_count = len(vowel_groups)

    if normalized.endswith("e") and syllable_count > 1:
        syllable_count -= 1

    return max(1, syllable_count)
