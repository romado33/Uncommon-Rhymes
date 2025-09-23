"""Compatibility shim for the legacy Module 2 import path."""

from __future__ import annotations

from anti_llm import AntiLLMPattern, AntiLLMRhymeEngine, SeedCandidate, safe_float
from rhyme_rarity.utils.syllables import estimate_syllable_count

__all__ = [
    "AntiLLMRhymeEngine",
    "AntiLLMPattern",
    "SeedCandidate",
    "estimate_syllable_count",
    "safe_float",
]
