"""Compatibility shim for the legacy Module 1 import path."""

from __future__ import annotations

from rhyme_rarity.core import (
    CMUDictLoader,
    DEFAULT_CMU_LOADER,
    DEFAULT_RARITY_MAP,
    EnhancedPhoneticAnalyzer,
    PhoneticMatch,
    PhraseComponents,
    RARITY_NOVELTY_WEIGHT,
    RARITY_SIMILARITY_WEIGHT,
    RhymeFeatureProfile,
    WordRarityMap,
    extract_phrase_components,
    get_cmu_rhymes,
)

__all__ = [
    "CMUDictLoader",
    "DEFAULT_CMU_LOADER",
    "DEFAULT_RARITY_MAP",
    "EnhancedPhoneticAnalyzer",
    "PhoneticMatch",
    "PhraseComponents",
    "RARITY_NOVELTY_WEIGHT",
    "RARITY_SIMILARITY_WEIGHT",
    "RhymeFeatureProfile",
    "WordRarityMap",
    "extract_phrase_components",
    "get_cmu_rhymes",
]
