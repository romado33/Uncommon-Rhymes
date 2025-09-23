"""Core phonetic analysis utilities for RhymeRarity."""

from .analyzer import (
    EnhancedPhoneticAnalyzer,
    RARITY_NOVELTY_WEIGHT,
    RARITY_SIMILARITY_WEIGHT,
    get_cmu_rhymes,
)
from .cmudict_loader import CMUDictLoader, DEFAULT_CMU_LOADER
from .feature_profile import (
    PhoneticMatch,
    PhraseComponents,
    RhymeFeatureProfile,
    extract_phrase_components,
)
from .rarity_map import DEFAULT_RARITY_MAP, WordRarityMap

__all__ = [
    "CMUDictLoader",
    "DEFAULT_CMU_LOADER",
    "WordRarityMap",
    "DEFAULT_RARITY_MAP",
    "EnhancedPhoneticAnalyzer",
    "RhymeFeatureProfile",
    "PhraseComponents",
    "PhoneticMatch",
    "extract_phrase_components",
    "get_cmu_rhymes",
    "RARITY_SIMILARITY_WEIGHT",
    "RARITY_NOVELTY_WEIGHT",
]
