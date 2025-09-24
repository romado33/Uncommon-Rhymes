"""Dataclasses and helpers describing phonetic feature profiles."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Tuple

from rhyme_rarity.utils.syllables import estimate_syllable_count

try:  # pragma: no cover - optional dependency
    import pronouncing  # type: ignore
except ImportError:  # pragma: no cover - gracefully handle missing package
    pronouncing = None  # type: ignore


@dataclass
class PhraseComponents:
    """Container describing the phonetic anchor for a phrase."""

    original: str
    tokens: List[str]
    normalized_tokens: List[str]
    normalized_phrase: str
    anchor: str
    anchor_display: str
    anchor_index: Optional[int]
    syllable_counts: List[int]
    total_syllables: int
    anchor_pronunciations: List[List[str]]


_TOKEN_PATTERN = re.compile(r"[A-Za-z']+")
_STRESS_PATTERN = re.compile(r"[12]")


@lru_cache(maxsize=8192)
def _cached_phrase_structure(
    phrase: str,
) -> Tuple[Tuple[str, ...], Tuple[str, ...], str, Tuple[int, ...], int]:
    """Return cached tokenisation and syllable data for ``phrase``."""

    original = phrase or ""
    tokens = tuple(match.group(0) for match in _TOKEN_PATTERN.finditer(original))
    normalized_tokens = tuple(token.lower() for token in tokens if token)
    normalized_phrase = " ".join(normalized_tokens)

    syllable_counts = tuple(
        estimate_syllable_count(token) for token in normalized_tokens
    )
    fallback_word = normalized_phrase or original.strip() or ""
    total_syllables = sum(syllable_counts) or estimate_syllable_count(fallback_word)

    return tokens, normalized_tokens, normalized_phrase, syllable_counts, total_syllables


def extract_phrase_components(
    phrase: str,
    cmu_loader: Optional["CMUDictLoader"] = None,
) -> PhraseComponents:
    """Return the final stressed token and syllable summary for ``phrase``."""

    original = phrase or ""
    (
        cached_tokens,
        cached_normalized,
        normalized_phrase,
        cached_syllables,
        total_syllables,
    ) = _cached_phrase_structure(original)

    tokens = list(cached_tokens)
    normalized_tokens = list(cached_normalized)
    syllable_counts = list(cached_syllables)

    anchor_index: Optional[int] = None
    anchor_pronunciations: List[List[str]] = []
    fallback_index: Optional[int] = (
        len(normalized_tokens) - 1 if normalized_tokens else None
    )
    fallback_pronunciations: List[List[str]] = []

    def _phones_with_stress(phones: Iterable[str]) -> bool:
        return any(_STRESS_PATTERN.search(phone) for phone in phones)

    for idx in range(len(normalized_tokens) - 1, -1, -1):
        token = normalized_tokens[idx]
        pronunciations: List[List[str]] = []

        if cmu_loader is not None:
            try:
                pronunciations = cmu_loader.get_pronunciations(token)
            except Exception:
                pronunciations = []

        if not pronunciations and pronouncing is not None:
            try:
                pronunciations = [
                    phones.split() for phones in pronouncing.phones_for_word(token)
                ]
            except Exception:
                pronunciations = []

        if pronunciations and not fallback_pronunciations:
            fallback_index = idx
            fallback_pronunciations = pronunciations

        if pronunciations and any(_phones_with_stress(phones) for phones in pronunciations):
            anchor_index = idx
            anchor_pronunciations = pronunciations
            break

    if anchor_index is None:
        anchor_index = fallback_index
        anchor_pronunciations = fallback_pronunciations

    anchor = normalized_tokens[anchor_index] if anchor_index is not None else ""
    anchor_display = (
        tokens[anchor_index]
        if anchor_index is not None and 0 <= anchor_index < len(tokens)
        else ""
    )

    return PhraseComponents(
        original=original,
        tokens=tokens,
        normalized_tokens=normalized_tokens,
        normalized_phrase=normalized_phrase,
        anchor=anchor,
        anchor_display=anchor_display,
        anchor_index=anchor_index,
        syllable_counts=syllable_counts,
        total_syllables=total_syllables,
        anchor_pronunciations=anchor_pronunciations,
    )


@dataclass
class RhymeFeatureProfile:
    """Rich description of rhyme craft elements between two words."""

    source_word: str
    target_word: str
    syllable_span: Tuple[int, int]
    stress_alignment: float
    stress_pattern_source: str
    stress_pattern_target: str
    vowel_skeleton_source: str
    vowel_skeleton_target: str
    consonant_tail_source: str
    consonant_tail_target: str
    assonance_score: float
    consonance_score: float
    internal_rhyme_score: float
    bradley_device: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "source_word": self.source_word,
            "target_word": self.target_word,
            "syllable_span": self.syllable_span,
            "stress_alignment": self.stress_alignment,
            "stress_pattern_source": self.stress_pattern_source,
            "stress_pattern_target": self.stress_pattern_target,
            "vowel_skeleton_source": self.vowel_skeleton_source,
            "vowel_skeleton_target": self.vowel_skeleton_target,
            "consonant_tail_source": self.consonant_tail_source,
            "consonant_tail_target": self.consonant_tail_target,
            "assonance_score": self.assonance_score,
            "consonance_score": self.consonance_score,
            "internal_rhyme_score": self.internal_rhyme_score,
            "bradley_device": self.bradley_device,
        }


@dataclass
class PhoneticMatch:
    """Represents a phonetic match between two words."""

    word1: str
    word2: str
    similarity_score: float
    phonetic_features: Dict[str, float]
    rhyme_type: str
    rarity_score: float = 0.0
    combined_score: float = 0.0
    feature_profile: Optional[RhymeFeatureProfile] = None


__all__ = [
    "PhraseComponents",
    "RhymeFeatureProfile",
    "PhoneticMatch",
    "extract_phrase_components",
    "pronouncing",
]
