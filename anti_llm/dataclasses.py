"""Shared dataclasses for the anti-LLM rhyme engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Set, Tuple


@dataclass
class AntiLLMPattern:
    """Represents a rhyme pattern designed to challenge LLM capabilities."""

    source_word: str
    target_word: str
    rarity_score: float
    cultural_depth: str
    llm_weakness_type: str
    confidence: float
    bradley_device: str = "undetermined"
    syllable_span: Tuple[int, int] = (0, 0)
    stress_alignment: float = 0.0
    feature_profile: Dict[str, Any] = field(default_factory=dict)
    prosody_profile: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SeedCandidate:
    """Normalized representation of Module 1 seed rhymes."""

    word: str
    rarity: float = 0.0
    combined: float = 0.0
    signatures: Set[str] = field(default_factory=set)
    feature_profile: Dict[str, Any] = field(default_factory=dict)
    prosody_profile: Dict[str, Any] = field(default_factory=dict)
    _fingerprint_cache: Set[str] = field(default_factory=set, init=False, repr=False)
    _suffix_cache: Set[str] = field(default_factory=set, init=False, repr=False)
    _signature_hint_cache: Set[str] = field(default_factory=set, init=False, repr=False)

    def normalized(self) -> str:
        return self.word.lower().strip()

    # Cached derivative helpers -------------------------------------------------

    def cache_fingerprint(self, fingerprint: Iterable[str]) -> None:
        self._fingerprint_cache = {str(value) for value in fingerprint if value}

    def cached_fingerprint(self) -> Set[str]:
        return set(self._fingerprint_cache)

    def cache_suffixes(self, suffixes: Iterable[str]) -> None:
        self._suffix_cache = {str(value) for value in suffixes if value}

    def cached_suffixes(self) -> Set[str]:
        return set(self._suffix_cache)

    def cache_signature_hints(self, hints: Iterable[str]) -> None:
        self._signature_hint_cache = {str(value) for value in hints if value}

    def cached_signature_hints(self) -> Set[str]:
        return set(self._signature_hint_cache)


__all__ = ["AntiLLMPattern", "SeedCandidate"]
