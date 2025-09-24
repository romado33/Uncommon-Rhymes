"""Shared dataclasses for the anti-LLM rhyme engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Set, Tuple


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
    fingerprint: Set[str] = field(default_factory=set)
    suffixes: Set[str] = field(default_factory=set)
    signature_hints: Set[str] = field(default_factory=set)

    def normalized(self) -> str:
        return self.word.lower().strip()


__all__ = ["AntiLLMPattern", "SeedCandidate"]
