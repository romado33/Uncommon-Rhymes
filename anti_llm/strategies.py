"""Strategy helpers for generating anti-LLM rhyme patterns."""

from __future__ import annotations

from typing import Callable, Dict, List, Mapping, Optional

from .dataclasses import AntiLLMPattern
from .repository import PatternRepository


def initialize_llm_weaknesses() -> Dict[str, Dict[str, object]]:
    return {
        "phonological_processing": {
            "description": "LLMs struggle with complex phoneme analysis",
            "target_patterns": ["consonant_clusters", "vowel_shifts", "silent_letters"],
            "difficulty_multiplier": 2.5,
        },
        "rare_word_combinations": {
            "description": "LLMs fail with low-frequency word pairs",
            "target_patterns": ["archaic_words", "slang_terms", "regional_variants"],
            "difficulty_multiplier": 3.0,
        },
        "cultural_context": {
            "description": "LLMs lack authentic cultural attribution",
            "target_patterns": ["artist_specific", "era_specific", "genre_specific"],
            "difficulty_multiplier": 2.2,
        },
        "multi_syllable_complexity": {
            "description": "LLMs struggle with complex syllable patterns",
            "target_patterns": ["internal_rhymes", "compound_words", "polysyllabic"],
            "difficulty_multiplier": 2.8,
        },
        "semantic_interference": {
            "description": "LLMs confused by semantic similarity vs phonetic",
            "target_patterns": ["semantic_opposites", "false_friends", "homophone_traps"],
            "difficulty_multiplier": 2.4,
        },
    }


def initialize_rarity_multipliers() -> Dict[str, float]:
    return {
        "ultra_rare": 4.0,
        "very_rare": 3.0,
        "rare": 2.0,
        "uncommon": 1.5,
        "common": 1.0,
    }


def initialize_cultural_weights() -> Dict[str, float]:
    return {
        "underground": 3.5,
        "regional": 3.0,
        "era_specific": 2.8,
        "artist_signature": 2.5,
        "mainstream": 1.0,
    }


def find_rare_combinations(
    repository: PatternRepository,
    source_word: str,
    limit: int,
    get_rarity: Callable[[str], float],
    cultural_weights: Mapping[str, float],
    attach_profile: Callable[[AntiLLMPattern], None],
    stats: Optional[Dict[str, int]] = None,
) -> List[AntiLLMPattern]:
    rows = repository.fetch_rare_combinations(source_word, limit * 3)
    patterns: List[AntiLLMPattern] = []

    for row in rows:
        target = str(row.get("target_word") or "").strip()
        if not target:
            continue

        rarity_score = get_rarity(target)
        cultural_sig = str(row.get("cultural_significance") or "")
        cultural_multiplier = cultural_weights.get(cultural_sig, 1.0)
        final_rarity = rarity_score * cultural_multiplier

        if final_rarity < 2.0:
            continue

        pattern = AntiLLMPattern(
            source_word=source_word,
            target_word=target,
            rarity_score=final_rarity,
            cultural_depth=f"{row.get('artist', '')} - {row.get('song_title', '')}",
            llm_weakness_type="rare_word_combinations",
            confidence=float(row.get("confidence_score") or 0.0),
        )
        attach_profile(pattern)
        patterns.append(pattern)

        if stats is not None:
            stats["rare_patterns_generated"] = stats.get("rare_patterns_generated", 0) + 1

        if len(patterns) >= limit:
            break

    return patterns


def find_phonological_challenges(
    repository: PatternRepository,
    source_word: str,
    limit: int,
    analyze_complexity: Callable[[str, str], float],
    attach_profile: Callable[[AntiLLMPattern], None],
    stats: Optional[Dict[str, int]] = None,
) -> List[AntiLLMPattern]:
    rows = repository.fetch_phonological_challenges(source_word, limit * 2)
    patterns: List[AntiLLMPattern] = []

    for row in rows:
        target = str(row.get("target_word") or "").strip()
        if not target:
            continue

        complexity_score = analyze_complexity(source_word, target)
        if complexity_score < 2.0:
            continue

        pattern = AntiLLMPattern(
            source_word=source_word,
            target_word=target,
            rarity_score=complexity_score,
            cultural_depth=f"{row.get('artist', '')} - {row.get('song_title', '')}",
            llm_weakness_type="phonological_processing",
            confidence=float(row.get("confidence_score") or 0.0),
        )
        attach_profile(pattern)
        patterns.append(pattern)

        if stats is not None:
            stats["phonological_challenges"] = stats.get("phonological_challenges", 0) + 1

        if len(patterns) >= limit:
            break

    return patterns


def find_cultural_depth_patterns(
    repository: PatternRepository,
    source_word: str,
    limit: int,
    cultural_weights: Mapping[str, float],
    attach_profile: Callable[[AntiLLMPattern], None],
    stats: Optional[Dict[str, int]] = None,
) -> List[AntiLLMPattern]:
    rows = repository.fetch_cultural_depth_patterns(source_word, limit * 2)
    patterns: List[AntiLLMPattern] = []

    for row in rows:
        target = str(row.get("target_word") or "").strip()
        if not target:
            continue

        cultural_sig = str(row.get("cultural_significance") or "")
        cultural_weight = cultural_weights.get(cultural_sig, 1.0)

        pattern = AntiLLMPattern(
            source_word=source_word,
            target_word=target,
            rarity_score=cultural_weight,
            cultural_depth=f"{row.get('artist', '')} - {row.get('song_title', '')} [{cultural_sig}]",
            llm_weakness_type="cultural_context",
            confidence=float(row.get("confidence_score") or 0.0),
        )
        attach_profile(pattern)
        patterns.append(pattern)

        if stats is not None:
            stats["cultural_patterns_found"] = stats.get("cultural_patterns_found", 0) + 1

        if len(patterns) >= limit:
            break

    return patterns


def find_complex_syllable_patterns(
    repository: PatternRepository,
    source_word: str,
    limit: int,
    calculate_syllable_complexity: Callable[[str], float],
    attach_profile: Callable[[AntiLLMPattern], None],
) -> List[AntiLLMPattern]:
    rows = repository.fetch_complex_syllable_patterns(source_word, limit * 2)
    patterns: List[AntiLLMPattern] = []

    for row in rows:
        target = str(row.get("target_word") or "").strip()
        if not target:
            continue

        syllable_complexity = calculate_syllable_complexity(target)
        if syllable_complexity < 2.0:
            continue

        pattern = AntiLLMPattern(
            source_word=source_word,
            target_word=target,
            rarity_score=syllable_complexity,
            cultural_depth=f"{row.get('artist', '')} - {row.get('song_title', '')}",
            llm_weakness_type="multi_syllable_complexity",
            confidence=float(row.get("confidence_score") or 0.0),
        )
        attach_profile(pattern)
        patterns.append(pattern)

    return patterns[:limit]


def effectiveness_score(pattern: AntiLLMPattern, multiplier_lookup: Mapping[str, float]) -> float:
    multiplier = multiplier_lookup.get(pattern.llm_weakness_type, 1.0)
    return pattern.rarity_score * pattern.confidence * multiplier


__all__ = [
    "initialize_llm_weaknesses",
    "initialize_rarity_multipliers",
    "initialize_cultural_weights",
    "find_rare_combinations",
    "find_phonological_challenges",
    "find_cultural_depth_patterns",
    "find_complex_syllable_patterns",
    "effectiveness_score",
]
