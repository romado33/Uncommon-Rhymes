"""Seed expansion helpers for the anti-LLM engine."""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Set

from .dataclasses import AntiLLMPattern, SeedCandidate
from .repository import PatternRepository


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_seed_candidates(
    module1_seeds: Optional[List[Any]],
    get_word_rarity: Callable[[str], float],
    value_sanitizer: Callable[[Any, float], float] = safe_float,
) -> List[SeedCandidate]:
    normalized: List[SeedCandidate] = []
    if not module1_seeds:
        return normalized

    seen: Set[str] = set()

    for raw_seed in module1_seeds:
        word: Optional[str] = None
        rarity_value = 0.0
        combined_value = 0.0
        signatures: Set[str] = set()
        feature_profile: Dict[str, Any] = {}
        prosody_profile: Dict[str, Any] = {}

        if isinstance(raw_seed, str):
            word = raw_seed
        elif isinstance(raw_seed, dict):
            word = (
                raw_seed.get("word")
                or raw_seed.get("target_word")
                or raw_seed.get("candidate")
            )
            rarity_value = value_sanitizer(
                raw_seed.get("rarity")
                or raw_seed.get("rarity_score")
                or raw_seed.get("module1_rarity"),
                default=0.0,
            )
            combined_value = value_sanitizer(
                raw_seed.get("combined")
                or raw_seed.get("combined_score")
                or raw_seed.get("confidence"),
                default=0.0,
            )
            signature_values = (
                raw_seed.get("signatures")
                or raw_seed.get("target_rhyme_signatures")
                or raw_seed.get("matched_signatures")
            )
            if signature_values:
                if isinstance(signature_values, (set, list, tuple)):
                    signatures.update({str(sig) for sig in signature_values if sig})
                else:
                    signatures.add(str(signature_values))
            feature_candidate = raw_seed.get("feature_profile")
            if isinstance(feature_candidate, dict):
                feature_profile = dict(feature_candidate)
            prosody_candidate = raw_seed.get("prosody_profile")
            if isinstance(prosody_candidate, dict):
                prosody_profile = dict(prosody_candidate)
        else:
            try:
                word = raw_seed[0]
                if len(raw_seed) > 2:
                    rarity_value = value_sanitizer(raw_seed[2], default=0.0)
                if len(raw_seed) > 3:
                    combined_value = value_sanitizer(raw_seed[3], default=0.0)
            except Exception:
                continue

        if not word:
            continue

        word_str = str(word).strip()
        if not word_str:
            continue

        normalized_word = word_str.lower()
        if normalized_word in seen:
            continue

        seen.add(normalized_word)

        if rarity_value <= 0:
            rarity_value = get_word_rarity(normalized_word)

        normalized.append(
            SeedCandidate(
                word=word_str,
                rarity=rarity_value,
                combined=combined_value,
                signatures=set(signatures),
                feature_profile=feature_profile,
                prosody_profile=prosody_profile,
            )
        )

    normalized.sort(key=lambda seed: (seed.rarity, seed.combined), reverse=True)
    return normalized[:8]


def normalize_module1_candidates(
    candidates: Optional[List[Any]],
    value_sanitizer: Callable[[Any, float], float] = safe_float,
) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    if not candidates:
        return normalized

    seen: Set[str] = set()

    for candidate in candidates:
        word: Optional[str] = None
        similarity = 0.0
        combined = 0.0
        rarity = 0.0

        if isinstance(candidate, dict):
            word = (
                candidate.get("word")
                or candidate.get("target")
                or candidate.get("candidate")
            )
            similarity = value_sanitizer(
                candidate.get("similarity") or candidate.get("score"),
                default=0.0,
            )
            combined = value_sanitizer(
                candidate.get("combined")
                or candidate.get("combined_score")
                or candidate.get("confidence"),
                default=similarity,
            )
            rarity = value_sanitizer(
                candidate.get("rarity") or candidate.get("rarity_score"),
                default=0.0,
            )
        else:
            try:
                word = candidate[0]
                if len(candidate) > 1:
                    similarity = value_sanitizer(candidate[1], default=0.0)
                if len(candidate) > 2:
                    combined = value_sanitizer(candidate[2], default=similarity)
                if len(candidate) > 3:
                    rarity = value_sanitizer(candidate[3], default=0.0)
            except Exception:
                continue

        if not word:
            continue

        key = str(word).strip().lower()
        if not key or key in seen:
            continue

        seen.add(key)
        normalized.append(
            {
                "candidate": str(word).strip(),
                "similarity": similarity,
                "combined": combined,
                "rarity": rarity,
            }
        )

    return normalized


def extract_suffixes(word: str) -> Set[str]:
    cleaned = re.sub(r"[^a-z]", "", (word or "").lower())
    suffixes: Set[str] = set()
    for length in (4, 3, 2):
        if len(cleaned) >= length:
            suffixes.add(cleaned[-length:])
    return suffixes


def phonetic_fingerprint(word: str) -> Set[str]:
    cleaned = re.sub(r"[^a-z]", "", (word or "").lower())
    if not cleaned:
        return set()

    vowels = re.findall(r"[aeiou]+", cleaned)
    fingerprint: Set[str] = set()

    if len(cleaned) >= 3:
        fingerprint.add(f"end:{cleaned[-3:]}")
    if len(cleaned) >= 4:
        fingerprint.add(f"tail:{cleaned[-4:]}")
    if vowels:
        fingerprint.add(f"v:{vowels[-1]}")
    if len(vowels) >= 2:
        fingerprint.add(f"vv:{vowels[-2]}>{vowels[-1]}")

    return fingerprint


def expand_from_seed_candidates(
    repository: PatternRepository,
    source_word: str,
    seeds: List[SeedCandidate],
    limit: int,
    signature_hints: Set[str],
    seen_targets: Set[str],
    get_word_rarity: Callable[[str], float],
    analyze_phonological_complexity: Callable[[str, str], float],
    calculate_syllable_complexity: Callable[[str], float],
    attach_profile: Callable[[AntiLLMPattern], None],
    ensure_seed_resources: Callable[[], None],
    cmu_candidates_fn: Optional[Callable[[str, int, Any], List[Any]]],
    seed_analyzer: Any,
    stats: Optional[Dict[str, int]] = None,
    value_sanitizer: Callable[[Any, float], float] = safe_float,
    fetch_neighbors: Optional[Callable[[str, int], List[Dict[str, Any]]]] = None,
    fetch_suffix_matches: Optional[Callable[[str, int], List[Dict[str, Any]]]] = None,
    suffix_extractor: Optional[Callable[[str], Set[str]]] = None,
    module1_normalizer: Optional[Callable[[Optional[List[Any]]], List[Dict[str, Any]]]] = None,
    fingerprint_fn: Optional[Callable[[str], Set[str]]] = None,
) -> List[AntiLLMPattern]:
    results: List[AntiLLMPattern] = []
    if not seeds or limit <= 0:
        return results

    ensure_seed_resources()

    per_seed_limit = max(1, (limit // max(len(seeds), 1)) + 1)

    fingerprint_source = fingerprint_fn or phonetic_fingerprint
    suffix_source = suffix_extractor or extract_suffixes

    for seed in seeds:
        if len(results) >= limit:
            break

        seed_word = seed.word
        normalized_seed = seed.normalized()

        seed_fingerprint = seed.cached_fingerprint()
        if not seed_fingerprint:
            seed_fingerprint = fingerprint_source(seed_word)
            seed.cache_fingerprint(seed_fingerprint)

        signature_hint_cache = seed.cached_signature_hints()
        if not signature_hint_cache:
            combined = set(seed.signatures)
            combined.update(seed_fingerprint)
            stress_hint = None
            if seed.feature_profile:
                stress_hint = seed.feature_profile.get("stress_alignment")
                device_hint = seed.feature_profile.get("bradley_device")
            else:
                device_hint = None
            if isinstance(stress_hint, (int, float)):
                combined.add(f"stress::{round(float(stress_hint), 2)}")
            if device_hint:
                combined.add(f"device::{str(device_hint).lower()}")
            seed.cache_signature_hints(combined)
            signature_hint_cache = combined

        combined_signatures = set(signature_hints)
        combined_signatures.update(signature_hint_cache)

        if seed.feature_profile:
            stress_hint = seed.feature_profile.get("stress_alignment")
            if isinstance(stress_hint, (int, float)):
                combined_signatures.add(f"stress::{round(float(stress_hint), 2)}")
            device_hint = seed.feature_profile.get("bradley_device")
            if device_hint:
                combined_signatures.add(f"device::{str(device_hint).lower()}")

        candidate_dicts: List[Dict[str, Any]] = []
        neighbors_fetcher = fetch_neighbors or (
            lambda seed, limit: repository.fetch_seed_neighbors(seed, limit)
        )
        candidate_dicts.extend(
            neighbors_fetcher(normalized_seed, per_seed_limit * 2)
        )

        suffix_matches = fetch_suffix_matches or (
            lambda suffix, limit: repository.fetch_suffix_matches(suffix, limit)
        )

        seed_suffixes = seed.cached_suffixes()
        if not seed_suffixes:
            seed_suffixes = suffix_source(seed_word)
            seed.cache_suffixes(seed_suffixes)

        for suffix in seed_suffixes:
            candidate_dicts.extend(suffix_matches(suffix, per_seed_limit))

        if cmu_candidates_fn is not None:
            try:
                raw_candidates = cmu_candidates_fn(
                    seed_word,
                    limit=per_seed_limit * 2,
                    analyzer=seed_analyzer,
                )
            except Exception:
                raw_candidates = []

            candidate_dicts.extend(
                (module1_normalizer or (lambda candidates: normalize_module1_candidates(candidates, value_sanitizer)))(
                    raw_candidates
                )
            )

        local_seen: Set[str] = set()

        for candidate in candidate_dicts:
            if len(results) >= limit:
                break

            target = str(candidate.get("candidate") or "").strip()
            if not target:
                continue

            normalized_target = target.lower()
            if (
                normalized_target in seen_targets
                or normalized_target in local_seen
                or normalized_target == normalized_seed
                or normalized_target == source_word
            ):
                continue

            fingerprint = fingerprint_source(target)
            if combined_signatures and fingerprint and not (fingerprint & combined_signatures):
                continue

            base_rarity = get_word_rarity(normalized_target)
            base_rarity = max(
                base_rarity,
                value_sanitizer(candidate.get("rarity"), default=0.0),
            )

            seed_boost = min(1.5, seed.rarity * 0.5)
            complexity = analyze_phonological_complexity(source_word, normalized_target)
            syllable_complexity = calculate_syllable_complexity(normalized_target)

            final_score = min(
                5.0,
                base_rarity + seed_boost + complexity * 0.6 + syllable_complexity * 0.3,
            )

            stress_hint = (
                seed.feature_profile.get("stress_alignment")
                if seed.feature_profile
                else None
            )
            if isinstance(stress_hint, (int, float)):
                final_score += min(0.4, max(0.0, float(stress_hint)) * 0.3)

            if final_score < 1.6:
                continue

            weakness = "rare_word_combinations"
            if syllable_complexity >= 2.5:
                weakness = "multi_syllable_complexity"
            elif complexity >= 2.0:
                weakness = "phonological_processing"

            confidence = value_sanitizer(
                candidate.get("confidence"),
                default=value_sanitizer(candidate.get("combined"), default=0.7),
            )
            similarity_hint = value_sanitizer(
                candidate.get("phonetic_similarity"),
                default=value_sanitizer(candidate.get("similarity"), default=0.0),
            )
            if similarity_hint:
                confidence = max(confidence, similarity_hint)
            if confidence <= 0:
                confidence = 0.65 + min(0.25, complexity * 0.1)

            cultural_depth = (
                candidate.get("context")
                or candidate.get("source")
                or f"Seed cascade via {seed_word}"
            )

            pattern = AntiLLMPattern(
                source_word=source_word,
                target_word=target,
                rarity_score=final_score,
                cultural_depth=str(cultural_depth),
                llm_weakness_type=weakness,
                confidence=confidence,
            )

            attach_profile(pattern)

            results.append(pattern)

            if stats is not None:
                stats["seed_expansions"] = stats.get("seed_expansions", 0) + 1

            seen_targets.add(normalized_target)
            local_seen.add(normalized_target)

    return results[:limit]


__all__ = [
    "safe_float",
    "normalize_seed_candidates",
    "normalize_module1_candidates",
    "extract_suffixes",
    "phonetic_fingerprint",
    "expand_from_seed_candidates",
]
