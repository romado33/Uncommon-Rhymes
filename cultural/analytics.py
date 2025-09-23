"""Heuristic helpers for cultural intelligence features."""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Set

from rhyme_rarity.utils.profile import normalize_profile_dict
from rhyme_rarity.utils.syllables import estimate_syllable_count

from .profiles import CulturalContext


def estimate_syllables(word: str, analyzer: Optional[Any]) -> int:
    if analyzer is not None and hasattr(analyzer, "estimate_syllables"):
        try:
            return int(analyzer.estimate_syllables(word))
        except Exception:
            pass
    return estimate_syllable_count(word)


def approximate_rhyme_signature(word: str, syllable_estimator) -> Set[str]:
    cleaned = re.sub(r"[^a-z]", "", (word or "").lower())
    if not cleaned:
        return set()
    vowels = re.findall(r"[aeiou]+", cleaned)
    last_vowel = vowels[-1] if vowels else ""
    ending = cleaned[-3:] if len(cleaned) >= 3 else cleaned
    syllables = syllable_estimator(cleaned)
    components: List[str] = []
    if last_vowel:
        components.append(f"v:{last_vowel}")
    components.append(f"e:{ending}")
    signatures = {"|".join(components)}
    if syllables:
        signatures.add("|".join(components + [f"s:{syllables}"]))
    return signatures


def derive_rhyme_signatures(
    word: str,
    analyzer: Optional[Any],
    approximator: Callable[[str], Set[str]],
) -> Set[str]:
    normalized = (word or "").strip().lower()
    if not normalized:
        return set()
    signatures: Set[str] = set()
    loader = getattr(analyzer, "cmu_loader", None) if analyzer else None
    if loader is not None:
        try:
            signatures.update({sig for sig in loader.get_rhyme_parts(normalized) if sig})
        except Exception:
            pass
    signatures.update(approximator(normalized))
    return signatures


def phrase_syllable_vector(text: Optional[str], estimator) -> List[int]:
    if not text:
        return []
    tokens = re.findall(r"[a-zA-Z']+", text)
    return [estimator(token.lower()) for token in tokens if token]


def prosody_profile(
    source_context: Optional[str],
    target_context: Optional[str],
    feature_profile: Optional[Dict[str, Any]],
    estimator,
) -> Dict[str, Any]:
    source_vector = phrase_syllable_vector(source_context, estimator)
    target_vector = phrase_syllable_vector(target_context, estimator)

    total_source = sum(source_vector)
    total_target = sum(target_vector)

    avg_source = total_source / max(len(source_vector), 1)
    avg_target = total_target / max(len(target_vector), 1)

    cadence_ratio = 0.0
    if total_source and total_target:
        cadence_ratio = total_target / total_source

    stress_alignment = None
    if feature_profile:
        stress_alignment = feature_profile.get("stress_alignment")

    complexity_tag = "steady"
    if avg_target >= 3 or avg_source >= 3:
        complexity_tag = "polysyllabic"
    if len(target_vector) >= 3 and any(count >= 4 for count in target_vector):
        complexity_tag = "dense"

    return {
        "source_total_syllables": total_source,
        "target_total_syllables": total_target,
        "source_average_syllables": avg_source,
        "target_average_syllables": avg_target,
        "cadence_ratio": cadence_ratio,
        "stress_alignment": stress_alignment,
        "complexity_tag": complexity_tag,
        "source_vector": source_vector,
        "target_vector": target_vector,
    }


def cultural_rarity_score(context: CulturalContext) -> float:
    base = 1.0
    if context.cultural_significance in {"underground", "regional"}:
        base += 0.75
    if "legendary" in context.style_characteristics:
        base += 1.0
    if context.genre in {"hip-hop", "rap"} and context.era == "golden_age":
        base += 0.5
    if context.regional_origin in {"detroit", "compton", "brooklyn"}:
        base += 0.25
    return base


def evaluate_rhyme_alignment(
    analyzer: Optional[Any],
    source_word: str,
    target_word: str,
    threshold: Optional[float] = None,
    rhyme_signatures: Optional[Set[str]] = None,
    source_context: Optional[str] = None,
    target_context: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    normalized_source = (source_word or "").strip().lower()
    normalized_target = (target_word or "").strip().lower()
    if not normalized_source or not normalized_target:
        return None

    source_signature_set = {sig for sig in (rhyme_signatures or set()) if sig}
    target_signatures = derive_rhyme_signatures(
        normalized_target,
        analyzer,
        lambda word: approximate_rhyme_signature(word, lambda w: estimate_syllable_count(w)),
    )
    signature_matches = (
        sorted(target_signatures.intersection(source_signature_set))
        if source_signature_set
        else []
    )

    if source_signature_set and target_signatures and not signature_matches:
        return None

    similarity: Optional[float] = None
    rarity_value: Optional[float] = None
    combined_value: Optional[float] = None
    rhyme_type: Optional[str] = None
    features: Dict[str, Any] = {}
    feature_profile_dict: Dict[str, Any] = {}

    if analyzer:
        try:
            match = analyzer.analyze_rhyme_pattern(normalized_source, normalized_target)
        except Exception:
            match = None
        if match:
            similarity = float(match.similarity_score)
            features = dict(match.phonetic_features)
            rhyme_type = match.rhyme_type
            profile_obj = getattr(match, "feature_profile", None)
        else:
            profile_obj = None
            try:
                similarity = float(
                    analyzer.get_phonetic_similarity(normalized_source, normalized_target)
                )
            except Exception:
                similarity = 0.0
            try:
                rhyme_type = analyzer.classify_rhyme_type(
                    normalized_source, normalized_target, similarity
                )
            except Exception:
                rhyme_type = None
            features = {}
        if profile_obj is None:
            try:
                profile_obj = analyzer.derive_rhyme_profile(
                    normalized_source,
                    normalized_target,
                    similarity=similarity,
                    rhyme_type=rhyme_type,
                )
            except Exception:
                profile_obj = None
        feature_profile_dict = normalize_profile_dict(profile_obj)
        try:
            rarity_value = float(analyzer.get_rarity_score(normalized_target))
        except Exception:
            rarity_value = None
        if rarity_value is not None and similarity is not None:
            try:
                combined_value = float(
                    analyzer.combine_similarity_and_rarity(similarity, rarity_value)
                )
            except Exception:
                combined_value = None
        if threshold is not None and similarity is not None:
            effective_threshold = max(0.0, float(threshold) - 0.02)
            if similarity < effective_threshold:
                return None

    prosody = prosody_profile(
        source_context,
        target_context,
        feature_profile_dict,
        lambda token: estimate_syllable_count(token),
    )

    return {
        "similarity": similarity,
        "rarity": rarity_value,
        "combined": combined_value,
        "rhyme_type": rhyme_type,
        "signature_matches": signature_matches,
        "target_signatures": sorted(target_signatures),
        "features": features,
        "feature_profile": feature_profile_dict,
        "prosody_profile": prosody,
    }


def aggregate_cultural_distribution(rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    distribution: Dict[str, int] = {}
    for row in rows:
        key = str(row.get("cultural_significance") or "unknown").lower()
        distribution[key] = distribution.get(key, 0) + 1
    return distribution


__all__ = [
    "estimate_syllables",
    "approximate_rhyme_signature",
    "derive_rhyme_signatures",
    "phrase_syllable_vector",
    "prosody_profile",
    "cultural_rarity_score",
    "evaluate_rhyme_alignment",
    "aggregate_cultural_distribution",
]
