"""Anti-LLM rhyme generation engine."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set

from profile_utils import normalize_profile_dict
from syllable_utils import estimate_syllable_count

from .dataclasses import AntiLLMPattern, SeedCandidate
from .repository import PatternRepository, SQLitePatternRepository
from .seed_expansion import (
    expand_from_seed_candidates,
    normalize_module1_candidates,
    normalize_seed_candidates,
    phonetic_fingerprint,
    safe_float,
    extract_suffixes,
)
from .strategies import (
    effectiveness_score,
    find_complex_syllable_patterns,
    find_cultural_depth_patterns,
    find_phonological_challenges,
    find_rare_combinations,
    initialize_cultural_weights,
    initialize_llm_weaknesses,
    initialize_rarity_multipliers,
)


class AntiLLMRhymeEngine:
    """Enhanced rhyme engine specifically designed to outperform LLMs."""

    def __init__(
        self,
        db_path: str = "patterns.db",
        phonetic_analyzer: Optional[Any] = None,
        rarity_map: Optional[Any] = None,
        repository: Optional[PatternRepository] = None,
    ) -> None:
        self.db_path = db_path
        self.phonetic_analyzer = phonetic_analyzer
        self.repository: PatternRepository = repository or SQLitePatternRepository(db_path)
        self._rarity_map: Optional[Any] = None

        if rarity_map is not None and hasattr(rarity_map, "get_rarity"):
            self._rarity_map = rarity_map
        else:
            analyzer_map = getattr(self.phonetic_analyzer, "rarity_map", None)
            if hasattr(analyzer_map, "get_rarity"):
                self._rarity_map = analyzer_map

        self.llm_weaknesses = initialize_llm_weaknesses()
        self.rarity_multipliers = initialize_rarity_multipliers()
        self.cultural_depth_weights = initialize_cultural_weights()

        self.anti_llm_stats = {
            "rare_patterns_generated": 0,
            "cultural_patterns_found": 0,
            "phonological_challenges": 0,
            "frequency_inversions": 0,
            "seed_expansions": 0,
        }

        self._seed_resources_initialized = False
        self._seed_analyzer = None
        self._cmu_seed_fn = None

    # ------------------------------------------------------------------
    # Dependency management
    # ------------------------------------------------------------------

    def set_phonetic_analyzer(self, analyzer: Any) -> None:
        self.phonetic_analyzer = analyzer
        rarity_map = getattr(analyzer, "rarity_map", None)
        if hasattr(rarity_map, "get_rarity"):
            self._rarity_map = rarity_map

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------

    def _get_rarity_map(self) -> Any:
        rarity_map = getattr(self, "_rarity_map", None)
        if hasattr(rarity_map, "get_rarity"):
            return rarity_map

        analyzer = getattr(self, "phonetic_analyzer", None)
        analyzer_map = getattr(analyzer, "rarity_map", None)
        if hasattr(analyzer_map, "get_rarity"):
            self._rarity_map = analyzer_map
            return analyzer_map

        from rhyme_rarity.core import WordRarityMap  # Local import to avoid cycle

        fallback = WordRarityMap()
        self._rarity_map = fallback
        return fallback

    def _get_word_rarity(self, word: str) -> float:
        rarity_map = self._get_rarity_map()
        try:
            rarity = rarity_map.get_rarity(word)
        except Exception:
            from rhyme_rarity.core import DEFAULT_RARITY_MAP

            rarity = DEFAULT_RARITY_MAP.get_rarity(word)

        try:
            return float(rarity)
        except (TypeError, ValueError):
            from rhyme_rarity.core import DEFAULT_RARITY_MAP

            return float(DEFAULT_RARITY_MAP.get_rarity(word))

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        return safe_float(value, default)

    def _normalize_seed_candidates(self, module1_seeds: Optional[List[Any]]) -> List[SeedCandidate]:
        return normalize_seed_candidates(
            module1_seeds,
            self._get_word_rarity,
            value_sanitizer=self._safe_float,
        )

    def _normalize_module1_candidates(self, candidates: Optional[List[Any]]) -> List[Dict[str, Any]]:
        return normalize_module1_candidates(candidates, value_sanitizer=self._safe_float)

    def _extract_suffixes(self, word: str) -> Set[str]:
        return extract_suffixes(word)

    def _query_seed_neighbors(self, cursor: Any, seed_word: str, limit: int) -> List[Dict[str, Any]]:
        return self.repository.fetch_seed_neighbors(seed_word, limit)

    def _query_suffix_matches(self, cursor: Any, suffix: str, limit: int) -> List[Dict[str, Any]]:
        return self.repository.fetch_suffix_matches(suffix, limit)

    def _attach_profile(self, pattern: AntiLLMPattern) -> None:
        analyzer = getattr(self, "phonetic_analyzer", None)
        if analyzer is None:
            return

        builder = getattr(analyzer, "derive_rhyme_profile", None)
        if not callable(builder):
            builder = getattr(analyzer, "build_feature_profile", None)
        if not callable(builder):
            return

        try:
            profile = builder(pattern.source_word, pattern.target_word)
        except Exception:
            profile = None

        profile_dict = normalize_profile_dict(profile)
        if not profile_dict:
            return

        pattern.feature_profile = profile_dict

        bradley = profile_dict.get("bradley_device")
        if isinstance(bradley, str) and bradley:
            pattern.bradley_device = bradley

        syllable_span = profile_dict.get("syllable_span")
        if isinstance(syllable_span, (list, tuple)) and len(syllable_span) == 2:
            try:
                pattern.syllable_span = (
                    int(syllable_span[0]),
                    int(syllable_span[1]),
                )
            except (TypeError, ValueError):
                pass

        stress_alignment = profile_dict.get("stress_alignment")
        if isinstance(stress_alignment, (int, float)):
            pattern.stress_alignment = float(stress_alignment)

        internal_score = profile_dict.get("internal_rhyme_score")
        if isinstance(internal_score, (int, float)):
            pattern.rarity_score *= 1.0 + max(0.0, float(internal_score)) * 0.15

        pattern.prosody_profile = {
            "complexity_tag": "dense"
            if pattern.syllable_span and max(pattern.syllable_span) >= 3
            else "steady",
            "stress_alignment": pattern.stress_alignment,
            "assonance": profile_dict.get("assonance_score"),
            "consonance": profile_dict.get("consonance_score"),
        }

    def _get_phonetic_fingerprint(self, word: str) -> Set[str]:
        return phonetic_fingerprint(word)

    def _ensure_seed_resources(self) -> None:
        if self._seed_resources_initialized:
            return

        self._seed_resources_initialized = True
        try:
            from rhyme_rarity.core import (
                EnhancedPhoneticAnalyzer,
                get_cmu_rhymes,
            )

            self._seed_analyzer = EnhancedPhoneticAnalyzer()
            self._cmu_seed_fn = get_cmu_rhymes
        except Exception:
            self._seed_analyzer = None
            self._cmu_seed_fn = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_anti_llm_patterns(
        self,
        source_word: str,
        limit: int = 20,
        module1_seeds: Optional[List[Any]] = None,
        seed_signatures: Optional[Set[str]] = None,
        delivered_words: Optional[Set[str]] = None,
    ) -> List[AntiLLMPattern]:
        if not source_word or not source_word.strip() or limit <= 0:
            return []

        source_word = source_word.lower().strip()
        patterns: List[AntiLLMPattern] = []

        signature_hints: Set[str] = set()
        if seed_signatures:
            signature_hints.update({str(sig).strip() for sig in seed_signatures if sig})

        normalized_seeds = normalize_seed_candidates(
            module1_seeds,
            self._get_word_rarity,
            value_sanitizer=self._safe_float,
        )
        for seed in normalized_seeds:
            signature_hints.update(seed.signatures)

        delivered_set: Set[str] = {source_word}
        if delivered_words:
            delivered_set.update({str(word).lower().strip() for word in delivered_words if word})
        delivered_set.update({seed.normalized() for seed in normalized_seeds})

        seen_targets: Set[str] = set(delivered_set)

        if normalized_seeds:
            seed_patterns = self._expand_from_seed_candidates(
                None,
                source_word,
                normalized_seeds,
                limit=limit,
                signature_hints=signature_hints,
                seen_targets=seen_targets,
            )
            for pattern in seed_patterns:
                if len(patterns) >= limit:
                    break
                normalized_target = pattern.target_word.lower().strip()
                if not normalized_target or normalized_target in seen_targets:
                    continue
                patterns.append(pattern)
                seen_targets.add(normalized_target)

        if len(patterns) >= limit:
            return patterns[:limit]

        per_strategy = max(1, limit // 4)

        strategy_batches = [
            find_rare_combinations(
                self.repository,
                source_word,
                per_strategy * 2,
                self._get_word_rarity,
                self.cultural_depth_weights,
                self._attach_profile,
                self.anti_llm_stats,
            ),
            find_phonological_challenges(
                self.repository,
                source_word,
                per_strategy * 2,
                self._analyze_phonological_complexity,
                self._attach_profile,
                self.anti_llm_stats,
            ),
            find_cultural_depth_patterns(
                self.repository,
                source_word,
                per_strategy * 2,
                self.cultural_depth_weights,
                self._attach_profile,
                self.anti_llm_stats,
            ),
            find_complex_syllable_patterns(
                self.repository,
                source_word,
                per_strategy * 2,
                self._calculate_syllable_complexity,
                self._attach_profile,
            ),
        ]

        for batch in strategy_batches:
            for pattern in batch:
                if len(patterns) >= limit:
                    break
                normalized_target = pattern.target_word.lower().strip()
                if not normalized_target or normalized_target in seen_targets:
                    continue
                patterns.append(pattern)
                seen_targets.add(normalized_target)
            if len(patterns) >= limit:
                break

        patterns.sort(key=lambda item: item.rarity_score * item.confidence, reverse=True)
        return patterns[:limit]

    # ------------------------------------------------------------------
    # Helper utilities reused by strategies
    # ------------------------------------------------------------------

    def _cmu_candidates(self, word: str, limit: int, analyzer: Any) -> List[Any]:
        cmu_fn = getattr(self, "_cmu_seed_fn", None)
        if callable(cmu_fn):
            return cmu_fn(word, limit=limit, analyzer=analyzer)
        return []

    def _expand_from_seed_candidates(
        self,
        cursor: Any,
        source_word: str,
        seeds: List[SeedCandidate],
        limit: int,
        signature_hints: Set[str],
        seen_targets: Set[str],
    ) -> List[AntiLLMPattern]:
        return expand_from_seed_candidates(
            repository=self.repository,
            source_word=source_word,
            seeds=seeds,
            limit=limit,
            signature_hints=signature_hints,
            seen_targets=seen_targets,
            get_word_rarity=self._get_word_rarity,
            analyze_phonological_complexity=self._analyze_phonological_complexity,
            calculate_syllable_complexity=self._calculate_syllable_complexity,
            attach_profile=self._attach_profile,
            ensure_seed_resources=self._ensure_seed_resources,
            cmu_candidates_fn=self._cmu_candidates,
            seed_analyzer=self._seed_analyzer,
            stats=self.anti_llm_stats,
            value_sanitizer=self._safe_float,
            fetch_neighbors=lambda seed, limit: self._query_seed_neighbors(None, seed, limit),
            fetch_suffix_matches=lambda suffix, limit: self._query_suffix_matches(None, suffix, limit),
            suffix_extractor=self._extract_suffixes,
            module1_normalizer=self._normalize_module1_candidates,
            fingerprint_fn=self._get_phonetic_fingerprint,
        )

    def _analyze_phonological_complexity(self, word1: str, word2: str) -> float:
        complexity = 0.0
        vowels1 = re.findall(r"[aeiou]+", word1)
        vowels2 = re.findall(r"[aeiou]+", word2)
        if vowels1 and vowels2 and vowels1[-1] != vowels2[-1]:
            complexity += 1.0
        consonants1 = re.sub(r"[aeiou]", "", word1)
        consonants2 = re.sub(r"[aeiou]", "", word2)
        if len(consonants1) >= 3 or len(consonants2) >= 3:
            complexity += 1.5
        if (word1.endswith("e") and not word2.endswith("e")) or (
            word2.endswith("e") and not word1.endswith("e")
        ):
            complexity += 1.0
        return complexity

    def _calculate_syllable_complexity(self, word: str) -> float:
        syllable_count = estimate_syllable_count(word)
        complexity = syllable_count * 0.5
        if len(word) >= 8 and "-" not in word:
            complexity += 1.0
        pattern_changes = 0
        for i in range(1, len(word)):
            is_vowel_now = word[i] in "aeiou"
            was_vowel_before = word[i - 1] in "aeiou"
            if is_vowel_now != was_vowel_before:
                pattern_changes += 1
        complexity += pattern_changes * 0.1
        return complexity

    def get_anti_llm_effectiveness_score(self, pattern: AntiLLMPattern) -> float:
        return effectiveness_score(pattern, self.llm_weaknesses)

    def get_performance_stats(self) -> Dict[str, Any]:
        return {
            "anti_llm_stats": self.anti_llm_stats.copy(),
            "llm_weaknesses_targeted": len(self.llm_weaknesses),
            "rarity_levels_available": len(self.rarity_multipliers),
            "cultural_depth_categories": len(self.cultural_depth_weights),
        }


__all__ = ["AntiLLMRhymeEngine", "AntiLLMPattern", "SeedCandidate"]
