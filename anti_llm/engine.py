"""Anti-LLM rhyme generation engine."""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

from rhyme_rarity.utils.profile import normalize_profile_dict
from rhyme_rarity.utils.syllables import estimate_syllable_count

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


_PROFILE_CACHE_MISS = object()


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
        self._normalized_seed_cache: Dict[Any, Tuple[SeedCandidate, ...]] = {}
        self._seed_context_cache: Dict[
            Tuple[str, Any, frozenset, frozenset],
            Tuple[Tuple[SeedCandidate, ...], frozenset, frozenset],
        ] = {}
        self._rarity_cache: Dict[str, float] = {}
        self._fingerprint_cache: Dict[str, Tuple[str, ...]] = {}
        self._suffix_cache: Dict[str, Tuple[str, ...]] = {}
        self._phonological_complexity_cache: Dict[Tuple[str, str], float] = {}
        self._syllable_complexity_cache: Dict[str, float] = {}
        self._profile_cache: Dict[Tuple[str, str], Any] = {}
        self._seed_neighbor_cache: Dict[Tuple[str, int], Tuple[Dict[str, Any], ...]] = {}
        self._suffix_match_cache: Dict[Tuple[str, int], Tuple[Dict[str, Any], ...]] = {}
        self._strategy_row_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._cultural_multiplier_cache: Dict[str, float] = {}
        self._pattern_cache: Dict[str, List[AntiLLMPattern]] = {}
        self._current_source_key: Optional[str] = None

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
        normalized = str(word or "").lower().strip()
        if not normalized:
            return 0.0

        cached = self._rarity_cache.get(normalized)
        if cached is not None:
            return cached

        rarity_map = self._get_rarity_map()
        try:
            rarity = rarity_map.get_rarity(normalized)
        except Exception:
            from rhyme_rarity.core import DEFAULT_RARITY_MAP

            rarity = DEFAULT_RARITY_MAP.get_rarity(normalized)

        try:
            rarity_value = float(rarity)
        except (TypeError, ValueError):
            from rhyme_rarity.core import DEFAULT_RARITY_MAP

            rarity_value = float(DEFAULT_RARITY_MAP.get_rarity(normalized))

        self._rarity_cache[normalized] = rarity_value
        return rarity_value

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        return safe_float(value, default)

    def _freeze_for_cache(self, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, dict):
            return tuple(
                (key, self._freeze_for_cache(val)) for key, val in sorted(value.items())
            )
        if isinstance(value, (list, tuple, set)):
            return tuple(self._freeze_for_cache(item) for item in value)
        if isinstance(value, (str, int, float, bool)):
            return value
        return repr(value)

    def _prepare_source_context(self, source_word: str) -> None:
        if self._current_source_key == source_word:
            return

        self._current_source_key = source_word

        self._strategy_row_cache = {
            key: value
            for key, value in self._strategy_row_cache.items()
            if key[1] == source_word
        }

        cached_patterns = self._pattern_cache.get(source_word)
        self._pattern_cache = {
            source_word: list(cached_patterns) if cached_patterns else []
        }

    def _cached_normalize_seeds(
        self, module1_seeds: Optional[List[Any]]
    ) -> List[SeedCandidate]:
        key = self._freeze_for_cache(module1_seeds) if module1_seeds else ()
        cached = self._normalized_seed_cache.get(key)
        if cached is not None:
            return [seed for seed in cached]

        normalized = normalize_seed_candidates(
            module1_seeds,
            self._get_word_rarity,
            value_sanitizer=self._safe_float,
        )
        self._normalized_seed_cache[key] = tuple(normalized)
        return normalized

    def _seed_fingerprint(self, seed: SeedCandidate) -> Set[str]:
        cached = seed.cached_fingerprint()
        if cached:
            return set(cached)

        fingerprint = self._get_phonetic_fingerprint(seed.word)
        seed.cache_fingerprint(fingerprint)
        return set(fingerprint)

    def _seed_suffixes(self, seed: SeedCandidate) -> Set[str]:
        cached = seed.cached_suffixes()
        if cached:
            return set(cached)

        suffixes = self._extract_suffixes(seed.word)
        seed.cache_suffixes(suffixes)
        return set(suffixes)

    def _seed_signature_hints(self, seed: SeedCandidate) -> Set[str]:
        cached = seed.cached_signature_hints()
        if cached:
            return set(cached)

        hints = set(seed.signatures)
        hints.update(self._seed_fingerprint(seed))

        feature_profile = seed.feature_profile or {}
        stress_hint = feature_profile.get("stress_alignment")
        if isinstance(stress_hint, (int, float)):
            hints.add(f"stress::{round(float(stress_hint), 2)}")

        device_hint = feature_profile.get("bradley_device")
        if device_hint:
            hints.add(f"device::{str(device_hint).lower()}")

        seed.cache_signature_hints(hints)
        return hints

    def _prepare_seed_context(
        self,
        source_word: str,
        module1_seeds: Optional[List[Any]],
        seed_signatures: Optional[Set[str]],
        delivered_words: Optional[Set[str]],
    ) -> Tuple[List[SeedCandidate], Set[str], Set[str]]:
        normalized_seed_key = self._freeze_for_cache(module1_seeds) if module1_seeds else ()
        provided_signatures = frozenset(
            {str(sig).strip() for sig in seed_signatures if sig}
            if seed_signatures
            else set()
        )
        delivered_key = frozenset(
            {str(word).lower().strip() for word in delivered_words if word}
            if delivered_words
            else set()
        )

        cache_key = (source_word, normalized_seed_key, provided_signatures, delivered_key)
        cached = self._seed_context_cache.get(cache_key)
        if cached is not None:
            cached_seeds, cached_hints, cached_delivered = cached
            return (
                [seed for seed in cached_seeds],
                set(cached_hints),
                set(cached_delivered),
            )

        seeds = self._cached_normalize_seeds(module1_seeds)
        signature_hints: Set[str] = set(provided_signatures)
        delivered_set: Set[str] = set(delivered_key)
        delivered_set.add(source_word)

        for seed in seeds:
            signature_hints.update(self._seed_signature_hints(seed))
            delivered_set.add(seed.normalized())
            self._seed_suffixes(seed)

        cache_value = (
            tuple(seeds),
            frozenset(signature_hints),
            frozenset(delivered_set),
        )
        self._seed_context_cache[cache_key] = cache_value
        return seeds, signature_hints, delivered_set

    def _prefetch_rarities(self, words: Iterable[str]) -> Dict[str, float]:
        rarities: Dict[str, float] = {}
        for word in words:
            normalized = str(word or "").lower().strip()
            if not normalized:
                continue
            rarities[normalized] = self._get_word_rarity(normalized)
        return rarities

    def _prefetch_cultural_multipliers(self, labels: Iterable[str]) -> Dict[str, float]:
        multipliers: Dict[str, float] = {}
        for label in labels:
            normalized = str(label or "").strip()
            if normalized not in self._cultural_multiplier_cache:
                self._cultural_multiplier_cache[normalized] = self.cultural_depth_weights.get(
                    normalized, 1.0
                )
            multipliers[normalized] = self._cultural_multiplier_cache[normalized]
        return multipliers

    def _fetch_strategy_rows(
        self,
        strategy_name: str,
        source_word: str,
        limit: int,
        fetcher: Callable[[str, int], List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        cache_key = (strategy_name, source_word)
        cached = self._strategy_row_cache.get(cache_key)
        if cached is not None and cached.get("limit", 0) >= limit:
            rows: Tuple[Dict[str, Any], ...] = cached["rows"]
        else:
            rows_list = fetcher(source_word, limit)
            rows = tuple(dict(row) for row in rows_list)
            self._strategy_row_cache[cache_key] = {"limit": limit, "rows": rows}
        return [row.copy() for row in rows]

    def _build_profile_payload(self, profile_dict: Dict[str, Any]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "feature_profile": dict(profile_dict),
            "bradley_device": None,
            "syllable_span": None,
            "stress_alignment": None,
            "rarity_multiplier": 1.0,
            "prosody_profile": {},
        }

        bradley = profile_dict.get("bradley_device")
        if isinstance(bradley, str) and bradley:
            payload["bradley_device"] = bradley

        span_candidate = profile_dict.get("syllable_span")
        span_tuple: Optional[Tuple[int, int]] = None
        if isinstance(span_candidate, (list, tuple)) and len(span_candidate) == 2:
            try:
                span_tuple = (int(span_candidate[0]), int(span_candidate[1]))
            except (TypeError, ValueError):
                span_tuple = None
        if span_tuple:
            payload["syllable_span"] = span_tuple

        stress_alignment = profile_dict.get("stress_alignment")
        if isinstance(stress_alignment, (int, float)):
            payload["stress_alignment"] = float(stress_alignment)

        internal_score = profile_dict.get("internal_rhyme_score")
        if isinstance(internal_score, (int, float)):
            payload["rarity_multiplier"] = 1.0 + max(0.0, float(internal_score)) * 0.15

        payload["prosody_profile"] = {
            "complexity_tag": "dense" if span_tuple and max(span_tuple) >= 3 else "steady",
            "stress_alignment": payload["stress_alignment"],
            "assonance": profile_dict.get("assonance_score"),
            "consonance": profile_dict.get("consonance_score"),
        }

        return payload

    def _apply_profile_data(self, pattern: AntiLLMPattern, payload: Dict[str, Any]) -> None:
        pattern.feature_profile = dict(payload.get("feature_profile", {}))

        rarity_multiplier = float(payload.get("rarity_multiplier", 1.0))
        if rarity_multiplier != 1.0:
            pattern.rarity_score *= rarity_multiplier

        bradley_device = payload.get("bradley_device")
        if isinstance(bradley_device, str) and bradley_device:
            pattern.bradley_device = bradley_device

        syllable_span = payload.get("syllable_span")
        if isinstance(syllable_span, tuple) and len(syllable_span) == 2:
            pattern.syllable_span = syllable_span

        stress_alignment = payload.get("stress_alignment")
        if isinstance(stress_alignment, (int, float)):
            pattern.stress_alignment = float(stress_alignment)

        pattern.prosody_profile = dict(payload.get("prosody_profile", {}))

    def _normalize_seed_candidates(self, module1_seeds: Optional[List[Any]]) -> List[SeedCandidate]:
        return self._cached_normalize_seeds(module1_seeds)

    def _normalize_module1_candidates(self, candidates: Optional[List[Any]]) -> List[Dict[str, Any]]:
        return normalize_module1_candidates(candidates, value_sanitizer=self._safe_float)

    def _extract_suffixes(self, word: str) -> Set[str]:
        normalized = str(word or "").lower().strip()
        cached = self._suffix_cache.get(normalized)
        if cached is not None:
            return set(cached)

        suffixes = extract_suffixes(word)
        self._suffix_cache[normalized] = tuple(sorted(suffixes))
        return set(suffixes)

    def _query_seed_neighbors(self, cursor: Any, seed_word: str, limit: int) -> List[Dict[str, Any]]:
        key = (str(seed_word or "").lower().strip(), int(limit))
        cached = self._seed_neighbor_cache.get(key)
        if cached is not None:
            return [row.copy() for row in cached]

        rows = self.repository.fetch_seed_neighbors(seed_word, limit)
        cached_rows = tuple(dict(row) for row in rows)
        self._seed_neighbor_cache[key] = cached_rows
        return [row.copy() for row in cached_rows]

    def _query_suffix_matches(self, cursor: Any, suffix: str, limit: int) -> List[Dict[str, Any]]:
        key = (str(suffix or "").lower().strip(), int(limit))
        cached = self._suffix_match_cache.get(key)
        if cached is not None:
            return [row.copy() for row in cached]

        rows = self.repository.fetch_suffix_matches(suffix, limit)
        cached_rows = tuple(dict(row) for row in rows)
        self._suffix_match_cache[key] = cached_rows
        return [row.copy() for row in cached_rows]

    def _attach_profile(self, pattern: AntiLLMPattern) -> None:
        source_key = pattern.source_word.lower().strip()
        target_key = pattern.target_word.lower().strip()
        if not source_key or not target_key:
            return

        cache_key = (source_key, target_key)
        cached = self._profile_cache.get(cache_key)
        if cached is _PROFILE_CACHE_MISS:
            return
        if isinstance(cached, dict):
            self._apply_profile_data(pattern, cached)
            return

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
            self._profile_cache[cache_key] = _PROFILE_CACHE_MISS
            return

        payload = self._build_profile_payload(profile_dict)
        self._profile_cache[cache_key] = payload
        self._apply_profile_data(pattern, payload)

    def _get_phonetic_fingerprint(self, word: str) -> Set[str]:
        normalized = str(word or "").lower().strip()
        cached = self._fingerprint_cache.get(normalized)
        if cached is not None:
            return set(cached)

        fingerprint = phonetic_fingerprint(word)
        self._fingerprint_cache[normalized] = tuple(sorted(fingerprint))
        return set(fingerprint)

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
        self._prepare_source_context(source_word)

        cached_patterns = self._pattern_cache.get(source_word, [])
        patterns: List[AntiLLMPattern] = list(cached_patterns)

        seed_signature_set = set(seed_signatures) if seed_signatures else None
        delivered_word_set = set(delivered_words) if delivered_words else None

        normalized_seeds, signature_hints, delivered_set = self._prepare_seed_context(
            source_word,
            module1_seeds,
            seed_signature_set,
            delivered_word_set,
        )

        seen_targets: Set[str] = set(delivered_set)
        for pattern in patterns:
            normalized_target = pattern.target_word.lower().strip()
            if normalized_target:
                seen_targets.add(normalized_target)

        if len(patterns) >= limit:
            return patterns[:limit]

        new_patterns_added = False

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
                new_patterns_added = True

        if len(patterns) >= limit:
            cache_cap = max(limit, 64)
            self._pattern_cache[source_word] = patterns[:cache_cap]
            return patterns[:limit]

        per_strategy = max(1, limit // 4)

        rare_rows = self._fetch_strategy_rows(
            "rare",
            source_word,
            per_strategy * 2,
            self.repository.fetch_rare_combinations,
        )
        rarity_lookup = self._prefetch_rarities(row.get("target_word") for row in rare_rows)
        rare_cultural_lookup = self._prefetch_cultural_multipliers(
            row.get("cultural_significance") for row in rare_rows
        )

        phonological_rows = self._fetch_strategy_rows(
            "phonological",
            source_word,
            per_strategy * 2,
            self.repository.fetch_phonological_challenges,
        )

        cultural_rows = self._fetch_strategy_rows(
            "cultural",
            source_word,
            per_strategy * 2,
            self.repository.fetch_cultural_depth_patterns,
        )

        cultural_lookup = self._prefetch_cultural_multipliers(
            row.get("cultural_significance") for row in cultural_rows
        )

        syllable_rows = self._fetch_strategy_rows(
            "syllable",
            source_word,
            per_strategy * 2,
            self.repository.fetch_complex_syllable_patterns,
        )

        strategy_batches = [
            find_rare_combinations(
                self.repository,
                source_word,
                per_strategy * 2,
                self._get_word_rarity,
                self.cultural_depth_weights,
                self._attach_profile,
                self.anti_llm_stats,
                rows=rare_rows,
                rarity_lookup=rarity_lookup,
                cultural_multiplier_lookup=rare_cultural_lookup,
            ),
            find_phonological_challenges(
                self.repository,
                source_word,
                per_strategy * 2,
                self._analyze_phonological_complexity,
                self._attach_profile,
                self.anti_llm_stats,
                rows=phonological_rows,
            ),
            find_cultural_depth_patterns(
                self.repository,
                source_word,
                per_strategy * 2,
                self.cultural_depth_weights,
                self._attach_profile,
                self.anti_llm_stats,
                rows=cultural_rows,
                cultural_multiplier_lookup=cultural_lookup,
            ),
            find_complex_syllable_patterns(
                self.repository,
                source_word,
                per_strategy * 2,
                self._calculate_syllable_complexity,
                self._attach_profile,
                rows=syllable_rows,
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
                new_patterns_added = True
            if len(patterns) >= limit:
                break

        if new_patterns_added or len(patterns) != len(cached_patterns):
            patterns.sort(key=lambda item: item.rarity_score * item.confidence, reverse=True)

        cache_cap = max(limit, 64)
        self._pattern_cache[source_word] = patterns[:cache_cap]
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
        key = (str(word1 or "").lower().strip(), str(word2 or "").lower().strip())
        cached = self._phonological_complexity_cache.get(key)
        if cached is not None:
            return cached

        complexity = 0.0
        vowels1 = re.findall(r"[aeiou]+", key[0])
        vowels2 = re.findall(r"[aeiou]+", key[1])
        if vowels1 and vowels2 and vowels1[-1] != vowels2[-1]:
            complexity += 1.0
        consonants1 = re.sub(r"[aeiou]", "", key[0])
        consonants2 = re.sub(r"[aeiou]", "", key[1])
        if len(consonants1) >= 3 or len(consonants2) >= 3:
            complexity += 1.5
        if (key[0].endswith("e") and not key[1].endswith("e")) or (
            key[1].endswith("e") and not key[0].endswith("e")
        ):
            complexity += 1.0
        self._phonological_complexity_cache[key] = complexity
        return complexity

    def _calculate_syllable_complexity(self, word: str) -> float:
        original = str(word or "").strip()
        normalized = original.lower()
        cached = self._syllable_complexity_cache.get(normalized)
        if cached is not None:
            return cached

        syllable_count = estimate_syllable_count(original)
        complexity = syllable_count * 0.5
        if len(normalized) >= 8 and "-" not in normalized:
            complexity += 1.0
        pattern_changes = 0
        for i in range(1, len(normalized)):
            is_vowel_now = normalized[i] in "aeiou"
            was_vowel_before = normalized[i - 1] in "aeiou"
            if is_vowel_now != was_vowel_before:
                pattern_changes += 1
        complexity += pattern_changes * 0.1
        self._syllable_complexity_cache[normalized] = complexity
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
