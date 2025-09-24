"""Anti-LLM rhyme generation engine."""

from __future__ import annotations

import copy
import re
import threading
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from rhyme_rarity.core import CmuRhymeRepository, DefaultCmuRhymeRepository
from rhyme_rarity.utils.profile import normalize_profile_dict
from rhyme_rarity.utils.syllables import estimate_syllable_count
from rhyme_rarity.utils.observability import get_logger

from .dataclasses import AntiLLMPattern, SeedCandidate
from .repository import PatternRepository, SQLitePatternRepository
from .seed_expansion import (
    expand_from_seed_candidates,
    normalize_seed_candidate_payloads,
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
        cmu_repository: Optional[CmuRhymeRepository] = None,
    ) -> None:
        self.db_path = db_path
        self.phonetic_analyzer = phonetic_analyzer
        self.repository: PatternRepository = repository or SQLitePatternRepository(db_path)
        self.cmu_repository: CmuRhymeRepository = (
            cmu_repository or DefaultCmuRhymeRepository()
        )
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

        self._stats_lock = threading.Lock()

        self._cache_lock = threading.RLock()
        self._max_cache_entries = 256
        self._pattern_cache: OrderedDict[
            Tuple[str, int, Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]],
            Tuple[Any, ...],
        ] = OrderedDict()
        self._neighbor_cache: OrderedDict[Tuple[str, int], Tuple[Tuple[Tuple[str, Any], ...], ...]] = OrderedDict()
        self._suffix_cache: OrderedDict[Tuple[str, int], Tuple[Tuple[Tuple[str, Any], ...], ...]] = OrderedDict()
        self._rare_rows_cache: OrderedDict[
            Tuple[str, int], Tuple[Tuple[Tuple[str, Any], ...], ...]
        ] = OrderedDict()
        self._phonological_rows_cache: OrderedDict[
            Tuple[str, int], Tuple[Tuple[Tuple[str, Any], ...], ...]
        ] = OrderedDict()
        self._cultural_rows_cache: OrderedDict[
            Tuple[str, int], Tuple[Tuple[Tuple[str, Any], ...], ...]
        ] = OrderedDict()
        self._complex_rows_cache: OrderedDict[
            Tuple[str, int], Tuple[Tuple[Tuple[str, Any], ...], ...]
        ] = OrderedDict()
        self._normalized_seed_cache: OrderedDict[
            Tuple[Any, ...], Tuple[SeedCandidate, ...]
        ] = OrderedDict()
        self._rarity_cache: Dict[str, float] = {}
        self._fingerprint_cache: Dict[str, Set[str]] = {}
        self._profile_cache: OrderedDict[Tuple[str, str], Dict[str, Any]] = OrderedDict()

        self._seed_resources_initialized = False
        self._seed_analyzer = None
        self._cmu_seed_fn = None
        self._cmu_seed_repository: Optional[CmuRhymeRepository] = None

        self._logger = get_logger(__name__).bind(component="anti_llm_engine")
        self._logger.info(
            "Anti-LLM engine initialised",
            context={"db_path": db_path},
        )

    # ------------------------------------------------------------------
    # Cache management helpers
    # ------------------------------------------------------------------

    def _trim_cache(self, cache: OrderedDict) -> None:
        if self._max_cache_entries <= 0:
            cache.clear()
            return

        while len(cache) > self._max_cache_entries:
            cache.popitem(last=False)

    @staticmethod
    def _freeze_rows(rows: List[Dict[str, Any]]) -> Tuple[Tuple[Tuple[str, Any], ...], ...]:
        frozen: List[Tuple[Tuple[str, Any], ...]] = []
        for row in rows:
            frozen.append(tuple(sorted(row.items())))
        return tuple(frozen)

    @staticmethod
    def _thaw_rows(rows: Tuple[Tuple[Tuple[str, Any], ...], ...]) -> List[Dict[str, Any]]:
        return [dict(items) for items in rows]

    def _load_cached_rows(
        self, cache: OrderedDict, key: Tuple[Any, ...]
    ) -> Optional[List[Dict[str, Any]]]:
        with self._cache_lock:
            frozen = cache.get(key)
            if frozen is None:
                return None
            cache.move_to_end(key)
            return self._thaw_rows(frozen)

    def _store_cached_rows(
        self, cache: OrderedDict, key: Tuple[Any, ...], rows: List[Dict[str, Any]]
    ) -> None:
        with self._cache_lock:
            cache[key] = self._freeze_rows(rows)
            self._trim_cache(cache)

    @staticmethod
    def _freeze_payload(payload: Any) -> Any:
        if isinstance(payload, dict):
            return tuple(
                sorted(
                    (str(key), AntiLLMRhymeEngine._freeze_payload(value))
                    for key, value in payload.items()
                )
            )
        if isinstance(payload, (list, tuple, set)):
            return tuple(AntiLLMRhymeEngine._freeze_payload(item) for item in payload)
        return str(payload)

    def _load_normalized_seed_cache(
        self, key: Tuple[Any, ...]
    ) -> Optional[List[SeedCandidate]]:
        with self._cache_lock:
            cached = self._normalized_seed_cache.get(key)
            if cached is None:
                return None
            self._normalized_seed_cache.move_to_end(key)
            return [copy.deepcopy(seed) for seed in cached]

    def _store_normalized_seed_cache(
        self, key: Tuple[Any, ...], seeds: List[SeedCandidate]
    ) -> None:
        with self._cache_lock:
            self._normalized_seed_cache[key] = tuple(
                copy.deepcopy(seed) for seed in seeds
            )
            self._trim_cache(self._normalized_seed_cache)

    def _get_or_cache_rows(
        self,
        cache: OrderedDict,
        key: Tuple[Any, ...],
        loader: Callable[[], List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        cached = self._load_cached_rows(cache, key)
        if cached is not None:
            return cached
        rows = loader()
        self._store_cached_rows(cache, key, rows)
        return rows

    def _apply_cached_profile(
        self, pattern: AntiLLMPattern, cache_entry: Dict[str, Any]
    ) -> None:
        if not cache_entry:
            return

        feature_profile = cache_entry.get("feature_profile")
        if isinstance(feature_profile, dict):
            pattern.feature_profile = copy.deepcopy(feature_profile)

        prosody_profile = cache_entry.get("prosody_profile")
        if isinstance(prosody_profile, dict):
            pattern.prosody_profile = copy.deepcopy(prosody_profile)

        bradley_device = cache_entry.get("bradley_device")
        if isinstance(bradley_device, str) and bradley_device:
            pattern.bradley_device = bradley_device

        syllable_span = cache_entry.get("syllable_span")
        if (
            isinstance(syllable_span, tuple)
            and len(syllable_span) == 2
            and all(isinstance(item, int) for item in syllable_span)
        ):
            pattern.syllable_span = syllable_span

        stress_alignment = cache_entry.get("stress_alignment")
        if isinstance(stress_alignment, float):
            pattern.stress_alignment = stress_alignment

        rarity_multiplier = cache_entry.get("rarity_multiplier")
        if isinstance(rarity_multiplier, float) and rarity_multiplier > 0:
            pattern.rarity_score *= rarity_multiplier

    def _store_profile_cache(
        self, cache_key: Tuple[str, str], cache_entry: Dict[str, Any]
    ) -> None:
        with self._cache_lock:
            self._profile_cache[cache_key] = cache_entry
            self._trim_cache(self._profile_cache)

    def _load_pattern_cache(
        self, key: Tuple[str, int, Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]]
    ) -> Optional[List[Any]]:
        with self._cache_lock:
            cached = self._pattern_cache.get(key)
            if cached is None:
                return None
            self._pattern_cache.move_to_end(key)
            return [copy.deepcopy(pattern) for pattern in cached]

    def _store_pattern_cache(
        self,
        key: Tuple[str, int, Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]],
        patterns: List[Any],
    ) -> None:
        with self._cache_lock:
            self._pattern_cache[key] = tuple(copy.deepcopy(pattern) for pattern in patterns)
            self._trim_cache(self._pattern_cache)

    def _clear_caches(self) -> None:
        with self._cache_lock:
            self._pattern_cache.clear()
            self._neighbor_cache.clear()
            self._suffix_cache.clear()
            self._rare_rows_cache.clear()
            self._phonological_rows_cache.clear()
            self._cultural_rows_cache.clear()
            self._complex_rows_cache.clear()
            self._normalized_seed_cache.clear()
            self._rarity_cache.clear()
            self._fingerprint_cache.clear()
            self._profile_cache.clear()
        self._logger.debug("Cleared anti-LLM caches")

    # ------------------------------------------------------------------
    # Dependency management
    # ------------------------------------------------------------------

    def set_phonetic_analyzer(self, analyzer: Any) -> None:
        self.phonetic_analyzer = analyzer
        self._seed_analyzer = analyzer
        self._seed_resources_initialized = False
        rarity_map = getattr(analyzer, "rarity_map", None)
        if hasattr(rarity_map, "get_rarity"):
            self._rarity_map = rarity_map
        self._clear_caches()
        self._logger.info(
            "Updated anti-LLM phonetic analyzer",
            context={"analyzer_present": analyzer is not None},
        )

    def clear_cached_results(self) -> None:
        """Expose cache reset for integration with orchestration layers."""

        self._clear_caches()
        self._logger.info("Anti-LLM caches cleared on request")

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
        normalized = (word or "").lower().strip()
        if not normalized:
            return 0.0

        with self._cache_lock:
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
            rarity_float = float(rarity)
        except (TypeError, ValueError):
            from rhyme_rarity.core import DEFAULT_RARITY_MAP

            rarity_float = float(DEFAULT_RARITY_MAP.get_rarity(normalized))

        with self._cache_lock:
            self._rarity_cache[normalized] = rarity_float

        return rarity_float

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        return safe_float(value, default)

    def _normalize_seed_candidates(self, module1_seeds: Optional[List[Any]]) -> List[SeedCandidate]:
        if not module1_seeds:
            return []

        cache_key = self._freeze_payload(module1_seeds)
        cached = self._load_normalized_seed_cache(cache_key)
        if cached is not None:
            return cached

        normalized = normalize_seed_candidates(
            module1_seeds,
            self._get_word_rarity,
            value_sanitizer=self._safe_float,
            fingerprint_fn=self._get_phonetic_fingerprint,
            suffix_extractor=self._extract_suffixes,
        )
        self._store_normalized_seed_cache(cache_key, normalized)
        return normalized

    def _normalize_seed_candidate_payloads(self, candidates: Optional[List[Any]]) -> List[Dict[str, Any]]:
        return normalize_seed_candidate_payloads(candidates, value_sanitizer=self._safe_float)

    def _extract_suffixes(self, word: str) -> Set[str]:
        return extract_suffixes(word)

    def _query_seed_neighbors(self, cursor: Any, seed_word: str, limit: int) -> List[Dict[str, Any]]:
        if not seed_word:
            return []

        cache_key = (seed_word.lower().strip(), int(limit))
        cached = self._load_cached_rows(self._neighbor_cache, cache_key)
        if cached is not None:
            return cached

        rows = self.repository.fetch_seed_neighbors(seed_word, limit)
        self._store_cached_rows(self._neighbor_cache, cache_key, rows)
        return rows

    def _query_suffix_matches(self, cursor: Any, suffix: str, limit: int) -> List[Dict[str, Any]]:
        if not suffix:
            return []

        cache_key = (suffix.lower().strip(), int(limit))
        cached = self._load_cached_rows(self._suffix_cache, cache_key)
        if cached is not None:
            return cached

        rows = self.repository.fetch_suffix_matches(suffix, limit)
        self._store_cached_rows(self._suffix_cache, cache_key, rows)
        return rows

    def _attach_profile(self, pattern: AntiLLMPattern) -> None:
        analyzer = getattr(self, "phonetic_analyzer", None)
        if analyzer is None:
            return

        builder = getattr(analyzer, "derive_rhyme_profile", None)
        if not callable(builder):
            builder = getattr(analyzer, "build_feature_profile", None)
        if not callable(builder):
            return

        cache_key = (
            pattern.source_word.lower().strip(),
            pattern.target_word.lower().strip(),
        )
        cached_profile: Optional[Dict[str, Any]] = None
        if cache_key[0] and cache_key[1]:
            with self._cache_lock:
                cached_profile = self._profile_cache.get(cache_key)
                if cached_profile is not None:
                    self._profile_cache.move_to_end(cache_key)
        if cached_profile is not None:
            self._apply_cached_profile(pattern, cached_profile)
            return

        try:
            profile = builder(pattern.source_word, pattern.target_word)
        except Exception:
            profile = None

        profile_dict = normalize_profile_dict(profile)
        if not profile_dict:
            if cache_key[0] and cache_key[1]:
                self._store_profile_cache(cache_key, {})
            return

        base_rarity = pattern.rarity_score
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
        rarity_multiplier: Optional[float] = None
        if isinstance(internal_score, (int, float)):
            rarity_multiplier = 1.0 + max(0.0, float(internal_score)) * 0.15
            pattern.rarity_score *= rarity_multiplier

        pattern.prosody_profile = {
            "complexity_tag": "dense"
            if pattern.syllable_span and max(pattern.syllable_span) >= 3
            else "steady",
            "stress_alignment": pattern.stress_alignment,
            "assonance": profile_dict.get("assonance_score"),
            "consonance": profile_dict.get("consonance_score"),
        }

        if cache_key[0] and cache_key[1]:
            cache_entry: Dict[str, Any] = {
                "feature_profile": copy.deepcopy(profile_dict),
                "prosody_profile": copy.deepcopy(pattern.prosody_profile),
                "bradley_device": pattern.bradley_device,
                "syllable_span": pattern.syllable_span,
                "stress_alignment": pattern.stress_alignment,
            }
            if rarity_multiplier is not None and base_rarity > 0:
                cache_entry["rarity_multiplier"] = rarity_multiplier
            self._store_profile_cache(cache_key, cache_entry)

    def _get_phonetic_fingerprint(self, word: str) -> Set[str]:
        normalized = (word or "").lower().strip()
        if not normalized:
            return set()

        with self._cache_lock:
            cached = self._fingerprint_cache.get(normalized)
            if cached is not None:
                return set(cached)

        fingerprint = phonetic_fingerprint(normalized)

        with self._cache_lock:
            self._fingerprint_cache[normalized] = set(fingerprint)

        return set(fingerprint)

    def _ensure_seed_resources(self) -> None:
        if self._seed_resources_initialized:
            return

        self._seed_resources_initialized = True
        analyzer = getattr(self, "phonetic_analyzer", None)
        if analyzer is None:
            try:
                from rhyme_rarity.core import EnhancedPhoneticAnalyzer

                analyzer = EnhancedPhoneticAnalyzer()
            except Exception:
                analyzer = None

        self._seed_analyzer = analyzer
        self._cmu_seed_repository = self.cmu_repository
        if self._cmu_seed_repository is not None:
            self._cmu_seed_fn = self._cmu_seed_repository.lookup
        else:
            self._cmu_seed_fn = None

    def _increment_stat(self, key: str, amount: int = 1) -> None:
        if amount == 0:
            return

        with self._stats_lock:
            self.anti_llm_stats[key] = self.anti_llm_stats.get(key, 0) + amount

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

        normalized_seeds = self._normalize_seed_candidates(module1_seeds)
        normalized_seed_tokens = tuple(sorted(seed.normalized() for seed in normalized_seeds))
        for seed in normalized_seeds:
            signature_hints.update(seed.signature_hints or seed.signatures)

        delivered_words_normalized: Set[str] = set()
        if delivered_words:
            delivered_words_normalized.update(
                {
                    str(word).lower().strip()
                    for word in delivered_words
                    if word and str(word).strip()
                }
            )

        cache_key = (
            source_word,
            int(limit),
            normalized_seed_tokens,
            tuple(sorted(sig for sig in signature_hints if sig)),
            tuple(sorted(delivered_words_normalized)),
        )

        cached_patterns = self._load_pattern_cache(cache_key)
        if cached_patterns is not None:
            return cached_patterns[:limit]

        delivered_set: Set[str] = {source_word}
        delivered_set.update(delivered_words_normalized)
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

        rare_rows = self._get_or_cache_rows(
            self._rare_rows_cache,
            (source_word, per_strategy * 2),
            lambda: self.repository.fetch_rare_combinations(source_word, per_strategy * 2),
        )
        phonological_rows = self._get_or_cache_rows(
            self._phonological_rows_cache,
            (source_word, per_strategy * 2),
            lambda: self.repository.fetch_phonological_challenges(
                source_word, per_strategy * 2
            ),
        )
        cultural_rows = self._get_or_cache_rows(
            self._cultural_rows_cache,
            (source_word, per_strategy * 2),
            lambda: self.repository.fetch_cultural_depth_patterns(
                source_word, per_strategy * 2
            ),
        )
        complex_rows = self._get_or_cache_rows(
            self._complex_rows_cache,
            (source_word, per_strategy * 2),
            lambda: self.repository.fetch_complex_syllable_patterns(
                source_word, per_strategy * 2
            ),
        )

        strategy_batches = [
            find_rare_combinations(
                self.repository,
                source_word,
                per_strategy * 2,
                self._get_word_rarity,
                self.cultural_depth_weights,
                self._attach_profile,
                stats=self.anti_llm_stats,
                stat_recorder=self._increment_stat,
                rows=rare_rows,
                rarity_lookup=self._rarity_cache,
            ),
            find_phonological_challenges(
                self.repository,
                source_word,
                per_strategy * 2,
                self._analyze_phonological_complexity,
                self._attach_profile,
                stats=self.anti_llm_stats,
                stat_recorder=self._increment_stat,
                rows=phonological_rows,
            ),
            find_cultural_depth_patterns(
                self.repository,
                source_word,
                per_strategy * 2,
                self.cultural_depth_weights,
                self._attach_profile,
                stats=self.anti_llm_stats,
                stat_recorder=self._increment_stat,
                rows=cultural_rows,
            ),
            find_complex_syllable_patterns(
                self.repository,
                source_word,
                per_strategy * 2,
                self._calculate_syllable_complexity,
                self._attach_profile,
                rows=complex_rows,
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
        limited = patterns[:limit]
        self._store_pattern_cache(cache_key, limited)
        return limited

    # ------------------------------------------------------------------
    # Helper utilities reused by strategies
    # ------------------------------------------------------------------

    def _fetch_cmu_seed_candidates(self, word: str, limit: int, analyzer: Any) -> List[Any]:
        """Fetch CMU-derived seed candidates using the configured analyzer, returning an empty list when unavailable."""
        cmu_fn = getattr(self, "_cmu_seed_fn", None)
        if callable(cmu_fn):
            cmu_loader = getattr(analyzer, "cmu_loader", None) if analyzer else None
            try:
                return cmu_fn(
                    word,
                    limit=limit,
                    analyzer=analyzer,
                    cmu_loader=cmu_loader,
                )
            except Exception:
                return []
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
            cmu_candidates_fn=self._fetch_cmu_seed_candidates,
            seed_analyzer=self._seed_analyzer,
            stats=self.anti_llm_stats,
            stat_recorder=self._increment_stat,
            value_sanitizer=self._safe_float,
            fetch_neighbors=lambda seed, limit: self._query_seed_neighbors(None, seed, limit),
            fetch_suffix_matches=lambda suffix, limit: self._query_suffix_matches(None, suffix, limit),
            suffix_extractor=self._extract_suffixes,
            module1_normalizer=self._normalize_seed_candidate_payloads,
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
        with self._stats_lock:
            stats_snapshot = dict(self.anti_llm_stats)

        return {
            "anti_llm_stats": stats_snapshot,
            "llm_weaknesses_targeted": len(self.llm_weaknesses),
            "rarity_levels_available": len(self.rarity_multipliers),
            "cultural_depth_categories": len(self.cultural_depth_weights),
        }


__all__ = ["AntiLLMRhymeEngine", "AntiLLMPattern", "SeedCandidate"]
