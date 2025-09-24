"""Search service orchestrating rhyme discovery and formatting."""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from anti_llm import AntiLLMRhymeEngine
from cultural.engine import CulturalIntelligenceEngine
from rhyme_rarity.core import (
    EnhancedPhoneticAnalyzer,
    extract_phrase_components,
    get_cmu_rhymes,
)

from ..data.database import SQLiteRhymeRepository

logger = logging.getLogger(__name__)


@dataclass
class SearchFilters:
    """Normalised search parameters and feature filters."""

    limit: int
    min_confidence: float
    cultural_filters: Set[str] = field(default_factory=set)
    genre_filters: Set[str] = field(default_factory=set)
    rhyme_type_filters: Set[str] = field(default_factory=set)
    bradley_filters: Set[str] = field(default_factory=set)
    include_phonetic: bool = True
    include_cultural: bool = True
    include_anti_llm: bool = True
    max_line_distance: Optional[int] = None
    min_syllables: Optional[int] = None
    max_syllables: Optional[int] = None
    require_internal: bool = False
    min_rarity: Optional[float] = None
    min_stress_alignment: Optional[float] = None
    cadence_focus: Optional[str] = None

    @property
    def has_feature_filters(self) -> bool:
        return any(
            (
                self.cultural_filters,
                self.genre_filters,
                self.rhyme_type_filters,
                self.bradley_filters,
                self.max_line_distance is not None,
                self.min_syllables is not None,
                self.max_syllables is not None,
                self.require_internal,
                self.min_rarity is not None,
                self.min_stress_alignment is not None,
                self.cadence_focus is not None,
            )
        )


@dataclass
class SourceContext:
    """Snapshot of phonetic context derived from the source word."""

    original: str
    components: Any
    anchor_word: str
    anchor_display: str
    phonetic_profile: Dict[str, Any]
    signatures: Set[str]
    prefix: str
    suffix: str

    @property
    def is_multi_word(self) -> bool:
        return bool(self.components and len(self.components.normalized_tokens) > 1)


class SearchService:
    """Coordinates rhyme searches across analyzers and repositories."""

    def __init__(
        self,
        *,
        repository: SQLiteRhymeRepository,
        phonetic_analyzer: Optional[EnhancedPhoneticAnalyzer] = None,
        cultural_engine: Optional[CulturalIntelligenceEngine] = None,
        anti_llm_engine: Optional[AntiLLMRhymeEngine] = None,
        cmu_loader: Optional[object] = None,
    ) -> None:
        self.repository = repository
        self.phonetic_analyzer = phonetic_analyzer
        self.cultural_engine = cultural_engine
        self.anti_llm_engine = anti_llm_engine
        self.cmu_loader = cmu_loader
        if self.cmu_loader is None and phonetic_analyzer is not None:
            self.cmu_loader = getattr(phonetic_analyzer, "cmu_loader", None)

        self._max_cache_entries = 256
        self._fallback_signature_cache: OrderedDict[str, Tuple[str, ...]] = OrderedDict()
        self._cmu_rhyme_cache: OrderedDict[
            Tuple[str, int, Optional[int], Optional[int]], Tuple[Any, ...]
        ] = OrderedDict()
        self._related_words_cache: OrderedDict[Tuple[str, ...], Tuple[str, ...]] = OrderedDict()

    # ------------------------------------------------------------------
    # Dependency management & cache helpers
    # ------------------------------------------------------------------
    def set_phonetic_analyzer(self, analyzer: Optional[EnhancedPhoneticAnalyzer]) -> None:
        self.phonetic_analyzer = analyzer
        if analyzer is not None and getattr(analyzer, "cmu_loader", None):
            self.cmu_loader = getattr(analyzer, "cmu_loader")
        self._reset_phonetic_caches()

    def set_cultural_engine(self, engine: Optional[CulturalIntelligenceEngine]) -> None:
        self.cultural_engine = engine

    def set_anti_llm_engine(self, engine: Optional[AntiLLMRhymeEngine]) -> None:
        self.anti_llm_engine = engine

    def clear_cached_results(self) -> None:
        """Clear memoized helper caches."""

        self._fallback_signature_cache.clear()
        self._cmu_rhyme_cache.clear()
        self._related_words_cache.clear()

    def _reset_phonetic_caches(self) -> None:
        """Drop caches derived from analyzer or CMU resources."""

        self._cmu_rhyme_cache.clear()
        self._related_words_cache.clear()

    def _trim_cache(self, cache: OrderedDict[Any, Tuple[Any, ...]]) -> None:
        """Ensure caches stay within the configured ``_max_cache_entries`` size."""

        if self._max_cache_entries <= 0:
            return

        while len(cache) > self._max_cache_entries:
            cache.popitem(last=False)

    def _fallback_signature(self, word: Optional[str]) -> Set[str]:
        """Compute a light-weight fallback signature with memoization."""

        cache_key = (word or "").strip().lower()
        cached = self._fallback_signature_cache.get(cache_key)
        if cached is not None:
            self._fallback_signature_cache.move_to_end(cache_key)
            return set(cached)

        import re

        cleaned = re.sub(r"[^a-z]", "", cache_key)
        if not cleaned:
            signature_tuple: Tuple[str, ...] = tuple()
        else:
            vowels = re.findall(r"[aeiou]+", cleaned)
            last_vowel = vowels[-1] if vowels else ""
            ending = cleaned[-3:] if len(cleaned) >= 3 else cleaned
            signature_bits: List[str] = []
            if last_vowel:
                signature_bits.append(f"v:{last_vowel}")
            signature_bits.append(f"e:{ending}")
            signature_tuple = ("|".join(signature_bits),)

        self._fallback_signature_cache[cache_key] = signature_tuple
        self._trim_cache(self._fallback_signature_cache)

        return set(signature_tuple)

    def _lookup_cmu_rhymes(
        self,
        source_word: str,
        limit: int,
        analyzer: Optional[EnhancedPhoneticAnalyzer],
        cmu_loader: Optional[object],
    ) -> List[Any]:
        """Return CMU rhyme candidates with caching."""

        cache_key = (
            source_word,
            int(limit),
            id(analyzer) if analyzer is not None else None,
            id(cmu_loader) if cmu_loader is not None else None,
        )
        cached = self._cmu_rhyme_cache.get(cache_key)
        if cached is not None:
            self._cmu_rhyme_cache.move_to_end(cache_key)
            return list(cached)

        try:
            results = get_cmu_rhymes(
                source_word,
                limit=limit,
                analyzer=analyzer,
                cmu_loader=cmu_loader,
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Failed to fetch CMU rhymes for %s", source_word)
            results = []

        self._cmu_rhyme_cache[cache_key] = tuple(results)
        self._trim_cache(self._cmu_rhyme_cache)

        return list(results)

    def _fetch_related_words_cached(self, lookup_terms: Set[str]) -> Set[str]:
        """Return repository suggestions using an LRU-style cache."""

        if not lookup_terms or self.repository is None:
            return set()

        normalized_terms = tuple(
            sorted({(term or "").strip().lower() for term in lookup_terms if term})
        )
        if not normalized_terms:
            return set()

        cached = self._related_words_cache.get(normalized_terms)
        if cached is not None:
            self._related_words_cache.move_to_end(normalized_terms)
            return set(cached)

        try:
            results = self.repository.fetch_related_words(set(normalized_terms))
        except Exception as exc:  # pragma: no cover - repository errors surface via logs
            logger.exception("Failed to fetch related words for terms: %s", normalized_terms)
            results = set()

        results_set = set(results)
        self._related_words_cache[normalized_terms] = tuple(sorted(results_set))
        self._trim_cache(self._related_words_cache)

        return results_set

    # ------------------------------------------------------------------
    # Normalisation helpers
    # ------------------------------------------------------------------
    def normalize_source_name(self, name: Optional[str]) -> str:
        if name is None:
            return ""
        return str(name).strip().lower().replace("_", "-")

    def _normalize_to_list(self, value: Optional[Iterable[Any]]) -> List[str]:
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            return [str(v) for v in value if v is not None and str(v).strip()]
        if isinstance(value, str):
            return [value] if value.strip() else []
        return [str(value)]

    def _normalized_set(self, value: Optional[Iterable[Any]]) -> Set[str]:
        return {
            normalized
            for normalized in (
                self.normalize_source_name(item) for item in self._normalize_to_list(value)
            )
            if normalized
        }

    def _coerce_float(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    # ------------------------------------------------------------------
    # Source context construction
    # ------------------------------------------------------------------
    def _build_source_context(
        self,
        source_word: str,
        analyzer: Optional[EnhancedPhoneticAnalyzer],
        cmu_loader: Optional[object],
        cultural_engine: Optional[CulturalIntelligenceEngine],
    ) -> SourceContext:
        components = extract_phrase_components(source_word, cmu_loader)
        anchor_word = components.anchor or source_word
        anchor_display = components.anchor_display or anchor_word
        prefix_tokens: List[str] = []
        suffix_tokens: List[str] = []
        if components.anchor_index is not None:
            prefix_tokens = components.normalized_tokens[: components.anchor_index]
            suffix_tokens = components.normalized_tokens[components.anchor_index + 1 :]

        phonetic_profile: Dict[str, Any] = {
            "word": source_word,
            "normalized": components.normalized_phrase,
            "tokens": components.normalized_tokens,
            "token_syllables": components.syllable_counts,
            "anchor_word": anchor_word,
            "anchor_display": anchor_display,
            "syllables": components.total_syllables,
        }

        if analyzer and hasattr(analyzer, "describe_word"):
            try:
                description = analyzer.describe_word(source_word)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.exception("Failed to describe source word '%s'", source_word)
            else:
                if isinstance(description, dict):
                    phonetic_profile.update(description)
                else:
                    try:
                        phonetic_profile.update(dict(description))
                    except Exception:
                        phonetic_profile.update({})

        is_multi = bool(components.normalized_tokens and len(components.normalized_tokens) > 1)
        if not is_multi and " " in components.normalized_phrase:
            is_multi = True
        phonetic_profile.setdefault("is_multi_word", is_multi)

        signatures: Set[str] = set()
        if cultural_engine and hasattr(cultural_engine, "derive_rhyme_signatures"):
            try:
                derived = cultural_engine.derive_rhyme_signatures(source_word)
            except Exception as exc:  # pragma: no cover - cultural engine failures are logged
                logger.exception(
                    "Cultural engine failed to derive signatures for '%s'", source_word
                )
                derived = set()
            if derived:
                signatures.update(str(sig) for sig in derived if sig)

        if not signatures:
            signatures = self._fallback_signature(anchor_word)

        return SourceContext(
            original=source_word,
            components=components,
            anchor_word=anchor_word,
            anchor_display=anchor_display,
            phonetic_profile=phonetic_profile,
            signatures=signatures,
            prefix=" ".join(prefix_tokens).strip(),
            suffix=" ".join(suffix_tokens).strip(),
        )

    # ------------------------------------------------------------------
    # Filter parsing
    # ------------------------------------------------------------------
    def _parse_filters(
        self,
        limit: int,
        min_confidence: float,
        cultural_significance: Optional[Iterable[str]],
        genres: Optional[Iterable[str]],
        result_sources: Optional[Iterable[str]],
        max_line_distance: Optional[int],
        min_syllables: Optional[int],
        max_syllables: Optional[int],
        allowed_rhyme_types: Optional[Iterable[str]],
        bradley_devices: Optional[Iterable[str]],
        require_internal: bool,
        min_rarity: Optional[float],
        min_stress_alignment: Optional[float],
        cadence_focus: Optional[str],
    ) -> SearchFilters:
        parsed_limit = max(int(limit), 0)
        min_conf = self._coerce_float(min_confidence)
        if min_conf is None or min_conf < 0:
            min_conf = 0.0

        cultural_filters = self._normalized_set(cultural_significance)
        genre_filters = self._normalized_set(genres)
        rhyme_filters = self._normalized_set(allowed_rhyme_types)
        bradley_filters = self._normalized_set(bradley_devices)

        sources_raw = self._normalized_set(result_sources)
        source_map = {
            "phonetic": "phonetic",
            "phonetics": "phonetic",
            "cmu": "phonetic",
            "cultural": "cultural",
            "rap": "cultural",
            "rap-db": "cultural",
            "rap_db": "cultural",
            "anti-llm": "anti-llm",
            "anti_llm": "anti-llm",
            "multi-word": "anti-llm",
        }
        mapped_sources = {source_map.get(value, value) for value in sources_raw}
        if not mapped_sources:
            mapped_sources = {"phonetic", "cultural", "anti-llm"}

        include_phonetic = "phonetic" in mapped_sources
        include_cultural = "cultural" in mapped_sources
        include_anti_llm = "anti-llm" in mapped_sources

        try:
            max_line = int(max_line_distance) if max_line_distance is not None else None
        except (TypeError, ValueError):
            max_line = None

        def _coerce_int(value: Optional[int]) -> Optional[int]:
            if value is None:
                return None
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                return None
            return max(parsed, 1)

        min_syll = _coerce_int(min_syllables)
        max_syll = _coerce_int(max_syllables)

        rarity_threshold = self._coerce_float(min_rarity)
        stress_threshold = self._coerce_float(min_stress_alignment)

        cadence_normalized = None
        if isinstance(cadence_focus, str) and cadence_focus.strip():
            cadence_normalized = self.normalize_source_name(cadence_focus) or None

        return SearchFilters(
            limit=parsed_limit,
            min_confidence=min_conf,
            cultural_filters=cultural_filters,
            genre_filters=genre_filters,
            rhyme_type_filters=rhyme_filters,
            bradley_filters=bradley_filters,
            include_phonetic=include_phonetic,
            include_cultural=include_cultural,
            include_anti_llm=include_anti_llm,
            max_line_distance=max_line,
            min_syllables=min_syll,
            max_syllables=max_syll,
            require_internal=bool(require_internal),
            min_rarity=rarity_threshold,
            min_stress_alignment=stress_threshold,
            cadence_focus=cadence_normalized,
        )

    # ------------------------------------------------------------------
    # Public search entry point
    # ------------------------------------------------------------------
    def search_rhymes(
        self,
        source_word: str,
        limit: int = 20,
        min_confidence: float = 0.7,
        cultural_significance: Optional[List[str]] = None,
        genres: Optional[List[str]] = None,
        result_sources: Optional[List[str]] = None,
        max_line_distance: Optional[int] = None,
        min_syllables: Optional[int] = None,
        max_syllables: Optional[int] = None,
        allowed_rhyme_types: Optional[List[str]] = None,
        bradley_devices: Optional[List[str]] = None,
        require_internal: bool = False,
        min_rarity: Optional[float] = None,
        min_stress_alignment: Optional[float] = None,
        cadence_focus: Optional[str] = None,
    ) -> Dict[str, List[Dict]]:
        """Search for rhymes in the patterns.db database."""

        def _empty_response() -> Dict[str, List[Dict]]:
            return {"cmu": [], "anti_llm": [], "rap_db": []}

        if not source_word or not str(source_word).strip():
            return _empty_response()

        if limit <= 0:
            return _empty_response()

        analyzer = getattr(self, "phonetic_analyzer", None)
        cultural_engine = getattr(self, "cultural_engine", None)
        cmu_loader = None
        if analyzer is not None:
            cmu_loader = getattr(analyzer, "cmu_loader", None)
        if cmu_loader is None:
            cmu_loader = getattr(self, "cmu_loader", None)

        filters = self._parse_filters(
            limit,
            min_confidence,
            cultural_significance,
            genres,
            result_sources,
            max_line_distance,
            min_syllables,
            max_syllables,
            allowed_rhyme_types,
            bradley_devices,
            require_internal,
            min_rarity,
            min_stress_alignment,
            cadence_focus,
        )

        context = self._build_source_context(
            str(source_word).strip().lower(),
            analyzer,
            cmu_loader,
            cultural_engine,
        )

        response = {
            "source_profile": {
                "word": context.original,
                "anchor_word": context.anchor_word,
                "anchor_display": context.anchor_display,
                "phonetics": context.phonetic_profile,
                "signatures": sorted(context.signatures),
                "phrase_prefix": context.prefix,
                "phrase_suffix": context.suffix,
                "is_multi_word": context.is_multi_word,
            },
            "cmu": [],
            "anti_llm": [],
            "rap_db": [],
        }

        module1_seeds: List[Dict[str, Any]] = []
        aggregate_signatures: Set[str] = set(context.signatures)
        delivered_words: Set[str] = set()

        if filters.include_phonetic and analyzer is not None:
            phonetic_matches, module1_seeds = self._collect_phonetic_matches(
                context,
                filters,
                analyzer,
                cultural_engine,
                cmu_loader,
            )
            response["cmu"] = phonetic_matches
            delivered_words.update(
                entry["target_word"]
                for entry in phonetic_matches
                if entry.get("target_word")
            )
            for seed in module1_seeds:
                aggregate_signatures.update(seed.get("signatures", []))

        if filters.include_cultural and self.repository is not None:
            cultural_matches = self._collect_cultural_matches(
                context,
                filters,
                analyzer,
                cultural_engine,
            )
            response["rap_db"] = cultural_matches
            delivered_words.update(
                entry["target_word"]
                for entry in cultural_matches
                if entry.get("target_word")
            )

        if filters.include_anti_llm and self.anti_llm_engine is not None:
            anti_matches = self._collect_anti_llm_matches(
                context,
                filters,
                analyzer,
                module1_seeds,
                aggregate_signatures,
                delivered_words,
            )
            response["anti_llm"] = anti_matches

        return response

    # ------------------------------------------------------------------
    # Category collectors
    # ------------------------------------------------------------------
    def _collect_phonetic_matches(
        self,
        context: SourceContext,
        filters: SearchFilters,
        analyzer: EnhancedPhoneticAnalyzer,
        cultural_engine: Optional[CulturalIntelligenceEngine],
        cmu_loader: Optional[object],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        fetch_limit = max(filters.limit * 2, filters.limit + 5, 10)
        cmu_candidates = self._lookup_cmu_rhymes(
            context.original,
            fetch_limit,
            analyzer,
            cmu_loader,
        )

        results: List[Dict[str, Any]] = []
        seed_payloads: List[Dict[str, Any]] = []

        for candidate in cmu_candidates:
            normalized = self._normalise_cmu_candidate(candidate)
            target_word = normalized.get("target")
            if not target_word:
                continue

            similarity = normalized.get("similarity", 0.0)
            combined = normalized.get("combined", similarity)
            rarity = normalized.get("rarity")
            alignment = None

            if cultural_engine and hasattr(cultural_engine, "evaluate_rhyme_alignment"):
                try:
                    alignment = cultural_engine.evaluate_rhyme_alignment(
                        context.original,
                        target_word,
                        threshold=None,
                        rhyme_signatures=context.signatures,
                        source_context=None,
                        target_context=None,
                    )
                except Exception as exc:  # pragma: no cover - defensive logging
                    logger.exception(
                        "Cultural alignment failed for %s -> %s", context.original, target_word
                    )
                    alignment = None
                if alignment is None:
                    continue

            if alignment is None:
                alignment = self._fallback_phonetic_alignment(
                    analyzer,
                    context.original,
                    target_word,
                    similarity,
                    combined,
                    rarity,
                )

            entry = self._build_phonetic_entry(
                context,
                target_word,
                normalized,
                alignment,
                analyzer,
            )

            if not self._entry_passes_filters(entry, filters, analyzer):
                continue

            results.append(entry)
            seed_payloads.append(
                {
                    "word": target_word,
                    "rarity": entry.get("rarity_score"),
                    "combined": entry.get("combined_score"),
                    "signatures": list(entry.get("target_rhyme_signatures", [])),
                    "feature_profile": entry.get("feature_profile", {}),
                    "prosody_profile": entry.get("prosody_profile", {}),
                }
            )

        results.sort(key=lambda item: self._confidence_value(item), reverse=True)
        return results[: filters.limit], seed_payloads[:5]

    def _normalise_cmu_candidate(self, candidate: Any) -> Dict[str, Any]:
        """Normalise CMU candidate payloads into a predictable dictionary."""

        result: Dict[str, Any] = {}
        if isinstance(candidate, dict):
            result["target"] = (
                candidate.get("word")
                or candidate.get("target")
                or candidate.get("candidate")
            )
            result["similarity"] = self._coerce_float(candidate.get("similarity")) or self._coerce_float(
                candidate.get("score")
            ) or 0.0
            result["combined"] = self._coerce_float(candidate.get("combined")) or self._coerce_float(
                candidate.get("combined_score")
            ) or result["similarity"]
            result["rarity"] = self._coerce_float(candidate.get("rarity")) or self._coerce_float(
                candidate.get("rarity_score")
            )
            result["is_multi_word"] = bool(candidate.get("is_multi_word"))
        elif isinstance(candidate, (tuple, list)):
            target = candidate[0] if candidate else None
            result["target"] = target
            result["similarity"] = self._coerce_float(candidate[1] if len(candidate) > 1 else None) or 0.0
            result["rarity"] = self._coerce_float(candidate[2] if len(candidate) > 2 else None)
            result["combined"] = self._coerce_float(candidate[3] if len(candidate) > 3 else None) or result[
                "similarity"
            ]
            result["is_multi_word"] = isinstance(target, str) and " " in target.strip()
        else:
            result["target"] = candidate
            result["similarity"] = 0.0
            result["combined"] = 0.0
            result["rarity"] = None
            result["is_multi_word"] = isinstance(candidate, str) and " " in candidate.strip()
        if isinstance(result.get("target"), str):
            result["target"] = result["target"].strip()
        return result

    def _fallback_phonetic_alignment(
        self,
        analyzer: EnhancedPhoneticAnalyzer,
        source_word: str,
        target_word: str,
        similarity: float,
        combined: float,
        rarity: Optional[float],
    ) -> Dict[str, Any]:
        alignment: Dict[str, Any] = {
            "similarity": similarity,
            "combined": combined,
            "rarity": rarity,
            "signature_matches": [],
            "target_signatures": [],
            "features": {},
            "feature_profile": {},
            "prosody_profile": {},
        }

        classify = getattr(analyzer, "classify_rhyme_type", None)
        profile_builder = getattr(analyzer, "derive_rhyme_profile", None)

        rhyme_type = None
        if callable(classify):
            try:
                rhyme_type = classify(source_word, target_word, similarity)
            except Exception:
                rhyme_type = None
        if rhyme_type:
            alignment["rhyme_type"] = rhyme_type

        if callable(profile_builder):
            try:
                profile = profile_builder(source_word, target_word, similarity=similarity)
            except Exception:
                profile = None
            profile_dict = self._normalise_feature_profile(profile)
            if profile_dict:
                alignment["feature_profile"] = profile_dict
                if profile_dict.get("stress_alignment") is not None:
                    alignment.setdefault("prosody_profile", {})["stress_alignment"] = profile_dict[
                        "stress_alignment"
                    ]
                if profile_dict.get("bradley_device"):
                    alignment.setdefault("features", {})["bradley_device"] = profile_dict[
                        "bradley_device"
                    ]
        return alignment

    def _build_phonetic_entry(
        self,
        context: SourceContext,
        target_word: str,
        candidate: Dict[str, Any],
        alignment: Dict[str, Any],
        analyzer: EnhancedPhoneticAnalyzer,
    ) -> Dict[str, Any]:
        rarity = alignment.get("rarity")
        if rarity is None:
            rarity = candidate.get("rarity")
        target_phonetics = self._describe_word(analyzer, target_word)

        entry: Dict[str, Any] = {
            "source_word": context.original,
            "target_word": target_word,
            "pattern": self._build_pattern(context.original, target_word, candidate.get("is_multi_word")),
            "confidence": alignment.get("combined", candidate.get("combined")),
            "combined_score": alignment.get("combined", candidate.get("combined")),
            "rarity_score": rarity,
            "phonetic_sim": alignment.get("similarity", candidate.get("similarity", 0.0)),
            "result_source": "phonetic",
            "source_rhyme_signatures": sorted(context.signatures),
            "source_phonetic_profile": context.phonetic_profile,
            "target_phonetics": target_phonetics,
            "prosody_profile": self._normalise_dict(alignment.get("prosody_profile")),
            "feature_profile": self._normalise_feature_profile(alignment.get("feature_profile")),
            "matched_signatures": alignment.get("signature_matches", []),
            "target_rhyme_signatures": alignment.get("target_signatures", []),
            "rhyme_type": alignment.get("rhyme_type"),
        }
        syllables = target_phonetics.get("syllables")
        if syllables is None and analyzer is not None:
            estimator = getattr(analyzer, "estimate_syllables", None)
            if callable(estimator):
                try:
                    syllables = estimator(target_word)
                except Exception:
                    syllables = None
        if syllables is not None:
            try:
                entry["candidate_syllables"] = int(syllables)
            except (TypeError, ValueError):
                pass
        stress_alignment = self._stress_score(entry)
        if stress_alignment is not None:
            entry["stress_alignment"] = stress_alignment
        entry["is_multi_word"] = bool(candidate.get("is_multi_word")) or (
            isinstance(target_word, str) and " " in target_word.strip()
        )
        entry["result_variant"] = "multi_word" if entry["is_multi_word"] else "single_word"
        entry["phrase_prefix"] = context.prefix
        entry["phrase_suffix"] = context.suffix
        entry["source_phrase"] = context.original
        entry["source_anchor"] = context.anchor_word
        entry["anchor_display"] = context.anchor_display
        return entry

    def _collect_cultural_matches(
        self,
        context: SourceContext,
        filters: SearchFilters,
        analyzer: Optional[EnhancedPhoneticAnalyzer],
        cultural_engine: Optional[CulturalIntelligenceEngine],
    ) -> List[Dict[str, Any]]:
        if self.repository is None:
            return []

        try:
            source_rows, target_rows = self.repository.fetch_cultural_matches(
                context.original,
                min_confidence=filters.min_confidence,
                phonetic_threshold=None,
                cultural_filters=sorted(filters.cultural_filters),
                genre_filters=sorted(filters.genre_filters),
                max_line_distance=filters.max_line_distance,
                limit=filters.limit * 2 if filters.limit else None,
            )
        except Exception as exc:  # pragma: no cover - repository errors surface via log
            logger.exception("Failed to fetch cultural matches for '%s'", context.anchor_word)
            return []

        candidates: List[Dict[str, Any]] = []
        for row in list(source_rows):
            entry = self._build_cultural_entry(context, row, analyzer, swap=False)
            if cultural_engine:
                entry = self._enrich_cultural_entry(entry, cultural_engine, analyzer)
                if entry is None:
                    continue
            if not self._entry_passes_filters(entry, filters, analyzer):
                continue
            candidates.append(entry)

        for row in list(target_rows):
            entry = self._build_cultural_entry(context, row, analyzer, swap=True)
            if cultural_engine:
                entry = self._enrich_cultural_entry(entry, cultural_engine, analyzer)
                if entry is None:
                    continue
            if not self._entry_passes_filters(entry, filters, analyzer):
                continue
            candidates.append(entry)

        candidates.sort(key=lambda item: self._confidence_value(item), reverse=True)
        if filters.limit:
            return candidates[: filters.limit]
        return candidates

    def _build_cultural_entry(
        self,
        context: SourceContext,
        row: Sequence[Any],
        analyzer: Optional[EnhancedPhoneticAnalyzer],
        *,
        swap: bool,
    ) -> Dict[str, Any]:
        values: List[Any] = list(row) if isinstance(row, Sequence) else [row]
        if len(values) < 12:
            values.extend([None] * (12 - len(values)))
        (
            source_word,
            target_word,
            artist,
            song,
            pattern,
            genre,
            distance,
            confidence,
            phonetic_sim,
            cultural_sig,
            source_context,
            target_context,
        ) = values[:12]

        counterpart = str(target_word or "").strip()
        source_context_value = source_context
        target_context_value = target_context
        if swap:
            counterpart = str(source_word or "").strip()
            source_context_value, target_context_value = target_context, source_context

        pattern_text = self._build_pattern(context.original, counterpart, " " in counterpart)
        entry = {
            "source_word": context.original,
            "target_word": counterpart,
            "artist": artist,
            "song": song,
            "pattern": pattern_text,
            "genre": genre,
            "distance": distance,
            "confidence": confidence,
            "combined_score": confidence,
            "phonetic_sim": phonetic_sim,
            "cultural_sig": cultural_sig,
            "source_context": source_context_value,
            "target_context": target_context_value,
            "result_source": "cultural",
            "source_rhyme_signatures": sorted(context.signatures),
            "source_phonetic_profile": context.phonetic_profile,
            "phrase_prefix": context.prefix,
            "phrase_suffix": context.suffix,
            "source_phrase": context.original,
            "source_anchor": context.anchor_word,
            "anchor_display": context.anchor_display,
            "target_phonetics": self._describe_word(analyzer, counterpart),
        }
        entry["is_multi_word"] = " " in counterpart
        entry["result_variant"] = "multi_word" if entry["is_multi_word"] else "single_word"
        return entry

    def _enrich_cultural_entry(
        self,
        entry: Dict[str, Any],
        cultural_engine: CulturalIntelligenceEngine,
        analyzer: Optional[EnhancedPhoneticAnalyzer],
    ) -> Optional[Dict[str, Any]]:
        context_payload = {
            "artist": entry.get("artist"),
            "song": entry.get("song"),
            "source_word": entry.get("source_word"),
            "target_word": entry.get("target_word"),
            "pattern": entry.get("pattern"),
            "cultural_significance": entry.get("cultural_sig"),
        }
        try:
            context_obj = cultural_engine.get_cultural_context(context_payload)
        except Exception as exc:  # pragma: no cover - cultural engine diagnostics via logs
            logger.exception("Failed to build cultural context for '%s'", entry.get("target_word"))
            context_obj = None

        context_dict = self._normalise_dict(context_obj)
        if context_dict:
            entry["cultural_context"] = context_dict
            try:
                rarity = cultural_engine.get_cultural_rarity_score(context_obj)
            except Exception:
                rarity = None
            if rarity is not None:
                entry["cultural_rarity"] = rarity

        alignment = None
        if hasattr(cultural_engine, "evaluate_rhyme_alignment"):
            try:
                alignment = cultural_engine.evaluate_rhyme_alignment(
                    entry.get("source_word"),
                    entry.get("target_word"),
                    threshold=None,
                    rhyme_signatures=set(entry.get("source_rhyme_signatures", [])),
                    source_context=entry.get("source_context"),
                    target_context=entry.get("target_context"),
                )
            except Exception as exc:  # pragma: no cover - diagnostics via logs
                logger.exception(
                    "Cultural alignment failed for %s -> %s",
                    entry.get("source_word"),
                    entry.get("target_word"),
                )
                alignment = None

        if alignment is None:
            return entry

        if alignment.get("combined") is not None:
            entry["combined_score"] = alignment["combined"]
            entry["confidence"] = alignment["combined"]
        if alignment.get("similarity") is not None:
            entry["phonetic_sim"] = alignment["similarity"]
        if alignment.get("rarity") is not None:
            entry.setdefault("rarity_score", alignment["rarity"])
        if alignment.get("rhyme_type"):
            entry["rhyme_type"] = alignment["rhyme_type"]

        entry["prosody_profile"] = self._normalise_dict(alignment.get("prosody_profile"))
        entry["feature_profile"] = self._normalise_feature_profile(alignment.get("feature_profile"))
        entry["matched_signatures"] = alignment.get("signature_matches", [])
        entry["target_rhyme_signatures"] = alignment.get("target_signatures", [])

        stress_alignment = self._stress_score(entry)
        if stress_alignment is not None:
            entry["stress_alignment"] = stress_alignment
            entry["rhythm_score"] = stress_alignment

        if entry.get("combined_score") is None and entry.get("confidence") is not None:
            entry["combined_score"] = entry["confidence"]
        return entry

    def _collect_anti_llm_matches(
        self,
        context: SourceContext,
        filters: SearchFilters,
        analyzer: Optional[EnhancedPhoneticAnalyzer],
        module1_seeds: List[Dict[str, Any]],
        aggregate_signatures: Set[str],
        delivered_words: Set[str],
    ) -> List[Dict[str, Any]]:
        anti_engine = self.anti_llm_engine
        if anti_engine is None:
            return []

        try:
            patterns = anti_engine.generate_anti_llm_patterns(
                context.original,
                limit=filters.limit,
                module1_seeds=module1_seeds or None,
                seed_signatures=aggregate_signatures or None,
                delivered_words=delivered_words or None,
            )
        except Exception as exc:  # pragma: no cover - diagnostics via logs
            logger.exception("Anti-LLM engine failed for '%s'", context.original)
            return []

        entries: List[Dict[str, Any]] = []
        for pattern in patterns or []:
            entry = self._build_anti_llm_entry(context, pattern)
            if not self._entry_passes_filters(entry, filters, analyzer):
                continue
            entries.append(entry)

        entries.sort(key=lambda item: self._confidence_value(item), reverse=True)
        if filters.limit:
            return entries[: filters.limit]
        return entries

    def _build_anti_llm_entry(self, context: SourceContext, pattern: Any) -> Dict[str, Any]:
        rarity = self._coerce_float(getattr(pattern, "rarity_score", None))
        combined_metric = getattr(pattern, "combined", None)
        if combined_metric is None:
            combined_metric = getattr(pattern, "combined_score", None)
        combined = self._coerce_float(combined_metric)
        confidence = self._coerce_float(getattr(pattern, "confidence", None))
        if combined is None:
            combined = confidence
        entry = {
            "source_word": context.original,
            "target_word": getattr(pattern, "target_word", None),
            "pattern": self._build_pattern(context.original, getattr(pattern, "target_word", None), False),
            "confidence": confidence or 0.0,
            "combined_score": combined or confidence or 0.0,
            "rarity_score": rarity,
            "llm_weakness_type": getattr(pattern, "llm_weakness_type", None),
            "cultural_depth": getattr(pattern, "cultural_depth", None),
            "result_source": "anti_llm",
            "source_rhyme_signatures": sorted(context.signatures),
            "source_phonetic_profile": context.phonetic_profile,
            "prosody_profile": self._normalise_dict(getattr(pattern, "prosody_profile", None)),
            "feature_profile": self._normalise_feature_profile(getattr(pattern, "feature_profile", None)),
            "stress_alignment": self._coerce_float(getattr(pattern, "stress_alignment", None)),
            "internal_rhyme_score": self._coerce_float(getattr(pattern, "internal_rhyme_score", None)),
            "bradley_device": getattr(pattern, "bradley_device", None),
            "syllable_span": list(getattr(pattern, "syllable_span", []) or []),
        }
        entry["is_multi_word"] = isinstance(entry["target_word"], str) and " " in entry["target_word"].strip()
        entry["result_variant"] = "multi_word" if entry["is_multi_word"] else "single_word"
        entry["pattern"] = self._build_pattern(context.original, entry["target_word"], entry["is_multi_word"])
        entry["target_phonetics"] = self._describe_word(getattr(self, "phonetic_analyzer", None), entry["target_word"])
        return entry

    # ------------------------------------------------------------------
    # Filtering helpers
    # ------------------------------------------------------------------
    def _entry_passes_filters(
        self,
        entry: Dict[str, Any],
        filters: SearchFilters,
        analyzer: Optional[EnhancedPhoneticAnalyzer],
    ) -> bool:
        if self._confidence_value(entry) < filters.min_confidence:
            return False

        if not filters.has_feature_filters:
            return True

        normalize = self.normalize_source_name

        if filters.cultural_filters:
            sig_value = entry.get("cultural_sig")
            if not sig_value or normalize(sig_value) not in filters.cultural_filters:
                return False

        if filters.genre_filters:
            genre_value = entry.get("genre")
            if not genre_value or normalize(genre_value) not in filters.genre_filters:
                return False

        if filters.rhyme_type_filters:
            rhyme_type = self._rhyme_type(entry)
            if not rhyme_type or normalize(rhyme_type) not in filters.rhyme_type_filters:
                return False

        if filters.bradley_filters:
            bradley = self._bradley_device(entry)
            if not bradley or normalize(bradley) not in filters.bradley_filters:
                return False

        if filters.min_rarity is not None:
            rarity = self._rarity_value(entry)
            if rarity is None or rarity < filters.min_rarity:
                return False

        if filters.min_stress_alignment is not None:
            stress = self._stress_score(entry)
            if stress is None or stress < filters.min_stress_alignment:
                return False

        if filters.cadence_focus is not None:
            cadence = self._cadence_tag(entry)
            if not cadence or self.normalize_source_name(cadence) != filters.cadence_focus:
                return False

        if filters.require_internal:
            internal = self._internal_score(entry)
            if internal is None or internal < 0.4:
                return False

        if filters.min_syllables is not None or filters.max_syllables is not None:
            syllables = self._target_syllable_count(entry, analyzer)
            if syllables is None:
                return False
            if filters.min_syllables is not None and syllables < filters.min_syllables:
                return False
            if filters.max_syllables is not None and syllables > filters.max_syllables:
                return False

        if filters.max_line_distance is not None:
            distance = self._coerce_float(entry.get("distance"))
            if distance is None or distance > filters.max_line_distance:
                return False

        return True

    def _confidence_value(self, entry: Dict[str, Any]) -> float:
        for key in ("combined_score", "confidence"):
            value = entry.get(key)
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return 0.0

    def _rarity_value(self, entry: Dict[str, Any]) -> Optional[float]:
        for key in ("rarity_score", "cultural_rarity"):
            value = entry.get(key)
            try:
                if value is None:
                    continue
                return float(value)
            except (TypeError, ValueError):
                continue
        return None

    def _stress_score(self, entry: Dict[str, Any]) -> Optional[float]:
        candidates = [
            entry.get("stress_alignment"),
            entry.get("rhythm_score"),
        ]
        profile = self._normalise_dict(entry.get("feature_profile"))
        if profile:
            candidates.append(profile.get("stress_alignment"))
        prosody = self._normalise_dict(entry.get("prosody_profile"))
        if prosody:
            candidates.append(prosody.get("stress_alignment"))
        for value in candidates:
            try:
                if value is None:
                    continue
                return float(value)
            except (TypeError, ValueError):
                continue
        return None

    def _cadence_tag(self, entry: Dict[str, Any]) -> Optional[str]:
        prosody = self._normalise_dict(entry.get("prosody_profile"))
        if prosody and prosody.get("complexity_tag"):
            return str(prosody["complexity_tag"])
        profile = self._normalise_dict(entry.get("feature_profile"))
        if profile and profile.get("complexity_tag"):
            return str(profile["complexity_tag"])
        return None

    def _rhyme_type(self, entry: Dict[str, Any]) -> Optional[str]:
        if entry.get("rhyme_type"):
            return str(entry["rhyme_type"])
        profile = self._normalise_dict(entry.get("feature_profile"))
        if profile and profile.get("rhyme_type"):
            return str(profile["rhyme_type"])
        features = self._normalise_dict(entry.get("phonetic_features"))
        if features and features.get("rhyme_type"):
            return str(features["rhyme_type"])
        return None

    def _bradley_device(self, entry: Dict[str, Any]) -> Optional[str]:
        if entry.get("bradley_device"):
            return str(entry["bradley_device"])
        profile = self._normalise_dict(entry.get("feature_profile"))
        if profile and profile.get("bradley_device"):
            return str(profile["bradley_device"])
        return None

    def _internal_score(self, entry: Dict[str, Any]) -> Optional[float]:
        if entry.get("internal_rhyme_score") is not None:
            try:
                return float(entry.get("internal_rhyme_score"))
            except (TypeError, ValueError):
                return None
        profile = self._normalise_dict(entry.get("feature_profile"))
        if profile and profile.get("internal_rhyme_score") is not None:
            try:
                return float(profile.get("internal_rhyme_score"))
            except (TypeError, ValueError):
                return None
        return None

    def _target_syllable_count(
        self,
        entry: Dict[str, Any],
        analyzer: Optional[EnhancedPhoneticAnalyzer],
    ) -> Optional[int]:
        span = entry.get("syllable_span")
        if isinstance(span, (list, tuple)) and len(span) == 2:
            try:
                return int(span[1])
            except (TypeError, ValueError):
                pass
        profile = self._normalise_feature_profile(entry.get("feature_profile"))
        if profile:
            span = profile.get("syllable_span")
            if isinstance(span, (list, tuple)) and len(span) == 2:
                try:
                    return int(span[1])
                except (TypeError, ValueError):
                    pass
        target_word = entry.get("target_word")
        if analyzer is None or not target_word:
            return None
        estimator = getattr(analyzer, "estimate_syllables", None)
        if callable(estimator):
            try:
                return int(estimator(target_word))
            except Exception:
                return None
        return None

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------
    def _build_pattern(self, source_word: str, target_word: Optional[str], is_multi: bool) -> str:
        connector = " // " if is_multi else " / "
        target = target_word or ""
        return f"{source_word}{connector}{target}".strip()

    def _describe_word(
        self,
        analyzer: Optional[EnhancedPhoneticAnalyzer],
        word: Optional[str],
    ) -> Dict[str, Any]:
        if analyzer is None or not word:
            return {}
        try:
            description = analyzer.describe_word(word)
        except Exception:
            return {}
        return self._normalise_dict(description)

    def _normalise_dict(self, value: Any) -> Dict[str, Any]:
        if isinstance(value, dict):
            return dict(value)
        if hasattr(value, "__dict__"):
            try:
                return dict(vars(value))
            except Exception:
                return {}
        try:
            return dict(value) if value is not None else {}
        except Exception:
            return {}

    def _normalise_feature_profile(self, value: Any) -> Dict[str, Any]:
        profile = self._normalise_dict(value)
        if not profile:
            return {}
        if "syllable_span" in profile:
            span = profile["syllable_span"]
            if isinstance(span, (list, tuple)) and len(span) == 2:
                try:
                    profile["syllable_span"] = [int(span[0]), int(span[1])]
                except (TypeError, ValueError):
                    profile.pop("syllable_span", None)
        return profile

    # ------------------------------------------------------------------
    # Result formatting
    # ------------------------------------------------------------------
    def format_rhyme_results(self, source_word: str, rhymes: Dict[str, Any]) -> str:
        """Render grouped rhyme results with shared phonetic context."""

        category_order: List[Tuple[str, str]] = [
            ("cmu", "📚 CMU — Uncommon Rhymes"),
            ("anti_llm", "🧠 Anti-LLM — Uncommon Patterns"),
            ("rap_db", "🎤 Rap & Cultural Matches"),
        ]

        if not rhymes or not any(rhymes.get(key) for key, _ in category_order):
            return f"❌ No rhymes found for '{source_word}'. Try another word or adjust your filters."

        lines: List[str] = [f"Results for {source_word}"]
        for key, title in category_order:
            entries = rhymes.get(key) or []
            if not entries:
                continue
            lines.append("")
            lines.append(title)
            for entry in entries:
                lines.extend(self._format_entry_lines(entry, key))

        return "\n".join(lines).strip()

    def _format_entry_lines(self, entry: Dict[str, Any], category_key: str) -> List[str]:
        display_word = str(entry.get("target_word", "")).upper() or "(unknown)"
        pattern = entry.get("pattern") or self._build_pattern(
            entry.get("source_word", ""), entry.get("target_word", ""), bool(entry.get("is_multi_word"))
        )
        rarity = self._rarity_value(entry)
        rarity_text = f"Rarity: {rarity:.2f}" if rarity is not None else "Rarity: —"
        entry_lines = [f"  - {display_word} — {pattern}", f"    {rarity_text}"]

        rhyme_type = self._rhyme_type(entry)
        if rhyme_type:
            entry_lines.append(f"    Rhyme type: {self._pretty_label(rhyme_type)}")

        cadence = self._cadence_tag(entry)
        if cadence:
            entry_lines.append(f"    Cadence: {self._pretty_label(cadence)}")

        context_dict = self._normalise_dict(entry.get("cultural_context"))
        if context_dict:
            era = context_dict.get("era")
            if era:
                entry_lines.append(f"    Cultural: Era: {self._pretty_label(era)}")
            region = context_dict.get("regional_origin")
            if region:
                entry_lines.append(f"    Region: {self._pretty_label(region)}")
            styles = context_dict.get("style_characteristics")
            if isinstance(styles, (list, tuple)) and styles:
                formatted_styles = ", ".join(self._pretty_label(style) for style in styles)
                entry_lines.append(f"    • Styles: {formatted_styles}")

        if category_key == "anti_llm":
            if entry.get("llm_weakness_type"):
                entry_lines.append(
                    f"    • LLM weakness: {self._pretty_label(entry['llm_weakness_type'])}"
                )
            if entry.get("cultural_depth"):
                entry_lines.append(
                    f"    • Cultural depth: {self._pretty_label(entry['cultural_depth'])}"
                )

        return entry_lines

    def _pretty_label(self, value: Any) -> str:
        if value is None:
            return ""
        text = str(value).replace("_", " ").strip()
        if not text:
            return ""
        return text.title()


__all__ = ["SearchService"]
