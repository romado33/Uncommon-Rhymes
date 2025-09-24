"""Search service orchestrating rhyme discovery and formatting."""

from __future__ import annotations

from collections import Counter, OrderedDict
from typing import Any, Dict, List, Optional, Set, Tuple
import re
import types

from rhyme_rarity.core import (
    EnhancedPhoneticAnalyzer,
    extract_phrase_components,
    get_cmu_rhymes,
)
from anti_llm import AntiLLMRhymeEngine
from cultural.engine import CulturalIntelligenceEngine

from ..data.database import SQLiteRhymeRepository



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
        """Clear memoized helper caches.

        Exposed so API consumers can explicitly reset caches if upstream data
        changes, e.g. when refreshing database content in long-running sessions.
        """

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
        except Exception:
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
        except Exception:
            results = set()

        results_set = set(results)
        self._related_words_cache[normalized_terms] = tuple(sorted(results_set))
        self._trim_cache(self._related_words_cache)

        return results_set

    def normalize_source_name(self, name: Optional[str]) -> str:
        if name is None:
            return ""
        return str(name).strip().lower().replace("_", "-")

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
            """Search for rhymes in the patterns.db database.

            Returns a mapping keyed by rhyme source so downstream consumers can
            format each category independently.
            """

            def _empty_response() -> Dict[str, List[Dict]]:
                return {"cmu": [], "anti_llm": [], "rap_db": []}

            if not source_word or not source_word.strip():
                return _empty_response()

            source_word = source_word.lower().strip()

            try:
                min_conf_threshold = float(min_confidence)
            except (TypeError, ValueError):
                min_conf_threshold = 0.0
            if min_conf_threshold < 0:
                min_conf_threshold = 0.0
            min_confidence = min_conf_threshold

            def _normalize_to_list(value: Optional[List[str] | str]) -> List[str]:
                """Return ``value`` coerced to a list of non-empty strings."""

                if value is None:
                    return []
                if isinstance(value, (list, tuple, set)):
                    return [str(v) for v in value if v is not None and str(v).strip()]
                if isinstance(value, str):
                    return [value] if value.strip() else []
                return [str(value)]

            normalize_name = self.normalize_source_name

            def _normalized_set(value: Optional[List[str] | str]) -> Set[str]:
                """Return a set of normalised filter names for user-supplied values."""

                return {
                    normalized
                    for normalized in (normalize_name(item) for item in _normalize_to_list(value))
                    if normalized
                }

            def _coerce_float(value: Any) -> Optional[float]:
                """Safely convert a value to ``float`` for numeric filters."""

                if value is None:
                    return None
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return None

            filter_stats: Dict[str, Dict[str, Any]] = {}

            def _category_stats(category: str) -> Dict[str, Any]:
                stats = filter_stats.setdefault(
                    category,
                    {"rejections": Counter(), "samples": {}, "accepted": 0},
                )
                rejections = stats.get("rejections")
                if not isinstance(rejections, Counter):
                    stats["rejections"] = Counter(rejections or {})
                if "samples" not in stats or not isinstance(stats["samples"], dict):
                    stats["samples"] = dict(stats.get("samples", {}))
                if "accepted" not in stats:
                    stats["accepted"] = 0
                return stats

            def _summarize_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
                return {
                    "target": entry.get("target_word"),
                    "confidence": _coerce_float(entry.get("confidence"))
                    or _coerce_float(entry.get("combined_score")),
                    "phonetic_sim": _coerce_float(entry.get("phonetic_sim")),
                    "rhyme_type": entry.get("rhyme_type"),
                }

            def _record_rejection(
                category: str,
                reason: str,
                entry: Optional[Dict[str, Any]] = None,
            ) -> None:
                stats = _category_stats(category)
                stats["rejections"][reason] += 1
                if entry:
                    samples = stats.setdefault("samples", {}).setdefault(reason, [])
                    if len(samples) < 3:
                        samples.append(dict(_summarize_entry(entry)))

            def _record_accept(category: str) -> None:
                stats = _category_stats(category)
                stats["accepted"] = int(stats.get("accepted", 0)) + 1

            def _prepare_confidence_defaults(entry: Dict) -> float:
                """Populate common confidence fields and return the comparison score."""

                combined_value = _coerce_float(entry.get("combined_score"))
                confidence_value = _coerce_float(entry.get("confidence"))

                score_for_filter = (
                    combined_value
                    if combined_value is not None
                    else confidence_value
                )
                if score_for_filter is None:
                    score_for_filter = 0.0

                entry["combined_score"] = (
                    combined_value if combined_value is not None else score_for_filter
                )
                entry["confidence"] = (
                    confidence_value if confidence_value is not None else score_for_filter
                )

                return score_for_filter

            cultural_filters = _normalized_set(cultural_significance)
            genre_filters = _normalized_set(genres)

            rhyme_type_filters = _normalized_set(allowed_rhyme_types)
            bradley_filters = _normalized_set(bradley_devices)
            cadence_focus_normalized = None
            if isinstance(cadence_focus, str) and cadence_focus.strip():
                cadence_focus_normalized = normalize_name(cadence_focus) or None

            min_syllable_threshold = None if min_syllables is None else max(1, int(min_syllables))
            max_syllable_threshold = None if max_syllables is None else max(1, int(max_syllables))
            min_rarity_threshold = None
            if min_rarity is not None:
                try:
                    min_rarity_threshold = float(min_rarity)
                except (TypeError, ValueError):
                    min_rarity_threshold = None
            min_stress_threshold = None
            if min_stress_alignment is not None:
                try:
                    min_stress_threshold = float(min_stress_alignment)
                except (TypeError, ValueError):
                    min_stress_threshold = None

            selected_sources = _normalized_set(result_sources)
            if not selected_sources:
                selected_sources = {"phonetic", "cultural", "anti-llm"}

            include_phonetic = "phonetic" in selected_sources
            include_cultural = "cultural" in selected_sources
            include_anti_llm = "anti-llm" in selected_sources

            filters_active = bool(
                cultural_filters
                or genre_filters
                or rhyme_type_filters
                or cadence_focus_normalized
                or (max_line_distance is not None)
                or (min_syllable_threshold is not None)
                or (max_syllable_threshold is not None)
                or (min_rarity_threshold is not None)
                or (min_stress_threshold is not None)
                or require_internal
                or bradley_filters
            )

            if limit <= 0:
                return _empty_response()

            analyzer = getattr(self, "phonetic_analyzer", None)
            cultural_engine = getattr(self, "cultural_engine", None)
            cmu_loader = None
            if analyzer is not None:
                cmu_loader = getattr(analyzer, "cmu_loader", None)
            if cmu_loader is None:
                cmu_loader = getattr(self, "cmu_loader", None)

            def _derive_signatures(word: Optional[str]) -> Set[str]:
                normalized = (word or "").strip().lower()
                if not normalized:
                    return set()

                signatures: Set[str] = set()

                if cultural_engine and hasattr(cultural_engine, "derive_rhyme_signatures"):
                    try:
                        signatures.update(
                            str(sig)
                            for sig in cultural_engine.derive_rhyme_signatures(normalized)
                            if sig
                        )
                    except Exception:
                        pass

                loader = None
                if analyzer is not None:
                    loader = getattr(analyzer, "cmu_loader", None)
                if loader is None:
                    loader = cmu_loader
                if loader is not None:
                    try:
                        loader_signatures = loader.get_rhyme_parts(normalized)
                    except Exception:
                        loader_signatures = []
                    signatures.update(str(sig) for sig in loader_signatures if sig)

                if analyzer and hasattr(analyzer, "derive_rhyme_signatures"):
                    try:
                        derived = analyzer.derive_rhyme_signatures(normalized)
                    except Exception:
                        derived = []
                    signatures.update(str(sig) for sig in derived if sig)

                if not signatures:
                    signatures.update(self._fallback_signature(normalized))

                return {str(sig) for sig in signatures if sig}

            source_components = extract_phrase_components(source_word, cmu_loader)
            source_anchor_word = source_components.anchor or source_word

            def _build_word_phonetics(word: Optional[str]) -> Dict[str, Any]:
                base_word = "" if word is None else str(word)
                defaults: Dict[str, Any] = {
                    "word": base_word,
                    "normalized": base_word.lower().strip(),
                    "syllables": None,
                    "stress_pattern": "",
                    "stress_pattern_display": "",
                    "meter_hint": None,
                    "metrical_foot": None,
                    "vowel_skeleton": "",
                    "consonant_tail": "",
                    "pronunciations": [],
                    "tokens": [],
                    "token_syllables": [],
                    "anchor_word": "",
                    "anchor_display": "",
                    "is_multi_word": False,
                }

                if analyzer and hasattr(analyzer, "describe_word"):
                    try:
                        description = analyzer.describe_word(base_word)
                    except Exception:
                        description = None
                    if description:
                        if isinstance(description, dict):
                            for key in defaults:
                                if key in description and description[key] not in (None, ""):
                                    defaults[key] = description[key]
                        else:
                            try:
                                desc_dict = dict(description)
                            except Exception:
                                desc_dict = {}
                            for key in defaults:
                                if key in desc_dict and desc_dict[key] not in (None, ""):
                                    defaults[key] = desc_dict[key]

                if not defaults.get("tokens") and defaults.get("normalized"):
                    defaults["tokens"] = defaults["normalized"].split()

                if not defaults.get("anchor_word") and defaults.get("tokens"):
                    defaults["anchor_word"] = defaults["tokens"][-1].lower()

                if not defaults.get("anchor_display") and defaults.get("tokens"):
                    defaults["anchor_display"] = defaults["tokens"][-1]

                defaults["is_multi_word"] = bool(
                    defaults.get("tokens") and len(defaults["tokens"]) > 1
                )
                if not defaults["is_multi_word"] and isinstance(defaults.get("normalized"), str):
                    defaults["is_multi_word"] = " " in defaults["normalized"]

                if defaults["syllables"] is None:
                    estimate_fn = getattr(analyzer, "estimate_syllables", None)
                    try:
                        if callable(estimate_fn):
                            defaults["syllables"] = estimate_fn(base_word)
                    except Exception:
                        defaults["syllables"] = None

                stress_pattern = defaults.get("stress_pattern")
                if stress_pattern and not defaults.get("stress_pattern_display"):
                    defaults["stress_pattern_display"] = "-".join(str(ch) for ch in str(stress_pattern))

                if not defaults.get("pronunciations"):
                    pronunciations: List[str] = []
                    if analyzer and hasattr(analyzer, "_get_pronunciation_variants"):
                        try:
                            variants = analyzer._get_pronunciation_variants(base_word)
                        except Exception:
                            variants = []
                        for phones in variants[:2]:
                            try:
                                stripped = " ".join(analyzer._strip_stress_markers(phones))
                            except Exception:
                                stripped = ""
                            if stripped:
                                pronunciations.append(stripped)
                    defaults["pronunciations"] = pronunciations

                return defaults

            def _sanitize_feature_profile(entry: Dict[str, Any]) -> Dict[str, Any]:
                profile_obj = entry.get("feature_profile")
                if profile_obj is None:
                    profile_dict: Dict[str, Any] = {}
                elif isinstance(profile_obj, dict):
                    profile_dict = dict(profile_obj)
                elif hasattr(profile_obj, "__dict__"):
                    try:
                        profile_dict = dict(vars(profile_obj))
                    except Exception:
                        profile_dict = {}
                else:
                    try:
                        profile_dict = dict(profile_obj)
                    except Exception:
                        profile_dict = {}

                bradley_value = None
                if "bradley_device" in profile_dict:
                    bradley_value = profile_dict.get("bradley_device")
                    profile_dict.pop("bradley_device", None)
                if entry.get("bradley_device") is not None and bradley_value is None:
                    bradley_value = entry.get("bradley_device")

                for banned_key in ("assonance_score", "consonance_score"):
                    if banned_key in profile_dict:
                        profile_dict.pop(banned_key, None)

                if bradley_value is not None:
                    entry["_bradley_device"] = bradley_value

                entry["feature_profile"] = profile_dict
                entry.pop("bradley_device", None)
                entry.pop("assonance_score", None)
                entry.pop("consonance_score", None)
                return profile_dict

            def _ensure_target_phonetics(entry: Dict[str, Any]) -> Dict[str, Any]:
                target_profile = entry.get("target_phonetics")
                if isinstance(target_profile, dict):
                    return target_profile
                target_word = entry.get("target_word")
                profile = _build_word_phonetics(target_word)
                entry["target_phonetics"] = profile
                return profile

            fallback_signature = self._fallback_signature

            source_signature_set = _derive_signatures(source_anchor_word or source_word)

            source_signature_list = sorted(source_signature_set)

            # Fetch more CMU candidates than requested results so that scoring
            # filters can discard low-quality matches without leaving the
            # response empty.
            reference_limit = max(limit * 2, 10)
            cmu_candidates = self._lookup_cmu_rhymes(
                source_word,
                reference_limit,
                analyzer,
                cmu_loader,
            )

            reference_similarity = 0.0
            for candidate in cmu_candidates:
                if isinstance(candidate, dict):
                    try:
                        score = float(candidate.get("similarity", candidate.get("score", 0.0)))
                    except (TypeError, ValueError):
                        continue
                else:
                    try:
                        score = float(candidate[1]) if len(candidate) > 1 else 0.0
                    except (TypeError, ValueError, IndexError):
                        score = 0.0
                if score > reference_similarity:
                    reference_similarity = score

            db_candidate_words: Set[str] = set()
            if analyzer and self.repository:
                lookup_terms = {source_word}
                if source_anchor_word and source_anchor_word != source_word:
                    lookup_terms.add(source_anchor_word)
                db_candidate_words = self._fetch_related_words_cached(lookup_terms)

                for candidate_word in db_candidate_words:
                    if not candidate_word or candidate_word == source_word:
                        continue
                    try:
                        score = float(analyzer.get_phonetic_similarity(source_word, candidate_word))
                    except Exception:
                        continue
                    if score > reference_similarity:
                        reference_similarity = score

            phonetic_threshold = 0.7
            if reference_similarity > 0:
                phonetic_threshold = max(0.6, min(0.92, reference_similarity - 0.1))

            source_phonetics = _build_word_phonetics(source_word)

            prefix_tokens = []
            suffix_tokens = []
            if source_components.anchor_index is not None:
                prefix_tokens = source_components.normalized_tokens[: source_components.anchor_index]
                suffix_tokens = source_components.normalized_tokens[source_components.anchor_index + 1 :]

            source_phonetic_profile = {
                "word": source_word,
                "threshold": phonetic_threshold,
                "reference_similarity": reference_similarity,
                "signatures": source_signature_list,
                "phonetics": source_phonetics,
                "anchor_word": source_anchor_word,
                "anchor_display": source_components.anchor_display or source_anchor_word,
                "phrase_tokens": source_components.normalized_tokens,
                "phrase_prefix": " ".join(prefix_tokens).strip(),
                "phrase_suffix": " ".join(suffix_tokens).strip(),
                "token_syllables": source_components.syllable_counts,
                "is_multi_word": bool(source_phonetics.get("is_multi_word")),
            }

            phonetic_entries: List[Dict] = []
            module1_seed_payload: List[Dict] = []
            aggregated_seed_signatures: Set[str] = set(source_signature_set)
            delivered_words_set: Set[str] = set()
            if include_phonetic:
                slice_limit = reference_limit if reference_limit else max(limit, 1)
                phonetic_matches = cmu_candidates[:slice_limit]
                rarity_source = analyzer if analyzer is not None else getattr(self, "phonetic_analyzer", None)
                def _collect_phonetic_entries(
                    enforce_signature: bool,
                    *,
                    track_rejections: bool = True,
                ) -> List[Dict[str, Any]]:
                    processed_entries: List[Dict[str, Any]] = []
                    for candidate in phonetic_matches:
                        is_multi_variant = False
                        variant_candidate: Optional[str] = None
                        candidate_tokens: List[str] = []
                        candidate_syllables: Optional[int] = None
                        phrase_prefix = None
                        phrase_suffix = None
                        prefix_display = None
                        suffix_display = None
                        candidate_source_phrase = None
                        anchor_display = None

                        if isinstance(candidate, dict):
                            target = (
                                candidate.get("word")
                                or candidate.get("target")
                                or candidate.get("candidate")
                            )
                            similarity = float(
                                candidate.get("similarity", candidate.get("score", 0.0))
                            )
                            rarity_value = float(
                                candidate.get("rarity", candidate.get("rarity_score", 0.0))
                            )
                            combined = float(
                                candidate.get("combined", candidate.get("combined_score", similarity))
                            )
                            is_multi_variant = bool(candidate.get("is_multi_word"))
                            variant_candidate = (
                                candidate.get("candidate")
                                or (target if isinstance(target, str) else None)
                            )
                            candidate_tokens = candidate.get("target_tokens") or []
                            candidate_syllables = candidate.get("candidate_syllables")
                            phrase_prefix = candidate.get("prefix")
                            phrase_suffix = candidate.get("suffix")
                            prefix_display = candidate.get("prefix_display")
                            suffix_display = candidate.get("suffix_display")
                            candidate_source_phrase = candidate.get("source_phrase")
                            anchor_display = candidate.get("anchor_display")
                        else:
                            try:
                                target = candidate[0]
                                similarity = float(candidate[1]) if len(candidate) > 1 else 0.0
                                if len(candidate) > 2:
                                    rarity_value = float(candidate[2])
                                elif rarity_source and hasattr(rarity_source, "get_rarity_score"):
                                    rarity_value = float(rarity_source.get_rarity_score(candidate[0]))
                                else:
                                    rarity_value = 0.0
                                combined = float(candidate[3]) if len(candidate) > 3 else similarity
                            except Exception:
                                target = candidate if isinstance(candidate, str) else None
                                similarity = 0.0
                                rarity_value = 0.0
                                combined = 0.0
                            variant_candidate = target if isinstance(target, str) else None

                        if not target:
                            continue

                        if variant_candidate is None and isinstance(target, str):
                            variant_candidate = target
                        target_text = str(target)
                        entry_is_multi = is_multi_variant or (" " in target_text.strip())
                        candidate_signatures = _derive_signatures(target_text)
                        matched_signatures = sorted(
                            sig for sig in candidate_signatures if sig in source_signature_set
                        )
                        signature_enforcement = analyzer is not None or cultural_engine is not None
                        signature_mismatch = (
                            signature_enforcement
                            and not entry_is_multi
                            and source_signature_set
                            and candidate_signatures
                            and not matched_signatures
                        )
                        if signature_mismatch and enforce_signature:
                            if track_rejections:
                                _record_rejection(
                                    "cmu",
                                    "signature_mismatch",
                                    {
                                        "target_word": target_text,
                                        "confidence": combined,
                                        "phonetic_sim": similarity,
                                    },
                                )
                            continue
                        entry_prefix = (
                            phrase_prefix
                            if phrase_prefix is not None
                            else source_phonetic_profile.get("phrase_prefix")
                        )
                        entry_suffix = (
                            phrase_suffix
                            if phrase_suffix is not None
                            else source_phonetic_profile.get("phrase_suffix")
                        )

                        candidate_source = (
                            candidate_source_phrase
                            if candidate_source_phrase is not None
                            else source_word
                        )

                        alignment = None
                        if cultural_engine and hasattr(cultural_engine, "evaluate_rhyme_alignment"):
                            try:
                                alignment = cultural_engine.evaluate_rhyme_alignment(
                                    source_word,
                                    target,
                                    threshold=phonetic_threshold,
                                    rhyme_signatures=source_signature_set,
                                    source_context=None,
                                    target_context=None,
                                )
                            except Exception:
                                alignment = None
                            if alignment is None:
                                if track_rejections:
                                    _record_rejection(
                                        "cmu",
                                        "alignment_failed",
                                        {
                                            "target_word": target_text,
                                            "confidence": combined,
                                            "phonetic_sim": similarity,
                                        },
                                    )
                                continue
                        else:
                            profile_payload: Dict[str, Any] = {}
                            if analyzer is not None:
                                try:
                                    profile_obj = analyzer.derive_rhyme_profile(
                                        source_word,
                                        target,
                                        similarity=similarity,
                                    )
                                except Exception:
                                    profile_obj = None
                                if profile_obj is not None:
                                    if hasattr(profile_obj, "as_dict"):
                                        try:
                                            profile_payload = dict(profile_obj.as_dict())
                                        except Exception:
                                            profile_payload = {}
                                    elif isinstance(profile_obj, dict):
                                        profile_payload = dict(profile_obj)
                                    else:
                                        try:
                                            profile_payload = dict(vars(profile_obj))
                                        except Exception:
                                            profile_payload = {}

                            alignment = {
                                "similarity": similarity,
                                "rarity": rarity_value,
                                "combined": combined,
                                "signature_matches": [],
                                "target_signatures": [],
                                "features": {},
                                "feature_profile": profile_payload,
                                "prosody_profile": {},
                            }

                        pattern_source = candidate_source or source_word
                        pattern_separator = " // " if entry_is_multi else " / "
                        entry = {
                            "source_word": pattern_source,
                            "target_word": target_text,
                            "artist": "CMU Pronouncing Dictionary",
                            "song": "Phonetic Match",
                            "pattern": f"{pattern_source}{pattern_separator}{target_text}",
                            "distance": None,
                            "confidence": combined,
                            "combined_score": combined,
                            "phonetic_sim": similarity,
                            "rarity_score": rarity_value,
                            "cultural_sig": "phonetic",
                            "genre": None,
                            "source_context": "Phonetic match suggested by the CMU Pronouncing Dictionary.",
                            "target_context": "",
                            "result_source": "phonetic",
                            "source_rhyme_signatures": source_signature_list,
                            "source_phonetic_profile": source_phonetic_profile,
                            "phonetic_threshold": phonetic_threshold,
                            "is_multi_word": entry_is_multi,
                            "result_variant": "multi_word" if entry_is_multi else "single_word",
                            "candidate_word": variant_candidate,
                            "phrase_prefix": entry_prefix,
                            "phrase_suffix": entry_suffix,
                            "prefix_display": prefix_display or entry_prefix,
                            "suffix_display": suffix_display or entry_suffix,
                            "target_tokens": candidate_tokens,
                            "candidate_syllables": candidate_syllables,
                            "source_phrase": pattern_source,
                            "source_anchor": source_anchor_word,
                            "anchor_display": anchor_display or source_phonetic_profile.get("anchor_display"),
                        }

                        entry["target_rhyme_signatures"] = sorted(candidate_signatures)
                        entry["matched_signatures"] = matched_signatures

                        entry["phonetic_sim"] = alignment.get("similarity", entry["phonetic_sim"])
                        if alignment.get("rarity") is not None:
                            entry["rarity_score"] = alignment["rarity"]
                        if alignment.get("combined") is not None:
                            entry["combined_score"] = alignment["combined"]
                            try:
                                entry["confidence"] = max(
                                    float(entry.get("confidence", 0.0)),
                                    float(alignment["combined"]),
                                )
                            except (TypeError, ValueError):
                                entry["confidence"] = alignment["combined"]
                        if alignment.get("rhyme_type"):
                            entry["rhyme_type"] = alignment["rhyme_type"]
                        alignment_matches = alignment.get("signature_matches")
                        if alignment_matches:
                            entry["matched_signatures"] = alignment_matches
                        elif not entry.get("matched_signatures"):
                            entry["matched_signatures"] = []

                        alignment_targets = alignment.get("target_signatures")
                        if alignment_targets:
                            entry["target_rhyme_signatures"] = alignment_targets
                        elif not entry.get("target_rhyme_signatures"):
                            entry["target_rhyme_signatures"] = []
                        features = alignment.get("features")
                        if features:
                            entry["phonetic_features"] = features

                        feature_profile_obj = alignment.get("feature_profile") or {}
                        feature_profile: Dict[str, Any] = {}
                        if feature_profile_obj:
                            if isinstance(feature_profile_obj, dict):
                                feature_profile = dict(feature_profile_obj)
                            else:
                                try:
                                    feature_profile = dict(feature_profile_obj)
                                except Exception:
                                    feature_profile = {}
                        entry["feature_profile"] = feature_profile
                        if feature_profile and entry.get("rhyme_type") and "rhyme_type" not in feature_profile:
                            feature_profile["rhyme_type"] = entry["rhyme_type"]
                        sanitized_profile = _sanitize_feature_profile(entry)
                        if sanitized_profile:
                            syllable_span = sanitized_profile.get("syllable_span")
                            if isinstance(syllable_span, (list, tuple)) and len(syllable_span) == 2:
                                entry["syllable_span"] = [int(syllable_span[0]), int(syllable_span[1])]
                            stress_alignment = sanitized_profile.get("stress_alignment")
                            if stress_alignment is not None:
                                entry["stress_alignment"] = stress_alignment
                            internal_score = sanitized_profile.get("internal_rhyme_score")
                            if internal_score is not None:
                                entry["internal_rhyme_score"] = internal_score

                        prosody_profile = alignment.get("prosody_profile")
                        if prosody_profile:
                            entry["prosody_profile"] = prosody_profile

                        if signature_mismatch and not enforce_signature:
                            diagnostics = entry.setdefault("diagnostics", {})
                            diagnostics.setdefault("signature_override", True)

                        _ensure_target_phonetics(entry)

                        score_for_filter = _prepare_confidence_defaults(entry)
                        if score_for_filter < min_confidence:
                            if track_rejections:
                                _record_rejection("cmu", "below_confidence", entry)
                            continue

                        processed_entries.append(entry)

                    return processed_entries

                phonetic_entries = _collect_phonetic_entries(True, track_rejections=True)
                if (
                    not phonetic_entries
                    and phonetic_matches
                    and _category_stats("cmu")["rejections"].get("signature_mismatch")
                ):
                    fallback_entries = _collect_phonetic_entries(False, track_rejections=False)
                    if fallback_entries:
                        stats = _category_stats("cmu")
                        stats.setdefault("signature_overrides", 0)
                        stats["signature_overrides"] = int(stats["signature_overrides"]) + len(
                            fallback_entries
                        )
                        phonetic_entries = fallback_entries

                phonetic_entries.sort(
                    key=lambda entry: (
                        entry.get("combined_score", entry.get("confidence", 0.0)),
                        entry.get("phonetic_sim", 0.0),
                    ),
                    reverse=True,
                )

                delivered_words_set = {
                    str(item.get("target_word", "")).strip().lower()
                    for item in phonetic_entries
                    if item.get("target_word")
                }

                rare_seed_candidates: List[Dict] = []
                for entry in phonetic_entries:
                    target_value = entry.get("target_word")
                    if not target_value:
                        continue

                    rarity_value = entry.get("rarity_score")
                    try:
                        rarity_float = float(rarity_value) if rarity_value is not None else 0.0
                    except (TypeError, ValueError):
                        rarity_float = 0.0

                    combined_value = entry.get("combined_score", entry.get("confidence", 0.0))
                    try:
                        combined_float = float(combined_value) if combined_value is not None else 0.0
                    except (TypeError, ValueError):
                        combined_float = 0.0

                    signature_values = entry.get("target_rhyme_signatures") or entry.get("matched_signatures")
                    if signature_values:
                        signature_set = {str(sig) for sig in signature_values if sig}
                    else:
                        signature_set = fallback_signature(str(target_value))

                    aggregated_seed_signatures.update(signature_set)

                    rare_seed_candidates.append(
                        {
                            "word": target_value,
                            "rarity": rarity_float,
                            "combined": combined_float,
                            "signatures": list(signature_set),
                            "feature_profile": entry.get("feature_profile", {}),
                            "prosody_profile": entry.get("prosody_profile", {}),
                        }
                    )

                rare_seed_candidates.sort(
                    key=lambda payload: (payload.get("rarity", 0.0), payload.get("combined", 0.0)),
                    reverse=True,
                )

                max_seed_count = max(3, min(6, limit))
                for candidate in rare_seed_candidates:
                    if len(module1_seed_payload) >= max_seed_count:
                        break
                    if candidate.get("rarity", 0.0) <= 0 and module1_seed_payload:
                        continue
                    module1_seed_payload.append(candidate)

            fetch_limit = None

            cultural_entries: List[Dict] = []

            def _context_to_dict(context_obj):
                if context_obj is None:
                    return None
                if isinstance(context_obj, dict):
                    return dict(context_obj)
                if hasattr(context_obj, "__dict__"):
                    return dict(vars(context_obj))
                return {"value": context_obj}

            def _enrich_with_cultural_context(entry: Dict) -> Optional[Dict]:
                engine = getattr(self, "cultural_engine", None)
                if not engine:
                    return entry

                entry.setdefault("source_phonetic_profile", source_phonetic_profile)
                entry.setdefault("source_rhyme_signatures", source_signature_list)
                entry.setdefault("phonetic_threshold", phonetic_threshold)
                entry.setdefault("matched_signatures", [])
                entry.setdefault("target_rhyme_signatures", [])

                context_data = None
                rarity_value = None

                try:
                    get_context = getattr(engine, "get_cultural_context", None)
                    if callable(get_context):
                        pattern_payload = {
                            "artist": entry.get("artist"),
                            "song": entry.get("song"),
                            "source_word": entry.get("source_word", source_word),
                            "target_word": entry.get("target_word"),
                            "pattern": entry.get("pattern"),
                            "cultural_significance": entry.get("cultural_sig"),
                        }
                        context_data = get_context(pattern_payload)
                    elif hasattr(engine, "find_cultural_patterns"):
                        finder = getattr(engine, "find_cultural_patterns")
                        if callable(finder):
                            patterns = finder(entry.get("source_word", source_word), limit=1)
                            if patterns:
                                pattern_info = patterns[0]
                                context_data = pattern_info.get("cultural_context")
                                rarity_value = pattern_info.get("cultural_rarity")

                    context_dict = _context_to_dict(context_data)
                    if context_dict:
                        entry["cultural_context"] = context_dict

                        rarity_fn = getattr(engine, "get_cultural_rarity_score", None)
                        if callable(rarity_fn):
                            try:
                                rarity_value = rarity_fn(context_data)
                            except (TypeError, AttributeError):
                                rarity_value = rarity_fn(context_dict)

                    if rarity_value is not None:
                        entry["cultural_rarity"] = rarity_value

                except Exception:
                    pass

                try:
                    evaluate_alignment = getattr(engine, "evaluate_rhyme_alignment", None)
                    if callable(evaluate_alignment):
                        alignment = evaluate_alignment(
                            entry.get("source_word", source_word),
                            entry.get("target_word"),
                            threshold=phonetic_threshold,
                            rhyme_signatures=source_signature_set,
                            source_context=entry.get("source_context"),
                            target_context=entry.get("target_context"),
                        )
                        if alignment is None:
                            return None
                        sim_value = alignment.get("similarity")
                        if sim_value is not None:
                            entry["phonetic_sim"] = sim_value
                        combined_value = alignment.get("combined")
                        if combined_value is not None:
                            entry["combined_score"] = combined_value
                            try:
                                entry["confidence"] = max(
                                    float(entry.get("confidence", 0.0)),
                                    float(combined_value),
                                )
                            except (TypeError, ValueError):
                                entry["confidence"] = combined_value
                        rarity_metric = alignment.get("rarity")
                        if rarity_metric is not None:
                            entry["rarity_score"] = rarity_metric
                        rhyme_type = alignment.get("rhyme_type")
                        if rhyme_type:
                            entry["rhyme_type"] = rhyme_type
                        entry["matched_signatures"] = alignment.get(
                            "signature_matches", entry.get("matched_signatures", [])
                        )
                        entry["target_rhyme_signatures"] = alignment.get(
                            "target_signatures", entry.get("target_rhyme_signatures", [])
                        )
                        features = alignment.get("features")
                        if features:
                            entry["phonetic_features"] = features
                        feature_profile_obj = alignment.get("feature_profile")
                        feature_profile: Dict[str, Any] = {}
                        if feature_profile_obj:
                            if isinstance(feature_profile_obj, dict):
                                feature_profile = dict(feature_profile_obj)
                            else:
                                try:
                                    feature_profile = dict(feature_profile_obj)
                                except Exception:
                                    feature_profile = {}
                        if feature_profile:
                            if entry.get("rhyme_type") and "rhyme_type" not in feature_profile:
                                feature_profile["rhyme_type"] = entry["rhyme_type"]
                            entry["feature_profile"] = feature_profile
                            syllable_span = feature_profile.get("syllable_span")
                            if isinstance(syllable_span, (list, tuple)) and len(syllable_span) == 2:
                                entry["syllable_span"] = [int(syllable_span[0]), int(syllable_span[1])]
                            stress_alignment = feature_profile.get("stress_alignment")
                            if stress_alignment is not None:
                                entry["stress_alignment"] = stress_alignment
                            internal_score = feature_profile.get("internal_rhyme_score")
                            if internal_score is not None:
                                entry["internal_rhyme_score"] = internal_score
                        prosody_profile = alignment.get("prosody_profile")
                        if prosody_profile:
                            entry["prosody_profile"] = prosody_profile
                except Exception:
                    pass

                _sanitize_feature_profile(entry)
                _ensure_target_phonetics(entry)
                return entry

            source_results: List[Tuple] = []
            target_results: List[Tuple] = []

            if include_cultural and self.repository:
                try:
                    source_results, target_results = self.repository.fetch_cultural_matches(
                        source_word,
                        min_confidence=min_confidence,
                        phonetic_threshold=phonetic_threshold,
                        cultural_filters=sorted(cultural_filters),
                        genre_filters=sorted(genre_filters),
                        max_line_distance=max_line_distance,
                        limit=fetch_limit,
                    )
                except Exception:
                    source_results, target_results = [], []

            def build_entry(row: Tuple, swap: bool = False) -> Optional[Dict]:
                entry = {
                    "source_word": row[0],
                    "target_word": row[1],
                    "artist": row[2],
                    "song": row[3],
                    "pattern": row[4],
                    "genre": row[5],
                    "distance": row[6],
                    "confidence": row[7],
                    "phonetic_sim": row[8],
                    "cultural_sig": row[9],
                    "source_context": row[10],
                    "target_context": row[11],
                    "result_source": "cultural",
                    "combined_score": row[7],
                    "source_phonetic_profile": source_phonetic_profile,
                    "source_rhyme_signatures": source_signature_list,
                    "phonetic_threshold": phonetic_threshold,
                }

                if swap:
                    swapped_source = row[1]
                    swapped_target = row[0]
                    swapped_pattern = entry["pattern"]
                    if swapped_source and swapped_target:
                        swapped_pattern = f"{swapped_source} / {swapped_target}"
                    entry = {
                        **entry,
                        "source_word": swapped_source,
                        "target_word": swapped_target,
                        "pattern": swapped_pattern,
                        "source_context": row[11],
                        "target_context": row[10],
                    }

                target_text = str(entry.get("target_word", ""))
                is_multi = " " in target_text.strip() if target_text else False
                entry.setdefault("is_multi_word", is_multi)
                entry.setdefault("result_variant", "multi_word" if is_multi else "single_word")
                if is_multi and target_text:
                    entry["pattern"] = f"{entry.get('source_word', source_word)} // {target_text}"
                entry.setdefault("source_phrase", entry.get("source_word", source_word))
                entry.setdefault("source_anchor", source_anchor_word)
                entry.setdefault("anchor_display", source_phonetic_profile.get("anchor_display"))
                entry.setdefault("phrase_prefix", source_phonetic_profile.get("phrase_prefix"))
                entry.setdefault("phrase_suffix", source_phonetic_profile.get("phrase_suffix"))

                enriched_entry = _enrich_with_cultural_context(entry)
                if enriched_entry is None:
                    _record_rejection("rap_db", "alignment_failed", entry)
                    return None
                return enriched_entry

            for row in source_results:
                enriched_entry = build_entry(row)
                if enriched_entry:
                    cultural_entries.append(enriched_entry)

            for row in target_results:
                enriched_entry = build_entry(row, swap=True)
                if enriched_entry:
                    cultural_entries.append(enriched_entry)

            cultural_entries.sort(
                key=lambda r: (
                    -float(r.get("confidence", 0.0)),
                    -float(r.get("phonetic_sim", 0.0)),
                )
            )

            anti_llm_entries: List[Dict] = []
            if include_anti_llm:
                threshold_rejected: List[Dict[str, Any]] = []
                anti_patterns = self.anti_llm_engine.generate_anti_llm_patterns(
                    source_word,
                    limit=limit,
                    module1_seeds=module1_seed_payload,
                    seed_signatures=aggregated_seed_signatures,
                    delivered_words=delivered_words_set,
                )
                for pattern in anti_patterns:
                    feature_profile = getattr(pattern, "feature_profile", {}) or {}
                    syllable_span = getattr(pattern, "syllable_span", None)
                    stress_alignment = getattr(pattern, "stress_alignment", None)
                    internal_rhyme = None
                    if isinstance(feature_profile, dict):
                        internal_rhyme = feature_profile.get("internal_rhyme_score")
                    confidence_float = _coerce_float(getattr(pattern, "confidence", None))
                    if confidence_float is None:
                        confidence_float = 0.0
                    combined_metric = getattr(pattern, "combined", None)
                    if combined_metric is None:
                        combined_metric = getattr(pattern, "combined_score", None)
                    combined_float = _coerce_float(combined_metric)
                    entry_payload = {
                        "source_word": source_word,
                        "target_word": pattern.target_word,
                        "pattern": f"{source_word} / {pattern.target_word}",
                        "confidence": confidence_float,
                        "rarity_score": pattern.rarity_score,
                        "llm_weakness_type": getattr(pattern, "llm_weakness_type", None),
                        "cultural_depth": getattr(pattern, "cultural_depth", None),
                        "result_source": "anti_llm",
                        "source_rhyme_signatures": source_signature_list,
                        "source_phonetic_profile": source_phonetic_profile,
                        "phonetic_threshold": phonetic_threshold,
                        "feature_profile": feature_profile,
                        "bradley_device": getattr(pattern, "bradley_device", None),
                        "stress_alignment": stress_alignment,
                        "internal_rhyme_score": internal_rhyme,
                    }
                    is_multi = " " in str(pattern.target_word or "").strip()
                    entry_payload["is_multi_word"] = is_multi
                    entry_payload["result_variant"] = "multi_word" if is_multi else "single_word"
                    if is_multi and pattern.target_word:
                        entry_payload["pattern"] = f"{source_word} // {pattern.target_word}"
                    entry_payload.setdefault("source_phrase", source_word)
                    entry_payload.setdefault("source_anchor", source_anchor_word)
                    entry_payload.setdefault("anchor_display", source_phonetic_profile.get("anchor_display"))
                    entry_payload.setdefault("phrase_prefix", source_phonetic_profile.get("phrase_prefix"))
                    entry_payload.setdefault("phrase_suffix", source_phonetic_profile.get("phrase_suffix"))
                    if isinstance(syllable_span, (list, tuple)) and len(syllable_span) == 2:
                        entry_payload["syllable_span"] = [int(syllable_span[0]), int(syllable_span[1])]
                    prosody_payload = getattr(pattern, "prosody_profile", None)
                    if isinstance(prosody_payload, dict):
                        entry_payload["prosody_profile"] = prosody_payload
                    entry_payload["combined_score"] = (
                        combined_float if combined_float is not None else confidence_float
                    )
                    candidate_signatures = _derive_signatures(pattern.target_word)
                    if candidate_signatures:
                        entry_payload["target_rhyme_signatures"] = sorted(candidate_signatures)
                        matched = [
                            sig for sig in candidate_signatures if sig in source_signature_set
                        ]
                        if matched:
                            entry_payload["matched_signatures"] = sorted(matched)

                    similarity_value = None
                    if analyzer is not None and pattern.target_word:
                        try:
                            similarity_value = float(
                                analyzer.get_phonetic_similarity(
                                    source_word, pattern.target_word
                                )
                            )
                        except Exception:
                            similarity_value = None
                    if similarity_value is not None:
                        entry_payload["phonetic_sim"] = similarity_value
                    if similarity_value is not None:
                        min_similarity = phonetic_threshold if phonetic_threshold is not None else 0.15
                        try:
                            min_similarity = float(min_similarity)
                        except (TypeError, ValueError):
                            min_similarity = 0.15
                        if min_similarity < 0.15:
                            min_similarity = 0.15
                        if similarity_value < min_similarity:
                            _record_rejection(
                                "anti_llm",
                                "below_phonetic_threshold",
                                entry_payload,
                            )
                            threshold_rejected.append(dict(entry_payload))
                            continue

                    _sanitize_feature_profile(entry_payload)
                    _ensure_target_phonetics(entry_payload)
                    score_for_filter = _prepare_confidence_defaults(entry_payload)
                    if score_for_filter < min_confidence:
                        _record_rejection("anti_llm", "below_confidence", entry_payload)
                        continue
                    anti_llm_entries.append(entry_payload)

                if not anti_llm_entries and threshold_rejected:
                    stats = _category_stats("anti_llm")
                    stats.setdefault("threshold_overrides", 0)
                    # Prefer higher confidence/rareness combinations when reviving entries.
                    threshold_rejected.sort(
                        key=lambda payload: (
                            float(payload.get("combined_score", payload.get("confidence", 0.0)) or 0.0),
                            float(payload.get("rarity_score") or 0.0),
                        ),
                        reverse=True,
                    )
                    for payload in threshold_rejected:
                        score_for_filter = _prepare_confidence_defaults(payload)
                        if score_for_filter < min_confidence:
                            continue
                        diagnostics = payload.setdefault("diagnostics", {})
                        diagnostics.setdefault("threshold_override", True)
                        _sanitize_feature_profile(payload)
                        _ensure_target_phonetics(payload)
                        anti_llm_entries.append(payload)
                        stats["threshold_overrides"] = int(stats["threshold_overrides"]) + 1

            def _passes_min_confidence(entry: Dict) -> bool:
                score = _prepare_confidence_defaults(entry)
                return score >= min_confidence

            def _ensure_feature_profile(entry: Dict) -> Dict[str, Any]:
                """Return a mutable feature profile dictionary for ``entry``."""

                profile_obj = entry.get("feature_profile")
                if isinstance(profile_obj, dict):
                    return profile_obj
                if profile_obj is None:
                    profile_dict: Dict[str, Any] = {}
                elif hasattr(profile_obj, "__dict__"):
                    try:
                        profile_dict = dict(vars(profile_obj))
                    except Exception:
                        profile_dict = {}
                else:
                    try:
                        profile_dict = dict(profile_obj)
                    except Exception:
                        profile_dict = {}
                entry["feature_profile"] = profile_dict
                return profile_dict

            def _extract_rhyme_category(entry: Dict) -> Optional[str]:
                """Pull a rhyme category label from an entry and normalise storage."""

                rhyme_value = entry.get("rhyme_type")
                if not rhyme_value:
                    features_obj = entry.get("phonetic_features")
                    if features_obj is not None:
                        if isinstance(features_obj, dict):
                            rhyme_value = features_obj.get("rhyme_type")
                        elif hasattr(features_obj, "__dict__"):
                            rhyme_value = vars(features_obj).get("rhyme_type")
                        else:
                            try:
                                rhyme_value = dict(features_obj).get("rhyme_type")
                            except Exception:
                                rhyme_value = None
                if not rhyme_value:
                    profile_obj = entry.get("feature_profile")
                    if profile_obj is not None:
                        if isinstance(profile_obj, dict):
                            rhyme_value = profile_obj.get("rhyme_type")
                        elif hasattr(profile_obj, "__dict__"):
                            rhyme_value = vars(profile_obj).get("rhyme_type")
                        else:
                            try:
                                rhyme_value = dict(profile_obj).get("rhyme_type")
                            except Exception:
                                rhyme_value = None
                if rhyme_value:
                    entry["rhyme_type"] = rhyme_value
                    profile_dict = _ensure_feature_profile(entry)
                    profile_dict.setdefault("rhyme_type", rhyme_value)
                return rhyme_value

            def _extract_rhythm_score(entry: Dict) -> Optional[float]:
                """Return a numeric rhythm/stress alignment score for an entry."""

                raw_value = entry.get("rhythm_score")
                if raw_value is None:
                    raw_value = entry.get("stress_alignment")
                if raw_value is None:
                    profile_obj = entry.get("feature_profile")
                    if profile_obj is not None:
                        if isinstance(profile_obj, dict):
                            raw_value = profile_obj.get("stress_alignment")
                        elif hasattr(profile_obj, "__dict__"):
                            raw_value = vars(profile_obj).get("stress_alignment")
                        else:
                            try:
                                raw_value = dict(profile_obj).get("stress_alignment")
                            except Exception:
                                raw_value = None
                if raw_value is None:
                    phonetic_features = entry.get("phonetic_features")
                    if phonetic_features is not None:
                        if isinstance(phonetic_features, dict):
                            raw_value = phonetic_features.get("stress_alignment")
                        elif hasattr(phonetic_features, "__dict__"):
                            raw_value = vars(phonetic_features).get("stress_alignment")
                        else:
                            try:
                                raw_value = dict(phonetic_features).get("stress_alignment")
                            except Exception:
                                raw_value = None
                try:
                    rhythm_float = float(raw_value)
                except (TypeError, ValueError):
                    return None

                entry["rhythm_score"] = rhythm_float
                entry["stress_alignment"] = rhythm_float
                profile_dict = _ensure_feature_profile(entry)
                profile_dict.setdefault("stress_alignment", rhythm_float)
                return rhythm_float

            baseline_stress_floor = 0.25
            baseline_rarity_floor = 0.05

            def _filter_entry(entry: Dict, category_key: str) -> bool:
                if not _passes_min_confidence(entry):
                    _record_rejection(category_key, "below_confidence", entry)
                    return False

                entry_threshold_value = entry.get("phonetic_threshold", phonetic_threshold)
                try:
                    threshold_value = float(entry_threshold_value)
                except (TypeError, ValueError):
                    threshold_value = float(phonetic_threshold) if phonetic_threshold is not None else None

                sim_value = _coerce_float(entry.get("phonetic_sim"))
                if threshold_value is not None and sim_value is not None:
                    if sim_value < threshold_value:
                        _record_rejection(category_key, "below_phonetic_threshold", entry)
                        return False

                entry_rhyme = _extract_rhyme_category(entry)
                rhythm_score = _extract_rhythm_score(entry)

                if rhythm_score is not None and rhythm_score < baseline_stress_floor:
                    _record_rejection(category_key, "weak_stress_alignment", entry)
                    return False

                rarity_metric = entry.get("rarity_score")
                if rarity_metric is None:
                    rarity_metric = entry.get("cultural_rarity")
                if rarity_metric is None and analyzer is not None:
                    target_word = entry.get("target_word")
                    if target_word:
                        try:
                            rarity_metric = float(analyzer.get_rarity_score(target_word))
                            entry["rarity_score"] = rarity_metric
                        except Exception:
                            rarity_metric = None
                rarity_value_for_baseline = _coerce_float(rarity_metric)
                if (
                    rarity_value_for_baseline is not None
                    and rarity_value_for_baseline < baseline_rarity_floor
                ):
                    _record_rejection(category_key, "low_rarity_signal", entry)
                    return False

                source_sig_values = entry.get("source_rhyme_signatures") or []
                source_sig_set = {str(sig) for sig in source_sig_values if sig}
                if not source_sig_set:
                    source_sig_set = set(source_signature_list)
                target_sig_values = entry.get("target_rhyme_signatures") or []
                target_sig_set = {str(sig) for sig in target_sig_values if sig}
                matched_sig_values = entry.get("matched_signatures") or []
                matched_sig_set = {str(sig) for sig in matched_sig_values if sig}
                enforce_signatures = analyzer is not None or cultural_engine is not None
                if category_key == "anti_llm":
                    enforce_signatures = False
                if (
                    enforce_signatures
                    and not entry.get("is_multi_word")
                    and source_sig_set
                    and target_sig_set
                    and not matched_sig_set
                ):
                    intersection = source_sig_set.intersection(target_sig_set)
                    if intersection:
                        entry["matched_signatures"] = sorted(intersection)
                    else:
                        _record_rejection(category_key, "signature_mismatch", entry)
                        return False

                if not filters_active:
                    return True

                if cultural_filters:
                    sig = entry.get("cultural_sig")
                    if sig is None or normalize_name(str(sig)) not in cultural_filters:
                        _record_rejection(category_key, "cultural_filter_mismatch", entry)
                        return False
                if genre_filters:
                    genre_value = entry.get("genre")
                    if genre_value is None or normalize_name(str(genre_value)) not in genre_filters:
                        _record_rejection(category_key, "genre_filter_mismatch", entry)
                        return False
                if rhyme_type_filters:
                    if not entry_rhyme or normalize_name(str(entry_rhyme)) not in rhyme_type_filters:
                        _record_rejection(category_key, "rhyme_type_mismatch", entry)
                        return False
                if bradley_filters:
                    device_value = entry.get("bradley_device")
                    if device_value is None and entry.get("feature_profile"):
                        device_value = entry["feature_profile"].get("bradley_device")
                    if device_value is None and entry.get("_bradley_device"):
                        device_value = entry.get("_bradley_device")
                    if (
                        device_value is None
                        or normalize_name(str(device_value)) not in bradley_filters
                    ):
                        _record_rejection(category_key, "bradley_filter_mismatch", entry)
                        return False
                if min_rarity_threshold is not None:
                    rarity_metric = entry.get("rarity_score")
                    if rarity_metric is None:
                        rarity_metric = entry.get("cultural_rarity")
                    try:
                        rarity_value = float(rarity_metric) if rarity_metric is not None else 0.0
                    except (TypeError, ValueError):
                        rarity_value = 0.0
                    if rarity_value < min_rarity_threshold:
                        _record_rejection(category_key, "below_rarity_threshold", entry)
                        return False
                if min_stress_threshold is not None:
                    if rhythm_score is None or rhythm_score < min_stress_threshold:
                        _record_rejection(category_key, "below_stress_threshold", entry)
                        return False
                if cadence_focus_normalized:
                    prosody = entry.get("prosody_profile") or {}
                    cadence_value = prosody.get("complexity_tag") if isinstance(prosody, dict) else None
                    if cadence_value is None:
                        cadence_value = entry.get("feature_profile", {}).get("complexity_tag")
                    if not cadence_value or normalize_name(str(cadence_value)) != cadence_focus_normalized:
                        _record_rejection(category_key, "cadence_mismatch", entry)
                        return False
                if require_internal:
                    internal_value = entry.get("internal_rhyme_score")
                    if internal_value is None and entry.get("feature_profile"):
                        internal_value = entry["feature_profile"].get("internal_rhyme_score")
                    try:
                        internal_float = float(internal_value) if internal_value is not None else 0.0
                    except (TypeError, ValueError):
                        internal_float = 0.0
                    if internal_float < 0.4:
                        _record_rejection(category_key, "internal_rhyme_required", entry)
                        return False
                if min_syllable_threshold is not None or max_syllable_threshold is not None:
                    syllable_span = entry.get("syllable_span")
                    if not syllable_span and entry.get("feature_profile"):
                        syllable_span = entry["feature_profile"].get("syllable_span")
                    if isinstance(syllable_span, (list, tuple)) and len(syllable_span) == 2:
                        try:
                            target_syllables = int(syllable_span[1])
                        except (TypeError, ValueError):
                            target_syllables = None
                    else:
                        target_word = entry.get("target_word")
                        try:
                            estimator = getattr(self.phonetic_analyzer, "estimate_syllables", None)
                            if callable(estimator):
                                target_syllables = estimator(str(target_word)) if target_word else None
                            else:
                                target_syllables = self.phonetic_analyzer._count_syllables(str(target_word)) if target_word else None
                        except Exception:
                            target_syllables = None
                    if min_syllable_threshold is not None and (target_syllables is None or target_syllables < min_syllable_threshold):
                        _record_rejection(category_key, "below_min_syllables", entry)
                        return False
                    if max_syllable_threshold is not None and (target_syllables is None or target_syllables > max_syllable_threshold):
                        _record_rejection(category_key, "above_max_syllables", entry)
                        return False
                if max_line_distance is not None:
                    distance_value = entry.get("distance")
                    if distance_value is None or distance_value > max_line_distance:
                        _record_rejection(category_key, "beyond_line_distance", entry)
                        return False

                return True

            def _rarity_value(entry: Dict) -> float:
                rarity_metric = entry.get("rarity_score")
                if rarity_metric is None:
                    rarity_metric = entry.get("cultural_rarity")
                try:
                    return float(rarity_metric)
                except (TypeError, ValueError):
                    return 0.0

            def _confidence_value(entry: Dict) -> float:
                metric = entry.get("combined_score")
                if metric is None:
                    metric = entry.get("confidence")
                try:
                    return float(metric)
                except (TypeError, ValueError):
                    return 0.0

            def _normalize_entry(entry: Dict, category_key: str) -> Dict:
                entry["result_source"] = category_key
                if not entry.get("source_phrase"):
                    entry["source_phrase"] = entry.get("source_word", source_word)
                entry.setdefault("source_anchor", source_anchor_word)
                entry.setdefault("anchor_display", source_phonetic_profile.get("anchor_display"))
                entry.setdefault("phrase_prefix", source_phonetic_profile.get("phrase_prefix"))
                entry.setdefault("phrase_suffix", source_phonetic_profile.get("phrase_suffix"))

                target_text = str(entry.get("target_word", ""))
                is_multi = bool(entry.get("is_multi_word"))
                if not is_multi and target_text:
                    is_multi = " " in target_text.strip()
                entry["is_multi_word"] = is_multi
                if not entry.get("result_variant"):
                    entry["result_variant"] = "multi_word" if is_multi else "single_word"

                if entry.get("pattern") and target_text:
                    connector = " // " if is_multi else " / "
                    source_value = str(entry.get("source_word", source_word))
                    entry["pattern"] = f"{source_value}{connector}{target_text}"
                elif not entry.get("pattern") and target_text:
                    connector = " // " if is_multi else " / "
                    entry["pattern"] = f"{source_word}{connector}{target_text}"
                if entry.get("combined_score") is None and entry.get("confidence") is not None:
                    entry["combined_score"] = entry["confidence"]
                if entry.get("confidence") is None and entry.get("combined_score") is not None:
                    entry["confidence"] = entry["combined_score"]

                feature_profile = entry.get("feature_profile")
                if feature_profile is None:
                    entry["feature_profile"] = {}
                elif not isinstance(feature_profile, dict):
                    try:
                        entry["feature_profile"] = dict(feature_profile)
                    except Exception:
                        entry["feature_profile"] = {}
                _sanitize_feature_profile(entry)

                prosody_profile = entry.get("prosody_profile")
                if prosody_profile is None:
                    entry["prosody_profile"] = {}

                for field, default in {
                    "rarity_score": None,
                    "cultural_rarity": None,
                    "phonetic_sim": None,
                    "rhyme_type": None,
                    "stress_alignment": None,
                    "internal_rhyme_score": None,
                    "cultural_context": None,
                    "llm_weakness_type": None,
                    "cultural_depth": None,
                }.items():
                    entry.setdefault(field, default)

                entry.setdefault("cultural_sig", None)
                entry.setdefault("genre", None)
                entry.setdefault("source_context", None)
                entry.setdefault("target_context", None)
                entry.setdefault("distance", None)
                _ensure_target_phonetics(entry)

                bradley_value = entry.pop("_bradley_device", None)
                if bradley_value is not None:
                    entry.setdefault("bradley_device", bradley_value)

                return entry

            def _process_entries(entries: List[Dict], category_key: str) -> List[Dict]:
                if not entries:
                    return []

                if category_key == "rap_db":
                    grouped: "OrderedDict[Tuple[str, str], Dict[str, Any]]" = OrderedDict()
                    for entry in entries:
                        normalized_entry = _normalize_entry(entry, category_key)
                        if not _filter_entry(normalized_entry, category_key):
                            continue

                        artist_key = str(normalized_entry.get("artist") or "").strip().lower()
                        song_key = str(normalized_entry.get("song") or "").strip().lower()
                        group_key = (artist_key, song_key)
                        group = grouped.get(group_key)
                        if group is None:
                            group = {
                                "artist": normalized_entry.get("artist"),
                                "song": normalized_entry.get("song"),
                                "genre": normalized_entry.get("genre"),
                                "cultural_sig": normalized_entry.get("cultural_sig"),
                                "cultural_context": normalized_entry.get("cultural_context"),
                                "result_source": category_key,
                                "grouped_targets": [],
                                "max_confidence": 0.0,
                                "max_phonetic_sim": 0.0,
                                "max_cultural_rarity": None,
                                "max_rarity_score": None,
                                "is_multi_word": False,
                                "pattern": None,
                                "source_context": None,
                                "target_context": None,
                                "prosody_profile": None,
                                "_rhyme_types": set(),
                            }
                            grouped[group_key] = group
                        else:
                            if not group.get("genre") and normalized_entry.get("genre"):
                                group["genre"] = normalized_entry.get("genre")
                            if not group.get("cultural_sig") and normalized_entry.get("cultural_sig"):
                                group["cultural_sig"] = normalized_entry.get("cultural_sig")
                            if not group.get("cultural_context") and normalized_entry.get("cultural_context"):
                                group["cultural_context"] = normalized_entry.get("cultural_context")

                        target_key = (
                            str(normalized_entry.get("target_word") or "").strip().lower(),
                            str(normalized_entry.get("pattern") or "").strip().lower(),
                        )
                        seen_targets = group.setdefault("_seen_targets", set())
                        if target_key in seen_targets:
                            continue
                        seen_targets.add(target_key)

                        target_payload = dict(normalized_entry)
                        target_payload.pop("source_word", None)

                        group["grouped_targets"].append(target_payload)
                        group["target_word"] = group["grouped_targets"][0].get("target_word")
                        if group.get("pattern") is None and target_payload.get("pattern"):
                            group["pattern"] = target_payload.get("pattern")
                        if group.get("source_context") is None and target_payload.get("source_context"):
                            group["source_context"] = target_payload.get("source_context")
                        if group.get("target_context") is None and target_payload.get("target_context"):
                            group["target_context"] = target_payload.get("target_context")
                        if target_payload.get("is_multi_word"):
                            group["is_multi_word"] = True
                        if group.get("prosody_profile") is None and target_payload.get("prosody_profile"):
                            group["prosody_profile"] = target_payload.get("prosody_profile")
                        rhythm_metric = target_payload.get("rhythm_score")
                        if rhythm_metric is None:
                            rhythm_metric = target_payload.get("stress_alignment")
                        try:
                            rhythm_float = float(rhythm_metric)
                        except (TypeError, ValueError):
                            rhythm_float = None
                        if rhythm_float is not None:
                            current_rhythm = group.get("rhythm_score")
                            if current_rhythm is None or rhythm_float > float(current_rhythm):
                                group["rhythm_score"] = rhythm_float
                                group["stress_alignment"] = rhythm_float
                        group["max_confidence"] = max(
                            group["max_confidence"],
                            _confidence_value(target_payload),
                        )
                        group["max_phonetic_sim"] = max(
                            group["max_phonetic_sim"],
                            float(target_payload.get("phonetic_sim") or 0.0),
                        )
                        rarity_metric = target_payload.get("cultural_rarity")
                        if rarity_metric is None:
                            rarity_metric = target_payload.get("rarity_score")
                        try:
                            rarity_float = float(rarity_metric)
                        except (TypeError, ValueError):
                            rarity_float = None
                        if rarity_float is not None:
                            current_cultural = group.get("max_cultural_rarity")
                            if current_cultural is None or rarity_float > current_cultural:
                                group["max_cultural_rarity"] = rarity_float
                        rarity_score_metric = target_payload.get("rarity_score")
                        try:
                            rarity_score_float = float(rarity_score_metric)
                        except (TypeError, ValueError):
                            rarity_score_float = None
                        if rarity_score_float is not None:
                            current_score = group.get("max_rarity_score")
                            if current_score is None or rarity_score_float > current_score:
                                group["max_rarity_score"] = rarity_score_float
                        target_rhyme_type = target_payload.get("rhyme_type")
                        if target_rhyme_type:
                            rhyme_set = group.setdefault("_rhyme_types", set())
                            rhyme_set.add(target_rhyme_type)

                    processed_groups: List[Dict[str, Any]] = []
                    for group in grouped.values():
                        targets = group.get("grouped_targets") or []
                        if not targets:
                            continue
                        group["group_size"] = len(targets)
                        group["confidence"] = group.get("max_confidence")
                        group["combined_score"] = group.get("max_confidence")
                        group["phonetic_sim"] = group.get("max_phonetic_sim")
                        if group.get("max_cultural_rarity") is not None:
                            group["cultural_rarity"] = group.get("max_cultural_rarity")
                        if group.get("max_rarity_score") is not None:
                            group["rarity_score"] = group.get("max_rarity_score")
                        rhyme_types = group.pop("_rhyme_types", set())
                        if len(rhyme_types) == 1:
                            group["rhyme_type"] = next(iter(rhyme_types))
                        elif not rhyme_types:
                            group["rhyme_type"] = group.get("rhyme_type")
                        else:
                            group["rhyme_type"] = None
                        group.pop("_seen_targets", None)
                        group.pop("max_cultural_rarity", None)
                        group.pop("max_rarity_score", None)
                        processed_groups.append(group)
                        _record_accept(category_key)

                    return processed_groups

                processed: List[Dict] = []
                seen_pairs = set()
                for entry in entries:
                    target_value = entry.get("target_word")
                    if not target_value:
                        continue

                    source_value = entry.get("source_word", source_word)
                    pair = (
                        str(source_value).strip().lower(),
                        str(target_value).strip().lower(),
                    )
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)

                    normalized_entry = _normalize_entry(entry, category_key)
                    passes_filters = _filter_entry(normalized_entry, category_key)
                    if (
                        not passes_filters
                        and category_key in {"cmu", "anti_llm"}
                        and not filters_active
                    ):
                        passes_filters = True
                    if not passes_filters:
                        continue

                    normalized_entry.pop("source_word", None)
                    processed.append(normalized_entry)
                    _record_accept(category_key)

                processed.sort(
                    key=lambda entry: (
                        _rarity_value(entry),
                        _confidence_value(entry),
                    ),
                    reverse=True,
                )

                if limit is None:
                    return processed

                try:
                    capped_limit = max(int(limit), 0)
                except (TypeError, ValueError):
                    capped_limit = len(processed)

                return processed[:capped_limit]

            cmu_results = _process_entries(
                phonetic_entries if include_phonetic else [], "cmu"
            )
            anti_llm_results = _process_entries(
                anti_llm_entries if include_anti_llm else [], "anti_llm"
            )
            cultural_results = _process_entries(
                cultural_entries if include_cultural else [], "rap_db"
            )

            telemetry_payload: Dict[str, Any] = {
                "source_word": source_word,
                "reference_similarity": reference_similarity,
                "phonetic_threshold": phonetic_threshold,
                "min_confidence": min_confidence,
                "source_signatures": source_signature_list,
                "filters": {},
                "raw_candidates": {
                    "cmu": len(phonetic_entries) if include_phonetic else 0,
                    "anti_llm": len(anti_llm_entries) if include_anti_llm else 0,
                    "rap_db": len(cultural_entries) if include_cultural else 0,
                },
            }

            def _accepted_total(category: str) -> int:
                stats = filter_stats.get(category, {})
                try:
                    accepted_count = int(stats.get("accepted", 0))
                except Exception:
                    accepted_count = 0
                if category == "cmu":
                    accepted_count = max(accepted_count, len(cmu_results))
                elif category == "anti_llm":
                    accepted_count = max(accepted_count, len(anti_llm_results))
                elif category == "rap_db":
                    accepted_count = max(accepted_count, len(cultural_results))
                return accepted_count

            for category_name, stats in filter_stats.items():
                rejections_counter = stats.get("rejections", Counter())
                if not isinstance(rejections_counter, Counter):
                    rejections_counter = Counter(rejections_counter)
                telemetry_payload["filters"][category_name] = {
                    "accepted": _accepted_total(category_name),
                    "rejections": dict(rejections_counter),
                    "samples": {
                        reason: list(samples)
                        for reason, samples in (stats.get("samples") or {}).items()
                    },
                }
                override_count = stats.get("signature_overrides")
                if override_count:
                    try:
                        telemetry_payload["filters"][category_name][
                            "signature_overrides"
                        ] = int(override_count)
                    except Exception:
                        telemetry_payload["filters"][category_name][
                            "signature_overrides"
                        ] = override_count
                threshold_override = stats.get("threshold_overrides")
                if threshold_override:
                    try:
                        telemetry_payload["filters"][category_name][
                            "threshold_overrides"
                        ] = int(threshold_override)
                    except Exception:
                        telemetry_payload["filters"][category_name][
                            "threshold_overrides"
                        ] = threshold_override

            for category_name, results in (
                ("cmu", cmu_results),
                ("anti_llm", anti_llm_results),
                ("rap_db", cultural_results),
            ):
                telemetry_payload["filters"].setdefault(
                    category_name,
                    {"accepted": _accepted_total(category_name), "rejections": {}, "samples": {}},
                )
                if category_name not in filter_stats:
                    telemetry_payload["filters"][category_name]["accepted"] = len(results)

            return {
                "source_profile": source_phonetic_profile,
                "cmu": cmu_results,
                "anti_llm": anti_llm_results,
                "rap_db": cultural_results,
                "telemetry": telemetry_payload,
            }




    def format_rhyme_results(self, source_word: str, rhymes: Dict[str, Any]) -> str:
        """Render grouped rhyme results with shared phonetic context."""

        category_order: List[Tuple[str, str]] = [
            ("cmu", "📚 CMU — Uncommon Rhymes"),
            ("anti_llm", "🧠 Anti-LLM — Uncommon Patterns"),
            ("rap_db", "🎤 Rap & Cultural Matches"),
        ]

        def _normalize_source_key(value: Optional[str]) -> str:
            mapping = {
                "phonetic": "cmu",
                "cultural": "rap_db",
                "anti_llm": "anti_llm",
                "anti-llm": "anti_llm",
                "multi_word": "anti_llm",
            }
            if value is None:
                return ""
            return mapping.get(str(value), str(value))

        if not rhymes:
            return f"❌ No rhymes found for '{source_word}'. Try another word or adjust your filters."

        has_entries = any(rhymes.get(key) for key, _ in category_order)
        if not has_entries:
            return f"❌ No rhymes found for '{source_word}'. Try another word or adjust your filters."

        def _resolve_rhyme_type(entry: Dict[str, Any]) -> Optional[str]:
            candidate = entry.get("rhyme_type")
            if not candidate:
                for attr in ("feature_profile", "phonetic_features"):
                    obj = entry.get(attr)
                    if obj is None:
                        continue
                    if isinstance(obj, dict):
                        candidate = obj.get("rhyme_type")
                    elif hasattr(obj, "__dict__"):
                        candidate = vars(obj).get("rhyme_type")
                    else:
                        try:
                            candidate = dict(obj).get("rhyme_type")
                        except Exception:
                            candidate = None
                    if candidate:
                        break
            if not candidate:
                return None
            return str(candidate).replace("_", " ").title()

        def _as_dict(value: Any) -> Dict[str, Any]:
            if isinstance(value, dict):
                return value
            if hasattr(value, "__dict__"):
                try:
                    return dict(vars(value))
                except Exception:
                    return {}
            return {}

        def _format_float(value: Any) -> Optional[str]:
            try:
                return f"{float(value):.2f}"
            except (TypeError, ValueError):
                return None

        def _format_phonetics_line(phonetics: Any) -> Optional[str]:
            if not isinstance(phonetics, dict):
                return None
            parts: List[str] = []
            syllables = phonetics.get("syllables")
            if isinstance(syllables, (int, float)):
                parts.append(f"Syllables: {int(syllables)}")
            stress_display = phonetics.get("stress_pattern_display") or phonetics.get(
                "stress_pattern"
            )
            if stress_display:
                parts.append(f"Stress: {stress_display}")
            meter_hint = phonetics.get("meter_hint")
            foot = phonetics.get("metrical_foot")
            if meter_hint:
                parts.append(f"Meter: {meter_hint}")
            elif foot:
                parts.append(f"Meter: {str(foot).title()}")
            if not parts:
                return None
            return f"Phonetics: {' | '.join(parts)}"

        def _resolve_source_label(
            source_key: str,
            entry: Dict[str, Any],
            parent: Optional[Dict[str, Any]] = None,
        ) -> Optional[str]:
            explicit_source = entry.get("result_source")
            if explicit_source is None and parent is not None:
                explicit_source = parent.get("result_source")
            normalized_source = _normalize_source_key(explicit_source) or source_key
            mapping = {
                "cmu": "CMU Pronouncing Dictionary",
                "anti_llm": "Anti-LLM Pattern Library",
                "rap_db": "Rap & Cultural Archive",
            }
            if normalized_source in mapping:
                return mapping[normalized_source]
            if explicit_source:
                return str(explicit_source)
            return mapping.get(source_key, source_key.replace("_", " ").title())

        def _standard_info_segments(
            subject: Dict[str, Any],
            source_key: str,
            parent: Optional[Dict[str, Any]] = None,
        ) -> List[str]:
            segments: List[str] = []
            rhyme_category = "Phrase" if subject.get("is_multi_word") else "Word"
            segments.append(f"Rhyme Type: {rhyme_category}")

            syllable_span = subject.get("syllable_span")
            if not syllable_span and parent is not None:
                syllable_span = parent.get("syllable_span")
            if not syllable_span:
                feature_profile = subject.get("feature_profile") or {}
                if isinstance(feature_profile, dict):
                    syllable_span = feature_profile.get("syllable_span")
                if not syllable_span and parent is not None:
                    parent_profile = parent.get("feature_profile") or {}
                    if isinstance(parent_profile, dict):
                        syllable_span = parent_profile.get("syllable_span")
            if isinstance(syllable_span, (list, tuple)) and len(syllable_span) == 2:
                segments.append(
                    f"Syllable span: {syllable_span[0]}–{syllable_span[1]}"
                )

            phonetics_line = _format_phonetics_line(subject.get("target_phonetics") or {})
            if phonetics_line:
                segments.append(phonetics_line)

            stress_value = subject.get("stress_alignment")
            if stress_value is None:
                stress_value = subject.get("rhythm_score")
            if stress_value is None and parent is not None:
                stress_value = parent.get("stress_alignment") or parent.get("rhythm_score")
            stress_formatted = _format_float(stress_value)
            if stress_formatted:
                segments.append(f"Stress align: {stress_formatted}")

            prosody_profile = subject.get("prosody_profile")
            if not isinstance(prosody_profile, dict) and parent is not None:
                parent_prosody = parent.get("prosody_profile")
                if isinstance(parent_prosody, dict):
                    prosody_profile = parent_prosody
            if isinstance(prosody_profile, dict):
                cadence = prosody_profile.get("complexity_tag")
                if cadence:
                    segments.append(
                        f"Cadence: {str(cadence).replace('_', ' ').title()}"
                    )
                cadence_ratio = _format_float(prosody_profile.get("cadence_ratio"))
                if cadence_ratio:
                    segments.append(f"Cadence ratio: {cadence_ratio}")

            rarity_value = subject.get("rarity_score")
            if rarity_value is None:
                rarity_value = subject.get("cultural_rarity")
            if rarity_value is None and parent is not None:
                rarity_value = parent.get("rarity_score") or parent.get("cultural_rarity")
            rarity_formatted = _format_float(rarity_value)
            if rarity_formatted:
                segments.append(f"Rarity: {rarity_formatted}")

            rhyme_label = _resolve_rhyme_type(subject) or (
                _resolve_rhyme_type(parent) if parent is not None else None
            )
            if rhyme_label:
                segments.append(f"Rhyme type: {rhyme_label}")

            confidence_value = subject.get("combined_score")
            if confidence_value is None:
                confidence_value = subject.get("confidence")
            if confidence_value is None and parent is not None:
                confidence_value = parent.get("combined_score") or parent.get("confidence")
            confidence_formatted = _format_float(confidence_value)
            if confidence_formatted:
                segments.append(f"Confidence: {confidence_formatted}")

            signature_matches = subject.get("matched_signatures")
            if not signature_matches and parent is not None:
                signature_matches = parent.get("matched_signatures")
            if signature_matches:
                try:
                    match_count = len(signature_matches)
                except TypeError:
                    match_count = None
                if match_count:
                    segments.append(f"Signature matches: {match_count}")

            source_label = _resolve_source_label(source_key, subject, parent)
            if source_label:
                segments.append(f"Source: {source_label}")

            return [segment for segment in segments if segment]

        lines: List[str] = [
            f"🎯 **TARGET RHYMES for '{source_word.upper()}':**",
            "=" * 50,
            "",
        ]

        source_profile = rhymes.get("source_profile") or {}
        telemetry_info = _as_dict(rhymes.get("telemetry"))
        source_phonetics = source_profile.get("phonetics") or {}
        lines.append("🔎 Source profile")
        lines.append(f"   • Word: {source_profile.get('word', source_word)}")

        basic_parts: List[str] = []
        syllables = source_phonetics.get("syllables")
        if isinstance(syllables, (int, float)):
            basic_parts.append(f"Syllables: {int(syllables)}")
        stress_display = source_phonetics.get("stress_pattern_display") or source_phonetics.get(
            "stress_pattern"
        )
        if stress_display:
            basic_parts.append(f"Stress: {stress_display}")
        meter_hint = source_phonetics.get("meter_hint")
        foot = source_phonetics.get("metrical_foot")
        if meter_hint:
            basic_parts.append(f"Meter: {meter_hint}")
        elif foot:
            basic_parts.append(f"Meter: {str(foot).title()}")
        if basic_parts:
            lines.append(f"   • Basic: {' | '.join(basic_parts)}")

        metrics_parts: List[str] = []
        threshold_value = telemetry_info.get("phonetic_threshold")
        if threshold_value is None:
            threshold_value = source_profile.get("threshold")
        threshold_formatted = _format_float(threshold_value)
        if threshold_formatted:
            metrics_parts.append(f"Threshold: {threshold_formatted}")

        reference_value = telemetry_info.get("reference_similarity")
        if reference_value is None:
            reference_value = source_profile.get("reference_similarity")
        reference_formatted = _format_float(reference_value)
        if reference_formatted:
            metrics_parts.append(f"Reference: {reference_formatted}")

        min_conf_value = telemetry_info.get("min_confidence")
        min_conf_formatted = _format_float(min_conf_value)
        if min_conf_formatted:
            metrics_parts.append(f"Min confidence: {min_conf_formatted}")

        if metrics_parts:
            lines.append(f"   • Filters: {', '.join(metrics_parts)}")

        signature_list = telemetry_info.get("source_signatures") or source_profile.get("signatures")
        if isinstance(signature_list, (list, tuple)) and signature_list:
            lines.append(f"   • Signatures: {len(signature_list)} captured")

        anchor_display = source_profile.get("anchor_display") or source_profile.get("anchor_word")
        if anchor_display:
            prefix_text = source_profile.get("phrase_prefix") or ""
            suffix_text = source_profile.get("phrase_suffix") or ""
            context_bits = []
            if prefix_text:
                context_bits.append(prefix_text)
            context_bits.append(str(anchor_display))
            if suffix_text:
                context_bits.append(suffix_text)
            lines.append(f"   • Anchor: {' | '.join(context_bits)}")
        lines.append("")

        for key, header in category_order:
            entries = rhymes.get(key) or []
            if not entries:
                continue

            lines.append(header)
            lines.append("-" * len(header))

            if key == "rap_db":
                for entry in entries:
                    targets = entry.get("grouped_targets") or []
                    if not targets:
                        continue

                    artist = entry.get("artist")
                    song = entry.get("song")
                    if artist and song:
                        artist_segment = f"{artist} — {song}"
                    elif artist:
                        artist_segment = str(artist)
                    elif song:
                        artist_segment = str(song)
                    else:
                        artist_segment = None

                    genre_bits: List[str] = []
                    genre_value = entry.get("genre")
                    if genre_value:
                        genre_bits.append(f"Genre: {genre_value}")
                    group_size = entry.get("group_size")
                    if group_size:
                        try:
                            genre_bits.append(f"Targets: {int(group_size)}")
                        except (TypeError, ValueError):
                            genre_bits.append(f"Targets: {group_size}")

                    for target in targets:
                        pattern_text = target.get("pattern") or target.get("target_word") or "(unknown)"
                        subject_label = f"Pattern: {pattern_text}"

                        line_segments = [subject_label]
                        line_segments.extend(
                            _standard_info_segments(target, key, parent=entry)
                        )

                        metadata_segments: List[str] = []
                        if artist_segment:
                            metadata_segments.append(artist_segment)
                        if genre_bits:
                            metadata_segments.append(" | ".join(genre_bits))

                        hidden_segments: List[str] = []
                        context_info = _as_dict(entry.get("cultural_context"))
                        cultural_bits: List[str] = []
                        for key_name, label in (
                            ("era", "Era"),
                            ("regional_origin", "Region"),
                            ("cultural_significance", "Significance"),
                        ):
                            value = context_info.get(key_name)
                            if not value and key_name == "cultural_significance":
                                value = entry.get("cultural_sig")
                            if value:
                                cultural_bits.append(
                                    f"{label}: {str(value).replace('_', ' ').title()}"
                                )
                        if cultural_bits:
                            hidden_segments.append(
                                f"<!-- • Cultural: {' | '.join(cultural_bits)} -->"
                            )

                        styles = context_info.get("style_characteristics")
                        if isinstance(styles, (list, tuple)) and styles:
                            formatted_styles = ", ".join(
                                str(style).replace("_", " ").title() for style in styles if style
                            )
                            if formatted_styles:
                                hidden_segments.append(
                                    f"<!-- • Styles: {formatted_styles} -->"
                                )

                        context_parts: List[str] = []
                        for value in (
                            target.get("source_context"),
                            target.get("target_context"),
                        ):
                            if value:
                                context_parts.append(str(value))
                        if context_parts:
                            metadata_segments.append(f"Context: {' | '.join(context_parts)}")

                        if metadata_segments:
                            line_segments.extend(metadata_segments)

                        prosody = target.get("prosody_profile")
                        if isinstance(prosody, dict):
                            cadence = prosody.get("complexity_tag")
                            if cadence:
                                hidden_segments.append(
                                    f"<!-- • Cadence: {str(cadence).replace('_', ' ').title()} -->"
                                )

                        if hidden_segments:
                            line_segments.extend(hidden_segments)

                        formatted_line = " • ".join(
                            segment for segment in line_segments if segment
                        )
                        if formatted_line:
                            lines.append(formatted_line)
                            lines.append("")

                lines.append("")
                continue

            for entry in entries:
                target_word = entry.get("target_word") or "(unknown)"
                subject_label = f"**{str(target_word).upper()}**"

                line_segments = [subject_label]
                line_segments.extend(_standard_info_segments(entry, key))

                if key == "anti_llm":
                    hidden_segments: List[str] = []
                    weakness = entry.get("llm_weakness_type")
                    if weakness:
                        hidden_segments.append(
                            f"<!-- • LLM weakness: {str(weakness).replace('_', ' ').title()} -->"
                        )
                    cultural_depth = entry.get("cultural_depth")
                    if cultural_depth:
                        hidden_segments.append(
                            f"<!-- • Cultural depth: {cultural_depth} -->"
                        )
                    if hidden_segments:
                        line_segments.extend(hidden_segments)

                formatted_line = " • ".join(segment for segment in line_segments if segment)
                if formatted_line:
                    lines.append(formatted_line)
                    lines.append("")

            lines.append("")

        filter_diagnostics = telemetry_info.get("filters") if telemetry_info else {}
        if isinstance(filter_diagnostics, dict) and filter_diagnostics:
            lines.append("### Diagnostics")
            diagnostics_labels = {
                "cmu": "CMU filters",
                "anti_llm": "Anti-LLM filters",
                "rap_db": "Cultural filters",
            }
            for diag_key, stats in filter_diagnostics.items():
                label = diagnostics_labels.get(
                    _normalize_source_key(diag_key),
                    diag_key.replace("_", " ").title(),
                )
                accepted_value = int(stats.get("accepted", 0)) if isinstance(stats, dict) else 0
                rejections = stats.get("rejections", {}) if isinstance(stats, dict) else {}
                rejection_items = [
                    f"{str(reason).replace('_', ' ')}: {int(count)}"
                    for reason, count in sorted(rejections.items())
                    if count
                ]
                rejection_summary = ", ".join(rejection_items) if rejection_items else "no rejections"
                lines.append(f"- {label}: {accepted_value} kept; {rejection_summary}")
            lines.append("")

        return "\n".join(lines).strip() + "\n"

