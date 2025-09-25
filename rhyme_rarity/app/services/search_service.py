"""Search service orchestrating rhyme discovery and formatting."""

from __future__ import annotations

from collections import OrderedDict
from html import escape
from contextlib import contextmanager, nullcontext
import copy
import re
import threading
import types
from typing import Any, Dict, Generator, List, Optional, Set, Tuple

from rhyme_rarity.core import (
    CmuRhymeRepository,
    DefaultCmuRhymeRepository,
    EnhancedPhoneticAnalyzer,
    PhraseComponents,
    extract_phrase_components,
)
from anti_llm import AntiLLMRhymeEngine
from cultural.engine import CulturalIntelligenceEngine

from ..data.database import SQLiteRhymeRepository
from ...utils.observability import (
    add_span_attributes,
    create_counter,
    create_histogram,
    get_logger,
    record_exception,
    start_span,
)
from ...utils.telemetry import StructuredTelemetry



class RhymeQueryOrchestrator:
    """Coordinates rhyme searches across analyzers and repositories."""

    def __init__(
        self,
        *,
        repository: SQLiteRhymeRepository,
        phonetic_analyzer: Optional[EnhancedPhoneticAnalyzer] = None,
        cultural_engine: Optional[CulturalIntelligenceEngine] = None,
        anti_llm_engine: Optional[AntiLLMRhymeEngine] = None,
        cmu_loader: Optional[object] = None,
        cmu_repository: Optional[CmuRhymeRepository] = None,
        max_concurrent_searches: Optional[int] = None,
        search_timeout: Optional[float] = None,
        telemetry: Optional[StructuredTelemetry] = None,
    ) -> None:
        self.repository = repository
        self.phonetic_analyzer = phonetic_analyzer
        self.cultural_engine = cultural_engine
        self.anti_llm_engine = anti_llm_engine
        self.cmu_loader = cmu_loader
        self.cmu_repository: CmuRhymeRepository = (
            cmu_repository or DefaultCmuRhymeRepository()
        )
        if self.cmu_loader is None and phonetic_analyzer is not None:
            self.cmu_loader = getattr(phonetic_analyzer, "cmu_loader", None)

        self.telemetry = telemetry or StructuredTelemetry()
        self._latest_trace: Dict[str, Any] = {}

        repository_name = type(repository).__name__
        self._logger = get_logger(__name__).bind(
            component="rhyme_query_orchestrator",
            repository=repository_name,
        )

        self._metric_request_total = create_counter(
            "rhyme_search_requests_total",
            "Total rhyme search requests received.",
        )
        self._metric_request_failures = create_counter(
            "rhyme_search_request_failures_total",
            "Total rhyme search requests that raised an exception.",
        )
        self._metric_request_duration = create_histogram(
            "rhyme_search_request_seconds",
            "Latency of rhyme search requests.",
        )
        self._metric_cache_hits = create_counter(
            "rhyme_search_cache_hits_total",
            "Cache hits recorded by the rhyme search orchestrator.",
            label_names=("cache",),
        )
        self._metric_cache_misses = create_counter(
            "rhyme_search_cache_misses_total",
            "Cache misses recorded by the rhyme search orchestrator.",
            label_names=("cache",),
        )
        self._metric_fallback_events = create_counter(
            "rhyme_search_fallback_events_total",
            "Fallback events triggered within the rhyme search pipeline.",
            label_names=("stage",),
        )

        self._cache_lock = threading.RLock()
        self._max_cache_entries = 512
        self._fallback_signature_cache: OrderedDict[str, Tuple[str, ...]] = OrderedDict()
        self._cmu_rhyme_cache: OrderedDict[
            Tuple[str, int, Optional[int], Optional[int]], Tuple[Any, ...]
        ] = OrderedDict()
        self._related_words_cache: OrderedDict[Tuple[str, ...], Tuple[str, ...]] = OrderedDict()

        self._search_timeout = None
        if search_timeout is not None:
            try:
                timeout_value = float(search_timeout)
            except (TypeError, ValueError):
                timeout_value = None
            else:
                if timeout_value >= 0:
                    self._search_timeout = timeout_value

        self._search_semaphore: Optional[threading.BoundedSemaphore] = None
        if max_concurrent_searches is not None:
            try:
                max_concurrent = int(max_concurrent_searches)
            except (TypeError, ValueError):
                max_concurrent = 0
            if max_concurrent > 0:
                self._search_semaphore = threading.BoundedSemaphore(max_concurrent)

        self._logger.info(
            "Rhyme query orchestrator initialised",
            context={
                "db_path": getattr(repository, "db_path", None),
                "max_cache_entries": self._max_cache_entries,
                "max_concurrent_searches": max_concurrent_searches,
                "search_timeout": self._search_timeout,
            },
        )

    def set_phonetic_analyzer(self, analyzer: Optional[EnhancedPhoneticAnalyzer]) -> None:
        self.phonetic_analyzer = analyzer
        if analyzer is not None and getattr(analyzer, "cmu_loader", None):
            self.cmu_loader = getattr(analyzer, "cmu_loader")
        self._reset_phonetic_caches()

    def set_cultural_engine(self, engine: Optional[CulturalIntelligenceEngine]) -> None:
        self.cultural_engine = engine

    def set_anti_llm_engine(self, engine: Optional[AntiLLMRhymeEngine]) -> None:
        self.anti_llm_engine = engine

    def set_cmu_repository(self, repository: Optional[CmuRhymeRepository]) -> None:
        self.cmu_repository = repository or DefaultCmuRhymeRepository()
        self._reset_phonetic_caches()

    def clear_cached_results(self) -> None:
        """Clear memoized helper caches.

        Exposed so API consumers can explicitly reset caches if upstream data
        changes, e.g. when refreshing database content in long-running sessions.
        """

        self._logger.info(
            "Clearing search caches",
            context={"caches": ["fallback_signature", "cmu_rhyme", "related_words"]},
        )

        with self._cache_lock:
            self._fallback_signature_cache.clear()
            self._cmu_rhyme_cache.clear()
            self._related_words_cache.clear()

        analyzer = getattr(self, "phonetic_analyzer", None)
        if analyzer and hasattr(analyzer, "clear_cached_results"):
            try:
                analyzer.clear_cached_results()
            except Exception:
                pass

        cultural_engine = getattr(self, "cultural_engine", None)
        if cultural_engine and hasattr(cultural_engine, "clear_cached_results"):
            try:
                cultural_engine.clear_cached_results()
            except Exception:
                pass

        anti_engine = getattr(self, "anti_llm_engine", None)
        if anti_engine and hasattr(anti_engine, "clear_cached_results"):
            try:
                anti_engine.clear_cached_results()
            except Exception:
                pass

    def get_latest_telemetry(self) -> Dict[str, Any]:
        """Return the most recent telemetry snapshot for the search service."""

        telemetry = getattr(self, "telemetry", None)
        if telemetry is None:
            return {}

        if not self._latest_trace:
            return telemetry.latest_snapshot()

        return copy.deepcopy(self._latest_trace)

    def _reset_phonetic_caches(self) -> None:
        """Drop caches derived from analyzer or CMU resources."""

        with self._cache_lock:
            self._cmu_rhyme_cache.clear()
            self._related_words_cache.clear()

        analyzer = getattr(self, "phonetic_analyzer", None)
        if analyzer and hasattr(analyzer, "clear_cached_results"):
            try:
                analyzer.clear_cached_results()
            except Exception:
                pass

    def _trim_cache(self, cache: OrderedDict[Any, Tuple[Any, ...]]) -> None:
        """Ensure caches stay within the configured ``_max_cache_entries`` size."""

        if self._max_cache_entries <= 0:
            cache.clear()
            return

        while len(cache) > self._max_cache_entries:
            cache.popitem(last=False)

    def _record_cache_event(
        self,
        cache_name: str,
        *,
        hit: bool,
        span: Optional[Any] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        metric = self._metric_cache_hits if hit else self._metric_cache_misses
        metric.labels(cache=cache_name).inc()
        context = {"cache": cache_name, "hit": hit}
        if details:
            context.update({key: value for key, value in details.items() if value is not None})
        log = self._logger.debug if hit else self._logger.info
        log(
            "Cache lookup %s",
            "hit" if hit else "miss",
            context=context,
        )
        if span is not None:
            span_attributes = {f"cache.{cache_name}.hit": hit}
            if details and "size" in details:
                span_attributes[f"cache.{cache_name}.size"] = details["size"]
            add_span_attributes(span, span_attributes)

    def _record_fallback(
        self,
        stage: str,
        *,
        reason: str,
        span: Optional[Any] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._metric_fallback_events.labels(stage=stage).inc()
        context = {"stage": stage, "reason": reason}
        if details:
            context.update({key: value for key, value in details.items() if value is not None})
        self._logger.info("Fallback engaged", context=context)
        if span is not None:
            span_attributes = {f"fallback.{stage}": True, f"fallback.{stage}.reason": reason}
            if details and "word" in details:
                span_attributes[f"fallback.{stage}.word"] = details["word"]
            add_span_attributes(span, span_attributes)

    def _build_spelling_signature(self, word: Optional[str]) -> Set[str]:
        """Build and cache a spelling-based signature (last vowel and ending) for fallback rhyme comparisons."""

        cache_key = (word or "").strip().lower()
        with start_span(
            "search.cache.fallback_signature",
            {"word": cache_key or ""},
        ) as span:
            with self._cache_lock:
                cached = self._fallback_signature_cache.get(cache_key)
                if cached is not None:
                    self._fallback_signature_cache.move_to_end(cache_key)
                    self._record_cache_event(
                        "fallback_signature",
                        hit=True,
                        span=span,
                        details={"word": cache_key, "size": len(cached)},
                    )
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

            with self._cache_lock:
                existing = self._fallback_signature_cache.get(cache_key)
                if existing is not None:
                    self._fallback_signature_cache.move_to_end(cache_key)
                    self._record_cache_event(
                        "fallback_signature",
                        hit=True,
                        span=span,
                        details={"word": cache_key, "size": len(existing)},
                    )
                    return set(existing)

                self._fallback_signature_cache[cache_key] = signature_tuple
                self._trim_cache(self._fallback_signature_cache)

            self._record_cache_event(
                "fallback_signature",
                hit=False,
                span=span,
                details={"word": cache_key, "size": len(signature_tuple)},
            )

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
        with start_span(
            "search.cache.cmu_lookup",
            {"source_word": source_word, "limit": int(limit)},
        ) as span:
            with self._cache_lock:
                cached = self._cmu_rhyme_cache.get(cache_key)
                if cached is not None:
                    self._cmu_rhyme_cache.move_to_end(cache_key)
                    self._record_cache_event(
                        "cmu_rhyme",
                        hit=True,
                        span=span,
                        details={"word": source_word, "limit": limit, "size": len(cached)},
                    )
                    return list(cached)

            repository = getattr(self, "cmu_repository", None)
            try:
                if repository is None:
                    results: List[Any] = []
                else:
                    results = repository.lookup(
                        source_word,
                        limit,
                        analyzer=analyzer,
                        cmu_loader=cmu_loader,
                    )
            except Exception as exc:
                self._logger.warning(
                    "CMU repository lookup failed",
                    context={"source_word": source_word, "limit": limit, "error": str(exc)},
                )
                record_exception(span, exc)
                results = []

            cached_results = tuple(results)

            with self._cache_lock:
                existing = self._cmu_rhyme_cache.get(cache_key)
                if existing is not None:
                    self._cmu_rhyme_cache.move_to_end(cache_key)
                    self._record_cache_event(
                        "cmu_rhyme",
                        hit=True,
                        span=span,
                        details={"word": source_word, "limit": limit, "size": len(existing)},
                    )
                    return list(existing)

                self._cmu_rhyme_cache[cache_key] = cached_results
                self._trim_cache(self._cmu_rhyme_cache)

            self._record_cache_event(
                "cmu_rhyme",
                hit=False,
                span=span,
                details={"word": source_word, "limit": limit, "size": len(cached_results)},
            )

            return list(cached_results)

    def _fetch_related_words_cached(self, lookup_terms: Set[str]) -> Set[str]:
        """Return repository suggestions using an LRU-style cache."""

        if not lookup_terms or self.repository is None:
            return set()

        normalized_terms = tuple(
            sorted({(term or "").strip().lower() for term in lookup_terms if term})
        )
        if not normalized_terms:
            return set()

        with start_span(
            "search.cache.related_words",
            {"terms": ",".join(normalized_terms)},
        ) as span:
            with self._cache_lock:
                cached = self._related_words_cache.get(normalized_terms)
                if cached is not None:
                    self._related_words_cache.move_to_end(normalized_terms)
                    self._record_cache_event(
                        "related_words",
                        hit=True,
                        span=span,
                        details={"terms": len(normalized_terms), "size": len(cached)},
                    )
                    return set(cached)

            try:
                results = self.repository.fetch_related_words(set(normalized_terms))
            except Exception as exc:
                self._logger.warning(
                    "Related words lookup failed",
                    context={"terms": list(normalized_terms), "error": str(exc)},
                )
                record_exception(span, exc)
                results = set()

            results_set = set(results)
            cached_tuple = tuple(sorted(results_set))

            with self._cache_lock:
                existing = self._related_words_cache.get(normalized_terms)
                if existing is not None:
                    self._related_words_cache.move_to_end(normalized_terms)
                    self._record_cache_event(
                        "related_words",
                        hit=True,
                        span=span,
                        details={"terms": len(normalized_terms), "size": len(existing)},
                    )
                    return set(existing)

                self._related_words_cache[normalized_terms] = cached_tuple
                self._trim_cache(self._related_words_cache)

            self._record_cache_event(
                "related_words",
                hit=False,
                span=span,
                details={"terms": len(normalized_terms), "size": len(cached_tuple)},
            )

            return results_set

    def _get_phrase_components_cached(
        self,
        phrase: str,
        cmu_loader: Optional[object],
        analyzer: Optional[EnhancedPhoneticAnalyzer],
    ) -> PhraseComponents:
        """Return phrase components using analyzer caching when available."""

        if analyzer and hasattr(analyzer, "get_phrase_components"):
            try:
                return analyzer.get_phrase_components(phrase, cmu_loader)
            except Exception:
                pass

        return extract_phrase_components(phrase, cmu_loader)

    @contextmanager
    def _search_slot(self) -> Generator[None, None, None]:
        """Bound concurrent searches when a semaphore has been configured."""

        semaphore = self._search_semaphore
        if semaphore is None:
            telemetry = getattr(self, "telemetry", None)
            if telemetry:
                telemetry.increment("search.gate.bypass")
            yield
            return

        timeout = self._search_timeout
        telemetry = getattr(self, "telemetry", None)
        wait_context = telemetry.timer("search.gate.wait") if telemetry else nullcontext()
        with wait_context:
            if timeout is None:
                semaphore.acquire()
                acquired = True
            else:
                acquired = semaphore.acquire(timeout=timeout)

        if not acquired:
            if telemetry:
                telemetry.increment("search.gate.timeout")
            raise TimeoutError("Search capacity exhausted; please retry later")

        if telemetry:
            telemetry.increment("search.gate.acquired")

        try:
            yield
        finally:
            semaphore.release()
            if telemetry:
                telemetry.increment("search.gate.released")

    def normalize_filter_label(self, name: Optional[str]) -> str:
        """Normalise user-supplied filter labels by trimming, lowercasing, and replacing underscores for consistent comparisons."""

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
        """Search for rhymes while applying optional concurrency backpressure."""
        telemetry = getattr(self, "telemetry", None)
        request_context: Dict[str, Any] = {
            "source_word": source_word,
            "limit": limit,
            "min_confidence": min_confidence,
        }
        if result_sources:
            request_context["result_sources"] = list(result_sources)
        if cultural_significance:
            request_context["cultural_significance"] = list(cultural_significance)
        if genres:
            request_context["genres"] = list(genres)

        if telemetry:
            telemetry.start_trace("search_rhymes")
            telemetry.increment("search.invoked")
            telemetry.annotate("input.raw_source_word", source_word)
            telemetry.annotate("input.limit", limit)
            telemetry.annotate("input.min_confidence", min_confidence)
            telemetry.annotate("input.result_sources", result_sources)
            telemetry.annotate("input.cultural_significance", cultural_significance)
            telemetry.annotate("input.genres", genres)

        self._metric_request_total.inc()
        self._logger.info("Search request received", context=request_context)

        with start_span("search.request", request_context) as request_span:
            try:
                with self._metric_request_duration.time():
                    with self._search_slot():
                        result = self._search_rhymes_internal(
                            source_word,
                            limit=limit,
                            min_confidence=min_confidence,
                            cultural_significance=cultural_significance,
                            genres=genres,
                            result_sources=result_sources,
                            max_line_distance=max_line_distance,
                            min_syllables=min_syllables,
                            max_syllables=max_syllables,
                            allowed_rhyme_types=allowed_rhyme_types,
                            bradley_devices=bradley_devices,
                            require_internal=require_internal,
                            min_rarity=min_rarity,
                            min_stress_alignment=min_stress_alignment,
                            cadence_focus=cadence_focus,
                        )
            except Exception as exc:
                failure_context = dict(request_context)
                failure_context["error"] = str(exc)
                self._metric_request_failures.inc()
                self._logger.error("Search request failed", context=failure_context)
                record_exception(request_span, exc)
                if telemetry:
                    telemetry.increment("search.failed")
                    self._latest_trace = telemetry.snapshot()
                else:
                    self._latest_trace = {}
                raise

            counts = {
                key: len(value)
                for key, value in result.items()
                if isinstance(value, list)
            }
            self._logger.info(
                "Search request completed",
                context={"result_counts": counts, "source_word": source_word},
            )

            if telemetry:
                telemetry.annotate("result.counts", counts)
                telemetry.increment("search.completed")
                self._latest_trace = telemetry.snapshot()
            else:
                self._latest_trace = {}

            add_span_attributes(
                request_span,
                {
                    "search.success": True,
                    "result.total": sum(counts.values()),
                    "result.sources": ",".join(sorted(counts)),
                },
            )

            return result

    def _describe_word(
        self,
        analyzer: Optional[EnhancedPhoneticAnalyzer],
        word: Optional[str],
    ) -> Dict[str, Any]:
        """Return a consistent phonetic description for ``word``."""

        base_word = (word or '').strip()
        profile: Dict[str, Any] = {
            'word': word or '',
            'normalized': base_word.lower(),
            'tokens': [] if not base_word else base_word.lower().split(),
            'syllables': None,
            'anchor_word': '',
            'anchor_display': word or '',
            'is_multi_word': False,
        }

        if analyzer is None:
            tokens = profile['tokens']
            if tokens:
                profile['anchor_word'] = tokens[-1]
                profile['anchor_display'] = tokens[-1]
                profile['is_multi_word'] = len(tokens) > 1
            return profile

        try:
            description = analyzer.describe_word(base_word)
        except Exception:
            description = None

        data: Dict[str, Any] = {}
        if isinstance(description, dict):
            data = dict(description)
        elif description is not None:
            try:
                data = dict(description)
            except Exception:
                data = {}

        result = {**profile, **{k: v for k, v in data.items() if v is not None}}
        tokens = result.get('tokens')
        if not tokens:
            normalized = result.get('normalized', base_word.lower())
            tokens = normalized.split() if normalized else []
            result['tokens'] = tokens
        result['is_multi_word'] = bool(tokens and len(tokens) > 1 or ' ' in result.get('normalized', ''))
        if tokens:
            result.setdefault('anchor_word', tokens[-1])
            result.setdefault('anchor_display', tokens[-1])
        elif result.get('normalized'):
            result.setdefault('anchor_word', result['normalized'].split()[-1])
            result.setdefault('anchor_display', result['anchor_word'])
        if result.get('syllables') is None:
            try:
                result['syllables'] = analyzer.estimate_syllables(base_word)
            except Exception:
                pass
        return result

    def _build_source_context(
        self,
        source_word: str,
        analyzer: Optional[EnhancedPhoneticAnalyzer],
        cmu_loader: Optional[object],
        cultural_engine: Optional[CulturalIntelligenceEngine],
    ) -> Dict[str, Any]:
        """Assemble reusable context shared by downstream collectors."""

        components = self._get_phrase_components_cached(source_word, cmu_loader, analyzer)
        anchor_word = components.anchor or source_word
        anchor_display = components.anchor_display or anchor_word

        prefix_tokens: List[str] = []
        suffix_tokens: List[str] = []
        if components.anchor_index is not None:
            prefix_tokens = components.normalized_tokens[: components.anchor_index]
            suffix_tokens = components.normalized_tokens[components.anchor_index + 1 :]

        phonetics = self._describe_word(analyzer, source_word)

        signature_provenance: Dict[str, Set[str]] = {}
        cultural_signatures: Set[str] = set()
        if cultural_engine and hasattr(cultural_engine, 'derive_rhyme_signatures'):
            try:
                cultural_signatures = {
                    str(sig)
                    for sig in cultural_engine.derive_rhyme_signatures(source_word)
                    if sig
                }
            except Exception:
                cultural_signatures = set()
        if cultural_signatures:
            signature_provenance['cultural'] = set(cultural_signatures)

        fallback_signatures = self._build_spelling_signature(source_word)
        if fallback_signatures:
            signature_provenance['spelling'] = set(fallback_signatures)

        signature_set: Set[str] = set()
        for values in signature_provenance.values():
            signature_set.update(values)
        if not signature_set:
            signature_set = set(fallback_signatures)

        profile_signature_provenance = {
            key: sorted(value)
            for key, value in signature_provenance.items()
            if value
        }

        profile = {
            'word': source_word,
            'anchor_word': anchor_word,
            'anchor_display': anchor_display,
            'phrase_tokens': list(components.normalized_tokens),
            'phrase_prefix': ' '.join(prefix_tokens).strip(),
            'phrase_suffix': ' '.join(suffix_tokens).strip(),
            'token_syllables': list(components.syllable_counts),
            'phonetics': phonetics,
            'signatures': sorted(signature_set),
            'threshold': 0.7,
            'reference_similarity': 0.0,
            'is_multi_word': bool(phonetics.get('is_multi_word')),
            'signature_provenance': profile_signature_provenance,
        }

        return {
            'word': source_word,
            'components': components,
            'profile': profile,
            'signature_set': signature_set,
            'threshold': 0.7,
            'reference_similarity': 0.0,
            'signature_provenance': signature_provenance,
        }

    def _update_threshold_from_similarity(self, context: Dict[str, Any], similarity: float) -> None:
        best_similarity = max(context.get('reference_similarity', 0.0), float(similarity))
        if best_similarity <= 0:
            return
        context['reference_similarity'] = best_similarity
        threshold = max(0.6, min(0.92, best_similarity - 0.1))
        context['threshold'] = threshold
        profile = context.get('profile', {})
        if isinstance(profile, dict):
            profile['reference_similarity'] = best_similarity
            profile['threshold'] = threshold

    def _compose_pattern(self, source_word: str, target_word: str) -> str:
        source_text = str(source_word or '').strip()
        target_text = str(target_word or '').strip()
        separator = ' // ' if (' ' in source_text or ' ' in target_text) else ' / '
        if not source_text:
            return target_text
        if not target_text:
            return source_text
        return f"{source_text}{separator}{target_text}"

    def _collect_phonetic_matches(
        self,
        source_word: str,
        limit: int,
        *,
        analyzer: Optional[EnhancedPhoneticAnalyzer],
        cultural_engine: Optional[CulturalIntelligenceEngine],
        cmu_loader: Optional[object],
        context: Dict[str, Any],
        min_confidence: float,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Set[str], Set[str]]:
        """Return CMU phonetic matches plus seed payload metadata."""

        source_profile = context.get('profile', {})
        signature_set: Set[str] = set(context.get('signature_set') or [])
        telemetry = getattr(self, 'telemetry', None)
        context_provenance = context.get('signature_provenance') or {}
        available_context_sources: Dict[str, Set[str]] = {
            key: set(value)
            for key, value in context_provenance.items()
            if value
        }
        enforce_multi_source = len(available_context_sources) >= 2

        try:
            candidates = self.cmu_repository.lookup(
                source_word,
                max(limit * 2, limit + 5),
                analyzer=analyzer,
                cmu_loader=cmu_loader,
            )
        except Exception as exc:
            self._logger.error(
                'CMU lookup failed',
                context={'error': str(exc), 'source_word': source_word},
            )
            return [], [], set(), set()

        matches: List[Dict[str, Any]] = []
        delivered_words: Set[str] = set()
        aggregated_signatures: Set[str] = set(signature_set)

        for candidate in candidates:
            target_word: str = ''
            similarity = 0.0
            rarity = 0.0
            combined = 0.0
            candidate_tokens: List[str] = []
            candidate_syllables: Optional[int] = None
            is_multi_variant = False

            if isinstance(candidate, dict):
                target_word = str(
                    candidate.get('word')
                    or candidate.get('target')
                    or candidate.get('candidate')
                    or ''
                ).strip()
                similarity = float(candidate.get('similarity', candidate.get('score', 0.0)) or 0.0)
                rarity = float(candidate.get('rarity', candidate.get('rarity_score', 0.0)) or 0.0)
                combined = float(candidate.get('combined', candidate.get('combined_score', similarity)) or similarity)
                candidate_tokens = list(candidate.get('target_tokens') or [])
                candidate_syllables = candidate.get('candidate_syllables')
                is_multi_variant = bool(candidate.get('is_multi_word'))
            elif isinstance(candidate, (tuple, list)) and candidate:
                target_word = str(candidate[0]).strip()
                if len(candidate) > 1:
                    try:
                        similarity = float(candidate[1])
                    except (TypeError, ValueError):
                        similarity = 0.0
                if len(candidate) > 2:
                    try:
                        rarity = float(candidate[2])
                    except (TypeError, ValueError):
                        rarity = 0.0
                if len(candidate) > 3:
                    try:
                        combined = float(candidate[3])
                    except (TypeError, ValueError):
                        combined = similarity
            else:
                continue

            if not target_word or target_word.lower() == source_word.lower():
                continue

            self._update_threshold_from_similarity(context, similarity)
            delivered_words.add(target_word.strip().lower())

            entry: Dict[str, Any] = {
                'source_word': source_word,
                'target_word': target_word,
                'pattern': self._compose_pattern(source_word, target_word),
                'confidence': combined,
                'combined_score': combined,
                'phonetic_sim': similarity,
                'rarity_score': rarity,
                'result_source': 'phonetic',
                'target_tokens': candidate_tokens,
                'candidate_syllables': candidate_syllables,
                'source_phonetic_profile': source_profile,
                'source_rhyme_signatures': list(signature_set),
                'phonetic_threshold': context.get('threshold'),
                'is_multi_word': bool(is_multi_variant or (' ' in target_word.strip())),
                'result_variant': 'multi_word' if is_multi_variant or (' ' in target_word.strip()) else 'single_word',
            }

            if analyzer is not None:
                entry['target_phonetics'] = self._describe_word(analyzer, target_word)
            else:
                entry['target_phonetics'] = self._describe_word(None, target_word)

            candidate_signature_map: Dict[str, Set[str]] = {}
            fallback_signature_set = self._build_spelling_signature(target_word)
            if fallback_signature_set:
                candidate_signature_map['spelling'] = set(fallback_signature_set)

            alignment = None
            alignment_successful = False
            if cultural_engine and hasattr(cultural_engine, 'evaluate_rhyme_alignment'):
                try:
                    alignment = cultural_engine.evaluate_rhyme_alignment(
                        source_word,
                        target_word,
                        threshold=context.get('threshold'),
                        rhyme_signatures=signature_set,
                        source_context=None,
                        target_context=None,
                    )
                except Exception:
                    alignment = None
                if alignment is None:
                    continue

            if isinstance(alignment, dict):
                alignment_successful = True
                entry.update({
                    key: value
                    for key, value in alignment.items()
                    if key in {
                        'similarity',
                        'rarity',
                        'combined',
                        'rhyme_type',
                        'matched_signatures',
                        'target_signatures',
                        'features',
                        'feature_profile',
                        'prosody_profile',
                        'internal_rhyme_score',
                        'stress_alignment',
                    }
                })
                if alignment.get('combined') is not None:
                    entry['combined_score'] = alignment['combined']
                    entry['confidence'] = alignment['combined']
                if alignment.get('similarity') is not None:
                    entry['phonetic_sim'] = alignment['similarity']
                if alignment.get('rarity') is not None:
                    entry['rarity_score'] = alignment['rarity']
                if alignment.get('target_signatures'):
                    alignment_signatures = {
                        str(sig)
                        for sig in alignment['target_signatures']
                        if sig
                    }
                    if alignment_signatures:
                        candidate_signature_map.setdefault('alignment', set()).update(
                            alignment_signatures
                        )
                if alignment.get('stress_alignment') is not None:
                    entry['rhythm_score'] = alignment['stress_alignment']
            else:
                aggregated_signatures.update(fallback_signature_set)

            if isinstance(candidate, dict):
                for key in ('target_rhyme_signatures', 'signatures', 'candidate_signatures'):
                    value = candidate.get(key)
                    if not value:
                        continue
                    candidate_signature_map.setdefault('candidate', set()).update(
                        str(sig) for sig in value if sig
                    )

            if analyzer is not None and hasattr(analyzer, 'derive_rhyme_profile'):
                try:
                    analyzer_profile = analyzer.derive_rhyme_profile(
                        source_word,
                        target_word,
                        similarity if similarity else None,
                    )
                except Exception:
                    analyzer_profile = None
                if isinstance(analyzer_profile, dict):
                    analyzer_signatures = analyzer_profile.get('target_signatures')
                    if not analyzer_signatures:
                        analyzer_signatures = analyzer_profile.get('signatures')
                    if analyzer_signatures:
                        candidate_signature_map.setdefault('analyzer', set()).update(
                            str(sig) for sig in analyzer_signatures if sig
                        )
                    if entry.get('stress_alignment') is None and (
                        analyzer_profile.get('stress_alignment') is not None
                    ):
                        entry['stress_alignment'] = analyzer_profile['stress_alignment']
                    for key in ('feature_profile', 'prosody_profile'):
                        if key not in entry or not isinstance(entry.get(key), dict):
                            value = analyzer_profile.get(key)
                            if isinstance(value, dict):
                                entry[key] = value

            matched_context_sources: Set[str] = set()
            for context_name, context_signatures in available_context_sources.items():
                if not context_signatures:
                    continue
                for candidate_signatures in candidate_signature_map.values():
                    if context_signatures & candidate_signatures:
                        matched_context_sources.add(context_name)
                        break

            if 'spelling' in available_context_sources and 'spelling' in candidate_signature_map:
                matched_context_sources.add('spelling')
            if alignment_successful and 'cultural' in available_context_sources:
                matched_context_sources.add('cultural')

            if enforce_multi_source and len(matched_context_sources) < 2:
                if telemetry:
                    telemetry.increment('search.phonetic.signature_reject')
                continue

            matching_candidate_sources = {
                source_name
                for source_name, candidate_signatures in candidate_signature_map.items()
                if candidate_signatures & signature_set
            }
            if matching_candidate_sources:
                entry['matched_signature_sources'] = sorted(matching_candidate_sources)
            if matched_context_sources:
                entry['signature_context_matches'] = sorted(matched_context_sources)

            combined_candidate_signatures: Set[str] = set()
            for sig_set in candidate_signature_map.values():
                combined_candidate_signatures.update(sig_set)

            if combined_candidate_signatures:
                aggregated_signatures.update(combined_candidate_signatures)
                entry.setdefault(
                    'target_rhyme_signatures',
                    sorted(combined_candidate_signatures),
                )

            self._ensure_rhythm_score(entry)

            matches.append(entry)

        matches.sort(
            key=lambda item: (
                float(item.get('combined_score', item.get('confidence', 0.0)) or 0.0),
                float(item.get('rarity_score', 0.0) or 0.0),
            ),
            reverse=True,
        )

        max_seed = max(3, min(6, limit))
        seed_payload: List[Dict[str, Any]] = []
        for entry in matches[:max_seed]:
            seed_payload.append(
                {
                    'word': entry.get('target_word'),
                    'rarity': entry.get('rarity_score', 0.0),
                    'combined': entry.get('combined_score', entry.get('confidence', 0.0)),
                    'signatures': list(
                        entry.get('target_rhyme_signatures')
                        or entry.get('matched_signatures')
                        or self._build_spelling_signature(entry.get('target_word'))
                    ),
                    'feature_profile': entry.get('feature_profile', {}),
                    'prosody_profile': entry.get('prosody_profile', {}),
                }
            )

        return matches, seed_payload, aggregated_signatures, delivered_words

    def _collect_cultural_matches(
        self,
        source_word: str,
        *,
        analyzer: Optional[EnhancedPhoneticAnalyzer],
        cultural_engine: Optional[CulturalIntelligenceEngine],
        min_confidence: float,
        context: Dict[str, Any],
        cultural_filters: Set[str],
        genre_filters: Set[str],
        max_line_distance: Optional[int],
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Return cultural database matches with optional enrichment."""

        if not self.repository:
            return []

        try:
            source_rows, target_rows = self.repository.fetch_cultural_matches(
                source_word,
                min_confidence=min_confidence,
                phonetic_threshold=context.get('threshold'),
                cultural_filters=sorted(cultural_filters),
                genre_filters=sorted(genre_filters),
                max_line_distance=max_line_distance,
                limit=limit,
            )
        except Exception as exc:
            self._logger.error(
                'Cultural lookup failed',
                context={'error': str(exc), 'source_word': source_word},
            )
            return []

        def _row_to_entry(row: Tuple[Any, ...]) -> Dict[str, Any]:
            (
                source,
                target,
                artist,
                song,
                pattern,
                genre,
                distance,
                confidence,
                phonetic_similarity,
                cultural_sig,
                source_context,
                target_context,
            ) = row

            source_clean = str(source or '').strip()
            target_clean = str(target or '').strip()
            query_word = str(source_word).strip().lower()
            if target_clean.lower() == query_word and source_clean.lower() != query_word:
                source, target = target, source
                source_context, target_context = target_context, source_context
                source_clean, target_clean = target_clean, source_clean

            entry = {
                'source_word': source,
                'target_word': target,
                'artist': artist,
                'song': song,
                'pattern': self._compose_pattern(source, target),
                'genre': genre,
                'distance': distance,
                'confidence': confidence,
                'combined_score': confidence,
                'phonetic_sim': phonetic_similarity,
                'cultural_sig': cultural_sig,
                'source_context': source_context,
                'target_context': target_context,
                'result_source': 'cultural',
                'source_phonetic_profile': context.get('profile'),
                'source_rhyme_signatures': list(context.get('signature_set') or []),
                'phonetic_threshold': context.get('threshold'),
                'is_multi_word': bool(target_clean and ' ' in target_clean),
            }
            if analyzer is not None:
                entry['target_phonetics'] = self._describe_word(analyzer, target)
            else:
                entry['target_phonetics'] = self._describe_word(None, target)
            return entry

        entries = [_row_to_entry(row) for row in source_rows + target_rows]

        if cultural_engine:
            for entry in entries:
                try:
                    context_payload = {
                        'artist': entry.get('artist'),
                        'song': entry.get('song'),
                        'source_word': entry.get('source_word'),
                        'target_word': entry.get('target_word'),
                        'pattern': entry.get('pattern'),
                        'cultural_significance': entry.get('cultural_sig'),
                    }
                    cultural_context = cultural_engine.get_cultural_context(context_payload)
                except Exception:
                    cultural_context = None
                if cultural_context is not None:
                    if not isinstance(cultural_context, dict):
                        try:
                            cultural_context = dict(vars(cultural_context))
                        except Exception:
                            cultural_context = {'value': cultural_context}
                    entry['cultural_context'] = cultural_context
                    try:
                        rarity = cultural_engine.get_cultural_rarity_score(cultural_context)
                    except Exception:
                        rarity = None
                    if rarity is not None:
                        entry['cultural_rarity'] = rarity
                try:
                    alignment = cultural_engine.evaluate_rhyme_alignment(
                        source_word,
                        entry.get('target_word'),
                        threshold=context.get('threshold'),
                        rhyme_signatures=context.get('signature_set'),
                        source_context=entry.get('source_context'),
                        target_context=entry.get('target_context'),
                    )
                except Exception:
                    alignment = None
                if isinstance(alignment, dict):
                    if alignment.get('combined') is not None:
                        entry['combined_score'] = alignment['combined']
                        entry['confidence'] = alignment['combined']
                    if alignment.get('similarity') is not None:
                        entry['phonetic_sim'] = alignment['similarity']
                    if alignment.get('rarity') is not None:
                        entry['rarity_score'] = alignment['rarity']
                    if alignment.get('rhyme_type'):
                        entry['rhyme_type'] = alignment['rhyme_type']
                    if alignment.get('feature_profile'):
                        entry['feature_profile'] = alignment['feature_profile']
                    if alignment.get('prosody_profile'):
                        entry['prosody_profile'] = alignment['prosody_profile']
                    if alignment.get('stress_alignment') is not None:
                        entry['stress_alignment'] = alignment['stress_alignment']
                        entry['rhythm_score'] = alignment['stress_alignment']
                    if alignment.get('target_signatures'):
                        entry['target_rhyme_signatures'] = alignment['target_signatures']
                self._ensure_rhythm_score(entry)
        return entries

    def _collect_anti_llm_matches(
        self,
        source_word: str,
        *,
        analyzer: Optional[EnhancedPhoneticAnalyzer],
        anti_llm_engine: Optional[AntiLLMRhymeEngine],
        limit: int,
        min_confidence: float,
        phonetic_threshold: Optional[float],
        module1_seeds: Optional[List[Dict[str, Any]]],
        seed_signatures: Optional[Set[str]],
        delivered_words: Optional[Set[str]],
    ) -> List[Dict[str, Any]]:
        """Generate anti-LLM patterns with consistent dictionaries."""

        if anti_llm_engine is None:
            return []

        telemetry = getattr(self, 'telemetry', None)

        try:
            threshold = (
                float(phonetic_threshold)
                if phonetic_threshold is not None
                else float(min_confidence)
            )
        except (TypeError, ValueError):
            threshold = float(min_confidence)
        threshold = max(float(min_confidence), threshold)
        fallback_floor = max(float(min_confidence), threshold - 0.05)
        if not delivered_words:
            fallback_floor = float(min_confidence)

        try:
            raw_patterns = anti_llm_engine.generate_anti_llm_patterns(
                source_word,
                limit=limit,
                module1_seeds=module1_seeds or [],
                seed_signatures=seed_signatures or set(),
                delivered_words=delivered_words or set(),
            )
        except Exception as exc:
            self._logger.error(
                'Anti-LLM pattern generation failed',
                context={'error': str(exc), 'source_word': source_word},
            )
            return []

        entries: List[Dict[str, Any]] = []
        for pattern in raw_patterns:
            if isinstance(pattern, dict):
                data = dict(pattern)
            else:
                data = {}
                for attr in (
                    'source_word',
                    'target_word',
                    'confidence',
                    'combined',
                    'combined_score',
                    'rarity_score',
                    'stress_alignment',
                    'prosody_profile',
                    'feature_profile',
                    'llm_weakness_type',
                    'cultural_depth',
                    'bradley_device',
                ):
                    if hasattr(pattern, attr):
                        data[attr] = getattr(pattern, attr)
            target_word = str(data.get('target_word', '') or '').strip()
            if not target_word:
                continue
            confidence_val = data.get('combined_score')
            if confidence_val is None:
                confidence_val = data.get('combined')
            if confidence_val is None:
                confidence_val = data.get('confidence')
            try:
                confidence_val = float(confidence_val) if confidence_val is not None else 0.0
            except (TypeError, ValueError):
                confidence_val = 0.0

            gate_mode = 'strict'
            if confidence_val < threshold:
                if confidence_val >= fallback_floor:
                    gate_mode = 'fallback'
                    if telemetry:
                        telemetry.increment('search.anti_llm.threshold_fallback')
                else:
                    if telemetry:
                        telemetry.increment('search.anti_llm.threshold_blocked')
                    continue
            elif telemetry:
                telemetry.increment('search.anti_llm.threshold_pass')

            entry = {
                'source_word': data.get('source_word', source_word),
                'target_word': target_word,
                'pattern': self._compose_pattern(source_word, target_word),
                'confidence': confidence_val,
                'combined_score': confidence_val,
                'rarity_score': data.get('rarity_score'),
                'stress_alignment': data.get('stress_alignment'),
                'prosody_profile': data.get('prosody_profile'),
                'feature_profile': data.get('feature_profile'),
                'llm_weakness_type': data.get('llm_weakness_type'),
                'cultural_depth': data.get('cultural_depth'),
                'bradley_device': data.get('bradley_device'),
                'result_source': 'anti_llm',
                'is_multi_word': bool(' ' in target_word),
                'phonetic_threshold': threshold,
                'threshold_gate': gate_mode,
            }
            if analyzer is not None:
                entry['target_phonetics'] = self._describe_word(analyzer, target_word)
            else:
                entry['target_phonetics'] = self._describe_word(None, target_word)
            self._ensure_rhythm_score(entry)
            entries.append(entry)
        return entries

    def _normalise_filters(
        self,
        cultural_significance: Optional[List[str] | str],
        genres: Optional[List[str] | str],
        allowed_rhyme_types: Optional[List[str] | str],
        bradley_devices: Optional[List[str] | str],
    ) -> Tuple[Set[str], Set[str], Set[str], Set[str]]:
        def _to_set(value: Optional[List[str] | str]) -> Set[str]:
            if value is None:
                return set()
            if isinstance(value, str):
                items = [value]
            else:
                items = list(value)
            cleaned = set()
            for item in items:
                if item is None:
                    continue
                label = self.normalize_filter_label(str(item))
                if label:
                    cleaned.add(label)
            return cleaned

        return (
            _to_set(cultural_significance),
            _to_set(genres),
            _to_set(allowed_rhyme_types),
            _to_set(bradley_devices),
        )

    def _confidence_value(self, entry: Dict[str, Any]) -> float:
        for key in ('combined_score', 'combined', 'confidence'):
            value = entry.get(key)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return 0.0

    def _rarity_value(self, entry: Dict[str, Any]) -> float:
        for key in ('rarity_score', 'cultural_rarity'):
            value = entry.get(key)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return 0.0

    def _stress_value(self, entry: Dict[str, Any]) -> Optional[float]:
        candidates = [
            entry.get('stress_alignment'),
        ]
        prosody = entry.get('prosody_profile')
        if isinstance(prosody, dict):
            candidates.append(prosody.get('stress_alignment'))
        feature = entry.get('feature_profile')
        if isinstance(feature, dict):
            candidates.append(feature.get('stress_alignment'))
        for value in candidates:
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return None

    def _ensure_rhythm_score(self, entry: Dict[str, Any]) -> None:
        stress_alignment = entry.get('stress_alignment')
        if stress_alignment is None:
            for key in ('feature_profile', 'prosody_profile', 'features'):
                profile = entry.get(key)
                if isinstance(profile, dict):
                    candidate = profile.get('stress_alignment')
                    if candidate is not None:
                        stress_alignment = candidate
                        break
        if stress_alignment is not None:
            entry.setdefault('stress_alignment', stress_alignment)
            if entry.get('rhythm_score') is None:
                entry['rhythm_score'] = stress_alignment

    def _cadence_tag(self, entry: Dict[str, Any]) -> Optional[str]:
        prosody = entry.get('prosody_profile')
        if isinstance(prosody, dict):
            tag = prosody.get('complexity_tag')
            if tag:
                return self.normalize_filter_label(str(tag))
        feature = entry.get('feature_profile')
        if isinstance(feature, dict):
            tag = feature.get('complexity_tag') or feature.get('cadence_tag')
            if tag:
                return self.normalize_filter_label(str(tag))
        return None

    def _entry_rhyme_type(self, entry: Dict[str, Any]) -> Optional[str]:
        candidate = entry.get('rhyme_type')
        if not candidate and isinstance(entry.get('feature_profile'), dict):
            candidate = entry['feature_profile'].get('rhyme_type')
        if candidate:
            return self.normalize_filter_label(str(candidate))
        return None

    def _syllable_count(self, entry: Dict[str, Any]) -> Optional[int]:
        syllable_span = entry.get('syllable_span')
        if isinstance(syllable_span, (list, tuple)) and len(syllable_span) >= 2:
            try:
                return int(syllable_span[1])
            except (TypeError, ValueError):
                return None
        phonetics = entry.get('target_phonetics')
        if isinstance(phonetics, dict) and phonetics.get('syllables') is not None:
            try:
                return int(phonetics['syllables'])
            except (TypeError, ValueError):
                return None
        return None

    def _apply_filters(
        self,
        entries: List[Dict[str, Any]],
        *,
        limit: int,
        min_confidence: float,
        min_rarity: Optional[float],
        min_stress_alignment: Optional[float],
        cadence_focus: Optional[str],
        allowed_rhyme_types: Set[str],
        bradley_devices: Set[str],
        require_internal: bool,
        min_syllables: Optional[int],
        max_syllables: Optional[int],
        max_line_distance: Optional[int],
        cultural_filters: Set[str],
        genre_filters: Set[str],
        scope: str,
    ) -> List[Dict[str, Any]]:
        cadence_focus_label = self.normalize_filter_label(cadence_focus) if cadence_focus else ''
        telemetry = getattr(self, 'telemetry', None)
        filtered: List[Dict[str, Any]] = []
        seen: Set[Tuple[str, str]] = set()

        reasons: Dict[str, int] = {}

        active_filters: List[str] = []
        if min_confidence > 0.0:
            active_filters.append('min_confidence')
        if min_rarity is not None:
            active_filters.append('min_rarity')
        if min_stress_alignment is not None:
            active_filters.append('min_stress_alignment')
        if cadence_focus_label:
            active_filters.append('cadence_focus')
        if allowed_rhyme_types:
            active_filters.append('allowed_rhyme_types')
        if bradley_devices:
            active_filters.append('bradley_devices')
        if require_internal:
            active_filters.append('require_internal')
        if min_syllables is not None:
            active_filters.append('min_syllables')
        if max_syllables is not None:
            active_filters.append('max_syllables')
        if max_line_distance is not None:
            active_filters.append('max_line_distance')
        if cultural_filters:
            active_filters.append('cultural_filters')
        if genre_filters:
            active_filters.append('genre_filters')

        def _reject(reason: str) -> None:
            reasons[reason] = reasons.get(reason, 0) + 1

        for entry in entries:
            if self._confidence_value(entry) < min_confidence:
                _reject('min_confidence')
                continue

            if cultural_filters:
                label = self.normalize_filter_label(entry.get('cultural_sig'))
                if label not in cultural_filters:
                    _reject('cultural_filters')
                    continue

            if genre_filters:
                genre_label = self.normalize_filter_label(entry.get('genre'))
                if genre_label not in genre_filters:
                    _reject('genre_filters')
                    continue

            if max_line_distance is not None:
                distance = entry.get('line_distance')
                if distance is None and isinstance(entry.get('feature_profile'), dict):
                    distance = entry['feature_profile'].get('line_distance')
                try:
                    distance = int(distance) if distance is not None else None
                except (TypeError, ValueError):
                    distance = None
                if distance is None or distance > max_line_distance:
                    _reject('max_line_distance')
                    continue

            if min_rarity is not None and self._rarity_value(entry) < min_rarity:
                _reject('min_rarity')
                continue

            stress_value = self._stress_value(entry)
            if min_stress_alignment is not None and (
                stress_value is None or stress_value < min_stress_alignment
            ):
                _reject('min_stress_alignment')
                continue

            if cadence_focus_label:
                cadence_tag = self._cadence_tag(entry)
                if cadence_tag != cadence_focus_label:
                    _reject('cadence_focus')
                    continue

            if allowed_rhyme_types:
                rhyme_type = self._entry_rhyme_type(entry)
                if rhyme_type not in allowed_rhyme_types:
                    _reject('allowed_rhyme_types')
                    continue

            if bradley_devices:
                device = entry.get('bradley_device')
                if device is None and isinstance(entry.get('feature_profile'), dict):
                    device = entry['feature_profile'].get('bradley_device')
                device_label = self.normalize_filter_label(device) if device else ''
                if device_label not in bradley_devices:
                    _reject('bradley_devices')
                    continue

            if require_internal:
                internal_score = entry.get('internal_rhyme_score')
                if internal_score is None and isinstance(entry.get('feature_profile'), dict):
                    internal_score = entry['feature_profile'].get('internal_rhyme_score')
                try:
                    internal_value = float(internal_score) if internal_score is not None else 0.0
                except (TypeError, ValueError):
                    internal_value = 0.0
                if internal_value < 0.4:
                    _reject('require_internal')
                    continue

            syllable_count = self._syllable_count(entry)
            if min_syllables is not None and (
                syllable_count is None or syllable_count < min_syllables
            ):
                _reject('min_syllables')
                continue
            if max_syllables is not None and (
                syllable_count is None or syllable_count > max_syllables
            ):
                _reject('max_syllables')
                continue

            key = (str(entry.get('target_word')).lower(), str(entry.get('result_source')))
            if key in seen:
                _reject('duplicate')
                continue
            seen.add(key)
            filtered.append(entry)

        filtered.sort(
            key=lambda item: (
                self._confidence_value(item),
                self._rarity_value(item),
            ),
            reverse=True,
        )

        kept_before_limit = len(filtered)
        if limit <= 0:
            result: List[Dict[str, Any]] = []
        else:
            result = filtered[:limit]

        stats = {
            'scope': scope,
            'input': len(entries),
            'kept': kept_before_limit,
            'output': len(result),
            'limited': kept_before_limit > len(result),
            'rejected': sum(reasons.values()),
            'reasons': reasons,
            'active_filters': sorted(active_filters),
            'parameters': {
                'min_confidence': min_confidence,
                'min_rarity': min_rarity,
                'min_stress_alignment': min_stress_alignment,
                'cadence_focus': cadence_focus_label or None,
                'allowed_rhyme_types': sorted(allowed_rhyme_types),
                'bradley_devices': sorted(bradley_devices),
                'require_internal': require_internal,
                'min_syllables': min_syllables,
                'max_syllables': max_syllables,
                'max_line_distance': max_line_distance,
                'cultural_filters': sorted(cultural_filters),
                'genre_filters': sorted(genre_filters),
            },
        }

        if telemetry:
            telemetry.annotate(f'filters.{scope}', stats)

        return result
    def _coerce_int(self, value: Any) -> Optional[int]:
        """Best-effort conversion of ``value`` to an integer."""

        if value is None:
            return None
        if isinstance(value, bool):
            return int(value)
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if number != number:  # NaN guard
            return None
        try:
            return int(round(number))
        except (TypeError, ValueError, OverflowError):
            return None

    def _result_identity(self, entry: Dict[str, Any]) -> str:
        """Return a stable identifier for a rhyme entry."""

        target = str(entry.get('target_word') or '').strip().lower()
        pattern = str(entry.get('pattern') or '').strip().lower()
        if target:
            return target
        if pattern:
            return pattern
        return str(id(entry))

    def _is_multi_word_entry(self, entry: Dict[str, Any]) -> bool:
        """Determine whether ``entry`` represents a multi-word rhyme."""

        if 'is_multi_word' in entry:
            return bool(entry['is_multi_word'])
        target = str(entry.get('target_word') or '')
        return bool(target and ' ' in target.strip())

    def _dedupe_and_truncate(
        self,
        entries: List[Dict[str, Any]],
        limit: int,
        *,
        seen: Optional[Set[str]] = None,
    ) -> Tuple[List[Dict[str, Any]], Set[str]]:
        """Return ``entries`` capped at ``limit`` while removing duplicates."""

        if seen is None:
            seen = set()
        result: List[Dict[str, Any]] = []
        if limit <= 0:
            return result, set(seen)

        for entry in entries:
            key = self._result_identity(entry)
            if not key or key in seen:
                continue
            seen.add(key)
            result.append(entry)
            if len(result) >= limit:
                break
        return result, set(seen)

    def _merge_uncommon_sources(
        self,
        phonetic: List[Dict[str, Any]],
        anti_llm: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Combine phonetic and anti-LLM matches preserving rarity focus."""

        merged: List[Dict[str, Any]] = []
        seen: Set[str] = set()
        for entry in list(anti_llm) + list(phonetic):
            key = self._result_identity(entry)
            if not key or key in seen:
                continue
            seen.add(key)
            merged.append(entry)
        return merged

    def _source_profile_shape(
        self, profile: Dict[str, Any]
    ) -> Tuple[Optional[int], bool]:
        """Return (syllable_count, is_multi_word) for the source request."""

        syllable_count: Optional[int] = None
        is_multi_word = False

        if isinstance(profile, dict):
            phonetics = profile.get('phonetics')
            if isinstance(phonetics, dict):
                syllable_count = self._coerce_int(phonetics.get('syllables'))
            if syllable_count is None:
                syllable_count = self._coerce_int(profile.get('syllables'))

            token_syllables = profile.get('token_syllables')
            if syllable_count is None and isinstance(token_syllables, (list, tuple)):
                total = 0
                valid = False
                for raw in token_syllables:
                    value = self._coerce_int(raw)
                    if value is None:
                        valid = False
                        break
                    total += value
                    valid = True
                if valid and total > 0:
                    syllable_count = total

            tokens = profile.get('phrase_tokens') or profile.get('tokens') or []
            is_multi_word = bool(profile.get('is_multi_word'))
            if tokens and len(tokens) > 1:
                is_multi_word = True

        return syllable_count, is_multi_word

    def _finalize_results(
        self,
        context: Dict[str, Any],
        *,
        phonetic: List[Dict[str, Any]],
        cultural: List[Dict[str, Any]],
        anti_llm: List[Dict[str, Any]],
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 20,
    ) -> Dict[str, Any]:
        profile = context.get('profile', {})
        if isinstance(profile, dict):
            profile.setdefault('signatures', sorted(context.get('signature_set', [])))
            profile.setdefault('threshold', context.get('threshold'))
            profile.setdefault('reference_similarity', context.get('reference_similarity'))
            provenance = context.get('signature_provenance') or {}
            if provenance and 'signature_provenance' not in profile:
                profile['signature_provenance'] = {
                    key: sorted({str(sig) for sig in value})
                    for key, value in provenance.items()
                    if value
                }
            profile.setdefault('signature_set', sorted(context.get('signature_set') or []))

        syllable_count, is_multi_word = self._source_profile_shape(profile)
        single_syllable = bool(not is_multi_word and syllable_count == 1)

        uncommon_candidates = self._merge_uncommon_sources(phonetic, anti_llm)

        try:
            limit_value = int(limit)
        except (TypeError, ValueError):
            limit_value = 0
        if limit_value <= 0:
            limit_value = 1

        rap_cap = min(limit_value, 10)
        rap_results, seen_keys = self._dedupe_and_truncate(cultural, rap_cap)

        multi_results: List[Dict[str, Any]] = []

        if single_syllable:
            uncommon_cap = min(limit_value, 20)
            single_candidates = [
                entry
                for entry in uncommon_candidates
                if self._result_identity(entry) not in seen_keys
            ]
            uncommon_results, seen_keys = self._dedupe_and_truncate(
                single_candidates,
                uncommon_cap,
                seen=seen_keys,
            )
        else:
            multi_cap = min(limit_value, 10)
            multi_candidates = [
                entry
                for entry in uncommon_candidates
                if self._is_multi_word_entry(entry)
            ]
            multi_results, seen_keys = self._dedupe_and_truncate(
                multi_candidates,
                multi_cap,
                seen=seen_keys,
            )

            uncommon_cap = min(limit_value, 10)
            single_candidates = [
                entry
                for entry in uncommon_candidates
                if not self._is_multi_word_entry(entry)
                and self._result_identity(entry) not in seen_keys
            ]
            uncommon_results, seen_keys = self._dedupe_and_truncate(
                single_candidates,
                uncommon_cap,
                seen=seen_keys,
            )

        return {
            'source_profile': profile,
            'uncommon': uncommon_results,
            'multi_word': multi_results,
            'rap_db': rap_results,
            'filters': filters or {},
        }

    def _search_rhymes_internal(
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
        """Search for rhymes without external coordination plumbing."""

        if not source_word or not str(source_word).strip():
            return {'uncommon': [], 'multi_word': [], 'rap_db': []}

        try:
            limit_value = int(limit)
        except (TypeError, ValueError):
            limit_value = 0
        if limit_value <= 0:
            return {'uncommon': [], 'multi_word': [], 'rap_db': []}
        limit = limit_value

        try:
            min_confidence = float(min_confidence)
        except (TypeError, ValueError):
            min_confidence = 0.0

        cultural_filters, genre_filters, rhyme_filters, bradley_filters = self._normalise_filters(
            cultural_significance,
            genres,
            allowed_rhyme_types,
            bradley_devices,
        )

        cadence_focus_label = self.normalize_filter_label(cadence_focus) if cadence_focus else ''

        try:
            min_syllable_threshold = max(1, int(min_syllables)) if min_syllables else None
        except (TypeError, ValueError):
            min_syllable_threshold = None
        try:
            max_syllable_threshold = max(1, int(max_syllables)) if max_syllables else None
        except (TypeError, ValueError):
            max_syllable_threshold = None
        try:
            min_rarity_threshold = float(min_rarity) if min_rarity is not None else None
        except (TypeError, ValueError):
            min_rarity_threshold = None
        try:
            min_stress_threshold = (
                float(min_stress_alignment) if min_stress_alignment is not None else None
            )
        except (TypeError, ValueError):
            min_stress_threshold = None

        selected_sources = self._normalise_filters(result_sources, None, None, None)[0]
        if not selected_sources:
            selected_sources = {'phonetic', 'cultural', 'anti-llm'}

        analyzer = getattr(self, 'phonetic_analyzer', None)
        cultural_engine = getattr(self, 'cultural_engine', None)
        anti_llm_engine = getattr(self, 'anti_llm_engine', None)
        include_phonetic = 'phonetic' in selected_sources
        include_cultural = 'cultural' in selected_sources
        include_anti = 'anti-llm' in selected_sources or 'anti_llm' in selected_sources
        cmu_loader = getattr(analyzer, 'cmu_loader', None) or getattr(self, 'cmu_loader', None)

        # When callers only request phonetic results we can skip the expensive
        # cultural context derivation and alignment checks.  Those steps invoke
        # the cultural intelligence engine which performs several database
        # queries per request; avoiding that work keeps the fast path light.
        cultural_for_context = (
            cultural_engine if (include_cultural or include_anti) else None
        )

        context = self._build_source_context(
            source_word, analyzer, cmu_loader, cultural_for_context
        )

        telemetry = getattr(self, 'telemetry', None)

        phonetic_matches: List[Dict[str, Any]] = []
        module1_seeds: List[Dict[str, Any]] = []
        seed_signatures: Set[str] = set(context.get('signature_set') or [])
        delivered_words: Set[str] = set()

        filter_summary: Dict[str, Any] = {
            'min_confidence': min_confidence,
            'min_rarity': min_rarity_threshold,
            'min_stress_alignment': min_stress_threshold,
            'cadence_focus': cadence_focus_label or None,
            'allowed_rhyme_types': sorted(rhyme_filters),
            'bradley_devices': sorted(bradley_filters),
            'cultural_filters': sorted(cultural_filters),
            'genre_filters': sorted(genre_filters),
            'require_internal': require_internal,
            'min_syllables': min_syllable_threshold,
            'max_syllables': max_syllable_threshold,
            'max_line_distance': max_line_distance,
        }

        if include_phonetic:
            timer = telemetry.timer('search.branch.phonetic') if telemetry else nullcontext()
            with timer as payload:
                (
                    phonetic_matches,
                    module1_seeds,
                    new_signatures,
                    delivered_words,
                ) = self._collect_phonetic_matches(
                    source_word,
                    max(limit, 1),
                    analyzer=analyzer,
                    cultural_engine=(
                        cultural_engine if (include_cultural or include_anti) else None
                    ),
                    cmu_loader=cmu_loader,
                    context=context,
                    min_confidence=min_confidence,
                )
                seed_signatures.update(new_signatures)
                if payload is not None:
                    payload['result_count'] = len(phonetic_matches)
        else:
            if telemetry:
                telemetry.increment('search.branch.phonetic.skipped')

        cultural_matches: List[Dict[str, Any]] = []
        if include_cultural:
            timer = telemetry.timer('search.branch.cultural') if telemetry else nullcontext()
            with timer as payload:
                if cultural_engine is not None:
                    cultural_matches = self._collect_cultural_matches(
                        source_word,
                        analyzer=analyzer,
                        cultural_engine=cultural_engine,
                        min_confidence=min_confidence,
                        context=context,
                        cultural_filters=cultural_filters,
                        genre_filters=genre_filters,
                        max_line_distance=max_line_distance,
                        limit=max(limit, 1),
                    )
                else:
                    cultural_matches = []
                if payload is not None:
                    payload['result_count'] = len(cultural_matches)
        elif telemetry:
            telemetry.increment('search.branch.cultural.skipped')

        anti_llm_matches: List[Dict[str, Any]] = []
        if include_anti and anti_llm_engine is not None:
            timer = telemetry.timer('search.branch.anti_llm') if telemetry else nullcontext()
            with timer as payload:
                anti_llm_matches = self._collect_anti_llm_matches(
                    source_word,
                    analyzer=analyzer,
                    anti_llm_engine=anti_llm_engine,
                    limit=max(limit, 1),
                    min_confidence=min_confidence,
                    phonetic_threshold=context.get('threshold'),
                    module1_seeds=module1_seeds,
                    seed_signatures=seed_signatures,
                    delivered_words=delivered_words,
                )
                if payload is not None:
                    payload['result_count'] = len(anti_llm_matches)
        elif telemetry:
            telemetry.increment('search.branch.anti_llm.skipped')

        phonetic_filtered = self._apply_filters(
            phonetic_matches,
            limit=limit,
            min_confidence=min_confidence,
            min_rarity=min_rarity_threshold,
            min_stress_alignment=min_stress_threshold,
            cadence_focus=cadence_focus_label,
            allowed_rhyme_types=rhyme_filters,
            bradley_devices=bradley_filters,
            require_internal=require_internal,
            min_syllables=min_syllable_threshold,
            max_syllables=max_syllable_threshold,
            max_line_distance=max_line_distance,
            cultural_filters=set(),
            genre_filters=set(),
            scope='phonetic',
        )

        cultural_filtered = self._apply_filters(
            cultural_matches,
            limit=limit,
            min_confidence=min_confidence,
            min_rarity=min_rarity_threshold,
            min_stress_alignment=min_stress_threshold,
            cadence_focus=cadence_focus_label,
            allowed_rhyme_types=rhyme_filters,
            bradley_devices=bradley_filters,
            require_internal=require_internal,
            min_syllables=min_syllable_threshold,
            max_syllables=max_syllable_threshold,
            max_line_distance=max_line_distance,
            cultural_filters=cultural_filters,
            genre_filters=genre_filters,
            scope='cultural',
        )

        anti_llm_filtered = self._apply_filters(
            anti_llm_matches,
            limit=limit,
            min_confidence=min_confidence,
            min_rarity=min_rarity_threshold,
            min_stress_alignment=min_stress_threshold,
            cadence_focus=cadence_focus_label,
            allowed_rhyme_types=rhyme_filters,
            bradley_devices=bradley_filters,
            require_internal=require_internal,
            min_syllables=min_syllable_threshold,
            max_syllables=max_syllable_threshold,
            max_line_distance=max_line_distance,
            cultural_filters=set(),
            genre_filters=set(),
            scope='anti_llm',
        )

        filter_summary['phonetic_threshold'] = context.get('threshold')
        signature_provenance = context.get('signature_provenance') or {}
        if signature_provenance:
            filter_summary['signature_sources'] = {
                key: sorted({str(sig) for sig in value})
                for key, value in signature_provenance.items()
                if value
            }
        filter_summary['signature_set'] = sorted(context.get('signature_set') or [])

        return self._finalize_results(
            context,
            phonetic=phonetic_filtered,
            cultural=cultural_filtered,
            anti_llm=anti_llm_filtered,
            filters=filter_summary,
            limit=limit,
        )
class RhymeResultFormatter:
    """Format rhyme search results into a rich markdown summary."""

    def format_rhyme_results(self, source_word: str, rhymes: Dict[str, Any]) -> str:
        """Render grouped rhyme results with shared phonetic context."""

        section_headers: Dict[str, str] = {
            "uncommon": "🧠 Uncommon Rhymes",
            "multi_word": "🧩 Multi-Word Rhymes",
            "rap_db": "🎤 Rap Database Patterns",
        }

        if not rhymes:
            return f"❌ No rhymes found for '{escape(str(source_word))}'. Try another word or adjust your filters."

        has_results = any(rhymes.get(key) for key in section_headers)
        if not has_results:
            return f"❌ No rhymes found for '{escape(str(source_word))}'. Try another word or adjust your filters."

        def _format_phonetics(phonetics: Dict[str, Any]) -> Optional[str]:
            parts: List[str] = []
            syllables = phonetics.get("syllables")
            if isinstance(syllables, (int, float)):
                parts.append(f"Syllables: {int(syllables)}")
            stress = phonetics.get("stress_pattern_display") or phonetics.get("stress_pattern")
            if stress:
                parts.append(f"Stress: {stress}")
            meter = phonetics.get("meter_hint") or phonetics.get("metrical_foot")
            if meter:
                parts.append(f"Meter: {meter}")
            return " | ".join(parts) if parts else None

        def _format_entry(entry: Dict[str, Any]) -> List[str]:
            details: List[str] = []
            rarity = entry.get("rarity_score") or entry.get("cultural_rarity")
            confidence = entry.get("combined_score", entry.get("confidence"))
            phonetics = entry.get("target_phonetics") or {}
            phonetics_line = _format_phonetics(phonetics)

            if rarity is not None:
                try:
                    details.append(f"• Rarity: {float(rarity):.2f}")
                except (TypeError, ValueError):
                    pass
            if confidence is not None:
                try:
                    details.append(f"• Confidence: {float(confidence):.2f}")
                except (TypeError, ValueError):
                    pass
            if phonetics_line:
                details.append(f"• Phonetics: {phonetics_line}")

            rhyme_type = entry.get("rhyme_type")
            feature_profile = entry.get("feature_profile")
            if not rhyme_type and isinstance(feature_profile, dict):
                rhyme_type = feature_profile.get("rhyme_type")
            if rhyme_type:
                details.append(f"• Rhyme type: {str(rhyme_type).replace('_', ' ').title()}")

            stress_alignment = entry.get("stress_alignment")
            if stress_alignment is None and isinstance(feature_profile, dict):
                stress_alignment = feature_profile.get("stress_alignment")
            rhythm_score = entry.get("rhythm_score")
            if rhythm_score is not None and rhythm_score != stress_alignment:
                try:
                    details.append(f"• Rhythm score: {float(rhythm_score):.2f}")
                except (TypeError, ValueError):
                    pass

            context_matches = entry.get("signature_context_matches")
            if context_matches:
                context_text = ", ".join(
                    sorted(str(src).replace('_', ' ').title() for src in context_matches)
                )
                details.append(f"• Context signatures: {context_text}")

            return details

        source_profile = rhymes.get("source_profile") or {}
        phonetics = source_profile.get("phonetics") or {}
        source_meta = _format_phonetics(phonetics)
        filters = rhymes.get("filters") or {}
        diagnostics: List[str] = []
        cadence_focus = filters.get("cadence_focus")
        if cadence_focus:
            diagnostics.append(
                f"Cadence focus: {str(cadence_focus).replace('_', ' ').title()}"
            )
        stress_target = filters.get("min_stress_alignment")
        if stress_target is not None:
            try:
                diagnostics.append(f"Min stress alignment: {float(stress_target):.2f}")
            except (TypeError, ValueError):
                pass
        phonetic_threshold = filters.get("phonetic_threshold")
        if phonetic_threshold is None:
            phonetic_threshold = source_profile.get("threshold")
        if phonetic_threshold is not None:
            try:
                diagnostics.append(f"Phonetic threshold: {float(phonetic_threshold):.2f}")
            except (TypeError, ValueError):
                pass
        signature_sources = filters.get("signature_sources")
        if not signature_sources and isinstance(source_profile.get("signature_provenance"), dict):
            signature_sources = source_profile.get("signature_provenance")
        if isinstance(signature_sources, dict) and signature_sources:
            parts: List[str] = []
            for key, value in signature_sources.items():
                count = len(value) if isinstance(value, (list, tuple, set)) else 0
                label = str(key).replace('_', ' ')
                parts.append(f"{label} ({count})")
            diagnostics.append("Signature sources: " + ", ".join(sorted(parts)))
        signature_set = filters.get("signature_set")
        if isinstance(signature_set, (list, tuple, set)) and signature_set:
            diagnostics.append(f"Signature variants tracked: {len(signature_set)}")
        summary_lines: List[str] = ["<div class='rr-source-summary'>"]
        summary_lines.append(
            f"<h3>Rhymes for {escape(str(source_word).upper())}</h3>"
        )
        source_label = str(source_profile.get("word", source_word))
        summary_lines.append(f"<p>Source: {escape(source_label)}</p>")

        meta_items: List[str] = []
        if source_meta:
            meta_items.append(source_meta)
        if meta_items:
            summary_lines.append("<ul>")
            for item in meta_items:
                summary_lines.append(f"<li>{escape(item)}</li>")
            summary_lines.append("</ul>")

        if diagnostics:
            summary_lines.append("<p><strong>Diagnostics:</strong></p>")
            summary_lines.append("<ul>")
            for diag in diagnostics:
                summary_lines.append(f"<li>{escape(diag)}</li>")
            summary_lines.append("</ul>")
        summary_lines.append("</div>")

        def _collect_details(entry: Dict[str, Any], key: str) -> List[str]:
            details = list(_format_entry(entry))
            if key == "rap_db":
                artist = entry.get("artist")
                song = entry.get("song")
                if artist or song:
                    details.append(
                        f"• Source: {artist or 'Unknown'} — {song or 'Unknown'}"
                    )
                source_context_raw = entry.get("source_context")
                target_context_raw = entry.get("target_context")
                source_context = (
                    str(source_context_raw).strip() if source_context_raw else ""
                )
                target_context = (
                    str(target_context_raw).strip() if target_context_raw else ""
                )
                lyric_segments: List[str] = []
                if source_context:
                    lyric_segments.append(f"source: {source_context}")
                if target_context and target_context.lower() != source_context.lower():
                    lyric_segments.append(f"target: {target_context}")
                if lyric_segments:
                    details.append("• Lyrics: " + " | ".join(lyric_segments))
            return details

        def _render_section(key: str, *, span_full: bool = False) -> str:
            entries = rhymes.get(key) or []
            classes = ["rr-result-card"]
            if span_full:
                classes.append("rr-span-2")
            card: List[str] = [
                f"<div class='{' '.join(classes)}'>",
                f"<h4>{escape(section_headers[key])}</h4>",
            ]
            if not entries:
                card.append(
                    "<p class='rr-empty'>No matches found. Try expanding the filters.</p>"
                )
            else:
                card.append("<ul class='rr-rhyme-list'>")
                for entry in entries:
                    target = str(entry.get("target_word") or "?")
                    pattern = entry.get("pattern")
                    card.append("<li class='rr-rhyme-entry'>")
                    card.append(
                        f"<div class='rr-rhyme-term'>{escape(target.upper())}</div>"
                    )
                    if pattern:
                        card.append(
                            f"<div class='rr-rhyme-pattern'>{escape(str(pattern))}</div>"
                        )
                    details = _collect_details(entry, key)
                    if details:
                        card.append("<ul class='rr-rhyme-details'>")
                        for detail in details:
                            card.append(f"<li>{escape(detail)}</li>")
                        card.append("</ul>")
                    card.append("</li>")
                card.append("</ul>")
            card.append("</div>")
            return "".join(card)

        grid: List[str] = ["<div class='rr-results-grid'>"]
        first_row: List[str] = ["<div class='rr-result-row'>"]
        for key in ("uncommon", "multi_word"):
            first_row.append(_render_section(key))
        first_row.append("</div>")
        grid.extend(first_row)
        second_row: List[str] = ["<div class='rr-result-row'>"]
        second_row.append(_render_section("rap_db", span_full=True))
        second_row.append("</div>")
        grid.extend(second_row)
        grid.append("</div>")

        return "\n".join(summary_lines + grid)


class SearchService:
    """Thin facade that delegates to the orchestrator and formatter."""

    @property
    def cmu_loader(self) -> Optional[object]:
        return getattr(self.orchestrator, "cmu_loader", None)

    @cmu_loader.setter
    def cmu_loader(self, loader: Optional[object]) -> None:
        if hasattr(self.orchestrator, "cmu_loader"):
            self.orchestrator.cmu_loader = loader

    def __init__(
        self,
        *,
        repository: SQLiteRhymeRepository,
        phonetic_analyzer: Optional[EnhancedPhoneticAnalyzer] = None,
        cultural_engine: Optional[CulturalIntelligenceEngine] = None,
        anti_llm_engine: Optional[AntiLLMRhymeEngine] = None,
        cmu_loader: Optional[object] = None,
        cmu_repository: Optional[CmuRhymeRepository] = None,
        max_concurrent_searches: Optional[int] = None,
        search_timeout: Optional[float] = None,
        telemetry: Optional[StructuredTelemetry] = None,
        orchestrator: Optional[RhymeQueryOrchestrator] = None,
        formatter: Optional[RhymeResultFormatter] = None,
    ) -> None:
        telemetry_obj = telemetry or StructuredTelemetry()
        cmu_repo = cmu_repository or DefaultCmuRhymeRepository()

        if orchestrator is None:
            orchestrator = RhymeQueryOrchestrator(
                repository=repository,
                phonetic_analyzer=phonetic_analyzer,
                cultural_engine=cultural_engine,
                anti_llm_engine=anti_llm_engine,
                cmu_loader=cmu_loader,
                cmu_repository=cmu_repo,
                max_concurrent_searches=max_concurrent_searches,
                search_timeout=search_timeout,
                telemetry=telemetry_obj,
            )
        else:
            orchestrator.set_phonetic_analyzer(phonetic_analyzer)
            orchestrator.set_cultural_engine(cultural_engine)
            orchestrator.set_anti_llm_engine(anti_llm_engine)
            orchestrator.set_cmu_repository(cmu_repo)
            if cmu_loader is not None:
                orchestrator.cmu_loader = cmu_loader
            orchestrator.telemetry = telemetry_obj

        self.orchestrator = orchestrator
        self.formatter = formatter or RhymeResultFormatter()
        self.repository = repository
        self.telemetry = self.orchestrator.telemetry
        self.phonetic_analyzer = phonetic_analyzer
        self.cultural_engine = cultural_engine
        self.anti_llm_engine = anti_llm_engine
        self.cmu_loader = getattr(self.orchestrator, "cmu_loader", cmu_loader)
        self.cmu_repository = cmu_repo

    def set_phonetic_analyzer(self, analyzer: Optional[EnhancedPhoneticAnalyzer]) -> None:
        self.phonetic_analyzer = analyzer
        self.orchestrator.set_phonetic_analyzer(analyzer)
        self.cmu_loader = getattr(self.orchestrator, "cmu_loader", self.cmu_loader)

    def set_cultural_engine(self, engine: Optional[CulturalIntelligenceEngine]) -> None:
        self.cultural_engine = engine
        self.orchestrator.set_cultural_engine(engine)

    def set_anti_llm_engine(self, engine: Optional[AntiLLMRhymeEngine]) -> None:
        self.anti_llm_engine = engine
        self.orchestrator.set_anti_llm_engine(engine)

    def set_cmu_repository(self, repository: Optional[CmuRhymeRepository]) -> None:
        self.cmu_repository = repository or DefaultCmuRhymeRepository()
        self.orchestrator.set_cmu_repository(repository)

    def clear_cached_results(self) -> None:
        self.orchestrator.clear_cached_results()

    def get_latest_telemetry(self) -> Dict[str, Any]:
        return self.orchestrator.get_latest_telemetry()

    def normalize_filter_label(self, name: Optional[str]) -> str:
        return self.orchestrator.normalize_filter_label(name)

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
        return self.orchestrator.search_rhymes(
            source_word,
            limit=limit,
            min_confidence=min_confidence,
            cultural_significance=cultural_significance,
            genres=genres,
            result_sources=result_sources,
            max_line_distance=max_line_distance,
            min_syllables=min_syllables,
            max_syllables=max_syllables,
            allowed_rhyme_types=allowed_rhyme_types,
            bradley_devices=bradley_devices,
            require_internal=require_internal,
            min_rarity=min_rarity,
            min_stress_alignment=min_stress_alignment,
            cadence_focus=cadence_focus,
        )

    def format_rhyme_results(self, source_word: str, rhymes: Dict[str, Any]) -> str:
        return self.formatter.format_rhyme_results(source_word, rhymes)

