from __future__ import annotations

import types
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence

import pytest

from rhyme_rarity.utils.telemetry import StructuredTelemetry
from rhyme_rarity.app.services.search_service import RhymeResultFormatter, SearchService


class FakeClock:
    """Deterministic clock used to drive telemetry timers in tests."""

    def __init__(self, step: float = 0.01) -> None:
        self._current = 0.0
        self._step = step

    def __call__(self) -> float:
        value = self._current
        self._current += self._step
        return value


@dataclass
class DummyPattern:
    target_word: str
    confidence: float
    rarity_score: float
    llm_weakness_type: str = "rare"
    cultural_depth: str = ""


class DummyRepository:
    def fetch_related_words(self, terms: Iterable[str]) -> set[str]:
        return {"echo"}

    def fetch_cultural_matches(
        self,
        source_word: str,
        *,
        min_confidence: float,
        phonetic_threshold: float | None,
        cultural_filters: Sequence[str],
        genre_filters: Sequence[str],
        max_line_distance: int | None,
        limit: int | None = None,
    ) -> tuple[List[tuple], List[tuple]]:
        row = (
            source_word,
            "echo",
            "Artist",
            "Song",
            "Pattern",
            "genre",
            1,
            0.9,
            0.8,
            "underground",
            "context",
            "context",
        )
        return ([row], [row])


class DummyAnalyzer:
    def __init__(self) -> None:
        self.cmu_loader = types.SimpleNamespace(get_rhyme_parts=lambda word: {"EH OW"})

    def describe_word(self, word: str) -> Dict[str, Any]:
        return {
            "word": word,
            "normalized": word.lower(),
            "tokens": [word.lower()],
            "anchor_word": word.lower(),
            "anchor_display": word,
            "is_multi_word": False,
        }

    def get_phonetic_similarity(self, word1: str, word2: str) -> float:
        return 0.85

    def derive_rhyme_profile(self, source: str, target: str, similarity: float | None = None) -> Dict[str, Any]:
        return {"similarity": similarity or 0.8}


class DummyAntiEngine:
    def __init__(self, patterns: Iterable[DummyPattern]) -> None:
        self._patterns = list(patterns)

    def generate_anti_llm_patterns(self, *args: Any, **kwargs: Any) -> List[DummyPattern]:
        return list(self._patterns)


class DummyCmuRepository:
    def __init__(self, results: Iterable[Any] | None = None) -> None:
        self._results = list(results or [])

    def lookup(
        self,
        source_word: str,
        limit: int,
        *,
        analyzer: Any | None = None,
        cmu_loader: Any | None = None,
    ) -> list[Any]:
        return list(self._results)


def _anti_entries(result: dict[str, list[dict]]) -> list[dict]:
    entries: list[dict] = []
    for bucket in ("uncommon", "multi_word"):
        entries.extend(result.get(bucket, []))
    return [entry for entry in entries if entry.get("result_source") == "anti_llm"]


@pytest.fixture(autouse=True)
def patch_core_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_extract_components(phrase: str, *_: Any, **__: Any) -> types.SimpleNamespace:
        return types.SimpleNamespace(
            original=phrase,
            tokens=[phrase],
            normalized_tokens=[phrase.lower()],
            normalized_phrase=phrase.lower(),
            anchor=phrase.lower(),
            anchor_display=phrase,
            anchor_index=0,
            syllable_counts=[1],
            total_syllables=1,
            anchor_pronunciations=[],
        )

    monkeypatch.setattr(
        "rhyme_rarity.app.services.search_service.extract_phrase_components",
        fake_extract_components,
    )


def test_search_service_records_branch_timings() -> None:
    telemetry = StructuredTelemetry(time_fn=FakeClock())
    repo = DummyRepository()
    analyzer = DummyAnalyzer()
    anti_engine = DummyAntiEngine([DummyPattern("rhythm", confidence=0.9, rarity_score=0.8)])
    cmu_repo = DummyCmuRepository([("echo", 0.9, 0.8, 0.9)])

    service = SearchService(
        repository=repo,
        phonetic_analyzer=analyzer,
        cultural_engine=None,
        anti_llm_engine=anti_engine,
        telemetry=telemetry,
        cmu_repository=cmu_repo,
    )

    result = service.search_rhymes("Echo", limit=5, result_sources=["phonetic", "cultural", "anti_llm"])
    assert result["uncommon"] or result["multi_word"] or result["rap_db"]

    metrics = service.get_latest_telemetry()
    assert metrics["counters"]["search.completed"] == 1
    assert "search.branch.phonetic" in metrics["timings"]
    assert "search.branch.cultural" in metrics["timings"]
    assert "search.branch.anti_llm" in metrics["timings"]
    counts = metrics["metadata"]["result.counts"]
    assert set(counts.keys()) == {"uncommon", "multi_word", "rap_db"}


def test_filter_telemetry_tracks_rejections() -> None:
    telemetry = StructuredTelemetry(time_fn=FakeClock())
    repo = DummyRepository()
    analyzer = DummyAnalyzer()
    cmu_repo = DummyCmuRepository([("ember", 0.85, 0.6, 0.85)])

    service = SearchService(
        repository=repo,
        phonetic_analyzer=analyzer,
        cultural_engine=None,
        anti_llm_engine=None,
        telemetry=telemetry,
        cmu_repository=cmu_repo,
    )

    service.search_rhymes(
        "Echo",
        limit=3,
        min_confidence=0.9,
        min_rarity=0.7,
        result_sources=["phonetic"],
    )

    metrics = service.get_latest_telemetry()
    filter_meta = metrics["metadata"].get("filters.phonetic")
    assert filter_meta is not None
    assert filter_meta["scope"] == "phonetic"
    assert filter_meta["input"] >= 1
    assert filter_meta["rejected"] >= 1
    assert "min_confidence" in filter_meta["reasons"]
    assert "min_confidence" in filter_meta["active_filters"]


def test_anti_llm_respects_dynamic_threshold() -> None:
    telemetry = StructuredTelemetry(time_fn=FakeClock())
    repo = DummyRepository()
    analyzer = DummyAnalyzer()
    cmu_repo = DummyCmuRepository([("mecho", 0.95, 0.8, 0.95)])
    anti_engine = DummyAntiEngine(
        [
            DummyPattern("Alpha", confidence=0.94, rarity_score=0.6),
            DummyPattern("Beta", confidence=0.83, rarity_score=0.7),
            DummyPattern("Gamma", confidence=0.6, rarity_score=0.4),
        ]
    )

    service = SearchService(
        repository=repo,
        phonetic_analyzer=analyzer,
        cultural_engine=None,
        anti_llm_engine=anti_engine,
        telemetry=telemetry,
        cmu_repository=cmu_repo,
    )

    results = service.search_rhymes(
        "Echo",
        limit=3,
        result_sources=["phonetic", "anti_llm"],
        min_confidence=0.6,
    )

    anti_targets = {entry["target_word"]: entry for entry in _anti_entries(results)}
    assert set(anti_targets) == {"Alpha", "Beta"}
    assert anti_targets["Alpha"]["threshold_gate"] == "strict"
    assert anti_targets["Beta"]["threshold_gate"] == "fallback"
    metrics = service.get_latest_telemetry()
    counters = metrics["counters"]
    assert counters["search.anti_llm.threshold_fallback"] >= 1
    assert counters["search.anti_llm.threshold_blocked"] >= 1


def test_formatter_emits_cadence_and_stress_diagnostics() -> None:
    formatter = RhymeResultFormatter()
    rhymes = {
        "source_profile": {
            "word": "Echo",
            "phonetics": {"syllables": 2, "stress_pattern_display": "1-0"},
            "threshold": 0.86,
            "signature_provenance": {
                "cultural": ["v:eh"],
                "spelling": ["e:cho"],
            },
        },
        "filters": {
            "cadence_focus": "smooth_flow",
            "min_stress_alignment": 0.72,
            "phonetic_threshold": 0.86,
            "signature_sources": {
                "cultural": ["v:eh"],
                "spelling": ["e:cho"],
            },
            "signature_set": ["v:eh|e:cho"],
        },
        "uncommon": [
            {
                "target_word": "Echo",
                "pattern": "echo / echo",
                "rarity_score": 0.9,
                "combined_score": 0.91,
                "target_phonetics": {"syllables": 2, "stress_pattern_display": "1-0"},
                "prosody_profile": {"complexity_tag": "smooth_flow"},
                "feature_profile": {"stress_alignment": 0.75},
                "stress_alignment": 0.75,
                "matched_signature_sources": ["alignment", "spelling"],
                "signature_context_matches": ["cultural", "spelling"],
                "threshold_gate": "strict",
                "phonetic_threshold": 0.86,
            }
        ],
        "multi_word": [],
        "rap_db": [],
    }

    output = formatter.format_rhyme_results("Echo", rhymes)
    assert "Diagnostics:" in output
    assert "Cadence focus: Smooth Flow" in output
    assert "Min stress alignment: 0.72" in output
    assert "Phonetic threshold: 0.86" in output
    assert "Signature support: Alignment, Spelling" in output
