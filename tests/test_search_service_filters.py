"""Tests for search service filter helpers and normalisation logic."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List

import types

import pytest

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rhyme_rarity.app.services import search_service as search_service_module
from rhyme_rarity.app.services.search_service import SearchService


@dataclass
class DummyPattern:
    """Lightweight stand-in for Anti-LLM pattern results."""

    target_word: str
    confidence: float
    rarity_score: float
    stress_alignment: float | None = None
    prosody_profile: dict[str, Any] | None = None

    llm_weakness_type: str | None = None
    cultural_depth: float | None = None
    feature_profile: dict[str, Any] | None = None
    combined: float | None = None
    combined_score: float | None = None
    syllable_span: Iterable[int] | None = None
    bradley_device: str | None = None


class DummyRepository:
    """Repository stub that satisfies the search service API."""

    def fetch_related_words(self, terms: Iterable[str]) -> set[str]:
        return set()

    def fetch_cultural_matches(self, *args: Any, **kwargs: Any) -> tuple[List[tuple], List[tuple]]:
        return ([], [])


class DummyCmuRepository:
    """Repository stub returning no CMU results."""

    def lookup(
        self,
        source_word: str,
        limit: int,
        *,
        analyzer: Any | None = None,
        cmu_loader: Any | None = None,
    ) -> list[Any]:
        return []


class DummyAntiEngine:
    """Anti-LLM engine stub returning predetermined patterns."""

    def __init__(self, patterns: Iterable[DummyPattern]):
        self._patterns = list(patterns)

    def generate_anti_llm_patterns(self, *args: Any, **kwargs: Any) -> List[DummyPattern]:
        return list(self._patterns)


class RecordingCulturalEngine:
    """Cultural engine stub that tracks alignment invocations."""

    def __init__(self) -> None:
        self.align_calls: int = 0

    def derive_rhyme_signatures(self, source_word: str) -> list[str]:
        # Provide a deterministic signature so phonetic context building succeeds.
        return [f"sig::{source_word.lower()}"]

    def evaluate_rhyme_alignment(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self.align_calls += 1
        return {"combined": 0.9, "rarity": 0.5}


class RichCmuRepository(DummyCmuRepository):
    """CMU repository stub that returns a single high-quality rhyme."""

    def lookup(
        self,
        source_word: str,
        limit: int,
        *,
        analyzer: Any | None = None,
        cmu_loader: Any | None = None,
    ) -> list[Any]:
        return [
            {
                "word": "cove",
                "similarity": 0.92,
                "rarity": 0.4,
                "combined": 0.88,
            }
        ]


@pytest.fixture(autouse=True)
def _patch_search_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide lightweight core helpers for the search service."""

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

    monkeypatch.setattr(search_service_module, "extract_phrase_components", fake_extract_components)


def make_service(patterns: Iterable[DummyPattern]) -> SearchService:
    repo = DummyRepository()
    anti_engine = DummyAntiEngine(patterns)
    cmu_repo = DummyCmuRepository()
    return SearchService(repository=repo, anti_llm_engine=anti_engine, cmu_repository=cmu_repo)


def _targets(result: dict[str, list[dict]]) -> list[str]:
    entries: list[dict] = []
    for bucket in ("perfect", "slant", "multi_word"):
        entries.extend(result.get(bucket, []))
    return [
        entry["target_word"]
        for entry in entries
        if entry.get("result_source") == "anti_llm"
    ]


def _first_anti_entry(result: dict[str, list[dict]]) -> dict:
    for bucket in ("perfect", "slant", "multi_word"):
        for entry in result.get(bucket, []):
            if entry.get("result_source") == "anti_llm":
                return entry
    raise AssertionError("No anti-LLM entry found in result")


def test_min_confidence_filter_blocks_low_scoring_entries() -> None:
    patterns = [
        DummyPattern("Alpha", confidence=0.9, rarity_score=0.5, stress_alignment=0.5),
        DummyPattern("Beta", confidence=0.4, rarity_score=0.6, stress_alignment=0.6),
    ]
    service = make_service(patterns)

    result = service.search_rhymes(
        "Echo",
        min_confidence=0.8,
        result_sources=["  Anti_LLM  "],
    )

    assert _targets(result) == ["Alpha"]
    assert result["multi_word"] == []
    assert result["rap_db"] == []


def test_rarity_and_stress_filters_keep_only_strong_matches() -> None:
    patterns = [
        DummyPattern("Alpha", confidence=0.9, rarity_score=0.6, stress_alignment=0.8),
        DummyPattern("Beta", confidence=0.9, rarity_score=0.3, stress_alignment=0.9),
        DummyPattern("Gamma", confidence=0.9, rarity_score=0.7, stress_alignment=0.5),
    ]
    service = make_service(patterns)

    result = service.search_rhymes(
        "Echo",
        min_confidence=0.0,
        min_rarity=0.5,
        min_stress_alignment=0.7,
        result_sources=["anti_llm"],
    )

    assert _targets(result) == ["Alpha"]
    anti_entry = _first_anti_entry(result)
    assert anti_entry["rarity_score"] >= 0.5
    assert anti_entry["stress_alignment"] >= 0.7


def test_cadence_focus_matches_normalised_complexity_tag() -> None:
    patterns = [
        DummyPattern(
            "Alpha",
            confidence=0.9,
            rarity_score=0.4,
            stress_alignment=0.75,
            prosody_profile={"complexity_tag": "Smooth Flow"},
        ),
        DummyPattern(
            "Beta",
            confidence=0.9,
            rarity_score=0.8,
            stress_alignment=0.9,
            prosody_profile={"complexity_tag": "Percussive"},
        ),
    ]
    service = make_service(patterns)

    result = service.search_rhymes(
        "Echo",
        min_confidence=0.0,
        cadence_focus="Smooth Flow",
        result_sources=["anti_llm"],
    )

    assert _targets(result) == ["Alpha"]
    anti_entry = _first_anti_entry(result)
    assert anti_entry["prosody_profile"]["complexity_tag"] == "Smooth Flow"


def test_phonetic_only_search_skips_cultural_alignment() -> None:
    repo = DummyRepository()
    cultural_engine = RecordingCulturalEngine()
    cmu_repo = RichCmuRepository()
    service = SearchService(
        repository=repo,
        cultural_engine=cultural_engine,
        cmu_repository=cmu_repo,
    )

    result = service.search_rhymes("Echo", result_sources=["phonetic"], limit=5)

    assert cultural_engine.align_calls == 0
    single_results = list(result.get("perfect", [])) + list(result.get("slant", []))
    assert any(entry["result_source"] == "phonetic" for entry in single_results)


def test_phonetic_with_cultural_results_triggers_alignment() -> None:
    repo = DummyRepository()
    cultural_engine = RecordingCulturalEngine()
    cmu_repo = RichCmuRepository()
    service = SearchService(
        repository=repo,
        cultural_engine=cultural_engine,
        cmu_repository=cmu_repo,
    )

    service.search_rhymes(
        "Echo",
        result_sources=["phonetic", "cultural"],
        limit=5,
    )

    assert cultural_engine.align_calls >= 1
