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


class DummyAntiEngine:
    """Anti-LLM engine stub returning predetermined patterns."""

    def __init__(self, patterns: Iterable[DummyPattern]):
        self._patterns = list(patterns)

    def generate_anti_llm_patterns(self, *args: Any, **kwargs: Any) -> List[DummyPattern]:
        return list(self._patterns)


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
    monkeypatch.setattr(
        search_service_module.CmuRhymeRepository,
        "fetch_rhymes",
        lambda self, *args, **kwargs: [],
    )


def make_service(patterns: Iterable[DummyPattern]) -> SearchService:
    repo = DummyRepository()
    anti_engine = DummyAntiEngine(patterns)
    return SearchService(repository=repo, anti_llm_engine=anti_engine)


def _targets(result: dict[str, list[dict]]) -> list[str]:
    return [entry["target_word"] for entry in result["anti_llm"]]


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
    assert result["cmu"] == []
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
    assert result["anti_llm"][0]["rarity_score"] >= 0.5
    assert result["anti_llm"][0]["stress_alignment"] >= 0.7


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
    assert result["anti_llm"][0]["prosody_profile"]["complexity_tag"] == "Smooth Flow"
