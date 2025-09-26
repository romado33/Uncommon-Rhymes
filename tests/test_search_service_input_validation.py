"""Input validation edge case tests for :mod:`search_service`."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List
import sys
import types

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rhyme_rarity.app.services import search_service as search_service_module
from rhyme_rarity.app.services.search_service import SearchService


@dataclass
class DummyPattern:
    """Simplified stand-in for Anti-LLM pattern objects."""

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


def _anti_targets(result: dict[str, list[dict]]) -> list[str]:
    entries: list[dict] = []
    for bucket in ("perfect", "slant", "multi_word"):
        entries.extend(result.get(bucket, []))
    return [
        entry["target_word"]
        for entry in entries
        if entry.get("result_source") == "anti_llm"
    ]


def test_search_rhymes_returns_empty_for_null_source_word() -> None:
    service = make_service([])

    result = service.search_rhymes(None)

    assert result == {"perfect": [], "slant": [], "multi_word": [], "rap_db": []}


def test_search_rhymes_coerces_invalid_min_confidence() -> None:
    patterns = [DummyPattern("Alpha", confidence=0.2, rarity_score=0.1)]
    service = make_service(patterns)

    result = service.search_rhymes(
        "Echo",
        min_confidence="not-a-number",
        result_sources=["anti_llm"],
    )

    assert _anti_targets(result) == ["Alpha"]


def test_search_rhymes_treats_string_result_sources_like_iterables() -> None:
    patterns = [DummyPattern("Alpha", confidence=0.9, rarity_score=0.3)]
    service = make_service(patterns)

    result = service.search_rhymes("Echo", result_sources="Anti_LLM")

    assert _anti_targets(result) == ["Alpha"]


def test_search_rhymes_zero_limit_short_circuits_results() -> None:
    patterns = [DummyPattern("Alpha", confidence=0.9, rarity_score=0.3)]
    service = make_service(patterns)

    result = service.search_rhymes("Echo", limit=0, result_sources=["anti_llm"])

    assert result == {"perfect": [], "slant": [], "multi_word": [], "rap_db": []}


def test_normalize_filter_label_sanitises_whitespace_and_underscores() -> None:
    service = make_service([])

    assert service.normalize_filter_label(None) == ""
    assert service.normalize_filter_label("  Mixed_CASE_Name  ") == "mixed-case-name"
