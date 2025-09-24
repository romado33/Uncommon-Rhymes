import types
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pytest

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rhyme_rarity.app.services import search_service as search_service_module
from rhyme_rarity.app.services.search_service import SearchService


class DummyCMULoader:
    def __init__(self, signatures: Dict[str, Sequence[str]]):
        self._signatures = {
            key.lower(): tuple(value) for key, value in signatures.items()
        }

    def get_rhyme_parts(self, word: str) -> Tuple[str, ...]:
        return self._signatures.get(word.lower(), tuple())


class DummyAnalyzer:
    def __init__(
        self,
        cmu_loader: DummyCMULoader,
        similarities: Dict[Tuple[str, str], float],
        rarity: Optional[Dict[str, float]] = None,
    ) -> None:
        self.cmu_loader = cmu_loader
        self._similarities = {
            (src.lower(), tgt.lower()): value for (src, tgt), value in similarities.items()
        }
        self._rarity = {key.lower(): value for key, value in (rarity or {}).items()}

    def describe_word(self, word: str) -> Dict[str, Any]:  # pragma: no cover - simple container
        normalized = (word or "").lower().strip()
        tokens = normalized.split()
        return {
            "word": word,
            "normalized": normalized,
            "tokens": tokens or [normalized],
            "token_syllables": [1 for _ in (tokens or [normalized])],
            "syllables": max(1, len(tokens) or 1),
            "stress_pattern": "1",
            "stress_pattern_display": "1",
            "anchor_word": tokens[-1] if tokens else normalized,
            "anchor_display": (tokens[-1] if tokens else word),
        }

    def estimate_syllables(self, _: str) -> int:  # pragma: no cover - deterministic
        return 1

    def get_phonetic_similarity(self, source: str, target: str) -> float:
        return self._similarities.get((source.lower(), target.lower()), 0.0)

    def get_rarity_score(self, word: str) -> float:  # pragma: no cover - deterministic
        return self._rarity.get(word.lower(), 0.5)

    def combine_similarity_and_rarity(self, similarity: float, rarity: float) -> float:
        return (similarity + rarity) / 2

    def derive_rhyme_profile(
        self,
        source: str,
        target: str,
        similarity: Optional[float] = None,
        rhyme_type: Optional[str] = None,
    ) -> Dict[str, Any]:  # pragma: no cover - deterministic
        return {
            "syllable_span": (1, 2),
            "stress_alignment": 0.8,
            "internal_rhyme_score": 0.4,
            "rhyme_type": rhyme_type or "perfect",
        }

    def analyze_rhyme_pattern(self, source: str, target: str):  # pragma: no cover - deterministic
        similarity = self.get_phonetic_similarity(source, target)
        profile = self.derive_rhyme_profile(source, target, similarity=similarity)
        return types.SimpleNamespace(
            similarity_score=similarity,
            phonetic_features={"stress_alignment": 0.8},
            rhyme_type="perfect",
            feature_profile=profile,
        )


@dataclass
class DummyPattern:
    target_word: str
    confidence: float
    rarity_score: float


class DummyAntiEngine:
    def __init__(self, patterns: Iterable[DummyPattern]):
        self._patterns = list(patterns)

    def generate_anti_llm_patterns(self, *args: Any, **kwargs: Any) -> List[DummyPattern]:
        return list(self._patterns)


class DummyRepository:
    def __init__(self, rows: Optional[Tuple[List[Tuple], List[Tuple]]]) -> None:
        self._rows = rows

    def fetch_related_words(self, terms: Iterable[str]) -> set[str]:
        return set()

    def fetch_cultural_matches(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[List[Tuple], List[Tuple]]:
        if self._rows is None:
            return ([], [])
        return self._rows


class StubCulturalEngine:
    def __init__(self, analyzer: DummyAnalyzer) -> None:
        self.phonetic_analyzer = analyzer

    def evaluate_rhyme_alignment(
        self,
        source_word: str,
        target_word: str,
        threshold: Optional[float] = None,
        rhyme_signatures: Optional[Sequence[str]] = None,
        **_: Any,
    ) -> Optional[Dict[str, Any]]:
        source_set = {str(sig) for sig in (rhyme_signatures or []) if sig}
        target_set = set(self.phonetic_analyzer.cmu_loader.get_rhyme_parts(target_word))
        if source_set and target_set and not source_set.intersection(target_set):
            return None

        similarity = self.phonetic_analyzer.get_phonetic_similarity(source_word, target_word)
        if threshold is not None and similarity < float(threshold):
            return None

        profile = self.phonetic_analyzer.derive_rhyme_profile(source_word, target_word, similarity)
        return {
            "similarity": similarity,
            "rarity": self.phonetic_analyzer.get_rarity_score(target_word),
            "combined": self.phonetic_analyzer.combine_similarity_and_rarity(
                similarity,
                self.phonetic_analyzer.get_rarity_score(target_word),
            ),
            "rhyme_type": profile.get("rhyme_type"),
            "signature_matches": sorted(source_set.intersection(target_set)),
            "target_signatures": sorted(target_set),
            "feature_profile": profile,
            "prosody_profile": {
                "complexity_tag": "steady",
                "cadence_ratio": 1.0,
            },
        }


@pytest.fixture(autouse=True)
def _patch_phrase_components(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_extract_components(phrase: str, *_: Any, **__: Any) -> types.SimpleNamespace:
        normalized = phrase.lower()
        return types.SimpleNamespace(
            original=phrase,
            tokens=[normalized],
            normalized_tokens=[normalized],
            normalized_phrase=normalized,
            anchor=normalized,
            anchor_display=phrase,
            anchor_index=0,
            syllable_counts=[1],
            total_syllables=1,
            anchor_pronunciations=[],
        )

    monkeypatch.setattr(search_service_module, "extract_phrase_components", fake_extract_components)


def test_anti_llm_results_apply_phonetic_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    loader = DummyCMULoader({
        "echo": ["sig-a"],
        "shine": ["sig-a"],
        "prime": ["sig-a"],
    })
    analyzer = DummyAnalyzer(
        loader,
        {
            ("echo", "shine"): 0.55,
            ("echo", "prime"): 0.88,
        },
        rarity={"shine": 0.4, "prime": 0.6},
    )

    # Ensure the dynamic threshold is driven by a strong CMU reference.
    monkeypatch.setattr(
        search_service_module,
        "get_cmu_rhymes",
        lambda *args, **kwargs: [("prime", 0.95, 0.6, 0.95)],
    )

    repo = DummyRepository(rows=None)
    anti_engine = DummyAntiEngine(
        [
            DummyPattern(target_word="shine", confidence=0.92, rarity_score=0.5),
            DummyPattern(target_word="prime", confidence=0.91, rarity_score=0.7),
        ]
    )

    service = SearchService(
        repository=repo,
        anti_llm_engine=anti_engine,
        phonetic_analyzer=analyzer,
    )

    results = service.search_rhymes(
        "echo",
        limit=5,
        min_confidence=0.0,
        result_sources=["anti_llm"],
    )

    anti_targets = [entry["target_word"] for entry in results["anti_llm"]]
    assert anti_targets == ["prime"]

    telemetry = results["telemetry"]["filters"]["anti_llm"]
    assert telemetry["rejections"]["below_phonetic_threshold"] == 1


def test_cultural_alignment_requires_matching_signatures(monkeypatch: pytest.MonkeyPatch) -> None:
    loader = DummyCMULoader({
        "echo": ["sig-a"],
        "stone": ["sig-b"],
    })
    analyzer = DummyAnalyzer(
        loader,
        {
            ("echo", "stone"): 0.87,
        },
        rarity={"stone": 0.5},
    )

    monkeypatch.setattr(
        search_service_module,
        "get_cmu_rhymes",
        lambda *args, **kwargs: [("echo", 0.9, 0.6, 0.9)],
    )

    repo = DummyRepository(
        rows=(
            [
                (
                    "echo",
                    "stone",
                    "Test Artist",
                    "Test Song",
                    "echo / stone",
                    "hip-hop",
                    1,
                    0.96,
                    0.9,
                    "legendary",
                    "Echo verse",
                    "Stone reply",
                )
            ],
            [],
        )
    )
    cultural_engine = StubCulturalEngine(analyzer)

    service = SearchService(
        repository=repo,
        phonetic_analyzer=analyzer,
        cultural_engine=cultural_engine,
    )

    results = service.search_rhymes(
        "echo",
        limit=5,
        min_confidence=0.0,
        result_sources=["cultural"],
    )

    assert results["rap_db"] == []
    telemetry = results["telemetry"]["filters"]["rap_db"]
    assert telemetry["rejections"]["alignment_failed"] == 1
