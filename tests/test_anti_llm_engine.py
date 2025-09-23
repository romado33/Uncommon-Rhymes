from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anti_llm import AntiLLMRhymeEngine, SeedCandidate
from anti_llm.strategies import find_cultural_depth_patterns
from rhyme_rarity.app.data.database import SQLiteRhymeRepository


class DummyRarityMap:
    def __init__(self, value: float = 0.42) -> None:
        self.value = value
        self.calls = []

    def get_rarity(self, word: str) -> float:
        self.calls.append(word)
        return self.value


def test_seed_normalization_uses_shared_rarity_map():
    rarity_map = DummyRarityMap(0.73)
    engine = AntiLLMRhymeEngine(rarity_map=rarity_map)

    seeds = engine._normalize_seed_candidates(["Phantom"])

    assert rarity_map.calls == ["phantom"]
    assert seeds
    assert seeds[0].rarity == pytest.approx(0.73)


def test_analyzer_rarity_map_is_respected():
    rarity_map = DummyRarityMap(0.31)

    class DummyAnalyzer:
        def __init__(self) -> None:
            self.rarity_map = rarity_map

    engine = AntiLLMRhymeEngine()
    engine.set_phonetic_analyzer(DummyAnalyzer())

    engine._normalize_seed_candidates(["Echo"])

    assert rarity_map.calls == ["echo"]


def test_seed_expansion_uses_shared_rarity_map():
    rarity_map = DummyRarityMap(3.0)
    engine = AntiLLMRhymeEngine(rarity_map=rarity_map)

    engine._ensure_seed_resources = lambda: None  # type: ignore[assignment]
    engine._seed_analyzer = None
    engine._cmu_seed_fn = None
    engine._query_seed_neighbors = lambda cursor, seed, limit: [  # type: ignore[assignment]
        {"candidate": "Novel", "rarity": 0.0, "confidence": 0.9}
    ]
    engine._extract_suffixes = lambda word: set()  # type: ignore[assignment]
    engine._query_suffix_matches = lambda cursor, suffix, limit: []  # type: ignore[assignment]
    engine._normalize_module1_candidates = lambda candidates: []  # type: ignore[assignment]
    engine._get_phonetic_fingerprint = lambda word: set()  # type: ignore[assignment]
    engine._analyze_phonological_complexity = lambda w1, w2: 0.0  # type: ignore[assignment]
    engine._calculate_syllable_complexity = lambda word: 0.0  # type: ignore[assignment]

    seed = SeedCandidate(word="Spark", rarity=2.0)

    patterns = engine._expand_from_seed_candidates(
        None,
        "source",
        [seed],
        limit=1,
        signature_hints=set(),
        seen_targets=set(),
    )

    assert "novel" in rarity_map.calls
    assert patterns
    assert patterns[0].target_word == "Novel"
    assert patterns[0].rarity_score >= 3.0


def test_syllable_complexity_uses_estimator(monkeypatch: pytest.MonkeyPatch):
    calls = []

    def fake_estimator(word: str) -> int:
        calls.append(word)
        return 4

    monkeypatch.setattr(
        "anti_llm.engine.estimate_syllable_count", fake_estimator
    )

    engine = AntiLLMRhymeEngine()

    complexity = engine._calculate_syllable_complexity("Cascade")

    assert calls == ["Cascade"]

    pattern_changes = 0
    word = "Cascade"
    for i in range(1, len(word)):
        is_vowel_now = word[i] in "aeiou"
        was_vowel_before = word[i - 1] in "aeiou"
        if is_vowel_now != was_vowel_before:
            pattern_changes += 1

    expected_complexity = 4 * 0.5 + pattern_changes * 0.1

    assert complexity == pytest.approx(expected_complexity)


def test_cultural_depth_strategy_returns_patterns(tmp_path):
    db_path = tmp_path / "patterns.db"
    repository_builder = SQLiteRhymeRepository(str(db_path))
    repository_builder.ensure_database()

    engine = AntiLLMRhymeEngine(db_path=str(db_path))

    patterns = find_cultural_depth_patterns(
        engine.repository,
        source_word="love",
        limit=5,
        cultural_weights=engine.cultural_depth_weights,
        attach_profile=lambda pattern: None,
    )

    assert patterns
    assert any("[mainstream]" in pattern.cultural_depth for pattern in patterns)
