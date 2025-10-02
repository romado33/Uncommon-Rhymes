from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rhyme_rarity.app.services.search_service import RhymeQueryOrchestrator
from rhyme_rarity.core import EnhancedPhoneticAnalyzer, passes_gate, score_pair
from rhyme_rarity.core import phrase_corpus
from rhyme_rarity.core.phrases import (
    generate_phrases_for_endwords,
    rank_phrases,
    retrieve_phrases_by_last_word,
)

def test_generate_phrases_end_with_permitted_tokens() -> None:
    allowed = {"widow", "pillow", "hello"}
    candidates = generate_phrases_for_endwords(
        "window", ["widow", "pillow", "hello"], beam_width=6, max_phrases=15
    )

    assert candidates, "Expected constrained beam search to surface phrases"
    for candidate in candidates:
        terminal = candidate.end_word.lower()
        assert terminal in allowed
        assert candidate.text.split()[-1].lower() in allowed


def test_retrieve_phrases_by_last_word_returns_idioms() -> None:
    phrase_corpus._load_ngram_corpus.cache_clear()
    try:
        with patch.object(phrase_corpus.resources, "files", side_effect=FileNotFoundError):
            with patch.object(
                phrase_corpus, "_mine_phrases_from_database", side_effect=AssertionError
            ) as mined:
                results = retrieve_phrases_by_last_word("window", rhyme_keys=("OW",))
    finally:
        phrase_corpus._load_ngram_corpus.cache_clear()

    phrases = {phrase for phrase, _ in results}
    assert "open window" in phrases

    mined.assert_not_called()

    for phrase, metadata in results:
        if phrase == "open window":
            assert metadata["source"] == "corpus_ngram"
            break
    else:  # pragma: no cover - guard
        raise AssertionError("Expected idiomatic corpus entry for open window")


def test_rank_phrases_blends_scores_and_metadata() -> None:
    analyzer = EnhancedPhoneticAnalyzer()
    ranked = rank_phrases(
        analyzer,
        "paper trail",
        (
            ("blazing trail", {"source": "corpus_ngram"}),
            ("snail mail", {"source": "corpus_ngram"}),
            ("shallow glow", {"source": "corpus_ngram"}),
        ),
    )

    assert [entry.phrase for entry in ranked] == ["blazing trail", "snail mail"]
    assert ranked[0].metadata["source"] == "corpus_ngram"
    assert ranked[0].why[0] == "phonetic tier: perfect"
    assert ranked[1].tier == "very_close"
    assert ranked[0].bonuses["prosody"] > ranked[1].bonuses["prosody"]


def test_slant_strength_gate_excludes_assonance_when_strict() -> None:
    analyzer = EnhancedPhoneticAnalyzer()
    strong_score = score_pair(analyzer, "orange", "porridge")
    assonance_score = score_pair(analyzer, "orange", "courage")
    assert not passes_gate(assonance_score)

    orchestrator = RhymeQueryOrchestrator(
        repository=SimpleNamespace(db_path=":memory:"),
        phonetic_analyzer=None,
        cultural_engine=None,
        anti_llm_engine=None,
        cmu_loader=None,
        cmu_repository=SimpleNamespace(),
        max_concurrent_searches=None,
        search_timeout=None,
        telemetry=None,
    )

    entries = [
        {
            "word": "porridge",
            "target_word": "porridge",
            "feature_profile": {"rhyme_type": "slant"},
            "slant_score": strong_score,
        },
        {
            "word": "courage",
            "target_word": "courage",
            "feature_profile": {"rhyme_type": "slant"},
            "slant_score": assonance_score,
        },
    ]

    strict, allowed_strict, saw_single = orchestrator._apply_slant_strength_gate(entries, strength=1.0)
    assert saw_single is True
    assert [entry["word"] for entry in strict] == ["porridge"]
    assert allowed_strict == {"porridge"}

    relaxed, allowed_relaxed, _ = orchestrator._apply_slant_strength_gate(entries, strength=0.5)
    assert {entry["word"] for entry in relaxed} == {"porridge", "courage"}
    assert allowed_relaxed == {"porridge", "courage"}
