import sqlite3
from pathlib import Path

import pytest

from rhyme_rarity.core.phrase_corpus import lookup_ngram_phrases


@pytest.mark.parametrize("word", ["time", "fire", "dream", "mind", "heart"])
def test_lookup_ngram_phrases_returns_general_phrases(word: str) -> None:
    """Ensure common vocabulary pulls idiomatic phrases from the corpus."""

    phrases = lookup_ngram_phrases(word, ())
    assert phrases, f"Expected idioms for {word}"
    assert any(phrase.lower().split()[-1] == word for phrase in phrases)


def test_lookup_ngram_phrases_dynamic_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When the corpus lacks a word the database fallback should supply phrases."""

    db_path = tmp_path / "patterns.db"
    connection = sqlite3.connect(db_path)
    try:
        connection.execute(
            """
            CREATE TABLE song_rhyme_patterns (
                target_word TEXT,
                target_word_normalized TEXT,
                target_context TEXT,
                lyrical_context TEXT,
                confidence_score REAL,
                phonetic_similarity REAL,
                pattern TEXT
            )
            """
        )
        connection.execute(
            """
            INSERT INTO song_rhyme_patterns (
                target_word,
                target_word_normalized,
                target_context,
                lyrical_context,
                confidence_score,
                phonetic_similarity,
                pattern
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "embers",
                "embers",
                "glow like embers",
                "shadows glow like embers",
                0.9,
                0.8,
                "embers / members",
            ),
        )
        connection.commit()
    finally:
        connection.close()

    monkeypatch.setenv("RHYME_RARITY_PHRASE_DB", str(db_path))

    no_dynamic = lookup_ngram_phrases("embers", (), enable_dynamic_fallback=False)
    assert no_dynamic == set()

    phrases = lookup_ngram_phrases("embers", ())
    normalized = {phrase.lower() for phrase in phrases}
    assert "glow like embers" in normalized
    assert all(phrase.split()[-1] == "embers" for phrase in normalized)
