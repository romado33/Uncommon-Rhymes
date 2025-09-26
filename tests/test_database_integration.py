from pathlib import Path
import sys
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from rhyme_rarity.app.data.database import SQLiteRhymeRepository


def _normalize_text(value: str) -> str:
    return str(value or "").strip().lower()


def _normalize_cultural(value: str) -> str:
    normalized = _normalize_text(value)
    return normalized.replace("_", "-") if normalized else normalized


@pytest.fixture
def seeded_repository(tmp_path) -> SQLiteRhymeRepository:
    db_path = tmp_path / "integration.db"
    repository = SQLiteRhymeRepository(str(db_path))
    repository.ensure_database()

    base_rows: List[
        Tuple[
            str,
            str,
            str,
            str,
            str,
            int,
            str,
            int,
            float,
            float,
            str,
            str,
            str,
            str,
        ]
    ] = [
        (
            "pattern-accepted-source",
            "sun",
            "fun",
            "Artist One",
            "Solar Rhymes",
            1999,
            "hip-hop",
            1,
            0.95,
            0.93,
            "high_impact",
            "the sun is shining",
            "bring the fun again",
            "the sun is shining // bring the fun again",
        ),
        (
            "pattern-low-confidence",
            "sun",
            "done",
            "Artist One",
            "Solar Rhymes",
            1999,
            "hip-hop",
            1,
            0.6,
            0.95,
            "high_impact",
            "confidence too low",
            "confidence too low",
            "confidence too low // confidence too low",
        ),
        (
            "pattern-low-phonetic",
            "sun",
            "moon",
            "Artist Two",
            "Lunar Rhymes",
            2005,
            "hip-hop",
            1,
            0.92,
            0.85,
            "high_impact",
            "phonetics too low",
            "phonetics too low",
            "phonetics too low // phonetics too low",
        ),
        (
            "pattern-wrong-genre",
            "sun",
            "pun",
            "Artist Three",
            "Comedy Rhymes",
            2010,
            "jazz",
            1,
            0.96,
            0.94,
            "high_impact",
            "genre mismatch",
            "genre mismatch",
            "genre mismatch // genre mismatch",
        ),
        (
            "pattern-wrong-culture",
            "sun",
            "bun",
            "Artist Four",
            "Bakery Rhymes",
            2001,
            "hip-hop",
            1,
            0.97,
            0.96,
            "low_impact",
            "culture mismatch",
            "culture mismatch",
            "culture mismatch // culture mismatch",
        ),
        (
            "pattern-accepted-target",
            "fun",
            "sun",
            "Artist Five",
            "Return Rhymes",
            1998,
            "hip-hop",
            1,
            0.94,
            0.91,
            "high_impact",
            "source flip works",
            "source flip works",
            "source flip works // source flip works",
        ),
        (
            "pattern-target-low-confidence",
            "run",
            "sun",
            "Artist Six",
            "Runner Rhymes",
            1998,
            "hip-hop",
            1,
            0.65,
            0.92,
            "high_impact",
            "target confidence low",
            "target confidence low",
            "target confidence low // target confidence low",
        ),
    ]
    rows: List[Tuple] = []
    for entry in base_rows:
        (
            pattern,
            source_word,
            target_word,
            artist,
            song_title,
            release_year,
            genre,
            line_distance,
            confidence,
            phonetic_similarity,
            cultural_significance,
            source_context,
            target_context,
            lyrical_context,
        ) = entry
        rows.append(
            (
                pattern,
                source_word,
                _normalize_text(source_word),
                target_word,
                _normalize_text(target_word),
                artist,
                _normalize_text(artist),
                song_title,
                release_year,
                genre,
                line_distance,
                confidence,
                phonetic_similarity,
                cultural_significance,
                source_context,
                target_context,
                lyrical_context,
                _normalize_text(genre),
                _normalize_cultural(cultural_significance),
            )
        )

    with repository._connect() as connection:
        cursor = connection.cursor()
        cursor.execute("DELETE FROM song_rhyme_patterns")
        cursor.executemany(
            """
            INSERT INTO song_rhyme_patterns (
                pattern,
                source_word,
                source_word_normalized,
                target_word,
                target_word_normalized,
                artist,
                artist_normalized,
                song_title,
                release_year,
                genre,
                line_distance,
                confidence_score,
                phonetic_similarity,
                cultural_significance,
                source_context,
                target_context,
                lyrical_context,
                genre_normalized,
                cultural_significance_normalized
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        connection.commit()

    return repository


def test_fetch_cultural_matches_applies_all_filters(seeded_repository: SQLiteRhymeRepository) -> None:
    source_rows, target_rows = seeded_repository.fetch_cultural_matches(
        "sun",
        min_confidence=0.8,
        phonetic_threshold=0.9,
        cultural_filters=["high-impact"],
        genre_filters=["hip-hop"],
        max_line_distance=None,
        limit=None,
    )

    assert [(row[0], row[1]) for row in source_rows] == [("sun", "fun")]
    assert [(row[0], row[1]) for row in target_rows] == [("fun", "sun")]


def test_fetch_cultural_matches_is_case_insensitive(
    seeded_repository: SQLiteRhymeRepository,
) -> None:
    base_source_rows, base_target_rows = seeded_repository.fetch_cultural_matches(
        "sun",
        min_confidence=0.8,
        phonetic_threshold=0.9,
        cultural_filters=["high-impact"],
        genre_filters=["hip-hop"],
        max_line_distance=None,
        limit=None,
    )

    mixed_source_rows, mixed_target_rows = seeded_repository.fetch_cultural_matches(
        "SuN",
        min_confidence=0.8,
        phonetic_threshold=0.9,
        cultural_filters=["high-impact"],
        genre_filters=["hip-hop"],
        max_line_distance=None,
        limit=None,
    )

    assert mixed_source_rows == base_source_rows
    assert mixed_target_rows == base_target_rows
