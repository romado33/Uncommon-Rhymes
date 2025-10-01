from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from rhyme_rarity.app.data.database import SQLiteRhymeRepository
from rhyme_rarity.core.analyzer import normalize_rime_key, phrase_rime_keys
from rhyme_rarity.core.cmudict_loader import CMUDictLoader
from rhyme_rarity.core.feature_profile import extract_phrase_components


def _normalize_text(value: str) -> str:
    return str(value or "").strip().lower()


def _normalize_cultural(value: str) -> str:
    normalized = _normalize_text(value)
    return normalized.replace("_", "-") if normalized else normalized


_PHRASE_LOADER = CMUDictLoader()
_KEY_CACHE: Dict[str, Tuple[Tuple[Optional[str], Optional[str], Optional[str]], Dict[str, List[str]]]] = {}


def _phrase_key_bundle(
    phrase: Optional[str],
) -> Tuple[Tuple[Optional[str], Optional[str], Optional[str]], Dict[str, List[str]]]:
    normalized = _normalize_text(phrase or "")
    if not normalized:
        empty: Dict[str, List[str]] = {"end_word": [], "compound": [], "backoff": []}
        return (None, None, None), empty

    cached = _KEY_CACHE.get(normalized)
    if cached is not None:
        tuple_keys, key_sets = cached
        return tuple_keys, {key: list(values) for key, values in key_sets.items()}

    components = extract_phrase_components(phrase or "", _PHRASE_LOADER)
    key_info = phrase_rime_keys(components, _PHRASE_LOADER)

    def _collect(values: Tuple[str, ...]) -> List[str]:
        collected: List[str] = []
        for value in values:
            normalized_value = normalize_rime_key(value)
            if normalized_value:
                collected.append(normalized_value)
        return collected

    key_sets = {
        "end_word": _collect(key_info.anchor_rhymes),
        "compound": _collect(key_info.compound_strings),
        "backoff": _collect(key_info.backoff_keys),
    }

    tuple_keys = (
        key_sets["end_word"][0] if key_sets["end_word"] else None,
        key_sets["compound"][0] if key_sets["compound"] else None,
        key_sets["backoff"][0] if key_sets["backoff"] else None,
    )

    _KEY_CACHE[normalized] = (tuple_keys, key_sets)
    return tuple_keys, {key: list(values) for key, values in key_sets.items()}


def _phrase_key_dict(phrase: str) -> Dict[str, List[str]]:
    _, key_sets = _phrase_key_bundle(phrase)
    return {key: list(values) for key, values in key_sets.items() if values}


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
        (
            "pattern-multi-phrase",
            "have fun",
            "run",
            "Artist Seven",
            "Playful Lines",
            2003,
            "hip-hop",
            1,
            0.91,
            0.92,
            "high_impact",
            "we came to have fun",
            "now we run and gun",
            "we came to have fun // now we run and gun",
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
        source_key_tuple, _ = _phrase_key_bundle(source_word)
        target_key_tuple, _ = _phrase_key_bundle(target_word)
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
                source_key_tuple[0],
                source_key_tuple[1],
                source_key_tuple[2],
                target_key_tuple[0],
                target_key_tuple[1],
                target_key_tuple[2],
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
                cultural_significance_normalized,
                source_last_word_rime_key,
                source_compound_rime_key,
                source_backoff_rime_key,
                target_last_word_rime_key,
                target_compound_rime_key,
                target_backoff_rime_key
            )
            VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
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


def test_fetch_cultural_matches_returns_empty_for_missing_term(
    seeded_repository: SQLiteRhymeRepository,
) -> None:
    source_rows, target_rows = seeded_repository.fetch_cultural_matches(
        "unknown",
        min_confidence=0.8,
        phonetic_threshold=0.9,
        cultural_filters=["high-impact"],
        genre_filters=["hip-hop"],
        max_line_distance=None,
        limit=None,
    )

    assert source_rows == []
    assert target_rows == []


def test_fetch_cultural_matches_uses_phonetic_keys_for_multi_word_queries(
    seeded_repository: SQLiteRhymeRepository,
) -> None:
    key_payload = _phrase_key_dict("have fun")

    source_rows, target_rows = seeded_repository.fetch_cultural_matches(
        "have fun",
        min_confidence=0.8,
        phonetic_threshold=0.9,
        cultural_filters=["high-impact"],
        genre_filters=["hip-hop"],
        max_line_distance=None,
        limit=None,
        phonetic_keys=key_payload,
    )

    source_pairs = {(row[0], row[1]) for row in source_rows}
    assert ("have fun", "run") in source_pairs
    assert ("sun", "fun") in source_pairs

    target_pairs = {(row[0], row[1]) for row in target_rows}
    assert ("fun", "sun") in target_pairs
