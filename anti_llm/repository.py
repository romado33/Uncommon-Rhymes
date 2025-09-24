"""Data access helpers for the anti-LLM rhyme engine."""

from __future__ import annotations

import sqlite3
from typing import Any, Dict, List, Optional, Protocol, Sequence, Set

from rhyme_rarity.app.data.database import SQLiteRhymeRepository

Row = Dict[str, Any]


def _normalize_text(value: Optional[str]) -> str:
    return str(value or "").strip().lower()


def _normalize_cultural_value(value: Optional[str]) -> Optional[str]:
    normalized = _normalize_text(value)
    if not normalized:
        return None
    return normalized.replace("_", "-")


class PatternRepository(Protocol):
    """Abstract repository describing the queries required by the engine."""

    def fetch_rare_combinations(self, source_word: str, limit: int) -> List[Row]:
        ...

    def fetch_phonological_challenges(self, source_word: str, limit: int) -> List[Row]:
        ...

    def fetch_cultural_depth_patterns(self, source_word: str, limit: int) -> List[Row]:
        ...

    def fetch_complex_syllable_patterns(self, source_word: str, limit: int) -> List[Row]:
        ...

    def fetch_seed_neighbors(self, seed_word: str, limit: int) -> List[Row]:
        ...

    def fetch_suffix_matches(self, suffix: str, limit: int) -> List[Row]:
        ...


class SQLitePatternRepository:
    """SQLite-backed implementation of :class:`PatternRepository`."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._cultural_significance_labels: Optional[List[str]] = None
        self._connection: Optional[sqlite3.Connection] = None
        self._normalizer: Optional[SQLiteRhymeRepository] = None

        try:
            normalizer = SQLiteRhymeRepository(db_path)
            normalizer.ensure_database()
            self._normalizer = normalizer
        except sqlite3.Error:
            self._normalizer = None

    def _get_connection(self) -> sqlite3.Connection:
        connection = self._connection
        if connection is None:
            connection = sqlite3.connect(self.db_path)
            connection.row_factory = sqlite3.Row
            self._connection = connection
        return connection

    def close(self) -> None:
        connection = self._connection
        if connection is not None:
            connection.close()
            self._connection = None

    def _execute(self, query: str, params: Sequence[Any]) -> List[sqlite3.Row]:
        self._ensure_normalized_columns()
        connection = self._get_connection()
        cursor = connection.execute(query, params)
        try:
            return cursor.fetchall()
        finally:
            cursor.close()

    def _ensure_normalized_columns(self) -> None:
        normalizer = getattr(self, "_normalizer", None)
        if normalizer is None:
            return
        try:
            normalizer.refresh_normalized_columns()
        except sqlite3.Error:
            pass

    def __del__(self) -> None:  # pragma: no cover - defensive cleanup
        try:
            self.close()
        except Exception:
            pass

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> Row:
        return {key: row[key] for key in row.keys()}

    def fetch_rare_combinations(self, source_word: str, limit: int) -> List[Row]:
        normalized_source = _normalize_text(source_word)
        if not normalized_source:
            return []

        query = """
            SELECT target_word, artist, song_title, confidence_score, cultural_significance
            FROM song_rhyme_patterns
            WHERE source_word_normalized = ?
              AND target_word IS NOT NULL
              AND source_word != target_word
              AND length(target_word) >= 4
              AND confidence_score >= 0.7
            ORDER BY confidence_score DESC, phonetic_similarity DESC
            LIMIT ?
        """
        rows = self._execute(query, (normalized_source, max(1, limit)))
        return [self._row_to_dict(row) for row in rows]

    def fetch_phonological_challenges(self, source_word: str, limit: int) -> List[Row]:
        normalized_source = _normalize_text(source_word)
        if not normalized_source:
            return []

        query = """
            SELECT target_word, artist, song_title, confidence_score, phonetic_similarity
            FROM song_rhyme_patterns
            WHERE source_word_normalized = ?
              AND source_word != target_word
              AND confidence_score BETWEEN 0.7 AND 0.9
              AND phonetic_similarity >= 0.8
            ORDER BY phonetic_similarity DESC
            LIMIT ?
        """
        rows = self._execute(query, (normalized_source, max(1, limit)))
        return [self._row_to_dict(row) for row in rows]

    def fetch_cultural_depth_patterns(self, source_word: str, limit: int) -> List[Row]:
        normalized_source = _normalize_text(source_word)
        if not normalized_source:
            return []

        categories = self._get_cultural_significance_labels()
        conditions = [
            "source_word_normalized = ?",
            "source_word != target_word",
        ]
        params: List[Any] = [normalized_source]

        if categories:
            placeholders = ", ".join("?" for _ in categories)
            conditions.append(f"cultural_significance_normalized IN ({placeholders})")
            params.extend(categories)
        else:
            conditions.append("cultural_significance_normalized IS NOT NULL")

        query = f"""
            SELECT target_word, artist, song_title, confidence_score, cultural_significance
            FROM song_rhyme_patterns
            WHERE {' AND '.join(conditions)}
            ORDER BY confidence_score DESC
            LIMIT ?
        """
        params.append(max(1, limit))

        rows = self._execute(query, tuple(params))
        return [self._row_to_dict(row) for row in rows]

    def _get_cultural_significance_labels(self) -> List[str]:
        if self._cultural_significance_labels is not None:
            return self._cultural_significance_labels

        query = """
            SELECT DISTINCT cultural_significance_normalized AS label
            FROM song_rhyme_patterns
            WHERE cultural_significance_normalized IS NOT NULL
            ORDER BY label
        """
        rows = self._execute(query, ())
        labels: List[str] = []
        for row in rows:
            value = str(row["label"]).strip()
            if value:
                labels.append(value)

        self._cultural_significance_labels = labels
        return labels

    def fetch_complex_syllable_patterns(self, source_word: str, limit: int) -> List[Row]:
        normalized_source = _normalize_text(source_word)
        if not normalized_source:
            return []

        query = """
            SELECT target_word, artist, song_title, confidence_score
            FROM song_rhyme_patterns
            WHERE source_word_normalized = ?
              AND source_word != target_word
              AND length(target_word) >= 6
              AND confidence_score >= 0.75
            ORDER BY length(target_word) DESC
            LIMIT ?
        """
        rows = self._execute(query, (normalized_source, max(1, limit)))
        return [self._row_to_dict(row) for row in rows]

    def fetch_seed_neighbors(self, seed_word: str, limit: int) -> List[Row]:
        normalized_seed = _normalize_text(seed_word)
        if not normalized_seed:
            return []

        results: List[Row] = []
        seen: Set[str] = set()

        query_source = """
            SELECT target_word, artist, song_title, confidence_score, phonetic_similarity, cultural_significance
            FROM song_rhyme_patterns
            WHERE source_word_normalized = ?
              AND target_word_normalized IS NOT NULL
              AND target_word_normalized != ?
            ORDER BY confidence_score DESC
            LIMIT ?
        """
        for row in self._execute(query_source, (normalized_seed, normalized_seed, max(1, limit))):
            candidate = str(row["target_word"]).strip()
            lowered = _normalize_text(candidate)
            if not candidate or lowered in seen:
                continue
            seen.add(lowered)
            results.append(
                {
                    "candidate": candidate,
                    "confidence": row["confidence_score"],
                    "phonetic_similarity": row["phonetic_similarity"],
                    "cultural_sig": row["cultural_significance"],
                    "context": f"{row['artist']} - {row['song_title']}",
                }
            )
            if len(results) >= limit:
                return results

        query_target = """
            SELECT source_word, artist, song_title, confidence_score, phonetic_similarity, cultural_significance
            FROM song_rhyme_patterns
            WHERE target_word_normalized = ?
              AND source_word_normalized IS NOT NULL
              AND source_word_normalized != ?
            ORDER BY confidence_score DESC
            LIMIT ?
        """
        for row in self._execute(query_target, (normalized_seed, normalized_seed, max(1, limit))):
            candidate = str(row["source_word"]).strip()
            lowered = _normalize_text(candidate)
            if not candidate or lowered in seen:
                continue
            seen.add(lowered)
            results.append(
                {
                    "candidate": candidate,
                    "confidence": row["confidence_score"],
                    "phonetic_similarity": row["phonetic_similarity"],
                    "cultural_sig": row["cultural_significance"],
                    "context": f"{row['artist']} - {row['song_title']}",
                }
            )
            if len(results) >= limit:
                break

        return results

    def fetch_suffix_matches(self, suffix: str, limit: int) -> List[Row]:
        normalized_suffix = _normalize_text(suffix)
        if not normalized_suffix:
            return []

        like_pattern = f"%{normalized_suffix}"
        results: List[Row] = []
        seen: Set[str] = set()

        for column, normalized_column in (
            ("target_word", "target_word_normalized"),
            ("source_word", "source_word_normalized"),
        ):
            query = f"""
                SELECT {column} as candidate, artist, song_title, confidence_score, phonetic_similarity, cultural_significance
                FROM song_rhyme_patterns
                WHERE {normalized_column} LIKE ?
                ORDER BY confidence_score DESC
                LIMIT ?
            """
            for row in self._execute(query, (like_pattern, max(1, limit))):
                candidate = str(row["candidate"]).strip()
                lowered = _normalize_text(candidate)
                if not candidate or lowered in seen:
                    continue
                seen.add(lowered)
                results.append(
                    {
                        "candidate": candidate,
                        "confidence": row["confidence_score"],
                        "phonetic_similarity": row["phonetic_similarity"],
                        "cultural_sig": row["cultural_significance"],
                        "context": f"{row['artist']} - {row['song_title']}",
                    }
                )
                if len(results) >= limit:
                    break
            if len(results) >= limit:
                break

        return results


__all__ = ["PatternRepository", "SQLitePatternRepository"]
