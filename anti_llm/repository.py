"""Data access helpers for the anti-LLM rhyme engine."""

from __future__ import annotations

import sqlite3
import threading
from typing import Any, Dict, List, Optional, Protocol, Sequence, Set

Row = Dict[str, Any]


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
        self._local = threading.local()
        self._connections_lock = threading.Lock()
        self._connections: Set[sqlite3.Connection] = set()

    def _get_connection(self) -> sqlite3.Connection:
        connection = getattr(self._local, "connection", None)
        if connection is None:
            connection = sqlite3.connect(self.db_path, check_same_thread=False)
            connection.row_factory = sqlite3.Row
            self._local.connection = connection
            with self._connections_lock:
                self._connections.add(connection)
        return connection

    def close(self) -> None:
        with self._connections_lock:
            connections = tuple(self._connections)
            self._connections.clear()

        for connection in connections:
            try:
                connection.close()
            except Exception:
                pass

        if hasattr(self._local, "connection"):
            del self._local.connection

    def _execute(self, query: str, params: Sequence[Any]) -> List[sqlite3.Row]:
        connection = self._get_connection()
        cursor = connection.execute(query, params)
        try:
            return cursor.fetchall()
        finally:
            cursor.close()

    def __del__(self) -> None:  # pragma: no cover - defensive cleanup
        try:
            self.close()
        except Exception:
            pass

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> Row:
        return {key: row[key] for key in row.keys()}

    @staticmethod
    def _normalize_term(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = str(value).strip().lower()
        return normalized or None

    def fetch_rare_combinations(self, source_word: str, limit: int) -> List[Row]:
        normalized_source = self._normalize_term(source_word)
        if not normalized_source:
            return []

        query = """
            SELECT target_word, artist, song_title, confidence_score, cultural_significance
            FROM song_rhyme_patterns
            WHERE source_word_normalized = ?
              AND target_word_normalized IS NOT NULL
              AND target_word_normalized != ?
              AND length(target_word) >= 4
              AND confidence_score >= 0.7
            ORDER BY confidence_score DESC, phonetic_similarity DESC, target_word_normalized
            LIMIT ?
        """
        rows = self._execute(
            query,
            (normalized_source, normalized_source, max(1, limit)),
        )
        return [self._row_to_dict(row) for row in rows]

    def fetch_phonological_challenges(self, source_word: str, limit: int) -> List[Row]:
        normalized_source = self._normalize_term(source_word)
        if not normalized_source:
            return []

        query = """
            SELECT target_word, artist, song_title, confidence_score, phonetic_similarity
            FROM song_rhyme_patterns
            WHERE source_word_normalized = ?
              AND target_word_normalized IS NOT NULL
              AND target_word_normalized != ?
              AND confidence_score BETWEEN 0.7 AND 0.9
              AND phonetic_similarity >= 0.8
            ORDER BY phonetic_similarity DESC, confidence_score DESC, target_word_normalized
            LIMIT ?
        """
        rows = self._execute(
            query,
            (normalized_source, normalized_source, max(1, limit)),
        )
        return [self._row_to_dict(row) for row in rows]

    def fetch_cultural_depth_patterns(self, source_word: str, limit: int) -> List[Row]:
        normalized_source = self._normalize_term(source_word)
        if not normalized_source:
            return []

        categories = self._get_cultural_significance_labels()
        conditions = [
            "source_word_normalized = ?",
            "target_word_normalized IS NOT NULL",
            "target_word_normalized != ?",
        ]
        params: List[Any] = [normalized_source, normalized_source]

        if categories:
            placeholders = ", ".join("?" for _ in categories)
            conditions.append(
                f"cultural_significance_normalized IN ({placeholders})"
            )
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
        normalized_source = self._normalize_term(source_word)
        if not normalized_source:
            return []

        query = """
            SELECT target_word, artist, song_title, confidence_score
            FROM song_rhyme_patterns
            WHERE source_word_normalized = ?
              AND target_word_normalized IS NOT NULL
              AND target_word_normalized != ?
              AND length(target_word) >= 6
              AND confidence_score >= 0.75
            ORDER BY length(target_word) DESC, confidence_score DESC
            LIMIT ?
        """
        rows = self._execute(
            query,
            (normalized_source, normalized_source, max(1, limit)),
        )
        return [self._row_to_dict(row) for row in rows]

    def fetch_seed_neighbors(self, seed_word: str, limit: int) -> List[Row]:
        normalized_seed = self._normalize_term(seed_word)
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
            ORDER BY confidence_score DESC, phonetic_similarity DESC, target_word_normalized
            LIMIT ?
        """
        for row in self._execute(
            query_source,
            (normalized_seed, normalized_seed, max(1, limit)),
        ):
            candidate = str(row["target_word"]).strip()
            lowered = candidate.lower()
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
            ORDER BY confidence_score DESC, phonetic_similarity DESC, source_word_normalized
            LIMIT ?
        """
        for row in self._execute(
            query_target,
            (normalized_seed, normalized_seed, max(1, limit)),
        ):
            candidate = str(row["source_word"]).strip()
            lowered = candidate.lower()
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
        if not suffix:
            return []

        normalized_suffix = self._normalize_term(suffix)
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
                ORDER BY confidence_score DESC, phonetic_similarity DESC, {normalized_column}
                LIMIT ?
            """
            for row in self._execute(query, (like_pattern, max(1, limit))):
                candidate = str(row["candidate"]).strip()
                lowered = candidate.lower()
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
