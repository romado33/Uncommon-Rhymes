"""Data access helpers for the anti-LLM rhyme engine."""

from __future__ import annotations

import sqlite3
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

    def _execute(self, query: str, params: Sequence[Any]) -> List[sqlite3.Row]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor.fetchall()

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> Row:
        return {key: row[key] for key in row.keys()}

    def fetch_rare_combinations(self, source_word: str, limit: int) -> List[Row]:
        query = """
            SELECT target_word, artist, song_title, confidence_score, cultural_significance
            FROM song_rhyme_patterns
            WHERE source_word = ?
              AND source_word != target_word
              AND length(target_word) >= 4
              AND confidence_score >= 0.7
            ORDER BY RANDOM()
            LIMIT ?
        """
        rows = self._execute(query, (source_word, max(1, limit)))
        return [self._row_to_dict(row) for row in rows]

    def fetch_phonological_challenges(self, source_word: str, limit: int) -> List[Row]:
        query = """
            SELECT target_word, artist, song_title, confidence_score, phonetic_similarity
            FROM song_rhyme_patterns
            WHERE source_word = ?
              AND source_word != target_word
              AND confidence_score BETWEEN 0.7 AND 0.9
              AND phonetic_similarity >= 0.8
            ORDER BY phonetic_similarity DESC
            LIMIT ?
        """
        rows = self._execute(query, (source_word, max(1, limit)))
        return [self._row_to_dict(row) for row in rows]

    def fetch_cultural_depth_patterns(self, source_word: str, limit: int) -> List[Row]:
        categories = self._get_cultural_significance_labels()
        conditions = [
            "source_word = ?",
            "source_word != target_word",
        ]
        params: List[Any] = [source_word]

        if categories:
            placeholders = ", ".join("?" for _ in categories)
            conditions.append(f"TRIM(cultural_significance) IN ({placeholders})")
            params.extend(categories)
        else:
            conditions.append("TRIM(COALESCE(cultural_significance, '')) != ''")

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
            SELECT DISTINCT TRIM(cultural_significance) AS label
            FROM song_rhyme_patterns
            WHERE cultural_significance IS NOT NULL
              AND TRIM(cultural_significance) != ''
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
        query = """
            SELECT target_word, artist, song_title, confidence_score
            FROM song_rhyme_patterns
            WHERE source_word = ?
              AND source_word != target_word
              AND length(target_word) >= 6
              AND confidence_score >= 0.75
            ORDER BY length(target_word) DESC
            LIMIT ?
        """
        rows = self._execute(query, (source_word, max(1, limit)))
        return [self._row_to_dict(row) for row in rows]

    def fetch_seed_neighbors(self, seed_word: str, limit: int) -> List[Row]:
        if not seed_word:
            return []

        results: List[Row] = []
        seen: Set[str] = set()

        query_source = """
            SELECT target_word, artist, song_title, confidence_score, phonetic_similarity, cultural_significance
            FROM song_rhyme_patterns
            WHERE LOWER(source_word) = ?
              AND target_word IS NOT NULL
              AND LOWER(target_word) != ?
            ORDER BY confidence_score DESC
            LIMIT ?
        """
        for row in self._execute(query_source, (seed_word.lower(), seed_word.lower(), max(1, limit))):
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
            WHERE LOWER(target_word) = ?
              AND source_word IS NOT NULL
              AND LOWER(source_word) != ?
            ORDER BY confidence_score DESC
            LIMIT ?
        """
        for row in self._execute(query_target, (seed_word.lower(), seed_word.lower(), max(1, limit))):
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

        like_pattern = f"%{suffix}"
        results: List[Row] = []
        seen: Set[str] = set()

        for column in ("target_word", "source_word"):
            query = f"""
                SELECT {column} as candidate, artist, song_title, confidence_score, phonetic_similarity, cultural_significance
                FROM song_rhyme_patterns
                WHERE LOWER({column}) LIKE ?
                ORDER BY confidence_score DESC
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
