"""Database utilities and repositories for the RhymeRarity application."""

from __future__ import annotations

import logging
import os
import sqlite3
from contextlib import contextmanager
from typing import Generator, Iterable, List, Optional, Sequence, Set, Tuple

from demo_data import DEMO_RHYME_PATTERNS, iter_demo_rhyme_rows
from rhyme_rarity.utils.observability import record_database_initialization


logger = logging.getLogger(__name__)


def _ensure_parent_directory(path: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


class SQLiteRhymeRepository:
    """Repository encapsulating all SQLite access for rhyme searches."""

    def __init__(self, db_path: str):
        self.db_path = db_path

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        connection = sqlite3.connect(self.db_path)
        try:
            yield connection
        finally:
            connection.close()

    def ensure_database(self) -> int:
        """Ensure the database exists and contains the expected schema."""

        if not os.path.exists(self.db_path):
            logger.warning(
                "database.missing", extra={"db_path": os.path.abspath(self.db_path)}
            )
            record_database_initialization("demo_rebuild", reason="missing")
            return self._create_demo_database()

        try:
            with self._connect() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM song_rhyme_patterns")
                (count,) = cursor.fetchone()
                logger.info(
                    "database.ready",
                    extra={
                        "db_path": os.path.abspath(self.db_path),
                        "rows": int(count),
                    },
                )
                record_database_initialization("existing", rows=int(count))
                return int(count)
        except sqlite3.Error:
            logger.exception(
                "database.corrupt",
                extra={"db_path": os.path.abspath(self.db_path)},
            )
            record_database_initialization("demo_rebuild", reason="sqlite_error")
            return self._create_demo_database(overwrite=True)

    def _initialise_schema(self, connection: sqlite3.Connection) -> None:
        cursor = connection.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS song_rhyme_patterns (
                id INTEGER PRIMARY KEY,
                pattern TEXT,
                source_word TEXT,
                target_word TEXT,
                artist TEXT,
                song_title TEXT,
                genre TEXT,
                line_distance INTEGER,
                confidence_score REAL,
                phonetic_similarity REAL,
                cultural_significance TEXT,
                source_context TEXT,
                target_context TEXT
            )
            """
        )

    def _create_demo_database(self, overwrite: bool = False) -> int:
        if overwrite and os.path.exists(self.db_path):
            logger.warning(
                "database.overwrite", extra={"db_path": os.path.abspath(self.db_path)}
            )
            os.remove(self.db_path)

        _ensure_parent_directory(self.db_path)

        with self._connect() as conn:
            self._initialise_schema(conn)

            cursor = conn.cursor()
            cursor.execute("DELETE FROM song_rhyme_patterns")
            cursor.executemany(
                """
                INSERT INTO song_rhyme_patterns (
                    pattern,
                    source_word,
                    target_word,
                    artist,
                    song_title,
                    genre,
                    line_distance,
                    confidence_score,
                    phonetic_similarity,
                    cultural_significance,
                    source_context,
                    target_context
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                iter_demo_rhyme_rows(),
            )
            conn.commit()

        record_count = len(DEMO_RHYME_PATTERNS)
        logger.info(
            "database.demo_built",
            extra={
                "db_path": os.path.abspath(self.db_path),
                "rows": record_count,
                "overwrite": overwrite,
            },
        )
        return record_count

    def fetch_related_words(self, lookup_terms: Iterable[str]) -> Set[str]:
        terms = {str(term or "").strip().lower() for term in lookup_terms if term}
        if not terms:
            return set()

        related: Set[str] = set()

        with self._connect() as conn:
            cursor = conn.cursor()
            for term in sorted(terms):
                if not term:
                    continue

                try:
                    cursor.execute(
                        """
                        SELECT target_word
                        FROM song_rhyme_patterns
                        WHERE source_word = ? AND target_word IS NOT NULL
                        LIMIT 50
                        """,
                        (term,),
                    )
                    related.update(
                        str(row[0]).strip().lower()
                        for row in cursor.fetchall()
                        if row and row[0]
                    )

                    cursor.execute(
                        """
                        SELECT source_word
                        FROM song_rhyme_patterns
                        WHERE target_word = ? AND source_word IS NOT NULL
                        LIMIT 50
                        """,
                        (term,),
                    )
                    related.update(
                        str(row[0]).strip().lower()
                        for row in cursor.fetchall()
                        if row and row[0]
                    )
                except sqlite3.Error:
                    logger.exception(
                        "database.related_lookup_failed",
                        extra={"db_path": os.path.abspath(self.db_path), "term": term},
                    )
                    continue

        return related

    def fetch_cultural_matches(
        self,
        source_word: str,
        *,
        min_confidence: float,
        phonetic_threshold: Optional[float],
        cultural_filters: Sequence[str],
        genre_filters: Sequence[str],
        max_line_distance: Optional[int],
        limit: Optional[int] = None,
    ) -> Tuple[List[Tuple], List[Tuple]]:
        """Return rhyme matches treating the word as both source and target."""

        base_query = """
            SELECT DISTINCT
                source_word,
                target_word,
                artist,
                song_title,
                pattern,
                genre,
                line_distance,
                confidence_score,
                phonetic_similarity,
                cultural_significance,
                source_context,
                target_context
            FROM song_rhyme_patterns
        """

        def build_query(column: str) -> Tuple[str, List]:
            conditions = [
                f"{column} = ?",
                "confidence_score >= ?",
                "source_word != target_word",
            ]
            params: List = [source_word, min_confidence]

            if phonetic_threshold is not None:
                conditions.append("phonetic_similarity >= ?")
                params.append(phonetic_threshold)

            if cultural_filters:
                placeholders = ",".join("?" for _ in cultural_filters)
                conditions.append(
                    f"REPLACE(LOWER(cultural_significance), '_', '-') IN ({placeholders})"
                )
                params.extend(cultural_filters)

            if genre_filters:
                placeholders = ",".join("?" for _ in genre_filters)
                conditions.append("LOWER(genre) IN ({})".format(placeholders))
                params.extend(genre_filters)

            if max_line_distance is not None:
                conditions.append("line_distance <= ?")
                params.append(max_line_distance)

            query = base_query + " WHERE " + " AND ".join(conditions)
            query += " ORDER BY confidence_score DESC, phonetic_similarity DESC"
            if limit is not None:
                query += " LIMIT ?"
                params.append(limit)
            return query, params

        with self._connect() as conn:
            cursor = conn.cursor()
            query, params = build_query("source_word")
            cursor.execute(query, params)
            source_rows = cursor.fetchall()

            query, params = build_query("target_word")
            cursor.execute(query, params)
            target_rows = cursor.fetchall()

        return source_rows, target_rows

    def get_cultural_significance_labels(self) -> List[str]:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT DISTINCT cultural_significance
                FROM song_rhyme_patterns
                WHERE cultural_significance IS NOT NULL
                """
            )
            return [row[0] for row in cursor.fetchall() if row and row[0]]

    def get_genres(self) -> List[str]:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT DISTINCT genre
                FROM song_rhyme_patterns
                WHERE genre IS NOT NULL
                ORDER BY genre
                """
            )
            return [row[0] for row in cursor.fetchall() if row and row[0]]

