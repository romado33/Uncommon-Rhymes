"""Database utilities and repositories for the RhymeRarity application."""

from __future__ import annotations

import os
import queue
import sqlite3
import threading
from contextlib import contextmanager
from typing import Dict, Generator, Iterable, List, Optional, Sequence, Set, Tuple

from rhyme_rarity.utils.observability import get_logger

from demo_data import DEMO_RHYME_PATTERNS, iter_demo_rhyme_rows


def _ensure_parent_directory(path: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


_GENRE_NORMALIZED_EXPR = (
    "CASE "
    "WHEN genre IS NULL OR TRIM(genre) = '' THEN NULL "
    "ELSE LOWER(TRIM(genre)) "
    "END"
)

_CULTURAL_NORMALIZED_EXPR = (
    "CASE "
    "WHEN cultural_significance IS NULL OR TRIM(cultural_significance) = '' THEN NULL "
    "ELSE REPLACE(LOWER(TRIM(cultural_significance)), '_', '-') "
    "END"
)


def _normalized_text_expr(column: str) -> str:
    return (
        "CASE "
        "WHEN {column} IS NULL OR TRIM({column}) = '' THEN NULL "
        "ELSE LOWER(TRIM({column})) "
        "END"
    ).format(column=column)


def _normalise_text(value: str) -> Optional[str]:
    cleaned = str(value or "").strip()
    if not cleaned:
        return None
    return cleaned.lower()


def _normalise_genre(value: str) -> Optional[str]:
    return _normalise_text(value)


def _normalise_cultural_significance(value: str) -> Optional[str]:
    cleaned = str(value or "").strip()
    if not cleaned:
        return None
    return cleaned.lower().replace("_", "-")


class SQLiteRhymeRepository:
    """Repository encapsulating all SQLite access for rhyme searches."""

    def __init__(
        self,
        db_path: str,
        *,
        pool_size: int = 4,
        pool_timeout: float = 5.0,
    ) -> None:
        self.db_path = db_path
        self._pool_size = max(1, int(pool_size))
        self._pool_timeout = max(0.0, float(pool_timeout))
        self._pool: queue.Queue = queue.Queue(maxsize=self._pool_size)
        self._pool_semaphore = threading.BoundedSemaphore(self._pool_size)
        self._logger = get_logger(__name__).bind(
            component="sqlite_repository",
            db_path=db_path,
        )
        self._logger.info(
            "SQLite repository initialised",
            context={"pool_size": self._pool_size, "pool_timeout": self._pool_timeout},
        )

    def _create_connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, check_same_thread=False)
        try:
            connection.execute("PRAGMA journal_mode=WAL")
        except sqlite3.Error as exc:
            # WAL mode is best-effort – log but continue if unsupported.
            self._logger.warning(
                "SQLite WAL mode unavailable",
                context={"error": str(exc)},
            )
        return connection

    def _acquire_connection(self) -> sqlite3.Connection:
        if not self._pool_semaphore.acquire(timeout=self._pool_timeout or None):
            self._logger.error(
                "Database connection pool exhausted",
                context={"pool_size": self._pool_size, "timeout": self._pool_timeout},
            )
            raise TimeoutError("Database connection pool exhausted")

        try:
            connection = self._pool.get_nowait()
        except queue.Empty:
            connection = self._create_connection()

        return connection

    def _release_connection(self, connection: sqlite3.Connection) -> None:
        try:
            self._pool.put_nowait(connection)
        except queue.Full:
            connection.close()
        finally:
            self._pool_semaphore.release()

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        connection = self._acquire_connection()
        try:
            yield connection
            if connection.in_transaction:
                connection.commit()
        except Exception as exc:
            if connection.in_transaction:
                connection.rollback()
            self._logger.error(
                "SQLite operation failed",
                context={"error": str(exc)},
            )
            raise
        finally:
            self._release_connection(connection)

    def ensure_database(self) -> int:
        """Ensure the database exists and contains the expected schema."""

        self._logger.info("Ensuring database availability")
        if not os.path.exists(self.db_path):
            self._logger.info(
                "Database file missing; creating demo database",
                context={"db_path": self.db_path},
            )
            return self._create_demo_database()

        try:
            with self._connect() as conn:
                self._ensure_schema_extensions(conn)
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM song_rhyme_patterns")
                (count,) = cursor.fetchone()
                row_count = int(count)
                self._logger.info(
                    "Database schema verified",
                    context={"row_count": row_count},
                )
                return row_count
        except sqlite3.Error as exc:
            self._logger.warning(
                "Database verification failed; recreating demo",
                context={"error": str(exc)},
            )
            return self._create_demo_database(overwrite=True)

    def _initialise_schema(self, connection: sqlite3.Connection) -> None:
        cursor = connection.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS song_rhyme_patterns (
                id INTEGER PRIMARY KEY,
                pattern TEXT,
                source_word TEXT,
                source_word_normalized TEXT,
                target_word TEXT,
                target_word_normalized TEXT,
                artist TEXT,
                artist_normalized TEXT,
                song_title TEXT,
                release_year INTEGER,
                genre TEXT,
                line_distance INTEGER,
                confidence_score REAL,
                phonetic_similarity REAL,
                cultural_significance TEXT,
                source_context TEXT,
                target_context TEXT,
                lyrical_context TEXT,
                genre_normalized TEXT,
                cultural_significance_normalized TEXT
            )
            """
        )
        self._ensure_schema_extensions(connection)

    def _ensure_schema_extensions(self, connection: sqlite3.Connection) -> None:
        cursor = connection.cursor()
        cursor.execute("PRAGMA table_info(song_rhyme_patterns)")
        existing_columns = {row[1] for row in cursor.fetchall()}

        if "genre_normalized" not in existing_columns:
            cursor.execute(
                """
                ALTER TABLE song_rhyme_patterns
                ADD COLUMN genre_normalized TEXT
                """
            )

        if "cultural_significance_normalized" not in existing_columns:
            cursor.execute(
                """
                ALTER TABLE song_rhyme_patterns
                ADD COLUMN cultural_significance_normalized TEXT
                """
            )

        if "release_year" not in existing_columns:
            cursor.execute(
                """
                ALTER TABLE song_rhyme_patterns
                ADD COLUMN release_year INTEGER
                """
            )

        if "lyrical_context" not in existing_columns:
            cursor.execute(
                """
                ALTER TABLE song_rhyme_patterns
                ADD COLUMN lyrical_context TEXT
                """
            )

        if "source_word_normalized" not in existing_columns:
            cursor.execute(
                """
                ALTER TABLE song_rhyme_patterns
                ADD COLUMN source_word_normalized TEXT
                """
            )

        if "target_word_normalized" not in existing_columns:
            cursor.execute(
                """
                ALTER TABLE song_rhyme_patterns
                ADD COLUMN target_word_normalized TEXT
                """
            )

        if "artist_normalized" not in existing_columns:
            cursor.execute(
                """
                ALTER TABLE song_rhyme_patterns
                ADD COLUMN artist_normalized TEXT
                """
            )

        self._refresh_normalized_columns(connection)

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_rhyme_source_lookup
            ON song_rhyme_patterns (
                source_word_normalized,
                confidence_score DESC,
                phonetic_similarity DESC,
                target_word_normalized,
                target_word,
                artist
            )
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_rhyme_target_lookup
            ON song_rhyme_patterns (
                target_word_normalized,
                confidence_score DESC,
                phonetic_similarity DESC,
                source_word_normalized,
                source_word,
                artist
            )
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_rhyme_genre_normalized
            ON song_rhyme_patterns (genre_normalized)
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_rhyme_cultural_normalized
            ON song_rhyme_patterns (cultural_significance_normalized)
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_rhyme_artist_normalized
            ON song_rhyme_patterns (
                artist_normalized,
                confidence_score DESC,
                source_word_normalized,
                target_word_normalized
            )
            """
        )
        connection.commit()

    def _refresh_normalized_columns(self, connection: sqlite3.Connection) -> None:
        cursor = connection.cursor()
        cursor.execute(
            f"""
            UPDATE song_rhyme_patterns
            SET genre_normalized = {_GENRE_NORMALIZED_EXPR}
            WHERE COALESCE(genre_normalized, '') != COALESCE({_GENRE_NORMALIZED_EXPR}, '')
            """
        )
        cursor.execute(
            f"""
            UPDATE song_rhyme_patterns
            SET cultural_significance_normalized = {_CULTURAL_NORMALIZED_EXPR}
            WHERE COALESCE(cultural_significance_normalized, '') != COALESCE({_CULTURAL_NORMALIZED_EXPR}, '')
            """
        )
        cursor.execute(
            f"""
            UPDATE song_rhyme_patterns
            SET source_word_normalized = {_normalized_text_expr('source_word')}
            WHERE COALESCE(source_word_normalized, '') != COALESCE({_normalized_text_expr('source_word')}, '')
            """
        )
        cursor.execute(
            f"""
            UPDATE song_rhyme_patterns
            SET target_word_normalized = {_normalized_text_expr('target_word')}
            WHERE COALESCE(target_word_normalized, '') != COALESCE({_normalized_text_expr('target_word')}, '')
            """
        )
        cursor.execute(
            f"""
            UPDATE song_rhyme_patterns
            SET artist_normalized = {_normalized_text_expr('artist')}
            WHERE COALESCE(artist_normalized, '') != COALESCE({_normalized_text_expr('artist')}, '')
            """
        )

    def _create_demo_database(self, overwrite: bool = False) -> int:
        if overwrite and os.path.exists(self.db_path):
            os.remove(self.db_path)

        _ensure_parent_directory(self.db_path)

        self._logger.info(
            "Seeding demo database",
            context={"overwrite": overwrite},
        )

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
                    release_year,
                    genre,
                    line_distance,
                    confidence_score,
                    phonetic_similarity,
                    cultural_significance,
                    source_context,
                    target_context,
                    lyrical_context
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                iter_demo_rhyme_rows(),
            )
            self._refresh_normalized_columns(conn)
            conn.commit()

        row_count = len(DEMO_RHYME_PATTERNS)
        self._logger.info(
            "Demo database seeded",
            context={"rows": row_count},
        )
        return row_count

    def fetch_related_words(self, lookup_terms: Iterable[str]) -> Set[str]:
        terms = {term for term in (_normalise_text(t) for t in lookup_terms) if term}
        if not terms:
            return set()

        normalized_terms = sorted(term for term in terms if term)
        if not normalized_terms:
            return set()

        related: Set[str] = set()
        placeholders = ",".join("?" for _ in normalized_terms)
        limit_hint = 50 * len(normalized_terms)

        self._logger.debug(
            "Fetching related words",
            context={"term_count": len(normalized_terms)},
        )

        with self._connect() as conn:
            cursor = conn.cursor()

            cursor.execute(
                f"""
                SELECT target_word
                FROM song_rhyme_patterns
                WHERE source_word_normalized IN ({placeholders})
                  AND target_word_normalized IS NOT NULL
                LIMIT ?
                """,
                (*normalized_terms, limit_hint),
            )
            for row in cursor.fetchall():
                if not row or not row[0]:
                    continue
                normalized = _normalise_text(row[0])
                if normalized:
                    related.add(normalized)

            cursor.execute(
                f"""
                SELECT source_word
                FROM song_rhyme_patterns
                WHERE target_word_normalized IN ({placeholders})
                  AND source_word_normalized IS NOT NULL
                LIMIT ?
                """,
                (*normalized_terms, limit_hint),
            )
            for row in cursor.fetchall():
                if not row or not row[0]:
                    continue
                normalized = _normalise_text(row[0])
                if normalized:
                    related.add(normalized)

        self._logger.info(
            "Related words fetched",
            context={
                "term_count": len(normalized_terms),
                "related_count": len(related),
            },
        )

        return related

    def fetch_phrases_for_word(
        self,
        end_word: str,
        *,
        limit: int = 40,
        min_tokens: int = 2,
        max_tokens: int = 5,
        rhyme_backoffs: Sequence[str] = (),
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Return phrase snippets whose terminal word matches ``end_word``."""

        normalized_end = _normalise_text(end_word)
        if normalized_end is None:
            return []

        phrases: List[Tuple[str, Dict[str, Any]]] = []
        seen: Set[str] = set()

        def _register(value: Optional[str], source: str, context_meta: Optional[Dict[str, Any]] = None) -> None:
            if not value:
                return
            cleaned = " ".join(part for part in value.split() if part).strip()
            if not cleaned:
                return
            normalized = _normalise_text(cleaned)
            if not normalized or normalized in seen:
                return
            tokens = normalized.split()
            if not tokens or tokens[-1] != normalized_end:
                return
            if not (min_tokens <= len(tokens) <= max_tokens):
                return
            seen.add(normalized)
            metadata: Dict[str, Any] = {"source": source}
            if context_meta:
                for key, value in context_meta.items():
                    if key not in metadata:
                        metadata[key] = value
            phrases.append((cleaned, metadata))

        with self._connect() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    """
                    SELECT source_context, target_context, lyrical_context,
                           source_word_normalized, target_word_normalized,
                           confidence_score
                    FROM song_rhyme_patterns
                    WHERE target_word_normalized = ? OR source_word_normalized = ?
                    ORDER BY confidence_score DESC
                    LIMIT ?
                    """,
                    (normalized_end, normalized_end, limit * 2),
                )
            except sqlite3.Error:
                return []

            rows = cursor.fetchall()

        for row in rows:
            if not row:
                continue
            (
                source_context,
                target_context,
                lyrical_context,
                source_norm,
                target_norm,
                confidence,
            ) = row

            context_meta = {
                "confidence": float(confidence) if confidence is not None else None,
                "row_source": "target" if target_norm == normalized_end else "source",
            }

            _register(target_context, "db_target", context_meta)
            _register(source_context, "db_source", context_meta)

            if lyrical_context:
                parts = [part.strip() for part in str(lyrical_context).split("//") if part]
                for part in parts:
                    _register(part, "db_lyric", context_meta)

        return phrases[:limit] if limit > 0 else phrases

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

        normalised_source_word = _normalise_text(source_word)
        if normalised_source_word is None:
            return [], []

        base_query = """
            SELECT DISTINCT
                source_word,
                target_word,
                artist,
                song_title,
                release_year,
                pattern,
                genre,
                line_distance,
                confidence_score,
                phonetic_similarity,
                cultural_significance,
                source_context,
                target_context,
                lyrical_context
            FROM song_rhyme_patterns
        """

        normalised_cultural_filters = [
            value
            for value in (
                _normalise_cultural_significance(item) for item in cultural_filters
            )
            if value is not None
        ]
        normalised_genre_filters = [
            value for value in (_normalise_genre(item) for item in genre_filters) if value is not None
        ]

        def build_query(column: str) -> Tuple[str, List]:
            conditions = [
                f"{column} = ?",
                "confidence_score >= ?",
                "source_word != target_word",
            ]
            params: List = [normalised_source_word, min_confidence]

            if phonetic_threshold is not None:
                conditions.append("phonetic_similarity >= ?")
                params.append(phonetic_threshold)

            if normalised_cultural_filters:
                placeholders = ",".join("?" for _ in normalised_cultural_filters)
                conditions.append(
                    f"cultural_significance_normalized IN ({placeholders})"
                )
                params.extend(normalised_cultural_filters)

            if normalised_genre_filters:
                placeholders = ",".join("?" for _ in normalised_genre_filters)
                conditions.append(
                    "genre_normalized IN ({})".format(placeholders)
                )
                params.extend(normalised_genre_filters)

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
            self._refresh_normalized_columns(conn)
            conn.commit()
            cursor = conn.cursor()
            query, params = build_query("source_word_normalized")
            cursor.execute(query, params)
            source_rows = cursor.fetchall()

            query, params = build_query("target_word_normalized")
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

