"""Rarity scoring helpers shared across RhymeRarity modules."""

from __future__ import annotations

import math
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Optional, Set

from demo_data import build_demo_frequencies


def _normalize_word(value: Optional[str]) -> str:
    return value.lower().strip() if value else ""


class WordRarityMap:
    """Stores approximate frequency information for rhyme candidates."""

    _DEFAULT_FREQUENCIES: Dict[str, int] = build_demo_frequencies()

    def __init__(self, frequencies: Optional[Dict[str, int]] = None) -> None:
        self._frequencies: Counter[str] = Counter()
        self._loaded_sources: Set[Path] = set()
        self._default_frequency: int = 1
        if frequencies:
            self.add_frequencies(frequencies)
        if not self._frequencies:
            self.add_frequencies(self._DEFAULT_FREQUENCIES)
        self._update_frequency_stats()

    def add_frequencies(self, frequencies: Dict[str, int]) -> None:
        for word, value in frequencies.items():
            normalized = _normalize_word(word)
            if not normalized:
                continue
            try:
                freq_value = int(value)
            except (TypeError, ValueError):
                continue
            if freq_value <= 0:
                continue
            self._frequencies[normalized] = max(
                self._frequencies.get(normalized, 0), freq_value
            )
        self._update_frequency_stats()

    def _update_frequency_stats(self) -> None:
        if not self._frequencies:
            self._min_frequency = 0
            self._max_frequency = 0
            self._default_frequency = 1
            return

        frequencies = list(self._frequencies.values())
        self._min_frequency = min(frequencies)
        self._max_frequency = max(frequencies)
        self._default_frequency = max(
            1, int(sum(frequencies) / max(len(frequencies), 1))
        )

    def get_frequency(self, word: str) -> int:
        normalized = _normalize_word(word)
        if not normalized:
            return self._default_frequency
        return self._frequencies.get(normalized, self._default_frequency)

    def get_rarity(self, word: str) -> float:
        frequency = max(self.get_frequency(word), 1)
        rarity = 1.0 / (1.0 + math.log1p(frequency))
        return float(rarity)

    def update_from_database(self, db_path: Optional[Path | str]) -> bool:
        if not db_path:
            return False

        db_file = Path(db_path)
        if not db_file.exists():
            return False

        canonical = db_file.resolve()
        if canonical in self._loaded_sources:
            return True

        try:
            with sqlite3.connect(str(canonical)) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT target_word_normalized, COUNT(*)
                    FROM song_rhyme_patterns
                    WHERE target_word_normalized IS NOT NULL
                    GROUP BY target_word_normalized
                    """
                )
                rows = cursor.fetchall()
        except sqlite3.Error:
            return False

        frequency_map: Dict[str, int] = {
            row[0]: int(row[1])
            for row in rows
            if isinstance(row, Iterable) and row and row[0]
        }

        if not frequency_map:
            return False

        self.add_frequencies(frequency_map)
        self._loaded_sources.add(canonical)
        return True


DEFAULT_RARITY_MAP = WordRarityMap()

__all__ = ["WordRarityMap", "DEFAULT_RARITY_MAP"]
