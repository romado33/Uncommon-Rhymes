"""Utilities for working with the CMU pronouncing dictionary."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

_ALT_PATTERN = re.compile(r"\(\d+\)$")
_DIGIT_PATTERN = re.compile(r"\d")
_STRESS_PATTERN = re.compile(r"[12]")

VOWEL_PHONEMES: Set[str] = {
    "AA",
    "AE",
    "AH",
    "AO",
    "AW",
    "AY",
    "EH",
    "ER",
    "EY",
    "IH",
    "IY",
    "OW",
    "OY",
    "UH",
    "UW",
}


class CMUDictLoader:
    """Lazy loader for the CMU pronouncing dictionary."""

    def __init__(self, dict_path: Optional[Path | str] = None) -> None:
        if dict_path is not None:
            base_path = Path(dict_path)
        else:
            module_path = Path(__file__).resolve()
            candidates = [
                module_path.with_name("cmudict.7b"),
                module_path.parents[1] / "cmudict.7b",
                module_path.parents[2] / "cmudict.7b",
            ]
            base_path = candidates[0]
            for candidate in candidates[1:]:
                try:
                    if candidate.exists():
                        base_path = candidate
                        break
                except OSError:
                    continue
        self.dict_path: Path = base_path
        self._pronunciations: Dict[str, Tuple[Tuple[str, ...], ...]] = {}
        self._rhyme_parts: Dict[str, FrozenSet[str]] = {}
        self._rhyme_index: Dict[str, Tuple[str, ...]] = {}
        self._loaded: bool = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return

        if not self.dict_path.exists():
            return

        pronunciations: Dict[str, List[Tuple[str, ...]]] = {}
        rhyme_parts: Dict[str, Set[str]] = {}
        rhyme_index: Dict[str, Set[str]] = {}

        try:
            with self.dict_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    entry = line.strip()
                    if not entry or entry.startswith(";;;"):
                        continue

                    parts = entry.split()
                    if len(parts) < 2:
                        continue

                    raw_word, *phones = parts
                    word = _ALT_PATTERN.sub("", raw_word).lower()
                    if not word:
                        continue

                    pronunciations.setdefault(word, []).append(tuple(phones))

                    rhyme_part = self._extract_rhyme_part(phones)
                    if not rhyme_part:
                        continue

                    rhyme_parts.setdefault(word, set()).add(rhyme_part)
                    rhyme_index.setdefault(rhyme_part, set()).add(word)
        except (OSError, UnicodeDecodeError):
            # If the dictionary cannot be read we simply operate without cache.
            return

        self._pronunciations = {
            word: tuple(variants) for word, variants in pronunciations.items()
        }
        self._rhyme_parts = {
            word: frozenset(parts) for word, parts in rhyme_parts.items()
        }
        self._rhyme_index = {
            part: tuple(sorted(words)) for part, words in rhyme_index.items()
        }
        self._loaded = True

    def _extract_rhyme_part(self, phones: List[str]) -> Optional[str]:
        """Return the final stressed vowel plus trailing phonemes."""

        last_stress_index: Optional[int] = None

        for index, phone in enumerate(phones):
            base = _DIGIT_PATTERN.sub("", phone)
            if base not in VOWEL_PHONEMES:
                continue

            if _STRESS_PATTERN.search(phone):
                last_stress_index = index

        if last_stress_index is None:
            for index in range(len(phones) - 1, -1, -1):
                base = _DIGIT_PATTERN.sub("", phones[index])
                if base in VOWEL_PHONEMES:
                    last_stress_index = index
                    break

        if last_stress_index is None:
            return None

        return " ".join(phones[last_stress_index:])

    def get_pronunciations(self, word: str) -> List[List[str]]:
        self._ensure_loaded()
        variants = self._pronunciations.get(word.lower(), tuple())
        return [list(variant) for variant in variants]

    def get_rhyme_parts(self, word: str) -> Set[str]:
        self._ensure_loaded()
        return set(self._rhyme_parts.get(word.lower(), frozenset()))

    def get_rhyming_words(self, word: str) -> List[str]:
        self._ensure_loaded()
        normalized = word.lower()
        rhyme_parts = self._rhyme_parts.get(normalized)
        if not rhyme_parts:
            return []

        candidates: Set[str] = set()
        for part in rhyme_parts:
            candidates.update(self._rhyme_index.get(part, tuple()))

        candidates.discard(normalized)
        return sorted(candidates)


DEFAULT_CMU_LOADER = CMUDictLoader()

__all__ = [
    "CMUDictLoader",
    "DEFAULT_CMU_LOADER",
    "VOWEL_PHONEMES",
]
