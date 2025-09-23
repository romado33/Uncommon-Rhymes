"""Utilities for working with the CMU pronouncing dictionary."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Set

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
        self._pronunciations: Dict[str, List[List[str]]] = {}
        self._rhyme_parts: Dict[str, Set[str]] = {}
        self._rhyme_index: Dict[str, Set[str]] = {}
        self._loaded: bool = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return

        if not self.dict_path.exists():
            return

        pronunciations: Dict[str, List[List[str]]] = {}
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
                    word = re.sub(r"\(\d+\)$", "", raw_word).lower()
                    if not word:
                        continue

                    pronunciations.setdefault(word, []).append(phones)

                    rhyme_part = self._extract_rhyme_part(phones)
                    if not rhyme_part:
                        continue

                    rhyme_parts.setdefault(word, set()).add(rhyme_part)
                    rhyme_index.setdefault(rhyme_part, set()).add(word)
        except (OSError, UnicodeDecodeError):
            # If the dictionary cannot be read we simply operate without cache.
            return

        self._pronunciations = pronunciations
        self._rhyme_parts = rhyme_parts
        self._rhyme_index = rhyme_index
        self._loaded = True

    def _extract_rhyme_part(self, phones: List[str]) -> Optional[str]:
        """Return the final stressed vowel plus trailing phonemes."""

        last_stress_index: Optional[int] = None

        for index, phone in enumerate(phones):
            base = re.sub(r"\d", "", phone)
            if base not in VOWEL_PHONEMES:
                continue

            if re.search(r"[12]", phone):
                last_stress_index = index

        if last_stress_index is None:
            for index in range(len(phones) - 1, -1, -1):
                base = re.sub(r"\d", "", phones[index])
                if base in VOWEL_PHONEMES:
                    last_stress_index = index
                    break

        if last_stress_index is None:
            return None

        return " ".join(phones[last_stress_index:])

    def get_pronunciations(self, word: str) -> List[List[str]]:
        self._ensure_loaded()
        return list(self._pronunciations.get(word.lower(), []))

    def get_rhyme_parts(self, word: str) -> Set[str]:
        self._ensure_loaded()
        return set(self._rhyme_parts.get(word.lower(), set()))

    def get_rhyming_words(self, word: str) -> List[str]:
        self._ensure_loaded()
        normalized = word.lower()
        rhyme_parts = self._rhyme_parts.get(normalized)
        if not rhyme_parts:
            return []

        candidates: Set[str] = set()
        for part in rhyme_parts:
            candidates.update(self._rhyme_index.get(part, set()))

        candidates.discard(normalized)
        return sorted(candidates)


DEFAULT_CMU_LOADER = CMUDictLoader()

__all__ = [
    "CMUDictLoader",
    "DEFAULT_CMU_LOADER",
    "VOWEL_PHONEMES",
]
