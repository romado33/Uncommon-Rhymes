"""Utilities for working with the CMU pronouncing dictionary."""

from __future__ import annotations

import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

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

_WORD_VARIANT_PATTERN = re.compile(r"\(\d+\)$")
_DIGIT_PATTERN = re.compile(r"\d")
_STRESS_PATTERN = re.compile(r"[12]")


def _strip_variant(word: str) -> str:
    return _WORD_VARIANT_PATTERN.sub("", word).lower()


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
        self._rhyme_parts: Dict[str, frozenset[str]] = {}
        self._rhyme_index: Dict[str, frozenset[str]] = {}
        self._phoneme_index: Dict[Tuple[str, ...], Tuple[str, ...]] = {}
        self._loaded: bool = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return

        if not self.dict_path.exists():
            return

        pronunciations: Dict[str, List[Tuple[str, ...]]] = {}
        rhyme_parts: Dict[str, Set[str]] = {}
        rhyme_index: Dict[str, Set[str]] = {}
        phoneme_index: Dict[Tuple[str, ...], Set[str]] = defaultdict(set)

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
                    word = _strip_variant(raw_word)
                    if not word:
                        continue

                    phone_tuple = tuple(phones)
                    pronunciations.setdefault(word, []).append(phone_tuple)

                    rhyme_part = self._extract_rhyme_part(phones)
                    if not rhyme_part:
                        continue

                    rhyme_parts.setdefault(word, set()).add(rhyme_part)
                    rhyme_index.setdefault(rhyme_part, set()).add(word)

                    normalized_tuple = tuple(
                        _DIGIT_PATTERN.sub("", phone) for phone in phone_tuple if phone
                    )
                    if normalized_tuple:
                        phoneme_index[normalized_tuple].add(word)
        except (OSError, UnicodeDecodeError):
            # If the dictionary cannot be read we simply operate without cache.
            return

        self._pronunciations = {
            word: tuple(entries) for word, entries in pronunciations.items()
        }
        self._rhyme_parts = {
            word: frozenset(parts) for word, parts in rhyme_parts.items()
        }
        self._rhyme_index = {
            part: frozenset(words) for part, words in rhyme_index.items()
        }
        self._phoneme_index = {
            key: tuple(sorted(values)) for key, values in phoneme_index.items()
        }
        self._loaded = True

    def _extract_rhyme_part(self, phones: Iterable[str]) -> Optional[str]:
        """Return the final stressed vowel plus trailing phonemes."""

        last_stress_index: Optional[int] = None

        phone_list = list(phones)

        for index, phone in enumerate(phone_list):
            base = _DIGIT_PATTERN.sub("", phone)
            if base not in VOWEL_PHONEMES:
                continue

            if _STRESS_PATTERN.search(phone):
                last_stress_index = index

        if last_stress_index is None:
            for index in range(len(phone_list) - 1, -1, -1):
                base = _DIGIT_PATTERN.sub("", phone_list[index])
                if base in VOWEL_PHONEMES:
                    last_stress_index = index
                    break

        if last_stress_index is None:
            return None

        return " ".join(phone_list[last_stress_index:])

    def get_pronunciations(self, word: str) -> List[List[str]]:
        self._ensure_loaded()
        stored = self._pronunciations.get(word.lower(), ())
        return [list(entry) for entry in stored]

    def get_rhyme_parts(self, word: str) -> Set[str]:
        self._ensure_loaded()
        stored = self._rhyme_parts.get(word.lower())
        return set(stored) if stored is not None else set()

    def get_rhyming_words(self, word: str) -> List[str]:
        self._ensure_loaded()
        normalized = word.lower()
        rhyme_parts = self._rhyme_parts.get(normalized)
        if not rhyme_parts:
            return []

        candidates: Set[str] = set()
        for part in rhyme_parts:
            candidates.update(self._rhyme_index.get(part, frozenset()))

        candidates.discard(normalized)
        return sorted(candidates)

    def find_words_by_phonemes(
        self,
        phones: Sequence[str],
        *,
        limit: Optional[int] = None,
        prefer_short: bool = True,
    ) -> List[str]:
        """Return words matching the provided CMU phoneme sequence.

        Args:
            phones: Iterable of phoneme strings that should match a full word
                pronunciation. Stress markers are ignored so callers may supply
                either stressed or unstressed variants.
            limit: Optional maximum number of words to return. ``None`` returns
                all matches, while ``0`` yields an empty list.
            prefer_short: When ``True`` results are ordered to favour shorter
                words which typically produce tighter multi-word phrases.

        Returns:
            A list of matching words in deterministic order.
        """

        if limit == 0:
            return []

        normalized = tuple(
            _DIGIT_PATTERN.sub("", phone) for phone in phones if isinstance(phone, str)
        )
        if not normalized:
            return []

        self._ensure_loaded()
        matches = self._phoneme_index.get(normalized)
        if not matches:
            return []

        ordered = list(matches)
        if prefer_short:
            ordered.sort(key=lambda word: (len(word), word))
        else:
            ordered.sort()

        if limit is None or limit < 0:
            return ordered
        return ordered[:limit]

    def split_pronunciation_into_words(
        self,
        phones: Sequence[str],
        *,
        max_pairs: Optional[int] = None,
        prefix_limit: int = 4,
        suffix_limit: int = 6,
        prefer_short: bool = True,
    ) -> List[Tuple[str, str, int]]:
        """Return word pairs that exactly realise ``phones`` when combined.

        The function searches for a split point inside ``phones`` and looks up
        words whose pronunciations match the prefix and suffix segments. Only
        combinations where *both* halves correspond to real CMU entries are
        returned. Stress markers are ignored, matching the behaviour of
        :meth:`find_words_by_phonemes`.

        Args:
            phones: Full pronunciation to split.
            max_pairs: Optional cap on the number of distinct word pairs to
                return across all split positions.
            prefix_limit: Maximum number of candidate words to request for the
                leading segment at each split.
            suffix_limit: Maximum number of candidate words to request for the
                trailing segment at each split.
            prefer_short: Whether shorter words should be prioritised when
                fetching matches for each segment.

        Returns:
            A list of ``(prefix_word, suffix_word, split_index)`` tuples sorted
            by the order they were discovered.
        """

        phone_list = [
            phone for phone in phones if isinstance(phone, str) and phone.strip()
        ]
        if len(phone_list) < 2:
            return []

        results: List[Tuple[str, str, int]] = []
        seen_pairs: Set[Tuple[str, str]] = set()
        remaining = None if max_pairs is None else max(int(max_pairs), 0)

        for split_index in range(1, len(phone_list)):
            if remaining is not None and remaining <= 0:
                break

            prefix_candidates = self.find_words_by_phonemes(
                phone_list[:split_index],
                limit=prefix_limit,
                prefer_short=prefer_short,
            )
            if not prefix_candidates:
                continue

            suffix_candidates = self.find_words_by_phonemes(
                phone_list[split_index:],
                limit=suffix_limit,
                prefer_short=prefer_short,
            )
            if not suffix_candidates:
                continue

            for prefix_word in prefix_candidates:
                for suffix_word in suffix_candidates:
                    key = (prefix_word, suffix_word)
                    if key in seen_pairs:
                        continue
                    seen_pairs.add(key)
                    results.append((prefix_word, suffix_word, split_index))
                    if remaining is not None:
                        remaining -= 1
                        if remaining <= 0:
                            return results

        return results


DEFAULT_CMU_LOADER = CMUDictLoader()

__all__ = [
    "CMUDictLoader",
    "DEFAULT_CMU_LOADER",
    "VOWEL_PHONEMES",
]
