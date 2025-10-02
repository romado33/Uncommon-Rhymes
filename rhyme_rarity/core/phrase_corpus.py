"""Static phrase templates and idiomatic n-gram lookups for rhyme generation."""

from __future__ import annotations

import json
import os
import re
import sqlite3
from contextlib import closing
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Set, Tuple

__all__ = [
    "lookup_template_words",
    "lookup_ngram_phrases",
]


def _normalize_rhyme_key(key: str) -> str:
    """Return an uppercase rhyme key stripped of stress markers.

    The CMU dictionary encodes stress digits inside each phoneme. For template
    lookup we operate on simplified keys that remove the digits but preserve the
    consonant/vowel sequence so we can share back-off inventories across
    multiple stress patterns of the same rime.
    """

    if not key:
        return ""
    cleaned = re.sub(r"\d", "", key).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.upper()


_TEMPLATE_CONFIG_ENV = "RHYME_RARITY_TEMPLATE_CONFIG"
_PHRASE_CORPUS_ENV = "RHYME_RARITY_PHRASE_CORPUS"
_PHRASE_DATABASE_ENV = "RHYME_RARITY_PHRASE_DB"

_VOWEL_PHONEMES = {
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


@lru_cache()
def _load_template_bank() -> tuple[
    Dict[str, Dict[str, Sequence[str]]],
    Dict[str, Dict[str, Sequence[str]]],
]:
    """Load template configuration from disk.

    Returns:
        A tuple of ``(templates, fallback_stems)`` where ``templates`` maps a
        full rhyme key to slot inventories and ``fallback_stems`` maps vowel
        families to base words that can be inflected when an exact match is not
        present.
    """

    path = os.getenv(_TEMPLATE_CONFIG_ENV)
    raw_config: Mapping[str, object] | None = None

    if path:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                raw_config = json.load(handle)
        except FileNotFoundError:
            raw_config = None
    if raw_config is None:
        try:
            config_path = resources.files("rhyme_rarity").joinpath(
                "data", "phonetic_templates.json"
            )
            with config_path.open("r", encoding="utf-8") as handle:
                raw_config = json.load(handle)
        except FileNotFoundError:
            raw_config = {}

    if not isinstance(raw_config, Mapping):
        return {}, {}

    templates: Dict[str, Dict[str, Sequence[str]]] = {}
    templates_section = raw_config.get("templates")
    if isinstance(templates_section, Mapping):
        for raw_key, slots in templates_section.items():
            normalized_key = _normalize_rhyme_key(str(raw_key))
            if not normalized_key or not isinstance(slots, Mapping):
                continue
            normalized_slots: Dict[str, Sequence[str]] = {}
            for slot, words in slots.items():
                if not isinstance(words, Sequence) or isinstance(words, (str, bytes)):
                    continue
                cleaned = tuple(
                    word.strip()
                    for word in words
                    if isinstance(word, str) and word.strip()
                )
                if cleaned:
                    normalized_slots[str(slot)] = cleaned
            if normalized_slots:
                templates[normalized_key] = normalized_slots

    fallback: Dict[str, Dict[str, Sequence[str]]] = {}
    fallback_section = raw_config.get("fallback_stems")
    if isinstance(fallback_section, Mapping):
        for vowel, slots in fallback_section.items():
            normalized_vowel = _normalize_rhyme_key(str(vowel))
            if not normalized_vowel or not isinstance(slots, Mapping):
                continue
            normalized_slots = {}
            for slot, words in slots.items():
                if not isinstance(words, Sequence) or isinstance(words, (str, bytes)):
                    continue
                cleaned = tuple(
                    word.strip()
                    for word in words
                    if isinstance(word, str) and word.strip()
                )
                if cleaned:
                    normalized_slots[str(slot)] = cleaned
            if normalized_slots:
                fallback[normalized_vowel] = normalized_slots

    return templates, fallback


_CONSONANT_DOUBLE_PATTERN = re.compile(r"[aeiou][bcdfghjklmnpqrtvz]$")


def _extract_primary_vowel(key: str) -> str:
    """Return the most specific vowel symbol contained in ``key``."""

    if not key:
        return ""
    tokens = key.split()
    for token in tokens:
        if token in _VOWEL_PHONEMES:
            return token
    return tokens[0] if tokens else ""


def _derive_verb_variants(stem: str) -> Set[str]:
    """Generate light-weight inflections for ``stem`` verbs."""

    lower = stem.lower()
    variants: Set[str] = set()
    if not lower:
        return variants

    variants.add(stem + "s")
    if lower.endswith("e") and not lower.endswith(("ee", "oe")):
        variants.add(stem + "d")
        variants.add(stem[:-1] + "ing")
    else:
        if len(lower) <= 4 and _CONSONANT_DOUBLE_PATTERN.search(lower):
            doubled = stem + stem[-1]
            variants.add(doubled + "ed")
            variants.add(doubled + "ing")
        else:
            variants.add(stem + "ed")
            variants.add(stem + "ing")
    return variants


def _derive_noun_variants(noun: str) -> Set[str]:
    """Return a set of pluralised noun forms."""

    lower = noun.lower()
    variants: Set[str] = set()
    if not lower:
        return variants

    if lower.endswith("y") and not lower.endswith(("ay", "ey", "iy", "oy", "uy")):
        variants.add(noun[:-1] + "ies")
    elif lower.endswith(("s", "x", "z", "ch", "sh")):
        variants.add(noun + "es")
    else:
        variants.add(noun + "s")
    return variants


def _derive_fallback_templates(
    normalized_key: str,
    fallback_stems: Dict[str, Dict[str, Sequence[str]]],
) -> Dict[str, Set[str]]:
    """Build template suggestions for keys missing from the explicit bank."""

    vowel = _extract_primary_vowel(normalized_key)
    if not vowel:
        return {}

    stems = fallback_stems.get(vowel)
    if not stems:
        return {}

    derived: Dict[str, Set[str]] = {}
    for slot, words in stems.items():
        slot_values: Set[str] = set()
        for word in words:
            cleaned = (word or "").strip()
            if not cleaned:
                continue
            slot_values.add(cleaned)
            if slot == "verbs":
                slot_values.update(_derive_verb_variants(cleaned))
            elif slot == "nouns":
                slot_values.update(_derive_noun_variants(cleaned))
        if slot_values:
            derived[slot] = slot_values
    return derived

GENERIC_TEMPLATE_BANK: Dict[str, Sequence[str]] = {
    "adjectives": ("silver", "hidden", "lonely", "turbulent"),
    "nouns": ("story", "signal", "harbor"),
    "verbs": ("keep", "carry", "chase", "ride"),
}


def _load_phrase_corpus_from_source() -> Tuple[
    Dict[str, Sequence[str]],
    Dict[str, Sequence[str]],
]:
    """Load the idiomatic n-gram corpus from disk."""

    raw_data: Mapping[str, object] | None = None
    path = os.getenv(_PHRASE_CORPUS_ENV)
    if path:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                raw_data = json.load(handle)
        except (FileNotFoundError, json.JSONDecodeError):
            raw_data = None

    if raw_data is None:
        corpus_path: Path | None = None
        try:
            corpus_path = resources.files("rhyme_rarity").joinpath(
                "data", "phrase_corpus.json"
            )
            with corpus_path.open("r", encoding="utf-8") as handle:
                raw_data = json.load(handle)
        except (FileNotFoundError, json.JSONDecodeError, AttributeError):
            raw_data = None

        if raw_data is None:
            try:
                filesystem_path = (
                    Path(__file__).resolve().parents[1] / "data" / "phrase_corpus.json"
                )
                with filesystem_path.open("r", encoding="utf-8") as handle:
                    raw_data = json.load(handle)
            except (FileNotFoundError, json.JSONDecodeError):
                raw_data = {}

    if not isinstance(raw_data, Mapping):
        return {}, {}

    by_word: Dict[str, Sequence[str]] = {}
    word_section = raw_data.get("by_word")
    if isinstance(word_section, Mapping):
        for raw_word, phrases in word_section.items():
            word = str(raw_word or "").strip().lower()
            if not word or not isinstance(phrases, Sequence) or isinstance(
                phrases, (str, bytes)
            ):
                continue
            cleaned = tuple(
                " ".join(str(phrase).split())
                for phrase in phrases
                if isinstance(phrase, str) and phrase.strip()
            )
            if cleaned:
                by_word[word] = cleaned

    by_rhyme: Dict[str, Sequence[str]] = {}
    rhyme_section = raw_data.get("by_rhyme")
    if isinstance(rhyme_section, Mapping):
        for raw_key, phrases in rhyme_section.items():
            key = _normalize_rhyme_key(str(raw_key or ""))
            if not key or not isinstance(phrases, Sequence) or isinstance(
                phrases, (str, bytes)
            ):
                continue
            cleaned = tuple(
                " ".join(str(phrase).split())
                for phrase in phrases
                if isinstance(phrase, str) and phrase.strip()
            )
            if cleaned:
                by_rhyme[key] = cleaned

    return by_word, by_rhyme


@lru_cache()
def _load_ngram_corpus() -> Tuple[Dict[str, Sequence[str]], Dict[str, Sequence[str]]]:
    return _load_phrase_corpus_from_source()


def _split_context_segments(value: str) -> Sequence[str]:
    """Yield candidate segments from lyrical contexts."""

    cleaned = " ".join((value or "").split()).strip()
    if not cleaned:
        return ()
    segments = re.split(r"\s*(?:\/{2}|\n|\||;|--)+\s*", cleaned)
    return tuple(segment for segment in segments if segment)


def _mine_phrases_from_database(
    target: str,
    rhyme_keys: Iterable[str],
    *,
    limit: int = 10,
) -> Set[str]:
    """Collect phrases from the SQLite rhyme database for ``target``."""

    db_path = os.getenv(_PHRASE_DATABASE_ENV) or "patterns.db"
    if not db_path or not os.path.exists(db_path):
        return set()

    normalized_target = (target or "").strip().lower()
    if not normalized_target:
        return set()

    try:
        with closing(sqlite3.connect(db_path)) as connection:
            connection.row_factory = sqlite3.Row
            try:
                cursor = connection.execute(
                    """
                    SELECT target_word, target_context, lyrical_context,
                           confidence_score, phonetic_similarity
                    FROM song_rhyme_patterns
                    WHERE LOWER(TRIM(COALESCE(target_word, ''))) = ?
                       OR LOWER(TRIM(COALESCE(target_word_normalized, ''))) = ?
                    ORDER BY COALESCE(confidence_score, 0) DESC,
                             COALESCE(phonetic_similarity, 0) DESC
                    LIMIT ?
                    """,
                    (normalized_target, normalized_target, max(1, int(limit) * 2)),
                )
                rows = cursor.fetchall()
            except sqlite3.OperationalError:
                cursor = connection.execute(
                    """
                    SELECT target_word, target_context, lyrical_context,
                           confidence_score, phonetic_similarity
                    FROM song_rhyme_patterns
                    WHERE LOWER(TRIM(COALESCE(target_word, ''))) = ?
                    ORDER BY COALESCE(confidence_score, 0) DESC,
                             COALESCE(phonetic_similarity, 0) DESC
                    LIMIT ?
                    """,
                    (normalized_target, max(1, int(limit) * 2)),
                )
                rows = cursor.fetchall()
    except sqlite3.Error:
        return set()

    candidates: Set[str] = set()
    for row in rows or ():
        values = []
        if isinstance(row, Mapping):
            values.extend([
                row.get("target_word"),
                row.get("target_context"),
                row.get("lyrical_context"),
            ])
        elif isinstance(row, Sequence):
            values.extend(list(row)[:3])
        for value in values:
            if not value:
                continue
            for segment in _split_context_segments(str(value)):
                tokens = segment.lower().split()
                if tokens and tokens[-1] == normalized_target:
                    candidates.add(segment)
                    if len(candidates) >= limit:
                        return candidates

    if candidates:
        return candidates

    # As a final back-off, probe rhyme keys for segments containing the target.
    normalized_keys = {_normalize_rhyme_key(key) for key in rhyme_keys or ()}
    normalized_keys.discard("")
    if not normalized_keys:
        return set()

    try:
        with closing(sqlite3.connect(db_path)) as connection:
            connection.row_factory = sqlite3.Row
            query = (
                """
                SELECT target_context, lyrical_context
                FROM song_rhyme_patterns
                WHERE LOWER(TRIM(COALESCE(target_word, ''))) LIKE ?
                   OR LOWER(TRIM(COALESCE(target_word_normalized, ''))) LIKE ?
                   OR LOWER(TRIM(COALESCE(pattern, ''))) LIKE ?
                """
            )
            params: Tuple[str, ...] = tuple(
                [f"% {normalized_target}", f"% {normalized_target}", f"% {normalized_target}"]
            )
            try:
                cursor = connection.execute(query, params)
                rows = cursor.fetchall()
            except sqlite3.OperationalError:
                return set()
    except sqlite3.Error:
        return set()

    fallback_candidates: Set[str] = set()
    for row in rows or ():
        row_values = []
        if isinstance(row, Mapping):
            row_values.extend([row.get("target_context"), row.get("lyrical_context")])
        elif isinstance(row, Sequence):
            row_values.extend(list(row)[:2])
        for value in row_values:
            if not value:
                continue
            for segment in _split_context_segments(str(value)):
                tokens = segment.lower().split()
                if tokens and tokens[-1] == normalized_target:
                    fallback_candidates.add(segment)
                    if len(fallback_candidates) >= limit:
                        return fallback_candidates
    return fallback_candidates


def lookup_template_words(rhyme_keys: Iterable[str]) -> Dict[str, Set[str]]:
    """Return modifier inventories for the supplied phonetic keys.

    Args:
        rhyme_keys: Iterable of simplified rhyme identifiers. The keys should be
            uppercase strings that omit stress markers. They can represent a
            full rhyme tail (e.g. ``"EY L"``) or a shorter back-off such as a
            vowel cluster (``"EY"``).

    Returns:
        A mapping of template slots (``adjectives``, ``nouns``, ``verbs``) to the
        set of candidate words that can be slotted before a rhyming terminal
        word. Generic defaults are merged in so every slot remains populated.
    """

    collected: Dict[str, Set[str]] = {
        "adjectives": set(),
        "nouns": set(),
        "verbs": set(),
    }

    template_bank, fallback_stems = _load_template_bank()

    for key in rhyme_keys:
        normalized = _normalize_rhyme_key(key)
        if not normalized:
            continue
        bank = template_bank.get(normalized)
        slot_words: Dict[str, Set[str]]
        if bank:
            slot_words = {
                slot: {word for word in words if word}
                for slot, words in bank.items()
            }
        else:
            slot_words = _derive_fallback_templates(normalized, fallback_stems)
        if not slot_words:
            continue
        for slot, words in slot_words.items():
            if slot not in collected:
                collected[slot] = set()
            collected[slot].update(words)

    # Merge generic fallbacks and ensure no slot is empty. The generics also act
    # as a mild smoothing factor so uncommon rhyme keys can still generate
    # phrases even if the targeted inventory is sparse.
    for slot, words in GENERIC_TEMPLATE_BANK.items():
        if slot not in collected:
            collected[slot] = set()
        collected[slot].update(word for word in words if word)

    return {slot: values for slot, values in collected.items() if values}


def lookup_ngram_phrases(
    word: str,
    rhyme_keys: Iterable[str],
    *,
    enable_dynamic_fallback: bool = True,
    dynamic_limit: int = 10,
) -> Set[str]:
    """Fetch idiomatic phrases whose terminal token matches ``word``.

    Args:
        word: Candidate rhyme word that should appear at the end of each phrase.
        rhyme_keys: Iterable of simplified rhyme identifiers that provide
            phonetic back-off cues when a direct word lookup is unavailable.
        enable_dynamic_fallback: When ``True`` the function will query the rhyme
            database for lyric fragments if the static corpus does not provide a
            match.
        dynamic_limit: Maximum number of database derived phrases to return when
            ``enable_dynamic_fallback`` is active.

    Returns:
        A set of phrases drawn from the comprehensive corpus with optional
        database backed fallbacks.
    """

    results: Set[str] = set()
    target = (word or "").strip().lower()
    if not target:
        return results

    corpus_by_word, corpus_by_rhyme = _load_ngram_corpus()

    direct_matches = tuple(
        phrase
        for phrase in corpus_by_word.get(target, ())
        if phrase and phrase.lower().split()[-1] == target
    )
    results.update(direct_matches)

    if not results:
        for key in rhyme_keys:
            normalized = _normalize_rhyme_key(key)
            if not normalized:
                continue
            for phrase in corpus_by_rhyme.get(normalized, ()):  # phonetic back-off
                if not phrase:
                    continue
                if phrase.lower().split()[-1] == target:
                    results.add(phrase)

    if enable_dynamic_fallback and not results:
        mined = _mine_phrases_from_database(
            target,
            tuple(rhyme_keys or tuple()),
            limit=max(1, int(dynamic_limit)),
        )
        results.update(mined)

    return results
