"""Static phrase templates and idiomatic n-gram lookups for rhyme generation."""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from importlib import resources
from typing import Dict, Iterable, Mapping, Sequence, Set

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


# Compact n-gram inventory. Each phrase ends in a word that regularly appears as
# a rhyme target in tests and demos. The corpus is intentionally lightweight so
# unit tests remain fast while still demonstrating idiomatic retrieval.
NGRAM_CORPUS_BY_WORD: Dict[str, Sequence[str]] = {
    "trail": ("paper trail", "blazing trail", "hidden trail"),
    "mail": ("snail mail", "chain mail"),
    "fail": ("epic fail", "major fail"),
    "flow": ("steady flow", "afterglow"),
    "go": ("on the go", "let it go"),
    "feel": ("deeply feel", "can you feel"),
    "light": ("guiding light", "midnight light"),
}

# Additional groupings indexed by rhyme key so words that share the same phonetic
# ending can borrow idioms even if they do not appear explicitly in the word map.
NGRAM_CORPUS_BY_RHYME: Dict[str, Sequence[str]] = {
    "EY L": ("paper trail", "snail mail", "chain mail", "hidden trail"),
    "EY": ("everyday sway", "nightly play"),
    "OW": ("steady flow", "open window"),
    "IY L": ("can you feel", "sudden steel"),
    "AY T": ("midnight light", "starlit night"),
}


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


def lookup_ngram_phrases(word: str, rhyme_keys: Iterable[str]) -> Set[str]:
    """Fetch idiomatic phrases whose terminal token matches ``word``.

    Args:
        word: Candidate rhyme word that should appear at the end of each phrase.
        rhyme_keys: Iterable of simplified rhyme identifiers that provide
            phonetic back-off cues when a direct word lookup is unavailable.

    Returns:
        A set of phrases drawn from the miniature corpus.
    """

    results: Set[str] = set()
    target = (word or "").strip().lower()
    if not target:
        return results

    for phrase in NGRAM_CORPUS_BY_WORD.get(target, ()):  # direct lookups first
        if phrase and phrase.lower().split()[-1] == target:
            results.add(phrase)

    for key in rhyme_keys:
        normalized = _normalize_rhyme_key(key)
        if not normalized:
            continue
        for phrase in NGRAM_CORPUS_BY_RHYME.get(normalized, ()):  # phonetic back-off
            if not phrase:
                continue
            if phrase.lower().split()[-1] == target:
                results.add(phrase)

    return results
