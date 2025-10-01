"""Static phrase templates and idiomatic n-gram lookups for rhyme generation."""

from __future__ import annotations

import re
from typing import Dict, Iterable, Sequence, Set

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


# Template banks organised by simplified rhyme key. The goal is to provide a
# small but expressive set of modifiers that can be combined with rhyming
# terminal words. We blend these targeted inventories with a generic fallback so
# every rhyme key can generate at least a handful of phrase variants.
PHONETIC_TEMPLATE_BANK: Dict[str, Dict[str, Sequence[str]]] = {
    "EY L": {
        "adjectives": ("paper", "blazing", "ancient", "wandering"),
        "nouns": ("ghost", "river", "shadow"),
        "verbs": ("blaze", "forge", "chase"),
    },
    "EY": {
        "adjectives": ("everyday", "wayward"),
        "nouns": ("sunset", "morning"),
        "verbs": ("sway", "play"),
    },
    "OW": {
        "adjectives": ("steady", "afterglow", "hollow"),
        "nouns": ("evening", "river"),
        "verbs": ("follow", "borrow", "let"),
    },
    "IY L": {
        "adjectives": ("steel", "eager", "emerald"),
        "nouns": ("city", "winter"),
        "verbs": ("feel", "reel", "conceal"),
    },
    "AY T": {
        "adjectives": ("midnight", "satellite"),
        "nouns": ("candle", "street"),
        "verbs": ("ignite", "rewrite"),
    },
}

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

    for key in rhyme_keys:
        normalized = _normalize_rhyme_key(key)
        if not normalized:
            continue
        bank = PHONETIC_TEMPLATE_BANK.get(normalized)
        if not bank:
            continue
        for slot, words in bank.items():
            if slot not in collected:
                collected[slot] = set()
            collected[slot].update(word for word in words if word)

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
