"""Constrained phrase generation utilities for rhyme-aware search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from rhyme_rarity.utils.syllables import estimate_syllable_count

__all__ = [
    "TEMPLATES",
    "PhraseCandidate",
    "generate_phrases_for_endwords",
]


@dataclass(frozen=True)
class PhraseCandidate:
    """Container describing a generated multi-word phrase."""

    text: str
    end_word: str
    template: str
    score: float
    token_count: int
    stress_pattern: str | None = None


@dataclass
class _BeamState:
    tokens: Tuple[str, ...]
    stress: str
    score: float


# The template inventory favours compact two and three word phrases. Each entry
# specifies the template name, the ordered slot tokens, and a mild prior score
# that helps nudge the beam search toward more idiomatic combinations.
TEMPLATES: Sequence[Dict[str, object]] = (
    {
        "name": "descriptor_end",
        "slots": ("{ADJECTIVE}", "{END}"),
        "prior": 1.1,
    },
    {
        "name": "article_descriptor_end",
        "slots": ("{ARTICLE}", "{ADJECTIVE}", "{END}"),
        "prior": 1.05,
    },
    {
        "name": "mood_end",
        "slots": ("{MOOD}", "{END}"),
        "prior": 1.0,
    },
    {
        "name": "verb_the_end",
        "slots": ("{VERB}", "the", "{END}"),
        "prior": 0.95,
    },
    {
        "name": "adverb_verb_end",
        "slots": ("{ADVERB}", "{VERB}", "{END}"),
        "prior": 0.9,
    },
    {
        "name": "qualifier_noun_end",
        "slots": ("{QUALIFIER}", "{NOUN}", "{END}"),
        "prior": 0.9,
    },
    {
        "name": "noun_end",
        "slots": ("{NOUN}", "{END}"),
        "prior": 1.0,
    },
)


_PLACEHOLDER_VOCAB: Dict[str, Tuple[str, ...]] = {
    "{ARTICLE}": ("the", "a", "this", "that"),
    "{ADJECTIVE}": (
        "ancient",
        "silver",
        "hidden",
        "wandering",
        "shifting",
        "restless",
        "lonely",
        "golden",
        "open",
    ),
    "{QUALIFIER}": (
        "quiet",
        "midnight",
        "shadow",
        "river",
        "ember",
        "winter",
    ),
    "{NOUN}": (
        "signal",
        "whisper",
        "sunset",
        "ember",
        "shadow",
        "harbor",
        "compass",
        "echo",
    ),
    "{MOOD}": (
        "midnight",
        "twilight",
        "neon",
        "ancient",
        "gentle",
        "hollow",
    ),
    "{VERB}": (
        "chase",
        "follow",
        "trace",
        "seek",
        "find",
        "ride",
    ),
    "{ADVERB}": (
        "softly",
        "slowly",
        "boldly",
        "quietly",
    ),
}


_SLOT_BASE_WEIGHTS: Dict[str, float] = {
    "{ARTICLE}": 0.4,
    "{ADJECTIVE}": 0.75,
    "{QUALIFIER}": 0.7,
    "{NOUN}": 0.65,
    "{MOOD}": 0.6,
    "{VERB}": 0.6,
    "{ADVERB}": 0.45,
    "{END}": 1.1,
}


_STRESS_OVERRIDES: Dict[str, str] = {
    "the": "0",
    "a": "0",
    "this": "1",
    "that": "1",
    "into": "10",
    "softly": "10",
    "slowly": "10",
    "quietly": "100",
    "boldly": "10",
    "ancient": "10",
    "silver": "10",
    "hidden": "10",
    "wandering": "100",
    "shifting": "10",
    "restless": "10",
    "lonely": "10",
    "golden": "10",
    "open": "10",
    "quiet": "10",
    "midnight": "10",
    "shadow": "10",
    "river": "10",
    "ember": "10",
    "winter": "10",
    "signal": "10",
    "whisper": "10",
    "sunset": "10",
    "harbor": "10",
    "compass": "10",
    "echo": "10",
    "neon": "10",
    "gentle": "10",
    "hollow": "10",
    "chase": "1",
    "follow": "10",
    "trace": "1",
    "seek": "1",
    "find": "1",
    "ride": "1",
}


def _stress_for_token(token: str) -> str:
    lookup = _STRESS_OVERRIDES.get(token.lower())
    if lookup is not None:
        return lookup

    syllables = estimate_syllable_count(token.lower())
    if syllables <= 0:
        return ""
    if syllables == 1:
        return "1"
    return "1" + "0" * (syllables - 1)


def _options_for_slot(
    slot: str,
    end_words: Sequence[str],
    *,
    base_word: str,
) -> List[Tuple[str, str, float]]:
    if slot == "{END}":
        options: List[Tuple[str, str, float]] = []
        for word in end_words:
            stress = _stress_for_token(word)
            weight = _SLOT_BASE_WEIGHTS.get(slot, 1.0)
            if word.lower() == base_word.lower():
                weight += 0.15
            options.append((word, stress, weight))
        return options

    tokens = _PLACEHOLDER_VOCAB.get(slot)
    if not tokens:
        return []

    weight = _SLOT_BASE_WEIGHTS.get(slot, 0.5)
    results = []
    for token in tokens:
        stress = _stress_for_token(token)
        bonus = 0.0
        if token.endswith("ing"):
            bonus -= 0.05
        elif len(token) <= 4:
            bonus += 0.05
        results.append((token, stress, weight + bonus))
    return results


def _options_for_literal(token: str) -> List[Tuple[str, str, float]]:
    stress = _stress_for_token(token)
    return [(token, stress, 0.35)]


def generate_phrases_for_endwords(
    base_word: str,
    end_words: Iterable[str],
    *,
    beam_width: int = 8,
    max_phrases: int = 18,
) -> List[PhraseCandidate]:
    """Generate compact phrases whose final token matches ``end_words``.

    Args:
        base_word: Canonical word used when weighting the beam search. If empty
            the first ``end_words`` entry is treated as the base.
        end_words: Iterable of admissible terminal tokens.
        beam_width: Maximum number of partial states to keep during expansion.
        max_phrases: Upper bound on the number of unique phrases returned.

    Returns:
        A list of :class:`PhraseCandidate` instances sorted by descending score.
    """

    normalized_end_words = []
    seen_end_words: Set[str] = set()
    for word in end_words:
        cleaned = str(word or "").strip()
        if not cleaned:
            continue
        lower = cleaned.lower()
        if lower in seen_end_words:
            continue
        seen_end_words.add(lower)
        normalized_end_words.append(cleaned)

    if not normalized_end_words:
        return []

    canonical = base_word.strip() if base_word else normalized_end_words[0]

    beam_width = max(1, int(beam_width))
    results: Dict[str, PhraseCandidate] = {}

    for template in TEMPLATES:
        slots: Sequence[str] = tuple(template.get("slots", ()))  # type: ignore[arg-type]
        if not slots:
            continue

        prior = float(template.get("prior", 1.0))  # type: ignore[arg-type]
        template_name = str(template.get("name", "template"))

        states: List[_BeamState] = [_BeamState(tokens=tuple(), stress="", score=prior)]

        for slot in slots:
            if not states:
                break

            if slot.startswith("{") and slot.endswith("}"):
                options = _options_for_slot(slot, normalized_end_words, base_word=canonical)
            else:
                options = _options_for_literal(slot)

            if not options:
                states = []
                break

            expanded: List[_BeamState] = []
            for state in states:
                for token, stress, weight in options:
                    tokens = state.tokens + (token,)
                    new_stress = state.stress + stress
                    new_score = state.score + weight
                    expanded.append(_BeamState(tokens=tokens, stress=new_stress, score=new_score))

            expanded.sort(key=lambda item: (-item.score, len(item.tokens), item.tokens))
            states = expanded[:beam_width]

        for state in states:
            if not state.tokens:
                continue
            terminal = state.tokens[-1]
            if terminal.lower() not in seen_end_words:
                continue

            token_count = len(state.tokens)
            length_penalty = 0.0
            if token_count == 2:
                length_penalty = 0.0
            elif token_count == 3:
                length_penalty = 0.05
            else:
                length_penalty = 0.25 + 0.05 * max(0, token_count - 3)

            score = state.score - length_penalty
            stress_pattern = state.stress or None

            phrase_text = " ".join(state.tokens)
            phrase_key = phrase_text.lower()

            candidate = PhraseCandidate(
                text=phrase_text,
                end_word=terminal,
                template=template_name,
                score=score,
                token_count=token_count,
                stress_pattern=stress_pattern,
            )

            existing = results.get(phrase_key)
            if existing is None or candidate.score > existing.score:
                results[phrase_key] = candidate

    ordered = sorted(
        results.values(),
        key=lambda item: (-item.score, item.token_count, item.text),
    )

    return ordered[:max_phrases]
