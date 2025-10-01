"""Constrained phrase generation and ranking utilities for rhyme-aware search."""

from __future__ import annotations

import difflib
from dataclasses import dataclass, replace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from rhyme_rarity.utils.syllables import estimate_syllable_count

from .phrase_corpus import lookup_ngram_phrases
from .rarity_map import DEFAULT_RARITY_MAP, WordRarityMap
from .scorer import SlantScore, passes_gate, score_pair

__all__ = [
    "TEMPLATES",
    "PhraseCandidate",
    "generate_phrases_for_endwords",
    "retrieve_phrases_by_last_word",
    "rank_phrases",
]


_RELAXED_MULTI_TOTAL_FLOOR = 0.5
_RELAXED_MULTI_RIME_FLOOR = 0.3
_RELAXED_MULTI_RIME_RATIO = 0.6
_RELAXED_MULTI_FALLBACK_FLOOR = 0.6


def _tier_from_total(total: float) -> str:
    if total >= 0.97:
        return "perfect"
    if total >= 0.82:
        return "very_close"
    if total >= 0.68:
        return "strong"
    if total >= 0.55:
        return "loose"
    return "weak"


def _relax_multi_gate(
    analyzer: Any,
    base_phrase: str,
    candidate_phrase: str,
    slant: SlantScore,
) -> SlantScore | None:
    """Apply a softer similarity check for multi-word phrases."""

    similarity_fn = getattr(analyzer, "get_phonetic_similarity", None)
    fallback_total = 0.0
    if callable(similarity_fn):
        try:
            fallback_total = float(similarity_fn(base_phrase, candidate_phrase))
        except Exception:
            fallback_total = 0.0

    fallback_total = max(0.0, min(1.0, fallback_total))
    if slant.tier == "weak" and fallback_total < _RELAXED_MULTI_FALLBACK_FLOOR:
        return None
    relaxed_total = max(float(slant.total), fallback_total)

    relaxed_rime = float(slant.rime)
    if fallback_total >= _RELAXED_MULTI_TOTAL_FLOOR:
        relaxed_rime = max(relaxed_rime, fallback_total * _RELAXED_MULTI_RIME_RATIO)

    relaxed_total = min(relaxed_total, 1.0)
    relaxed_rime = min(relaxed_rime, 1.0)

    if relaxed_total < _RELAXED_MULTI_TOTAL_FLOOR or relaxed_rime < _RELAXED_MULTI_RIME_FLOOR:
        return None

    if relaxed_total == slant.total and relaxed_rime == slant.rime:
        return slant

    return replace(
        slant,
        total=relaxed_total,
        rime=relaxed_rime,
        tier=_tier_from_total(relaxed_total),
    )


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


@dataclass(frozen=True)
class RankedPhrase:
    """Container describing a scored phrase produced by :func:`rank_phrases`."""

    phrase: str
    score: float
    tier: str
    why: Tuple[str, ...]
    slant_score: SlantScore
    bonuses: Dict[str, float]
    metadata: Dict[str, Any]


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


def _normalise_phrase(text: str) -> str:
    return " ".join(part for part in text.split() if part).strip().lower()


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


def retrieve_phrases_by_last_word(
    end_word: str,
    rhyme_keys: Iterable[str] = (),
    *,
    repository: Any = None,
    limit: int = 40,
    min_tokens: int = 2,
    max_tokens: int = 5,
) -> List[Tuple[str, Dict[str, Any]]]:
    """Retrieve idiomatic phrases whose terminal token matches ``end_word``.

    The lookup blends a tiny built-in corpus with optional repository backed
    retrieval so callers can surface domain specific phrases when a database is
    available. Results are deduplicated and normalised to ensure consistent
    downstream handling.
    """

    normalized_end = _normalise_phrase(end_word)
    if not normalized_end:
        return []

    seen: Set[str] = set()
    results: List[Tuple[str, Dict[str, Any]]] = []

    def _register(phrase: str, metadata: Optional[Mapping[str, Any]] = None) -> None:
        cleaned = " ".join(part for part in phrase.split() if part).strip()
        if not cleaned:
            return
        key = _normalise_phrase(cleaned)
        if not key or key in seen:
            return
        tokens = key.split()
        if not tokens or tokens[-1] != normalized_end:
            return
        if not (min_tokens <= len(tokens) <= max_tokens):
            return
        seen.add(key)
        entry_metadata = {"source": "corpus_ngram"}
        if metadata:
            for meta_key, meta_value in metadata.items():
                if meta_key not in entry_metadata:
                    entry_metadata[meta_key] = meta_value
        results.append((cleaned, entry_metadata))

    for phrase in lookup_ngram_phrases(normalized_end, rhyme_keys):
        if phrase:
            _register(phrase, {"source": "corpus_ngram"})

    fetcher = None
    if repository is not None:
        fetcher = getattr(repository, "fetch_phrases_for_word", None)

    if callable(fetcher):
        try:
            fetched: Iterable[Any] = fetcher(
                normalized_end,
                limit=limit,
                min_tokens=min_tokens,
                max_tokens=max_tokens,
                rhyme_backoffs=tuple(rhyme_keys or tuple()),
            )
        except TypeError:
            try:
                fetched = fetcher(normalized_end, limit)
            except Exception:
                fetched = []
        except Exception:
            fetched = []

        for item in fetched or []:
            phrase_text: str
            metadata: Mapping[str, Any] | None = None
            if isinstance(item, tuple) and len(item) == 2:
                phrase_text = str(item[0])
                metadata = item[1] if isinstance(item[1], Mapping) else None
            elif isinstance(item, Mapping):
                phrase_text = str(
                    item.get("phrase")
                    or item.get("text")
                    or item.get("value")
                    or item.get("context")
                    or ""
                )
                metadata = item
            else:
                phrase_text = str(item)
            if not phrase_text:
                continue
            merged_meta: Dict[str, Any] = {"source": "database"}
            if metadata:
                for meta_key, meta_value in metadata.items():
                    if meta_key not in merged_meta:
                        merged_meta[meta_key] = meta_value
            _register(phrase_text, merged_meta)

    if limit > 0:
        return results[:limit]
    return results


def _average_token_rarity(tokens: Iterable[str], rarity_map: WordRarityMap) -> float:
    values: List[float] = []
    for token in tokens:
        lookup = token.strip().lower()
        if not lookup:
            continue
        try:
            values.append(float(rarity_map.get_rarity(lookup)))
        except Exception:
            values.append(float(DEFAULT_RARITY_MAP.get_rarity(lookup)))
    if not values:
        return 1.0
    return sum(values) / len(values)


def rank_phrases(
    analyzer: Any,
    base_phrase: str,
    candidates: Iterable[Tuple[str, Mapping[str, Any] | None]],
    *,
    rarity_map: Optional[WordRarityMap] = None,
    max_results: Optional[int] = None,
) -> List[RankedPhrase]:
    """Score ``candidates`` relative to ``base_phrase``.

    Each candidate is scored using :func:`score_pair` to assess phonetic
    alignment. Lightweight bonuses favour prosodic alignment, perceived fluency
    (token rarity), and semantic cohesion between non-terminal tokens. Entries
    that fail :func:`passes_gate` are discarded.
    """

    if analyzer is None:
        return []

    base_clean = " ".join(part for part in str(base_phrase or "").split() if part).strip()
    if not base_clean:
        return []

    rarity_lookup = rarity_map or getattr(analyzer, "rarity_map", None) or DEFAULT_RARITY_MAP
    base_syllables = max(estimate_syllable_count(base_clean.lower()), 1)
    base_tokens = base_clean.lower().split()
    base_prefix = " ".join(base_tokens[:-1]) if len(base_tokens) > 1 else ""

    ranked: List[RankedPhrase] = []
    seen: Set[str] = set()

    for phrase, metadata in candidates:
        normalized = " ".join(part for part in str(phrase or "").split() if part).strip()
        if not normalized:
            continue
        key = _normalise_phrase(normalized)
        if not key or key in seen:
            continue
        seen.add(key)

        candidate_tokens = key.split()
        if not candidate_tokens:
            continue
        is_multi_word = len(candidate_tokens) > 1

        used_fallback = False
        try:
            slant = score_pair(analyzer, base_clean, normalized)
        except Exception:
            used_fallback = True
            similarity_fn = getattr(analyzer, "get_phonetic_similarity", None)
            fallback_total = 0.0
            if callable(similarity_fn):
                try:
                    fallback_total = float(similarity_fn(base_clean, normalized))
                except Exception:
                    fallback_total = 0.0
            fallback_total = max(0.0, min(1.0, fallback_total))
            if fallback_total >= 0.97:
                tier = "perfect"
            elif fallback_total >= 0.82:
                tier = "very_close"
            elif fallback_total >= 0.68:
                tier = "strong"
            elif fallback_total >= 0.55:
                tier = "loose"
            else:
                tier = "weak"
            slant = SlantScore(
                total=fallback_total,
                rime=fallback_total,
                vowel=fallback_total,
                coda=fallback_total,
                stress_penalty=0.0,
                syllable_penalty=0.0,
                tier=tier,
            )

        if not used_fallback and not passes_gate(slant):
            if not is_multi_word:
                continue
            relaxed = _relax_multi_gate(analyzer, base_clean, normalized, slant)
            if relaxed is None:
                continue
            slant = relaxed

        candidate_prefix = (
            " ".join(candidate_tokens[:-1]) if len(candidate_tokens) > 1 else ""
        )

        candidate_syllables = max(estimate_syllable_count(key), 1)
        syllable_diff = abs(candidate_syllables - base_syllables)
        prosody_bonus = 1.0 - (syllable_diff / max(base_syllables, candidate_syllables, 1))
        prosody_bonus = max(0.0, min(1.0, prosody_bonus))

        avg_rarity = _average_token_rarity(candidate_tokens, rarity_lookup)
        fluency_bonus = max(0.0, min(1.0, 1.0 - avg_rarity))

        semantic_bonus = 0.0
        if base_prefix and candidate_prefix:
            try:
                semantic_bonus = difflib.SequenceMatcher(
                    None, base_prefix, candidate_prefix
                ).ratio()
            except Exception:
                semantic_bonus = 0.0

        bonus_weights = {"prosody": 0.12, "fluency": 0.08, "semantic": 0.05}
        combined = slant.total
        combined += bonus_weights["prosody"] * prosody_bonus
        combined += bonus_weights["fluency"] * fluency_bonus
        combined += bonus_weights["semantic"] * semantic_bonus
        combined = max(0.0, min(1.0, combined))

        reasons: List[str] = []
        reasons.append(f"phonetic tier: {slant.tier}")
        if prosody_bonus >= 0.9:
            reasons.append("prosody aligned")
        elif prosody_bonus >= 0.75:
            reasons.append("prosody close")
        if fluency_bonus >= 0.7:
            reasons.append("fluent wording")
        if semantic_bonus >= 0.6:
            reasons.append("semantic echo")

        bonus_snapshot = {
            "prosody": prosody_bonus,
            "fluency": fluency_bonus,
            "semantic": semantic_bonus,
        }

        ranked.append(
            RankedPhrase(
                phrase=normalized,
                score=combined,
                tier=slant.tier,
                why=tuple(reasons),
                slant_score=slant,
                bonuses=bonus_snapshot,
                metadata=dict(metadata or {}),
            )
        )

    ranked.sort(key=lambda item: (-item.score, item.phrase))

    if max_results is not None and max_results >= 0:
        return ranked[:max_results]
    return ranked
