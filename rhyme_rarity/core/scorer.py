from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - used only for typing
    from .analyzer import EnhancedPhoneticAnalyzer

from .cmudict_loader import VOWEL_PHONEMES


_STRESS_PENALTY_STRONG = 0.06
_STRESS_PENALTY_LIGHT = 0.02
_SYLLABLE_PENALTY_STEP = 0.04
_SPELLING_DAMPING = 0.85

_TIER_THRESHOLDS: Tuple[Tuple[str, float], ...] = (
    ("perfect", 0.97),
    ("very_close", 0.82),
    ("strong", 0.68),
    ("loose", 0.55),
)


@dataclass(frozen=True)
class SlantScore:
    """Phonetic similarity breakdown tailored for near-rhyme analysis."""

    total: float
    rime: float
    vowel: float
    coda: float
    stress_penalty: float
    syllable_penalty: float
    tier: str
    source_rime: Tuple[str, ...] = ()
    target_rime: Tuple[str, ...] = ()
    tie_breaker: float = 0.0
    used_spelling_backoff: bool = False

    @classmethod
    def empty(cls) -> "SlantScore":
        return cls(
            total=0.0,
            rime=0.0,
            vowel=0.0,
            coda=0.0,
            stress_penalty=0.0,
            syllable_penalty=0.0,
            tier="weak",
        )

    @property
    def penalties(self) -> float:
        return self.stress_penalty + self.syllable_penalty


def _collect_rimes(
    analyzer: "EnhancedPhoneticAnalyzer",
    word: str,
    pronunciations: List[List[str]],
) -> List[Tuple[Tuple[str, ...], str, Tuple[str, ...], str, bool]]:
    results: List[Tuple[Tuple[str, ...], str, Tuple[str, ...], str, bool]] = []
    seen: set[Tuple[str, ...]] = set()

    try:
        loader_tails = analyzer._get_rhyme_tails(word, pronunciations)
    except Exception:
        loader_tails = []

    for tail in loader_tails:
        normalized = [analyzer._normalize_phoneme_symbol(symbol) for symbol in tail]
        normalized = [symbol for symbol in normalized if symbol]
        if not normalized:
            continue
        vowel = normalized[0]
        coda = tuple(normalized[1:])
        stress = ""
        for char in tail[0]:
            if char in {"0", "1", "2"}:
                stress = char
                break
        key = tuple(normalized)
        if key in seen:
            continue
        seen.add(key)
        results.append((key, vowel, coda, stress, False))

    for phones in pronunciations:
        tail = analyzer._extract_phoneme_tail(phones)
        if not tail:
            continue
        normalized = [analyzer._normalize_phoneme_symbol(symbol) for symbol in tail]
        normalized = [symbol for symbol in normalized if symbol]
        if not normalized:
            continue
        vowel = normalized[0]
        coda = tuple(normalized[1:])
        stress = ""
        for char in tail[0]:
            if char in {"0", "1", "2"}:
                stress = char
                break
        key = tuple(normalized)
        if key not in seen:
            seen.add(key)
            results.append((key, vowel, coda, stress, False))

        # Include a secondary variant that captures an additional preceding syllable
        start_index = len(phones) - len(tail)
        prev_start: Optional[int] = None
        for idx in range(start_index - 1, -1, -1):
            base = analyzer._normalize_phoneme_symbol(phones[idx])
            if base in VOWEL_PHONEMES:
                prev_start = idx
                break
        if prev_start is not None and prev_start < start_index:
            extended = phones[prev_start:]
            normalized_ext = [
                analyzer._normalize_phoneme_symbol(symbol) for symbol in extended
            ]
            normalized_ext = [symbol for symbol in normalized_ext if symbol]
            if normalized_ext:
                stress_ext = ""
                for char in extended[0]:
                    if char in {"0", "1", "2"}:
                        stress_ext = char
                        break
                key_ext = tuple(normalized_ext)
                if key_ext not in seen:
                    seen.add(key_ext)
                    vowel_ext = normalized_ext[0]
                    coda_ext = tuple(normalized_ext[1:])
                    results.append((key_ext, vowel_ext, coda_ext, stress_ext, False))

    if results:
        return results

    fallback_vowel = analyzer._approximate_spelling_vowel(word)
    fallback_coda = analyzer._approximate_spelling_coda(word)
    normalized_vowel = analyzer._normalize_phoneme_symbol(fallback_vowel) if fallback_vowel else ""
    normalized_coda = tuple(
        symbol
        for symbol in (analyzer._normalize_phoneme_symbol(item) for item in fallback_coda)
        if symbol
    )

    if not normalized_vowel:
        return []

    normalized = (normalized_vowel, *normalized_coda)
    return [(tuple(normalized), normalized_vowel, normalized_coda, "", True)]


def _syllable_penalty(diff: int) -> float:
    if diff <= 0:
        return 0.0
    if diff == 1:
        return _SYLLABLE_PENALTY_STEP
    return _SYLLABLE_PENALTY_STEP * (1 + min(diff - 1, 2))


def _stress_penalty(marker_a: str, marker_b: str) -> float:
    if not marker_a or not marker_b:
        return 0.0
    strong_a = marker_a in {"1", "2"}
    strong_b = marker_b in {"1", "2"}
    if strong_a != strong_b:
        return _STRESS_PENALTY_STRONG
    if marker_a != marker_b:
        return _STRESS_PENALTY_LIGHT
    return 0.0


def _resolve_tier(total: float) -> str:
    for tier, threshold in _TIER_THRESHOLDS:
        if total >= threshold:
            return tier
    return "weak"


def _orthography_hint(word_a: str, word_b: str) -> float:
    tail_a = word_a[-4:]
    tail_b = word_b[-4:]
    if not tail_a or not tail_b:
        return 0.0
    try:
        return difflib.SequenceMatcher(None, tail_a, tail_b).ratio()
    except Exception:
        return 0.0


def score_pair(
    analyzer: "EnhancedPhoneticAnalyzer",
    word_a: str,
    word_b: str,
) -> SlantScore:
    if not word_a or not word_b:
        return SlantScore.empty()

    clean_a = analyzer._clean_word(word_a)
    clean_b = analyzer._clean_word(word_b)

    if not clean_a or not clean_b:
        return SlantScore.empty()

    pronunciations_a = analyzer._get_pronunciation_variants(clean_a)
    pronunciations_b = analyzer._get_pronunciation_variants(clean_b)

    rimes_a = _collect_rimes(analyzer, clean_a, pronunciations_a)
    rimes_b = _collect_rimes(analyzer, clean_b, pronunciations_b)

    if not rimes_a or not rimes_b:
        hint = _orthography_hint(clean_a, clean_b)
        score = SlantScore.empty()
        return SlantScore(
            total=score.total,
            rime=score.rime,
            vowel=score.vowel,
            coda=score.coda,
            stress_penalty=score.stress_penalty,
            syllable_penalty=score.syllable_penalty,
            tier=score.tier,
            tie_breaker=hint,
            used_spelling_backoff=True,
        )

    syllables_a = analyzer._count_syllables(clean_a)
    syllables_b = analyzer._count_syllables(clean_b)
    syllable_diff = abs(syllables_a - syllables_b)
    base_syllable_penalty = _syllable_penalty(syllable_diff)
    tie_breaker = _orthography_hint(clean_a, clean_b)

    best: Optional[SlantScore] = None

    def _best_sequence_similarity(
        seq1: Tuple[str, ...],
        seq2: Tuple[str, ...],
        *,
        emphasize_first: bool = False,
    ) -> float:
        base = analyzer._phoneme_sequence_similarity(seq1, seq2, emphasize_first=emphasize_first)
        best_score = base
        len1, len2 = len(seq1), len(seq2)
        if len1 and len2:
            if len1 >= len2:
                for start in range(len1 - len2 + 1):
                    window = seq1[start : start + len2]
                    score = analyzer._phoneme_sequence_similarity(
                        window,
                        seq2,
                        emphasize_first=emphasize_first,
                    )
                    if score > best_score:
                        best_score = score
            if len2 >= len1:
                for start in range(len2 - len1 + 1):
                    window = seq2[start : start + len1]
                    score = analyzer._phoneme_sequence_similarity(
                        seq1,
                        window,
                        emphasize_first=emphasize_first,
                    )
                    if score > best_score:
                        best_score = score
        return best_score

    for normalized_a, vowel_a, coda_a, stress_a, spelling_a in rimes_a:
        for normalized_b, vowel_b, coda_b, stress_b, spelling_b in rimes_b:
            rime_similarity = _best_sequence_similarity(
                normalized_a,
                normalized_b,
                emphasize_first=True,
            )

            vowels_a = [symbol for symbol in normalized_a if symbol in VOWEL_PHONEMES]
            vowels_b = [symbol for symbol in normalized_b if symbol in VOWEL_PHONEMES]
            vowel_similarity = 0.0
            if vowels_a and vowels_b:
                vowel_similarity = max(
                    analyzer._phoneme_feature_similarity(v1, v2)
                    for v1 in vowels_a
                    for v2 in vowels_b
                )

            coda_similarity = _best_sequence_similarity(coda_a, coda_b)

            base_total = (
                (0.6 * rime_similarity)
                + (0.2 * vowel_similarity)
                + (0.2 * coda_similarity)
            )

            stress_pen = _stress_penalty(stress_a, stress_b)
            syll_pen = base_syllable_penalty

            if spelling_a or spelling_b:
                base_total *= _SPELLING_DAMPING

            total = max(0.0, min(1.0, base_total - stress_pen - syll_pen))
            tier = _resolve_tier(total)

            candidate = SlantScore(
                total=total,
                rime=rime_similarity,
                vowel=vowel_similarity,
                coda=coda_similarity,
                stress_penalty=stress_pen,
                syllable_penalty=syll_pen,
                tier=tier,
                source_rime=normalized_a,
                target_rime=normalized_b,
                tie_breaker=tie_breaker,
                used_spelling_backoff=spelling_a or spelling_b,
            )

            if best is None:
                best = candidate
                continue

            if candidate.total > best.total + 1e-6:
                best = candidate
            elif abs(candidate.total - best.total) <= 1e-6:
                if best.used_spelling_backoff and not candidate.used_spelling_backoff:
                    best = candidate
                elif candidate.tie_breaker > best.tie_breaker:
                    best = candidate

    return best if best is not None else SlantScore.empty()


def passes_gate(score: SlantScore) -> bool:
    if score.tier == "weak":
        return False
    return score.total >= 0.55 and score.rime >= 0.4
