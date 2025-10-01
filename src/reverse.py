# -*- coding: utf-8 -*-
"""
Reverse rhyme pipeline:
Given a multi-word input, produce BOTH:
  A) single-word rhymes
  B) multi-word rhymes (retrieved from patterns.db)
Uses two keys:
  - K1: final-word rime
  - K2: cross-word compound rime = last syllable of penultimate + final-word rime
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import functools
import re
import sqlite3
from pathlib import Path

# ---- Import core APIs (Module 1) ----
# Expect these functions; if names differ, add a wrapper here.
try:
    from module1_enhanced_core_phonetic import (
        explain_slant,         # (target_word, candidate_word) -> SlantExplanation
        rank_slant_candidates, # (target_word, candidate_words=[], slant_strength=..., allow_propers=...) -> list
        get_pronunciations,    # (word) -> list[list[str]] phones; if not present, implement a thin wrapper
        extract_rime,          # (phones) -> (vowel_nucleus_tuple, coda_tuple, stress_tag, syll_span)
    )
except Exception:  # pragma: no cover - fallback wiring when legacy module path missing
    from dataclasses import dataclass as _dataclass

    from rhyme_rarity.core import (
        DEFAULT_CMU_LOADER as _DEFAULT_CMU_LOADER,
        EnhancedPhoneticAnalyzer as _EnhancedPhoneticAnalyzer,
        score_pair as _score_pair,
    )
    from rhyme_rarity.core.cmudict_loader import VOWEL_PHONEMES as _VOWEL_PHONEMES

    _DIGIT_RE = re.compile(r"\d")
    _FALLBACK_ANALYZER = _EnhancedPhoneticAnalyzer()

    def get_pronunciations(word: str) -> List[List[str]]:
        """Return CMU pronunciations for ``word`` using the default loader."""

        try:
            return _DEFAULT_CMU_LOADER.get_pronunciations(word)
        except Exception:
            return []

    def extract_rime(phones: List[str]) -> Tuple[Tuple[str, ...], Tuple[str, ...], str, int]:
        """Return (vowel, coda, stress_tag, syllable_span) for a pronunciation."""

        if not phones:
            return ((), (), "none", 0)

        last_vowel_idx: Optional[int] = None
        stress_tag = "none"
        normalized: List[str] = []
        for phone in phones:
            if isinstance(phone, str):
                base = _DIGIT_RE.sub("", phone)
                if base:
                    normalized.append(base)
                else:
                    normalized.append("")
            else:
                normalized.append("")

        for idx in range(len(phones) - 1, -1, -1):
            phone = phones[idx]
            if not isinstance(phone, str):
                continue
            base = _DIGIT_RE.sub("", phone)
            if base in _VOWEL_PHONEMES:
                last_vowel_idx = idx
                if "1" in phone:
                    stress_tag = "primary"
                elif "2" in phone:
                    stress_tag = "secondary"
                elif "0" in phone:
                    stress_tag = "unstressed"
                else:
                    stress_tag = "none"
                break

        if last_vowel_idx is None:
            return ((), (), "none", 0)

        vowel_segment = tuple(
            _DIGIT_RE.sub("", phones[i])
            for i in range(last_vowel_idx, min(last_vowel_idx + 1, len(phones)))
            if isinstance(phones[i], str) and _DIGIT_RE.sub("", phones[i])
        )
        coda_segment = tuple(
            _DIGIT_RE.sub("", phones[i])
            for i in range(last_vowel_idx + 1, len(phones))
            if isinstance(phones[i], str) and _DIGIT_RE.sub("", phones[i])
        )
        syll_span = len(phones) - last_vowel_idx
        return (vowel_segment, coda_segment, stress_tag, syll_span)

    @_dataclass(frozen=True)
    class _FallbackExplanation:
        tier: str
        rime: str
        vowel_match: str
        coda_match: str
        stress_note: str
        score: float
        source_rime: Tuple[str, ...]
        target_rime: Tuple[str, ...]
        raw: Any

    def explain_slant(target_word: str, candidate_word: str | None) -> _FallbackExplanation:
        """Compute a phonetic explanation using the core analyzer."""

        if not candidate_word:
            empty = _score_pair(_FALLBACK_ANALYZER, target_word, "")
            return _FallbackExplanation(
                tier=getattr(empty, "tier", "slant"),
                rime="",
                vowel_match=f"{getattr(empty, 'vowel', 0.0):.3f}",
                coda_match=f"{getattr(empty, 'coda', 0.0):.3f}",
                stress_note=f"penalty {getattr(empty, 'stress_penalty', 0.0):.3f}",
                score=float(getattr(empty, "total", 0.0)),
                source_rime=getattr(empty, "source_rime", ()),
                target_rime=getattr(empty, "target_rime", ()),
                raw=empty,
            )

        slant = _score_pair(_FALLBACK_ANALYZER, target_word, candidate_word)
        source = ".".join(slant.source_rime) if getattr(slant, "source_rime", None) else ""
        target = ".".join(slant.target_rime) if getattr(slant, "target_rime", None) else ""
        if source and target:
            rime = f"{source} ↔ {target}"
        else:
            rime = source or target or ""
        vowel_match = f"{slant.vowel:.3f}"
        coda_match = f"{slant.coda:.3f}"
        stress_note = f"penalty {slant.stress_penalty:.3f}"
        return _FallbackExplanation(
            tier=getattr(slant, "tier", "slant"),
            rime=rime,
            vowel_match=vowel_match,
            coda_match=coda_match,
            stress_note=stress_note,
            score=float(getattr(slant, "total", 0.0)),
            source_rime=getattr(slant, "source_rime", ()),
            target_rime=getattr(slant, "target_rime", ()),
            raw=slant,
        )

    @_dataclass(frozen=True)
    class _FallbackCandidate:
        word: str
        score: float
        tier: str
        explanation: _FallbackExplanation

    def _normalize_token(text: str) -> List[str]:
        return re.findall(r"[A-Za-z']+", text.lower())

    def _slant_threshold(strength: float) -> float:
        strength = max(0.0, min(1.0, float(strength)))
        tight, loose = 0.65, 0.35
        return tight - (tight - loose) * strength

    def rank_slant_candidates(
        target_word: str,
        candidate_words: Optional[List[str]] = None,
        *,
        slant_strength: float = 0.6,
        allow_propers: bool = True,
    ) -> List[_FallbackCandidate]:
        """Return scored single-word rhyme candidates."""

        tokens = _normalize_token(target_word)
        base_word = ""
        if tokens:
            base_word = tokens[-1]
        else:
            parts = [part for part in str(target_word or "").strip().split() if part]
            base_word = parts[-1] if parts else str(target_word or "").strip()

        candidates: List[str] = []
        if candidate_words:
            candidates.extend(candidate_words)
        else:
            try:
                candidates.extend(_DEFAULT_CMU_LOADER.get_rhyming_words(base_word))
            except Exception:
                candidates = []

        seen: set[str] = set()
        scored: List[_FallbackCandidate] = []
        gate = _slant_threshold(slant_strength)
        for cand in candidates:
            if not cand:
                continue
            normalized = cand.strip()
            key = normalized.lower()
            if not key or key in seen:
                continue
            seen.add(key)
            if not allow_propers and normalized and normalized[0].isupper():
                continue
            explanation = explain_slant(target_word, normalized)
            if explanation.score < gate:
                continue
            scored.append(
                _FallbackCandidate(
                    word=normalized,
                    score=explanation.score,
                    tier=explanation.tier,
                    explanation=explanation,
                )
            )

        scored.sort(key=lambda item: item.score, reverse=True)
        return scored


_WORD_RE = re.compile(r"[A-Za-z']+")


@dataclass(frozen=True)
class PhraseKeys:
    k1_last_word_rime: List[str]          # e.g., ["/oʊ/"]
    k2_cross_word_rime: List[str]         # e.g., ["/ɪm.oʊ/"]


# ---- Helpers ----
def _normalize_text(s: str) -> str:
    s = s.strip()
    s = re.sub(r"[^\w'\s-]+", "", s)
    return s.lower()


def tokenize(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower())


@functools.lru_cache(maxsize=8192)
def _rime_label(vowel: Tuple[str, ...], coda: Tuple[str, ...]) -> str:
    # Turn phones into a compact rime label like "/IH N.D OW/" or "/OW/"
    def j(seg): return ".".join(seg) if seg else ""
    v = j(vowel)
    c = j(coda)
    if v and c:
        return f"/{v}.{c}/"
    elif v:
        return f"/{v}/"
    else:
        return "/∅/"


def _last_stressed_rime_labels_for_word(w: str) -> List[str]:
    labels = []
    for pron in get_pronunciations(w) or []:
        v, c, stress, span = extract_rime(pron)
        labels.append(_rime_label(v, c))
    return list(dict.fromkeys(labels))  # dedupe, preserve order


def phrase_rime_keys(phrase: str) -> PhraseKeys:
    toks = tokenize(_normalize_text(phrase))
    if len(toks) == 0:
        return PhraseKeys([], [])
    if len(toks) == 1:
        return PhraseKeys(_last_stressed_rime_labels_for_word(toks[-1]), [])
    # Last word K1
    k1 = _last_stressed_rime_labels_for_word(toks[-1])
    # Cross-word K2: last syllable of penultimate + rime of final
    # We approximate by taking penultimate's last stressed vowel nucleus (no coda) + final rime.
    k2 = []
    pen_prons = get_pronunciations(toks[-2]) or []
    fin_prons = get_pronunciations(toks[-1]) or []
    for p in pen_prons:
        v_pen, c_pen, s_pen, span_pen = extract_rime(p)
        v_pen_only = v_pen  # no coda for cross-boundary
        for f in fin_prons:
            v_fin, c_fin, s_fin, span_fin = extract_rime(f)
            label = _rime_label(tuple(list(v_pen_only)+list(v_fin)), c_fin)
            k2.append(label)
    # Dedupe
    k2 = list(dict.fromkeys(k2))
    return PhraseKeys(k1_last_word_rime=k1, k2_cross_word_rime=k2)


# ---- DB retrieval ----
def _connect_db(db_path: str | Path) -> sqlite3.Connection:
    return sqlite3.connect(str(db_path))


def ensure_db_indices(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
    # Add keys if missing
    cur.execute("PRAGMA table_info(phrases)")
    cols = [r[1] for r in cur.fetchall()]
    if "last_word_rime_key" not in cols:
        try:
            cur.execute("ALTER TABLE phrases ADD COLUMN last_word_rime_key TEXT")
        except sqlite3.OperationalError:
            pass
    if "last_two_syllables_key" not in cols:
        try:
            cur.execute("ALTER TABLE phrases ADD COLUMN last_two_syllables_key TEXT")
        except sqlite3.OperationalError:
            pass
    # Indexes
    cur.execute("CREATE INDEX IF NOT EXISTS idx_last_word_rime_key ON phrases(last_word_rime_key)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_last_two_syllables_key ON phrases(last_two_syllables_key)")
    conn.commit()


def retrieve_phrase_candidates(db_path: str | Path, keys: PhraseKeys, limit: int = 500) -> List[Tuple[str, str, int]]:
    """
    Returns list of (phrase, last_word, freq/intensity)
    Matches either last_word_rime_key in K1 OR last_two_syllables_key in K2
    """
    if not keys.k1_last_word_rime and not keys.k2_cross_word_rime:
        return []
    conn = _connect_db(db_path)
    ensure_db_indices(conn)
    cur = conn.cursor()

    conds = []
    params: List[Any] = []
    if keys.k1_last_word_rime:
        qmarks = ",".join(["?"] * len(keys.k1_last_word_rime))
        conds.append(f"last_word_rime_key IN ({qmarks})")
        params.extend(keys.k1_last_word_rime)
    if keys.k2_cross_word_rime:
        qmarks = ",".join(["?"] * len(keys.k2_cross_word_rime))
        conds.append(f"last_two_syllables_key IN ({qmarks})")
        params.extend(keys.k2_cross_word_rime)
    where = " OR ".join(conds)
    sql = f"""
        SELECT phrase, last_word, COALESCE(freq, 1) as freq
        FROM phrases
        WHERE {where}
        ORDER BY freq DESC
        LIMIT ?
    """
    params.append(limit)
    try:
        rows = cur.execute(sql, params).fetchall()
    except sqlite3.OperationalError:
        rows = []
    finally:
        conn.close()
    return [(r[0], r[1], int(r[2])) for r in rows]


# ---- Public API ----
@dataclass(frozen=True)
class SingleWordResult:
    word: str
    tier: str
    score: float
    explanation: Any


@dataclass(frozen=True)
class PhraseResult:
    phrase: str
    last_word: str
    tier: str
    final_score: float
    freq: int
    explanation: Any


def rhyme_from_phrase(
    phrase: str,
    *,
    db_path: str | Path = "patterns.db",
    slant_strength: float = 0.6,
    allow_propers: bool = True,
    limit: int = 100,
) -> Tuple[List[SingleWordResult], List[PhraseResult]]:
    """Main entry: returns (single_words, multi_word_phrases)."""
    keys = phrase_rime_keys(phrase)

    # A) Single-word: reuse your normal ranker; pass empty candidate list to use default vocab.
    singles_scored = rank_slant_candidates(phrase, [], slant_strength=slant_strength, allow_propers=allow_propers)
    single_out: List[SingleWordResult] = []
    for row in singles_scored[:limit]:
        # Row could be (word, score, tier, ...), or object with attributes
        w = getattr(row, "word", None) or (row[0] if isinstance(row, (list, tuple)) else None)
        tier = getattr(row, "tier", None) or (row[2] if isinstance(row, (list, tuple)) and len(row) > 2 else "slant")
        score = getattr(row, "score", None) or (row[1] if isinstance(row, (list, tuple)) else 0.0)
        exp = getattr(row, "explanation", None) or explain_slant(phrase, w)
        single_out.append(SingleWordResult(word=w, tier=tier, score=float(score), explanation=exp))

    # B) Multi-word: retrieve from DB and re-score by last word
    db_rows = retrieve_phrase_candidates(db_path, keys, limit=limit * 5)
    ph_out: List[PhraseResult] = []
    for text, last, freq in db_rows:
        exp = explain_slant(phrase, last)
        tier = getattr(exp, "tier", "slant")
        phon = getattr(exp, "score", 0.0)
        # Simple composite: phonetics primary + small fluency (freq)
        final_score = float(0.85 * phon + 0.15 * min(1.0, freq / 10.0))
        ph_out.append(PhraseResult(phrase=text, last_word=last, tier=tier, final_score=final_score, freq=freq, explanation=exp))

    # Sort
    single_out.sort(key=lambda r: r.score, reverse=True)
    ph_out.sort(key=lambda r: r.final_score, reverse=True)

    # Clip
    return single_out[:limit], ph_out[:limit]
