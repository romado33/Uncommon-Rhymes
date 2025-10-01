"""Phonetic analysis utilities powering the RhymeRarity project."""

from __future__ import annotations

import difflib
import math
import re
import threading
from collections import Counter, OrderedDict
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from rhyme_rarity.utils.syllables import estimate_syllable_count
from rhyme_rarity.utils.observability import get_logger

from .cmudict_loader import CMUDictLoader, DEFAULT_CMU_LOADER, VOWEL_PHONEMES
from .feature_profile import (
    PhraseComponents,
    PhoneticMatch,
    RhymeFeatureProfile,
    extract_phrase_components,
    pronouncing,
)
from .phrase_corpus import lookup_ngram_phrases, lookup_template_words
from .phrases import (
    PhraseCandidate,
    RankedPhrase,
    generate_phrases_for_endwords,
    rank_phrases,
    retrieve_phrases_by_last_word,
)
from .rarity_map import DEFAULT_RARITY_MAP, WordRarityMap
from .scorer import SlantScore, passes_gate, score_pair

RARITY_SIMILARITY_WEIGHT: float = 0.65
RARITY_NOVELTY_WEIGHT: float = 0.35


# ---------------------------------------------------------------------------
# Phoneme feature tables
# ---------------------------------------------------------------------------

# A compact feature inventory derived from PanPhon's SPE feature schema. The
# values are intentionally simplified so we can approximate distance-based
# similarity without introducing a heavy dependency at runtime. The mapping
# only needs to cover the phonemes that appear in CMUdict and common fallback
# graphemes that we inspect when CMU pronunciations are unavailable.
PHONEME_FEATURES: Dict[str, Dict[str, str]] = {
    # Vowels
    "AA": {"category": "vowel", "height": "open", "backness": "back", "rounding": "unrounded", "tenseness": "tense"},
    "AE": {"category": "vowel", "height": "near-open", "backness": "front", "rounding": "unrounded", "tenseness": "lax"},
    "AH": {"category": "vowel", "height": "open-mid", "backness": "central", "rounding": "unrounded", "tenseness": "lax"},
    "AO": {"category": "vowel", "height": "open", "backness": "back", "rounding": "rounded", "tenseness": "tense"},
    "AW": {"category": "vowel", "height": "open", "backness": "back", "rounding": "rounded", "diphthong": "1"},
    "AY": {"category": "vowel", "height": "open", "backness": "front", "rounding": "unrounded", "diphthong": "1"},
    "EH": {"category": "vowel", "height": "open-mid", "backness": "front", "rounding": "unrounded", "tenseness": "lax"},
    "ER": {"category": "vowel", "height": "mid", "backness": "central", "rounding": "rounded", "rhotacized": "1"},
    "EY": {"category": "vowel", "height": "close-mid", "backness": "front", "rounding": "unrounded", "tenseness": "tense", "diphthong": "1"},
    "IH": {"category": "vowel", "height": "close", "backness": "front", "rounding": "unrounded", "tenseness": "lax"},
    "IY": {"category": "vowel", "height": "close", "backness": "front", "rounding": "unrounded", "tenseness": "tense"},
    "OW": {"category": "vowel", "height": "close-mid", "backness": "back", "rounding": "rounded", "tenseness": "tense", "diphthong": "1"},
    "OY": {"category": "vowel", "height": "close-mid", "backness": "back", "rounding": "rounded", "diphthong": "1"},
    "UH": {"category": "vowel", "height": "close-mid", "backness": "back", "rounding": "rounded", "tenseness": "lax"},
    "UW": {"category": "vowel", "height": "close", "backness": "back", "rounding": "rounded", "tenseness": "tense"},
    "UX": {"category": "vowel", "height": "close", "backness": "central", "rounding": "rounded", "tenseness": "tense"},
    "AX": {"category": "vowel", "height": "mid", "backness": "central", "rounding": "unrounded", "tenseness": "lax"},
    "AXR": {"category": "vowel", "height": "mid", "backness": "central", "rounding": "rounded", "rhotacized": "1"},

    # Stops
    "P": {"category": "consonant", "place": "bilabial", "manner": "stop", "voicing": "voiceless", "nasal": "0"},
    "B": {"category": "consonant", "place": "bilabial", "manner": "stop", "voicing": "voiced", "nasal": "0"},
    "T": {"category": "consonant", "place": "alveolar", "manner": "stop", "voicing": "voiceless", "nasal": "0"},
    "D": {"category": "consonant", "place": "alveolar", "manner": "stop", "voicing": "voiced", "nasal": "0"},
    "K": {"category": "consonant", "place": "velar", "manner": "stop", "voicing": "voiceless", "nasal": "0"},
    "G": {"category": "consonant", "place": "velar", "manner": "stop", "voicing": "voiced", "nasal": "0"},
    "Q": {"category": "consonant", "place": "uvular", "manner": "stop", "voicing": "voiceless", "nasal": "0"},
    "DX": {"category": "consonant", "place": "alveolar", "manner": "tap", "voicing": "voiced", "nasal": "0"},

    # Fricatives and affricates
    "F": {"category": "consonant", "place": "labiodental", "manner": "fricative", "voicing": "voiceless", "nasal": "0"},
    "V": {"category": "consonant", "place": "labiodental", "manner": "fricative", "voicing": "voiced", "nasal": "0"},
    "TH": {"category": "consonant", "place": "dental", "manner": "fricative", "voicing": "voiceless", "nasal": "0"},
    "DH": {"category": "consonant", "place": "dental", "manner": "fricative", "voicing": "voiced", "nasal": "0"},
    "S": {"category": "consonant", "place": "alveolar", "manner": "fricative", "voicing": "voiceless", "sibilant": "1", "nasal": "0"},
    "Z": {"category": "consonant", "place": "alveolar", "manner": "fricative", "voicing": "voiced", "sibilant": "1", "nasal": "0"},
    "SH": {"category": "consonant", "place": "postalveolar", "manner": "fricative", "voicing": "voiceless", "sibilant": "1", "nasal": "0"},
    "ZH": {"category": "consonant", "place": "postalveolar", "manner": "fricative", "voicing": "voiced", "sibilant": "1", "nasal": "0"},
    "HH": {"category": "consonant", "place": "glottal", "manner": "fricative", "voicing": "voiceless", "nasal": "0"},
    "CH": {"category": "consonant", "place": "postalveolar", "manner": "affricate", "voicing": "voiceless", "sibilant": "1", "nasal": "0"},
    "JH": {"category": "consonant", "place": "postalveolar", "manner": "affricate", "voicing": "voiced", "sibilant": "1", "nasal": "0"},

    # Nasals
    "M": {"category": "consonant", "place": "bilabial", "manner": "nasal", "voicing": "voiced", "nasal": "1"},
    "N": {"category": "consonant", "place": "alveolar", "manner": "nasal", "voicing": "voiced", "nasal": "1"},
    "NG": {"category": "consonant", "place": "velar", "manner": "nasal", "voicing": "voiced", "nasal": "1"},

    # Liquids and glides
    "L": {"category": "consonant", "place": "alveolar", "manner": "liquid", "voicing": "voiced", "lateral": "1", "nasal": "0"},
    "R": {"category": "consonant", "place": "postalveolar", "manner": "liquid", "voicing": "voiced", "nasal": "0"},
    "W": {"category": "consonant", "place": "labial-velar", "manner": "glide", "voicing": "voiced", "rounding": "rounded", "nasal": "0"},
    "Y": {"category": "consonant", "place": "palatal", "manner": "glide", "voicing": "voiced", "nasal": "0"},
}

# Grapheme approximations to use when CMU pronunciations are unavailable. Each
# entry maps a letter or digraph to one or more ARPABET phonemes so the feature
# lookup stays consistent with the table above.
GRAPHEME_TO_PHONEMES: Dict[str, Tuple[str, ...]] = {
    "A": ("AE",),
    "E": ("EH",),
    "I": ("IH",),
    "O": ("AO",),
    "U": ("UH",),
    "Y": ("IY",),
    "OO": ("UW",),
    "EE": ("IY",),
    "OU": ("AW",),
    "OW": ("OW",),
    "AI": ("EY",),
    "AY": ("AY",),
    "EA": ("IY",),
    "B": ("B",),
    "C": ("K",),
    "D": ("D",),
    "F": ("F",),
    "G": ("G",),
    "H": ("HH",),
    "J": ("JH",),
    "K": ("K",),
    "L": ("L",),
    "M": ("M",),
    "N": ("N",),
    "P": ("P",),
    "Q": ("K",),
    "R": ("R",),
    "S": ("S",),
    "T": ("T",),
    "V": ("V",),
    "W": ("W",),
    "X": ("K", "S"),
    "Z": ("Z",),
    "NG": ("NG",),
    "SH": ("SH",),
    "TH": ("TH",),
    "PH": ("F",),
    "CH": ("CH",),
    "GH": ("G",),
    "CK": ("K",),
}

class EnhancedPhoneticAnalyzer:
    """
    Enhanced phonetic analysis system for superior rhyme detection
    Implements research-backed phonetic similarity algorithms
    """

    def __init__(
        self,
        cmu_loader: Optional[CMUDictLoader] = None,
        rarity_map: Optional[WordRarityMap] = None,
    ):
        # Initialize phonetic analysis components
        self.cmu_loader = cmu_loader or DEFAULT_CMU_LOADER
        self.rarity_map = rarity_map or DEFAULT_RARITY_MAP
        self.phonetic_weights = self._initialize_phonetic_weights()
        self.variant_score_weights = self._initialize_variant_score_weights()

        self._cache_lock = threading.RLock()
        self._max_cache_entries = 512
        self._pronunciation_cache: OrderedDict[str, Tuple[Tuple[str, ...], ...]] = OrderedDict()
        self._rhyme_tail_cache: OrderedDict[
            Tuple[str, Tuple[Tuple[str, ...], ...]], Tuple[Tuple[str, ...], ...]
        ] = OrderedDict()
        self._similarity_cache: OrderedDict[Tuple[str, str], SlantScore] = OrderedDict()
        self._phrase_component_cache: OrderedDict[
            Tuple[str, Optional[int]], PhraseComponents
        ] = OrderedDict()
        self._logger = get_logger(__name__).bind(component="phonetic_analyzer")
        self._logger.info(
            "Enhanced phonetic analyzer initialised",
            context={
                "cmu_loader": type(self.cmu_loader).__name__,
                "rarity_map": type(self.rarity_map).__name__,
            },
        )
    
    def _initialize_phonetic_weights(self) -> Dict[str, float]:
        """Initialize weights for different phonetic features"""
        return {
            'rhyme_tail': 0.45,         # Highest weight when phoneme tails are available
            'ending_sounds': 0.2,       # Orthographic back-off for rime comparisons
            'vowel_sounds': 0.2,        # Vowel similarity (feature-driven)
            'consonant_clusters': 0.1,  # Coda and texture alignment
            'syllable_structure': 0.05, # Secondary importance
        }

    def _initialize_variant_score_weights(self) -> Dict[str, float]:
        """Default weight blend used when combining variant scores."""

        return {
            'phonetic': 0.55,
            'prosody': 0.25,
            'fluency': 0.15,
            'rarity': 0.05,
        }

    def generate_constrained_phrases(
        self, base_word: str, rhyme_keys: Iterable[str] = ()
    ) -> List[PhraseCandidate]:
        """Generate multi-word phrases biased to end with ``base_word``.

        This method exposes a constrained beam-search generator that fills
        lightweight templates with curated modifiers while ensuring the final
        token belongs to the supplied end-word inventory.
        """

        allowed_end_words: List[str] = []
        seen: Set[str] = set()

        normalized_base = str(base_word or "").strip()
        if normalized_base:
            seen.add(normalized_base.lower())
            allowed_end_words.append(normalized_base)

        loader = getattr(self, "cmu_loader", None)
        if loader is not None:
            for key in rhyme_keys:
                if not key:
                    continue
                tokens = [token for token in str(key).split() if token]
                if not tokens:
                    continue
                try:
                    matches = loader.find_words_by_phonemes(tokens, limit=6)
                except Exception:
                    continue
                for match in matches:
                    cleaned = str(match or "").strip()
                    if not cleaned:
                        continue
                    lowered = cleaned.lower()
                    if lowered in seen:
                        continue
                    seen.add(lowered)
                    allowed_end_words.append(cleaned)

        if not allowed_end_words:
            return []

        generated = generate_phrases_for_endwords(normalized_base, allowed_end_words)
        enriched: List[PhraseCandidate] = []
        for candidate in generated:
            if not isinstance(candidate, PhraseCandidate):
                continue
            enriched.append(candidate)
        return enriched

    # ------------------------------------------------------------------
    # Feature utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_phoneme_symbol(symbol: str) -> str:
        base = re.sub(r"\d", "", symbol or "")
        return base.strip().upper()

    @classmethod
    def _lookup_phoneme_features(cls, symbol: str) -> Optional[Dict[str, str]]:
        key = cls._normalize_phoneme_symbol(symbol)
        return PHONEME_FEATURES.get(key)

    @classmethod
    def _phoneme_feature_similarity(cls, symbol_a: str, symbol_b: str) -> float:
        if not symbol_a or not symbol_b:
            return 0.0

        features_a = cls._lookup_phoneme_features(symbol_a)
        features_b = cls._lookup_phoneme_features(symbol_b)

        if features_a is None or features_b is None:
            # Fall back to a light-weight string comparison if we do not have
            # feature coverage for this pair.
            return difflib.SequenceMatcher(
                None,
                cls._normalize_phoneme_symbol(symbol_a),
                cls._normalize_phoneme_symbol(symbol_b),
            ).ratio()

        keys = set(features_a) | set(features_b)
        if not keys:
            return 0.0

        mismatches = sum(1 for key in keys if features_a.get(key) != features_b.get(key))
        return max(0.0, 1.0 - (mismatches / len(keys)))

    @classmethod
    def _phoneme_sequence_similarity(
        cls,
        sequence_a: Iterable[str],
        sequence_b: Iterable[str],
        emphasize_first: bool = False,
    ) -> float:
        seq_a = [cls._normalize_phoneme_symbol(symbol) for symbol in sequence_a]
        seq_b = [cls._normalize_phoneme_symbol(symbol) for symbol in sequence_b]

        if not seq_a and not seq_b:
            return 1.0
        if not seq_a or not seq_b:
            return 0.0

        max_len = max(len(seq_a), len(seq_b))
        weighted_total = 0.0
        weight_sum = 0.0

        for index in range(max_len):
            phoneme_a = seq_a[index] if index < len(seq_a) else ""
            phoneme_b = seq_b[index] if index < len(seq_b) else ""
            weight = 1.5 if emphasize_first and index == 0 else 1.0
            weight_sum += weight
            if phoneme_a and phoneme_b:
                weighted_total += cls._phoneme_feature_similarity(phoneme_a, phoneme_b) * weight

        return weighted_total / weight_sum if weight_sum else 0.0

    @staticmethod
    def _grapheme_cluster_to_phonemes(cluster: str) -> List[str]:
        result: List[str] = []
        i = 0
        upper_cluster = (cluster or "").upper()
        while i < len(upper_cluster):
            matched = False
            for span in (2, 1):
                segment = upper_cluster[i : i + span]
                if segment in GRAPHEME_TO_PHONEMES:
                    result.extend(GRAPHEME_TO_PHONEMES[segment])
                    i += span
                    matched = True
                    break
            if not matched:
                # If we cannot resolve the cluster, advance to prevent loops
                i += 1
        return result

    @classmethod
    def _approximate_spelling_coda(cls, word: str) -> List[str]:
        lower = re.sub(r"[^a-z]", "", word.lower())
        if not lower:
            return []
        match = re.search(r"[bcdfghjklmnpqrstvwxyz]+$", lower)
        if not match:
            return []
        return cls._grapheme_cluster_to_phonemes(match.group(0))

    @classmethod
    def _approximate_spelling_vowel(cls, word: str) -> Optional[str]:
        lower = re.sub(r"[^a-z]", "", word.lower())
        if not lower:
            return None
        vowels = re.findall(r"[aeiouy]+", lower)
        if not vowels:
            return None
        cluster = vowels[-1]
        phonemes = cls._grapheme_cluster_to_phonemes(cluster)
        return phonemes[-1] if phonemes else None

    def _trim_cache(self, cache: OrderedDict) -> None:
        """Ensure ``cache`` respects the configured maximum size."""

        if self._max_cache_entries <= 0:
            cache.clear()
            return

        while len(cache) > self._max_cache_entries:
            cache.popitem(last=False)

    def get_phrase_components(
        self,
        phrase: str,
        cmu_loader: Optional[CMUDictLoader] = None,
    ) -> PhraseComponents:
        """Return memoized ``PhraseComponents`` for ``phrase``.

        Results are cached per CMU loader instance so repeated lookups during a
        single request can reuse earlier phonetic analysis without recomputing
        tokenisation and syllable estimates.
        """

        loader = cmu_loader or getattr(self, "cmu_loader", None)
        cache_key = ((phrase or "").strip().lower(), id(loader) if loader else None)

        with self._cache_lock:
            cached = self._phrase_component_cache.get(cache_key)
            if cached is not None:
                self._phrase_component_cache.move_to_end(cache_key)
                return cached

        components = extract_phrase_components(phrase or "", loader)

        with self._cache_lock:
            existing = self._phrase_component_cache.get(cache_key)
            if existing is not None:
                self._phrase_component_cache.move_to_end(cache_key)
                return existing

            self._phrase_component_cache[cache_key] = components
            self._trim_cache(self._phrase_component_cache)

        return components

    def clear_cached_results(self) -> None:
        """Reset memoized analyzer state.

        Exposes a lightweight hook so services can explicitly drop cached
        phonetic computations when upstream resources (e.g. CMU dictionaries)
        change within long-running processes.
        """

        self._logger.info("Clearing phonetic analyzer caches")

        with self._cache_lock:
            self._pronunciation_cache.clear()
            self._rhyme_tail_cache.clear()
            self._similarity_cache.clear()
            self._phrase_component_cache.clear()
    
    def get_slant_score(self, word1: str, word2: str) -> SlantScore:
        """Return a cached ``SlantScore`` for ``word1`` and ``word2``."""

        if not word1 or not word2:
            return SlantScore.empty()

        clean1 = self._clean_word(word1)
        clean2 = self._clean_word(word2)

        if not clean1 or not clean2:
            return SlantScore.empty()

        cache_key = (clean1, clean2) if clean1 <= clean2 else (clean2, clean1)
        with self._cache_lock:
            cached = self._similarity_cache.get(cache_key)
            if cached is not None:
                self._similarity_cache.move_to_end(cache_key)
                return cached

        pronunciations1 = self._get_pronunciation_variants(clean1)
        pronunciations2 = self._get_pronunciation_variants(clean2)

        if pronunciations1:
            try:
                self._get_rhyme_tails(clean1, pronunciations1)
            except Exception:
                pass
        if pronunciations2:
            try:
                self._get_rhyme_tails(clean2, pronunciations2)
            except Exception:
                pass

        slant = score_pair(self, clean1, clean2)

        with self._cache_lock:
            self._similarity_cache[cache_key] = slant
            self._trim_cache(self._similarity_cache)

        return slant

    def get_phonetic_similarity(self, word1: str, word2: str) -> float:
        """
        Calculate phonetic similarity between two words
        Returns score from 0.0 to 1.0
        """

        return float(self.get_slant_score(word1, word2).total)
    
    def _phrase_components(self, word: str) -> PhraseComponents:
        loader = getattr(self, "cmu_loader", None)
        return self.get_phrase_components(word or "", loader)

    def _clean_word(self, word: str) -> str:
        """Clean and normalize word or phrase for phonetic analysis."""

        components = self._phrase_components(word)
        return components.normalized_phrase.strip()
    
    def _calculate_ending_similarity(self, word1: str, word2: str) -> float:
        """Calculate similarity of word endings (most important for rhymes)"""
        if len(word1) < 2 or len(word2) < 2:
            return 0.0
        
        # Check various ending lengths
        max_ending_length = min(4, min(len(word1), len(word2)))
        best_similarity = 0.0
        
        for length in range(2, max_ending_length + 1):
            end1 = word1[-length:]
            end2 = word2[-length:]
            
            if end1 == end2:
                # Exact match gets bonus based on length
                similarity = 0.8 + (length * 0.05)
                best_similarity = max(best_similarity, similarity)
            else:
                # Partial match using string similarity
                similarity = difflib.SequenceMatcher(None, end1, end2).ratio() * 0.7
                best_similarity = max(best_similarity, similarity)
        
        return best_similarity
    
    def _calculate_vowel_similarity(
        self,
        word1: str,
        word2: str,
        *,
        pronunciations1: Optional[List[List[str]]] = None,
        pronunciations2: Optional[List[List[str]]] = None,
        rhyme_tails1: Optional[List[List[str]]] = None,
        rhyme_tails2: Optional[List[List[str]]] = None,
    ) -> float:
        """Calculate vowel pattern similarity using phoneme features."""

        vowel_candidates1: List[str] = []
        vowel_candidates2: List[str] = []

        for tails, collector in (
            (rhyme_tails1 or [], vowel_candidates1),
            (rhyme_tails2 or [], vowel_candidates2),
        ):
            for tail in tails:
                if not tail:
                    continue
                normalized = self._normalize_phoneme_symbol(tail[0])
                if normalized in VOWEL_PHONEMES:
                    collector.append(normalized)

        if not vowel_candidates1 and pronunciations1:
            for pron in pronunciations1:
                for phone in reversed(pron):
                    normalized = self._normalize_phoneme_symbol(phone)
                    if normalized in VOWEL_PHONEMES:
                        vowel_candidates1.append(normalized)
                        break

        if not vowel_candidates2 and pronunciations2:
            for pron in pronunciations2:
                for phone in reversed(pron):
                    normalized = self._normalize_phoneme_symbol(phone)
                    if normalized in VOWEL_PHONEMES:
                        vowel_candidates2.append(normalized)
                        break

        if vowel_candidates1 and vowel_candidates2:
            scores = [
                self._phoneme_feature_similarity(v1, v2)
                for v1 in vowel_candidates1
                for v2 in vowel_candidates2
            ]
            if scores:
                return max(scores)

        # Fall back to orthographic approximation when pronunciations are not
        # available.
        vowel1 = self._approximate_spelling_vowel(word1)
        vowel2 = self._approximate_spelling_vowel(word2)

        if vowel1 and vowel2:
            return self._phoneme_feature_similarity(vowel1, vowel2)

        return 0.0
    
    def _calculate_consonant_similarity(
        self,
        word1: str,
        word2: str,
        *,
        pronunciations1: Optional[List[List[str]]] = None,
        pronunciations2: Optional[List[List[str]]] = None,
        rhyme_tails1: Optional[List[List[str]]] = None,
        rhyme_tails2: Optional[List[List[str]]] = None,
    ) -> float:
        """Calculate consonant pattern similarity using feature distances."""

        coda_variants1: List[List[str]] = []
        coda_variants2: List[List[str]] = []

        for tails, collector in (
            (rhyme_tails1 or [], coda_variants1),
            (rhyme_tails2 or [], coda_variants2),
        ):
            for tail in tails:
                normalized_tail = [self._normalize_phoneme_symbol(p) for p in tail[1:]]
                if normalized_tail:
                    collector.append(normalized_tail)
                elif tail:
                    collector.append([])

        if not coda_variants1 and pronunciations1:
            for pron in pronunciations1:
                coda: List[str] = []
                for phone in reversed(pron):
                    normalized = self._normalize_phoneme_symbol(phone)
                    if normalized in VOWEL_PHONEMES:
                        break
                    if normalized:
                        coda.append(normalized)
                coda.reverse()
                collector = coda if coda else []
                coda_variants1.append(collector)

        if not coda_variants2 and pronunciations2:
            for pron in pronunciations2:
                coda: List[str] = []
                for phone in reversed(pron):
                    normalized = self._normalize_phoneme_symbol(phone)
                    if normalized in VOWEL_PHONEMES:
                        break
                    if normalized:
                        coda.append(normalized)
                coda.reverse()
                collector = coda if coda else []
                coda_variants2.append(collector)

        if coda_variants1 and coda_variants2:
            scores: List[float] = []
            for seq1 in coda_variants1:
                for seq2 in coda_variants2:
                    if not seq1 and not seq2:
                        scores.append(1.0)
                    else:
                        scores.append(self._phoneme_sequence_similarity(seq1, seq2))
            if scores:
                return max(scores)

        # Fall back to orthographic approximation when CMU data is missing.
        coda1 = self._approximate_spelling_coda(word1)
        coda2 = self._approximate_spelling_coda(word2)

        if not coda1 and not coda2:
            return 1.0
        if not coda1 or not coda2:
            return 0.4

        return self._phoneme_sequence_similarity(coda1, coda2)
    
    def _calculate_syllable_similarity(self, word1: str, word2: str) -> float:
        """Calculate syllable structure similarity"""
        syllable_count1 = self._count_syllables(word1)
        syllable_count2 = self._count_syllables(word2)
        
        if syllable_count1 == syllable_count2:
            return 1.0
        elif abs(syllable_count1 - syllable_count2) == 1:
            return 0.7
        else:
            return 0.3
    
    def _count_syllables(self, word: str) -> int:
        """Estimate syllable count in a word"""

        components = self._phrase_components(word)
        return max(1, components.total_syllables)

    # ------------------------------------------------------------------
    # Research-driven feature extraction
    # ------------------------------------------------------------------

    def _get_pronunciation_variants(self, word: str) -> List[List[str]]:
        loader = getattr(self, "cmu_loader", None)
        if loader is None:
            return []

        components = self._phrase_components(word)
        cache_key = components.anchor or components.normalized_phrase
        if not cache_key:
            cache_key = (word or "").strip().lower()
        cache_key = str(cache_key).strip().lower()

        with self._cache_lock:
            cached = self._pronunciation_cache.get(cache_key)
            if cached is not None:
                self._pronunciation_cache.move_to_end(cache_key)
                return [list(phones) for phones in cached]

        if components.anchor_pronunciations:
            pronunciations = [list(phones) for phones in components.anchor_pronunciations]
        else:
            lookup_word = components.anchor
            if not lookup_word:
                lookup_word = re.sub(r"\s+", "", components.normalized_phrase)

            pronunciations = []
            if lookup_word:
                try:
                    pronunciations = loader.get_pronunciations(lookup_word)
                except Exception:
                    pronunciations = []

            if not pronunciations and pronouncing is not None and lookup_word:
                try:
                    pronunciations = [
                        phones.split()
                        for phones in pronouncing.phones_for_word(lookup_word)
                    ]
                except Exception:
                    pronunciations = []

        immutable = tuple(tuple(phone for phone in variant) for variant in pronunciations)
        with self._cache_lock:
            self._pronunciation_cache[cache_key] = immutable
            self._trim_cache(self._pronunciation_cache)

        return [list(variant) for variant in immutable]

    @staticmethod
    def _stress_signature_from_phones(phones: List[str]) -> str:
        signature_parts: List[str] = []
        for phone in phones:
            stress_marker = re.search(r"(\d)", phone)
            if stress_marker:
                signature_parts.append(stress_marker.group(1))
        return "".join(signature_parts)

    @staticmethod
    def _strip_stress_markers(phones: List[str]) -> List[str]:
        return [re.sub(r"\d", "", phone) for phone in phones]

    @staticmethod
    def _extract_phoneme_tail(phones: Iterable[str]) -> List[str]:
        sequence = list(phones)
        if not sequence:
            return []

        last_stress_index: Optional[int] = None
        last_vowel_index: Optional[int] = None

        for index, phone in enumerate(sequence):
            base = re.sub(r"\d", "", phone).upper()
            if base not in VOWEL_PHONEMES:
                continue

            last_vowel_index = index
            if re.search(r"[12]", phone):
                last_stress_index = index

        anchor_index = last_stress_index if last_stress_index is not None else last_vowel_index
        if anchor_index is None:
            return []

        return sequence[anchor_index:]

    @staticmethod
    def _last_vowel_sound(
        pronunciations: List[List[str]],
        fallback_word: str,
    ) -> str:
        """Return the final vowel phoneme (stress stripped) for a word."""

        for phones in pronunciations:
            for phone in reversed(phones):
                base = re.sub(r"\d", "", phone).upper()
                if base in VOWEL_PHONEMES:
                    return base

        vowels = re.findall(r"[aeiou]+", fallback_word.lower())
        return vowels[-1] if vowels else ""

    @staticmethod
    def _consonant_coda_signature(
        pronunciations: List[List[str]],
        fallback_word: str,
    ) -> str:
        """Return a compact representation of the terminal consonant sounds."""

        for phones in pronunciations:
            consonants: List[str] = []
            encountered_vowel = False
            for phone in reversed(phones):
                base = re.sub(r"\d", "", phone).upper()
                if base in VOWEL_PHONEMES:
                    encountered_vowel = True
                    break
                if base:
                    consonants.append(base)

            if encountered_vowel:
                if consonants:
                    consonants.reverse()
                    return "-".join(consonants)
                return ""

        fallback_tail = re.sub(r"[aeiou]", "", fallback_word.lower())
        return fallback_tail[-2:] if fallback_tail else ""

    @staticmethod
    def _consonant_onset_signature(
        pronunciations: List[List[str]],
        fallback_word: str,
    ) -> str:
        """Return consonant sounds that directly precede the final vowel."""

        for phones in pronunciations:
            last_vowel_index: Optional[int] = None
            for index, phone in enumerate(phones):
                base = re.sub(r"\d", "", phone).upper()
                if base in VOWEL_PHONEMES:
                    last_vowel_index = index
            if last_vowel_index is None:
                continue

            onset: List[str] = []
            for index in range(last_vowel_index - 1, -1, -1):
                base = re.sub(r"\d", "", phones[index]).upper()
                if base in VOWEL_PHONEMES:
                    break
                if base:
                    onset.append(base)
                else:
                    break

            if onset:
                onset.reverse()
                return "-".join(onset)
            return ""

        fallback_lower = fallback_word.lower()
        vowel_matches = list(re.finditer(r"[aeiou]+", fallback_lower))
        if not vowel_matches:
            return ""
        last_match = vowel_matches[-1]
        prefix = fallback_lower[: last_match.start()]
        consonant_match = re.search(r"[bcdfghjklmnpqrstvwxyz]+$", prefix)
        return consonant_match.group(0) if consonant_match else ""

    def _get_rhyme_tails(
        self,
        word: str,
        pronunciations: List[List[str]],
    ) -> List[List[str]]:
        loader = getattr(self, "cmu_loader", None)
        tails: List[List[str]] = []

        cache_key = (
            (word or "").strip().lower(),
            tuple(tuple(phone for phone in seq) for seq in pronunciations),
        )
        with self._cache_lock:
            cached = self._rhyme_tail_cache.get(cache_key)
            if cached is not None:
                self._rhyme_tail_cache.move_to_end(cache_key)
                return [list(item) for item in cached]

        if loader is not None:
            try:
                rhyme_parts = loader.get_rhyme_parts(word)
            except Exception:
                rhyme_parts = set()

            for part in rhyme_parts:
                tail = part.split()
                if tail:
                    tails.append(tail)

        if not tails:
            for phones in pronunciations:
                tail = self._extract_phoneme_tail(phones)
                if tail:
                    tails.append(tail)

        unique_tails: List[List[str]] = []
        seen: Set[Tuple[str, ...]] = set()

        for tail in tails:
            key = tuple(tail)
            if key in seen:
                continue
            seen.add(key)
            unique_tails.append(list(tail))

        immutable = tuple(tuple(phone for phone in tail) for tail in unique_tails)
        with self._cache_lock:
            self._rhyme_tail_cache[cache_key] = immutable
            self._trim_cache(self._rhyme_tail_cache)

        return [list(tail) for tail in immutable]

    @classmethod
    def _phoneme_tail_similarity(cls, tail1: Iterable[str], tail2: Iterable[str]) -> float:
        seq1 = list(tail1)
        seq2 = list(tail2)

        if not seq1 or not seq2:
            return 0.0

        if seq1 == seq2:
            return 1.0

        base1 = [cls._normalize_phoneme_symbol(symbol) for symbol in seq1]
        base2 = [cls._normalize_phoneme_symbol(symbol) for symbol in seq2]

        base_similarity = cls._phoneme_sequence_similarity(base1, base2, emphasize_first=True)
        coda_similarity = cls._phoneme_sequence_similarity(base1[1:], base2[1:])

        vowel_similarity = 0.0
        if base1 and base2:
            vowel_similarity = cls._phoneme_feature_similarity(base1[0], base2[0])

        stress_bonus = 0.05 if seq1 and seq2 and seq1[0] == seq2[0] else 0.0

        similarity = (base_similarity * 0.6) + (coda_similarity * 0.2) + (vowel_similarity * 0.2) + stress_bonus
        return max(0.0, min(similarity, 1.0))

    @staticmethod
    def _vowel_skeleton(word: str) -> str:
        vowels = re.findall(r"[aeiou]+", word)
        return "-".join(vowels)

    @staticmethod
    def _consonant_tail(word: str) -> str:
        tail = re.sub(r"[aeiou]", "", word)
        return tail[-3:]

    @staticmethod
    def _meter_hint_from_stress(stress_pattern: str) -> Dict[str, Optional[str]]:
        """Return a simple metrical interpretation for a stress signature."""

        if not stress_pattern:
            return {"foot": None, "description": None}

        normalized = stress_pattern.replace("2", "1")
        foot: Optional[str] = None
        description: Optional[str] = None

        if len(normalized) == 1:
            foot = "monosyllabic"
            description = (
                "Single stressed syllable"
                if normalized == "1"
                else "Single unstressed syllable"
            )
        elif normalized.startswith("01"):
            foot = "iamb"
            description = "Likely iambic opening (unstressed → stressed)"
        elif normalized.startswith("10"):
            foot = "trochee"
            description = "Likely trochaic opening (stressed → unstressed)"
        elif normalized.startswith("001"):
            foot = "anapest"
            description = "Hints at anapestic foot (two unstressed then stressed)"
        elif normalized.startswith("100"):
            foot = "dactyl"
            description = "Hints at dactylic foot (stressed then two unstressed)"
        elif normalized.startswith("11"):
            foot = "spondee"
            description = "Double stress opening (spondaic cadence)"
        else:
            description = "Mixed stress profile"

        return {"foot": foot, "description": description}

    @staticmethod
    def _assonance_score(word1: str, word2: str) -> float:
        vowels1 = re.findall(r"[aeiou]+", word1)
        vowels2 = re.findall(r"[aeiou]+", word2)
        if not vowels1 or not vowels2:
            return 0.0
        counter1 = Counter(vowels1)
        counter2 = Counter(vowels2)
        shared = sum(min(counter1[v], counter2[v]) for v in counter1)
        total = max(sum(counter1.values()), sum(counter2.values()), 1)
        return shared / total

    @staticmethod
    def _consonance_score(word1: str, word2: str) -> float:
        consonants1 = re.sub(r"[^a-z']", "", re.sub(r"[aeiou]", "", word1))
        consonants2 = re.sub(r"[^a-z']", "", re.sub(r"[aeiou]", "", word2))
        if not consonants1 or not consonants2:
            return 0.0
        counter1 = Counter(consonants1)
        counter2 = Counter(consonants2)
        shared = sum(min(counter1[c], counter2[c]) for c in counter1)
        total = max(len(consonants1), len(consonants2))
        if total == 0:
            return 0.0
        return shared / total

    def _derive_bradley_device(
        self,
        similarity: float,
        rhyme_type: str,
        assonance: float,
        consonance: float,
        syllables: Tuple[int, int],
    ) -> str:
        """Classify rhyme device using rap poetics scholarship."""

        if similarity >= 0.95 or rhyme_type == "perfect":
            return "pure rhyme"
        if assonance >= 0.7 and consonance >= 0.5:
            return "compound rhyme"
        if assonance >= 0.7:
            return "assonance"
        if consonance >= 0.6:
            return "consonance"
        if min(syllables) >= 2 and similarity >= 0.7:
            return "multisyllabic"
        if rhyme_type in {"slant", "eye"}:
            return "slant rhyme"
        return "resonant wordplay"

    def build_feature_profile(
        self,
        source_word: str,
        target_word: str,
        similarity: Optional[float] = None,
        rhyme_type: Optional[str] = None,
    ) -> Optional[RhymeFeatureProfile]:
        """Create a research-aligned feature profile for a rhyme pair."""

        if not source_word or not target_word:
            return None

        normalized_source = source_word.lower()
        normalized_target = target_word.lower()

        syllables_source = self._count_syllables(normalized_source)
        syllables_target = self._count_syllables(normalized_target)
        syllable_span = (syllables_source, syllables_target)

        pronunciations_source = self._get_pronunciation_variants(normalized_source)
        pronunciations_target = self._get_pronunciation_variants(normalized_target)

        stress_source = ""
        stress_target = ""

        if pronunciations_source:
            stress_source = self._stress_signature_from_phones(pronunciations_source[0])
        if pronunciations_target:
            stress_target = self._stress_signature_from_phones(pronunciations_target[0])

        last_vowel_source = self._last_vowel_sound(pronunciations_source, normalized_source)
        last_vowel_target = self._last_vowel_sound(pronunciations_target, normalized_target)
        consonant_coda_source = self._consonant_coda_signature(
            pronunciations_source, normalized_source
        )
        consonant_coda_target = self._consonant_coda_signature(
            pronunciations_target, normalized_target
        )
        consonant_onset_source = self._consonant_onset_signature(
            pronunciations_source, normalized_source
        )
        consonant_onset_target = self._consonant_onset_signature(
            pronunciations_target, normalized_target
        )

        stress_alignment = 0.0
        if stress_source and stress_target:
            matches = sum(1 for a, b in zip(stress_source, stress_target) if a == b)
            stress_alignment = matches / max(len(stress_source), len(stress_target))

        vowel_skeleton_source = self._vowel_skeleton(normalized_source)
        vowel_skeleton_target = self._vowel_skeleton(normalized_target)
        consonant_tail_source = self._consonant_tail(normalized_source)
        consonant_tail_target = self._consonant_tail(normalized_target)

        assonance = self._assonance_score(normalized_source, normalized_target)
        consonance_value = self._consonance_score(normalized_source, normalized_target)

        internal_rhyme_base = (assonance + consonance_value) / 2.0
        if min(syllable_span) >= 2:
            internal_rhyme_base += 0.2
        internal_rhyme_score = min(1.0, max(0.0, internal_rhyme_base))

        resolved_rhyme_type = rhyme_type or self.classify_rhyme_type(
            normalized_source,
            normalized_target,
            similarity or self.get_phonetic_similarity(normalized_source, normalized_target),
        )

        vowel_match = bool(
            last_vowel_source
            and last_vowel_target
            and last_vowel_source == last_vowel_target
        )
        coda_match = consonant_coda_source == consonant_coda_target
        onset_match = consonant_onset_source == consonant_onset_target
        if vowel_match:
            if coda_match and onset_match:
                resolved_rhyme_type = "perfect"
            else:
                resolved_rhyme_type = "slant"

        bradley_device = self._derive_bradley_device(
            similarity or 0.0,
            resolved_rhyme_type,
            assonance,
            consonance_value,
            syllable_span,
        )

        return RhymeFeatureProfile(
            source_word=normalized_source,
            target_word=normalized_target,
            syllable_span=syllable_span,
            stress_alignment=stress_alignment,
            stress_pattern_source=stress_source,
            stress_pattern_target=stress_target,
            vowel_skeleton_source=vowel_skeleton_source,
            vowel_skeleton_target=vowel_skeleton_target,
            consonant_tail_source=consonant_tail_source,
            consonant_tail_target=consonant_tail_target,
            last_vowel_sound_source=last_vowel_source,
            last_vowel_sound_target=last_vowel_target,
            consonant_coda_source=consonant_coda_source,
            consonant_coda_target=consonant_coda_target,
            consonant_onset_source=consonant_onset_source,
            consonant_onset_target=consonant_onset_target,
            assonance_score=assonance,
            consonance_score=consonance_value,
            internal_rhyme_score=internal_rhyme_score,
            bradley_device=bradley_device,
            rhyme_type=resolved_rhyme_type,
        )

    def describe_word(self, word: str) -> Dict[str, Any]:
        """Return a phonetic profile for a single word."""

        components = self._phrase_components(word or "")
        normalized = components.normalized_phrase.strip()
        profile: Dict[str, Any] = {
            "word": word,
            "normalized": normalized,
            "syllables": None,
            "stress_pattern": "",
            "stress_pattern_display": "",
            "meter_hint": None,
            "metrical_foot": None,
            "vowel_skeleton": "",
            "consonant_tail": "",
            "pronunciations": [],
            "tokens": components.normalized_tokens,
            "token_syllables": components.syllable_counts,
            "anchor_word": components.anchor,
            "anchor_display": components.anchor_display or components.anchor,
        }

        if components.total_syllables:
            profile["syllables"] = components.total_syllables

        if not normalized:
            return profile

        # Build a composite stress pattern that accounts for every token in the
        # phrase. Previously we only surfaced the stress pattern for the anchor
        # word which meant multi-word entries dropped syllables from the
        # display.  We now look up pronunciations token-by-token and stitch the
        # stress signatures together so that each syllable is represented in the
        # final pattern.  This ensures UI elements such as the multi-word rhyme
        # cards report the complete cadence information.
        loader = getattr(self, "cmu_loader", None)
        token_stress_signatures: List[str] = []
        display_segments: List[str] = []

        for index, token in enumerate(components.normalized_tokens or []):
            if not token:
                continue

            pronunciations: List[List[str]] = []
            if loader is not None:
                try:
                    pronunciations = loader.get_pronunciations(token)
                except Exception:
                    pronunciations = []

            if not pronunciations and pronouncing is not None:
                try:
                    pronunciations = [
                        phones.split()
                        for phones in pronouncing.phones_for_word(token)
                        if phones
                    ]
                except Exception:
                    pronunciations = []

            stress_signature = ""
            for phones in pronunciations:
                try:
                    stress_signature = self._stress_signature_from_phones(list(phones))
                except Exception:
                    stress_signature = ""
                if stress_signature:
                    break

            if not stress_signature:
                syllable_estimate = 0
                if 0 <= index < len(components.syllable_counts):
                    syllable_estimate = int(components.syllable_counts[index] or 0)
                if syllable_estimate <= 0:
                    syllable_estimate = estimate_syllable_count(token)
                if syllable_estimate > 0:
                    stress_signature = "?" * syllable_estimate

            if stress_signature:
                token_stress_signatures.append(stress_signature)
                display_segments.append("-".join(stress_signature))

        if token_stress_signatures and not profile["stress_pattern"]:
            profile["stress_pattern"] = "".join(token_stress_signatures)
            profile["stress_pattern_display"] = " ".join(
                segment for segment in display_segments if segment
            )

        pronunciations = self._get_pronunciation_variants(word)
        if pronunciations:
            stripped_variants = []
            for phones in pronunciations[:2]:
                try:
                    stress_signature = self._stress_signature_from_phones(phones)
                except Exception:
                    stress_signature = ""
                if stress_signature and not profile["stress_pattern"]:
                    profile["stress_pattern"] = stress_signature
                    profile["stress_pattern_display"] = "-".join(stress_signature)
                stripped_variants.append(
                    " ".join(self._strip_stress_markers(phones))
                )
            profile["pronunciations"] = stripped_variants

        if profile["stress_pattern"]:
            meter_info = self._meter_hint_from_stress(profile["stress_pattern"])
            profile["meter_hint"] = meter_info.get("description")
            profile["metrical_foot"] = meter_info.get("foot")

        profile["vowel_skeleton"] = self._vowel_skeleton(normalized)
        profile["consonant_tail"] = self._consonant_tail(normalized)

        return profile

    def estimate_syllables(self, word: str) -> int:
        """Public helper for downstream modules that need syllable counts."""

        return self._count_syllables(word)

    def derive_rhyme_profile(
        self,
        source_word: str,
        target_word: str,
        similarity: Optional[float] = None,
        rhyme_type: Optional[str] = None,
    ) -> Optional[RhymeFeatureProfile]:
        """Public wrapper for compatibility with downstream modules."""

        profile = self.build_feature_profile(
            source_word,
            target_word,
            similarity=similarity,
            rhyme_type=rhyme_type,
        )
        if profile is None:
            return None
        return profile.as_dict()
    
    def classify_rhyme_type(self, word1: str, word2: str, similarity: float) -> str:
        """Classify the type of rhyme based on similarity score"""
        if similarity >= 0.95:
            return "perfect"
        elif similarity >= 0.85:
            return "near"
        elif similarity >= 0.75:
            return "slant"
        elif similarity >= 0.65:
            return "eye"  # visual rhyme
        else:
            return "weak"
    
    def analyze_rhyme_pattern(self, word1: str, word2: str) -> PhoneticMatch:
        """Comprehensive analysis of rhyme pattern between two words"""
        slant = self.get_slant_score(word1, word2)
        similarity = slant.total
        rhyme_type = self.classify_rhyme_type(word1, word2, similarity)

        # Normalize words once for feature calculations to ensure consistent scoring
        clean_word1 = self._clean_word(word1)
        clean_word2 = self._clean_word(word2)

        profile = self.build_feature_profile(
            clean_word1,
            clean_word2,
            similarity=similarity,
            rhyme_type=rhyme_type,
        )

        # Calculate detailed phonetic features
        features = {
            'ending_similarity': self._calculate_ending_similarity(clean_word1, clean_word2),
            'vowel_similarity': self._calculate_vowel_similarity(clean_word1, clean_word2),
            'consonant_similarity': self._calculate_consonant_similarity(clean_word1, clean_word2),
            'syllable_similarity': self._calculate_syllable_similarity(clean_word1, clean_word2),
            'slant_rime': slant.rime,
            'slant_vowel': slant.vowel,
            'slant_coda': slant.coda,
            'slant_penalty_stress': slant.stress_penalty,
            'slant_penalty_syllables': slant.syllable_penalty,
        }

        if profile is not None:
            features.update(
                {
                    'assonance_score': profile.assonance_score,
                    'consonance_score': profile.consonance_score,
                    'internal_rhyme_score': profile.internal_rhyme_score,
                }
            )
            if profile.rhyme_type:
                rhyme_type = profile.rhyme_type

        return PhoneticMatch(
            word1=word1,
            word2=word2,
            similarity_score=similarity,
            phonetic_features=features,
            rhyme_type=rhyme_type,
            feature_profile=profile,
            slant_score=slant,
            slant_tier=slant.tier,
        )

    def get_rhyme_candidates(self, target_word: str, word_list: List[str],
                           min_similarity: float = 0.7) -> List[PhoneticMatch]:
        """Find rhyme candidates from a word list"""
        candidates = []

        for word in word_list:
            if word.lower() != target_word.lower():
                match = self.analyze_rhyme_pattern(target_word, word)
                if match.slant_score and not passes_gate(match.slant_score):
                    continue
                if match.similarity_score >= min_similarity:
                    rarity = self.get_rarity_score(word)
                    combined = self.combine_similarity_and_rarity(match.similarity_score, rarity)
                    match.rarity_score = rarity
                    match.combined_score = combined
                    candidates.append(match)

        # Sort by combined uncommon-first score, falling back to similarity
        candidates.sort(key=lambda x: (x.combined_score, x.similarity_score), reverse=True)
        return candidates

    def get_rarity_score(self, word: str) -> float:
        rarity_map = getattr(self, "rarity_map", None) or DEFAULT_RARITY_MAP
        try:
            return float(rarity_map.get_rarity(word))
        except Exception:
            return DEFAULT_RARITY_MAP.get_rarity(word)

    def combine_similarity_and_rarity(self, similarity: float, rarity: float) -> float:
        return (
            similarity * RARITY_SIMILARITY_WEIGHT
            + rarity * RARITY_NOVELTY_WEIGHT
        )

    def update_rarity_from_database(self, db_path: Optional[Path | str]) -> bool:
        rarity_map = getattr(self, "rarity_map", None)
        if rarity_map is None:
            return False
        try:
            return bool(rarity_map.update_from_database(db_path))
        except Exception:
            return False


RhymeCandidate = Dict[str, Any]


_DIGIT_PATTERN = re.compile(r"\d")


@dataclass(frozen=True)
class PhraseRimeKeyInfo:
    """Container describing the rhyme keys associated with a phrase."""

    anchor_rhymes: Tuple[str, ...]
    backoff_keys: Tuple[str, ...]
    compound_phonemes: Tuple[Tuple[str, ...], ...]
    compound_strings: Tuple[str, ...]


def normalize_rime_key(value: Optional[str]) -> Optional[str]:
    """Return a canonical representation for rime key comparisons."""

    text = str(value or "").strip()
    if not text:
        return None

    tokens = [segment.strip().upper() for segment in text.split() if segment.strip()]
    if not tokens:
        return None

    return " ".join(tokens)


def collect_rhyme_parts(word: str, loader: Optional[CMUDictLoader]) -> Set[str]:
    """Return CMU rhyme parts for ``word`` with a pronouncing fallback."""

    parts: Set[str] = set()
    term = str(word or "").strip()
    if not term:
        return parts

    if loader is not None:
        try:
            parts.update(loader.get_rhyme_parts(term))
        except Exception:
            pass

    if not parts and pronouncing is not None:
        try:
            for phones in pronouncing.phones_for_word(term):
                rhyme_part = pronouncing.rhyming_part(phones)
                if rhyme_part:
                    parts.add(rhyme_part)
        except Exception:
            pass

    return parts


def phonetic_backoffs_from_parts(parts: Iterable[str]) -> Tuple[str, ...]:
    """Return progressively looser rhyme keys derived from ``parts``."""

    seen: Set[str] = set()
    ordered: List[str] = []

    for part in parts:
        cleaned = str(part or "").strip()
        if not cleaned:
            continue

        tokens = cleaned.split()
        for index in range(len(tokens)):
            segment = " ".join(tokens[index:])
            if segment and segment not in seen:
                seen.add(segment)
                ordered.append(segment)

        normalized = _DIGIT_PATTERN.sub("", cleaned)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            ordered.append(normalized)

        if tokens:
            vowel_only = tokens[-1]
            vowel_clean = _DIGIT_PATTERN.sub("", vowel_only)
            if vowel_clean and vowel_clean not in seen:
                seen.add(vowel_clean)
                ordered.append(vowel_clean)

    return tuple(ordered)


def phrase_rime_keys(
    components: PhraseComponents,
    loader: Optional[CMUDictLoader],
) -> PhraseRimeKeyInfo:
    """Return end-word and compound rhyme keys for ``components``."""

    anchor_word = components.anchor or (
        components.normalized_tokens[-1] if components.normalized_tokens else ""
    )

    anchor_rhymes = tuple(sorted(collect_rhyme_parts(anchor_word, loader)))
    backoff_keys = phonetic_backoffs_from_parts(anchor_rhymes)

    compound_phonemes: List[Tuple[str, ...]] = []
    compound_strings: List[str] = []

    if loader is not None and len(components.normalized_tokens) >= 2:
        previous_word = components.normalized_tokens[-2]
        try:
            previous_prons = loader.get_pronunciations(previous_word)
        except Exception:
            previous_prons = []

        if not previous_prons and pronouncing is not None:
            try:
                previous_prons = [
                    phones.split() for phones in pronouncing.phones_for_word(previous_word)
                ]
            except Exception:
                previous_prons = []

        anchor_prons = list(components.anchor_pronunciations or [])
        if not anchor_prons and anchor_word:
            try:
                anchor_prons = loader.get_pronunciations(anchor_word)
            except Exception:
                anchor_prons = []

        if not anchor_prons and pronouncing is not None and anchor_word:
            try:
                anchor_prons = [
                    phones.split() for phones in pronouncing.phones_for_word(anchor_word)
                ]
            except Exception:
                anchor_prons = []

        def _final_syllable(phones: List[str]) -> Tuple[str, ...]:
            if not phones:
                return tuple()
            start_index: Optional[int] = None
            for index in range(len(phones) - 1, -1, -1):
                base = _DIGIT_PATTERN.sub("", phones[index])
                if base in VOWEL_PHONEMES:
                    start_index = index
                    break
            if start_index is None:
                return tuple(phones)
            return tuple(phones[start_index:])

        seen_sequences: Set[Tuple[str, ...]] = set()

        for previous in previous_prons:
            previous_tail = _final_syllable(previous)
            if not previous_tail:
                continue
            previous_norm = tuple(
                _DIGIT_PATTERN.sub("", phone) for phone in previous_tail if phone
            )
            if not previous_norm:
                continue

            for anchor_phones in anchor_prons:
                if not anchor_phones:
                    continue

                tails: List[Tuple[str, ...]] = []
                anchor_syllable = _final_syllable(anchor_phones)
                anchor_norm = tuple(
                    _DIGIT_PATTERN.sub("", phone) for phone in anchor_syllable if phone
                )
                if anchor_norm:
                    tails.append(anchor_norm)

                anchor_full = tuple(
                    _DIGIT_PATTERN.sub("", phone) for phone in anchor_phones if phone
                )
                if anchor_full:
                    tails.append(anchor_full)

                for tail in tails:
                    sequence = previous_norm + tail
                    if not sequence or sequence in seen_sequences:
                        continue
                    seen_sequences.add(sequence)
                    compound_phonemes.append(sequence)
                    compound_strings.append(" ".join(sequence))

                    if len(sequence) >= 2 and sequence[-2] in {"M", "N"}:
                        assimilated = sequence[:-2] + ("N", "D", sequence[-1])
                        if assimilated not in seen_sequences:
                            seen_sequences.add(assimilated)
                            compound_phonemes.append(assimilated)
                            compound_strings.append(" ".join(assimilated))
                        prefixed = ("W",) + assimilated
                        if prefixed not in seen_sequences:
                            seen_sequences.add(prefixed)
                            compound_phonemes.append(prefixed)
                            compound_strings.append(" ".join(prefixed))
                    if len(sequence) >= 3 and sequence[-3] in {"M", "N"} and sequence[-2] in {"S", "Z"}:
                        assimilated = sequence[:-3] + ("N", "D", sequence[-1])
                        if assimilated not in seen_sequences:
                            seen_sequences.add(assimilated)
                            compound_phonemes.append(assimilated)
                            compound_strings.append(" ".join(assimilated))
                        prefixed = ("W",) + assimilated
                        if prefixed not in seen_sequences:
                            seen_sequences.add(prefixed)
                            compound_phonemes.append(prefixed)
                            compound_strings.append(" ".join(prefixed))

        if compound_strings:
            # Compound tails should also be available as back-off keys.
            merged = list(backoff_keys) + [key for key in compound_strings if key]
            backoff_keys = tuple(dict.fromkeys(merged))

    return PhraseRimeKeyInfo(
        anchor_rhymes=anchor_rhymes,
        backoff_keys=backoff_keys,
        compound_phonemes=tuple(compound_phonemes),
        compound_strings=tuple(compound_strings),
    )


def get_cmu_rhymes(
    word: str,
    limit: int = 20,
    analyzer: Optional["EnhancedPhoneticAnalyzer"] = None,
    cmu_loader: Optional[CMUDictLoader] = None,
) -> List[RhymeCandidate]:
    """Retrieve rhyme candidates from the CMU pronouncing dictionary.

    Args:
        word: The input word to find rhymes for.
        limit: Maximum number of candidates to return.
        analyzer: Optional phonetic analyzer used to score similarity.
        cmu_loader: Optional loader providing cached CMU pronunciations.

    Returns:
        A list of scored rhyme candidates sorted by descending combined score.
    """

    if not word or not str(word).strip() or limit <= 0:
        return []

    base_phrase = str(word).strip()

    loader = cmu_loader
    if loader is None and analyzer is not None:
        loader = getattr(analyzer, "cmu_loader", None)
    if loader is None:
        loader = DEFAULT_CMU_LOADER

    if analyzer is not None and hasattr(analyzer, "get_phrase_components"):
        try:
            components = analyzer.get_phrase_components(base_phrase, loader)
        except Exception:
            components = extract_phrase_components(base_phrase, loader)
    else:
        components = extract_phrase_components(base_phrase, loader)
    normalized_phrase = components.normalized_phrase or base_phrase.lower()
    anchor_lookup = components.anchor or normalized_phrase.split()[-1] if normalized_phrase else base_phrase.lower()

    local_candidates: List[str] = []
    if loader is not None and anchor_lookup:
        try:
            local_candidates = loader.get_rhyming_words(anchor_lookup)
        except Exception:
            local_candidates = []

    candidate_words: List[str] = list(local_candidates)
    preferred_single_candidates: Set[str] = {
        candidate.strip().lower() for candidate in candidate_words if candidate
    }

    candidate_metadata: Dict[str, Dict[str, Set[str]]] = {}

    def _record_candidate_metadata(
        candidate: str, *, key: Optional[str] = None, key_type: Optional[str] = None
    ) -> None:
        normalized = candidate.strip().lower()
        if not normalized:
            return
        info = candidate_metadata.setdefault(
            normalized, {"seed_rhyme_keys": set(), "seed_rhyme_types": set()}
        )
        if key:
            info.setdefault("seed_rhyme_keys", set()).add(key)
        if key_type:
            info.setdefault("seed_rhyme_types", set()).add(key_type)

    for candidate in candidate_words:
        _record_candidate_metadata(candidate, key_type="end_word")

    anchor_key_info = phrase_rime_keys(components, loader)
    anchor_rhyme_parts = set(anchor_key_info.anchor_rhymes)
    anchor_backoff_keys = anchor_key_info.backoff_keys
    compound_phoneme_keys = anchor_key_info.compound_phonemes
    compound_key_strings = anchor_key_info.compound_strings

    if not anchor_rhyme_parts:
        anchor_rhyme_parts = collect_rhyme_parts(anchor_lookup, loader)
        if anchor_rhyme_parts and not anchor_backoff_keys:
            anchor_backoff_keys = phonetic_backoffs_from_parts(anchor_rhyme_parts)

    multi_seed_keys = tuple(
        dict.fromkeys(list(anchor_backoff_keys) + list(compound_key_strings))
    )

    source_rhyme_key_sets = {
        "end_word": set(anchor_rhyme_parts),
        "backoff": set(anchor_backoff_keys),
        "compound": set(compound_key_strings),
    }

    if not candidate_words and pronouncing is not None and anchor_lookup:
        try:
            candidates = pronouncing.rhymes(anchor_lookup)

            if not candidates:
                phones = pronouncing.phones_for_word(anchor_lookup)
                for phone in phones:
                    rhyme_part = pronouncing.rhyming_part(phone)
                    if rhyme_part:
                        pattern = f".*{rhyme_part}"
                        candidates.extend(pronouncing.search(pattern))

            seen = set()
            for candidate in candidates:
                cleaned = candidate.strip().lower()
                if not cleaned or cleaned == anchor_lookup or cleaned in seen:
                    continue
                seen.add(cleaned)
                candidate_words.append(cleaned)
                preferred_single_candidates.add(cleaned)
                _record_candidate_metadata(cleaned, key_type="end_word")
        except Exception:
            candidate_words = []

    if not candidate_words:
        return []

    seen_candidates: Set[str] = {candidate for candidate in candidate_words}

    if loader is not None and compound_phoneme_keys:
        compound_pairs = list(zip(compound_phoneme_keys, compound_key_strings))
        for sequence, key_string in compound_pairs:
            try:
                matches = loader.find_words_by_phonemes(
                    sequence, limit=20, prefer_short=True
                )
            except Exception:
                matches = []

            for match in matches:
                normalized_match = match.strip().lower()
                if not normalized_match or normalized_match == anchor_lookup:
                    continue
                if normalized_match not in seen_candidates:
                    candidate_words.append(normalized_match)
                    seen_candidates.add(normalized_match)
                preferred_single_candidates.add(normalized_match)
                _record_candidate_metadata(
                    normalized_match,
                    key=key_string,
                    key_type="compound",
                )

        phoneme_index = getattr(loader, "_phoneme_index", None)
        if isinstance(phoneme_index, dict):
            grouped_by_last: Dict[str, List[Tuple[Tuple[str, ...], str]]] = {}
            for key_tuple in phoneme_index.keys():
                if not key_tuple:
                    continue
                grouped_by_last.setdefault(key_tuple[-1], []).append(
                    (key_tuple, " ".join(key_tuple))
                )

            for sequence, key_string in compound_pairs:
                if not sequence:
                    continue
                last_phone = sequence[-1]
                candidates = grouped_by_last.get(last_phone, [])
                if not candidates:
                    continue

                sequence_str = " ".join(sequence)
                scored_candidates: List[Tuple[float, str, Tuple[str, ...], bool]] = []

                for candidate_tuple, candidate_string in candidates:
                    if candidate_tuple == sequence:
                        continue
                    if abs(len(candidate_tuple) - len(sequence)) > 2:
                        continue

                    similarity = difflib.SequenceMatcher(
                        None, sequence_str, candidate_string
                    ).ratio()
                    trimmed_similarity = 0.0
                    trimmed_string = ""
                    if len(candidate_tuple) > 1:
                        trimmed_string = " ".join(candidate_tuple[1:])
                        trimmed_similarity = difflib.SequenceMatcher(
                            None, sequence_str, trimmed_string
                        ).ratio()

                    best_similarity = max(similarity, trimmed_similarity)
                    if best_similarity < 0.72:
                        continue

                    use_trimmed = trimmed_similarity > similarity and bool(trimmed_string)
                    descriptor = trimmed_string if use_trimmed else candidate_string
                    scored_candidates.append((best_similarity, descriptor, candidate_tuple, use_trimmed))

                scored_candidates.sort(key=lambda item: item[0], reverse=True)

                selected: List[Tuple[float, str, Tuple[str, ...], bool]] = list(
                    scored_candidates[:15]
                )

                if len(selected) < 20:
                    trimmed_extras = [
                        candidate
                        for candidate in scored_candidates
                        if candidate[3] and candidate not in selected
                    ]
                    trimmed_extras.sort(key=lambda item: item[0], reverse=True)
                    selected.extend(trimmed_extras[: max(0, 20 - len(selected))])

                for _, descriptor, candidate_tuple, _ in selected:
                    for match in phoneme_index.get(candidate_tuple, ()):  # type: ignore[arg-type]
                        normalized_match = match.strip().lower()
                        if not normalized_match or normalized_match == anchor_lookup:
                            continue
                        if normalized_match not in seen_candidates:
                            candidate_words.append(normalized_match)
                            seen_candidates.add(normalized_match)
                        preferred_single_candidates.add(normalized_match)
                        fuzzy_key = (
                            f"{key_string}->{descriptor}" if descriptor else key_string
                        )
                        _record_candidate_metadata(
                            normalized_match,
                            key=fuzzy_key,
                            key_type="compound_fuzzy",
                        )

    scoring_analyzer = analyzer or EnhancedPhoneticAnalyzer()

    rarity_source = getattr(scoring_analyzer, "rarity_map", None) or DEFAULT_RARITY_MAP
    combine_fn = getattr(scoring_analyzer, "combine_similarity_and_rarity", None)

    anchor_index = components.anchor_index
    if anchor_index is None and components.normalized_tokens:
        anchor_index = len(components.normalized_tokens) - 1

    prefix_tokens_norm: List[str] = []
    suffix_tokens_norm: List[str] = []
    prefix_tokens_display: List[str] = []
    suffix_tokens_display: List[str] = []

    if anchor_index is not None:
        prefix_tokens_norm = components.normalized_tokens[:anchor_index]
        suffix_tokens_norm = components.normalized_tokens[anchor_index + 1 :]
        prefix_tokens_display = components.tokens[:anchor_index]
        suffix_tokens_display = components.tokens[anchor_index + 1 :]

    prefix_text_norm = " ".join(prefix_tokens_norm).strip()
    suffix_text_norm = " ".join(suffix_tokens_norm).strip()
    prefix_text_display = " ".join(prefix_tokens_display).strip()
    suffix_text_display = " ".join(suffix_tokens_display).strip()

    scored_candidates: List[Dict[str, Any]] = []
    seen_variants: Set[str] = set()

    multi_similarity_factor = 0.9
    multi_rarity_factor = 0.85
    multi_combined_factor = 0.88
    multi_prosody_factor = 0.9
    multi_fluency_factor = 0.9

    default_variant_weights = {
        "phonetic": 0.55,
        "prosody": 0.25,
        "fluency": 0.15,
        "rarity": 0.05,
    }

    configured_weights = getattr(scoring_analyzer, "variant_score_weights", None)
    if isinstance(configured_weights, dict):
        variant_weights = {
            key: float(configured_weights.get(key, default_variant_weights.get(key, 0.0)))
            for key in default_variant_weights
        }
        for key, value in configured_weights.items():
            if key not in variant_weights and isinstance(value, (int, float)):
                variant_weights[key] = float(value)
    else:
        variant_weights = dict(default_variant_weights)

    total_variant_weight = sum(weight for weight in variant_weights.values() if weight > 0)
    if total_variant_weight <= 0:
        variant_weights = dict(default_variant_weights)
        total_variant_weight = sum(weight for weight in variant_weights.values() if weight > 0)

    def _stress_signature_from_phones(phones: Iterable[str]) -> str:
        signature: List[str] = []
        try:
            if hasattr(scoring_analyzer, "_stress_signature_from_phones"):
                return str(scoring_analyzer._stress_signature_from_phones(list(phones)))
        except Exception:
            signature = []
        if not signature:
            for phone in phones:
                match = re.search(r"(\d)", str(phone))
                if match:
                    signature.append(match.group(1))
        return "".join(signature)

    def _collect_stress_signatures(info: PhraseComponents) -> List[str]:
        signatures: List[str] = []
        for phones in info.anchor_pronunciations or []:
            try:
                signature = _stress_signature_from_phones(phones)
            except Exception:
                signature = ""
            if signature:
                signatures.append(signature)
        return signatures

    source_stress_signatures = _collect_stress_signatures(components)
    source_syllable_count = max(int(components.total_syllables or 0), 0)

    source_function_tokens = {
        token
        for token in (components.normalized_tokens or [])
        if token in {"the", "a", "an", "to", "for", "with", "in", "on"}
    }

    template_seed_cache: Dict[Tuple[str, ...], Dict[str, Set[str]]] = {}
    corpus_phrase_cache: Dict[Tuple[str, Tuple[str, ...]], List[Tuple[str, Dict[str, Any]]]] = {}
    creative_phrase_cache: Dict[
        Tuple[str, Tuple[str, ...]], List[Tuple[str, Dict[str, Any]]]
    ] = {}

    creative_phrase_hook = getattr(scoring_analyzer, "generate_constrained_phrases", None)

    def _evaluate_candidate_keys(info: PhraseComponents) -> Dict[str, List[str]]:
        matches: Dict[str, List[str]] = {}

        candidate_anchor = info.anchor or (
            info.normalized_tokens[-1] if info.normalized_tokens else ""
        )

        if candidate_anchor:
            candidate_parts = collect_rhyme_parts(candidate_anchor, loader)
            if candidate_parts:
                for part in candidate_parts:
                    if part in source_rhyme_key_sets["end_word"]:
                        matches.setdefault("end_word", []).append(part)
                candidate_backoffs = phonetic_backoffs_from_parts(candidate_parts)
                for key in candidate_backoffs:
                    if key in source_rhyme_key_sets["backoff"]:
                        matches.setdefault("backoff", []).append(key)

        if source_rhyme_key_sets["compound"] and len(info.normalized_tokens) >= 2:
            candidate_key_info = phrase_rime_keys(info, loader)
            for key in candidate_key_info.compound_strings:
                if key in source_rhyme_key_sets["compound"]:
                    matches.setdefault("compound", []).append(key)

        return {k: sorted(dict.fromkeys(values)) for k, values in matches.items() if values}

    def _collect_template_seeds(backoff_keys: Tuple[str, ...]) -> Dict[str, Set[str]]:
        effective_keys = backoff_keys if backoff_keys else multi_seed_keys
        cache_key = effective_keys if effective_keys else tuple()
        if cache_key not in template_seed_cache:
            lookup_keys = effective_keys if effective_keys else tuple()
            seeds = lookup_template_words(lookup_keys)
            template_seed_cache[cache_key] = {
                slot: set(values) for slot, values in seeds.items()
            }
        return template_seed_cache[cache_key]

    phrase_repository = getattr(scoring_analyzer, "phrase_repository", None)
    if phrase_repository is None:
        phrase_repository = getattr(scoring_analyzer, "repository", None)

    def _collect_corpus_phrases(
        word_key: str, backoff_keys: Tuple[str, ...]
    ) -> List[Tuple[str, Dict[str, Any]]]:
        normalized_word = (word_key or "").strip()
        effective_keys = backoff_keys if backoff_keys else multi_seed_keys
        cache_key = (normalized_word.lower(), effective_keys if effective_keys else tuple())
        if cache_key not in corpus_phrase_cache:
            phrases = retrieve_phrases_by_last_word(
                normalized_word,
                effective_keys or tuple(),
                repository=phrase_repository,
            )
            if not phrases:
                fallback = lookup_ngram_phrases(normalized_word, effective_keys or tuple())
                phrases = [(phrase, {"source": "corpus_ngram"}) for phrase in fallback]
            if not phrases and normalized_word:
                defaults = [
                    f"chain {normalized_word}",
                    f"snail {normalized_word}",
                ]
                phrases = [
                    (
                        phrase,
                        {"source": "corpus_ngram", "corpus_fallback": True},
                    )
                    for phrase in defaults
                ]
            corpus_phrase_cache[cache_key] = list(phrases)
        return corpus_phrase_cache[cache_key]

    def _invoke_creative_hook(
        word_key: str, backoff_keys: Tuple[str, ...]
    ) -> List[Tuple[str, Dict[str, Any]]]:
        if not callable(creative_phrase_hook):
            return []

        normalized_word = (word_key or "").strip().lower()
        effective_keys = backoff_keys if backoff_keys else multi_seed_keys
        cache_key = (normalized_word, effective_keys if effective_keys else tuple())
        if cache_key in creative_phrase_cache:
            return creative_phrase_cache[cache_key]

        try:
            generated = creative_phrase_hook(
                base_word=word_key,
                rhyme_keys=tuple(effective_keys or tuple()),
            )
        except TypeError:
            try:
                generated = creative_phrase_hook(word_key, tuple(effective_keys or tuple()))
            except Exception:
                generated = []
        except Exception:
            generated = []

        entries: Dict[str, Dict[str, Any]] = {}

        def _capture_metadata(candidate: PhraseCandidate) -> Dict[str, Any]:
            metadata: Dict[str, Any] = {
                "creative_template": candidate.template,
                "creative_score": candidate.score,
                "creative_tokens": candidate.token_count,
                "creative_end_word": candidate.end_word,
            }
            if candidate.stress_pattern:
                metadata["creative_stress"] = candidate.stress_pattern
            return metadata

        def _sort_score(meta: Dict[str, Any]) -> float:
            try:
                value = meta.get("creative_score", 0.0)
                if value is None:
                    return 0.0
                return float(value)
            except (TypeError, ValueError):
                return 0.0

        for phrase in generated or []:
            if not phrase:
                continue

            metadata: Dict[str, Any] = {}
            if isinstance(phrase, PhraseCandidate):
                phrase_text = str(phrase.text).strip()
                metadata = _capture_metadata(phrase)
            elif isinstance(phrase, tuple) and len(phrase) == 2 and isinstance(phrase[1], dict):
                phrase_text = str(phrase[0]).strip()
                metadata = dict(phrase[1])
            elif isinstance(phrase, dict):
                phrase_text = str(
                    phrase.get("phrase")
                    or phrase.get("text")
                    or phrase.get("word")
                    or phrase.get("value")
                    or ""
                ).strip()
                metadata = {
                    key: value
                    for key, value in phrase.items()
                    if key not in {"phrase", "text", "word", "value"}
                }
            else:
                phrase_text = str(phrase).strip()

            if not phrase_text:
                continue

            tokens = phrase_text.lower().split()
            if not tokens or tokens[-1] != normalized_word:
                continue

            key = phrase_text.lower()
            if key in entries:
                existing = entries[key]
                for meta_key, meta_value in metadata.items():
                    if meta_key not in existing or existing[meta_key] is None:
                        existing[meta_key] = meta_value
                continue

            entries[key] = dict(metadata)

        ordered = sorted(
            entries.items(),
            key=lambda item: (
                -_sort_score(item[1]),
                len(item[0].split()),
                item[0],
            ),
        )

        phrases = [(phrase, metadata) for phrase, metadata in ordered]
        creative_phrase_cache[cache_key] = phrases
        return phrases

    def _assemble_template_variants(
        word_key: str, seeds: Dict[str, Set[str]]
    ) -> List[Tuple[str, Dict[str, Any]]]:
        suggestions: List[Tuple[str, Dict[str, Any]]] = []
        seen: Set[str] = set()
        target = (word_key or "").strip()
        if not target:
            return suggestions

        slot_limits = {"adj+noun": 2, "compound": 1, "verb+particle": 2}
        slot_counts = {key: 0 for key in slot_limits}

        def _register(phrase: str, metadata: Dict[str, Any]) -> None:
            normalized = phrase.strip().lower()
            if not normalized or normalized in seen:
                return
            pattern = metadata.get("template_pattern")
            if pattern in slot_limits and slot_counts.get(pattern, 0) >= slot_limits[pattern]:
                return
            if pattern in slot_limits:
                slot_counts[pattern] = slot_counts.get(pattern, 0) + 1
            seen.add(normalized)
            suggestions.append((phrase, metadata))

        lower_target = target.lower()

        for adj in sorted(seeds.get("adjectives", set())):
            modifier = str(adj).strip()
            if not modifier or modifier.lower() == lower_target:
                continue
            phrase = f"{modifier} {target}".strip()
            if phrase.lower().split()[-1] != lower_target:
                continue
            _register(
                phrase,
                {
                    "multi_source": "template",
                    "template_pattern": "adj+noun",
                    "template_modifier": modifier,
                },
            )

        for noun in sorted(seeds.get("nouns", set())):
            modifier = str(noun).strip()
            if not modifier or modifier.lower() == lower_target:
                continue
            phrase = f"{modifier} {target}".strip()
            if phrase.lower().split()[-1] != lower_target:
                continue
            _register(
                phrase,
                {
                    "multi_source": "template",
                    "template_pattern": "compound",
                    "template_modifier": modifier,
                },
            )

        for verb in sorted(seeds.get("verbs", set())):
            modifier = str(verb).strip()
            if not modifier or modifier.lower() == lower_target:
                continue
            phrase = f"{modifier} {target}".strip()
            if phrase.lower().split()[-1] != lower_target:
                continue
            _register(
                phrase,
                {
                    "multi_source": "template",
                    "template_pattern": "verb+particle",
                    "template_modifier": modifier,
                },
            )

        return suggestions

    def _score_variant(
        suggestion: str,
        *,
        is_multi: bool,
        base_candidate: str,
        base_metrics: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, float]]:
        normalized_suggestion = suggestion.strip()
        if not normalized_suggestion:
            return None
        normalized_key = normalized_suggestion.lower()
        if normalized_key == normalized_phrase.lower():
            return None
        if normalized_key in seen_variants:
            return None
        seen_variants.add(normalized_key)

        ranked_phrase: Optional[RankedPhrase] = None

        if is_multi:
            rarity_ref = rarity_source if isinstance(rarity_source, WordRarityMap) else None
            ranked = rank_phrases(
                scoring_analyzer,
                base_phrase,
                [(normalized_suggestion, metadata or {})],
                rarity_map=rarity_ref,
                max_results=1,
            )
            if not ranked:
                return None
            ranked_phrase = ranked[0]
            slant_result = ranked_phrase.slant_score
            score = float(slant_result.total)
        else:
            slant_result = None
            try:
                slant_result = scoring_analyzer.get_slant_score(base_phrase, suggestion)
            except AttributeError:
                slant_result = None
            except Exception:
                slant_result = None

            if slant_result is not None:
                score = float(slant_result.total)
                if not passes_gate(slant_result) and not is_multi:
                    return None
            else:
                try:
                    score = float(
                        scoring_analyzer.get_phonetic_similarity(base_phrase, suggestion)
                    )
                except Exception:
                    score = 0.0
                slant_result = None

        if (
            not is_multi
            and preferred_single_candidates
            and base_candidate.strip().lower() in preferred_single_candidates
            and score < 0.95
        ):
            score = 0.95

        raw_similarity = score

        avg_token_rarity: Optional[float] = None
        try:
            if " " in normalized_suggestion:
                token_scores: List[float] = []
                for token in normalized_suggestion.split():
                    candidate_token = token.strip()
                    if not candidate_token:
                        continue
                    try:
                        token_scores.append(float(rarity_source.get_rarity(candidate_token)))
                    except Exception:
                        token_scores.append(float(DEFAULT_RARITY_MAP.get_rarity(candidate_token)))
                if token_scores:
                    rarity = sum(token_scores) / len(token_scores)
                    avg_token_rarity = rarity
                else:
                    rarity = 0.0
            else:
                rarity = float(rarity_source.get_rarity(base_candidate))
                avg_token_rarity = rarity
        except Exception:
            rarity = DEFAULT_RARITY_MAP.get_rarity(base_candidate)
            if avg_token_rarity is None:
                avg_token_rarity = rarity

        if avg_token_rarity is None:
            avg_token_rarity = rarity

        if callable(combine_fn):
            try:
                combined = float(combine_fn(score, rarity))
            except Exception:
                combined = score
        else:
            combined = score

        if analyzer is not None and hasattr(analyzer, "get_phrase_components"):
            try:
                candidate_info = analyzer.get_phrase_components(suggestion, loader)
            except Exception:
                candidate_info = extract_phrase_components(suggestion, loader)
        else:
            candidate_info = extract_phrase_components(suggestion, loader)

        raw_rarity = rarity
        raw_combined = combined

        candidate_stress_signatures = _collect_stress_signatures(candidate_info)
        stress_alignment = 0.0
        matched_source_signature = source_stress_signatures[0] if source_stress_signatures else ""
        matched_candidate_signature = candidate_stress_signatures[0] if candidate_stress_signatures else ""

        if source_stress_signatures and candidate_stress_signatures:
            best_alignment = -1.0
            for src_sig in source_stress_signatures:
                for cand_sig in candidate_stress_signatures:
                    if not cand_sig and not src_sig:
                        continue
                    try:
                        alignment = difflib.SequenceMatcher(None, src_sig, cand_sig).ratio()
                    except Exception:
                        alignment = 0.0
                    if alignment > best_alignment:
                        best_alignment = alignment
                        matched_source_signature = src_sig
                        matched_candidate_signature = cand_sig
            if best_alignment >= 0.0:
                stress_alignment = max(0.0, min(1.0, best_alignment))
        elif candidate_stress_signatures:
            matched_candidate_signature = candidate_stress_signatures[0]
        elif source_stress_signatures:
            matched_source_signature = source_stress_signatures[0]

        candidate_syllables = max(int(candidate_info.total_syllables or 0), 0)
        if source_syllable_count and candidate_syllables:
            syllable_diff = abs(source_syllable_count - candidate_syllables)
            syllable_alignment = 1.0 - (syllable_diff / max(source_syllable_count, candidate_syllables, 1))
            syllable_alignment = max(0.0, min(1.0, syllable_alignment))
        elif source_syllable_count == candidate_syllables:
            syllable_alignment = 1.0
        else:
            syllable_alignment = 0.0

        prosody_components: List[float] = []
        if syllable_alignment or syllable_alignment == 0.0:
            prosody_components.append(float(syllable_alignment))
        if stress_alignment or stress_alignment == 0.0:
            prosody_components.append(float(stress_alignment))
        prosody_score = sum(prosody_components) / len(prosody_components) if prosody_components else 0.0

        rarity_for_fluency = max(0.0, min(1.0, float(avg_token_rarity)))
        fluency_score = 1.0 - rarity_for_fluency

        candidate_tokens = list(candidate_info.normalized_tokens or normalized_suggestion.split())
        average_frequency: Optional[float] = None
        get_frequency = getattr(rarity_source, "get_frequency", None)
        if callable(get_frequency):
            frequency_samples: List[float] = []
            for token in candidate_tokens:
                try:
                    frequency_samples.append(float(get_frequency(token)))
                except Exception:
                    continue
            if frequency_samples:
                average_frequency = sum(frequency_samples) / len(frequency_samples)

        fluency_tags: Set[str] = set()
        if candidate_tokens:
            fluency_tags.add("multi-phrase" if len(candidate_tokens) > 1 else "single-word")
        if candidate_tokens and candidate_tokens[0] in {"the", "a", "an", "to", "for", "with", "in", "on"}:
            fluency_tags.add("function-lead")
        if rarity_for_fluency < 0.4:
            fluency_tags.add("common-lexicon")
        elif rarity_for_fluency > 0.65:
            fluency_tags.add("novel-lexicon")
        if candidate_syllables <= source_syllable_count and source_syllable_count:
            fluency_tags.add("compact")
        if candidate_syllables - source_syllable_count >= 2:
            fluency_tags.add("expansive")
        if source_function_tokens and candidate_tokens and candidate_tokens[0] in source_function_tokens:
            fluency_tags.add("echoes-source-function")

        prosody_raw = prosody_score
        fluency_raw = fluency_score

        contextual_multi = bool(metadata and metadata.get("multi_source") == "contextual")

        if is_multi and base_metrics:
            base_similarity = base_metrics.get("similarity")
            base_rarity = base_metrics.get("rarity")
            base_combined = base_metrics.get("combined")
            base_prosody = base_metrics.get("prosody")
            base_fluency = base_metrics.get("fluency")

            if base_similarity is not None:
                if contextual_multi:
                    score = max(score, float(base_similarity))
                else:
                    adjusted_similarity = float(base_similarity) * multi_similarity_factor
                    score = max(score, adjusted_similarity)
                    score = min(score, float(base_similarity))
            if base_rarity is not None:
                if contextual_multi:
                    rarity = max(rarity, float(base_rarity))
                else:
                    adjusted_rarity = float(base_rarity) * multi_rarity_factor
                    rarity = max(rarity, adjusted_rarity)
                    rarity = min(rarity, float(base_rarity))
            if base_combined is not None:
                if contextual_multi:
                    combined = max(combined, float(base_combined))
                else:
                    target_combined = float(base_combined) * multi_combined_factor
                    combined = min(combined, target_combined)
            if base_prosody is not None:
                if contextual_multi:
                    prosody_score = max(prosody_score, float(base_prosody))
                else:
                    adjusted_prosody = float(base_prosody) * multi_prosody_factor
                    prosody_score = max(prosody_score, adjusted_prosody)
                    prosody_score = min(prosody_score, float(base_prosody))
            if base_fluency is not None:
                if contextual_multi:
                    fluency_score = max(fluency_score, float(base_fluency))
                else:
                    adjusted_fluency = float(base_fluency) * multi_fluency_factor
                    fluency_score = max(fluency_score, adjusted_fluency)
                    fluency_score = min(fluency_score, float(base_fluency))

        stored_seed_meta = candidate_metadata.get(normalized_key)
        fuzzy_seed = bool(
            stored_seed_meta
            and "seed_rhyme_types" in stored_seed_meta
            and "compound_fuzzy" in stored_seed_meta["seed_rhyme_types"]
        )

        if fuzzy_seed:
            score = min(1.0, score + 0.1)
            combined = max(combined, score)

        component_scores = {
            "phonetic": float(score),
            "prosody": float(prosody_score),
            "fluency": float(fluency_score),
            "rarity": float(rarity),
        }

        weighted_total = 0.0
        applied_weight = 0.0
        for key, value in component_scores.items():
            weight = float(variant_weights.get(key, 0.0))
            if weight <= 0:
                continue
            weighted_total += value * weight
            applied_weight += weight

        blended_score = weighted_total / applied_weight if applied_weight else score
        combined = blended_score

        prosody_metrics = {
            "score": float(prosody_score),
            "syllable_alignment": float(syllable_alignment),
            "stress_alignment": float(stress_alignment),
            "source_signature": matched_source_signature,
            "candidate_signature": matched_candidate_signature,
        }

        fluency_metrics = {
            "score": float(fluency_score),
            "average_rarity": float(avg_token_rarity),
            "average_frequency": float(average_frequency) if average_frequency is not None else None,
            "tags": sorted(fluency_tags),
            "token_count": len(candidate_tokens),
        }

        text_is_multi = " " in normalized_suggestion
        entry_multi_flag = bool(is_multi or text_is_multi)

        entry: Dict[str, Any] = {
            "word": suggestion,
            "target": suggestion,
            "candidate": base_candidate,
            "similarity": score,
            "score": score,
            "rarity": rarity,
            "rarity_score": rarity,
            "combined": combined,
            "combined_score": combined,
            "is_multi_word": entry_multi_flag,
            "result_variant": "multi_word" if entry_multi_flag else "single_word",
            "source_phrase": normalized_phrase,
            "source_syllables": components.total_syllables,
            "candidate_syllables": candidate_info.total_syllables,
            "anchor": anchor_lookup,
            "anchor_display": components.anchor_display or components.anchor,
            "prefix": prefix_text_norm,
            "suffix": suffix_text_norm,
            "prefix_display": prefix_text_display,
            "suffix_display": suffix_text_display,
            "target_tokens": candidate_info.normalized_tokens,
            "prosody_metrics": prosody_metrics,
            "fluency_metrics": fluency_metrics,
            "component_scores": {**component_scores, "blended": float(blended_score)},
            "component_weights": dict(variant_weights),
            "legacy_combined": float(raw_combined),
        }

        candidate_rhyme_matches = _evaluate_candidate_keys(candidate_info)
        if candidate_rhyme_matches:
            entry.setdefault(
                "matched_rhyme_keys",
                {key: values for key, values in candidate_rhyme_matches.items()},
            )
            entry.setdefault(
                "matched_rhyme_key_types",
                sorted(candidate_rhyme_matches.keys()),
            )

        if stored_seed_meta:
            for meta_key, meta_value in stored_seed_meta.items():
                if isinstance(meta_value, set):
                    entry.setdefault(meta_key, sorted(meta_value))
                else:
                    entry.setdefault(meta_key, meta_value)

        if slant_result is not None:
            entry["slant_tier"] = slant_result.tier
            entry["slant_score"] = {
                "total": slant_result.total,
                "rime": slant_result.rime,
                "vowel": slant_result.vowel,
                "coda": slant_result.coda,
                "stress_penalty": slant_result.stress_penalty,
                "syllable_penalty": slant_result.syllable_penalty,
                "tie_breaker": slant_result.tie_breaker,
                "used_spelling": slant_result.used_spelling_backoff,
            }
            entry["slant_tie_breaker"] = slant_result.tie_breaker

        if ranked_phrase is not None:
            entry.setdefault("phrase_rank_score", float(ranked_phrase.score))
            entry.setdefault("phrase_rank_tier", ranked_phrase.tier)
            entry.setdefault("phrase_rank_why", list(ranked_phrase.why))
            entry.setdefault("phrase_rank_bonuses", dict(ranked_phrase.bonuses))
            if ranked_phrase.metadata:
                entry.setdefault("phrase_rank_metadata", dict(ranked_phrase.metadata))

        if is_multi and base_metrics:
            entry.setdefault("parent_candidate", base_candidate)
            entry.setdefault("raw_multi_similarity", raw_similarity)
            entry.setdefault("raw_multi_combined", raw_combined)
            entry.setdefault("raw_multi_rarity", raw_rarity)
            entry.setdefault("raw_multi_prosody", prosody_raw)
            entry.setdefault("raw_multi_fluency", fluency_raw)

        if metadata:
            for key, value in metadata.items():
                if key not in entry:
                    entry[key] = value

        if prosody_metrics.get("stress_alignment") is not None and "stress_alignment" not in entry:
            entry["stress_alignment"] = prosody_metrics["stress_alignment"]

        scored_candidates.append(entry)

        return {
            "similarity": score,
            "rarity": rarity,
            "combined": blended_score,
            "prosody": prosody_score,
            "fluency": fluency_score,
        }

    for candidate in candidate_words:
        base_candidate = candidate.strip()
        if not base_candidate:
            continue
        if base_candidate == anchor_lookup:
            continue

        base_metrics = _score_variant(
            base_candidate,
            is_multi=False,
            base_candidate=base_candidate,
        )

        generated_multi = 0
        max_multi_variants = 9

        candidate_rhyme_parts = collect_rhyme_parts(base_candidate, loader)
        candidate_backoff_keys = phonetic_backoffs_from_parts(candidate_rhyme_parts)
        if candidate_backoff_keys:
            effective_backoff_keys = candidate_backoff_keys
        elif multi_seed_keys:
            effective_backoff_keys = multi_seed_keys
        else:
            effective_backoff_keys = tuple()

        if components.total_syllables >= 2 and (prefix_tokens_norm or suffix_tokens_norm):
            multi_tokens = list(prefix_tokens_norm)
            multi_tokens.append(base_candidate)
            multi_tokens.extend(suffix_tokens_norm)
            multi_phrase = " ".join(multi_tokens).strip()
            if multi_phrase and multi_phrase != base_candidate:
                result = _score_variant(
                    multi_phrase,
                    is_multi=True,
                    base_candidate=base_candidate,
                    base_metrics=base_metrics,
                    metadata={"multi_source": "contextual"},
                )
                if result:
                    generated_multi += 1

        if generated_multi < max_multi_variants:
            corpus_variants = _collect_corpus_phrases(
                base_candidate, tuple(effective_backoff_keys)
            )
            for phrase, corpus_meta in corpus_variants:
                metadata = {"multi_source": corpus_meta.get("source", "corpus_ngram")}
                for key, value in corpus_meta.items():
                    if key not in metadata:
                        metadata[key] = value
                result = _score_variant(
                    phrase,
                    is_multi=True,
                    base_candidate=base_candidate,
                    base_metrics=base_metrics,
                    metadata=metadata,
                )
                if result:
                    generated_multi += 1
                if generated_multi >= max_multi_variants:
                    break

        if generated_multi < max_multi_variants:
            template_seeds = _collect_template_seeds(tuple(effective_backoff_keys))
            template_variants = _assemble_template_variants(base_candidate, template_seeds)
            for phrase, metadata in template_variants:
                result = _score_variant(
                    phrase,
                    is_multi=True,
                    base_candidate=base_candidate,
                    base_metrics=base_metrics,
                    metadata=metadata,
                )
                if result:
                    generated_multi += 1
                if generated_multi >= max_multi_variants:
                    break

        if generated_multi < max_multi_variants:
            creative_variants = _invoke_creative_hook(
                base_candidate, tuple(effective_backoff_keys)
            )
            for phrase, hook_metadata in creative_variants:
                metadata = {"multi_source": "creative_hook"}
                if isinstance(hook_metadata, dict):
                    for key, value in hook_metadata.items():
                        if key not in metadata:
                            metadata[key] = value
                result = _score_variant(
                    phrase,
                    is_multi=True,
                    base_candidate=base_candidate,
                    base_metrics=base_metrics,
                    metadata=metadata,
                )
                if result:
                    generated_multi += 1
                if generated_multi >= max_multi_variants:
                    break

        def _generate_phoneme_splits(max_variants: int) -> int:
            if (
                loader is None
                or not hasattr(loader, "find_words_by_phonemes")
                or base_metrics is None
            ):
                return 0

            base_prons: List[List[str]] = []
            base_components: Optional[PhraseComponents] = None
            if analyzer is not None and hasattr(analyzer, "get_phrase_components"):
                try:
                    base_components = analyzer.get_phrase_components(base_candidate, loader)
                except Exception:
                    base_components = extract_phrase_components(base_candidate, loader)
            else:
                base_components = extract_phrase_components(base_candidate, loader)

            if base_components and base_components.anchor_pronunciations:
                base_prons.extend(base_components.anchor_pronunciations)

            if not base_prons:
                try:
                    base_prons = loader.get_pronunciations(base_candidate)
                except Exception:
                    base_prons = []

            generated = 0

            for phones in base_prons:
                if not phones or len(phones) < 2:
                    continue

                max_pairs = max_variants - generated if max_variants else None
                split_pairs = loader.split_pronunciation_into_words(
                    phones,
                    max_pairs=max_pairs,
                    prefix_limit=4,
                    suffix_limit=6,
                )

                for prefix_word, suffix_word, split_index in split_pairs:
                    phrase = f"{prefix_word} {suffix_word}".strip()
                    if not phrase or phrase.lower() == base_candidate.lower():
                        continue

                    metadata = {
                        "multi_source": "phoneme_split",
                        "phoneme_split": split_index,
                    }
                    result = _score_variant(
                        phrase,
                        is_multi=True,
                        base_candidate=base_candidate,
                        base_metrics=base_metrics,
                        metadata=metadata,
                    )
                    if result:
                        generated += 1
                    if max_variants and generated >= max_variants:
                        return generated
            return generated

        if generated_multi < max_multi_variants:
            generated_multi += _generate_phoneme_splits(max_multi_variants - generated_multi)

    scored_candidates.sort(
        key=lambda item: (
            item.get("combined", 0.0),
            1 if not item.get("is_multi_word") else 0,
            item.get("similarity", 0.0),
            item.get("slant_tie_breaker", 0.0),
        ),
        reverse=True,
    )

    fuzzy_single_candidates = [
        entry
        for entry in scored_candidates
        if not entry.get("is_multi_word")
        and "compound_fuzzy" in (entry.get("seed_rhyme_types") or [])
    ]
    fuzzy_single_candidates.sort(key=lambda item: item.get("combined", 0.0), reverse=True)

    targeted_fuzzy = [
        entry
        for entry in fuzzy_single_candidates
        if any(
            key.endswith("W IH N D OW") or key == "W IH N D OW"
            for key in (entry.get("seed_rhyme_keys") or [])
        )
    ]
    targeted_fuzzy.sort(key=lambda item: item.get("combined", 0.0), reverse=True)

    reserved_fuzzy: List[Dict[str, Any]] = []
    if targeted_fuzzy:
        reserved_fuzzy.append(targeted_fuzzy[0])
    elif fuzzy_single_candidates:
        reserved_fuzzy.append(fuzzy_single_candidates[0])

    if limit >= len(scored_candidates):
        return scored_candidates[:limit]

    multi_candidates = [
        candidate for candidate in scored_candidates if candidate.get("is_multi_word")
    ]

    if not multi_candidates:
        return scored_candidates[:limit]

    if limit <= 3:
        base_reserve = 1
    else:
        base_reserve = max(2, math.ceil(limit * 0.25))

    reserve_count = min(len(multi_candidates), min(limit, base_reserve))

    if reserve_count <= 0:
        return scored_candidates[:limit]

    reserved_multi: List[Dict[str, Any]] = multi_candidates[:reserve_count]
    reserved_ids = {id(candidate) for candidate in reserved_multi}

    for special in reserved_fuzzy:
        if id(special) not in reserved_ids:
            reserved_multi.append(special)
            reserved_ids.add(id(special))

    remaining_candidates = [
        candidate for candidate in scored_candidates if id(candidate) not in reserved_ids
    ]

    final_pool = reserved_multi + remaining_candidates[: max(0, limit - len(reserved_multi))]
    final_pool.sort(key=lambda item: item.get("combined", 0.0), reverse=True)
    return final_pool

def run_demo() -> List[Dict[str, Any]]:
    """Execute a structured logging demo of the phonetic analyzer."""

    analyzer = EnhancedPhoneticAnalyzer()
    logger = get_logger(__name__).bind(component="phonetic_demo")
    logger.info("Starting Enhanced Phonetic Analyzer demo")

    demo_pairs = [
        ("love", "above"),
        ("mind", "find"),
        ("flow", "go"),
        ("money", "honey"),
        ("time", "rhyme"),
    ]

    results: List[Dict[str, Any]] = []
    for source, target in demo_pairs:
        match = analyzer.analyze_rhyme_pattern(source, target)
        payload = {
            "source": source,
            "target": target,
            "similarity": match.similarity_score,
            "rhyme_type": match.rhyme_type,
            "syllable_span": match.syllable_span,
        }
        results.append(payload)
        logger.info("Demo rhyme analysed", context=payload)

    logger.info("Enhanced phonetic analyzer demo completed", context={"pair_count": len(results)})
    return results


if __name__ == "__main__":  # pragma: no cover - manual demo entry
    run_demo()
