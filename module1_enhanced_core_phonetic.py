#!/usr/bin/env python3
"""
Module 1: Enhanced Core Phonetic Analysis for RhymeRarity
Handles phonetic similarity calculations and rhyme detection algorithms
Part of the RhymeRarity system deployed on Hugging Face
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Iterable
from dataclasses import dataclass
import difflib
import math
import sqlite3
from collections import Counter

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

try:  # pragma: no cover - optional dependency
    import pronouncing  # type: ignore
except ImportError:  # pragma: no cover - gracefully handle missing package
    pronouncing = None  # type: ignore


class CMUDictLoader:
    """Lazy loader for the CMU pronouncing dictionary.

    The loader parses ``cmudict.7b`` once and caches pronunciations along with
    their rhyme parts (defined as the final stressed vowel and trailing
    phonemes). Subsequent lookups reuse the cached data to avoid repeatedly
    loading the large dictionary file.
    """

    def __init__(self, dict_path: Optional[Path | str] = None) -> None:
        base_path = Path(dict_path) if dict_path is not None else Path(__file__).resolve().with_name("cmudict.7b")
        self.dict_path: Path = base_path
        self._pronunciations: Dict[str, List[List[str]]] = {}
        self._rhyme_parts: Dict[str, Set[str]] = {}
        self._rhyme_index: Dict[str, Set[str]] = {}
        self._loaded: bool = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return

        self._loaded = True

        if not self.dict_path.exists():
            return

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

                    self._pronunciations.setdefault(word, []).append(phones)

                    rhyme_part = self._extract_rhyme_part(phones)
                    if not rhyme_part:
                        continue

                    self._rhyme_parts.setdefault(word, set()).add(rhyme_part)
                    self._rhyme_index.setdefault(rhyme_part, set()).add(word)
        except OSError:
            # If the dictionary cannot be read we simply operate without cache.
            self._pronunciations.clear()
            self._rhyme_parts.clear()
            self._rhyme_index.clear()

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

RARITY_SIMILARITY_WEIGHT: float = 0.65
RARITY_NOVELTY_WEIGHT: float = 0.35


def _normalize_word(value: Optional[str]) -> str:
    return value.lower().strip() if value else ""


class WordRarityMap:
    """Stores approximate frequency information for rhyme candidates.

    The rarity map favours words that appear less frequently in the
    ``song_rhyme_patterns`` dataset (or any other supplied frequency list).
    Lower observed frequency translates into a higher rarity score so that
    downstream scoring can prioritise uncommon rhymes.
    """

    #: Fallback frequency data derived from the demo dataset embedded in
    #: :mod:`app`. The values loosely reflect how often each target word
    #: appears and provide sensible defaults when no database is available.
    _DEFAULT_FREQUENCIES: Dict[str, int] = {
        "love": 42,
        "above": 21,
        "mind": 18,
        "find": 17,
        "flow": 15,
        "go": 14,
        "money": 11,
        "honey": 9,
        "time": 10,
        "rhyme": 8,
        "night": 13,
        "light": 12,
        "pain": 7,
        "rain": 7,
        "real": 6,
        "feel": 6,
        "street": 5,
        "beat": 5,
        "life": 4,
        "knife": 3,
        "word": 4,
        "heard": 4,
        "game": 5,
        "fame": 5,
        "soul": 4,
        "goal": 3,
        "way": 5,
        "day": 5,
        "back": 6,
        "track": 6,
    }

    def __init__(self, frequencies: Optional[Dict[str, int]] = None) -> None:
        self._frequencies: Counter[str] = Counter()
        self._loaded_sources: Set[Path] = set()
        self._default_frequency: int = 1
        if frequencies:
            self.add_frequencies(frequencies)
        if not self._frequencies:
            self.add_frequencies(self._DEFAULT_FREQUENCIES)
        self._update_frequency_stats()

    def add_frequencies(self, frequencies: Dict[str, int]) -> None:
        for word, value in frequencies.items():
            normalized = _normalize_word(word)
            if not normalized:
                continue
            try:
                freq_value = int(value)
            except (TypeError, ValueError):
                continue
            if freq_value <= 0:
                continue
            self._frequencies[normalized] = max(self._frequencies.get(normalized, 0), freq_value)
        self._update_frequency_stats()

    def _update_frequency_stats(self) -> None:
        if not self._frequencies:
            self._min_frequency = 0
            self._max_frequency = 0
            self._default_frequency = 1
            return

        frequencies = list(self._frequencies.values())
        self._min_frequency = min(frequencies)
        self._max_frequency = max(frequencies)
        self._default_frequency = max(1, int(sum(frequencies) / max(len(frequencies), 1)))

    def get_frequency(self, word: str) -> int:
        normalized = _normalize_word(word)
        if not normalized:
            return self._default_frequency
        return self._frequencies.get(normalized, self._default_frequency)

    def get_rarity(self, word: str) -> float:
        frequency = max(self.get_frequency(word), 1)
        # Inverse log scaling keeps the range approximately within (0, 1].
        rarity = 1.0 / (1.0 + math.log1p(frequency))
        return float(rarity)

    def update_from_database(self, db_path: Optional[Path | str]) -> bool:
        if not db_path:
            return False

        db_file = Path(db_path)
        if not db_file.exists():
            return False

        canonical = db_file.resolve()
        if canonical in self._loaded_sources:
            return True

        try:
            with sqlite3.connect(str(canonical)) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT LOWER(target_word), COUNT(*)
                    FROM song_rhyme_patterns
                    WHERE target_word IS NOT NULL AND TRIM(target_word) != ''
                    GROUP BY LOWER(target_word)
                    """
                )
                rows = cursor.fetchall()
        except sqlite3.Error:
            return False

        frequency_map: Dict[str, int] = {
            row[0]: int(row[1]) for row in rows if isinstance(row, Iterable) and row and row[0]
        }

        if not frequency_map:
            return False

        self.add_frequencies(frequency_map)
        self._loaded_sources.add(canonical)
        return True


DEFAULT_RARITY_MAP = WordRarityMap()


@dataclass
class PhoneticMatch:
    """Represents a phonetic match between two words"""

    word1: str
    word2: str
    similarity_score: float
    phonetic_features: Dict[str, float]
    rhyme_type: str  # perfect, near, slant, etc.
    rarity_score: float = 0.0
    combined_score: float = 0.0

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
        self.vowel_groups = self._initialize_vowel_groups()
        self.consonant_groups = self._initialize_consonant_groups()
        self.phonetic_weights = self._initialize_phonetic_weights()

        print("📊 Enhanced Core Phonetic Analyzer initialized")
    
    def _initialize_vowel_groups(self) -> Dict[str, List[str]]:
        """Initialize vowel sound groupings for phonetic analysis"""
        return {
            'long_a': ['ay', 'ey', 'ai', 'ei'],
            'long_e': ['ee', 'ea', 'ie', 'ei'],
            'long_i': ['igh', 'ie', 'y', 'eye'],
            'long_o': ['ow', 'oe', 'oa', 'o'],
            'long_u': ['oo', 'ou', 'ew', 'ue'],
            'short_a': ['a', 'at', 'an', 'ap'],
            'short_e': ['e', 'en', 'et', 'ed'],
            'short_i': ['i', 'in', 'it', 'ip'],
            'short_o': ['o', 'on', 'ot', 'op'],
            'short_u': ['u', 'un', 'ut', 'up']
        }
    
    def _initialize_consonant_groups(self) -> Dict[str, List[str]]:
        """Initialize consonant sound groupings"""
        return {
            'stops': ['p', 'b', 't', 'd', 'k', 'g'],
            'fricatives': ['f', 'v', 's', 'z', 'sh', 'zh', 'th'],
            'nasals': ['m', 'n', 'ng'],
            'liquids': ['l', 'r'],
            'glides': ['w', 'y', 'h']
        }
    
    def _initialize_phonetic_weights(self) -> Dict[str, float]:
        """Initialize weights for different phonetic features"""
        return {
            'ending_sounds': 0.4,    # Most important for rhymes
            'vowel_sounds': 0.3,     # Critical for rhyme quality
            'consonant_clusters': 0.2, # Important for texture
            'syllable_structure': 0.1  # Secondary importance
        }
    
    def get_phonetic_similarity(self, word1: str, word2: str) -> float:
        """
        Calculate phonetic similarity between two words
        Returns score from 0.0 to 1.0
        """
        if not word1 or not word2:
            return 0.0
        
        if word1.lower() == word2.lower():
            return 1.0
        
        # Clean words
        clean1 = self._clean_word(word1)
        clean2 = self._clean_word(word2)
        
        # Calculate multiple phonetic features
        ending_sim = self._calculate_ending_similarity(clean1, clean2)
        vowel_sim = self._calculate_vowel_similarity(clean1, clean2)
        consonant_sim = self._calculate_consonant_similarity(clean1, clean2)
        syllable_sim = self._calculate_syllable_similarity(clean1, clean2)
        
        # Weight the similarities
        weights = self.phonetic_weights
        total_similarity = (
            ending_sim * weights['ending_sounds'] +
            vowel_sim * weights['vowel_sounds'] +
            consonant_sim * weights['consonant_clusters'] +
            syllable_sim * weights['syllable_structure']
        )
        
        return min(total_similarity, 1.0)
    
    def _clean_word(self, word: str) -> str:
        """Clean and normalize word for phonetic analysis"""
        cleaned = word.lower().strip()
        # Remove non-alphabetic characters but preserve apostrophes
        cleaned = re.sub(r"[^a-z']", '', cleaned)
        return cleaned
    
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
    
    def _calculate_vowel_similarity(self, word1: str, word2: str) -> float:
        """Calculate vowel pattern similarity"""
        vowels1 = re.findall(r'[aeiou]+', word1)
        vowels2 = re.findall(r'[aeiou]+', word2)
        
        if not vowels1 or not vowels2:
            return 0.0
        
        # Compare last vowel sounds (most important for rhymes)
        last_vowel1 = vowels1[-1] if vowels1 else ''
        last_vowel2 = vowels2[-1] if vowels2 else ''
        
        if last_vowel1 == last_vowel2:
            return 0.9
        
        # Check if vowels are in the same phonetic group
        similarity = self._check_vowel_group_similarity(last_vowel1, last_vowel2)
        return similarity
    
    def _check_vowel_group_similarity(self, vowel1: str, vowel2: str) -> float:
        """Check if vowels belong to similar phonetic groups"""
        for group_name, vowels in self.vowel_groups.items():
            if any(v in vowel1 for v in vowels) and any(v in vowel2 for v in vowels):
                return 0.7  # Same vowel group
        
        return 0.3  # Different vowel groups
    
    def _calculate_consonant_similarity(self, word1: str, word2: str) -> float:
        """Calculate consonant pattern similarity"""
        # Extract consonant patterns
        consonants1 = re.sub(r'[aeiou]', '', word1)
        consonants2 = re.sub(r'[aeiou]', '', word2)
        
        if not consonants1 or not consonants2:
            return 0.5
        
        # Focus on ending consonant clusters
        end_consonants1 = consonants1[-2:] if len(consonants1) >= 2 else consonants1
        end_consonants2 = consonants2[-2:] if len(consonants2) >= 2 else consonants2
        
        similarity = difflib.SequenceMatcher(None, end_consonants1, end_consonants2).ratio()
        return similarity
    
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
        word = word.lower()
        vowel_groups = re.findall(r'[aeiou]+', word)
        syllable_count = len(vowel_groups)
        
        # Adjust for silent 'e'
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
        
        return max(1, syllable_count)  # Minimum 1 syllable
    
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
        similarity = self.get_phonetic_similarity(word1, word2)
        rhyme_type = self.classify_rhyme_type(word1, word2, similarity)

        # Normalize words once for feature calculations to ensure consistent scoring
        clean_word1 = self._clean_word(word1)
        clean_word2 = self._clean_word(word2)

        # Calculate detailed phonetic features
        features = {
            'ending_similarity': self._calculate_ending_similarity(clean_word1, clean_word2),
            'vowel_similarity': self._calculate_vowel_similarity(clean_word1, clean_word2),
            'consonant_similarity': self._calculate_consonant_similarity(clean_word1, clean_word2),
            'syllable_similarity': self._calculate_syllable_similarity(clean_word1, clean_word2)
        }

        return PhoneticMatch(
            word1=word1,
            word2=word2,
            similarity_score=similarity,
            phonetic_features=features,
            rhyme_type=rhyme_type
        )

    def get_rhyme_candidates(self, target_word: str, word_list: List[str],
                           min_similarity: float = 0.7) -> List[PhoneticMatch]:
        """Find rhyme candidates from a word list"""
        candidates = []

        for word in word_list:
            if word.lower() != target_word.lower():
                match = self.analyze_rhyme_pattern(target_word, word)
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


RhymeCandidate = Tuple[str, float, float, float]


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
        A list of tuples ``(candidate_word, similarity_score, rarity_score,
        combined_score)`` sorted by descending combined score. Returns an
        empty list when no candidates are found in either the local CMU cache
        or the pronouncing fallback.
    """

    if not word or not word.strip() or limit <= 0:
        return []

    normalized_word = word.strip().lower()

    loader = cmu_loader
    if loader is None and analyzer is not None:
        loader = getattr(analyzer, "cmu_loader", None)
    if loader is None:
        loader = DEFAULT_CMU_LOADER

    local_candidates: List[str] = []
    if loader is not None:
        try:
            local_candidates = loader.get_rhyming_words(normalized_word)
        except Exception:
            local_candidates = []

    candidate_words: List[str] = list(local_candidates)

    if not candidate_words and pronouncing is not None:
        try:
            candidates = pronouncing.rhymes(normalized_word)

            if not candidates:
                phones = pronouncing.phones_for_word(normalized_word)
                for phone in phones:
                    rhyme_part = pronouncing.rhyming_part(phone)
                    if rhyme_part:
                        pattern = f".*{rhyme_part}"
                        candidates.extend(pronouncing.search(pattern))

            seen = set()
            for candidate in candidates:
                cleaned = candidate.strip().lower()
                if not cleaned or cleaned == normalized_word or cleaned in seen:
                    continue
                seen.add(cleaned)
                candidate_words.append(cleaned)
        except Exception:
            candidate_words = []

    if not candidate_words:
        return []

    scoring_analyzer = analyzer or EnhancedPhoneticAnalyzer()

    rarity_source = getattr(scoring_analyzer, "rarity_map", None) or DEFAULT_RARITY_MAP
    combine_fn = getattr(scoring_analyzer, "combine_similarity_and_rarity", None)
    scored_candidates: List[RhymeCandidate] = []
    for candidate in candidate_words:
        try:
            score = scoring_analyzer.get_phonetic_similarity(normalized_word, candidate)
        except Exception:
            score = 0.0
        try:
            rarity = float(rarity_source.get_rarity(candidate))
        except Exception:
            rarity = DEFAULT_RARITY_MAP.get_rarity(candidate)

        if callable(combine_fn):
            try:
                combined = float(combine_fn(score, rarity))
            except Exception:
                combined = score
        else:
            combined = score
        scored_candidates.append((candidate, score, rarity, combined))

    scored_candidates.sort(key=lambda item: item[3], reverse=True)
    return scored_candidates[:limit]

# Example usage and testing
if __name__ == "__main__":
    analyzer = EnhancedPhoneticAnalyzer()
    
    # Test some rhyme pairs
    test_pairs = [
        ("love", "above"),
        ("mind", "find"),
        ("flow", "go"),
        ("money", "honey"),
        ("time", "rhyme")
    ]
    
    print("🎵 Testing Enhanced Phonetic Analyzer:")
    print("=" * 50)
    
    for word1, word2 in test_pairs:
        match = analyzer.analyze_rhyme_pattern(word1, word2)
        print(f"'{word1}' / '{word2}': {match.similarity_score:.3f} ({match.rhyme_type})")
        
    print("\n✅ Module 1 Enhanced Core Phonetic ready for integration")
