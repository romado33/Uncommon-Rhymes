"""Phonetic analysis utilities powering the RhymeRarity project."""

from __future__ import annotations

import difflib
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from syllable_utils import estimate_syllable_count

from .cmudict_loader import CMUDictLoader, DEFAULT_CMU_LOADER
from .feature_profile import (
    PhraseComponents,
    PhoneticMatch,
    RhymeFeatureProfile,
    extract_phrase_components,
    pronouncing,
)
from .rarity_map import DEFAULT_RARITY_MAP, WordRarityMap

RARITY_SIMILARITY_WEIGHT: float = 0.65
RARITY_NOVELTY_WEIGHT: float = 0.35

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

        clean1 = self._clean_word(word1)
        clean2 = self._clean_word(word2)

        if not clean1 or not clean2:
            return 0.0

        if clean1 == clean2:
            return 1.0

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
    
    def _phrase_components(self, word: str) -> PhraseComponents:
        loader = getattr(self, "cmu_loader", None)
        return extract_phrase_components(word or "", loader)

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
        consonants1 = re.sub(r"[^a-z']", "", re.sub(r"[aeiou]", "", word1))
        consonants2 = re.sub(r"[^a-z']", "", re.sub(r"[aeiou]", "", word2))
        
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

        if components.anchor_pronunciations:
            return [list(phones) for phones in components.anchor_pronunciations]

        lookup_word = components.anchor
        if not lookup_word:
            lookup_word = re.sub(r"\s+", "", components.normalized_phrase)

        if not lookup_word:
            return []

        try:
            pronunciations = loader.get_pronunciations(lookup_word)
        except Exception:
            pronunciations = []

        if not pronunciations and pronouncing is not None:
            try:
                pronunciations = [
                    phones.split()
                    for phones in pronouncing.phones_for_word(lookup_word)
                ]
            except Exception:
                pronunciations = []

        return pronunciations

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
            assonance_score=assonance,
            consonance_score=consonance_value,
            internal_rhyme_score=internal_rhyme_score,
            bradley_device=bradley_device,
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

        return self.build_feature_profile(
            source_word,
            target_word,
            similarity=similarity,
            rhyme_type=rhyme_type,
        )
    
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
            'syllable_similarity': self._calculate_syllable_similarity(clean_word1, clean_word2)
        }

        if profile is not None:
            features.update(
                {
                    'assonance_score': profile.assonance_score,
                    'consonance_score': profile.consonance_score,
                    'internal_rhyme_score': profile.internal_rhyme_score,
                }
            )

        return PhoneticMatch(
            word1=word1,
            word2=word2,
            similarity_score=similarity,
            phonetic_features=features,
            rhyme_type=rhyme_type,
            feature_profile=profile,
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


RhymeCandidate = Dict[str, Any]


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
        except Exception:
            candidate_words = []

    if not candidate_words:
        return []

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

    def _score_variant(suggestion: str, *, is_multi: bool, base_candidate: str) -> None:
        normalized_suggestion = suggestion.strip()
        if not normalized_suggestion:
            return
        normalized_key = normalized_suggestion.lower()
        if normalized_key == normalized_phrase.lower():
            return
        if normalized_key in seen_variants:
            return
        seen_variants.add(normalized_key)

        try:
            score = float(scoring_analyzer.get_phonetic_similarity(base_phrase, suggestion))
        except Exception:
            score = 0.0

        try:
            if " " in normalized_suggestion:
                token_scores: List[float] = []
                for token in normalized_suggestion.split():
                    try:
                        token_scores.append(float(rarity_source.get_rarity(token)))
                    except Exception:
                        token_scores.append(float(DEFAULT_RARITY_MAP.get_rarity(token)))
                rarity = sum(token_scores) / len(token_scores) if token_scores else 0.0
            else:
                rarity = float(rarity_source.get_rarity(base_candidate))
        except Exception:
            rarity = DEFAULT_RARITY_MAP.get_rarity(base_candidate)

        if callable(combine_fn):
            try:
                combined = float(combine_fn(score, rarity))
            except Exception:
                combined = score
        else:
            combined = score

        candidate_info = extract_phrase_components(suggestion, loader)

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
            "is_multi_word": is_multi,
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
        }

        scored_candidates.append(entry)

    for candidate in candidate_words:
        base_candidate = candidate.strip()
        if not base_candidate:
            continue
        if base_candidate == anchor_lookup:
            continue

        _score_variant(base_candidate, is_multi=False, base_candidate=base_candidate)

        if components.total_syllables >= 2:
            multi_tokens = list(prefix_tokens_norm)
            multi_tokens.append(base_candidate)
            multi_tokens.extend(suffix_tokens_norm)
            multi_phrase = " ".join(multi_tokens).strip()
            if multi_phrase and multi_phrase != base_candidate:
                _score_variant(multi_phrase, is_multi=True, base_candidate=base_candidate)

    scored_candidates.sort(key=lambda item: item.get("combined", 0.0), reverse=True)
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
