#!/usr/bin/env python3
"""
Module 1: Enhanced Core Phonetic Analysis for RhymeRarity
Handles phonetic similarity calculations and rhyme detection algorithms
Part of the RhymeRarity system deployed on Hugging Face
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import difflib

@dataclass
class PhoneticMatch:
    """Represents a phonetic match between two words"""
    word1: str
    word2: str
    similarity_score: float
    phonetic_features: Dict[str, float]
    rhyme_type: str  # perfect, near, slant, etc.

class EnhancedPhoneticAnalyzer:
    """
    Enhanced phonetic analysis system for superior rhyme detection
    Implements research-backed phonetic similarity algorithms
    """
    
    def __init__(self):
        # Initialize phonetic analysis components
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
        
        # Calculate detailed phonetic features
        features = {
            'ending_similarity': self._calculate_ending_similarity(word1, word2),
            'vowel_similarity': self._calculate_vowel_similarity(word1, word2),
            'consonant_similarity': self._calculate_consonant_similarity(word1, word2),
            'syllable_similarity': self._calculate_syllable_similarity(word1, word2)
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
                    candidates.append(match)
        
        # Sort by similarity score
        candidates.sort(key=lambda x: x.similarity_score, reverse=True)
        return candidates

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
