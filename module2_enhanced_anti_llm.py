#!/usr/bin/env python3
"""
Module 2: Enhanced Anti-LLM Rhyme Engine for RhymeRarity
Implements algorithms specifically designed to outperform Large Language Models
Part of the RhymeRarity system deployed on Hugging Face
"""

import sqlite3
import random
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict, Counter
import re

@dataclass
class AntiLLMPattern:
    """Represents a rhyme pattern designed to challenge LLM capabilities"""
    source_word: str
    target_word: str
    rarity_score: float
    cultural_depth: str
    llm_weakness_type: str  # phonological, cultural, frequency, etc.
    confidence: float

class AntiLLMRhymeEngine:
    """
    Enhanced rhyme engine specifically designed to outperform LLMs
    Exploits documented weaknesses in large language model rhyme generation
    """
    
    def __init__(self, db_path: str = "patterns.db"):
        self.db_path = db_path
        
        # Initialize anti-LLM strategies
        self.llm_weaknesses = self._initialize_llm_weaknesses()
        self.rarity_multipliers = self._initialize_rarity_multipliers()
        self.cultural_depth_weights = self._initialize_cultural_weights()
        
        # Performance tracking
        self.anti_llm_stats = {
            'rare_patterns_generated': 0,
            'cultural_patterns_found': 0,
            'phonological_challenges': 0,
            'frequency_inversions': 0
        }
        
        print("🚀 Enhanced Anti-LLM Rhyme Engine initialized")
        print("Targeting documented LLM weaknesses in phonological processing")
    
    def _initialize_llm_weaknesses(self) -> Dict[str, Dict]:
        """Initialize known LLM weaknesses in rhyme generation"""
        return {
            'phonological_processing': {
                'description': 'LLMs struggle with complex phoneme analysis',
                'target_patterns': ['consonant_clusters', 'vowel_shifts', 'silent_letters'],
                'difficulty_multiplier': 2.5
            },
            'rare_word_combinations': {
                'description': 'LLMs fail with low-frequency word pairs',
                'target_patterns': ['archaic_words', 'slang_terms', 'regional_variants'],
                'difficulty_multiplier': 3.0
            },
            'cultural_context': {
                'description': 'LLMs lack authentic cultural attribution',
                'target_patterns': ['artist_specific', 'era_specific', 'genre_specific'],
                'difficulty_multiplier': 2.2
            },
            'multi_syllable_complexity': {
                'description': 'LLMs struggle with complex syllable patterns',
                'target_patterns': ['internal_rhymes', 'compound_words', 'polysyllabic'],
                'difficulty_multiplier': 2.8
            },
            'semantic_interference': {
                'description': 'LLMs confused by semantic similarity vs phonetic',
                'target_patterns': ['semantic_opposites', 'false_friends', 'homophone_traps'],
                'difficulty_multiplier': 2.4
            }
        }
    
    def _initialize_rarity_multipliers(self) -> Dict[str, float]:
        """Initialize rarity scoring multipliers"""
        return {
            'ultra_rare': 4.0,     # <0.1% frequency in training data
            'very_rare': 3.0,      # 0.1-1% frequency
            'rare': 2.0,           # 1-5% frequency
            'uncommon': 1.5,       # 5-15% frequency
            'common': 1.0,         # >15% frequency
        }
    
    def _initialize_cultural_weights(self) -> Dict[str, float]:
        """Initialize cultural significance weights"""
        return {
            'underground': 3.5,     # Underground/independent artists
            'regional': 3.0,        # Region-specific slang/patterns
            'era_specific': 2.8,    # Time period specific language
            'artist_signature': 2.5, # Artist's unique style
            'mainstream': 1.0       # Widely known patterns
        }
    
    def generate_anti_llm_patterns(self, source_word: str, limit: int = 20) -> List[AntiLLMPattern]:
        """
        Generate rhyme patterns specifically designed to challenge LLMs
        Focuses on documented weaknesses in LLM rhyme generation
        """
        if not source_word or not source_word.strip():
            return []

        if limit <= 0:
            return []

        source_word = source_word.lower().strip()
        patterns = []
        per_strategy = max(1, limit // 4)
        strategy_functions = [
            self._find_rare_combinations,
            self._find_phonological_challenges,
            self._find_cultural_depth_patterns,
            self._find_complex_syllable_patterns,
        ]

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                for finder in strategy_functions:
                    if len(patterns) >= limit:
                        break

                    remaining = limit - len(patterns)
                    strategy_limit = min(per_strategy, remaining)

                    if strategy_limit <= 0:
                        break

                    new_patterns = finder(cursor, source_word, strategy_limit)
                    if new_patterns:
                        patterns.extend(new_patterns[:strategy_limit])

            # Sort by anti-LLM effectiveness
            patterns.sort(key=lambda x: x.rarity_score * x.confidence, reverse=True)

            return patterns[:limit]
            
        except sqlite3.Error as e:
            print(f"Database error in anti-LLM engine: {e}")
            return []
    
    def _find_rare_combinations(self, cursor, source_word: str, limit: int) -> List[AntiLLMPattern]:
        """Find rare word combinations that LLMs typically miss"""
        query = """
        SELECT target_word, artist, song_title, confidence_score, cultural_significance
        FROM song_rhyme_patterns 
        WHERE source_word = ? 
          AND source_word != target_word
          AND length(target_word) >= 4
          AND confidence_score >= 0.7
        ORDER BY RANDOM()
        LIMIT ?
        """
        
        cursor.execute(query, (source_word, limit * 3))
        results = cursor.fetchall()
        
        patterns = []
        for target, artist, song, confidence, cultural_sig in results:
            # Calculate rarity score based on word frequency heuristics
            rarity_score = self._calculate_word_rarity(target)
            
            # Boost score if it's from underground/regional artists
            cultural_multiplier = self.cultural_depth_weights.get(cultural_sig, 1.0)
            final_rarity = rarity_score * cultural_multiplier
            
            if final_rarity >= 2.0:  # Only high-rarity patterns
                patterns.append(AntiLLMPattern(
                    source_word=source_word,
                    target_word=target,
                    rarity_score=final_rarity,
                    cultural_depth=f"{artist} - {song}",
                    llm_weakness_type="rare_word_combinations",
                    confidence=confidence
                ))
                
                self.anti_llm_stats['rare_patterns_generated'] += 1
        
        return patterns[:limit]
    
    def _find_phonological_challenges(self, cursor, source_word: str, limit: int) -> List[AntiLLMPattern]:
        """Find patterns that exploit LLM phonological processing weaknesses"""
        # Target words with complex phonological features
        query = """
        SELECT target_word, artist, song_title, confidence_score, phonetic_similarity
        FROM song_rhyme_patterns 
        WHERE source_word = ? 
          AND source_word != target_word
          AND confidence_score BETWEEN 0.7 AND 0.9
          AND phonetic_similarity >= 0.8
        ORDER BY phonetic_similarity DESC
        LIMIT ?
        """
        
        cursor.execute(query, (source_word, limit * 2))
        results = cursor.fetchall()
        
        patterns = []
        for target, artist, song, confidence, phonetic_sim in results:
            # Identify phonological complexity features
            complexity_score = self._analyze_phonological_complexity(source_word, target)
            
            if complexity_score >= 2.0:
                patterns.append(AntiLLMPattern(
                    source_word=source_word,
                    target_word=target,
                    rarity_score=complexity_score,
                    cultural_depth=f"{artist} - {song}",
                    llm_weakness_type="phonological_processing",
                    confidence=confidence
                ))
                
                self.anti_llm_stats['phonological_challenges'] += 1
        
        return patterns[:limit]
    
    def _find_cultural_depth_patterns(self, cursor, source_word: str, limit: int) -> List[AntiLLMPattern]:
        """Find patterns with authentic cultural depth that LLMs lack"""
        query = """
        SELECT target_word, artist, song_title, confidence_score, cultural_significance
        FROM song_rhyme_patterns 
        WHERE source_word = ? 
          AND source_word != target_word
          AND cultural_significance IN ('underground', 'regional', 'era_specific', 'artist_signature')
        ORDER BY confidence_score DESC
        LIMIT ?
        """
        
        cursor.execute(query, (source_word, limit * 2))
        results = cursor.fetchall()
        
        patterns = []
        for target, artist, song, confidence, cultural_sig in results:
            cultural_weight = self.cultural_depth_weights.get(cultural_sig, 1.0)
            
            patterns.append(AntiLLMPattern(
                source_word=source_word,
                target_word=target,
                rarity_score=cultural_weight,
                cultural_depth=f"{artist} - {song} [{cultural_sig}]",
                llm_weakness_type="cultural_context",
                confidence=confidence
            ))
            
            self.anti_llm_stats['cultural_patterns_found'] += 1
        
        return patterns[:limit]
    
    def _find_complex_syllable_patterns(self, cursor, source_word: str, limit: int) -> List[AntiLLMPattern]:
        """Find multi-syllable complexity patterns that challenge LLMs"""
        query = """
        SELECT target_word, artist, song_title, confidence_score
        FROM song_rhyme_patterns 
        WHERE source_word = ? 
          AND source_word != target_word
          AND length(target_word) >= 6
          AND confidence_score >= 0.75
        ORDER BY length(target_word) DESC
        LIMIT ?
        """
        
        cursor.execute(query, (source_word, limit * 2))
        results = cursor.fetchall()
        
        patterns = []
        for target, artist, song, confidence in results:
            # Calculate syllable complexity
            syllable_complexity = self._calculate_syllable_complexity(target)
            
            if syllable_complexity >= 2.0:
                patterns.append(AntiLLMPattern(
                    source_word=source_word,
                    target_word=target,
                    rarity_score=syllable_complexity,
                    cultural_depth=f"{artist} - {song}",
                    llm_weakness_type="multi_syllable_complexity",
                    confidence=confidence
                ))
        
        return patterns[:limit]
    
    def _calculate_word_rarity(self, word: str) -> float:
        """Estimate word rarity based on linguistic features"""
        rarity_factors = 0.0
        
        # Length factor (longer words tend to be rarer)
        if len(word) >= 7:
            rarity_factors += 1.5
        elif len(word) >= 5:
            rarity_factors += 1.0
        
        # Complex consonant clusters
        consonant_clusters = len(re.findall(r'[bcdfghjklmnpqrstvwxyz]{3,}', word))
        rarity_factors += consonant_clusters * 0.5
        
        # Uncommon letter combinations
        rare_combinations = ['xh', 'qw', 'kn', 'gn', 'wr', 'mb', 'bt']
        for combo in rare_combinations:
            if combo in word:
                rarity_factors += 0.8
        
        # Silent letters (indicator of complexity)
        if word.endswith('e') and len(word) > 4:
            rarity_factors += 0.3
        
        return min(rarity_factors, 4.0)  # Cap at maximum rarity
    
    def _analyze_phonological_complexity(self, word1: str, word2: str) -> float:
        """Analyze phonological complexity that challenges LLMs"""
        complexity = 0.0
        
        # Vowel shift patterns (LLM weakness)
        vowels1 = re.findall(r'[aeiou]+', word1)
        vowels2 = re.findall(r'[aeiou]+', word2)
        
        if vowels1 and vowels2 and vowels1[-1] != vowels2[-1]:
            complexity += 1.0  # Different vowel sounds but still rhyme
        
        # Consonant cluster complexity
        consonants1 = re.sub(r'[aeiou]', '', word1)
        consonants2 = re.sub(r'[aeiou]', '', word2)
        
        if len(consonants1) >= 3 or len(consonants2) >= 3:
            complexity += 1.5
        
        # Silent letter challenges
        if (word1.endswith('e') and not word2.endswith('e')) or (word2.endswith('e') and not word1.endswith('e')):
            complexity += 1.0
        
        return complexity
    
    def _calculate_syllable_complexity(self, word: str) -> float:
        """Calculate syllable-based complexity"""
        syllable_count = len(re.findall(r'[aeiou]+', word))
        
        complexity = syllable_count * 0.5
        
        # Bonus for compound-looking words
        if len(word) >= 8 and '-' not in word:
            complexity += 1.0
        
        # Bonus for alternating consonant-vowel patterns (harder for LLMs)
        pattern_changes = 0
        for i in range(1, len(word)):
            is_vowel_now = word[i] in 'aeiou'
            was_vowel_before = word[i-1] in 'aeiou'
            if is_vowel_now != was_vowel_before:
                pattern_changes += 1
        
        complexity += pattern_changes * 0.1
        
        return complexity
    
    def get_anti_llm_effectiveness_score(self, pattern: AntiLLMPattern) -> float:
        """Calculate overall anti-LLM effectiveness score"""
        base_score = pattern.rarity_score * pattern.confidence
        
        # Bonus multipliers based on LLM weakness type
        weakness_multiplier = self.llm_weaknesses[pattern.llm_weakness_type]['difficulty_multiplier']
        
        return base_score * weakness_multiplier
    
    def get_performance_stats(self) -> Dict:
        """Get anti-LLM engine performance statistics"""
        return {
            'anti_llm_stats': self.anti_llm_stats.copy(),
            'llm_weaknesses_targeted': len(self.llm_weaknesses),
            'rarity_levels_available': len(self.rarity_multipliers),
            'cultural_depth_categories': len(self.cultural_depth_weights)
        }

# Example usage and testing
if __name__ == "__main__":
    engine = AntiLLMRhymeEngine()
    
    # Test anti-LLM pattern generation
    test_words = ["love", "mind", "flow"]
    
    print("🎯 Testing Anti-LLM Rhyme Engine:")
    print("=" * 50)

    for limit in range(1, 5):
        print(f"\n🔢 Evaluating engine output with limit={limit}")

        for word in test_words:
            patterns = engine.generate_anti_llm_patterns(word, limit=limit)
            print(f"\nAnti-LLM patterns for '{word}' (requested {limit}, received {len(patterns)}):")

            if not patterns:
                print("  ⚠️ No patterns found; ensure the database contains entries for this source word.")
                continue

            for pattern in patterns:
                effectiveness = engine.get_anti_llm_effectiveness_score(pattern)
                print(f"  • '{pattern.target_word}' - {pattern.cultural_depth}")
                print(f"    Weakness: {pattern.llm_weakness_type} | Score: {effectiveness:.2f}")

            print(f"  ✅ Retrieved {len(patterns)} pattern(s) within the requested limit.")

    print(f"\n📊 Performance Stats:")
    stats = engine.get_performance_stats()
    for key, value in stats.items():
        print(f"  • {key}: {value}")
    
    print("\n✅ Module 2 Enhanced Anti-LLM ready for integration")
