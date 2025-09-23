#!/usr/bin/env python3
"""
Module 2: Enhanced Anti-LLM Rhyme Engine for RhymeRarity
Implements algorithms specifically designed to outperform Large Language Models
Part of the RhymeRarity system deployed on Hugging Face
"""

import sqlite3
import random
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
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
    bradley_device: str = "undetermined"
    syllable_span: Tuple[int, int] = (0, 0)
    stress_alignment: float = 0.0
    feature_profile: Dict[str, Any] = field(default_factory=dict)
    prosody_profile: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SeedCandidate:
    """Normalized representation of Module 1 seed rhymes."""

    word: str
    rarity: float = 0.0
    combined: float = 0.0
    signatures: Set[str] = field(default_factory=set)
    feature_profile: Dict[str, Any] = field(default_factory=dict)
    prosody_profile: Dict[str, Any] = field(default_factory=dict)

    def normalized(self) -> str:
        return self.word.lower().strip()

class AntiLLMRhymeEngine:
    """
    Enhanced rhyme engine specifically designed to outperform LLMs
    Exploits documented weaknesses in large language model rhyme generation
    """
    
    def __init__(self, db_path: str = "patterns.db", phonetic_analyzer: Optional[Any] = None):
        self.db_path = db_path
        self.phonetic_analyzer = phonetic_analyzer

        # Initialize anti-LLM strategies
        self.llm_weaknesses = self._initialize_llm_weaknesses()
        self.rarity_multipliers = self._initialize_rarity_multipliers()
        self.cultural_depth_weights = self._initialize_cultural_weights()
        
        # Performance tracking
        self.anti_llm_stats = {
            'rare_patterns_generated': 0,
            'cultural_patterns_found': 0,
            'phonological_challenges': 0,
            'frequency_inversions': 0,
            'seed_expansions': 0,
        }

        # Lazy-loaded resources for seed-driven exploration
        self._seed_resources_initialized = False
        self._seed_analyzer = None
        self._cmu_seed_fn = None
        
        print("🚀 Enhanced Anti-LLM Rhyme Engine initialized")
        print("Targeting documented LLM weaknesses in phonological processing")

    def set_phonetic_analyzer(self, analyzer: Any) -> None:
        """Attach an analyzer so that generated patterns expose feature profiles."""

        self.phonetic_analyzer = analyzer
    
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

    # ------------------------------------------------------------------
    # Feature profile support
    # ------------------------------------------------------------------

    def _extract_feature_profile(self, source_word: str, target_word: str) -> Dict[str, Any]:
        analyzer = getattr(self, "phonetic_analyzer", None)
        if analyzer is None:
            return {}

        builder = getattr(analyzer, "derive_rhyme_profile", None)
        if not callable(builder):
            builder = getattr(analyzer, "build_feature_profile", None)
        if not callable(builder):
            return {}

        try:
            profile = builder(source_word, target_word)
        except Exception:
            return {}

        if profile is None:
            return {}

        if hasattr(profile, "as_dict"):
            try:
                return dict(profile.as_dict())
            except Exception:
                pass

        if isinstance(profile, dict):
            return dict(profile)

        try:
            return dict(profile.__dict__)
        except Exception:
            return {}

    def _attach_profile(self, pattern: AntiLLMPattern) -> None:
        profile = self._extract_feature_profile(pattern.source_word, pattern.target_word)
        if not profile:
            return

        pattern.feature_profile = profile

        bradley = profile.get("bradley_device")
        if isinstance(bradley, str) and bradley:
            pattern.bradley_device = bradley

        syllable_span = profile.get("syllable_span")
        if isinstance(syllable_span, (list, tuple)) and len(syllable_span) == 2:
            try:
                pattern.syllable_span = (int(syllable_span[0]), int(syllable_span[1]))
            except (TypeError, ValueError):
                pass

        stress_alignment = profile.get("stress_alignment")
        if isinstance(stress_alignment, (int, float)):
            pattern.stress_alignment = float(stress_alignment)

        internal_score = profile.get("internal_rhyme_score")
        if isinstance(internal_score, (int, float)):
            pattern.rarity_score *= 1.0 + max(0.0, float(internal_score)) * 0.15

        pattern.prosody_profile = {
            "complexity_tag": "dense"
            if profile.get("syllable_span") and max(profile["syllable_span"]) >= 3
            else "steady",
            "stress_alignment": pattern.stress_alignment,
            "assonance": profile.get("assonance_score"),
            "consonance": profile.get("consonance_score"),
        }
    
    def generate_anti_llm_patterns(
        self,
        source_word: str,
        limit: int = 20,
        module1_seeds: Optional[List[Any]] = None,
        seed_signatures: Optional[Set[str]] = None,
        delivered_words: Optional[Set[str]] = None,
    ) -> List[AntiLLMPattern]:
        """Generate rhyme patterns that extend Module 1's rarest discoveries.

        Args:
            source_word: Root word we are generating rhymes for.
            limit: Maximum number of anti-LLM patterns to return.
            module1_seeds: Optional seed rhymes discovered by Module 1. Entries
                may be strings or dictionaries containing ``word`` /
                ``target_word`` alongside rarity and signature metadata.
            seed_signatures: Optional phonetic fingerprints derived from
                Module 1, used to align second-order explorations.
            delivered_words: Set of words already surfaced by Module 1 so they
                can be excluded from Module 2 output.
        """

        if not source_word or not source_word.strip():
            return []

        if limit <= 0:
            return []

        source_word = source_word.lower().strip()
        patterns: List[AntiLLMPattern] = []

        signature_hints: Set[str] = set()
        if seed_signatures:
            signature_hints.update({str(sig).strip() for sig in seed_signatures if sig})

        normalized_seeds = self._normalize_seed_candidates(module1_seeds)
        for seed in normalized_seeds:
            signature_hints.update(seed.signatures)

        delivered_set: Set[str] = {source_word}
        if delivered_words:
            delivered_set.update({str(word).lower().strip() for word in delivered_words if word})
        delivered_set.update({seed.normalized() for seed in normalized_seeds})

        seen_targets: Set[str] = set(delivered_set)

        strategy_functions = [
            self._find_rare_combinations,
            self._find_phonological_challenges,
            self._find_cultural_depth_patterns,
            self._find_complex_syllable_patterns,
        ]

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                if normalized_seeds:
                    seed_patterns = self._expand_from_seed_candidates(
                        cursor,
                        source_word,
                        normalized_seeds,
                        limit,
                        signature_hints,
                        seen_targets,
                    )

                    for pattern in seed_patterns:
                        if len(patterns) >= limit:
                            break
                        normalized_target = pattern.target_word.lower().strip()
                        if not normalized_target or normalized_target in seen_targets:
                            continue
                        patterns.append(pattern)
                        seen_targets.add(normalized_target)

                per_strategy = max(1, limit // max(len(strategy_functions), 1))

                for finder in strategy_functions:
                    if len(patterns) >= limit:
                        break

                    remaining = limit - len(patterns)
                    strategy_limit = min(per_strategy, remaining)

                    if strategy_limit <= 0:
                        break

                    fetched = finder(cursor, source_word, strategy_limit * 2)
                    added = 0

                    for pattern in fetched:
                        if len(patterns) >= limit or added >= strategy_limit:
                            break
                        normalized_target = pattern.target_word.lower().strip()
                        if not normalized_target or normalized_target in seen_targets:
                            continue
                        patterns.append(pattern)
                        seen_targets.add(normalized_target)
                        added += 1

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

                self._attach_profile(patterns[-1])

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

                self._attach_profile(patterns[-1])

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

            self._attach_profile(patterns[-1])

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

                self._attach_profile(patterns[-1])

        return patterns[:limit]

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    def _normalize_seed_candidates(self, module1_seeds: Optional[List[Any]]) -> List[SeedCandidate]:
        normalized: List[SeedCandidate] = []
        if not module1_seeds:
            return normalized

        seen: Set[str] = set()

        for raw_seed in module1_seeds:
            word: Optional[str] = None
            rarity_value = 0.0
            combined_value = 0.0
            signatures: Set[str] = set()
            feature_profile: Dict[str, Any] = {}
            prosody_profile: Dict[str, Any] = {}

            if isinstance(raw_seed, str):
                word = raw_seed
            elif isinstance(raw_seed, dict):
                word = raw_seed.get("word") or raw_seed.get("target_word") or raw_seed.get("candidate")
                rarity_value = self._safe_float(
                    raw_seed.get("rarity")
                    or raw_seed.get("rarity_score")
                    or raw_seed.get("module1_rarity"),
                    default=0.0,
                )
                combined_value = self._safe_float(
                    raw_seed.get("combined")
                    or raw_seed.get("combined_score")
                    or raw_seed.get("confidence"),
                    default=0.0,
                )
                signature_values = (
                    raw_seed.get("signatures")
                    or raw_seed.get("target_rhyme_signatures")
                    or raw_seed.get("matched_signatures")
                )
                if signature_values:
                    if isinstance(signature_values, (set, list, tuple)):
                        for sig in signature_values:
                            if sig:
                                signatures.add(str(sig))
                    else:
                        signatures.add(str(signature_values))
                feature_candidate = raw_seed.get("feature_profile")
                if isinstance(feature_candidate, dict):
                    feature_profile = dict(feature_candidate)
                prosody_candidate = raw_seed.get("prosody_profile")
                if isinstance(prosody_candidate, dict):
                    prosody_profile = dict(prosody_candidate)
            else:
                try:
                    word = raw_seed[0]
                    if len(raw_seed) > 2:
                        rarity_value = self._safe_float(raw_seed[2], default=0.0)
                    if len(raw_seed) > 3:
                        combined_value = self._safe_float(raw_seed[3], default=0.0)
                except Exception:
                    continue

            if not word:
                continue

            word_str = str(word).strip()
            if not word_str:
                continue

            normalized_word = word_str.lower()
            if normalized_word in seen:
                continue

            seen.add(normalized_word)

            if rarity_value <= 0:
                rarity_value = self._calculate_word_rarity(normalized_word)

            normalized.append(
                SeedCandidate(
                    word=word_str,
                    rarity=rarity_value,
                    combined=combined_value,
                    signatures=set(signatures),
                    feature_profile=feature_profile,
                    prosody_profile=prosody_profile,
                )
            )

        normalized.sort(key=lambda seed: (seed.rarity, seed.combined), reverse=True)
        return normalized[:8]

    def _ensure_seed_resources(self) -> None:
        if self._seed_resources_initialized:
            return

        self._seed_resources_initialized = True
        try:
            from module1_enhanced_core_phonetic import (  # type: ignore
                EnhancedPhoneticAnalyzer,
                get_cmu_rhymes,
            )

            self._seed_analyzer = EnhancedPhoneticAnalyzer()
            self._cmu_seed_fn = get_cmu_rhymes
        except Exception:
            self._seed_analyzer = None
            self._cmu_seed_fn = None

    def _normalize_module1_candidates(self, candidates: Optional[List[Any]]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        if not candidates:
            return normalized

        seen: Set[str] = set()

        for candidate in candidates:
            word: Optional[str] = None
            similarity = 0.0
            combined = 0.0
            rarity = 0.0

            if isinstance(candidate, dict):
                word = (
                    candidate.get("word")
                    or candidate.get("target")
                    or candidate.get("candidate")
                )
                similarity = self._safe_float(
                    candidate.get("similarity") or candidate.get("score"),
                    default=0.0,
                )
                combined = self._safe_float(
                    candidate.get("combined")
                    or candidate.get("combined_score")
                    or candidate.get("confidence"),
                    default=similarity,
                )
                rarity = self._safe_float(
                    candidate.get("rarity") or candidate.get("rarity_score"),
                    default=0.0,
                )
            else:
                try:
                    word = candidate[0]
                    similarity = self._safe_float(
                        candidate[1] if len(candidate) > 1 else 0.0,
                        default=0.0,
                    )
                    rarity = self._safe_float(
                        candidate[2] if len(candidate) > 2 else 0.0,
                        default=0.0,
                    )
                    combined = self._safe_float(
                        candidate[3] if len(candidate) > 3 else similarity,
                        default=similarity,
                    )
                except Exception:
                    continue

            if not word:
                continue

            text_word = str(word).strip()
            lowered = text_word.lower()
            if not text_word or lowered in seen:
                continue

            seen.add(lowered)
            normalized.append(
                {
                    "candidate": text_word,
                    "confidence": combined,
                    "combined": combined,
                    "similarity": similarity,
                    "rarity": rarity,
                    "source": "Module 1 cascade",
                }
            )

        return normalized

    def _expand_from_seed_candidates(
        self,
        cursor,
        source_word: str,
        seeds: List[SeedCandidate],
        limit: int,
        signature_hints: Set[str],
        seen_targets: Set[str],
    ) -> List[AntiLLMPattern]:
        results: List[AntiLLMPattern] = []
        if not seeds or limit <= 0:
            return results

        self._ensure_seed_resources()
        cmu_fn = getattr(self, "_cmu_seed_fn", None)
        analyzer = getattr(self, "_seed_analyzer", None)

        per_seed_limit = max(1, (limit // max(len(seeds), 1)) + 1)

        for seed in seeds:
            if len(results) >= limit:
                break

            seed_word = seed.word
            normalized_seed = seed.normalized()
            combined_signatures = set(signature_hints)
            combined_signatures.update(seed.signatures)
            combined_signatures.update(self._get_phonetic_fingerprint(seed_word))

            if seed.feature_profile:
                stress_hint = seed.feature_profile.get("stress_alignment")
                if isinstance(stress_hint, (int, float)):
                    combined_signatures.add(f"stress::{round(float(stress_hint), 2)}")
                device_hint = seed.feature_profile.get("bradley_device")
                if device_hint:
                    combined_signatures.add(f"device::{str(device_hint).lower()}")

            candidate_dicts: List[Dict[str, Any]] = []

            candidate_dicts.extend(
                self._query_seed_neighbors(cursor, normalized_seed, per_seed_limit * 2)
            )

            for suffix in self._extract_suffixes(seed_word):
                candidate_dicts.extend(
                    self._query_suffix_matches(cursor, suffix, per_seed_limit)
                )

            if cmu_fn is not None:
                try:
                    raw_candidates = cmu_fn(
                        seed_word,
                        limit=per_seed_limit * 2,
                        analyzer=analyzer,
                    )
                except Exception:
                    raw_candidates = []

                candidate_dicts.extend(
                    self._normalize_module1_candidates(raw_candidates)
                )

            local_seen: Set[str] = set()

            for candidate in candidate_dicts:
                if len(results) >= limit:
                    break

                target = str(candidate.get("candidate") or "").strip()
                if not target:
                    continue

                normalized_target = target.lower()
                if (
                    normalized_target in seen_targets
                    or normalized_target in local_seen
                    or normalized_target == normalized_seed
                    or normalized_target == source_word
                ):
                    continue

                fingerprint = self._get_phonetic_fingerprint(target)
                if combined_signatures and fingerprint and not (
                    fingerprint & combined_signatures
                ):
                    continue

                base_rarity = self._calculate_word_rarity(normalized_target)
                base_rarity = max(
                    base_rarity,
                    self._safe_float(candidate.get("rarity"), default=0.0),
                )

                seed_boost = min(1.5, seed.rarity * 0.5)
                complexity = self._analyze_phonological_complexity(source_word, normalized_target)
                syllable_complexity = self._calculate_syllable_complexity(normalized_target)

                final_score = min(
                    5.0,
                    base_rarity + seed_boost + complexity * 0.6 + syllable_complexity * 0.3,
                )

                stress_hint = seed.feature_profile.get("stress_alignment") if seed.feature_profile else None
                if isinstance(stress_hint, (int, float)):
                    final_score += min(0.4, max(0.0, float(stress_hint)) * 0.3)

                if final_score < 1.6:
                    continue

                weakness = "rare_word_combinations"
                if syllable_complexity >= 2.5:
                    weakness = "multi_syllable_complexity"
                elif complexity >= 2.0:
                    weakness = "phonological_processing"

                confidence = self._safe_float(
                    candidate.get("confidence"),
                    default=self._safe_float(candidate.get("combined"), default=0.7),
                )
                similarity_hint = self._safe_float(
                    candidate.get("phonetic_similarity"),
                    default=self._safe_float(candidate.get("similarity"), default=0.0),
                )
                if similarity_hint:
                    confidence = max(confidence, similarity_hint)
                if confidence <= 0:
                    confidence = 0.65 + min(0.25, complexity * 0.1)

                cultural_depth = (
                    candidate.get("context")
                    or candidate.get("source")
                    or f"Seed cascade via {seed_word}"
                )

                pattern = AntiLLMPattern(
                    source_word=source_word,
                    target_word=target,
                    rarity_score=final_score,
                    cultural_depth=str(cultural_depth),
                    llm_weakness_type=weakness,
                    confidence=confidence,
                )

                self._attach_profile(pattern)

                results.append(pattern)

                self.anti_llm_stats['seed_expansions'] += 1
                seen_targets.add(normalized_target)
                local_seen.add(normalized_target)

        return results[:limit]

    def _query_seed_neighbors(self, cursor, seed_word: str, limit: int) -> List[Dict[str, Any]]:
        if not seed_word:
            return []

        results: List[Dict[str, Any]] = []
        seen: Set[str] = set()

        query_source = """
            SELECT target_word, artist, song_title, confidence_score, phonetic_similarity, cultural_significance
            FROM song_rhyme_patterns
            WHERE LOWER(source_word) = ?
              AND target_word IS NOT NULL
              AND LOWER(target_word) != ?
            ORDER BY confidence_score DESC
            LIMIT ?
        """

        cursor.execute(query_source, (seed_word, seed_word, max(limit, 1)))
        for row in cursor.fetchall():
            candidate = str(row[0]).strip()
            lowered = candidate.lower()
            if not candidate or lowered in seen:
                continue

            seen.add(lowered)
            results.append(
                {
                    "candidate": candidate,
                    "confidence": row[3],
                    "phonetic_similarity": row[4],
                    "cultural_sig": row[5],
                    "context": f"{row[1]} - {row[2]}",
                }
            )

            if len(results) >= limit:
                break

        if len(results) < limit:
            query_target = """
                SELECT source_word, artist, song_title, confidence_score, phonetic_similarity, cultural_significance
                FROM song_rhyme_patterns
                WHERE LOWER(target_word) = ?
                  AND source_word IS NOT NULL
                  AND LOWER(source_word) != ?
                ORDER BY confidence_score DESC
                LIMIT ?
            """

            cursor.execute(query_target, (seed_word, seed_word, max(limit, 1)))
            for row in cursor.fetchall():
                candidate = str(row[0]).strip()
                lowered = candidate.lower()
                if not candidate or lowered in seen:
                    continue

                seen.add(lowered)
                results.append(
                    {
                        "candidate": candidate,
                        "confidence": row[3],
                        "phonetic_similarity": row[4],
                        "cultural_sig": row[5],
                        "context": f"{row[1]} - {row[2]}",
                    }
                )

                if len(results) >= limit:
                    break

        return results

    def _query_suffix_matches(self, cursor, suffix: str, limit: int) -> List[Dict[str, Any]]:
        if not suffix:
            return []

        results: List[Dict[str, Any]] = []
        seen: Set[str] = set()
        like_pattern = f"%{suffix}"

        for column in ("target_word", "source_word"):
            query = f"""
                SELECT {column}, artist, song_title, confidence_score, phonetic_similarity, cultural_significance
                FROM song_rhyme_patterns
                WHERE LOWER({column}) LIKE ?
                ORDER BY confidence_score DESC
                LIMIT ?
            """

            cursor.execute(query, (like_pattern, max(limit, 1)))
            for row in cursor.fetchall():
                candidate = str(row[0]).strip()
                lowered = candidate.lower()
                if (
                    not candidate
                    or not lowered.endswith(suffix)
                    or lowered in seen
                ):
                    continue

                seen.add(lowered)
                results.append(
                    {
                        "candidate": candidate,
                        "confidence": row[3],
                        "phonetic_similarity": row[4],
                        "cultural_sig": row[5],
                        "context": f"{row[1]} - {row[2]}",
                    }
                )

                if len(results) >= limit:
                    break

            if len(results) >= limit:
                break

        return results

    def _extract_suffixes(self, word: str) -> Set[str]:
        cleaned = re.sub(r"[^a-z]", "", (word or "").lower())
        suffixes: Set[str] = set()

        for length in (4, 3, 2):
            if len(cleaned) >= length:
                suffixes.add(cleaned[-length:])

        return suffixes

    def _get_phonetic_fingerprint(self, word: str) -> Set[str]:
        cleaned = re.sub(r"[^a-z]", "", (word or "").lower())
        if not cleaned:
            return set()

        vowels = re.findall(r"[aeiou]+", cleaned)
        fingerprint: Set[str] = set()

        if len(cleaned) >= 3:
            fingerprint.add(f"end:{cleaned[-3:]}")
        if len(cleaned) >= 4:
            fingerprint.add(f"tail:{cleaned[-4:]}")
        if vowels:
            fingerprint.add(f"v:{vowels[-1]}")
        if len(vowels) >= 2:
            fingerprint.add(f"vv:{vowels[-2]}>{vowels[-1]}")

        return fingerprint

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
