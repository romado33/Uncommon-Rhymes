#!/usr/bin/env python3
"""
RhymeRarity Hugging Face Spaces App
Main entry point for the deployed application
References: patterns.db, module1_enhanced_core_phonetic.py, module2_enhanced_anti_llm.py, module3_enhanced_cultural_database.py
"""

import gradio as gr
import sqlite3
import pandas as pd
import os
import sys
from typing import List, Dict, Tuple, Optional, Set
import json
import random
import types
import re

# Import our custom modules
try:
    from module1_enhanced_core_phonetic import (
        CMUDictLoader,
        EnhancedPhoneticAnalyzer,
        get_cmu_rhymes,
    )
    from module2_enhanced_anti_llm import AntiLLMRhymeEngine
    from module3_enhanced_cultural_database import CulturalIntelligenceEngine
    print("✅ All RhymeRarity modules loaded successfully")
except ImportError as e:
    print(f"⚠️ Module import warning: {e}")
    # Create fallback classes for demo
    class CMUDictLoader:
        def __init__(self, *_args, **_kwargs):
            pass

        def get_rhyming_words(self, *_args, **_kwargs):
            return []

    class EnhancedPhoneticAnalyzer:
        def __init__(self, *_, **__):
            self.cmu_loader = CMUDictLoader()
            self.rarity_map = types.SimpleNamespace(get_rarity=lambda *_: 0.5)

        def get_phonetic_similarity(self, *_args, **_kwargs):
            return 0.85

        def combine_similarity_and_rarity(self, similarity, rarity):
            return 0.5 * similarity + 0.5 * rarity

        def update_rarity_from_database(self, *_args, **_kwargs):
            return False

    def get_cmu_rhymes(word, limit=20, analyzer=None, cmu_loader=None):
        return []
    

    class AntiLLMRhymeEngine:
        def __init__(self, *_args, **_kwargs):
            pass

        def generate_anti_llm_patterns(self, word):
            return []

    class CulturalIntelligenceEngine:
        def __init__(self, *_, **kwargs):
            self.phonetic_analyzer = kwargs.get("phonetic_analyzer")

        def set_phonetic_analyzer(self, analyzer):
            self.phonetic_analyzer = analyzer

        def derive_rhyme_signatures(self, word):
            cleaned = re.sub(r"[^a-z]", "", str(word or "").lower())
            if not cleaned:
                return set()
            vowels = re.findall(r"[aeiou]+", cleaned)
            last_vowel = vowels[-1] if vowels else ""
            ending = cleaned[-3:] if len(cleaned) >= 3 else cleaned
            components = []
            if last_vowel:
                components.append(f"v:{last_vowel}")
            components.append(f"e:{ending}")
            return {"|".join(components)}

        def evaluate_rhyme_alignment(self, source_word, target_word, threshold=None, rhyme_signatures=None):
            analyzer = getattr(self, "phonetic_analyzer", None)
            similarity = None
            if analyzer and source_word and target_word:
                try:
                    similarity = float(analyzer.get_phonetic_similarity(source_word, target_word))
                except Exception:
                    similarity = None
            signature_set = self.derive_rhyme_signatures(target_word)
            rhyme_set = {sig for sig in (rhyme_signatures or set()) if sig}
            matches = sorted(signature_set.intersection(rhyme_set)) if rhyme_set else []
            if rhyme_set and signature_set and not matches:
                return None
            if threshold is not None and similarity is not None:
                if similarity < max(0.0, float(threshold) - 0.02):
                    return None
            return {
                'similarity': similarity,
                'rarity': None,
                'combined': similarity,
                'rhyme_type': None,
                'signature_matches': matches,
                'target_signatures': sorted(signature_set),
                'features': {},
            }

        def get_cultural_context(self, pattern):
            return {"significance": "mainstream"}

        def get_cultural_rarity_score(self, context):
            return 1.0

class RhymeRarityApp:
    """Production Hugging Face app for RhymeRarity rhyme search"""
    
    def __init__(self, db_path: str = "patterns.db"):
        self.db_path = db_path
        self.cmu_loader = CMUDictLoader()
        self.phonetic_analyzer = EnhancedPhoneticAnalyzer(cmu_loader=self.cmu_loader)
        self.anti_llm_engine = AntiLLMRhymeEngine(db_path=self.db_path)
        self.cultural_engine = CulturalIntelligenceEngine(
            db_path=self.db_path,
            phonetic_analyzer=self.phonetic_analyzer,
        )
        setter = getattr(self.cultural_engine, "set_phonetic_analyzer", None)
        if callable(setter):
            setter(self.phonetic_analyzer)

        # Initialize database
        self.check_database()
        self._refresh_rarity_map()

        print(f"🎵 RhymeRarity App initialized with database: {db_path}")

    def _refresh_rarity_map(self) -> None:
        analyzer = getattr(self, "phonetic_analyzer", None)
        if not analyzer:
            return
        updater = getattr(analyzer, "update_rarity_from_database", None)
        if callable(updater):
            try:
                updater(self.db_path)
            except Exception:
                pass

    def check_database(self):
        """Check if patterns.db exists and is accessible"""
        if not os.path.exists(self.db_path):
            print(f"⚠️ Database {self.db_path} not found - creating demo database")
            self.create_demo_database()
        else:
            # Verify database structure
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM song_rhyme_patterns LIMIT 1")
                count = cursor.fetchone()[0]
                conn.close()
                print(f"✅ Database verified: {count:,} patterns available")
                self._refresh_rarity_map()
            except Exception as e:
                print(f"⚠️ Database verification failed: {e}")
                self.create_demo_database()
    
    def create_demo_database(self):
        """Create a demo database for Hugging Face Spaces if patterns.db is missing"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS song_rhyme_patterns (
                id INTEGER PRIMARY KEY,
                pattern TEXT,
                source_word TEXT,
                target_word TEXT,
                artist TEXT,
                song_title TEXT,
                genre TEXT,
                line_distance INTEGER,
                confidence_score REAL,
                phonetic_similarity REAL,
                cultural_significance TEXT,
                source_context TEXT,
                target_context TEXT
            )
        ''')
        
        # Add sample rap rhyme data for demonstration
        sample_data = [
            ("love / above", "love", "above", "Drake", "Headlines", "hip-hop", 1, 0.95, 0.98, "mainstream", "All about that love", "Looking from above"),
            ("mind / find", "mind", "find", "Eminem", "Lose Yourself", "hip-hop", 1, 0.92, 0.96, "cultural_icon", "State of mind", "What you gonna find"),
            ("night / light", "night", "light", "Kendrick Lamar", "ADHD", "hip-hop", 2, 0.89, 0.94, "artistic", "In the night", "See the light"),
            ("flow / go", "flow", "go", "Jay-Z", "Izzo", "hip-hop", 1, 0.87, 0.92, "classic", "Feel the flow", "Watch me go"),
            ("time / rhyme", "time", "rhyme", "Nas", "NY State of Mind", "hip-hop", 1, 0.91, 0.95, "legendary", "In due time", "Perfect rhyme"),
            ("money / honey", "money", "honey", "Biggie", "Juicy", "hip-hop", 1, 0.88, 0.91, "classic", "Stack that money", "Sweet like honey"),
            ("street / beat", "street", "beat", "50 Cent", "In Da Club", "hip-hop", 1, 0.93, 0.97, "mainstream", "From the street", "Feel the beat"),
            ("pain / rain", "pain", "rain", "Tupac", "Dear Mama", "hip-hop", 1, 0.90, 0.93, "emotional", "Through the pain", "Like the rain"),
            ("game / fame", "game", "fame", "The Game", "Dreams", "hip-hop", 1, 0.86, 0.89, "ambitious", "In this game", "Chasing fame"),
            ("real / feel", "real", "feel", "J. Cole", "Middle Child", "hip-hop", 1, 0.84, 0.87, "authentic", "Keep it real", "How you feel"),
            ("way / day", "way", "day", "Kanye West", "Through The Wire", "hip-hop", 1, 0.85, 0.88, "innovative", "Show the way", "Brand new day"),
            ("life / knife", "life", "knife", "Eminem", "Stan", "hip-hop", 1, 0.89, 0.92, "dark", "Take my life", "Sharp as knife"),
            ("back / track", "back", "track", "LL Cool J", "Mama Said Knock You Out", "hip-hop", 1, 0.91, 0.94, "classic", "Got your back", "On the right track"),
            ("soul / goal", "soul", "goal", "Lauryn Hill", "Doo Wop", "hip-hop", 1, 0.87, 0.90, "conscious", "Search my soul", "Reach that goal"),
            ("word / heard", "word", "heard", "Rakim", "Paid In Full", "hip-hop", 1, 0.88, 0.91, "foundational", "Speak the word", "What you heard")
        ]
        
        cursor.executemany(
            "INSERT INTO song_rhyme_patterns (pattern, source_word, target_word, artist, song_title, genre, line_distance, confidence_score, phonetic_similarity, cultural_significance, source_context, target_context) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            sample_data
        )
        
        conn.commit()
        conn.close()

        print(f"✅ Demo database created with {len(sample_data)} sample patterns")
        self._refresh_rarity_map()
    

    def search_rhymes(
            self,
            source_word: str,
            limit: int = 20,
            min_confidence: float = 0.7,
            cultural_significance: Optional[List[str]] = None,
            genres: Optional[List[str]] = None,
            result_sources: Optional[List[str]] = None,
            max_line_distance: Optional[int] = None,
            rhyme_types: Optional[List[str]] = None,
            min_syllable_similarity: Optional[float] = None,
            min_rhythm_strength: Optional[float] = None,
        ) -> List[Dict]:
            """Search for rhymes in the patterns.db database"""
            if not source_word or not source_word.strip():
                return []

            source_word = source_word.lower().strip()

            def _normalize_to_list(value: Optional[List[str] | str]) -> List[str]:
                if value is None:
                    return []
                if isinstance(value, (list, tuple, set)):
                    return [str(v) for v in value if v is not None and str(v).strip()]
                if isinstance(value, str):
                    return [value] if value.strip() else []
                return [str(value)]

            def _normalize_source_name(name: str) -> str:
                return name.strip().lower().replace("_", "-")

            cultural_filters = {_normalize_source_name(sig) for sig in _normalize_to_list(cultural_significance)}
            genre_filters = {_normalize_source_name(genre) for genre in _normalize_to_list(genres)}
            rhyme_type_filters = {
                _normalize_source_name(rhyme)
                for rhyme in _normalize_to_list(rhyme_types)
            }

            def _normalize_threshold(value: Optional[float]) -> Optional[float]:
                if value is None:
                    return None
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    return None
                if numeric <= 0:
                    return None
                return numeric

            min_syllable_similarity = _normalize_threshold(min_syllable_similarity)
            min_rhythm_strength = _normalize_threshold(min_rhythm_strength)

            syllable_threshold_active = min_syllable_similarity is not None
            rhythm_threshold_active = min_rhythm_strength is not None
            syllable_threshold_value = float(min_syllable_similarity) if syllable_threshold_active else 0.0
            rhythm_threshold_value = float(min_rhythm_strength) if rhythm_threshold_active else 0.0

            selected_sources = {
                _normalize_source_name(source) for source in _normalize_to_list(result_sources)
            }
            if not selected_sources:
                selected_sources = {"phonetic", "cultural", "anti-llm"}

            include_phonetic = "phonetic" in selected_sources
            include_cultural = "cultural" in selected_sources
            include_anti_llm = "anti-llm" in selected_sources

            filters_active = bool(
                cultural_filters
                or genre_filters
                or (max_line_distance is not None)
                or rhyme_type_filters
                or syllable_threshold_active
                or rhythm_threshold_active
            )

            def _safe_float(value: Optional[float], default: float = 0.0) -> float:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return default

            def _extract_phonetic_feature(entry: Dict, key: str) -> Optional[float]:
                features = entry.get("phonetic_features")
                if isinstance(features, dict):
                    value = features.get(key)
                    if value is not None:
                        try:
                            return float(value)
                        except (TypeError, ValueError):
                            pass
                value = entry.get(key)
                if value is not None:
                    try:
                        return float(value)
                    except (TypeError, ValueError):
                        return None
                return None

            def _evaluate_rhythm_metrics(entry: Dict) -> Tuple[Optional[float], Optional[float]]:
                syllable_value = _extract_phonetic_feature(entry, "syllable_similarity")
                components = [
                    _extract_phonetic_feature(entry, "ending_similarity"),
                    _extract_phonetic_feature(entry, "vowel_similarity"),
                    _extract_phonetic_feature(entry, "consonant_similarity"),
                ]
                available = [value for value in components if value is not None]
                rhythm_strength_value: Optional[float]
                if available:
                    rhythm_strength_value = sum(available) / len(available)
                else:
                    rhythm_strength_value = None
                return syllable_value, rhythm_strength_value

            if limit <= 0:
                return []

            analyzer = getattr(self, "phonetic_analyzer", None)
            cultural_engine = getattr(self, "cultural_engine", None)
            cmu_loader = None
            if analyzer is not None:
                cmu_loader = getattr(analyzer, "cmu_loader", None)
            if cmu_loader is None:
                cmu_loader = getattr(self, "cmu_loader", None)

            def _fallback_signature(word: str) -> Set[str]:
                cleaned = re.sub(r"[^a-z]", "", (word or "").lower())
                if not cleaned:
                    return set()
                vowels = re.findall(r"[aeiou]+", cleaned)
                last_vowel = vowels[-1] if vowels else ""
                ending = cleaned[-3:] if len(cleaned) >= 3 else cleaned
                signature_bits: List[str] = []
                if last_vowel:
                    signature_bits.append(f"v:{last_vowel}")
                signature_bits.append(f"e:{ending}")
                return {"|".join(signature_bits)}

            source_signature_set: Set[str] = set()
            if cultural_engine and hasattr(cultural_engine, "derive_rhyme_signatures"):
                try:
                    derived_signatures = cultural_engine.derive_rhyme_signatures(source_word)
                    source_signature_set = {sig for sig in derived_signatures if sig}
                except Exception:
                    source_signature_set = set()
            if not source_signature_set and cmu_loader is not None:
                try:
                    source_signature_set = {
                        sig for sig in cmu_loader.get_rhyme_parts(source_word) if sig
                    }
                except Exception:
                    source_signature_set = set()
            if not source_signature_set:
                source_signature_set = _fallback_signature(source_word)

            source_signature_list = sorted(source_signature_set)

            reference_limit = max(limit * 2, 10)
            try:
                cmu_candidates = get_cmu_rhymes(
                    source_word,
                    limit=reference_limit,
                    analyzer=analyzer,
                    cmu_loader=cmu_loader,
                )
            except Exception:
                cmu_candidates = []

            reference_similarity = 0.0
            for candidate in cmu_candidates:
                if isinstance(candidate, dict):
                    try:
                        score = float(candidate.get("similarity", candidate.get("score", 0.0)))
                    except (TypeError, ValueError):
                        continue
                else:
                    try:
                        score = float(candidate[1]) if len(candidate) > 1 else 0.0
                    except (TypeError, ValueError, IndexError):
                        score = 0.0
                if score > reference_similarity:
                    reference_similarity = score

            db_candidate_words: Set[str] = set()
            if analyzer:
                try:
                    with sqlite3.connect(self.db_path) as conn:
                        cursor = conn.cursor()
                        cursor.execute(
                            "SELECT target_word FROM song_rhyme_patterns WHERE source_word = ? AND target_word IS NOT NULL LIMIT 50",
                            (source_word,),
                        )
                        db_candidate_words.update(
                            str(row[0]).strip().lower()
                            for row in cursor.fetchall()
                            if row and row[0]
                        )
                        cursor.execute(
                            "SELECT source_word FROM song_rhyme_patterns WHERE target_word = ? AND source_word IS NOT NULL LIMIT 50",
                            (source_word,),
                        )
                        db_candidate_words.update(
                            str(row[0]).strip().lower()
                            for row in cursor.fetchall()
                            if row and row[0]
                        )
                except sqlite3.Error:
                    db_candidate_words = set()

                for candidate_word in db_candidate_words:
                    if not candidate_word or candidate_word == source_word:
                        continue
                    try:
                        score = float(analyzer.get_phonetic_similarity(source_word, candidate_word))
                    except Exception:
                        continue
                    if score > reference_similarity:
                        reference_similarity = score

            phonetic_threshold = 0.7
            if reference_similarity > 0:
                phonetic_threshold = max(0.6, min(0.92, reference_similarity - 0.1))

            source_phonetic_profile = {
                "threshold": phonetic_threshold,
                "reference_similarity": reference_similarity,
                "signatures": source_signature_list,
            }

            phonetic_entries: List[Dict] = []
            if include_phonetic and not filters_active:
                phonetic_matches = cmu_candidates[: max(limit, 1)]
                rarity_source = analyzer if analyzer is not None else getattr(self, "phonetic_analyzer", None)
                for candidate in phonetic_matches:
                    if isinstance(candidate, dict):
                        target = candidate.get("word") or candidate.get("target")
                        similarity = float(candidate.get("similarity", candidate.get("score", 0.0)))
                        rarity_value = float(candidate.get("rarity", candidate.get("rarity_score", 0.0)))
                        combined = float(candidate.get("combined", candidate.get("combined_score", similarity)))
                    else:
                        try:
                            target = candidate[0]
                            similarity = float(candidate[1]) if len(candidate) > 1 else 0.0
                            if len(candidate) > 2:
                                rarity_value = float(candidate[2])
                            elif rarity_source and hasattr(rarity_source, "get_rarity_score"):
                                rarity_value = float(rarity_source.get_rarity_score(candidate[0]))
                            else:
                                rarity_value = 0.0
                            combined = float(candidate[3]) if len(candidate) > 3 else similarity
                        except Exception:
                            target = candidate if isinstance(candidate, str) else None
                            similarity = 0.0
                            rarity_value = 0.0
                            combined = 0.0

                    if not target:
                        continue

                    alignment = None
                    if cultural_engine and hasattr(cultural_engine, "evaluate_rhyme_alignment"):
                        try:
                            alignment = cultural_engine.evaluate_rhyme_alignment(
                                source_word,
                                target,
                                threshold=phonetic_threshold,
                                rhyme_signatures=source_signature_set,
                            )
                        except Exception:
                            alignment = None
                        if alignment is None:
                            continue
                    else:
                        alignment = {
                            "similarity": similarity,
                            "rarity": rarity_value,
                            "combined": combined,
                            "signature_matches": [],
                            "target_signatures": [],
                            "features": {},
                        }

                    entry = {
                        "source_word": source_word,
                        "target_word": target,
                        "artist": "CMU Pronouncing Dictionary",
                        "song": "Phonetic Match",
                        "pattern": f"{source_word} / {target}",
                        "distance": None,
                        "confidence": combined,
                        "combined_score": combined,
                        "phonetic_sim": similarity,
                        "rarity_score": rarity_value,
                        "cultural_sig": "phonetic",
                        "genre": None,
                        "source_context": "Phonetic match suggested by the CMU Pronouncing Dictionary.",
                        "target_context": "",
                        "result_source": "phonetic",
                        "source_rhyme_signatures": source_signature_list,
                        "source_phonetic_profile": source_phonetic_profile,
                        "phonetic_threshold": phonetic_threshold,
                    }

                    entry["phonetic_sim"] = alignment.get("similarity", entry["phonetic_sim"])
                    if alignment.get("rarity") is not None:
                        entry["rarity_score"] = alignment["rarity"]
                    if alignment.get("combined") is not None:
                        entry["combined_score"] = alignment["combined"]
                        try:
                            entry["confidence"] = max(
                                float(entry.get("confidence", 0.0)),
                                float(alignment["combined"]),
                            )
                        except (TypeError, ValueError):
                            entry["confidence"] = alignment["combined"]
                    if alignment.get("rhyme_type"):
                        entry["rhyme_type"] = alignment["rhyme_type"]
                    entry["matched_signatures"] = alignment.get("signature_matches", [])
                    entry["target_rhyme_signatures"] = alignment.get("target_signatures", [])
                    features = alignment.get("features")
                    if features:
                        entry["phonetic_features"] = features

                    phonetic_entries.append(entry)

                phonetic_entries.sort(
                    key=lambda entry: (
                        entry.get("combined_score", entry.get("confidence", 0.0)),
                        entry.get("phonetic_sim", 0.0),
                    ),
                    reverse=True,
                )

            fetch_limit = max(limit * 2, limit, 1)

            base_query = """
                SELECT DISTINCT
                    source_word,
                    target_word,
                    artist,
                    song_title,
                    pattern,
                    genre,
                    line_distance,
                    confidence_score,
                    phonetic_similarity,
                    cultural_significance,
                    source_context,
                    target_context
                FROM song_rhyme_patterns
                """

            cultural_entries: List[Dict] = []

            def _context_to_dict(context_obj):
                if context_obj is None:
                    return None
                if isinstance(context_obj, dict):
                    return dict(context_obj)
                if hasattr(context_obj, "__dict__"):
                    return dict(vars(context_obj))
                return {"value": context_obj}

            def _enrich_with_cultural_context(entry: Dict) -> Optional[Dict]:
                engine = getattr(self, "cultural_engine", None)
                if not engine:
                    return entry

                entry.setdefault("source_phonetic_profile", source_phonetic_profile)
                entry.setdefault("source_rhyme_signatures", source_signature_list)
                entry.setdefault("phonetic_threshold", phonetic_threshold)
                entry.setdefault("matched_signatures", [])
                entry.setdefault("target_rhyme_signatures", [])

                context_data = None
                rarity_value = None

                try:
                    get_context = getattr(engine, "get_cultural_context", None)
                    if callable(get_context):
                        pattern_payload = {
                            "artist": entry.get("artist"),
                            "song": entry.get("song"),
                            "source_word": entry.get("source_word", source_word),
                            "target_word": entry.get("target_word"),
                            "pattern": entry.get("pattern"),
                            "cultural_significance": entry.get("cultural_sig"),
                        }
                        context_data = get_context(pattern_payload)
                    elif hasattr(engine, "find_cultural_patterns"):
                        finder = getattr(engine, "find_cultural_patterns")
                        if callable(finder):
                            patterns = finder(entry.get("source_word", source_word), limit=1)
                            if patterns:
                                pattern_info = patterns[0]
                                context_data = pattern_info.get("cultural_context")
                                rarity_value = pattern_info.get("cultural_rarity")

                    context_dict = _context_to_dict(context_data)
                    if context_dict:
                        entry["cultural_context"] = context_dict

                        rarity_fn = getattr(engine, "get_cultural_rarity_score", None)
                        if callable(rarity_fn):
                            try:
                                rarity_value = rarity_fn(context_data)
                            except (TypeError, AttributeError):
                                rarity_value = rarity_fn(context_dict)

                    if rarity_value is not None:
                        entry["cultural_rarity"] = rarity_value

                except Exception:
                    pass

                try:
                    evaluate_alignment = getattr(engine, "evaluate_rhyme_alignment", None)
                    if callable(evaluate_alignment):
                        alignment = evaluate_alignment(
                            entry.get("source_word", source_word),
                            entry.get("target_word"),
                            threshold=phonetic_threshold,
                            rhyme_signatures=source_signature_set,
                        )
                        if alignment is None:
                            return None
                        sim_value = alignment.get("similarity")
                        if sim_value is not None:
                            entry["phonetic_sim"] = sim_value
                        combined_value = alignment.get("combined")
                        if combined_value is not None:
                            entry["combined_score"] = combined_value
                            try:
                                entry["confidence"] = max(
                                    float(entry.get("confidence", 0.0)),
                                    float(combined_value),
                                )
                            except (TypeError, ValueError):
                                entry["confidence"] = combined_value
                        rarity_metric = alignment.get("rarity")
                        if rarity_metric is not None:
                            entry["rarity_score"] = rarity_metric
                        rhyme_type = alignment.get("rhyme_type")
                        if rhyme_type:
                            entry["rhyme_type"] = rhyme_type
                        entry["matched_signatures"] = alignment.get(
                            "signature_matches", entry.get("matched_signatures", [])
                        )
                        entry["target_rhyme_signatures"] = alignment.get(
                            "target_signatures", entry.get("target_rhyme_signatures", [])
                        )
                        features = alignment.get("features")
                        if features:
                            entry["phonetic_features"] = features
                except Exception:
                    pass

                return entry

            def build_query(word_column: str) -> Tuple[str, List]:
                conditions = [f"{word_column} = ?", "confidence_score >= ?", "source_word != target_word"]
                params: List = [source_word, min_confidence]

                if phonetic_threshold is not None:
                    conditions.append("phonetic_similarity >= ?")
                    params.append(phonetic_threshold)

                if cultural_filters:
                    placeholders = ",".join(["?"] * len(cultural_filters))
                    conditions.append(f"LOWER(cultural_significance) IN ({placeholders})")
                    params.extend(sorted(cultural_filters))

                if genre_filters:
                    placeholders = ",".join(["?"] * len(genre_filters))
                    conditions.append(f"LOWER(genre) IN ({placeholders})")
                    params.extend(sorted(genre_filters))

                if max_line_distance is not None:
                    conditions.append("line_distance <= ?")
                    params.append(max_line_distance)

                query = base_query
                query += " WHERE " + " AND ".join(conditions)
                query += " ORDER BY confidence_score DESC, phonetic_similarity DESC LIMIT ?"
                params.append(fetch_limit)
                return query, params

            source_results: List[Tuple] = []
            target_results: List[Tuple] = []

            if include_cultural:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()

                    query, params = build_query("source_word")
                    cursor.execute(query, params)
                    source_results = cursor.fetchall()

                    query, params = build_query("target_word")
                    cursor.execute(query, params)
                    target_results = cursor.fetchall()

            def build_entry(row: Tuple, swap: bool = False) -> Optional[Dict]:
                entry = {
                    "source_word": row[0],
                    "target_word": row[1],
                    "artist": row[2],
                    "song": row[3],
                    "pattern": row[4],
                    "genre": row[5],
                    "distance": row[6],
                    "confidence": row[7],
                    "phonetic_sim": row[8],
                    "cultural_sig": row[9],
                    "source_context": row[10],
                    "target_context": row[11],
                    "result_source": "cultural",
                    "combined_score": row[7],
                    "source_phonetic_profile": source_phonetic_profile,
                    "source_rhyme_signatures": source_signature_list,
                    "phonetic_threshold": phonetic_threshold,
                }

                if swap:
                    entry = {
                        **entry,
                        "source_word": row[1],
                        "target_word": row[0],
                        "source_context": row[11],
                        "target_context": row[10],
                    }

                return _enrich_with_cultural_context(entry)

            for row in source_results:
                enriched_entry = build_entry(row)
                if enriched_entry:
                    cultural_entries.append(enriched_entry)

            for row in target_results:
                enriched_entry = build_entry(row, swap=True)
                if enriched_entry:
                    cultural_entries.append(enriched_entry)

            cultural_entries.sort(
                key=lambda r: (
                    -float(r.get("confidence", 0.0)),
                    -float(r.get("phonetic_sim", 0.0)),
                )
            )

            anti_llm_entries: List[Dict] = []
            if include_anti_llm and not filters_active:
                anti_patterns = self.anti_llm_engine.generate_anti_llm_patterns(
                    source_word,
                    limit=limit,
                )
                for pattern in anti_patterns:
                    anti_llm_entries.append({
                        "source_word": source_word,
                        "target_word": pattern.target_word,
                        "pattern": f"{source_word} / {pattern.target_word}",
                        "confidence": pattern.confidence,
                        "rarity_score": pattern.rarity_score,
                        "llm_weakness_type": pattern.llm_weakness_type,
                        "cultural_depth": pattern.cultural_depth,
                        "result_source": "anti_llm",
                        "source_rhyme_signatures": source_signature_list,
                        "source_phonetic_profile": source_phonetic_profile,
                        "phonetic_threshold": phonetic_threshold,
                    })

            combined_results = phonetic_entries + cultural_entries + anti_llm_entries

            deduped: List[Dict] = []
            seen_pairs = set()
            for entry in combined_results:
                target_value = entry.get("target_word")
                if not target_value:
                    continue
                source_value = entry.get("source_word", source_word)
                pair = (
                    str(source_value).lower(),
                    str(target_value).lower(),
                )
                if pair in seen_pairs:
                    continue

                normalized_rhyme: Optional[str] = None
                entry_rhyme_value = entry.get("rhyme_type")
                if entry_rhyme_value:
                    normalized_rhyme = _normalize_source_name(str(entry_rhyme_value))
                elif rhyme_type_filters and analyzer is not None:
                    classifier = getattr(analyzer, "classify_rhyme_type", None)
                    similarity_getter = getattr(analyzer, "get_phonetic_similarity", None)
                    if callable(classifier):
                        try:
                            similarity_estimate = entry.get("phonetic_sim")
                            if similarity_estimate is None and callable(similarity_getter):
                                similarity_estimate = similarity_getter(str(source_value), str(target_value))
                            similarity_value = float(similarity_estimate) if similarity_estimate is not None else 0.0
                            derived_type = classifier(str(source_value), str(target_value), similarity_value)
                        except Exception:
                            derived_type = None
                        if derived_type:
                            entry["rhyme_type"] = derived_type
                            normalized_rhyme = _normalize_source_name(str(derived_type))

                if rhyme_type_filters and (not normalized_rhyme or normalized_rhyme not in rhyme_type_filters):
                    continue

                syllable_value, rhythm_strength_value = _evaluate_rhythm_metrics(entry)

                if syllable_threshold_active:
                    if syllable_value is None or syllable_value < syllable_threshold_value:
                        continue
                if rhythm_threshold_active:
                    if rhythm_strength_value is None or rhythm_strength_value < rhythm_threshold_value:
                        continue

                if syllable_value is not None and "syllable_similarity" not in entry:
                    entry["syllable_similarity"] = syllable_value
                if rhythm_strength_value is not None and "rhythm_strength" not in entry:
                    entry["rhythm_strength"] = rhythm_strength_value

                seen_pairs.add(pair)
                deduped.append(entry)

            filtered_entries: List[Dict] = []
            if filters_active:
                for entry in deduped:
                    if cultural_filters:
                        sig = entry.get("cultural_sig")
                        if sig is None or _normalize_source_name(str(sig)) not in cultural_filters:
                            continue
                    if genre_filters:
                        genre_value = entry.get("genre")
                        if genre_value is None or _normalize_source_name(str(genre_value)) not in genre_filters:
                            continue
                    if max_line_distance is not None:
                        distance_value = entry.get("distance")
                        if distance_value is None or distance_value > max_line_distance:
                            continue
                    filtered_entries.append(entry)
            else:
                filtered_entries = deduped

            if syllable_threshold_active or rhythm_threshold_active or rhyme_type_filters:
                filtered_entries.sort(
                    key=lambda entry: (
                        _safe_float(entry.get("combined_score", entry.get("confidence"))),
                        _safe_float(entry.get("rhythm_strength")),
                        _safe_float(entry.get("phonetic_sim")),
                    ),
                    reverse=True,
                )

            for entry in filtered_entries:
                entry.pop("source_word", None)

            return filtered_entries[:limit]
    def format_rhyme_results(self, source_word: str, rhymes: List[Dict]) -> str:
        """Format rhyme search results for display"""
        if not rhymes:
            return f"❌ No rhymes found for '{source_word}'\n\nTry words like: love, mind, flow, time, money"

        result = f"🎯 **TARGET RHYMES for '{source_word.upper()}':**\n"
        result += "=" * 50 + "\n\n"

        def _format_rhythm_details(rhyme_entry: Dict) -> str:
            lines: List[str] = []

            rhyme_type_value = rhyme_entry.get('rhyme_type')
            if rhyme_type_value:
                rhyme_label = str(rhyme_type_value).replace('_', ' ').title()
                lines.append(f"   🎯 Rhyme Type: {rhyme_label}")

            rhythm_strength_value = rhyme_entry.get('rhythm_strength')
            try:
                if rhythm_strength_value is not None:
                    lines.append(f"   🥁 Rhythm Strength: {float(rhythm_strength_value):.2f}")
            except (TypeError, ValueError):
                pass

            syllable_alignment_value = rhyme_entry.get('syllable_similarity')
            try:
                if syllable_alignment_value is not None:
                    lines.append(f"   🔄 Syllable Alignment: {float(syllable_alignment_value):.2f}")
            except (TypeError, ValueError):
                pass

            if not lines:
                return ""

            return "\n".join(lines) + "\n"

        for i, rhyme in enumerate(rhymes[:15], 1):
            result += f"**{i:2d}. {rhyme['target_word'].upper()}**\n"
            source_type = rhyme.get('result_source')
            if source_type == 'phonetic':
                origin = rhyme.get('artist', 'CMU Pronouncing Dictionary')
                result += f"   🏷️ Origin: 📚 {origin}\n"
                result += f"   📝 Pattern: {rhyme['pattern']}\n"
                result += f"   📊 Phonetic Score: {rhyme['phonetic_sim']:.2f}\n"
                rarity_value = rhyme.get('rarity_score')
                if rarity_value is not None:
                    result += f"   🌟 Rarity Score: {rarity_value:.2f}\n"
                combined_value = rhyme.get('combined_score')
                if combined_value is not None:
                    result += f"   🎯 Combined Score: {combined_value:.2f}\n"
                note = rhyme.get('source_context') or 'Pronunciation-based suggestion.'
                result += f"   🗒️ Note: {note}\n\n"
                result += _format_rhythm_details(rhyme)
            elif source_type in {"anti_llm", "anti-llm"}:
                pattern_text = rhyme.get('pattern') or f"{source_word} / {rhyme['target_word']}"
                result += "   🤖 Source: Anti-LLM Engine\n"
                result += f"   📝 Pattern: {pattern_text}\n"

                confidence = rhyme.get('confidence')
                if confidence is not None:
                    result += f"   📈 Confidence: {confidence:.2f}\n"

                rarity = rhyme.get('rarity_score')
                if rarity is not None:
                    result += f"   🌟 Rarity Score: {rarity:.2f}\n"

                weakness = rhyme.get('llm_weakness_type')
                if weakness:
                    weakness_label = str(weakness).replace('_', ' ').title()
                    result += f"   🧩 LLM Weakness: {weakness_label}\n"

                cultural_depth = rhyme.get('cultural_depth')
                if cultural_depth:
                    result += f"   🌌 Cultural Depth: {cultural_depth}\n"

                context_note = rhyme.get('source_context') or 'Designed to exploit LLM weaknesses.'
                if context_note:
                    result += f"   🧠 Insight: {context_note}\n"

                result += "\n"
                result += _format_rhythm_details(rhyme)
            else:
                result += f"   🎤 Source: {rhyme['artist']} - {rhyme['song']}\n"
                result += f"   📝 Pattern: {rhyme['pattern']}\n"
                result += f"   📊 Confidence: {rhyme['confidence']:.2f} | Phonetic: {rhyme['phonetic_sim']:.2f}\n"

                context_info = rhyme.get('cultural_context')
                if context_info and hasattr(context_info, "__dict__"):
                    context_info = dict(vars(context_info))

                def _humanize(value: Optional[str]) -> str:
                    if value is None:
                        return ""
                    if isinstance(value, str):
                        return value.replace('_', ' ').title()
                    return str(value)

                cultural_highlights: List[str] = []
                if isinstance(context_info, dict):
                    era_value = context_info.get('era')
                    if era_value:
                        cultural_highlights.append(f"Era: {_humanize(era_value)}")

                    region_value = context_info.get('regional_origin')
                    if region_value:
                        cultural_highlights.append(f"Region: {_humanize(region_value)}")

                    significance_value = context_info.get('cultural_significance') or rhyme.get('cultural_sig')
                    if significance_value:
                        cultural_highlights.append(f"Significance: {_humanize(significance_value)}")

                rarity_score = rhyme.get('cultural_rarity')
                if rarity_score is not None:
                    cultural_highlights.append(f"Rarity Score: {rarity_score:.2f}")

                if cultural_highlights:
                    result += f"   🌍 Cultural Context: {' | '.join(cultural_highlights)}\n"

                if isinstance(context_info, dict):
                    styles = context_info.get('style_characteristics')
                    if styles:
                        formatted_styles = ", ".join(_humanize(style) for style in styles if style)
                        if formatted_styles:
                            result += f"   🎨 Styles: {formatted_styles}\n"

                result += f"   🎵 Context: \"{rhyme['source_context']}\" → \"{rhyme['target_context']}\"\n\n"
                result += _format_rhythm_details(rhyme)

        return result
    
    def create_gradio_interface(self):
        """Create the Gradio interface for Hugging Face Spaces"""

        def search_interface(
            word,
            max_results,
            min_conf,
            cultural_filter,
            genre_filter,
            source_filter,
            max_line_distance_choice,
            rhyme_type_filter,
            min_syllable_alignment,
            min_rhythm_strength,
        ):
            """Interface function for Gradio"""
            if not word:
                return "Please enter a word to find rhymes for."

            def ensure_list(value):
                if value is None:
                    return []
                if isinstance(value, (list, tuple, set)):
                    return [v for v in value if v not in (None, "", [])]
                if isinstance(value, str):
                    return [value] if value else []
                return [value]

            def ensure_threshold(value):
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    return None
                if numeric <= 0:
                    return None
                return numeric

            max_distance: Optional[int]
            if max_line_distance_choice in (None, "", "Any"):
                max_distance = None
            else:
                try:
                    max_distance = int(max_line_distance_choice)
                except (TypeError, ValueError):
                    max_distance = None

            min_syllable_value = ensure_threshold(min_syllable_alignment)
            min_rhythm_value = ensure_threshold(min_rhythm_strength)

            rhymes = self.search_rhymes(
                word,
                limit=max_results,
                min_confidence=min_conf,
                cultural_significance=ensure_list(cultural_filter),
                genres=ensure_list(genre_filter),
                result_sources=ensure_list(source_filter),
                max_line_distance=max_distance,
                rhyme_types=ensure_list(rhyme_type_filter),
                min_syllable_similarity=min_syllable_value,
                min_rhythm_strength=min_rhythm_value,
            )
            return self.format_rhyme_results(word, rhymes)

        cultural_options = sorted(
            getattr(getattr(self, "cultural_engine", None), "cultural_categories", {}).keys()
        )

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT DISTINCT genre FROM song_rhyme_patterns WHERE genre IS NOT NULL ORDER BY genre"
                )
                genre_options = [row[0] for row in cursor.fetchall() if row[0]]
        except sqlite3.Error:
            genre_options = []

        default_sources = ["phonetic", "cultural"]
        rhyme_type_options = ["perfect", "near", "slant", "eye", "weak"]
        rhyme_type_defaults = list(rhyme_type_options)

        # Create Gradio interface
        with gr.Blocks(title="RhymeRarity - Advanced Rhyme Generator", theme=gr.themes.Soft()) as interface:
            gr.Markdown("""
            # 🎵 RhymeRarity - Advanced Rhyme Generator

            **Find perfect rhymes from authentic hip-hop lyrics and cultural patterns**

            This system uses real rap lyrics from 200+ artists to find rhymes that LLMs can't generate,
            with full cultural attribution and context.
            """)

            with gr.Row():
                with gr.Column(scale=2):
                    word_input = gr.Textbox(
                        label="Word to Find Rhymes For",
                        placeholder="Enter a word (e.g., love, mind, flow, money)",
                        lines=1
                    )

                    max_results = gr.Slider(
                        minimum=5,
                        maximum=50,
                        value=15,
                        step=1,
                        label="Max Results"
                    )

                    min_confidence = gr.Slider(
                        minimum=0.5,
                        maximum=1.0,
                        value=0.7,
                        step=0.05,
                        label="Min Confidence"
                    )

                with gr.Column(scale=1):
                    cultural_dropdown = gr.Dropdown(
                        choices=cultural_options,
                        multiselect=True,
                        label="Cultural Significance",
                        info="Filter by cultural importance",
                        value=[],
                    )

                    genre_dropdown = gr.Dropdown(
                        choices=genre_options,
                        multiselect=True,
                        label="Genre",
                        info="Limit to specific genres",
                        value=[],
                    )

                    result_source_group = gr.CheckboxGroup(
                        choices=["phonetic", "cultural", "anti-llm"],
                        value=default_sources,
                        label="Result Sources",
                    )

                    max_line_distance_dropdown = gr.Dropdown(
                        choices=["Any", "1", "2", "3", "4", "5"],
                        value="Any",
                        label="Max Line Distance",
                    )

                with gr.Column(scale=1):
                    rhyme_type_group = gr.CheckboxGroup(
                        choices=rhyme_type_options,
                        value=rhyme_type_defaults,
                        label="Rhyme Types",
                        info="Choose the rhyme categories to include in results",
                    )

                    min_syllable_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.0,
                        step=0.05,
                        label="Min Syllable Alignment",
                        info="Set a minimum syllable alignment to focus on tighter rhythms",
                    )

                    min_rhythm_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.0,
                        step=0.05,
                        label="Min Rhythm Strength",
                        info="Require a minimum average of ending, vowel, and consonant alignment",
                    )

            search_btn = gr.Button("🔍 Find Rhymes", variant="primary", size="lg")

            output = gr.Textbox(
                label="Rhyme Results",
                lines=20,
                max_lines=30,
                show_copy_button=True
            )

            # Example inputs
            gr.Examples(
                examples=[
                    ["love", 15, 0.8, [], [], default_sources, "Any", list(rhyme_type_defaults), 0.0, 0.0],
                    ["mind", 15, 0.8, [], [], default_sources, "Any", list(rhyme_type_defaults), 0.0, 0.0],
                    ["flow", 15, 0.8, [], [], default_sources, "Any", list(rhyme_type_defaults), 0.0, 0.0],
                    ["money", 15, 0.8, [], [], default_sources, "Any", list(rhyme_type_defaults), 0.0, 0.0],
                    ["time", 15, 0.8, [], [], default_sources, "Any", ["perfect", "near"], 0.6, 0.55]
                ],
                inputs=[
                    word_input,
                    max_results,
                    min_confidence,
                    cultural_dropdown,
                    genre_dropdown,
                    result_source_group,
                    max_line_distance_dropdown,
                    rhyme_type_group,
                    min_syllable_slider,
                    min_rhythm_slider,
                ],
                label="Try these examples"
            )

            search_btn.click(
                fn=search_interface,
                inputs=[
                    word_input,
                    max_results,
                    min_confidence,
                    cultural_dropdown,
                    genre_dropdown,
                    result_source_group,
                    max_line_distance_dropdown,
                    rhyme_type_group,
                    min_syllable_slider,
                    min_rhythm_slider,
                ],
                outputs=output
            )

            gr.Markdown("""
            ---
            ### About RhymeRarity

            - **1.2M+ authentic rhyme patterns** from real hip-hop lyrics
            - **200+ artists** including Eminem, Kendrick Lamar, Jay-Z, Nas, and more
            - **Cultural intelligence** with full song and artist attribution
            - **Anti-LLM algorithms** that find rhymes large language models miss
            - **Research-backed** phonetic analysis for superior rhyme detection
            """)

        return interface

# Initialize the app
app = RhymeRarityApp()

# Create and launch the interface
if __name__ == "__main__":
    interface = app.create_gradio_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_api=True,
        share=True
    )
