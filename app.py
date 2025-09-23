#!/usr/bin/env python3
"""
RhymeRarity Hugging Face Spaces App
Main entry point for the deployed application
References: patterns.db, module1_enhanced_core_phonetic.py, module2_enhanced_anti_llm.py, module3_enhanced_cultural_database.py
"""

import gradio as gr
import sqlite3
import os
from typing import Any, Dict, List, Optional, Set, Tuple
import types
import re

from demo_data import DEMO_RHYME_PATTERNS, iter_demo_rhyme_rows

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

        def set_phonetic_analyzer(self, analyzer):
            self.phonetic_analyzer = analyzer

        def generate_anti_llm_patterns(
            self,
            word,
            limit=20,
            module1_seeds=None,
            seed_signatures=None,
            delivered_words=None,
        ):
            return []

    class CulturalIntelligenceEngine:
        def __init__(self, *_, **kwargs):
            self.phonetic_analyzer = kwargs.get("phonetic_analyzer")

        def set_phonetic_analyzer(self, analyzer):
            self.phonetic_analyzer = analyzer

        def set_prosody_analyzer(self, analyzer):
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

        def evaluate_rhyme_alignment(
            self,
            source_word,
            target_word,
            threshold=None,
            rhyme_signatures=None,
            source_context=None,
            target_context=None,
        ):
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
            feature_profile = {
                'bradley_device': 'assonance' if similarity and similarity >= 0.85 else 'slant rhyme',
                'syllable_span': (len(str(source_word)), len(str(target_word))),
                'stress_alignment': 0.5,
                'assonance_score': similarity or 0.0,
                'consonance_score': (similarity or 0.0) * 0.6,
                'internal_rhyme_score': (similarity or 0.0) * 0.4,
            }
            prosody_profile = {
                'source_total_syllables': len(str(source_context or "").split()),
                'target_total_syllables': len(str(target_context or "").split()),
                'complexity_tag': 'steady',
            }
            return {
                'similarity': similarity,
                'rarity': None,
                'combined': similarity,
                'rhyme_type': None,
                'signature_matches': matches,
                'target_signatures': sorted(signature_set),
                'features': {},
                'feature_profile': feature_profile,
                'prosody_profile': prosody_profile,
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
        self.anti_llm_engine = AntiLLMRhymeEngine(
            db_path=self.db_path,
            phonetic_analyzer=self.phonetic_analyzer,
        )
        self.cultural_engine = CulturalIntelligenceEngine(
            db_path=self.db_path,
            phonetic_analyzer=self.phonetic_analyzer,
        )
        anti_setter = getattr(self.anti_llm_engine, "set_phonetic_analyzer", None)
        if callable(anti_setter):
            anti_setter(self.phonetic_analyzer)
        setter = getattr(self.cultural_engine, "set_phonetic_analyzer", None)
        if callable(setter):
            setter(self.phonetic_analyzer)
        prosody_setter = getattr(self.cultural_engine, "set_prosody_analyzer", None)
        if callable(prosody_setter):
            prosody_setter(self.phonetic_analyzer)

        # Initialize database
        self.check_database()
        self._refresh_rarity_map()

        print(f"🎵 RhymeRarity App initialized with database: {db_path}")

    @staticmethod
    def _normalize_source_name(name: Optional[str]) -> str:
        """Normalise filter labels for consistent lookups and comparisons."""

        if name is None:
            return ""
        return str(name).strip().lower().replace("_", "-")

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
        
        cursor.executemany(
            "INSERT INTO song_rhyme_patterns (pattern, source_word, target_word, artist, song_title, genre, line_distance, confidence_score, phonetic_similarity, cultural_significance, source_context, target_context) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            iter_demo_rhyme_rows()
        )

        conn.commit()
        conn.close()

        print(f"✅ Demo database created with {len(DEMO_RHYME_PATTERNS)} sample patterns")
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
            min_syllables: Optional[int] = None,
            max_syllables: Optional[int] = None,
            allowed_rhyme_types: Optional[List[str]] = None,
            bradley_devices: Optional[List[str]] = None,
            require_internal: bool = False,
            min_rarity: Optional[float] = None,
            min_stress_alignment: Optional[float] = None,
            cadence_focus: Optional[str] = None,
        ) -> Dict[str, List[Dict]]:
            """Search for rhymes in the patterns.db database.

            Returns a mapping keyed by rhyme source so downstream consumers can
            format each category independently.
            """

            def _empty_response() -> Dict[str, List[Dict]]:
                return {"cmu": [], "multi_word": [], "rap_db": []}

            if not source_word or not source_word.strip():
                return _empty_response()

            source_word = source_word.lower().strip()

            try:
                min_conf_threshold = float(min_confidence)
            except (TypeError, ValueError):
                min_conf_threshold = 0.0
            if min_conf_threshold < 0:
                min_conf_threshold = 0.0
            min_confidence = min_conf_threshold

            def _normalize_to_list(value: Optional[List[str] | str]) -> List[str]:
                """Return ``value`` coerced to a list of non-empty strings."""

                if value is None:
                    return []
                if isinstance(value, (list, tuple, set)):
                    return [str(v) for v in value if v is not None and str(v).strip()]
                if isinstance(value, str):
                    return [value] if value.strip() else []
                return [str(value)]

            normalize_name = self._normalize_source_name

            def _normalized_set(value: Optional[List[str] | str]) -> Set[str]:
                """Return a set of normalised filter names for user-supplied values."""

                return {
                    normalized
                    for normalized in (normalize_name(item) for item in _normalize_to_list(value))
                    if normalized
                }

            def _coerce_float(value: Any) -> Optional[float]:
                """Safely convert a value to ``float`` for numeric filters."""

                if value is None:
                    return None
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return None

            def _prepare_confidence_defaults(entry: Dict) -> float:
                """Populate common confidence fields and return the comparison score."""

                combined_value = _coerce_float(entry.get("combined_score"))
                confidence_value = _coerce_float(entry.get("confidence"))

                score_for_filter = (
                    combined_value
                    if combined_value is not None
                    else confidence_value
                )
                if score_for_filter is None:
                    score_for_filter = 0.0

                entry["combined_score"] = (
                    combined_value if combined_value is not None else score_for_filter
                )
                entry["confidence"] = (
                    confidence_value if confidence_value is not None else score_for_filter
                )

                return score_for_filter

            cultural_filters = _normalized_set(cultural_significance)
            genre_filters = _normalized_set(genres)

            rhyme_type_filters = _normalized_set(allowed_rhyme_types)
            bradley_filters = _normalized_set(bradley_devices)
            cadence_focus_normalized = None
            if isinstance(cadence_focus, str) and cadence_focus.strip():
                cadence_focus_normalized = normalize_name(cadence_focus) or None

            min_syllable_threshold = None if min_syllables is None else max(1, int(min_syllables))
            max_syllable_threshold = None if max_syllables is None else max(1, int(max_syllables))
            min_rarity_threshold = None
            if min_rarity is not None:
                try:
                    min_rarity_threshold = float(min_rarity)
                except (TypeError, ValueError):
                    min_rarity_threshold = None
            min_stress_threshold = None
            if min_stress_alignment is not None:
                try:
                    min_stress_threshold = float(min_stress_alignment)
                except (TypeError, ValueError):
                    min_stress_threshold = None

            selected_sources = _normalized_set(result_sources)
            if not selected_sources:
                selected_sources = {"phonetic", "cultural", "anti-llm"}

            include_phonetic = "phonetic" in selected_sources
            include_cultural = "cultural" in selected_sources
            include_anti_llm = "anti-llm" in selected_sources

            filters_active = bool(
                cultural_filters
                or genre_filters
                or rhyme_type_filters
                or bradley_filters
                or cadence_focus_normalized
                or (max_line_distance is not None)
                or (min_syllable_threshold is not None)
                or (max_syllable_threshold is not None)
                or (min_rarity_threshold is not None)
                or (min_stress_threshold is not None)
                or require_internal
            )

            if limit <= 0:
                return _empty_response()

            analyzer = getattr(self, "phonetic_analyzer", None)
            cultural_engine = getattr(self, "cultural_engine", None)
            cmu_loader = None
            if analyzer is not None:
                cmu_loader = getattr(analyzer, "cmu_loader", None)
            if cmu_loader is None:
                cmu_loader = getattr(self, "cmu_loader", None)

            def _fallback_signature(word: str) -> Set[str]:
                """Lightweight rhyme signature used when no analyzer data is available."""

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

            # Build the source word signature set using progressively simpler
            # strategies so that downstream comparisons always have at least a
            # minimal representation to work with.
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

            # Fetch more CMU candidates than requested results so that scoring
            # filters can discard low-quality matches without leaving the
            # response empty.
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
            module1_seed_payload: List[Dict] = []
            aggregated_seed_signatures: Set[str] = set(source_signature_set)
            delivered_words_set: Set[str] = set()
            if include_phonetic:
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

                    # Attempt to evaluate cultural alignment when the cultural
                    # engine is available; otherwise fall back to a lightweight
                    # profile derived from the phonetic analyzer.
                    alignment = None
                    if cultural_engine and hasattr(cultural_engine, "evaluate_rhyme_alignment"):
                        try:
                            alignment = cultural_engine.evaluate_rhyme_alignment(
                                source_word,
                                target,
                                threshold=phonetic_threshold,
                                rhyme_signatures=source_signature_set,
                                source_context=None,
                                target_context=None,
                            )
                        except Exception:
                            alignment = None
                        if alignment is None:
                            continue
                    else:
                        profile_payload: Dict[str, Any] = {}
                        if analyzer is not None:
                            try:
                                profile_obj = analyzer.derive_rhyme_profile(
                                    source_word,
                                    target,
                                    similarity=similarity,
                                )
                            except Exception:
                                profile_obj = None
                            if profile_obj is not None:
                                if hasattr(profile_obj, "as_dict"):
                                    try:
                                        profile_payload = dict(profile_obj.as_dict())
                                    except Exception:
                                        profile_payload = {}
                                elif isinstance(profile_obj, dict):
                                    profile_payload = dict(profile_obj)
                                else:
                                    try:
                                        profile_payload = dict(vars(profile_obj))
                                    except Exception:
                                        profile_payload = {}

                        alignment = {
                            "similarity": similarity,
                            "rarity": rarity_value,
                            "combined": combined,
                            "signature_matches": [],
                            "target_signatures": [],
                            "features": {},
                            "feature_profile": profile_payload,
                            "prosody_profile": {},
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

                    # Normalize optional feature profile information from the
                    # alignment payload so that downstream UI components always
                    # interact with plain dictionaries.
                    feature_profile_obj = alignment.get("feature_profile") or {}
                    feature_profile: Dict[str, Any] = {}
                    if feature_profile_obj:
                        if isinstance(feature_profile_obj, dict):
                            feature_profile = dict(feature_profile_obj)
                        else:
                            try:
                                feature_profile = dict(feature_profile_obj)
                            except Exception:
                                feature_profile = {}
                    if feature_profile:
                        if entry.get("rhyme_type") and "rhyme_type" not in feature_profile:
                            feature_profile["rhyme_type"] = entry["rhyme_type"]
                        entry["feature_profile"] = feature_profile
                        bradley_device = feature_profile.get("bradley_device")
                        if bradley_device:
                            entry["bradley_device"] = bradley_device
                        syllable_span = feature_profile.get("syllable_span")
                        if isinstance(syllable_span, (list, tuple)) and len(syllable_span) == 2:
                            entry["syllable_span"] = [int(syllable_span[0]), int(syllable_span[1])]
                        stress_alignment = feature_profile.get("stress_alignment")
                        if stress_alignment is not None:
                            entry["stress_alignment"] = stress_alignment
                        entry["assonance_score"] = feature_profile.get("assonance_score")
                        entry["consonance_score"] = feature_profile.get("consonance_score")
                        entry["internal_rhyme_score"] = feature_profile.get("internal_rhyme_score")

                    prosody_profile = alignment.get("prosody_profile")
                    if prosody_profile:
                        entry["prosody_profile"] = prosody_profile

                    score_for_filter = _prepare_confidence_defaults(entry)
                    if score_for_filter < min_confidence:
                        continue

                    phonetic_entries.append(entry)

                phonetic_entries.sort(
                    key=lambda entry: (
                        entry.get("combined_score", entry.get("confidence", 0.0)),
                        entry.get("phonetic_sim", 0.0),
                    ),
                    reverse=True,
                )

                delivered_words_set = {
                    str(item.get("target_word", "")).strip().lower()
                    for item in phonetic_entries
                    if item.get("target_word")
                }

                rare_seed_candidates: List[Dict] = []
                for entry in phonetic_entries:
                    target_value = entry.get("target_word")
                    if not target_value:
                        continue

                    rarity_value = entry.get("rarity_score")
                    try:
                        rarity_float = float(rarity_value) if rarity_value is not None else 0.0
                    except (TypeError, ValueError):
                        rarity_float = 0.0

                    combined_value = entry.get("combined_score", entry.get("confidence", 0.0))
                    try:
                        combined_float = float(combined_value) if combined_value is not None else 0.0
                    except (TypeError, ValueError):
                        combined_float = 0.0

                    signature_values = entry.get("target_rhyme_signatures") or entry.get("matched_signatures")
                    if signature_values:
                        signature_set = {str(sig) for sig in signature_values if sig}
                    else:
                        signature_set = _fallback_signature(str(target_value))

                    aggregated_seed_signatures.update(signature_set)

                    rare_seed_candidates.append(
                        {
                            "word": target_value,
                            "rarity": rarity_float,
                            "combined": combined_float,
                            "signatures": list(signature_set),
                            "feature_profile": entry.get("feature_profile", {}),
                            "prosody_profile": entry.get("prosody_profile", {}),
                        }
                    )

                rare_seed_candidates.sort(
                    key=lambda payload: (payload.get("rarity", 0.0), payload.get("combined", 0.0)),
                    reverse=True,
                )

                max_seed_count = max(3, min(6, limit))
                for candidate in rare_seed_candidates:
                    if len(module1_seed_payload) >= max_seed_count:
                        break
                    if candidate.get("rarity", 0.0) <= 0 and module1_seed_payload:
                        continue
                    module1_seed_payload.append(candidate)

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
                            source_context=entry.get("source_context"),
                            target_context=entry.get("target_context"),
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
                        feature_profile_obj = alignment.get("feature_profile")
                        feature_profile: Dict[str, Any] = {}
                        if feature_profile_obj:
                            if isinstance(feature_profile_obj, dict):
                                feature_profile = dict(feature_profile_obj)
                            else:
                                try:
                                    feature_profile = dict(feature_profile_obj)
                                except Exception:
                                    feature_profile = {}
                        if feature_profile:
                            if entry.get("rhyme_type") and "rhyme_type" not in feature_profile:
                                feature_profile["rhyme_type"] = entry["rhyme_type"]
                            entry["feature_profile"] = feature_profile
                            device = feature_profile.get("bradley_device")
                            if device:
                                entry["bradley_device"] = device
                            syllable_span = feature_profile.get("syllable_span")
                            if isinstance(syllable_span, (list, tuple)) and len(syllable_span) == 2:
                                entry["syllable_span"] = [int(syllable_span[0]), int(syllable_span[1])]
                            stress_alignment = feature_profile.get("stress_alignment")
                            if stress_alignment is not None:
                                entry["stress_alignment"] = stress_alignment
                            entry["assonance_score"] = feature_profile.get("assonance_score")
                            entry["consonance_score"] = feature_profile.get("consonance_score")
                            entry["internal_rhyme_score"] = feature_profile.get("internal_rhyme_score")
                        prosody_profile = alignment.get("prosody_profile")
                        if prosody_profile:
                            entry["prosody_profile"] = prosody_profile
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
                    conditions.append(
                        f"REPLACE(LOWER(cultural_significance), '_', '-') IN ({placeholders})"
                    )
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
                    swapped_source = row[1]
                    swapped_target = row[0]
                    swapped_pattern = entry["pattern"]
                    if swapped_source and swapped_target:
                        swapped_pattern = f"{swapped_source} / {swapped_target}"
                    entry = {
                        **entry,
                        "source_word": swapped_source,
                        "target_word": swapped_target,
                        "pattern": swapped_pattern,
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
            if include_anti_llm:
                anti_patterns = self.anti_llm_engine.generate_anti_llm_patterns(
                    source_word,
                    limit=limit,
                    module1_seeds=module1_seed_payload,
                    seed_signatures=aggregated_seed_signatures,
                    delivered_words=delivered_words_set,
                )
                for pattern in anti_patterns:
                    feature_profile = getattr(pattern, "feature_profile", {}) or {}
                    syllable_span = getattr(pattern, "syllable_span", None)
                    stress_alignment = getattr(pattern, "stress_alignment", None)
                    internal_rhyme = None
                    if isinstance(feature_profile, dict):
                        internal_rhyme = feature_profile.get("internal_rhyme_score")
                    confidence_float = _coerce_float(getattr(pattern, "confidence", None))
                    if confidence_float is None:
                        confidence_float = 0.0
                    combined_metric = getattr(pattern, "combined", None)
                    if combined_metric is None:
                        combined_metric = getattr(pattern, "combined_score", None)
                    combined_float = _coerce_float(combined_metric)
                    entry_payload = {
                        "source_word": source_word,
                        "target_word": pattern.target_word,
                        "pattern": f"{source_word} / {pattern.target_word}",
                        "confidence": confidence_float,
                        "rarity_score": pattern.rarity_score,
                        "llm_weakness_type": pattern.llm_weakness_type,
                        "cultural_depth": pattern.cultural_depth,
                        "result_source": "anti_llm",
                        "source_rhyme_signatures": source_signature_list,
                        "source_phonetic_profile": source_phonetic_profile,
                        "phonetic_threshold": phonetic_threshold,
                        "feature_profile": feature_profile,
                        "bradley_device": getattr(pattern, "bradley_device", None),
                        "stress_alignment": stress_alignment,
                        "internal_rhyme_score": internal_rhyme,
                    }
                    if isinstance(syllable_span, (list, tuple)) and len(syllable_span) == 2:
                        entry_payload["syllable_span"] = [int(syllable_span[0]), int(syllable_span[1])]
                    prosody_payload = getattr(pattern, "prosody_profile", None)
                    if isinstance(prosody_payload, dict):
                        entry_payload["prosody_profile"] = prosody_payload
                    entry_payload["combined_score"] = (
                        combined_float if combined_float is not None else confidence_float
                    )
                    score_for_filter = _prepare_confidence_defaults(entry_payload)
                    if score_for_filter < min_confidence:
                        continue
                    anti_llm_entries.append(entry_payload)

            def _passes_min_confidence(entry: Dict) -> bool:
                score = _prepare_confidence_defaults(entry)
                return score >= min_confidence

            def _ensure_feature_profile(entry: Dict) -> Dict[str, Any]:
                """Return a mutable feature profile dictionary for ``entry``."""

                profile_obj = entry.get("feature_profile")
                if isinstance(profile_obj, dict):
                    return profile_obj
                if profile_obj is None:
                    profile_dict: Dict[str, Any] = {}
                elif hasattr(profile_obj, "__dict__"):
                    try:
                        profile_dict = dict(vars(profile_obj))
                    except Exception:
                        profile_dict = {}
                else:
                    try:
                        profile_dict = dict(profile_obj)
                    except Exception:
                        profile_dict = {}
                entry["feature_profile"] = profile_dict
                return profile_dict

            def _extract_rhyme_category(entry: Dict) -> Optional[str]:
                """Pull a rhyme category label from an entry and normalise storage."""

                rhyme_value = entry.get("rhyme_type")
                if not rhyme_value:
                    features_obj = entry.get("phonetic_features")
                    if features_obj is not None:
                        if isinstance(features_obj, dict):
                            rhyme_value = features_obj.get("rhyme_type")
                        elif hasattr(features_obj, "__dict__"):
                            rhyme_value = vars(features_obj).get("rhyme_type")
                        else:
                            try:
                                rhyme_value = dict(features_obj).get("rhyme_type")
                            except Exception:
                                rhyme_value = None
                if not rhyme_value:
                    profile_obj = entry.get("feature_profile")
                    if profile_obj is not None:
                        if isinstance(profile_obj, dict):
                            rhyme_value = profile_obj.get("rhyme_type")
                        elif hasattr(profile_obj, "__dict__"):
                            rhyme_value = vars(profile_obj).get("rhyme_type")
                        else:
                            try:
                                rhyme_value = dict(profile_obj).get("rhyme_type")
                            except Exception:
                                rhyme_value = None
                if rhyme_value:
                    entry["rhyme_type"] = rhyme_value
                    profile_dict = _ensure_feature_profile(entry)
                    profile_dict.setdefault("rhyme_type", rhyme_value)
                return rhyme_value

            def _extract_rhythm_score(entry: Dict) -> Optional[float]:
                """Return a numeric rhythm/stress alignment score for an entry."""

                raw_value = entry.get("rhythm_score")
                if raw_value is None:
                    raw_value = entry.get("stress_alignment")
                if raw_value is None:
                    profile_obj = entry.get("feature_profile")
                    if profile_obj is not None:
                        if isinstance(profile_obj, dict):
                            raw_value = profile_obj.get("stress_alignment")
                        elif hasattr(profile_obj, "__dict__"):
                            raw_value = vars(profile_obj).get("stress_alignment")
                        else:
                            try:
                                raw_value = dict(profile_obj).get("stress_alignment")
                            except Exception:
                                raw_value = None
                if raw_value is None:
                    phonetic_features = entry.get("phonetic_features")
                    if phonetic_features is not None:
                        if isinstance(phonetic_features, dict):
                            raw_value = phonetic_features.get("stress_alignment")
                        elif hasattr(phonetic_features, "__dict__"):
                            raw_value = vars(phonetic_features).get("stress_alignment")
                        else:
                            try:
                                raw_value = dict(phonetic_features).get("stress_alignment")
                            except Exception:
                                raw_value = None
                try:
                    rhythm_float = float(raw_value)
                except (TypeError, ValueError):
                    return None

                entry["rhythm_score"] = rhythm_float
                entry["stress_alignment"] = rhythm_float
                profile_dict = _ensure_feature_profile(entry)
                profile_dict.setdefault("stress_alignment", rhythm_float)
                return rhythm_float

            def _filter_entry(entry: Dict) -> bool:
                if not _passes_min_confidence(entry):
                    return False

                entry_rhyme = _extract_rhyme_category(entry)
                rhythm_score = _extract_rhythm_score(entry)

                if not filters_active:
                    return True

                if cultural_filters:
                    sig = entry.get("cultural_sig")
                    if sig is None or normalize_name(str(sig)) not in cultural_filters:
                        return False
                if genre_filters:
                    genre_value = entry.get("genre")
                    if genre_value is None or normalize_name(str(genre_value)) not in genre_filters:
                        return False
                if rhyme_type_filters:
                    if not entry_rhyme or normalize_name(str(entry_rhyme)) not in rhyme_type_filters:
                        return False
                if bradley_filters:
                    device_value = entry.get("bradley_device")
                    if not device_value and entry.get("feature_profile"):
                        device_value = entry["feature_profile"].get("bradley_device")
                    if not device_value or normalize_name(str(device_value)) not in bradley_filters:
                        return False
                if min_rarity_threshold is not None:
                    rarity_metric = entry.get("rarity_score")
                    if rarity_metric is None:
                        rarity_metric = entry.get("cultural_rarity")
                    try:
                        rarity_value = float(rarity_metric) if rarity_metric is not None else 0.0
                    except (TypeError, ValueError):
                        rarity_value = 0.0
                    if rarity_value < min_rarity_threshold:
                        return False
                if min_stress_threshold is not None:
                    if rhythm_score is None or rhythm_score < min_stress_threshold:
                        return False
                if cadence_focus_normalized:
                    prosody = entry.get("prosody_profile") or {}
                    cadence_value = prosody.get("complexity_tag") if isinstance(prosody, dict) else None
                    if cadence_value is None:
                        cadence_value = entry.get("feature_profile", {}).get("complexity_tag")
                    if not cadence_value or normalize_name(str(cadence_value)) != cadence_focus_normalized:
                        return False
                if require_internal:
                    internal_value = entry.get("internal_rhyme_score")
                    if internal_value is None and entry.get("feature_profile"):
                        internal_value = entry["feature_profile"].get("internal_rhyme_score")
                    try:
                        internal_float = float(internal_value) if internal_value is not None else 0.0
                    except (TypeError, ValueError):
                        internal_float = 0.0
                    if internal_float < 0.4:
                        return False
                if min_syllable_threshold is not None or max_syllable_threshold is not None:
                    syllable_span = entry.get("syllable_span")
                    if not syllable_span and entry.get("feature_profile"):
                        syllable_span = entry["feature_profile"].get("syllable_span")
                    if isinstance(syllable_span, (list, tuple)) and len(syllable_span) == 2:
                        try:
                            target_syllables = int(syllable_span[1])
                        except (TypeError, ValueError):
                            target_syllables = None
                    else:
                        target_word = entry.get("target_word")
                        try:
                            estimator = getattr(self.phonetic_analyzer, "estimate_syllables", None)
                            if callable(estimator):
                                target_syllables = estimator(str(target_word)) if target_word else None
                            else:
                                target_syllables = self.phonetic_analyzer._count_syllables(str(target_word)) if target_word else None
                        except Exception:
                            target_syllables = None
                    if min_syllable_threshold is not None and (target_syllables is None or target_syllables < min_syllable_threshold):
                        return False
                    if max_syllable_threshold is not None and (target_syllables is None or target_syllables > max_syllable_threshold):
                        return False
                if max_line_distance is not None:
                    distance_value = entry.get("distance")
                    if distance_value is None or distance_value > max_line_distance:
                        return False

                return True

            def _rarity_value(entry: Dict) -> float:
                rarity_metric = entry.get("rarity_score")
                if rarity_metric is None:
                    rarity_metric = entry.get("cultural_rarity")
                try:
                    return float(rarity_metric)
                except (TypeError, ValueError):
                    return 0.0

            def _confidence_value(entry: Dict) -> float:
                metric = entry.get("combined_score")
                if metric is None:
                    metric = entry.get("confidence")
                try:
                    return float(metric)
                except (TypeError, ValueError):
                    return 0.0

            def _normalize_entry(entry: Dict, category_key: str) -> Dict:
                entry["result_source"] = category_key
                if not entry.get("pattern") and entry.get("target_word"):
                    entry["pattern"] = f"{source_word} / {entry['target_word']}"
                if entry.get("combined_score") is None and entry.get("confidence") is not None:
                    entry["combined_score"] = entry["confidence"]
                if entry.get("confidence") is None and entry.get("combined_score") is not None:
                    entry["confidence"] = entry["combined_score"]

                feature_profile = entry.get("feature_profile")
                if feature_profile is None:
                    entry["feature_profile"] = {}
                elif not isinstance(feature_profile, dict):
                    try:
                        entry["feature_profile"] = dict(feature_profile)
                    except Exception:
                        entry["feature_profile"] = {}

                prosody_profile = entry.get("prosody_profile")
                if prosody_profile is None:
                    entry["prosody_profile"] = {}

                for field, default in {
                    "rarity_score": None,
                    "cultural_rarity": None,
                    "phonetic_sim": None,
                    "rhyme_type": None,
                    "bradley_device": None,
                    "stress_alignment": None,
                    "internal_rhyme_score": None,
                    "assonance_score": None,
                    "consonance_score": None,
                    "cultural_context": None,
                    "llm_weakness_type": None,
                    "cultural_depth": None,
                }.items():
                    entry.setdefault(field, default)

                entry.setdefault("cultural_sig", None)
                entry.setdefault("genre", None)
                entry.setdefault("source_context", None)
                entry.setdefault("target_context", None)
                entry.setdefault("distance", None)

                return entry

            def _process_entries(entries: List[Dict], category_key: str) -> List[Dict]:
                if not entries:
                    return []

                processed: List[Dict] = []
                seen_pairs = set()
                for entry in entries:
                    target_value = entry.get("target_word")
                    if not target_value:
                        continue

                    source_value = entry.get("source_word", source_word)
                    pair = (
                        str(source_value).strip().lower(),
                        str(target_value).strip().lower(),
                    )
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)

                    normalized_entry = _normalize_entry(entry, category_key)
                    if not _filter_entry(normalized_entry):
                        continue

                    normalized_entry.pop("source_word", None)
                    processed.append(normalized_entry)

                processed.sort(
                    key=lambda entry: (
                        _rarity_value(entry),
                        _confidence_value(entry),
                    ),
                    reverse=True,
                )

                return processed[:limit]

            return {
                "cmu": _process_entries(phonetic_entries if include_phonetic else [], "cmu"),
                "multi_word": _process_entries(anti_llm_entries if include_anti_llm else [], "multi_word"),
                "rap_db": _process_entries(cultural_entries if include_cultural else [], "rap_db"),
            }



    def format_rhyme_results(self, source_word: str, rhymes: Dict[str, List[Dict]]) -> str:
        """Format rhyme search results for display grouped by source."""

        category_order: List[Tuple[str, str]] = [
            ("cmu", "📚 CMU Pronouncing Dictionary"),
            ("rap_db", "🎤 Cultural Pattern Database"),
            ("multi_word", "🧠 Anti-LLM Multi-Word Patterns"),
        ]

        def _normalize_source_key(value: Optional[str]) -> str:
            mapping = {
                "phonetic": "cmu",
                "cultural": "rap_db",
                "anti_llm": "multi_word",
                "anti-llm": "multi_word",
            }
            if value is None:
                return ""
            return mapping.get(str(value), str(value))

        if not rhymes or not any(rhymes.get(key) for key, _ in category_order):
            return f"❌ No rhymes found for '{source_word}'\n\nTry words like: love, mind, flow, time, money"

        result = f"🎯 **TARGET RHYMES for '{source_word.upper()}':**\n"
        result += "=" * 50 + "\n\n"

        def _format_rhyme_type_line(rhyme_entry: Dict) -> str:
            type_value = (
                rhyme_entry.get("rhyme_type")
                or (rhyme_entry.get("phonetic_features") or {}).get("rhyme_type")
                or (rhyme_entry.get("feature_profile") or {}).get("rhyme_type")
            )
            if not type_value:
                return ""
            label = str(type_value).replace("_", " " ).title()
            return f"   🎼 Rhyme Type: {label}\n"

        def _format_rhythm_line(rhyme_entry: Dict) -> str:
            rhythm_value = (
                rhyme_entry.get("rhythm_score")
                if rhyme_entry.get("rhythm_score") is not None
                else rhyme_entry.get("stress_alignment")
            )
            if rhythm_value is None:
                profile = rhyme_entry.get("feature_profile") or {}
                if not isinstance(profile, dict):
                    try:
                        profile = dict(profile)
                    except Exception:
                        profile = {}
                rhythm_value = profile.get("stress_alignment")
            if rhythm_value is None:
                phonetic_features = rhyme_entry.get("phonetic_features") or {}
                if not isinstance(phonetic_features, dict):
                    try:
                        phonetic_features = dict(phonetic_features)
                    except Exception:
                        phonetic_features = {}
                rhythm_value = phonetic_features.get("stress_alignment")
            try:
                rhythm_float = float(rhythm_value)
            except (TypeError, ValueError):
                return ""
            return f"   ♟️ Rhythm Match: {rhythm_float:.2f}\n"

        entry_counter = 1
        max_entries = 15

        for category_key, heading in category_order:
            entries = list(rhymes.get(category_key) or [])
            if not entries:
                continue

            result += f"**{heading}**\n"
            for rhyme in entries:
                if entry_counter > max_entries:
                    break

                normalized_source = _normalize_source_key(rhyme.get("result_source")) or category_key
                rhyme["result_source"] = normalized_source

                result += f"**{entry_counter:2d}. {rhyme['target_word'].upper()}**\n"

                if normalized_source in {"cmu", "phonetic"}:
                    origin = rhyme.get("artist", "CMU Pronouncing Dictionary")
                    pattern_text = rhyme.get("pattern") or f"{source_word} / {rhyme['target_word']}"
                    result += f"   🏷️ Origin: 📚 {origin}\n"
                    result += f"   📝 Pattern: {pattern_text}\n"
                    phonetic_score = rhyme.get("phonetic_sim")
                    if phonetic_score is not None:
                        result += f"   📊 Phonetic Score: {float(phonetic_score):.2f}\n"
                    result += _format_rhyme_type_line(rhyme)
                    rarity_value = rhyme.get("rarity_score")
                    if rarity_value is not None:
                        result += f"   🌟 Rarity Score: {float(rarity_value):.2f}\n"
                    combined_value = rhyme.get("combined_score")
                    if combined_value is not None:
                        result += f"   🎯 Combined Score: {float(combined_value):.2f}\n"
                    feature_profile = rhyme.get("feature_profile") or {}
                    if feature_profile:
                        device = feature_profile.get("bradley_device") or rhyme.get("bradley_device")
                        if device:
                            result += f"   🎙️ Bradley Device: {str(device).title()}\n"
                        syllable_span = feature_profile.get("syllable_span") or rhyme.get("syllable_span")
                        if isinstance(syllable_span, (list, tuple)) and len(syllable_span) == 2:
                            result += f"   🔢 Syllable Span: {syllable_span[0]} → {syllable_span[1]}\n"
                        assonance = feature_profile.get("assonance_score") or rhyme.get("assonance_score")
                        consonance = feature_profile.get("consonance_score") or rhyme.get("consonance_score")
                        if assonance is not None or consonance is not None:
                            result += (
                                f"   🔊 Sonic Blend – Assonance: {assonance or 0:.2f} | "
                                f"Consonance: {consonance or 0:.2f}\n"
                            )
                    rhythm_line = _format_rhythm_line(rhyme)
                    if rhythm_line:
                        result += rhythm_line
                    prosody = rhyme.get("prosody_profile") or {}
                    if isinstance(prosody, dict) and prosody:
                        cadence = prosody.get("complexity_tag")
                        if cadence:
                            result += f"   🥁 Cadence: {str(cadence).replace('_', ' ').title()}\n"
                        cadence_ratio = prosody.get("cadence_ratio")
                        if cadence_ratio:
                            result += f"   ⚖️ Cadence Ratio: {float(cadence_ratio):.2f}\n"
                    note = rhyme.get("source_context") or "Pronunciation-based suggestion."
                    result += f"   🗒️ Note: {note}\n\n"
                elif normalized_source in {"multi_word", "anti_llm", "anti-llm"}:
                    pattern_text = rhyme.get("pattern") or f"{source_word} / {rhyme['target_word']}"
                    result += "   🤖 Source: Anti-LLM Engine\n"
                    result += f"   📝 Pattern: {pattern_text}\n"
                    confidence = rhyme.get("confidence")
                    if confidence is not None:
                        result += f"   📈 Confidence: {float(confidence):.2f}\n"
                    rarity = rhyme.get("rarity_score")
                    if rarity is not None:
                        result += f"   🌟 Rarity Score: {float(rarity):.2f}\n"
                    result += _format_rhyme_type_line(rhyme)
                    weakness = rhyme.get("llm_weakness_type")
                    if weakness:
                        weakness_label = str(weakness).replace("_", " " ).title()
                        result += f"   🧩 LLM Weakness: {weakness_label}\n"
                    cultural_depth = rhyme.get("cultural_depth")
                    if cultural_depth:
                        result += f"   🌌 Cultural Depth: {cultural_depth}\n"
                    feature_profile = rhyme.get("feature_profile") or {}
                    if feature_profile:
                        device = feature_profile.get("bradley_device") or rhyme.get("bradley_device")
                        if device:
                            result += f"   🎙️ Bradley Device: {str(device).title()}\n"
                        internal_score = feature_profile.get("internal_rhyme_score") or rhyme.get("internal_rhyme_score")
                        if internal_score is not None:
                            result += f"   🌀 Internal Rhyme Potential: {float(internal_score):.2f}\n"
                        syllable_span = feature_profile.get("syllable_span") or rhyme.get("syllable_span")
                        if isinstance(syllable_span, (list, tuple)) and len(syllable_span) == 2:
                            result += f"   🔢 Syllable Span: {syllable_span[0]} → {syllable_span[1]}\n"
                    rhythm_line = _format_rhythm_line(rhyme)
                    if rhythm_line:
                        result += rhythm_line
                    prosody = rhyme.get("prosody_profile") or {}
                    if isinstance(prosody, dict) and prosody.get("complexity_tag"):
                        result += f"   🥁 Cadence: {prosody['complexity_tag'].replace('_', ' ').title()}\n"
                    context_note = rhyme.get("source_context") or "Designed to exploit LLM weaknesses."
                    if context_note:
                        result += f"   🧠 Insight: {context_note}\n"
                    result += "\n"
                else:
                    pattern_text = rhyme.get("pattern") or f"{source_word} / {rhyme['target_word']}"
                    result += f"   🎤 Source: {rhyme.get('artist', 'Unknown')} - {rhyme.get('song', 'Unknown')}\n"
                    result += f"   📝 Pattern: {pattern_text}\n"
                    confidence_value = rhyme.get("confidence")
                    phonetic_value = rhyme.get("phonetic_sim")
                    if confidence_value is not None or phonetic_value is not None:
                        result += "   📊 Confidence: "
                        if confidence_value is not None:
                            result += f"{float(confidence_value):.2f}"
                        else:
                            result += "--"
                        result += " | Phonetic: "
                        if phonetic_value is not None:
                            result += f"{float(phonetic_value):.2f}\n"
                        else:
                            result += "--\n"
                    context_info = rhyme.get("cultural_context")
                    if context_info and hasattr(context_info, "__dict__"):
                        context_info = dict(vars(context_info))

                    def _humanize(value: Optional[str]) -> str:
                        if value is None:
                            return ""
                        if isinstance(value, str):
                            return value.replace("_", " " ).title()
                        return str(value)

                    cultural_highlights: List[str] = []
                    if isinstance(context_info, dict):
                        era_value = context_info.get("era")
                        if era_value:
                            cultural_highlights.append(f"Era: {_humanize(era_value)}")
                        region_value = context_info.get("regional_origin")
                        if region_value:
                            cultural_highlights.append(f"Region: {_humanize(region_value)}")
                        significance_value = context_info.get("cultural_significance") or rhyme.get("cultural_sig")
                        if significance_value:
                            cultural_highlights.append(f"Significance: {_humanize(significance_value)}")

                    rarity_score = rhyme.get("cultural_rarity")
                    if rarity_score is not None:
                        cultural_highlights.append(f"Rarity Score: {float(rarity_score):.2f}")

                    if cultural_highlights:
                        result += f"   🌍 Cultural Context: {' | '.join(cultural_highlights)}\n"

                    result += _format_rhyme_type_line(rhyme)

                    if isinstance(context_info, dict):
                        styles = context_info.get("style_characteristics")
                        if styles:
                            formatted_styles = ", ".join(_humanize(style) for style in styles if style)
                            if formatted_styles:
                                result += f"   🎨 Styles: {formatted_styles}\n"

                    feature_profile = rhyme.get("feature_profile") or {}
                    if feature_profile:
                        device = feature_profile.get("bradley_device") or rhyme.get("bradley_device")
                        if device:
                            result += f"   🎙️ Bradley Device: {str(device).title()}\n"
                        syllable_span = feature_profile.get("syllable_span") or rhyme.get("syllable_span")
                        if isinstance(syllable_span, (list, tuple)) and len(syllable_span) == 2:
                            result += f"   🔢 Syllable Span: {syllable_span[0]} → {syllable_span[1]}\n"
                        assonance = feature_profile.get("assonance_score") or rhyme.get("assonance_score")
                        consonance = feature_profile.get("consonance_score") or rhyme.get("consonance_score")
                        if assonance is not None or consonance is not None:
                            result += (
                                f"   🔊 Sonic Blend – Assonance: {assonance or 0:.2f} | "
                                f"Consonance: {consonance or 0:.2f}\n"
                            )
                    prosody = rhyme.get("prosody_profile") or {}
                    if isinstance(prosody, dict) and prosody.get("complexity_tag"):
                        cadence = prosody["complexity_tag"]
                        result += f"   🥁 Cadence: {str(cadence).replace('_', ' ').title()}\n"
                    rhythm_line = _format_rhythm_line(rhyme)
                    if rhythm_line:
                        result += rhythm_line
                    source_context = rhyme.get("source_context", "")
                    target_context = rhyme.get("target_context", "")
                    result += f"   🎵 Context: \"{source_context}\" → \"{target_context}\"\n\n"

                entry_counter += 1

            if entry_counter > max_entries:
                return result

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
            rhythm_threshold,
            cadence_focus_choice,
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

            max_distance: Optional[int]
            if max_line_distance_choice in (None, "", "Any"):
                max_distance = None
            else:
                try:
                    max_distance = int(max_line_distance_choice)
                except (TypeError, ValueError):
                    max_distance = None

            cadence_filter: Optional[str]
            if cadence_focus_choice in (None, "", "Any"):
                cadence_filter = None
            else:
                cadence_filter = str(cadence_focus_choice)

            rhythm_threshold_value: Optional[float]
            if rhythm_threshold in (None, ""):
                rhythm_threshold_value = None
            else:
                try:
                    rhythm_threshold_value = float(rhythm_threshold)
                except (TypeError, ValueError):
                    rhythm_threshold_value = None
            if rhythm_threshold_value is not None:
                if rhythm_threshold_value < 0.0:
                    rhythm_threshold_value = 0.0
                elif rhythm_threshold_value > 1.0:
                    rhythm_threshold_value = 1.0

            rhymes = self.search_rhymes(
                word,
                limit=max_results,
                min_confidence=min_conf,
                cultural_significance=ensure_list(cultural_filter),
                genres=ensure_list(genre_filter),
                result_sources=ensure_list(source_filter),
                max_line_distance=max_distance,
                allowed_rhyme_types=ensure_list(rhyme_type_filter),
                min_stress_alignment=rhythm_threshold_value,
                cadence_focus=cadence_filter,
            )
            return self.format_rhyme_results(word, rhymes)

        normalized_cultural_labels: Set[str] = set()
        cultural_engine = getattr(self, "cultural_engine", None)
        if cultural_engine:
            for raw_label in getattr(cultural_engine, "cultural_categories", {}).keys():
                normalized = self._normalize_source_name(raw_label)
                if normalized:
                    normalized_cultural_labels.add(normalized)

        genre_options: List[str] = []
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT DISTINCT cultural_significance FROM song_rhyme_patterns "
                    "WHERE cultural_significance IS NOT NULL"
                )
                for (value,) in cursor.fetchall():
                    normalized = self._normalize_source_name(value)
                    if normalized:
                        normalized_cultural_labels.add(normalized)

                cursor.execute(
                    "SELECT DISTINCT genre FROM song_rhyme_patterns WHERE genre IS NOT NULL ORDER BY genre"
                )
                genre_options = [row[0] for row in cursor.fetchall() if row[0]]
        except sqlite3.Error:
            genre_options = []

        cultural_options = sorted(normalized_cultural_labels)

        default_sources = ["phonetic", "cultural"]

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
                        info="Filter by cultural importance (labels such as classic, cultural-icon, underground)",
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

                    rhyme_type_dropdown = gr.Dropdown(
                        choices=["perfect", "near", "slant", "eye", "weak"],
                        multiselect=True,
                        label="Rhyme Type",
                        info="Limit to specific rhyme categories",
                        value=[],
                    )

                    rhythm_threshold_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.0,
                        step=0.05,
                        label="Minimum Rhythm Match",
                        info="Set a minimum stress-alignment score for suggested rhymes",
                    )

                    cadence_dropdown = gr.Dropdown(
                        choices=["Any", "steady", "polysyllabic", "dense"],
                        value="Any",
                        label="Rhythm Focus",
                        info="Prioritise matches with a particular cadence profile",
                    )

                    max_line_distance_dropdown = gr.Dropdown(
                        choices=["Any", "1", "2", "3", "4", "5"],
                        value="Any",
                        label="Max Line Distance",
                    )

            with gr.Row():
                with gr.Column(scale=1, min_width=200):
                    search_btn = gr.Button("🔍 Find Rhymes", variant="primary", size="lg")
                with gr.Column(scale=3):
                    gr.Markdown(
                        "💡 Enter a word and adjust the filters, then press **Find Rhymes** to discover new lyric pairings."
                    )

            output = gr.Textbox(
                label="Rhyme Results",
                lines=20,
                max_lines=30,
                show_copy_button=True
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
                    rhyme_type_dropdown,
                    rhythm_threshold_slider,
                    cadence_dropdown,
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
