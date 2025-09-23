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
        extract_phrase_components,
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

    def extract_phrase_components(phrase, cmu_loader=None):
        base = str(phrase or "").strip()
        tokens = base.split()
        normalized_tokens = [token.lower() for token in tokens]
        normalized_phrase = " ".join(normalized_tokens)
        anchor_index = len(normalized_tokens) - 1 if normalized_tokens else None
        anchor = normalized_tokens[-1] if normalized_tokens else ""
        anchor_display = tokens[-1] if tokens else anchor
        syllable_counts = [1 for _ in normalized_tokens]
        total_syllables = max(1, len(syllable_counts)) if syllable_counts else 1
        return types.SimpleNamespace(
            original=base,
            tokens=tokens,
            normalized_tokens=normalized_tokens,
            normalized_phrase=normalized_phrase,
            anchor=anchor,
            anchor_display=anchor_display,
            anchor_index=anchor_index,
            syllable_counts=syllable_counts,
            total_syllables=total_syllables,
            anchor_pronunciations=[],
        )
    

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
                'syllable_span': (len(str(source_word)), len(str(target_word))),
                'stress_alignment': 0.5,
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

            source_components = extract_phrase_components(source_word, cmu_loader)
            source_anchor_word = source_components.anchor or source_word

            def _build_word_phonetics(word: Optional[str]) -> Dict[str, Any]:
                base_word = "" if word is None else str(word)
                defaults: Dict[str, Any] = {
                    "word": base_word,
                    "normalized": base_word.lower().strip(),
                    "syllables": None,
                    "stress_pattern": "",
                    "stress_pattern_display": "",
                    "meter_hint": None,
                    "metrical_foot": None,
                    "vowel_skeleton": "",
                    "consonant_tail": "",
                    "pronunciations": [],
                    "tokens": [],
                    "token_syllables": [],
                    "anchor_word": "",
                    "anchor_display": "",
                    "is_multi_word": False,
                }

                if analyzer and hasattr(analyzer, "describe_word"):
                    try:
                        description = analyzer.describe_word(base_word)
                    except Exception:
                        description = None
                    if description:
                        if isinstance(description, dict):
                            for key in defaults:
                                if key in description and description[key] not in (None, ""):
                                    defaults[key] = description[key]
                        else:
                            try:
                                desc_dict = dict(description)
                            except Exception:
                                desc_dict = {}
                            for key in defaults:
                                if key in desc_dict and desc_dict[key] not in (None, ""):
                                    defaults[key] = desc_dict[key]

                if not defaults.get("tokens") and defaults.get("normalized"):
                    defaults["tokens"] = defaults["normalized"].split()

                if not defaults.get("anchor_word") and defaults.get("tokens"):
                    defaults["anchor_word"] = defaults["tokens"][-1].lower()

                if not defaults.get("anchor_display") and defaults.get("tokens"):
                    defaults["anchor_display"] = defaults["tokens"][-1]

                defaults["is_multi_word"] = bool(
                    defaults.get("tokens") and len(defaults["tokens"]) > 1
                )
                if not defaults["is_multi_word"] and isinstance(defaults.get("normalized"), str):
                    defaults["is_multi_word"] = " " in defaults["normalized"]

                if defaults["syllables"] is None:
                    estimate_fn = getattr(analyzer, "estimate_syllables", None)
                    try:
                        if callable(estimate_fn):
                            defaults["syllables"] = estimate_fn(base_word)
                    except Exception:
                        defaults["syllables"] = None

                stress_pattern = defaults.get("stress_pattern")
                if stress_pattern and not defaults.get("stress_pattern_display"):
                    defaults["stress_pattern_display"] = "-".join(str(ch) for ch in str(stress_pattern))

                if not defaults.get("pronunciations"):
                    pronunciations: List[str] = []
                    if analyzer and hasattr(analyzer, "_get_pronunciation_variants"):
                        try:
                            variants = analyzer._get_pronunciation_variants(base_word)
                        except Exception:
                            variants = []
                        for phones in variants[:2]:
                            try:
                                stripped = " ".join(analyzer._strip_stress_markers(phones))
                            except Exception:
                                stripped = ""
                            if stripped:
                                pronunciations.append(stripped)
                    defaults["pronunciations"] = pronunciations

                return defaults

            def _sanitize_feature_profile(entry: Dict[str, Any]) -> Dict[str, Any]:
                profile_obj = entry.get("feature_profile")
                if profile_obj is None:
                    profile_dict: Dict[str, Any] = {}
                elif isinstance(profile_obj, dict):
                    profile_dict = dict(profile_obj)
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

                for banned_key in ("bradley_device", "assonance_score", "consonance_score"):
                    if banned_key in profile_dict:
                        profile_dict.pop(banned_key, None)

                entry["feature_profile"] = profile_dict
                entry.pop("bradley_device", None)
                entry.pop("assonance_score", None)
                entry.pop("consonance_score", None)
                return profile_dict

            def _ensure_target_phonetics(entry: Dict[str, Any]) -> Dict[str, Any]:
                target_profile = entry.get("target_phonetics")
                if isinstance(target_profile, dict):
                    return target_profile
                target_word = entry.get("target_word")
                profile = _build_word_phonetics(target_word)
                entry["target_phonetics"] = profile
                return profile

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
            if not source_signature_set and cmu_loader is not None and source_anchor_word:
                try:
                    source_signature_set = {
                        sig for sig in cmu_loader.get_rhyme_parts(source_anchor_word) if sig
                    }
                except Exception:
                    source_signature_set = set()
            if not source_signature_set:
                source_signature_set = _fallback_signature(source_anchor_word or source_word)

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
                        lookup_terms = {source_word}
                        if source_anchor_word and source_anchor_word != source_word:
                            lookup_terms.add(source_anchor_word)

                        for lookup in lookup_terms:
                            cursor.execute(
                                "SELECT target_word FROM song_rhyme_patterns WHERE source_word = ? AND target_word IS NOT NULL LIMIT 50",
                                (lookup,),
                            )
                            db_candidate_words.update(
                                str(row[0]).strip().lower()
                                for row in cursor.fetchall()
                                if row and row[0]
                            )

                        for lookup in lookup_terms:
                            cursor.execute(
                                "SELECT source_word FROM song_rhyme_patterns WHERE target_word = ? AND source_word IS NOT NULL LIMIT 50",
                                (lookup,),
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

            source_phonetics = _build_word_phonetics(source_word)

            prefix_tokens = []
            suffix_tokens = []
            if source_components.anchor_index is not None:
                prefix_tokens = source_components.normalized_tokens[: source_components.anchor_index]
                suffix_tokens = source_components.normalized_tokens[source_components.anchor_index + 1 :]

            source_phonetic_profile = {
                "word": source_word,
                "threshold": phonetic_threshold,
                "reference_similarity": reference_similarity,
                "signatures": source_signature_list,
                "phonetics": source_phonetics,
                "anchor_word": source_anchor_word,
                "anchor_display": source_components.anchor_display or source_anchor_word,
                "phrase_tokens": source_components.normalized_tokens,
                "phrase_prefix": " ".join(prefix_tokens).strip(),
                "phrase_suffix": " ".join(suffix_tokens).strip(),
                "token_syllables": source_components.syllable_counts,
                "is_multi_word": bool(source_phonetics.get("is_multi_word")),
            }

            phonetic_entries: List[Dict] = []
            module1_seed_payload: List[Dict] = []
            aggregated_seed_signatures: Set[str] = set(source_signature_set)
            delivered_words_set: Set[str] = set()
            if include_phonetic:
                slice_limit = reference_limit if reference_limit else max(limit, 1)
                phonetic_matches = cmu_candidates[:slice_limit]
                rarity_source = analyzer if analyzer is not None else getattr(self, "phonetic_analyzer", None)
                for candidate in phonetic_matches:
                    is_multi_variant = False
                    variant_candidate: Optional[str] = None
                    candidate_tokens: List[str] = []
                    candidate_syllables: Optional[int] = None
                    phrase_prefix = None
                    phrase_suffix = None
                    prefix_display = None
                    suffix_display = None
                    candidate_source_phrase = None
                    anchor_display = None

                    if isinstance(candidate, dict):
                        target = (
                            candidate.get("word")
                            or candidate.get("target")
                            or candidate.get("candidate")
                        )
                        similarity = float(
                            candidate.get("similarity", candidate.get("score", 0.0))
                        )
                        rarity_value = float(
                            candidate.get("rarity", candidate.get("rarity_score", 0.0))
                        )
                        combined = float(
                            candidate.get("combined", candidate.get("combined_score", similarity))
                        )
                        is_multi_variant = bool(candidate.get("is_multi_word"))
                        variant_candidate = (
                            candidate.get("candidate")
                            or (target if isinstance(target, str) else None)
                        )
                        candidate_tokens = candidate.get("target_tokens") or []
                        candidate_syllables = candidate.get("candidate_syllables")
                        phrase_prefix = candidate.get("prefix")
                        phrase_suffix = candidate.get("suffix")
                        prefix_display = candidate.get("prefix_display")
                        suffix_display = candidate.get("suffix_display")
                        candidate_source_phrase = candidate.get("source_phrase")
                        anchor_display = candidate.get("anchor_display")
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
                        variant_candidate = target if isinstance(target, str) else None

                    if not target:
                        continue

                    if variant_candidate is None and isinstance(target, str):
                        variant_candidate = target
                    target_text = str(target)
                    entry_is_multi = is_multi_variant or (" " in target_text.strip())
                    entry_prefix = (
                        phrase_prefix
                        if phrase_prefix is not None
                        else source_phonetic_profile.get("phrase_prefix")
                    )
                    entry_suffix = (
                        phrase_suffix
                        if phrase_suffix is not None
                        else source_phonetic_profile.get("phrase_suffix")
                    )

                    candidate_source_phrase = (
                        candidate_source_phrase if candidate_source_phrase is not None else source_word
                    )

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

                    pattern_source = candidate_source_phrase or source_word
                    pattern_separator = " // " if entry_is_multi else " / "
                    entry = {
                        "source_word": pattern_source,
                        "target_word": target_text,
                        "artist": "CMU Pronouncing Dictionary",
                        "song": "Phonetic Match",
                        "pattern": f"{pattern_source}{pattern_separator}{target_text}",
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
                        "is_multi_word": entry_is_multi,
                        "result_variant": "multi_word" if entry_is_multi else "single_word",
                        "candidate_word": variant_candidate,
                        "phrase_prefix": entry_prefix,
                        "phrase_suffix": entry_suffix,
                        "prefix_display": prefix_display or entry_prefix,
                        "suffix_display": suffix_display or entry_suffix,
                        "target_tokens": candidate_tokens,
                        "candidate_syllables": candidate_syllables,
                        "source_phrase": pattern_source,
                        "source_anchor": source_anchor_word,
                        "anchor_display": anchor_display or source_phonetic_profile.get("anchor_display"),
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
                    entry["feature_profile"] = feature_profile
                    if feature_profile and entry.get("rhyme_type") and "rhyme_type" not in feature_profile:
                        feature_profile["rhyme_type"] = entry["rhyme_type"]
                    sanitized_profile = _sanitize_feature_profile(entry)
                    if sanitized_profile:
                        syllable_span = sanitized_profile.get("syllable_span")
                        if isinstance(syllable_span, (list, tuple)) and len(syllable_span) == 2:
                            entry["syllable_span"] = [int(syllable_span[0]), int(syllable_span[1])]
                        stress_alignment = sanitized_profile.get("stress_alignment")
                        if stress_alignment is not None:
                            entry["stress_alignment"] = stress_alignment
                        internal_score = sanitized_profile.get("internal_rhyme_score")
                        if internal_score is not None:
                            entry["internal_rhyme_score"] = internal_score

                    prosody_profile = alignment.get("prosody_profile")
                    if prosody_profile:
                        entry["prosody_profile"] = prosody_profile

                    _ensure_target_phonetics(entry)

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
                            syllable_span = feature_profile.get("syllable_span")
                            if isinstance(syllable_span, (list, tuple)) and len(syllable_span) == 2:
                                entry["syllable_span"] = [int(syllable_span[0]), int(syllable_span[1])]
                            stress_alignment = feature_profile.get("stress_alignment")
                            if stress_alignment is not None:
                                entry["stress_alignment"] = stress_alignment
                            internal_score = feature_profile.get("internal_rhyme_score")
                            if internal_score is not None:
                                entry["internal_rhyme_score"] = internal_score
                        prosody_profile = alignment.get("prosody_profile")
                        if prosody_profile:
                            entry["prosody_profile"] = prosody_profile
                except Exception:
                    pass

                _sanitize_feature_profile(entry)
                _ensure_target_phonetics(entry)
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

                target_text = str(entry.get("target_word", ""))
                is_multi = " " in target_text.strip() if target_text else False
                entry.setdefault("is_multi_word", is_multi)
                entry.setdefault("result_variant", "multi_word" if is_multi else "single_word")
                if is_multi and target_text:
                    entry["pattern"] = f"{entry.get('source_word', source_word)} // {target_text}"
                entry.setdefault("source_phrase", entry.get("source_word", source_word))
                entry.setdefault("source_anchor", source_anchor_word)
                entry.setdefault("anchor_display", source_phonetic_profile.get("anchor_display"))
                entry.setdefault("phrase_prefix", source_phonetic_profile.get("phrase_prefix"))
                entry.setdefault("phrase_suffix", source_phonetic_profile.get("phrase_suffix"))

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
                    is_multi = " " in str(pattern.target_word or "").strip()
                    entry_payload["is_multi_word"] = is_multi
                    entry_payload["result_variant"] = "multi_word" if is_multi else "single_word"
                    if is_multi and pattern.target_word:
                        entry_payload["pattern"] = f"{source_word} // {pattern.target_word}"
                    entry_payload.setdefault("source_phrase", source_word)
                    entry_payload.setdefault("source_anchor", source_anchor_word)
                    entry_payload.setdefault("anchor_display", source_phonetic_profile.get("anchor_display"))
                    entry_payload.setdefault("phrase_prefix", source_phonetic_profile.get("phrase_prefix"))
                    entry_payload.setdefault("phrase_suffix", source_phonetic_profile.get("phrase_suffix"))
                    if isinstance(syllable_span, (list, tuple)) and len(syllable_span) == 2:
                        entry_payload["syllable_span"] = [int(syllable_span[0]), int(syllable_span[1])]
                    prosody_payload = getattr(pattern, "prosody_profile", None)
                    if isinstance(prosody_payload, dict):
                        entry_payload["prosody_profile"] = prosody_payload
                    entry_payload["combined_score"] = (
                        combined_float if combined_float is not None else confidence_float
                    )
                    _sanitize_feature_profile(entry_payload)
                    _ensure_target_phonetics(entry_payload)
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
                if not entry.get("source_phrase"):
                    entry["source_phrase"] = entry.get("source_word", source_word)
                entry.setdefault("source_anchor", source_anchor_word)
                entry.setdefault("anchor_display", source_phonetic_profile.get("anchor_display"))
                entry.setdefault("phrase_prefix", source_phonetic_profile.get("phrase_prefix"))
                entry.setdefault("phrase_suffix", source_phonetic_profile.get("phrase_suffix"))

                target_text = str(entry.get("target_word", ""))
                is_multi = bool(entry.get("is_multi_word"))
                if not is_multi and target_text:
                    is_multi = " " in target_text.strip()
                entry["is_multi_word"] = is_multi
                if not entry.get("result_variant"):
                    entry["result_variant"] = "multi_word" if is_multi else "single_word"

                if entry.get("pattern") and target_text:
                    connector = " // " if is_multi else " / "
                    source_value = str(entry.get("source_word", source_word))
                    entry["pattern"] = f"{source_value}{connector}{target_text}"
                elif not entry.get("pattern") and target_text:
                    connector = " // " if is_multi else " / "
                    entry["pattern"] = f"{source_word}{connector}{target_text}"
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
                _sanitize_feature_profile(entry)

                prosody_profile = entry.get("prosody_profile")
                if prosody_profile is None:
                    entry["prosody_profile"] = {}

                for field, default in {
                    "rarity_score": None,
                    "cultural_rarity": None,
                    "phonetic_sim": None,
                    "rhyme_type": None,
                    "stress_alignment": None,
                    "internal_rhyme_score": None,
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
                _ensure_target_phonetics(entry)

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

                if limit is None:
                    return processed

                try:
                    capped_limit = max(int(limit), 0)
                except (TypeError, ValueError):
                    capped_limit = len(processed)

                return processed[:capped_limit]

            return {
                "source_profile": source_phonetic_profile,
                "cmu": _process_entries(phonetic_entries if include_phonetic else [], "cmu"),
                "multi_word": _process_entries(anti_llm_entries if include_anti_llm else [], "multi_word"),
                "rap_db": _process_entries(cultural_entries if include_cultural else [], "rap_db"),
            }



    def format_rhyme_results(self, source_word: str, rhymes: Dict[str, Any]) -> str:
        """Render grouped rhyme results with shared phonetic context."""

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

        if not rhymes:
            return f"❌ No rhymes found for '{source_word}'\n\nTry words like: love, mind, flow, time, money"

        has_entries = any(rhymes.get(key) for key, _ in category_order)
        if not has_entries:
            return f"❌ No rhymes found for '{source_word}'\n\nTry words like: love, mind, flow, time, money"

        def _resolve_rhyme_type(entry: Dict[str, Any]) -> Optional[str]:
            candidate = entry.get("rhyme_type")
            if not candidate:
                for attr in ("feature_profile", "phonetic_features"):
                    obj = entry.get(attr)
                    if obj is None:
                        continue
                    if isinstance(obj, dict):
                        candidate = obj.get("rhyme_type")
                    elif hasattr(obj, "__dict__"):
                        candidate = vars(obj).get("rhyme_type")
                    else:
                        try:
                            candidate = dict(obj).get("rhyme_type")
                        except Exception:
                            candidate = None
                    if candidate:
                        break
            if not candidate:
                return None
            return str(candidate).replace("_", " ").title()

        def _as_dict(value: Any) -> Dict[str, Any]:
            if isinstance(value, dict):
                return value
            if hasattr(value, "__dict__"):
                try:
                    return dict(vars(value))
                except Exception:
                    return {}
            return {}

        def _format_float(value: Any) -> Optional[str]:
            try:
                return f"{float(value):.2f}"
            except (TypeError, ValueError):
                return None

        lines: List[str] = [
            f"🎯 **TARGET RHYMES for '{source_word.upper()}':**",
            "=" * 50,
            "",
        ]

        source_profile = rhymes.get("source_profile") or {}
        source_phonetics = source_profile.get("phonetics") or {}
        lines.append("🔎 Source profile")
        lines.append(f"   • Word: {source_profile.get('word', source_word)}")

        basic_parts: List[str] = []
        syllables = source_phonetics.get("syllables")
        if isinstance(syllables, (int, float)):
            basic_parts.append(f"Syllables: {int(syllables)}")
        stress_display = source_phonetics.get("stress_pattern_display") or source_phonetics.get("stress_pattern")
        if stress_display:
            basic_parts.append(f"Stress: {stress_display}")
        meter_hint = source_phonetics.get("meter_hint")
        foot = source_phonetics.get("metrical_foot")
        if meter_hint:
            basic_parts.append(f"Meter: {meter_hint}")
        elif foot:
            basic_parts.append(f"Foot: {str(foot).title()}")
        if basic_parts:
            lines.append(f"   • Basic: {' | '.join(basic_parts)}")

        signature_values = source_profile.get("signatures") or []
        if signature_values:
            lines.append(f"   • Signatures: {', '.join(signature_values)}")

        similarity_parts: List[str] = []
        reference_similarity = _format_float(source_profile.get("reference_similarity"))
        threshold_value = _format_float(source_profile.get("threshold"))
        if reference_similarity:
            similarity_parts.append(f"Reference sim: {reference_similarity}")
        if threshold_value:
            similarity_parts.append(f"Threshold: {threshold_value}")
        if similarity_parts:
            lines.append(f"   • Similarity band: {' | '.join(similarity_parts)}")

        expanded_parts: List[str] = []
        vowel_skeleton = source_phonetics.get("vowel_skeleton")
        if vowel_skeleton:
            expanded_parts.append(f"Vowels: {vowel_skeleton}")
        consonant_tail = source_phonetics.get("consonant_tail")
        if consonant_tail:
            expanded_parts.append(f"Tail: {consonant_tail}")
        pronunciations = source_phonetics.get("pronunciations") or []
        if pronunciations:
            lines.append(f"   • Pronunciations: {', '.join(pronunciations)}")
        if expanded_parts:
            lines.append(f"   • Expanded: {' | '.join(expanded_parts)}")
        lines.append("")

        for key, header in category_order:
            entries = rhymes.get(key) or []
            if not entries:
                continue

            lines.append(header)
            lines.append("-" * len(header))

            for entry in entries:
                target_word = entry.get("target_word") or "(unknown)"
                variant_note = " (phrase)" if entry.get("is_multi_word") else ""
                lines.append(f"• **{str(target_word).upper()}**{variant_note}")

                target_phonetics = entry.get("target_phonetics") or {}
                phonetic_bits: List[str] = []
                target_syllables = target_phonetics.get("syllables")
                if isinstance(target_syllables, (int, float)):
                    phonetic_bits.append(f"Syllables: {int(target_syllables)}")
                target_stress = target_phonetics.get("stress_pattern_display") or target_phonetics.get("stress_pattern")
                if target_stress:
                    phonetic_bits.append(f"Stress: {target_stress}")
                target_meter = target_phonetics.get("meter_hint")
                target_foot = target_phonetics.get("metrical_foot")
                if target_meter:
                    phonetic_bits.append(f"Meter: {target_meter}")
                elif target_foot:
                    phonetic_bits.append(f"Foot: {str(target_foot).title()}")
                syllable_span = entry.get("syllable_span")
                if isinstance(syllable_span, (list, tuple)) and len(syllable_span) == 2:
                    try:
                        phonetic_bits.append(f"Span: {int(syllable_span[0])}→{int(syllable_span[1])}")
                    except Exception:
                        pass
                if phonetic_bits:
                    lines.append(f"   • Phonetics: {' | '.join(phonetic_bits)}")

                pronunciation_variants = target_phonetics.get("pronunciations") or []
                if pronunciation_variants:
                    lines.append(f"   • Phonemes: {', '.join(pronunciation_variants)}")

                score_parts: List[str] = []
                similarity_value = _format_float(entry.get("phonetic_sim"))
                if similarity_value:
                    score_parts.append(f"Similarity: {similarity_value}")
                confidence_value = entry.get("combined_score")
                if confidence_value is None:
                    confidence_value = entry.get("confidence")
                confidence_formatted = _format_float(confidence_value)
                if confidence_formatted:
                    score_parts.append(f"Confidence: {confidence_formatted}")
                rarity_value = entry.get("rarity_score")
                if rarity_value is None:
                    rarity_value = entry.get("cultural_rarity")
                rarity_formatted = _format_float(rarity_value)
                if rarity_formatted:
                    score_parts.append(f"Rarity: {rarity_formatted}")
                if score_parts:
                    lines.append(f"   • Scores: {' | '.join(score_parts)}")

                rhyme_label = _resolve_rhyme_type(entry)
                if rhyme_label:
                    lines.append(f"   • Rhyme type: {rhyme_label}")

                normalized_source = _normalize_source_key(entry.get("result_source")) or key
                if normalized_source == "cmu":
                    lines.append("   • Source: CMU Pronouncing Dictionary")
                elif normalized_source == "multi_word":
                    weakness = entry.get("llm_weakness_type")
                    if weakness:
                        lines.append(f"   • LLM weakness: {str(weakness).replace('_', ' ').title()}")
                    cultural_depth = entry.get("cultural_depth")
                    if cultural_depth:
                        lines.append(f"   • Cultural depth: {cultural_depth}")
                else:
                    credits: List[str] = []
                    artist = entry.get("artist")
                    song = entry.get("song")
                    if artist:
                        credits.append(str(artist))
                    if song:
                        credits.append(str(song))
                    if credits:
                        lines.append(f"   • Credits: {' — '.join(credits)}")
                    genre_value = entry.get("genre")
                    if genre_value:
                        lines.append(f"   • Genre: {genre_value}")

                    context_info = _as_dict(entry.get("cultural_context"))
                    cultural_bits: List[str] = []
                    for key_name, label in (
                        ("era", "Era"),
                        ("regional_origin", "Region"),
                        ("cultural_significance", "Significance"),
                    ):
                        value = context_info.get(key_name)
                        if not value and key_name == "cultural_significance":
                            value = entry.get("cultural_sig")
                        if value:
                            cultural_bits.append(f"{label}: {str(value).replace('_', ' ').title()}")
                    if cultural_bits:
                        lines.append(f"   • Cultural: {' | '.join(cultural_bits)}")
                    styles = context_info.get("style_characteristics")
                    if isinstance(styles, (list, tuple)) and styles:
                        formatted_styles = ", ".join(
                            str(style).replace('_', ' ').title() for style in styles if style
                        )
                        if formatted_styles:
                            lines.append(f"   • Styles: {formatted_styles}")

                pattern_text = entry.get("pattern")
                if pattern_text:
                    lines.append(f"   • Pattern: {pattern_text}")

                source_context = entry.get("source_context")
                target_context = entry.get("target_context")
                context_parts: List[str] = []
                if source_context:
                    context_parts.append(str(source_context))
                if target_context:
                    context_parts.append(str(target_context))
                if context_parts:
                    lines.append(f"   • Context: {' | '.join(context_parts)}")

                prosody = entry.get("prosody_profile")
                if isinstance(prosody, dict):
                    cadence = prosody.get("complexity_tag")
                    if cadence:
                        lines.append(f"   • Cadence: {str(cadence).replace('_', ' ').title()}")

                lines.append("")

            lines.append("")

        return "\n".join(lines).strip() + "\n"

    def create_gradio_interface(self):
        """Create the Gradio interface for Hugging Face Spaces"""

        def search_interface(
            word,
            max_results,
            min_conf,
            cultural_filter,
            genre_filter,
            rhyme_type_filter,
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

            rhymes = self.search_rhymes(
                word,
                limit=max_results,
                min_confidence=min_conf,
                cultural_significance=ensure_list(cultural_filter),
                genres=ensure_list(genre_filter),
                allowed_rhyme_types=ensure_list(rhyme_type_filter),
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

        # Create Gradio interface
        with gr.Blocks(title="RhymeRarity - Advanced Rhyme Generator", theme=gr.themes.Soft()) as interface:
            gr.Markdown(
                "## 🎵 RhymeRarity - Advanced Rhyme Generator\n"
                "Discover culturally grounded rhyme pairs sourced from authentic hip-hop lyrics."
            )

            with gr.Row(equal_height=True):
                with gr.Column(scale=5, min_width=360):
                    word_input = gr.Textbox(
                        label="Word to Find Rhymes For",
                        placeholder="Enter a word (e.g., love, mind, flow, money)",
                        lines=1,
                    )

                    with gr.Accordion("Advanced filters", open=False):
                        with gr.Row():
                            max_results = gr.Slider(
                                minimum=5,
                                maximum=50,
                                value=15,
                                step=1,
                                label="Max Results",
                            )

                            min_confidence = gr.Slider(
                                minimum=0.5,
                                maximum=1.0,
                                value=0.7,
                                step=0.05,
                                label="Min Confidence",
                            )

                        with gr.Row():
                            with gr.Column(scale=1, min_width=160):
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

                            with gr.Column(scale=1, min_width=160):
                                rhyme_type_dropdown = gr.Dropdown(
                                    choices=["perfect", "near", "slant", "eye", "weak"],
                                    multiselect=True,
                                    label="Rhyme Type",
                                    info="Limit to specific rhyme categories",
                                    value=[],
                                )

                    search_btn = gr.Button("🔍 Find Rhymes", variant="primary", size="lg")
                    gr.Markdown(
                        "💡 Enter a word and adjust the filters, then press **Find Rhymes** to discover new lyric pairings."
                    )

                with gr.Column(scale=5, min_width=360):
                    gr.Markdown("### Rhyme Results")
                    output = gr.Markdown(
                        value="Start by entering a word on the left and click **Find Rhymes**.",
                    )

            search_btn.click(
                fn=search_interface,
                inputs=[
                    word_input,
                    max_results,
                    min_confidence,
                    cultural_dropdown,
                    genre_dropdown,
                    rhyme_type_dropdown,
                ],
                outputs=output,
            )

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
