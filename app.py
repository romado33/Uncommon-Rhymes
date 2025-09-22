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
from typing import List, Dict, Tuple, Optional
import json
import random

# Import our custom modules
try:
    from module1_enhanced_core_phonetic import (
        EnhancedPhoneticAnalyzer,
        get_cmu_rhymes,
    )
    from module2_enhanced_anti_llm import AntiLLMRhymeEngine
    from module3_enhanced_cultural_database import CulturalIntelligenceEngine
    print("✅ All RhymeRarity modules loaded successfully")
except ImportError as e:
    print(f"⚠️ Module import warning: {e}")
    # Create fallback classes for demo
    class EnhancedPhoneticAnalyzer:
        def __init__(self): pass
        def get_phonetic_similarity(self, word1, word2): return 0.85

    def get_cmu_rhymes(word, limit=20, analyzer=None):
        return []
    
    class AntiLLMRhymeEngine:
        def __init__(self): pass
        def generate_anti_llm_patterns(self, word): return []
    
    class CulturalIntelligenceEngine:
        def __init__(self):
            pass

        def get_cultural_context(self, pattern):
            return {"significance": "mainstream"}

        def get_cultural_rarity_score(self, context):
            return 1.0

class RhymeRarityApp:
    """Production Hugging Face app for RhymeRarity rhyme search"""
    
    def __init__(self, db_path: str = "patterns.db"):
        self.db_path = db_path
        self.phonetic_analyzer = EnhancedPhoneticAnalyzer()
        self.anti_llm_engine = AntiLLMRhymeEngine(db_path=self.db_path)
        self.cultural_engine = CulturalIntelligenceEngine(db_path=self.db_path)
        
        # Initialize database
        self.check_database()
        
        print(f"🎵 RhymeRarity App initialized with database: {db_path}")
        
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
    
    def search_rhymes(
        self,
        source_word: str,
        limit: int = 20,
        min_confidence: float = 0.7,
        cultural_significance: Optional[List[str]] = None,
        genres: Optional[List[str]] = None,
        result_sources: Optional[List[str]] = None,
        max_line_distance: Optional[int] = None,
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

        selected_sources = {
            _normalize_source_name(source) for source in _normalize_to_list(result_sources)
        }
        if not selected_sources:
            selected_sources = {"phonetic", "cultural", "anti-llm"}

        include_phonetic = "phonetic" in selected_sources
        include_cultural = "cultural" in selected_sources
        include_anti_llm = "anti-llm" in selected_sources

        filters_active = bool(
            cultural_filters or genre_filters or (max_line_distance is not None)
        )

        if limit <= 0:
            return []

        try:
            phonetic_entries: List[Dict] = []
            if include_phonetic and not filters_active:
                phonetic_matches = get_cmu_rhymes(
                    source_word,
                    limit=limit,
                    analyzer=getattr(self, "phonetic_analyzer", None),
                )

                for target, score in phonetic_matches:
                    phonetic_entries.append({
                        'source_word': source_word,
                        'target_word': target,
                        'artist': 'CMU Pronouncing Dictionary',
                        'song': 'Phonetic Match',
                        'pattern': f"{source_word} / {target}",
                        'distance': None,
                        'confidence': score,
                        'phonetic_sim': score,
                        'cultural_sig': 'phonetic',
                        'genre': None,
                        'source_context': 'Phonetic match suggested by the CMU Pronouncing Dictionary.',
                        'target_context': '',
                        'result_source': 'phonetic',
                    })

                phonetic_entries.sort(key=lambda entry: entry['phonetic_sim'], reverse=True)

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

            def _enrich_with_cultural_context(entry: Dict) -> Dict:
                engine = getattr(self, "cultural_engine", None)
                if not engine:
                    return entry

                context_data = None
                rarity_value = None

                try:
                    get_context = getattr(engine, "get_cultural_context", None)
                    if callable(get_context):
                        pattern_payload = {
                            'artist': entry.get('artist'),
                            'song': entry.get('song'),
                            'source_word': entry.get('source_word', source_word),
                            'target_word': entry.get('target_word'),
                            'pattern': entry.get('pattern'),
                            'cultural_significance': entry.get('cultural_sig'),
                        }
                        context_data = get_context(pattern_payload)
                    elif hasattr(engine, "find_cultural_patterns"):
                        finder = getattr(engine, "find_cultural_patterns")
                        if callable(finder):
                            patterns = finder(entry.get('source_word', source_word), limit=1)
                            if patterns:
                                pattern_info = patterns[0]
                                context_data = pattern_info.get('cultural_context')
                                rarity_value = pattern_info.get('cultural_rarity')

                    context_dict = _context_to_dict(context_data)
                    if context_dict:
                        entry['cultural_context'] = context_dict

                        rarity_fn = getattr(engine, "get_cultural_rarity_score", None)
                        if callable(rarity_fn):
                            try:
                                rarity_value = rarity_fn(context_data)
                            except (TypeError, AttributeError):
                                rarity_value = rarity_fn(context_dict)

                    if rarity_value is not None:
                        entry['cultural_rarity'] = rarity_value

                except Exception:
                    # Cultural enrichment is optional; ignore errors to avoid breaking search
                    pass

                return entry

            def build_query(word_column: str) -> Tuple[str, List]:
                conditions = [f"{word_column} = ?", "confidence_score >= ?", "source_word != target_word"]
                params: List = [source_word, min_confidence]

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

            def build_entry(row: Tuple, swap: bool = False) -> Dict:
                entry = {
                    'source_word': row[0],
                    'target_word': row[1],
                    'artist': row[2],
                    'song': row[3],
                    'pattern': row[4],
                    'genre': row[5],
                    'distance': row[6],
                    'confidence': row[7],
                    'phonetic_sim': row[8],
                    'cultural_sig': row[9],
                    'source_context': row[10],
                    'target_context': row[11],
                    'result_source': 'cultural',
                }

                if swap:
                    entry = {
                        **entry,
                        'source_word': row[1],
                        'target_word': row[0],
                        'source_context': row[11],
                        'target_context': row[10],
                    }

                return _enrich_with_cultural_context(entry)

            for row in source_results:
                cultural_entries.append(build_entry(row))

            for row in target_results:
                cultural_entries.append(build_entry(row, swap=True))

            cultural_entries.sort(key=lambda r: (-r['confidence'], -r['phonetic_sim']))

            anti_llm_entries: List[Dict] = []
            if include_anti_llm and not filters_active:
                anti_patterns = self.anti_llm_engine.generate_anti_llm_patterns(
                    source_word,
                    limit=limit,
                )
                for pattern in anti_patterns:
                    anti_llm_entries.append({
                        'source_word': source_word,
                        'target_word': pattern.target_word,
                        'pattern': f"{source_word} / {pattern.target_word}",
                        'confidence': pattern.confidence,
                        'rarity_score': pattern.rarity_score,
                        'llm_weakness_type': pattern.llm_weakness_type,
                        'cultural_depth': pattern.cultural_depth,
                        'result_source': 'anti_llm',
                    })

            combined_results = phonetic_entries + cultural_entries + anti_llm_entries

            deduped: List[Dict] = []
            seen_pairs = set()
            for entry in combined_results:
                pair = (
                    entry.get('source_word', source_word).lower(),
                    entry['target_word'].lower(),
                )
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                deduped.append(entry)

            filtered_entries: List[Dict] = []
            if filters_active:
                for entry in deduped:
                    if cultural_filters:
                        sig = entry.get('cultural_sig')
                        if sig is None or _normalize_source_name(str(sig)) not in cultural_filters:
                            continue
                    if genre_filters:
                        genre_value = entry.get('genre')
                        if genre_value is None or _normalize_source_name(str(genre_value)) not in genre_filters:
                            continue
                    if max_line_distance is not None:
                        distance_value = entry.get('distance')
                        if distance_value is None or distance_value > max_line_distance:
                            continue
                    filtered_entries.append(entry)
            else:
                filtered_entries = deduped

            for entry in filtered_entries:
                entry.pop('source_word', None)

            return filtered_entries[:limit]

        except sqlite3.Error as e:
            print(f"Database error: {e}")
            return []
    
    def format_rhyme_results(self, source_word: str, rhymes: List[Dict]) -> str:
        """Format rhyme search results for display"""
        if not rhymes:
            return f"❌ No rhymes found for '{source_word}'\n\nTry words like: love, mind, flow, time, money"
        
        result = f"🎯 **TARGET RHYMES for '{source_word.upper()}':**\n"
        result += "=" * 50 + "\n\n"

        for i, rhyme in enumerate(rhymes[:15], 1):
            result += f"**{i:2d}. {rhyme['target_word'].upper()}**\n"
            source_type = rhyme.get('result_source')
            if source_type == 'phonetic':
                origin = rhyme.get('artist', 'CMU Pronouncing Dictionary')
                result += f"   🏷️ Origin: 📚 {origin}\n"
                result += f"   📝 Pattern: {rhyme['pattern']}\n"
                result += f"   📊 Phonetic Score: {rhyme['phonetic_sim']:.2f}\n"
                note = rhyme.get('source_context') or 'Pronunciation-based suggestion.'
                result += f"   🗒️ Note: {note}\n\n"
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

            rhymes = self.search_rhymes(
                word,
                limit=max_results,
                min_confidence=min_conf,
                cultural_significance=ensure_list(cultural_filter),
                genres=ensure_list(genre_filter),
                result_sources=ensure_list(source_filter),
                max_line_distance=max_distance,
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
                    ["love", 15, 0.8, [], [], default_sources, "Any"],
                    ["mind", 15, 0.8, [], [], default_sources, "Any"],
                    ["flow", 15, 0.8, [], [], default_sources, "Any"],
                    ["money", 15, 0.8, [], [], default_sources, "Any"],
                    ["time", 15, 0.8, [], [], default_sources, "Any"]
                ],
                inputs=[
                    word_input,
                    max_results,
                    min_confidence,
                    cultural_dropdown,
                    genre_dropdown,
                    result_source_group,
                    max_line_distance_dropdown,
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
