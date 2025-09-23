import os
import shutil
import sqlite3
import sys
import types
from pathlib import Path

import pytest


gradio_stub = types.ModuleType("gradio")
gradio_stub.themes = types.SimpleNamespace(Soft=lambda *_, **__: None)
for attr in (
    "Blocks",
    "Markdown",
    "Row",
    "Column",
    "Textbox",
    "Slider",
    "Button",
    "Examples",
    "Dropdown",
    "CheckboxGroup",
):
    setattr(gradio_stub, attr, lambda *_, **__: None)
sys.modules.setdefault("gradio", gradio_stub)

pandas_stub = types.ModuleType("pandas")
sys.modules.setdefault("pandas", pandas_stub)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import RhymeRarityApp
from module2_enhanced_anti_llm import AntiLLMPattern


if os.path.exists("patterns.db"):
    os.remove("patterns.db")

if os.path.isdir("__pycache__"):
    shutil.rmtree("__pycache__")


def create_test_database(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE song_rhyme_patterns (
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
        """
    )

    rows = [
        (
            1,
            "love / above",
            "love",
            "above",
            "Artist A",
            "Song A",
            "hip-hop",
            1,
            0.95,
            0.97,
            "mainstream",
            "Love in the verse",
            "Above in the hook",
        ),
        (
            2,
            "above / love",
            "above",
            "love",
            "Artist B",
            "Song B",
            "hip-hop",
            1,
            0.80,
            0.85,
            "mainstream",
            "Above in the verse",
            "Love in the hook",
        ),
        (
            3,
            "love / glove",
            "love",
            "glove",
            "Artist C",
            "Song C",
            "soul",
            1,
            0.88,
            0.91,
            "legendary",
            "Love in the bridge",
            "Glove in the chorus",
        ),
        (
            4,
            "love / shove",
            "love",
            "shove",
            "Artist D",
            "Song D",
            "hip-hop",
            3,
            0.70,
            0.80,
            "underground",
            "Love in the intro",
            "Shove in the outro",
        ),
    ]

    cursor.executemany(
        """
        INSERT INTO song_rhyme_patterns (
            id,
            pattern,
            source_word,
            target_word,
            artist,
            song_title,
            genre,
            line_distance,
            confidence_score,
            phonetic_similarity,
            cultural_significance,
            source_context,
            target_context
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    conn.close()


def test_search_rhymes_returns_counterpart_for_target_word(tmp_path):
    db_path = tmp_path / "patterns.db"
    create_test_database(str(db_path))

    app = RhymeRarityApp(db_path=str(db_path))

    results = app.search_rhymes("above", limit=5, min_confidence=0.0)

    assert results, "Expected at least one rhyme suggestion"
    assert results[0]["target_word"] == "love"
    assert results[0]["source_context"] == "Above in the hook"
    assert results[0]["target_context"] == "Love in the verse"
    assert all(result["target_word"] == "love" for result in results)


def test_search_rhymes_filters_by_cultural_significance(tmp_path):
    db_path = tmp_path / "patterns.db"
    create_test_database(str(db_path))

    app = RhymeRarityApp(db_path=str(db_path))

    results = app.search_rhymes(
        "love",
        limit=10,
        min_confidence=0.0,
        cultural_significance=["legendary"],
        result_sources=["cultural"],
    )

    assert results, "Expected filtered results for cultural significance"
    assert all(result["cultural_sig"] == "legendary" for result in results)
    assert all(result["target_word"] == "glove" for result in results)


def test_cultural_context_enrichment_in_formatting(tmp_path):
    db_path = tmp_path / "patterns.db"
    create_test_database(str(db_path))

    app = RhymeRarityApp(db_path=str(db_path))

    class MockCulturalEngine:
        def __init__(self):
            self.calls = []
            self.rarity_calls = []

        def get_cultural_context(self, pattern_data):
            self.calls.append(pattern_data)
            return types.SimpleNamespace(
                artist=pattern_data.get("artist", ""),
                song=pattern_data.get("song", ""),
                genre="hip-hop",
                era="golden_age",
                cultural_significance="legendary",
                regional_origin="queens",
                style_characteristics=["multi_syllable", "storytelling"],
            )

        def get_cultural_rarity_score(self, context):
            self.rarity_calls.append(context)
            return 3.5

    app.cultural_engine = MockCulturalEngine()

    results = app.search_rhymes(
        "love",
        limit=5,
        min_confidence=0.0,
        result_sources=["cultural"],
    )

    assert results, "Expected results enriched with cultural context"
    enriched_entry = results[0]
    assert "cultural_context" in enriched_entry
    assert enriched_entry["cultural_rarity"] == 3.5

    formatted = app.format_rhyme_results("love", results)

    assert "Era: Golden Age" in formatted
    assert "Region: Queens" in formatted
    assert "Rarity Score: 3.50" in formatted
    assert "Styles: Multi Syllable, Storytelling" in formatted


def test_rhyme_type_and_rhythm_filters(tmp_path):
    db_path = tmp_path / "patterns.db"
    create_test_database(str(db_path))

    app = RhymeRarityApp(db_path=str(db_path))

    class MockCMULoader:
        def get_rhyme_parts(self, _word):
            return {"sig"}

        def get_rhyming_words(self, _word):
            return []

    class MockAnalyzer:
        def __init__(self):
            self.cmu_loader = MockCMULoader()

        def get_phonetic_similarity(self, _source, target):
            return {
                "glove": 0.92,
                "above": 0.83,
                "shove": 0.72,
            }.get(target, 0.7)

        def get_rarity_score(self, _word):
            return 0.5

        def combine_similarity_and_rarity(self, similarity, rarity):
            return (float(similarity) + float(rarity)) / 2

        def update_rarity_from_database(self, _path):
            return True

        def analyze_rhyme_pattern(self, source, target):
            similarity = self.get_phonetic_similarity(source, target)
            feature_map = {
                "glove": {
                    "ending_similarity": 0.85,
                    "vowel_similarity": 0.8,
                    "consonant_similarity": 0.82,
                    "syllable_similarity": 1.0,
                },
                "above": {
                    "ending_similarity": 0.75,
                    "vowel_similarity": 0.7,
                    "consonant_similarity": 0.68,
                    "syllable_similarity": 0.5,
                },
                "shove": {
                    "ending_similarity": 0.65,
                    "vowel_similarity": 0.6,
                    "consonant_similarity": 0.58,
                    "syllable_similarity": 0.4,
                },
            }
            features = feature_map.get(target, feature_map["shove"])
            rhyme_type = "perfect" if target == "glove" else "slant"
            return types.SimpleNamespace(
                similarity_score=similarity,
                phonetic_features=features,
                rhyme_type=rhyme_type,
            )

        def classify_rhyme_type(self, _source, target, similarity):
            return "perfect" if float(similarity) >= 0.9 else "slant"

    class MockCulturalEngine:
        cultural_categories = {}

        def __init__(self, analyzer):
            self.phonetic_analyzer = analyzer

        def set_phonetic_analyzer(self, analyzer):
            self.phonetic_analyzer = analyzer

        def derive_rhyme_signatures(self, _word):
            return {"sig"}

        def evaluate_rhyme_alignment(self, source_word, target_word, threshold=None, rhyme_signatures=None):
            if threshold is not None:
                similarity = self.phonetic_analyzer.get_phonetic_similarity(source_word, target_word)
                if similarity < threshold:
                    return None
            match = self.phonetic_analyzer.analyze_rhyme_pattern(source_word, target_word)
            return {
                "similarity": match.similarity_score,
                "rarity": 0.5,
                "combined": (match.similarity_score + 0.5) / 2,
                "rhyme_type": match.rhyme_type,
                "signature_matches": ["sig"],
                "target_signatures": ["sig"],
                "features": dict(match.phonetic_features),
            }

        def get_cultural_context(self, _pattern):
            return None

        def get_cultural_rarity_score(self, _context):
            return 0.5

    mock_analyzer = MockAnalyzer()
    app.phonetic_analyzer = mock_analyzer
    app.cmu_loader = mock_analyzer.cmu_loader
    app.cultural_engine = MockCulturalEngine(mock_analyzer)

    results = app.search_rhymes(
        "love",
        limit=10,
        min_confidence=0.0,
        result_sources=["cultural"],
        rhyme_types=["perfect"],
        min_syllable_similarity=0.9,
        min_rhythm_strength=0.75,
    )

    assert results, "Expected filtered results for rhyme type and rhythm thresholds"
    assert all(result["rhyme_type"] == "perfect" for result in results)
    assert all(result.get("syllable_similarity", 0) >= 0.9 for result in results)
    assert all(result.get("rhythm_strength", 0) >= 0.75 for result in results)


def test_anti_llm_patterns_in_formatting(tmp_path):
    db_path = tmp_path / "patterns.db"
    create_test_database(str(db_path))

    app = RhymeRarityApp(db_path=str(db_path))

    sentinel_pattern = AntiLLMPattern(
        source_word="love",
        target_word="shove",
        rarity_score=4.2,
        cultural_depth="Sentinel Depth",
        llm_weakness_type="rare_word_combinations",
        confidence=0.73,
    )

    class StubAntiLLMEngine:
        def generate_anti_llm_patterns(self, word, limit=20):
            return [sentinel_pattern]

    app.anti_llm_engine = StubAntiLLMEngine()

    results = app.search_rhymes(
        "love",
        limit=5,
        min_confidence=0.0,
        result_sources=["anti-llm"],
    )

    assert results, "Expected anti-LLM results when engine is patched"
    anti_entry = results[0]
    assert anti_entry["result_source"] in {"anti_llm", "anti-llm"}
    assert anti_entry["rarity_score"] == pytest.approx(4.2)

    formatted = app.format_rhyme_results("love", results)

    assert "SHOVE" in formatted
    assert "Rarity Score: 4.20" in formatted
    assert "LLM Weakness: Rare Word Combinations" in formatted
    assert "Cultural Depth: Sentinel Depth" in formatted

