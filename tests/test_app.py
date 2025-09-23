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

import app as app_module
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
        def generate_anti_llm_patterns(
            self,
            word,
            limit=20,
            module1_seeds=None,
            seed_signatures=None,
            delivered_words=None,
        ):
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


def test_min_confidence_filters_phonetic_candidates(monkeypatch, tmp_path):
    db_path = tmp_path / "patterns.db"
    create_test_database(str(db_path))

    app = RhymeRarityApp(db_path=str(db_path))
    app.cultural_engine = None

    def stub_cmu_rhymes(word, limit=20, analyzer=None, cmu_loader=None):
        return [
            {"word": "alpha", "similarity": 0.94, "combined": 0.94, "rarity": 0.8},
            {"word": "beta", "similarity": 0.82, "combined": 0.6, "rarity": 0.4},
        ]

    monkeypatch.setattr(app_module, "get_cmu_rhymes", stub_cmu_rhymes)

    results = app.search_rhymes(
        "love",
        limit=5,
        min_confidence=0.9,
        result_sources=["phonetic"],
    )

    assert results, "Expected high-confidence phonetic suggestions"
    targets = {entry["target_word"] for entry in results}
    assert "alpha" in targets
    assert "beta" not in targets
    assert all(
        float(entry.get("combined_score", entry.get("confidence", 0.0))) >= 0.9
        for entry in results
    )


def test_min_confidence_filters_anti_llm_candidates(tmp_path):
    db_path = tmp_path / "patterns.db"
    create_test_database(str(db_path))

    app = RhymeRarityApp(db_path=str(db_path))
    app.cultural_engine = None

    low_conf_pattern = AntiLLMPattern(
        source_word="love",
        target_word="gamma",
        rarity_score=2.5,
        cultural_depth="Low",
        llm_weakness_type="rare_word_combinations",
        confidence=0.5,
    )
    high_conf_pattern = AntiLLMPattern(
        source_word="love",
        target_word="delta",
        rarity_score=3.1,
        cultural_depth="High",
        llm_weakness_type="rare_word_combinations",
        confidence=0.95,
    )

    class StubAntiLLMEngine:
        def generate_anti_llm_patterns(
            self,
            word,
            limit=20,
            module1_seeds=None,
            seed_signatures=None,
            delivered_words=None,
        ):
            return [low_conf_pattern, high_conf_pattern]

    app.anti_llm_engine = StubAntiLLMEngine()

    results = app.search_rhymes(
        "love",
        limit=5,
        min_confidence=0.9,
        result_sources=["anti-llm"],
    )

    assert results, "Expected at least one high-confidence anti-LLM suggestion"
    targets = {entry["target_word"] for entry in results}
    assert "delta" in targets
    assert "gamma" not in targets
    assert all(
        float(entry.get("combined_score", entry.get("confidence", 0.0))) >= 0.9
        for entry in results
    )


def test_search_rhymes_respects_rhyme_type_and_cadence_filters(tmp_path):
    db_path = tmp_path / "patterns.db"
    create_test_database(str(db_path))

    app = RhymeRarityApp(db_path=str(db_path))

    class MockCulturalEngine:
        cultural_categories = {"legendary": {}, "mainstream": {}}

        def __init__(self):
            self.phonetic_analyzer = None

        def derive_rhyme_signatures(self, word):
            if not word:
                return set()
            cleaned = str(word).strip().lower()
            if not cleaned:
                return set()
            return {f"sig:{cleaned[-2:]}"}

        def get_cultural_context(self, pattern_data):
            return {"style_characteristics": ["storytelling"]}

        def get_cultural_rarity_score(self, _context):
            return 2.0

        def evaluate_rhyme_alignment(
            self,
            source_word,
            target_word,
            threshold=None,
            rhyme_signatures=None,
            source_context=None,
            target_context=None,
        ):
            target = (target_word or "").lower()
            if not target:
                return None

            if target in {"love", "glove"}:
                rhyme_type = "perfect"
                complexity = "steady"
                cadence_ratio = 1.05
            else:
                rhyme_type = "slant"
                complexity = "dense"
                cadence_ratio = 1.4

            target_signatures = list(self.derive_rhyme_signatures(target))
            signature_matches = list(rhyme_signatures or [])

            return {
                "similarity": 0.9,
                "rarity": 0.6,
                "combined": 0.75,
                "rhyme_type": rhyme_type,
                "signature_matches": signature_matches,
                "target_signatures": target_signatures,
                "features": {"rhyme_type": rhyme_type},
                "feature_profile": {"bradley_device": "multi"},
                "prosody_profile": {
                    "complexity_tag": complexity,
                    "cadence_ratio": cadence_ratio,
                },
            }

    app.cultural_engine = MockCulturalEngine()

    filtered_results = app.search_rhymes(
        "love",
        limit=5,
        min_confidence=0.0,
        result_sources=["cultural"],
        allowed_rhyme_types=["perfect"],
        cadence_focus="steady",
    )

    assert filtered_results, "Expected matches that satisfy rhyme type and cadence filters"
    assert all(entry.get("rhyme_type") == "perfect" for entry in filtered_results)
    assert all(
        (entry.get("prosody_profile") or {}).get("complexity_tag") == "steady"
        for entry in filtered_results
    )

    formatted = app.format_rhyme_results("love", filtered_results)
    assert "Rhyme Type: Perfect" in formatted
    assert "Cadence: Steady" in formatted

