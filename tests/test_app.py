import os
import shutil
import sqlite3
import sys
import types
from pathlib import Path


gradio_stub = types.ModuleType("gradio")
gradio_stub.themes = types.SimpleNamespace(Soft=lambda *_, **__: None)
for attr in ("Blocks", "Markdown", "Row", "Column", "Textbox", "Slider", "Button", "Examples"):
    setattr(gradio_stub, attr, lambda *_, **__: None)
sys.modules.setdefault("gradio", gradio_stub)

pandas_stub = types.ModuleType("pandas")
sys.modules.setdefault("pandas", pandas_stub)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import RhymeRarityApp


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

