import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cultural.engine import CulturalIntelligenceEngine
from rhyme_rarity.utils.syllables import estimate_syllable_count


def test_get_cultural_context_handles_none_artist(tmp_path):
    engine = CulturalIntelligenceEngine(db_path=str(tmp_path / "patterns.db"))

    pattern_data = {
        "artist": None,
        "song": "Test Song",
    }

    context = engine.get_cultural_context(pattern_data)

    assert context.artist == ""
    assert context.song == "Test Song"


class DummyAnalyzer:
    def __init__(self, syllable_value: int) -> None:
        self.syllable_value = syllable_value

    def estimate_syllables(self, word: str) -> int:  # pragma: no cover - simple pass-through
        return self.syllable_value


class ExplodingAnalyzer:
    def estimate_syllables(self, word: str) -> int:
        raise RuntimeError("boom")


def test_cultural_engine_estimate_syllables_prefers_attached_analyzer(tmp_path):
    engine = CulturalIntelligenceEngine(
        db_path=str(tmp_path / "patterns.db"),
        phonetic_analyzer=DummyAnalyzer(7),
    )

    assert engine._estimate_syllables("anything") == 7


def test_cultural_engine_estimate_syllables_falls_back_to_helper(tmp_path):
    engine = CulturalIntelligenceEngine(db_path=str(tmp_path / "patterns.db"))

    assert engine._estimate_syllables("flow") == estimate_syllable_count("flow")
    assert engine._estimate_syllables("") == 0


def test_cultural_engine_estimate_syllables_handles_analyzer_error(tmp_path):
    engine = CulturalIntelligenceEngine(
        db_path=str(tmp_path / "patterns.db"),
        phonetic_analyzer=ExplodingAnalyzer(),
    )

    assert engine._estimate_syllables("flow") == estimate_syllable_count("flow")
