import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anti_llm import AntiLLMRhymeEngine


class DummyProfileAnalyzer:
    """Analyzer stub that returns deterministic profile data."""

    def __init__(self) -> None:
        self.calls = []

    def derive_rhyme_profile(self, source: str, target: str):
        self.calls.append((source, target))
        return {
            "bradley_device": "slant",
            "syllable_span": (1, 3),
            "stress_alignment": 0.75,
            "assonance_score": 0.5,
            "consonance_score": 0.25,
            "internal_rhyme_score": 0.4,
        }


@pytest.fixture
def engine_with_dummy_analyzer():
    """Engine instance that uses the dummy analyzer for profile generation."""

    analyzer = DummyProfileAnalyzer()
    engine = AntiLLMRhymeEngine(phonetic_analyzer=analyzer)
    return engine
