import os
import re
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

import rhyme_rarity.app.services.search_service as search_service_module
from rhyme_rarity.app.app import RhymeRarityApp
from anti_llm import AntiLLMPattern
from rhyme_rarity.core.cmudict_loader import VOWEL_PHONEMES


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
        (
            5,
            "paper trail / major fail",
            "paper trail",
            "major fail",
            "Artist E",
            "Song E",
            "hip-hop",
            1,
            0.88,
            0.90,
            "legendary",
            "Paper trail in the verse",
            "Major fail in the hook",
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

    results = app.search_rhymes("glove", limit=5, min_confidence=0.0)

    assert "source_profile" in results
    assert isinstance(results["source_profile"], dict)

    rap_results = results["rap_db"]
    assert rap_results, "Expected at least one rhyme suggestion"
    first = rap_results[0]
    assert first["target_word"] == "love"
    assert first["pattern"] == "glove / love"
    assert first["source_context"] == "Glove in the chorus"
    assert first["target_context"] == "Love in the bridge"
    assert all(result["target_word"] == "love" for result in rap_results)


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

    assert "source_profile" in results
    assert isinstance(results["source_profile"], dict)
    rap_results = results["rap_db"]

    assert rap_results, "Expected filtered results for cultural significance"
    assert all(result["cultural_sig"] == "legendary" for result in rap_results)
    assert all(result["target_word"] == "glove" for result in rap_results)


class LoaderStub:
    def __init__(self):
        self.requests = []

    def get_rhyming_words(self, word):
        self.requests.append(word)
        return ["fail", "mail"]

    def get_pronunciations(self, word):
        pronunciations = {
            "paper": [["P", "EY1", "P", "ER0"]],
            "trail": [["T", "R", "EY1", "L"]],
            "fail": [["F", "EY1", "L"]],
            "mail": [["M", "EY1", "L"]],
        }
        return pronunciations.get(word, [])

    def get_rhyme_parts(self, word):
        return {"EY1 L"}


class DeterministicCMULoader:
    """Minimal CMU loader stub with deterministic pronunciations."""

    def __init__(self):
        self._pronunciations = {
            "desk": [["D", "EH1", "S", "K"]],
            "grotesque": [["G", "R", "OW0", "T", "EH1", "S", "K"]],
            "orange": [["AO1", "R", "AH0", "N", "JH"]],
            "sporange": [["S", "P", "AO1", "R", "AH0", "N", "JH"]],
            "door": [["D", "AO1", "R"]],
            "hinge": [["HH", "IH1", "N", "JH"]],
            "fire": [["F", "AY1", "ER0"]],
            "choir": [["K", "W", "AY1", "ER0"]],
            "liar": [["L", "AY1", "ER0"]],
            "wild": [["W", "AY1", "L", "D"]],
            "lamp": [["L", "AE1", "M", "P"]],
            "camp": [["K", "AE1", "M", "P"]],
        }
        self._rhyme_map = {
            "desk": ["grotesque"],
            "orange": ["sporange", "door hinge"],
            "fire": ["liar", "choir"],
            "lamp": ["camp"],
        }
        self._rhyme_parts = {}
        for word, pronunciations in self._pronunciations.items():
            parts = set()
            for phones in pronunciations:
                rhyme_part = self._compute_rhyme_part(phones)
                if rhyme_part:
                    parts.add(rhyme_part)
            if parts:
                self._rhyme_parts[word] = parts

    def _compute_rhyme_part(self, phones):
        last_stress_index = None
        for index, phone in enumerate(phones):
            base = re.sub(r"\d", "", phone)
            if base in VOWEL_PHONEMES and re.search(r"[12]", phone):
                last_stress_index = index
        if last_stress_index is None:
            for index in range(len(phones) - 1, -1, -1):
                base = re.sub(r"\d", "", phones[index])
                if base in VOWEL_PHONEMES:
                    last_stress_index = index
                    break
        if last_stress_index is None:
            return None
        return " ".join(phones[last_stress_index:])

    def get_pronunciations(self, word):
        return [list(phones) for phones in self._pronunciations.get(word.lower(), [])]

    def get_rhyme_parts(self, word):
        return set(self._rhyme_parts.get(word.lower(), set()))

    def get_rhyming_words(self, word):
        return list(self._rhyme_map.get(word.lower(), []))


@pytest.fixture
def deterministic_cmu_loader():
    return DeterministicCMULoader()


def test_search_rhymes_handles_multi_word_phrases(tmp_path):
    db_path = tmp_path / "patterns.db"
    create_test_database(str(db_path))

    app = RhymeRarityApp(db_path=str(db_path))

    loader = LoaderStub()
    app.cmu_loader = loader
    app.search_service.cmu_loader = loader
    app.phonetic_analyzer.cmu_loader = loader
    if hasattr(app.cultural_engine, "set_phonetic_analyzer"):
        app.cultural_engine.set_phonetic_analyzer(app.phonetic_analyzer)

    results = app.search_rhymes("paper trail", limit=5, min_confidence=0.0)

    assert loader.requests == ["trail"]
    source_profile = results["source_profile"]
    assert source_profile["phonetics"].get("is_multi_word")

    cmu_results = results["cmu"]
    assert any(entry["is_multi_word"] for entry in cmu_results)
    cmu_multi = next(entry for entry in cmu_results if entry["is_multi_word"])
    assert " // " in cmu_multi["pattern"]

    rap_results = results["rap_db"]
    assert any(entry["is_multi_word"] for entry in rap_results)
    rap_multi = next(entry for entry in rap_results if entry["is_multi_word"])
    assert " // " in rap_multi["pattern"]


@pytest.mark.parametrize(
    "source_word, expected_syllables, expectations",
    [
        (
            "desk",
            1,
            (
                {
                    "target": "grotesque",
                    "min_similarity": 0.95,
                    "min_confidence": 0.7,
                    "is_multi": False,
                    "expected_syllables": 2,
                },
            ),
        ),
        (
            "orange",
            2,
            (
                {
                    "target": "sporange",
                    "min_similarity": 0.95,
                    "min_confidence": 0.7,
                    "is_multi": False,
                    "expected_syllables": 2,
                },
                {
                    "target": "door hinge",
                    "min_similarity": 0.4,
                    "min_confidence": 0.35,
                    "is_multi": True,
                    "expected_syllables": 2,
                },
            ),
        ),
        (
            "fire",
            1,
            (
                {
                    "target": "choir",
                    "min_similarity": 0.95,
                    "min_confidence": 0.7,
                    "is_multi": False,
                    "expected_syllables": 1,
                },
            ),
        ),
        (
            "wild fire",
            2,
            (
                {
                    "target": "wild choir",
                    "min_similarity": 0.95,
                    "min_confidence": 0.7,
                    "is_multi": True,
                    "expected_syllables": 2,
                },
            ),
        ),
    ],
    ids=["desk", "orange", "fire", "wild-fire"],
)
def test_search_rhymes_phonetic_rhyme_cases(
    deterministic_cmu_loader, source_word, expected_syllables, expectations, tmp_path
):
    db_path = tmp_path / "patterns.db"
    create_test_database(str(db_path))

    app = RhymeRarityApp(db_path=str(db_path), cmu_loader=deterministic_cmu_loader)
    app.set_cultural_engine(None)

    results = app.search_rhymes(
        source_word,
        limit=10,
        min_confidence=0.0,
        result_sources=["phonetic"],
    )

    source_profile = results["source_profile"]
    assert source_profile["phonetics"]["syllables"] == expected_syllables

    cmu_results = results["cmu"]
    assert cmu_results, f"Expected CMU results for '{source_word}'"

    for expectation in expectations:
        target = expectation["target"]
        entry = next(
            (candidate for candidate in cmu_results if candidate["target_word"] == target),
            None,
        )
        assert entry is not None, f"Expected phonetic rhyme '{target}' for '{source_word}'"
        assert entry["phonetic_sim"] >= expectation["min_similarity"]
        assert entry["combined_score"] >= expectation["min_confidence"]
        assert entry["is_multi_word"] is expectation["is_multi"]
        if expectation.get("expected_syllables") is not None:
            assert entry["candidate_syllables"] == expectation["expected_syllables"]


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

    app.set_cultural_engine(MockCulturalEngine())

    results = app.search_rhymes(
        "love",
        limit=5,
        min_confidence=0.0,
        result_sources=["cultural"],
    )

    rap_results = results["rap_db"]

    assert rap_results, "Expected results enriched with cultural context"
    enriched_entry = rap_results[0]
    assert "cultural_context" in enriched_entry
    assert enriched_entry["cultural_rarity"] == 3.5

    formatted = app.format_rhyme_results("love", results)

    assert "Cultural: Era: Golden Age" in formatted
    assert "Region: Queens" in formatted
    assert "Rarity: 3.50" in formatted
    assert "• Styles: Multi Syllable, Storytelling" in formatted


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

    app.set_anti_llm_engine(StubAntiLLMEngine())

    results = app.search_rhymes(
        "love",
        limit=5,
        min_confidence=0.0,
        result_sources=["anti-llm"],
    )

    anti_llm_results = results["anti_llm"]

    assert anti_llm_results, "Expected anti-LLM results when engine is patched"
    anti_entry = anti_llm_results[0]
    assert anti_entry["result_source"] == "anti_llm"
    assert anti_entry["rarity_score"] == pytest.approx(4.2)

    formatted = app.format_rhyme_results("love", results)

    assert "SHOVE" in formatted
    assert "Rarity: 4.20" in formatted
    assert "• LLM weakness: Rare Word Combinations" in formatted
    assert "• Cultural depth: Sentinel Depth" in formatted


def test_min_confidence_filters_phonetic_candidates(monkeypatch, tmp_path):
    db_path = tmp_path / "patterns.db"
    create_test_database(str(db_path))

    app = RhymeRarityApp(db_path=str(db_path))
    app.set_cultural_engine(None)

    def stub_cmu_rhymes(word, limit=20, analyzer=None, cmu_loader=None):
        return [
            {"word": "alpha", "similarity": 0.94, "combined": 0.94, "rarity": 0.8},
            {"word": "beta", "similarity": 0.82, "combined": 0.6, "rarity": 0.4},
        ]

    monkeypatch.setattr(search_service_module, "get_cmu_rhymes", stub_cmu_rhymes)

    results = app.search_rhymes(
        "love",
        limit=5,
        min_confidence=0.9,
        result_sources=["phonetic"],
    )

    cmu_results = results["cmu"]

    assert cmu_results, "Expected high-confidence phonetic suggestions"
    targets = {entry["target_word"] for entry in cmu_results}
    assert "alpha" in targets
    assert "beta" not in targets
    assert all(
        float(entry.get("combined_score", entry.get("confidence", 0.0))) >= 0.9
        for entry in cmu_results
    )


def test_phonetic_candidates_extend_beyond_limit(monkeypatch, tmp_path):
    db_path = tmp_path / "patterns.db"
    create_test_database(str(db_path))

    app = RhymeRarityApp(db_path=str(db_path))

    cmu_payload = [
        {"word": "alpha", "similarity": 0.96, "combined": 0.96, "rarity": 1.1},
        {"word": "beta", "similarity": 0.95, "combined": 0.95, "rarity": 0.9},
        {"word": "gamma", "similarity": 0.94, "combined": 0.94, "rarity": 0.8},
    ]

    def stub_cmu_rhymes(word, limit=20, analyzer=None, cmu_loader=None):
        return [dict(candidate) for candidate in cmu_payload]

    monkeypatch.setattr(search_service_module, "get_cmu_rhymes", stub_cmu_rhymes)

    class FilteringCulturalEngine:
        def __init__(self):
            self.alignment_map = {
                "alpha": {"similarity": 0.96, "combined": 0.96, "rarity": 1.1},
                "beta": None,
                "gamma": {"similarity": 0.94, "combined": 0.94, "rarity": 0.8},
            }

        def derive_rhyme_signatures(self, word):
            return {f"sig-{word}"}

        def evaluate_rhyme_alignment(self, source_word, target_word, **_kwargs):
            alignment = self.alignment_map.get(target_word)
            if alignment is None:
                return None
            return {
                "similarity": alignment["similarity"],
                "combined": alignment["combined"],
                "rarity": alignment["rarity"],
                "signature_matches": [],
                "target_signatures": [],
                "features": {},
                "feature_profile": {},
                "prosody_profile": {},
            }

    app.set_cultural_engine(FilteringCulturalEngine())

    results = app.search_rhymes(
        "love",
        limit=2,
        min_confidence=0.0,
        result_sources=["phonetic"],
    )

    cmu_results = results["cmu"]

    assert cmu_results, "Expected phonetic matches when candidates are available"
    assert len(cmu_results) == 2, "Results should still be capped by the requested limit"
    targets = [entry["target_word"] for entry in cmu_results]
    assert "beta" not in targets
    assert "gamma" in targets, "Candidate beyond the initial limit should be retained"


def test_min_confidence_filters_anti_llm_candidates(tmp_path):
    db_path = tmp_path / "patterns.db"
    create_test_database(str(db_path))

    app = RhymeRarityApp(db_path=str(db_path))
    app.set_cultural_engine(None)

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

    app.set_anti_llm_engine(StubAntiLLMEngine())

    results = app.search_rhymes(
        "love",
        limit=5,
        min_confidence=0.9,
        result_sources=["anti-llm"],
    )

    anti_llm_results = results["anti_llm"]

    assert anti_llm_results, "Expected at least one high-confidence anti-LLM suggestion"
    targets = {entry["target_word"] for entry in anti_llm_results}
    assert "delta" in targets
    assert "gamma" not in targets
    assert all(
        float(entry.get("combined_score", entry.get("confidence", 0.0))) >= 0.9
        for entry in anti_llm_results
    )


def test_search_rhymes_respects_rhyme_type_and_rhythm_filters(tmp_path):
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
                stress_alignment = 0.9
                bradley_device = "multi"
            elif target == "shove":
                rhyme_type = None
                complexity = "steady"
                cadence_ratio = 1.1
                stress_alignment = 0.65
                bradley_device = "perfect"
            else:
                rhyme_type = "slant"
                complexity = "dense"
                cadence_ratio = 1.4
                stress_alignment = 0.72
                bradley_device = "assonance"

            target_signatures = list(self.derive_rhyme_signatures(target))
            signature_matches = list(rhyme_signatures or [])

            feature_profile = {"bradley_device": bradley_device, "stress_alignment": stress_alignment}
            if rhyme_type:
                feature_profile["rhyme_type"] = rhyme_type

            features = {"rhyme_type": rhyme_type, "stress_alignment": stress_alignment} if rhyme_type else {}

            return {
                "similarity": 0.9,
                "rarity": 0.6,
                "combined": 0.75,
                "rhyme_type": rhyme_type,
                "signature_matches": signature_matches,
                "target_signatures": target_signatures,
                "features": features,
                "feature_profile": feature_profile,
                "prosody_profile": {
                    "complexity_tag": complexity,
                    "cadence_ratio": cadence_ratio,
                    "stress_alignment": stress_alignment,
                },
            }

    app.set_cultural_engine(MockCulturalEngine())

    filtered_results = app.search_rhymes(
        "love",
        limit=5,
        min_confidence=0.0,
        result_sources=["cultural"],
        allowed_rhyme_types=["perfect"],
        min_stress_alignment=0.85,
        cadence_focus="steady",
    )

    rap_results = filtered_results["rap_db"]

    assert rap_results, "Expected matches that satisfy rhyme type and cadence filters"
    assert all(entry.get("rhyme_type") == "perfect" for entry in rap_results)
    assert all(entry.get("target_word") != "shove" for entry in rap_results)
    assert all(
        (entry.get("prosody_profile") or {}).get("complexity_tag") == "steady"
        for entry in rap_results
    )
    assert all(
        float(entry.get("rhythm_score", 0.0)) >= 0.85 for entry in rap_results
    )

    formatted = app.format_rhyme_results("love", filtered_results)
    assert "Rhyme type: Perfect" in formatted
    assert "Cadence: Steady" in formatted
    assert "Stress align" not in formatted

