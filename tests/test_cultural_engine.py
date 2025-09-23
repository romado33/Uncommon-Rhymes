from module3_enhanced_cultural_database import CulturalIntelligenceEngine


def test_get_cultural_context_handles_none_artist(tmp_path):
    engine = CulturalIntelligenceEngine(db_path=str(tmp_path / "patterns.db"))

    pattern_data = {
        "artist": None,
        "song": "Test Song",
    }

    context = engine.get_cultural_context(pattern_data)

    assert context.artist == ""
    assert context.song == "Test Song"
