from rhyme_rarity.core import CMUDictLoader


def test_find_words_by_phonemes_ignores_stress_markers():
    loader = CMUDictLoader()

    matches = loader.find_words_by_phonemes(["D", "OW0"], limit=5)

    assert matches, "Expected at least one match for the /DOH/ phoneme sequence"
    assert any(word in matches for word in {"doe", "dough", "doh"})


def test_find_words_by_phonemes_respects_limit():
    loader = CMUDictLoader()

    matches = loader.find_words_by_phonemes(["IH1", "N"], limit=2)

    assert len(matches) <= 2


def test_cmudict_loader_retries_after_file_creation(tmp_path):
    dict_path = tmp_path / "cmudict.7b"
    loader = CMUDictLoader(dict_path=dict_path)

    # Initial attempt should leave the loader in an unloaded state when the file is missing.
    assert loader.get_pronunciations("test") == []
    assert loader._loaded is False
    assert loader._pronunciations == {}
    assert loader._rhyme_parts == {}
    assert loader._rhyme_index == {}

    dict_path.write_text("TEST  T EH1 S T\n", encoding="utf-8")

    # After the file becomes available, the loader should retry and populate caches.
    assert loader.get_pronunciations("test") == [["T", "EH1", "S", "T"]]
    assert loader._loaded is True
    assert "test" in loader._rhyme_parts
    assert loader._rhyme_parts["test"] == {"EH1 S T"}


def test_split_pronunciation_into_words_returns_dictionary_pairs():
    loader = CMUDictLoader()
    pronunciations = loader.get_pronunciations("below")

    assert pronunciations, "Expected CMU pronunciations for 'below'"

    pairs = loader.split_pronunciation_into_words(pronunciations[0], max_pairs=5)

    assert pairs, "Expected at least one phoneme split for 'below'"
    for prefix, suffix, split_index in pairs:
        assert prefix and suffix
        assert isinstance(split_index, int)
        combined = f"{prefix} {suffix}".strip()
        assert combined

        prefix_prons = loader.get_pronunciations(prefix)
        suffix_prons = loader.get_pronunciations(suffix)
        assert prefix_prons, f"Expected CMU pronunciation for prefix '{prefix}'"
        assert suffix_prons, f"Expected CMU pronunciation for suffix '{suffix}'"
