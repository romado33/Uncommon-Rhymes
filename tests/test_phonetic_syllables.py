from rhyme_rarity.core import (
    CMUDictLoader,
    EnhancedPhoneticAnalyzer,
    extract_phrase_components,
    get_cmu_rhymes,
)
from rhyme_rarity.utils.syllables import estimate_syllable_count


def test_enhanced_phonetic_analyzer_uses_shared_syllable_helper():
    analyzer = EnhancedPhoneticAnalyzer()
    sample_words = ["flow", "table", "amazing", "rhythm"]

    for word in sample_words:
        assert analyzer.estimate_syllables(word) == estimate_syllable_count(word)


def test_shared_syllable_helper_minimum_one_syllable():
    assert estimate_syllable_count("") == 1


def test_extract_phrase_components_for_phrase():
    components = extract_phrase_components("Paper Trail")
    assert components.anchor == "trail"
    assert components.normalized_phrase == "paper trail"
    assert components.total_syllables == estimate_syllable_count("paper") + estimate_syllable_count("trail")


class DummyLoader:
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


def test_get_cmu_rhymes_uses_anchor_and_produces_phrase_variants():
    loader = DummyLoader()
    analyzer = EnhancedPhoneticAnalyzer(cmu_loader=loader)
    results = get_cmu_rhymes("paper trail", analyzer=analyzer, cmu_loader=loader, limit=10)

    assert loader.requests == ["trail"], "Expected anchor lookup for the final stressed token"
    words = {entry["word"] for entry in results}
    assert "fail" in words
    assert "mail" in words
    assert any(" " in entry["word"] for entry in results if entry.get("is_multi_word"))
    phrase_words = {entry["word"] for entry in results if entry.get("is_multi_word")}
    assert "paper fail" in phrase_words or "paper mail" in phrase_words


def test_single_word_queries_produce_phoneme_split_multi_words():
    loader = CMUDictLoader()
    analyzer = EnhancedPhoneticAnalyzer(cmu_loader=loader)

    results = get_cmu_rhymes("window", analyzer=analyzer, cmu_loader=loader, limit=10)

    multi_variants = [
        entry["word"]
        for entry in results
        if entry.get("is_multi_word") and " " in str(entry.get("word", ""))
    ]

    assert multi_variants, "Expected phoneme-split multi-word variants for single-word input"


def test_multi_word_variants_survive_large_single_candidate_lists():
    class LongListLoader(DummyLoader):
        def get_rhyming_words(self, word):
            super().get_rhyming_words(word)
            return [f"option{i}" for i in range(40)] + ["fail", "mail"]

    loader = LongListLoader()
    analyzer = EnhancedPhoneticAnalyzer(cmu_loader=loader)

    results = get_cmu_rhymes("paper trail", analyzer=analyzer, cmu_loader=loader, limit=5)

    multi_variants = [
        entry["word"]
        for entry in results
        if entry.get("is_multi_word") and " " in str(entry.get("word", ""))
    ]

    assert multi_variants, "Expected multi-word variants to survive limit slicing"


def test_derive_rhyme_profile_returns_rhyme_type_metadata():
    analyzer = EnhancedPhoneticAnalyzer()

    profile = analyzer.derive_rhyme_profile("money", "honey")

    assert isinstance(profile, dict)
    assert profile["rhyme_type"] == "perfect"
    assert profile["last_vowel_sound_source"] == profile["last_vowel_sound_target"]
    assert profile["consonant_onset_source"] == profile["consonant_onset_target"]
    assert profile["consonant_coda_source"] == profile["consonant_coda_target"]


def test_slant_rhyme_requires_shared_vowel_and_varied_consonants():
    analyzer = EnhancedPhoneticAnalyzer()

    profile = analyzer.derive_rhyme_profile("money", "lusty")

    assert profile["rhyme_type"] == "slant"
    assert profile["last_vowel_sound_source"] == profile["last_vowel_sound_target"]
    differs = (
        profile["consonant_onset_source"] != profile["consonant_onset_target"]
        or profile["consonant_coda_source"] != profile["consonant_coda_target"]
    )
    assert differs, "Expected consonant context to differ for slant rhymes"
