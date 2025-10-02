from rhyme_rarity.core import (
    CMUDictLoader,
    EnhancedPhoneticAnalyzer,
    extract_phrase_components,
    get_cmu_rhymes,
    passes_gate,
    score_pair,
    SlantScore,
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


def test_describe_word_reports_multi_word_stress_patterns():
    loader = CMUDictLoader()
    analyzer = EnhancedPhoneticAnalyzer(cmu_loader=loader)

    profile = analyzer.describe_word("carry lindow")

    assert profile["stress_pattern_display"] == "1-0 1-0"
    assert profile["stress_pattern"] == "1010"


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


def test_template_and_corpus_variants_expand_multi_word_pool():
    loader = DummyLoader()
    analyzer = EnhancedPhoneticAnalyzer(cmu_loader=loader)

    results = get_cmu_rhymes("paper trail", analyzer=analyzer, cmu_loader=loader, limit=20)

    multi_entries = [entry for entry in results if entry.get("is_multi_word")]
    assert multi_entries, "Expected multi-word entries from template assembly"

    sources = {entry.get("multi_source") for entry in multi_entries}
    assert "template" in sources, "Template-driven variants should be surfaced"
    assert "corpus_ngram" in sources, "N-gram corpus variants should be surfaced"

    multi_words = [entry["word"] for entry in multi_entries]
    assert len(multi_words) == len(set(multi_words)), "Variants should not duplicate"
    assert any("chain mail" == word for word in multi_words) or any(
        "snail mail" == word for word in multi_words
    ), "Expected idiomatic corpus phrase ending in rhyme key"
    assert len(multi_words) >= 3, "Should provide broader multi-word coverage"


class CreativeHookAnalyzer(EnhancedPhoneticAnalyzer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hook_calls = []

    def generate_constrained_phrases(self, base_word: str, rhyme_keys=()):
        self.hook_calls.append((base_word, tuple(rhyme_keys)))
        return [f"beam {base_word}"]


def test_constrained_generation_hook_is_respected():
    loader = DummyLoader()
    analyzer = CreativeHookAnalyzer(cmu_loader=loader)

    results = get_cmu_rhymes("paper trail", analyzer=analyzer, cmu_loader=loader, limit=20)

    creative_entries = [
        entry for entry in results if entry.get("multi_source") == "creative_hook"
    ]

    assert analyzer.hook_calls, "Expected constrained generation hook to be invoked"
    assert creative_entries, "Creative hook output should appear in results"
    assert all(
        entry["word"].split()[-1] in {"fail", "mail"} for entry in creative_entries
    )


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


def test_large_rhyme_classes_are_bounded_before_scoring():
    class LargeClassLoader(DummyLoader):
        def get_rhyming_words(self, word):
            super().get_rhyming_words(word)
            return [f"option{i}" for i in range(600)]

        def get_pronunciations(self, word):
            if str(word).startswith("option"):
                return [["AA1", "P", "SH", "AH", "N"]]
            return super().get_pronunciations(word)

        def get_rhyme_parts(self, word):
            if str(word).startswith("option"):
                return {"AA1 P SH AH N"}
            return super().get_rhyme_parts(word)

    class CountingAnalyzer(EnhancedPhoneticAnalyzer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.slant_calls = []

        def get_slant_score(self, word1: str, word2: str) -> SlantScore:
            self.slant_calls.append((word1, word2))
            return SlantScore(
                total=0.98,
                rime=0.98,
                vowel=0.98,
                coda=0.98,
                stress_penalty=0.0,
                syllable_penalty=0.0,
                tier="perfect",
            )

    limit = 7
    loader = LargeClassLoader()
    analyzer = CountingAnalyzer(cmu_loader=loader)

    results = get_cmu_rhymes("paper trail", analyzer=analyzer, cmu_loader=loader, limit=limit)

    assert results, "Expected bounded results even for large rhyme classes"
    max_pool = max(limit * 5, 100)
    assert len(analyzer.slant_calls) <= max_pool


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


def test_passes_gate_enforces_tier_thresholds():
    analyzer = EnhancedPhoneticAnalyzer()

    base = "window"
    expectations = {
        "window": ("perfect", True),
        "bingo": ("very_close", True),
        "widow": ("strong", True),
        "pillow": ("strong", True),
        "hello": ("strong", True),
        "yellow": ("loose", True),
        "caper": ("weak", False),
    }

    for candidate, (expected_tier, expected_gate) in expectations.items():
        score = score_pair(analyzer, base, candidate)
        assert score.tier == expected_tier
        assert passes_gate(score) is expected_gate


def test_compound_rhyme_keys_expand_candidate_pool():
    loader = CMUDictLoader()
    analyzer = EnhancedPhoneticAnalyzer(cmu_loader=loader)

    results = get_cmu_rhymes("him so", analyzer=analyzer, cmu_loader=loader, limit=25)

    words = {entry["word"] for entry in results if entry.get("word")}
    assert "window" in words, "Expected single-word slant driven by compound backoff"

    compound_entries = [
        entry
        for entry in results
        if entry.get("is_multi_word")
        and "matched_rhyme_keys" in entry
        and "compound" in entry["matched_rhyme_keys"]
        and "IH M OW" in entry["matched_rhyme_keys"]["compound"]
    ]

    assert compound_entries, "Expected multi-word variant seeded by /IH M OW/ compound key"
    assert any(
        "compound" in entry.get("matched_rhyme_key_types", []) for entry in compound_entries
    )
