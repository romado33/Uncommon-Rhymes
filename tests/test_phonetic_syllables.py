from module1_enhanced_core_phonetic import EnhancedPhoneticAnalyzer
from syllable_utils import estimate_syllable_count


def test_enhanced_phonetic_analyzer_uses_shared_syllable_helper():
    analyzer = EnhancedPhoneticAnalyzer()
    sample_words = ["flow", "table", "amazing", "rhythm"]

    for word in sample_words:
        assert analyzer.estimate_syllables(word) == estimate_syllable_count(word)


def test_shared_syllable_helper_minimum_one_syllable():
    assert estimate_syllable_count("") == 1
