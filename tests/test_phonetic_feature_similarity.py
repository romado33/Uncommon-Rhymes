import pytest

from rhyme_rarity.core.analyzer import EnhancedPhoneticAnalyzer


@pytest.fixture()
def analyzer():
    return EnhancedPhoneticAnalyzer()


def test_voicing_difference_rewards_near_rhyme(analyzer):
    """Final consonant voicing shifts (D↔T) should remain highly similar."""

    dt_score = analyzer.get_phonetic_similarity("code", "coat")
    dn_score = analyzer.get_phonetic_similarity("code", "cone")

    assert dt_score > dn_score
    assert dt_score > 0.6


def test_nasal_place_shift_is_near_rhyme(analyzer):
    """Bilabial/alveolar nasal alternation (N↔M) should stay similar."""

    nm_score = analyzer.get_phonetic_similarity("sun", "sum")
    nk_score = analyzer.get_phonetic_similarity("sun", "suck")

    assert nm_score > nk_score
    assert nm_score > 0.55


def test_vowel_tenseness_shift_scores_higher(analyzer):
    """Front vowel tenseness shifts (IY↔IH) should outrank broader changes."""

    tense_shift = analyzer.get_phonetic_similarity("beet", "bit")
    broad_shift = analyzer.get_phonetic_similarity("beet", "bat")

    assert tense_shift > broad_shift
    assert tense_shift > 0.5
