import pytest

from rhyme_rarity.core.phrase_corpus import lookup_template_words


@pytest.mark.parametrize(
    "rhyme_key, expected_words",
    [
        ("AE N", {"lantern", "gather"}),
        ("UW N", {"lagoon", "attune"}),
    ],
)
def test_lookup_template_words_uses_extended_bank(rhyme_key, expected_words):
    results = lookup_template_words([rhyme_key])
    # Ensure targeted entries from the data-backed inventory appear alongside
    # generic defaults.
    aggregated = set().union(*results.values())
    missing = expected_words - aggregated
    if missing:
        pytest.fail(f"Expected {expected_words} but missing {missing} in template results")


def test_lookup_template_words_falls_back_to_vowel_family():
    # The key "AE S K" is not represented directly in the JSON configuration.
    # The lookup should fall back to the vowel family templates and emit
    # morphological variants derived from the stored stems.
    results = lookup_template_words(["AE S K"])
    assert "lanterns" in results["nouns"]
    assert {"chant", "chanting"}.issubset(results["verbs"])
