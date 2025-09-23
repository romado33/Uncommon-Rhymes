from rhyme_rarity.utils.profile import normalize_profile_dict
from rhyme_rarity.utils.syllables import estimate_syllable_count


class DummyProfile:
    def as_dict(self):  # pragma: no cover - simple data method
        return {"score": 0.5}


def test_normalize_profile_dict_prefers_as_dict():
    dummy = DummyProfile()

    result = normalize_profile_dict(dummy)

    assert result == {"score": 0.5}
    assert normalize_profile_dict.__module__ == "rhyme_rarity.utils.profile"


def test_estimate_syllable_count_module_location():
    assert estimate_syllable_count("lyrical") >= 1
    assert estimate_syllable_count.__module__ == "rhyme_rarity.utils.syllables"
