from __future__ import annotations

import types
from collections import defaultdict

from rhyme_rarity.core.analyzer import EnhancedPhoneticAnalyzer


class CountingLoader:
    def __init__(self) -> None:
        self.pronunciations = defaultdict(int)
        self.rhyme_parts = defaultdict(int)

    def get_pronunciations(self, word: str) -> list[list[str]]:
        self.pronunciations[word.lower()] += 1
        return [["EH1", "K", "OW0"]]

    def get_rhyme_parts(self, word: str) -> set[str]:
        self.rhyme_parts[word.lower()] += 1
        return {"EH OW"}


def test_phonetic_analyzer_reuses_cached_pronunciations() -> None:
    loader = CountingLoader()
    analyzer = EnhancedPhoneticAnalyzer(cmu_loader=loader)

    def fake_phrase_components(self, word: str):
        normalized = word.lower()
        return types.SimpleNamespace(
            original=word,
            tokens=[word],
            normalized_tokens=[normalized],
            normalized_phrase=normalized,
            anchor=normalized,
            anchor_display=word,
            anchor_index=0,
            syllable_counts=[1],
            total_syllables=1,
            anchor_pronunciations=[],
        )

    analyzer._phrase_components = types.MethodType(fake_phrase_components, analyzer)

    analyzer.get_phonetic_similarity("Echo", "Bellow")

    # First invocation should populate the loader-backed caches
    first_pron_counts = dict(loader.pronunciations)
    first_rhyme_counts = dict(loader.rhyme_parts)
    assert set(first_pron_counts) == {"echo", "bellow"}
    assert set(first_rhyme_counts) == {"echo", "bellow"}
    assert all(count > 0 for count in first_pron_counts.values())
    assert all(count > 0 for count in first_rhyme_counts.values())

    analyzer.get_phonetic_similarity("Echo", "Bellow")

    # Subsequent invocations should reuse cached pronunciations/tails
    assert dict(loader.pronunciations) == first_pron_counts
    assert dict(loader.rhyme_parts) == first_rhyme_counts

    analyzer.get_phonetic_similarity("Echo", "Bellow")

    # And repeated calls should continue to avoid additional lookups
    assert dict(loader.pronunciations) == first_pron_counts
    assert dict(loader.rhyme_parts) == first_rhyme_counts
