from __future__ import annotations

from typing import Any, Dict, List

from anti_llm.engine import AntiLLMRhymeEngine


class CountingRepository:
    def __init__(self) -> None:
        self.calls: Dict[str, int] = {}

    def _record(self, name: str) -> None:
        self.calls[name] = self.calls.get(name, 0) + 1

    def fetch_rare_combinations(self, source_word: str, limit: int) -> List[Dict[str, Any]]:
        self._record("rare")
        return []

    def fetch_phonological_challenges(self, source_word: str, limit: int) -> List[Dict[str, Any]]:
        self._record("phonological")
        return []

    def fetch_cultural_depth_patterns(self, source_word: str, limit: int) -> List[Dict[str, Any]]:
        self._record("cultural")
        return []

    def fetch_complex_syllable_patterns(self, source_word: str, limit: int) -> List[Dict[str, Any]]:
        self._record("complex")
        return []

    def fetch_seed_neighbors(self, seed_word: str, limit: int) -> List[Dict[str, Any]]:
        self._record("neighbors")
        return []

    def fetch_suffix_matches(self, suffix: str, limit: int) -> List[Dict[str, Any]]:
        self._record("suffix")
        return []


def test_generate_patterns_uses_pattern_cache() -> None:
    repo = CountingRepository()
    engine = AntiLLMRhymeEngine(repository=repo)

    engine.generate_anti_llm_patterns("echo", limit=5, module1_seeds=[], seed_signatures=set(), delivered_words=set())
    first_counts = dict(repo.calls)

    engine.generate_anti_llm_patterns("echo", limit=5, module1_seeds=[], seed_signatures=set(), delivered_words=set())
    assert repo.calls == first_counts
