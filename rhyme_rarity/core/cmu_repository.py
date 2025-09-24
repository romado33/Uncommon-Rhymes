"""Repository abstraction around CMU rhyme retrieval and scoring."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from .analyzer import EnhancedPhoneticAnalyzer
from .cmudict_loader import CMUDictLoader, DEFAULT_CMU_LOADER
from .feature_profile import extract_phrase_components, pronouncing
from .rarity_map import DEFAULT_RARITY_MAP, WordRarityMap

RhymeCandidate = Dict[str, Any]


class CmuRhymeRepository:
    """Fetch and score rhyme candidates from CMU resources."""

    def __init__(
        self,
        *,
        loader: Optional[CMUDictLoader] = None,
        analyzer: Optional[EnhancedPhoneticAnalyzer] = None,
        rarity_map: Optional[WordRarityMap] = None,
    ) -> None:
        self.loader = loader
        self._analyzer = analyzer
        self._rarity_map = rarity_map

    # Resource management ---------------------------------------------------
    def set_analyzer(self, analyzer: Optional[EnhancedPhoneticAnalyzer]) -> None:
        self._analyzer = analyzer
        if analyzer is None:
            return
        loader = getattr(analyzer, "cmu_loader", None)
        if loader is not None:
            self.loader = loader
        rarity_map = getattr(analyzer, "rarity_map", None)
        if rarity_map is not None:
            self._rarity_map = rarity_map

    def update_resources(
        self,
        *,
        loader: Optional[CMUDictLoader] = None,
        rarity_map: Optional[WordRarityMap] = None,
    ) -> None:
        if loader is not None:
            self.loader = loader
        if rarity_map is not None:
            self._rarity_map = rarity_map

    # Internal helpers ------------------------------------------------------
    def _resolve_loader(
        self,
        loader: Optional[CMUDictLoader],
        analyzer: Optional[EnhancedPhoneticAnalyzer],
    ) -> CMUDictLoader:
        if loader is not None:
            return loader
        if analyzer is not None:
            candidate = getattr(analyzer, "cmu_loader", None)
            if candidate is not None:
                return candidate
        if self.loader is not None:
            return self.loader
        if self._analyzer is not None:
            candidate = getattr(self._analyzer, "cmu_loader", None)
            if candidate is not None:
                return candidate
        return DEFAULT_CMU_LOADER

    def _resolve_rarity_map(
        self,
        analyzer: Optional[EnhancedPhoneticAnalyzer],
    ) -> WordRarityMap:
        if self._rarity_map is not None:
            return self._rarity_map
        if analyzer is not None:
            candidate = getattr(analyzer, "rarity_map", None)
            if hasattr(candidate, "get_rarity"):
                return candidate  # type: ignore[return-value]
        if self._analyzer is not None:
            candidate = getattr(self._analyzer, "rarity_map", None)
            if hasattr(candidate, "get_rarity"):
                return candidate  # type: ignore[return-value]
        return DEFAULT_RARITY_MAP

    def _resolve_analyzer(
        self,
        analyzer: Optional[EnhancedPhoneticAnalyzer],
        loader: CMUDictLoader,
        rarity_map: WordRarityMap,
    ) -> EnhancedPhoneticAnalyzer:
        if analyzer is not None:
            return analyzer
        if self._analyzer is not None:
            return self._analyzer
        analyzer = EnhancedPhoneticAnalyzer(cmu_loader=loader, rarity_map=rarity_map)
        self._analyzer = analyzer
        return analyzer

    # Public API ------------------------------------------------------------
    def fetch_rhymes(
        self,
        word: str,
        *,
        limit: int = 20,
        analyzer: Optional[EnhancedPhoneticAnalyzer] = None,
        cmu_loader: Optional[CMUDictLoader] = None,
    ) -> List[RhymeCandidate]:
        """Retrieve rhyme candidates from the CMU pronouncing dictionary."""

        if not word or not str(word).strip() or limit <= 0:
            return []

        base_phrase = str(word).strip()
        loader = self._resolve_loader(cmu_loader, analyzer)
        rarity_map = self._resolve_rarity_map(analyzer)
        scoring_analyzer = self._resolve_analyzer(analyzer, loader, rarity_map)

        components = extract_phrase_components(base_phrase, loader)
        normalized_phrase = components.normalized_phrase or base_phrase.lower()
        anchor_lookup = (
            components.anchor
            or normalized_phrase.split()[-1]
            if normalized_phrase
            else base_phrase.lower()
        )

        candidate_words: List[str] = []
        if loader is not None and anchor_lookup:
            try:
                candidate_words = loader.get_rhyming_words(anchor_lookup)
            except Exception:
                candidate_words = []

        if not candidate_words and pronouncing is not None and anchor_lookup:
            try:
                candidates = pronouncing.rhymes(anchor_lookup)

                if not candidates:
                    phones = pronouncing.phones_for_word(anchor_lookup)
                    for phone in phones:
                        rhyme_part = pronouncing.rhyming_part(phone)
                        if rhyme_part:
                            pattern = f".*{rhyme_part}"
                            candidates.extend(pronouncing.search(pattern))

                seen = set()
                for candidate in candidates:
                    cleaned = candidate.strip().lower()
                    if not cleaned or cleaned == anchor_lookup or cleaned in seen:
                        continue
                    seen.add(cleaned)
                    candidate_words.append(cleaned)
            except Exception:
                candidate_words = []

        if not candidate_words:
            return []

        rarity_source = getattr(scoring_analyzer, "rarity_map", None) or rarity_map
        combine_fn = getattr(scoring_analyzer, "combine_similarity_and_rarity", None)

        anchor_index = components.anchor_index
        if anchor_index is None and components.normalized_tokens:
            anchor_index = len(components.normalized_tokens) - 1

        prefix_tokens_norm: List[str] = []
        suffix_tokens_norm: List[str] = []
        prefix_tokens_display: List[str] = []
        suffix_tokens_display: List[str] = []

        if anchor_index is not None:
            prefix_tokens_norm = components.normalized_tokens[:anchor_index]
            suffix_tokens_norm = components.normalized_tokens[anchor_index + 1 :]
            prefix_tokens_display = components.tokens[:anchor_index]
            suffix_tokens_display = components.tokens[anchor_index + 1 :]

        prefix_text_norm = " ".join(prefix_tokens_norm).strip()
        suffix_text_norm = " ".join(suffix_tokens_norm).strip()
        prefix_text_display = " ".join(prefix_tokens_display).strip()
        suffix_text_display = " ".join(suffix_tokens_display).strip()
        prefix_phrase = prefix_text_display or prefix_text_norm
        suffix_phrase = suffix_text_display or suffix_text_norm

        scored_candidates: List[Dict[str, Any]] = []
        seen_variants: Set[str] = set()

        def _score_variant(suggestion: str, *, is_multi: bool, base_candidate: str) -> None:
            normalized_suggestion = suggestion.strip()
            if not normalized_suggestion:
                return
            normalized_key = normalized_suggestion.lower()
            if normalized_key == normalized_phrase.lower():
                return
            if normalized_key in seen_variants:
                return
            seen_variants.add(normalized_key)

            try:
                score = float(
                    scoring_analyzer.get_phonetic_similarity(base_phrase, suggestion)
                )
            except Exception:
                score = 0.0

            try:
                if " " in normalized_suggestion:
                    token_scores: List[float] = []
                    for token in normalized_suggestion.split():
                        try:
                            token_scores.append(float(rarity_source.get_rarity(token)))
                        except Exception:
                            token_scores.append(float(DEFAULT_RARITY_MAP.get_rarity(token)))
                    rarity = sum(token_scores) / len(token_scores) if token_scores else 0.0
                else:
                    rarity = float(rarity_source.get_rarity(base_candidate))
            except Exception:
                rarity = DEFAULT_RARITY_MAP.get_rarity(base_candidate)

            if callable(combine_fn):
                try:
                    combined = float(combine_fn(score, rarity))
                except Exception:
                    combined = score
            else:
                combined = score

            candidate_info = extract_phrase_components(suggestion, loader)

            entry: Dict[str, Any] = {
                "word": suggestion,
                "target": suggestion,
                "candidate": base_candidate,
                "similarity": score,
                "score": score,
                "rarity": rarity,
                "rarity_score": rarity,
                "combined": combined,
                "combined_score": combined,
                "is_multi_word": is_multi,
                "prefix": prefix_text_norm,
                "suffix": suffix_text_norm,
                "prefix_display": prefix_text_display,
                "suffix_display": suffix_text_display,
                "source_phrase": base_phrase,
                "target_tokens": candidate_info.normalized_tokens,
                "target_tokens_display": candidate_info.tokens,
                "candidate_syllables": candidate_info.total_syllables,
                "anchor_display": candidate_info.anchor_display,
                "source_word": base_phrase,
                "target_word": suggestion,
            }

            scored_candidates.append(entry)

        for candidate_word in candidate_words:
            if not candidate_word:
                continue
            if " " in candidate_word:
                suggestion = candidate_word
                _score_variant(suggestion, is_multi=True, base_candidate=candidate_word)
                continue

            _score_variant(candidate_word, is_multi=False, base_candidate=candidate_word)

            if prefix_phrase:
                combined = f"{prefix_phrase} {candidate_word}".strip()
                _score_variant(combined, is_multi=True, base_candidate=candidate_word)
            if suffix_phrase:
                combined = f"{candidate_word} {suffix_phrase}".strip()
                _score_variant(combined, is_multi=True, base_candidate=candidate_word)
            if prefix_phrase and suffix_phrase:
                combined = f"{prefix_phrase} {candidate_word} {suffix_phrase}".strip()
                _score_variant(combined, is_multi=True, base_candidate=candidate_word)

        scored_candidates.sort(key=lambda item: item.get("combined", 0.0), reverse=True)
        return scored_candidates[: max(0, int(limit))]


# Backwards compatible helper -----------------------------------------------
_default_repository = CmuRhymeRepository()


def get_cmu_rhymes(
    word: str,
    limit: int = 20,
    analyzer: Optional[EnhancedPhoneticAnalyzer] = None,
    cmu_loader: Optional[CMUDictLoader] = None,
    repository: Optional[CmuRhymeRepository] = None,
) -> List[RhymeCandidate]:
    """Compatibility wrapper returning CMU rhyme candidates."""

    repo = repository or _default_repository
    if analyzer is not None:
        repo.set_analyzer(analyzer)
    if cmu_loader is not None:
        repo.update_resources(loader=cmu_loader)
    return repo.fetch_rhymes(
        word,
        limit=limit,
        analyzer=analyzer,
        cmu_loader=cmu_loader,
    )
