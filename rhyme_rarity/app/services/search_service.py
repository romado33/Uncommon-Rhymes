"""Facade around core rhyme orchestration and formatting services."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from rhyme_rarity.core import EnhancedPhoneticAnalyzer, extract_phrase_components
from rhyme_rarity.core.cmu_repository import CmuRhymeRepository
from anti_llm import AntiLLMRhymeEngine
from cultural.engine import CulturalIntelligenceEngine

from ..data.database import SQLiteRhymeRepository
from .result_formatter import RhymeResultFormatter
from .rhyme_query import RhymeQueryOrchestrator


class SearchService:
    """High-level API that coordinates query orchestration and formatting."""

    def __init__(
        self,
        *,
        repository: SQLiteRhymeRepository,
        phonetic_analyzer: Optional[EnhancedPhoneticAnalyzer] = None,
        cultural_engine: Optional[CulturalIntelligenceEngine] = None,
        anti_llm_engine: Optional[AntiLLMRhymeEngine] = None,
        cmu_loader: Optional[object] = None,
        cmu_repository: Optional[CmuRhymeRepository] = None,
        query_orchestrator: Optional[RhymeQueryOrchestrator] = None,
        result_formatter: Optional[RhymeResultFormatter] = None,
    ) -> None:
        self.repository = repository
        self.phonetic_analyzer = phonetic_analyzer
        self.cultural_engine = cultural_engine
        self.anti_llm_engine = anti_llm_engine
        self.cmu_loader = cmu_loader or (
            getattr(phonetic_analyzer, "cmu_loader", None) if phonetic_analyzer else None
        )

        self.cmu_repository = cmu_repository or CmuRhymeRepository(
            loader=self.cmu_loader,
            analyzer=phonetic_analyzer,
        )

        self.query_orchestrator = query_orchestrator or RhymeQueryOrchestrator(
            repository=repository,
            phonetic_analyzer=phonetic_analyzer,
            cultural_engine=cultural_engine,
            anti_llm_engine=anti_llm_engine,
            cmu_loader=self.cmu_loader,
            cmu_repository=self.cmu_repository,
        )
        self.result_formatter = result_formatter or RhymeResultFormatter()

    # Dependency management -------------------------------------------------
    def set_phonetic_analyzer(self, analyzer: Optional[EnhancedPhoneticAnalyzer]) -> None:
        self.phonetic_analyzer = analyzer
        if analyzer is not None:
            self.cmu_loader = getattr(analyzer, "cmu_loader", self.cmu_loader)
        self.cmu_repository.set_analyzer(analyzer)
        self.query_orchestrator.set_phonetic_analyzer(analyzer)

    def set_cultural_engine(self, engine: Optional[CulturalIntelligenceEngine]) -> None:
        self.cultural_engine = engine
        self.query_orchestrator.set_cultural_engine(engine)

    def set_anti_llm_engine(self, engine: Optional[AntiLLMRhymeEngine]) -> None:
        self.anti_llm_engine = engine
        self.query_orchestrator.set_anti_llm_engine(engine)

    # Public API ------------------------------------------------------------
    def clear_cached_results(self) -> None:
        self.query_orchestrator.clear_cached_results()

    def normalize_source_name(self, name: Optional[str]) -> str:
        return self.query_orchestrator.normalize_source_name(name)

    def search_rhymes(
        self,
        source_word: str,
        limit: int = 20,
        min_confidence: float = 0.7,
        cultural_significance: Optional[List[str]] = None,
        genres: Optional[List[str]] = None,
        result_sources: Optional[List[str]] = None,
        max_line_distance: Optional[int] = None,
        min_syllables: Optional[int] = None,
        max_syllables: Optional[int] = None,
        allowed_rhyme_types: Optional[List[str]] = None,
        bradley_devices: Optional[List[str]] = None,
        require_internal: bool = False,
        min_rarity: Optional[float] = None,
        min_stress_alignment: Optional[float] = None,
        cadence_focus: Optional[str] = None,
    ) -> Dict[str, List[Dict]]:
        return self.query_orchestrator.search_rhymes(
            source_word,
            limit=limit,
            min_confidence=min_confidence,
            cultural_significance=cultural_significance,
            genres=genres,
            result_sources=result_sources,
            max_line_distance=max_line_distance,
            min_syllables=min_syllables,
            max_syllables=max_syllables,
            allowed_rhyme_types=allowed_rhyme_types,
            bradley_devices=bradley_devices,
            require_internal=require_internal,
            min_rarity=min_rarity,
            min_stress_alignment=min_stress_alignment,
            cadence_focus=cadence_focus,
        )

    def format_rhyme_results(self, source_word: str, rhymes: Dict[str, Any]) -> str:
        return self.result_formatter.format_results(source_word, rhymes)
