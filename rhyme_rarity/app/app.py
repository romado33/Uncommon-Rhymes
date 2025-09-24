"""Application wiring for the RhymeRarity project."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from rhyme_rarity.core import CmuRhymeRepository, CMUDictLoader, EnhancedPhoneticAnalyzer
from anti_llm import AntiLLMRhymeEngine
from cultural.engine import CulturalIntelligenceEngine

from .data.database import SQLiteRhymeRepository
from .services.search_service import SearchService
from .ui.gradio import create_interface


class RhymeRarityApp:
    """High-level application facade bundling dependencies."""

    def __init__(
        self,
        db_path: str = "patterns.db",
        *,
        repository: Optional[SQLiteRhymeRepository] = None,
        phonetic_analyzer: Optional[EnhancedPhoneticAnalyzer] = None,
        cultural_engine: Optional[CulturalIntelligenceEngine] = None,
        anti_llm_engine: Optional[AntiLLMRhymeEngine] = None,
        search_service: Optional[SearchService] = None,
        cmu_loader: Optional[CMUDictLoader] = None,
    ) -> None:
        self.db_path = db_path
        self.repository = repository or SQLiteRhymeRepository(db_path)
        self.repository.ensure_database()

        self.cmu_loader = cmu_loader or CMUDictLoader()
        self.phonetic_analyzer = phonetic_analyzer or EnhancedPhoneticAnalyzer(
            cmu_loader=self.cmu_loader
        )

        self.cultural_engine = cultural_engine or CulturalIntelligenceEngine(
            db_path=db_path,
            phonetic_analyzer=self.phonetic_analyzer,
        )
        self.anti_llm_engine = anti_llm_engine or AntiLLMRhymeEngine(
            db_path=db_path,
            phonetic_analyzer=self.phonetic_analyzer,
        )

        self.cmu_repository = CmuRhymeRepository(
            loader=self.cmu_loader,
            analyzer=self.phonetic_analyzer,
        )

        self._wire_component(self.anti_llm_engine, "set_phonetic_analyzer", self.phonetic_analyzer)
        self._wire_component(self.cultural_engine, "set_phonetic_analyzer", self.phonetic_analyzer)
        self._wire_component(self.cultural_engine, "set_prosody_analyzer", self.phonetic_analyzer)

        self.search_service = search_service or SearchService(
            repository=self.repository,
            phonetic_analyzer=self.phonetic_analyzer,
            cultural_engine=self.cultural_engine,
            anti_llm_engine=self.anti_llm_engine,
            cmu_loader=getattr(self.phonetic_analyzer, "cmu_loader", self.cmu_loader),
            cmu_repository=self.cmu_repository,
        )
        if search_service is not None:
            self.search_service.set_phonetic_analyzer(self.phonetic_analyzer)
            self.search_service.set_cultural_engine(self.cultural_engine)
            self.search_service.set_anti_llm_engine(self.anti_llm_engine)

        self._refresh_rarity_map()

    # Dependency management -------------------------------------------------
    def set_phonetic_analyzer(self, analyzer: EnhancedPhoneticAnalyzer) -> None:
        self.phonetic_analyzer = analyzer
        self.cmu_repository.set_analyzer(analyzer)
        self._wire_component(self.anti_llm_engine, "set_phonetic_analyzer", analyzer)
        self._wire_component(self.cultural_engine, "set_phonetic_analyzer", analyzer)
        self._wire_component(self.cultural_engine, "set_prosody_analyzer", analyzer)
        self.search_service.set_phonetic_analyzer(analyzer)

    def set_cultural_engine(self, engine: Optional[CulturalIntelligenceEngine]) -> None:
        self.cultural_engine = engine
        self._wire_component(engine, "set_phonetic_analyzer", self.phonetic_analyzer)
        self._wire_component(engine, "set_prosody_analyzer", self.phonetic_analyzer)
        self.search_service.set_cultural_engine(engine)

    def set_anti_llm_engine(self, engine: Optional[AntiLLMRhymeEngine]) -> None:
        self.anti_llm_engine = engine
        self._wire_component(engine, "set_phonetic_analyzer", self.phonetic_analyzer)
        self.search_service.set_anti_llm_engine(engine)

    # Public API ------------------------------------------------------------
    def search_rhymes(self, *args, **kwargs) -> Dict[str, List[Dict]]:
        return self.search_service.search_rhymes(*args, **kwargs)

    def format_rhyme_results(self, source_word: str, rhymes: Dict[str, Any]) -> str:
        return self.search_service.format_rhyme_results(source_word, rhymes)

    def create_gradio_interface(self):
        return create_interface(self.search_service, self.repository)

    # Internal helpers ------------------------------------------------------
    def _refresh_rarity_map(self) -> None:
        analyzer = getattr(self, "phonetic_analyzer", None)
        if not analyzer:
            return
        updater = getattr(analyzer, "update_rarity_from_database", None)
        if callable(updater):
            try:
                updater(self.db_path)
            except Exception:
                pass

    @staticmethod
    def _wire_component(component: Optional[Any], method: str, *args: Any) -> None:
        if component is None:
            return
        candidate = getattr(component, method, None)
        if callable(candidate):
            try:
                candidate(*args)
            except Exception:
                pass


def _should_share_interface() -> bool:
    """Return whether the Gradio UI should request a public share link."""

    env_value = os.environ.get("RHYMES_SHARE", "")
    if not env_value:
        return False
    normalized = str(env_value).strip().lower()
    return normalized in {"1", "true", "yes", "on"}


def main() -> None:
    app = RhymeRarityApp()
    interface = app.create_gradio_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_api=True,
        share=_should_share_interface(),
    )


__all__ = ["RhymeRarityApp", "main"]

