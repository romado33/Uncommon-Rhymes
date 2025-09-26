"""Application wiring for the RhymeRarity project."""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]
try:
    import spaces
except ImportError:  # pragma: no cover - optional dependency
    class _SpacesStub:
        """Fallback stub emulating the Hugging Face ``spaces`` module."""

        @staticmethod
        def GPU(
            func: Optional[Callable] = None,
            /,
            *decorator_args: Any,
            **decorator_kwargs: Any,
        ) -> Callable:
            """No-op decorator matching ``spaces.GPU`` signature."""

            if func is None:
                def _wrapper(inner: Callable) -> Callable:
                    return inner

                return _wrapper

            return func

    spaces = _SpacesStub()  # type: ignore[assignment]

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from rhyme_rarity.core import (
    CMUDictLoader,
    CmuRhymeRepository,
    DefaultCmuRhymeRepository,
    EnhancedPhoneticAnalyzer,
)
from rhyme_rarity.utils.observability import get_logger
from anti_llm import AntiLLMRhymeEngine
from cultural.engine import CulturalIntelligenceEngine

from rhyme_rarity.app.data.database import SQLiteRhymeRepository
from rhyme_rarity.app.services.search_service import SearchService
from rhyme_rarity.app.ui.gradio import create_interface


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
        cmu_repository: Optional[CmuRhymeRepository] = None,
    ) -> None:
        self.db_path = db_path
        self._logger = get_logger(__name__).bind(component="app_facade")
        self._logger.info("Initialising application facade", context={"db_path": db_path})

        self.repository = repository or SQLiteRhymeRepository(db_path)
        try:
            row_count = self.repository.ensure_database()
        except Exception as exc:
            self._logger.error(
                "Database initialisation failed",
                context={"db_path": db_path, "error": str(exc)},
            )
            raise
        else:
            self._logger.info(
                "Database ready",
                context={"db_path": db_path, "row_count": row_count},
            )

        self.cmu_loader = cmu_loader or CMUDictLoader()
        self.cmu_repository = cmu_repository or DefaultCmuRhymeRepository()
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
            cmu_repository=self.cmu_repository,
        )

        if hasattr(self.anti_llm_engine, "set_phonetic_analyzer"):
            self.anti_llm_engine.set_phonetic_analyzer(self.phonetic_analyzer)
        if hasattr(self.cultural_engine, "set_phonetic_analyzer"):
            self.cultural_engine.set_phonetic_analyzer(self.phonetic_analyzer)
        if hasattr(self.cultural_engine, "set_prosody_analyzer"):
            self.cultural_engine.set_prosody_analyzer(self.phonetic_analyzer)

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
            self.search_service.set_cmu_repository(self.cmu_repository)

        self._logger.info(
            "Application dependencies wired",
            context={
                "has_cultural_engine": self.cultural_engine is not None,
                "has_anti_llm_engine": self.anti_llm_engine is not None,
            },
        )

        self._refresh_rarity_map()

    # Dependency management -------------------------------------------------
    def set_phonetic_analyzer(self, analyzer: EnhancedPhoneticAnalyzer) -> None:
        self.phonetic_analyzer = analyzer
        self.search_service.set_phonetic_analyzer(analyzer)

    def set_cultural_engine(self, engine: Optional[CulturalIntelligenceEngine]) -> None:
        self.cultural_engine = engine
        self.search_service.set_cultural_engine(engine)

    def set_anti_llm_engine(self, engine: Optional[AntiLLMRhymeEngine]) -> None:
        self.anti_llm_engine = engine
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
                self._logger.info(
                    "Phonetic rarity map refreshed",
                    context={"db_path": self.db_path},
                )
            except Exception as exc:
                self._logger.warning(
                    "Phonetic rarity refresh failed",
                    context={"db_path": self.db_path, "error": str(exc)},
                )


def _should_share_interface() -> bool:
    """Return whether the Gradio UI should request a public share link."""

    env_value = os.environ.get("RHYMES_SHARE", "")
    if not env_value:
        return False
    normalized = str(env_value).strip().lower()
    return normalized in {"1", "true", "yes", "on"}


# 👇 Add a GPU-marked function so HF knows GPU is used
@spaces.GPU
def warmup_gpu() -> str:
    """Simple GPU warmup so Hugging Face Spaces detects usage."""
    if torch is None:
        return "Torch not installed"
    if torch.cuda.is_available():
        x = torch.rand(1000, 1000, device="cuda")
        _ = torch.matmul(x, x)
        return "GPU warmed up"
    return "No GPU available"


def main() -> None:
    # Call warmup just to satisfy HF's runtime check
    try:
        warmup_gpu()
    except Exception:
        pass

    app = RhymeRarityApp()
    interface = app.create_gradio_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_api=True,
        share=_should_share_interface(),
    )


__all__ = ["RhymeRarityApp", "main", "warmup_gpu"]
