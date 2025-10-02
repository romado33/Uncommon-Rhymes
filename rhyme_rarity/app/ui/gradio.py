"""User interface assembly for the Gradio front-end."""

from __future__ import annotations

import concurrent.futures
import time
from typing import Any, Dict, List, Sequence, Set

import gradio as gr

from ..data.database import SQLiteRhymeRepository
from ..services.search_service import SearchService
from src.reverse import rhyme_from_phrase


def _ensure_list(value) -> List:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [item for item in value if item not in (None, "", [])]
    if isinstance(value, str):
        return [value] if value else []
    return [value]


def _format_live_events(snapshot: Dict[str, Any]) -> str:
    """Return a markdown representation of live telemetry events."""

    if not snapshot:
        return ""

    events = snapshot.get("events") or []
    counters = snapshot.get("counters") or {}

    if not events and not counters:
        return ""

    output: List[str] = ["#### Live search activity"]

    if events:
        output.append("")
        recent_events = events[-8:]
        for event in recent_events:
            name = str(event.get("name", "event"))
            duration = event.get("duration")
            metadata = event.get("metadata") or {}
            meta_chunks = [f"{key}={value}" for key, value in metadata.items()]
            meta_suffix = f" – {', '.join(meta_chunks)}" if meta_chunks else ""
            if isinstance(duration, (float, int)):
                output.append(f"- `{name}` took {float(duration):.2f}s{meta_suffix}")
            else:
                output.append(f"- `{name}`{meta_suffix}")

    if counters:
        output.append("")
        output.append("**Counters**")
        counter_chunks = [f"`{key}`: {value}" for key, value in counters.items()]
        output.append(", ".join(counter_chunks))

    return "\n".join(output)


def create_interface(
    search_service: SearchService,
    repository: SQLiteRhymeRepository,
) -> gr.Blocks:
    """Construct the interactive Gradio Blocks UI."""

    def search_interface(
        word: str,
        max_results: int,
        min_conf: float,
        slant_strength: float,
        cultural_filter: Sequence[str] | None,
        genre_filter: Sequence[str] | None,
        rhyme_type_filter: Sequence[str] | None,
    ):
        """Execute a search while streaming telemetry updates to the UI."""

        default_results_message = (
            "Start by entering a word on the left and click **Find Rhymes**."
        )

        if not word or not word.strip():
            return (
                "Please enter a word to find rhymes for.",
                "_Enter a term above to begin tracking telemetry._",
                default_results_message,
            )

        telemetry_available = hasattr(search_service, "get_latest_telemetry")
        log_placeholder = (
            "_Waiting for telemetry updates..._"
            if telemetry_available
            else "_Telemetry instrumentation is not available for this search._"
        )
        last_rendered_log = log_placeholder if telemetry_available else ""

        yield (
            "Starting search...",
            log_placeholder,
            default_results_message,
        )

        telemetry_snapshot: Dict[str, Any] = {}
        rhymes: Dict[str, Any] | None = None
        error_message: str | None = None
        start_time = time.perf_counter()

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    search_service.search_rhymes,
                    word,
                    limit=max_results,
                    min_confidence=min_conf,
                    cultural_significance=_ensure_list(cultural_filter),
                    genres=_ensure_list(genre_filter),
                    allowed_rhyme_types=_ensure_list(rhyme_type_filter),
                    slant_strength=slant_strength,
                )

                while True:
                    try:
                        rhymes = future.result(timeout=0.25)
                        break
                    except concurrent.futures.TimeoutError:
                        elapsed = time.perf_counter() - start_time
                        status_message = f"Searching… {elapsed:.1f}s elapsed"

                        if telemetry_available:
                            telemetry_snapshot = search_service.get_latest_telemetry()
                            events = telemetry_snapshot.get("events") or []
                            if events:
                                latest_event = str(events[-1].get("name", "event"))
                                status_message += f" | Last event: {latest_event}"
                            log_markdown = _format_live_events(telemetry_snapshot)
                            if log_markdown:
                                last_rendered_log = log_markdown

                        yield (
                            status_message,
                            last_rendered_log or log_placeholder,
                            default_results_message,
                        )
                        time.sleep(0.25)

        except Exception as exc:  # pragma: no cover - surface UI level failures
            error_message = str(exc)

        if error_message:
            yield (
                f"Search failed: {error_message}",
                last_rendered_log or log_placeholder,
                default_results_message,
            )
            return

        if rhymes is None:
            yield (
                "Search failed: no results returned.",
                last_rendered_log or log_placeholder,
                default_results_message,
            )
            return

        elapsed = time.perf_counter() - start_time
        status_message = f"Search completed in {elapsed:.2f}s"

        if telemetry_available:
            telemetry_snapshot = search_service.get_latest_telemetry()
            final_log = _format_live_events(telemetry_snapshot)
            if final_log:
                last_rendered_log = final_log

        formatted = search_service.format_rhyme_results(word, rhymes)
        yield (
            status_message,
            last_rendered_log or log_placeholder,
            formatted,
        )

    def _phrase_query(
        phrase: str,
        slant_value: float,
        allow_propers: bool,
        db_path: str,
    ):
        singles, phrases = rhyme_from_phrase(
            phrase,
            db_path=db_path,
            slant_strength=slant_value,
            allow_propers=allow_propers,
            limit=100,
        )
        single_rows = [
            {
                "word": entry.word,
                "tier": entry.tier,
                "score": round(entry.score, 4),
            }
            for entry in singles
        ]
        phrase_rows = [
            {
                "phrase": entry.phrase,
                "last_word": entry.last_word,
                "tier": entry.tier,
                "score": round(entry.final_score, 4),
                "freq": entry.freq,
            }
            for entry in phrases
        ]
        explanation = ""
        if singles:
            top = singles[0]
            detail = top.explanation
            explanation = (
                f"**Top single-word**: `{top.word}`\n\n"
                f"- Tier: **{getattr(detail, 'tier', 'slant')}**\n"
                f"- Rime: `{getattr(detail, 'rime', '')}`\n"
                f"- Vowel match: {getattr(detail, 'vowel_match', '')}\n"
                f"- Coda match: {getattr(detail, 'coda_match', '')}\n"
                f"- Stress: {getattr(detail, 'stress_note', '')}\n"
                f"- Score: {getattr(detail, 'score', 0.0):.3f}"
            )
        return single_rows, phrase_rows, explanation

    normalized_cultural_labels: Set[str] = set()
    cultural_engine = getattr(search_service, "cultural_engine", None)
    if cultural_engine:
        for raw_label in getattr(cultural_engine, "cultural_categories", {}).keys():
            normalized = search_service.normalize_filter_label(raw_label)
            if normalized:
                normalized_cultural_labels.add(normalized)

    genre_options: List[str] = []
    try:
        for value in repository.get_cultural_significance_labels():
            normalized = search_service.normalize_filter_label(value)
            if normalized:
                normalized_cultural_labels.add(normalized)
        genre_options = repository.get_genres()
    except Exception:
        genre_options = []

    cultural_options = sorted(normalized_cultural_labels)

    interface_css = """
    .rr-container {max-width: 1200px; margin: 0 auto; gap: 24px;}
    .rr-hero {text-align: center; padding-bottom: 16px;}
    .rr-hero h2 {font-size: 2.1rem; margin-bottom: 0.25rem;}
    .rr-hero p {color: #4b5563; font-size: 1rem;}
    .rr-panel {border: 1px solid rgba(15, 23, 42, 0.08); border-radius: 16px; background: #ffffff; padding: 24px; box-shadow: 0 12px 32px rgba(15, 23, 42, 0.08);}
    .rr-panel h3 {margin-top: 0; font-weight: 700; letter-spacing: 0.02em;}
    .rr-section {width: 100%;}
    .rr-input-panel .gr-form {gap: 16px;}
    .rr-search-panel {width: 100%;}
    .rr-button {width: 100%; font-weight: 600;}
    .rr-tip {color: #4b5563; font-size: 0.92rem; margin-top: 8px;}
    .rr-results-panel {display: flex; flex-direction: column; gap: 16px; min-height: 420px;}
    .rr-results-panel .gr-markdown {flex: 1; background: #f8fafc; border-radius: 12px; padding: 16px 18px; border: 1px solid rgba(15, 23, 42, 0.05); white-space: normal;}
    .rr-status-card {background: #f1f5f9; border-radius: 12px; border: 1px solid rgba(15, 23, 42, 0.05); padding: 16px 18px;}
    .rr-status-message {margin: 0; color: #0f172a; font-weight: 600;}
    .rr-log {background: #ffffff; border-radius: 12px; border: 1px solid rgba(15, 23, 42, 0.06); padding: 16px 18px; max-height: 220px; overflow-y: auto;}
    .rr-source-summary {margin-bottom: 12px;}
    .rr-source-summary h3 {margin: 0 0 4px; font-size: 1.35rem;}
    .rr-source-summary p {margin: 0; color: #334155;}
    .rr-source-summary ul {margin: 8px 0 0; padding-left: 18px; color: #475569;}
    .rr-results-grid {display: flex; flex-direction: column; gap: 16px;}
    .rr-result-row {display: flex; flex-wrap: wrap; gap: 16px;}
    .rr-result-card {flex: 1 1 280px; background: #ffffff; border-radius: 12px; border: 1px solid rgba(15, 23, 42, 0.05); padding: 16px; box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.6);}
    .rr-result-card h4 {margin-top: 0; margin-bottom: 12px; font-size: 1.05rem;}
    .rr-result-card.rr-span-2 {flex-basis: 100%;}
    .rr-rhyme-list {list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 12px;}
    .rr-rhyme-entry {background: #f8fafc; border-radius: 10px; padding: 12px 14px; border: 1px solid rgba(15, 23, 42, 0.06);}
    .rr-rhyme-line {display: flex; flex-wrap: wrap; gap: 8px; align-items: baseline;}
    .rr-rhyme-term {font-weight: 700; letter-spacing: 0.04em; color: #0f172a;}
    .rr-rhyme-details-inline {color: #475569; font-size: 0.95rem;}
    .rr-rhyme-tier {margin-top: 6px; font-size: 0.88rem; font-weight: 600; color: #1f2937; letter-spacing: 0.01em;}
    .rr-rhyme-rationale {margin-top: 4px; color: #475569; font-size: 0.85rem;}
    .rr-rhyme-context {margin-top: 6px; color: #1e293b; font-size: 0.92rem; line-height: 1.35;}
    .rr-empty {margin: 0; color: #94a3b8; font-style: italic;}
    .rr-accordion .gr-accordion-label {font-weight: 600;}
    """

    with gr.Blocks(
        title="RhymeRarity - Advanced Rhyme Generator",
        theme=gr.themes.Soft(),
        css=interface_css,
    ) as interface:
        with gr.Column(elem_classes=["rr-container"]):
            gr.Markdown(
                "<h2>🎵 RhymeRarity</h2>\n"
                "<p>Discover perfect matches, slant surprises, and rap-inspired multi-word rhymes.</p>",
                elem_classes=["rr-hero"],
            )
            gr.Markdown(
                "**Confidence score:** Weighted 0–1 blend (65% phonetic match, 35% rarity) showing rhyme strength.\n"
                "**Rarity score:** 0–1 indicator of how uncommon the word or cultural context is in the corpus.",
                elem_classes=["rr-tip"],
            )

            with gr.Tabs():
                with gr.Tab("Word → Rhymes"):
                    with gr.Column(elem_classes=["rr-section"]):
                        with gr.Group(elem_classes=["rr-panel", "rr-input-panel", "rr-search-panel"]):
                            gr.Markdown("### Search settings")
                            word_input = gr.Textbox(
                                label="Word to Find Rhymes For",
                                placeholder="Enter a word (e.g., love, mind, flow, money)",
                                lines=1,
                            )

                            with gr.Accordion(
                                "Advanced filters",
                                open=False,
                                elem_classes=["rr-accordion"],
                            ):
                                with gr.Row():
                                    max_results = gr.Slider(
                                        minimum=5,
                                        maximum=50,
                                        value=15,
                                        step=1,
                                        label="Max Results",
                                    )

                                    min_confidence = gr.Slider(
                                        minimum=0.5,
                                        maximum=1.0,
                                        value=0.7,
                                        step=0.05,
                                        label="Min Confidence",
                                    )

                                    slant_strength = gr.Slider(
                                        minimum=0.0,
                                        maximum=1.0,
                                        value=1.0,
                                        step=0.05,
                                        label="Slant Strength",
                                        info=(
                                            "1.0 favours the tightest matches; lower values admit looser"
                                            " slant pairs."
                                        ),
                                    )

                                with gr.Row():
                                    with gr.Column(scale=1, min_width=160):
                                        cultural_dropdown = gr.Dropdown(
                                            choices=cultural_options,
                                            multiselect=True,
                                            label="Cultural Significance",
                                            info="Highlight results by their cultural weight",
                                            value=[],
                                        )

                                        genre_dropdown = gr.Dropdown(
                                            choices=genre_options,
                                            multiselect=True,
                                            label="Genre",
                                            info="Limit to specific genres",
                                            value=[],
                                        )

                                    with gr.Column(scale=1, min_width=160):
                                        rhyme_type_dropdown = gr.Dropdown(
                                            choices=["perfect", "near", "slant", "eye", "weak"],
                                            multiselect=True,
                                            label="Rhyme Type",
                                            info="Limit to specific rhyme categories",
                                            value=[],
                                        )

                            search_btn = gr.Button(
                                "🔍 Find Rhymes",
                                variant="primary",
                                size="lg",
                                elem_classes=["rr-button"],
                            )
                            gr.Markdown(
                                "💡 Enter a word, tune the filters, and click **Find Rhymes** to surface perfect, slant, and multi-word pairings.",
                                elem_classes=["rr-tip"],
                            )

                    with gr.Column(elem_classes=["rr-section"]):
                        with gr.Group(elem_classes=["rr-panel", "rr-results-panel"]):
                            gr.Markdown("### Status & telemetry")
                            status_md = gr.Markdown(
                                value="Waiting to start a search…",
                                elem_classes=["rr-status-card", "rr-status-message"],
                            )
                            log_md = gr.Markdown(
                                value="_Live telemetry updates will appear here during a search._",
                                elem_classes=["rr-log"],
                            )
                            gr.Markdown("### Rhyme results")
                            results_md = gr.Markdown(
                                value="Start by entering a word on the left and click **Find Rhymes**.",
                                elem_classes=["rr-results-markdown"],
                            )

                with gr.Tab("Phrase → Rhymes"):
                    with gr.Column(elem_classes=["rr-section"]):
                        with gr.Group(elem_classes=["rr-panel", "rr-input-panel", "rr-search-panel"]):
                            gr.Markdown("### Phrase settings")
                            phrase_in = gr.Textbox(
                                label="Input phrase",
                                placeholder="e.g., him so",
                                lines=1,
                            )
                            slant_ph = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                value=0.6,
                                step=0.05,
                                label="Slant strength (tight → loose)",
                            )
                            allow_prop_ph = gr.Checkbox(
                                value=True,
                                label="Allow proper nouns",
                            )
                            db_path_in = gr.Textbox(
                                value="patterns.db",
                                label="DB path",
                            )
                            run_rev = gr.Button("Find rhymes")

                    with gr.Column(elem_classes=["rr-section"]):
                        with gr.Group(elem_classes=["rr-panel", "rr-results-panel"]):
                            singles_df = gr.Dataframe(label="Single-word rhymes", wrap=True)
                            phrases_df = gr.Dataframe(label="Multi-word rhymes (DB)", wrap=True)
                            why_md = gr.Markdown(label="Why this rhymes")

        search_btn.click(
            fn=search_interface,
            inputs=[
                word_input,
                max_results,
                min_confidence,
                slant_strength,
                cultural_dropdown,
                genre_dropdown,
                rhyme_type_dropdown,
            ],
            outputs=[status_md, log_md, results_md],
        )

        word_input.submit(
            fn=search_interface,
            inputs=[
                word_input,
                max_results,
                min_confidence,
                slant_strength,
                cultural_dropdown,
                genre_dropdown,
                rhyme_type_dropdown,
            ],
            outputs=[status_md, log_md, results_md],
        )

        run_rev.click(
            _phrase_query,
            [phrase_in, slant_ph, allow_prop_ph, db_path_in],
            [singles_df, phrases_df, why_md],
        )

    return interface

