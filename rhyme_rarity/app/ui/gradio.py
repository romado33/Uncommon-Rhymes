"""User interface assembly for the Gradio front-end."""

from __future__ import annotations

from typing import List, Sequence, Set

import gradio as gr

from ..data.database import SQLiteRhymeRepository
from ..services.search_service import SearchService


def _ensure_list(value) -> List:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [item for item in value if item not in (None, "", [])]
    if isinstance(value, str):
        return [value] if value else []
    return [value]


def create_interface(
    search_service: SearchService,
    repository: SQLiteRhymeRepository,
) -> gr.Blocks:
    """Construct the interactive Gradio Blocks UI."""

    def search_interface(
        word: str,
        max_results: int,
        min_conf: float,
        cultural_filter: Sequence[str] | None,
        genre_filter: Sequence[str] | None,
        rhyme_type_filter: Sequence[str] | None,
    ) -> str:
        if not word:
            return "Please enter a word to find rhymes for."

        rhymes = search_service.search_rhymes(
            word,
            limit=max_results,
            min_confidence=min_conf,
            cultural_significance=_ensure_list(cultural_filter),
            genres=_ensure_list(genre_filter),
            allowed_rhyme_types=_ensure_list(rhyme_type_filter),
        )
        return search_service.format_rhyme_results(word, rhymes)

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
                    gr.Markdown("### Rhyme results")
                    output = gr.Markdown(
                        value="Start by entering a word on the left and click **Find Rhymes**.",
                        elem_classes=["rr-results-markdown"],
                    )

        search_btn.click(
            fn=search_interface,
            inputs=[
                word_input,
                max_results,
                min_confidence,
                cultural_dropdown,
                genre_dropdown,
                rhyme_type_dropdown,
            ],
            outputs=output,
        )

        word_input.submit(
            fn=search_interface,
            inputs=[
                word_input,
                max_results,
                min_confidence,
                cultural_dropdown,
                genre_dropdown,
                rhyme_type_dropdown,
            ],
            outputs=output,
        )

    return interface

