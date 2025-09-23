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
            normalized = search_service.normalize_source_name(raw_label)
            if normalized:
                normalized_cultural_labels.add(normalized)

    genre_options: List[str] = []
    try:
        for value in repository.get_cultural_significance_labels():
            normalized = search_service.normalize_source_name(value)
            if normalized:
                normalized_cultural_labels.add(normalized)
        genre_options = repository.get_genres()
    except Exception:
        genre_options = []

    cultural_options = sorted(normalized_cultural_labels)

    with gr.Blocks(
        title="RhymeRarity - Advanced Rhyme Generator",
        theme=gr.themes.Soft(),
    ) as interface:
        gr.Markdown(
            "## 🎵 RhymeRarity - Advanced Rhyme Generator\n"
            "Discover culturally grounded rhyme pairs sourced from authentic hip-hop lyrics."
        )

        with gr.Row(equal_height=True):
            with gr.Column(scale=5, min_width=360):
                word_input = gr.Textbox(
                    label="Word to Find Rhymes For",
                    placeholder="Enter a word (e.g., love, mind, flow, money)",
                    lines=1,
                )

                with gr.Accordion("Advanced filters", open=False):
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
                                info="Filter by cultural importance (labels such as classic, cultural-icon, underground)",
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

                search_btn = gr.Button("🔍 Find Rhymes", variant="primary", size="lg")
                gr.Markdown(
                    "💡 Enter a word and adjust the filters, then press **Find Rhymes** to discover new lyric pairings."
                )

            with gr.Column(scale=5, min_width=360):
                gr.Markdown("### Rhyme Results")
                output = gr.Markdown(
                    value="Start by entering a word on the left and click **Find Rhymes**.",
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

    return interface

