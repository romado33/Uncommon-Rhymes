"""Streamlit front-end for the RhymeRarity project."""

from __future__ import annotations

from typing import List, Sequence, Set, Tuple

import streamlit as st

from rhyme_rarity.app.app import RhymeRarityApp
from rhyme_rarity.app.services.search_service import SearchService
from rhyme_rarity.app.data.database import SQLiteRhymeRepository


@st.cache_resource(show_spinner=False)
def _load_app() -> RhymeRarityApp:
    """Initialise and cache the core application facade."""

    return RhymeRarityApp()


def _prepare_filter_options(
    search_service: SearchService,
    repository: SQLiteRhymeRepository,
) -> Tuple[List[str], List[str]]:
    """Collect filter option values from engines and the database."""

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
        genre_values = repository.get_genres()
        genre_options = [value for value in genre_values if value]
    except Exception as exc:  # pragma: no cover - defensive UI fallback
        st.warning(
            "Unable to load cultural metadata from the database. "
            "Filtering options will be limited.\n\n"
            f"Details: {exc}",
            icon="⚠️",
        )
        genre_options = []

    cultural_options = sorted(normalized_cultural_labels)
    return cultural_options, genre_options


def _render_styles() -> None:
    """Inject custom CSS so results mirror the Gradio styling."""

    interface_css = """
    <style>
    .rr-container {max-width: 1200px; margin: 0 auto; gap: 24px;}
    .rr-hero {text-align: center; padding-bottom: 16px;}
    .rr-hero h2 {font-size: 2.1rem; margin-bottom: 0.25rem;}
    .rr-hero p {color: #4b5563; font-size: 1rem;}
    .rr-panel {border: 1px solid rgba(15, 23, 42, 0.08); border-radius: 16px; background: #ffffff; padding: 24px; box-shadow: 0 12px 32px rgba(15, 23, 42, 0.08);}
    .rr-panel h3 {margin-top: 0; font-weight: 700; letter-spacing: 0.02em;}
    .rr-section {width: 100%;}
    .rr-tip {color: #4b5563; font-size: 0.92rem; margin-top: 8px;}
    .rr-results-panel {display: flex; flex-direction: column; gap: 16px; min-height: 420px;}
    .rr-results-panel .rr-results-markdown {flex: 1; background: #f8fafc; border-radius: 12px; padding: 16px 18px; border: 1px solid rgba(15, 23, 42, 0.05); white-space: normal;}
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
    .rr-rhyme-context {margin-top: 6px; color: #1e293b; font-size: 0.92rem; line-height: 1.35;}
    .rr-empty {margin: 0; color: #94a3b8; font-style: italic;}
    </style>
    """
    st.markdown(interface_css, unsafe_allow_html=True)


def main() -> None:
    """Render the interactive Streamlit experience."""

    st.set_page_config(
        page_title="RhymeRarity - Advanced Rhyme Generator",
        page_icon="🎵",
        layout="wide",
    )
    _render_styles()

    app = _load_app()
    search_service = app.search_service
    repository = app.repository

    if "filter_options" not in st.session_state:
        st.session_state.filter_options = _prepare_filter_options(search_service, repository)

    cultural_options, genre_options = st.session_state.filter_options

    st.markdown(
        "<div class='rr-hero'><h2>🎵 RhymeRarity</h2><p>Discover perfect matches, slant surprises, and rap-inspired multi-word rhymes.</p></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p class='rr-tip'><strong>Confidence score:</strong> Weighted 0–1 blend (65% phonetic match, 35% rarity) showing rhyme strength.<br>"
        "<strong>Rarity score:</strong> 0–1 indicator of how uncommon the word or cultural context is in the corpus.</p>",
        unsafe_allow_html=True,
    )

    with st.form("rhyme_search"):
        word = st.text_input(
            "Word to find rhymes for",
            help="Enter the word you want to rhyme (e.g., love, flow, money).",
        )
        col1, col2 = st.columns(2)
        with col1:
            max_results = st.slider(
                "Max results",
                min_value=5,
                max_value=50,
                value=15,
                step=1,
            )
        with col2:
            min_confidence = st.slider(
                "Minimum confidence",
                min_value=0.5,
                max_value=1.0,
                value=0.7,
                step=0.05,
            )

        col3, col4, col5 = st.columns(3)
        with col3:
            cultural_filter: Sequence[str] = st.multiselect(
                "Cultural significance",
                options=cultural_options,
                help="Highlight results with matching cultural significance tags.",
            )
        with col4:
            genre_filter: Sequence[str] = st.multiselect(
                "Genre",
                options=genre_options,
                help="Limit matches to a specific genre.",
            )
        with col5:
            rhyme_type_filter: Sequence[str] = st.multiselect(
                "Rhyme type",
                options=["perfect", "near", "slant", "eye", "weak"],
                help="Limit matches to specific rhyme categories.",
            )

        submitted = st.form_submit_button("🔍 Find rhymes")

    results_placeholder = st.empty()

    if submitted:
        if not word:
            results_placeholder.info("Please enter a word to find rhymes for.")
            return

        with st.spinner("Searching the rhyme vault..."):
            rhymes = search_service.search_rhymes(
                word,
                limit=max_results,
                min_confidence=min_confidence,
                cultural_significance=list(cultural_filter),
                genres=list(genre_filter),
                allowed_rhyme_types=list(rhyme_type_filter),
            )
            formatted = search_service.format_rhyme_results(word, rhymes)

        results_placeholder.markdown(formatted, unsafe_allow_html=True)
    else:
        results_placeholder.markdown(
            "<div class='rr-panel rr-results-panel'><p class='rr-empty'>Start by entering a word and click <strong>Find rhymes</strong>.</p></div>",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
