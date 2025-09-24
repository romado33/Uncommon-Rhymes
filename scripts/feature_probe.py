#!/usr/bin/env python3
"""CLI helper to exercise RhymeRarity search features for competitive analysis."""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional, Sequence

from rhyme_rarity.app.app import RhymeRarityApp


def _parse_list(values: Optional[Sequence[str]]) -> List[str]:
    """Normalize CLI list arguments.

    Supports comma-separated strings so auditors can pass values as
    ``--cultural golden-era,underground`` or as separate tokens.
    """

    if not values:
        return []

    items: List[str] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, str):
            parts = [part.strip() for part in value.split(",")]
            items.extend(part for part in parts if part)
        else:
            items.append(str(value))
    return items


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Probe the RhymeRarity search stack with configurable filters to "
            "mirror competitor behaviour."
        )
    )
    parser.add_argument("word", help="Source word or phrase to analyse.")
    parser.add_argument(
        "--database",
        default="patterns.db",
        help="Path to the SQLite database (defaults to patterns.db).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of results to request from each source.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.7,
        help="Minimum confidence threshold applied to all sources.",
    )
    parser.add_argument(
        "--cultural",
        nargs="*",
        metavar="LABEL",
        help="Cultural significance labels (space or comma separated).",
    )
    parser.add_argument(
        "--genres",
        nargs="*",
        metavar="GENRE",
        help="Genre filters (space or comma separated).",
    )
    parser.add_argument(
        "--rhyme-types",
        nargs="*",
        metavar="TYPE",
        help="Allowed rhyme types (e.g., perfect, near, slant).",
    )
    parser.add_argument(
        "--bradley",
        nargs="*",
        metavar="DEVICE",
        help="Limit by Bradley rhyme device categories.",
    )
    parser.add_argument(
        "--sources",
        nargs="*",
        metavar="SOURCE",
        help=(
            "Restrict result sources (phonetic, cultural, anti-llm). "
            "Defaults to all sources when omitted."
        ),
    )
    parser.add_argument(
        "--max-line-distance",
        type=int,
        help="Filter cultural matches by maximum line distance.",
    )
    parser.add_argument(
        "--min-syllables",
        type=int,
        help="Minimum syllable span for rhyme candidates.",
    )
    parser.add_argument(
        "--max-syllables",
        type=int,
        help="Maximum syllable span for rhyme candidates.",
    )
    parser.add_argument(
        "--require-internal",
        action="store_true",
        help="Require internal rhyme matches when supported by data.",
    )
    parser.add_argument(
        "--min-rarity",
        type=float,
        help="Minimum rarity score (anti-LLM engine).",
    )
    parser.add_argument(
        "--min-stress",
        type=float,
        help="Minimum stress alignment score.",
    )
    parser.add_argument(
        "--cadence-focus",
        help="Target cadence focus label (e.g., swing, triplet).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of formatted text (good for logging).",
    )
    parser.add_argument(
        "--pretty-json",
        action="store_true",
        help="Indent JSON output for readability (implies --json).",
    )
    parser.add_argument(
        "--show-params",
        action="store_true",
        help="Print the resolved parameter set before executing the search.",
    )
    return parser


def _resolve_params(namespace: argparse.Namespace) -> dict:
    """Translate CLI arguments to `SearchService.search_rhymes` keyword args."""

    params = {
        "source_word": namespace.word,
        "limit": namespace.limit,
        "min_confidence": namespace.min_confidence,
        "cultural_significance": _parse_list(namespace.cultural),
        "genres": _parse_list(namespace.genres),
        "allowed_rhyme_types": _parse_list(namespace.rhyme_types),
        "bradley_devices": _parse_list(namespace.bradley),
        "result_sources": _parse_list(namespace.sources),
        "max_line_distance": namespace.max_line_distance,
        "min_syllables": namespace.min_syllables,
        "max_syllables": namespace.max_syllables,
        "require_internal": namespace.require_internal,
        "min_rarity": namespace.min_rarity,
        "min_stress_alignment": namespace.min_stress,
        "cadence_focus": namespace.cadence_focus,
    }

    cleaned: dict = {}
    for key, value in params.items():
        if value in (None, [], {}):
            continue
        cleaned[key] = value
    return cleaned


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    params = _resolve_params(args)

    if args.show_params:
        print("Resolved parameters:")
        print(json.dumps(params, indent=2, sort_keys=True))
        print()

    app = RhymeRarityApp(db_path=args.database)
    results = app.search_rhymes(**params)

    if args.pretty_json or args.json:
        indent = 2 if args.pretty_json else None
        json.dump(results, sys.stdout, indent=indent, ensure_ascii=False, sort_keys=True)
        sys.stdout.write("\n")
        return 0

    formatted = app.format_rhyme_results(args.word, results)
    print(formatted)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
