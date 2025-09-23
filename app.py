#!/usr/bin/env python3
"""CLI entry point wiring together the RhymeRarity layers."""

from rhyme_rarity.app.app import RhymeRarityApp, main

__all__ = ["RhymeRarityApp", "main"]

if __name__ == "__main__":
    main()
