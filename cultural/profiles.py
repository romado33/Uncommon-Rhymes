"""Dataclasses and profile initialisation helpers for cultural intelligence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class CulturalContext:
    artist: str
    song: str
    genre: str
    era: str
    cultural_significance: str
    regional_origin: str
    style_characteristics: List[str]
    release_year: Optional[int] = None
    lyrical_context: Optional[str] = None


@dataclass
class ArtistProfile:
    name: str
    primary_genre: str
    secondary_genres: List[str]
    active_years: Tuple[int, int]
    cultural_impact: str
    signature_styles: List[str]
    regional_influence: str
    pattern_count: int


def build_artist_profiles() -> Dict[str, ArtistProfile]:
    return {
        "eminem": ArtistProfile(
            name="Eminem",
            primary_genre="hip-hop",
            secondary_genres=["rap", "hardcore"],
            active_years=(1996, 2024),
            cultural_impact="lyrical_genius",
            signature_styles=[
                "multi_syllable",
                "internal_rhymes",
                "wordplay",
                "storytelling",
            ],
            regional_influence="detroit",
            pattern_count=0,
        ),
        "jay-z": ArtistProfile(
            name="Jay-Z",
            primary_genre="hip-hop",
            secondary_genres=["east_coast", "business_rap"],
            active_years=(1986, 2024),
            cultural_impact="mogul_influence",
            signature_styles=[
                "clever_wordplay",
                "double_entendres",
                "flow_mastery",
            ],
            regional_influence="brooklyn",
            pattern_count=0,
        ),
        "nas": ArtistProfile(
            name="Nas",
            primary_genre="hip-hop",
            secondary_genres=["east_coast", "conscious"],
            active_years=(1991, 2024),
            cultural_impact="poetic_storyteller",
            signature_styles=[
                "narrative_rhymes",
                "street_poetry",
                "complex_metaphors",
            ],
            regional_influence="queens",
            pattern_count=0,
        ),
        "kendrick lamar": ArtistProfile(
            name="Kendrick Lamar",
            primary_genre="hip-hop",
            secondary_genres=["west_coast", "conscious", "experimental"],
            active_years=(2003, 2024),
            cultural_impact="modern_prophet",
            signature_styles=[
                "voice_modulation",
                "conceptual_albums",
                "social_commentary",
            ],
            regional_influence="compton",
            pattern_count=0,
        ),
        "drake": ArtistProfile(
            name="Drake",
            primary_genre="hip-hop",
            secondary_genres=["r&b", "pop", "melodic_rap"],
            active_years=(2006, 2024),
            cultural_impact="mainstream_crossover",
            signature_styles=[
                "singing_rap",
                "emotional_vulnerability",
                "melodic_hooks",
            ],
            regional_influence="toronto",
            pattern_count=0,
        ),
        "j. cole": ArtistProfile(
            name="J. Cole",
            primary_genre="hip-hop",
            secondary_genres=["conscious", "southern"],
            active_years=(2007, 2024),
            cultural_impact="authentic_storyteller",
            signature_styles=[
                "personal_narratives",
                "social_consciousness",
                "no_features",
            ],
            regional_influence="fayetteville",
            pattern_count=0,
        ),
    }


def build_cultural_categories() -> Dict[str, Dict[str, object]]:
    return {
        "legendary": {
            "description": "Foundational figures who shaped hip-hop culture",
            "influence_score": 10,
            "rarity_multiplier": 4.0,
            "examples": ["Grandmaster Flash", "Run-DMC", "Public Enemy"],
        },
        "cultural_icon": {
            "description": "Artists who transcended music to become cultural phenomena",
            "influence_score": 9,
            "rarity_multiplier": 3.5,
            "examples": ["Tupac", "Biggie", "Jay-Z", "Eminem"],
        },
        "classic": {
            "description": "Timeless anthems regarded as essential hip-hop staples",
            "influence_score": 8,
            "rarity_multiplier": 3.0,
            "examples": ["Nas", "Jay-Z", "OutKast"],
        },
        "underground": {
            "description": "Independent artists and scenes with cult followings",
            "influence_score": 7,
            "rarity_multiplier": 2.5,
            "examples": ["MF DOOM", "Aesop Rock", "Little Simz"],
        },
        "regional_movement": {
            "description": "Collectives defined by regional slang and cadence",
            "influence_score": 6,
            "rarity_multiplier": 2.0,
            "examples": ["Three 6 Mafia", "UGK", "Griselda"],
        },
    }


def build_era_classifications() -> Dict[str, Dict[str, object]]:
    return {
        "golden_age": {
            "years": (1985, 1992),
            "description": "Innovative sampling and conscious lyricism",
        },
        "jiggy_era": {
            "years": (1993, 1999),
            "description": "Glossy production, crossover radio dominance",
        },
        "blog_era": {
            "years": (2000, 2009),
            "description": "Mixtapes, internet breakouts, stylistic experimentation",
        },
        "streaming_era": {
            "years": (2010, 2024),
            "description": "Playlist culture, genre fluidity, international influence",
        },
    }


def build_regional_mappings() -> Dict[str, Dict[str, object]]:
    return {
        "east_coast": {
            "cities": ["new york", "philadelphia", "baltimore"],
            "characteristics": ["lyrical_density", "boom_bap", "storytelling"],
        },
        "west_coast": {
            "cities": ["los angeles", "oakland", "compton"],
            "characteristics": ["g-funk", "laid_back_flow", "funk_samples"],
        },
        "southern": {
            "cities": ["atlanta", "houston", "new orleans"],
            "characteristics": ["bounce", "trap", "drawl"],
        },
        "midwest": {
            "cities": ["detroit", "chicago", "st. louis"],
            "characteristics": ["double_time", "melodic_hooks", "lyrical_precision"],
        },
    }


def generate_dynamic_profile(artist_name: str) -> ArtistProfile:
    cleaned = artist_name.strip().title()
    return ArtistProfile(
        name=cleaned,
        primary_genre="hip-hop",
        secondary_genres=[],
        active_years=(2000, 2024),
        cultural_impact="emerging_voice",
        signature_styles=["versatility", "regional_influence"],
        regional_influence="global",
        pattern_count=0,
    )


__all__ = [
    "CulturalContext",
    "ArtistProfile",
    "build_artist_profiles",
    "build_cultural_categories",
    "build_era_classifications",
    "build_regional_mappings",
    "generate_dynamic_profile",
]
