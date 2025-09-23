#!/usr/bin/env python3
"""
Module 3: Enhanced Cultural Database Engine for RhymeRarity
Handles cultural intelligence, artist attribution, and advanced database operations
Part of the RhymeRarity system deployed on Hugging Face
"""

import sqlite3
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass
import re

@dataclass
class CulturalContext:
    """Represents cultural context for a rhyme pattern"""
    artist: str
    song: str
    genre: str
    era: str
    cultural_significance: str
    regional_origin: str
    style_characteristics: List[str]

@dataclass
class ArtistProfile:
    """Comprehensive artist profile for cultural attribution"""
    name: str
    primary_genre: str
    secondary_genres: List[str]
    active_years: Tuple[int, int]
    cultural_impact: str
    signature_styles: List[str]
    regional_influence: str
    pattern_count: int

class CulturalIntelligenceEngine:
    """
    Enhanced cultural intelligence system providing authentic attribution
    and cultural context that LLMs fundamentally cannot access
    """
    
    def __init__(self, db_path: str = "patterns.db", phonetic_analyzer: Optional[Any] = None):
        self.db_path = db_path
        self.phonetic_analyzer = phonetic_analyzer
        
        # Initialize cultural intelligence components
        self.artist_profiles = self._initialize_artist_profiles()
        self.cultural_categories = self._initialize_cultural_categories()
        self.era_classifications = self._initialize_era_classifications()
        self.regional_mappings = self._initialize_regional_mappings()
        
        # Performance tracking
        self.cultural_stats = {
            'patterns_analyzed': 0,
            'artists_profiled': 0,
            'cultural_contexts_generated': 0,
            'regional_patterns_identified': 0
        }
        
        print("🎯 Enhanced Cultural Database Engine initialized")
        print("Providing authentic cultural attribution beyond LLM capabilities")
 
    def set_phonetic_analyzer(self, analyzer: Any) -> None:
        """Attach or replace the phonetic analyzer for phonetic validation."""
        self.phonetic_analyzer = analyzer

    def _estimate_syllables(self, word: str) -> int:
        """Rough syllable count used for fallback rhyme signatures."""
        if not word:
            return 0
        vowel_groups = re.findall(r"[aeiou]+", word)
        count = len(vowel_groups)
        if word.endswith("e") and count > 1:
            count -= 1
        return max(1, count)

    def _approximate_rhyme_signature(self, word: str) -> Set[str]:
        """Create lightweight rhyme signatures when CMU data is unavailable."""
        cleaned = re.sub(r"[^a-z]", "", (word or "").lower())
        if not cleaned:
            return set()
        vowels = re.findall(r"[aeiou]+", cleaned)
        last_vowel = vowels[-1] if vowels else ""
        ending = cleaned[-3:] if len(cleaned) >= 3 else cleaned
        syllables = self._estimate_syllables(cleaned)
        components = []
        if last_vowel:
            components.append(f"v:{last_vowel}")
        components.append(f"e:{ending}")
        base_signature = "|".join(components)
        signatures = {base_signature}
        if syllables:
            signatures.add("|".join(components + [f"s:{syllables}"]))
        return signatures

    def derive_rhyme_signatures(self, word: str) -> Set[str]:
        """Return available rhyme signatures for a given word."""
        normalized = (word or "").strip().lower()
        if not normalized:
            return set()
        signatures: Set[str] = set()
        analyzer = getattr(self, "phonetic_analyzer", None)
        loader = getattr(analyzer, "cmu_loader", None) if analyzer else None
        if loader is not None:
            try:
                signatures.update({sig for sig in loader.get_rhyme_parts(normalized) if sig})
            except Exception:
                pass
        signatures.update(self._approximate_rhyme_signature(normalized))
        return signatures

    # ------------------------------------------------------------------
    # Prosody helpers inspired by rhythm-focused scholarship
    # ------------------------------------------------------------------

    def _phrase_syllable_vector(self, text: Optional[str]) -> List[int]:
        if not text:
            return []
        tokens = re.findall(r"[a-zA-Z']+", text)
        return [self._estimate_syllables(token.lower()) for token in tokens if token]

    def _prosody_profile(
        self,
        source_context: Optional[str],
        target_context: Optional[str],
        feature_profile: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        source_vector = self._phrase_syllable_vector(source_context)
        target_vector = self._phrase_syllable_vector(target_context)

        total_source = sum(source_vector)
        total_target = sum(target_vector)

        avg_source = total_source / max(len(source_vector), 1)
        avg_target = total_target / max(len(target_vector), 1)

        cadence_ratio = 0.0
        if total_source and total_target:
            cadence_ratio = total_target / total_source

        stress_alignment = None
        if feature_profile:
            stress_alignment = feature_profile.get("stress_alignment")

        complexity_tag = "steady"
        if avg_target >= 3 or avg_source >= 3:
            complexity_tag = "polysyllabic"
        if len(target_vector) >= 3 and any(count >= 4 for count in target_vector):
            complexity_tag = "dense"

        return {
            "source_total_syllables": total_source,
            "target_total_syllables": total_target,
            "source_average_syllables": avg_source,
            "target_average_syllables": avg_target,
            "cadence_ratio": cadence_ratio,
            "stress_alignment": stress_alignment,
            "complexity_tag": complexity_tag,
            "source_vector": source_vector,
            "target_vector": target_vector,
        }

    def evaluate_rhyme_alignment(
        self,
        source_word: str,
        target_word: str,
        threshold: Optional[float] = None,
        rhyme_signatures: Optional[Set[str]] = None,
        source_context: Optional[str] = None,
        target_context: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Validate and score phonetic alignment between two words."""
        normalized_source = (source_word or "").strip().lower()
        normalized_target = (target_word or "").strip().lower()
        if not normalized_source or not normalized_target:
            return None

        analyzer = getattr(self, "phonetic_analyzer", None)
        source_signature_set = {sig for sig in (rhyme_signatures or set()) if sig}
        target_signatures = self.derive_rhyme_signatures(normalized_target)
        signature_matches = (
            sorted(target_signatures.intersection(source_signature_set))
            if source_signature_set
            else []
        )

        if source_signature_set and target_signatures and not signature_matches:
            return None

        similarity: Optional[float] = None
        rarity_value: Optional[float] = None
        combined_value: Optional[float] = None
        rhyme_type: Optional[str] = None
        features: Dict[str, Any] = {}
        feature_profile_dict: Dict[str, Any] = {}
        prosody_profile: Dict[str, Any] = {}

        if analyzer:
            try:
                match = analyzer.analyze_rhyme_pattern(normalized_source, normalized_target)
            except Exception:
                match = None
            if match:
                similarity = float(match.similarity_score)
                features = dict(match.phonetic_features)
                rhyme_type = match.rhyme_type
                profile_obj = getattr(match, "feature_profile", None)
            else:
                profile_obj = None
                try:
                    similarity = float(analyzer.get_phonetic_similarity(normalized_source, normalized_target))
                except Exception:
                    similarity = 0.0
                try:
                    rhyme_type = analyzer.classify_rhyme_type(normalized_source, normalized_target, similarity)
                except Exception:
                    rhyme_type = None
                features = {}
            if profile_obj is None:
                try:
                    profile_obj = analyzer.derive_rhyme_profile(
                        normalized_source,
                        normalized_target,
                        similarity=similarity,
                        rhyme_type=rhyme_type,
                    )
                except Exception:
                    profile_obj = None
            if profile_obj is not None:
                if hasattr(profile_obj, "as_dict"):
                    try:
                        feature_profile_dict = dict(profile_obj.as_dict())
                    except Exception:
                        feature_profile_dict = {}
                elif isinstance(profile_obj, dict):
                    feature_profile_dict = dict(profile_obj)
                else:
                    try:
                        feature_profile_dict = dict(vars(profile_obj))
                    except Exception:
                        feature_profile_dict = {}
            try:
                rarity_value = float(analyzer.get_rarity_score(normalized_target))
            except Exception:
                rarity_value = None
            if rarity_value is not None and similarity is not None:
                try:
                    combined_value = float(analyzer.combine_similarity_and_rarity(similarity, rarity_value))
                except Exception:
                    combined_value = None
            if threshold is not None and similarity is not None:
                effective_threshold = max(0.0, float(threshold) - 0.02)
                if similarity < effective_threshold:
                    return None
        else:
            similarity = None
            rarity_value = None
            combined_value = None
            rhyme_type = None
            features = {}

        prosody_profile = self._prosody_profile(
            source_context,
            target_context,
            feature_profile_dict,
        )

        return {
            "similarity": similarity,
            "rarity": rarity_value,
            "combined": combined_value,
            "rhyme_type": rhyme_type,
            "signature_matches": signature_matches,
            "target_signatures": sorted(target_signatures),
            "features": features,
            "feature_profile": feature_profile_dict,
            "prosody_profile": prosody_profile,
        }

    def _initialize_artist_profiles(self) -> Dict[str, ArtistProfile]:
        """Initialize comprehensive artist profiles for cultural intelligence"""
        return {
            # Hip-Hop Legends
            'eminem': ArtistProfile(
                name="Eminem",
                primary_genre="hip-hop",
                secondary_genres=["rap", "hardcore"],
                active_years=(1996, 2024),
                cultural_impact="lyrical_genius",
                signature_styles=["multi_syllable", "internal_rhymes", "wordplay", "storytelling"],
                regional_influence="detroit",
                pattern_count=0
            ),
            'jay-z': ArtistProfile(
                name="Jay-Z",
                primary_genre="hip-hop", 
                secondary_genres=["east_coast", "business_rap"],
                active_years=(1986, 2024),
                cultural_impact="mogul_influence",
                signature_styles=["clever_wordplay", "double_entendres", "flow_mastery"],
                regional_influence="brooklyn",
                pattern_count=0
            ),
            'nas': ArtistProfile(
                name="Nas",
                primary_genre="hip-hop",
                secondary_genres=["east_coast", "conscious"],
                active_years=(1991, 2024),
                cultural_impact="poetic_storyteller",
                signature_styles=["narrative_rhymes", "street_poetry", "complex_metaphors"],
                regional_influence="queens",
                pattern_count=0
            ),
            'kendrick lamar': ArtistProfile(
                name="Kendrick Lamar",
                primary_genre="hip-hop",
                secondary_genres=["west_coast", "conscious", "experimental"],
                active_years=(2003, 2024),
                cultural_impact="modern_prophet",
                signature_styles=["voice_modulation", "conceptual_albums", "social_commentary"],
                regional_influence="compton",
                pattern_count=0
            ),
            'drake': ArtistProfile(
                name="Drake",
                primary_genre="hip-hop",
                secondary_genres=["r&b", "pop", "melodic_rap"],
                active_years=(2006, 2024),
                cultural_impact="mainstream_crossover",
                signature_styles=["singing_rap", "emotional_vulnerability", "melodic_hooks"],
                regional_influence="toronto",
                pattern_count=0
            ),
            'j. cole': ArtistProfile(
                name="J. Cole",
                primary_genre="hip-hop",
                secondary_genres=["conscious", "southern"],
                active_years=(2007, 2024),
                cultural_impact="authentic_storyteller",
                signature_styles=["personal_narratives", "social_consciousness", "no_features"],
                regional_influence="fayetteville",
                pattern_count=0
            ),
            # Add more artist profiles as needed
        }
    
    def _initialize_cultural_categories(self) -> Dict[str, Dict]:
        """Initialize cultural significance categories"""
        return {
            'legendary': {
                'description': 'Foundational figures who shaped hip-hop culture',
                'influence_score': 10,
                'rarity_multiplier': 4.0,
                'examples': ['Grandmaster Flash', 'Run-DMC', 'Public Enemy']
            },
            'cultural_icon': {
                'description': 'Artists who transcended music to become cultural phenomena',
                'influence_score': 9,
                'rarity_multiplier': 3.5,
                'examples': ['Tupac', 'Biggie', 'Jay-Z', 'Eminem']
            },
            'lyrical_genius': {
                'description': 'Masters of wordplay and lyrical complexity',
                'influence_score': 8,
                'rarity_multiplier': 3.2,
                'examples': ['Nas', 'Kendrick Lamar', 'MF DOOM', 'Black Thought']
            },
            'regional_pioneer': {
                'description': 'Artists who established regional hip-hop scenes',
                'influence_score': 7,
                'rarity_multiplier': 2.8,
                'examples': ['OutKast', 'UGK', 'Bone Thugs', 'Tech N9ne']
            },
            'underground_legend': {
                'description': 'Respected in hip-hop circles but less mainstream',
                'influence_score': 6,
                'rarity_multiplier': 3.8,
                'examples': ['MF DOOM', 'Aesop Rock', 'Roc Marciano', 'Ka']
            },
            'mainstream': {
                'description': 'Popular artists with wide commercial appeal',
                'influence_score': 5,
                'rarity_multiplier': 1.5,
                'examples': ['Drake', 'Post Malone', 'Travis Scott']
            }
        }
    
    def _initialize_era_classifications(self) -> Dict[str, Dict]:
        """Initialize hip-hop era classifications"""
        return {
            'old_school': {
                'years': (1973, 1983),
                'characteristics': ['party_rap', 'disco_influenced', 'simple_rhymes'],
                'cultural_context': 'Birth of hip-hop in the Bronx'
            },
            'golden_age': {
                'years': (1984, 1993), 
                'characteristics': ['complex_lyricism', 'social_consciousness', 'jazz_samples'],
                'cultural_context': 'Peak of lyrical innovation and cultural impact'
            },
            'east_west_era': {
                'years': (1994, 1999),
                'characteristics': ['coast_rivalry', 'gangsta_rap', 'g_funk'],
                'cultural_context': 'Regional tensions and commercial explosion'
            },
            'millennium': {
                'years': (2000, 2009),
                'characteristics': ['bling_era', 'south_rise', 'ringtone_rap'],
                'cultural_context': 'Southern dominance and commercial peak'
            },
            'blog_era': {
                'years': (2010, 2015),
                'characteristics': ['internet_discovery', 'mixtape_culture', 'conscious_return'],
                'cultural_context': 'Digital revolution in music discovery'
            },
            'streaming_era': {
                'years': (2016, 2024),
                'characteristics': ['playlist_culture', 'melodic_rap', 'viral_moments'],
                'cultural_context': 'Streaming dominance and global expansion'
            }
        }
    
    def _initialize_regional_mappings(self) -> Dict[str, Dict]:
        """Initialize regional hip-hop scene mappings"""
        return {
            'bronx': {'region': 'east_coast', 'significance': 'birthplace', 'style': 'foundational'},
            'brooklyn': {'region': 'east_coast', 'significance': 'lyrical_complex', 'style': 'street_smart'},
            'queens': {'region': 'east_coast', 'significance': 'storytelling', 'style': 'narrative'},
            'detroit': {'region': 'midwest', 'significance': 'technical_skill', 'style': 'aggressive_complex'},
            'chicago': {'region': 'midwest', 'significance': 'drill_innovation', 'style': 'street_conscious'},
            'atlanta': {'region': 'south', 'significance': 'trap_capital', 'style': 'melodic_trap'},
            'houston': {'region': 'south', 'significance': 'screw_music', 'style': 'chopped_screwed'},
            'los_angeles': {'region': 'west_coast', 'significance': 'g_funk', 'style': 'laid_back_gangsta'},
            'compton': {'region': 'west_coast', 'significance': 'hardcore_reality', 'style': 'street_documentary'},
            'oakland': {'region': 'west_coast', 'significance': 'hyphy_movement', 'style': 'energetic_party'},
        }
    
    def get_cultural_context(self, pattern_data: Dict) -> CulturalContext:
        """Generate comprehensive cultural context for a rhyme pattern"""
        artist = (pattern_data.get('artist') or '').strip().lower()
        song = pattern_data.get('song', '')
        
        # Get artist profile
        profile = self.artist_profiles.get(artist)
        if not profile:
            profile = self._generate_dynamic_profile(artist)
        
        # Determine era
        era = self._classify_era(profile.active_years[0])
        
        # Get cultural significance
        cultural_significance = profile.cultural_impact
        
        # Determine regional origin
        regional_origin = profile.regional_influence
        
        context = CulturalContext(
            artist=profile.name,
            song=song,
            genre=profile.primary_genre,
            era=era,
            cultural_significance=cultural_significance,
            regional_origin=regional_origin,
            style_characteristics=profile.signature_styles
        )
        
        self.cultural_stats['cultural_contexts_generated'] += 1
        return context
    
    def _generate_dynamic_profile(self, artist_name: str) -> ArtistProfile:
        """Generate dynamic profile for artists not in static database"""
        # Query database for artist information
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT COUNT(*), genre, cultural_significance 
                FROM song_rhyme_patterns 
                WHERE LOWER(artist) = ?
                GROUP BY genre, cultural_significance
                ORDER BY COUNT(*) DESC
                LIMIT 1
            """, (artist_name,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                pattern_count, genre, cultural_sig = result
                
                return ArtistProfile(
                    name=artist_name.title(),
                    primary_genre=genre or "hip-hop",
                    secondary_genres=["rap"],
                    active_years=(2000, 2024),  # Default range
                    cultural_impact=cultural_sig or "mainstream",
                    signature_styles=["contemporary"],
                    regional_influence="unknown",
                    pattern_count=pattern_count
                )
        except sqlite3.Error:
            pass
        
        # Return default profile
        return ArtistProfile(
            name=artist_name.title(),
            primary_genre="hip-hop",
            secondary_genres=["rap"],
            active_years=(2000, 2024),
            cultural_impact="emerging",
            signature_styles=["contemporary"],
            regional_influence="unknown",
            pattern_count=0
        )
    
    def _classify_era(self, start_year: int) -> str:
        """Classify artist era based on career start year"""
        for era_name, era_info in self.era_classifications.items():
            start, end = era_info['years']
            if start <= start_year <= end:
                return era_name
        return "streaming_era"  # Default to current era
    
    def get_cultural_rarity_score(self, context: CulturalContext) -> float:
        """Calculate cultural rarity score based on context"""
        base_score = 1.0
        
        # Cultural significance multiplier
        significance_info = self.cultural_categories.get(context.cultural_significance, {})
        significance_multiplier = significance_info.get('rarity_multiplier', 1.0)
        
        # Era rarity (older eras are rarer)
        era_bonus = {
            'old_school': 2.5,
            'golden_age': 2.2,
            'east_west_era': 1.8,
            'millennium': 1.4,
            'blog_era': 1.2,
            'streaming_era': 1.0
        }
        
        era_multiplier = era_bonus.get(context.era, 1.0)
        
        # Regional rarity (underground scenes are rarer)
        regional_info = self.regional_mappings.get(context.regional_origin.lower(), {})
        regional_multiplier = 1.5 if regional_info.get('significance') == 'underground' else 1.0
        
        final_score = base_score * significance_multiplier * era_multiplier * regional_multiplier
        return min(final_score, 5.0)  # Cap at maximum rarity
    
    def find_cultural_patterns(self, source_word: str, cultural_filter: Optional[str] = None, 
                             limit: int = 20) -> List[Dict]:
        """Find patterns with specific cultural characteristics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = """
            SELECT target_word, artist, song_title, confidence_score, cultural_significance
            FROM song_rhyme_patterns 
            WHERE source_word = ? 
              AND source_word != target_word
            """
            
            params = [source_word]
            
            if cultural_filter:
                query += " AND cultural_significance = ?"
                params.append(cultural_filter)
            
            query += " ORDER BY confidence_score DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            results = cursor.fetchall()
            conn.close()
            
            cultural_patterns = []
            for target, artist, song, confidence, cultural_sig in results:
                pattern_data = {
                    'source_word': source_word,
                    'target_word': target,
                    'artist': artist,
                    'song': song,
                    'confidence': confidence,
                    'cultural_significance': cultural_sig
                }
                
                context = self.get_cultural_context(pattern_data)
                rarity_score = self.get_cultural_rarity_score(context)
                
                pattern_data['cultural_context'] = context
                pattern_data['cultural_rarity'] = rarity_score
                
                cultural_patterns.append(pattern_data)
            
            return cultural_patterns
            
        except sqlite3.Error as e:
            print(f"Database error in cultural engine: {e}")
            return []
    
    def get_artist_signature_patterns(self, artist_name: str, limit: int = 20) -> List[Dict]:
        """Get signature rhyme patterns for a specific artist"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT source_word, target_word, song_title, confidence_score, pattern
                FROM song_rhyme_patterns 
                WHERE LOWER(artist) = ?
                  AND source_word != target_word
                  AND confidence_score >= 0.8
                ORDER BY confidence_score DESC
                LIMIT ?
            """, (artist_name.lower(), limit))
            
            results = cursor.fetchall()
            conn.close()
            
            signature_patterns = []
            for source, target, song, confidence, pattern in results:
                signature_patterns.append({
                    'source_word': source,
                    'target_word': target,
                    'song': song,
                    'pattern': pattern,
                    'confidence': confidence,
                    'artist': artist_name
                })
            
            return signature_patterns
            
        except sqlite3.Error as e:
            print(f"Database error getting artist patterns: {e}")
            return []
    
    def analyze_cultural_distribution(self) -> Dict:
        """Analyze cultural distribution in the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Cultural significance distribution
            cursor.execute("""
                SELECT cultural_significance, COUNT(*) 
                FROM song_rhyme_patterns 
                GROUP BY cultural_significance 
                ORDER BY COUNT(*) DESC
            """)
            cultural_dist = dict(cursor.fetchall())
            
            # Artist distribution
            cursor.execute("""
                SELECT artist, COUNT(*) 
                FROM song_rhyme_patterns 
                GROUP BY artist 
                ORDER BY COUNT(*) DESC 
                LIMIT 20
            """)
            artist_dist = dict(cursor.fetchall())
            
            # Genre distribution
            cursor.execute("""
                SELECT genre, COUNT(*) 
                FROM song_rhyme_patterns 
                GROUP BY genre 
                ORDER BY COUNT(*) DESC
            """)
            genre_dist = dict(cursor.fetchall())
            
            conn.close()
            
            return {
                'cultural_significance': cultural_dist,
                'top_artists': artist_dist,
                'genre_distribution': genre_dist,
                'total_patterns': sum(cultural_dist.values())
            }
            
        except sqlite3.Error as e:
            print(f"Database error in cultural analysis: {e}")
            return {}
    
    def get_performance_stats(self) -> Dict:
        """Get cultural intelligence engine performance statistics"""
        return {
            'cultural_stats': self.cultural_stats.copy(),
            'artist_profiles_loaded': len(self.artist_profiles),
            'cultural_categories': len(self.cultural_categories),
            'era_classifications': len(self.era_classifications),
            'regional_mappings': len(self.regional_mappings)
        }

# Example usage and testing
if __name__ == "__main__":
    engine = CulturalIntelligenceEngine()
    
    # Test cultural context generation
    test_pattern = {
        'artist': 'eminem',
        'song': 'Lose Yourself',
        'source_word': 'mind',
        'target_word': 'find'
    }
    
    print("🎯 Testing Cultural Intelligence Engine:")
    print("=" * 50)
    
    context = engine.get_cultural_context(test_pattern)
    print(f"Cultural Context for Eminem:")
    print(f"  • Era: {context.era}")
    print(f"  • Cultural Impact: {context.cultural_significance}")
    print(f"  • Regional Origin: {context.regional_origin}")
    print(f"  • Style Characteristics: {context.style_characteristics}")
    
    rarity_score = engine.get_cultural_rarity_score(context)
    print(f"  • Cultural Rarity Score: {rarity_score:.2f}")
    
    # Test cultural pattern finding
    cultural_patterns = engine.find_cultural_patterns("love", limit=3)
    print(f"\nCultural patterns for 'love':")
    for pattern in cultural_patterns:
        print(f"  • '{pattern['target_word']}' - {pattern['artist']} (rarity: {pattern['cultural_rarity']:.2f})")
    
    # Performance stats
    print(f"\n📊 Performance Stats:")
    stats = engine.get_performance_stats()
    for key, value in stats.items():
        print(f"  • {key}: {value}")
    
    print("\n✅ Module 3 Enhanced Cultural Database ready for integration")
