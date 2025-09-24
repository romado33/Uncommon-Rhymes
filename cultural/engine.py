"""Cultural intelligence engine."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional, Set

from rhyme_rarity.utils.profile import normalize_profile_dict

from .analytics import (
    aggregate_cultural_distribution,
    approximate_rhyme_signature,
    cultural_rarity_score,
    derive_rhyme_signatures,
    estimate_syllables,
    evaluate_rhyme_alignment,
    phrase_syllable_vector,
    prosody_profile,
)
from .profiles import (
    ArtistProfile,
    CulturalContext,
    build_artist_profiles,
    build_cultural_categories,
    build_era_classifications,
    build_regional_mappings,
    generate_dynamic_profile,
)

class CulturalIntelligenceEngine:
    """
    Enhanced cultural intelligence system providing authentic attribution
    and cultural context that LLMs fundamentally cannot access
    """
    
    def __init__(self, db_path: str = "patterns.db", phonetic_analyzer: Optional[Any] = None):
        self.db_path = db_path
        self.phonetic_analyzer = phonetic_analyzer
        self._connection: Optional[sqlite3.Connection] = None
        
        # Initialize cultural intelligence components
        self.artist_profiles = build_artist_profiles()
        self.cultural_categories = build_cultural_categories()
        self.era_classifications = build_era_classifications()
        self.regional_mappings = build_regional_mappings()
        
        # Performance tracking
        self.cultural_stats = {
            'patterns_analyzed': 0,
            'artists_profiled': 0,
            'cultural_contexts_generated': 0,
            'regional_patterns_identified': 0
        }
        
        print("🎯 Enhanced Cultural Database Engine initialized")
        print("Providing authentic cultural attribution beyond LLM capabilities")

    def _get_connection(self) -> sqlite3.Connection:
        connection = self._connection
        if connection is None:
            connection = sqlite3.connect(self.db_path)
            self._connection = connection
        return connection

    @contextmanager
    def _cursor(self) -> Generator[sqlite3.Cursor, None, None]:
        cursor = self._get_connection().cursor()
        try:
            yield cursor
        finally:
            cursor.close()

    def close(self) -> None:
        connection = self._connection
        if connection is not None:
            connection.close()
            self._connection = None

    def __del__(self) -> None:  # pragma: no cover - defensive cleanup
        try:
            self.close()
        except Exception:
            pass
 
    def set_phonetic_analyzer(self, analyzer: Any) -> None:
        """Attach or replace the phonetic analyzer for phonetic validation."""
        self.phonetic_analyzer = analyzer

    def _estimate_syllables(self, word: str) -> int:
        if not word:
            return 0
        return estimate_syllables(word, self.phonetic_analyzer)

    def _approximate_rhyme_signature(self, word: str) -> Set[str]:
        """Create lightweight rhyme signatures when CMU data is unavailable."""
        return approximate_rhyme_signature(word, self._estimate_syllables)

    def derive_rhyme_signatures(self, word: str) -> Set[str]:
        """Return available rhyme signatures for a given word."""
        normalized = (word or "").strip().lower()
        if not normalized:
            return set()
        analyzer = getattr(self, "phonetic_analyzer", None)
        return derive_rhyme_signatures(
            word,
            analyzer,
            lambda w: approximate_rhyme_signature(w, self._estimate_syllables),
        )

    def _classify_era(self, start_year: int) -> str:
        for era_name, info in self.era_classifications.items():
            years = info.get("years")
            if isinstance(years, (list, tuple)) and len(years) == 2:
                if years[0] <= start_year <= years[1]:
                    return era_name
        return "modern"

    # ------------------------------------------------------------------
    # Prosody helpers inspired by rhythm-focused scholarship
    # ------------------------------------------------------------------

    def _phrase_syllable_vector(self, text: Optional[str]) -> List[int]:
        return phrase_syllable_vector(text, self._estimate_syllables)

    def _prosody_profile(
        self,
        source_context: Optional[str],
        target_context: Optional[str],
        feature_profile: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return prosody_profile(
            source_context,
            target_context,
            feature_profile,
            self._estimate_syllables,
        )

    def evaluate_rhyme_alignment(
        self,
        source_word: str,
        target_word: str,
        threshold: Optional[float] = None,
        rhyme_signatures: Optional[Set[str]] = None,
        source_context: Optional[str] = None,
        target_context: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        return evaluate_rhyme_alignment(
            analyzer=getattr(self, "phonetic_analyzer", None),
            source_word=source_word,
            target_word=target_word,
            threshold=threshold,
            rhyme_signatures=rhyme_signatures,
            source_context=source_context,
            target_context=target_context,
        )

    def get_cultural_context(self, pattern_data: Dict) -> CulturalContext:
        """Generate comprehensive cultural context for a rhyme pattern"""
        artist = (pattern_data.get('artist') or '').strip().lower()
        song = pattern_data.get('song', '')
        
        # Get artist profile
        profile = self.artist_profiles.get(artist)
        if not profile:
            profile = generate_dynamic_profile(artist)
        
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
    


    def get_cultural_rarity_score(self, context: CulturalContext) -> float:
        """Calculate cultural rarity score based on context"""
        return cultural_rarity_score(context)

    def find_cultural_patterns(self, source_word: str, cultural_filter: Optional[str] = None, 
                             limit: int = 20) -> List[Dict]:
        """Find patterns with specific cultural characteristics"""
        try:
            with self._cursor() as cursor:
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
            with self._cursor() as cursor:
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
            with self._cursor() as cursor:
                cursor.execute(
                    "SELECT cultural_significance FROM song_rhyme_patterns"
                )
                cultural_rows = [
                    {"cultural_significance": row[0]}
                    for row in cursor.fetchall()
                    if row and row[0]
                ]
                cultural_dist = aggregate_cultural_distribution(cultural_rows)

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
