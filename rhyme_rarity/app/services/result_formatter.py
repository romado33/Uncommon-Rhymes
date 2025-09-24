"""Result formatting helpers for rhyme discovery."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


class RhymeResultFormatter:
    """Render grouped rhyme results with shared phonetic context."""

    def format_results(self, source_word: str, rhymes: Dict[str, Any]) -> str:
        """Render grouped rhyme results with shared phonetic context."""

        category_order: List[Tuple[str, str]] = [
            ("cmu", "📚 CMU — Uncommon Rhymes"),
            ("anti_llm", "🧠 Anti-LLM — Uncommon Patterns"),
            ("rap_db", "🎤 Rap & Cultural Matches"),
        ]

        def _normalize_source_key(value: Optional[str]) -> str:
            mapping = {
                "phonetic": "cmu",
                "cultural": "rap_db",
                "anti_llm": "anti_llm",
                "anti-llm": "anti_llm",
                "multi_word": "anti_llm",
            }
            if value is None:
                return ""
            return mapping.get(str(value), str(value))

        if not rhymes:
            return f"❌ No rhymes found for '{source_word}'. Try another word or adjust your filters."

        has_entries = any(rhymes.get(key) for key, _ in category_order)
        if not has_entries:
            return f"❌ No rhymes found for '{source_word}'. Try another word or adjust your filters."

        def _resolve_rhyme_type(entry: Dict[str, Any]) -> Optional[str]:
            candidate = entry.get("rhyme_type")
            if not candidate:
                for attr in ("feature_profile", "phonetic_features"):
                    obj = entry.get(attr)
                    if obj is None:
                        continue
                    if isinstance(obj, dict):
                        candidate = obj.get("rhyme_type")
                    elif hasattr(obj, "__dict__"):
                        candidate = vars(obj).get("rhyme_type")
                    else:
                        try:
                            candidate = dict(obj).get("rhyme_type")
                        except Exception:
                            candidate = None
                    if candidate:
                        break
            if not candidate:
                return None
            return str(candidate).replace("_", " ").title()

        def _as_dict(value: Any) -> Dict[str, Any]:
            if isinstance(value, dict):
                return value
            if hasattr(value, "__dict__"):
                try:
                    return dict(vars(value))
                except Exception:
                    return {}
            return {}

        def _format_float(value: Any) -> Optional[str]:
            try:
                return f"{float(value):.2f}"
            except (TypeError, ValueError):
                return None

        def _format_phonetics_line(phonetics: Any) -> Optional[str]:
            if not isinstance(phonetics, dict):
                return None
            parts: List[str] = []
            syllables = phonetics.get("syllables")
            if isinstance(syllables, (int, float)):
                parts.append(f"Syllables: {int(syllables)}")
            stress_display = phonetics.get("stress_pattern_display") or phonetics.get(
                "stress_pattern"
            )
            if stress_display:
                parts.append(f"Stress: {stress_display}")
            meter_hint = phonetics.get("meter_hint")
            foot = phonetics.get("metrical_foot")
            if meter_hint:
                parts.append(f"Meter: {meter_hint}")
            elif foot:
                parts.append(f"Meter: {str(foot).title()}")
            if not parts:
                return None
            return f"Phonetics: {' | '.join(parts)}"

        def _resolve_source_label(
            source_key: str,
            entry: Dict[str, Any],
            parent: Optional[Dict[str, Any]] = None,
        ) -> Optional[str]:
            explicit_source = entry.get("result_source")
            if explicit_source is None and parent is not None:
                explicit_source = parent.get("result_source")
            normalized_source = _normalize_source_key(explicit_source) or source_key
            mapping = {
                "cmu": "CMU Pronouncing Dictionary",
                "anti_llm": "Anti-LLM Pattern Library",
                "rap_db": "Rap & Cultural Archive",
            }
            if normalized_source in mapping:
                return mapping[normalized_source]
            if explicit_source:
                return str(explicit_source)
            return mapping.get(source_key, source_key.replace("_", " ").title())

        def _standard_info_segments(
            subject: Dict[str, Any],
            source_key: str,
            parent: Optional[Dict[str, Any]] = None,
        ) -> List[str]:
            segments: List[str] = []
            rhyme_category = "Phrase" if subject.get("is_multi_word") else "Word"
            segments.append(f"Rhyme Type: {rhyme_category}")

            phonetics_line = _format_phonetics_line(subject.get("target_phonetics") or {})
            if phonetics_line:
                segments.append(phonetics_line)

            rarity_value = subject.get("rarity_score")
            if rarity_value is None:
                rarity_value = subject.get("cultural_rarity")
            if rarity_value is None and parent is not None:
                rarity_value = parent.get("rarity_score") or parent.get("cultural_rarity")
            rarity_formatted = _format_float(rarity_value)
            if rarity_formatted:
                segments.append(f"Rarity: {rarity_formatted}")

            rhyme_label = _resolve_rhyme_type(subject) or (
                _resolve_rhyme_type(parent) if parent is not None else None
            )
            if rhyme_label:
                segments.append(f"Rhyme type: {rhyme_label}")

            confidence_value = subject.get("combined_score")
            if confidence_value is None:
                confidence_value = subject.get("confidence")
            if confidence_value is None and parent is not None:
                confidence_value = parent.get("combined_score") or parent.get("confidence")
            confidence_formatted = _format_float(confidence_value)
            if confidence_formatted:
                segments.append(f"Confidence: {confidence_formatted}")

            source_label = _resolve_source_label(source_key, subject, parent)
            if source_label:
                segments.append(f"Source: {source_label}")

            return [segment for segment in segments if segment]

        lines: List[str] = [
            f"🎯 **TARGET RHYMES for '{source_word.upper()}':**",
            "=" * 50,
            "",
        ]

        source_profile = rhymes.get("source_profile") or {}
        source_phonetics = source_profile.get("phonetics") or {}
        lines.append("🔎 Source profile")
        lines.append(f"   • Word: {source_profile.get('word', source_word)}")

        basic_parts: List[str] = []
        syllables = source_phonetics.get("syllables")
        if isinstance(syllables, (int, float)):
            basic_parts.append(f"Syllables: {int(syllables)}")
        stress_display = source_phonetics.get("stress_pattern_display") or source_phonetics.get(
            "stress_pattern"
        )
        if stress_display:
            basic_parts.append(f"Stress: {stress_display}")
        meter_hint = source_phonetics.get("meter_hint")
        foot = source_phonetics.get("metrical_foot")
        if meter_hint:
            basic_parts.append(f"Meter: {meter_hint}")
        elif foot:
            basic_parts.append(f"Meter: {str(foot).title()}")
        if basic_parts:
            lines.append(f"   • Basic: {' | '.join(basic_parts)}")
        lines.append("")

        for key, header in category_order:
            entries = rhymes.get(key) or []
            if not entries:
                continue

            lines.append(header)
            lines.append("-" * len(header))

            if key == "rap_db":
                for entry in entries:
                    targets = entry.get("grouped_targets") or []
                    if not targets:
                        continue

                    artist = entry.get("artist")
                    song = entry.get("song")
                    if artist and song:
                        artist_segment = f"{artist} — {song}"
                    elif artist:
                        artist_segment = str(artist)
                    elif song:
                        artist_segment = str(song)
                    else:
                        artist_segment = None

                    genre_bits: List[str] = []
                    genre_value = entry.get("genre")
                    if genre_value:
                        genre_bits.append(f"Genre: {genre_value}")
                    group_size = entry.get("group_size")
                    if group_size:
                        try:
                            genre_bits.append(f"Targets: {int(group_size)}")
                        except (TypeError, ValueError):
                            genre_bits.append(f"Targets: {group_size}")

                    for target in targets:
                        pattern_text = target.get("pattern") or target.get("target_word") or "(unknown)"
                        subject_label = f"Pattern: {pattern_text}"

                        line_segments = [subject_label]
                        line_segments.extend(
                            _standard_info_segments(target, key, parent=entry)
                        )

                        metadata_segments: List[str] = []
                        if artist_segment:
                            metadata_segments.append(artist_segment)
                        if genre_bits:
                            metadata_segments.append(" | ".join(genre_bits))

                        hidden_segments: List[str] = []
                        context_info = _as_dict(entry.get("cultural_context"))
                        cultural_bits: List[str] = []
                        for key_name, label in (
                            ("era", "Era"),
                            ("regional_origin", "Region"),
                            ("cultural_significance", "Significance"),
                        ):
                            value = context_info.get(key_name)
                            if not value and key_name == "cultural_significance":
                                value = entry.get("cultural_sig")
                            if value:
                                cultural_bits.append(
                                    f"{label}: {str(value).replace('_', ' ').title()}"
                                )
                        if cultural_bits:
                            hidden_segments.append(
                                f"<!-- • Cultural: {' | '.join(cultural_bits)} -->"
                            )

                        styles = context_info.get("style_characteristics")
                        if isinstance(styles, (list, tuple)) and styles:
                            formatted_styles = ", ".join(
                                str(style).replace("_", " ").title() for style in styles if style
                            )
                            if formatted_styles:
                                hidden_segments.append(
                                    f"<!-- • Styles: {formatted_styles} -->"
                                )

                        context_parts: List[str] = []
                        for value in (
                            target.get("source_context"),
                            target.get("target_context"),
                        ):
                            if value:
                                context_parts.append(str(value))
                        if context_parts:
                            metadata_segments.append(f"Context: {' | '.join(context_parts)}")

                        if metadata_segments:
                            line_segments.extend(metadata_segments)

                        prosody = target.get("prosody_profile")
                        if isinstance(prosody, dict):
                            cadence = prosody.get("complexity_tag")
                            if cadence:
                                hidden_segments.append(
                                    f"<!-- • Cadence: {str(cadence).replace('_', ' ').title()} -->"
                                )

                        if hidden_segments:
                            line_segments.extend(hidden_segments)

                        formatted_line = " • ".join(
                            segment for segment in line_segments if segment
                        )
                        if formatted_line:
                            lines.append(formatted_line)
                            lines.append("")

                lines.append("")
                continue

            for entry in entries:
                target_word = entry.get("target_word") or "(unknown)"
                subject_label = f"**{str(target_word).upper()}**"

                line_segments = [subject_label]
                line_segments.extend(_standard_info_segments(entry, key))

                if key == "anti_llm":
                    hidden_segments: List[str] = []
                    weakness = entry.get("llm_weakness_type")
                    if weakness:
                        hidden_segments.append(
                            f"<!-- • LLM weakness: {str(weakness).replace('_', ' ').title()} -->"
                        )
                    cultural_depth = entry.get("cultural_depth")
                    if cultural_depth:
                        hidden_segments.append(
                            f"<!-- • Cultural depth: {cultural_depth} -->"
                        )
                    if hidden_segments:
                        line_segments.extend(hidden_segments)

                formatted_line = " • ".join(segment for segment in line_segments if segment)
                if formatted_line:
                    lines.append(formatted_line)
                    lines.append("")

            lines.append("")

        return "\n".join(lines).strip() + "\n"
