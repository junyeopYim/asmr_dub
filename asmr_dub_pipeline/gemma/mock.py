from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from asmr_dub_pipeline.schemas import Segment

from .base import GemmaBackend


class MockGemmaBackend(GemmaBackend):
    def analyze_segment(self, audio_path: Path, segment: Segment, context: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "source_language": "ja",
            "transcript_original": f"mock transcript for {segment.id}",
            "literal_ja": "ゆっくり、深呼吸してください。",
            "speech_style": "soft whisper",
            "speaker_count": 1,
            "emotion": "gentle",
            "pace": "slow",
            "volume": "soft",
            "nonverbal_cues": [],
            "spatial_style": "center",
            "style_tags": ["soft_whisper", "close_mic"],
            "estimated_pan": segment.estimated_pan,
            "keep_original_texture": True,
            "risk_flags": [],
            "confidence": 0.95,
            "voice_training": {
                "clean_voice": True,
                "eligible": True,
                "reason": "",
                "effect_tags": ["none"],
                "same_speaker_under_effect": False,
            },
            "effect_events": [],
            "audio_path": str(audio_path),
        }

    def analyze_audio_style(self, audio_path: Path, segment: Segment, context: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "nonverbal_cues": [],
            "spatial_style": "center",
            "style_tags": ["soft_whisper", "close_mic"],
            "estimated_pan": segment.estimated_pan,
            "keep_original_texture": True,
            "risk_flags": [],
            "confidence": 0.95,
            "voice_training": {
                "clean_voice": True,
                "eligible": True,
                "reason": "",
                "effect_tags": ["none"],
                "same_speaker_under_effect": False,
            },
            "effect_events": [],
            "audio_path": str(audio_path),
        }

    def generate_script(self, audio_path: Path, segment: Segment, context: Mapping[str, Any]) -> dict[str, Any]:
        duration = max(0.5, min(segment.duration, 8.0))
        return {
            "literal_ja": "ゆっくり、深呼吸してください。",
            "ja_text": "ゆっくり……深呼吸してくださいね。",
            "tts_text": "ゆっくり……深呼吸してくださいね。",
            "ref_style": "whisper_close",
            "emotion": "gentle",
            "pace": "slow",
            "volume": "soft",
            "nonverbal_cues": [],
            "spatial_style": "center",
            "expected_tts_duration_sec": round(duration, 3),
            "style_tags": ["soft_whisper", "slow_pace"],
            "retry_policy": {
                "duration_too_long": "request_shorter_script",
                "duration_too_short": "request_longer_script",
                "repetition_detected": "regenerate_with_variation_and_new_seed",
                "omission_detected": "regenerate_with_variation_and_new_seed",
                "max_script_rewrites": 2,
                "max_tts_regenerations": 2,
                "seed_strategy": "increment_candidate_seed",
                "shortening_prompt": (
                    "Shorten tts_text while preserving the source meaning, ASMR tone, "
                    "and nonverbal metadata."
                ),
                "variation_prompt": (
                    "Generate a fresh wording with the same meaning and style metadata; "
                    "avoid repeated or omitted phrases."
                ),
            },
            "risk_flags": [],
            "audio_path": str(audio_path),
        }

    def qc_audio(
        self,
        audio_path: Path,
        target_text: str,
        segment: Segment,
        context: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "text_match_score": 0.98,
            "pronunciation_score": 0.95,
            "asmr_style_score": 0.96,
            "timing_score": 0.94,
            "repetition_detected": False,
            "omission_detected": False,
            "unsafe_or_rights_issue": False,
            "recommendation": "pass",
            "issues": [],
            "audio_path": str(audio_path),
            "target_text": target_text,
        }
