from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from asmr_dub_pipeline.schemas import Segment

from .schemas import TASK_REQUIRED_KEYS, TaskName

PROMPT_VERSION = "2026-04-26"

SAFETY_LINE = (
    "Use only user-authorized source and reference material"
)

STRICT_JSON_LINE = (
    "Return exactly one valid JSON object. Do not include markdown, prose, comments, code fences, "
    "trailing commas, or keys outside the contract."
)

STYLE_ENUMS = (
    "Allowed emotion: gentle, sleepy, reassuring, playful, serious, neutral. "
    "Allowed pace: very_slow, slow, normal, slightly_fast. "
    "Allowed volume: whisper, soft, normal. "
    "Allowed spatial_style: center, left_close, right_close, center_close, center_far, "
    "sleepy_center, binaural_sweep, ambient."
)

NONVERBAL_CUE_CONTRACT = (
    "Each nonverbal_cues item must be an object with kind, source_text, normalized_text, "
    "position, intensity, optional pause_sec, and notes. Use metadata for breaths, laughs, "
    "pauses, mouth sounds, ear proximity, pan movement, and bracketed stage directions. "
    "For natural pauses, set kind=\"pause\" and pause_sec instead of putting bracketed "
    "pause directions in ja_text or tts_text."
)

VOICE_TRAINING_CONTRACT = (
    "For speaker_count, count only distinct human speakers. Do not count reverb, telephone, "
    "radio, robot, distortion, echo, or other effects as a different speaker. Also include "
    "voice_training with clean_voice, eligible, reason, effect_tags, and same_speaker_under_effect. "
    "Set clean_voice=false and eligible=false when voice effects, heavy SFX, overlap, music, or "
    "multi-speaker speech make the clip unsafe for fine-tune/RVC training."
)

RETRY_POLICY_CONTRACT = (
    "retry_policy must be an object with duration_too_long, duration_too_short, "
    "repetition_detected, omission_detected, max_script_rewrites, max_tts_regenerations, "
    "seed_strategy, shortening_prompt, and variation_prompt. Use duration_too_long="
    '"request_shorter_script", duration_too_short="request_longer_script", '
    'repetition_detected="regenerate_with_variation_and_new_seed", '
    'omission_detected="regenerate_with_variation_and_new_seed", and seed_strategy='
    '"increment_candidate_seed" unless the segment must go to manual_review.'
)


def analysis_prompt(segment: Segment, context: Mapping[str, Any] | None = None) -> str:
    return f"""{STRICT_JSON_LINE}
Analyze this ASMR audio segment for speech, translation, timing, style, spatial position, and risks.
{STYLE_ENUMS}
{NONVERBAL_CUE_CONTRACT}
{VOICE_TRAINING_CONTRACT}
Required keys: source_language, transcript_original, literal_ja, speech_style, speaker_count, emotion, pace, volume, nonverbal_cues, spatial_style, style_tags, estimated_pan, keep_original_texture, risk_flags, confidence, voice_training.
Segment: id={segment.id}, start={segment.start}, end={segment.end}, duration={segment.duration}.
Context: {dict(context or {})}
"""


def audio_style_prompt(segment: Segment) -> str:
    return f"""{STRICT_JSON_LINE}
Analyze only the audible style of this ASMR source audio segment. Do not translate,
rewrite, summarize, or infer story content.
Allowed effect_tags: none, telephone, radio, robot, distortion, reverb, echo. Use
exactly ["none"] when no voice, background, or attached SFX effect is clearly
audible; never combine none with another effect tag. Treat reverb, echo, robot,
telephone, radio, and distortion as processing on the same speaker, not as extra speakers.
effect_events must be a list of objects with tag, target, start_sec, end_sec,
intensity, confidence, and params. Use segment-relative seconds. Use an empty
effect_events list when effect_tags is ["none"]. Put DSP hints in params, such as
delays_ms, decays, modulation_hz, depth, drive, low_hz, high_hz, and wet.
{STYLE_ENUMS}
{NONVERBAL_CUE_CONTRACT}
Required keys: style_tags, nonverbal_cues, spatial_style, estimated_pan, keep_original_texture, risk_flags, confidence, effect_events, voice_training.
JSON shape: {{"style_tags":[],"nonverbal_cues":[],"spatial_style":"center","estimated_pan":0.0,"keep_original_texture":true,"risk_flags":[],"confidence":0.0,"effect_events":[],"voice_training":{{"clean_voice":true,"eligible":true,"reason":"","effect_tags":["none"],"same_speaker_under_effect":false}}}}
Segment: id={segment.id}, start={segment.start}, end={segment.end}, duration={segment.duration}.
"""


def script_prompt(segment: Segment, analysis: Mapping[str, Any], context: Mapping[str, Any] | None = None) -> str:
    return f"""{STRICT_JSON_LINE}
Generate gentle Japanese ASMR dubbing text for GPT-SoVITS.
{STYLE_ENUMS}
{NONVERBAL_CUE_CONTRACT}
{RETRY_POLICY_CONTRACT}
Required keys: literal_ja, ja_text, tts_text, ref_style, emotion, pace, volume, nonverbal_cues, spatial_style, expected_tts_duration_sec, style_tags, retry_policy, risk_flags.
Rules: ja_text and tts_text must contain only speakable Japanese text for synthesis. Do not put bracketed stage directions, speaker labels, breaths, laughs, spatial hints, emoji, markdown, or style tags in ja_text or tts_text. Put that performance intent in nonverbal_cues, emotion, pace, volume, spatial_style, ref_style, and style_tags. Aim for 0.85-1.15x the segment duration and prefer short, natural ASMR phrases over literal over-translation.
Segment: id={segment.id}, duration={segment.duration}.
Analysis: {dict(analysis)}
Context: {dict(context or {})}
"""


def qc_prompt(segment: Segment, target_text: str, context: Mapping[str, Any] | None = None) -> str:
    return f"""{STRICT_JSON_LINE}
Quality-check the synthesized Japanese ASMR audio against the target text and style.
Required keys: text_match_score, pronunciation_score, asmr_style_score, timing_score, repetition_detected, omission_detected, unsafe_or_rights_issue, recommendation, issues.
recommendation must be one of: pass, regenerate, manual_review. If timing fails because the synthesized audio is too long, add issue "duration_too_long" so the script retry_policy can request a shorter script. If repetition_detected or omission_detected is true, prefer recommendation="regenerate" unless there is a rights or safety issue.
Segment: id={segment.id}, duration={segment.duration}.
Target text: {target_text}
Context: {dict(context or {})}
"""


def repair_prompt(task: TaskName, original_prompt: str, bad_response: str, error: str) -> str:
    required = ", ".join(sorted(TASK_REQUIRED_KEYS[task]))
    return f"""{STRICT_JSON_LINE}
Repair the previous {task} response so it matches the schema.
Required keys: {required}.
Validation error: {error}
Previous response:
{bad_response[:4000]}

Original instruction:
{original_prompt[:4000]}
"""


def json_repair_prompt(task: TaskName, raw_response: str, error: str) -> str:
    required = ", ".join(sorted(TASK_REQUIRED_KEYS[task]))
    return f"""{STRICT_JSON_LINE}
Repair the previous Gemma {task} response so it is exactly one JSON object matching the schema.
Required keys: {required}.
Do not add explanations, markdown fences, comments, or extra wrapper keys.
Validation error: {error}
Previous response:
{raw_response[:6000]}
"""
