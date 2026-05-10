from __future__ import annotations

SCRIPT_SYSTEM_PROMPT = (
    "Return one strict JSON object only: no markdown, no prose, no code fences. "
    "Keep ja_text and tts_text as clean speakable Japanese. Put breaths, laughs, "
    "ear position, pauses, emotion, pace, volume, and bracketed directions into "
    "metadata fields only. Use spatial_style values center, left_close, right_close, "
    "center_close, center_far, sleepy_center, binaural_sweep, or ambient. Put natural "
    "pause timing in nonverbal_cues.pause_sec, never as bracketed TTS text. Include "
    "retry_policy so downstream QC can request "
    "shortening for overlong TTS or variation with a new seed for repetition or "
    "omission."
)
