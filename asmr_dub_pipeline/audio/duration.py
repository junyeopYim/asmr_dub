from __future__ import annotations


def duration_ratio(actual_sec: float, target_sec: float) -> float:
    if target_sec <= 0:
        return 0.0
    return actual_sec / target_sec


def suggest_speed_factor(
    estimated_tts_sec: float,
    target_sec: float,
    minimum: float = 0.85,
    maximum: float = 1.20,
) -> float:
    if target_sec <= 0 or estimated_tts_sec <= 0:
        return 1.0
    return min(max(estimated_tts_sec / target_sec, minimum), maximum)
