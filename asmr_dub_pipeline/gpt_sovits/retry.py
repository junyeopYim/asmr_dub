from __future__ import annotations

from enum import StrEnum

from .schemas import GPTSoVITSTTSOptions


class GPTSoVITSRetrySignal(StrEnum):
    DURATION_TOO_LONG = "duration_too_long"
    DURATION_TOO_SHORT = "duration_too_short"
    SPEED_FACTOR_ADJUSTED = "speed_factor_adjusted"
    SCRIPT_SHORTENING_REQUESTED = "script_shortening_requested"
    SCRIPT_DURATION_REWRITE_REQUESTED = "script_duration_rewrite_requested"
    REPETITION_OR_OMISSION = "repetition_or_omission"
    SEED_CHANGED = "seed_changed"
    REPETITION_PENALTY_INCREASED = "repetition_penalty_increased"


def duration_too_long(actual_sec: float, target_sec: float, tolerance: float) -> bool:
    if actual_sec <= 0 or target_sec <= 0:
        return False
    return actual_sec > target_sec * (1.0 + tolerance)


def duration_too_short(actual_sec: float, target_sec: float, tolerance: float) -> bool:
    if actual_sec <= 0 or target_sec <= 0:
        return False
    return actual_sec < target_sec * (1.0 - tolerance)


def adjust_speed_for_duration(
    options: GPTSoVITSTTSOptions,
    actual_sec: float,
    target_sec: float,
    *,
    maximum: float = 1.35,
    minimum_step: float = 0.05,
) -> GPTSoVITSTTSOptions:
    if actual_sec <= 0 or target_sec <= 0:
        return options
    ratio = actual_sec / target_sec
    speed_factor = min(maximum, max(options.speed_factor + minimum_step, options.speed_factor * ratio))
    return options.model_copy(update={"speed_factor": round(speed_factor, 4)})


def adjust_speed_for_short_duration(
    options: GPTSoVITSTTSOptions,
    actual_sec: float,
    target_sec: float,
    *,
    minimum: float = 0.75,
    minimum_step: float = 0.05,
) -> GPTSoVITSTTSOptions:
    if actual_sec <= 0 or target_sec <= 0:
        return options
    ratio = actual_sec / target_sec
    speed_factor = max(minimum, min(options.speed_factor - minimum_step, options.speed_factor * ratio))
    return options.model_copy(update={"speed_factor": round(speed_factor, 4)})


def adjust_for_repetition_or_omission(
    options: GPTSoVITSTTSOptions,
    *,
    seed_step: int,
    penalty_step: float = 0.15,
    maximum_penalty: float = 2.0,
) -> GPTSoVITSTTSOptions:
    seed = options.seed + seed_step if options.seed >= 0 else seed_step
    repetition_penalty = min(maximum_penalty, options.repetition_penalty + penalty_step)
    return options.model_copy(
        update={
            "seed": seed,
            "repetition_penalty": round(repetition_penalty, 4),
        }
    )


def retry_signal_values(signals: list[GPTSoVITSRetrySignal]) -> list[str]:
    return [signal.value for signal in signals]
