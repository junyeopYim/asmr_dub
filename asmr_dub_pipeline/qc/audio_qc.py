from __future__ import annotations

from pathlib import Path

from asmr_dub_pipeline.audio.duration import duration_ratio
from asmr_dub_pipeline.audio.features import (
    clipping_ratio,
    duration_sec,
    leading_trailing_silence,
    peak_dbfs,
    rms_dbfs,
)


def measure_audio_qc(audio_path: Path, target_duration_sec: float) -> dict[str, float]:
    duration = duration_sec(audio_path)
    leading, trailing = leading_trailing_silence(audio_path)
    return {
        "duration_sec": duration,
        "duration_ratio": duration_ratio(duration, target_duration_sec),
        "peak_dbfs": peak_dbfs(audio_path),
        "rms_dbfs": rms_dbfs(audio_path),
        "clipping_ratio": clipping_ratio(audio_path),
        "leading_silence_sec": leading,
        "trailing_silence_sec": trailing,
    }
