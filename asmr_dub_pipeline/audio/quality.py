from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from asmr_dub_pipeline.audio.features import (
    clipping_ratio,
    duration_sec,
    leading_trailing_silence,
    load_audio,
    peak_dbfs,
    rms_dbfs,
    to_mono,
)


@dataclass(frozen=True)
class AudioQualityMetrics:
    duration_sec: float
    peak_dbfs: float
    rms_dbfs: float
    clipping_ratio: float
    leading_silence_sec: float
    trailing_silence_sec: float
    active_ratio: float
    silence_ratio: float
    estimated_snr_db: float | None
    score: float
    issues: list[str] = field(default_factory=list)

    def as_payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload["score"] = round(self.score, 6)
        if self.estimated_snr_db is not None:
            payload["estimated_snr_db"] = round(self.estimated_snr_db, 6)
        return payload


def measure_source_voice_quality(
    path: Path | str,
    *,
    silence_threshold_db: float = -50.0,
    frame_ms: float = 40.0,
) -> AudioQualityMetrics:
    data, sample_rate = load_audio(path)
    mono = to_mono(data)
    duration = duration_sec(path)
    frame = max(1, int(sample_rate * frame_ms / 1000.0))
    threshold = 10 ** (silence_threshold_db / 20.0)
    frame_rms: list[float] = []
    for start in range(0, len(mono), frame):
        chunk = mono[start : start + frame]
        if len(chunk):
            frame_rms.append(float(np.sqrt(np.mean(np.square(chunk)))))
    active = [value for value in frame_rms if value > threshold]
    inactive = [value for value in frame_rms if value <= threshold]
    active_ratio = len(active) / len(frame_rms) if frame_rms else 0.0
    silence_ratio = 1.0 - active_ratio
    estimated_snr_db: float | None = None
    if active and inactive:
        active_rms = float(np.median(active))
        inactive_rms = max(float(np.median(inactive)), 1e-8)
        estimated_snr_db = 20.0 * float(np.log10(max(active_rms, 1e-8) / inactive_rms))
    leading, trailing = leading_trailing_silence(path, threshold_db=silence_threshold_db)
    peak = peak_dbfs(path)
    rms = rms_dbfs(path)
    clip = clipping_ratio(path)
    issues: list[str] = []
    if active_ratio < 0.15:
        issues.append("mostly_silent")
    if rms < -55.0:
        issues.append("very_low_rms")
    if peak > -0.2 or clip > 0.001:
        issues.append("clipping")
    if duration > 0 and (leading + trailing) / duration > 0.60:
        issues.append("edge_silence_heavy")
    if estimated_snr_db is not None and estimated_snr_db < 10.0:
        issues.append("low_estimated_snr")
    score = 1.0
    score -= min(0.60, silence_ratio * 0.45)
    if rms < -45.0:
        score -= min(0.25, (-45.0 - rms) / 60.0)
    if clip > 0.0:
        score -= min(0.35, clip * 80.0)
    if estimated_snr_db is not None and estimated_snr_db < 20.0:
        score -= min(0.20, (20.0 - estimated_snr_db) / 100.0)
    return AudioQualityMetrics(
        duration_sec=duration,
        peak_dbfs=peak,
        rms_dbfs=rms,
        clipping_ratio=clip,
        leading_silence_sec=leading,
        trailing_silence_sec=trailing,
        active_ratio=active_ratio,
        silence_ratio=silence_ratio,
        estimated_snr_db=estimated_snr_db,
        score=max(0.0, min(1.0, score)),
        issues=issues,
    )
