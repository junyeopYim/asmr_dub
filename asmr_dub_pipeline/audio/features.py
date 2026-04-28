from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


class AudioProcessingError(RuntimeError):
    pass


def load_audio(path: Path | str) -> tuple[np.ndarray, int]:
    data, sample_rate = sf.read(str(path), always_2d=True, dtype="float32")
    if data.size == 0:
        raise AudioProcessingError(f"Audio file is empty: {path}")
    if not np.isfinite(data).all():
        raise AudioProcessingError(f"Audio contains NaN or infinity: {path}")
    return data, int(sample_rate)


def write_audio(path: Path | str, data: np.ndarray, sample_rate: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(data, dtype=np.float32), sample_rate)


def ensure_stereo(data: np.ndarray) -> np.ndarray:
    if data.ndim == 1:
        data = data[:, None]
    if data.shape[1] == 1:
        return np.repeat(data, 2, axis=1)
    return data[:, :2]


def to_mono(data: np.ndarray) -> np.ndarray:
    if data.ndim == 1:
        return data
    return data.mean(axis=1)


def duration_sec(path: Path | str) -> float:
    info = sf.info(str(path))
    return float(info.frames) / float(info.samplerate)


def peak_dbfs(path: Path | str) -> float:
    data, _ = load_audio(path)
    peak = float(np.max(np.abs(data)))
    if peak <= 0:
        return -120.0
    return 20.0 * float(np.log10(peak))


def rms_dbfs(path: Path | str) -> float:
    data, _ = load_audio(path)
    rms = float(np.sqrt(np.mean(np.square(data))))
    if rms <= 0:
        return -120.0
    return 20.0 * float(np.log10(rms))


def clipping_ratio(path: Path | str, threshold: float = 0.999) -> float:
    data, _ = load_audio(path)
    return float(np.mean(np.abs(data) >= threshold))


def resample_linear(data: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate:
        return data
    if data.ndim == 1:
        data_2d = data[:, None]
        squeeze = True
    else:
        data_2d = data
        squeeze = False
    src_len = data_2d.shape[0]
    dst_len = max(1, int(round(src_len * dst_rate / src_rate)))
    old_x = np.linspace(0.0, 1.0, src_len, endpoint=False)
    new_x = np.linspace(0.0, 1.0, dst_len, endpoint=False)
    channels = [np.interp(new_x, old_x, data_2d[:, ch]) for ch in range(data_2d.shape[1])]
    out = np.stack(channels, axis=1).astype(np.float32)
    return out[:, 0] if squeeze else out


def leading_trailing_silence(
    path: Path | str,
    threshold_db: float = -50.0,
    frame_ms: float = 20.0,
) -> tuple[float, float]:
    data, sr = load_audio(path)
    mono = to_mono(data)
    frame = max(1, int(sr * frame_ms / 1000.0))
    if len(mono) <= frame:
        active = np.max(np.abs(mono)) > 10 ** (threshold_db / 20.0)
        return (0.0, 0.0) if active else (len(mono) / sr, len(mono) / sr)
    frames = []
    for start in range(0, len(mono), frame):
        chunk = mono[start : start + frame]
        rms = float(np.sqrt(np.mean(np.square(chunk)))) if len(chunk) else 0.0
        frames.append(rms > 10 ** (threshold_db / 20.0))
    active_idx = [idx for idx, active in enumerate(frames) if active]
    if not active_idx:
        total = len(mono) / sr
        return total, total
    leading = active_idx[0] * frame / sr
    trailing = max(0.0, (len(frames) - active_idx[-1] - 1) * frame / sr)
    return leading, trailing


def trim_edge_silence(
    path: Path | str,
    *,
    threshold_db: float = -50.0,
    frame_ms: float = 20.0,
    keep_sec: float = 0.08,
) -> dict[str, float | bool]:
    data, sr = load_audio(path)
    mono = to_mono(data)
    frame = max(1, int(sr * frame_ms / 1000.0))
    threshold = 10 ** (threshold_db / 20.0)
    active_idx: list[int] = []
    for index, start in enumerate(range(0, len(mono), frame)):
        chunk = mono[start : start + frame]
        rms = float(np.sqrt(np.mean(np.square(chunk)))) if len(chunk) else 0.0
        if rms > threshold:
            active_idx.append(index)
    if not active_idx:
        return {"trimmed": False, "leading_trim_sec": 0.0, "trailing_trim_sec": 0.0}
    keep_frames = int(round(max(0.0, keep_sec) * sr))
    start_frame = max(0, active_idx[0] * frame - keep_frames)
    end_frame = min(len(data), (active_idx[-1] + 1) * frame + keep_frames)
    leading_trim = start_frame / sr
    trailing_trim = max(0.0, (len(data) - end_frame) / sr)
    if leading_trim <= 0.0 and trailing_trim <= 0.0:
        return {"trimmed": False, "leading_trim_sec": 0.0, "trailing_trim_sec": 0.0}
    trimmed = data[start_frame:end_frame]
    if len(trimmed) == 0:
        return {"trimmed": False, "leading_trim_sec": 0.0, "trailing_trim_sec": 0.0}
    write_audio(path, trimmed, sr)
    return {
        "trimmed": True,
        "leading_trim_sec": round(leading_trim, 6),
        "trailing_trim_sec": round(trailing_trim, 6),
    }
