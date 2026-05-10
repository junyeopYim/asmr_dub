from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import soundfile as sf

DEFAULT_BLOCK_FRAMES = 65_536


class AudioProcessingError(RuntimeError):
    pass


def _validate_audio_block(data: np.ndarray, path: Path | str) -> None:
    if data.size and not np.isfinite(data).all():
        raise AudioProcessingError(f"Audio contains NaN or infinity: {path}")


def iter_audio_blocks(
    path: Path | str,
    *,
    block_frames: int = DEFAULT_BLOCK_FRAMES,
) -> Iterator[tuple[np.ndarray, int]]:
    with sf.SoundFile(str(path)) as audio:
        if audio.frames <= 0:
            raise AudioProcessingError(f"Audio file is empty: {path}")
        while True:
            block = audio.read(block_frames, always_2d=True, dtype="float32")
            if len(block) == 0:
                break
            _validate_audio_block(block, path)
            yield block, int(audio.samplerate)


def load_audio(path: Path | str) -> tuple[np.ndarray, int]:
    data, sample_rate = sf.read(str(path), always_2d=True, dtype="float32")
    if data.size == 0:
        raise AudioProcessingError(f"Audio file is empty: {path}")
    _validate_audio_block(data, path)
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
    peak = 0.0
    for block, _ in iter_audio_blocks(path):
        peak = max(peak, float(np.max(np.abs(block))))
    if peak <= 0:
        return -120.0
    return 20.0 * float(np.log10(peak))


def rms_dbfs(path: Path | str) -> float:
    total_squares = 0.0
    total_samples = 0
    for block, _ in iter_audio_blocks(path):
        data = block.astype(np.float64, copy=False)
        total_squares += float(np.sum(data * data))
        total_samples += int(data.size)
    rms = float(np.sqrt(total_squares / total_samples)) if total_samples else 0.0
    if rms <= 0:
        return -120.0
    return 20.0 * float(np.log10(rms))


def clipping_ratio(path: Path | str, threshold: float = 0.999) -> float:
    clipped = 0
    total_samples = 0
    for block, _ in iter_audio_blocks(path):
        clipped += int(np.count_nonzero(np.abs(block) >= threshold))
        total_samples += int(block.size)
    return float(clipped / total_samples) if total_samples else 0.0


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
    with sf.SoundFile(str(path)) as audio:
        if audio.frames <= 0:
            raise AudioProcessingError(f"Audio file is empty: {path}")
        sr = int(audio.samplerate)
        frame = max(1, int(sr * frame_ms / 1000.0))
        threshold = 10 ** (threshold_db / 20.0)
        frame_count = 0
        first_active: int | None = None
        last_active: int | None = None
        while True:
            chunk = audio.read(frame, always_2d=True, dtype="float32")
            if len(chunk) == 0:
                break
            _validate_audio_block(chunk, path)
            mono = to_mono(chunk)
            rms = float(np.sqrt(np.mean(np.square(mono)))) if len(mono) else 0.0
            if rms > threshold:
                if first_active is None:
                    first_active = frame_count
                last_active = frame_count
            frame_count += 1
        if first_active is None or last_active is None:
            total = float(audio.frames) / float(sr)
            return total, total
    leading = first_active * frame / sr
    trailing = max(0.0, (frame_count - last_active - 1) * frame / sr)
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
