from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from asmr_dub_pipeline.schemas import Segment

from . import ffmpeg
from .features import AudioProcessingError, ensure_stereo, to_mono, write_audio


def _validate_block(data: np.ndarray, path: Path) -> None:
    if data.size and not np.isfinite(data).all():
        raise AudioProcessingError(f"Audio contains NaN or infinity: {path}")


def _read_slice_from_handle(audio: sf.SoundFile, path: Path, start: float, end: float) -> tuple[np.ndarray, int]:
    if audio.frames <= 0:
        raise AudioProcessingError(f"Audio file is empty: {path}")
    sample_rate = int(audio.samplerate)
    start_idx = max(0, int(round(start * sample_rate)))
    requested_end_idx = max(start_idx, int(round(end * sample_rate)))
    end_idx = min(int(audio.frames), requested_end_idx)
    frames = max(0, end_idx - start_idx)
    fallback_start = start_idx / sample_rate
    fallback_end = requested_end_idx / sample_rate
    try:
        audio.seek(start_idx)
        data = audio.read(frames, always_2d=True, dtype="float32")
    except Exception as exc:
        data, sample_rate = _read_slice_with_ffmpeg(
            path,
            fallback_start,
            fallback_end,
            sample_rate=sample_rate,
            channels=max(1, int(audio.channels)),
            cause=exc,
        )
    else:
        if requested_end_idx > start_idx and len(data) == 0:
            data, sample_rate = _read_slice_with_ffmpeg(
                path,
                fallback_start,
                fallback_end,
                sample_rate=sample_rate,
                channels=max(1, int(audio.channels)),
                cause=AudioProcessingError(f"SoundFile returned an empty slice: {path}"),
            )
    _validate_block(data, path)
    return data, sample_rate


def _read_slice_with_ffmpeg(
    path: Path,
    start: float,
    end: float,
    *,
    sample_rate: int,
    channels: int,
    cause: Exception,
) -> tuple[np.ndarray, int]:
    duration = max(0.0, end - start)
    if duration <= 0.0:
        return np.zeros((0, channels), dtype=np.float32), sample_rate
    cmd = [
        ffmpeg.require_binary("ffmpeg"),
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start:.6f}",
        "-i",
        str(path),
        "-t",
        f"{duration:.6f}",
        "-vn",
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "pipe:1",
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise AudioProcessingError(
            f"Audio seek failed and ffmpeg fallback could not read slice {start:.3f}-{end:.3f}: {path}"
        ) from exc
    raw = np.frombuffer(result.stdout, dtype="<f4")
    usable = (raw.size // channels) * channels
    if usable != raw.size:
        raw = raw[:usable]
    data = raw.reshape((-1, channels)).astype(np.float32, copy=False)
    _validate_block(data, path)
    if data.size == 0 and duration > 0.0:
        raise AudioProcessingError(
            f"Audio seek failed and ffmpeg fallback returned an empty slice {start:.3f}-{end:.3f}: {path}"
        ) from cause
    return data, sample_rate


def _read_slice(path: Path, start: float, end: float) -> tuple[np.ndarray, int]:
    with sf.SoundFile(str(path)) as audio:
        return _read_slice_from_handle(audio, path, start, end)


def _write_slice_from_file(path: Path, start: float, end: float, output_path: Path) -> Path:
    data, sample_rate = _read_slice(path, start, end)
    write_audio(output_path, data, sample_rate)
    return output_path


def _estimate_pan(stereo_slice: np.ndarray) -> float:
    stereo = ensure_stereo(stereo_slice)
    left = float(np.sqrt(np.mean(np.square(stereo[:, 0])))) if len(stereo) else 0.0
    right = float(np.sqrt(np.mean(np.square(stereo[:, 1])))) if len(stereo) else 0.0
    denom = left + right
    if denom <= 1e-9:
        return 0.0
    return max(-1.0, min(1.0, (right - left) / denom))


def load_manual_segments(path: Path) -> list[Segment]:
    data = json.loads(path.read_text("utf-8"))
    items = data["segments"] if isinstance(data, dict) and "segments" in data else data
    if not isinstance(items, list):
        raise ValueError("Manual segments must be a list or an object with a segments list")
    return [Segment.model_validate(item) for item in items]


def write_segment_audio_clips(
    segments: list[Segment],
    gemma_audio_path: Path,
    mix_audio_path: Path,
    project_dir: Path,
    progress_callback: Callable[[int, int, Segment], None] | None = None,
) -> list[Segment]:
    total = len(segments)
    with sf.SoundFile(str(gemma_audio_path)) as gemma_audio, sf.SoundFile(str(mix_audio_path)) as mix_audio:
        for idx, segment in enumerate(segments, start=1):
            gemma_clip = project_dir / "work" / "segments" / "audio" / f"{segment.id}_gemma.wav"
            mix_clip = project_dir / "work" / "segments" / "audio" / f"{segment.id}_mix.wav"
            gemma_slice, gemma_sr = _read_slice_from_handle(
                gemma_audio,
                gemma_audio_path,
                segment.start,
                segment.end,
            )
            write_audio(gemma_clip, gemma_slice, gemma_sr)
            mix_slice, mix_sr = _read_slice_from_handle(
                mix_audio,
                mix_audio_path,
                segment.start,
                segment.end,
            )
            write_audio(mix_clip, mix_slice, mix_sr)
            segment.audio_for_gemma = str(gemma_clip)
            segment.audio_for_mix = str(mix_clip)
            segment.estimated_pan = round(_estimate_pan(mix_slice), 3)
            if progress_callback:
                progress_callback(idx, total, segment)
    return segments


def _read_slice_from_parts(
    parts: Sequence[dict[str, Any]],
    path_key: str,
    start: float,
    end: float,
) -> tuple[np.ndarray, int]:
    chunks: list[np.ndarray] = []
    sample_rate: int | None = None
    for part in parts:
        part_start = float(part["start_sec"])
        part_end = float(part["end_sec"])
        overlap_start = max(start, part_start)
        overlap_end = min(end, part_end)
        if overlap_end <= overlap_start:
            continue
        local_start = overlap_start - part_start
        local_end = overlap_end - part_start
        data, sr = _read_slice(Path(str(part[path_key])), local_start, local_end)
        if sample_rate is None:
            sample_rate = sr
        elif sample_rate != sr:
            raise AudioProcessingError("Part audio slices must share the same sample rate.")
        chunks.append(data)
    if not chunks or sample_rate is None:
        raise AudioProcessingError(f"No part audio covers segment range {start:.3f}-{end:.3f}.")
    return np.concatenate(chunks, axis=0), sample_rate


def write_segment_audio_clips_from_parts(
    segments: list[Segment],
    parts: Sequence[dict[str, Any]],
    project_dir: Path,
    progress_callback: Callable[[int, int, Segment], None] | None = None,
) -> list[Segment]:
    total = len(segments)
    for idx, segment in enumerate(segments, start=1):
        gemma_clip = project_dir / "work" / "segments" / "audio" / f"{segment.id}_gemma.wav"
        mix_clip = project_dir / "work" / "segments" / "audio" / f"{segment.id}_mix.wav"
        gemma_slice, gemma_sr = _read_slice_from_parts(
            parts,
            "vocals_mono_path",
            segment.start,
            segment.end,
        )
        mix_slice, mix_sr = _read_slice_from_parts(
            parts,
            "vocals_path",
            segment.start,
            segment.end,
        )
        write_audio(gemma_clip, gemma_slice, gemma_sr)
        write_audio(mix_clip, mix_slice, mix_sr)
        segment.audio_for_gemma = str(gemma_clip)
        segment.audio_for_mix = str(mix_clip)
        segment.estimated_pan = round(_estimate_pan(mix_slice), 3)
        if progress_callback:
            progress_callback(idx, total, segment)
    return segments


def _active_frames(path: Path, frame_len: int, threshold: float) -> tuple[list[bool], int, int]:
    active_frames: list[bool] = []
    with sf.SoundFile(str(path)) as audio:
        if audio.frames <= 0:
            raise AudioProcessingError(f"Audio file is empty: {path}")
        sample_rate = int(audio.samplerate)
        while True:
            chunk = audio.read(frame_len, always_2d=True, dtype="float32")
            if len(chunk) == 0:
                break
            _validate_block(chunk, path)
            mono = to_mono(chunk)
            rms = float(np.sqrt(np.mean(np.square(mono)))) if len(mono) else 0.0
            active_frames.append(rms > threshold)
        return active_frames, sample_rate, int(audio.frames)


def energy_segments(
    gemma_audio_path: Path,
    mix_audio_path: Path,
    project_dir: Path,
    min_segment_sec: float = 0.25,
    max_segment_sec: float = 20.0,
    silence_db: float = -45.0,
    min_silence_sec: float = 0.30,
    progress_callback: Callable[[int, int, Segment], None] | None = None,
) -> list[Segment]:
    threshold = 10 ** (silence_db / 20.0)
    with sf.SoundFile(str(gemma_audio_path)) as info:
        sr = int(info.samplerate)
    frame_len = max(1, int(sr * 0.05))
    active_frames, sr, total_frames = _active_frames(gemma_audio_path, frame_len, threshold)
    segments: list[tuple[float, float]] = []
    active_start: int | None = None
    last_active: int | None = None
    gap_frames = int(max(1, round(min_silence_sec / 0.05)))
    for idx, active in enumerate(active_frames):
        if active:
            if active_start is None:
                active_start = idx
            last_active = idx
        elif active_start is not None and last_active is not None and idx - last_active >= gap_frames:
            start_sec = active_start * frame_len / sr
            end_sec = min(total_frames / sr, (last_active + 1) * frame_len / sr)
            if end_sec - start_sec >= min_segment_sec:
                segments.append((start_sec, end_sec))
            active_start = None
            last_active = None
    if active_start is not None and last_active is not None:
        segments.append(
            (
                active_start * frame_len / sr,
                min(total_frames / sr, (last_active + 1) * frame_len / sr),
            )
        )
    if not segments:
        segments = [(0.0, total_frames / sr)]
    split: list[tuple[float, float]] = []
    for start, end in segments:
        cursor = start
        while end - cursor > max_segment_sec:
            split.append((cursor, cursor + max_segment_sec))
            cursor += max_segment_sec
        if end - cursor > 0.01:
            split.append((cursor, end))
    out: list[Segment] = []
    total = len(split)
    for idx, (start, end) in enumerate(split, start=1):
        seg_id = f"seg_{idx:04d}"
        gemma_clip = project_dir / "work" / "segments" / "audio" / f"{seg_id}_gemma.wav"
        mix_clip = project_dir / "work" / "segments" / "audio" / f"{seg_id}_mix.wav"
        _write_slice_from_file(gemma_audio_path, start, end, gemma_clip)
        mix_slice, mix_sr = _read_slice(mix_audio_path, start, end)
        write_audio(mix_clip, mix_slice, mix_sr)
        status = "raw" if end - start >= min_segment_sec else "needs_manual_review"
        segment = Segment(
            id=seg_id,
            start=round(start, 3),
            end=round(end, 3),
            duration=round(end - start, 3),
            audio_for_gemma=str(gemma_clip),
            audio_for_mix=str(mix_clip),
            estimated_pan=round(_estimate_pan(mix_slice), 3),
            keep_original_texture=True,
            status=status,
        )
        out.append(segment)
        if progress_callback:
            progress_callback(idx, total, segment)
    return out
