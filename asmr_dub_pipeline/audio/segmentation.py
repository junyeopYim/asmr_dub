from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

from asmr_dub_pipeline.schemas import Segment

from .features import ensure_stereo, load_audio, to_mono, write_audio


def _write_slice(data: np.ndarray, sample_rate: int, start: float, end: float, output_path: Path) -> Path:
    start_idx = max(0, int(round(start * sample_rate)))
    end_idx = min(len(data), int(round(end * sample_rate)))
    write_audio(output_path, data[start_idx:end_idx], sample_rate)
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
    data, sr = load_audio(gemma_audio_path)
    mix_data, mix_sr = load_audio(mix_audio_path)
    total = len(segments)
    for idx, segment in enumerate(segments, start=1):
        gemma_clip = project_dir / "work" / "segments" / "audio" / f"{segment.id}_gemma.wav"
        mix_clip = project_dir / "work" / "segments" / "audio" / f"{segment.id}_mix.wav"
        _write_slice(data, sr, segment.start, segment.end, gemma_clip)
        _write_slice(mix_data, mix_sr, segment.start, segment.end, mix_clip)
        mix_start = max(0, int(round(segment.start * mix_sr)))
        mix_end = min(len(mix_data), int(round(segment.end * mix_sr)))
        segment.audio_for_gemma = str(gemma_clip)
        segment.audio_for_mix = str(mix_clip)
        segment.estimated_pan = round(_estimate_pan(mix_data[mix_start:mix_end]), 3)
        if progress_callback:
            progress_callback(idx, total, segment)
    return segments


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
    data, sr = load_audio(gemma_audio_path)
    mono = to_mono(data)
    frame_len = max(1, int(sr * 0.05))
    threshold = 10 ** (silence_db / 20.0)
    active_frames = []
    for start in range(0, len(mono), frame_len):
        chunk = mono[start : start + frame_len]
        rms = float(np.sqrt(np.mean(np.square(chunk)))) if len(chunk) else 0.0
        active_frames.append(rms > threshold)
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
            end_sec = min(len(mono) / sr, (last_active + 1) * frame_len / sr)
            if end_sec - start_sec >= min_segment_sec:
                segments.append((start_sec, end_sec))
            active_start = None
            last_active = None
    if active_start is not None and last_active is not None:
        segments.append((active_start * frame_len / sr, min(len(mono) / sr, (last_active + 1) * frame_len / sr)))
    if not segments:
        segments = [(0.0, len(mono) / sr)]
    split: list[tuple[float, float]] = []
    for start, end in segments:
        cursor = start
        while end - cursor > max_segment_sec:
            split.append((cursor, cursor + max_segment_sec))
            cursor += max_segment_sec
        if end - cursor > 0.01:
            split.append((cursor, end))
    mix_data, mix_sr = load_audio(mix_audio_path)
    out: list[Segment] = []
    total = len(split)
    for idx, (start, end) in enumerate(split, start=1):
        seg_id = f"seg_{idx:04d}"
        gemma_clip = project_dir / "work" / "segments" / "audio" / f"{seg_id}_gemma.wav"
        mix_clip = project_dir / "work" / "segments" / "audio" / f"{seg_id}_mix.wav"
        _write_slice(data, sr, start, end, gemma_clip)
        _write_slice(mix_data, mix_sr, start, end, mix_clip)
        mix_start = max(0, int(round(start * mix_sr)))
        mix_end = min(len(mix_data), int(round(end * mix_sr)))
        status = "raw" if end - start >= min_segment_sec else "needs_manual_review"
        segment = Segment(
            id=seg_id,
            start=round(start, 3),
            end=round(end, 3),
            duration=round(end - start, 3),
            audio_for_gemma=str(gemma_clip),
            audio_for_mix=str(mix_clip),
            estimated_pan=round(_estimate_pan(mix_data[mix_start:mix_end]), 3),
            keep_original_texture=True,
            status=status,
        )
        out.append(segment)
        if progress_callback:
            progress_callback(idx, total, segment)
    return out
