from __future__ import annotations

from pathlib import Path

from . import ffmpeg
from .features import ensure_stereo, load_audio, resample_linear, to_mono, write_audio


def extract_project_audio(input_path: Path, project_dir: Path) -> tuple[Path, Path]:
    stereo = project_dir / "work" / "audio" / "original_stereo_48k.wav"
    mono = project_dir / "work" / "audio" / "gemma_mono_16k.wav"
    try:
        ffmpeg.extract_stereo_48k(input_path, stereo)
        ffmpeg.extract_mono_16k(input_path, mono)
    except ffmpeg.FFmpegError:
        data, sr = load_audio(input_path)
        stereo_data = ensure_stereo(resample_linear(data, sr, 48_000))
        mono_data = resample_linear(to_mono(data), sr, 16_000)
        write_audio(stereo, stereo_data, 48_000)
        write_audio(mono, mono_data[:, None] if mono_data.ndim == 1 else mono_data, 16_000)
    return stereo, mono


def probe_with_fallback(input_path: Path):
    try:
        return ffmpeg.probe_media(input_path)
    except ffmpeg.FFmpegError:
        from asmr_dub_pipeline.schemas import SourceInfo

        data, sr = load_audio(input_path)
        return SourceInfo(
            path=str(input_path),
            duration_sec=len(data) / sr,
            sample_rate=sr,
            channels=data.shape[1],
            codec="wav",
            format_name=input_path.suffix.lstrip(".") or "audio",
            has_video=False,
            raw={},
        )
