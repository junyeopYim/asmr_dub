from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from asmr_dub_pipeline.rights import ensure_not_same_path
from asmr_dub_pipeline.schemas import SourceInfo


class FFmpegError(RuntimeError):
    pass


def require_binary(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise FFmpegError(f"{name} is required but was not found on PATH.")
    return path


def run_ffmpeg(args: list[str]) -> subprocess.CompletedProcess[str]:
    cmd = [require_binary("ffmpeg"), "-hide_banner", *args]
    try:
        return subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or str(exc)).strip()
        raise FFmpegError(f"ffmpeg failed: {detail}") from exc


def run_ffprobe(args: list[str]) -> subprocess.CompletedProcess[str]:
    cmd = [require_binary("ffprobe"), *args]
    try:
        return subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or str(exc)).strip()
        raise FFmpegError(f"ffprobe failed: {detail}") from exc


def _float_or_zero(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def probe_media(path: Path | str) -> SourceInfo:
    path = Path(path)
    try:
        result = run_ffprobe(
            [
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                str(path),
            ]
        )
        payload: dict[str, Any] = json.loads(result.stdout)
    except (FFmpegError, json.JSONDecodeError) as exc:
        raise FFmpegError(f"ffprobe failed for {path}: {exc}") from exc
    streams = payload.get("streams", [])
    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), {})
    has_video = any(s.get("codec_type") == "video" for s in streams)
    fmt = payload.get("format", {})
    duration = _float_or_zero(fmt.get("duration") or audio_stream.get("duration") or 0.0)
    sample_rate = audio_stream.get("sample_rate")
    bit_rate = fmt.get("bit_rate")
    return SourceInfo(
        path=str(path),
        duration_sec=duration,
        sample_rate=_int_or_none(sample_rate) if sample_rate else None,
        channels=_int_or_none(audio_stream.get("channels")) if audio_stream.get("channels") else None,
        codec=audio_stream.get("codec_name"),
        format_name=fmt.get("format_name"),
        has_video=has_video,
        bit_rate=_int_or_none(bit_rate) if bit_rate else None,
        raw=payload,
    )


def extract_stereo_48k(input_path: Path, output_path: Path) -> Path:
    ensure_not_same_path(input_path, output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(["-y", "-i", str(input_path), "-vn", "-ac", "2", "-ar", "48000", str(output_path)])
    return output_path


def extract_mono_16k(input_path: Path, output_path: Path) -> Path:
    ensure_not_same_path(input_path, output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(["-y", "-i", str(input_path), "-vn", "-ac", "1", "-ar", "16000", str(output_path)])
    return output_path


def slice_audio(
    input_path: Path,
    start_sec: float,
    end_sec: float,
    output_path: Path,
    sample_rate: int | None = None,
    channels: int | None = None,
) -> Path:
    ensure_not_same_path(input_path, output_path)
    args = ["-y", "-ss", f"{start_sec:.3f}", "-to", f"{end_sec:.3f}", "-i", str(input_path)]
    if channels:
        args += ["-ac", str(channels)]
    if sample_rate:
        args += ["-ar", str(sample_rate)]
    args += [str(output_path)]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(args)
    return output_path


def mux_audio(input_media: Path, final_audio: Path, output_path: Path) -> Path:
    ensure_not_same_path(input_media, output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    info = probe_media(input_media)
    if info.has_video:
        run_ffmpeg(
            [
                "-y",
                "-i",
                str(input_media),
                "-i",
                str(final_audio),
                "-map",
                "0:v?",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-shortest",
                str(output_path),
            ]
        )
    else:
        ensure_not_same_path(final_audio, output_path)
        shutil.copy2(final_audio, output_path)
    return output_path
