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


def wav_output_args(output_path: Path) -> list[str]:
    if output_path.suffix.lower() == ".wav":
        return ["-rf64", "auto", str(output_path)]
    return [str(output_path)]


def extract_stereo_48k(input_path: Path, output_path: Path) -> Path:
    ensure_not_same_path(input_path, output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(["-y", "-i", str(input_path), "-vn", "-ac", "2", "-ar", "48000", *wav_output_args(output_path)])
    return output_path


def extract_mono_16k(input_path: Path, output_path: Path) -> Path:
    ensure_not_same_path(input_path, output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(["-y", "-i", str(input_path), "-vn", "-ac", "1", "-ar", "16000", *wav_output_args(output_path)])
    return output_path


def concat_audio_to_wav(
    input_paths: list[Path],
    output_path: Path,
    *,
    sample_rate: int = 48_000,
    channels: int = 2,
) -> Path:
    if len(input_paths) < 2:
        raise FFmpegError("At least two input files are required for audio concatenation.")
    for input_path in input_paths:
        ensure_not_same_path(input_path, output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    args = ["-y"]
    for input_path in input_paths:
        args += ["-i", str(input_path)]
    concat_inputs = "".join(f"[{index}:a:0]" for index in range(len(input_paths)))
    args += [
        "-filter_complex",
        f"{concat_inputs}concat=n={len(input_paths)}:v=0:a=1[a]",
        "-map",
        "[a]",
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate),
        *wav_output_args(output_path),
    ]
    run_ffmpeg(args)
    return output_path


def concat_audio_to_wav_with_silence(
    input_paths: list[Path],
    output_path: Path,
    *,
    silent_paths: list[Path],
    sample_rate: int = 48_000,
    channels: int = 2,
) -> Path:
    if not input_paths:
        raise FFmpegError("At least one input file is required for audio concatenation.")
    for input_path in input_paths:
        ensure_not_same_path(input_path, output_path)
    if channels == 1:
        channel_layout = "mono"
    elif channels == 2:
        channel_layout = "stereo"
    else:
        raise FFmpegError("Silence concat currently supports mono or stereo output.")

    silent_set = {path.resolve() for path in silent_paths}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    args = ["-y"]
    for input_path in input_paths:
        if input_path.resolve() in silent_set:
            duration = max(0.001, float(probe_media(input_path).duration_sec))
            args += [
                "-f",
                "lavfi",
                "-t",
                f"{duration:.6f}",
                "-i",
                f"anullsrc=r={sample_rate}:cl={channel_layout}",
            ]
        else:
            args += ["-i", str(input_path)]
    concat_inputs = "".join(f"[{index}:a:0]" for index in range(len(input_paths)))
    args += [
        "-filter_complex",
        f"{concat_inputs}concat=n={len(input_paths)}:v=0:a=1[a]",
        "-map",
        "[a]",
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate),
        *wav_output_args(output_path),
    ]
    run_ffmpeg(args)
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
    args += wav_output_args(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(args)
    return output_path


def _atempo_chain(tempo: float) -> str:
    if tempo <= 0:
        raise FFmpegError("atempo tempo must be greater than zero.")
    factors: list[float] = []
    remaining = tempo
    while remaining < 0.5:
        factors.append(0.5)
        remaining /= 0.5
    while remaining > 2.0:
        factors.append(2.0)
        remaining /= 2.0
    factors.append(remaining)
    return ",".join(f"atempo={factor:.6f}" for factor in factors)


def fit_audio_duration(
    input_path: Path,
    output_path: Path,
    *,
    target_duration_sec: float,
    sample_rate: int | None = None,
    channels: int | None = None,
) -> Path:
    ensure_not_same_path(input_path, output_path)
    if target_duration_sec <= 0:
        raise FFmpegError("target_duration_sec must be greater than zero.")
    source_duration_sec = probe_media(input_path).duration_sec
    if source_duration_sec <= 0:
        raise FFmpegError(f"Cannot time-fit audio with unknown duration: {input_path}")
    tempo = source_duration_sec / target_duration_sec
    args = ["-y", "-i", str(input_path), "-filter:a", _atempo_chain(tempo), "-vn"]
    if channels:
        args += ["-ac", str(channels)]
    if sample_rate:
        args += ["-ar", str(sample_rate)]
    args += wav_output_args(output_path)
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
