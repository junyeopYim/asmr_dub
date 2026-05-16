from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .features import duration_sec, ensure_stereo, load_audio, resample_linear, to_mono, write_audio
from .ffmpeg import FFmpegError, run_ffmpeg, wav_output_args

SourceSeparationBackend = Literal["auto", "none", "demucs", "mock"]


class SourceSeparationUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class SourceSeparationResult:
    backend: str
    model: str
    vocals_path: Path
    vocals_mono_path: Path
    background_path: Path
    metadata_path: Path
    reused_existing: bool
    command: list[str]


@dataclass(frozen=True)
class _PartwiseSeparationResult:
    part_index: int
    input_path: Path
    command: list[str]
    vocals_path: Path
    vocals_mono_path: Path
    background_path: Path
    duration_sec: float


def demucs_available() -> bool:
    return shutil.which("demucs") is not None or importlib.util.find_spec("demucs") is not None


def _demucs_command(
    input_audio_path: Path | Sequence[Path],
    output_dir: Path,
    model: str,
    device: str | None,
) -> list[str]:
    input_audio_paths = (
        [input_audio_path]
        if isinstance(input_audio_path, Path)
        else list(input_audio_path)
    )
    command = [shutil.which("demucs") or sys.executable]
    if command[0] == sys.executable:
        command.extend(["-m", "demucs.separate"])
    command.extend(["--two-stems", "vocals", "-n", model, "-o", str(output_dir)])
    if device:
        command.extend(["-d", device])
    command.extend(str(path) for path in input_audio_paths)
    return command


def _stage_demucs_batch_inputs(input_paths: Sequence[Path], batch_input_dir: Path) -> list[Path]:
    batch_input_dir.mkdir(parents=True, exist_ok=True)
    staged_paths: list[Path] = []
    for index, input_path in enumerate(input_paths, start=1):
        suffix = input_path.suffix or ".wav"
        staged_path = batch_input_dir / f"part_{index:04d}{suffix.lower()}"
        if staged_path.exists() or staged_path.is_symlink():
            staged_path.unlink()
        source_path = input_path.resolve()
        try:
            staged_path.symlink_to(source_path)
        except OSError:
            try:
                staged_path.hardlink_to(source_path)
            except OSError:
                shutil.copy2(source_path, staged_path)
        staged_paths.append(staged_path)
    return staged_paths


def _quote_ffmpeg_concat_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "'\\''")


def _concat_audio_files_streaming(
    input_paths: Sequence[Path],
    output_path: Path,
    *,
    sample_rate: int,
    channels: int,
) -> None:
    if not input_paths:
        raise SourceSeparationUnavailable("Cannot concatenate empty Demucs part outputs.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    list_path = output_path.with_suffix(output_path.suffix + ".concat.txt")
    list_path.write_text(
        "".join(f"file '{_quote_ffmpeg_concat_path(path)}'\n" for path in input_paths),
        "utf-8",
    )
    run_ffmpeg(
        [
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
            "-vn",
            "-ac",
            str(channels),
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            *wav_output_args(output_path),
        ]
    )


def _find_demucs_stem(output_dir: Path, model: str, input_stem: str, stem_name: str) -> Path:
    preferred = output_dir / model / input_stem / f"{stem_name}.wav"
    if preferred.exists():
        return preferred
    matches = sorted(output_dir.rglob(f"{stem_name}.wav"))
    if not matches:
        raise SourceSeparationUnavailable(
            f"Demucs finished but did not write {stem_name}.wav under {output_dir}"
        )
    return matches[-1]


def _convert_audio_streaming(
    input_path: Path,
    output_path: Path,
    *,
    sample_rate: int,
    channels: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(
        [
            "-y",
            "-i",
            str(input_path),
            "-vn",
            "-map",
            "0:a:0",
            "-ac",
            str(channels),
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            *wav_output_args(output_path),
        ]
    )


def _write_normalized_stems_streaming(
    vocals_source: Path,
    background_source: Path,
    vocals_path: Path,
    vocals_mono_path: Path,
    background_path: Path,
    sample_rate: int,
    mono_sample_rate: int,
) -> None:
    vocals_path.parent.mkdir(parents=True, exist_ok=True)
    background_path.parent.mkdir(parents=True, exist_ok=True)
    vocals_mono_path.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(
        [
            "-y",
            "-i",
            str(vocals_source),
            "-i",
            str(background_source),
            "-vn",
            "-map",
            "0:a:0",
            "-ac",
            "2",
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            *wav_output_args(vocals_path),
            "-map",
            "1:a:0",
            "-ac",
            "2",
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            *wav_output_args(background_path),
            "-map",
            "0:a:0",
            "-ac",
            "1",
            "-ar",
            str(mono_sample_rate),
            "-c:a",
            "pcm_s16le",
            *wav_output_args(vocals_mono_path),
        ]
    )


def _write_normalized_stems_in_memory(
    vocals_source: Path,
    background_source: Path,
    vocals_path: Path,
    vocals_mono_path: Path,
    background_path: Path,
    sample_rate: int,
    mono_sample_rate: int,
) -> None:
    vocals, vocals_sr = load_audio(vocals_source)
    background, background_sr = load_audio(background_source)
    if vocals_sr != sample_rate:
        vocals = resample_linear(vocals, vocals_sr, sample_rate)
    if background_sr != sample_rate:
        background = resample_linear(background, background_sr, sample_rate)
    vocals = ensure_stereo(vocals)
    background = ensure_stereo(background)
    write_audio(vocals_path, vocals, sample_rate)
    write_audio(background_path, background, sample_rate)
    mono = to_mono(vocals)
    if sample_rate != mono_sample_rate:
        mono = resample_linear(mono, sample_rate, mono_sample_rate)
    write_audio(vocals_mono_path, mono[:, None], mono_sample_rate)


def _write_normalized_stems(
    vocals_source: Path,
    background_source: Path,
    vocals_path: Path,
    vocals_mono_path: Path,
    background_path: Path,
    sample_rate: int,
    mono_sample_rate: int,
    *,
    allow_in_memory_fallback: bool = True,
) -> str:
    try:
        _write_normalized_stems_streaming(
            vocals_source,
            background_source,
            vocals_path,
            vocals_mono_path,
            background_path,
            sample_rate,
            mono_sample_rate,
        )
    except FFmpegError as exc:
        if not allow_in_memory_fallback:
            raise SourceSeparationUnavailable(
                "ffmpeg streaming postprocess failed; refusing in-memory fallback for part-wise source separation."
            ) from exc
        _write_normalized_stems_in_memory(
            vocals_source,
            background_source,
            vocals_path,
            vocals_mono_path,
            background_path,
            sample_rate,
            mono_sample_rate,
        )
        return "python_in_memory_fallback"
    return "ffmpeg_streaming"


def _write_mock_stems(
    input_audio_path: Path,
    vocals_path: Path,
    vocals_mono_path: Path,
    background_path: Path,
    sample_rate: int,
    mono_sample_rate: int,
) -> None:
    data, sr = load_audio(input_audio_path)
    if sr != sample_rate:
        data = resample_linear(data, sr, sample_rate)
    vocals = ensure_stereo(data)
    write_audio(vocals_path, vocals, sample_rate)
    write_audio(background_path, vocals * 0.0, sample_rate)
    mono = to_mono(vocals)
    if sample_rate != mono_sample_rate:
        mono = resample_linear(mono, sample_rate, mono_sample_rate)
    write_audio(vocals_mono_path, mono[:, None], mono_sample_rate)


def separate_source_audio(
    input_audio_path: Path,
    project_dir: Path,
    *,
    input_part_paths: Sequence[Path] | None = None,
    backend: SourceSeparationBackend = "auto",
    model: str = "htdemucs",
    device: str | None = None,
    sample_rate: int = 48_000,
    mono_sample_rate: int = 16_000,
    force: bool = False,
    runner: Any | None = None,
) -> SourceSeparationResult | None:
    if backend == "none":
        return None
    runner = runner or subprocess.run

    audio_dir = project_dir / "work" / "audio"
    separation_dir = project_dir / "work" / "source_separation"
    demucs_dir = separation_dir / "demucs"
    vocals_path = audio_dir / "source_vocals_48k.wav"
    vocals_mono_path = audio_dir / "source_vocals_mono_16k.wav"
    background_path = audio_dir / "background_only_48k.wav"
    metadata_path = separation_dir / "source_separation_manifest.json"
    outputs_exist = vocals_path.exists() and vocals_mono_path.exists() and background_path.exists()
    command: list[str] = []
    reused_existing = outputs_exist and not force
    selected_backend = backend
    selected_model = model
    postprocess_method = ""
    input_parts = [Path(path) for path in (input_part_paths or [])]
    partwise = len(input_parts) > 1
    part_records: list[dict[str, Any]] = []

    if reused_existing:
        if metadata_path.exists():
            try:
                previous_metadata = json.loads(metadata_path.read_text("utf-8"))
            except (OSError, json.JSONDecodeError, TypeError):
                previous_metadata = {}
            if isinstance(previous_metadata, dict):
                selected_backend = str(previous_metadata.get("backend") or selected_backend)
                selected_model = str(previous_metadata.get("model") or selected_model)
        if selected_backend == "auto":
            selected_backend = "cached"

    if not reused_existing:
        selected_backend = "demucs" if backend == "auto" and demucs_available() else backend
        if selected_backend == "auto":
            return None
        if selected_backend == "demucs" and not demucs_available():
            raise SourceSeparationUnavailable(
                "Demucs is not installed. Install the optional separation dependency or set "
                "source_separation_backend: none."
            )
        if selected_backend not in {"demucs", "mock"}:
            raise SourceSeparationUnavailable(f"Unsupported source separation backend: {selected_backend}")

        if selected_backend == "mock":
            _write_mock_stems(
                input_audio_path,
                vocals_path,
                vocals_mono_path,
                background_path,
                sample_rate,
                mono_sample_rate,
            )
        else:
            demucs_dir.mkdir(parents=True, exist_ok=True)
            if partwise:
                commands: list[list[str]] = []
                part_vocals: list[Path] = []
                part_vocals_mono: list[Path] = []
                part_backgrounds: list[Path] = []
                cursor_sec = 0.0
                batch_demucs_dir = demucs_dir / "parts_batch"
                batch_input_dir = separation_dir / "demucs_batch_inputs"
                staged_parts = _stage_demucs_batch_inputs(input_parts, batch_input_dir)
                batch_command = _demucs_command(staged_parts, batch_demucs_dir, model, device)
                batch_demucs_dir.mkdir(parents=True, exist_ok=True)
                try:
                    runner(batch_command, check=True, text=True)
                except subprocess.CalledProcessError as exc:
                    raise SourceSeparationUnavailable(f"Demucs source separation failed: {exc}") from exc

                def collect_part(index: int, part_path: Path, staged_path: Path) -> _PartwiseSeparationResult:
                    raw_vocals = _find_demucs_stem(batch_demucs_dir, model, staged_path.stem, "vocals")
                    raw_background = _find_demucs_stem(batch_demucs_dir, model, staged_path.stem, "no_vocals")
                    normalized_dir = separation_dir / "partwise" / f"part_{index:04d}"
                    part_vocals_path = normalized_dir / "source_vocals_48k.wav"
                    part_vocals_mono_path = normalized_dir / "source_vocals_mono_16k.wav"
                    part_background_path = normalized_dir / "background_only_48k.wav"
                    _write_normalized_stems(
                        raw_vocals,
                        raw_background,
                        part_vocals_path,
                        part_vocals_mono_path,
                        part_background_path,
                        sample_rate,
                        mono_sample_rate,
                        allow_in_memory_fallback=False,
                    )
                    part_duration = duration_sec(part_vocals_mono_path)
                    return _PartwiseSeparationResult(
                        part_index=index,
                        input_path=part_path,
                        command=batch_command,
                        vocals_path=part_vocals_path,
                        vocals_mono_path=part_vocals_mono_path,
                        background_path=part_background_path,
                        duration_sec=part_duration,
                    )

                part_results = [
                    collect_part(index, part_path, staged_path)
                    for index, (part_path, staged_path) in enumerate(
                        zip(input_parts, staged_parts, strict=True),
                        start=1,
                    )
                ]

                for part_result in part_results:
                    commands.append(part_result.command)
                    part_duration = part_result.duration_sec
                    part_vocals.append(part_result.vocals_path)
                    part_vocals_mono.append(part_result.vocals_mono_path)
                    part_backgrounds.append(part_result.background_path)
                    part_records.append(
                        {
                            "part_index": part_result.part_index,
                            "input_path": str(part_result.input_path),
                            "start_sec": round(cursor_sec, 6),
                            "end_sec": round(cursor_sec + part_duration, 6),
                            "duration_sec": round(part_duration, 6),
                            "vocals_path": str(part_result.vocals_path),
                            "vocals_mono_path": str(part_result.vocals_mono_path),
                            "background_path": str(part_result.background_path),
                        }
                    )
                    cursor_sec += part_duration
                command = commands[0] if commands else []
                _concat_audio_files_streaming(part_vocals, vocals_path, sample_rate=sample_rate, channels=2)
                _concat_audio_files_streaming(
                    part_vocals_mono,
                    vocals_mono_path,
                    sample_rate=mono_sample_rate,
                    channels=1,
                )
                _concat_audio_files_streaming(part_backgrounds, background_path, sample_rate=sample_rate, channels=2)
                postprocess_method = "partwise_batched_demucs_ffmpeg_streaming"
            else:
                command = _demucs_command(input_audio_path, demucs_dir, model, device)
                try:
                    runner(command, check=True, text=True)
                except subprocess.CalledProcessError as exc:
                    raise SourceSeparationUnavailable(f"Demucs source separation failed: {exc}") from exc
                vocals_source = _find_demucs_stem(demucs_dir, model, input_audio_path.stem, "vocals")
                background_source = _find_demucs_stem(demucs_dir, model, input_audio_path.stem, "no_vocals")
                postprocess_method = _write_normalized_stems(
                    vocals_source,
                    background_source,
                    vocals_path,
                    vocals_mono_path,
                    background_path,
                    sample_rate,
                    mono_sample_rate,
                )

    metadata: dict[str, Any] = {
        "backend": selected_backend,
        "model": selected_model,
        "input_audio_path": str(input_audio_path),
        "vocals_path": str(vocals_path),
        "vocals_mono_path": str(vocals_mono_path),
        "background_path": str(background_path),
        "reused_existing": reused_existing,
        "command": command,
    }
    if partwise:
        metadata["partwise"] = True
        metadata["input_part_paths"] = [str(path) for path in input_parts]
        metadata["parts"] = part_records
    if postprocess_method:
        metadata["postprocess_method"] = postprocess_method
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n", "utf-8")
    return SourceSeparationResult(
        backend=selected_backend,
        model=selected_model,
        vocals_path=vocals_path,
        vocals_mono_path=vocals_mono_path,
        background_path=background_path,
        metadata_path=metadata_path,
        reused_existing=reused_existing,
        command=command,
    )
