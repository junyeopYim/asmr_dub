from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .features import ensure_stereo, load_audio, resample_linear, to_mono, write_audio

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


def demucs_available() -> bool:
    return shutil.which("demucs") is not None or importlib.util.find_spec("demucs") is not None


def _demucs_command(
    input_audio_path: Path,
    output_dir: Path,
    model: str,
    device: str | None,
) -> list[str]:
    command = [shutil.which("demucs") or sys.executable]
    if command[0] == sys.executable:
        command.extend(["-m", "demucs.separate"])
    command.extend(["--two-stems", "vocals", "-n", model, "-o", str(output_dir)])
    if device:
        command.extend(["-d", device])
    command.append(str(input_audio_path))
    return command


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


def _write_normalized_stems(
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
    backend: SourceSeparationBackend = "auto",
    model: str = "htdemucs",
    device: str | None = None,
    sample_rate: int = 48_000,
    mono_sample_rate: int = 16_000,
    force: bool = False,
    runner: Any = subprocess.run,
) -> SourceSeparationResult | None:
    if backend == "none":
        return None
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

    if not reused_existing:
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
            command = _demucs_command(input_audio_path, demucs_dir, model, device)
            demucs_dir.mkdir(parents=True, exist_ok=True)
            try:
                runner(command, check=True, text=True)
            except subprocess.CalledProcessError as exc:
                raise SourceSeparationUnavailable(f"Demucs source separation failed: {exc}") from exc
            vocals_source = _find_demucs_stem(demucs_dir, model, input_audio_path.stem, "vocals")
            background_source = _find_demucs_stem(demucs_dir, model, input_audio_path.stem, "no_vocals")
            _write_normalized_stems(
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
        "model": model,
        "input_audio_path": str(input_audio_path),
        "vocals_path": str(vocals_path),
        "vocals_mono_path": str(vocals_mono_path),
        "background_path": str(background_path),
        "reused_existing": reused_existing,
        "command": command,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n", "utf-8")
    return SourceSeparationResult(
        backend=selected_backend,
        model=model,
        vocals_path=vocals_path,
        vocals_mono_path=vocals_mono_path,
        background_path=background_path,
        metadata_path=metadata_path,
        reused_existing=reused_existing,
        command=command,
    )
