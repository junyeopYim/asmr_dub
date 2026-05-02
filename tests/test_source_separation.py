from __future__ import annotations

import json
import shutil
from pathlib import Path

from asmr_dub_pipeline.asr.base import ASRChunk
from asmr_dub_pipeline.audio import separation as separation_module
from asmr_dub_pipeline.audio.separation import separate_source_audio
from asmr_dub_pipeline.cli import app
from asmr_dub_pipeline.config import create_project_structure, save_project_config
from asmr_dub_pipeline.pipeline import steps as pipeline_steps
from asmr_dub_pipeline.pipeline.manifest_io import load_manifest
from asmr_dub_pipeline.pipeline.steps import (
    extract_step,
    segment_step,
    source_separation_step,
    transcribe_step,
)
from asmr_dub_pipeline.schemas import ProjectConfig


def _mock_separation_config(project: Path) -> None:
    create_project_structure(project)
    save_project_config(
        ProjectConfig(project_name=project.name, source_separation_backend="mock"),
        project / "pipeline.yaml",
    )


def test_source_separation_mock_writes_stems_and_segments_from_vocals(
    tiny_wav_path: Path,
    tmp_project_dir: Path,
) -> None:
    _mock_separation_config(tmp_project_dir)
    extract_step(tiny_wav_path, tmp_project_dir, confirm_rights=True)

    manifest = source_separation_step(tmp_project_dir, confirm_rights=True)

    assert manifest.stage_state["source-separation"]["status"] == "completed"
    assert Path(manifest.artifacts["source_vocals_48k"]).exists()
    assert Path(manifest.artifacts["source_vocals_mono_16k"]).exists()
    assert Path(manifest.artifacts["background_only_48k"]).exists()

    manifest = segment_step(tmp_project_dir)

    assert manifest.segments
    assert Path(manifest.segments[0].audio_for_mix).exists()
    assert Path(manifest.segments[0].audio_for_gemma).exists()


def test_real_transcribe_runs_source_separation_and_uses_vocal_mono(
    tiny_wav_path: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "transcribe_uses_vocals"
    create_project_structure(project)
    save_project_config(
        ProjectConfig(
            project_name=project.name,
            source_separation_backend="mock",
            asr_resegment_from_chunks=False,
        ),
        project / "pipeline.yaml",
    )
    extract_step(tiny_wav_path, project, confirm_rights=True)
    segment_step(project)
    captured: dict[str, Path] = {}

    class FakeASRBackend:
        name = "faster_whisper"

        def transcribe(self, audio_path: Path, segments: list[object]) -> list[ASRChunk]:
            captured["audio_path"] = Path(audio_path)
            return [
                ASRChunk(
                    start=0.0,
                    end=1.0,
                    text="テスト",
                    language="ja",
                    confidence=0.9,
                )
            ]

    monkeypatch.setattr(pipeline_steps, "create_asr_backend", lambda *_args, **_kwargs: FakeASRBackend())

    manifest = transcribe_step(project, asr_backend="faster_whisper", confirm_rights=True)

    assert manifest.stage_state["source-separation"]["status"] == "completed"
    assert captured["audio_path"] == Path(manifest.artifacts["source_vocals_mono_16k"])
    assert captured["audio_path"].name == "source_vocals_mono_16k.wav"
    assert manifest.stage_state["transcribe"]["status"] == "completed"


def test_source_separation_auto_reuses_existing_outputs_without_demucs(
    tiny_wav_path: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "separation_reuse"
    first = separate_source_audio(tiny_wav_path, project, backend="mock")
    assert first is not None

    monkeypatch.setattr(separation_module, "demucs_available", lambda: False)
    second = separate_source_audio(tiny_wav_path, project, backend="auto")

    assert second is not None
    assert second.reused_existing is True
    assert second.backend == "mock"


def test_full_imports_matching_voice_bank_source_separation_cache(
    cli_runner,
    tiny_wav_path: Path,
    tmp_path: Path,
) -> None:
    cache_project = tmp_path / "voice_bank_all"
    source_dir = cache_project / "voice_bank" / "sources" / f"src_0001_{tiny_wav_path.stem}"
    cache_extract_project = tmp_path / "cache_extract"
    extract_step(tiny_wav_path, cache_extract_project, confirm_rights=True)
    source_audio = source_dir / "source_stereo_48k.wav"
    source_audio.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cache_extract_project / "work" / "audio" / "original_stereo_48k.wav", source_audio)
    cached_result = separate_source_audio(source_audio, source_dir, backend="mock")
    assert cached_result is not None
    voice_bank_manifest = cache_project / "voice_bank" / "voice_bank_manifest.json"
    voice_bank_manifest.write_text(
        json.dumps({"source_paths": [str(tiny_wav_path.resolve())]}, ensure_ascii=False) + "\n",
        "utf-8",
    )

    project = tmp_path / "full_reuse"
    result = cli_runner.invoke(
        app,
        [
            "full",
            str(tiny_wav_path),
            "--project",
            str(project),
            "--confirm-rights",
            "--no-cache-status",
            "--source-separation-cache",
            str(cache_project),
        ],
    )

    assert result.exit_code == 0, result.output
    manifest = load_manifest(project)
    assert "imported cached voice-bank stems" in result.output
    assert manifest.stage_state["source-separation"]["status"] == "completed"
    assert manifest.stage_state["source-separation"]["backend"] == "cached"
    assert manifest.stage_state["source-separation"]["reused_existing"] is True
    assert Path(manifest.artifacts["source_separation_cache_import"]).exists()


def test_run_pipeline_uses_separated_background_when_available(
    cli_runner,
    tiny_wav_path: Path,
    tmp_path: Path,
) -> None:
    project = tmp_path / "source_separated_run"
    _mock_separation_config(project)

    result = cli_runner.invoke(
        app,
        [
            "run",
            str(tiny_wav_path),
            "--project",
            str(project),
            "--confirm-rights",
            "--mock",
        ],
    )

    assert result.exit_code == 0, result.output
    manifest = load_manifest(project)
    assert manifest.stage_state["source-separation"]["status"] == "completed"
    mix_manifest = json.loads(Path(manifest.artifacts["mix_manifest"]).read_text("utf-8"))
    assert mix_manifest["background"]["source_kind"] == "source_separated"
    assert mix_manifest["background"]["speech_suppression"]["enabled"] is True
    assert mix_manifest["background"]["speech_suppression"]["center_bleed_reduction"] is False
