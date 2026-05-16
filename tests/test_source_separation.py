from __future__ import annotations

import json
import shutil
from io import StringIO
from pathlib import Path

import numpy as np
from rich.console import Console

from asmr_dub_pipeline.asr.base import ASRChunk, ASRWord
from asmr_dub_pipeline.audio import separation as separation_module
from asmr_dub_pipeline.audio.features import write_audio
from asmr_dub_pipeline.audio.separation import separate_source_audio
from asmr_dub_pipeline.cli import app
from asmr_dub_pipeline.config import create_project_structure, save_project_config
from asmr_dub_pipeline.pipeline import steps as pipeline_steps
from asmr_dub_pipeline.pipeline.manifest_io import load_manifest, save_manifest
from asmr_dub_pipeline.pipeline.stages import common as common_stage
from asmr_dub_pipeline.pipeline.steps import (
    extract_step,
    segment_step,
    source_separation_step,
    transcribe_step,
)
from asmr_dub_pipeline.schemas import PipelineManifest, ProjectConfig, Segment, SourceInfo


def _demucs_track_paths(command: list[str]) -> list[Path]:
    tail = command[command.index("-o") + 2 :]
    if tail[:1] == ["-d"]:
        tail = tail[2:]
    return [Path(raw) for raw in tail]


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
    captured_audio_paths: list[Path] = []

    class FakeASRBackend:
        name = "faster_whisper"

        def transcribe(self, audio_path: Path, segments: list[object]) -> list[ASRChunk]:
            captured_audio_paths.append(Path(audio_path))
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
    assert manifest.stage_state["transcribe-seed"]["status"] == "completed"
    assert captured_audio_paths[0] == Path(manifest.artifacts["source_vocals_mono_16k"])
    assert captured_audio_paths[0].name == "source_vocals_mono_16k.wav"
    assert manifest.stage_state["transcribe"]["status"] == "completed"


def test_real_transcribe_repairs_from_selected_asr_audio_when_vocals_are_longer_than_gemma(
    tiny_wav_path: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "transcribe_repairs_from_selected_vocals"
    create_project_structure(project)
    save_project_config(
        ProjectConfig(
            project_name=project.name,
            source_separation_backend="none",
            asr_resegment_from_chunks=False,
            asr_repair_enabled=True,
            asr_review_enabled=False,
        ),
        project / "pipeline.yaml",
    )
    extract_step(tiny_wav_path, project, confirm_rights=True)
    manifest = load_manifest(project)
    source_vocals_mono = project / "work" / "audio" / "source_vocals_mono_16k.wav"
    source_vocals_48k = project / "work" / "audio" / "source_vocals_48k.wav"
    write_audio(source_vocals_mono, np.full((2 * 16_000, 1), 0.1, dtype=np.float32), 16_000)
    write_audio(source_vocals_48k, np.full((2 * 48_000, 2), 0.1, dtype=np.float32), 48_000)
    manifest.artifacts["source_vocals_mono_16k"] = str(source_vocals_mono)
    manifest.artifacts["source_vocals_48k"] = str(source_vocals_48k)
    assert manifest.source_info is not None
    manifest.source_info.duration_sec = 2.0
    save_manifest(project, manifest)

    captured: dict[str, Path | float] = {}

    class FakeASRBackend:
        name = "faster_whisper"

        def transcribe(self, audio_path: Path, segments: list[object]) -> list[ASRChunk]:
            captured["audio_path"] = Path(audio_path)
            return [
                ASRChunk(
                    start=1.2,
                    end=1.8,
                    text="付属しますね",
                    language="ja",
                    confidence=0.7,
                )
            ]

        def transcribe_with_options(
            self,
            audio_path: Path,
            segments: list[object],
            **_kwargs: object,
        ) -> list[ASRChunk]:
            _ = audio_path, segments
            return []

    def fake_repair_asr_chunks(
        chunks: list[ASRChunk],
        **kwargs: object,
    ) -> tuple[list[ASRChunk], dict[str, object]]:
        captured["repair_audio_path"] = Path(kwargs["repair_audio_path"])
        captured["audio_duration_sec"] = float(kwargs["audio_duration_sec"])
        return chunks, {"enabled": True, "attempted": 0, "repaired": 0, "skipped": 0, "items": []}

    monkeypatch.setattr(pipeline_steps, "create_asr_backend", lambda *_args, **_kwargs: FakeASRBackend())
    monkeypatch.setattr(pipeline_steps, "_repair_asr_chunks", fake_repair_asr_chunks)

    transcribe_step(project, asr_backend="faster_whisper", confirm_rights=True)

    assert captured["audio_path"] == source_vocals_mono
    assert captured["repair_audio_path"] == source_vocals_mono
    assert captured["audio_duration_sec"] == 2.0


def test_real_transcribe_defers_seed_audio_when_resegmenting(
    tiny_wav_path: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "transcribe_defers_seed_audio"
    create_project_structure(project)
    save_project_config(
        ProjectConfig(
            project_name=project.name,
            source_separation_backend="none",
            asr_resegment_from_chunks=True,
        ),
        project / "pipeline.yaml",
    )
    extract_step(tiny_wav_path, project, confirm_rights=True)

    class FakeASRBackend:
        name = "faster_whisper"

        def transcribe(self, audio_path: Path, segments: list[object]) -> list[ASRChunk]:
            _ = audio_path, segments
            return [
                ASRChunk(
                    start=0.0,
                    end=0.5,
                    text="テスト",
                    language="ja",
                    confidence=0.9,
                )
            ]

    monkeypatch.setattr(pipeline_steps, "create_asr_backend", lambda *_args, **_kwargs: FakeASRBackend())

    manifest = transcribe_step(project, asr_backend="faster_whisper", confirm_rights=True)

    assert manifest.stage_state["transcribe-seed"]["audio_clips_written"] is False
    assert manifest.stage_state["transcribe"]["resegmented_from_chunks"] is True


def test_real_transcribe_falls_back_to_gemma_when_source_vocals_are_too_quiet(
    tiny_wav_path: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "transcribe_fallback_gemma"
    create_project_structure(project)
    save_project_config(
        ProjectConfig(
            project_name=project.name,
            source_separation_backend="none",
            asr_resegment_from_chunks=False,
            asr_repair_enabled=False,
        ),
        project / "pipeline.yaml",
    )
    extract_step(tiny_wav_path, project, confirm_rights=True)
    manifest = load_manifest(project)
    silent_vocals = project / "work" / "audio" / "source_vocals_mono_16k.wav"
    write_audio(silent_vocals, np.zeros((16_000, 1), dtype=np.float32), 16_000)
    manifest.artifacts["source_vocals_mono_16k"] = str(silent_vocals)
    manifest.artifacts["source_vocals_48k"] = manifest.artifacts["original_stereo_48k"]
    save_manifest(project, manifest)
    captured: dict[str, Path] = {}

    class FakeASRBackend:
        name = "faster_whisper"

        def transcribe(self, audio_path: Path, segments: list[object]) -> list[ASRChunk]:
            captured["audio_path"] = Path(audio_path)
            return [ASRChunk(start=0.0, end=1.0, text="テスト", language="ja", confidence=0.9)]

    monkeypatch.setattr(pipeline_steps, "create_asr_backend", lambda *_args, **_kwargs: FakeASRBackend())

    manifest = transcribe_step(project, asr_backend="faster_whisper", confirm_rights=True)

    assert captured["audio_path"] == Path(manifest.artifacts["gemma_mono_16k"])
    input_summary = json.loads(Path(manifest.artifacts["asr_input_diagnostics"]).read_text("utf-8"))
    assert input_summary["selected"]["source"] == "gemma_mono_16k"
    assert any("too_quiet" in warning for warning in input_summary["warnings"])
    assert "asr_diagnostics" in manifest.artifacts


def test_real_transcribe_falls_back_when_source_vocals_are_quiet_relative_to_gemma(
    tiny_wav_path: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "transcribe_relative_quiet_fallback"
    create_project_structure(project)
    save_project_config(
        ProjectConfig(
            project_name=project.name,
            source_separation_backend="none",
            asr_resegment_from_chunks=False,
            asr_repair_enabled=False,
        ),
        project / "pipeline.yaml",
    )
    extract_step(tiny_wav_path, project, confirm_rights=True)
    manifest = load_manifest(project)
    gemma_path = Path(manifest.artifacts["gemma_mono_16k"])
    sample_rate = 16_000
    t = np.linspace(0.0, 1.0, sample_rate, endpoint=False, dtype=np.float32)
    write_audio(gemma_path, (0.10 * np.sin(2 * np.pi * 220 * t))[:, None], sample_rate)
    quiet_vocals = project / "work" / "audio" / "source_vocals_mono_16k.wav"
    write_audio(quiet_vocals, (0.001 * np.sin(2 * np.pi * 220 * t))[:, None], sample_rate)
    manifest.artifacts["source_vocals_mono_16k"] = str(quiet_vocals)
    manifest.artifacts["source_vocals_48k"] = manifest.artifacts["original_stereo_48k"]
    save_manifest(project, manifest)
    captured: dict[str, Path] = {}

    class FakeASRBackend:
        name = "faster_whisper"

        def transcribe(self, audio_path: Path, segments: list[object]) -> list[ASRChunk]:
            captured["audio_path"] = Path(audio_path)
            return [ASRChunk(start=0.0, end=1.0, text="テスト", language="ja", confidence=0.9)]

    monkeypatch.setattr(pipeline_steps, "create_asr_backend", lambda *_args, **_kwargs: FakeASRBackend())

    manifest = transcribe_step(project, asr_backend="faster_whisper", confirm_rights=True)

    assert captured["audio_path"] == Path(manifest.artifacts["gemma_mono_16k"])
    input_summary = json.loads(Path(manifest.artifacts["asr_input_diagnostics"]).read_text("utf-8"))
    assert input_summary["selected"]["source"] == "gemma_mono_16k"
    assert any("relative_too_quiet" in warning for warning in input_summary["warnings"])


def test_transcribe_passes_project_asr_config_to_backend(
    tiny_wav_path: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "transcribe_config"
    create_project_structure(project)
    save_project_config(
        ProjectConfig(
            project_name=project.name,
            source_separation_backend="none",
            asr_resegment_from_chunks=False,
            asr_repair_enabled=False,
            asr_model_id="custom/fw",
            asr_language="ja",
            asr_device="cuda",
            asr_compute_type="float16",
            asr_batched_inference=True,
            asr_batch_size=16,
            asr_beam_size=7,
            asr_best_of=6,
            asr_condition_on_previous_text=False,
            asr_vad_filter=True,
            asr_vad_parameters={"threshold": 0.25, "speech_pad_ms": 640},
            asr_word_timestamps=True,
            asr_hallucination_silence_threshold=0.75,
            asr_initial_prompt="絶頂 媚薬",
            asr_hotwords="耳舐め",
        ),
        project / "pipeline.yaml",
    )
    extract_step(tiny_wav_path, project, confirm_rights=True)
    captured: dict[str, object] = {}

    class FakeASRBackend:
        name = "faster_whisper"

        def transcribe(self, _audio_path: Path, _segments: list[object]) -> list[ASRChunk]:
            return [ASRChunk(start=0.0, end=1.0, text="テスト", language="ja", confidence=0.9)]

    def fake_create(kind: str, config: dict[str, object]) -> FakeASRBackend:
        captured["kind"] = kind
        captured["config"] = config
        return FakeASRBackend()

    monkeypatch.setattr(pipeline_steps, "create_asr_backend", fake_create)

    transcribe_step(project, asr_backend="faster_whisper", confirm_rights=True)

    assert captured["kind"] == "faster_whisper"
    config = captured["config"]
    assert isinstance(config, dict)
    assert config["model_id"] == "custom/fw"
    assert config["device"] == "cuda"
    assert config["compute_type"] == "float16"
    assert config["batched_inference"] is True
    assert config["batch_size"] == 16
    assert config["beam_size"] == 7
    assert config["best_of"] == 6
    assert config["vad_parameters"] == {"threshold": 0.25, "speech_pad_ms": 640}
    assert config["word_timestamps"] is True
    assert config["hallucination_silence_threshold"] == 0.75
    assert config["initial_prompt"] == "絶頂 媚薬"
    assert config["hotwords"] == "耳舐め"


def test_transcribe_step_accepts_injected_asr_backend_factory(
    tiny_wav_path: Path,
    tmp_path: Path,
) -> None:
    project = tmp_path / "transcribe_injected_factory"
    create_project_structure(project)
    save_project_config(
        ProjectConfig(
            project_name=project.name,
            source_separation_backend="none",
            asr_resegment_from_chunks=False,
            asr_repair_enabled=False,
        ),
        project / "pipeline.yaml",
    )
    extract_step(tiny_wav_path, project, confirm_rights=True)
    calls: list[tuple[str, dict[str, object]]] = []

    class FakeASRBackend:
        name = "faster_whisper"

        def transcribe(self, _audio_path: Path, _segments: list[object]) -> list[ASRChunk]:
            return [ASRChunk(start=0.0, end=1.0, text="テスト", language="ja", confidence=0.9)]

    def fake_factory(kind: str, config: dict[str, object]) -> FakeASRBackend:
        calls.append((kind, dict(config)))
        return FakeASRBackend()

    manifest = transcribe_step(
        project,
        asr_backend="faster_whisper",
        confirm_rights=True,
        asr_backend_factory=fake_factory,
    )

    assert manifest.stage_state["transcribe"]["status"] == "completed"
    assert calls
    assert calls[0][0] == "faster_whisper"
    assert calls[0][1]["model_id"] == "Systran/faster-whisper-large-v3"


def test_transcribe_logs_major_progress_checkpoints(
    tiny_wav_path: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "transcribe_progress_logs"
    create_project_structure(project)
    save_project_config(
        ProjectConfig(
            project_name=project.name,
            source_separation_backend="none",
            asr_resegment_from_chunks=False,
            asr_repair_enabled=False,
            asr_review_enabled=False,
        ),
        project / "pipeline.yaml",
    )
    extract_step(tiny_wav_path, project, confirm_rights=True)
    segment_step(project)
    manifest = load_manifest(project)
    manifest.artifacts["source_vocals_mono_16k"] = manifest.artifacts["gemma_mono_16k"]
    manifest.artifacts["source_vocals_48k"] = manifest.artifacts["original_stereo_48k"]
    save_manifest(project, manifest)

    output = StringIO()
    monkeypatch.setattr(
        common_stage,
        "console",
        Console(file=output, force_terminal=False, color_system=None, width=240),
    )

    class FakeASRBackend:
        name = "faster_whisper"

        def transcribe(self, _audio_path: Path, _segments: list[object]) -> list[ASRChunk]:
            return [ASRChunk(start=0.0, end=1.0, text="テスト", language="ja", confidence=0.9)]

    monkeypatch.setattr(pipeline_steps, "create_asr_backend", lambda *_args, **_kwargs: FakeASRBackend())

    transcribe_step(project, asr_backend="faster_whisper", confirm_rights=True)

    rendered = output.getvalue()
    assert "transcribe: audio selected" in rendered
    assert "transcribe: starting ASR backend=faster_whisper" in rendered
    assert "transcribe: ASR complete raw_chunks=1" in rendered
    assert "transcribe: applying ASR post-processing" in rendered
    assert "transcribe: writing transcription artifacts" in rendered


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


def test_demucs_postprocess_uses_ffmpeg_streaming_without_loading_stems(
    tiny_wav_path: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "separation_streaming"
    model = "htdemucs"
    ffmpeg_calls: list[list[str]] = []

    def fake_demucs_runner(command: list[str], check: bool, text: bool) -> None:
        assert check is True
        assert text is True
        output_dir = Path(command[command.index("-o") + 1])
        input_stem = Path(command[-1]).stem
        stem_dir = output_dir / model / input_stem
        stem_dir.mkdir(parents=True, exist_ok=True)
        write_audio(stem_dir / "vocals.wav", np.full((4_410, 2), 0.05, dtype=np.float32), 44_100)
        write_audio(stem_dir / "no_vocals.wav", np.zeros((4_410, 2), dtype=np.float32), 44_100)

    def fake_run_ffmpeg(args: list[str]) -> None:
        ffmpeg_calls.append(args)
        outputs = {
            project / "work" / "audio" / "source_vocals_48k.wav": (48_000, 2),
            project / "work" / "audio" / "background_only_48k.wav": (48_000, 2),
            project / "work" / "audio" / "source_vocals_mono_16k.wav": (16_000, 1),
        }
        for output_path, (sample_rate, channels) in outputs.items():
            if str(output_path) in args:
                write_audio(output_path, np.zeros((sample_rate // 20, channels), dtype=np.float32), sample_rate)

    def fail_load_audio(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("demucs postprocess should stream via ffmpeg")

    monkeypatch.setattr(separation_module, "demucs_available", lambda: True)
    monkeypatch.setattr(separation_module, "run_ffmpeg", fake_run_ffmpeg)
    monkeypatch.setattr(separation_module, "load_audio", fail_load_audio)

    result = separate_source_audio(tiny_wav_path, project, backend="demucs", model=model, runner=fake_demucs_runner)

    assert result is not None
    assert result.backend == "demucs"
    assert result.vocals_path.exists()
    assert result.vocals_mono_path.exists()
    assert result.background_path.exists()
    assert len(ffmpeg_calls) == 1
    assert [value for index, value in enumerate(ffmpeg_calls[0]) if index > 0 and ffmpeg_calls[0][index - 1] == "-map"] == [
        "0:a:0",
        "1:a:0",
        "0:a:0",
    ]
    assert [value for index, value in enumerate(ffmpeg_calls[0]) if index > 0 and ffmpeg_calls[0][index - 1] == "-ac"] == [
        "2",
        "2",
        "1",
    ]
    assert [value for index, value in enumerate(ffmpeg_calls[0]) if index > 0 and ffmpeg_calls[0][index - 1] == "-ar"] == [
        "48000",
        "48000",
        "16000",
    ]
    metadata = json.loads(result.metadata_path.read_text("utf-8"))
    assert metadata["postprocess_method"] == "ffmpeg_streaming"


def test_demucs_separates_folder_parts_before_concatenating_stems(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "folder_partwise_separation"
    part_1 = tmp_path / "folder" / "01 intro.wav"
    part_2 = tmp_path / "folder" / "02 main.wav"
    write_audio(part_1, np.full((4_410, 2), 0.05, dtype=np.float32), 44_100)
    write_audio(part_2, np.full((4_410, 2), 0.10, dtype=np.float32), 44_100)
    merged_input = project / "work" / "audio" / "original_stereo_48k.wav"
    write_audio(merged_input, np.zeros((9_600, 2), dtype=np.float32), 48_000)
    model = "htdemucs"
    demucs_inputs: list[Path] = []

    def fake_demucs_runner(command: list[str], check: bool, text: bool) -> None:
        assert check is True
        assert text is True
        output_dir = Path(command[command.index("-o") + 1])
        for input_path in _demucs_track_paths(command):
            demucs_inputs.append(input_path)
            stem_dir = output_dir / model / input_path.stem
            stem_dir.mkdir(parents=True, exist_ok=True)
            write_audio(stem_dir / "vocals.wav", np.full((4_410, 2), 0.05, dtype=np.float32), 44_100)
            write_audio(stem_dir / "no_vocals.wav", np.zeros((4_410, 2), dtype=np.float32), 44_100)

    def fake_run_ffmpeg(args: list[str]) -> None:
        for raw in args:
            output_path = Path(raw)
            if project not in output_path.parents or output_path.suffix != ".wav":
                continue
            if "partwise" not in output_path.parts and output_path.parent != project / "work" / "audio":
                continue
            channels = 1 if "mono_16k" in output_path.name else 2
            sample_rate = 16_000 if channels == 1 else 48_000
            value = 0.05 if "source_vocals" in output_path.name else 0.0
            write_audio(
                output_path,
                np.full((sample_rate // 10, channels), value, dtype=np.float32),
                sample_rate,
            )

    monkeypatch.setattr(separation_module, "demucs_available", lambda: True)
    monkeypatch.setattr(separation_module, "run_ffmpeg", fake_run_ffmpeg)

    result = separate_source_audio(
        merged_input,
        project,
        backend="demucs",
        model=model,
        runner=fake_demucs_runner,
        input_part_paths=[part_1, part_2],
    )

    assert result is not None
    assert [path.stem for path in demucs_inputs] == ["part_0001", "part_0002"]
    assert len(demucs_inputs) == 2
    assert merged_input not in demucs_inputs
    metadata = json.loads(result.metadata_path.read_text("utf-8"))
    assert metadata["partwise"] is True
    assert metadata["input_part_paths"] == [str(part_1), str(part_2)]


def test_demucs_folder_partwise_separation_batches_parts_in_single_process(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "folder_partwise_batch_separation"
    folder = tmp_path / "folder"
    parts = [folder / f"{index:02d} part.wav" for index in range(1, 9)]
    for index, part in enumerate(parts, start=1):
        write_audio(part, np.full((4_410, 2), 0.01 * index, dtype=np.float32), 44_100)
    merged_input = project / "work" / "audio" / "original_stereo_48k.wav"
    write_audio(merged_input, np.zeros((48_000, 2), dtype=np.float32), 48_000)
    model = "htdemucs"
    demucs_commands: list[list[str]] = []

    def fake_demucs_runner(command: list[str], check: bool, text: bool) -> None:
        assert check is True
        assert text is True
        demucs_commands.append(command)
        output_dir = Path(command[command.index("-o") + 1])
        for input_path in _demucs_track_paths(command):
            stem_dir = output_dir / model / input_path.stem
            stem_dir.mkdir(parents=True, exist_ok=True)
            write_audio(stem_dir / "vocals.wav", np.full((4_410, 2), 0.05, dtype=np.float32), 44_100)
            write_audio(stem_dir / "no_vocals.wav", np.zeros((4_410, 2), dtype=np.float32), 44_100)

    def fake_run_ffmpeg(args: list[str]) -> None:
        for raw in args:
            output_path = Path(raw)
            if project not in output_path.parents or output_path.suffix != ".wav":
                continue
            if "partwise" not in output_path.parts and output_path.parent != project / "work" / "audio":
                continue
            channels = 1 if "mono_16k" in output_path.name else 2
            sample_rate = 16_000 if channels == 1 else 48_000
            value = 0.05 if "source_vocals" in output_path.name else 0.0
            write_audio(
                output_path,
                np.full((sample_rate // 20, channels), value, dtype=np.float32),
                sample_rate,
            )

    monkeypatch.setattr(separation_module, "demucs_available", lambda: True)
    monkeypatch.setattr(separation_module, "run_ffmpeg", fake_run_ffmpeg)

    result = separate_source_audio(
        merged_input,
        project,
        backend="demucs",
        model=model,
        runner=fake_demucs_runner,
        input_part_paths=parts,
    )

    assert result is not None
    assert len(demucs_commands) == 1
    demucs_inputs = _demucs_track_paths(demucs_commands[0])
    assert [path.stem for path in demucs_inputs] == [f"part_{index:04d}" for index in range(1, 9)]
    assert all(path.exists() for path in demucs_inputs)
    assert not any(path in parts for path in demucs_inputs)
    metadata = json.loads(result.metadata_path.read_text("utf-8"))
    assert metadata["parts"] == sorted(metadata["parts"], key=lambda part: part["part_index"])
    assert metadata["input_part_paths"] == [str(part) for part in parts]


def test_source_separation_stage_uses_folder_mix_parts_for_demucs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "folder_stage_partwise_separation"
    folder = tmp_path / "RJTEST"
    part_1 = folder / "本編" / "01 intro.wav"
    part_2 = folder / "本編" / "02 main.wav"
    write_audio(part_1, np.full((4_410, 2), 0.05, dtype=np.float32), 44_100)
    write_audio(part_2, np.full((4_410, 2), 0.10, dtype=np.float32), 44_100)
    create_project_structure(project)
    save_project_config(
        ProjectConfig(project_name=project.name, source_separation_backend="demucs"),
        project / "pipeline.yaml",
    )
    extract_step(folder, project, confirm_rights=True)
    model = "htdemucs"
    demucs_inputs: list[Path] = []

    def fake_demucs_runner(command: list[str], check: bool, text: bool) -> None:
        assert check is True
        assert text is True
        output_dir = Path(command[command.index("-o") + 1])
        for input_path in _demucs_track_paths(command):
            demucs_inputs.append(input_path)
            stem_dir = output_dir / model / input_path.stem
            stem_dir.mkdir(parents=True, exist_ok=True)
            write_audio(stem_dir / "vocals.wav", np.full((4_410, 2), 0.05, dtype=np.float32), 44_100)
            write_audio(stem_dir / "no_vocals.wav", np.zeros((4_410, 2), dtype=np.float32), 44_100)

    def fake_run_ffmpeg(args: list[str]) -> None:
        for raw in args:
            output_path = Path(raw)
            if project not in output_path.parents or output_path.suffix != ".wav":
                continue
            if "partwise" not in output_path.parts and output_path.parent != project / "work" / "audio":
                continue
            channels = 1 if "mono_16k" in output_path.name else 2
            sample_rate = 16_000 if channels == 1 else 48_000
            value = 0.05 if "source_vocals" in output_path.name else 0.0
            write_audio(
                output_path,
                np.full((sample_rate // 10, channels), value, dtype=np.float32),
                sample_rate,
            )

    monkeypatch.setattr(separation_module, "demucs_available", lambda: True)
    monkeypatch.setattr(separation_module, "run_ffmpeg", fake_run_ffmpeg)
    monkeypatch.setattr(separation_module.subprocess, "run", fake_demucs_runner)

    manifest = source_separation_step(project, confirm_rights=True)

    assert manifest.stage_state["source-separation"]["status"] == "completed"
    assert [path.stem for path in demucs_inputs] == ["part_0001", "part_0002"]
    assert len(demucs_inputs) == 2
    assert Path(manifest.artifacts["original_stereo_48k"]) not in demucs_inputs


def test_transcribe_uses_source_separated_folder_part_audio(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "folder_transcribe_partwise"
    folder = tmp_path / "RJTRANSCRIBE"
    part_1 = folder / "本編" / "01 intro.wav"
    part_2 = folder / "本編" / "02 main.wav"
    write_audio(part_1, np.full((4_410, 2), 0.05, dtype=np.float32), 44_100)
    write_audio(part_2, np.full((8_820, 2), 0.10, dtype=np.float32), 44_100)
    create_project_structure(project)
    save_project_config(
        ProjectConfig(
            project_name=project.name,
            source_separation_backend="demucs",
            asr_resegment_from_chunks=True,
            asr_repair_enabled=False,
            asr_review_enabled=False,
            asr_input_duration_tolerance=1.0,
            asr_resegment_min_sec=0.01,
            asr_resegment_max_sec=0.08,
            asr_resegment_merge_gap_sec=0.0,
        ),
        project / "pipeline.yaml",
    )
    extract_step(folder, project, confirm_rights=True)
    model = "htdemucs"

    def fake_demucs_runner(command: list[str], check: bool, text: bool) -> None:
        assert check is True
        assert text is True
        output_dir = Path(command[command.index("-o") + 1])
        for input_path in _demucs_track_paths(command):
            stem_dir = output_dir / model / input_path.stem
            stem_dir.mkdir(parents=True, exist_ok=True)
            write_audio(stem_dir / "vocals.wav", np.full((4_410, 2), 0.05, dtype=np.float32), 44_100)
            write_audio(stem_dir / "no_vocals.wav", np.zeros((4_410, 2), dtype=np.float32), 44_100)

    def fake_run_ffmpeg(args: list[str]) -> None:
        for raw in args:
            output_path = Path(raw)
            if project not in output_path.parents or output_path.suffix != ".wav":
                continue
            if "partwise" not in output_path.parts and output_path.parent != project / "work" / "audio":
                continue
            channels = 1 if "mono_16k" in output_path.name else 2
            sample_rate = 16_000 if channels == 1 else 48_000
            value = 0.05 if "source_vocals" in output_path.name else 0.0
            frames = sample_rate // 10
            write_audio(
                output_path,
                np.full((frames, channels), value, dtype=np.float32),
                sample_rate,
            )

    monkeypatch.setattr(separation_module, "demucs_available", lambda: True)
    monkeypatch.setattr(separation_module, "run_ffmpeg", fake_run_ffmpeg)
    monkeypatch.setattr(separation_module.subprocess, "run", fake_demucs_runner)
    source_separation_step(project, confirm_rights=True)
    manifest = load_manifest(project)
    separation_metadata = json.loads(Path(manifest.artifacts["source_separation_manifest"]).read_text("utf-8"))
    part_mono_paths = [Path(part["vocals_mono_path"]) for part in separation_metadata["parts"]]
    captured_audio_paths: list[Path] = []

    class FakeASRBackend:
        name = "faster_whisper"

        def transcribe(self, audio_path: Path, segments: list[object]) -> list[ASRChunk]:
            _ = segments
            captured_audio_paths.append(Path(audio_path))
            return [
                ASRChunk(
                    start=0.01,
                    end=0.08,
                    text=f"テスト{len(captured_audio_paths)}",
                    language="ja",
                    confidence=0.9,
                )
            ]

    monkeypatch.setattr(pipeline_steps, "create_asr_backend", lambda *_args, **_kwargs: FakeASRBackend())

    manifest = transcribe_step(project, asr_backend="faster_whisper", confirm_rights=True)

    assert captured_audio_paths == part_mono_paths
    assert Path(manifest.artifacts["source_vocals_mono_16k"]) not in captured_audio_paths
    assert len(manifest.segments) == 2
    assert all(Path(segment.audio_for_gemma).exists() for segment in manifest.segments)
    assert all(Path(segment.audio_for_mix).exists() for segment in manifest.segments)
    assert manifest.segments[0].start >= 0.0
    assert manifest.segments[-1].start > 0.0


def test_partwise_transcribe_offsets_word_timestamps(tmp_path: Path) -> None:
    from asmr_dub_pipeline.pipeline.stages.transcribe import _transcribe_partwise_audio

    part_1 = tmp_path / "part_1.wav"
    part_2 = tmp_path / "part_2.wav"
    write_audio(part_1, np.zeros((160, 1), dtype=np.float32), 16_000)
    write_audio(part_2, np.zeros((160, 1), dtype=np.float32), 16_000)

    class FakeASRBackend:
        def transcribe(self, audio_path: Path, segments: list[object]) -> list[ASRChunk]:
            _ = segments
            token = "5" if audio_path == part_1 else "4"
            return [
                ASRChunk(
                    start=0.1,
                    end=0.7,
                    text=token,
                    language="ja",
                    confidence=0.9,
                    words=[
                        ASRWord(start=0.12, end=0.32, text=token, confidence=0.95),
                    ],
                )
            ]

    chunks = _transcribe_partwise_audio(
        FakeASRBackend(),
        [
            {"start_sec": 10.0, "vocals_mono_path": str(part_1)},
            {"start_sec": 20.0, "vocals_mono_path": str(part_2)},
        ],
    )

    assert [(chunk.start, chunk.end, chunk.text) for chunk in chunks] == [
        (10.1, 10.7, "5"),
        (20.1, 20.7, "4"),
    ]
    assert [(word.start, word.end, word.text) for chunk in chunks for word in chunk.words] == [
        (10.12, 10.32, "5"),
        (20.12, 20.32, "4"),
    ]


def test_partwise_transcribe_skips_asr_silenced_parts(tmp_path: Path) -> None:
    from asmr_dub_pipeline.pipeline.stages.transcribe import _transcribe_partwise_audio

    spoken = tmp_path / "spoken.wav"
    effect_only = tmp_path / "効果音トラック02e.wav"
    write_audio(spoken, np.zeros((160, 1), dtype=np.float32), 16_000)
    write_audio(effect_only, np.zeros((160, 1), dtype=np.float32), 16_000)
    captured_paths: list[Path] = []

    class FakeASRBackend:
        def transcribe(self, audio_path: Path, segments: list[object]) -> list[ASRChunk]:
            _ = segments
            captured_paths.append(audio_path)
            return [ASRChunk(start=0.1, end=0.7, text="声", language="ja", confidence=0.9)]

    chunks = _transcribe_partwise_audio(
        FakeASRBackend(),
        [
            {"start_sec": 10.0, "vocals_mono_path": str(spoken)},
            {
                "start_sec": 20.0,
                "vocals_mono_path": str(effect_only),
                "asr_silenced": True,
                "asr_skip_reason": "effect_only",
            },
        ],
    )

    assert captured_paths == [spoken]
    assert [(chunk.start, chunk.end, chunk.text) for chunk in chunks] == [(10.1, 10.7, "声")]


def test_transcribe_marks_segments_in_asr_silenced_folder_parts_no_speech(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "transcribe_skip_effect_only_part"
    create_project_structure(project)
    save_project_config(
        ProjectConfig(
            project_name=project.name,
            source_separation_backend="none",
            asr_resegment_from_chunks=False,
            asr_repair_enabled=False,
            asr_review_enabled=False,
            asr_input_duration_tolerance=1.0,
        ),
        project / "pipeline.yaml",
    )
    audio_dir = project / "work" / "audio"
    gemma_audio = audio_dir / "gemma_mono_16k.wav"
    mix_audio = audio_dir / "original_stereo_48k.wav"
    write_audio(gemma_audio, np.zeros((32_000, 1), dtype=np.float32), 16_000)
    write_audio(mix_audio, np.zeros((96_000, 2), dtype=np.float32), 48_000)
    manifest = PipelineManifest(
        source_info=SourceInfo(
            path=str(mix_audio),
            duration_sec=2.0,
            sample_rate=48_000,
            channels=2,
            has_video=False,
            format_name="wav",
            raw={
                "folder_input": {
                    "input_kind": "folder",
                    "asr_source_status": "mix_parts_silenced_for_asr",
                    "asr_silent_part_count": 1,
                    "asr_parts": [
                        {
                            "part_index": 1,
                            "path": str(tmp_path / "spoken.wav"),
                            "stem": "spoken",
                            "start_sec": 0.0,
                            "end_sec": 1.0,
                        },
                        {
                            "part_index": 2,
                            "path": str(tmp_path / "効果音トラック02e.mp3"),
                            "stem": "効果音トラック02e",
                            "start_sec": 1.0,
                            "end_sec": 2.0,
                            "asr_silenced": True,
                            "asr_skip_reason": "effect_only",
                        },
                    ],
                }
            },
        ),
        artifacts={"gemma_mono_16k": str(gemma_audio), "original_stereo_48k": str(mix_audio)},
        segments=[
            Segment(
                id="seg_0001",
                start=0.0,
                end=1.0,
                duration=1.0,
                audio_for_gemma="work/segments/audio/seg_0001_gemma.wav",
                audio_for_mix="work/segments/audio/seg_0001_mix.wav",
            ),
            Segment(
                id="seg_0002",
                start=1.0,
                end=2.0,
                duration=1.0,
                audio_for_gemma="work/segments/audio/seg_0002_gemma.wav",
                audio_for_mix="work/segments/audio/seg_0002_mix.wav",
            ),
        ],
    )
    save_manifest(project, manifest)

    class FakeASRBackend:
        name = "faster_whisper"

        def transcribe(self, audio_path: Path, segments: list[object]) -> list[ASRChunk]:
            _ = audio_path, segments
            return [ASRChunk(start=0.1, end=0.7, text="声です", language="ja", confidence=0.9)]

    monkeypatch.setattr(pipeline_steps, "create_asr_backend", lambda *_args, **_kwargs: FakeASRBackend())

    manifest = transcribe_step(project, asr_backend="faster_whisper", confirm_rights=True)

    assert manifest.segments[0].status == "transcribed"
    assert manifest.segments[1].status == "no_speech_detected"
    assert manifest.segments[1].source_script is None
    assert manifest.segments[1].analysis["asr_quality_gate"] == {
        "decision": "no_speech",
        "reasons": ["asr_skipped_folder_part:effect_only"],
        "tts_blocked": True,
    }


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


def test_run_pipeline_uses_separated_background_without_timeline_suppression(
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
    assert mix_manifest["background"]["path"] == manifest.artifacts["background_only_48k"]
    assert mix_manifest["background"]["speech_suppression"]["enabled"] is False
    assert mix_manifest["background"]["speech_suppression"]["center_bleed_reduction"] is False
    assert "source_suppressed_background" not in manifest.artifacts
