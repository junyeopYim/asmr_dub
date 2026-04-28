from __future__ import annotations

import json
from pathlib import Path

from conftest import sha256

import asmr_dub_pipeline.cli as cli_module
from asmr_dub_pipeline.cli import app
from asmr_dub_pipeline.config import load_project_config
from asmr_dub_pipeline.pipeline.manifest_io import load_manifest
from asmr_dub_pipeline.pipeline.steps import extract_step, mix_step, segment_step
from asmr_dub_pipeline.schemas import PipelineManifest


def test_mock_pipeline_e2e(cli_runner, tiny_wav_path: Path, tmp_project_dir: Path) -> None:
    before = sha256(tiny_wav_path)
    result = cli_runner.invoke(
        app,
        [
            "run",
            str(tiny_wav_path),
            "--project",
            str(tmp_project_dir),
            "--confirm-rights",
            "--mock",
        ],
    )
    assert result.exit_code == 0, result.output
    assert sha256(tiny_wav_path) == before
    assert (tmp_project_dir / "work/audio/original_stereo_48k.wav").exists()
    assert (tmp_project_dir / "work/audio/gemma_mono_16k.wav").exists()
    assert (tmp_project_dir / "work/segments/manifests/segments_raw.json").exists()
    assert (tmp_project_dir / "work/mix/dialogue_stem.wav").exists()
    assert (tmp_project_dir / "work/mix/final_audio.wav").exists()
    manifest = load_manifest(tmp_project_dir)
    assert manifest.rights_audit.confirmed is True
    assert manifest.segments
    assert manifest.segments[0].tts is not None
    assert manifest.segments[0].tts.selected_candidate_path
    assert manifest.artifacts["export"].endswith("_dub.wav")
    assert manifest.artifacts["mix_manifest"].endswith("mix_manifest.json")
    assert manifest.artifacts["export_manifest"].endswith("export_manifest.json")
    assert manifest.segments[0].mix["included"] is True
    assert manifest.segments[0].mix["dialogue_fade_ms"] is None
    assert manifest.segments[0].mix["fade_in_ms"] > 8.0
    mix_manifest = json.loads(Path(manifest.artifacts["mix_manifest"]).read_text("utf-8"))
    assert mix_manifest["config"]["background_bed"] == "preserve_original"
    assert mix_manifest["config"]["loudness_strategy"] == "peak_guard_only"
    assert mix_manifest["config"]["loudness_normalization"] == "disabled"
    assert mix_manifest["background"]["used"] is True
    assert Path(manifest.artifacts["export"]).exists()


def test_full_command_runs_mock_e2e_with_project(cli_runner, tiny_wav_path: Path, tmp_path: Path) -> None:
    project = tmp_path / "full_project"
    result = cli_runner.invoke(
        app,
        [
            "full",
            str(tiny_wav_path),
            "--project",
            str(project),
            "--confirm-rights",
            "--no-cache-status",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "extract started" in result.output
    assert "translate-ko: 1/" in result.output
    assert "korean-script: 1/" in result.output
    assert "synth: 1/" in result.output
    assert "mix dialogue: 1/" in result.output
    assert "Pipeline complete" in result.output
    assert "Project:" in result.output
    assert str(project.resolve()) in result.output
    manifest = load_manifest(project)
    assert manifest.rights_audit.confirmed is True
    assert manifest.artifacts["export"].endswith("_dub.wav")
    assert Path(manifest.artifacts["export"]).exists()


def test_full_real_applies_high_quality_preset_by_default(
    cli_runner,
    tiny_wav_path: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "full_real_hq"
    captured: dict[str, object] = {}

    def fake_run_pipeline(*args: object, **kwargs: object) -> PipelineManifest:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return PipelineManifest(artifacts={"export": "out.wav"})

    monkeypatch.setattr(cli_module, "run_pipeline", fake_run_pipeline)

    result = cli_runner.invoke(
        app,
        [
            "full",
            str(tiny_wav_path),
            "--project",
            str(project),
            "--confirm-rights",
            "--real",
            "--no-cache-status",
        ],
    )

    assert result.exit_code == 0, result.output
    cfg = load_project_config(project)
    assert cfg.target_language == "ko"
    assert cfg.candidate_count == 8
    assert cfg.duration_tolerance == 0.15
    assert cfg.gsv_few_shot_target_sec == 180.0
    assert cfg.gsv_few_shot_min_clip_sec == 2.0
    assert cfg.gsv_few_shot_max_clip_sec == 8.0
    assert cfg.gsv_concurrency == 1
    assert cfg.gemma_llama_cpp_ctx_size == 16384
    assert cfg.gemma_text_batch_size == 1
    assert cfg.gemma_text_concurrency == 4
    assert cfg.gemma_text_n_predict == 8192
    assert cfg.mix_allow_korean_timing_draft is False
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["mock"] is False
    assert kwargs["gemma_backend"] == "llama_cpp"
    assert kwargs["few_shot"] is True
    assert kwargs["gsv_few_shot_force"] is True


def test_full_real_use_trained_gpt_flag_passes_to_pipeline(
    cli_runner,
    tiny_wav_path: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "full_real_trained_gpt"
    captured: dict[str, object] = {}

    def fake_run_pipeline(*args: object, **kwargs: object) -> PipelineManifest:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return PipelineManifest(artifacts={"export": "out.wav"})

    monkeypatch.setattr(cli_module, "run_pipeline", fake_run_pipeline)

    result = cli_runner.invoke(
        app,
        [
            "full",
            str(tiny_wav_path),
            "--project",
            str(project),
            "--confirm-rights",
            "--real",
            "--use-trained-gpt",
            "--no-cache-status",
        ],
    )

    assert result.exit_code == 0, result.output
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["use_trained_gpt"] is True


def test_mix_requires_completed_qc(tiny_wav_path: Path, tmp_project_dir: Path) -> None:
    extract_step(tiny_wav_path, tmp_project_dir, confirm_rights=True)
    segment_step(tmp_project_dir)
    try:
        mix_step(tmp_project_dir, confirm_rights=True)
    except ValueError as exc:
        assert "QC" in str(exc)
    else:
        raise AssertionError("Expected mix to require completed QC")
