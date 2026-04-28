from __future__ import annotations

import json
from pathlib import Path

from conftest import sha256

from asmr_dub_pipeline.cli import app
from asmr_dub_pipeline.pipeline.manifest_io import load_manifest
from asmr_dub_pipeline.pipeline.steps import extract_step, mix_step, segment_step


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


def test_mix_requires_completed_qc(tiny_wav_path: Path, tmp_project_dir: Path) -> None:
    extract_step(tiny_wav_path, tmp_project_dir, confirm_rights=True)
    segment_step(tmp_project_dir)
    try:
        mix_step(tmp_project_dir, confirm_rights=True)
    except ValueError as exc:
        assert "QC" in str(exc)
    else:
        raise AssertionError("Expected mix to require completed QC")
