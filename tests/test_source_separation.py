from __future__ import annotations

import json
from pathlib import Path

from asmr_dub_pipeline.cli import app
from asmr_dub_pipeline.config import create_project_structure, save_project_config
from asmr_dub_pipeline.pipeline.manifest_io import load_manifest
from asmr_dub_pipeline.pipeline.steps import extract_step, segment_step, source_separation_step
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
