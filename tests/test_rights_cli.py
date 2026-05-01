from __future__ import annotations

import pytest
from conftest import sha256

from asmr_dub_pipeline.cli import app
from asmr_dub_pipeline.config import save_project_config
from asmr_dub_pipeline.pipeline.manifest_io import load_manifest, save_manifest
from asmr_dub_pipeline.pipeline.state import mark_stage
from asmr_dub_pipeline.pipeline.steps import (
    analyze_step,
    init_project,
    qc_step,
    script_step,
    segment_step,
    synth_step,
)
from asmr_dub_pipeline.rights import RightsError
from asmr_dub_pipeline.schemas import PipelineManifest, ProjectConfig, Segment, TTSMetadata


def _write_valid_rvc_project(tmp_project_dir, tiny_wav_path) -> None:
    model = tmp_project_dir / "models" / "voice.pth"
    model.parent.mkdir(parents=True, exist_ok=True)
    model.write_bytes(b"model")
    save_project_config(
        ProjectConfig(
            project_name=tmp_project_dir.name,
            rvc_backend="command",
            rvc_command=["rvc", "{input}", "{output}", "{model}"],
            rvc_model_path=str(model),
        ),
        tmp_project_dir / "pipeline.yaml",
    )
    tts = tmp_project_dir / "work" / "tts" / "seg_0001_final.wav"
    tts.parent.mkdir(parents=True, exist_ok=True)
    tts.write_bytes(tiny_wav_path.read_bytes())
    manifest = PipelineManifest(
        segments=[
            Segment(
                id="seg_0001",
                start=0,
                end=1,
                duration=1,
                audio_for_gemma="a.wav",
                audio_for_mix="b.wav",
                status="synthesized",
                tts=TTSMetadata(selected_candidate_path=str(tts)),
            )
        ]
    )
    mark_stage(manifest, "synth", "completed")
    index = tmp_project_dir / "models" / "voice.index"
    index.write_bytes(b"index")
    manifest.artifacts["rvc_model_path"] = str(model)
    manifest.artifacts["rvc_index_path"] = str(index)
    mark_stage(manifest, "train-rvc", "completed", model_path=str(model), index_path=str(index))
    save_manifest(tmp_project_dir, manifest)


def test_run_requires_confirm_rights(cli_runner, tiny_wav_path, tmp_project_dir) -> None:
    result = cli_runner.invoke(app, ["run", str(tiny_wav_path), "--project", str(tmp_project_dir), "--mock"])
    assert result.exit_code != 0
    assert "permission/consent" in result.output


def test_full_requires_confirm_rights(cli_runner, tiny_wav_path, tmp_project_dir) -> None:
    result = cli_runner.invoke(
        app,
        ["full", str(tiny_wav_path), "--project", str(tmp_project_dir), "--no-cache-status"],
    )
    assert result.exit_code != 0
    assert "permission/consent" in result.output


def test_extract_records_rights_audit(cli_runner, tiny_wav_path, tmp_project_dir) -> None:
    before = sha256(tiny_wav_path)
    result = cli_runner.invoke(
        app,
        ["extract", str(tiny_wav_path), "--project", str(tmp_project_dir), "--confirm-rights"],
    )
    assert result.exit_code == 0, result.output
    assert sha256(tiny_wav_path) == before
    manifest = load_manifest(tmp_project_dir)
    assert manifest.rights_audit.confirmed is True
    assert manifest.rights_audit.source_sha256 == before


def test_other_real_media_commands_require_rights(cli_runner, tiny_wav_path, tmp_project_dir) -> None:
    for args in (
        ["mix", "--project", str(tmp_project_dir)],
        ["export", str(tiny_wav_path), "--project", str(tmp_project_dir)],
        ["synth", "--project", str(tmp_project_dir)],
        ["train-gsv", "--project", str(tmp_project_dir)],
    ):
        result = cli_runner.invoke(app, args)
        assert result.exit_code != 0
        assert "rights" in result.output.lower() or "permission" in result.output.lower()


def test_derived_media_stages_require_existing_rights_audit(tmp_project_dir) -> None:
    init_project(tmp_project_dir)
    for call in (
        lambda: segment_step(tmp_project_dir),
        lambda: analyze_step(tmp_project_dir, "mock"),
        lambda: script_step(tmp_project_dir, "mock"),
        lambda: qc_step(tmp_project_dir, "mock"),
    ):
        with pytest.raises(RightsError):
            call()


def test_real_synth_requires_fresh_confirm_rights(tmp_project_dir, tiny_wav_path) -> None:
    from asmr_dub_pipeline.pipeline.steps import extract_step

    extract_step(tiny_wav_path, tmp_project_dir, confirm_rights=True)
    with pytest.raises(RightsError, match="GPT-SoVITS"):
        synth_step(tmp_project_dir, None, refs_path=tmp_project_dir / "refs/refs.json", mock=False)


def test_rvc_cli_requires_confirm_rights(cli_runner, tiny_wav_path, tmp_project_dir) -> None:
    _write_valid_rvc_project(tmp_project_dir, tiny_wav_path)
    result = cli_runner.invoke(app, ["rvc", "--project", str(tmp_project_dir)])
    assert result.exit_code != 0
    assert "rights" in result.output.lower() or "permission" in result.output.lower()


def test_train_rvc_cli_requires_confirm_rights(cli_runner, tiny_wav_path, tmp_project_dir) -> None:
    _write_valid_rvc_project(tmp_project_dir, tiny_wav_path)
    manifest = load_manifest(tmp_project_dir)
    manifest.stage_state.pop("train-rvc", None)
    save_manifest(tmp_project_dir, manifest)
    result = cli_runner.invoke(app, ["train-rvc", "--project", str(tmp_project_dir)])
    assert result.exit_code != 0
    assert "rights" in result.output.lower() or "permission" in result.output.lower()


def test_manual_segment_path_traversal_rejected(tmp_project_dir) -> None:
    init_project(tmp_project_dir)
    manual = tmp_project_dir / "work/segments/manifests/segments_manual.json"
    manual.write_text(
        """
{
  "segments": [
    {
      "id": "seg_0001",
      "start": 0,
      "end": 1,
      "duration": 1,
      "audio_for_gemma": "../outside.wav",
      "audio_for_mix": "work/segments/audio/seg_0001_mix.wav",
      "estimated_pan": 0,
      "keep_original_texture": true,
      "status": "raw"
    }
  ]
}
""",
        "utf-8",
    )
    try:
        segment_step(tmp_project_dir, confirm_rights=True)
    except Exception as exc:
        assert "inside the project" in str(exc)
    else:
        raise AssertionError("Expected path traversal to be rejected")
