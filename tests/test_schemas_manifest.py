from __future__ import annotations

import json

import pytest

from asmr_dub_pipeline.pipeline.manifest_io import load_manifest, save_manifest
from asmr_dub_pipeline.schemas import PipelineManifest, ProjectConfig, RVCMetadata, Segment


def test_segment_validates_duration() -> None:
    Segment(
        id="seg_0001",
        start=0,
        end=1,
        duration=1,
        audio_for_gemma="a.wav",
        audio_for_mix="b.wav",
    )
    with pytest.raises(ValueError):
        Segment(
            id="bad",
            start=1,
            end=0.5,
            duration=1,
            audio_for_gemma="a.wav",
            audio_for_mix="b.wav",
        )


def test_manifest_atomic_round_trip(tmp_project_dir) -> None:
    manifest = PipelineManifest()
    manifest.segments.append(
        Segment(
            id="seg_0001",
            start=0,
            end=1,
            duration=1,
            audio_for_gemma="a.wav",
            audio_for_mix="b.wav",
        )
    )
    path = save_manifest(tmp_project_dir, manifest)
    assert path.exists()
    text = path.read_text("utf-8")
    assert text.startswith("{\n")
    data = json.loads(text)
    assert data["schema_version"] == "1.0"
    loaded = load_manifest(tmp_project_dir)
    assert loaded.segments[0].id == "seg_0001"


def test_project_config_requires_rvc_by_default() -> None:
    cfg = ProjectConfig()

    assert cfg.rvc_required is True
    assert cfg.rvc_backend == "command"
    assert cfg.rvc_train_required is True
    assert cfg.rvc_train_backend == "command"
    assert cfg.rvc_train_timeout_sec == 14400.0
    assert cfg.rvc_train_batch_size == 0
    assert cfg.rvc_train_preprocess_processes == 0
    assert cfg.rvc_train_f0_workers == 0
    assert cfg.rvc_train_feature_workers == 0
    assert cfg.rvc_train_save_every_epoch == 50
    assert cfg.rvc_train_reuse_intermediate_cache is True
    assert cfg.rvc_concurrency == 1
    assert cfg.source_separation_backend == "demucs"
    assert cfg.source_separation_model == "htdemucs"
    assert cfg.qwen_tts_candidate_batch_size == 4
    assert cfg.qwen_tts_max_new_tokens == 2048
    assert cfg.qwen_tts_segment_batch_size == 8
    assert cfg.qwen_tts_target_vram_gb == 14.0
    assert [profile.name for profile in cfg.rvc_auto_profiles] == [
        "rmvpe_index045",
        "rmvpe_index035_safer",
        "rmvpe_index055_stronger_timbre",
        "crepe_index045_whisper_candidate",
    ]


def test_rvc_metadata_manifest_round_trip(tmp_project_dir) -> None:
    manifest = PipelineManifest(
        segments=[
            Segment(
                id="seg_0001",
                start=0,
                end=1,
                duration=1,
                audio_for_gemma="a.wav",
                audio_for_mix="b.wav",
                rvc=RVCMetadata(
                    backend="mock",
                    input_path="work/tts/seg_0001_final.wav",
                    output_path="work/rvc/seg_0001_final.wav",
                    selected_profile_name="rmvpe_index045",
                    candidate_paths=["work/rvc/candidates/seg_0001/rmvpe_index045.wav"],
                    accepted=True,
                ),
            )
        ]
    )

    save_manifest(tmp_project_dir, manifest)
    loaded = load_manifest(tmp_project_dir)

    assert loaded.segments[0].rvc is not None
    assert loaded.segments[0].rvc.accepted is True
    assert loaded.segments[0].rvc.input_path == "work/tts/seg_0001_final.wav"
