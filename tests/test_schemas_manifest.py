from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from asmr_dub_pipeline.audio.features import write_audio
from asmr_dub_pipeline.config import load_project_config, save_project_config
from asmr_dub_pipeline.pipeline.manifest_io import load_manifest, save_manifest
from asmr_dub_pipeline.pipeline.steps import synth_step
from asmr_dub_pipeline.schemas import (
    JapaneseScript,
    PipelineManifest,
    ProjectConfig,
    RVCMetadata,
    Segment,
    SourceScript,
)


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
    assert cfg.rvc_train_sample_rate == 48_000
    assert cfg.rvc_train_epochs == 100
    assert cfg.rvc_train_batch_size == 0
    assert cfg.rvc_train_min_clean_sec == 600.0
    assert cfg.rvc_train_augment_enabled is False
    assert cfg.rvc_train_augment_min_real_sec == 300.0
    assert cfg.rvc_train_preprocess_processes == 0
    assert cfg.rvc_train_f0_workers == 0
    assert cfg.rvc_train_feature_workers == 0
    assert cfg.rvc_train_save_every_epoch == 10
    assert cfg.rvc_train_reuse_intermediate_cache is True
    assert cfg.rvc_concurrency == 1
    assert cfg.source_separation_backend == "auto"
    assert cfg.source_separation_model == "htdemucs"
    assert cfg.qwen_tts_candidate_batch_size == 4
    assert cfg.qwen_tts_max_new_tokens == 2048
    assert cfg.qwen_tts_segment_batch_size == 8
    assert cfg.qwen_tts_target_vram_gb == 14.0
    assert cfg.fish_tts_repo_dir == ".cache/tts_backends/fish-speech"
    assert cfg.fish_tts_base_url == "http://127.0.0.1:8080"
    assert cfg.cosyvoice_repo_dir == ".cache/tts_backends/CosyVoice"
    assert cfg.cosyvoice_model_dir.endswith("CosyVoice2-0.5B")
    assert cfg.cosyvoice_base_url == "http://127.0.0.1:50000"
    assert [profile.name for profile in cfg.rvc_auto_profiles] == [
        "rmvpe_index045",
        "rmvpe_index035_safer",
        "rmvpe_index055_stronger_timbre",
        "crepe_index045_whisper_candidate",
    ]


def test_load_project_config_rejects_legacy_asr_text_review_keys(tmp_path) -> None:
    project = tmp_path / "legacy_config"
    project.mkdir()
    (project / "pipeline.yaml").write_text(
        "\n".join(
            [
                "project_name: legacy_config",
                "asr_text_review_enabled: true",
                "asr_text_review_backend: llama_server",
                "asr_text_review_max_chunks: 7",
                "asr_text_review_suspicious_text_patterns:",
                "- 釣り",
            ]
        )
        + "\n",
        "utf-8",
    )

    with pytest.raises(ValueError):
        load_project_config(project)


def test_manifest_rejects_legacy_asr_text_review_project_config(tmp_project_dir) -> None:
    manifest = PipelineManifest(project_config=ProjectConfig(project_name=tmp_project_dir.name))
    payload = manifest.model_dump(mode="json")
    project_config = payload["project_config"]
    project_config["asr_text_review_enabled"] = True
    project_config["asr_text_review_backend"] = "llama_server"
    project_config["asr_text_review_max_chunks"] = 11
    project_config["asr"].pop("review_enabled")
    project_config["asr"].pop("review_backend")
    project_config["asr"].pop("review_max_chunks")
    manifest_path = tmp_project_dir / "work" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", "utf-8")

    with pytest.raises(ValueError):
        load_manifest(tmp_project_dir)


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


def test_numeric_phrase_renderer_metadata_is_json_stable() -> None:
    segment = Segment(
        id="seg_numeric",
        start=0,
        end=3,
        duration=3,
        audio_for_gemma="work/segments/seg_numeric_gemma.wav",
        audio_for_mix="work/segments/seg_numeric_mix.wav",
    )
    metadata = {
        "status": "rendered",
        "renderer": "numeric_phrase",
        "render_policy": "whole_span_guard120_pad350",
        "values": [1, 2, 3, 4],
        "tokens": ["하나", "둘", "셋", "넷"],
        "text_variant": "native_countup",
        "text": "하나, 둘, 셋, 넷.",
        "candidate_text": "하나, 둘, 셋, 넷.",
        "candidate_generation": {
            "text_split_method": "cut0",
            "top_k": 5,
            "top_p": 0.85,
            "temperature": 0.65,
            "repetition_penalty": 1.8,
        },
        "max_tempo": 1.0,
        "numeric_qc": {
            "gate": "pass",
            "expected_values": [1, 2, 3, 4],
            "observed_values": [1, 2, 3, 4],
            "backend": "mock",
        },
        "placements": [
            {
                "values": [1, 2, 3, 4],
                "source_start_sec": 0.08,
                "source_end_sec": 2.72,
                "target_start_sec": 0.12,
                "target_end_sec": 2.84,
                "required_tempo": 1.0,
                "copy_status": "copied",
            }
        ],
        "output_path": "work/tts/numeric_phrase/seg_numeric_numeric_phrase.wav",
    }
    segment.analysis["numeric_phrase_renderer"] = metadata

    dumped = segment.model_dump(mode="json")
    renderer = dumped["analysis"]["numeric_phrase_renderer"]

    assert renderer == metadata
    assert json.loads(json.dumps(renderer, ensure_ascii=False, sort_keys=True)) == metadata


def test_numeric_phrase_renderer_success_metadata_is_manifest_ready(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import asmr_dub_pipeline.pipeline.stages.synth_gpt_sovits as synth_stage

    def fake_render_numeric_phrase_segment(**kwargs: Any) -> dict[str, Any]:
        segment = kwargs["segment"]
        output_path = (
            kwargs["project_dir"]
            / "work"
            / "tts"
            / "numeric_phrase"
            / f"{segment.id}_fake.wav"
        )
        samples = np.full((int(round(segment.duration * 48_000)), 2), 0.01, dtype=np.float32)
        write_audio(output_path, samples, 48_000)
        return {
            "status": "rendered",
            "output_path": str(output_path),
            "duration_sec": segment.duration,
            "seed": 4242,
            "payload": {"mock": True},
            "selection_reason": "test_numeric_phrase_renderer",
        }

    monkeypatch.setattr(
        synth_stage,
        "render_numeric_phrase_segment",
        fake_render_numeric_phrase_segment,
    )
    cfg = ProjectConfig(project_name="numeric-metadata")
    save_project_config(cfg, tmp_path / "pipeline.yaml")
    ref_audio = tmp_path / "refs" / "ref.wav"
    write_audio(ref_audio, np.full((24_000, 2), 0.01, dtype=np.float32), 24_000)
    refs_path = tmp_path / "refs" / "refs.json"
    refs_path.write_text(
        json.dumps(
            {
                "whisper_close": {
                    "ref_audio_path": "refs/ref.wav",
                    "prompt_text": "テストです",
                    "prompt_lang": "ja",
                }
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        "utf-8",
    )
    segment = Segment(
        id="seg_numeric",
        start=0,
        end=3,
        duration=3,
        status="scripted",
        audio_for_gemma="work/segments/seg_numeric_gemma.wav",
        audio_for_mix="work/segments/seg_numeric_mix.wav",
        source_script=SourceScript(
            text="1 2 3 4",
            language="ja",
            backend="mock",
            start=0,
            end=3,
        ),
        script=JapaneseScript(
            literal_ja="1 2 3 4",
            ja_text="1 2 3 4",
            tts_text="하나, 둘, 셋, 넷.",
            tts_language="ko",
            source_language="ja",
            target_language="ko",
            expected_tts_duration_sec=3,
            ref_style="whisper_close",
        ),
    )
    manifest = PipelineManifest(project_config=cfg, segments=[segment])
    manifest.stage_state["korean-script"] = {"status": "completed"}
    save_manifest(tmp_path, manifest)

    synth_step(tmp_path, None, refs_path, mock=True, confirm_rights=True)

    loaded = load_manifest(tmp_path)
    renderer = loaded.segments[0].analysis["numeric_phrase_renderer"]
    assert renderer["status"] == "rendered"
    assert renderer["candidate_text"] == "하나 둘 셋 넷."
    assert renderer["candidate_generation"] == {
        "text_split_method": "cut0",
        "top_k": 5,
        "top_p": 0.85,
        "temperature": 0.65,
        "repetition_penalty": 1.8,
    }
    assert renderer["numeric_qc"] == {
        "gate": "pass",
        "expected_values": [1, 2, 3, 4],
        "observed_values": [1, 2, 3, 4],
        "backend": "mock",
        "reason": "mock_numeric_phrase_renderer",
    }
    assert renderer["placements"] == []
    assert renderer["output_path"].endswith("seg_numeric_fake.wav")
    assert json.loads(json.dumps(renderer, ensure_ascii=False, sort_keys=True)) == renderer
