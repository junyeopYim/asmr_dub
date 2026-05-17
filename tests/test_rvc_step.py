from __future__ import annotations

import json
import os
import time
from pathlib import Path
from threading import Lock

import numpy as np
import pytest
import soundfile as sf

import asmr_dub_pipeline.cli as cli_module
from asmr_dub_pipeline.cli import app
from asmr_dub_pipeline.config import load_project_config, save_project_config
from asmr_dub_pipeline.pipeline import steps as pipeline_steps
from asmr_dub_pipeline.pipeline.manifest_io import load_manifest, save_manifest
from asmr_dub_pipeline.pipeline.state import mark_stage
from asmr_dub_pipeline.rights import RightsError, require_confirmed_rights
from asmr_dub_pipeline.rvc import RVCCommandError, RVCCommandResult, validate_rvc_config
from asmr_dub_pipeline.schemas import (
    JapaneseScript,
    PipelineManifest,
    ProjectConfig,
    QCMetadata,
    RVCMetadata,
    RVCSpeakerConfig,
    Segment,
    SourceScript,
    TTSMetadata,
)


def _write_exact_wav(path: Path, duration: float, sample_rate: int = 48_000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(max(1, int(sample_rate * duration)), dtype=np.float32) / sample_rate
    tone = 0.05 * np.sin(2 * np.pi * 220.0 * t)
    sf.write(str(path), np.stack([tone, tone], axis=1), sample_rate)
    return path


def _segment_with_tts(project_dir: Path, duration: float = 1.0, segment_id: str = "seg_0001") -> Segment:
    tts_path = _write_exact_wav(project_dir / "work" / "tts" / f"{segment_id}_final.wav", duration=duration)
    source_path = _write_exact_wav(
        project_dir / "work" / "segments" / "audio" / f"{segment_id}_mix.wav",
        duration=duration,
    )
    return Segment(
        id=segment_id,
        start=0.0,
        end=duration,
        duration=duration,
        audio_for_gemma="work/segments/audio/seg_0001_gemma.wav",
        audio_for_mix=str(source_path.relative_to(project_dir)),
        speaker_id="speaker_0001",
        analysis={"speaker_count": 1},
        source_script=SourceScript(
            text="こんにちは。",
            language="ja",
            backend="mock",
            start=0.0,
            end=duration,
        ),
        status="synthesized",
        script=JapaneseScript(ja_text="안녕하세요.", tts_text="안녕하세요.", tts_language="ko"),
        tts=TTSMetadata(selected_candidate_path=str(tts_path)),
    )


def _save_synth_manifest(
    project_dir: Path,
    cfg: ProjectConfig,
    segment: Segment | None = None,
    segments: list[Segment] | None = None,
) -> None:
    save_project_config(cfg.model_copy(update={"project_name": project_dir.name}), project_dir / "pipeline.yaml")
    manifest = PipelineManifest(
        project_config=cfg,
        rights_audit=require_confirmed_rights(True, "test"),
        segments=segments or [segment or _segment_with_tts(project_dir)],
    )
    mark_stage(manifest, "synth", "completed")
    model_path, index_path = project_dir / "work" / "rvc_train" / "model" / "test.pth", project_dir / "work" / "rvc_train" / "model" / "test.index"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(b"model")
    index_path.write_bytes(b"index")
    manifest.artifacts["rvc_model_path"] = str(model_path)
    manifest.artifacts["rvc_index_path"] = str(index_path)
    mark_stage(manifest, "train-rvc", "completed", model_path=str(model_path), index_path=str(index_path))
    save_manifest(project_dir, manifest)


def test_mock_rvc_copies_selected_tts_and_records_manifest(tmp_project_dir: Path) -> None:
    cfg = ProjectConfig(rvc_backend="mock")
    _save_synth_manifest(tmp_project_dir, cfg)

    manifest = pipeline_steps.rvc_step(tmp_project_dir, confirm_rights=True)
    segment = manifest.segments[0]

    assert manifest.stage_state["rvc"]["status"] == "completed"
    assert manifest.artifacts["rvc_manifest"].endswith("work/rvc/rvc_manifest.json")
    assert Path(manifest.artifacts["rvc_manifest"]).exists()
    assert segment.rvc is not None
    assert segment.rvc.input_path.endswith("work/tts/seg_0001_final.wav")
    assert segment.rvc.output_path is not None
    assert segment.rvc.output_path.endswith("work/rvc/seg_0001_final.wav")
    assert segment.rvc.selected_profile_name == "rmvpe_index045"
    assert segment.rvc.candidate_paths
    assert segment.tts is not None
    assert segment.tts.selected_candidate_path is not None
    assert segment.tts.selected_candidate_path.endswith("work/tts/seg_0001_final.wav")
    assert Path(segment.tts.selected_candidate_path).exists()
    assert Path(segment.rvc.output_path).exists()


def test_rvc_rerun_uses_raw_tts_not_previous_rvc_output(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = ProjectConfig(rvc_backend="mock")
    _save_synth_manifest(tmp_project_dir, cfg)
    first = pipeline_steps.rvc_step(tmp_project_dir, confirm_rights=True)
    raw_tts = first.segments[0].tts.selected_candidate_path
    assert raw_tts is not None
    seen_inputs: list[Path] = []
    original_convert = pipeline_steps.RVCMockClient.convert

    def counting_convert(self, input_path: Path, output_path: Path, **kwargs: object):
        seen_inputs.append(input_path)
        return original_convert(self, input_path, output_path, **kwargs)

    monkeypatch.setattr(pipeline_steps.RVCMockClient, "convert", counting_convert)
    pipeline_steps.rvc_step(tmp_project_dir, confirm_rights=True, force=True)

    assert seen_inputs
    assert seen_inputs[0] == Path(raw_tts)
    assert "work/rvc" not in str(seen_inputs[0])


def test_mock_train_rvc_creates_model_artifacts(tmp_project_dir: Path) -> None:
    cfg = ProjectConfig(rvc_backend="mock", rvc_train_backend="mock", rvc_train_min_clean_sec=0.0)
    save_project_config(cfg.model_copy(update={"project_name": tmp_project_dir.name}), tmp_project_dir / "pipeline.yaml")
    manifest = PipelineManifest(
        project_config=cfg,
        rights_audit=require_confirmed_rights(True, "test"),
        segments=[_segment_with_tts(tmp_project_dir)],
    )
    mark_stage(manifest, "synth", "completed")
    save_manifest(tmp_project_dir, manifest)

    trained = pipeline_steps.rvc_train_step(tmp_project_dir, confirm_rights=True, mock=True)

    assert trained.stage_state["train-rvc"]["status"] == "completed"
    assert Path(trained.artifacts["rvc_model_path"]).exists()
    assert Path(trained.artifacts["rvc_index_path"]).exists()
    assert Path(trained.artifacts["rvc_train_manifest"]).exists()


def test_train_rvc_auto_epoch_records_effective_epochs(tmp_project_dir: Path) -> None:
    cfg = ProjectConfig(
        rvc_backend="mock",
        rvc_train_backend="mock",
        rvc_train_min_clean_sec=0.0,
        rvc_train_epoch_policy="auto",
        rvc_train_auto_epoch_min=80,
        rvc_train_auto_epoch_max=120,
    )
    save_project_config(cfg.model_copy(update={"project_name": tmp_project_dir.name}), tmp_project_dir / "pipeline.yaml")
    manifest = PipelineManifest(
        project_config=cfg,
        rights_audit=require_confirmed_rights(True, "test"),
        segments=[_segment_with_tts(tmp_project_dir)],
    )
    mark_stage(manifest, "synth", "completed")
    save_manifest(tmp_project_dir, manifest)

    trained = pipeline_steps.rvc_train_step(tmp_project_dir, confirm_rights=True, mock=True)

    train_payload = json.loads(Path(trained.artifacts["rvc_train_manifest"]).read_text("utf-8"))
    assert train_payload["epoch_decision"]["policy"] == "auto"
    assert train_payload["epoch_decision"]["configured_epochs"] == 100
    assert train_payload["epoch_decision"]["recommended_epoch_count"] == 25
    assert train_payload["epoch_decision"]["effective_epochs"] == 80
    assert train_payload["effective_train_epochs"] == 80
    assert trained.stage_state["train-rvc"]["effective_train_epochs"] == 80


def test_voice_bank_rvc_skips_training_and_uses_speaker_model(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_project_dir / "voice_bank" / "speakers" / "speaker_0001" / "rvc" / "v001" / "model.pth"
    index_path = tmp_project_dir / "voice_bank" / "speakers" / "speaker_0001" / "rvc" / "v001" / "added.index"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(b"model")
    index_path.write_bytes(b"index")
    cfg = ProjectConfig(
        rvc_backend="command",
        rvc_command=["rvc", "{input}", "{output}", "{model}", "{index}", "{sid}"],
        rvc_train_required=False,
        rvc_speaker_models={
            "speaker_0001": RVCSpeakerConfig(
                model_path=str(model_path.relative_to(tmp_project_dir)),
                index_path=str(index_path.relative_to(tmp_project_dir)),
            )
        },
    )
    save_project_config(cfg.model_copy(update={"project_name": tmp_project_dir.name}), tmp_project_dir / "pipeline.yaml")
    segment = _segment_with_tts(tmp_project_dir)
    segment.speaker_id = "speaker_0001"
    manifest = PipelineManifest(
        project_config=cfg,
        rights_audit=require_confirmed_rights(True, "test"),
        segments=[segment],
    )
    mark_stage(manifest, "synth", "completed")
    save_manifest(tmp_project_dir, manifest)
    skipped = pipeline_steps.skip_rvc_train_for_voice_bank_step(tmp_project_dir)
    seen: list[tuple[Path | None, Path | None, str]] = []

    def fake_convert(self, input_path: Path, output_path: Path, **kwargs: object) -> RVCCommandResult:
        seen.append((kwargs["model_path"], kwargs["index_path"], kwargs["sid"]))
        _write_exact_wav(output_path, duration=1.0)
        return RVCCommandResult(output_path, ["fake"], "", "", 0, 0.0)

    monkeypatch.setattr(pipeline_steps.RVCCommandClient, "convert", fake_convert)
    converted = pipeline_steps.rvc_step(tmp_project_dir, confirm_rights=True)

    assert skipped.stage_state["train-rvc"]["status"] == "skipped_pretrained_voice_bank"
    assert seen == [(model_path, index_path, "speaker_0001")]
    assert converted.segments[0].rvc is not None
    assert converted.segments[0].rvc.model_path == str(model_path)


def test_rvc_train_dataset_skips_identical_existing_copy(tmp_project_dir: Path, monkeypatch) -> None:
    cfg = ProjectConfig(rvc_backend="mock", rvc_train_backend="mock", rvc_train_min_clean_sec=0.0)
    manifest = PipelineManifest(
        project_config=cfg,
        rights_audit=require_confirmed_rights(True, "test"),
        segments=[_segment_with_tts(tmp_project_dir)],
    )
    original_copy2 = pipeline_steps.shutil.copy2
    calls: list[tuple[Path, Path]] = []

    def counting_copy2(source: Path, output: Path) -> Path:
        calls.append((source, output))
        return original_copy2(source, output)

    monkeypatch.setattr(pipeline_steps.shutil, "copy2", counting_copy2)

    pipeline_steps._rvc_train_dataset(tmp_project_dir, manifest, force=False)
    pipeline_steps._rvc_train_dataset(tmp_project_dir, manifest, force=False)

    assert len(calls) == 1


def test_rvc_train_dataset_keeps_metadata_out_of_audio_directory(tmp_project_dir: Path) -> None:
    cfg = ProjectConfig(rvc_backend="mock", rvc_train_backend="mock", rvc_train_min_clean_sec=0.0)
    manifest = PipelineManifest(
        project_config=cfg,
        rights_audit=require_confirmed_rights(True, "test"),
        segments=[_segment_with_tts(tmp_project_dir)],
    )

    dataset_dir, rows = pipeline_steps._rvc_train_dataset(tmp_project_dir, manifest, force=False)

    assert rows
    assert {path.suffix for path in dataset_dir.iterdir()} == {".wav"}
    assert not (dataset_dir / "dataset_manifest.json").exists()
    assert (tmp_project_dir / "work" / "rvc_train" / "dataset_manifest.json").exists()


def test_real_rvc_train_dataset_filters_effected_and_multi_speaker_segments(tmp_project_dir: Path) -> None:
    clean = _segment_with_tts(tmp_project_dir, duration=1.0, segment_id="seg_0001")
    effected = _segment_with_tts(tmp_project_dir, duration=1.0, segment_id="seg_0002")
    effected.analysis = {"speaker_count": 1, "risk_flags": ["voice_effect"]}
    multi_speaker = _segment_with_tts(tmp_project_dir, duration=1.0, segment_id="seg_0003")
    multi_speaker.analysis = {"speaker_count": 2}
    cfg = ProjectConfig(rvc_train_min_clean_sec=0.0)
    manifest = PipelineManifest(
        project_config=cfg,
        rights_audit=require_confirmed_rights(True, "test"),
        segments=[clean, effected, multi_speaker],
    )

    dataset_dir, rows = pipeline_steps._rvc_train_dataset(tmp_project_dir, manifest, force=False)

    assert [row["segment_id"] for row in rows] == [clean.id]
    assert {path.name for path in dataset_dir.iterdir()} == {f"{clean.id}.wav"}
    dataset_manifest = json.loads((tmp_project_dir / "work" / "rvc_train" / "dataset_manifest.json").read_text("utf-8"))
    rejected = {
        item["segment_id"]: item["reject_reasons"]
        for item in dataset_manifest["rejected_segments"]
    }
    assert rejected[effected.id] == ["disallowed_training_tag:voice_effect"]
    assert rejected[multi_speaker.id] == ["speaker_count_not_one:2"]


def test_real_rvc_train_dataset_requires_none_effect_tag(tmp_project_dir: Path) -> None:
    clean = _segment_with_tts(tmp_project_dir, duration=1.0, segment_id="seg_0001")
    clean.analysis = {
        "speaker_count": 1,
        "voice_training": {
            "clean_voice": True,
            "eligible": True,
            "effect_tags": ["none"],
        },
    }
    effected = _segment_with_tts(tmp_project_dir, duration=1.0, segment_id="seg_0002")
    effected.analysis = {
        "speaker_count": 1,
        "voice_training": {
            "clean_voice": True,
            "eligible": True,
            "effect_tags": ["pitch_shift"],
        },
    }
    cfg = ProjectConfig(rvc_train_min_clean_sec=0.0)
    manifest = PipelineManifest(
        project_config=cfg,
        rights_audit=require_confirmed_rights(True, "test"),
        segments=[clean, effected],
    )

    dataset_dir, rows = pipeline_steps._rvc_train_dataset(tmp_project_dir, manifest, force=False)

    assert [row["segment_id"] for row in rows] == [clean.id]
    assert {path.name for path in dataset_dir.iterdir()} == {f"{clean.id}.wav"}
    dataset_manifest = json.loads((tmp_project_dir / "work" / "rvc_train" / "dataset_manifest.json").read_text("utf-8"))
    rejected = {
        item["segment_id"]: item["reject_reasons"]
        for item in dataset_manifest["rejected_segments"]
    }
    assert rejected[effected.id] == ["voice_training_effect_tag_not_none:pitch_shift"]


def test_real_rvc_train_dataset_rejects_overly_fast_source_speech(tmp_project_dir: Path) -> None:
    fast = _segment_with_tts(tmp_project_dir, duration=3.0, segment_id="seg_0001")
    fast.source_script = SourceScript(
        text="これはとても速く詰め込まれた長い長い長い台詞です",
        language="ja",
        backend="mock",
        start=0.0,
        end=3.0,
    )
    slow = _segment_with_tts(tmp_project_dir, duration=4.0, segment_id="seg_0002")
    slow.source_script = SourceScript(
        text="ゆっくり話すね",
        language="ja",
        backend="mock",
        start=0.0,
        end=4.0,
    )
    cfg = ProjectConfig(
        rvc_train_backend="command",
        rvc_train_min_clean_sec=0.0,
        rvc_train_preferred_chars_per_sec=4.5,
        rvc_train_max_chars_per_sec=5.2,
    )
    manifest = PipelineManifest(
        project_config=cfg,
        rights_audit=require_confirmed_rights(True, "test"),
        segments=[fast, slow],
    )

    dataset_dir, rows = pipeline_steps._rvc_train_dataset(tmp_project_dir, manifest, force=False)

    assert [row["segment_id"] for row in rows] == [slow.id]
    assert {path.name for path in dataset_dir.iterdir()} == {f"{slow.id}.wav"}
    assert rows[0]["source_chars_per_sec"] == pytest.approx(1.75)
    assert rows[0]["source_text"] == "ゆっくりはなすね"
    assert rows[0]["source_text_original"] == "ゆっくり話すね"
    dataset_manifest = json.loads((tmp_project_dir / "work" / "rvc_train" / "dataset_manifest.json").read_text("utf-8"))
    rejected = {
        item["segment_id"]: item
        for item in dataset_manifest["rejected_segments"]
    }
    assert rejected[fast.id]["source_chars_per_sec"] == pytest.approx(8.0)
    assert rejected[fast.id]["source_text"] == "これわとてもはやくつめこまれたながいながいながいぜりふです"
    assert rejected[fast.id]["source_text_original"] == "これはとても速く詰め込まれた長い長い長い台詞です"
    assert rejected[fast.id]["reject_reasons"] == [
        "source_chars_per_sec_above_max:8.000>5.200"
    ]


def test_rvc_train_dataset_augments_borderline_clean_data(tmp_project_dir: Path) -> None:
    clean = _segment_with_tts(tmp_project_dir, duration=1.0, segment_id="seg_0001")
    cfg = ProjectConfig(
        rvc_train_backend="command",
        rvc_train_min_clean_sec=2.4,
        rvc_train_min_clean_segments=3,
        rvc_train_augment_enabled=True,
        rvc_train_augment_min_real_sec=0.5,
        rvc_train_augment_max_multiplier=3,
    )
    manifest = PipelineManifest(
        project_config=cfg,
        rights_audit=require_confirmed_rights(True, "test"),
        segments=[clean],
    )

    dataset_dir, rows = pipeline_steps._rvc_train_dataset(tmp_project_dir, manifest, force=False)

    assert len(rows) == 3
    assert {path.name for path in dataset_dir.iterdir()} == {
        "seg_0001.wav",
        "seg_0001_aug_gain_minus3.wav",
        "seg_0001_aug_gain_plus3.wav",
    }
    augmented_rows = [row for row in rows if row.get("augmentation_method")]
    assert [row["augmentation_method"] for row in augmented_rows] == ["gain_minus3", "gain_plus3"]
    assert all(row["augmentation_source_segment_id"] == clean.id for row in augmented_rows)

    dataset_manifest = json.loads((tmp_project_dir / "work" / "rvc_train" / "dataset_manifest.json").read_text("utf-8"))
    assert dataset_manifest["summary"]["augmentation_applied"] is True
    assert dataset_manifest["summary"]["real_clean_segment_count"] == 1
    assert dataset_manifest["summary"]["real_clean_duration_sec"] == pytest.approx(1.0)
    assert dataset_manifest["summary"]["augmented_segment_count"] == 2
    assert dataset_manifest["summary"]["clean_segment_count"] == 3
    assert dataset_manifest["summary"]["clean_duration_sec"] >= 2.4


def test_rvc_train_dataset_rejects_duplicate_source_audio_segments(tmp_project_dir: Path) -> None:
    first = _segment_with_tts(tmp_project_dir, duration=1.0, segment_id="seg_0001")
    duplicate = _segment_with_tts(tmp_project_dir, duration=1.0, segment_id="seg_0002")
    duplicate.audio_for_mix = first.audio_for_mix
    cfg = ProjectConfig(rvc_train_backend="command", rvc_train_min_clean_sec=0.0)
    manifest = PipelineManifest(
        project_config=cfg,
        rights_audit=require_confirmed_rights(True, "test"),
        segments=[first, duplicate],
    )

    dataset_dir, rows = pipeline_steps._rvc_train_dataset(tmp_project_dir, manifest, force=False)

    assert [row["segment_id"] for row in rows] == [first.id]
    assert {path.name for path in dataset_dir.iterdir()} == {f"{first.id}.wav"}
    dataset_manifest = json.loads((tmp_project_dir / "work" / "rvc_train" / "dataset_manifest.json").read_text("utf-8"))
    rejected = {
        item["segment_id"]: item["reject_reasons"]
        for item in dataset_manifest["rejected_segments"]
    }
    assert rejected[duplicate.id] == [f"duplicate_source_audio:{first.id}"]


def test_train_rvc_registers_speaker_models_for_mixed_speakers(tmp_project_dir: Path) -> None:
    speaker_1 = _segment_with_tts(tmp_project_dir, duration=1.0, segment_id="seg_0001")
    speaker_2 = _segment_with_tts(tmp_project_dir, duration=1.0, segment_id="seg_0002")
    speaker_2.speaker_id = "speaker_0002"
    cfg = ProjectConfig(rvc_backend="mock", rvc_train_backend="mock", rvc_train_min_clean_sec=0.0)
    save_project_config(cfg.model_copy(update={"project_name": tmp_project_dir.name}), tmp_project_dir / "pipeline.yaml")
    manifest = PipelineManifest(
        project_config=cfg,
        rights_audit=require_confirmed_rights(True, "test"),
        segments=[speaker_1, speaker_2],
    )
    mark_stage(manifest, "synth", "completed")
    save_manifest(tmp_project_dir, manifest)

    trained = pipeline_steps.rvc_train_step(tmp_project_dir, confirm_rights=True, mock=True)

    saved_cfg = load_project_config(tmp_project_dir)
    assert sorted(saved_cfg.rvc_speaker_models) == ["speaker_0001", "speaker_0002"]
    assert sorted(trained.project_config.rvc_speaker_models) == ["speaker_0001", "speaker_0002"]
    assert trained.stage_state["train-rvc"]["speaker_count"] == 2
    for speaker_id, speaker_cfg in saved_cfg.rvc_speaker_models.items():
        assert Path(speaker_cfg.model_path).exists()
        assert speaker_cfg.index_path is not None
        assert Path(speaker_cfg.index_path).exists()
        assert f"work/rvc_train/speakers/{speaker_id}/" in speaker_cfg.model_path


def test_rvc_skips_training_and_uses_pre_rvc_fallback_when_clean_data_is_too_small(
    tmp_project_dir: Path,
) -> None:
    cfg = ProjectConfig(
        rvc_required=False,
        rvc_backend="command",
        rvc_command=["rvc", "{input}", "{output}", "{model}"],
        rvc_train_backend="command",
        rvc_train_command=["train-rvc", "{dataset}", "{output_model}", "{output_index}"],
        rvc_train_min_clean_sec=2.0,
        rvc_train_min_clean_segments=2,
        rvc_allow_pre_rvc_fallback=True,
    )
    save_project_config(cfg.model_copy(update={"project_name": tmp_project_dir.name}), tmp_project_dir / "pipeline.yaml")
    segment = _segment_with_tts(tmp_project_dir, duration=1.0, segment_id="seg_0001")
    manifest = PipelineManifest(
        project_config=cfg,
        rights_audit=require_confirmed_rights(True, "test"),
        segments=[segment],
    )
    mark_stage(manifest, "synth", "completed")
    save_manifest(tmp_project_dir, manifest)

    trained = pipeline_steps.rvc_train_step(tmp_project_dir, confirm_rights=True)
    converted = pipeline_steps.rvc_step(tmp_project_dir, confirm_rights=True)

    assert trained.stage_state["train-rvc"]["status"] == "skipped_insufficient_training_data"
    assert trained.stage_state["train-rvc"]["clean_segment_count"] == 1
    assert trained.stage_state["train-rvc"]["clean_duration_sec"] == pytest.approx(1.0)
    assert trained.stage_state["train-rvc"]["augmentation_applied"] is False
    assert trained.stage_state["train-rvc"]["augmentation_skipped_reason"] == "disabled"
    assert converted.stage_state["rvc"]["status"] == "completed"
    assert converted.stage_state["rvc"]["execution_mode"] == "pre_rvc_fallback"
    converted_segment = converted.segments[0]
    assert converted_segment.tts is not None
    assert converted_segment.rvc is not None
    assert converted_segment.rvc.output_path == converted_segment.tts.selected_candidate_path
    assert converted_segment.rvc.fallback_used is True
    assert converted_segment.rvc.accepted is False


def test_duration_mismatch_triggers_next_retry_profile(tmp_project_dir: Path, monkeypatch) -> None:
    model_path = tmp_project_dir / "models" / "voice.pth"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"model")
    cfg = ProjectConfig(
        rvc_backend="command",
        rvc_command=["rvc", "{input}", "{output}", "{model}", "{profile_name}"],
        rvc_model_path=str(model_path),
        rvc_duration_tolerance=0.1,
    )
    _save_synth_manifest(tmp_project_dir, cfg)

    def fake_convert(self, input_path: Path, output_path: Path, **kwargs: object) -> RVCCommandResult:
        profile = kwargs["profile"]
        duration = 0.2 if profile.name == "rmvpe_index045" else 1.0
        _write_exact_wav(output_path, duration=duration)
        return RVCCommandResult(output_path, ["fake"], "", "", 0, 0.0)

    monkeypatch.setattr(pipeline_steps.RVCCommandClient, "convert", fake_convert)
    manifest = pipeline_steps.rvc_step(tmp_project_dir, confirm_rights=True)

    segment = manifest.segments[0]
    assert segment.rvc is not None
    assert segment.rvc.selected_profile_name == "rmvpe_index035_safer"
    assert len(segment.rvc.attempts) == 2
    assert segment.rvc.attempts[0]["accepted"] is False
    assert segment.rvc.attempts[1]["accepted"] is True


def test_rvc_retries_failed_segments_twice_in_same_invocation(tmp_project_dir: Path, monkeypatch) -> None:
    model_path = tmp_project_dir / "models" / "voice.pth"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"model")
    cfg = ProjectConfig(
        rvc_backend="command",
        rvc_command=["rvc", "{input}", "{output}", "{model}", "{profile_name}"],
        rvc_model_path=str(model_path),
        rvc_failure_policy="error",
    )
    _save_synth_manifest(tmp_project_dir, cfg)
    calls: list[bool] = []

    def flaky_convert(self, input_path: Path, output_path: Path, **kwargs: object) -> RVCCommandResult:
        _ = self, input_path, kwargs
        calls.append(bool(kwargs["force"]))
        if len(calls) < 3:
            raise RuntimeError(f"transient rvc failure {len(calls)}")
        _write_exact_wav(output_path, duration=1.0)
        return RVCCommandResult(output_path, ["fake"], "", "", 0, 0.0)

    monkeypatch.setattr(pipeline_steps.RVCCommandClient, "convert", flaky_convert)

    manifest = pipeline_steps.rvc_step(tmp_project_dir, confirm_rights=True)

    segment = manifest.segments[0]
    assert calls == [False, True, True]
    assert manifest.stage_state["rvc"]["status"] == "completed"
    assert manifest.stage_state["rvc"]["retry_summary"]["retry_rounds"] == 2
    assert segment.status == "rvc_converted"
    assert segment.rvc is not None
    assert segment.rvc.accepted is True
    assert len(segment.rvc.attempts) == 3
    assert [attempt["error"] for attempt in segment.rvc.attempts[:2]] == [
        "transient rvc failure 1",
        "transient rvc failure 2",
    ]


def test_rvc_retry_failed_option_requeues_only_failed_segments(tmp_project_dir: Path, monkeypatch) -> None:
    model_path = tmp_project_dir / "models" / "voice.pth"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"model")
    cfg = ProjectConfig(
        rvc_backend="command",
        rvc_command=["rvc", "{input}", "{output}", "{model}", "{profile_name}"],
        rvc_model_path=str(model_path),
    )
    failed = _segment_with_tts(tmp_project_dir, duration=1.0, segment_id="seg_0001")
    failed.status = "failed"
    failed.errors = ["All RVC candidates failed or were rejected."]
    assert failed.tts is not None
    failed.rvc = RVCMetadata(
        backend="command",
        input_path=failed.tts.selected_candidate_path or "",
        accepted=False,
        error="All RVC candidates failed or were rejected.",
        attempts=[{"profile_name": "previous", "accepted": False, "error": "previous failure"}],
    )
    completed = _segment_with_tts(tmp_project_dir, duration=1.0, segment_id="seg_0002")
    completed_output = _write_exact_wav(tmp_project_dir / "work" / "rvc" / "seg_0002_final.wav", duration=1.0)
    completed.status = "rvc_converted"
    assert completed.tts is not None
    completed.rvc = RVCMetadata(
        backend="command",
        input_path=completed.tts.selected_candidate_path or "",
        output_path=str(completed_output),
        selected_profile_name="rmvpe_index045",
        accepted=True,
    )
    _save_synth_manifest(tmp_project_dir, cfg, segments=[failed, completed])
    converted_segments: list[str] = []

    def fake_convert(self, input_path: Path, output_path: Path, **kwargs: object) -> RVCCommandResult:
        _ = self, input_path
        converted_segments.append(str(kwargs["segment_id"]))
        assert kwargs["force"] is True
        _write_exact_wav(output_path, duration=1.0)
        return RVCCommandResult(output_path, ["fake"], "", "", 0, 0.0)

    monkeypatch.setattr(pipeline_steps.RVCCommandClient, "convert", fake_convert)

    manifest = pipeline_steps.rvc_step(tmp_project_dir, confirm_rights=True, retry_failed=True)

    retried = manifest.segments[0]
    assert converted_segments == ["seg_0001"]
    assert manifest.stage_state["rvc"]["retry_failed"] is True
    assert retried.status == "rvc_converted"
    assert retried.errors == []
    assert retried.rvc is not None
    assert retried.rvc.accepted is True
    assert retried.rvc.attempts[0]["profile_name"] == "previous"


def test_rvc_force_restarts_failed_segment_from_scratch(tmp_project_dir: Path, monkeypatch) -> None:
    model_path = tmp_project_dir / "models" / "voice.pth"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"model")
    cfg = ProjectConfig(
        rvc_backend="command",
        rvc_command=["rvc", "{input}", "{output}", "{model}", "{profile_name}"],
        rvc_model_path=str(model_path),
    )
    failed = _segment_with_tts(tmp_project_dir, duration=1.0, segment_id="seg_0001")
    failed.status = "failed"
    failed.errors = ["All RVC candidates failed or were rejected."]
    assert failed.tts is not None
    failed.rvc = RVCMetadata(
        backend="command",
        input_path=failed.tts.selected_candidate_path or "",
        candidate_paths=["work/rvc/candidates/seg_0001/previous.wav"],
        accepted=False,
        error="All RVC candidates failed or were rejected.",
        attempts=[{"profile_name": "previous", "accepted": False, "error": "previous failure"}],
    )
    _save_synth_manifest(tmp_project_dir, cfg, segments=[failed])
    converted_segments: list[str] = []

    def fake_convert(self, input_path: Path, output_path: Path, **kwargs: object) -> RVCCommandResult:
        _ = self, input_path
        converted_segments.append(str(kwargs["segment_id"]))
        assert kwargs["force"] is True
        _write_exact_wav(output_path, duration=1.0)
        return RVCCommandResult(output_path, ["fake"], "", "", 0, 0.0)

    monkeypatch.setattr(pipeline_steps.RVCCommandClient, "convert", fake_convert)

    manifest = pipeline_steps.rvc_step(tmp_project_dir, confirm_rights=True, force=True)

    restarted = manifest.segments[0]
    assert converted_segments == ["seg_0001"]
    assert restarted.status == "rvc_converted"
    assert restarted.errors == []
    assert restarted.rvc is not None
    assert restarted.rvc.accepted is True
    assert len(restarted.rvc.attempts) == 1
    assert restarted.rvc.attempts[0]["profile_name"] == "rmvpe_index045"
    assert "work/rvc/candidates/seg_0001/previous.wav" not in restarted.rvc.candidate_paths
    assert manifest.stage_state["rvc"]["force"] is True


def test_rvc_cli_accepts_retry_failed(cli_runner, tmp_project_dir: Path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_rvc_step(*args: object, **kwargs: object) -> PipelineManifest:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return PipelineManifest()

    monkeypatch.setattr(cli_module, "rvc_step", fake_rvc_step)

    result = cli_runner.invoke(
        app,
        [
            "rvc",
            "-p",
            str(tmp_project_dir),
            "--confirm-rights",
            "--retry-failed",
        ],
    )

    assert result.exit_code == 0, result.output
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["retry_failed"] is True


def test_rvc_cli_accepts_retry_failed_alias(cli_runner, tmp_project_dir: Path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_rvc_step(*args: object, **kwargs: object) -> PipelineManifest:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return PipelineManifest()

    monkeypatch.setattr(cli_module, "rvc_step", fake_rvc_step)

    result = cli_runner.invoke(
        app,
        [
            "rvc",
            "-p",
            str(tmp_project_dir),
            "--confirm-rights",
            "--retry",
            "failed",
        ],
    )

    assert result.exit_code == 0, result.output
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["retry_failed"] is True


def test_rvc_command_concurrency_runs_segments_in_parallel(tmp_project_dir: Path, monkeypatch) -> None:
    model_path = tmp_project_dir / "models" / "voice.pth"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"model")
    cfg = ProjectConfig(
        rvc_backend="command",
        rvc_command=["rvc", "{input}", "{output}", "{model}", "{profile_name}"],
        rvc_model_path=str(model_path),
        rvc_concurrency=3,
    )
    segments = [
        _segment_with_tts(tmp_project_dir, duration=1.0, segment_id=f"seg_{index:04d}")
        for index in range(1, 4)
    ]
    _save_synth_manifest(tmp_project_dir, cfg, segments=segments)
    lock = Lock()
    active = 0
    max_active = 0

    def fake_convert(self, input_path: Path, output_path: Path, **kwargs: object) -> RVCCommandResult:
        nonlocal active, max_active
        _ = self, input_path, kwargs
        with lock:
            active += 1
            max_active = max(max_active, active)
        try:
            time.sleep(0.05)
            _write_exact_wav(output_path, duration=1.0)
            return RVCCommandResult(output_path, ["fake"], "", "", 0, 0.0)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(pipeline_steps.RVCCommandClient, "convert", fake_convert)

    manifest = pipeline_steps.rvc_step(tmp_project_dir, confirm_rights=True)

    assert max_active > 1
    assert manifest.stage_state["rvc"]["status"] == "completed"
    assert manifest.stage_state["rvc"]["concurrency"] == 3
    assert all(segment.rvc and segment.rvc.accepted for segment in manifest.segments)


def test_rvc_batch_command_converts_segments_in_chunks(tmp_project_dir: Path, monkeypatch) -> None:
    model_path = tmp_project_dir / "models" / "voice.pth"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"model")
    cfg = ProjectConfig(
        rvc_backend="command",
        rvc_command=["rvc", "{input}", "{output}", "{model}", "{profile_name}"],
        rvc_batch_command=["rvc-batch", "--jobs", "{jobs}", "--results", "{results}", "--model", "{model}"],
        rvc_model_path=str(model_path),
        rvc_batch_infer=True,
        rvc_batch_size=2,
    )
    segments = [
        _segment_with_tts(tmp_project_dir, duration=1.0, segment_id=f"seg_{index:04d}")
        for index in range(1, 4)
    ]
    _save_synth_manifest(tmp_project_dir, cfg, segments=segments)
    seen_batches: list[list[str]] = []
    save_calls: list[Path] = []
    original_save_manifest = pipeline_steps.save_manifest

    def fake_convert_many(self, jobs: list[pipeline_steps.RVCBatchJob], **kwargs: object) -> dict[str, RVCCommandResult]:
        _ = self, kwargs
        seen_batches.append([job.segment_id for job in jobs])
        results: dict[str, RVCCommandResult] = {}
        for job in jobs:
            _write_exact_wav(job.output_path, duration=1.0)
            results[job.segment_id] = RVCCommandResult(job.output_path, ["batch"], "ok", "", 0, 0.1)
        return results

    def counting_save_manifest(project_dir: Path, manifest: PipelineManifest) -> Path:
        save_calls.append(project_dir)
        return original_save_manifest(project_dir, manifest)

    monkeypatch.setattr(pipeline_steps.RVCBatchCommandClient, "convert_many", fake_convert_many)
    monkeypatch.setattr(pipeline_steps, "save_manifest", counting_save_manifest)

    manifest = pipeline_steps.rvc_step(tmp_project_dir, confirm_rights=True)

    assert seen_batches == [["seg_0001", "seg_0002"], ["seg_0003"]]
    assert len(save_calls) == 3
    assert manifest.stage_state["rvc"]["status"] == "completed"
    assert manifest.stage_state["rvc"]["execution_mode"] == "batch"
    assert all(segment.rvc and segment.rvc.accepted for segment in manifest.segments)


def test_rvc_batch_skips_already_converted_segments(tmp_project_dir: Path, monkeypatch) -> None:
    model_path = tmp_project_dir / "models" / "voice.pth"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"model")
    cfg = ProjectConfig(
        rvc_backend="command",
        rvc_command=["rvc", "{input}", "{output}", "{model}", "{profile_name}"],
        rvc_batch_command=["rvc-batch", "--jobs", "{jobs}", "--results", "{results}", "--model", "{model}"],
        rvc_model_path=str(model_path),
        rvc_batch_infer=True,
        rvc_batch_size=2,
    )
    segments = [
        _segment_with_tts(tmp_project_dir, duration=1.0, segment_id=f"seg_{index:04d}")
        for index in range(1, 4)
    ]
    completed_output = _write_exact_wav(tmp_project_dir / "work" / "rvc" / "seg_0001_final.wav", duration=1.0)
    segments[0].status = "rvc_converted"
    assert segments[0].tts is not None
    assert segments[0].tts.selected_candidate_path is not None
    segments[0].rvc = RVCMetadata(
        backend="command",
        input_path=segments[0].tts.selected_candidate_path,
        output_path=str(completed_output),
        selected_profile_name="rmvpe_index045",
        accepted=True,
    )
    _save_synth_manifest(tmp_project_dir, cfg, segments=segments)
    seen_batches: list[list[str]] = []

    def fake_convert_many(self, jobs: list[pipeline_steps.RVCBatchJob], **kwargs: object) -> dict[str, RVCCommandResult]:
        _ = self, kwargs
        seen_batches.append([job.segment_id for job in jobs])
        results: dict[str, RVCCommandResult] = {}
        for job in jobs:
            _write_exact_wav(job.output_path, duration=1.0)
            results[job.segment_id] = RVCCommandResult(job.output_path, ["batch"], "ok", "", 0, 0.1)
        return results

    monkeypatch.setattr(pipeline_steps.RVCBatchCommandClient, "convert_many", fake_convert_many)

    manifest = pipeline_steps.rvc_step(tmp_project_dir, confirm_rights=True)

    assert seen_batches == [["seg_0002", "seg_0003"]]
    assert manifest.stage_state["rvc"]["status"] == "completed"
    assert all(segment.rvc and segment.rvc.accepted for segment in manifest.segments)


def test_rvc_rerun_refreshes_completed_segment_when_tts_is_newer(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = ProjectConfig(rvc_backend="mock")
    segment = _segment_with_tts(tmp_project_dir, duration=1.0, segment_id="seg_0001")
    completed_output = _write_exact_wav(tmp_project_dir / "work" / "rvc" / "seg_0001_final.wav", duration=1.0)
    segment.status = "rvc_converted"
    assert segment.tts is not None
    assert segment.tts.selected_candidate_path is not None
    segment.rvc = RVCMetadata(
        backend="mock",
        input_path=segment.tts.selected_candidate_path,
        output_path=str(completed_output),
        selected_profile_name="rmvpe_index045",
        accepted=True,
    )
    now = time.time()
    os.utime(completed_output, (now - 30, now - 30))
    os.utime(segment.tts.selected_candidate_path, (now + 30, now + 30))
    _save_synth_manifest(tmp_project_dir, cfg, segments=[segment])
    seen_inputs: list[Path] = []
    original_convert = pipeline_steps.RVCMockClient.convert

    def counting_convert(self, input_path: Path, output_path: Path, **kwargs: object):
        seen_inputs.append(input_path)
        return original_convert(self, input_path, output_path, **kwargs)

    monkeypatch.setattr(pipeline_steps.RVCMockClient, "convert", counting_convert)

    manifest = pipeline_steps.rvc_step(tmp_project_dir, confirm_rights=True)

    assert seen_inputs == [Path(segment.tts.selected_candidate_path)]
    assert manifest.segments[0].status == "rvc_converted"
    assert manifest.segments[0].rvc is not None
    assert manifest.segments[0].rvc.accepted is True


def test_regenerate_needs_step_rebuilds_only_regeneration_segments(
    tmp_project_dir: Path,
    monkeypatch,
) -> None:
    pipeline_steps.init_project(tmp_project_dir)
    cfg = ProjectConfig(rvc_backend="mock")
    ok_segment = _segment_with_tts(tmp_project_dir, duration=1.0, segment_id="seg_0001")
    ok_segment.status = "ok"
    ok_segment.qc = QCMetadata(recommendation="pass", status="ok")
    regen_segment = _segment_with_tts(tmp_project_dir, duration=1.0, segment_id="seg_0002")
    regen_segment.status = "needs_regeneration"
    regen_segment.qc = QCMetadata(
        recommendation="regenerate",
        status="needs_regeneration",
        issues=["duration_ratio_out_of_range"],
    )
    _save_synth_manifest(tmp_project_dir, cfg, segments=[ok_segment, regen_segment])
    synthesized: list[str] = []
    original_mock_synthesize = pipeline_steps._mock_synthesize

    def counting_mock_synthesize(output_path: Path, duration: float, seed: int, sample_rate: int = 48_000) -> Path:
        synthesized.append(output_path.name)
        return original_mock_synthesize(output_path, duration, seed, sample_rate)

    monkeypatch.setattr(pipeline_steps, "_mock_synthesize", counting_mock_synthesize)

    manifest = pipeline_steps.regenerate_needs_step(
        tmp_project_dir,
        tts_backend="mock",
        gemma_backend="mock",
        confirm_rights=True,
    )

    assert synthesized
    assert all(name.startswith("seg_0002") for name in synthesized)
    assert manifest.segments[0].status == "ok"
    assert manifest.segments[1].status == "ok"
    assert manifest.segments[1].rvc is not None
    assert manifest.segments[1].rvc.accepted is True
    assert manifest.stage_state["regenerate"]["status"] == "completed"
    assert manifest.stage_state["regenerate"]["target_segments"] == ["seg_0002"]


def test_regenerate_needs_step_honors_explicit_only_segment_ids(
    tmp_project_dir: Path,
    monkeypatch,
) -> None:
    pipeline_steps.init_project(tmp_project_dir)
    cfg = ProjectConfig(rvc_backend="mock")
    skipped = _segment_with_tts(tmp_project_dir, duration=1.0, segment_id="seg_0001")
    skipped.status = "scripted"
    skipped.tts = None
    target = _segment_with_tts(tmp_project_dir, duration=1.0, segment_id="seg_0002")
    target.status = "scripted"
    target.tts = None
    _save_synth_manifest(tmp_project_dir, cfg, segments=[skipped, target])
    synthesized: list[str] = []
    original_mock_synthesize = pipeline_steps._mock_synthesize

    def counting_mock_synthesize(output_path: Path, duration: float, seed: int, sample_rate: int = 48_000) -> Path:
        synthesized.append(output_path.name)
        return original_mock_synthesize(output_path, duration, seed, sample_rate)

    monkeypatch.setattr(pipeline_steps, "_mock_synthesize", counting_mock_synthesize)

    manifest = pipeline_steps.regenerate_needs_step(
        tmp_project_dir,
        tts_backend="mock",
        gemma_backend="mock",
        confirm_rights=True,
        only_segment_ids={"seg_0002"},
    )

    assert synthesized
    assert all(name.startswith("seg_0002") for name in synthesized)
    assert manifest.segments[0].status == "scripted"
    assert manifest.segments[0].tts is None
    assert manifest.segments[1].status == "ok"
    assert manifest.stage_state["regenerate"]["target_segments"] == ["seg_0002"]


def test_regenerate_needs_step_includes_recoverable_auto_repair_plan_segments(
    tmp_project_dir: Path,
    monkeypatch,
) -> None:
    pipeline_steps.init_project(tmp_project_dir)
    cfg = ProjectConfig(rvc_backend="mock")
    manual_recoverable = _segment_with_tts(tmp_project_dir, duration=1.0, segment_id="seg_0001")
    manual_recoverable.status = "needs_manual_review"
    manual_recoverable.tts = None
    manual_recoverable.rvc = None
    manual_recoverable.qc = QCMetadata(
        recommendation="manual_review",
        status="needs_manual_review",
        issues=["missing_selected_tts"],
    )
    manual_recoverable.analysis["ko_qc_repair_plan"] = {
        "action": "regenerate_tts",
        "terminal_manual": False,
        "reasons": ["missing_selected_tts"],
    }
    _save_synth_manifest(tmp_project_dir, cfg, segments=[manual_recoverable])
    synthesized: list[str] = []
    original_mock_synthesize = pipeline_steps._mock_synthesize

    def counting_mock_synthesize(output_path: Path, duration: float, seed: int, sample_rate: int = 48_000) -> Path:
        synthesized.append(output_path.name)
        return original_mock_synthesize(output_path, duration, seed, sample_rate)

    monkeypatch.setattr(pipeline_steps, "_mock_synthesize", counting_mock_synthesize)

    manifest = pipeline_steps.regenerate_needs_step(
        tmp_project_dir,
        tts_backend="mock",
        gemma_backend="mock",
        confirm_rights=True,
    )

    assert synthesized
    assert all(name.startswith("seg_0001") for name in synthesized)
    assert manifest.segments[0].status == "ok"
    assert manifest.segments[0].rvc is not None
    assert manifest.stage_state["regenerate"]["target_segments"] == ["seg_0001"]


def test_all_candidates_failing_causes_hard_failure(tmp_project_dir: Path, monkeypatch) -> None:
    model_path = tmp_project_dir / "models" / "voice.pth"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"model")
    cfg = ProjectConfig(
        rvc_backend="command",
        rvc_command=["rvc", "{input}", "{output}", "{model}", "{profile_name}"],
        rvc_model_path=str(model_path),
        rvc_duration_tolerance=0.1,
    )
    _save_synth_manifest(tmp_project_dir, cfg)

    def fake_convert(self, input_path: Path, output_path: Path, **kwargs: object) -> RVCCommandResult:
        _write_exact_wav(output_path, duration=0.2)
        return RVCCommandResult(output_path, ["fake"], "", "", 0, 0.0)

    monkeypatch.setattr(pipeline_steps.RVCCommandClient, "convert", fake_convert)

    with pytest.raises(RVCCommandError, match="RVC conversion failed"):
        pipeline_steps.rvc_step(tmp_project_dir, confirm_rights=True)

    manifest = load_manifest(tmp_project_dir)
    assert manifest.stage_state["rvc"]["status"] == "failed"
    assert manifest.segments[0].rvc is not None
    assert manifest.segments[0].rvc.accepted is False
    assert manifest.segments[0].tts is not None
    assert "work/tts" in manifest.segments[0].tts.selected_candidate_path


def test_command_backend_requires_rights_confirmation(tmp_project_dir: Path) -> None:
    model_path = tmp_project_dir / "models" / "voice.pth"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"model")
    cfg = ProjectConfig(
        rvc_backend="command",
        rvc_command=["rvc", "{input}", "{output}", "{model}"],
        rvc_model_path=str(model_path),
    )
    _save_synth_manifest(tmp_project_dir, cfg)

    with pytest.raises(RightsError, match="RVC"):
        pipeline_steps.rvc_step(tmp_project_dir, confirm_rights=False)


def test_command_backend_validates_missing_command_and_model_path(tmp_project_dir: Path) -> None:
    cfg = ProjectConfig(rvc_backend="command", rvc_command=[], rvc_model_path="missing.pth")

    with pytest.raises(RVCCommandError, match="rvc_command.*rvc_model_path"):
        validate_rvc_config(tmp_project_dir, cfg, real=True, segments=[_segment_with_tts(tmp_project_dir)])


def test_real_validation_rejects_mock_backend(tmp_project_dir: Path) -> None:
    cfg = ProjectConfig(rvc_backend="mock")

    with pytest.raises(RVCCommandError, match="real RVC requires rvc_backend: command"):
        validate_rvc_config(tmp_project_dir, cfg, real=True, segments=[_segment_with_tts(tmp_project_dir)])


def test_qc_fails_when_required_rvc_missing(tmp_project_dir: Path) -> None:
    cfg = ProjectConfig(rvc_backend="command", rvc_command=["rvc"], rvc_model_path="model.pth")
    _save_synth_manifest(tmp_project_dir, cfg)

    with pytest.raises(ValueError, match="RVC is required"):
        pipeline_steps.qc_step(tmp_project_dir, "mock")


def test_qc_uses_rvc_output_not_raw_tts(tmp_project_dir: Path, monkeypatch) -> None:
    cfg = ProjectConfig(rvc_backend="mock")
    _save_synth_manifest(tmp_project_dir, cfg)
    pipeline_steps.rvc_step(tmp_project_dir, confirm_rights=True)
    seen: list[Path] = []

    def fake_measure(audio_path: Path, target_duration_sec: float) -> dict[str, float]:
        seen.append(audio_path)
        return {
            "duration_sec": target_duration_sec,
            "duration_ratio": 1.0,
            "peak_dbfs": -12.0,
            "rms_dbfs": -30.0,
            "clipping_ratio": 0.0,
            "leading_silence_sec": 0.0,
            "trailing_silence_sec": 0.0,
        }

    monkeypatch.setattr(pipeline_steps, "measure_audio_qc", fake_measure)
    manifest = pipeline_steps.qc_step(tmp_project_dir, "mock")

    assert seen
    assert "work/rvc" in str(seen[0])
    assert manifest.segments[0].status == "ok"


def test_mix_and_export_fail_when_required_rvc_missing(tmp_project_dir: Path, tiny_wav_path: Path) -> None:
    cfg = ProjectConfig(rvc_backend="command", rvc_command=["rvc"], rvc_model_path="model.pth")
    segment = _segment_with_tts(tmp_project_dir)
    segment.status = "ok"
    segment.qc = QCMetadata(recommendation="pass", status="ok")
    _save_synth_manifest(tmp_project_dir, cfg, segment)
    manifest = load_manifest(tmp_project_dir)
    mark_stage(manifest, "qc", "completed")
    save_manifest(tmp_project_dir, manifest)

    with pytest.raises(ValueError, match="RVC is required"):
        pipeline_steps.mix_step(tmp_project_dir, confirm_rights=True)
    with pytest.raises(ValueError, match="RVC is required"):
        pipeline_steps.export_step(tiny_wav_path, tmp_project_dir, confirm_rights=True)
