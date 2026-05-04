from __future__ import annotations

import os
import time
from pathlib import Path
from threading import Lock

import numpy as np
import pytest
import soundfile as sf

from asmr_dub_pipeline.config import save_project_config
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
    cfg = ProjectConfig(rvc_backend="mock", rvc_train_backend="mock")
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
    cfg = ProjectConfig(rvc_backend="mock", rvc_train_backend="mock")
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
    cfg = ProjectConfig(rvc_backend="mock", rvc_train_backend="mock")
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
