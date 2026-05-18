from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from asmr_dub_pipeline.pipeline import steps as pipeline_steps
from asmr_dub_pipeline.rights import require_confirmed_rights
from asmr_dub_pipeline.schemas import JapaneseScript, PipelineManifest, ProjectConfig, Segment, SourceScript, TTSMetadata


def _write_wav(path: Path, duration: float = 1.0, sample_rate: int = 48_000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(max(1, int(sample_rate * duration)), dtype=np.float32) / sample_rate
    tone = 0.05 * np.sin(2 * np.pi * 220.0 * t)
    sf.write(str(path), np.stack([tone, tone], axis=1), sample_rate)
    return path


def _segment(project_dir: Path, segment_id: str, logical_duration: float, analysis: dict | None = None) -> Segment:
    tts_path = _write_wav(project_dir / "work" / "tts" / f"{segment_id}_final.wav")
    source_path = _write_wav(project_dir / "work" / "segments" / "audio" / f"{segment_id}_mix.wav")
    return Segment(
        id=segment_id,
        start=0.0,
        end=logical_duration,
        duration=logical_duration,
        audio_for_gemma=f"work/segments/audio/{segment_id}_gemma.wav",
        audio_for_mix=str(source_path.relative_to(project_dir)),
        speaker_id="speaker_0001",
        analysis=analysis or {"speaker_count": 1},
        source_script=SourceScript(text="こんにちは。", language="ja", backend="mock", start=0.0, end=logical_duration),
        status="synthesized",
        script=JapaneseScript(ja_text="こんにちは。", tts_text="안녕하세요.", tts_language="ko"),
        tts=TTSMetadata(selected_candidate_path=str(tts_path)),
    )


def _dataset_manifest(project_dir: Path) -> dict:
    return json.loads((project_dir / "work" / "rvc_train" / "dataset_manifest.json").read_text("utf-8"))


def test_rvc_strict_pass_uses_no_soft_asmr_relaxation_when_enough_clean_data(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clean = _segment(tmp_project_dir, "seg_clean", 600.0)
    soft = _segment(
        tmp_project_dir,
        "seg_soft",
        120.0,
        {
            "speaker_count": 1,
            "voice_training": {"exclude": True, "clean_voice": True, "eligible": True, "effect_tags": ["whisper"]},
        },
    )
    cfg = ProjectConfig(rvc_train_min_clean_sec=0.0, rvc_train_absolute_min_clean_sec=0.0, rvc_train_target_clean_sec=600.0)
    durations = {"seg_clean_mix.wav": 600.0, "seg_soft_mix.wav": 120.0}
    monkeypatch.setattr(pipeline_steps, "duration_sec", lambda path: durations[Path(path).name])

    _dataset_dir, rows = pipeline_steps._rvc_train_dataset(
        tmp_project_dir,
        PipelineManifest(project_config=cfg, rights_audit=require_confirmed_rights(True, "test"), segments=[clean, soft]),
        force=False,
    )

    payload = _dataset_manifest(tmp_project_dir)
    assert [row["segment_id"] for row in rows] == [clean.id]
    assert payload["summary"]["training_selection_pass"] == "strict"
    assert payload["summary"]["strict_clean_duration_sec"] == pytest.approx(600.0)
    assert payload["summary"]["soft_asmr_relaxation_applied"] is False
    rejected = {item["segment_id"]: item["reject_reasons"] for item in payload["rejected_segments"]}
    assert "voice_training_soft_effect_tag_requires_low_data_relaxation:whisper" in rejected[soft.id]


def test_rvc_low_data_relaxed_pass_applies_soft_tags_and_preserves_hard_rejects(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clean = _segment(tmp_project_dir, "seg_clean", 120.0)
    soft = _segment(
        tmp_project_dir,
        "seg_soft",
        120.0,
        {
            "speaker_count": 1,
            "voice_training": {"exclude": True, "clean_voice": True, "eligible": True, "effect_tags": ["close_mic"]},
        },
    )
    hard = _segment(
        tmp_project_dir,
        "seg_hard",
        120.0,
        {
            "speaker_count": 1,
            "training_flags": ["overlap"],
            "voice_training": {"exclude": True, "clean_voice": True, "eligible": True, "effect_tags": ["reverb"]},
        },
    )
    cfg = ProjectConfig(rvc_train_min_clean_sec=0.0, rvc_train_absolute_min_clean_sec=0.0, rvc_train_target_clean_sec=600.0)
    durations = {"seg_clean_mix.wav": 120.0, "seg_soft_mix.wav": 120.0, "seg_hard_mix.wav": 120.0}
    monkeypatch.setattr(pipeline_steps, "duration_sec", lambda path: durations[Path(path).name])

    _dataset_dir, rows = pipeline_steps._rvc_train_dataset(
        tmp_project_dir,
        PipelineManifest(project_config=cfg, rights_audit=require_confirmed_rights(True, "test"), segments=[clean, soft, hard]),
        force=False,
    )

    payload = _dataset_manifest(tmp_project_dir)
    assert [row["segment_id"] for row in rows] == [clean.id, soft.id]
    assert payload["summary"]["training_selection_pass"] == "low_data_relaxed"
    assert payload["summary"]["soft_asmr_accepted_count"] == 1
    assert payload["summary"]["accepted_override_reason_counts"]["voice_training_soft_effect_tag:close_mic"] == 1
    rejected = {item["segment_id"]: item["reject_reasons"] for item in payload["rejected_segments"]}
    assert "manual_training_exclude:hard_effect_tag:reverb" in rejected[hard.id]
    assert "disallowed_training_tag:overlap" in rejected[hard.id]
