from __future__ import annotations

from pathlib import Path

from conftest import write_tiny_wav

from asmr_dub_pipeline.pipeline.artifacts import (
    make_qc_generation_id,
    make_rvc_generation_id,
    make_script_generation_id,
    make_selected_tts_generation_id,
)
from asmr_dub_pipeline.pipeline.context import PipelineContext
from asmr_dub_pipeline.pipeline.manifest_io import save_manifest
from asmr_dub_pipeline.pipeline.state import mark_stage
from asmr_dub_pipeline.rights import require_confirmed_rights
from asmr_dub_pipeline.schemas import (
    JapaneseScript,
    PipelineManifest,
    ProjectConfig,
    QCMetadata,
    RVCMetadata,
    Segment,
    SourceScript,
    TTSMetadata,
)


def _repair_segment(project_dir: Path) -> Segment:
    audio = write_tiny_wav(project_dir / "work" / "segments" / "audio" / "seg_0001_mix.wav")
    return Segment(
        id="seg_0001",
        start=0.0,
        end=1.0,
        duration=1.0,
        audio_for_gemma=str(audio),
        audio_for_mix=str(audio),
        status="needs_manual_review",
        source_script=SourceScript(text="3 2 1", language="ja", backend="mock", start=0.0, end=1.0),
        script=JapaneseScript(
            ja_text="3 2 1",
            tts_text="삼, 이, 일",
            tts_language="ko",
            source_language="ja",
            target_language="ko",
        ),
        qc=QCMetadata(
            recommendation="manual_review",
            status="needs_manual_review",
            issues=["pronunciation_qc_failed"],
        ),
        analysis={
            "ko_qc_repair_plan": {
                "action": "fallback_tts_qwen",
                "terminal_manual": False,
                "root_cause": "gpt_sovits_pronunciation_qc_failed",
            }
        },
    )


def test_auto_repair_fallback_qwen_runs_candidate_pool_closure(tmp_project_dir: Path, monkeypatch) -> None:
    from asmr_dub_pipeline.pipeline.stages.auto_repair import run_auto_repair_stage

    cfg = ProjectConfig(
        rvc_backend="mock",
        rvc_required=False,
        rvc_train_required=False,
        rvc_allow_pre_rvc_fallback=True,
    )
    manifest = PipelineManifest(
        project_config=cfg,
        rights_audit=require_confirmed_rights(True, "test"),
        segments=[_repair_segment(tmp_project_dir)],
    )
    mark_stage(manifest, "korean-script", "completed")
    save_manifest(tmp_project_dir, manifest)
    executed: list[tuple[str, tuple[str, ...]]] = []

    def fake_run_closure(ctx: PipelineContext, target_nodes: list[str], segment_ids: set[str], **kwargs: object):
        executed.extend((node, tuple(sorted(segment_ids))) for node in target_nodes)
        manifest = ctx.reload_manifest()
        segment = manifest.segments[0]
        segment.status = "ok"
        segment.tts = TTSMetadata(
            backend="qwen-tts",
            selected_candidate_path="work/tts/seg_0001_final.wav",
            selected_candidate_id="qwen_tts_cand_00",
            selected_tts_generation_id="tts:selected",
            generation_id="tts:selected",
        )
        segment.analysis["fake_closure"] = {"target_nodes": target_nodes, "kwargs": kwargs}
        save_manifest(ctx.project_dir, manifest)
        return {
            "executed_nodes": target_nodes,
            "segment_ids": sorted(segment_ids),
            "verified": True,
            "verification": {"seg_0001": {"ok": True, "issues": []}},
        }

    monkeypatch.setattr("asmr_dub_pipeline.pipeline.stages.auto_repair.run_closure", fake_run_closure)

    repaired = run_auto_repair_stage(PipelineContext.load(tmp_project_dir), confirm_rights=True)

    assert executed == [
        ("tts.candidate_pool", ("seg_0001",)),
        ("tts.select", ("seg_0001",)),
        ("rvc", ("seg_0001",)),
        ("qc", ("seg_0001",)),
    ]
    assert repaired.segments[0].status == "ok"
    summary = repaired.stage_state["auto-repair"]
    assert summary["repaired_count"] == 1
    assert repaired.segments[0].analysis["auto_repair"]["closure"]["executed_nodes"] == [
        "tts.candidate_pool",
        "tts.select",
        "rvc",
        "qc",
    ]
    attempt = repaired.stage_state["auto-repair"]["per_segment_attempts"][0]
    assert attempt["output_status"] == "ok"
    assert attempt["failure_signature_after"] == "ok"
    assert attempt["selected_candidate_id"] == "qwen_tts_cand_00"
    assert attempt["closure_verified"] is True


def test_auto_repair_repeated_failure_signature_becomes_terminal_manual() -> None:
    from asmr_dub_pipeline.pipeline.stages.auto_repair import classify_auto_repair_segment

    segment = _repair_segment(Path("/tmp/project"))
    segment.analysis["auto_repair"] = {
        "attempt_count": 1,
        "last_failure_signature": "pronunciation_qc_failed",
    }

    plan = classify_auto_repair_segment(segment, max_attempts=3)

    assert plan["terminal_manual"] is True
    assert plan["action"] == "manual_review"
    assert "repeated_failure_signature" in plan["reasons"]


def test_run_closure_uses_strict_non_mutating_generation_verification(
    tmp_project_dir: Path,
    monkeypatch,
) -> None:
    from asmr_dub_pipeline.pipeline.runner import run_closure

    script = JapaneseScript(
        ja_text="こんにちは",
        tts_text="안녕하세요",
        tts_language="ko",
        source_language="ja",
        target_language="ko",
    )
    script_generation_id = make_script_generation_id(script)
    selected_generation_id = make_selected_tts_generation_id(
        segment_id="seg_0001",
        candidate_id="cand_001",
        wav_path="work/tts/seg_0001_final.wav",
        input_script_generation_id=script_generation_id,
    )
    rvc_generation_id = make_rvc_generation_id(
        segment_id="seg_0001",
        output_path="work/rvc/seg_0001_final.wav",
        input_selected_tts_generation_id=selected_generation_id,
    )
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=1.0,
        duration=1.0,
        audio_for_gemma="work/segments/audio/seg_0001_gemma.wav",
        audio_for_mix="work/segments/audio/seg_0001_mix.wav",
        script=script,
        tts=TTSMetadata(
            backend="gpt-sovits",
            selected_candidate_path="work/tts/seg_0001_final.wav",
            selected_candidate_id="cand_001",
            input_script_generation_id=script_generation_id,
            selected_tts_generation_id=selected_generation_id,
            generation_id=selected_generation_id,
        ),
        rvc=RVCMetadata(
            backend="mock",
            input_path="work/tts/seg_0001_final.wav",
            output_path="work/rvc/seg_0001_final.wav",
            accepted=True,
            generation_id=rvc_generation_id,
            input_selected_tts_generation_id=None,
        ),
        qc=QCMetadata(
            recommendation="pass",
            status="ok",
            generation_id=make_qc_generation_id(
                segment_id="seg_0001",
                input_rvc_generation_id=rvc_generation_id,
                recommendation="pass",
                issues=[],
            ),
            input_rvc_generation_id=None,
        ),
    )
    save_manifest(tmp_project_dir, PipelineManifest(segments=[segment]))

    def fake_qc(ctx: PipelineContext, *args: object, **kwargs: object):
        return ctx.reload_manifest()

    monkeypatch.setattr("asmr_dub_pipeline.pipeline.runner.run_qc_stage", fake_qc)

    result = run_closure(PipelineContext.load(tmp_project_dir), ["qc"], {"seg_0001"})
    reloaded = PipelineContext.load(tmp_project_dir).reload_manifest().segments[0]

    assert result["verified"] is False
    assert "missing_rvc_input_selected_tts_generation_id" in result["verification"]["seg_0001"]["issues"]
    assert "missing_qc_input_rvc_generation_id" in result["verification"]["seg_0001"]["issues"]
    assert reloaded.rvc.input_selected_tts_generation_id is None
    assert reloaded.qc.input_rvc_generation_id is None


def test_run_closure_requires_metadata_for_requested_target_nodes(
    tmp_project_dir: Path,
    monkeypatch,
) -> None:
    from asmr_dub_pipeline.pipeline.runner import run_closure

    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=1.0,
        duration=1.0,
        audio_for_gemma="work/segments/audio/seg_0001_gemma.wav",
        audio_for_mix="work/segments/audio/seg_0001_mix.wav",
        script=JapaneseScript(
            ja_text="こんにちは",
            tts_text="안녕하세요",
            tts_language="ko",
            source_language="ja",
            target_language="ko",
        ),
    )
    save_manifest(tmp_project_dir, PipelineManifest(segments=[segment]))

    def fake_stage(ctx: PipelineContext, *args: object, **kwargs: object):
        return ctx.reload_manifest()

    monkeypatch.setattr("asmr_dub_pipeline.pipeline.runner.run_tts_select_stage", fake_stage)
    monkeypatch.setattr("asmr_dub_pipeline.pipeline.runner.run_rvc_stage", fake_stage)
    monkeypatch.setattr("asmr_dub_pipeline.pipeline.runner.run_qc_stage", fake_stage)

    result = run_closure(PipelineContext.load(tmp_project_dir), ["tts.select", "rvc", "qc"], {"seg_0001"})

    issues = result["verification"]["seg_0001"]["issues"]
    assert result["verified"] is False
    assert "missing_tts_metadata" in issues
    assert "missing_rvc_metadata" in issues
    assert "missing_qc_metadata" in issues
