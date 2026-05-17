from __future__ import annotations

from pathlib import Path

from conftest import write_tiny_wav

from asmr_dub_pipeline.pipeline.context import PipelineContext
from asmr_dub_pipeline.pipeline.manifest_io import save_manifest
from asmr_dub_pipeline.pipeline.state import mark_stage
from asmr_dub_pipeline.rights import require_confirmed_rights
from asmr_dub_pipeline.schemas import (
    JapaneseScript,
    PipelineManifest,
    ProjectConfig,
    QCMetadata,
    Segment,
    SourceScript,
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
        segment.analysis["fake_closure"] = {"target_nodes": target_nodes, "kwargs": kwargs}
        save_manifest(ctx.project_dir, manifest)
        return {"executed_nodes": target_nodes, "segment_ids": sorted(segment_ids), "verified": True}

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
