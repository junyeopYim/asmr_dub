from __future__ import annotations

from typing import Any

from asmr_dub_pipeline.pipeline.context import PipelineContext
from asmr_dub_pipeline.pipeline.manifest_io import save_manifest
from asmr_dub_pipeline.rights import require_confirmed_rights
from asmr_dub_pipeline.schemas import (
    JapaneseScript,
    KoreanTranslation,
    PipelineManifest,
    ProjectConfig,
    QCMetadata,
    Segment,
    SourceScript,
    TTSMetadata,
)


def _value(obj: object, key: str) -> Any:
    if isinstance(obj, dict):
        return obj[key]
    if hasattr(obj, "model_dump"):
        dumped = obj.model_dump(mode="json")  # type: ignore[attr-defined]
        if key in dumped:
            return dumped[key]
    return getattr(obj, key)


def _segment(
    segment_id: str,
    *,
    duration: float = 1.0,
    source_script: bool = True,
    translation: bool = True,
    script: bool = True,
    selected_tts: str | None = "work/tts/seg_final.wav",
    status: str = "needs_manual_review",
    analysis: dict[str, Any] | None = None,
) -> Segment:
    return Segment(
        id=segment_id,
        start=0.0,
        end=duration,
        duration=duration,
        audio_for_gemma=f"work/segments/audio/{segment_id}_gemma.wav",
        audio_for_mix=f"work/segments/audio/{segment_id}_mix.wav",
        status=status,  # type: ignore[arg-type]
        analysis=analysis or {},
        source_script=SourceScript(
            text="こんにちは",
            language="ja",
            backend="mock",
            start=0.0,
            end=duration,
        )
        if source_script
        else None,
        translation_ko=KoreanTranslation(
            ko_literal="안녕하세요.",
            ko_natural="안녕하세요.",
            model="mock",
            batch_id="batch_0001",
        )
        if translation
        else None,
        script=JapaneseScript(
            ja_text="こんにちは",
            tts_text="안녕하세요.",
            tts_language="ko",
            source_language="ja",
            target_language="ko",
            expected_tts_duration_sec=duration,
        )
        if script
        else None,
        tts=TTSMetadata(
            backend="gpt-sovits",
            selected_candidate_path=selected_tts,
            target_language="ko",
        )
        if selected_tts is not None
        else None,
        qc=QCMetadata(recommendation="manual_review", status="needs_manual_review"),
    )


def test_classify_auto_repair_segment_routes_asr_translation_and_tts_failures() -> None:
    from asmr_dub_pipeline.pipeline.stages.auto_repair import classify_auto_repair_segment

    cases = [
        (
            _segment(
                "seg_asr",
                source_script=False,
                translation=False,
                script=False,
                selected_tts=None,
                analysis={"asr": {"error": "empty_text"}},
            ),
            "retry_asr",
            "transcribe",
        ),
        (
            _segment(
                "seg_translate",
                translation=False,
                script=False,
                selected_tts=None,
                analysis={"translate_ko": {"error": "missing_translation"}},
            ),
            "retry_translate_ko",
            "translate-ko",
        ),
        (
            _segment(
                "seg_tts",
                selected_tts=None,
                analysis={"ko_qc_repair_plan": {"action": "regenerate_tts", "terminal_manual": False}},
            ),
            "regenerate_tts",
            "synth",
        ),
    ]

    for segment, expected_action, expected_stage in cases:
        plan = classify_auto_repair_segment(segment, max_attempts=3)

        assert _value(plan, "terminal_manual") is False
        assert _value(plan, "action") == expected_action
        assert _value(plan, "stage") == expected_stage


def test_classify_auto_repair_segment_marks_max_attempts_terminal_manual() -> None:
    from asmr_dub_pipeline.pipeline.stages.auto_repair import classify_auto_repair_segment

    segment = _segment(
        "seg_retry_exhausted",
        selected_tts=None,
        analysis={
            "auto_repair": {
                "attempt_count": 3,
                "last_action": "regenerate_tts",
            },
            "ko_qc_repair_plan": {"action": "regenerate_tts", "terminal_manual": False},
        },
    )

    plan = classify_auto_repair_segment(segment, max_attempts=3)

    assert _value(plan, "terminal_manual") is True
    assert _value(plan, "action") == "manual_review"
    assert "max_attempts" in _value(plan, "reasons")


def test_run_auto_repair_stage_plan_only_records_retry_accounting(tmp_project_dir) -> None:
    from asmr_dub_pipeline.pipeline.stages.auto_repair import run_auto_repair_stage

    segment = _segment(
        "seg_0001",
        selected_tts=None,
        analysis={"ko_qc_repair_plan": {"action": "regenerate_tts", "terminal_manual": False}},
    )
    save_manifest(
        tmp_project_dir,
        PipelineManifest(
            rights_audit=require_confirmed_rights(True, "test"),
            segments=[segment],
        ),
    )

    manifest = run_auto_repair_stage(
        PipelineContext.load(tmp_project_dir),
        confirm_rights=True,
        max_attempts=3,
        plan_only=True,
    )

    repaired = manifest.segments[0]
    auto_repair = repaired.analysis["auto_repair"]
    assert auto_repair["action"] == "regenerate_tts"
    assert auto_repair["attempt_count"] == 1
    assert auto_repair["terminal_manual"] is False
    assert manifest.stage_state["auto-repair"]["status"] == "planned"


def test_auto_repair_does_not_count_unverified_closure_as_repaired(
    tmp_project_dir,
    monkeypatch,
) -> None:
    from asmr_dub_pipeline.pipeline.stages.auto_repair import run_auto_repair_stage

    segment = _segment(
        "seg_0001",
        analysis={"ko_qc_repair_plan": {"action": "regenerate_tts", "terminal_manual": False}},
    )
    save_manifest(
        tmp_project_dir,
        PipelineManifest(
            rights_audit=require_confirmed_rights(True, "test"),
            project_config=ProjectConfig(rvc_backend="mock", rvc_required=False, rvc_train_required=False),
            segments=[segment],
        ),
    )

    def fake_closure(ctx: PipelineContext, target_nodes: list[str], segment_ids: set[str], **kwargs: object):
        manifest = ctx.reload_manifest()
        manifest.segments[0].status = "ok"
        save_manifest(ctx.project_dir, manifest)
        return {
            "executed_nodes": target_nodes,
            "segment_ids": sorted(segment_ids),
            "verified": False,
            "verification": {"seg_0001": {"ok": False, "issues": ["missing_qc_input_rvc_generation_id"]}},
        }

    monkeypatch.setattr("asmr_dub_pipeline.pipeline.stages.auto_repair.run_closure", fake_closure)

    repaired = run_auto_repair_stage(PipelineContext.load(tmp_project_dir), confirm_rights=True)

    assert repaired.stage_state["auto-repair"]["repaired_count"] == 0
    assert repaired.stage_state["auto-repair"]["verification_failed_count"] == 1
    assert repaired.segments[0].status == "needs_manual_review"
    auto_repair = repaired.segments[0].analysis["auto_repair"]
    assert auto_repair["closure_verified"] is False
    assert auto_repair["manual_review_reason"] == "closure_verification_failed"


def test_auto_repair_translation_retry_passes_only_target_segment_ids(
    tmp_project_dir,
    monkeypatch,
) -> None:
    from asmr_dub_pipeline.pipeline.stages.auto_repair import run_auto_repair_stage

    target = _segment(
        "seg_0002",
        translation=False,
        script=False,
        selected_tts=None,
        status="quarantined",
        analysis={"translate_ko_quarantine": {"recoverable": True}},
    )
    untouched = _segment("seg_0001", status="ok")
    save_manifest(
        tmp_project_dir,
        PipelineManifest(
            rights_audit=require_confirmed_rights(True, "test"),
            project_config=ProjectConfig(rvc_backend="mock", rvc_required=False, rvc_train_required=False),
            segments=[untouched, target],
        ),
    )
    captured: dict[str, object] = {}

    def fake_translate(ctx: PipelineContext, *args: object, **kwargs: object):
        captured["only_segment_ids"] = kwargs.get("only_segment_ids")
        manifest = ctx.reload_manifest()
        manifest.segments[1].status = "transcribed"
        manifest.segments[1].translation_ko = KoreanTranslation(
            ko_literal="안녕하세요.",
            ko_natural="안녕하세요.",
            model="mock",
            batch_id="batch_0001",
        )
        save_manifest(ctx.project_dir, manifest)
        return manifest

    def fake_korean_script(ctx: PipelineContext, *args: object, **kwargs: object):
        captured["korean_script_only_segment_ids"] = kwargs.get("only_segment_ids")
        return ctx.reload_manifest()

    def fake_closure(ctx: PipelineContext, target_nodes: list[str], segment_ids: set[str], **kwargs: object):
        return {"executed_nodes": target_nodes, "segment_ids": sorted(segment_ids), "verified": True, "verification": {}}

    monkeypatch.setattr("asmr_dub_pipeline.pipeline.stages.translate_ko.run_translate_ko_stage", fake_translate)
    monkeypatch.setattr("asmr_dub_pipeline.pipeline.stages.korean_script.run_korean_script_stage", fake_korean_script)
    monkeypatch.setattr("asmr_dub_pipeline.pipeline.stages.auto_repair.run_closure", fake_closure)

    run_auto_repair_stage(PipelineContext.load(tmp_project_dir), confirm_rights=True)

    assert captured["only_segment_ids"] == {"seg_0002"}
    assert captured["korean_script_only_segment_ids"] == {"seg_0002"}
