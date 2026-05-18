from __future__ import annotations

from pathlib import Path
from typing import Any

from asmr_dub_pipeline.pipeline.context import PipelineContext
from asmr_dub_pipeline.pipeline.manifest_io import save_manifest
from asmr_dub_pipeline.pipeline.state import mark_stage
from asmr_dub_pipeline.schemas import JapaneseScript, PipelineManifest, Segment, SourceScript, TTSMetadata


def _late_qwen_segment(segment_id: str = "seg_late") -> Segment:
    return Segment(
        id=segment_id,
        start=0.0,
        end=3.0,
        duration=3.0,
        audio_for_gemma=f"work/segments/audio/{segment_id}_gemma.wav",
        audio_for_mix=f"work/segments/audio/{segment_id}_mix.wav",
        status="needs_regeneration",
        source_script=SourceScript(
            text="お願いします",
            language="ja",
            backend="mock",
            start=0.0,
            end=3.0,
        ),
        script=JapaneseScript(
            ja_text="お願いします",
            tts_text="부탁드려요.",
            tts_language="ko",
            source_language="ja",
            target_language="ko",
            expected_tts_duration_sec=3.0,
        ),
        analysis={
            "ko_qc_repair_plan": {
                "action": "fallback_tts_qwen",
                "route": "late_qwen_after_gsv_hard_fail",
            }
        },
    )


def test_late_qwen_scheduled_segments_are_consumed_before_train_rvc(tmp_project_dir: Path) -> None:
    from asmr_dub_pipeline.pipeline.stages.tts_late_qwen import run_late_qwen_pre_rvc_closure

    segment = _late_qwen_segment()
    manifest = PipelineManifest(segments=[segment])
    mark_stage(
        manifest,
        "synth",
        "completed_with_hard_failed_candidates",
        selected_segments=[],
        hard_failed_segments=[],
        late_qwen_scheduled_segments=[segment.id],
    )
    save_manifest(tmp_project_dir, manifest)

    def fake_closure(
        ctx: PipelineContext,
        target_nodes: list[str],
        segment_ids: set[str],
        **kwargs: Any,
    ) -> dict[str, Any]:
        assert target_nodes == ["tts.candidate_pool", "tts.select"]
        assert kwargs["tts_backend"] == "qwen"
        reloaded = ctx.reload_manifest()
        target = reloaded.segments[0]
        target.status = "needs_regeneration"
        target.tts = None
        save_manifest(ctx.project_dir, reloaded)
        return {"executed_nodes": target_nodes, "segment_ids": sorted(segment_ids), "verified": False}

    result = run_late_qwen_pre_rvc_closure(
        PipelineContext.load(tmp_project_dir),
        segment_ids={segment.id},
        refs_path=Path("refs/refs.json"),
        confirm_rights=True,
        closure_runner=fake_closure,
    )

    resolved = PipelineContext.load(tmp_project_dir).reload_manifest()
    resolved_segment = resolved.segments[0]
    assert result["attempted_segments"] == [segment.id]
    assert result["terminal_segments"] == [segment.id]
    assert resolved_segment.status == "needs_manual_review"
    assert resolved_segment.status != "needs_regeneration"
    marker = resolved_segment.analysis["late_qwen_pre_rvc"]
    assert marker["attempted"] is True
    assert marker["resolved"] is False
    assert marker["terminal_reason"] == "late_qwen_unresolved_before_train_rvc"
    assert resolved.stage_state["synth"]["late_qwen_scheduled_segments"] == []
    assert resolved.stage_state["synth"]["late_qwen_pre_rvc_terminal_segments"] == [segment.id]


def test_orchestrator_calls_late_qwen_pre_rvc_before_train_rvc(
    tmp_project_dir: Path,
    tiny_wav_path: Path,
    monkeypatch,
) -> None:
    import asmr_dub_pipeline.orchestrator as orchestrator
    from asmr_dub_pipeline.config import save_project_config
    from asmr_dub_pipeline.pipeline.manifest_io import load_manifest
    from asmr_dub_pipeline.schemas import ProjectConfig

    cfg = ProjectConfig(
        rvc_backend="mock",
        rvc_train_backend="mock",
        source_separation_backend="mock",
        tts={"candidate_pool_enabled": True},
    )
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    initial = PipelineManifest(project_config=cfg, segments=[_late_qwen_segment()])
    save_manifest(tmp_project_dir, initial)
    order: list[str] = []

    def passthrough(name: str):
        def _inner(*args: Any, **kwargs: Any) -> PipelineManifest:
            order.append(name)
            return load_manifest(tmp_project_dir)

        return _inner

    monkeypatch.setattr(orchestrator, "init_project", lambda project_dir: project_dir.mkdir(parents=True, exist_ok=True))
    monkeypatch.setattr(orchestrator, "_transcribe_with_optional_source_separation_fallback", passthrough("transcribe"))
    for attr, name in [
        ("extract_step", "extract"),
        ("source_separation_step", "source-separation"),
        ("segment_step", "segment"),
        ("audio_style_step", "audio-style"),
        ("translate_ko_step", "translate-ko"),
        ("korean_script_step", "korean-script"),
        ("tts_candidate_pool_step", "tts.candidate_pool"),
        ("rvc_step", "rvc"),
        ("qc_step", "qc"),
        ("mix_step", "mix"),
        ("export_step", "export"),
    ]:
        monkeypatch.setattr(orchestrator, attr, passthrough(name))

    def fake_tts_select(*args: Any, **kwargs: Any) -> PipelineManifest:
        order.append("tts.select")
        manifest = load_manifest(tmp_project_dir)
        manifest.segments[0].status = "needs_regeneration"
        manifest.segments[0].tts = None
        mark_stage(
            manifest,
            "synth",
            "completed_with_hard_failed_candidates",
            selected_segments=[],
            hard_failed_segments=[],
            late_qwen_scheduled_segments=[manifest.segments[0].id],
        )
        save_manifest(tmp_project_dir, manifest)
        return manifest

    def fake_late_qwen(*args: Any, **kwargs: Any) -> dict[str, Any]:
        order.append("late-qwen-pre-rvc")
        manifest = load_manifest(tmp_project_dir)
        segment = manifest.segments[0]
        segment.status = "synthesized"
        segment.tts = TTSMetadata(selected_candidate_path="work/tts/seg_late_final.wav")
        mark_stage(
            manifest,
            "synth",
            "completed",
            selected_segments=[segment.id],
            hard_failed_segments=[],
            late_qwen_scheduled_segments=[],
            late_qwen_pre_rvc_resolved_segments=[segment.id],
        )
        save_manifest(tmp_project_dir, manifest)
        return {"resolved_segments": [segment.id]}

    def fake_train_rvc(*args: Any, **kwargs: Any) -> PipelineManifest:
        order.append("train-rvc")
        manifest = load_manifest(tmp_project_dir)
        assert manifest.segments[0].status == "synthesized"
        return manifest

    monkeypatch.setattr(orchestrator, "tts_select_step", fake_tts_select)
    monkeypatch.setattr(orchestrator, "tts_late_qwen_pre_rvc_step", fake_late_qwen)
    monkeypatch.setattr(orchestrator, "rvc_train_step", fake_train_rvc)

    orchestrator.run_pipeline(tiny_wav_path, tmp_project_dir, confirm_rights=True, mock=True)

    assert order.index("tts.select") < order.index("late-qwen-pre-rvc") < order.index("train-rvc")
