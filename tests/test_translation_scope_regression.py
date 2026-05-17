from __future__ import annotations

import json
from pathlib import Path

from asmr_dub_pipeline.pipeline.context import PipelineContext
from asmr_dub_pipeline.pipeline.manifest_io import save_manifest
from asmr_dub_pipeline.pipeline.stages.translate_ko import run_translate_ko_stage
from asmr_dub_pipeline.pipeline.state import mark_stage
from asmr_dub_pipeline.rights import require_confirmed_rights
from asmr_dub_pipeline.schemas import (
    KoreanTranslation,
    PipelineManifest,
    ProjectConfig,
    QCMetadata,
    RVCMetadata,
    Segment,
    SourceScript,
    TTSMetadata,
)


def _segment(
    segment_id: str,
    *,
    text: str,
    translation: KoreanTranslation | None = None,
    status: str = "transcribed",
) -> Segment:
    return Segment(
        id=segment_id,
        start=0.0,
        end=1.0,
        duration=1.0,
        audio_for_gemma=f"work/segments/audio/{segment_id}_gemma.wav",
        audio_for_mix=f"work/segments/audio/{segment_id}_mix.wav",
        status=status,  # type: ignore[arg-type]
        source_script=SourceScript(text=text, language="ja", backend="mock", start=0.0, end=1.0),
        translation_ko=translation,
        tts=TTSMetadata(backend="mock", selected_candidate_path=f"work/tts/{segment_id}_final.wav")
        if translation is not None
        else None,
        rvc=RVCMetadata(
            backend="mock",
            input_path=f"work/tts/{segment_id}_final.wav",
            output_path=f"work/rvc/{segment_id}_final.wav",
            accepted=True,
        )
        if translation is not None
        else None,
        qc=QCMetadata(recommendation="pass", status="ok") if translation is not None else None,
        mix={"included": True} if translation is not None else {},
    )


def test_translate_ko_only_segment_ids_preserves_non_target_segments(tmp_project_dir: Path) -> None:
    untouched_translation = KoreanTranslation(
        ko_literal="기존 번역",
        ko_natural="기존 자연 번역",
        model="existing",
        batch_id="existing_batch",
    )
    untouched = _segment("seg_0001", text="既存です", translation=untouched_translation, status="ok")
    target = _segment("seg_0002", text="こんにちは")
    manifest = PipelineManifest(
        project_config=ProjectConfig(gemma_text_batch_size=1),
        rights_audit=require_confirmed_rights(True, "test"),
        segments=[untouched, target],
    )
    mark_stage(manifest, "transcribe", "completed")
    save_manifest(tmp_project_dir, manifest)

    translated = run_translate_ko_stage(
        PipelineContext.load(tmp_project_dir),
        "mock",
        confirm_rights=True,
        only_segment_ids={"seg_0002"},
    )

    assert translated.segments[0].translation_ko == untouched_translation
    assert translated.segments[0].tts is not None
    assert translated.segments[0].rvc is not None
    assert translated.segments[0].qc is not None
    assert translated.segments[0].mix == {"included": True}
    assert translated.segments[1].translation_ko is not None
    stage_state = translated.stage_state["translate-ko"]
    assert stage_state["processed_by_only_segment_ids"] is True
    assert stage_state["target_segment_ids"] == ["seg_0002"]
    assert stage_state["skipped_non_target_segments"] == ["seg_0001"]

    summary = json.loads((tmp_project_dir / "work" / "translate_ko" / "summary.json").read_text("utf-8"))
    assert summary["processed_by_only_segment_ids"] is True
    assert summary["target_segment_ids"] == ["seg_0002"]
    assert summary["skipped_non_target_segments"] == ["seg_0001"]
