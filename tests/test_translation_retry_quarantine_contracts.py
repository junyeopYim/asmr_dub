from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from asmr_dub_pipeline.gemma.text_translate import GemmaTextTranslationError
from asmr_dub_pipeline.pipeline import steps
from asmr_dub_pipeline.pipeline.manifest_io import save_manifest
from asmr_dub_pipeline.pipeline.state import mark_stage
from asmr_dub_pipeline.rights import require_confirmed_rights
from asmr_dub_pipeline.schemas import KoreanTranslation, PipelineManifest, Segment, SourceScript


def _segment(segment_id: str, text: str) -> Segment:
    return Segment(
        id=segment_id,
        start=0.0,
        end=1.0,
        duration=1.0,
        audio_for_gemma=f"work/segments/audio/{segment_id}_gemma.wav",
        audio_for_mix=f"work/segments/audio/{segment_id}_mix.wav",
        status="transcribed",
        source_script=SourceScript(text=text, language="ja", backend="mock", start=0.0, end=1.0),
    )


def test_translate_ko_refusal_retries_segment_and_completes_on_success(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    save_manifest(
        tmp_project_dir,
        PipelineManifest(
            rights_audit=require_confirmed_rights(True, "test"),
            segments=[_segment("seg_0001", "こんにちは")],
        ),
    )
    manifest = steps.PipelineContext.load(tmp_project_dir).manifest
    mark_stage(manifest, "transcribe", "completed")
    save_manifest(tmp_project_dir, manifest)
    calls: list[str] = []

    class RefusalThenSuccessClient:
        def __init__(self, model: str = "mock") -> None:
            self.model = model

        def translate_batch(
            self,
            segments: Sequence[Segment],
            batch_id: str,
            context_segments: Sequence[Segment] | None = None,
        ) -> dict[str, KoreanTranslation]:
            calls.append(batch_id)
            if len(calls) == 1:
                raise GemmaTextTranslationError("model refusal: I cannot comply with this request")
            return {
                segments[0].id: KoreanTranslation(
                    ko_literal="안녕하세요.",
                    ko_natural="안녕하세요.",
                    model=self.model,
                    batch_id=batch_id,
                )
            }

    monkeypatch.setattr(steps, "MockTranslationClient", RefusalThenSuccessClient)

    translated = steps.translate_ko_step(tmp_project_dir, gemma_text_backend="mock", confirm_rights=True)

    segment = translated.segments[0]
    assert segment.translation_ko is not None
    assert segment.status == "transcribed"
    attempts = segment.analysis["translation_safety"]["attempts"]
    assert [attempt["mode"] for attempt in attempts] == ["same_model_low_temp"]
    assert attempts[0]["status"] == "success"
    assert translated.stage_state["translate-ko"]["status"] == "completed"


def test_translate_ko_refusal_exhaustion_quarantines_segment_without_fatal_stage_failure(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    save_manifest(
        tmp_project_dir,
        PipelineManifest(
            rights_audit=require_confirmed_rights(True, "test"),
            segments=[_segment("seg_0001", "こんにちは")],
        ),
    )
    manifest = steps.PipelineContext.load(tmp_project_dir).manifest
    mark_stage(manifest, "transcribe", "completed")
    save_manifest(tmp_project_dir, manifest)

    class AlwaysRefusesClient:
        def __init__(self, model: str = "mock") -> None:
            self.model = model

        def translate_batch(
            self,
            segments: Sequence[Segment],
            batch_id: str,
            context_segments: Sequence[Segment] | None = None,
        ) -> dict[str, KoreanTranslation]:
            raise GemmaTextTranslationError("content_filter refusal: cannot translate")

    monkeypatch.setattr(steps, "MockTranslationClient", AlwaysRefusesClient)

    translated = steps.translate_ko_step(tmp_project_dir, gemma_text_backend="mock", confirm_rights=True)

    segment = translated.segments[0]
    assert segment.translation_ko is None
    assert segment.status == "quarantined"
    assert segment.analysis["translate_ko_quarantine"]["kind"] == "provider_safety_block"
    assert segment.analysis["translate_ko_quarantine"]["recoverable"] is True
    assert translated.stage_state["translate-ko"]["status"] == "completed_with_quarantined_segments"
    quarantine_path = Path(translated.artifacts["translation_quarantine"])
    assert json.loads(quarantine_path.read_text("utf-8"))["segments"][0]["segment_id"] == "seg_0001"
