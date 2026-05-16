from __future__ import annotations

from pathlib import Path

import pytest

from asmr_dub_pipeline.config import save_project_config
from asmr_dub_pipeline.pipeline.manifest_io import load_manifest, save_manifest
from asmr_dub_pipeline.pipeline.stages import common as common_stage
from asmr_dub_pipeline.pipeline.stages import synth_gpt_sovits as synth_stage
from asmr_dub_pipeline.pipeline.steps import translate_ko_step
from asmr_dub_pipeline.schemas import (
    JapaneseScript,
    KoreanTranslation,
    PipelineManifest,
    ProjectConfig,
    Segment,
    SourceScript,
)


def _segment_with_source(
    text: str,
    *,
    values: list[int] | None = None,
    tts_text: str = "사, 삼, 이, 일, 영",
) -> Segment:
    analysis = {}
    if values is not None:
        analysis["countdown_event"] = {
            "kind": "descending_countdown",
            "values": values,
        }
    return Segment(
        id="seg_countdown",
        start=0.0,
        end=5.0,
        duration=5.0,
        audio_for_gemma="gemma.wav",
        audio_for_mix="mix.wav",
        source_script=SourceScript(
            text=text,
            language="ja",
            backend="mock",
            start=0.0,
            end=5.0,
        ),
        script=JapaneseScript(
            literal_ja=text,
            ja_text=text,
            tts_text=tts_text,
            tts_language="ko",
            source_language="ja",
            target_language="ko",
            expected_tts_duration_sec=5.0,
            ref_style="whisper_close",
        ),
        analysis=analysis,
    )


def test_countdown_translation_route_ignores_embedded_countdown_event() -> None:
    segment = _segment_with_source(
        "4 3 2 1 絶頂します ゼロ",
        values=[4, 3, 2, 1, 0],
    )

    assert common_stage._countdown_values_for_segment(segment) is None


def test_countdown_synth_route_ignores_embedded_countdown_event() -> None:
    segment = _segment_with_source(
        "4 3 2 1 絶頂します ゼロ",
        values=[4, 3, 2, 1, 0],
    )

    assert synth_stage._source_countdown_values(segment) is None


def test_pure_countdown_routes_remain_enabled_with_event() -> None:
    segment = _segment_with_source("5 4 3 2 1 0", values=[5, 4, 3, 2, 1, 0])

    assert common_stage._countdown_values_for_segment(segment) == [5, 4, 3, 2, 1, 0]
    assert synth_stage._source_countdown_values(segment) == [5, 4, 3, 2, 1, 0]


@pytest.mark.parametrize("initial_status", ["scripted", "needs_manual_review"])
def test_translate_ko_repairs_embedded_countdown_deterministic_translation(
    tmp_path: Path,
    initial_status: str,
) -> None:
    save_project_config(ProjectConfig(project_name="countdown-repair"), tmp_path / "pipeline.yaml")
    segment = _segment_with_source(
        "4 3 2 1 絶頂します ゼロ",
        values=[4, 3, 2, 1, 0],
    )
    segment.status = initial_status
    segment.translation_ko = KoreanTranslation(
        ko_literal="사, 삼, 이, 일, 영",
        ko_natural="사, 삼, 이, 일, 영",
        notes=["deterministic_countdown_event"],
        confidence=1.0,
        model="deterministic:countdown-event",
        batch_id="countdown_seg_countdown",
    )
    segment.errors = [
        "RVC requires segment.tts.selected_candidate_path from synth.",
        "Countdown phrase slice fallback failed.",
    ]
    manifest = PipelineManifest(segments=[segment])
    manifest.stage_state["transcribe"] = {"status": "completed"}
    save_manifest(tmp_path, manifest)

    translate_ko_step(tmp_path, gemma_text_backend="mock", confirm_rights=True)

    repaired = load_manifest(tmp_path).segments[0]
    assert repaired.status == "transcribed"
    assert repaired.translation_ko is not None
    assert repaired.translation_ko.model == "mock"
    assert repaired.translation_ko.ko_natural == "자연 번역: 부드럽게 속삭여 드릴게요."
    assert repaired.script is None
    assert repaired.tts is None
    assert repaired.rvc is None
    assert repaired.qc is None
    assert repaired.errors == []
    assert repaired.analysis["countdown_event"]["kind"] == "embedded_countdown"
    assert repaired.analysis["countdown_event"]["synth_eligible"] is False
    assert repaired.analysis["countdown_event"]["deterministic_translation_eligible"] is False
    assert repaired.analysis["embedded_countdown_translation_repair"]["action"] == (
        "clear_translation_and_downstream_state"
    )


def test_translate_ko_keeps_pure_countdown_deterministic_translation(tmp_path: Path) -> None:
    save_project_config(ProjectConfig(project_name="countdown-pure"), tmp_path / "pipeline.yaml")
    segment = _segment_with_source("5 4 3 2 1 0", values=[5, 4, 3, 2, 1, 0])
    segment.status = "scripted"
    segment.translation_ko = KoreanTranslation(
        ko_literal="오, 사, 삼, 이, 일, 영",
        ko_natural="오, 사, 삼, 이, 일, 영",
        notes=["deterministic_countdown_event"],
        confidence=1.0,
        model="deterministic:countdown-event",
        batch_id="countdown_seg_countdown",
    )
    manifest = PipelineManifest(segments=[segment])
    manifest.stage_state["transcribe"] = {"status": "completed"}
    save_manifest(tmp_path, manifest)

    translate_ko_step(tmp_path, gemma_text_backend="mock", confirm_rights=True)

    translated = load_manifest(tmp_path).segments[0]
    assert translated.status == "transcribed"
    assert translated.translation_ko is not None
    assert translated.translation_ko.model == "deterministic:countdown-event"
    assert translated.translation_ko.ko_natural == "오, 사, 삼, 이, 일, 영"


def test_translate_ko_keeps_event_only_countdown_deterministic_translation(tmp_path: Path) -> None:
    save_project_config(ProjectConfig(project_name="countdown-event-only"), tmp_path / "pipeline.yaml")
    segment = _segment_with_source("5 4 3 2 1 0", values=[5, 4, 3, 2, 1, 0])
    segment.source_script = None
    segment.status = "scripted"
    segment.translation_ko = KoreanTranslation(
        ko_literal="오, 사, 삼, 이, 일, 영",
        ko_natural="오, 사, 삼, 이, 일, 영",
        notes=["deterministic_countdown_event"],
        confidence=1.0,
        model="deterministic:countdown-event",
        batch_id="countdown_seg_countdown",
    )
    manifest = PipelineManifest(segments=[segment])
    manifest.stage_state["transcribe"] = {"status": "completed"}
    save_manifest(tmp_path, manifest)

    translate_ko_step(tmp_path, gemma_text_backend="mock", confirm_rights=True)

    translated = load_manifest(tmp_path).segments[0]
    assert translated.status == "transcribed"
    assert translated.translation_ko is not None
    assert translated.translation_ko.model == "deterministic:countdown-event"
    assert translated.translation_ko.ko_natural == "오, 사, 삼, 이, 일, 영"
    assert "embedded_countdown_translation_repair" not in translated.analysis
