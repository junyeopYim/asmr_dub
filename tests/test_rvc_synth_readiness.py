from __future__ import annotations

from asmr_dub_pipeline.pipeline.state import mark_stage
from asmr_dub_pipeline.schemas import PipelineManifest, Segment, TTSMetadata


def _segment(segment_id: str, status: str, selected_tts: str | None = None) -> Segment:
    return Segment(
        id=segment_id,
        start=0.0,
        end=1.0,
        duration=1.0,
        audio_for_gemma=f"work/segments/audio/{segment_id}_gemma.wav",
        audio_for_mix=f"work/segments/audio/{segment_id}_mix.wav",
        status=status,
        tts=TTSMetadata(selected_candidate_path=selected_tts) if selected_tts else None,
    )


def test_synth_readiness_completed_status_is_ready() -> None:
    from asmr_dub_pipeline.pipeline.stage_readiness import synth_ready_for_downstream

    manifest = PipelineManifest(
        segments=[_segment("seg_0001", "synthesized", "work/tts/seg_0001_final.wav")]
    )
    mark_stage(manifest, "synth", "completed")

    readiness = synth_ready_for_downstream(manifest)

    assert readiness["ready"] is True
    assert readiness["synth_status"] == "completed"
    assert readiness["selected_segment_count"] == 1
    assert readiness["missing_selected_tts_count"] == 0
    assert readiness["blocking_segments"] == []


def test_synth_readiness_partial_status_is_ready_when_hard_failures_are_non_blocking() -> None:
    from asmr_dub_pipeline.pipeline.stage_readiness import synth_ready_for_downstream

    manifest = PipelineManifest(
        segments=[
            _segment("seg_0001", "synthesized", "work/tts/seg_0001_final.wav"),
            _segment("seg_0002", "needs_manual_review"),
        ]
    )
    mark_stage(
        manifest,
        "synth",
        "completed_with_hard_failed_candidates",
        selected_segments=["seg_0001"],
        hard_failed_segments=["seg_0002"],
    )

    readiness = synth_ready_for_downstream(manifest)

    assert readiness["ready"] is True
    assert readiness["synth_status"] == "completed_with_hard_failed_candidates"
    assert readiness["hard_failed_segments"] == ["seg_0002"]
    assert readiness["non_blocking_segments"] == ["seg_0002"]
    assert readiness["blocking_segments"] == []
    assert readiness["selected_segment_count"] == 1
    assert readiness["missing_selected_tts_count"] == 0


def test_synth_readiness_partial_status_blocks_missing_selected_tts() -> None:
    from asmr_dub_pipeline.pipeline.stage_readiness import synth_ready_for_downstream

    manifest = PipelineManifest(segments=[_segment("seg_0001", "synthesized")])
    mark_stage(
        manifest,
        "synth",
        "completed_with_hard_failed_candidates",
        selected_segments=[],
        hard_failed_segments=["seg_0001"],
    )

    readiness = synth_ready_for_downstream(manifest)

    assert readiness["ready"] is False
    assert readiness["synth_status"] == "completed_with_hard_failed_candidates"
    assert readiness["blocking_segments"] == ["seg_0001"]
    assert readiness["missing_selected_tts_count"] == 1


def test_synth_readiness_partial_status_requires_selection_metadata() -> None:
    from asmr_dub_pipeline.pipeline.stage_readiness import synth_ready_for_downstream

    manifest = PipelineManifest(
        segments=[
            _segment("seg_0001", "synthesized", "work/tts/seg_0001_final.wav"),
            _segment("seg_0002", "needs_manual_review"),
        ]
    )
    mark_stage(manifest, "synth", "completed_with_hard_failed_candidates")

    readiness = synth_ready_for_downstream(manifest)

    assert readiness["ready"] is False
    assert "<missing:selected_segments>" in readiness["blocking_segments"]
    assert "<missing:hard_failed_segments>" in readiness["blocking_segments"]


def test_needs_regeneration_remains_blocking_for_synth_readiness() -> None:
    from asmr_dub_pipeline.pipeline.stage_readiness import (
        NON_BLOCKING_SYNTH_SEGMENT_STATUSES,
        synth_ready_for_downstream,
    )

    manifest = PipelineManifest(
        segments=[
            _segment("seg_0001", "needs_regeneration"),
        ]
    )
    mark_stage(
        manifest,
        "synth",
        "completed_with_hard_failed_candidates",
        selected_segments=[],
        hard_failed_segments=[],
        late_qwen_scheduled_segments=["seg_0001"],
    )

    readiness = synth_ready_for_downstream(manifest)

    assert "needs_regeneration" not in NON_BLOCKING_SYNTH_SEGMENT_STATUSES
    assert readiness["ready"] is False
    assert readiness["blocking_segments"] == ["seg_0001"]
    assert readiness["missing_selected_tts_segments"] == ["seg_0001"]
