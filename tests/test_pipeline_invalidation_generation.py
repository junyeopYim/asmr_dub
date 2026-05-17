from __future__ import annotations

from asmr_dub_pipeline.pipeline.artifacts import (
    make_qc_generation_id,
    make_rvc_generation_id,
    make_script_generation_id,
    make_selected_tts_generation_id,
    verify_segment_generation_chain,
)
from asmr_dub_pipeline.pipeline.invalidation import invalidate_segment
from asmr_dub_pipeline.schemas import (
    JapaneseScript,
    PipelineManifest,
    QCMetadata,
    RVCMetadata,
    Segment,
    TTSMetadata,
)


def _segment() -> Segment:
    script = JapaneseScript(
        ja_text="こんにちは",
        tts_text="안녕하세요",
        tts_language="ko",
        source_language="ja",
        target_language="ko",
    )
    script_generation_id = make_script_generation_id(script)
    selected_tts_generation_id = make_selected_tts_generation_id(
        segment_id="seg_0001",
        candidate_id="cand_001",
        wav_path="work/tts/seg_0001_final.wav",
        input_script_generation_id=script_generation_id,
    )
    rvc_generation_id = make_rvc_generation_id(
        segment_id="seg_0001",
        output_path="work/rvc/seg_0001_final.wav",
        input_selected_tts_generation_id=selected_tts_generation_id,
        settings={"profile": "mock"},
    )
    return Segment(
        id="seg_0001",
        start=0.0,
        end=1.0,
        duration=1.0,
        audio_for_gemma="work/segments/audio/seg_0001_gemma.wav",
        audio_for_mix="work/segments/audio/seg_0001_mix.wav",
        script=script,
        status="ok",
        tts=TTSMetadata(
            backend="gpt-sovits",
            selected_candidate_path="work/tts/seg_0001_final.wav",
            input_script_generation_id=script_generation_id,
            selected_candidate_id="cand_001",
            selected_tts_generation_id=selected_tts_generation_id,
            generation_id=selected_tts_generation_id,
        ),
        rvc=RVCMetadata(
            backend="mock",
            input_path="work/tts/seg_0001_final.wav",
            output_path="work/rvc/seg_0001_final.wav",
            accepted=True,
            input_selected_tts_generation_id=selected_tts_generation_id,
            generation_id=rvc_generation_id,
        ),
        qc=QCMetadata(
            input_rvc_generation_id=rvc_generation_id,
            generation_id=make_qc_generation_id(
                segment_id="seg_0001",
                input_rvc_generation_id=rvc_generation_id,
                recommendation="pass",
                issues=[],
            ),
            recommendation="pass",
            status="ok",
        ),
        mix={"included": True},
    )


def test_invalidate_script_clears_tts_rvc_qc_and_mix() -> None:
    manifest = PipelineManifest(segments=[_segment()])

    record = invalidate_segment(manifest, "seg_0001", "korean_script", "test_script_changed")

    segment = manifest.segments[0]
    assert record["invalidated_nodes"] == ["korean_script", "tts.candidate_pool", "tts.select", "rvc", "qc"]
    assert segment.tts is None
    assert segment.rvc is None
    assert segment.qc is None
    assert segment.mix == {}
    assert segment.analysis["invalidation"][-1]["reason"] == "test_script_changed"


def test_invalidate_selected_tts_clears_rvc_qc_and_mix() -> None:
    manifest = PipelineManifest(segments=[_segment()])

    record = invalidate_segment(manifest, "seg_0001", "tts.select", "test_selected_changed")

    segment = manifest.segments[0]
    assert record["invalidated_nodes"] == ["tts.select", "rvc", "qc"]
    assert segment.tts is not None
    assert segment.rvc is None
    assert segment.qc is None
    assert segment.mix == {}


def test_invalidate_rvc_clears_qc_and_mix() -> None:
    manifest = PipelineManifest(segments=[_segment()])

    record = invalidate_segment(manifest, "seg_0001", "rvc", "test_rvc_changed")

    segment = manifest.segments[0]
    assert record["invalidated_nodes"] == ["rvc", "qc"]
    assert segment.tts is not None
    assert segment.rvc is None
    assert segment.qc is None
    assert segment.mix == {}


def test_generation_chain_verifies_selected_tts_rvc_qc_links() -> None:
    segment = _segment()

    ok = verify_segment_generation_chain(segment)
    assert ok["ok"] is True
    assert ok["issues"] == []

    assert segment.rvc is not None
    segment.rvc.input_selected_tts_generation_id = "stale-tts-generation"
    failed = verify_segment_generation_chain(segment)

    assert failed["ok"] is False
    assert "rvc_input_selected_tts_generation_mismatch" in failed["issues"]


def test_strict_generation_chain_does_not_backfill_missing_ids() -> None:
    segment = _segment()
    assert segment.rvc is not None
    assert segment.qc is not None
    segment.rvc.input_selected_tts_generation_id = None
    segment.qc.input_rvc_generation_id = None

    strict = verify_segment_generation_chain(segment, strict=True, mutate=False)

    assert strict["ok"] is False
    assert "missing_rvc_input_selected_tts_generation_id" in strict["issues"]
    assert "missing_qc_input_rvc_generation_id" in strict["issues"]
    assert segment.rvc.input_selected_tts_generation_id is None
    assert segment.qc.input_rvc_generation_id is None

    legacy = verify_segment_generation_chain(segment, strict=False, mutate=True)
    assert legacy["ok"] is True
    assert segment.rvc.input_selected_tts_generation_id == segment.tts.selected_tts_generation_id
    assert segment.qc.input_rvc_generation_id == segment.rvc.generation_id


def test_strict_generation_chain_reports_rvc_and_qc_mismatches() -> None:
    segment = _segment()
    assert segment.rvc is not None
    assert segment.qc is not None
    segment.rvc.input_selected_tts_generation_id = "tts:stale"
    segment.qc.input_rvc_generation_id = "rvc:stale"

    result = verify_segment_generation_chain(segment, strict=True, mutate=False)

    assert result["ok"] is False
    assert "rvc_input_selected_tts_generation_mismatch" in result["issues"]
    assert "qc_input_rvc_generation_mismatch" in result["issues"]
