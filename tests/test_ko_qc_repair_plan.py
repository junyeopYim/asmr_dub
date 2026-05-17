from __future__ import annotations

from typing import Any

from asmr_dub_pipeline.schemas import JapaneseScript, QCMetadata, Segment, SourceScript, TTSMetadata


def _value(obj: object, key: str) -> Any:
    if isinstance(obj, dict):
        return obj[key]
    if hasattr(obj, "model_dump"):
        dumped = obj.model_dump(mode="json")  # type: ignore[attr-defined]
        if key in dumped:
            return dumped[key]
    return getattr(obj, key)


def _ko_segment(
    *,
    segment_id: str = "seg_0001",
    duration: float = 1.0,
    source_text: str = "こんにちは",
    tts_text: str = "안녕하세요.",
    status: str = "rvc_converted",
    selected_tts: str | None = "work/tts/seg_0001_final.wav",
    qc: QCMetadata | None = None,
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
            text=source_text,
            language="ja",
            backend="mock",
            start=0.0,
            end=duration,
        ),
        script=JapaneseScript(
            ja_text=source_text,
            tts_text=tts_text,
            tts_language="ko",
            source_language="ja",
            target_language="ko",
            expected_tts_duration_sec=duration,
        ),
        tts=TTSMetadata(
            backend="gpt-sovits",
            selected_candidate_path=selected_tts,
            target_language="ko",
        )
        if selected_tts is not None
        else None,
        qc=qc,
    )


def test_missing_selected_tts_with_script_is_recoverable_not_terminal_manual() -> None:
    from asmr_dub_pipeline.qc.repair_plan import build_ko_qc_repair_plan

    segment = _ko_segment(selected_tts=None, qc=QCMetadata(recommendation="manual_review", status="needs_manual_review"))

    plan = build_ko_qc_repair_plan(segment)

    assert _value(plan, "terminal_manual") is False
    assert _value(plan, "action") in {"fallback_tts_qwen", "regenerate_tts"}


def test_unsafe_or_rights_issue_is_terminal_manual() -> None:
    from asmr_dub_pipeline.qc.repair_plan import build_ko_qc_repair_plan

    segment = _ko_segment(
        qc=QCMetadata(
            unsafe_or_rights_issue=True,
            recommendation="manual_review",
            status="needs_manual_review",
            issues=["unsafe_or_rights_issue"],
        )
    )

    plan = build_ko_qc_repair_plan(segment)

    assert _value(plan, "terminal_manual") is True
    assert _value(plan, "action") == "manual_review"


def test_normal_duration_issue_regenerates_tts() -> None:
    from asmr_dub_pipeline.qc.repair_plan import build_ko_qc_repair_plan

    segment = _ko_segment(
        duration=4.0,
        qc=QCMetadata(
            duration_ratio=1.8,
            recommendation="regenerate",
            status="needs_regeneration",
            issues=["duration_ratio_out_of_range"],
        ),
    )

    plan = build_ko_qc_repair_plan(segment)

    assert _value(plan, "terminal_manual") is False
    assert _value(plan, "action") == "regenerate_tts"


def test_short_absolute_duration_mismatch_stays_within_tolerance() -> None:
    from asmr_dub_pipeline.qc.repair_plan import duration_qc_policy

    policy = duration_qc_policy(
        source_duration_sec=0.35,
        tts_duration_sec=0.50,
        duration_ratio=1.43,
    )

    assert _value(policy, "gate") == "pass"
    assert _value(policy, "action") == "none"


def test_texture_like_micro_segment_keeps_original_texture() -> None:
    from asmr_dub_pipeline.qc.repair_plan import (
        build_ko_qc_repair_plan,
        is_texture_like_micro_segment,
    )

    segment = _ko_segment(
        duration=0.28,
        source_text="…",
        tts_text="...",
        status="non_speech_texture",
        selected_tts=None,
        qc=QCMetadata(recommendation="manual_review", status="needs_manual_review"),
        analysis={"speech_kind": "texture", "non_speech_texture": True},
    )

    assert is_texture_like_micro_segment(segment) is True
    plan = build_ko_qc_repair_plan(segment)

    assert _value(plan, "terminal_manual") is False
    assert _value(plan, "action") == "keep_original_texture"


def test_short_korean_speech_micro_uses_fallback_route() -> None:
    from asmr_dub_pipeline.qc.repair_plan import build_ko_qc_repair_plan, is_micro_segment

    segment = _ko_segment(
        duration=0.42,
        source_text="はい",
        tts_text="네.",
        selected_tts=None,
        qc=QCMetadata(recommendation="manual_review", status="needs_manual_review"),
    )

    assert is_micro_segment(segment) is True
    plan = build_ko_qc_repair_plan(segment)

    assert _value(plan, "terminal_manual") is False
    assert _value(plan, "action") in {"fallback_tts_qwen", "regenerate_tts"}
    if isinstance(plan, dict) and "route" in plan:
        assert plan["route"] in {"micro_fallback", "micro_tts"}


def test_countdown_micro_is_not_treated_as_texture() -> None:
    from asmr_dub_pipeline.qc.repair_plan import (
        build_ko_qc_repair_plan,
        is_texture_like_micro_segment,
    )

    segment = _ko_segment(
        duration=0.55,
        source_text="3 2 1",
        tts_text="삼, 이, 일.",
        selected_tts=None,
        qc=QCMetadata(recommendation="manual_review", status="needs_manual_review"),
        analysis={"countdown_event": {"values": [3, 2, 1]}},
    )

    assert is_texture_like_micro_segment(segment) is False
    plan = build_ko_qc_repair_plan(segment)

    assert _value(plan, "action") != "keep_original_texture"
