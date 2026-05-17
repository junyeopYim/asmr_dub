from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from asmr_dub_pipeline.gemma.schemas import GemmaQCResult
from asmr_dub_pipeline.qc.repair_plan import duration_qc_policy
from asmr_dub_pipeline.schemas import QCMetadata

ASMR_MAX_RMS_DBFS = -18.0
ASMR_MAX_SINGLE_EDGE_SILENCE_SEC = 1.20
ASMR_MAX_TOTAL_EDGE_SILENCE_RATIO = 0.35


def score_qc(audio_metrics: dict[str, Any], gemma_qc: dict[str, Any] | None = None) -> QCMetadata:
    try:
        gemma_qc = GemmaQCResult.model_validate(gemma_qc or {}).model_dump(mode="json")
    except ValidationError as exc:
        gemma_qc = {
            "text_match_score": 0.0,
            "pronunciation_score": 0.0,
            "asmr_style_score": 0.0,
            "timing_score": 0.0,
            "repetition_detected": False,
            "omission_detected": False,
            "unsafe_or_rights_issue": True,
            "recommendation": "manual_review",
            "issues": [f"gemma_qc_validation_failed: {exc.errors()[0]['msg']}"],
        }
    issues: list[str] = list(gemma_qc.get("issues") or [])
    ratio = float(audio_metrics.get("duration_ratio") or 0.0)
    duration_policy = duration_qc_policy(audio_metrics)
    timing_score = float(gemma_qc.get("timing_score", max(0.0, 1.0 - abs(1.0 - ratio))))
    if duration_policy.get("gate") in {"too_short", "too_long"}:
        issues.append("duration_ratio_out_of_range")
    if float(audio_metrics.get("clipping_ratio") or 0.0) > 0.001:
        issues.append("clipping_detected")
    if float(audio_metrics.get("peak_dbfs") or -120.0) > -0.1:
        issues.append("peak_too_hot")
    if float(audio_metrics.get("rms_dbfs") or -120.0) > ASMR_MAX_RMS_DBFS:
        issues.append("too_loud_for_asmr")
    leading_silence = float(audio_metrics.get("leading_silence_sec") or 0.0)
    trailing_silence = float(audio_metrics.get("trailing_silence_sec") or 0.0)
    intentional_leading_silence = max(
        0.0, float(audio_metrics.get("intentional_leading_silence_sec") or 0.0)
    )
    intentional_trailing_silence = max(
        0.0, float(audio_metrics.get("intentional_trailing_silence_sec") or 0.0)
    )
    unintentional_leading_silence = max(0.0, leading_silence - intentional_leading_silence)
    unintentional_trailing_silence = max(0.0, trailing_silence - intentional_trailing_silence)
    duration_sec = max(0.0, float(audio_metrics.get("duration_sec") or 0.0))
    edge_silence = unintentional_leading_silence + unintentional_trailing_silence
    edge_ratio = edge_silence / duration_sec if duration_sec > 0.0 else 0.0
    if (
        unintentional_leading_silence > ASMR_MAX_SINGLE_EDGE_SILENCE_SEC
        or unintentional_trailing_silence > ASMR_MAX_SINGLE_EDGE_SILENCE_SEC
        or edge_ratio > ASMR_MAX_TOTAL_EDGE_SILENCE_RATIO
    ):
        issues.append("too_much_silence")
    unsafe = bool(gemma_qc.get("unsafe_or_rights_issue", False))
    repetition = bool(gemma_qc.get("repetition_detected", False))
    omission = bool(gemma_qc.get("omission_detected", False))
    text_score = float(gemma_qc.get("text_match_score", 1.0))
    pronunciation = float(gemma_qc.get("pronunciation_score", 1.0))
    style = float(gemma_qc.get("asmr_style_score", 1.0))
    score = max(0.0, min(1.0, (text_score + pronunciation + style + timing_score) / 4.0))
    recommendation = gemma_qc.get("recommendation") or "pass"
    status = "ok"
    if unsafe:
        recommendation = "manual_review"
        status = "needs_manual_review"
    elif recommendation == "manual_review":
        status = "needs_manual_review"
    elif issues or repetition or omission or recommendation == "regenerate" or score < 0.75:
        recommendation = "regenerate"
        status = "needs_regeneration"
    return QCMetadata(
        duration_ratio=ratio,
        peak_dbfs=audio_metrics.get("peak_dbfs"),
        rms_dbfs=audio_metrics.get("rms_dbfs"),
        clipping_ratio=audio_metrics.get("clipping_ratio"),
        leading_silence_sec=audio_metrics.get("leading_silence_sec"),
        trailing_silence_sec=audio_metrics.get("trailing_silence_sec"),
        text_match_score=text_score,
        pronunciation_score=pronunciation,
        asmr_style_score=style,
        timing_score=timing_score,
        repetition_detected=repetition,
        omission_detected=omission,
        unsafe_or_rights_issue=unsafe,
        duration_policy=duration_policy,
        recommendation=recommendation,
        issues=issues,
        score=score,
        status=status,
    )
