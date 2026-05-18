from __future__ import annotations

import re
from typing import Any

from asmr_dub_pipeline.schemas import Segment
from asmr_dub_pipeline.tts.scoring import segment_has_numeric_sequence

_TEXT_RE = re.compile(r"[0-9A-Za-z가-힣ぁ-んァ-ン一-龯]")


def _analysis_dict(segment: Segment, key: str) -> dict[str, Any]:
    value = segment.analysis.get(key)
    return value if isinstance(value, dict) else {}


def _selection_scores(segment: Segment) -> list[dict[str, Any]]:
    selection = _analysis_dict(segment, "tts_selection")
    scores = selection.get("scores")
    return [score for score in scores if isinstance(score, dict)] if isinstance(scores, list) else []


def _selection_hard_fail_reasons(segment: Segment) -> list[str]:
    selection = _analysis_dict(segment, "tts_selection")
    raw = selection.get("hard_fail_reasons")
    if isinstance(raw, list):
        return [str(reason) for reason in raw]
    reasons: list[str] = []
    for score in _selection_scores(segment):
        score_reasons = score.get("hard_fail_reasons")
        if isinstance(score_reasons, list):
            reasons.extend(str(reason) for reason in score_reasons)
    return list(dict.fromkeys(reasons))


def _score_backends(segment: Segment) -> set[str]:
    return {str(score.get("backend") or "") for score in _selection_scores(segment)}


def _all_scored_candidates_duration_only(segment: Segment, *, backend: str | None = None) -> bool:
    scores = _selection_scores(segment)
    if backend is not None:
        scores = [score for score in scores if str(score.get("backend") or "") == backend]
    if not scores:
        return False
    for score in scores:
        reasons = set(str(reason) for reason in score.get("hard_fail_reasons") or [])
        if reasons != {"duration_tolerance_exceeded"}:
            return False
    return True


def _meaningful_script(segment: Segment) -> bool:
    parts = []
    if segment.script is not None:
        parts.extend([segment.script.tts_text, segment.script.ja_text])
    if segment.source_script is not None:
        parts.append(segment.source_script.text)
    text = " ".join(part for part in parts if part)
    return bool(_TEXT_RE.search(text or ""))


def _asr_texture_evidence(segment: Segment) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    analysis = segment.analysis or {}
    if segment.status == "non_speech_texture":
        evidence.append({"path": "status", "value": segment.status})
    if segment.keep_original_texture:
        evidence.append({"path": "keep_original_texture", "value": True})
    for key in ("non_speech_texture", "speech_kind"):
        value = analysis.get(key)
        if value is True or str(value).strip().lower() == "texture":
            evidence.append({"path": f"analysis.{key}", "value": value})
    for key in ("asr_quality_gate", "candidate_keep_original_texture", "korean_script_non_speech_texture"):
        payload = analysis.get(key)
        if not isinstance(payload, dict):
            continue
        joined = " ".join(
            str(value)
            for value in [
                payload.get("gate"),
                payload.get("decision"),
                payload.get("action"),
                payload.get("reason"),
                *(payload.get("reasons") or []),
            ]
        ).lower()
        if payload.get("tts_blocked") is True or "texture" in joined or "asr_non_speech_texture" in joined:
            evidence.append({"path": f"analysis.{key}", "value": payload})
    for error in segment.errors:
        lowered = str(error).lower()
        if "asr_non_speech_texture" in lowered or "keep_original_texture" in lowered:
            evidence.append({"path": "errors", "value": error})
    return evidence


def _micro_absorb_evidence(segment: Segment) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    if segment.status == "absorbed":
        evidence.append({"path": "status", "value": segment.status})
    for key in ("korean_tts_absorption", "micro_segment_absorption", "micro_segment_auto_fallback"):
        payload = segment.analysis.get(key)
        if not isinstance(payload, dict):
            continue
        joined = " ".join(str(value) for value in payload.values()).lower()
        if "absorb" in joined or payload.get("absorbed_into_segment_id"):
            evidence.append({"path": f"analysis.{key}", "value": payload})
    policy = segment.analysis.get("korean_tts_micro_segment_policy")
    if isinstance(policy, dict):
        action = str(policy.get("action") or "").lower()
        if "absorb" in action or policy.get("absorbed_into_segment_id") or policy.get("target_segment_id"):
            evidence.append({"path": "analysis.korean_tts_micro_segment_policy", "value": policy})
    return evidence


def _numeric_evidence(segment: Segment) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    if segment_has_numeric_sequence(segment):
        evidence.append({"path": "segment_text", "value": "numeric_sequence"})
    for key in ("countdown_event", "numeric_phrase_renderer", "numeric_render_plan"):
        payload = segment.analysis.get(key)
        if isinstance(payload, dict):
            evidence.append({"path": f"analysis.{key}", "value": payload})
    reasons = _selection_hard_fail_reasons(segment)
    if any(reason.startswith("numeric_sequence_qc") for reason in reasons):
        evidence.append({"path": "analysis.tts_selection.hard_fail_reasons", "value": reasons})
    return evidence


def _payload(
    class_name: str,
    *,
    reasons: list[str],
    suggested_action: str,
    evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "class": class_name,
        "reasons": list(dict.fromkeys(reasons)),
        "suggested_action": suggested_action,
        "evidence": evidence,
    }


def classify_tts_hard_failed_segment(segment: Segment) -> dict[str, Any]:
    """Classify a segment whose TTS candidate selection could not pick audio."""

    absorb_evidence = _micro_absorb_evidence(segment)
    if absorb_evidence:
        return _payload(
            "micro_absorb",
            reasons=["neighbor_absorb_evidence"],
            suggested_action="absorbed",
            evidence=absorb_evidence,
        )

    texture_evidence = _asr_texture_evidence(segment)
    if segment.script is None and texture_evidence:
        return _payload(
            "missing_script_texture",
            reasons=["missing_script", "texture_evidence"],
            suggested_action="non_speech_texture",
            evidence=texture_evidence,
        )
    if texture_evidence and not _meaningful_script(segment):
        return _payload(
            "non_speech_texture",
            reasons=["texture_evidence", "no_meaningful_script"],
            suggested_action="non_speech_texture",
            evidence=texture_evidence,
        )

    numeric_evidence = _numeric_evidence(segment)
    if numeric_evidence:
        return _payload(
            "numeric_renderer_required",
            reasons=["numeric_or_countdown_segment"],
            suggested_action="numeric_renderer",
            evidence=numeric_evidence,
        )

    backends = _score_backends(segment)
    reasons = _selection_hard_fail_reasons(segment)
    if "qwen_tts" in backends and _all_scored_candidates_duration_only(segment, backend="qwen_tts"):
        return _payload(
            "qwen_duration_mismatch",
            reasons=["qwen_candidates_duration_tolerance_exceeded"],
            suggested_action="duration_rescue",
            evidence=[{"path": "analysis.tts_selection.hard_fail_reasons", "value": reasons}],
        )
    if backends and backends <= {"gpt_sovits", "mock"} and "qwen_tts" not in backends:
        return _payload(
            "gsv_only_needs_late_qwen",
            reasons=["gsv_only_all_candidates_hard_failed"],
            suggested_action="late_qwen",
            evidence=[
                {"path": "analysis.tts_selection.backends", "value": sorted(backends)},
                {"path": "analysis.tts_selection.hard_fail_reasons", "value": reasons},
            ],
        )
    return _payload(
        "genuine_manual_review",
        reasons=reasons or ["all_candidates_hard_failed"],
        suggested_action="manual_review",
        evidence=[{"path": "analysis.tts_selection.hard_fail_reasons", "value": reasons}],
    )
