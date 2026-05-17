from __future__ import annotations

import re
from typing import Any

from asmr_dub_pipeline.schemas import ProjectConfig, QCMetadata, Segment

_HANGUL_RE = re.compile(r"[가-힣]")
_NON_TEXT_RE = re.compile(r"^[\s.,!?;:，。！？、…·~\-_'\"“”‘’(){}\[\]<>]*$")
_TEXTURE_MARKER_RE = re.compile(
    r"(?:으+|아+|어+|오+|우+|이+|하아+|후우+|흐으+|스으+|스르르|톡|탁|훅|쿵)",
    re.IGNORECASE,
)
_SAFETY_ISSUE_MARKERS = (
    "unsafe",
    "rights",
    "consent",
    "policy",
    "copyright",
    "unauthorized",
)
_TERMINAL_SKIP_STATUSES = {"absorbed", "no_speech_detected", "non_speech_texture"}


def _cfg_value(cfg: Any | None, name: str, default: Any) -> Any:
    if cfg is None:
        cfg = ProjectConfig()
    if hasattr(cfg, name):
        return getattr(cfg, name)
    if hasattr(cfg, "gsv") and hasattr(cfg.gsv, name):
        return getattr(cfg.gsv, name)
    prefixed = f"gsv_{name}"
    if hasattr(cfg, prefixed):
        return getattr(cfg, prefixed)
    return default


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_dict(value: Any | None) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")  # type: ignore[attr-defined]
    return {}


def duration_qc_policy(
    audio_metrics: dict[str, Any] | None = None,
    *,
    source_duration_sec: float | None = None,
    target_duration_sec: float | None = None,
    tts_duration_sec: float | None = None,
    duration_ratio: float | None = None,
) -> dict[str, Any]:
    """Classify duration fit with absolute tolerance for very short segments."""

    metrics = dict(audio_metrics or {})
    source_duration = _as_float(
        source_duration_sec
        if source_duration_sec is not None
        else target_duration_sec
        if target_duration_sec is not None
        else metrics.get("source_duration_sec")
        if metrics.get("source_duration_sec") is not None
        else metrics.get("target_duration_sec"),
    )
    tts_duration = _as_float(
        tts_duration_sec if tts_duration_sec is not None else metrics.get("tts_duration_sec", metrics.get("duration_sec"))
    )
    ratio = _as_float(
        duration_ratio if duration_ratio is not None else metrics.get("duration_ratio"),
        default=0.0,
    )
    if source_duration <= 0.0 and tts_duration > 0.0 and ratio > 0.0:
        source_duration = tts_duration / ratio
    if ratio <= 0.0 and source_duration > 0.0 and tts_duration > 0.0:
        ratio = tts_duration / source_duration

    if source_duration <= 0.0 or tts_duration <= 0.0:
        return {
            "gate": "unknown",
            "action": "manual_review",
            "mode": "unavailable",
            "source_duration_sec": round(source_duration, 6),
            "tts_duration_sec": round(tts_duration, 6),
            "duration_ratio": round(ratio, 6) if ratio > 0.0 else None,
        }

    tolerance: float | None = None
    if source_duration < 0.7:
        tolerance = 0.22
    elif source_duration < 1.2:
        tolerance = 0.30
    elif source_duration < 2.0:
        tolerance = 0.40

    if tolerance is not None:
        delta = tts_duration - source_duration
        gate = "pass" if abs(delta) <= tolerance else "too_long" if delta > 0.0 else "too_short"
        return {
            "gate": gate,
            "action": "none" if gate == "pass" else "regenerate_tts",
            "mode": "absolute",
            "absolute_tolerance_sec": tolerance,
            "delta_sec": round(delta, 6),
            "source_duration_sec": round(source_duration, 6),
            "tts_duration_sec": round(tts_duration, 6),
            "duration_ratio": round(ratio, 6),
        }

    min_ratio = 0.75
    max_ratio = 1.35
    gate = "pass" if min_ratio <= ratio <= max_ratio else "too_long" if ratio > max_ratio else "too_short"
    return {
        "gate": gate,
        "action": "none" if gate == "pass" else "regenerate_tts",
        "mode": "ratio",
        "min_ratio": min_ratio,
        "max_ratio": max_ratio,
        "source_duration_sec": round(source_duration, 6),
        "tts_duration_sec": round(tts_duration, 6),
        "duration_ratio": round(ratio, 6),
    }


def _hangul_syllable_count(text: str) -> int:
    return len(_HANGUL_RE.findall(text or ""))


def _script_text(segment: Segment) -> str:
    return segment.script.tts_text if segment.script is not None else ""


def _source_text(segment: Segment) -> str:
    if segment.source_script is not None:
        return segment.source_script.text
    return segment.script.ja_text if segment.script is not None else ""


def _has_countdown_event(segment: Segment) -> bool:
    return isinstance(segment.analysis.get("countdown_event"), dict)


def is_micro_segment(segment: Segment, cfg: Any | None = None) -> bool:
    if not bool(_cfg_value(cfg, "micro_segment_enabled", True)):
        return False
    if _has_countdown_event(segment):
        return False
    max_sec = _as_float(_cfg_value(cfg, "micro_segment_max_sec", 1.2), 1.2)
    if segment.duration > max_sec:
        return False
    max_hangul = int(_cfg_value(cfg, "micro_segment_max_hangul_syllables", 4) or 0)
    text = _script_text(segment)
    return not (max_hangul > 0 and text and _hangul_syllable_count(text) > max_hangul)


def is_texture_like_micro_segment(segment: Segment, cfg: Any | None = None) -> bool:
    if _has_countdown_event(segment):
        return False
    if not bool(_cfg_value(cfg, "micro_segment_keep_original_texture_enabled", True)):
        return False
    texture_max_sec = _as_float(_cfg_value(cfg, "micro_segment_texture_max_sec", 0.7), 0.7)
    if segment.duration > texture_max_sec:
        return False
    analysis = segment.analysis or {}
    if segment.status == "non_speech_texture":
        return True
    if analysis.get("non_speech_texture") is True or str(analysis.get("speech_kind") or "") == "texture":
        return True
    combined = f"{_source_text(segment)} {_script_text(segment)}".strip()
    if combined and _NON_TEXT_RE.fullmatch(combined):
        return True
    return bool(combined and _TEXTURE_MARKER_RE.search(combined) and _hangul_syllable_count(_script_text(segment)) <= 4)


def _fallback_tts_action(cfg: Any | None) -> str:
    backend = str(_cfg_value(cfg, "micro_segment_fallback_backend", "qwen")).strip().lower().replace("-", "_")
    if backend == "qwen":
        return "fallback_tts_qwen"
    if backend == "keep_original":
        return "keep_original_texture"
    return "regenerate_tts"


def _plan(
    *,
    action: str,
    root_cause: str,
    terminal_manual: bool = False,
    severity: str = "recoverable",
    confidence: float = 0.8,
    issues: list[str] | None = None,
    source: str = "ko_qc_repair_plan",
    **extra: Any,
) -> dict[str, Any]:
    payload = {
        "action": action,
        "root_cause": root_cause,
        "severity": severity,
        "confidence": round(max(0.0, min(float(confidence), 1.0)), 6),
        "terminal_manual": terminal_manual,
        "issues": list(dict.fromkeys(issues or [])),
        "source": source,
    }
    payload.update(extra)
    return payload


def _unsafe_or_rights_issue(qc: QCMetadata | None, issues: list[str]) -> bool:
    if qc is not None and qc.unsafe_or_rights_issue:
        return True
    lowered = [issue.lower() for issue in issues]
    return any(any(marker in issue for marker in _SAFETY_ISSUE_MARKERS) for issue in lowered)


def build_ko_qc_repair_plan(
    segment: Segment,
    audio_metrics: dict[str, Any] | None = None,
    gemma_result: dict[str, Any] | None = None,
    qc: QCMetadata | None = None,
    cfg: Any | None = None,
) -> dict[str, Any]:
    qc = qc or segment.qc
    qc_payload = _as_dict(qc)
    gemma_payload = _as_dict(gemma_result)
    issues = [
        *[str(issue) for issue in qc_payload.get("issues") or []],
        *[str(issue) for issue in gemma_payload.get("issues") or []],
    ]

    if _unsafe_or_rights_issue(qc, issues):
        return _plan(
            action="manual_review",
            root_cause="unsafe_or_rights_issue",
            terminal_manual=True,
            severity="terminal",
            confidence=1.0,
            issues=issues or ["unsafe_or_rights_issue"],
        )

    if is_texture_like_micro_segment(segment, cfg):
        return _plan(
            action="keep_original_texture",
            root_cause="texture_like_micro_segment",
            severity="low",
            confidence=0.9,
            issues=issues,
            route="micro_texture",
        )

    if segment.status in _TERMINAL_SKIP_STATUSES:
        return _plan(
            action="manual_review",
            root_cause=f"terminal_status:{segment.status}",
            terminal_manual=True,
            severity="terminal",
            confidence=0.9,
            issues=issues,
        )

    if segment.source_script is None:
        return _plan(
            action="repair_asr_then_downstream",
            root_cause="missing_source_script",
            confidence=0.82,
            issues=issues or ["missing_source_script"],
        )

    if segment.translation_ko is None and segment.script is None:
        return _plan(
            action="repair_translation_then_tts",
            root_cause="missing_korean_translation",
            confidence=0.82,
            issues=issues or ["missing_korean_translation"],
        )

    if segment.script is None:
        return _plan(
            action="rewrite_script_then_tts",
            root_cause="missing_korean_script",
            confidence=0.8,
            issues=issues or ["missing_korean_script"],
        )

    if not segment.tts or not segment.tts.selected_candidate_path:
        action = _fallback_tts_action(cfg) if is_micro_segment(segment, cfg) else "regenerate_tts"
        return _plan(
            action=action,
            root_cause="missing_selected_tts",
            confidence=0.88,
            issues=issues or ["missing_selected_tts"],
            route="micro_fallback" if is_micro_segment(segment, cfg) else "tts_regeneration",
        )

    metrics = dict(audio_metrics or {})
    if qc is not None and qc.duration_ratio is not None:
        metrics.setdefault("duration_ratio", qc.duration_ratio)
    metrics.setdefault("source_duration_sec", segment.duration)
    policy = duration_qc_policy(metrics) if metrics else {}
    duration_gate = str(policy.get("gate") or "")
    effective_issues = list(issues)
    if duration_gate in {"too_short", "too_long"} and "duration_ratio_out_of_range" not in effective_issues:
        effective_issues.append("duration_ratio_out_of_range")
    if duration_gate in {"too_short", "too_long"} or any(
        issue in effective_issues
        for issue in ("duration_ratio_out_of_range", "too_much_silence", "clipping_detected", "peak_too_hot")
    ):
        return _plan(
            action="regenerate_tts",
            root_cause="audio_duration_or_quality",
            confidence=0.84,
            issues=effective_issues,
            duration_policy=policy,
        )

    if qc is not None and (qc.repetition_detected or qc.omission_detected):
        return _plan(
            action="rewrite_script_then_tts",
            root_cause="text_repetition_or_omission",
            confidence=0.76,
            issues=effective_issues,
        )

    recommendation = str(qc_payload.get("recommendation") or "pass")
    status = str(qc_payload.get("status") or segment.status)
    if recommendation == "pass" and status == "ok" and not effective_issues:
        return _plan(
            action="pass",
            root_cause="qc_pass",
            terminal_manual=False,
            severity="info",
            confidence=1.0,
            issues=[],
            duration_policy=policy,
        )

    if recommendation == "regenerate" or status == "needs_regeneration":
        return _plan(
            action="regenerate_tts",
            root_cause="qc_regenerate",
            confidence=0.8,
            issues=effective_issues,
            duration_policy=policy,
        )

    if recommendation == "manual_review" or status == "needs_manual_review":
        action = _fallback_tts_action(cfg) if is_micro_segment(segment, cfg) else "regenerate_tts"
        return _plan(
            action=action,
            root_cause="recoverable_manual_review",
            confidence=0.68,
            issues=effective_issues or ["manual_review"],
            route="micro_tts" if is_micro_segment(segment, cfg) else "tts_regeneration",
            duration_policy=policy,
        )

    return _plan(
        action="pass",
        root_cause="no_repair_needed",
        terminal_manual=False,
        severity="info",
        confidence=0.8,
        issues=effective_issues,
        duration_policy=policy,
    )


def plan_is_repairable(plan: Any) -> bool:
    payload = _as_dict(plan)
    if not payload:
        return False
    action = str(payload.get("action") or "")
    if action in {"", "pass", "none", "manual_review", "terminal_manual_review"}:
        return False
    return not bool(payload.get("terminal_manual"))
