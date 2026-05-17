from __future__ import annotations

import hashlib
import json
from typing import Any

from asmr_dub_pipeline.schemas import ProjectConfig, Segment
from asmr_dub_pipeline.tts.scoring import segment_has_numeric_sequence
from asmr_dub_pipeline.tts.types import TTSBackendName, TTSRoute


def _normalize_backend(value: str) -> str:
    return value.strip().lower().replace("-", "_")


def _route_id(segment_id: str, backends: list[str], reason_codes: list[str]) -> str:
    payload = {"segment_id": segment_id, "backends": backends, "reason_codes": reason_codes}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return f"tts-route:{digest}"


def _analysis_dict(segment: Segment, key: str) -> dict[str, Any]:
    value = segment.analysis.get(key)
    return value if isinstance(value, dict) else {}


def _has_previous_gsv_pronunciation_failure(segment: Segment) -> bool:
    repair_plan = _analysis_dict(segment, "ko_qc_repair_plan")
    root = " ".join(str(repair_plan.get(key) or "") for key in ("root_cause", "action", "route"))
    if "pronunciation" in root and ("gpt_sovits" in root or "gsv" in root):
        return True
    if segment.qc and any("pronunciation" in issue for issue in segment.qc.issues):
        return True
    summary = segment.tts.retry_summary if segment.tts else {}
    return str(summary.get("selected_pronunciation_gate") or "").lower() == "fail"


def _has_previous_gsv_duration_failure(segment: Segment) -> bool:
    repair_plan = _analysis_dict(segment, "ko_qc_repair_plan")
    root = " ".join(str(repair_plan.get(key) or "") for key in ("root_cause", "action", "route"))
    if "duration" in root and ("gpt_sovits" in root or "gsv" in root):
        return True
    if segment.qc and any("duration" in issue for issue in segment.qc.issues):
        return True
    summary = segment.tts.retry_summary if segment.tts else {}
    return str(summary.get("selected_duration_gate") or "").lower() in {"too_short", "too_long"}


def _auto_repair_requests_qwen(segment: Segment) -> bool:
    repair_plan = _analysis_dict(segment, "ko_qc_repair_plan")
    auto_repair = _analysis_dict(segment, "auto_repair")
    return str(repair_plan.get("action") or auto_repair.get("action") or "") == "fallback_tts_qwen"


def route_segment_tts(
    segment: Segment,
    cfg: ProjectConfig,
    *,
    requested_backend: str = "auto",
    candidate_budget: int | None = None,
) -> TTSRoute:
    """Return the backends that should generate candidates for one segment."""

    normalized = _normalize_backend(requested_backend)
    if normalized in {"gpt_sovits", "gsv"}:
        backends: list[TTSBackendName] = ["gpt_sovits"]
        reason_codes = ["requested_gpt_sovits"]
    elif normalized in {"qwen", "qwen_tts"}:
        backends = ["qwen_tts"]
        reason_codes = ["requested_qwen_tts"]
    elif normalized == "mock":
        backends = ["mock"]
        reason_codes = ["requested_mock"]
    elif normalized in {"auto", "pool", "candidate_pool"}:
        backends = [
            _normalize_backend(backend)  # type: ignore[list-item]
            for backend in cfg.tts.router.default_backends
        ]
        reason_codes = ["default_gpt_sovits"] if "gpt_sovits" in backends else ["default_route"]
        if not backends:
            backends = ["gpt_sovits"]
            reason_codes = ["default_gpt_sovits"]
        if cfg.tts.router.qwen_parallel_enabled:
            qwen_reasons: list[str] = []
            if (
                cfg.tts.router.qwen_for_micro_segments
                and segment.duration <= cfg.tts.router.micro_segment_max_sec
            ):
                qwen_reasons.append("micro_segment")
            if cfg.tts.router.qwen_for_numeric_sequences and segment_has_numeric_sequence(segment):
                qwen_reasons.append("numeric_sequence")
            if (
                cfg.tts.router.qwen_for_gsv_pronunciation_failures
                and _has_previous_gsv_pronunciation_failure(segment)
            ):
                qwen_reasons.append("previous_gsv_pronunciation_failure")
            if (
                cfg.tts.router.qwen_for_gsv_duration_failures
                and _has_previous_gsv_duration_failure(segment)
            ):
                qwen_reasons.append("previous_gsv_duration_failure")
            if cfg.tts.router.qwen_for_auto_repair_qwen_action and _auto_repair_requests_qwen(segment):
                qwen_reasons.append("auto_repair_qwen_action")
            if qwen_reasons and "qwen_tts" not in backends:
                backends.append("qwen_tts")
            reason_codes.extend(qwen_reasons)
    else:
        raise ValueError("tts_backend must be one of: gpt-sovits, qwen, auto, pool, candidate-pool, mock")

    deduped_backends: list[TTSBackendName] = []
    for backend in backends:
        if backend == "gpt-sovits":  # type: ignore[comparison-overlap]
            backend = "gpt_sovits"  # type: ignore[assignment]
        if backend == "qwen-tts":  # type: ignore[comparison-overlap]
            backend = "qwen_tts"  # type: ignore[assignment]
        if backend not in deduped_backends:
            deduped_backends.append(backend)
    deduped_reasons = list(dict.fromkeys(reason_codes))
    return TTSRoute(
        segment_id=segment.id,
        backends=deduped_backends,
        reason_codes=deduped_reasons,
        candidate_budget=candidate_budget or cfg.candidate_count,
        route_id=_route_id(segment.id, list(deduped_backends), deduped_reasons),
    )
