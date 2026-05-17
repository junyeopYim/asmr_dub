from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import soundfile as sf

from asmr_dub_pipeline.schemas import ProjectConfig, Segment
from asmr_dub_pipeline.tts.types import CandidateScore, TTSCandidate, TTSRoute

_DIGIT_RE = re.compile(r"\d")
_KOREAN_NUMERIC_TOKENS = {
    "영",
    "공",
    "일",
    "하나",
    "이",
    "둘",
    "삼",
    "셋",
    "사",
    "넷",
    "오",
    "다섯",
    "육",
    "여섯",
    "칠",
    "일곱",
    "팔",
    "여덟",
    "구",
    "아홉",
    "십",
}


def segment_text(segment: Segment) -> str:
    parts = [
        segment.source_script.text if segment.source_script else "",
        segment.script.tts_text if segment.script else "",
        segment.script.ja_text if segment.script else "",
    ]
    return " ".join(part for part in parts if part)


def segment_has_numeric_sequence(segment: Segment) -> bool:
    text = segment_text(segment)
    if _DIGIT_RE.search(text):
        return True
    tokens = re.findall(r"[\uac00-\ud7a3]+", text)
    return sum(1 for token in tokens if token in _KOREAN_NUMERIC_TOKENS) >= 2


def audio_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "rms_dbfs": -120.0, "duration_sec": None}
    try:
        info = sf.info(str(path))
        frames = int(info.frames)
        sample_rate = int(info.samplerate)
        sum_squares = 0.0
        sample_count = 0
        for block in sf.blocks(str(path), blocksize=65536, always_2d=True, dtype="float32"):
            sum_squares += float((block * block).sum())
            sample_count += int(block.size)
        rms = math.sqrt(sum_squares / sample_count) if sample_count else 0.0
        return {
            "exists": True,
            "duration_sec": float(frames) / sample_rate if sample_rate else None,
            "sample_rate": sample_rate,
            "channels": int(info.channels),
            "rms_dbfs": 20.0 * math.log10(rms) if rms > 0 else -120.0,
        }
    except Exception as exc:
        return {"exists": False, "error": str(exc), "rms_dbfs": -120.0, "duration_sec": None}


def _selector_weight(cfg: ProjectConfig, key: str, default: float) -> float:
    weights = cfg.tts.selector.weights
    value = getattr(weights, key, default)
    return float(value)


def _candidate_numeric_qc(candidate: TTSCandidate) -> dict[str, Any] | None:
    payload = candidate.payload
    for key in ("numeric_sequence_qc", "numeric_qc"):
        qc = payload.get(key)
        if isinstance(qc, dict):
            return qc
    numeric_phrase = payload.get("numeric_phrase")
    if isinstance(numeric_phrase, dict):
        for key in ("numeric_sequence_qc", "numeric_qc"):
            qc = numeric_phrase.get(key)
            if isinstance(qc, dict):
                return qc
    return None


def _is_numeric_phrase_candidate(candidate: TTSCandidate) -> bool:
    payload = candidate.payload
    return (
        str(payload.get("renderer") or "").strip().lower() == "numeric_phrase"
        or isinstance(payload.get("numeric_phrase"), dict)
    )


def _numeric_qc_passed(numeric_qc: dict[str, Any] | None) -> bool:
    return (
        isinstance(numeric_qc, dict)
        and str(numeric_qc.get("gate") or "").strip().lower() == "pass"
        and numeric_qc.get("exact_match") is not False
    )


def score_candidate(
    segment: Segment,
    candidate: TTSCandidate,
    cfg: ProjectConfig,
    *,
    route: TTSRoute | None = None,
) -> CandidateScore:
    path = Path(candidate.wav_path)
    metrics = audio_metrics(path)
    hard_fail_reasons: list[str] = []
    if not metrics.get("exists"):
        hard_fail_reasons.append("missing_wav")
    if metrics.get("exists") and float(metrics.get("rms_dbfs") or -120.0) <= -85.0:
        hard_fail_reasons.append("silent_or_too_quiet")

    duration = candidate.duration_sec or metrics.get("duration_sec")
    duration_ratio = (float(duration) / segment.duration) if duration and segment.duration > 0 else None
    duration_error = abs((duration_ratio or 0.0) - 1.0) if duration_ratio is not None else 1.0
    numeric_segment = segment_has_numeric_sequence(segment)
    numeric_qc = _candidate_numeric_qc(candidate)
    numeric_phrase_duration_soft = _is_numeric_phrase_candidate(candidate) and _numeric_qc_passed(numeric_qc)
    tolerance = float(cfg.tts.selector.duration_tolerance)
    if duration_ratio is None:
        hard_fail_reasons.append("missing_duration")
    elif duration_error > max(tolerance, 0.01) and not numeric_phrase_duration_soft:
        hard_fail_reasons.append("duration_tolerance_exceeded")

    preflight = candidate.payload.get("preflight") or candidate.payload.get("text_preflight")
    if isinstance(preflight, dict) and bool(preflight.get("blocked")):
        hard_fail_reasons.append("korean_preflight_blocked")

    if numeric_segment and cfg.tts.selector.require_numeric_sequence_qc:
        if isinstance(numeric_qc, dict):
            gate = str(numeric_qc.get("gate") or "").strip().lower()
            exact_match = numeric_qc.get("exact_match")
            if gate == "fail" or exact_match is False:
                hard_fail_reasons.append("numeric_sequence_qc_failed")
        elif candidate.backend != "qwen_tts":
            hard_fail_reasons.append("numeric_sequence_qc_missing")

    pronunciation_qc = candidate.payload.get("pronunciation_qc")
    if (
        numeric_segment
        and cfg.tts.selector.require_pronunciation_qc_for_numeric
        and isinstance(pronunciation_qc, dict)
        and str(pronunciation_qc.get("gate") or "").lower() == "fail"
    ):
        hard_fail_reasons.append("pronunciation_qc_failed")

    asr_backcheck = candidate.payload.get("ko_asr_backcheck")
    if isinstance(asr_backcheck, dict) and str(asr_backcheck.get("severity") or "").lower() == "severe":
        hard_fail_reasons.append("ko_asr_backcheck_severe_omission")

    blocked = bool(hard_fail_reasons)
    duration_fit = max(0.0, 1.0 - min(duration_error, 1.0))
    pronunciation = 1.0
    if isinstance(pronunciation_qc, dict):
        if str(pronunciation_qc.get("gate") or "").lower() == "fail":
            pronunciation = 0.0
        elif str(pronunciation_qc.get("gate") or "").lower() == "warn":
            pronunciation = 0.5
        elif pronunciation_qc.get("coverage") is not None:
            pronunciation = max(0.0, min(1.0, float(pronunciation_qc.get("coverage") or 0.0)))
    numeric_accuracy = 1.0 if not numeric_segment else 0.7
    if isinstance(numeric_qc, dict):
        numeric_accuracy = 1.0 if str(numeric_qc.get("gate") or "").lower() == "pass" else 0.0
    ko_asr_backcheck = 0.5 if isinstance(asr_backcheck, dict) else 1.0
    style_match = float(candidate.payload.get("style_match_score") or 1.0)
    noise_floor = max(0.0, min(1.0, (float(metrics.get("rms_dbfs") or -120.0) + 85.0) / 65.0))
    rms_stability = float(candidate.payload.get("rms_stability") or 1.0)

    reason_codes = route.reason_codes if route is not None else []
    qwen_bonus_route = any(
        code in reason_codes
        for code in (
            "micro_segment",
            "numeric_sequence",
            "previous_gsv_pronunciation_failure",
            "previous_gsv_duration_failure",
            "auto_repair_qwen_action",
        )
    )
    backend_prior = 0.85
    if candidate.backend == "gpt_sovits":
        backend_prior = 1.0 if not qwen_bonus_route else 0.85
    elif candidate.backend == "qwen_tts":
        backend_prior = 1.0 if qwen_bonus_route or numeric_segment or segment.duration <= cfg.tts.router.micro_segment_max_sec else 0.78

    score_parts = {
        "duration_fit": duration_fit,
        "pronunciation": pronunciation,
        "numeric_accuracy": numeric_accuracy,
        "ko_asr_backcheck": ko_asr_backcheck,
        "style_match": max(0.0, min(1.0, style_match)),
        "noise": noise_floor,
        "rms_stability": max(0.0, min(1.0, rms_stability)),
        "backend_prior": backend_prior,
    }
    weighted = (
        _selector_weight(cfg, "duration_fit", 0.30) * score_parts["duration_fit"]
        + _selector_weight(cfg, "pronunciation", 0.25) * score_parts["pronunciation"]
        + _selector_weight(cfg, "ko_asr_backcheck", 0.20) * score_parts["ko_asr_backcheck"]
        + _selector_weight(cfg, "style_match", 0.10) * score_parts["style_match"]
        + _selector_weight(cfg, "noise", 0.10) * score_parts["noise"]
        + _selector_weight(cfg, "backend_prior", 0.05) * score_parts["backend_prior"]
    )
    if numeric_segment:
        weighted = 0.85 * weighted + 0.15 * score_parts["numeric_accuracy"]
    return CandidateScore(
        candidate_id=candidate.candidate_id,
        backend=candidate.backend,
        blocked=blocked,
        hard_fail_reasons=hard_fail_reasons,
        score=0.0 if blocked else round(float(weighted), 6),
        score_parts={key: round(float(value), 6) for key, value in score_parts.items()},
    )
