from __future__ import annotations

# ruff: noqa: F403,F405,I001

import copy
import hashlib
import re
import threading

from asmr_dub_pipeline.gpu_memory import clear_gpu_vram
from asmr_dub_pipeline.pipeline.context import PipelineContext
from asmr_dub_pipeline.pipeline.stages.common import *
from asmr_dub_pipeline.qc.pronunciation_qc import (
    evaluate_numeric_sequence_text,
    normalize_korean_pronunciation_text,
)
from asmr_dub_pipeline.qc.repair_plan import (
    build_ko_qc_repair_plan,
    is_micro_segment,
    is_texture_like_micro_segment,
)
from asmr_dub_pipeline.pipeline.stages.numeric_phrase_renderer import (
    build_numeric_phrase_request,
    render_live_numeric_phrase,
)
from asmr_dub_pipeline.script.numeric_cadence import (
    extract_korean_numeric_values,
    periodize_korean_numeric_cadence_text,
)
from asmr_dub_pipeline.script.numeric_render_plan import (
    NumericRenderPlan,
    build_numeric_render_plan,
)

_OPEN_KOREAN_TTS_END_RE = re.compile(r"(?:\s*(?:[,，、]+|\.{2,}|…+))+\s*$")
_KOREAN_SENTENCE_FINAL_RE = re.compile(
    r"(?:[.!?。！？]|요|다|죠|네|까|니다|세요|어요|아요|해요|예요|이에요)\s*$"
)
_KOREAN_TOKEN_RE = re.compile(r"[가-힣]+")
_KOREAN_PRONUNCIATION_TERM_RE = re.compile(r"\d+|[가-힣]+")
_OMISSION_EXPECTED_RATIO_THRESHOLD = 0.35
_OMISSION_SEGMENT_RATIO_THRESHOLD = 0.25
_KOREAN_COUNTING_SEPARATOR_RE = re.compile(r"^[\s,，、・:;-]*$")
_KOREAN_COUNTING_FILLER_RE = re.compile(r"^\s*말이에요[.!?。！？]*\s*$")
_KOREAN_COUNTING_TOKEN_TO_SPOKEN = {
    "0": "영",
    "1": "하나",
    "2": "둘",
    "3": "셋",
    "4": "넷",
    "5": "다섯",
    "6": "여섯",
    "7": "일곱",
    "8": "여덟",
    "9": "아홉",
    "10": "열",
    "영": "영",
    "공": "영",
    "일": "하나",
    "이": "둘",
    "삼": "셋",
    "사": "넷",
    "오": "다섯",
    "육": "여섯",
    "칠": "일곱",
    "팔": "여덟",
    "구": "아홉",
    "십": "열",
    **{text: text for text in NATIVE_KOREAN_COUNT_ONES.values()},
    **{text: text for text in NATIVE_KOREAN_COUNT_TENS.values()},
}
_KOREAN_COUNTING_TOKEN_RE = re.compile(
    r"(?<![0-9A-Za-z가-힣])("
    + "|".join(
        re.escape(token)
        for token in sorted(_KOREAN_COUNTING_TOKEN_TO_SPOKEN, key=len, reverse=True)
    )
    + r")(?![0-9A-Za-z가-힣])"
)
_COUNTDOWN_ALIGNMENT_UNIT_RE = re.compile(r"[0-9A-Za-z가-힣]")
_COUNTDOWN_TOKEN_DIGITS = {
    "십": "10",
    "구": "9",
    "팔": "8",
    "칠": "7",
    "육": "6",
    "오": "5",
    "사": "4",
    "삼": "3",
    "이": "2",
    "일": "1",
    "영": "0",
}
_COUNTDOWN_TOKEN_PHONETIC = {
    "팔": "파알",
    "사": "사아",
}
_COUNTDOWN_SOURCE_ANCHOR_PROMPT = "日本語ASMRのカウントダウン。10 9 8 7 6 5 4 3 2 1 0 ゼロ。"
_COUNTDOWN_SOURCE_ANCHOR_HOTWORDS = "10 9 8 7 6 5 4 3 2 1 0 ゼロ 十 九 八 七 六 五 四 三 二 一 零"
_COUNTDOWN_SOURCE_CLUSTER_SEPARATOR_RE = re.compile(r"[\s,，、.。!！?？:：;；・…~〜-]+")
_NUMERIC_PHRASE_ALLOWED_REMAINDER_RE = re.compile(
    r"^[\s,，、.!?。！？・:：;；／/\\\-~〜…]*$"
)


def _numeric_phrase_token_pattern() -> re.Pattern[str]:
    tokens: set[str] = {"공"}
    for value in range(0, 100):
        tokens.add(str(value))
        native = native_korean_count_number(value)
        if native:
            tokens.add(native)
        sino_tokens = countdown_korean_tokens([value])
        if sino_tokens:
            tokens.update(sino_tokens)
    return re.compile(
        r"(?<![0-9A-Za-z가-힣])("
        + "|".join(re.escape(token) for token in sorted(tokens, key=len, reverse=True))
        + r")(?![0-9A-Za-z가-힣])"
    )


_NUMERIC_PHRASE_TOKEN_RE = _numeric_phrase_token_pattern()
_NUMERIC_PHRASE_ASR_PROMPT = "한국어 ASMR 숫자 카운팅 음성을 그대로 받아 적습니다."
_NUMERIC_PHRASE_ASR_HOTWORDS = (
    "영 공 하나 둘 셋 넷 다섯 여섯 일곱 여덟 아홉 열 "
    "일 이 삼 사 오 육 칠 팔 구 십 0 1 2 3 4 5 6 7 8 9 10"
)


def _gsv_invalid_ref_duration_error(error: object) -> bool:
    message = str(error).lower()
    return (
        "reference audio is outside" in message
        and "3-10" in message
        and "second" in message
    )


def _gsv_refs_duration_preflight(
    refs: dict[str, GPTSoVITSRef],
    cfg: ProjectConfig,
    *,
    required_styles: Sequence[str] = ("whisper_close", "sleepy"),
) -> dict[str, Any]:
    min_sec = max(float(cfg.gsv_ref_min_sec), GSV_API_MIN_REF_SEC)
    max_sec = float(cfg.gsv_ref_max_sec)
    duration_epsilon_sec = 0.02
    rows: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    for style in required_styles:
        ref = refs.get(style)
        if ref is None:
            rows.append(
                {
                    "style": style,
                    "valid": None,
                    "skipped": True,
                    "reject_reasons": ["missing_ref_style"],
                }
            )
            continue
        ref_path = Path(ref.ref_audio_path)
        try:
            actual_duration = duration_sec(ref_path)
        except Exception as exc:
            row = {
                "style": style,
                "valid": False,
                "ref_audio_path": ref.ref_audio_path,
                "reject_reasons": [f"ref_duration_unreadable:{exc}"],
            }
            rows.append(row)
            invalid.append(row)
            continue
        reject_reasons: list[str] = []
        if actual_duration + duration_epsilon_sec < min_sec:
            reject_reasons.append(
                f"actual_duration_below_ref_min:{actual_duration:.3f}<{min_sec:.3f}"
            )
        if actual_duration - duration_epsilon_sec > max_sec:
            reject_reasons.append(
                f"actual_duration_above_ref_max:{actual_duration:.3f}>{max_sec:.3f}"
            )
        row = {
            "style": style,
            "valid": not reject_reasons,
            "ref_audio_path": ref.ref_audio_path,
            "actual_duration_sec": round(actual_duration, 6),
            "min_duration_sec": round(min_sec, 6),
            "max_duration_sec": round(max_sec, 6),
            "reject_reasons": reject_reasons,
        }
        rows.append(row)
        if reject_reasons:
            invalid.append(row)
    return {"refs": rows, "invalid": invalid}


def _uses_real_gsv_client() -> bool:
    return getattr(GPTSoVITSClient, "__module__", "") == "asmr_dub_pipeline.gpt_sovits.client"


def _close_open_korean_tts_sentence(text: str) -> tuple[str, bool]:
    stripped = text.strip()
    if not stripped or not _OPEN_KOREAN_TTS_END_RE.search(stripped):
        return stripped, False
    base = _OPEN_KOREAN_TTS_END_RE.sub("", stripped).strip()
    if not base:
        return stripped, False
    if _KOREAN_SENTENCE_FINAL_RE.search(base):
        return f"{base}.", True
    tokens = _KOREAN_TOKEN_RE.findall(base)
    if stripped.endswith("…") and tokens:
        short_interjection = len(tokens) == 1 and len(tokens[0]) <= 2
        repeated_sound = len(tokens) > 1 and all(len(token) <= 2 for token in tokens)
        if short_interjection or repeated_sound:
            return stripped, False
    return f"{base} 말이에요.", True


def _gsv_omission_detection_reasons(
    *,
    duration_sec: float,
    target_duration_sec: float,
    expected_tts_duration_sec: float,
    duration_gate: str,
    audio_gate: str,
    language_contract_ok: bool,
) -> list[str]:
    if (
        duration_gate != "too_short"
        or audio_gate != "pass"
        or not language_contract_ok
        or duration_sec <= 0
    ):
        return []
    reasons: list[str] = []
    if expected_tts_duration_sec > 0:
        ratio = duration_sec / expected_tts_duration_sec
        if ratio < _OMISSION_EXPECTED_RATIO_THRESHOLD:
            reasons.append(
                "duration_below_expected_ratio:"
                f"{ratio:.3f}<{_OMISSION_EXPECTED_RATIO_THRESHOLD:.3f}"
            )
    if target_duration_sec > 0:
        ratio = duration_sec / target_duration_sec
        if ratio < _OMISSION_SEGMENT_RATIO_THRESHOLD:
            reasons.append(
                "duration_below_segment_ratio:"
                f"{ratio:.3f}<{_OMISSION_SEGMENT_RATIO_THRESHOLD:.3f}"
            )
    return reasons


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _gsv_candidate_edge_silence_score(candidate: TTSCandidate, segment: Segment) -> float:
    trim = candidate.payload.get("postprocess", {}).get("edge_silence_trim", {})
    if not isinstance(trim, dict):
        return 1.0
    leading = _safe_float(trim.get("leading_trim_sec"))
    trailing = _safe_float(trim.get("trailing_trim_sec"))
    total_trim = max(0.0, leading) + max(0.0, trailing)
    if total_trim <= 0.0 or segment.duration <= 0.0:
        return 1.0
    return max(0.0, 1.0 - min(total_trim / segment.duration, 1.0))


def _gsv_candidate_speed_score(candidate: TTSCandidate) -> float:
    speed = _safe_float(candidate.payload.get("speed_factor"), 1.0)
    return max(0.0, 1.0 - min(abs(speed - 1.0) / 0.35, 1.0))


def _gsv_candidate_style_score(candidate: TTSCandidate, segment: Segment) -> float:
    requested = str(candidate.payload.get("requested_ref_style") or "").strip()
    resolved = str(candidate.payload.get("resolved_ref_style") or "").strip()
    if candidate.payload.get("fallback_used"):
        return 0.82
    if requested and resolved and requested != resolved:
        return 0.88
    if segment.script and segment.script.ref_style and resolved and segment.script.ref_style != resolved:
        return 0.92
    return 1.0


def _gsv_candidate_rescue_score(candidate: TTSCandidate) -> float:
    reason = candidate.selection_reason or ""
    if "time_fit" in reason:
        return 0.58
    if "pause_padding" in reason:
        return 0.62
    if "rescue" in reason or candidate.payload.get("rescue"):
        return 0.72
    if reason == "duration_or_language_contract_failed":
        return 0.45
    return 1.0


def _gsv_candidate_audio_score(candidate: TTSCandidate) -> float:
    return 1.0 if candidate.payload.get("audio_qc", {}).get("gate") == "pass" else 0.0


def _gsv_candidate_duration_score(candidate: TTSCandidate, segment: Segment) -> float:
    ratio = candidate.duration_ratio
    if ratio is None and candidate.duration_sec is not None:
        ratio = duration_ratio(candidate.duration_sec, segment.duration)
    if ratio is None:
        return 0.0
    return max(0.0, 1.0 - min(abs(float(ratio) - 1.0), 1.0))


def _gsv_timing_quality_gate(
    ratio: float | None,
    duration_gate: str,
    timing_quality_tolerance: float,
) -> str:
    if ratio is None:
        return "unknown"
    if duration_gate == "unknown":
        return "unknown"
    if duration_gate != "pass":
        return "fail"
    if abs(float(ratio) - 1.0) <= timing_quality_tolerance:
        return "good"
    return "warn"


def _gsv_timing_quality_payload(
    ratio: float | None,
    duration_gate: str,
    timing_quality_tolerance: float,
    mix_duration_tolerance: float,
) -> dict[str, Any]:
    gate = _gsv_timing_quality_gate(ratio, duration_gate, timing_quality_tolerance)
    deviation = None if ratio is None else abs(float(ratio) - 1.0)
    payload: dict[str, Any] = {
        "gate": gate,
        "quality_tolerance": round(float(timing_quality_tolerance), 6),
        "mix_duration_tolerance": round(float(mix_duration_tolerance), 6),
    }
    if ratio is not None:
        payload["duration_ratio"] = round(float(ratio), 6)
    if deviation is not None:
        payload["duration_deviation"] = round(float(deviation), 6)
    return payload


def _gsv_timing_quality_is_good(candidate: TTSCandidate) -> bool:
    gate = str(candidate.timing_quality_gate or "").strip().lower()
    if gate in {"", "unknown"}:
        timing_quality = candidate.payload.get("timing_quality")
        if isinstance(timing_quality, dict):
            gate = str(timing_quality.get("gate") or "").strip().lower()
    return gate == "good"


def _gsv_candidate_pronunciation_score(candidate: TTSCandidate) -> float:
    qc = candidate.payload.get("pronunciation_qc")
    if not isinstance(qc, dict):
        return 1.0
    gate = str(qc.get("gate") or "").strip().lower()
    if gate in {"", "disabled", "skipped", "unavailable"}:
        return 1.0
    try:
        coverage = float(qc.get("coverage"))
    except (TypeError, ValueError):
        coverage = 0.0 if gate == "fail" else 1.0
    coverage = max(0.0, min(coverage, 1.0))
    if gate == "pass":
        return coverage
    if gate == "warn":
        return min(coverage, 0.74)
    if gate == "fail":
        return min(coverage, 0.25)
    return coverage


def _gsv_candidate_selection_components(
    candidate: TTSCandidate,
    segment: Segment,
) -> dict[str, float]:
    return {
        "duration": _gsv_candidate_duration_score(candidate, segment),
        "pronunciation": _gsv_candidate_pronunciation_score(candidate),
        "audio": _gsv_candidate_audio_score(candidate),
        "style": _gsv_candidate_style_score(candidate, segment),
        "speed": _gsv_candidate_speed_score(candidate),
        "edge_silence": _gsv_candidate_edge_silence_score(candidate, segment),
        "rescue": _gsv_candidate_rescue_score(candidate),
    }


def _gsv_candidate_selection_score(candidate: TTSCandidate, segment: Segment) -> float:
    components = _gsv_candidate_selection_components(candidate, segment)
    score = (
        components["duration"] * 0.32
        + components["pronunciation"] * 0.28
        + components["audio"] * 0.10
        + components["style"] * 0.12
        + components["speed"] * 0.10
        + components["edge_silence"] * 0.06
        + components["rescue"] * 0.02
    )
    return round(max(0.0, min(score, 1.0)), 6)


def _update_gsv_candidate_selection_scores(
    candidates: list[TTSCandidate],
    segment: Segment,
) -> None:
    for candidate in candidates:
        if candidate.error or candidate.duration_sec is None:
            continue
        components = _gsv_candidate_selection_components(candidate, segment)
        candidate.selection_score = _gsv_candidate_selection_score(candidate, segment)
        candidate.payload["selection_scoring"] = {
            key: round(value, 6) for key, value in components.items()
        }
        candidate.payload["selection_scoring"]["score"] = candidate.selection_score


def _select_gsv_candidate_for_mix(
    candidates: list[TTSCandidate],
    segment: Segment,
) -> TTSCandidate:
    if not candidates:
        raise ValueError("No TTS candidates to select.")
    _update_gsv_candidate_selection_scores(candidates, segment)
    return max(
        candidates,
        key=lambda candidate: (
            candidate.selection_score if candidate.selection_score is not None else 0.0,
            _gsv_candidate_duration_score(candidate, segment),
        ),
    )


def _compact_korean_counting_tts_text(text: str) -> tuple[str, dict[str, Any] | None]:
    stripped = text.strip()
    if not stripped:
        return stripped, None
    matches = list(_KOREAN_COUNTING_TOKEN_RE.finditer(stripped))
    if len(matches) < 3:
        return stripped, None

    groups: list[list[re.Match[str]]] = []
    current: list[re.Match[str]] = [matches[0]]
    for match in matches[1:]:
        separator = stripped[current[-1].end() : match.start()]
        if _KOREAN_COUNTING_SEPARATOR_RE.fullmatch(separator):
            current.append(match)
        else:
            if len(current) >= 3:
                groups.append(current)
            current = [match]
    if len(current) >= 3:
        groups.append(current)
    if not groups:
        return stripped, None

    replacements: list[dict[str, Any]] = []
    output_parts: list[str] = []
    last_index = 0
    for group in groups:
        start = group[0].start()
        end = group[-1].end()
        before = stripped[start:end]
        after = "".join(_KOREAN_COUNTING_TOKEN_TO_SPOKEN[match.group(1)] for match in group)
        output_parts.append(stripped[last_index:start])
        output_parts.append(after)
        replacements.append(
            {
                "before": before,
                "after": after,
                "token_count": len(group),
            }
        )
        last_index = end
    output_parts.append(stripped[last_index:])
    compacted = "".join(output_parts).strip()

    removed_counting_filler = False
    if len(groups) == 1:
        group = groups[0]
        prefix = stripped[: group[0].start()]
        suffix = stripped[group[-1].end() :]
        if not prefix.strip() and _KOREAN_COUNTING_FILLER_RE.fullmatch(suffix):
            compacted = replacements[0]["after"] + "."
            removed_counting_filler = True

    if compacted == stripped:
        return stripped, None
    metadata: dict[str, Any] = {
        "runs": replacements,
        "removed_counting_filler": removed_counting_filler,
    }
    return compacted, metadata


def _source_countdown_values(segment: Segment) -> list[int] | None:
    source_text = ""
    if segment.source_script is not None:
        source_text = segment.source_script.text
    elif segment.script is not None:
        source_text = segment.script.ja_text or segment.script.literal_ja
    if source_text.strip():
        return source_countdown_values(source_text)

    event = segment.analysis.get(COUNTDOWN_EVENT_KEY)
    if isinstance(event, dict):
        raw_values = event.get("values")
        if isinstance(raw_values, list) and all(isinstance(value, int) for value in raw_values):
            return [int(value) for value in raw_values]
    return None


def _countdown_event_values(segment: Segment) -> list[int] | None:
    event = segment.analysis.get(COUNTDOWN_EVENT_KEY)
    if not isinstance(event, dict):
        return None
    raw_values = event.get("values")
    if not isinstance(raw_values, list) or not raw_values:
        return None
    try:
        values = [int(value) for value in raw_values]
    except (TypeError, ValueError):
        return None
    return values


def _embedded_countdown_values(segment: Segment) -> list[int] | None:
    event = segment.analysis.get(COUNTDOWN_EVENT_KEY)
    if not isinstance(event, dict) or str(event.get("kind") or "") != "embedded_countdown":
        return None
    values = _countdown_event_values(segment)
    if values is None or not is_descending_countdown(values):
        return None
    if _countdown_spoken_tokens(values) is None:
        return None
    source_text = segment.source_script.text if segment.source_script else ""
    if not source_text.strip():
        return None
    if source_countdown_values(source_text) == values:
        return None
    matches = source_countdown_token_matches(source_text)
    for start in range(0, len(matches) - len(values) + 1):
        window = matches[start : start + len(values)]
        if [int(item[0]) for item in window] == values:
            return values
    return None


def _countdown_manifest_anchor_rows(segment: Segment, values: list[int]) -> list[dict[str, Any]]:
    event = segment.analysis.get(COUNTDOWN_EVENT_KEY)
    if not isinstance(event, dict):
        return []
    raw_timeline = event.get("source_anchor_timeline")
    local_timeline = bool(event.get("source_anchor_timeline_local"))
    if not isinstance(raw_timeline, list):
        raw_timeline = event.get("token_timeline")
        local_timeline = False
    if not isinstance(raw_timeline, list) or len(raw_timeline) != len(values):
        return []
    rows: list[dict[str, Any]] = []
    for item in raw_timeline:
        if not isinstance(item, dict):
            return []
        try:
            value = int(item["value"])
            raw_start = float(item["start"])
            raw_end = float(item.get("end", raw_start))
        except (KeyError, TypeError, ValueError):
            return []
        start = raw_start if local_timeline else raw_start - float(segment.start)
        end = raw_end if local_timeline else raw_end - float(segment.start)
        rows.append(
            {
                "value": value,
                "source_text": str(item.get("source_text") or value),
                "korean_token": str(item.get("korean_token") or ""),
                "start": round(max(0.0, min(float(segment.duration), start)), 6),
                "end": round(max(0.0, min(float(segment.duration), end)), 6),
                "confidence": item.get("confidence"),
                "method": str(item.get("method") or "manifest_timeline"),
            }
        )
    if [row["value"] for row in rows] != values:
        return []
    return rows


def _embedded_countdown_anchor_rows(
    segment: Segment,
    values: list[int],
    *,
    smoothing_blend: float,
    cluster_gap_sec: float,
) -> tuple[list[dict[str, Any]], list[list[int]], str]:
    rows = _countdown_manifest_anchor_rows(segment, values)
    source_kind = "manifest_timeline" if rows else "source_text_char_fraction"
    if not rows:
        rows = _countdown_source_text_anchor_rows(segment, values)
    if not rows or len(rows) != len(values):
        return [], [], "unavailable"
    starts = [float(row["start"]) for row in rows]
    source_text = segment.source_script.text if segment.source_script else ""
    clusters = _countdown_source_anchor_clusters(
        source_text,
        values,
        starts,
        max_cluster_gap_sec=cluster_gap_sec,
    )
    smoothed_starts = _countdown_smooth_source_anchor_starts(
        starts,
        clusters,
        blend=smoothing_blend,
    )
    smoothed_rows: list[dict[str, Any]] = []
    for row, start in zip(rows, smoothed_starts, strict=True):
        updated = dict(row)
        delta = float(row.get("end", row["start"])) - float(row["start"])
        updated["start"] = round(max(0.0, min(float(segment.duration), start)), 6)
        updated["end"] = round(
            max(0.0, min(float(segment.duration), float(updated["start"]) + max(0.0, delta))),
            6,
        )
        if smoothing_blend > 0 and any(len(cluster) >= 3 for cluster in clusters):
            updated["method"] = f"{updated.get('method') or source_kind}_cluster_smoothed"
        smoothed_rows.append(updated)
    return smoothed_rows, clusters, source_kind


def _countdown_neighbor_gap_sec(
    anchors: list[dict[str, Any]],
    index: int,
    duration_sec: float,
) -> float:
    anchor = float(anchors[index]["start"])
    gaps: list[float] = []
    if index > 0:
        gaps.append(anchor - float(anchors[index - 1]["start"]))
    if index + 1 < len(anchors):
        gaps.append(float(anchors[index + 1]["start"]) - anchor)
    positive = [gap for gap in gaps if gap > 0.05]
    if positive:
        return min(positive)
    return max(0.24, duration_sec / max(len(anchors), 1))


def _countdown_hybrid_target_token_sec(
    anchors: list[dict[str, Any]],
    index: int,
    duration_sec: float,
    *,
    gap_fill_ratio: float,
    max_token_sec: float,
) -> float:
    gap = _countdown_neighbor_gap_sec(anchors, index, duration_sec)
    return max(0.12, min(float(max_token_sec), gap * float(gap_fill_ratio)))


def _countdown_trim_audio_to_active_region(
    audio: np.ndarray,
    sample_rate: int,
    *,
    keep_before_sec: float,
    keep_after_sec: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    data = _countdown_stereo(np.asarray(audio, dtype=np.float32))
    detected = _countdown_detect_active_islands(data, sample_rate)
    islands = detected.get("islands_frames", []) if isinstance(detected, dict) else []
    if not islands:
        return data, {"applied": False, "reason": "no_active_region"}
    start_frame = max(0, int(islands[0][0]) - int(round(keep_before_sec * sample_rate)))
    end_frame = min(len(data), int(islands[-1][1]) + int(round(keep_after_sec * sample_rate)))
    if end_frame <= start_frame:
        return data, {"applied": False, "reason": "empty_active_region"}
    trimmed = data[start_frame:end_frame]
    return trimmed, {
        "applied": True,
        "start_trim_sec": round(start_frame / sample_rate, 6),
        "end_trim_sec": round(max(0, len(data) - end_frame) / sample_rate, 6),
        "active_island_count": len(islands),
    }


def _countdown_retime_audio_to_frames(audio: np.ndarray, target_frames: int) -> np.ndarray:
    data = _countdown_stereo(np.asarray(audio, dtype=np.float32))
    target_frames = max(1, int(target_frames))
    if len(data) == target_frames:
        return data
    if len(data) <= 1:
        return np.repeat(data[:1], target_frames, axis=0)
    old_x = np.linspace(0.0, 1.0, len(data), endpoint=False)
    new_x = np.linspace(0.0, 1.0, target_frames, endpoint=False)
    channels = [
        np.interp(new_x, old_x, data[:, channel])
        for channel in range(data.shape[1])
    ]
    return np.stack(channels, axis=1).astype(np.float32)


def _is_strict_descending_countdown(values: list[int]) -> bool:
    return is_descending_countdown(values)


def _countdown_spoken_tokens(values: list[int]) -> list[str] | None:
    return countdown_korean_tokens(values)


def _countdown_numeric_items_from_text(text: str) -> list[tuple[int, str]] | None:
    matches = source_countdown_token_matches(text)
    if matches:
        return [(int(value), raw) for value, raw, _start, _end in matches]
    values = source_countdown_values(text.strip(" \t\r\n,，、。.!！?？"))
    if values is None:
        return None
    return [(int(value), str(value)) for value in values]


def _countdown_source_text_anchor_rows(
    segment: Segment,
    values: list[int],
) -> list[dict[str, Any]]:
    source_text = ""
    if segment.source_script is not None:
        source_text = segment.source_script.text
    elif segment.script is not None:
        source_text = segment.script.ja_text or segment.script.literal_ja
    matches = source_countdown_token_matches(source_text)
    selected: list[tuple[int, str, int, int]] | None = None
    for start in range(0, len(matches) - len(values) + 1):
        window = matches[start : start + len(values)]
        if [int(item[0]) for item in window] == values:
            selected = window
            break
    if selected is None:
        return []
    text_len = max(len(source_text), 1)
    tokens = _countdown_spoken_tokens(values) or [str(value) for value in values]
    rows: list[dict[str, Any]] = []
    for (value, raw, start, end), token in zip(selected, tokens, strict=True):
        center = ((start + end) / 2.0) / text_len
        anchor_sec = max(0.0, min(float(segment.duration), center * float(segment.duration)))
        rows.append(
            {
                "value": int(value),
                "source_text": raw,
                "korean_token": token,
                "start": round(anchor_sec, 6),
                "end": round(min(float(segment.duration), anchor_sec + 0.35), 6),
                "confidence": 0.25,
                "method": "source_text_char_fraction",
            }
        )
    return rows


def _countdown_numeric_word_events(chunks: Sequence[ASRChunk]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for chunk in sorted(chunks, key=lambda item: (float(item.start), float(item.end))):
        words = sorted(chunk.words, key=lambda item: (float(item.start), float(item.end)))
        index = 0
        while index < len(words):
            word = words[index]
            parsed = _countdown_numeric_items_from_text(str(word.text or ""))
            if parsed:
                duration = max(0.001, float(word.end) - float(word.start))
                for offset, (value, raw) in enumerate(parsed):
                    start = float(word.start) + duration * offset / len(parsed)
                    end = float(word.start) + duration * (offset + 1) / len(parsed)
                    events.append(
                        {
                            "value": int(value),
                            "source_text": raw,
                            "start": round(start, 6),
                            "end": round(end, 6),
                            "confidence": word.confidence,
                            "method": "asr_word_event",
                        }
                    )
                index += 1
                continue
            if index + 1 < len(words):
                joined = f"{word.text or ''}{words[index + 1].text or ''}"
                joined_parsed = _countdown_numeric_items_from_text(joined)
                if joined_parsed and len(joined_parsed) == 1:
                    value, raw = joined_parsed[0]
                    confidences = [
                        item
                        for item in (word.confidence, words[index + 1].confidence)
                        if item is not None
                    ]
                    events.append(
                        {
                            "value": int(value),
                            "source_text": raw,
                            "start": round(float(word.start), 6),
                            "end": round(float(words[index + 1].end), 6),
                            "confidence": min(confidences) if confidences else None,
                            "method": "asr_word_joined_event",
                        }
                    )
                    index += 2
                    continue
            index += 1
    return sorted(events, key=lambda item: (float(item["start"]), float(item["end"])))


def _countdown_numeric_chunk_events(chunks: Sequence[ASRChunk]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for chunk in sorted(chunks, key=lambda item: (float(item.start), float(item.end))):
        parsed = _countdown_numeric_items_from_text(str(chunk.text or ""))
        if not parsed:
            continue
        duration = max(0.001, float(chunk.end) - float(chunk.start))
        for index, (value, raw) in enumerate(parsed):
            start = float(chunk.start) + duration * index / len(parsed)
            end = float(chunk.start) + duration * (index + 1) / len(parsed)
            events.append(
                {
                    "value": int(value),
                    "source_text": raw,
                    "start": round(start, 6),
                    "end": round(end, 6),
                    "confidence": chunk.confidence,
                    "method": "asr_chunk_event",
                }
            )
    return sorted(events, key=lambda item: (float(item["start"]), float(item["end"])))


def _countdown_event_window(
    events: list[dict[str, Any]],
    values: list[int],
) -> list[dict[str, Any]] | None:
    event_values = [int(event["value"]) for event in events]
    for start in range(0, len(event_values) - len(values) + 1):
        if event_values[start : start + len(values)] == values:
            return [dict(event) for event in events[start : start + len(values)]]
    return None


def _countdown_align_events_by_value(
    segment: Segment,
    values: list[int],
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not events:
        return []
    char_rows = _countdown_source_text_anchor_rows(segment, values)
    tokens = _countdown_spoken_tokens(values) or [str(value) for value in values]
    aligned: list[dict[str, Any] | None] = []
    cursor = 0
    for value in values:
        match_index = None
        for index in range(cursor, len(events)):
            if int(events[index]["value"]) == int(value):
                match_index = index
                break
        if match_index is None:
            aligned.append(None)
            continue
        aligned.append(dict(events[match_index]))
        cursor = match_index + 1
    if not any(item is not None for item in aligned):
        return []
    matched = [item for item in aligned if item is not None]
    if len(matched) != len(values):
        return []
    gaps = [
        float(right["start"]) - float(left["start"])
        for left, right in zip(matched, matched[1:], strict=False)
        if float(right["start"]) > float(left["start"])
    ]
    median_gap = sorted(gaps)[len(gaps) // 2] if gaps else 0.62
    rows: list[dict[str, Any]] = []
    for index, (value, token) in enumerate(zip(values, tokens, strict=True)):
        item = aligned[index]
        if item is not None:
            rows.append(
                {
                    "value": value,
                    "source_text": item.get("source_text") or str(value),
                    "korean_token": token,
                    "start": round(max(0.0, min(float(segment.duration), float(item["start"]))), 6),
                    "end": round(max(0.0, min(float(segment.duration), float(item.get("end", item["start"])))), 6),
                    "confidence": item.get("confidence"),
                    "method": "asr_value_aligned",
                    "asr_value": item.get("value"),
                }
            )
            continue
        if index < len(char_rows):
            anchor = float(char_rows[index]["start"])
        elif rows:
            anchor = float(rows[-1]["start"]) + median_gap
        else:
            next_anchor = next(
                (
                    float(candidate["start"])
                    for candidate in aligned[index + 1 :]
                    if candidate is not None
                ),
                0.0,
            )
            anchor = max(0.0, next_anchor - median_gap)
        rows.append(
            {
                "value": value,
                "source_text": str(value),
                "korean_token": token,
                "start": round(max(0.0, min(float(segment.duration), anchor)), 6),
                "end": round(max(0.0, min(float(segment.duration), anchor + min(median_gap * 0.65, 0.55))), 6),
                "confidence": 0.18,
                "method": "asr_value_aligned_fill",
            }
        )
    if any(float(right["start"]) < float(left["start"]) for left, right in zip(rows, rows[1:], strict=False)):
        return []
    return rows


def _countdown_assign_events_to_source_slots(
    char_rows: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> list[tuple[int, int]]:
    if not char_rows or not events:
        return []
    slot_count = len(char_rows)
    event_count = len(events)
    dp = [[math.inf] * (event_count + 1) for _ in range(slot_count + 1)]
    back: list[list[tuple[int, int, str] | None]] = [
        [None] * (event_count + 1) for _ in range(slot_count + 1)
    ]
    dp[0][0] = 0.0
    for slot_index in range(slot_count + 1):
        for event_index in range(event_count + 1):
            current = dp[slot_index][event_index]
            if math.isinf(current):
                continue
            if slot_index < slot_count and current + 0.55 < dp[slot_index + 1][event_index]:
                dp[slot_index + 1][event_index] = current + 0.55
                back[slot_index + 1][event_index] = (slot_index, event_index, "skip_slot")
            if slot_index < slot_count and event_index < event_count:
                char_t = float(char_rows[slot_index]["start"])
                event_t = float(events[event_index]["start"])
                value_penalty = (
                    -0.03
                    if int(events[event_index]["value"]) == int(char_rows[slot_index]["value"])
                    else 0.04
                )
                cost = current + min(abs(event_t - char_t) / 1.4, 2.5) + value_penalty
                if cost < dp[slot_index + 1][event_index + 1]:
                    dp[slot_index + 1][event_index + 1] = cost
                    back[slot_index + 1][event_index + 1] = (slot_index, event_index, "take")
            if event_index < event_count and current + 0.75 < dp[slot_index][event_index + 1]:
                dp[slot_index][event_index + 1] = current + 0.75
                back[slot_index][event_index + 1] = (slot_index, event_index, "skip_event")
    slot_index = slot_count
    event_index = event_count
    pairs: list[tuple[int, int]] = []
    while slot_index > 0 or event_index > 0:
        previous = back[slot_index][event_index]
        if previous is None:
            break
        prev_slot, prev_event, action = previous
        if action == "take":
            pairs.append((slot_index - 1, event_index - 1))
        slot_index, event_index = prev_slot, prev_event
    return list(reversed(pairs))


def _countdown_fill_source_slot_anchors(
    segment: Segment,
    values: list[int],
    char_rows: list[dict[str, Any]],
    events: list[dict[str, Any]],
    pairs: list[tuple[int, int]],
) -> list[dict[str, Any]]:
    if not char_rows:
        return []
    tokens = _countdown_spoken_tokens(values) or [str(value) for value in values]
    anchors: list[dict[str, Any] | None] = [None] * len(values)
    for slot_index, event_index in pairs:
        event = events[event_index]
        anchors[slot_index] = {
            "value": values[slot_index],
            "source_text": event.get("source_text") or char_rows[slot_index].get("source_text"),
            "korean_token": tokens[slot_index],
            "start": round(max(0.0, min(float(segment.duration), float(event["start"]))), 6),
            "end": round(max(0.0, min(float(segment.duration), float(event.get("end", event["start"])))), 6),
            "confidence": max(0.35, float(event.get("confidence") or 0.0)),
            "method": "asr_char_dp",
            "asr_value": event.get("value"),
        }

    known = [index for index, anchor in enumerate(anchors) if anchor is not None]
    if not known:
        return []

    def assign_fill(index: int, anchor_sec: float, method: str) -> None:
        anchors[index] = {
            "value": values[index],
            "source_text": char_rows[index].get("source_text"),
            "korean_token": tokens[index],
            "start": round(max(0.0, min(float(segment.duration), anchor_sec)), 6),
            "end": round(max(0.0, min(float(segment.duration), anchor_sec + 0.35)), 6),
            "confidence": 0.22,
            "method": method,
        }

    for left, right in zip(known, known[1:], strict=False):
        if right - left <= 1:
            continue
        left_t = float(anchors[left]["start"])  # type: ignore[index]
        right_t = float(anchors[right]["start"])  # type: ignore[index]
        for index in range(left + 1, right):
            fraction = (index - left) / (right - left)
            interp_t = left_t + (right_t - left_t) * fraction
            char_t = float(char_rows[index]["start"])
            assign_fill(index, 0.35 * char_t + 0.65 * interp_t, "asr_char_dp_interpolated")

    if known[0] > 0:
        gaps = [
            (float(anchors[right]["start"]) - float(anchors[left]["start"])) / (right - left)  # type: ignore[index]
            for left, right in zip(known, known[1:], strict=False)
            if right > left and float(anchors[right]["start"]) > float(anchors[left]["start"])  # type: ignore[index]
        ]
        default_gap = sorted(gaps)[len(gaps) // 2] if gaps else 0.62
        default_gap = max(0.42, min(0.85, default_gap))
        first = known[0]
        first_t = float(anchors[first]["start"])  # type: ignore[index]
        for index in range(first - 1, -1, -1):
            char_t = float(char_rows[index]["start"])
            assign_fill(index, max(0.0, min(char_t, first_t - default_gap * (first - index))), "asr_char_dp_prefix_fill")

    if known[-1] < len(values) - 1:
        gaps = [
            (float(anchors[right]["start"]) - float(anchors[left]["start"])) / (right - left)  # type: ignore[index]
            for left, right in zip(known, known[1:], strict=False)
            if right > left and float(anchors[right]["start"]) > float(anchors[left]["start"])  # type: ignore[index]
        ]
        default_gap = sorted(gaps)[len(gaps) // 2] if gaps else 0.62
        default_gap = max(0.42, min(0.85, default_gap))
        last = known[-1]
        last_t = float(anchors[last]["start"])  # type: ignore[index]
        for index in range(last + 1, len(values)):
            char_t = float(char_rows[index]["start"])
            assign_fill(index, max(char_t, last_t + default_gap * (index - last)), "asr_char_dp_suffix_fill")

    complete = [anchor for anchor in anchors if anchor is not None]
    if len(complete) != len(values):
        return []
    previous = -0.08
    for anchor in complete:
        anchor["start"] = round(max(previous + 0.08, float(anchor["start"])), 6)
        previous = float(anchor["start"])
    if complete[-1]["start"] > float(segment.duration):
        shift = float(complete[-1]["start"]) - float(segment.duration)
        for anchor in complete:
            anchor["start"] = round(max(0.0, float(anchor["start"]) - shift), 6)
    return complete


def _countdown_source_anchor_rows_from_chunks(
    segment: Segment,
    values: list[int],
    chunks: Sequence[ASRChunk],
) -> tuple[str, list[dict[str, Any]]]:
    word_events = _countdown_numeric_word_events(chunks)
    word_window = _countdown_event_window(word_events, values)
    tokens = _countdown_spoken_tokens(values) or [str(value) for value in values]
    if word_window is not None:
        rows = []
        for item, token in zip(word_window, tokens, strict=True):
            rows.append({**item, "korean_token": token, "method": "asr_word_exact"})
        return "asr_word_exact", rows

    aligned = _countdown_align_events_by_value(segment, values, word_events)
    if aligned:
        return "asr_value_aligned", aligned

    chunk_events = _countdown_numeric_chunk_events(chunks)
    chunk_window = _countdown_event_window(chunk_events, values)
    if chunk_window is not None:
        rows = []
        for item, token in zip(chunk_window, tokens, strict=True):
            rows.append({**item, "korean_token": token, "method": "asr_chunk_exact"})
        return "asr_chunk_exact", rows

    aligned = _countdown_align_events_by_value(segment, values, chunk_events)
    if aligned:
        return "asr_chunk_value_aligned", aligned

    char_rows = _countdown_source_text_anchor_rows(segment, values)
    events = word_events or chunk_events
    pairs = _countdown_assign_events_to_source_slots(char_rows, events)
    filled = _countdown_fill_source_slot_anchors(segment, values, char_rows, events, pairs)
    if filled:
        return "source_text_dp_fallback", filled
    if char_rows:
        return "source_text_char_fraction_fallback", char_rows
    return "unavailable", []


def _countdown_source_anchor_clusters(
    source_text: str,
    values: list[int],
    starts_sec: list[float],
    *,
    max_cluster_gap_sec: float,
) -> list[list[int]]:
    matches = source_countdown_token_matches(source_text)
    selected: list[tuple[int, str, int, int]] | None = None
    for start in range(0, len(matches) - len(values) + 1):
        window = matches[start : start + len(values)]
        if [int(item[0]) for item in window] == values:
            selected = window
            break
    if selected is None or len(selected) != len(values):
        return [list(range(len(values)))]
    clusters: list[list[int]] = [[0]]
    for index in range(1, len(values)):
        previous = selected[index - 1]
        current = selected[index]
        between = source_text[previous[3] : current[2]]
        has_words_between = bool(_COUNTDOWN_SOURCE_CLUSTER_SEPARATOR_RE.sub("", between))
        gap = starts_sec[index] - starts_sec[index - 1]
        if has_words_between or gap > max_cluster_gap_sec:
            clusters.append([index])
        else:
            clusters[-1].append(index)
    return clusters


def _countdown_smooth_source_anchor_starts(
    starts_sec: list[float],
    clusters: list[list[int]],
    *,
    blend: float,
) -> list[float]:
    smoothed = list(starts_sec)
    for cluster in clusters:
        if len(cluster) < 3:
            continue
        first = cluster[0]
        last = cluster[-1]
        start = starts_sec[first]
        end = starts_sec[last]
        if end <= start:
            continue
        step = (end - start) / (len(cluster) - 1)
        for offset, index in enumerate(cluster):
            uniform = start + step * offset
            smoothed[index] = starts_sec[index] * (1.0 - blend) + uniform * blend
    return [round(max(0.0, value), 6) for value in smoothed]


def _countdown_phrase_chunks(tokens: list[str]) -> list[list[str]]:
    chunks: list[list[str]] = []
    index = 0
    while index < len(tokens):
        remaining = len(tokens) - index
        if remaining == 1 and chunks:
            chunks[-1].append(tokens[index])
            break
        if remaining == 3:
            chunks.append(tokens[index : index + 3])
            break
        take = min(2, remaining)
        chunks.append(tokens[index : index + take])
        index += take
    return chunks


def _countdown_chunk_label(text: str) -> str:
    label = re.sub(r"[^0-9A-Za-z가-힣]+", "_", text).strip("_")
    return label or "chunk"


def _countdown_lcs_length(left: str, right: str) -> int:
    if not left or not right:
        return 0
    previous = [0] * (len(right) + 1)
    for left_char in left:
        current = [0]
        diagonal = 0
        for index, right_char in enumerate(right, start=1):
            above = previous[index]
            if left_char == right_char:
                current.append(diagonal + 1)
            else:
                current.append(max(previous[index], current[-1]))
            diagonal = above
        previous = current
    return previous[-1]


def _countdown_carrier_full_sentence_prefilter_match(
    expected_text: str,
    transcript: str,
) -> dict[str, Any]:
    expected = normalize_korean_pronunciation_text(expected_text)
    observed_terms = [
        normalize_korean_pronunciation_text(term)
        for term in _KOREAN_PRONUNCIATION_TERM_RE.findall(transcript)
    ]
    observed_terms = [term for term in observed_terms if term]
    normalized_transcript = normalize_korean_pronunciation_text(transcript)
    if normalized_transcript and normalized_transcript not in observed_terms:
        observed_terms.append(normalized_transcript)
    if not expected:
        return {
            "expected_text": expected_text,
            "transcript": transcript,
            "expected": expected,
            "observed_terms": observed_terms,
            "coverage": 1.0,
            "matched": False,
            "reason": "empty_expected_text",
        }

    exact_match = expected in observed_terms
    if len(expected) == 1:
        contains_match = False
        coverage = 1.0 if exact_match else 0.0
        matched = exact_match
    else:
        contains_match = any(expected in term for term in observed_terms)
        coverage_values = [
            _countdown_lcs_length(expected, term) / len(expected)
            for term in observed_terms
        ]
        coverage = max(coverage_values, default=0.0)
        if exact_match or contains_match:
            coverage = 1.0
        matched = exact_match or contains_match or coverage >= 1.0
    return {
        "expected_text": expected_text,
        "transcript": transcript,
        "expected": expected,
        "observed_terms": observed_terms,
        "coverage": round(max(0.0, min(coverage, 1.0)), 6),
        "matched": matched,
        "reason": None if matched else "target_token_not_found",
    }


def _countdown_token_has_hangul_coda(token_text: str) -> bool:
    normalized = normalize_korean_pronunciation_text(token_text)
    for char in normalized:
        code = ord(char)
        if 0xAC00 <= code <= 0xD7A3 and (code - 0xAC00) % 28 != 0:
            return True
    return False


def _countdown_apply_slice_start_backoff(
    start_frame: int,
    *,
    sample_rate: int,
    backoff_sec: float,
) -> int:
    start = max(0, int(start_frame))
    if sample_rate <= 0:
        return start
    backoff_frames = max(0, int(round(max(0.0, float(backoff_sec)) * sample_rate)))
    return max(0, start - backoff_frames)


def _countdown_audio_envelope(audio: np.ndarray) -> np.ndarray:
    data = np.asarray(audio, dtype=np.float32)
    if data.ndim == 0:
        return np.asarray([], dtype=np.float32)
    if data.ndim == 1:
        return np.abs(data)
    if data.shape[0] <= 0:
        return np.asarray([], dtype=np.float32)
    return np.max(np.abs(data), axis=1)


def _countdown_rms(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    data = values.astype(np.float32, copy=False)
    return float(np.sqrt(np.mean(np.square(data))))


def _countdown_extend_slice_end_to_energy_valley(
    audio: np.ndarray,
    *,
    start_frame: int,
    end_frame: int,
    sample_rate: int,
    token_text: str = "",
    enabled: bool = True,
    max_extension_sec: float = 0.10,
    coda_max_extension_sec: float = 0.20,
    edge_window_sec: float = 0.025,
    quiet_window_sec: float = 0.020,
    quiet_pad_sec: float = 0.012,
    edge_threshold_ratio: float = 0.12,
    quiet_threshold_ratio: float = 0.08,
) -> dict[str, Any]:
    envelope = _countdown_audio_envelope(audio)
    total_frames = int(envelope.size)
    start = max(0, min(max(total_frames - 1, 0), int(start_frame)))
    end = max(start + 1, min(total_frames, int(end_frame))) if total_frames else 0
    metadata: dict[str, Any] = {
        "enabled": bool(enabled),
        "gate": "pass",
        "reason": "disabled" if not enabled else "unchanged",
        "start_frame": start,
        "original_end_frame": end,
        "end_frame": end,
        "extended_frames": 0,
        "extended_sec": 0.0,
        "edge_rms_ratio": 0.0,
        "edge_peak_ratio": 0.0,
        "final_edge_rms_ratio": 0.0,
        "cut_risk_score": 0.0,
        "token_has_coda": _countdown_token_has_hangul_coda(token_text),
    }
    if not enabled or sample_rate <= 0 or total_frames <= 0 or end <= start:
        return metadata

    has_coda = bool(metadata["token_has_coda"])
    extension_sec = coda_max_extension_sec if has_coda else max_extension_sec
    max_extension_frames = max(0, int(round(max(0.0, extension_sec) * sample_rate)))
    if max_extension_frames <= 0:
        metadata["reason"] = "extension_disabled"
        return metadata
    search_end = min(total_frames, end + max_extension_frames)
    active_region = envelope[start:search_end]
    local_peak = float(np.max(active_region)) if active_region.size else 0.0
    noise_floor = 10 ** (-60.0 / 20.0)
    if local_peak <= noise_floor:
        metadata["reason"] = "silent"
        return metadata

    edge_frames = max(1, int(round(max(0.001, edge_window_sec) * sample_rate)))
    quiet_frames = max(1, int(round(max(0.001, quiet_window_sec) * sample_rate)))
    quiet_pad_frames = max(0, int(round(max(0.0, quiet_pad_sec) * sample_rate)))
    edge = envelope[max(start, end - edge_frames) : end]
    edge_rms_ratio = _countdown_rms(edge) / local_peak
    edge_peak_ratio = (float(np.max(edge)) / local_peak) if edge.size else 0.0
    edge_threshold = max(0.01, float(edge_threshold_ratio))
    quiet_threshold = max(local_peak * max(0.005, float(quiet_threshold_ratio)), noise_floor)
    edge_is_active = edge_rms_ratio >= edge_threshold or edge_peak_ratio >= edge_threshold * 1.6
    metadata.update(
        {
            "edge_rms_ratio": round(edge_rms_ratio, 6),
            "edge_peak_ratio": round(edge_peak_ratio, 6),
            "edge_threshold_ratio": round(edge_threshold, 6),
            "quiet_threshold_ratio": round(float(quiet_threshold_ratio), 6),
            "max_extension_sec": round(extension_sec, 6),
        }
    )
    if not edge_is_active or search_end <= end:
        metadata["reason"] = "edge_already_quiet"
        return metadata

    new_end = end
    reason = "no_energy_valley_found"
    tail = envelope[end:search_end]
    if tail.size >= quiet_frames:
        for offset in range(0, tail.size - quiet_frames + 1):
            window = tail[offset : offset + quiet_frames]
            if _countdown_rms(window) <= quiet_threshold and float(np.max(window)) <= quiet_threshold * 2.0:
                new_end = min(total_frames, end + offset + quiet_frames + quiet_pad_frames)
                reason = "extended_to_energy_valley"
                break

    final_edge = envelope[max(start, new_end - edge_frames) : new_end]
    final_edge_rms_ratio = _countdown_rms(final_edge) / local_peak
    cut_risk = max(0.0, min(final_edge_rms_ratio / edge_threshold, 1.0))
    gate = "pass"
    if final_edge_rms_ratio >= edge_threshold:
        gate = "warn"
    metadata.update(
        {
            "gate": gate,
            "reason": reason,
            "end_frame": int(new_end),
            "extended_frames": int(new_end - end),
            "extended_sec": round((new_end - end) / sample_rate, 6),
            "final_edge_rms_ratio": round(final_edge_rms_ratio, 6),
            "cut_risk_score": round(cut_risk, 6),
        }
    )
    return metadata


def _countdown_slice_boundary_is_clean(boundary_qc: dict[str, Any] | None) -> bool:
    if not isinstance(boundary_qc, dict):
        return True
    gate = str(boundary_qc.get("gate") or "pass").strip().lower()
    if gate in {"", "disabled", "skipped", "unavailable"}:
        return True
    if gate == "fail":
        return False
    cut_risk = max(0.0, min(_safe_float(boundary_qc.get("cut_risk_score"), 0.0), 1.0))
    final_edge_rms_ratio = max(
        0.0,
        _safe_float(boundary_qc.get("final_edge_rms_ratio"), 0.0),
    )
    if gate == "warn":
        return False
    return cut_risk <= 0.80 and final_edge_rms_ratio <= 0.14


def _countdown_tts_text(token: str) -> str:
    return token.strip()


def _countdown_phrase_tts_text(tokens: list[str]) -> str:
    return ", ".join(token for token in (_countdown_tts_text(token) for token in tokens) if token)


def _countdown_space_phrase_tts_text(tokens: list[str]) -> str:
    return " ".join(token for token in (_countdown_tts_text(token) for token in tokens) if token)


def _countdown_digit_space_text(values: list[int]) -> str | None:
    if not values or any(value < 0 or value > 9 for value in values):
        return None
    return " ".join(str(value) for value in values)


def _countdown_canonical_pack_prompt_specs(
    values: list[int],
    tokens: list[str],
) -> list[dict[str, Any]]:
    expected_text = _countdown_space_phrase_tts_text(tokens)
    if not expected_text:
        return []
    specs: list[dict[str, Any]] = []
    seen: set[tuple[str, str, float]] = set()

    def add_spec(prompt_kind: str, prompt_text: str, speed_factor: float) -> None:
        prompt_text = prompt_text.strip()
        if not prompt_text:
            return
        key = (prompt_kind, prompt_text, round(float(speed_factor), 6))
        if key in seen:
            return
        seen.add(key)
        specs.append(
            {
                "prompt_kind": prompt_kind,
                "prompt_text": prompt_text,
                "expected_text": expected_text,
                "speed_factor": float(speed_factor),
            }
        )

    digit_text = _countdown_digit_space_text(values)
    if digit_text is not None and len(values) >= 4:
        for speed_factor in (1.0, 0.9, 0.85, 1.1):
            add_spec("digit_space", digit_text, speed_factor)

    if 10 in values and len(values) <= 3:
        sino_speeds = (1.1, 1.25, 1.0, 0.95)
    elif 10 in values:
        sino_speeds = (1.0, 1.1, 0.9, 1.25)
    else:
        sino_speeds = (1.0, 0.9, 1.15, 1.08)
    for speed_factor in sino_speeds:
        add_spec("sino_space", expected_text, speed_factor)

    comma_text = _countdown_phrase_tts_text(tokens)
    add_spec("sino_comma", comma_text, 1.0)
    if len(tokens) >= 4:
        add_spec("sino_comma", comma_text, 1.1)
        period_text = ". ".join(token for token in (_countdown_tts_text(token) for token in tokens) if token)
        add_spec("sino_period", period_text, 1.0)
        add_spec("sino_period", period_text, 1.1)
    return specs


def _countdown_canonical_pack_candidate_prompt_spec(
    values: list[int],
    prompt_specs: list[dict[str, Any]],
    candidate_index: int,
) -> dict[str, Any]:
    def find_spec(prompt_kind: str, speed_factor: float) -> dict[str, Any] | None:
        for spec in prompt_specs:
            if str(spec.get("prompt_kind") or "") != prompt_kind:
                continue
            try:
                spec_speed = float(spec.get("speed_factor"))
            except (TypeError, ValueError):
                continue
            if abs(spec_speed - speed_factor) <= 1e-6:
                return spec
        return None

    schedule: list[tuple[str, float]] = []
    if values and all(0 <= value <= 9 for value in values) and len(values) >= 4:
        schedule = [
            ("digit_space", 1.0),
            ("sino_space", 1.0),
            ("digit_space", 0.9),
            ("sino_comma", 1.0),
            ("digit_space", 1.1),
            ("sino_space", 1.15),
            ("sino_period", 1.0),
            ("digit_space", 0.85),
            ("sino_space", 0.9),
            ("sino_comma", 1.1),
            ("digit_space", 1.0),
            ("sino_period", 1.1),
        ]
    elif 10 in values and len(values) <= 3:
        schedule = [
            ("sino_space", 1.1),
            ("sino_space", 1.25),
            ("sino_space", 1.1),
            ("sino_space", 1.25),
            ("sino_space", 1.0),
            ("sino_space", 0.95),
        ]

    if schedule:
        prompt_kind, speed_factor = schedule[candidate_index % len(schedule)]
        scheduled_spec = find_spec(prompt_kind, speed_factor)
        if scheduled_spec is not None:
            return scheduled_spec
    return prompt_specs[candidate_index % len(prompt_specs)]


def _countdown_audio_timing_qc(
    audio: np.ndarray,
    sample_rate: int,
    *,
    expected_units: int = 0,
) -> dict[str, Any]:
    detected = _countdown_detect_active_islands(audio, sample_rate)
    if not detected["ok"]:
        return {
            "gap_gate": "fail",
            "max_gap_sec": 0.0,
            "mean_gap_sec": 0.0,
            "gap_cv": 0.0,
            "active_island_count": 0,
            "expected_units": expected_units,
        }
    islands = detected["islands_sec"]
    gaps = detected["gaps_sec"]
    max_gap = max(gaps) if gaps else 0.0
    mean_gap = float(np.mean(gaps)) if gaps else 0.0
    gap_cv = float(np.std(gaps) / mean_gap) if len(gaps) > 1 and mean_gap > 0 else 0.0
    if max_gap <= 0.60:
        gate = "pass"
    elif max_gap <= 0.90:
        gate = "warn"
    else:
        gate = "fail"
    return {
        "gap_gate": gate,
        "max_gap_sec": round(max_gap, 6),
        "mean_gap_sec": round(mean_gap, 6),
        "gap_cv": round(gap_cv, 6),
        "active_island_count": len(islands),
        "expected_units": expected_units,
        "threshold_db": round(float(detected["threshold_db"]), 3),
        "active_islands": [[round(start, 4), round(end, 4)] for start, end in islands[:24]],
        "internal_gaps": [round(gap, 4) for gap in gaps[:24]],
    }


def _countdown_detect_active_islands(
    audio: np.ndarray,
    sample_rate: int,
) -> dict[str, Any]:
    if sample_rate <= 0 or len(audio) <= 0:
        return {
            "ok": False,
            "threshold_db": -120.0,
            "islands_sec": [],
            "islands_frames": [],
            "gaps_sec": [],
        }
    envelope_source = np.max(np.abs(audio), axis=1) if audio.ndim > 1 else np.abs(audio)
    frame = max(1, int(round(sample_rate * 0.025)))
    hop = max(1, int(round(sample_rate * 0.010)))
    if len(envelope_source) < frame:
        rms = np.array([float(np.sqrt(np.mean(envelope_source * envelope_source) + 1e-12))])
        times = np.array([len(envelope_source) / (2.0 * sample_rate)])
    else:
        rms_values: list[float] = []
        time_values: list[float] = []
        for start in range(0, len(envelope_source) - frame + 1, hop):
            chunk = envelope_source[start : start + frame]
            rms_values.append(float(np.sqrt(np.mean(chunk * chunk) + 1e-12)))
            time_values.append((start + frame / 2.0) / sample_rate)
        rms = np.asarray(rms_values, dtype=np.float64)
        times = np.asarray(time_values, dtype=np.float64)

    db = 20.0 * np.log10(rms + 1e-9)
    floor_db = float(np.percentile(db, 15))
    peak_db = float(np.percentile(db, 95))
    threshold_db = max(floor_db + 0.38 * (peak_db - floor_db), peak_db - 24.0, -55.0)
    active = db > threshold_db
    max_fill_gap_frames = max(1, int(round(0.08 / 0.010)))
    index = 0
    while index < len(active):
        if active[index]:
            index += 1
            continue
        end = index
        while end < len(active) and not active[end]:
            end += 1
        if index > 0 and end < len(active) and (end - index) <= max_fill_gap_frames:
            active[index:end] = True
        index = end

    islands: list[tuple[float, float]] = []
    island_frames: list[tuple[int, int]] = []
    duration = len(envelope_source) / sample_rate
    index = 0
    while index < len(active):
        if not active[index]:
            index += 1
            continue
        end = index
        while end < len(active) and active[end]:
            end += 1
        start_sec = max(0.0, float(times[index]) - frame / (2.0 * sample_rate))
        end_sec = min(duration, float(times[end - 1]) + frame / (2.0 * sample_rate))
        if end_sec - start_sec >= 0.045:
            islands.append((start_sec, end_sec))
            start_frame = max(0, min(len(envelope_source), int(round(start_sec * sample_rate))))
            end_frame = max(start_frame, min(len(envelope_source), int(round(end_sec * sample_rate))))
            island_frames.append((start_frame, end_frame))
        index = end

    gaps = [islands[i][0] - islands[i - 1][1] for i in range(1, len(islands))]
    return {
        "ok": True,
        "threshold_db": threshold_db,
        "islands_sec": islands,
        "islands_frames": island_frames,
        "gaps_sec": gaps,
    }


def _countdown_active_anchor_fit_from_audio(
    audio: np.ndarray,
    sample_rate: int,
    *,
    token_slot_sec: float | None = None,
) -> dict[str, Any]:
    data = _countdown_stereo(np.asarray(audio, dtype=np.float32))
    duration_sec = len(data) / sample_rate if sample_rate > 0 else 0.0
    detected = _countdown_detect_active_islands(data, sample_rate)
    islands = detected.get("islands_sec", []) if isinstance(detected, dict) else []
    gaps = detected.get("gaps_sec", []) if isinstance(detected, dict) else []
    if islands:
        active_start_sec = max(0.0, float(islands[0][0]))
        active_end_sec = min(duration_sec, float(islands[-1][1]))
        detection_gate = "pass"
    else:
        active_start_sec = duration_sec
        active_end_sec = duration_sec
        detection_gate = "fail"
    leading_silence_sec = max(0.0, active_start_sec)
    trailing_silence_sec = max(0.0, duration_sec - active_end_sec)
    post_anchor_duration_sec = max(0.0, duration_sec - leading_silence_sec)
    slot_sec = max(0.0, float(token_slot_sec or 0.0))
    next_anchor_overflow_sec = (
        max(0.0, post_anchor_duration_sec - slot_sec) if slot_sec > 0.0 else 0.0
    )
    return {
        "gate": detection_gate,
        "duration_sec": round(duration_sec, 6),
        "token_slot_sec": round(slot_sec, 6) if slot_sec > 0.0 else None,
        "active_start_sec": round(active_start_sec, 6),
        "active_end_sec": round(active_end_sec, 6),
        "active_duration_sec": round(max(0.0, active_end_sec - active_start_sec), 6),
        "leading_silence_sec": round(leading_silence_sec, 6),
        "trailing_silence_sec": round(trailing_silence_sec, 6),
        "post_anchor_duration_sec": round(post_anchor_duration_sec, 6),
        "next_anchor_overflow_sec": round(next_anchor_overflow_sec, 6),
        "pre_anchor_underflow_sec": 0.0,
        "active_island_count": len(islands),
        "max_internal_gap_sec": round(max(gaps), 6) if gaps else 0.0,
        "threshold_db": round(float(detected.get("threshold_db", -120.0)), 3)
        if isinstance(detected, dict)
        else -120.0,
    }


def _countdown_candidate_anchor_fit(candidate: dict[str, Any]) -> dict[str, Any] | None:
    carrier_bank = candidate.get("payload", {}).get("countdown_carrier_bank", {})
    if not isinstance(carrier_bank, dict):
        return None
    fit = carrier_bank.get("active_anchor_fit")
    return fit if isinstance(fit, dict) else None


def _countdown_candidate_anchor_fit_score(candidate: dict[str, Any]) -> float | None:
    fit = _countdown_candidate_anchor_fit(candidate)
    if fit is None:
        return None
    leading_silence_sec = max(0.0, _safe_float(fit.get("leading_silence_sec"), 0.0))
    trailing_silence_sec = max(0.0, _safe_float(fit.get("trailing_silence_sec"), 0.0))
    next_anchor_overflow_sec = max(
        0.0,
        _safe_float(fit.get("next_anchor_overflow_sec"), 0.0),
    )
    pre_anchor_underflow_sec = max(
        0.0,
        _safe_float(fit.get("pre_anchor_underflow_sec"), 0.0),
    )
    active_island_count = max(0, int(_safe_float(fit.get("active_island_count"), 1.0)))
    leading_score = 1.0 - min(max(0.0, leading_silence_sec - 0.06) / 0.18, 1.0)
    trailing_score = 1.0 - min(max(0.0, trailing_silence_sec - 0.08) / 0.24, 1.0)
    anchor_overflow = next_anchor_overflow_sec + pre_anchor_underflow_sec
    overflow_score = 1.0 - min(anchor_overflow / 0.12, 1.0)
    island_score = 1.0 if active_island_count <= 1 else 0.85 if active_island_count == 2 else 0.65
    score = (
        leading_score * 0.45
        + trailing_score * 0.20
        + overflow_score * 0.30
        + island_score * 0.05
    )
    return round(max(0.0, min(score, 1.0)), 6)


def _countdown_candidate_transcript_preference_score(
    token_text: str,
    candidate: dict[str, Any],
) -> float:
    pronunciation_qc = candidate.get("payload", {}).get("pronunciation_qc", {})
    if not isinstance(pronunciation_qc, dict):
        return 1.0
    transcript = str(pronunciation_qc.get("transcript") or "")
    expected = normalize_korean_pronunciation_text(token_text)
    observed = normalize_korean_pronunciation_text(transcript)
    if not expected or not observed:
        return 1.0
    if observed == expected:
        return 1.0
    if expected == "일" and observed == "예":
        return 0.72
    if expected == "육" and observed in {"유", "류"}:
        return 0.78
    if expected in observed:
        extra_units = max(0, len(observed) - len(expected))
        return round(max(0.62, 0.92 - extra_units * 0.12), 6)
    matched = _countdown_lcs_length(expected, observed)
    return round(max(0.0, min((matched / max(len(expected), 1)) * 0.65, 0.65)), 6)


def _countdown_anchor_aligned_start_frame(
    *,
    slot_start_frame: int,
    active_leading_frames: int,
    total_frames: int,
) -> int:
    return max(
        0,
        min(int(slot_start_frame) - max(0, int(active_leading_frames)), max(0, int(total_frames))),
    )


def _countdown_fade_chunk_edges(
    audio: np.ndarray,
    sample_rate: int,
    *,
    fade_sec: float = 0.005,
) -> np.ndarray:
    chunk = np.array(audio, dtype=np.float32, copy=True)
    if len(chunk) <= 2 or sample_rate <= 0:
        return chunk
    fade_frames = min(int(round(sample_rate * fade_sec)), max(0, len(chunk) // 2))
    if fade_frames <= 1:
        return chunk
    fade_in = np.linspace(0.0, 1.0, fade_frames, dtype=np.float32)
    fade_out = fade_in[::-1]
    if chunk.ndim == 1:
        chunk[:fade_frames] *= fade_in
        chunk[-fade_frames:] *= fade_out
    else:
        chunk[:fade_frames] *= fade_in[:, None]
        chunk[-fade_frames:] *= fade_out[:, None]
    return chunk


def _countdown_retime_phrase_audio(
    audio: np.ndarray,
    sample_rate: int,
    *,
    expected_units: int = 0,
    enabled: bool = True,
    trigger_max_gap_sec: float = 0.55,
    trigger_gap_cv: float = 0.35,
    target_min_gap_sec: float = 0.14,
    target_max_gap_sec: float = 0.28,
    leading_sec: float = 0.08,
    trailing_sec: float = 0.12,
) -> tuple[np.ndarray, dict[str, Any]]:
    original = _countdown_stereo(np.asarray(audio, dtype=np.float32))
    pre_qc = _countdown_audio_timing_qc(original, sample_rate, expected_units=expected_units)
    base_meta: dict[str, Any] = {
        "enabled": bool(enabled),
        "applied": False,
        "method": "humanized_cap_280ms",
        "pre_timing_qc": pre_qc,
        "post_timing_qc": pre_qc,
        "original_duration_sec": round(len(original) / sample_rate, 6) if sample_rate > 0 else 0.0,
        "retimed_duration_sec": round(len(original) / sample_rate, 6) if sample_rate > 0 else 0.0,
        "duration_ratio": 1.0,
        "expected_units": expected_units,
    }
    if not enabled:
        base_meta["reason"] = "disabled"
        return np.array(original, copy=True), base_meta
    if sample_rate <= 0 or len(original) <= 0:
        base_meta["reason"] = "empty_audio"
        return np.array(original, copy=True), base_meta

    detected = _countdown_detect_active_islands(original, sample_rate)
    islands = detected["islands_frames"] if detected["ok"] else []
    gaps = detected["gaps_sec"] if detected["ok"] else []
    if len(islands) <= 1 or not gaps:
        base_meta["reason"] = "insufficient_active_islands"
        base_meta["active_island_count"] = len(islands)
        return np.array(original, copy=True), base_meta

    max_gap = float(pre_qc.get("max_gap_sec") or 0.0)
    gap_cv = float(pre_qc.get("gap_cv") or 0.0)
    if max_gap <= trigger_max_gap_sec and gap_cv <= trigger_gap_cv:
        base_meta["reason"] = "timing_within_gate"
        base_meta["active_island_count"] = len(islands)
        return np.array(original, copy=True), base_meta

    min_gap = max(0.0, float(target_min_gap_sec))
    max_target_gap = max(min_gap, float(target_max_gap_sec))
    median_gap = float(np.median(gaps)) if gaps else min_gap
    target_gap = max(min_gap, min(max_target_gap, median_gap))
    jitter_pattern = (-0.10, 0.07, 0.12, -0.05, 0.03, -0.08, 0.09, -0.02)
    gap_sequence = [
        max(0.10, min(max_target_gap, target_gap * (1.0 + jitter_pattern[index % len(jitter_pattern)])))
        for index in range(len(gaps))
    ]

    pieces: list[np.ndarray] = []
    channels = original.shape[1]
    lead_frames = max(0, int(round(sample_rate * max(0.0, leading_sec))))
    trail_frames = max(0, int(round(sample_rate * max(0.0, trailing_sec))))
    if lead_frames:
        pieces.append(np.zeros((lead_frames, channels), dtype=np.float32))
    for index, (start_frame, end_frame) in enumerate(islands):
        chunk = original[start_frame:end_frame]
        if len(chunk):
            pieces.append(_countdown_fade_chunk_edges(chunk, sample_rate))
        if index < len(gap_sequence):
            gap_frames = max(1, int(round(sample_rate * gap_sequence[index])))
            pieces.append(np.zeros((gap_frames, channels), dtype=np.float32))
    if trail_frames:
        pieces.append(np.zeros((trail_frames, channels), dtype=np.float32))

    retimed = np.concatenate(pieces, axis=0) if pieces else np.array(original, copy=True)
    peak = float(np.max(np.abs(retimed))) if retimed.size else 0.0
    if peak > 0.98:
        retimed *= 0.98 / peak
    post_qc = _countdown_audio_timing_qc(retimed, sample_rate, expected_units=expected_units)
    ratio = len(retimed) / max(len(original), 1)
    if len(retimed) <= 0 or ratio > 1.25 or float(post_qc.get("max_gap_sec") or 0.0) > max_gap:
        base_meta.update(
            {
                "reason": "retime_guard_rejected",
                "post_timing_qc": post_qc,
                "target_gap_sec": round(target_gap, 6),
                "gap_sequence_sec": [round(gap, 6) for gap in gap_sequence],
                "active_island_count": len(islands),
            }
        )
        return np.array(original, copy=True), base_meta

    base_meta.update(
        {
            "applied": True,
            "reason": "long_internal_gap_retimed",
            "post_timing_qc": post_qc,
            "target_gap_sec": round(target_gap, 6),
            "gap_sequence_sec": [round(gap, 6) for gap in gap_sequence],
            "active_island_count": len(islands),
            "retimed_duration_sec": round(len(retimed) / sample_rate, 6),
            "duration_ratio": round(ratio, 6),
            "leading_sec": round(max(0.0, leading_sec), 6),
            "trailing_sec": round(max(0.0, trailing_sec), 6),
        }
    )
    return retimed.astype(np.float32, copy=False), base_meta


def _countdown_pack_take_selection_score(
    metadata: dict[str, Any],
    *,
    target_duration_sec: float,
) -> float:
    if not bool(metadata.get("approved")):
        return 0.0
    sequence_qc = metadata.get("sequence_qc") if isinstance(metadata.get("sequence_qc"), dict) else {}
    if not bool(sequence_qc.get("exact_match")):
        return 0.0
    pronunciation_gate = str(metadata.get("pronunciation_gate") or "").strip().lower()
    pronunciation_score = 1.0 if pronunciation_gate == "pass" else 0.55 if pronunciation_gate == "warn" else 0.0
    timing_qc = metadata.get("timing_qc") if isinstance(metadata.get("timing_qc"), dict) else {}
    gap_gate = str(timing_qc.get("gap_gate") or "").strip().lower()
    if gap_gate == "pass":
        gap_score = 1.0
    elif gap_gate == "warn":
        gap_score = 0.55
    else:
        gap_score = max(0.0, 1.0 - min(_safe_float(timing_qc.get("max_gap_sec"), 1.2) / 1.2, 1.0))
    duration = _safe_float(metadata.get("duration_sec"), 0.0)
    if target_duration_sec > 0.0 and duration > 0.0:
        ratio = duration / target_duration_sec
        duration_score = max(0.0, 1.0 - min(abs(ratio - 1.0) / 0.75, 1.0))
    else:
        duration_score = 0.6
    return round(pronunciation_score * 0.35 + duration_score * 0.25 + gap_score * 0.40, 6)


def _countdown_sequence_qc(
    values: list[int],
    pronunciation_qc: dict[str, Any] | None,
    *,
    expected_text: str | None = None,
) -> dict[str, Any]:
    tokens = countdown_korean_tokens(values) or []
    resolved_expected_text = expected_text or _countdown_space_phrase_tts_text(tokens)
    expected_normalized = normalize_korean_pronunciation_text(resolved_expected_text)
    transcript = str((pronunciation_qc or {}).get("transcript") or "").strip()
    observed_normalized = normalize_korean_pronunciation_text(transcript)
    source_gate = str((pronunciation_qc or {}).get("gate") or "").strip().lower()
    issues: list[str] = []
    exact_match = bool(expected_normalized) and observed_normalized == expected_normalized

    if not expected_normalized:
        gate = "unavailable"
        issues.append("empty_countdown_expected_text")
    elif not observed_normalized:
        gate = "unavailable"
        issues.append("empty_countdown_transcript")
    elif exact_match:
        gate = "pass"
    else:
        gate = "fail"
        issues.append("countdown_sequence_mismatch")

    return {
        "gate": gate,
        "exact_match": exact_match,
        "issues": issues,
        "expected_values": list(values),
        "expected_text": resolved_expected_text,
        "expected_normalized": expected_normalized,
        "observed_transcript": transcript,
        "observed_normalized": observed_normalized,
        "source_pronunciation_gate": source_gate or "not_run",
    }


def _countdown_carrier_template_values(token_text: str) -> dict[str, str]:
    return {
        "token": token_text,
        "digit": _COUNTDOWN_TOKEN_DIGITS.get(token_text, token_text),
        "phonetic": _COUNTDOWN_TOKEN_PHONETIC.get(token_text, token_text),
    }


def _countdown_has_token_specific_carrier_templates(cfg: Any, token_text: str) -> bool:
    token_template_map = getattr(cfg, "gsv_countdown_carrier_token_templates", {}) or {}
    if not isinstance(token_template_map, dict):
        return False
    return any(str(template).strip() for template in token_template_map.get(token_text, []))


def _countdown_carrier_templates_for_token_config(
    cfg: Any,
    token_text: str,
) -> list[tuple[int, str, str]]:
    token_template_map = getattr(cfg, "gsv_countdown_carrier_token_templates", {}) or {}
    token_templates = (
        [
            str(template).strip()
            for template in token_template_map.get(token_text, [])
            if str(template).strip()
        ]
        if isinstance(token_template_map, dict)
        else []
    )
    using_token_templates = bool(token_templates)
    if using_token_templates:
        templates = token_templates
    else:
        numeric_templates = (
            [
                str(template).strip()
                for template in getattr(
                    cfg,
                    "gsv_countdown_carrier_numeric_unit_templates",
                    [],
                )
                if str(template).strip()
            ]
            if bool(getattr(cfg, "gsv_countdown_carrier_numeric_unit_enabled", True))
            else []
        )
        templates = [
            str(template).strip()
            for template in getattr(cfg, "gsv_countdown_carrier_templates", [])
            if str(template).strip()
        ]
        templates = numeric_templates + templates
    if not templates:
        templates = ["숫자만 조용히 말해요. {token}. 다시."]
    templates = list(dict.fromkeys(templates))
    rendered: list[tuple[int, str, str]] = []
    template_values = _countdown_carrier_template_values(token_text)
    for carrier_index, template in enumerate(templates):
        if "{" in template:
            try:
                carrier_text = template.format(**template_values)
            except Exception:
                carrier_text = template.replace("{token}", token_text)
        elif using_token_templates:
            carrier_text = template
        else:
            carrier_text = f"{template} {token_text}".strip()
        rendered.append((carrier_index, template, carrier_text))
    return rendered


def _countdown_alignment_units(text: str) -> float:
    units = float(len(_COUNTDOWN_ALIGNMENT_UNIT_RE.findall(text)))
    units += 0.35 * sum(1 for char in text if char in ",，、.。!?！？…")
    return units


def _countdown_carrier_text_parts(
    carrier_text: str,
    token_text: str,
) -> tuple[str, str, str] | None:
    def standalone_match(text: str, token: str) -> re.Match[str] | None:
        if not token:
            return None
        pattern = re.compile(
            rf"(?<![0-9A-Za-z가-힣]){re.escape(token)}(?![0-9A-Za-z가-힣])"
        )
        matches = list(pattern.finditer(text))
        return matches[-1] if matches else None

    raw_match = standalone_match(carrier_text, token_text)
    if raw_match is not None:
        raw_index = raw_match.start()
        raw_end = raw_match.end()
        return (
            carrier_text[:raw_index],
            carrier_text[raw_index:raw_end],
            carrier_text[raw_end:],
        )
    raw_index = carrier_text.rfind(token_text)
    if raw_index >= 0:
        raw_end = raw_index + len(token_text)
        return (
            carrier_text[:raw_index],
            carrier_text[raw_index:raw_end],
            carrier_text[raw_end:],
        )
    normalized_carrier = normalize_korean_pronunciation_text(carrier_text)
    normalized_token = normalize_korean_pronunciation_text(token_text)
    normalized_match = standalone_match(normalized_carrier, normalized_token)
    if normalized_match is not None:
        normalized_index = normalized_match.start()
        normalized_end = normalized_match.end()
        return (
            normalized_carrier[:normalized_index],
            normalized_carrier[normalized_index:normalized_end],
            normalized_carrier[normalized_end:],
        )
    normalized_index = normalized_carrier.rfind(normalized_token)
    if normalized_index < 0:
        return None
    normalized_end = normalized_index + len(normalized_token)
    return (
        normalized_carrier[:normalized_index],
        normalized_carrier[normalized_index:normalized_end],
        normalized_carrier[normalized_end:],
    )


def _countdown_token_alignment_from_chunks(
    expected_text: str,
    chunks: list[Any],
    candidate_duration_sec: float,
) -> dict[str, Any] | None:
    expected = normalize_korean_pronunciation_text(expected_text)
    if not expected:
        return None

    def normalized_contains(value: str) -> bool:
        normalized = normalize_korean_pronunciation_text(value)
        if not normalized:
            return False
        if len(expected) == 1:
            return expected == normalized
        return expected == normalized or expected in normalized

    max_token_span_sec = min(0.75, max(0.16, candidate_duration_sec * 0.55))
    for chunk in chunks:
        for word in getattr(chunk, "words", []) or []:
            if not normalized_contains(str(getattr(word, "text", ""))):
                continue
            start = max(0.0, float(getattr(word, "start", 0.0)))
            end = min(candidate_duration_sec, float(getattr(word, "end", 0.0)))
            if end > start:
                return {
                    "source": "asr_word",
                    "start_sec": round(start, 6),
                    "end_sec": round(end, 6),
                    "text": getattr(word, "text", ""),
                }
        if not normalized_contains(str(getattr(chunk, "text", ""))):
            continue
        start = max(0.0, float(getattr(chunk, "start", 0.0)))
        end = min(candidate_duration_sec, float(getattr(chunk, "end", 0.0)))
        if end > start and (end - start) <= max_token_span_sec:
            return {
                "source": "asr_chunk",
                "start_sec": round(start, 6),
                "end_sec": round(end, 6),
                "text": getattr(chunk, "text", ""),
            }
    return None


def _countdown_pitch_profile(audio: np.ndarray, sample_rate: int) -> dict[str, Any]:
    """Extract a compact pitch/energy profile for a short countdown token."""
    data = _countdown_stereo(audio)
    texture_profile = _countdown_texture_profile(data, sample_rate)
    mono = to_mono(data).astype(np.float32, copy=False)
    if len(mono) == 0 or sample_rate <= 0:
        return {"available": False, "reason": "empty_audio", "texture_profile": texture_profile}
    frame_len = min(len(mono), max(128, int(round(sample_rate * 0.04))))
    hop = max(64, int(round(sample_rate * 0.02)))
    min_lag = max(1, int(round(sample_rate / 500.0)))
    max_lag = max(min_lag + 1, int(round(sample_rate / 70.0)))
    if frame_len <= max_lag:
        frame_len = min(len(mono), max_lag + 1)
    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    rms_threshold = max(peak * 0.05, 1e-5)
    f0_values: list[float] = []
    f0_times: list[float] = []
    rms_values: list[float] = []
    frame_starts = range(0, max(1, len(mono) - frame_len + 1), hop)
    for start in frame_starts:
        frame = mono[start : start + frame_len]
        if len(frame) < frame_len:
            break
        rms_value = float(np.sqrt(np.mean(np.square(frame))))
        if rms_value <= rms_threshold:
            continue
        rms_values.append(rms_value)
        centered = frame - float(np.mean(frame))
        if not np.any(centered):
            continue
        windowed = centered * np.hanning(len(centered))
        corr = np.correlate(windowed, windowed, mode="full")[len(windowed) - 1 :]
        if len(corr) <= max_lag or float(corr[0]) <= 0.0:
            continue
        search = corr[min_lag:max_lag]
        if search.size == 0:
            continue
        best_offset = int(np.argmax(search))
        best_lag = min_lag + best_offset
        confidence = float(search[best_offset]) / max(float(corr[0]), 1e-12)
        if confidence < 0.25:
            continue
        f0_values.append(sample_rate / best_lag)
        f0_times.append((start + frame_len / 2) / sample_rate)
    if not f0_values:
        return {
            "available": False,
            "reason": "no_voiced_frames",
            "active_frame_count": len(rms_values),
            "duration_sec": round(len(mono) / sample_rate, 6),
            "texture_profile": texture_profile,
        }
    f0_array = np.asarray(f0_values, dtype=np.float32)
    median_f0 = float(np.median(f0_array))
    f0_range = float(np.percentile(f0_array, 90) - np.percentile(f0_array, 10))
    if len(f0_values) >= 2 and max(f0_times) > min(f0_times):
        slope = float(np.polyfit(np.asarray(f0_times, dtype=np.float32), f0_array, 1)[0])
    else:
        slope = 0.0
    rms = float(np.sqrt(np.mean(np.square(mono)))) if mono.size else 0.0
    rms_db = 20.0 * float(np.log10(max(rms, 1e-12)))
    return {
        "available": True,
        "median_f0_hz": round(median_f0, 3),
        "f0_range_hz": round(f0_range, 3),
        "f0_slope_hz_per_sec": round(slope, 3),
        "rms_dbfs": round(rms_db, 3),
        "voiced_frame_count": len(f0_values),
        "active_frame_count": len(rms_values),
        "duration_sec": round(len(mono) / sample_rate, 6),
        "texture_profile": texture_profile,
    }


def _countdown_texture_profile(audio: np.ndarray, sample_rate: int) -> dict[str, Any]:
    data = _countdown_stereo(audio)
    mono = to_mono(data).astype(np.float32, copy=False)
    if len(mono) == 0 or sample_rate <= 0:
        return {"available": False, "reason": "empty_audio"}
    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(mono)))) if mono.size else 0.0
    rms_db = 20.0 * float(np.log10(max(rms, 1e-12)))
    frame_len = min(len(mono), max(256, int(round(sample_rate * 0.04))))
    if frame_len <= 1:
        return {"available": False, "reason": "audio_too_short"}
    frame = mono[:frame_len] * np.hanning(frame_len)
    magnitude = np.abs(np.fft.rfft(frame)).astype(np.float64)
    magnitude_sum = float(np.sum(magnitude))
    freqs = np.fft.rfftfreq(frame_len, d=1.0 / sample_rate)
    if magnitude_sum <= 1e-12:
        centroid = 0.0
        high_freq_ratio = 0.0
        flatness = 0.0
    else:
        centroid = float(np.sum(freqs * magnitude) / magnitude_sum)
        high_freq_ratio = float(np.sum(magnitude[freqs >= 4000.0]) / magnitude_sum)
        flatness = float(np.exp(np.mean(np.log(magnitude + 1e-12))) / max(np.mean(magnitude), 1e-12))
    zcr = float(np.mean(np.abs(np.diff(np.signbit(mono))).astype(np.float32))) if len(mono) > 1 else 0.0
    left = data[:, 0].astype(np.float32, copy=False)
    right = data[:, 1].astype(np.float32, copy=False)
    mid = (left + right) * 0.5
    side = (left - right) * 0.5
    stereo_width = float(
        np.sqrt(np.mean(np.square(side))) / max(np.sqrt(np.mean(np.square(mid))), 1e-8)
    )
    return {
        "available": True,
        "peak": round(peak, 6),
        "rms_dbfs": round(rms_db, 3),
        "spectral_centroid_hz": round(centroid, 3),
        "high_freq_ratio": round(max(0.0, min(high_freq_ratio, 1.0)), 6),
        "spectral_flatness": round(max(0.0, min(flatness, 1.0)), 6),
        "zero_crossing_rate": round(max(0.0, min(zcr, 1.0)), 6),
        "stereo_width": round(max(0.0, min(stereo_width, 2.0)), 6),
    }


def _countdown_texture_match(
    source_texture: dict[str, Any] | None,
    candidate_texture: dict[str, Any] | None,
) -> dict[str, Any]:
    if not source_texture or not candidate_texture:
        return {"gate": "unavailable", "score": 1.0, "reason": "missing_texture_profile"}
    if not source_texture.get("available") or not candidate_texture.get("available"):
        return {
            "gate": "unavailable",
            "score": 1.0,
            "reason": "unavailable_texture_profile",
            "source": source_texture,
            "candidate": candidate_texture,
        }

    def bounded_delta_score(left: float, right: float, scale: float) -> float:
        return max(0.0, 1.0 - min(abs(left - right) / max(scale, 1e-8), 1.0))

    energy_score = bounded_delta_score(
        float(source_texture.get("rms_dbfs") or -120.0),
        float(candidate_texture.get("rms_dbfs") or -120.0),
        18.0,
    )
    centroid_scale = max(800.0, float(source_texture.get("spectral_centroid_hz") or 0.0) + 800.0)
    brightness_score = bounded_delta_score(
        float(source_texture.get("spectral_centroid_hz") or 0.0),
        float(candidate_texture.get("spectral_centroid_hz") or 0.0),
        centroid_scale,
    )
    high_score = bounded_delta_score(
        float(source_texture.get("high_freq_ratio") or 0.0),
        float(candidate_texture.get("high_freq_ratio") or 0.0),
        0.25,
    )
    flatness_score = bounded_delta_score(
        float(source_texture.get("spectral_flatness") or 0.0),
        float(candidate_texture.get("spectral_flatness") or 0.0),
        0.35,
    )
    zcr_score = bounded_delta_score(
        float(source_texture.get("zero_crossing_rate") or 0.0),
        float(candidate_texture.get("zero_crossing_rate") or 0.0),
        0.25,
    )
    width_score = bounded_delta_score(
        float(source_texture.get("stereo_width") or 0.0),
        float(candidate_texture.get("stereo_width") or 0.0),
        0.6,
    )
    score = (
        energy_score * 0.20
        + brightness_score * 0.25
        + high_score * 0.15
        + flatness_score * 0.15
        + zcr_score * 0.10
        + width_score * 0.15
    )
    return {
        "gate": "pass" if score >= 0.72 else "warn" if score >= 0.45 else "fail",
        "score": round(score, 6),
        "energy_score": round(energy_score, 6),
        "brightness_score": round(brightness_score, 6),
        "high_freq_score": round(high_score, 6),
        "flatness_score": round(flatness_score, 6),
        "zero_crossing_score": round(zcr_score, 6),
        "stereo_width_score": round(width_score, 6),
        "source": source_texture,
        "candidate": candidate_texture,
    }


def _countdown_prosody_match(
    source_profile: dict[str, Any] | None,
    candidate_profile: dict[str, Any] | None,
    *,
    max_median_semitone_error: float,
    pass_score: float,
    warn_score: float,
) -> dict[str, Any]:
    if not source_profile or not candidate_profile:
        return {"gate": "unavailable", "score": 1.0, "reason": "missing_profile"}
    if not source_profile.get("available") or not candidate_profile.get("available"):
        return {
            "gate": "unavailable",
            "score": 1.0,
            "reason": "unavailable_profile",
            "source": source_profile,
            "candidate": candidate_profile,
        }
    source_f0 = float(source_profile.get("median_f0_hz") or 0.0)
    candidate_f0 = float(candidate_profile.get("median_f0_hz") or 0.0)
    if source_f0 <= 0.0 or candidate_f0 <= 0.0:
        return {
            "gate": "unavailable",
            "score": 1.0,
            "reason": "missing_f0",
            "source": source_profile,
            "candidate": candidate_profile,
        }
    semitone_error = abs(12.0 * float(np.log2(candidate_f0 / source_f0)))
    median_score = max(0.0, 1.0 - min(semitone_error / max_median_semitone_error, 1.0))
    source_slope = float(source_profile.get("f0_slope_hz_per_sec") or 0.0)
    candidate_slope = float(candidate_profile.get("f0_slope_hz_per_sec") or 0.0)
    slope_scale = max(60.0, abs(source_slope) + 60.0)
    slope_score = max(0.0, 1.0 - min(abs(candidate_slope - source_slope) / slope_scale, 1.0))
    source_rms = float(source_profile.get("rms_dbfs") or -120.0)
    candidate_rms = float(candidate_profile.get("rms_dbfs") or -120.0)
    energy_score = max(0.0, 1.0 - min(abs(candidate_rms - source_rms) / 18.0, 1.0))
    score = median_score * 0.72 + slope_score * 0.18 + energy_score * 0.10
    gate = "pass" if score >= pass_score else "warn" if score >= warn_score else "fail"
    return {
        "gate": gate,
        "score": round(score, 6),
        "median_f0_score": round(median_score, 6),
        "slope_score": round(slope_score, 6),
        "energy_score": round(energy_score, 6),
        "median_f0_semitone_error": round(semitone_error, 6),
        "source": source_profile,
        "candidate": candidate_profile,
    }


def _fit_audio_frames(data: np.ndarray, target_frames: int) -> np.ndarray:
    target_frames = max(1, int(target_frames))
    if len(data) == target_frames:
        return data
    return resample_linear(data, max(1, len(data)), target_frames)


def _countdown_stereo(data: np.ndarray) -> np.ndarray:
    if data.ndim == 1:
        data = data[:, None]
    if data.shape[1] == 1:
        return np.repeat(data, 2, axis=1)
    return data[:, :2]


def _omission_reasons_allow_source_pause_padding(candidate: TTSCandidate) -> list[str] | None:
    if not candidate.payload.get("omission_suspected"):
        return []
    reasons = [
        str(reason)
        for reason in candidate.payload.get("omission_detection", {}).get("reasons") or []
    ]
    if reasons and all(reason.startswith("duration_below_segment_ratio:") for reason in reasons):
        return reasons
    return None


def _duration_rewrite_log_preview(value: object, max_chars: int = 140) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "..."


def _log_duration_rewrite_result(segment_id: str, metadata: dict[str, Any]) -> None:
    accepted = str(bool(metadata.get("accepted"))).lower()
    before = _duration_rewrite_log_preview(metadata.get("before"))
    after = _duration_rewrite_log_preview(metadata.get("after") or metadata.get("normalized_text"))
    parts = [
        f"[cyan]synth duration-rewrite[/cyan] {escape(segment_id)}",
        f"reason={escape(str(metadata.get('reason') or 'unknown'))}",
        f"accepted={accepted}",
        (
            "chars="
            f"{metadata.get('current_speech_chars', '?')}->{metadata.get('speech_chars', '?')} "
            f"target={metadata.get('target_speech_chars', '?')} "
            f"range={metadata.get('min_speech_chars', '?')}-{metadata.get('max_speech_chars', '?')}"
        ),
        f'before="{escape(before)}"',
    ]
    if metadata.get("accepted_relaxed"):
        parts.append(f"relaxed={escape(str(metadata.get('relaxed_acceptance_reason') or 'true'))}")
    if "retry_scheduled" in metadata:
        retry = str(bool(metadata.get("retry_scheduled"))).lower()
        parts.append(f"retry={retry}")
    if after:
        parts.append(f'after="{escape(after)}"')
    if metadata.get("rejected_reasons"):
        reasons = ", ".join(str(reason) for reason in metadata["rejected_reasons"])
        parts.append(f"rejected={escape(_duration_rewrite_log_preview(reasons))}")
    if metadata.get("error"):
        parts.append(f"error={escape(_duration_rewrite_log_preview(metadata['error']))}")
    console.print(" ".join(parts))


def _maybe_relax_duration_rewrite_acceptance(metadata: dict[str, Any]) -> bool:
    if metadata.get("accepted"):
        return True
    if metadata.get("error") or metadata.get("reason") != "too_short":
        return False
    rejected_reasons = [str(reason) for reason in metadata.get("rejected_reasons") or []]
    if not rejected_reasons or any(
        not reason.startswith("speech_chars_below_min:") for reason in rejected_reasons
    ):
        return False
    try:
        current_chars = int(metadata["current_speech_chars"])
        speech_chars = int(metadata["speech_chars"])
        min_chars = int(metadata["min_speech_chars"])
    except (KeyError, TypeError, ValueError):
        return False
    short_by = min_chars - speech_chars
    if short_by < 1 or short_by > 2 or speech_chars <= current_chars:
        return False
    metadata["accepted"] = True
    metadata["accepted_relaxed"] = True
    metadata["relaxed_acceptance_reason"] = f"speech_chars_below_min_near_miss:{speech_chars}<{min_chars}"
    metadata["duration_rewrite_relaxed_shortfall_chars"] = short_by
    metadata["original_rejected_reasons"] = rejected_reasons
    metadata["rejected_reasons"] = []
    return True


def _should_retry_duration_rewrite_result(
    rewritten: JapaneseScript | None,
    current_script: JapaneseScript | None,
) -> bool:
    return rewritten is not None and current_script is not None and rewritten.tts_text != current_script.tts_text


def _numeric_render_plan_payload(plan: NumericRenderPlan) -> dict[str, Any]:
    return {
        "kind": str(getattr(plan.kind, "value", plan.kind)),
        "values": list(plan.values),
        "tokens": list(plan.tokens),
        "target_duration_sec": round(float(plan.target_duration_sec), 6),
        "text": plan.text,
        "text_variant": plan.text_variant,
        "render_policy": plan.render_policy,
        "groups": [list(group) for group in plan.groups],
    }


def _numeric_phrase_request_payload(
    plan: NumericRenderPlan,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    raw_request = (payload or {}).get("request")
    if isinstance(raw_request, dict):
        request = copy.deepcopy(raw_request)
    else:
        request = vars(build_numeric_phrase_request(plan, ref={})).copy()
    request.setdefault("text", plan.text)
    request.setdefault("text_split_method", "cut0")
    request.setdefault("top_k", 5)
    request.setdefault("top_p", 0.85)
    request.setdefault("temperature", 0.65)
    request.setdefault("repetition_penalty", 1.8)
    return request


def _numeric_phrase_candidate_generation_payload(
    request: dict[str, Any],
    payload: dict[str, Any] | None = None,
) -> dict[str, Any] | list[Any]:
    if isinstance(payload, dict):
        raw_generation = payload.get("candidate_generation")
        if isinstance(raw_generation, (dict, list)):
            return copy.deepcopy(raw_generation)
        numeric_phrase = payload.get("numeric_phrase")
        if isinstance(numeric_phrase, dict):
            raw_generation = numeric_phrase.get("candidate_generation")
            if isinstance(raw_generation, (dict, list)):
                return copy.deepcopy(raw_generation)
    return {
        "text_split_method": str(request.get("text_split_method") or "cut0"),
        "top_k": int(request.get("top_k") or 5),
        "top_p": float(request.get("top_p") or 0.85),
        "temperature": float(request.get("temperature") or 0.65),
        "repetition_penalty": float(request.get("repetition_penalty") or 1.8),
    }


def _numeric_phrase_placements_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    placements = payload.get("placements")
    if not isinstance(placements, list):
        numeric_phrase = payload.get("numeric_phrase")
        placements = (
            numeric_phrase.get("placements")
            if isinstance(numeric_phrase, dict)
            else []
        )
    if not isinstance(placements, list):
        return []
    return [
        copy.deepcopy(placement)
        for placement in placements
        if isinstance(placement, dict)
    ]


def _numeric_phrase_numeric_qc_payload(
    plan: NumericRenderPlan,
    payload: dict[str, Any],
) -> dict[str, Any]:
    for key in ("numeric_qc", "numeric_sequence_qc"):
        qc = payload.get(key)
        if isinstance(qc, dict):
            return copy.deepcopy(qc)
    numeric_phrase = payload.get("numeric_phrase")
    if isinstance(numeric_phrase, dict):
        for key in ("numeric_qc", "numeric_sequence_qc"):
            qc = numeric_phrase.get(key)
            if isinstance(qc, dict):
                return copy.deepcopy(qc)
    if bool(payload.get("mock")):
        return {
            "gate": "pass",
            "expected_values": list(plan.values),
            "observed_values": list(plan.values),
            "backend": "mock",
            "reason": "mock_numeric_phrase_renderer",
        }
    return {
        "gate": "unavailable",
        "expected_values": list(plan.values),
        "observed_values": [],
        "backend": "",
        "reason": "numeric_qc_not_recorded",
    }


def _pure_korean_numeric_phrase_values(text: str, *, min_values: int) -> list[int] | None:
    values = extract_korean_numeric_values(text)
    if len(values) < min_values:
        return None
    remainder = _NUMERIC_PHRASE_TOKEN_RE.sub("", unicodedata.normalize("NFKC", text or "").strip())
    if not _NUMERIC_PHRASE_ALLOWED_REMAINDER_RE.fullmatch(remainder):
        return None
    return values


def _korean_tts_allows_source_countdown_fallback(text: str, source_values: list[int]) -> bool:
    normalized = unicodedata.normalize("NFKC", text or "").strip()
    if not normalized:
        return True
    remainder = _NUMERIC_PHRASE_TOKEN_RE.sub("", normalized)
    if not _NUMERIC_PHRASE_ALLOWED_REMAINDER_RE.fullmatch(remainder):
        return False
    values = extract_korean_numeric_values(normalized)
    return not values or values == source_values


def _numeric_phrase_render_plan_for_segment(
    segment: Segment,
    cfg: ProjectConfig,
) -> NumericRenderPlan | None:
    if (
        not bool(getattr(cfg, "gsv_numeric_phrase_renderer_enabled", True))
        or not segment.script
        or _canonical_language(segment.script.tts_language) != "ko"
    ):
        return None
    event = segment.analysis.get(COUNTDOWN_EVENT_KEY)
    if isinstance(event, dict) and str(event.get("kind") or "") == "embedded_countdown":
        return None
    if _embedded_countdown_values(segment) is not None:
        return None
    min_values = int(getattr(cfg, "gsv_numeric_cadence_min_values", 3))
    values = _pure_korean_numeric_phrase_values(
        segment.script.tts_text,
        min_values=min_values,
    )
    if values is None:
        source_values = _source_countdown_values(segment)
        if (
            source_values is None
            or len(source_values) < min_values
            or not is_descending_countdown(source_values, min_values=min_values)
            or not _korean_tts_allows_source_countdown_fallback(
                segment.script.tts_text,
                source_values,
            )
        ):
            return None
        values = source_values
    return build_numeric_render_plan(values, target_duration_sec=float(segment.duration))


def _numeric_phrase_asr_backend_name(cfg: ProjectConfig) -> str:
    backend = str(getattr(cfg, "asr_backend", "faster_whisper")).strip().lower()
    return "qwen_asr" if backend == "qwen_asr" else backend.replace("-", "_")


def _numeric_phrase_asr_backend_config(cfg: ProjectConfig) -> dict[str, Any]:
    backend_config = _asr_backend_config(cfg)
    backend_config.update(
        {
            "language": "ko",
            "word_timestamps": True,
            "condition_on_previous_text": False,
            "vad_filter": True,
            "vad_parameters": {
                "min_silence_duration_ms": 80,
                "speech_pad_ms": 120,
            },
            "batched_inference": False,
            "initial_prompt": _NUMERIC_PHRASE_ASR_PROMPT,
            "hotwords": _NUMERIC_PHRASE_ASR_HOTWORDS,
        }
    )
    return backend_config


def _numeric_phrase_values_are_countdown(segment: Segment, values: list[int]) -> bool:
    if not is_descending_countdown(values):
        return False
    event_values = _countdown_event_values(segment)
    if event_values == values:
        return True
    return _source_countdown_values(segment) == values


def render_numeric_phrase_segment(
    *,
    project_dir: Path,
    segment: Segment,
    plan: NumericRenderPlan,
    cfg: ProjectConfig,
    mock: bool,
    lane_index: int = 0,
    gsv_url: str | None = None,
    ref: GPTSoVITSRef | None = None,
    client: GPTSoVITSClient | None = None,
) -> dict[str, Any]:
    """Render a pure numeric phrase segment or explicitly report failure.

    Mock mode keeps the deterministic synthetic route. Live mode delegates
    GPT-SoVITS phrase synthesis, ASR QC, and numeric bed rendering to the
    numeric phrase renderer helper.
    """
    _ = (lane_index, gsv_url)
    plan_payload = _numeric_render_plan_payload(plan)
    if not mock:
        if client is None:
            return {
                "status": "failed",
                "reason": "numeric_phrase_client_missing",
                "plan": plan_payload,
            }
        if ref is None:
            return {
                "status": "failed",
                "reason": "numeric_phrase_ref_missing",
                "plan": plan_payload,
            }
        output_path = (
            project_dir
            / "work"
            / "tts"
            / "numeric_phrase"
            / f"{segment.id}_numeric_phrase.wav"
        )
        work_dir = project_dir / "work" / "tts" / "numeric_phrase" / segment.id

        def asr_backend_factory() -> Any:
            return create_asr_backend(
                _numeric_phrase_asr_backend_name(cfg),
                _numeric_phrase_asr_backend_config(cfg),
            )

        return render_live_numeric_phrase(
            plan,
            client,
            output_path,
            ref=ref.model_dump(mode="json"),
            asr_backend_factory=asr_backend_factory,
            work_dir=work_dir,
            max_tempo_limit=float(getattr(cfg, "gsv_numeric_phrase_max_tempo", 1.1)),
            mock=False,
        )

    output_path = project_dir / "work" / "tts" / "numeric_phrase" / f"{segment.id}_numeric_phrase.wav"
    seed_material = f"{segment.id}:{','.join(str(value) for value in plan.values)}"
    seed = int(hashlib.sha256(seed_material.encode("utf-8")).hexdigest()[:8], 16)
    seed = int(getattr(cfg, "base_seed", 0)) + seed % 100_000
    _mock_synthesize(
        output_path,
        max(0.05, float(segment.duration)),
        seed,
        int(getattr(cfg, "mix_sample_rate", 48_000)),
    )
    actual_duration = duration_sec(output_path)
    candidate_ratio = duration_ratio(actual_duration, segment.duration)
    duration_gate = (
        "too_long"
        if duration_too_long(actual_duration, segment.duration, cfg.duration_tolerance)
        else "too_short"
        if duration_too_short(actual_duration, segment.duration, cfg.duration_tolerance)
        else "pass"
    )
    peak = peak_dbfs(output_path)
    rms = rms_dbfs(output_path)
    request = build_numeric_phrase_request(plan, ref=(ref.model_dump(mode="json") if ref else {}))
    request_payload = vars(request).copy()
    candidate_generation = _numeric_phrase_candidate_generation_payload(request_payload)
    numeric_qc = _numeric_phrase_numeric_qc_payload(plan, {"mock": True})
    payload = {
        "renderer": "numeric_phrase",
        "mock": True,
        "plan": plan_payload,
        "request": request_payload,
        "candidate_text": str(request_payload.get("text") or plan.text),
        "candidate_generation": candidate_generation,
        "values": list(plan.values),
        "tokens": list(plan.tokens),
        "text_variant": plan.text_variant,
        "render_policy": plan.render_policy,
        "target_duration_sec": round(float(segment.duration), 6),
        "duration_ratio": candidate_ratio,
        "duration_gate": duration_gate,
        "timing_quality": _gsv_timing_quality_payload(
            candidate_ratio,
            duration_gate,
            float(getattr(cfg, "gsv_timing_quality_tolerance", 0.10)),
            float(cfg.duration_tolerance),
        ),
        "audio_qc": {
            "gate": "silent" if peak <= -90.0 or rms <= -90.0 else "pass",
            "peak_dbfs": round(peak, 3),
            "rms_dbfs": round(rms, 3),
        },
        "numeric_qc": numeric_qc,
        "placements": [],
        "numeric_phrase": {
            **plan_payload,
            "max_tempo": 1.0,
            "max_tempo_limit": float(getattr(cfg, "gsv_numeric_phrase_max_tempo", 1.1)),
            "whole_lead_in_sec": float(
                getattr(cfg, "gsv_numeric_phrase_whole_lead_in_sec", 0.12)
            ),
            "tail_guard_sec": float(getattr(cfg, "gsv_numeric_phrase_tail_guard_sec", 0.16)),
            "numeric_qc": numeric_qc,
            "placements": [],
        },
    }
    return {
        "status": "rendered",
        "output_path": str(output_path),
        "duration_sec": actual_duration,
        "seed": seed,
        "payload": payload,
        "max_tempo": 1.0,
        "selection_reason": "numeric_phrase_mock",
    }


def run_synth_stage(ctx: PipelineContext, gsv_url: str | None, refs_path: Path, mock: bool = False, confirm_rights: bool = False, gpt_weights_path: str | None = None, sovits_weights_path: str | None = None, auto_gsv_server: bool | None = None, gsv_server_command: list[str] | str | None = None, use_trained_gpt: bool = False, only_segment_ids: set[str] | None = None, retry_failed: bool = False, force: bool = False, render_countdowns: bool = True, countdown_only: bool = False, stage_name: str = "synth") -> PipelineManifest:
    project_dir = ctx.project_dir
    manifest = ctx.reload_manifest()
    _load_config_into_manifest(project_dir, manifest)
    cfg = manifest.project_config
    if use_trained_gpt:
        cfg = cfg.model_copy(update={"gsv_gpt_weights_policy": "few_shot"})
        manifest.project_config = cfg
    total = len(manifest.segments)
    synth_backend_name = "mock" if mock else "gpt-sovits"
    _log_stage_start(stage_name, f"backend={synth_backend_name}, segments={total}")
    if not mock and not confirm_rights:
        raise RightsError(
            "Real GPT-SoVITS synthesis requires --confirm-rights for the current source and voice references."
        )
    if mock:
        _require_audio_stage_rights(manifest, stage_name, confirm_rights, metadata={"backend": "mock"})
    use_speaker_gsv = bool(cfg.gsv_speaker_models)
    if use_speaker_gsv:
        _validate_gsv_speaker_models(project_dir, manifest)
        refs: dict[str, GPTSoVITSRef] = {}
        refs_metadata: dict[str, object] = {
            "speaker_refs": {
                speaker_id: speaker_cfg.refs_path
                for speaker_id, speaker_cfg in sorted(cfg.gsv_speaker_models.items())
            }
        }
    else:
        refs = load_refs(refs_path, project_dir=project_dir)
        actual_refs_path = resolve_refs_json_path(refs_path, project_dir)
        refs_preflight: dict[str, Any] = {
            "refs": [],
            "invalid": [],
            "skipped": True,
            "reason": "countdown_only_or_non_api_client",
        }
        if not mock and not countdown_only and _uses_real_gsv_client():
            refs_preflight = _gsv_refs_duration_preflight(refs, cfg)
            if refs_preflight["invalid"]:
                from asmr_dub_pipeline.pipeline.stages.voice_refs import (
                    run_prepare_source_voice_refs_stage,
                )

                try:
                    run_prepare_source_voice_refs_stage(
                        ctx,
                        actual_refs_path,
                        confirm_rights=True,
                    )
                except Exception as exc:
                    raise GPTSoVITSError(
                        "valid source-derived reference not available: "
                        f"{exc}"
                    ) from exc
                manifest = ctx.reload_manifest()
                _load_config_into_manifest(project_dir, manifest)
                cfg = manifest.project_config
                refs = load_refs(actual_refs_path, project_dir=project_dir)
                refs_preflight = _gsv_refs_duration_preflight(refs, cfg)
                if refs_preflight["invalid"]:
                    invalid_styles = ", ".join(
                        str(row.get("style")) for row in refs_preflight["invalid"]
                    )
                    raise GPTSoVITSError(
                        "valid source-derived reference not available: "
                        f"{invalid_styles}"
                    )
        refs_metadata = _refs_audit_metadata(actual_refs_path, refs)
        if not mock:
            refs_metadata["refs_preflight"] = refs_preflight
    if not mock:
        manifest.rights_audit = require_existing_or_confirmed_rights(
            manifest.rights_audit,
            True,
            stage_name,
            _manifest_source_path(manifest),
            metadata={"backend": "gpt-sovits", **refs_metadata},
        )
    effective_gsv_url = gsv_url or cfg.gsv_url
    should_auto_start_server = (
        False if mock else cfg.gsv_auto_start if auto_gsv_server is None else auto_gsv_server
    )
    selected_total = (
        total
        if only_segment_ids is None
        else sum(1 for segment in manifest.segments if segment.id in only_segment_ids)
    )
    gsv_lane_count = 1 if mock else _effective_lane_count(cfg.gsv_concurrency, selected_total)
    gsv_base_urls = [effective_gsv_url] if mock else _parallel_base_urls(effective_gsv_url, gsv_lane_count)
    if not mock:
        _log_stage_checkpoint(
            stage_name,
            "gsv lanes",
            (
                f"concurrency={gsv_lane_count} auto_start={str(should_auto_start_server).lower()} "
                f"urls={','.join(gsv_base_urls)}"
            ),
        )
    server_managers: list[ManagedGPTSoVITSServer] = []
    if not mock:
        for lane_index, base_url in enumerate(gsv_base_urls):
            log_name = "api_v2.log" if gsv_lane_count == 1 else f"api_v2_lane_{lane_index + 1:02d}.log"
            server_managers.append(
                ManagedGPTSoVITSServer(
                    enabled=should_auto_start_server,
                    base_url=base_url,
                    command=gsv_server_command if gsv_server_command is not None else cfg.gsv_server_command,
                    cwd=cfg.gsv_server_cwd,
                    log_path=project_dir / "work" / "gpt_sovits" / log_name,
                    startup_timeout_sec=cfg.gsv_server_startup_timeout_sec,
                    shutdown_timeout_sec=cfg.gsv_server_shutdown_timeout_sec,
                )
            )
    duration_rewrite_enabled = (
        not mock
        and _canonical_language(cfg.target_language) == "ko"
        and getattr(cfg, "gsv_duration_rewrite_backend", "none") == "gemma"
        and int(getattr(cfg, "gsv_duration_rewrite_max_attempts", 0)) > 0
    )
    duration_rewrite_timing = str(
        getattr(cfg, "gsv_duration_rewrite_timing", "after_initial")
    )
    duration_rewrite_after_initial_enabled = (
        duration_rewrite_enabled and duration_rewrite_timing == "after_initial"
    )
    duration_rewrite_before_zero_shot_enabled = (
        duration_rewrite_enabled and duration_rewrite_timing == "before_zero_shot"
    )
    timing_expansion_enabled = bool(
        not mock
        and _canonical_language(cfg.target_language) == "ko"
        and getattr(cfg, "gsv_timing_expansion_enabled", True)
        and int(getattr(cfg, "gsv_timing_expansion_max_attempts", 0)) > 0
    )
    duration_rewrite_base_url = cfg.gemma_text_server_url.rstrip("/")
    duration_rewrite_manager = (
        ManagedGemmaTextServer(
            enabled=cfg.gemma_text_server_auto_start,
            base_url=duration_rewrite_base_url,
            command=(
                _gemma_text_server_command(cfg, base_url=duration_rewrite_base_url, lane_index=0)
                if cfg.gemma_text_server_auto_start
                else []
            ),
            log_path=project_dir / "work" / "gpt_sovits" / "duration_rewrite_llama_server.log",
            startup_timeout_sec=cfg.gemma_text_server_startup_timeout_sec,
            shutdown_timeout_sec=cfg.gemma_text_server_shutdown_timeout_sec,
        )
        if duration_rewrite_enabled or timing_expansion_enabled
        else None
    )
    duration_rewrite_client: Any | None = None
    duration_rewrite_lock = Lock()
    model_switch: dict[str, Any] = {}
    gsv_servers_running = False
    duration_rewrite_running = False
    fine_tuned_retry_summary: dict[str, Any] | None = None
    static_ref_retry_summary: dict[str, Any] | None = None
    korean_clarity_retry_summary: dict[str, Any] | None = None
    low_temperature_retry_summary: dict[str, Any] | None = None
    duration_rewrite_before_zero_shot_summary: dict[str, Any] | None = None
    timing_expansion_summary: dict[str, Any] | None = None
    zero_shot_fallback_summary: dict[str, Any] | None = None
    keep_original_fallback_summary: dict[str, Any] | None = None
    numeric_phrase_rendered_segment_ids: set[str] = set()
    numeric_phrase_failed_segment_ids: set[str] = set()
    numeric_phrase_normal_tts_fallback_segment_ids: set[str] = set()
    numeric_phrase_handled_segment_ids: set[str] = set()
    try:
        clients: list[GPTSoVITSClient] = []
        if not mock:
            clients = [
                GPTSoVITSClient(base_url, cfg.gsv_timeout_sec, cfg.gsv_retries)
                for base_url in gsv_base_urls
            ]
        _validate_gsv_speaker_models(project_dir, manifest)
        gpt_weights = None
        sovits_weights = None
        if clients:
            if use_speaker_gsv:
                model_switch["gpt_weights_mode"] = "speaker_voice_bank"
                model_switch["sovits_weights_mode"] = "speaker_voice_bank"
                model_switch["speaker_models"] = sorted(cfg.gsv_speaker_models)
            else:
                gpt_weights = _resolve_gpt_weights_for_tts(
                    project_dir,
                    manifest,
                    cfg,
                    gpt_weights_path,
                    model_switch,
                )
                sovits_weights = (
                    sovits_weights_path
                    or cfg.gsv_sovits_weights_path
                    or (
                        manifest.artifacts.get(FEW_SHOT_ARTIFACT_SOVITS)
                        if cfg.gsv_sovits_weights_policy != "unchanged"
                        else None
                    )
                )
            if gpt_weights:
                model_switch["gpt_weights_path"] = gpt_weights
            if sovits_weights:
                model_switch["sovits_weights_path"] = sovits_weights
                model_switch["sovits_weights_mode"] = (
                    "explicit"
                    if sovits_weights_path or cfg.gsv_sovits_weights_path
                    else "few_shot_source_voice"
                )
        started_at = monotonic()
        last_logged_at = started_at
        lane_locks = [Lock() for _ in range(gsv_lane_count)]
        lane_gpt_weights: list[str | None] = [None for _ in range(gsv_lane_count)]
        lane_sovits_weights: list[str | None] = [None for _ in range(gsv_lane_count)]
        speaker_refs_cache: dict[str, dict[str, GPTSoVITSRef]] = {}
        speaker_refs_cache_lock = Lock()
        countdown_bank_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
        countdown_bank_lock = Lock()
        countdown_pack_cache: dict[str, dict[str, Any]] = {}
        countdown_pack_lock = Lock()
        countdown_pack_token_cache: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        countdown_pack_token_generated_rounds: dict[tuple[Any, ...], int] = {}
        countdown_pack_token_lock = Lock()
        countdown_chunk_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
        countdown_chunk_lock = Lock()
        countdown_carrier_cache: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        countdown_carrier_generated_rounds: dict[tuple[Any, ...], int] = {}
        countdown_carrier_lock = Lock()
        countdown_token_bank_cache: dict[str, list[dict[str, Any]]] = {}
        countdown_token_bank_warmup_keys: set[tuple[Any, ...]] = set()
        countdown_token_bank_lock = Lock()
        countdown_source_anchor_cache: dict[tuple[str, tuple[int, ...]], dict[str, Any] | None] = {}
        countdown_source_anchor_backend_cache: dict[str, Any] = {}
        countdown_source_anchor_unavailable_errors: dict[str, str] = {}
        countdown_source_anchor_lock = Lock()
        countdown_source_prosody_cache: dict[tuple[str, str, float, float], dict[str, Any] | None] = {}
        countdown_source_prosody_lock = Lock()
        pronunciation_qc_worker_count = min(
            max(1, int(getattr(cfg, "gsv_pronunciation_qc_workers", 1))),
            max(1, gsv_lane_count),
        )
        pronunciation_qc_backend_cache: dict[str, Any] = {}
        pronunciation_qc_unavailable_errors: dict[str, str] = {}
        pronunciation_qc_backend_locks = [
            Lock() for _ in range(pronunciation_qc_worker_count)
        ]
        countdown_bulk_asr_lock = Lock()

        def record_model_switch_instances(instances: list[dict[str, Any]]) -> None:
            if not instances:
                return
            if "instances" in model_switch:
                model_switch.setdefault("restarts", []).append(instances)
                return
            model_switch["instances"] = instances
            if len(instances) == 1:
                instance = instances[0]
                if "gpt_response" in instance:
                    model_switch["gpt_response"] = instance["gpt_response"]
                if "sovits_response" in instance:
                    model_switch["sovits_response"] = instance["sovits_response"]

        def start_gsv_servers() -> None:
            nonlocal gsv_servers_running, lane_gpt_weights, lane_sovits_weights
            if mock or gsv_servers_running:
                return
            if len(server_managers) > 1:
                executor = ThreadPoolExecutor(max_workers=len(server_managers))
                futures = [executor.submit(server_manager.start) for server_manager in server_managers]
                try:
                    for future in as_completed(futures):
                        future.result()
                except BaseException:
                    executor.shutdown(wait=True, cancel_futures=True)
                    for server_manager in reversed(server_managers):
                        server_manager.stop()
                    raise
                else:
                    executor.shutdown(wait=True)
            else:
                for server_manager in server_managers:
                    server_manager.start()
            gsv_servers_running = True
            lane_gpt_weights = [None for _ in range(gsv_lane_count)]
            lane_sovits_weights = [None for _ in range(gsv_lane_count)]
            switch_instances: list[dict[str, Any]] = []
            for lane_index, client in enumerate(clients):
                lane_switch: dict[str, Any] = {
                    "lane_index": lane_index,
                    "gsv_url": gsv_base_urls[lane_index],
                }
                if gpt_weights:
                    lane_switch["gpt_response"] = client.set_gpt_weights(gpt_weights)
                    lane_gpt_weights[lane_index] = gpt_weights
                if sovits_weights:
                    lane_switch["sovits_response"] = client.set_sovits_weights(sovits_weights)
                    lane_sovits_weights[lane_index] = sovits_weights
                switch_instances.append(lane_switch)
            record_model_switch_instances(switch_instances)

        def stop_gsv_servers() -> None:
            nonlocal gsv_servers_running
            if not gsv_servers_running:
                return
            for server_manager in reversed(server_managers):
                server_manager.stop()
            gsv_servers_running = False

        duration_rewrite_phase = "initial" if duration_rewrite_after_initial_enabled else "normal"
        duration_rewrite_retry_segment_ids: set[str] = set()
        internal_retry_segment_ids: set[str] = set()
        static_ref_retry_segment_ids: set[str] = set()
        pass_candidate_count_override: int | None = None
        pass_temperature_override: float | None = None
        pass_top_k_override: int | None = None
        pass_top_p_override: float | None = None
        pass_parallel_infer_override: bool | None = None
        synth_pass = "fine_tuned_initial"

        def should_retry_failed_segment(segment: Segment) -> bool:
            return bool(segment.status == "failed" and segment.script)

        def should_force_segment(segment: Segment) -> bool:
            return bool(
                force
                and segment.script
                and (
                    segment.status in {"synthesized", "failed"}
                    or (
                        segment.status == "absorbed"
                        and "synth_keep_original_fallback" in segment.analysis
                    )
                    or (countdown_only and segment.status == "needs_manual_review")
                )
            )

        def should_reset_previous_tts(segment: Segment) -> bool:
            return should_retry_failed_segment(segment) or should_force_segment(segment)

        def reset_previous_tts_attempt(segment: Segment) -> None:
            segment.status = "scripted"
            segment.tts = None
            segment.analysis.pop("countdown_renderer", None)
            segment.analysis.pop("countdown_renderer_skip", None)
            segment.analysis.pop("numeric_phrase_renderer", None)
            segment.analysis.pop("pending_timing_expansion", None)
            keep_original_fallback = segment.analysis.pop("synth_keep_original_fallback", None)
            if keep_original_fallback is not None:
                segment.analysis.setdefault("synth_keep_original_fallback_history", []).append(
                    copy.deepcopy(keep_original_fallback)
                )
            segment.errors = [
                error
                for error in segment.errors
                if error
                not in {
                    "No acceptable TTS candidates for mix.",
                    "All TTS candidates failed.",
                    "Micro segment too short for Korean TTS.",
                }
                and not error.startswith("GPT-SoVITS synthesis failed")
                and not error.startswith("Korean TTS preflight blocked synthesis")
            ]

        def detected_countdown_segment_values(segment: Segment) -> list[int] | None:
            if use_speaker_gsv:
                return None
            if not segment.script or _canonical_language(segment.script.tts_language) != "ko":
                return None
            values = _source_countdown_values(segment)
            if values is None:
                event = segment.analysis.get(COUNTDOWN_EVENT_KEY)
                if (
                    isinstance(event, dict)
                    and str(event.get("kind") or "") == "descending_countdown"
                    and str(getattr(cfg, "gsv_countdown_renderer", "chunk_bank")) != "numeric_phrase"
                ):
                    values = _countdown_event_values(segment)
                if values is None:
                    return None
            tokens = _countdown_spoken_tokens(values)
            if tokens is None:
                return None
            return values

        def countdown_segment_values(segment: Segment) -> list[int] | None:
            values = detected_countdown_segment_values(segment)
            if values is None:
                return None
            if should_reset_previous_tts(segment):
                reset_previous_tts_attempt(segment)
            if segment.status == "synthesized" or segment.status in SKIP_STATUSES:
                return None
            return values

        def countdown_spans_for_jobs(
            segment_jobs: list[tuple[int, Segment, int]],
        ) -> list[list[tuple[int, Segment, list[int]]]]:
            spans: list[list[tuple[int, Segment, list[int]]]] = []
            current: list[tuple[int, Segment, list[int]]] = []
            current_values: list[int] = []
            last_segment: Segment | None = None

            def flush_current() -> None:
                nonlocal current, current_values, last_segment
                if _is_strict_descending_countdown(current_values):
                    spans.append(current)
                current = []
                current_values = []
                last_segment = None

            for index, segment, _lane_index in segment_jobs:
                values = countdown_segment_values(segment)
                if values is None:
                    flush_current()
                    continue
                if current:
                    gap = max(0.0, segment.start - (last_segment.end if last_segment else segment.start))
                    if current_values[-1] - values[0] != 1 or gap > 2.0:
                        flush_current()
                current.append((index, segment, values))
                current_values.extend(values)
                last_segment = segment
            flush_current()
            return spans

        def countdown_source_anchor_enabled() -> bool:
            timing_mode = (
                str(getattr(cfg, "gsv_countdown_timing_mode", "source_smoothed"))
                .strip()
                .lower()
                .replace("-", "_")
            )
            return (
                not mock
                and bool(getattr(cfg, "gsv_countdown_source_anchor_enabled", True))
                and timing_mode in {"source_smoothed", "source_exact"}
            )

        def countdown_source_anchor_backend_name() -> str:
            backend = str(getattr(cfg, "asr_backend", "faster_whisper")).strip().lower()
            return "qwen_asr" if backend == "qwen_asr" else "faster_whisper"

        def countdown_source_anchor_backend_config() -> dict[str, Any]:
            backend_config = _asr_backend_config(cfg)
            backend_config.update(
                {
                    "language": "ja",
                    "word_timestamps": True,
                    "condition_on_previous_text": False,
                    "vad_filter": True,
                    "vad_parameters": {
                        "min_silence_duration_ms": 120,
                        "speech_pad_ms": 80,
                    },
                    "batched_inference": False,
                    "initial_prompt": _COUNTDOWN_SOURCE_ANCHOR_PROMPT,
                    "hotwords": _COUNTDOWN_SOURCE_ANCHOR_HOTWORDS,
                }
            )
            return backend_config

        def countdown_source_anchor_option_variants() -> list[dict[str, Any]]:
            return [
                {
                    "variant": "baseline",
                    "overrides": {
                        "language": "ja",
                        "word_timestamps": True,
                        "condition_on_previous_text": False,
                        "vad_filter": True,
                        "vad_parameters": {
                            "min_silence_duration_ms": 120,
                            "speech_pad_ms": 80,
                        },
                        "batched_inference": False,
                        "initial_prompt": _COUNTDOWN_SOURCE_ANCHOR_PROMPT,
                        "hotwords": _COUNTDOWN_SOURCE_ANCHOR_HOTWORDS,
                    },
                },
                {
                    "variant": "wide_pad_vad50",
                    "overrides": {
                        "language": "ja",
                        "word_timestamps": True,
                        "condition_on_previous_text": False,
                        "vad_filter": True,
                        "vad_parameters": {
                            "min_silence_duration_ms": 50,
                            "speech_pad_ms": 240,
                        },
                        "batched_inference": False,
                        "initial_prompt": _COUNTDOWN_SOURCE_ANCHOR_PROMPT,
                        "hotwords": _COUNTDOWN_SOURCE_ANCHOR_HOTWORDS,
                    },
                },
                {
                    "variant": "no_vad_prompt",
                    "overrides": {
                        "language": "ja",
                        "word_timestamps": True,
                        "condition_on_previous_text": False,
                        "vad_filter": False,
                        "vad_parameters": None,
                        "batched_inference": False,
                        "initial_prompt": _COUNTDOWN_SOURCE_ANCHOR_PROMPT,
                        "hotwords": _COUNTDOWN_SOURCE_ANCHOR_HOTWORDS,
                    },
                },
                {
                    "variant": "no_vad_no_prompt",
                    "overrides": {
                        "language": "ja",
                        "word_timestamps": True,
                        "condition_on_previous_text": False,
                        "vad_filter": False,
                        "vad_parameters": None,
                        "batched_inference": False,
                        "initial_prompt": None,
                        "hotwords": None,
                    },
                },
                {
                    "variant": "wide_pad_no_hotwords",
                    "overrides": {
                        "language": "ja",
                        "word_timestamps": True,
                        "condition_on_previous_text": False,
                        "vad_filter": True,
                        "vad_parameters": {
                            "min_silence_duration_ms": 50,
                            "speech_pad_ms": 240,
                        },
                        "batched_inference": False,
                        "initial_prompt": _COUNTDOWN_SOURCE_ANCHOR_PROMPT,
                        "hotwords": None,
                    },
                },
            ]

        def countdown_source_anchor_resolve_audio(segment: Segment) -> Path | None:
            raw_path = segment.audio_for_mix or segment.audio_for_gemma
            if raw_path:
                try:
                    return _resolve_project_read_path(project_dir, raw_path, "audio_for_mix")
                except Exception:
                    pass
            if segment.audio_for_gemma:
                try:
                    return _resolve_project_read_path(
                        project_dir,
                        segment.audio_for_gemma,
                        "audio_for_gemma",
                    )
                except Exception:
                    return None
            return None

        def countdown_source_anchor_rows_for_chunks(
            segment: Segment,
            values: list[int],
            chunks: list[ASRChunk],
        ) -> tuple[str, list[dict[str, Any]]]:
            source_kind, rows = _countdown_source_anchor_rows_from_chunks(segment, values, chunks)
            if rows and len(rows) == len(values):
                return source_kind, rows
            return "unavailable", []

        def countdown_source_anchor_is_asr_kind(source_kind: str) -> bool:
            return source_kind.startswith("asr_")

        def countdown_source_anchor_existing_result(
            segment: Segment,
            values: list[int],
        ) -> dict[str, Any] | None:
            event = segment.analysis.get(COUNTDOWN_EVENT_KEY)
            if not isinstance(event, dict):
                return None
            raw_timeline = event.get("source_anchor_timeline")
            if not isinstance(raw_timeline, list) or len(raw_timeline) != len(values):
                return None
            try:
                raw_values = [int(item.get("value")) for item in raw_timeline if isinstance(item, dict)]
            except (TypeError, ValueError):
                return None
            if raw_values != values:
                return None
            policy = event.get("source_anchor_policy")
            if not isinstance(policy, dict):
                policy = {"source_kind": "manifest_existing"}
            return {"policy": policy, "timeline": raw_timeline}

        def countdown_source_anchor_store_result(
            segment: Segment,
            values: list[int],
            *,
            backend_name: str,
            variant_name: str,
            source_kind: str,
            rows: list[dict[str, Any]],
        ) -> dict[str, Any]:
            event = segment.analysis.setdefault(COUNTDOWN_EVENT_KEY, {})
            if not isinstance(event, dict):
                event = {}
                segment.analysis[COUNTDOWN_EVENT_KEY] = event
            timeline = [
                {
                    "value": int(row["value"]),
                    "source_text": str(row.get("source_text") or row["value"]),
                    "korean_token": str(row.get("korean_token") or ""),
                    "start": round(float(row["start"]), 6),
                    "end": round(float(row.get("end", row["start"])), 6),
                    "confidence": row.get("confidence"),
                    "method": str(row.get("method") or source_kind),
                }
                for row in rows
            ]
            policy = {
                "source_kind": source_kind,
                "backend": backend_name,
                "variant": variant_name,
                "values": [int(value) for value in values],
                "smoothing_blend": float(
                    getattr(cfg, "gsv_countdown_source_anchor_smoothing_blend", 0.70)
                ),
                "cluster_gap_sec": float(
                    getattr(cfg, "gsv_countdown_source_anchor_cluster_gap_sec", 2.6)
                ),
            }
            event["source_anchor_timeline"] = timeline
            event["source_anchor_timeline_local"] = True
            event["source_anchor_policy"] = policy
            result = {"policy": policy, "timeline": timeline}
            with countdown_source_anchor_lock:
                countdown_source_anchor_cache[(segment.id, tuple(values))] = result
            return result

        def countdown_source_anchor_for_segment(
            segment: Segment,
            values: list[int],
        ) -> dict[str, Any] | None:
            cache_key = (segment.id, tuple(values))
            with countdown_source_anchor_lock:
                if cache_key in countdown_source_anchor_cache:
                    return countdown_source_anchor_cache[cache_key]
            existing = countdown_source_anchor_existing_result(segment, values)
            if existing is not None:
                with countdown_source_anchor_lock:
                    countdown_source_anchor_cache[cache_key] = existing
                return existing
            if not countdown_source_anchor_enabled():
                with countdown_source_anchor_lock:
                    countdown_source_anchor_cache[cache_key] = None
                return None
            event = segment.analysis.get(COUNTDOWN_EVENT_KEY)
            raw_token_timeline = event.get("token_timeline") if isinstance(event, dict) else None
            if isinstance(raw_token_timeline, list) and len(raw_token_timeline) == len(values):
                with countdown_source_anchor_lock:
                    countdown_source_anchor_cache[cache_key] = None
                return None
            if (
                segment.source_script is None
                or str(segment.source_script.backend).strip().lower() == "mock"
            ):
                with countdown_source_anchor_lock:
                    countdown_source_anchor_cache[cache_key] = None
                return None

            audio_path = countdown_source_anchor_resolve_audio(segment)
            if audio_path is None:
                source_kind, rows = _countdown_source_anchor_rows_from_chunks(segment, values, [])
                if not rows:
                    with countdown_source_anchor_lock:
                        countdown_source_anchor_cache[cache_key] = None
                    return None
                return countdown_source_anchor_store_result(
                    segment,
                    values,
                    backend_name="none",
                    variant_name="source_text_only",
                    source_kind=source_kind,
                    rows=rows,
                )

            backend_name = countdown_source_anchor_backend_name()
            if backend_name in countdown_source_anchor_unavailable_errors:
                source_kind, rows = _countdown_source_anchor_rows_from_chunks(segment, values, [])
                if rows:
                    return countdown_source_anchor_store_result(
                        segment,
                        values,
                        backend_name=backend_name,
                        variant_name="source_text_only_after_asr_unavailable",
                        source_kind=source_kind,
                        rows=rows,
                    )
                with countdown_source_anchor_lock:
                    countdown_source_anchor_cache[cache_key] = None
                return None

            try:
                with countdown_source_anchor_lock:
                    backend = countdown_source_anchor_backend_cache.get(backend_name)
                    if backend is None:
                        backend = create_asr_backend(
                            backend_name,
                            countdown_source_anchor_backend_config(),
                        )
                        countdown_source_anchor_backend_cache[backend_name] = backend
            except ASRUnavailableError as exc:
                countdown_source_anchor_unavailable_errors[backend_name] = str(exc)
                backend = None

            best_source_kind = "unavailable"
            best_variant_name = "none"
            best_rows: list[dict[str, Any]] = []
            if backend is not None:
                variants = countdown_source_anchor_option_variants()
                if not bool(getattr(cfg, "gsv_countdown_source_anchor_asr_retry_enabled", True)):
                    variants = variants[:1]
                for variant in variants:
                    variant_name = str(variant["variant"])
                    try:
                        chunks = _transcribe_with_backend_options(
                            backend,
                            audio_path,
                            [
                                Segment(
                                    id=f"{segment.id}_countdown_source_anchor",
                                    speaker_id=segment.speaker_id,
                                    start=0.0,
                                    end=segment.duration,
                                    duration=segment.duration,
                                    audio_for_gemma=str(audio_path),
                                    audio_for_mix=str(audio_path),
                                )
                            ],
                            **dict(variant["overrides"]),
                        )
                    except ASRUnavailableError as exc:
                        countdown_source_anchor_unavailable_errors[backend_name] = str(exc)
                        break
                    source_kind, rows = countdown_source_anchor_rows_for_chunks(
                        segment,
                        values,
                        chunks,
                    )
                    if rows and (
                        not best_rows
                        or (
                            countdown_source_anchor_is_asr_kind(source_kind)
                            and not countdown_source_anchor_is_asr_kind(best_source_kind)
                        )
                    ):
                        best_source_kind = source_kind
                        best_variant_name = variant_name
                        best_rows = rows
                    if rows and countdown_source_anchor_is_asr_kind(source_kind):
                        break

            if not best_rows:
                best_source_kind, best_rows = _countdown_source_anchor_rows_from_chunks(
                    segment,
                    values,
                    [],
                )
                best_variant_name = "source_text_only"
            if best_rows:
                return countdown_source_anchor_store_result(
                    segment,
                    values,
                    backend_name=backend_name,
                    variant_name=best_variant_name,
                    source_kind=best_source_kind,
                    rows=best_rows,
                )
            with countdown_source_anchor_lock:
                countdown_source_anchor_cache[cache_key] = None
            return None

        def precompute_countdown_source_anchors(
            spans: list[list[tuple[int, Segment, list[int]]]],
        ) -> None:
            if not countdown_source_anchor_enabled():
                return
            computed = 0
            try:
                for span in spans:
                    for _index, segment, values in span:
                        if countdown_source_anchor_for_segment(segment, values) is not None:
                            computed += 1
            finally:
                with countdown_source_anchor_lock:
                    countdown_source_anchor_backend_cache.clear()
                clear_gpu_vram("countdown_source_anchor")
            if computed:
                _log_stage_checkpoint(
                    stage_name,
                    "countdown source anchors",
                    f"segments={computed}",
                )

        def token_audio_for_countdown(
            *,
            token_text: str,
            token_index: int,
            token_count: int,
            span_id: str,
            ref: GPTSoVITSRef,
            token_slot_sec: float,
            lane_index: int,
            tts_text_language: str,
            ref_style: str,
            candidate_index: int = 0,
        ) -> tuple[np.ndarray, dict[str, Any]]:
            token_dir = project_dir / "work" / "tts" / "countdown" / "tokens"
            token_label = _countdown_chunk_label(token_text)
            candidate_suffix = f"_cand_{candidate_index:02d}" if candidate_index else ""
            raw_path = token_dir / f"{span_id}_chunk_{token_index:02d}{candidate_suffix}_{token_label}.wav"
            speed = float(getattr(cfg, "gsv_countdown_token_speed_factor", cfg.gsv_tts_max_speed_factor))
            seed = cfg.base_seed + 70_000 + token_index + candidate_index * 1_009
            options = GPTSoVITSTTSOptions(
                seed=seed,
                speed_factor=speed,
                text_lang=tts_text_language,
                top_k=cfg.gsv_top_k,
                top_p=cfg.gsv_top_p,
                temperature=float(getattr(cfg, "gsv_countdown_temperature", cfg.gsv_temperature)),
                text_split_method=cfg.gsv_text_split_method,
                fragment_interval=cfg.gsv_fragment_interval,
                parallel_infer=cfg.gsv_parallel_infer,
                repetition_penalty=cfg.gsv_repetition_penalty,
                sample_steps=cfg.gsv_sample_steps,
                super_sampling=cfg.gsv_super_sampling,
                overlap_length=cfg.gsv_overlap_length,
                min_chunk_length=cfg.gsv_min_chunk_length,
            )
            payload: dict[str, Any] = {
                "renderer": "countdown_phrase_timeline",
                "token_text": token_text,
                "token_index": token_index,
                "candidate_index": candidate_index,
                "chunk_text": token_text,
                "chunk_index": token_index,
                "chunk_token_count": token_count,
                "ref_style": ref_style,
                "target_token_slot_sec": round(token_slot_sec, 6),
                "lane_index": lane_index,
                "gsv_url": None if mock else gsv_base_urls[lane_index],
            }
            payload.update(_tts_request_debug_payload(token_text, ref, options))
            if mock:
                _mock_synthesize(raw_path, max(0.12, min(0.42, token_slot_sec * 0.75)), options.seed, cfg.mix_sample_rate)
                payload["mock"] = True
            else:
                client = clients[lane_index]
                request = client.build_payload(token_text, ref, options)
                payload.update(request.as_payload())
                client.synthesize_to_file(request, raw_path)
            postprocess_tts_candidate(raw_path, payload)
            raw_duration = duration_sec(raw_path)
            data, sample_rate = load_audio(raw_path)
            data = _countdown_stereo(data)
            if sample_rate != cfg.mix_sample_rate:
                data = resample_linear(data, sample_rate, cfg.mix_sample_rate)
                sample_rate = cfg.mix_sample_rate
            final_duration = len(data) / sample_rate
            payload["countdown_token_fit"] = {
                "raw_path": str(raw_path),
                "raw_duration_sec": round(raw_duration, 6),
                "fitted_path": str(raw_path),
                "fitted_duration_sec": round(final_duration, 6),
                "token_target_sec": round(final_duration, 6),
                "fit_strategy": "raw_no_linear_resample",
            }
            return data, payload

        def countdown_token_text_max_sec(token_text: str) -> float:
            unit_count = len(normalize_korean_pronunciation_text(token_text))
            if unit_count <= 1:
                return float(getattr(cfg, "gsv_countdown_token_single_syllable_max_sec", 0.55))
            if unit_count == 2:
                return float(getattr(cfg, "gsv_countdown_token_double_syllable_max_sec", 0.75))
            return float(getattr(cfg, "gsv_countdown_token_multi_syllable_max_sec", 0.95))

        def countdown_token_max_allowed_sec(
            token_slot_sec: float,
            *,
            token_text: str | None = None,
        ) -> float:
            absolute_max = float(getattr(cfg, "gsv_countdown_token_max_sec", 0.95))
            if token_text is not None:
                absolute_max = min(absolute_max, countdown_token_text_max_sec(token_text))
            slot_occupancy = float(getattr(cfg, "gsv_countdown_token_max_slot_occupancy", 0.85))
            if token_slot_sec <= 0:
                return absolute_max
            return min(absolute_max, token_slot_sec * slot_occupancy)

        def countdown_strict_pronunciation_required(segment: Segment) -> bool:
            if (
                mock
                or _canonical_language(cfg.target_language) != "ko"
                or not bool(getattr(cfg, "gsv_pronunciation_qc_enabled", True))
                or not bool(getattr(cfg, "gsv_countdown_strict_token_pronunciation", True))
            ):
                return False
            configured_backend = (
                str(getattr(cfg, "gsv_pronunciation_qc_backend", "auto"))
                .strip()
                .lower()
                .replace("-", "_")
            )
            return not (
                configured_backend == "auto"
                and (
                    segment.source_script is None
                    or str(segment.source_script.backend).strip().lower() == "mock"
                )
            )

        def countdown_pronunciation_contract_ok(
            segment: Segment,
            pronunciation_qc: dict[str, Any] | None,
        ) -> bool:
            gate = str((pronunciation_qc or {}).get("gate") or "").strip().lower()
            if countdown_strict_pronunciation_required(segment):
                return gate == "pass"
            return not (
                gate == "fail"
                and bool(getattr(cfg, "gsv_pronunciation_qc_failure_blocks_mix", True))
            )

        def countdown_source_prosody_for_placement(
            segment: Segment,
            placement: dict[str, Any],
        ) -> dict[str, Any] | None:
            if mock or not bool(getattr(cfg, "gsv_countdown_prosody_qc_enabled", True)):
                return None
            raw_start = placement.get("source_start_sec")
            raw_end = placement.get("source_end_sec")
            if raw_start is None or raw_end is None:
                return None
            try:
                local_start = max(0.0, float(raw_start) - float(segment.start))
                local_end = min(float(segment.duration), float(raw_end) - float(segment.start))
            except (TypeError, ValueError):
                return None
            if local_end - local_start <= 0.03:
                return None
            try:
                audio_path = _resolve_project_read_path(
                    project_dir,
                    segment.audio_for_mix,
                    "audio_for_mix",
                )
            except Exception:
                return None
            cache_key = (
                segment.id,
                str(audio_path),
                round(local_start, 4),
                round(local_end, 4),
            )
            with countdown_source_prosody_lock:
                if cache_key in countdown_source_prosody_cache:
                    return countdown_source_prosody_cache[cache_key]
            try:
                source_audio, sample_rate = load_audio(audio_path)
                start_frame = max(0, int(round(local_start * sample_rate)))
                end_frame = min(len(source_audio), int(round(local_end * sample_rate)))
                if end_frame <= start_frame:
                    profile = None
                else:
                    profile = _countdown_pitch_profile(source_audio[start_frame:end_frame], sample_rate)
            except Exception:
                profile = None
            with countdown_source_prosody_lock:
                countdown_source_prosody_cache[cache_key] = profile
            return profile

        def attach_countdown_prosody_qc(
            payload: dict[str, Any],
            audio: np.ndarray,
            sample_rate: int,
            source_profile: dict[str, Any] | None,
        ) -> dict[str, Any] | None:
            if source_profile is None or not bool(getattr(cfg, "gsv_countdown_prosody_qc_enabled", True)):
                return None
            candidate_profile = _countdown_pitch_profile(audio, sample_rate)
            prosody_qc = _countdown_prosody_match(
                source_profile,
                candidate_profile,
                max_median_semitone_error=float(
                    getattr(cfg, "gsv_countdown_prosody_max_median_semitone_error", 12.0)
                ),
                pass_score=float(getattr(cfg, "gsv_countdown_prosody_min_pass_score", 0.74)),
                warn_score=float(getattr(cfg, "gsv_countdown_prosody_min_warn_score", 0.45)),
            )
            payload["countdown_prosody_qc"] = prosody_qc
            payload["countdown_texture_qc"] = _countdown_texture_match(
                source_profile.get("texture_profile") if isinstance(source_profile, dict) else None,
                candidate_profile.get("texture_profile") if isinstance(candidate_profile, dict) else None,
            )
            return prosody_qc

        def countdown_candidate_prosody_score(candidate: dict[str, Any]) -> float:
            qc = candidate.get("payload", {}).get("countdown_prosody_qc")
            if not isinstance(qc, dict):
                return 1.0
            gate = str(qc.get("gate") or "").strip().lower()
            if gate in {"", "disabled", "skipped", "unavailable"}:
                return 1.0
            try:
                score = float(qc.get("score"))
            except (TypeError, ValueError):
                return 0.0 if gate == "fail" else 1.0
            return max(0.0, min(score, 1.0))

        def countdown_prosody_contract_ok(candidate: dict[str, Any]) -> bool:
            if not bool(getattr(cfg, "gsv_countdown_prosody_failure_blocks_mix", True)):
                return True
            qc = candidate.get("payload", {}).get("countdown_prosody_qc")
            if not isinstance(qc, dict):
                return True
            gate = str(qc.get("gate") or "").strip().lower()
            return gate != "fail"

        def countdown_candidate_texture_score(candidate: dict[str, Any]) -> float:
            qc = candidate.get("payload", {}).get("countdown_texture_qc")
            if not isinstance(qc, dict):
                return 1.0
            gate = str(qc.get("gate") or "").strip().lower()
            if gate in {"", "disabled", "skipped", "unavailable"}:
                return 1.0
            try:
                score = float(qc.get("score"))
            except (TypeError, ValueError):
                return 0.0 if gate == "fail" else 1.0
            return max(0.0, min(score, 1.0))

        def countdown_token_candidate_score(candidate: dict[str, Any], token_slot_sec: float) -> float:
            duration = float(candidate["duration_sec"])
            min_sec = float(getattr(cfg, "gsv_countdown_token_min_sec", 0.25))
            max_sec = countdown_token_max_allowed_sec(
                token_slot_sec,
                token_text=str(candidate.get("text") or ""),
            )
            target_sec = min(max_sec, max(min_sec, token_slot_sec * 0.58))
            if target_sec <= 0:
                return 0.0
            duration_score = max(0.0, 1.0 - min(abs(duration - target_sec) / target_sec, 1.0))
            peak = float(candidate["payload"].get("audio_qc", {}).get("peak_dbfs", -120.0))
            loudness_score = max(0.0, min((peak + 60.0) / 60.0, 1.0))
            pronunciation_score = countdown_token_pronunciation_score(candidate)
            prosody_score = countdown_candidate_prosody_score(candidate)
            return round(
                duration_score * 0.50
                + pronunciation_score * 0.25
                + prosody_score * 0.20
                + loudness_score * 0.05,
                6,
            )

        def countdown_token_duration_gate(
            *,
            duration: float,
            token_slot_sec: float,
            token_text: str,
            candidate_source: str | None = None,
        ) -> tuple[str, float, float]:
            token_min_sec = float(getattr(cfg, "gsv_countdown_token_min_sec", 0.25))
            if str(candidate_source or "").strip() == "numeric_unit_carrier":
                numeric_windows = [
                    max(0.04, float(value))
                    for value in getattr(
                        cfg,
                        "gsv_countdown_carrier_numeric_unit_onset_window_sec",
                        [0.18, 0.24, 0.30, 0.36],
                    )
                ]
                if numeric_windows:
                    token_min_sec = min(token_min_sec, min(numeric_windows) * 0.9)
            token_max_sec = countdown_token_max_allowed_sec(
                token_slot_sec,
                token_text=token_text,
            )
            if duration < token_min_sec:
                return "too_short", token_min_sec, token_max_sec
            if duration > token_max_sec:
                return "too_long", token_min_sec, token_max_sec
            return "pass", token_min_sec, token_max_sec

        def countdown_token_pronunciation_score(candidate: dict[str, Any]) -> float:
            strict_required = bool(
                candidate["payload"]
                .get("countdown_token_candidate", {})
                .get("strict_pronunciation_required", False)
            )
            qc = candidate["payload"].get("pronunciation_qc")
            if not isinstance(qc, dict):
                return 0.0 if strict_required else 1.0
            gate = str(qc.get("gate") or "").strip().lower()
            if gate in {"", "disabled", "skipped", "unavailable"}:
                return 0.0 if strict_required else 1.0
            try:
                coverage = float(qc.get("coverage"))
            except (TypeError, ValueError):
                coverage = 0.0 if gate == "fail" else 1.0
            coverage = max(0.0, min(coverage, 1.0))
            if gate == "pass":
                return coverage
            if gate == "warn":
                return min(coverage, 0.74)
            if gate == "fail":
                return min(coverage, 0.25)
            return coverage

        def countdown_token_candidate_audio(
            *,
            token_text: str,
            token_index: int,
            candidate_index: int,
            span_id: str,
            segment: Segment,
            ref: GPTSoVITSRef,
            token_slot_sec: float,
            lane_index: int,
            tts_text_language: str,
            ref_style: str,
            source_prosody: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            token_label = _countdown_chunk_label(token_text)
            speed = float(
                getattr(cfg, "gsv_countdown_token_speed_factor", cfg.gsv_tts_max_speed_factor)
            )
            stable_token_seed = sum((index + 1) * ord(char) for index, char in enumerate(token_text))
            seed = cfg.base_seed + 70_000 + stable_token_seed + candidate_index
            options = GPTSoVITSTTSOptions(
                seed=seed,
                speed_factor=speed,
                text_lang=tts_text_language,
                top_k=cfg.gsv_top_k,
                top_p=cfg.gsv_top_p,
                temperature=float(getattr(cfg, "gsv_countdown_temperature", cfg.gsv_temperature)),
                text_split_method=cfg.gsv_text_split_method,
                fragment_interval=0.0,
                parallel_infer=cfg.gsv_parallel_infer,
                repetition_penalty=cfg.gsv_repetition_penalty,
                sample_steps=cfg.gsv_sample_steps,
                super_sampling=cfg.gsv_super_sampling,
                overlap_length=cfg.gsv_overlap_length,
                min_chunk_length=cfg.gsv_min_chunk_length,
            )
            bank_key = (
                ref_style,
                token_text,
                candidate_index,
                options.speed_factor,
                options.text_lang,
                options.top_k,
                options.top_p,
                options.temperature,
                options.text_split_method,
                options.fragment_interval,
                options.parallel_infer,
                options.repetition_penalty,
                options.sample_steps,
                options.super_sampling,
                options.overlap_length,
                options.min_chunk_length,
            )
            bank_dir = (
                project_dir
                / "work"
                / "tts"
                / "countdown"
                / "bank"
                / _countdown_chunk_label(ref_style)
            )
            output_path = bank_dir / f"{token_label}_cand_{candidate_index:02d}.wav"
            payload: dict[str, Any] = {
                "renderer": "countdown_token_timeline",
                "token_text": token_text,
                "token_index": token_index,
                "candidate_index": candidate_index,
                "ref_style": ref_style,
                "target_token_slot_sec": round(token_slot_sec, 6),
                "lane_index": lane_index,
                "gsv_url": None if mock else gsv_base_urls[lane_index],
            }
            payload.update(_tts_request_debug_payload(token_text, ref, options))
            with countdown_bank_lock:
                cached = countdown_bank_cache.get(bank_key)
            if cached is None:
                if mock:
                    duration = min(
                        countdown_token_max_allowed_sec(token_slot_sec, token_text=token_text) * 0.75,
                        max(
                            float(getattr(cfg, "gsv_countdown_token_min_sec", 0.25)),
                            token_slot_sec * 0.55,
                        ),
                    )
                    _mock_synthesize(output_path, max(0.12, duration), options.seed, cfg.mix_sample_rate)
                    payload["mock"] = True
                else:
                    client = clients[lane_index]
                    request = client.build_payload(token_text, ref, options)
                    payload.update(request.as_payload())
                    client.synthesize_to_file(request, output_path)

                raw_duration = duration_sec(output_path)
                postprocess_tts_candidate(output_path, payload)
                duration = duration_sec(output_path)
                data, sample_rate = load_audio(output_path)
                data = _countdown_stereo(data)
                if sample_rate != cfg.mix_sample_rate:
                    data = resample_linear(data, sample_rate, cfg.mix_sample_rate)
                    sample_rate = cfg.mix_sample_rate
                new_cached = {
                    "payload": copy.deepcopy(payload),
                    "raw_duration": raw_duration,
                    "duration": duration,
                    "audio": data,
                    "sample_rate": sample_rate,
                    "output_path": str(output_path),
                }
                with countdown_bank_lock:
                    cached = countdown_bank_cache.setdefault(bank_key, new_cached)
                reused_existing = cached is not new_cached
            else:
                reused_existing = True
            raw_duration = float(cached["raw_duration"])
            duration = float(cached["duration"])
            output_path = Path(str(cached["output_path"]))
            sample_rate = int(cached["sample_rate"])
            data = np.array(cached["audio"], copy=True)
            payload = copy.deepcopy(cached["payload"])
            payload.update(
                {
                    "token_index": token_index,
                    "candidate_index": candidate_index,
                    "target_token_slot_sec": round(token_slot_sec, 6),
                    "lane_index": lane_index,
                    "gsv_url": None if mock else gsv_base_urls[lane_index],
                }
            )
            payload["countdown_bank"] = {
                "key": [str(item) for item in bank_key],
                "path": str(cached["output_path"]),
                "reused_existing": reused_existing,
            }
            duration_gate, token_min_sec, token_max_sec = countdown_token_duration_gate(
                duration=duration,
                token_slot_sec=token_slot_sec,
                token_text=token_text,
            )
            audio_metrics = {
                "gate": "pass",
                "peak_dbfs": round(peak_dbfs(output_path), 3),
                "rms_dbfs": round(rms_dbfs(output_path), 3),
            }
            payload["audio_qc"] = audio_metrics
            attach_countdown_prosody_qc(payload, data, sample_rate, source_prosody)
            pronunciation_qc = run_pronunciation_qc(
                segment=segment,
                candidate_path=output_path,
                expected_text=token_text,
                candidate_duration_sec=duration,
                short_slice=True,
            )
            if pronunciation_qc is not None:
                payload["pronunciation_qc"] = pronunciation_qc
            pronunciation_gate = str((pronunciation_qc or {}).get("gate") or "").strip().lower()
            strict_pronunciation_required = countdown_strict_pronunciation_required(segment)
            pronunciation_contract_ok = countdown_pronunciation_contract_ok(segment, pronunciation_qc)
            payload["countdown_token_candidate"] = {
                "raw_duration_sec": round(raw_duration, 6),
                "duration_sec": round(duration, 6),
                "duration_gate": duration_gate,
                "token_min_sec": round(token_min_sec, 6),
                "token_max_sec": round(token_max_sec, 6),
                "pronunciation_gate": pronunciation_gate or "not_run",
                "strict_pronunciation_required": strict_pronunciation_required,
                "pronunciation_contract_ok": pronunciation_contract_ok,
            }
            candidate = {
                "candidate_index": candidate_index,
                "seed": seed,
                "text": token_text,
                "output_path": str(cached["output_path"]),
                "duration_sec": duration,
                "duration_gate": duration_gate,
                "acceptable": duration_gate == "pass"
                and pronunciation_contract_ok
                and countdown_prosody_contract_ok({"payload": payload}),
                "payload": payload,
                "audio": data,
            }
            candidate["selection_score"] = countdown_token_candidate_score(
                candidate,
                token_slot_sec,
            )
            return candidate

        def countdown_token_slots_for_span(
            span: list[tuple[int, Segment, list[int]]],
            sample_rate: int,
        ) -> tuple[list[dict[str, Any]], dict[str, int], int]:
            segment_frames: dict[str, int] = {}
            segment_offsets: dict[str, int] = {}
            offset = 0
            for _index, segment, _segment_values in span:
                frames = max(1, int(round(segment.duration * sample_rate)))
                segment_frames[segment.id] = frames
                segment_offsets[segment.id] = offset
                offset += frames
            total_frames = offset

            def placement_payload_from_row(row: dict[str, Any]) -> dict[str, Any]:
                return {
                    key: value
                    for key, value in row.items()
                    if key
                    not in {
                        "source_start_frame",
                        "source_end_frame",
                        "segment_end_frame",
                        "equal_slot_frames",
                    }
                }

            def equal_grid_placements(anchor: str = "even_grid") -> list[dict[str, Any]]:
                placements: list[dict[str, Any]] = []
                next_token_index = 0
                for _index, segment, segment_values in span:
                    frames = segment_frames[segment.id]
                    segment_offset = segment_offsets[segment.id]
                    value_count = max(1, len(segment_values))
                    segment_tokens = _countdown_spoken_tokens(segment_values) or []
                    for local_index, value in enumerate(segment_values):
                        slot_start = segment_offset + int(round(local_index * frames / value_count))
                        slot_end = segment_offset + int(round((local_index + 1) * frames / value_count))
                        token_text = (
                            segment_tokens[local_index]
                            if local_index < len(segment_tokens)
                            else str(value)
                        )
                        slot_frames = max(slot_end - slot_start, 1)
                        placements.append(
                            {
                                "segment_id": segment.id,
                                "value": value,
                                "text": token_text,
                                "token_index": next_token_index,
                                "local_index": local_index,
                                "slot_start_frame": slot_start,
                                "slot_end_frame": max(slot_start + 1, slot_end),
                                "slot_duration_sec": slot_frames / sample_rate,
                                "placement_anchor": anchor,
                            }
                        )
                        next_token_index += 1
                return placements

            timeline_rows: list[dict[str, Any]] = []
            token_index = 0
            for _index, segment, segment_values in span:
                event = segment.analysis.get(COUNTDOWN_EVENT_KEY)
                raw_timeline = None
                timeline_kind = ""
                if isinstance(event, dict):
                    token_timeline = event.get("token_timeline")
                    source_anchor_timeline = event.get("source_anchor_timeline")
                    if isinstance(token_timeline, list) and len(token_timeline) == len(segment_values):
                        raw_timeline = token_timeline
                        timeline_kind = "token_timeline"
                    elif (
                        countdown_source_anchor_enabled()
                        and isinstance(source_anchor_timeline, list)
                        and len(source_anchor_timeline) == len(segment_values)
                    ):
                        raw_timeline = source_anchor_timeline
                        timeline_kind = "source_anchor_timeline"
                if not isinstance(raw_timeline, list) or len(raw_timeline) != len(segment_values):
                    timeline_rows = []
                    break
                value_count = max(1, len(segment_values))
                segment_tokens = _countdown_spoken_tokens(segment_values) or []
                raw_values: list[int] = []
                for local_index, (value, raw_item) in enumerate(
                    zip(segment_values, raw_timeline, strict=True)
                ):
                    if not isinstance(raw_item, dict):
                        timeline_rows = []
                        break
                    raw_value = raw_item.get("value")
                    raw_start = raw_item.get("start")
                    raw_end = raw_item.get("end", raw_start)
                    if raw_value is None or raw_start is None:
                        timeline_rows = []
                        break
                    try:
                        raw_value_int = int(raw_value)
                        raw_start_sec = float(raw_start)
                        raw_end_sec = float(raw_end)
                    except (TypeError, ValueError):
                        timeline_rows = []
                        break
                    raw_values.append(raw_value_int)
                    if timeline_kind == "source_anchor_timeline":
                        local_start_sec = max(0.0, min(segment.duration, raw_start_sec))
                        local_end_sec = max(local_start_sec, min(segment.duration, raw_end_sec))
                        source_start_sec = segment.start + local_start_sec
                        source_end_sec = segment.start + local_end_sec
                    else:
                        local_start_sec = max(0.0, min(segment.duration, raw_start_sec - segment.start))
                        local_end_sec = max(local_start_sec, min(segment.duration, raw_end_sec - segment.start))
                        source_start_sec = raw_start_sec
                        source_end_sec = raw_end_sec
                    token_text = (
                        segment_tokens[local_index]
                        if local_index < len(segment_tokens)
                        else str(raw_item.get("korean_token") or "") or str(value)
                    )
                    timeline_rows.append(
                        {
                            "segment_id": segment.id,
                            "value": value,
                            "text": token_text,
                            "token_index": token_index,
                            "local_index": local_index,
                            "timeline_kind": timeline_kind,
                            "source_text": str(raw_item.get("source_text") or ""),
                            "source_start_sec": round(source_start_sec, 6),
                            "source_end_sec": round(source_end_sec, 6),
                            "local_source_start_sec": round(local_start_sec, 6),
                            "local_source_end_sec": round(local_end_sec, 6),
                            "source_start_frame": segment_offsets[segment.id]
                            + int(round(local_start_sec * sample_rate)),
                            "source_end_frame": segment_offsets[segment.id]
                            + int(round(local_end_sec * sample_rate)),
                            "segment_end_frame": segment_offsets[segment.id]
                            + segment_frames[segment.id],
                            "equal_slot_frames": max(
                                1,
                                int(round(segment_frames[segment.id] / value_count)),
                            ),
                        }
                    )
                    token_index += 1
                if timeline_rows == [] or raw_values != segment_values:
                    timeline_rows = []
                    break

            timing_mode = (
                str(getattr(cfg, "gsv_countdown_timing_mode", "source_smoothed"))
                .strip()
                .lower()
                .replace("-", "_")
            )
            if timing_mode == "even_grid":
                return equal_grid_placements("even_grid"), segment_frames, total_frames

            if timeline_rows and len(timeline_rows) == sum(len(values) for _i, _s, values in span):
                if timing_mode == "source_smoothed":
                    if all(
                        row.get("timeline_kind") == "source_anchor_timeline"
                        for row in timeline_rows
                    ):
                        placements = []
                        rows_by_segment: dict[str, list[dict[str, Any]]] = {}
                        for row in timeline_rows:
                            rows_by_segment.setdefault(str(row["segment_id"]), []).append(row)
                        for _index, segment, _segment_values in span:
                            segment_rows = rows_by_segment.get(segment.id, [])
                            if not segment_rows:
                                continue
                            segment_offset = segment_offsets[segment.id]
                            segment_end_frame = segment_offset + segment_frames[segment.id]
                            starts_sec = [
                                max(0.0, float(row["source_start_frame"] - segment_offset) / sample_rate)
                                for row in segment_rows
                            ]
                            values = [int(row["value"]) for row in segment_rows]
                            source_text = ""
                            if segment.source_script is not None:
                                source_text = segment.source_script.text
                            elif segment.script is not None:
                                source_text = segment.script.literal_ja or segment.script.ja_text
                            clusters = _countdown_source_anchor_clusters(
                                source_text,
                                values,
                                starts_sec,
                                max_cluster_gap_sec=float(
                                    getattr(cfg, "gsv_countdown_source_anchor_cluster_gap_sec", 2.6)
                                ),
                            )
                            smoothed_starts = _countdown_smooth_source_anchor_starts(
                                starts_sec,
                                clusters,
                                blend=float(
                                    getattr(cfg, "gsv_countdown_source_anchor_smoothing_blend", 0.70)
                                ),
                            )
                            cluster_by_local_index: dict[int, int] = {}
                            for cluster_index, cluster in enumerate(clusters):
                                for local_index in cluster:
                                    cluster_by_local_index[local_index] = cluster_index
                            for local_index, row in enumerate(segment_rows):
                                slot_start = segment_offset + int(
                                    round(smoothed_starts[local_index] * sample_rate)
                                )
                                slot_start = max(segment_offset, min(slot_start, segment_end_frame - 1))
                                next_start = None
                                if (
                                    local_index + 1 < len(segment_rows)
                                    and cluster_by_local_index.get(local_index)
                                    == cluster_by_local_index.get(local_index + 1)
                                ):
                                    next_start = segment_offset + int(
                                        round(smoothed_starts[local_index + 1] * sample_rate)
                                    )
                                source_period_frames = (
                                    max(1, next_start - slot_start)
                                    if next_start is not None
                                    else int(row["equal_slot_frames"])
                                )
                                slot_frames = max(source_period_frames, int(row["equal_slot_frames"]))
                                placement = placement_payload_from_row(row)
                                placement.update(
                                    {
                                        "slot_start_frame": slot_start,
                                        "slot_end_frame": slot_start + slot_frames,
                                        "slot_duration_sec": slot_frames / sample_rate,
                                        "placement_anchor": "source_anchor_smoothed",
                                        "source_anchor_cluster_index": cluster_by_local_index.get(
                                            local_index,
                                            0,
                                        ),
                                    }
                                )
                                placements.append(placement)
                        return placements, segment_frames, total_frames

                    placements = []
                    rows_by_segment: dict[str, list[dict[str, Any]]] = {}
                    for row in timeline_rows:
                        rows_by_segment.setdefault(str(row["segment_id"]), []).append(row)
                    for _index, segment, _segment_values in span:
                        segment_rows = rows_by_segment.get(segment.id, [])
                        if not segment_rows:
                            continue
                        segment_end_frame = segment_offsets[segment.id] + segment_frames[segment.id]
                        first_start = max(
                            segment_offsets[segment.id],
                            min(int(segment_rows[0]["source_start_frame"]), segment_end_frame - 1),
                        )
                        available_frames = max(1, segment_end_frame - first_start)
                        value_count = max(1, len(segment_rows))
                        for local_index, row in enumerate(segment_rows):
                            slot_start = first_start + int(round(local_index * available_frames / value_count))
                            slot_end = first_start + int(round((local_index + 1) * available_frames / value_count))
                            placement = placement_payload_from_row(row)
                            placement.update(
                                {
                                    "slot_start_frame": slot_start,
                                    "slot_end_frame": max(slot_start + 1, min(segment_end_frame, slot_end)),
                                    "slot_duration_sec": max(slot_end - slot_start, 1) / sample_rate,
                                    "placement_anchor": "source_smoothed_grid",
                                }
                            )
                            placements.append(placement)
                    return placements, segment_frames, total_frames

                placements = []
                for row_index, row in enumerate(timeline_rows):
                    slot_start = int(row["source_start_frame"])
                    next_start = (
                        int(timeline_rows[row_index + 1]["source_start_frame"])
                        if row_index + 1 < len(timeline_rows)
                        else int(row["segment_end_frame"])
                    )
                    source_period_frames = max(1, next_start - slot_start)
                    slot_frames = max(source_period_frames, int(row["equal_slot_frames"]))
                    placement = placement_payload_from_row(row)
                    placement.update(
                        {
                            "slot_start_frame": slot_start,
                            "slot_end_frame": slot_start + slot_frames,
                            "slot_duration_sec": slot_frames / sample_rate,
                            "placement_anchor": (
                                "source_anchor_exact"
                                if row.get("timeline_kind") == "source_anchor_timeline"
                                else "source_word_start"
                            ),
                        }
                    )
                    placements.append(placement)
                return placements, segment_frames, total_frames

            return equal_grid_placements("even_grid"), segment_frames, total_frames

        def countdown_pack_tts_options(
            *,
            candidate_index: int,
            sequence_seed: int,
            tts_text_language: str,
            speed_factor: float | None = None,
        ) -> GPTSoVITSTTSOptions:
            resolved_speed_factor = (
                float(speed_factor)
                if speed_factor is not None
                else float(getattr(cfg, "gsv_countdown_token_speed_factor", cfg.gsv_tts_max_speed_factor))
            )
            return GPTSoVITSTTSOptions(
                seed=cfg.base_seed + 80_000 + sequence_seed + candidate_index * 1_009,
                speed_factor=resolved_speed_factor,
                text_lang=tts_text_language,
                top_k=cfg.gsv_top_k,
                top_p=cfg.gsv_top_p,
                temperature=float(getattr(cfg, "gsv_countdown_temperature", cfg.gsv_temperature)),
                text_split_method=cfg.gsv_text_split_method,
                fragment_interval=0.0,
                parallel_infer=cfg.gsv_parallel_infer,
                repetition_penalty=cfg.gsv_repetition_penalty,
                sample_steps=cfg.gsv_sample_steps,
                super_sampling=cfg.gsv_super_sampling,
                overlap_length=cfg.gsv_overlap_length,
                min_chunk_length=cfg.gsv_min_chunk_length,
            )

        def countdown_pack_key(
            *,
            values: list[int],
            compact_text: str,
            expected_text: str,
            prompt_kind: str,
            ref_style: str,
            tts_text_language: str,
            options: GPTSoVITSTTSOptions,
        ) -> tuple[str, dict[str, Any]]:
            options_payload = options.model_dump(mode="json")
            options_payload.pop("seed", None)
            payload = {
                "values": values,
                "compact_text": compact_text,
                "expected_text": expected_text,
                "prompt_kind": prompt_kind,
                "ref_style": ref_style,
                "source_language": cfg.source_language,
                "target_language": cfg.target_language,
                "tts_text_language": tts_text_language,
                "mix_sample_rate": cfg.mix_sample_rate,
                "options": options_payload,
            }
            encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
            sequence_label = "_".join(str(value) for value in values)
            return f"{sequence_label}_{hashlib.sha256(encoded).hexdigest()[:16]}", payload

        def countdown_pack_take_from_metadata(metadata_path: Path) -> dict[str, Any] | None:
            try:
                metadata = json.loads(metadata_path.read_text("utf-8"))
            except (OSError, json.JSONDecodeError):
                return None
            if not bool(metadata.get("approved")):
                return None
            phrase_path_value = metadata.get("phrase_path")
            if not phrase_path_value:
                return None
            phrase_path = Path(str(phrase_path_value))
            if not phrase_path.is_absolute():
                phrase_path = metadata_path.parent / phrase_path
            if not phrase_path.exists():
                return None
            try:
                data, sample_rate = load_audio(phrase_path)
            except Exception:
                return None
            data = _countdown_stereo(data)
            if sample_rate != cfg.mix_sample_rate:
                data = resample_linear(data, sample_rate, cfg.mix_sample_rate)
                sample_rate = cfg.mix_sample_rate
            metadata["phrase_path"] = str(phrase_path)
            metadata["metadata_path"] = str(metadata_path)
            metadata["duration_sec"] = round(len(data) / sample_rate, 6)
            if not isinstance(metadata.get("timing_qc"), dict):
                metadata["timing_qc"] = _countdown_audio_timing_qc(
                    data,
                    sample_rate,
                    expected_units=len(metadata.get("pack_key_payload", {}).get("values", []))
                    if isinstance(metadata.get("pack_key_payload"), dict)
                    else 0,
                )
            return {
                "audio": data,
                "sample_rate": sample_rate,
                "metadata": metadata,
                "phrase_path": phrase_path,
                "metadata_path": metadata_path,
            }

        def clone_countdown_pack_take(
            take: dict[str, Any],
            *,
            reused_existing: bool,
        ) -> dict[str, Any]:
            metadata = copy.deepcopy(take["metadata"])
            metadata["reused_existing"] = reused_existing
            return {
                "audio": np.array(take["audio"], copy=True),
                "sample_rate": int(take["sample_rate"]),
                "metadata": metadata,
                "phrase_path": Path(str(take["phrase_path"])),
                "metadata_path": Path(str(take["metadata_path"])),
            }

        def countdown_pack_take_summary(take: dict[str, Any]) -> dict[str, Any]:
            metadata = take["metadata"]
            summary_keys = (
                "pack_key",
                "candidate_index",
                "approved",
                "reused_existing",
                "phrase_path",
                "metadata_path",
                "duration_sec",
                "raw_duration_sec",
                "seed",
                "prompt_kind",
                "prompt_text",
                "expected_text",
                "speed_factor",
                "pronunciation_gate",
                "pronunciation_contract_ok",
                "sequence_gate",
                "sequence_contract_ok",
                "sequence_qc",
                "timing_qc",
                "selection_score",
                "chunks",
                "chunk_gap_sec",
                "reject_reason",
            )
            return {
                key: metadata[key]
                for key in summary_keys
                if key in metadata
            }

        def remember_countdown_pack_take(pack_key_value: str, take: dict[str, Any]) -> None:
            with countdown_pack_lock:
                countdown_pack_cache[pack_key_value] = {
                    "audio": np.array(take["audio"], copy=True),
                    "sample_rate": int(take["sample_rate"]),
                    "metadata": copy.deepcopy(take["metadata"]),
                    "phrase_path": str(take["phrase_path"]),
                    "metadata_path": str(take["metadata_path"]),
                }

        def cached_countdown_pack_take(pack_key_value: str) -> dict[str, Any] | None:
            with countdown_pack_lock:
                cached = countdown_pack_cache.get(pack_key_value)
            if cached is None:
                return None
            return clone_countdown_pack_take(cached, reused_existing=True)

        def synthesize_countdown_pack_take(
            *,
            pack_dir: Path,
            pack_key_value: str,
            pack_key_payload: dict[str, Any],
            values: list[int],
            compact_text: str,
            expected_text: str,
            prompt_kind: str,
            speed_factor: float,
            candidate_index: int,
            sequence_seed: int,
            first_index: int,
            first_segment: Segment,
            ref: GPTSoVITSRef,
            lane_index: int,
            tts_text_language: str,
            ref_style: str,
        ) -> dict[str, Any]:
            options = countdown_pack_tts_options(
                candidate_index=candidate_index,
                sequence_seed=sequence_seed,
                tts_text_language=tts_text_language,
                speed_factor=speed_factor,
            )
            phrase_path = pack_dir / f"cand_{candidate_index:02d}_phrase.wav"
            metadata_path = pack_dir / f"cand_{candidate_index:02d}.json"
            payload: dict[str, Any] = {
                "renderer": "countdown_canonical_pack",
                "token_text": compact_text,
                "prompt_kind": prompt_kind,
                "candidate_index": candidate_index,
                "ref_style": ref_style,
                "lane_index": lane_index,
                "gsv_url": None if mock else gsv_base_urls[lane_index],
                "pack_key": pack_key_value,
            }
            payload.update(_tts_request_debug_payload(compact_text, ref, options))
            if mock:
                _mock_synthesize(
                    phrase_path,
                    max(0.12, min(0.42 * len(expected_text.split()), first_segment.duration)),
                    options.seed,
                    cfg.mix_sample_rate,
                )
                payload["mock"] = True
            else:
                client = clients[lane_index]
                request = client.build_payload(compact_text, ref, options)
                payload.update(request.as_payload())
                client.synthesize_to_file(request, phrase_path)
            postprocess_tts_candidate(phrase_path, payload)
            raw_duration = duration_sec(phrase_path)
            data, sample_rate = load_audio(phrase_path)
            data = _countdown_stereo(data)
            if sample_rate != cfg.mix_sample_rate:
                data = resample_linear(data, sample_rate, cfg.mix_sample_rate)
                sample_rate = cfg.mix_sample_rate
            duration = len(data) / sample_rate
            timing_qc = _countdown_audio_timing_qc(
                data,
                sample_rate,
                expected_units=len(values),
            )
            pronunciation_qc = run_pronunciation_qc(
                segment=first_segment,
                candidate_path=phrase_path,
                expected_text=expected_text,
                candidate_duration_sec=duration,
            )
            pronunciation_gate = str((pronunciation_qc or {}).get("gate") or "").strip().lower()
            pronunciation_contract_ok = countdown_pronunciation_contract_ok(
                first_segment,
                pronunciation_qc,
            )
            sequence_qc = _countdown_sequence_qc(
                values,
                pronunciation_qc,
                expected_text=expected_text,
            )
            sequence_gate = str(sequence_qc.get("gate") or "").strip().lower()
            sequence_contract_ok = sequence_gate == "pass" or (
                sequence_gate in {"unavailable", "disabled", "skipped"}
                and not countdown_strict_pronunciation_required(first_segment)
            )
            approved = pronunciation_contract_ok and sequence_contract_ok
            if not pronunciation_contract_ok:
                reject_reason = "pronunciation_qc_failed"
            elif not sequence_contract_ok:
                reject_reason = "countdown_sequence_qc_failed"
            else:
                reject_reason = None
            metadata = {
                "pack_key": pack_key_value,
                "pack_key_payload": pack_key_payload,
                "renderer": "countdown_canonical_pack",
                "candidate_index": candidate_index,
                "seed": options.seed,
                "approved": approved,
                "reused_existing": False,
                "reject_reason": reject_reason,
                "prompt_kind": prompt_kind,
                "prompt_text": compact_text,
                "compact_text": compact_text,
                "expected_text": expected_text,
                "speed_factor": options.speed_factor,
                "phrase_path": str(phrase_path),
                "metadata_path": str(metadata_path),
                "duration_sec": round(duration, 6),
                "raw_duration_sec": round(raw_duration, 6),
                "pronunciation_gate": pronunciation_gate or "not_run",
                "pronunciation_contract_ok": pronunciation_contract_ok,
                "pronunciation_qc": pronunciation_qc,
                "sequence_gate": sequence_gate or "not_run",
                "sequence_contract_ok": sequence_contract_ok,
                "sequence_qc": sequence_qc,
                "timing_qc": timing_qc,
                "first_segment_id": first_segment.id,
                "first_segment_index": first_index,
                "source_language": cfg.source_language,
                "target_language": cfg.target_language,
                "tts_text_language": tts_text_language,
                "ref_style": ref_style,
                "payload": payload,
            }
            write_json_atomic(metadata_path, metadata)
            return {
                "audio": data,
                "sample_rate": sample_rate,
                "metadata": metadata,
                "phrase_path": phrase_path,
                "metadata_path": metadata_path,
            }

        def approved_countdown_pack_take(
            *,
            values: list[int],
            tokens: list[str],
            first_index: int,
            first_segment: Segment,
            ref: GPTSoVITSRef,
            lane_index: int,
            tts_text_language: str,
            ref_style: str,
            candidate_count: int,
            target_duration_sec: float,
            allow_chunked_fallback: bool = True,
        ) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
            sequence_seed = sum((index + 1) * int(value) for index, value in enumerate(values))
            prompt_specs = _countdown_canonical_pack_prompt_specs(values, tokens)
            if not prompt_specs:
                return None, []

            def annotate_take(take: dict[str, Any], pack_key_value: str) -> dict[str, Any]:
                metadata = take["metadata"]
                if not isinstance(metadata.get("timing_qc"), dict):
                    metadata["timing_qc"] = _countdown_audio_timing_qc(
                        _countdown_stereo(np.array(take["audio"], copy=False)),
                        int(take["sample_rate"]),
                        expected_units=len(values),
                    )
                metadata["selection_score"] = _countdown_pack_take_selection_score(
                    metadata,
                    target_duration_sec=target_duration_sec,
                )
                if bool(metadata.get("approved")):
                    remember_countdown_pack_take(pack_key_value, take)
                return take

            def chunk_ranges() -> list[tuple[int, int]]:
                if len(values) < 5:
                    return []
                return [(index, index + 1) for index in range(len(values))]

            def chunked_countdown_pack_take() -> dict[str, Any] | None:
                ranges = chunk_ranges()
                if not allow_chunked_fallback or not ranges:
                    return None
                expected_text = _countdown_space_phrase_tts_text(tokens)
                sequence_label = "_".join(str(value) for value in values)
                chunk_payload = {
                    "values": values,
                    "chunks": [
                        {
                            "values": values[start:end],
                            "tokens": tokens[start:end],
                        }
                        for start, end in ranges
                    ],
                    "renderer": "countdown_canonical_pack_chunked",
                    "ref_style": ref_style,
                    "source_language": cfg.source_language,
                    "target_language": cfg.target_language,
                    "tts_text_language": tts_text_language,
                    "mix_sample_rate": cfg.mix_sample_rate,
                }
                encoded = json.dumps(chunk_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
                pack_key_value = f"{sequence_label}_chunked_{hashlib.sha256(encoded).hexdigest()[:16]}"
                cache_key = f"{pack_key_value}:cand_chunked"
                cached = cached_countdown_pack_take(cache_key)
                if cached is not None:
                    return cached
                pack_dir = (
                    project_dir
                    / "work"
                    / "tts"
                    / "countdown"
                    / "canonical_pack_chunked"
                    / _countdown_chunk_label(ref_style)
                    / pack_key_value
                )
                pack_dir.mkdir(parents=True, exist_ok=True)
                metadata_path = pack_dir / "cand_chunked.json"
                loaded = countdown_pack_take_from_metadata(metadata_path)
                if loaded is not None:
                    return clone_countdown_pack_take(loaded, reused_existing=True)

                chunk_takes: list[dict[str, Any]] = []
                chunk_summaries: list[dict[str, Any]] = []
                chunk_candidate_count = max(1, min(6, max(3, candidate_count // 2)))
                for chunk_number, (start, end) in enumerate(ranges):
                    chunk_values = values[start:end]
                    chunk_tokens = tokens[start:end]
                    chunk_target_duration = max(
                        0.25,
                        target_duration_sec * (len(chunk_values) / max(len(values), 1)),
                    )
                    if len(chunk_values) == 1:
                        token_text = chunk_tokens[0]
                        with lane_locks[lane_index]:
                            carrier_candidates = countdown_carrier_candidates_for_slot(
                                token_text=token_text,
                                token_slot_sec=chunk_target_duration,
                                segment=first_segment,
                                ref=ref,
                                ref_style=ref_style,
                                lane_index=lane_index,
                                tts_text_language=tts_text_language,
                                source_prosody=None,
                            )
                        accepted_carrier_candidates: list[dict[str, Any]] = []
                        for candidate in carrier_candidates:
                            payload = candidate.get("payload", {})
                            pronunciation_qc = payload.get("pronunciation_qc")
                            chunk_sequence_qc = _countdown_sequence_qc(
                                chunk_values,
                                pronunciation_qc,
                                expected_text=token_text,
                            )
                            boundary_qc = payload.get("countdown_slice_boundary_qc")
                            boundary_gate = (
                                str(boundary_qc.get("gate") or "pass").strip().lower()
                                if isinstance(boundary_qc, dict)
                                else "pass"
                            )
                            if (
                                countdown_pronunciation_contract_ok(first_segment, pronunciation_qc)
                                and str(chunk_sequence_qc.get("gate") or "").strip().lower() == "pass"
                                and countdown_prosody_contract_ok(candidate)
                                and boundary_gate != "fail"
                                and payload.get("audio_qc", {}).get("gate") == "pass"
                            ):
                                accepted_carrier_candidates.append(candidate)
                        selected_carrier = (
                            max(
                                accepted_carrier_candidates,
                                key=lambda candidate: (
                                    countdown_carrier_quality_tier_rank(candidate.get("quality_tier")),
                                    float(candidate.get("selection_score") or 0.0),
                                ),
                            )
                            if accepted_carrier_candidates
                            else None
                        )
                        pronunciation_qc = (
                            selected_carrier.get("payload", {}).get("pronunciation_qc")
                            if selected_carrier is not None
                            else None
                        )
                        sequence_qc = _countdown_sequence_qc(
                            chunk_values,
                            pronunciation_qc,
                            expected_text=token_text,
                        )
                        sequence_gate = str(sequence_qc.get("gate") or "").strip().lower()
                        sequence_contract_ok = selected_carrier is not None and (
                            sequence_gate == "pass"
                            or (
                                sequence_gate in {"unavailable", "disabled", "skipped"}
                                and not countdown_strict_pronunciation_required(first_segment)
                            )
                        )
                        chunk_candidates = [
                            {key: value for key, value in candidate.items() if key != "audio"}
                            for candidate in carrier_candidates
                        ]
                        if selected_carrier is not None and sequence_contract_ok:
                            selected_audio = _countdown_stereo(
                                np.asarray(selected_carrier["audio"], dtype=np.float32)
                            )
                            selected_duration = float(selected_carrier["duration_sec"])
                            selected_payload = selected_carrier.get("payload", {})
                            selected_pronunciation_gate = str(
                                (pronunciation_qc or {}).get("gate")
                                or selected_carrier.get("pronunciation_gate")
                                or "not_run"
                            ).strip().lower()
                            chunk_take = {
                                "audio": selected_audio,
                                "sample_rate": cfg.mix_sample_rate,
                                "metadata": {
                                    "pack_key": f"carrier_chunk:{token_text}",
                                    "renderer": "countdown_canonical_pack",
                                    "candidate_index": int(selected_carrier.get("candidate_index") or 0),
                                    "seed": int(selected_payload.get("seed") or cfg.base_seed + 95_000 + chunk_number),
                                    "approved": True,
                                    "reused_existing": bool(
                                        selected_payload.get("countdown_bank", {}).get("reused_existing")
                                        if isinstance(selected_payload.get("countdown_bank"), dict)
                                        else False
                                    ),
                                    "reject_reason": None,
                                    "prompt_kind": "carrier_token_chunk",
                                    "prompt_text": str(
                                        selected_payload.get("carrier_text")
                                        or selected_payload.get("pack_text")
                                        or token_text
                                    ),
                                    "expected_text": token_text,
                                    "speed_factor": 1.0,
                                    "phrase_path": str(selected_carrier["output_path"]),
                                    "metadata_path": str(selected_carrier["output_path"]),
                                    "duration_sec": round(selected_duration, 6),
                                    "raw_duration_sec": round(selected_duration, 6),
                                    "pronunciation_gate": selected_pronunciation_gate,
                                    "pronunciation_contract_ok": True,
                                    "pronunciation_qc": pronunciation_qc,
                                    "sequence_gate": sequence_gate or "not_run",
                                    "sequence_contract_ok": sequence_contract_ok,
                                    "sequence_qc": sequence_qc,
                                    "timing_qc": _countdown_audio_timing_qc(
                                        selected_audio,
                                        cfg.mix_sample_rate,
                                        expected_units=1,
                                    ),
                                    "payload": selected_payload,
                                },
                                "phrase_path": Path(str(selected_carrier["output_path"])),
                                "metadata_path": Path(str(selected_carrier["output_path"])),
                            }
                        else:
                            chunk_take = None
                    else:
                        chunk_take, chunk_candidates = approved_countdown_pack_take(
                            values=chunk_values,
                            tokens=chunk_tokens,
                            first_index=first_index,
                            first_segment=first_segment,
                            ref=ref,
                            lane_index=lane_index,
                            tts_text_language=tts_text_language,
                            ref_style=ref_style,
                            candidate_count=chunk_candidate_count,
                            target_duration_sec=chunk_target_duration,
                            allow_chunked_fallback=False,
                        )
                    chunk_summaries.append(
                        {
                            "chunk_index": chunk_number,
                            "values": chunk_values,
                            "tokens": chunk_tokens,
                            "selected_take": countdown_pack_take_summary(chunk_take)
                            if chunk_take is not None
                            else None,
                            "take_candidates": chunk_candidates,
                        }
                    )
                    if chunk_take is None:
                        return None
                    chunk_takes.append(chunk_take)

                chunk_audios = [
                    _countdown_stereo(np.array(take["audio"], copy=True))
                    if int(take["sample_rate"]) == cfg.mix_sample_rate
                    else resample_linear(
                        _countdown_stereo(np.array(take["audio"], copy=True)),
                        int(take["sample_rate"]),
                        cfg.mix_sample_rate,
                    )
                    for take in chunk_takes
                ]
                configured_gap_frames = int(
                    round(
                        max(0.0, float(getattr(cfg, "gsv_countdown_pack_chunk_gap_sec", 0.16)))
                        * cfg.mix_sample_rate
                    )
                )
                target_frames = int(round(max(0.0, target_duration_sec) * cfg.mix_sample_rate))
                raw_frames = sum(len(audio) for audio in chunk_audios)
                if len(chunk_audios) > 1 and target_frames > raw_frames:
                    gap_frames = min(configured_gap_frames, (target_frames - raw_frames) // (len(chunk_audios) - 1))
                else:
                    gap_frames = 0
                pieces: list[np.ndarray] = []
                for chunk_index, audio in enumerate(chunk_audios):
                    if chunk_index > 0 and gap_frames > 0:
                        pieces.append(np.zeros((gap_frames, 2), dtype=np.float32))
                    pieces.append(audio)
                data = (
                    np.concatenate(pieces, axis=0)
                    if pieces
                    else np.zeros((0, 2), dtype=np.float32)
                )
                phrase_path = pack_dir / "cand_chunked_phrase.wav"
                write_audio(phrase_path, data, cfg.mix_sample_rate)
                raw_duration = duration_sec(phrase_path)
                pronunciation_qc = run_pronunciation_qc(
                    segment=first_segment,
                    candidate_path=phrase_path,
                    expected_text=expected_text,
                    candidate_duration_sec=raw_duration,
                )
                pronunciation_gate = str((pronunciation_qc or {}).get("gate") or "").strip().lower()
                pronunciation_contract_ok = countdown_pronunciation_contract_ok(
                    first_segment,
                    pronunciation_qc,
                )
                sequence_qc = _countdown_sequence_qc(
                    values,
                    pronunciation_qc,
                    expected_text=expected_text,
                )
                sequence_gate = str(sequence_qc.get("gate") or "").strip().lower()
                sequence_contract_ok = sequence_gate == "pass" or (
                    sequence_gate in {"unavailable", "disabled", "skipped"}
                    and not countdown_strict_pronunciation_required(first_segment)
                )
                approved = pronunciation_contract_ok and sequence_contract_ok
                if not pronunciation_contract_ok:
                    reject_reason = "pronunciation_qc_failed"
                elif not sequence_contract_ok:
                    reject_reason = "countdown_sequence_qc_failed"
                else:
                    reject_reason = None
                timing_qc = _countdown_audio_timing_qc(
                    data,
                    cfg.mix_sample_rate,
                    expected_units=len(values),
                )
                metadata = {
                    "pack_key": pack_key_value,
                    "pack_key_payload": chunk_payload,
                    "renderer": "countdown_canonical_pack",
                    "candidate_index": 10_000,
                    "seed": cfg.base_seed + 90_000 + sequence_seed,
                    "approved": approved,
                    "reused_existing": False,
                    "reject_reason": reject_reason,
                    "prompt_kind": "chunked_canonical_pack",
                    "prompt_text": " | ".join(
                        str(
                            (
                                summary.get("selected_take")
                                if isinstance(summary.get("selected_take"), dict)
                                else {}
                            ).get("prompt_text")
                            or ""
                        )
                        for summary in chunk_summaries
                    ),
                    "compact_text": expected_text,
                    "expected_text": expected_text,
                    "speed_factor": 1.0,
                    "phrase_path": str(phrase_path),
                    "metadata_path": str(metadata_path),
                    "duration_sec": round(len(data) / cfg.mix_sample_rate, 6),
                    "raw_duration_sec": round(raw_duration, 6),
                    "pronunciation_gate": pronunciation_gate or "not_run",
                    "pronunciation_contract_ok": pronunciation_contract_ok,
                    "pronunciation_qc": pronunciation_qc,
                    "sequence_gate": sequence_gate or "not_run",
                    "sequence_contract_ok": sequence_contract_ok,
                    "sequence_qc": sequence_qc,
                    "timing_qc": timing_qc,
                    "chunks": chunk_summaries,
                    "chunk_gap_sec": round(gap_frames / cfg.mix_sample_rate, 6),
                    "first_segment_id": first_segment.id,
                    "first_segment_index": first_index,
                    "source_language": cfg.source_language,
                    "target_language": cfg.target_language,
                    "tts_text_language": tts_text_language,
                    "ref_style": ref_style,
                    "payload": {
                        "renderer": "countdown_canonical_pack_chunked",
                        "chunk_count": len(chunk_takes),
                        "pack_key": pack_key_value,
                    },
                }
                write_json_atomic(metadata_path, metadata)
                if not approved:
                    return None
                take = {
                    "audio": data,
                    "sample_rate": cfg.mix_sample_rate,
                    "metadata": metadata,
                    "phrase_path": phrase_path,
                    "metadata_path": metadata_path,
                }
                remember_countdown_pack_take(cache_key, take)
                return take

            candidates: list[dict[str, Any]] = []
            approved_takes: list[dict[str, Any]] = []
            for candidate_index in range(max(1, candidate_count)):
                prompt_spec = _countdown_canonical_pack_candidate_prompt_spec(
                    values,
                    prompt_specs,
                    candidate_index,
                )
                compact_text = str(prompt_spec["prompt_text"])
                expected_text = str(prompt_spec["expected_text"])
                prompt_kind = str(prompt_spec["prompt_kind"])
                speed_factor = float(prompt_spec["speed_factor"])
                key_options = countdown_pack_tts_options(
                    candidate_index=0,
                    sequence_seed=sequence_seed,
                    tts_text_language=tts_text_language,
                    speed_factor=speed_factor,
                )
                pack_key_value, pack_key_payload = countdown_pack_key(
                    values=values,
                    compact_text=compact_text,
                    expected_text=expected_text,
                    prompt_kind=prompt_kind,
                    ref_style=ref_style,
                    tts_text_language=tts_text_language,
                    options=key_options,
                )
                cache_key = f"{pack_key_value}:cand_{candidate_index:02d}"
                cached = cached_countdown_pack_take(cache_key)
                if cached is not None:
                    cached = annotate_take(cached, cache_key)
                    candidates.append(countdown_pack_take_summary(cached))
                    if bool(cached["metadata"].get("approved")):
                        approved_takes.append(cached)
                    continue
                pack_dir = (
                    project_dir
                    / "work"
                    / "tts"
                    / "countdown"
                    / "canonical_pack"
                    / _countdown_chunk_label(ref_style)
                    / pack_key_value
                )
                pack_dir.mkdir(parents=True, exist_ok=True)
                metadata_path = pack_dir / f"cand_{candidate_index:02d}.json"
                loaded = countdown_pack_take_from_metadata(metadata_path)
                if loaded is not None:
                    reused = annotate_take(
                        clone_countdown_pack_take(loaded, reused_existing=True),
                        cache_key,
                    )
                    candidates.append(countdown_pack_take_summary(reused))
                    if bool(reused["metadata"].get("approved")):
                        approved_takes.append(reused)
                    continue
                with lane_locks[lane_index]:
                    take = synthesize_countdown_pack_take(
                        pack_dir=pack_dir,
                        pack_key_value=pack_key_value,
                        pack_key_payload=pack_key_payload,
                        values=values,
                        compact_text=compact_text,
                        expected_text=expected_text,
                        prompt_kind=prompt_kind,
                        speed_factor=speed_factor,
                        candidate_index=candidate_index,
                        sequence_seed=sequence_seed,
                        first_index=first_index,
                        first_segment=first_segment,
                        ref=ref,
                        lane_index=lane_index,
                        tts_text_language=tts_text_language,
                        ref_style=ref_style,
                    )
                take = annotate_take(take, cache_key)
                candidates.append(countdown_pack_take_summary(take))
                if bool(take["metadata"].get("approved")):
                    approved_takes.append(take)
            if not approved_takes:
                chunked_take = chunked_countdown_pack_take()
                if chunked_take is not None:
                    chunked_take = annotate_take(chunked_take, str(chunked_take["metadata"].get("pack_key") or ""))
                    candidates.append(countdown_pack_take_summary(chunked_take))
                    return chunked_take, candidates
                return None, candidates
            selected = max(
                approved_takes,
                key=lambda take: (
                    float(take["metadata"].get("selection_score") or 0.0),
                    -int(take["metadata"].get("candidate_index") or 0),
                ),
            )
            return selected, candidates

        def render_countdown_span_canonical_pack(
            span: list[tuple[int, Segment, list[int]]],
        ) -> set[str]:
            first_index, first_segment, _first_values = span[0]
            if not first_segment.script:
                return set()
            values = [value for _index, _segment, segment_values in span for value in segment_values]
            tokens = _countdown_spoken_tokens(values)
            if tokens is None:
                return set()
            start_gsv_servers()
            span_id = "countdown_" + "_".join(segment.id for _index, segment, _values in span)
            span_dir = project_dir / "work" / "tts" / "countdown"
            slice_dir = span_dir / "canonical_pack_slices"
            span_dir.mkdir(parents=True, exist_ok=True)
            slice_dir.mkdir(parents=True, exist_ok=True)
            ref_style = first_segment.script.ref_style
            resolved_ref_style = ref_style if ref_style in refs else "whisper_close"
            ref = _ref_for_tts_language(resolve_ref(refs, ref_style), first_segment.script.tts_language)
            sample_rate = cfg.mix_sample_rate
            placements, segment_frames, total_frames = countdown_token_slots_for_span(
                span,
                sample_rate,
            )
            if len(placements) != len(tokens):
                return set()
            lane_index = _segment_lane_index(first_segment, first_index - 1, gsv_lane_count)
            candidate_count = int(getattr(cfg, "gsv_countdown_candidate_count", 8))
            total_duration_sec = total_frames / sample_rate
            prompt_specs = _countdown_canonical_pack_prompt_specs(values, tokens)
            compact_text = str(prompt_specs[0]["prompt_text"]) if prompt_specs else _countdown_phrase_tts_text(tokens)
            tts_text_language = _segment_tts_text_language(first_segment, cfg.target_language)
            selected_take, take_candidates = approved_countdown_pack_take(
                values=values,
                tokens=tokens,
                first_index=first_index,
                first_segment=first_segment,
                ref=ref,
                lane_index=lane_index,
                tts_text_language=tts_text_language,
                ref_style=resolved_ref_style,
                candidate_count=candidate_count,
                target_duration_sec=total_duration_sec,
            )
            if selected_take is None:
                skip_payload = {
                    "reason": "no_approved_countdown_canonical_pack_take",
                    "renderer": "countdown_canonical_pack",
                    "span_id": span_id,
                    "segment_ids": [segment.id for _index, segment, _values in span],
                    "values": values,
                    "tokens": tokens,
                    "compact_text": compact_text,
                    "prompt_specs": prompt_specs,
                    "take_candidates": take_candidates,
                }
                for _index, segment, _values in span:
                    segment.analysis["countdown_renderer_skip"] = skip_payload
                return set()

            selected_take_summary = countdown_pack_take_summary(selected_take)
            take_audio = _countdown_stereo(np.array(selected_take["audio"], copy=True))
            if selected_take["sample_rate"] != sample_rate:
                take_audio = resample_linear(take_audio, int(selected_take["sample_rate"]), sample_rate)
            take_audio, retime_qc = _countdown_retime_phrase_audio(
                take_audio,
                sample_rate,
                expected_units=len(values),
                enabled=bool(getattr(cfg, "gsv_countdown_pack_retime_enabled", True)),
                trigger_max_gap_sec=float(
                    getattr(cfg, "gsv_countdown_pack_retime_trigger_max_gap_sec", 0.55)
                ),
                trigger_gap_cv=float(getattr(cfg, "gsv_countdown_pack_retime_trigger_gap_cv", 0.35)),
                target_min_gap_sec=float(
                    getattr(cfg, "gsv_countdown_pack_retime_target_min_gap_sec", 0.14)
                ),
                target_max_gap_sec=float(
                    getattr(cfg, "gsv_countdown_pack_retime_target_max_gap_sec", 0.28)
                ),
                leading_sec=float(getattr(cfg, "gsv_countdown_pack_retime_leading_sec", 0.08)),
                trailing_sec=float(getattr(cfg, "gsv_countdown_pack_retime_trailing_sec", 0.12)),
            )
            if bool(retime_qc.get("applied")):
                retime_dir = span_dir / "canonical_pack_retimed"
                retime_dir.mkdir(parents=True, exist_ok=True)
                retime_path = (
                    retime_dir
                    / f"{span_id}_take_cand_{int(selected_take_summary.get('candidate_index', 0)):02d}_retimed.wav"
                )
                write_audio(retime_path, take_audio, sample_rate)
                retime_qc["retimed_phrase_path"] = str(retime_path)
                selected_take_summary["retimed_phrase_path"] = str(retime_path)
                selected_take_summary["duration_sec"] = round(len(take_audio) / sample_rate, 6)
                if isinstance(retime_qc.get("post_timing_qc"), dict):
                    selected_take_summary["timing_qc"] = retime_qc["post_timing_qc"]
            selected_take_summary["retime_qc"] = retime_qc

            def direct_take_token_duration_qc() -> dict[str, Any]:
                boundaries = countdown_phrase_slice_boundaries(
                    take_audio,
                    len(placements),
                    sample_rate,
                )
                if len(boundaries) != len(placements):
                    return {
                        "acceptable": False,
                        "reason": "slice_boundary_count_mismatch",
                        "slice_count": len(boundaries),
                        "token_count": len(placements),
                    }
                slices: list[dict[str, Any]] = []
                for placement, (slice_start, slice_end) in zip(placements, boundaries, strict=True):
                    token_text = str(placement["text"])
                    slice_duration = max(slice_end - slice_start, 1) / sample_rate
                    duration_gate, token_min_sec, token_max_sec = countdown_token_duration_gate(
                        duration=slice_duration,
                        token_slot_sec=float(placement["slot_duration_sec"]),
                        token_text=token_text,
                    )
                    slices.append(
                        {
                            "segment_id": placement["segment_id"],
                            "value": placement["value"],
                            "text": token_text,
                            "token_index": placement["token_index"],
                            "slice_start_sec": round(slice_start / sample_rate, 6),
                            "slice_end_sec": round(slice_end / sample_rate, 6),
                            "duration_sec": round(slice_duration, 6),
                            "duration_gate": duration_gate,
                            "token_min_sec": round(token_min_sec, 6),
                            "token_max_sec": round(token_max_sec, 6),
                        }
                    )
                acceptable = all(item["duration_gate"] == "pass" for item in slices)
                return {
                    "acceptable": acceptable,
                    "reason": "pass" if acceptable else "token_duration_gate_failed",
                    "slices": slices,
                }

            min_occupancy = float(getattr(cfg, "gsv_countdown_pack_min_span_occupancy", 0.55))
            min_direct_frames = int(round(total_frames * min_occupancy))
            selected_metadata = selected_take.get("metadata") if isinstance(selected_take, dict) else {}
            selected_sequence_qc = (
                selected_metadata.get("sequence_qc") if isinstance(selected_metadata, dict) else None
            )
            take_to_span_ratio = len(take_audio) / max(total_frames, 1)
            low_span_occupancy = len(take_audio) < min_direct_frames
            direct_sequence_take_ok = (
                isinstance(selected_sequence_qc, dict)
                and str(selected_sequence_qc.get("gate") or "").strip().lower() == "pass"
                and bool(selected_sequence_qc.get("exact_match"))
                and len(take_audio) <= total_frames
            )
            placement_mode = "take"
            placement_fit: dict[str, Any] | None = None
            direct_take_duration_qc: dict[str, Any] | None = None
            if len(take_audio) > total_frames:
                required_tempo = len(take_audio) / max(total_frames, 1)
                placement_mode = "slice_grid"
                placement_fit = {
                    "reason": "take_exceeds_span_no_linear_resample",
                    "original_duration_sec": round(len(take_audio) / sample_rate, 6),
                    "target_duration_sec": round(total_duration_sec, 6),
                    "tempo": round(required_tempo, 6),
                }
            elif low_span_occupancy and len(tokens) > 1 and not direct_sequence_take_ok:
                placement_mode = "slice_grid"
            if placement_mode == "take" and len(tokens) > 1:
                if direct_sequence_take_ok:
                    direct_take_duration_qc = {
                        "acceptable": True,
                        "reason": "countdown_sequence_exact_full_take",
                        "take_to_span_ratio": round(take_to_span_ratio, 6),
                        "low_span_occupancy": bool(low_span_occupancy),
                    }
                else:
                    direct_take_duration_qc = direct_take_token_duration_qc()
                    if not bool(direct_take_duration_qc.get("acceptable")):
                        placement_mode = "slice_grid"
                        placement_fit = {
                            "reason": "direct_take_token_duration_qc_failed",
                            "token_duration_qc": direct_take_duration_qc,
                        }

            span_audio = np.zeros((total_frames, 2), dtype=np.float32)
            placement_metadata: list[dict[str, Any]] = []
            phrase_timeline: dict[str, Any] | None = None
            if placement_mode == "slice_grid":
                segment_by_id = {segment.id: segment for _index, segment, _values in span}
                boundaries = countdown_phrase_slice_boundaries(
                    take_audio,
                    len(placements),
                    sample_rate,
                )
                selected_slices: list[dict[str, Any]] = []
                for placement, (slice_start, slice_end) in zip(placements, boundaries, strict=True):
                    token_text = str(placement["text"])
                    token_label = _countdown_chunk_label(token_text)
                    slice_path = (
                        slice_dir
                        / f"{span_id}_take_cand_{int(selected_take_summary.get('candidate_index', 0)):02d}_"
                        f"token_{int(placement['token_index']):02d}_{token_label}.wav"
                    )
                    slice_audio = _countdown_stereo(take_audio[slice_start:slice_end])
                    write_audio(slice_path, slice_audio, sample_rate)
                    slice_duration = duration_sec(slice_path)
                    duration_gate, _token_min_sec, _token_max_sec = countdown_token_duration_gate(
                        duration=slice_duration,
                        token_slot_sec=float(placement["slot_duration_sec"]),
                        token_text=token_text,
                    )
                    pronunciation_qc = run_pronunciation_qc(
                        segment=segment_by_id[str(placement["segment_id"])],
                        candidate_path=slice_path,
                        expected_text=token_text,
                        candidate_duration_sec=slice_duration,
                        short_slice=True,
                    )
                    pronunciation_gate = str((pronunciation_qc or {}).get("gate") or "").strip().lower()
                    pronunciation_ok = countdown_pronunciation_contract_ok(
                        segment_by_id[str(placement["segment_id"])],
                        pronunciation_qc,
                    )
                    selected_slices.append(
                        {
                            "candidate_index": selected_take_summary.get("candidate_index", 0),
                            "output_path": str(slice_path),
                            "audio": slice_audio,
                            "duration_sec": slice_duration,
                            "duration_gate": duration_gate,
                            "pronunciation_gate": pronunciation_gate or "not_run",
                            "pronunciation_qc": pronunciation_qc,
                            "acceptable": duration_gate == "pass"
                            and pronunciation_ok
                            and countdown_prosody_contract_ok({"payload": {}}),
                            "slice_start_sec": round(slice_start / sample_rate, 6),
                            "slice_end_sec": round(slice_end / sample_rate, 6),
                        }
                    )
                if len(selected_slices) != len(placements) or not all(
                    item["acceptable"] for item in selected_slices
                ):
                    skip_payload = {
                        "reason": "canonical_pack_slice_qc_failed",
                        "renderer": "countdown_canonical_pack",
                        "span_id": span_id,
                        "segment_ids": [segment.id for _index, segment, _values in span],
                        "values": values,
                        "tokens": tokens,
                        "compact_text": compact_text,
                        "prompt_specs": prompt_specs,
                        "selected_take": selected_take_summary,
                        "direct_take_token_duration_qc": direct_take_duration_qc,
                        "slices": [
                            {key: value for key, value in item.items() if key != "audio"}
                            for item in selected_slices
                        ],
                    }
                    for _index, segment, _values in span:
                        segment.analysis["countdown_renderer_skip"] = skip_payload
                    return set()
                for placement, selected in zip(placements, selected_slices, strict=True):
                    slot_start = int(placement["slot_start_frame"])
                    slot_end = int(placement["slot_end_frame"])
                    slot_frames = max(1, slot_end - slot_start)
                    audio = _countdown_stereo(np.array(selected["audio"], copy=True))
                    placement_anchor = placement.get("placement_anchor")
                    if placement_anchor in {"source_anchor_exact", "source_anchor_smoothed"}:
                        start_frame = max(0, min(slot_start, total_frames))
                    elif placement_anchor == "source_word_start":
                        start_frame = min(slot_start, max(0, total_frames - len(audio)))
                    else:
                        start_frame = slot_start + max(0, (slot_frames - len(audio)) // 2)
                    end_frame = min(total_frames, start_frame + len(audio))
                    source_frames = max(0, end_frame - start_frame)
                    if source_frames:
                        span_audio[start_frame:end_frame] += audio[:source_frames]
                    placement_metadata.append(
                        {
                            "segment_id": placement["segment_id"],
                            "value": placement["value"],
                            "text": placement["text"],
                            "token_index": placement["token_index"],
                            "slot_start_sec": round(slot_start / sample_rate, 6),
                            "slot_end_sec": round(slot_end / sample_rate, 6),
                            "slot_duration_sec": round(float(placement["slot_duration_sec"]), 6),
                            "placed_start_sec": round(start_frame / sample_rate, 6),
                            "placed_end_sec": round(end_frame / sample_rate, 6),
                            "selected_candidate_index": selected["candidate_index"],
                            "selected_duration_sec": round(float(selected["duration_sec"]), 6),
                            "selected_path": selected["output_path"],
                            "selected_pronunciation_gate": selected["pronunciation_gate"],
                            "placement_anchor": placement.get("placement_anchor", "slot_center"),
                        }
                    )
            else:
                source_aligned_placements = [
                    placement
                    for placement in placements
                    if str(placement.get("placement_anchor") or "").startswith("source_")
                ]
                if source_aligned_placements:
                    first_source_frame = min(
                        int(placement["slot_start_frame"])
                        for placement in source_aligned_placements
                    )
                    start_frame = min(
                        max(0, first_source_frame),
                        max(0, total_frames - len(take_audio)),
                    )
                    full_take_anchor = "source_phrase_start"
                else:
                    start_frame = max(0, (total_frames - len(take_audio)) // 2)
                    full_take_anchor = "span_center"
                end_frame = min(total_frames, start_frame + len(take_audio))
                if end_frame > start_frame:
                    span_audio[start_frame:end_frame] += take_audio[: end_frame - start_frame]
                phrase_timeline = {
                    "start_sec": round(start_frame / sample_rate, 6),
                    "end_sec": round(end_frame / sample_rate, 6),
                    "placement_anchor": full_take_anchor,
                }

            peak = float(np.max(np.abs(span_audio))) if span_audio.size else 0.0
            if peak > 0.98:
                span_audio *= 0.98 / peak
            span_path = span_dir / f"{span_id}.wav"
            write_audio(span_path, span_audio, sample_rate)
            span_metadata = {
                "span_id": span_id,
                "renderer": "countdown_canonical_pack",
                "placement_mode": placement_mode,
                "segment_ids": [segment.id for _index, segment, _values in span],
                "values": values,
                "tokens": tokens,
                "compact_text": compact_text,
                "prompt_specs": prompt_specs,
                "target_duration_sec": round(total_duration_sec, 6),
                "span_path": str(span_path),
                "selected_take": selected_take_summary,
                "retime_qc": retime_qc,
                "take_candidates": take_candidates,
                "token_placements": placement_metadata,
                "phrase_timeline": phrase_timeline,
                "placement_fit": placement_fit,
                "direct_take_token_duration_qc": direct_take_duration_qc,
                "source_language": cfg.source_language,
                "target_language": cfg.target_language,
                "ref_style": resolved_ref_style,
            }
            metadata_path = span_dir / f"{span_id}.json"
            write_json_atomic(metadata_path, span_metadata)

            offset = 0
            for _segment_index, (_index, segment, segment_values) in enumerate(span):
                frames = segment_frames[segment.id]
                segment_audio = span_audio[offset : offset + frames]
                offset += frames
                if len(segment_audio) < frames:
                    padding = np.zeros((frames - len(segment_audio), 2), dtype=np.float32)
                    segment_audio = np.concatenate([segment_audio, padding], axis=0)
                elif len(segment_audio) > frames:
                    segment_audio = segment_audio[:frames]
                final_path = project_dir / "work" / "tts" / f"{segment.id}_final.wav"
                write_audio(final_path, segment_audio, sample_rate)
                final_duration = duration_sec(final_path)
                final_ratio = duration_ratio(final_duration, segment.duration)
                segment_tokens = _countdown_spoken_tokens(segment_values) or []
                segment_placements = [
                    placement
                    for placement in placement_metadata
                    if placement["segment_id"] == segment.id
                ]
                payload = {
                    "renderer": "countdown_canonical_pack",
                    "placement_mode": placement_mode,
                    "span_id": span_id,
                    "span_metadata_path": str(metadata_path),
                    "span_path": str(span_path),
                    "segment_id": segment.id,
                    "segment_values": segment_values,
                    "segment_tokens": segment_tokens,
                    "values": values,
                    "tokens": tokens,
                    "selected_take": selected_take_summary,
                    "retime_qc": retime_qc,
                    "direct_take_token_duration_qc": direct_take_duration_qc,
                    "token_placements": segment_placements,
                    "phrase_timeline": phrase_timeline,
                    "target_duration_sec": segment.duration,
                    "duration_ratio": final_ratio,
                    "duration_gate": "pass",
                    "audio_qc": {
                        "gate": "pass",
                        "peak_dbfs": round(peak_dbfs(final_path), 3),
                        "rms_dbfs": round(rms_dbfs(final_path), 3),
                    },
                }
                candidate = TTSCandidate(
                    candidate_index=0,
                    seed=int(selected_take_summary.get("seed") or cfg.base_seed + 80_000 + first_index),
                    payload=payload,
                    output_path=str(final_path),
                    duration_sec=final_duration,
                    backend="gpt-sovits-countdown-renderer",
                    selected=True,
                    duration_ratio=final_ratio,
                    duration_gate="pass",
                    acceptable_for_mix=True,
                    selection_score=max(0.0, 1.0 - min(abs(final_ratio - 1.0), 1.0)),
                    selection_reason="countdown_canonical_pack",
                    retry_summary={"countdown_renderer": True, "span_id": span_id},
                )
                segment.tts = TTSMetadata(
                    backend="gpt-sovits-countdown-renderer",
                    ref_style=resolved_ref_style,
                    speed_factor=float(
                        selected_take_summary.get(
                            "speed_factor",
                            getattr(cfg, "gsv_countdown_token_speed_factor", cfg.gsv_tts_max_speed_factor),
                        )
                    ),
                    candidate_count=1,
                    selected_candidate_path=str(final_path),
                    candidates=[candidate],
                    source_language=cfg.source_language,
                    target_language=cfg.target_language,
                    cross_lingual_voice_transfer=cfg.source_language != cfg.target_language,
                    retry_summary={
                        "countdown_renderer": True,
                        "countdown_renderer_mode": "canonical_pack",
                        "span_id": span_id,
                        "span_metadata_path": str(metadata_path),
                        "selected_duration_gate": "pass",
                        "selected_acceptable_for_mix": True,
                        "selected_duration_ratio": final_ratio,
                        "countdown_retime_applied": bool(retime_qc.get("applied")),
                    },
                )
                segment.analysis["countdown_renderer"] = {
                    "renderer": "countdown_canonical_pack",
                    "placement_mode": placement_mode,
                    "span_id": span_id,
                    "segment_values": segment_values,
                    "segment_tokens": segment_tokens,
                    "selected_take": selected_take_summary,
                    "direct_take_token_duration_qc": direct_take_duration_qc,
                    "token_placements": segment_placements,
                    "phrase_timeline": phrase_timeline,
                    "span_metadata_path": str(metadata_path),
                }
                segment.status = "synthesized"
            return {segment.id for _index, segment, _values in span}

        def render_countdown_span_token(span: list[tuple[int, Segment, list[int]]]) -> set[str]:
            first_index, first_segment, _first_values = span[0]
            if not first_segment.script:
                return set()
            values = [value for _index, _segment, segment_values in span for value in segment_values]
            tokens = _countdown_spoken_tokens(values)
            if tokens is None:
                return set()
            start_gsv_servers()
            span_id = "countdown_" + "_".join(segment.id for _index, segment, _values in span)
            span_dir = project_dir / "work" / "tts" / "countdown"
            span_dir.mkdir(parents=True, exist_ok=True)
            ref_style = first_segment.script.ref_style
            resolved_ref_style = ref_style if ref_style in refs else "whisper_close"
            ref = _ref_for_tts_language(resolve_ref(refs, ref_style), first_segment.script.tts_language)
            sample_rate = cfg.mix_sample_rate
            placements, segment_frames, total_frames = countdown_token_slots_for_span(
                span,
                sample_rate,
            )
            lane_index = _segment_lane_index(first_segment, first_index - 1, gsv_lane_count)
            candidate_count = int(getattr(cfg, "gsv_countdown_candidate_count", 8))
            span_audio = np.zeros((total_frames, 2), dtype=np.float32)
            placement_metadata: list[dict[str, Any]] = []
            all_candidates: list[dict[str, Any]] = []
            failed_token_payloads: list[dict[str, Any]] = []
            segment_by_id = {segment.id: segment for _index, segment, _values in span}

            for placement in placements:
                token_candidates: list[dict[str, Any]] = []
                with lane_locks[lane_index]:
                    for candidate_index in range(candidate_count):
                        token_candidates.append(
                            countdown_token_candidate_audio(
                                token_text=_countdown_tts_text(str(placement["text"])),
                                token_index=int(placement["token_index"]),
                                candidate_index=candidate_index,
                                span_id=span_id,
                                segment=segment_by_id[str(placement["segment_id"])],
                                ref=ref,
                                token_slot_sec=float(placement["slot_duration_sec"]),
                                lane_index=lane_index,
                                tts_text_language=_segment_tts_text_language(
                                    first_segment,
                                    cfg.target_language,
                                ),
                                ref_style=resolved_ref_style,
                                source_prosody=countdown_source_prosody_for_placement(
                                    segment_by_id[str(placement["segment_id"])],
                                    placement,
                                ),
                            )
                        )
                accepted = [candidate for candidate in token_candidates if candidate["acceptable"]]
                all_candidates.extend(
                    {
                        key: value
                        for key, value in candidate.items()
                        if key != "audio"
                    }
                    for candidate in token_candidates
                )
                if not accepted:
                    failed_token_payloads.append(
                        {
                            "reason": "no_acceptable_countdown_token_candidate",
                            "failed_token": placement["text"],
                            "failed_value": placement["value"],
                            "slot_duration_sec": round(float(placement["slot_duration_sec"]), 6),
                            "placement_anchor": placement.get("placement_anchor", "slot_center"),
                            "candidates": [
                                {key: value for key, value in candidate.items() if key != "audio"}
                                for candidate in token_candidates
                            ],
                        }
                    )
                    continue
                selected = max(
                    accepted,
                    key=lambda candidate: (
                        float(candidate["selection_score"]),
                        -abs(float(candidate["duration_sec"]) - float(placement["slot_duration_sec"]) * 0.58),
                    ),
                )
                slot_start = int(placement["slot_start_frame"])
                slot_end = int(placement["slot_end_frame"])
                slot_frames = max(1, slot_end - slot_start)
                audio = selected["audio"]
                placement_anchor = placement.get("placement_anchor")
                if placement_anchor in {"source_anchor_exact", "source_anchor_smoothed"}:
                    start_frame = max(0, min(slot_start, total_frames))
                elif placement_anchor == "source_word_start":
                    start_frame = min(slot_start, max(0, total_frames - len(audio)))
                else:
                    start_frame = slot_start + max(0, (slot_frames - len(audio)) // 2)
                end_frame = min(total_frames, start_frame + len(audio))
                source_frames = max(0, end_frame - start_frame)
                if source_frames:
                    span_audio[start_frame:end_frame] += audio[:source_frames]
                selected_pronunciation = selected["payload"].get("pronunciation_qc")
                selected_pronunciation_gate = (
                    str(selected_pronunciation.get("gate") or "not_run")
                    if isinstance(selected_pronunciation, dict)
                    else "not_run"
                )
                placement_metadata.append(
                    {
                        "segment_id": placement["segment_id"],
                        "value": placement["value"],
                        "text": placement["text"],
                        "token_index": placement["token_index"],
                        "slot_start_sec": round(slot_start / sample_rate, 6),
                        "slot_end_sec": round(slot_end / sample_rate, 6),
                        "slot_duration_sec": round(float(placement["slot_duration_sec"]), 6),
                        "placed_start_sec": round(start_frame / sample_rate, 6),
                        "placed_end_sec": round(end_frame / sample_rate, 6),
                        "selected_candidate_index": selected["candidate_index"],
                        "selected_duration_sec": round(float(selected["duration_sec"]), 6),
                        "selected_path": selected["output_path"],
                        "selected_pronunciation_gate": selected_pronunciation_gate,
                        "selected_prosody_score": selected["payload"]
                        .get("countdown_prosody_qc", {})
                        .get("score"),
                        "candidate_count": len(token_candidates),
                        "placement_anchor": placement.get("placement_anchor", "slot_center"),
                        "rejected_candidates": [
                            {
                                "candidate_index": candidate["candidate_index"],
                                "duration_sec": round(float(candidate["duration_sec"]), 6),
                                "duration_gate": candidate["duration_gate"],
                                "output_path": candidate["output_path"],
                            }
                            for candidate in token_candidates
                            if candidate is not selected
                        ],
                    }
                )

            if failed_token_payloads:
                skip_payload = {
                    "reason": "no_acceptable_countdown_token_candidate",
                    "renderer": "countdown_token_timeline",
                    "span_id": span_id,
                    "segment_ids": [segment.id for _index, segment, _values in span],
                    "failed_tokens": failed_token_payloads,
                    "token_candidates": all_candidates,
                }
                for _index, segment, _values in span:
                    segment.analysis["countdown_renderer_skip"] = skip_payload
                return set()

            peak = float(np.max(np.abs(span_audio))) if span_audio.size else 0.0
            if peak > 0.98:
                span_audio *= 0.98 / peak
            span_path = span_dir / f"{span_id}.wav"
            write_audio(span_path, span_audio, sample_rate)
            span_metadata = {
                "span_id": span_id,
                "renderer": "countdown_token_timeline",
                "segment_ids": [segment.id for _index, segment, _values in span],
                "values": values,
                "tokens": tokens,
                "target_duration_sec": round(total_frames / sample_rate, 6),
                "span_path": str(span_path),
                "token_placements": placement_metadata,
                "token_candidates": all_candidates,
                "source_language": cfg.source_language,
                "target_language": cfg.target_language,
                "ref_style": resolved_ref_style,
            }
            metadata_path = span_dir / f"{span_id}.json"
            write_json_atomic(metadata_path, span_metadata)

            offset = 0
            for _segment_index, (_index, segment, segment_values) in enumerate(span):
                frames = segment_frames[segment.id]
                segment_audio = span_audio[offset : offset + frames]
                offset += frames
                if len(segment_audio) < frames:
                    padding = np.zeros((frames - len(segment_audio), 2), dtype=np.float32)
                    segment_audio = np.concatenate([segment_audio, padding], axis=0)
                elif len(segment_audio) > frames:
                    segment_audio = segment_audio[:frames]
                final_path = project_dir / "work" / "tts" / f"{segment.id}_final.wav"
                write_audio(final_path, segment_audio, sample_rate)
                final_duration = duration_sec(final_path)
                final_ratio = duration_ratio(final_duration, segment.duration)
                segment_tokens = _countdown_spoken_tokens(segment_values) or []
                segment_placements = [
                    placement
                    for placement in placement_metadata
                    if placement["segment_id"] == segment.id
                ]
                payload = {
                    "renderer": "countdown_token_timeline",
                    "span_id": span_id,
                    "span_metadata_path": str(metadata_path),
                    "span_path": str(span_path),
                    "segment_id": segment.id,
                    "segment_values": segment_values,
                    "segment_tokens": segment_tokens,
                    "values": values,
                    "tokens": tokens,
                    "token_placements": segment_placements,
                    "target_duration_sec": segment.duration,
                    "duration_ratio": final_ratio,
                    "duration_gate": "pass",
                    "audio_qc": {
                        "gate": "pass",
                        "peak_dbfs": round(peak_dbfs(final_path), 3),
                        "rms_dbfs": round(rms_dbfs(final_path), 3),
                    },
                }
                candidate = TTSCandidate(
                    candidate_index=0,
                    seed=cfg.base_seed + 70_000 + first_index,
                    payload=payload,
                    output_path=str(final_path),
                    duration_sec=final_duration,
                    backend="gpt-sovits-countdown-renderer",
                    selected=True,
                    duration_ratio=final_ratio,
                    duration_gate="pass",
                    acceptable_for_mix=True,
                    selection_score=max(0.0, 1.0 - min(abs(final_ratio - 1.0), 1.0)),
                    selection_reason="countdown_token_timeline",
                    retry_summary={"countdown_renderer": True, "span_id": span_id},
                )
                segment.tts = TTSMetadata(
                    backend="gpt-sovits-countdown-renderer",
                    ref_style=resolved_ref_style,
                    speed_factor=float(
                        getattr(
                            cfg,
                            "gsv_countdown_token_speed_factor",
                            cfg.gsv_tts_max_speed_factor,
                        )
                    ),
                    candidate_count=1,
                    selected_candidate_path=str(final_path),
                    candidates=[candidate],
                    source_language=cfg.source_language,
                    target_language=cfg.target_language,
                    cross_lingual_voice_transfer=cfg.source_language != cfg.target_language,
                    retry_summary={
                        "countdown_renderer": True,
                        "countdown_renderer_mode": "token",
                        "span_id": span_id,
                        "span_metadata_path": str(metadata_path),
                        "selected_duration_gate": "pass",
                        "selected_acceptable_for_mix": True,
                        "selected_duration_ratio": final_ratio,
                    },
                )
                segment.analysis["countdown_renderer"] = {
                    "renderer": "countdown_token_timeline",
                    "span_id": span_id,
                    "segment_values": segment_values,
                    "segment_tokens": segment_tokens,
                    "token_placements": segment_placements,
                    "span_metadata_path": str(metadata_path),
                }
                segment.status = "synthesized"
            return {segment.id for _index, segment, _values in span}

        def countdown_phrase_slice_boundaries(
            phrase_audio: np.ndarray,
            token_count: int,
            sample_rate: int,
        ) -> list[tuple[int, int]]:
            total_frames = len(phrase_audio)
            if token_count <= 0 or total_frames <= 0:
                return []
            active_start, active_end = countdown_phrase_active_span(phrase_audio, token_count)
            pad_frames = int(
                round(
                    float(getattr(cfg, "gsv_countdown_phrase_slice_edge_pad_sec", 0.04))
                    * sample_rate
                )
            )
            active_frames = max(1, active_end - active_start)
            boundaries: list[tuple[int, int]] = []
            for index in range(token_count):
                base_start = active_start + int(round(index * active_frames / token_count))
                base_end = active_start + int(round((index + 1) * active_frames / token_count))
                slice_start = max(0, base_start - pad_frames)
                slice_end = min(total_frames, base_end + pad_frames)
                if slice_end <= slice_start:
                    slice_end = min(total_frames, slice_start + 1)
                boundaries.append((slice_start, slice_end))
            return boundaries

        def countdown_phrase_active_span(
            phrase_audio: np.ndarray,
            token_count: int = 1,
        ) -> tuple[int, int]:
            total_frames = len(phrase_audio)
            if total_frames <= 0:
                return 0, 0
            if phrase_audio.ndim == 1:
                envelope = np.abs(phrase_audio)
            else:
                envelope = np.max(np.abs(phrase_audio), axis=1)
            peak = float(np.max(envelope)) if envelope.size else 0.0
            if peak > 0.0:
                threshold = max(peak * 0.08, 10 ** (-55.0 / 20.0))
                active = np.flatnonzero(envelope >= threshold)
            else:
                active = np.array([], dtype=np.int64)
            if active.size:
                active_start = int(active[0])
                active_end = int(active[-1]) + 1
            else:
                active_start = 0
                active_end = total_frames
            if active_end - active_start < token_count:
                active_start = 0
                active_end = total_frames
            return active_start, active_end

        def countdown_carrier_base_slice(
            *,
            phrase_audio: np.ndarray,
            token_text: str,
            carrier_template: str,
            carrier_text: str,
            sample_rate: int,
            full_sentence_prefilter: dict[str, Any],
        ) -> dict[str, Any]:
            total_frames = len(phrase_audio)
            if total_frames <= 0:
                return {
                    "strategy": "empty_audio",
                    "start_frame": 0,
                    "end_frame": 0,
                    "start_sec": 0.0,
                    "end_sec": 0.0,
                }
            pad_frames = int(
                round(
                    float(getattr(cfg, "gsv_countdown_phrase_slice_edge_pad_sec", 0.04))
                    * sample_rate
                )
            )
            alignment = full_sentence_prefilter.get("token_alignment")
            if isinstance(alignment, dict):
                start_sec = _safe_float(alignment.get("start_sec"), 0.0)
                end_sec = _safe_float(alignment.get("end_sec"), 0.0)
                start = max(0, int(round(start_sec * sample_rate)) - pad_frames)
                end = min(total_frames, int(round(end_sec * sample_rate)) + pad_frames)
                if end > start:
                    return {
                        "strategy": str(alignment.get("source") or "asr_alignment"),
                        "start_frame": start,
                        "end_frame": end,
                        "start_sec": round(start / sample_rate, 6),
                        "end_sec": round(end / sample_rate, 6),
                        "alignment": alignment,
                    }

            active_start, active_end = countdown_phrase_active_span(phrase_audio, 1)
            active_frames = max(1, active_end - active_start)
            text_parts = _countdown_carrier_text_parts(carrier_text, token_text)
            if text_parts is not None:
                prefix_text, matched_token_text, suffix_text = text_parts
                if countdown_carrier_is_numeric_unit(
                    template=carrier_template,
                    carrier_text=carrier_text,
                    token_text=token_text,
                ):
                    window_values = [
                        max(0.04, float(value))
                        for value in getattr(
                            cfg,
                            "gsv_countdown_carrier_numeric_unit_onset_window_sec",
                            [0.18, 0.24, 0.30, 0.36],
                        )
                    ] or [0.18]
                    tail_pad_frames = max(
                        0,
                        int(
                            round(
                                float(
                                    getattr(
                                        cfg,
                                        "gsv_countdown_carrier_numeric_unit_tail_pad_sec",
                                        0.04,
                                    )
                                )
                                * sample_rate
                            )
                        ),
                    )
                    first_window_frames = max(1, int(round(window_values[0] * sample_rate)))
                    slice_start = active_start
                    slice_end = min(total_frames, active_start + first_window_frames + tail_pad_frames)
                    return {
                        "strategy": "numeric_unit_onset_window",
                        "start_frame": slice_start,
                        "end_frame": max(slice_start + 1, slice_end),
                        "start_sec": round(slice_start / sample_rate, 6),
                        "end_sec": round(max(slice_start + 1, slice_end) / sample_rate, 6),
                        "anchor": "numeric_unit_onset",
                        "window_sec": round(window_values[0], 6),
                        "tail_pad_sec": round(tail_pad_frames / sample_rate, 6),
                    }
                prefix_units = _countdown_alignment_units(prefix_text)
                token_units = max(2.5, _countdown_alignment_units(matched_token_text) * 2.5)
                suffix_units = _countdown_alignment_units(suffix_text)
                total_units = max(1.0, prefix_units + token_units + suffix_units)
                anchor_window_frames = max(
                    1,
                    int(round(0.42 * sample_rate)),
                )
                anchor = "text_ratio"
                if prefix_units <= 0.5:
                    base_start = active_start
                    base_end = min(active_end, active_start + anchor_window_frames)
                    anchor = "prefix_start"
                elif suffix_units <= 0.5:
                    base_start = max(active_start, active_end - anchor_window_frames)
                    base_end = active_end
                    anchor = "suffix_end"
                else:
                    base_start = active_start + int(round((prefix_units / total_units) * active_frames))
                    base_end = active_start + int(
                        round(((prefix_units + token_units) / total_units) * active_frames)
                    )
                slice_start = max(0, base_start - pad_frames)
                slice_end = min(total_frames, base_end + pad_frames)
                if slice_end <= slice_start:
                    slice_end = min(total_frames, slice_start + max(1, pad_frames * 2))
                return {
                    "strategy": "carrier_text_position",
                    "start_frame": slice_start,
                    "end_frame": slice_end,
                    "start_sec": round(slice_start / sample_rate, 6),
                    "end_sec": round(slice_end / sample_rate, 6),
                    "anchor": anchor,
                    "prefix_units": round(prefix_units, 6),
                    "token_units": round(token_units, 6),
                    "suffix_units": round(suffix_units, 6),
                }

            boundaries = countdown_phrase_slice_boundaries(phrase_audio, 3, sample_rate)
            if len(boundaries) >= 3:
                start, end = boundaries[1]
            else:
                start, end = 0, total_frames
            return {
                "strategy": "equal_active_slice",
                "start_frame": start,
                "end_frame": end,
                "start_sec": round(start / sample_rate, 6),
                "end_sec": round(end / sample_rate, 6),
            }

        def countdown_slice_stability_metrics(
            audio: np.ndarray,
            sample_rate: int,
        ) -> dict[str, Any]:
            data = _countdown_stereo(audio)
            mono = to_mono(data).astype(np.float32, copy=False)
            if len(mono) == 0:
                return {
                    "pitch_stability_score": 0.0,
                    "energy_stability_score": 0.0,
                    "stability_score": 0.0,
                    "voiced_frame_count": 0,
                }
            frame_len = max(128, int(round(sample_rate * 0.04)))
            hop = max(64, int(round(sample_rate * 0.02)))
            min_lag = max(1, int(round(sample_rate / 500.0)))
            max_lag = max(min_lag + 1, int(round(sample_rate / 70.0)))
            peak = float(np.max(np.abs(mono))) if mono.size else 0.0
            rms_threshold = max(peak * 0.05, 1e-5)
            f0_values: list[float] = []
            rms_values: list[float] = []
            for start in range(0, max(1, len(mono) - frame_len + 1), hop):
                frame = mono[start : start + frame_len]
                if len(frame) < frame_len:
                    break
                rms_value = float(np.sqrt(np.mean(np.square(frame))))
                if rms_value <= rms_threshold:
                    continue
                rms_values.append(rms_value)
                centered = frame - float(np.mean(frame))
                if not np.any(centered):
                    continue
                windowed = centered * np.hanning(len(centered))
                corr = np.correlate(windowed, windowed, mode="full")[len(windowed) - 1 :]
                if len(corr) <= max_lag or float(corr[0]) <= 0.0:
                    continue
                search = corr[min_lag:max_lag]
                if search.size == 0:
                    continue
                best_offset = int(np.argmax(search))
                best_lag = min_lag + best_offset
                confidence = float(search[best_offset]) / max(float(corr[0]), 1e-12)
                if confidence < 0.25:
                    continue
                f0_values.append(sample_rate / best_lag)
            if len(f0_values) >= 2:
                f0_array = np.asarray(f0_values, dtype=np.float32)
                f0_median = float(np.median(f0_array))
                f0_mad = float(np.median(np.abs(f0_array - f0_median)))
                pitch_cv = f0_mad / max(f0_median, 1e-6)
                pitch_score = max(0.0, 1.0 - min(pitch_cv * 8.0, 1.0))
            else:
                f0_median = float(f0_values[0]) if f0_values else 0.0
                pitch_cv = 0.0 if f0_values else 1.0
                pitch_score = 1.0 if f0_values else 0.35
            if len(rms_values) >= 2:
                rms_array = np.asarray(rms_values, dtype=np.float32)
                rms_median = float(np.median(rms_array))
                rms_mad = float(np.median(np.abs(rms_array - rms_median)))
                rms_cv = rms_mad / max(rms_median, 1e-8)
                energy_score = max(0.0, 1.0 - min(rms_cv * 6.0, 1.0))
            else:
                rms_cv = 0.0 if rms_values else 1.0
                energy_score = 1.0 if rms_values else 0.0
            stability_score = pitch_score * 0.75 + energy_score * 0.25
            return {
                "pitch_stability_score": round(pitch_score, 6),
                "energy_stability_score": round(energy_score, 6),
                "stability_score": round(stability_score, 6),
                "pitch_cv": round(pitch_cv, 6),
                "rms_cv": round(rms_cv, 6),
                "median_f0_hz": round(f0_median, 3),
                "voiced_frame_count": len(f0_values),
                "active_frame_count": len(rms_values),
            }

        def countdown_slice_boundary_qc(
            audio: np.ndarray,
            *,
            start_frame: int,
            end_frame: int,
            sample_rate: int,
            token_text: str,
        ) -> dict[str, Any]:
            original_start_frame = max(0, int(start_frame))
            start_backoff_sec = float(
                getattr(cfg, "gsv_countdown_carrier_slice_start_backoff_sec", 0.02)
            )
            adjusted_start_frame = _countdown_apply_slice_start_backoff(
                original_start_frame,
                sample_rate=sample_rate,
                backoff_sec=start_backoff_sec,
            )
            result = _countdown_extend_slice_end_to_energy_valley(
                audio,
                start_frame=adjusted_start_frame,
                end_frame=end_frame,
                sample_rate=sample_rate,
                token_text=token_text,
                enabled=bool(getattr(cfg, "gsv_countdown_carrier_energy_extend_enabled", True)),
                max_extension_sec=float(
                    getattr(cfg, "gsv_countdown_carrier_energy_extend_max_sec", 0.10)
                ),
                coda_max_extension_sec=float(
                    getattr(cfg, "gsv_countdown_carrier_energy_extend_coda_max_sec", 0.20)
                ),
                edge_threshold_ratio=float(
                    getattr(
                        cfg,
                        "gsv_countdown_carrier_energy_extend_edge_threshold_ratio",
                        0.12,
                    )
                ),
                quiet_threshold_ratio=float(
                    getattr(
                        cfg,
                        "gsv_countdown_carrier_energy_extend_quiet_threshold_ratio",
                        0.08,
                    )
                ),
            )
            result.update(
                {
                    "original_start_frame": int(original_start_frame),
                    "start_backoff_frames": int(original_start_frame - adjusted_start_frame),
                    "start_backoff_sec": round(
                        (original_start_frame - adjusted_start_frame) / sample_rate,
                        6,
                    )
                    if sample_rate > 0
                    else 0.0,
                    "configured_start_backoff_sec": round(max(0.0, start_backoff_sec), 6),
                }
            )
            return result

        def countdown_candidate_boundary_score(candidate: dict[str, Any]) -> float:
            payload = candidate.get("payload", {})
            boundary_qc = payload.get("countdown_slice_boundary_qc")
            if not isinstance(boundary_qc, dict):
                boundary_qc = payload.get("countdown_carrier_bank", {}).get("boundary_qc", {})
            if not isinstance(boundary_qc, dict):
                return 1.0
            gate = str(boundary_qc.get("gate") or "pass").strip().lower()
            if gate in {"", "disabled", "skipped", "unavailable"}:
                return 1.0
            cut_risk = max(0.0, min(_safe_float(boundary_qc.get("cut_risk_score"), 0.0), 1.0))
            if gate == "fail":
                return 0.0
            if gate == "warn":
                return max(0.05, 0.35 - cut_risk * 0.30)
            return max(0.65, 1.0 - cut_risk * 0.35)

        def countdown_candidate_anchor_score(candidate: dict[str, Any]) -> float | None:
            active_fit_score = _countdown_candidate_anchor_fit_score(candidate)
            if active_fit_score is not None:
                return active_fit_score
            carrier_bank = candidate.get("payload", {}).get("countdown_carrier_bank", {})
            if not isinstance(carrier_bank, dict) or not carrier_bank:
                return None
            slice_window = carrier_bank.get("slice_window", {})
            if isinstance(slice_window, dict) and slice_window.get("kind") == "carrier_text_position":
                return 1.0
            slice_start = _safe_float(carrier_bank.get("slice_start_sec"), 0.0)
            base_start = _safe_float(carrier_bank.get("base_slice_start_sec"), slice_start)
            late_by_sec = max(0.0, slice_start - base_start)
            return max(0.0, 1.0 - min(late_by_sec / 0.25, 1.0))

        def countdown_chunk_slice_score(
            candidate: dict[str, Any],
            token_slot_sec: float,
        ) -> float:
            duration = float(candidate["duration_sec"])
            token_text = str(candidate.get("text") or "")
            min_sec = float(getattr(cfg, "gsv_countdown_token_min_sec", 0.25))
            max_sec = countdown_token_max_allowed_sec(
                token_slot_sec,
                token_text=token_text,
            )
            target_sec = min(max_sec, max(min_sec, token_slot_sec * 0.58))
            duration_score = (
                max(0.0, 1.0 - min(abs(duration - target_sec) / target_sec, 1.0))
                if target_sec > 0
                else 0.0
            )
            pronunciation_score = countdown_token_pronunciation_score(candidate)
            stability_score = float(
                candidate["payload"]
                .get("stability_qc", {})
                .get("stability_score", 0.0)
            )
            boundary_score = countdown_candidate_boundary_score(candidate)
            anchor_score = countdown_candidate_anchor_score(candidate)
            anchor_weight = 0.04 if anchor_score is not None else 0.0
            stability_weight = 0.12 - anchor_weight
            prosody_score = countdown_candidate_prosody_score(candidate)
            texture_score = countdown_candidate_texture_score(candidate)
            peak = float(candidate["payload"].get("audio_qc", {}).get("peak_dbfs", -120.0))
            loudness_score = max(0.0, min((peak + 60.0) / 60.0, 1.0))
            return round(
                pronunciation_score * 0.32
                + texture_score * 0.20
                + prosody_score * 0.16
                + stability_score * stability_weight
                + boundary_score * 0.10
                + (anchor_score or 0.0) * anchor_weight
                + duration_score * 0.08
                + loudness_score * 0.02,
                6,
            )

        def countdown_carrier_templates_for_token(token_text: str) -> list[tuple[int, str, str]]:
            return _countdown_carrier_templates_for_token_config(cfg, token_text)

        def countdown_carrier_is_numeric_unit(
            *,
            template: str,
            carrier_text: str,
            token_text: str,
        ) -> bool:
            numeric_templates = {
                str(item).strip()
                for item in getattr(cfg, "gsv_countdown_carrier_numeric_unit_templates", [])
                if str(item).strip()
            }
            if template.strip() in numeric_templates:
                return True
            text = carrier_text.strip()
            if not text.startswith(token_text):
                return False
            suffix = text[len(token_text) :].strip()
            suffix = suffix.lstrip(",，、.。 ").strip()
            return suffix.startswith(("번", "초", "만", "만큼", "입니다", "이에요", "예요", "하고"))

        def countdown_carrier_ref_for_span(
            span: list[tuple[int, Segment, list[int]]],
            *,
            ref_style: str,
            resolved_ref_style: str,
        ) -> tuple[GPTSoVITSRef, dict[str, Any]]:
            first_segment = span[0][1]
            static_ref = resolve_ref(refs, ref_style)
            segment_ref, segment_ref_metadata = _segment_source_ref_for_gsv(
                project_dir,
                first_segment,
                cfg,
                manifest.segments,
            )
            selected_ref = segment_ref or static_ref
            metadata = {
                "requested_ref_style": ref_style,
                "resolved_ref_style": resolved_ref_style,
                "ref_audio_path": selected_ref.ref_audio_path,
                "prompt_text": selected_ref.prompt_text,
                "prompt_lang": selected_ref.prompt_lang,
                "segment_ref": segment_ref_metadata,
                "using_segment_ref": bool(segment_ref is not None),
            }
            return selected_ref, metadata

        def clone_countdown_carrier_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
            cloned = {
                key: copy.deepcopy(value)
                for key, value in candidate.items()
                if key != "audio"
            }
            cloned["audio"] = np.array(candidate["audio"], copy=True)
            return cloned

        def carrier_bank_key(
            *,
            token_text: str,
            ref: GPTSoVITSRef,
            ref_style: str,
            tts_text_language: str,
        ) -> tuple[Any, ...]:
            key_options = countdown_chunk_tts_options(
                candidate_index=0,
                sequence_seed=sum(ord(char) for char in token_text),
                tts_text_language=tts_text_language,
            )
            return (
                ref_style,
                ref.ref_audio_path,
                ref.prompt_text,
                ref.prompt_lang,
                tuple(
                    str(template).strip()
                    for template in (
                        (getattr(cfg, "gsv_countdown_carrier_token_templates", {}) or {}).get(
                            token_text,
                            [],
                        )
                        if isinstance(
                            getattr(cfg, "gsv_countdown_carrier_token_templates", {}),
                            dict,
                        )
                        else []
                    )
                ),
                tuple(str(template).strip() for template in getattr(cfg, "gsv_countdown_carrier_templates", [])),
                bool(getattr(cfg, "gsv_countdown_carrier_numeric_unit_enabled", True)),
                tuple(
                    str(template).strip()
                    for template in getattr(cfg, "gsv_countdown_carrier_numeric_unit_templates", [])
                ),
                tuple(
                    float(value)
                    for value in getattr(cfg, "gsv_countdown_carrier_numeric_unit_onset_window_sec", [])
                ),
                float(getattr(cfg, "gsv_countdown_carrier_numeric_unit_tail_pad_sec", 0.04)),
                int(getattr(cfg, "gsv_countdown_carrier_candidate_count", 1)),
                bool(getattr(cfg, "gsv_countdown_carrier_slice_search_enabled", True)),
                tuple(float(value) for value in getattr(cfg, "gsv_countdown_carrier_slice_window_sec", [])),
                tuple(
                    float(value)
                    for value in getattr(cfg, "gsv_countdown_carrier_slice_window_offset_sec", [])
                ),
                int(getattr(cfg, "gsv_countdown_carrier_max_slice_windows_per_candidate", 1)),
                bool(getattr(cfg, "gsv_countdown_carrier_full_sentence_prefilter_enabled", True)),
                float(getattr(cfg, "gsv_countdown_carrier_full_sentence_prefilter_min_coverage", 1.0)),
                bool(getattr(cfg, "gsv_countdown_carrier_quality_retry_enabled", True)),
                int(getattr(cfg, "gsv_countdown_carrier_quality_retry_max_rounds", 3)),
                str(getattr(cfg, "gsv_countdown_carrier_quality_retry_target_tier", "A")),
                bool(
                    getattr(
                        cfg,
                        "gsv_countdown_carrier_stop_window_search_after_pronunciation_pass",
                        True,
                    )
                ),
                int(getattr(cfg, "gsv_countdown_carrier_target_pronunciation_passes", 2)),
                token_text,
                key_options.speed_factor,
                key_options.text_lang,
                key_options.top_k,
                key_options.top_p,
                key_options.temperature,
                key_options.text_split_method,
                key_options.fragment_interval,
                key_options.parallel_infer,
                key_options.repetition_penalty,
                key_options.sample_steps,
                key_options.super_sampling,
                key_options.overlap_length,
                key_options.min_chunk_length,
            )

        def countdown_carrier_generation_profile(
            base_options: GPTSoVITSTTSOptions,
            variant_index: int,
        ) -> tuple[GPTSoVITSTTSOptions, dict[str, Any]]:
            profile_slot = variant_index % 3
            if profile_slot == 0:
                profile_name = "stable_low_temperature"
                top_k = max(1, min(int(base_options.top_k), 8))
                top_p = max(0.1, min(float(base_options.top_p), 0.85))
                temperature = max(0.0, float(base_options.temperature) - 0.20)
            elif profile_slot == 1:
                profile_name = "neutral"
                top_k = int(base_options.top_k)
                top_p = float(base_options.top_p)
                temperature = float(base_options.temperature)
            else:
                profile_name = "light_exploration"
                top_k = int(base_options.top_k)
                top_p = float(base_options.top_p)
                temperature = min(2.0, float(base_options.temperature) + 0.10)
            options = base_options.model_copy(
                update={
                    "top_k": top_k,
                    "top_p": top_p,
                    "temperature": temperature,
                    "seed": int(base_options.seed) + variant_index * 10_007,
                }
            )
            return options, {
                "name": profile_name,
                "variant_index": variant_index,
                "temperature": round(temperature, 6),
                "top_k": top_k,
                "top_p": round(top_p, 6),
            }

        def countdown_carrier_slice_windows(
            *,
            total_frames: int,
            base_start: int,
            base_end: int,
            sample_rate: int,
            base_kind: str = "equal_active_slice",
            base_anchor: str | None = None,
        ) -> list[dict[str, Any]]:
            windows: list[dict[str, Any]] = []
            seen: set[tuple[int, int]] = set()

            def add_window(
                *,
                start_frame: int,
                end_frame: int,
                kind: str,
                window_sec: float | None,
                offset_sec: float,
            ) -> None:
                start = max(0, min(total_frames - 1, int(start_frame)))
                end = max(start + 1, min(total_frames, int(end_frame)))
                key = (start, end)
                if key in seen:
                    return
                seen.add(key)
                windows.append(
                    {
                        "slice_window_index": len(windows),
                        "kind": kind,
                        "start_frame": start,
                        "end_frame": end,
                        "window_sec": None if window_sec is None else round(float(window_sec), 6),
                        "offset_sec": round(float(offset_sec), 6),
                    }
                )

            add_window(
                start_frame=base_start,
                end_frame=base_end,
                kind=base_kind,
                window_sec=None,
                offset_sec=0.0,
            )
            if not bool(getattr(cfg, "gsv_countdown_carrier_slice_search_enabled", True)):
                return windows

            center = (base_start + base_end) / 2.0
            window_values = [
                max(0.04, float(value))
                for value in getattr(cfg, "gsv_countdown_carrier_slice_window_sec", [])
            ]
            offset_values = [
                float(value)
                for value in getattr(cfg, "gsv_countdown_carrier_slice_window_offset_sec", [])
            ] or [0.0]
            max_windows = int(getattr(cfg, "gsv_countdown_carrier_max_slice_windows_per_candidate", 5))
            anchor = str(base_anchor or "").strip().lower()
            if anchor == "numeric_unit_onset":
                tail_pad_frames = max(
                    0,
                    int(
                        round(
                            float(
                                getattr(
                                    cfg,
                                    "gsv_countdown_carrier_numeric_unit_tail_pad_sec",
                                    0.04,
                                )
                            )
                            * sample_rate
                        )
                    ),
                )
                for window_sec in [
                    max(0.04, float(value))
                    for value in getattr(
                        cfg,
                        "gsv_countdown_carrier_numeric_unit_onset_window_sec",
                        [0.18, 0.24, 0.30, 0.36],
                    )
                ]:
                    window_frames = max(1, int(round(window_sec * sample_rate)))
                    add_window(
                        start_frame=base_start,
                        end_frame=base_start + window_frames + tail_pad_frames,
                        kind="numeric_unit_onset_window",
                        window_sec=window_sec,
                        offset_sec=0.0,
                    )
                    if len(windows) >= max_windows:
                        return windows
                return windows
            if anchor in {"suffix_end", "prefix_start"}:
                for window_sec in window_values:
                    window_frames = max(1, int(round(window_sec * sample_rate)))
                    step_frames = max(1, window_frames // 2)
                    max_steps = max(1, min(6, max_windows - len(windows)))
                    for step_index in range(max_steps):
                        if anchor == "suffix_end":
                            end_frame = base_end - step_index * step_frames
                            start_frame = end_frame - window_frames
                        else:
                            start_frame = base_start + step_index * step_frames
                            end_frame = start_frame + window_frames
                        add_window(
                            start_frame=start_frame,
                            end_frame=end_frame,
                            kind=f"{anchor}_scan_window",
                            window_sec=window_sec,
                            offset_sec=(step_index * step_frames / sample_rate)
                            * (-1.0 if anchor == "suffix_end" else 1.0),
                        )
                        if len(windows) >= max_windows:
                            return windows
            for window_sec in window_values:
                window_frames = max(1, int(round(window_sec * sample_rate)))
                half_window = window_frames // 2
                for offset_sec in offset_values:
                    center_frame = int(round(center + offset_sec * sample_rate))
                    add_window(
                        start_frame=center_frame - half_window,
                        end_frame=center_frame - half_window + window_frames,
                        kind="search_window",
                        window_sec=window_sec,
                        offset_sec=offset_sec,
                    )
                    if len(windows) >= max_windows:
                        return windows
            return windows

        def countdown_carrier_bank_early_stop_metadata() -> dict[str, Any]:
            return {
                "stop_window_search_after_pronunciation_pass": bool(
                    getattr(
                        cfg,
                        "gsv_countdown_carrier_stop_window_search_after_pronunciation_pass",
                        True,
                    )
                ),
                "target_pronunciation_passes": int(
                    getattr(cfg, "gsv_countdown_carrier_target_pronunciation_passes", 2)
                ),
            }

        def countdown_carrier_quality_retry_metadata() -> dict[str, Any]:
            return {
                "enabled": bool(getattr(cfg, "gsv_countdown_carrier_quality_retry_enabled", True)),
                "max_rounds": int(getattr(cfg, "gsv_countdown_carrier_quality_retry_max_rounds", 3)),
                "target_tier": str(
                    getattr(cfg, "gsv_countdown_carrier_quality_retry_target_tier", "A")
                ).upper(),
            }

        def countdown_carrier_full_sentence_prefilter_metadata() -> dict[str, Any]:
            return {
                "enabled": bool(
                    getattr(cfg, "gsv_countdown_carrier_full_sentence_prefilter_enabled", True)
                ),
                "min_coverage": round(
                    float(
                        getattr(
                            cfg,
                            "gsv_countdown_carrier_full_sentence_prefilter_min_coverage",
                            1.0,
                        )
                    ),
                    6,
                ),
            }

        def run_countdown_carrier_full_sentence_prefilter(
            *,
            segment: Segment,
            candidate_path: Path,
            expected_text: str,
            candidate_duration_sec: float,
        ) -> dict[str, Any]:
            metadata = countdown_carrier_full_sentence_prefilter_metadata()
            if not metadata["enabled"]:
                return {
                    **metadata,
                    "gate": "disabled",
                    "approved": True,
                    "coverage": 1.0,
                    "reason": "full_sentence_prefilter_disabled",
                }
            if (
                mock
                or _canonical_language(cfg.target_language) != "ko"
                or not bool(getattr(cfg, "gsv_pronunciation_qc_enabled", True))
            ):
                return {
                    **metadata,
                    "gate": "skipped",
                    "approved": True,
                    "coverage": 1.0,
                    "reason": "pronunciation_qc_skipped",
                }
            configured_backend = (
                str(getattr(cfg, "gsv_pronunciation_qc_backend", "auto"))
                .strip()
                .lower()
                .replace("-", "_")
            )
            if configured_backend == "auto" and (
                segment.source_script is None
                or str(segment.source_script.backend).strip().lower() == "mock"
            ):
                return {
                    **metadata,
                    "gate": "skipped",
                    "approved": True,
                    "coverage": 1.0,
                    "reason": "pronunciation_qc_auto_skipped_for_mock_source",
                }
            backend_name = pronunciation_qc_backend_name()
            worker_index = pronunciation_qc_worker_index()
            backend_cache_key = pronunciation_qc_worker_cache_key(backend_name, worker_index)
            if backend_cache_key in pronunciation_qc_unavailable_errors:
                return {
                    **metadata,
                    "gate": "unavailable",
                    "approved": True,
                    "coverage": 1.0,
                    "backend": backend_name,
                    "error": pronunciation_qc_unavailable_errors[backend_cache_key],
                    "worker_index": worker_index,
                    "workers": pronunciation_qc_worker_count,
                    "reason": "pronunciation_qc_unavailable",
                }
            try:
                with pronunciation_qc_backend_locks[worker_index]:
                    backend = pronunciation_qc_backend_cache.get(backend_cache_key)
                    if backend is None:
                        backend = create_asr_backend(
                            backend_name,
                            pronunciation_qc_backend_config(),
                        )
                        pronunciation_qc_backend_cache[backend_cache_key] = backend
                    qc_duration = max(float(candidate_duration_sec), 0.001)
                    qc_segment = Segment(
                        id=f"{segment.id}_carrier_full_sentence_prefilter",
                        speaker_id=segment.speaker_id,
                        start=0.0,
                        end=qc_duration,
                        duration=qc_duration,
                        audio_for_gemma=str(candidate_path),
                        audio_for_mix=str(candidate_path),
                    )
                    chunks = backend.transcribe(candidate_path, [qc_segment])
                transcript = " ".join(str(getattr(chunk, "text", "")).strip() for chunk in chunks).strip()
                match = _countdown_carrier_full_sentence_prefilter_match(
                    expected_text,
                    transcript,
                )
                token_alignment = _countdown_token_alignment_from_chunks(
                    expected_text,
                    chunks,
                    candidate_duration_sec,
                )
                min_coverage = float(metadata["min_coverage"])
                coverage = float(match.get("coverage") or 0.0)
                approved = bool(match.get("matched")) and coverage >= min_coverage
                return {
                    **metadata,
                    **match,
                    "gate": "pass" if approved else "fail",
                    "approved": approved,
                    "backend": backend_name,
                    "worker_index": worker_index,
                    "workers": pronunciation_qc_worker_count,
                    "token_alignment": token_alignment,
                    "reason": None if approved else match.get("reason", "target_token_not_found"),
                }
            except Exception as exc:
                pronunciation_qc_unavailable_errors[backend_cache_key] = str(exc)
                return {
                    **metadata,
                    "gate": "unavailable",
                    "approved": True,
                    "coverage": 1.0,
                    "backend": backend_name,
                    "error": str(exc),
                    "worker_index": worker_index,
                    "workers": pronunciation_qc_worker_count,
                    "reason": "pronunciation_qc_unavailable",
                }

        def countdown_pack_token_templates_for_token(token_text: str) -> list[tuple[int, str, str]]:
            templates = [
                str(template).strip()
                for template in getattr(cfg, "gsv_countdown_token_bank_pack_templates", [])
                if str(template).strip()
            ] or ["{token}, {token}, {token}"]
            rendered: list[tuple[int, str, str]] = []
            for pack_index, template in enumerate(templates):
                if "{token}" in template:
                    try:
                        pack_text = template.format(token=token_text)
                    except Exception:
                        pack_text = template.replace("{token}", token_text)
                else:
                    pack_text = f"{token_text}, {token_text}, {token_text}"
                rendered.append((pack_index, template, pack_text))
            return rendered

        def pack_token_bank_key(
            *,
            token_text: str,
            ref: GPTSoVITSRef,
            ref_style: str,
            tts_text_language: str,
        ) -> tuple[Any, ...]:
            key_options = countdown_chunk_tts_options(
                candidate_index=0,
                sequence_seed=sum(ord(char) for char in token_text),
                tts_text_language=tts_text_language,
            )
            return (
                "pack_token_bank",
                ref_style,
                ref.ref_audio_path,
                ref.prompt_text,
                ref.prompt_lang,
                tuple(str(template).strip() for template in getattr(cfg, "gsv_countdown_token_bank_pack_templates", [])),
                int(getattr(cfg, "gsv_countdown_carrier_candidate_count", 1)),
                bool(getattr(cfg, "gsv_countdown_carrier_quality_retry_enabled", True)),
                int(getattr(cfg, "gsv_countdown_carrier_quality_retry_max_rounds", 3)),
                token_text,
                key_options.speed_factor,
                key_options.text_lang,
                key_options.top_k,
                key_options.top_p,
                key_options.temperature,
                key_options.text_split_method,
                key_options.fragment_interval,
                key_options.parallel_infer,
                key_options.repetition_penalty,
                key_options.sample_steps,
                key_options.super_sampling,
                key_options.overlap_length,
                key_options.min_chunk_length,
            )

        def generate_countdown_pack_token_candidates(
            *,
            token_text: str,
            segment: Segment,
            ref: GPTSoVITSRef,
            ref_style: str,
            lane_index: int,
            tts_text_language: str,
            min_generation_rounds: int = 1,
        ) -> list[dict[str, Any]]:
            if not bool(getattr(cfg, "gsv_countdown_token_bank_pack_warmup_enabled", True)):
                return []
            if _countdown_has_token_specific_carrier_templates(cfg, token_text):
                return []
            key = pack_token_bank_key(
                token_text=token_text,
                ref=ref,
                ref_style=ref_style,
                tts_text_language=tts_text_language,
            )
            with countdown_pack_token_lock:
                cached = countdown_pack_token_cache.get(key)
                generated_rounds = countdown_pack_token_generated_rounds.get(key, 0)
            requested_rounds = max(1, int(min_generation_rounds))
            if cached is not None and generated_rounds >= requested_rounds:
                return [clone_countdown_carrier_candidate(candidate) for candidate in cached]

            token_label = _countdown_chunk_label(token_text)
            ref_fingerprint = hashlib.sha1(
                "|".join([ref_style, ref.ref_audio_path, ref.prompt_text, ref.prompt_lang]).encode("utf-8")
            ).hexdigest()[:12]
            pack_dir = (
                project_dir
                / "work"
                / "tts"
                / "countdown"
                / "pack_token_bank"
                / ref_fingerprint
                / token_label
            )
            pack_dir.mkdir(parents=True, exist_ok=True)
            candidates: list[dict[str, Any]] = (
                [clone_countdown_carrier_candidate(candidate) for candidate in cached]
                if cached is not None
                else []
            )
            carrier_candidate_count = int(getattr(cfg, "gsv_countdown_carrier_candidate_count", 1))
            max_quality_retry_rounds = int(
                getattr(cfg, "gsv_countdown_carrier_quality_retry_max_rounds", 3)
            )
            variant_capacity = max(1, carrier_candidate_count * max_quality_retry_rounds)
            round_pending_candidates: list[dict[str, Any]] = []
            bulk_asr_enabled = bool(getattr(cfg, "gsv_countdown_carrier_bulk_asr_enabled", True))
            for generation_round_index in range(generated_rounds, requested_rounds):
                for pack_index, template, pack_text in countdown_pack_token_templates_for_token(token_text):
                    for local_variant_index in range(carrier_candidate_count):
                        pack_variant_index = (
                            generation_round_index * carrier_candidate_count + local_variant_index
                        )
                        candidate_index = pack_index * variant_capacity + pack_variant_index
                        sequence_seed = sum(
                            (index + 1) * ord(char)
                            for index, char in enumerate(
                                f"pack:{pack_index}:{pack_variant_index}:{token_text}:{pack_text}"
                            )
                        )
                        base_options = countdown_chunk_tts_options(
                            candidate_index=pack_variant_index,
                            sequence_seed=sequence_seed,
                            tts_text_language=tts_text_language,
                        )
                        options, generation_profile = countdown_carrier_generation_profile(
                            base_options,
                            pack_variant_index,
                        )
                        generation_profile = {
                            **generation_profile,
                            "generation_round_index": generation_round_index,
                            "local_variant_index": local_variant_index,
                            "candidate_source": "pack_take",
                        }
                        full_path = (
                            pack_dir
                            / f"pack_{pack_index:02d}_cand_{pack_variant_index:02d}_full_{token_label}.wav"
                        )
                        payload: dict[str, Any] = {
                            "renderer": "countdown_pack_token_bank",
                            "candidate_source": "pack_take",
                            "candidate_index": candidate_index,
                            "carrier_index": -1,
                            "carrier_variant_index": pack_variant_index,
                            "carrier_local_variant_index": local_variant_index,
                            "carrier_generation_round_index": generation_round_index,
                            "carrier_template": template,
                            "carrier_text": pack_text,
                            "carrier_generation_profile": generation_profile,
                            "pack_index": pack_index,
                            "pack_template": template,
                            "pack_text": pack_text,
                            "token_text": token_text,
                            "ref_style": ref_style,
                            "ref_audio_path": ref.ref_audio_path,
                            "lane_index": lane_index,
                            "gsv_url": None if mock else gsv_base_urls[lane_index],
                        }
                        payload.update(_tts_request_debug_payload(pack_text, ref, options))
                        if mock:
                            _mock_synthesize(full_path, 0.72, options.seed, cfg.mix_sample_rate)
                            payload["mock"] = True
                        else:
                            client = clients[lane_index]
                            request = client.build_payload(pack_text, ref, options)
                            payload.update(request.as_payload())
                            client.synthesize_to_file(request, full_path)
                        raw_duration = duration_sec(full_path)
                        postprocess_tts_candidate(full_path, payload)
                        postprocessed_full_duration = duration_sec(full_path)
                        full_sentence_prefilter = run_countdown_carrier_full_sentence_prefilter(
                            segment=segment,
                            candidate_path=full_path,
                            expected_text=token_text,
                            candidate_duration_sec=postprocessed_full_duration,
                        )
                        payload["countdown_carrier_full_sentence_prefilter"] = full_sentence_prefilter
                        if not bool(full_sentence_prefilter.get("approved", True)):
                            continue
                        data, sample_rate = load_audio(full_path)
                        data = _countdown_stereo(data)
                        if sample_rate != cfg.mix_sample_rate:
                            data = resample_linear(data, sample_rate, cfg.mix_sample_rate)
                            sample_rate = cfg.mix_sample_rate
                        boundaries = countdown_phrase_slice_boundaries(data, 3, sample_rate)
                        if len(boundaries) >= 3:
                            slice_start, slice_end = boundaries[1]
                        else:
                            slice_start, slice_end = 0, len(data)
                        boundary_qc = countdown_slice_boundary_qc(
                            data,
                            start_frame=slice_start,
                            end_frame=slice_end,
                            sample_rate=sample_rate,
                            token_text=token_text,
                        )
                        original_slice_start = slice_start
                        original_slice_end = slice_end
                        slice_start = int(boundary_qc["start_frame"])
                        slice_end = int(boundary_qc["end_frame"])
                        slice_audio = _countdown_stereo(data[slice_start:slice_end])
                        slice_path = (
                            pack_dir
                            / f"pack_{pack_index:02d}_cand_{pack_variant_index:02d}_middle_{token_label}.wav"
                        )
                        write_audio(slice_path, slice_audio, sample_rate)
                        slice_duration = duration_sec(slice_path)
                        active_anchor_fit = _countdown_active_anchor_fit_from_audio(
                            slice_audio,
                            sample_rate,
                        )
                        if bulk_asr_enabled:
                            pronunciation_qc = {
                                "gate": "pending",
                                "coverage": 0.0,
                                "backend": pronunciation_qc_backend_name(),
                                "issues": ["pronunciation_qc_pending_bulk_asr"],
                                "short_slice": True,
                                "bulk_asr": True,
                            }
                        else:
                            pronunciation_qc = run_pronunciation_qc(
                                segment=segment,
                                candidate_path=slice_path,
                                expected_text=token_text,
                                candidate_duration_sec=slice_duration,
                                short_slice=True,
                            )
                        pronunciation_gate = str(
                            (pronunciation_qc or {}).get("gate") or "not_run"
                        ).strip().lower()
                        stability_qc = countdown_slice_stability_metrics(slice_audio, sample_rate)
                        audio_metrics = {
                            "gate": "pass",
                            "peak_dbfs": round(peak_dbfs(slice_path), 3),
                            "rms_dbfs": round(rms_dbfs(slice_path), 3),
                        }
                        slice_payload = copy.deepcopy(payload)
                        slice_payload.update(
                            {
                                "candidate_index": candidate_index,
                                "carrier_candidate_index": candidate_index,
                                "slice_window_index": 0,
                                "countdown_carrier_bank": {
                                    "full_path": str(full_path),
                                    "raw_duration_sec": round(raw_duration, 6),
                                    "postprocessed_duration_sec": round(len(data) / sample_rate, 6),
                                    "full_sentence_prefilter": full_sentence_prefilter,
                                    "slice_path": str(slice_path),
                                    "slice_search_enabled": False,
                                    "slice_window_count": 1,
                                    "slice_window": {
                                        "index": 0,
                                        "kind": "pack_middle_slice",
                                        "window_sec": None,
                                        "offset_sec": 0.0,
                                    },
                                    "base_slice": {
                                        "strategy": "pack_middle_slice",
                                        "anchor": "middle_token",
                                        "start_sec": round(
                                            original_slice_start / sample_rate,
                                            6,
                                        ),
                                        "end_sec": round(original_slice_end / sample_rate, 6),
                                    },
                                    "base_slice_start_sec": round(
                                        original_slice_start / sample_rate,
                                        6,
                                    ),
                                    "base_slice_end_sec": round(
                                        original_slice_end / sample_rate,
                                        6,
                                    ),
                                    "original_slice_start_sec": round(
                                        original_slice_start / sample_rate,
                                        6,
                                    ),
                                    "original_slice_end_sec": round(
                                        original_slice_end / sample_rate,
                                        6,
                                    ),
                                    "slice_start_sec": round(slice_start / sample_rate, 6),
                                    "slice_end_sec": round(slice_end / sample_rate, 6),
                                    "slice_duration_sec": round(slice_duration, 6),
                                    "active_anchor_fit": active_anchor_fit,
                                    "boundary_qc": boundary_qc,
                                },
                                "countdown_slice_boundary_qc": boundary_qc,
                                "pronunciation_qc": pronunciation_qc,
                                "stability_qc": stability_qc,
                                "audio_qc": audio_metrics,
                            }
                        )
                        candidates.append(
                            {
                                "candidate_index": candidate_index,
                                "carrier_candidate_index": candidate_index,
                                "carrier_index": -1,
                                "carrier_variant_index": pack_variant_index,
                                "carrier_local_variant_index": local_variant_index,
                                "carrier_generation_round_index": generation_round_index,
                                "slice_window_index": 0,
                                "candidate_source": "pack_take",
                                "carrier_template": template,
                                "carrier_text": pack_text,
                                "carrier_generation_profile": generation_profile,
                                "ref_style": ref_style,
                                "ref_audio_path": ref.ref_audio_path,
                                "text": token_text,
                                "output_path": str(slice_path),
                                "duration_sec": slice_duration,
                                "pronunciation_gate": pronunciation_gate,
                                "payload": slice_payload,
                                "audio": slice_audio,
                            }
                        )
                        if bulk_asr_enabled:
                            round_pending_candidates.append(candidates[-1])
            if bulk_asr_enabled and round_pending_candidates:
                qc_results = run_pronunciation_qc_batch(
                    [
                        {
                            "segment": segment,
                            "candidate_path": Path(str(candidate["output_path"])),
                            "expected_text": token_text,
                            "candidate_duration_sec": float(candidate["duration_sec"]),
                        }
                        for candidate in round_pending_candidates
                    ],
                    short_slice=True,
                )
                for candidate, pronunciation_qc in zip(
                    round_pending_candidates,
                    qc_results,
                    strict=True,
                ):
                    pronunciation_gate = str(
                        (pronunciation_qc or {}).get("gate") or "not_run"
                    ).strip().lower()
                    candidate["pronunciation_gate"] = pronunciation_gate
                    candidate["payload"]["pronunciation_qc"] = pronunciation_qc
            with countdown_pack_token_lock:
                stored = [clone_countdown_carrier_candidate(candidate) for candidate in candidates]
                countdown_pack_token_cache[key] = stored
                countdown_pack_token_generated_rounds[key] = max(
                    countdown_pack_token_generated_rounds.get(key, 0),
                    requested_rounds,
                )
                cached = countdown_pack_token_cache[key]
            return [clone_countdown_carrier_candidate(candidate) for candidate in cached]

        def generate_countdown_carrier_candidates(
            *,
            token_text: str,
            segment: Segment,
            ref: GPTSoVITSRef,
            ref_style: str,
            lane_index: int,
            tts_text_language: str,
            min_generation_rounds: int = 1,
        ) -> list[dict[str, Any]]:
            key = carrier_bank_key(
                token_text=token_text,
                ref=ref,
                ref_style=ref_style,
                tts_text_language=tts_text_language,
            )
            with countdown_carrier_lock:
                cached = countdown_carrier_cache.get(key)
                generated_rounds = countdown_carrier_generated_rounds.get(key, 0)
            requested_rounds = max(1, int(min_generation_rounds))
            if cached is not None and generated_rounds >= requested_rounds:
                return [clone_countdown_carrier_candidate(candidate) for candidate in cached]

            token_label = _countdown_chunk_label(token_text)
            token_template_map = getattr(cfg, "gsv_countdown_carrier_token_templates", {}) or {}
            token_template_set = (
                {
                    str(template).strip()
                    for template in token_template_map.get(token_text, [])
                    if str(template).strip()
                }
                if isinstance(token_template_map, dict)
                else set()
            )
            ref_fingerprint = hashlib.sha1(
                "|".join([ref_style, ref.ref_audio_path, ref.prompt_text, ref.prompt_lang]).encode("utf-8")
            ).hexdigest()[:12]
            carrier_dir = (
                project_dir
                / "work"
                / "tts"
                / "countdown"
                / "carrier_bank"
                / ref_fingerprint
                / token_label
            )
            carrier_dir.mkdir(parents=True, exist_ok=True)
            candidates: list[dict[str, Any]] = (
                [clone_countdown_carrier_candidate(candidate) for candidate in cached]
                if cached is not None
                else []
            )
            carrier_candidate_count = int(getattr(cfg, "gsv_countdown_carrier_candidate_count", 1))
            max_slice_windows_per_candidate = int(
                getattr(cfg, "gsv_countdown_carrier_max_slice_windows_per_candidate", 5)
            )
            max_quality_retry_rounds = int(
                getattr(cfg, "gsv_countdown_carrier_quality_retry_max_rounds", 3)
            )
            variant_capacity = max(1, carrier_candidate_count * max_quality_retry_rounds)
            stop_window_search_after_pass = bool(
                getattr(
                    cfg,
                    "gsv_countdown_carrier_stop_window_search_after_pronunciation_pass",
                    True,
                )
            )
            target_pronunciation_passes = int(
                getattr(cfg, "gsv_countdown_carrier_target_pronunciation_passes", 2)
            )
            template_entries = countdown_carrier_templates_for_token(token_text)
            has_numeric_unit_template = any(
                countdown_carrier_is_numeric_unit(
                    template=template,
                    carrier_text=carrier_text,
                    token_text=token_text,
                )
                for _, template, carrier_text in template_entries
            )
            for generation_round_index in range(generated_rounds, requested_rounds):
                pronunciation_pass_count = 0
                bulk_asr_enabled = bool(
                    getattr(cfg, "gsv_countdown_carrier_bulk_asr_enabled", True)
                )
                round_pending_candidates: list[dict[str, Any]] = []

                def flush_pending_bulk_asr(
                    pending_candidates: list[dict[str, Any]],
                    *,
                    enabled: bool,
                ) -> None:
                    nonlocal pronunciation_pass_count
                    if not enabled or not pending_candidates:
                        return
                    qc_results = run_pronunciation_qc_batch(
                        [
                            {
                                "segment": segment,
                                "candidate_path": Path(str(candidate["output_path"])),
                                "expected_text": token_text,
                                "candidate_duration_sec": float(candidate["duration_sec"]),
                            }
                            for candidate in pending_candidates
                        ],
                        short_slice=True,
                    )
                    for candidate, pronunciation_qc in zip(
                        pending_candidates,
                        qc_results,
                        strict=True,
                    ):
                        pronunciation_gate = str(
                            (pronunciation_qc or {}).get("gate") or "not_run"
                        ).strip().lower()
                        candidate["pronunciation_gate"] = pronunciation_gate
                        candidate["payload"]["pronunciation_qc"] = pronunciation_qc
                        token_payload = candidate["payload"].get("countdown_carrier_token")
                        if isinstance(token_payload, dict):
                            token_payload["pronunciation_gate"] = pronunciation_gate
                        boundary_clean = bool(
                            candidate["payload"]
                            .get("countdown_carrier_bank", {})
                            .get("boundary_clean", True)
                        )
                        carrier_candidate_source = str(
                            candidate.get("candidate_source")
                            or candidate.get("payload", {}).get("candidate_source")
                            or ""
                        )
                        pronunciation_window_pass = pronunciation_gate == "pass" and (
                            carrier_candidate_source != "numeric_unit_carrier" or boundary_clean
                        )
                        if pronunciation_window_pass:
                            pronunciation_pass_count += 1
                    pending_candidates.clear()

                retry_numeric_unit_only = generation_round_index > 0 and has_numeric_unit_template
                for carrier_index, template, carrier_text in template_entries:
                    carrier_is_token_specific = str(template).strip() in token_template_set
                    carrier_is_numeric_unit = countdown_carrier_is_numeric_unit(
                        template=template,
                        carrier_text=carrier_text,
                        token_text=token_text,
                    )
                    if retry_numeric_unit_only and not (
                        carrier_is_numeric_unit or carrier_is_token_specific
                    ):
                        continue
                    carrier_candidate_source = (
                        "token_specific_carrier"
                        if carrier_is_token_specific
                        else "numeric_unit_carrier"
                        if carrier_is_numeric_unit
                        else "sentence_carrier"
                    )
                    carrier_family = (
                        "token_specific_numeric_unit"
                        if carrier_is_token_specific and carrier_is_numeric_unit
                        else "token_specific"
                        if carrier_is_token_specific
                        else "numeric_unit"
                        if carrier_is_numeric_unit
                        else "sentence"
                    )
                    for local_variant_index in range(carrier_candidate_count):
                        carrier_variant_index = (
                            generation_round_index * carrier_candidate_count + local_variant_index
                        )
                        candidate_index = carrier_index * variant_capacity + carrier_variant_index
                        sequence_seed = sum(
                            (index + 1) * ord(char)
                            for index, char in enumerate(
                                f"{carrier_index}:{carrier_variant_index}:{token_text}:{carrier_text}"
                            )
                        )
                        base_options = countdown_chunk_tts_options(
                            candidate_index=carrier_variant_index,
                            sequence_seed=sequence_seed,
                            tts_text_language=tts_text_language,
                        )
                        options, generation_profile = countdown_carrier_generation_profile(
                            base_options,
                            carrier_variant_index,
                        )
                        generation_profile = {
                            **generation_profile,
                            "generation_round_index": generation_round_index,
                            "local_variant_index": local_variant_index,
                            "carrier_family": carrier_family,
                        }
                        full_path = (
                            carrier_dir
                            / f"carrier_{carrier_index:02d}_cand_{carrier_variant_index:02d}_full_{token_label}.wav"
                        )
                        payload: dict[str, Any] = {
                            "renderer": "countdown_carrier_bank",
                            "candidate_source": carrier_candidate_source,
                            "candidate_index": candidate_index,
                            "carrier_index": carrier_index,
                            "carrier_variant_index": carrier_variant_index,
                            "carrier_local_variant_index": local_variant_index,
                            "carrier_generation_round_index": generation_round_index,
                            "carrier_template": template,
                            "carrier_text": carrier_text,
                            "carrier_family": carrier_family,
                            "carrier_generation_profile": generation_profile,
                            "token_text": token_text,
                            "ref_style": ref_style,
                            "ref_audio_path": ref.ref_audio_path,
                            "lane_index": lane_index,
                            "gsv_url": None if mock else gsv_base_urls[lane_index],
                        }
                        payload.update(_tts_request_debug_payload(carrier_text, ref, options))
                        if mock:
                            _mock_synthesize(
                                full_path,
                                0.9,
                                options.seed,
                                cfg.mix_sample_rate,
                            )
                            payload["mock"] = True
                        else:
                            client = clients[lane_index]
                            request = client.build_payload(carrier_text, ref, options)
                            payload.update(request.as_payload())
                            client.synthesize_to_file(request, full_path)
                        raw_duration = duration_sec(full_path)
                        postprocess_tts_candidate(full_path, payload)
                        postprocessed_full_duration = duration_sec(full_path)
                        full_sentence_prefilter = run_countdown_carrier_full_sentence_prefilter(
                            segment=segment,
                            candidate_path=full_path,
                            expected_text=token_text,
                            candidate_duration_sec=postprocessed_full_duration,
                        )
                        payload["countdown_carrier_full_sentence_prefilter"] = full_sentence_prefilter
                        if not bool(full_sentence_prefilter.get("approved", True)):
                            continue
                        data, sample_rate = load_audio(full_path)
                        data = _countdown_stereo(data)
                        if sample_rate != cfg.mix_sample_rate:
                            data = resample_linear(data, sample_rate, cfg.mix_sample_rate)
                            sample_rate = cfg.mix_sample_rate
                        base_slice = countdown_carrier_base_slice(
                            phrase_audio=data,
                            token_text=token_text,
                            carrier_template=template,
                            carrier_text=carrier_text,
                            sample_rate=sample_rate,
                            full_sentence_prefilter=full_sentence_prefilter,
                        )
                        base_slice_start = int(base_slice["start_frame"])
                        base_slice_end = int(base_slice["end_frame"])
                        slice_windows = countdown_carrier_slice_windows(
                            total_frames=len(data),
                            base_start=base_slice_start,
                            base_end=base_slice_end,
                            sample_rate=sample_rate,
                            base_kind=str(base_slice["strategy"]),
                            base_anchor=str(base_slice.get("anchor") or ""),
                        )
                        for slice_window in slice_windows:
                            slice_window_index = int(slice_window["slice_window_index"])
                            slice_start = int(slice_window["start_frame"])
                            slice_end = int(slice_window["end_frame"])
                            boundary_qc = countdown_slice_boundary_qc(
                                data,
                                start_frame=slice_start,
                                end_frame=slice_end,
                                sample_rate=sample_rate,
                                token_text=token_text,
                            )
                            original_slice_start = slice_start
                            original_slice_end = slice_end
                            slice_start = int(boundary_qc["start_frame"])
                            slice_end = int(boundary_qc["end_frame"])
                            slice_audio = _countdown_stereo(data[slice_start:slice_end])
                            window_candidate_index = (
                                candidate_index * max_slice_windows_per_candidate + slice_window_index
                            )
                            slice_path = (
                                carrier_dir
                                / f"carrier_{carrier_index:02d}_cand_{carrier_variant_index:02d}_"
                                f"slice_window_{slice_window_index:02d}_{token_label}.wav"
                            )
                            write_audio(slice_path, slice_audio, sample_rate)
                            slice_duration = duration_sec(slice_path)
                            active_anchor_fit = _countdown_active_anchor_fit_from_audio(
                                slice_audio,
                                sample_rate,
                            )
                            if bulk_asr_enabled:
                                pronunciation_qc = {
                                    "gate": "pending",
                                    "coverage": 0.0,
                                    "backend": pronunciation_qc_backend_name(),
                                    "issues": ["pronunciation_qc_pending_bulk_asr"],
                                    "short_slice": True,
                                    "bulk_asr": True,
                                }
                            else:
                                pronunciation_qc = run_pronunciation_qc(
                                    segment=segment,
                                    candidate_path=slice_path,
                                    expected_text=token_text,
                                    candidate_duration_sec=slice_duration,
                                    short_slice=True,
                                )
                            pronunciation_gate = str(
                                (pronunciation_qc or {}).get("gate") or "not_run"
                            ).strip().lower()
                            boundary_clean = _countdown_slice_boundary_is_clean(boundary_qc)
                            stability_qc = countdown_slice_stability_metrics(slice_audio, sample_rate)
                            audio_metrics = {
                                "gate": "pass",
                                "peak_dbfs": round(peak_dbfs(slice_path), 3),
                                "rms_dbfs": round(rms_dbfs(slice_path), 3),
                            }
                            slice_payload = copy.deepcopy(payload)
                            slice_payload.update(
                                {
                                    "candidate_index": window_candidate_index,
                                    "carrier_candidate_index": candidate_index,
                                    "slice_window_index": slice_window_index,
                                    "countdown_carrier_bank": {
                                        "full_path": str(full_path),
                                        "raw_duration_sec": round(raw_duration, 6),
                                        "postprocessed_duration_sec": round(len(data) / sample_rate, 6),
                                        "full_sentence_prefilter": full_sentence_prefilter,
                                        "slice_path": str(slice_path),
                                        "slice_search_enabled": bool(
                                            getattr(
                                                cfg,
                                                "gsv_countdown_carrier_slice_search_enabled",
                                                True,
                                            )
                                        ),
                                        "slice_window_count": len(slice_windows),
                                        "slice_window": {
                                            "index": slice_window_index,
                                            "kind": slice_window["kind"],
                                            "window_sec": slice_window["window_sec"],
                                            "offset_sec": slice_window["offset_sec"],
                                        },
                                        "base_slice": {
                                            key: value
                                            for key, value in base_slice.items()
                                            if not str(key).endswith("_frame")
                                        },
                                        "base_slice_start_sec": round(base_slice_start / sample_rate, 6),
                                        "base_slice_end_sec": round(base_slice_end / sample_rate, 6),
                                        "original_slice_start_sec": round(
                                            original_slice_start / sample_rate,
                                            6,
                                        ),
                                        "original_slice_end_sec": round(
                                            original_slice_end / sample_rate,
                                            6,
                                        ),
                                        "slice_start_sec": round(slice_start / sample_rate, 6),
                                        "slice_end_sec": round(slice_end / sample_rate, 6),
                                        "slice_duration_sec": round(slice_duration, 6),
                                        "active_anchor_fit": active_anchor_fit,
                                        "boundary_qc": boundary_qc,
                                        "boundary_clean": boundary_clean,
                                    },
                                    "countdown_slice_boundary_qc": boundary_qc,
                                    "pronunciation_qc": pronunciation_qc,
                                    "stability_qc": stability_qc,
                                    "audio_qc": audio_metrics,
                                }
                            )
                            candidates.append(
                                {
                                    "candidate_index": window_candidate_index,
                                    "carrier_candidate_index": candidate_index,
                                    "carrier_index": carrier_index,
                                    "carrier_variant_index": carrier_variant_index,
                                    "carrier_local_variant_index": local_variant_index,
                                    "carrier_generation_round_index": generation_round_index,
                                    "slice_window_index": slice_window_index,
                                    "carrier_template": template,
                                    "carrier_text": carrier_text,
                                    "carrier_generation_profile": generation_profile,
                                    "candidate_source": carrier_candidate_source,
                                    "ref_style": ref_style,
                                    "ref_audio_path": ref.ref_audio_path,
                                    "text": token_text,
                                    "output_path": str(slice_path),
                                    "duration_sec": slice_duration,
                                    "pronunciation_gate": pronunciation_gate,
                                    "payload": slice_payload,
                                    "audio": slice_audio,
                                }
                            )
                            if bulk_asr_enabled:
                                round_pending_candidates.append(candidates[-1])
                                if stop_window_search_after_pass:
                                    flush_pending_bulk_asr(
                                        round_pending_candidates,
                                        enabled=True,
                                    )
                                    pronunciation_gate = str(
                                        candidates[-1].get("pronunciation_gate") or "not_run"
                                    ).strip().lower()
                                    pronunciation_window_pass = pronunciation_gate == "pass" and (
                                        carrier_candidate_source != "numeric_unit_carrier"
                                        or boundary_clean
                                    )
                                    if pronunciation_window_pass:
                                        if target_pronunciation_passes > 0:
                                            pronunciation_pass_count = max(
                                                pronunciation_pass_count,
                                                target_pronunciation_passes,
                                            )
                                        break
                            else:
                                pronunciation_window_pass = pronunciation_gate == "pass" and (
                                    carrier_candidate_source != "numeric_unit_carrier"
                                    or boundary_clean
                                )
                                if pronunciation_window_pass:
                                    pronunciation_pass_count += 1
                                    if stop_window_search_after_pass:
                                        if target_pronunciation_passes > 0:
                                            pronunciation_pass_count = max(
                                                pronunciation_pass_count,
                                                target_pronunciation_passes,
                                            )
                                        break
                        if (
                            target_pronunciation_passes > 0
                            and pronunciation_pass_count >= target_pronunciation_passes
                        ):
                            break
                    flush_pending_bulk_asr(
                        round_pending_candidates,
                        enabled=bulk_asr_enabled,
                    )
                    if (
                        target_pronunciation_passes > 0
                        and pronunciation_pass_count >= target_pronunciation_passes
                    ):
                        break
                flush_pending_bulk_asr(
                    round_pending_candidates,
                    enabled=bulk_asr_enabled,
                )
                if (
                    target_pronunciation_passes > 0
                    and pronunciation_pass_count >= target_pronunciation_passes
                ):
                    break
            with countdown_carrier_lock:
                stored = [clone_countdown_carrier_candidate(candidate) for candidate in candidates]
                countdown_carrier_cache[key] = stored
                countdown_carrier_generated_rounds[key] = max(
                    countdown_carrier_generated_rounds.get(key, 0),
                    requested_rounds,
                )
                cached = countdown_carrier_cache[key]
            return [clone_countdown_carrier_candidate(candidate) for candidate in cached]

        def countdown_carrier_quality_tier_rank(tier: str | None) -> int:
            return {"A": 3, "B": 2, "C": 1}.get(str(tier or "").strip().upper(), 0)

        def countdown_candidate_source_rank(candidate: dict[str, Any]) -> int:
            source = str(
                candidate.get("candidate_source")
                or candidate.get("payload", {}).get("candidate_source")
                or ""
            ).strip()
            return {
                "token_specific_carrier": 4,
                "numeric_unit_carrier": 3,
                "pack_take": 2,
                "sentence_carrier": 1,
            }.get(source, 0)

        def countdown_carrier_candidate_quality_tier(candidate: dict[str, Any]) -> str:
            payload = candidate.get("payload", {})
            token_payload = payload.get("countdown_carrier_token", {})
            duration_gate = str(token_payload.get("duration_gate") or "").strip().lower()
            pronunciation_gate = str(candidate.get("pronunciation_gate") or "").strip().lower()
            pronunciation_ok = bool(token_payload.get("pronunciation_contract_ok"))
            strict_pronunciation = bool(token_payload.get("strict_pronunciation_required"))
            audio_gate = str(payload.get("audio_qc", {}).get("gate") or "").strip().lower()
            prosody_qc = payload.get("countdown_prosody_qc")
            prosody_gate = (
                str(prosody_qc.get("gate") or "").strip().lower()
                if isinstance(prosody_qc, dict)
                else "unavailable"
            )
            stability_score = float(
                payload.get("stability_qc", {}).get("stability_score", 0.0)
            )
            boundary_qc = payload.get("countdown_slice_boundary_qc")
            if not isinstance(boundary_qc, dict):
                boundary_qc = payload.get("countdown_carrier_bank", {}).get("boundary_qc", {})
            boundary_gate = (
                str(boundary_qc.get("gate") or "pass").strip().lower()
                if isinstance(boundary_qc, dict)
                else "pass"
            )
            boundary_clean = _countdown_slice_boundary_is_clean(boundary_qc)
            if duration_gate != "pass" or audio_gate != "pass" or not pronunciation_ok:
                return "reject"
            if strict_pronunciation and pronunciation_gate != "pass":
                return "reject"
            if prosody_gate == "fail":
                return "reject"
            if boundary_gate == "fail":
                return "reject"
            anchor_fit_gate = payload.get("countdown_carrier_token", {}).get(
                "active_anchor_fit_gate",
                {},
            )
            if (
                isinstance(anchor_fit_gate, dict)
                and str(anchor_fit_gate.get("gate") or "").strip().lower() == "fail"
            ):
                return "reject"
            requires_clean_boundary = (
                str(
                    candidate.get("candidate_source")
                    or payload.get("candidate_source")
                    or ""
                ).strip()
                == "numeric_unit_carrier"
            )
            if (
                pronunciation_gate == "pass"
                and prosody_gate in {"", "pass", "unavailable", "disabled", "skipped"}
                and (boundary_clean or not requires_clean_boundary)
                and stability_score >= 0.5
            ):
                return "A"
            if bool(candidate.get("acceptable")):
                return "B"
            return "C"

        def countdown_token_bank_metadata() -> dict[str, Any]:
            return {
                "enabled": bool(getattr(cfg, "gsv_countdown_token_bank_enabled", True)),
                "warmup_enabled": bool(
                    getattr(cfg, "gsv_countdown_token_bank_warmup_enabled", True)
                ),
                "pack_warmup_enabled": bool(
                    getattr(cfg, "gsv_countdown_token_bank_pack_warmup_enabled", True)
                ),
                "pack_template_count": len(countdown_pack_token_templates_for_token("삼")),
                "max_ref_count": int(getattr(cfg, "gsv_countdown_token_bank_max_ref_count", 4)),
                "beam_width": int(getattr(cfg, "gsv_countdown_token_bank_beam_width", 8)),
            }

        def countdown_token_bank_store(token_text: str, candidates: list[dict[str, Any]]) -> None:
            if not candidates:
                return
            with countdown_token_bank_lock:
                existing = countdown_token_bank_cache.setdefault(token_text, [])
                seen = {str(candidate.get("output_path")) for candidate in existing}
                for candidate in candidates:
                    output_path = str(candidate.get("output_path"))
                    if output_path in seen:
                        continue
                    seen.add(output_path)
                    existing.append(clone_countdown_carrier_candidate(candidate))

        def countdown_token_bank_candidates(token_text: str) -> list[dict[str, Any]]:
            with countdown_token_bank_lock:
                candidates = countdown_token_bank_cache.get(token_text, [])
                return [clone_countdown_carrier_candidate(candidate) for candidate in candidates]

        def countdown_token_bank_ref_pool(
            *,
            span: list[tuple[int, Segment, list[int]]],
            primary_ref_style: str,
            primary_ref: GPTSoVITSRef,
            tts_text_language: str,
        ) -> list[dict[str, Any]]:
            max_ref_count = int(getattr(cfg, "gsv_countdown_token_bank_max_ref_count", 4))
            pool: list[dict[str, Any]] = []
            seen: set[tuple[str, str, str]] = set()

            def add_ref(
                ref_style: str,
                ref: GPTSoVITSRef,
                source: str,
                *,
                require_existing_audio: bool = True,
            ) -> None:
                tts_ref = _ref_for_tts_language(ref, tts_text_language)
                if require_existing_audio and not Path(tts_ref.ref_audio_path).exists():
                    return
                key = (tts_ref.ref_audio_path, tts_ref.prompt_text, tts_ref.prompt_lang)
                if key in seen or len(pool) >= max_ref_count:
                    return
                seen.add(key)
                pool.append({"ref_style": ref_style, "ref": tts_ref, "source": source})

            if not bool(getattr(cfg, "gsv_countdown_token_bank_enabled", True)):
                add_ref(
                    primary_ref_style,
                    primary_ref,
                    "span_primary",
                    require_existing_audio=False,
                )
                return pool
            ordered_styles = [primary_ref_style] + [
                style for style in sorted(refs) if style != primary_ref_style
            ]
            for style in ordered_styles:
                if style in refs:
                    add_ref(style, refs[style], "refs_json")
            if not pool:
                add_ref(
                    primary_ref_style,
                    primary_ref,
                    "span_primary",
                    require_existing_audio=False,
                )
            return pool

        def countdown_token_bank_annotate_and_store(
            token_text: str,
            ref_item: dict[str, Any],
            candidates: list[dict[str, Any]],
        ) -> None:
            for candidate in candidates:
                candidate["token_bank_ref_source"] = ref_item["source"]
                candidate["payload"]["countdown_token_bank"] = {
                    "ref_source": ref_item["source"],
                    "ref_style": ref_item["ref_style"],
                    "ref_audio_path": ref_item["ref"].ref_audio_path,
                }
            countdown_token_bank_store(token_text, candidates)

        def countdown_token_bank_warmup_for_span(
            *,
            span_id: str,
            tokens: list[str],
            segment: Segment,
            ref_pool: list[dict[str, Any]],
            lane_index: int,
            tts_text_language: str,
        ) -> dict[str, Any]:
            metadata = countdown_token_bank_metadata()
            if not metadata["enabled"] or not metadata["warmup_enabled"]:
                return {**metadata, "status": "disabled", "ref_count": 0, "token_count": len(set(tokens))}
            ordered_tokens = list(dict.fromkeys(tokens))
            warmup_key = (
                tuple(ordered_tokens),
                tuple(
                    (
                        item["ref_style"],
                        item["ref"].ref_audio_path,
                        item["ref"].prompt_text,
                        item["ref"].prompt_lang,
                    )
                    for item in ref_pool
                ),
                tts_text_language,
            )
            with countdown_token_bank_lock:
                already_done = warmup_key in countdown_token_bank_warmup_keys
            if already_done:
                existing_count = sum(len(countdown_token_bank_candidates(token)) for token in ordered_tokens)
                return {
                    **metadata,
                    "status": "cached",
                    "ref_count": len(ref_pool),
                    "token_count": len(ordered_tokens),
                    "candidate_count": existing_count,
                }
            generated_count = 0
            pack_generated_count = 0
            carrier_generated_count = 0
            for token_text in ordered_tokens:
                for ref_item in ref_pool:
                    pack_candidates = generate_countdown_pack_token_candidates(
                        token_text=token_text,
                        segment=segment,
                        ref=ref_item["ref"],
                        ref_style=str(ref_item["ref_style"]),
                        lane_index=lane_index,
                        tts_text_language=tts_text_language,
                        min_generation_rounds=1,
                    )
                    generated_count += len(pack_candidates)
                    pack_generated_count += len(pack_candidates)
                    countdown_token_bank_annotate_and_store(token_text, ref_item, pack_candidates)
                    candidates = generate_countdown_carrier_candidates(
                        token_text=token_text,
                        segment=segment,
                        ref=ref_item["ref"],
                        ref_style=str(ref_item["ref_style"]),
                        lane_index=lane_index,
                        tts_text_language=tts_text_language,
                        min_generation_rounds=1,
                    )
                    generated_count += len(candidates)
                    carrier_generated_count += len(candidates)
                    countdown_token_bank_annotate_and_store(token_text, ref_item, candidates)
            with countdown_token_bank_lock:
                countdown_token_bank_warmup_keys.add(warmup_key)
            _log_stage_checkpoint(
                stage_name,
                "countdown token-bank warmup",
                (
                    f"span={span_id} tokens={len(ordered_tokens)} refs={len(ref_pool)} "
                    f"candidates={generated_count}"
                ),
            )
            return {
                **metadata,
                "status": "generated",
                "ref_count": len(ref_pool),
                "token_count": len(ordered_tokens),
                "candidate_count": generated_count,
                "pack_candidate_count": pack_generated_count,
                "carrier_candidate_count": carrier_generated_count,
            }

        def countdown_candidate_anchor_fit_gate(
            candidate: dict[str, Any],
            token_text: str,
        ) -> dict[str, Any]:
            score = _countdown_candidate_anchor_fit_score(candidate)
            fit = _countdown_candidate_anchor_fit(candidate)
            if score is None or fit is None:
                return {
                    "gate": "pass",
                    "score": None,
                    "reason": "active_anchor_fit_unavailable",
                }
            leading_silence_sec = max(0.0, _safe_float(fit.get("leading_silence_sec"), 0.0))
            trailing_silence_sec = max(0.0, _safe_float(fit.get("trailing_silence_sec"), 0.0))
            next_anchor_overflow_sec = max(
                0.0,
                _safe_float(fit.get("next_anchor_overflow_sec"), 0.0),
            )
            active_duration_sec = max(0.0, _safe_float(fit.get("active_duration_sec"), 0.0))
            strict_token = normalize_korean_pronunciation_text(token_text) == "사"
            min_score = 0.68 if strict_token else 0.54
            max_leading_sec = 0.15 if strict_token else 0.20
            max_overflow_sec = 0.06 if strict_token else 0.12
            max_trailing_sec = 0.34
            issues: list[str] = []
            if score < min_score:
                issues.append("low_active_anchor_fit_score")
            if leading_silence_sec > max_leading_sec:
                issues.append("excessive_pre_anchor_silence")
            if trailing_silence_sec > max_trailing_sec:
                issues.append("excessive_post_anchor_silence")
            if next_anchor_overflow_sec > max_overflow_sec:
                issues.append("active_audio_overflows_next_anchor")
            if active_duration_sec <= 0.035:
                issues.append("active_speech_not_detected")
            return {
                "gate": "fail" if issues else "pass",
                "score": score,
                "issues": issues,
                "strict_token": strict_token,
                "min_score": min_score,
                "max_leading_sec": max_leading_sec,
                "max_trailing_sec": max_trailing_sec,
                "max_overflow_sec": max_overflow_sec,
            }

        def countdown_carrier_candidates_for_slot(
            *,
            token_text: str,
            token_slot_sec: float,
            segment: Segment,
            ref: GPTSoVITSRef,
            ref_style: str,
            lane_index: int,
            tts_text_language: str,
            token_bank_ref_pool: list[dict[str, Any]] | None = None,
            source_prosody: dict[str, Any] | None = None,
        ) -> list[dict[str, Any]]:
            retry_metadata = countdown_carrier_quality_retry_metadata()
            max_rounds = (
                int(retry_metadata["max_rounds"])
                if bool(retry_metadata["enabled"])
                else 1
            )
            target_rank = countdown_carrier_quality_tier_rank(str(retry_metadata["target_tier"]))
            requested_rounds = 1
            target_met = False
            use_token_bank = bool(getattr(cfg, "gsv_countdown_token_bank_enabled", True))
            candidates: list[dict[str, Any]] = (
                countdown_token_bank_candidates(token_text) if use_token_bank else []
            )

            def generate_token_bank_round(round_count: int) -> list[dict[str, Any]]:
                ref_items = token_bank_ref_pool or [
                    {"ref": ref, "ref_style": ref_style, "source": "span_primary"}
                ]
                for ref_item in ref_items:
                    pack_generated = generate_countdown_pack_token_candidates(
                        token_text=token_text,
                        segment=segment,
                        ref=ref_item["ref"],
                        ref_style=str(ref_item["ref_style"]),
                        lane_index=lane_index,
                        tts_text_language=tts_text_language,
                        min_generation_rounds=round_count,
                    )
                    countdown_token_bank_annotate_and_store(token_text, ref_item, pack_generated)
                    generated = generate_countdown_carrier_candidates(
                        token_text=token_text,
                        segment=segment,
                        ref=ref_item["ref"],
                        ref_style=str(ref_item["ref_style"]),
                        lane_index=lane_index,
                        tts_text_language=tts_text_language,
                        min_generation_rounds=round_count,
                    )
                    countdown_token_bank_annotate_and_store(token_text, ref_item, generated)
                return countdown_token_bank_candidates(token_text)

            while True:
                if not candidates:
                    if use_token_bank:
                        candidates = generate_token_bank_round(requested_rounds)
                    else:
                        candidates = generate_countdown_carrier_candidates(
                            token_text=token_text,
                            segment=segment,
                            ref=ref,
                            ref_style=ref_style,
                            lane_index=lane_index,
                            tts_text_language=tts_text_language,
                            min_generation_rounds=requested_rounds,
                        )
                for candidate in candidates:
                    duration_gate, token_min_sec, token_max_sec = countdown_token_duration_gate(
                        duration=float(candidate["duration_sec"]),
                        token_slot_sec=token_slot_sec,
                        token_text=token_text,
                        candidate_source=str(
                            candidate.get("candidate_source")
                            or candidate.get("payload", {}).get("candidate_source")
                            or ""
                        ),
                    )
                    pronunciation_ok = countdown_pronunciation_contract_ok(
                        segment,
                        candidate["payload"].get("pronunciation_qc"),
                    )
                    payload = candidate["payload"]
                    carrier_bank_payload = payload.get("countdown_carrier_bank")
                    if isinstance(carrier_bank_payload, dict):
                        carrier_bank_payload["active_anchor_fit"] = (
                            _countdown_active_anchor_fit_from_audio(
                                _countdown_stereo(np.asarray(candidate["audio"], dtype=np.float32)),
                                cfg.mix_sample_rate,
                                token_slot_sec=token_slot_sec,
                            )
                        )
                    attach_countdown_prosody_qc(
                        payload,
                        _countdown_stereo(np.asarray(candidate["audio"], dtype=np.float32)),
                        cfg.mix_sample_rate,
                        source_prosody,
                    )
                    active_anchor_fit_score = _countdown_candidate_anchor_fit_score(candidate)
                    active_anchor_fit_gate = countdown_candidate_anchor_fit_gate(
                        candidate,
                        token_text,
                    )
                    transcript_preference_score = _countdown_candidate_transcript_preference_score(
                        token_text,
                        candidate,
                    )
                    payload["countdown_carrier_token"] = {
                        "duration_sec": round(float(candidate["duration_sec"]), 6),
                        "duration_gate": duration_gate,
                        "token_min_sec": round(token_min_sec, 6),
                        "token_max_sec": round(token_max_sec, 6),
                        "pronunciation_gate": candidate["pronunciation_gate"],
                        "slice_window_index": candidate.get("slice_window_index"),
                        "strict_pronunciation_required": countdown_strict_pronunciation_required(segment),
                        "pronunciation_contract_ok": pronunciation_ok,
                        "active_anchor_fit_score": active_anchor_fit_score,
                        "active_anchor_fit_gate": active_anchor_fit_gate,
                        "transcript_preference_score": transcript_preference_score,
                    }
                    boundary_qc = payload.get("countdown_slice_boundary_qc")
                    boundary_gate = (
                        str(boundary_qc.get("gate") or "pass").strip().lower()
                        if isinstance(boundary_qc, dict)
                        else "pass"
                    )
                    candidate["duration_gate"] = duration_gate
                    candidate["acceptable"] = (
                        duration_gate == "pass"
                        and pronunciation_ok
                        and countdown_prosody_contract_ok(candidate)
                        and boundary_gate != "fail"
                        and active_anchor_fit_gate["gate"] != "fail"
                        and payload.get("audio_qc", {}).get("gate") == "pass"
                    )
                    candidate["selection_score"] = countdown_chunk_slice_score(
                        candidate,
                        token_slot_sec,
                    )
                    ref_audio_path = str(candidate.get("ref_audio_path") or "")
                    candidate["ref_affinity_score"] = (
                        1.0
                        if ref_audio_path == ref.ref_audio_path
                        else 0.7
                        if str(candidate.get("ref_style") or "") == ref_style
                        else 0.35
                    )
                    candidate["texture_score"] = countdown_candidate_texture_score(candidate)
                    candidate["active_anchor_fit_score"] = active_anchor_fit_score
                    candidate["active_anchor_fit_gate"] = active_anchor_fit_gate
                    candidate["transcript_preference_score"] = transcript_preference_score
                    candidate["selection_score"] = round(
                        float(candidate["selection_score"]) * 0.84
                        + float(candidate["ref_affinity_score"]) * 0.08
                        + float(transcript_preference_score) * 0.04
                        + float(active_anchor_fit_score if active_anchor_fit_score is not None else 1.0)
                        * 0.04,
                        6,
                    )
                    if countdown_candidate_source_rank(candidate) >= 2:
                        candidate["selection_score"] = round(
                            min(float(candidate["selection_score"]) + 0.03, 1.0),
                            6,
                        )
                    quality_tier = countdown_carrier_candidate_quality_tier(candidate)
                    candidate["quality_tier"] = quality_tier
                    payload["countdown_carrier_token"]["quality_tier"] = quality_tier
                    payload["countdown_carrier_token"]["texture_score"] = candidate["texture_score"]
                    payload["countdown_carrier_token"]["ref_affinity_score"] = candidate[
                        "ref_affinity_score"
                    ]
                target_met = any(
                    countdown_carrier_quality_tier_rank(candidate.get("quality_tier")) >= target_rank
                    for candidate in candidates
                )
                if target_met or requested_rounds >= max_rounds:
                    break
                requested_rounds += 1
                if use_token_bank:
                    candidates = generate_token_bank_round(requested_rounds)
                else:
                    candidates = generate_countdown_carrier_candidates(
                        token_text=token_text,
                        segment=segment,
                        ref=ref,
                        ref_style=ref_style,
                        lane_index=lane_index,
                        tts_text_language=tts_text_language,
                        min_generation_rounds=requested_rounds,
                    )
            for candidate in candidates:
                candidate["quality_retry_rounds_requested"] = requested_rounds
                candidate["quality_retry_target_met"] = target_met
                candidate["payload"]["countdown_carrier_quality_retry"] = {
                    **retry_metadata,
                    "rounds_requested": requested_rounds,
                    "target_met": target_met,
                }
            return candidates

        def countdown_candidate_continuity_score(
            previous: dict[str, Any] | None,
            current: dict[str, Any],
        ) -> float:
            if previous is None:
                return 1.0
            score = 0.0
            if str(previous.get("ref_audio_path") or "") == str(current.get("ref_audio_path") or ""):
                score += 0.50
            if str(previous.get("ref_style") or "") == str(current.get("ref_style") or ""):
                score += 0.20
            prev_texture = (
                previous.get("payload", {})
                .get("countdown_texture_qc", {})
                .get("candidate")
            )
            curr_texture = (
                current.get("payload", {})
                .get("countdown_texture_qc", {})
                .get("candidate")
            )
            texture_match = _countdown_texture_match(prev_texture, curr_texture)
            score += float(texture_match.get("score", 1.0)) * 0.30
            return round(max(0.0, min(score, 1.0)), 6)

        def select_countdown_carrier_sequence_with_beam(
            placement_candidates: list[tuple[dict[str, Any], list[dict[str, Any]]]],
        ) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
            beam_width = int(getattr(cfg, "gsv_countdown_token_bank_beam_width", 8))
            beams: list[tuple[float, list[dict[str, Any]]]] = [(0.0, [])]
            evaluated_path_count = 0
            for _placement, candidates in placement_candidates:
                ranked_candidates = sorted(
                    candidates,
                    key=lambda candidate: (
                        countdown_carrier_quality_tier_rank(candidate.get("quality_tier")),
                        countdown_candidate_source_rank(candidate),
                        float(candidate.get("transcript_preference_score") or 0.0),
                        float(candidate.get("active_anchor_fit_score") or 0.0),
                        float(candidate.get("selection_score") or 0.0),
                    ),
                    reverse=True,
                )[:beam_width]
                next_beams: list[tuple[float, list[dict[str, Any]]]] = []
                for score, path in beams:
                    previous = path[-1] if path else None
                    for candidate in ranked_candidates:
                        continuity = countdown_candidate_continuity_score(previous, candidate)
                        tier_bonus = countdown_carrier_quality_tier_rank(candidate.get("quality_tier")) / 3.0
                        next_score = (
                            score
                            + float(candidate.get("selection_score") or 0.0)
                            + continuity * 0.20
                            + tier_bonus * 0.12
                        )
                        next_beams.append((next_score, [*path, candidate]))
                        evaluated_path_count += 1
                beams = sorted(next_beams, key=lambda item: item[0], reverse=True)[:beam_width]
            if not beams:
                return {}, {
                    "enabled": True,
                    "beam_width": beam_width,
                    "evaluated_path_count": evaluated_path_count,
                    "selected_score": 0.0,
                }
            selected_score, selected_path = beams[0]
            return {
                int(placement["token_index"]): candidate
                for (placement, _candidates), candidate in zip(
                    placement_candidates,
                    selected_path,
                    strict=True,
                )
            }, {
                "enabled": True,
                "beam_width": beam_width,
                "evaluated_path_count": evaluated_path_count,
                "selected_score": round(selected_score, 6),
            }

        def render_countdown_span_carrier_bank(
            span: list[tuple[int, Segment, list[int]]],
        ) -> set[str]:
            first_index, first_segment, _first_values = span[0]
            if not first_segment.script:
                return set()
            values = [value for _index, _segment, segment_values in span for value in segment_values]
            tokens = _countdown_spoken_tokens(values)
            if tokens is None:
                return set()
            start_gsv_servers()
            span_id = "countdown_" + "_".join(segment.id for _index, segment, _values in span)
            span_dir = project_dir / "work" / "tts" / "countdown"
            span_dir.mkdir(parents=True, exist_ok=True)
            ref_style = first_segment.script.ref_style
            resolved_ref_style = ref_style if ref_style in refs else "whisper_close"
            ref, ref_metadata = countdown_carrier_ref_for_span(
                span,
                ref_style=ref_style,
                resolved_ref_style=resolved_ref_style,
            )
            ref = _ref_for_tts_language(ref, first_segment.script.tts_language)
            sample_rate = cfg.mix_sample_rate
            placements, segment_frames, total_frames = countdown_token_slots_for_span(
                span,
                sample_rate,
            )
            if len(placements) != len(tokens):
                return set()
            lane_index = _segment_lane_index(first_segment, first_index - 1, gsv_lane_count)
            tts_text_language = _segment_tts_text_language(first_segment, cfg.target_language)
            segment_by_id = {segment.id: segment for _index, segment, _values in span}
            token_candidates: list[dict[str, Any]] = []
            failed_tokens: list[dict[str, Any]] = []
            selected_by_token: dict[int, dict[str, Any]] = {}
            placement_candidates: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
            ref_pool = countdown_token_bank_ref_pool(
                span=span,
                primary_ref_style=resolved_ref_style,
                primary_ref=ref,
                tts_text_language=tts_text_language,
            )
            token_bank_warmup_metadata = countdown_token_bank_metadata()
            span_beam_metadata: dict[str, Any] = {
                "enabled": bool(getattr(cfg, "gsv_countdown_token_bank_enabled", True)),
                "beam_width": int(getattr(cfg, "gsv_countdown_token_bank_beam_width", 8)),
                "evaluated_path_count": 0,
                "selected_score": 0.0,
            }

            with lane_locks[lane_index]:
                token_bank_warmup_metadata = countdown_token_bank_warmup_for_span(
                    span_id=span_id,
                    tokens=tokens,
                    segment=first_segment,
                    ref_pool=ref_pool,
                    lane_index=lane_index,
                    tts_text_language=tts_text_language,
                )
                for placement in placements:
                    token_index = int(placement["token_index"])
                    token_text = str(placement["text"])
                    segment = segment_by_id[str(placement["segment_id"])]
                    candidates = countdown_carrier_candidates_for_slot(
                        token_text=token_text,
                        token_slot_sec=float(placement["slot_duration_sec"]),
                        segment=segment,
                        ref=ref,
                        ref_style=resolved_ref_style,
                        lane_index=lane_index,
                        tts_text_language=tts_text_language,
                        token_bank_ref_pool=ref_pool,
                        source_prosody=countdown_source_prosody_for_placement(
                            segment,
                            placement,
                        ),
                    )
                    token_candidates.extend(
                        {key: value for key, value in candidate.items() if key != "audio"}
                        for candidate in candidates
                    )
                    accepted = [candidate for candidate in candidates if candidate["acceptable"]]
                    if not accepted:
                        failed_tokens.append(
                            {
                                "token_index": token_index,
                                "text": token_text,
                                "reason": "no_acceptable_countdown_carrier_candidate",
                                "candidates": [
                                    {key: value for key, value in candidate.items() if key != "audio"}
                                    for candidate in candidates
                                ],
                            }
                        )
                        continue
                    placement_candidates.append((placement, accepted))

                if not failed_tokens:
                    selected_by_token, span_beam_metadata = select_countdown_carrier_sequence_with_beam(
                        placement_candidates
                    )
                    selected_styles = sorted(
                        {
                            str(candidate.get("ref_style") or "")
                            for candidate in selected_by_token.values()
                        }
                    )
                    _log_stage_checkpoint(
                        stage_name,
                        "countdown token-bank beam",
                        (
                            f"span={span_id} paths={span_beam_metadata['evaluated_path_count']} "
                            f"score={span_beam_metadata['selected_score']} "
                            f"refs={','.join(style for style in selected_styles if style)}"
                        ),
                    )

            if failed_tokens:
                skip_payload = {
                    "reason": "no_acceptable_countdown_carrier_candidate",
                    "renderer": "countdown_carrier_bank",
                    "span_id": span_id,
                    "segment_ids": [segment.id for _index, segment, _values in span],
                    "values": values,
                    "tokens": tokens,
                    "carrier_bank": {
                        "carrier_template_count": len(countdown_carrier_templates_for_token(tokens[0])),
                        "carrier_candidate_count": int(
                            getattr(cfg, "gsv_countdown_carrier_candidate_count", 1)
                        ),
                        "slice_search_enabled": bool(
                            getattr(cfg, "gsv_countdown_carrier_slice_search_enabled", True)
                        ),
                        "slice_window_count": int(
                            getattr(cfg, "gsv_countdown_carrier_max_slice_windows_per_candidate", 1)
                        ),
                        "full_sentence_prefilter": countdown_carrier_full_sentence_prefilter_metadata(),
                        "quality_retry": countdown_carrier_quality_retry_metadata(),
                        "token_bank": {
                            **countdown_token_bank_metadata(),
                            "warmup": token_bank_warmup_metadata,
                            "ref_pool": [
                                {
                                    "ref_style": item["ref_style"],
                                    "ref_audio_path": item["ref"].ref_audio_path,
                                    "source": item["source"],
                                }
                                for item in ref_pool
                            ],
                        },
                        "span_beam_search": span_beam_metadata,
                        "early_stop": countdown_carrier_bank_early_stop_metadata(),
                        "segment_ref": ref_metadata["segment_ref"],
                        "using_segment_ref": ref_metadata["using_segment_ref"],
                    },
                    "failed_tokens": failed_tokens,
                    "token_candidates": token_candidates,
                }
                for _index, segment, _values in span:
                    segment.analysis["countdown_renderer_skip"] = skip_payload
                return set()

            span_audio = np.zeros((total_frames, 2), dtype=np.float32)
            placement_metadata: list[dict[str, Any]] = []
            for placement in placements:
                token_index = int(placement["token_index"])
                selected = selected_by_token[token_index]
                slot_start = int(placement["slot_start_frame"])
                slot_end = int(placement["slot_end_frame"])
                slot_frames = max(1, slot_end - slot_start)
                audio = _countdown_stereo(np.array(selected["audio"], copy=True))
                placement_anchor = placement.get("placement_anchor")
                selected_active_fit = (
                    selected["payload"]
                    .get("countdown_carrier_bank", {})
                    .get("active_anchor_fit", {})
                )
                active_leading_sec = (
                    _safe_float(selected_active_fit.get("leading_silence_sec"), 0.0)
                    if isinstance(selected_active_fit, dict)
                    else 0.0
                )
                active_leading_frames = max(0, int(round(active_leading_sec * sample_rate)))
                if placement_anchor in {
                    "source_anchor_exact",
                    "source_anchor_smoothed",
                    "source_word_start",
                }:
                    start_frame = _countdown_anchor_aligned_start_frame(
                        slot_start_frame=slot_start,
                        active_leading_frames=active_leading_frames,
                        total_frames=total_frames,
                    )
                else:
                    start_frame = slot_start + max(0, (slot_frames - len(audio)) // 2)
                end_frame = min(total_frames, start_frame + len(audio))
                source_frames = max(0, end_frame - start_frame)
                if source_frames:
                    span_audio[start_frame:end_frame] += audio[:source_frames]
                placed_active_start_frame = min(total_frames, start_frame + active_leading_frames)
                placed_active_start_sec = placed_active_start_frame / sample_rate
                source_anchor_sec = slot_start / sample_rate
                placement_metadata.append(
                    {
                        "segment_id": placement["segment_id"],
                        "value": placement["value"],
                        "text": placement["text"],
                        "token_index": token_index,
                        "slot_start_sec": round(slot_start / sample_rate, 6),
                        "slot_end_sec": round(slot_end / sample_rate, 6),
                        "slot_duration_sec": round(float(placement["slot_duration_sec"]), 6),
                        "placed_start_sec": round(start_frame / sample_rate, 6),
                        "placed_end_sec": round(end_frame / sample_rate, 6),
                        "placed_active_start_sec": round(placed_active_start_sec, 6),
                        "source_anchor_sec": round(source_anchor_sec, 6),
                        "active_anchor_delta_sec": round(
                            placed_active_start_sec - source_anchor_sec,
                            6,
                        ),
                        "selected_active_anchor_fit": selected_active_fit
                        if isinstance(selected_active_fit, dict)
                        else None,
                        "selected_active_anchor_fit_score": selected.get(
                            "active_anchor_fit_score"
                        ),
                        "selected_carrier_index": selected["carrier_index"],
                        "selected_carrier_variant_index": selected["carrier_variant_index"],
                        "selected_slice_window_index": selected.get("slice_window_index"),
                        "selected_slice_window_kind": selected["payload"]
                        .get("countdown_carrier_bank", {})
                        .get("slice_window", {})
                        .get("kind"),
                        "selected_slice_start_sec": selected["payload"]
                        .get("countdown_carrier_bank", {})
                        .get("slice_start_sec"),
                        "selected_slice_end_sec": selected["payload"]
                        .get("countdown_carrier_bank", {})
                        .get("slice_end_sec"),
                        "selected_slice_duration_sec": selected["payload"]
                        .get("countdown_carrier_bank", {})
                        .get("slice_duration_sec"),
                        "selected_carrier_template": selected["carrier_template"],
                        "selected_carrier_text": selected["carrier_text"],
                        "selected_candidate_source": selected.get("candidate_source")
                        or selected["payload"].get("candidate_source")
                        or "sentence_carrier",
                        "selected_carrier_generation_profile": selected[
                            "carrier_generation_profile"
                        ],
                        "selected_candidate_index": selected["candidate_index"],
                        "selected_duration_sec": round(float(selected["duration_sec"]), 6),
                        "selected_path": selected["output_path"],
                        "selected_quality_tier": selected.get("quality_tier", "reject"),
                        "selected_quality_retry_rounds": selected.get(
                            "quality_retry_rounds_requested"
                        ),
                        "selected_ref_style": selected.get("ref_style"),
                        "selected_ref_audio_path": selected.get("ref_audio_path"),
                        "selected_texture_score": selected.get("texture_score"),
                        "selected_ref_affinity_score": selected.get("ref_affinity_score"),
                        "selected_pronunciation_gate": selected["pronunciation_gate"],
                        "selected_stability_score": selected["payload"]["stability_qc"][
                            "stability_score"
                        ],
                        "selected_prosody_score": selected["payload"]
                        .get("countdown_prosody_qc", {})
                        .get("score"),
                        "selection_score": selected["selection_score"],
                        "placement_anchor": placement.get("placement_anchor", "slot_center"),
                    }
                )

            peak = float(np.max(np.abs(span_audio))) if span_audio.size else 0.0
            if peak > 0.98:
                span_audio *= 0.98 / peak
            span_path = span_dir / f"{span_id}.wav"
            write_audio(span_path, span_audio, sample_rate)
            slice_window_count = max(
                (
                    int(
                        candidate.get("payload", {})
                        .get("countdown_carrier_bank", {})
                        .get("slice_window_count", 1)
                    )
                    for candidate in token_candidates
                ),
                default=0,
            )
            span_metadata = {
                "span_id": span_id,
                "renderer": "countdown_carrier_bank",
                "segment_ids": [segment.id for _index, segment, _values in span],
                "values": values,
                "tokens": tokens,
                "target_duration_sec": round(total_frames / sample_rate, 6),
                "span_path": str(span_path),
                "token_placements": placement_metadata,
                "carrier_bank": {
                    "carrier_template_count": len(countdown_carrier_templates_for_token(tokens[0])),
                    "carrier_candidate_count": int(
                        getattr(cfg, "gsv_countdown_carrier_candidate_count", 1)
                    ),
                    "slice_search_enabled": bool(
                        getattr(cfg, "gsv_countdown_carrier_slice_search_enabled", True)
                    ),
                    "slice_window_count": slice_window_count,
                    "full_sentence_prefilter": countdown_carrier_full_sentence_prefilter_metadata(),
                    "quality_retry": countdown_carrier_quality_retry_metadata(),
                    "token_bank": {
                        **countdown_token_bank_metadata(),
                        "warmup": token_bank_warmup_metadata,
                        "ref_pool": [
                            {
                                "ref_style": item["ref_style"],
                                "ref_audio_path": item["ref"].ref_audio_path,
                                "source": item["source"],
                            }
                            for item in ref_pool
                        ],
                    },
                    "span_beam_search": span_beam_metadata,
                    "early_stop": countdown_carrier_bank_early_stop_metadata(),
                    "token_candidate_count": len(token_candidates),
                    "segment_ref": ref_metadata["segment_ref"],
                    "using_segment_ref": ref_metadata["using_segment_ref"],
                    "ref_audio_path": ref.ref_audio_path,
                    "prompt_text": ref.prompt_text,
                    "prompt_lang": ref.prompt_lang,
                },
                "token_candidates": token_candidates,
                "source_language": cfg.source_language,
                "target_language": cfg.target_language,
                "ref_style": resolved_ref_style,
            }
            metadata_path = span_dir / f"{span_id}.json"
            write_json_atomic(metadata_path, span_metadata)

            offset = 0
            for _segment_index, (_index, segment, segment_values) in enumerate(span):
                frames = segment_frames[segment.id]
                segment_audio = span_audio[offset : offset + frames]
                offset += frames
                if len(segment_audio) < frames:
                    padding = np.zeros((frames - len(segment_audio), 2), dtype=np.float32)
                    segment_audio = np.concatenate([segment_audio, padding], axis=0)
                elif len(segment_audio) > frames:
                    segment_audio = segment_audio[:frames]
                final_path = project_dir / "work" / "tts" / f"{segment.id}_final.wav"
                write_audio(final_path, segment_audio, sample_rate)
                final_duration = duration_sec(final_path)
                final_ratio = duration_ratio(final_duration, segment.duration)
                segment_tokens = _countdown_spoken_tokens(segment_values) or []
                segment_placements = [
                    placement
                    for placement in placement_metadata
                    if placement["segment_id"] == segment.id
                ]
                payload = {
                    "renderer": "countdown_carrier_bank",
                    "span_id": span_id,
                    "span_metadata_path": str(metadata_path),
                    "span_path": str(span_path),
                    "segment_id": segment.id,
                    "segment_values": segment_values,
                    "segment_tokens": segment_tokens,
                    "values": values,
                    "tokens": tokens,
                    "token_placements": segment_placements,
                    "carrier_bank": span_metadata["carrier_bank"],
                    "target_duration_sec": segment.duration,
                    "duration_ratio": final_ratio,
                    "duration_gate": "pass",
                    "audio_qc": {
                        "gate": "pass",
                        "peak_dbfs": round(peak_dbfs(final_path), 3),
                        "rms_dbfs": round(rms_dbfs(final_path), 3),
                    },
                }
                candidate = TTSCandidate(
                    candidate_index=0,
                    seed=cfg.base_seed + 95_000 + first_index,
                    payload=payload,
                    output_path=str(final_path),
                    duration_sec=final_duration,
                    backend="gpt-sovits-countdown-renderer",
                    selected=True,
                    duration_ratio=final_ratio,
                    duration_gate="pass",
                    acceptable_for_mix=True,
                    selection_score=max(0.0, 1.0 - min(abs(final_ratio - 1.0), 1.0)),
                    selection_reason="countdown_carrier_bank",
                    retry_summary={"countdown_renderer": True, "span_id": span_id},
                )
                segment.tts = TTSMetadata(
                    backend="gpt-sovits-countdown-renderer",
                    ref_style=resolved_ref_style,
                    speed_factor=float(
                        getattr(cfg, "gsv_countdown_token_speed_factor", cfg.gsv_tts_max_speed_factor)
                    ),
                    candidate_count=1,
                    selected_candidate_path=str(final_path),
                    candidates=[candidate],
                    source_language=cfg.source_language,
                    target_language=cfg.target_language,
                    cross_lingual_voice_transfer=cfg.source_language != cfg.target_language,
                    retry_summary={
                        "countdown_renderer": True,
                        "countdown_renderer_mode": "carrier_bank",
                        "span_id": span_id,
                        "span_metadata_path": str(metadata_path),
                        "selected_duration_gate": "pass",
                        "selected_acceptable_for_mix": True,
                        "selected_duration_ratio": final_ratio,
                    },
                )
                segment.analysis["countdown_renderer"] = {
                    "renderer": "countdown_carrier_bank",
                    "span_id": span_id,
                    "segment_values": segment_values,
                    "segment_tokens": segment_tokens,
                    "token_placements": segment_placements,
                    "carrier_bank": span_metadata["carrier_bank"],
                    "span_metadata_path": str(metadata_path),
                }
                segment.status = "synthesized"
            return {segment.id for _index, segment, _values in span}

        def countdown_chunk_tts_options(
            *,
            candidate_index: int,
            sequence_seed: int,
            tts_text_language: str,
        ) -> GPTSoVITSTTSOptions:
            return GPTSoVITSTTSOptions(
                seed=cfg.base_seed + 90_000 + sequence_seed + candidate_index * 1_009,
                speed_factor=float(
                    getattr(cfg, "gsv_countdown_token_speed_factor", cfg.gsv_tts_max_speed_factor)
                ),
                text_lang=tts_text_language,
                top_k=cfg.gsv_top_k,
                top_p=cfg.gsv_top_p,
                temperature=float(getattr(cfg, "gsv_countdown_temperature", cfg.gsv_temperature)),
                text_split_method=cfg.gsv_text_split_method,
                fragment_interval=0.0,
                parallel_infer=cfg.gsv_parallel_infer,
                repetition_penalty=cfg.gsv_repetition_penalty,
                sample_steps=cfg.gsv_sample_steps,
                super_sampling=cfg.gsv_super_sampling,
                overlap_length=cfg.gsv_overlap_length,
                min_chunk_length=cfg.gsv_min_chunk_length,
            )

        def synthesize_countdown_chunk_take(
            *,
            chunk_dir: Path,
            span_id: str,
            chunk_start: int,
            chunk_tokens: list[str],
            candidate_index: int,
            first_segment: Segment,
            ref: GPTSoVITSRef,
            lane_index: int,
            tts_text_language: str,
            ref_style: str,
            target_duration_sec: float,
        ) -> dict[str, Any]:
            chunk_size = len(chunk_tokens)
            chunk_text = _countdown_phrase_tts_text(chunk_tokens)
            sequence_seed = sum(
                (index + 1) * ord(char)
                for index, char in enumerate(f"{chunk_start}:{chunk_size}:{chunk_text}")
            )
            options = countdown_chunk_tts_options(
                candidate_index=candidate_index,
                sequence_seed=sequence_seed,
                tts_text_language=tts_text_language,
            )
            cache_key = (
                ref_style,
                chunk_text,
                candidate_index,
                options.speed_factor,
                options.text_lang,
                options.top_k,
                options.top_p,
                options.temperature,
                options.text_split_method,
                options.fragment_interval,
                options.parallel_infer,
                options.repetition_penalty,
                options.sample_steps,
                options.super_sampling,
                options.overlap_length,
                options.min_chunk_length,
            )
            with countdown_chunk_lock:
                cached = countdown_chunk_cache.get(cache_key)
            if cached is None:
                chunk_label = _countdown_chunk_label(chunk_text)
                phrase_path = (
                    chunk_dir
                    / f"{span_id}_chunk_start_{chunk_start:02d}_size_{chunk_size:02d}_"
                    f"cand_{candidate_index:02d}_{chunk_label}.wav"
                )
                payload: dict[str, Any] = {
                    "renderer": "countdown_chunk_bank",
                    "chunk_start_index": chunk_start,
                    "chunk_size": chunk_size,
                    "chunk_text": chunk_text,
                    "candidate_index": candidate_index,
                    "ref_style": ref_style,
                    "target_chunk_duration_sec": round(target_duration_sec, 6),
                    "lane_index": lane_index,
                    "gsv_url": None if mock else gsv_base_urls[lane_index],
                }
                payload.update(_tts_request_debug_payload(chunk_text, ref, options))
                if mock:
                    _mock_synthesize(
                        phrase_path,
                        max(0.12, min(0.35 * chunk_size, target_duration_sec)),
                        options.seed,
                        cfg.mix_sample_rate,
                    )
                    payload["mock"] = True
                else:
                    client = clients[lane_index]
                    request = client.build_payload(chunk_text, ref, options)
                    payload.update(request.as_payload())
                    client.synthesize_to_file(request, phrase_path)
                raw_duration = duration_sec(phrase_path)
                postprocess_tts_candidate(phrase_path, payload)
                data, sample_rate = load_audio(phrase_path)
                data = _countdown_stereo(data)
                if sample_rate != cfg.mix_sample_rate:
                    data = resample_linear(data, sample_rate, cfg.mix_sample_rate)
                    sample_rate = cfg.mix_sample_rate
                payload["countdown_chunk_take"] = {
                    "raw_duration_sec": round(raw_duration, 6),
                    "duration_sec": round(len(data) / sample_rate, 6),
                    "phrase_path": str(phrase_path),
                }
                new_cached = {
                    "audio": data,
                    "sample_rate": sample_rate,
                    "payload": copy.deepcopy(payload),
                    "phrase_path": str(phrase_path),
                    "raw_duration": raw_duration,
                    "duration": len(data) / sample_rate,
                }
                with countdown_chunk_lock:
                    cached = countdown_chunk_cache.setdefault(cache_key, new_cached)
                reused_existing = cached is not new_cached
            else:
                reused_existing = True
            payload = copy.deepcopy(cached["payload"])
            payload["countdown_chunk_bank"] = {
                "cache_key": [str(item) for item in cache_key],
                "reused_existing": reused_existing,
            }
            return {
                "audio": np.array(cached["audio"], copy=True),
                "sample_rate": int(cached["sample_rate"]),
                "payload": payload,
                "phrase_path": str(cached["phrase_path"]),
                "raw_duration": float(cached["raw_duration"]),
                "duration": float(cached["duration"]),
                "chunk_start": chunk_start,
                "chunk_size": chunk_size,
                "chunk_text": chunk_text,
                "candidate_index": candidate_index,
            }

        def render_countdown_span_chunk_bank(
            span: list[tuple[int, Segment, list[int]]],
        ) -> set[str]:
            first_index, first_segment, _first_values = span[0]
            if not first_segment.script:
                return set()
            values = [value for _index, _segment, segment_values in span for value in segment_values]
            tokens = _countdown_spoken_tokens(values)
            if tokens is None:
                return set()
            start_gsv_servers()
            span_id = "countdown_" + "_".join(segment.id for _index, segment, _values in span)
            span_dir = project_dir / "work" / "tts" / "countdown"
            chunk_dir = span_dir / "chunk_bank" / span_id
            slice_dir = span_dir / "chunk_bank_slices"
            span_dir.mkdir(parents=True, exist_ok=True)
            chunk_dir.mkdir(parents=True, exist_ok=True)
            slice_dir.mkdir(parents=True, exist_ok=True)
            ref_style = first_segment.script.ref_style
            resolved_ref_style = ref_style if ref_style in refs else "whisper_close"
            ref = _ref_for_tts_language(resolve_ref(refs, ref_style), first_segment.script.tts_language)
            sample_rate = cfg.mix_sample_rate
            placements, segment_frames, total_frames = countdown_token_slots_for_span(
                span,
                sample_rate,
            )
            if len(placements) != len(tokens):
                return set()
            lane_index = _segment_lane_index(first_segment, first_index - 1, gsv_lane_count)
            candidate_count = int(getattr(cfg, "gsv_countdown_chunk_candidate_count", 10))
            max_chunk_size = min(
                len(tokens),
                max(1, int(getattr(cfg, "gsv_countdown_chunk_max_size", 10))),
            )
            tts_text_language = _segment_tts_text_language(first_segment, cfg.target_language)
            segment_by_id = {segment.id: segment for _index, segment, _values in span}
            slice_candidates_by_token: dict[int, list[dict[str, Any]]] = {
                int(placement["token_index"]): [] for placement in placements
            }
            chunk_take_summaries: list[dict[str, Any]] = []
            slice_candidate_summaries: list[dict[str, Any]] = []

            for chunk_size in range(1, max_chunk_size + 1):
                for chunk_start in range(0, len(tokens) - chunk_size + 1):
                    chunk_tokens = tokens[chunk_start : chunk_start + chunk_size]
                    chunk_slot_duration = sum(
                        float(placements[index]["slot_duration_sec"])
                        for index in range(chunk_start, chunk_start + chunk_size)
                    )
                    for candidate_index in range(candidate_count):
                        with lane_locks[lane_index]:
                            take = synthesize_countdown_chunk_take(
                                chunk_dir=chunk_dir,
                                span_id=span_id,
                                chunk_start=chunk_start,
                                chunk_tokens=chunk_tokens,
                                candidate_index=candidate_index,
                                first_segment=first_segment,
                                ref=ref,
                                lane_index=lane_index,
                                tts_text_language=tts_text_language,
                                ref_style=resolved_ref_style,
                                target_duration_sec=chunk_slot_duration,
                            )
                        phrase_audio = _countdown_stereo(take["audio"])
                        boundaries = countdown_phrase_slice_boundaries(
                            phrase_audio,
                            chunk_size,
                            sample_rate,
                        )
                        chunk_summary = {
                            "chunk_start_index": chunk_start,
                            "chunk_size": chunk_size,
                            "chunk_text": take["chunk_text"],
                            "candidate_index": candidate_index,
                            "phrase_path": take["phrase_path"],
                            "duration_sec": round(float(take["duration"]), 6),
                            "slice_count": len(boundaries),
                        }
                        chunk_take_summaries.append(chunk_summary)
                        if len(boundaries) != chunk_size:
                            continue
                        for local_index, (slice_start, slice_end) in enumerate(boundaries):
                            token_index = chunk_start + local_index
                            placement = placements[token_index]
                            token_text = str(placement["text"])
                            token_label = _countdown_chunk_label(token_text)
                            slice_audio = _countdown_stereo(phrase_audio[slice_start:slice_end])
                            slice_path = (
                                slice_dir
                                / f"{span_id}_chunk_size_{chunk_size:02d}_start_{chunk_start:02d}_"
                                f"cand_{candidate_index:02d}_token_{token_index:02d}_{token_label}.wav"
                            )
                            write_audio(slice_path, slice_audio, sample_rate)
                            slice_duration = duration_sec(slice_path)
                            duration_gate, token_min_sec, token_max_sec = countdown_token_duration_gate(
                                duration=slice_duration,
                                token_slot_sec=float(placement["slot_duration_sec"]),
                                token_text=token_text,
                            )
                            pronunciation_qc = run_pronunciation_qc(
                                segment=segment_by_id[str(placement["segment_id"])],
                                candidate_path=slice_path,
                                expected_text=token_text,
                                candidate_duration_sec=slice_duration,
                                short_slice=True,
                            )
                            pronunciation_gate = str(
                                (pronunciation_qc or {}).get("gate") or "not_run"
                            ).strip().lower()
                            pronunciation_ok = countdown_pronunciation_contract_ok(
                                segment_by_id[str(placement["segment_id"])],
                                pronunciation_qc,
                            )
                            stability_qc = countdown_slice_stability_metrics(
                                slice_audio,
                                sample_rate,
                            )
                            audio_metrics = {
                                "gate": "pass",
                                "peak_dbfs": round(peak_dbfs(slice_path), 3),
                                "rms_dbfs": round(rms_dbfs(slice_path), 3),
                            }
                            payload = {
                                "renderer": "countdown_chunk_bank",
                                "chunk_start_index": chunk_start,
                                "chunk_size": chunk_size,
                                "chunk_text": take["chunk_text"],
                                "candidate_index": candidate_index,
                                "token_index": token_index,
                                "token_text": token_text,
                                "pronunciation_qc": pronunciation_qc,
                                "stability_qc": stability_qc,
                                "audio_qc": audio_metrics,
                                "countdown_token_candidate": {
                                    "duration_sec": round(slice_duration, 6),
                                    "duration_gate": duration_gate,
                                    "token_min_sec": round(token_min_sec, 6),
                                    "token_max_sec": round(token_max_sec, 6),
                                    "pronunciation_gate": pronunciation_gate,
                                    "strict_pronunciation_required": countdown_strict_pronunciation_required(
                                        segment_by_id[str(placement["segment_id"])]
                                    ),
                                    "pronunciation_contract_ok": pronunciation_ok,
                                },
                                "source_take": {
                                    "phrase_path": take["phrase_path"],
                                    "slice_start_sec": round(slice_start / sample_rate, 6),
                                    "slice_end_sec": round(slice_end / sample_rate, 6),
                                },
                            }
                            attach_countdown_prosody_qc(
                                payload,
                                slice_audio,
                                sample_rate,
                                countdown_source_prosody_for_placement(
                                    segment_by_id[str(placement["segment_id"])],
                                    placement,
                                ),
                            )
                            candidate = {
                                "candidate_index": candidate_index,
                                "chunk_start_index": chunk_start,
                                "chunk_size": chunk_size,
                                "chunk_text": take["chunk_text"],
                                "token_index": token_index,
                                "text": token_text,
                                "output_path": str(slice_path),
                                "duration_sec": slice_duration,
                                "duration_gate": duration_gate,
                                "pronunciation_gate": pronunciation_gate,
                                "acceptable": (
                                    duration_gate == "pass"
                                    and pronunciation_ok
                                    and countdown_prosody_contract_ok({"payload": payload})
                                    and audio_metrics["gate"] == "pass"
                                ),
                                "payload": payload,
                                "audio": slice_audio,
                            }
                            candidate["selection_score"] = countdown_chunk_slice_score(
                                candidate,
                                float(placement["slot_duration_sec"]),
                            )
                            slice_candidates_by_token[token_index].append(candidate)
                            slice_candidate_summaries.append(
                                {key: value for key, value in candidate.items() if key != "audio"}
                            )

            selected_by_token: dict[int, dict[str, Any]] = {}
            failed_tokens: list[dict[str, Any]] = []
            for placement in placements:
                token_index = int(placement["token_index"])
                candidates = slice_candidates_by_token.get(token_index, [])
                accepted = [candidate for candidate in candidates if candidate["acceptable"]]
                pool = accepted or candidates
                if not pool:
                    failed_tokens.append(
                        {
                            "token_index": token_index,
                            "text": placement["text"],
                            "reason": "no_chunk_slice_candidate",
                        }
                    )
                    continue
                selected = max(
                    pool,
                    key=lambda candidate: (
                        float(candidate["selection_score"]),
                        int(candidate["chunk_size"]),
                        -abs(
                            float(candidate["duration_sec"])
                            - float(placement["slot_duration_sec"]) * 0.58
                        ),
                    ),
                )
                if not selected["acceptable"]:
                    failed_tokens.append(
                        {
                            "token_index": token_index,
                            "text": placement["text"],
                            "reason": "no_acceptable_chunk_slice_candidate",
                            "best_candidate": {
                                key: value for key, value in selected.items() if key != "audio"
                            },
                        }
                    )
                    continue
                selected_by_token[token_index] = selected
            if failed_tokens:
                skip_payload = {
                    "reason": "no_acceptable_countdown_chunk_slice_candidate",
                    "renderer": "countdown_chunk_bank",
                    "span_id": span_id,
                    "segment_ids": [segment.id for _index, segment, _values in span],
                    "values": values,
                    "tokens": tokens,
                    "failed_tokens": failed_tokens,
                    "chunk_takes": chunk_take_summaries,
                    "slice_candidates": slice_candidate_summaries,
                }
                for _index, segment, _values in span:
                    segment.analysis["countdown_renderer_skip"] = skip_payload
                return set()

            span_audio = np.zeros((total_frames, 2), dtype=np.float32)
            placement_metadata: list[dict[str, Any]] = []
            for placement in placements:
                selected = selected_by_token[int(placement["token_index"])]
                slot_start = int(placement["slot_start_frame"])
                slot_end = int(placement["slot_end_frame"])
                slot_frames = max(1, slot_end - slot_start)
                audio = _countdown_stereo(np.array(selected["audio"], copy=True))
                placement_anchor = placement.get("placement_anchor")
                if placement_anchor in {"source_anchor_exact", "source_anchor_smoothed"}:
                    start_frame = max(0, min(slot_start, total_frames))
                elif placement_anchor == "source_word_start":
                    start_frame = min(slot_start, max(0, total_frames - len(audio)))
                else:
                    start_frame = slot_start + max(0, (slot_frames - len(audio)) // 2)
                end_frame = min(total_frames, start_frame + len(audio))
                source_frames = max(0, end_frame - start_frame)
                if source_frames:
                    span_audio[start_frame:end_frame] += audio[:source_frames]
                placement_metadata.append(
                    {
                        "segment_id": placement["segment_id"],
                        "value": placement["value"],
                        "text": placement["text"],
                        "token_index": placement["token_index"],
                        "slot_start_sec": round(slot_start / sample_rate, 6),
                        "slot_end_sec": round(slot_end / sample_rate, 6),
                        "slot_duration_sec": round(float(placement["slot_duration_sec"]), 6),
                        "placed_start_sec": round(start_frame / sample_rate, 6),
                        "placed_end_sec": round(end_frame / sample_rate, 6),
                        "selected_candidate_index": selected["candidate_index"],
                        "selected_chunk_start_index": selected["chunk_start_index"],
                        "selected_chunk_size": selected["chunk_size"],
                        "selected_chunk_text": selected["chunk_text"],
                        "selected_duration_sec": round(float(selected["duration_sec"]), 6),
                        "selected_path": selected["output_path"],
                        "selected_pronunciation_gate": selected["pronunciation_gate"],
                        "selected_stability_score": selected["payload"]["stability_qc"][
                            "stability_score"
                        ],
                        "selected_prosody_score": selected["payload"]
                        .get("countdown_prosody_qc", {})
                        .get("score"),
                        "selection_score": selected["selection_score"],
                        "placement_anchor": placement.get("placement_anchor", "slot_center"),
                    }
                )

            peak = float(np.max(np.abs(span_audio))) if span_audio.size else 0.0
            if peak > 0.98:
                span_audio *= 0.98 / peak
            span_path = span_dir / f"{span_id}.wav"
            write_audio(span_path, span_audio, sample_rate)
            span_metadata = {
                "span_id": span_id,
                "renderer": "countdown_chunk_bank",
                "segment_ids": [segment.id for _index, segment, _values in span],
                "values": values,
                "tokens": tokens,
                "target_duration_sec": round(total_frames / sample_rate, 6),
                "span_path": str(span_path),
                "token_placements": placement_metadata,
                "chunk_bank": {
                    "chunk_sizes": list(range(1, max_chunk_size + 1)),
                    "candidate_count_per_chunk": candidate_count,
                    "chunk_take_count": len(chunk_take_summaries),
                    "slice_candidate_count": len(slice_candidate_summaries),
                },
                "chunk_takes": chunk_take_summaries,
                "slice_candidates": slice_candidate_summaries,
                "source_language": cfg.source_language,
                "target_language": cfg.target_language,
                "ref_style": resolved_ref_style,
            }
            metadata_path = span_dir / f"{span_id}.json"
            write_json_atomic(metadata_path, span_metadata)

            offset = 0
            for _segment_index, (_index, segment, segment_values) in enumerate(span):
                frames = segment_frames[segment.id]
                segment_audio = span_audio[offset : offset + frames]
                offset += frames
                if len(segment_audio) < frames:
                    padding = np.zeros((frames - len(segment_audio), 2), dtype=np.float32)
                    segment_audio = np.concatenate([segment_audio, padding], axis=0)
                elif len(segment_audio) > frames:
                    segment_audio = segment_audio[:frames]
                final_path = project_dir / "work" / "tts" / f"{segment.id}_final.wav"
                write_audio(final_path, segment_audio, sample_rate)
                final_duration = duration_sec(final_path)
                final_ratio = duration_ratio(final_duration, segment.duration)
                segment_tokens = _countdown_spoken_tokens(segment_values) or []
                segment_placements = [
                    placement
                    for placement in placement_metadata
                    if placement["segment_id"] == segment.id
                ]
                payload = {
                    "renderer": "countdown_chunk_bank",
                    "span_id": span_id,
                    "span_metadata_path": str(metadata_path),
                    "span_path": str(span_path),
                    "segment_id": segment.id,
                    "segment_values": segment_values,
                    "segment_tokens": segment_tokens,
                    "values": values,
                    "tokens": tokens,
                    "token_placements": segment_placements,
                    "chunk_bank": span_metadata["chunk_bank"],
                    "target_duration_sec": segment.duration,
                    "duration_ratio": final_ratio,
                    "duration_gate": "pass",
                    "audio_qc": {
                        "gate": "pass",
                        "peak_dbfs": round(peak_dbfs(final_path), 3),
                        "rms_dbfs": round(rms_dbfs(final_path), 3),
                    },
                }
                candidate = TTSCandidate(
                    candidate_index=0,
                    seed=cfg.base_seed + 90_000 + first_index,
                    payload=payload,
                    output_path=str(final_path),
                    duration_sec=final_duration,
                    backend="gpt-sovits-countdown-renderer",
                    selected=True,
                    duration_ratio=final_ratio,
                    duration_gate="pass",
                    acceptable_for_mix=True,
                    selection_score=max(0.0, 1.0 - min(abs(final_ratio - 1.0), 1.0)),
                    selection_reason="countdown_chunk_bank",
                    retry_summary={"countdown_renderer": True, "span_id": span_id},
                )
                segment.tts = TTSMetadata(
                    backend="gpt-sovits-countdown-renderer",
                    ref_style=resolved_ref_style,
                    speed_factor=float(
                        getattr(cfg, "gsv_countdown_token_speed_factor", cfg.gsv_tts_max_speed_factor)
                    ),
                    candidate_count=1,
                    selected_candidate_path=str(final_path),
                    candidates=[candidate],
                    source_language=cfg.source_language,
                    target_language=cfg.target_language,
                    cross_lingual_voice_transfer=cfg.source_language != cfg.target_language,
                    retry_summary={
                        "countdown_renderer": True,
                        "countdown_renderer_mode": "chunk_bank",
                        "span_id": span_id,
                        "span_metadata_path": str(metadata_path),
                        "selected_duration_gate": "pass",
                        "selected_acceptable_for_mix": True,
                        "selected_duration_ratio": final_ratio,
                    },
                )
                segment.analysis["countdown_renderer"] = {
                    "renderer": "countdown_chunk_bank",
                    "span_id": span_id,
                    "segment_values": segment_values,
                    "segment_tokens": segment_tokens,
                    "token_placements": segment_placements,
                    "chunk_bank": span_metadata["chunk_bank"],
                    "span_metadata_path": str(metadata_path),
                }
                segment.status = "synthesized"
            return {segment.id for _index, segment, _values in span}

        def render_countdown_span_phrase_slice(
            span: list[tuple[int, Segment, list[int]]],
            *,
            fallback_from: str,
        ) -> set[str]:
            first_index, first_segment, _first_values = span[0]
            if not first_segment.script:
                return set()
            values = [value for _index, _segment, segment_values in span for value in segment_values]
            tokens = _countdown_spoken_tokens(values)
            if tokens is None:
                return set()
            start_gsv_servers()
            span_id = "countdown_" + "_".join(segment.id for _index, segment, _values in span)
            span_dir = project_dir / "work" / "tts" / "countdown"
            slice_dir = span_dir / "phrase_slices"
            span_dir.mkdir(parents=True, exist_ok=True)
            slice_dir.mkdir(parents=True, exist_ok=True)
            ref_style = first_segment.script.ref_style
            resolved_ref_style = ref_style if ref_style in refs else "whisper_close"
            ref = _ref_for_tts_language(resolve_ref(refs, ref_style), first_segment.script.tts_language)
            sample_rate = cfg.mix_sample_rate
            placements, segment_frames, total_frames = countdown_token_slots_for_span(
                span,
                sample_rate,
            )
            if len(placements) != len(tokens):
                return set()
            lane_index = _segment_lane_index(first_segment, first_index - 1, gsv_lane_count)
            candidate_count = int(getattr(cfg, "gsv_countdown_candidate_count", 8))
            total_duration_sec = total_frames / sample_rate
            compact_text = _countdown_phrase_tts_text(tokens)
            segment_by_id = {segment.id: segment for _index, segment, _values in span}
            phrase_candidates: list[dict[str, Any]] = []
            selected_slices: list[dict[str, Any]] | None = None
            selected_phrase_payload: dict[str, Any] | None = None

            for candidate_index in range(candidate_count):
                with lane_locks[lane_index]:
                    phrase_audio, phrase_payload = token_audio_for_countdown(
                        token_text=compact_text,
                        token_index=0,
                        token_count=len(tokens),
                        span_id=span_id,
                        ref=ref,
                        token_slot_sec=total_duration_sec,
                        lane_index=lane_index,
                        tts_text_language=_segment_tts_text_language(
                            first_segment,
                            cfg.target_language,
                        ),
                        ref_style=resolved_ref_style,
                        candidate_index=candidate_index,
                    )
                if len(phrase_audio) > total_frames:
                    phrase_payload["countdown_phrase_slice_fit"] = {
                        "fit_strategy": "raw_no_linear_resample",
                        "raw_duration_sec": round(len(phrase_audio) / sample_rate, 6),
                        "target_span_duration_sec": round(total_duration_sec, 6),
                    }
                boundaries = countdown_phrase_slice_boundaries(
                    phrase_audio,
                    len(placements),
                    sample_rate,
                )
                if len(boundaries) != len(placements):
                    phrase_candidates.append(
                        {
                            "candidate_index": candidate_index,
                            "phrase_payload": phrase_payload,
                            "slices": [],
                            "acceptable": False,
                            "reason": "phrase_slice_boundary_count_mismatch",
                        }
                    )
                    continue
                phrase_slices: list[dict[str, Any]] = []
                for placement, (slice_start, slice_end) in zip(placements, boundaries, strict=True):
                    token_text = str(placement["text"])
                    token_label = _countdown_chunk_label(token_text)
                    slice_path = (
                        slice_dir
                        / f"{span_id}_phrase_slice_cand_{candidate_index:02d}_"
                        f"{int(placement['token_index']):02d}_{token_label}.wav"
                    )
                    slice_audio = phrase_audio[slice_start:slice_end]
                    write_audio(slice_path, slice_audio, sample_rate)
                    slice_duration = duration_sec(slice_path)
                    pronunciation_qc = run_pronunciation_qc(
                        segment=segment_by_id[str(placement["segment_id"])],
                        candidate_path=slice_path,
                        expected_text=token_text,
                        candidate_duration_sec=slice_duration,
                        short_slice=True,
                    )
                    pronunciation_gate = str(
                        (pronunciation_qc or {}).get("gate") or "not_run"
                    ).strip().lower()
                    duration_gate, _token_min_sec, _token_max_sec = countdown_token_duration_gate(
                        duration=slice_duration,
                        token_slot_sec=float(placement["slot_duration_sec"]),
                        token_text=token_text,
                    )
                    pronunciation_ok = countdown_pronunciation_contract_ok(
                        segment_by_id[str(placement["segment_id"])],
                        pronunciation_qc,
                    )
                    slice_payload: dict[str, Any] = {}
                    attach_countdown_prosody_qc(
                        slice_payload,
                        slice_audio,
                        sample_rate,
                        countdown_source_prosody_for_placement(
                            segment_by_id[str(placement["segment_id"])],
                            placement,
                        ),
                    )
                    phrase_slices.append(
                        {
                            "candidate_index": candidate_index,
                            "segment_id": placement["segment_id"],
                            "value": placement["value"],
                            "text": token_text,
                            "token_index": placement["token_index"],
                            "output_path": str(slice_path),
                            "duration_sec": slice_duration,
                            "duration_gate": duration_gate,
                            "slice_start_sec": round(slice_start / sample_rate, 6),
                            "slice_end_sec": round(slice_end / sample_rate, 6),
                            "pronunciation_qc": pronunciation_qc,
                            "pronunciation_gate": pronunciation_gate,
                            "strict_pronunciation_required": countdown_strict_pronunciation_required(
                                segment_by_id[str(placement["segment_id"])]
                            ),
                            "countdown_prosody_qc": slice_payload.get("countdown_prosody_qc"),
                            "acceptable": duration_gate == "pass"
                            and pronunciation_ok
                            and countdown_prosody_contract_ok({"payload": slice_payload}),
                            "audio": slice_audio,
                        }
                    )
                phrase_candidates.append(
                    {
                        "candidate_index": candidate_index,
                        "phrase_payload": phrase_payload,
                        "slices": [
                            {key: value for key, value in item.items() if key != "audio"}
                            for item in phrase_slices
                        ],
                        "acceptable": all(item["acceptable"] for item in phrase_slices),
                    }
                )
                if phrase_slices and all(item["acceptable"] for item in phrase_slices):
                    selected_slices = phrase_slices
                    selected_phrase_payload = phrase_payload
                    break

            if selected_slices is None or selected_phrase_payload is None:
                skip_payload = {
                    "reason": "no_acceptable_countdown_phrase_slice_candidate",
                    "renderer": "countdown_phrase_slice",
                    "fallback_from": fallback_from,
                    "span_id": span_id,
                    "segment_ids": [segment.id for _index, segment, _values in span],
                    "values": values,
                    "tokens": tokens,
                    "compact_text": compact_text,
                    "phrase_candidates": phrase_candidates,
                }
                for _index, segment, _values in span:
                    segment.analysis["countdown_renderer_skip"] = skip_payload
                return set()

            span_audio = np.zeros((total_frames, 2), dtype=np.float32)
            placement_metadata: list[dict[str, Any]] = []
            for placement, selected in zip(placements, selected_slices, strict=True):
                slot_start = int(placement["slot_start_frame"])
                slot_end = int(placement["slot_end_frame"])
                slot_frames = max(1, slot_end - slot_start)
                audio = _countdown_stereo(np.array(selected["audio"], copy=True))
                placement_anchor = placement.get("placement_anchor")
                if placement_anchor in {"source_anchor_exact", "source_anchor_smoothed"}:
                    start_frame = max(0, min(slot_start, total_frames))
                elif placement_anchor == "source_word_start":
                    start_frame = min(slot_start, max(0, total_frames - len(audio)))
                else:
                    start_frame = slot_start + max(0, (slot_frames - len(audio)) // 2)
                end_frame = min(total_frames, start_frame + len(audio))
                source_frames = max(0, end_frame - start_frame)
                if source_frames:
                    span_audio[start_frame:end_frame] += audio[:source_frames]
                placement_metadata.append(
                    {
                        "segment_id": placement["segment_id"],
                        "value": placement["value"],
                        "text": placement["text"],
                        "token_index": placement["token_index"],
                        "slot_start_sec": round(slot_start / sample_rate, 6),
                        "slot_end_sec": round(slot_end / sample_rate, 6),
                        "slot_duration_sec": round(float(placement["slot_duration_sec"]), 6),
                        "placed_start_sec": round(start_frame / sample_rate, 6),
                        "placed_end_sec": round(end_frame / sample_rate, 6),
                        "selected_candidate_index": selected["candidate_index"],
                        "selected_duration_sec": round(float(selected["duration_sec"]), 6),
                        "selected_path": selected["output_path"],
                        "selected_pronunciation_gate": selected["pronunciation_gate"],
                        "selected_prosody_score": (
                            selected.get("countdown_prosody_qc") or {}
                        ).get("score"),
                        "placement_anchor": placement.get("placement_anchor", "slot_center"),
                    }
                )

            peak = float(np.max(np.abs(span_audio))) if span_audio.size else 0.0
            if peak > 0.98:
                span_audio *= 0.98 / peak
            span_path = span_dir / f"{span_id}.wav"
            write_audio(span_path, span_audio, sample_rate)
            span_metadata = {
                "span_id": span_id,
                "renderer": "countdown_phrase_slice",
                "fallback_from": fallback_from,
                "segment_ids": [segment.id for _index, segment, _values in span],
                "values": values,
                "tokens": tokens,
                "compact_text": compact_text,
                "target_duration_sec": round(total_duration_sec, 6),
                "span_path": str(span_path),
                "token_placements": placement_metadata,
                "phrase_payload": selected_phrase_payload,
                "phrase_candidates": phrase_candidates,
                "source_language": cfg.source_language,
                "target_language": cfg.target_language,
                "ref_style": resolved_ref_style,
            }
            metadata_path = span_dir / f"{span_id}.json"
            write_json_atomic(metadata_path, span_metadata)

            offset = 0
            for _segment_index, (_index, segment, segment_values) in enumerate(span):
                frames = segment_frames[segment.id]
                segment_audio = span_audio[offset : offset + frames]
                offset += frames
                if len(segment_audio) < frames:
                    padding = np.zeros((frames - len(segment_audio), 2), dtype=np.float32)
                    segment_audio = np.concatenate([segment_audio, padding], axis=0)
                elif len(segment_audio) > frames:
                    segment_audio = segment_audio[:frames]
                final_path = project_dir / "work" / "tts" / f"{segment.id}_final.wav"
                write_audio(final_path, segment_audio, sample_rate)
                final_duration = duration_sec(final_path)
                final_ratio = duration_ratio(final_duration, segment.duration)
                segment_tokens = _countdown_spoken_tokens(segment_values) or []
                segment_placements = [
                    placement
                    for placement in placement_metadata
                    if placement["segment_id"] == segment.id
                ]
                payload = {
                    "renderer": "countdown_phrase_slice",
                    "fallback_from": fallback_from,
                    "span_id": span_id,
                    "span_metadata_path": str(metadata_path),
                    "span_path": str(span_path),
                    "segment_id": segment.id,
                    "segment_values": segment_values,
                    "segment_tokens": segment_tokens,
                    "values": values,
                    "tokens": tokens,
                    "token_placements": segment_placements,
                    "target_duration_sec": segment.duration,
                    "duration_ratio": final_ratio,
                    "duration_gate": "pass",
                    "audio_qc": {
                        "gate": "pass",
                        "peak_dbfs": round(peak_dbfs(final_path), 3),
                        "rms_dbfs": round(rms_dbfs(final_path), 3),
                    },
                }
                candidate = TTSCandidate(
                    candidate_index=0,
                    seed=cfg.base_seed + 70_000 + first_index,
                    payload=payload,
                    output_path=str(final_path),
                    duration_sec=final_duration,
                    backend="gpt-sovits-countdown-renderer",
                    selected=True,
                    duration_ratio=final_ratio,
                    duration_gate="pass",
                    acceptable_for_mix=True,
                    selection_score=max(0.0, 1.0 - min(abs(final_ratio - 1.0), 1.0)),
                    selection_reason="countdown_phrase_slice",
                    retry_summary={"countdown_renderer": True, "span_id": span_id},
                )
                segment.tts = TTSMetadata(
                    backend="gpt-sovits-countdown-renderer",
                    ref_style=resolved_ref_style,
                    speed_factor=float(
                        getattr(cfg, "gsv_countdown_token_speed_factor", cfg.gsv_tts_max_speed_factor)
                    ),
                    candidate_count=1,
                    selected_candidate_path=str(final_path),
                    candidates=[candidate],
                    source_language=cfg.source_language,
                    target_language=cfg.target_language,
                    cross_lingual_voice_transfer=cfg.source_language != cfg.target_language,
                    retry_summary={
                        "countdown_renderer": True,
                        "countdown_renderer_mode": "phrase_slice",
                        "span_id": span_id,
                        "span_metadata_path": str(metadata_path),
                        "selected_duration_gate": "pass",
                        "selected_acceptable_for_mix": True,
                        "selected_duration_ratio": final_ratio,
                    },
                )
                segment.analysis["countdown_renderer"] = {
                    "renderer": "countdown_phrase_slice",
                    "fallback_from": fallback_from,
                    "span_id": span_id,
                    "segment_values": segment_values,
                    "segment_tokens": segment_tokens,
                    "token_placements": segment_placements,
                    "span_metadata_path": str(metadata_path),
                }
                segment.status = "synthesized"
            return {segment.id for _index, segment, _values in span}

        def render_countdown_span_compact(span: list[tuple[int, Segment, list[int]]]) -> set[str]:
            first_index, first_segment, _first_values = span[0]
            if not first_segment.script:
                return set()
            values = [value for _index, _segment, segment_values in span for value in segment_values]
            tokens = _countdown_spoken_tokens(values)
            if tokens is None:
                return set()
            start_gsv_servers()
            span_id = "countdown_" + "_".join(segment.id for _index, segment, _values in span)
            span_dir = project_dir / "work" / "tts" / "countdown"
            span_dir.mkdir(parents=True, exist_ok=True)
            ref_style = first_segment.script.ref_style
            resolved_ref_style = ref_style if ref_style in refs else "whisper_close"
            ref = _ref_for_tts_language(resolve_ref(refs, ref_style), first_segment.script.tts_language)
            total_duration_sec = sum(segment.duration for _index, segment, _values in span)
            sample_rate = cfg.mix_sample_rate
            total_frames = max(1, int(round(total_duration_sec * sample_rate)))
            compact_text = _countdown_phrase_tts_text(tokens)
            lane_index = _segment_lane_index(first_segment, first_index - 1, gsv_lane_count)
            with lane_locks[lane_index]:
                phrase_audio, phrase_payload = token_audio_for_countdown(
                    token_text=compact_text,
                    token_index=0,
                    token_count=len(tokens),
                    span_id=span_id,
                    ref=ref,
                    token_slot_sec=total_duration_sec,
                    lane_index=lane_index,
                    tts_text_language=_segment_tts_text_language(first_segment, cfg.target_language),
                    ref_style=resolved_ref_style,
                )
            if len(phrase_audio) > total_frames:
                skip_payload = {
                    "reason": "countdown_phrase_exceeds_span_after_tempo_cap",
                    "span_id": span_id,
                    "segment_ids": [segment.id for _index, segment, _values in span],
                    "values": values,
                    "tokens": tokens,
                    "compact_text": compact_text,
                    "target_duration_sec": round(total_duration_sec, 6),
                    "phrase_duration_sec": round(len(phrase_audio) / sample_rate, 6),
                    "phrase_payload": phrase_payload,
                }
                for _index, segment, _values in span:
                    segment.analysis["countdown_renderer_skip"] = skip_payload
                return set()

            span_audio = np.zeros((total_frames, 2), dtype=np.float32)
            start_frame = max(0, (total_frames - len(phrase_audio)) // 2)
            end_frame = min(total_frames, start_frame + len(phrase_audio))
            if end_frame > start_frame:
                span_audio[start_frame:end_frame] += phrase_audio[: end_frame - start_frame]
            peak = float(np.max(np.abs(span_audio))) if span_audio.size else 0.0
            if peak > 0.98:
                span_audio *= 0.98 / peak
            span_path = span_dir / f"{span_id}.wav"
            write_audio(span_path, span_audio, sample_rate)
            span_metadata = {
                "span_id": span_id,
                "renderer": "countdown_compact_phrase",
                "segment_ids": [segment.id for _index, segment, _values in span],
                "values": values,
                "tokens": tokens,
                "compact_text": compact_text,
                "target_duration_sec": round(total_duration_sec, 6),
                "span_path": str(span_path),
                "phrase_timeline": {
                    "start_sec": round(start_frame / sample_rate, 6),
                    "end_sec": round(end_frame / sample_rate, 6),
                },
                "phrase_payload": phrase_payload,
                "source_language": cfg.source_language,
                "target_language": cfg.target_language,
                "ref_style": resolved_ref_style,
            }
            metadata_path = span_dir / f"{span_id}.json"
            write_json_atomic(metadata_path, span_metadata)

            offset = 0
            for segment_index, (_index, segment, segment_values) in enumerate(span):
                segment_frames = max(1, int(round(segment.duration * sample_rate)))
                if segment_index == len(span) - 1:
                    segment_audio = span_audio[offset:]
                else:
                    segment_audio = span_audio[offset : offset + segment_frames]
                offset += segment_frames
                if len(segment_audio) < segment_frames:
                    padding = np.zeros((segment_frames - len(segment_audio), 2), dtype=np.float32)
                    segment_audio = np.concatenate([segment_audio, padding], axis=0)
                elif len(segment_audio) > segment_frames:
                    segment_audio = segment_audio[:segment_frames]
                final_path = project_dir / "work" / "tts" / f"{segment.id}_final.wav"
                write_audio(final_path, segment_audio, sample_rate)
                final_duration = duration_sec(final_path)
                final_ratio = duration_ratio(final_duration, segment.duration)
                segment_tokens = _countdown_spoken_tokens(segment_values) or []
                payload = {
                    "renderer": "countdown_compact_phrase",
                    "span_id": span_id,
                    "span_metadata_path": str(metadata_path),
                    "span_path": str(span_path),
                    "segment_id": segment.id,
                    "segment_values": segment_values,
                    "segment_tokens": segment_tokens,
                    "values": values,
                    "tokens": tokens,
                    "target_duration_sec": segment.duration,
                    "duration_ratio": final_ratio,
                    "duration_gate": "pass",
                    "audio_qc": {
                        "gate": "pass",
                        "peak_dbfs": round(peak_dbfs(final_path), 3),
                        "rms_dbfs": round(rms_dbfs(final_path), 3),
                    },
                }
                candidate = TTSCandidate(
                    candidate_index=0,
                    seed=cfg.base_seed + 70_000 + first_index,
                    payload=payload,
                    output_path=str(final_path),
                    duration_sec=final_duration,
                    backend="gpt-sovits-countdown-renderer",
                    selected=True,
                    duration_ratio=final_ratio,
                    duration_gate="pass",
                    acceptable_for_mix=True,
                    selection_score=max(0.0, 1.0 - min(abs(final_ratio - 1.0), 1.0)),
                    selection_reason="countdown_compact_phrase",
                    retry_summary={"countdown_renderer": True, "span_id": span_id},
                )
                segment.tts = TTSMetadata(
                    backend="gpt-sovits-countdown-renderer",
                    ref_style=resolved_ref_style,
                    speed_factor=float(getattr(cfg, "gsv_countdown_token_speed_factor", cfg.gsv_tts_max_speed_factor)),
                    candidate_count=1,
                    selected_candidate_path=str(final_path),
                    candidates=[candidate],
                    source_language=cfg.source_language,
                    target_language=cfg.target_language,
                    cross_lingual_voice_transfer=cfg.source_language != cfg.target_language,
                    retry_summary={
                        "countdown_renderer": True,
                        "span_id": span_id,
                        "span_metadata_path": str(metadata_path),
                        "selected_duration_gate": "pass",
                        "selected_acceptable_for_mix": True,
                        "selected_duration_ratio": final_ratio,
                    },
                )
                segment.analysis["countdown_renderer"] = {
                    "span_id": span_id,
                    "segment_values": segment_values,
                    "segment_tokens": segment_tokens,
                    "span_metadata_path": str(metadata_path),
                }
                segment.status = "synthesized"
            return {segment.id for _index, segment, _values in span}

        def embedded_countdown_hybrid_enabled() -> bool:
            return bool(getattr(cfg, "gsv_countdown_hybrid_enabled", True))

        def embedded_countdown_ref_for_segment(
            segment: Segment,
            *,
            tts_text_language: str,
        ) -> tuple[str, GPTSoVITSRef]:
            ref_style = segment.script.ref_style if segment.script else "whisper_close"
            resolved_ref_style = ref_style if ref_style in refs else "whisper_close"
            return resolved_ref_style, _ref_for_tts_language(
                resolve_ref(refs, resolved_ref_style),
                tts_text_language,
            )

        def select_embedded_countdown_candidate(
            candidates: list[dict[str, Any]],
        ) -> tuple[dict[str, Any] | None, str]:
            if not candidates:
                return None, "no_countdown_token_candidate"
            ranked = sorted(
                candidates,
                key=lambda candidate: (
                    bool(candidate.get("acceptable")),
                    countdown_carrier_quality_tier_rank(candidate.get("quality_tier")),
                    float(candidate.get("selection_score") or 0.0),
                    -float(candidate.get("duration_sec") or 0.0),
                ),
                reverse=True,
            )
            selected = ranked[0]
            reason = (
                "acceptable"
                if bool(selected.get("acceptable"))
                else "best_available_nonacceptable"
            )
            return selected, reason

        def render_embedded_countdown_hybrid_bed(
            *,
            index: int,
            segment: Segment,
            lane_index: int,
            values: list[int],
        ) -> bool:
            tokens = _countdown_spoken_tokens(values)
            if tokens is None:
                return False
            if countdown_source_anchor_enabled():
                countdown_source_anchor_for_segment(segment, values)
            anchors, clusters, anchor_source_kind = _embedded_countdown_anchor_rows(
                segment,
                values,
                smoothing_blend=float(
                    getattr(cfg, "gsv_countdown_source_anchor_smoothing_blend", 0.70)
                ),
                cluster_gap_sec=float(
                    getattr(cfg, "gsv_countdown_source_anchor_cluster_gap_sec", 2.6)
                ),
            )
            if not anchors or len(anchors) != len(tokens):
                segment.analysis["embedded_countdown_hybrid_renderer"] = {
                    "status": "skipped",
                    "reason": "missing_source_anchors",
                    "values": values,
                    "tokens": tokens,
                }
                return False
            start_gsv_servers()
            sample_rate = cfg.mix_sample_rate
            total_frames = max(1, int(round(float(segment.duration) * sample_rate)))
            bed_audio = np.zeros((total_frames, 2), dtype=np.float32)
            span_id = f"embedded_countdown_{segment.id}"
            tts_text_language = _segment_tts_text_language(segment, cfg.target_language)
            ref_style, ref = embedded_countdown_ref_for_segment(
                segment,
                tts_text_language=tts_text_language,
            )
            span = [(index, segment, values)]
            ref_pool = countdown_token_bank_ref_pool(
                span=span,
                primary_ref_style=ref_style,
                primary_ref=ref,
                tts_text_language=tts_text_language,
            )
            warmup_metadata = countdown_token_bank_warmup_for_span(
                span_id=span_id,
                tokens=tokens,
                segment=segment,
                ref_pool=ref_pool,
                lane_index=lane_index,
                tts_text_language=tts_text_language,
            )
            placements: list[dict[str, Any]] = []
            failed_tokens: list[dict[str, Any]] = []
            retimed_token_count = 0
            with lane_locks[lane_index]:
                for token_index, (token_text, anchor) in enumerate(
                    zip(tokens, anchors, strict=True)
                ):
                    target_duration_sec = _countdown_hybrid_target_token_sec(
                        anchors,
                        token_index,
                        float(segment.duration),
                        gap_fill_ratio=float(
                            getattr(cfg, "gsv_countdown_hybrid_token_gap_fill_ratio", 0.72)
                        ),
                        max_token_sec=float(
                            getattr(cfg, "gsv_countdown_hybrid_token_max_sec", 0.58)
                        ),
                    )
                    candidates = countdown_carrier_candidates_for_slot(
                        token_text=token_text,
                        token_slot_sec=target_duration_sec,
                        segment=segment,
                        ref=ref,
                        ref_style=ref_style,
                        lane_index=lane_index,
                        tts_text_language=tts_text_language,
                        token_bank_ref_pool=ref_pool,
                        source_prosody=None,
                    )
                    selected, selection_reason = select_embedded_countdown_candidate(candidates)
                    if selected is None:
                        try:
                            direct_candidate = countdown_token_candidate_audio(
                                token_text=token_text,
                                token_index=token_index,
                                candidate_index=0,
                                span_id=span_id,
                                segment=segment,
                                ref=ref,
                                token_slot_sec=target_duration_sec,
                                lane_index=lane_index,
                                tts_text_language=tts_text_language,
                                ref_style=ref_style,
                                source_prosody=None,
                            )
                        except Exception as exc:
                            direct_candidate = None
                            selection_reason = f"{selection_reason}:direct_token_fallback_failed:{exc}"
                        if direct_candidate is not None:
                            selected = direct_candidate
                            selection_reason = (
                                "direct_token_fallback"
                                if bool(direct_candidate.get("acceptable"))
                                else "direct_token_fallback_nonacceptable"
                            )
                    if selected is None:
                        failed_tokens.append(
                            {
                                "token_index": token_index,
                                "text": token_text,
                                "reason": selection_reason,
                            }
                        )
                        continue
                    audio = _countdown_stereo(np.array(selected["audio"], copy=True))
                    trim_metadata = {"enabled": False, "applied": False}
                    if bool(getattr(cfg, "gsv_countdown_hybrid_active_trim_enabled", True)):
                        audio, trim_metadata = _countdown_trim_audio_to_active_region(
                            audio,
                            sample_rate,
                            keep_before_sec=float(
                                getattr(
                                    cfg,
                                    "gsv_countdown_hybrid_active_trim_keep_before_sec",
                                    0.012,
                                )
                            ),
                            keep_after_sec=float(
                                getattr(
                                    cfg,
                                    "gsv_countdown_hybrid_active_trim_keep_after_sec",
                                    0.045,
                                )
                            ),
                        )
                        trim_metadata["enabled"] = True
                    target_frames = max(1, int(round(target_duration_sec * sample_rate)))
                    original_frames = len(audio)
                    retime_ratio = 1.0
                    if len(audio) > target_frames:
                        audio = _countdown_retime_audio_to_frames(audio, target_frames)
                        retime_ratio = original_frames / target_frames
                        retimed_token_count += 1
                    audio = _countdown_fade_chunk_edges(audio, sample_rate, fade_sec=0.012)
                    active_fit = _countdown_active_anchor_fit_from_audio(audio, sample_rate)
                    leading_frames = max(
                        0,
                        int(round(float(active_fit.get("leading_silence_sec") or 0.0) * sample_rate)),
                    )
                    anchor_frame = int(round(float(anchor["start"]) * sample_rate))
                    start_frame = _countdown_anchor_aligned_start_frame(
                        slot_start_frame=anchor_frame,
                        active_leading_frames=leading_frames,
                        total_frames=total_frames,
                    )
                    end_frame = min(total_frames, start_frame + len(audio))
                    copied_frames = max(0, end_frame - start_frame)
                    if copied_frames:
                        bed_audio[start_frame:end_frame] += audio[:copied_frames]
                    placed_active_start_sec = (start_frame + leading_frames) / sample_rate
                    placements.append(
                        {
                            "segment_id": segment.id,
                            "value": int(anchor["value"]),
                            "text": token_text,
                            "token_index": token_index,
                            "source_anchor_sec": round(float(anchor["start"]), 6),
                            "placed_start_sec": round(start_frame / sample_rate, 6),
                            "placed_end_sec": round(end_frame / sample_rate, 6),
                            "placed_active_start_sec": round(placed_active_start_sec, 6),
                            "active_anchor_delta_sec": round(
                                placed_active_start_sec - float(anchor["start"]),
                                6,
                            ),
                            "target_duration_sec": round(target_duration_sec, 6),
                            "selected_duration_sec": round(float(selected["duration_sec"]), 6),
                            "selected_path": selected["output_path"],
                            "selected_quality_tier": selected.get("quality_tier"),
                            "selected_acceptable": bool(selected.get("acceptable")),
                            "selection_reason": selection_reason,
                            "retime_ratio": round(retime_ratio, 6),
                            "trim": trim_metadata,
                            "active_anchor_fit": active_fit,
                        }
                    )
            if failed_tokens:
                segment.analysis["embedded_countdown_hybrid_renderer"] = {
                    "status": "skipped",
                    "reason": "missing_token_candidates",
                    "values": values,
                    "tokens": tokens,
                    "failed_tokens": failed_tokens,
                    "token_bank": warmup_metadata,
                }
                return False
            peak = float(np.max(np.abs(bed_audio))) if bed_audio.size else 0.0
            if peak > 0.98:
                bed_audio *= 0.98 / peak
            output_dir = project_dir / "work" / "tts" / "countdown" / "hybrid"
            output_dir.mkdir(parents=True, exist_ok=True)
            bed_path = output_dir / f"{segment.id}_countdown_bed.wav"
            write_audio(bed_path, bed_audio, sample_rate)
            metadata = {
                "status": "rendered",
                "renderer": "embedded_countdown_hybrid",
                "span_id": span_id,
                "segment_id": segment.id,
                "values": values,
                "tokens": tokens,
                "bed_path": str(bed_path),
                "target_duration_sec": round(float(segment.duration), 6),
                "anchors": anchors,
                "anchor_source_kind": anchor_source_kind,
                "clusters": clusters,
                "token_placements": placements,
                "token_bank": warmup_metadata,
                "retimed_token_count": retimed_token_count,
                "audio_qc": {
                    "gate": "pass",
                    "peak_dbfs": round(peak_dbfs(bed_path), 3),
                    "rms_dbfs": round(rms_dbfs(bed_path), 3),
                },
                "policy": {
                    "token_gap_fill_ratio": float(
                        getattr(cfg, "gsv_countdown_hybrid_token_gap_fill_ratio", 0.72)
                    ),
                    "max_token_duration_sec": float(
                        getattr(cfg, "gsv_countdown_hybrid_token_max_sec", 0.58)
                    ),
                    "active_trim_enabled": bool(
                        getattr(cfg, "gsv_countdown_hybrid_active_trim_enabled", True)
                    ),
                },
            }
            metadata_path = output_dir / f"{segment.id}_countdown_bed.json"
            write_json_atomic(metadata_path, metadata)
            metadata["metadata_path"] = str(metadata_path)
            segment.analysis["embedded_countdown_hybrid_renderer"] = metadata
            return True

        def render_embedded_countdown_hybrid_beds(
            segment_jobs: list[tuple[int, Segment, int]],
        ) -> set[str]:
            if not embedded_countdown_hybrid_enabled():
                return set()
            rendered: set[str] = set()
            for index, segment, lane_index in segment_jobs:
                values = _embedded_countdown_values(segment)
                if values is None:
                    continue
                if (
                    not countdown_only
                    and segment.status in {*SKIP_STATUSES, "synthesized"}
                    and not should_reset_previous_tts(segment)
                ):
                    continue
                if render_embedded_countdown_hybrid_bed(
                    index=index,
                    segment=segment,
                    lane_index=lane_index,
                    values=values,
                ):
                    rendered.add(segment.id)
                    save_manifest(project_dir, manifest)
            if rendered:
                _log_stage_checkpoint(
                    stage_name,
                    "embedded countdown hybrid",
                    f"segments={len(rendered)}",
                )
            return rendered

        def apply_embedded_countdown_hybrid_overlay(
            *,
            segment: Segment,
            selected: TTSCandidate,
            final_path: Path,
            candidates: list[TTSCandidate],
        ) -> TTSCandidate:
            if not bool(getattr(cfg, "gsv_countdown_hybrid_apply_to_synth", True)):
                return selected
            metadata = segment.analysis.get("embedded_countdown_hybrid_renderer")
            if not isinstance(metadata, dict) or metadata.get("status") != "rendered":
                return selected
            bed_path = Path(str(metadata.get("bed_path") or ""))
            if not bed_path.exists():
                return selected
            base_audio, base_rate = load_audio(final_path)
            bed_audio, bed_rate = load_audio(bed_path)
            base_audio = _countdown_stereo(base_audio)
            bed_audio = _countdown_stereo(bed_audio)
            if base_rate != cfg.mix_sample_rate:
                base_audio = _countdown_stereo(
                    resample_linear(base_audio, base_rate, cfg.mix_sample_rate)
                )
                base_rate = cfg.mix_sample_rate
            if bed_rate != base_rate:
                bed_audio = _countdown_stereo(resample_linear(bed_audio, bed_rate, base_rate))
            target_frames = max(1, int(round(float(segment.duration) * base_rate)))
            base_original_frames = len(base_audio)
            bed_original_frames = len(bed_audio)
            if len(base_audio) < target_frames:
                base_audio = np.concatenate(
                    [
                        base_audio,
                        np.zeros((target_frames - len(base_audio), 2), dtype=np.float32),
                    ],
                    axis=0,
                )
            elif len(base_audio) > target_frames:
                base_audio = base_audio[:target_frames]
            if len(bed_audio) < target_frames:
                bed_audio = np.concatenate(
                    [
                        bed_audio,
                        np.zeros((target_frames - len(bed_audio), 2), dtype=np.float32),
                    ],
                    axis=0,
                )
            elif len(bed_audio) > target_frames:
                bed_audio = bed_audio[:target_frames]
            duck_regions: list[dict[str, Any]] = []
            if bool(getattr(cfg, "gsv_countdown_hybrid_base_duck_enabled", True)):
                duck_gain = float(getattr(cfg, "gsv_countdown_hybrid_base_duck_gain", 0.18))
                duck_pad_sec = float(getattr(cfg, "gsv_countdown_hybrid_base_duck_pad_sec", 0.055))
                duck_fade_sec = float(getattr(cfg, "gsv_countdown_hybrid_base_duck_fade_sec", 0.025))
                fade_frames = max(0, int(round(duck_fade_sec * base_rate)))
                envelope = np.ones((target_frames, 1), dtype=np.float32)
                for placement in metadata.get("token_placements", []):
                    if not isinstance(placement, dict):
                        continue
                    try:
                        start_sec = float(
                            placement.get("placed_start_sec")
                            if placement.get("placed_start_sec") is not None
                            else placement.get("source_anchor_sec")
                            or 0.0
                        )
                        end_sec = float(
                            placement.get("placed_end_sec")
                            if placement.get("placed_end_sec") is not None
                            else start_sec + float(placement.get("target_duration_sec") or 0.0)
                        )
                    except (TypeError, ValueError):
                        continue
                    start_frame = max(0, int(round((start_sec - duck_pad_sec) * base_rate)))
                    end_frame = min(target_frames, int(round((end_sec + duck_pad_sec) * base_rate)))
                    if end_frame <= start_frame:
                        continue
                    duck_regions.append(
                        {
                            "token_index": placement.get("token_index"),
                            "text": placement.get("text"),
                            "start_sec": round(start_frame / base_rate, 6),
                            "end_sec": round(end_frame / base_rate, 6),
                            "gain": duck_gain,
                        }
                    )
                    solid_start = min(target_frames, start_frame + fade_frames)
                    solid_end = max(start_frame, end_frame - fade_frames)
                    if fade_frames and solid_start > start_frame:
                        ramp = np.linspace(1.0, duck_gain, solid_start - start_frame, dtype=np.float32)
                        envelope[start_frame:solid_start, 0] = np.minimum(
                            envelope[start_frame:solid_start, 0],
                            ramp,
                        )
                    if solid_end > solid_start:
                        envelope[solid_start:solid_end, 0] = np.minimum(
                            envelope[solid_start:solid_end, 0],
                            duck_gain,
                        )
                    if fade_frames and end_frame > solid_end:
                        ramp = np.linspace(duck_gain, 1.0, end_frame - solid_end, dtype=np.float32)
                        envelope[solid_end:end_frame, 0] = np.minimum(
                            envelope[solid_end:end_frame, 0],
                            ramp,
                        )
                    elif end_frame > solid_start:
                        envelope[solid_start:end_frame, 0] = np.minimum(
                            envelope[solid_start:end_frame, 0],
                            duck_gain,
                        )
                base_audio = base_audio * envelope
            base_gain = float(getattr(cfg, "gsv_countdown_hybrid_base_gain", 0.55))
            bed_gain = float(getattr(cfg, "gsv_countdown_hybrid_bed_gain", 0.85))
            mixed = base_audio[:target_frames] * base_gain + bed_audio[:target_frames] * bed_gain
            peak = float(np.max(np.abs(mixed))) if mixed.size else 0.0
            if peak > 0.98:
                mixed *= 0.98 / peak
            hybrid_path = (
                project_dir
                / "work"
                / "tts"
                / "candidates"
                / f"{segment.id}_embedded_countdown_hybrid.wav"
            )
            write_audio(hybrid_path, mixed, base_rate)
            shutil.copy2(hybrid_path, final_path)
            final_duration = duration_sec(final_path)
            final_ratio = duration_ratio(final_duration, segment.duration)
            selected.selected = False
            payload = copy.deepcopy(selected.payload)
            payload["embedded_countdown_hybrid"] = {
                "renderer": "embedded_countdown_hybrid",
                "bed_path": str(bed_path),
                "metadata_path": metadata.get("metadata_path"),
                "base_candidate_path": selected.output_path,
                "base_gain": base_gain,
                "bed_gain": bed_gain,
                "base_duration_before_overlay_sec": round(base_original_frames / base_rate, 6),
                "bed_duration_before_overlay_sec": round(bed_original_frames / base_rate, 6),
                "target_duration_sec": round(float(segment.duration), 6),
                "base_clipped_to_target": base_original_frames > target_frames,
                "bed_clipped_to_target": bed_original_frames > target_frames,
                "base_ducking": {
                    "enabled": bool(
                        getattr(cfg, "gsv_countdown_hybrid_base_duck_enabled", True)
                    ),
                    "regions": duck_regions,
                    "gain": float(getattr(cfg, "gsv_countdown_hybrid_base_duck_gain", 0.18)),
                },
                "token_placements": metadata.get("token_placements", []),
                "audio_qc": {
                    "gate": "pass",
                    "peak_dbfs": round(peak_dbfs(final_path), 3),
                    "rms_dbfs": round(rms_dbfs(final_path), 3),
                },
            }
            hybrid_candidate = TTSCandidate(
                candidate_index=max((candidate.candidate_index for candidate in candidates), default=-1) + 1,
                seed=selected.seed,
                payload=payload,
                output_path=str(hybrid_path),
                duration_sec=final_duration,
                backend=selected.backend,
                selected=True,
                duration_ratio=final_ratio,
                duration_gate="pass",
                acceptable_for_mix=True,
                selection_score=selected.selection_score,
                selection_reason="embedded_countdown_hybrid_overlay",
                retry_summary={
                    **selected.retry_summary,
                    "embedded_countdown_hybrid": True,
                },
            )
            candidates.append(hybrid_candidate)
            segment.analysis["embedded_countdown_hybrid_renderer"] = {
                **metadata,
                "applied_to_synth": True,
                "hybrid_candidate_path": str(hybrid_path),
                "selected_candidate_path": str(final_path),
            }
            return hybrid_candidate

        def mark_countdown_span_manual_review(
            span: list[tuple[int, Segment, list[int]]],
            errors: list[str],
        ) -> set[str]:
            for _index, segment, _values in span:
                segment.status = "needs_manual_review"
                for error in errors:
                    if error not in segment.errors:
                        segment.errors.append(error)
            return {segment.id for _index, segment, _values in span}

        def render_countdown_span(span: list[tuple[int, Segment, list[int]]]) -> set[str]:
            renderer = getattr(cfg, "gsv_countdown_renderer", "chunk_bank")
            if renderer == "carrier_bank":
                rendered = render_countdown_span_carrier_bank(span)
                if rendered:
                    return rendered
                fallback = getattr(cfg, "gsv_countdown_fallback_renderer", "compact")
                if fallback == "phrase_slice":
                    rendered = render_countdown_span_phrase_slice(
                        span,
                        fallback_from="countdown_carrier_bank",
                    )
                    if rendered:
                        return rendered
                    return mark_countdown_span_manual_review(
                        span,
                        [
                            "Countdown carrier bank renderer failed.",
                            "Countdown phrase slice fallback failed.",
                        ],
                    )
                if fallback == "compact":
                    return render_countdown_span_compact(span)
                if fallback == "manual_review":
                    return mark_countdown_span_manual_review(
                        span,
                        ["Countdown carrier bank renderer failed."],
                    )
                return set()
            if renderer == "chunk_bank":
                rendered = render_countdown_span_chunk_bank(span)
                if rendered:
                    return rendered
                fallback = getattr(cfg, "gsv_countdown_fallback_renderer", "compact")
                if fallback == "phrase_slice":
                    rendered = render_countdown_span_phrase_slice(
                        span,
                        fallback_from="countdown_chunk_bank",
                    )
                    if rendered:
                        return rendered
                    return mark_countdown_span_manual_review(
                        span,
                        [
                            "Countdown chunk bank renderer failed.",
                            "Countdown phrase slice fallback failed.",
                        ],
                    )
                if fallback == "compact":
                    return render_countdown_span_compact(span)
                if fallback == "manual_review":
                    return mark_countdown_span_manual_review(
                        span,
                        ["Countdown chunk bank renderer failed."],
                    )
                return set()
            if renderer == "canonical_pack":
                rendered = render_countdown_span_canonical_pack(span)
                if rendered:
                    return rendered
                fallback = getattr(cfg, "gsv_countdown_fallback_renderer", "compact")
                if fallback == "phrase_slice":
                    rendered = render_countdown_span_phrase_slice(
                        span,
                        fallback_from="countdown_canonical_pack",
                    )
                    if rendered:
                        return rendered
                    return mark_countdown_span_manual_review(
                        span,
                        [
                            "Countdown canonical pack renderer failed.",
                            "Countdown phrase slice fallback failed.",
                        ],
                    )
                if fallback == "compact":
                    return render_countdown_span_compact(span)
                if fallback == "manual_review":
                    return mark_countdown_span_manual_review(
                        span,
                        ["Countdown canonical pack renderer failed."],
                    )
                return set()
            if renderer == "compact":
                return render_countdown_span_compact(span)
            rendered = render_countdown_span_token(span)
            if rendered:
                return rendered
            fallback = getattr(cfg, "gsv_countdown_fallback_renderer", "compact")
            if fallback == "phrase_slice":
                rendered = render_countdown_span_phrase_slice(
                    span,
                    fallback_from="countdown_token_timeline",
                )
                if rendered:
                    return rendered
                return mark_countdown_span_manual_review(
                    span,
                    [
                        "Countdown token renderer failed.",
                        "Countdown phrase slice fallback failed.",
                    ],
                )
            if fallback == "compact":
                return render_countdown_span_compact(span)
            if fallback == "manual_review":
                return mark_countdown_span_manual_review(
                    span,
                    ["Countdown token renderer failed."],
                )
            return set()

        def render_countdown_spans(segment_jobs: list[tuple[int, Segment, int]]) -> set[str]:
            nonlocal last_logged_at
            spans = countdown_spans_for_jobs(segment_jobs)
            if not spans:
                return set()
            if str(getattr(cfg, "gsv_countdown_renderer", "chunk_bank")) == "numeric_phrase":
                return set()
            precompute_countdown_source_anchors(spans)
            save_manifest(project_dir, manifest)
            start_gsv_servers()
            rendered_segment_ids: set[str] = set()
            span_total = len(spans)
            completed_spans = 0
            if not mock and gsv_lane_count > 1 and len(spans) > 1:
                with ThreadPoolExecutor(max_workers=gsv_lane_count) as executor:
                    futures = {
                        executor.submit(render_countdown_span, span): span for span in spans
                    }
                    for future in as_completed(futures):
                        span = futures[future]
                        rendered_segment_ids.update(future.result())
                        completed_spans += 1
                        save_manifest(project_dir, manifest)
                        first_index, first_segment, _values = span[0]
                        last_logged_at = _log_segment_progress(
                            stage_name,
                            first_index,
                            span_total,
                            first_segment,
                            manifest,
                            started_at,
                            last_logged_at,
                            progress_index=completed_spans,
                            progress_label="spans",
                            counts_label="status_counts",
                        )
            else:
                for span in spans:
                    rendered_segment_ids.update(render_countdown_span(span))
                    completed_spans += 1
                    save_manifest(project_dir, manifest)
                    first_index, first_segment, _values = span[0]
                    last_logged_at = _log_segment_progress(
                        stage_name,
                        first_index,
                        span_total,
                        first_segment,
                        manifest,
                        started_at,
                        last_logged_at,
                        progress_index=completed_spans,
                        progress_label="spans",
                        counts_label="status_counts",
                    )
            return rendered_segment_ids

        def numeric_phrase_failure_metadata(
            *,
            segment: Segment,
            plan: NumericRenderPlan,
            reason: str,
            fallback: str,
            result: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            result_payload = (
                result.get("payload")
                if isinstance(result, dict) and isinstance(result.get("payload"), dict)
                else {}
            )
            request_payload = _numeric_phrase_request_payload(plan, result_payload)
            metadata = {
                "renderer": "numeric_phrase",
                "status": "failed",
                "reason": reason,
                "fallback": fallback,
                **_numeric_render_plan_payload(plan),
                "request": request_payload,
                "candidate_text": str(request_payload.get("text") or plan.text),
                "candidate_generation": _numeric_phrase_candidate_generation_payload(
                    request_payload,
                    result_payload,
                ),
                "numeric_qc": _numeric_phrase_numeric_qc_payload(plan, result_payload),
                "placements": _numeric_phrase_placements_payload(result_payload),
                "target_duration_sec": round(float(segment.duration), 6),
                "max_tempo_limit": float(getattr(cfg, "gsv_numeric_phrase_max_tempo", 1.1)),
                "whole_lead_in_sec": float(
                    getattr(cfg, "gsv_numeric_phrase_whole_lead_in_sec", 0.12)
                ),
                "tail_guard_sec": float(getattr(cfg, "gsv_numeric_phrase_tail_guard_sec", 0.16)),
            }
            if result:
                metadata["result"] = {
                    key: copy.deepcopy(value)
                    for key, value in result.items()
                    if key not in {"candidate", "payload"}
                }
            return metadata

        def mark_numeric_phrase_manual_review(
            *,
            segment: Segment,
            plan: NumericRenderPlan,
            reason: str,
            result: dict[str, Any] | None = None,
        ) -> bool:
            metadata = numeric_phrase_failure_metadata(
                segment=segment,
                plan=plan,
                reason=reason,
                fallback="manual_review",
                result=result,
            )
            segment.analysis["numeric_phrase_renderer"] = metadata
            seed = int(getattr(cfg, "base_seed", 0))
            candidate = TTSCandidate(
                candidate_index=0,
                seed=seed,
                payload=copy.deepcopy(metadata),
                output_path=str(
                    project_dir
                    / "work"
                    / "tts"
                    / "numeric_phrase"
                    / f"{segment.id}_numeric_phrase_failed.wav"
                ),
                backend="gpt-sovits-countdown-renderer",
                error=reason,
                selection_reason="numeric_phrase_renderer_failed",
                retry_summary={"numeric_phrase_renderer": True},
            )
            segment.tts = TTSMetadata(
                backend="gpt-sovits-countdown-renderer",
                ref_style=segment.script.ref_style if segment.script else "whisper_close",
                candidate_count=1,
                candidates=[candidate],
                source_language=cfg.source_language,
                target_language=cfg.target_language,
                cross_lingual_voice_transfer=cfg.source_language != cfg.target_language,
                retry_summary={
                    "numeric_phrase_renderer": True,
                    "status": "failed",
                    "fallback": "manual_review",
                    "reason": reason,
                },
            )
            segment.status = "needs_manual_review"
            error = f"Numeric phrase renderer failed: {reason}"
            if error not in segment.errors:
                segment.errors.append(error)
            numeric_phrase_failed_segment_ids.add(segment.id)
            numeric_phrase_handled_segment_ids.add(segment.id)
            return True

        def numeric_phrase_source_path_from_result(
            result: dict[str, Any],
        ) -> Path | None:
            candidate = result.get("candidate")
            raw_path = (
                candidate.output_path
                if isinstance(candidate, TTSCandidate)
                else result.get("output_path")
            )
            if not raw_path:
                return None
            path = Path(str(raw_path))
            if not path.is_absolute():
                path = project_dir / path
            return path

        def promote_numeric_phrase_rendered_result(
            *,
            index: int,
            segment: Segment,
            plan: NumericRenderPlan,
            result: dict[str, Any],
        ) -> bool:
            source_path = numeric_phrase_source_path_from_result(result)
            if source_path is None or not source_path.exists():
                return mark_numeric_phrase_manual_review(
                    segment=segment,
                    plan=plan,
                    reason="numeric_phrase_rendered_output_missing",
                    result=result,
                )
            final_path = project_dir / "work" / "tts" / f"{segment.id}_final.wav"
            final_path.parent.mkdir(parents=True, exist_ok=True)
            if source_path.resolve() != final_path.resolve():
                ensure_not_same_path(source_path, final_path)
                shutil.copy2(source_path, final_path)
            actual_duration = duration_sec(final_path)
            candidate_ratio = duration_ratio(actual_duration, segment.duration)
            duration_gate = (
                "too_long"
                if duration_too_long(actual_duration, segment.duration, cfg.duration_tolerance)
                else "too_short"
                if duration_too_short(actual_duration, segment.duration, cfg.duration_tolerance)
                else "pass"
            )
            peak = peak_dbfs(final_path)
            rms = rms_dbfs(final_path)
            audio_gate = "silent" if peak <= -90.0 or rms <= -90.0 else "pass"
            source_candidate = result.get("candidate")
            if isinstance(source_candidate, TTSCandidate):
                seed = source_candidate.seed
                payload = copy.deepcopy(source_candidate.payload)
                selection_reason = source_candidate.selection_reason or "numeric_phrase_renderer"
            else:
                seed = int(result.get("seed") or (int(getattr(cfg, "base_seed", 0)) + index))
                payload = copy.deepcopy(result.get("payload") or {})
                selection_reason = str(result.get("selection_reason") or "numeric_phrase_renderer")
            plan_payload = _numeric_render_plan_payload(plan)
            try:
                max_tempo = float(
                    result.get("max_tempo")
                    or payload.get("max_tempo")
                    or payload.get("numeric_phrase", {}).get("max_tempo")
                    or 1.0
                )
            except (TypeError, ValueError):
                max_tempo = 1.0
            max_tempo_limit = float(getattr(cfg, "gsv_numeric_phrase_max_tempo", 1.1))
            request_payload = _numeric_phrase_request_payload(plan, payload)
            candidate_generation = _numeric_phrase_candidate_generation_payload(
                request_payload,
                payload,
            )
            placements = _numeric_phrase_placements_payload(payload)
            numeric_qc = _numeric_phrase_numeric_qc_payload(plan, payload)
            payload.update(
                {
                    "renderer": "numeric_phrase",
                    "plan": plan_payload,
                    "request": request_payload,
                    "candidate_text": str(request_payload.get("text") or plan.text),
                    "candidate_generation": candidate_generation,
                    "values": list(plan.values),
                    "tokens": list(plan.tokens),
                    "text_variant": plan.text_variant,
                    "render_policy": plan.render_policy,
                    "target_duration_sec": round(float(segment.duration), 6),
                    "duration_ratio": candidate_ratio,
                    "duration_gate": duration_gate,
                    "timing_quality": _gsv_timing_quality_payload(
                        candidate_ratio,
                        duration_gate,
                        float(getattr(cfg, "gsv_timing_quality_tolerance", 0.10)),
                        float(cfg.duration_tolerance),
                    ),
                    "audio_qc": {
                        "gate": audio_gate,
                        "peak_dbfs": round(peak, 3),
                        "rms_dbfs": round(rms, 3),
                    },
                    "numeric_qc": numeric_qc,
                    "placements": placements,
                    "numeric_phrase": {
                        **copy.deepcopy(payload.get("numeric_phrase") or {}),
                        **plan_payload,
                        "max_tempo": round(max_tempo, 6),
                        "max_tempo_limit": max_tempo_limit,
                        "whole_lead_in_sec": float(
                            getattr(cfg, "gsv_numeric_phrase_whole_lead_in_sec", 0.12)
                        ),
                        "tail_guard_sec": float(
                            getattr(cfg, "gsv_numeric_phrase_tail_guard_sec", 0.16)
                        ),
                        "numeric_qc": numeric_qc,
                        "placements": placements,
                    },
                }
            )
            if duration_gate != "pass":
                return mark_numeric_phrase_manual_review(
                    segment=segment,
                    plan=plan,
                    reason=f"duration_gate_{duration_gate}",
                    result={**result, "duration_gate": duration_gate},
                )
            if audio_gate != "pass":
                return mark_numeric_phrase_manual_review(
                    segment=segment,
                    plan=plan,
                    reason=f"audio_gate_{audio_gate}",
                    result={**result, "audio_gate": audio_gate},
                )
            if max_tempo > max_tempo_limit:
                return mark_numeric_phrase_manual_review(
                    segment=segment,
                    plan=plan,
                    reason=f"max_tempo_exceeded:{max_tempo:.3f}>{max_tempo_limit:.3f}",
                    result={**result, "max_tempo": max_tempo},
                )
            candidate = TTSCandidate(
                candidate_index=0,
                seed=seed,
                payload=payload,
                output_path=str(final_path),
                duration_sec=actual_duration,
                backend="gpt-sovits-countdown-renderer",
                selected=True,
                duration_ratio=candidate_ratio,
                duration_gate="pass",
                timing_quality_gate=_gsv_timing_quality_gate(
                    candidate_ratio,
                    "pass",
                    float(getattr(cfg, "gsv_timing_quality_tolerance", 0.10)),
                ),
                acceptable_for_mix=True,
                selection_score=max(0.0, 1.0 - min(abs(candidate_ratio - 1.0), 1.0)),
                selection_reason=selection_reason,
                retry_summary={"numeric_phrase_renderer": True},
            )
            segment.tts = TTSMetadata(
                backend="gpt-sovits-countdown-renderer",
                ref_style=segment.script.ref_style if segment.script else "whisper_close",
                speed_factor=1.0,
                candidate_count=1,
                selected_candidate_path=str(final_path),
                candidates=[candidate],
                source_language=cfg.source_language,
                target_language=cfg.target_language,
                cross_lingual_voice_transfer=cfg.source_language != cfg.target_language,
                retry_summary={
                    "numeric_phrase_renderer": True,
                    "selected_duration_gate": "pass",
                    "selected_acceptable_for_mix": True,
                    "selected_duration_ratio": candidate_ratio,
                    "selected_timing_quality_gate": candidate.timing_quality_gate,
                    "max_tempo": round(max_tempo, 6),
                    "max_tempo_limit": max_tempo_limit,
                },
            )
            segment.analysis["numeric_phrase_renderer"] = {
                "renderer": "numeric_phrase",
                "status": "rendered",
                **plan_payload,
                "request": request_payload,
                "candidate_text": str(request_payload.get("text") or plan.text),
                "candidate_generation": candidate_generation,
                "numeric_qc": numeric_qc,
                "placements": placements,
                "output_path": str(source_path),
                "selected_candidate_path": str(final_path),
                "duration_sec": round(actual_duration, 6),
                "duration_ratio": candidate_ratio,
                "duration_gate": "pass",
                "audio_qc": payload["audio_qc"],
                "max_tempo": round(max_tempo, 6),
                "max_tempo_limit": max_tempo_limit,
                "whole_lead_in_sec": float(
                    getattr(cfg, "gsv_numeric_phrase_whole_lead_in_sec", 0.12)
                ),
                "tail_guard_sec": float(getattr(cfg, "gsv_numeric_phrase_tail_guard_sec", 0.16)),
            }
            segment.status = "synthesized"
            numeric_phrase_rendered_segment_ids.add(segment.id)
            numeric_phrase_handled_segment_ids.add(segment.id)
            return True

        def numeric_phrase_ref_for_segment(segment: Segment) -> GPTSoVITSRef | None:
            if segment.script is None:
                return None
            requested_ref_style = segment.script.ref_style or "whisper_close"
            segment_refs = refs
            speaker_cfg = _gsv_speaker_cfg(cfg, segment)
            if speaker_cfg is not None:
                speaker_refs_path = _resolve_gsv_speaker_path(project_dir, speaker_cfg.refs_path)
                cache_key = str(speaker_refs_path)
                with speaker_refs_cache_lock:
                    if cache_key not in speaker_refs_cache:
                        speaker_refs_cache[cache_key] = load_refs(speaker_refs_path, project_dir)
                    segment_refs = speaker_refs_cache[cache_key]
                if requested_ref_style not in segment_refs:
                    requested_ref_style = speaker_cfg.default_ref_style
            return _ref_for_tts_language(
                resolve_ref(segment_refs, requested_ref_style),
                segment.script.tts_language,
            )

        def render_numeric_phrase_job(index: int, segment: Segment, lane_index: int) -> bool:
            plan = _numeric_phrase_render_plan_for_segment(segment, cfg)
            if plan is None:
                return False
            is_countdown_plan = _numeric_phrase_values_are_countdown(segment, plan.values)
            if countdown_only and not is_countdown_plan:
                return False
            if is_countdown_plan and (
                not render_countdowns
                or str(getattr(cfg, "gsv_countdown_renderer", "chunk_bank")) != "numeric_phrase"
            ):
                return False
            if should_reset_previous_tts(segment):
                reset_previous_tts_attempt(segment)
            if segment.status == "synthesized" or segment.status in SKIP_STATUSES:
                return False
            ref = numeric_phrase_ref_for_segment(segment)
            if not mock:
                start_gsv_servers()
            result = render_numeric_phrase_segment(
                project_dir=project_dir,
                segment=segment,
                plan=plan,
                cfg=cfg,
                mock=mock,
                lane_index=lane_index,
                gsv_url=None if mock else gsv_base_urls[lane_index],
                ref=ref,
                client=None if mock or not clients else clients[lane_index],
            )
            if not isinstance(result, dict):
                result = {"status": "failed", "reason": "numeric_phrase_renderer_invalid_result"}
            status = str(result.get("status") or "failed")
            if status == "rendered":
                return promote_numeric_phrase_rendered_result(
                    index=index,
                    segment=segment,
                    plan=plan,
                    result=result,
                )
            reason = str(result.get("reason") or "numeric_phrase_renderer_failed")
            fallback = str(
                getattr(cfg, "gsv_numeric_phrase_failure_fallback", "manual_review")
            )
            metadata = numeric_phrase_failure_metadata(
                segment=segment,
                plan=plan,
                reason=reason,
                fallback=fallback,
                result=result,
            )
            segment.analysis["numeric_phrase_renderer"] = metadata
            if fallback == "normal_tts" and not countdown_only:
                numeric_phrase_normal_tts_fallback_segment_ids.add(segment.id)
                return False
            return mark_numeric_phrase_manual_review(
                segment=segment,
                plan=plan,
                reason=reason,
                result=result,
            )

        def render_numeric_phrase_segments(
            segment_jobs: list[tuple[int, Segment, int]],
        ) -> set[str]:
            nonlocal last_logged_at
            if not bool(getattr(cfg, "gsv_numeric_phrase_renderer_enabled", True)):
                return set()
            handled: set[str] = set()
            job_total = len(segment_jobs)
            for progress_index, (index, segment, lane_index) in enumerate(segment_jobs, start=1):
                if render_numeric_phrase_job(index, segment, lane_index):
                    handled.add(segment.id)
                    save_manifest(project_dir, manifest)
                    last_logged_at = _log_segment_progress(
                        stage_name,
                        index,
                        job_total,
                        segment,
                        manifest,
                        started_at,
                        last_logged_at,
                        progress_index=progress_index,
                        counts_label="status_counts",
                    )
            if handled or numeric_phrase_normal_tts_fallback_segment_ids:
                _log_stage_checkpoint(
                    stage_name,
                    "numeric phrase renderer",
                    (
                        f"rendered={len(numeric_phrase_rendered_segment_ids)} "
                        f"failed={len(numeric_phrase_failed_segment_ids)} "
                        f"normal_tts_fallback={len(numeric_phrase_normal_tts_fallback_segment_ids)}"
                    ),
                )
            return handled

        def candidate_count_for_synth_pass(pass_name: str) -> int:
            if pass_candidate_count_override is not None:
                return pass_candidate_count_override
            if pass_name == "fine_tuned_initial":
                configured = getattr(cfg, "gsv_initial_candidate_count", None)
                return int(configured or min(int(cfg.candidate_count), 3))
            if pass_name == "zero_shot_fallback":
                configured = getattr(cfg, "gsv_zero_shot_candidate_count", None)
                if configured is not None:
                    return int(configured)
                return int(cfg.candidate_count)
            if pass_name == "low_temperature_retry":
                configured = getattr(cfg, "gsv_low_temperature_retry_candidate_count", None)
                if configured is not None:
                    return int(configured)
            configured = getattr(cfg, "gsv_retry_candidate_count", None)
            return int(configured or cfg.candidate_count)

        def temperature_for_synth_pass() -> float:
            if pass_temperature_override is not None:
                return float(pass_temperature_override)
            return float(cfg.gsv_temperature)

        def top_k_for_synth_pass() -> int:
            if pass_top_k_override is not None:
                return int(pass_top_k_override)
            return int(cfg.gsv_top_k)

        def top_p_for_synth_pass() -> float:
            if pass_top_p_override is not None:
                return float(pass_top_p_override)
            return float(cfg.gsv_top_p)

        def parallel_infer_for_synth_pass() -> bool:
            if pass_parallel_infer_override is not None:
                return bool(pass_parallel_infer_override)
            return bool(cfg.gsv_parallel_infer)

        def postprocess_tts_candidate(candidate_path: Path, payload: dict[str, Any]) -> None:
            if not cfg.gsv_trim_edge_silence:
                return
            trim = trim_edge_silence(
                candidate_path,
                threshold_db=cfg.gsv_trim_silence_threshold_db,
                keep_sec=cfg.gsv_trim_silence_keep_sec,
            )
            payload.setdefault("postprocess", {})["edge_silence_trim"] = trim

        def candidate_language_contract_ok(candidate: TTSCandidate) -> bool:
            if _canonical_language(cfg.target_language) != "ko":
                return True
            return (
                candidate.payload.get("text_lang") == "all_ko"
                and candidate.payload.get("prompt_lang") == "all_ja"
                and bool(candidate.payload.get("text"))
            )

        def candidate_pronunciation_contract_ok(candidate: TTSCandidate) -> bool:
            if _canonical_language(cfg.target_language) != "ko":
                return True
            if not bool(getattr(cfg, "gsv_pronunciation_qc_failure_blocks_mix", True)):
                pronunciation_blocks_mix = False
            else:
                pronunciation_blocks_mix = True
            qc = candidate.payload.get("pronunciation_qc")
            if pronunciation_blocks_mix and isinstance(qc, dict):
                gate = str(qc.get("gate") or "").strip().lower()
                if gate == "fail":
                    return False
                if (
                    gate == "warn"
                    and candidate.payload.get("prompt_text_policy") == "use_segment_source_reference"
                    and bool(getattr(cfg, "gsv_korean_segment_ref_warn_blocks_mix", True))
                ):
                    return False
            if not bool(getattr(cfg, "gsv_numeric_sequence_qc_failure_blocks_mix", True)):
                return True
            numeric_qc = candidate.payload.get("numeric_sequence_qc")
            if not isinstance(numeric_qc, dict):
                return True
            numeric_gate = str(numeric_qc.get("gate") or "").strip().lower()
            return numeric_gate != "fail"

        def segment_ref_pronunciation_contract_ok(
            *,
            target_language: str,
            using_segment_ref: bool,
            pronunciation_gate: str,
        ) -> bool:
            if not (
                target_language == "ko"
                and bool(getattr(cfg, "gsv_pronunciation_qc_failure_blocks_mix", True))
            ):
                return True
            if pronunciation_gate == "fail":
                return False
            return not (
                pronunciation_gate == "warn"
                and using_segment_ref
                and bool(getattr(cfg, "gsv_korean_segment_ref_warn_blocks_mix", True))
            )

        def run_numeric_sequence_qc(
            *,
            expected_text: str,
            pronunciation_qc: dict[str, Any] | None,
        ) -> dict[str, Any] | None:
            if (
                _canonical_language(cfg.target_language) != "ko"
                or not bool(getattr(cfg, "gsv_numeric_sequence_qc_enabled", True))
                or not isinstance(pronunciation_qc, dict)
            ):
                return None
            result = evaluate_numeric_sequence_text(
                expected_text,
                str(pronunciation_qc.get("transcript") or ""),
                min_values=int(getattr(cfg, "gsv_numeric_cadence_min_values", 3)),
                require_contiguous=bool(
                    getattr(cfg, "gsv_numeric_sequence_qc_require_contiguous", True)
                ),
                backend=str(pronunciation_qc.get("backend") or ""),
            )
            if result.gate == "unavailable":
                return None
            return result.as_payload()

        def pronunciation_qc_backend_name() -> str:
            configured = (
                str(getattr(cfg, "gsv_pronunciation_qc_backend", "auto"))
                .strip()
                .lower()
                .replace("-", "_")
            )
            if configured != "auto":
                return configured
            backend = str(getattr(cfg, "asr_backend", "faster_whisper")).strip().lower()
            return "qwen_asr" if backend == "qwen_asr" else "faster_whisper"

        def pronunciation_qc_backend_config(*, short_slice: bool = False) -> dict[str, Any]:
            backend_config = _asr_backend_config(cfg)
            backend_config["language"] = "ko"
            backend_config["batched_inference"] = False
            backend_config["condition_on_previous_text"] = False
            backend_config["vad_filter"] = False
            if short_slice:
                backend_config["initial_prompt"] = ""
                backend_config["hotwords"] = ""
            else:
                backend_config["initial_prompt"] = "한국어 ASMR 더빙 음성을 그대로 받아 적습니다."
            return backend_config

        def pronunciation_qc_worker_index(worker_slot: int | None = None) -> int:
            if pronunciation_qc_worker_count <= 1:
                return 0
            if worker_slot is None:
                return 0
            return max(0, int(worker_slot)) % pronunciation_qc_worker_count

        def pronunciation_qc_worker_cache_key(base_key: str, worker_index: int) -> str:
            return f"{base_key}:worker_{worker_index:02d}"

        def run_pronunciation_qc(
            *,
            segment: Segment,
            candidate_path: Path,
            expected_text: str,
            candidate_duration_sec: float,
            short_slice: bool = False,
            worker_slot: int | None = None,
        ) -> dict[str, Any] | None:
            if (
                mock
                or _canonical_language(cfg.target_language) != "ko"
                or not bool(getattr(cfg, "gsv_pronunciation_qc_enabled", True))
            ):
                return None
            configured_backend = (
                str(getattr(cfg, "gsv_pronunciation_qc_backend", "auto"))
                .strip()
                .lower()
                .replace("-", "_")
            )
            if configured_backend == "auto" and (
                segment.source_script is None
                or str(segment.source_script.backend).strip().lower() == "mock"
            ):
                return None
            backend_name = pronunciation_qc_backend_name()
            base_backend_cache_key = f"{backend_name}:short_slice" if short_slice else backend_name
            worker_index = pronunciation_qc_worker_index(worker_slot)
            backend_cache_key = pronunciation_qc_worker_cache_key(
                base_backend_cache_key,
                worker_index,
            )
            if backend_cache_key in pronunciation_qc_unavailable_errors:
                return {
                    "gate": "unavailable",
                    "coverage": 1.0,
                    "backend": backend_name,
                    "error": pronunciation_qc_unavailable_errors[backend_cache_key],
                    "issues": ["pronunciation_qc_unavailable"],
                    "worker_index": worker_index,
                    "workers": pronunciation_qc_worker_count,
                }
            try:
                with pronunciation_qc_backend_locks[worker_index]:
                    backend = pronunciation_qc_backend_cache.get(backend_cache_key)
                    if backend is None:
                        backend = create_asr_backend(
                            backend_name,
                            pronunciation_qc_backend_config(short_slice=short_slice),
                        )
                        pronunciation_qc_backend_cache[backend_cache_key] = backend
                    qc_duration = max(float(candidate_duration_sec), 0.001)
                    qc_segment = Segment(
                        id=f"{segment.id}_pronunciation_qc",
                        speaker_id=segment.speaker_id,
                        start=0.0,
                        end=qc_duration,
                        duration=qc_duration,
                        audio_for_gemma=str(candidate_path),
                        audio_for_mix=str(candidate_path),
                    )
                    chunks = backend.transcribe(candidate_path, [qc_segment])
                result = evaluate_pronunciation_chunks(
                    expected_text,
                    chunks,
                    pass_coverage=float(getattr(cfg, "gsv_pronunciation_qc_pass_coverage", 0.82)),
                    warn_coverage=float(getattr(cfg, "gsv_pronunciation_qc_warn_coverage", 0.62)),
                    max_observed_unit_ratio=getattr(
                        cfg,
                        "gsv_pronunciation_qc_max_observed_unit_ratio",
                        1.8,
                    ),
                    max_extra_units=getattr(cfg, "gsv_pronunciation_qc_max_extra_units", 1),
                    backend=backend_name,
                )
                payload = result.as_payload()
                payload["short_slice"] = short_slice
                payload["worker_index"] = worker_index
                payload["workers"] = pronunciation_qc_worker_count
                return payload
            except Exception as exc:
                pronunciation_qc_unavailable_errors[backend_cache_key] = str(exc)
                return {
                    "gate": "unavailable",
                    "coverage": 1.0,
                    "backend": backend_name,
                    "error": str(exc),
                    "issues": ["pronunciation_qc_unavailable"],
                    "short_slice": short_slice,
                    "worker_index": worker_index,
                    "workers": pronunciation_qc_worker_count,
                }

        def run_pronunciation_qc_batch(
            tasks: list[dict[str, Any]],
            *,
            short_slice: bool = False,
        ) -> list[dict[str, Any] | None]:
            if not tasks:
                return []
            worker_count = max(
                1,
                int(getattr(cfg, "gsv_countdown_carrier_bulk_asr_workers", 3)),
            )
            if gsv_servers_running and worker_count > 1:
                _log_stage_checkpoint(
                    stage_name,
                    "countdown bulk asr",
                    (
                        "clamping inline asr workers to 1 while gsv servers are running "
                        f"configured_workers={worker_count}"
                    ),
                )
                worker_count = 1
            if worker_count <= 1:
                return [
                    run_pronunciation_qc(
                        segment=task["segment"],
                        candidate_path=Path(task["candidate_path"]),
                        expected_text=str(task["expected_text"]),
                        candidate_duration_sec=float(task["candidate_duration_sec"]),
                        short_slice=short_slice,
                    )
                    for task in tasks
                ]
            if (
                mock
                or _canonical_language(cfg.target_language) != "ko"
                or not bool(getattr(cfg, "gsv_pronunciation_qc_enabled", True))
            ):
                return [None for _ in tasks]
            configured_backend = (
                str(getattr(cfg, "gsv_pronunciation_qc_backend", "auto"))
                .strip()
                .lower()
                .replace("-", "_")
            )
            results: list[dict[str, Any] | None] = []
            runnable_tasks: list[tuple[int, dict[str, Any]]] = []
            for index, task in enumerate(tasks):
                task_segment = task["segment"]
                if configured_backend == "auto" and (
                    task_segment.source_script is None
                    or str(task_segment.source_script.backend).strip().lower() == "mock"
                ):
                    results.append(None)
                    continue
                results.append({})
                runnable_tasks.append((index, task))
            if not runnable_tasks:
                return [None for _ in tasks]

            backend_name = pronunciation_qc_backend_name()
            backend_cache_key = f"{backend_name}:short_slice" if short_slice else backend_name
            if backend_cache_key in pronunciation_qc_unavailable_errors:
                unavailable = {
                    "gate": "unavailable",
                    "coverage": 1.0,
                    "backend": backend_name,
                    "error": pronunciation_qc_unavailable_errors[backend_cache_key],
                    "issues": ["pronunciation_qc_unavailable"],
                    "short_slice": short_slice,
                    "bulk_asr": True,
                }
                return [None if result is None else dict(unavailable) for result in results]

            asr_local = threading.local()

            def backend_for_worker() -> Any:
                backend = getattr(asr_local, "backend", None)
                if backend is None:
                    backend = create_asr_backend(
                        backend_name,
                        pronunciation_qc_backend_config(short_slice=short_slice),
                    )
                    asr_local.backend = backend
                return backend

            def evaluate_task(index: int, task: dict[str, Any]) -> tuple[int, dict[str, Any]]:
                candidate_path = Path(task["candidate_path"])
                candidate_duration_sec = max(float(task["candidate_duration_sec"]), 0.001)
                task_segment = task["segment"]
                qc_segment = Segment(
                    id=f"{task_segment.id}_pronunciation_qc_{index}",
                    speaker_id=task_segment.speaker_id,
                    start=0.0,
                    end=candidate_duration_sec,
                    duration=candidate_duration_sec,
                    audio_for_gemma=str(candidate_path),
                    audio_for_mix=str(candidate_path),
                )
                chunks = backend_for_worker().transcribe(candidate_path, [qc_segment])
                result = evaluate_pronunciation_chunks(
                    str(task["expected_text"]),
                    chunks,
                    pass_coverage=float(getattr(cfg, "gsv_pronunciation_qc_pass_coverage", 0.82)),
                    warn_coverage=float(getattr(cfg, "gsv_pronunciation_qc_warn_coverage", 0.62)),
                    max_observed_unit_ratio=getattr(
                        cfg,
                        "gsv_pronunciation_qc_max_observed_unit_ratio",
                        1.8,
                    ),
                    max_extra_units=getattr(cfg, "gsv_pronunciation_qc_max_extra_units", 1),
                    backend=backend_name,
                )
                payload = result.as_payload()
                payload["short_slice"] = short_slice
                payload["bulk_asr"] = True
                return index, payload

            with countdown_bulk_asr_lock:
                try:
                    with ThreadPoolExecutor(max_workers=worker_count) as executor:
                        futures = [
                            executor.submit(evaluate_task, index, task)
                            for index, task in runnable_tasks
                        ]
                        for completed_count, future in enumerate(as_completed(futures), start=1):
                            index, payload = future.result()
                            results[index] = payload
                            if completed_count == len(futures) or completed_count % 50 == 0:
                                _log_stage_checkpoint(
                                    stage_name,
                                    "countdown bulk asr",
                                    f"asr={completed_count}/{len(futures)} workers={worker_count}",
                                )
                except Exception as exc:
                    pronunciation_qc_unavailable_errors[backend_cache_key] = str(exc)
                    unavailable = {
                        "gate": "unavailable",
                        "coverage": 1.0,
                        "backend": backend_name,
                        "error": str(exc),
                        "issues": ["pronunciation_qc_unavailable"],
                        "short_slice": short_slice,
                        "bulk_asr": True,
                    }
                    results = [None if result is None else dict(unavailable) for result in results]
            return results

        def time_fit_candidate_if_needed(
            segment: Segment,
            selected: TTSCandidate,
            *,
            output_label: str = "timefit",
            max_tempo_override: float | None = None,
            max_stretch_override: float | None = None,
            allow_acceptable: bool = False,
            rescue_metadata: dict[str, Any] | None = None,
            selection_reason_if_acceptable: str = "duration_time_fit_fallback",
            selection_reason_if_failed: str = "duration_time_fit_failed",
        ) -> TTSCandidate | None:
            if selected.acceptable_for_mix and not allow_acceptable:
                return None
            if selected.payload.get("audio_qc", {}).get("gate") != "pass":
                return None
            if not candidate_language_contract_ok(selected):
                return None
            if not candidate_pronunciation_contract_ok(selected):
                return None
            if selected.duration_sec is None or selected.duration_sec <= 0 or segment.duration <= 0:
                return None
            source_path = Path(selected.output_path)
            fitted_path = project_dir / "work" / "tts" / "candidates" / f"{segment.id}_{output_label}.wav"
            payload = copy.deepcopy(selected.payload)
            source_timing_quality_gate = selected.timing_quality_gate
            tempo = selected.duration_sec / segment.duration
            stretch = segment.duration / selected.duration_sec
            base_max_tempo = float(getattr(cfg, "gsv_timefit_max_tempo", 1.18))
            base_max_stretch = float(getattr(cfg, "gsv_timefit_max_stretch", 1.08))
            max_tempo = base_max_tempo
            max_stretch = base_max_stretch
            policy = "default"
            micro_max_sec = float(getattr(cfg, "gsv_timefit_micro_max_sec", 2.0))
            if segment.duration <= micro_max_sec and tempo > base_max_tempo:
                max_tempo = max(
                    max_tempo,
                    float(getattr(cfg, "gsv_timefit_micro_max_tempo", 1.30)),
                )
                policy = "micro_segment_relaxed"
            long_min_sec = float(getattr(cfg, "gsv_timefit_long_min_sec", 7.0))
            if segment.duration >= long_min_sec and stretch > base_max_stretch:
                max_stretch = max(
                    max_stretch,
                    float(getattr(cfg, "gsv_timefit_long_max_stretch", 1.15)),
                )
                policy = "long_segment_relaxed"
            if max_tempo_override is not None and max_tempo_override > max_tempo:
                max_tempo = float(max_tempo_override)
                policy = "rescue_relaxed" if policy == "default" else f"{policy}_rescue_relaxed"
            if max_stretch_override is not None and max_stretch_override > max_stretch:
                max_stretch = float(max_stretch_override)
                policy = "rescue_relaxed" if policy == "default" else f"{policy}_rescue_relaxed"
            if tempo > max_tempo:
                selected.payload["time_fit"] = {
                    "source_path": selected.output_path,
                    "source_duration_sec": selected.duration_sec,
                    "target_duration_sec": segment.duration,
                    "tempo": tempo,
                    "stretch": stretch,
                    "policy": policy,
                    "max_tempo": max_tempo,
                    "max_stretch": max_stretch,
                    "base_max_tempo": base_max_tempo,
                    "base_max_stretch": base_max_stretch,
                    "source_timing_quality_gate": source_timing_quality_gate,
                    "rejected_reason": f"tempo_above_max:{tempo:.3f}>{max_tempo:.3f}",
                }
                return None
            if stretch > max_stretch:
                selected.payload["time_fit"] = {
                    "source_path": selected.output_path,
                    "source_duration_sec": selected.duration_sec,
                    "target_duration_sec": segment.duration,
                    "tempo": tempo,
                    "stretch": stretch,
                    "policy": policy,
                    "max_tempo": max_tempo,
                    "max_stretch": max_stretch,
                    "base_max_tempo": base_max_tempo,
                    "base_max_stretch": base_max_stretch,
                    "source_timing_quality_gate": source_timing_quality_gate,
                    "rejected_reason": f"stretch_above_max:{stretch:.3f}>{max_stretch:.3f}",
                }
                return None
            try:
                ffmpeg.fit_audio_duration(
                    source_path,
                    fitted_path,
                    target_duration_sec=segment.duration,
                    sample_rate=cfg.mix_sample_rate,
                    channels=2,
                )
                fitted_duration = duration_sec(fitted_path)
                fitted_peak = peak_dbfs(fitted_path)
                fitted_rms = rms_dbfs(fitted_path)
            except Exception as exc:
                selected.payload.setdefault("time_fit", {})["error"] = str(exc)
                return None
            too_long = duration_too_long(fitted_duration, segment.duration, cfg.duration_tolerance)
            too_short = duration_too_short(fitted_duration, segment.duration, cfg.duration_tolerance)
            fitted_ratio = duration_ratio(fitted_duration, segment.duration)
            duration_gate = "too_long" if too_long else "too_short" if too_short else "pass"
            audio_gate = "silent" if fitted_peak <= -90.0 or fitted_rms <= -90.0 else "pass"
            timing_quality_gate = _gsv_timing_quality_gate(
                fitted_ratio,
                duration_gate,
                float(getattr(cfg, "gsv_timing_quality_tolerance", 0.10)),
            )
            payload["duration_ratio"] = fitted_ratio
            payload["duration_gate"] = duration_gate
            payload["timing_quality"] = _gsv_timing_quality_payload(
                fitted_ratio,
                duration_gate,
                float(getattr(cfg, "gsv_timing_quality_tolerance", 0.10)),
                float(cfg.duration_tolerance),
            )
            payload["audio_qc"] = {
                "gate": audio_gate,
                "peak_dbfs": round(fitted_peak, 3),
                "rms_dbfs": round(fitted_rms, 3),
            }
            payload["time_fit"] = {
                "source_path": selected.output_path,
                "source_duration_sec": selected.duration_sec,
                "target_duration_sec": segment.duration,
                "tempo": tempo,
                "stretch": stretch,
                "policy": policy,
                "max_tempo": max_tempo,
                "max_stretch": max_stretch,
                "base_max_tempo": base_max_tempo,
                "base_max_stretch": base_max_stretch,
                "duration_ratio_before": selected.duration_ratio,
                "duration_ratio_after": fitted_ratio,
                "source_timing_quality_gate": source_timing_quality_gate,
            }
            if rescue_metadata is not None:
                payload["rescue"] = rescue_metadata
            acceptable_for_mix = (
                duration_gate == "pass"
                and audio_gate == "pass"
                and candidate_language_contract_ok(selected)
                and candidate_pronunciation_contract_ok(selected)
            )
            selection_score = max(0.0, 1.0 - min(abs(fitted_ratio - 1.0), 1.0))
            return TTSCandidate(
                candidate_index=selected.candidate_index,
                seed=selected.seed,
                payload=payload,
                output_path=str(fitted_path),
                duration_sec=fitted_duration,
                backend=selected.backend,
                duration_ratio=fitted_ratio,
                duration_gate=duration_gate,
                timing_quality_gate=timing_quality_gate,
                acceptable_for_mix=acceptable_for_mix,
                selection_score=selection_score,
                selection_reason=(
                    selection_reason_if_acceptable
                    if acceptable_for_mix
                    else selection_reason_if_failed
                ),
                retry_summary=selected.retry_summary,
            )

        def duration_gate_for_tolerance(
            actual_duration_sec: float,
            target_duration_sec: float,
            tolerance: float,
        ) -> tuple[str, float]:
            ratio = duration_ratio(actual_duration_sec, target_duration_sec)
            too_long = duration_too_long(actual_duration_sec, target_duration_sec, tolerance)
            too_short = duration_too_short(actual_duration_sec, target_duration_sec, tolerance)
            return ("too_long" if too_long else "too_short" if too_short else "pass", ratio)

        def rescue_with_relaxed_duration_gate(
            segment: Segment,
            audible: list[TTSCandidate],
        ) -> TTSCandidate | None:
            rescue_tolerance = getattr(cfg, "gsv_rescue_duration_tolerance", 0.35)
            if rescue_tolerance is None:
                return None
            rescue_tolerance = float(rescue_tolerance)
            for source_candidate in sorted(
                audible,
                key=lambda candidate: abs((candidate.duration_sec or 0.0) - segment.duration),
            ):
                if (
                    source_candidate.duration_sec is None
                    or source_candidate.duration_sec <= 0
                    or segment.duration <= 0
                    or not candidate_language_contract_ok(source_candidate)
                    or not candidate_pronunciation_contract_ok(source_candidate)
                ):
                    continue
                duration_gate, candidate_ratio = duration_gate_for_tolerance(
                    source_candidate.duration_sec,
                    segment.duration,
                    rescue_tolerance,
                )
                if duration_gate != "pass":
                    continue
                payload = copy.deepcopy(source_candidate.payload)
                payload["duration_ratio"] = candidate_ratio
                payload["duration_gate"] = "pass"
                timing_quality_gate = _gsv_timing_quality_gate(
                    candidate_ratio,
                    "pass",
                    float(getattr(cfg, "gsv_timing_quality_tolerance", 0.10)),
                )
                payload["timing_quality"] = _gsv_timing_quality_payload(
                    candidate_ratio,
                    "pass",
                    float(getattr(cfg, "gsv_timing_quality_tolerance", 0.10)),
                    float(cfg.duration_tolerance),
                )
                payload["rescue"] = {
                    "tier": "relaxed_duration_gate",
                    "source_candidate_index": source_candidate.candidate_index,
                    "source_candidate_path": source_candidate.output_path,
                    "source_selection_reason": source_candidate.selection_reason,
                    "strict_duration_gate": source_candidate.duration_gate,
                    "strict_duration_ratio": source_candidate.duration_ratio,
                    "strict_duration_tolerance": cfg.duration_tolerance,
                    "duration_tolerance_used": rescue_tolerance,
                }
                return TTSCandidate(
                    candidate_index=source_candidate.candidate_index,
                    seed=source_candidate.seed,
                    payload=payload,
                    output_path=source_candidate.output_path,
                    duration_sec=source_candidate.duration_sec,
                    backend=source_candidate.backend,
                    duration_ratio=candidate_ratio,
                    duration_gate="pass",
                    timing_quality_gate=timing_quality_gate,
                    acceptable_for_mix=True,
                    selection_score=max(0.0, 1.0 - min(abs(candidate_ratio - 1.0), 1.0)),
                    selection_reason="duration_relaxed_rescue",
                    retry_summary=copy.deepcopy(source_candidate.retry_summary),
                )
            return None

        def rescue_with_source_pause_padding(
            segment: Segment,
            audible: list[TTSCandidate],
        ) -> TTSCandidate | None:
            rescue_tolerance = getattr(cfg, "gsv_rescue_duration_tolerance", 0.35)
            if rescue_tolerance is None or not segment.script:
                return None
            rescue_tolerance = float(rescue_tolerance)
            expected_duration_sec = float(segment.script.expected_tts_duration_sec or 0.0)
            if expected_duration_sec <= 1.0 or segment.duration <= expected_duration_sec:
                return None
            for source_candidate in sorted(
                audible,
                key=lambda candidate: abs((candidate.duration_sec or 0.0) - expected_duration_sec),
            ):
                allowed_omission_reasons = _omission_reasons_allow_source_pause_padding(
                    source_candidate
                )
                if (
                    source_candidate.duration_sec is None
                    or source_candidate.duration_sec <= 0.0
                    or source_candidate.duration_sec >= segment.duration
                    or source_candidate.duration_gate != "too_short"
                    or allowed_omission_reasons is None
                    or not candidate_language_contract_ok(source_candidate)
                    or not candidate_pronunciation_contract_ok(source_candidate)
                ):
                    continue
                speech_ratio = source_candidate.duration_sec / expected_duration_sec
                if speech_ratio < 1.0 - rescue_tolerance or speech_ratio > 1.0 + rescue_tolerance:
                    continue
                source_path = Path(source_candidate.output_path)
                padded_path = source_path.with_name(f"{source_path.stem}_pause_padded.wav")
                try:
                    source_audio, sample_rate = load_audio(source_path)
                    target_frames = max(
                        len(source_audio),
                        int(round(segment.duration * sample_rate)),
                    )
                    padding_frames = target_frames - len(source_audio)
                    if padding_frames <= 0:
                        continue
                    silence = np.zeros(
                        (padding_frames, source_audio.shape[1]),
                        dtype=source_audio.dtype,
                    )
                    write_audio(padded_path, np.concatenate([source_audio, silence], axis=0), sample_rate)
                    padded_duration = duration_sec(padded_path)
                    padded_peak = peak_dbfs(padded_path)
                    padded_rms = rms_dbfs(padded_path)
                except Exception as exc:
                    source_candidate.payload.setdefault("pause_padding", {})["error"] = str(exc)
                    continue
                too_long = duration_too_long(padded_duration, segment.duration, cfg.duration_tolerance)
                too_short = duration_too_short(padded_duration, segment.duration, cfg.duration_tolerance)
                padded_ratio = duration_ratio(padded_duration, segment.duration)
                duration_gate = "too_long" if too_long else "too_short" if too_short else "pass"
                audio_gate = "silent" if padded_peak <= -90.0 or padded_rms <= -90.0 else "pass"
                if duration_gate != "pass" or audio_gate != "pass":
                    continue
                padding_sec = max(0.0, padded_duration - source_candidate.duration_sec)
                payload = copy.deepcopy(source_candidate.payload)
                payload["duration_ratio"] = padded_ratio
                payload["duration_gate"] = "pass"
                timing_quality_gate = _gsv_timing_quality_gate(
                    padded_ratio,
                    "pass",
                    float(getattr(cfg, "gsv_timing_quality_tolerance", 0.10)),
                )
                payload["timing_quality"] = _gsv_timing_quality_payload(
                    padded_ratio,
                    "pass",
                    float(getattr(cfg, "gsv_timing_quality_tolerance", 0.10)),
                    float(cfg.duration_tolerance),
                )
                payload["audio_qc"] = {
                    "gate": audio_gate,
                    "peak_dbfs": round(padded_peak, 3),
                    "rms_dbfs": round(padded_rms, 3),
                }
                payload["pause_padding"] = {
                    "tier": "source_pause_padding",
                    "source_candidate_index": source_candidate.candidate_index,
                    "source_candidate_path": source_candidate.output_path,
                    "speech_duration_sec": round(source_candidate.duration_sec, 6),
                    "padding_sec": round(padding_sec, 6),
                    "expected_tts_duration_sec": round(expected_duration_sec, 6),
                    "target_segment_duration_sec": round(segment.duration, 6),
                    "speech_duration_ratio_to_expected": round(speech_ratio, 6),
                    "strict_duration_gate": source_candidate.duration_gate,
                    "strict_duration_ratio": source_candidate.duration_ratio,
                    "strict_duration_tolerance": cfg.duration_tolerance,
                    "speech_duration_tolerance_used": rescue_tolerance,
                }
                if allowed_omission_reasons:
                    payload["pause_padding"]["allowed_omission_detection_reasons"] = (
                        allowed_omission_reasons
                    )
                payload["rescue"] = copy.deepcopy(payload["pause_padding"])
                return TTSCandidate(
                    candidate_index=source_candidate.candidate_index,
                    seed=source_candidate.seed,
                    payload=payload,
                    output_path=str(padded_path),
                    duration_sec=padded_duration,
                    backend=source_candidate.backend,
                    duration_ratio=padded_ratio,
                    duration_gate="pass",
                    timing_quality_gate=timing_quality_gate,
                    acceptable_for_mix=True,
                    selection_score=max(0.0, 1.0 - min(abs(padded_ratio - 1.0), 1.0)),
                    selection_reason="source_pause_padding_rescue",
                    retry_summary=copy.deepcopy(source_candidate.retry_summary),
                )
            return None

        def rescue_with_relaxed_time_fit(
            segment: Segment,
            audible: list[TTSCandidate],
            candidates: list[TTSCandidate],
        ) -> TTSCandidate | None:
            top_k = int(getattr(cfg, "gsv_rescue_timefit_top_k", 3))
            if top_k <= 0:
                return None
            max_tempo = float(getattr(cfg, "gsv_rescue_timefit_max_tempo", 1.45))
            max_stretch = float(getattr(cfg, "gsv_rescue_timefit_max_stretch", 1.25))
            counting_compaction_timefit = bool(
                segment.analysis.get("pre_synth_tts_counting_compaction")
            )
            if counting_compaction_timefit:
                max_tempo = max(
                    max_tempo,
                    float(getattr(cfg, "gsv_rescue_timefit_counting_max_tempo", 1.60)),
                )
            ranked = sorted(
                audible,
                key=lambda candidate: abs((candidate.duration_sec or 0.0) - segment.duration),
            )[:top_k]
            for rank, source_candidate in enumerate(ranked, start=1):
                rescue_metadata = {
                    "tier": "relaxed_time_fit",
                    "source_candidate_index": source_candidate.candidate_index,
                    "source_candidate_path": source_candidate.output_path,
                    "source_selection_reason": source_candidate.selection_reason,
                    "strict_duration_gate": source_candidate.duration_gate,
                    "strict_duration_ratio": source_candidate.duration_ratio,
                    "max_tempo_used": max_tempo,
                    "max_stretch_used": max_stretch,
                    "rank": rank,
                }
                if counting_compaction_timefit:
                    rescue_metadata["counting_compaction_timefit"] = True
                fitted = time_fit_candidate_if_needed(
                    segment,
                    source_candidate.model_copy(deep=True),
                    output_label=f"rescue_timefit_{rank:02d}",
                    max_tempo_override=max_tempo,
                    max_stretch_override=max_stretch,
                    rescue_metadata=rescue_metadata,
                    selection_reason_if_acceptable="duration_relaxed_timefit_rescue",
                    selection_reason_if_failed="duration_relaxed_timefit_failed",
                )
                if fitted is None:
                    continue
                candidates.append(fitted)
                if fitted.acceptable_for_mix:
                    return fitted
            return None

        def micro_segment_unfit_summary(
            segment: Segment,
            selected: TTSCandidate,
            evaluated_candidates: Sequence[TTSCandidate],
        ) -> dict[str, Any] | None:
            if not is_micro_segment(segment, cfg):
                return None
            micro_max_sec = float(getattr(cfg, "gsv_micro_segment_max_sec", 1.2))
            duration_gated = [
                candidate
                for candidate in evaluated_candidates
                if str(candidate.duration_gate or "").strip()
            ]
            if not duration_gated or any(candidate.duration_gate != "too_long" for candidate in duration_gated):
                return None
            return {
                "acceptable_candidates": 0,
                "selected_duration_gate": selected.duration_gate,
                "selected_duration_ratio": selected.duration_ratio,
                "micro_segment_max_sec": micro_max_sec,
                "target_duration_sec": segment.duration,
            }

        def duration_rewrite_context(index: int) -> list[Segment]:
            radius = int(getattr(cfg, "gemma_text_context_radius", 0))
            if radius <= 0:
                return []
            start = max(0, index - 1 - radius)
            end = min(len(manifest.segments), index + radius)
            return manifest.segments[start:end]

        def duration_rewrite_char_budget(
            segment: Segment,
            text: str,
            actual_duration_sec: float,
            reason: str,
        ) -> dict[str, int]:
            current_chars = korean_tts_speech_char_count(text)
            source_text = segment.source_script.text if segment.source_script else ""
            timing_budget = korean_tts_timing_budget(segment.duration, source_text)
            max_chars = max(1, int(timing_budget["max_speech_chars"]))
            if actual_duration_sec > 0 and segment.duration > 0:
                target_chars = int(round(current_chars * segment.duration / actual_duration_sec))
            else:
                target_chars = current_chars
            target_chars = max(1, min(max_chars, target_chars))
            if reason == "too_short":
                target_chars = max(min(current_chars + 1, max_chars), target_chars)
                min_chars = min(max_chars, max(current_chars + 1, int(target_chars * 0.85)))
            else:
                min_chars = max(1, int(target_chars * 0.80))
            return {
                "current": current_chars,
                "target": target_chars,
                "min": max(1, min_chars),
                "max": max_chars,
            }

        def accept_duration_rewrite_text(
            *,
            segment: Segment,
            candidate_text: str,
            reason: str,
            budget: dict[str, int],
        ) -> tuple[bool, dict[str, Any]]:
            normalized = normalize_korean_tts_text(candidate_text)
            text = normalized.text.strip()
            chars = korean_tts_speech_char_count(text)
            metadata: dict[str, Any] = {
                "normalized_text": text,
                "speech_chars": chars,
                "target_speech_chars": budget["target"],
                "min_speech_chars": budget["min"],
                "max_speech_chars": budget["max"],
                "rejected_reasons": [],
            }
            if not text:
                metadata["rejected_reasons"].append("empty_rewrite")
            if chars > budget["max"]:
                metadata["rejected_reasons"].append(
                    f"speech_chars_above_max:{chars}>{budget['max']}"
                )
            if reason == "too_short" and chars < budget["min"]:
                metadata["rejected_reasons"].append(
                    f"speech_chars_below_min:{chars}<{budget['min']}"
                )
            if reason == "too_long" and chars >= budget["current"]:
                metadata["rejected_reasons"].append(
                    f"not_shorter_for_too_long:{chars}>={budget['current']}"
                )
            trial_script = segment.script.model_copy(
                update={
                    "tts_text": text,
                    "expected_tts_duration_sec": estimate_tts_duration(text, "ko"),
                    "risk_flags": [*segment.script.risk_flags, *normalized.risk_flags],
                },
                deep=True,
            )
            preflight = preflight_tts_text(
                trial_script,
                target_language=cfg.target_language,
                source_text=segment.source_script.text if segment.source_script else "",
                min_hangul_ratio=cfg.gsv_ko_text_min_hangul_ratio,
            )
            metadata["preflight"] = preflight.as_payload()
            if preflight.blocked:
                metadata["rejected_reasons"].append(
                    "preflight_blocked:" + ",".join(preflight.issues)
                )
            return not metadata["rejected_reasons"], metadata

        def maybe_rewrite_script_with_gemma(
            *,
            segment: Segment,
            index: int,
            attempt_text: str,
            actual_duration_sec: float,
            reason: str,
            rewrite_attempts_used: int,
        ) -> tuple[JapaneseScript | None, dict[str, Any]]:
            metadata: dict[str, Any] = {
                "backend": "gemma_text",
                "reason": reason,
                "before": attempt_text,
                "actual_duration_sec": round(actual_duration_sec, 6),
                "target_duration_sec": round(segment.duration, 6),
                "accepted": False,
            }
            if duration_rewrite_client is None:
                metadata["error"] = "duration_rewrite_client_unavailable"
                return None, metadata
            if rewrite_attempts_used >= int(getattr(cfg, "gsv_duration_rewrite_max_attempts", 0)):
                metadata["error"] = "duration_rewrite_attempt_limit_reached"
                return None, metadata
            if segment.script.rewrite_count >= segment.script.retry_policy.max_script_rewrites:
                metadata["error"] = "script_retry_policy_limit_reached"
                return None, metadata
            budget = duration_rewrite_char_budget(segment, attempt_text, actual_duration_sec, reason)
            metadata.update(
                {
                    "current_speech_chars": budget["current"],
                    "target_speech_chars": budget["target"],
                    "min_speech_chars": budget["min"],
                    "max_speech_chars": budget["max"],
                }
            )
            batch_id = f"duration_rewrite_{segment.id}_{reason}_{segment.script.rewrite_count + 1}"
            try:
                with duration_rewrite_lock:
                    translation: KoreanTranslation | None = duration_rewrite_client.rewrite_tts_for_duration(
                        segment=segment,
                        batch_id=batch_id,
                        current_text=attempt_text,
                        reason=reason,
                        actual_duration_sec=actual_duration_sec,
                        target_duration_sec=segment.duration,
                        target_speech_chars=budget["target"],
                        min_speech_chars=budget["min"],
                        max_speech_chars=budget["max"],
                        context_segments=duration_rewrite_context(index),
                    )
            except Exception as exc:
                metadata["error"] = str(exc)
                return None, metadata
            if translation is None:
                metadata["error"] = "gemma_returned_no_translation"
                return None, metadata
            accepted, acceptance = accept_duration_rewrite_text(
                segment=segment,
                candidate_text=translation.ko_natural,
                reason=reason,
                budget=budget,
            )
            metadata.update(acceptance)
            metadata["after"] = acceptance["normalized_text"]
            metadata["model"] = translation.model
            metadata["batch_id"] = translation.batch_id
            if not accepted and not _maybe_relax_duration_rewrite_acceptance(metadata):
                return None, metadata
            updated = segment.script.model_copy(deep=True)
            updated.tts_text = acceptance["normalized_text"]
            updated.expected_tts_duration_sec = estimate_tts_duration(updated.tts_text, "ko")
            updated.rewrite_count += 1
            risk_flag = (
                f"gemma_duration_rewrite_{reason}_relaxed"
                if metadata.get("accepted_relaxed")
                else f"gemma_duration_rewrite_{reason}"
            )
            updated.risk_flags.append(risk_flag)
            metadata["accepted"] = True
            return updated, metadata

        def synthesize_segment_locked(
            index: int,
            segment: Segment,
            lane_index: int,
        ) -> tuple[int, Segment]:
            if should_reset_previous_tts(segment):
                reset_previous_tts_attempt(segment)
            if segment.status == "synthesized":
                return index, segment
            if segment.status in SKIP_STATUSES:
                return index, segment
            if not segment.script:
                segment.status = "needs_manual_review"
                segment.errors.append("Cannot synthesize without script metadata.")
                return index, segment
            target_language = _canonical_language(cfg.target_language)
            source_language = _canonical_language(cfg.source_language)
            open_sentence_normalized = False
            if target_language == "ko":
                if is_texture_like_micro_segment(segment, cfg):
                    segment.status = "non_speech_texture"
                    segment.keep_original_texture = True
                    segment.tts = None
                    segment.rvc = None
                    segment.qc = None
                    segment.mix = {}
                    segment.analysis["micro_segment_auto_fallback"] = {
                        "action": "keep_original_texture",
                        "reason": "texture_like_micro_segment",
                        "duration_sec": round(float(segment.duration), 6),
                        "backend": "gpt-sovits",
                    }
                    segment.analysis["ko_qc_repair_plan"] = build_ko_qc_repair_plan(
                        segment,
                        cfg=cfg,
                    )
                    return index, segment
                previous_tts_text = segment.script.tts_text
                normalized = normalize_korean_tts_text(previous_tts_text)
                normalized_text = normalized.text
                normalization_risk_flags = list(normalized.risk_flags)
                if normalized_text:
                    closed_text, open_sentence_normalized = _close_open_korean_tts_sentence(
                        normalized_text
                    )
                    if open_sentence_normalized:
                        normalized_text = closed_text
                        normalization_risk_flags = list(
                            dict.fromkeys([*normalization_risk_flags, "closed_open_sentence"])
                        )
                if normalized_text and normalized_text != previous_tts_text.strip():
                    segment.script.tts_text = normalized_text
                    segment.script.expected_tts_duration_sec = estimate_tts_duration(normalized_text, "ko")
                    segment.script.nonverbal_cues = [*segment.script.nonverbal_cues, *normalized.cues]
                    segment.script.risk_flags = list(
                        dict.fromkeys([*segment.script.risk_flags, *normalization_risk_flags])
                    )
                    segment.analysis["pre_synth_tts_text_normalization"] = {
                        "before": previous_tts_text,
                        "after": normalized_text,
                        "normalized_text": normalized.text,
                        "risk_flags": normalization_risk_flags,
                    }
                embedded_countdown_values = _embedded_countdown_values(segment)
                numeric_cadence_periodized = False
                if (
                    embedded_countdown_values is None
                    and bool(getattr(cfg, "gsv_numeric_cadence_periods_enabled", True))
                ):
                    periodized_text, periodization_metadata = periodize_korean_numeric_cadence_text(
                        segment.script.tts_text,
                        min_values=int(getattr(cfg, "gsv_numeric_cadence_min_values", 3)),
                    )
                    if (
                        periodization_metadata is not None
                        and periodized_text != segment.script.tts_text.strip()
                    ):
                        previous_tts_text = segment.script.tts_text
                        segment.script.tts_text = periodized_text
                        segment.script.expected_tts_duration_sec = estimate_tts_duration(
                            periodized_text, "ko"
                        )
                        segment.script.risk_flags = list(
                            dict.fromkeys(
                                [
                                    *segment.script.risk_flags,
                                    "korean_numeric_cadence_periodized",
                                ]
                            )
                        )
                        segment.analysis["pre_synth_tts_numeric_cadence_periodization"] = {
                            "before": previous_tts_text,
                            "after": periodized_text,
                            **periodization_metadata,
                        }
                        numeric_cadence_periodized = True
                compacted_text, counting_metadata = (
                    (segment.script.tts_text.strip(), None)
                    if embedded_countdown_values is not None or numeric_cadence_periodized
                    else _compact_korean_counting_tts_text(segment.script.tts_text)
                )
                if counting_metadata is not None and compacted_text != segment.script.tts_text.strip():
                    previous_tts_text = segment.script.tts_text
                    segment.script.tts_text = compacted_text
                    segment.script.expected_tts_duration_sec = estimate_tts_duration(
                        compacted_text, "ko"
                    )
                    segment.script.risk_flags = list(
                        dict.fromkeys(
                            [
                                *segment.script.risk_flags,
                                "korean_counting_tts_compacted",
                            ]
                        )
                    )
                    segment.analysis["pre_synth_tts_counting_compaction"] = {
                        "before": previous_tts_text,
                        "after": compacted_text,
                        **counting_metadata,
                    }
                preflight = preflight_tts_text(
                    segment.script,
                    target_language=target_language,
                    source_text=segment.source_script.text if segment.source_script else "",
                    min_hangul_ratio=cfg.gsv_ko_text_min_hangul_ratio,
                )
                if preflight.issues == ["korean_tts_suspicious_truncated_sentence"]:
                    repaired_text, repaired = repair_suspicious_truncated_korean_tts_text(
                        segment.script.tts_text
                    )
                    if repaired:
                        previous_tts_text = segment.script.tts_text
                        segment.script.tts_text = repaired_text
                        segment.script.expected_tts_duration_sec = estimate_tts_duration(
                            repaired_text,
                            "ko",
                        )
                        segment.script.risk_flags = list(
                            dict.fromkeys(
                                [
                                    *segment.script.risk_flags,
                                    "repaired_truncated_sentence",
                                ]
                            )
                        )
                        preflight = preflight_tts_text(
                            segment.script,
                            target_language=target_language,
                            source_text=segment.source_script.text if segment.source_script else "",
                            min_hangul_ratio=cfg.gsv_ko_text_min_hangul_ratio,
                        )
                        segment.analysis["pre_synth_text_qc_recovery"] = {
                            "action": "repaired_truncated_sentence",
                            "before": previous_tts_text,
                            "after": repaired_text,
                        }
                segment.analysis["pre_synth_text_qc"] = preflight.as_payload()
                if preflight.blocked:
                    segment.status = "needs_manual_review"
                    segment.errors.append(
                        "Korean TTS preflight blocked synthesis: " + ", ".join(preflight.issues)
                    )
                    return index, segment
            original_ref_style = segment.script.ref_style
            requested_ref_style = original_ref_style
            speaker_cfg = _gsv_speaker_cfg(cfg, segment)
            segment_refs = refs
            speaker_gpt_weights: str | None = None
            speaker_sovits_weights: str | None = None
            speaker_refs_path: Path | None = None
            if speaker_cfg is not None:
                if speaker_cfg.gpt_weights_path:
                    speaker_gpt_weights = str(
                        _resolve_gsv_speaker_path(project_dir, speaker_cfg.gpt_weights_path)
                    )
                speaker_sovits_weights = str(
                    _resolve_gsv_speaker_path(project_dir, speaker_cfg.sovits_weights_path)
                )
                speaker_refs_path = _resolve_gsv_speaker_path(project_dir, speaker_cfg.refs_path)
                cache_key = str(speaker_refs_path)
                with speaker_refs_cache_lock:
                    if cache_key not in speaker_refs_cache:
                        speaker_refs_cache[cache_key] = load_refs(speaker_refs_path, project_dir)
                    segment_refs = speaker_refs_cache[cache_key]
                if requested_ref_style not in segment_refs:
                    requested_ref_style = speaker_cfg.default_ref_style
            resolved_ref_style = requested_ref_style if requested_ref_style in segment_refs else "whisper_close"
            static_ref = resolve_ref(segment_refs, requested_ref_style)
            segment_ref, segment_ref_metadata = _segment_source_ref_for_gsv(
                project_dir,
                segment,
                cfg,
                manifest.segments,
            )
            static_ref_retry_active = segment.id in static_ref_retry_segment_ids
            if static_ref_retry_active:
                segment_ref_metadata = copy.deepcopy(segment_ref_metadata)
                segment_ref_metadata["static_ref_retry"] = True
                segment_ref_metadata["used_before_static_ref_retry"] = bool(
                    segment_ref_metadata.get("used")
                )
                segment_ref_metadata["used"] = False
                if segment_ref is not None:
                    segment_ref_metadata["disabled_by_synth_pass"] = "static_ref_retry"
                ref = static_ref
            else:
                ref = segment_ref or static_ref
            synthesis_ref = _ref_for_tts_language(ref, segment.script.tts_language)
            static_synthesis_ref = _ref_for_tts_language(static_ref, segment.script.tts_language)
            initial_segment_ref_active = (
                target_language == "ko"
                and segment_ref is not None
                and synthesis_ref.ref_audio_path == segment_ref.ref_audio_path
                and not static_ref_retry_active
            )
            segment_ref_clarity_enabled = bool(
                initial_segment_ref_active
                and getattr(cfg, "gsv_korean_segment_ref_clarity_profile_enabled", True)
            )
            fallback_used = resolved_ref_style != original_ref_style
            candidates: list[TTSCandidate] = []
            expected = segment.script.expected_tts_duration_sec or segment.duration
            duration_rewrite_controls_current_pass = duration_rewrite_enabled and (
                duration_rewrite_phase in {"initial", "post_rewrite", "before_zero_shot"}
            )
            speed = (
                1.0
                if duration_rewrite_controls_current_pass
                else suggest_speed_factor(
                    expected,
                    segment.duration,
                    minimum=cfg.gsv_tts_min_speed_factor,
                    maximum=cfg.gsv_tts_max_speed_factor,
                )
            )
            has_repetition_or_omission_signal = bool(
                segment.qc and (segment.qc.repetition_detected or segment.qc.omission_detected)
            )
            can_defer_duration_rewrite = (
                duration_rewrite_phase == "initial"
                and duration_rewrite_after_initial_enabled
                and _can_rewrite_script_for_duration(segment.script)
            )
            pass_candidate_count = candidate_count_for_synth_pass(synth_pass)
            pass_temperature = temperature_for_synth_pass()
            request_temperature = (
                float(getattr(cfg, "gsv_korean_segment_ref_temperature", 0.75))
                if segment_ref_clarity_enabled and pass_temperature_override is None
                else pass_temperature
            )
            request_top_k = (
                int(getattr(cfg, "gsv_korean_segment_ref_top_k", 8))
                if segment_ref_clarity_enabled and pass_top_k_override is None
                else top_k_for_synth_pass()
            )
            request_top_p = (
                float(getattr(cfg, "gsv_korean_segment_ref_top_p", 0.85))
                if segment_ref_clarity_enabled and pass_top_p_override is None
                else top_p_for_synth_pass()
            )
            request_parallel_infer = (
                bool(getattr(cfg, "gsv_korean_segment_ref_parallel_infer", False))
                if segment_ref_clarity_enabled and pass_parallel_infer_override is None
                else parallel_infer_for_synth_pass()
            )
            effective_candidate_count = (
                int(cfg.gsv_duration_rewrite_pre_candidate_count or pass_candidate_count)
                if can_defer_duration_rewrite
                else pass_candidate_count
            )
            pending_duration_rewrite = copy.deepcopy(
                segment.analysis.pop("pending_duration_rewrite", None)
            )
            pending_timing_expansion = copy.deepcopy(
                segment.analysis.pop("pending_timing_expansion", None)
            )
            ref_duration_failure = False
            for candidate_index in range(effective_candidate_count):
                seed = cfg.base_seed + index * 100 + candidate_index
                tts_text_language = _segment_tts_text_language(segment, target_language)
                max_attempts = (
                    1
                    if can_defer_duration_rewrite
                    else int(getattr(cfg, "gsv_max_attempts_per_candidate", 3))
                )
                if open_sentence_normalized:
                    max_attempts = max(max_attempts, 2)
                last_attempt = max_attempts - 1
                options = GPTSoVITSTTSOptions(
                    seed=seed,
                    speed_factor=speed,
                    text_lang=tts_text_language,
                    top_k=request_top_k,
                    top_p=request_top_p,
                    temperature=request_temperature,
                    text_split_method=cfg.gsv_text_split_method,
                    fragment_interval=cfg.gsv_fragment_interval,
                    parallel_infer=request_parallel_infer,
                    repetition_penalty=cfg.gsv_repetition_penalty,
                    sample_steps=cfg.gsv_sample_steps,
                    super_sampling=cfg.gsv_super_sampling,
                    overlap_length=cfg.gsv_overlap_length,
                    min_chunk_length=cfg.gsv_min_chunk_length,
                )
                attempt_signals: list[GPTSoVITSRetrySignal] = []
                if has_repetition_or_omission_signal:
                    options = adjust_for_repetition_or_omission(options, seed_step=10_000 + index)
                    attempt_signals.extend(
                        [
                            GPTSoVITSRetrySignal.REPETITION_OR_OMISSION,
                            GPTSoVITSRetrySignal.SEED_CHANGED,
                            GPTSoVITSRetrySignal.REPETITION_PENALTY_INCREASED,
                        ]
                    )
                attempt_text = segment.script.tts_text
                attempt_ref = synthesis_ref
                omission_retry_metadata: dict[str, Any] | None = None
                for attempt in range(max_attempts):
                    candidate_path = _tts_candidate_path(project_dir, segment.id, candidate_index, attempt)
                    current_ref = attempt_ref
                    using_segment_ref = (
                        segment_ref is not None
                        and current_ref.ref_audio_path == synthesis_ref.ref_audio_path
                    )
                    payload: dict[str, Any] = {
                        "speaker_id": segment.speaker_id,
                        "requested_ref_style": original_ref_style,
                        "resolved_ref_style": resolved_ref_style,
                        "fallback_used": fallback_used,
                        "ref_audio_path": current_ref.ref_audio_path,
                        "aux_ref_audio_paths": current_ref.aux_ref_audio_paths,
                        "prompt_text_policy": (
                            "use_static_reference_retry"
                            if static_ref_retry_active
                            else
                            "use_segment_source_reference"
                            if using_segment_ref
                            else "use_source_reference_prompt"
                        ),
                        "segment_ref": segment_ref_metadata,
                        "speaker_gpt_weights_path": speaker_gpt_weights,
                        "speaker_sovits_weights_path": speaker_sovits_weights,
                        "speaker_refs_path": str(speaker_refs_path) if speaker_refs_path else None,
                        "source_language": source_language,
                        "target_language": target_language,
                        "cross_lingual_voice_transfer": source_language != target_language,
                        "expected_tts_duration_sec": expected,
                        "target_duration_sec": segment.duration,
                        "synth_pass": synth_pass,
                        "candidate_count_used": effective_candidate_count,
                        "temperature_used": request_temperature,
                        "segment_ref_clarity_profile": {
                            "enabled": bool(segment_ref_clarity_enabled and using_segment_ref),
                            "warn_blocks_mix": bool(
                                getattr(cfg, "gsv_korean_segment_ref_warn_blocks_mix", True)
                            ),
                            "temperature": request_temperature,
                            "top_k": request_top_k,
                            "top_p": request_top_p,
                            "parallel_infer": request_parallel_infer,
                        },
                        "lane_index": lane_index,
                        "gsv_url": None if mock else gsv_base_urls[lane_index],
                        "retry": {
                            "attempt": attempt,
                            "max_attempts": max_attempts,
                            "signals": retry_signal_values(attempt_signals),
                        },
                    }
                    payload.update(_tts_request_debug_payload(attempt_text, current_ref, options))
                    if omission_retry_metadata is not None:
                        payload["omission_retry"] = copy.deepcopy(omission_retry_metadata)
                    if pending_duration_rewrite is not None:
                        payload["duration_rewrite"] = copy.deepcopy(pending_duration_rewrite)
                    if pending_timing_expansion is not None:
                        payload["timing_expansion"] = copy.deepcopy(pending_timing_expansion)
                    if mock:
                        mock_duration = max(0.05, segment.duration)
                        _mock_synthesize(candidate_path, mock_duration, options.seed, cfg.mix_sample_rate)
                        postprocess_tts_candidate(candidate_path, payload)
                        duration = duration_sec(candidate_path)
                        peak = peak_dbfs(candidate_path)
                        rms = rms_dbfs(candidate_path)
                        payload.update(
                            {
                                "mock": True,
                                "repetition_penalty": options.repetition_penalty,
                            }
                        )
                        candidate_backend_name = "mock"
                    else:
                        client = clients[lane_index]
                        try:
                            speaker_switch: dict[str, Any] = {}
                            if speaker_gpt_weights and lane_gpt_weights[lane_index] != speaker_gpt_weights:
                                response = client.set_gpt_weights(speaker_gpt_weights)
                                lane_gpt_weights[lane_index] = speaker_gpt_weights
                                speaker_switch.update(
                                    {
                                        "lane_index": lane_index,
                                        "speaker_id": segment.speaker_id,
                                        "gpt_weights_path": speaker_gpt_weights,
                                        "gpt_response": response,
                                    }
                                )
                            if speaker_sovits_weights and lane_sovits_weights[lane_index] != speaker_sovits_weights:
                                response = client.set_sovits_weights(speaker_sovits_weights)
                                lane_sovits_weights[lane_index] = speaker_sovits_weights
                                speaker_switch.update(
                                    {
                                        "lane_index": lane_index,
                                        "speaker_id": segment.speaker_id,
                                        "sovits_weights_path": speaker_sovits_weights,
                                        "sovits_response": response,
                                    }
                                )
                            if speaker_switch:
                                model_switch.setdefault("speaker_switches", []).append(speaker_switch)
                            request = client.build_payload(attempt_text, current_ref, options)
                            payload.update(request.as_payload())
                            client.synthesize_to_file(request, candidate_path)
                            postprocess_tts_candidate(candidate_path, payload)
                            duration = duration_sec(candidate_path)
                            peak = peak_dbfs(candidate_path)
                            rms = rms_dbfs(candidate_path)
                        except GPTSoVITSError as exc:
                            if _gsv_invalid_ref_duration_error(exc):
                                payload["ref_failure"] = {
                                    "kind": "invalid_ref_duration",
                                    "ref_audio_path": current_ref.ref_audio_path,
                                    "message": str(exc),
                                    "action": "stop_repeating_ref_candidates",
                                }
                                payload["retry"]["next_action"] = "stop_invalid_ref_duration"
                                ref_duration_failure = True
                            candidates.append(
                                TTSCandidate(
                                    candidate_index=candidate_index,
                                    seed=options.seed,
                                    payload=payload,
                                    output_path=str(candidate_path),
                                    backend="gpt-sovits",
                                    error=str(exc),
                                )
                            )
                            break
                        candidate_backend_name = "gpt-sovits"
                    if ref_duration_failure:
                        break
                    too_long = duration_too_long(duration, segment.duration, cfg.duration_tolerance)
                    too_short = duration_too_short(duration, segment.duration, cfg.duration_tolerance)
                    candidate_ratio = duration_ratio(duration, segment.duration)
                    duration_gate = "too_long" if too_long else "too_short" if too_short else "pass"
                    timing_quality_gate = _gsv_timing_quality_gate(
                        candidate_ratio,
                        duration_gate,
                        float(getattr(cfg, "gsv_timing_quality_tolerance", 0.10)),
                    )
                    audio_gate = "silent" if peak <= -90.0 or rms <= -90.0 else "pass"
                    payload["duration_ratio"] = candidate_ratio
                    payload["duration_gate"] = duration_gate
                    payload["timing_quality"] = _gsv_timing_quality_payload(
                        candidate_ratio,
                        duration_gate,
                        float(getattr(cfg, "gsv_timing_quality_tolerance", 0.10)),
                        float(cfg.duration_tolerance),
                    )
                    payload["audio_qc"] = {
                        "gate": audio_gate,
                        "peak_dbfs": round(peak, 3),
                        "rms_dbfs": round(rms, 3),
                    }
                    language_contract_ok = True
                    if target_language == "ko":
                        language_contract_ok = (
                            payload.get("text") == attempt_text
                            and payload.get("text_lang") == "all_ko"
                            and payload.get("prompt_lang") == "all_ja"
                        )
                    acceptable_for_mix = (
                        duration_gate == "pass"
                        and audio_gate == "pass"
                        and language_contract_ok
                    )
                    omission_reasons = _gsv_omission_detection_reasons(
                        duration_sec=duration,
                        target_duration_sec=segment.duration,
                        expected_tts_duration_sec=expected,
                        duration_gate=duration_gate,
                        audio_gate=audio_gate,
                        language_contract_ok=language_contract_ok,
                    )
                    if omission_reasons:
                        payload["omission_suspected"] = True
                        payload["omission_detection"] = {
                            "reasons": omission_reasons,
                            "duration_sec": round(duration, 6),
                            "expected_tts_duration_sec": round(expected, 6),
                            "target_duration_sec": round(segment.duration, 6),
                        }
                    selection_score = max(0.0, 1.0 - min(abs(candidate_ratio - 1.0), 1.0))
                    if audio_gate != "pass" and attempt < last_attempt:
                        payload["retry"]["next_action"] = GPTSoVITSRetrySignal.SEED_CHANGED.value
                    elif omission_reasons and attempt < last_attempt:
                        payload["retry"]["next_action"] = GPTSoVITSRetrySignal.REPETITION_OR_OMISSION.value
                    elif (too_long or too_short) and attempt < last_attempt:
                        payload["retry"]["next_action"] = (
                            GPTSoVITSRetrySignal.SPEED_FACTOR_ADJUSTED.value
                            if attempt == 0
                            else GPTSoVITSRetrySignal.SEED_CHANGED.value
                        )
                    candidates.append(
                        TTSCandidate(
                            candidate_index=candidate_index,
                            seed=options.seed,
                            payload=payload,
                            output_path=str(candidate_path),
                            duration_sec=duration,
                            backend=candidate_backend_name,
                            duration_ratio=candidate_ratio,
                            duration_gate=duration_gate,
                            timing_quality_gate=timing_quality_gate,
                            acceptable_for_mix=acceptable_for_mix,
                            selection_score=selection_score,
                            selection_reason=(
                                "duration_and_language_contract_pass"
                                if acceptable_for_mix
                                else "audio_qc_failed"
                                if audio_gate != "pass"
                                else "omission_suspected"
                                if omission_reasons
                                else "duration_or_language_contract_failed"
                            ),
                            retry_summary=payload["retry"],
                        )
                    )
                    if audio_gate != "pass":
                        if attempt >= last_attempt:
                            break
                        options = options.model_copy(
                            update={
                                "seed": options.seed + 30_000 + index + attempt
                                if options.seed >= 0
                                else 30_000 + index + attempt
                            }
                        )
                        attempt_signals = [GPTSoVITSRetrySignal.SEED_CHANGED]
                        continue
                    if omission_reasons:
                        if attempt >= last_attempt:
                            break
                        retry_text, closed_for_retry = _close_open_korean_tts_sentence(attempt_text)
                        attempt_text = retry_text
                        use_static_ref = (
                            segment_ref is not None
                            and current_ref.ref_audio_path != static_synthesis_ref.ref_audio_path
                        )
                        attempt_ref = static_synthesis_ref if use_static_ref else current_ref
                        options = adjust_for_repetition_or_omission(
                            options,
                            seed_step=40_000 + index + attempt,
                        )
                        attempt_signals = [
                            GPTSoVITSRetrySignal.REPETITION_OR_OMISSION,
                            GPTSoVITSRetrySignal.SEED_CHANGED,
                            GPTSoVITSRetrySignal.REPETITION_PENALTY_INCREASED,
                        ]
                        omission_retry_metadata = {
                            "trigger": "omission_suspected",
                            "source_candidate_index": candidate_index,
                            "source_attempt": attempt,
                            "source_duration_sec": round(duration, 6),
                            "ref_fallback": "static_ref" if use_static_ref else "same_ref",
                            "text_normalization": (
                                "closed_open_sentence"
                                if open_sentence_normalized or closed_for_retry
                                else "unchanged"
                            ),
                            "reasons": omission_reasons,
                        }
                        continue
                    if not (too_long or too_short):
                        break
                    if attempt >= last_attempt:
                        break
                    if attempt == 0:
                        options = (
                            adjust_speed_for_duration(
                                options,
                                duration,
                                segment.duration,
                                maximum=cfg.gsv_tts_max_speed_factor,
                            )
                            if too_long
                            else adjust_speed_for_short_duration(
                                options,
                                duration,
                                segment.duration,
                                minimum=cfg.gsv_tts_min_speed_factor,
                            )
                        )
                        attempt_signals = [
                            GPTSoVITSRetrySignal.DURATION_TOO_LONG
                            if too_long
                            else GPTSoVITSRetrySignal.DURATION_TOO_SHORT,
                            GPTSoVITSRetrySignal.SPEED_FACTOR_ADJUSTED,
                        ]
                        continue
                    duration_signal = (
                        GPTSoVITSRetrySignal.DURATION_TOO_LONG
                        if too_long
                        else GPTSoVITSRetrySignal.DURATION_TOO_SHORT
                    )
                    options = options.model_copy(
                        update={
                            "seed": options.seed + 20_000 + index + attempt
                            if options.seed >= 0
                            else 20_000 + index + attempt
                        }
                    )
                    attempt_signals = [
                        duration_signal,
                        GPTSoVITSRetrySignal.SEED_CHANGED,
                    ]
                if ref_duration_failure:
                    break
            successful = [
                candidate for candidate in candidates if not candidate.error and candidate.duration_sec is not None
            ]
            if not successful:
                segment.tts = TTSMetadata(
                    backend="mock" if mock else "gpt-sovits",
                    ref_style=resolved_ref_style,
                    speed_factor=speed,
                    candidate_count=effective_candidate_count,
                    candidates=candidates,
                    source_language=source_language,
                    target_language=target_language,
                    cross_lingual_voice_transfer=source_language != target_language,
                )
                segment.status = "failed"
                segment.errors.append("All TTS candidates failed.")
                return index, segment
            acceptable = [candidate for candidate in successful if candidate.acceptable_for_mix]
            audible = [
                candidate
                for candidate in successful
                if candidate.payload.get("audio_qc", {}).get("gate") == "pass"
            ]
            _update_gsv_candidate_selection_scores(successful, segment)

            def apply_selected_pronunciation_qc(candidate: TTSCandidate) -> bool:
                if isinstance(candidate.payload.get("pronunciation_qc"), dict):
                    pronunciation_qc = candidate.payload["pronunciation_qc"]
                else:
                    expected_text = str(candidate.payload.get("text") or segment.script.tts_text)
                    pronunciation_qc = run_pronunciation_qc(
                        segment=segment,
                        candidate_path=Path(candidate.output_path),
                        expected_text=expected_text,
                        candidate_duration_sec=float(candidate.duration_sec or 0.0),
                        worker_slot=lane_index,
                    )
                    if pronunciation_qc is not None:
                        candidate.payload["pronunciation_qc"] = pronunciation_qc
                pronunciation_gate = str(
                    (pronunciation_qc or {}).get("gate") or "pass"
                ).strip().lower()
                if isinstance(candidate.payload.get("numeric_sequence_qc"), dict):
                    numeric_sequence_qc = candidate.payload["numeric_sequence_qc"]
                else:
                    numeric_sequence_qc = run_numeric_sequence_qc(
                        expected_text=str(candidate.payload.get("text") or segment.script.tts_text),
                        pronunciation_qc=pronunciation_qc,
                    )
                    if numeric_sequence_qc is not None:
                        candidate.payload["numeric_sequence_qc"] = numeric_sequence_qc
                numeric_gate = str(
                    (numeric_sequence_qc or {}).get("gate") or "pass"
                ).strip().lower()
                if (
                    numeric_gate == "fail"
                    and bool(getattr(cfg, "gsv_numeric_sequence_qc_failure_blocks_mix", True))
                ):
                    candidate.acceptable_for_mix = False
                    candidate.selection_score = min(float(candidate.selection_score or 0.0), 0.25)
                    candidate.selection_reason = "numeric_sequence_qc_failed"
                    return False
                using_segment_ref = bool(candidate.payload.get("segment_ref", {}).get("used"))
                pronunciation_contract_ok = segment_ref_pronunciation_contract_ok(
                    target_language=target_language,
                    using_segment_ref=using_segment_ref,
                    pronunciation_gate=pronunciation_gate,
                )
                if pronunciation_contract_ok:
                    return True
                candidate.acceptable_for_mix = False
                candidate.selection_score = min(float(candidate.selection_score or 0.0), 0.25)
                candidate.selection_reason = (
                    "pronunciation_qc_failed"
                    if pronunciation_gate == "fail"
                    else "segment_ref_pronunciation_qc_warn"
                )
                return False

            def select_with_deferred_pronunciation_qc(selected: TTSCandidate) -> TTSCandidate:
                rejected_candidate_ids: set[int] = set()
                while True:
                    if apply_selected_pronunciation_qc(selected):
                        return selected
                    rejected_candidate_ids.add(id(selected))
                    selectable = [
                        candidate
                        for candidate in candidates
                        if id(candidate) not in rejected_candidate_ids
                        and not candidate.error
                        and candidate.duration_sec is not None
                        and candidate.acceptable_for_mix
                    ]
                    if not selectable:
                        return selected
                    timing_good = [
                        candidate for candidate in selectable if _gsv_timing_quality_is_good(candidate)
                    ]
                    selected = _select_gsv_candidate_for_mix(timing_good or selectable, segment)

            if not audible:
                segment.tts = TTSMetadata(
                    backend="mock" if mock else "gpt-sovits",
                    ref_style=resolved_ref_style,
                    speed_factor=speed,
                    candidate_count=effective_candidate_count,
                    candidates=candidates,
                    source_language=source_language,
                    target_language=target_language,
                    cross_lingual_voice_transfer=source_language != target_language,
                    retry_summary={"acceptable_candidates": 0},
                )
                segment.status = "failed"
                segment.errors.append("No acceptable TTS candidates for mix.")
                return index, segment
            embedded_overlay_rescue = False
            if acceptable:
                timing_good = [
                    candidate for candidate in acceptable if _gsv_timing_quality_is_good(candidate)
                ]
                selected = _select_gsv_candidate_for_mix(timing_good or acceptable, segment)
                if not _gsv_timing_quality_is_good(selected):
                    fitted = time_fit_candidate_if_needed(
                        segment,
                        selected,
                        output_label="timing_quality_timefit",
                        allow_acceptable=True,
                        selection_reason_if_acceptable="timing_quality_timefit",
                        selection_reason_if_failed="timing_quality_timefit_failed",
                    )
                    if fitted is not None:
                        candidates.append(fitted)
                        _update_gsv_candidate_selection_scores([fitted], segment)
                        if fitted.acceptable_for_mix and (
                            _gsv_timing_quality_is_good(fitted)
                            or not _gsv_timing_quality_is_good(selected)
                        ):
                            selected = fitted
                    if not _gsv_timing_quality_is_good(selected):
                        selected.payload.setdefault("timing_quality", {})[
                            "accepted_with_warning"
                        ] = True
            else:
                selected = _select_gsv_candidate_for_mix(audible, segment)
                fitted = time_fit_candidate_if_needed(segment, selected)
                if fitted is not None:
                    candidates.append(fitted)
                    _update_gsv_candidate_selection_scores([fitted], segment)
                    if fitted.acceptable_for_mix:
                        selected = fitted
                if not selected.acceptable_for_mix:
                    rescued = rescue_with_relaxed_duration_gate(segment, audible)
                    if rescued is not None:
                        candidates.append(rescued)
                        _update_gsv_candidate_selection_scores([rescued], segment)
                        selected = rescued
                if not selected.acceptable_for_mix:
                    rescued = rescue_with_source_pause_padding(segment, audible)
                    if rescued is not None:
                        candidates.append(rescued)
                        _update_gsv_candidate_selection_scores([rescued], segment)
                        selected = rescued
                if not selected.acceptable_for_mix:
                    rescued = rescue_with_relaxed_time_fit(segment, audible, candidates)
                    if rescued is not None:
                        _update_gsv_candidate_selection_scores([rescued], segment)
                        selected = rescued
                if not selected.acceptable_for_mix and bool(
                    getattr(cfg, "gsv_countdown_hybrid_apply_to_synth", True)
                ):
                    hybrid_metadata = segment.analysis.get("embedded_countdown_hybrid_renderer")
                    if (
                        isinstance(hybrid_metadata, dict)
                        and hybrid_metadata.get("status") == "rendered"
                        and Path(str(hybrid_metadata.get("bed_path") or "")).exists()
                    ):
                        embedded_overlay_rescue = True
                if not selected.acceptable_for_mix:
                    micro_summary = micro_segment_unfit_summary(segment, selected, audible)
                    if micro_summary is not None and not embedded_overlay_rescue:
                        unfit_policy = str(
                            getattr(cfg, "gsv_micro_segment_unfit_policy", "keep_original")
                        )
                        rescue_status = (
                            "micro_segment_keep_original"
                            if unfit_policy == "keep_original"
                            else "micro_segment_manual_review"
                        )
                        segment.tts = TTSMetadata(
                            backend="mock" if mock else "gpt-sovits",
                            ref_style=resolved_ref_style,
                            speed_factor=speed,
                            candidate_count=effective_candidate_count,
                            candidates=candidates,
                            source_language=source_language,
                            target_language=target_language,
                            cross_lingual_voice_transfer=source_language != target_language,
                            retry_summary={
                                **micro_summary,
                                "rescue_status": rescue_status,
                                "micro_segment_unfit_policy": unfit_policy,
                            },
                        )
                        fallback_backend = str(
                            getattr(cfg, "gsv_micro_segment_fallback_backend", "qwen")
                        ).strip().lower().replace("-", "_")
                        if unfit_policy == "keep_original":
                            segment.status = "absorbed"
                            segment.keep_original_texture = True
                            segment.analysis["micro_segment_auto_fallback"] = {
                                "action": "keep_original_micro_segment",
                                "reason": "tts_duration_unfit",
                                "selected_duration_gate": micro_summary["selected_duration_gate"],
                                "selected_duration_ratio": micro_summary["selected_duration_ratio"],
                                "target_duration_sec": micro_summary["target_duration_sec"],
                            }
                        else:
                            segment.status = "needs_manual_review"
                            if (
                                not mock
                                and target_language == "ko"
                                and fallback_backend == "qwen"
                            ):
                                repair_plan = build_ko_qc_repair_plan(segment, cfg=cfg)
                                repair_plan.update(
                                    {
                                        "action": "fallback_tts_qwen",
                                        "root_cause": "gpt_sovits_micro_segment_unfit",
                                        "terminal_manual": False,
                                        "route": "micro_fallback",
                                        "source": "gpt_sovits_micro_segment",
                                        "gsv_retry_summary": micro_summary,
                                    }
                                )
                                segment.analysis["ko_qc_repair_plan"] = repair_plan
                            segment.errors.append("Micro segment too short for Korean TTS.")
                        return index, segment
                if not selected.acceptable_for_mix and not embedded_overlay_rescue:
                    segment.tts = TTSMetadata(
                        backend="mock" if mock else "gpt-sovits",
                        ref_style=resolved_ref_style,
                        speed_factor=speed,
                        candidate_count=effective_candidate_count,
                        candidates=candidates,
                        source_language=source_language,
                        target_language=target_language,
                        cross_lingual_voice_transfer=source_language != target_language,
                        retry_summary={
                            "acceptable_candidates": 0,
                            "selected_duration_gate": selected.duration_gate,
                            "selected_duration_ratio": selected.duration_ratio,
                            "selected_timing_quality_gate": selected.timing_quality_gate,
                            "timing_quality_tolerance": float(
                                getattr(cfg, "gsv_timing_quality_tolerance", 0.10)
                            ),
                            "selected_pronunciation_gate": (
                                selected.payload.get("pronunciation_qc", {}).get("gate")
                                if isinstance(selected.payload.get("pronunciation_qc"), dict)
                                else None
                            ),
                            "selected_numeric_sequence_gate": (
                                selected.payload.get("numeric_sequence_qc", {}).get("gate")
                                if isinstance(selected.payload.get("numeric_sequence_qc"), dict)
                                else None
                            ),
                        },
                    )
                    segment.status = "failed"
                    segment.errors.append("No acceptable TTS candidates for mix.")
                    return index, segment
            selected = select_with_deferred_pronunciation_qc(selected)
            if not selected.acceptable_for_mix and not embedded_overlay_rescue:
                segment.tts = TTSMetadata(
                    backend="mock" if mock else "gpt-sovits",
                    ref_style=resolved_ref_style,
                    speed_factor=speed,
                    candidate_count=effective_candidate_count,
                    candidates=candidates,
                    source_language=source_language,
                    target_language=target_language,
                    cross_lingual_voice_transfer=source_language != target_language,
                    retry_summary={
                        "acceptable_candidates": 0,
                        "selected_duration_gate": selected.duration_gate,
                        "selected_duration_ratio": selected.duration_ratio,
                        "selected_timing_quality_gate": selected.timing_quality_gate,
                        "timing_quality_tolerance": float(
                            getattr(cfg, "gsv_timing_quality_tolerance", 0.10)
                        ),
                        "selected_pronunciation_gate": (
                            selected.payload.get("pronunciation_qc", {}).get("gate")
                            if isinstance(selected.payload.get("pronunciation_qc"), dict)
                            else None
                        ),
                        "selected_numeric_sequence_gate": (
                            selected.payload.get("numeric_sequence_qc", {}).get("gate")
                            if isinstance(selected.payload.get("numeric_sequence_qc"), dict)
                            else None
                        ),
                    },
                )
                segment.status = "failed"
                segment.errors.append("No acceptable TTS candidates for mix.")
                return index, segment
            selected.selected = True
            final_path = project_dir / "work" / "tts" / f"{segment.id}_final.wav"
            ensure_not_same_path(Path(selected.output_path), final_path)
            final_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(selected.output_path, final_path)
            selected = apply_embedded_countdown_hybrid_overlay(
                segment=segment,
                selected=selected,
                final_path=final_path,
                candidates=candidates,
            )
            segment.tts = TTSMetadata(
                backend="mock" if mock else "gpt-sovits",
                ref_style=resolved_ref_style,
                speed_factor=float(selected.payload.get("speed_factor", speed)),
                candidate_count=effective_candidate_count,
                selected_candidate_path=str(final_path),
                candidates=candidates,
                source_language=source_language,
                target_language=target_language,
                cross_lingual_voice_transfer=source_language != target_language,
                retry_summary={
                    "selected_duration_gate": selected.duration_gate,
                    "selected_acceptable_for_mix": selected.acceptable_for_mix,
                    "selected_duration_ratio": selected.duration_ratio,
                    "selected_timing_quality_gate": selected.timing_quality_gate,
                    "timing_quality_tolerance": float(
                        getattr(cfg, "gsv_timing_quality_tolerance", 0.10)
                    ),
                    "selected_pronunciation_gate": (
                        selected.payload.get("pronunciation_qc", {}).get("gate")
                        if isinstance(selected.payload.get("pronunciation_qc"), dict)
                        else None
                    ),
                    "selected_numeric_sequence_gate": (
                        selected.payload.get("numeric_sequence_qc", {}).get("gate")
                        if isinstance(selected.payload.get("numeric_sequence_qc"), dict)
                        else None
                    ),
                },
            )
            segment.status = "synthesized"
            return index, segment

        def synthesize_segment(index: int, segment: Segment, lane_index: int) -> tuple[int, Segment]:
            if mock or (
                segment.status in {*SKIP_STATUSES, "synthesized"}
                and not should_reset_previous_tts(segment)
            ) or not segment.script:
                return synthesize_segment_locked(index, segment, lane_index)
            with lane_locks[lane_index]:
                return synthesize_segment_locked(index, segment, lane_index)

        def run_synth_jobs(segment_jobs: list[tuple[int, Segment, int]]) -> None:
            nonlocal last_logged_at
            if not segment_jobs:
                return
            start_gsv_servers()
            job_total = len(segment_jobs)
            completed_jobs = 0
            if not mock and gsv_lane_count > 1 and len(segment_jobs) > 1:
                with ThreadPoolExecutor(max_workers=gsv_lane_count) as executor:
                    futures = [
                        executor.submit(synthesize_segment, index, segment, lane_index)
                        for index, segment, lane_index in segment_jobs
                    ]
                    for future in as_completed(futures):
                        index, segment = future.result()
                        completed_jobs += 1
                        save_manifest(project_dir, manifest)
                        last_logged_at = _log_segment_progress(
                            stage_name,
                            index,
                            job_total,
                            segment,
                            manifest,
                            started_at,
                            last_logged_at,
                            progress_index=completed_jobs,
                            counts_label="status_counts",
                        )
            else:
                for index, segment, lane_index in segment_jobs:
                    index, segment = synthesize_segment(index, segment, lane_index)
                    completed_jobs += 1
                    save_manifest(project_dir, manifest)
                    last_logged_at = _log_segment_progress(
                        stage_name,
                        index,
                        job_total,
                        segment,
                        manifest,
                        started_at,
                        last_logged_at,
                        progress_index=completed_jobs,
                        counts_label="status_counts",
                    )

        def segment_has_invalid_ref_duration_failure(segment: Segment) -> bool:
            if segment.tts is None:
                return False
            return any(
                isinstance(candidate.payload.get("ref_failure"), dict)
                and candidate.payload.get("ref_failure", {}).get("kind")
                == "invalid_ref_duration"
                for candidate in segment.tts.candidates
            )

        def selected_failed_segment_ids(
            *,
            include_invalid_ref_duration: bool = False,
        ) -> list[str]:
            return [
                segment.id
                for segment in manifest.segments
                if segment.status == "failed"
                and (only_segment_ids is None or segment.id in only_segment_ids)
                and failed_segment_in_stage_scope(segment)
                and (
                    include_invalid_ref_duration
                    or not segment_has_invalid_ref_duration_failure(segment)
                )
            ]

        def failed_segment_in_stage_scope(segment: Segment) -> bool:
            is_countdown = detected_countdown_segment_values(segment) is not None
            if countdown_only:
                return is_countdown
            return not (not render_countdowns and is_countdown)

        def failed_segment_ids_with_used_segment_ref(segment_ids: list[str]) -> list[str]:
            target_ids = set(segment_ids)
            return [
                segment.id
                for segment in manifest.segments
                if segment.id in target_ids
                and segment.status == "failed"
                and segment.tts is not None
                and any(
                    bool(candidate.payload.get("segment_ref", {}).get("used"))
                    for candidate in segment.tts.candidates
                )
            ]

        def failed_segment_ids_with_pronunciation_failure(segment_ids: list[str]) -> list[str]:
            target_ids = set(segment_ids)
            return [
                segment.id
                for segment in manifest.segments
                if segment.id in target_ids
                and segment.status == "failed"
                and segment.tts is not None
                and any(
                    (
                        isinstance(candidate.payload.get("pronunciation_qc"), dict)
                        and str(
                            candidate.payload.get("pronunciation_qc", {}).get("gate", "")
                        ).strip().lower()
                        == "fail"
                    )
                    or (
                        isinstance(candidate.payload.get("numeric_sequence_qc"), dict)
                        and str(
                            candidate.payload.get("numeric_sequence_qc", {}).get("gate", "")
                        ).strip().lower()
                        == "fail"
                    )
                    for candidate in segment.tts.candidates
                )
            ]

        def jobs_for_segment_ids(segment_ids: set[str]) -> list[tuple[int, Segment, int]]:
            return [
                (index, segment, _segment_lane_index(segment, index - 1, gsv_lane_count))
                for index, segment in enumerate(manifest.segments, start=1)
                if segment.id in segment_ids
            ]

        def reset_failed_segments_for_internal_retry(
            segment_ids: set[str],
            *,
            pass_name: str,
        ) -> None:
            for segment in manifest.segments:
                if segment.id not in segment_ids or segment.status != "failed" or not segment.script:
                    continue
                segment.analysis.setdefault("synth_internal_retry_history", []).append(
                    {
                        "pass": pass_name,
                        "previous_errors": list(segment.errors),
                        "previous_duration_gate": (
                            segment.tts.retry_summary.get("selected_duration_gate")
                            if segment.tts
                            else None
                        ),
                    }
                )
                reset_previous_tts_attempt(segment)

        def run_internal_retry_pass(
            *,
            pass_name: str,
            segment_ids: list[str],
            zero_shot: bool = False,
            static_ref_retry: bool = False,
            candidate_count_override: int | None = None,
            temperature_override: float | None = None,
            top_k_override: int | None = None,
            top_p_override: float | None = None,
            parallel_infer_override: bool | None = None,
        ) -> dict[str, Any] | None:
            nonlocal gpt_weights, sovits_weights, synth_pass, internal_retry_segment_ids
            nonlocal static_ref_retry_segment_ids
            nonlocal pass_candidate_count_override, pass_temperature_override
            nonlocal pass_top_k_override, pass_top_p_override, pass_parallel_infer_override
            if mock or not segment_ids:
                return None
            target_ids = set(segment_ids)
            if zero_shot:
                stop_gsv_servers()
                gpt_weights = None
                sovits_weights = None
                model_switch.setdefault("zero_shot_fallback", {})["weights_policy"] = "unchanged"
            internal_retry_segment_ids = target_ids
            static_ref_retry_segment_ids = target_ids if static_ref_retry else set()
            pass_candidate_count_override = candidate_count_override
            pass_temperature_override = temperature_override
            pass_top_k_override = top_k_override
            pass_top_p_override = top_p_override
            pass_parallel_infer_override = parallel_infer_override
            reset_failed_segments_for_internal_retry(target_ids, pass_name=pass_name)
            synth_pass = pass_name
            try:
                run_synth_jobs(jobs_for_segment_ids(target_ids))
            finally:
                internal_retry_segment_ids = set()
                static_ref_retry_segment_ids = set()
                pass_candidate_count_override = None
                pass_temperature_override = None
                pass_top_k_override = None
                pass_top_p_override = None
                pass_parallel_infer_override = None
            failed_after = [
                segment_id
                for segment_id in selected_failed_segment_ids()
                if segment_id in target_ids
            ]
            succeeded = [segment_id for segment_id in segment_ids if segment_id not in failed_after]
            summary: dict[str, Any] = {
                "attempted_segments": segment_ids,
                "succeeded_segments": succeeded,
                "failed_segments": failed_after,
                "candidate_count": (
                    candidate_count_override
                    if candidate_count_override is not None
                    else candidate_count_for_synth_pass(pass_name)
                ),
            }
            if temperature_override is not None:
                summary["temperature"] = temperature_override
            if top_k_override is not None:
                summary["top_k"] = top_k_override
            if top_p_override is not None:
                summary["top_p"] = top_p_override
            if parallel_infer_override is not None:
                summary["parallel_infer"] = parallel_infer_override
            if zero_shot:
                summary["server_restarted_for_zero_shot"] = should_auto_start_server
                model_switch["zero_shot_fallback"] = {
                    **model_switch.get("zero_shot_fallback", {}),
                    **summary,
                }
            return summary

        def duration_rewrite_request_for_segment(
            index: int,
            segment: Segment,
        ) -> dict[str, Any] | None:
            if (
                not duration_rewrite_enabled
                or not segment.script
                or not _can_rewrite_script_for_duration(segment.script)
                or not segment.tts
                or segment.status not in {"failed", "needs_manual_review"}
            ):
                return None
            candidates = [
                candidate
                for candidate in segment.tts.candidates
                if not candidate.error
                and candidate.duration_sec is not None
                and candidate.duration_gate in {"too_long", "too_short"}
                and candidate.payload.get("audio_qc", {}).get("gate") == "pass"
            ]
            if not candidates:
                return None
            candidate = min(
                candidates,
                key=lambda item: abs((item.duration_ratio or 0.0) - 1.0),
            )
            return {
                "index": index,
                "segment": segment,
                "attempt_text": str(candidate.payload.get("text") or segment.script.tts_text),
                "actual_duration_sec": float(candidate.duration_sec or 0.0),
                "reason": candidate.duration_gate,
            }

        def reset_segment_for_duration_rewrite_retry(
            segment: Segment,
            rewrite_metadata: dict[str, Any],
        ) -> None:
            segment.analysis.setdefault("duration_rewrite_history", []).append(
                copy.deepcopy(rewrite_metadata)
            )
            segment.analysis["pending_duration_rewrite"] = copy.deepcopy(rewrite_metadata)
            segment.status = "scripted"
            segment.tts = None
            segment.errors = [
                error
                for error in segment.errors
                if error
                not in {
                    "No acceptable TTS candidates for mix.",
                    "All TTS candidates failed.",
                    "Micro segment too short for Korean TTS.",
                }
                and not error.startswith("GPT-SoVITS synthesis failed")
                and not error.startswith("Korean TTS preflight blocked synthesis")
            ]

        def record_skipped_duration_rewrite_retry(
            segment: Segment,
            rewrite_metadata: dict[str, Any],
        ) -> None:
            segment.analysis.setdefault("duration_rewrite_history", []).append(
                copy.deepcopy(rewrite_metadata)
            )
            segment.analysis["duration_rewrite_retry_skipped"] = copy.deepcopy(rewrite_metadata)
            segment.analysis.pop("pending_duration_rewrite", None)

        def run_deferred_duration_rewrites(
            *,
            segment_ids: set[str] | None = None,
            phase: str,
        ) -> tuple[set[str], dict[str, Any] | None]:
            nonlocal duration_rewrite_client, duration_rewrite_running
            requests = [
                request
                for index, segment in enumerate(manifest.segments, start=1)
                if only_segment_ids is None or segment.id in only_segment_ids
                if segment_ids is None or segment.id in segment_ids
                for request in [duration_rewrite_request_for_segment(index, segment)]
                if request is not None
            ]
            if not requests:
                return set(), None
            stop_gsv_servers()
            if duration_rewrite_manager is not None:
                duration_rewrite_manager.start()
                duration_rewrite_running = True
            duration_rewrite_client = LlamaServerTranslationClient(
                duration_rewrite_base_url,
                timeout_sec=cfg.gemma_text_timeout_sec,
                retries=cfg.gemma_text_retries,
                n_predict=cfg.gemma_text_n_predict,
                model=cfg.gemma_llama_cpp_model_path,
                two_pass=False,
            )
            retry_segment_ids: set[str] = set()
            skipped_segment_ids: set[str] = set()
            try:
                for request in requests:
                    segment = request["segment"]
                    rewritten, rewrite_metadata = maybe_rewrite_script_with_gemma(
                        segment=segment,
                        index=int(request["index"]),
                        attempt_text=str(request["attempt_text"]),
                        actual_duration_sec=float(request["actual_duration_sec"]),
                        reason=str(request["reason"]),
                        rewrite_attempts_used=0,
                    )
                    rewrite_metadata["deferred"] = True
                    rewrite_metadata["deferred_phase"] = phase
                    should_retry_rewrite = _should_retry_duration_rewrite_result(
                        rewritten,
                        segment.script,
                    )
                    rewrite_metadata["retry_scheduled"] = should_retry_rewrite
                    _log_duration_rewrite_result(segment.id, rewrite_metadata)
                    if should_retry_rewrite:
                        segment.script = rewritten
                        reset_segment_for_duration_rewrite_retry(segment, rewrite_metadata)
                        retry_segment_ids.add(segment.id)
                    else:
                        record_skipped_duration_rewrite_retry(segment, rewrite_metadata)
                        skipped_segment_ids.add(segment.id)
                    save_manifest(project_dir, manifest)
            finally:
                duration_rewrite_client = None
                if duration_rewrite_manager is not None and duration_rewrite_running:
                    duration_rewrite_manager.stop()
                    duration_rewrite_running = False
            summary = {
                "phase": phase,
                "backend": "gemma",
                "attempted_segments": sorted(str(request["segment"].id) for request in requests),
                "retry_segments": sorted(retry_segment_ids),
                "skipped_segments": sorted(skipped_segment_ids),
            }
            return retry_segment_ids, summary

        def recoverable_terminal_tts_failure(segment: Segment) -> bool:
            if not segment.script or not segment.tts:
                return False
            errors = [str(error) for error in segment.errors if error]
            blocked_prefixes = (
                "Korean TTS preflight blocked synthesis",
                "Cannot synthesize without script metadata",
                "GPT-SoVITS synthesis failed",
            )
            if any(error.startswith(blocked_prefixes) for error in errors):
                return False
            recoverable_errors = {
                "No acceptable TTS candidates for mix.",
                "All TTS candidates failed.",
            }
            return any(error in recoverable_errors for error in errors)

        def best_timing_expansion_candidate(segment: Segment) -> TTSCandidate | None:
            if not segment.tts or segment.duration <= 0:
                return None
            min_ratio = float(getattr(cfg, "gsv_timing_expansion_min_ratio", 0.70))
            candidates: list[TTSCandidate] = []
            for candidate in segment.tts.candidates:
                if candidate.error or candidate.duration_sec is None or candidate.duration_sec <= 0:
                    continue
                if candidate.payload.get("audio_qc", {}).get("gate") != "pass":
                    continue
                ratio = candidate.duration_ratio
                if ratio is None:
                    ratio = duration_ratio(candidate.duration_sec, segment.duration)
                too_short = candidate.duration_gate == "too_short"
                omission = bool(candidate.payload.get("omission_suspected"))
                if not (too_short or omission):
                    continue
                if ratio is not None and float(ratio) >= min_ratio:
                    continue
                candidates.append(candidate)
            if not candidates:
                return None
            return max(
                candidates,
                key=lambda candidate: float(candidate.duration_ratio or 0.0),
            )

        def timing_expansion_budget(
            segment: Segment,
            current_text: str,
        ) -> dict[str, int]:
            slot_budget = korean_tts_slot_timing_budget(segment.duration)
            current_chars = korean_tts_speech_char_count(current_text)
            target_chars = int(slot_budget["target_speech_chars"])
            max_chars = int(slot_budget["max_speech_chars"])
            min_chars = max(current_chars + 1, int(slot_budget["min_speech_chars"]))
            min_chars = min(min_chars, max_chars)
            return {
                "current": current_chars,
                "target": target_chars,
                "min": min_chars,
                "max": max_chars,
            }

        def accept_timing_expansion_text(
            *,
            segment: Segment,
            candidate_text: str,
            budget: dict[str, int],
        ) -> tuple[bool, dict[str, Any]]:
            normalized = normalize_korean_tts_text(candidate_text)
            text = normalized.text.strip()
            chars = korean_tts_speech_char_count(text)
            metadata: dict[str, Any] = {
                "normalized_text": text,
                "speech_chars": chars,
                "target_speech_chars": budget["target"],
                "min_speech_chars": budget["min"],
                "max_speech_chars": budget["max"],
                "rejected_reasons": [],
            }
            if not text:
                metadata["rejected_reasons"].append("empty_expansion")
            if chars <= budget["current"]:
                metadata["rejected_reasons"].append(
                    f"not_longer_for_too_short:{chars}<={budget['current']}"
                )
            if chars < budget["min"]:
                metadata["rejected_reasons"].append(
                    f"speech_chars_below_min:{chars}<{budget['min']}"
                )
            if chars > budget["max"]:
                metadata["rejected_reasons"].append(
                    f"speech_chars_above_max:{chars}>{budget['max']}"
                )
            trial_script = segment.script.model_copy(
                update={
                    "tts_text": text,
                    "expected_tts_duration_sec": estimate_tts_duration(text, "ko"),
                    "risk_flags": [*segment.script.risk_flags, *normalized.risk_flags],
                },
                deep=True,
            )
            preflight = preflight_tts_text(
                trial_script,
                target_language=cfg.target_language,
                source_text=segment.source_script.text if segment.source_script else "",
                min_hangul_ratio=cfg.gsv_ko_text_min_hangul_ratio,
            )
            metadata["preflight"] = preflight.as_payload()
            if preflight.blocked:
                metadata["rejected_reasons"].append(
                    "preflight_blocked:" + ",".join(preflight.issues)
                )
            return not metadata["rejected_reasons"], metadata

        def timing_expansion_request_for_segment(
            index: int,
            segment: Segment,
        ) -> dict[str, Any] | None:
            if (
                not timing_expansion_enabled
                or not segment.script
                or segment.status != "failed"
                or not recoverable_terminal_tts_failure(segment)
            ):
                return None
            history = segment.analysis.get("timing_expansion_history")
            attempts_used = len(history) if isinstance(history, list) else 0
            if attempts_used >= int(getattr(cfg, "gsv_timing_expansion_max_attempts", 0)):
                return None
            candidate = best_timing_expansion_candidate(segment)
            if candidate is None:
                return None
            return {
                "index": index,
                "segment": segment,
                "candidate": candidate,
                "attempt_text": str(candidate.payload.get("text") or segment.script.tts_text),
                "actual_duration_sec": float(candidate.duration_sec or 0.0),
                "attempts_used": attempts_used,
            }

        def maybe_expand_script_timing_with_gemma(
            *,
            segment: Segment,
            index: int,
            attempt_text: str,
            actual_duration_sec: float,
            attempts_used: int,
        ) -> tuple[JapaneseScript | None, dict[str, Any]]:
            metadata: dict[str, Any] = {
                "backend": "gemma_text",
                "reason": "too_short",
                "before": attempt_text,
                "actual_duration_sec": round(actual_duration_sec, 6),
                "target_duration_sec": round(segment.duration, 6),
                "accepted": False,
            }
            if duration_rewrite_client is None:
                metadata["error"] = "timing_expansion_client_unavailable"
                return None, metadata
            if attempts_used >= int(getattr(cfg, "gsv_timing_expansion_max_attempts", 0)):
                metadata["error"] = "timing_expansion_attempt_limit_reached"
                return None, metadata
            if segment.script.rewrite_count >= segment.script.retry_policy.max_script_rewrites:
                metadata["error"] = "script_retry_policy_limit_reached"
                return None, metadata
            budget = timing_expansion_budget(segment, attempt_text)
            metadata.update(
                {
                    "current_speech_chars": budget["current"],
                    "target_speech_chars": budget["target"],
                    "min_speech_chars": budget["min"],
                    "max_speech_chars": budget["max"],
                }
            )
            if budget["current"] >= budget["max"]:
                metadata["error"] = "timing_expansion_budget_exhausted"
                return None, metadata
            batch_id = f"timing_expansion_{segment.id}_too_short_{attempts_used + 1}"
            try:
                with duration_rewrite_lock:
                    translation: KoreanTranslation | None = (
                        duration_rewrite_client.rewrite_tts_for_timing_expansion(
                            segment=segment,
                            batch_id=batch_id,
                            current_text=attempt_text,
                            reason="too_short",
                            actual_duration_sec=actual_duration_sec,
                            target_duration_sec=segment.duration,
                            target_speech_chars=budget["target"],
                            min_speech_chars=budget["min"],
                            max_speech_chars=budget["max"],
                            context_segments=duration_rewrite_context(index),
                        )
                    )
            except Exception as exc:
                metadata["error"] = str(exc)
                return None, metadata
            if translation is None:
                metadata["error"] = "gemma_returned_no_timing_expansion"
                return None, metadata
            accepted, acceptance = accept_timing_expansion_text(
                segment=segment,
                candidate_text=translation.ko_natural,
                budget=budget,
            )
            metadata.update(acceptance)
            metadata["after"] = acceptance["normalized_text"]
            metadata["model"] = translation.model
            metadata["batch_id"] = translation.batch_id
            if not accepted:
                return None, metadata
            updated = segment.script.model_copy(deep=True)
            updated.tts_text = acceptance["normalized_text"]
            updated.expected_tts_duration_sec = estimate_tts_duration(updated.tts_text, "ko")
            updated.rewrite_count += 1
            updated.risk_flags.append("gemma_timing_expansion_too_short")
            metadata["accepted"] = True
            return updated, metadata

        def reset_segment_for_timing_expansion_retry(
            segment: Segment,
            expansion_metadata: dict[str, Any],
        ) -> None:
            segment.analysis.setdefault("timing_expansion_history", []).append(
                copy.deepcopy(expansion_metadata)
            )
            segment.analysis["pending_timing_expansion"] = copy.deepcopy(expansion_metadata)
            segment.status = "scripted"
            segment.tts = None
            segment.errors = [
                error
                for error in segment.errors
                if error
                not in {
                    "No acceptable TTS candidates for mix.",
                    "All TTS candidates failed.",
                    "Micro segment too short for Korean TTS.",
                }
                and not error.startswith("GPT-SoVITS synthesis failed")
                and not error.startswith("Korean TTS preflight blocked synthesis")
            ]

        def record_skipped_timing_expansion_retry(
            segment: Segment,
            expansion_metadata: dict[str, Any],
        ) -> None:
            segment.analysis.setdefault("timing_expansion_history", []).append(
                copy.deepcopy(expansion_metadata)
            )
            segment.analysis["timing_expansion_retry_skipped"] = copy.deepcopy(
                expansion_metadata
            )
            segment.analysis.pop("pending_timing_expansion", None)

        def run_deferred_timing_expansions(
            *,
            segment_ids: set[str],
            phase: str,
        ) -> tuple[set[str], dict[str, Any] | None]:
            nonlocal duration_rewrite_client, duration_rewrite_running
            requests = [
                request
                for index, segment in enumerate(manifest.segments, start=1)
                if only_segment_ids is None or segment.id in only_segment_ids
                if segment.id in segment_ids
                for request in [timing_expansion_request_for_segment(index, segment)]
                if request is not None
            ]
            if not requests:
                return set(), None
            stop_gsv_servers()
            if duration_rewrite_manager is not None:
                duration_rewrite_manager.start()
                duration_rewrite_running = True
            duration_rewrite_client = LlamaServerTranslationClient(
                duration_rewrite_base_url,
                timeout_sec=cfg.gemma_text_timeout_sec,
                retries=cfg.gemma_text_retries,
                n_predict=cfg.gemma_text_n_predict,
                model=cfg.gemma_llama_cpp_model_path,
                two_pass=False,
            )
            retry_segment_ids: set[str] = set()
            skipped_segment_ids: set[str] = set()
            try:
                for request in requests:
                    segment = request["segment"]
                    expanded, expansion_metadata = maybe_expand_script_timing_with_gemma(
                        segment=segment,
                        index=int(request["index"]),
                        attempt_text=str(request["attempt_text"]),
                        actual_duration_sec=float(request["actual_duration_sec"]),
                        attempts_used=int(request["attempts_used"]),
                    )
                    expansion_metadata["deferred"] = True
                    expansion_metadata["deferred_phase"] = phase
                    expansion_metadata["retry_scheduled"] = _should_retry_duration_rewrite_result(
                        expanded,
                        segment.script,
                    )
                    if expansion_metadata["retry_scheduled"]:
                        segment.script = expanded
                        reset_segment_for_timing_expansion_retry(segment, expansion_metadata)
                        retry_segment_ids.add(segment.id)
                    else:
                        record_skipped_timing_expansion_retry(segment, expansion_metadata)
                        skipped_segment_ids.add(segment.id)
                    save_manifest(project_dir, manifest)
            finally:
                duration_rewrite_client = None
                if duration_rewrite_manager is not None and duration_rewrite_running:
                    duration_rewrite_manager.stop()
                    duration_rewrite_running = False
            summary = {
                "phase": phase,
                "backend": "gemma",
                "attempted_segments": sorted(str(request["segment"].id) for request in requests),
                "retry_segments": sorted(retry_segment_ids),
                "skipped_segments": sorted(skipped_segment_ids),
            }
            return retry_segment_ids, summary

        def apply_terminal_keep_original_fallback(segment_ids: list[str]) -> dict[str, Any] | None:
            if str(getattr(cfg, "gsv_terminal_failure_policy", "keep_original")) != "keep_original":
                return None

            def materialize_selected_fallback(
                segment: Segment,
                selected: TTSCandidate | None,
            ) -> str | None:
                if segment.tts is None or selected is None:
                    return None
                source_path = _resolve_manifest_path(project_dir, selected.output_path)
                if source_path is None or not source_path.exists():
                    return None
                final_path = ensure_inside_project(
                    project_dir,
                    project_dir / "work" / "tts" / f"{segment.id}_final.wav",
                )
                final_path.parent.mkdir(parents=True, exist_ok=True)
                if source_path.resolve() != final_path.resolve():
                    ensure_not_same_path(source_path, final_path)
                    shutil.copy2(source_path, final_path)
                for candidate in segment.tts.candidates:
                    candidate.selected = candidate is selected
                if not selected.selection_reason:
                    selected.selection_reason = "terminal_keep_original_fallback"
                segment.tts.selected_candidate_path = str(final_path)
                segment.tts.retry_summary = {
                    **segment.tts.retry_summary,
                    "terminal_failure_policy": "keep_original",
                    "fallback_selected_candidate_path": str(final_path),
                    "fallback_source_candidate_path": selected.output_path,
                }
                return str(final_path)

            attempted: list[str] = []
            succeeded: list[str] = []
            failed: list[str] = []
            for segment in manifest.segments:
                if segment.id not in segment_ids:
                    continue
                attempted.append(segment.id)
                if not recoverable_terminal_tts_failure(segment):
                    failed.append(segment.id)
                    continue
                selected = None
                if segment.tts is not None:
                    successful = [
                        candidate
                        for candidate in segment.tts.candidates
                        if not candidate.error and candidate.duration_sec is not None
                    ]
                    if successful:
                        selected = _select_gsv_candidate_for_mix(successful, segment)
                selected_candidate_path = materialize_selected_fallback(segment, selected)
                segment.analysis["synth_keep_original_fallback"] = {
                    "action": "keep_original_after_tts_failure",
                    "terminal_failure_policy": "keep_original",
                    "previous_status": segment.status,
                    "previous_errors": list(segment.errors),
                    "selected_candidate_path": selected_candidate_path,
                    "selected_source_candidate_path": selected.output_path if selected else None,
                    "selected_duration_gate": selected.duration_gate if selected else None,
                    "selected_duration_ratio": selected.duration_ratio if selected else None,
                    "selected_selection_reason": selected.selection_reason if selected else None,
                }
                segment.status = "absorbed"
                segment.keep_original_texture = True
                succeeded.append(segment.id)
            if not attempted:
                return None
            return {
                "attempted_segments": attempted,
                "succeeded_segments": succeeded,
                "failed_segments": failed,
            }

        def should_queue_initial_synth_job(segment: Segment) -> bool:
            if retry_failed and not force:
                return should_retry_failed_segment(segment)
            return True

        segment_jobs = [
            (index, segment, _segment_lane_index(segment, index - 1, gsv_lane_count))
            for index, segment in enumerate(manifest.segments, start=1)
            if (only_segment_ids is None or segment.id in only_segment_ids)
            and should_queue_initial_synth_job(segment)
        ]
        countdown_segment_ids = {
            segment.id
            for _index, segment, _lane_index in segment_jobs
            if detected_countdown_segment_values(segment) is not None
        }
        numeric_phrase_handled_segment_ids = render_numeric_phrase_segments(segment_jobs)
        countdown_rendered_segment_ids = (
            render_countdown_spans(segment_jobs) if render_countdowns else set()
        )
        embedded_countdown_rendered_segment_ids = render_embedded_countdown_hybrid_beds(
            segment_jobs
        )
        if not countdown_only:
            run_synth_jobs(
                [
                    (index, segment, lane_index)
                    for index, segment, lane_index in segment_jobs
                    if segment.id not in countdown_rendered_segment_ids
                    and segment.id not in numeric_phrase_handled_segment_ids
                    and (render_countdowns or segment.id not in countdown_segment_ids)
                ]
            )
        if duration_rewrite_after_initial_enabled:
            duration_rewrite_retry_segment_ids, _duration_rewrite_after_initial_summary = (
                run_deferred_duration_rewrites(phase="after_initial")
            )
            if duration_rewrite_retry_segment_ids:
                duration_rewrite_phase = "post_rewrite"
                retry_jobs = [
                    (index, segment, _segment_lane_index(segment, index - 1, gsv_lane_count))
                    for index, segment in enumerate(manifest.segments, start=1)
                    if segment.id in duration_rewrite_retry_segment_ids
                    and (only_segment_ids is None or segment.id in only_segment_ids)
                    and not countdown_only
                ]
                run_synth_jobs(retry_jobs)
        failed_after_primary = [] if countdown_only else selected_failed_segment_ids()
        fine_tuned_retry_enabled = bool(
            not mock
            and failed_after_primary
            and (gpt_weights or sovits_weights or use_speaker_gsv)
        )
        if fine_tuned_retry_enabled:
            fine_tuned_retry_summary = run_internal_retry_pass(
                pass_name="fine_tuned_retry",
                segment_ids=failed_after_primary,
            )
        failed_after_fine_tuned_retry = [] if countdown_only else selected_failed_segment_ids()
        static_ref_retry_segments = failed_segment_ids_with_used_segment_ref(
            failed_after_fine_tuned_retry
        )
        static_ref_retry_enabled = bool(
            not mock
            and static_ref_retry_segments
            and getattr(cfg, "gsv_ref_mode", "static") in {"segment", "auto"}
        )
        if static_ref_retry_enabled:
            static_ref_retry_summary = run_internal_retry_pass(
                pass_name="static_ref_retry",
                segment_ids=static_ref_retry_segments,
                static_ref_retry=True,
            )
        failed_after_static_ref_retry = [] if countdown_only else selected_failed_segment_ids()
        korean_clarity_retry_segments = failed_segment_ids_with_pronunciation_failure(
            failed_after_static_ref_retry
        )
        korean_clarity_retry_enabled = bool(
            not mock
            and korean_clarity_retry_segments
            and _canonical_language(cfg.target_language) == "ko"
            and getattr(cfg, "gsv_korean_clarity_retry_enabled", True)
        )
        if korean_clarity_retry_enabled:
            korean_clarity_retry_summary = run_internal_retry_pass(
                pass_name="korean_clarity_retry",
                segment_ids=korean_clarity_retry_segments,
                candidate_count_override=getattr(
                    cfg,
                    "gsv_korean_clarity_retry_candidate_count",
                    None,
                ),
                temperature_override=float(
                    getattr(cfg, "gsv_korean_clarity_temperature", 0.65)
                ),
                top_k_override=int(getattr(cfg, "gsv_korean_clarity_top_k", 5)),
                top_p_override=float(getattr(cfg, "gsv_korean_clarity_top_p", 0.75)),
                parallel_infer_override=bool(
                    getattr(cfg, "gsv_korean_clarity_parallel_infer", False)
                ),
            )
        failed_after_korean_clarity_retry = [] if countdown_only else selected_failed_segment_ids()
        low_temperature_retry_enabled = bool(
            not mock
            and failed_after_korean_clarity_retry
            and getattr(cfg, "gsv_low_temperature_retry_enabled", True)
        )
        if low_temperature_retry_enabled:
            low_temperature_retry_summary = run_internal_retry_pass(
                pass_name="low_temperature_retry",
                segment_ids=failed_after_korean_clarity_retry,
                temperature_override=float(
                    getattr(cfg, "gsv_low_temperature_retry_temperature", 0.5)
                ),
            )
        failed_after_low_temperature_retry = [] if countdown_only else selected_failed_segment_ids()
        if (
            duration_rewrite_before_zero_shot_enabled
            and failed_after_low_temperature_retry
            and not countdown_only
        ):
            before_zero_shot_failed = set(failed_after_low_temperature_retry)
            duration_rewrite_retry_segment_ids, duration_rewrite_before_zero_shot_summary = (
                run_deferred_duration_rewrites(
                    segment_ids=before_zero_shot_failed,
                    phase="before_zero_shot",
                )
            )
            if duration_rewrite_retry_segment_ids:
                duration_rewrite_phase = "before_zero_shot"
                synth_pass = "before_zero_shot_duration_rewrite"
                retry_jobs = [
                    (index, segment, _segment_lane_index(segment, index - 1, gsv_lane_count))
                    for index, segment in enumerate(manifest.segments, start=1)
                    if segment.id in duration_rewrite_retry_segment_ids
                    and (only_segment_ids is None or segment.id in only_segment_ids)
                ]
                run_synth_jobs(retry_jobs)
                if duration_rewrite_before_zero_shot_summary is not None:
                    duration_rewrite_before_zero_shot_summary["succeeded_segments"] = sorted(
                        segment.id
                        for segment in manifest.segments
                        if segment.id in before_zero_shot_failed and segment.status == "synthesized"
                    )
                    duration_rewrite_before_zero_shot_summary["failed_segments"] = sorted(
                        segment.id
                        for segment in manifest.segments
                        if segment.id in before_zero_shot_failed and segment.status == "failed"
                    )
                duration_rewrite_phase = "normal"
                synth_pass = "complete"
            failed_after_low_temperature_retry = [] if countdown_only else selected_failed_segment_ids()
        if (
            timing_expansion_enabled
            and failed_after_low_temperature_retry
            and not countdown_only
        ):
            before_timing_expansion_failed = set(failed_after_low_temperature_retry)
            timing_expansion_retry_segment_ids, timing_expansion_summary = (
                run_deferred_timing_expansions(
                    segment_ids=before_timing_expansion_failed,
                    phase="before_zero_shot",
                )
            )
            if timing_expansion_retry_segment_ids:
                synth_pass = "timing_expansion_retry"
                retry_jobs = [
                    (index, segment, _segment_lane_index(segment, index - 1, gsv_lane_count))
                    for index, segment in enumerate(manifest.segments, start=1)
                    if segment.id in timing_expansion_retry_segment_ids
                    and (only_segment_ids is None or segment.id in only_segment_ids)
                ]
                run_synth_jobs(retry_jobs)
                if timing_expansion_summary is not None:
                    timing_expansion_summary["succeeded_segments"] = sorted(
                        segment.id
                        for segment in manifest.segments
                        if segment.id in before_timing_expansion_failed
                        and segment.status == "synthesized"
                    )
                    timing_expansion_summary["failed_segments"] = sorted(
                        segment.id
                        for segment in manifest.segments
                        if segment.id in before_timing_expansion_failed
                        and segment.status == "failed"
                    )
                synth_pass = "complete"
            failed_after_low_temperature_retry = [] if countdown_only else selected_failed_segment_ids()
        zero_shot_fallback_enabled = bool(
            not mock
            and failed_after_low_temperature_retry
            and not use_speaker_gsv
            and (gpt_weights or sovits_weights)
        )
        if zero_shot_fallback_enabled:
            zero_shot_fallback_summary = run_internal_retry_pass(
                pass_name="zero_shot_fallback",
                segment_ids=failed_after_low_temperature_retry,
                zero_shot=True,
            )
        synth_pass = "complete"
        terminal_failed_before_fallback = (
            []
            if countdown_only
            else selected_failed_segment_ids(include_invalid_ref_duration=True)
        )
        keep_original_fallback_summary = apply_terminal_keep_original_fallback(
            terminal_failed_before_fallback
        )
        gsv_instances = [
            {
                "base_url": manager.base_url,
                "started": manager.started,
                "reused_existing": manager.reused_existing,
                "log_path": str(manager.log_path) if manager.log_path else None,
            }
            for manager in server_managers
        ]
        gsv_server_metadata = {
            "auto_start": should_auto_start_server,
            "concurrency": gsv_lane_count,
            "base_urls": [] if mock else gsv_base_urls,
            "instances": gsv_instances,
        }
        if len(gsv_instances) == 1:
            gsv_server_metadata.update(
                started=gsv_instances[0]["started"],
                reused_existing=gsv_instances[0]["reused_existing"],
                log_path=gsv_instances[0]["log_path"],
            )
        failed_synth_segments = [
            segment.id
            for segment in manifest.segments
            if segment.status == "failed"
            and (only_segment_ids is None or segment.id in only_segment_ids)
            and failed_segment_in_stage_scope(segment)
        ]
        recovery_metadata = {}
        if fine_tuned_retry_summary is not None:
            recovery_metadata["fine_tuned_retry"] = fine_tuned_retry_summary
        if static_ref_retry_summary is not None:
            recovery_metadata["static_ref_retry"] = static_ref_retry_summary
        if korean_clarity_retry_summary is not None:
            recovery_metadata["korean_clarity_retry"] = korean_clarity_retry_summary
        if low_temperature_retry_summary is not None:
            recovery_metadata["low_temperature_retry"] = low_temperature_retry_summary
        if duration_rewrite_before_zero_shot_summary is not None:
            recovery_metadata["duration_rewrite_before_zero_shot"] = (
                duration_rewrite_before_zero_shot_summary
            )
        if timing_expansion_summary is not None:
            recovery_metadata["timing_expansion"] = timing_expansion_summary
        if zero_shot_fallback_summary is not None:
            recovery_metadata["zero_shot_fallback"] = zero_shot_fallback_summary
        if keep_original_fallback_summary is not None:
            recovery_metadata["keep_original_fallback"] = keep_original_fallback_summary
        if not mock and failed_synth_segments:
            mark_stage(
                manifest,
                stage_name,
                "failed",
                backend="gpt-sovits",
                gsv_url=effective_gsv_url,
                gsv_urls=gsv_base_urls,
                gsv_server=gsv_server_metadata,
                failed_segments=failed_synth_segments,
                retry_failed=retry_failed,
                force=force,
                render_countdowns=render_countdowns,
                countdown_only=countdown_only,
                numeric_phrase_rendered_segments=sorted(numeric_phrase_rendered_segment_ids),
                numeric_phrase_failed_segments=sorted(numeric_phrase_failed_segment_ids),
                numeric_phrase_handled_segments=sorted(numeric_phrase_handled_segment_ids),
                numeric_phrase_normal_tts_fallback_segments=sorted(
                    numeric_phrase_normal_tts_fallback_segment_ids
                ),
                segment_counts=_segment_counts(manifest),
                **recovery_metadata,
            )
            save_manifest(project_dir, manifest)
            raise GPTSoVITSError(
                "GPT-SoVITS synthesis failed for segments: "
                + ", ".join(failed_synth_segments[:20])
                + (" ..." if len(failed_synth_segments) > 20 else "")
            )
        mark_stage(
            manifest,
            stage_name,
            "completed",
            backend="mock" if mock else "gpt-sovits",
            gsv_url=None if mock else effective_gsv_url,
            gsv_urls=[] if mock else gsv_base_urls,
            gsv_server=gsv_server_metadata,
            concurrency=gsv_lane_count,
            model_switch=model_switch,
            retry_failed=retry_failed,
            force=force,
            render_countdowns=render_countdowns,
            countdown_only=countdown_only,
            numeric_phrase_rendered_segments=sorted(numeric_phrase_rendered_segment_ids),
            numeric_phrase_failed_segments=sorted(numeric_phrase_failed_segment_ids),
            numeric_phrase_handled_segments=sorted(numeric_phrase_handled_segment_ids),
            numeric_phrase_normal_tts_fallback_segments=sorted(
                numeric_phrase_normal_tts_fallback_segment_ids
            ),
            countdown_rendered_segments=sorted(countdown_rendered_segment_ids),
            embedded_countdown_hybrid_segments=sorted(
                embedded_countdown_rendered_segment_ids
            ),
            countdown_skipped_segments=sorted(
                countdown_segment_ids
                - countdown_rendered_segment_ids
                - numeric_phrase_handled_segment_ids
            ),
            segment_counts=_segment_counts(manifest),
            **recovery_metadata,
        )
        save_manifest(project_dir, manifest)
        _log_stage_complete(stage_name, manifest, f"backend={synth_backend_name}")
        return ctx.update_manifest(manifest)
    finally:
        for server_manager in reversed(server_managers):
            server_manager.stop()
        gsv_servers_running = False
        if duration_rewrite_manager is not None and duration_rewrite_running:
            duration_rewrite_manager.stop()


def run_countdown_synth_stage(ctx: PipelineContext, gsv_url: str | None, refs_path: Path, mock: bool = False, confirm_rights: bool = False, gpt_weights_path: str | None = None, sovits_weights_path: str | None = None, auto_gsv_server: bool | None = None, gsv_server_command: list[str] | str | None = None, use_trained_gpt: bool = False, only_segment_ids: set[str] | None = None, retry_failed: bool = False, force: bool = False) -> PipelineManifest:
    return run_synth_stage(
        ctx,
        gsv_url,
        refs_path,
        mock=mock,
        confirm_rights=confirm_rights,
        gpt_weights_path=gpt_weights_path,
        sovits_weights_path=sovits_weights_path,
        auto_gsv_server=auto_gsv_server,
        gsv_server_command=gsv_server_command,
        use_trained_gpt=use_trained_gpt,
        only_segment_ids=only_segment_ids,
        retry_failed=retry_failed,
        force=force,
        render_countdowns=True,
        countdown_only=True,
        stage_name="countdown-synth",
    )
