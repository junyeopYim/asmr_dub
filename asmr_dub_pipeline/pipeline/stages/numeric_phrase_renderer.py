from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from asmr_dub_pipeline.audio.features import load_audio, resample_linear, write_audio
from asmr_dub_pipeline.gpt_sovits.client import normalize_api_language_code
from asmr_dub_pipeline.schemas import Segment
from asmr_dub_pipeline.script.numeric_cadence import extract_korean_numeric_values
from asmr_dub_pipeline.script.numeric_render_plan import NumericRenderPlan


@dataclass(frozen=True)
class RenderedNumericBed:
    audio: np.ndarray
    sample_rate: int
    policy: str
    placements: list[dict[str, Any]]
    max_tempo: float


@dataclass(frozen=True)
class NumericPhraseRequest:
    text: str
    text_lang: str = "all_ko"
    ref_audio_path: str = ""
    prompt_text: str = ""
    prompt_lang: str = "ko"
    aux_ref_audio_paths: list[str] = field(default_factory=list)
    text_split_method: str = "cut0"
    top_k: int = 5
    top_p: float = 0.85
    temperature: float = 0.65
    repetition_penalty: float = 1.8

    def __post_init__(self) -> None:
        object.__setattr__(self, "text_lang", normalize_api_language_code(self.text_lang))
        object.__setattr__(self, "prompt_lang", normalize_api_language_code(self.prompt_lang))

    def as_payload(self) -> dict[str, Any]:
        """Return an api_v2-compatible payload for duck-typed GPT-SoVITS clients."""
        payload = asdict(self)
        return {key: value for key, value in payload.items() if value not in (None, "")}


def _ref_value(ref: Any, key: str, default: Any = "") -> Any:
    if isinstance(ref, dict):
        return ref.get(key, default)
    return getattr(ref, key, default)


def build_numeric_phrase_request(plan: NumericRenderPlan, *, ref: Any) -> NumericPhraseRequest:
    """Build a deterministic low-randomness GPT-SoVITS numeric phrase request."""
    aux_paths = _ref_value(ref, "aux_ref_audio_paths", []) or []
    return NumericPhraseRequest(
        text=plan.text,
        ref_audio_path=str(_ref_value(ref, "ref_audio_path", "")),
        prompt_text=str(_ref_value(ref, "prompt_text", "")),
        prompt_lang=str(_ref_value(ref, "prompt_lang", "ko") or "ko"),
        aux_ref_audio_paths=[str(path) for path in aux_paths],
    )


def _value_counts(values: list[int]) -> dict[int, int]:
    return dict(Counter(int(value) for value in values))


def _repeated_timing_mismatches(
    expected_values: list[int],
    word_timing: list[dict[str, Any]],
) -> dict[int, dict[str, int]]:
    expected_counts = Counter(int(value) for value in expected_values)
    timing_counts = Counter(int(item["value"]) for item in word_timing)
    mismatches: dict[int, dict[str, int]] = {}
    for value, expected_count in expected_counts.items():
        if expected_count <= 1:
            continue
        observed_count = timing_counts.get(value, 0)
        if observed_count != expected_count:
            mismatches[value] = {
                "expected": expected_count,
                "observed": observed_count,
            }
    return mismatches


def evaluate_numeric_render_transcript(
    expected: list[int],
    transcript: str,
    *,
    word_timing: list[dict[str, Any]] | None = None,
    required_timing_values: list[int] | None = None,
) -> dict[str, Any]:
    """Compare a transcript's extracted numeric sequence against expected render values."""
    expected_values = list(expected)
    observed_values = extract_korean_numeric_values(transcript)
    required_values = list(required_timing_values or [])
    timing_values = {int(item["value"]) for item in word_timing or []}
    missing_timing_values = [value for value in required_values if value not in timing_values]
    repeated_timing_mismatches = _repeated_timing_mismatches(expected_values, list(word_timing or []))
    issues: list[str] = []
    if observed_values != expected_values:
        issues.append("numeric_sequence_mismatch")
    if missing_timing_values:
        issues.append("missing_word_timing")
    if repeated_timing_mismatches:
        issues.append("repeated_numeric_word_timing_mismatch")
    pass_gate = not issues
    return {
        "gate": "pass" if pass_gate else "fail",
        "expected_values": list(expected_values),
        "observed_values": list(observed_values),
        "transcript": transcript,
        "word_timing": list(word_timing or []),
        "expected_value_counts": _value_counts(expected_values),
        "word_timing_value_counts": _value_counts([int(item["value"]) for item in word_timing or []]),
        "repeated_timing_mismatches": repeated_timing_mismatches,
        "required_timing_values": required_values,
        "missing_timing_values": missing_timing_values,
        "issues": issues,
    }


def _frame(sec: float, sample_rate: int) -> int:
    """Convert seconds to a non-negative frame index."""
    return max(0, int(round(float(sec) * sample_rate)))


def _fade_edges(audio: np.ndarray, sample_rate: int, fade_sec: float = 0.006) -> np.ndarray:
    """Return a copy with short linear fades at both edges."""
    faded = np.array(audio, copy=True)
    if faded.size == 0:
        return faded
    fade_frames = min(_frame(fade_sec, sample_rate), len(faded) // 2)
    if fade_frames <= 0:
        return faded

    fade_in = np.linspace(0.0, 1.0, fade_frames, endpoint=True, dtype=np.float32)
    fade_out = np.linspace(1.0, 0.0, fade_frames, endpoint=True, dtype=np.float32)
    if faded.ndim == 1:
        faded[:fade_frames] *= fade_in
        faded[-fade_frames:] *= fade_out
    else:
        faded[:fade_frames] *= fade_in[:, np.newaxis]
        faded[-fade_frames:] *= fade_out[:, np.newaxis]
    return faded


def _timing_by_value(word_timing: list[dict[str, Any]]) -> dict[int, dict[str, float]]:
    """Index numeric word timings by integer value."""
    timings: dict[int, dict[str, float]] = {}
    for item in word_timing:
        value = int(item["value"])
        timings[value] = {
            "source_start_sec": float(item["source_start_sec"]),
            "source_end_sec": float(item["source_end_sec"]),
        }
    return timings


def _normalized_word_timing(word_timing: list[dict[str, Any]]) -> list[dict[str, float | int]]:
    return [
        {
            "value": int(item["value"]),
            "source_start_sec": float(item["source_start_sec"]),
            "source_end_sec": float(item["source_end_sec"]),
        }
        for item in word_timing
    ]


def _has_repeated_values(values: list[int]) -> bool:
    counts = Counter(int(value) for value in values)
    return any(count > 1 for count in counts.values())


def _ordered_span_timing(
    word_timing: list[dict[str, Any]],
    values: list[int],
) -> tuple[dict[str, float | int], dict[str, float | int]] | None:
    timings = _normalized_word_timing(word_timing)
    if not timings or not values:
        return None
    if _has_repeated_values(values):
        matched: list[dict[str, float | int]] = []
        search_start = 0
        for expected_value in values:
            for index in range(search_start, len(timings)):
                if int(timings[index]["value"]) == int(expected_value):
                    matched.append(timings[index])
                    search_start = index + 1
                    break
            else:
                return None
        return matched[0], matched[-1]

    first_value = int(values[0])
    last_value = int(values[-1])
    first = next((item for item in timings if int(item["value"]) == first_value), None)
    last = next((item for item in reversed(timings) if int(item["value"]) == last_value), None)
    if first is None or last is None:
        return None
    return first, last


def _as_audio_matrix(audio: np.ndarray) -> np.ndarray:
    matrix = np.asarray(audio, dtype=np.float32)
    if matrix.ndim == 1:
        return matrix[:, np.newaxis]
    if matrix.ndim != 2:
        raise ValueError("audio must be a 1D mono or 2D channel-last NumPy array")
    return matrix


def _match_channels(source: np.ndarray, channels: int) -> np.ndarray:
    if source.shape[1] == channels:
        return source
    if source.shape[1] == 1:
        return np.repeat(source, channels, axis=1)
    if channels == 1:
        return source.mean(axis=1, keepdims=True)
    raise ValueError(f"Cannot copy {source.shape[1]}-channel audio into {channels}-channel bed")


def _copy_chunk(
    *,
    bed: np.ndarray,
    source_audio: np.ndarray,
    sample_rate: int,
    values: list[int],
    source_start_sec: float,
    source_end_sec: float,
    target_start_sec: float,
    target_end_sec: float,
    placements: list[dict[str, Any]],
) -> None:
    """Copy a source chunk into the render bed and append placement metadata."""
    source = _match_channels(_as_audio_matrix(source_audio), bed.shape[1])
    requested_source_start_sec = float(source_start_sec)
    requested_source_end_sec = float(source_end_sec)
    requested_target_start_sec = float(target_start_sec)
    requested_target_end_sec = float(target_end_sec)

    source_start_frame = min(_frame(requested_source_start_sec, sample_rate), len(source))
    source_end_frame = min(_frame(requested_source_end_sec, sample_rate), len(source))
    target_start_frame = min(_frame(requested_target_start_sec, sample_rate), len(bed))
    target_end_frame = min(_frame(requested_target_end_sec, sample_rate), len(bed))

    source_frames = max(0, source_end_frame - source_start_frame)
    target_available_frames = max(0, target_end_frame - target_start_frame)
    required_tempo = 1.0
    truncated = False
    copied_frames = 0
    copy_status = "skipped"
    skipped_reason: str | None = None

    if target_available_frames <= 0:
        skipped_reason = "empty_target_window"
    elif source_frames <= 0:
        skipped_reason = "empty_source_window"
    else:
        required_tempo = max(1.0, source_frames / target_available_frames)
        truncated = source_frames > target_available_frames
        copied_frames = min(source_frames, target_available_frames)
        copy_status = "copied"

    copy_source_end_frame = source_start_frame + copied_frames
    copy_target_end_frame = target_start_frame + copied_frames
    if copied_frames > 0:
        chunk = _fade_edges(source[source_start_frame:copy_source_end_frame], sample_rate)
        bed[target_start_frame:copy_target_end_frame] = chunk

    placement = {
        "values": list(values),
        "requested_source_start_sec": round(requested_source_start_sec, 6),
        "requested_source_end_sec": round(requested_source_end_sec, 6),
        "source_start_sec": round(source_start_frame / sample_rate, 6),
        "source_end_sec": round(copy_source_end_frame / sample_rate, 6),
        "target_start_sec": round(requested_target_start_sec, 6),
        "target_end_sec": round(requested_target_end_sec, 6),
        "copied_target_start_sec": round(target_start_frame / sample_rate, 6),
        "copied_target_end_sec": round(copy_target_end_frame / sample_rate, 6),
        "source_frame_range": [source_start_frame, source_end_frame],
        "target_frame_range": [target_start_frame, copy_target_end_frame],
        "target_available_frame_range": [target_start_frame, target_end_frame],
        "source_frames": source_frames,
        "target_available_frames": target_available_frames,
        "copied_frames": copied_frames,
        "required_tempo": round(required_tempo, 6),
        "truncated": truncated,
        "copy_status": copy_status,
    }
    if skipped_reason is not None:
        placement["skipped_reason"] = skipped_reason
    placements.append(placement)


def render_from_phrase_candidate(
    plan: NumericRenderPlan,
    phrase_audio: np.ndarray,
    sample_rate: int,
    word_timing: list[dict[str, Any]],
    *,
    head_audio: np.ndarray | None = None,
) -> RenderedNumericBed:
    """Render a numeric phrase candidate into a silent bed using broad-span copy policies."""
    phrase = _as_audio_matrix(phrase_audio)
    bed = np.zeros((_frame(plan.target_duration_sec, sample_rate), 2), dtype=np.float32)
    timings = _timing_by_value(word_timing)
    placements: list[dict[str, Any]] = []

    if plan.render_policy == "head_single_rest":
        if head_audio is None:
            raise ValueError("head_audio is required for head_single_rest numeric rendering")
        head = _as_audio_matrix(head_audio)
        _copy_chunk(
            bed=bed,
            source_audio=head,
            sample_rate=sample_rate,
            values=[10],
            source_start_sec=0.0,
            source_end_sec=min(len(head) / sample_rate, 0.70),
            target_start_sec=0.08,
            target_end_sec=1.045,
            placements=placements,
        )
        if 9 not in timings or 1 not in timings:
            raise ValueError("word_timing must include values 9 and 1 for head_single_rest rendering")
        _copy_chunk(
            bed=bed,
            source_audio=phrase,
            sample_rate=sample_rate,
            values=[9, 8, 7, 6, 5, 4, 3, 2, 1],
            source_start_sec=timings[9]["source_start_sec"],
            source_end_sec=timings[1]["source_end_sec"] + 0.32,
            target_start_sec=1.08,
            target_end_sec=plan.target_duration_sec - 0.12,
            placements=placements,
        )
    elif plan.render_policy == "whole_span_guard120_pad350":
        span_timing = _ordered_span_timing(word_timing, plan.values)
        if span_timing is None:
            raise ValueError("word_timing must include the first and last plan values")
        first_timing, last_timing = span_timing
        _copy_chunk(
            bed=bed,
            source_audio=phrase,
            sample_rate=sample_rate,
            values=plan.values,
            source_start_sec=float(first_timing["source_start_sec"]) - 0.08,
            source_end_sec=float(last_timing["source_end_sec"]) + 0.35,
            target_start_sec=0.12,
            target_end_sec=plan.target_duration_sec - 0.16,
            placements=placements,
        )
    else:
        raise ValueError(f"Unsupported numeric render policy: {plan.render_policy}")

    max_tempo = max((float(placement["required_tempo"]) for placement in placements), default=1.0)

    return RenderedNumericBed(
        audio=bed,
        sample_rate=sample_rate,
        policy=plan.render_policy,
        placements=placements,
        max_tempo=max_tempo,
    )


def _plan_payload(plan: NumericRenderPlan) -> dict[str, Any]:
    payload = asdict(plan)
    payload["kind"] = str(plan.kind)
    return payload


def _request_payload(request: NumericPhraseRequest) -> dict[str, Any]:
    return request.as_payload()


def _text_for_values(plan: NumericRenderPlan, values: list[int]) -> str:
    token_by_value = {value: token for value, token in zip(plan.values, plan.tokens, strict=False)}
    tokens = [token_by_value.get(value, str(value)) for value in values]
    separator = ", " if plan.text_variant == "native_countdown" else " "
    return separator.join(tokens) + "."


def _plan_for_values(plan: NumericRenderPlan, values: list[int]) -> NumericRenderPlan:
    token_by_value = {value: token for value, token in zip(plan.values, plan.tokens, strict=False)}
    tokens = [token_by_value.get(value, str(value)) for value in values]
    return replace(plan, values=list(values), tokens=tokens, text=_text_for_values(plan, values), groups=[list(values)])


def _required_timing_values(plan: NumericRenderPlan) -> list[int]:
    if plan.render_policy == "head_single_rest":
        return [9, 1]
    if plan.render_policy == "whole_span_guard120_pad350":
        return [plan.values[0], plan.values[-1]]
    return []


def _head_and_body_values(plan: NumericRenderPlan) -> tuple[list[int], list[int]]:
    groups = [list(group) for group in getattr(plan, "groups", []) or []]
    head = next((group for group in groups if group == [10]), None)
    body = next((group for group in groups if group != [10]), None)
    if head is not None and body is not None:
        return head, body
    return [10], [value for value in plan.values if value != 10]


def _word_attr(word: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(word, dict) and name in word:
            return word[name]
        if hasattr(word, name):
            return getattr(word, name)
    return default


def _chunks_to_transcript_and_timing(chunks: list[Any]) -> tuple[str, list[dict[str, Any]]]:
    transcript_parts: list[str] = []
    word_timing: list[dict[str, Any]] = []
    for chunk in chunks:
        text = _word_attr(chunk, "text", default="")
        if str(text).strip():
            transcript_parts.append(str(text).strip())
        for word in _word_attr(chunk, "words", default=[]) or []:
            raw_word = str(_word_attr(word, "word", "text", default="")).strip()
            if not raw_word:
                continue
            explicit_value = _word_attr(word, "value", default=None)
            values = [int(explicit_value)] if explicit_value is not None else extract_korean_numeric_values(raw_word)
            if len(values) != 1:
                continue
            word_timing.append(
                {
                    "value": int(values[0]),
                    "source_start_sec": float(_word_attr(word, "source_start_sec", "start", default=0.0)),
                    "source_end_sec": float(_word_attr(word, "source_end_sec", "end", default=0.0)),
                    "word": raw_word,
                }
            )
    return " ".join(transcript_parts).strip(), word_timing


def _transcribe_numeric_candidate(
    *,
    asr_backend: Any,
    audio_path: Path,
    expected_values: list[int],
    required_timing_values: list[int],
    segment_id: str,
    duration_sec: float,
) -> dict[str, Any]:
    qc_segment = Segment(
        id=segment_id,
        start=0.0,
        end=max(0.001, duration_sec),
        duration=max(0.001, duration_sec),
        audio_for_gemma=str(audio_path),
        audio_for_mix=str(audio_path),
    )
    if hasattr(asr_backend, "transcribe_with_options"):
        chunks = asr_backend.transcribe_with_options(
            audio_path,
            [qc_segment],
            word_timestamps=True,
            language="ko",
            condition_on_previous_text=False,
            vad_filter=False,
        )
    else:
        chunks = asr_backend.transcribe(audio_path, [qc_segment])
    transcript, word_timing = _chunks_to_transcript_and_timing(list(chunks))
    qc = evaluate_numeric_render_transcript(
        expected_values,
        transcript,
        word_timing=word_timing,
        required_timing_values=required_timing_values,
    )
    qc["audio_path"] = str(audio_path)
    qc["segment_id"] = segment_id
    return qc


def _combine_numeric_qc(parts: list[dict[str, Any]], expected_values: list[int]) -> dict[str, Any]:
    transcript = " ".join(str(part.get("transcript") or "").strip() for part in parts).strip()
    observed_values: list[int] = []
    word_timing: list[dict[str, Any]] = []
    issues: list[str] = []
    missing_timing_values: list[int] = []
    for part in parts:
        observed_values.extend(int(value) for value in part.get("observed_values", []))
        word_timing.extend(dict(item) for item in part.get("word_timing", []))
        issues.extend(str(issue) for issue in part.get("issues", []))
        missing_timing_values.extend(int(value) for value in part.get("missing_timing_values", []))
    if observed_values != list(expected_values) and "numeric_sequence_mismatch" not in issues:
        issues.append("numeric_sequence_mismatch")
    repeated_timing_mismatches = _repeated_timing_mismatches(list(expected_values), word_timing)
    if repeated_timing_mismatches and "repeated_numeric_word_timing_mismatch" not in issues:
        issues.append("repeated_numeric_word_timing_mismatch")
    unique_issues = list(dict.fromkeys(issues))
    return {
        "gate": "pass" if not unique_issues else "fail",
        "expected_values": list(expected_values),
        "observed_values": observed_values,
        "transcript": transcript,
        "word_timing": word_timing,
        "expected_value_counts": _value_counts(list(expected_values)),
        "word_timing_value_counts": _value_counts([int(item["value"]) for item in word_timing]),
        "repeated_timing_mismatches": repeated_timing_mismatches,
        "required_timing_values": [value for part in parts for value in part.get("required_timing_values", [])],
        "missing_timing_values": list(dict.fromkeys(missing_timing_values)),
        "issues": unique_issues,
        "parts": parts,
    }


def _get_asr_backend(asr_backend: Any | None, asr_backend_factory: Callable[[], Any] | None) -> Any:
    if asr_backend is not None:
        return asr_backend
    if asr_backend_factory is not None:
        return asr_backend_factory()
    raise ValueError("asr_backend or asr_backend_factory is required for live numeric phrase rendering")


def _failure_result(
    *,
    output_path: Path,
    reason: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "status": "failed",
        "reason": reason,
        "output_path": str(output_path),
        "payload": payload,
    }


def _missing_ref_fields(ref: Any) -> list[str]:
    required_fields = ["ref_audio_path", "prompt_text", "prompt_lang"]
    return [field_name for field_name in required_fields if not str(_ref_value(ref, field_name, "")).strip()]


def render_live_numeric_phrase(
    plan: NumericRenderPlan,
    client: Any,
    output_path: Path | str,
    *,
    ref: dict[str, Any],
    asr_backend: Any | None = None,
    asr_backend_factory: Callable[[], Any] | None = None,
    work_dir: Path | str | None = None,
    max_tempo_limit: float = 1.1,
    mock: bool = False,
) -> dict[str, Any]:
    """Synthesize, ASR-QC, and render a numeric phrase plan into a WAV bed."""
    output_path = Path(output_path)
    candidate_dir = Path(work_dir) if work_dir is not None else output_path.parent / f"{output_path.stem}_candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "renderer": "numeric_phrase_renderer",
        "mock": bool(mock),
        "plan": _plan_payload(plan),
        "request": {},
        "candidate_text": plan.text,
        "candidate_generation": [],
        "numeric_qc": {},
        "placements": [],
        "render_policy": plan.render_policy,
        "max_tempo": None,
        "max_tempo_limit": float(max_tempo_limit),
        "rendered_audio_path": str(output_path),
    }

    try:
        missing_ref_fields = _missing_ref_fields(ref)
        if missing_ref_fields:
            payload["missing_ref_fields"] = missing_ref_fields
            return _failure_result(output_path=output_path, reason="missing_ref", payload=payload)
        backend = _get_asr_backend(asr_backend, asr_backend_factory)
        if plan.render_policy == "head_single_rest":
            head_values, body_values = _head_and_body_values(plan)
            head_plan = _plan_for_values(plan, head_values)
            body_plan = _plan_for_values(plan, body_values)
            head_request = build_numeric_phrase_request(head_plan, ref=ref)
            body_request = build_numeric_phrase_request(body_plan, ref=ref)
            head_path = candidate_dir / f"{output_path.stem}_head.wav"
            body_path = candidate_dir / f"{output_path.stem}_body.wav"
            payload["request"] = {
                "head": _request_payload(head_request),
                "body": _request_payload(body_request),
            }
            payload["candidate_text"] = {"head": head_request.text, "body": body_request.text}

            client.synthesize_to_file(head_request, head_path)
            payload["candidate_generation"].append(
                {"role": "head", "text": head_request.text, "path": str(head_path), "request": _request_payload(head_request)}
            )
            client.synthesize_to_file(body_request, body_path)
            payload["candidate_generation"].append(
                {"role": "body", "text": body_request.text, "path": str(body_path), "request": _request_payload(body_request)}
            )

            head_audio, head_sample_rate = load_audio(head_path)
            body_audio, body_sample_rate = load_audio(body_path)
            if head_sample_rate != body_sample_rate:
                head_audio = resample_linear(head_audio, head_sample_rate, body_sample_rate)
                head_sample_rate = body_sample_rate
            head_duration = len(head_audio) / head_sample_rate
            body_duration = len(body_audio) / body_sample_rate
            head_qc = _transcribe_numeric_candidate(
                asr_backend=backend,
                audio_path=head_path,
                expected_values=head_values,
                required_timing_values=[],
                segment_id="numeric_phrase_head_qc",
                duration_sec=head_duration,
            )
            body_qc = _transcribe_numeric_candidate(
                asr_backend=backend,
                audio_path=body_path,
                expected_values=body_values,
                required_timing_values=_required_timing_values(plan),
                segment_id="numeric_phrase_body_qc",
                duration_sec=body_duration,
            )
            numeric_qc = _combine_numeric_qc([head_qc, body_qc], plan.values)
            payload["numeric_qc"] = numeric_qc
            if numeric_qc["gate"] != "pass":
                return _failure_result(output_path=output_path, reason="numeric_qc_failed", payload=payload)

            rendered = render_from_phrase_candidate(
                plan,
                body_audio,
                body_sample_rate,
                body_qc["word_timing"],
                head_audio=head_audio,
            )
        else:
            request = build_numeric_phrase_request(plan, ref=ref)
            candidate_path = candidate_dir / f"{output_path.stem}_phrase.wav"
            payload["request"] = _request_payload(request)
            payload["candidate_text"] = request.text
            client.synthesize_to_file(request, candidate_path)
            payload["candidate_generation"].append(
                {"role": "phrase", "text": request.text, "path": str(candidate_path), "request": _request_payload(request)}
            )
            phrase_audio, sample_rate = load_audio(candidate_path)
            phrase_duration = len(phrase_audio) / sample_rate
            numeric_qc = _transcribe_numeric_candidate(
                asr_backend=backend,
                audio_path=candidate_path,
                expected_values=plan.values,
                required_timing_values=_required_timing_values(plan),
                segment_id="numeric_phrase_qc",
                duration_sec=phrase_duration,
            )
            payload["numeric_qc"] = numeric_qc
            if numeric_qc["gate"] != "pass":
                return _failure_result(output_path=output_path, reason="numeric_qc_failed", payload=payload)
            rendered = render_from_phrase_candidate(plan, phrase_audio, sample_rate, numeric_qc["word_timing"])

        payload["placements"] = rendered.placements
        payload["max_tempo"] = rendered.max_tempo
        if rendered.max_tempo > float(max_tempo_limit):
            return _failure_result(output_path=output_path, reason="max_tempo_exceeded", payload=payload)

        write_audio(output_path, rendered.audio, rendered.sample_rate)
        return {
            "status": "rendered",
            "output_path": str(output_path),
            "duration_sec": round(len(rendered.audio) / rendered.sample_rate, 6),
            "payload": payload,
        }
    except Exception as exc:
        payload["error"] = f"{type(exc).__name__}: {exc}"
        return _failure_result(output_path=output_path, reason="exception", payload=payload)
