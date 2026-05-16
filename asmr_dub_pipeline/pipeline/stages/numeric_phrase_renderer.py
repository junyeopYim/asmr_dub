from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

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
    text_split_method: str = "cut0"
    top_k: int = 5
    top_p: float = 0.85
    temperature: float = 0.65
    repetition_penalty: float = 1.8


def build_numeric_phrase_request(plan: NumericRenderPlan, *, ref: dict[str, Any]) -> NumericPhraseRequest:
    """Build a deterministic low-randomness GPT-SoVITS numeric phrase request."""
    _ = ref
    return NumericPhraseRequest(text=plan.text)


def evaluate_numeric_render_transcript(expected: list[int], transcript: str) -> dict[str, Any]:
    """Compare a transcript's extracted numeric sequence against expected render values."""
    expected_values = list(expected)
    observed_values = extract_korean_numeric_values(transcript)
    pass_gate = observed_values == expected_values
    return {
        "gate": "pass" if pass_gate else "fail",
        "expected_values": list(expected_values),
        "observed_values": list(observed_values),
        "transcript": transcript,
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
        first_value = plan.values[0]
        last_value = plan.values[-1]
        if first_value not in timings or last_value not in timings:
            raise ValueError("word_timing must include the first and last plan values")
        _copy_chunk(
            bed=bed,
            source_audio=phrase,
            sample_rate=sample_rate,
            values=plan.values,
            source_start_sec=timings[first_value]["source_start_sec"] - 0.08,
            source_end_sec=timings[last_value]["source_end_sec"] + 0.35,
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
