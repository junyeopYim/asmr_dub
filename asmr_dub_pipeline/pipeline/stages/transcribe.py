from __future__ import annotations

# ruff: noqa: F403,F405,I001

from asmr_dub_pipeline.pipeline.context import PipelineContext
from asmr_dub_pipeline.pipeline.stages.common import *
from asmr_dub_pipeline.pipeline.stages.source_separation import run_source_separation_stage


ASR_QUALITY_ERROR_PREFIXES = (
    "asr_low_confidence:",
    "asr_repair_rejected",
    "asr_suspicious_pattern:",
)
ASR_QUALITY_ERROR_VALUES = {
    "missing_asr_text",
    "no_speech_detected",
    "asr_degenerate_repetition",
    "asr_non_speech_texture",
    "asr_short_filler_keep_original_texture",
    "asr_mixed_texture_speech",
    "asr_numeric_runaway",
    "asr_prompt_or_hallucination_leak",
    "asr_sparse_text_density",
}
ASR_WARNING_ANALYSIS_KEYS = (
    "asr_countdown_unverified",
    "asr_numeric_sequence_unverified",
    "asr_sparse_speech_unverified",
)
ASR_WARNING_ANALYSIS_KEY_BY_REASON = {
    "asr_countdown_unverified": "asr_countdown_unverified",
    "asr_numeric_sequence_unverified": "asr_numeric_sequence_unverified",
    "asr_sparse_speech_unverified": "asr_sparse_speech_unverified",
}
ASR_SEGMENT_RETRY_REASON_PREFIXES = (
    "asr_low_confidence:",
    "asr_repair_rejected",
    "asr_suspicious_pattern:",
)
ASR_SEGMENT_RETRY_REASON_VALUES = {
    "asr_degenerate_repetition",
    "asr_excessive_text_density",
    "asr_numeric_runaway",
    "asr_prompt_or_hallucination_leak",
    "asr_sparse_text_density",
}
ASR_CHANNEL_SPLIT_CHANNELS = ("left", "right", "mid", "side")
ASR_CHANNEL_SPLIT_LANE_CHANNELS = ("left", "right")
ASR_CHANNEL_SPLIT_NO_DIALOG_REJECT_REASONS = {
    "empty_candidate",
    "empty_clip",
    "asr_non_speech_texture",
}
ASR_CHANNEL_SPLIT_STYLE = {
    "left": ("L", -0.72, "left_close"),
    "right": ("R", 0.72, "right_close"),
}


def _clear_asr_quality_gate_errors(segment: Segment) -> None:
    segment.errors = [
        error
        for error in segment.errors
        if error not in ASR_QUALITY_ERROR_VALUES
        and not any(error.startswith(prefix) for prefix in ASR_QUALITY_ERROR_PREFIXES)
    ]


def _source_separation_part_audio_inputs(
    project_dir: Path,
    manifest: PipelineManifest,
    selected_audio_path: Path,
) -> list[dict[str, Any]]:
    source_vocals_mono = _resolve_manifest_artifact_path(project_dir, manifest, "source_vocals_mono_16k")
    if source_vocals_mono is None or selected_audio_path.resolve() != source_vocals_mono.resolve():
        return []
    metadata_path = _resolve_manifest_artifact_path(project_dir, manifest, "source_separation_manifest")
    if metadata_path is None or not metadata_path.exists():
        return []
    try:
        metadata = json.loads(metadata_path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return []
    if not isinstance(metadata, dict) or not metadata.get("partwise"):
        return []
    raw_parts = metadata.get("parts")
    if not isinstance(raw_parts, list):
        return []
    folder_parts = _asr_folder_input_parts(manifest)
    folder_part_by_index = {int(part["part_index"]): part for part in folder_parts}
    folder_part_by_path = {
        str(Path(str(part["path"])).expanduser().resolve()): part
        for part in folder_parts
        if part.get("path")
    }
    parts: list[dict[str, Any]] = []
    for raw_part in raw_parts:
        if not isinstance(raw_part, dict):
            continue
        vocals_mono_path = Path(str(raw_part.get("vocals_mono_path") or ""))
        vocals_path = Path(str(raw_part.get("vocals_path") or ""))
        background_path = Path(str(raw_part.get("background_path") or ""))
        if not vocals_mono_path.exists() or not vocals_path.exists():
            continue
        part_index = int(raw_part.get("part_index") or len(parts) + 1)
        folder_part = folder_part_by_index.get(part_index)
        input_path = str(raw_part.get("input_path") or "")
        if input_path:
            folder_part = folder_part_by_path.get(str(Path(input_path).expanduser().resolve())) or folder_part
        skip_reason = _asr_folder_part_skip_reason(folder_part) or _asr_folder_part_skip_reason(raw_part)
        parts.append(
            {
                "part_index": part_index,
                "start_sec": float(raw_part.get("start_sec") or 0.0),
                "end_sec": float(raw_part.get("end_sec") or 0.0),
                "duration_sec": float(raw_part.get("duration_sec") or 0.0),
                "vocals_mono_path": str(vocals_mono_path),
                "vocals_path": str(vocals_path),
                "background_path": str(background_path) if background_path else "",
                "asr_silenced": skip_reason is not None,
                "asr_skip_reason": skip_reason,
            }
        )
    return parts


def _partwise_audio_duration(parts: list[dict[str, Any]]) -> float | None:
    if not parts:
        return None
    return max(float(part.get("end_sec") or 0.0) for part in parts)


def _transcribe_partwise_audio(backend: Any, parts: list[dict[str, Any]]) -> list[ASRChunk]:
    chunks: list[ASRChunk] = []
    for part in parts:
        if _asr_folder_part_skip_reason(part) is not None:
            continue
        offset = float(part["start_sec"])
        part_path = Path(str(part["vocals_mono_path"]))
        for chunk in backend.transcribe(part_path, []):
            words = [
                word.model_copy(
                    update={
                        "start": round(offset + float(word.start), 6),
                        "end": round(offset + float(word.end), 6),
                    }
                )
                for word in chunk.words
            ]
            chunks.append(
                chunk.model_copy(
                    update={
                        "start": round(offset + float(chunk.start), 6),
                        "end": round(offset + float(chunk.end), 6),
                        "words": words,
                    }
                )
            )
    return chunks


def _asr_segment_retry_reasons(review_reasons: list[str]) -> list[str]:
    return [
        reason
        for reason in review_reasons
        if reason in ASR_SEGMENT_RETRY_REASON_VALUES
        or any(reason.startswith(prefix) for prefix in ASR_SEGMENT_RETRY_REASON_PREFIXES)
    ]


def _absolute_retry_chunks(
    chunks: list[ASRChunk],
    *,
    clip_start: float,
    clip_end: float,
    source_start: float,
    source_end: float,
) -> ASRBoundaryClipResult:
    return _clip_asr_chunks_to_window(
        chunks,
        clip_start=clip_start,
        clip_end=clip_end,
        window_start=source_start,
        window_end=source_end,
        require_word_timestamps_for_boundary=(
            clip_start < source_start - 0.02 or clip_end > source_end + 0.02
        ),
    )


def _source_script_from_retry_chunks(
    source_script: SourceScript,
    chunks: list[ASRChunk],
    *,
    backend_name: str,
    candidate_id: str,
    cfg: Any,
) -> SourceScript | None:
    text = _asr_candidate_text(chunks)
    if not text:
        return None
    return SourceScript(
        text=text,
        language=next((chunk.language for chunk in chunks if chunk.language), cfg.asr_language),
        confidence=_asr_candidate_confidence(chunks),
        backend=f"{backend_name}:segment_retry:{candidate_id}",
        start=float(source_script.start),
        end=float(source_script.end),
    )


def _asr_segment_retry_candidate_score(
    original: SourceScript,
    candidate_source: SourceScript,
) -> float:
    original_duration = max(0.001, float(original.end) - float(original.start))
    candidate_duration = max(0.001, float(candidate_source.end) - float(candidate_source.start))
    original_density = len(original.text.strip()) / original_duration
    candidate_density = len(candidate_source.text.strip()) / candidate_duration
    confidence = float(candidate_source.confidence) if candidate_source.confidence is not None else 0.0
    return (candidate_density - original_density) + confidence


def _attempt_asr_segment_local_retry(
    source_script: SourceScript,
    *,
    segment: Segment,
    backend: Any,
    project_dir: Path,
    retry_audio_path: Path,
    audio_duration_sec: float,
    cfg: Any,
    review_reasons: list[str],
    summary: dict[str, Any],
) -> SourceScript | None:
    retry_reasons = _asr_segment_retry_reasons(review_reasons)
    if not retry_reasons:
        summary["skipped"] += 1
        return None

    source_start = max(0.0, float(source_script.start))
    source_end = min(float(audio_duration_sec), max(source_start, float(source_script.end)))
    if source_end - source_start <= 0.05:
        summary["skipped"] += 1
        return None

    summary["attempted"] += 1
    item: dict[str, Any] = {
        "segment_id": segment.id,
        "start": round(source_start, 3),
        "end": round(source_end, 3),
        "original_text": source_script.text,
        "original_reasons": list(review_reasons),
        "retry_reasons": retry_reasons,
        "accepted": False,
        "accepted_candidate_id": None,
        "accepted_text": None,
        "accepted_vote_count": 0,
        "candidates": [],
    }
    accepted_candidates: list[tuple[str, list[ASRChunk], float]] = []
    retry_dir = project_dir / "work" / "transcribe" / "asr_segment_retry_clips"
    for option in _asr_repair_candidate_options(cfg):
        candidate_id = str(option["candidate_id"])
        padding = float(option.get("padding_sec") or 0.0)
        clip_start = max(0.0, source_start - padding)
        clip_end = min(float(audio_duration_sec), source_end + padding)
        candidate_row: dict[str, Any] = {
            "candidate_id": candidate_id,
            "clip_start": round(clip_start, 3),
            "clip_end": round(clip_end, 3),
            "accepted_for_vote": False,
            "reject_reason": None,
            "text": "",
            "confidence": None,
        }
        if clip_end - clip_start <= 0.05:
            candidate_row["reject_reason"] = "empty_clip"
            item["candidates"].append(candidate_row)
            continue
        clip_path = retry_dir / f"{segment.id}_{candidate_id}.wav"
        try:
            ffmpeg.slice_audio(
                retry_audio_path,
                clip_start,
                clip_end,
                clip_path,
                sample_rate=cfg.gemma_sample_rate,
                channels=1,
            )
            local_chunks = _transcribe_with_backend_options(
                backend,
                clip_path,
                [],
                **dict(option.get("overrides") or {}),
            )
        except Exception as exc:  # pragma: no cover - exercised through summary in live runs
            candidate_row["reject_reason"] = f"retry_error:{type(exc).__name__}"
            candidate_row["error"] = str(exc)
            item["candidates"].append(candidate_row)
            continue

        boundary_result = _absolute_retry_chunks(
            local_chunks,
            clip_start=clip_start,
            clip_end=clip_end,
            source_start=source_start,
            source_end=source_end,
        )
        if boundary_result.reject_reason is not None:
            candidate_row["reject_reason"] = boundary_result.reject_reason
            item["candidates"].append(candidate_row)
            continue
        candidate_chunks = boundary_result.chunks
        candidate_text = _asr_candidate_text(candidate_chunks)
        candidate_row["text"] = candidate_text
        candidate_row["confidence"] = _asr_candidate_confidence(candidate_chunks)
        candidate_row["boundary_clipped"] = boundary_result.boundary_clipped
        candidate_row["dropped_word_count"] = boundary_result.dropped_word_count
        candidate_source = _source_script_from_retry_chunks(
            source_script,
            candidate_chunks,
            backend_name=str(getattr(backend, "name", "asr")),
            candidate_id=candidate_id,
            cfg=cfg,
        )
        if candidate_source is None:
            candidate_row["reject_reason"] = "empty_candidate"
            item["candidates"].append(candidate_row)
            continue
        if _asr_candidate_looks_prompt_leaked(candidate_source.text, cfg):
            candidate_row["reject_reason"] = "prompt_or_hallucination_leak"
            item["candidates"].append(candidate_row)
            continue
        texture_reason = _source_script_non_speech_texture_reason(candidate_source)
        if texture_reason is not None:
            candidate_row["reject_reason"] = texture_reason
            item["candidates"].append(candidate_row)
            continue
        candidate_review_reasons = _source_script_asr_review_reasons(candidate_source, cfg)
        if candidate_review_reasons:
            candidate_row["reject_reason"] = "candidate_review_blocked:" + candidate_review_reasons[0]
            candidate_row["review_reasons"] = candidate_review_reasons
            item["candidates"].append(candidate_row)
            continue
        score = _asr_segment_retry_candidate_score(source_script, candidate_source)
        candidate_row["accepted_for_vote"] = True
        candidate_row["score"] = round(score, 6)
        accepted_candidates.append((candidate_id, candidate_chunks, score))
        item["candidates"].append(candidate_row)

    vote = _select_voted_asr_repair_candidate(accepted_candidates, cfg=cfg)
    if vote is None:
        item["reject_reason"] = "no_candidate_vote"
        summary["items"].append(item)
        return None

    retry_source_script = _source_script_from_retry_chunks(
        source_script,
        list(vote.chunks),
        backend_name=str(getattr(backend, "name", "asr")),
        candidate_id=vote.candidate_id,
        cfg=cfg,
    )
    if retry_source_script is None:
        item["reject_reason"] = "empty_voted_candidate"
        summary["items"].append(item)
        return None

    summary["repaired"] += 1
    item["accepted"] = True
    item["accepted_candidate_id"] = vote.candidate_id
    item["accepted_text"] = retry_source_script.text
    item["accepted_vote_count"] = vote.vote_count
    item["accepted_normalized_text"] = vote.normalized_text
    item["accepted_score"] = round(vote.score, 6)
    summary["items"].append(item)
    return retry_source_script


@dataclass(frozen=True)
class ASRChannelSplitCandidate:
    channel: str
    candidate_id: str
    clip_path: Path
    clip_start: float
    clip_end: float
    chunks: list[ASRChunk]
    text: str
    confidence: float | None
    boundary_clipped: bool
    dropped_word_count: int
    review_reasons: list[str]
    reject_reason: str | None = None


def _reset_generated_channel_split_segments(manifest: PipelineManifest) -> int:
    generated_parent_ids = {
        str(segment.parent_segment_id)
        for segment in manifest.segments
        if segment.parent_segment_id
        and (
            segment.analysis.get("generated_by") == "asr_channel_split"
            or segment.source_lane is not None
        )
    }
    if not generated_parent_ids:
        return 0
    kept: list[Segment] = []
    removed = 0
    for segment in manifest.segments:
        if (
            segment.parent_segment_id in generated_parent_ids
            and segment.analysis.get("generated_by") == "asr_channel_split"
        ):
            removed += 1
            continue
        if segment.id in generated_parent_ids:
            segment.source_lanes = []
            segment.source_lane = None
            segment.analysis.pop("stereo_lane_split", None)
            if segment.status == "absorbed":
                segment.status = "raw"
        kept.append(segment)
    manifest.segments = kept
    return removed


def _channel_split_trigger_reasons(segment: Segment) -> list[str]:
    if segment.source_script is None or not segment.source_script.text.strip():
        return []
    reasons: list[str] = []
    quality_gate = segment.analysis.get("asr_quality_gate")
    if isinstance(quality_gate, dict):
        reasons.extend(str(reason) for reason in quality_gate.get("reasons", []) if reason)
        reasons.extend(str(reason) for reason in quality_gate.get("warnings", []) if reason)
    reasons.extend(str(error) for error in segment.errors if error)
    for key in ASR_WARNING_ANALYSIS_KEYS:
        if key in segment.analysis:
            reasons.append(key)
    if _asr_text_has_unstable_embedded_numeric_sequence(segment.source_script.text):
        reasons.append("asr_unstable_embedded_numeric_sequence")
    return list(dict.fromkeys(reasons))


def _segment_should_channel_split(segment: Segment, cfg: Any) -> tuple[bool, list[str]]:
    if not bool(getattr(cfg, "asr_channel_split_enabled", True)):
        return False, []
    if segment.parent_segment_id or segment.status in NO_SPEECH_STATUSES:
        return False, []
    reasons = _channel_split_trigger_reasons(segment)
    if not reasons:
        return False, []
    trigger = any(
        reason in {
            "asr_countdown_unverified",
            "asr_numeric_sequence_unverified",
            "asr_numeric_runaway",
            "asr_unstable_embedded_numeric_sequence",
        }
        or reason.startswith("asr_repair_rejected")
        for reason in reasons
    )
    return trigger, reasons


def _audio_channel_count(path: Path) -> int | None:
    try:
        return int(sf.info(str(path)).channels)
    except Exception:
        try:
            return ffmpeg.probe_media(path).channels
        except Exception:
            return None


def _channel_split_candidate_specs(cfg: Any) -> list[tuple[str, float]]:
    specs = [
        ("exact", 0.0),
        ("padded", float(getattr(cfg, "asr_channel_split_padding_sec", 1.4))),
        ("wide", float(getattr(cfg, "asr_channel_split_wide_padding_sec", 3.0))),
    ]
    deduped: list[tuple[str, float]] = []
    seen: set[float] = set()
    for candidate_id, padding in specs:
        normalized = max(0.0, round(float(padding), 3))
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append((candidate_id, normalized))
    return deduped


def _shift_asr_chunks(chunks: Sequence[ASRChunk], offset_sec: float) -> list[ASRChunk]:
    shifted: list[ASRChunk] = []
    for chunk in chunks:
        words = [
            word.model_copy(
                update={
                    "start": round(float(word.start) + offset_sec, 6),
                    "end": round(float(word.end) + offset_sec, 6),
                }
            )
            for word in chunk.words
        ]
        shifted.append(
            chunk.model_copy(
                update={
                    "start": round(float(chunk.start) + offset_sec, 6),
                    "end": round(float(chunk.end) + offset_sec, 6),
                    "words": words,
                }
            )
        )
    return shifted


def _channel_split_clip_window(
    segment: Segment,
    source_path: Path,
    *,
    audio_duration_sec: float,
    padding_sec: float,
) -> dict[str, float | bool]:
    segment_start = max(0.0, float(segment.start))
    segment_end = max(segment_start, float(segment.end))
    segment_duration = max(0.0, segment_end - segment_start)
    try:
        source_duration = max(0.0, float(duration_sec(source_path)))
    except Exception:
        source_duration = max(0.0, float(audio_duration_sec))
    source_is_segment_clip = source_duration < segment_end - 0.05
    if source_is_segment_clip:
        window_start = 0.0
        window_end = min(source_duration, segment_duration) if source_duration else segment_duration
        clip_start = max(0.0, window_start - padding_sec)
        clip_end = min(source_duration or segment_duration, window_end + padding_sec)
        return {
            "slice_start": clip_start,
            "slice_end": clip_end,
            "window_start": window_start,
            "window_end": window_end,
            "reported_clip_start": segment_start + clip_start,
            "reported_clip_end": segment_start + clip_end,
            "shift_offset": segment_start,
            "is_segment_clip": True,
        }
    clip_start = max(0.0, segment_start - padding_sec)
    clip_end = min(float(audio_duration_sec), segment_end + padding_sec)
    return {
        "slice_start": clip_start,
        "slice_end": clip_end,
        "window_start": segment_start,
        "window_end": segment_end,
        "reported_clip_start": clip_start,
        "reported_clip_end": clip_end,
        "shift_offset": 0.0,
        "is_segment_clip": False,
    }


def _source_script_for_channel_candidate(
    segment: Segment,
    candidate: ASRChannelSplitCandidate,
    *,
    backend_name: str,
) -> SourceScript:
    return SourceScript(
        text=candidate.text,
        language=next((chunk.language for chunk in candidate.chunks if chunk.language), "ja"),
        confidence=candidate.confidence,
        backend=f"{backend_name}:channel_split:{candidate.channel}:{candidate.candidate_id}",
        start=float(segment.start),
        end=float(segment.end),
    )


def _transcript_words_from_chunks(chunks: Sequence[ASRChunk]) -> list[TranscriptWord]:
    words: list[TranscriptWord] = []
    for chunk in chunks:
        for word in chunk.words:
            text = word.text.strip()
            if not text:
                continue
            words.append(
                TranscriptWord(
                    start=float(word.start),
                    end=float(word.end),
                    text=text,
                    confidence=word.confidence,
                )
            )
    return words


def _lane_transcript_from_candidate(
    segment: Segment,
    candidate: ASRChannelSplitCandidate,
    *,
    backend_name: str,
) -> SourceLaneTranscript:
    suffix, pan, spatial_style = ASR_CHANNEL_SPLIT_STYLE.get(
        candidate.channel,
        ("M", 0.0, "center"),
    )
    return SourceLaneTranscript(
        lane_id=f"{segment.id}_{suffix}",
        channel=candidate.channel,  # type: ignore[arg-type]
        text=candidate.text,
        language=next((chunk.language for chunk in candidate.chunks if chunk.language), "ja"),
        confidence=candidate.confidence,
        start=float(segment.start),
        end=float(segment.end),
        pan=pan,
        spatial_style=spatial_style,  # type: ignore[arg-type]
        backend=f"{backend_name}:channel_split:{candidate.channel}:{candidate.candidate_id}",
        candidate_id=candidate.candidate_id,
        clip_start=candidate.clip_start,
        clip_end=candidate.clip_end,
        words=_transcript_words_from_chunks(candidate.chunks),
        boundary_clipped=candidate.boundary_clipped,
        review_reasons=list(candidate.review_reasons),
    )


def _channel_candidate_payload(candidate: ASRChannelSplitCandidate) -> dict[str, Any]:
    return {
        "channel": candidate.channel,
        "candidate_id": candidate.candidate_id,
        "clip_path": str(candidate.clip_path),
        "clip_start": round(candidate.clip_start, 3),
        "clip_end": round(candidate.clip_end, 3),
        "text": candidate.text,
        "confidence": candidate.confidence,
        "boundary_clipped": candidate.boundary_clipped,
        "dropped_word_count": candidate.dropped_word_count,
        "review_reasons": list(candidate.review_reasons),
        "reject_reason": candidate.reject_reason,
    }


def _channel_split_candidate_score(candidate: ASRChannelSplitCandidate) -> tuple[int, float, float]:
    confidence = float(candidate.confidence) if candidate.confidence is not None else 0.0
    duration = max(0.001, candidate.clip_end - candidate.clip_start)
    density = len(candidate.text) / duration
    return (len(candidate.text), confidence, density)


def _select_channel_lane_candidate(
    candidates: Sequence[ASRChannelSplitCandidate],
) -> ASRChannelSplitCandidate | None:
    viable = [
        candidate
        for candidate in candidates
        if candidate.reject_reason is None
        and candidate.text.strip()
        and not candidate.review_reasons
    ]
    if not viable:
        return None
    return max(viable, key=_channel_split_candidate_score)


def _channel_split_candidates_confirm_no_dialog_texture(item: Mapping[str, Any]) -> bool:
    candidates = item.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return False
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            return False
        reject_reason = str(candidate.get("reject_reason") or "")
        text = str(candidate.get("text") or "").strip()
        if reject_reason not in ASR_CHANNEL_SPLIT_NO_DIALOG_REJECT_REASONS:
            return False
        if text and reject_reason != "asr_non_speech_texture":
            return False
    return True


def _channel_split_confirms_no_dialog_texture(
    segment: Segment,
    item: Mapping[str, Any],
    cfg: Any,
) -> bool:
    if segment.status != "needs_manual_review":
        return False
    trigger_reasons = [str(reason) for reason in item.get("trigger_reasons", []) if reason]
    if not any(reason.startswith("asr_repair_rejected") for reason in trigger_reasons):
        return False
    duration = max(0.001, float(segment.duration or (float(segment.end) - float(segment.start))))
    max_duration = float(getattr(cfg, "asr_channel_split_no_dialog_texture_max_sec", 1.5))
    if duration > max_duration:
        return False
    return _channel_split_candidates_confirm_no_dialog_texture(item)


def _mark_segment_no_dialog_texture(segment: Segment, *, trigger_reasons: Sequence[str]) -> None:
    _clear_asr_quality_gate_errors(segment)
    segment.status = "non_speech_texture"
    segment.keep_original_texture = True
    if "asr_non_speech_texture" not in segment.errors:
        segment.errors.append("asr_non_speech_texture")
    segment.analysis["asr_quality_gate"] = {
        "decision": "texture",
        "reasons": ["asr_non_speech_texture"],
        "tts_blocked": True,
    }
    segment.analysis["stereo_lane_split"] = {
        "mode": "no_dialog_texture",
        "trigger_reasons": list(trigger_reasons),
        "diagnostic_channels": list(ASR_CHANNEL_SPLIT_CHANNELS),
    }


def _run_channel_split_candidate(
    *,
    segment: Segment,
    channel: str,
    candidate_id: str,
    padding_sec: float,
    backend: Any,
    project_dir: Path,
    audio_duration_sec: float,
    cfg: Any,
) -> ASRChannelSplitCandidate:
    source_path = Path(segment.audio_for_mix)
    window = _channel_split_clip_window(
        segment,
        source_path,
        audio_duration_sec=audio_duration_sec,
        padding_sec=padding_sec,
    )
    slice_start = float(window["slice_start"])
    slice_end = float(window["slice_end"])
    window_start = float(window["window_start"])
    window_end = float(window["window_end"])
    reported_clip_start = float(window["reported_clip_start"])
    reported_clip_end = float(window["reported_clip_end"])
    shift_offset = float(window["shift_offset"])
    is_segment_clip = bool(window["is_segment_clip"])
    clip_path = (
        project_dir
        / "work"
        / "transcribe"
        / "asr_channel_split_clips"
        / f"{segment.id}_{channel}_{candidate_id}.wav"
    )
    if slice_end - slice_start <= 0.05:
        return ASRChannelSplitCandidate(
            channel=channel,
            candidate_id=candidate_id,
            clip_path=clip_path,
            clip_start=reported_clip_start,
            clip_end=reported_clip_end,
            chunks=[],
            text="",
            confidence=None,
            boundary_clipped=False,
            dropped_word_count=0,
            review_reasons=[],
            reject_reason="empty_clip",
        )
    try:
        ffmpeg.slice_audio_channel(
            source_path,
            slice_start,
            slice_end,
            clip_path,
            channel=channel,
            sample_rate=int(getattr(cfg, "gemma_sample_rate", 16_000)),
        )
        local_chunks = _transcribe_with_backend_options(
            backend,
            clip_path,
            [],
            word_timestamps=True,
            condition_on_previous_text=False,
            vad_filter=bool(getattr(cfg, "asr_vad_filter", True)),
            vad_parameters=dict(getattr(cfg, "asr_vad_parameters", {}) or {}) or None,
        )
    except Exception as exc:
        return ASRChannelSplitCandidate(
            channel=channel,
            candidate_id=candidate_id,
            clip_path=clip_path,
            clip_start=reported_clip_start,
            clip_end=reported_clip_end,
            chunks=[],
            text="",
            confidence=None,
            boundary_clipped=False,
            dropped_word_count=0,
            review_reasons=[],
            reject_reason=f"channel_transcribe_error:{type(exc).__name__}",
        )
    boundary_result = _clip_asr_chunks_to_window(
        local_chunks,
        clip_start=slice_start,
        clip_end=slice_end,
        window_start=window_start,
        window_end=window_end,
        require_word_timestamps_for_boundary=padding_sec > 0.0 and not is_segment_clip,
    )
    if boundary_result.reject_reason is not None:
        return ASRChannelSplitCandidate(
            channel=channel,
            candidate_id=candidate_id,
            clip_path=clip_path,
            clip_start=reported_clip_start,
            clip_end=reported_clip_end,
            chunks=[],
            text="",
            confidence=None,
            boundary_clipped=boundary_result.boundary_clipped,
            dropped_word_count=boundary_result.dropped_word_count,
            review_reasons=[],
            reject_reason=boundary_result.reject_reason,
        )
    chunks = _shift_asr_chunks(boundary_result.chunks, shift_offset) if shift_offset else boundary_result.chunks
    text = _asr_candidate_text(chunks)
    confidence = _asr_candidate_confidence(chunks)
    source_script = SourceScript(
        text=text,
        language=next((chunk.language for chunk in chunks if chunk.language), getattr(cfg, "asr_language", "ja")),
        confidence=confidence,
        backend=f"{getattr(backend, 'name', 'asr')}:channel_split:{channel}:{candidate_id}",
        start=float(segment.start),
        end=float(segment.end),
    )
    review_reasons = _source_script_asr_review_reasons(source_script, cfg) if text else []
    reject_reason = "empty_candidate" if not text else None
    if text and _asr_candidate_looks_prompt_leaked(text, cfg):
        reject_reason = "prompt_or_hallucination_leak"
    texture_reason = _source_script_non_speech_texture_reason(source_script) if text else None
    if texture_reason is not None:
        reject_reason = texture_reason
    return ASRChannelSplitCandidate(
        channel=channel,
        candidate_id=candidate_id,
        clip_path=clip_path,
        clip_start=reported_clip_start,
        clip_end=reported_clip_end,
        chunks=chunks,
        text=text,
        confidence=confidence,
        boundary_clipped=boundary_result.boundary_clipped,
        dropped_word_count=boundary_result.dropped_word_count,
        review_reasons=review_reasons,
        reject_reason=reject_reason,
    )


def _normalized_channel_text(text: str, cfg: Any) -> str:
    return _normalize_asr_text_for_repair_equivalence(text, cfg)


def _channel_lane_texts_are_distinct(
    left: ASRChannelSplitCandidate,
    right: ASRChannelSplitCandidate,
    cfg: Any,
) -> bool:
    left_text = _normalized_channel_text(left.text, cfg)
    right_text = _normalized_channel_text(right.text, cfg)
    return bool(left_text and right_text and left_text != right_text)


def _build_channel_lane_segment(
    parent: Segment,
    lane: SourceLaneTranscript,
    candidate: ASRChannelSplitCandidate,
) -> Segment:
    return Segment(
        id=lane.lane_id,
        speaker_id=parent.speaker_id,
        parent_segment_id=parent.id,
        start=parent.start,
        end=parent.end,
        duration=parent.duration,
        audio_for_gemma=str(candidate.clip_path),
        audio_for_mix=parent.audio_for_mix,
        estimated_pan=lane.pan,
        keep_original_texture=False,
        status="transcribed",
        analysis={
            "generated_by": "asr_channel_split",
            "parent_segment_id": parent.id,
            "spatial_style": lane.spatial_style,
            "source_channel": lane.channel,
        },
        source_script=SourceScript(
            text=lane.text,
            language=lane.language,
            confidence=lane.confidence,
            backend=lane.backend,
            start=parent.start,
            end=parent.end,
        ),
        source_lane=lane,
    )


def _apply_channel_aware_asr_split(
    manifest: PipelineManifest,
    *,
    backend: Any,
    project_dir: Path,
    audio_duration_sec: float,
    cfg: Any,
) -> dict[str, Any]:
    removed = _reset_generated_channel_split_segments(manifest)
    summary: dict[str, Any] = {
        "enabled": bool(getattr(cfg, "asr_channel_split_enabled", True)),
        "attempted": 0,
        "split": 0,
        "single_lane": 0,
        "no_dialog_texture": 0,
        "skipped": 0,
        "removed_existing_generated_lanes": removed,
        "items": [],
    }
    if not summary["enabled"]:
        return summary
    max_segments = int(getattr(cfg, "asr_channel_split_max_segments", 80))
    if max_segments <= 0:
        return summary
    next_segments: list[Segment] = []
    attempted = 0
    for segment in manifest.segments:
        next_segments.append(segment)
        should_split, trigger_reasons = _segment_should_channel_split(segment, cfg)
        if not should_split:
            continue
        item: dict[str, Any] = {
            "segment_id": segment.id,
            "trigger_reasons": trigger_reasons,
            "candidates": [],
            "created_lane_segment_ids": [],
            "reject_reason": None,
        }
        if attempted >= max_segments:
            summary["skipped"] += 1
            item["reject_reason"] = "max_segments_exceeded"
            summary["items"].append(item)
            continue
        attempted += 1
        summary["attempted"] += 1
        audio_path = Path(segment.audio_for_mix)
        channel_count = _audio_channel_count(audio_path)
        if channel_count is None or channel_count < 2:
            summary["skipped"] += 1
            item["reject_reason"] = "audio_not_stereo"
            summary["items"].append(item)
            continue
        channel_candidates: dict[str, list[ASRChannelSplitCandidate]] = {
            channel: [] for channel in ASR_CHANNEL_SPLIT_CHANNELS
        }
        for channel in ASR_CHANNEL_SPLIT_CHANNELS:
            for candidate_id, padding_sec in _channel_split_candidate_specs(cfg):
                candidate = _run_channel_split_candidate(
                    segment=segment,
                    channel=channel,
                    candidate_id=candidate_id,
                    padding_sec=padding_sec,
                    backend=backend,
                    project_dir=project_dir,
                    audio_duration_sec=audio_duration_sec,
                    cfg=cfg,
                )
                channel_candidates[channel].append(candidate)
                item["candidates"].append(_channel_candidate_payload(candidate))
        selected = {
            channel: _select_channel_lane_candidate(channel_candidates[channel])
            for channel in ASR_CHANNEL_SPLIT_LANE_CHANNELS
        }
        valid_lanes = {
            channel: candidate
            for channel, candidate in selected.items()
            if candidate is not None
        }
        if not valid_lanes:
            if _channel_split_confirms_no_dialog_texture(segment, item, cfg):
                item["reject_reason"] = "no_dialog_texture"
                _mark_segment_no_dialog_texture(segment, trigger_reasons=trigger_reasons)
                summary["no_dialog_texture"] += 1
            else:
                summary["skipped"] += 1
                item["reject_reason"] = "no_valid_lane_candidates"
            summary["items"].append(item)
            continue
        if (
            "left" in valid_lanes
            and "right" in valid_lanes
            and _channel_lane_texts_are_distinct(valid_lanes["left"], valid_lanes["right"], cfg)
        ):
            lanes = [
                _lane_transcript_from_candidate(
                    segment,
                    valid_lanes[channel],
                    backend_name=str(getattr(backend, "name", "asr")),
                )
                for channel in ASR_CHANNEL_SPLIT_LANE_CHANNELS
            ]
            lane_segments = [
                _build_channel_lane_segment(segment, lane, valid_lanes[lane.channel])
                for lane in lanes
            ]
            segment.status = "absorbed"
            segment.source_lanes = lanes
            segment.source_lane = None
            segment.analysis["stereo_lane_split"] = {
                "mode": "split",
                "created_lane_segment_ids": [lane_segment.id for lane_segment in lane_segments],
                "trigger_reasons": trigger_reasons,
                "diagnostic_channels": ["mid", "side"],
            }
            item["created_lane_segment_ids"] = [lane_segment.id for lane_segment in lane_segments]
            next_segments.extend(lane_segments)
            summary["split"] += 1
            summary["items"].append(item)
            continue
        channel, candidate = max(valid_lanes.items(), key=lambda entry: _channel_split_candidate_score(entry[1]))
        lane = _lane_transcript_from_candidate(
            segment,
            candidate,
            backend_name=str(getattr(backend, "name", "asr")),
        )
        segment.source_lane = lane
        segment.source_lanes = [lane]
        segment.source_script = _source_script_for_channel_candidate(
            segment,
            candidate,
            backend_name=str(getattr(backend, "name", "asr")),
        )
        segment.status = "transcribed"
        segment.analysis["spatial_style"] = lane.spatial_style
        segment.analysis["stereo_lane_split"] = {
            "mode": "single_lane",
            "selected_channel": channel,
            "trigger_reasons": trigger_reasons,
            "diagnostic_channels": ["mid", "side"],
        }
        _clear_asr_quality_gate_errors(segment)
        summary["single_lane"] += 1
        summary["items"].append(item)
    manifest.segments = next_segments
    return summary


def _transcribe_rows_from_segments(segments: Sequence[Segment]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for segment in segments:
        quality_gate = segment.analysis.get("asr_quality_gate")
        reasons: list[str] = []
        if isinstance(quality_gate, dict):
            reasons = [str(reason) for reason in quality_gate.get("reasons", []) if reason]
        rows.append(
            {
                "segment_id": segment.id,
                "status": segment.status,
                "review_reasons": reasons,
                "source_script": segment.source_script.model_dump(mode="json")
                if segment.source_script
                else None,
            }
        )
    return rows


def _transcribe_status_metrics(segments: Sequence[Segment]) -> dict[str, int]:
    with_text = sum(
        1
        for segment in segments
        if segment.source_script is not None
        and segment.source_script.text.strip()
        and segment.status not in {"absorbed", *NO_SPEECH_STATUSES}
    )
    manual_review = sum(1 for segment in segments if segment.status == "needs_manual_review")
    no_speech = sum(1 for segment in segments if segment.status in NO_SPEECH_STATUSES)
    non_speech_texture = sum(1 for segment in segments if segment.status == "non_speech_texture")
    return {
        "with_text": with_text,
        "manual_review": manual_review,
        "no_speech": no_speech,
        "non_speech_texture": non_speech_texture,
    }


def run_transcribe_stage(ctx: PipelineContext, asr_backend: str | None = None, confirm_rights: bool = False, asr_review: bool | None = None, asr_preset: str | None = None, asr_vad_off: bool | None = None, asr_diagnostics: bool | None = None, asr_device: str | None = None, asr_compute_type: str | None = None, asr_batched_inference: bool | None = None, asr_batch_size: int | None = None, asr_repair_enabled: bool | None = None, asr_backend_factory: Any | None = None) -> PipelineManifest:
    project_dir = ctx.project_dir
    manifest = ctx.reload_manifest()
    _load_config_into_manifest(project_dir, manifest)
    cfg = manifest.project_config
    if asr_review is not None:
        next_cfg = type(cfg).model_validate(
            {**cfg.model_dump(mode="json"), "asr_review_enabled": asr_review}
        )
        next_cfg.asr.correction_profile = cfg.asr.correction_profile
        cfg = next_cfg
        manifest.project_config = cfg
    cfg = _effective_asr_config(
        cfg,
        asr_preset=asr_preset,
        asr_vad_off=asr_vad_off,
        asr_diagnostics=asr_diagnostics,
        asr_device=asr_device,
        asr_compute_type=asr_compute_type,
        asr_batched_inference=asr_batched_inference,
        asr_batch_size=asr_batch_size,
        asr_repair_enabled=asr_repair_enabled,
    )
    manifest.project_config = cfg
    backend_kind = asr_backend or cfg.asr_backend
    total = len(manifest.segments)
    _log_stage_start(
        "transcribe",
        f"backend={backend_kind}, preset={cfg.asr_preset}, segments={total}",
    )
    _require_audio_stage_rights(manifest, "transcribe", confirm_rights, metadata={"backend": backend_kind})
    if backend_kind != "mock" and "source_vocals_mono_16k" not in manifest.artifacts:
        _log_stage_checkpoint(
            "transcribe",
            "source separation required",
            "missing=source_vocals_mono_16k",
        )
        manifest = run_source_separation_stage(ctx, confirm_rights=confirm_rights)
        _load_config_into_manifest(project_dir, manifest)
        cfg = manifest.project_config
        if asr_review is not None:
            next_cfg = type(cfg).model_validate(
                {**cfg.model_dump(mode="json"), "asr_review_enabled": asr_review}
            )
            next_cfg.asr.correction_profile = cfg.asr.correction_profile
            cfg = next_cfg
            manifest.project_config = cfg
        cfg = _effective_asr_config(
            cfg,
            asr_preset=asr_preset,
            asr_vad_off=asr_vad_off,
            asr_diagnostics=asr_diagnostics,
            asr_device=asr_device,
            asr_compute_type=asr_compute_type,
            asr_batched_inference=asr_batched_inference,
            asr_batch_size=asr_batch_size,
            asr_repair_enabled=asr_repair_enabled,
        )
        manifest.project_config = cfg
        total = len(manifest.segments)
        _log_stage_checkpoint("transcribe", "source separation complete", f"segments={total}")
    audio_path, mix_audio_path, input_diagnostics = _select_asr_audio_input(
        project_dir,
        manifest,
        backend_kind=backend_kind,
        cfg=cfg,
    )
    selected_source = input_diagnostics.get("selected", {}).get("source", "unknown")
    _log_stage_checkpoint(
        "transcribe",
        "audio selected",
        f"source={selected_source} audio={audio_path.name} segments={total}",
    )
    part_audio_inputs = _source_separation_part_audio_inputs(project_dir, manifest, audio_path)
    if part_audio_inputs:
        input_diagnostics["partwise_source_separation"] = {
            "enabled": True,
            "part_count": len(part_audio_inputs),
            "audio_paths": [part["vocals_mono_path"] for part in part_audio_inputs],
        }
        _log_stage_checkpoint(
            "transcribe",
            "partwise source separation enabled",
            f"parts={len(part_audio_inputs)}",
        )
    write_seed_audio_clips = (backend_kind == "mock" or not cfg.asr_resegment_from_chunks) and not part_audio_inputs
    if not manifest.segments:
        _log_stage_checkpoint(
            "transcribe",
            "seeding full-input segment",
            f"write_audio_clips={write_seed_audio_clips}",
        )
    seeded_for_transcribe = _seed_segments_for_transcribe(
        project_dir,
        manifest,
        audio_path,
        mix_audio_path,
        write_audio_clips=write_seed_audio_clips,
    )
    removed_channel_lanes = _reset_generated_channel_split_segments(manifest)
    total = len(manifest.segments)
    if seeded_for_transcribe:
        _log_stage_checkpoint("transcribe", "seed segment ready", f"segments={total}")
    if removed_channel_lanes:
        _log_stage_checkpoint(
            "transcribe",
            "removed generated channel lanes for rerun",
            f"removed={removed_channel_lanes} segments={total}",
        )
    backend_config = _asr_backend_config(cfg)
    _log_stage_checkpoint("transcribe", "creating ASR backend", f"backend={backend_kind}")
    backend = (
        asr_backend_factory(backend_kind, backend_config)
        if asr_backend_factory is not None
        else create_asr_backend(backend_kind, backend_config)
    )
    audio_duration = _partwise_audio_duration(part_audio_inputs) or duration_sec(audio_path)
    _log_stage_checkpoint(
        "transcribe",
        "starting ASR",
        f"backend={backend.name} audio={audio_path.name} duration={audio_duration:.2f}s segments={total}",
    )
    raw_chunks = (
        _transcribe_partwise_audio(backend, part_audio_inputs)
        if part_audio_inputs and backend_kind != "mock"
        else backend.transcribe(audio_path, manifest.segments)
    )
    _log_stage_checkpoint(
        "transcribe",
        "ASR complete",
        f"raw_chunks={len(raw_chunks)} backend={backend.name}",
    )
    chunks = [chunk.model_copy() for chunk in raw_chunks]
    raw_asr_chunk_count = len(raw_chunks)
    asr_text_replacement_count = 0
    asr_text_replacements_summary: dict[str, Any] = {
        "chunks_changed": 0,
        "total_replacements": 0,
        "items": [],
    }
    repair_summary: dict[str, Any] = {
        "enabled": False,
        "attempted": 0,
        "repaired": 0,
        "skipped": 0,
        "items": [],
    }
    segment_retry_summary: dict[str, Any] = {
        "enabled": backend_kind != "mock" and bool(cfg.asr_repair_enabled),
        "attempted": 0,
        "repaired": 0,
        "skipped": 0,
        "items": [],
    }
    asr_review_summary: dict[str, Any] = {
        "enabled": bool(cfg.asr_review_enabled),
        "backend": cfg.asr_review_backend,
        "attempted": 0,
        "reviewed": 0,
        "replaced": 0,
        "guarded_auto_replaced": 0,
        "manual_review": 0,
        "skipped": 0,
        "generated_candidates": 0,
        "error": None,
        "items": [],
    }
    qwen_fallback_backend = None
    qwen_fallback_summary: dict[str, Any] = {
        "enabled": False,
        "available": False,
        "backend": "qwen_asr",
        "skipped_reason": "disabled",
    }
    repaired_chunks = [chunk.model_copy() for chunk in chunks]
    filtered_final_chunks: list[dict[str, Any]] = []
    if backend_kind != "mock":
        _log_stage_checkpoint(
            "transcribe",
            "applying ASR post-processing",
            f"repair={cfg.asr_repair_enabled} review={cfg.asr_review_enabled}",
        )
        qwen_fallback_backend, qwen_fallback_summary = _create_qwen_repair_fallback_backend(
            cfg,
            manifest,
        )
        repair_audio_path = audio_path
        chunks, repair_summary = _repair_asr_chunks(
            chunks,
            backend=backend,
            project_dir=project_dir,
            repair_audio_path=repair_audio_path,
            audio_duration_sec=audio_duration,
            cfg=cfg,
            qwen_fallback_backend=qwen_fallback_backend,
        )
        _log_stage_checkpoint(
            "transcribe",
            "ASR repair complete",
            "attempted={attempted} repaired={repaired} skipped={skipped}".format(
                attempted=repair_summary.get("attempted", 0),
                repaired=repair_summary.get("repaired", 0),
                skipped=repair_summary.get("skipped", 0),
            ),
        )
        repaired_chunks = [chunk.model_copy() for chunk in chunks]
        repair_summary_path = project_dir / "work" / "transcribe" / "asr_repair_summary.json"
        write_json_atomic(repair_summary_path, repair_summary)
        manifest.artifacts["asr_repair_summary"] = str(repair_summary_path)
        chunks, asr_text_replacements_summary = _apply_asr_text_replacements_to_chunks_with_summary(
            chunks,
            {},
            contextual_replacements=cfg.asr_review_candidate_replacements,
        )
        chunks, asr_review_summary = _review_asr_chunks_with_model(
            chunks,
            backend=backend,
            project_dir=project_dir,
            review_audio_path=repair_audio_path,
            audio_duration_sec=audio_duration,
            cfg=cfg,
            qwen_fallback_backend=qwen_fallback_backend,
        )
        _log_stage_checkpoint(
            "transcribe",
            "ASR review complete",
            "attempted={attempted} replaced={replaced} manual_review={manual_review}".format(
                attempted=asr_review_summary.get("attempted", 0),
                replaced=asr_review_summary.get("replaced", 0),
                manual_review=asr_review_summary.get("manual_review", 0),
            ),
        )
        asr_review_summary_path = project_dir / "work" / "transcribe" / "asr_review_summary.json"
        write_json_atomic(asr_review_summary_path, asr_review_summary)
        manifest.artifacts["asr_review_summary"] = str(asr_review_summary_path)
        chunks, post_review_replacements_summary = _apply_asr_text_replacements_to_chunks_with_summary(
            chunks,
            cfg.asr_text_replacements,
            contextual_replacements=cfg.asr_review_candidate_replacements,
        )
        asr_text_replacements_summary = _merge_asr_text_replacement_summaries(
            asr_text_replacements_summary,
            post_review_replacements_summary,
        )
        asr_text_replacement_count = int(asr_text_replacements_summary["chunks_changed"])
        chunks, filtered_final_chunks = _filter_final_asr_chunks_for_hallucinations(
            chunks,
            cfg=cfg,
        )
        if filtered_final_chunks:
            _log_stage_checkpoint(
                "transcribe",
                "filtered hallucinated ASR chunks",
                f"items={len(filtered_final_chunks)}",
            )
    final_chunks = [chunk.model_copy() for chunk in chunks]
    resegmented_from_chunks = False
    previous_segment_count = len(manifest.segments)
    manual_segments_path = project_dir / "work" / "segments" / "manifests" / "segments_manual.json"
    if cfg.asr_resegment_from_chunks and backend_kind != "mock" and chunks and not manual_segments_path.exists():
        _log_stage_checkpoint(
            "transcribe",
            "building segments from ASR chunks",
            f"chunks={len(chunks)} previous_segments={previous_segment_count}",
        )
        resegmented = _segments_from_asr_chunks(
            chunks,
            project_dir=project_dir,
            backend=backend.name,
            fallback_language=cfg.asr_language,
            audio_duration_sec=audio_duration,
            min_segment_sec=cfg.asr_resegment_min_sec,
            merge_gap_sec=cfg.asr_resegment_merge_gap_sec,
            max_segment_sec=cfg.asr_resegment_max_sec,
            sparse_chunk_max_sec=cfg.asr_sparse_chunk_max_sec,
            sparse_chunk_min_chars_per_sec=cfg.asr_sparse_chunk_min_chars_per_sec,
            countdown_merge_enabled=cfg.asr_countdown_merge_enabled,
            countdown_merge_gap_sec=cfg.asr_countdown_merge_gap_sec,
            countdown_merge_max_span_sec=cfg.asr_countdown_merge_max_span_sec,
        )
        if resegmented:
            if part_audio_inputs:
                write_segment_audio_clips_from_parts(resegmented, part_audio_inputs, project_dir)
            else:
                write_segment_audio_clips(resegmented, audio_path, mix_audio_path, project_dir)
            split_resegmented = _split_sparse_edge_segments_by_audio(
                resegmented,
                project_dir=project_dir,
                cfg=cfg,
                merge_gap_sec=cfg.asr_resegment_merge_gap_sec,
            )
            if split_resegmented is not resegmented:
                resegmented = split_resegmented
                if part_audio_inputs:
                    write_segment_audio_clips_from_parts(resegmented, part_audio_inputs, project_dir)
                else:
                    write_segment_audio_clips(resegmented, audio_path, mix_audio_path, project_dir)
            manifest.segments = resegmented
            total = len(manifest.segments)
            resegmented_from_chunks = True
            _log_stage_checkpoint(
                "transcribe",
                "ASR resegmentation complete",
                f"segments={previous_segment_count}->{total}",
            )
    _log_stage_checkpoint(
        "transcribe",
        "mapping ASR chunks to segments",
        f"segments={len(manifest.segments)} chunks={len(chunks)}",
    )
    mapped = (
        {segment.id: segment.source_script for segment in manifest.segments}
        if resegmented_from_chunks
        else map_chunks_to_segments(
            manifest.segments,
            chunks,
            backend=backend.name,
            fallback_language=cfg.asr_language,
        )
    )
    rows: list[dict[str, Any]] = []
    started_at = monotonic()
    last_logged_at = started_at
    with_text = 0
    manual_review_count = 0
    no_speech_count = 0
    non_speech_texture_count = 0
    whole_input_no_speech = backend_kind != "mock" and not final_chunks
    folder_parts = _asr_folder_input_parts(manifest)
    for index, segment in enumerate(manifest.segments, start=1):
        source_script = mapped.get(segment.id)
        segment.source_script = source_script
        source_has_text = bool(source_script and source_script.text.strip())
        folder_part = _asr_part_for_segment_start(folder_parts, float(segment.start))
        folder_skip_reason = _asr_folder_part_skip_reason(folder_part)
        skipped_folder_no_speech = folder_skip_reason is not None and not source_has_text
        no_speech_segment = (whole_input_no_speech and not source_has_text) or skipped_folder_no_speech
        review_reasons = (
            [f"asr_skipped_folder_part:{folder_skip_reason}"]
            if skipped_folder_no_speech and folder_skip_reason
            else ["no_speech_detected"]
            if no_speech_segment
            else _source_script_asr_review_reasons(source_script, cfg)
        )
        non_speech_texture_reason = (
            None if no_speech_segment else _source_script_non_speech_texture_reason(source_script)
        )
        if non_speech_texture_reason is None and not no_speech_segment and source_script is not None:
            non_speech_texture_reason = _source_script_keep_original_texture_override_reason(
                segment,
                source_script,
                cfg,
            )
        keep_original_texture_candidate = (
            None
            if no_speech_segment
            else _source_script_keep_original_texture_candidate(source_script)
        )
        if non_speech_texture_reason is None and keep_original_texture_candidate is not None:
            non_speech_texture_reason = str(keep_original_texture_candidate["reason"])
        repair_review_reasons = _source_script_rejected_repair_reasons(
            source_script,
            repair_summary,
            cfg=cfg,
        )
        asr_warning_reasons: list[str] = []
        if non_speech_texture_reason is None:
            asr_warning_reasons = _source_script_asr_warning_reasons(source_script, cfg)
            repair_review_reasons = _filter_asr_repair_review_reasons(
                source_script,
                cfg,
                review_reasons=review_reasons,
                repair_review_reasons=repair_review_reasons,
            )
            review_reasons.extend(
                reason for reason in repair_review_reasons if reason not in review_reasons
            )
        else:
            review_reasons = [non_speech_texture_reason]
        if (
            segment_retry_summary["enabled"]
            and source_script is not None
            and source_script.text.strip()
            and review_reasons
            and non_speech_texture_reason is None
            and _asr_segment_retry_reasons(review_reasons)
        ):
            retry_source_script = _attempt_asr_segment_local_retry(
                source_script,
                segment=segment,
                backend=backend,
                project_dir=project_dir,
                retry_audio_path=audio_path,
                audio_duration_sec=audio_duration,
                cfg=cfg,
                review_reasons=review_reasons,
                summary=segment_retry_summary,
            )
            if retry_source_script is not None:
                source_script = retry_source_script
                mapped[segment.id] = retry_source_script
                segment.source_script = retry_source_script
                review_reasons = _source_script_asr_review_reasons(source_script, cfg)
                non_speech_texture_reason = _source_script_non_speech_texture_reason(source_script)
                if non_speech_texture_reason is None:
                    non_speech_texture_reason = _source_script_keep_original_texture_override_reason(
                        segment,
                        source_script,
                        cfg,
                    )
                keep_original_texture_candidate = _source_script_keep_original_texture_candidate(
                    source_script
                )
                if non_speech_texture_reason is None and keep_original_texture_candidate is not None:
                    non_speech_texture_reason = str(keep_original_texture_candidate["reason"])
                repair_review_reasons = _source_script_rejected_repair_reasons(
                    source_script,
                    repair_summary,
                    cfg=cfg,
                )
                asr_warning_reasons = []
                if non_speech_texture_reason is None:
                    asr_warning_reasons = _source_script_asr_warning_reasons(source_script, cfg)
                    repair_review_reasons = _filter_asr_repair_review_reasons(
                        source_script,
                        cfg,
                        review_reasons=review_reasons,
                        repair_review_reasons=repair_review_reasons,
                    )
                    review_reasons.extend(
                        reason for reason in repair_review_reasons if reason not in review_reasons
                    )
                else:
                    review_reasons = [non_speech_texture_reason]
                source_has_text = bool(source_script and source_script.text.strip())
        _clear_asr_quality_gate_errors(segment)
        segment.analysis.pop("candidate_keep_original_texture", None)
        for key in ASR_WARNING_ANALYSIS_KEYS:
            segment.analysis.pop(key, None)
        if no_speech_segment:
            segment.status = "no_speech_detected"
            if skipped_folder_no_speech:
                segment.keep_original_texture = True
            no_speech_count += 1
        elif non_speech_texture_reason is not None:
            segment.status = "non_speech_texture"
            segment.keep_original_texture = True
            no_speech_count += 1
            non_speech_texture_count += 1
            if keep_original_texture_candidate is not None:
                segment.analysis["candidate_keep_original_texture"] = keep_original_texture_candidate
            if non_speech_texture_reason not in segment.errors:
                segment.errors.append(non_speech_texture_reason)
        elif review_reasons:
            segment.status = "needs_manual_review"
            manual_review_count += 1
            for reason in review_reasons:
                if reason not in segment.errors:
                    segment.errors.append(reason)
        elif source_has_text:
            segment.status = "transcribed"
        if asr_warning_reasons and not review_reasons:
            for warning_reason in asr_warning_reasons:
                key = ASR_WARNING_ANALYSIS_KEY_BY_REASON.get(warning_reason)
                if key is None:
                    continue
                segment.analysis[key] = {
                    "reason": warning_reason,
                    "source_text": source_script.text.strip() if source_script else "",
                }
        quality_gate = {
            "decision": (
                "no_speech"
                if no_speech_segment
                else "texture"
                if non_speech_texture_reason is not None
                else "block_tts"
                if review_reasons
                else "pass_with_warning"
                if asr_warning_reasons
                else "pass"
            ),
            "reasons": review_reasons,
            "tts_blocked": bool(review_reasons),
        }
        if asr_warning_reasons and not review_reasons:
            quality_gate["warnings"] = asr_warning_reasons
        segment.analysis["asr_quality_gate"] = quality_gate
        if source_has_text and non_speech_texture_reason is None:
            with_text += 1
            status = "needs_manual_review" if review_reasons else "transcribed"
        elif non_speech_texture_reason is not None:
            status = "non_speech_texture"
        elif no_speech_segment:
            status = "no_speech_detected"
        else:
            status = "needs_manual_review"
        rows.append(
            {
                "segment_id": segment.id,
                "status": status,
                "review_reasons": review_reasons,
                "source_script": source_script.model_dump(mode="json") if source_script else None,
            }
        )
        last_logged_at = _log_segment_progress(
            "transcribe", index, total, segment, manifest, started_at, last_logged_at
        )
    channel_split_summary = _apply_channel_aware_asr_split(
        manifest,
        backend=backend,
        project_dir=project_dir,
        audio_duration_sec=audio_duration,
        cfg=cfg,
    )
    if (
        channel_split_summary.get("split", 0)
        or channel_split_summary.get("single_lane", 0)
        or channel_split_summary.get("no_dialog_texture", 0)
        or channel_split_summary.get("removed_existing_generated_lanes", 0)
    ):
        rows = _transcribe_rows_from_segments(manifest.segments)
        metrics = _transcribe_status_metrics(manifest.segments)
        total = len(manifest.segments)
        with_text = metrics["with_text"]
        manual_review_count = metrics["manual_review"]
        no_speech_count = metrics["no_speech"]
        non_speech_texture_count = metrics["non_speech_texture"]
        _log_stage_checkpoint(
            "transcribe",
            "channel-aware ASR split complete",
            "split={split} single_lane={single_lane} segments={segments}".format(
                split=channel_split_summary.get("split", 0),
                single_lane=channel_split_summary.get("single_lane", 0),
                segments=total,
            ),
        )
    _log_stage_checkpoint(
        "transcribe",
        "writing transcription artifacts",
        f"segments={len(manifest.segments)} rows={len(rows)}",
    )
    jsonl_path = project_dir / "work" / "transcribe" / "source_segments.jsonl"
    _write_jsonl_atomic(jsonl_path, rows)
    out_path = project_dir / "work" / "segments" / "manifests" / "segments_transcribed.json"
    write_json_atomic(out_path, {"segments": [s.model_dump(mode="json") for s in manifest.segments]})
    manifest.artifacts["source_segments"] = str(jsonl_path)
    manifest.artifacts["segments_transcribed"] = str(out_path)
    if resegmented_from_chunks:
        manifest.artifacts["segments_asr_resegmented"] = str(out_path)
    segment_retry_summary_path = project_dir / "work" / "transcribe" / "asr_segment_retry_summary.json"
    write_json_atomic(segment_retry_summary_path, segment_retry_summary)
    manifest.artifacts["asr_segment_retry_summary"] = str(segment_retry_summary_path)
    channel_split_summary_path = project_dir / "work" / "transcribe" / "asr_channel_split_summary.json"
    write_json_atomic(channel_split_summary_path, channel_split_summary)
    manifest.artifacts["asr_channel_split_summary"] = str(channel_split_summary_path)
    _write_asr_diagnostics_artifacts(
        project_dir,
        manifest,
        backend_kind=backend_kind,
        backend_name=backend.name,
        cfg=cfg,
        input_diagnostics=input_diagnostics,
        raw_chunks=raw_chunks,
        repaired_chunks=repaired_chunks,
        final_chunks=final_chunks,
        repair_summary=repair_summary,
        asr_review_summary=asr_review_summary,
        replacements_summary=asr_text_replacements_summary,
        filtered_summary=filtered_final_chunks,
        qwen_fallback_summary=qwen_fallback_summary,
        segment_retry_summary=segment_retry_summary,
    )
    _log_stage_checkpoint(
        "transcribe",
        "ASR diagnostics written",
        f"raw_chunks={raw_asr_chunk_count} final_chunks={len(final_chunks)}",
    )
    countdown_summary = _countdown_timeline_summary(manifest.segments)
    _warn_missing_countdown_timelines(
        manifest,
        countdown_summary,
        word_timestamps_enabled=bool(cfg.asr_word_timestamps),
    )
    asr_high_risk_report = _write_asr_high_risk_report_artifact(
        project_dir,
        manifest,
        cfg=cfg,
        replacements_summary=asr_text_replacements_summary,
        repair_summary=repair_summary,
        asr_review_summary=asr_review_summary,
        filtered_summary=filtered_final_chunks,
    )
    asr_high_risk_summary = asr_high_risk_report["summary"]
    asr_postprocess_review = _write_asr_postprocess_review_artifact(
        project_dir,
        manifest,
        cfg=cfg,
        replacements_summary=asr_text_replacements_summary,
    )
    asr_postprocess_review_summary = asr_postprocess_review["summary"]
    mark_stage(
        manifest,
        "transcribe",
        "completed",
        backend=backend_kind,
        backend_name=backend.name,
        asr_preset=cfg.asr_preset,
        asr_device=cfg.asr_device,
        asr_compute_type=cfg.asr_compute_type,
        asr_batched_inference=cfg.asr_batched_inference,
        asr_batch_size=cfg.asr_batch_size,
        asr_word_timestamps=cfg.asr_word_timestamps,
        asr_word_timestamp_chunk_count=sum(1 for chunk in final_chunks if chunk.words),
        asr_word_timestamp_count=sum(len(chunk.words) for chunk in final_chunks),
        asr_input_source=input_diagnostics.get("selected", {}).get("source"),
        segment_count=total,
        previous_segment_count=previous_segment_count,
        raw_asr_chunk_count=raw_asr_chunk_count,
        asr_chunk_count=len(chunks),
        asr_repair_attempted=repair_summary.get("attempted", 0),
        asr_repair_repaired=repair_summary.get("repaired", 0),
        asr_segment_retry_attempted=segment_retry_summary.get("attempted", 0),
        asr_segment_retry_repaired=segment_retry_summary.get("repaired", 0),
        asr_channel_split_attempted=channel_split_summary.get("attempted", 0),
        asr_channel_split_split=channel_split_summary.get("split", 0),
        asr_channel_split_single_lane=channel_split_summary.get("single_lane", 0),
        asr_review_attempted=asr_review_summary.get("attempted", 0),
        asr_review_replaced=asr_review_summary.get("replaced", 0),
        asr_review_guarded_auto_replaced=asr_review_summary.get("guarded_auto_replaced", 0),
        asr_review_manual_review=asr_review_summary.get("manual_review", 0),
        asr_review_error=asr_review_summary.get("error"),
        asr_text_replacements=asr_text_replacement_count,
        seeded_for_transcribe=seeded_for_transcribe,
        resegmented_from_chunks=resegmented_from_chunks,
        transcribed=with_text,
        no_speech_detected=no_speech_count,
        non_speech_texture=non_speech_texture_count,
        needs_manual_review=max(total - with_text - no_speech_count, manual_review_count),
        asr_auto_dub_ready=bool(asr_high_risk_summary["automated_dubbing_ready"]),
        asr_high_risk_warning=int(asr_high_risk_summary["warning"]),
        asr_high_risk_severe=int(asr_high_risk_summary["severe"]),
        asr_high_risk_items=len(asr_high_risk_report["items"]),
        asr_high_risk_blocking_reasons=asr_high_risk_summary["blocking_reasons"],
        asr_postprocess_review_items=int(asr_postprocess_review_summary["item_count"]),
        asr_postprocess_auto_replace=int(asr_postprocess_review_summary["auto_replace"]),
        asr_postprocess_candidate_review=int(
            asr_postprocess_review_summary["candidate_review"]
        ),
        asr_postprocess_manual_review=int(asr_postprocess_review_summary["manual_review"]),
        **countdown_summary,
    )
    save_manifest(project_dir, manifest)
    _log_stage_complete("transcribe", manifest, f"backend={backend_kind}")
    return ctx.update_manifest(manifest)
