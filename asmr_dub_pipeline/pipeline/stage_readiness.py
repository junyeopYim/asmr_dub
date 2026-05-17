from __future__ import annotations

from typing import Any

from asmr_dub_pipeline.schemas import PipelineManifest, Segment

SYNTH_DOWNSTREAM_READY_STATUSES = {"completed", "completed_with_hard_failed_candidates"}
PARTIAL_SYNTH_STATUS = "completed_with_hard_failed_candidates"
NON_BLOCKING_SYNTH_SEGMENT_STATUSES = {
    "needs_manual_review",
    "translation_blocked",
    "quarantined",
    "absorbed",
    "no_speech_detected",
    "non_speech_texture",
}


def _stage_list(stage_state: dict[str, Any], key: str) -> list[str]:
    value = stage_state.get(key)
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item is not None]


def _has_selected_tts(segment: Segment) -> bool:
    return bool(segment.tts and segment.tts.selected_candidate_path)


def synth_ready_for_downstream(
    manifest: PipelineManifest,
    *,
    only_segment_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Return deterministic readiness accounting for stages after synth."""

    synth_state = manifest.stage_state.get("synth", {})
    if not isinstance(synth_state, dict):
        synth_state = {}
    synth_status = str(synth_state.get("status") or "unknown")
    hard_failed_segments = _stage_list(synth_state, "hard_failed_segments")
    selected_segments = _stage_list(synth_state, "selected_segments")
    selected_metadata_present = "selected_segments" in synth_state
    hard_failed_metadata_present = "hard_failed_segments" in synth_state
    segment_by_id = {segment.id: segment for segment in manifest.segments}
    scoped_segments = [
        segment
        for segment in manifest.segments
        if only_segment_ids is None or segment.id in only_segment_ids
    ]

    non_blocking_segments: list[str] = []
    blocking_segments: list[str] = []
    missing_selected_tts_segments: list[str] = []

    if synth_status not in SYNTH_DOWNSTREAM_READY_STATUSES:
        return {
            "ready": False,
            "synth_status": synth_status,
            "reason": f"synth status is not downstream-ready: {synth_status}",
            "hard_failed_segments": hard_failed_segments,
            "non_blocking_segments": [],
            "blocking_segments": [],
            "selected_segment_count": sum(1 for segment in scoped_segments if _has_selected_tts(segment)),
            "missing_selected_tts_count": 0,
            "selected_segments": selected_segments,
            "missing_selected_tts_segments": [],
        }

    for segment in scoped_segments:
        status = str(segment.status)
        if status in NON_BLOCKING_SYNTH_SEGMENT_STATUSES:
            non_blocking_segments.append(segment.id)
            continue
        if not _has_selected_tts(segment):
            blocking_segments.append(segment.id)
            missing_selected_tts_segments.append(segment.id)

    if synth_status == PARTIAL_SYNTH_STATUS:
        if not selected_metadata_present or not hard_failed_metadata_present:
            blocking_segments.extend(
                marker
                for marker, present in (
                    ("<missing:selected_segments>", selected_metadata_present),
                    ("<missing:hard_failed_segments>", hard_failed_metadata_present),
                )
                if not present
            )
        for segment_id in hard_failed_segments:
            if only_segment_ids is not None and segment_id not in only_segment_ids:
                continue
            segment = segment_by_id.get(segment_id)
            if segment is None:
                blocking_segments.append(segment_id)
                continue
            if str(segment.status) not in NON_BLOCKING_SYNTH_SEGMENT_STATUSES:
                blocking_segments.append(segment_id)

    blocking_segments = list(dict.fromkeys(blocking_segments))
    selected_segment_count = sum(1 for segment in scoped_segments if _has_selected_tts(segment))
    missing_selected_tts_count = len(missing_selected_tts_segments)
    ready = not blocking_segments
    if ready:
        reason = "synth is downstream-ready"
    else:
        reason = "synth has blocking downstream segments: " + ", ".join(blocking_segments[:20])
        if len(blocking_segments) > 20:
            reason += " ..."

    return {
        "ready": ready,
        "synth_status": synth_status,
        "reason": reason,
        "hard_failed_segments": hard_failed_segments,
        "non_blocking_segments": non_blocking_segments,
        "blocking_segments": blocking_segments,
        "selected_segment_count": selected_segment_count,
        "missing_selected_tts_count": missing_selected_tts_count,
        "selected_segments": selected_segments,
        "missing_selected_tts_segments": missing_selected_tts_segments,
    }


def require_synth_ready_for_downstream(
    manifest: PipelineManifest,
    stage_name: str,
    *,
    only_segment_ids: set[str] | None = None,
) -> dict[str, Any]:
    readiness = synth_ready_for_downstream(manifest, only_segment_ids=only_segment_ids)
    if readiness["ready"]:
        return readiness
    raise ValueError(f"{stage_name} requires synth downstream readiness. {readiness['reason']}")
