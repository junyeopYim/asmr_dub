from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from asmr_dub_pipeline.pipeline.artifacts import make_script_generation_id
from asmr_dub_pipeline.pipeline.context import PipelineContext
from asmr_dub_pipeline.pipeline.manifest_io import save_manifest
from asmr_dub_pipeline.pipeline.runner import run_closure
from asmr_dub_pipeline.pipeline.stage_readiness import (
    NON_BLOCKING_SYNTH_SEGMENT_STATUSES,
    synth_ready_for_downstream,
)
from asmr_dub_pipeline.pipeline.state import mark_stage
from asmr_dub_pipeline.schemas import PipelineManifest, Segment

ClosureRunner = Callable[..., dict[str, Any]]


def _stage_list(stage_state: dict[str, Any], key: str) -> list[str]:
    value = stage_state.get(key)
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item is not None]


def late_qwen_scheduled_segment_ids(manifest: PipelineManifest) -> set[str]:
    """Return late-Qwen repair targets recorded by synth/tts.select state."""

    segment_ids: set[str] = set()
    for stage_name in ("synth", "tts.select"):
        stage_state = manifest.stage_state.get(stage_name)
        if isinstance(stage_state, dict):
            segment_ids.update(_stage_list(stage_state, "late_qwen_scheduled_segments"))
    return segment_ids


def _selected_tts(segment: Segment) -> bool:
    return bool(segment.tts and segment.tts.selected_candidate_path)


def _segment_terminal_for_downstream(segment: Segment) -> bool:
    return str(segment.status) in NON_BLOCKING_SYNTH_SEGMENT_STATUSES


def _current_marker(segment: Segment) -> dict[str, Any]:
    marker = segment.analysis.get("late_qwen_pre_rvc")
    return dict(marker) if isinstance(marker, dict) else {}


def _mark_attempt_started(manifest: PipelineManifest, segment_ids: set[str]) -> None:
    for segment in manifest.segments:
        if segment.id not in segment_ids:
            continue
        previous = _current_marker(segment)
        attempt = int(previous.get("attempt") or 0) + 1
        segment.analysis["late_qwen_pre_rvc"] = {
            **previous,
            "attempted": True,
            "attempt": attempt,
            "requested_backend": "qwen",
            "input_script_generation_id": make_script_generation_id(segment.script),
            "input_status": str(segment.status),
            "output_status": None,
            "resolved": False,
            "terminal_reason": None,
            "selected_candidate_id": None,
            "selected_tts_generation_id": None,
        }


def _append_route_codes(payload: dict[str, Any], codes: list[str]) -> dict[str, Any]:
    existing = payload.get("route_reason_codes")
    values = [str(item) for item in existing] if isinstance(existing, list) else []
    values.extend(codes)
    payload["route_reason_codes"] = list(dict.fromkeys(values))
    return payload


def finalize_unresolved_late_qwen_segments(
    manifest: PipelineManifest,
    segment_ids: set[str],
    reason: str,
) -> list[str]:
    """Turn unresolved late-Qwen pre-RVC targets into terminal manual review."""

    terminal_segments: list[str] = []
    for segment in manifest.segments:
        if segment.id not in segment_ids:
            continue
        if _selected_tts(segment) or _segment_terminal_for_downstream(segment):
            continue
        previous_status = str(segment.status)
        segment.status = "needs_manual_review"
        segment.tts = None
        segment.rvc = None
        segment.qc = None
        segment.mix = {}
        plan = segment.analysis.get("ko_qc_repair_plan")
        if not isinstance(plan, dict):
            plan = {}
        segment.analysis["ko_qc_repair_plan"] = {
            **plan,
            "action": "manual_review",
            "terminal_manual": True,
            "terminal_reason": reason,
            "route": "late_qwen_terminal_after_retry",
            "source": "late_qwen_pre_rvc",
        }
        selection = segment.analysis.get("tts_selection")
        if not isinstance(selection, dict):
            selection = {}
        segment.analysis["tts_selection"] = _append_route_codes(
            {
                **selection,
                "status": "manual_review",
                "terminal_reason": reason,
            },
            ["late_qwen_pre_rvc_attempted", "late_qwen_terminal_after_retry"],
        )
        marker = _current_marker(segment)
        segment.analysis["late_qwen_pre_rvc"] = {
            **marker,
            "attempted": True,
            "input_status": marker.get("input_status") or previous_status,
            "output_status": "needs_manual_review",
            "resolved": False,
            "terminal_reason": reason,
            "selected_candidate_id": None,
            "selected_tts_generation_id": None,
        }
        terminal_segments.append(segment.id)
    return terminal_segments


def _refresh_synth_state(
    manifest: PipelineManifest,
    *,
    attempted_segments: list[str],
    resolved_segments: list[str],
    terminal_segments: list[str],
) -> dict[str, Any]:
    selected_segments = [
        segment.id for segment in manifest.segments if _selected_tts(segment)
    ]
    hard_failed_segments = [
        segment.id for segment in manifest.segments if segment.status == "needs_manual_review"
    ]
    texture_bypassed_segments = [
        segment.id for segment in manifest.segments if segment.status == "non_speech_texture"
    ]
    absorbed_segments = [
        segment.id for segment in manifest.segments if segment.status == "absorbed"
    ]
    late_qwen_remaining = [
        segment.id
        for segment in manifest.segments
        if segment.id in attempted_segments and segment.status == "needs_regeneration"
    ]
    readiness = synth_ready_for_downstream(manifest)
    partial_status = bool(hard_failed_segments or texture_bypassed_segments or absorbed_segments)
    mark_stage(
        manifest,
        "synth",
        "completed_with_hard_failed_candidates" if partial_status else "completed",
        backend="candidate-pool",
        selected_segments=selected_segments,
        hard_failed_segments=hard_failed_segments,
        texture_bypassed_segments=texture_bypassed_segments,
        absorbed_segments=absorbed_segments,
        late_qwen_scheduled_segments=late_qwen_remaining,
        late_qwen_scheduled_count=len(late_qwen_remaining),
        late_qwen_pre_rvc_attempted_segments=attempted_segments,
        late_qwen_pre_rvc_resolved_segments=resolved_segments,
        late_qwen_pre_rvc_terminal_segments=terminal_segments,
        downstream_ready=readiness["ready"],
        downstream_blocking_segments=readiness["blocking_segments"],
        downstream_readiness=readiness,
    )
    manifest.stage_state["synth"]["downstream_readiness"] = synth_ready_for_downstream(manifest)
    select_state = manifest.stage_state.get("tts.select")
    if isinstance(select_state, dict):
        select_state["late_qwen_scheduled_segments"] = late_qwen_remaining
        select_state["late_qwen_scheduled_count"] = len(late_qwen_remaining)
        select_state["late_qwen_pre_rvc_attempted_segments"] = attempted_segments
        select_state["late_qwen_pre_rvc_resolved_segments"] = resolved_segments
        select_state["late_qwen_pre_rvc_terminal_segments"] = terminal_segments
        select_state["downstream_ready"] = manifest.stage_state["synth"]["downstream_ready"]
        select_state["downstream_blocking_segments"] = manifest.stage_state["synth"]["downstream_blocking_segments"]
        select_state["downstream_readiness"] = manifest.stage_state["synth"]["downstream_readiness"]
    return manifest.stage_state["synth"]


def run_late_qwen_pre_rvc_closure(
    ctx: PipelineContext,
    *,
    segment_ids: set[str] | None = None,
    refs_path: Path = Path("refs/refs.json"),
    confirm_rights: bool = False,
    qwen_model_id: str | None = None,
    qwen_candidate_count: int | None = None,
    qwen_local_files_only: bool | None = None,
    closure_runner: ClosureRunner = run_closure,
) -> dict[str, Any]:
    """Consume scheduled late-Qwen TTS repairs before RVC training."""

    manifest = ctx.reload_manifest()
    target_ids = set(segment_ids or late_qwen_scheduled_segment_ids(manifest))
    if not target_ids:
        return {
            "status": "skipped",
            "reason": "no_late_qwen_scheduled_segments",
            "attempted_segments": [],
            "resolved_segments": [],
            "terminal_segments": [],
        }

    _mark_attempt_started(manifest, target_ids)
    save_manifest(ctx.project_dir, manifest)
    closure_result = closure_runner(
        ctx,
        ["tts.candidate_pool", "tts.select"],
        target_ids,
        refs_path=refs_path,
        confirm_rights=confirm_rights,
        tts_backend="qwen",
        force=True,
        qwen_model_id=qwen_model_id,
        qwen_candidate_count=qwen_candidate_count,
        qwen_local_files_only=qwen_local_files_only,
    )
    manifest = ctx.reload_manifest()
    attempted_segments = sorted(target_ids)
    resolved_segments: list[str] = []
    terminal_segments: list[str] = []
    unresolved_segments: set[str] = set()
    for segment in manifest.segments:
        if segment.id not in target_ids:
            continue
        marker = _current_marker(segment)
        selected_candidate_id = segment.tts.selected_candidate_id if segment.tts else None
        selected_tts_generation_id = segment.tts.selected_tts_generation_id if segment.tts else None
        if _selected_tts(segment):
            resolved_segments.append(segment.id)
            segment.analysis["late_qwen_pre_rvc"] = {
                **marker,
                "attempted": True,
                "output_status": str(segment.status),
                "resolved": True,
                "terminal_reason": None,
                "selected_candidate_id": selected_candidate_id,
                "selected_tts_generation_id": selected_tts_generation_id,
            }
        elif _segment_terminal_for_downstream(segment):
            terminal_segments.append(segment.id)
            segment.analysis["late_qwen_pre_rvc"] = {
                **marker,
                "attempted": True,
                "output_status": str(segment.status),
                "resolved": False,
                "terminal_reason": marker.get("terminal_reason") or str(segment.status),
                "selected_candidate_id": None,
                "selected_tts_generation_id": None,
            }
        else:
            unresolved_segments.add(segment.id)

    terminal_segments.extend(
        finalize_unresolved_late_qwen_segments(
            manifest,
            unresolved_segments,
            "late_qwen_unresolved_before_train_rvc",
        )
    )
    terminal_segments = sorted(dict.fromkeys(terminal_segments))
    _refresh_synth_state(
        manifest,
        attempted_segments=attempted_segments,
        resolved_segments=sorted(resolved_segments),
        terminal_segments=terminal_segments,
    )
    save_manifest(ctx.project_dir, manifest)
    readiness = synth_ready_for_downstream(manifest)
    return {
        "status": "completed",
        "attempted_segments": attempted_segments,
        "resolved_segments": sorted(resolved_segments),
        "terminal_segments": terminal_segments,
        "unresolved_segments": sorted(unresolved_segments),
        "closure_result": closure_result,
        "downstream_readiness": readiness,
    }
