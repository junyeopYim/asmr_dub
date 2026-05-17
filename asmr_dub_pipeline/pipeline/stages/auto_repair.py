from __future__ import annotations

# ruff: noqa: F403,F405,I001

from asmr_dub_pipeline.pipeline.context import PipelineContext
from asmr_dub_pipeline.pipeline.stages.common import *
from asmr_dub_pipeline.qc.repair_plan import plan_is_repairable


_AUTO_REPAIR_TERMINAL_STATUSES = {"absorbed", "no_speech_detected", "non_speech_texture", "ok"}
_SAFETY_ISSUE_MARKERS = ("unsafe", "rights", "consent", "copyright", "policy", "unauthorized")


def _analysis_dict(segment: Segment, key: str) -> dict[str, Any]:
    value = segment.analysis.get(key)
    return value if isinstance(value, dict) else {}


def _auto_repair_attempt_count(segment: Segment) -> int:
    payload = _analysis_dict(segment, "auto_repair")
    try:
        return int(payload.get("attempt_count") or len(payload.get("attempts") or []))
    except (TypeError, ValueError):
        return 0


def _qc_issues(segment: Segment) -> list[str]:
    issues: list[str] = []
    if segment.qc is not None:
        issues.extend(str(issue) for issue in segment.qc.issues)
    issues.extend(str(error) for error in segment.errors)
    return issues


def _has_safety_or_rights_issue(segment: Segment) -> bool:
    if segment.qc is not None and segment.qc.unsafe_or_rights_issue:
        return True
    lowered = [issue.lower() for issue in _qc_issues(segment)]
    return any(any(marker in issue for marker in _SAFETY_ISSUE_MARKERS) for issue in lowered)


def _plan_action(segment: Segment) -> str | None:
    plan = _analysis_dict(segment, "ko_qc_repair_plan")
    if not plan or bool(plan.get("terminal_manual")):
        return None
    action = str(plan.get("action") or "").strip()
    return action or None


def classify_auto_repair_segment(
    segment: Segment,
    cfg: ProjectConfig | None = None,
    *,
    max_attempts: int | None = None,
) -> dict[str, Any]:
    """Return the next safe repair route for one segment without mutating it."""

    effective_max_attempts = int(
        max_attempts
        if max_attempts is not None
        else getattr(cfg, "auto_repair_max_attempts", 3)
        if cfg is not None
        else 3
    )
    attempt_count = _auto_repair_attempt_count(segment)
    if attempt_count >= effective_max_attempts:
        return {
            "action": "manual_review",
            "stage": "manual",
            "terminal_manual": True,
            "reasons": ["max_attempts"],
            "attempt_count": attempt_count,
            "max_attempts": effective_max_attempts,
        }

    if segment.status in _AUTO_REPAIR_TERMINAL_STATUSES:
        return {
            "action": "manual_review",
            "stage": "manual",
            "terminal_manual": True,
            "reasons": [f"terminal_status:{segment.status}"],
            "attempt_count": attempt_count,
            "max_attempts": effective_max_attempts,
        }

    if _has_safety_or_rights_issue(segment):
        return {
            "action": "manual_review",
            "stage": "manual",
            "terminal_manual": True,
            "reasons": ["unsafe_or_rights_issue"],
            "attempt_count": attempt_count,
            "max_attempts": effective_max_attempts,
        }

    if segment.source_script is None:
        return {
            "action": "retry_asr",
            "stage": "transcribe",
            "terminal_manual": False,
            "reasons": ["missing_source_script"],
            "attempt_count": attempt_count,
            "max_attempts": effective_max_attempts,
        }

    if segment.translation_ko is None and segment.script is None:
        return {
            "action": "retry_translate_ko",
            "stage": "translate-ko",
            "terminal_manual": False,
            "reasons": ["missing_korean_translation"],
            "attempt_count": attempt_count,
            "max_attempts": effective_max_attempts,
        }

    if segment.script is None:
        return {
            "action": "rewrite_script_then_tts",
            "stage": "korean-script",
            "terminal_manual": False,
            "reasons": ["missing_script"],
            "attempt_count": attempt_count,
            "max_attempts": effective_max_attempts,
        }

    action = _plan_action(segment)
    if action == "fallback_tts_qwen":
        stage = "synth-qwen"
    elif action == "keep_original_texture":
        stage = "in-place"
    elif action == "repair_translation_then_tts":
        stage = "translate-ko"
    elif action == "repair_asr_then_downstream":
        stage = "transcribe"
    elif action == "rewrite_script_then_tts":
        stage = "korean-script"
    elif action:
        stage = "synth"
    elif not segment.tts or not segment.tts.selected_candidate_path:
        action = "regenerate_tts"
        stage = "synth"
    elif segment.status == "failed" and segment.tts and segment.tts.selected_candidate_path:
        action = "retry_rvc"
        stage = "rvc"
    else:
        action = "manual_review"
        stage = "manual"

    return {
        "action": action,
        "stage": stage,
        "terminal_manual": action == "manual_review",
        "reasons": _qc_issues(segment) or [action],
        "attempt_count": attempt_count,
        "max_attempts": effective_max_attempts,
    }


def _eligible_for_auto_repair(segment: Segment, only_segment_ids: set[str] | None) -> bool:
    if only_segment_ids is not None and segment.id not in only_segment_ids:
        return False
    if segment.status == "needs_regeneration":
        return True
    if segment.status == "needs_manual_review" and plan_is_repairable(segment.analysis.get("ko_qc_repair_plan")):
        return True
    if segment.status == "failed":
        return bool(
            plan_is_repairable(segment.analysis.get("ko_qc_repair_plan"))
            or (segment.tts and segment.tts.selected_candidate_path)
            or segment.source_script is None
        )
    return False


def _record_auto_repair_attempt(segment: Segment, plan: dict[str, Any]) -> dict[str, Any]:
    payload = segment.analysis.setdefault("auto_repair", {})
    if not isinstance(payload, dict):
        payload = {}
        segment.analysis["auto_repair"] = payload
    attempts = payload.setdefault("attempts", [])
    if not isinstance(attempts, list):
        attempts = []
        payload["attempts"] = attempts
    attempt_count = _auto_repair_attempt_count(segment) + 1
    record = {
        "attempt": attempt_count,
        "action": plan["action"],
        "stage": plan["stage"],
        "terminal_manual": bool(plan.get("terminal_manual")),
        "reasons": list(plan.get("reasons") or []),
    }
    attempts.append(record)
    payload.update(
        {
            "action": plan["action"],
            "stage": plan["stage"],
            "attempt_count": attempt_count,
            "max_attempts": plan.get("max_attempts"),
            "terminal_manual": bool(plan.get("terminal_manual")),
            "last_action": plan["action"],
            "last_stage": plan["stage"],
        }
    )
    if plan.get("terminal_manual"):
        payload["terminal_reason"] = ",".join(str(reason) for reason in plan.get("reasons") or [])
    else:
        payload.pop("terminal_reason", None)
    return payload


def _write_auto_repair_summary(project_dir: Path, manifest: PipelineManifest, summary: dict[str, Any]) -> Path:
    out_path = project_dir / "work" / "auto_repair" / "summary.json"
    write_json_atomic(out_path, summary)
    manifest.artifacts["auto_repair"] = str(out_path)
    return out_path


def run_auto_repair_stage(
    ctx: PipelineContext,
    *,
    refs_path: Path = Path("refs/refs.json"),
    confirm_rights: bool = False,
    max_attempts: int | None = None,
    plan_only: bool = False,
    only_segment_ids: set[str] | None = None,
    gemma_backend: str = "mock",
    tts_backend: str = "gpt-sovits",
    gsv_url: str | None = None,
    gpt_weights_path: str | None = None,
    sovits_weights_path: str | None = None,
    use_trained_gpt: bool = False,
    auto_gsv_server: bool | None = None,
    gsv_server_command: list[str] | str | None = None,
) -> PipelineManifest:
    project_dir = ctx.project_dir
    manifest = ctx.reload_manifest()
    _load_config_into_manifest(project_dir, manifest)
    cfg = manifest.project_config
    effective_max_attempts = max_attempts or cfg.auto_repair_max_attempts
    _log_stage_start("auto-repair", f"plan_only={plan_only}, max_attempts={effective_max_attempts}")
    _require_audio_stage_rights(
        manifest,
        "auto-repair",
        confirm_rights,
        metadata={"plan_only": plan_only, "max_attempts": effective_max_attempts},
    )

    plans: dict[str, dict[str, Any]] = {}
    terminal_segments: list[str] = []
    for segment in manifest.segments:
        if not _eligible_for_auto_repair(segment, only_segment_ids):
            continue
        plan = classify_auto_repair_segment(segment, cfg, max_attempts=effective_max_attempts)
        plans[segment.id] = plan
        _record_auto_repair_attempt(segment, plan)
        if plan.get("terminal_manual"):
            segment.status = "needs_manual_review"
            terminal_segments.append(segment.id)

    action_groups: dict[str, set[str]] = {}
    for segment_id, plan in plans.items():
        if plan.get("terminal_manual"):
            continue
        action_groups.setdefault(str(plan["action"]), set()).add(segment_id)

    summary: dict[str, Any] = {
        "plan_only": plan_only,
        "max_attempts": effective_max_attempts,
        "planned_segments": sorted(plans),
        "terminal_segments": sorted(terminal_segments),
        "actions": {action: sorted(segment_ids) for action, segment_ids in sorted(action_groups.items())},
        "per_action_counts": {action: len(segment_ids) for action, segment_ids in sorted(action_groups.items())},
        "repairable_count": sum(len(segment_ids) for segment_ids in action_groups.values()),
        "terminal_manual_count": len(terminal_segments),
        "repaired_count": 0,
        "remaining_problematic_segment_ids": [],
        "plans": plans,
    }

    if plan_only or not action_groups:
        status = "planned" if plan_only and plans else "skipped" if not plans else "completed"
        summary["remaining_problematic_segment_ids"] = sorted(
            segment.id
            for segment in manifest.segments
            if segment.status in {"failed", "needs_manual_review", "needs_regeneration"}
        )
        _write_auto_repair_summary(project_dir, manifest, summary)
        mark_stage(
            manifest,
            "auto-repair",
            status,
            plan_only=plan_only,
            planned_segments=sorted(plans),
            actions=summary["actions"],
            per_action_counts=summary["per_action_counts"],
            repairable_count=summary["repairable_count"],
            repaired_count=summary["repaired_count"],
            terminal_segments=sorted(terminal_segments),
            terminal_manual_count=summary["terminal_manual_count"],
            remaining_problematic_segment_ids=summary["remaining_problematic_segment_ids"],
            segment_counts=_segment_counts(manifest),
        )
        save_manifest(project_dir, manifest)
        _log_stage_complete("auto-repair", manifest, f"status={status}")
        return ctx.update_manifest(manifest)

    save_manifest(project_dir, manifest)

    from asmr_dub_pipeline.pipeline.stages.korean_script import run_korean_script_stage
    from asmr_dub_pipeline.pipeline.stages.qc import run_qc_stage
    from asmr_dub_pipeline.pipeline.stages.regenerate import run_regenerate_needs_stage
    from asmr_dub_pipeline.pipeline.stages.rvc import run_rvc_stage
    from asmr_dub_pipeline.pipeline.stages.synth_qwen import run_synth_qwen_stage

    keep_original_ids = action_groups.get("keep_original_texture", set())
    if keep_original_ids:
        manifest = ctx.reload_manifest()
        for segment in manifest.segments:
            if segment.id not in keep_original_ids:
                continue
            segment.keep_original_texture = True
            segment.status = "absorbed"
            segment.analysis.setdefault("auto_repair_keep_original_texture", {})["applied"] = True
        save_manifest(project_dir, manifest)
        ctx.update_manifest(manifest)

    qwen_ids = action_groups.get("fallback_tts_qwen", set())
    if qwen_ids:
        run_synth_qwen_stage(ctx, refs_path, confirm_rights=confirm_rights, promote=True, only_segment_ids=qwen_ids)
        run_rvc_stage(ctx, confirm_rights=confirm_rights, only_segment_ids=qwen_ids)
        run_qc_stage(ctx, gemma_backend, confirm_rights=confirm_rights, only_segment_ids=qwen_ids)

    rewrite_ids = action_groups.get("rewrite_script_then_tts", set())
    if rewrite_ids:
        run_korean_script_stage(ctx, confirm_rights=confirm_rights, only_segment_ids=rewrite_ids)

    regenerate_ids = set(action_groups.get("regenerate_tts", set())) | set(rewrite_ids)
    if regenerate_ids:
        run_regenerate_needs_stage(
            ctx,
            refs_path=refs_path,
            confirm_rights=confirm_rights,
            gemma_backend=gemma_backend,
            tts_backend=tts_backend,
            gsv_url=gsv_url,
            gpt_weights_path=gpt_weights_path,
            sovits_weights_path=sovits_weights_path,
            use_trained_gpt=use_trained_gpt,
            auto_gsv_server=auto_gsv_server,
            gsv_server_command=gsv_server_command,
            only_segment_ids=regenerate_ids,
        )

    rvc_ids = action_groups.get("retry_rvc", set())
    if rvc_ids:
        run_rvc_stage(ctx, confirm_rights=confirm_rights, retry_failed=True, only_segment_ids=rvc_ids)
        run_qc_stage(ctx, gemma_backend, confirm_rights=confirm_rights, only_segment_ids=rvc_ids)

    unsupported_ids = set().union(
        action_groups.get("retry_asr", set()),
        action_groups.get("retry_translate_ko", set()),
        action_groups.get("repair_translation_then_tts", set()),
        action_groups.get("repair_asr_then_downstream", set()),
    )
    if unsupported_ids:
        manifest = ctx.reload_manifest()
        for segment in manifest.segments:
            if segment.id not in unsupported_ids:
                continue
            payload = segment.analysis.setdefault("auto_repair", {})
            if isinstance(payload, dict):
                payload["terminal_manual"] = True
                payload["terminal_reason"] = "segment_specific_route_unavailable"
            segment.status = "needs_manual_review"
        save_manifest(project_dir, manifest)
        ctx.update_manifest(manifest)

    manifest = ctx.reload_manifest()
    remaining_problematic_ids = sorted(
        segment.id
        for segment in manifest.segments
        if segment.status in {"failed", "needs_manual_review", "needs_regeneration"}
    )
    repaired_ids = sorted(
        segment.id
        for segment in manifest.segments
        if segment.id in plans and segment.status not in {"failed", "needs_manual_review", "needs_regeneration"}
    )
    summary["completed_segment_counts"] = _segment_counts(manifest)
    summary["repaired_segments"] = repaired_ids
    summary["repaired_count"] = len(repaired_ids)
    summary["remaining_problematic_segment_ids"] = remaining_problematic_ids
    summary["terminal_manual_count"] = len(
        [
            segment
            for segment in manifest.segments
            if segment.id in plans and segment.status == "needs_manual_review"
        ]
    )
    _write_auto_repair_summary(project_dir, manifest, summary)
    mark_stage(
        manifest,
        "auto-repair",
        "completed",
        plan_only=False,
        planned_segments=sorted(plans),
        actions=summary["actions"],
        per_action_counts=summary["per_action_counts"],
        repairable_count=summary["repairable_count"],
        repaired_count=summary["repaired_count"],
        terminal_segments=sorted(terminal_segments),
        terminal_manual_count=summary["terminal_manual_count"],
        remaining_problematic_segment_ids=remaining_problematic_ids,
        segment_counts=_segment_counts(manifest),
    )
    save_manifest(project_dir, manifest)
    _log_stage_complete("auto-repair", manifest, f"processed={len(plans)}")
    return ctx.update_manifest(manifest)
