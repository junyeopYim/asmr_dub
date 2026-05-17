from __future__ import annotations

# ruff: noqa: F403,F405,I001

from asmr_dub_pipeline.pipeline.context import PipelineContext
from asmr_dub_pipeline.pipeline.invalidation import invalidate_segment
from asmr_dub_pipeline.pipeline.runner import run_closure
from asmr_dub_pipeline.pipeline.stages.common import *
from asmr_dub_pipeline.qc.repair_plan import plan_is_repairable


_AUTO_REPAIR_TERMINAL_STATUSES = {"absorbed", "no_speech_detected", "non_speech_texture", "ok"}
_AUTO_REPAIR_UNRESOLVED_STATUSES = {
    "failed",
    "needs_manual_review",
    "needs_regeneration",
    "quarantined",
    "translation_blocked",
}
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


def _failure_signature(segment: Segment) -> str:
    issues = _qc_issues(segment)
    if issues:
        return "|".join(sorted(set(issues)))
    plan = _analysis_dict(segment, "ko_qc_repair_plan")
    if plan:
        return "|".join(
            str(plan.get(key) or "")
            for key in ("action", "root_cause", "route")
            if plan.get(key)
        )
    return segment.status


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
    signature = _failure_signature(segment)
    previous_signature = _analysis_dict(segment, "auto_repair").get("last_failure_signature")
    if attempt_count > 0 and previous_signature and previous_signature == signature:
        return {
            "action": "manual_review",
            "stage": "manual",
            "terminal_manual": True,
            "reasons": ["repeated_failure_signature"],
            "attempt_count": attempt_count,
            "max_attempts": effective_max_attempts,
            "failure_signature": signature,
        }
    if attempt_count >= effective_max_attempts:
        return {
            "action": "manual_review",
            "stage": "manual",
            "terminal_manual": True,
            "reasons": ["max_attempts"],
            "attempt_count": attempt_count,
            "max_attempts": effective_max_attempts,
            "failure_signature": signature,
        }

    if segment.status in _AUTO_REPAIR_TERMINAL_STATUSES:
        return {
            "action": "manual_review",
            "stage": "manual",
            "terminal_manual": True,
            "reasons": [f"terminal_status:{segment.status}"],
            "attempt_count": attempt_count,
            "max_attempts": effective_max_attempts,
            "failure_signature": signature,
        }

    if _has_safety_or_rights_issue(segment):
        return {
            "action": "manual_review",
            "stage": "manual",
            "terminal_manual": True,
            "reasons": ["unsafe_or_rights_issue"],
            "attempt_count": attempt_count,
            "max_attempts": effective_max_attempts,
            "failure_signature": signature,
        }

    if segment.source_script is None:
        return {
            "action": "retry_asr",
            "stage": "transcribe",
            "terminal_manual": False,
            "reasons": ["missing_source_script"],
            "attempt_count": attempt_count,
            "max_attempts": effective_max_attempts,
            "failure_signature": signature,
        }

    if segment.translation_ko is None and segment.script is None:
        return {
            "action": "retry_translate_ko",
            "stage": "translate-ko",
            "terminal_manual": False,
            "reasons": ["missing_korean_translation"],
            "attempt_count": attempt_count,
            "max_attempts": effective_max_attempts,
            "failure_signature": signature,
        }

    if segment.script is None:
        return {
            "action": "rewrite_script_then_tts",
            "stage": "korean-script",
            "terminal_manual": False,
            "reasons": ["missing_script"],
            "attempt_count": attempt_count,
            "max_attempts": effective_max_attempts,
            "failure_signature": signature,
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
        "failure_signature": signature,
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
    if segment.status == "quarantined":
        quarantine = _analysis_dict(segment, "translate_ko_quarantine")
        return bool(quarantine.get("recoverable"))
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
        "round": attempt_count,
        "segment_id": segment.id,
        "input_status": segment.status,
        "action": plan["action"],
        "stage": plan["stage"],
        "terminal_manual": bool(plan.get("terminal_manual")),
        "reasons": list(plan.get("reasons") or []),
        "failure_signature_before": plan.get("failure_signature"),
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
            "last_failure_signature": plan.get("failure_signature"),
        }
    )
    if plan.get("terminal_manual"):
        payload["terminal_reason"] = ",".join(str(reason) for reason in plan.get("reasons") or [])
    else:
        payload.pop("terminal_reason", None)
    return payload


def _latest_auto_repair_attempt(segment: Segment) -> dict[str, Any]:
    payload = _analysis_dict(segment, "auto_repair")
    attempts = payload.get("attempts")
    if isinstance(attempts, list) and attempts and isinstance(attempts[-1], dict):
        return attempts[-1]
    return {}


def _closure_verification_for_segment(segment: Segment) -> tuple[bool | None, list[str], list[str]]:
    payload = _analysis_dict(segment, "auto_repair")
    closure = payload.get("closure")
    if not isinstance(closure, dict):
        return None, [], []
    executed_nodes = [str(node) for node in closure.get("executed_nodes") or []]
    verification = closure.get("verification")
    segment_verification = verification.get(segment.id) if isinstance(verification, dict) else None
    if isinstance(segment_verification, dict):
        verified = bool(segment_verification.get("ok"))
        issues = [str(issue) for issue in segment_verification.get("issues") or []]
        return verified, issues, executed_nodes
    if "verified" in closure:
        return bool(closure.get("verified")), [], executed_nodes
    return None, [], executed_nodes


def _failure_signature_after(segment: Segment) -> str:
    if segment.status not in _AUTO_REPAIR_UNRESOLVED_STATUSES:
        return segment.status
    return _failure_signature(segment)


def _apply_closure_verification_failures(manifest: PipelineManifest, plans: dict[str, dict[str, Any]]) -> None:
    for segment in manifest.segments:
        if segment.id not in plans:
            continue
        closure_verified, issues, _executed_nodes = _closure_verification_for_segment(segment)
        if closure_verified is not False:
            continue
        payload = segment.analysis.setdefault("auto_repair", {})
        if not isinstance(payload, dict):
            payload = {}
            segment.analysis["auto_repair"] = payload
        payload["closure_verified"] = False
        payload["closure_verification_issues"] = issues
        payload["manual_review_reason"] = "closure_verification_failed"
        payload["terminal_reason"] = "closure_verification_failed"
        segment.status = "needs_manual_review"
        message = "auto-repair closure verification failed"
        if message not in segment.errors:
            segment.errors.append(message)


def _auto_repair_attempt_detail(segment: Segment, plan: dict[str, Any]) -> dict[str, Any]:
    latest = _latest_auto_repair_attempt(segment)
    auto_repair = _analysis_dict(segment, "auto_repair")
    closure_verified, closure_issues, executed_nodes = _closure_verification_for_segment(segment)
    selection = _analysis_dict(segment, "tts_selection")
    quarantine = _analysis_dict(segment, "translate_ko_quarantine")
    stale_filtered = selection.get("stale_candidate_filtered_count")
    try:
        stale_filtered_count = int(stale_filtered or 0)
    except (TypeError, ValueError):
        stale_filtered_count = 0
    terminal_reason = auto_repair.get("terminal_reason")
    manual_review_reason = auto_repair.get("manual_review_reason")
    if closure_verified is False:
        manual_review_reason = manual_review_reason or "closure_verification_failed"
        terminal_reason = terminal_reason or "closure_verification_failed"
    if bool(plan.get("terminal_manual")) and terminal_reason is None:
        terminal_reason = ",".join(str(reason) for reason in plan.get("reasons") or [])
    if segment.status == "needs_manual_review" and manual_review_reason is None:
        manual_review_reason = terminal_reason or (segment.errors[-1] if segment.errors else None)
    quarantine_reason = None
    if quarantine:
        quarantine_reason = quarantine.get("reason_code") or quarantine.get("kind") or quarantine.get("reason")
    result = "repaired"
    if closure_verified is False:
        result = "verification_failed"
    elif segment.status == "quarantined":
        result = "quarantined"
    elif segment.status in _AUTO_REPAIR_UNRESOLVED_STATUSES:
        result = "manual_review" if segment.status == "needs_manual_review" else "unresolved"
    detail = {
        "round": latest.get("round") or latest.get("attempt") or plan.get("attempt_count"),
        "segment_id": segment.id,
        "input_status": latest.get("input_status"),
        "output_status": segment.status,
        "action": plan.get("action"),
        "stage": plan.get("stage"),
        "invalidated_from": _analysis_dict(segment, "_state").get("last_invalidation", {}).get("from_node"),
        "executed_nodes": executed_nodes,
        "closure_verified": closure_verified,
        "closure_verification_issues": closure_issues,
        "failure_signature_before": latest.get("failure_signature_before") or plan.get("failure_signature"),
        "failure_signature_after": _failure_signature_after(segment),
        "selected_backend": segment.tts.backend if segment.tts else None,
        "selected_candidate_id": segment.tts.selected_candidate_id if segment.tts else None,
        "selected_tts_generation_id": segment.tts.selected_tts_generation_id if segment.tts else None,
        "rvc_generation_id": segment.rvc.generation_id if segment.rvc else None,
        "qc_generation_id": segment.qc.generation_id if segment.qc else None,
        "terminal_reason": terminal_reason,
        "manual_review_reason": manual_review_reason,
        "quarantine_reason": quarantine_reason,
        "stale_candidate_filtered_count": stale_filtered_count,
        "result": result,
    }
    payload = segment.analysis.setdefault("auto_repair", {})
    if not isinstance(payload, dict):
        payload = {}
        segment.analysis["auto_repair"] = payload
    attempts = payload.get("attempts")
    if isinstance(attempts, list) and attempts and isinstance(attempts[-1], dict):
        attempts[-1].update(detail)
    payload.update(
        {
            "output_status": detail["output_status"],
            "failure_signature_after": detail["failure_signature_after"],
            "closure_verified": closure_verified,
            "closure_verification_issues": closure_issues,
            "manual_review_reason": manual_review_reason,
            "terminal_reason": terminal_reason,
            "quarantine_reason": quarantine_reason,
            "stale_candidate_filtered_count": stale_filtered_count,
        }
    )
    return detail


def _auto_repair_observability(
    manifest: PipelineManifest,
    plans: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    details = [
        _auto_repair_attempt_detail(segment, plans[segment.id])
        for segment in manifest.segments
        if segment.id in plans
    ]
    return {
        "rounds": max((int(detail.get("round") or 0) for detail in details), default=0),
        "per_segment_attempts": details,
        "segment_results": {str(detail["segment_id"]): detail for detail in details},
        "manual_review_count": sum(1 for detail in details if detail.get("output_status") == "needs_manual_review"),
        "quarantined_count": sum(1 for detail in details if detail.get("output_status") == "quarantined"),
        "verification_failed_count": sum(1 for detail in details if detail.get("closure_verified") is False),
        "stale_candidate_filtered_count": sum(int(detail.get("stale_candidate_filtered_count") or 0) for detail in details),
    }


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
            if segment.status in _AUTO_REPAIR_UNRESOLVED_STATUSES
        )
        summary["remaining_problematic_segments"] = summary["remaining_problematic_segment_ids"]
        summary.update(_auto_repair_observability(manifest, plans))
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
            remaining_problematic_segments=summary["remaining_problematic_segments"],
            rounds=summary["rounds"],
            manual_review_count=summary["manual_review_count"],
            quarantined_count=summary["quarantined_count"],
            verification_failed_count=summary["verification_failed_count"],
            stale_candidate_filtered_count=summary["stale_candidate_filtered_count"],
            per_segment_attempts=summary["per_segment_attempts"],
            segment_counts=_segment_counts(manifest),
        )
        save_manifest(project_dir, manifest)
        _log_stage_complete("auto-repair", manifest, f"status={status}")
        return ctx.update_manifest(manifest)

    save_manifest(project_dir, manifest)

    from asmr_dub_pipeline.pipeline.stages.korean_script import run_korean_script_stage
    from asmr_dub_pipeline.pipeline.stages.qc import run_qc_stage
    from asmr_dub_pipeline.pipeline.stages.rvc import run_rvc_stage
    from asmr_dub_pipeline.pipeline.stages.translate_ko import run_translate_ko_stage

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
        manifest = ctx.reload_manifest()
        for segment_id in qwen_ids:
            invalidate_segment(manifest, segment_id, "tts.candidate_pool", "auto_repair_fallback_tts_qwen")
        save_manifest(project_dir, manifest)
        closure = run_closure(
            ctx,
            ["tts.candidate_pool", "tts.select", "rvc", "qc"],
            qwen_ids,
            refs_path=refs_path,
            confirm_rights=confirm_rights,
            gemma_backend=gemma_backend,
            tts_backend="qwen",
            gsv_url=gsv_url,
            gpt_weights_path=gpt_weights_path,
            sovits_weights_path=sovits_weights_path,
            use_trained_gpt=use_trained_gpt,
            auto_gsv_server=auto_gsv_server,
            gsv_server_command=gsv_server_command,
        )
        manifest = ctx.reload_manifest()
        for segment in manifest.segments:
            if segment.id in qwen_ids:
                payload = segment.analysis.setdefault("auto_repair", {})
                if isinstance(payload, dict):
                    payload["closure"] = closure
        save_manifest(project_dir, manifest)
        ctx.update_manifest(manifest)

    rewrite_ids = action_groups.get("rewrite_script_then_tts", set())
    if rewrite_ids:
        run_korean_script_stage(ctx, confirm_rights=confirm_rights, only_segment_ids=rewrite_ids)

    regenerate_ids = set(action_groups.get("regenerate_tts", set())) | set(rewrite_ids)
    if regenerate_ids:
        manifest = ctx.reload_manifest()
        for segment_id in regenerate_ids:
            invalidate_segment(manifest, segment_id, "tts.candidate_pool", "auto_repair_regenerate_tts")
        save_manifest(project_dir, manifest)
        closure = run_closure(
            ctx,
            refs_path=refs_path,
            confirm_rights=confirm_rights,
            gemma_backend=gemma_backend,
            tts_backend=tts_backend,
            target_nodes=["tts.candidate_pool", "tts.select", "rvc", "qc"],
            segment_ids=regenerate_ids,
            gsv_url=gsv_url,
            gpt_weights_path=gpt_weights_path,
            sovits_weights_path=sovits_weights_path,
            use_trained_gpt=use_trained_gpt,
            auto_gsv_server=auto_gsv_server,
            gsv_server_command=gsv_server_command,
        )
        manifest = ctx.reload_manifest()
        for segment in manifest.segments:
            if segment.id in regenerate_ids:
                payload = segment.analysis.setdefault("auto_repair", {})
                if isinstance(payload, dict):
                    payload["closure"] = closure
        save_manifest(project_dir, manifest)
        ctx.update_manifest(manifest)

    rvc_ids = action_groups.get("retry_rvc", set())
    if rvc_ids:
        manifest = ctx.reload_manifest()
        for segment_id in rvc_ids:
            invalidate_segment(manifest, segment_id, "rvc", "auto_repair_retry_rvc")
        save_manifest(project_dir, manifest)
        run_rvc_stage(ctx, confirm_rights=confirm_rights, retry_failed=True, only_segment_ids=rvc_ids)
        run_qc_stage(ctx, gemma_backend, confirm_rights=confirm_rights, only_segment_ids=rvc_ids)

    translation_ids = set(action_groups.get("retry_translate_ko", set())) | set(
        action_groups.get("repair_translation_then_tts", set())
    )
    if translation_ids:
        manifest = ctx.reload_manifest()
        for segment_id in translation_ids:
            invalidate_segment(manifest, segment_id, "translate_ko", "auto_repair_translation")
        save_manifest(project_dir, manifest)
        run_translate_ko_stage(
            ctx,
            gemma_backend,
            confirm_rights=confirm_rights,
            retry_failed=True,
            force_retranslate_failed=True,
            only_segment_ids=translation_ids,
        )
        run_korean_script_stage(ctx, confirm_rights=confirm_rights, only_segment_ids=translation_ids)
        closure = run_closure(
            ctx,
            ["tts.candidate_pool", "tts.select", "rvc", "qc"],
            translation_ids,
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
        )
        manifest = ctx.reload_manifest()
        for segment in manifest.segments:
            if segment.id in translation_ids:
                payload = segment.analysis.setdefault("auto_repair", {})
                if isinstance(payload, dict):
                    payload["closure"] = closure
        save_manifest(project_dir, manifest)
        ctx.update_manifest(manifest)

    unsupported_ids = set().union(
        action_groups.get("retry_asr", set()),
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
    _apply_closure_verification_failures(manifest, plans)
    observability = _auto_repair_observability(manifest, plans)
    remaining_problematic_ids = sorted(
        segment.id
        for segment in manifest.segments
        if segment.status in _AUTO_REPAIR_UNRESOLVED_STATUSES
    )
    repaired_ids = sorted(
        str(detail["segment_id"])
        for detail in observability["per_segment_attempts"]
        if detail.get("result") == "repaired"
    )
    summary["completed_segment_counts"] = _segment_counts(manifest)
    summary["repaired_segments"] = repaired_ids
    summary["repaired_count"] = len(repaired_ids)
    summary["remaining_problematic_segment_ids"] = remaining_problematic_ids
    summary["remaining_problematic_segments"] = remaining_problematic_ids
    summary["terminal_manual_count"] = len(
        [
            segment
            for segment in manifest.segments
            if segment.id in plans and segment.status == "needs_manual_review"
        ]
    )
    summary["manual_review_count"] = observability["manual_review_count"]
    summary["quarantined_count"] = observability["quarantined_count"]
    summary["verification_failed_count"] = observability["verification_failed_count"]
    summary["stale_candidate_filtered_count"] = observability["stale_candidate_filtered_count"]
    summary["rounds"] = observability["rounds"]
    summary["per_segment_attempts"] = observability["per_segment_attempts"]
    summary["segment_results"] = observability["segment_results"]
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
        remaining_problematic_segments=remaining_problematic_ids,
        rounds=summary["rounds"],
        manual_review_count=summary["manual_review_count"],
        quarantined_count=summary["quarantined_count"],
        verification_failed_count=summary["verification_failed_count"],
        stale_candidate_filtered_count=summary["stale_candidate_filtered_count"],
        per_segment_attempts=summary["per_segment_attempts"],
        segment_counts=_segment_counts(manifest),
    )
    save_manifest(project_dir, manifest)
    _log_stage_complete("auto-repair", manifest, f"processed={len(plans)}")
    return ctx.update_manifest(manifest)
