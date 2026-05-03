from __future__ import annotations

# ruff: noqa: F403,F405,I001

from asmr_dub_pipeline.pipeline.context import PipelineContext
from asmr_dub_pipeline.pipeline.stages.common import *


def run_mix_stage(ctx: PipelineContext, confirm_rights: bool) -> PipelineManifest:
    project_dir = ctx.project_dir
    manifest = ctx.reload_manifest()
    _load_config_into_manifest(project_dir, manifest)
    cfg = manifest.project_config
    total = len(manifest.segments)
    _log_stage_start("mix", f"segments={total}")
    source_path = Path(manifest.source_info.path) if manifest.source_info else None
    audit = require_confirmed_rights(confirm_rights, "mix", source_path)
    manifest.rights_audit = merge_rights_audit(manifest.rights_audit, audit)
    if manifest.stage_state.get("qc", {}).get("status") != "completed":
        raise ValueError("Mix requires a completed QC stage.")
    _require_rvc_ready_for_downstream(project_dir, manifest)
    allow_korean_timing_draft = (
        cfg.mix_allow_korean_timing_draft
        and _canonical_language(cfg.target_language) == "ko"
        and manifest.stage_state.get("korean-script", {}).get("status") == "completed"
    )
    duration = manifest.source_info.duration_sec if manifest.source_info else 0.0
    if duration <= 0 and manifest.segments:
        duration = max(segment.end for segment in manifest.segments)
    dialogue = ensure_inside_project(project_dir, project_dir / "work" / "mix" / "dialogue_stem.wav")
    final_audio = ensure_inside_project(project_dir, project_dir / "work" / "mix" / "final_audio.wav")
    peak_limit_dbfs = cfg.mix_peak_limit_dbfs if cfg.mix_loudness_strategy == "peak_guard_only" else None
    started_at = monotonic()
    last_logged_at = started_at

    def log_mix_progress(index: int, progress_total: int, segment: Segment) -> None:
        nonlocal last_logged_at
        last_logged_at = _log_segment_progress(
            "mix dialogue",
            index,
            progress_total,
            segment,
            manifest,
            started_at,
            last_logged_at,
        )

    console.print("[cyan]mix[/cyan] building dialogue stem")
    build_dialogue_stem(
        manifest.segments,
        dialogue,
        duration,
        cfg.mix_sample_rate,
        dialogue_gain_db=cfg.mix_dialogue_gain_db,
        dialogue_fade_ms=cfg.mix_dialogue_fade_ms,
        peak_limit_dbfs=peak_limit_dbfs,
        progress_callback=log_mix_progress,
        include_segment=lambda segment: _include_segment_in_mix(
            segment,
            allow_korean_timing_draft=allow_korean_timing_draft,
        ),
    )
    console.print(f"[cyan]mix[/cyan] dialogue stem written: {dialogue}")
    separated_background = manifest.artifacts.get("background_only_48k")
    original_background = manifest.artifacts.get("original_stereo_48k")
    background_source = None
    background_kind = None
    if cfg.mix_background_bed == "preserve_original":
        if separated_background:
            background_source = Path(separated_background)
            background_kind = "source_separated"
        elif original_background:
            background_source = Path(original_background)
            background_kind = "original"
    background_available = background_source is not None
    background = background_source
    background_suppressed_path: Path | None = None
    if background_source and cfg.background_speech_suppression:
        background_suppressed_path = ensure_inside_project(
            project_dir,
            project_dir / "work" / "mix" / "source_suppressed_background.wav",
        )
        console.print("[cyan]mix[/cyan] suppressing source speech in background bed")
        build_source_suppressed_background(
            background_source,
            background_suppressed_path,
            manifest.segments,
            sample_rate=cfg.mix_sample_rate,
            attenuation_db=cfg.background_speech_suppression_db,
            pad_sec=cfg.background_speech_suppression_pad_sec,
            fade_ms=cfg.background_speech_suppression_fade_ms,
            reduce_center_bleed=background_kind != "source_separated",
            peak_limit_dbfs=peak_limit_dbfs,
        )
        manifest.artifacts["source_suppressed_background"] = str(background_suppressed_path)
        background = background_suppressed_path
    console.print("[cyan]mix[/cyan] combining dialogue with background")
    mix_with_background(
        dialogue,
        final_audio,
        background,
        cfg.background_gain_db,
        cfg.mix_sample_rate,
        peak_limit_dbfs=peak_limit_dbfs,
        suppress_background_speech=False,
    )
    console.print(f"[cyan]mix[/cyan] final audio written: {final_audio}")
    manifest.artifacts["dialogue_stem"] = str(dialogue)
    manifest.artifacts["final_audio"] = str(final_audio)
    mix_config = _mix_config_metadata(manifest)
    background_metadata = {
        "available": background_available,
        "used": background is not None,
        "path": str(background) if background else None,
        "source_path": str(background_source) if background_source else None,
        "source_kind": background_kind,
        "policy": cfg.mix_background_bed,
        "gain_db": cfg.background_gain_db if background else None,
        "speech_suppression": {
            "enabled": bool(background_suppressed_path),
            "path": str(background_suppressed_path) if background_suppressed_path else None,
            "attenuation_db": cfg.background_speech_suppression_db
            if background_suppressed_path
            else None,
            "pad_sec": cfg.background_speech_suppression_pad_sec
            if background_suppressed_path
            else None,
            "fade_ms": cfg.background_speech_suppression_fade_ms
            if background_suppressed_path
            else None,
            "center_bleed_reduction": bool(
                background_suppressed_path and background_kind != "source_separated"
            ),
        },
    }
    skipped = [
        s.id
        for s in manifest.segments
        if not _include_segment_in_mix(s, allow_korean_timing_draft=allow_korean_timing_draft)
    ]
    draft_included = [
        s.id
        for s in manifest.segments
        if s.status != "ok"
        and _include_segment_in_mix(s, allow_korean_timing_draft=allow_korean_timing_draft)
    ]
    for segment in manifest.segments:
        included = _include_segment_in_mix(
            segment,
            allow_korean_timing_draft=allow_korean_timing_draft,
        )
        segment.mix = {
            **segment.mix,
            "included": included,
            "reason": "qc_pass"
            if included and segment.status == "ok"
            else "korean_timing_draft"
            if included
            else f"status_{segment.status}",
            "selected_candidate_path": segment.tts.selected_candidate_path if segment.tts else None,
            "rvc_output_path": segment.rvc.output_path if segment.rvc else None,
            "start": segment.start,
            "estimated_pan": segment.estimated_pan,
            "dialogue_gain_db": cfg.mix_dialogue_gain_db if included else None,
            "dialogue_fade_ms": cfg.mix_dialogue_fade_ms if included else None,
            "qc_recommendation": segment.qc.recommendation if segment.qc else None,
        }
    if draft_included:
        manifest.warnings.append(
            "Included Korean draft segments with timing-only QC regeneration flags during mix: "
            + ", ".join(draft_included)
        )
    if skipped:
        manifest.warnings.append(f"Skipped non-passing segments during mix: {', '.join(skipped)}")
    mix_manifest = project_dir / "work" / "mix" / "mix_manifest.json"
    write_json_atomic(
        mix_manifest,
        {
            "dialogue_stem": str(dialogue),
            "final_audio": str(final_audio),
            "config": mix_config,
            "background": background_metadata,
            "segments": [{"id": segment.id, "mix": segment.mix} for segment in manifest.segments],
        },
    )
    manifest.artifacts["mix_manifest"] = str(mix_manifest)
    mark_stage(
        manifest,
        "mix",
        "completed",
        skipped_segments=skipped,
        draft_included_segments=draft_included,
        mix_config=mix_config,
        background=background_metadata,
    )
    save_manifest(project_dir, manifest)
    _log_stage_complete("mix", manifest, f"skipped={len(skipped)}")
    return ctx.update_manifest(manifest)
