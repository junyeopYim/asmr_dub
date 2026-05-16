from __future__ import annotations

# ruff: noqa: F403,F405,I001

from asmr_dub_pipeline.pipeline.context import PipelineContext
from asmr_dub_pipeline.pipeline.stages.common import *


def run_qc_stage(ctx: PipelineContext, backend_kind: str, confirm_rights: bool = False, only_segment_ids: set[str] | None = None) -> PipelineManifest:
    project_dir = ctx.project_dir
    manifest = ctx.reload_manifest()
    _load_config_into_manifest(project_dir, manifest)
    total = len(manifest.segments)
    _log_stage_start("qc", f"backend={backend_kind}, segments={total}")
    _require_audio_stage_rights(manifest, "qc", confirm_rights, metadata={"backend": backend_kind})
    _require_rvc_ready_for_downstream(project_dir, manifest)
    cfg = manifest.project_config
    backend = create_gemma_backend(backend_kind, _gemma_backend_config(cfg))
    context = _gemma_context(manifest)
    started_at = monotonic()
    last_logged_at = started_at
    for index, segment in enumerate(manifest.segments, start=1):
        if only_segment_ids is not None and segment.id not in only_segment_ids:
            continue
        if segment.status in SKIP_STATUSES:
            last_logged_at = _log_segment_progress(
                "qc", index, total, segment, manifest, started_at, last_logged_at
            )
            continue
        if not segment.tts or not segment.tts.selected_candidate_path or not segment.script:
            segment.status = "needs_manual_review"
            segment.errors.append("Cannot QC without selected TTS and script.")
            last_logged_at = _log_segment_progress(
                "qc", index, total, segment, manifest, started_at, last_logged_at
            )
            continue
        audio_path = (
            Path(segment.rvc.output_path)
            if segment.rvc and segment.rvc.output_path
            else Path(segment.tts.selected_candidate_path)
        )
        audio_metrics = measure_audio_qc(audio_path, segment.duration)
        selected_candidate = next(
            (candidate for candidate in segment.tts.candidates if candidate.selected),
            None,
        )
        pause_padding = (
            selected_candidate.payload.get("pause_padding")
            if selected_candidate is not None
            else None
        )
        if isinstance(pause_padding, dict) and pause_padding.get("tier") == "source_pause_padding":
            padding_sec = float(pause_padding.get("padding_sec") or 0.0)
            if padding_sec > 0.0:
                audio_metrics["intentional_trailing_silence_sec"] = padding_sec
        try:
            gemma_result = validate_gemma_task_response(
                "qc",
                backend.qc_audio(audio_path, segment.script.tts_text, segment, context),
            )
        except Exception as exc:
            gemma_result = {"recommendation": "manual_review", "issues": [str(exc)]}
        qc = score_qc(audio_metrics, gemma_result)
        segment.qc = qc
        segment.status = qc.status
        last_logged_at = _log_segment_progress(
            "qc", index, total, segment, manifest, started_at, last_logged_at
        )
    out_path = project_dir / "work" / "qc" / "qc_manifest.json"
    write_json_atomic(out_path, {"segments": [s.model_dump(mode="json") for s in manifest.segments]})
    manifest.artifacts["qc"] = str(out_path)
    mark_stage(manifest, "qc", "completed", backend=backend_kind, segment_counts=_segment_counts(manifest))
    save_manifest(project_dir, manifest)
    _log_stage_complete("qc", manifest, f"backend={backend_kind}")
    return ctx.update_manifest(manifest)
