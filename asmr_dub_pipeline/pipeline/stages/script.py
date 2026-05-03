from __future__ import annotations

# ruff: noqa: F403,F405,I001

from asmr_dub_pipeline.pipeline.context import PipelineContext
from asmr_dub_pipeline.pipeline.stages.common import *


def run_script_stage(ctx: PipelineContext, backend_kind: str, confirm_rights: bool = False) -> PipelineManifest:
    project_dir = ctx.project_dir
    manifest = ctx.reload_manifest()
    _load_config_into_manifest(project_dir, manifest)
    total = len(manifest.segments)
    _log_stage_start("script", f"backend={backend_kind}, segments={total}")
    _require_audio_stage_rights(manifest, "script", confirm_rights, metadata={"backend": backend_kind})
    cfg = manifest.project_config
    backend = create_gemma_backend(backend_kind, _gemma_backend_config(cfg))
    context = _gemma_context(manifest)
    started_at = monotonic()
    last_logged_at = started_at
    for index, segment in enumerate(manifest.segments, start=1):
        if segment.status in SKIP_STATUSES:
            last_logged_at = _log_segment_progress(
                "script", index, total, segment, manifest, started_at, last_logged_at
            )
            continue
        _validate_segment_audio_paths(project_dir, segment, check_formats=True)
        try:
            payload = validate_gemma_task_response(
                "script",
                backend.generate_script(Path(segment.audio_for_gemma), segment, context),
            )
            segment.script = normalize_script_payload(payload)
            segment.status = "scripted"
        except Exception as exc:
            segment.errors.append(str(exc))
            segment.status = "needs_manual_review"
        last_logged_at = _log_segment_progress(
            "script", index, total, segment, manifest, started_at, last_logged_at
        )
    out_path = project_dir / "work" / "segments" / "manifests" / "segments_script.json"
    write_json_atomic(out_path, {"segments": [s.model_dump(mode="json") for s in manifest.segments]})
    manifest.artifacts["segments_script"] = str(out_path)
    mark_stage(manifest, "script", "completed", backend=backend_kind, segment_counts=_segment_counts(manifest))
    save_manifest(project_dir, manifest)
    _log_stage_complete("script", manifest, f"backend={backend_kind}")
    return ctx.update_manifest(manifest)
