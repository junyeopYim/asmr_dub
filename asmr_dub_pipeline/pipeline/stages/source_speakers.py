from __future__ import annotations

# ruff: noqa: F403,F405,I001

from asmr_dub_pipeline.pipeline.context import PipelineContext
from asmr_dub_pipeline.pipeline.stages.common import *


def run_source_speakers_stage(
    ctx: PipelineContext,
    backend_kind: str | None = None,
    confirm_rights: bool = False,
    jobs: int = 4,
) -> PipelineManifest:
    project_dir = ctx.project_dir
    manifest = ctx.reload_manifest()
    _load_config_into_manifest(project_dir, manifest)
    cfg = manifest.project_config
    backend = backend_kind or cfg.speaker_assignment_backend
    if backend == "none":
        backend = "mock"
    _log_stage_start("source-speakers", f"backend={backend}, jobs={jobs}, segments={len(manifest.segments)}")
    _require_audio_stage_rights(
        manifest,
        "source-speakers",
        confirm_rights,
        metadata={"backend": backend},
    )
    assign_source_speakers_to_manifest(project_dir, manifest, backend_kind=backend, jobs=jobs)
    out_path = project_dir / "work" / "segments" / "manifests" / "segments_source_speakers.json"
    write_json_atomic(out_path, {"segments": [s.model_dump(mode="json") for s in manifest.segments]})
    manifest.artifacts["segments_source_speakers"] = str(out_path)
    save_manifest(project_dir, manifest)
    _log_stage_complete("source-speakers", manifest, f"backend={backend}")
    return ctx.update_manifest(manifest)
