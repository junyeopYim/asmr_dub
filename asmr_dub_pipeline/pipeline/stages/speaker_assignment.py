from __future__ import annotations

# ruff: noqa: F403,F405,I001

from asmr_dub_pipeline.pipeline.context import PipelineContext
from asmr_dub_pipeline.pipeline.stages.common import *


def run_assign_speakers_stage(ctx: PipelineContext, voice_bank_path: Path | None = None, backend_kind: str | None = None, require_all: bool = True) -> PipelineManifest:
    project_dir = ctx.project_dir
    manifest = ctx.reload_manifest()
    _load_config_into_manifest(project_dir, manifest)
    cfg = manifest.project_config
    bank = load_voice_bank(project_dir, cfg, voice_bank_path)
    validate_voice_bank_models(project_dir, bank)
    next_cfg = apply_voice_bank_to_config(project_dir, cfg, bank)
    save_project_config(next_cfg, project_dir / "pipeline.yaml")
    manifest.project_config = next_cfg
    backend = backend_kind or cfg.speaker_assignment_backend
    if backend == "none":
        backend = "mock"
    assign_speakers_to_manifest(
        project_dir,
        manifest,
        bank,
        backend_kind=backend,
        require_all=require_all,
    )
    manifest.artifacts["voice_bank"] = str(resolve_voice_bank_path(project_dir, next_cfg, voice_bank_path))
    save_manifest(project_dir, manifest)
    _log_stage_complete("speaker-assign", manifest, f"backend={backend}")
    return ctx.update_manifest(manifest)
