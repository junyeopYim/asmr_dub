from __future__ import annotations

# ruff: noqa: F403,F405,I001

from asmr_dub_pipeline.pipeline.context import PipelineContext
from asmr_dub_pipeline.pipeline.stages.common import *


def run_source_separation_stage(ctx: PipelineContext, confirm_rights: bool = False, force: bool = False) -> PipelineManifest:
    project_dir = ctx.project_dir
    manifest = ctx.reload_manifest()
    _load_config_into_manifest(project_dir, manifest)
    cfg = manifest.project_config
    backend = cfg.source_separation_backend
    _log_stage_start("source-separation", f"backend={backend}, model={cfg.source_separation_model}")
    _require_audio_stage_rights(
        manifest,
        "source-separation",
        confirm_rights,
        metadata={"backend": backend, "model": cfg.source_separation_model},
    )
    original_audio = Path(
        manifest.artifacts.get("original_stereo_48k", project_dir / "work/audio/original_stereo_48k.wav")
    )
    if backend == "none":
        mark_stage(manifest, "source-separation", "skipped", backend=backend, reason="disabled")
        save_manifest(project_dir, manifest)
        _log_stage_complete("source-separation", manifest, "skipped=disabled")
        return ctx.update_manifest(manifest)
    try:
        result = separate_source_audio(
            original_audio,
            project_dir,
            backend=backend,
            model=cfg.source_separation_model,
            device=cfg.source_separation_device,
            sample_rate=cfg.mix_sample_rate,
            mono_sample_rate=cfg.gemma_sample_rate,
            force=force,
        )
    except SourceSeparationUnavailable as exc:
        if backend != "auto":
            raise
        warning = f"Source separation skipped because no separator backend is available: {exc}"
        if warning not in manifest.warnings:
            manifest.warnings.append(warning)
        mark_stage(manifest, "source-separation", "skipped", backend=backend, reason=str(exc))
        save_manifest(project_dir, manifest)
        _log_stage_complete("source-separation", manifest, "skipped=no backend")
        return ctx.update_manifest(manifest)
    if result is None:
        mark_stage(manifest, "source-separation", "skipped", backend=backend, reason="disabled")
        save_manifest(project_dir, manifest)
        _log_stage_complete("source-separation", manifest, "skipped")
        return ctx.update_manifest(manifest)

    _validate_audio_contract(result.vocals_path, cfg.mix_sample_rate, 2, "source_vocals_48k")
    _validate_audio_contract(result.vocals_mono_path, cfg.gemma_sample_rate, 1, "source_vocals_mono_16k")
    _validate_audio_contract(result.background_path, cfg.mix_sample_rate, 2, "background_only_48k")
    manifest.artifacts["source_vocals_48k"] = str(result.vocals_path)
    manifest.artifacts["source_vocals_mono_16k"] = str(result.vocals_mono_path)
    manifest.artifacts["background_only_48k"] = str(result.background_path)
    manifest.artifacts["source_separation_manifest"] = str(result.metadata_path)

    resliced_segments = 0
    if manifest.segments:
        started_at = monotonic()
        last_logged_at = started_at

        def log_reslice_progress(index: int, total: int, segment: Segment) -> None:
            nonlocal last_logged_at
            last_logged_at = _log_segment_progress(
                "source-separation clips",
                index,
                total,
                segment,
                manifest,
                started_at,
                last_logged_at,
            )

        write_segment_audio_clips(
            manifest.segments,
            result.vocals_mono_path,
            result.vocals_path,
            project_dir,
            progress_callback=log_reslice_progress,
        )
        resliced_segments = len(manifest.segments)
        out_path = project_dir / "work" / "segments" / "manifests" / "segments_source_separated.json"
        write_json_atomic(out_path, {"segments": [s.model_dump(mode="json") for s in manifest.segments]})
        manifest.artifacts["segments_source_separated"] = str(out_path)

    mark_stage(
        manifest,
        "source-separation",
        "completed",
        backend=result.backend,
        model=result.model,
        reused_existing=result.reused_existing,
        vocals_path=str(result.vocals_path),
        vocals_mono_path=str(result.vocals_mono_path),
        background_path=str(result.background_path),
        resliced_segments=resliced_segments,
    )
    save_manifest(project_dir, manifest)
    _log_stage_complete(
        "source-separation",
        manifest,
        f"backend={result.backend}, reused={result.reused_existing}",
    )
    return ctx.update_manifest(manifest)
