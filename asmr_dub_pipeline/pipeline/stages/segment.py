from __future__ import annotations

# ruff: noqa: F403,F405,I001

from asmr_dub_pipeline.pipeline.context import PipelineContext
from asmr_dub_pipeline.pipeline.stages.common import *


def run_segment_stage(ctx: PipelineContext, confirm_rights: bool = False) -> PipelineManifest:
    project_dir = ctx.project_dir
    _log_stage_start("segment", f"project={project_dir}")
    manifest = ctx.reload_manifest()
    _load_config_into_manifest(project_dir, manifest)
    _require_audio_stage_rights(manifest, "segment", confirm_rights)
    cfg = manifest.project_config
    if manifest.stage_state.get("transcribe", {}).get("status") == "completed" and manifest.segments:
        raw_path = project_dir / "work" / "segments" / "manifests" / "segments_raw.json"
        final_path = project_dir / "work" / "segments" / "manifests" / "segments_final.json"
        _write_segments_manifest(raw_path, manifest.segments)
        _write_segments_manifest(final_path, manifest.segments)
        manifest.artifacts["segments_raw"] = str(raw_path)
        manifest.artifacts["segments_final"] = str(final_path)
        mark_stage(
            manifest,
            "segment",
            "completed",
            source="transcribe",
            segment_count=len(manifest.segments),
            segment_counts=_segment_counts(manifest),
        )
        save_manifest(project_dir, manifest)
        _log_stage_complete("segment", manifest, f"finalized={len(manifest.segments)}")
        return ctx.update_manifest(manifest)
    started_at = monotonic()
    last_logged_at = started_at

    def log_segment_progress(index: int, total: int, segment: Segment) -> None:
        nonlocal last_logged_at
        last_logged_at = _log_segment_progress(
            "segment",
            index,
            total,
            segment,
            None,
            started_at,
            last_logged_at,
            note="writing segment audio",
        )

    manual = project_dir / "work" / "segments" / "manifests" / "segments_manual.json"
    if manual.exists():
        segments = load_manual_segments(manual)
        total = len(segments)
        for index, segment in enumerate(segments, start=1):
            _validate_segment_audio_paths(project_dir, segment, check_formats=True)
            log_segment_progress(index, total, segment)
    else:
        gemma_audio = Path(
            manifest.artifacts.get(
                "source_vocals_mono_16k",
                manifest.artifacts.get("gemma_mono_16k", project_dir / "work/audio/gemma_mono_16k.wav"),
            )
        )
        mix_audio = Path(
            manifest.artifacts.get(
                "source_vocals_48k",
                manifest.artifacts.get("original_stereo_48k", project_dir / "work/audio/original_stereo_48k.wav"),
            )
        )
        segments = energy_segments(
            gemma_audio,
            mix_audio,
            project_dir,
            min_segment_sec=cfg.segmentation_min_segment_sec,
            max_segment_sec=cfg.segmentation_max_segment_sec,
            silence_db=cfg.segmentation_silence_db,
            min_silence_sec=cfg.segmentation_min_silence_sec,
            progress_callback=log_segment_progress,
        )
    manifest.segments = segments
    raw_path = project_dir / "work" / "segments" / "manifests" / "segments_raw.json"
    _write_segments_manifest(raw_path, segments)
    manifest.artifacts["segments_raw"] = str(raw_path)
    mark_stage(manifest, "segment", "completed", segment_count=len(segments), segment_counts=_segment_counts(manifest))
    save_manifest(project_dir, manifest)
    _log_stage_complete("segment", manifest, f"created={len(segments)}")
    return ctx.update_manifest(manifest)
