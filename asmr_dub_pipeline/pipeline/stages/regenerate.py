from __future__ import annotations

# ruff: noqa: F403,F405,I001

from asmr_dub_pipeline.pipeline.context import PipelineContext
from asmr_dub_pipeline.pipeline.stages.common import *
from asmr_dub_pipeline.pipeline.stages.experimental_tts import run_synth_experimental_tts_stage
from asmr_dub_pipeline.pipeline.stages.qc import run_qc_stage
from asmr_dub_pipeline.pipeline.stages.rvc import run_rvc_stage
from asmr_dub_pipeline.pipeline.stages.synth_gpt_sovits import run_synth_stage
from asmr_dub_pipeline.pipeline.stages.synth_qwen import run_synth_qwen_stage


def run_regenerate_needs_stage(ctx: PipelineContext, *, refs_path: Path = Path('refs/refs.json'), confirm_rights: bool = False, gemma_backend: str = 'mock', tts_backend: str = 'gpt-sovits', gsv_url: str | None = None, gpt_weights_path: str | None = None, sovits_weights_path: str | None = None, use_trained_gpt: bool = False, auto_gsv_server: bool | None = None, gsv_server_command: list[str] | str | None = None, qwen_model_id: str | None = None, qwen_candidate_count: int | None = None, qwen_local_files_only: bool | None = None, experimental_tts_base_url: str | None = None, experimental_tts_candidate_count: int | None = None) -> PipelineManifest:
    project_dir = ctx.project_dir
    manifest = ctx.reload_manifest()
    _load_config_into_manifest(project_dir, manifest)
    target_ids = {
        segment.id
        for segment in manifest.segments
        if segment.status == "needs_regeneration"
        and segment.qc is not None
        and segment.qc.recommendation == "regenerate"
    }
    _log_stage_start("regenerate", f"segments={len(target_ids)}, tts_backend={tts_backend}")
    if not target_ids:
        mark_stage(
            manifest,
            "regenerate",
            "skipped",
            target_status="needs_regeneration",
            target_segments=[],
            segment_counts=_segment_counts(manifest),
        )
        save_manifest(project_dir, manifest)
        _log_stage_complete("regenerate", manifest, "no target segments")
        return ctx.update_manifest(manifest)

    for segment in manifest.segments:
        if segment.id not in target_ids:
            continue
        segment.rvc = None
        segment.mix = {}
    _invalidate_downstream_after_tts_promotion(manifest)
    save_manifest(project_dir, manifest)

    backend = tts_backend.strip().lower().replace("_", "-")
    if backend in {"gpt-sovits", "gsv"}:
        run_synth_stage(
            ctx,
            gsv_url,
            refs_path,
            mock=False,
            confirm_rights=confirm_rights,
            gpt_weights_path=gpt_weights_path,
            sovits_weights_path=sovits_weights_path,
            use_trained_gpt=use_trained_gpt,
            auto_gsv_server=auto_gsv_server,
            gsv_server_command=gsv_server_command,
            only_segment_ids=target_ids,
        )
    elif backend == "mock":
        run_synth_stage(
            ctx,
            gsv_url,
            refs_path,
            mock=True,
            confirm_rights=confirm_rights,
            only_segment_ids=target_ids,
        )
    elif backend == "qwen":
        run_synth_qwen_stage(
            ctx,
            refs_path,
            confirm_rights=confirm_rights,
            model_id=qwen_model_id,
            candidate_count=qwen_candidate_count,
            promote=True,
            local_files_only=qwen_local_files_only,
            only_segment_ids=target_ids,
        )
    elif backend in {"fish", "fish-tts", "fish-speech", "cosyvoice", "cosy", "cosy-voice"}:
        run_synth_experimental_tts_stage(
            ctx,
            refs_path,
            backend=backend,
            confirm_rights=confirm_rights,
            base_url=experimental_tts_base_url,
            candidate_count=experimental_tts_candidate_count,
            promote=True,
            only_segment_ids=target_ids,
        )
    else:
        raise ValueError("tts_backend must be one of: gpt-sovits, qwen, fish, cosyvoice, mock")

    run_rvc_stage(ctx, confirm_rights=confirm_rights, only_segment_ids=target_ids)
    manifest = run_qc_stage(
        ctx,
        gemma_backend,
        confirm_rights=confirm_rights,
        only_segment_ids=target_ids,
    )
    remaining = [
        segment.id for segment in manifest.segments if segment.id in target_ids and segment.status == "needs_regeneration"
    ]
    mark_stage(
        manifest,
        "regenerate",
        "completed",
        tts_backend=backend,
        gemma_backend=gemma_backend,
        target_segments=sorted(target_ids),
        remaining_needs_regeneration=remaining,
        segment_counts=_segment_counts(manifest),
    )
    save_manifest(project_dir, manifest)
    _log_stage_complete(
        "regenerate",
        manifest,
        f"processed={len(target_ids)} remaining_needs_regeneration={len(remaining)}",
    )
    return ctx.update_manifest(manifest)
