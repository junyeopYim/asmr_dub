from __future__ import annotations

# ruff: noqa: F403,F405,I001

from asmr_dub_pipeline.pipeline.context import PipelineContext
from asmr_dub_pipeline.pipeline.stages.common import *


def run_gsv_few_shot_stage(ctx: PipelineContext, confirm_rights: bool = False, force: bool | None = None, gsv_url: str | None = None, gsv_server_command: list[str] | str | None = None) -> PipelineManifest:
    project_dir = ctx.project_dir
    manifest = ctx.reload_manifest()
    _load_config_into_manifest(project_dir, manifest)
    cfg = manifest.project_config
    training_cfg = cfg.model_copy(update={"gsv_url": gsv_url or cfg.gsv_url})
    _log_stage_start(FEW_SHOT_STAGE, f"min_total={few_shot_min_total_sec(cfg):g}s")
    started_at = monotonic()

    def log_training_progress(event: FewShotTrainingProgress) -> None:
        elapsed = _format_elapsed(monotonic() - started_at)
        if event.status == "output":
            detail = escape(_log_text_snippet(event.detail, max_chars=220))
            label = "fine-tune log" if event.phase.startswith("fine-tune") else "prep log"
            console.print(
                f"[dim]{FEW_SHOT_STAGE} {label} - phase={event.phase} "
                f"elapsed={elapsed} {detail}[/dim]"
            )
            return
        if event.phase == "dataset":
            console.print(
                f"[dim]{FEW_SHOT_STAGE}: dataset ready - elapsed={elapsed} "
                f"{escape(event.detail or '')} log={escape(str(event.log_path or ''))}[/dim]"
            )
            return
        if event.phase == "reuse":
            console.print(
                f"[dim]{FEW_SHOT_STAGE}: reused cached weights - elapsed={elapsed} "
                f"log={escape(str(event.log_path or ''))}[/dim]"
            )
            return
        percent = (event.index / event.total * 100.0) if event.total else 100.0
        console.print(
            f"[dim]{FEW_SHOT_STAGE}: {event.index}/{event.total} ({percent:.1f}%) "
            f"elapsed={elapsed} phase={event.phase} status={event.status} "
            f"log={escape(str(event.log_path or ''))}[/dim]"
        )

    manifest.rights_audit = require_existing_or_confirmed_rights(
        manifest.rights_audit,
        confirm_rights,
        FEW_SHOT_STAGE,
        _manifest_source_path(manifest),
        metadata={"source_derived_few_shot_training": True},
    )
    try:
        speaker_ids = _gsv_training_speaker_ids(project_dir, manifest, cfg)
    except GPTSoVITSError as exc:
        skipped = _maybe_skip_gsv_few_shot_for_insufficient_data(project_dir, manifest, cfg, exc)
        if skipped is not None:
            return ctx.update_manifest(skipped)
        raise
    if len(speaker_ids) > 1:
        results: dict[str, Any] = {}
        speaker_models: dict[str, GSVSpeakerConfig] = {}
        for speaker_id in speaker_ids:
            work_dir = project_dir / "work" / "gpt_sovits" / "few_shot" / "speakers" / speaker_id
            try:
                result = train_few_shot(
                    project_dir,
                    manifest,
                    training_cfg,
                    speaker_id=speaker_id,
                    work_dir=work_dir,
                    force=force,
                    command=gsv_server_command if gsv_server_command is not None else cfg.gsv_server_command,
                    progress_callback=log_training_progress,
                )
            except GPTSoVITSError as exc:
                skipped = _maybe_skip_gsv_few_shot_for_insufficient_data(project_dir, manifest, cfg, exc)
                if skipped is not None:
                    return ctx.update_manifest(skipped)
                raise
            refs_path = _prepare_gsv_speaker_refs(project_dir, manifest, cfg, speaker_id)
            speaker_models[speaker_id] = GSVSpeakerConfig(
                gpt_weights_path=str(result.gpt_weights_path),
                sovits_weights_path=str(result.sovits_weights_path),
                refs_path=str(refs_path),
            )
            results[speaker_id] = {
                "status": result.status,
                "reused_existing": result.reused_existing,
                "fingerprint": result.fingerprint,
                "selected_duration_sec": result.dataset.total_duration_sec,
                "selected_segment_ids": [item.segment_id for item in result.dataset.items],
                "dataset_list_path": str(result.dataset.list_path),
                "dataset_wav_dir": str(result.dataset.wav_dir),
                "metadata_path": str(result.metadata_path),
                "gpt_weights_path": str(result.gpt_weights_path),
                "sovits_weights_path": str(result.sovits_weights_path),
                "gpt_weights_sha256": result.gpt_weights_sha256,
                "sovits_weights_sha256": result.sovits_weights_sha256,
                "refs_path": str(refs_path),
                "log_path": str(result.log_path),
                "gpt_sovits_root": str(result.install.root),
                "gpt_sovits_checkout": result.install.checkout,
                "gpt_sovits_version": result.install.version,
            }
        speaker_manifest_path = project_dir / "work" / "gpt_sovits" / "few_shot" / "speaker_models.json"
        write_json_atomic(speaker_manifest_path, {"speakers": results})
        updated_cfg = _config_with_gsv_speaker_models(cfg, speaker_models)
        save_project_config(updated_cfg, project_dir / "pipeline.yaml")
        manifest.project_config = updated_cfg
        manifest.artifacts["gsv_few_shot_speaker_models"] = str(speaker_manifest_path)
        manifest.rights_audit = record_rights_reliance(
            manifest.rights_audit,
            FEW_SHOT_STAGE,
            _manifest_source_path(manifest),
            metadata={
                "source_derived_few_shot_training": True,
                "speaker_ids": speaker_ids,
                "speaker_results": {
                    speaker_id: {
                        "selected_duration_sec": row["selected_duration_sec"],
                        "selected_segment_ids": row["selected_segment_ids"],
                        "gpt_weights_sha256": row["gpt_weights_sha256"],
                        "sovits_weights_sha256": row["sovits_weights_sha256"],
                    }
                    for speaker_id, row in results.items()
                },
                "source_language": cfg.source_language,
                "target_language": cfg.target_language,
                "cross_lingual_voice_transfer": cfg.source_language != cfg.target_language,
            },
        )
        mark_stage(
            manifest,
            FEW_SHOT_STAGE,
            "completed",
            speaker_count=len(speaker_ids),
            speaker_ids=speaker_ids,
            speaker_models={speaker_id: speaker_cfg.model_dump(mode="json") for speaker_id, speaker_cfg in speaker_models.items()},
            speaker_results=results,
            source_language=cfg.source_language,
            target_language=cfg.target_language,
            cross_lingual_voice_transfer=cfg.source_language != cfg.target_language,
        )
        save_manifest(project_dir, manifest)
        _log_stage_complete(FEW_SHOT_STAGE, manifest, f"trained speaker_count={len(speaker_ids)}")
        return ctx.update_manifest(manifest)

    try:
        result = train_few_shot(
            project_dir,
            manifest,
            training_cfg,
            force=force,
            command=gsv_server_command if gsv_server_command is not None else cfg.gsv_server_command,
            progress_callback=log_training_progress,
        )
    except GPTSoVITSError as exc:
        skipped = _maybe_skip_gsv_few_shot_for_insufficient_data(project_dir, manifest, cfg, exc)
        if skipped is not None:
            return ctx.update_manifest(skipped)
        raise
    manifest.artifacts[FEW_SHOT_ARTIFACT_GPT] = str(result.gpt_weights_path)
    manifest.artifacts[FEW_SHOT_ARTIFACT_SOVITS] = str(result.sovits_weights_path)
    manifest.artifacts["gsv_few_shot_dataset"] = str(result.dataset.list_path)
    manifest.artifacts["gsv_few_shot_manifest"] = str(result.metadata_path)
    source_clip_qc_path = project_dir / "work" / "gpt_sovits" / "few_shot" / "source_clip_qc.json"
    if source_clip_qc_path.exists():
        manifest.artifacts["gsv_few_shot_source_clip_qc"] = str(source_clip_qc_path)
    manifest.rights_audit = record_rights_reliance(
        manifest.rights_audit,
        FEW_SHOT_STAGE,
        _manifest_source_path(manifest),
        metadata={
            "source_derived_few_shot_training": True,
            "selected_duration_sec": result.dataset.total_duration_sec,
            "selected_segment_ids": [item.segment_id for item in result.dataset.items],
            "source_language": cfg.source_language,
            "target_language": cfg.target_language,
            "cross_lingual_voice_transfer": cfg.source_language != cfg.target_language,
            "gpt_weights_sha256": result.gpt_weights_sha256,
            "sovits_weights_sha256": result.sovits_weights_sha256,
        },
    )
    mark_stage(
        manifest,
        FEW_SHOT_STAGE,
        result.status,
        reused_existing=result.reused_existing,
        fingerprint=result.fingerprint,
        selected_duration_sec=result.dataset.total_duration_sec,
        selected_segment_ids=[item.segment_id for item in result.dataset.items],
        source_language=cfg.source_language,
        target_language=cfg.target_language,
        cross_lingual_voice_transfer=cfg.source_language != cfg.target_language,
        source_clip_qc_path=str(source_clip_qc_path) if source_clip_qc_path.exists() else None,
        gpt_weights_path=str(result.gpt_weights_path),
        sovits_weights_path=str(result.sovits_weights_path),
        gpt_weights_sha256=result.gpt_weights_sha256,
        sovits_weights_sha256=result.sovits_weights_sha256,
        gpt_sovits_root=str(result.install.root),
        gpt_sovits_checkout=result.install.checkout,
        gpt_sovits_version=result.install.version,
        log_path=str(result.log_path),
    )
    save_manifest(project_dir, manifest)
    _log_stage_complete(
        FEW_SHOT_STAGE,
        manifest,
        f"{'reused' if result.reused_existing else 'trained'} version={result.install.version}",
    )
    return ctx.update_manifest(manifest)


def _maybe_skip_gsv_few_shot_for_insufficient_data(
    project_dir: Path,
    manifest: PipelineManifest,
    cfg: ProjectConfig,
    exc: GPTSoVITSError,
) -> PipelineManifest | None:
    if cfg.gsv_few_shot_insufficient_policy != "zero_shot":
        return None
    if "Not enough source voice data" not in str(exc):
        return None
    mark_stage(
        manifest,
        FEW_SHOT_STAGE,
        "skipped_insufficient_training_data",
        policy="zero_shot",
        reason=str(exc),
        min_total_sec=few_shot_min_total_sec(cfg),
        min_clip_sec=cfg.gsv_few_shot_min_clip_sec,
        max_clip_sec=cfg.gsv_few_shot_max_clip_sec,
    )
    save_manifest(project_dir, manifest)
    _log_stage_complete(FEW_SHOT_STAGE, manifest, "skipped_insufficient_training_data")
    return manifest
