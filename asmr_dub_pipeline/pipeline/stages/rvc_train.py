from __future__ import annotations

# ruff: noqa: F403,F405,I001

from asmr_dub_pipeline.pipeline.context import PipelineContext
from asmr_dub_pipeline.pipeline.stages.common import *


def run_rvc_train_stage(ctx: PipelineContext, confirm_rights: bool = False, force: bool = False, mock: bool | None = None, runner: Any | None = None) -> PipelineManifest:
    project_dir = ctx.project_dir
    manifest = ctx.reload_manifest()
    _load_config_into_manifest(project_dir, manifest)
    cfg = manifest.project_config
    backend = "mock" if mock is True else cfg.rvc_train_backend
    _log_stage_start("train-rvc", f"backend={backend}, segments={len(manifest.segments)}")
    if manifest.stage_state.get("synth", {}).get("status") != "completed":
        raise ValueError("train-rvc requires a completed synth stage.")
    if backend == "command":
        if not confirm_rights:
            raise RightsError("Real RVC training requires --confirm-rights for source voice training data.")
        validate_rvc_training_config(project_dir, cfg, real=True)
        source_path = Path(manifest.source_info.path) if manifest.source_info else None
        manifest.rights_audit = merge_rights_audit(
            manifest.rights_audit,
            require_confirmed_rights(
                True,
                "train-rvc",
                source_path,
                metadata={"backend": "command", "experiment_name": cfg.rvc_train_experiment_name},
            ),
        )
        working_dir = resolve_config_path(project_dir, cfg.rvc_train_working_dir)
        client: Any = RVCTrainCommandClient(
            cfg.rvc_train_command,
            working_dir=working_dir,
            timeout_sec=cfg.rvc_train_timeout_sec,
            runner=runner or subprocess.run,
            stream_output=True,
            log_prefix="train-rvc",
        )
    else:
        validate_rvc_training_config(project_dir, cfg, real=False)
        _require_audio_stage_rights(manifest, "train-rvc", confirm_rights, metadata={"backend": "mock"})
        client = RVCTrainMockClient()

    speaker_ids = _rvc_training_speaker_ids(project_dir, manifest, backend)
    if len(speaker_ids) > 1:
        speaker_models: dict[str, RVCSpeakerConfig] = {}
        speaker_results: dict[str, Any] = {}
        for speaker_id in speaker_ids:
            speaker_work_dir = project_dir / "work" / "rvc_train" / "speakers" / speaker_id
            speaker_dataset_dir = speaker_work_dir / "dataset"
            speaker_cfg = _rvc_train_cfg_for_speaker(project_dir, cfg, speaker_id)
            try:
                dataset_dir, dataset_rows = _rvc_train_dataset(
                    project_dir,
                    manifest,
                    force,
                    speaker_id=speaker_id,
                    dataset_dir=speaker_dataset_dir,
                )
            except RVCCommandError as exc:
                skipped = _maybe_skip_rvc_train_for_insufficient_data(
                    project_dir,
                    manifest,
                    cfg,
                    backend,
                    exc,
                    dataset_dir=speaker_dataset_dir,
                )
                if skipped is not None:
                    return ctx.update_manifest(skipped)
                raise
            dataset_summary = _rvc_train_dataset_summary_from_manifest(project_dir, dataset_dir)
            speaker_effective_cfg, epoch_decision = _rvc_train_effective_epoch_config(speaker_cfg, dataset_summary)
            model_path, index_path = rvc_train_output_paths(project_dir, speaker_effective_cfg)
            console.print(
                f"[cyan]train-rvc[/cyan] speaker={escape(speaker_id)} "
                f"dataset ready: {len(dataset_rows)} wav(s) -> {escape(str(dataset_dir))}"
            )
            console.print(
                f"[cyan]train-rvc[/cyan] speaker={escape(speaker_id)} "
                f"quality={escape(str(epoch_decision.get('quality_grade')))} "
                f"epochs={epoch_decision['configured_epochs']}->{epoch_decision['effective_epochs']}"
            )
            if isinstance(client, RVCTrainCommandClient):
                command_preview = client.build_command(
                    project_dir=project_dir,
                    dataset_dir=dataset_dir,
                    work_dir=speaker_work_dir,
                    model_path=model_path,
                    index_path=index_path,
                    cfg=speaker_effective_cfg,
                )
                console.print(f"[dim]train-rvc command: {escape(_format_command_preview(command_preview))}[/dim]")
            result = client.train(
                project_dir=project_dir,
                dataset_dir=dataset_dir,
                work_dir=speaker_work_dir,
                model_path=model_path,
                index_path=index_path,
                cfg=speaker_effective_cfg,
                force=force,
            )
            speaker_models[speaker_id] = RVCSpeakerConfig(
                model_path=str(result.model_path),
                index_path=str(result.index_path) if result.index_path else None,
            )
            speaker_results[speaker_id] = {
                "dataset_dir": str(dataset_dir),
                "dataset_segments": dataset_rows,
                "dataset_summary": dataset_summary,
                "epoch_decision": epoch_decision,
                "configured_train_epochs": epoch_decision["configured_epochs"],
                "effective_train_epochs": epoch_decision["effective_epochs"],
                "dataset_quality_grade": dataset_summary.get("quality_grade"),
                "model_path": str(result.model_path),
                "index_path": str(result.index_path) if result.index_path else None,
                "command": result.command,
                "returncode": result.returncode,
                "elapsed_sec": round(result.elapsed_sec, 6),
                "reused_existing": result.reused_existing,
                "stdout_tail": result.stdout.strip()[-1200:] if result.stdout else "",
                "stderr_tail": result.stderr.strip()[-1200:] if result.stderr else "",
            }
        train_manifest = project_dir / "work" / "rvc_train" / "rvc_train_manifest.json"
        write_json_atomic(
            train_manifest,
            {
                "backend": backend,
                "mode": "speaker_models",
                "epoch_policy": cfg.rvc_train_epoch_policy,
                "quality_preset": cfg.rvc_train_quality_preset,
                "speaker_models": speaker_results,
            },
        )
        updated_cfg = _config_with_rvc_speaker_models(cfg, speaker_models)
        save_project_config(updated_cfg, project_dir / "pipeline.yaml")
        manifest.project_config = updated_cfg
        manifest.artifacts["rvc_train_manifest"] = str(train_manifest)
        mark_stage(
            manifest,
            "train-rvc",
            "completed",
            backend=backend,
            mode="speaker_models",
            rvc_train_manifest=str(train_manifest),
            speaker_count=len(speaker_ids),
            speaker_ids=speaker_ids,
            speaker_models={speaker_id: speaker_cfg.model_dump(mode="json") for speaker_id, speaker_cfg in speaker_models.items()},
            dataset_segment_count=sum(len(row["dataset_segments"]) for row in speaker_results.values()),
            epoch_policy=cfg.rvc_train_epoch_policy,
            quality_preset=cfg.rvc_train_quality_preset,
            effective_train_epochs_by_speaker={
                speaker_id: result["effective_train_epochs"] for speaker_id, result in speaker_results.items()
            },
            dataset_quality_grades={
                speaker_id: result["dataset_quality_grade"] for speaker_id, result in speaker_results.items()
            },
        )
        save_manifest(project_dir, manifest)
        _log_stage_complete("train-rvc", manifest, f"backend={backend} speaker_count={len(speaker_ids)}")
        return ctx.update_manifest(manifest)

    try:
        dataset_dir, dataset_rows = _rvc_train_dataset(project_dir, manifest, force)
    except RVCCommandError as exc:
        skipped = _maybe_skip_rvc_train_for_insufficient_data(project_dir, manifest, cfg, backend, exc)
        if skipped is not None:
            return ctx.update_manifest(skipped)
        raise
    work_dir = project_dir / "work" / "rvc_train"
    dataset_summary = _rvc_train_dataset_summary_from_manifest(project_dir, dataset_dir)
    effective_cfg, epoch_decision = _rvc_train_effective_epoch_config(cfg, dataset_summary)
    model_path, index_path = rvc_train_output_paths(project_dir, effective_cfg)
    console.print(
        f"[cyan]train-rvc[/cyan] dataset ready: {len(dataset_rows)} wav(s) -> {escape(str(dataset_dir))}"
    )
    console.print(
        f"[cyan]train-rvc[/cyan] outputs: model={escape(str(model_path))} index={escape(str(index_path))}"
    )
    console.print(
        f"[cyan]train-rvc[/cyan] quality={escape(str(epoch_decision.get('quality_grade')))} "
        f"epochs={epoch_decision['configured_epochs']}->{epoch_decision['effective_epochs']}"
    )
    if isinstance(client, RVCTrainCommandClient):
        command_preview = client.build_command(
            project_dir=project_dir,
            dataset_dir=dataset_dir,
            work_dir=work_dir,
            model_path=model_path,
            index_path=index_path,
            cfg=effective_cfg,
        )
        console.print(f"[dim]train-rvc command: {escape(_format_command_preview(command_preview))}[/dim]")
    console.print(f"[cyan]train-rvc[/cyan] running backend={backend}")
    result = client.train(
        project_dir=project_dir,
        dataset_dir=dataset_dir,
        work_dir=work_dir,
        model_path=model_path,
        index_path=index_path,
        cfg=effective_cfg,
        force=force,
    )
    reuse_note = "reused existing artifacts" if result.reused_existing else f"elapsed={result.elapsed_sec:.1f}s"
    console.print(f"[cyan]train-rvc[/cyan] backend finished: {reuse_note}")
    train_manifest = project_dir / "work" / "rvc_train" / "rvc_train_manifest.json"
    write_json_atomic(
        train_manifest,
        {
            "backend": backend,
            "dataset_dir": str(dataset_dir),
            "dataset_segments": dataset_rows,
            "dataset_summary": dataset_summary,
            "epoch_decision": epoch_decision,
            "configured_train_epochs": epoch_decision["configured_epochs"],
            "effective_train_epochs": epoch_decision["effective_epochs"],
            "dataset_quality_grade": dataset_summary.get("quality_grade"),
            "model_path": str(result.model_path),
            "index_path": str(result.index_path) if result.index_path else None,
            "command": result.command,
            "returncode": result.returncode,
            "elapsed_sec": round(result.elapsed_sec, 6),
            "reused_existing": result.reused_existing,
            "stdout_tail": result.stdout.strip()[-1200:] if result.stdout else "",
            "stderr_tail": result.stderr.strip()[-1200:] if result.stderr else "",
        },
    )
    manifest.artifacts["rvc_train_manifest"] = str(train_manifest)
    manifest.artifacts["rvc_model_path"] = str(result.model_path)
    if result.index_path:
        manifest.artifacts["rvc_index_path"] = str(result.index_path)
    mark_stage(
        manifest,
        "train-rvc",
        "completed",
        backend=backend,
        rvc_train_manifest=str(train_manifest),
        model_path=str(result.model_path),
        index_path=str(result.index_path) if result.index_path else None,
        dataset_segment_count=len(dataset_rows),
        epoch_policy=cfg.rvc_train_epoch_policy,
        quality_preset=cfg.rvc_train_quality_preset,
        configured_train_epochs=epoch_decision["configured_epochs"],
        effective_train_epochs=epoch_decision["effective_epochs"],
        recommended_epoch_count=epoch_decision["recommended_epoch_count"],
        dataset_quality_grade=dataset_summary.get("quality_grade"),
    )
    save_manifest(project_dir, manifest)
    _log_stage_complete("train-rvc", manifest, f"backend={backend}")
    return ctx.update_manifest(manifest)


def _rvc_train_cfg_for_speaker(project_dir: Path, cfg: ProjectConfig, speaker_id: str) -> ProjectConfig:
    experiment_name = f"{cfg.rvc_train_experiment_name}-{speaker_id}"
    output_dir = project_dir / "work" / "rvc_train" / "speakers" / speaker_id / "model"
    payload = cfg.model_dump(mode="json")
    rvc_payload = dict(payload.get("rvc") or {})
    rvc_payload.update(
        {
            "train_experiment_name": experiment_name,
            "train_output_model_path": str(output_dir / f"{experiment_name}.pth"),
            "train_output_index_path": str(output_dir / f"added_{experiment_name}.index"),
        }
    )
    payload["rvc"] = rvc_payload
    return ProjectConfig.model_validate(payload)


def _maybe_skip_rvc_train_for_insufficient_data(
    project_dir: Path,
    manifest: PipelineManifest,
    cfg: ProjectConfig,
    backend: str,
    exc: RVCCommandError,
    *,
    dataset_dir: Path | None = None,
) -> PipelineManifest | None:
    if not _is_rvc_insufficient_training_data(exc):
        return None
    if not (cfg.rvc_allow_pre_rvc_fallback and not _rvc_downstream_required(cfg)):
        return None
    dataset_manifest = _rvc_train_dataset_manifest_path(project_dir, dataset_dir)
    summary: dict[str, Any] = {}
    if dataset_manifest.exists():
        try:
            payload = json.loads(dataset_manifest.read_text("utf-8"))
            if isinstance(payload.get("summary"), dict):
                summary = payload["summary"]
        except Exception:
            summary = {}
    manifest.artifacts["rvc_train_dataset_manifest"] = str(dataset_manifest)
    mark_stage(
        manifest,
        "train-rvc",
        "skipped_insufficient_training_data",
        backend=backend,
        reason=str(exc),
        policy="pre_rvc_fallback",
        rvc_train_dataset_manifest=str(dataset_manifest),
        clean_segment_count=summary.get("clean_segment_count", 0),
        clean_duration_sec=summary.get("clean_duration_sec", 0.0),
        real_clean_segment_count=summary.get("real_clean_segment_count", 0),
        real_clean_duration_sec=summary.get("real_clean_duration_sec", 0.0),
        augmented_segment_count=summary.get("augmented_segment_count", 0),
        augmented_duration_sec=summary.get("augmented_duration_sec", 0.0),
        augmentation_applied=summary.get("augmentation_applied", False),
        augmentation_skipped_reason=summary.get("augmentation_skipped_reason"),
        min_clean_segments=summary.get("min_clean_segments", cfg.rvc_train_min_clean_segments),
        min_clean_sec=summary.get("min_clean_sec", cfg.rvc_train_min_clean_sec),
        insufficient_reasons=summary.get("insufficient_reasons", []),
    )
    save_manifest(project_dir, manifest)
    _log_stage_complete("train-rvc", manifest, "skipped_insufficient_training_data")
    return manifest


def run_skip_rvc_train_for_voice_bank_stage(ctx: PipelineContext) -> PipelineManifest:
    project_dir = ctx.project_dir
    manifest = ctx.reload_manifest()
    _load_config_into_manifest(project_dir, manifest)
    cfg = manifest.project_config
    if not cfg.rvc_speaker_models:
        raise ValueError("Cannot skip train-rvc without configured voice-bank RVC speaker models.")
    mark_stage(
        manifest,
        "train-rvc",
        "skipped_pretrained_voice_bank",
        backend="voice_bank",
        speaker_models=sorted(cfg.rvc_speaker_models),
    )
    save_manifest(project_dir, manifest)
    _log_stage_complete("train-rvc", manifest, "skipped_pretrained_voice_bank")
    return ctx.update_manifest(manifest)
