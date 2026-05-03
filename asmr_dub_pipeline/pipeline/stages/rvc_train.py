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

    dataset_dir, dataset_rows = _rvc_train_dataset(project_dir, manifest, force)
    work_dir = project_dir / "work" / "rvc_train"
    model_path, index_path = rvc_train_output_paths(project_dir, cfg)
    console.print(
        f"[cyan]train-rvc[/cyan] dataset ready: {len(dataset_rows)} wav(s) -> {escape(str(dataset_dir))}"
    )
    console.print(
        f"[cyan]train-rvc[/cyan] outputs: model={escape(str(model_path))} index={escape(str(index_path))}"
    )
    if isinstance(client, RVCTrainCommandClient):
        command_preview = client.build_command(
            project_dir=project_dir,
            dataset_dir=dataset_dir,
            work_dir=work_dir,
            model_path=model_path,
            index_path=index_path,
            cfg=cfg,
        )
        console.print(f"[dim]train-rvc command: {escape(_format_command_preview(command_preview))}[/dim]")
    console.print(f"[cyan]train-rvc[/cyan] running backend={backend}")
    result = client.train(
        project_dir=project_dir,
        dataset_dir=dataset_dir,
        work_dir=work_dir,
        model_path=model_path,
        index_path=index_path,
        cfg=cfg,
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
    )
    save_manifest(project_dir, manifest)
    _log_stage_complete("train-rvc", manifest, f"backend={backend}")
    return ctx.update_manifest(manifest)


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
