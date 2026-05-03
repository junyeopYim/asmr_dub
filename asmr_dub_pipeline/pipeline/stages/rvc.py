from __future__ import annotations

# ruff: noqa: F403,F405,I001

from asmr_dub_pipeline.pipeline.context import PipelineContext
from asmr_dub_pipeline.pipeline.stages.common import *


def run_rvc_stage(ctx: PipelineContext, confirm_rights: bool = False, force: bool = False, mock: bool | None = None, runner: Any | None = None, only_segment_ids: set[str] | None = None) -> PipelineManifest:
    project_dir = ctx.project_dir
    manifest = ctx.reload_manifest()
    _load_config_into_manifest(project_dir, manifest)
    cfg = manifest.project_config
    backend = "mock" if mock is True else cfg.rvc_backend
    _log_stage_start("rvc", f"backend={backend}, segments={len(manifest.segments)}")
    if manifest.stage_state.get("synth", {}).get("status") != "completed":
        raise ValueError("RVC requires a completed synth stage.")
    if not _train_rvc_ready_for_rvc(manifest):
        raise ValueError("RVC requires a completed train-rvc stage.")
    validate_rvc_config(
        project_dir,
        cfg,
        real=backend == "command",
        segments=manifest.segments,
        allow_trained_artifact=True,
    )
    if backend == "command":
        working_dir: Path | None = None
        if not confirm_rights:
            raise RightsError("Real RVC conversion requires --confirm-rights for the source and voice model.")
        source_path = Path(manifest.source_info.path) if manifest.source_info else None
        manifest.rights_audit = merge_rights_audit(
            manifest.rights_audit,
            require_confirmed_rights(
                True,
                "rvc",
                source_path,
                metadata={
                    "backend": "command",
                    "model_path": cfg.rvc_model_path,
                    "index_path": cfg.rvc_index_path,
                    "speaker_models": sorted(cfg.rvc_speaker_models),
                },
            ),
        )
        working_dir = resolve_config_path(project_dir, cfg.rvc_working_dir)
        client: Any = RVCCommandClient(
            cfg.rvc_command,
            working_dir=working_dir,
            timeout_sec=cfg.rvc_timeout_sec,
            runner=runner or subprocess.run,
            stream_output=True,
            log_prefix="rvc",
        )
    else:
        _require_audio_stage_rights(manifest, "rvc", confirm_rights, metadata={"backend": "mock"})
        client = RVCMockClient()

    failed_segments: list[str] = []
    started_at = monotonic()
    last_logged_at = started_at
    total = len(manifest.segments)
    use_batch_rvc = backend == "command" and cfg.rvc_batch_infer and bool(cfg.rvc_batch_command)
    rvc_lane_count = 1 if backend != "command" else _effective_lane_count(cfg.rvc_concurrency, total)
    batch_lane_count = _effective_lane_count(cfg.rvc_batch_concurrency, total) if use_batch_rvc else 1
    console.print(
        f"[cyan]rvc[/cyan] converting segments with {len(cfg.rvc_auto_profiles)} profile candidate(s); "
        f"failure_policy={cfg.rvc_failure_policy} "
        f"mode={'batch' if use_batch_rvc else 'per-segment'} "
        f"concurrency={batch_lane_count if use_batch_rvc else rvc_lane_count}"
    )

    def prepare_segment(
        segment: Segment,
    ) -> tuple[tuple[Path, Path | None, Path | None] | None, str | None]:
        if segment.status in SKIP_STATUSES:
            return None, None
        if not segment.tts or not segment.tts.selected_candidate_path:
            message = "RVC requires segment.tts.selected_candidate_path from synth."
            segment.status = "failed"
            segment.errors.append(message)
            return None, segment.id
        raw_tts_path = segment.rvc.input_path if segment.rvc and segment.rvc.input_path else segment.tts.selected_candidate_path
        input_path = Path(raw_tts_path)
        if not input_path.exists():
            message = f"RVC input does not exist: {input_path}"
            segment.status = "failed"
            segment.errors.append(message)
            segment.rvc = RVCMetadata(
                backend=backend,
                input_path=str(input_path),
                error=message,
            )
            return None, segment.id
        model_path, index_path = _rvc_model_paths(project_dir, cfg, segment, manifest)
        if backend == "command" and (model_path is None or not model_path.exists()):
            message = "RVC requires the model artifact produced by train-rvc."
            segment.status = "failed"
            segment.errors.append(message)
            segment.rvc = RVCMetadata(backend=backend, input_path=str(input_path), error=message)
            return None, segment.id
        return (input_path, model_path, index_path), None

    def convert_segment(index: int, segment: Segment) -> tuple[int, Segment, str | None]:
        prepared, failed_segment_id = prepare_segment(segment)
        if prepared is None:
            return index, segment, failed_segment_id
        input_path, model_path, index_path = prepared
        attempts: list[dict[str, Any]] = []
        candidate_paths: list[str] = []
        accepted_attempt: dict[str, Any] | None = None
        selected_candidate_path: Path | None = None
        profiles = cfg.rvc_auto_profiles
        if cfg.rvc_failure_policy == "error":
            profiles = profiles[:1]
        for profile in profiles:
            effective_profile = _rvc_profile_for_segment(cfg, profile, segment)
            candidate_path = (
                project_dir
                / "work"
                / "rvc"
                / "candidates"
                / segment.id
                / f"{effective_profile.name}.wav"
            )
            candidate_paths.append(str(candidate_path))
            command: list[str] | None = None
            try:
                console.print(
                    f"[dim]rvc candidate: {index}/{total} segment={escape(segment.id)} "
                    f"profile={escape(effective_profile.name)} output={escape(str(candidate_path))}[/dim]"
                )
                if isinstance(client, RVCCommandClient):
                    command = client.build_command(
                        input_path=input_path,
                        output_path=candidate_path,
                        model_path=model_path,
                        index_path=index_path,
                        cfg=cfg,
                        profile=effective_profile,
                        segment_id=segment.id,
                        sid=segment.speaker_id or "",
                    )
                result = client.convert(
                    input_path,
                    candidate_path,
                    model_path=model_path,
                    index_path=index_path,
                    cfg=cfg,
                    profile=effective_profile,
                    segment_id=segment.id,
                    sid=segment.speaker_id or "",
                    force=force,
                )
                command = result.command or command
                metrics = _rvc_metrics(input_path, candidate_path, segment, cfg)
                attempt = _rvc_attempt_payload(
                    profile=effective_profile,
                    output_path=candidate_path,
                    model_path=model_path,
                    index_path=index_path,
                    command=command,
                    reused_existing=result.reused_existing,
                    returncode=result.returncode,
                    elapsed_sec=result.elapsed_sec,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    metrics=metrics,
                )
                attempts.append(attempt)
                if metrics["accepted"]:
                    console.print(
                        f"[dim]rvc accepted: segment={escape(segment.id)} "
                        f"profile={escape(effective_profile.name)} "
                        f"duration_ratio={metrics.get('duration_ratio', 0):.3f} "
                        f"elapsed={result.elapsed_sec:.1f}s"
                        f"{' reused=true' if result.reused_existing else ''}[/dim]"
                    )
                    accepted_attempt = attempt
                    selected_candidate_path = candidate_path
                    break
                console.print(
                    f"[dim]rvc rejected: segment={escape(segment.id)} "
                    f"profile={escape(effective_profile.name)} issues={metrics.get('issues', [])}[/dim]"
                )
            except Exception as exc:
                console.print(
                    f"[yellow]rvc candidate failed[/yellow]: segment={escape(segment.id)} "
                    f"profile={escape(effective_profile.name)} error={escape(str(exc))}"
                )
                attempts.append(
                    _rvc_attempt_payload(
                        profile=effective_profile,
                        output_path=candidate_path,
                        model_path=model_path,
                        index_path=index_path,
                        command=command,
                        error=str(exc),
                    )
                )
                if cfg.rvc_failure_policy == "error":
                    break
        if selected_candidate_path is None or accepted_attempt is None:
            error = "All RVC candidates failed or were rejected."
            failed_segment_id: str | None = None
            if cfg.rvc_allow_pre_rvc_fallback and not _rvc_downstream_required(cfg):
                segment.rvc = RVCMetadata(
                    backend=backend,
                    input_path=str(input_path),
                    output_path=str(input_path),
                    selected_profile_name=None,
                    candidate_paths=candidate_paths,
                    model_path=str(model_path) if model_path else None,
                    index_path=str(index_path) if index_path else None,
                    accepted=False,
                    fallback_used=True,
                    fallback_reason=error,
                    error=error,
                    attempts=attempts,
                )
            else:
                segment.rvc = RVCMetadata(
                    backend=backend,
                    input_path=str(input_path),
                    output_path=None,
                    selected_profile_name=None,
                    candidate_paths=candidate_paths,
                    model_path=str(model_path) if model_path else None,
                    index_path=str(index_path) if index_path else None,
                    accepted=False,
                    error=error,
                    attempts=attempts,
                )
                segment.status = "failed"
                segment.errors.append(error)
                failed_segment_id = segment.id
            return index, segment, failed_segment_id
        final_path = ensure_inside_project(project_dir, project_dir / "work" / "rvc" / f"{segment.id}_final.wav")
        ensure_not_same_path(selected_candidate_path, final_path)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(selected_candidate_path, final_path)
        metrics = dict(accepted_attempt.get("metrics") or {})
        segment.rvc = RVCMetadata(
            backend=backend,
            input_path=str(input_path),
            output_path=str(final_path),
            selected_profile_name=str(accepted_attempt["profile_name"]),
            candidate_paths=candidate_paths,
            model_path=str(model_path) if model_path else None,
            index_path=str(index_path) if index_path else None,
            settings={
                "failure_policy": cfg.rvc_failure_policy,
                "duration_tolerance": metrics.get("duration_tolerance"),
                "selected_settings": accepted_attempt.get("settings", {}),
            },
            pre_duration_sec=metrics.get("pre_duration_sec"),
            post_duration_sec=metrics.get("post_duration_sec"),
            duration_ratio=metrics.get("duration_ratio"),
            accepted=True,
            fallback_used=False,
            command=accepted_attempt.get("command"),
            attempts=attempts,
        )
        segment.status = "rvc_converted"
        return index, segment, None

    def batch_chunks(items: list[Any], size: int) -> list[list[Any]]:
        return [items[start : start + size] for start in range(0, len(items), size)]

    def rvc_output_exists(segment: Segment) -> bool:
        if force or segment.status != "rvc_converted" or not segment.rvc or not segment.rvc.accepted:
            return False
        if not segment.rvc.output_path:
            return False
        output_path = Path(segment.rvc.output_path).expanduser()
        if not output_path.is_absolute():
            output_path = project_dir / output_path
        try:
            return output_path.exists() and output_path.stat().st_size > 0
        except OSError:
            return False

    def convert_segments_batched(segment_jobs: list[tuple[int, Segment]]) -> None:
        nonlocal last_logged_at
        batch_client = RVCBatchCommandClient(
            cfg.rvc_batch_command,
            working_dir=working_dir,
            timeout_sec=cfg.rvc_timeout_sec,
            runner=runner or subprocess.run,
            stream_output=True,
            log_prefix="rvc-batch",
        )
        prepared_segments: list[tuple[int, Segment, Path, Path, Path | None]] = []
        for index, segment in segment_jobs:
            prepared, failed_segment_id = prepare_segment(segment)
            if failed_segment_id:
                failed_segments.append(failed_segment_id)
            if prepared is None:
                last_logged_at = _log_segment_progress("rvc", index, total, segment, manifest, started_at, last_logged_at)
                save_manifest(project_dir, manifest)
                continue
            input_path, model_path, index_path = prepared
            if model_path is None:
                failed_segments.append(segment.id)
                continue
            prepared_segments.append((index, segment, input_path, model_path, index_path))

        attempts_by_segment: dict[str, list[dict[str, Any]]] = {
            segment.id: [] for _, segment, *_ in prepared_segments
        }
        candidate_paths_by_segment: dict[str, list[str]] = {
            segment.id: [] for _, segment, *_ in prepared_segments
        }
        profiles = cfg.rvc_auto_profiles
        if cfg.rvc_failure_policy == "error":
            profiles = profiles[:1]

        pending = prepared_segments
        batches_dir = project_dir / "work" / "rvc" / "batches"
        batch_counter = 0
        batch_counter_lock = Lock()

        def apply_batch_result(
            entries: list[tuple[int, Segment, Path, Path, Path | None, RVCProfile, Path]],
            results: dict[str, RVCCommandResult] | None,
            batch_error: Exception | None = None,
        ) -> list[tuple[int, Segment, Path, Path, Path | None]]:
            nonlocal last_logged_at
            rejected: list[tuple[int, Segment, Path, Path, Path | None]] = []
            for index, segment, input_path, model_path, index_path, profile, candidate_path in entries:
                attempts = attempts_by_segment[segment.id]
                result = results.get(segment.id) if results else None
                if batch_error is not None or result is None or result.returncode != 0:
                    error = str(batch_error) if batch_error is not None else "RVC batch command did not return a result."
                    command = None
                    if result is not None:
                        command = result.command
                        error = result.stderr or result.stdout or f"RVC batch command failed with exit code {result.returncode}."
                    console.print(
                        f"[yellow]rvc candidate failed[/yellow]: segment={escape(segment.id)} "
                        f"profile={escape(profile.name)} error={escape(error)}"
                    )
                    attempts.append(
                        _rvc_attempt_payload(
                            profile=profile,
                            output_path=candidate_path,
                            model_path=model_path,
                            index_path=index_path,
                            command=command,
                            error=error,
                        )
                    )
                    rejected.append((index, segment, input_path, model_path, index_path))
                    continue

                metrics = _rvc_metrics(input_path, candidate_path, segment, cfg)
                attempt = _rvc_attempt_payload(
                    profile=profile,
                    output_path=candidate_path,
                    model_path=model_path,
                    index_path=index_path,
                    command=result.command,
                    reused_existing=result.reused_existing,
                    returncode=result.returncode,
                    elapsed_sec=result.elapsed_sec,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    metrics=metrics,
                )
                attempts.append(attempt)
                if not metrics["accepted"]:
                    console.print(
                        f"[dim]rvc rejected: segment={escape(segment.id)} "
                        f"profile={escape(profile.name)} issues={metrics.get('issues', [])}[/dim]"
                    )
                    rejected.append((index, segment, input_path, model_path, index_path))
                    continue

                final_path = ensure_inside_project(project_dir, project_dir / "work" / "rvc" / f"{segment.id}_final.wav")
                ensure_not_same_path(candidate_path, final_path)
                final_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(candidate_path, final_path)
                segment.rvc = RVCMetadata(
                    backend=backend,
                    input_path=str(input_path),
                    output_path=str(final_path),
                    selected_profile_name=str(attempt["profile_name"]),
                    candidate_paths=candidate_paths_by_segment[segment.id],
                    model_path=str(model_path),
                    index_path=str(index_path) if index_path else None,
                    settings={
                        "failure_policy": cfg.rvc_failure_policy,
                        "duration_tolerance": metrics.get("duration_tolerance"),
                        "selected_settings": attempt.get("settings", {}),
                        "execution_mode": "batch",
                    },
                    pre_duration_sec=metrics.get("pre_duration_sec"),
                    post_duration_sec=metrics.get("post_duration_sec"),
                    duration_ratio=metrics.get("duration_ratio"),
                    accepted=True,
                    fallback_used=False,
                    command=attempt.get("command"),
                    attempts=attempts,
                )
                segment.status = "rvc_converted"
                console.print(
                    f"[dim]rvc accepted: segment={escape(segment.id)} "
                    f"profile={escape(profile.name)} "
                    f"duration_ratio={metrics.get('duration_ratio', 0):.3f} "
                    f"elapsed={result.elapsed_sec:.1f}s"
                    f"{' reused=true' if result.reused_existing else ''}[/dim]"
                )
                last_logged_at = _log_segment_progress("rvc", index, total, segment, manifest, started_at, last_logged_at)
            return rejected

        for profile in profiles:
            if not pending:
                break
            grouped: dict[tuple[str, str, str], list[tuple[int, Segment, Path, Path, Path | None, RVCProfile, Path]]] = {}
            for index, segment, input_path, model_path, index_path in pending:
                effective_profile = _rvc_profile_for_segment(cfg, profile, segment)
                candidate_path = (
                    project_dir
                    / "work"
                    / "rvc"
                    / "candidates"
                    / segment.id
                    / f"{effective_profile.name}.wav"
                )
                candidate_paths_by_segment[segment.id].append(str(candidate_path))
                console.print(
                    f"[dim]rvc candidate: {index}/{total} segment={escape(segment.id)} "
                    f"profile={escape(effective_profile.name)} output={escape(str(candidate_path))}[/dim]"
                )
                profile_key = json.dumps(effective_profile.model_dump(mode="json"), sort_keys=True)
                key = (str(model_path.resolve()), str(index_path.resolve()) if index_path else "", profile_key)
                grouped.setdefault(key, []).append(
                    (index, segment, input_path, model_path, index_path, effective_profile, candidate_path)
                )

            next_pending: list[tuple[int, Segment, Path, Path, Path | None]] = []
            batch_tasks: list[list[tuple[int, Segment, Path, Path, Path | None, RVCProfile, Path]]] = []
            for entries in grouped.values():
                batch_tasks.extend(batch_chunks(entries, cfg.rvc_batch_size))

            def run_batch(
                entries: list[tuple[int, Segment, Path, Path, Path | None, RVCProfile, Path]]
            ) -> tuple[
                list[tuple[int, Segment, Path, Path, Path | None, RVCProfile, Path]],
                dict[str, RVCCommandResult],
            ]:
                nonlocal batch_counter
                with batch_counter_lock:
                    batch_counter += 1
                    batch_id = batch_counter
                first = entries[0]
                _, _, _, model_path, index_path, effective_profile, _ = first
                jobs = [
                    RVCBatchJob(
                        segment_id=segment.id,
                        input_path=input_path,
                        output_path=candidate_path,
                        model_path=model_path,
                        index_path=index_path,
                        profile=effective_profile,
                        sid=segment.speaker_id or "",
                    )
                    for _, segment, input_path, model_path, index_path, effective_profile, candidate_path in entries
                ]
                jobs_path = batches_dir / f"batch_{batch_id:04d}_{effective_profile.name}_jobs.jsonl"
                results_path = batches_dir / f"batch_{batch_id:04d}_{effective_profile.name}_results.jsonl"
                console.print(
                    f"[cyan]rvc batch[/cyan] profile={escape(effective_profile.name)} "
                    f"jobs={len(jobs)} model={escape(str(model_path))}"
                )
                return entries, batch_client.convert_many(
                    jobs,
                    jobs_path=jobs_path,
                    results_path=results_path,
                    model_path=model_path,
                    index_path=index_path,
                    cfg=cfg,
                    profile=effective_profile,
                    force=force,
                )

            if batch_lane_count > 1 and len(batch_tasks) > 1:
                with ThreadPoolExecutor(max_workers=batch_lane_count) as executor:
                    futures = [executor.submit(run_batch, entries) for entries in batch_tasks]
                    for future in as_completed(futures):
                        try:
                            entries, results = future.result()
                        except Exception as exc:
                            entries = batch_tasks[futures.index(future)]
                            next_pending.extend(apply_batch_result(entries, None, exc))
                        else:
                            next_pending.extend(apply_batch_result(entries, results))
                        save_manifest(project_dir, manifest)
            else:
                for entries in batch_tasks:
                    try:
                        entries, results = run_batch(entries)
                    except Exception as exc:
                        next_pending.extend(apply_batch_result(entries, None, exc))
                    else:
                        next_pending.extend(apply_batch_result(entries, results))
                    save_manifest(project_dir, manifest)
            pending = next_pending

        for index, segment, input_path, model_path, index_path in pending:
            error = "All RVC candidates failed or were rejected."
            attempts = attempts_by_segment[segment.id]
            if cfg.rvc_allow_pre_rvc_fallback and not _rvc_downstream_required(cfg):
                segment.rvc = RVCMetadata(
                    backend=backend,
                    input_path=str(input_path),
                    output_path=str(input_path),
                    selected_profile_name=None,
                    candidate_paths=candidate_paths_by_segment[segment.id],
                    model_path=str(model_path),
                    index_path=str(index_path) if index_path else None,
                    accepted=False,
                    fallback_used=True,
                    fallback_reason=error,
                    error=error,
                    attempts=attempts,
                )
            else:
                segment.rvc = RVCMetadata(
                    backend=backend,
                    input_path=str(input_path),
                    output_path=None,
                    selected_profile_name=None,
                    candidate_paths=candidate_paths_by_segment[segment.id],
                    model_path=str(model_path),
                    index_path=str(index_path) if index_path else None,
                    accepted=False,
                    error=error,
                    attempts=attempts,
                )
                segment.status = "failed"
                segment.errors.append(error)
                failed_segments.append(segment.id)
            last_logged_at = _log_segment_progress("rvc", index, total, segment, manifest, started_at, last_logged_at)
        if pending:
            save_manifest(project_dir, manifest)

    indexed_segments = [
        (index, segment)
        for index, segment in enumerate(manifest.segments, start=1)
        if only_segment_ids is None or segment.id in only_segment_ids
    ]
    skipped_completed = sum(1 for _, segment in indexed_segments if rvc_output_exists(segment))
    segment_jobs = [
        (index, segment)
        for index, segment in indexed_segments
        if segment.status not in SKIP_STATUSES and not rvc_output_exists(segment)
    ]
    if skipped_completed:
        console.print(f"[dim]rvc skipped {skipped_completed} already converted segment(s)[/dim]")
    if use_batch_rvc and len(segment_jobs) > 1:
        convert_segments_batched(segment_jobs)
    elif backend == "command" and rvc_lane_count > 1 and len(segment_jobs) > 1:
        with ThreadPoolExecutor(max_workers=rvc_lane_count) as executor:
            futures = [executor.submit(convert_segment, index, segment) for index, segment in segment_jobs]
            for future in as_completed(futures):
                index, segment, failed_segment_id = future.result()
                if failed_segment_id:
                    failed_segments.append(failed_segment_id)
                last_logged_at = _log_segment_progress("rvc", index, total, segment, manifest, started_at, last_logged_at)
                save_manifest(project_dir, manifest)
    else:
        for index, segment in segment_jobs:
            index, segment, failed_segment_id = convert_segment(index, segment)
            if failed_segment_id:
                failed_segments.append(failed_segment_id)
            last_logged_at = _log_segment_progress("rvc", index, total, segment, manifest, started_at, last_logged_at)
            save_manifest(project_dir, manifest)

    out_path = project_dir / "work" / "rvc" / "rvc_manifest.json"
    write_json_atomic(
        out_path,
        {
            "backend": backend,
            "execution_mode": "batch" if use_batch_rvc else "per_segment",
            "segments": [
                {"id": segment.id, "rvc": segment.rvc.model_dump(mode="json") if segment.rvc else None}
                for segment in manifest.segments
            ],
        },
    )
    manifest.artifacts["rvc_manifest"] = str(out_path)
    effective_concurrency = batch_lane_count if use_batch_rvc else rvc_lane_count
    if failed_segments:
        mark_stage(
            manifest,
            "rvc",
            "failed",
            backend=backend,
            failed_segments=failed_segments,
            rvc_manifest=str(out_path),
            concurrency=effective_concurrency,
            execution_mode="batch" if use_batch_rvc else "per_segment",
            segment_counts=_segment_counts(manifest),
        )
        save_manifest(project_dir, manifest)
        raise RVCCommandError(
            "RVC conversion failed for segments: "
            + ", ".join(failed_segments[:20])
            + (" ..." if len(failed_segments) > 20 else "")
        )
    mark_stage(
        manifest,
        "rvc",
        "completed",
        backend=backend,
        rvc_manifest=str(out_path),
        concurrency=effective_concurrency,
        execution_mode="batch" if use_batch_rvc else "per_segment",
        segment_counts=_segment_counts(manifest),
    )
    save_manifest(project_dir, manifest)
    _log_stage_complete("rvc", manifest, f"backend={backend}")
    return ctx.update_manifest(manifest)
