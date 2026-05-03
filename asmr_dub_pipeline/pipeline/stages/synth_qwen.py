from __future__ import annotations

# ruff: noqa: F403,F405,I001

from asmr_dub_pipeline.pipeline.context import PipelineContext
from asmr_dub_pipeline.pipeline.stages.common import *


def run_synth_qwen_stage(ctx: PipelineContext, refs_path: Path, confirm_rights: bool = False, *, model_id: str | None = None, candidate_count: int | None = None, candidate_batch_size: int | None = None, segment_batch_size: int | None = None, target_vram_gb: float | None = None, promote: bool = False, local_files_only: bool | None = None, only_segment_ids: set[str] | None = None) -> PipelineManifest:
    project_dir = ctx.project_dir
    manifest = ctx.reload_manifest()
    _load_config_into_manifest(project_dir, manifest)
    cfg = manifest.project_config
    total = len(manifest.segments)
    effective_model_id = model_id or cfg.qwen_tts_model_id
    effective_candidate_count = candidate_count or cfg.qwen_tts_candidate_count
    effective_segment_batch_size = segment_batch_size or cfg.qwen_tts_segment_batch_size
    effective_candidate_batch_size = min(
        effective_candidate_count,
        candidate_batch_size or cfg.qwen_tts_candidate_batch_size,
    )
    effective_target_vram_gb = cfg.qwen_tts_target_vram_gb if target_vram_gb is None else target_vram_gb
    effective_local_files_only = cfg.qwen_tts_local_files_only if local_files_only is None else local_files_only
    _log_stage_start(
        "synth-qwen",
        f"model={effective_model_id}, segments={total}, candidates={effective_candidate_count}, "
        f"segment_batch_size={effective_segment_batch_size}, "
        f"candidate_batch_size={effective_candidate_batch_size}, target_vram_gb={effective_target_vram_gb}, "
        f"promote={promote}",
    )
    refs = load_refs(refs_path, project_dir=project_dir)
    actual_refs_path = resolve_refs_json_path(refs_path, project_dir)
    refs_metadata = _refs_audit_metadata(actual_refs_path, refs)
    manifest.rights_audit = require_existing_or_confirmed_rights(
        manifest.rights_audit,
        confirm_rights,
        "synth-qwen",
        _manifest_source_path(manifest),
        metadata={"backend": "qwen-tts", "model_id": effective_model_id, **refs_metadata},
    )
    use_speaker_refs = bool(cfg.gsv_speaker_models)
    if use_speaker_refs:
        _validate_gsv_speaker_models(project_dir, manifest)
    client = QwenTTSClient(
        model_id=effective_model_id,
        device_map=cfg.qwen_tts_device_map,
        dtype=cfg.qwen_tts_dtype,
        attn_implementation=cfg.qwen_tts_attn_implementation,
        local_files_only=effective_local_files_only,
        target_vram_gb=effective_target_vram_gb,
    )
    console.print(
        f"[cyan]synth-qwen model[/cyan] loading "
        f"device_map={cfg.qwen_tts_device_map} dtype={cfg.qwen_tts_dtype} "
        f"attn={cfg.qwen_tts_attn_implementation} local_files_only={effective_local_files_only}"
    )
    load_model = getattr(client, "load_model", None)
    if callable(load_model):
        load_model()
    memory_snapshot = client.cuda_memory_snapshot() if hasattr(client, "cuda_memory_snapshot") else None
    console.print(f"[dim]synth-qwen cuda after load: {_format_cuda_memory_snapshot(memory_snapshot)}[/dim]")
    source_language = _canonical_language(cfg.source_language)
    target_language = _canonical_language(cfg.target_language)
    started_at = monotonic()
    last_logged_at = started_at
    failed_segments: list[str] = []
    promoted_segments: list[str] = []
    speaker_refs_cache: dict[str, dict[str, GPTSoVITSRef]] = {}

    synthesis_jobs: list[_QwenSegmentSynthesisJob] = []
    for index, segment in enumerate(manifest.segments, start=1):
        if only_segment_ids is not None and segment.id not in only_segment_ids:
            continue
        if not segment.script:
            payload = {
                "backend": "qwen-tts",
                "model_id": effective_model_id,
                "error": "Cannot synthesize without script metadata.",
            }
            segment.analysis["qwen_tts"] = payload
            if promote:
                segment.status = "needs_manual_review"
                segment.errors.append(payload["error"])
                failed_segments.append(segment.id)
            last_logged_at = _log_segment_progress(
                "synth-qwen",
                index,
                total,
                segment,
                manifest,
                started_at,
                last_logged_at,
            )
            continue
        if target_language == "ko":
            preflight = preflight_tts_text(
                segment.script,
                target_language=target_language,
                source_text=segment.source_script.text if segment.source_script else "",
                min_hangul_ratio=cfg.gsv_ko_text_min_hangul_ratio,
            )
            segment.analysis["pre_synth_qwen_text_qc"] = preflight.as_payload()
            if preflight.blocked:
                payload = {
                    "backend": "qwen-tts",
                    "model_id": effective_model_id,
                    "error": "Korean TTS preflight blocked synthesis: " + ", ".join(preflight.issues),
                    "preflight": preflight.as_payload(),
                }
                segment.analysis["qwen_tts"] = payload
                if promote:
                    segment.status = "needs_manual_review"
                    segment.errors.append(payload["error"])
                    failed_segments.append(segment.id)
                last_logged_at = _log_segment_progress(
                    "synth-qwen",
                    index,
                    total,
                    segment,
                    manifest,
                    started_at,
                    last_logged_at,
                )
                continue

        segment_refs = refs
        requested_ref_style = segment.script.ref_style
        resolved_ref_style = requested_ref_style if requested_ref_style in segment_refs else "whisper_close"
        speaker_refs_path: Path | None = None
        if use_speaker_refs and segment.speaker_id:
            speaker_cfg = _gsv_speaker_cfg(cfg, segment)
            if speaker_cfg is not None:
                speaker_refs_path = _resolve_gsv_speaker_path(project_dir, speaker_cfg.refs_path)
                cache_key = str(speaker_refs_path)
                if cache_key not in speaker_refs_cache:
                    speaker_refs_cache[cache_key] = load_refs(speaker_refs_path, project_dir)
                segment_refs = speaker_refs_cache[cache_key]
                if requested_ref_style not in segment_refs:
                    requested_ref_style = speaker_cfg.default_ref_style
                resolved_ref_style = requested_ref_style if requested_ref_style in segment_refs else "whisper_close"
        ref = resolve_ref(segment_refs, requested_ref_style)
        synthesis_jobs.append(
            _QwenSegmentSynthesisJob(
                index=index,
                segment=segment,
                ref=ref,
                resolved_ref_style=resolved_ref_style,
                speaker_refs_path=speaker_refs_path,
                candidates=[],
            )
        )

    generation_kwargs = {
        "temperature": cfg.qwen_tts_temperature,
        "top_p": cfg.qwen_tts_top_p,
        "max_new_tokens": cfg.qwen_tts_max_new_tokens,
    }

    def make_qwen_candidate_job(
        job: _QwenSegmentSynthesisJob,
        candidate_index: int,
    ) -> tuple[_QwenSegmentSynthesisJob, int, int, Path, QwenTTSRequest, dict[str, Any]]:
        segment = job.segment
        seed = cfg.base_seed + job.index * 100 + candidate_index
        candidate_path = _qwen_tts_candidate_path(project_dir, segment.id, candidate_index)
        tts_text_language = _segment_tts_text_language(segment, target_language)
        request = QwenTTSRequest(
            text=segment.script.tts_text,
            language=qwen_language(tts_text_language),
            ref_audio_path=job.ref.ref_audio_path,
            ref_text=job.ref.prompt_text,
            seed=seed,
            x_vector_only_mode=cfg.qwen_tts_x_vector_only_mode,
            generation_kwargs=generation_kwargs,
        )
        payload: dict[str, Any] = {
            "backend": "qwen-tts",
            "model_id": effective_model_id,
            "speaker_id": segment.speaker_id,
            "requested_ref_style": segment.script.ref_style,
            "resolved_ref_style": job.resolved_ref_style,
            "fallback_used": job.resolved_ref_style != segment.script.ref_style,
            "speaker_refs_path": str(job.speaker_refs_path) if job.speaker_refs_path else None,
            "source_language": source_language,
            "target_language": target_language,
            "cross_lingual_voice_transfer": source_language != target_language,
            "target_duration_sec": segment.duration,
            "prompt_lang": job.ref.prompt_lang,
            **request.as_payload(),
        }
        return job, candidate_index, seed, candidate_path, request, payload

    def record_qwen_failure(
        job: _QwenSegmentSynthesisJob,
        candidate_index: int,
        seed: int,
        candidate_path: Path,
        payload: dict[str, Any],
        exc: QwenTTSError,
    ) -> None:
        job.candidates.append(
            TTSCandidate(
                candidate_index=candidate_index,
                seed=seed,
                payload=payload,
                output_path=str(candidate_path),
                backend="qwen-tts",
                error=str(exc),
            )
        )

    def record_qwen_success(
        job: _QwenSegmentSynthesisJob,
        candidate_index: int,
        seed: int,
        candidate_path: Path,
        payload: dict[str, Any],
        result: Any,
    ) -> None:
        segment = job.segment
        if cfg.gsv_trim_edge_silence:
            trim = trim_edge_silence(
                candidate_path,
                threshold_db=cfg.gsv_trim_silence_threshold_db,
                keep_sec=cfg.gsv_trim_silence_keep_sec,
            )
            payload.setdefault("postprocess", {})["edge_silence_trim"] = trim
        duration = duration_sec(candidate_path)
        payload["sample_rate"] = result.sample_rate
        payload["batch_size"] = getattr(result, "batch_size", 1)
        payload["batch_seed"] = getattr(result, "batch_seed", seed)
        too_long = duration_too_long(duration, segment.duration, cfg.duration_tolerance)
        too_short = duration_too_short(duration, segment.duration, cfg.duration_tolerance)
        candidate_ratio = duration_ratio(duration, segment.duration)
        duration_gate = "too_long" if too_long else "too_short" if too_short else "pass"
        language_contract_ok = payload["text"] == segment.script.tts_text
        if target_language == "ko":
            language_contract_ok = language_contract_ok and payload["language"] == "Korean"
        acceptable_for_mix = duration_gate == "pass" and language_contract_ok
        payload["duration_ratio"] = candidate_ratio
        payload["duration_gate"] = duration_gate
        job.candidates.append(
            TTSCandidate(
                candidate_index=candidate_index,
                seed=seed,
                payload=payload,
                output_path=str(candidate_path),
                duration_sec=duration,
                backend="qwen-tts",
                duration_ratio=candidate_ratio,
                duration_gate=duration_gate,
                acceptable_for_mix=acceptable_for_mix,
                selection_score=max(0.0, 1.0 - min(abs(candidate_ratio - 1.0), 1.0)),
                selection_reason=(
                    "duration_and_language_contract_pass"
                    if acceptable_for_mix
                    else "duration_or_language_contract_failed"
                ),
            )
        )

    def run_qwen_request_batch(
        request_batch: list[tuple[_QwenSegmentSynthesisJob, int, int, Path, QwenTTSRequest, dict[str, Any]]],
    ) -> None:
        try:
            batch_synthesize = getattr(client, "synthesize_many_to_files", None)
            if callable(batch_synthesize) and len(request_batch) > 1:
                results = batch_synthesize(
                    [request for _, _, _, _, request, _ in request_batch],
                    [candidate_path for _, _, _, candidate_path, _, _ in request_batch],
                )
            else:
                results = [
                    client.synthesize_to_file(request, candidate_path)
                    for _, _, _, candidate_path, request, _ in request_batch
                ]
        except QwenTTSError as exc:
            if len(request_batch) == 1:
                job, candidate_index, seed, candidate_path, _, payload = request_batch[0]
                record_qwen_failure(job, candidate_index, seed, candidate_path, payload, exc)
                return
            segment_ids = ",".join(job.segment.id for job, *_ in request_batch[:8])
            console.print(
                f"[yellow]synth-qwen batch failed for segments={escape(segment_ids)}; "
                f"retrying one by one: {escape(str(exc))}[/yellow]"
            )
            for item in request_batch:
                run_qwen_request_batch([item])
            return
        for (job, candidate_index, seed, candidate_path, _, payload), result in zip(
            request_batch, results, strict=True
        ):
            record_qwen_success(job, candidate_index, seed, candidate_path, payload, result)

    def finalize_qwen_segment(job: _QwenSegmentSynthesisJob) -> Path | None:
        segment = job.segment
        successful = [
            candidate for candidate in job.candidates if not candidate.error and candidate.duration_sec is not None
        ]
        acceptable = [candidate for candidate in successful if candidate.acceptable_for_mix]
        selected = (
            min(acceptable or successful, key=lambda c: abs((c.duration_sec or 0.0) - segment.duration))
            if successful
            else None
        )
        selected_path: Path | None = None
        if selected is None:
            failed_segments.append(segment.id)
            summary = {
                "backend": "qwen-tts",
                "model_id": effective_model_id,
                "candidate_count": effective_candidate_count,
                "candidate_batch_size": effective_candidate_batch_size,
                "segment_batch_size": effective_segment_batch_size,
                "target_vram_gb": effective_target_vram_gb,
                "selected_candidate_path": None,
                "candidates": [candidate.model_dump(mode="json") for candidate in job.candidates],
                "error": "All Qwen TTS candidates failed.",
            }
            segment.analysis["qwen_tts"] = summary
            if promote:
                segment.status = "failed"
                segment.errors.append("All Qwen TTS candidates failed.")
            return None

        selected.selected = True
        selected_path = _qwen_tts_best_path(project_dir, segment.id)
        ensure_not_same_path(Path(selected.output_path), selected_path)
        selected_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(selected.output_path, selected_path)
        summary = {
            "backend": "qwen-tts",
            "model_id": effective_model_id,
            "candidate_count": effective_candidate_count,
            "candidate_batch_size": effective_candidate_batch_size,
            "segment_batch_size": effective_segment_batch_size,
            "target_vram_gb": effective_target_vram_gb,
            "selected_candidate_path": str(selected_path),
            "selected_duration_gate": selected.duration_gate,
            "selected_acceptable_for_mix": selected.acceptable_for_mix,
            "selected_duration_ratio": selected.duration_ratio,
            "candidates": [candidate.model_dump(mode="json") for candidate in job.candidates],
        }
        segment.analysis["qwen_tts"] = summary
        if promote:
            final_path = project_dir / "work" / "tts" / f"{segment.id}_final.wav"
            ensure_not_same_path(Path(selected.output_path), final_path)
            final_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(selected.output_path, final_path)
            segment.tts = TTSMetadata(
                backend="qwen-tts",
                ref_style=job.resolved_ref_style,
                speed_factor=1.0,
                candidate_count=effective_candidate_count,
                selected_candidate_path=str(final_path),
                candidates=job.candidates,
                source_language=source_language,
                target_language=target_language,
                cross_lingual_voice_transfer=source_language != target_language,
                retry_summary={
                    "selected_duration_gate": selected.duration_gate,
                    "selected_acceptable_for_mix": selected.acceptable_for_mix,
                    "selected_duration_ratio": selected.duration_ratio,
                },
            )
            segment.rvc = None
            segment.qc = None
            segment.mix = {}
            segment.status = "synthesized"
            promoted_segments.append(segment.id)
        return selected_path

    for segment_batch in _chunked(synthesis_jobs, effective_segment_batch_size):
        first_index = segment_batch[0].index
        last_index = segment_batch[-1].index
        batch_candidate_size = 1 if len(segment_batch) > 1 else effective_candidate_batch_size
        for candidate_indexes in _chunked(list(range(effective_candidate_count)), batch_candidate_size):
            request_batch = [
                make_qwen_candidate_job(job, candidate_index)
                for candidate_index in candidate_indexes
                for job in segment_batch
            ]
            console.print(
                f"[cyan]synth-qwen batch[/cyan] segments={first_index}-{last_index}/{total} "
                f"candidates={candidate_indexes[0] + 1}-{candidate_indexes[-1] + 1}/{effective_candidate_count} "
                f"batch_size={len(request_batch)}"
            )
            run_qwen_request_batch(request_batch)
            if first_index == 1 or last_index == total or first_index % _progress_interval(total) == 0:
                memory_snapshot = client.cuda_memory_snapshot() if hasattr(client, "cuda_memory_snapshot") else None
                console.print(f"[dim]synth-qwen cuda: {_format_cuda_memory_snapshot(memory_snapshot)}[/dim]")

        for job in segment_batch:
            selected_path = finalize_qwen_segment(job)
            save_manifest(project_dir, manifest)
            last_logged_at = _log_segment_progress(
                "synth-qwen",
                job.index,
                total,
                job.segment,
                manifest,
                started_at,
                last_logged_at,
                note=f"selected={selected_path}" if selected_path else None,
            )

    if promoted_segments:
        _invalidate_downstream_after_tts_promotion(manifest)
    out_path = project_dir / "work" / "tts" / "qwen" / "qwen_tts_manifest.json"
    write_json_atomic(
        out_path,
        {
            "backend": "qwen-tts",
            "model_id": effective_model_id,
            "promote": promote,
            "candidate_batch_size": effective_candidate_batch_size,
            "segment_batch_size": effective_segment_batch_size,
            "target_vram_gb": effective_target_vram_gb,
            "segments": [
                {
                    "id": segment.id,
                    "qwen_tts": segment.analysis.get("qwen_tts"),
                }
                for segment in manifest.segments
            ],
        },
    )
    manifest.artifacts["qwen_tts"] = str(out_path)
    status = "failed" if promote and failed_segments else "completed"
    mark_stage(
        manifest,
        "synth-qwen",
        status,
        backend="qwen-tts",
        model_id=effective_model_id,
        candidate_count=effective_candidate_count,
        candidate_batch_size=effective_candidate_batch_size,
        segment_batch_size=effective_segment_batch_size,
        target_vram_gb=effective_target_vram_gb,
        promote=promote,
        promoted_segments=promoted_segments,
        failed_segments=failed_segments,
        qwen_tts_manifest=str(out_path),
        segment_counts=_segment_counts(manifest),
    )
    save_manifest(project_dir, manifest)
    _log_stage_complete("synth-qwen", manifest, f"backend=qwen-tts promote={promote}")
    if promote and failed_segments:
        raise QwenTTSError(
            "Qwen TTS synthesis failed for segments: "
            + ", ".join(failed_segments[:20])
            + (" ..." if len(failed_segments) > 20 else "")
        )
    return ctx.update_manifest(manifest)
