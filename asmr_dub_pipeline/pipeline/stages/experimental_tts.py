from __future__ import annotations

# ruff: noqa: F403,F405,I001

from asmr_dub_pipeline.pipeline.context import PipelineContext
from asmr_dub_pipeline.pipeline.stages.common import *


def run_synth_experimental_tts_stage(ctx: PipelineContext, refs_path: Path, *, backend: str, confirm_rights: bool = False, base_url: str | None = None, candidate_count: int | None = None, promote: bool = False, only_segment_ids: set[str] | None = None) -> PipelineManifest:
    project_dir = ctx.project_dir
    spec = _experimental_tts_backend_spec(backend)
    manifest = ctx.reload_manifest()
    _load_config_into_manifest(project_dir, manifest)
    cfg = manifest.project_config
    total = len(manifest.segments)
    effective_candidate_count = candidate_count or (
        cfg.fish_tts_candidate_count if spec.backend_name == "fish-tts" else cfg.cosyvoice_candidate_count
    )
    effective_base_url = base_url or (
        cfg.fish_tts_base_url if spec.backend_name == "fish-tts" else cfg.cosyvoice_base_url
    )
    _log_stage_start(
        spec.stage,
        f"backend={spec.backend_name}, base_url={effective_base_url}, "
        f"segments={total}, candidates={effective_candidate_count}, promote={promote}",
    )
    refs = load_refs(refs_path, project_dir=project_dir)
    actual_refs_path = resolve_refs_json_path(refs_path, project_dir)
    refs_metadata = _refs_audit_metadata(actual_refs_path, refs)
    manifest.rights_audit = require_existing_or_confirmed_rights(
        manifest.rights_audit,
        confirm_rights,
        spec.stage,
        _manifest_source_path(manifest),
        metadata={"backend": spec.backend_name, "base_url": effective_base_url, **refs_metadata},
    )
    use_speaker_refs = bool(cfg.gsv_speaker_models)
    if use_speaker_refs:
        _validate_gsv_speaker_models(project_dir, manifest)

    if spec.backend_name == "fish-tts":
        client = FishSpeechTTSClient(base_url=effective_base_url, timeout_sec=cfg.fish_tts_timeout_sec)
        generation_kwargs: dict[str, Any] = {
            "chunk_length": cfg.fish_tts_chunk_length,
            "temperature": cfg.fish_tts_temperature,
            "top_p": cfg.fish_tts_top_p,
            "repetition_penalty": cfg.fish_tts_repetition_penalty,
            "max_new_tokens": cfg.fish_tts_max_new_tokens,
            "normalize": cfg.fish_tts_normalize,
            "latency": cfg.fish_tts_latency,
        }
    else:
        client = CosyVoiceTTSClient(
            base_url=effective_base_url,
            mode=cfg.cosyvoice_mode,
            sample_rate=cfg.cosyvoice_sample_rate,
            timeout_sec=cfg.cosyvoice_timeout_sec,
            instruct_text=cfg.cosyvoice_instruct_text,
        )
        generation_kwargs = {}
        if cfg.cosyvoice_instruct_text:
            generation_kwargs["instruct_text"] = cfg.cosyvoice_instruct_text

    load_model = getattr(client, "load_model", None)
    if callable(load_model):
        load_model()

    source_language = _canonical_language(cfg.source_language)
    target_language = _canonical_language(cfg.target_language)
    started_at = monotonic()
    last_logged_at = started_at
    failed_segments: list[str] = []
    promoted_segments: list[str] = []
    speaker_refs_cache: dict[str, dict[str, GPTSoVITSRef]] = {}
    synthesis_jobs: list[_ExperimentalTTSSegmentSynthesisJob] = []

    for index, segment in enumerate(manifest.segments, start=1):
        if only_segment_ids is not None and segment.id not in only_segment_ids:
            continue
        if not segment.script:
            payload = {
                "backend": spec.backend_name,
                "base_url": effective_base_url,
                "error": "Cannot synthesize without script metadata.",
            }
            segment.analysis[spec.analysis_key] = payload
            if promote:
                segment.status = "needs_manual_review"
                segment.errors.append(payload["error"])
                failed_segments.append(segment.id)
            last_logged_at = _log_segment_progress(
                spec.stage,
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
            segment.analysis[f"pre_{spec.stage.replace('-', '_')}_text_qc"] = preflight.as_payload()
            if preflight.blocked:
                payload = {
                    "backend": spec.backend_name,
                    "base_url": effective_base_url,
                    "error": "Korean TTS preflight blocked synthesis: " + ", ".join(preflight.issues),
                    "preflight": preflight.as_payload(),
                }
                segment.analysis[spec.analysis_key] = payload
                if promote:
                    segment.status = "needs_manual_review"
                    segment.errors.append(payload["error"])
                    failed_segments.append(segment.id)
                last_logged_at = _log_segment_progress(
                    spec.stage,
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
            _ExperimentalTTSSegmentSynthesisJob(
                index=index,
                segment=segment,
                ref=ref,
                resolved_ref_style=resolved_ref_style,
                speaker_refs_path=speaker_refs_path,
                candidates=[],
            )
        )

    def make_candidate_job(
        job: _ExperimentalTTSSegmentSynthesisJob,
        candidate_index: int,
    ) -> tuple[_ExperimentalTTSSegmentSynthesisJob, int, int, Path, ExperimentalTTSRequest, dict[str, Any]]:
        segment = job.segment
        seed = cfg.base_seed + job.index * 100 + candidate_index
        candidate_path = _experimental_tts_candidate_path(project_dir, spec, segment.id, candidate_index)
        tts_text_language = _segment_tts_text_language(segment, target_language)
        ref_audio_path = Path(job.ref.ref_audio_path).expanduser()
        if not ref_audio_path.is_absolute():
            ref_audio_path = project_dir / ref_audio_path
        request = ExperimentalTTSRequest(
            text=segment.script.tts_text,
            language=tts_text_language,
            ref_audio_path=str(ref_audio_path),
            ref_text=job.ref.prompt_text,
            seed=seed,
            generation_kwargs=generation_kwargs,
        )
        payload: dict[str, Any] = {
            "backend": spec.backend_name,
            "base_url": effective_base_url,
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

    def record_failure(
        job: _ExperimentalTTSSegmentSynthesisJob,
        candidate_index: int,
        seed: int,
        candidate_path: Path,
        payload: dict[str, Any],
        exc: ExperimentalTTSError,
    ) -> None:
        job.candidates.append(
            TTSCandidate(
                candidate_index=candidate_index,
                seed=seed,
                payload=payload,
                output_path=str(candidate_path),
                backend=spec.backend_name,
                error=str(exc),
            )
        )

    def record_success(
        job: _ExperimentalTTSSegmentSynthesisJob,
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
        too_long = duration_too_long(duration, segment.duration, cfg.duration_tolerance)
        too_short = duration_too_short(duration, segment.duration, cfg.duration_tolerance)
        candidate_ratio = duration_ratio(duration, segment.duration)
        duration_gate = "too_long" if too_long else "too_short" if too_short else "pass"
        language_contract_ok = payload["text"] == segment.script.tts_text
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
                backend=spec.backend_name,
                duration_ratio=candidate_ratio,
                duration_gate=duration_gate,
                acceptable_for_mix=acceptable_for_mix,
                selection_score=max(0.0, 1.0 - min(abs(candidate_ratio - 1.0), 1.0)),
                selection_reason=(
                    "duration_and_text_contract_pass"
                    if acceptable_for_mix
                    else "duration_or_text_contract_failed"
                ),
            )
        )

    def finalize_segment(job: _ExperimentalTTSSegmentSynthesisJob) -> Path | None:
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
        if selected is None:
            failed_segments.append(segment.id)
            summary = {
                "backend": spec.backend_name,
                "base_url": effective_base_url,
                "candidate_count": effective_candidate_count,
                "selected_candidate_path": None,
                "candidates": [candidate.model_dump(mode="json") for candidate in job.candidates],
                "error": f"All {spec.backend_name} candidates failed.",
            }
            segment.analysis[spec.analysis_key] = summary
            if promote:
                segment.status = "failed"
                segment.errors.append(f"All {spec.backend_name} candidates failed.")
            return None

        selected.selected = True
        selected_path = _experimental_tts_best_path(project_dir, spec, segment.id)
        ensure_not_same_path(Path(selected.output_path), selected_path)
        selected_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(selected.output_path, selected_path)
        summary = {
            "backend": spec.backend_name,
            "base_url": effective_base_url,
            "candidate_count": effective_candidate_count,
            "selected_candidate_path": str(selected_path),
            "selected_duration_gate": selected.duration_gate,
            "selected_acceptable_for_mix": selected.acceptable_for_mix,
            "selected_duration_ratio": selected.duration_ratio,
            "candidates": [candidate.model_dump(mode="json") for candidate in job.candidates],
        }
        segment.analysis[spec.analysis_key] = summary
        if promote:
            final_path = project_dir / "work" / "tts" / f"{segment.id}_final.wav"
            ensure_not_same_path(Path(selected.output_path), final_path)
            final_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(selected.output_path, final_path)
            segment.tts = TTSMetadata(
                backend=spec.backend_name,
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

    for job in synthesis_jobs:
        for candidate_index in range(effective_candidate_count):
            item = make_candidate_job(job, candidate_index)
            _, _, seed, candidate_path, request, payload = item
            try:
                result = client.synthesize_to_file(request, candidate_path)
            except ExperimentalTTSError as exc:
                record_failure(job, candidate_index, seed, candidate_path, payload, exc)
                continue
            record_success(job, candidate_index, seed, candidate_path, payload, result)
        selected_path = finalize_segment(job)
        save_manifest(project_dir, manifest)
        last_logged_at = _log_segment_progress(
            spec.stage,
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
    out_path = project_dir / "work" / "tts" / spec.work_dir_name / f"{spec.analysis_key}_manifest.json"
    write_json_atomic(
        out_path,
        {
            "backend": spec.backend_name,
            "base_url": effective_base_url,
            "promote": promote,
            "candidate_count": effective_candidate_count,
            "segments": [
                {
                    "id": segment.id,
                    spec.analysis_key: segment.analysis.get(spec.analysis_key),
                }
                for segment in manifest.segments
            ],
        },
    )
    manifest.artifacts[spec.artifact_key] = str(out_path)
    status = "failed" if promote and failed_segments else "completed"
    mark_stage(
        manifest,
        spec.stage,
        status,
        backend=spec.backend_name,
        base_url=effective_base_url,
        candidate_count=effective_candidate_count,
        promote=promote,
        promoted_segments=promoted_segments,
        failed_segments=failed_segments,
        manifest_path=str(out_path),
        segment_counts=_segment_counts(manifest),
    )
    save_manifest(project_dir, manifest)
    _log_stage_complete(spec.stage, manifest, f"backend={spec.backend_name} promote={promote}")
    if promote and failed_segments:
        raise ExperimentalTTSError(
            f"{spec.backend_name} synthesis failed for segments: "
            + ", ".join(failed_segments[:20])
            + (" ..." if len(failed_segments) > 20 else "")
        )
    return ctx.update_manifest(manifest)
