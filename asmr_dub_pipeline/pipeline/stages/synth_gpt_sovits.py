from __future__ import annotations

# ruff: noqa: F403,F405,I001

from asmr_dub_pipeline.pipeline.context import PipelineContext
from asmr_dub_pipeline.pipeline.stages.common import *


def run_synth_stage(ctx: PipelineContext, gsv_url: str | None, refs_path: Path, mock: bool = False, confirm_rights: bool = False, gpt_weights_path: str | None = None, sovits_weights_path: str | None = None, auto_gsv_server: bool | None = None, gsv_server_command: list[str] | str | None = None, use_trained_gpt: bool = False, only_segment_ids: set[str] | None = None) -> PipelineManifest:
    project_dir = ctx.project_dir
    manifest = ctx.reload_manifest()
    _load_config_into_manifest(project_dir, manifest)
    cfg = manifest.project_config
    if use_trained_gpt:
        cfg = cfg.model_copy(update={"gsv_gpt_weights_policy": "few_shot"})
        manifest.project_config = cfg
    total = len(manifest.segments)
    synth_backend_name = "mock" if mock else "gpt-sovits"
    _log_stage_start("synth", f"backend={synth_backend_name}, segments={total}")
    if not mock and not confirm_rights:
        raise RightsError(
            "Real GPT-SoVITS synthesis requires --confirm-rights for the current source and voice references."
        )
    if mock:
        _require_audio_stage_rights(manifest, "synth", confirm_rights, metadata={"backend": "mock"})
    use_speaker_gsv = bool(cfg.gsv_speaker_models)
    if use_speaker_gsv:
        _validate_gsv_speaker_models(project_dir, manifest)
        refs: dict[str, GPTSoVITSRef] = {}
        refs_metadata: dict[str, object] = {
            "speaker_refs": {
                speaker_id: speaker_cfg.refs_path
                for speaker_id, speaker_cfg in sorted(cfg.gsv_speaker_models.items())
            }
        }
    else:
        refs = load_refs(refs_path, project_dir=project_dir)
        actual_refs_path = resolve_refs_json_path(refs_path, project_dir)
        refs_metadata = _refs_audit_metadata(actual_refs_path, refs)
    if not mock:
        manifest.rights_audit = require_existing_or_confirmed_rights(
            manifest.rights_audit,
            True,
            "synth",
            _manifest_source_path(manifest),
            metadata={"backend": "gpt-sovits", **refs_metadata},
        )
    effective_gsv_url = gsv_url or cfg.gsv_url
    should_auto_start_server = (
        False if mock else cfg.gsv_auto_start if auto_gsv_server is None else auto_gsv_server
    )
    gsv_lane_count = 1 if mock else _effective_lane_count(cfg.gsv_concurrency, total)
    gsv_base_urls = [effective_gsv_url] if mock else _parallel_base_urls(effective_gsv_url, gsv_lane_count)
    server_managers: list[ManagedGPTSoVITSServer] = []
    if not mock:
        for lane_index, base_url in enumerate(gsv_base_urls):
            log_name = "api_v2.log" if gsv_lane_count == 1 else f"api_v2_lane_{lane_index + 1:02d}.log"
            server_managers.append(
                ManagedGPTSoVITSServer(
                    enabled=should_auto_start_server,
                    base_url=base_url,
                    command=gsv_server_command if gsv_server_command is not None else cfg.gsv_server_command,
                    cwd=cfg.gsv_server_cwd,
                    log_path=project_dir / "work" / "gpt_sovits" / log_name,
                    startup_timeout_sec=cfg.gsv_server_startup_timeout_sec,
                    shutdown_timeout_sec=cfg.gsv_server_shutdown_timeout_sec,
                )
            )
    model_switch: dict[str, Any] = {}
    try:
        for server_manager in server_managers:
            server_manager.start()
        clients: list[GPTSoVITSClient] = []
        if not mock:
            clients = [
                GPTSoVITSClient(base_url, cfg.gsv_timeout_sec, cfg.gsv_retries)
                for base_url in gsv_base_urls
            ]
        _validate_gsv_speaker_models(project_dir, manifest)
        if clients:
            gpt_weights = None
            sovits_weights = None
            if use_speaker_gsv:
                model_switch["gpt_weights_mode"] = "speaker_voice_bank"
                model_switch["sovits_weights_mode"] = "speaker_voice_bank"
                model_switch["speaker_models"] = sorted(cfg.gsv_speaker_models)
            else:
                gpt_weights = _resolve_gpt_weights_for_tts(
                    project_dir,
                    manifest,
                    cfg,
                    gpt_weights_path,
                    model_switch,
                )
                sovits_weights = (
                    sovits_weights_path
                    or cfg.gsv_sovits_weights_path
                    or (
                        manifest.artifacts.get(FEW_SHOT_ARTIFACT_SOVITS)
                        if cfg.gsv_sovits_weights_policy != "unchanged"
                        else None
                    )
                )
            if gpt_weights:
                model_switch["gpt_weights_path"] = gpt_weights
            if sovits_weights:
                model_switch["sovits_weights_path"] = sovits_weights
                model_switch["sovits_weights_mode"] = (
                    "explicit"
                    if sovits_weights_path or cfg.gsv_sovits_weights_path
                    else "few_shot_source_voice"
                )
            model_switch["instances"] = []
            for lane_index, client in enumerate(clients):
                lane_switch: dict[str, Any] = {
                    "lane_index": lane_index,
                    "gsv_url": gsv_base_urls[lane_index],
                }
                if gpt_weights:
                    lane_switch["gpt_response"] = client.set_gpt_weights(gpt_weights)
                if sovits_weights:
                    lane_switch["sovits_response"] = client.set_sovits_weights(sovits_weights)
                model_switch["instances"].append(lane_switch)
            if len(model_switch["instances"]) == 1:
                instance = model_switch["instances"][0]
                if "gpt_response" in instance:
                    model_switch["gpt_response"] = instance["gpt_response"]
                if "sovits_response" in instance:
                    model_switch["sovits_response"] = instance["sovits_response"]
        started_at = monotonic()
        last_logged_at = started_at
        lane_locks = [Lock() for _ in range(gsv_lane_count)]
        lane_gpt_weights: list[str | None] = [None for _ in range(gsv_lane_count)]
        lane_sovits_weights: list[str | None] = [None for _ in range(gsv_lane_count)]
        speaker_refs_cache: dict[str, dict[str, GPTSoVITSRef]] = {}
        speaker_refs_cache_lock = Lock()

        def postprocess_tts_candidate(candidate_path: Path, payload: dict[str, Any]) -> None:
            if not cfg.gsv_trim_edge_silence:
                return
            trim = trim_edge_silence(
                candidate_path,
                threshold_db=cfg.gsv_trim_silence_threshold_db,
                keep_sec=cfg.gsv_trim_silence_keep_sec,
            )
            payload.setdefault("postprocess", {})["edge_silence_trim"] = trim

        def synthesize_segment_locked(
            index: int,
            segment: Segment,
            lane_index: int,
        ) -> tuple[int, Segment]:
            if segment.status in SKIP_STATUSES:
                return index, segment
            if not segment.script:
                segment.status = "needs_manual_review"
                segment.errors.append("Cannot synthesize without script metadata.")
                return index, segment
            target_language = _canonical_language(cfg.target_language)
            source_language = _canonical_language(cfg.source_language)
            if target_language == "ko":
                preflight = preflight_tts_text(
                    segment.script,
                    target_language=target_language,
                    source_text=segment.source_script.text if segment.source_script else "",
                    min_hangul_ratio=cfg.gsv_ko_text_min_hangul_ratio,
                )
                segment.analysis["pre_synth_text_qc"] = preflight.as_payload()
                if preflight.blocked:
                    segment.status = "needs_manual_review"
                    segment.errors.append(
                        "Korean TTS preflight blocked synthesis: " + ", ".join(preflight.issues)
                    )
                    return index, segment
            original_ref_style = segment.script.ref_style
            requested_ref_style = original_ref_style
            speaker_cfg = _gsv_speaker_cfg(cfg, segment)
            segment_refs = refs
            speaker_gpt_weights: str | None = None
            speaker_sovits_weights: str | None = None
            speaker_refs_path: Path | None = None
            if speaker_cfg is not None:
                if speaker_cfg.gpt_weights_path:
                    speaker_gpt_weights = str(
                        _resolve_gsv_speaker_path(project_dir, speaker_cfg.gpt_weights_path)
                    )
                speaker_sovits_weights = str(
                    _resolve_gsv_speaker_path(project_dir, speaker_cfg.sovits_weights_path)
                )
                speaker_refs_path = _resolve_gsv_speaker_path(project_dir, speaker_cfg.refs_path)
                cache_key = str(speaker_refs_path)
                with speaker_refs_cache_lock:
                    if cache_key not in speaker_refs_cache:
                        speaker_refs_cache[cache_key] = load_refs(speaker_refs_path, project_dir)
                    segment_refs = speaker_refs_cache[cache_key]
                if requested_ref_style not in segment_refs:
                    requested_ref_style = speaker_cfg.default_ref_style
            resolved_ref_style = requested_ref_style if requested_ref_style in segment_refs else "whisper_close"
            ref = resolve_ref(segment_refs, requested_ref_style)
            synthesis_ref = _ref_for_tts_language(ref, segment.script.tts_language)
            fallback_used = resolved_ref_style != original_ref_style
            candidates: list[TTSCandidate] = []
            expected = segment.script.expected_tts_duration_sec or segment.duration
            speed = suggest_speed_factor(
                expected,
                segment.duration,
                minimum=cfg.gsv_tts_min_speed_factor,
                maximum=cfg.gsv_tts_max_speed_factor,
            )
            has_repetition_or_omission_signal = bool(
                segment.qc and (segment.qc.repetition_detected or segment.qc.omission_detected)
            )
            can_rewrite_for_duration = _can_rewrite_script_for_duration(segment.script)
            for candidate_index in range(cfg.candidate_count):
                seed = cfg.base_seed + index * 100 + candidate_index
                tts_text_language = _segment_tts_text_language(segment, target_language)
                options = GPTSoVITSTTSOptions(
                    seed=seed,
                    speed_factor=speed,
                    text_lang=tts_text_language,
                    top_k=cfg.gsv_top_k,
                    top_p=cfg.gsv_top_p,
                    temperature=cfg.gsv_temperature,
                    text_split_method=cfg.gsv_text_split_method,
                    fragment_interval=cfg.gsv_fragment_interval,
                    parallel_infer=cfg.gsv_parallel_infer,
                    repetition_penalty=cfg.gsv_repetition_penalty,
                    sample_steps=cfg.gsv_sample_steps,
                    super_sampling=cfg.gsv_super_sampling,
                    overlap_length=cfg.gsv_overlap_length,
                    min_chunk_length=cfg.gsv_min_chunk_length,
                )
                attempt_signals: list[GPTSoVITSRetrySignal] = []
                if has_repetition_or_omission_signal:
                    options = adjust_for_repetition_or_omission(options, seed_step=10_000 + index)
                    attempt_signals.extend(
                        [
                            GPTSoVITSRetrySignal.REPETITION_OR_OMISSION,
                            GPTSoVITSRetrySignal.SEED_CHANGED,
                            GPTSoVITSRetrySignal.REPETITION_PENALTY_INCREASED,
                        ]
                    )
                attempt_text = segment.script.tts_text
                for attempt in range(3):
                    candidate_path = _tts_candidate_path(project_dir, segment.id, candidate_index, attempt)
                    payload: dict[str, Any] = {
                        "speaker_id": segment.speaker_id,
                        "requested_ref_style": original_ref_style,
                        "resolved_ref_style": resolved_ref_style,
                        "fallback_used": fallback_used,
                        "ref_audio_path": ref.ref_audio_path,
                        "aux_ref_audio_paths": ref.aux_ref_audio_paths,
                        "prompt_text_policy": "use_source_reference_prompt",
                        "speaker_gpt_weights_path": speaker_gpt_weights,
                        "speaker_sovits_weights_path": speaker_sovits_weights,
                        "speaker_refs_path": str(speaker_refs_path) if speaker_refs_path else None,
                        "source_language": source_language,
                        "target_language": target_language,
                        "cross_lingual_voice_transfer": source_language != target_language,
                        "expected_tts_duration_sec": expected,
                        "target_duration_sec": segment.duration,
                        "lane_index": lane_index,
                        "gsv_url": None if mock else gsv_base_urls[lane_index],
                        "retry": {
                            "attempt": attempt,
                            "max_attempts": 3,
                            "signals": retry_signal_values(attempt_signals),
                        },
                    }
                    payload.update(_tts_request_debug_payload(attempt_text, synthesis_ref, options))
                    if mock:
                        mock_duration = max(0.05, expected / max(options.speed_factor, 0.01))
                        _mock_synthesize(candidate_path, mock_duration, options.seed, cfg.mix_sample_rate)
                        postprocess_tts_candidate(candidate_path, payload)
                        duration = duration_sec(candidate_path)
                        payload.update(
                            {
                                "mock": True,
                                "repetition_penalty": options.repetition_penalty,
                            }
                        )
                        candidate_backend_name = "mock"
                    else:
                        client = clients[lane_index]
                        try:
                            speaker_switch: dict[str, Any] = {}
                            if speaker_gpt_weights and lane_gpt_weights[lane_index] != speaker_gpt_weights:
                                response = client.set_gpt_weights(speaker_gpt_weights)
                                lane_gpt_weights[lane_index] = speaker_gpt_weights
                                speaker_switch.update(
                                    {
                                        "lane_index": lane_index,
                                        "speaker_id": segment.speaker_id,
                                        "gpt_weights_path": speaker_gpt_weights,
                                        "gpt_response": response,
                                    }
                                )
                            if speaker_sovits_weights and lane_sovits_weights[lane_index] != speaker_sovits_weights:
                                response = client.set_sovits_weights(speaker_sovits_weights)
                                lane_sovits_weights[lane_index] = speaker_sovits_weights
                                speaker_switch.update(
                                    {
                                        "lane_index": lane_index,
                                        "speaker_id": segment.speaker_id,
                                        "sovits_weights_path": speaker_sovits_weights,
                                        "sovits_response": response,
                                    }
                                )
                            if speaker_switch:
                                model_switch.setdefault("speaker_switches", []).append(speaker_switch)
                            request = client.build_payload(attempt_text, synthesis_ref, options)
                            payload.update(request.as_payload())
                            client.synthesize_to_file(request, candidate_path)
                            postprocess_tts_candidate(candidate_path, payload)
                            duration = duration_sec(candidate_path)
                        except GPTSoVITSError as exc:
                            candidates.append(
                                TTSCandidate(
                                    candidate_index=candidate_index,
                                    seed=options.seed,
                                    payload=payload,
                                    output_path=str(candidate_path),
                                    backend="gpt-sovits",
                                    error=str(exc),
                                )
                            )
                            break
                        candidate_backend_name = "gpt-sovits"
                    too_long = duration_too_long(duration, segment.duration, cfg.duration_tolerance)
                    too_short = duration_too_short(duration, segment.duration, cfg.duration_tolerance)
                    candidate_ratio = duration_ratio(duration, segment.duration)
                    duration_gate = "too_long" if too_long else "too_short" if too_short else "pass"
                    payload["duration_ratio"] = candidate_ratio
                    payload["duration_gate"] = duration_gate
                    language_contract_ok = True
                    if target_language == "ko":
                        language_contract_ok = (
                            payload.get("text") == attempt_text
                            and payload.get("text_lang") == "all_ko"
                            and payload.get("prompt_lang") == "all_ja"
                        )
                    acceptable_for_mix = duration_gate == "pass" and language_contract_ok
                    selection_score = max(0.0, 1.0 - min(abs(candidate_ratio - 1.0), 1.0))
                    if too_long and attempt < 2:
                        if attempt == 0:
                            payload["retry"]["next_action"] = GPTSoVITSRetrySignal.SPEED_FACTOR_ADJUSTED.value
                        elif can_rewrite_for_duration:
                            payload["retry"]["next_action"] = (
                                GPTSoVITSRetrySignal.SCRIPT_SHORTENING_REQUESTED.value
                            )
                    elif too_short and attempt < 2:
                        payload["retry"]["next_action"] = GPTSoVITSRetrySignal.SPEED_FACTOR_ADJUSTED.value
                    candidates.append(
                        TTSCandidate(
                            candidate_index=candidate_index,
                            seed=options.seed,
                            payload=payload,
                            output_path=str(candidate_path),
                            duration_sec=duration,
                            backend=candidate_backend_name,
                            duration_ratio=candidate_ratio,
                            duration_gate=duration_gate,
                            acceptable_for_mix=acceptable_for_mix,
                            selection_score=selection_score,
                            selection_reason=(
                                "duration_and_language_contract_pass"
                                if acceptable_for_mix
                                else "duration_or_language_contract_failed"
                            ),
                            retry_summary=payload["retry"],
                        )
                    )
                    if not (too_long or too_short):
                        break
                    if attempt >= 2:
                        break
                    if attempt == 0:
                        options = (
                            adjust_speed_for_duration(
                                options,
                                duration,
                                segment.duration,
                                maximum=cfg.gsv_tts_max_speed_factor,
                            )
                            if too_long
                            else adjust_speed_for_short_duration(
                                options,
                                duration,
                                segment.duration,
                                minimum=cfg.gsv_tts_min_speed_factor,
                            )
                        )
                        attempt_signals = [
                            GPTSoVITSRetrySignal.DURATION_TOO_LONG
                            if too_long
                            else GPTSoVITSRetrySignal.DURATION_TOO_SHORT,
                            GPTSoVITSRetrySignal.SPEED_FACTOR_ADJUSTED,
                        ]
                        continue
                    if not can_rewrite_for_duration:
                        options = options.model_copy(
                            update={
                                "seed": options.seed + 20_000 + index + attempt
                                if options.seed >= 0
                                else 20_000 + index + attempt
                            }
                        )
                        attempt_signals = [
                            GPTSoVITSRetrySignal.DURATION_TOO_LONG
                            if too_long
                            else GPTSoVITSRetrySignal.DURATION_TOO_SHORT,
                            GPTSoVITSRetrySignal.SEED_CHANGED,
                        ]
                        continue
                    rewritten = rewrite_for_duration(segment.script, segment.duration, cfg.duration_tolerance)
                    if rewritten.tts_text != segment.script.tts_text:
                        segment.script = rewritten
                        attempt_text = rewritten.tts_text
                        expected = rewritten.expected_tts_duration_sec or expected
                    attempt_signals = [
                        GPTSoVITSRetrySignal.DURATION_TOO_LONG,
                        GPTSoVITSRetrySignal.SCRIPT_SHORTENING_REQUESTED,
                    ]
            successful = [
                candidate for candidate in candidates if not candidate.error and candidate.duration_sec is not None
            ]
            if not successful:
                segment.tts = TTSMetadata(
                    backend="mock" if mock else "gpt-sovits",
                    ref_style=resolved_ref_style,
                    speed_factor=speed,
                    candidate_count=cfg.candidate_count,
                    candidates=candidates,
                    source_language=source_language,
                    target_language=target_language,
                    cross_lingual_voice_transfer=source_language != target_language,
                )
                segment.status = "failed"
                segment.errors.append("All TTS candidates failed.")
                return index, segment
            acceptable = [candidate for candidate in successful if candidate.acceptable_for_mix]
            selected_pool = acceptable or successful
            selected = min(selected_pool, key=lambda c: abs((c.duration_sec or 0.0) - segment.duration))
            selected.selected = True
            final_path = project_dir / "work" / "tts" / f"{segment.id}_final.wav"
            ensure_not_same_path(Path(selected.output_path), final_path)
            final_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(selected.output_path, final_path)
            segment.tts = TTSMetadata(
                backend="mock" if mock else "gpt-sovits",
                ref_style=resolved_ref_style,
                speed_factor=float(selected.payload.get("speed_factor", speed)),
                candidate_count=cfg.candidate_count,
                selected_candidate_path=str(final_path),
                candidates=candidates,
                source_language=source_language,
                target_language=target_language,
                cross_lingual_voice_transfer=source_language != target_language,
                retry_summary={
                    "selected_duration_gate": selected.duration_gate,
                    "selected_acceptable_for_mix": selected.acceptable_for_mix,
                    "selected_duration_ratio": selected.duration_ratio,
                },
            )
            segment.status = "synthesized"
            return index, segment

        def synthesize_segment(index: int, segment: Segment, lane_index: int) -> tuple[int, Segment]:
            if mock or segment.status in SKIP_STATUSES or not segment.script:
                return synthesize_segment_locked(index, segment, lane_index)
            with lane_locks[lane_index]:
                return synthesize_segment_locked(index, segment, lane_index)

        segment_jobs = [
            (index, segment, _segment_lane_index(segment, index - 1, gsv_lane_count))
            for index, segment in enumerate(manifest.segments, start=1)
            if only_segment_ids is None or segment.id in only_segment_ids
        ]
        if not mock and gsv_lane_count > 1 and len(segment_jobs) > 1:
            with ThreadPoolExecutor(max_workers=gsv_lane_count) as executor:
                futures = [
                    executor.submit(synthesize_segment, index, segment, lane_index)
                    for index, segment, lane_index in segment_jobs
                ]
                for future in as_completed(futures):
                    index, segment = future.result()
                    save_manifest(project_dir, manifest)
                    last_logged_at = _log_segment_progress(
                        "synth", index, total, segment, manifest, started_at, last_logged_at
                    )
        else:
            for index, segment, lane_index in segment_jobs:
                index, segment = synthesize_segment(index, segment, lane_index)
                save_manifest(project_dir, manifest)
                last_logged_at = _log_segment_progress(
                    "synth", index, total, segment, manifest, started_at, last_logged_at
                )
        gsv_instances = [
            {
                "base_url": manager.base_url,
                "started": manager.started,
                "reused_existing": manager.reused_existing,
                "log_path": str(manager.log_path) if manager.log_path else None,
            }
            for manager in server_managers
        ]
        gsv_server_metadata = {
            "auto_start": should_auto_start_server,
            "concurrency": gsv_lane_count,
            "base_urls": [] if mock else gsv_base_urls,
            "instances": gsv_instances,
        }
        if len(gsv_instances) == 1:
            gsv_server_metadata.update(
                started=gsv_instances[0]["started"],
                reused_existing=gsv_instances[0]["reused_existing"],
                log_path=gsv_instances[0]["log_path"],
            )
        failed_synth_segments = [
            segment.id
            for segment in manifest.segments
            if segment.status == "failed" and (only_segment_ids is None or segment.id in only_segment_ids)
        ]
        if not mock and failed_synth_segments:
            mark_stage(
                manifest,
                "synth",
                "failed",
                backend="gpt-sovits",
                gsv_url=effective_gsv_url,
                gsv_urls=gsv_base_urls,
                gsv_server=gsv_server_metadata,
                failed_segments=failed_synth_segments,
                segment_counts=_segment_counts(manifest),
            )
            save_manifest(project_dir, manifest)
            raise GPTSoVITSError(
                "GPT-SoVITS synthesis failed for segments: "
                + ", ".join(failed_synth_segments[:20])
                + (" ..." if len(failed_synth_segments) > 20 else "")
            )
        mark_stage(
            manifest,
            "synth",
            "completed",
            backend="mock" if mock else "gpt-sovits",
            gsv_url=None if mock else effective_gsv_url,
            gsv_urls=[] if mock else gsv_base_urls,
            gsv_server=gsv_server_metadata,
            concurrency=gsv_lane_count,
            model_switch=model_switch,
            segment_counts=_segment_counts(manifest),
        )
        save_manifest(project_dir, manifest)
        _log_stage_complete("synth", manifest, f"backend={synth_backend_name}")
        return ctx.update_manifest(manifest)
    finally:
        for server_manager in reversed(server_managers):
            server_manager.stop()
