from __future__ import annotations

# ruff: noqa: F403,F405,I001

from asmr_dub_pipeline.pipeline.context import PipelineContext
from asmr_dub_pipeline.pipeline.stages.common import *
from asmr_dub_pipeline.pipeline.stages.source_separation import run_source_separation_stage


ASR_QUALITY_ERROR_PREFIXES = (
    "asr_low_confidence:",
    "asr_repair_rejected",
    "asr_suspicious_pattern:",
)
ASR_QUALITY_ERROR_VALUES = {
    "missing_asr_text",
    "no_speech_detected",
    "asr_degenerate_repetition",
    "asr_non_speech_texture",
    "asr_numeric_runaway",
    "asr_prompt_or_hallucination_leak",
    "asr_sparse_text_density",
}
ASR_WARNING_ANALYSIS_KEYS = (
    "asr_countdown_unverified",
    "asr_numeric_sequence_unverified",
    "asr_sparse_speech_unverified",
)
ASR_WARNING_ANALYSIS_KEY_BY_REASON = {
    "asr_countdown_unverified": "asr_countdown_unverified",
    "asr_numeric_sequence_unverified": "asr_numeric_sequence_unverified",
    "asr_sparse_speech_unverified": "asr_sparse_speech_unverified",
}


def _clear_asr_quality_gate_errors(segment: Segment) -> None:
    segment.errors = [
        error
        for error in segment.errors
        if error not in ASR_QUALITY_ERROR_VALUES
        and not any(error.startswith(prefix) for prefix in ASR_QUALITY_ERROR_PREFIXES)
    ]


def _source_separation_part_audio_inputs(
    project_dir: Path,
    manifest: PipelineManifest,
    selected_audio_path: Path,
) -> list[dict[str, Any]]:
    source_vocals_mono = _resolve_manifest_artifact_path(project_dir, manifest, "source_vocals_mono_16k")
    if source_vocals_mono is None or selected_audio_path.resolve() != source_vocals_mono.resolve():
        return []
    metadata_path = _resolve_manifest_artifact_path(project_dir, manifest, "source_separation_manifest")
    if metadata_path is None or not metadata_path.exists():
        return []
    try:
        metadata = json.loads(metadata_path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return []
    if not isinstance(metadata, dict) or not metadata.get("partwise"):
        return []
    raw_parts = metadata.get("parts")
    if not isinstance(raw_parts, list):
        return []
    parts: list[dict[str, Any]] = []
    for raw_part in raw_parts:
        if not isinstance(raw_part, dict):
            continue
        vocals_mono_path = Path(str(raw_part.get("vocals_mono_path") or ""))
        vocals_path = Path(str(raw_part.get("vocals_path") or ""))
        background_path = Path(str(raw_part.get("background_path") or ""))
        if not vocals_mono_path.exists() or not vocals_path.exists():
            continue
        parts.append(
            {
                "part_index": int(raw_part.get("part_index") or len(parts) + 1),
                "start_sec": float(raw_part.get("start_sec") or 0.0),
                "end_sec": float(raw_part.get("end_sec") or 0.0),
                "duration_sec": float(raw_part.get("duration_sec") or 0.0),
                "vocals_mono_path": str(vocals_mono_path),
                "vocals_path": str(vocals_path),
                "background_path": str(background_path) if background_path else "",
            }
        )
    return parts


def _partwise_audio_duration(parts: list[dict[str, Any]]) -> float | None:
    if not parts:
        return None
    return max(float(part.get("end_sec") or 0.0) for part in parts)


def _transcribe_partwise_audio(backend: Any, parts: list[dict[str, Any]]) -> list[ASRChunk]:
    chunks: list[ASRChunk] = []
    for part in parts:
        offset = float(part["start_sec"])
        part_path = Path(str(part["vocals_mono_path"]))
        for chunk in backend.transcribe(part_path, []):
            words = [
                word.model_copy(
                    update={
                        "start": round(offset + float(word.start), 6),
                        "end": round(offset + float(word.end), 6),
                    }
                )
                for word in chunk.words
            ]
            chunks.append(
                chunk.model_copy(
                    update={
                        "start": round(offset + float(chunk.start), 6),
                        "end": round(offset + float(chunk.end), 6),
                        "words": words,
                    }
                )
            )
    return chunks


def run_transcribe_stage(ctx: PipelineContext, asr_backend: str | None = None, confirm_rights: bool = False, asr_review: bool | None = None, asr_preset: str | None = None, asr_vad_off: bool | None = None, asr_diagnostics: bool | None = None, asr_device: str | None = None, asr_compute_type: str | None = None, asr_batched_inference: bool | None = None, asr_batch_size: int | None = None, asr_repair_enabled: bool | None = None, asr_backend_factory: Any | None = None) -> PipelineManifest:
    project_dir = ctx.project_dir
    manifest = ctx.reload_manifest()
    _load_config_into_manifest(project_dir, manifest)
    cfg = manifest.project_config
    if asr_review is not None:
        next_cfg = type(cfg).model_validate(
            {**cfg.model_dump(mode="json"), "asr_review_enabled": asr_review}
        )
        next_cfg.asr.correction_profile = cfg.asr.correction_profile
        cfg = next_cfg
        manifest.project_config = cfg
    cfg = _effective_asr_config(
        cfg,
        asr_preset=asr_preset,
        asr_vad_off=asr_vad_off,
        asr_diagnostics=asr_diagnostics,
        asr_device=asr_device,
        asr_compute_type=asr_compute_type,
        asr_batched_inference=asr_batched_inference,
        asr_batch_size=asr_batch_size,
        asr_repair_enabled=asr_repair_enabled,
    )
    manifest.project_config = cfg
    backend_kind = asr_backend or cfg.asr_backend
    total = len(manifest.segments)
    _log_stage_start(
        "transcribe",
        f"backend={backend_kind}, preset={cfg.asr_preset}, segments={total}",
    )
    _require_audio_stage_rights(manifest, "transcribe", confirm_rights, metadata={"backend": backend_kind})
    if backend_kind != "mock" and "source_vocals_mono_16k" not in manifest.artifacts:
        _log_stage_checkpoint(
            "transcribe",
            "source separation required",
            "missing=source_vocals_mono_16k",
        )
        manifest = run_source_separation_stage(ctx, confirm_rights=confirm_rights)
        _load_config_into_manifest(project_dir, manifest)
        cfg = manifest.project_config
        if asr_review is not None:
            next_cfg = type(cfg).model_validate(
                {**cfg.model_dump(mode="json"), "asr_review_enabled": asr_review}
            )
            next_cfg.asr.correction_profile = cfg.asr.correction_profile
            cfg = next_cfg
            manifest.project_config = cfg
        cfg = _effective_asr_config(
            cfg,
            asr_preset=asr_preset,
            asr_vad_off=asr_vad_off,
            asr_diagnostics=asr_diagnostics,
            asr_device=asr_device,
            asr_compute_type=asr_compute_type,
            asr_batched_inference=asr_batched_inference,
            asr_batch_size=asr_batch_size,
            asr_repair_enabled=asr_repair_enabled,
        )
        manifest.project_config = cfg
        total = len(manifest.segments)
        _log_stage_checkpoint("transcribe", "source separation complete", f"segments={total}")
    audio_path, mix_audio_path, input_diagnostics = _select_asr_audio_input(
        project_dir,
        manifest,
        backend_kind=backend_kind,
        cfg=cfg,
    )
    selected_source = input_diagnostics.get("selected", {}).get("source", "unknown")
    _log_stage_checkpoint(
        "transcribe",
        "audio selected",
        f"source={selected_source} audio={audio_path.name} segments={total}",
    )
    part_audio_inputs = _source_separation_part_audio_inputs(project_dir, manifest, audio_path)
    if part_audio_inputs:
        input_diagnostics["partwise_source_separation"] = {
            "enabled": True,
            "part_count": len(part_audio_inputs),
            "audio_paths": [part["vocals_mono_path"] for part in part_audio_inputs],
        }
        _log_stage_checkpoint(
            "transcribe",
            "partwise source separation enabled",
            f"parts={len(part_audio_inputs)}",
        )
    write_seed_audio_clips = (backend_kind == "mock" or not cfg.asr_resegment_from_chunks) and not part_audio_inputs
    if not manifest.segments:
        _log_stage_checkpoint(
            "transcribe",
            "seeding full-input segment",
            f"write_audio_clips={write_seed_audio_clips}",
        )
    seeded_for_transcribe = _seed_segments_for_transcribe(
        project_dir,
        manifest,
        audio_path,
        mix_audio_path,
        write_audio_clips=write_seed_audio_clips,
    )
    total = len(manifest.segments)
    if seeded_for_transcribe:
        _log_stage_checkpoint("transcribe", "seed segment ready", f"segments={total}")
    backend_config = _asr_backend_config(cfg)
    _log_stage_checkpoint("transcribe", "creating ASR backend", f"backend={backend_kind}")
    backend = (
        asr_backend_factory(backend_kind, backend_config)
        if asr_backend_factory is not None
        else create_asr_backend(backend_kind, backend_config)
    )
    audio_duration = _partwise_audio_duration(part_audio_inputs) or duration_sec(audio_path)
    _log_stage_checkpoint(
        "transcribe",
        "starting ASR",
        f"backend={backend.name} audio={audio_path.name} duration={audio_duration:.2f}s segments={total}",
    )
    raw_chunks = (
        _transcribe_partwise_audio(backend, part_audio_inputs)
        if part_audio_inputs and backend_kind != "mock"
        else backend.transcribe(audio_path, manifest.segments)
    )
    _log_stage_checkpoint(
        "transcribe",
        "ASR complete",
        f"raw_chunks={len(raw_chunks)} backend={backend.name}",
    )
    chunks = [chunk.model_copy() for chunk in raw_chunks]
    raw_asr_chunk_count = len(raw_chunks)
    asr_text_replacement_count = 0
    asr_text_replacements_summary: dict[str, Any] = {
        "chunks_changed": 0,
        "total_replacements": 0,
        "items": [],
    }
    repair_summary: dict[str, Any] = {
        "enabled": False,
        "attempted": 0,
        "repaired": 0,
        "skipped": 0,
        "items": [],
    }
    asr_review_summary: dict[str, Any] = {
        "enabled": bool(cfg.asr_review_enabled),
        "backend": cfg.asr_review_backend,
        "attempted": 0,
        "reviewed": 0,
        "replaced": 0,
        "manual_review": 0,
        "skipped": 0,
        "generated_candidates": 0,
        "error": None,
        "items": [],
    }
    qwen_fallback_backend = None
    qwen_fallback_summary: dict[str, Any] = {
        "enabled": False,
        "available": False,
        "backend": "qwen_asr",
        "skipped_reason": "disabled",
    }
    repaired_chunks = [chunk.model_copy() for chunk in chunks]
    filtered_final_chunks: list[dict[str, Any]] = []
    if backend_kind != "mock":
        _log_stage_checkpoint(
            "transcribe",
            "applying ASR post-processing",
            f"repair={cfg.asr_repair_enabled} review={cfg.asr_review_enabled}",
        )
        qwen_fallback_backend, qwen_fallback_summary = _create_qwen_repair_fallback_backend(
            cfg,
            manifest,
        )
        repair_audio_path = audio_path
        chunks, repair_summary = _repair_asr_chunks(
            chunks,
            backend=backend,
            project_dir=project_dir,
            repair_audio_path=repair_audio_path,
            audio_duration_sec=audio_duration,
            cfg=cfg,
            qwen_fallback_backend=qwen_fallback_backend,
        )
        _log_stage_checkpoint(
            "transcribe",
            "ASR repair complete",
            "attempted={attempted} repaired={repaired} skipped={skipped}".format(
                attempted=repair_summary.get("attempted", 0),
                repaired=repair_summary.get("repaired", 0),
                skipped=repair_summary.get("skipped", 0),
            ),
        )
        repaired_chunks = [chunk.model_copy() for chunk in chunks]
        repair_summary_path = project_dir / "work" / "transcribe" / "asr_repair_summary.json"
        write_json_atomic(repair_summary_path, repair_summary)
        manifest.artifacts["asr_repair_summary"] = str(repair_summary_path)
        chunks, asr_text_replacements_summary = _apply_asr_text_replacements_to_chunks_with_summary(
            chunks,
            {},
            contextual_replacements=cfg.asr_review_candidate_replacements,
        )
        chunks, asr_review_summary = _review_asr_chunks_with_model(
            chunks,
            backend=backend,
            project_dir=project_dir,
            review_audio_path=repair_audio_path,
            audio_duration_sec=audio_duration,
            cfg=cfg,
        )
        _log_stage_checkpoint(
            "transcribe",
            "ASR review complete",
            "attempted={attempted} replaced={replaced} manual_review={manual_review}".format(
                attempted=asr_review_summary.get("attempted", 0),
                replaced=asr_review_summary.get("replaced", 0),
                manual_review=asr_review_summary.get("manual_review", 0),
            ),
        )
        asr_review_summary_path = project_dir / "work" / "transcribe" / "asr_review_summary.json"
        write_json_atomic(asr_review_summary_path, asr_review_summary)
        manifest.artifacts["asr_review_summary"] = str(asr_review_summary_path)
        chunks, post_review_replacements_summary = _apply_asr_text_replacements_to_chunks_with_summary(
            chunks,
            cfg.asr_text_replacements,
            contextual_replacements=cfg.asr_review_candidate_replacements,
        )
        asr_text_replacements_summary = _merge_asr_text_replacement_summaries(
            asr_text_replacements_summary,
            post_review_replacements_summary,
        )
        asr_text_replacement_count = int(asr_text_replacements_summary["chunks_changed"])
        chunks, filtered_final_chunks = _filter_final_asr_chunks_for_hallucinations(
            chunks,
            cfg=cfg,
        )
        if filtered_final_chunks:
            _log_stage_checkpoint(
                "transcribe",
                "filtered hallucinated ASR chunks",
                f"items={len(filtered_final_chunks)}",
            )
    final_chunks = [chunk.model_copy() for chunk in chunks]
    resegmented_from_chunks = False
    previous_segment_count = len(manifest.segments)
    manual_segments_path = project_dir / "work" / "segments" / "manifests" / "segments_manual.json"
    if cfg.asr_resegment_from_chunks and backend_kind != "mock" and chunks and not manual_segments_path.exists():
        _log_stage_checkpoint(
            "transcribe",
            "building segments from ASR chunks",
            f"chunks={len(chunks)} previous_segments={previous_segment_count}",
        )
        resegmented = _segments_from_asr_chunks(
            chunks,
            project_dir=project_dir,
            backend=backend.name,
            fallback_language=cfg.asr_language,
            audio_duration_sec=audio_duration,
            min_segment_sec=cfg.asr_resegment_min_sec,
            merge_gap_sec=cfg.asr_resegment_merge_gap_sec,
            max_segment_sec=cfg.asr_resegment_max_sec,
            sparse_chunk_max_sec=cfg.asr_sparse_chunk_max_sec,
            sparse_chunk_min_chars_per_sec=cfg.asr_sparse_chunk_min_chars_per_sec,
            countdown_merge_enabled=cfg.asr_countdown_merge_enabled,
            countdown_merge_gap_sec=cfg.asr_countdown_merge_gap_sec,
            countdown_merge_max_span_sec=cfg.asr_countdown_merge_max_span_sec,
        )
        if resegmented:
            if part_audio_inputs:
                write_segment_audio_clips_from_parts(resegmented, part_audio_inputs, project_dir)
            else:
                write_segment_audio_clips(resegmented, audio_path, mix_audio_path, project_dir)
            split_resegmented = _split_sparse_edge_segments_by_audio(
                resegmented,
                project_dir=project_dir,
                cfg=cfg,
                merge_gap_sec=cfg.asr_resegment_merge_gap_sec,
            )
            if split_resegmented is not resegmented:
                resegmented = split_resegmented
                if part_audio_inputs:
                    write_segment_audio_clips_from_parts(resegmented, part_audio_inputs, project_dir)
                else:
                    write_segment_audio_clips(resegmented, audio_path, mix_audio_path, project_dir)
            manifest.segments = resegmented
            total = len(manifest.segments)
            resegmented_from_chunks = True
            _log_stage_checkpoint(
                "transcribe",
                "ASR resegmentation complete",
                f"segments={previous_segment_count}->{total}",
            )
    _log_stage_checkpoint(
        "transcribe",
        "mapping ASR chunks to segments",
        f"segments={len(manifest.segments)} chunks={len(chunks)}",
    )
    mapped = (
        {segment.id: segment.source_script for segment in manifest.segments}
        if resegmented_from_chunks
        else map_chunks_to_segments(
            manifest.segments,
            chunks,
            backend=backend.name,
            fallback_language=cfg.asr_language,
        )
    )
    rows: list[dict[str, Any]] = []
    started_at = monotonic()
    last_logged_at = started_at
    with_text = 0
    manual_review_count = 0
    no_speech_count = 0
    non_speech_texture_count = 0
    whole_input_no_speech = backend_kind != "mock" and not final_chunks
    for index, segment in enumerate(manifest.segments, start=1):
        source_script = mapped.get(segment.id)
        segment.source_script = source_script
        source_has_text = bool(source_script and source_script.text.strip())
        no_speech_segment = whole_input_no_speech and not source_has_text
        review_reasons = (
            ["no_speech_detected"]
            if no_speech_segment
            else _source_script_asr_review_reasons(source_script, cfg)
        )
        non_speech_texture_reason = (
            None if no_speech_segment else _source_script_non_speech_texture_reason(source_script)
        )
        repair_review_reasons = _source_script_rejected_repair_reasons(
            source_script,
            repair_summary,
        )
        asr_warning_reasons: list[str] = []
        if non_speech_texture_reason is None:
            asr_warning_reasons = _source_script_asr_warning_reasons(source_script, cfg)
            repair_review_reasons = _filter_asr_repair_review_reasons(
                source_script,
                cfg,
                review_reasons=review_reasons,
                repair_review_reasons=repair_review_reasons,
            )
            review_reasons.extend(
                reason for reason in repair_review_reasons if reason not in review_reasons
            )
        else:
            review_reasons = [non_speech_texture_reason]
        _clear_asr_quality_gate_errors(segment)
        for key in ASR_WARNING_ANALYSIS_KEYS:
            segment.analysis.pop(key, None)
        if no_speech_segment:
            segment.status = "no_speech_detected"
            no_speech_count += 1
        elif non_speech_texture_reason is not None:
            segment.status = "non_speech_texture"
            segment.keep_original_texture = True
            no_speech_count += 1
            non_speech_texture_count += 1
            if non_speech_texture_reason not in segment.errors:
                segment.errors.append(non_speech_texture_reason)
        elif review_reasons:
            segment.status = "needs_manual_review"
            manual_review_count += 1
            for reason in review_reasons:
                if reason not in segment.errors:
                    segment.errors.append(reason)
        elif source_has_text:
            segment.status = "transcribed"
        if asr_warning_reasons and not review_reasons:
            for warning_reason in asr_warning_reasons:
                key = ASR_WARNING_ANALYSIS_KEY_BY_REASON.get(warning_reason)
                if key is None:
                    continue
                segment.analysis[key] = {
                    "reason": warning_reason,
                    "source_text": source_script.text.strip() if source_script else "",
                }
        quality_gate = {
            "decision": (
                "no_speech"
                if no_speech_segment
                else "texture"
                if non_speech_texture_reason is not None
                else "block_tts"
                if review_reasons
                else "pass_with_warning"
                if asr_warning_reasons
                else "pass"
            ),
            "reasons": review_reasons,
            "tts_blocked": bool(review_reasons),
        }
        if asr_warning_reasons and not review_reasons:
            quality_gate["warnings"] = asr_warning_reasons
        segment.analysis["asr_quality_gate"] = quality_gate
        if source_has_text and non_speech_texture_reason is None:
            with_text += 1
            status = "needs_manual_review" if review_reasons else "transcribed"
        elif non_speech_texture_reason is not None:
            status = "non_speech_texture"
        elif no_speech_segment:
            status = "no_speech_detected"
        else:
            status = "needs_manual_review"
        rows.append(
            {
                "segment_id": segment.id,
                "status": status,
                "review_reasons": review_reasons,
                "source_script": source_script.model_dump(mode="json") if source_script else None,
            }
        )
        last_logged_at = _log_segment_progress(
            "transcribe", index, total, segment, manifest, started_at, last_logged_at
        )
    _log_stage_checkpoint(
        "transcribe",
        "writing transcription artifacts",
        f"segments={len(manifest.segments)} rows={len(rows)}",
    )
    jsonl_path = project_dir / "work" / "transcribe" / "source_segments.jsonl"
    _write_jsonl_atomic(jsonl_path, rows)
    out_path = project_dir / "work" / "segments" / "manifests" / "segments_transcribed.json"
    write_json_atomic(out_path, {"segments": [s.model_dump(mode="json") for s in manifest.segments]})
    manifest.artifacts["source_segments"] = str(jsonl_path)
    manifest.artifacts["segments_transcribed"] = str(out_path)
    if resegmented_from_chunks:
        manifest.artifacts["segments_asr_resegmented"] = str(out_path)
    _write_asr_diagnostics_artifacts(
        project_dir,
        manifest,
        backend_kind=backend_kind,
        backend_name=backend.name,
        cfg=cfg,
        input_diagnostics=input_diagnostics,
        raw_chunks=raw_chunks,
        repaired_chunks=repaired_chunks,
        final_chunks=final_chunks,
        repair_summary=repair_summary,
        asr_review_summary=asr_review_summary,
        replacements_summary=asr_text_replacements_summary,
        filtered_summary=filtered_final_chunks,
        qwen_fallback_summary=qwen_fallback_summary,
    )
    _log_stage_checkpoint(
        "transcribe",
        "ASR diagnostics written",
        f"raw_chunks={raw_asr_chunk_count} final_chunks={len(final_chunks)}",
    )
    countdown_summary = _countdown_timeline_summary(manifest.segments)
    _warn_missing_countdown_timelines(
        manifest,
        countdown_summary,
        word_timestamps_enabled=bool(cfg.asr_word_timestamps),
    )
    asr_high_risk_report = _write_asr_high_risk_report_artifact(
        project_dir,
        manifest,
        cfg=cfg,
        replacements_summary=asr_text_replacements_summary,
        repair_summary=repair_summary,
        asr_review_summary=asr_review_summary,
        filtered_summary=filtered_final_chunks,
    )
    asr_high_risk_summary = asr_high_risk_report["summary"]
    asr_postprocess_review = _write_asr_postprocess_review_artifact(
        project_dir,
        manifest,
        cfg=cfg,
        replacements_summary=asr_text_replacements_summary,
    )
    asr_postprocess_review_summary = asr_postprocess_review["summary"]
    mark_stage(
        manifest,
        "transcribe",
        "completed",
        backend=backend_kind,
        backend_name=backend.name,
        asr_preset=cfg.asr_preset,
        asr_device=cfg.asr_device,
        asr_compute_type=cfg.asr_compute_type,
        asr_batched_inference=cfg.asr_batched_inference,
        asr_batch_size=cfg.asr_batch_size,
        asr_word_timestamps=cfg.asr_word_timestamps,
        asr_word_timestamp_chunk_count=sum(1 for chunk in final_chunks if chunk.words),
        asr_word_timestamp_count=sum(len(chunk.words) for chunk in final_chunks),
        asr_input_source=input_diagnostics.get("selected", {}).get("source"),
        segment_count=total,
        previous_segment_count=previous_segment_count,
        raw_asr_chunk_count=raw_asr_chunk_count,
        asr_chunk_count=len(chunks),
        asr_repair_attempted=repair_summary.get("attempted", 0),
        asr_repair_repaired=repair_summary.get("repaired", 0),
        asr_review_attempted=asr_review_summary.get("attempted", 0),
        asr_review_replaced=asr_review_summary.get("replaced", 0),
        asr_review_manual_review=asr_review_summary.get("manual_review", 0),
        asr_review_error=asr_review_summary.get("error"),
        asr_text_replacements=asr_text_replacement_count,
        seeded_for_transcribe=seeded_for_transcribe,
        resegmented_from_chunks=resegmented_from_chunks,
        transcribed=with_text,
        no_speech_detected=no_speech_count,
        non_speech_texture=non_speech_texture_count,
        needs_manual_review=max(total - with_text - no_speech_count, manual_review_count),
        asr_auto_dub_ready=bool(asr_high_risk_summary["automated_dubbing_ready"]),
        asr_high_risk_warning=int(asr_high_risk_summary["warning"]),
        asr_high_risk_severe=int(asr_high_risk_summary["severe"]),
        asr_high_risk_items=len(asr_high_risk_report["items"]),
        asr_high_risk_blocking_reasons=asr_high_risk_summary["blocking_reasons"],
        asr_postprocess_review_items=int(asr_postprocess_review_summary["item_count"]),
        asr_postprocess_auto_replace=int(asr_postprocess_review_summary["auto_replace"]),
        asr_postprocess_candidate_review=int(
            asr_postprocess_review_summary["candidate_review"]
        ),
        asr_postprocess_manual_review=int(asr_postprocess_review_summary["manual_review"]),
        **countdown_summary,
    )
    save_manifest(project_dir, manifest)
    _log_stage_complete("transcribe", manifest, f"backend={backend_kind}")
    return ctx.update_manifest(manifest)
