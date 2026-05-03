from __future__ import annotations

# ruff: noqa: F403,F405,I001

from asmr_dub_pipeline.pipeline.context import PipelineContext
from asmr_dub_pipeline.pipeline.stages.common import *
from asmr_dub_pipeline.pipeline.stages.source_separation import run_source_separation_stage


def run_transcribe_stage(ctx: PipelineContext, asr_backend: str | None = None, confirm_rights: bool = False, asr_review: bool | None = None, asr_preset: str | None = None, asr_vad_off: bool | None = None, asr_diagnostics: bool | None = None, asr_device: str | None = None, asr_compute_type: str | None = None, asr_batched_inference: bool | None = None, asr_batch_size: int | None = None) -> PipelineManifest:
    project_dir = ctx.project_dir
    manifest = ctx.reload_manifest()
    _load_config_into_manifest(project_dir, manifest)
    cfg = manifest.project_config
    if asr_review is not None:
        cfg = type(cfg).model_validate(
            {**cfg.model_dump(mode="json"), "asr_review_enabled": asr_review}
        )
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
        manifest = run_source_separation_stage(ctx, confirm_rights=confirm_rights)
        _load_config_into_manifest(project_dir, manifest)
        cfg = manifest.project_config
        if asr_review is not None:
            cfg = type(cfg).model_validate(
                {**cfg.model_dump(mode="json"), "asr_review_enabled": asr_review}
            )
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
        )
        manifest.project_config = cfg
        total = len(manifest.segments)
    audio_path, mix_audio_path, input_diagnostics = _select_asr_audio_input(
        project_dir,
        manifest,
        backend_kind=backend_kind,
        cfg=cfg,
    )
    seeded_for_transcribe = _seed_segments_for_transcribe(project_dir, manifest, audio_path, mix_audio_path)
    total = len(manifest.segments)
    backend = create_asr_backend(backend_kind, _asr_backend_config(cfg))
    audio_duration = duration_sec(audio_path)
    raw_chunks = backend.transcribe(audio_path, manifest.segments)
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
        qwen_fallback_backend, qwen_fallback_summary = _create_qwen_repair_fallback_backend(
            cfg,
            manifest,
        )
        repair_audio_path = Path(manifest.artifacts.get("gemma_mono_16k", audio_path))
        chunks, repair_summary = _repair_asr_chunks(
            chunks,
            backend=backend,
            project_dir=project_dir,
            repair_audio_path=repair_audio_path,
            audio_duration_sec=audio_duration,
            cfg=cfg,
            qwen_fallback_backend=qwen_fallback_backend,
        )
        repaired_chunks = [chunk.model_copy() for chunk in chunks]
        repair_summary_path = project_dir / "work" / "transcribe" / "asr_repair_summary.json"
        write_json_atomic(repair_summary_path, repair_summary)
        manifest.artifacts["asr_repair_summary"] = str(repair_summary_path)
        chunks, asr_review_summary = _review_asr_chunks_with_model(
            chunks,
            backend=backend,
            project_dir=project_dir,
            review_audio_path=repair_audio_path,
            audio_duration_sec=audio_duration,
            cfg=cfg,
        )
        asr_review_summary_path = project_dir / "work" / "transcribe" / "asr_review_summary.json"
        write_json_atomic(asr_review_summary_path, asr_review_summary)
        manifest.artifacts["asr_review_summary"] = str(asr_review_summary_path)
        chunks, asr_text_replacements_summary = _apply_asr_text_replacements_to_chunks_with_summary(
            chunks,
            cfg.asr_text_replacements,
        )
        asr_text_replacement_count = int(asr_text_replacements_summary["chunks_changed"])
        chunks, filtered_final_chunks = _filter_final_asr_chunks_for_hallucinations(
            chunks,
            cfg=cfg,
        )
    final_chunks = [chunk.model_copy() for chunk in chunks]
    resegmented_from_chunks = False
    previous_segment_count = len(manifest.segments)
    manual_segments_path = project_dir / "work" / "segments" / "manifests" / "segments_manual.json"
    if cfg.asr_resegment_from_chunks and backend_kind != "mock" and chunks and not manual_segments_path.exists():
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
        )
        if resegmented:
            write_segment_audio_clips(resegmented, audio_path, mix_audio_path, project_dir)
            manifest.segments = resegmented
            total = len(manifest.segments)
            resegmented_from_chunks = True
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
    for index, segment in enumerate(manifest.segments, start=1):
        source_script = mapped.get(segment.id)
        segment.source_script = source_script
        review_reasons = _source_script_asr_review_reasons(source_script, cfg)
        repair_review_reasons = _source_script_rejected_repair_reasons(
            source_script,
            repair_summary,
        )
        review_reasons.extend(
            reason for reason in repair_review_reasons if reason not in review_reasons
        )
        if review_reasons:
            segment.status = "needs_manual_review"
            manual_review_count += 1
            for reason in review_reasons:
                if reason not in segment.errors:
                    segment.errors.append(reason)
        if source_script and source_script.text:
            with_text += 1
            status = "needs_manual_review" if review_reasons else "transcribed"
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
        needs_manual_review=max(total - with_text, manual_review_count),
    )
    save_manifest(project_dir, manifest)
    _log_stage_complete("transcribe", manifest, f"backend={backend_kind}")
    return ctx.update_manifest(manifest)
