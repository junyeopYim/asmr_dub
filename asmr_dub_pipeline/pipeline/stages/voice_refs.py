from __future__ import annotations

# ruff: noqa: F403,F405,I001

from asmr_dub_pipeline.pipeline.context import PipelineContext
from asmr_dub_pipeline.pipeline.stages.common import *


def _prepared_ref_metric_reject_reasons(
    metrics: AudioQualityMetrics,
    cfg: ProjectConfig,
) -> list[str]:
    reasons: list[str] = []
    min_sec = max(float(cfg.gsv_ref_min_sec), GSV_API_MIN_REF_SEC)
    max_sec = float(cfg.gsv_ref_max_sec)
    if metrics.duration_sec < min_sec:
        reasons.append(f"actual_duration_below_ref_min:{metrics.duration_sec:.3f}<{min_sec:.3f}")
    if metrics.duration_sec > max_sec:
        reasons.append(f"actual_duration_above_ref_max:{metrics.duration_sec:.3f}>{max_sec:.3f}")
    if metrics.score < float(cfg.gsv_ref_min_quality_score):
        reasons.append(
            f"quality_score_below_ref_min:{metrics.score:.3f}<{cfg.gsv_ref_min_quality_score:.3f}"
        )
    return reasons


def _source_audio_duration_rejection_rows(
    project_dir: Path,
    manifest: PipelineManifest,
    cfg: ProjectConfig,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for segment in manifest.segments:
        if not _segment_can_seed_voice_ref(segment, cfg.source_language):
            continue
        span = _VoiceRefSpan((segment,), float(segment.duration))
        reasons = _voice_ref_span_audio_duration_reject_reasons(
            project_dir,
            span,
            min_sec=max(float(cfg.gsv_ref_min_sec), GSV_API_MIN_REF_SEC),
            max_sec=float(cfg.gsv_ref_max_sec),
        )
        if not reasons:
            continue
        rows.append(
            {
                "stage": "source_audio_preselection",
                "segment_ids": [segment.id],
                "metadata_duration_sec": round(float(segment.duration), 6),
                "reject_reasons": reasons,
            }
        )
    return rows


def run_prepare_source_voice_refs_stage(ctx: PipelineContext, refs_path: Path | None = None, confirm_rights: bool = False) -> PipelineManifest:
    project_dir = ctx.project_dir
    manifest = ctx.reload_manifest()
    _load_config_into_manifest(project_dir, manifest)
    _log_stage_start("prepare-refs", f"project={project_dir}")
    manifest.rights_audit = require_existing_or_confirmed_rights(
        manifest.rights_audit,
        confirm_rights,
        "prepare-refs",
        _manifest_source_path(manifest),
        metadata={"source_derived_voice_refs": True},
    )
    cfg = manifest.project_config
    selected_spans = _select_voice_ref_spans(project_dir, manifest, cfg)
    rejected_span_rows = _source_audio_duration_rejection_rows(project_dir, manifest, cfg)
    selected_span: _VoiceRefSpan | None = None
    selected_validation_metrics: AudioQualityMetrics | None = None
    validation_dir = project_dir / "work" / "gpt_sovits" / "ref_candidates"
    validation_dir.mkdir(parents=True, exist_ok=True)
    span_passes: list[tuple[str, list[_VoiceRefSpan]]] = [("primary", selected_spans)]
    selected_span_ids = {
        tuple(segment.id for segment in selected.segments) for selected in selected_spans
    }
    relaxed_cfg = cfg.model_copy(update={"gsv_few_shot_prefer_plain_text": False})
    relaxed_spans = [
        span
        for span in _select_voice_ref_spans(project_dir, manifest, relaxed_cfg)
        if tuple(segment.id for segment in span.segments) not in selected_span_ids
    ]
    if relaxed_spans:
        span_passes.append(("relaxed_text_penalty", relaxed_spans))
    for pass_name, candidate_spans in span_passes:
        for candidate_index, candidate_span in enumerate(candidate_spans, start=1):
            if not candidate_span.segments or not candidate_span.segments[0].source_script:
                continue
            validation_path = validation_dir / f"{pass_name}_{candidate_index:02d}.wav"
            _write_voice_ref_span(project_dir, candidate_span, validation_path)
            metrics = measure_source_voice_quality(validation_path)
            reject_reasons = _prepared_ref_metric_reject_reasons(metrics, cfg)
            if reject_reasons:
                rejected_span_rows.append(
                    {
                        "stage": "written_wav_validation",
                        "selection_pass": pass_name,
                        "candidate_index": candidate_index,
                        "segment_ids": [segment.id for segment in candidate_span.segments],
                        "metadata_duration_sec": round(float(candidate_span.duration), 6),
                        "actual_duration_sec": round(float(metrics.duration_sec), 6),
                        "metrics": metrics.as_payload(),
                        "reject_reasons": reject_reasons,
                    }
                )
                continue
            selected_span = candidate_span
            selected_validation_metrics = metrics
            break
        if selected_span is not None:
            break
    if selected_span is None or not selected_span.segments[0].source_script:
        raise ValueError(
            "Cannot prepare source voice refs without a transcribed audio span "
            f"within {cfg.gsv_ref_min_sec:.2f}-{cfg.gsv_ref_max_sec:.2f} seconds."
        )
    selected = selected_span.segments[0]

    actual_refs_path = resolve_refs_json_path(refs_path or Path("refs/refs.json"), project_dir)
    data = json.loads(actual_refs_path.read_text("utf-8")) if actual_refs_path.exists() else {}
    if not isinstance(data, dict):
        raise ValueError(f"refs JSON must be an object keyed by style name: {actual_refs_path}")

    aux_spans = selected_spans[1:]
    prompt_lang = selected.source_script.language or cfg.source_language
    prompt_text, prompt_text_original, prompt_text_flags = _model_boundary_text_for_language(
        _voice_ref_span_prompt_text(selected_span),
        prompt_lang,
    )
    prepared: dict[str, str] = {}
    ref_qc_rows: list[dict[str, Any]] = []
    for style in ("whisper_close", "sleepy"):
        entry = data.get(style) if isinstance(data.get(style), dict) else {}
        raw_ref_path = str(entry.get("ref_audio_path") or f"refs/{style}.wav")
        ref_path = Path(raw_ref_path).expanduser()
        resolved_ref_path = (
            project_dir / ref_path if not ref_path.is_absolute() else ref_path
        ).resolve()
        resolved_ref_path = ensure_inside_project(project_dir, resolved_ref_path)
        resolved_ref_path.parent.mkdir(parents=True, exist_ok=True)
        _write_voice_ref_span(project_dir, selected_span, resolved_ref_path)
        selected_metrics = measure_source_voice_quality(resolved_ref_path)
        final_reject_reasons = _prepared_ref_metric_reject_reasons(selected_metrics, cfg)
        if final_reject_reasons:
            raise ValueError(
                "Prepared source voice reference failed wav validation: "
                + ", ".join(final_reject_reasons)
            )
        aux_ref_audio_paths: list[str] = []
        for aux_index, aux_span in enumerate(aux_spans, start=1):
            aux_raw_path = f"refs/{style}_aux_{aux_index}.wav"
            aux_path = ensure_inside_project(project_dir, (project_dir / aux_raw_path).resolve())
            aux_path.parent.mkdir(parents=True, exist_ok=True)
            _write_voice_ref_span(project_dir, aux_span, aux_path)
            aux_ref_audio_paths.append(aux_raw_path)
        data[style] = {
            **entry,
            "ref_audio_path": raw_ref_path,
            "prompt_text": prompt_text,
            "prompt_text_original": prompt_text_original,
            "prompt_lang": prompt_lang,
            "aux_ref_audio_paths": aux_ref_audio_paths,
            "source_language": cfg.source_language,
            "target_language": cfg.target_language,
            "cross_lingual_role": "ja_source_prompt_for_ko_tts",
            "text_normalization": {
                "policy": "ja_hiragana" if _canonical_language(prompt_lang) == "ja" else "none",
                "risk_flags": prompt_text_flags,
            },
        }
        prepared[style] = str(resolved_ref_path)
        ref_qc_rows.append(
            {
                "style": style,
                "segment_id": selected.id,
                "source_language": cfg.source_language,
                "target_language": cfg.target_language,
                "prompt_lang": prompt_lang,
                "prompt_text": prompt_text,
                "prompt_text_original": prompt_text_original,
                "prompt_text_normalization": {
                    "policy": "ja_hiragana" if _canonical_language(prompt_lang) == "ja" else "none",
                    "risk_flags": prompt_text_flags,
                },
                "metrics": selected_metrics.as_payload(),
                "selected_actual_duration_sec": round(
                    float(
                        selected_validation_metrics.duration_sec
                        if selected_validation_metrics is not None
                        else selected_metrics.duration_sec
                    ),
                    6,
                ),
                "selected_segment_ids": [segment.id for segment in selected_span.segments],
                "selected_span_start_sec": selected_span.segments[0].start,
                "selected_span_end_sec": selected_span.segments[-1].end,
                "selected_span_duration_sec": round(selected_span.duration, 6),
                "selected_aux_segment_ids": [span.segments[0].id for span in aux_spans],
                "selected_aux_span_segment_ids": [
                    [segment.id for segment in span.segments] for span in aux_spans
                ],
            }
        )

    write_json_atomic(actual_refs_path, data)
    ref_qc_path = project_dir / "work" / "gpt_sovits" / "ref_qc.json"
    write_json_atomic(ref_qc_path, {"refs": ref_qc_rows, "rejected_spans": rejected_span_rows})
    manifest.artifacts["source_voice_refs"] = str(actual_refs_path)
    manifest.artifacts["source_voice_ref_qc"] = str(ref_qc_path)
    mark_stage(
        manifest,
        "prepare-refs",
        "completed",
        segment_id=selected.id,
        selected_segment_ids=[segment.id for segment in selected_span.segments],
        refs=prepared,
        source_language=cfg.source_language,
        target_language=cfg.target_language,
        cross_lingual_voice_transfer=cfg.source_language != cfg.target_language,
        ref_qc_path=str(ref_qc_path),
    )
    save_manifest(project_dir, manifest)
    _log_stage_complete("prepare-refs", manifest, f"segment={selected.id}")
    return ctx.update_manifest(manifest)
