from __future__ import annotations

# ruff: noqa: F403,F405,I001

from asmr_dub_pipeline.pipeline.context import PipelineContext
from asmr_dub_pipeline.pipeline.stages.common import *


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
    selected_span = selected_spans[0] if selected_spans else None
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
    write_json_atomic(ref_qc_path, {"refs": ref_qc_rows})
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
