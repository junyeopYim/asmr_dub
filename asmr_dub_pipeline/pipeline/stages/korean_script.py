from __future__ import annotations

# ruff: noqa: F403,F405,I001

from asmr_dub_pipeline.pipeline.context import PipelineContext
from asmr_dub_pipeline.pipeline.stages.common import *


def run_korean_script_stage(ctx: PipelineContext, confirm_rights: bool = False) -> PipelineManifest:
    project_dir = ctx.project_dir
    manifest = ctx.reload_manifest()
    _load_config_into_manifest(project_dir, manifest)
    cfg = manifest.project_config
    if _canonical_language(cfg.target_language) != "ko":
        raise ValueError("korean-script requires project_config.target_language='ko'.")
    total = len(manifest.segments)
    _log_stage_start("korean-script", f"segments={total}")
    _require_audio_stage_rights(manifest, "korean-script", confirm_rights)
    if manifest.stage_state.get("translate-ko", {}).get("status") != "completed":
        raise ValueError("korean-script requires a completed translate-ko stage.")

    scripted = 0
    needs_manual_review = 0
    started_at = monotonic()
    last_logged_at = started_at
    for index, segment in enumerate(manifest.segments, start=1):
        if segment.status in SKIP_STATUSES:
            needs_manual_review += 1
            segment.script = None
            message = f"korean-script skipped segment status {segment.status}."
            if message not in segment.errors:
                segment.errors.append(message)
            last_logged_at = _log_segment_progress(
                "korean-script", index, total, segment, manifest, started_at, last_logged_at
            )
            continue
        translation = segment.translation_ko
        text = translation.ko_natural.strip() if translation else ""
        source_text = segment.source_script.text.strip() if segment.source_script else ""
        normalized = normalize_korean_tts_text(text) if text else None
        if not text or normalized is None or not normalized.text:
            needs_manual_review += 1
            segment.status = "needs_manual_review"
            segment.errors.append("Cannot build Korean TTS script without translation_ko.ko_natural.")
            last_logged_at = _log_segment_progress(
                "korean-script", index, total, segment, manifest, started_at, last_logged_at
            )
            continue
        segment.script = JapaneseScript(
            literal_ja=source_text,
            ja_text=source_text or normalized.text,
            tts_text=normalized.text,
            tts_language="ko",
            source_language=cfg.source_language,
            target_language=cfg.target_language,
            ref_style="whisper_close",
            emotion="gentle",
            pace="slow",
            volume="soft",
            nonverbal_cues=normalized.cues,
            spatial_style="center",
            expected_tts_duration_sec=segment.duration,
            style_tags=["korean_translation", "soft_whisper"],
            risk_flags=["source_script_translated_to_ko", *normalized.risk_flags],
        )
        preflight = preflight_tts_text(
            segment.script,
            target_language=cfg.target_language,
            source_text=source_text,
            min_hangul_ratio=cfg.gsv_ko_text_min_hangul_ratio,
        )
        segment.analysis["pre_synth_text_qc"] = preflight.as_payload()
        if preflight.blocked:
            needs_manual_review += 1
            segment.status = "needs_manual_review"
            segment.errors.append(
                "Korean TTS preflight blocked synthesis: " + ", ".join(preflight.issues)
            )
            last_logged_at = _log_segment_progress(
                "korean-script", index, total, segment, manifest, started_at, last_logged_at
            )
            continue
        segment.status = "scripted"
        scripted += 1
        last_logged_at = _log_segment_progress(
            "korean-script", index, total, segment, manifest, started_at, last_logged_at
        )

    out_path = project_dir / "work" / "segments" / "manifests" / "segments_ko_script.json"
    write_json_atomic(out_path, {"segments": [s.model_dump(mode="json") for s in manifest.segments]})
    manifest.artifacts["segments_ko_script"] = str(out_path)
    mark_stage(
        manifest,
        "korean-script",
        "completed",
        scripted=scripted,
        needs_manual_review=needs_manual_review,
    )
    save_manifest(project_dir, manifest)
    _log_stage_complete("korean-script", manifest, "tts_language=ko")
    return ctx.update_manifest(manifest)
