from __future__ import annotations

# ruff: noqa: F403,F405,I001

from asmr_dub_pipeline.pipeline.context import PipelineContext
from asmr_dub_pipeline.pipeline.stages.common import *
from asmr_dub_pipeline.schemas import NonverbalCue
from asmr_dub_pipeline.script.duration_rewrite import (
    estimate_tts_duration,
    korean_tts_speech_char_count,
    korean_tts_timing_budget,
)
from asmr_dub_pipeline.script.korean_colloquial import (
    COLLOQUIAL_REWRITE_NOTE,
    colloquialize_korean_text,
)


_KOREAN_SCRIPT_SAFE_STYLE_TAG_RE = re.compile(r"^[a-z0-9_]{1,48}$")
_KOREAN_SCRIPT_EMOTIONS = {"gentle", "sleepy", "reassuring", "playful", "serious", "neutral"}
_KOREAN_SCRIPT_PACES = {"very_slow", "slow", "normal", "slightly_fast"}
_KOREAN_SCRIPT_VOLUMES = {"whisper", "soft", "normal"}
_KOREAN_SCRIPT_SPATIAL_STYLES = {
    "center",
    "left_close",
    "right_close",
    "center_close",
    "center_far",
    "sleepy_center",
    "binaural_sweep",
    "ambient",
}
_KOREAN_MICRO_SEGMENT_MAX_SEC = 0.6
_KOREAN_MICRO_SEGMENT_MAX_RATIO = 1.5


def _normalize_korean_script_tag(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _append_korean_script_tag(tags: list[str], value: Any) -> None:
    tag = _normalize_korean_script_tag(value)
    if not tag or not _KOREAN_SCRIPT_SAFE_STYLE_TAG_RE.fullmatch(tag):
        return
    if tag not in tags:
        tags.append(tag)


def _korean_script_style_tags(segment: Segment) -> list[str]:
    tags = ["korean_translation", "soft_whisper"]
    analysis = segment.analysis or {}
    if isinstance(analysis.get(COUNTDOWN_EVENT_KEY), dict):
        tags.append("countdown_event")
    values: list[Any] = []
    raw_style_tags = analysis.get("style_tags")
    if isinstance(raw_style_tags, list):
        values.extend(raw_style_tags)
    elif raw_style_tags is not None:
        values.append(raw_style_tags)
    raw_audio_style = analysis.get("audio_style")
    if isinstance(raw_audio_style, dict):
        raw_audio_style_tags = raw_audio_style.get("style_tags")
        if isinstance(raw_audio_style_tags, list):
            values.extend(raw_audio_style_tags)
        elif raw_audio_style_tags is not None:
            values.append(raw_audio_style_tags)
    voice_training = analysis.get("voice_training")
    if isinstance(voice_training, dict):
        effect_tags = voice_training.get("effect_tags")
        if isinstance(effect_tags, list):
            values.extend(effect_tags)
        elif effect_tags is not None:
            values.append(effect_tags)
    for value in values:
        _append_korean_script_tag(tags, value)
    return tags


def _korean_script_style_value(
    analysis: dict[str, Any],
    key: str,
    allowed: set[str],
    default: str,
) -> str:
    value = _normalize_korean_script_tag(analysis.get(key))
    return value if value in allowed else default


def _korean_script_ref_style(emotion: str, spatial_style: str, style_tags: list[str]) -> str:
    if emotion == "sleepy" or spatial_style == "sleepy_center" or "sleepy" in style_tags:
        return "sleepy"
    return "whisper_close"


def _coerce_korean_script_cues(raw_cues: Any) -> list[NonverbalCue]:
    cues: list[NonverbalCue] = []
    if not isinstance(raw_cues, list):
        return cues
    for raw_cue in raw_cues:
        if isinstance(raw_cue, NonverbalCue):
            cues.append(raw_cue)
            continue
        if isinstance(raw_cue, dict):
            kind = str(
                raw_cue.get("kind")
                or raw_cue.get("type")
                or raw_cue.get("label")
                or raw_cue.get("name")
                or "style"
            ).strip()
            if not kind:
                continue
            try:
                cues.append(
                    NonverbalCue(
                        kind=kind,
                        source_text=str(raw_cue.get("source_text") or raw_cue.get("text") or ""),
                        normalized_text=str(
                            raw_cue.get("normalized_text")
                            or raw_cue.get("normalized")
                            or raw_cue.get("text")
                            or kind
                        ),
                        position=int(raw_cue.get("position") or 0),
                        intensity=float(raw_cue.get("intensity") or 0.5),
                        pause_sec=raw_cue.get("pause_sec"),
                        notes=str(raw_cue.get("notes") or ""),
                    )
                )
            except (TypeError, ValueError):
                continue
            continue
        token = str(raw_cue).strip()
        if token:
            cues.append(NonverbalCue(kind="style", source_text=token, normalized_text=token))
    return cues


def _dedupe_korean_script_cues(cues: list[NonverbalCue]) -> list[NonverbalCue]:
    seen: set[tuple[str, str, str, int]] = set()
    out: list[NonverbalCue] = []
    for cue in cues:
        key = (cue.kind, cue.source_text, cue.normalized_text, cue.position)
        if key in seen:
            continue
        seen.add(key)
        out.append(cue)
    return out


def _adapt_korean_tts_text_for_performance(text: str) -> tuple[str, list[str]]:
    adapted = colloquialize_korean_text(text)
    reasons: list[str] = []
    if adapted != text.strip():
        reasons.append(COLLOQUIAL_REWRITE_NOTE)
    return adapted, reasons


def _should_defer_dense_korean_micro_segment(
    *,
    duration_sec: float,
    estimated_tts_duration_sec: float,
    speech_chars: int,
    countdown_values: list[int] | None,
) -> bool:
    if countdown_values is not None or duration_sec >= _KOREAN_MICRO_SEGMENT_MAX_SEC:
        return False
    if speech_chars <= 2 and estimated_tts_duration_sec <= max(0.4, duration_sec * _KOREAN_MICRO_SEGMENT_MAX_RATIO):
        return False
    return estimated_tts_duration_sec > max(
        duration_sec * _KOREAN_MICRO_SEGMENT_MAX_RATIO,
        duration_sec + 0.25,
    )


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
    no_speech_detected = 0
    safety_blocked = 0
    timing_over_budget = 0
    micro_segments_deferred = 0
    started_at = monotonic()
    last_logged_at = started_at
    for index, segment in enumerate(manifest.segments, start=1):
        if segment.status in NO_SPEECH_STATUSES:
            no_speech_detected += 1
            segment.script = None
            last_logged_at = _log_segment_progress(
                "korean-script", index, total, segment, manifest, started_at, last_logged_at
            )
            continue
        if segment.status in SKIP_STATUSES:
            if segment.status == "needs_manual_review" and _can_retry_korean_script_segment(segment):
                segment.errors = [
                    error
                    for error in segment.errors
                    if not _is_recoverable_korean_preflight_error(error)
                ]
            else:
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
        countdown_values = _countdown_values_for_segment(segment)
        text, adaptation_reasons = _adapt_korean_tts_text_for_performance(text) if text else ("", [])
        normalized = normalize_korean_tts_text(text) if text else None
        if not text or normalized is None or not normalized.text:
            needs_manual_review += 1
            segment.status = "needs_manual_review"
            segment.errors.append("Cannot build Korean TTS script without translation_ko.ko_natural.")
            last_logged_at = _log_segment_progress(
                "korean-script", index, total, segment, manifest, started_at, last_logged_at
            )
            continue
        timing_budget = korean_tts_timing_budget(segment.duration, source_text)
        speech_chars = korean_tts_speech_char_count(normalized.text)
        estimated_tts_duration_sec = (
            segment.duration if countdown_values is not None else estimate_tts_duration(normalized.text, "ko")
        )
        over_budget = False if countdown_values is not None else speech_chars > int(timing_budget["max_speech_chars"])
        if over_budget:
            timing_over_budget += 1
        segment.analysis["korean_tts_timing"] = {
            **timing_budget,
            "speech_chars": speech_chars,
            "estimated_tts_duration_sec": round(estimated_tts_duration_sec, 3),
            "over_budget": over_budget,
        }
        if adaptation_reasons:
            segment.analysis["korean_tts_adaptation"] = {
                "tts_text_before": translation.ko_natural.strip() if translation else "",
                "tts_text_after": normalized.text,
                "reasons": adaptation_reasons,
            }
        if _should_defer_dense_korean_micro_segment(
            duration_sec=segment.duration,
            estimated_tts_duration_sec=estimated_tts_duration_sec,
            speech_chars=speech_chars,
            countdown_values=countdown_values,
        ):
            needs_manual_review += 1
            micro_segments_deferred += 1
            segment.status = "needs_manual_review"
            segment.script = None
            segment.analysis["korean_tts_micro_segment_policy"] = {
                "action": "merge_or_absorb_required",
                "reason": "dense_micro_segment",
                "duration_sec": round(segment.duration, 6),
                "estimated_tts_duration_sec": round(estimated_tts_duration_sec, 6),
                "speech_chars": speech_chars,
            }
            message = "Micro segment too dense for standalone Korean TTS; merge or absorb required."
            if message not in segment.errors:
                segment.errors.append(message)
            last_logged_at = _log_segment_progress(
                "korean-script", index, total, segment, manifest, started_at, last_logged_at
            )
            continue
        risk_flags = ["source_script_translated_to_ko", *normalized.risk_flags]
        if countdown_values is not None:
            risk_flags.append("countdown_event")
        if over_budget:
            risk_flags.append("korean_tts_timing_over_budget")
        analysis = segment.analysis or {}
        emotion = _korean_script_style_value(
            analysis,
            "emotion",
            _KOREAN_SCRIPT_EMOTIONS,
            "gentle",
        )
        pace = _korean_script_style_value(analysis, "pace", _KOREAN_SCRIPT_PACES, "slow")
        volume = _korean_script_style_value(analysis, "volume", _KOREAN_SCRIPT_VOLUMES, "soft")
        spatial_style = _korean_script_style_value(
            analysis,
            "spatial_style",
            _KOREAN_SCRIPT_SPATIAL_STYLES,
            "center",
        )
        style_tags = _korean_script_style_tags(segment)
        ref_style = _korean_script_ref_style(emotion, spatial_style, style_tags)
        nonverbal_cues = _dedupe_korean_script_cues(
            [*normalized.cues, *_coerce_korean_script_cues(analysis.get("nonverbal_cues"))]
        )
        segment.script = JapaneseScript(
            literal_ja=source_text,
            ja_text=source_text or normalized.text,
            tts_text=normalized.text,
            tts_language="ko",
            source_language=cfg.source_language,
            target_language=cfg.target_language,
            ref_style=ref_style,
            emotion=emotion,
            pace=pace,
            volume=volume,
            nonverbal_cues=nonverbal_cues,
            spatial_style=spatial_style,
            expected_tts_duration_sec=estimated_tts_duration_sec,
            style_tags=style_tags,
            risk_flags=risk_flags,
        )
        preflight = preflight_tts_text(
            segment.script,
            target_language=cfg.target_language,
            source_text=source_text,
            min_hangul_ratio=cfg.gsv_ko_text_min_hangul_ratio,
        )
        if preflight.issues == [
            "korean_tts_suspicious_truncated_sentence"
        ] and _can_soften_truncated_korean_tts(segment.script.tts_text):
            segment.script.tts_text = segment.script.tts_text.rstrip(" ,") + "..."
            preflight = preflight_tts_text(
                segment.script,
                target_language=cfg.target_language,
                source_text=source_text,
                min_hangul_ratio=cfg.gsv_ko_text_min_hangul_ratio,
            )
            segment.analysis["pre_synth_text_qc_recovery"] = "softened_truncated_sentence"
        segment.analysis["pre_synth_text_qc"] = preflight.as_payload()
        if preflight.blocked:
            needs_manual_review += 1
            if "tts_safety_minor_sexualized_content" in preflight.issues:
                safety_blocked += 1
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
    stage_status = "failed" if safety_blocked else "completed"
    mark_stage(
        manifest,
        "korean-script",
        stage_status,
        scripted=scripted,
        needs_manual_review=needs_manual_review,
        no_speech_detected=no_speech_detected,
        safety_blocked=safety_blocked,
        timing_over_budget=timing_over_budget,
        micro_segments_deferred=micro_segments_deferred,
    )
    save_manifest(project_dir, manifest)
    if safety_blocked:
        _log_stage_complete(
            "korean-script",
            manifest,
            f"safety_blocked={safety_blocked}",
        )
        raise ValueError(
            "korean-script blocked minor sexualized content before TTS synthesis "
            f"({safety_blocked} segment(s))."
        )
    _log_stage_complete("korean-script", manifest, "tts_language=ko")
    return ctx.update_manifest(manifest)


def _can_soften_truncated_korean_tts(text: str) -> bool:
    normalized = text.strip()
    if normalized.endswith(","):
        return True
    return bool(re.search(r"(?:그리고|그러니까|하지만|말고|자|뭐)$", normalized))


def _can_retry_korean_script_segment(segment: Segment) -> bool:
    translation = segment.translation_ko
    if translation is None or not translation.ko_natural.strip():
        return False
    if not segment.errors:
        return False
    return all(_is_recoverable_korean_preflight_error(error) for error in segment.errors)


def _is_recoverable_korean_preflight_error(error: str) -> bool:
    return error in {
        "Korean TTS preflight blocked synthesis: korean_tts_suspicious_truncated_sentence",
        "korean-script skipped segment status needs_manual_review.",
    }
