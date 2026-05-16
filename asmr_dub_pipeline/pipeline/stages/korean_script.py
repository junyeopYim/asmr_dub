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
from asmr_dub_pipeline.script.numeric_cadence import periodize_korean_numeric_cadence_text


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
_KOREAN_SHORT_SAME_SPEAKER_MERGE_MAX_SEC = 1.5
_KOREAN_MICRO_SEGMENT_MAX_RATIO = 1.5
_KOREAN_MICRO_ABSORB_MAX_SPEECH_CHARS = 8
_KOREAN_MICRO_ABSORB_GAP_SEC = 3.0
_KOREAN_MICRO_ABSORB_RISK_FLAG = "korean_tts_absorbed_micro_segment"
_KOREAN_SHORT_SAME_SPEAKER_ABSORB_RISK_FLAG = "korean_tts_absorbed_short_same_speaker_segment"
_KOREAN_TEXTURE_KEEP_ORIGINAL_ERROR = "korean_script_repeated_texture_keep_original"
_KOREAN_TEXTURE_TTS_TOKEN_RE = re.compile(
    r"(그으+|으으+|아아+|어어+|오오+|우우+|이이+|으음+|음음+|하아+|후우+|흐으+|"
    r"스으+|스르르|파르르|부르르|덜덜|쿵|톡|탁|훅)"
)
_KOREAN_LONG_VOWEL_TEXTURE_MIN_CHARS = 4
_SOURCE_TEXTURE_TOKEN_RE = re.compile(
    r"(グー+ッ?|ぎゅー+|スー+ッ?|ふっ|ドロドロ|ダラーン|びくん|ビクン|ぞくぞく|ピクピク)",
    re.IGNORECASE,
)
_KOREAN_PURE_NUMERIC_COUNTDOWN_TOKEN_RE = re.compile(
    r"(하나|둘|셋|넷|다섯|여섯|일곱|여덟|아홉|열|영|공|"
    r"일|이|삼|사|오|육|칠|팔|구|십|[0-9０-９]+)"
)
_KOREAN_NUMERIC_COUNTDOWN_SEPARATORS_RE = re.compile(r"[\s,.;:!?，。！？、…·~\-]+")
_KOREAN_TEXTURE_INCIDENTAL_COUNT_TOKEN_RE = re.compile(
    r"(하나|한|둘|두|셋|세|넷|네|다섯|여섯|일곱|여덟|아홉|열|영|공|"
    r"일|이|삼|사|오|육|칠|팔|구|십|[0-9０-９]+|번씩|회씩|번|회)"
)
_KOREAN_TEXTURE_ALLOWED_REMAINDER_RE = re.compile(r"[\s,.;:!?，。！？、…·~\-]+")


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


def _is_pure_korean_numeric_countdown_text(text: str) -> bool:
    remaining = _KOREAN_PURE_NUMERIC_COUNTDOWN_TOKEN_RE.sub(
        "",
        _KOREAN_NUMERIC_COUNTDOWN_SEPARATORS_RE.sub("", text.strip()),
    )
    return not remaining


def _is_pure_or_nearly_pure_korean_texture_text(text: str) -> bool:
    remaining = _KOREAN_TEXTURE_TTS_TOKEN_RE.sub("", text.strip())
    remaining = _KOREAN_TEXTURE_INCIDENTAL_COUNT_TOKEN_RE.sub("", remaining)
    remaining = _KOREAN_TEXTURE_ALLOWED_REMAINDER_RE.sub("", remaining)
    return not remaining


def _post_translation_texture_reason(
    *,
    source_text: str,
    tts_text: str,
    countdown_values: list[int] | None,
) -> str | None:
    source_texture_tokens = _SOURCE_TEXTURE_TOKEN_RE.findall(source_text)
    tts_texture_tokens = _KOREAN_TEXTURE_TTS_TOKEN_RE.findall(tts_text)
    if countdown_values is not None or _is_pure_korean_numeric_countdown_text(tts_text):
        return None
    if not tts_texture_tokens:
        return None
    if _is_pure_or_nearly_pure_korean_texture_text(tts_text):
        return "repeated_vowel_or_onomatopoeia_after_translation"
    source_backed_texture = bool(source_texture_tokens) and (
        len(tts_texture_tokens) >= 2
        or any(len(token) >= _KOREAN_LONG_VOWEL_TEXTURE_MIN_CHARS for token in tts_texture_tokens)
    )
    if not source_backed_texture:
        return None
    return "repeated_vowel_or_onomatopoeia_after_translation"


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


def _join_korean_script_text(left: str, right: str) -> str:
    parts = [part.strip() for part in (left, right) if part and part.strip()]
    return " ".join(parts)


def _korean_script_candidate_text(segment: Segment) -> str:
    if segment.script and _canonical_language(segment.script.tts_language) == "ko":
        return segment.script.tts_text.strip()
    translation = segment.translation_ko
    text = translation.ko_natural.strip() if translation else ""
    if not text:
        return ""
    adapted, _ = _adapt_korean_tts_text_for_performance(text)
    normalized = normalize_korean_tts_text(adapted)
    return normalized.text if normalized is not None else ""


def _merge_korean_translation_text(
    target: KoreanTranslation,
    absorbed: KoreanTranslation,
    *,
    side: str,
) -> KoreanTranslation:
    if side == "left":
        literal = _join_korean_script_text(target.ko_literal, absorbed.ko_literal)
        natural = _join_korean_script_text(target.ko_natural, absorbed.ko_natural)
    else:
        literal = _join_korean_script_text(absorbed.ko_literal, target.ko_literal)
        natural = _join_korean_script_text(absorbed.ko_natural, target.ko_natural)
    notes = list(dict.fromkeys([*target.notes, *absorbed.notes]))
    return target.model_copy(update={"ko_literal": literal, "ko_natural": natural, "notes": notes})


def _merge_korean_source_script(
    target: SourceScript | None,
    absorbed: SourceScript | None,
    *,
    side: str,
    start: float,
    end: float,
) -> SourceScript | None:
    if target is None and absorbed is None:
        return None
    if target is None:
        assert absorbed is not None
        return absorbed.model_copy(update={"start": start, "end": end})
    if absorbed is None:
        return target.model_copy(update={"start": start, "end": end})
    if side == "left":
        text = _join_korean_script_text(target.text, absorbed.text)
    else:
        text = _join_korean_script_text(absorbed.text, target.text)
    confidence = None
    if target.confidence is not None and absorbed.confidence is not None:
        confidence = min(target.confidence, absorbed.confidence)
    elif target.confidence is not None:
        confidence = target.confidence
    elif absorbed.confidence is not None:
        confidence = absorbed.confidence
    return target.model_copy(update={"text": text, "start": start, "end": end, "confidence": confidence})


def _korean_micro_absorption_candidate(
    segments: list[Segment],
    index: int,
    *,
    side: str,
    micro_text: str,
    source_text: str,
    max_segment_sec: float,
    require_same_speaker: bool = False,
    strict_timing_fit: bool = True,
) -> tuple[float, str, int] | None:
    segment = segments[index]
    target_index = index - 1 if side == "left" else index + 1
    if target_index < 0 or target_index >= len(segments):
        return None
    target = segments[target_index]
    if target.status in SKIP_STATUSES or target.translation_ko is None:
        return None
    if require_same_speaker:
        if not segment.speaker_id or target.speaker_id != segment.speaker_id:
            return None
    elif target.speaker_id and segment.speaker_id and target.speaker_id != segment.speaker_id:
        return None
    if _countdown_values_for_segment(target) is not None:
        return None
    target_text = _korean_script_candidate_text(target)
    if not target_text:
        return None
    if side == "left":
        gap = segment.start - target.end
        combined_start = target.start
        combined_end = segment.end
        combined_text = _join_korean_script_text(target_text, micro_text)
        combined_source = _join_korean_script_text(
            target.source_script.text if target.source_script else "",
            source_text,
        )
    else:
        gap = target.start - segment.end
        combined_start = segment.start
        combined_end = target.end
        combined_text = _join_korean_script_text(micro_text, target_text)
        combined_source = _join_korean_script_text(
            source_text,
            target.source_script.text if target.source_script else "",
        )
    if gap < 0 or gap > _KOREAN_MICRO_ABSORB_GAP_SEC:
        return None
    combined_duration = combined_end - combined_start
    if combined_duration <= 0 or combined_duration > max_segment_sec:
        return None
    normalized = normalize_korean_tts_text(combined_text)
    if normalized is None or not normalized.text:
        return None
    if strict_timing_fit:
        budget = korean_tts_timing_budget(combined_duration, combined_source)
        speech_chars = korean_tts_speech_char_count(normalized.text)
        duration_budget_chars = max(6, int(combined_duration * 4.0))
        if speech_chars > max(int(budget["max_speech_chars"]), duration_budget_chars):
            return None
        combined_estimated_tts_sec = estimate_tts_duration(normalized.text, "ko")
        if combined_estimated_tts_sec > max(combined_duration * 1.12, combined_duration + 0.35):
            return None
    speaker_penalty = 0.0 if target.speaker_id == segment.speaker_id else 0.1
    side_penalty = 0.0 if side == "left" else 0.01
    return (speaker_penalty + gap + side_penalty, side, target_index)


def _find_korean_micro_absorption_target(
    segments: list[Segment],
    index: int,
    *,
    micro_text: str,
    source_text: str,
    speech_chars: int,
    max_segment_sec: float,
) -> tuple[str, int] | None:
    if speech_chars > _KOREAN_MICRO_ABSORB_MAX_SPEECH_CHARS:
        return None
    candidates = [
        candidate
        for side in ("left", "right")
        if (
            candidate := _korean_micro_absorption_candidate(
                segments,
                index,
                side=side,
                micro_text=micro_text,
                source_text=source_text,
                max_segment_sec=max_segment_sec,
            )
        )
        is not None
    ]
    if not candidates:
        return None
    _, side, target_index = min(candidates, key=lambda item: item[0])
    return side, target_index


def _find_korean_short_same_speaker_absorption_target(
    segments: list[Segment],
    index: int,
    *,
    micro_text: str,
    source_text: str,
    duration_sec: float,
    countdown_values: list[int] | None,
    max_segment_sec: float,
) -> tuple[str, int] | None:
    segment = segments[index]
    if (
        countdown_values is not None
        or duration_sec >= _KOREAN_SHORT_SAME_SPEAKER_MERGE_MAX_SEC
        or not segment.speaker_id
    ):
        return None
    candidates = [
        candidate
        for side in ("left", "right")
        if (
            candidate := _korean_micro_absorption_candidate(
                segments,
                index,
                side=side,
                micro_text=micro_text,
                source_text=source_text,
                max_segment_sec=max_segment_sec,
                require_same_speaker=True,
                strict_timing_fit=False,
            )
        )
        is not None
    ]
    if not candidates:
        return None
    _, side, target_index = min(candidates, key=lambda item: item[0])
    return side, target_index


def _apply_korean_micro_absorption(
    *,
    segments: list[Segment],
    index: int,
    side: str,
    target_index: int,
    normalized_text: str,
    estimated_tts_duration_sec: float,
    speech_chars: int,
    reason: str = "dense_micro_segment",
    risk_flag: str = _KOREAN_MICRO_ABSORB_RISK_FLAG,
) -> None:
    segment = segments[index]
    target = segments[target_index]
    absorbed_translation = segment.translation_ko
    new_translation = target.translation_ko
    if new_translation is not None and absorbed_translation is not None:
        new_translation = _merge_korean_translation_text(
            new_translation,
            absorbed_translation,
            side=side,
        )
    if side == "left":
        combined_start = target.start
        combined_end = segment.end
    else:
        combined_start = segment.start
        combined_end = target.end
    new_source_script = _merge_korean_source_script(
        target.source_script,
        segment.source_script,
        side=side,
        start=combined_start,
        end=combined_end,
    )
    target = target.model_copy(
        update={
            "translation_ko": new_translation,
            "source_script": new_source_script,
            "start": round(combined_start, 6),
            "end": round(combined_end, 6),
            "duration": round(combined_end - combined_start, 6),
        }
    )
    segments[target_index] = target
    if target.script is not None and _canonical_language(target.script.tts_language) == "ko":
        if side == "left":
            combined_tts_text = _join_korean_script_text(target.script.tts_text, normalized_text)
        else:
            combined_tts_text = _join_korean_script_text(normalized_text, target.script.tts_text)
        normalized = normalize_korean_tts_text(combined_tts_text)
        if normalized is not None and normalized.text:
            target.script.tts_text = normalized.text
            if target.source_script is not None:
                target.script.ja_text = target.source_script.text
                target.script.literal_ja = target.source_script.text
            target.script.expected_tts_duration_sec = estimate_tts_duration(normalized.text, "ko")
            target.script.risk_flags = list(
                dict.fromkeys([*target.script.risk_flags, risk_flag])
            )
    absorbed_payload = {
        "absorbed_segment_id": segment.id,
        "absorbed_text": normalized_text,
        "absorbed_source_text": segment.source_script.text if segment.source_script else "",
        "estimated_tts_duration_sec": round(estimated_tts_duration_sec, 6),
        "speech_chars": speech_chars,
        "side": side,
        "reason": reason,
        "target_span_sec": round(target.duration, 6),
    }
    target.analysis.setdefault("korean_tts_absorbed_segments", []).append(absorbed_payload)
    segment.status = "absorbed"
    segment.script = None
    segment.analysis["korean_tts_absorption"] = {
        "action": "absorbed_into_neighbor",
        "absorbed_into_segment_id": target.id,
        "side": side,
    }
    segment.analysis["korean_tts_micro_segment_policy"] = {
        "action": "absorbed_into_neighbor",
        "reason": reason,
        "duration_sec": round(segment.duration, 6),
        "estimated_tts_duration_sec": round(estimated_tts_duration_sec, 6),
        "speech_chars": speech_chars,
    }


def run_korean_script_stage(
    ctx: PipelineContext,
    confirm_rights: bool = False,
    only_segment_ids: set[str] | None = None,
) -> PipelineManifest:
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
    absorbed_micro_segments = 0
    non_speech_texture = 0
    started_at = monotonic()
    last_logged_at = started_at
    for index, segment in enumerate(manifest.segments, start=1):
        if only_segment_ids is not None and segment.id not in only_segment_ids:
            continue
        if segment.status in NO_SPEECH_STATUSES:
            no_speech_detected += 1
            segment.script = None
            last_logged_at = _log_segment_progress(
                "korean-script", index, total, segment, manifest, started_at, last_logged_at
            )
            continue
        if segment.status in SKIP_STATUSES:
            if segment.status == "absorbed":
                segment.script = None
                last_logged_at = _log_segment_progress(
                    "korean-script", index, total, segment, manifest, started_at, last_logged_at
                )
                continue
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
        texture_reason = _post_translation_texture_reason(
            source_text=source_text,
            tts_text=normalized.text,
            countdown_values=countdown_values,
        )
        if texture_reason is not None:
            non_speech_texture += 1
            segment.status = "non_speech_texture"
            segment.keep_original_texture = True
            segment.script = None
            segment.analysis["korean_script_non_speech_texture"] = {
                "action": "keep_original_texture",
                "reason": texture_reason,
                "source_text": source_text,
                "tts_text": normalized.text,
            }
            if _KOREAN_TEXTURE_KEEP_ORIGINAL_ERROR not in segment.errors:
                segment.errors.append(_KOREAN_TEXTURE_KEEP_ORIGINAL_ERROR)
            last_logged_at = _log_segment_progress(
                "korean-script", index, total, segment, manifest, started_at, last_logged_at
            )
            continue
        numeric_periodization_metadata = None
        if (
            _canonical_language(cfg.target_language) == "ko"
            and countdown_values is None
            and bool(getattr(cfg, "gsv_numeric_cadence_periods_enabled", True))
        ):
            periodized_text, numeric_periodization_metadata = periodize_korean_numeric_cadence_text(
                normalized.text,
                min_values=int(getattr(cfg, "gsv_numeric_cadence_min_values", 3)),
            )
            if numeric_periodization_metadata is not None and periodized_text != normalized.text:
                before_periodization = normalized.text
                normalized = normalize_korean_tts_text(periodized_text)
                segment.analysis["korean_numeric_cadence_periodization"] = {
                    "before": before_periodization,
                    "after": normalized.text,
                    **numeric_periodization_metadata,
                }
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
        short_same_speaker_absorption_target = _find_korean_short_same_speaker_absorption_target(
            manifest.segments,
            index - 1,
            micro_text=normalized.text,
            source_text=source_text,
            duration_sec=segment.duration,
            countdown_values=countdown_values,
            max_segment_sec=float(cfg.asr_resegment_max_sec),
        )
        if short_same_speaker_absorption_target is not None:
            side, target_index = short_same_speaker_absorption_target
            _apply_korean_micro_absorption(
                segments=manifest.segments,
                index=index - 1,
                side=side,
                target_index=target_index,
                normalized_text=normalized.text,
                estimated_tts_duration_sec=estimated_tts_duration_sec,
                speech_chars=speech_chars,
                reason="short_same_speaker_segment",
                risk_flag=_KOREAN_SHORT_SAME_SPEAKER_ABSORB_RISK_FLAG,
            )
            absorbed_micro_segments += 1
            last_logged_at = _log_segment_progress(
                "korean-script", index, total, segment, manifest, started_at, last_logged_at
            )
            continue
        if _should_defer_dense_korean_micro_segment(
            duration_sec=segment.duration,
            estimated_tts_duration_sec=estimated_tts_duration_sec,
            speech_chars=speech_chars,
            countdown_values=countdown_values,
        ):
            absorption_target = _find_korean_micro_absorption_target(
                manifest.segments,
                index - 1,
                micro_text=normalized.text,
                source_text=source_text,
                speech_chars=speech_chars,
                max_segment_sec=float(cfg.asr_resegment_max_sec),
            )
            if absorption_target is not None:
                side, target_index = absorption_target
                _apply_korean_micro_absorption(
                    segments=manifest.segments,
                    index=index - 1,
                    side=side,
                    target_index=target_index,
                    normalized_text=normalized.text,
                    estimated_tts_duration_sec=estimated_tts_duration_sec,
                    speech_chars=speech_chars,
                )
                absorbed_micro_segments += 1
                last_logged_at = _log_segment_progress(
                    "korean-script", index, total, segment, manifest, started_at, last_logged_at
                )
                continue
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
        if numeric_periodization_metadata is not None:
            risk_flags.append("korean_numeric_cadence_periodized")
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
        if preflight.issues == ["korean_tts_suspicious_truncated_sentence"]:
            recovery_label: str | None = None
            if _can_soften_truncated_korean_tts(segment.script.tts_text):
                repaired_text = segment.script.tts_text.rstrip(" ,") + "..."
                recovery_label = "softened_truncated_sentence"
            else:
                repaired_text, repaired = repair_suspicious_truncated_korean_tts_text(
                    segment.script.tts_text
                )
                if repaired:
                    recovery_label = "repaired_truncated_sentence"
            if recovery_label is not None:
                segment.script.tts_text = repaired_text
                segment.script.expected_tts_duration_sec = estimate_tts_duration(
                    repaired_text,
                    "ko",
                )
                segment.script.risk_flags = list(
                    dict.fromkeys([*segment.script.risk_flags, recovery_label])
                )
                preflight = preflight_tts_text(
                    segment.script,
                    target_language=cfg.target_language,
                    source_text=source_text,
                    min_hangul_ratio=cfg.gsv_ko_text_min_hangul_ratio,
                )
                segment.analysis["pre_synth_text_qc_recovery"] = recovery_label
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
        non_speech_texture=non_speech_texture,
        safety_blocked=safety_blocked,
        timing_over_budget=timing_over_budget,
        micro_segments_deferred=micro_segments_deferred,
        absorbed_micro_segments=absorbed_micro_segments,
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
        "Micro segment too dense for standalone Korean TTS; merge or absorb required.",
        "korean-script skipped segment status needs_manual_review.",
    }
