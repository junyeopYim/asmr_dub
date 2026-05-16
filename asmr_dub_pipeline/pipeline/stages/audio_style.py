from __future__ import annotations

# ruff: noqa: F403,F405,I001

from asmr_dub_pipeline.pipeline.context import PipelineContext
from asmr_dub_pipeline.pipeline.stages.common import *


_AUDIO_STYLE_ANALYSIS_KEYS = {
    "speech_style",
    "emotion",
    "pace",
    "volume",
    "nonverbal_cues",
    "spatial_style",
    "style_tags",
    "estimated_pan",
    "keep_original_texture",
    "risk_flags",
    "confidence",
    "voice_training",
    "effect_events",
}

_AUDIO_STYLE_EFFECT_TAGS = ("telephone", "radio", "robot", "distortion", "reverb", "echo")
_AUDIO_STYLE_EFFECT_TAG_SET = set(_AUDIO_STYLE_EFFECT_TAGS)
_AUDIO_STYLE_SCOPE_ALL = "all"
_AUDIO_STYLE_SCOPE_SPEAKER_SUSPICIOUS = "speaker_suspicious"
_AUDIO_STYLE_SCOPES = {_AUDIO_STYLE_SCOPE_ALL, _AUDIO_STYLE_SCOPE_SPEAKER_SUSPICIOUS}
_AUDIO_STYLE_LOW_SPEAKER_OVERLAP_RATIO = 0.6
_AUDIO_STYLE_LOW_CENTROID_SIMILARITY = 0.65
_AUDIO_STYLE_MINOR_SPEAKER_MAX_DURATION_SEC = 30.0
_AUDIO_STYLE_MINOR_SPEAKER_MAX_COUNT = 5
_AUDIO_STYLE_MINOR_SPEAKER_MAX_DOMINANT_RATIO = 0.1
_AUDIO_STYLE_EFFECT_ALIASES = {
    "phone": "telephone",
    "telephone_filter": "telephone",
    "telephone_voice": "telephone",
    "radio_voice": "radio",
    "walkie_talkie": "radio",
    "robot_voice": "robot",
    "robotic": "robot",
}
_AUDIO_STYLE_EVENT_TARGETS = {"voice", "background", "sfx", "mixed"}


def _normalize_audio_style_token(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _normalize_audio_style_scope(value: Any) -> str:
    scope = _normalize_audio_style_token(value or _AUDIO_STYLE_SCOPE_ALL)
    if scope in {"speaker", "speaker_suspicious", "speaker_suspect"}:
        scope = _AUDIO_STYLE_SCOPE_SPEAKER_SUSPICIOUS
    if scope not in _AUDIO_STYLE_SCOPES:
        allowed = ", ".join(sorted(_AUDIO_STYLE_SCOPES))
        raise ValueError(f"Unsupported audio-style scope '{value}'. Expected one of: {allowed}.")
    return scope


def _clamp_audio_style_float(value: Any, default: float, low: float, high: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(low, min(high, number))


def _normalize_audio_style_effect_tag(value: Any) -> str:
    tag = _normalize_audio_style_token(value)
    return _AUDIO_STYLE_EFFECT_ALIASES.get(tag, tag)


def _normalize_audio_style_effect_events(
    raw_events: Any,
    *,
    segment_duration: float,
) -> list[dict[str, Any]]:
    if not isinstance(raw_events, list):
        return []
    duration = max(0.0, float(segment_duration))
    events: list[dict[str, Any]] = []
    for raw_event in raw_events:
        if not isinstance(raw_event, dict):
            continue
        tag = _normalize_audio_style_effect_tag(raw_event.get("tag") or raw_event.get("name"))
        if tag == "none" or tag not in _AUDIO_STYLE_EFFECT_TAG_SET:
            continue
        target = _normalize_audio_style_token(raw_event.get("target") or "voice")
        if target not in _AUDIO_STYLE_EVENT_TARGETS:
            target = "voice"
        start_sec = _clamp_audio_style_float(raw_event.get("start_sec"), 0.0, 0.0, duration)
        end_raw = raw_event.get("end_sec")
        end_sec = (
            _clamp_audio_style_float(end_raw, duration, 0.0, duration)
            if end_raw is not None
            else duration
        )
        if end_sec <= start_sec:
            end_sec = duration
        params = raw_event.get("params")
        events.append(
            {
                "tag": tag,
                "target": target,
                "start_sec": round(start_sec, 6),
                "end_sec": round(max(start_sec, end_sec), 6),
                "intensity": _clamp_audio_style_float(raw_event.get("intensity"), 1.0, 0.0, 1.0),
                "confidence": _clamp_audio_style_float(raw_event.get("confidence"), 0.0, 0.0, 1.0),
                "params": dict(params) if isinstance(params, dict) else {},
            }
        )
    return events


def _normalize_audio_style_effect_tags(
    raw_tags: Any,
    effect_events: list[dict[str, Any]],
) -> list[str]:
    values = raw_tags if isinstance(raw_tags, list) else [raw_tags] if raw_tags is not None else []
    values.extend(event["tag"] for event in effect_events)
    tags: list[str] = []
    for value in values:
        tag = _normalize_audio_style_effect_tag(value)
        if (tag == "none" or tag in _AUDIO_STYLE_EFFECT_TAG_SET) and tag not in tags:
            tags.append(tag)
    effect_tags = [tag for tag in tags if tag in _AUDIO_STYLE_EFFECT_TAG_SET]
    return effect_tags or ["none"]


def _default_audio_style_effect_events(
    effect_tags: list[str],
    *,
    segment_duration: float,
    confidence: Any,
) -> list[dict[str, Any]]:
    if effect_tags == ["none"]:
        return []
    duration = max(0.0, float(segment_duration))
    return [
        {
            "tag": tag,
            "target": "voice",
            "start_sec": 0.0,
            "end_sec": round(duration, 6),
            "intensity": 1.0,
            "confidence": _clamp_audio_style_float(confidence, 0.0, 0.0, 1.0),
            "params": {},
        }
        for tag in effect_tags
    ]


def _merge_audio_style_analysis(existing: dict[str, Any], payload: dict[str, Any], segment: Segment) -> dict[str, Any]:
    payload = dict(payload)
    voice_training = dict(payload.get("voice_training") or {})
    effect_events = _normalize_audio_style_effect_events(
        payload.get("effect_events"),
        segment_duration=segment.duration,
    )
    effect_tags = _normalize_audio_style_effect_tags(
        voice_training.get("effect_tags"),
        effect_events,
    )
    if effect_tags == ["none"]:
        effect_events = []
    elif not effect_events:
        effect_events = _default_audio_style_effect_events(
            effect_tags,
            segment_duration=segment.duration,
            confidence=payload.get("confidence"),
        )
    voice_training["effect_tags"] = effect_tags
    payload["voice_training"] = voice_training
    payload["effect_events"] = effect_events
    merged = dict(existing)
    for key in _AUDIO_STYLE_ANALYSIS_KEYS:
        if key in payload:
            merged[key] = payload[key]
    merged["audio_style"] = {
        "backend_task": "audio_style",
        "effect_tags": effect_tags,
        "effect_events": effect_events,
        "confidence": payload.get("confidence"),
    }
    return merged


def _has_reusable_audio_style_analysis(segment: Segment) -> bool:
    audio_style = segment.analysis.get("audio_style")
    if not isinstance(audio_style, dict):
        return False
    return audio_style.get("backend_task") == "audio_style" and "effect_tags" in audio_style


def _segment_has_audio_style_effect(segment: Segment) -> bool:
    audio_style = segment.analysis.get("audio_style")
    effect_tags: Any = None
    if isinstance(audio_style, dict):
        effect_tags = audio_style.get("effect_tags")
    if effect_tags is None:
        voice_training = segment.analysis.get("voice_training")
        if isinstance(voice_training, dict):
            effect_tags = voice_training.get("effect_tags")
    if not isinstance(effect_tags, list):
        return False
    return any(str(tag) != "none" for tag in effect_tags)


def _audio_style_float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _audio_style_minor_speaker_ids(segments: list[Segment]) -> set[str]:
    stats: dict[str, dict[str, float]] = {}
    for segment in segments:
        if not segment.speaker_id:
            continue
        item = stats.setdefault(segment.speaker_id, {"count": 0.0, "duration": 0.0})
        item["count"] += 1.0
        item["duration"] += max(0.0, float(segment.duration))
    if len(stats) < 2:
        return set()
    dominant_duration = max(item["duration"] for item in stats.values())
    if dominant_duration <= 0:
        return set()
    minor_ids: set[str] = set()
    for speaker_id, item in stats.items():
        is_small = (
            item["duration"] < _AUDIO_STYLE_MINOR_SPEAKER_MAX_DURATION_SEC
            or item["count"] <= _AUDIO_STYLE_MINOR_SPEAKER_MAX_COUNT
        )
        is_minor_relative_to_dominant = (
            item["duration"] <= dominant_duration * _AUDIO_STYLE_MINOR_SPEAKER_MAX_DOMINANT_RATIO
        )
        if is_small and is_minor_relative_to_dominant:
            minor_ids.add(speaker_id)
    return minor_ids


def _audio_style_speaker_suspicion(
    segment: Segment,
    *,
    minor_speaker_ids: set[str],
) -> tuple[int, float, list[str]]:
    analysis = segment.analysis or {}
    reasons: list[str] = []
    score = 0
    rank_distance = 1.0

    normalization = analysis.get("source_speaker_bucket_normalization")
    if isinstance(normalization, Mapping):
        reasons.append("source_speaker_bucket_normalization")
        score += 3
        if _normalize_audio_style_token(normalization.get("merge_confidence")) == "low":
            reasons.append("low_confidence_bucket_merge")
            score += 3
        centroid_similarity = _audio_style_float_or_none(normalization.get("centroid_similarity"))
        if centroid_similarity is not None:
            rank_distance = min(rank_distance, centroid_similarity)
            if centroid_similarity < _AUDIO_STYLE_LOW_CENTROID_SIMILARITY:
                reasons.append("low_centroid_similarity")
                score += 2
        clean_duration = _audio_style_float_or_none(normalization.get("clean_training_duration_sec"))
        if (
            clean_duration is not None
            and clean_duration < _AUDIO_STYLE_MINOR_SPEAKER_MAX_DURATION_SEC
        ):
            reasons.append("minor_bucket_clean_duration")
            score += 2

    assignment = analysis.get("source_speaker_assignment")
    if isinstance(assignment, Mapping):
        dominant_overlap_ratio = _audio_style_float_or_none(
            assignment.get("dominant_overlap_ratio")
        )
        if dominant_overlap_ratio is not None:
            rank_distance = min(rank_distance, dominant_overlap_ratio)
        if not segment.speaker_id and dominant_overlap_ratio is not None:
            reasons.append("missing_speaker_id")
            score += 2
            if dominant_overlap_ratio < _AUDIO_STYLE_LOW_SPEAKER_OVERLAP_RATIO:
                reasons.append("low_dominant_overlap")
                score += 1
        speaker_count = assignment.get("speaker_count")
        if isinstance(speaker_count, int) and speaker_count > 1:
            reasons.append("overlapping_speakers")
            score += 2
    elif not segment.speaker_id:
        reasons.append("missing_speaker_assignment")
        score += 2

    if segment.speaker_id and segment.speaker_id in minor_speaker_ids:
        reasons.append("minor_speaker_bucket")
        score += 2

    fallback = analysis.get("source_speaker_model_fallback")
    if isinstance(fallback, Mapping):
        reasons.append("source_speaker_model_fallback")
        score += 2

    return score, rank_distance, reasons


def _audio_style_speaker_suspicious_segments(
    segments: list[Segment],
) -> tuple[set[str], list[str], dict[str, list[str]]]:
    minor_speaker_ids = _audio_style_minor_speaker_ids(segments)
    selected: list[tuple[int, float, int, str, list[str]]] = []
    reasons_by_id: dict[str, list[str]] = {}
    for index, segment in enumerate(segments):
        if segment.status in SKIP_STATUSES:
            continue
        score, rank_distance, reasons = _audio_style_speaker_suspicion(
            segment,
            minor_speaker_ids=minor_speaker_ids,
        )
        if not reasons:
            continue
        selected.append((score, rank_distance, index, segment.id, reasons))
        reasons_by_id[segment.id] = reasons
    selected.sort(key=lambda item: (-item[0], item[1], item[2]))
    selected_ids = [segment_id for _score, _rank_distance, _index, segment_id, _reasons in selected]
    return set(selected_ids), selected_ids, reasons_by_id


def _audio_style_backend_config(cfg: Any, model_id: str | None) -> dict[str, Any]:
    config = _gemma_backend_config(cfg, model_id)
    config["llama_cpp_n_predict"] = min(int(config.get("llama_cpp_n_predict", 1024)), 384)
    config["llama_cpp_ctx_size"] = min(int(config.get("llama_cpp_ctx_size", 4096)), 4096)
    return config


def run_audio_style_stage(
    ctx: PipelineContext,
    backend_kind: str,
    model_id: str | None = None,
    confirm_rights: bool = False,
    force: bool = False,
    scope: str = _AUDIO_STYLE_SCOPE_ALL,
) -> PipelineManifest:
    project_dir = ctx.project_dir
    manifest = ctx.reload_manifest()
    _load_config_into_manifest(project_dir, manifest)
    backend_kind = backend_kind.replace("-", "_")
    scope = _normalize_audio_style_scope(scope)
    selected_segment_ids: set[str] | None = None
    selected_segments: list[str] = []
    selected_reasons: dict[str, list[str]] = {}
    if scope == _AUDIO_STYLE_SCOPE_SPEAKER_SUSPICIOUS:
        selected_segment_ids, selected_segments, selected_reasons = (
            _audio_style_speaker_suspicious_segments(manifest.segments)
        )
    total = len(manifest.segments)
    candidate_total = len(selected_segment_ids) if selected_segment_ids is not None else total
    _log_stage_start("audio-style", f"backend={backend_kind}, scope={scope}, segments={total}")
    _require_audio_stage_rights(manifest, "audio-style", confirm_rights, metadata={"backend": backend_kind})
    cfg = manifest.project_config
    backend = None
    style_client = None
    server_manager = None
    server_metadata = None
    audio_style_worker_count = (
        _effective_lane_count(cfg.gemma_audio_style_concurrency, candidate_total)
        if backend_kind == "llama_server_audio" and candidate_total
        else 1
    )

    def ensure_backend_ready() -> None:
        nonlocal backend, style_client, server_manager
        if backend is not None or style_client is not None:
            return
        if backend_kind == "llama_server_audio":
            base_url = cfg.gemma_text_server_url.rstrip("/")
            server_manager = ManagedGemmaTextServer(
                enabled=cfg.gemma_text_server_auto_start,
                base_url=base_url,
                command=(
                    _gemma_text_server_command(
                        cfg,
                        base_url=base_url,
                        lane_index=0,
                        include_mmproj=True,
                        parallel_slots=audio_style_worker_count,
                    )
                    if cfg.gemma_text_server_auto_start
                    else []
                ),
                log_path=project_dir / "work" / "audio_style" / "llama_server.log",
                startup_timeout_sec=cfg.gemma_text_server_startup_timeout_sec,
                shutdown_timeout_sec=cfg.gemma_text_server_shutdown_timeout_sec,
            )
            server_manager.start()
            style_client = LlamaServerTranslationClient(
                base_url,
                timeout_sec=cfg.gemma_text_timeout_sec,
                retries=cfg.gemma_text_retries,
                n_predict=min(cfg.gemma_text_n_predict, 384),
                model=cfg.gemma_llama_cpp_audio_model_path,
                two_pass=False,
            )
        else:
            backend = create_gemma_backend(backend_kind, _audio_style_backend_config(cfg, model_id))

    context = {"source_language": cfg.source_language, "target_language": cfg.target_language}
    styled = 0
    tagged_segments = 0
    failed = 0
    no_speech_detected = 0
    skipped_existing = 0
    skipped_scope = 0
    started_at = monotonic()
    last_logged_at = started_at

    def apply_audio_style_payload(segment: Segment, raw_payload: dict[str, Any]) -> None:
        nonlocal styled, tagged_segments
        payload = validate_gemma_task_response(
            "audio_style",
            raw_payload,
        )
        segment.analysis = _merge_audio_style_analysis(segment.analysis, payload, segment)
        if payload.get("estimated_pan") is not None:
            segment.estimated_pan = float(payload["estimated_pan"])
        if payload.get("keep_original_texture") is not None:
            segment.keep_original_texture = bool(payload["keep_original_texture"])
        effect_tags = (segment.analysis.get("voice_training") or {}).get("effect_tags") or []
        if any(tag != "none" for tag in effect_tags):
            tagged_segments += 1
        styled += 1

    def warn_audio_style_failure(segment: Segment, exc: Exception) -> None:
        nonlocal failed
        failed += 1
        warning = f"audio-style skipped {segment.id}: {exc}"
        if warning not in manifest.warnings:
            manifest.warnings.append(warning)

    def collect_parallel_audio_style_segments(start_cursor: int) -> list[tuple[int, Segment, Path]]:
        window: list[tuple[int, Segment, Path]] = []
        scan_cursor = start_cursor
        while scan_cursor < total and len(window) < audio_style_worker_count:
            candidate = manifest.segments[scan_cursor]
            if candidate.status in SKIP_STATUSES:
                break
            if selected_segment_ids is not None and candidate.id not in selected_segment_ids:
                break
            if not force and _has_reusable_audio_style_analysis(candidate):
                break
            try:
                _validate_segment_audio_paths(project_dir, candidate, check_formats=True)
            except Exception:
                if window:
                    break
                raise
            window.append((scan_cursor, candidate, Path(candidate.audio_for_gemma)))
            scan_cursor += 1
        return window

    def analyze_audio_style_segment(item: tuple[int, Segment, Path]) -> tuple[int, Segment, dict[str, Any]]:
        item_index, item_segment, item_audio_path = item
        if style_client is None:
            raise RuntimeError("audio-style client was not initialized")
        return item_index, item_segment, style_client.analyze_audio_style(item_audio_path, item_segment)

    try:
        cursor = 0
        while cursor < total:
            index = cursor + 1
            segment = manifest.segments[cursor]
            if segment.status in NO_SPEECH_STATUSES:
                no_speech_detected += 1
                last_logged_at = _log_segment_progress(
                    "audio-style", index, total, segment, manifest, started_at, last_logged_at
                )
                cursor += 1
                continue
            if segment.status in SKIP_STATUSES:
                last_logged_at = _log_segment_progress(
                    "audio-style", index, total, segment, manifest, started_at, last_logged_at
                )
                cursor += 1
                continue
            if selected_segment_ids is not None and segment.id not in selected_segment_ids:
                skipped_scope += 1
                last_logged_at = _log_segment_progress(
                    "audio-style", index, total, segment, manifest, started_at, last_logged_at
                )
                cursor += 1
                continue
            if not force and _has_reusable_audio_style_analysis(segment):
                skipped_existing += 1
                if _segment_has_audio_style_effect(segment):
                    tagged_segments += 1
                last_logged_at = _log_segment_progress(
                    "audio-style", index, total, segment, manifest, started_at, last_logged_at
                )
                cursor += 1
                continue
            try:
                ensure_backend_ready()
                if style_client is not None:
                    if audio_style_worker_count > 1:
                        window = collect_parallel_audio_style_segments(cursor)
                        if len(window) > 1:
                            results: dict[int, dict[str, Any] | Exception] = {}
                            with ThreadPoolExecutor(max_workers=len(window)) as executor:
                                future_map = {
                                    executor.submit(analyze_audio_style_segment, item): item
                                    for item in window
                                }
                                for future in as_completed(future_map):
                                    item_index, _item_segment, _item_audio_path = future_map[future]
                                    try:
                                        _result_index, _result_segment, raw_result = future.result()
                                    except Exception as exc:
                                        results[item_index] = exc
                                    else:
                                        results[item_index] = raw_result
                            for item_index, item_segment, _item_audio_path in window:
                                result = results.get(item_index)
                                if isinstance(result, Exception):
                                    warn_audio_style_failure(item_segment, result)
                                elif isinstance(result, dict):
                                    try:
                                        apply_audio_style_payload(item_segment, result)
                                    except Exception as exc:
                                        warn_audio_style_failure(item_segment, exc)
                                else:
                                    warn_audio_style_failure(
                                        item_segment,
                                        RuntimeError("audio-style worker returned no result"),
                                    )
                                last_logged_at = _log_segment_progress(
                                    "audio-style",
                                    item_index + 1,
                                    total,
                                    item_segment,
                                    manifest,
                                    started_at,
                                    last_logged_at,
                                )
                            cursor += len(window)
                            continue
                    _validate_segment_audio_paths(project_dir, segment, check_formats=True)
                    audio_path = Path(segment.audio_for_gemma)
                    raw_payload = style_client.analyze_audio_style(audio_path, segment)
                else:
                    _validate_segment_audio_paths(project_dir, segment, check_formats=True)
                    audio_path = Path(segment.audio_for_gemma)
                    if backend is None:
                        raise RuntimeError("audio-style backend was not initialized")
                    raw_payload = backend.analyze_audio_style(audio_path, segment, context)
                apply_audio_style_payload(segment, raw_payload)
            except Exception as exc:
                warn_audio_style_failure(segment, exc)
            last_logged_at = _log_segment_progress(
                "audio-style", index, total, segment, manifest, started_at, last_logged_at
            )
            cursor += 1
    finally:
        if server_manager is not None:
            server_metadata = {
                "auto_start": cfg.gemma_text_server_auto_start,
                "base_url": server_manager.base_url,
                "server_count": 1,
                "mode": "single_server_audio",
                "parallel_slots": audio_style_worker_count,
                "started": bool(server_manager.started),
                "reused_existing": bool(server_manager.reused_existing),
            }
            server_manager.stop()

    out_path = project_dir / "work" / "segments" / "manifests" / "segments_audio_style.json"
    write_json_atomic(out_path, {"segments": [s.model_dump(mode="json") for s in manifest.segments]})
    manifest.artifacts["segments_audio_style"] = str(out_path)
    mark_stage(
        manifest,
        "audio-style",
        "completed",
        backend=backend_kind,
        styled=styled,
        tagged_segments=tagged_segments,
        failed=failed,
        no_speech_detected=no_speech_detected,
        skipped_existing=skipped_existing,
        skipped_scope=skipped_scope,
        scope=scope,
        selected_segments=selected_segments,
        selected_segment_reasons=selected_reasons,
        concurrency=audio_style_worker_count,
        server=server_metadata,
        segment_counts=_segment_counts(manifest),
    )
    save_manifest(project_dir, manifest)
    _log_stage_complete("audio-style", manifest, f"styled={styled}, tagged={tagged_segments}, failed={failed}")
    return ctx.update_manifest(manifest)
