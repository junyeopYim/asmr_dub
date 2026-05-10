from __future__ import annotations

# ruff: noqa: F403,F405,I001

import copy
import re

from asmr_dub_pipeline.pipeline.context import PipelineContext
from asmr_dub_pipeline.pipeline.stages.common import *

_OPEN_KOREAN_TTS_END_RE = re.compile(r"(?:\s*(?:[,，、]+|\.{2,}|…+))+\s*$")
_KOREAN_SENTENCE_FINAL_RE = re.compile(
    r"(?:[.!?。！？]|요|다|죠|네|까|니다|세요|어요|아요|해요|예요|이에요)\s*$"
)
_KOREAN_TOKEN_RE = re.compile(r"[가-힣]+")
_OMISSION_EXPECTED_RATIO_THRESHOLD = 0.35
_OMISSION_SEGMENT_RATIO_THRESHOLD = 0.25
_KOREAN_COUNTING_SEPARATOR_RE = re.compile(r"^[\s,，、・:;-]*$")
_KOREAN_COUNTING_FILLER_RE = re.compile(r"^\s*말이에요[.!?。！？]*\s*$")
_KOREAN_COUNTING_TOKEN_TO_SPOKEN = {
    "0": "영",
    "1": "하나",
    "2": "둘",
    "3": "셋",
    "4": "넷",
    "5": "다섯",
    "6": "여섯",
    "7": "일곱",
    "8": "여덟",
    "9": "아홉",
    "10": "열",
    "영": "영",
    "공": "영",
    "일": "하나",
    "이": "둘",
    "삼": "셋",
    "사": "넷",
    "오": "다섯",
    "육": "여섯",
    "칠": "일곱",
    "팔": "여덟",
    "구": "아홉",
    "십": "열",
    **{text: text for text in NATIVE_KOREAN_COUNT_ONES.values()},
    **{text: text for text in NATIVE_KOREAN_COUNT_TENS.values()},
}
_KOREAN_COUNTING_TOKEN_RE = re.compile(
    r"(?<![0-9A-Za-z가-힣])("
    + "|".join(
        re.escape(token)
        for token in sorted(_KOREAN_COUNTING_TOKEN_TO_SPOKEN, key=len, reverse=True)
    )
    + r")(?![0-9A-Za-z가-힣])"
)


def _close_open_korean_tts_sentence(text: str) -> tuple[str, bool]:
    stripped = text.strip()
    if not stripped or not _OPEN_KOREAN_TTS_END_RE.search(stripped):
        return stripped, False
    base = _OPEN_KOREAN_TTS_END_RE.sub("", stripped).strip()
    if not base:
        return stripped, False
    if _KOREAN_SENTENCE_FINAL_RE.search(base):
        return f"{base}.", True
    tokens = _KOREAN_TOKEN_RE.findall(base)
    if stripped.endswith("…") and tokens:
        short_interjection = len(tokens) == 1 and len(tokens[0]) <= 2
        repeated_sound = len(tokens) > 1 and all(len(token) <= 2 for token in tokens)
        if short_interjection or repeated_sound:
            return stripped, False
    return f"{base} 말이에요.", True


def _gsv_omission_detection_reasons(
    *,
    duration_sec: float,
    target_duration_sec: float,
    expected_tts_duration_sec: float,
    duration_gate: str,
    audio_gate: str,
    language_contract_ok: bool,
) -> list[str]:
    if (
        duration_gate != "too_short"
        or audio_gate != "pass"
        or not language_contract_ok
        or duration_sec <= 0
    ):
        return []
    reasons: list[str] = []
    if expected_tts_duration_sec > 0:
        ratio = duration_sec / expected_tts_duration_sec
        if ratio < _OMISSION_EXPECTED_RATIO_THRESHOLD:
            reasons.append(
                "duration_below_expected_ratio:"
                f"{ratio:.3f}<{_OMISSION_EXPECTED_RATIO_THRESHOLD:.3f}"
            )
    if target_duration_sec > 0:
        ratio = duration_sec / target_duration_sec
        if ratio < _OMISSION_SEGMENT_RATIO_THRESHOLD:
            reasons.append(
                "duration_below_segment_ratio:"
                f"{ratio:.3f}<{_OMISSION_SEGMENT_RATIO_THRESHOLD:.3f}"
            )
    return reasons


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _gsv_candidate_edge_silence_score(candidate: TTSCandidate, segment: Segment) -> float:
    trim = candidate.payload.get("postprocess", {}).get("edge_silence_trim", {})
    if not isinstance(trim, dict):
        return 1.0
    leading = _safe_float(trim.get("leading_trim_sec"))
    trailing = _safe_float(trim.get("trailing_trim_sec"))
    total_trim = max(0.0, leading) + max(0.0, trailing)
    if total_trim <= 0.0 or segment.duration <= 0.0:
        return 1.0
    return max(0.0, 1.0 - min(total_trim / segment.duration, 1.0))


def _gsv_candidate_speed_score(candidate: TTSCandidate) -> float:
    speed = _safe_float(candidate.payload.get("speed_factor"), 1.0)
    return max(0.0, 1.0 - min(abs(speed - 1.0) / 0.35, 1.0))


def _gsv_candidate_style_score(candidate: TTSCandidate, segment: Segment) -> float:
    requested = str(candidate.payload.get("requested_ref_style") or "").strip()
    resolved = str(candidate.payload.get("resolved_ref_style") or "").strip()
    if candidate.payload.get("fallback_used"):
        return 0.82
    if requested and resolved and requested != resolved:
        return 0.88
    if segment.script and segment.script.ref_style and resolved and segment.script.ref_style != resolved:
        return 0.92
    return 1.0


def _gsv_candidate_rescue_score(candidate: TTSCandidate) -> float:
    reason = candidate.selection_reason or ""
    if "time_fit" in reason:
        return 0.58
    if "pause_padding" in reason:
        return 0.62
    if "rescue" in reason or candidate.payload.get("rescue"):
        return 0.72
    if reason == "duration_or_language_contract_failed":
        return 0.45
    return 1.0


def _gsv_candidate_audio_score(candidate: TTSCandidate) -> float:
    return 1.0 if candidate.payload.get("audio_qc", {}).get("gate") == "pass" else 0.0


def _gsv_candidate_duration_score(candidate: TTSCandidate, segment: Segment) -> float:
    ratio = candidate.duration_ratio
    if ratio is None and candidate.duration_sec is not None:
        ratio = duration_ratio(candidate.duration_sec, segment.duration)
    if ratio is None:
        return 0.0
    return max(0.0, 1.0 - min(abs(float(ratio) - 1.0), 1.0))


def _gsv_candidate_selection_components(
    candidate: TTSCandidate,
    segment: Segment,
) -> dict[str, float]:
    return {
        "duration": _gsv_candidate_duration_score(candidate, segment),
        "audio": _gsv_candidate_audio_score(candidate),
        "style": _gsv_candidate_style_score(candidate, segment),
        "speed": _gsv_candidate_speed_score(candidate),
        "edge_silence": _gsv_candidate_edge_silence_score(candidate, segment),
        "rescue": _gsv_candidate_rescue_score(candidate),
    }


def _gsv_candidate_selection_score(candidate: TTSCandidate, segment: Segment) -> float:
    components = _gsv_candidate_selection_components(candidate, segment)
    score = (
        components["duration"] * 0.45
        + components["audio"] * 0.10
        + components["style"] * 0.15
        + components["speed"] * 0.15
        + components["edge_silence"] * 0.10
        + components["rescue"] * 0.05
    )
    return round(max(0.0, min(score, 1.0)), 6)


def _update_gsv_candidate_selection_scores(
    candidates: list[TTSCandidate],
    segment: Segment,
) -> None:
    for candidate in candidates:
        if candidate.error or candidate.duration_sec is None:
            continue
        components = _gsv_candidate_selection_components(candidate, segment)
        candidate.selection_score = _gsv_candidate_selection_score(candidate, segment)
        candidate.payload["selection_scoring"] = {
            key: round(value, 6) for key, value in components.items()
        }
        candidate.payload["selection_scoring"]["score"] = candidate.selection_score


def _select_gsv_candidate_for_mix(
    candidates: list[TTSCandidate],
    segment: Segment,
) -> TTSCandidate:
    if not candidates:
        raise ValueError("No TTS candidates to select.")
    _update_gsv_candidate_selection_scores(candidates, segment)
    return max(
        candidates,
        key=lambda candidate: (
            candidate.selection_score if candidate.selection_score is not None else 0.0,
            _gsv_candidate_duration_score(candidate, segment),
        ),
    )


def _compact_korean_counting_tts_text(text: str) -> tuple[str, dict[str, Any] | None]:
    stripped = text.strip()
    if not stripped:
        return stripped, None
    matches = list(_KOREAN_COUNTING_TOKEN_RE.finditer(stripped))
    if len(matches) < 3:
        return stripped, None

    groups: list[list[re.Match[str]]] = []
    current: list[re.Match[str]] = [matches[0]]
    for match in matches[1:]:
        separator = stripped[current[-1].end() : match.start()]
        if _KOREAN_COUNTING_SEPARATOR_RE.fullmatch(separator):
            current.append(match)
        else:
            if len(current) >= 3:
                groups.append(current)
            current = [match]
    if len(current) >= 3:
        groups.append(current)
    if not groups:
        return stripped, None

    replacements: list[dict[str, Any]] = []
    output_parts: list[str] = []
    last_index = 0
    for group in groups:
        start = group[0].start()
        end = group[-1].end()
        before = stripped[start:end]
        after = "".join(_KOREAN_COUNTING_TOKEN_TO_SPOKEN[match.group(1)] for match in group)
        output_parts.append(stripped[last_index:start])
        output_parts.append(after)
        replacements.append(
            {
                "before": before,
                "after": after,
                "token_count": len(group),
            }
        )
        last_index = end
    output_parts.append(stripped[last_index:])
    compacted = "".join(output_parts).strip()

    removed_counting_filler = False
    if len(groups) == 1:
        group = groups[0]
        prefix = stripped[: group[0].start()]
        suffix = stripped[group[-1].end() :]
        if not prefix.strip() and _KOREAN_COUNTING_FILLER_RE.fullmatch(suffix):
            compacted = replacements[0]["after"] + "."
            removed_counting_filler = True

    if compacted == stripped:
        return stripped, None
    metadata: dict[str, Any] = {
        "runs": replacements,
        "removed_counting_filler": removed_counting_filler,
    }
    return compacted, metadata


def _source_countdown_values(segment: Segment) -> list[int] | None:
    event = segment.analysis.get(COUNTDOWN_EVENT_KEY)
    if isinstance(event, dict):
        raw_values = event.get("values")
        if isinstance(raw_values, list) and all(isinstance(value, int) for value in raw_values):
            return [int(value) for value in raw_values]
    source_text = ""
    if segment.source_script is not None:
        source_text = segment.source_script.text
    elif segment.script is not None:
        source_text = segment.script.ja_text or segment.script.literal_ja
    return source_countdown_values(source_text)


def _is_strict_descending_countdown(values: list[int]) -> bool:
    return is_descending_countdown(values)


def _countdown_spoken_tokens(values: list[int]) -> list[str] | None:
    return countdown_korean_tokens(values)


def _countdown_phrase_chunks(tokens: list[str]) -> list[list[str]]:
    chunks: list[list[str]] = []
    index = 0
    while index < len(tokens):
        remaining = len(tokens) - index
        if remaining == 1 and chunks:
            chunks[-1].append(tokens[index])
            break
        if remaining == 3:
            chunks.append(tokens[index : index + 3])
            break
        take = min(2, remaining)
        chunks.append(tokens[index : index + take])
        index += take
    return chunks


def _countdown_chunk_label(text: str) -> str:
    label = re.sub(r"[^0-9A-Za-z가-힣]+", "_", text).strip("_")
    return label or "chunk"


def _fit_audio_frames(data: np.ndarray, target_frames: int) -> np.ndarray:
    target_frames = max(1, int(target_frames))
    if len(data) == target_frames:
        return data
    return resample_linear(data, max(1, len(data)), target_frames)


def _countdown_stereo(data: np.ndarray) -> np.ndarray:
    if data.ndim == 1:
        data = data[:, None]
    if data.shape[1] == 1:
        return np.repeat(data, 2, axis=1)
    return data[:, :2]


def _omission_reasons_allow_source_pause_padding(candidate: TTSCandidate) -> list[str] | None:
    if not candidate.payload.get("omission_suspected"):
        return []
    reasons = [
        str(reason)
        for reason in candidate.payload.get("omission_detection", {}).get("reasons") or []
    ]
    if reasons and all(reason.startswith("duration_below_segment_ratio:") for reason in reasons):
        return reasons
    return None


def _duration_rewrite_log_preview(value: object, max_chars: int = 140) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "..."


def _log_duration_rewrite_result(segment_id: str, metadata: dict[str, Any]) -> None:
    accepted = str(bool(metadata.get("accepted"))).lower()
    before = _duration_rewrite_log_preview(metadata.get("before"))
    after = _duration_rewrite_log_preview(metadata.get("after") or metadata.get("normalized_text"))
    parts = [
        f"[cyan]synth duration-rewrite[/cyan] {escape(segment_id)}",
        f"reason={escape(str(metadata.get('reason') or 'unknown'))}",
        f"accepted={accepted}",
        (
            "chars="
            f"{metadata.get('current_speech_chars', '?')}->{metadata.get('speech_chars', '?')} "
            f"target={metadata.get('target_speech_chars', '?')} "
            f"range={metadata.get('min_speech_chars', '?')}-{metadata.get('max_speech_chars', '?')}"
        ),
        f'before="{escape(before)}"',
    ]
    if metadata.get("accepted_relaxed"):
        parts.append(f"relaxed={escape(str(metadata.get('relaxed_acceptance_reason') or 'true'))}")
    if "retry_scheduled" in metadata:
        retry = str(bool(metadata.get("retry_scheduled"))).lower()
        parts.append(f"retry={retry}")
    if after:
        parts.append(f'after="{escape(after)}"')
    if metadata.get("rejected_reasons"):
        reasons = ", ".join(str(reason) for reason in metadata["rejected_reasons"])
        parts.append(f"rejected={escape(_duration_rewrite_log_preview(reasons))}")
    if metadata.get("error"):
        parts.append(f"error={escape(_duration_rewrite_log_preview(metadata['error']))}")
    console.print(" ".join(parts))


def _maybe_relax_duration_rewrite_acceptance(metadata: dict[str, Any]) -> bool:
    if metadata.get("accepted"):
        return True
    if metadata.get("error") or metadata.get("reason") != "too_short":
        return False
    rejected_reasons = [str(reason) for reason in metadata.get("rejected_reasons") or []]
    if not rejected_reasons or any(
        not reason.startswith("speech_chars_below_min:") for reason in rejected_reasons
    ):
        return False
    try:
        current_chars = int(metadata["current_speech_chars"])
        speech_chars = int(metadata["speech_chars"])
        min_chars = int(metadata["min_speech_chars"])
    except (KeyError, TypeError, ValueError):
        return False
    short_by = min_chars - speech_chars
    if short_by < 1 or short_by > 2 or speech_chars <= current_chars:
        return False
    metadata["accepted"] = True
    metadata["accepted_relaxed"] = True
    metadata["relaxed_acceptance_reason"] = f"speech_chars_below_min_near_miss:{speech_chars}<{min_chars}"
    metadata["duration_rewrite_relaxed_shortfall_chars"] = short_by
    metadata["original_rejected_reasons"] = rejected_reasons
    metadata["rejected_reasons"] = []
    return True


def _should_retry_duration_rewrite_result(
    rewritten: JapaneseScript | None,
    current_script: JapaneseScript | None,
) -> bool:
    return rewritten is not None and current_script is not None and rewritten.tts_text != current_script.tts_text


def run_synth_stage(ctx: PipelineContext, gsv_url: str | None, refs_path: Path, mock: bool = False, confirm_rights: bool = False, gpt_weights_path: str | None = None, sovits_weights_path: str | None = None, auto_gsv_server: bool | None = None, gsv_server_command: list[str] | str | None = None, use_trained_gpt: bool = False, only_segment_ids: set[str] | None = None, retry_failed: bool = False, force: bool = False) -> PipelineManifest:
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
    selected_total = (
        total
        if only_segment_ids is None
        else sum(1 for segment in manifest.segments if segment.id in only_segment_ids)
    )
    gsv_lane_count = 1 if mock else _effective_lane_count(cfg.gsv_concurrency, selected_total)
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
    duration_rewrite_enabled = (
        not mock
        and _canonical_language(cfg.target_language) == "ko"
        and getattr(cfg, "gsv_duration_rewrite_backend", "none") == "gemma"
        and int(getattr(cfg, "gsv_duration_rewrite_max_attempts", 0)) > 0
    )
    duration_rewrite_base_url = cfg.gemma_text_server_url.rstrip("/")
    duration_rewrite_manager = (
        ManagedGemmaTextServer(
            enabled=cfg.gemma_text_server_auto_start,
            base_url=duration_rewrite_base_url,
            command=(
                _gemma_text_server_command(cfg, base_url=duration_rewrite_base_url, lane_index=0)
                if cfg.gemma_text_server_auto_start
                else []
            ),
            log_path=project_dir / "work" / "gpt_sovits" / "duration_rewrite_llama_server.log",
            startup_timeout_sec=cfg.gemma_text_server_startup_timeout_sec,
            shutdown_timeout_sec=cfg.gemma_text_server_shutdown_timeout_sec,
        )
        if duration_rewrite_enabled
        else None
    )
    duration_rewrite_client: Any | None = None
    duration_rewrite_lock = Lock()
    model_switch: dict[str, Any] = {}
    gsv_servers_running = False
    duration_rewrite_running = False
    fine_tuned_retry_summary: dict[str, Any] | None = None
    static_ref_retry_summary: dict[str, Any] | None = None
    low_temperature_retry_summary: dict[str, Any] | None = None
    zero_shot_fallback_summary: dict[str, Any] | None = None
    try:
        clients: list[GPTSoVITSClient] = []
        if not mock:
            clients = [
                GPTSoVITSClient(base_url, cfg.gsv_timeout_sec, cfg.gsv_retries)
                for base_url in gsv_base_urls
            ]
        _validate_gsv_speaker_models(project_dir, manifest)
        gpt_weights = None
        sovits_weights = None
        if clients:
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
        started_at = monotonic()
        last_logged_at = started_at
        lane_locks = [Lock() for _ in range(gsv_lane_count)]
        lane_gpt_weights: list[str | None] = [None for _ in range(gsv_lane_count)]
        lane_sovits_weights: list[str | None] = [None for _ in range(gsv_lane_count)]
        speaker_refs_cache: dict[str, dict[str, GPTSoVITSRef]] = {}
        speaker_refs_cache_lock = Lock()

        def record_model_switch_instances(instances: list[dict[str, Any]]) -> None:
            if not instances:
                return
            if "instances" in model_switch:
                model_switch.setdefault("restarts", []).append(instances)
                return
            model_switch["instances"] = instances
            if len(instances) == 1:
                instance = instances[0]
                if "gpt_response" in instance:
                    model_switch["gpt_response"] = instance["gpt_response"]
                if "sovits_response" in instance:
                    model_switch["sovits_response"] = instance["sovits_response"]

        def start_gsv_servers() -> None:
            nonlocal gsv_servers_running, lane_gpt_weights, lane_sovits_weights
            if mock or gsv_servers_running:
                return
            for server_manager in server_managers:
                server_manager.start()
            gsv_servers_running = True
            lane_gpt_weights = [None for _ in range(gsv_lane_count)]
            lane_sovits_weights = [None for _ in range(gsv_lane_count)]
            switch_instances: list[dict[str, Any]] = []
            for lane_index, client in enumerate(clients):
                lane_switch: dict[str, Any] = {
                    "lane_index": lane_index,
                    "gsv_url": gsv_base_urls[lane_index],
                }
                if gpt_weights:
                    lane_switch["gpt_response"] = client.set_gpt_weights(gpt_weights)
                    lane_gpt_weights[lane_index] = gpt_weights
                if sovits_weights:
                    lane_switch["sovits_response"] = client.set_sovits_weights(sovits_weights)
                    lane_sovits_weights[lane_index] = sovits_weights
                switch_instances.append(lane_switch)
            record_model_switch_instances(switch_instances)

        def stop_gsv_servers() -> None:
            nonlocal gsv_servers_running
            if not gsv_servers_running:
                return
            for server_manager in reversed(server_managers):
                server_manager.stop()
            gsv_servers_running = False

        duration_rewrite_phase = "initial" if duration_rewrite_enabled else "normal"
        duration_rewrite_retry_segment_ids: set[str] = set()
        internal_retry_segment_ids: set[str] = set()
        static_ref_retry_segment_ids: set[str] = set()
        pass_candidate_count_override: int | None = None
        pass_temperature_override: float | None = None
        synth_pass = "fine_tuned_initial"

        def should_retry_failed_segment(segment: Segment) -> bool:
            return bool(
                segment.status == "failed"
                and segment.script
                and (retry_failed or segment.id in internal_retry_segment_ids)
            )

        def should_force_segment(segment: Segment) -> bool:
            return bool(force and segment.script and segment.status in {"synthesized", "failed"})

        def should_reset_previous_tts(segment: Segment) -> bool:
            return should_retry_failed_segment(segment) or should_force_segment(segment)

        def reset_previous_tts_attempt(segment: Segment) -> None:
            segment.status = "scripted"
            segment.tts = None
            segment.errors = [
                error
                for error in segment.errors
                if error
                not in {
                    "No acceptable TTS candidates for mix.",
                    "All TTS candidates failed.",
                    "Micro segment too short for Korean TTS.",
                }
                and not error.startswith("GPT-SoVITS synthesis failed")
                and not error.startswith("Korean TTS preflight blocked synthesis")
            ]

        def countdown_segment_values(segment: Segment) -> list[int] | None:
            if use_speaker_gsv:
                return None
            if not segment.script or _canonical_language(segment.script.tts_language) != "ko":
                return None
            if should_reset_previous_tts(segment):
                reset_previous_tts_attempt(segment)
            if segment.status == "synthesized" or segment.status in SKIP_STATUSES:
                return None
            values = _source_countdown_values(segment)
            if values is None:
                return None
            tokens = _countdown_spoken_tokens(values)
            if tokens is None:
                return None
            return values

        def countdown_spans_for_jobs(
            segment_jobs: list[tuple[int, Segment, int]],
        ) -> list[list[tuple[int, Segment, list[int]]]]:
            spans: list[list[tuple[int, Segment, list[int]]]] = []
            current: list[tuple[int, Segment, list[int]]] = []
            current_values: list[int] = []
            last_segment: Segment | None = None

            def flush_current() -> None:
                nonlocal current, current_values, last_segment
                if _is_strict_descending_countdown(current_values):
                    spans.append(current)
                current = []
                current_values = []
                last_segment = None

            for index, segment, _lane_index in segment_jobs:
                values = countdown_segment_values(segment)
                if values is None:
                    flush_current()
                    continue
                if current:
                    gap = max(0.0, segment.start - (last_segment.end if last_segment else segment.start))
                    if current_values[-1] - values[0] != 1 or gap > 2.0:
                        flush_current()
                current.append((index, segment, values))
                current_values.extend(values)
                last_segment = segment
            flush_current()
            return spans

        def token_audio_for_countdown(
            *,
            token_text: str,
            token_index: int,
            token_count: int,
            span_id: str,
            ref: GPTSoVITSRef,
            token_slot_sec: float,
            lane_index: int,
            tts_text_language: str,
            ref_style: str,
        ) -> tuple[np.ndarray, dict[str, Any]]:
            token_dir = project_dir / "work" / "tts" / "countdown" / "tokens"
            token_label = _countdown_chunk_label(token_text)
            raw_path = token_dir / f"{span_id}_chunk_{token_index:02d}_{token_label}.wav"
            fitted_path = token_dir / f"{span_id}_chunk_{token_index:02d}_{token_label}_fit.wav"
            speed = float(getattr(cfg, "gsv_countdown_token_speed_factor", cfg.gsv_tts_max_speed_factor))
            seed = cfg.base_seed + 70_000 + token_index
            options = GPTSoVITSTTSOptions(
                seed=seed,
                speed_factor=speed,
                text_lang=tts_text_language,
                top_k=cfg.gsv_top_k,
                top_p=cfg.gsv_top_p,
                temperature=float(getattr(cfg, "gsv_countdown_temperature", cfg.gsv_temperature)),
                text_split_method=cfg.gsv_text_split_method,
                fragment_interval=cfg.gsv_fragment_interval,
                parallel_infer=cfg.gsv_parallel_infer,
                repetition_penalty=cfg.gsv_repetition_penalty,
                sample_steps=cfg.gsv_sample_steps,
                super_sampling=cfg.gsv_super_sampling,
                overlap_length=cfg.gsv_overlap_length,
                min_chunk_length=cfg.gsv_min_chunk_length,
            )
            payload: dict[str, Any] = {
                "renderer": "countdown_phrase_timeline",
                "token_text": token_text,
                "token_index": token_index,
                "chunk_text": token_text,
                "chunk_index": token_index,
                "chunk_token_count": token_count,
                "ref_style": ref_style,
                "target_token_slot_sec": round(token_slot_sec, 6),
                "lane_index": lane_index,
                "gsv_url": None if mock else gsv_base_urls[lane_index],
            }
            payload.update(_tts_request_debug_payload(token_text, ref, options))
            if mock:
                _mock_synthesize(raw_path, max(0.12, min(0.42, token_slot_sec * 0.75)), options.seed, cfg.mix_sample_rate)
                payload["mock"] = True
            else:
                client = clients[lane_index]
                request = client.build_payload(token_text, ref, options)
                payload.update(request.as_payload())
                client.synthesize_to_file(request, raw_path)
            postprocess_tts_candidate(raw_path, payload)
            raw_duration = duration_sec(raw_path)
            data, sample_rate = load_audio(raw_path)
            data = _countdown_stereo(data)
            if sample_rate != cfg.mix_sample_rate:
                data = resample_linear(data, sample_rate, cfg.mix_sample_rate)
                sample_rate = cfg.mix_sample_rate
            token_target_sec = raw_duration
            if token_slot_sec > 0 and raw_duration > token_slot_sec:
                max_tempo = max(1.0, float(getattr(cfg, "gsv_countdown_max_tempo", 1.15)))
                token_target_sec = max(token_slot_sec, raw_duration / max_tempo)
            target_frames = max(1, int(round(token_target_sec * sample_rate)))
            fitted = _fit_audio_frames(data, target_frames)
            write_audio(fitted_path, fitted, sample_rate)
            fitted_duration = duration_sec(fitted_path)
            payload["countdown_token_fit"] = {
                "raw_path": str(raw_path),
                "raw_duration_sec": round(raw_duration, 6),
                "fitted_path": str(fitted_path),
                "fitted_duration_sec": round(fitted_duration, 6),
                "token_target_sec": round(token_target_sec, 6),
            }
            return fitted, payload

        def countdown_token_max_allowed_sec(token_slot_sec: float) -> float:
            absolute_max = float(getattr(cfg, "gsv_countdown_token_max_sec", 0.95))
            slot_occupancy = float(getattr(cfg, "gsv_countdown_token_max_slot_occupancy", 0.85))
            if token_slot_sec <= 0:
                return absolute_max
            return min(absolute_max, token_slot_sec * slot_occupancy)

        def countdown_token_candidate_score(candidate: dict[str, Any], token_slot_sec: float) -> float:
            duration = float(candidate["duration_sec"])
            min_sec = float(getattr(cfg, "gsv_countdown_token_min_sec", 0.25))
            max_sec = countdown_token_max_allowed_sec(token_slot_sec)
            target_sec = min(max_sec, max(min_sec, token_slot_sec * 0.58))
            if target_sec <= 0:
                return 0.0
            duration_score = max(0.0, 1.0 - min(abs(duration - target_sec) / target_sec, 1.0))
            peak = float(candidate["payload"].get("audio_qc", {}).get("peak_dbfs", -120.0))
            loudness_score = max(0.0, min((peak + 60.0) / 60.0, 1.0))
            return round(duration_score * 0.85 + loudness_score * 0.15, 6)

        def countdown_token_candidate_audio(
            *,
            token_text: str,
            token_index: int,
            candidate_index: int,
            span_id: str,
            ref: GPTSoVITSRef,
            token_slot_sec: float,
            lane_index: int,
            tts_text_language: str,
            ref_style: str,
        ) -> dict[str, Any]:
            token_dir = project_dir / "work" / "tts" / "countdown" / "tokens"
            token_label = _countdown_chunk_label(token_text)
            output_path = (
                token_dir
                / f"{span_id}_token_{token_index:02d}_cand_{candidate_index:02d}_{token_label}.wav"
            )
            speed = float(
                getattr(cfg, "gsv_countdown_token_speed_factor", cfg.gsv_tts_max_speed_factor)
            )
            seed = cfg.base_seed + 70_000 + token_index * 100 + candidate_index
            options = GPTSoVITSTTSOptions(
                seed=seed,
                speed_factor=speed,
                text_lang=tts_text_language,
                top_k=cfg.gsv_top_k,
                top_p=cfg.gsv_top_p,
                temperature=float(getattr(cfg, "gsv_countdown_temperature", cfg.gsv_temperature)),
                text_split_method=cfg.gsv_text_split_method,
                fragment_interval=0.0,
                parallel_infer=cfg.gsv_parallel_infer,
                repetition_penalty=cfg.gsv_repetition_penalty,
                sample_steps=cfg.gsv_sample_steps,
                super_sampling=cfg.gsv_super_sampling,
                overlap_length=cfg.gsv_overlap_length,
                min_chunk_length=cfg.gsv_min_chunk_length,
            )
            payload: dict[str, Any] = {
                "renderer": "countdown_token_timeline",
                "token_text": token_text,
                "token_index": token_index,
                "candidate_index": candidate_index,
                "ref_style": ref_style,
                "target_token_slot_sec": round(token_slot_sec, 6),
                "lane_index": lane_index,
                "gsv_url": None if mock else gsv_base_urls[lane_index],
            }
            payload.update(_tts_request_debug_payload(token_text, ref, options))
            if mock:
                duration = min(
                    countdown_token_max_allowed_sec(token_slot_sec) * 0.75,
                    max(float(getattr(cfg, "gsv_countdown_token_min_sec", 0.25)), token_slot_sec * 0.55),
                )
                _mock_synthesize(output_path, max(0.12, duration), options.seed, cfg.mix_sample_rate)
                payload["mock"] = True
            else:
                client = clients[lane_index]
                request = client.build_payload(token_text, ref, options)
                payload.update(request.as_payload())
                client.synthesize_to_file(request, output_path)

            raw_duration = duration_sec(output_path)
            postprocess_tts_candidate(output_path, payload)
            duration = duration_sec(output_path)
            token_min_sec = float(getattr(cfg, "gsv_countdown_token_min_sec", 0.25))
            token_max_sec = countdown_token_max_allowed_sec(token_slot_sec)
            if duration < token_min_sec:
                duration_gate = "too_short"
            elif duration > token_max_sec:
                duration_gate = "too_long"
            else:
                duration_gate = "pass"
            audio_metrics = {
                "gate": "pass",
                "peak_dbfs": round(peak_dbfs(output_path), 3),
                "rms_dbfs": round(rms_dbfs(output_path), 3),
            }
            payload["audio_qc"] = audio_metrics
            payload["countdown_token_candidate"] = {
                "raw_duration_sec": round(raw_duration, 6),
                "duration_sec": round(duration, 6),
                "duration_gate": duration_gate,
                "token_min_sec": round(token_min_sec, 6),
                "token_max_sec": round(token_max_sec, 6),
            }
            data, sample_rate = load_audio(output_path)
            data = _countdown_stereo(data)
            if sample_rate != cfg.mix_sample_rate:
                data = resample_linear(data, sample_rate, cfg.mix_sample_rate)
                sample_rate = cfg.mix_sample_rate
            candidate = {
                "candidate_index": candidate_index,
                "seed": seed,
                "text": token_text,
                "output_path": str(output_path),
                "duration_sec": duration,
                "duration_gate": duration_gate,
                "acceptable": duration_gate == "pass",
                "payload": payload,
                "audio": data,
            }
            candidate["selection_score"] = countdown_token_candidate_score(
                candidate,
                token_slot_sec,
            )
            return candidate

        def countdown_token_slots_for_span(
            span: list[tuple[int, Segment, list[int]]],
            sample_rate: int,
        ) -> tuple[list[dict[str, Any]], dict[str, int], int]:
            segment_frames: dict[str, int] = {}
            segment_offsets: dict[str, int] = {}
            offset = 0
            for _index, segment, _segment_values in span:
                frames = max(1, int(round(segment.duration * sample_rate)))
                segment_frames[segment.id] = frames
                segment_offsets[segment.id] = offset
                offset += frames
            total_frames = offset

            timeline_rows: list[dict[str, Any]] = []
            token_index = 0
            for _index, segment, segment_values in span:
                event = segment.analysis.get(COUNTDOWN_EVENT_KEY)
                raw_timeline = event.get("token_timeline") if isinstance(event, dict) else None
                if not isinstance(raw_timeline, list) or len(raw_timeline) != len(segment_values):
                    timeline_rows = []
                    break
                value_count = max(1, len(segment_values))
                segment_tokens = _countdown_spoken_tokens(segment_values) or []
                raw_values: list[int] = []
                for local_index, (value, raw_item) in enumerate(
                    zip(segment_values, raw_timeline, strict=True)
                ):
                    if not isinstance(raw_item, dict):
                        timeline_rows = []
                        break
                    raw_value = raw_item.get("value")
                    raw_start = raw_item.get("start")
                    raw_end = raw_item.get("end", raw_start)
                    if raw_value is None or raw_start is None:
                        timeline_rows = []
                        break
                    try:
                        raw_value_int = int(raw_value)
                        raw_start_sec = float(raw_start)
                        raw_end_sec = float(raw_end)
                    except (TypeError, ValueError):
                        timeline_rows = []
                        break
                    raw_values.append(raw_value_int)
                    local_start_sec = max(0.0, min(segment.duration, raw_start_sec - segment.start))
                    local_end_sec = max(local_start_sec, min(segment.duration, raw_end_sec - segment.start))
                    token_text = (
                        str(raw_item.get("korean_token") or "")
                        or (
                            segment_tokens[local_index]
                            if local_index < len(segment_tokens)
                            else str(value)
                        )
                    )
                    timeline_rows.append(
                        {
                            "segment_id": segment.id,
                            "value": value,
                            "text": token_text,
                            "token_index": token_index,
                            "local_index": local_index,
                            "source_text": str(raw_item.get("source_text") or ""),
                            "source_start_sec": round(raw_start_sec, 6),
                            "source_end_sec": round(raw_end_sec, 6),
                            "source_start_frame": segment_offsets[segment.id]
                            + int(round(local_start_sec * sample_rate)),
                            "source_end_frame": segment_offsets[segment.id]
                            + int(round(local_end_sec * sample_rate)),
                            "segment_end_frame": segment_offsets[segment.id]
                            + segment_frames[segment.id],
                            "equal_slot_frames": max(
                                1,
                                int(round(segment_frames[segment.id] / value_count)),
                            ),
                        }
                    )
                    token_index += 1
                if timeline_rows == [] or raw_values != segment_values:
                    timeline_rows = []
                    break

            if timeline_rows and len(timeline_rows) == sum(len(values) for _i, _s, values in span):
                placements = []
                for row_index, row in enumerate(timeline_rows):
                    slot_start = int(row["source_start_frame"])
                    next_start = (
                        int(timeline_rows[row_index + 1]["source_start_frame"])
                        if row_index + 1 < len(timeline_rows)
                        else int(row["segment_end_frame"])
                    )
                    source_period_frames = max(1, next_start - slot_start)
                    slot_frames = max(source_period_frames, int(row["equal_slot_frames"]))
                    placement = {
                        key: value
                        for key, value in row.items()
                        if key
                        not in {
                            "source_start_frame",
                            "source_end_frame",
                            "segment_end_frame",
                            "equal_slot_frames",
                        }
                    }
                    placement.update(
                        {
                            "slot_start_frame": slot_start,
                            "slot_end_frame": slot_start + slot_frames,
                            "slot_duration_sec": slot_frames / sample_rate,
                            "placement_anchor": "source_word_start",
                        }
                    )
                    placements.append(placement)
                return placements, segment_frames, total_frames

            placements: list[dict[str, Any]] = []
            token_index = 0
            for _index, segment, segment_values in span:
                frames = segment_frames[segment.id]
                offset = segment_offsets[segment.id]
                value_count = max(1, len(segment_values))
                segment_tokens = _countdown_spoken_tokens(segment_values) or []
                for local_index, value in enumerate(segment_values):
                    slot_start = offset + int(round(local_index * frames / value_count))
                    slot_end = offset + int(round((local_index + 1) * frames / value_count))
                    token_text = (
                        segment_tokens[local_index]
                        if local_index < len(segment_tokens)
                        else str(value)
                    )
                    placements.append(
                        {
                            "segment_id": segment.id,
                            "value": value,
                            "text": token_text,
                            "token_index": token_index,
                            "local_index": local_index,
                            "slot_start_frame": slot_start,
                            "slot_end_frame": max(slot_start + 1, slot_end),
                            "slot_duration_sec": max(slot_end - slot_start, 1) / sample_rate,
                        }
                    )
                    token_index += 1
            return placements, segment_frames, total_frames

        def render_countdown_span_token(span: list[tuple[int, Segment, list[int]]]) -> set[str]:
            first_index, first_segment, _first_values = span[0]
            if not first_segment.script:
                return set()
            values = [value for _index, _segment, segment_values in span for value in segment_values]
            tokens = _countdown_spoken_tokens(values)
            if tokens is None:
                return set()
            start_gsv_servers()
            span_id = "countdown_" + "_".join(segment.id for _index, segment, _values in span)
            span_dir = project_dir / "work" / "tts" / "countdown"
            span_dir.mkdir(parents=True, exist_ok=True)
            ref_style = first_segment.script.ref_style
            resolved_ref_style = ref_style if ref_style in refs else "whisper_close"
            ref = _ref_for_tts_language(resolve_ref(refs, ref_style), first_segment.script.tts_language)
            sample_rate = cfg.mix_sample_rate
            placements, segment_frames, total_frames = countdown_token_slots_for_span(
                span,
                sample_rate,
            )
            lane_index = _segment_lane_index(first_segment, first_index - 1, gsv_lane_count)
            candidate_count = int(getattr(cfg, "gsv_countdown_candidate_count", 8))
            span_audio = np.zeros((total_frames, 2), dtype=np.float32)
            placement_metadata: list[dict[str, Any]] = []
            all_candidates: list[dict[str, Any]] = []

            for placement in placements:
                token_candidates: list[dict[str, Any]] = []
                with lane_locks[lane_index]:
                    for candidate_index in range(candidate_count):
                        token_candidates.append(
                            countdown_token_candidate_audio(
                                token_text=str(placement["text"]),
                                token_index=int(placement["token_index"]),
                                candidate_index=candidate_index,
                                span_id=span_id,
                                ref=ref,
                                token_slot_sec=float(placement["slot_duration_sec"]),
                                lane_index=lane_index,
                                tts_text_language=_segment_tts_text_language(
                                    first_segment,
                                    cfg.target_language,
                                ),
                                ref_style=resolved_ref_style,
                            )
                        )
                accepted = [candidate for candidate in token_candidates if candidate["acceptable"]]
                all_candidates.extend(
                    {
                        key: value
                        for key, value in candidate.items()
                        if key != "audio"
                    }
                    for candidate in token_candidates
                )
                if not accepted:
                    skip_payload = {
                        "reason": "no_acceptable_countdown_token_candidate",
                        "renderer": "countdown_token_timeline",
                        "span_id": span_id,
                        "segment_ids": [segment.id for _index, segment, _values in span],
                        "failed_token": placement["text"],
                        "failed_value": placement["value"],
                        "slot_duration_sec": round(float(placement["slot_duration_sec"]), 6),
                        "candidates": [
                            {key: value for key, value in candidate.items() if key != "audio"}
                            for candidate in token_candidates
                        ],
                    }
                    for _index, segment, _values in span:
                        segment.analysis["countdown_renderer_skip"] = skip_payload
                    return set()
                selected = max(
                    accepted,
                    key=lambda candidate: (
                        float(candidate["selection_score"]),
                        -abs(float(candidate["duration_sec"]) - float(placement["slot_duration_sec"]) * 0.58),
                    ),
                )
                slot_start = int(placement["slot_start_frame"])
                slot_end = int(placement["slot_end_frame"])
                slot_frames = max(1, slot_end - slot_start)
                audio = selected["audio"]
                if placement.get("placement_anchor") == "source_word_start":
                    start_frame = min(slot_start, max(0, total_frames - len(audio)))
                else:
                    start_frame = slot_start + max(0, (slot_frames - len(audio)) // 2)
                end_frame = min(total_frames, start_frame + len(audio))
                source_frames = max(0, end_frame - start_frame)
                if source_frames:
                    span_audio[start_frame:end_frame] += audio[:source_frames]
                placement_metadata.append(
                    {
                        "segment_id": placement["segment_id"],
                        "value": placement["value"],
                        "text": placement["text"],
                        "token_index": placement["token_index"],
                        "slot_start_sec": round(slot_start / sample_rate, 6),
                        "slot_end_sec": round(slot_end / sample_rate, 6),
                        "slot_duration_sec": round(float(placement["slot_duration_sec"]), 6),
                        "placed_start_sec": round(start_frame / sample_rate, 6),
                        "placed_end_sec": round(end_frame / sample_rate, 6),
                        "selected_candidate_index": selected["candidate_index"],
                        "selected_duration_sec": round(float(selected["duration_sec"]), 6),
                        "selected_path": selected["output_path"],
                        "candidate_count": len(token_candidates),
                        "placement_anchor": placement.get("placement_anchor", "slot_center"),
                        "rejected_candidates": [
                            {
                                "candidate_index": candidate["candidate_index"],
                                "duration_sec": round(float(candidate["duration_sec"]), 6),
                                "duration_gate": candidate["duration_gate"],
                                "output_path": candidate["output_path"],
                            }
                            for candidate in token_candidates
                            if candidate is not selected
                        ],
                    }
                )

            peak = float(np.max(np.abs(span_audio))) if span_audio.size else 0.0
            if peak > 0.98:
                span_audio *= 0.98 / peak
            span_path = span_dir / f"{span_id}.wav"
            write_audio(span_path, span_audio, sample_rate)
            span_metadata = {
                "span_id": span_id,
                "renderer": "countdown_token_timeline",
                "segment_ids": [segment.id for _index, segment, _values in span],
                "values": values,
                "tokens": tokens,
                "target_duration_sec": round(total_frames / sample_rate, 6),
                "span_path": str(span_path),
                "token_placements": placement_metadata,
                "token_candidates": all_candidates,
                "source_language": cfg.source_language,
                "target_language": cfg.target_language,
                "ref_style": resolved_ref_style,
            }
            metadata_path = span_dir / f"{span_id}.json"
            write_json_atomic(metadata_path, span_metadata)

            offset = 0
            for _segment_index, (_index, segment, segment_values) in enumerate(span):
                frames = segment_frames[segment.id]
                segment_audio = span_audio[offset : offset + frames]
                offset += frames
                if len(segment_audio) < frames:
                    padding = np.zeros((frames - len(segment_audio), 2), dtype=np.float32)
                    segment_audio = np.concatenate([segment_audio, padding], axis=0)
                elif len(segment_audio) > frames:
                    segment_audio = segment_audio[:frames]
                final_path = project_dir / "work" / "tts" / f"{segment.id}_final.wav"
                write_audio(final_path, segment_audio, sample_rate)
                final_duration = duration_sec(final_path)
                final_ratio = duration_ratio(final_duration, segment.duration)
                segment_tokens = _countdown_spoken_tokens(segment_values) or []
                segment_placements = [
                    placement
                    for placement in placement_metadata
                    if placement["segment_id"] == segment.id
                ]
                payload = {
                    "renderer": "countdown_token_timeline",
                    "span_id": span_id,
                    "span_metadata_path": str(metadata_path),
                    "span_path": str(span_path),
                    "segment_id": segment.id,
                    "segment_values": segment_values,
                    "segment_tokens": segment_tokens,
                    "values": values,
                    "tokens": tokens,
                    "token_placements": segment_placements,
                    "target_duration_sec": segment.duration,
                    "duration_ratio": final_ratio,
                    "duration_gate": "pass",
                    "audio_qc": {
                        "gate": "pass",
                        "peak_dbfs": round(peak_dbfs(final_path), 3),
                        "rms_dbfs": round(rms_dbfs(final_path), 3),
                    },
                }
                candidate = TTSCandidate(
                    candidate_index=0,
                    seed=cfg.base_seed + 70_000 + first_index,
                    payload=payload,
                    output_path=str(final_path),
                    duration_sec=final_duration,
                    backend="gpt-sovits-countdown-renderer",
                    selected=True,
                    duration_ratio=final_ratio,
                    duration_gate="pass",
                    acceptable_for_mix=True,
                    selection_score=max(0.0, 1.0 - min(abs(final_ratio - 1.0), 1.0)),
                    selection_reason="countdown_token_timeline",
                    retry_summary={"countdown_renderer": True, "span_id": span_id},
                )
                segment.tts = TTSMetadata(
                    backend="gpt-sovits-countdown-renderer",
                    ref_style=resolved_ref_style,
                    speed_factor=float(
                        getattr(
                            cfg,
                            "gsv_countdown_token_speed_factor",
                            cfg.gsv_tts_max_speed_factor,
                        )
                    ),
                    candidate_count=1,
                    selected_candidate_path=str(final_path),
                    candidates=[candidate],
                    source_language=cfg.source_language,
                    target_language=cfg.target_language,
                    cross_lingual_voice_transfer=cfg.source_language != cfg.target_language,
                    retry_summary={
                        "countdown_renderer": True,
                        "countdown_renderer_mode": "token",
                        "span_id": span_id,
                        "span_metadata_path": str(metadata_path),
                        "selected_duration_gate": "pass",
                        "selected_acceptable_for_mix": True,
                        "selected_duration_ratio": final_ratio,
                    },
                )
                segment.analysis["countdown_renderer"] = {
                    "renderer": "countdown_token_timeline",
                    "span_id": span_id,
                    "segment_values": segment_values,
                    "segment_tokens": segment_tokens,
                    "token_placements": segment_placements,
                    "span_metadata_path": str(metadata_path),
                }
                segment.status = "synthesized"
            return {segment.id for _index, segment, _values in span}

        def render_countdown_span_compact(span: list[tuple[int, Segment, list[int]]]) -> set[str]:
            first_index, first_segment, _first_values = span[0]
            if not first_segment.script:
                return set()
            values = [value for _index, _segment, segment_values in span for value in segment_values]
            tokens = _countdown_spoken_tokens(values)
            if tokens is None:
                return set()
            start_gsv_servers()
            span_id = "countdown_" + "_".join(segment.id for _index, segment, _values in span)
            span_dir = project_dir / "work" / "tts" / "countdown"
            span_dir.mkdir(parents=True, exist_ok=True)
            ref_style = first_segment.script.ref_style
            resolved_ref_style = ref_style if ref_style in refs else "whisper_close"
            ref = _ref_for_tts_language(resolve_ref(refs, ref_style), first_segment.script.tts_language)
            total_duration_sec = sum(segment.duration for _index, segment, _values in span)
            sample_rate = cfg.mix_sample_rate
            total_frames = max(1, int(round(total_duration_sec * sample_rate)))
            compact_text = "".join(tokens)
            lane_index = _segment_lane_index(first_segment, first_index - 1, gsv_lane_count)
            with lane_locks[lane_index]:
                phrase_audio, phrase_payload = token_audio_for_countdown(
                    token_text=compact_text,
                    token_index=0,
                    token_count=len(tokens),
                    span_id=span_id,
                    ref=ref,
                    token_slot_sec=total_duration_sec,
                    lane_index=lane_index,
                    tts_text_language=_segment_tts_text_language(first_segment, cfg.target_language),
                    ref_style=resolved_ref_style,
                )
            if len(phrase_audio) > total_frames:
                skip_payload = {
                    "reason": "countdown_phrase_exceeds_span_after_tempo_cap",
                    "span_id": span_id,
                    "segment_ids": [segment.id for _index, segment, _values in span],
                    "values": values,
                    "tokens": tokens,
                    "compact_text": compact_text,
                    "target_duration_sec": round(total_duration_sec, 6),
                    "phrase_duration_sec": round(len(phrase_audio) / sample_rate, 6),
                    "phrase_payload": phrase_payload,
                }
                for _index, segment, _values in span:
                    segment.analysis["countdown_renderer_skip"] = skip_payload
                return set()

            span_audio = np.zeros((total_frames, 2), dtype=np.float32)
            start_frame = max(0, (total_frames - len(phrase_audio)) // 2)
            end_frame = min(total_frames, start_frame + len(phrase_audio))
            if end_frame > start_frame:
                span_audio[start_frame:end_frame] += phrase_audio[: end_frame - start_frame]
            peak = float(np.max(np.abs(span_audio))) if span_audio.size else 0.0
            if peak > 0.98:
                span_audio *= 0.98 / peak
            span_path = span_dir / f"{span_id}.wav"
            write_audio(span_path, span_audio, sample_rate)
            span_metadata = {
                "span_id": span_id,
                "renderer": "countdown_compact_phrase",
                "segment_ids": [segment.id for _index, segment, _values in span],
                "values": values,
                "tokens": tokens,
                "compact_text": compact_text,
                "target_duration_sec": round(total_duration_sec, 6),
                "span_path": str(span_path),
                "phrase_timeline": {
                    "start_sec": round(start_frame / sample_rate, 6),
                    "end_sec": round(end_frame / sample_rate, 6),
                },
                "phrase_payload": phrase_payload,
                "source_language": cfg.source_language,
                "target_language": cfg.target_language,
                "ref_style": resolved_ref_style,
            }
            metadata_path = span_dir / f"{span_id}.json"
            write_json_atomic(metadata_path, span_metadata)

            offset = 0
            for segment_index, (_index, segment, segment_values) in enumerate(span):
                segment_frames = max(1, int(round(segment.duration * sample_rate)))
                if segment_index == len(span) - 1:
                    segment_audio = span_audio[offset:]
                else:
                    segment_audio = span_audio[offset : offset + segment_frames]
                offset += segment_frames
                if len(segment_audio) < segment_frames:
                    padding = np.zeros((segment_frames - len(segment_audio), 2), dtype=np.float32)
                    segment_audio = np.concatenate([segment_audio, padding], axis=0)
                elif len(segment_audio) > segment_frames:
                    segment_audio = segment_audio[:segment_frames]
                final_path = project_dir / "work" / "tts" / f"{segment.id}_final.wav"
                write_audio(final_path, segment_audio, sample_rate)
                final_duration = duration_sec(final_path)
                final_ratio = duration_ratio(final_duration, segment.duration)
                segment_tokens = _countdown_spoken_tokens(segment_values) or []
                payload = {
                    "renderer": "countdown_compact_phrase",
                    "span_id": span_id,
                    "span_metadata_path": str(metadata_path),
                    "span_path": str(span_path),
                    "segment_id": segment.id,
                    "segment_values": segment_values,
                    "segment_tokens": segment_tokens,
                    "values": values,
                    "tokens": tokens,
                    "target_duration_sec": segment.duration,
                    "duration_ratio": final_ratio,
                    "duration_gate": "pass",
                    "audio_qc": {
                        "gate": "pass",
                        "peak_dbfs": round(peak_dbfs(final_path), 3),
                        "rms_dbfs": round(rms_dbfs(final_path), 3),
                    },
                }
                candidate = TTSCandidate(
                    candidate_index=0,
                    seed=cfg.base_seed + 70_000 + first_index,
                    payload=payload,
                    output_path=str(final_path),
                    duration_sec=final_duration,
                    backend="gpt-sovits-countdown-renderer",
                    selected=True,
                    duration_ratio=final_ratio,
                    duration_gate="pass",
                    acceptable_for_mix=True,
                    selection_score=max(0.0, 1.0 - min(abs(final_ratio - 1.0), 1.0)),
                    selection_reason="countdown_compact_phrase",
                    retry_summary={"countdown_renderer": True, "span_id": span_id},
                )
                segment.tts = TTSMetadata(
                    backend="gpt-sovits-countdown-renderer",
                    ref_style=resolved_ref_style,
                    speed_factor=float(getattr(cfg, "gsv_countdown_token_speed_factor", cfg.gsv_tts_max_speed_factor)),
                    candidate_count=1,
                    selected_candidate_path=str(final_path),
                    candidates=[candidate],
                    source_language=cfg.source_language,
                    target_language=cfg.target_language,
                    cross_lingual_voice_transfer=cfg.source_language != cfg.target_language,
                    retry_summary={
                        "countdown_renderer": True,
                        "span_id": span_id,
                        "span_metadata_path": str(metadata_path),
                        "selected_duration_gate": "pass",
                        "selected_acceptable_for_mix": True,
                        "selected_duration_ratio": final_ratio,
                    },
                )
                segment.analysis["countdown_renderer"] = {
                    "span_id": span_id,
                    "segment_values": segment_values,
                    "segment_tokens": segment_tokens,
                    "span_metadata_path": str(metadata_path),
                }
                segment.status = "synthesized"
            return {segment.id for _index, segment, _values in span}

        def render_countdown_span(span: list[tuple[int, Segment, list[int]]]) -> set[str]:
            if getattr(cfg, "gsv_countdown_renderer", "token") == "compact":
                return render_countdown_span_compact(span)
            rendered = render_countdown_span_token(span)
            if rendered:
                return rendered
            fallback = getattr(cfg, "gsv_countdown_fallback_renderer", "compact")
            if fallback == "compact":
                return render_countdown_span_compact(span)
            if fallback == "manual_review":
                for _index, segment, _values in span:
                    segment.status = "needs_manual_review"
                    segment.errors.append("Countdown token renderer failed.")
                return {segment.id for _index, segment, _values in span}
            return set()

        def render_countdown_spans(segment_jobs: list[tuple[int, Segment, int]]) -> set[str]:
            rendered_segment_ids: set[str] = set()
            for span in countdown_spans_for_jobs(segment_jobs):
                rendered_segment_ids.update(render_countdown_span(span))
                save_manifest(project_dir, manifest)
            return rendered_segment_ids

        def candidate_count_for_synth_pass(pass_name: str) -> int:
            if pass_candidate_count_override is not None:
                return pass_candidate_count_override
            if pass_name == "fine_tuned_initial":
                configured = getattr(cfg, "gsv_initial_candidate_count", None)
                return int(configured or min(int(cfg.candidate_count), 3))
            if pass_name == "zero_shot_fallback":
                configured = getattr(cfg, "gsv_zero_shot_candidate_count", None)
                if configured is not None:
                    return int(configured)
                return int(cfg.candidate_count)
            if pass_name == "low_temperature_retry":
                configured = getattr(cfg, "gsv_low_temperature_retry_candidate_count", None)
                if configured is not None:
                    return int(configured)
            configured = getattr(cfg, "gsv_retry_candidate_count", None)
            return int(configured or cfg.candidate_count)

        def temperature_for_synth_pass() -> float:
            if pass_temperature_override is not None:
                return float(pass_temperature_override)
            return float(cfg.gsv_temperature)

        def postprocess_tts_candidate(candidate_path: Path, payload: dict[str, Any]) -> None:
            if not cfg.gsv_trim_edge_silence:
                return
            trim = trim_edge_silence(
                candidate_path,
                threshold_db=cfg.gsv_trim_silence_threshold_db,
                keep_sec=cfg.gsv_trim_silence_keep_sec,
            )
            payload.setdefault("postprocess", {})["edge_silence_trim"] = trim

        def candidate_language_contract_ok(candidate: TTSCandidate) -> bool:
            if _canonical_language(cfg.target_language) != "ko":
                return True
            return (
                candidate.payload.get("text_lang") == "all_ko"
                and candidate.payload.get("prompt_lang") == "all_ja"
                and bool(candidate.payload.get("text"))
            )

        def time_fit_candidate_if_needed(
            segment: Segment,
            selected: TTSCandidate,
            *,
            output_label: str = "timefit",
            max_tempo_override: float | None = None,
            max_stretch_override: float | None = None,
            rescue_metadata: dict[str, Any] | None = None,
            selection_reason_if_acceptable: str = "duration_time_fit_fallback",
            selection_reason_if_failed: str = "duration_time_fit_failed",
        ) -> TTSCandidate | None:
            if selected.acceptable_for_mix:
                return None
            if selected.payload.get("audio_qc", {}).get("gate") != "pass":
                return None
            if not candidate_language_contract_ok(selected):
                return None
            if selected.duration_sec is None or selected.duration_sec <= 0 or segment.duration <= 0:
                return None
            source_path = Path(selected.output_path)
            fitted_path = project_dir / "work" / "tts" / "candidates" / f"{segment.id}_{output_label}.wav"
            payload = copy.deepcopy(selected.payload)
            tempo = selected.duration_sec / segment.duration
            stretch = segment.duration / selected.duration_sec
            base_max_tempo = float(getattr(cfg, "gsv_timefit_max_tempo", 1.18))
            base_max_stretch = float(getattr(cfg, "gsv_timefit_max_stretch", 1.08))
            max_tempo = base_max_tempo
            max_stretch = base_max_stretch
            policy = "default"
            micro_max_sec = float(getattr(cfg, "gsv_timefit_micro_max_sec", 2.0))
            if segment.duration <= micro_max_sec and tempo > base_max_tempo:
                max_tempo = max(
                    max_tempo,
                    float(getattr(cfg, "gsv_timefit_micro_max_tempo", 1.30)),
                )
                policy = "micro_segment_relaxed"
            long_min_sec = float(getattr(cfg, "gsv_timefit_long_min_sec", 7.0))
            if segment.duration >= long_min_sec and stretch > base_max_stretch:
                max_stretch = max(
                    max_stretch,
                    float(getattr(cfg, "gsv_timefit_long_max_stretch", 1.15)),
                )
                policy = "long_segment_relaxed"
            if max_tempo_override is not None and max_tempo_override > max_tempo:
                max_tempo = float(max_tempo_override)
                policy = "rescue_relaxed" if policy == "default" else f"{policy}_rescue_relaxed"
            if max_stretch_override is not None and max_stretch_override > max_stretch:
                max_stretch = float(max_stretch_override)
                policy = "rescue_relaxed" if policy == "default" else f"{policy}_rescue_relaxed"
            if tempo > max_tempo:
                selected.payload["time_fit"] = {
                    "source_path": selected.output_path,
                    "source_duration_sec": selected.duration_sec,
                    "target_duration_sec": segment.duration,
                    "tempo": tempo,
                    "stretch": stretch,
                    "policy": policy,
                    "max_tempo": max_tempo,
                    "max_stretch": max_stretch,
                    "base_max_tempo": base_max_tempo,
                    "base_max_stretch": base_max_stretch,
                    "rejected_reason": f"tempo_above_max:{tempo:.3f}>{max_tempo:.3f}",
                }
                return None
            if stretch > max_stretch:
                selected.payload["time_fit"] = {
                    "source_path": selected.output_path,
                    "source_duration_sec": selected.duration_sec,
                    "target_duration_sec": segment.duration,
                    "tempo": tempo,
                    "stretch": stretch,
                    "policy": policy,
                    "max_tempo": max_tempo,
                    "max_stretch": max_stretch,
                    "base_max_tempo": base_max_tempo,
                    "base_max_stretch": base_max_stretch,
                    "rejected_reason": f"stretch_above_max:{stretch:.3f}>{max_stretch:.3f}",
                }
                return None
            try:
                ffmpeg.fit_audio_duration(
                    source_path,
                    fitted_path,
                    target_duration_sec=segment.duration,
                    sample_rate=cfg.mix_sample_rate,
                    channels=2,
                )
                fitted_duration = duration_sec(fitted_path)
                fitted_peak = peak_dbfs(fitted_path)
                fitted_rms = rms_dbfs(fitted_path)
            except Exception as exc:
                selected.payload.setdefault("time_fit", {})["error"] = str(exc)
                return None
            too_long = duration_too_long(fitted_duration, segment.duration, cfg.duration_tolerance)
            too_short = duration_too_short(fitted_duration, segment.duration, cfg.duration_tolerance)
            fitted_ratio = duration_ratio(fitted_duration, segment.duration)
            duration_gate = "too_long" if too_long else "too_short" if too_short else "pass"
            audio_gate = "silent" if fitted_peak <= -90.0 or fitted_rms <= -90.0 else "pass"
            payload["duration_ratio"] = fitted_ratio
            payload["duration_gate"] = duration_gate
            payload["audio_qc"] = {
                "gate": audio_gate,
                "peak_dbfs": round(fitted_peak, 3),
                "rms_dbfs": round(fitted_rms, 3),
            }
            payload["time_fit"] = {
                "source_path": selected.output_path,
                "source_duration_sec": selected.duration_sec,
                "target_duration_sec": segment.duration,
                "tempo": tempo,
                "stretch": stretch,
                "policy": policy,
                "max_tempo": max_tempo,
                "max_stretch": max_stretch,
                "base_max_tempo": base_max_tempo,
                "base_max_stretch": base_max_stretch,
                "duration_ratio_before": selected.duration_ratio,
                "duration_ratio_after": fitted_ratio,
            }
            if rescue_metadata is not None:
                payload["rescue"] = rescue_metadata
            acceptable_for_mix = (
                duration_gate == "pass" and audio_gate == "pass" and candidate_language_contract_ok(selected)
            )
            selection_score = max(0.0, 1.0 - min(abs(fitted_ratio - 1.0), 1.0))
            return TTSCandidate(
                candidate_index=selected.candidate_index,
                seed=selected.seed,
                payload=payload,
                output_path=str(fitted_path),
                duration_sec=fitted_duration,
                backend=selected.backend,
                duration_ratio=fitted_ratio,
                duration_gate=duration_gate,
                acceptable_for_mix=acceptable_for_mix,
                selection_score=selection_score,
                selection_reason=(
                    selection_reason_if_acceptable
                    if acceptable_for_mix
                    else selection_reason_if_failed
                ),
                retry_summary=selected.retry_summary,
            )

        def duration_gate_for_tolerance(
            actual_duration_sec: float,
            target_duration_sec: float,
            tolerance: float,
        ) -> tuple[str, float]:
            ratio = duration_ratio(actual_duration_sec, target_duration_sec)
            too_long = duration_too_long(actual_duration_sec, target_duration_sec, tolerance)
            too_short = duration_too_short(actual_duration_sec, target_duration_sec, tolerance)
            return ("too_long" if too_long else "too_short" if too_short else "pass", ratio)

        def rescue_with_relaxed_duration_gate(
            segment: Segment,
            audible: list[TTSCandidate],
        ) -> TTSCandidate | None:
            rescue_tolerance = getattr(cfg, "gsv_rescue_duration_tolerance", 0.35)
            if rescue_tolerance is None:
                return None
            rescue_tolerance = float(rescue_tolerance)
            for source_candidate in sorted(
                audible,
                key=lambda candidate: abs((candidate.duration_sec or 0.0) - segment.duration),
            ):
                if (
                    source_candidate.duration_sec is None
                    or source_candidate.duration_sec <= 0
                    or segment.duration <= 0
                    or not candidate_language_contract_ok(source_candidate)
                ):
                    continue
                duration_gate, candidate_ratio = duration_gate_for_tolerance(
                    source_candidate.duration_sec,
                    segment.duration,
                    rescue_tolerance,
                )
                if duration_gate != "pass":
                    continue
                payload = copy.deepcopy(source_candidate.payload)
                payload["duration_ratio"] = candidate_ratio
                payload["duration_gate"] = "pass"
                payload["rescue"] = {
                    "tier": "relaxed_duration_gate",
                    "source_candidate_index": source_candidate.candidate_index,
                    "source_candidate_path": source_candidate.output_path,
                    "source_selection_reason": source_candidate.selection_reason,
                    "strict_duration_gate": source_candidate.duration_gate,
                    "strict_duration_ratio": source_candidate.duration_ratio,
                    "strict_duration_tolerance": cfg.duration_tolerance,
                    "duration_tolerance_used": rescue_tolerance,
                }
                return TTSCandidate(
                    candidate_index=source_candidate.candidate_index,
                    seed=source_candidate.seed,
                    payload=payload,
                    output_path=source_candidate.output_path,
                    duration_sec=source_candidate.duration_sec,
                    backend=source_candidate.backend,
                    duration_ratio=candidate_ratio,
                    duration_gate="pass",
                    acceptable_for_mix=True,
                    selection_score=max(0.0, 1.0 - min(abs(candidate_ratio - 1.0), 1.0)),
                    selection_reason="duration_relaxed_rescue",
                    retry_summary=copy.deepcopy(source_candidate.retry_summary),
                )
            return None

        def rescue_with_source_pause_padding(
            segment: Segment,
            audible: list[TTSCandidate],
        ) -> TTSCandidate | None:
            rescue_tolerance = getattr(cfg, "gsv_rescue_duration_tolerance", 0.35)
            if rescue_tolerance is None or not segment.script:
                return None
            rescue_tolerance = float(rescue_tolerance)
            expected_duration_sec = float(segment.script.expected_tts_duration_sec or 0.0)
            if expected_duration_sec <= 1.0 or segment.duration <= expected_duration_sec:
                return None
            for source_candidate in sorted(
                audible,
                key=lambda candidate: abs((candidate.duration_sec or 0.0) - expected_duration_sec),
            ):
                allowed_omission_reasons = _omission_reasons_allow_source_pause_padding(
                    source_candidate
                )
                if (
                    source_candidate.duration_sec is None
                    or source_candidate.duration_sec <= 0.0
                    or source_candidate.duration_sec >= segment.duration
                    or source_candidate.duration_gate != "too_short"
                    or allowed_omission_reasons is None
                    or not candidate_language_contract_ok(source_candidate)
                ):
                    continue
                speech_ratio = source_candidate.duration_sec / expected_duration_sec
                if speech_ratio < 1.0 - rescue_tolerance or speech_ratio > 1.0 + rescue_tolerance:
                    continue
                source_path = Path(source_candidate.output_path)
                padded_path = source_path.with_name(f"{source_path.stem}_pause_padded.wav")
                try:
                    source_audio, sample_rate = load_audio(source_path)
                    target_frames = max(
                        len(source_audio),
                        int(round(segment.duration * sample_rate)),
                    )
                    padding_frames = target_frames - len(source_audio)
                    if padding_frames <= 0:
                        continue
                    silence = np.zeros(
                        (padding_frames, source_audio.shape[1]),
                        dtype=source_audio.dtype,
                    )
                    write_audio(padded_path, np.concatenate([source_audio, silence], axis=0), sample_rate)
                    padded_duration = duration_sec(padded_path)
                    padded_peak = peak_dbfs(padded_path)
                    padded_rms = rms_dbfs(padded_path)
                except Exception as exc:
                    source_candidate.payload.setdefault("pause_padding", {})["error"] = str(exc)
                    continue
                too_long = duration_too_long(padded_duration, segment.duration, cfg.duration_tolerance)
                too_short = duration_too_short(padded_duration, segment.duration, cfg.duration_tolerance)
                padded_ratio = duration_ratio(padded_duration, segment.duration)
                duration_gate = "too_long" if too_long else "too_short" if too_short else "pass"
                audio_gate = "silent" if padded_peak <= -90.0 or padded_rms <= -90.0 else "pass"
                if duration_gate != "pass" or audio_gate != "pass":
                    continue
                padding_sec = max(0.0, padded_duration - source_candidate.duration_sec)
                payload = copy.deepcopy(source_candidate.payload)
                payload["duration_ratio"] = padded_ratio
                payload["duration_gate"] = "pass"
                payload["audio_qc"] = {
                    "gate": audio_gate,
                    "peak_dbfs": round(padded_peak, 3),
                    "rms_dbfs": round(padded_rms, 3),
                }
                payload["pause_padding"] = {
                    "tier": "source_pause_padding",
                    "source_candidate_index": source_candidate.candidate_index,
                    "source_candidate_path": source_candidate.output_path,
                    "speech_duration_sec": round(source_candidate.duration_sec, 6),
                    "padding_sec": round(padding_sec, 6),
                    "expected_tts_duration_sec": round(expected_duration_sec, 6),
                    "target_segment_duration_sec": round(segment.duration, 6),
                    "speech_duration_ratio_to_expected": round(speech_ratio, 6),
                    "strict_duration_gate": source_candidate.duration_gate,
                    "strict_duration_ratio": source_candidate.duration_ratio,
                    "strict_duration_tolerance": cfg.duration_tolerance,
                    "speech_duration_tolerance_used": rescue_tolerance,
                }
                if allowed_omission_reasons:
                    payload["pause_padding"]["allowed_omission_detection_reasons"] = (
                        allowed_omission_reasons
                    )
                payload["rescue"] = copy.deepcopy(payload["pause_padding"])
                return TTSCandidate(
                    candidate_index=source_candidate.candidate_index,
                    seed=source_candidate.seed,
                    payload=payload,
                    output_path=str(padded_path),
                    duration_sec=padded_duration,
                    backend=source_candidate.backend,
                    duration_ratio=padded_ratio,
                    duration_gate="pass",
                    acceptable_for_mix=True,
                    selection_score=max(0.0, 1.0 - min(abs(padded_ratio - 1.0), 1.0)),
                    selection_reason="source_pause_padding_rescue",
                    retry_summary=copy.deepcopy(source_candidate.retry_summary),
                )
            return None

        def rescue_with_relaxed_time_fit(
            segment: Segment,
            audible: list[TTSCandidate],
            candidates: list[TTSCandidate],
        ) -> TTSCandidate | None:
            top_k = int(getattr(cfg, "gsv_rescue_timefit_top_k", 3))
            if top_k <= 0:
                return None
            max_tempo = float(getattr(cfg, "gsv_rescue_timefit_max_tempo", 1.45))
            max_stretch = float(getattr(cfg, "gsv_rescue_timefit_max_stretch", 1.25))
            counting_compaction_timefit = bool(
                segment.analysis.get("pre_synth_tts_counting_compaction")
            )
            if counting_compaction_timefit:
                max_tempo = max(
                    max_tempo,
                    float(getattr(cfg, "gsv_rescue_timefit_counting_max_tempo", 1.60)),
                )
            ranked = sorted(
                audible,
                key=lambda candidate: abs((candidate.duration_sec or 0.0) - segment.duration),
            )[:top_k]
            for rank, source_candidate in enumerate(ranked, start=1):
                rescue_metadata = {
                    "tier": "relaxed_time_fit",
                    "source_candidate_index": source_candidate.candidate_index,
                    "source_candidate_path": source_candidate.output_path,
                    "source_selection_reason": source_candidate.selection_reason,
                    "strict_duration_gate": source_candidate.duration_gate,
                    "strict_duration_ratio": source_candidate.duration_ratio,
                    "max_tempo_used": max_tempo,
                    "max_stretch_used": max_stretch,
                    "rank": rank,
                }
                if counting_compaction_timefit:
                    rescue_metadata["counting_compaction_timefit"] = True
                fitted = time_fit_candidate_if_needed(
                    segment,
                    source_candidate.model_copy(deep=True),
                    output_label=f"rescue_timefit_{rank:02d}",
                    max_tempo_override=max_tempo,
                    max_stretch_override=max_stretch,
                    rescue_metadata=rescue_metadata,
                    selection_reason_if_acceptable="duration_relaxed_timefit_rescue",
                    selection_reason_if_failed="duration_relaxed_timefit_failed",
                )
                if fitted is None:
                    continue
                candidates.append(fitted)
                if fitted.acceptable_for_mix:
                    return fitted
            return None

        def micro_segment_manual_review_summary(
            segment: Segment,
            selected: TTSCandidate,
        ) -> dict[str, Any] | None:
            micro_max_sec = float(getattr(cfg, "gsv_rescue_micro_segment_max_sec", 0.6))
            if micro_max_sec <= 0 or segment.duration > micro_max_sec:
                return None
            return {
                "acceptable_candidates": 0,
                "selected_duration_gate": selected.duration_gate,
                "selected_duration_ratio": selected.duration_ratio,
                "rescue_status": "micro_segment_manual_review",
                "micro_segment_max_sec": micro_max_sec,
            }

        def duration_rewrite_context(index: int) -> list[Segment]:
            radius = int(getattr(cfg, "gemma_text_context_radius", 0))
            if radius <= 0:
                return []
            start = max(0, index - 1 - radius)
            end = min(len(manifest.segments), index + radius)
            return manifest.segments[start:end]

        def duration_rewrite_char_budget(
            segment: Segment,
            text: str,
            actual_duration_sec: float,
            reason: str,
        ) -> dict[str, int]:
            current_chars = korean_tts_speech_char_count(text)
            source_text = segment.source_script.text if segment.source_script else ""
            timing_budget = korean_tts_timing_budget(segment.duration, source_text)
            max_chars = max(1, int(timing_budget["max_speech_chars"]))
            if actual_duration_sec > 0 and segment.duration > 0:
                target_chars = int(round(current_chars * segment.duration / actual_duration_sec))
            else:
                target_chars = current_chars
            target_chars = max(1, min(max_chars, target_chars))
            if reason == "too_short":
                target_chars = max(min(current_chars + 1, max_chars), target_chars)
                min_chars = min(max_chars, max(current_chars + 1, int(target_chars * 0.85)))
            else:
                min_chars = max(1, int(target_chars * 0.80))
            return {
                "current": current_chars,
                "target": target_chars,
                "min": max(1, min_chars),
                "max": max_chars,
            }

        def accept_duration_rewrite_text(
            *,
            segment: Segment,
            candidate_text: str,
            reason: str,
            budget: dict[str, int],
        ) -> tuple[bool, dict[str, Any]]:
            normalized = normalize_korean_tts_text(candidate_text)
            text = normalized.text.strip()
            chars = korean_tts_speech_char_count(text)
            metadata: dict[str, Any] = {
                "normalized_text": text,
                "speech_chars": chars,
                "target_speech_chars": budget["target"],
                "min_speech_chars": budget["min"],
                "max_speech_chars": budget["max"],
                "rejected_reasons": [],
            }
            if not text:
                metadata["rejected_reasons"].append("empty_rewrite")
            if chars > budget["max"]:
                metadata["rejected_reasons"].append(
                    f"speech_chars_above_max:{chars}>{budget['max']}"
                )
            if reason == "too_short" and chars < budget["min"]:
                metadata["rejected_reasons"].append(
                    f"speech_chars_below_min:{chars}<{budget['min']}"
                )
            if reason == "too_long" and chars >= budget["current"]:
                metadata["rejected_reasons"].append(
                    f"not_shorter_for_too_long:{chars}>={budget['current']}"
                )
            trial_script = segment.script.model_copy(
                update={
                    "tts_text": text,
                    "expected_tts_duration_sec": estimate_tts_duration(text, "ko"),
                    "risk_flags": [*segment.script.risk_flags, *normalized.risk_flags],
                },
                deep=True,
            )
            preflight = preflight_tts_text(
                trial_script,
                target_language=cfg.target_language,
                source_text=segment.source_script.text if segment.source_script else "",
                min_hangul_ratio=cfg.gsv_ko_text_min_hangul_ratio,
            )
            metadata["preflight"] = preflight.as_payload()
            if preflight.blocked:
                metadata["rejected_reasons"].append(
                    "preflight_blocked:" + ",".join(preflight.issues)
                )
            return not metadata["rejected_reasons"], metadata

        def maybe_rewrite_script_with_gemma(
            *,
            segment: Segment,
            index: int,
            attempt_text: str,
            actual_duration_sec: float,
            reason: str,
            rewrite_attempts_used: int,
        ) -> tuple[JapaneseScript | None, dict[str, Any]]:
            metadata: dict[str, Any] = {
                "backend": "gemma_text",
                "reason": reason,
                "before": attempt_text,
                "actual_duration_sec": round(actual_duration_sec, 6),
                "target_duration_sec": round(segment.duration, 6),
                "accepted": False,
            }
            if duration_rewrite_client is None:
                metadata["error"] = "duration_rewrite_client_unavailable"
                return None, metadata
            if rewrite_attempts_used >= int(getattr(cfg, "gsv_duration_rewrite_max_attempts", 0)):
                metadata["error"] = "duration_rewrite_attempt_limit_reached"
                return None, metadata
            if segment.script.rewrite_count >= segment.script.retry_policy.max_script_rewrites:
                metadata["error"] = "script_retry_policy_limit_reached"
                return None, metadata
            budget = duration_rewrite_char_budget(segment, attempt_text, actual_duration_sec, reason)
            metadata.update(
                {
                    "current_speech_chars": budget["current"],
                    "target_speech_chars": budget["target"],
                    "min_speech_chars": budget["min"],
                    "max_speech_chars": budget["max"],
                }
            )
            batch_id = f"duration_rewrite_{segment.id}_{reason}_{segment.script.rewrite_count + 1}"
            try:
                with duration_rewrite_lock:
                    translation: KoreanTranslation | None = duration_rewrite_client.rewrite_tts_for_duration(
                        segment=segment,
                        batch_id=batch_id,
                        current_text=attempt_text,
                        reason=reason,
                        actual_duration_sec=actual_duration_sec,
                        target_duration_sec=segment.duration,
                        target_speech_chars=budget["target"],
                        min_speech_chars=budget["min"],
                        max_speech_chars=budget["max"],
                        context_segments=duration_rewrite_context(index),
                    )
            except Exception as exc:
                metadata["error"] = str(exc)
                return None, metadata
            if translation is None:
                metadata["error"] = "gemma_returned_no_translation"
                return None, metadata
            accepted, acceptance = accept_duration_rewrite_text(
                segment=segment,
                candidate_text=translation.ko_natural,
                reason=reason,
                budget=budget,
            )
            metadata.update(acceptance)
            metadata["after"] = acceptance["normalized_text"]
            metadata["model"] = translation.model
            metadata["batch_id"] = translation.batch_id
            if not accepted and not _maybe_relax_duration_rewrite_acceptance(metadata):
                return None, metadata
            updated = segment.script.model_copy(deep=True)
            updated.tts_text = acceptance["normalized_text"]
            updated.expected_tts_duration_sec = estimate_tts_duration(updated.tts_text, "ko")
            updated.rewrite_count += 1
            risk_flag = (
                f"gemma_duration_rewrite_{reason}_relaxed"
                if metadata.get("accepted_relaxed")
                else f"gemma_duration_rewrite_{reason}"
            )
            updated.risk_flags.append(risk_flag)
            metadata["accepted"] = True
            return updated, metadata

        def synthesize_segment_locked(
            index: int,
            segment: Segment,
            lane_index: int,
        ) -> tuple[int, Segment]:
            if should_reset_previous_tts(segment):
                reset_previous_tts_attempt(segment)
            if segment.status == "synthesized":
                return index, segment
            if segment.status in SKIP_STATUSES:
                return index, segment
            if not segment.script:
                segment.status = "needs_manual_review"
                segment.errors.append("Cannot synthesize without script metadata.")
                return index, segment
            target_language = _canonical_language(cfg.target_language)
            source_language = _canonical_language(cfg.source_language)
            open_sentence_normalized = False
            if target_language == "ko":
                previous_tts_text = segment.script.tts_text
                normalized = normalize_korean_tts_text(previous_tts_text)
                normalized_text = normalized.text
                normalization_risk_flags = list(normalized.risk_flags)
                if normalized_text:
                    closed_text, open_sentence_normalized = _close_open_korean_tts_sentence(
                        normalized_text
                    )
                    if open_sentence_normalized:
                        normalized_text = closed_text
                        normalization_risk_flags = list(
                            dict.fromkeys([*normalization_risk_flags, "closed_open_sentence"])
                        )
                if normalized_text and normalized_text != previous_tts_text.strip():
                    segment.script.tts_text = normalized_text
                    segment.script.expected_tts_duration_sec = estimate_tts_duration(normalized_text, "ko")
                    segment.script.nonverbal_cues = [*segment.script.nonverbal_cues, *normalized.cues]
                    segment.script.risk_flags = list(
                        dict.fromkeys([*segment.script.risk_flags, *normalization_risk_flags])
                    )
                    segment.analysis["pre_synth_tts_text_normalization"] = {
                        "before": previous_tts_text,
                        "after": normalized_text,
                        "normalized_text": normalized.text,
                        "risk_flags": normalization_risk_flags,
                    }
                compacted_text, counting_metadata = _compact_korean_counting_tts_text(
                    segment.script.tts_text
                )
                if counting_metadata is not None and compacted_text != segment.script.tts_text.strip():
                    previous_tts_text = segment.script.tts_text
                    segment.script.tts_text = compacted_text
                    segment.script.expected_tts_duration_sec = estimate_tts_duration(
                        compacted_text, "ko"
                    )
                    segment.script.risk_flags = list(
                        dict.fromkeys(
                            [
                                *segment.script.risk_flags,
                                "korean_counting_tts_compacted",
                            ]
                        )
                    )
                    segment.analysis["pre_synth_tts_counting_compaction"] = {
                        "before": previous_tts_text,
                        "after": compacted_text,
                        **counting_metadata,
                    }
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
            static_ref = resolve_ref(segment_refs, requested_ref_style)
            segment_ref, segment_ref_metadata = _segment_source_ref_for_gsv(
                project_dir,
                segment,
                cfg,
                manifest.segments,
            )
            static_ref_retry_active = segment.id in static_ref_retry_segment_ids
            if static_ref_retry_active:
                segment_ref_metadata = copy.deepcopy(segment_ref_metadata)
                segment_ref_metadata["static_ref_retry"] = True
                segment_ref_metadata["used_before_static_ref_retry"] = bool(
                    segment_ref_metadata.get("used")
                )
                segment_ref_metadata["used"] = False
                if segment_ref is not None:
                    segment_ref_metadata["disabled_by_synth_pass"] = "static_ref_retry"
                ref = static_ref
            else:
                ref = segment_ref or static_ref
            synthesis_ref = _ref_for_tts_language(ref, segment.script.tts_language)
            static_synthesis_ref = _ref_for_tts_language(static_ref, segment.script.tts_language)
            fallback_used = resolved_ref_style != original_ref_style
            candidates: list[TTSCandidate] = []
            expected = segment.script.expected_tts_duration_sec or segment.duration
            speed = (
                1.0
                if duration_rewrite_enabled
                else suggest_speed_factor(
                    expected,
                    segment.duration,
                    minimum=cfg.gsv_tts_min_speed_factor,
                    maximum=cfg.gsv_tts_max_speed_factor,
                )
            )
            has_repetition_or_omission_signal = bool(
                segment.qc and (segment.qc.repetition_detected or segment.qc.omission_detected)
            )
            can_defer_duration_rewrite = (
                duration_rewrite_phase == "initial"
                and duration_rewrite_enabled
                and _can_rewrite_script_for_duration(segment.script)
            )
            pass_candidate_count = candidate_count_for_synth_pass(synth_pass)
            pass_temperature = temperature_for_synth_pass()
            effective_candidate_count = (
                int(cfg.gsv_duration_rewrite_pre_candidate_count or pass_candidate_count)
                if can_defer_duration_rewrite
                else pass_candidate_count
            )
            pending_duration_rewrite = copy.deepcopy(
                segment.analysis.pop("pending_duration_rewrite", None)
            )
            for candidate_index in range(effective_candidate_count):
                seed = cfg.base_seed + index * 100 + candidate_index
                tts_text_language = _segment_tts_text_language(segment, target_language)
                max_attempts = (
                    1
                    if can_defer_duration_rewrite
                    else int(getattr(cfg, "gsv_max_attempts_per_candidate", 3))
                )
                if open_sentence_normalized:
                    max_attempts = max(max_attempts, 2)
                last_attempt = max_attempts - 1
                options = GPTSoVITSTTSOptions(
                    seed=seed,
                    speed_factor=speed,
                    text_lang=tts_text_language,
                    top_k=cfg.gsv_top_k,
                    top_p=cfg.gsv_top_p,
                    temperature=pass_temperature,
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
                attempt_ref = synthesis_ref
                omission_retry_metadata: dict[str, Any] | None = None
                for attempt in range(max_attempts):
                    candidate_path = _tts_candidate_path(project_dir, segment.id, candidate_index, attempt)
                    current_ref = attempt_ref
                    using_segment_ref = (
                        segment_ref is not None
                        and current_ref.ref_audio_path == synthesis_ref.ref_audio_path
                    )
                    payload: dict[str, Any] = {
                        "speaker_id": segment.speaker_id,
                        "requested_ref_style": original_ref_style,
                        "resolved_ref_style": resolved_ref_style,
                        "fallback_used": fallback_used,
                        "ref_audio_path": current_ref.ref_audio_path,
                        "aux_ref_audio_paths": current_ref.aux_ref_audio_paths,
                        "prompt_text_policy": (
                            "use_static_reference_retry"
                            if static_ref_retry_active
                            else
                            "use_segment_source_reference"
                            if using_segment_ref
                            else "use_source_reference_prompt"
                        ),
                        "segment_ref": segment_ref_metadata,
                        "speaker_gpt_weights_path": speaker_gpt_weights,
                        "speaker_sovits_weights_path": speaker_sovits_weights,
                        "speaker_refs_path": str(speaker_refs_path) if speaker_refs_path else None,
                        "source_language": source_language,
                        "target_language": target_language,
                        "cross_lingual_voice_transfer": source_language != target_language,
                        "expected_tts_duration_sec": expected,
                        "target_duration_sec": segment.duration,
                        "synth_pass": synth_pass,
                        "candidate_count_used": effective_candidate_count,
                        "temperature_used": pass_temperature,
                        "lane_index": lane_index,
                        "gsv_url": None if mock else gsv_base_urls[lane_index],
                        "retry": {
                            "attempt": attempt,
                            "max_attempts": max_attempts,
                            "signals": retry_signal_values(attempt_signals),
                        },
                    }
                    payload.update(_tts_request_debug_payload(attempt_text, current_ref, options))
                    if omission_retry_metadata is not None:
                        payload["omission_retry"] = copy.deepcopy(omission_retry_metadata)
                    if pending_duration_rewrite is not None:
                        payload["duration_rewrite"] = copy.deepcopy(pending_duration_rewrite)
                    if mock:
                        mock_duration = max(0.05, segment.duration)
                        _mock_synthesize(candidate_path, mock_duration, options.seed, cfg.mix_sample_rate)
                        postprocess_tts_candidate(candidate_path, payload)
                        duration = duration_sec(candidate_path)
                        peak = peak_dbfs(candidate_path)
                        rms = rms_dbfs(candidate_path)
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
                            request = client.build_payload(attempt_text, current_ref, options)
                            payload.update(request.as_payload())
                            client.synthesize_to_file(request, candidate_path)
                            postprocess_tts_candidate(candidate_path, payload)
                            duration = duration_sec(candidate_path)
                            peak = peak_dbfs(candidate_path)
                            rms = rms_dbfs(candidate_path)
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
                    audio_gate = "silent" if peak <= -90.0 or rms <= -90.0 else "pass"
                    payload["duration_ratio"] = candidate_ratio
                    payload["duration_gate"] = duration_gate
                    payload["audio_qc"] = {
                        "gate": audio_gate,
                        "peak_dbfs": round(peak, 3),
                        "rms_dbfs": round(rms, 3),
                    }
                    language_contract_ok = True
                    if target_language == "ko":
                        language_contract_ok = (
                            payload.get("text") == attempt_text
                            and payload.get("text_lang") == "all_ko"
                            and payload.get("prompt_lang") == "all_ja"
                        )
                    acceptable_for_mix = (
                        duration_gate == "pass" and audio_gate == "pass" and language_contract_ok
                    )
                    omission_reasons = _gsv_omission_detection_reasons(
                        duration_sec=duration,
                        target_duration_sec=segment.duration,
                        expected_tts_duration_sec=expected,
                        duration_gate=duration_gate,
                        audio_gate=audio_gate,
                        language_contract_ok=language_contract_ok,
                    )
                    if omission_reasons:
                        payload["omission_suspected"] = True
                        payload["omission_detection"] = {
                            "reasons": omission_reasons,
                            "duration_sec": round(duration, 6),
                            "expected_tts_duration_sec": round(expected, 6),
                            "target_duration_sec": round(segment.duration, 6),
                        }
                    selection_score = max(0.0, 1.0 - min(abs(candidate_ratio - 1.0), 1.0))
                    if audio_gate != "pass" and attempt < last_attempt:
                        payload["retry"]["next_action"] = GPTSoVITSRetrySignal.SEED_CHANGED.value
                    elif omission_reasons and attempt < last_attempt:
                        payload["retry"]["next_action"] = GPTSoVITSRetrySignal.REPETITION_OR_OMISSION.value
                    elif (too_long or too_short) and attempt < last_attempt:
                        payload["retry"]["next_action"] = (
                            GPTSoVITSRetrySignal.SPEED_FACTOR_ADJUSTED.value
                            if attempt == 0
                            else GPTSoVITSRetrySignal.SEED_CHANGED.value
                        )
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
                                else "audio_qc_failed"
                                if audio_gate != "pass"
                                else "omission_suspected"
                                if omission_reasons
                                else "duration_or_language_contract_failed"
                            ),
                            retry_summary=payload["retry"],
                        )
                    )
                    if audio_gate != "pass":
                        if attempt >= last_attempt:
                            break
                        options = options.model_copy(
                            update={
                                "seed": options.seed + 30_000 + index + attempt
                                if options.seed >= 0
                                else 30_000 + index + attempt
                            }
                        )
                        attempt_signals = [GPTSoVITSRetrySignal.SEED_CHANGED]
                        continue
                    if omission_reasons:
                        if attempt >= last_attempt:
                            break
                        retry_text, closed_for_retry = _close_open_korean_tts_sentence(attempt_text)
                        attempt_text = retry_text
                        use_static_ref = (
                            segment_ref is not None
                            and current_ref.ref_audio_path != static_synthesis_ref.ref_audio_path
                        )
                        attempt_ref = static_synthesis_ref if use_static_ref else current_ref
                        options = adjust_for_repetition_or_omission(
                            options,
                            seed_step=40_000 + index + attempt,
                        )
                        attempt_signals = [
                            GPTSoVITSRetrySignal.REPETITION_OR_OMISSION,
                            GPTSoVITSRetrySignal.SEED_CHANGED,
                            GPTSoVITSRetrySignal.REPETITION_PENALTY_INCREASED,
                        ]
                        omission_retry_metadata = {
                            "trigger": "omission_suspected",
                            "source_candidate_index": candidate_index,
                            "source_attempt": attempt,
                            "source_duration_sec": round(duration, 6),
                            "ref_fallback": "static_ref" if use_static_ref else "same_ref",
                            "text_normalization": (
                                "closed_open_sentence"
                                if open_sentence_normalized or closed_for_retry
                                else "unchanged"
                            ),
                            "reasons": omission_reasons,
                        }
                        continue
                    if not (too_long or too_short):
                        break
                    if attempt >= last_attempt:
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
                    duration_signal = (
                        GPTSoVITSRetrySignal.DURATION_TOO_LONG
                        if too_long
                        else GPTSoVITSRetrySignal.DURATION_TOO_SHORT
                    )
                    options = options.model_copy(
                        update={
                            "seed": options.seed + 20_000 + index + attempt
                            if options.seed >= 0
                            else 20_000 + index + attempt
                        }
                    )
                    attempt_signals = [
                        duration_signal,
                        GPTSoVITSRetrySignal.SEED_CHANGED,
                    ]
            successful = [
                candidate for candidate in candidates if not candidate.error and candidate.duration_sec is not None
            ]
            if not successful:
                segment.tts = TTSMetadata(
                    backend="mock" if mock else "gpt-sovits",
                    ref_style=resolved_ref_style,
                    speed_factor=speed,
                    candidate_count=effective_candidate_count,
                    candidates=candidates,
                    source_language=source_language,
                    target_language=target_language,
                    cross_lingual_voice_transfer=source_language != target_language,
                )
                segment.status = "failed"
                segment.errors.append("All TTS candidates failed.")
                return index, segment
            acceptable = [candidate for candidate in successful if candidate.acceptable_for_mix]
            audible = [
                candidate
                for candidate in successful
                if candidate.payload.get("audio_qc", {}).get("gate") == "pass"
            ]
            _update_gsv_candidate_selection_scores(successful, segment)
            if not audible:
                segment.tts = TTSMetadata(
                    backend="mock" if mock else "gpt-sovits",
                    ref_style=resolved_ref_style,
                    speed_factor=speed,
                    candidate_count=effective_candidate_count,
                    candidates=candidates,
                    source_language=source_language,
                    target_language=target_language,
                    cross_lingual_voice_transfer=source_language != target_language,
                    retry_summary={"acceptable_candidates": 0},
                )
                segment.status = "failed"
                segment.errors.append("No acceptable TTS candidates for mix.")
                return index, segment
            if acceptable:
                selected = _select_gsv_candidate_for_mix(acceptable, segment)
            else:
                selected = _select_gsv_candidate_for_mix(audible, segment)
                fitted = time_fit_candidate_if_needed(segment, selected)
                if fitted is not None:
                    candidates.append(fitted)
                    _update_gsv_candidate_selection_scores([fitted], segment)
                    if fitted.acceptable_for_mix:
                        selected = fitted
                if not selected.acceptable_for_mix:
                    rescued = rescue_with_relaxed_duration_gate(segment, audible)
                    if rescued is not None:
                        candidates.append(rescued)
                        _update_gsv_candidate_selection_scores([rescued], segment)
                        selected = rescued
                if not selected.acceptable_for_mix:
                    rescued = rescue_with_source_pause_padding(segment, audible)
                    if rescued is not None:
                        candidates.append(rescued)
                        _update_gsv_candidate_selection_scores([rescued], segment)
                        selected = rescued
                if not selected.acceptable_for_mix:
                    rescued = rescue_with_relaxed_time_fit(segment, audible, candidates)
                    if rescued is not None:
                        _update_gsv_candidate_selection_scores([rescued], segment)
                        selected = rescued
                if not selected.acceptable_for_mix:
                    micro_summary = micro_segment_manual_review_summary(segment, selected)
                    if micro_summary is not None:
                        segment.tts = TTSMetadata(
                            backend="mock" if mock else "gpt-sovits",
                            ref_style=resolved_ref_style,
                            speed_factor=speed,
                            candidate_count=effective_candidate_count,
                            candidates=candidates,
                            source_language=source_language,
                            target_language=target_language,
                            cross_lingual_voice_transfer=source_language != target_language,
                            retry_summary=micro_summary,
                        )
                        segment.status = "needs_manual_review"
                        segment.errors.append("Micro segment too short for Korean TTS.")
                        return index, segment
                if not selected.acceptable_for_mix:
                    segment.tts = TTSMetadata(
                        backend="mock" if mock else "gpt-sovits",
                        ref_style=resolved_ref_style,
                        speed_factor=speed,
                        candidate_count=effective_candidate_count,
                        candidates=candidates,
                        source_language=source_language,
                        target_language=target_language,
                        cross_lingual_voice_transfer=source_language != target_language,
                        retry_summary={
                            "acceptable_candidates": 0,
                            "selected_duration_gate": selected.duration_gate,
                            "selected_duration_ratio": selected.duration_ratio,
                        },
                    )
                    segment.status = "failed"
                    segment.errors.append("No acceptable TTS candidates for mix.")
                    return index, segment
            selected.selected = True
            final_path = project_dir / "work" / "tts" / f"{segment.id}_final.wav"
            ensure_not_same_path(Path(selected.output_path), final_path)
            final_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(selected.output_path, final_path)
            segment.tts = TTSMetadata(
                backend="mock" if mock else "gpt-sovits",
                ref_style=resolved_ref_style,
                speed_factor=float(selected.payload.get("speed_factor", speed)),
                candidate_count=effective_candidate_count,
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
            if mock or (
                segment.status in {*SKIP_STATUSES, "synthesized"}
                and not should_reset_previous_tts(segment)
            ) or not segment.script:
                return synthesize_segment_locked(index, segment, lane_index)
            with lane_locks[lane_index]:
                return synthesize_segment_locked(index, segment, lane_index)

        def run_synth_jobs(segment_jobs: list[tuple[int, Segment, int]]) -> None:
            nonlocal last_logged_at
            if not segment_jobs:
                return
            start_gsv_servers()
            job_total = len(segment_jobs)
            completed_jobs = 0
            if not mock and gsv_lane_count > 1 and len(segment_jobs) > 1:
                with ThreadPoolExecutor(max_workers=gsv_lane_count) as executor:
                    futures = [
                        executor.submit(synthesize_segment, index, segment, lane_index)
                        for index, segment, lane_index in segment_jobs
                    ]
                    for future in as_completed(futures):
                        index, segment = future.result()
                        completed_jobs += 1
                        save_manifest(project_dir, manifest)
                        last_logged_at = _log_segment_progress(
                            "synth",
                            index,
                            job_total,
                            segment,
                            manifest,
                            started_at,
                            last_logged_at,
                            progress_index=completed_jobs,
                            counts_label="status_counts",
                        )
            else:
                for index, segment, lane_index in segment_jobs:
                    index, segment = synthesize_segment(index, segment, lane_index)
                    completed_jobs += 1
                    save_manifest(project_dir, manifest)
                    last_logged_at = _log_segment_progress(
                        "synth",
                        index,
                        job_total,
                        segment,
                        manifest,
                        started_at,
                        last_logged_at,
                        progress_index=completed_jobs,
                        counts_label="status_counts",
                    )

        def selected_failed_segment_ids() -> list[str]:
            return [
                segment.id
                for segment in manifest.segments
                if segment.status == "failed"
                and (only_segment_ids is None or segment.id in only_segment_ids)
            ]

        def failed_segment_ids_with_used_segment_ref(segment_ids: list[str]) -> list[str]:
            target_ids = set(segment_ids)
            return [
                segment.id
                for segment in manifest.segments
                if segment.id in target_ids
                and segment.status == "failed"
                and segment.tts is not None
                and any(
                    bool(candidate.payload.get("segment_ref", {}).get("used"))
                    for candidate in segment.tts.candidates
                )
            ]

        def jobs_for_segment_ids(segment_ids: set[str]) -> list[tuple[int, Segment, int]]:
            return [
                (index, segment, _segment_lane_index(segment, index - 1, gsv_lane_count))
                for index, segment in enumerate(manifest.segments, start=1)
                if segment.id in segment_ids
            ]

        def reset_failed_segments_for_internal_retry(
            segment_ids: set[str],
            *,
            pass_name: str,
        ) -> None:
            for segment in manifest.segments:
                if segment.id not in segment_ids or segment.status != "failed" or not segment.script:
                    continue
                segment.analysis.setdefault("synth_internal_retry_history", []).append(
                    {
                        "pass": pass_name,
                        "previous_errors": list(segment.errors),
                        "previous_duration_gate": (
                            segment.tts.retry_summary.get("selected_duration_gate")
                            if segment.tts
                            else None
                        ),
                    }
                )
                reset_previous_tts_attempt(segment)

        def run_internal_retry_pass(
            *,
            pass_name: str,
            segment_ids: list[str],
            zero_shot: bool = False,
            static_ref_retry: bool = False,
            candidate_count_override: int | None = None,
            temperature_override: float | None = None,
        ) -> dict[str, Any] | None:
            nonlocal gpt_weights, sovits_weights, synth_pass, internal_retry_segment_ids
            nonlocal static_ref_retry_segment_ids
            nonlocal pass_candidate_count_override, pass_temperature_override
            if mock or not segment_ids:
                return None
            target_ids = set(segment_ids)
            if zero_shot:
                stop_gsv_servers()
                gpt_weights = None
                sovits_weights = None
                model_switch.setdefault("zero_shot_fallback", {})["weights_policy"] = "unchanged"
            internal_retry_segment_ids = target_ids
            static_ref_retry_segment_ids = target_ids if static_ref_retry else set()
            pass_candidate_count_override = candidate_count_override
            pass_temperature_override = temperature_override
            reset_failed_segments_for_internal_retry(target_ids, pass_name=pass_name)
            synth_pass = pass_name
            run_synth_jobs(jobs_for_segment_ids(target_ids))
            internal_retry_segment_ids = set()
            static_ref_retry_segment_ids = set()
            pass_candidate_count_override = None
            pass_temperature_override = None
            failed_after = [
                segment_id
                for segment_id in selected_failed_segment_ids()
                if segment_id in target_ids
            ]
            succeeded = [segment_id for segment_id in segment_ids if segment_id not in failed_after]
            summary: dict[str, Any] = {
                "attempted_segments": segment_ids,
                "succeeded_segments": succeeded,
                "failed_segments": failed_after,
                "candidate_count": (
                    candidate_count_override
                    if candidate_count_override is not None
                    else candidate_count_for_synth_pass(pass_name)
                ),
            }
            if temperature_override is not None:
                summary["temperature"] = temperature_override
            if zero_shot:
                summary["server_restarted_for_zero_shot"] = should_auto_start_server
                model_switch["zero_shot_fallback"] = {
                    **model_switch.get("zero_shot_fallback", {}),
                    **summary,
                }
            return summary

        def duration_rewrite_request_for_segment(
            index: int,
            segment: Segment,
        ) -> dict[str, Any] | None:
            if (
                not duration_rewrite_enabled
                or not segment.script
                or not _can_rewrite_script_for_duration(segment.script)
                or not segment.tts
                or segment.status not in {"failed", "needs_manual_review"}
            ):
                return None
            candidates = [
                candidate
                for candidate in segment.tts.candidates
                if not candidate.error
                and candidate.duration_sec is not None
                and candidate.duration_gate in {"too_long", "too_short"}
                and candidate.payload.get("audio_qc", {}).get("gate") == "pass"
            ]
            if not candidates:
                return None
            candidate = min(
                candidates,
                key=lambda item: abs((item.duration_ratio or 0.0) - 1.0),
            )
            return {
                "index": index,
                "segment": segment,
                "attempt_text": str(candidate.payload.get("text") or segment.script.tts_text),
                "actual_duration_sec": float(candidate.duration_sec or 0.0),
                "reason": candidate.duration_gate,
            }

        def reset_segment_for_duration_rewrite_retry(
            segment: Segment,
            rewrite_metadata: dict[str, Any],
        ) -> None:
            segment.analysis.setdefault("duration_rewrite_history", []).append(
                copy.deepcopy(rewrite_metadata)
            )
            segment.analysis["pending_duration_rewrite"] = copy.deepcopy(rewrite_metadata)
            segment.status = "scripted"
            segment.tts = None
            segment.errors = [
                error
                for error in segment.errors
                if error
                not in {
                    "No acceptable TTS candidates for mix.",
                    "All TTS candidates failed.",
                    "Micro segment too short for Korean TTS.",
                }
                and not error.startswith("GPT-SoVITS synthesis failed")
                and not error.startswith("Korean TTS preflight blocked synthesis")
            ]

        def record_skipped_duration_rewrite_retry(
            segment: Segment,
            rewrite_metadata: dict[str, Any],
        ) -> None:
            segment.analysis.setdefault("duration_rewrite_history", []).append(
                copy.deepcopy(rewrite_metadata)
            )
            segment.analysis["duration_rewrite_retry_skipped"] = copy.deepcopy(rewrite_metadata)
            segment.analysis.pop("pending_duration_rewrite", None)

        def run_deferred_duration_rewrites() -> set[str]:
            nonlocal duration_rewrite_client, duration_rewrite_running
            requests = [
                request
                for index, segment in enumerate(manifest.segments, start=1)
                if only_segment_ids is None or segment.id in only_segment_ids
                for request in [duration_rewrite_request_for_segment(index, segment)]
                if request is not None
            ]
            if not requests:
                return set()
            stop_gsv_servers()
            if duration_rewrite_manager is not None:
                duration_rewrite_manager.start()
                duration_rewrite_running = True
            duration_rewrite_client = LlamaServerTranslationClient(
                duration_rewrite_base_url,
                timeout_sec=cfg.gemma_text_timeout_sec,
                retries=cfg.gemma_text_retries,
                n_predict=cfg.gemma_text_n_predict,
                model=cfg.gemma_llama_cpp_model_path,
                two_pass=False,
            )
            retry_segment_ids: set[str] = set()
            try:
                for request in requests:
                    segment = request["segment"]
                    rewritten, rewrite_metadata = maybe_rewrite_script_with_gemma(
                        segment=segment,
                        index=int(request["index"]),
                        attempt_text=str(request["attempt_text"]),
                        actual_duration_sec=float(request["actual_duration_sec"]),
                        reason=str(request["reason"]),
                        rewrite_attempts_used=0,
                    )
                    rewrite_metadata["deferred"] = True
                    should_retry_rewrite = _should_retry_duration_rewrite_result(
                        rewritten,
                        segment.script,
                    )
                    rewrite_metadata["retry_scheduled"] = should_retry_rewrite
                    _log_duration_rewrite_result(segment.id, rewrite_metadata)
                    if should_retry_rewrite:
                        segment.script = rewritten
                        reset_segment_for_duration_rewrite_retry(segment, rewrite_metadata)
                        retry_segment_ids.add(segment.id)
                    else:
                        record_skipped_duration_rewrite_retry(segment, rewrite_metadata)
                    save_manifest(project_dir, manifest)
            finally:
                duration_rewrite_client = None
                if duration_rewrite_manager is not None and duration_rewrite_running:
                    duration_rewrite_manager.stop()
                    duration_rewrite_running = False
            return retry_segment_ids

        segment_jobs = [
            (index, segment, _segment_lane_index(segment, index - 1, gsv_lane_count))
            for index, segment in enumerate(manifest.segments, start=1)
            if only_segment_ids is None or segment.id in only_segment_ids
        ]
        countdown_rendered_segment_ids = render_countdown_spans(segment_jobs)
        run_synth_jobs(
            [
                (index, segment, lane_index)
                for index, segment, lane_index in segment_jobs
                if segment.id not in countdown_rendered_segment_ids
            ]
        )
        if duration_rewrite_enabled:
            duration_rewrite_retry_segment_ids = run_deferred_duration_rewrites()
            if duration_rewrite_retry_segment_ids:
                duration_rewrite_phase = "post_rewrite"
                retry_jobs = [
                    (index, segment, _segment_lane_index(segment, index - 1, gsv_lane_count))
                    for index, segment in enumerate(manifest.segments, start=1)
                    if segment.id in duration_rewrite_retry_segment_ids
                    and (only_segment_ids is None or segment.id in only_segment_ids)
                ]
                run_synth_jobs(retry_jobs)
        failed_after_primary = selected_failed_segment_ids()
        fine_tuned_retry_enabled = bool(
            not mock
            and failed_after_primary
            and (gpt_weights or sovits_weights or use_speaker_gsv)
        )
        if fine_tuned_retry_enabled:
            fine_tuned_retry_summary = run_internal_retry_pass(
                pass_name="fine_tuned_retry",
                segment_ids=failed_after_primary,
            )
        failed_after_fine_tuned_retry = selected_failed_segment_ids()
        static_ref_retry_segments = failed_segment_ids_with_used_segment_ref(
            failed_after_fine_tuned_retry
        )
        static_ref_retry_enabled = bool(
            not mock
            and static_ref_retry_segments
            and getattr(cfg, "gsv_ref_mode", "static") in {"segment", "auto"}
        )
        if static_ref_retry_enabled:
            static_ref_retry_summary = run_internal_retry_pass(
                pass_name="static_ref_retry",
                segment_ids=static_ref_retry_segments,
                static_ref_retry=True,
            )
        failed_after_static_ref_retry = selected_failed_segment_ids()
        low_temperature_retry_enabled = bool(
            not mock
            and failed_after_static_ref_retry
            and getattr(cfg, "gsv_low_temperature_retry_enabled", True)
        )
        if low_temperature_retry_enabled:
            low_temperature_retry_summary = run_internal_retry_pass(
                pass_name="low_temperature_retry",
                segment_ids=failed_after_static_ref_retry,
                temperature_override=float(
                    getattr(cfg, "gsv_low_temperature_retry_temperature", 0.5)
                ),
            )
        failed_after_low_temperature_retry = selected_failed_segment_ids()
        zero_shot_fallback_enabled = bool(
            not mock
            and failed_after_low_temperature_retry
            and not use_speaker_gsv
            and (gpt_weights or sovits_weights)
        )
        if zero_shot_fallback_enabled:
            zero_shot_fallback_summary = run_internal_retry_pass(
                pass_name="zero_shot_fallback",
                segment_ids=failed_after_low_temperature_retry,
                zero_shot=True,
            )
        synth_pass = "complete"
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
        recovery_metadata = {}
        if fine_tuned_retry_summary is not None:
            recovery_metadata["fine_tuned_retry"] = fine_tuned_retry_summary
        if static_ref_retry_summary is not None:
            recovery_metadata["static_ref_retry"] = static_ref_retry_summary
        if low_temperature_retry_summary is not None:
            recovery_metadata["low_temperature_retry"] = low_temperature_retry_summary
        if zero_shot_fallback_summary is not None:
            recovery_metadata["zero_shot_fallback"] = zero_shot_fallback_summary
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
                retry_failed=retry_failed,
                force=force,
                segment_counts=_segment_counts(manifest),
                **recovery_metadata,
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
            retry_failed=retry_failed,
            force=force,
            segment_counts=_segment_counts(manifest),
            **recovery_metadata,
        )
        save_manifest(project_dir, manifest)
        _log_stage_complete("synth", manifest, f"backend={synth_backend_name}")
        return ctx.update_manifest(manifest)
    finally:
        if gsv_servers_running:
            for server_manager in reversed(server_managers):
                server_manager.stop()
        if duration_rewrite_manager is not None and duration_rewrite_running:
            duration_rewrite_manager.stop()
