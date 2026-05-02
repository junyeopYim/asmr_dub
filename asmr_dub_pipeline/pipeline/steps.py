from __future__ import annotations

import json
import math
import re
import shlex
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from inspect import signature
from pathlib import Path
from threading import Lock
from time import monotonic
from typing import Any
from urllib.parse import urlparse, urlunparse

import numpy as np
import soundfile as sf
from rich.markup import escape

from asmr_dub_pipeline.asr import ASRChunk, create_asr_backend, map_chunks_to_segments
from asmr_dub_pipeline.audio import ffmpeg
from asmr_dub_pipeline.audio.duration import duration_ratio, suggest_speed_factor
from asmr_dub_pipeline.audio.features import (
    clipping_ratio,
    duration_sec,
    load_audio,
    peak_dbfs,
    rms_dbfs,
    trim_edge_silence,
    write_audio,
)
from asmr_dub_pipeline.audio.mixing import (
    build_dialogue_stem,
    build_source_suppressed_background,
    mix_with_background,
)
from asmr_dub_pipeline.audio.preprocess import extract_project_audio, probe_with_fallback
from asmr_dub_pipeline.audio.quality import AudioQualityMetrics, measure_source_voice_quality
from asmr_dub_pipeline.audio.segmentation import (
    energy_segments,
    load_manual_segments,
    write_segment_audio_clips,
)
from asmr_dub_pipeline.audio.separation import (
    SourceSeparationUnavailable,
    separate_source_audio,
)
from asmr_dub_pipeline.config import (
    create_project_structure,
    load_project_config,
    save_project_config,
)
from asmr_dub_pipeline.experimental_tts import (
    CosyVoiceTTSClient,
    ExperimentalTTSError,
    ExperimentalTTSRequest,
    FishSpeechTTSClient,
)
from asmr_dub_pipeline.gemma.base import create_gemma_backend
from asmr_dub_pipeline.gemma.schemas import validate_gemma_task_response
from asmr_dub_pipeline.gemma.text_server import (
    ManagedGemmaTextServer,
    default_llama_server_command,
)
from asmr_dub_pipeline.gemma.text_translate import (
    LlamaServerTranslationClient,
    MockTranslationClient,
)
from asmr_dub_pipeline.gpt_sovits.client import (
    GPTSoVITSClient,
    GPTSoVITSError,
    normalize_api_language_code,
)
from asmr_dub_pipeline.gpt_sovits.few_shot import (
    FEW_SHOT_ARTIFACT_GPT,
    FEW_SHOT_ARTIFACT_SOVITS,
    FEW_SHOT_STAGE,
    FewShotTrainingProgress,
    train_few_shot,
)
from asmr_dub_pipeline.gpt_sovits.refs import load_refs, resolve_ref, resolve_refs_json_path
from asmr_dub_pipeline.gpt_sovits.retry import (
    GPTSoVITSRetrySignal,
    adjust_for_repetition_or_omission,
    adjust_speed_for_duration,
    adjust_speed_for_short_duration,
    duration_too_long,
    duration_too_short,
    retry_signal_values,
)
from asmr_dub_pipeline.gpt_sovits.schemas import GPTSoVITSRef, GPTSoVITSTTSOptions
from asmr_dub_pipeline.gpt_sovits.server import ManagedGPTSoVITSServer
from asmr_dub_pipeline.logging import console
from asmr_dub_pipeline.pipeline.manifest_io import load_manifest, save_manifest, write_json_atomic
from asmr_dub_pipeline.pipeline.state import mark_stage
from asmr_dub_pipeline.qc.audio_qc import measure_audio_qc
from asmr_dub_pipeline.qc.scoring import score_qc
from asmr_dub_pipeline.qwen_tts import QwenTTSClient, QwenTTSError, QwenTTSRequest, qwen_language
from asmr_dub_pipeline.rights import (
    RightsError,
    ensure_inside_project,
    ensure_not_same_path,
    merge_rights_audit,
    record_rights_reliance,
    require_confirmed_rights,
    require_existing_or_confirmed_rights,
    sha256_file,
)
from asmr_dub_pipeline.rvc import (
    RVCBatchCommandClient,
    RVCBatchJob,
    RVCCommandClient,
    RVCCommandError,
    RVCCommandResult,
    RVCMockClient,
    RVCTrainCommandClient,
    RVCTrainMockClient,
    resolve_config_path,
    rvc_train_output_paths,
    validate_rvc_config,
    validate_rvc_training_config,
)
from asmr_dub_pipeline.schemas import (
    JapaneseScript,
    PipelineManifest,
    ProjectConfig,
    RVCMetadata,
    RVCProfile,
    Segment,
    SourceScript,
    TTSCandidate,
    TTSMetadata,
)
from asmr_dub_pipeline.script.duration_rewrite import rewrite_for_duration
from asmr_dub_pipeline.script.korean_colloquial import (
    COLLOQUIAL_REWRITE_NOTE,
    colloquialize_korean_translation,
)
from asmr_dub_pipeline.script.normalizer import normalize_korean_tts_text, normalize_script_payload
from asmr_dub_pipeline.script.text_qc import preflight_tts_text
from asmr_dub_pipeline.voice_bank import (
    apply_voice_bank_to_config,
    assign_speakers_to_manifest,
    load_voice_bank,
    resolve_voice_bank_path,
    validate_voice_bank_models,
)

SKIP_STATUSES = {"needs_manual_review", "failed"}
GEMMA_TEXT_SERVER_UNAVAILABLE_MARKERS = (
    "Connection refused",
    "Connection reset",
    "Server disconnected",
    "All connection attempts failed",
)
LEGACY_DETERMINISTIC_NUMERIC_TRANSLATION_NOTE = "deterministic_numeric_source"
NUMERIC_COUNTING_POSTPROCESS_NOTE = "numeric_counting_postprocess"
NUMERIC_ONLY_SOURCE_RE = re.compile(r"^\s*\d+(?:[\s,]+\d+)*\s*$")
NUMERIC_TOKEN_RE = re.compile(r"\d+")
KOREAN_DRAFT_MIX_ALLOWED_QC_ISSUES = {"duration_ratio_out_of_range", "too_much_silence"}
PROGRESS_LOG_SECONDS = 30.0
JAPANESE_TTS_LANGUAGES = {"ja", "jp", "jpn", "japanese"}
KOREAN_TTS_LANGUAGES = {"ko", "kr", "kor", "korean"}
NATIVE_KOREAN_COUNT_ONES = {
    0: "영",
    1: "하나",
    2: "둘",
    3: "셋",
    4: "넷",
    5: "다섯",
    6: "여섯",
    7: "일곱",
    8: "여덟",
    9: "아홉",
}
NATIVE_KOREAN_COUNT_TENS = {
    10: "열",
    20: "스물",
    30: "서른",
    40: "마흔",
    50: "쉰",
    60: "예순",
    70: "일흔",
    80: "여든",
    90: "아흔",
}
RVC_REQUIRED_MESSAGE = (
    "RVC is required but has not completed. Run `asmr-dub rvc --project ... "
    "--confirm-rights` or use `asmr-dub full --real ...`."
)


@dataclass(frozen=True)
class _VoiceRefSpan:
    segments: tuple[Segment, ...]
    duration: float


@dataclass(frozen=True)
class _SourceSeparationCacheCandidate:
    source_dir: Path
    matched_by: str
    paths: dict[str, Path]


@dataclass
class _QwenSegmentSynthesisJob:
    index: int
    segment: Segment
    ref: GPTSoVITSRef
    resolved_ref_style: str
    speaker_refs_path: Path | None
    candidates: list[TTSCandidate]


@dataclass
class _ExperimentalTTSSegmentSynthesisJob:
    index: int
    segment: Segment
    ref: GPTSoVITSRef
    resolved_ref_style: str
    speaker_refs_path: Path | None
    candidates: list[TTSCandidate]


@dataclass(frozen=True)
class _ExperimentalTTSBackendSpec:
    stage: str
    backend_name: str
    analysis_key: str
    artifact_key: str
    work_dir_name: str
    file_label: str


def _canonical_language(value: str | None) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_")
    if normalized in JAPANESE_TTS_LANGUAGES:
        return "ja"
    if normalized in KOREAN_TTS_LANGUAGES:
        return "ko"
    return normalized


def _segment_counts(manifest: PipelineManifest) -> dict[str, int]:
    counts: dict[str, int] = {}
    for segment in manifest.segments:
        counts[segment.status] = counts.get(segment.status, 0) + 1
    return counts


def _format_elapsed(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, second = divmod(seconds, 60)
    hour, minute = divmod(minutes, 60)
    return f"{hour:02d}:{minute:02d}:{second:02d}"


def _format_segment_counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{key}:{counts[key]}" for key in sorted(counts))


def _parallel_base_urls(base_url: str, count: int) -> list[str]:
    if count <= 1:
        return [base_url.rstrip("/")]
    parsed = urlparse(base_url)
    if not parsed.scheme or not parsed.hostname:
        raise ValueError(f"Parallel server URL must include scheme and host: {base_url}")
    default_port = 443 if parsed.scheme == "https" else 80
    start_port = parsed.port or default_port
    host = parsed.hostname
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    userinfo = ""
    if parsed.username:
        userinfo = parsed.username
        if parsed.password:
            userinfo += f":{parsed.password}"
        userinfo += "@"
    path = parsed.path.rstrip("/")
    urls: list[str] = []
    for offset in range(count):
        netloc = f"{userinfo}{host}:{start_port + offset}"
        urls.append(urlunparse((parsed.scheme, netloc, path, "", "", "")))
    return urls


def _segment_lane_index(segment: Segment, fallback_index: int, lane_count: int) -> int:
    if lane_count <= 1:
        return 0
    suffix = segment.id.rsplit("_", 1)[-1]
    if suffix.isdigit():
        return max(0, int(suffix) - 1) % lane_count
    return max(0, fallback_index) % lane_count


def _effective_lane_count(configured: int, item_count: int) -> int:
    return min(max(1, int(configured)), max(1, item_count))


def _log_text_snippet(text: str | None, max_chars: int = 160) -> str:
    normalized = " ".join((text or "").split())
    if not normalized:
        return "(empty)"
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 1].rstrip() + "..."


def _manifest_tts_languages(manifest: PipelineManifest) -> set[str]:
    languages: set[str] = set()
    for segment in manifest.segments:
        if not segment.script:
            continue
        language = _canonical_language(segment.script.tts_language)
        if language:
            languages.add(language)
    return languages


def _segment_tts_text_language(segment: Segment, project_target_language: str) -> str:
    script_language = _canonical_language(segment.script.tts_language if segment.script else None)
    target_language = _canonical_language(project_target_language) or "ko"
    if script_language == "ko" or target_language == "ko":
        return "ko"
    return script_language or target_language


def _same_project_path(project_dir: Path, left: str | None, right: str | None) -> bool:
    if not left or not right:
        return False

    def resolve(raw: str) -> Path:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = project_dir / path
        return path.resolve()

    return resolve(left) == resolve(right)


def _raise_unsafe_korean_few_shot_gpt() -> None:
    raise GPTSoVITSError(
        "Korean TTS cannot safely use the project's Japanese source-trained few-shot GPT "
        "weights. Use gsv_gpt_weights_policy: auto or gsv_gpt_weights_policy: "
        "base_for_korean, and keep few-shot SoVITS/RVC for timbre."
    )


def _resolve_manifest_path(project_dir: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_dir / path
    return path


def _few_shot_base_gpt_weights(project_dir: Path, manifest: PipelineManifest) -> str | None:
    candidates: list[Path] = []
    metadata_path = _resolve_manifest_path(project_dir, manifest.artifacts.get("gsv_few_shot_manifest"))
    if metadata_path and metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text("utf-8"))
        except json.JSONDecodeError:
            metadata = {}
        gsv_metadata = metadata.get("fingerprint_payload", {}).get("gpt_sovits", {})
        pretrained_gpt = _resolve_manifest_path(project_dir, gsv_metadata.get("pretrained_gpt_path"))
        if pretrained_gpt:
            candidates.append(pretrained_gpt)
    repo_root = Path(__file__).resolve().parents[2]
    candidates.extend(
        [
            repo_root / ".cache" / "gpt_sovits" / "GPT_SoVITS" / "pretrained_models" / "s1v3.ckpt",
            repo_root
            / ".cache"
            / "third_party"
            / "GPT-SoVITS"
            / "GPT_SoVITS"
            / "pretrained_models"
            / "s1v3.ckpt",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return None


def _resolve_gpt_weights_for_tts(
    project_dir: Path,
    manifest: PipelineManifest,
    cfg: Any,
    explicit_gpt_weights_path: str | None,
    model_switch: dict[str, str],
) -> str | None:
    policy = getattr(cfg, "gsv_gpt_weights_policy", "auto")
    few_shot_gpt = manifest.artifacts.get(FEW_SHOT_ARTIFACT_GPT)
    source_language = _canonical_language(getattr(cfg, "source_language", "ja"))
    target_language = _canonical_language(getattr(cfg, "target_language", "ko"))
    model_switch["source_language"] = source_language
    model_switch["target_language"] = target_language
    korean_tts_from_japanese_source = source_language == "ja" and target_language == "ko"
    configured = explicit_gpt_weights_path or cfg.gsv_gpt_weights_path
    if configured:
        model_switch["gpt_weights_mode"] = "explicit"
        if korean_tts_from_japanese_source and _same_project_path(project_dir, configured, few_shot_gpt):
            _raise_unsafe_korean_few_shot_gpt()
        return configured

    if policy == "unchanged":
        model_switch["gpt_weights_mode"] = "unchanged_by_policy"
        return None
    if not few_shot_gpt:
        return None

    languages = _manifest_tts_languages(manifest)
    model_switch["tts_languages"] = ",".join(sorted(languages)) if languages else "unknown"
    if policy == "few_shot":
        if korean_tts_from_japanese_source:
            _raise_unsafe_korean_few_shot_gpt()
        model_switch["gpt_weights_mode"] = "few_shot"
        return few_shot_gpt
    if target_language == "ja" and languages and languages <= {"ja"}:
        model_switch["gpt_weights_mode"] = "few_shot"
        return few_shot_gpt

    base_gpt = _few_shot_base_gpt_weights(project_dir, manifest)
    model_switch["gpt_weights_skipped_path"] = few_shot_gpt
    model_switch["gpt_weights_skip_reason"] = (
        "source_few_shot_gpt_language_ja_does_not_match_tts_output_ko"
        if target_language == "ko"
        else "few-shot GPT was trained from source-language clips; using base GPT for non-Japanese TTS text"
    )
    if base_gpt:
        model_switch["gpt_weights_mode"] = "base_for_non_japanese_tts"
        return base_gpt
    if target_language == "ko" and policy in {"auto", "base_for_korean"}:
        raise GPTSoVITSError(
            "Korean TTS output cannot safely load Japanese source-trained few-shot GPT weights, "
            "and no base GPT weights were found. Configure gsv_gpt_weights_path or set "
            "gsv_gpt_weights_policy='unchanged' explicitly."
        )
    model_switch["gpt_weights_mode"] = "unchanged_no_base_gpt_found"
    return None


def _gsv_speaker_cfg(cfg: ProjectConfig, segment: Segment):
    if not segment.speaker_id:
        return None
    return cfg.gsv_speaker_models.get(segment.speaker_id)


def _resolve_gsv_speaker_path(project_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    return path.resolve() if path.is_absolute() else (project_dir / path).resolve()


def _active_segments_requiring_voice_bank(manifest: PipelineManifest) -> list[Segment]:
    return [segment for segment in manifest.segments if segment.status not in SKIP_STATUSES]


def _validate_gsv_speaker_models(project_dir: Path, manifest: PipelineManifest) -> None:
    cfg = manifest.project_config
    if not cfg.gsv_speaker_models:
        return
    missing: list[str] = []
    errors: list[str] = []
    for segment in _active_segments_requiring_voice_bank(manifest):
        if not segment.speaker_id or segment.speaker_id not in cfg.gsv_speaker_models:
            missing.append(segment.id)
            continue
        speaker_cfg = cfg.gsv_speaker_models[segment.speaker_id]
        if not speaker_cfg.gpt_weights_path:
            errors.append(f"{segment.speaker_id} GPT weights missing: <not configured>")
        for label, raw_path in (
            ("GPT weights", speaker_cfg.gpt_weights_path),
            ("SoVITS weights", speaker_cfg.sovits_weights_path),
            ("refs", speaker_cfg.refs_path),
        ):
            if not raw_path:
                continue
            path = _resolve_gsv_speaker_path(project_dir, raw_path)
            if not path.exists():
                errors.append(f"{segment.speaker_id} {label} missing: {path}")
    if missing:
        errors.append("missing speaker model mapping for segments: " + ", ".join(missing[:20]))
    if errors:
        raise ValueError("Invalid GPT-SoVITS voice bank speaker models: " + "; ".join(errors))


def assign_speakers_step(
    project_dir: Path,
    voice_bank_path: Path | None = None,
    backend_kind: str | None = None,
    require_all: bool = True,
) -> PipelineManifest:
    manifest = load_manifest(project_dir)
    _load_config_into_manifest(project_dir, manifest)
    cfg = manifest.project_config
    bank = load_voice_bank(project_dir, cfg, voice_bank_path)
    validate_voice_bank_models(project_dir, bank)
    next_cfg = apply_voice_bank_to_config(project_dir, cfg, bank)
    save_project_config(next_cfg, project_dir / "pipeline.yaml")
    manifest.project_config = next_cfg
    backend = backend_kind or cfg.speaker_assignment_backend
    if backend == "none":
        backend = "mock"
    assign_speakers_to_manifest(
        project_dir,
        manifest,
        bank,
        backend_kind=backend,
        require_all=require_all,
    )
    manifest.artifacts["voice_bank"] = str(resolve_voice_bank_path(project_dir, next_cfg, voice_bank_path))
    save_manifest(project_dir, manifest)
    _log_stage_complete("speaker-assign", manifest, f"backend={backend}")
    return manifest


def _ref_for_tts_language(ref: GPTSoVITSRef, tts_language: str) -> GPTSoVITSRef:
    _ = tts_language
    return ref


def _tts_request_debug_payload(
    text: str,
    ref: GPTSoVITSRef,
    options: GPTSoVITSTTSOptions,
) -> dict[str, Any]:
    return {
        "text": text,
        "text_lang": normalize_api_language_code(options.text_lang),
        "prompt_text": ref.prompt_text,
        "prompt_lang": normalize_api_language_code(ref.prompt_lang),
        "seed": options.seed,
        "speed_factor": options.speed_factor,
        "top_k": options.top_k,
        "top_p": options.top_p,
        "temperature": options.temperature,
    }


def _can_rewrite_script_for_duration(script: JapaneseScript) -> bool:
    normalized = script.tts_language.strip().lower().replace("-", "_")
    return normalized in JAPANESE_TTS_LANGUAGES or normalized in KOREAN_TTS_LANGUAGES


def _valid_asr_chunks(
    chunks: list[ASRChunk],
    audio_duration_sec: float | None,
    *,
    max_chunk_sec: float,
    sparse_chunk_max_sec: float,
    sparse_chunk_min_chars_per_sec: float,
) -> list[ASRChunk]:
    valid: list[ASRChunk] = []
    for chunk in sorted(chunks, key=lambda item: (item.start, item.end)):
        text = chunk.text.strip()
        if not text:
            continue
        start = max(0.0, float(chunk.start))
        end = max(start, float(chunk.end))
        if audio_duration_sec is not None:
            end = min(float(audio_duration_sec), end)
        if end - start <= 0.05:
            continue
        duration = end - start
        if (
            duration > sparse_chunk_max_sec
            and sparse_chunk_min_chars_per_sec > 0
            and len(text) / duration < sparse_chunk_min_chars_per_sec
        ):
            continue
        normalized = chunk.model_copy(update={"start": start, "end": end, "text": text})
        valid.extend(_split_long_asr_chunk(normalized, max_chunk_sec=max_chunk_sec))
    return valid


def _split_long_asr_chunk(chunk: ASRChunk, *, max_chunk_sec: float) -> list[ASRChunk]:
    duration = chunk.end - chunk.start
    if duration <= max_chunk_sec:
        return [chunk]
    part_count = max(1, int(math.ceil(duration / max_chunk_sec)))
    text = chunk.text.strip()
    if not text:
        return []
    chars_per_part = max(1, int(math.ceil(len(text) / part_count)))
    parts = [text[index : index + chars_per_part].strip() for index in range(0, len(text), chars_per_part)]
    parts = [part for part in parts if part]
    if len(parts) <= 1:
        return [chunk]
    part_duration = duration / len(parts)
    split_chunks: list[ASRChunk] = []
    for index, part in enumerate(parts):
        start = chunk.start + part_duration * index
        end = chunk.end if index == len(parts) - 1 else chunk.start + part_duration * (index + 1)
        split_chunks.append(chunk.model_copy(update={"start": start, "end": end, "text": part}))
    return split_chunks


def _group_asr_chunks_for_tts(
    chunks: list[ASRChunk],
    *,
    min_segment_sec: float,
    merge_gap_sec: float,
) -> list[list[ASRChunk]]:
    groups: list[list[ASRChunk]] = []
    current: list[ASRChunk] = []
    for chunk in chunks:
        if not current:
            current = [chunk]
            continue
        current_duration = current[-1].end - current[0].start
        gap = chunk.start - current[-1].end
        if current_duration < min_segment_sec and gap <= merge_gap_sec:
            current.append(chunk)
            continue
        groups.append(current)
        current = [chunk]
    if current:
        groups.append(current)
    return groups


def _segments_from_asr_chunks(
    chunks: list[ASRChunk],
    *,
    project_dir: Path,
    backend: str,
    fallback_language: str,
    audio_duration_sec: float | None,
    min_segment_sec: float,
    merge_gap_sec: float,
    max_segment_sec: float = 20.0,
    sparse_chunk_max_sec: float = 30.0,
    sparse_chunk_min_chars_per_sec: float = 0.5,
) -> list[Segment]:
    groups = _group_asr_chunks_for_tts(
        _valid_asr_chunks(
            chunks,
            audio_duration_sec,
            max_chunk_sec=max_segment_sec,
            sparse_chunk_max_sec=sparse_chunk_max_sec,
            sparse_chunk_min_chars_per_sec=sparse_chunk_min_chars_per_sec,
        ),
        min_segment_sec=min_segment_sec,
        merge_gap_sec=merge_gap_sec,
    )
    segments: list[Segment] = []
    last_end = 0.0
    for index, group in enumerate(groups, start=1):
        start = max(last_end, group[0].start)
        end = max(start + 0.05, group[-1].end)
        start = round(start, 3)
        end = round(end, 3)
        if end <= start:
            continue
        duration = round(end - start, 3)
        seg_id = f"seg_{index:04d}"
        text = " ".join(chunk.text.strip() for chunk in group if chunk.text.strip()).strip()
        language = next((chunk.language for chunk in group if chunk.language), fallback_language)
        confidence = _asr_group_confidence(group)
        audio_base = project_dir / "work" / "segments" / "audio"
        segments.append(
            Segment(
                id=seg_id,
                start=start,
                end=end,
                duration=duration,
                audio_for_gemma=str(audio_base / f"{seg_id}_gemma.wav"),
                audio_for_mix=str(audio_base / f"{seg_id}_mix.wav"),
                keep_original_texture=True,
                source_script=SourceScript(
                    text=text,
                    language=language,
                    confidence=confidence,
                    backend=backend,
                    start=start,
                    end=end,
                ),
            )
        )
        last_end = end
    return segments


def _asr_group_confidence(group: list[ASRChunk]) -> float | None:
    weighted = [
        (float(chunk.confidence), max(0.0, chunk.end - chunk.start))
        for chunk in group
        if chunk.confidence is not None
    ]
    total = sum(duration for _, duration in weighted)
    if total <= 0:
        return None
    return sum(confidence * duration for confidence, duration in weighted) / total


def _asr_text_density(chunk: ASRChunk) -> float:
    duration = max(0.001, chunk.end - chunk.start)
    return len(chunk.text.strip()) / duration


def _asr_text_matches_suspicious_pattern(
    text: str,
    suspicious_text_patterns: list[str] | tuple[str, ...],
) -> bool:
    for pattern in suspicious_text_patterns:
        pattern = pattern.strip()
        if not pattern:
            continue
        try:
            if re.search(pattern, text):
                return True
        except re.error:
            if pattern in text:
                return True
    return False


def _asr_chunk_needs_repair(
    chunk: ASRChunk,
    *,
    confidence_threshold: float,
    sparse_min_sec: float,
    sparse_min_chars_per_sec: float,
    suspicious_text_patterns: list[str] | tuple[str, ...] = (),
) -> bool:
    text = chunk.text.strip()
    if NUMERIC_ONLY_SOURCE_RE.fullmatch(text):
        return False
    if _asr_text_matches_suspicious_pattern(text, suspicious_text_patterns):
        return True
    if chunk.confidence is not None and chunk.confidence < confidence_threshold:
        return True
    duration = chunk.end - chunk.start
    return duration >= sparse_min_sec and _asr_text_density(chunk) < sparse_min_chars_per_sec


def _asr_candidate_confidence(chunks: list[ASRChunk]) -> float | None:
    weighted = [
        (float(chunk.confidence), max(0.0, chunk.end - chunk.start))
        for chunk in chunks
        if chunk.confidence is not None
    ]
    total = sum(duration for _, duration in weighted)
    if total <= 0:
        return None
    return sum(confidence * duration for confidence, duration in weighted) / total


def _asr_candidate_text(chunks: list[ASRChunk]) -> str:
    return " ".join(chunk.text.strip() for chunk in chunks if chunk.text.strip()).strip()


def _apply_asr_text_replacements_to_chunks(
    chunks: list[ASRChunk],
    replacements: dict[str, str],
) -> tuple[list[ASRChunk], int]:
    if not replacements:
        return chunks, 0
    replaced_count = 0
    normalized_chunks: list[ASRChunk] = []
    for chunk in chunks:
        text = chunk.text
        normalized = text
        for source, target in replacements.items():
            if source:
                normalized = normalized.replace(source, target)
        if normalized != text:
            replaced_count += 1
            normalized_chunks.append(chunk.model_copy(update={"text": normalized}))
        else:
            normalized_chunks.append(chunk)
    return normalized_chunks, replaced_count


def _asr_repair_candidate_is_better(
    original: ASRChunk,
    candidate_chunks: list[ASRChunk],
    *,
    confidence_threshold: float,
    sparse_min_sec: float,
    sparse_min_chars_per_sec: float,
    suspicious_text_patterns: list[str] | tuple[str, ...] = (),
) -> bool:
    candidate_text = _asr_candidate_text(candidate_chunks)
    original_text = original.text.strip()
    if not candidate_text:
        return False
    if NUMERIC_ONLY_SOURCE_RE.fullmatch(original_text):
        return False
    candidate_confidence = _asr_candidate_confidence(candidate_chunks)
    original_confidence = original.confidence
    if (
        original_confidence is not None
        and candidate_confidence is not None
        and candidate_confidence + 0.12 < original_confidence
    ):
        return False
    if len(candidate_text) < max(2, int(len(original_text) * 0.6)):
        return False

    original_needs_repair = _asr_chunk_needs_repair(
        original,
        confidence_threshold=confidence_threshold,
        sparse_min_sec=sparse_min_sec,
        sparse_min_chars_per_sec=sparse_min_chars_per_sec,
        suspicious_text_patterns=suspicious_text_patterns,
    )
    if not original_needs_repair:
        return False

    if _asr_text_matches_suspicious_pattern(original_text, suspicious_text_patterns):
        return candidate_text != original_text
    if original_confidence is not None and original_confidence < confidence_threshold:
        return True
    original_density = _asr_text_density(original)
    candidate_duration = max(0.001, candidate_chunks[-1].end - candidate_chunks[0].start)
    candidate_density = len(candidate_text) / candidate_duration
    return candidate_density > max(original_density * 1.2, sparse_min_chars_per_sec)


def _transcribe_with_backend_options(
    backend: Any,
    audio_path: Path,
    segments: list[Segment],
    **overrides: Any,
) -> list[ASRChunk]:
    method = getattr(backend, "transcribe_with_options", None)
    if callable(method):
        return method(audio_path, segments, **overrides)
    return backend.transcribe(audio_path, segments)


def _repair_asr_chunks(
    chunks: list[ASRChunk],
    *,
    backend: Any,
    project_dir: Path,
    repair_audio_path: Path,
    audio_duration_sec: float,
    cfg: Any,
) -> tuple[list[ASRChunk], dict[str, Any]]:
    summary: dict[str, Any] = {
        "enabled": bool(cfg.asr_repair_enabled),
        "attempted": 0,
        "repaired": 0,
        "skipped": 0,
        "audio_path": str(repair_audio_path),
        "items": [],
    }
    if not cfg.asr_repair_enabled or not chunks or not repair_audio_path.exists():
        return chunks, summary
    if not callable(getattr(backend, "transcribe_with_options", None)):
        summary["skipped"] = len(chunks)
        return chunks, summary

    repair_groups: list[tuple[bool, list[ASRChunk]]] = []
    current_repair_group: list[ASRChunk] = []
    repair_group_gap_sec = max(1.0, float(getattr(cfg, "asr_resegment_merge_gap_sec", 1.0)))
    for chunk in sorted(chunks, key=lambda item: (item.start, item.end)):
        needs_repair = _asr_chunk_needs_repair(
            chunk,
            confidence_threshold=cfg.asr_repair_confidence_threshold,
            sparse_min_sec=cfg.asr_repair_sparse_min_sec,
            sparse_min_chars_per_sec=cfg.asr_repair_sparse_min_chars_per_sec,
            suspicious_text_patterns=cfg.asr_repair_suspicious_text_patterns,
        )
        if needs_repair:
            if current_repair_group and chunk.start - current_repair_group[-1].end > repair_group_gap_sec:
                repair_groups.append((True, current_repair_group))
                current_repair_group = []
            current_repair_group.append(chunk)
            continue
        if current_repair_group:
            repair_groups.append((True, current_repair_group))
            current_repair_group = []
        repair_groups.append((False, [chunk]))
    if current_repair_group:
        repair_groups.append((True, current_repair_group))

    repaired_chunks: list[ASRChunk] = []
    repair_dir = project_dir / "work" / "transcribe" / "repair_clips"
    attempted = 0
    for should_repair, group in repair_groups:
        if not should_repair:
            repaired_chunks.extend(group)
            continue
        if attempted >= cfg.asr_repair_max_chunks:
            repaired_chunks.extend(group)
            summary["skipped"] += len(group)
            continue

        attempted += 1
        group_start = group[0].start
        group_end = group[-1].end
        clip_start = max(0.0, group_start - cfg.asr_repair_padding_sec)
        clip_end = min(audio_duration_sec, group_end + cfg.asr_repair_padding_sec)
        clip_path = repair_dir / f"repair_{attempted:04d}.wav"
        ffmpeg.slice_audio(
            repair_audio_path,
            clip_start,
            clip_end,
            clip_path,
            sample_rate=cfg.gemma_sample_rate,
            channels=1,
        )
        candidate_local = _transcribe_with_backend_options(
            backend,
            clip_path,
            [],
            vad_filter=False,
            vad_parameters=None,
            condition_on_previous_text=False,
            word_timestamps=False,
            hallucination_silence_threshold=None,
        )
        candidate_abs = [
            candidate.model_copy(
                update={
                    "start": max(group_start, min(group_end, clip_start + candidate.start)),
                    "end": max(group_start, min(group_end, clip_start + candidate.end)),
                }
            )
            for candidate in candidate_local
        ]
        candidate_abs = _valid_asr_chunks(
            candidate_abs,
            audio_duration_sec,
            max_chunk_sec=cfg.asr_resegment_max_sec,
            sparse_chunk_max_sec=cfg.asr_sparse_chunk_max_sec,
            sparse_chunk_min_chars_per_sec=cfg.asr_sparse_chunk_min_chars_per_sec,
        )
        original = ASRChunk(
            start=group_start,
            end=group_end,
            text=_asr_candidate_text(group),
            language=group[0].language,
            confidence=_asr_candidate_confidence(group),
        )
        accepted = _asr_repair_candidate_is_better(
            original,
            candidate_abs,
            confidence_threshold=cfg.asr_repair_confidence_threshold,
            sparse_min_sec=cfg.asr_repair_sparse_min_sec,
            sparse_min_chars_per_sec=cfg.asr_repair_sparse_min_chars_per_sec,
            suspicious_text_patterns=cfg.asr_repair_suspicious_text_patterns,
        )
        summary["items"].append(
            {
                "start": round(group_start, 3),
                "end": round(group_end, 3),
                "accepted": accepted,
                "original_text": original.text,
                "candidate_text": _asr_candidate_text(candidate_abs),
            }
        )
        if accepted:
            repaired_chunks.extend(candidate_abs)
            summary["repaired"] += 1
        else:
            repaired_chunks.extend(group)

    summary["attempted"] = attempted
    return sorted(repaired_chunks, key=lambda item: (item.start, item.end)), summary


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        "utf-8",
    )
    tmp.replace(path)
    return path


def _is_gemma_text_server_unavailable(exc: Exception) -> bool:
    message = str(exc)
    return any(marker in message for marker in GEMMA_TEXT_SERVER_UNAVAILABLE_MARKERS)


def _log_stage_start(stage: str, detail: str | None = None) -> None:
    suffix = f" - {detail}" if detail else ""
    console.print(f"[cyan]{stage}[/cyan] started{suffix}")


def _log_stage_complete(stage: str, manifest: PipelineManifest, detail: str | None = None) -> None:
    parts = [f"[green]{stage} complete[/green]"]
    if detail:
        parts.append(detail)
    if manifest.segments:
        parts.append(f"counts={_format_segment_counts(_segment_counts(manifest))}")
    console.print(" - ".join(parts))


def _format_command_preview(command: list[str], max_chars: int = 360) -> str:
    rendered = shlex.join(str(part) for part in command)
    if len(rendered) <= max_chars:
        return rendered
    return rendered[: max_chars - 3] + "..."


def _progress_interval(total: int) -> int:
    if total <= 20:
        return 1
    if total <= 300:
        return 10
    return 25


def _log_segment_progress(
    stage: str,
    index: int,
    total: int,
    segment: Segment,
    manifest: PipelineManifest | None,
    started_at: float,
    last_logged_at: float,
    note: str | None = None,
) -> float:
    now = monotonic()
    interval = _progress_interval(total)
    should_log = (
        index == 1
        or index == total
        or index % interval == 0
        or now - last_logged_at >= PROGRESS_LOG_SECONDS
    )
    if not should_log:
        return last_logged_at
    percent = (index / total * 100.0) if total else 100.0
    counts = (
        f" counts={_format_segment_counts(_segment_counts(manifest))}"
        if manifest and manifest.segments
        else ""
    )
    suffix = f" - {note}" if note else ""
    console.print(
        f"[dim]{stage}: {index}/{total} ({percent:.1f}%) "
        f"elapsed={_format_elapsed(now - started_at)} "
        f"latest={segment.id} status={segment.status}{counts}{suffix}[/dim]"
    )
    return now


def _format_cuda_memory_snapshot(snapshot: dict[str, float | int | str] | None) -> str:
    if not snapshot:
        return "cuda=unavailable"
    return (
        f"{snapshot['device']} allocated={snapshot['allocated_gb']:.2f}GB "
        f"reserved={snapshot['reserved_gb']:.2f}GB free={snapshot['free_gb']:.2f}GB "
        f"total={snapshot['total_gb']:.2f}GB target={snapshot['target_vram_gb']:.2f}GB"
    )


def _chunked(items: list[Any], size: int) -> list[list[Any]]:
    return [items[index : index + size] for index in range(0, len(items), max(1, size))]


def _log_translate_progress(
    index: int,
    total: int,
    segment: Segment,
    status: str,
    source_text: str | None,
    translated_text: str | None,
    started_at: float,
    last_logged_at: float,
) -> float:
    now = monotonic()
    interval = _progress_interval(total)
    should_log = (
        index == 1
        or index == total
        or index % interval == 0
        or now - last_logged_at >= PROGRESS_LOG_SECONDS
    )
    if not should_log:
        return last_logged_at
    percent = (index / total * 100.0) if total else 100.0
    console.print(
        f"[dim]translate-ko: {index}/{total} ({percent:.1f}%) "
        f"elapsed={_format_elapsed(now - started_at)} "
        f"latest={segment.id} status={status}[/dim]\n"
        f"[dim]  - 원문: {escape(_log_text_snippet(source_text))}[/dim]\n"
        f"[dim]  - 번역문: {escape(_log_text_snippet(translated_text))}[/dim]"
    )
    return now


def _apply_korean_colloquial_postprocess(segments: list[Segment]) -> int:
    rewritten_count = 0
    for segment in segments:
        translation = segment.translation_ko
        if translation is None or not translation.ko_natural.strip():
            continue
        if COLLOQUIAL_REWRITE_NOTE in translation.notes:
            continue
        rewritten = colloquialize_korean_translation(translation)
        if rewritten.ko_natural == translation.ko_natural:
            continue
        segment.translation_ko = rewritten
        rewritten_count += 1
    return rewritten_count


def _native_korean_count_number(value: int) -> str | None:
    if value in NATIVE_KOREAN_COUNT_ONES:
        return NATIVE_KOREAN_COUNT_ONES[value]
    if value in NATIVE_KOREAN_COUNT_TENS:
        return NATIVE_KOREAN_COUNT_TENS[value]
    if 10 < value < 100:
        tens, ones = divmod(value, 10)
        tens_text = NATIVE_KOREAN_COUNT_TENS.get(tens * 10)
        ones_text = NATIVE_KOREAN_COUNT_ONES.get(ones)
        if tens_text and ones_text:
            return tens_text + ones_text
    return None


def _numeric_source_values(segment: Segment) -> list[int] | None:
    source_text = _translation_source_text(segment)
    if not NUMERIC_ONLY_SOURCE_RE.fullmatch(source_text):
        return None
    tokens = NUMERIC_TOKEN_RE.findall(source_text)
    if not tokens:
        return None
    values: list[int] = []
    for token in tokens:
        if len(token) > 2:
            return None
        value = int(token)
        if not 0 <= value <= 99:
            return None
        values.append(value)
    return values


def _numeric_group_is_counting(group: list[tuple[Segment, list[int]]]) -> bool:
    values = [value for _, segment_values in group for value in segment_values]
    if len(values) < 4:
        return False
    transitions = [right - left for left, right in zip(values, values[1:], strict=False)]
    if not transitions:
        return False
    step_like = sum(1 for diff in transitions if 0 < abs(diff) <= 3)
    repeats = sum(1 for diff in transitions if diff == 0)
    large_jumps = sum(1 for diff in transitions if abs(diff) > 10)
    if large_jumps:
        return False
    if step_like < max(3, len(transitions) // 2):
        return False
    return repeats <= max(2, len(transitions) // 2)


def _numeric_counting_groups(segments: list[Segment]) -> list[list[tuple[Segment, list[int]]]]:
    groups: list[list[tuple[Segment, list[int]]]] = []
    current: list[tuple[Segment, list[int]]] = []
    last_numeric_segment: Segment | None = None
    last_numeric_value: int | None = None

    def flush_current() -> None:
        nonlocal current, last_numeric_segment, last_numeric_value
        if _numeric_group_is_counting(current):
            groups.append(current)
        current = []
        last_numeric_segment = None
        last_numeric_value = None

    for segment in segments:
        values = _numeric_source_values(segment)
        if values is None:
            continue
        if last_numeric_segment is not None and segment.start - last_numeric_segment.end > 8.0:
            flush_current()
        if last_numeric_value is not None and abs(values[0] - last_numeric_value) > 10:
            flush_current()
        current.append((segment, values))
        last_numeric_segment = segment
        last_numeric_value = values[-1]
    flush_current()
    return groups


def _apply_korean_numeric_counting_postprocess(segments: list[Segment]) -> int:
    rewritten_count = 0
    for group in _numeric_counting_groups(segments):
        for segment, values in group:
            translation = segment.translation_ko
            if translation is None or not translation.ko_natural.strip():
                continue
            spoken_values = [_native_korean_count_number(value) for value in values]
            if any(value is None for value in spoken_values):
                continue
            ko_natural = ", ".join(value for value in spoken_values if value is not None)
            if translation.ko_natural.strip() == ko_natural and translation.ko_literal.strip() == ko_natural:
                continue
            notes = list(translation.notes)
            if NUMERIC_COUNTING_POSTPROCESS_NOTE not in notes:
                notes.append(NUMERIC_COUNTING_POSTPROCESS_NOTE)
            segment.translation_ko = translation.model_copy(
                update={
                    "ko_literal": ko_natural,
                    "ko_natural": ko_natural,
                    "notes": notes,
                }
            )
            rewritten_count += 1
    return rewritten_count


def _refresh_translation_rows(rows: list[dict[str, Any]], segments: list[Segment]) -> None:
    by_segment_id = {segment.id: segment for segment in segments}
    for row in rows:
        if row.get("status") != "translated":
            continue
        segment = by_segment_id.get(str(row.get("segment_id") or ""))
        if segment is None or segment.translation_ko is None:
            continue
        row["translation_ko"] = segment.translation_ko.model_dump(mode="json")
        row["colloquialized"] = COLLOQUIAL_REWRITE_NOTE in segment.translation_ko.notes
        row["numeric_counting_postprocessed"] = (
            NUMERIC_COUNTING_POSTPROCESS_NOTE in segment.translation_ko.notes
        )


def _translation_context_segments(
    all_segments: list[Segment],
    target_segments: list[Segment],
    radius: int,
) -> list[Segment]:
    if not target_segments:
        return []
    if radius <= 0:
        return [
            segment
            for segment in target_segments
            if segment.source_script and segment.source_script.text.strip()
        ]
    indices_by_id = {segment.id: index for index, segment in enumerate(all_segments)}
    target_indices = [
        indices_by_id[segment.id]
        for segment in target_segments
        if segment.id in indices_by_id
    ]
    if not target_indices:
        return target_segments
    start = max(0, min(target_indices) - radius)
    end = min(len(all_segments), max(target_indices) + radius + 1)
    return [
        segment
        for segment in all_segments[start:end]
        if segment.source_script and segment.source_script.text.strip()
    ]


def _translation_source_text(segment: Segment) -> str:
    return segment.source_script.text.strip() if segment.source_script else ""


def _clear_korean_translation_errors(segment: Segment) -> None:
    segment.errors = [
        error
        for error in segment.errors
        if not error.startswith("Korean translation retry failed")
        and not error.startswith("Korean translation batch failed")
    ]


def _translation_span_batches(
    segments: list[Segment],
    *,
    max_segments: int,
    max_duration_sec: float,
    max_gap_sec: float,
) -> list[list[Segment]]:
    batches: list[list[Segment]] = []
    current: list[Segment] = []

    def flush_current() -> None:
        nonlocal current
        if current:
            batches.append(current)
            current = []

    for segment in segments:
        if current:
            gap = max(0.0, segment.start - current[-1].end)
            span_duration = max(segment.end, current[-1].end) - current[0].start
            if (
                len(current) >= max_segments
                or span_duration > max_duration_sec
                or gap > max_gap_sec
            ):
                flush_current()
        current.append(segment)
    flush_current()
    return batches


def _client_accepts_context_segments(client: Any) -> bool:
    try:
        parameters = signature(client.translate_batch).parameters
    except (TypeError, ValueError):
        return False
    return "context_segments" in parameters or any(
        parameter.kind == parameter.VAR_KEYWORD for parameter in parameters.values()
    )


def _translate_with_optional_context(
    client: Any,
    segments: list[Segment],
    batch_id: str,
    context_segments: list[Segment],
) -> dict[str, Any]:
    if _client_accepts_context_segments(client):
        return client.translate_batch(
            segments,
            batch_id,
            context_segments=context_segments,
        )
    return client.translate_batch(segments, batch_id)


def _resolve_project_read_path(project_dir: Path, raw_path: str, field_name: str) -> Path:
    path = Path(raw_path)
    resolved = (project_dir / path).resolve() if not path.is_absolute() else path.resolve()
    try:
        resolved.relative_to(project_dir.resolve())
    except ValueError as exc:
        raise RightsError(f"{field_name} must stay inside the project directory: {resolved}") from exc
    return resolved


def _validate_audio_contract(path: Path, sample_rate: int, channels: int, description: str) -> None:
    info = sf.info(str(path))
    if info.samplerate != sample_rate or info.channels != channels:
        raise ValueError(
            f"{description} must be {channels} channel(s) at {sample_rate} Hz: "
            f"{path} is {info.channels} channel(s) at {info.samplerate} Hz"
        )


def _safe_voice_bank_source_suffix(input_path: Path) -> str:
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", input_path.stem).strip("._-")
    return f"_{safe_stem or 'speaker'}"


def _same_audio_fingerprint(left: Path, right: Path) -> bool:
    try:
        if not left.exists() or not right.exists():
            return False
        if left.stat().st_size != right.stat().st_size:
            return False
        return sha256_file(left) == sha256_file(right)
    except OSError:
        return False


def _voice_bank_source_dirs(cache_project_dir: Path, input_path: Path) -> list[tuple[Path, str]]:
    sources_root = cache_project_dir / "voice_bank" / "sources"
    if not sources_root.is_dir():
        return []
    source_dirs = sorted(path for path in sources_root.iterdir() if path.is_dir())
    resolved_input = input_path.expanduser().resolve()
    candidates: list[tuple[Path, str]] = []
    seen: set[Path] = set()

    def add_candidate(source_dir: Path, matched_by: str) -> None:
        resolved_source_dir = source_dir.resolve()
        if resolved_source_dir in seen:
            return
        seen.add(resolved_source_dir)
        candidates.append((source_dir, matched_by))

    manifest_path = cache_project_dir / "voice_bank" / "voice_bank_manifest.json"
    if manifest_path.exists():
        try:
            voice_bank_data = json.loads(manifest_path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError, TypeError):
            voice_bank_data = {}
        source_paths = voice_bank_data.get("source_paths") if isinstance(voice_bank_data, dict) else None
        if isinstance(source_paths, list):
            for source_index, raw_source_path in enumerate(source_paths, start=1):
                if not isinstance(raw_source_path, str):
                    continue
                try:
                    resolved_source_path = Path(raw_source_path).expanduser().resolve()
                except OSError:
                    continue
                if resolved_source_path != resolved_input:
                    continue
                source_prefix = f"src_{source_index:04d}_"
                for source_dir in source_dirs:
                    if source_dir.name.startswith(source_prefix):
                        add_candidate(source_dir, "voice_bank_manifest")

    source_suffix = _safe_voice_bank_source_suffix(input_path)
    for source_dir in source_dirs:
        if source_dir.name.endswith(source_suffix):
            add_candidate(source_dir, "source_stem")
    return candidates


def _voice_bank_source_separation_paths(source_dir: Path) -> dict[str, Path] | None:
    audio_dir = source_dir / "work" / "audio"
    paths = {
        "source_vocals_48k": audio_dir / "source_vocals_48k.wav",
        "source_vocals_mono_16k": audio_dir / "source_vocals_mono_16k.wav",
        "background_only_48k": audio_dir / "background_only_48k.wav",
    }
    return paths if all(path.exists() for path in paths.values()) else None


def _voice_bank_source_separation_candidates(
    cache_project_dir: Path,
    input_path: Path,
    original_audio: Path,
    cfg: ProjectConfig,
) -> list[_SourceSeparationCacheCandidate]:
    candidates: list[_SourceSeparationCacheCandidate] = []
    for source_dir, matched_by in _voice_bank_source_dirs(cache_project_dir, input_path):
        source_audio = source_dir / "source_stereo_48k.wav"
        if not _same_audio_fingerprint(original_audio, source_audio):
            continue
        paths = _voice_bank_source_separation_paths(source_dir)
        if paths is None:
            continue
        try:
            _validate_audio_contract(paths["source_vocals_48k"], cfg.mix_sample_rate, 2, "cached source_vocals_48k")
            _validate_audio_contract(
                paths["source_vocals_mono_16k"],
                cfg.gemma_sample_rate,
                1,
                "cached source_vocals_mono_16k",
            )
            _validate_audio_contract(paths["background_only_48k"], cfg.mix_sample_rate, 2, "cached background_only_48k")
        except (OSError, RuntimeError, ValueError):
            continue
        candidates.append(_SourceSeparationCacheCandidate(source_dir, matched_by, paths))
    return candidates


def import_voice_bank_source_separation_cache_step(
    project_dir: Path,
    input_path: Path,
    cache_project_dir: Path,
) -> PipelineManifest:
    manifest = load_manifest(project_dir)
    _load_config_into_manifest(project_dir, manifest)
    original_audio = Path(
        manifest.artifacts.get("original_stereo_48k", project_dir / "work/audio/original_stereo_48k.wav")
    )
    if not original_audio.exists():
        return manifest
    cache_project_dir = cache_project_dir.expanduser().resolve()
    candidates = _voice_bank_source_separation_candidates(
        cache_project_dir,
        input_path.expanduser().resolve(),
        original_audio,
        manifest.project_config,
    )
    if not candidates:
        return manifest

    candidate = candidates[0]
    audio_dir = ensure_inside_project(project_dir, project_dir / "work" / "audio")
    separation_dir = ensure_inside_project(project_dir, project_dir / "work" / "source_separation")
    audio_dir.mkdir(parents=True, exist_ok=True)
    separation_dir.mkdir(parents=True, exist_ok=True)
    destination_paths = {
        "source_vocals_48k": audio_dir / "source_vocals_48k.wav",
        "source_vocals_mono_16k": audio_dir / "source_vocals_mono_16k.wav",
        "background_only_48k": audio_dir / "background_only_48k.wav",
    }
    for key, source_path in candidate.paths.items():
        destination_path = ensure_inside_project(project_dir, destination_paths[key])
        if source_path.resolve() != destination_path.resolve():
            shutil.copy2(source_path, destination_path)

    import_manifest_path = separation_dir / "source_separation_cache_import.json"
    separation_manifest_path = separation_dir / "source_separation_manifest.json"
    import_metadata = {
        "cache_project_dir": str(cache_project_dir),
        "source_dir": str(candidate.source_dir.resolve()),
        "input_path": str(input_path.expanduser().resolve()),
        "matched_by": candidate.matched_by,
        "source_paths": {key: str(path.resolve()) for key, path in candidate.paths.items()},
        "destination_paths": {key: str(path.resolve()) for key, path in destination_paths.items()},
    }
    write_json_atomic(import_manifest_path, import_metadata)
    write_json_atomic(
        separation_manifest_path,
        {
            "backend": "cached",
            "model": "voice_bank_cache",
            "input_audio_path": str(original_audio),
            "vocals_path": str(destination_paths["source_vocals_48k"]),
            "vocals_mono_path": str(destination_paths["source_vocals_mono_16k"]),
            "background_path": str(destination_paths["background_only_48k"]),
            "reused_existing": True,
            "command": [],
            "cache_import_manifest": str(import_manifest_path),
        },
    )
    manifest.artifacts["source_separation_cache_import"] = str(import_manifest_path)
    manifest.artifacts["source_separation_manifest"] = str(separation_manifest_path)
    save_manifest(project_dir, manifest)
    console.print(f"[cyan]source-separation[/cyan] imported cached voice-bank stems: {candidate.source_dir}")
    return manifest


def _validate_segment_audio_paths(project_dir: Path, segment: Segment, check_formats: bool = False) -> None:
    gemma_path = _resolve_project_read_path(project_dir, segment.audio_for_gemma, "audio_for_gemma")
    mix_path = _resolve_project_read_path(project_dir, segment.audio_for_mix, "audio_for_mix")
    if check_formats:
        _validate_audio_contract(gemma_path, 16_000, 1, "audio_for_gemma")
        _validate_audio_contract(mix_path, 48_000, 2, "audio_for_mix")
    segment.audio_for_gemma = str(gemma_path)
    segment.audio_for_mix = str(mix_path)


def _manifest_source_path(manifest: PipelineManifest) -> Path | None:
    return Path(manifest.source_info.path) if manifest.source_info else None


def _require_audio_stage_rights(
    manifest: PipelineManifest,
    stage: str,
    confirm_rights: bool = False,
    metadata: dict[str, object] | None = None,
) -> None:
    manifest.rights_audit = require_existing_or_confirmed_rights(
        manifest.rights_audit,
        confirm_rights,
        stage,
        _manifest_source_path(manifest),
        metadata=metadata,
    )


def init_project(project_dir: Path) -> None:
    create_project_structure(project_dir)


def inspect_input(input_path: Path) -> Any:
    return probe_with_fallback(input_path)


def _load_config_into_manifest(project_dir: Path, manifest: PipelineManifest) -> None:
    manifest.project_config = load_project_config(project_dir)


def extract_step(input_path: Path, project_dir: Path, confirm_rights: bool) -> PipelineManifest:
    _log_stage_start("extract", f"input={input_path}")
    create_project_structure(project_dir)
    manifest = load_manifest(project_dir)
    _load_config_into_manifest(project_dir, manifest)
    audit = require_confirmed_rights(confirm_rights, "extract", input_path)
    manifest.rights_audit = merge_rights_audit(manifest.rights_audit, audit)
    stereo, mono = extract_project_audio(input_path, project_dir)
    manifest.source_info = probe_with_fallback(input_path)
    manifest.artifacts["original_stereo_48k"] = str(stereo)
    manifest.artifacts["gemma_mono_16k"] = str(mono)
    mark_stage(manifest, "extract", "completed")
    save_manifest(project_dir, manifest)
    _log_stage_complete("extract", manifest, "audio prepared")
    return manifest


def source_separation_step(
    project_dir: Path,
    confirm_rights: bool = False,
    force: bool = False,
) -> PipelineManifest:
    manifest = load_manifest(project_dir)
    _load_config_into_manifest(project_dir, manifest)
    cfg = manifest.project_config
    backend = cfg.source_separation_backend
    _log_stage_start("source-separation", f"backend={backend}, model={cfg.source_separation_model}")
    _require_audio_stage_rights(
        manifest,
        "source-separation",
        confirm_rights,
        metadata={"backend": backend, "model": cfg.source_separation_model},
    )
    original_audio = Path(
        manifest.artifacts.get("original_stereo_48k", project_dir / "work/audio/original_stereo_48k.wav")
    )
    if backend == "none":
        mark_stage(manifest, "source-separation", "skipped", backend=backend, reason="disabled")
        save_manifest(project_dir, manifest)
        _log_stage_complete("source-separation", manifest, "skipped=disabled")
        return manifest
    try:
        result = separate_source_audio(
            original_audio,
            project_dir,
            backend=backend,
            model=cfg.source_separation_model,
            device=cfg.source_separation_device,
            sample_rate=cfg.mix_sample_rate,
            mono_sample_rate=cfg.gemma_sample_rate,
            force=force,
        )
    except SourceSeparationUnavailable as exc:
        if backend != "auto":
            raise
        warning = f"Source separation skipped because no separator backend is available: {exc}"
        if warning not in manifest.warnings:
            manifest.warnings.append(warning)
        mark_stage(manifest, "source-separation", "skipped", backend=backend, reason=str(exc))
        save_manifest(project_dir, manifest)
        _log_stage_complete("source-separation", manifest, "skipped=no backend")
        return manifest
    if result is None:
        mark_stage(manifest, "source-separation", "skipped", backend=backend, reason="disabled")
        save_manifest(project_dir, manifest)
        _log_stage_complete("source-separation", manifest, "skipped")
        return manifest

    _validate_audio_contract(result.vocals_path, cfg.mix_sample_rate, 2, "source_vocals_48k")
    _validate_audio_contract(result.vocals_mono_path, cfg.gemma_sample_rate, 1, "source_vocals_mono_16k")
    _validate_audio_contract(result.background_path, cfg.mix_sample_rate, 2, "background_only_48k")
    manifest.artifacts["source_vocals_48k"] = str(result.vocals_path)
    manifest.artifacts["source_vocals_mono_16k"] = str(result.vocals_mono_path)
    manifest.artifacts["background_only_48k"] = str(result.background_path)
    manifest.artifacts["source_separation_manifest"] = str(result.metadata_path)

    resliced_segments = 0
    if manifest.segments:
        started_at = monotonic()
        last_logged_at = started_at

        def log_reslice_progress(index: int, total: int, segment: Segment) -> None:
            nonlocal last_logged_at
            last_logged_at = _log_segment_progress(
                "source-separation clips",
                index,
                total,
                segment,
                manifest,
                started_at,
                last_logged_at,
            )

        write_segment_audio_clips(
            manifest.segments,
            result.vocals_mono_path,
            result.vocals_path,
            project_dir,
            progress_callback=log_reslice_progress,
        )
        resliced_segments = len(manifest.segments)
        out_path = project_dir / "work" / "segments" / "manifests" / "segments_source_separated.json"
        write_json_atomic(out_path, {"segments": [s.model_dump(mode="json") for s in manifest.segments]})
        manifest.artifacts["segments_source_separated"] = str(out_path)

    mark_stage(
        manifest,
        "source-separation",
        "completed",
        backend=result.backend,
        model=result.model,
        reused_existing=result.reused_existing,
        vocals_path=str(result.vocals_path),
        vocals_mono_path=str(result.vocals_mono_path),
        background_path=str(result.background_path),
        resliced_segments=resliced_segments,
    )
    save_manifest(project_dir, manifest)
    _log_stage_complete(
        "source-separation",
        manifest,
        f"backend={result.backend}, reused={result.reused_existing}",
    )
    return manifest


def _write_segments_manifest(path: Path, segments: list[Segment]) -> None:
    write_json_atomic(path, {"segments": [s.model_dump(mode="json") for s in segments]})


def _seed_segments_for_transcribe(
    project_dir: Path,
    manifest: PipelineManifest,
    audio_path: Path,
    mix_audio_path: Path,
) -> bool:
    if manifest.segments:
        return False
    audio_duration = round(max(duration_sec(audio_path), 0.05), 3)
    seed = Segment(
        id="seg_0001",
        start=0.0,
        end=audio_duration,
        duration=audio_duration,
        audio_for_gemma=str(project_dir / "work" / "segments" / "audio" / "seg_0001_gemma.wav"),
        audio_for_mix=str(project_dir / "work" / "segments" / "audio" / "seg_0001_mix.wav"),
        keep_original_texture=True,
        status="raw",
    )
    manifest.segments = write_segment_audio_clips([seed], audio_path, mix_audio_path, project_dir)
    seed_path = project_dir / "work" / "segments" / "manifests" / "segments_transcribe_seed.json"
    _write_segments_manifest(seed_path, manifest.segments)
    manifest.artifacts["segments_transcribe_seed"] = str(seed_path)
    mark_stage(
        manifest,
        "transcribe-seed",
        "completed",
        segment_count=len(manifest.segments),
        source_audio=str(audio_path),
        mix_audio=str(mix_audio_path),
    )
    save_manifest(project_dir, manifest)
    return True


def segment_step(project_dir: Path, confirm_rights: bool = False) -> PipelineManifest:
    _log_stage_start("segment", f"project={project_dir}")
    manifest = load_manifest(project_dir)
    _load_config_into_manifest(project_dir, manifest)
    _require_audio_stage_rights(manifest, "segment", confirm_rights)
    cfg = manifest.project_config
    if manifest.stage_state.get("transcribe", {}).get("status") == "completed" and manifest.segments:
        raw_path = project_dir / "work" / "segments" / "manifests" / "segments_raw.json"
        final_path = project_dir / "work" / "segments" / "manifests" / "segments_final.json"
        _write_segments_manifest(raw_path, manifest.segments)
        _write_segments_manifest(final_path, manifest.segments)
        manifest.artifacts["segments_raw"] = str(raw_path)
        manifest.artifacts["segments_final"] = str(final_path)
        mark_stage(
            manifest,
            "segment",
            "completed",
            source="transcribe",
            segment_count=len(manifest.segments),
            segment_counts=_segment_counts(manifest),
        )
        save_manifest(project_dir, manifest)
        _log_stage_complete("segment", manifest, f"finalized={len(manifest.segments)}")
        return manifest
    started_at = monotonic()
    last_logged_at = started_at

    def log_segment_progress(index: int, total: int, segment: Segment) -> None:
        nonlocal last_logged_at
        last_logged_at = _log_segment_progress(
            "segment",
            index,
            total,
            segment,
            None,
            started_at,
            last_logged_at,
            note="writing segment audio",
        )

    manual = project_dir / "work" / "segments" / "manifests" / "segments_manual.json"
    if manual.exists():
        segments = load_manual_segments(manual)
        total = len(segments)
        for index, segment in enumerate(segments, start=1):
            _validate_segment_audio_paths(project_dir, segment, check_formats=True)
            log_segment_progress(index, total, segment)
    else:
        gemma_audio = Path(
            manifest.artifacts.get(
                "source_vocals_mono_16k",
                manifest.artifacts.get("gemma_mono_16k", project_dir / "work/audio/gemma_mono_16k.wav"),
            )
        )
        mix_audio = Path(
            manifest.artifacts.get(
                "source_vocals_48k",
                manifest.artifacts.get("original_stereo_48k", project_dir / "work/audio/original_stereo_48k.wav"),
            )
        )
        segments = energy_segments(
            gemma_audio,
            mix_audio,
            project_dir,
            min_segment_sec=cfg.segmentation_min_segment_sec,
            max_segment_sec=cfg.segmentation_max_segment_sec,
            silence_db=cfg.segmentation_silence_db,
            min_silence_sec=cfg.segmentation_min_silence_sec,
            progress_callback=log_segment_progress,
        )
    manifest.segments = segments
    raw_path = project_dir / "work" / "segments" / "manifests" / "segments_raw.json"
    _write_segments_manifest(raw_path, segments)
    manifest.artifacts["segments_raw"] = str(raw_path)
    mark_stage(manifest, "segment", "completed", segment_count=len(segments), segment_counts=_segment_counts(manifest))
    save_manifest(project_dir, manifest)
    _log_stage_complete("segment", manifest, f"created={len(segments)}")
    return manifest


def _gemma_context(manifest: PipelineManifest) -> dict[str, Any]:
    return {
        "schema_version": manifest.schema_version,
        "source_info": manifest.source_info.model_dump(mode="json") if manifest.source_info else None,
    }


def _gemma_backend_config(cfg: Any, model_id: str | None = None) -> dict[str, Any]:
    return {
        "model_id": model_id or cfg.gemma_model_id,
        "url": cfg.gemma_http_url,
        "send_audio": cfg.gemma_http_send_audio,
        "local_files_only": cfg.hf_local_files_only,
        "llama_cpp_cli_path": cfg.gemma_llama_cpp_cli_path,
        "llama_cpp_model_path": cfg.gemma_llama_cpp_model_path,
        "llama_cpp_mmproj_path": cfg.gemma_llama_cpp_mmproj_path,
        "llama_cpp_timeout_sec": cfg.gemma_llama_cpp_timeout_sec,
        "llama_cpp_ctx_size": cfg.gemma_llama_cpp_ctx_size,
        "llama_cpp_n_predict": cfg.gemma_llama_cpp_n_predict,
        "llama_cpp_gpu_layers": cfg.gemma_llama_cpp_gpu_layers,
        "llama_cpp_temperature": cfg.gemma_llama_cpp_temperature,
        "llama_cpp_seed": cfg.gemma_llama_cpp_seed,
        "llama_cpp_extra_args": cfg.gemma_llama_cpp_extra_args,
    }


def _asr_backend_config(cfg: Any) -> dict[str, Any]:
    return {
        "model_id": cfg.asr_model_id,
        "language": cfg.asr_language,
        "local_files_only": cfg.asr_local_files_only,
        "beam_size": cfg.asr_beam_size,
        "best_of": cfg.asr_best_of,
        "condition_on_previous_text": cfg.asr_condition_on_previous_text,
        "vad_filter": cfg.asr_vad_filter,
        "vad_parameters": cfg.asr_vad_parameters,
        "word_timestamps": cfg.asr_word_timestamps,
        "hallucination_silence_threshold": cfg.asr_hallucination_silence_threshold,
        "qwen_model_id": cfg.qwen_asr_model_id,
        "qwen_forced_aligner_model_id": cfg.qwen_asr_forced_aligner_model_id,
        "qwen_device_map": cfg.qwen_asr_device_map,
        "qwen_dtype": cfg.qwen_asr_dtype,
        "qwen_return_timestamps": cfg.qwen_asr_return_timestamps,
        "qwen_context": cfg.qwen_asr_context,
        "qwen_max_inference_batch_size": cfg.qwen_asr_max_inference_batch_size,
        "qwen_max_new_tokens": cfg.qwen_asr_max_new_tokens,
    }


def _format_server_command(command: list[str], base_url: str, lane_index: int) -> list[str]:
    parsed = urlparse(base_url)
    host = parsed.hostname or ""
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return [
        str(part)
        .replace("{base_url}", base_url)
        .replace("{host}", host)
        .replace("{port}", str(port))
        .replace("{lane}", str(lane_index))
        for part in command
    ]


def _gemma_text_server_command(
    cfg: Any,
    *,
    base_url: str | None = None,
    lane_index: int = 0,
) -> list[str]:
    effective_base_url = base_url or cfg.gemma_text_server_url
    if cfg.gemma_text_server_command:
        return _format_server_command(
            [str(part) for part in cfg.gemma_text_server_command],
            effective_base_url,
            lane_index,
        )
    return default_llama_server_command(
        base_url=effective_base_url,
        model_path=cfg.gemma_llama_cpp_model_path,
        ctx_size=cfg.gemma_llama_cpp_ctx_size,
        gpu_layers=cfg.gemma_llama_cpp_gpu_layers,
        n_predict=cfg.gemma_text_n_predict,
    )


def transcribe_step(
    project_dir: Path,
    asr_backend: str | None = None,
    confirm_rights: bool = False,
) -> PipelineManifest:
    manifest = load_manifest(project_dir)
    _load_config_into_manifest(project_dir, manifest)
    cfg = manifest.project_config
    backend_kind = asr_backend or cfg.asr_backend
    total = len(manifest.segments)
    _log_stage_start("transcribe", f"backend={backend_kind}, segments={total}")
    _require_audio_stage_rights(manifest, "transcribe", confirm_rights, metadata={"backend": backend_kind})
    if backend_kind != "mock" and "source_vocals_mono_16k" not in manifest.artifacts:
        manifest = source_separation_step(project_dir, confirm_rights=confirm_rights)
        _load_config_into_manifest(project_dir, manifest)
        cfg = manifest.project_config
        total = len(manifest.segments)
    if backend_kind != "mock" and "source_vocals_mono_16k" not in manifest.artifacts:
        raise ValueError(
            "Real ASR transcription requires source-separated vocals. Run source-separation "
            "with backend auto/demucs/mock before transcribe, or set source_separation_backend "
            "to an available separator."
        )
    audio_path = Path(
        manifest.artifacts["source_vocals_mono_16k"]
        if backend_kind != "mock"
        else manifest.artifacts.get(
            "source_vocals_mono_16k",
            manifest.artifacts.get("gemma_mono_16k", project_dir / "work/audio/gemma_mono_16k.wav"),
        )
    )
    _validate_audio_contract(audio_path, cfg.gemma_sample_rate, 1, audio_path.stem)
    mix_audio_path = Path(
        manifest.artifacts.get(
            "source_vocals_48k",
            manifest.artifacts.get("original_stereo_48k", project_dir / "work/audio/original_stereo_48k.wav"),
        )
    )
    seeded_for_transcribe = _seed_segments_for_transcribe(project_dir, manifest, audio_path, mix_audio_path)
    total = len(manifest.segments)
    backend = create_asr_backend(backend_kind, _asr_backend_config(cfg))
    audio_duration = duration_sec(audio_path)
    chunks = backend.transcribe(audio_path, manifest.segments)
    raw_asr_chunk_count = len(chunks)
    asr_text_replacement_count = 0
    repair_summary: dict[str, Any] = {
        "enabled": False,
        "attempted": 0,
        "repaired": 0,
        "skipped": 0,
        "items": [],
    }
    if backend_kind != "mock":
        repair_audio_path = Path(manifest.artifacts.get("gemma_mono_16k", audio_path))
        chunks, repair_summary = _repair_asr_chunks(
            chunks,
            backend=backend,
            project_dir=project_dir,
            repair_audio_path=repair_audio_path,
            audio_duration_sec=audio_duration,
            cfg=cfg,
        )
        repair_summary_path = project_dir / "work" / "transcribe" / "asr_repair_summary.json"
        write_json_atomic(repair_summary_path, repair_summary)
        manifest.artifacts["asr_repair_summary"] = str(repair_summary_path)
        chunks, asr_text_replacement_count = _apply_asr_text_replacements_to_chunks(
            chunks,
            cfg.asr_text_replacements,
        )
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
    for index, segment in enumerate(manifest.segments, start=1):
        source_script = mapped.get(segment.id)
        segment.source_script = source_script
        if source_script and source_script.text:
            with_text += 1
            status = "transcribed"
        else:
            status = "needs_manual_review"
        rows.append(
            {
                "segment_id": segment.id,
                "status": status,
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
    mark_stage(
        manifest,
        "transcribe",
        "completed",
        backend=backend_kind,
        segment_count=total,
        previous_segment_count=previous_segment_count,
        raw_asr_chunk_count=raw_asr_chunk_count,
        asr_chunk_count=len(chunks),
        asr_repair_attempted=repair_summary.get("attempted", 0),
        asr_repair_repaired=repair_summary.get("repaired", 0),
        asr_text_replacements=asr_text_replacement_count,
        seeded_for_transcribe=seeded_for_transcribe,
        resegmented_from_chunks=resegmented_from_chunks,
        transcribed=with_text,
        needs_manual_review=total - with_text,
    )
    save_manifest(project_dir, manifest)
    _log_stage_complete("transcribe", manifest, f"backend={backend_kind}")
    return manifest


def analyze_step(
    project_dir: Path,
    backend_kind: str,
    model_id: str | None = None,
    confirm_rights: bool = False,
) -> PipelineManifest:
    manifest = load_manifest(project_dir)
    _load_config_into_manifest(project_dir, manifest)
    total = len(manifest.segments)
    _log_stage_start("analyze", f"backend={backend_kind}, segments={total}")
    _require_audio_stage_rights(manifest, "analyze", confirm_rights, metadata={"backend": backend_kind})
    cfg = manifest.project_config
    backend = create_gemma_backend(backend_kind, _gemma_backend_config(cfg, model_id))
    context = _gemma_context(manifest)
    started_at = monotonic()
    last_logged_at = started_at
    for index, segment in enumerate(manifest.segments, start=1):
        if segment.status in SKIP_STATUSES:
            last_logged_at = _log_segment_progress(
                "analyze", index, total, segment, manifest, started_at, last_logged_at
            )
            continue
        _validate_segment_audio_paths(project_dir, segment, check_formats=True)
        try:
            segment.analysis = validate_gemma_task_response(
                "analyze",
                backend.analyze_segment(Path(segment.audio_for_gemma), segment, context),
            )
            segment.status = "analyzed"
        except Exception as exc:
            segment.errors.append(str(exc))
            segment.status = "needs_manual_review"
        last_logged_at = _log_segment_progress(
            "analyze", index, total, segment, manifest, started_at, last_logged_at
        )
    out_path = project_dir / "work" / "segments" / "manifests" / "segments_gemma.json"
    write_json_atomic(out_path, {"segments": [s.model_dump(mode="json") for s in manifest.segments]})
    manifest.artifacts["segments_gemma"] = str(out_path)
    mark_stage(manifest, "analyze", "completed", backend=backend_kind, segment_counts=_segment_counts(manifest))
    save_manifest(project_dir, manifest)
    _log_stage_complete("analyze", manifest, f"backend={backend_kind}")
    return manifest


def script_step(project_dir: Path, backend_kind: str, confirm_rights: bool = False) -> PipelineManifest:
    manifest = load_manifest(project_dir)
    _load_config_into_manifest(project_dir, manifest)
    total = len(manifest.segments)
    _log_stage_start("script", f"backend={backend_kind}, segments={total}")
    _require_audio_stage_rights(manifest, "script", confirm_rights, metadata={"backend": backend_kind})
    cfg = manifest.project_config
    backend = create_gemma_backend(backend_kind, _gemma_backend_config(cfg))
    context = _gemma_context(manifest)
    started_at = monotonic()
    last_logged_at = started_at
    for index, segment in enumerate(manifest.segments, start=1):
        if segment.status in SKIP_STATUSES:
            last_logged_at = _log_segment_progress(
                "script", index, total, segment, manifest, started_at, last_logged_at
            )
            continue
        _validate_segment_audio_paths(project_dir, segment, check_formats=True)
        try:
            payload = validate_gemma_task_response(
                "script",
                backend.generate_script(Path(segment.audio_for_gemma), segment, context),
            )
            segment.script = normalize_script_payload(payload)
            segment.status = "scripted"
        except Exception as exc:
            segment.errors.append(str(exc))
            segment.status = "needs_manual_review"
        last_logged_at = _log_segment_progress(
            "script", index, total, segment, manifest, started_at, last_logged_at
        )
    out_path = project_dir / "work" / "segments" / "manifests" / "segments_script.json"
    write_json_atomic(out_path, {"segments": [s.model_dump(mode="json") for s in manifest.segments]})
    manifest.artifacts["segments_script"] = str(out_path)
    mark_stage(manifest, "script", "completed", backend=backend_kind, segment_counts=_segment_counts(manifest))
    save_manifest(project_dir, manifest)
    _log_stage_complete("script", manifest, f"backend={backend_kind}")
    return manifest


def translate_ko_step(
    project_dir: Path,
    gemma_text_backend: str | None = None,
    confirm_rights: bool = False,
    force_retranslate: bool = False,
) -> PipelineManifest:
    manifest = load_manifest(project_dir)
    _load_config_into_manifest(project_dir, manifest)
    cfg = manifest.project_config
    backend_kind = (gemma_text_backend or "llama_server").replace("-", "_")
    total = len(manifest.segments)
    _log_stage_start("translate-ko", f"backend={backend_kind}, segments={total}")
    _require_audio_stage_rights(
        manifest, "translate-ko", confirm_rights, metadata={"backend": backend_kind}
    )
    if manifest.stage_state.get("transcribe", {}).get("status") != "completed":
        raise ValueError("translate-ko requires a completed transcribe stage.")
    if backend_kind not in {"llama_server", "mock"}:
        raise ValueError(f"Unsupported Gemma text backend: {gemma_text_backend}")

    jsonl_path = project_dir / "work" / "translate_ko" / "translation_bundles.jsonl"
    summary_path = project_dir / "work" / "translate_ko" / "summary.json"
    rows: list[dict[str, Any]] = []
    translated = 0
    needs_manual_review = 0
    colloquialized = 0
    numeric_counting_postprocessed = 0
    model_name = cfg.gemma_llama_cpp_model_path if backend_kind == "llama_server" else "mock"
    translatable: list[Segment] = []
    for segment in manifest.segments:
        translation = segment.translation_ko
        legacy_numeric_translation = bool(
            translation
            and LEGACY_DETERMINISTIC_NUMERIC_TRANSLATION_NOTE in translation.notes
        )
        if (
            translation
            and translation.ko_natural.strip()
            and not force_retranslate
            and not legacy_numeric_translation
        ):
            translated += 1
            rows.append(
                {
                    "batch_id": translation.batch_id,
                    "segment_id": segment.id,
                    "status": "translated",
                    "source_text": segment.source_script.text if segment.source_script else "",
                    "translation_ko": translation.model_dump(mode="json"),
                    "resumed": True,
                }
            )
        elif segment.source_script and segment.source_script.text.strip():
            if force_retranslate or legacy_numeric_translation:
                segment.translation_ko = None
            translatable.append(segment)
        else:
            needs_manual_review += 1
            rows.append(
                {
                    "segment_id": segment.id,
                    "status": "needs_manual_review",
                    "reason": "missing source_script text",
                    "source_text": segment.source_script.text if segment.source_script else "",
                    "translation_ko": None,
                }
            )

    translation_batches = (
        _translation_span_batches(
            translatable,
            max_segments=cfg.gemma_text_span_size,
            max_duration_sec=cfg.gemma_text_span_max_sec,
            max_gap_sec=cfg.gemma_text_span_max_gap_sec,
        )
        if backend_kind == "llama_server"
        else _chunked(translatable, cfg.gemma_text_batch_size)
    )
    translation_worker_count = (
        _effective_lane_count(cfg.gemma_text_concurrency, len(translation_batches))
        if backend_kind == "llama_server" and translatable
        else 1
    )
    translation_base_urls = [cfg.gemma_text_server_url.rstrip("/")]

    def create_translation_client(_worker_index: int = 0) -> Any:
        if backend_kind == "llama_server":
            return LlamaServerTranslationClient(
                translation_base_urls[0],
                timeout_sec=cfg.gemma_text_timeout_sec,
                retries=cfg.gemma_text_retries,
                n_predict=cfg.gemma_text_n_predict,
                model=model_name,
                two_pass=cfg.gemma_text_two_pass,
            )
        return MockTranslationClient(model=model_name)

    server_managers: list[ManagedGemmaTextServer] = []
    if backend_kind == "llama_server" and translatable:
        for lane_index, base_url in enumerate(translation_base_urls):
            command = (
                _gemma_text_server_command(cfg, base_url=base_url, lane_index=lane_index)
                if cfg.gemma_text_server_auto_start
                else []
            )
            log_name = "llama_server.log"
            server_managers.append(
                ManagedGemmaTextServer(
                    enabled=cfg.gemma_text_server_auto_start,
                    base_url=base_url,
                    command=command,
                    log_path=project_dir / "work" / "translate_ko" / log_name,
                    startup_timeout_sec=cfg.gemma_text_server_startup_timeout_sec,
                    shutdown_timeout_sec=cfg.gemma_text_server_shutdown_timeout_sec,
                )
            )

    started_at = monotonic()
    last_logged_at = started_at
    processed = translated + needs_manual_review

    def persist_partial() -> None:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        _write_jsonl_atomic(jsonl_path, rows)
        write_json_atomic(
            summary_path,
            {
                "backend": backend_kind,
                "model": model_name,
                "segments": total,
                "translated": translated,
                "needs_manual_review": needs_manual_review,
                "colloquialized": colloquialized,
                "numeric_counting_postprocessed": numeric_counting_postprocessed,
                "concurrency": translation_worker_count,
                "context_radius": cfg.gemma_text_context_radius,
                "span_size": cfg.gemma_text_span_size if backend_kind == "llama_server" else cfg.gemma_text_batch_size,
                "span_max_sec": cfg.gemma_text_span_max_sec if backend_kind == "llama_server" else None,
                "span_max_gap_sec": cfg.gemma_text_span_max_gap_sec if backend_kind == "llama_server" else None,
                "span_count": len(translation_batches),
                "two_pass": cfg.gemma_text_two_pass if backend_kind == "llama_server" else False,
                "force_retranslate": force_retranslate,
                "base_urls": translation_base_urls if backend_kind == "llama_server" else [],
                "partial": True,
            },
        )
        manifest.artifacts["translation_bundles"] = str(jsonl_path)
        manifest.artifacts["translation_summary"] = str(summary_path)
        save_manifest(project_dir, manifest)

    if rows:
        persist_partial()

    def translate_batch_with_retries(
        batch: list[Segment],
        batch_id: str,
        worker_index: int,
    ) -> tuple[list[Segment], str, dict[str, Any], dict[str, list[str]]]:
        client = create_translation_client(worker_index)
        translation_failures: dict[str, list[str]] = {}
        translations: dict[str, Any] = {}

        def record_failure(segment: Segment, message: str) -> None:
            translation_failures.setdefault(segment.id, []).append(message)

        def retry_single(missing_segment: Segment, retry_batch_id: str) -> None:
            context_segments = _translation_context_segments(
                manifest.segments,
                [missing_segment],
                cfg.gemma_text_context_radius,
            )
            try:
                translations.update(
                    _translate_with_optional_context(
                        client,
                        [missing_segment],
                        retry_batch_id,
                        context_segments,
                    )
                )
            except Exception as exc:
                message = f"Korean translation retry failed for {retry_batch_id}: {exc}"
                record_failure(missing_segment, message)
                return
            if missing_segment.id not in translations:
                record_failure(
                    missing_segment,
                    f"Korean translation retry failed for {retry_batch_id}: missing model response",
                )

        def translate_group(group: list[Segment], group_batch_id: str) -> None:
            if not group:
                return
            context_segments = _translation_context_segments(
                manifest.segments,
                group,
                cfg.gemma_text_context_radius,
            )
            try:
                group_translations = _translate_with_optional_context(
                    client,
                    group,
                    group_batch_id,
                    context_segments,
                )
            except Exception as exc:
                if backend_kind == "llama_server" and _is_gemma_text_server_unavailable(exc):
                    raise
                message = f"Korean translation batch failed for {group_batch_id}: {exc}"
                if len(group) == 1:
                    record_failure(group[0], message)
                    return
                for segment in group:
                    record_failure(segment, message)
                midpoint = max(1, len(group) // 2)
                translate_group(group[:midpoint], f"{group_batch_id}_split_01")
                translate_group(group[midpoint:], f"{group_batch_id}_split_02")
                return
            translations.update(group_translations)
            missing_after_group = [segment for segment in group if segment.id not in translations]
            if not missing_after_group:
                return
            if len(group) == 1:
                retry_single(group[0], f"{group_batch_id}_single_01")
                return
            if len(missing_after_group) == 1:
                retry_single(missing_after_group[0], f"{group_batch_id}_single_01")
                return
            midpoint = max(1, len(missing_after_group) // 2)
            translate_group(missing_after_group[:midpoint], f"{group_batch_id}_missing_01")
            translate_group(missing_after_group[midpoint:], f"{group_batch_id}_missing_02")

        translate_group(batch, batch_id)
        return batch, batch_id, translations, translation_failures

    def apply_batch_result(
        batch: list[Segment],
        batch_id: str,
        translations: dict[str, Any],
        translation_failures: dict[str, list[str]],
    ) -> None:
        nonlocal last_logged_at, needs_manual_review, processed, translated
        for segment in batch:
            for message in translation_failures.get(segment.id, []):
                if message not in segment.errors:
                    segment.errors.append(message)
            translation = translations.get(segment.id)
            if translation is None:
                needs_manual_review += 1
                processed += 1
                source_text = segment.source_script.text if segment.source_script else ""
                rows.append(
                    {
                        "batch_id": batch_id,
                        "segment_id": segment.id,
                        "status": "needs_manual_review",
                        "reason": "missing translation in model response",
                        "source_text": source_text,
                        "translation_ko": None,
                        "error": "; ".join(translation_failures.get(segment.id, [])) or None,
                    }
                )
                last_logged_at = _log_translate_progress(
                    processed,
                    total,
                    segment,
                    "needs_manual_review",
                    source_text,
                    None,
                    started_at,
                    last_logged_at,
                )
                continue
            segment.translation_ko = translation
            _clear_korean_translation_errors(segment)
            translated += 1
            processed += 1
            source_text = segment.source_script.text if segment.source_script else ""
            rows.append(
                {
                    "batch_id": batch_id,
                    "segment_id": segment.id,
                    "status": "translated",
                    "source_text": source_text,
                    "translation_ko": translation.model_dump(mode="json"),
                }
            )
            last_logged_at = _log_translate_progress(
                processed,
                total,
                segment,
                "translated",
                source_text,
                translation.ko_natural,
                started_at,
                last_logged_at,
            )

    try:
        for server_manager in server_managers:
            server_manager.start()
        batch_jobs = []
        for job_index, batch in enumerate(translation_batches):
            batch_id = f"batch_{job_index + 1:04d}"
            worker_index = job_index % translation_worker_count
            batch_jobs.append((job_index, batch, batch_id, worker_index))
        if translation_worker_count > 1 and len(batch_jobs) > 1:
            pending: dict[int, tuple[list[Segment], str, dict[str, Any], dict[str, list[str]]]] = {}
            next_to_apply = 0
            with ThreadPoolExecutor(max_workers=translation_worker_count) as executor:
                futures = {
                    executor.submit(
                        translate_batch_with_retries,
                        batch,
                        batch_id,
                        worker_index,
                    ): job_index
                    for job_index, batch, batch_id, worker_index in batch_jobs
                }
                for future in as_completed(futures):
                    job_index = futures[future]
                    pending[job_index] = future.result()
                    while next_to_apply in pending:
                        apply_batch_result(*pending.pop(next_to_apply))
                        persist_partial()
                        next_to_apply += 1
        else:
            for _, batch, batch_id, worker_index in batch_jobs:
                apply_batch_result(*translate_batch_with_retries(batch, batch_id, worker_index))
                persist_partial()
    finally:
        for server_manager in reversed(server_managers):
            server_manager.stop()

    colloquialized = _apply_korean_colloquial_postprocess(manifest.segments)
    numeric_counting_postprocessed = _apply_korean_numeric_counting_postprocess(manifest.segments)
    _refresh_translation_rows(rows, manifest.segments)
    _write_jsonl_atomic(jsonl_path, rows)
    summary = {
        "backend": backend_kind,
        "model": model_name,
        "segments": total,
        "translated": translated,
        "needs_manual_review": needs_manual_review,
        "colloquialized": colloquialized,
        "numeric_counting_postprocessed": numeric_counting_postprocessed,
        "concurrency": translation_worker_count,
        "context_radius": cfg.gemma_text_context_radius,
        "span_size": cfg.gemma_text_span_size if backend_kind == "llama_server" else cfg.gemma_text_batch_size,
        "span_max_sec": cfg.gemma_text_span_max_sec if backend_kind == "llama_server" else None,
        "span_max_gap_sec": cfg.gemma_text_span_max_gap_sec if backend_kind == "llama_server" else None,
        "span_count": len(translation_batches),
        "two_pass": cfg.gemma_text_two_pass if backend_kind == "llama_server" else False,
        "force_retranslate": force_retranslate,
        "base_urls": translation_base_urls if backend_kind == "llama_server" else [],
    }
    write_json_atomic(summary_path, summary)
    manifest.artifacts["translation_bundles"] = str(jsonl_path)
    manifest.artifacts["translation_summary"] = str(summary_path)
    server_metadata = None
    if backend_kind == "llama_server":
        instances = [
            {
                "base_url": getattr(manager, "base_url", translation_base_urls[index]),
                "started": manager.started,
                "reused_existing": manager.reused_existing,
                "log_path": str(manager.log_path) if manager.log_path else None,
            }
            for index, manager in enumerate(server_managers)
        ]
        server_metadata = {
            "auto_start": cfg.gemma_text_server_auto_start,
            "concurrency": translation_worker_count,
            "server_count": len(server_managers),
            "mode": "single_server_slots",
            "base_urls": translation_base_urls,
            "instances": instances,
        }
        if len(instances) == 1:
            server_metadata.update(
                started=instances[0]["started"],
                reused_existing=instances[0]["reused_existing"],
                log_path=instances[0]["log_path"],
            )
    mark_stage(
        manifest,
        "translate-ko",
        "completed",
        backend=backend_kind,
        model=model_name,
        translated=translated,
        needs_manual_review=needs_manual_review,
        colloquialized=colloquialized,
        numeric_counting_postprocessed=numeric_counting_postprocessed,
        concurrency=translation_worker_count,
        context_radius=cfg.gemma_text_context_radius,
        span_size=cfg.gemma_text_span_size if backend_kind == "llama_server" else cfg.gemma_text_batch_size,
        span_max_sec=cfg.gemma_text_span_max_sec if backend_kind == "llama_server" else None,
        span_max_gap_sec=cfg.gemma_text_span_max_gap_sec if backend_kind == "llama_server" else None,
        span_count=len(translation_batches),
        two_pass=cfg.gemma_text_two_pass if backend_kind == "llama_server" else False,
        force_retranslate=force_retranslate,
        server=server_metadata,
    )
    save_manifest(project_dir, manifest)
    _log_stage_complete("translate-ko", manifest, f"backend={backend_kind}")
    return manifest


def korean_script_step(project_dir: Path, confirm_rights: bool = False) -> PipelineManifest:
    manifest = load_manifest(project_dir)
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
    return manifest


def _select_voice_ref_segment(manifest: PipelineManifest) -> Segment | None:
    selected = _select_voice_ref_spans(Path("."), manifest, manifest.project_config, max_refs=1)
    return selected[0].segments[0] if selected else None


def _select_voice_ref_segments(
    project_dir: Path,
    manifest: PipelineManifest,
    cfg: Any,
    max_refs: int = 5,
) -> list[Segment]:
    return [
        span.segments[0]
        for span in _select_voice_ref_spans(project_dir, manifest, cfg, max_refs=max_refs)
    ]


def _select_voice_ref_spans(
    project_dir: Path,
    manifest: PipelineManifest,
    cfg: Any,
    max_refs: int = 5,
) -> list[_VoiceRefSpan]:
    source_language = _canonical_language(getattr(cfg, "source_language", "ja"))
    ref_min_sec = float(getattr(cfg, "gsv_ref_min_sec", 3.0))
    ref_max_sec = float(getattr(cfg, "gsv_ref_max_sec", 10.0))
    candidates = [
        segment
        for segment in sorted(manifest.segments, key=lambda item: (item.start, item.end, item.id))
        if _segment_can_seed_voice_ref(segment, source_language)
    ]
    scored: list[tuple[tuple[float, ...], _VoiceRefSpan, AudioQualityMetrics | None]] = []
    for index, segment in enumerate(candidates):
        span = _build_voice_ref_span(candidates, index, ref_min_sec, ref_max_sec)
        if span is None:
            continue
        metrics = None
        try:
            source_audio = _resolve_project_read_path(project_dir, segment.audio_for_mix, "audio_for_mix")
            metrics = measure_source_voice_quality(source_audio)
        except Exception:
            pass
        scored.append((_voice_ref_span_score(span, metrics), span, metrics))
    selected = [
        (score, span, metrics)
        for score, span, metrics in sorted(scored, key=lambda item: item[0], reverse=True)
        if metrics is None or metrics.score >= getattr(cfg, "gsv_ref_min_quality_score", 0.25)
    ]
    if not selected:
        selected = sorted(scored, key=lambda item: item[0], reverse=True)
    spans: list[_VoiceRefSpan] = []
    used_segment_ids: set[str] = set()
    for _, span, _ in selected:
        if any(segment.id in used_segment_ids for segment in span.segments):
            continue
        spans.append(span)
        used_segment_ids.update(segment.id for segment in span.segments)
        if len(spans) >= max_refs:
            break
    return spans


def _segment_can_seed_voice_ref(segment: Segment, source_language: str) -> bool:
    return bool(
        segment.source_script
        and segment.source_script.text.strip()
        and segment.audio_for_mix
        and segment.duration > 0
        and _canonical_language(segment.source_script.language) == source_language
    )


def _build_voice_ref_span(
    candidates: list[Segment],
    start_index: int,
    min_sec: float,
    max_sec: float,
) -> _VoiceRefSpan | None:
    segments: list[Segment] = []
    duration = 0.0
    for segment in candidates[start_index:]:
        next_duration = duration + segment.duration
        if next_duration > max_sec:
            return None if duration < min_sec else _VoiceRefSpan(tuple(segments), duration)
        segments.append(segment)
        duration = next_duration
        if duration >= min_sec:
            return _VoiceRefSpan(tuple(segments), duration)
    return None


def _voice_ref_span_score(
    span: _VoiceRefSpan,
    metrics: AudioQualityMetrics | None = None,
) -> tuple[float, float, float, float]:
    first = span.segments[0]
    text_len = sum(len(segment.source_script.text.strip()) for segment in span.segments if segment.source_script)
    quality = metrics.score if metrics is not None else 0.5
    return (
        quality,
        span.duration,
        min(text_len / 80.0, 1.0),
        -first.start,
    )


def _voice_ref_span_prompt_text(span: _VoiceRefSpan) -> str:
    return " ".join(
        segment.source_script.text.strip()
        for segment in span.segments
        if segment.source_script and segment.source_script.text.strip()
    )


def _write_voice_ref_span(project_dir: Path, span: _VoiceRefSpan, output_path: Path) -> None:
    clips: list[np.ndarray] = []
    sample_rate: int | None = None
    channels: int | None = None
    for segment in span.segments:
        source = _resolve_project_read_path(project_dir, segment.audio_for_mix, "audio_for_mix")
        data, clip_sample_rate = load_audio(source)
        if sample_rate is None:
            sample_rate = clip_sample_rate
            channels = data.shape[1]
        elif clip_sample_rate != sample_rate or data.shape[1] != channels:
            raise ValueError("Cannot concatenate voice refs with different audio formats.")
        clips.append(data)
    if not clips or sample_rate is None:
        raise ValueError("Cannot write an empty voice reference span.")
    write_audio(output_path, np.concatenate(clips, axis=0), sample_rate)


def prepare_source_voice_refs_step(
    project_dir: Path,
    refs_path: Path | None = None,
    confirm_rights: bool = False,
) -> PipelineManifest:
    manifest = load_manifest(project_dir)
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
    prompt_text = _voice_ref_span_prompt_text(selected_span)
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
            "prompt_lang": selected.source_script.language or cfg.source_language,
            "aux_ref_audio_paths": aux_ref_audio_paths,
            "source_language": cfg.source_language,
            "target_language": cfg.target_language,
            "cross_lingual_role": "ja_source_prompt_for_ko_tts",
        }
        prepared[style] = str(resolved_ref_path)
        ref_qc_rows.append(
            {
                "style": style,
                "segment_id": selected.id,
                "source_language": cfg.source_language,
                "target_language": cfg.target_language,
                "prompt_lang": selected.source_script.language or cfg.source_language,
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
    return manifest


def gsv_few_shot_step(
    project_dir: Path,
    confirm_rights: bool = False,
    force: bool | None = None,
    gsv_url: str | None = None,
    gsv_server_command: list[str] | str | None = None,
) -> PipelineManifest:
    manifest = load_manifest(project_dir)
    _load_config_into_manifest(project_dir, manifest)
    cfg = manifest.project_config
    training_cfg = cfg.model_copy(update={"gsv_url": gsv_url or cfg.gsv_url})
    _log_stage_start(FEW_SHOT_STAGE, f"target={cfg.gsv_few_shot_target_sec:g}s")
    started_at = monotonic()

    def log_training_progress(event: FewShotTrainingProgress) -> None:
        elapsed = _format_elapsed(monotonic() - started_at)
        if event.status == "output":
            detail = escape(_log_text_snippet(event.detail, max_chars=220))
            label = "fine-tune log" if event.phase.startswith("fine-tune") else "prep log"
            console.print(
                f"[dim]{FEW_SHOT_STAGE} {label} - phase={event.phase} "
                f"elapsed={elapsed} {detail}[/dim]"
            )
            return
        if event.phase == "dataset":
            console.print(
                f"[dim]{FEW_SHOT_STAGE}: dataset ready - elapsed={elapsed} "
                f"{escape(event.detail or '')} log={escape(str(event.log_path or ''))}[/dim]"
            )
            return
        if event.phase == "reuse":
            console.print(
                f"[dim]{FEW_SHOT_STAGE}: reused cached weights - elapsed={elapsed} "
                f"log={escape(str(event.log_path or ''))}[/dim]"
            )
            return
        percent = (event.index / event.total * 100.0) if event.total else 100.0
        console.print(
            f"[dim]{FEW_SHOT_STAGE}: {event.index}/{event.total} ({percent:.1f}%) "
            f"elapsed={elapsed} phase={event.phase} status={event.status} "
            f"log={escape(str(event.log_path or ''))}[/dim]"
        )

    manifest.rights_audit = require_existing_or_confirmed_rights(
        manifest.rights_audit,
        confirm_rights,
        FEW_SHOT_STAGE,
        _manifest_source_path(manifest),
        metadata={"source_derived_few_shot_training": True},
    )
    result = train_few_shot(
        project_dir,
        manifest,
        training_cfg,
        force=force,
        command=gsv_server_command if gsv_server_command is not None else cfg.gsv_server_command,
        progress_callback=log_training_progress,
    )
    manifest.artifacts[FEW_SHOT_ARTIFACT_GPT] = str(result.gpt_weights_path)
    manifest.artifacts[FEW_SHOT_ARTIFACT_SOVITS] = str(result.sovits_weights_path)
    manifest.artifacts["gsv_few_shot_dataset"] = str(result.dataset.list_path)
    manifest.artifacts["gsv_few_shot_manifest"] = str(result.metadata_path)
    source_clip_qc_path = project_dir / "work" / "gpt_sovits" / "few_shot" / "source_clip_qc.json"
    if source_clip_qc_path.exists():
        manifest.artifacts["gsv_few_shot_source_clip_qc"] = str(source_clip_qc_path)
    manifest.rights_audit = record_rights_reliance(
        manifest.rights_audit,
        FEW_SHOT_STAGE,
        _manifest_source_path(manifest),
        metadata={
            "source_derived_few_shot_training": True,
            "selected_duration_sec": result.dataset.total_duration_sec,
            "selected_segment_ids": [item.segment_id for item in result.dataset.items],
            "source_language": cfg.source_language,
            "target_language": cfg.target_language,
            "cross_lingual_voice_transfer": cfg.source_language != cfg.target_language,
            "gpt_weights_sha256": result.gpt_weights_sha256,
            "sovits_weights_sha256": result.sovits_weights_sha256,
        },
    )
    mark_stage(
        manifest,
        FEW_SHOT_STAGE,
        result.status,
        reused_existing=result.reused_existing,
        fingerprint=result.fingerprint,
        selected_duration_sec=result.dataset.total_duration_sec,
        selected_segment_ids=[item.segment_id for item in result.dataset.items],
        source_language=cfg.source_language,
        target_language=cfg.target_language,
        cross_lingual_voice_transfer=cfg.source_language != cfg.target_language,
        source_clip_qc_path=str(source_clip_qc_path) if source_clip_qc_path.exists() else None,
        gpt_weights_path=str(result.gpt_weights_path),
        sovits_weights_path=str(result.sovits_weights_path),
        gpt_weights_sha256=result.gpt_weights_sha256,
        sovits_weights_sha256=result.sovits_weights_sha256,
        gpt_sovits_root=str(result.install.root),
        gpt_sovits_checkout=result.install.checkout,
        gpt_sovits_version=result.install.version,
        log_path=str(result.log_path),
    )
    save_manifest(project_dir, manifest)
    _log_stage_complete(
        FEW_SHOT_STAGE,
        manifest,
        f"{'reused' if result.reused_existing else 'trained'} version={result.install.version}",
    )
    return manifest


def _mock_synthesize(output_path: Path, duration: float, seed: int, sample_rate: int = 48_000) -> Path:
    rng = np.random.default_rng(seed)
    frames = max(1, int(round(duration * sample_rate)))
    t = np.arange(frames, dtype=np.float32) / sample_rate
    freq = 220.0 + (seed % 60)
    tone = 0.04 * np.sin(2 * np.pi * freq * t)
    noise = 0.003 * rng.standard_normal(frames).astype(np.float32)
    data = np.stack([tone + noise, tone + noise], axis=1)
    fade = min(frames, int(0.02 * sample_rate))
    if fade > 1:
        data[:fade] *= np.linspace(0.0, 1.0, fade)[:, None]
        data[-fade:] *= np.linspace(1.0, 0.0, fade)[:, None]
    write_audio(output_path, data, sample_rate)
    return output_path


def _refs_audit_metadata(refs_path: Path, refs: dict[str, Any]) -> dict[str, object]:
    ref_audio_paths = sorted({ref.ref_audio_path for ref in refs.values()})
    aux_ref_audio_paths = sorted({path for ref in refs.values() for path in ref.aux_ref_audio_paths})
    return {
        "refs_path": str(refs_path),
        "refs_sha256": sha256_file(refs_path) if refs_path.exists() else None,
        "ref_audio_paths": ref_audio_paths,
        "aux_ref_audio_paths": aux_ref_audio_paths,
    }


def _tts_candidate_path(project_dir: Path, segment_id: str, candidate_index: int, attempt: int) -> Path:
    suffix = "" if attempt == 0 else f"_retry_{attempt}"
    return project_dir / "work" / "tts" / "candidates" / f"{segment_id}_cand_{candidate_index}{suffix}.wav"


def _qwen_tts_candidate_path(project_dir: Path, segment_id: str, candidate_index: int) -> Path:
    return project_dir / "work" / "tts" / "qwen" / "candidates" / f"{segment_id}_qwen_cand_{candidate_index}.wav"


def _qwen_tts_best_path(project_dir: Path, segment_id: str) -> Path:
    return project_dir / "work" / "tts" / "qwen" / f"{segment_id}_qwen_best.wav"


def _experimental_tts_backend_spec(backend: str) -> _ExperimentalTTSBackendSpec:
    normalized = backend.strip().lower().replace("_", "-")
    if normalized in {"fish", "fish-tts", "fish-speech"}:
        return _ExperimentalTTSBackendSpec(
            stage="synth-fish",
            backend_name="fish-tts",
            analysis_key="fish_tts",
            artifact_key="fish_tts",
            work_dir_name="fish",
            file_label="fish",
        )
    if normalized in {"cosy", "cosyvoice", "cosy-voice"}:
        return _ExperimentalTTSBackendSpec(
            stage="synth-cosyvoice",
            backend_name="cosyvoice",
            analysis_key="cosyvoice_tts",
            artifact_key="cosyvoice_tts",
            work_dir_name="cosyvoice",
            file_label="cosyvoice",
        )
    raise ValueError("experimental TTS backend must be one of: fish, cosyvoice")


def _experimental_tts_candidate_path(
    project_dir: Path,
    spec: _ExperimentalTTSBackendSpec,
    segment_id: str,
    candidate_index: int,
) -> Path:
    return (
        project_dir
        / "work"
        / "tts"
        / spec.work_dir_name
        / "candidates"
        / f"{segment_id}_{spec.file_label}_cand_{candidate_index}.wav"
    )


def _experimental_tts_best_path(
    project_dir: Path,
    spec: _ExperimentalTTSBackendSpec,
    segment_id: str,
) -> Path:
    return project_dir / "work" / "tts" / spec.work_dir_name / f"{segment_id}_{spec.file_label}_best.wav"


def _invalidate_downstream_after_tts_promotion(manifest: PipelineManifest) -> None:
    for stage in ("rvc", "qc", "mix", "export"):
        manifest.stage_state.pop(stage, None)
    for artifact in ("rvc_manifest", "rvc", "qc", "mix", "export"):
        manifest.artifacts.pop(artifact, None)


def synth_step(
    project_dir: Path,
    gsv_url: str | None,
    refs_path: Path,
    mock: bool = False,
    confirm_rights: bool = False,
    gpt_weights_path: str | None = None,
    sovits_weights_path: str | None = None,
    auto_gsv_server: bool | None = None,
    gsv_server_command: list[str] | str | None = None,
    use_trained_gpt: bool = False,
    only_segment_ids: set[str] | None = None,
) -> PipelineManifest:
    manifest = load_manifest(project_dir)
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
        return manifest
    finally:
        for server_manager in reversed(server_managers):
            server_manager.stop()


def synth_qwen_step(
    project_dir: Path,
    refs_path: Path,
    confirm_rights: bool = False,
    *,
    model_id: str | None = None,
    candidate_count: int | None = None,
    candidate_batch_size: int | None = None,
    segment_batch_size: int | None = None,
    target_vram_gb: float | None = None,
    promote: bool = False,
    local_files_only: bool | None = None,
    only_segment_ids: set[str] | None = None,
) -> PipelineManifest:
    manifest = load_manifest(project_dir)
    _load_config_into_manifest(project_dir, manifest)
    cfg = manifest.project_config
    total = len(manifest.segments)
    effective_model_id = model_id or cfg.qwen_tts_model_id
    effective_candidate_count = candidate_count or cfg.qwen_tts_candidate_count
    effective_segment_batch_size = segment_batch_size or cfg.qwen_tts_segment_batch_size
    effective_candidate_batch_size = min(
        effective_candidate_count,
        candidate_batch_size or cfg.qwen_tts_candidate_batch_size,
    )
    effective_target_vram_gb = cfg.qwen_tts_target_vram_gb if target_vram_gb is None else target_vram_gb
    effective_local_files_only = cfg.qwen_tts_local_files_only if local_files_only is None else local_files_only
    _log_stage_start(
        "synth-qwen",
        f"model={effective_model_id}, segments={total}, candidates={effective_candidate_count}, "
        f"segment_batch_size={effective_segment_batch_size}, "
        f"candidate_batch_size={effective_candidate_batch_size}, target_vram_gb={effective_target_vram_gb}, "
        f"promote={promote}",
    )
    refs = load_refs(refs_path, project_dir=project_dir)
    actual_refs_path = resolve_refs_json_path(refs_path, project_dir)
    refs_metadata = _refs_audit_metadata(actual_refs_path, refs)
    manifest.rights_audit = require_existing_or_confirmed_rights(
        manifest.rights_audit,
        confirm_rights,
        "synth-qwen",
        _manifest_source_path(manifest),
        metadata={"backend": "qwen-tts", "model_id": effective_model_id, **refs_metadata},
    )
    use_speaker_refs = bool(cfg.gsv_speaker_models)
    if use_speaker_refs:
        _validate_gsv_speaker_models(project_dir, manifest)
    client = QwenTTSClient(
        model_id=effective_model_id,
        device_map=cfg.qwen_tts_device_map,
        dtype=cfg.qwen_tts_dtype,
        attn_implementation=cfg.qwen_tts_attn_implementation,
        local_files_only=effective_local_files_only,
        target_vram_gb=effective_target_vram_gb,
    )
    console.print(
        f"[cyan]synth-qwen model[/cyan] loading "
        f"device_map={cfg.qwen_tts_device_map} dtype={cfg.qwen_tts_dtype} "
        f"attn={cfg.qwen_tts_attn_implementation} local_files_only={effective_local_files_only}"
    )
    load_model = getattr(client, "load_model", None)
    if callable(load_model):
        load_model()
    memory_snapshot = client.cuda_memory_snapshot() if hasattr(client, "cuda_memory_snapshot") else None
    console.print(f"[dim]synth-qwen cuda after load: {_format_cuda_memory_snapshot(memory_snapshot)}[/dim]")
    source_language = _canonical_language(cfg.source_language)
    target_language = _canonical_language(cfg.target_language)
    started_at = monotonic()
    last_logged_at = started_at
    failed_segments: list[str] = []
    promoted_segments: list[str] = []
    speaker_refs_cache: dict[str, dict[str, GPTSoVITSRef]] = {}

    synthesis_jobs: list[_QwenSegmentSynthesisJob] = []
    for index, segment in enumerate(manifest.segments, start=1):
        if only_segment_ids is not None and segment.id not in only_segment_ids:
            continue
        if not segment.script:
            payload = {
                "backend": "qwen-tts",
                "model_id": effective_model_id,
                "error": "Cannot synthesize without script metadata.",
            }
            segment.analysis["qwen_tts"] = payload
            if promote:
                segment.status = "needs_manual_review"
                segment.errors.append(payload["error"])
                failed_segments.append(segment.id)
            last_logged_at = _log_segment_progress(
                "synth-qwen",
                index,
                total,
                segment,
                manifest,
                started_at,
                last_logged_at,
            )
            continue
        if target_language == "ko":
            preflight = preflight_tts_text(
                segment.script,
                target_language=target_language,
                source_text=segment.source_script.text if segment.source_script else "",
                min_hangul_ratio=cfg.gsv_ko_text_min_hangul_ratio,
            )
            segment.analysis["pre_synth_qwen_text_qc"] = preflight.as_payload()
            if preflight.blocked:
                payload = {
                    "backend": "qwen-tts",
                    "model_id": effective_model_id,
                    "error": "Korean TTS preflight blocked synthesis: " + ", ".join(preflight.issues),
                    "preflight": preflight.as_payload(),
                }
                segment.analysis["qwen_tts"] = payload
                if promote:
                    segment.status = "needs_manual_review"
                    segment.errors.append(payload["error"])
                    failed_segments.append(segment.id)
                last_logged_at = _log_segment_progress(
                    "synth-qwen",
                    index,
                    total,
                    segment,
                    manifest,
                    started_at,
                    last_logged_at,
                )
                continue

        segment_refs = refs
        requested_ref_style = segment.script.ref_style
        resolved_ref_style = requested_ref_style if requested_ref_style in segment_refs else "whisper_close"
        speaker_refs_path: Path | None = None
        if use_speaker_refs and segment.speaker_id:
            speaker_cfg = _gsv_speaker_cfg(cfg, segment)
            if speaker_cfg is not None:
                speaker_refs_path = _resolve_gsv_speaker_path(project_dir, speaker_cfg.refs_path)
                cache_key = str(speaker_refs_path)
                if cache_key not in speaker_refs_cache:
                    speaker_refs_cache[cache_key] = load_refs(speaker_refs_path, project_dir)
                segment_refs = speaker_refs_cache[cache_key]
                if requested_ref_style not in segment_refs:
                    requested_ref_style = speaker_cfg.default_ref_style
                resolved_ref_style = requested_ref_style if requested_ref_style in segment_refs else "whisper_close"
        ref = resolve_ref(segment_refs, requested_ref_style)
        synthesis_jobs.append(
            _QwenSegmentSynthesisJob(
                index=index,
                segment=segment,
                ref=ref,
                resolved_ref_style=resolved_ref_style,
                speaker_refs_path=speaker_refs_path,
                candidates=[],
            )
        )

    generation_kwargs = {
        "temperature": cfg.qwen_tts_temperature,
        "top_p": cfg.qwen_tts_top_p,
        "max_new_tokens": cfg.qwen_tts_max_new_tokens,
    }

    def make_qwen_candidate_job(
        job: _QwenSegmentSynthesisJob,
        candidate_index: int,
    ) -> tuple[_QwenSegmentSynthesisJob, int, int, Path, QwenTTSRequest, dict[str, Any]]:
        segment = job.segment
        seed = cfg.base_seed + job.index * 100 + candidate_index
        candidate_path = _qwen_tts_candidate_path(project_dir, segment.id, candidate_index)
        tts_text_language = _segment_tts_text_language(segment, target_language)
        request = QwenTTSRequest(
            text=segment.script.tts_text,
            language=qwen_language(tts_text_language),
            ref_audio_path=job.ref.ref_audio_path,
            ref_text=job.ref.prompt_text,
            seed=seed,
            x_vector_only_mode=cfg.qwen_tts_x_vector_only_mode,
            generation_kwargs=generation_kwargs,
        )
        payload: dict[str, Any] = {
            "backend": "qwen-tts",
            "model_id": effective_model_id,
            "speaker_id": segment.speaker_id,
            "requested_ref_style": segment.script.ref_style,
            "resolved_ref_style": job.resolved_ref_style,
            "fallback_used": job.resolved_ref_style != segment.script.ref_style,
            "speaker_refs_path": str(job.speaker_refs_path) if job.speaker_refs_path else None,
            "source_language": source_language,
            "target_language": target_language,
            "cross_lingual_voice_transfer": source_language != target_language,
            "target_duration_sec": segment.duration,
            "prompt_lang": job.ref.prompt_lang,
            **request.as_payload(),
        }
        return job, candidate_index, seed, candidate_path, request, payload

    def record_qwen_failure(
        job: _QwenSegmentSynthesisJob,
        candidate_index: int,
        seed: int,
        candidate_path: Path,
        payload: dict[str, Any],
        exc: QwenTTSError,
    ) -> None:
        job.candidates.append(
            TTSCandidate(
                candidate_index=candidate_index,
                seed=seed,
                payload=payload,
                output_path=str(candidate_path),
                backend="qwen-tts",
                error=str(exc),
            )
        )

    def record_qwen_success(
        job: _QwenSegmentSynthesisJob,
        candidate_index: int,
        seed: int,
        candidate_path: Path,
        payload: dict[str, Any],
        result: Any,
    ) -> None:
        segment = job.segment
        if cfg.gsv_trim_edge_silence:
            trim = trim_edge_silence(
                candidate_path,
                threshold_db=cfg.gsv_trim_silence_threshold_db,
                keep_sec=cfg.gsv_trim_silence_keep_sec,
            )
            payload.setdefault("postprocess", {})["edge_silence_trim"] = trim
        duration = duration_sec(candidate_path)
        payload["sample_rate"] = result.sample_rate
        payload["batch_size"] = getattr(result, "batch_size", 1)
        payload["batch_seed"] = getattr(result, "batch_seed", seed)
        too_long = duration_too_long(duration, segment.duration, cfg.duration_tolerance)
        too_short = duration_too_short(duration, segment.duration, cfg.duration_tolerance)
        candidate_ratio = duration_ratio(duration, segment.duration)
        duration_gate = "too_long" if too_long else "too_short" if too_short else "pass"
        language_contract_ok = payload["text"] == segment.script.tts_text
        if target_language == "ko":
            language_contract_ok = language_contract_ok and payload["language"] == "Korean"
        acceptable_for_mix = duration_gate == "pass" and language_contract_ok
        payload["duration_ratio"] = candidate_ratio
        payload["duration_gate"] = duration_gate
        job.candidates.append(
            TTSCandidate(
                candidate_index=candidate_index,
                seed=seed,
                payload=payload,
                output_path=str(candidate_path),
                duration_sec=duration,
                backend="qwen-tts",
                duration_ratio=candidate_ratio,
                duration_gate=duration_gate,
                acceptable_for_mix=acceptable_for_mix,
                selection_score=max(0.0, 1.0 - min(abs(candidate_ratio - 1.0), 1.0)),
                selection_reason=(
                    "duration_and_language_contract_pass"
                    if acceptable_for_mix
                    else "duration_or_language_contract_failed"
                ),
            )
        )

    def run_qwen_request_batch(
        request_batch: list[tuple[_QwenSegmentSynthesisJob, int, int, Path, QwenTTSRequest, dict[str, Any]]],
    ) -> None:
        try:
            batch_synthesize = getattr(client, "synthesize_many_to_files", None)
            if callable(batch_synthesize) and len(request_batch) > 1:
                results = batch_synthesize(
                    [request for _, _, _, _, request, _ in request_batch],
                    [candidate_path for _, _, _, candidate_path, _, _ in request_batch],
                )
            else:
                results = [
                    client.synthesize_to_file(request, candidate_path)
                    for _, _, _, candidate_path, request, _ in request_batch
                ]
        except QwenTTSError as exc:
            if len(request_batch) == 1:
                job, candidate_index, seed, candidate_path, _, payload = request_batch[0]
                record_qwen_failure(job, candidate_index, seed, candidate_path, payload, exc)
                return
            segment_ids = ",".join(job.segment.id for job, *_ in request_batch[:8])
            console.print(
                f"[yellow]synth-qwen batch failed for segments={escape(segment_ids)}; "
                f"retrying one by one: {escape(str(exc))}[/yellow]"
            )
            for item in request_batch:
                run_qwen_request_batch([item])
            return
        for (job, candidate_index, seed, candidate_path, _, payload), result in zip(
            request_batch, results, strict=True
        ):
            record_qwen_success(job, candidate_index, seed, candidate_path, payload, result)

    def finalize_qwen_segment(job: _QwenSegmentSynthesisJob) -> Path | None:
        segment = job.segment
        successful = [
            candidate for candidate in job.candidates if not candidate.error and candidate.duration_sec is not None
        ]
        acceptable = [candidate for candidate in successful if candidate.acceptable_for_mix]
        selected = (
            min(acceptable or successful, key=lambda c: abs((c.duration_sec or 0.0) - segment.duration))
            if successful
            else None
        )
        selected_path: Path | None = None
        if selected is None:
            failed_segments.append(segment.id)
            summary = {
                "backend": "qwen-tts",
                "model_id": effective_model_id,
                "candidate_count": effective_candidate_count,
                "candidate_batch_size": effective_candidate_batch_size,
                "segment_batch_size": effective_segment_batch_size,
                "target_vram_gb": effective_target_vram_gb,
                "selected_candidate_path": None,
                "candidates": [candidate.model_dump(mode="json") for candidate in job.candidates],
                "error": "All Qwen TTS candidates failed.",
            }
            segment.analysis["qwen_tts"] = summary
            if promote:
                segment.status = "failed"
                segment.errors.append("All Qwen TTS candidates failed.")
            return None

        selected.selected = True
        selected_path = _qwen_tts_best_path(project_dir, segment.id)
        ensure_not_same_path(Path(selected.output_path), selected_path)
        selected_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(selected.output_path, selected_path)
        summary = {
            "backend": "qwen-tts",
            "model_id": effective_model_id,
            "candidate_count": effective_candidate_count,
            "candidate_batch_size": effective_candidate_batch_size,
            "segment_batch_size": effective_segment_batch_size,
            "target_vram_gb": effective_target_vram_gb,
            "selected_candidate_path": str(selected_path),
            "selected_duration_gate": selected.duration_gate,
            "selected_acceptable_for_mix": selected.acceptable_for_mix,
            "selected_duration_ratio": selected.duration_ratio,
            "candidates": [candidate.model_dump(mode="json") for candidate in job.candidates],
        }
        segment.analysis["qwen_tts"] = summary
        if promote:
            final_path = project_dir / "work" / "tts" / f"{segment.id}_final.wav"
            ensure_not_same_path(Path(selected.output_path), final_path)
            final_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(selected.output_path, final_path)
            segment.tts = TTSMetadata(
                backend="qwen-tts",
                ref_style=job.resolved_ref_style,
                speed_factor=1.0,
                candidate_count=effective_candidate_count,
                selected_candidate_path=str(final_path),
                candidates=job.candidates,
                source_language=source_language,
                target_language=target_language,
                cross_lingual_voice_transfer=source_language != target_language,
                retry_summary={
                    "selected_duration_gate": selected.duration_gate,
                    "selected_acceptable_for_mix": selected.acceptable_for_mix,
                    "selected_duration_ratio": selected.duration_ratio,
                },
            )
            segment.rvc = None
            segment.qc = None
            segment.mix = {}
            segment.status = "synthesized"
            promoted_segments.append(segment.id)
        return selected_path

    for segment_batch in _chunked(synthesis_jobs, effective_segment_batch_size):
        first_index = segment_batch[0].index
        last_index = segment_batch[-1].index
        batch_candidate_size = 1 if len(segment_batch) > 1 else effective_candidate_batch_size
        for candidate_indexes in _chunked(list(range(effective_candidate_count)), batch_candidate_size):
            request_batch = [
                make_qwen_candidate_job(job, candidate_index)
                for candidate_index in candidate_indexes
                for job in segment_batch
            ]
            console.print(
                f"[cyan]synth-qwen batch[/cyan] segments={first_index}-{last_index}/{total} "
                f"candidates={candidate_indexes[0] + 1}-{candidate_indexes[-1] + 1}/{effective_candidate_count} "
                f"batch_size={len(request_batch)}"
            )
            run_qwen_request_batch(request_batch)
            if first_index == 1 or last_index == total or first_index % _progress_interval(total) == 0:
                memory_snapshot = client.cuda_memory_snapshot() if hasattr(client, "cuda_memory_snapshot") else None
                console.print(f"[dim]synth-qwen cuda: {_format_cuda_memory_snapshot(memory_snapshot)}[/dim]")

        for job in segment_batch:
            selected_path = finalize_qwen_segment(job)
            save_manifest(project_dir, manifest)
            last_logged_at = _log_segment_progress(
                "synth-qwen",
                job.index,
                total,
                job.segment,
                manifest,
                started_at,
                last_logged_at,
                note=f"selected={selected_path}" if selected_path else None,
            )

    if promoted_segments:
        _invalidate_downstream_after_tts_promotion(manifest)
    out_path = project_dir / "work" / "tts" / "qwen" / "qwen_tts_manifest.json"
    write_json_atomic(
        out_path,
        {
            "backend": "qwen-tts",
            "model_id": effective_model_id,
            "promote": promote,
            "candidate_batch_size": effective_candidate_batch_size,
            "segment_batch_size": effective_segment_batch_size,
            "target_vram_gb": effective_target_vram_gb,
            "segments": [
                {
                    "id": segment.id,
                    "qwen_tts": segment.analysis.get("qwen_tts"),
                }
                for segment in manifest.segments
            ],
        },
    )
    manifest.artifacts["qwen_tts"] = str(out_path)
    status = "failed" if promote and failed_segments else "completed"
    mark_stage(
        manifest,
        "synth-qwen",
        status,
        backend="qwen-tts",
        model_id=effective_model_id,
        candidate_count=effective_candidate_count,
        candidate_batch_size=effective_candidate_batch_size,
        segment_batch_size=effective_segment_batch_size,
        target_vram_gb=effective_target_vram_gb,
        promote=promote,
        promoted_segments=promoted_segments,
        failed_segments=failed_segments,
        qwen_tts_manifest=str(out_path),
        segment_counts=_segment_counts(manifest),
    )
    save_manifest(project_dir, manifest)
    _log_stage_complete("synth-qwen", manifest, f"backend=qwen-tts promote={promote}")
    if promote and failed_segments:
        raise QwenTTSError(
            "Qwen TTS synthesis failed for segments: "
            + ", ".join(failed_segments[:20])
            + (" ..." if len(failed_segments) > 20 else "")
        )
    return manifest


def _rvc_profile_for_segment(cfg: ProjectConfig, profile: RVCProfile, segment: Segment) -> RVCProfile:
    if not segment.speaker_id or segment.speaker_id not in cfg.rvc_speaker_models:
        return profile
    speaker_cfg = cfg.rvc_speaker_models[segment.speaker_id]
    overrides = {
        key: value
        for key, value in {
            "f0_method": speaker_cfg.f0_method,
            "index_rate": speaker_cfg.index_rate,
            "f0_up_key": speaker_cfg.f0_up_key,
            "filter_radius": speaker_cfg.filter_radius,
            "resample_sr": speaker_cfg.resample_sr,
            "rms_mix_rate": speaker_cfg.rms_mix_rate,
            "protect": speaker_cfg.protect,
        }.items()
        if value is not None
    }
    return profile.model_copy(update=overrides) if overrides else profile


def _rvc_model_paths(
    project_dir: Path,
    cfg: ProjectConfig,
    segment: Segment,
    manifest: PipelineManifest | None = None,
) -> tuple[Path | None, Path | None]:
    if segment.speaker_id and segment.speaker_id in cfg.rvc_speaker_models:
        speaker_cfg = cfg.rvc_speaker_models[segment.speaker_id]
        return (
            resolve_config_path(project_dir, speaker_cfg.model_path),
            resolve_config_path(project_dir, speaker_cfg.index_path),
        )
    if manifest and manifest.artifacts.get("rvc_model_path"):
        return (
            resolve_config_path(project_dir, manifest.artifacts.get("rvc_model_path")),
            resolve_config_path(project_dir, manifest.artifacts.get("rvc_index_path")),
        )
    return (
        resolve_config_path(project_dir, cfg.rvc_model_path),
        resolve_config_path(project_dir, cfg.rvc_index_path),
    )


def _rvc_metrics(input_path: Path, output_path: Path, segment: Segment, cfg: ProjectConfig) -> dict[str, Any]:
    pre_duration = duration_sec(input_path)
    post_duration = duration_sec(output_path)
    ratio = duration_ratio(post_duration, pre_duration)
    tolerance = cfg.rvc_duration_tolerance if cfg.rvc_duration_tolerance is not None else cfg.duration_tolerance
    peak = peak_dbfs(output_path)
    rms = rms_dbfs(output_path)
    clip = clipping_ratio(output_path)
    issues: list[str] = []
    if abs(ratio - 1.0) > tolerance:
        issues.append("duration_ratio_out_of_range")
    if peak <= -90.0 or rms <= -90.0:
        issues.append("silent_or_empty_audio")
    if clip > 0.001 or peak > -0.05:
        issues.append("clipping_detected")
    return {
        "pre_duration_sec": pre_duration,
        "post_duration_sec": post_duration,
        "duration_ratio": ratio,
        "target_segment_duration_sec": segment.duration,
        "peak_dbfs": peak,
        "rms_dbfs": rms,
        "clipping_ratio": clip,
        "duration_tolerance": tolerance,
        "issues": issues,
        "accepted": not issues,
    }


def _rvc_attempt_payload(
    *,
    profile: RVCProfile,
    output_path: Path,
    model_path: Path | None,
    index_path: Path | None,
    command: list[str] | None = None,
    reused_existing: bool = False,
    returncode: int | None = None,
    elapsed_sec: float | None = None,
    stdout: str = "",
    stderr: str = "",
    metrics: dict[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "profile_name": profile.name,
        "output_path": str(output_path),
        "model_path": str(model_path) if model_path else None,
        "index_path": str(index_path) if index_path else None,
        "settings": profile.model_dump(mode="json"),
        "command": command,
        "reused_existing": reused_existing,
        "returncode": returncode,
        "elapsed_sec": round(elapsed_sec, 6) if elapsed_sec is not None else None,
        "stdout_tail": stdout.strip()[-1200:] if stdout else "",
        "stderr_tail": stderr.strip()[-1200:] if stderr else "",
        "metrics": metrics or {},
        "accepted": bool(metrics and metrics.get("accepted")),
        "error": error,
    }


def _rvc_downstream_required(cfg: ProjectConfig) -> bool:
    return bool(cfg.rvc_required and cfg.rvc_backend == "command")


def _require_rvc_ready_for_downstream(project_dir: Path, manifest: PipelineManifest) -> None:
    cfg = manifest.project_config
    if not _rvc_downstream_required(cfg):
        return
    if manifest.stage_state.get("rvc", {}).get("status") != "completed":
        raise ValueError(RVC_REQUIRED_MESSAGE)
    rvc_root = (project_dir / "work" / "rvc").resolve()
    for segment in manifest.segments:
        if segment.status in SKIP_STATUSES:
            continue
        if not segment.rvc or not segment.rvc.accepted:
            raise ValueError(f"{RVC_REQUIRED_MESSAGE} Segment {segment.id} has no accepted RVC output.")
        if not segment.rvc.output_path:
            raise ValueError(f"{RVC_REQUIRED_MESSAGE} Segment {segment.id} has no RVC output path.")
        resolved = Path(segment.rvc.output_path).resolve()
        try:
            resolved.relative_to(rvc_root)
        except ValueError as exc:
            raise ValueError(
                f"{RVC_REQUIRED_MESSAGE} Segment {segment.id} RVC output is not under work/rvc: {resolved}"
            ) from exc
        if not resolved.exists():
            raise ValueError(f"{RVC_REQUIRED_MESSAGE} Segment {segment.id} RVC output is missing: {resolved}")


def _rvc_train_dataset(project_dir: Path, manifest: PipelineManifest, force: bool) -> tuple[Path, list[dict[str, str]]]:
    dataset_dir = project_dir / "work" / "rvc_train" / "dataset"
    if force and dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    for segment in manifest.segments:
        if segment.status in SKIP_STATUSES:
            continue
        source_path = _resolve_project_read_path(project_dir, segment.audio_for_mix, "audio_for_mix")
        if not source_path.exists():
            raise RVCCommandError(f"train-rvc source segment audio is missing: {source_path}")
        output_path = dataset_dir / f"{segment.id}.wav"
        source_stat = source_path.stat()
        if (
            force
            or not output_path.exists()
            or output_path.stat().st_size != source_stat.st_size
            or output_path.stat().st_mtime_ns != source_stat.st_mtime_ns
        ):
            shutil.copy2(source_path, output_path)
        rows.append({"segment_id": segment.id, "source_path": str(source_path), "dataset_path": str(output_path)})
    if not rows:
        raise RVCCommandError("train-rvc requires at least one source segment audio file.")
    write_json_atomic(dataset_dir / "dataset_manifest.json", {"segments": rows})
    return dataset_dir, rows


def rvc_train_step(
    project_dir: Path,
    confirm_rights: bool = False,
    force: bool = False,
    mock: bool | None = None,
    runner: Any | None = None,
) -> PipelineManifest:
    manifest = load_manifest(project_dir)
    _load_config_into_manifest(project_dir, manifest)
    cfg = manifest.project_config
    backend = "mock" if mock is True else cfg.rvc_train_backend
    _log_stage_start("train-rvc", f"backend={backend}, segments={len(manifest.segments)}")
    if manifest.stage_state.get("synth", {}).get("status") != "completed":
        raise ValueError("train-rvc requires a completed synth stage.")
    if backend == "command":
        if not confirm_rights:
            raise RightsError("Real RVC training requires --confirm-rights for source voice training data.")
        validate_rvc_training_config(project_dir, cfg, real=True)
        source_path = Path(manifest.source_info.path) if manifest.source_info else None
        manifest.rights_audit = merge_rights_audit(
            manifest.rights_audit,
            require_confirmed_rights(
                True,
                "train-rvc",
                source_path,
                metadata={"backend": "command", "experiment_name": cfg.rvc_train_experiment_name},
            ),
        )
        working_dir = resolve_config_path(project_dir, cfg.rvc_train_working_dir)
        client: Any = RVCTrainCommandClient(
            cfg.rvc_train_command,
            working_dir=working_dir,
            timeout_sec=cfg.rvc_train_timeout_sec,
            runner=runner or subprocess.run,
            stream_output=True,
            log_prefix="train-rvc",
        )
    else:
        validate_rvc_training_config(project_dir, cfg, real=False)
        _require_audio_stage_rights(manifest, "train-rvc", confirm_rights, metadata={"backend": "mock"})
        client = RVCTrainMockClient()

    dataset_dir, dataset_rows = _rvc_train_dataset(project_dir, manifest, force)
    work_dir = project_dir / "work" / "rvc_train"
    model_path, index_path = rvc_train_output_paths(project_dir, cfg)
    console.print(
        f"[cyan]train-rvc[/cyan] dataset ready: {len(dataset_rows)} wav(s) -> {escape(str(dataset_dir))}"
    )
    console.print(
        f"[cyan]train-rvc[/cyan] outputs: model={escape(str(model_path))} index={escape(str(index_path))}"
    )
    if isinstance(client, RVCTrainCommandClient):
        command_preview = client.build_command(
            project_dir=project_dir,
            dataset_dir=dataset_dir,
            work_dir=work_dir,
            model_path=model_path,
            index_path=index_path,
            cfg=cfg,
        )
        console.print(f"[dim]train-rvc command: {escape(_format_command_preview(command_preview))}[/dim]")
    console.print(f"[cyan]train-rvc[/cyan] running backend={backend}")
    result = client.train(
        project_dir=project_dir,
        dataset_dir=dataset_dir,
        work_dir=work_dir,
        model_path=model_path,
        index_path=index_path,
        cfg=cfg,
        force=force,
    )
    reuse_note = "reused existing artifacts" if result.reused_existing else f"elapsed={result.elapsed_sec:.1f}s"
    console.print(f"[cyan]train-rvc[/cyan] backend finished: {reuse_note}")
    train_manifest = project_dir / "work" / "rvc_train" / "rvc_train_manifest.json"
    write_json_atomic(
        train_manifest,
        {
            "backend": backend,
            "dataset_dir": str(dataset_dir),
            "dataset_segments": dataset_rows,
            "model_path": str(result.model_path),
            "index_path": str(result.index_path) if result.index_path else None,
            "command": result.command,
            "returncode": result.returncode,
            "elapsed_sec": round(result.elapsed_sec, 6),
            "reused_existing": result.reused_existing,
            "stdout_tail": result.stdout.strip()[-1200:] if result.stdout else "",
            "stderr_tail": result.stderr.strip()[-1200:] if result.stderr else "",
        },
    )
    manifest.artifacts["rvc_train_manifest"] = str(train_manifest)
    manifest.artifacts["rvc_model_path"] = str(result.model_path)
    if result.index_path:
        manifest.artifacts["rvc_index_path"] = str(result.index_path)
    mark_stage(
        manifest,
        "train-rvc",
        "completed",
        backend=backend,
        rvc_train_manifest=str(train_manifest),
        model_path=str(result.model_path),
        index_path=str(result.index_path) if result.index_path else None,
        dataset_segment_count=len(dataset_rows),
    )
    save_manifest(project_dir, manifest)
    _log_stage_complete("train-rvc", manifest, f"backend={backend}")
    return manifest


def skip_rvc_train_for_voice_bank_step(project_dir: Path) -> PipelineManifest:
    manifest = load_manifest(project_dir)
    _load_config_into_manifest(project_dir, manifest)
    cfg = manifest.project_config
    if not cfg.rvc_speaker_models:
        raise ValueError("Cannot skip train-rvc without configured voice-bank RVC speaker models.")
    mark_stage(
        manifest,
        "train-rvc",
        "skipped_pretrained_voice_bank",
        backend="voice_bank",
        speaker_models=sorted(cfg.rvc_speaker_models),
    )
    save_manifest(project_dir, manifest)
    _log_stage_complete("train-rvc", manifest, "skipped_pretrained_voice_bank")
    return manifest


def _train_rvc_ready_for_rvc(manifest: PipelineManifest) -> bool:
    status = manifest.stage_state.get("train-rvc", {}).get("status")
    if status == "completed":
        return True
    return status == "skipped_pretrained_voice_bank" and bool(manifest.project_config.rvc_speaker_models)


def rvc_step(
    project_dir: Path,
    confirm_rights: bool = False,
    force: bool = False,
    mock: bool | None = None,
    runner: Any | None = None,
    only_segment_ids: set[str] | None = None,
) -> PipelineManifest:
    manifest = load_manifest(project_dir)
    _load_config_into_manifest(project_dir, manifest)
    cfg = manifest.project_config
    backend = "mock" if mock is True else cfg.rvc_backend
    _log_stage_start("rvc", f"backend={backend}, segments={len(manifest.segments)}")
    if manifest.stage_state.get("synth", {}).get("status") != "completed":
        raise ValueError("RVC requires a completed synth stage.")
    if not _train_rvc_ready_for_rvc(manifest):
        raise ValueError("RVC requires a completed train-rvc stage.")
    validate_rvc_config(
        project_dir,
        cfg,
        real=backend == "command",
        segments=manifest.segments,
        allow_trained_artifact=True,
    )
    if backend == "command":
        working_dir: Path | None = None
        if not confirm_rights:
            raise RightsError("Real RVC conversion requires --confirm-rights for the source and voice model.")
        source_path = Path(manifest.source_info.path) if manifest.source_info else None
        manifest.rights_audit = merge_rights_audit(
            manifest.rights_audit,
            require_confirmed_rights(
                True,
                "rvc",
                source_path,
                metadata={
                    "backend": "command",
                    "model_path": cfg.rvc_model_path,
                    "index_path": cfg.rvc_index_path,
                    "speaker_models": sorted(cfg.rvc_speaker_models),
                },
            ),
        )
        working_dir = resolve_config_path(project_dir, cfg.rvc_working_dir)
        client: Any = RVCCommandClient(
            cfg.rvc_command,
            working_dir=working_dir,
            timeout_sec=cfg.rvc_timeout_sec,
            runner=runner or subprocess.run,
            stream_output=True,
            log_prefix="rvc",
        )
    else:
        _require_audio_stage_rights(manifest, "rvc", confirm_rights, metadata={"backend": "mock"})
        client = RVCMockClient()

    failed_segments: list[str] = []
    started_at = monotonic()
    last_logged_at = started_at
    total = len(manifest.segments)
    use_batch_rvc = backend == "command" and cfg.rvc_batch_infer and bool(cfg.rvc_batch_command)
    rvc_lane_count = 1 if backend != "command" else _effective_lane_count(cfg.rvc_concurrency, total)
    batch_lane_count = _effective_lane_count(cfg.rvc_batch_concurrency, total) if use_batch_rvc else 1
    console.print(
        f"[cyan]rvc[/cyan] converting segments with {len(cfg.rvc_auto_profiles)} profile candidate(s); "
        f"failure_policy={cfg.rvc_failure_policy} "
        f"mode={'batch' if use_batch_rvc else 'per-segment'} "
        f"concurrency={batch_lane_count if use_batch_rvc else rvc_lane_count}"
    )

    def prepare_segment(
        segment: Segment,
    ) -> tuple[tuple[Path, Path | None, Path | None] | None, str | None]:
        if segment.status in SKIP_STATUSES:
            return None, None
        if not segment.tts or not segment.tts.selected_candidate_path:
            message = "RVC requires segment.tts.selected_candidate_path from synth."
            segment.status = "failed"
            segment.errors.append(message)
            return None, segment.id
        raw_tts_path = segment.rvc.input_path if segment.rvc and segment.rvc.input_path else segment.tts.selected_candidate_path
        input_path = Path(raw_tts_path)
        if not input_path.exists():
            message = f"RVC input does not exist: {input_path}"
            segment.status = "failed"
            segment.errors.append(message)
            segment.rvc = RVCMetadata(
                backend=backend,
                input_path=str(input_path),
                error=message,
            )
            return None, segment.id
        model_path, index_path = _rvc_model_paths(project_dir, cfg, segment, manifest)
        if backend == "command" and (model_path is None or not model_path.exists()):
            message = "RVC requires the model artifact produced by train-rvc."
            segment.status = "failed"
            segment.errors.append(message)
            segment.rvc = RVCMetadata(backend=backend, input_path=str(input_path), error=message)
            return None, segment.id
        return (input_path, model_path, index_path), None

    def convert_segment(index: int, segment: Segment) -> tuple[int, Segment, str | None]:
        prepared, failed_segment_id = prepare_segment(segment)
        if prepared is None:
            return index, segment, failed_segment_id
        input_path, model_path, index_path = prepared
        attempts: list[dict[str, Any]] = []
        candidate_paths: list[str] = []
        accepted_attempt: dict[str, Any] | None = None
        selected_candidate_path: Path | None = None
        profiles = cfg.rvc_auto_profiles
        if cfg.rvc_failure_policy == "error":
            profiles = profiles[:1]
        for profile in profiles:
            effective_profile = _rvc_profile_for_segment(cfg, profile, segment)
            candidate_path = (
                project_dir
                / "work"
                / "rvc"
                / "candidates"
                / segment.id
                / f"{effective_profile.name}.wav"
            )
            candidate_paths.append(str(candidate_path))
            command: list[str] | None = None
            try:
                console.print(
                    f"[dim]rvc candidate: {index}/{total} segment={escape(segment.id)} "
                    f"profile={escape(effective_profile.name)} output={escape(str(candidate_path))}[/dim]"
                )
                if isinstance(client, RVCCommandClient):
                    command = client.build_command(
                        input_path=input_path,
                        output_path=candidate_path,
                        model_path=model_path,
                        index_path=index_path,
                        cfg=cfg,
                        profile=effective_profile,
                        segment_id=segment.id,
                        sid=segment.speaker_id or "",
                    )
                result = client.convert(
                    input_path,
                    candidate_path,
                    model_path=model_path,
                    index_path=index_path,
                    cfg=cfg,
                    profile=effective_profile,
                    segment_id=segment.id,
                    sid=segment.speaker_id or "",
                    force=force,
                )
                command = result.command or command
                metrics = _rvc_metrics(input_path, candidate_path, segment, cfg)
                attempt = _rvc_attempt_payload(
                    profile=effective_profile,
                    output_path=candidate_path,
                    model_path=model_path,
                    index_path=index_path,
                    command=command,
                    reused_existing=result.reused_existing,
                    returncode=result.returncode,
                    elapsed_sec=result.elapsed_sec,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    metrics=metrics,
                )
                attempts.append(attempt)
                if metrics["accepted"]:
                    console.print(
                        f"[dim]rvc accepted: segment={escape(segment.id)} "
                        f"profile={escape(effective_profile.name)} "
                        f"duration_ratio={metrics.get('duration_ratio', 0):.3f} "
                        f"elapsed={result.elapsed_sec:.1f}s"
                        f"{' reused=true' if result.reused_existing else ''}[/dim]"
                    )
                    accepted_attempt = attempt
                    selected_candidate_path = candidate_path
                    break
                console.print(
                    f"[dim]rvc rejected: segment={escape(segment.id)} "
                    f"profile={escape(effective_profile.name)} issues={metrics.get('issues', [])}[/dim]"
                )
            except Exception as exc:
                console.print(
                    f"[yellow]rvc candidate failed[/yellow]: segment={escape(segment.id)} "
                    f"profile={escape(effective_profile.name)} error={escape(str(exc))}"
                )
                attempts.append(
                    _rvc_attempt_payload(
                        profile=effective_profile,
                        output_path=candidate_path,
                        model_path=model_path,
                        index_path=index_path,
                        command=command,
                        error=str(exc),
                    )
                )
                if cfg.rvc_failure_policy == "error":
                    break
        if selected_candidate_path is None or accepted_attempt is None:
            error = "All RVC candidates failed or were rejected."
            failed_segment_id: str | None = None
            if cfg.rvc_allow_pre_rvc_fallback and not _rvc_downstream_required(cfg):
                segment.rvc = RVCMetadata(
                    backend=backend,
                    input_path=str(input_path),
                    output_path=str(input_path),
                    selected_profile_name=None,
                    candidate_paths=candidate_paths,
                    model_path=str(model_path) if model_path else None,
                    index_path=str(index_path) if index_path else None,
                    accepted=False,
                    fallback_used=True,
                    fallback_reason=error,
                    error=error,
                    attempts=attempts,
                )
            else:
                segment.rvc = RVCMetadata(
                    backend=backend,
                    input_path=str(input_path),
                    output_path=None,
                    selected_profile_name=None,
                    candidate_paths=candidate_paths,
                    model_path=str(model_path) if model_path else None,
                    index_path=str(index_path) if index_path else None,
                    accepted=False,
                    error=error,
                    attempts=attempts,
                )
                segment.status = "failed"
                segment.errors.append(error)
                failed_segment_id = segment.id
            return index, segment, failed_segment_id
        final_path = ensure_inside_project(project_dir, project_dir / "work" / "rvc" / f"{segment.id}_final.wav")
        ensure_not_same_path(selected_candidate_path, final_path)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(selected_candidate_path, final_path)
        metrics = dict(accepted_attempt.get("metrics") or {})
        segment.rvc = RVCMetadata(
            backend=backend,
            input_path=str(input_path),
            output_path=str(final_path),
            selected_profile_name=str(accepted_attempt["profile_name"]),
            candidate_paths=candidate_paths,
            model_path=str(model_path) if model_path else None,
            index_path=str(index_path) if index_path else None,
            settings={
                "failure_policy": cfg.rvc_failure_policy,
                "duration_tolerance": metrics.get("duration_tolerance"),
                "selected_settings": accepted_attempt.get("settings", {}),
            },
            pre_duration_sec=metrics.get("pre_duration_sec"),
            post_duration_sec=metrics.get("post_duration_sec"),
            duration_ratio=metrics.get("duration_ratio"),
            accepted=True,
            fallback_used=False,
            command=accepted_attempt.get("command"),
            attempts=attempts,
        )
        segment.status = "rvc_converted"
        return index, segment, None

    def batch_chunks(items: list[Any], size: int) -> list[list[Any]]:
        return [items[start : start + size] for start in range(0, len(items), size)]

    def rvc_output_exists(segment: Segment) -> bool:
        if force or segment.status != "rvc_converted" or not segment.rvc or not segment.rvc.accepted:
            return False
        if not segment.rvc.output_path:
            return False
        output_path = Path(segment.rvc.output_path).expanduser()
        if not output_path.is_absolute():
            output_path = project_dir / output_path
        try:
            return output_path.exists() and output_path.stat().st_size > 0
        except OSError:
            return False

    def convert_segments_batched(segment_jobs: list[tuple[int, Segment]]) -> None:
        nonlocal last_logged_at
        batch_client = RVCBatchCommandClient(
            cfg.rvc_batch_command,
            working_dir=working_dir,
            timeout_sec=cfg.rvc_timeout_sec,
            runner=runner or subprocess.run,
            stream_output=True,
            log_prefix="rvc-batch",
        )
        prepared_segments: list[tuple[int, Segment, Path, Path, Path | None]] = []
        for index, segment in segment_jobs:
            prepared, failed_segment_id = prepare_segment(segment)
            if failed_segment_id:
                failed_segments.append(failed_segment_id)
            if prepared is None:
                last_logged_at = _log_segment_progress("rvc", index, total, segment, manifest, started_at, last_logged_at)
                save_manifest(project_dir, manifest)
                continue
            input_path, model_path, index_path = prepared
            if model_path is None:
                failed_segments.append(segment.id)
                continue
            prepared_segments.append((index, segment, input_path, model_path, index_path))

        attempts_by_segment: dict[str, list[dict[str, Any]]] = {
            segment.id: [] for _, segment, *_ in prepared_segments
        }
        candidate_paths_by_segment: dict[str, list[str]] = {
            segment.id: [] for _, segment, *_ in prepared_segments
        }
        profiles = cfg.rvc_auto_profiles
        if cfg.rvc_failure_policy == "error":
            profiles = profiles[:1]

        pending = prepared_segments
        batches_dir = project_dir / "work" / "rvc" / "batches"
        batch_counter = 0
        batch_counter_lock = Lock()

        def apply_batch_result(
            entries: list[tuple[int, Segment, Path, Path, Path | None, RVCProfile, Path]],
            results: dict[str, RVCCommandResult] | None,
            batch_error: Exception | None = None,
        ) -> list[tuple[int, Segment, Path, Path, Path | None]]:
            nonlocal last_logged_at
            rejected: list[tuple[int, Segment, Path, Path, Path | None]] = []
            for index, segment, input_path, model_path, index_path, profile, candidate_path in entries:
                attempts = attempts_by_segment[segment.id]
                result = results.get(segment.id) if results else None
                if batch_error is not None or result is None or result.returncode != 0:
                    error = str(batch_error) if batch_error is not None else "RVC batch command did not return a result."
                    command = None
                    if result is not None:
                        command = result.command
                        error = result.stderr or result.stdout or f"RVC batch command failed with exit code {result.returncode}."
                    console.print(
                        f"[yellow]rvc candidate failed[/yellow]: segment={escape(segment.id)} "
                        f"profile={escape(profile.name)} error={escape(error)}"
                    )
                    attempts.append(
                        _rvc_attempt_payload(
                            profile=profile,
                            output_path=candidate_path,
                            model_path=model_path,
                            index_path=index_path,
                            command=command,
                            error=error,
                        )
                    )
                    rejected.append((index, segment, input_path, model_path, index_path))
                    continue

                metrics = _rvc_metrics(input_path, candidate_path, segment, cfg)
                attempt = _rvc_attempt_payload(
                    profile=profile,
                    output_path=candidate_path,
                    model_path=model_path,
                    index_path=index_path,
                    command=result.command,
                    reused_existing=result.reused_existing,
                    returncode=result.returncode,
                    elapsed_sec=result.elapsed_sec,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    metrics=metrics,
                )
                attempts.append(attempt)
                if not metrics["accepted"]:
                    console.print(
                        f"[dim]rvc rejected: segment={escape(segment.id)} "
                        f"profile={escape(profile.name)} issues={metrics.get('issues', [])}[/dim]"
                    )
                    rejected.append((index, segment, input_path, model_path, index_path))
                    continue

                final_path = ensure_inside_project(project_dir, project_dir / "work" / "rvc" / f"{segment.id}_final.wav")
                ensure_not_same_path(candidate_path, final_path)
                final_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(candidate_path, final_path)
                segment.rvc = RVCMetadata(
                    backend=backend,
                    input_path=str(input_path),
                    output_path=str(final_path),
                    selected_profile_name=str(attempt["profile_name"]),
                    candidate_paths=candidate_paths_by_segment[segment.id],
                    model_path=str(model_path),
                    index_path=str(index_path) if index_path else None,
                    settings={
                        "failure_policy": cfg.rvc_failure_policy,
                        "duration_tolerance": metrics.get("duration_tolerance"),
                        "selected_settings": attempt.get("settings", {}),
                        "execution_mode": "batch",
                    },
                    pre_duration_sec=metrics.get("pre_duration_sec"),
                    post_duration_sec=metrics.get("post_duration_sec"),
                    duration_ratio=metrics.get("duration_ratio"),
                    accepted=True,
                    fallback_used=False,
                    command=attempt.get("command"),
                    attempts=attempts,
                )
                segment.status = "rvc_converted"
                console.print(
                    f"[dim]rvc accepted: segment={escape(segment.id)} "
                    f"profile={escape(profile.name)} "
                    f"duration_ratio={metrics.get('duration_ratio', 0):.3f} "
                    f"elapsed={result.elapsed_sec:.1f}s"
                    f"{' reused=true' if result.reused_existing else ''}[/dim]"
                )
                last_logged_at = _log_segment_progress("rvc", index, total, segment, manifest, started_at, last_logged_at)
            return rejected

        for profile in profiles:
            if not pending:
                break
            grouped: dict[tuple[str, str, str], list[tuple[int, Segment, Path, Path, Path | None, RVCProfile, Path]]] = {}
            for index, segment, input_path, model_path, index_path in pending:
                effective_profile = _rvc_profile_for_segment(cfg, profile, segment)
                candidate_path = (
                    project_dir
                    / "work"
                    / "rvc"
                    / "candidates"
                    / segment.id
                    / f"{effective_profile.name}.wav"
                )
                candidate_paths_by_segment[segment.id].append(str(candidate_path))
                console.print(
                    f"[dim]rvc candidate: {index}/{total} segment={escape(segment.id)} "
                    f"profile={escape(effective_profile.name)} output={escape(str(candidate_path))}[/dim]"
                )
                profile_key = json.dumps(effective_profile.model_dump(mode="json"), sort_keys=True)
                key = (str(model_path.resolve()), str(index_path.resolve()) if index_path else "", profile_key)
                grouped.setdefault(key, []).append(
                    (index, segment, input_path, model_path, index_path, effective_profile, candidate_path)
                )

            next_pending: list[tuple[int, Segment, Path, Path, Path | None]] = []
            batch_tasks: list[list[tuple[int, Segment, Path, Path, Path | None, RVCProfile, Path]]] = []
            for entries in grouped.values():
                batch_tasks.extend(batch_chunks(entries, cfg.rvc_batch_size))

            def run_batch(
                entries: list[tuple[int, Segment, Path, Path, Path | None, RVCProfile, Path]]
            ) -> tuple[
                list[tuple[int, Segment, Path, Path, Path | None, RVCProfile, Path]],
                dict[str, RVCCommandResult],
            ]:
                nonlocal batch_counter
                with batch_counter_lock:
                    batch_counter += 1
                    batch_id = batch_counter
                first = entries[0]
                _, _, _, model_path, index_path, effective_profile, _ = first
                jobs = [
                    RVCBatchJob(
                        segment_id=segment.id,
                        input_path=input_path,
                        output_path=candidate_path,
                        model_path=model_path,
                        index_path=index_path,
                        profile=effective_profile,
                        sid=segment.speaker_id or "",
                    )
                    for _, segment, input_path, model_path, index_path, effective_profile, candidate_path in entries
                ]
                jobs_path = batches_dir / f"batch_{batch_id:04d}_{effective_profile.name}_jobs.jsonl"
                results_path = batches_dir / f"batch_{batch_id:04d}_{effective_profile.name}_results.jsonl"
                console.print(
                    f"[cyan]rvc batch[/cyan] profile={escape(effective_profile.name)} "
                    f"jobs={len(jobs)} model={escape(str(model_path))}"
                )
                return entries, batch_client.convert_many(
                    jobs,
                    jobs_path=jobs_path,
                    results_path=results_path,
                    model_path=model_path,
                    index_path=index_path,
                    cfg=cfg,
                    profile=effective_profile,
                    force=force,
                )

            if batch_lane_count > 1 and len(batch_tasks) > 1:
                with ThreadPoolExecutor(max_workers=batch_lane_count) as executor:
                    futures = [executor.submit(run_batch, entries) for entries in batch_tasks]
                    for future in as_completed(futures):
                        try:
                            entries, results = future.result()
                        except Exception as exc:
                            entries = batch_tasks[futures.index(future)]
                            next_pending.extend(apply_batch_result(entries, None, exc))
                        else:
                            next_pending.extend(apply_batch_result(entries, results))
                        save_manifest(project_dir, manifest)
            else:
                for entries in batch_tasks:
                    try:
                        entries, results = run_batch(entries)
                    except Exception as exc:
                        next_pending.extend(apply_batch_result(entries, None, exc))
                    else:
                        next_pending.extend(apply_batch_result(entries, results))
                    save_manifest(project_dir, manifest)
            pending = next_pending

        for index, segment, input_path, model_path, index_path in pending:
            error = "All RVC candidates failed or were rejected."
            attempts = attempts_by_segment[segment.id]
            if cfg.rvc_allow_pre_rvc_fallback and not _rvc_downstream_required(cfg):
                segment.rvc = RVCMetadata(
                    backend=backend,
                    input_path=str(input_path),
                    output_path=str(input_path),
                    selected_profile_name=None,
                    candidate_paths=candidate_paths_by_segment[segment.id],
                    model_path=str(model_path),
                    index_path=str(index_path) if index_path else None,
                    accepted=False,
                    fallback_used=True,
                    fallback_reason=error,
                    error=error,
                    attempts=attempts,
                )
            else:
                segment.rvc = RVCMetadata(
                    backend=backend,
                    input_path=str(input_path),
                    output_path=None,
                    selected_profile_name=None,
                    candidate_paths=candidate_paths_by_segment[segment.id],
                    model_path=str(model_path),
                    index_path=str(index_path) if index_path else None,
                    accepted=False,
                    error=error,
                    attempts=attempts,
                )
                segment.status = "failed"
                segment.errors.append(error)
                failed_segments.append(segment.id)
            last_logged_at = _log_segment_progress("rvc", index, total, segment, manifest, started_at, last_logged_at)
        if pending:
            save_manifest(project_dir, manifest)

    indexed_segments = [
        (index, segment)
        for index, segment in enumerate(manifest.segments, start=1)
        if only_segment_ids is None or segment.id in only_segment_ids
    ]
    skipped_completed = sum(1 for _, segment in indexed_segments if rvc_output_exists(segment))
    segment_jobs = [
        (index, segment)
        for index, segment in indexed_segments
        if segment.status not in SKIP_STATUSES and not rvc_output_exists(segment)
    ]
    if skipped_completed:
        console.print(f"[dim]rvc skipped {skipped_completed} already converted segment(s)[/dim]")
    if use_batch_rvc and len(segment_jobs) > 1:
        convert_segments_batched(segment_jobs)
    elif backend == "command" and rvc_lane_count > 1 and len(segment_jobs) > 1:
        with ThreadPoolExecutor(max_workers=rvc_lane_count) as executor:
            futures = [executor.submit(convert_segment, index, segment) for index, segment in segment_jobs]
            for future in as_completed(futures):
                index, segment, failed_segment_id = future.result()
                if failed_segment_id:
                    failed_segments.append(failed_segment_id)
                last_logged_at = _log_segment_progress("rvc", index, total, segment, manifest, started_at, last_logged_at)
                save_manifest(project_dir, manifest)
    else:
        for index, segment in segment_jobs:
            index, segment, failed_segment_id = convert_segment(index, segment)
            if failed_segment_id:
                failed_segments.append(failed_segment_id)
            last_logged_at = _log_segment_progress("rvc", index, total, segment, manifest, started_at, last_logged_at)
            save_manifest(project_dir, manifest)

    out_path = project_dir / "work" / "rvc" / "rvc_manifest.json"
    write_json_atomic(
        out_path,
        {
            "backend": backend,
            "execution_mode": "batch" if use_batch_rvc else "per_segment",
            "segments": [
                {"id": segment.id, "rvc": segment.rvc.model_dump(mode="json") if segment.rvc else None}
                for segment in manifest.segments
            ],
        },
    )
    manifest.artifacts["rvc_manifest"] = str(out_path)
    effective_concurrency = batch_lane_count if use_batch_rvc else rvc_lane_count
    if failed_segments:
        mark_stage(
            manifest,
            "rvc",
            "failed",
            backend=backend,
            failed_segments=failed_segments,
            rvc_manifest=str(out_path),
            concurrency=effective_concurrency,
            execution_mode="batch" if use_batch_rvc else "per_segment",
            segment_counts=_segment_counts(manifest),
        )
        save_manifest(project_dir, manifest)
        raise RVCCommandError(
            "RVC conversion failed for segments: "
            + ", ".join(failed_segments[:20])
            + (" ..." if len(failed_segments) > 20 else "")
        )
    mark_stage(
        manifest,
        "rvc",
        "completed",
        backend=backend,
        rvc_manifest=str(out_path),
        concurrency=effective_concurrency,
        execution_mode="batch" if use_batch_rvc else "per_segment",
        segment_counts=_segment_counts(manifest),
    )
    save_manifest(project_dir, manifest)
    _log_stage_complete("rvc", manifest, f"backend={backend}")
    return manifest


def qc_step(
    project_dir: Path,
    backend_kind: str,
    confirm_rights: bool = False,
    only_segment_ids: set[str] | None = None,
) -> PipelineManifest:
    manifest = load_manifest(project_dir)
    _load_config_into_manifest(project_dir, manifest)
    total = len(manifest.segments)
    _log_stage_start("qc", f"backend={backend_kind}, segments={total}")
    _require_audio_stage_rights(manifest, "qc", confirm_rights, metadata={"backend": backend_kind})
    _require_rvc_ready_for_downstream(project_dir, manifest)
    cfg = manifest.project_config
    backend = create_gemma_backend(backend_kind, _gemma_backend_config(cfg))
    context = _gemma_context(manifest)
    started_at = monotonic()
    last_logged_at = started_at
    for index, segment in enumerate(manifest.segments, start=1):
        if only_segment_ids is not None and segment.id not in only_segment_ids:
            continue
        if segment.status in {"needs_manual_review", "failed"}:
            last_logged_at = _log_segment_progress(
                "qc", index, total, segment, manifest, started_at, last_logged_at
            )
            continue
        if not segment.tts or not segment.tts.selected_candidate_path or not segment.script:
            segment.status = "needs_manual_review"
            segment.errors.append("Cannot QC without selected TTS and script.")
            last_logged_at = _log_segment_progress(
                "qc", index, total, segment, manifest, started_at, last_logged_at
            )
            continue
        audio_path = (
            Path(segment.rvc.output_path)
            if segment.rvc and segment.rvc.output_path
            else Path(segment.tts.selected_candidate_path)
        )
        audio_metrics = measure_audio_qc(audio_path, segment.duration)
        try:
            gemma_result = validate_gemma_task_response(
                "qc",
                backend.qc_audio(audio_path, segment.script.tts_text, segment, context),
            )
        except Exception as exc:
            gemma_result = {"recommendation": "manual_review", "issues": [str(exc)]}
        qc = score_qc(audio_metrics, gemma_result)
        segment.qc = qc
        segment.status = qc.status
        last_logged_at = _log_segment_progress(
            "qc", index, total, segment, manifest, started_at, last_logged_at
        )
    out_path = project_dir / "work" / "qc" / "qc_manifest.json"
    write_json_atomic(out_path, {"segments": [s.model_dump(mode="json") for s in manifest.segments]})
    manifest.artifacts["qc"] = str(out_path)
    mark_stage(manifest, "qc", "completed", backend=backend_kind, segment_counts=_segment_counts(manifest))
    save_manifest(project_dir, manifest)
    _log_stage_complete("qc", manifest, f"backend={backend_kind}")
    return manifest


def synth_experimental_tts_step(
    project_dir: Path,
    refs_path: Path,
    *,
    backend: str,
    confirm_rights: bool = False,
    base_url: str | None = None,
    candidate_count: int | None = None,
    promote: bool = False,
    only_segment_ids: set[str] | None = None,
) -> PipelineManifest:
    spec = _experimental_tts_backend_spec(backend)
    manifest = load_manifest(project_dir)
    _load_config_into_manifest(project_dir, manifest)
    cfg = manifest.project_config
    total = len(manifest.segments)
    effective_candidate_count = candidate_count or (
        cfg.fish_tts_candidate_count if spec.backend_name == "fish-tts" else cfg.cosyvoice_candidate_count
    )
    effective_base_url = base_url or (
        cfg.fish_tts_base_url if spec.backend_name == "fish-tts" else cfg.cosyvoice_base_url
    )
    _log_stage_start(
        spec.stage,
        f"backend={spec.backend_name}, base_url={effective_base_url}, "
        f"segments={total}, candidates={effective_candidate_count}, promote={promote}",
    )
    refs = load_refs(refs_path, project_dir=project_dir)
    actual_refs_path = resolve_refs_json_path(refs_path, project_dir)
    refs_metadata = _refs_audit_metadata(actual_refs_path, refs)
    manifest.rights_audit = require_existing_or_confirmed_rights(
        manifest.rights_audit,
        confirm_rights,
        spec.stage,
        _manifest_source_path(manifest),
        metadata={"backend": spec.backend_name, "base_url": effective_base_url, **refs_metadata},
    )
    use_speaker_refs = bool(cfg.gsv_speaker_models)
    if use_speaker_refs:
        _validate_gsv_speaker_models(project_dir, manifest)

    if spec.backend_name == "fish-tts":
        client = FishSpeechTTSClient(base_url=effective_base_url, timeout_sec=cfg.fish_tts_timeout_sec)
        generation_kwargs: dict[str, Any] = {
            "chunk_length": cfg.fish_tts_chunk_length,
            "temperature": cfg.fish_tts_temperature,
            "top_p": cfg.fish_tts_top_p,
            "repetition_penalty": cfg.fish_tts_repetition_penalty,
            "max_new_tokens": cfg.fish_tts_max_new_tokens,
            "normalize": cfg.fish_tts_normalize,
            "latency": cfg.fish_tts_latency,
        }
    else:
        client = CosyVoiceTTSClient(
            base_url=effective_base_url,
            mode=cfg.cosyvoice_mode,
            sample_rate=cfg.cosyvoice_sample_rate,
            timeout_sec=cfg.cosyvoice_timeout_sec,
            instruct_text=cfg.cosyvoice_instruct_text,
        )
        generation_kwargs = {}
        if cfg.cosyvoice_instruct_text:
            generation_kwargs["instruct_text"] = cfg.cosyvoice_instruct_text

    load_model = getattr(client, "load_model", None)
    if callable(load_model):
        load_model()

    source_language = _canonical_language(cfg.source_language)
    target_language = _canonical_language(cfg.target_language)
    started_at = monotonic()
    last_logged_at = started_at
    failed_segments: list[str] = []
    promoted_segments: list[str] = []
    speaker_refs_cache: dict[str, dict[str, GPTSoVITSRef]] = {}
    synthesis_jobs: list[_ExperimentalTTSSegmentSynthesisJob] = []

    for index, segment in enumerate(manifest.segments, start=1):
        if only_segment_ids is not None and segment.id not in only_segment_ids:
            continue
        if not segment.script:
            payload = {
                "backend": spec.backend_name,
                "base_url": effective_base_url,
                "error": "Cannot synthesize without script metadata.",
            }
            segment.analysis[spec.analysis_key] = payload
            if promote:
                segment.status = "needs_manual_review"
                segment.errors.append(payload["error"])
                failed_segments.append(segment.id)
            last_logged_at = _log_segment_progress(
                spec.stage,
                index,
                total,
                segment,
                manifest,
                started_at,
                last_logged_at,
            )
            continue
        if target_language == "ko":
            preflight = preflight_tts_text(
                segment.script,
                target_language=target_language,
                source_text=segment.source_script.text if segment.source_script else "",
                min_hangul_ratio=cfg.gsv_ko_text_min_hangul_ratio,
            )
            segment.analysis[f"pre_{spec.stage.replace('-', '_')}_text_qc"] = preflight.as_payload()
            if preflight.blocked:
                payload = {
                    "backend": spec.backend_name,
                    "base_url": effective_base_url,
                    "error": "Korean TTS preflight blocked synthesis: " + ", ".join(preflight.issues),
                    "preflight": preflight.as_payload(),
                }
                segment.analysis[spec.analysis_key] = payload
                if promote:
                    segment.status = "needs_manual_review"
                    segment.errors.append(payload["error"])
                    failed_segments.append(segment.id)
                last_logged_at = _log_segment_progress(
                    spec.stage,
                    index,
                    total,
                    segment,
                    manifest,
                    started_at,
                    last_logged_at,
                )
                continue

        segment_refs = refs
        requested_ref_style = segment.script.ref_style
        resolved_ref_style = requested_ref_style if requested_ref_style in segment_refs else "whisper_close"
        speaker_refs_path: Path | None = None
        if use_speaker_refs and segment.speaker_id:
            speaker_cfg = _gsv_speaker_cfg(cfg, segment)
            if speaker_cfg is not None:
                speaker_refs_path = _resolve_gsv_speaker_path(project_dir, speaker_cfg.refs_path)
                cache_key = str(speaker_refs_path)
                if cache_key not in speaker_refs_cache:
                    speaker_refs_cache[cache_key] = load_refs(speaker_refs_path, project_dir)
                segment_refs = speaker_refs_cache[cache_key]
                if requested_ref_style not in segment_refs:
                    requested_ref_style = speaker_cfg.default_ref_style
                resolved_ref_style = requested_ref_style if requested_ref_style in segment_refs else "whisper_close"
        ref = resolve_ref(segment_refs, requested_ref_style)
        synthesis_jobs.append(
            _ExperimentalTTSSegmentSynthesisJob(
                index=index,
                segment=segment,
                ref=ref,
                resolved_ref_style=resolved_ref_style,
                speaker_refs_path=speaker_refs_path,
                candidates=[],
            )
        )

    def make_candidate_job(
        job: _ExperimentalTTSSegmentSynthesisJob,
        candidate_index: int,
    ) -> tuple[_ExperimentalTTSSegmentSynthesisJob, int, int, Path, ExperimentalTTSRequest, dict[str, Any]]:
        segment = job.segment
        seed = cfg.base_seed + job.index * 100 + candidate_index
        candidate_path = _experimental_tts_candidate_path(project_dir, spec, segment.id, candidate_index)
        tts_text_language = _segment_tts_text_language(segment, target_language)
        ref_audio_path = Path(job.ref.ref_audio_path).expanduser()
        if not ref_audio_path.is_absolute():
            ref_audio_path = project_dir / ref_audio_path
        request = ExperimentalTTSRequest(
            text=segment.script.tts_text,
            language=tts_text_language,
            ref_audio_path=str(ref_audio_path),
            ref_text=job.ref.prompt_text,
            seed=seed,
            generation_kwargs=generation_kwargs,
        )
        payload: dict[str, Any] = {
            "backend": spec.backend_name,
            "base_url": effective_base_url,
            "speaker_id": segment.speaker_id,
            "requested_ref_style": segment.script.ref_style,
            "resolved_ref_style": job.resolved_ref_style,
            "fallback_used": job.resolved_ref_style != segment.script.ref_style,
            "speaker_refs_path": str(job.speaker_refs_path) if job.speaker_refs_path else None,
            "source_language": source_language,
            "target_language": target_language,
            "cross_lingual_voice_transfer": source_language != target_language,
            "target_duration_sec": segment.duration,
            "prompt_lang": job.ref.prompt_lang,
            **request.as_payload(),
        }
        return job, candidate_index, seed, candidate_path, request, payload

    def record_failure(
        job: _ExperimentalTTSSegmentSynthesisJob,
        candidate_index: int,
        seed: int,
        candidate_path: Path,
        payload: dict[str, Any],
        exc: ExperimentalTTSError,
    ) -> None:
        job.candidates.append(
            TTSCandidate(
                candidate_index=candidate_index,
                seed=seed,
                payload=payload,
                output_path=str(candidate_path),
                backend=spec.backend_name,
                error=str(exc),
            )
        )

    def record_success(
        job: _ExperimentalTTSSegmentSynthesisJob,
        candidate_index: int,
        seed: int,
        candidate_path: Path,
        payload: dict[str, Any],
        result: Any,
    ) -> None:
        segment = job.segment
        if cfg.gsv_trim_edge_silence:
            trim = trim_edge_silence(
                candidate_path,
                threshold_db=cfg.gsv_trim_silence_threshold_db,
                keep_sec=cfg.gsv_trim_silence_keep_sec,
            )
            payload.setdefault("postprocess", {})["edge_silence_trim"] = trim
        duration = duration_sec(candidate_path)
        payload["sample_rate"] = result.sample_rate
        too_long = duration_too_long(duration, segment.duration, cfg.duration_tolerance)
        too_short = duration_too_short(duration, segment.duration, cfg.duration_tolerance)
        candidate_ratio = duration_ratio(duration, segment.duration)
        duration_gate = "too_long" if too_long else "too_short" if too_short else "pass"
        language_contract_ok = payload["text"] == segment.script.tts_text
        acceptable_for_mix = duration_gate == "pass" and language_contract_ok
        payload["duration_ratio"] = candidate_ratio
        payload["duration_gate"] = duration_gate
        job.candidates.append(
            TTSCandidate(
                candidate_index=candidate_index,
                seed=seed,
                payload=payload,
                output_path=str(candidate_path),
                duration_sec=duration,
                backend=spec.backend_name,
                duration_ratio=candidate_ratio,
                duration_gate=duration_gate,
                acceptable_for_mix=acceptable_for_mix,
                selection_score=max(0.0, 1.0 - min(abs(candidate_ratio - 1.0), 1.0)),
                selection_reason=(
                    "duration_and_text_contract_pass"
                    if acceptable_for_mix
                    else "duration_or_text_contract_failed"
                ),
            )
        )

    def finalize_segment(job: _ExperimentalTTSSegmentSynthesisJob) -> Path | None:
        segment = job.segment
        successful = [
            candidate for candidate in job.candidates if not candidate.error and candidate.duration_sec is not None
        ]
        acceptable = [candidate for candidate in successful if candidate.acceptable_for_mix]
        selected = (
            min(acceptable or successful, key=lambda c: abs((c.duration_sec or 0.0) - segment.duration))
            if successful
            else None
        )
        if selected is None:
            failed_segments.append(segment.id)
            summary = {
                "backend": spec.backend_name,
                "base_url": effective_base_url,
                "candidate_count": effective_candidate_count,
                "selected_candidate_path": None,
                "candidates": [candidate.model_dump(mode="json") for candidate in job.candidates],
                "error": f"All {spec.backend_name} candidates failed.",
            }
            segment.analysis[spec.analysis_key] = summary
            if promote:
                segment.status = "failed"
                segment.errors.append(f"All {spec.backend_name} candidates failed.")
            return None

        selected.selected = True
        selected_path = _experimental_tts_best_path(project_dir, spec, segment.id)
        ensure_not_same_path(Path(selected.output_path), selected_path)
        selected_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(selected.output_path, selected_path)
        summary = {
            "backend": spec.backend_name,
            "base_url": effective_base_url,
            "candidate_count": effective_candidate_count,
            "selected_candidate_path": str(selected_path),
            "selected_duration_gate": selected.duration_gate,
            "selected_acceptable_for_mix": selected.acceptable_for_mix,
            "selected_duration_ratio": selected.duration_ratio,
            "candidates": [candidate.model_dump(mode="json") for candidate in job.candidates],
        }
        segment.analysis[spec.analysis_key] = summary
        if promote:
            final_path = project_dir / "work" / "tts" / f"{segment.id}_final.wav"
            ensure_not_same_path(Path(selected.output_path), final_path)
            final_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(selected.output_path, final_path)
            segment.tts = TTSMetadata(
                backend=spec.backend_name,
                ref_style=job.resolved_ref_style,
                speed_factor=1.0,
                candidate_count=effective_candidate_count,
                selected_candidate_path=str(final_path),
                candidates=job.candidates,
                source_language=source_language,
                target_language=target_language,
                cross_lingual_voice_transfer=source_language != target_language,
                retry_summary={
                    "selected_duration_gate": selected.duration_gate,
                    "selected_acceptable_for_mix": selected.acceptable_for_mix,
                    "selected_duration_ratio": selected.duration_ratio,
                },
            )
            segment.rvc = None
            segment.qc = None
            segment.mix = {}
            segment.status = "synthesized"
            promoted_segments.append(segment.id)
        return selected_path

    for job in synthesis_jobs:
        for candidate_index in range(effective_candidate_count):
            item = make_candidate_job(job, candidate_index)
            _, _, seed, candidate_path, request, payload = item
            try:
                result = client.synthesize_to_file(request, candidate_path)
            except ExperimentalTTSError as exc:
                record_failure(job, candidate_index, seed, candidate_path, payload, exc)
                continue
            record_success(job, candidate_index, seed, candidate_path, payload, result)
        selected_path = finalize_segment(job)
        save_manifest(project_dir, manifest)
        last_logged_at = _log_segment_progress(
            spec.stage,
            job.index,
            total,
            job.segment,
            manifest,
            started_at,
            last_logged_at,
            note=f"selected={selected_path}" if selected_path else None,
        )

    if promoted_segments:
        _invalidate_downstream_after_tts_promotion(manifest)
    out_path = project_dir / "work" / "tts" / spec.work_dir_name / f"{spec.analysis_key}_manifest.json"
    write_json_atomic(
        out_path,
        {
            "backend": spec.backend_name,
            "base_url": effective_base_url,
            "promote": promote,
            "candidate_count": effective_candidate_count,
            "segments": [
                {
                    "id": segment.id,
                    spec.analysis_key: segment.analysis.get(spec.analysis_key),
                }
                for segment in manifest.segments
            ],
        },
    )
    manifest.artifacts[spec.artifact_key] = str(out_path)
    status = "failed" if promote and failed_segments else "completed"
    mark_stage(
        manifest,
        spec.stage,
        status,
        backend=spec.backend_name,
        base_url=effective_base_url,
        candidate_count=effective_candidate_count,
        promote=promote,
        promoted_segments=promoted_segments,
        failed_segments=failed_segments,
        manifest_path=str(out_path),
        segment_counts=_segment_counts(manifest),
    )
    save_manifest(project_dir, manifest)
    _log_stage_complete(spec.stage, manifest, f"backend={spec.backend_name} promote={promote}")
    if promote and failed_segments:
        raise ExperimentalTTSError(
            f"{spec.backend_name} synthesis failed for segments: "
            + ", ".join(failed_segments[:20])
            + (" ..." if len(failed_segments) > 20 else "")
        )
    return manifest


def regenerate_needs_step(
    project_dir: Path,
    *,
    refs_path: Path = Path("refs/refs.json"),
    confirm_rights: bool = False,
    gemma_backend: str = "mock",
    tts_backend: str = "gpt-sovits",
    gsv_url: str | None = None,
    gpt_weights_path: str | None = None,
    sovits_weights_path: str | None = None,
    use_trained_gpt: bool = False,
    auto_gsv_server: bool | None = None,
    gsv_server_command: list[str] | str | None = None,
    qwen_model_id: str | None = None,
    qwen_candidate_count: int | None = None,
    qwen_local_files_only: bool | None = None,
    experimental_tts_base_url: str | None = None,
    experimental_tts_candidate_count: int | None = None,
) -> PipelineManifest:
    manifest = load_manifest(project_dir)
    _load_config_into_manifest(project_dir, manifest)
    target_ids = {
        segment.id
        for segment in manifest.segments
        if segment.status == "needs_regeneration"
        and segment.qc is not None
        and segment.qc.recommendation == "regenerate"
    }
    _log_stage_start("regenerate", f"segments={len(target_ids)}, tts_backend={tts_backend}")
    if not target_ids:
        mark_stage(
            manifest,
            "regenerate",
            "skipped",
            target_status="needs_regeneration",
            target_segments=[],
            segment_counts=_segment_counts(manifest),
        )
        save_manifest(project_dir, manifest)
        _log_stage_complete("regenerate", manifest, "no target segments")
        return manifest

    for segment in manifest.segments:
        if segment.id not in target_ids:
            continue
        segment.rvc = None
        segment.mix = {}
    _invalidate_downstream_after_tts_promotion(manifest)
    save_manifest(project_dir, manifest)

    backend = tts_backend.strip().lower().replace("_", "-")
    if backend in {"gpt-sovits", "gsv"}:
        synth_step(
            project_dir,
            gsv_url,
            refs_path,
            mock=False,
            confirm_rights=confirm_rights,
            gpt_weights_path=gpt_weights_path,
            sovits_weights_path=sovits_weights_path,
            use_trained_gpt=use_trained_gpt,
            auto_gsv_server=auto_gsv_server,
            gsv_server_command=gsv_server_command,
            only_segment_ids=target_ids,
        )
    elif backend == "mock":
        synth_step(
            project_dir,
            gsv_url,
            refs_path,
            mock=True,
            confirm_rights=confirm_rights,
            only_segment_ids=target_ids,
        )
    elif backend == "qwen":
        synth_qwen_step(
            project_dir,
            refs_path,
            confirm_rights=confirm_rights,
            model_id=qwen_model_id,
            candidate_count=qwen_candidate_count,
            promote=True,
            local_files_only=qwen_local_files_only,
            only_segment_ids=target_ids,
        )
    elif backend in {"fish", "fish-tts", "fish-speech", "cosyvoice", "cosy", "cosy-voice"}:
        synth_experimental_tts_step(
            project_dir,
            refs_path,
            backend=backend,
            confirm_rights=confirm_rights,
            base_url=experimental_tts_base_url,
            candidate_count=experimental_tts_candidate_count,
            promote=True,
            only_segment_ids=target_ids,
        )
    else:
        raise ValueError("tts_backend must be one of: gpt-sovits, qwen, fish, cosyvoice, mock")

    rvc_step(project_dir, confirm_rights=confirm_rights, only_segment_ids=target_ids)
    manifest = qc_step(
        project_dir,
        gemma_backend,
        confirm_rights=confirm_rights,
        only_segment_ids=target_ids,
    )
    remaining = [
        segment.id for segment in manifest.segments if segment.id in target_ids and segment.status == "needs_regeneration"
    ]
    mark_stage(
        manifest,
        "regenerate",
        "completed",
        tts_backend=backend,
        gemma_backend=gemma_backend,
        target_segments=sorted(target_ids),
        remaining_needs_regeneration=remaining,
        segment_counts=_segment_counts(manifest),
    )
    save_manifest(project_dir, manifest)
    _log_stage_complete(
        "regenerate",
        manifest,
        f"processed={len(target_ids)} remaining_needs_regeneration={len(remaining)}",
    )
    return manifest


def _mix_config_metadata(manifest: PipelineManifest) -> dict[str, Any]:
    cfg = manifest.project_config
    return {
        "profile": cfg.mix_profile,
        "sample_rate": cfg.mix_sample_rate,
        "background_bed": cfg.mix_background_bed,
        "background_gain_db": cfg.background_gain_db,
        "background_speech_suppression": cfg.background_speech_suppression,
        "background_speech_suppression_db": cfg.background_speech_suppression_db,
        "background_speech_suppression_pad_sec": cfg.background_speech_suppression_pad_sec,
        "background_speech_suppression_fade_ms": cfg.background_speech_suppression_fade_ms,
        "dialogue_gain_db": cfg.mix_dialogue_gain_db,
        "dialogue_fade_ms": cfg.mix_dialogue_fade_ms,
        "loudness_strategy": cfg.mix_loudness_strategy,
        "peak_limit_dbfs": cfg.mix_peak_limit_dbfs
        if cfg.mix_loudness_strategy == "peak_guard_only"
        else None,
        "loudness_normalization": "disabled",
    }


def _include_segment_in_mix(segment: Segment, *, allow_korean_timing_draft: bool) -> bool:
    selected_candidate = None
    if segment.tts:
        selected_candidate = next((candidate for candidate in segment.tts.candidates if candidate.selected), None)
    selected_candidate_ok = selected_candidate.acceptable_for_mix if selected_candidate else True
    if (
        segment.status == "ok"
        and segment.qc is not None
        and segment.qc.recommendation == "pass"
        and selected_candidate_ok
    ):
        return True
    if not allow_korean_timing_draft:
        return False
    if (
        segment.status != "needs_regeneration"
        or not segment.script
        or segment.script.tts_language != "ko"
        or not segment.tts
        or not segment.tts.selected_candidate_path
        or segment.qc is None
    ):
        return False
    if selected_candidate is not None and not selected_candidate.acceptable_for_mix:
        return False
    if (
        segment.qc.unsafe_or_rights_issue
        or segment.qc.repetition_detected
        or segment.qc.omission_detected
    ):
        return False
    return not (set(segment.qc.issues) - KOREAN_DRAFT_MIX_ALLOWED_QC_ISSUES)


def mix_step(project_dir: Path, confirm_rights: bool) -> PipelineManifest:
    manifest = load_manifest(project_dir)
    _load_config_into_manifest(project_dir, manifest)
    cfg = manifest.project_config
    total = len(manifest.segments)
    _log_stage_start("mix", f"segments={total}")
    source_path = Path(manifest.source_info.path) if manifest.source_info else None
    audit = require_confirmed_rights(confirm_rights, "mix", source_path)
    manifest.rights_audit = merge_rights_audit(manifest.rights_audit, audit)
    if manifest.stage_state.get("qc", {}).get("status") != "completed":
        raise ValueError("Mix requires a completed QC stage.")
    _require_rvc_ready_for_downstream(project_dir, manifest)
    allow_korean_timing_draft = (
        cfg.mix_allow_korean_timing_draft
        and _canonical_language(cfg.target_language) == "ko"
        and manifest.stage_state.get("korean-script", {}).get("status") == "completed"
    )
    duration = manifest.source_info.duration_sec if manifest.source_info else 0.0
    if duration <= 0 and manifest.segments:
        duration = max(segment.end for segment in manifest.segments)
    dialogue = ensure_inside_project(project_dir, project_dir / "work" / "mix" / "dialogue_stem.wav")
    final_audio = ensure_inside_project(project_dir, project_dir / "work" / "mix" / "final_audio.wav")
    peak_limit_dbfs = cfg.mix_peak_limit_dbfs if cfg.mix_loudness_strategy == "peak_guard_only" else None
    started_at = monotonic()
    last_logged_at = started_at

    def log_mix_progress(index: int, progress_total: int, segment: Segment) -> None:
        nonlocal last_logged_at
        last_logged_at = _log_segment_progress(
            "mix dialogue",
            index,
            progress_total,
            segment,
            manifest,
            started_at,
            last_logged_at,
        )

    console.print("[cyan]mix[/cyan] building dialogue stem")
    build_dialogue_stem(
        manifest.segments,
        dialogue,
        duration,
        cfg.mix_sample_rate,
        dialogue_gain_db=cfg.mix_dialogue_gain_db,
        dialogue_fade_ms=cfg.mix_dialogue_fade_ms,
        peak_limit_dbfs=peak_limit_dbfs,
        progress_callback=log_mix_progress,
        include_segment=lambda segment: _include_segment_in_mix(
            segment,
            allow_korean_timing_draft=allow_korean_timing_draft,
        ),
    )
    console.print(f"[cyan]mix[/cyan] dialogue stem written: {dialogue}")
    separated_background = manifest.artifacts.get("background_only_48k")
    original_background = manifest.artifacts.get("original_stereo_48k")
    background_source = None
    background_kind = None
    if cfg.mix_background_bed == "preserve_original":
        if separated_background:
            background_source = Path(separated_background)
            background_kind = "source_separated"
        elif original_background:
            background_source = Path(original_background)
            background_kind = "original"
    background_available = background_source is not None
    background = background_source
    background_suppressed_path: Path | None = None
    if background_source and cfg.background_speech_suppression:
        background_suppressed_path = ensure_inside_project(
            project_dir,
            project_dir / "work" / "mix" / "source_suppressed_background.wav",
        )
        console.print("[cyan]mix[/cyan] suppressing source speech in background bed")
        build_source_suppressed_background(
            background_source,
            background_suppressed_path,
            manifest.segments,
            sample_rate=cfg.mix_sample_rate,
            attenuation_db=cfg.background_speech_suppression_db,
            pad_sec=cfg.background_speech_suppression_pad_sec,
            fade_ms=cfg.background_speech_suppression_fade_ms,
            reduce_center_bleed=background_kind != "source_separated",
            peak_limit_dbfs=peak_limit_dbfs,
        )
        manifest.artifacts["source_suppressed_background"] = str(background_suppressed_path)
        background = background_suppressed_path
    console.print("[cyan]mix[/cyan] combining dialogue with background")
    mix_with_background(
        dialogue,
        final_audio,
        background,
        cfg.background_gain_db,
        cfg.mix_sample_rate,
        peak_limit_dbfs=peak_limit_dbfs,
        suppress_background_speech=False,
    )
    console.print(f"[cyan]mix[/cyan] final audio written: {final_audio}")
    manifest.artifacts["dialogue_stem"] = str(dialogue)
    manifest.artifacts["final_audio"] = str(final_audio)
    mix_config = _mix_config_metadata(manifest)
    background_metadata = {
        "available": background_available,
        "used": background is not None,
        "path": str(background) if background else None,
        "source_path": str(background_source) if background_source else None,
        "source_kind": background_kind,
        "policy": cfg.mix_background_bed,
        "gain_db": cfg.background_gain_db if background else None,
        "speech_suppression": {
            "enabled": bool(background_suppressed_path),
            "path": str(background_suppressed_path) if background_suppressed_path else None,
            "attenuation_db": cfg.background_speech_suppression_db
            if background_suppressed_path
            else None,
            "pad_sec": cfg.background_speech_suppression_pad_sec
            if background_suppressed_path
            else None,
            "fade_ms": cfg.background_speech_suppression_fade_ms
            if background_suppressed_path
            else None,
            "center_bleed_reduction": bool(
                background_suppressed_path and background_kind != "source_separated"
            ),
        },
    }
    skipped = [
        s.id
        for s in manifest.segments
        if not _include_segment_in_mix(s, allow_korean_timing_draft=allow_korean_timing_draft)
    ]
    draft_included = [
        s.id
        for s in manifest.segments
        if s.status != "ok"
        and _include_segment_in_mix(s, allow_korean_timing_draft=allow_korean_timing_draft)
    ]
    for segment in manifest.segments:
        included = _include_segment_in_mix(
            segment,
            allow_korean_timing_draft=allow_korean_timing_draft,
        )
        segment.mix = {
            **segment.mix,
            "included": included,
            "reason": "qc_pass"
            if included and segment.status == "ok"
            else "korean_timing_draft"
            if included
            else f"status_{segment.status}",
            "selected_candidate_path": segment.tts.selected_candidate_path if segment.tts else None,
            "rvc_output_path": segment.rvc.output_path if segment.rvc else None,
            "start": segment.start,
            "estimated_pan": segment.estimated_pan,
            "dialogue_gain_db": cfg.mix_dialogue_gain_db if included else None,
            "dialogue_fade_ms": cfg.mix_dialogue_fade_ms if included else None,
            "qc_recommendation": segment.qc.recommendation if segment.qc else None,
        }
    if draft_included:
        manifest.warnings.append(
            "Included Korean draft segments with timing-only QC regeneration flags during mix: "
            + ", ".join(draft_included)
        )
    if skipped:
        manifest.warnings.append(f"Skipped non-passing segments during mix: {', '.join(skipped)}")
    mix_manifest = project_dir / "work" / "mix" / "mix_manifest.json"
    write_json_atomic(
        mix_manifest,
        {
            "dialogue_stem": str(dialogue),
            "final_audio": str(final_audio),
            "config": mix_config,
            "background": background_metadata,
            "segments": [{"id": segment.id, "mix": segment.mix} for segment in manifest.segments],
        },
    )
    manifest.artifacts["mix_manifest"] = str(mix_manifest)
    mark_stage(
        manifest,
        "mix",
        "completed",
        skipped_segments=skipped,
        draft_included_segments=draft_included,
        mix_config=mix_config,
        background=background_metadata,
    )
    save_manifest(project_dir, manifest)
    _log_stage_complete("mix", manifest, f"skipped={len(skipped)}")
    return manifest


def export_step(input_path: Path, project_dir: Path, confirm_rights: bool) -> PipelineManifest:
    _log_stage_start("export", f"input={input_path}")
    manifest = load_manifest(project_dir)
    _load_config_into_manifest(project_dir, manifest)
    audit = require_confirmed_rights(confirm_rights, "export", input_path)
    manifest.rights_audit = merge_rights_audit(manifest.rights_audit, audit)
    _require_rvc_ready_for_downstream(project_dir, manifest)
    final_audio = Path(manifest.artifacts.get("final_audio", project_dir / "work/mix/final_audio.wav"))
    suffix = ".mp4" if manifest.source_info and manifest.source_info.has_video else ".wav"
    output = ensure_inside_project(project_dir, project_dir / "output" / f"{input_path.stem}_dub{suffix}")
    ensure_not_same_path(input_path, output)
    try:
        console.print(f"[cyan]export[/cyan] muxing output: {output}")
        ffmpeg.mux_audio(input_path, final_audio, output)
    except ffmpeg.FFmpegError:
        if suffix == ".wav":
            shutil.copy2(final_audio, output)
        else:
            raise
    manifest.artifacts["export"] = str(output)
    export_manifest = project_dir / "work" / "export" / "export_manifest.json"
    write_json_atomic(
        export_manifest,
        {
            "input": str(input_path),
            "final_audio": str(final_audio),
            "output": str(output),
            "has_video": bool(manifest.source_info and manifest.source_info.has_video),
        },
    )
    manifest.artifacts["export_manifest"] = str(export_manifest)
    mark_stage(manifest, "export", "completed", output=str(output), export_manifest=str(export_manifest))
    save_manifest(project_dir, manifest)
    _log_stage_complete("export", manifest, f"output={output}")
    return manifest
