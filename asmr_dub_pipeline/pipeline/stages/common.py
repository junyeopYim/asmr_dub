"""Shared imports and helpers for pipeline stage modules.

This file is mechanically split from pipeline.steps so stage entrypoints can live
in focused modules while preserving the existing helper API.
"""

# ruff: noqa: F401
from __future__ import annotations

import importlib.util
import json
import math
import re
import shlex
import shutil
import subprocess
from collections import Counter
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

from asmr_dub_pipeline.asr import (
    ASRChunk,
    ASRUnavailableError,
    create_asr_backend,
    map_chunks_to_segments,
)
from asmr_dub_pipeline.audio import ffmpeg
from asmr_dub_pipeline.audio.duration import duration_ratio, suggest_speed_factor
from asmr_dub_pipeline.audio.features import (
    clipping_ratio,
    duration_sec,
    load_audio,
    peak_dbfs,
    resample_linear,
    rms_dbfs,
    to_mono,
    trim_edge_silence,
    write_audio,
)
from asmr_dub_pipeline.audio.mixing import (
    build_dialogue_stem,
    build_source_suppressed_background,
    mix_with_background,
)
from asmr_dub_pipeline.audio.preprocess import (
    extract_project_audio,
    folder_input_metadata,
    merge_numbered_parts_to_audio,
    plan_folder_input,
    plan_numbered_part_merge,
    prepare_folder_input_audio,
    probe_with_fallback,
    translate_media_stem_to_korean,
)
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
from asmr_dub_pipeline.audio.training_filter import evaluate_voice_training_candidate
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
    RightsAudit,
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
    assign_source_speakers_to_manifest,
    assign_speakers_to_manifest,
    load_voice_bank,
    resolve_voice_bank_path,
    validate_voice_bank_models,
)

NO_SPEECH_STATUSES = {"no_speech_detected"}
SKIP_STATUSES = {"needs_manual_review", "failed", *NO_SPEECH_STATUSES}
GEMMA_TEXT_SERVER_UNAVAILABLE_MARKERS = (
    "Connection refused",
    "Connection reset",
    "Server disconnected",
    "All connection attempts failed",
)
LEGACY_DETERMINISTIC_NUMERIC_TRANSLATION_NOTE = "deterministic_numeric_source"
NUMERIC_COUNTING_POSTPROCESS_NOTE = "numeric_counting_postprocess"
KOREAN_DIGIT_PRONUNCIATION_POSTPROCESS_NOTE = "korean_digit_pronunciation_postprocess"
KOREAN_ORDINAL_POSTPROCESS_NOTE = "korean_ordinal_postprocess"
KOREAN_ASR_HOMOPHONE_POSTPROCESS_NOTE = "korean_asr_homophone_postprocess"
KOREAN_ONOMATOPOEIA_POSTPROCESS_NOTE = "korean_onomatopoeia_postprocess"
KOREAN_FLUENCY_POSTPROCESS_NOTE = "korean_fluency_postprocess"
NUMERIC_ONLY_SOURCE_RE = re.compile(r"^\s*\d+(?:[\s,]+\d+)*\s*$")
NUMERIC_TOKEN_RE = re.compile(r"\d+")
RAW_DIGIT_RE = re.compile(r"(?<![A-Za-z_])\d+(?![A-Za-z_])")
JAPANESE_SECOND_ORDINAL_RE = re.compile(r"第\s*(?:二|2|２)\s*の")
KOREAN_BAD_SECOND_ORDINAL_RE = re.compile(r"제\s*이의\s*")
KOREAN_TRANSLATION_KANA_RE = re.compile(r"[\u3040-\u30ffー]")
KOREAN_TRANSLATION_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
KOREAN_TRANSLATION_LATIN_RE = re.compile(r"[A-Za-z]")
KOREAN_GURIGURI_TRANSLITERATION_RE = re.compile(
    r"(?<![가-힣])(?:그리그리|그리(?:\s*,?\s*그리)+)(?![가-힣])"
)
JAPANESE_HIRAGANA_RE = re.compile(r"[\u3041-\u309fー]")
JAPANESE_KATAKANA_RE = re.compile(r"[\u30a0-\u30ffー]")
JAPANESE_KANJI_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
JAPANESE_STRONG_SPLIT_CHARS = set(" \t\n\r。、，,.！？!?…♪")
JAPANESE_SOFT_SPLIT_END_CHARS = set("よねわぞぜ")
JAPANESE_PARTICLE_SPLIT_START_CHARS = set("はがをにでともへや")
JAPANESE_SMALL_KANA_CHARS = set("ぁぃぅぇぉゃゅょっァィゥェォャュョッ")
JAPANESE_COMMON_SEGMENT_STARTS = (
    "だから",
    "だって",
    "でも",
    "では",
    "です",
    "そして",
    "それで",
    "それから",
    "それは",
    "それも",
    "ただ",
    "また",
    "まだ",
    "もう",
    "まず",
    "次に",
    "この",
    "その",
    "あの",
    "どこ",
    "ここ",
    "そこ",
    "ある",
    "いい",
    "はい",
    "ゆっくり",
)
SEVERE_TRANSLATION_BACKCHECK_KO_PATTERNS = ("변비약", "비약", "체감")
KOREAN_DRAFT_MIX_ALLOWED_QC_ISSUES = {"duration_ratio_out_of_range", "too_much_silence"}
PROGRESS_LOG_SECONDS = 30.0
TRANSLATION_REJECTION_ERROR_PREFIX = "Korean translation rejected before TTS: "
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
    split_points = _preferred_asr_text_split_points(text, part_count)
    start_indexes = [0, *split_points]
    end_indexes = [*split_points, len(text)]
    parts = [text[start:end].strip() for start, end in zip(start_indexes, end_indexes, strict=True)]
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


def _preferred_asr_text_split_points(text: str, part_count: int) -> list[int]:
    if part_count <= 1 or len(text) <= 1:
        return []
    split_points: list[int] = []
    previous = 0
    for split_index in range(1, part_count):
        target = round(len(text) * split_index / part_count)
        min_index = previous + 1
        max_index = len(text) - (part_count - split_index)
        if min_index > max_index:
            break
        point = _preferred_asr_text_split_point(
            text,
            target=target,
            min_index=min_index,
            max_index=max_index,
            part_count=part_count,
        )
        split_points.append(point)
        previous = point
    return split_points


def _preferred_asr_text_split_point(
    text: str,
    *,
    target: int,
    min_index: int,
    max_index: int,
    part_count: int,
) -> int:
    window = max(8, math.ceil(len(text) / part_count * 0.45))
    window_min = max(min_index, target - window)
    window_max = min(max_index, target + window)
    candidates = range(window_min, window_max + 1)
    scored = [
        (_asr_text_split_boundary_score(text, index), -abs(index - target), -index, index)
        for index in candidates
    ]
    preferred = [item for item in scored if item[0] > 0]
    if preferred:
        return max(preferred)[3]
    return min(max(target, min_index), max_index)


def _asr_text_split_boundary_score(text: str, index: int) -> int:
    if index <= 0 or index >= len(text):
        return 0
    left = text[index - 1]
    right = text[index]
    if left.isspace() or right.isspace():
        return 100
    if left in JAPANESE_STRONG_SPLIT_CHARS or right in JAPANESE_STRONG_SPLIT_CHARS:
        return 95
    if any(text[:index].endswith(suffix) for suffix in ("ください", "でした", "ました", "ません", "です", "ます")):
        return 85
    if left in JAPANESE_SOFT_SPLIT_END_CHARS:
        return 70
    if JAPANESE_KANJI_RE.fullmatch(left) and right in JAPANESE_PARTICLE_SPLIT_START_CHARS:
        return 60
    if _looks_like_japanese_word_internal_split(left, right):
        return 0
    return 0


def _looks_like_japanese_word_internal_split(left: str, right: str) -> bool:
    if JAPANESE_KANJI_RE.fullmatch(left) and JAPANESE_HIRAGANA_RE.fullmatch(right):
        return True
    if JAPANESE_HIRAGANA_RE.fullmatch(left) and JAPANESE_HIRAGANA_RE.fullmatch(right):
        return left in JAPANESE_SMALL_KANA_CHARS or right in JAPANESE_SMALL_KANA_CHARS
    if JAPANESE_KATAKANA_RE.fullmatch(left) and JAPANESE_KATAKANA_RE.fullmatch(right):
        return left in JAPANESE_SMALL_KANA_CHARS or right in JAPANESE_SMALL_KANA_CHARS or right == "ー"
    return False


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
    return bool(_asr_suspicious_pattern_hits(text, suspicious_text_patterns))


def _asr_suspicious_pattern_hits(
    text: str,
    suspicious_text_patterns: list[str] | tuple[str, ...],
) -> list[str]:
    hits: list[str] = []
    for pattern in suspicious_text_patterns:
        pattern = pattern.strip()
        if not pattern:
            continue
        try:
            if re.search(pattern, text):
                if pattern not in hits:
                    hits.append(pattern)
        except re.error:
            if pattern in text and pattern not in hits:
                hits.append(pattern)
    return hits


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
    normalized_chunks, summary = _apply_asr_text_replacements_to_chunks_with_summary(
        chunks,
        replacements,
    )
    return normalized_chunks, int(summary["chunks_changed"])


def _apply_text_replacements_with_hits(
    text: str,
    replacements: dict[str, str],
    *,
    max_passes: int = 3,
) -> tuple[str, list[dict[str, Any]]]:
    hits: list[dict[str, Any]] = []
    normalized = text
    for _ in range(max(1, max_passes)):
        changed = False
        for source, target in replacements.items():
            if not source:
                continue
            count = normalized.count(source)
            if count > 0:
                hits.append({"source": source, "target": target, "count": count})
                normalized = normalized.replace(source, target)
                changed = True
        if not changed:
            break
    return normalized, hits


def _replacement_hits(text: str, replacements: dict[str, str]) -> list[dict[str, Any]]:
    _, hits = _apply_text_replacements_with_hits(text, replacements)
    return hits


def _apply_asr_text_replacements_to_chunks_with_summary(
    chunks: list[ASRChunk],
    replacements: dict[str, str],
) -> tuple[list[ASRChunk], dict[str, Any]]:
    if not replacements:
        return chunks, {"chunks_changed": 0, "total_replacements": 0, "items": []}
    chunks_changed = 0
    total_replacements = 0
    items: list[dict[str, Any]] = []
    normalized_chunks: list[ASRChunk] = []
    for index, chunk in enumerate(chunks, start=1):
        text = chunk.text
        normalized, hits = _apply_text_replacements_with_hits(text, replacements)
        if normalized != text:
            chunks_changed += 1
            total_replacements += sum(int(hit["count"]) for hit in hits)
            items.append(
                {
                    "chunk_id": f"chunk_{index:04d}",
                    "start": round(chunk.start, 3),
                    "end": round(chunk.end, 3),
                    "original_text": text,
                    "replaced_text": normalized,
                    "hits": hits,
                }
            )
            normalized_chunks.append(chunk.model_copy(update={"text": normalized}))
        else:
            normalized_chunks.append(chunk)
    return normalized_chunks, {
        "chunks_changed": chunks_changed,
        "total_replacements": total_replacements,
        "items": items,
    }


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


ASR_HARD_PROMPT_LEAK_MARKERS = (
    "Sound Hodori",
    "사운드",
    "호돌이",
    "Instagram",
    "Twitter",
    "ご視聴",
    "ご覧いただきありがとうございます",
    "ご覧頂きありがとうございます",
    "最後までご覧",
    "次の動画",
    "お会いしましょう",
    "チャンネル登録",
    "高評価",
    "Serviceman",
    "iling",
    "İ",
    "さくら ジンジン 痺れる",
    "気持ちいい イっちゃう 飛んじゃってください",
)
ASR_SOFT_ENDING_MARKERS = (
    "ありがとうございました",
    "お疲れ様",
)
ASR_PROMPT_LEAK_MARKERS = ASR_HARD_PROMPT_LEAK_MARKERS
ASR_SOURCE_VOCALS_RELATIVE_QUIET_MAX_RMS_DBFS = -45.0
ASR_SOURCE_VOCALS_RELATIVE_QUIET_MIN_RMS_DELTA_DB = 24.0
ASR_SOURCE_VOCALS_RELATIVE_QUIET_MIN_PEAK_DELTA_DB = 6.0


def _asr_candidate_looks_prompt_leaked(text: str, cfg: Any) -> bool:
    normalized = " ".join(text.split())
    if not normalized:
        return False
    if any(marker in normalized for marker in ASR_HARD_PROMPT_LEAK_MARKERS):
        return True
    if normalized.count("おやすみ") >= 2:
        return True
    prompt = str(getattr(cfg, "asr_initial_prompt", "") or "").strip()
    if prompt and prompt in normalized:
        return True
    review_prompt = str(getattr(cfg, "asr_review_initial_prompt", "") or "").strip()
    if review_prompt and review_prompt in normalized:
        return True
    hotwords = str(getattr(cfg, "asr_hotwords", "") or "").strip()
    return bool(hotwords and hotwords in normalized)


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


def _asr_repair_candidate_options(cfg: Any) -> list[dict[str, Any]]:
    clean_options: dict[str, Any] = {
        "vad_filter": False,
        "vad_parameters": None,
        "condition_on_previous_text": False,
        "word_timestamps": False,
        "hallucination_silence_threshold": None,
        "initial_prompt": None,
        "hotwords": None,
    }
    vad_parameters = dict(getattr(cfg, "asr_vad_parameters", {}) or {})
    prompted_options: dict[str, Any] = {
        "vad_filter": bool(getattr(cfg, "asr_vad_filter", True)),
        "vad_parameters": vad_parameters or None,
        "condition_on_previous_text": False,
        "word_timestamps": bool(getattr(cfg, "asr_word_timestamps", False)),
        "hallucination_silence_threshold": getattr(cfg, "asr_hallucination_silence_threshold", None),
        "initial_prompt": str(getattr(cfg, "asr_initial_prompt", "") or "").strip() or None,
        "hotwords": str(getattr(cfg, "asr_hotwords", "") or "").strip() or None,
    }
    base_padding = float(getattr(cfg, "asr_repair_padding_sec", 1.0))
    hallucination_threshold = getattr(cfg, "asr_hallucination_silence_threshold", None) or 1.0
    return [
        {
            "candidate_id": "no_vad_clean",
            "padding_sec": base_padding,
            "overrides": clean_options,
        },
        {
            "candidate_id": "vad_no_hotwords",
            "padding_sec": base_padding,
            "overrides": {**prompted_options, "hotwords": None},
        },
        {
            "candidate_id": "vad_no_prompt",
            "padding_sec": base_padding,
            "overrides": {**prompted_options, "initial_prompt": None},
        },
        {
            "candidate_id": "wide_no_vad_clean",
            "padding_sec": max(base_padding * 2.0, base_padding + 0.8),
            "overrides": clean_options,
        },
        {
            "candidate_id": "word_ts_hallucination_guard",
            "padding_sec": base_padding,
            "overrides": {
                **clean_options,
                "word_timestamps": True,
                "hallucination_silence_threshold": hallucination_threshold,
            },
        },
    ]


def _asr_repair_candidate_score(
    original: ASRChunk,
    candidate_chunks: list[ASRChunk],
    *,
    cfg: Any,
    prompt_leaked: bool,
) -> tuple[bool, float, str]:
    candidate_text = _asr_candidate_text(candidate_chunks)
    original_text = original.text.strip()
    if not candidate_text:
        return False, -100.0, "empty_candidate"
    if prompt_leaked:
        return False, -90.0, "prompt_or_hallucination_leak"
    if _asr_candidate_looks_prompt_leaked(original_text, cfg) and not _asr_candidate_looks_prompt_leaked(candidate_text, cfg):
        leak_improvement = 2.0
    else:
        leak_improvement = 0.0
    accepted = _asr_repair_candidate_is_better(
        original,
        candidate_chunks,
        confidence_threshold=cfg.asr_repair_confidence_threshold,
        sparse_min_sec=cfg.asr_repair_sparse_min_sec,
        sparse_min_chars_per_sec=cfg.asr_repair_sparse_min_chars_per_sec,
        suspicious_text_patterns=cfg.asr_repair_suspicious_text_patterns,
    )
    if not accepted:
        return False, -10.0 + leak_improvement, "not_confidently_better_than_original"
    candidate_confidence = _asr_candidate_confidence(candidate_chunks)
    original_confidence = original.confidence
    confidence_delta = 0.0
    if candidate_confidence is not None and original_confidence is not None:
        confidence_delta = candidate_confidence - original_confidence
    original_suspicious = _asr_text_matches_suspicious_pattern(
        original_text,
        cfg.asr_repair_suspicious_text_patterns,
    )
    candidate_suspicious = _asr_text_matches_suspicious_pattern(
        candidate_text,
        cfg.asr_repair_suspicious_text_patterns,
    )
    suspicious_improvement = 2.0 if original_suspicious and not candidate_suspicious else 0.0
    original_density = _asr_text_density(original)
    candidate_duration = max(0.001, candidate_chunks[-1].end - candidate_chunks[0].start)
    candidate_density = len(candidate_text) / candidate_duration
    density_delta = min(2.0, max(-2.0, candidate_density - original_density))
    length_ratio = len(candidate_text) / max(1, len(original_text))
    length_score = 0.5 if 0.6 <= length_ratio <= 1.8 else -0.5
    score = (
        suspicious_improvement
        + leak_improvement
        + confidence_delta
        + density_delta * 0.25
        + length_score
    )
    return True, score, "accepted"


def _asr_repair_attempt_payload(
    *,
    candidate_id: str,
    clip_start: float,
    clip_end: float,
    candidate_chunks: list[ASRChunk],
    prompt_leaked: bool,
    accepted: bool,
    score: float,
    reason: str,
    error: str | None = None,
) -> dict[str, Any]:
    candidate_text = _asr_candidate_text(candidate_chunks)
    return {
        "candidate_id": candidate_id,
        "clip_start": round(clip_start, 3),
        "clip_end": round(clip_end, 3),
        "accepted": accepted,
        "score": round(score, 6),
        "reason": reason,
        "prompt_leaked": prompt_leaked,
        "candidate_text": candidate_text,
        "confidence": _asr_candidate_confidence(candidate_chunks),
        "duration": round(
            max(0.0, candidate_chunks[-1].end - candidate_chunks[0].start) if candidate_chunks else 0.0,
            3,
        ),
        "text_density": round(
            len(candidate_text)
            / max(0.001, candidate_chunks[-1].end - candidate_chunks[0].start)
            if candidate_chunks
            else 0.0,
            6,
        ),
        "error": error,
    }


def _split_asr_chunks_for_repair(
    chunks: list[ASRChunk],
    *,
    audio_duration_sec: float,
    max_chunk_sec: float,
) -> list[ASRChunk]:
    repair_chunks: list[ASRChunk] = []
    for chunk in sorted(chunks, key=lambda item: (item.start, item.end)):
        text = chunk.text.strip()
        if not text:
            continue
        start = max(0.0, float(chunk.start))
        end = min(float(audio_duration_sec), max(start, float(chunk.end)))
        if end - start <= 0.05:
            continue
        normalized = chunk.model_copy(update={"start": start, "end": end, "text": text})
        repair_chunks.extend(_split_long_asr_chunk(normalized, max_chunk_sec=max_chunk_sec))
    return repair_chunks


def _repair_asr_chunks(
    chunks: list[ASRChunk],
    *,
    backend: Any,
    project_dir: Path,
    repair_audio_path: Path,
    audio_duration_sec: float,
    cfg: Any,
    qwen_fallback_backend: Any | None = None,
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
    repair_group_max_sec = max(1.0, float(getattr(cfg, "asr_resegment_max_sec", 20.0)))
    repair_input_chunks = _split_asr_chunks_for_repair(
        chunks,
        audio_duration_sec=audio_duration_sec,
        max_chunk_sec=repair_group_max_sec,
    )
    for chunk in repair_input_chunks:
        needs_repair = _asr_chunk_needs_repair(
            chunk,
            confidence_threshold=cfg.asr_repair_confidence_threshold,
            sparse_min_sec=cfg.asr_repair_sparse_min_sec,
            sparse_min_chars_per_sec=cfg.asr_repair_sparse_min_chars_per_sec,
            suspicious_text_patterns=cfg.asr_repair_suspicious_text_patterns,
        )
        if needs_repair:
            would_exceed_max = (
                bool(current_repair_group)
                and chunk.end - current_repair_group[0].start > repair_group_max_sec
            )
            gap_too_large = (
                bool(current_repair_group)
                and chunk.start - current_repair_group[-1].end > repair_group_gap_sec
            )
            if current_repair_group and (gap_too_large or would_exceed_max):
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
        original = ASRChunk(
            start=group_start,
            end=group_end,
            text=_asr_candidate_text(group),
            language=group[0].language,
            confidence=_asr_candidate_confidence(group),
        )
        option_specs = _asr_repair_candidate_options(cfg)
        if qwen_fallback_backend is not None:
            option_specs.append(
                {
                    "candidate_id": "qwen_asr_fallback",
                    "padding_sec": float(getattr(cfg, "asr_repair_padding_sec", 1.0)),
                    "overrides": {},
                    "backend": qwen_fallback_backend,
                }
            )
        attempts: list[dict[str, Any]] = []
        best_candidate: list[ASRChunk] = []
        best_candidate_id: str | None = None
        best_score = -math.inf
        for option in option_specs:
            candidate_id = str(option["candidate_id"])
            padding_sec = float(option["padding_sec"])
            clip_start = max(0.0, group_start - padding_sec)
            clip_end = min(audio_duration_sec, group_end + padding_sec)
            clip_path = repair_dir / f"repair_{attempted:04d}_{candidate_id}.wav"
            try:
                ffmpeg.slice_audio(
                    repair_audio_path,
                    clip_start,
                    clip_end,
                    clip_path,
                    sample_rate=cfg.gemma_sample_rate,
                    channels=1,
                )
                candidate_backend = option.get("backend", backend)
                candidate_local = _transcribe_with_backend_options(
                    candidate_backend,
                    clip_path,
                    [],
                    **dict(option.get("overrides") or {}),
                )
                candidate_text = _asr_candidate_text(candidate_local)
                prompt_leaked = _asr_candidate_looks_prompt_leaked(candidate_text, cfg)
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
                accepted, score, reason = _asr_repair_candidate_score(
                    original,
                    candidate_abs,
                    cfg=cfg,
                    prompt_leaked=prompt_leaked,
                )
                attempts.append(
                    _asr_repair_attempt_payload(
                        candidate_id=candidate_id,
                        clip_start=clip_start,
                        clip_end=clip_end,
                        candidate_chunks=candidate_abs,
                        prompt_leaked=prompt_leaked,
                        accepted=accepted,
                        score=score,
                        reason=reason,
                    )
                )
                if accepted and score > best_score:
                    best_score = score
                    best_candidate = candidate_abs
                    best_candidate_id = candidate_id
                    if score >= 0.5:
                        break
            except Exception as exc:
                attempts.append(
                    _asr_repair_attempt_payload(
                        candidate_id=candidate_id,
                        clip_start=clip_start,
                        clip_end=clip_end,
                        candidate_chunks=[],
                        prompt_leaked=False,
                        accepted=False,
                        score=-100.0,
                        reason="transcribe_failed",
                        error=str(exc),
                    )
                )
        accepted = bool(best_candidate)
        summary["items"].append(
            {
                "start": round(group_start, 3),
                "end": round(group_end, 3),
                "accepted": accepted,
                "accepted_candidate_id": best_candidate_id,
                "prompt_leaked": any(bool(attempt.get("prompt_leaked")) for attempt in attempts),
                "original_text": original.text,
                "candidate_text": _asr_candidate_text(best_candidate),
                "attempts": attempts,
            }
        )
        if accepted:
            repaired_chunks.extend(best_candidate)
            summary["repaired"] += 1
        else:
            repaired_chunks.extend(group)

    summary["attempted"] = attempted
    return sorted(repaired_chunks, key=lambda item: (item.start, item.end)), summary


ASR_REVIEW_GLOSSARY = [
    "絶頂",
    "媚薬",
    "耳舐め",
    "耳なめ",
    "暗示",
    "快感",
    "10数える",
]


def _asr_text_with_replacements(text: str, replacements: dict[str, str]) -> str:
    normalized, _ = _apply_text_replacements_with_hits(text, replacements)
    return normalized


def _asr_review_context(chunks: list[ASRChunk], index: int, radius: int, *, before: bool) -> list[dict[str, Any]]:
    if radius <= 0:
        return []
    if before:
        selected = chunks[max(0, index - radius) : index]
    else:
        selected = chunks[index + 1 : index + 1 + radius]
    return [
        {
            "start": round(chunk.start, 3),
            "end": round(chunk.end, 3),
            "text": chunk.text,
        }
        for chunk in selected
        if chunk.text.strip()
    ]


def _asr_review_item(
    chunks: list[ASRChunk],
    index: int,
    *,
    cfg: Any,
) -> dict[str, Any] | None:
    chunk = chunks[index]
    text = chunk.text.strip()
    if not text or NUMERIC_ONLY_SOURCE_RE.fullmatch(text):
        return None
    suspicious_patterns = [
        pattern
        for pattern in cfg.asr_review_suspicious_text_patterns
        if pattern and pattern in text
    ]
    candidate_text = _asr_text_with_replacements(
        text,
        cfg.asr_review_candidate_replacements,
    ).strip()
    low_confidence = (
        chunk.confidence is not None
        and chunk.confidence < cfg.asr_review_confidence_threshold
    )
    if not suspicious_patterns and candidate_text == text and not low_confidence:
        return None

    candidates: list[dict[str, str]] = [{"candidate_id": "original", "text": text}]
    if candidate_text and candidate_text != text:
        candidates.append({"candidate_id": "domain_replacement", "text": candidate_text})
    return {
        "chunk_id": f"chunk_{index + 1:04d}",
        "chunk_index": index,
        "start": round(chunk.start, 3),
        "end": round(chunk.end, 3),
        "duration": round(max(0.0, chunk.end - chunk.start), 3),
        "confidence": chunk.confidence,
        "suspicious_patterns": suspicious_patterns,
        "context_before": _asr_review_context(
            chunks,
            index,
            cfg.asr_review_context_radius,
            before=True,
        ),
        "context_after": _asr_review_context(
            chunks,
            index,
            cfg.asr_review_context_radius,
            before=False,
        ),
        "glossary": ASR_REVIEW_GLOSSARY,
        "candidates": candidates,
    }


def _append_unique_asr_review_candidate(
    item: dict[str, Any],
    *,
    candidate_id: str,
    text: str,
) -> bool:
    normalized_text = text.strip()
    if not normalized_text:
        return False
    candidates = item.setdefault("candidates", [])
    if any(
        isinstance(candidate, dict) and str(candidate.get("text", "")).strip() == normalized_text
        for candidate in candidates
    ):
        return False
    candidates.append({"candidate_id": candidate_id, "text": normalized_text})
    return True


def _generated_asr_review_candidate_options(cfg: Any) -> list[tuple[str, float, dict[str, Any]]]:
    padding_values = [
        max(0.0, float(value))
        for value in getattr(cfg, "asr_review_candidate_padding_sec", [])
    ]
    if not padding_values:
        padding_values = [0.8]
    prompt = str(getattr(cfg, "asr_review_initial_prompt", "") or "").strip()
    options: list[tuple[str, float, dict[str, Any]]] = []
    for index, padding_sec in enumerate(padding_values, start=1):
        base_options: dict[str, Any] = {
            "vad_filter": False,
            "vad_parameters": None,
            "condition_on_previous_text": False,
            "word_timestamps": False,
            "hallucination_silence_threshold": None,
            "initial_prompt": None,
            "hotwords": None,
        }
        options.append((f"asr_no_vad_pad_{index}", padding_sec, base_options))
        if prompt:
            options.append(
                (
                    f"asr_prompted_pad_{index}",
                    padding_sec,
                    {**base_options, "initial_prompt": prompt},
                )
            )
    return options


def _add_generated_asr_review_candidates(
    review_items: list[dict[str, Any]],
    *,
    backend: Any,
    project_dir: Path,
    review_audio_path: Path,
    audio_duration_sec: float,
    cfg: Any,
) -> int:
    if not cfg.asr_review_generate_candidates:
        return 0
    if not callable(getattr(backend, "transcribe_with_options", None)):
        return 0
    if not review_audio_path.exists():
        return 0

    generated = 0
    review_dir = project_dir / "work" / "transcribe" / "asr_review_clips"
    for item in review_items:
        chunk_id = str(item.get("chunk_id") or "chunk")
        start = float(item["start"])
        end = float(item["end"])
        for option_id, padding_sec, options in _generated_asr_review_candidate_options(cfg):
            clip_start = max(0.0, start - padding_sec)
            clip_end = min(audio_duration_sec, end + padding_sec)
            if clip_end <= clip_start:
                continue
            clip_path = review_dir / f"{chunk_id}_{option_id}.wav"
            ffmpeg.slice_audio(
                review_audio_path,
                clip_start,
                clip_end,
                clip_path,
                sample_rate=cfg.gemma_sample_rate,
                channels=1,
            )
            try:
                candidate_chunks = _transcribe_with_backend_options(backend, clip_path, [], **options)
            except Exception:
                continue
            candidate_text = _asr_candidate_text(candidate_chunks)
            if _asr_candidate_looks_prompt_leaked(candidate_text, cfg):
                continue
            if _append_unique_asr_review_candidate(
                item,
                candidate_id=option_id,
                text=candidate_text,
            ):
                generated += 1
            replaced_text = _asr_text_with_replacements(
                candidate_text,
                cfg.asr_review_candidate_replacements,
            )
            if replaced_text != candidate_text and _append_unique_asr_review_candidate(
                item,
                candidate_id=f"{option_id}_domain_replacement",
                text=replaced_text,
            ):
                generated += 1
    return generated


def _attach_asr_review_audio_clips(
    review_items: list[dict[str, Any]],
    *,
    project_dir: Path,
    review_audio_path: Path,
    audio_duration_sec: float,
    cfg: Any,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "enabled": True,
        "created": 0,
        "padding_sec": float(getattr(cfg, "asr_review_audio_padding_sec", 0.4)),
        "error": None,
        "items": [],
    }
    if not review_audio_path.exists():
        summary["error"] = f"review audio not found: {review_audio_path}"
        return summary

    review_dir = project_dir / "work" / "transcribe" / "asr_review_audio_clips"
    padding_sec = max(0.0, float(summary["padding_sec"]))
    for item in review_items:
        chunk_id = str(item.get("chunk_id") or "chunk")
        start = float(item["start"])
        end = float(item["end"])
        clip_start = max(0.0, start - padding_sec)
        clip_end = min(audio_duration_sec, end + padding_sec)
        if clip_end <= clip_start:
            summary["items"].append(
                {
                    "chunk_id": chunk_id,
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "created": False,
                    "error": "empty_audio_window",
                }
            )
            continue
        clip_path = review_dir / f"{chunk_id}.wav"
        try:
            ffmpeg.slice_audio(
                review_audio_path,
                clip_start,
                clip_end,
                clip_path,
                sample_rate=cfg.gemma_sample_rate,
                channels=1,
            )
        except Exception as exc:
            summary["items"].append(
                {
                    "chunk_id": chunk_id,
                    "start": round(clip_start, 3),
                    "end": round(clip_end, 3),
                    "created": False,
                    "error": str(exc),
                }
            )
            continue
        item["audio_clip_path"] = str(clip_path)
        item["audio_clip"] = {
            "path": str(clip_path),
            "start": round(clip_start, 3),
            "end": round(clip_end, 3),
            "duration": round(clip_end - clip_start, 3),
            "padding_sec": padding_sec,
        }
        summary["created"] += 1
        summary["items"].append({"chunk_id": chunk_id, **item["audio_clip"], "created": True})
    return summary


def _review_asr_chunks_with_model(
    chunks: list[ASRChunk],
    *,
    backend: Any,
    project_dir: Path,
    review_audio_path: Path,
    audio_duration_sec: float,
    cfg: Any,
) -> tuple[list[ASRChunk], dict[str, Any]]:
    summary: dict[str, Any] = {
        "enabled": bool(cfg.asr_review_enabled),
        "backend": cfg.asr_review_backend,
        "attempted": 0,
        "reviewed": 0,
        "replaced": 0,
        "manual_review": 0,
        "skipped": 0,
        "generated_candidates": 0,
        "audio_input": {"enabled": False, "created": 0, "error": None, "items": []},
        "error": None,
        "items": [],
    }
    if not cfg.asr_review_enabled or not chunks:
        return chunks, summary

    review_items = [
        item
        for index in range(len(chunks))
        if (item := _asr_review_item(chunks, index, cfg=cfg)) is not None
    ]
    if cfg.asr_review_max_chunks >= 0 and len(review_items) > cfg.asr_review_max_chunks:
        summary["skipped"] = len(review_items) - cfg.asr_review_max_chunks
        review_items = review_items[: cfg.asr_review_max_chunks]
    if not review_items:
        return chunks, summary

    summary["generated_candidates"] = _add_generated_asr_review_candidates(
        review_items,
        backend=backend,
        project_dir=project_dir,
        review_audio_path=review_audio_path,
        audio_duration_sec=audio_duration_sec,
        cfg=cfg,
    )

    backend_kind = cfg.asr_review_backend.replace("-", "_")
    if backend_kind not in {"llama_server_audio", "mock"}:
        summary["error"] = f"unsupported backend: {cfg.asr_review_backend}"
        return chunks, summary
    audio_review_enabled = backend_kind == "llama_server_audio"
    if audio_review_enabled:
        summary["audio_input"] = _attach_asr_review_audio_clips(
            review_items,
            project_dir=project_dir,
            review_audio_path=review_audio_path,
            audio_duration_sec=audio_duration_sec,
            cfg=cfg,
        )
        if summary["audio_input"].get("error"):
            summary["error"] = str(summary["audio_input"]["error"])
            return chunks, summary
        missing_audio = [
            str(item.get("chunk_id"))
            for item in review_items
            if not str(item.get("audio_clip_path") or "").strip()
        ]
        if missing_audio:
            summary["error"] = "ASR audio review could not create audio clips for: " + ", ".join(
                missing_audio[:5]
            )
            return chunks, summary

    review_by_chunk_id = {str(item["chunk_id"]): item for item in review_items}
    selected_text_by_chunk_id: dict[str, str] = {}
    review_results: dict[str, dict[str, Any]] = {}
    model_name = (
        cfg.gemma_llama_cpp_audio_model_path
        if backend_kind == "llama_server_audio"
        else "mock"
    )
    base_url = cfg.gemma_text_server_url.rstrip("/")

    def create_review_client() -> Any:
        if backend_kind == "llama_server_audio":
            return LlamaServerTranslationClient(
                base_url,
                timeout_sec=cfg.gemma_text_timeout_sec,
                retries=cfg.gemma_text_retries,
                n_predict=cfg.gemma_text_n_predict,
                model=model_name,
                two_pass=False,
            )
        return MockTranslationClient(model=model_name)

    server_manager = None
    if backend_kind == "llama_server_audio":
        server_manager = ManagedGemmaTextServer(
            enabled=cfg.gemma_text_server_auto_start,
            base_url=base_url,
            command=(
                _gemma_text_server_command(
                    cfg,
                    base_url=base_url,
                    lane_index=0,
                    include_mmproj=True,
                )
                if cfg.gemma_text_server_auto_start
                else []
            ),
            log_path=project_dir / "work" / "transcribe" / "asr_review_llama_server.log",
            startup_timeout_sec=cfg.gemma_text_server_startup_timeout_sec,
            shutdown_timeout_sec=cfg.gemma_text_server_shutdown_timeout_sec,
        )

    try:
        if server_manager is not None:
            server_manager.start()
        client = create_review_client()
        review_batches = ([item] for item in review_items) if audio_review_enabled else _chunked(
            review_items,
            cfg.asr_review_batch_size,
        )
        for batch_index, batch in enumerate(review_batches, start=1):
            batch_id = f"asr_review_{batch_index:04d}"
            if audio_review_enabled:
                audio_path = Path(str(batch[0]["audio_clip_path"]))
                review_results.update(
                    client.review_asr_candidates_with_audio(batch, batch_id, audio_path)
                )
            else:
                review_results.update(client.review_asr_candidates_for_mock(batch, batch_id))
    except Exception as exc:
        summary["error"] = str(exc)
        return chunks, summary
    finally:
        if server_manager is not None:
            server_manager.stop()

    summary["attempted"] = len(review_items)
    reviewed_chunks = [chunk.model_copy() for chunk in chunks]
    for chunk_id, review in review_results.items():
        item = review_by_chunk_id.get(chunk_id)
        if item is None:
            continue
        candidate_by_id = {
            str(candidate["candidate_id"]): str(candidate["text"])
            for candidate in item.get("candidates", [])
            if isinstance(candidate, dict)
        }
        selected_id = str(review.get("selected_candidate_id") or "")
        selected_text = candidate_by_id.get(selected_id)
        confidence = review.get("confidence")
        accepted = (
            review.get("decision") == "replace"
            and selected_id != "original"
            and selected_text is not None
            and confidence is not None
            and confidence >= cfg.asr_review_confidence_threshold
        )
        if review.get("decision") == "manual_review":
            summary["manual_review"] += 1
        if accepted:
            selected_text_by_chunk_id[chunk_id] = selected_text
            summary["replaced"] += 1
        summary["items"].append(
            {
                "chunk_id": chunk_id,
                "start": item["start"],
                "end": item["end"],
                "accepted": accepted,
                "decision": review.get("decision"),
                "selected_candidate_id": selected_id,
                "confidence": confidence,
                "original_text": candidate_by_id.get("original", ""),
                "selected_text": selected_text,
                "candidates": list(item.get("candidates", [])),
                "audio_clip": item.get("audio_clip"),
                "heard_text": review.get("heard_text"),
                "reason": review.get("reason"),
                "risk_terms": review.get("risk_terms", []),
                "suspicious_patterns": item.get("suspicious_patterns", []),
            }
        )

    for index, chunk in enumerate(reviewed_chunks):
        chunk_id = f"chunk_{index + 1:04d}"
        selected_text = selected_text_by_chunk_id.get(chunk_id)
        if selected_text is not None:
            reviewed_chunks[index] = chunk.model_copy(update={"text": selected_text})

    summary["reviewed"] = len(review_results)
    return reviewed_chunks, summary


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


def _sino_korean_number(value: int) -> str:
    if value == 0:
        return "영"
    digits = ["", "일", "이", "삼", "사", "오", "육", "칠", "팔", "구"]
    units = [(10000, "만"), (1000, "천"), (100, "백"), (10, "십")]
    remaining = value
    parts: list[str] = []
    for unit_value, unit_text in units:
        count, remaining = divmod(remaining, unit_value)
        if count == 0:
            continue
        prefix = "" if count == 1 and unit_value < 10000 else _sino_korean_number(count)
        parts.append(prefix + unit_text)
    if remaining:
        parts.append(digits[remaining])
    return "".join(parts)


def _spell_korean_numbers_for_tts(text: str) -> str:
    def replace_match(match: re.Match[str]) -> str:
        token = match.group(0)
        if len(token) > 5:
            return " ".join(_sino_korean_number(int(char)) for char in token)
        return _sino_korean_number(int(token))

    rewritten = RAW_DIGIT_RE.sub(replace_match, text)
    rewritten = rewritten.replace("%", "퍼센트")
    rewritten = re.sub(r"\s+", " ", rewritten)
    rewritten = re.sub(r"\s+([,.!?…])", r"\1", rewritten)
    return rewritten.strip()


def _translation_has_raw_digit(translation: Any) -> bool:
    return bool(
        translation
        and (
            RAW_DIGIT_RE.search(str(getattr(translation, "ko_natural", "")))
            or RAW_DIGIT_RE.search(str(getattr(translation, "ko_literal", "")))
        )
    )


def _strip_translation_test_placeholders(text: str) -> str:
    return re.sub(r"(?:seg|batch)_\d+", "", text)


def _source_suggests_numeric_translation_repair(segment: Segment) -> bool:
    source = _strip_translation_test_placeholders(_translation_source_text(segment))
    return bool(
        NUMERIC_TOKEN_RE.search(source)
        or "%" in source
        or "％" in source
        or "レベル" in source
        or "数え" in source
    )


def _apply_korean_digit_pronunciation_postprocess(
    segments: list[Segment],
    diagnostics: list[dict[str, Any]],
    quality_counters: Counter[str],
) -> int:
    rewritten_count = 0
    for segment in segments:
        translation = segment.translation_ko
        if (
            translation is None
            or not _source_suggests_numeric_translation_repair(segment)
            or not _translation_has_raw_digit(translation)
        ):
            continue
        before = translation.model_dump(mode="json")
        notes = list(translation.notes)
        if KOREAN_DIGIT_PRONUNCIATION_POSTPROCESS_NOTE not in notes:
            notes.append(KOREAN_DIGIT_PRONUNCIATION_POSTPROCESS_NOTE)
        repaired = translation.model_copy(
            update={
                "ko_literal": _spell_korean_numbers_for_tts(translation.ko_literal),
                "ko_natural": _spell_korean_numbers_for_tts(translation.ko_natural),
                "notes": notes,
            }
        )
        segment.translation_ko = repaired
        quality_counters["raw_digit"] += 1
        diagnostics.append(
            {
                "segment_id": segment.id,
                "source_text": _translation_source_text(segment),
                "repair_reasons": ["raw_digit"],
                "before": before,
                "after": repaired.model_dump(mode="json"),
                "accepted": not _translation_has_raw_digit(repaired),
            }
        )
        rewritten_count += 1
    return rewritten_count


def _repair_second_ordinal_translation_text(text: str) -> str:
    rewritten = KOREAN_BAD_SECOND_ORDINAL_RE.sub("두 번째 ", text)
    rewritten = re.sub(r"\s+", " ", rewritten)
    rewritten = re.sub(r"\s+([,.!?…])", r"\1", rewritten)
    return rewritten.strip()


def _apply_korean_ordinal_postprocess(
    segments: list[Segment],
    diagnostics: list[dict[str, Any]],
    quality_counters: Counter[str],
) -> int:
    rewritten_count = 0
    for segment in segments:
        translation = segment.translation_ko
        if translation is None or KOREAN_ORDINAL_POSTPROCESS_NOTE in translation.notes:
            continue
        source_text = _translation_source_text(segment)
        if not JAPANESE_SECOND_ORDINAL_RE.search(source_text):
            continue
        combined = f"{translation.ko_literal} {translation.ko_natural}"
        if not KOREAN_BAD_SECOND_ORDINAL_RE.search(combined):
            continue
        before = translation.model_dump(mode="json")
        notes = list(translation.notes)
        notes.append(KOREAN_ORDINAL_POSTPROCESS_NOTE)
        repaired = translation.model_copy(
            update={
                "ko_literal": _repair_second_ordinal_translation_text(translation.ko_literal),
                "ko_natural": _repair_second_ordinal_translation_text(translation.ko_natural),
                "notes": notes,
            }
        )
        segment.translation_ko = repaired
        quality_counters["ordinal_mistranslation_repaired"] += 1
        diagnostics.append(
            {
                "segment_id": segment.id,
                "source_text": source_text,
                "repair_reasons": ["ordinal_mistranslation"],
                "before": before,
                "after": repaired.model_dump(mode="json"),
                "accepted": not KOREAN_BAD_SECOND_ORDINAL_RE.search(
                    f"{repaired.ko_literal} {repaired.ko_natural}"
                ),
            }
        )
        rewritten_count += 1
    return rewritten_count


def _source_suggests_akume_asr_homophone(segment: Segment) -> bool:
    source_text = _translation_source_text(segment)
    compact = re.sub(r"\s+", "", source_text)
    homophone_tokens = ("悪夢", "悪目", "明け目", "アカメ")
    if not any(token in compact for token in homophone_tokens):
        return False
    if any(token in compact for token in ("悪夢落ち", "悪目落ち", "明け目落ち")):
        return True
    return any(term in compact for term in ("メスイキ", "イキ", "絶頂", "快楽", "快感", "気持ちいい", "イク", "行く"))


def _source_suggests_josou_asr_homophone(segment: Segment) -> bool:
    return "助走" in re.sub(r"\s+", "", _translation_source_text(segment))


def _source_suggests_guriguri_onomatopoeia(segment: Segment) -> bool:
    source_text = _translation_source_text(segment)
    compact = re.sub(r"\s+", "", source_text)
    return "グリグリ" in compact or bool(re.search(r"(^|[\s、。,.])グリ([\s、。,.]|$)", source_text))


def _repair_akume_homophone_translation_text(text: str) -> str:
    rewritten = text
    replacements = (
        ("암컷 절정 악몽이", "암컷 절정이"),
        ("암컷 절정 악몽을", "암컷 절정을"),
        ("암컷 절정 악몽은", "암컷 절정은"),
        ("암컷 절정 악몽", "암컷 절정"),
        ("암컷 절정의 악몽이", "암컷 절정이"),
        ("암컷 절정의 악몽을", "암컷 절정을"),
        ("암컷 절정의 악몽은", "암컷 절정은"),
        ("암컷 절정의 악몽", "암컷 절정"),
        ("메스이키 악몽이", "메스이키 절정이"),
        ("메스이키 악몽을", "메스이키 절정을"),
        ("메스이키 악몽은", "메스이키 절정은"),
        ("메스이키 악몽", "메스이키 절정"),
        ("메스 이키 악몽이", "메스이키 절정이"),
        ("메스 이키 악몽을", "메스이키 절정을"),
        ("메스 이키 악몽은", "메스이키 절정은"),
        ("메스 이키 악몽", "메스이키 절정"),
        ("악몽으로 떨어지는 길", "절정에 빠지는 길"),
        ("악몽으로 떨어지는", "절정에 빠지는"),
        ("악몽에 빠지는 길", "절정에 빠지는 길"),
        ("악몽에 빠지는", "절정에 빠지는"),
        ("눈이 뒤집히는 타락의 길", "절정에 빠지는 길"),
        ("눈이 뒤집히는 타락", "절정"),
        ("악몽을 꾸세요", "절정하세요"),
        ("악몽을 꾸고", "절정하고"),
        ("악몽을 꾸면서", "절정하면서"),
        ("악몽을 겪어보고", "아크메를 해보고"),
        ("악몽을 겪고", "아크메를 하고"),
        ("악몽을 결정", "아크메를 결정"),
        ("악몽을 너무 결정해서", "아크메를 너무 해서"),
        ("악몽 추락", "아크메락"),
        ("악몽의 추락", "아크메락"),
        ("악몽 속으로 떨어진 순간", "아크메락 순간"),
        ("악몽 속으로 추락하고", "아크메락하고"),
        ("악몽 여배우", "아크메 여배우"),
        ("악몽 배우", "아크메 배우"),
        ("악몽을", "아크메를"),
        ("악몽이", "아크메가"),
        ("악몽은", "아크메는"),
        ("악몽으로", "아크메로"),
        ("악몽에", "아크메에"),
        ("악몽", "아크메"),
    )
    for source, target in replacements:
        rewritten = rewritten.replace(source, target)
    rewritten = re.sub(r"\s+", " ", rewritten)
    rewritten = re.sub(r"\s+([,.!?…])", r"\1", rewritten)
    return rewritten.strip()


def _repair_josou_homophone_translation_text(text: str) -> str:
    rewritten = text
    replacements = (
        ("도움닫기를", "여장을"),
        ("도움닫기라는", "여장이라는"),
        ("도움닫기였", "여장이었"),
        ("도움닫기예요", "여장이에요"),
        ("도움닫기", "여장"),
        ("달려들고 싶어지면", "여장하고 싶어지면"),
        ("달려들고 싶어질", "여장하고 싶어질"),
        ("달려들고 싶어", "여장하고 싶어"),
        ("조주를", "여장을"),
        ("조주라는", "여장이라는"),
        ("조주였", "여장이었"),
        ("조주예요", "여장이에요"),
        ("조주", "여장"),
    )
    for source, target in replacements:
        rewritten = rewritten.replace(source, target)
    rewritten = re.sub(r"\s+", " ", rewritten)
    rewritten = re.sub(r"\s+([,.!?…])", r"\1", rewritten)
    return rewritten.strip()


def _repair_guriguri_onomatopoeia_translation_text(text: str) -> str:
    rewritten = KOREAN_GURIGURI_TRANSLITERATION_RE.sub("문질문질", text)
    rewritten = re.sub(r"\s+", " ", rewritten)
    rewritten = re.sub(r"\s+([,.!?…])", r"\1", rewritten)
    return rewritten.strip()


def _repair_korean_fluency_translation_text(text: str) -> tuple[str, list[str]]:
    rewritten = text
    repair_reasons: list[str] = []
    if "끝져가는" in rewritten:
        rewritten = rewritten.replace("끝져가는", "끝나가는")
        repair_reasons.append("broken_korean_ending")
    rewritten = re.sub(r"\s+", " ", rewritten)
    rewritten = re.sub(r"\s+([,.!?…])", r"\1", rewritten)
    return rewritten.strip(), repair_reasons


def _apply_korean_asr_homophone_postprocess(
    segments: list[Segment],
    diagnostics: list[dict[str, Any]],
    quality_counters: Counter[str],
) -> int:
    rewritten_count = 0
    for segment in segments:
        translation = segment.translation_ko
        if translation is None:
            continue
        source_text = _translation_source_text(segment)
        repair_reasons: list[str] = []
        ko_literal = translation.ko_literal
        ko_natural = translation.ko_natural
        if _source_suggests_akume_asr_homophone(segment):
            ko_literal = _repair_akume_homophone_translation_text(ko_literal)
            ko_natural = _repair_akume_homophone_translation_text(ko_natural)
            if ko_literal != translation.ko_literal or ko_natural != translation.ko_natural:
                repair_reasons.append("asr_homophone_akume")
        if _source_suggests_josou_asr_homophone(segment):
            before_literal = ko_literal
            before_natural = ko_natural
            ko_literal = _repair_josou_homophone_translation_text(ko_literal)
            ko_natural = _repair_josou_homophone_translation_text(ko_natural)
            if ko_literal != before_literal or ko_natural != before_natural:
                repair_reasons.append("asr_homophone_josou")
        if not repair_reasons:
            continue
        before = translation.model_dump(mode="json")
        notes = list(translation.notes)
        if KOREAN_ASR_HOMOPHONE_POSTPROCESS_NOTE not in notes:
            notes.append(KOREAN_ASR_HOMOPHONE_POSTPROCESS_NOTE)
        repaired = translation.model_copy(
            update={
                "ko_literal": ko_literal,
                "ko_natural": ko_natural,
                "notes": notes,
            }
        )
        segment.translation_ko = repaired
        quality_counters["asr_homophone_repaired"] += 1
        diagnostics.append(
            {
                "segment_id": segment.id,
                "source_text": source_text,
                "repair_reasons": repair_reasons,
                "before": before,
                "after": repaired.model_dump(mode="json"),
                "accepted": not any(
                    bad in f"{repaired.ko_literal} {repaired.ko_natural}"
                    for bad in ("악몽", "도움닫기", "조주")
                ),
            }
        )
        rewritten_count += 1
    return rewritten_count


def _apply_korean_onomatopoeia_postprocess(
    segments: list[Segment],
    diagnostics: list[dict[str, Any]],
    quality_counters: Counter[str],
) -> int:
    rewritten_count = 0
    for segment in segments:
        translation = segment.translation_ko
        if translation is None or not _source_suggests_guriguri_onomatopoeia(segment):
            continue
        ko_literal = _repair_guriguri_onomatopoeia_translation_text(translation.ko_literal)
        ko_natural = _repair_guriguri_onomatopoeia_translation_text(translation.ko_natural)
        if ko_literal == translation.ko_literal and ko_natural == translation.ko_natural:
            continue
        before = translation.model_dump(mode="json")
        notes = list(translation.notes)
        if KOREAN_ONOMATOPOEIA_POSTPROCESS_NOTE not in notes:
            notes.append(KOREAN_ONOMATOPOEIA_POSTPROCESS_NOTE)
        repaired = translation.model_copy(
            update={
                "ko_literal": ko_literal,
                "ko_natural": ko_natural,
                "notes": notes,
            }
        )
        segment.translation_ko = repaired
        quality_counters["onomatopoeia_transliteration_repaired"] += 1
        diagnostics.append(
            {
                "segment_id": segment.id,
                "source_text": _translation_source_text(segment),
                "repair_reasons": ["onomatopoeia_guriguri"],
                "before": before,
                "after": repaired.model_dump(mode="json"),
                "accepted": not KOREAN_GURIGURI_TRANSLITERATION_RE.search(
                    f"{repaired.ko_literal} {repaired.ko_natural}"
                ),
            }
        )
        rewritten_count += 1
    return rewritten_count


def _apply_korean_fluency_postprocess(
    segments: list[Segment],
    diagnostics: list[dict[str, Any]],
    quality_counters: Counter[str],
) -> int:
    rewritten_count = 0
    for segment in segments:
        translation = segment.translation_ko
        if translation is None:
            continue
        ko_literal, literal_reasons = _repair_korean_fluency_translation_text(translation.ko_literal)
        ko_natural, natural_reasons = _repair_korean_fluency_translation_text(translation.ko_natural)
        repair_reasons = sorted({*literal_reasons, *natural_reasons})
        if not repair_reasons or (ko_literal == translation.ko_literal and ko_natural == translation.ko_natural):
            continue
        before = translation.model_dump(mode="json")
        notes = list(translation.notes)
        if KOREAN_FLUENCY_POSTPROCESS_NOTE not in notes:
            notes.append(KOREAN_FLUENCY_POSTPROCESS_NOTE)
        repaired = translation.model_copy(
            update={
                "ko_literal": ko_literal,
                "ko_natural": ko_natural,
                "notes": notes,
            }
        )
        segment.translation_ko = repaired
        quality_counters["fluency_repaired"] += 1
        diagnostics.append(
            {
                "segment_id": segment.id,
                "source_text": _translation_source_text(segment),
                "repair_reasons": repair_reasons,
                "before": before,
                "after": repaired.model_dump(mode="json"),
                "accepted": "끝져가는" not in f"{repaired.ko_literal} {repaired.ko_natural}",
            }
        )
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


def _translation_issue_codes(segment: Segment) -> list[str]:
    translation = segment.translation_ko
    if translation is None:
        return ["missing_translation"]
    source_text = _translation_source_text(segment)
    natural = translation.ko_natural.strip()
    combined = " ".join(
        text for text in [translation.ko_literal.strip(), natural] if text
    )
    issues: list[str] = []
    if not natural:
        issues.append("empty_translation")
    if _source_suggests_numeric_translation_repair(segment) and RAW_DIGIT_RE.search(natural):
        issues.append("raw_digit")
    if KOREAN_TRANSLATION_KANA_RE.search(natural):
        issues.append("japanese_kana")
    if KOREAN_TRANSLATION_CJK_RE.search(natural):
        issues.append("untranslated_cjk")
    latin_check_text = _strip_translation_test_placeholders(natural)
    if KOREAN_TRANSLATION_LATIN_RE.search(latin_check_text):
        issues.append("latin")
    if "媚薬" in source_text and ("변비약" in combined or "비약" in combined):
        issues.append("domain_mistranslation")
    if "18禁" in source_text and (
        "열여덟까지" in combined or "쓸 수 있는" in combined or "사용할 수 있는" in combined
    ):
        issues.append("domain_mistranslation")
    if JAPANESE_SECOND_ORDINAL_RE.search(source_text) and KOREAN_BAD_SECOND_ORDINAL_RE.search(combined):
        issues.append("ordinal_mistranslation")
    if _natural_translation_omits_repeated_source_phrase(source_text, translation):
        issues.append("natural_repetition_omission")
    return list(dict.fromkeys(issues))


def _compact_translation_repeat_text(text: str) -> str:
    return re.sub(r"[\s、。，,.!?！？…]+", "", text)


def _has_repeated_phrase(text: str, *, min_chars: int = 8) -> bool:
    compact = _compact_translation_repeat_text(text)
    if len(compact) < min_chars * 2:
        return False
    seen: set[str] = set()
    for index in range(0, len(compact) - min_chars + 1):
        phrase = compact[index : index + min_chars]
        if phrase in seen:
            return True
        seen.add(phrase)
    return False


def _natural_translation_omits_repeated_source_phrase(
    source_text: str,
    translation: Any,
) -> bool:
    if not _has_repeated_phrase(source_text):
        return False
    literal = _compact_translation_repeat_text(str(getattr(translation, "ko_literal", "")))
    natural = _compact_translation_repeat_text(str(getattr(translation, "ko_natural", "")))
    if len(literal) < 40 or len(natural) < 1:
        return False
    return len(natural) / len(literal) < 0.75


def _source_split_inside_japanese_word(left_text: str, right_text: str) -> bool:
    left = left_text.strip()
    right = right_text.strip()
    if not left or not right:
        return False
    if right.startswith(JAPANESE_COMMON_SEGMENT_STARTS):
        return False
    combined = f"{left}{right}"
    split_index = len(left)
    if _asr_text_split_boundary_score(combined, split_index) > 0:
        return False
    return _looks_like_japanese_word_internal_split(left[-1], right[0])


def _translation_boundary_issue_codes(segments: list[Segment]) -> dict[str, list[str]]:
    issues: dict[str, list[str]] = {}
    for left, right in zip(segments, segments[1:], strict=False):
        left_source = _translation_source_text(left)
        right_source = _translation_source_text(right)
        if not left_source or not right_source:
            continue
        if abs(float(right.start) - float(left.end)) > 0.05:
            continue
        if not _source_split_inside_japanese_word(left_source, right_source):
            continue
        for segment in (left, right):
            issues.setdefault(segment.id, []).append("source_split_inside_japanese_word")
    return issues


def _translation_backcheck_ko_pattern_matches(
    text: str,
    pattern: str,
    *,
    source_text: str = "",
) -> bool:
    if not pattern:
        return False
    if pattern == "비약":
        return bool(re.search(r"비약(?!\s*적)", text))
    if pattern == "미약" and "媚薬" in source_text and "火薬" not in source_text:
        return False
    return pattern in text


def _backcheck_severity(item: dict[str, Any]) -> str:
    hits = [str(hit) for hit in item.get("translation_hits", [])]
    if any(hit in SEVERE_TRANSLATION_BACKCHECK_KO_PATTERNS for hit in hits):
        return "severe"
    ko_natural = str(item.get("ko_natural") or "")
    if any(
        _translation_backcheck_ko_pattern_matches(ko_natural, pattern)
        for pattern in SEVERE_TRANSLATION_BACKCHECK_KO_PATTERNS
    ):
        return "severe"
    return "warning"


def _translation_rejection_message(reasons: list[str]) -> str:
    return TRANSLATION_REJECTION_ERROR_PREFIX + ", ".join(reasons)


def _clear_translation_rejection_errors(segment: Segment) -> None:
    segment.errors = [
        error
        for error in segment.errors
        if not error.startswith(TRANSLATION_REJECTION_ERROR_PREFIX)
    ]
    if segment.status in SKIP_STATUSES and not segment.errors:
        segment.status = "raw"


def _upsert_translation_row(
    rows: list[dict[str, Any]],
    segment: Segment,
    row: dict[str, Any],
) -> None:
    for index, existing in enumerate(rows):
        if str(existing.get("segment_id") or "") == segment.id:
            rows[index] = row
            return
    rows.append(row)


def _finalize_translation_acceptance(
    rows: list[dict[str, Any]],
    segments: list[Segment],
    asr_backcheck_items: list[dict[str, Any]],
    quality_counters: Counter[str],
) -> None:
    backcheck_by_segment_id = {str(item["segment_id"]): item for item in asr_backcheck_items}
    rows_by_segment_id = {str(row.get("segment_id") or ""): row for row in rows}
    boundary_warning_codes = _translation_boundary_issue_codes(segments)
    for segment in segments:
        row = rows_by_segment_id.get(segment.id)
        if row is None:
            continue
        translation = segment.translation_ko
        source_text = _translation_source_text(segment)
        if row.get("status") != "translated":
            continue
        rejected_reasons: list[str] = []
        issue_codes = _translation_issue_codes(segment)
        issue_codes = list(dict.fromkeys(issue_codes))
        warning_codes = list(dict.fromkeys(boundary_warning_codes.get(segment.id, [])))
        for issue in issue_codes:
            quality_counters[issue] += 1
        for warning in warning_codes:
            quality_counters[warning] += 1
        rejected_reasons.extend(issue_codes)
        backcheck = backcheck_by_segment_id.get(segment.id)
        if backcheck is not None:
            severity = _backcheck_severity(backcheck)
            backcheck["severity"] = severity
            row["asr_backcheck"] = backcheck
            if severity == "severe":
                quality_counters["severe_backcheck"] += 1
                rejected_reasons.append("severe_translation_smell")
        if rejected_reasons:
            segment.status = "needs_manual_review"
            message = _translation_rejection_message(rejected_reasons)
            if message not in segment.errors:
                segment.errors.append(message)
            row.update(
                {
                    "status": "needs_manual_review",
                    "reason": "translation rejected before TTS",
                    "source_text": source_text,
                    "translation_ko": translation.model_dump(mode="json") if translation else None,
                    "quality_issues": issue_codes,
                    "quality_warnings": warning_codes,
                    "accepted": False,
                    "rejected_reasons": rejected_reasons,
                }
            )
            _upsert_translation_row(rows, segment, row)
            continue
        _clear_translation_rejection_errors(segment)
        row.update(
            {
                "status": "translated",
                "source_text": source_text,
                "translation_ko": translation.model_dump(mode="json") if translation else None,
                "quality_issues": [],
                "quality_warnings": warning_codes,
                "accepted": True,
                "rejected_reasons": [],
            }
        )
        _upsert_translation_row(rows, segment, row)


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
        row["digit_pronunciation_postprocessed"] = (
            KOREAN_DIGIT_PRONUNCIATION_POSTPROCESS_NOTE in segment.translation_ko.notes
        )


def _translation_asr_backcheck_item(segment: Segment, cfg: Any) -> dict[str, Any] | None:
    if segment.source_script is None or segment.translation_ko is None:
        return None
    source_text = segment.source_script.text.strip()
    translation_text = " ".join(
        text.strip()
        for text in [
            segment.translation_ko.ko_literal,
            segment.translation_ko.ko_natural,
        ]
        if text.strip()
    )
    source_hits = [
        pattern
        for pattern in cfg.asr_translation_backcheck_source_patterns
        if pattern and pattern in source_text
    ]
    translation_hits = [
        pattern
        for pattern in cfg.asr_translation_backcheck_ko_patterns
        if pattern
        and _translation_backcheck_ko_pattern_matches(
            translation_text,
            pattern,
            source_text=source_text,
        )
    ]
    if not source_hits and not translation_hits:
        return None
    reasons: list[str] = []
    if source_hits:
        reasons.append("source_text_contains_known_asr_risk")
    if translation_hits:
        reasons.append("korean_translation_contains_asr_smell")
    item = {
        "segment_id": segment.id,
        "start": round(segment.start, 3),
        "end": round(segment.end, 3),
        "source_text": source_text,
        "ko_natural": segment.translation_ko.ko_natural,
        "source_hits": source_hits,
        "translation_hits": translation_hits,
        "reasons": reasons,
        "recommendation": "rerun_asr_review_or_manual_review",
    }
    item["severity"] = _backcheck_severity(item)
    if item["severity"] == "severe":
        item["policy_action"] = "needs_manual_review"
    return item


def _apply_translation_asr_backcheck(
    segments: list[Segment],
    cfg: Any,
) -> list[dict[str, Any]]:
    if not cfg.asr_translation_backcheck_enabled:
        return []
    items = [
        item
        for segment in segments
        if (item := _translation_asr_backcheck_item(segment, cfg)) is not None
    ]
    for item in items:
        segment = next((candidate for candidate in segments if candidate.id == item["segment_id"]), None)
        if segment is None:
            continue
        message = (
            "ASR translation backcheck flagged possible source transcription issue: "
            f"source_hits={item['source_hits']} translation_hits={item['translation_hits']}"
        )
        if message not in segment.errors:
            segment.errors.append(message)
        if cfg.asr_translation_backcheck_mark_manual_review:
            segment.status = "needs_manual_review"
    return items


def _attach_asr_backcheck_to_translation_rows(
    rows: list[dict[str, Any]],
    items: list[dict[str, Any]],
) -> None:
    by_segment_id = {str(item["segment_id"]): item for item in items}
    for row in rows:
        item = by_segment_id.get(str(row.get("segment_id") or ""))
        if item is not None:
            row["asr_backcheck"] = item


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


def _load_config_into_manifest(project_dir: Path, manifest: PipelineManifest) -> None:
    manifest.project_config = load_project_config(project_dir)


def _input_part_metadata(parts: tuple[Path, ...]) -> list[dict[str, Any]]:
    metadata: list[dict[str, Any]] = []
    for index, part in enumerate(parts, start=1):
        duration = None
        try:
            duration = ffmpeg.probe_media(part).duration_sec
        except ffmpeg.FFmpegError:
            duration = None
        metadata.append(
            {
                "part_index": index,
                "path": str(part),
                "sha256": sha256_file(part),
                "duration_sec": duration,
            }
        )
    return metadata


def _input_merge_metadata(
    *,
    requested_path: Path,
    selected_path: Path,
    parts: tuple[Path, ...],
    status: str,
    reason: str,
    merged_path: Path | None,
) -> dict[str, Any]:
    part_metadata = _input_part_metadata(parts)
    total_duration = sum(
        part["duration_sec"] for part in part_metadata if isinstance(part.get("duration_sec"), (int, float))
    )
    return {
        "requested": True,
        "status": status,
        "reason": reason,
        "requested_path": str(requested_path),
        "selected_input_path": str(selected_path),
        "merged_output_path": str(merged_path) if merged_path else None,
        "part_count": len(parts),
        "parts": part_metadata,
        "total_part_duration_sec": total_duration,
    }


def _attach_input_merge_to_audit(audit: RightsAudit, metadata: dict[str, Any] | None) -> RightsAudit:
    if not metadata or not audit.history:
        return audit
    history = [*audit.history]
    history[-1] = {**history[-1], "input_merge": metadata}
    return audit.model_copy(update={"history": history})


def _write_segments_manifest(path: Path, segments: list[Segment]) -> None:
    write_json_atomic(path, {"segments": [s.model_dump(mode="json") for s in segments]})


def _seed_segments_for_transcribe(
    project_dir: Path,
    manifest: PipelineManifest,
    audio_path: Path,
    mix_audio_path: Path,
    *,
    write_audio_clips: bool = True,
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
    manifest.segments = (
        write_segment_audio_clips([seed], audio_path, mix_audio_path, project_dir)
        if write_audio_clips
        else [seed]
    )
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
        audio_clips_written=write_audio_clips,
    )
    save_manifest(project_dir, manifest)
    return True


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


ASR_PRESET_OVERRIDES: dict[str, dict[str, Any]] = {
    "conservative": {
        "asr_vad_filter": True,
        "asr_vad_parameters": {
            "threshold": 0.45,
            "min_silence_duration_ms": 650,
            "speech_pad_ms": 350,
            "max_speech_duration_s": 18.0,
        },
        "asr_word_timestamps": True,
        "asr_hallucination_silence_threshold": 1.0,
        "asr_condition_on_previous_text": False,
        "asr_sparse_chunk_max_sec": 24.0,
        "asr_sparse_chunk_min_chars_per_sec": 0.65,
    },
    "whisper": {
        "asr_vad_filter": True,
        "asr_vad_parameters": {
            "threshold": 0.25,
            "min_silence_duration_ms": 850,
            "speech_pad_ms": 650,
            "max_speech_duration_s": 20.0,
        },
        "asr_word_timestamps": True,
        "asr_hallucination_silence_threshold": 1.0,
        "asr_condition_on_previous_text": False,
        "asr_sparse_chunk_max_sec": 24.0,
        "asr_sparse_chunk_min_chars_per_sec": 0.55,
        "asr_repair_enabled": True,
        "asr_repair_padding_sec": 1.4,
    },
    "no_vad_repair": {
        "asr_vad_filter": False,
        "asr_vad_parameters": {},
        "asr_word_timestamps": True,
        "asr_hallucination_silence_threshold": 1.0,
        "asr_condition_on_previous_text": False,
        "asr_repair_enabled": True,
        "asr_repair_padding_sec": 1.2,
    },
}


def _effective_asr_config(
    cfg: Any,
    *,
    asr_preset: str | None = None,
    asr_vad_off: bool | None = None,
    asr_diagnostics: bool | None = None,
    asr_device: str | None = None,
    asr_compute_type: str | None = None,
    asr_batched_inference: bool | None = None,
    asr_batch_size: int | None = None,
) -> Any:
    payload = cfg.model_dump(mode="json")
    preset = (asr_preset or getattr(cfg, "asr_preset", None) or "default").replace("-", "_")
    payload["asr_preset"] = preset
    overrides = ASR_PRESET_OVERRIDES.get(preset, {})
    for key, value in overrides.items():
        if key == "asr_vad_parameters":
            payload[key] = {**dict(getattr(cfg, "asr_vad_parameters", {}) or {}), **dict(value)}
        else:
            payload[key] = value
    if asr_vad_off:
        payload["asr_vad_filter"] = False
        payload["asr_vad_parameters"] = {}
    if asr_diagnostics is not None:
        payload["asr_diagnostics_enabled"] = bool(asr_diagnostics)
    if asr_device is not None:
        payload["asr_device"] = asr_device
    if asr_compute_type is not None:
        payload["asr_compute_type"] = asr_compute_type
    if asr_batched_inference is not None:
        payload["asr_batched_inference"] = bool(asr_batched_inference)
    if asr_batch_size is not None:
        payload["asr_batch_size"] = int(asr_batch_size)
    next_cfg = type(cfg).model_validate(payload)
    next_cfg.asr.correction_profile = cfg.asr.correction_profile
    return next_cfg


def _asr_backend_config(cfg: Any) -> dict[str, Any]:
    return {
        "model_id": cfg.asr_model_id,
        "language": cfg.asr_language,
        "local_files_only": cfg.asr_local_files_only,
        "device": cfg.asr_device,
        "compute_type": cfg.asr_compute_type,
        "batched_inference": cfg.asr_batched_inference,
        "batch_size": cfg.asr_batch_size,
        "beam_size": cfg.asr_beam_size,
        "best_of": cfg.asr_best_of,
        "condition_on_previous_text": cfg.asr_condition_on_previous_text,
        "vad_filter": cfg.asr_vad_filter,
        "vad_parameters": cfg.asr_vad_parameters,
        "word_timestamps": cfg.asr_word_timestamps,
        "hallucination_silence_threshold": cfg.asr_hallucination_silence_threshold,
        "initial_prompt": cfg.asr_initial_prompt,
        "hotwords": cfg.asr_hotwords,
        "qwen_model_id": cfg.qwen_asr_model_id,
        "qwen_forced_aligner_model_id": cfg.qwen_asr_forced_aligner_model_id,
        "qwen_device_map": cfg.qwen_asr_device_map,
        "qwen_dtype": cfg.qwen_asr_dtype,
        "qwen_return_timestamps": cfg.qwen_asr_return_timestamps,
        "qwen_context": cfg.qwen_asr_context,
        "qwen_max_inference_batch_size": cfg.qwen_asr_max_inference_batch_size,
        "qwen_max_new_tokens": cfg.qwen_asr_max_new_tokens,
    }


def _manifest_warning_once(manifest: PipelineManifest, warning: str) -> None:
    if warning not in manifest.warnings:
        manifest.warnings.append(warning)


def _asr_audio_metrics(path: Path) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "duration_sec": 0.0,
        "sample_rate": None,
        "channels": None,
        "rms_dbfs": None,
        "peak_dbfs": None,
        "clipping_ratio": None,
        "error": None,
    }
    if not path.exists():
        metrics["error"] = "missing"
        return metrics
    try:
        info = sf.info(str(path))
        metrics["duration_sec"] = round(float(info.frames) / float(info.samplerate), 6) if info.samplerate else 0.0
        metrics["sample_rate"] = int(info.samplerate)
        metrics["channels"] = int(info.channels)
        peak = 0.0
        sum_squares = 0.0
        sample_count = 0
        clipped = 0
        for block in sf.blocks(str(path), blocksize=65536, always_2d=True, dtype="float32"):
            abs_block = np.abs(block)
            peak = max(peak, float(np.max(abs_block)) if abs_block.size else 0.0)
            sum_squares += float(np.sum(np.square(block)))
            sample_count += int(block.size)
            clipped += int(np.sum(abs_block >= 0.999))
        rms = math.sqrt(sum_squares / sample_count) if sample_count > 0 else 0.0
        metrics["rms_dbfs"] = round(20.0 * math.log10(rms), 3) if rms > 0 else -120.0
        metrics["peak_dbfs"] = round(20.0 * math.log10(peak), 3) if peak > 0 else -120.0
        metrics["clipping_ratio"] = round(clipped / sample_count, 8) if sample_count > 0 else 0.0
    except Exception as exc:
        metrics["error"] = str(exc)
    return metrics


def _asr_metric_float(metrics: dict[str, Any] | None, key: str, default: float = -120.0) -> float:
    if not metrics:
        return default
    value = metrics.get(key)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _reject_relative_quiet_source_vocals(candidates: list[dict[str, Any]]) -> None:
    source = next((item for item in candidates if item.get("source") == "source_vocals_mono_16k"), None)
    reference = next((item for item in candidates if item.get("source") == "gemma_mono_16k"), None)
    if source is None or reference is None:
        return
    if source.get("reject_reasons") or reference.get("reject_reasons"):
        return
    source_rms = _asr_metric_float(source.get("metrics"), "rms_dbfs")
    reference_rms = _asr_metric_float(reference.get("metrics"), "rms_dbfs")
    source_peak = _asr_metric_float(source.get("metrics"), "peak_dbfs")
    reference_peak = _asr_metric_float(reference.get("metrics"), "peak_dbfs")
    rms_delta = reference_rms - source_rms
    peak_delta = reference_peak - source_peak
    if (
        source_rms <= ASR_SOURCE_VOCALS_RELATIVE_QUIET_MAX_RMS_DBFS
        and rms_delta >= ASR_SOURCE_VOCALS_RELATIVE_QUIET_MIN_RMS_DELTA_DB
        and peak_delta >= ASR_SOURCE_VOCALS_RELATIVE_QUIET_MIN_PEAK_DELTA_DB
    ):
        source.setdefault("reject_reasons", []).append(
            "relative_too_quiet_vs_gemma:"
            f"rms_delta={rms_delta:.1f}db,"
            f"peak_delta={peak_delta:.1f}db,"
            f"source_rms={source_rms:.1f}dbfs,"
            f"gemma_rms={reference_rms:.1f}dbfs"
        )


def _asr_reference_duration(manifest: PipelineManifest, project_dir: Path) -> float | None:
    if manifest.source_info and manifest.source_info.duration_sec > 0:
        return manifest.source_info.duration_sec
    for key in ("gemma_mono_16k", "original_stereo_48k"):
        raw = manifest.artifacts.get(key)
        if not raw:
            continue
        path = Path(raw)
        if not path.is_absolute():
            path = project_dir / path
        try:
            return duration_sec(path)
        except Exception:
            continue
    return None


def _asr_candidate_reject_reasons(
    metrics: dict[str, Any],
    *,
    cfg: Any,
    expected_sample_rate: int,
    expected_channels: int,
    reference_duration_sec: float | None,
) -> list[str]:
    reasons: list[str] = []
    if not metrics.get("exists"):
        return ["missing"]
    if metrics.get("error"):
        return [f"read_error:{metrics['error']}"]
    if metrics.get("sample_rate") != expected_sample_rate or metrics.get("channels") != expected_channels:
        reasons.append("format_mismatch")
    duration = float(metrics.get("duration_sec") or 0.0)
    if duration <= 0.05:
        reasons.append("empty_or_too_short")
    if reference_duration_sec and reference_duration_sec > 0 and duration > 0:
        delta = abs(duration - reference_duration_sec)
        if delta > 1.0 and delta / reference_duration_sec > cfg.asr_input_duration_tolerance:
            reasons.append(
                f"duration_mismatch:{duration:.3f}s_vs_{reference_duration_sec:.3f}s"
            )
    rms_value = metrics.get("rms_dbfs")
    peak_value = metrics.get("peak_dbfs")
    rms = float(rms_value) if rms_value is not None else -120.0
    peak = float(peak_value) if peak_value is not None else -120.0
    if rms <= cfg.asr_input_min_rms_dbfs or peak <= cfg.asr_input_min_peak_dbfs:
        reasons.append(f"too_quiet:rms={rms:.1f}dbfs,peak={peak:.1f}dbfs")
    clipping = float(metrics.get("clipping_ratio") or 0.0)
    if clipping > 0.02:
        reasons.append(f"clipping:{clipping:.4f}")
    return reasons


def _resolve_manifest_artifact_path(project_dir: Path, manifest: PipelineManifest, key: str) -> Path | None:
    raw = manifest.artifacts.get(key)
    if not raw:
        return None
    path = Path(raw)
    return path if path.is_absolute() else project_dir / path


def _derive_original_mono_16k(project_dir: Path, manifest: PipelineManifest, cfg: Any) -> Path | None:
    existing = _resolve_manifest_artifact_path(project_dir, manifest, "asr_original_mono_16k")
    if existing and existing.exists():
        return existing
    source = _resolve_manifest_artifact_path(project_dir, manifest, "original_stereo_48k")
    if source is None and manifest.source_info:
        source = Path(manifest.source_info.path)
    if source is None or not source.exists():
        return None
    output = project_dir / "work" / "audio" / "asr_original_mono_16k.wav"
    try:
        ffmpeg.extract_mono_16k(source, output)
    except Exception:
        data, sample_rate = load_audio(source)
        mono = resample_linear(to_mono(data), sample_rate, cfg.gemma_sample_rate)
        write_audio(output, mono[:, None] if mono.ndim == 1 else mono, cfg.gemma_sample_rate)
    manifest.artifacts["asr_original_mono_16k"] = str(output)
    return output


def _select_asr_audio_input(
    project_dir: Path,
    manifest: PipelineManifest,
    *,
    backend_kind: str,
    cfg: Any,
) -> tuple[Path, Path, dict[str, Any]]:
    reference_duration = _asr_reference_duration(manifest, project_dir)
    candidates: list[dict[str, Any]] = []
    warnings: list[str] = []

    def add_candidate(source: str, path: Path | None, *, strict_quality: bool) -> None:
        if path is None:
            candidates.append({"source": source, "path": None, "metrics": None, "reject_reasons": ["missing"]})
            return
        metrics = _asr_audio_metrics(path)
        reject_reasons = _asr_candidate_reject_reasons(
            metrics,
            cfg=cfg,
            expected_sample_rate=cfg.gemma_sample_rate,
            expected_channels=1,
            reference_duration_sec=reference_duration,
        )
        if not strict_quality:
            reject_reasons = [
                reason
                for reason in reject_reasons
                if reason.startswith(("missing", "read_error", "format_mismatch", "empty_or_too_short"))
            ]
        candidates.append(
            {
                "source": source,
                "path": str(path),
                "metrics": metrics,
                "reject_reasons": reject_reasons,
            }
        )

    add_candidate(
        "source_vocals_mono_16k",
        _resolve_manifest_artifact_path(project_dir, manifest, "source_vocals_mono_16k"),
        strict_quality=backend_kind != "mock",
    )
    add_candidate(
        "gemma_mono_16k",
        _resolve_manifest_artifact_path(project_dir, manifest, "gemma_mono_16k"),
        strict_quality=False,
    )
    _reject_relative_quiet_source_vocals(candidates)
    selected = next((candidate for candidate in candidates if not candidate["reject_reasons"]), None)
    if selected is None:
        add_candidate(
            "original_derived_mono_16k",
            _derive_original_mono_16k(project_dir, manifest, cfg),
            strict_quality=False,
        )
        selected = next((candidate for candidate in candidates if not candidate["reject_reasons"]), None)
    else:
        candidates.append(
            {
                "source": "original_derived_mono_16k",
                "path": None,
                "metrics": None,
                "reject_reasons": ["not_evaluated"],
            }
        )
    if selected is None:
        selected = next((candidate for candidate in candidates if candidate.get("path")), None)
    if selected is None or not selected.get("path"):
        raise ValueError("No usable ASR input audio was found. Run extract first.")

    for candidate in candidates:
        if candidate["source"] == selected["source"]:
            continue
        reasons = candidate.get("reject_reasons") or []
        if reasons and reasons != ["not_evaluated"]:
            warning = f"ASR input candidate {candidate['source']} rejected: {', '.join(reasons)}"
            warnings.append(warning)
            _manifest_warning_once(manifest, warning)

    audio_path = Path(str(selected["path"]))
    _validate_audio_contract(audio_path, cfg.gemma_sample_rate, 1, selected["source"])
    if selected["source"] == "source_vocals_mono_16k":
        mix_audio_path = _resolve_manifest_artifact_path(project_dir, manifest, "source_vocals_48k")
    else:
        mix_audio_path = None
    if mix_audio_path is None or not mix_audio_path.exists():
        mix_audio_path = _resolve_manifest_artifact_path(project_dir, manifest, "original_stereo_48k")
    if mix_audio_path is None or not mix_audio_path.exists():
        mix_audio_path = audio_path

    diagnostics = {
        "backend": backend_kind,
        "reference_duration_sec": reference_duration,
        "selected": {
            "source": selected["source"],
            "path": selected["path"],
            "metrics": selected["metrics"],
        },
        "mix_audio_path": str(mix_audio_path),
        "candidates": candidates,
        "warnings": warnings,
    }
    return audio_path, mix_audio_path, diagnostics


def _qwen_asr_dependency_available() -> bool:
    return importlib.util.find_spec("qwen_asr") is not None


def _create_qwen_repair_fallback_backend(
    cfg: Any,
    manifest: PipelineManifest,
) -> tuple[Any | None, dict[str, Any]]:
    summary: dict[str, Any] = {
        "enabled": bool(getattr(cfg, "asr_qwen_repair_fallback_enabled", False)),
        "available": False,
        "backend": "qwen_asr",
        "skipped_reason": None,
    }
    if not summary["enabled"]:
        summary["skipped_reason"] = "disabled"
        return None, summary
    if not _qwen_asr_dependency_available():
        summary["skipped_reason"] = "qwen-asr is not installed"
        _manifest_warning_once(
            manifest,
            "qwen-asr is not installed; ASR repair will continue with faster-whisper candidates only.",
        )
        return None, summary
    try:
        backend = create_asr_backend("qwen_asr", _asr_backend_config(cfg))
    except ASRUnavailableError as exc:
        summary["skipped_reason"] = str(exc)
        _manifest_warning_once(
            manifest,
            f"qwen-asr repair fallback unavailable: {exc}",
        )
        return None, summary
    summary["available"] = True
    return backend, summary


def _asr_chunk_diagnostics(
    chunks: list[ASRChunk],
    *,
    cfg: Any,
    replacements: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    replacements = replacements or {}
    patterns = list(getattr(cfg, "asr_repair_suspicious_text_patterns", []) or []) + list(
        getattr(cfg, "asr_review_suspicious_text_patterns", []) or []
    )
    rows: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks, start=1):
        duration = max(0.0, chunk.end - chunk.start)
        text = chunk.text.strip()
        rows.append(
            {
                "chunk_id": f"chunk_{index:04d}",
                "start": round(chunk.start, 3),
                "end": round(chunk.end, 3),
                "duration": round(duration, 3),
                "text": text,
                "language": chunk.language,
                "confidence": chunk.confidence,
                "text_density": round(len(text) / max(0.001, duration), 6),
                "suspicious_pattern_hits": _asr_suspicious_pattern_hits(text, patterns),
                "prompt_leak": _asr_candidate_looks_prompt_leaked(text, cfg),
                "replacement_hits": _replacement_hits(text, replacements),
            }
        )
    return rows


def _filter_final_asr_chunks_for_hallucinations(
    chunks: list[ASRChunk],
    *,
    cfg: Any,
) -> tuple[list[ASRChunk], list[dict[str, Any]]]:
    kept: list[ASRChunk] = []
    dropped: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks, start=1):
        text = chunk.text.strip()
        duration = max(0.0, chunk.end - chunk.start)
        density = len(text) / max(0.001, duration)
        prompt_leak = _asr_candidate_looks_prompt_leaked(text, cfg)
        sparse_prompt_leak = (
            prompt_leak
            and duration >= max(2.0, float(getattr(cfg, "asr_repair_sparse_min_sec", 12.0)) * 0.5)
            and density < max(1.0, float(getattr(cfg, "asr_repair_sparse_min_chars_per_sec", 1.0)))
        )
        repeated_outro = (
            prompt_leak
            and any(marker in text for marker in ASR_HARD_PROMPT_LEAK_MARKERS)
            and duration >= 4.0
        )
        if sparse_prompt_leak or repeated_outro:
            dropped.append(
                {
                    "chunk_id": f"chunk_{index:04d}",
                    "start": round(chunk.start, 3),
                    "end": round(chunk.end, 3),
                    "text": text,
                    "reason": "sparse_prompt_leak" if sparse_prompt_leak else "repeated_outro_hallucination",
                    "text_density": round(density, 6),
                    "duration": round(duration, 3),
                }
            )
            continue
        kept.append(chunk)
    return kept, dropped


def _source_script_asr_review_reasons(source_script: SourceScript | None, cfg: Any) -> list[str]:
    if source_script is None or not source_script.text.strip():
        return ["missing_asr_text"]
    text = source_script.text.strip()
    reasons: list[str] = []
    if _asr_candidate_looks_prompt_leaked(text, cfg):
        reasons.append("asr_prompt_or_hallucination_leak")
    patterns = list(getattr(cfg, "asr_repair_suspicious_text_patterns", []) or []) + list(
        getattr(cfg, "asr_review_suspicious_text_patterns", []) or []
    )
    hits = _asr_suspicious_pattern_hits(text, patterns)
    if hits:
        reasons.append("asr_suspicious_pattern:" + ",".join(hits[:5]))
    if (
        source_script.confidence is not None
        and source_script.confidence < getattr(cfg, "asr_review_confidence_threshold", 0.78)
    ):
        reasons.append(f"asr_low_confidence:{source_script.confidence:.3f}")
    duration = max(0.001, source_script.end - source_script.start)
    if (
        duration >= getattr(cfg, "asr_repair_sparse_min_sec", 12.0)
        and len(text) / duration < getattr(cfg, "asr_repair_sparse_min_chars_per_sec", 1.0)
    ):
        reasons.append("asr_sparse_text_density")
    return reasons


def _source_script_rejected_repair_reasons(
    source_script: SourceScript | None,
    repair_summary: dict[str, Any],
) -> list[str]:
    if source_script is None or not source_script.text.strip():
        return []
    reasons: list[str] = []
    script_start = float(source_script.start)
    script_end = float(source_script.end)
    script_duration = max(0.001, script_end - script_start)
    for item in repair_summary.get("items", []):
        if item.get("accepted"):
            continue
        try:
            repair_start = float(item.get("start"))
            repair_end = float(item.get("end"))
        except (TypeError, ValueError):
            continue
        overlap = max(0.0, min(script_end, repair_end) - max(script_start, repair_start))
        repair_duration = max(0.001, repair_end - repair_start)
        if overlap / min(script_duration, repair_duration) < 0.3:
            continue
        attempt_reasons = {
            str(attempt.get("reason") or "")
            for attempt in item.get("attempts", [])
            if attempt.get("reason")
        }
        suffix = (
            ":prompt_or_hallucination_leak"
            if "prompt_or_hallucination_leak" in attempt_reasons
            else ""
        )
        reason = f"asr_repair_rejected{suffix}"
        if reason not in reasons:
            reasons.append(reason)
    return reasons


def _write_asr_diagnostics_artifacts(
    project_dir: Path,
    manifest: PipelineManifest,
    *,
    backend_kind: str,
    backend_name: str,
    cfg: Any,
    input_diagnostics: dict[str, Any],
    raw_chunks: list[ASRChunk],
    repaired_chunks: list[ASRChunk],
    final_chunks: list[ASRChunk],
    repair_summary: dict[str, Any],
    asr_review_summary: dict[str, Any],
    replacements_summary: dict[str, Any],
    filtered_summary: list[dict[str, Any]],
    qwen_fallback_summary: dict[str, Any],
) -> None:
    transcribe_dir = project_dir / "work" / "transcribe"
    input_path = transcribe_dir / "asr_input_diagnostics.json"
    write_json_atomic(input_path, input_diagnostics)
    manifest.artifacts["asr_input_diagnostics"] = str(input_path)
    if not getattr(cfg, "asr_diagnostics_enabled", True):
        return
    vad_parameters = dict(getattr(cfg, "asr_vad_parameters", {}) or {})
    final_chunk_rows = _asr_chunk_diagnostics(
        final_chunks,
        cfg=cfg,
        replacements=getattr(cfg, "asr_text_replacements", {}),
    )
    for row in final_chunk_rows:
        for item in replacements_summary.get("items", []):
            if (
                abs(float(row["start"]) - float(item.get("start", -1))) <= 0.001
                and abs(float(row["end"]) - float(item.get("end", -1))) <= 0.001
                and str(row["text"]) == str(item.get("replaced_text", ""))
            ):
                row["replacement_hits"] = list(item.get("hits", []))
                break
    diagnostics = {
        "backend": backend_kind,
        "backend_name": backend_name,
        "preset": getattr(cfg, "asr_preset", "default"),
        "runtime": {
            "device": getattr(cfg, "asr_device", "auto"),
            "compute_type": getattr(cfg, "asr_compute_type", "default"),
            "batched_inference": bool(getattr(cfg, "asr_batched_inference", False)),
            "batch_size": int(getattr(cfg, "asr_batch_size", 8)),
        },
        "input_audio": input_diagnostics,
        "vad": {
            "vad_filter": bool(getattr(cfg, "asr_vad_filter", False)),
            "vad_parameters": vad_parameters,
            "word_timestamps": bool(getattr(cfg, "asr_word_timestamps", False)),
            "hallucination_silence_threshold": getattr(
                cfg,
                "asr_hallucination_silence_threshold",
                None,
            ),
        },
        "raw_asr_chunks": _asr_chunk_diagnostics(raw_chunks, cfg=cfg),
        "repaired_asr_chunks": _asr_chunk_diagnostics(repaired_chunks, cfg=cfg),
        "final_asr_chunks": final_chunk_rows,
        "repair": repair_summary,
        "asr_review": asr_review_summary,
        "text_replacements": replacements_summary,
        "filtered_chunks": filtered_summary,
        "qwen_repair_fallback": qwen_fallback_summary,
    }
    summary = {
        "backend": backend_kind,
        "backend_name": backend_name,
        "preset": getattr(cfg, "asr_preset", "default"),
        "runtime": {
            "device": getattr(cfg, "asr_device", "auto"),
            "compute_type": getattr(cfg, "asr_compute_type", "default"),
            "batched_inference": bool(getattr(cfg, "asr_batched_inference", False)),
            "batch_size": int(getattr(cfg, "asr_batch_size", 8)),
        },
        "asr_input_source": input_diagnostics.get("selected", {}).get("source"),
        "raw_asr_chunk_count": len(raw_chunks),
        "repaired_asr_chunk_count": len(repaired_chunks),
        "final_asr_chunk_count": len(final_chunks),
        "repair_attempted": repair_summary.get("attempted", 0),
        "repair_repaired": repair_summary.get("repaired", 0),
        "asr_review_attempted": asr_review_summary.get("attempted", 0),
        "asr_review_replaced": asr_review_summary.get("replaced", 0),
        "text_replacements": replacements_summary,
        "filtered_chunk_count": len(filtered_summary),
        "needs_manual_review": sum(1 for segment in manifest.segments if segment.status == "needs_manual_review"),
        "no_speech_detected": sum(1 for segment in manifest.segments if segment.status in NO_SPEECH_STATUSES),
        "warnings": input_diagnostics.get("warnings", []),
        "qwen_repair_fallback": qwen_fallback_summary,
    }
    diagnostics_path = transcribe_dir / "asr_diagnostics.json"
    summary_path = transcribe_dir / "asr_diagnostics_summary.json"
    write_json_atomic(diagnostics_path, diagnostics)
    write_json_atomic(summary_path, summary)
    manifest.artifacts["asr_diagnostics"] = str(diagnostics_path)
    manifest.artifacts["asr_diagnostics_summary"] = str(summary_path)


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
    include_mmproj: bool = False,
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
        model_path=(
            cfg.gemma_llama_cpp_audio_model_path
            if include_mmproj
            else cfg.gemma_llama_cpp_model_path
        ),
        mmproj_path=cfg.gemma_llama_cpp_audio_mmproj_path if include_mmproj else None,
        ctx_size=cfg.gemma_llama_cpp_ctx_size,
        gpu_layers=cfg.gemma_llama_cpp_gpu_layers,
        n_predict=cfg.gemma_text_n_predict,
    )


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
    stale_manifest = dataset_dir / "dataset_manifest.json"
    if stale_manifest.exists():
        stale_manifest.unlink()
    rows: list[dict[str, str]] = []
    rejected_rows: list[dict[str, Any]] = []
    cfg = manifest.project_config
    strict_training_filter = cfg.rvc_train_backend == "command"
    for segment in manifest.segments:
        if segment.status in SKIP_STATUSES:
            rejected_rows.append(
                {
                    "segment_id": segment.id,
                    "source_path": None,
                    "reject_reasons": [f"skip_status:{segment.status}"],
                }
            )
            continue
        if strict_training_filter:
            check = evaluate_voice_training_candidate(
                project_dir,
                segment,
                cfg,
                min_quality_score=cfg.rvc_train_min_quality_score,
                require_source_script=False,
                require_speaker_id=True,
            )
            if not check.accepted:
                rejected_rows.append(
                    {
                        "segment_id": segment.id,
                        "source_path": str(check.source_audio_path) if check.source_audio_path else None,
                        "reject_reasons": list(check.reject_reasons),
                        "quality_score": round(check.metrics.score, 6) if check.metrics else None,
                        "quality_issues": list(check.metrics.issues) if check.metrics else [],
                        "speaker_id": segment.speaker_id,
                        "analysis_speaker_count": segment.analysis.get("speaker_count"),
                    }
                )
                continue
            if not check.source_audio_path:
                raise RVCCommandError(f"train-rvc source segment audio is missing for {segment.id}")
            source_path = check.source_audio_path
        else:
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
    accepted_names = {f"{row['segment_id']}.wav" for row in rows}
    for stale_wav in dataset_dir.glob("*.wav"):
        if stale_wav.name not in accepted_names:
            stale_wav.unlink()
    if not rows:
        raise RVCCommandError("train-rvc requires at least one clean source voice segment audio file.")
    write_json_atomic(
        project_dir / "work" / "rvc_train" / "dataset_manifest.json",
        {"segments": rows, "rejected_segments": rejected_rows},
    )
    return dataset_dir, rows


def _train_rvc_ready_for_rvc(manifest: PipelineManifest) -> bool:
    status = manifest.stage_state.get("train-rvc", {}).get("status")
    if status == "completed":
        return True
    return status == "skipped_pretrained_voice_bank" and bool(manifest.project_config.rvc_speaker_models)


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
    rvc_output_ok = bool(segment.rvc and segment.rvc.accepted and segment.rvc.output_path)
    if (
        segment.status == "ok"
        and segment.qc is not None
        and segment.qc.recommendation == "pass"
        and (selected_candidate_ok or rvc_output_ok)
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


__all__ = [name for name in globals() if not name.startswith("__")]
