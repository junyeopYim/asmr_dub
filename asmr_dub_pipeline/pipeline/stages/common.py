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
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
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
    folder_asr_part_skip_reason,
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
    write_segment_audio_clips_from_parts,
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
    few_shot_min_total_sec,
    select_training_items,
    select_training_speaker_ids,
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
from asmr_dub_pipeline.qc.pronunciation_qc import (
    evaluate_pronunciation_chunks,
    evaluate_pronunciation_text,
)
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
    GSVSpeakerConfig,
    JapaneseScript,
    KoreanTranslation,
    PipelineManifest,
    ProjectConfig,
    RightsAudit,
    RVCMetadata,
    RVCProfile,
    RVCSpeakerConfig,
    Segment,
    SourceLaneTranscript,
    SourceScript,
    TranscriptWord,
    TTSCandidate,
    TTSMetadata,
)
from asmr_dub_pipeline.script.countdown import (
    COUNTDOWN_EVENT_KEY,
    COUNTDOWN_EVENT_NOTE,
    countdown_korean_text,
    countdown_korean_tokens,
    is_descending_countdown,
    is_descending_countdown_prefix,
    native_korean_count_number,
    repair_descending_countdown_text,
    source_countdown_token_matches,
    source_countdown_values,
)
from asmr_dub_pipeline.script.duration_rewrite import (
    estimate_tts_duration,
    korean_tts_slot_timing_budget,
    korean_tts_speech_char_count,
    korean_tts_timing_budget,
    rewrite_for_duration,
)
from asmr_dub_pipeline.script.korean_colloquial import (
    COLLOQUIAL_REWRITE_NOTE,
    colloquialize_korean_translation,
)
from asmr_dub_pipeline.script.normalizer import (
    normalize_japanese_kana_text,
    normalize_korean_tts_text,
    normalize_script_payload,
)
from asmr_dub_pipeline.script.text_qc import (
    preflight_tts_text,
    repair_suspicious_truncated_korean_tts_text,
)
from asmr_dub_pipeline.voice_bank import (
    apply_voice_bank_to_config,
    assign_source_speakers_to_manifest,
    assign_speakers_to_manifest,
    load_voice_bank,
    resolve_voice_bank_path,
    validate_voice_bank_models,
)

NO_SPEECH_STATUSES = {"no_speech_detected", "non_speech_texture"}
SKIP_STATUSES = {"needs_manual_review", "absorbed", "failed", *NO_SPEECH_STATUSES}
ASR_SOURCE_SEPARATION_FALLBACK_MIN_MANUAL_REVIEW_RATE = 0.02
ASR_SOURCE_SEPARATION_FALLBACK_MIN_MANUAL_REVIEW_COUNT = 3
ASR_SOURCE_SEPARATION_FALLBACK_SEPARATED_SOURCES = {"source_vocals_mono_16k"}
GSV_API_MIN_REF_SEC = 3.0
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
NUMERIC_SEQUENCE_SOURCE_RE = re.compile(r"^\s*\d+(?:[\s,，、]+\d+)+\s*$")
NUMERIC_TOKEN_RE = re.compile(r"\d+")
REPEATED_CHAR_RUNAWAY_RE = re.compile(r"(.)\1{24,}")
ASR_SHORT_FILLER_KEEP_ORIGINAL_REASON = "asr_short_filler_keep_original_texture"
ASR_SHORT_FILLER_KEEP_ORIGINAL_POLICY = "conservative_short_filler"
ASR_SHORT_FILLER_KEEP_ORIGINAL_MAX_SEC = 1.2
ASR_SHORT_CLEAN_FRAGMENT_WARNING_REASON = "asr_short_clean_fragment_auto_accepted"
ASR_SHORT_CLEAN_FRAGMENT_DROP_RE = re.compile(r"[\s　、。,.，．!！?？…・♪♡❤（）()\[\]「」『』]+")
ASR_SHORT_CLEAN_FRAGMENT_TOKENS = frozenset(
    (
        "これ",
        "あの",
        "うん",
        "さ",
        "はい",
        "ね",
        "そう",
        "あ",
        "え",
        "ん",
        "いや",
        "そうそう",
        "まあ",
    )
)
ASR_SHORT_FILLER_KEEP_ORIGINAL_TOKENS = frozenset(
    (
        "あ",
        "あー",
        "ああ",
        "あっ",
        "え",
        "えー",
        "ええ",
        "えっ",
        "うん",
        "うーん",
        "ん",
        "んー",
        "はい",
        "はぁ",
        "はあ",
        "ふふ",
        "ふふふ",
        "うふふ",
        "あはは",
        "へへ",
        "んふ",
        "んふふ",
        "あの",
        "その",
        "えっと",
        "えーと",
        "ええと",
        "さ",
        "さあ",
        "ね",
        "ねえ",
        "ほら",
        "そう",
        "そうだね",
        "そうね",
        "そうそう",
        "まあ",
        "ま",
        "お",
        "ほ",
        "スー",
        "シュー",
    )
)
ASR_SHORT_FILLER_KEEP_ORIGINAL_REPEAT_UNITS = frozenset(("ほら", "さ", "ね", "そう", "うん", "あ", "え", "ふふ"))
NON_SPEECH_TEXTURE_REPEAT_RE = re.compile(r"(.)\1{2,}")
NON_SPEECH_TEXTURE_DOMINANT_REPEAT_RE = re.compile(r"(.)\1{24,}")
NON_SPEECH_TEXTURE_CHARS = frozenset(
    "ぁあぃいぅうぇえぉおァアィイゥウェエォオ"
    "んンっッはハひヒふフへヘほホゃゅょャュョ"
    "ーｰ〜～"
)
NON_SPEECH_TEXTURE_MARKERS = frozenset("ぁぃぅぇぉァィゥェォっッーｰ〜～")
NON_SPEECH_TEXTURE_DOMINANT_REPEAT_CHARS = frozenset(
    "ぁあぃいぅうぇえぉおァアィイゥウェエォオんンっッ"
)
NON_SPEECH_TEXTURE_SPARSE_TOKENS = frozenset(
    (
        "ぁ",
        "あ",
        "ぃ",
        "い",
        "ぅ",
        "う",
        "ぇ",
        "え",
        "ぉ",
        "お",
        "ァ",
        "ア",
        "ィ",
        "イ",
        "ゥ",
        "ウ",
        "ェ",
        "エ",
        "ォ",
        "オ",
        "ん",
        "ン",
        "っ",
        "ッ",
        "は",
        "ハ",
        "ふ",
        "フ",
        "声",
        "音",
        "息",
        "呼吸",
        "吐息",
        "再",
        "び",
        "ビ",
    )
)
NON_SPEECH_TEXTURE_PUNCTUATION = frozenset("…♪♡❤・ーｰ〜～")
NON_SPEECH_TEXTURE_NGRAM_MIN_LEN = 2
NON_SPEECH_TEXTURE_NGRAM_MAX_LEN = 5
NON_SPEECH_TEXTURE_NGRAM_MIN_REPEATS = 8
NON_SPEECH_TEXTURE_NGRAM_MIN_TOTAL_LEN = 30
NON_SPEECH_TEXTURE_NGRAM_MIN_COVERAGE = 0.78
MIXED_TEXTURE_SPEECH_CUES = (
    "気持ちいい",
    "きもちいい",
    "キモチいい",
    "いく",
    "イク",
    "だめ",
    "ダメ",
    "もっと",
    "して",
    "した",
    "ください",
    "なさい",
)
SPARSE_SPEECH_NATURAL_CUES = (
    "です",
    "ます",
    "まま",
    "なさい",
    "ください",
    "して",
    "した",
    "いる",
    "ない",
    "反応",
    "姿勢",
    "吸って",
    "吐いて",
)
SPARSE_SPEECH_PARTICLE_CUES = frozenset("のにをがはへともで")
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
JAPANESE_REPEAT_GRAMMAR_FRAME_RE = re.compile(
    r"(?:ているのか|いるのか|なのか|のか|でしょうか|ですか|ますか)$"
)
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
RVC_INSUFFICIENT_TRAINING_DATA_PREFIX = "train-rvc insufficient clean source voice data:"
RVC_TRAIN_AUGMENT_METHODS = ("gain_minus3", "gain_plus3", "highpass_80", "lowpass_7600")


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


def _model_boundary_text_for_language(text: str, language: str | None) -> tuple[str, str, list[str]]:
    original = str(text or "").strip()
    if _canonical_language(language) != "ja":
        return original, original, []
    normalized = normalize_japanese_kana_text(original)
    return normalized.text, normalized.original_text, normalized.risk_flags


def _source_script_text_audit_fields(segment: Segment) -> dict[str, Any]:
    if segment.source_script is None or not segment.source_script.text.strip():
        return {}
    text, original, flags = _model_boundary_text_for_language(
        segment.source_script.text,
        segment.source_script.language,
    )
    return {
        "source_text": text,
        "source_text_original": original,
        "source_text_language": _canonical_language(segment.source_script.language),
        "text_normalization": {
            "policy": "ja_hiragana"
            if _canonical_language(segment.source_script.language) == "ja"
            else "none",
            "risk_flags": flags,
        },
    }


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
    speaker_cfg = cfg.gsv_speaker_models.get(segment.speaker_id)
    if speaker_cfg is not None:
        return speaker_cfg
    fallback_id = _gsv_model_fallback_speaker_id(segment)
    if fallback_id:
        return cfg.gsv_speaker_models.get(fallback_id)
    return None


def _gsv_model_fallback_speaker_id(segment: Segment) -> str | None:
    fallback = segment.analysis.get("source_speaker_model_fallback")
    if not isinstance(fallback, dict):
        return None
    speaker_id = fallback.get("speaker_id")
    return str(speaker_id) if speaker_id else None


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
        speaker_cfg = _gsv_speaker_cfg(cfg, segment)
        effective_speaker_id = (
            segment.speaker_id
            if segment.speaker_id in cfg.gsv_speaker_models
            else _gsv_model_fallback_speaker_id(segment)
        )
        if not segment.speaker_id or speaker_cfg is None or not effective_speaker_id:
            missing.append(segment.id)
            continue
        if not speaker_cfg.gpt_weights_path:
            errors.append(f"{effective_speaker_id} GPT weights missing: <not configured>")
        for label, raw_path in (
            ("GPT weights", speaker_cfg.gpt_weights_path),
            ("SoVITS weights", speaker_cfg.sovits_weights_path),
            ("refs", speaker_cfg.refs_path),
        ):
            if not raw_path:
                continue
            path = _resolve_gsv_speaker_path(project_dir, raw_path)
            if not path.exists():
                errors.append(f"{effective_speaker_id} {label} missing: {path}")
    if missing:
        errors.append("missing speaker model mapping for segments: " + ", ".join(missing[:20]))
    if errors:
        raise ValueError("Invalid GPT-SoVITS voice bank speaker models: " + "; ".join(errors))


def _ref_for_tts_language(ref: GPTSoVITSRef, tts_language: str) -> GPTSoVITSRef:
    _ = tts_language
    return ref


def _segment_ref_overlap_ratio(analysis: dict[str, Any]) -> float | None:
    ratios: list[float] = []
    assignment = analysis.get("source_speaker_assignment")
    if isinstance(assignment, dict):
        for key in ("routing_overlap_ratio", "dominant_overlap_ratio"):
            value = assignment.get(key)
            if isinstance(value, bool):
                continue
            if isinstance(value, (float, int)):
                ratios.append(float(value))
    routing = analysis.get("source_speaker_routing")
    if isinstance(routing, dict):
        value = routing.get("overlap_ratio")
        if isinstance(value, (float, int)) and not isinstance(value, bool):
            ratios.append(float(value))
    return max(ratios) if ratios else None


def _segment_ref_relaxed_training_reason(segment: Segment, cfg: ProjectConfig) -> str | None:
    voice_training = segment.analysis.get("voice_training")
    if not isinstance(voice_training, dict) or voice_training.get("exclude") is not True:
        return None
    reason = str(voice_training.get("reason") or "").strip()
    allowed = {
        str(item).strip()
        for item in getattr(cfg, "gsv_segment_ref_relaxed_training_reasons", [])
        if str(item).strip()
    }
    return reason if reason in allowed else None


def _segment_ref_relaxation_payload(
    segment: Segment,
    cfg: ProjectConfig,
) -> dict[str, Any] | None:
    reason = _segment_ref_relaxed_training_reason(segment, cfg)
    if reason is None:
        return None
    min_overlap_ratio = float(getattr(cfg, "gsv_segment_ref_min_overlap_ratio", 0.75))
    overlap_ratio = _segment_ref_overlap_ratio(segment.analysis)
    if overlap_ratio is None or overlap_ratio < min_overlap_ratio:
        return {
            "accepted": False,
            "reason": reason,
            "overlap_ratio": None if overlap_ratio is None else round(overlap_ratio, 6),
            "min_overlap_ratio": round(min_overlap_ratio, 6),
        }
    return {
        "accepted": True,
        "reason": reason,
        "overlap_ratio": round(overlap_ratio, 6),
        "min_overlap_ratio": round(min_overlap_ratio, 6),
    }


def _segment_ref_reject_reasons_after_relaxation(
    segment: Segment,
    cfg: ProjectConfig,
    reject_reasons: Sequence[str],
    metadata: dict[str, Any],
) -> list[str]:
    reasons = list(dict.fromkeys(str(reason) for reason in reject_reasons if str(reason)))
    if reasons != ["manual_training_exclude"]:
        return reasons
    relaxation = _segment_ref_relaxation_payload(segment, cfg)
    if relaxation is None:
        return reasons
    if not relaxation["accepted"]:
        metadata["relaxed_training_rejected"] = {
            "reason": relaxation["reason"],
            "overlap_ratio": relaxation["overlap_ratio"],
            "min_overlap_ratio": relaxation["min_overlap_ratio"],
        }
        return reasons
    metadata["relaxed_training_exclusion"] = relaxation["reason"]
    metadata["relaxed_training_overlap_ratio"] = relaxation["overlap_ratio"]
    metadata["relaxed_training_min_overlap_ratio"] = relaxation["min_overlap_ratio"]
    return []


def _segment_ref_audio_duration_reject_reasons(
    *,
    metrics: AudioQualityMetrics | None,
    expected_duration_sec: float,
    min_sec: float,
    max_sec: float,
) -> list[str]:
    if metrics is None:
        return []
    actual = float(metrics.duration_sec)
    reasons: list[str] = []
    if abs(actual - expected_duration_sec) > 0.10:
        reasons.append(
            f"audio_duration_mismatch:{actual:.3f}!={expected_duration_sec:.3f}"
        )
    if actual < min_sec:
        reasons.append(f"actual_duration_below_ref_min:{actual:.3f}<{min_sec:.3f}")
    if actual > max_sec:
        reasons.append(f"actual_duration_above_ref_max:{actual:.3f}>{max_sec:.3f}")
    return reasons


def _segment_source_ref_for_gsv(
    project_dir: Path,
    segment: Segment,
    cfg: ProjectConfig,
    all_segments: Sequence[Segment] | None = None,
) -> tuple[GPTSoVITSRef | None, dict[str, Any]]:
    mode = getattr(cfg, "gsv_ref_mode", "static")
    metadata: dict[str, Any] = {
        "mode": mode,
        "segment_id": segment.id,
        "used": False,
        "reject_reasons": [],
    }
    if mode not in {"segment", "auto"}:
        metadata["reject_reasons"] = ["ref_mode_static"]
        return None, metadata
    source_language = _canonical_language(getattr(cfg, "source_language", "ja"))
    target_language = _canonical_language(getattr(cfg, "target_language", "ko"))
    if mode == "auto" and source_language == target_language:
        metadata["reject_reasons"] = ["auto_mode_same_source_and_target_language"]
        return None, metadata
    if (
        target_language == "ko"
        and source_language != target_language
        and not bool(getattr(cfg, "gsv_korean_segment_ref_enabled", True))
    ):
        metadata["reject_reasons"] = [
            "korean_segment_ref_disabled_for_pronunciation_priority"
        ]
        metadata["korean_ref_policy"] = "static_for_pronunciation_priority"
        return None, metadata
    effective_ref_min_sec = max(float(cfg.gsv_ref_min_sec), GSV_API_MIN_REF_SEC)
    if segment.duration < effective_ref_min_sec:
        span = _build_segment_neighbor_ref_span(
            all_segments or (),
            segment,
            source_language=source_language,
            cfg=cfg,
            min_sec=effective_ref_min_sec,
            max_sec=float(cfg.gsv_ref_max_sec),
            max_gap_sec=float(getattr(cfg, "asr_resegment_merge_gap_sec", 1.0)),
        )
        if span is not None:
            span_ref = _write_segment_neighbor_ref_for_gsv(
                project_dir,
                span,
                segment,
                cfg,
                source_language=source_language,
                target_language=target_language,
                metadata=metadata,
            )
            if span_ref is not None:
                return span_ref, metadata
        reason = (
            "duration_below_gsv_api_ref_min"
            if effective_ref_min_sec > float(cfg.gsv_ref_min_sec)
            else "duration_below_ref_min"
        )
        metadata["reject_reasons"] = [
            f"{reason}:{segment.duration:.3f}<{effective_ref_min_sec:.3f}"
        ]
        return None, metadata
    if segment.duration > cfg.gsv_ref_max_sec:
        metadata["reject_reasons"] = [
            f"duration_above_ref_max:{segment.duration:.3f}>{cfg.gsv_ref_max_sec:.3f}"
        ]
        return None, metadata
    check = evaluate_voice_training_candidate(
        project_dir,
        segment,
        cfg,
        min_quality_score=cfg.gsv_ref_min_quality_score,
        require_source_script=True,
        require_speaker_id=False,
        source_language=source_language,
    )
    reject_reasons = list(check.reject_reasons or ())
    if reject_reasons:
        reject_reasons = _segment_ref_reject_reasons_after_relaxation(
            segment,
            cfg,
            reject_reasons,
            metadata,
        )
    duration_reasons = _segment_ref_audio_duration_reject_reasons(
        metrics=check.metrics,
        expected_duration_sec=segment.duration,
        min_sec=effective_ref_min_sec,
        max_sec=float(cfg.gsv_ref_max_sec),
    )
    if duration_reasons:
        reject_reasons.extend(duration_reasons)
    if reject_reasons or not check.source_audio_path or not segment.source_script:
        metadata["reject_reasons"] = list(dict.fromkeys(reject_reasons or ["missing_segment_reference"]))
        metadata["clean_source_metrics"] = check.clean_source_metrics
        return None, metadata
    prompt_lang = segment.source_script.language or source_language
    prompt_text, prompt_text_original, prompt_text_flags = _model_boundary_text_for_language(
        segment.source_script.text,
        prompt_lang,
    )
    metadata.update(
        {
            "used": True,
            "ref_audio_path": str(check.source_audio_path),
            "prompt_text": prompt_text,
            "prompt_text_original": prompt_text_original,
            "prompt_lang": prompt_lang,
            "prompt_text_normalization": {
                "policy": "ja_hiragana" if _canonical_language(prompt_lang) == "ja" else "none",
                "risk_flags": prompt_text_flags,
            },
            "clean_source_metrics": check.clean_source_metrics,
        }
    )
    return (
        GPTSoVITSRef(
            ref_audio_path=str(check.source_audio_path),
            prompt_text=prompt_text,
            prompt_text_original=prompt_text_original,
            prompt_lang=prompt_lang,
            aux_ref_audio_paths=[],
            source_language=source_language,
            target_language=target_language,
            cross_lingual_role="segment_source_prompt_for_tts",
        ),
        metadata,
    )


def _build_segment_neighbor_ref_span(
    all_segments: Sequence[Segment],
    target: Segment,
    *,
    source_language: str,
    cfg: Any | None = None,
    min_sec: float,
    max_sec: float,
    max_gap_sec: float,
) -> _VoiceRefSpan | None:
    candidates = [
        segment
        for segment in sorted(all_segments, key=lambda item: (item.start, item.end, item.id))
        if _segment_can_seed_voice_ref(segment, source_language)
        and not _segment_ref_neighbor_reject_reasons(segment, cfg)
    ]
    try:
        target_index = next(index for index, segment in enumerate(candidates) if segment.id == target.id)
    except StopIteration:
        return None
    duration = target.duration
    if duration <= 0 or duration >= min_sec or duration > max_sec:
        return None
    start_index = target_index
    end_index = target_index
    while duration < min_sec:
        options: list[tuple[float, int, str, Segment]] = []
        previous_index = start_index - 1
        if previous_index >= 0:
            previous = candidates[previous_index]
            gap = max(0.0, candidates[start_index].start - previous.end)
            if (
                gap <= max_gap_sec
                and duration + previous.duration <= max_sec
                and _segments_can_share_segment_ref(target, previous)
            ):
                options.append((gap, 1, "previous", previous))
        next_index = end_index + 1
        if next_index < len(candidates):
            next_segment = candidates[next_index]
            gap = max(0.0, next_segment.start - candidates[end_index].end)
            if (
                gap <= max_gap_sec
                and duration + next_segment.duration <= max_sec
                and _segments_can_share_segment_ref(target, next_segment)
            ):
                options.append((gap, 0, "next", next_segment))
        if not options:
            return None
        _, _, side, selected = min(options, key=lambda item: (item[0], item[1], item[3].start))
        duration += selected.duration
        if side == "previous":
            start_index -= 1
        else:
            end_index += 1
    return _VoiceRefSpan(tuple(candidates[start_index : end_index + 1]), duration)


def _segment_ref_neighbor_reject_reasons(segment: Segment, cfg: Any | None = None) -> tuple[str, ...]:
    reasons = list(_voice_ref_segment_reject_reasons(segment, cfg))
    if reasons != ["voice_training_exclude"] or cfg is None:
        return tuple(reasons)
    relaxation = _segment_ref_relaxation_payload(segment, cfg)
    if relaxation is not None and relaxation["accepted"]:
        return ()
    return tuple(reasons)


def _segments_can_share_segment_ref(target: Segment, neighbor: Segment) -> bool:
    if target.speaker_id and neighbor.speaker_id and target.speaker_id != neighbor.speaker_id:
        return False
    target_count = _segment_ref_speaker_count(target.analysis)
    neighbor_count = _segment_ref_speaker_count(neighbor.analysis)
    return target_count in {None, 1} and neighbor_count in {None, 1}


def _segment_ref_speaker_count(analysis: dict[str, Any]) -> int | None:
    value = analysis.get("speaker_count")
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _write_segment_neighbor_ref_for_gsv(
    project_dir: Path,
    span: _VoiceRefSpan,
    target: Segment,
    cfg: ProjectConfig,
    *,
    source_language: str,
    target_language: str,
    metadata: dict[str, Any],
) -> GPTSoVITSRef | None:
    span_reject_reasons = _voice_ref_span_reject_reasons(span)
    if span_reject_reasons:
        metadata["reject_reasons"] = span_reject_reasons
        return None
    ref_path = ensure_inside_project(
        project_dir,
        (project_dir / "work" / "gpt_sovits" / "segment_refs" / f"{target.id}_ref.wav").resolve(),
    )
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        _write_voice_ref_span(project_dir, span, ref_path)
        metrics = measure_source_voice_quality(ref_path)
    except Exception as exc:
        metadata["reject_reasons"] = [f"neighbor_ref_write_failed:{exc}"]
        return None
    duration_reasons = _segment_ref_audio_duration_reject_reasons(
        metrics=metrics,
        expected_duration_sec=span.duration,
        min_sec=max(float(cfg.gsv_ref_min_sec), GSV_API_MIN_REF_SEC),
        max_sec=float(cfg.gsv_ref_max_sec),
    )
    if duration_reasons:
        metadata["reject_reasons"] = duration_reasons
        metadata["clean_source_metrics"] = metrics.as_payload()
        return None
    if metrics.score < cfg.gsv_ref_min_quality_score:
        metadata["reject_reasons"] = [
            f"quality_score_below_ref_min:{metrics.score:.3f}<{cfg.gsv_ref_min_quality_score:.3f}"
        ]
        metadata["clean_source_metrics"] = metrics.as_payload()
        return None
    prompt_lang = (
        span.segments[0].source_script.language
        if span.segments[0].source_script
        else source_language
    )
    prompt_text, prompt_text_original, prompt_text_flags = _model_boundary_text_for_language(
        _voice_ref_span_prompt_text(span),
        prompt_lang,
    )
    metadata.update(
        {
            "used": True,
            "expanded_with_neighbors": True,
            "ref_audio_path": str(ref_path),
            "prompt_text": prompt_text,
            "prompt_text_original": prompt_text_original,
            "prompt_lang": prompt_lang,
            "prompt_text_normalization": {
                "policy": "ja_hiragana" if _canonical_language(prompt_lang) == "ja" else "none",
                "risk_flags": prompt_text_flags,
            },
            "span_segment_ids": [segment.id for segment in span.segments],
            "span_duration_sec": round(span.duration, 6),
            "clean_source_metrics": metrics.as_payload(),
        }
    )
    return GPTSoVITSRef(
        ref_audio_path=str(ref_path),
        prompt_text=prompt_text,
        prompt_text_original=prompt_text_original,
        prompt_lang=prompt_lang,
        aux_ref_audio_paths=[],
        source_language=source_language,
        target_language=target_language,
        cross_lingual_role="segment_neighbor_span_source_prompt_for_tts",
    )


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


def _clipped_asr_words(chunk: ASRChunk, *, start: float, end: float) -> list[Any]:
    clipped: list[Any] = []
    for word in chunk.words:
        text = word.text.strip()
        word_start = max(start, float(word.start))
        word_end = min(end, float(word.end))
        if not text or word_end <= word_start:
            continue
        clipped.append(word.model_copy(update={"start": word_start, "end": word_end, "text": text}))
    return clipped


@dataclass(frozen=True)
class ASRBoundaryClipResult:
    chunks: list[ASRChunk]
    boundary_clipped: bool = False
    reject_reason: str | None = None
    dropped_word_count: int = 0


def _asr_word_text_needs_separator(left: str, right: str) -> bool:
    if not left or not right:
        return False
    left_alnum = any(char.isascii() and char.isalnum() for char in left)
    right_alnum = any(char.isascii() and char.isalnum() for char in right)
    return left_alnum or right_alnum


def _asr_text_from_words(words: Sequence[Any]) -> str:
    tokens = [str(word.text or "").strip() for word in words if str(word.text or "").strip()]
    if not tokens:
        return ""
    text = tokens[0]
    previous = tokens[0]
    for token in tokens[1:]:
        separator = " " if _asr_word_text_needs_separator(previous, token) else ""
        text += separator + token
        previous = token
    return text.strip()


def _clip_asr_chunks_to_window(
    chunks: Sequence[ASRChunk],
    *,
    clip_start: float,
    clip_end: float,
    window_start: float,
    window_end: float,
    require_word_timestamps_for_boundary: bool = True,
    boundary_epsilon_sec: float = 0.02,
) -> ASRBoundaryClipResult:
    clipped_chunks: list[ASRChunk] = []
    boundary_clipped = False
    dropped_word_count = 0
    clip_crosses_window = (
        clip_start < window_start - boundary_epsilon_sec
        or clip_end > window_end + boundary_epsilon_sec
    )
    for chunk in sorted(chunks, key=lambda item: (float(item.start), float(item.end))):
        raw_text = chunk.text.strip()
        if not raw_text:
            continue
        abs_start = clip_start + float(chunk.start)
        abs_end = clip_start + float(chunk.end)
        overlaps_window = (
            min(abs_end, window_end) - max(abs_start, window_start)
        ) > boundary_epsilon_sec
        crosses_boundary = (
            abs_start < window_start - boundary_epsilon_sec
            or abs_end > window_end + boundary_epsilon_sec
        )
        if not overlaps_window:
            boundary_clipped = True
            continue
        if chunk.words:
            kept_words: list[Any] = []
            for word in sorted(chunk.words, key=lambda item: (float(item.start), float(item.end))):
                word_text = word.text.strip()
                if not word_text:
                    continue
                word_start = clip_start + float(word.start)
                word_end = clip_start + float(word.end)
                if word_end <= word_start:
                    continue
                midpoint = (word_start + word_end) / 2.0
                if midpoint < window_start or midpoint > window_end:
                    dropped_word_count += 1
                    boundary_clipped = True
                    continue
                clipped_start = max(window_start, word_start)
                clipped_end = min(window_end, word_end)
                if clipped_end <= clipped_start:
                    dropped_word_count += 1
                    boundary_clipped = True
                    continue
                if clipped_start != word_start or clipped_end != word_end:
                    boundary_clipped = True
                kept_words.append(
                    word.model_copy(
                        update={
                            "start": round(clipped_start, 6),
                            "end": round(clipped_end, 6),
                            "text": word_text,
                        }
                    )
                )
            if not kept_words:
                continue
            rebuilt_text = _asr_text_from_words(kept_words)
            if not rebuilt_text:
                continue
            if rebuilt_text != raw_text:
                boundary_clipped = True
            clipped_chunks.append(
                chunk.model_copy(
                    update={
                        "start": round(float(kept_words[0].start), 6),
                        "end": round(float(kept_words[-1].end), 6),
                        "text": rebuilt_text,
                        "words": kept_words,
                    }
                )
            )
            continue
        if require_word_timestamps_for_boundary and clip_crosses_window and crosses_boundary:
            return ASRBoundaryClipResult(
                chunks=[],
                boundary_clipped=True,
                reject_reason="boundary_clipping_requires_word_timestamps",
                dropped_word_count=dropped_word_count,
            )
        start = max(window_start, abs_start)
        end = min(window_end, abs_end)
        if end - start <= boundary_epsilon_sec:
            boundary_clipped = True
            continue
        if crosses_boundary:
            boundary_clipped = True
        clipped_chunks.append(
            chunk.model_copy(
                update={
                    "start": round(start, 6),
                    "end": round(end, 6),
                    "text": raw_text,
                    "words": [],
                }
            )
        )
    return ASRBoundaryClipResult(
        chunks=clipped_chunks,
        boundary_clipped=boundary_clipped,
        reject_reason=None,
        dropped_word_count=dropped_word_count,
    )


def _tighten_sparse_asr_chunk_timing_from_words(
    chunk: ASRChunk,
    *,
    sparse_chunk_min_chars_per_sec: float,
    audio_duration_sec: float | None,
    sparse_timing_min_sec: float = 12.0,
    padding_sec: float = 0.4,
) -> ASRChunk:
    if not chunk.words or sparse_chunk_min_chars_per_sec <= 0:
        return chunk
    text = chunk.text.strip()
    duration = max(0.001, float(chunk.end) - float(chunk.start))
    if duration < sparse_timing_min_sec or len(text) / duration >= sparse_chunk_min_chars_per_sec:
        return chunk
    word_starts = [float(word.start) for word in chunk.words if word.text.strip()]
    word_ends = [float(word.end) for word in chunk.words if word.text.strip()]
    if not word_starts or not word_ends:
        return chunk
    word_start = max(float(chunk.start), min(word_starts))
    word_end = min(float(chunk.end), max(word_ends))
    if word_end <= word_start:
        return chunk
    start = max(0.0, word_start - padding_sec)
    end = word_end + padding_sec
    if audio_duration_sec is not None:
        end = min(float(audio_duration_sec), end)
    end = max(start + 0.05, end)
    if end - start >= duration * 0.8:
        return chunk
    return chunk.model_copy(
        update={
            "start": start,
            "end": end,
            "words": _clipped_asr_words(chunk, start=start, end=end),
        }
    )


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
        normalized = chunk.model_copy(
            update={
                "start": start,
                "end": end,
                "text": text,
                "words": _clipped_asr_words(chunk, start=start, end=end),
            }
        )
        normalized = _tighten_sparse_asr_chunk_timing_from_words(
            normalized,
            sparse_chunk_min_chars_per_sec=sparse_chunk_min_chars_per_sec,
            audio_duration_sec=audio_duration_sec,
        )
        duration = normalized.end - normalized.start
        if (
            duration > sparse_chunk_max_sec
            and sparse_chunk_min_chars_per_sec > 0
            and len(text) / duration < sparse_chunk_min_chars_per_sec
        ):
            continue
        for word_split in _split_asr_chunk_on_word_gaps(normalized):
            valid.extend(_split_long_asr_chunk(word_split, max_chunk_sec=max_chunk_sec))
    return valid


def _split_long_asr_chunk(chunk: ASRChunk, *, max_chunk_sec: float) -> list[ASRChunk]:
    duration = chunk.end - chunk.start
    if duration <= max_chunk_sec:
        return [chunk]
    text = chunk.text.strip()
    if not text:
        return []
    desired_part_count = max(1, int(math.ceil(duration / max_chunk_sec)))
    part_count = min(desired_part_count, len(text))
    if part_count <= 1:
        return [chunk]
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
        split_chunks.append(
            chunk.model_copy(
                update={
                    "start": start,
                    "end": end,
                    "text": part,
                    "words": _clipped_asr_words(chunk, start=start, end=end),
                }
            )
        )
    return split_chunks


def _split_asr_chunk_on_word_gaps(
    chunk: ASRChunk,
    *,
    min_gap_sec: float = 0.8,
) -> list[ASRChunk]:
    words = [
        word
        for word in sorted(chunk.words, key=lambda item: (float(item.start), float(item.end)))
        if word.text.strip() and float(word.end) > float(word.start)
    ]
    if len(words) <= 1:
        return [chunk]
    split_indexes = [
        index
        for index in range(1, len(words))
        if float(words[index].start) - float(words[index - 1].end) >= min_gap_sec
    ]
    if not split_indexes:
        return [chunk]

    groups: list[list[Any]] = []
    start_index = 0
    for split_index in split_indexes:
        groups.append(words[start_index:split_index])
        start_index = split_index
    groups.append(words[start_index:])

    split_chunks: list[ASRChunk] = []
    for group in groups:
        text = "".join(word.text.strip() for word in group if word.text.strip()).strip()
        if not text:
            continue
        start = max(float(chunk.start), float(group[0].start))
        end = min(float(chunk.end), float(group[-1].end))
        if end - start <= 0.05:
            continue
        split_chunks.append(
            chunk.model_copy(
                update={
                    "start": start,
                    "end": end,
                    "text": text,
                    "words": [word.model_copy(update={"text": word.text.strip()}) for word in group],
                }
            )
        )
    return split_chunks if len(split_chunks) > 1 else [chunk]


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


def _asr_group_text(group: list[ASRChunk]) -> str:
    return " ".join(chunk.text.strip() for chunk in group if chunk.text.strip()).strip()


def _segment_text_density(text: str, duration: float) -> float:
    compact = re.sub(r"[\s　、。,.，．!！?？…・♪♡❤（）()\[\]「」『』]+", "", text)
    return len(compact) / max(0.001, duration)


def _segment_is_sparse_edge_candidate(segment: Segment, cfg: Any) -> bool:
    source_script = segment.source_script
    if source_script is None:
        return False
    text = source_script.text.strip()
    if not text or _asr_text_looks_non_speech_texture(text, duration=segment.duration):
        return False
    compact = re.sub(r"[\s　、。,.，．!！?？…・♪♡❤（）()\[\]「」『』]+", "", text)
    sparse_min_sec = float(getattr(cfg, "asr_repair_sparse_min_sec", 12.0))
    short_sparse_fragment = segment.duration >= 8.0 and len(compact) <= 2
    if segment.duration < sparse_min_sec and not short_sparse_fragment:
        return False
    min_density = float(getattr(cfg, "asr_repair_sparse_min_chars_per_sec", 1.0))
    return _segment_text_density(text, segment.duration) < min_density


def _segment_audio_activity_spans(
    audio_path: Path,
    *,
    threshold_db: float,
    frame_ms: float = 50.0,
    merge_gap_sec: float = 0.35,
) -> list[tuple[float, float]]:
    data, sample_rate = load_audio(audio_path)
    mono = to_mono(data)
    frame = max(1, int(round(sample_rate * frame_ms / 1000.0)))
    threshold = 10 ** (threshold_db / 20.0)
    active: list[tuple[float, float]] = []
    for start in range(0, len(mono), frame):
        chunk = mono[start : start + frame]
        if len(chunk) == 0:
            continue
        rms = float(np.sqrt(np.mean(np.square(chunk))))
        if rms <= threshold:
            continue
        active.append((start / sample_rate, min(len(mono), start + frame) / sample_rate))
    if not active:
        return []
    merged: list[tuple[float, float]] = []
    for start, end in active:
        if merged and start - merged[-1][1] <= merge_gap_sec:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def _texts_for_activity_spans(text: str, span_count: int) -> list[str] | None:
    if span_count <= 0:
        return None
    text = text.strip()
    if span_count == 1:
        return [text]
    tokens = [token for token in re.split(r"[\s　]+", text) if token]
    if len(tokens) < span_count:
        return None
    if len(tokens) == span_count:
        return tokens
    groups: list[str] = []
    for index in range(span_count):
        start = round(len(tokens) * index / span_count)
        end = round(len(tokens) * (index + 1) / span_count)
        groups.append(" ".join(tokens[start:end]).strip())
    return groups if all(groups) else None


def _copy_source_script_for_span(
    source_script: SourceScript,
    *,
    start: float,
    end: float,
    text: str | None = None,
    backend: str | None = None,
) -> SourceScript:
    return source_script.model_copy(
        update={
            "text": source_script.text.strip() if text is None else text,
            "start": start,
            "end": end,
            "backend": source_script.backend if backend is None else backend,
        }
    )


def _make_segment_like(
    segment: Segment,
    *,
    start: float,
    end: float,
    source_script: SourceScript | None,
    status: str = "raw",
    errors: list[str] | None = None,
    analysis: dict[str, Any] | None = None,
) -> Segment:
    start = round(start, 3)
    end = round(max(start + 0.05, end), 3)
    return segment.model_copy(
        update={
            "start": start,
            "end": end,
            "duration": round(end - start, 3),
            "source_script": source_script,
            "status": status,
            "errors": list(errors or []),
            "analysis": dict(analysis or {}),
            "script": None,
            "translation_ko": None,
            "tts": None,
            "rvc": None,
            "qc": None,
            "mix": {},
        }
    )


def _split_sparse_segment_by_activity(
    segment: Segment,
    *,
    cfg: Any,
) -> list[Segment] | None:
    source_script = segment.source_script
    if source_script is None or not _segment_is_sparse_edge_candidate(segment, cfg):
        return None
    audio_path = Path(segment.audio_for_gemma)
    if not audio_path.exists():
        return None
    threshold_db = float(getattr(cfg, "asr_segmentation_silence_db", -45.0))
    spans = _segment_audio_activity_spans(audio_path, threshold_db=threshold_db)
    if not spans:
        return None
    active_total = sum(end - start for start, end in spans)
    if active_total / max(0.001, segment.duration) > 0.35:
        return None
    texts = _texts_for_activity_spans(source_script.text, len(spans))
    if texts is None:
        return None
    pieces: list[Segment] = []
    cursor = float(segment.start)
    pad_sec = 0.25
    min_texture_sec = 1.0
    for (active_start, active_end), text in zip(spans, texts, strict=True):
        speech_start = max(float(segment.start), float(segment.start) + active_start - pad_sec)
        speech_end = min(float(segment.end), float(segment.start) + active_end + pad_sec)
        if speech_start - cursor >= min_texture_sec:
            texture_script = SourceScript(
                text="…",
                language=source_script.language,
                backend="audio_energy",
                start=cursor,
                end=speech_start,
                confidence=None,
            )
            pieces.append(
                _make_segment_like(
                    segment,
                    start=cursor,
                    end=speech_start,
                    source_script=texture_script,
                    status="non_speech_texture",
                    errors=["asr_non_speech_texture"],
                )
            )
        speech_script = _copy_source_script_for_span(
            source_script,
            start=speech_start,
            end=speech_end,
            text=text,
        )
        pieces.append(
            _make_segment_like(
                segment,
                start=speech_start,
                end=speech_end,
                source_script=speech_script,
                analysis={"asr_sparse_edge_speech": True},
            )
        )
        cursor = speech_end
    if float(segment.end) - cursor >= min_texture_sec:
        texture_script = SourceScript(
            text="…",
            language=source_script.language,
            backend="audio_energy",
            start=cursor,
            end=float(segment.end),
            confidence=None,
        )
        pieces.append(
            _make_segment_like(
                segment,
                start=cursor,
                end=float(segment.end),
                source_script=texture_script,
                status="non_speech_texture",
                errors=["asr_non_speech_texture"],
            )
        )
    return pieces if pieces else None


def _merge_source_scripts(left: SourceScript, right: SourceScript, *, start: float, end: float) -> SourceScript:
    left_text = left.text.strip()
    right_text = right.text.strip()
    confidence: float | None = None
    if left.confidence is not None or right.confidence is not None:
        left_duration = max(0.0, left.end - left.start)
        right_duration = max(0.0, right.end - right.start)
        weighted: list[tuple[float, float]] = []
        if left.confidence is not None:
            weighted.append((float(left.confidence), left_duration))
        if right.confidence is not None:
            weighted.append((float(right.confidence), right_duration))
        total = sum(weight for _, weight in weighted)
        confidence = (
            sum(value * weight for value, weight in weighted) / total
            if total > 0
            else left.confidence or right.confidence
        )
    return left.model_copy(
        update={
            "text": " ".join(part for part in (left_text, right_text) if part),
            "start": start,
            "end": end,
            "confidence": confidence,
        }
    )


def _merge_sparse_edge_speech_segments(
    segments: list[Segment],
    *,
    merge_gap_sec: float,
    max_segment_sec: float,
) -> list[Segment]:
    merged = [segment.model_copy(deep=True) for segment in segments]
    index = 0
    while index < len(merged):
        segment = merged[index]
        if not segment.analysis.get("asr_sparse_edge_speech") or segment.source_script is None:
            index += 1
            continue
        candidates: list[tuple[float, str, int]] = []
        if index > 0 and merged[index - 1].status not in NO_SPEECH_STATUSES and merged[index - 1].source_script:
            left = merged[index - 1]
            gap = segment.start - left.end
            if 0 <= gap <= merge_gap_sec and segment.end - left.start <= max_segment_sec:
                candidates.append((gap, "left", index - 1))
        if (
            index + 1 < len(merged)
            and merged[index + 1].status not in NO_SPEECH_STATUSES
            and merged[index + 1].source_script
        ):
            right = merged[index + 1]
            gap = right.start - segment.end
            if 0 <= gap <= merge_gap_sec and right.end - segment.start <= max_segment_sec:
                candidates.append((gap, "right", index + 1))
        if not candidates:
            segment.analysis.pop("asr_sparse_edge_speech", None)
            index += 1
            continue
        _, side, target_index = min(candidates, key=lambda item: (item[0], item[1] != "left"))
        if side == "left":
            target = merged[target_index]
            assert target.source_script is not None
            source_script = _merge_source_scripts(
                target.source_script,
                segment.source_script,
                start=target.start,
                end=segment.end,
            )
            merged[target_index] = target.model_copy(
                update={
                    "end": segment.end,
                    "duration": round(segment.end - target.start, 3),
                    "source_script": source_script,
                }
            )
            del merged[index]
            index = max(0, target_index)
        else:
            target = merged[target_index]
            assert target.source_script is not None
            source_script = _merge_source_scripts(
                segment.source_script,
                target.source_script,
                start=segment.start,
                end=target.end,
            )
            merged[target_index] = target.model_copy(
                update={
                    "start": segment.start,
                    "duration": round(target.end - segment.start, 3),
                    "source_script": source_script,
                }
            )
            del merged[index]
    return merged


def _renumber_segments_for_project(segments: list[Segment], project_dir: Path) -> list[Segment]:
    audio_base = project_dir / "work" / "segments" / "audio"
    renumbered: list[Segment] = []
    for index, segment in enumerate(segments, start=1):
        seg_id = f"seg_{index:04d}"
        source_script = segment.source_script
        if source_script is not None:
            source_script = source_script.model_copy(
                update={"start": segment.start, "end": segment.end}
            )
        renumbered.append(
            segment.model_copy(
                update={
                    "id": seg_id,
                    "audio_for_gemma": str(audio_base / f"{seg_id}_gemma.wav"),
                    "audio_for_mix": str(audio_base / f"{seg_id}_mix.wav"),
                    "source_script": source_script,
                }
            )
        )
    return renumbered


def _split_sparse_edge_segments_by_audio(
    segments: list[Segment],
    *,
    project_dir: Path,
    cfg: Any,
    merge_gap_sec: float,
) -> list[Segment]:
    pieces: list[Segment] = []
    changed = False
    for segment in segments:
        split = _split_sparse_segment_by_activity(segment, cfg=cfg)
        if split is None:
            pieces.append(segment)
            continue
        pieces.extend(split)
        changed = True
    if not changed:
        return segments
    pieces.sort(key=lambda item: (item.start, item.end))
    pieces = _merge_sparse_edge_speech_segments(
        pieces,
        merge_gap_sec=merge_gap_sec,
        max_segment_sec=float(getattr(cfg, "asr_resegment_max_sec", 10.0)),
    )
    return _renumber_segments_for_project(pieces, project_dir)


def _embedded_source_countdown_matches(text: str) -> list[tuple[int, str]] | None:
    matches = source_countdown_token_matches(text)
    if not matches:
        return None
    values = [value for value, _raw, _start, _end in matches]
    if not is_descending_countdown(values):
        return None
    return [(value, raw) for value, raw, _start, _end in matches]


def _embedded_source_countdown_values(text: str) -> list[int] | None:
    matches = _embedded_source_countdown_matches(text)
    if matches is None:
        return None
    return [value for value, _raw in matches]


def _asr_group_countdown_values(group: list[ASRChunk]) -> list[int] | None:
    text = _asr_group_text(group)
    values = source_countdown_values(text)
    if values is None:
        values = _embedded_source_countdown_values(text)
    if values is None or not values:
        return None
    return values


def _merge_countdown_asr_groups(
    groups: list[list[ASRChunk]],
    *,
    enabled: bool,
    merge_gap_sec: float,
    max_span_sec: float,
) -> list[list[ASRChunk]]:
    if not enabled or not groups:
        return groups
    merged: list[list[ASRChunk]] = []
    current: list[list[ASRChunk]] = []
    current_values: list[int] = []

    def flush_current() -> None:
        nonlocal current, current_values
        if current and is_descending_countdown(current_values):
            merged.append([chunk for group in current for chunk in group])
        else:
            merged.extend(current)
        current = []
        current_values = []

    for group in groups:
        values = _asr_group_countdown_values(group)
        if values is None:
            flush_current()
            merged.append(group)
            continue
        if not current:
            current = [group]
            current_values = list(values)
            continue
        gap = group[0].start - current[-1][-1].end
        span = group[-1].end - current[0][0].start
        candidate_values = [*current_values, *values]
        if (
            gap <= merge_gap_sec
            and span <= max_span_sec
            and is_descending_countdown_prefix(candidate_values)
        ):
            current.append(group)
            current_values = candidate_values
            continue
        flush_current()
        current = [group]
        current_values = list(values)
    flush_current()
    return merged


def _absorb_micro_asr_groups_for_tts(
    groups: list[list[ASRChunk]],
    *,
    merge_gap_sec: float,
    max_segment_sec: float,
    micro_max_sec: float = 0.6,
) -> list[list[ASRChunk]]:
    if len(groups) <= 1:
        return groups
    absorbed = [list(group) for group in groups]
    absorb_gap_sec = max(float(merge_gap_sec), 1.5)
    index = 0
    while index < len(absorbed):
        group = absorbed[index]
        duration = group[-1].end - group[0].start
        if duration <= 0 or duration > micro_max_sec or _asr_group_countdown_values(group):
            index += 1
            continue

        candidates: list[tuple[float, str, int]] = []
        if index > 0:
            left = absorbed[index - 1]
            gap = group[0].start - left[-1].end
            combined_duration = group[-1].end - left[0].start
            if 0 <= gap <= absorb_gap_sec and combined_duration <= max_segment_sec:
                candidates.append((gap, "left", index - 1))
        if index + 1 < len(absorbed):
            right = absorbed[index + 1]
            gap = right[0].start - group[-1].end
            combined_duration = right[-1].end - group[0].start
            if 0 <= gap <= absorb_gap_sec and combined_duration <= max_segment_sec:
                candidates.append((gap, "right", index + 1))
        if not candidates:
            index += 1
            continue

        _, side, target_index = min(candidates, key=lambda item: (item[0], item[1] != "left"))
        if side == "left":
            absorbed[target_index].extend(group)
            del absorbed[index]
            index = max(0, target_index)
        else:
            absorbed[target_index] = [*group, *absorbed[target_index]]
            del absorbed[index]
    return absorbed


def _absorb_short_asr_groups_for_tts(
    groups: list[list[ASRChunk]],
    *,
    min_segment_sec: float,
    merge_gap_sec: float,
    max_segment_sec: float,
) -> list[list[ASRChunk]]:
    if len(groups) <= 1 or min_segment_sec <= 0:
        return groups
    absorbed = [list(group) for group in groups]
    absorb_gap_sec = max(0.0, float(merge_gap_sec))
    index = 0
    while index < len(absorbed):
        group = absorbed[index]
        duration = group[-1].end - group[0].start
        if duration <= 0 or duration >= min_segment_sec or _asr_group_countdown_values(group):
            index += 1
            continue

        candidates: list[tuple[float, str, int]] = []
        if index > 0:
            left = absorbed[index - 1]
            gap = group[0].start - left[-1].end
            combined_duration = group[-1].end - left[0].start
            if 0 <= gap <= absorb_gap_sec and combined_duration <= max_segment_sec:
                candidates.append((gap, "left", index - 1))
        if index + 1 < len(absorbed):
            right = absorbed[index + 1]
            gap = right[0].start - group[-1].end
            combined_duration = right[-1].end - group[0].start
            if 0 <= gap <= absorb_gap_sec and combined_duration <= max_segment_sec:
                candidates.append((gap, "right", index + 1))
        if not candidates:
            index += 1
            continue

        _, side, target_index = min(candidates, key=lambda item: (item[0], item[1] != "left"))
        if side == "left":
            absorbed[target_index].extend(group)
            del absorbed[index]
            index = max(0, target_index)
        else:
            absorbed[target_index] = [*group, *absorbed[target_index]]
            del absorbed[index]
    return absorbed


def _asr_word_countdown_timeline(
    group: list[ASRChunk],
    values: list[int],
) -> list[dict[str, Any]] | None:
    tokens = countdown_korean_tokens(values)
    if tokens is None:
        return None
    timeline: list[dict[str, Any]] = []
    for chunk in sorted(group, key=lambda item: (item.start, item.end)):
        for word in sorted(chunk.words, key=lambda item: (item.start, item.end)):
            source_text = word.text.strip()
            if not source_text:
                continue
            word_values = source_countdown_values(source_text.strip(".,、。"))
            if word_values is None or len(word_values) != 1:
                continue
            start = float(word.start)
            end = float(word.end)
            if end <= start:
                continue
            item: dict[str, Any] = {
                "value": word_values[0],
                "source_text": source_text,
                "start": round(start, 6),
                "end": round(end, 6),
            }
            if word.confidence is not None:
                item["confidence"] = round(float(word.confidence), 6)
            timeline.append(item)
    if [int(item["value"]) for item in timeline] != values:
        return _asr_chunk_countdown_timeline(group, values)
    for item, token in zip(timeline, tokens, strict=True):
        item["korean_token"] = token
    return timeline


def _asr_chunk_countdown_timeline(
    group: list[ASRChunk],
    values: list[int],
) -> list[dict[str, Any]] | None:
    tokens = countdown_korean_tokens(values)
    if tokens is None:
        return None
    timeline: list[dict[str, Any]] = []
    for chunk in sorted(group, key=lambda item: (item.start, item.end)):
        chunk_matches: list[tuple[int, str]]
        chunk_values = source_countdown_values(chunk.text.strip())
        if chunk_values is not None:
            chunk_matches = [(value, str(value)) for value in chunk_values]
        else:
            embedded_matches = _embedded_source_countdown_matches(chunk.text.strip())
            if embedded_matches is None:
                return None
            chunk_matches = embedded_matches
            chunk_values = [value for value, _raw in chunk_matches]
        if not chunk_values:
            return None
        value_count = len(chunk_values)
        if value_count <= 0:
            return None
        duration = max(0.0, float(chunk.end) - float(chunk.start))
        if duration <= 0:
            return None
        for local_index, value in enumerate(chunk_values):
            start = float(chunk.start) + duration * local_index / value_count
            end = float(chunk.start) + duration * (local_index + 1) / value_count
            raw_text = chunk_matches[local_index][1]
            source_text = chunk.text.strip() if value_count == 1 else raw_text
            item: dict[str, Any] = {
                "value": value,
                "source_text": source_text,
                "start": round(start, 6),
                "end": round(end, 6),
            }
            if chunk.confidence is not None:
                item["confidence"] = round(float(chunk.confidence), 6)
            timeline.append(item)
    if [int(item["value"]) for item in timeline] != values:
        return None
    for item, token in zip(timeline, tokens, strict=True):
        item["korean_token"] = token
    return timeline


def _countdown_timeline_summary(segments: Sequence[Segment]) -> dict[str, Any]:
    event_count = 0
    timeline_count = 0
    missing_ids: list[str] = []
    for segment in segments:
        event = segment.analysis.get(COUNTDOWN_EVENT_KEY)
        if not isinstance(event, dict):
            continue
        raw_values = event.get("values")
        if not isinstance(raw_values, list) or not all(isinstance(value, int) for value in raw_values):
            continue
        values = [int(value) for value in raw_values]
        if not is_descending_countdown(values):
            continue
        event_count += 1
        raw_timeline = event.get("token_timeline")
        if isinstance(raw_timeline, list) and len(raw_timeline) == len(values):
            timeline_count += 1
        else:
            missing_ids.append(segment.id)
    return {
        "countdown_event_count": event_count,
        "countdown_token_timeline_count": timeline_count,
        "countdown_token_timeline_missing": len(missing_ids),
        "countdown_token_timeline_missing_ids": missing_ids[:50],
    }


def _warn_missing_countdown_timelines(
    manifest: PipelineManifest,
    summary: Mapping[str, Any],
    *,
    word_timestamps_enabled: bool,
) -> None:
    missing = int(summary.get("countdown_token_timeline_missing", 0) or 0)
    total = int(summary.get("countdown_event_count", 0) or 0)
    if not word_timestamps_enabled or total <= 0 or missing <= 0:
        return
    ids = [str(value) for value in summary.get("countdown_token_timeline_missing_ids", [])]
    suffix = f" ids={','.join(ids[:10])}" if ids else ""
    _manifest_warning_once(
        manifest,
        f"countdown token timeline missing for {missing}/{total} countdown segments; "
        f"TTS will use chunk/equal-slot fallback where needed.{suffix}",
    )


def _countdown_event_payload(
    values: list[int],
    *,
    source_chunk_texts: list[str],
    source_chunk_count: int,
    merge_gap_sec: float,
    max_span_sec: float,
    token_timeline: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    korean_text = countdown_korean_text(values)
    tokens = countdown_korean_tokens(values)
    if korean_text is None or tokens is None:
        return None
    payload = {
        "kind": "descending_countdown",
        "values": values,
        "korean_text": korean_text,
        "korean_tokens": tokens,
        "source_chunk_texts": source_chunk_texts,
        "source_chunk_count": source_chunk_count,
        "merge_gap_sec": merge_gap_sec,
        "max_span_sec": max_span_sec,
        "preserve_wall_clock_span": True,
    }
    if token_timeline:
        payload["token_timeline"] = token_timeline
    return payload


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
    countdown_merge_enabled: bool = True,
    countdown_merge_gap_sec: float = 2.5,
    countdown_merge_max_span_sec: float = 60.0,
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
    groups = _merge_countdown_asr_groups(
        groups,
        enabled=countdown_merge_enabled,
        merge_gap_sec=countdown_merge_gap_sec,
        max_span_sec=countdown_merge_max_span_sec,
    )
    groups = _absorb_micro_asr_groups_for_tts(
        groups,
        merge_gap_sec=merge_gap_sec,
        max_segment_sec=max_segment_sec,
    )
    groups = _absorb_short_asr_groups_for_tts(
        groups,
        min_segment_sec=min_segment_sec,
        merge_gap_sec=merge_gap_sec,
        max_segment_sec=max_segment_sec,
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
        text = _asr_group_text(group)
        language = next((chunk.language for chunk in group if chunk.language), fallback_language)
        confidence = _asr_group_confidence(group)
        audio_base = project_dir / "work" / "segments" / "audio"
        segment = Segment(
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
        countdown_values = _asr_group_countdown_values(group)
        if countdown_values is not None and is_descending_countdown(countdown_values):
            payload = _countdown_event_payload(
                countdown_values,
                source_chunk_texts=[chunk.text.strip() for chunk in group if chunk.text.strip()],
                source_chunk_count=len(group),
                merge_gap_sec=countdown_merge_gap_sec,
                max_span_sec=countdown_merge_max_span_sec,
                token_timeline=_asr_word_countdown_timeline(group, countdown_values),
            )
            if payload is not None:
                segment.analysis[COUNTDOWN_EVENT_KEY] = payload
        segments.append(segment)
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


def _asr_text_has_dominant_non_speech_repetition(compact: str) -> bool:
    if len(compact) < 30:
        return False
    for match in NON_SPEECH_TEXTURE_DOMINANT_REPEAT_RE.finditer(compact):
        repeated = match.group(1)
        if repeated not in NON_SPEECH_TEXTURE_DOMINANT_REPEAT_CHARS:
            continue
        run_length = match.end() - match.start()
        if run_length / max(1, len(compact)) >= 0.55 or len(compact) - run_length <= 16:
            return True
    return False


def _asr_text_has_dominant_ngram_repetition(compact: str) -> bool:
    if len(compact) < NON_SPEECH_TEXTURE_NGRAM_MIN_TOTAL_LEN:
        return False
    for unit_len in range(
        NON_SPEECH_TEXTURE_NGRAM_MIN_LEN,
        NON_SPEECH_TEXTURE_NGRAM_MAX_LEN + 1,
    ):
        limit = len(compact) - (unit_len * NON_SPEECH_TEXTURE_NGRAM_MIN_REPEATS)
        for index in range(0, max(0, limit) + 1):
            unit = compact[index : index + unit_len]
            if not unit.strip() or len(set(unit)) == 1:
                continue
            repeats = 0
            cursor = index
            while compact.startswith(unit, cursor):
                repeats += 1
                cursor += unit_len
            if repeats < NON_SPEECH_TEXTURE_NGRAM_MIN_REPEATS:
                continue
            run_length = repeats * unit_len
            coverage = run_length / max(1, len(compact))
            if coverage >= NON_SPEECH_TEXTURE_NGRAM_MIN_COVERAGE or len(compact) - run_length <= 12:
                return True
    return False


def _asr_mixed_texture_speech_reason(text: str, *, duration: float) -> str | None:
    stripped = text.strip()
    if duration < 4.0:
        return None
    if source_countdown_values(stripped) is not None or NUMERIC_ONLY_SOURCE_RE.fullmatch(stripped):
        return None
    compact = re.sub(r"[\s　、。,.，．!！?？…・♪♡❤]+", "", stripped)
    if len(compact) < 8:
        return None
    if _asr_text_looks_non_speech_texture(stripped, duration=duration):
        return None
    if not any(cue in compact for cue in MIXED_TEXTURE_SPEECH_CUES):
        return None
    texture_chars = sum(1 for char in compact if char in NON_SPEECH_TEXTURE_CHARS)
    texture_ratio = texture_chars / max(1, len(compact))
    marker_count = sum(1 for char in compact if char in NON_SPEECH_TEXTURE_MARKERS)
    if texture_ratio >= 0.55 and marker_count >= 2:
        return "asr_mixed_texture_speech"
    return None


def _compact_short_filler_source_text(text: str) -> str:
    return re.sub(r"[\s　、。,.，．!！?？…・♪♡❤「」『』（）()［］\[\]【】:：;；\"'`]+", "", text.strip())


def _is_repeated_short_filler(compact: str) -> bool:
    for unit in ASR_SHORT_FILLER_KEEP_ORIGINAL_REPEAT_UNITS:
        if len(compact) <= len(unit) or len(compact) % len(unit) != 0:
            continue
        if unit * (len(compact) // len(unit)) == compact:
            return True
    return False


def _source_script_keep_original_texture_candidate(source_script: SourceScript | None) -> dict[str, Any] | None:
    if source_script is None:
        return None
    text = source_script.text.strip()
    duration = max(0.0, float(source_script.end) - float(source_script.start))
    if duration > ASR_SHORT_FILLER_KEEP_ORIGINAL_MAX_SEC:
        return None
    if source_countdown_values(text) is not None or NUMERIC_ONLY_SOURCE_RE.fullmatch(text):
        return None
    compact = _compact_short_filler_source_text(text)
    if not compact:
        return None
    if (
        compact not in ASR_SHORT_FILLER_KEEP_ORIGINAL_TOKENS
        and not _is_repeated_short_filler(compact)
    ):
        return None
    return {
        "action": "keep_original_texture",
        "reason": ASR_SHORT_FILLER_KEEP_ORIGINAL_REASON,
        "duration_sec": round(duration, 6),
        "source_text": text,
        "normalized_source_text": compact,
        "policy": ASR_SHORT_FILLER_KEEP_ORIGINAL_POLICY,
    }


def _asr_text_looks_non_speech_texture(text: str, *, duration: float) -> bool:
    stripped = text.strip()
    if source_countdown_values(text) is not None or NUMERIC_ONLY_SOURCE_RE.fullmatch(stripped):
        return False
    compact = re.sub(r"[\s　、。,.，．!！?？…・♪♡❤]+", "", stripped)
    if not compact:
        return duration >= 2.0 and any(char in NON_SPEECH_TEXTURE_PUNCTUATION for char in stripped)
    density = len(compact) / max(0.001, duration)
    if duration >= 6.0 and compact in NON_SPEECH_TEXTURE_SPARSE_TOKENS:
        return True
    if _asr_text_has_dominant_non_speech_repetition(compact):
        return True
    if _asr_text_has_dominant_ngram_repetition(compact):
        return True
    if any(char not in NON_SPEECH_TEXTURE_CHARS for char in compact):
        return False
    has_texture_marker = bool(
        NON_SPEECH_TEXTURE_REPEAT_RE.search(compact)
        or any(char in NON_SPEECH_TEXTURE_MARKERS for char in compact)
    )
    if not has_texture_marker:
        return False
    if len(compact) <= 24:
        return True
    return duration >= 6.0 and density < 1.0


def _asr_text_matches_suspicious_pattern(
    text: str,
    suspicious_text_patterns: list[str] | tuple[str, ...],
) -> bool:
    return bool(_asr_suspicious_pattern_hits(text, suspicious_text_patterns))


def _asr_numeric_only_chunk_needs_repair(text: str, *, duration: float, sparse_min_sec: float) -> bool:
    values = source_countdown_values(text)
    if values is None:
        return False
    if duration >= sparse_min_sec:
        return True
    return len(values) >= 6 and max(Counter(values).values(), default=0) >= 4


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
            if re.search(pattern, text) and pattern not in hits:
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
    if not text:
        return False
    duration = chunk.end - chunk.start
    if _asr_text_looks_non_speech_texture(text, duration=duration):
        return False
    if _asr_numeric_only_chunk_needs_repair(text, duration=duration, sparse_min_sec=sparse_min_sec):
        return True
    if source_countdown_values(text) is not None:
        return False
    if NUMERIC_ONLY_SOURCE_RE.fullmatch(text):
        return False
    if _asr_text_matches_suspicious_pattern(text, suspicious_text_patterns):
        return True
    if chunk.confidence is not None and chunk.confidence < confidence_threshold:
        return True
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


def _numeric_values_are_monotonic_step(values: list[int]) -> bool:
    if len(values) <= 1:
        return True
    deltas = [right - left for left, right in zip(values, values[1:], strict=False)]
    return all(delta == 1 for delta in deltas) or all(delta == -1 for delta in deltas)


def _asr_anomalous_text_reason(text: str, *, duration: float) -> str | None:
    normalized = text.strip()
    compact = re.sub(r"\s+", "", normalized)
    if not compact:
        return None
    if len(compact) >= 30 and REPEATED_CHAR_RUNAWAY_RE.search(compact):
        return "degenerate_repetition"

    numeric_tokens = NUMERIC_TOKEN_RE.findall(normalized)
    numeric_values = [int(token) for token in numeric_tokens]
    if len(numeric_values) >= 6:
        max_repeat = max(Counter(numeric_values).values(), default=0)
        has_countdown_cue = any(cue in normalized for cue in ("あと", "カウント", "count", "COUNT"))
        if max_repeat >= 4:
            return "numeric_runaway"
        if has_countdown_cue and not _numeric_values_are_monotonic_step(numeric_values):
            return "numeric_runaway"
    digitish_chars = sum(
        1
        for char in normalized
        if char.isdigit() or char in ",，、.．・ \t\n\r"
    )
    if (
        len(normalized) >= 80
        and len(numeric_tokens) >= 32
        and digitish_chars / max(1, len(normalized)) >= 0.85
    ):
        return "numeric_runaway"

    text_density = len(normalized) / max(0.001, duration)
    if len(compact) >= 80 and text_density >= 18.0:
        return "excessive_text_density"
    return None


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


def _apply_countdown_sequence_repairs_with_hits(text: str) -> tuple[str, list[dict[str, Any]]]:
    repaired = repair_descending_countdown_text(text)
    if repaired is None or repaired == text:
        return text, []
    return repaired, [{"source": "descending_countdown_sequence", "target": repaired, "count": 1}]


ASR_AKUME_CONTEXT_CUES = (
    "18禁",
    "ASMR",
    "アクメ",
    "女の悪夢",
    "イキ",
    "イーチ",
    "イク",
    "オナニー",
    "オナホ",
    "あと",
    "カウント",
    "クリ",
    "チンポ",
    "ポルチオ",
    "ピストン",
    "メス",
    "ゼロ",
    "だらしない",
    "エネルギー",
    "実験",
    "被験者",
    "確認",
    "継続",
    "発生",
    "自動",
    "連続",
    "直前",
    "寸前",
    "機械",
    "計測",
    "命令",
    "指令",
    "回路",
    "融合",
    "暴走",
    "発動",
    "生体",
    "波",
    "震え",
    "打ち寄せ",
    "ゾワゾワ",
    "感覚",
    "広が",
    "望んだ悪夢",
    "一瞬",
    "支配",
    "腰",
    "背筋",
    "漏らし",
    "体そら",
    "真っ白",
    "亀頭",
    "催眠",
    "子宮",
    "射精",
    "受精",
    "出産",
    "性感",
    "性器",
    "絶頂",
    "男性器",
    "潮",
    "膣",
    "快感",
    "女性器",
    "顔",
    "していい",
    "いいですよ",
)
ASR_AKUME_SOURCE_TOKENS = ("悪夢", "悪目", "明け目", "アカメ")
ASR_AKUME_GRAMMATICAL_SOURCE_CUES = (
    "悪夢し",
    "悪夢する",
    "悪夢させ",
    "悪夢します",
    "悪夢になる",
    "悪夢声",
    "悪目声",
    "明け目声",
    "アカメ声",
)
ASR_NIGHTMARE_CONTEXT_CUES = (
    "悪夢を見",
    "夢を見",
    "夢の中",
    "眠",
    "睡眠",
    "寝",
    "怖い",
    "恐怖",
    "うなされ",
    "目が覚め",
    "ホラー",
)
ASR_DENMA_CONTEXT_CUES = (
    "電マ",
    "電動",
    "バイブ",
    "振動",
    "マッサージ",
    "クリ",
    "乳首",
    "股",
    "腰",
    "おまんこ",
    "陰核",
    "刺激",
    "当て",
    "押し当て",
    "挟",
    "こす",
    "擦",
    "スイッチ",
    "強さ",
    "強め",
    "弱め",
    "弱く",
    "ブルブル",
    "震え",
)
ASR_DENMA_DIRECT_CUES = ("電話邪魔",)
ASR_PHONE_CONTEXT_CUES = (
    "電話をかけ",
    "電話かけ",
    "電話に出",
    "電話出",
    "電話が鳴",
    "電波",
    "着信",
    "通話",
    "スマホ",
    "携帯",
    "番号",
    "留守電",
    "受話器",
    "電話越し",
    "電話口",
)
ASR_KAIKAN_SOURCE_TOKENS = ("会館", "開館")
ASR_KAIKAN_CONTEXT_CUES = (
    "快感",
    "気持ち",
    "感じ",
    "絶頂",
    "アクメ",
    "イク",
    "イキ",
    "体",
    "脳",
    "脳みそ",
    "性感",
    "疼",
    "電流",
    "真っ白",
    "ゾワゾワ",
    "腰",
    "背筋",
    "波",
    "支配",
    "飲み込",
    "知りたくない",
    "やめないで",
    "耳の奥",
    "広げ",
    "迎え入れ",
    "言葉も飛",
    "言葉が飛",
    "飛んじゃ",
)
ASR_HALL_CONTEXT_CUES = (
    "市民会館",
    "公民館",
    "文化会館",
    "会館ホール",
    "会館のホール",
    "ホール",
    "イベント",
    "建物",
    "会場",
    "住所",
    "駅",
)


def _compact_asr_context(text: str) -> str:
    return re.sub(r"[\s　、。,.，．!！?？]+", "", text)


def _should_apply_contextual_akume_replacement(text: str, source: str, target: str) -> bool:
    if "アクメ" not in target or not any(token in source for token in ASR_AKUME_SOURCE_TOKENS):
        return False
    compact = _compact_asr_context(text)
    has_nightmare_context = any(cue in compact for cue in ASR_NIGHTMARE_CONTEXT_CUES)
    if has_nightmare_context:
        return False
    has_domain_context = any(cue in compact for cue in ASR_AKUME_CONTEXT_CUES)
    if has_domain_context:
        return True
    if any(cue in source for cue in ASR_AKUME_GRAMMATICAL_SOURCE_CUES):
        return True
    return any(token in source for token in ASR_AKUME_SOURCE_TOKENS)


def _is_contextual_akume_replacement(source: str, target: str) -> bool:
    return "アクメ" in target and any(token in source for token in ASR_AKUME_SOURCE_TOKENS)


def _is_contextual_denma_replacement(source: str, target: str) -> bool:
    return source == "電話" and target == "電マ"


def _is_contextual_kaikan_replacement(source: str, target: str) -> bool:
    return source in ASR_KAIKAN_SOURCE_TOKENS and target == "快感"


def _should_apply_contextual_denma_replacement(text: str, source: str, target: str) -> bool:
    if not _is_contextual_denma_replacement(source, target):
        return False
    compact = _compact_asr_context(text)
    if any(cue in compact for cue in ASR_PHONE_CONTEXT_CUES):
        return False
    if any(cue in compact for cue in ASR_DENMA_DIRECT_CUES):
        return True
    return any(cue in compact for cue in ASR_DENMA_CONTEXT_CUES)


def _should_apply_contextual_kaikan_replacement(text: str, source: str, target: str) -> bool:
    if not _is_contextual_kaikan_replacement(source, target):
        return False
    compact = _compact_asr_context(text)
    if any(cue in compact for cue in ASR_HALL_CONTEXT_CUES):
        return False
    return any(cue in compact for cue in ASR_KAIKAN_CONTEXT_CUES)


def _is_contextual_asr_replacement(source: str, target: str) -> bool:
    return (
        _is_contextual_akume_replacement(source, target)
        or _is_contextual_denma_replacement(
            source,
            target,
        )
        or _is_contextual_kaikan_replacement(source, target)
    )


def _should_apply_contextual_asr_replacement(text: str, source: str, target: str) -> bool:
    if _is_contextual_akume_replacement(source, target):
        return _should_apply_contextual_akume_replacement(text, source, target)
    if _is_contextual_denma_replacement(source, target):
        return _should_apply_contextual_denma_replacement(text, source, target)
    if _is_contextual_kaikan_replacement(source, target):
        return _should_apply_contextual_kaikan_replacement(text, source, target)
    return False


def _has_contextual_asr_replacement_negative_evidence(text: str, source: str, target: str) -> bool:
    compact = _compact_asr_context(text)
    if _is_contextual_akume_replacement(source, target):
        return any(cue in compact for cue in ASR_NIGHTMARE_CONTEXT_CUES)
    if _is_contextual_denma_replacement(source, target):
        return any(cue in compact for cue in ASR_PHONE_CONTEXT_CUES)
    if _is_contextual_kaikan_replacement(source, target):
        return any(cue in compact for cue in ASR_HALL_CONTEXT_CUES)
    return False


def _asr_candidate_replacement_should_be_contextual(text: str, source: str, target: str) -> bool:
    if not _is_contextual_asr_replacement(source, target):
        return True
    return _should_apply_contextual_asr_replacement(text, source, target)


def _asr_text_with_review_candidate_replacements(text: str, replacements: Mapping[str, str]) -> str:
    plain_replacements = {
        source: target
        for source, target in replacements.items()
        if not _is_contextual_asr_replacement(source, target)
    }
    normalized, _hits = _apply_text_replacements_with_hits(text, plain_replacements)
    normalized, _contextual_hits = _apply_contextual_asr_replacements_with_hits(normalized, replacements)
    return normalized


def _normalize_asr_text_for_repair_equivalence(text: str, cfg: Any) -> str:
    normalized, _hits = _apply_text_replacements_with_hits(
        text,
        dict(getattr(cfg, "asr_text_replacements", {}) or {}),
    )
    normalized = _asr_text_with_review_candidate_replacements(
        normalized,
        getattr(cfg, "asr_review_candidate_replacements", {}) or {},
    )
    normalized, _countdown_hits = _apply_countdown_sequence_repairs_with_hits(normalized)
    return _compact_asr_context(normalized)


def _asr_contextual_suspicious_pattern_hits(
    text: str,
    patterns: list[str] | tuple[str, ...],
    cfg: Any,
) -> list[str]:
    hits = _asr_suspicious_pattern_hits(text, patterns)
    replacements = getattr(cfg, "asr_review_candidate_replacements", {}) or {}
    filtered: list[str] = []
    for pattern in hits:
        target = replacements.get(pattern)
        if isinstance(target, str) and not _asr_candidate_replacement_should_be_contextual(
            text,
            pattern,
            target,
        ) and _has_contextual_asr_replacement_negative_evidence(text, pattern, target):
            continue
        filtered.append(pattern)
    return filtered


def _asr_auto_resolution_rule_value(rule: Any, key: str, default: Any = None) -> Any:
    if isinstance(rule, Mapping):
        return rule.get(key, default)
    return getattr(rule, key, default)


def _asr_auto_resolution_cue_matches(text: str, compact_text: str, cue: object) -> bool:
    cue_text = str(cue or "").strip()
    if not cue_text:
        return False
    return cue_text in text or _compact_asr_context(cue_text) in compact_text


def _asr_candidate_id_for_text(candidate_by_id: Mapping[str, str], text: str) -> str | None:
    compact_text = _compact_asr_context(text)
    for candidate_id, candidate_text in candidate_by_id.items():
        if candidate_text == text or _compact_asr_context(candidate_text) == compact_text:
            return candidate_id
    return None


def _guarded_asr_review_auto_resolution(
    item: Mapping[str, Any],
    candidate_by_id: Mapping[str, str],
    *,
    cfg: Any,
) -> dict[str, Any] | None:
    original_text = str(candidate_by_id.get("original") or "").strip()
    if not original_text:
        return None
    compact_text = _compact_asr_context(original_text)
    for rule in list(getattr(cfg, "asr_review_auto_resolution_rules", []) or []):
        if str(_asr_auto_resolution_rule_value(rule, "action", "auto_replace")) != "auto_replace":
            continue
        rule_id = str(_asr_auto_resolution_rule_value(rule, "id", "") or "").strip()
        source = str(_asr_auto_resolution_rule_value(rule, "source", "") or "").strip()
        target = str(_asr_auto_resolution_rule_value(rule, "target", "") or "").strip()
        if not rule_id or not source or not target or source not in original_text:
            continue
        negative_any = list(_asr_auto_resolution_rule_value(rule, "negative_any", []) or [])
        if any(_asr_auto_resolution_cue_matches(original_text, compact_text, cue) for cue in negative_any):
            continue
        required_all = list(_asr_auto_resolution_rule_value(rule, "required_all", []) or [])
        if required_all and not all(
            _asr_auto_resolution_cue_matches(original_text, compact_text, cue)
            for cue in required_all
        ):
            continue
        required_any = list(_asr_auto_resolution_rule_value(rule, "required_any", []) or [])
        if required_any and not any(
            _asr_auto_resolution_cue_matches(original_text, compact_text, cue)
            for cue in required_any
        ):
            continue
        resolved_text = original_text.replace(source, target)
        if resolved_text == original_text:
            continue
        confidence = item.get("confidence")
        source_script = SourceScript(
            text=resolved_text,
            language=getattr(cfg, "asr_language", "ja"),
            backend="asr_review_auto_resolution",
            start=float(item["start"]),
            end=float(item["end"]),
            confidence=float(confidence) if confidence is not None else None,
        )
        blocked_review_reasons = _source_script_asr_review_reasons(source_script, cfg)
        if blocked_review_reasons:
            continue
        return {
            "rule_id": rule_id,
            "text": resolved_text,
            "selected_candidate_id": _asr_candidate_id_for_text(candidate_by_id, resolved_text)
            or f"auto_resolution:{rule_id}",
        }
    return None


def _apply_contextual_asr_replacements_with_hits(
    text: str,
    replacements: Mapping[str, str],
) -> tuple[str, list[dict[str, Any]]]:
    hits: list[dict[str, Any]] = []
    normalized = text
    for source, target in replacements.items():
        if not source or source not in normalized:
            continue
        if not _should_apply_contextual_asr_replacement(normalized, source, target):
            continue
        count = normalized.count(source)
        hits.append({"source": source, "target": target, "count": count})
        normalized = normalized.replace(source, target)
    return normalized, hits


def _apply_asr_text_replacements_to_chunks_with_summary(
    chunks: list[ASRChunk],
    replacements: dict[str, str],
    *,
    contextual_replacements: Mapping[str, str] | None = None,
) -> tuple[list[ASRChunk], dict[str, Any]]:
    contextual_replacements = contextual_replacements or {}
    chunks_changed = 0
    total_replacements = 0
    items: list[dict[str, Any]] = []
    normalized_chunks: list[ASRChunk] = []
    for index, chunk in enumerate(chunks, start=1):
        text = chunk.text
        normalized, hits = _apply_text_replacements_with_hits(text, replacements)
        normalized, contextual_hits = _apply_contextual_asr_replacements_with_hits(
            normalized,
            contextual_replacements,
        )
        hits.extend(contextual_hits)
        normalized, countdown_hits = _apply_countdown_sequence_repairs_with_hits(normalized)
        hits.extend(countdown_hits)
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
            normalized_chunks.append(chunk.model_copy(update={"text": normalized, "words": []}))
        else:
            normalized_chunks.append(chunk)
    return normalized_chunks, {
        "chunks_changed": chunks_changed,
        "total_replacements": total_replacements,
        "items": items,
    }


def _merge_asr_text_replacement_summaries(*summaries: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "chunks_changed": sum(int(summary.get("chunks_changed", 0)) for summary in summaries),
        "total_replacements": sum(int(summary.get("total_replacements", 0)) for summary in summaries),
        "items": [
            item
            for summary in summaries
            for item in list(summary.get("items") or [])
        ],
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
    "Japanese ASMR speech",
    "Prefer audio evidence",
    "domain assumptions",
    "do not infer work-specific terms",
    "JapaneseASMR",
    "Japanesewhisperingdialogue",
)
ASR_SOFT_ENDING_MARKERS = (
    "ありがとうございました",
    "お疲れ様",
)
ASR_PROMPT_LEAK_MARKERS = ASR_HARD_PROMPT_LEAK_MARKERS
ASR_SOURCE_VOCALS_RELATIVE_QUIET_MAX_RMS_DBFS = -45.0
ASR_SOURCE_VOCALS_RELATIVE_QUIET_MIN_RMS_DELTA_DB = 24.0
ASR_SOURCE_VOCALS_RELATIVE_QUIET_MIN_PEAK_DELTA_DB = 6.0
ASR_PROMPT_LEAK_TERM_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}|[\u3040-\u30ff\u3400-\u9fff々〆〤ー]{2,}")
ASR_PROMPT_LEAK_SPLIT_RE = re.compile(r"[\s、。,.，・/|:;：；!！?？「」『』（）()]+")


def _asr_prompt_leak_source_texts(cfg: Any) -> list[str]:
    sources: list[str] = []
    for attr in (
        "asr_initial_prompt",
        "asr_review_initial_prompt",
        "asr_hotwords",
        "qwen_asr_context",
        "qwen_context",
    ):
        value = str(getattr(cfg, attr, "") or "").strip()
        if value:
            sources.append(value)
    asr_cfg = getattr(cfg, "asr", None)
    if asr_cfg is not None:
        value = str(getattr(asr_cfg, "qwen_context", "") or "").strip()
        if value:
            sources.append(value)
        profile = getattr(asr_cfg, "correction_profile", None)
        if profile is not None:
            for attr in ("initial_prompt", "review_initial_prompt", "hotwords", "qwen_context"):
                value = str(getattr(profile, attr, "") or "").strip()
                if value:
                    sources.append(value)
    return list(dict.fromkeys(sources))


def _asr_prompt_leak_terms(cfg: Any) -> set[str]:
    terms: set[str] = set()
    for source in _asr_prompt_leak_source_texts(cfg):
        for match in ASR_PROMPT_LEAK_TERM_RE.findall(source):
            term = match.strip()
            if len(term) >= 3:
                terms.add(term)
    return terms


def _asr_candidate_looks_like_prompt_term_list(normalized: str, cfg: Any) -> bool:
    compact = re.sub(r"\s+", "", normalized)
    if not compact or len(compact) > 64:
        return False
    terms = _asr_prompt_leak_terms(cfg)
    if not terms:
        return False

    pieces = [piece for piece in ASR_PROMPT_LEAK_SPLIT_RE.split(normalized) if piece]
    exact_matches = [piece for piece in pieces if piece in terms]
    if len(exact_matches) >= 3 and len(exact_matches) == len(pieces):
        return True

    contained: list[str] = []
    for term in sorted(terms, key=len, reverse=True):
        if term in compact and not any(term in longer for longer in contained):
            contained.append(term)
    covered_chars = sum(len(term) for term in contained)
    return len(contained) >= 3 and covered_chars / max(1, len(compact)) >= 0.65


def _asr_candidate_looks_prompt_leaked(text: str, cfg: Any) -> bool:
    normalized = " ".join(text.split())
    if not normalized:
        return False
    if any(marker in normalized for marker in ASR_HARD_PROMPT_LEAK_MARKERS):
        return True
    if normalized.count("おやすみ") >= 2:
        return True
    for source in _asr_prompt_leak_source_texts(cfg):
        if source in normalized:
            return True
    return _asr_candidate_looks_like_prompt_term_list(normalized, cfg)


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
        "word_timestamps": True,
        "hallucination_silence_threshold": None,
        "initial_prompt": None,
        "hotwords": None,
    }
    vad_parameters = dict(getattr(cfg, "asr_vad_parameters", {}) or {})
    prompted_options: dict[str, Any] = {
        "vad_filter": bool(getattr(cfg, "asr_vad_filter", True)),
        "vad_parameters": vad_parameters or None,
        "condition_on_previous_text": False,
        "word_timestamps": True,
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
    candidate_id: str | None = None,
) -> tuple[bool, float, str]:
    candidate_text = _asr_candidate_text(candidate_chunks)
    original_text = original.text.strip()
    if not candidate_text:
        return False, -100.0, "empty_candidate"
    if prompt_leaked:
        return False, -90.0, "prompt_or_hallucination_leak"
    candidate_duration = max(0.001, candidate_chunks[-1].end - candidate_chunks[0].start)
    anomalous_reason = _asr_anomalous_text_reason(candidate_text, duration=candidate_duration)
    if anomalous_reason:
        return False, -80.0, anomalous_reason
    if candidate_id == "qwen_asr_fallback":
        candidate_patterns = list(getattr(cfg, "asr_repair_suspicious_text_patterns", []) or []) + list(
            getattr(cfg, "asr_review_suspicious_text_patterns", []) or []
        )
        if _asr_suspicious_pattern_hits(candidate_text, candidate_patterns):
            return False, -70.0, "qwen_fallback_still_suspicious"
    candidate_review_reasons = _asr_repair_candidate_review_reasons(candidate_chunks, cfg)
    if candidate_review_reasons:
        return False, -75.0, f"candidate_review_blocked:{candidate_review_reasons[0]}"
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


@dataclass
class ASRRepairVote:
    candidate_id: str
    chunks: list[ASRChunk]
    score: float
    vote_count: int
    normalized_text: str


def _select_voted_asr_repair_candidate(
    candidates: Sequence[tuple[str, list[ASRChunk], float]],
    *,
    cfg: Any,
) -> ASRRepairVote | None:
    buckets: dict[str, list[tuple[str, list[ASRChunk], float]]] = {}
    for candidate_id, candidate_chunks, score in candidates:
        candidate_text = _asr_candidate_text(candidate_chunks)
        normalized_text = _normalize_asr_text_for_repair_equivalence(candidate_text, cfg)
        if not normalized_text:
            continue
        buckets.setdefault(normalized_text, []).append((candidate_id, candidate_chunks, score))
    if not buckets:
        return None
    normalized_text, group = max(
        buckets.items(),
        key=lambda item: (len(item[1]), max(score for _candidate_id, _chunks, score in item[1])),
    )
    if len(group) < 2:
        return None
    candidate_id, candidate_chunks, score = max(group, key=lambda item: item[2])
    return ASRRepairVote(
        candidate_id=candidate_id,
        chunks=candidate_chunks,
        score=score,
        vote_count=len(group),
        normalized_text=normalized_text,
    )


def _asr_repair_vote_sensitive_original(original: ASRChunk) -> bool:
    original_text = original.text.strip()
    if not original_text:
        return False
    duration = max(0.001, float(original.end) - float(original.start))
    compact = _compact_asr_context(original_text)
    if source_countdown_values(original_text) is not None:
        return True
    return duration <= 3.0 or len(compact) <= 3


def _asr_repair_candidate_requires_vote(original: ASRChunk, candidate_text: str, cfg: Any) -> bool:
    if not candidate_text.strip():
        return False
    if not _asr_repair_vote_sensitive_original(original):
        return False
    original_normalized = _normalize_asr_text_for_repair_equivalence(original.text, cfg)
    candidate_normalized = _normalize_asr_text_for_repair_equivalence(candidate_text, cfg)
    return bool(candidate_normalized) and candidate_normalized != original_normalized


def _asr_repair_candidate_review_reasons(candidate_chunks: list[ASRChunk], cfg: Any) -> list[str]:
    candidate_text = _asr_candidate_text(candidate_chunks)
    if not candidate_text or not candidate_chunks:
        return []
    replaced_candidate = _asr_text_with_review_candidate_replacements(
        candidate_text,
        getattr(cfg, "asr_review_candidate_replacements", {}) or {},
    ).strip()
    if replaced_candidate and replaced_candidate != candidate_text.strip():
        source_script = SourceScript(
            text=replaced_candidate,
            language=next((chunk.language for chunk in candidate_chunks if chunk.language), cfg.asr_language),
            backend="asr_repair_candidate",
            start=float(candidate_chunks[0].start),
            end=float(candidate_chunks[-1].end),
            confidence=_asr_candidate_confidence(candidate_chunks),
        )
        return _source_script_asr_review_reasons(source_script, cfg)

    original_source_script = SourceScript(
        text=candidate_text,
        language=next((chunk.language for chunk in candidate_chunks if chunk.language), cfg.asr_language),
        backend="asr_repair_candidate",
        start=float(candidate_chunks[0].start),
        end=float(candidate_chunks[-1].end),
        confidence=_asr_candidate_confidence(candidate_chunks),
    )
    original_reasons = _source_script_asr_review_reasons(original_source_script, cfg)
    if original_reasons:
        return original_reasons
    return []


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


def _run_asr_repair_option(
    option: dict[str, Any],
    *,
    backend: Any,
    repair_audio_path: Path,
    repair_dir: Path,
    attempted: int,
    group_start: float,
    group_end: float,
    audio_duration_sec: float,
    original: ASRChunk,
    cfg: Any,
) -> tuple[dict[str, Any], list[ASRChunk], float, str | None]:
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
        raw_candidate_text = _asr_candidate_text(candidate_local)
        raw_prompt_leaked = _asr_candidate_looks_prompt_leaked(raw_candidate_text, cfg)
        boundary_result = _clip_asr_chunks_to_window(
            candidate_local,
            clip_start=clip_start,
            clip_end=clip_end,
            window_start=group_start,
            window_end=group_end,
            require_word_timestamps_for_boundary=False,
        )
        if boundary_result.reject_reason is not None:
            return (
                _asr_repair_attempt_payload(
                    candidate_id=candidate_id,
                    clip_start=clip_start,
                    clip_end=clip_end,
                    candidate_chunks=[],
                    prompt_leaked=raw_prompt_leaked,
                    accepted=False,
                    score=-100.0,
                    reason=boundary_result.reject_reason,
                ),
                [],
                -100.0,
                None,
            )
        candidate_abs = boundary_result.chunks
        candidate_text = _asr_candidate_text(candidate_abs)
        prompt_leaked = raw_prompt_leaked or _asr_candidate_looks_prompt_leaked(candidate_text, cfg)
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
            candidate_id=candidate_id,
        )
        return (
            _asr_repair_attempt_payload(
                candidate_id=candidate_id,
                clip_start=clip_start,
                clip_end=clip_end,
                candidate_chunks=candidate_abs,
                prompt_leaked=prompt_leaked,
                accepted=accepted,
                score=score,
                reason=reason,
            ),
            candidate_abs if accepted else [],
            score,
            candidate_id if accepted else None,
        )
    except Exception as exc:
        return (
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
            ),
            [],
            -100.0,
            None,
        )


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
        normalized = chunk.model_copy(
            update={
                "start": start,
                "end": end,
                "text": text,
                "words": _clipped_asr_words(chunk, start=start, end=end),
            }
        )
        for word_split in _split_asr_chunk_on_word_gaps(normalized):
            repair_chunks.extend(_split_long_asr_chunk(word_split, max_chunk_sec=max_chunk_sec))
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
    supports_option_overrides = callable(getattr(backend, "transcribe_with_options", None))

    repair_groups: list[tuple[bool, list[ASRChunk]]] = []
    current_repair_group: list[ASRChunk] = []
    repair_group_gap_sec = max(1.0, float(getattr(cfg, "asr_resegment_merge_gap_sec", 1.0)))
    repair_group_max_sec = max(1.0, float(getattr(cfg, "asr_resegment_max_sec", 10.0)))
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
        option_specs = (
            _asr_repair_candidate_options(cfg)
            if supports_option_overrides
            else [
                {
                    "candidate_id": "plain_transcribe",
                    "padding_sec": float(getattr(cfg, "asr_repair_padding_sec", 1.0)),
                    "overrides": {},
                }
            ]
        )
        qwen_option_spec = (
            {
                "candidate_id": "qwen_asr_fallback",
                "padding_sec": float(getattr(cfg, "asr_repair_padding_sec", 1.0)),
                "overrides": {},
                "backend": qwen_fallback_backend,
            }
            if qwen_fallback_backend is not None
            else None
        )
        attempts: list[dict[str, Any]] = []
        accepted_candidates: list[tuple[str, list[ASRChunk], float]] = []
        best_candidate: list[ASRChunk] = []
        best_candidate_id: str | None = None
        best_score = -math.inf
        reject_reason: str | None = None

        for option in option_specs:
            attempt_payload, candidate, score, candidate_id = _run_asr_repair_option(
                option,
                backend=backend,
                repair_audio_path=repair_audio_path,
                repair_dir=repair_dir,
                attempted=attempted,
                group_start=group_start,
                group_end=group_end,
                audio_duration_sec=audio_duration_sec,
                original=original,
                cfg=cfg,
            )
            attempts.append(attempt_payload)
            if candidate and score > best_score:
                best_score = score
                best_candidate = candidate
                best_candidate_id = candidate_id
            if candidate and candidate_id is not None:
                accepted_candidates.append((candidate_id, candidate, score))
            if (
                best_candidate
                and best_score >= 0.5
                and not _asr_repair_vote_sensitive_original(original)
            ):
                break
        if qwen_option_spec is not None and not best_candidate:
            attempt_payload, candidate, score, candidate_id = _run_asr_repair_option(
                qwen_option_spec,
                backend=backend,
                repair_audio_path=repair_audio_path,
                repair_dir=repair_dir,
                attempted=attempted,
                group_start=group_start,
                group_end=group_end,
                audio_duration_sec=audio_duration_sec,
                original=original,
                cfg=cfg,
            )
            attempts.append(attempt_payload)
            if candidate and score > best_score:
                best_score = score
                best_candidate = candidate
                best_candidate_id = candidate_id
            if candidate and candidate_id is not None:
                accepted_candidates.append((candidate_id, candidate, score))
        vote = _select_voted_asr_repair_candidate(accepted_candidates, cfg=cfg)
        if vote is not None:
            best_candidate = vote.chunks
            best_candidate_id = vote.candidate_id
            best_score = vote.score
        candidate_text = _asr_candidate_text(best_candidate)
        if vote is None and best_candidate and _asr_repair_candidate_requires_vote(
            original,
            candidate_text,
            cfg,
        ):
            best_candidate = []
            best_candidate_id = None
            best_score = -math.inf
            reject_reason = "candidate_vote_required"
            candidate_text = ""
        accepted = bool(best_candidate)
        normalized_candidate_text = (
            _normalize_asr_text_for_repair_equivalence(candidate_text, cfg)
            if candidate_text
            else ""
        )
        summary["items"].append(
            {
                "start": round(group_start, 3),
                "end": round(group_end, 3),
                "accepted": accepted,
                "accepted_candidate_id": best_candidate_id,
                "accepted_candidate_reason": "normalized_candidate_vote" if vote else "score",
                "candidate_vote_count": vote.vote_count if vote else (1 if accepted else 0),
                "reject_reason": reject_reason,
                "normalized_candidate_text": normalized_candidate_text,
                "prompt_leaked": any(bool(attempt.get("prompt_leaked")) for attempt in attempts),
                "original_text": original.text,
                "candidate_text": candidate_text,
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
    "電マ",
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
    if not text:
        return None
    duration = max(0.0, chunk.end - chunk.start)
    if _asr_text_looks_non_speech_texture(text, duration=duration):
        return None
    sparse_text_density = (
        duration >= getattr(cfg, "asr_repair_sparse_min_sec", 12.0)
        and len(text) / max(0.001, duration)
        < getattr(cfg, "asr_repair_sparse_min_chars_per_sec", 1.0)
    )
    if NUMERIC_ONLY_SOURCE_RE.fullmatch(text) and not sparse_text_density:
        return None
    suspicious_patterns = _asr_contextual_suspicious_pattern_hits(
        text,
        cfg.asr_review_suspicious_text_patterns,
        cfg,
    )
    if sparse_text_density:
        suspicious_patterns.append("asr_sparse_text_density")
    candidate_text = _asr_text_with_review_candidate_replacements(
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
        "duration": round(duration, 3),
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
            replaced_text = _asr_text_with_review_candidate_replacements(
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


def _add_qwen_asr_review_candidates(
    review_items: list[dict[str, Any]],
    *,
    qwen_backend: Any | None,
    project_dir: Path,
    review_audio_path: Path,
    audio_duration_sec: float,
    cfg: Any,
) -> int:
    if qwen_backend is None:
        return 0
    if not cfg.asr_review_generate_candidates:
        return 0
    if not review_audio_path.exists():
        return 0

    padding_values = [
        max(0.0, float(value))
        for value in getattr(cfg, "asr_review_candidate_padding_sec", [])
    ]
    if not padding_values:
        padding_values = [0.8]

    generated = 0
    review_dir = project_dir / "work" / "transcribe" / "asr_review_clips"
    for item in review_items:
        chunk_id = str(item.get("chunk_id") or "chunk")
        start = float(item["start"])
        end = float(item["end"])
        for index, padding_sec in enumerate(padding_values, start=1):
            clip_start = max(0.0, start - padding_sec)
            clip_end = min(audio_duration_sec, end + padding_sec)
            if clip_end <= clip_start:
                continue
            candidate_id = f"qwen_asr_pad_{index}"
            clip_path = review_dir / f"{chunk_id}_{candidate_id}.wav"
            ffmpeg.slice_audio(
                review_audio_path,
                clip_start,
                clip_end,
                clip_path,
                sample_rate=cfg.gemma_sample_rate,
                channels=1,
            )
            try:
                candidate_chunks = _transcribe_with_backend_options(qwen_backend, clip_path, [])
            except Exception:
                continue
            candidate_text = _asr_candidate_text(candidate_chunks)
            if _asr_candidate_looks_prompt_leaked(candidate_text, cfg):
                continue
            if _append_unique_asr_review_candidate(
                item,
                candidate_id=candidate_id,
                text=candidate_text,
            ):
                generated += 1
            replaced_text = _asr_text_with_review_candidate_replacements(
                candidate_text,
                cfg.asr_review_candidate_replacements,
            )
            if replaced_text != candidate_text and _append_unique_asr_review_candidate(
                item,
                candidate_id=f"{candidate_id}_domain_replacement",
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
    qwen_fallback_backend: Any | None = None,
) -> tuple[list[ASRChunk], dict[str, Any]]:
    summary: dict[str, Any] = {
        "enabled": bool(cfg.asr_review_enabled),
        "backend": cfg.asr_review_backend,
        "attempted": 0,
        "reviewed": 0,
        "replaced": 0,
        "guarded_auto_replaced": 0,
        "manual_review": 0,
        "failed": 0,
        "skipped": 0,
        "generated_candidates": 0,
        "generated_qwen_candidates": 0,
        "audio_input": {"enabled": False, "created": 0, "error": None, "items": []},
        "error": None,
        "errors": [],
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
    summary["generated_qwen_candidates"] = _add_qwen_asr_review_candidates(
        review_items,
        qwen_backend=qwen_fallback_backend,
        project_dir=project_dir,
        review_audio_path=review_audio_path,
        audio_duration_sec=audio_duration_sec,
        cfg=cfg,
    )
    summary["generated_candidates"] += summary["generated_qwen_candidates"]

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
            batch_items = list(batch)
            try:
                if audio_review_enabled:
                    audio_path = Path(str(batch_items[0]["audio_clip_path"]))
                    review_results.update(
                        client.review_asr_candidates_with_audio(batch_items, batch_id, audio_path)
                    )
                else:
                    review_results.update(client.review_asr_candidates_for_mock(batch_items, batch_id))
            except Exception as exc:
                chunk_ids = [str(item.get("chunk_id") or "") for item in batch_items]
                summary["failed"] += len(batch_items)
                summary["errors"].append({"chunk_ids": chunk_ids, "error": str(exc)})
                for item in batch_items:
                    chunk_id = str(item.get("chunk_id") or "")
                    if not chunk_id:
                        continue
                    review_results[chunk_id] = {
                        "chunk_id": chunk_id,
                        "decision": "manual_review",
                        "selected_candidate_id": "original",
                        "confidence": 0.0,
                        "heard_text": "",
                        "reason": f"ASR review failed: {exc}",
                        "risk_terms": ["asr_review_error"],
                    }
    except Exception as exc:
        summary["error"] = str(exc)
        return chunks, summary
    finally:
        if server_manager is not None:
            server_manager.stop()

    summary["attempted"] = len(review_items)
    if summary["failed"]:
        summary["error"] = f"{summary['failed']} ASR review item(s) failed; see errors"
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
        model_decision = review.get("decision")
        model_selected_id = selected_id
        auto_resolution: dict[str, Any] | None = None
        accepted = (
            model_decision == "replace"
            and selected_id != "original"
            and selected_text is not None
            and confidence is not None
            and confidence >= cfg.asr_review_confidence_threshold
        )
        blocked_review_reasons: list[str] = []
        if accepted and selected_text is not None:
            selected_chunk = ASRChunk(
                start=float(item["start"]),
                end=float(item["end"]),
                text=selected_text,
                language=cfg.asr_language,
                confidence=float(confidence),
            )
            blocked_review_reasons = _asr_repair_candidate_review_reasons([selected_chunk], cfg)
            if blocked_review_reasons:
                accepted = False
        if not accepted:
            auto_resolution = _guarded_asr_review_auto_resolution(
                item,
                candidate_by_id,
                cfg=cfg,
            )
            if auto_resolution is not None:
                selected_id = str(auto_resolution["selected_candidate_id"])
                selected_text = str(auto_resolution["text"])
                accepted = True
                blocked_review_reasons = []
                summary["guarded_auto_replaced"] += 1
        if not accepted and (blocked_review_reasons or model_decision == "manual_review"):
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
                "decision": model_decision,
                "selected_candidate_id": selected_id,
                "confidence": confidence,
                "original_text": candidate_by_id.get("original", ""),
                "selected_text": selected_text,
                "candidates": list(item.get("candidates", [])),
                "audio_clip": item.get("audio_clip"),
                "heard_text": review.get("heard_text"),
                "reason": (
                    f"selected_candidate_review_blocked:{blocked_review_reasons[0]}"
                    if blocked_review_reasons
                    else review.get("reason")
                ),
                "risk_terms": [
                    *list(review.get("risk_terms", []) or []),
                    *(["selected_candidate_review_blocked"] if blocked_review_reasons else []),
                ],
                "blocked_review_reasons": blocked_review_reasons,
                "suspicious_patterns": item.get("suspicious_patterns", []),
                **(
                    {
                        "resolution": "guarded_auto_replace",
                        "auto_resolution_rule_id": auto_resolution["rule_id"],
                        "model_decision": model_decision,
                        "model_selected_candidate_id": model_selected_id,
                    }
                    if auto_resolution is not None
                    else {}
                ),
            }
        )

    for index, chunk in enumerate(reviewed_chunks):
        chunk_id = f"chunk_{index + 1:04d}"
        selected_text = selected_text_by_chunk_id.get(chunk_id)
        if selected_text is not None:
            reviewed_chunks[index] = chunk.model_copy(update={"text": selected_text, "words": []})

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


def _log_stage_checkpoint(stage: str, message: str, detail: str | None = None) -> None:
    suffix = f" {detail}" if detail else ""
    console.print(f"[dim]{escape(f'{stage}: {message}{suffix}')}[/dim]")


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
    progress_index: int | None = None,
    progress_label: str = "done",
    counts_label: str = "counts",
) -> float:
    now = monotonic()
    interval = _progress_interval(total)
    display_index = progress_index if progress_index is not None else index
    should_log = (
        display_index == 1
        or display_index == total
        or display_index % interval == 0
        or now - last_logged_at >= PROGRESS_LOG_SECONDS
    )
    if not should_log:
        return last_logged_at
    percent = (display_index / total * 100.0) if total else 100.0
    progress = (
        f"{progress_label}={display_index}/{total}"
        if progress_index is not None
        else f"{index}/{total}"
    )
    segment_index = f" segment_index={index}" if progress_index is not None else ""
    counts = (
        f" {counts_label}={_format_segment_counts(_segment_counts(manifest))}"
        if manifest and manifest.segments
        else ""
    )
    suffix = f" - {note}" if note else ""
    console.print(
        f"[dim]{stage}: {progress} ({percent:.1f}%) "
        f"elapsed={_format_elapsed(now - started_at)} "
        f"latest={segment.id}{segment_index} status={segment.status}{counts}{suffix}[/dim]"
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
    return native_korean_count_number(value)


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
            if COUNTDOWN_EVENT_NOTE in translation.notes:
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
        if phrase in seen and _is_meaningful_repeated_source_phrase(phrase):
            return True
        seen.add(phrase)
    return False


def _is_meaningful_repeated_source_phrase(phrase: str) -> bool:
    compact = _compact_translation_repeat_text(phrase)
    if len(compact) < 8:
        return False
    kanji_count = len(JAPANESE_KANJI_RE.findall(compact))
    if kanji_count < 2 and JAPANESE_REPEAT_GRAMMAR_FRAME_RE.search(compact):
        return False
    if kanji_count >= 2:
        return True
    return len(compact) >= 16


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
    if pattern == "체감" and "体感" in source_text:
        return False
    return pattern in text


def _backcheck_severity(item: dict[str, Any]) -> str:
    hits = [str(hit) for hit in item.get("translation_hits", [])]
    if any(hit in SEVERE_TRANSLATION_BACKCHECK_KO_PATTERNS for hit in hits):
        return "severe"
    ko_natural = str(item.get("ko_natural") or "")
    source_text = str(item.get("source_text") or "")
    if any(
        _translation_backcheck_ko_pattern_matches(ko_natural, pattern, source_text=source_text)
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
    cfg: Any | None = None,
) -> None:
    cfg = cfg or ProjectConfig()
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
        repetition_warn_only = (
            rejected_reasons == ["natural_repetition_omission"]
            and getattr(cfg, "gemma_text_repetition_omission_policy", "warn") == "warn"
        )
        if repetition_warn_only:
            warning_codes = list(dict.fromkeys([*warning_codes, "natural_repetition_omission"]))
            literal_fallback_applied = False
            if translation is not None and translation.ko_literal.strip():
                max_speech_chars = int(
                    korean_tts_timing_budget(segment.duration, source_text)["max_speech_chars"]
                )
                literal_text = translation.ko_literal.strip()
                if (
                    literal_text != translation.ko_natural.strip()
                    and korean_tts_speech_char_count(literal_text) <= max_speech_chars
                ):
                    notes = list(translation.notes)
                    if "repetition_preserving_literal_fallback" not in notes:
                        notes.append("repetition_preserving_literal_fallback")
                    translation = translation.model_copy(
                        update={"ko_natural": literal_text, "notes": notes}
                    )
                    segment.translation_ko = translation
                    literal_fallback_applied = True
            segment.analysis["translation_auto_fallback"] = {
                "reason": "natural_repetition_omission_warn_only",
                "policy": "warn",
                "literal_fallback_applied": literal_fallback_applied,
            }
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
            continue
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


def _countdown_values_for_segment(segment: Segment) -> list[int] | None:
    source_text = _translation_source_text(segment)
    if source_text:
        values = source_countdown_values(source_text)
        if values is None or not is_descending_countdown(values):
            return None
        _ensure_countdown_event_analysis(segment, values)
        return values

    event = segment.analysis.get(COUNTDOWN_EVENT_KEY)
    if isinstance(event, dict):
        raw_values = event.get("values")
        if isinstance(raw_values, list) and all(isinstance(value, int) for value in raw_values):
            values = [int(value) for value in raw_values]
            if is_descending_countdown(values):
                return values
    return None


def _ensure_countdown_event_analysis(segment: Segment, values: list[int]) -> None:
    event = segment.analysis.get(COUNTDOWN_EVENT_KEY)
    if not isinstance(event, dict):
        event = {}
    payload = _countdown_event_payload(
        values,
        source_chunk_texts=[_translation_source_text(segment)],
        source_chunk_count=1,
        merge_gap_sec=0.0,
        max_span_sec=segment.duration,
    )
    if payload is None:
        return
    payload.update({key: value for key, value in event.items() if key not in payload})
    segment.analysis[COUNTDOWN_EVENT_KEY] = payload


def _countdown_translation_for_segment(segment: Segment, values: list[int]) -> KoreanTranslation | None:
    korean_text = countdown_korean_text(values)
    if korean_text is None:
        return None
    _ensure_countdown_event_analysis(segment, values)
    return KoreanTranslation(
        ko_literal=korean_text,
        ko_natural=korean_text,
        notes=[COUNTDOWN_EVENT_NOTE],
        confidence=1.0,
        model="deterministic:countdown-event",
        batch_id=f"countdown_{segment.id}",
    )


def _counting_values_for_segment(segment: Segment) -> list[int] | None:
    values = source_countdown_values(_translation_source_text(segment))
    if values is None or is_descending_countdown(values):
        return None
    if not _numeric_group_is_counting([(segment, values)]):
        return None
    return values


def _counting_translation_for_segment(segment: Segment, values: list[int]) -> KoreanTranslation | None:
    spoken_values = [_native_korean_count_number(value) for value in values]
    if any(value is None for value in spoken_values):
        return None
    korean_text = ", ".join(value for value in spoken_values if value is not None)
    return KoreanTranslation(
        ko_literal=korean_text,
        ko_natural=korean_text,
        notes=[NUMERIC_COUNTING_POSTPROCESS_NOTE],
        confidence=1.0,
        model="deterministic:numeric-counting",
        batch_id=f"numeric_counting_{segment.id}",
    )


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
    asr_repair_enabled: bool | None = None,
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
    if asr_repair_enabled is not None:
        payload["asr_repair_enabled"] = bool(asr_repair_enabled)
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


def _prefer_folder_asr_source_audio(manifest: PipelineManifest) -> bool:
    folder_input = None
    if manifest.source_info is not None:
        raw_folder_input = manifest.source_info.raw.get("folder_input")
        if isinstance(raw_folder_input, dict):
            folder_input = raw_folder_input
    if folder_input is None:
        return False
    status = str(folder_input.get("asr_source_status") or "")
    silent_count = int(folder_input.get("asr_silent_part_count") or 0)
    return (
        status in {"separate_asr_parts", "mix_parts_silenced_duplicates", "mix_parts_silenced_for_asr"}
        or silent_count > 0
    )


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

    candidate_order = (
        ("gemma_mono_16k", False),
        ("source_vocals_mono_16k", backend_kind != "mock"),
    ) if _prefer_folder_asr_source_audio(manifest) else (
        ("source_vocals_mono_16k", backend_kind != "mock"),
        ("gemma_mono_16k", False),
    )
    for source, strict_quality in candidate_order:
        add_candidate(
            source,
            _resolve_manifest_artifact_path(project_dir, manifest, source),
            strict_quality=strict_quality,
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
                "word_count": len(chunk.words),
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
        normalized = " ".join(text.split())
        duration = max(0.0, chunk.end - chunk.start)
        density = len(text) / max(0.001, duration)
        prompt_leak = _asr_candidate_looks_prompt_leaked(text, cfg)
        prompt_term_list_leak = prompt_leak and _asr_candidate_looks_like_prompt_term_list(
            normalized,
            cfg,
        )
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
        if prompt_term_list_leak or sparse_prompt_leak or repeated_outro:
            dropped.append(
                {
                    "chunk_id": f"chunk_{index:04d}",
                    "start": round(chunk.start, 3),
                    "end": round(chunk.end, 3),
                    "text": text,
                    "reason": (
                        "prompt_term_list_leak"
                        if prompt_term_list_leak
                        else "sparse_prompt_leak"
                        if sparse_prompt_leak
                        else "repeated_outro_hallucination"
                    ),
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
    duration = max(0.001, source_script.end - source_script.start)
    if _asr_text_looks_non_speech_texture(text, duration=duration):
        return reasons
    if _asr_candidate_looks_prompt_leaked(text, cfg):
        reasons.append("asr_prompt_or_hallucination_leak")
        return reasons
    mixed_texture_reason = _asr_mixed_texture_speech_reason(text, duration=duration)
    if mixed_texture_reason:
        reasons.append(mixed_texture_reason)
    patterns = list(getattr(cfg, "asr_repair_suspicious_text_patterns", []) or []) + list(
        getattr(cfg, "asr_review_suspicious_text_patterns", []) or []
    )
    hits = _asr_contextual_suspicious_pattern_hits(text, patterns, cfg)
    if hits:
        reasons.append("asr_suspicious_pattern:" + ",".join(hits[:5]))
    anomalous_reason = _asr_anomalous_text_reason(text, duration=duration)
    if anomalous_reason and not (
        anomalous_reason == "numeric_runaway"
        and not hits
        and (
            _source_script_countdown_unverified_reason(source_script)
            or _source_script_numeric_sequence_unverified_reason(source_script)
        )
    ):
        reasons.append(f"asr_{anomalous_reason}")
    if (
        source_script.confidence is not None
        and source_script.confidence < getattr(cfg, "asr_review_confidence_threshold", 0.78)
    ):
        reasons.append(f"asr_low_confidence:{source_script.confidence:.3f}")
    compact_text = re.sub(r"[\s　、。,.，．!！?？…・♪♡❤（）()\[\]「」『』]+", "", text)
    sparse_min_sec = getattr(cfg, "asr_repair_sparse_min_sec", 12.0)
    sparse_min_chars_per_sec = getattr(cfg, "asr_repair_sparse_min_chars_per_sec", 1.0)
    short_sparse_fragment = duration >= 8.0 and len(compact_text) <= 2
    if (
        (duration >= sparse_min_sec or short_sparse_fragment)
        and len(text) / duration < sparse_min_chars_per_sec
        and not _source_script_asr_warning_reasons(source_script, cfg)
    ):
        reasons.append("asr_sparse_text_density")
    return reasons


def _source_script_countdown_unverified_reason(source_script: SourceScript | None) -> str | None:
    if source_script is None or not source_script.text.strip():
        return None
    text = source_script.text.strip()
    values = source_countdown_values(text)
    if values is not None:
        if is_descending_countdown(values):
            return None
        if len(values) >= 4 and min(values) >= 0 and max(values) <= 30:
            return "asr_countdown_unverified"
        return None
    embedded_values = _embedded_source_countdown_values(text)
    if embedded_values is not None and is_descending_countdown(embedded_values):
        return None
    numeric_values = [int(token) for token in NUMERIC_TOKEN_RE.findall(text)]
    if len(numeric_values) < 4 or min(numeric_values, default=99) < 0 or max(numeric_values, default=0) > 30:
        return None
    has_countdown_cue = any(cue in text for cue in ("あと", "カウント", "count", "COUNT"))
    has_descending_pair = any(
        left - right == 1 for left, right in zip(numeric_values, numeric_values[1:], strict=False)
    )
    edge_looks_like_countdown = numeric_values[0] <= 30 and numeric_values[-1] in {0, 1}
    if (has_countdown_cue or edge_looks_like_countdown) and has_descending_pair:
        return "asr_countdown_unverified"
    return None


def _source_script_numeric_sequence_unverified_reason(
    source_script: SourceScript | None,
) -> str | None:
    if source_script is None or not source_script.text.strip():
        return None
    text = source_script.text.strip()
    if not NUMERIC_SEQUENCE_SOURCE_RE.fullmatch(text):
        return None
    numeric_values = [int(token) for token in NUMERIC_TOKEN_RE.findall(text)]
    if len(numeric_values) < 3 or max(numeric_values, default=0) <= 99:
        return None
    if _numeric_values_are_monotonic_step(numeric_values):
        return "asr_numeric_sequence_unverified"
    return None


def _asr_text_looks_plausible_sparse_speech(text: str) -> bool:
    stripped = text.strip()
    if not stripped or NUMERIC_TOKEN_RE.search(stripped):
        return False
    compact = re.sub(r"[\s　、。,.，．!！?？…・♪♡❤（）()\[\]「」『』]+", "", stripped)
    if len(compact) < 4:
        return False
    if not (
        JAPANESE_HIRAGANA_RE.search(compact)
        or JAPANESE_KATAKANA_RE.search(compact)
        or JAPANESE_KANJI_RE.search(compact)
    ):
        return False

    has_natural_cue = any(cue in compact for cue in SPARSE_SPEECH_NATURAL_CUES)
    tokens = [
        re.sub(r"[、。,.，．!！?？…・♪♡❤（）()\[\]「」『』]+", "", token)
        for token in stripped.split()
        if token.strip()
    ]
    if len(tokens) >= 3 and not has_natural_cue and all(len(token) <= 4 for token in tokens):
        return False

    has_particle_cue = any(char in SPARSE_SPEECH_PARTICLE_CUES for char in compact)
    return has_natural_cue or (has_particle_cue and len(compact) >= 6)


def _source_script_sparse_speech_unverified_reason(
    source_script: SourceScript | None,
    cfg: Any,
) -> str | None:
    if source_script is None or not source_script.text.strip():
        return None
    if (
        _source_script_countdown_unverified_reason(source_script)
        or _source_script_numeric_sequence_unverified_reason(source_script)
    ):
        return None
    text = source_script.text.strip()
    duration = max(0.001, source_script.end - source_script.start)
    if (
        duration < getattr(cfg, "asr_repair_sparse_min_sec", 12.0)
        or len(text) / duration >= getattr(cfg, "asr_repair_sparse_min_chars_per_sec", 1.0)
    ):
        return None
    if _asr_text_looks_non_speech_texture(text, duration=duration):
        return None
    if _asr_candidate_looks_prompt_leaked(text, cfg):
        return None
    patterns = list(getattr(cfg, "asr_repair_suspicious_text_patterns", []) or []) + list(
        getattr(cfg, "asr_review_suspicious_text_patterns", []) or []
    )
    if _asr_contextual_suspicious_pattern_hits(text, patterns, cfg):
        return None
    if _asr_anomalous_text_reason(text, duration=duration):
        return None
    if (
        source_script.confidence is not None
        and source_script.confidence < getattr(cfg, "asr_review_confidence_threshold", 0.78)
    ):
        return None
    if not _asr_text_looks_plausible_sparse_speech(text):
        return None
    return "asr_sparse_speech_unverified"


def _source_script_short_clean_fragment_warning_reason(
    source_script: SourceScript | None,
    cfg: Any,
) -> str | None:
    if not bool(getattr(cfg, "asr_repair_rejected_short_fragment_auto_accept", True)):
        return None
    if source_script is None or not source_script.text.strip():
        return None
    duration = max(0.001, source_script.end - source_script.start)
    if duration > float(getattr(cfg, "asr_repair_rejected_short_fragment_max_sec", 0.8)):
        return None
    confidence = source_script.confidence
    if confidence is None or confidence < float(
        getattr(cfg, "asr_repair_rejected_short_fragment_min_confidence", 0.85)
    ):
        return None
    text = source_script.text.strip()
    if (
        _source_script_countdown_unverified_reason(source_script)
        or _source_script_numeric_sequence_unverified_reason(source_script)
        or _asr_text_has_unstable_embedded_numeric_sequence(text)
    ):
        return None
    if _source_script_non_speech_texture_reason(source_script) is not None:
        return None
    if _asr_candidate_looks_prompt_leaked(text, cfg):
        return None
    patterns = list(getattr(cfg, "asr_repair_suspicious_text_patterns", []) or []) + list(
        getattr(cfg, "asr_review_suspicious_text_patterns", []) or []
    )
    if _asr_contextual_suspicious_pattern_hits(text, patterns, cfg):
        return None
    if _asr_anomalous_text_reason(text, duration=duration):
        return None
    compact = ASR_SHORT_CLEAN_FRAGMENT_DROP_RE.sub("", text)
    if compact in ASR_SHORT_CLEAN_FRAGMENT_TOKENS:
        return ASR_SHORT_CLEAN_FRAGMENT_WARNING_REASON
    return None


def _source_script_asr_warning_reasons(source_script: SourceScript | None, cfg: Any) -> list[str]:
    reasons = [
        _source_script_short_clean_fragment_warning_reason(source_script, cfg),
        _source_script_countdown_unverified_reason(source_script),
        _source_script_numeric_sequence_unverified_reason(source_script),
        _source_script_sparse_speech_unverified_reason(source_script, cfg),
    ]
    return list(dict.fromkeys(reason for reason in reasons if reason))


def _filter_asr_repair_review_reasons(
    source_script: SourceScript | None,
    cfg: Any,
    *,
    review_reasons: Sequence[str],
    repair_review_reasons: Sequence[str],
) -> list[str]:
    repair_reasons = [str(reason) for reason in repair_review_reasons if reason]
    if not repair_reasons or review_reasons:
        return repair_reasons
    if _source_script_asr_warning_reasons(source_script, cfg) and all(
        reason == "asr_repair_rejected" for reason in repair_reasons
    ):
        return []
    if (
        _source_script_is_clean_segment_retry(source_script, cfg)
        and all(reason == "asr_repair_rejected" for reason in repair_reasons)
    ):
        return []
    return repair_reasons


def _source_script_is_clean_segment_retry(source_script: SourceScript | None, cfg: Any) -> bool:
    if source_script is None or not source_script.text.strip():
        return False
    if ":segment_retry:" not in str(source_script.backend):
        return False
    if _asr_text_has_unstable_embedded_numeric_sequence(source_script.text):
        return False
    if _source_script_non_speech_texture_reason(source_script) is not None:
        return False
    if _source_script_asr_warning_reasons(source_script, cfg):
        return False
    return not _source_script_asr_review_reasons(source_script, cfg)


def _asr_text_has_unstable_embedded_numeric_sequence(text: str) -> bool:
    normalized = text.strip()
    if not normalized:
        return False
    numeric_values = [int(token) for token in NUMERIC_TOKEN_RE.findall(normalized)]
    if len(numeric_values) < 6 or max(numeric_values, default=0) > 30:
        return False
    if _numeric_values_are_monotonic_step(numeric_values):
        return False
    non_numeric = NUMERIC_TOKEN_RE.sub("", normalized)
    compact_non_numeric = _compact_asr_context(non_numeric)
    if not compact_non_numeric:
        return False
    return bool(
        JAPANESE_HIRAGANA_RE.search(compact_non_numeric)
        or JAPANESE_KATAKANA_RE.search(compact_non_numeric)
        or JAPANESE_KANJI_RE.search(compact_non_numeric)
    )


def _source_script_non_speech_texture_reason(source_script: SourceScript | None) -> str | None:
    if source_script is None or not source_script.text.strip():
        return None
    duration = max(0.001, source_script.end - source_script.start)
    if _asr_text_looks_non_speech_texture(source_script.text, duration=duration):
        return "asr_non_speech_texture"
    return None


def _source_script_keep_original_texture_override_reason(
    segment: Segment,
    source_script: SourceScript | None,
    cfg: Any,
) -> str | None:
    if not bool(segment.keep_original_texture):
        return None
    if source_script is None or not source_script.text.strip():
        return None
    text = source_script.text.strip()
    if source_countdown_values(text) is not None or NUMERIC_ONLY_SOURCE_RE.fullmatch(text):
        return None
    duration = max(0.001, float(source_script.end) - float(source_script.start))
    if _asr_text_looks_non_speech_texture(text, duration=duration):
        return "asr_non_speech_texture"
    compact = re.sub(r"[\s　、。,.，．!！?？…・♪♡❤（）()\[\]「」『』]+", "", text)
    sparse_min_chars_per_sec = float(getattr(cfg, "asr_repair_sparse_min_chars_per_sec", 1.0))
    if (
        duration >= 6.0
        and len(compact) <= 2
        and len(text) / duration < sparse_min_chars_per_sec
    ):
        return "asr_non_speech_texture"
    return None


def _source_script_rejected_repair_reasons(
    source_script: SourceScript | None,
    repair_summary: dict[str, Any],
    cfg: Any | None = None,
) -> list[str]:
    if source_script is None or not source_script.text.strip():
        return []
    reasons: list[str] = []
    script_start = float(source_script.start)
    script_end = float(source_script.end)
    script_duration = max(0.001, script_end - script_start)
    source_equivalence_text = (
        _normalize_asr_text_for_repair_equivalence(source_script.text, cfg)
        if cfg is not None
        else ""
    )
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
        if source_equivalence_text and _repair_item_matches_source_after_replacements(
            item,
            source_equivalence_text=source_equivalence_text,
            cfg=cfg,
        ):
            continue
        suffix = ":prompt_or_hallucination_leak" if _repair_item_prompt_leak_blocks_source(
            item,
            source_script=source_script,
            cfg=cfg,
        ) else ""
        reason = f"asr_repair_rejected{suffix}"
        if reason not in reasons:
            reasons.append(reason)
    return reasons


def _repair_item_prompt_leak_blocks_source(
    item: Mapping[str, Any],
    *,
    source_script: SourceScript,
    cfg: Any | None,
) -> bool:
    attempts = [attempt for attempt in item.get("attempts", []) or [] if isinstance(attempt, Mapping)]
    prompt_leaked_attempts = [
        attempt
        for attempt in attempts
        if bool(attempt.get("prompt_leaked"))
        or str(attempt.get("reason") or "") == "prompt_or_hallucination_leak"
    ]
    if not prompt_leaked_attempts:
        return False
    if cfg is None:
        return True
    if _asr_candidate_looks_prompt_leaked(source_script.text, cfg):
        return True
    text_attempts = [
        attempt
        for attempt in attempts
        if str(attempt.get("candidate_text") or "").strip()
    ]
    if not text_attempts:
        return True
    return all(
        bool(attempt.get("prompt_leaked"))
        or str(attempt.get("reason") or "") == "prompt_or_hallucination_leak"
        or _asr_candidate_looks_prompt_leaked(str(attempt.get("candidate_text") or ""), cfg)
        for attempt in text_attempts
    )


def _repair_item_candidate_texts(item: Mapping[str, Any]) -> list[str]:
    texts: list[str] = []
    text = str(item.get("candidate_text") or "").strip()
    if text:
        texts.append(text)
    for attempt in item.get("attempts", []) or []:
        if not isinstance(attempt, Mapping):
            continue
        text = str(attempt.get("candidate_text") or "").strip()
        if text:
            texts.append(text)
    return texts


def _repair_item_matches_source_after_replacements(
    item: Mapping[str, Any],
    *,
    source_equivalence_text: str,
    cfg: Any,
) -> bool:
    allow_partial_match = not _repair_item_has_prompt_leak_attempt(item)
    for text in _repair_item_candidate_texts(item):
        if _normalized_repair_candidate_matches_source(
            _normalize_asr_text_for_repair_equivalence(text, cfg),
            source_equivalence_text,
            allow_partial=allow_partial_match,
        ):
            return True
    return False


def _repair_item_has_prompt_leak_attempt(item: Mapping[str, Any]) -> bool:
    attempts = [attempt for attempt in item.get("attempts", []) or [] if isinstance(attempt, Mapping)]
    return any(
        bool(attempt.get("prompt_leaked"))
        or str(attempt.get("reason") or "") == "prompt_or_hallucination_leak"
        for attempt in attempts
    )


def _normalized_repair_candidate_matches_source(
    candidate_text: str,
    source_text: str,
    *,
    allow_partial: bool,
) -> bool:
    if not candidate_text or not source_text:
        return False
    if candidate_text == source_text:
        return True
    if not allow_partial:
        return False
    shorter, longer = (
        (candidate_text, source_text)
        if len(candidate_text) <= len(source_text)
        else (source_text, candidate_text)
    )
    if len(shorter) < 8:
        return False
    if len(shorter) / max(1, len(longer)) < 0.35:
        return False
    return shorter in longer


def _asr_summary_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _asr_source_separation_manual_review_threshold(segment_count: int) -> int:
    if segment_count <= 0:
        return ASR_SOURCE_SEPARATION_FALLBACK_MIN_MANUAL_REVIEW_COUNT
    rate_threshold = math.ceil(
        segment_count * ASR_SOURCE_SEPARATION_FALLBACK_MIN_MANUAL_REVIEW_RATE
    )
    return max(ASR_SOURCE_SEPARATION_FALLBACK_MIN_MANUAL_REVIEW_COUNT, rate_threshold)


def _asr_folder_input_parts(manifest: PipelineManifest) -> list[dict[str, Any]]:
    if manifest.source_info is None:
        return []
    folder_input = manifest.source_info.raw.get("folder_input")
    if not isinstance(folder_input, dict):
        return []
    raw_parts = folder_input.get("asr_parts") or folder_input.get("mix_parts")
    if not isinstance(raw_parts, list):
        return []
    parts: list[dict[str, Any]] = []
    for raw_part in raw_parts:
        if not isinstance(raw_part, dict):
            continue
        try:
            start_sec = float(raw_part.get("start_sec") or 0.0)
            end_sec = float(raw_part.get("end_sec") or 0.0)
        except (TypeError, ValueError):
            continue
        if end_sec <= start_sec:
            continue
        skip_reason = _asr_folder_part_skip_reason(raw_part)
        parts.append(
            {
                "part_index": int(raw_part.get("part_index") or len(parts) + 1),
                "stem": str(raw_part.get("stem") or ""),
                "path": str(raw_part.get("path") or ""),
                "start_sec": start_sec,
                "end_sec": end_sec,
                "asr_silenced": skip_reason is not None,
                "asr_skip_reason": skip_reason,
            }
        )
    return sorted(parts, key=lambda part: (float(part["start_sec"]), int(part["part_index"])))


def _asr_folder_part_skip_reason(part: Mapping[str, Any] | None) -> str | None:
    if not isinstance(part, Mapping):
        return None
    raw_reason = str(part.get("asr_skip_reason") or "").strip()
    path_text = str(part.get("path") or part.get("stem") or "")
    detected_reason = folder_asr_part_skip_reason(path_text) if path_text else None
    if bool(part.get("asr_silenced")):
        return raw_reason or detected_reason or "asr_silenced"
    return detected_reason


def _asr_segment_reference_start(segment: Segment) -> float:
    if segment.source_script is not None:
        return float(segment.source_script.start)
    return float(segment.start)


def _asr_part_for_segment_start(
    parts: list[dict[str, Any]],
    segment_start: float,
) -> dict[str, Any] | None:
    for index, part in enumerate(parts):
        start_sec = float(part["start_sec"])
        end_sec = float(part["end_sec"])
        is_last = index == len(parts) - 1
        if start_sec <= segment_start < end_sec or (is_last and segment_start == end_sec):
            return part
    return None


def _asr_manual_review_file_metrics(manifest: PipelineManifest) -> list[dict[str, Any]]:
    parts = _asr_folder_input_parts(manifest)
    if not parts:
        return []
    stats: dict[int, dict[str, Any]] = {}
    for part in parts:
        part_index = int(part["part_index"])
        stats[part_index] = {
            "part_index": part_index,
            "stem": part["stem"],
            "path": part["path"],
            "start_sec": round(float(part["start_sec"]), 6),
            "end_sec": round(float(part["end_sec"]), 6),
            "segment_count": 0,
            "needs_manual_review": 0,
        }
    for segment in manifest.segments:
        part = _asr_part_for_segment_start(parts, _asr_segment_reference_start(segment))
        if part is None:
            continue
        part_stats = stats[int(part["part_index"])]
        part_stats["segment_count"] += 1
        if segment.status == "needs_manual_review":
            part_stats["needs_manual_review"] += 1

    metrics: list[dict[str, Any]] = []
    for part in parts:
        part_stats = stats[int(part["part_index"])]
        part_segment_count = int(part_stats["segment_count"])
        part_needs_manual_review = int(part_stats["needs_manual_review"])
        part_manual_review_rate = (
            part_needs_manual_review / part_segment_count if part_segment_count else 0.0
        )
        part_threshold = _asr_source_separation_manual_review_threshold(part_segment_count)
        metrics.append(
            {
                **part_stats,
                "manual_review_rate": round(part_manual_review_rate, 6),
                "manual_review_threshold": part_threshold,
                "recommended": (
                    part_segment_count > 0
                    and part_needs_manual_review >= part_threshold
                ),
            }
        )
    return metrics


def _source_separation_fallback_recommendation(
    manifest: PipelineManifest,
    *,
    cfg: Any,
    input_diagnostics: dict[str, Any],
    repair_summary: dict[str, Any],
    asr_review_summary: dict[str, Any],
) -> dict[str, Any]:
    selected = input_diagnostics.get("selected")
    selected_source = selected.get("source") if isinstance(selected, dict) else None
    selected_source_text = str(selected_source or "")
    backend = str(getattr(cfg, "source_separation_backend", "") or "")
    segment_count = len(manifest.segments)
    needs_manual_review = sum(
        1 for segment in manifest.segments if segment.status == "needs_manual_review"
    )
    no_speech_count = sum(
        1 for segment in manifest.segments if segment.status in NO_SPEECH_STATUSES
    )
    non_speech_texture_count = sum(
        1 for segment in manifest.segments if segment.status == "non_speech_texture"
    )
    error_counts: Counter[str] = Counter()
    for segment in manifest.segments:
        error_counts.update(str(error) for error in segment.errors if error)

    manual_review_rate = needs_manual_review / segment_count if segment_count else 0.0
    manual_review_threshold = _asr_source_separation_manual_review_threshold(segment_count)
    manual_review_file_metrics = _asr_manual_review_file_metrics(manifest)
    manual_review_file_trigger_count = sum(
        1 for item in manual_review_file_metrics if item.get("recommended")
    )
    repair_attempted = _asr_summary_int(repair_summary.get("attempted"))
    repair_repaired = _asr_summary_int(repair_summary.get("repaired"))
    asr_review_attempted = _asr_summary_int(asr_review_summary.get("attempted"))
    asr_review_failed = _asr_summary_int(asr_review_summary.get("failed"))
    metrics = {
        "source_separation_backend": backend,
        "asr_input_source": selected_source_text or None,
        "segment_count": segment_count,
        "needs_manual_review": needs_manual_review,
        "manual_review_rate": round(manual_review_rate, 6),
        "manual_review_threshold": manual_review_threshold,
        "manual_review_file_metrics": manual_review_file_metrics,
        "manual_review_file_trigger_count": manual_review_file_trigger_count,
        "no_speech_detected": no_speech_count,
        "non_speech_texture": non_speech_texture_count,
        "repair_attempted": repair_attempted,
        "repair_repaired": repair_repaired,
        "asr_review_attempted": asr_review_attempted,
        "asr_review_failed": asr_review_failed,
        "error_counts": dict(sorted(error_counts.items())),
    }
    thresholds = {
        "min_manual_review_rate": ASR_SOURCE_SEPARATION_FALLBACK_MIN_MANUAL_REVIEW_RATE,
        "min_manual_review_count": ASR_SOURCE_SEPARATION_FALLBACK_MIN_MANUAL_REVIEW_COUNT,
    }
    if (
        backend in {"demucs", "mock"}
        or selected_source_text in ASR_SOURCE_SEPARATION_FALLBACK_SEPARATED_SOURCES
    ):
        return {
            "recommended": False,
            "recommended_backend": None,
            "reason": "source_separation_already_used",
            "reasons": [],
            "metrics": metrics,
            "thresholds": thresholds,
        }

    reasons: list[str] = []
    has_file_metrics = any(item.get("segment_count", 0) for item in manual_review_file_metrics)
    manual_review_triggered = (
        manual_review_file_trigger_count > 0
        if has_file_metrics
        else needs_manual_review >= manual_review_threshold
    )
    if manual_review_triggered:
        reasons.append("manual_review_rate")

    recommended = bool(reasons)
    return {
        "recommended": recommended,
        "recommended_backend": "demucs" if recommended else None,
        "reason": (
            "raw_asr_quality_gate_failed"
            if recommended
            else "raw_asr_quality_within_threshold"
        ),
        "reasons": reasons,
        "metrics": metrics,
        "thresholds": thresholds,
    }


def _asr_segment_replacement_hits(
    segment: Segment,
    replacements_summary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if segment.source_script is None:
        return []
    segment_start = float(segment.source_script.start)
    segment_end = float(segment.source_script.end)
    segment_duration = max(0.001, segment_end - segment_start)
    segment_text = segment.source_script.text.strip()
    hits: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    for item in replacements_summary.get("items", []) or []:
        try:
            item_start = float(item.get("start"))
            item_end = float(item.get("end"))
        except (TypeError, ValueError):
            continue
        item_duration = max(0.001, item_end - item_start)
        overlap = max(0.0, min(segment_end, item_end) - max(segment_start, item_start))
        if overlap / min(segment_duration, item_duration) < 0.3:
            continue
        item_segment_id = item.get("segment_id")
        if item_segment_id is not None and str(item_segment_id) != segment.id:
            continue
        item_texts = [
            str(item.get(key) or "").strip()
            for key in ("replaced_text", "original_text")
            if str(item.get(key) or "").strip()
        ]
        if item_texts and not any(
            segment_text == item_text
            or item_text in segment_text
            or segment_text in item_text
            for item_text in item_texts
        ):
            continue
        for raw_hit in item.get("hits", []) or []:
            if not isinstance(raw_hit, Mapping):
                continue
            source = str(raw_hit.get("source") or "")
            target = str(raw_hit.get("target") or "")
            try:
                count = int(raw_hit.get("count") or 0)
            except (TypeError, ValueError):
                count = 0
            key = (source, target, count)
            if not source or key in seen:
                continue
            seen.add(key)
            hits.append({"source": source, "target": target, "count": count})
    return hits


def _asr_segment_repair_item(
    segment: Segment,
    repair_summary: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    source_script = segment.source_script
    segment_start = float(source_script.start) if source_script is not None else float(segment.start)
    segment_end = float(source_script.end) if source_script is not None else float(segment.end)
    segment_duration = max(0.001, segment_end - segment_start)
    best_item: Mapping[str, Any] | None = None
    best_overlap = 0.0
    for item in repair_summary.get("items", []) or []:
        if not isinstance(item, Mapping):
            continue
        try:
            item_start = float(item.get("start"))
            item_end = float(item.get("end"))
        except (TypeError, ValueError):
            continue
        item_duration = max(0.001, item_end - item_start)
        overlap = max(0.0, min(segment_end, item_end) - max(segment_start, item_start))
        if overlap / min(segment_duration, item_duration) < 0.3:
            continue
        if overlap > best_overlap:
            best_overlap = overlap
            best_item = item
    return best_item


def _asr_repair_item_preferred_candidate_text(item: Mapping[str, Any]) -> str | None:
    candidate_text = str(item.get("candidate_text") or "").strip()
    if candidate_text:
        return candidate_text
    attempts = [
        attempt
        for attempt in item.get("attempts", []) or []
        if isinstance(attempt, Mapping) and str(attempt.get("candidate_text") or "").strip()
    ]
    if not attempts:
        return None
    best_attempt = max(
        attempts,
        key=lambda attempt: float(attempt.get("score") or -math.inf),
    )
    return str(best_attempt.get("candidate_text") or "").strip() or None


def _asr_segment_countdown_verified(segment: Segment) -> bool:
    return _asr_segment_countdown_state(segment) == "verified"


def _asr_segment_countdown_state(segment: Segment) -> str:
    if isinstance(segment.analysis.get("asr_countdown_unverified"), Mapping):
        return "unverified"
    event = segment.analysis.get(COUNTDOWN_EVENT_KEY)
    if not isinstance(event, Mapping):
        return "none"
    raw_values = event.get("values")
    if not isinstance(raw_values, list) or not all(isinstance(value, int) for value in raw_values):
        return "none"
    values = [int(value) for value in raw_values]
    if not is_descending_countdown(values):
        return "none"
    raw_timeline = event.get("token_timeline")
    if isinstance(raw_timeline, list) and len(raw_timeline) == len(values):
        return "verified"
    return "missing_timeline"


def _asr_segment_warning_decision(segment: Segment) -> str | None:
    analysis_decisions = (
        ("asr_sparse_speech_unverified", "sparse_speech_unverified"),
        ("asr_numeric_sequence_unverified", "numeric_sequence_unverified"),
    )
    for key, decision in analysis_decisions:
        if isinstance(segment.analysis.get(key), Mapping):
            return decision

    quality_gate = segment.analysis.get("asr_quality_gate")
    raw_warnings = quality_gate.get("warnings") if isinstance(quality_gate, Mapping) else None
    if not isinstance(raw_warnings, list):
        return None
    warnings = {str(reason) for reason in raw_warnings if reason}
    if "asr_sparse_speech_unverified" in warnings:
        return "sparse_speech_unverified"
    if "asr_numeric_sequence_unverified" in warnings:
        return "numeric_sequence_unverified"
    return None


def _asr_segment_audit_item(
    segment: Segment,
    *,
    replacements_summary: Mapping[str, Any],
) -> dict[str, Any]:
    source_script = segment.source_script
    status = str(segment.status or "")
    errors = [str(error) for error in segment.errors if error]
    quality_gate = segment.analysis.get("asr_quality_gate")
    gate_reasons: list[str] = []
    if isinstance(quality_gate, Mapping):
        raw_reasons = quality_gate.get("reasons")
        if isinstance(raw_reasons, list):
            gate_reasons = [str(reason) for reason in raw_reasons if reason]
        raw_warnings = quality_gate.get("warnings")
        if isinstance(raw_warnings, list):
            gate_reasons.extend(str(reason) for reason in raw_warnings if reason)
    reasons = list(dict.fromkeys([*errors, *gate_reasons]))
    replacement_hits = _asr_segment_replacement_hits(segment, replacements_summary)
    countdown_state = _asr_segment_countdown_state(segment)
    countdown_verified = countdown_state == "verified"
    warning_decision = _asr_segment_warning_decision(segment)

    if status in {"needs_manual_review", "failed"} or any(
        reason.startswith(("asr_prompt_or_hallucination_leak", "asr_suspicious_pattern:"))
        for reason in reasons
    ):
        decision = "needs_review"
        severity = "severe"
    elif status == "non_speech_texture" or status in NO_SPEECH_STATUSES:
        decision = "texture"
        severity = "info"
    elif countdown_verified:
        decision = "countdown_verified"
        severity = "warning" if replacement_hits else "info"
    elif countdown_state in {"missing_timeline", "unverified"}:
        decision = "countdown_unverified"
        severity = "warning"
    elif warning_decision is not None:
        decision = warning_decision
        severity = "warning"
    else:
        decision = "auto_accept"
        severity = "warning" if replacement_hits else "info"

    item: dict[str, Any] = {
        "segment_id": segment.id,
        "start": round(float(segment.start), 3),
        "end": round(float(segment.end), 3),
        "status": status,
        "decision": decision,
        "severity": severity,
        "reasons": reasons,
        "replacement_hits": replacement_hits,
        "countdown_state": countdown_state,
        "countdown_verified": countdown_verified,
        "keep_original_texture": bool(segment.keep_original_texture),
    }
    if source_script is not None:
        item["source_text"] = source_script.text.strip()
        item["source_confidence"] = source_script.confidence
        item["source_backend"] = source_script.backend
    return item


def _asr_postprocess_candidate_text(source_text: str, cfg: Any) -> str | None:
    candidate = _asr_text_with_review_candidate_replacements(
        source_text,
        getattr(cfg, "asr_review_candidate_replacements", {}) or {},
    ).strip()
    if candidate and candidate != source_text.strip():
        return candidate
    return None


def _asr_postprocess_action(item: Mapping[str, Any], candidate_text: str | None) -> str:
    decision = str(item.get("decision") or "")
    status = str(item.get("status") or "")
    if candidate_text:
        return "candidate_review"
    if status in {"needs_manual_review", "failed"} or decision in {
        "needs_review",
        "countdown_unverified",
        "numeric_sequence_unverified",
        "sparse_speech_unverified",
    }:
        return "manual_review"
    if item.get("replacement_hits"):
        return "auto_replace"
    return "keep"


def _attach_asr_candidate_equivalence_fields(
    payload: dict[str, Any],
    *,
    source_text: str,
    candidate_text: str,
    cfg: Any,
) -> None:
    normalized_source_text = _normalize_asr_text_for_repair_equivalence(source_text, cfg)
    normalized_candidate_text = _normalize_asr_text_for_repair_equivalence(candidate_text, cfg)
    payload["normalized_candidate_text"] = normalized_candidate_text
    payload["equivalent_to_final_source"] = (
        bool(normalized_candidate_text)
        and normalized_candidate_text == normalized_source_text
    )


def _asr_postprocess_review_item(
    segment: Segment,
    *,
    cfg: Any,
    replacements_summary: Mapping[str, Any],
) -> dict[str, Any]:
    item = _asr_segment_audit_item(segment, replacements_summary=replacements_summary)
    source_text = str(item.get("source_text") or "").strip()
    candidate_text = _asr_postprocess_candidate_text(source_text, cfg) if source_text else None
    action = _asr_postprocess_action(item, candidate_text)
    payload: dict[str, Any] = {
        "segment_id": item["segment_id"],
        "start": item["start"],
        "end": item["end"],
        "status": item["status"],
        "action": action,
        "reasons": item["reasons"],
        "replacement_hits": item["replacement_hits"],
        "review_required": action in {"candidate_review", "manual_review"},
    }
    for key in ("source_text", "source_confidence", "source_backend"):
        if key in item:
            payload[key] = item[key]
    selected_candidate_text: str | None = None
    if (action == "auto_replace" or item.get("replacement_hits")) and source_text:
        selected_candidate_text = source_text
    elif candidate_text is not None:
        selected_candidate_text = candidate_text
    if selected_candidate_text is not None:
        payload["candidate_text"] = selected_candidate_text
        _attach_asr_candidate_equivalence_fields(
            payload,
            source_text=source_text,
            candidate_text=selected_candidate_text,
            cfg=cfg,
        )
    return payload


def _build_asr_postprocess_review_report(
    manifest: PipelineManifest,
    *,
    cfg: Any,
    replacements_summary: Mapping[str, Any],
) -> dict[str, Any]:
    all_items = [
        _asr_postprocess_review_item(
            segment,
            cfg=cfg,
            replacements_summary=replacements_summary,
        )
        for segment in manifest.segments
    ]
    action_counts = Counter(str(item["action"]) for item in all_items)
    report_items = [item for item in all_items if item["action"] != "keep"]
    return {
        "schema_version": 1,
        "summary": {
            "segment_count": len(all_items),
            "item_count": len(report_items),
            "auto_replace": int(action_counts["auto_replace"]),
            "candidate_review": int(action_counts["candidate_review"]),
            "manual_review": int(action_counts["manual_review"]),
            "keep": int(action_counts["keep"]),
        },
        "items": report_items,
    }


def _write_asr_postprocess_review_artifact(
    project_dir: Path,
    manifest: PipelineManifest,
    *,
    cfg: Any,
    replacements_summary: Mapping[str, Any],
) -> dict[str, Any]:
    report = _build_asr_postprocess_review_report(
        manifest,
        cfg=cfg,
        replacements_summary=replacements_summary,
    )
    report_path = project_dir / "work" / "transcribe" / "asr_postprocess_review.json"
    write_json_atomic(report_path, report)
    manifest.artifacts["asr_postprocess_review"] = str(report_path)
    return report


def _build_asr_high_risk_report(
    manifest: PipelineManifest,
    *,
    cfg: Any,
    replacements_summary: Mapping[str, Any],
    repair_summary: Mapping[str, Any],
    asr_review_summary: Mapping[str, Any],
    filtered_summary: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    all_items: list[dict[str, Any]] = []
    for segment in manifest.segments:
        item = _asr_segment_audit_item(segment, replacements_summary=replacements_summary)
        source_text = str(item.get("source_text") or "").strip()
        repair_item = _asr_segment_repair_item(segment, repair_summary)
        if source_text and repair_item is not None:
            candidate_text = _asr_repair_item_preferred_candidate_text(repair_item)
            if candidate_text:
                item["candidate_text"] = candidate_text
                item["repair_accepted"] = bool(repair_item.get("accepted"))
                _attach_asr_candidate_equivalence_fields(
                    item,
                    source_text=source_text,
                    candidate_text=candidate_text,
                    cfg=cfg,
                )
        all_items.append(item)
    report_items = [
        item
        for item in all_items
        if item["decision"] != "auto_accept" or item["severity"] != "info"
    ]
    decision_counts = Counter(str(item["decision"]) for item in all_items)
    severity_counts = Counter(str(item["severity"]) for item in all_items)
    blocking_reasons: list[str] = []
    if decision_counts["needs_review"]:
        blocking_reasons.append("needs_manual_review")
    if any(str(item.get("status")) == "failed" for item in all_items):
        blocking_reasons.append("failed_segments")
    severe_count = int(severity_counts["severe"])
    automated_dubbing_ready = severe_count == 0
    summary = {
        "segment_count": len(all_items),
        "auto_accept": int(decision_counts["auto_accept"]),
        "countdown_verified": int(decision_counts["countdown_verified"]),
        "countdown_unverified": int(decision_counts["countdown_unverified"]),
        "sparse_speech_unverified": int(decision_counts["sparse_speech_unverified"]),
        "numeric_sequence_unverified": int(decision_counts["numeric_sequence_unverified"]),
        "texture": int(decision_counts["texture"]),
        "needs_review": int(decision_counts["needs_review"]),
        "info": int(severity_counts["info"]),
        "warning": int(severity_counts["warning"]),
        "severe": severe_count,
        "replacement_item_count": len(replacements_summary.get("items", []) or []),
        "repair_attempted": _asr_summary_int(repair_summary.get("attempted")),
        "repair_repaired": _asr_summary_int(repair_summary.get("repaired")),
        "asr_review_attempted": _asr_summary_int(asr_review_summary.get("attempted")),
        "asr_review_guarded_auto_replaced": _asr_summary_int(
            asr_review_summary.get("guarded_auto_replaced")
        ),
        "asr_review_manual_review": _asr_summary_int(asr_review_summary.get("manual_review")),
        "filtered_chunk_count": len(list(filtered_summary or [])),
        "automated_dubbing_ready": automated_dubbing_ready,
        "blocking_reasons": blocking_reasons,
    }
    return {
        "schema_version": 1,
        "summary": summary,
        "items": report_items,
    }


def _write_asr_high_risk_report_artifact(
    project_dir: Path,
    manifest: PipelineManifest,
    *,
    cfg: Any,
    replacements_summary: Mapping[str, Any],
    repair_summary: Mapping[str, Any],
    asr_review_summary: Mapping[str, Any],
    filtered_summary: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    report = _build_asr_high_risk_report(
        manifest,
        cfg=cfg,
        replacements_summary=replacements_summary,
        repair_summary=repair_summary,
        asr_review_summary=asr_review_summary,
        filtered_summary=filtered_summary,
    )
    report_path = project_dir / "work" / "transcribe" / "asr_high_risk_report.json"
    write_json_atomic(report_path, report)
    manifest.artifacts["asr_high_risk_report"] = str(report_path)
    return report


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
    segment_retry_summary: dict[str, Any],
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
    source_separation_fallback = _source_separation_fallback_recommendation(
        manifest,
        cfg=cfg,
        input_diagnostics=input_diagnostics,
        repair_summary=repair_summary,
        asr_review_summary=asr_review_summary,
    )
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
        "segment_retry": segment_retry_summary,
        "text_replacements": replacements_summary,
        "filtered_chunks": filtered_summary,
        "qwen_repair_fallback": qwen_fallback_summary,
        "source_separation_fallback": source_separation_fallback,
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
        "raw_asr_word_chunk_count": sum(1 for chunk in raw_chunks if chunk.words),
        "repaired_asr_word_chunk_count": sum(1 for chunk in repaired_chunks if chunk.words),
        "final_asr_word_chunk_count": sum(1 for chunk in final_chunks if chunk.words),
        "final_asr_word_count": sum(len(chunk.words) for chunk in final_chunks),
        "repair_attempted": repair_summary.get("attempted", 0),
        "repair_repaired": repair_summary.get("repaired", 0),
        "segment_retry_attempted": segment_retry_summary.get("attempted", 0),
        "segment_retry_repaired": segment_retry_summary.get("repaired", 0),
        "asr_review_attempted": asr_review_summary.get("attempted", 0),
        "asr_review_replaced": asr_review_summary.get("replaced", 0),
        "text_replacements": replacements_summary,
        "filtered_chunk_count": len(filtered_summary),
        "needs_manual_review": sum(1 for segment in manifest.segments if segment.status == "needs_manual_review"),
        "no_speech_detected": sum(1 for segment in manifest.segments if segment.status in NO_SPEECH_STATUSES),
        "non_speech_texture": sum(1 for segment in manifest.segments if segment.status == "non_speech_texture"),
        "warnings": input_diagnostics.get("warnings", []),
        "qwen_repair_fallback": qwen_fallback_summary,
        "source_separation_fallback": source_separation_fallback,
        "recommend_source_separation_fallback": bool(
            source_separation_fallback.get("recommended")
        ),
        "recommended_source_separation_backend": source_separation_fallback.get(
            "recommended_backend"
        ),
    }
    diagnostics_path = transcribe_dir / "asr_diagnostics.json"
    summary_path = transcribe_dir / "asr_diagnostics_summary.json"
    write_json_atomic(diagnostics_path, diagnostics)
    write_json_atomic(summary_path, summary)
    manifest.artifacts["asr_diagnostics"] = str(diagnostics_path)
    manifest.artifacts["asr_diagnostics_summary"] = str(summary_path)


def _format_server_command(
    command: list[str],
    base_url: str,
    lane_index: int,
    parallel_slots: int = 1,
) -> list[str]:
    parsed = urlparse(base_url)
    host = parsed.hostname or ""
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    effective_parallel_slots = max(1, int(parallel_slots))
    return [
        str(part)
        .replace("{base_url}", base_url)
        .replace("{host}", host)
        .replace("{port}", str(port))
        .replace("{lane}", str(lane_index))
        .replace("{parallel}", str(effective_parallel_slots))
        for part in command
    ]


def _gemma_text_server_command(
    cfg: Any,
    *,
    base_url: str | None = None,
    lane_index: int = 0,
    include_mmproj: bool = False,
    parallel_slots: int = 1,
) -> list[str]:
    effective_base_url = base_url or cfg.gemma_text_server_url
    effective_parallel_slots = max(1, int(parallel_slots))
    if cfg.gemma_text_server_command:
        return _format_server_command(
            [str(part) for part in cfg.gemma_text_server_command],
            effective_base_url,
            lane_index,
            effective_parallel_slots,
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
        parallel_slots=effective_parallel_slots,
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
    max_gap_sec = float(getattr(cfg, "asr_resegment_merge_gap_sec", 1.0))
    seed_segments = [
        segment
        for segment in sorted(manifest.segments, key=lambda item: (item.start, item.end, item.id))
        if _segment_can_seed_voice_ref(segment, source_language)
    ]
    candidates = [
        segment
        for segment in seed_segments
        if not _voice_ref_segment_reject_reasons(segment, cfg)
    ]
    scored: list[tuple[tuple[float, ...], _VoiceRefSpan, AudioQualityMetrics | None]] = []
    for index, segment in enumerate(candidates):
        span = _build_voice_ref_span(candidates, index, ref_min_sec, ref_max_sec, max_gap_sec)
        if span is None:
            continue
        if _voice_ref_span_reject_reasons(span, seed_segments, max_gap_sec):
            continue
        span_duration_reasons = _voice_ref_span_audio_duration_reject_reasons(
            project_dir,
            span,
            min_sec=ref_min_sec,
            max_sec=ref_max_sec,
        )
        if span_duration_reasons:
            continue
        metrics = None
        try:
            source_audio = _resolve_project_read_path(project_dir, segment.audio_for_mix, "audio_for_mix")
            metrics = measure_source_voice_quality(source_audio)
        except Exception:
            pass
        scored.append((_voice_ref_span_score(span, metrics, cfg), span, metrics))
    selected = [
        (score, span, metrics)
        for score, span, metrics in sorted(scored, key=lambda item: item[0], reverse=True)
        if metrics is None or metrics.score >= getattr(cfg, "gsv_ref_min_quality_score", 0.25)
    ]
    if not selected:
        selected = sorted(scored, key=lambda item: item[0], reverse=True)
    prefer_plain = getattr(cfg, "gsv_few_shot_prefer_plain_text", True)
    if prefer_plain:
        plain_selected = [
            item
            for item in selected
            if _voice_ref_span_text_penalty(item[1], cfg) < 0.25
        ]
        if plain_selected:
            selected = plain_selected
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


def _voice_ref_segment_reject_reasons(segment: Segment, cfg: Any | None = None) -> tuple[str, ...]:
    del cfg
    reasons: list[str] = []
    analysis = segment.analysis if isinstance(segment.analysis, dict) else {}
    asr_gate = analysis.get("asr_quality_gate")
    if isinstance(asr_gate, dict):
        decision = str(asr_gate.get("decision") or "").strip().lower()
        if asr_gate.get("tts_blocked") is True:
            reasons.append("asr_quality_gate_tts_blocked")
        if decision in {"block", "blocked", "reject", "rejected", "manual_review"}:
            reasons.append(f"asr_quality_gate:{decision}")
    voice_training = analysis.get("voice_training")
    if isinstance(voice_training, dict):
        if voice_training.get("exclude") is True:
            reasons.append("voice_training_exclude")
        if voice_training.get("eligible") is False:
            reasons.append("voice_training_ineligible")
        if voice_training.get("clean_voice") is False:
            reasons.append("voice_training_unclean")
        effect_tags = voice_training.get("effect_tags")
        if isinstance(effect_tags, str):
            raw_tags = [effect_tags]
        elif isinstance(effect_tags, list):
            raw_tags = effect_tags
        else:
            raw_tags = []
        for tag in raw_tags:
            normalized = str(tag).strip().lower()
            if normalized and normalized not in {"none", "clean", "natural", "no_effect"}:
                reasons.append(f"voice_training_effect_tag:{normalized}")
    return tuple(dict.fromkeys(reasons))


def _voice_ref_span_reject_reasons(
    span: _VoiceRefSpan,
    neighboring_segments: Sequence[Segment] | None = None,
    max_gap_sec: float = 1.0,
) -> list[str]:
    reasons: list[str] = []
    prompt_text = _voice_ref_span_prompt_text(span)
    sparse_reason = _voice_ref_prompt_sparse_reject_reason(prompt_text, span.duration)
    if sparse_reason:
        reasons.append(sparse_reason)
    repetition_reason = _voice_ref_prompt_repetition_reject_reason(prompt_text)
    if repetition_reason:
        reasons.append(repetition_reason)
    for left, right in zip(span.segments, span.segments[1:], strict=False):
        left_text = left.source_script.text if left.source_script else ""
        right_text = right.source_script.text if right.source_script else ""
        if _source_split_inside_japanese_word(left_text, right_text):
            reasons.append(
                f"source_split_inside_japanese_word:{left.id}->{right.id}"
            )
    if neighboring_segments:
        sorted_neighbors = sorted(neighboring_segments, key=lambda item: (item.start, item.end, item.id))
        span_ids = {segment.id for segment in span.segments}
        first = span.segments[0]
        last = span.segments[-1]
        previous = None
        next_segment = None
        for index, segment in enumerate(sorted_neighbors):
            if segment.id == first.id:
                if index > 0:
                    previous = sorted_neighbors[index - 1]
                break
        for index, segment in enumerate(sorted_neighbors):
            if segment.id == last.id:
                if index + 1 < len(sorted_neighbors):
                    next_segment = sorted_neighbors[index + 1]
                break
        if previous is not None and previous.id not in span_ids:
            gap = max(0.0, first.start - previous.end)
            left_text = previous.source_script.text if previous.source_script else ""
            right_text = first.source_script.text if first.source_script else ""
            if gap <= max_gap_sec and _source_split_inside_japanese_word(left_text, right_text):
                reasons.append(
                    f"source_split_inside_japanese_word:{previous.id}->{first.id}"
                )
        if next_segment is not None and next_segment.id not in span_ids:
            gap = max(0.0, next_segment.start - last.end)
            left_text = last.source_script.text if last.source_script else ""
            right_text = next_segment.source_script.text if next_segment.source_script else ""
            if gap <= max_gap_sec and _source_split_inside_japanese_word(left_text, right_text):
                reasons.append(
                    f"source_split_inside_japanese_word:{last.id}->{next_segment.id}"
                )
    return reasons


def _voice_ref_prompt_compact_text(prompt_text: str) -> str:
    return re.sub(r"[\s、。,.!?！？…・♪♡❤ーｰ〜～]+", "", prompt_text.strip())


def _voice_ref_prompt_sparse_reject_reason(prompt_text: str, duration: float) -> str | None:
    compact = _voice_ref_prompt_compact_text(prompt_text)
    if not compact:
        return "prompt_text_empty"
    chars_per_sec = len(compact) / duration if duration > 0 else 0.0
    if duration >= 6.0 and chars_per_sec < 1.25:
        return "prompt_text_sparse_for_reference"
    return None


def _voice_ref_prompt_repetition_reject_reason(prompt_text: str) -> str | None:
    tokens = [
        token
        for token in re.split(r"[\s、。,.!?！？]+", prompt_text.strip())
        if token
    ]
    if len(tokens) >= 3 and len(set(tokens)) / len(tokens) <= 0.5:
        return "prompt_text_repetitive"
    compact = _voice_ref_prompt_compact_text(prompt_text)
    if _voice_ref_prompt_has_low_diversity_repetition(compact):
        return "prompt_text_repetitive"
    for unit_size in range(1, min(5, len(compact) // 2 + 1)):
        if len(compact) % unit_size:
            continue
        repeat_count = len(compact) // unit_size
        if repeat_count >= 3 and compact == compact[:unit_size] * repeat_count:
            return "prompt_text_repetitive"
    if _voice_ref_prompt_has_high_repeated_ngram_coverage(compact):
        return "prompt_text_repetitive"
    return None


def _voice_ref_prompt_has_low_diversity_repetition(compact: str) -> bool:
    if len(compact) < 12:
        return False
    counts = Counter(compact)
    unique_ratio = len(counts) / len(compact)
    top_two_ratio = sum(count for _, count in counts.most_common(2)) / len(compact)
    return unique_ratio <= 0.25 and top_two_ratio >= 0.65


def _voice_ref_prompt_has_high_repeated_ngram_coverage(compact: str) -> bool:
    if len(compact) < 6:
        return False
    max_unit_size = min(8, len(compact) // 2)
    for unit_size in range(2, max_unit_size + 1):
        for start in range(0, len(compact) - unit_size + 1):
            unit = compact[start : start + unit_size]
            if len(set(unit)) < 2:
                continue
            positions = [
                index
                for index in range(0, len(compact) - unit_size + 1)
                if compact.startswith(unit, index)
            ]
            if len(positions) < 2:
                continue
            non_overlapping: list[int] = []
            next_allowed = -1
            for position in positions:
                if position >= next_allowed:
                    non_overlapping.append(position)
                    next_allowed = position + unit_size
            if len(non_overlapping) < 2:
                continue
            coverage = len(non_overlapping) * unit_size / len(compact)
            required_coverage = 0.6 if unit_size <= 2 else 0.5
            if coverage >= required_coverage and (len(compact) <= 24 or len(non_overlapping) >= 3):
                return True
    return False


def _build_voice_ref_span(
    candidates: list[Segment],
    start_index: int,
    min_sec: float,
    max_sec: float,
    max_gap_sec: float,
) -> _VoiceRefSpan | None:
    segments: list[Segment] = []
    duration = 0.0
    for segment in candidates[start_index:]:
        if segments:
            gap = max(0.0, segment.start - segments[-1].end)
            if gap > max_gap_sec:
                return None if duration < min_sec else _VoiceRefSpan(tuple(segments), duration)
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
    cfg: Any | None = None,
) -> tuple[float, float, float, float]:
    first = span.segments[0]
    quality = metrics.score if metrics is not None else 0.5
    text_len = sum(len(segment.source_script.text.strip()) for segment in span.segments if segment.source_script)
    quality -= _voice_ref_span_text_penalty(span, cfg)
    return (
        max(0.0, min(1.0, quality)),
        span.duration,
        min(text_len / 80.0, 1.0),
        -first.start,
    )


def _voice_ref_span_text_penalty(span: _VoiceRefSpan, cfg: Any | None = None) -> float:
    text_len = sum(len(segment.source_script.text.strip()) for segment in span.segments if segment.source_script)
    prompt_text = _voice_ref_span_prompt_text(span)
    source_chars_per_sec = text_len / span.duration if span.duration > 0 else 0.0
    return _voice_ref_text_penalty(prompt_text, source_chars_per_sec, cfg)


def _voice_ref_text_penalty(
    prompt_text: str,
    source_chars_per_sec: float,
    cfg: Any | None = None,
) -> float:
    penalty = 0.0
    preferred = getattr(cfg, "gsv_few_shot_preferred_chars_per_sec", None) if cfg is not None else None
    maximum = getattr(cfg, "gsv_few_shot_max_chars_per_sec", None) if cfg is not None else None
    if preferred is not None and preferred > 0 and source_chars_per_sec > preferred:
        if maximum is not None and maximum > preferred:
            ratio = min(max((source_chars_per_sec - preferred) / (maximum - preferred), 0.0), 1.0)
        else:
            ratio = min(max((source_chars_per_sec - preferred) / preferred, 0.0), 1.0)
        penalty += 0.25 * ratio
    if re.search(r"^\s*\d+\s+", prompt_text):
        penalty += 0.18
    if re.search(
        r"(ピチッ|ピチョ|ピッチャ|ピキー|ピク|ぷしゃ|ぐちゃ|ぬめ|にゅる|ぎゅる|びく|喘ぎ|絶頂|貫く|貫通|触手|タイツ|電気ショック|電流)",
        prompt_text,
    ):
        penalty += 0.28
    if re.search(
        r"(変態|快感|気持ちよ|股間|股|睾丸|金玉|粘液|愛液|子宮|乳首|おっぱい|おまんこ|おちんちん|おちんぽ|チンポ|チンチン|ビンビン|勃起|フェラ|フェラチオ|エッチ|エロ|嫌ら|濡れ|尿道|お漏らし|メス|ロリ|痙攣|卵巣|バギナ|ピストン|媚薬|限界|出ない|行けない|侵され)",
        prompt_text,
    ):
        penalty += 0.28
    if prompt_text.count(" ") >= 2 and len(prompt_text.replace(" ", "")) / max(prompt_text.count(" ") + 1, 1) < 9:
        penalty += 0.10
    return min(penalty, 0.65)


def _voice_ref_span_prompt_text(span: _VoiceRefSpan) -> str:
    return " ".join(
        segment.source_script.text.strip()
        for segment in span.segments
        if segment.source_script and segment.source_script.text.strip()
    )


def _voice_ref_span_audio_duration_reject_reasons(
    project_dir: Path,
    span: _VoiceRefSpan,
    *,
    min_sec: float,
    max_sec: float,
) -> list[str]:
    actual_total = 0.0
    reasons: list[str] = []
    for segment in span.segments:
        try:
            source = _resolve_project_read_path(project_dir, segment.audio_for_mix, "audio_for_mix")
            actual = duration_sec(source)
        except Exception:
            continue
        actual_total += actual
        if abs(actual - segment.duration) > 0.10:
            reasons.append(
                f"audio_duration_mismatch:{segment.id}:{actual:.3f}!={segment.duration:.3f}"
            )
    if actual_total and actual_total < min_sec:
        reasons.append(f"actual_span_duration_below_ref_min:{actual_total:.3f}<{min_sec:.3f}")
    if actual_total > max_sec:
        reasons.append(f"actual_span_duration_above_ref_max:{actual_total:.3f}>{max_sec:.3f}")
    return reasons


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


def _gsv_training_speaker_ids(project_dir: Path, manifest: PipelineManifest, cfg: ProjectConfig) -> list[str]:
    speaker_ids = select_training_speaker_ids(project_dir, manifest, cfg)
    if len(speaker_ids) > 1:
        insufficient: list[str] = []
        for speaker_id in speaker_ids:
            try:
                select_training_items(project_dir, manifest, cfg, speaker_id=speaker_id)
            except GPTSoVITSError as exc:
                if "Not enough source voice data" not in str(exc):
                    raise
                insufficient.append(_gsv_speaker_insufficient_detail(speaker_id, exc))
        if insufficient:
            raise GPTSoVITSError(
                "source speaker sanity check failed before GPT-SoVITS few-shot training: "
                + "; ".join(insufficient)
                + ". Not enough source voice data for at least one project speaker_id; "
                "this often means source-speakers over-split a real speaker across multiple "
                "diarization labels. Re-run source-speakers after inspecting diarization, "
                "or use zero-shot fallback for insufficient training data."
            )
    return speaker_ids


def _gsv_speaker_insufficient_detail(speaker_id: str, exc: GPTSoVITSError) -> str:
    match = re.search(r"selected ([0-9.]+)s, need ([0-9.]+)s", str(exc))
    if not match:
        return f"{speaker_id}: {exc}"
    selected, needed = (float(match.group(1)), float(match.group(2)))
    return f"{speaker_id} selected {selected:.2f}s, need {needed:.2f}s"


def _rvc_training_speaker_ids(project_dir: Path, manifest: PipelineManifest, backend: str) -> list[str]:
    cfg = manifest.project_config
    speaker_ids: set[str] = set()
    strict_training_filter = backend == "command"
    for segment in manifest.segments:
        if segment.status in SKIP_STATUSES or not segment.speaker_id:
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
            if (
                check.accepted
                and not _rvc_train_text_reject_reasons(_source_script_chars_per_sec(segment), cfg)
                and check.source_audio_path
                and check.source_audio_path.exists()
            ):
                speaker_ids.add(segment.speaker_id)
            continue
        if not segment.audio_for_mix:
            continue
        try:
            source_path = _resolve_project_read_path(project_dir, segment.audio_for_mix, "audio_for_mix")
        except RightsError:
            continue
        if source_path.exists():
            speaker_ids.add(segment.speaker_id)
    return sorted(speaker_ids)


def _config_with_gsv_speaker_models(
    cfg: ProjectConfig,
    speaker_models: dict[str, GSVSpeakerConfig],
) -> ProjectConfig:
    payload = cfg.model_dump(mode="json")
    gsv_payload = dict(payload.get("gsv") or {})
    existing = dict(gsv_payload.get("speaker_models") or {})
    existing.update(
        {speaker_id: speaker_cfg.model_dump(mode="json") for speaker_id, speaker_cfg in speaker_models.items()}
    )
    gsv_payload["speaker_models"] = existing
    payload["gsv"] = gsv_payload
    return ProjectConfig.model_validate(payload)


def _config_with_rvc_speaker_models(
    cfg: ProjectConfig,
    speaker_models: dict[str, RVCSpeakerConfig],
) -> ProjectConfig:
    payload = cfg.model_dump(mode="json")
    rvc_payload = dict(payload.get("rvc") or {})
    existing = dict(rvc_payload.get("speaker_models") or {})
    existing.update(
        {speaker_id: speaker_cfg.model_dump(mode="json") for speaker_id, speaker_cfg in speaker_models.items()}
    )
    rvc_payload["speaker_models"] = existing
    payload["rvc"] = rvc_payload
    return ProjectConfig.model_validate(payload)


def _prepare_gsv_speaker_refs(
    project_dir: Path,
    manifest: PipelineManifest,
    cfg: ProjectConfig,
    speaker_id: str,
) -> Path:
    speaker_segments = [
        segment.model_copy(deep=True)
        for segment in manifest.segments
        if segment.speaker_id == speaker_id and segment.status not in SKIP_STATUSES
    ]
    speaker_manifest = manifest.model_copy(update={"segments": speaker_segments})
    selected_spans = _select_voice_ref_spans(project_dir, speaker_manifest, cfg)
    selected_span = selected_spans[0] if selected_spans else None
    if selected_span is None or not selected_span.segments[0].source_script:
        raise ValueError(
            f"Cannot prepare GPT-SoVITS refs for {speaker_id}: no transcribed clean span "
            f"within {cfg.gsv_ref_min_sec:.2f}-{cfg.gsv_ref_max_sec:.2f} seconds."
        )
    refs_dir = ensure_inside_project(project_dir, project_dir / "refs" / "speakers" / speaker_id)
    refs_dir.mkdir(parents=True, exist_ok=True)
    refs_path = ensure_inside_project(project_dir, refs_dir / "refs.json")
    aux_spans = selected_spans[1:]
    prompt_lang = selected_span.segments[0].source_script.language or cfg.source_language
    prompt_text, prompt_text_original, prompt_text_flags = _model_boundary_text_for_language(
        _voice_ref_span_prompt_text(selected_span),
        prompt_lang,
    )
    data: dict[str, dict[str, Any]] = {}
    for style in ("whisper_close", "sleepy"):
        raw_ref_path = f"refs/speakers/{speaker_id}/{style}.wav"
        ref_path = ensure_inside_project(project_dir, project_dir / raw_ref_path)
        _write_voice_ref_span(project_dir, selected_span, ref_path)
        aux_ref_audio_paths: list[str] = []
        for aux_index, aux_span in enumerate(aux_spans, start=1):
            aux_raw_path = f"refs/speakers/{speaker_id}/{style}_aux_{aux_index}.wav"
            aux_path = ensure_inside_project(project_dir, project_dir / aux_raw_path)
            _write_voice_ref_span(project_dir, aux_span, aux_path)
            aux_ref_audio_paths.append(aux_raw_path)
        data[style] = {
            "ref_audio_path": raw_ref_path,
            "prompt_text": prompt_text,
            "prompt_text_original": prompt_text_original,
            "prompt_lang": prompt_lang,
            "aux_ref_audio_paths": aux_ref_audio_paths,
            "source_language": cfg.source_language,
            "target_language": cfg.target_language,
            "speaker_id": speaker_id,
            "cross_lingual_role": "ja_source_prompt_for_ko_tts",
            "text_normalization": {
                "policy": "ja_hiragana" if _canonical_language(prompt_lang) == "ja" else "none",
                "risk_flags": prompt_text_flags,
            },
        }
    write_json_atomic(refs_path, data)
    return refs_path


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


def _rvc_train_dataset_manifest_path(project_dir: Path, dataset_dir: Path | None = None) -> Path:
    if dataset_dir is None or dataset_dir == project_dir / "work" / "rvc_train" / "dataset":
        return project_dir / "work" / "rvc_train" / "dataset_manifest.json"
    return dataset_dir.parent / "dataset_manifest.json"


RVC_TRAIN_STRICT_DEFAULT_MAX_CLIP_SEC = 10.0
RVC_TRAIN_STRICT_DEFAULT_MIN_SNR_DB = 20.0
RVC_TRAIN_STRICT_DEFAULT_MAX_BACKGROUND_BLEED_DB = -30.0
RVC_TRAIN_STRICT_DEFAULT_MAX_SIDE_TO_MID_DB = -12.0
RVC_TRAIN_STRICT_DEFAULT_MIN_QUALITY_SCORE = 0.60
RVC_TRAIN_RECOMMENDED_EPOCHS_BY_GRADE = {
    "excellent": 160,
    "good": 120,
    "mixed": 60,
    "poor": 25,
}


def _rvc_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _rvc_round(value: float | None) -> float | None:
    return round(value, 6) if value is not None else None


def _rvc_numeric_stats(values: Sequence[Any]) -> dict[str, Any]:
    numeric = sorted(value for value in (_rvc_float(item) for item in values) if value is not None)
    if not numeric:
        return {"count": 0}

    def percentile(fraction: float) -> float:
        if len(numeric) == 1:
            return numeric[0]
        position = (len(numeric) - 1) * fraction
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return numeric[lower]
        weight = position - lower
        return numeric[lower] * (1.0 - weight) + numeric[upper] * weight

    return {
        "count": len(numeric),
        "min": round(numeric[0], 6),
        "p10": round(percentile(0.10), 6),
        "median": round(percentile(0.50), 6),
        "mean": round(sum(numeric) / len(numeric), 6),
        "p90": round(percentile(0.90), 6),
        "max": round(numeric[-1], 6),
    }


def _rvc_stat(stats: dict[str, Any], key: str) -> float | None:
    return _rvc_float(stats.get(key))


def _rvc_stats_for_rows(rows: Sequence[dict[str, Any]], key: str) -> dict[str, Any]:
    return _rvc_numeric_stats([row.get(key) for row in rows])


def _rvc_speaker_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "dominant_speaker_id": None,
            "dominant_speaker_ratio": 0.0,
            "missing_speaker_ratio": 0.0,
            "speaker_counts": {},
        }
    speaker_ids = [str(row.get("speaker_id") or "") for row in rows]
    missing = sum(1 for speaker_id in speaker_ids if not speaker_id)
    counts = Counter(speaker_id for speaker_id in speaker_ids if speaker_id)
    dominant_speaker_id: str | None = None
    dominant_count = 0
    if counts:
        dominant_speaker_id, dominant_count = counts.most_common(1)[0]
    return {
        "dominant_speaker_id": dominant_speaker_id,
        "dominant_speaker_ratio": round(dominant_count / len(rows), 6),
        "missing_speaker_ratio": round(missing / len(rows), 6),
        "speaker_counts": dict(sorted(counts.items())),
    }


def _rvc_reject_reason_counts(rejected_rows: Sequence[dict[str, Any]] | None) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rejected_rows or []:
        reasons = row.get("reject_reasons")
        if not isinstance(reasons, Sequence) or isinstance(reasons, (str, bytes)):
            continue
        for reason in reasons:
            key = str(reason).split(":", 1)[0]
            counts[key] += 1
    return dict(sorted(counts.items()))


def _rvc_metric_passes_max(value: float | None, maximum: float) -> bool:
    return value is None or value <= maximum


def _rvc_metric_passes_min(value: float | None, minimum: float) -> bool:
    return value is not None and value >= minimum


def _rvc_dataset_quality_grade(
    *,
    clean_duration_sec: float,
    quality_stats: dict[str, Any],
    background_stats: dict[str, Any],
    side_stats: dict[str, Any],
    dominant_speaker_ratio: float,
    missing_speaker_ratio: float,
) -> str:
    quality_median = _rvc_stat(quality_stats, "median")
    quality_p10 = _rvc_stat(quality_stats, "p10")
    background_median = _rvc_stat(background_stats, "median")
    side_median = _rvc_stat(side_stats, "median")

    if (
        clean_duration_sec < 300.0
        or missing_speaker_ratio > 0.25
        or (quality_median is not None and quality_median < 0.55)
    ):
        return "poor"
    if (
        clean_duration_sec >= 600.0
        and _rvc_metric_passes_min(quality_median, 0.78)
        and _rvc_metric_passes_min(quality_p10, 0.62)
        and _rvc_metric_passes_max(background_median, -30.0)
        and _rvc_metric_passes_max(side_median, -12.0)
        and dominant_speaker_ratio >= 0.95
    ):
        return "excellent"
    if (
        clean_duration_sec >= 600.0
        and _rvc_metric_passes_min(quality_median, 0.70)
        and _rvc_metric_passes_min(quality_p10, 0.50)
        and _rvc_metric_passes_max(background_median, -25.0)
        and _rvc_metric_passes_max(side_median, -8.0)
        and dominant_speaker_ratio >= 0.90
    ):
        return "good"
    return "mixed"


def _rvc_recommended_epoch_count(quality_grade: str) -> int:
    return RVC_TRAIN_RECOMMENDED_EPOCHS_BY_GRADE.get(quality_grade, 60)


def _rvc_training_dataset_summary(
    rows: Sequence[dict[str, Any]],
    cfg: ProjectConfig,
    rejected_rows: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    durations: list[float] = []
    real_durations: list[float] = []
    augmented_durations: list[float] = []
    for row in rows:
        source_path = Path(row["source_path"])
        try:
            row_duration = duration_sec(source_path)
        except Exception:
            row_duration = 0.0
        durations.append(row_duration)
        if row.get("augmentation_method"):
            augmented_durations.append(row_duration)
        else:
            real_durations.append(row_duration)
    clean_duration = round(sum(durations), 6)
    clean_segment_count = len(rows)
    insufficient_reasons: list[str] = []
    if clean_segment_count < cfg.rvc_train_min_clean_segments:
        insufficient_reasons.append(
            f"clean_segment_count_below_min:{clean_segment_count}<{cfg.rvc_train_min_clean_segments}"
        )
    min_clean_sec = 0.0 if cfg.rvc_train_backend == "mock" else cfg.rvc_train_min_clean_sec
    if clean_duration + 1e-6 < min_clean_sec:
        insufficient_reasons.append(
            f"clean_duration_sec_below_min:{clean_duration:g}<{min_clean_sec:g}"
        )
    quality_stats = _rvc_stats_for_rows(rows, "quality_score")
    estimated_snr_stats = _rvc_stats_for_rows(rows, "estimated_snr_db")
    background_stats = _rvc_stats_for_rows(rows, "background_bleed_db")
    side_stats = _rvc_stats_for_rows(rows, "side_to_mid_db")
    cps_stats = _rvc_stats_for_rows(rows, "source_chars_per_sec")
    rank_stats = _rvc_stats_for_rows(rows, "training_rank_score")
    speaker_summary = _rvc_speaker_summary(rows)
    quality_grade = _rvc_dataset_quality_grade(
        clean_duration_sec=clean_duration,
        quality_stats=quality_stats,
        background_stats=background_stats,
        side_stats=side_stats,
        dominant_speaker_ratio=speaker_summary["dominant_speaker_ratio"],
        missing_speaker_ratio=speaker_summary["missing_speaker_ratio"],
    )
    return {
        "clean_segment_count": clean_segment_count,
        "clean_duration_sec": clean_duration,
        "min_clean_segments": cfg.rvc_train_min_clean_segments,
        "min_clean_sec": min_clean_sec,
        "insufficient": bool(insufficient_reasons),
        "insufficient_reasons": insufficient_reasons,
        "real_clean_segment_count": clean_segment_count - len(augmented_durations),
        "real_clean_duration_sec": round(sum(real_durations), 6),
        "augmented_segment_count": len(augmented_durations),
        "augmented_duration_sec": round(sum(augmented_durations), 6),
        "augmentation_enabled": cfg.rvc_train_augment_enabled,
        "augmentation_applied": bool(augmented_durations),
        "augmentation_max_multiplier": cfg.rvc_train_augment_max_multiplier,
        "augmentation_min_real_sec": cfg.rvc_train_augment_min_real_sec,
        "augmentation_methods": list(dict.fromkeys(row["augmentation_method"] for row in rows if row.get("augmentation_method"))),
        "quality_preset": cfg.rvc_train_quality_preset,
        "epoch_policy": cfg.rvc_train_epoch_policy,
        "quality_grade": quality_grade,
        "recommended_epoch_count": _rvc_recommended_epoch_count(quality_grade),
        "quality_score_stats": quality_stats,
        "estimated_snr_db_stats": estimated_snr_stats,
        "background_bleed_db_stats": background_stats,
        "side_to_mid_db_stats": side_stats,
        "source_chars_per_sec_stats": cps_stats,
        "training_rank_score_stats": rank_stats,
        "dominant_speaker_id": speaker_summary["dominant_speaker_id"],
        "dominant_speaker_ratio": speaker_summary["dominant_speaker_ratio"],
        "missing_speaker_ratio": speaker_summary["missing_speaker_ratio"],
        "speaker_counts": speaker_summary["speaker_counts"],
        "rejected_segment_count": len(rejected_rows or []),
        "reject_reason_counts": _rvc_reject_reason_counts(rejected_rows),
    }


def _maybe_augment_rvc_training_rows(
    rows: list[dict[str, Any]],
    cfg: ProjectConfig,
    *,
    dataset_dir: Path,
    force: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summary = _rvc_training_dataset_summary(rows, cfg)
    if not rows or not summary["insufficient"]:
        return rows, summary
    if not cfg.rvc_train_augment_enabled:
        summary["augmentation_skipped_reason"] = "disabled"
        return rows, summary
    if summary["real_clean_duration_sec"] + 1e-6 < cfg.rvc_train_augment_min_real_sec:
        summary["augmentation_skipped_reason"] = "real_clean_duration_below_augment_min"
        return rows, summary

    max_total_rows = max(len(rows), len(rows) * cfg.rvc_train_augment_max_multiplier)
    if max_total_rows <= len(rows):
        summary["augmentation_skipped_reason"] = "max_multiplier_exhausted"
        return rows, summary

    augmented_rows: list[dict[str, Any]] = []
    base_rows = list(rows)
    augment_dir = dataset_dir.parent / "augmented"
    augment_dir.mkdir(parents=True, exist_ok=True)
    for method in RVC_TRAIN_AUGMENT_METHODS:
        for row in base_rows:
            if len(rows) + len(augmented_rows) >= max_total_rows:
                break
            source_segment_id = row["segment_id"]
            augmented_id = f"{source_segment_id}_aug_{method}"
            augmented_source = augment_dir / f"{augmented_id}.wav"
            _write_rvc_augmented_audio(Path(row["source_path"]), augmented_source, method, force=force)
            augmented_row = {
                key: value
                for key, value in row.items()
                if key
                not in {
                    "segment_id",
                    "source_path",
                    "dataset_path",
                    "augmentation_method",
                    "augmentation_source_segment_id",
                }
            }
            augmented_row.update(
                {
                    "segment_id": augmented_id,
                    "speaker_id": row.get("speaker_id", ""),
                    "source_path": str(augmented_source),
                    "dataset_path": str(dataset_dir / f"{augmented_id}.wav"),
                    "augmentation_method": method,
                    "augmentation_source_segment_id": source_segment_id,
                }
            )
            augmented_rows.append(augmented_row)
            trial_rows = [*rows, *augmented_rows]
            trial_summary = _rvc_training_dataset_summary(trial_rows, cfg)
            if not trial_summary["insufficient"]:
                return trial_rows, trial_summary
        if len(rows) + len(augmented_rows) >= max_total_rows:
            break

    augmented = [*rows, *augmented_rows]
    summary = _rvc_training_dataset_summary(augmented, cfg)
    if summary["insufficient"]:
        summary["augmentation_skipped_reason"] = "max_multiplier_exhausted"
    return augmented, summary


def _write_rvc_augmented_audio(source_path: Path, output_path: Path, method: str, *, force: bool) -> None:
    if not force and output_path.exists():
        try:
            source_stat = source_path.stat()
            output_stat = output_path.stat()
            if output_stat.st_mtime_ns >= source_stat.st_mtime_ns and output_stat.st_size > 0:
                return
        except OSError:
            pass
    data, sample_rate = load_audio(source_path)
    augmented = _apply_rvc_training_augmentation(data, sample_rate, method)
    write_audio(output_path, augmented, sample_rate)


def _apply_rvc_training_augmentation(data: np.ndarray, sample_rate: int, method: str) -> np.ndarray:
    audio = np.asarray(data, dtype=np.float32)
    if method == "gain_minus3":
        return _rvc_training_peak_guard(audio * (10 ** (-3.0 / 20.0)))
    if method == "gain_plus3":
        return _rvc_training_peak_guard(audio * (10 ** (3.0 / 20.0)))
    if method == "highpass_80":
        return _rvc_training_peak_guard(audio - _rvc_training_moving_average(audio, sample_rate / 80.0))
    if method == "lowpass_7600":
        return _rvc_training_peak_guard(_rvc_training_moving_average(audio, sample_rate / 7600.0))
    raise RVCCommandError(f"Unknown RVC training augmentation method: {method}")


def _rvc_training_moving_average(data: np.ndarray, window: float) -> np.ndarray:
    window_size = max(1, int(round(window)))
    if window_size <= 1 or len(data) <= 1:
        return data.astype(np.float32, copy=True)
    kernel = np.ones(window_size, dtype=np.float32) / float(window_size)
    channels = [
        np.convolve(data[:, channel], kernel, mode="same")
        for channel in range(data.shape[1])
    ]
    return np.stack(channels, axis=1).astype(np.float32)


def _rvc_training_peak_guard(data: np.ndarray, peak_limit: float = 0.95) -> np.ndarray:
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    if peak > peak_limit > 0:
        data = data * (peak_limit / peak)
    return np.clip(data, -1.0, 1.0).astype(np.float32)


def _write_rvc_train_dataset_manifest(
    project_dir: Path,
    dataset_dir: Path | None,
    *,
    rows: Sequence[dict[str, Any]],
    rejected_rows: Sequence[dict[str, Any]],
    summary: dict[str, Any],
) -> Path:
    manifest_path = _rvc_train_dataset_manifest_path(project_dir, dataset_dir)
    write_json_atomic(
        manifest_path,
        {
            "segments": list(rows),
            "rejected_segments": list(rejected_rows),
            "summary": summary,
        },
    )
    return manifest_path


def _is_rvc_insufficient_training_data(exc: Exception) -> bool:
    return str(exc).startswith(RVC_INSUFFICIENT_TRAINING_DATA_PREFIX)


def _dedupe_rvc_training_rows(
    rows: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept_by_source: dict[Path, dict[str, Any]] = {}
    duplicate_rows: list[dict[str, Any]] = []
    for row in rows:
        source_key = Path(row["source_path"]).resolve()
        kept = kept_by_source.get(source_key)
        if kept is None:
            kept_by_source[source_key] = dict(row)
            continue
        if _rvc_training_row_is_better(row, kept):
            duplicate_rows.append(_rvc_duplicate_reject_row(kept, row))
            kept_by_source[source_key] = dict(row)
            continue
        duplicate_rows.append(_rvc_duplicate_reject_row(row, kept))
    kept_ids = {row["segment_id"] for row in kept_by_source.values()}
    return [dict(row) for row in rows if row["segment_id"] in kept_ids], duplicate_rows


def _rvc_training_row_is_better(candidate: dict[str, Any], current: dict[str, Any]) -> bool:
    return _rvc_training_row_score(candidate) > _rvc_training_row_score(current)


def _rvc_training_row_score(row: dict[str, Any]) -> tuple[float, float]:
    quality = row.get("training_rank_score", row.get("quality_score"))
    if not isinstance(quality, (int, float)):
        quality = -1.0
    duration = row.get("duration_sec")
    if not isinstance(duration, (int, float)):
        duration = 0.0
    return (float(quality), float(duration))


def _source_script_chars_per_sec(segment: Segment) -> float | None:
    if segment.duration <= 0 or not segment.source_script:
        return None
    text = "".join(
        char
        for char in segment.source_script.text
        if unicodedata.category(char)[0] in {"L", "N"}
    )
    if not text:
        return None
    return len(text) / segment.duration


def _rvc_train_text_reject_reasons(
    source_chars_per_sec: float | None,
    cfg: ProjectConfig,
) -> tuple[str, ...]:
    max_chars_per_sec = getattr(cfg, "rvc_train_max_chars_per_sec", None)
    if source_chars_per_sec is None or max_chars_per_sec is None or max_chars_per_sec <= 0:
        return ()
    if source_chars_per_sec <= max_chars_per_sec:
        return ()
    return (
        "source_chars_per_sec_above_max:"
        f"{source_chars_per_sec:.3f}>{float(max_chars_per_sec):.3f}",
    )


def _rvc_training_quality_tier(rank_score: float | None) -> str | None:
    if rank_score is None:
        return None
    if rank_score >= 0.80:
        return "excellent"
    if rank_score >= 0.65:
        return "good"
    if rank_score >= 0.45:
        return "mixed"
    return "poor"


def _rvc_training_rank_score(row: dict[str, Any], cfg: ProjectConfig) -> float:
    score = _rvc_float(row.get("quality_score"))
    if score is None:
        score = 0.50

    source_chars_per_sec = _rvc_float(row.get("source_chars_per_sec"))
    preferred_chars_per_sec = _rvc_float(getattr(cfg, "rvc_train_preferred_chars_per_sec", None))
    max_chars_per_sec = _rvc_float(getattr(cfg, "rvc_train_max_chars_per_sec", None))
    if source_chars_per_sec is not None and preferred_chars_per_sec is not None and source_chars_per_sec > preferred_chars_per_sec:
        hard_max = max(max_chars_per_sec or preferred_chars_per_sec, preferred_chars_per_sec + 1e-6)
        score -= min(0.18, 0.18 * ((source_chars_per_sec - preferred_chars_per_sec) / (hard_max - preferred_chars_per_sec)))

    estimated_snr_db = _rvc_float(row.get("estimated_snr_db"))
    if estimated_snr_db is not None and estimated_snr_db < 20.0:
        score -= min(0.12, (20.0 - estimated_snr_db) / 100.0)

    background_bleed_db = _rvc_float(row.get("background_bleed_db"))
    if background_bleed_db is not None and background_bleed_db > -30.0:
        score -= min(0.18, (background_bleed_db + 30.0) / 80.0)

    side_to_mid_db = _rvc_float(row.get("side_to_mid_db"))
    if side_to_mid_db is not None and side_to_mid_db > -12.0:
        score -= min(0.12, (side_to_mid_db + 12.0) / 80.0)

    duration = _rvc_float(row.get("duration_sec"))
    max_clip_sec = _rvc_float(getattr(cfg, "rvc_train_max_clip_sec", None))
    if duration is not None and max_clip_sec is not None and duration > max_clip_sec:
        score -= min(0.10, (duration - max_clip_sec) / max(max_clip_sec * 10.0, 1.0))

    return round(max(0.0, min(1.0, score)), 6)


def _rvc_training_audit_fields(
    metrics: AudioQualityMetrics | None,
    clean_source_metrics: dict[str, float | None],
    source_chars_per_sec: float | None,
    cfg: ProjectConfig,
) -> dict[str, Any]:
    row: dict[str, Any] = {}
    if metrics is not None:
        row.update(
            {
                "quality_score": round(metrics.score, 6),
                "peak_dbfs": round(metrics.peak_dbfs, 6),
                "rms_dbfs": round(metrics.rms_dbfs, 6),
                "clipping_ratio": round(metrics.clipping_ratio, 6),
                "leading_silence_sec": round(metrics.leading_silence_sec, 6),
                "trailing_silence_sec": round(metrics.trailing_silence_sec, 6),
                "active_ratio": round(metrics.active_ratio, 6),
                "silence_ratio": round(metrics.silence_ratio, 6),
                "estimated_snr_db": _rvc_round(metrics.estimated_snr_db),
                "quality_issues": list(metrics.issues),
            }
        )
    if clean_source_metrics:
        row["background_bleed_db"] = _rvc_round(clean_source_metrics.get("background_bleed_db"))
        row["side_to_mid_db"] = _rvc_round(clean_source_metrics.get("side_to_mid_db"))
    if source_chars_per_sec is not None:
        row["source_chars_per_sec"] = round(source_chars_per_sec, 6)
    rank_score = _rvc_training_rank_score(row, cfg)
    row["training_rank_score"] = rank_score
    row["training_quality_tier"] = _rvc_training_quality_tier(rank_score)
    return row


def _rvc_train_policy_reject_reasons(row: dict[str, Any], cfg: ProjectConfig) -> tuple[str, ...]:
    if cfg.rvc_train_quality_preset != "strict":
        return ()
    reasons: list[str] = []
    duration = _rvc_float(row.get("duration_sec"))
    max_clip_sec = _rvc_float(cfg.rvc_train_max_clip_sec) or RVC_TRAIN_STRICT_DEFAULT_MAX_CLIP_SEC
    if duration is not None and duration > max_clip_sec:
        reasons.append(f"rvc_train_duration_sec_above_max:{duration:.3f}>{max_clip_sec:.3f}")

    quality_score = _rvc_float(row.get("quality_score"))
    min_quality_score = max(float(cfg.rvc_train_min_quality_score), RVC_TRAIN_STRICT_DEFAULT_MIN_QUALITY_SCORE)
    if quality_score is not None and quality_score < min_quality_score:
        reasons.append(f"rvc_train_quality_score_below_strict_min:{quality_score:.3f}<{min_quality_score:.3f}")

    estimated_snr_db = _rvc_float(row.get("estimated_snr_db"))
    min_snr_db = _rvc_float(cfg.rvc_train_min_snr_db) or RVC_TRAIN_STRICT_DEFAULT_MIN_SNR_DB
    if estimated_snr_db is not None and estimated_snr_db < min_snr_db:
        reasons.append(f"rvc_train_estimated_snr_db_below_min:{estimated_snr_db:.3f}<{min_snr_db:.3f}")

    background_bleed_db = _rvc_float(row.get("background_bleed_db"))
    max_background_bleed_db = (
        _rvc_float(cfg.rvc_train_max_background_bleed_db)
        if cfg.rvc_train_max_background_bleed_db is not None
        else RVC_TRAIN_STRICT_DEFAULT_MAX_BACKGROUND_BLEED_DB
    )
    if background_bleed_db is not None and background_bleed_db > max_background_bleed_db:
        reasons.append(
            f"rvc_train_background_bleed_db_above_max:{background_bleed_db:.3f}>{max_background_bleed_db:.3f}"
        )

    side_to_mid_db = _rvc_float(row.get("side_to_mid_db"))
    max_side_to_mid_db = (
        _rvc_float(cfg.rvc_train_max_side_to_mid_db)
        if cfg.rvc_train_max_side_to_mid_db is not None
        else RVC_TRAIN_STRICT_DEFAULT_MAX_SIDE_TO_MID_DB
    )
    if side_to_mid_db is not None and side_to_mid_db > max_side_to_mid_db:
        reasons.append(f"rvc_train_side_to_mid_db_above_max:{side_to_mid_db:.3f}>{max_side_to_mid_db:.3f}")
    return tuple(reasons)


def _rvc_rejected_training_row(row: dict[str, Any], reject_reasons: Sequence[str]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "segment_id": row.get("segment_id"),
        "source_path": row.get("source_path") or None,
        "reject_reasons": list(dict.fromkeys(reject_reasons)),
    }
    for key in (
        "speaker_id",
        "analysis_speaker_count",
        "duration_sec",
        "quality_score",
        "quality_issues",
        "peak_dbfs",
        "rms_dbfs",
        "clipping_ratio",
        "leading_silence_sec",
        "trailing_silence_sec",
        "active_ratio",
        "silence_ratio",
        "estimated_snr_db",
        "background_bleed_db",
        "side_to_mid_db",
        "source_chars_per_sec",
        "source_text",
        "source_text_original",
        "source_text_language",
        "text_normalization",
        "training_rank_score",
        "training_quality_tier",
    ):
        if key in row:
            payload[key] = row.get(key)
    return payload


def _rvc_train_effective_epoch_config(
    cfg: ProjectConfig,
    dataset_summary: dict[str, Any],
) -> tuple[ProjectConfig, dict[str, Any]]:
    configured_epochs = int(cfg.rvc_train_epochs)
    recommended_epochs = int(dataset_summary.get("recommended_epoch_count") or configured_epochs)
    auto_min = int(cfg.rvc_train_auto_epoch_min)
    auto_max = int(cfg.rvc_train_auto_epoch_max)
    effective_epochs = configured_epochs
    if cfg.rvc_train_epoch_policy == "auto":
        effective_epochs = max(auto_min, min(auto_max, recommended_epochs))
    decision = {
        "policy": cfg.rvc_train_epoch_policy,
        "quality_preset": cfg.rvc_train_quality_preset,
        "quality_grade": dataset_summary.get("quality_grade"),
        "configured_epochs": configured_epochs,
        "recommended_epoch_count": recommended_epochs,
        "effective_epochs": effective_epochs,
        "auto_epoch_min": auto_min,
        "auto_epoch_max": auto_max,
    }
    if effective_epochs == configured_epochs:
        return cfg, decision
    payload = cfg.model_dump(mode="json")
    rvc_payload = dict(payload.get("rvc") or {})
    rvc_payload["train_epochs"] = effective_epochs
    payload["rvc"] = rvc_payload
    return ProjectConfig.model_validate(payload), decision


def _rvc_train_dataset_summary_from_manifest(project_dir: Path, dataset_dir: Path | None) -> dict[str, Any]:
    dataset_manifest_path = _rvc_train_dataset_manifest_path(project_dir, dataset_dir)
    if not dataset_manifest_path.exists():
        return {}
    try:
        payload = json.loads(dataset_manifest_path.read_text("utf-8"))
    except Exception:
        return {}
    summary = payload.get("summary")
    return summary if isinstance(summary, dict) else {}


def _rvc_duplicate_reject_row(row: dict[str, Any], kept: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "segment_id": row["segment_id"],
        "source_path": row["source_path"],
        "reject_reasons": [f"duplicate_source_audio:{kept['segment_id']}"],
        "speaker_id": row.get("speaker_id") or "",
        "quality_score": row.get("quality_score"),
        "source_chars_per_sec": row.get("source_chars_per_sec"),
    }
    for key in (
        "estimated_snr_db",
        "background_bleed_db",
        "side_to_mid_db",
        "training_rank_score",
        "training_quality_tier",
    ):
        if key in row:
            payload[key] = row.get(key)
    return payload


def _trim_rvc_training_rows_to_target(
    rows: Sequence[dict[str, Any]],
    cfg: ProjectConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    target_clean_sec = _rvc_float(cfg.rvc_train_target_clean_sec)
    if target_clean_sec is None or target_clean_sec <= 0:
        return [dict(row) for row in rows], []
    current_duration = sum(_rvc_float(row.get("duration_sec")) or 0.0 for row in rows)
    if current_duration <= target_clean_sec + 1e-6:
        return [dict(row) for row in rows], []

    ranked_rows = sorted(
        (dict(row) for row in rows),
        key=lambda row: (
            -(_rvc_float(row.get("training_rank_score")) or 0.0),
            -float(row.get("duration_sec") if isinstance(row.get("duration_sec"), (int, float)) else 0.0),
            str(row.get("segment_id") or ""),
        ),
    )
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    kept_duration = 0.0
    for row in ranked_rows:
        if kept and kept_duration >= target_clean_sec:
            rejected.append(_rvc_rejected_training_row(row, [f"target_clean_sec_trimmed:{target_clean_sec:.3f}"]))
            continue
        kept.append(row)
        kept_duration += _rvc_float(row.get("duration_sec")) or 0.0
    kept_ids = {row["segment_id"] for row in kept}
    ordered_kept = [dict(row) for row in rows if row["segment_id"] in kept_ids]
    return ordered_kept, rejected


def _rvc_train_dataset(
    project_dir: Path,
    manifest: PipelineManifest,
    force: bool,
    *,
    speaker_id: str | None = None,
    dataset_dir: Path | None = None,
) -> tuple[Path, list[dict[str, Any]]]:
    dataset_dir = dataset_dir or project_dir / "work" / "rvc_train" / "dataset"
    if force and dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    augment_dir = dataset_dir.parent / "augmented"
    if force and augment_dir.exists():
        shutil.rmtree(augment_dir)
    stale_manifest = dataset_dir / "dataset_manifest.json"
    if stale_manifest.exists():
        stale_manifest.unlink()
    rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    cfg = manifest.project_config
    strict_training_filter = cfg.rvc_train_backend == "command"
    for segment in manifest.segments:
        if speaker_id is not None and segment.speaker_id != speaker_id:
            continue
        source_chars_per_sec = _source_script_chars_per_sec(segment)
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
            row = {
                "segment_id": segment.id,
                "speaker_id": segment.speaker_id or "",
                "source_path": str(check.source_audio_path) if check.source_audio_path else "",
                "dataset_path": str(dataset_dir / f"{segment.id}.wav"),
                "duration_sec": round(segment.duration, 6),
                "analysis_speaker_count": segment.analysis.get("speaker_count"),
            }
            row.update(_source_script_text_audit_fields(segment))
            row.update(
                _rvc_training_audit_fields(
                    check.metrics,
                    check.clean_source_metrics,
                    source_chars_per_sec,
                    cfg,
                )
            )
            reject_reasons = [
                *check.reject_reasons,
                *_rvc_train_text_reject_reasons(source_chars_per_sec, cfg),
                *_rvc_train_policy_reject_reasons(row, cfg),
            ]
            if reject_reasons:
                rejected_rows.append(_rvc_rejected_training_row(row, reject_reasons))
                continue
            if not check.source_audio_path:
                raise RVCCommandError(f"train-rvc source segment audio is missing for {segment.id}")
            source_path = check.source_audio_path
        else:
            source_path = _resolve_project_read_path(project_dir, segment.audio_for_mix, "audio_for_mix")
            row = {
                "segment_id": segment.id,
                "speaker_id": segment.speaker_id or "",
                "source_path": str(source_path),
                "dataset_path": str(dataset_dir / f"{segment.id}.wav"),
                "duration_sec": round(segment.duration, 6),
            }
            row.update(_source_script_text_audit_fields(segment))
            if source_chars_per_sec is not None:
                row["source_chars_per_sec"] = round(source_chars_per_sec, 6)
            rank_score = _rvc_training_rank_score(row, cfg)
            row["training_rank_score"] = rank_score
            row["training_quality_tier"] = _rvc_training_quality_tier(rank_score)
        if not source_path.exists():
            raise RVCCommandError(f"train-rvc source segment audio is missing: {source_path}")
        rows.append(row)
    rows, duplicate_rows = _dedupe_rvc_training_rows(rows)
    rejected_rows.extend(duplicate_rows)
    rows, target_trimmed_rows = _trim_rvc_training_rows_to_target(rows, cfg)
    rejected_rows.extend(target_trimmed_rows)
    summary = _rvc_training_dataset_summary(rows, cfg, rejected_rows)
    if not rows:
        _write_rvc_train_dataset_manifest(
            project_dir,
            dataset_dir,
            rows=rows,
            rejected_rows=rejected_rows,
            summary=summary,
        )
        raise RVCCommandError(
            f"{RVC_INSUFFICIENT_TRAINING_DATA_PREFIX} "
            "requires at least one clean source voice segment audio file."
        )
    rows, summary = _maybe_augment_rvc_training_rows(rows, cfg, dataset_dir=dataset_dir, force=force)
    augmentation_skipped_reason = summary.get("augmentation_skipped_reason")
    summary = _rvc_training_dataset_summary(rows, cfg, rejected_rows)
    if augmentation_skipped_reason is not None:
        summary["augmentation_skipped_reason"] = augmentation_skipped_reason
    if summary["insufficient"]:
        _write_rvc_train_dataset_manifest(
            project_dir,
            dataset_dir,
            rows=rows,
            rejected_rows=rejected_rows,
            summary=summary,
        )
        raise RVCCommandError(
            f"{RVC_INSUFFICIENT_TRAINING_DATA_PREFIX} "
            + ", ".join(summary["insufficient_reasons"])
        )
    if speaker_id is None:
        _ensure_single_rvc_training_speaker(rows)
    for row in rows:
        source_path = Path(row["source_path"])
        output_path = Path(row["dataset_path"])
        source_stat = source_path.stat()
        if (
            force
            or not output_path.exists()
            or output_path.stat().st_size != source_stat.st_size
            or output_path.stat().st_mtime_ns != source_stat.st_mtime_ns
        ):
            shutil.copy2(source_path, output_path)
    accepted_names = {Path(row["dataset_path"]).name for row in rows}
    for stale_wav in dataset_dir.glob("*.wav"):
        if stale_wav.name not in accepted_names:
            stale_wav.unlink()
    _write_rvc_train_dataset_manifest(
        project_dir,
        dataset_dir,
        rows=rows,
        rejected_rows=rejected_rows,
        summary=summary,
    )
    return dataset_dir, rows


def _ensure_single_rvc_training_speaker(rows: Sequence[dict[str, Any]]) -> None:
    by_speaker: dict[str, list[str]] = {}
    for row in rows:
        speaker_id = row.get("speaker_id") or ""
        if not speaker_id:
            continue
        by_speaker.setdefault(speaker_id, []).append(row["segment_id"])
    if len(by_speaker) <= 1:
        return
    details = ", ".join(
        f"{speaker_id}({','.join(segment_ids[:5])})"
        for speaker_id, segment_ids in sorted(by_speaker.items())
    )
    raise RVCCommandError(
        "speaker_id_mismatch: train-rvc requires one speaker_id, "
        f"but accepted training segments include multiple speakers: {details}"
    )


def _train_rvc_ready_for_rvc(manifest: PipelineManifest) -> bool:
    status = manifest.stage_state.get("train-rvc", {}).get("status")
    if status == "completed":
        return True
    if status == "skipped_pretrained_voice_bank":
        return bool(manifest.project_config.rvc_speaker_models)
    if status == "skipped_insufficient_training_data":
        cfg = manifest.project_config
        return bool(cfg.rvc_allow_pre_rvc_fallback and not _rvc_downstream_required(cfg))
    return False


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
