from __future__ import annotations

from datetime import UTC, datetime
from importlib import resources
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = "1.0"

SegmentStatus = Literal[
    "raw",
    "transcribed",
    "analyzed",
    "scripted",
    "synthesized",
    "rvc_trained",
    "rvc_converted",
    "ok",
    "needs_regeneration",
    "needs_manual_review",
    "absorbed",
    "no_speech_detected",
    "non_speech_texture",
    "failed",
]
Emotion = Literal["gentle", "sleepy", "reassuring", "playful", "serious", "neutral"]
Pace = Literal["very_slow", "slow", "normal", "slightly_fast"]
Volume = Literal["whisper", "soft", "normal"]
SpatialStyle = Literal[
    "center",
    "left_close",
    "right_close",
    "center_close",
    "center_far",
    "sleepy_center",
    "binaural_sweep",
    "ambient",
]
QCRecommendation = Literal["pass", "regenerate", "manual_review"]
DurationRetryAction = Literal["none", "request_shorter_script", "request_longer_script"]
VariationRetryAction = Literal["none", "regenerate_with_variation_and_new_seed"]
MixProfile = Literal["asmr_stereo"]
MixBackgroundBed = Literal["preserve_original", "dialogue_only"]
MixLoudnessStrategy = Literal["peak_guard_only", "none"]
SourceLanguage = Literal["ja"]
TargetLanguage = Literal["ko"]
GSVGPTWeightsPolicy = Literal["auto", "explicit", "few_shot", "base_for_korean", "unchanged"]
GSVSoVITSWeightsPolicy = Literal["auto", "explicit", "few_shot", "unchanged"]
GSVRefMode = Literal["static", "segment", "auto"]
GSVDurationRewriteBackend = Literal["none", "gemma"]
GSVDurationRewriteTiming = Literal["after_initial", "before_zero_shot"]
GSVTerminalFailurePolicy = Literal["keep_original", "fail"]
GSVTimingQualityGate = Literal["good", "warn", "fail", "unknown"]
SpeakerAssignmentBackend = Literal["none", "mock", "pyannote"]
ASRPreset = Literal["default", "conservative", "whisper", "no_vad_repair"]

DEFAULT_COUNTDOWN_CARRIER_TOKEN_TEMPLATES: dict[str, list[str]] = {
    "십": ["십 번만요."],
    "구": ["구팔칠.", "구, 팔, 칠.", "{token}, 살짝만."],
    "팔": ["정답은 팔, 여기.", "팔, 작게요.", "팔 초만요."],
    "칠": ["칠 초만요.", "칠, 작게요.", "7."],
    "육": ["육 초만요.", "육, 작게요.", "6, 육."],
    "오": ["오 초만요.", "오, 작게요.", "오, 천천히요."],
    "사": ["사아 번만요."],
    "삼": ["삼 번만요.", "삼 초만요.", "삼 하고요."],
    "이": ["이 하고요.", "이 번만요."],
    "일": ["1.", "일, 천천히요.", "일 번만요.", "일, 일."],
    "영": ["영 하고요.", "영 번만요.", "영 초만요."],
}


def utc_now() -> datetime:
    return datetime.now(UTC)


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


def validate_file_safe_id(value: str, field_name: str = "id") -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    if any(char in normalized for char in ("/", "\\", "\x00")):
        raise ValueError(f"{field_name} must be file-safe")
    if normalized in {".", ".."}:
        raise ValueError(f"{field_name} must be file-safe")
    return normalized


class RVCProfile(StrictBaseModel):
    name: str
    f0_method: str = "rmvpe"
    index_rate: float = Field(default=0.45, ge=0.0, le=1.0)
    f0_up_key: int = 0
    filter_radius: int = Field(default=3, ge=0)
    resample_sr: int = Field(default=48_000, ge=0)
    rms_mix_rate: float = Field(default=0.25, ge=0.0, le=1.0)
    protect: float = Field(default=0.33, ge=0.0, le=0.5)

    @field_validator("name")
    @classmethod
    def _profile_name_safe(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("RVC profile name must not be empty")
        if any(char in normalized for char in ("/", "\\", "\x00")):
            raise ValueError("RVC profile name must be a file-safe name")
        return normalized


def default_rvc_profiles() -> list[RVCProfile]:
    return [
        RVCProfile(
            name="rmvpe_index045",
            f0_method="rmvpe",
            index_rate=0.45,
            rms_mix_rate=0.25,
            protect=0.33,
        ),
        RVCProfile(
            name="rmvpe_index035_safer",
            f0_method="rmvpe",
            index_rate=0.35,
            rms_mix_rate=0.20,
            protect=0.33,
        ),
        RVCProfile(
            name="rmvpe_index055_stronger_timbre",
            f0_method="rmvpe",
            index_rate=0.55,
            rms_mix_rate=0.25,
            protect=0.33,
        ),
        RVCProfile(
            name="crepe_index045_whisper_candidate",
            f0_method="crepe",
            index_rate=0.45,
            rms_mix_rate=0.20,
            protect=0.33,
        ),
    ]


class RVCSpeakerConfig(StrictBaseModel):
    model_path: str
    index_path: str | None = None
    f0_method: str | None = None
    index_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    f0_up_key: int | None = None
    filter_radius: int | None = Field(default=None, ge=0)
    resample_sr: int | None = Field(default=None, ge=0)
    rms_mix_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    protect: float | None = Field(default=None, ge=0.0, le=0.5)

    @field_validator("model_path")
    @classmethod
    def _model_path_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("RVC speaker model_path must not be empty")
        return value


class GSVSpeakerConfig(StrictBaseModel):
    gpt_weights_path: str | None = None
    sovits_weights_path: str
    refs_path: str
    default_ref_style: str = "whisper_close"

    @field_validator("gpt_weights_path", "sovits_weights_path", "refs_path")
    @classmethod
    def _path_not_empty(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value.strip():
            raise ValueError("GSV speaker paths must not be empty")
        return value

    @field_validator("default_ref_style")
    @classmethod
    def _ref_style_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("GSV default_ref_style must not be empty")
        return value.strip()


class VoiceBankSourceSegment(StrictBaseModel):
    source_id: str
    source_path: str
    local_speaker_label: str
    segment_id: str
    speaker_id: str
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    duration: float = Field(gt=0)
    audio_path: str
    text: str | None = None
    language: str | None = None
    quality_score: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("source_id", "local_speaker_label", "segment_id", "speaker_id")
    @classmethod
    def _ids_are_safe(cls, value: str, info) -> str:
        return validate_file_safe_id(value, info.field_name)

    @model_validator(mode="after")
    def _validate_times(self) -> VoiceBankSourceSegment:
        if self.end <= self.start:
            raise ValueError("voice bank source segment end must be greater than start")
        if abs((self.end - self.start) - self.duration) > 0.01:
            raise ValueError("voice bank source segment duration must match end-start")
        return self


class VoiceBankSpeaker(StrictBaseModel):
    speaker_id: str
    display_name: str | None = None
    source_segments: list[VoiceBankSourceSegment] = Field(default_factory=list)
    embedding_centroid_path: str | None = None
    gsv: GSVSpeakerConfig
    rvc: RVCSpeakerConfig
    dataset_fingerprint: str
    rights_audit: dict[str, Any] = Field(default_factory=dict)
    version: str = "v001"

    @field_validator("speaker_id")
    @classmethod
    def _speaker_id_safe(cls, value: str) -> str:
        return validate_file_safe_id(value, "speaker_id")


class VoiceBankManifest(StrictBaseModel):
    schema_version: str = "voice-bank-1.0"
    speakers: dict[str, VoiceBankSpeaker] = Field(default_factory=dict)
    source_paths: list[str] = Field(default_factory=list)
    backend: str = "mock"
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    rights_audit: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _speaker_keys_match(self) -> VoiceBankManifest:
        for key, speaker in self.speakers.items():
            if validate_file_safe_id(key, "speaker_id") != speaker.speaker_id:
                raise ValueError("voice bank speaker key must match speaker_id")
        return self

    def mark_updated(self) -> None:
        self.updated_at = utc_now()


BUILTIN_ASR_CORRECTION_PROFILE = "builtin:asmr_ja"


class ASRAutoResolutionRule(StrictBaseModel):
    id: str
    source: str
    target: str
    required_any: list[str] = Field(default_factory=list)
    required_all: list[str] = Field(default_factory=list)
    negative_any: list[str] = Field(default_factory=list)
    action: Literal["auto_replace"] = "auto_replace"

    @field_validator("id")
    @classmethod
    def _validate_rule_id(cls, value: str) -> str:
        return validate_file_safe_id(value, "asr_review_auto_resolution_rule.id")

    @field_validator("source", "target")
    @classmethod
    def _validate_non_empty_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("ASR auto-resolution source and target must not be empty")
        return stripped


class ASRCorrectionProfile(StrictBaseModel):
    initial_prompt: str = ""
    review_initial_prompt: str = ""
    qwen_context: str = ""
    hotwords: str = ""
    repair_suspicious_text_patterns: list[str] = Field(default_factory=list)
    text_replacements: dict[str, str] = Field(default_factory=dict)
    review_suspicious_text_patterns: list[str] = Field(default_factory=list)
    review_candidate_replacements: dict[str, str] = Field(default_factory=dict)
    review_auto_resolution_rules: list[ASRAutoResolutionRule] = Field(default_factory=list)
    translation_backcheck_source_patterns: list[str] = Field(default_factory=list)
    translation_backcheck_ko_patterns: list[str] = Field(default_factory=list)


def _read_asr_profile_payload(profile_path: str | Path | None, base_dir: Path | None = None) -> dict[str, Any]:
    raw_profile = str(profile_path or BUILTIN_ASR_CORRECTION_PROFILE).strip()
    if not raw_profile or raw_profile == BUILTIN_ASR_CORRECTION_PROFILE:
        raw_profile = BUILTIN_ASR_CORRECTION_PROFILE
    if raw_profile.startswith("builtin:"):
        profile_name = raw_profile.removeprefix("builtin:").strip() or "asmr_ja"
        resource = resources.files("asmr_dub_pipeline.config_profiles.asr").joinpath(
            f"{profile_name}.yaml"
        )
        payload = yaml.safe_load(resource.read_text(encoding="utf-8")) or {}
    else:
        path = Path(raw_profile).expanduser()
        if not path.is_absolute() and base_dir is not None:
            path = base_dir / path
        payload = yaml.safe_load(path.read_text("utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"ASR correction profile must be a mapping: {raw_profile}")
    return payload


def load_asr_correction_profile(
    profile_path: str | Path | None = BUILTIN_ASR_CORRECTION_PROFILE,
    base_dir: Path | None = None,
) -> ASRCorrectionProfile:
    return ASRCorrectionProfile.model_validate(_read_asr_profile_payload(profile_path, base_dir))


def default_asr_correction_profile() -> ASRCorrectionProfile:
    return load_asr_correction_profile(BUILTIN_ASR_CORRECTION_PROFILE)


class ASRConfig(StrictBaseModel):
    backend: Literal["mock", "faster_whisper", "qwen_asr"] = "faster_whisper"
    preset: ASRPreset = "default"
    model_id: str = "Systran/faster-whisper-large-v3"
    language: str = "ja"
    local_files_only: bool = True
    device: str = "cuda"
    compute_type: str = "default"
    batched_inference: bool = False
    batch_size: int = Field(default=8, ge=1)
    beam_size: int = Field(default=5, ge=1)
    best_of: int = Field(default=5, ge=1)
    condition_on_previous_text: bool = False
    vad_filter: bool = True
    vad_parameters: dict[str, Any] = Field(
        default_factory=lambda: {"min_silence_duration_ms": 300, "speech_pad_ms": 200}
    )
    word_timestamps: bool = False
    hallucination_silence_threshold: float | None = Field(default=None, gt=0)
    initial_prompt: str = ""
    diagnostics_enabled: bool = True
    input_min_rms_dbfs: float = -75.0
    input_min_peak_dbfs: float = -65.0
    input_duration_tolerance: float = Field(default=0.08, ge=0.0, le=1.0)
    qwen_model_id: str = "Qwen/Qwen3-ASR-1.7B"
    qwen_forced_aligner_model_id: str | None = "Qwen/Qwen3-ForcedAligner-0.6B"
    qwen_device_map: str = "cuda:0"
    qwen_dtype: str = "bfloat16"
    qwen_return_timestamps: bool = True
    qwen_context: str = ""
    qwen_max_inference_batch_size: int = Field(default=8, ge=1)
    qwen_max_new_tokens: int = Field(default=4096, ge=1)
    resegment_from_chunks: bool = True
    resegment_min_sec: float = Field(default=3.0, gt=0)
    resegment_max_sec: float = Field(default=10.0, gt=0)
    resegment_merge_gap_sec: float = Field(default=1.0, ge=0)
    countdown_merge_enabled: bool = True
    countdown_merge_gap_sec: float = Field(default=2.5, ge=0)
    countdown_merge_max_span_sec: float = Field(default=60.0, gt=0)
    sparse_chunk_max_sec: float = Field(default=30.0, gt=0)
    sparse_chunk_min_chars_per_sec: float = Field(default=0.5, ge=0)
    repair_enabled: bool = True
    repair_confidence_threshold: float = Field(default=0.90, ge=0.0, le=1.0)
    repair_sparse_min_sec: float = Field(default=12.0, gt=0)
    repair_sparse_min_chars_per_sec: float = Field(default=1.0, ge=0)
    repair_padding_sec: float = Field(default=1.0, ge=0)
    repair_max_chunks: int = Field(default=160, ge=0)
    qwen_repair_fallback_enabled: bool = False
    repair_rejected_short_fragment_auto_accept: bool = True
    repair_rejected_short_fragment_max_sec: float = Field(default=0.8, gt=0)
    repair_rejected_short_fragment_min_confidence: float = Field(default=0.85, ge=0.0, le=1.0)
    review_enabled: bool = False
    review_backend: Literal["llama_server_audio", "mock"] = "llama_server_audio"
    review_batch_size: int = Field(default=8, ge=1, le=50)
    review_max_chunks: int = Field(default=160, ge=0)
    review_context_radius: int = Field(default=2, ge=0, le=8)
    review_confidence_threshold: float = Field(default=0.78, ge=0.0, le=1.0)
    review_generate_candidates: bool = True
    review_candidate_padding_sec: list[float] = Field(default_factory=lambda: [0.4, 1.2])
    review_audio_padding_sec: float = Field(default=0.4, ge=0.0)
    review_initial_prompt: str = ""
    channel_split_enabled: bool = True
    channel_split_padding_sec: float = Field(default=1.4, ge=0.0)
    channel_split_wide_padding_sec: float = Field(default=3.0, ge=0.0)
    channel_split_max_segments: int = Field(default=80, ge=0)
    channel_split_no_dialog_texture_max_sec: float = Field(default=1.5, gt=0.0)
    translation_backcheck_enabled: bool = True
    translation_backcheck_mark_manual_review: bool = False
    source_separation_backend: Literal["auto", "none", "demucs", "mock"] = "auto"
    source_separation_model: str = "htdemucs"
    source_separation_device: str | None = None
    segmentation_min_segment_sec: float = Field(default=0.25, gt=0)
    segmentation_max_segment_sec: float = Field(default=20.0, gt=0)
    segmentation_silence_db: float = -45.0
    segmentation_min_silence_sec: float = Field(default=0.30, ge=0)
    correction_profile_path: str | None = BUILTIN_ASR_CORRECTION_PROFILE
    correction_profile: ASRCorrectionProfile = Field(
        default_factory=default_asr_correction_profile,
        exclude=True,
        repr=False,
    )

    @model_validator(mode="before")
    @classmethod
    def _merge_partial_correction_profile(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        profile = data.get("correction_profile")
        if isinstance(profile, dict):
            merged = default_asr_correction_profile().model_dump(mode="json")
            merged.update(profile)
            data["correction_profile"] = merged
        return data


class GemmaConfig(StrictBaseModel):
    sample_rate: int = 16_000
    default_backend: Literal["mock", "hf", "http", "llama_cpp"] = "mock"
    model_id: str = "google/gemma-4-E4B-it"
    http_url: str | None = None
    http_send_audio: bool = False
    llama_cpp_cli_path: str = ".cache/llama_cpp/src/llama.cpp/build/bin/llama-mtmd-cli"
    llama_cpp_model_path: str = (
        "/home/junyeop/projects/ASMR/.cache/llama_cpp/models/mudler/"
        "gemma-4-26B-A4B-it-heretic-APEX-GGUF/"
        "gemma-4-26B-A4B-heretic-APEX-I-Mini.gguf"
    )
    llama_cpp_mmproj_path: str = (
        "~/.cache/huggingface/hub/models--OBLITERATUS--gemma-4-E4B-it-OBLITERATED/"
        "snapshots/d8678bbb9e0d4f5729c115087485a4e25ba89d65/"
        "gemma-4-E4B-it-OBLITERATED-mmproj-f16.gguf"
    )
    llama_cpp_audio_model_path: str = (
        "~/.cache/huggingface/hub/models--OBLITERATUS--gemma-4-E4B-it-OBLITERATED/"
        "snapshots/d8678bbb9e0d4f5729c115087485a4e25ba89d65/"
        "gemma-4-E4B-it-OBLITERATED-Q8_0.gguf"
    )
    llama_cpp_audio_mmproj_path: str = (
        "~/.cache/huggingface/hub/models--OBLITERATUS--gemma-4-E4B-it-OBLITERATED/"
        "snapshots/d8678bbb9e0d4f5729c115087485a4e25ba89d65/"
        "gemma-4-E4B-it-OBLITERATED-mmproj-f16.gguf"
    )
    llama_cpp_timeout_sec: float = Field(default=600.0, gt=0)
    llama_cpp_ctx_size: int = Field(default=16384, ge=512)
    llama_cpp_n_predict: int = Field(default=1024, ge=64)
    llama_cpp_gpu_layers: int = Field(default=999, ge=0)
    llama_cpp_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llama_cpp_seed: int = 12345
    llama_cpp_extra_args: list[str] = Field(default_factory=list)
    text_server_url: str = "http://127.0.0.1:8080"
    text_server_auto_start: bool = True
    text_server_command: list[str] = Field(default_factory=list)
    text_batch_size: int = Field(default=12, ge=1, le=200)
    text_span_size: int = Field(default=10, ge=1, le=20)
    text_span_max_sec: float = Field(default=75.0, gt=0)
    text_span_max_gap_sec: float = Field(default=5.0, ge=0)
    text_context_radius: int = Field(default=10, ge=0, le=20)
    text_two_pass: bool = True
    text_auto_salvage_enabled: bool = True
    text_repetition_omission_policy: Literal["warn", "manual_review"] = "warn"
    text_concurrency: int = Field(default=1, ge=1, le=8)
    audio_style_concurrency: int = Field(default=2, ge=1, le=8)
    audio_style_scope: Literal["all", "speaker_suspicious"] = "all"
    text_n_predict: int = Field(default=1536, ge=64)
    text_timeout_sec: float = Field(default=180.0, gt=0)
    text_retries: int = Field(default=1, ge=0, le=5)
    text_server_startup_timeout_sec: float = Field(default=120.0, gt=0)
    text_server_shutdown_timeout_sec: float = Field(default=10.0, gt=0)

    @field_validator("audio_style_scope", mode="before")
    @classmethod
    def _normalize_audio_style_scope(cls, value: Any) -> str:
        normalized = str(value or "all").strip().lower().replace("-", "_").replace(" ", "_")
        if normalized in {"speaker", "speaker_suspect"}:
            return "speaker_suspicious"
        return normalized


class TTSConfig(StrictBaseModel):
    qwen_model_id: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    qwen_candidate_count: int = Field(default=4, ge=1, le=8)
    qwen_device_map: str = "cuda:0"
    qwen_dtype: str = "bfloat16"
    qwen_attn_implementation: str = "flash_attention_2"
    qwen_local_files_only: bool = True
    qwen_candidate_batch_size: int = Field(default=4, ge=1, le=8)
    qwen_segment_batch_size: int = Field(default=8, ge=1, le=16)
    qwen_target_vram_gb: float | None = Field(default=14.0, gt=0)
    qwen_temperature: float = Field(default=0.65, ge=0.0, le=2.0)
    qwen_top_p: float = Field(default=0.85, gt=0.0, le=1.0)
    qwen_max_new_tokens: int = Field(default=2048, ge=1)
    qwen_x_vector_only_mode: bool = False
    fish_repo_dir: str = ".cache/tts_backends/fish-speech"
    fish_base_url: str = "http://127.0.0.1:8080"
    fish_candidate_count: int = Field(default=2, ge=1, le=8)
    fish_timeout_sec: float = Field(default=240.0, gt=0)
    fish_chunk_length: int = Field(default=200, ge=100, le=1000)
    fish_temperature: float = Field(default=0.8, ge=0.1, le=1.0)
    fish_top_p: float = Field(default=0.8, ge=0.1, le=1.0)
    fish_repetition_penalty: float = Field(default=1.1, ge=0.9, le=2.0)
    fish_max_new_tokens: int = Field(default=1024, ge=1)
    fish_normalize: bool = True
    fish_latency: Literal["normal", "balanced"] = "normal"
    cosyvoice_repo_dir: str = ".cache/tts_backends/CosyVoice"
    cosyvoice_model_dir: str = ".cache/tts_backends/CosyVoice/pretrained_models/CosyVoice2-0.5B"
    cosyvoice_base_url: str = "http://127.0.0.1:50000"
    cosyvoice_candidate_count: int = Field(default=2, ge=1, le=8)
    cosyvoice_timeout_sec: float = Field(default=240.0, gt=0)
    cosyvoice_mode: Literal["zero_shot", "cross_lingual", "instruct2"] = "zero_shot"
    cosyvoice_sample_rate: int = Field(default=22_050, ge=8_000)
    cosyvoice_instruct_text: str = ""


class GSVConfig(StrictBaseModel):
    url: str = "http://127.0.0.1:9880"
    gpt_weights_path: str | None = None
    sovits_weights_path: str | None = None
    gpt_weights_policy: GSVGPTWeightsPolicy = "auto"
    sovits_weights_policy: GSVSoVITSWeightsPolicy = "auto"
    speaker_models: dict[str, GSVSpeakerConfig] = Field(default_factory=dict)
    timeout_sec: float = Field(default=120.0, gt=0)
    retries: int = Field(default=2, ge=0)
    concurrency: int = Field(default=4, ge=1, le=8)
    auto_start: bool = False
    server_command: list[str] = Field(default_factory=list)
    server_cwd: str | None = None
    server_startup_timeout_sec: float = Field(default=120.0, gt=0)
    server_shutdown_timeout_sec: float = Field(default=10.0, gt=0)
    trim_edge_silence: bool = True
    trim_silence_threshold_db: float = -50.0
    trim_silence_keep_sec: float = Field(default=0.08, ge=0.0, le=1.0)
    few_shot_enabled: bool = True
    few_shot_min_total_sec: float | None = Field(default=None, gt=0)
    few_shot_min_clip_sec: float = Field(default=1.0, gt=0)
    few_shot_max_clip_sec: float = Field(default=10.0, gt=0)
    few_shot_min_quality_score: float = Field(default=0.20, ge=0.0, le=1.0)
    few_shot_clean_source_filter: bool = True
    few_shot_max_background_bleed_db: float = Field(default=-24.0, ge=-80.0, le=0.0)
    few_shot_max_side_to_mid_db: float = Field(default=-8.0, ge=-80.0, le=20.0)
    few_shot_preferred_chars_per_sec: float | None = Field(default=4.5, gt=0)
    few_shot_max_chars_per_sec: float | None = Field(default=5.2, gt=0)
    few_shot_min_selection_score: float | None = Field(default=None, ge=0.0, le=1.0)
    few_shot_max_total_sec: float | None = Field(default=None, gt=0)
    few_shot_asr_risk_filter: bool = True
    few_shot_prefer_plain_text: bool = True
    few_shot_pacing_target_enabled: bool = True
    few_shot_pacing_target_tolerance: float = Field(default=0.10, ge=0.0, le=2.0)
    few_shot_pacing_max_target_tolerance: float = Field(default=0.30, ge=0.0, le=2.0)
    few_shot_pacing_variance_weight: float = Field(default=1.0, ge=0.0, le=10.0)
    few_shot_pacing_quality_weight: float = Field(default=0.25, ge=0.0, le=10.0)
    few_shot_pacing_beam_size: int = Field(default=768, ge=32, le=4096)
    few_shot_pacing_search_iterations: int = Field(default=12_000, ge=0, le=100_000)
    few_shot_pacing_max_duration_overage_ratio: float = Field(default=0.10, ge=0.0, le=1.0)
    few_shot_relax_clean_source_for_pacing: bool = True
    few_shot_relaxed_clean_source_penalty: float = Field(default=0.15, ge=0.0, le=1.0)
    few_shot_insufficient_policy: Literal["error", "zero_shot"] = "zero_shot"
    ref_mode: GSVRefMode = "static"
    ref_min_sec: float = Field(default=3.0, gt=0)
    ref_max_sec: float = Field(default=10.0, gt=0)
    ref_min_quality_score: float = Field(default=0.25, ge=0.0, le=1.0)
    korean_segment_ref_enabled: bool = True
    segment_ref_min_overlap_ratio: float = Field(default=0.75, ge=0.0, le=1.0)
    segment_ref_relaxed_training_reasons: list[str] = Field(
        default_factory=lambda: [
            "low_dominant_source_speaker_overlap",
            "borderline_single_speaker_overlap",
            "neighbor_confirmed_single_speaker_overlap",
            "single_speaker_overlap_tts_routing",
            "merged_overlap_candidates_tts_routing",
        ]
    )
    korean_segment_ref_clarity_profile_enabled: bool = True
    korean_segment_ref_warn_blocks_mix: bool = True
    korean_segment_ref_temperature: float = Field(default=0.75, ge=0.0, le=2.0)
    korean_segment_ref_top_p: float = Field(default=0.85, gt=0.0, le=1.0)
    korean_segment_ref_top_k: int = Field(default=8, ge=1)
    korean_segment_ref_parallel_infer: bool = False
    ko_text_min_hangul_ratio: float = Field(default=0.20, ge=0.0, le=1.0)
    top_k: int = Field(default=15, ge=1)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    text_split_method: str = Field(default="cut5", min_length=1)
    parallel_infer: bool = True
    repetition_penalty: float = Field(default=1.35, ge=0.0, le=5.0)
    sample_steps: int = Field(default=32, ge=1)
    super_sampling: bool = False
    overlap_length: int = Field(default=2, ge=0)
    min_chunk_length: int = Field(default=16, ge=1)
    fragment_interval: float = Field(default=0.3, ge=0.0)
    tts_min_speed_factor: float = Field(default=0.85, gt=0.0, le=1.0)
    tts_max_speed_factor: float = Field(default=1.12, ge=1.0, le=1.35)
    countdown_renderer: Literal[
        "carrier_bank",
        "chunk_bank",
        "canonical_pack",
        "token",
        "compact",
        "numeric_phrase",
    ] = "numeric_phrase"
    numeric_phrase_renderer_enabled: bool = True
    numeric_phrase_max_tempo: float = Field(default=1.1, ge=1.0, le=2.0)
    numeric_phrase_failure_fallback: Literal["manual_review", "normal_tts"] = "manual_review"
    numeric_phrase_whole_lead_in_sec: float = Field(default=0.12, ge=0.0, le=1.0)
    numeric_phrase_tail_guard_sec: float = Field(default=0.16, ge=0.0, le=1.0)
    countdown_fallback_renderer: Literal["compact", "phrase_slice", "manual_review", "none"] = "manual_review"
    countdown_candidate_count: int = Field(default=8, ge=1, le=24)
    countdown_carrier_templates: list[str] = Field(
        default_factory=lambda: [
            "이번 숫자는 {token}. 입니다.",
            "숫자만 조용히 말해요. {token}. 다시.",
        ],
        min_length=0,
        max_length=12,
    )
    countdown_carrier_token_templates: dict[str, list[str]] = Field(
        default_factory=lambda: {
            token: list(templates)
            for token, templates in DEFAULT_COUNTDOWN_CARRIER_TOKEN_TEMPLATES.items()
        }
    )
    countdown_carrier_numeric_unit_enabled: bool = True
    countdown_carrier_numeric_unit_templates: list[str] = Field(
        default_factory=lambda: [
            "{token} 번만요.",
            "{token} 입니다.",
            "{token} 하고요.",
            "{token} 초만요.",
        ],
        min_length=1,
        max_length=12,
    )
    countdown_carrier_numeric_unit_onset_window_sec: list[float] = Field(
        default_factory=lambda: [0.18, 0.24, 0.30, 0.36],
        min_length=1,
        max_length=8,
    )
    countdown_carrier_numeric_unit_tail_pad_sec: float = Field(default=0.04, ge=0.0, le=0.2)
    countdown_carrier_slice_start_backoff_sec: float = Field(default=0.02, ge=0.0, le=0.08)
    countdown_carrier_energy_extend_enabled: bool = True
    countdown_carrier_energy_extend_max_sec: float = Field(default=0.10, ge=0.0, le=0.4)
    countdown_carrier_energy_extend_coda_max_sec: float = Field(default=0.20, ge=0.0, le=0.5)
    countdown_carrier_energy_extend_edge_threshold_ratio: float = Field(default=0.12, ge=0.01, le=0.8)
    countdown_carrier_energy_extend_quiet_threshold_ratio: float = Field(default=0.08, ge=0.005, le=0.5)
    countdown_carrier_bulk_asr_enabled: bool = True
    countdown_carrier_bulk_asr_workers: int = Field(default=3, ge=1, le=8)
    countdown_carrier_pause_gsv_for_bulk_asr: bool = False
    countdown_carrier_candidate_count: int = Field(default=2, ge=1, le=8)
    countdown_carrier_slice_search_enabled: bool = True
    countdown_carrier_slice_window_sec: list[float] = Field(
        default_factory=lambda: [0.30, 0.42, 0.55],
        min_length=1,
        max_length=8,
    )
    countdown_carrier_slice_window_offset_sec: list[float] = Field(
        default_factory=lambda: [-0.06, 0.0, 0.06],
        min_length=1,
        max_length=9,
    )
    countdown_carrier_max_slice_windows_per_candidate: int = Field(default=5, ge=1, le=32)
    countdown_carrier_full_sentence_prefilter_enabled: bool = True
    countdown_carrier_full_sentence_prefilter_min_coverage: float = Field(default=1.0, ge=0.0, le=1.0)
    countdown_carrier_quality_retry_enabled: bool = True
    countdown_carrier_quality_retry_max_rounds: int = Field(default=3, ge=1, le=12)
    countdown_carrier_quality_retry_target_tier: Literal["A", "B", "C"] = "A"
    countdown_token_bank_enabled: bool = True
    countdown_token_bank_warmup_enabled: bool = True
    countdown_token_bank_pack_warmup_enabled: bool = True
    countdown_token_bank_pack_templates: list[str] = Field(
        default_factory=lambda: [
            "{token}, {token}, {token}",
            "{token}. {token}. {token}.",
            "{token}... {token}... {token}",
        ],
        min_length=1,
        max_length=8,
    )
    countdown_token_bank_max_ref_count: int = Field(default=4, ge=1, le=16)
    countdown_token_bank_beam_width: int = Field(default=8, ge=1, le=64)
    countdown_carrier_stop_window_search_after_pronunciation_pass: bool = True
    countdown_carrier_target_pronunciation_passes: int = Field(default=2, ge=0, le=32)
    countdown_chunk_candidate_count: int = Field(default=10, ge=1, le=24)
    countdown_chunk_max_size: int = Field(default=10, ge=1, le=10)
    countdown_temperature: float = Field(default=0.55, ge=0.0, le=2.0)
    countdown_timing_mode: Literal["source_smoothed", "even_grid", "source_exact"] = "source_smoothed"
    countdown_source_anchor_enabled: bool = True
    countdown_source_anchor_asr_retry_enabled: bool = True
    countdown_source_anchor_smoothing_blend: float = Field(default=0.70, ge=0.0, le=1.0)
    countdown_source_anchor_cluster_gap_sec: float = Field(default=2.6, ge=0.0, le=30.0)
    countdown_hybrid_enabled: bool = True
    countdown_hybrid_apply_to_synth: bool = True
    countdown_hybrid_token_gap_fill_ratio: float = Field(default=0.72, gt=0.0, le=2.0)
    countdown_hybrid_token_max_sec: float = Field(default=0.58, gt=0.0, le=5.0)
    countdown_hybrid_active_trim_enabled: bool = True
    countdown_hybrid_active_trim_keep_before_sec: float = Field(default=0.012, ge=0.0, le=1.0)
    countdown_hybrid_active_trim_keep_after_sec: float = Field(default=0.045, ge=0.0, le=1.0)
    countdown_hybrid_base_gain: float = Field(default=0.55, ge=0.0, le=2.0)
    countdown_hybrid_bed_gain: float = Field(default=0.85, ge=0.0, le=2.0)
    countdown_hybrid_base_duck_enabled: bool = True
    countdown_hybrid_base_duck_gain: float = Field(default=0.18, ge=0.0, le=1.0)
    countdown_hybrid_base_duck_pad_sec: float = Field(default=0.055, ge=0.0, le=1.0)
    countdown_hybrid_base_duck_fade_sec: float = Field(default=0.025, ge=0.0, le=1.0)
    countdown_strict_token_pronunciation: bool = True
    countdown_pack_min_span_occupancy: float = Field(default=0.55, ge=0.0, le=1.0)
    countdown_pack_retime_enabled: bool = True
    countdown_pack_retime_trigger_max_gap_sec: float = Field(default=0.55, ge=0.0, le=5.0)
    countdown_pack_retime_trigger_gap_cv: float = Field(default=0.35, ge=0.0, le=5.0)
    countdown_pack_retime_target_min_gap_sec: float = Field(default=0.14, ge=0.0, le=1.0)
    countdown_pack_retime_target_max_gap_sec: float = Field(default=0.28, ge=0.0, le=1.0)
    countdown_pack_retime_leading_sec: float = Field(default=0.08, ge=0.0, le=1.0)
    countdown_pack_retime_trailing_sec: float = Field(default=0.12, ge=0.0, le=1.0)
    countdown_phrase_slice_edge_pad_sec: float = Field(default=0.04, ge=0.0, le=0.5)
    countdown_token_speed_factor: float = Field(default=1.0, ge=0.5, le=1.35)
    countdown_token_min_sec: float = Field(default=0.25, ge=0.0, le=5.0)
    countdown_token_max_sec: float = Field(default=0.95, gt=0.0, le=10.0)
    countdown_token_single_syllable_max_sec: float = Field(default=0.55, gt=0.0, le=10.0)
    countdown_token_double_syllable_max_sec: float = Field(default=0.75, gt=0.0, le=10.0)
    countdown_token_multi_syllable_max_sec: float = Field(default=0.95, gt=0.0, le=10.0)
    countdown_token_max_slot_occupancy: float = Field(default=0.85, gt=0.0, le=2.0)
    countdown_max_tempo: float = Field(default=1.0, ge=1.0, le=8.0)
    countdown_prosody_qc_enabled: bool = True
    countdown_prosody_max_median_semitone_error: float = Field(default=12.0, gt=0.0, le=48.0)
    countdown_prosody_min_pass_score: float = Field(default=0.74, ge=0.0, le=1.0)
    countdown_prosody_min_warn_score: float = Field(default=0.45, ge=0.0, le=1.0)
    countdown_prosody_failure_blocks_mix: bool = True
    max_attempts_per_candidate: int = Field(default=3, ge=1, le=8)
    initial_candidate_count: int | None = Field(default=None, ge=1, le=8)
    retry_candidate_count: int | None = Field(default=7, ge=1, le=8)
    low_temperature_retry_enabled: bool = True
    low_temperature_retry_candidate_count: int | None = Field(default=20, ge=1, le=20)
    low_temperature_retry_temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    pronunciation_qc_enabled: bool = True
    pronunciation_qc_backend: Literal["auto", "mock", "faster_whisper", "qwen_asr"] = "auto"
    pronunciation_qc_workers: int = Field(default=1, ge=1, le=4)
    pronunciation_qc_pass_coverage: float = Field(default=0.82, ge=0.0, le=1.0)
    pronunciation_qc_warn_coverage: float = Field(default=0.62, ge=0.0, le=1.0)
    pronunciation_qc_max_observed_unit_ratio: float | None = Field(default=1.8, gt=0.0)
    pronunciation_qc_max_extra_units: int | None = Field(default=1, ge=0)
    pronunciation_qc_failure_blocks_mix: bool = True
    numeric_cadence_periods_enabled: bool = True
    numeric_cadence_min_values: int = Field(default=3, ge=3, le=20)
    numeric_sequence_qc_enabled: bool = True
    numeric_sequence_qc_require_contiguous: bool = True
    numeric_sequence_qc_failure_blocks_mix: bool = True
    korean_clarity_retry_enabled: bool = True
    korean_clarity_retry_candidate_count: int | None = Field(default=None, ge=1, le=20)
    korean_clarity_temperature: float = Field(default=0.65, ge=0.0, le=2.0)
    korean_clarity_top_p: float = Field(default=0.75, gt=0.0, le=1.0)
    korean_clarity_top_k: int = Field(default=5, ge=1)
    korean_clarity_parallel_infer: bool = False
    zero_shot_candidate_count: int | None = Field(default=None, ge=1, le=8)
    duration_rewrite_backend: GSVDurationRewriteBackend = "none"
    duration_rewrite_timing: GSVDurationRewriteTiming = "after_initial"
    duration_rewrite_max_attempts: int = Field(default=1, ge=0, le=5)
    duration_rewrite_pre_candidate_count: int | None = Field(default=None, ge=1, le=24)
    timing_expansion_enabled: bool = True
    timing_expansion_min_ratio: float = Field(default=0.70, gt=0.0, lt=1.0)
    timing_expansion_max_attempts: int = Field(default=1, ge=0, le=3)
    terminal_failure_policy: GSVTerminalFailurePolicy = "keep_original"
    timing_quality_tolerance: float = Field(default=0.10, gt=0.0, lt=1.0)
    timefit_max_tempo: float = Field(default=1.18, ge=1.0, le=8.0)
    timefit_max_stretch: float = Field(default=1.08, ge=1.0, le=2.0)
    timefit_micro_max_sec: float = Field(default=2.0, gt=0.0)
    timefit_micro_max_tempo: float = Field(default=1.30, ge=1.0, le=8.0)
    timefit_long_min_sec: float = Field(default=7.0, gt=0.0)
    timefit_long_max_stretch: float = Field(default=1.15, ge=1.0, le=2.0)
    rescue_duration_tolerance: float | None = Field(default=0.35, gt=0.0, lt=1.0)
    rescue_timefit_top_k: int = Field(default=3, ge=0, le=8)
    rescue_timefit_max_tempo: float = Field(default=1.45, ge=1.0, le=8.0)
    rescue_timefit_max_stretch: float = Field(default=1.25, ge=1.0, le=2.0)
    rescue_micro_segment_max_sec: float = Field(default=0.6, ge=0.0)
    micro_segment_unfit_policy: Literal["keep_original", "manual_review"] = "keep_original"
    few_shot_force: bool = False
    few_shot_version: Literal["auto", "v1", "v2", "v3", "v4", "v2Pro", "v2ProPlus"] = "auto"

    @model_validator(mode="after")
    def _validate_pronunciation_qc_thresholds(self) -> GSVConfig:
        if self.pronunciation_qc_pass_coverage < self.pronunciation_qc_warn_coverage:
            raise ValueError("pronunciation_qc_pass_coverage must be >= pronunciation_qc_warn_coverage")
        if self.countdown_pack_retime_target_max_gap_sec < self.countdown_pack_retime_target_min_gap_sec:
            raise ValueError(
                "countdown_pack_retime_target_max_gap_sec must be >= "
                "countdown_pack_retime_target_min_gap_sec"
            )
        if (
            "countdown_carrier_templates" in self.model_fields_set
            and "countdown_carrier_token_templates" not in self.model_fields_set
        ):
            self.countdown_carrier_token_templates = {}
        return self


class RVCConfig(StrictBaseModel):
    required: bool = True
    backend: Literal["command", "mock"] = "command"
    train_required: bool = True
    train_backend: Literal["command", "mock"] = "command"
    train_command: list[str] = Field(default_factory=list)
    train_working_dir: str | None = None
    train_timeout_sec: float = Field(default=14400.0, gt=0)
    train_experiment_name: str = "asmr-rvc-speaker-1"
    train_sample_rate: int = Field(default=48_000, ge=0)
    train_epochs: int = Field(default=100, ge=1)
    train_epoch_policy: Literal["fixed", "auto"] = "fixed"
    train_quality_preset: Literal["balanced", "strict"] = "balanced"
    train_batch_size: int = Field(default=0, ge=0, le=64)
    train_min_quality_score: float = Field(default=0.20, ge=0.0, le=1.0)
    train_preferred_chars_per_sec: float | None = Field(default=4.5, gt=0)
    train_max_chars_per_sec: float | None = Field(default=5.2, gt=0)
    train_max_clip_sec: float | None = Field(default=None, gt=0.0)
    train_min_snr_db: float | None = Field(default=None, ge=0.0)
    train_max_background_bleed_db: float | None = None
    train_max_side_to_mid_db: float | None = None
    train_target_clean_sec: float | None = Field(default=None, gt=0.0)
    train_auto_epoch_min: int = Field(default=100, ge=1)
    train_auto_epoch_max: int = Field(default=200, ge=1)
    train_min_clean_sec: float = Field(default=600.0, ge=0.0)
    train_min_clean_segments: int = Field(default=1, ge=1)
    train_augment_enabled: bool = False
    train_augment_min_real_sec: float = Field(default=300.0, ge=0.0)
    train_augment_max_multiplier: int = Field(default=3, ge=1, le=8)
    train_preprocess_processes: int = Field(default=0, ge=0)
    train_f0_workers: int = Field(default=0, ge=0)
    train_feature_workers: int = Field(default=0, ge=0)
    train_save_every_epoch: int = Field(default=10, ge=1)
    train_reuse_intermediate_cache: bool = True
    train_output_model_path: str | None = None
    train_output_index_path: str | None = None
    command: list[str] = Field(default_factory=list)
    batch_infer: bool = True
    batch_command: list[str] = Field(default_factory=list)
    batch_size: int = Field(default=200, ge=1, le=1000)
    batch_concurrency: int = Field(default=1, ge=1, le=4)
    working_dir: str | None = None
    timeout_sec: float = Field(default=180.0, gt=0)
    concurrency: int = Field(default=1, ge=1, le=8)
    model_path: str | None = None
    index_path: str | None = None
    device: str = "cuda:0"
    f0_up_key: int = 0
    f0_method: str = "rmvpe"
    index_rate: float = Field(default=0.45, ge=0.0, le=1.0)
    filter_radius: int = Field(default=3, ge=0)
    resample_sr: int = Field(default=48_000, ge=0)
    rms_mix_rate: float = Field(default=0.25, ge=0.0, le=1.0)
    protect: float = Field(default=0.33, ge=0.0, le=0.5)
    failure_policy: Literal["retry_then_error", "error"] = "retry_then_error"
    allow_pre_rvc_fallback: bool = False
    duration_tolerance: float | None = Field(default=None, gt=0, lt=1)
    auto_profiles: list[RVCProfile] = Field(default_factory=default_rvc_profiles)
    speaker_models: dict[str, RVCSpeakerConfig] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_train_auto_epoch_bounds(self) -> RVCConfig:
        if self.train_auto_epoch_max < self.train_auto_epoch_min:
            raise ValueError("rvc_train_auto_epoch_max must be >= rvc_train_auto_epoch_min")
        return self


class MixConfig(StrictBaseModel):
    sample_rate: int = 48_000
    duration_tolerance: float = Field(default=0.20, gt=0, lt=1)
    profile: MixProfile = "asmr_stereo"
    background_bed: MixBackgroundBed = "preserve_original"
    background_gain_db: float = Field(default=-18.0, ge=-60.0, le=6.0)
    background_speech_suppression: bool = True
    background_speech_suppression_db: float = Field(default=-42.0, ge=-80.0, le=0.0)
    background_speech_suppression_pad_sec: float = Field(default=0.06, ge=0.0, le=1.0)
    background_speech_suppression_fade_ms: float = Field(default=30.0, ge=0.0, le=500.0)
    dialogue_gain_db: float = Field(default=0.0, ge=-60.0, le=12.0)
    dialogue_fade_ms: float | None = Field(default=None, ge=0.0, le=250.0)
    loudness_strategy: MixLoudnessStrategy = "peak_guard_only"
    peak_limit_dbfs: float = Field(default=-1.0, ge=-24.0, le=0.0)
    allow_korean_timing_draft: bool = False


class VoiceBankConfig(StrictBaseModel):
    path: str = "voice_bank/voice_bank_manifest.json"
    speaker_assignment_backend: SpeakerAssignmentBackend = "none"
    diarization_model_id: str = "pyannote/speaker-diarization-community-1"
    diarization_embedding_model_id: str = "pyannote/wespeaker-voxceleb-resnet34-LM"
    diarization_auto_download: bool = True
    diarization_min_speakers: int | None = Field(default=None, ge=1)
    diarization_max_speakers: int | None = Field(default=None, ge=1)
    diarization_embedding_match_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    source_speaker_training_min_dominant_overlap_ratio: float = Field(default=0.85, ge=0.0, le=1.0)


class SafetyConfig(StrictBaseModel):
    hf_local_files_only: bool = True


class ProjectConfig(StrictBaseModel):
    project_name: str = "asmr-dub-project"
    source_language: SourceLanguage = "ja"
    target_language: TargetLanguage = "ko"
    candidate_count: int = Field(default=5, ge=1, le=8)
    base_seed: int = 12345
    asr: ASRConfig = Field(default_factory=ASRConfig)
    gemma: GemmaConfig = Field(default_factory=GemmaConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    gsv: GSVConfig = Field(default_factory=GSVConfig)
    rvc: RVCConfig = Field(default_factory=RVCConfig)
    mix: MixConfig = Field(default_factory=MixConfig)
    voice_bank: VoiceBankConfig = Field(default_factory=VoiceBankConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)

    @model_validator(mode="before")
    @classmethod
    def _migrate_flat_config_fields(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        for flat_key, path in _PROJECT_CONFIG_FLAT_ALIASES.items():
            if flat_key in data:
                _assign_nested_config_value(data, path, data.pop(flat_key))
        return data

    @field_validator("source_language", "target_language", mode="before")
    @classmethod
    def _canonicalize_language(cls, value: str) -> str:
        normalized = str(value or "").strip().lower().replace("-", "_")
        if normalized in {"jp", "jpn", "japanese"}:
            return "ja"
        if normalized in {"kr", "kor", "korean"}:
            return "ko"
        return normalized

    @model_validator(mode="after")
    def _validate_duration_contracts(self) -> ProjectConfig:
        if self.gsv.ref_max_sec <= self.gsv.ref_min_sec:
            raise ValueError("gsv_ref_max_sec must be greater than gsv_ref_min_sec")
        if self.gsv.tts_max_speed_factor < self.gsv.tts_min_speed_factor:
            raise ValueError("gsv_tts_max_speed_factor must be >= gsv_tts_min_speed_factor")
        if self.rvc.required and not self.rvc.auto_profiles:
            raise ValueError("rvc_auto_profiles must contain at least one profile when RVC is required")
        if self.rvc.required and not self.rvc.train_required and not self.rvc.speaker_models:
            raise ValueError(
                "rvc_train_required must be true when RVC is required unless rvc_speaker_models are configured"
            )
        profile_names = [profile.name for profile in self.rvc.auto_profiles]
        if len(profile_names) != len(set(profile_names)):
            raise ValueError("rvc_auto_profiles names must be unique")
        if (
            self.voice_bank.diarization_min_speakers is not None
            and self.voice_bank.diarization_max_speakers is not None
            and self.voice_bank.diarization_min_speakers > self.voice_bank.diarization_max_speakers
        ):
            raise ValueError("diarization_min_speakers must be <= diarization_max_speakers")
        return self

    def __getattr__(self, item: str) -> Any:
        if item in _PROJECT_CONFIG_FLAT_ALIASES:
            return _get_nested_config_value(self, _PROJECT_CONFIG_FLAT_ALIASES[item])
        return super().__getattr__(item)

    def __setattr__(self, name: str, value: Any) -> None:
        aliases = globals().get("_PROJECT_CONFIG_FLAT_ALIASES", {})
        if name in aliases:
            _set_nested_config_value(self, aliases[name], value)
            return
        super().__setattr__(name, value)


FlatConfigPath = tuple[str, ...]


def _mapping_from_config_value(value: Any, path: FlatConfigPath) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return dict(value)
    joined = ".".join(path)
    raise ValueError(f"Project config section must be a mapping before assigning {joined}")


def _assign_nested_config_value(data: dict[str, Any], path: FlatConfigPath, value: Any) -> None:
    cursor = data
    for index, part in enumerate(path[:-1]):
        existing = cursor.get(part)
        if existing is None and tuple(path[: index + 1]) == ("asr", "correction_profile"):
            existing = default_asr_correction_profile().model_dump(mode="json")
        section = _mapping_from_config_value(existing, path[: index + 1])
        cursor[part] = section
        cursor = section
    cursor[path[-1]] = value


def _get_nested_config_value(config: ProjectConfig, path: FlatConfigPath) -> Any:
    value: Any = config
    for part in path:
        value = getattr(value, part)
    return value


def _set_nested_config_value(config: ProjectConfig, path: FlatConfigPath, value: Any) -> None:
    target: Any = config
    for part in path[:-1]:
        target = getattr(target, part)
    setattr(target, path[-1], value)


_ASR_DIRECT_FLAT_FIELDS = [
    "backend",
    "preset",
    "model_id",
    "language",
    "local_files_only",
    "device",
    "compute_type",
    "batched_inference",
    "batch_size",
    "beam_size",
    "best_of",
    "condition_on_previous_text",
    "vad_filter",
    "vad_parameters",
    "word_timestamps",
    "hallucination_silence_threshold",
    "initial_prompt",
    "diagnostics_enabled",
    "input_min_rms_dbfs",
    "input_min_peak_dbfs",
    "input_duration_tolerance",
    "resegment_from_chunks",
    "resegment_min_sec",
    "resegment_max_sec",
    "resegment_merge_gap_sec",
    "countdown_merge_enabled",
    "countdown_merge_gap_sec",
    "countdown_merge_max_span_sec",
    "sparse_chunk_max_sec",
    "sparse_chunk_min_chars_per_sec",
    "repair_enabled",
    "repair_confidence_threshold",
    "repair_sparse_min_sec",
    "repair_sparse_min_chars_per_sec",
    "repair_padding_sec",
    "repair_max_chunks",
    "qwen_repair_fallback_enabled",
    "repair_rejected_short_fragment_auto_accept",
    "repair_rejected_short_fragment_max_sec",
    "repair_rejected_short_fragment_min_confidence",
    "review_enabled",
    "review_backend",
    "review_batch_size",
    "review_max_chunks",
    "review_context_radius",
    "review_confidence_threshold",
    "review_generate_candidates",
    "review_candidate_padding_sec",
    "review_audio_padding_sec",
    "review_initial_prompt",
    "channel_split_enabled",
    "channel_split_padding_sec",
    "channel_split_wide_padding_sec",
    "channel_split_max_segments",
    "channel_split_no_dialog_texture_max_sec",
    "translation_backcheck_enabled",
    "translation_backcheck_mark_manual_review",
    "correction_profile_path",
]
_QWEN_ASR_FLAT_FIELDS = [
    "model_id",
    "forced_aligner_model_id",
    "device_map",
    "dtype",
    "return_timestamps",
    "context",
    "max_inference_batch_size",
    "max_new_tokens",
]
_GEMMA_SIMPLE_FLAT_FIELDS = ["model_id", "http_url", "http_send_audio"]
_GEMMA_LLAMA_CPP_FLAT_FIELDS = [
    "cli_path",
    "model_path",
    "mmproj_path",
    "audio_model_path",
    "audio_mmproj_path",
    "timeout_sec",
    "ctx_size",
    "n_predict",
    "gpu_layers",
    "temperature",
    "seed",
    "extra_args",
]
_GEMMA_TEXT_FLAT_FIELDS = [
    "server_url",
    "server_auto_start",
    "server_command",
    "batch_size",
    "span_size",
    "span_max_sec",
    "span_max_gap_sec",
    "context_radius",
    "two_pass",
    "auto_salvage_enabled",
    "repetition_omission_policy",
    "concurrency",
    "n_predict",
    "timeout_sec",
    "retries",
    "server_startup_timeout_sec",
    "server_shutdown_timeout_sec",
]
_GSV_FLAT_FIELDS = [
    "url",
    "gpt_weights_path",
    "sovits_weights_path",
    "gpt_weights_policy",
    "sovits_weights_policy",
    "speaker_models",
    "timeout_sec",
    "retries",
    "concurrency",
    "auto_start",
    "server_command",
    "server_cwd",
    "server_startup_timeout_sec",
    "server_shutdown_timeout_sec",
    "trim_edge_silence",
    "trim_silence_threshold_db",
    "trim_silence_keep_sec",
    "few_shot_enabled",
    "few_shot_min_total_sec",
    "few_shot_min_clip_sec",
    "few_shot_max_clip_sec",
    "few_shot_min_quality_score",
    "few_shot_clean_source_filter",
    "few_shot_max_background_bleed_db",
    "few_shot_max_side_to_mid_db",
    "few_shot_preferred_chars_per_sec",
    "few_shot_max_chars_per_sec",
    "few_shot_min_selection_score",
    "few_shot_max_total_sec",
    "few_shot_asr_risk_filter",
    "few_shot_prefer_plain_text",
    "few_shot_pacing_target_enabled",
    "few_shot_pacing_target_tolerance",
    "few_shot_pacing_max_target_tolerance",
    "few_shot_pacing_variance_weight",
    "few_shot_pacing_quality_weight",
    "few_shot_pacing_beam_size",
    "few_shot_pacing_search_iterations",
    "few_shot_pacing_max_duration_overage_ratio",
    "few_shot_relax_clean_source_for_pacing",
    "few_shot_relaxed_clean_source_penalty",
    "few_shot_insufficient_policy",
    "ref_mode",
    "ref_min_sec",
    "ref_max_sec",
    "ref_min_quality_score",
    "korean_segment_ref_enabled",
    "segment_ref_min_overlap_ratio",
    "segment_ref_relaxed_training_reasons",
    "korean_segment_ref_clarity_profile_enabled",
    "korean_segment_ref_warn_blocks_mix",
    "korean_segment_ref_temperature",
    "korean_segment_ref_top_p",
    "korean_segment_ref_top_k",
    "korean_segment_ref_parallel_infer",
    "ko_text_min_hangul_ratio",
    "top_k",
    "top_p",
    "temperature",
    "text_split_method",
    "parallel_infer",
    "repetition_penalty",
    "sample_steps",
    "super_sampling",
    "overlap_length",
    "min_chunk_length",
    "fragment_interval",
    "tts_min_speed_factor",
    "tts_max_speed_factor",
    "countdown_renderer",
    "numeric_phrase_renderer_enabled",
    "numeric_phrase_max_tempo",
    "numeric_phrase_failure_fallback",
    "numeric_phrase_whole_lead_in_sec",
    "numeric_phrase_tail_guard_sec",
    "countdown_fallback_renderer",
    "countdown_candidate_count",
    "countdown_carrier_templates",
    "countdown_carrier_token_templates",
    "countdown_carrier_numeric_unit_enabled",
    "countdown_carrier_numeric_unit_templates",
    "countdown_carrier_numeric_unit_onset_window_sec",
    "countdown_carrier_numeric_unit_tail_pad_sec",
    "countdown_carrier_slice_start_backoff_sec",
    "countdown_carrier_energy_extend_enabled",
    "countdown_carrier_energy_extend_max_sec",
    "countdown_carrier_energy_extend_coda_max_sec",
    "countdown_carrier_energy_extend_edge_threshold_ratio",
    "countdown_carrier_energy_extend_quiet_threshold_ratio",
    "countdown_carrier_bulk_asr_enabled",
    "countdown_carrier_bulk_asr_workers",
    "countdown_carrier_pause_gsv_for_bulk_asr",
    "countdown_carrier_candidate_count",
    "countdown_carrier_slice_search_enabled",
    "countdown_carrier_slice_window_sec",
    "countdown_carrier_slice_window_offset_sec",
    "countdown_carrier_max_slice_windows_per_candidate",
    "countdown_carrier_full_sentence_prefilter_enabled",
    "countdown_carrier_full_sentence_prefilter_min_coverage",
    "countdown_carrier_quality_retry_enabled",
    "countdown_carrier_quality_retry_max_rounds",
    "countdown_carrier_quality_retry_target_tier",
    "countdown_token_bank_enabled",
    "countdown_token_bank_warmup_enabled",
    "countdown_token_bank_pack_warmup_enabled",
    "countdown_token_bank_pack_templates",
    "countdown_token_bank_max_ref_count",
    "countdown_token_bank_beam_width",
    "countdown_carrier_stop_window_search_after_pronunciation_pass",
    "countdown_carrier_target_pronunciation_passes",
    "countdown_chunk_candidate_count",
    "countdown_chunk_max_size",
    "countdown_temperature",
    "countdown_timing_mode",
    "countdown_source_anchor_enabled",
    "countdown_source_anchor_asr_retry_enabled",
    "countdown_source_anchor_smoothing_blend",
    "countdown_source_anchor_cluster_gap_sec",
    "countdown_hybrid_enabled",
    "countdown_hybrid_apply_to_synth",
    "countdown_hybrid_token_gap_fill_ratio",
    "countdown_hybrid_token_max_sec",
    "countdown_hybrid_active_trim_enabled",
    "countdown_hybrid_active_trim_keep_before_sec",
    "countdown_hybrid_active_trim_keep_after_sec",
    "countdown_hybrid_base_gain",
    "countdown_hybrid_bed_gain",
    "countdown_hybrid_base_duck_enabled",
    "countdown_hybrid_base_duck_gain",
    "countdown_hybrid_base_duck_pad_sec",
    "countdown_hybrid_base_duck_fade_sec",
    "countdown_strict_token_pronunciation",
    "countdown_pack_min_span_occupancy",
    "countdown_pack_retime_enabled",
    "countdown_pack_retime_trigger_max_gap_sec",
    "countdown_pack_retime_trigger_gap_cv",
    "countdown_pack_retime_target_min_gap_sec",
    "countdown_pack_retime_target_max_gap_sec",
    "countdown_pack_retime_leading_sec",
    "countdown_pack_retime_trailing_sec",
    "countdown_phrase_slice_edge_pad_sec",
    "countdown_token_speed_factor",
    "countdown_token_min_sec",
    "countdown_token_max_sec",
    "countdown_token_single_syllable_max_sec",
    "countdown_token_double_syllable_max_sec",
    "countdown_token_multi_syllable_max_sec",
    "countdown_token_max_slot_occupancy",
    "countdown_max_tempo",
    "countdown_prosody_qc_enabled",
    "countdown_prosody_max_median_semitone_error",
    "countdown_prosody_min_pass_score",
    "countdown_prosody_min_warn_score",
    "countdown_prosody_failure_blocks_mix",
    "max_attempts_per_candidate",
    "initial_candidate_count",
    "retry_candidate_count",
    "low_temperature_retry_enabled",
    "low_temperature_retry_candidate_count",
    "low_temperature_retry_temperature",
    "pronunciation_qc_enabled",
    "pronunciation_qc_backend",
    "pronunciation_qc_workers",
    "pronunciation_qc_pass_coverage",
    "pronunciation_qc_warn_coverage",
    "pronunciation_qc_max_observed_unit_ratio",
    "pronunciation_qc_max_extra_units",
    "pronunciation_qc_failure_blocks_mix",
    "numeric_cadence_periods_enabled",
    "numeric_cadence_min_values",
    "numeric_sequence_qc_enabled",
    "numeric_sequence_qc_require_contiguous",
    "numeric_sequence_qc_failure_blocks_mix",
    "korean_clarity_retry_enabled",
    "korean_clarity_retry_candidate_count",
    "korean_clarity_temperature",
    "korean_clarity_top_p",
    "korean_clarity_top_k",
    "korean_clarity_parallel_infer",
    "zero_shot_candidate_count",
    "duration_rewrite_backend",
    "duration_rewrite_timing",
    "duration_rewrite_max_attempts",
    "duration_rewrite_pre_candidate_count",
    "timing_expansion_enabled",
    "timing_expansion_min_ratio",
    "timing_expansion_max_attempts",
    "terminal_failure_policy",
    "timing_quality_tolerance",
    "timefit_max_tempo",
    "timefit_max_stretch",
    "timefit_micro_max_sec",
    "timefit_micro_max_tempo",
    "timefit_long_min_sec",
    "timefit_long_max_stretch",
    "rescue_duration_tolerance",
    "rescue_timefit_top_k",
    "rescue_timefit_max_tempo",
    "rescue_timefit_max_stretch",
    "rescue_micro_segment_max_sec",
    "micro_segment_unfit_policy",
    "few_shot_force",
    "few_shot_version",
]
_RVC_FLAT_FIELDS = [
    "required",
    "backend",
    "train_required",
    "train_backend",
    "train_command",
    "train_working_dir",
    "train_timeout_sec",
    "train_experiment_name",
    "train_sample_rate",
    "train_epochs",
    "train_epoch_policy",
    "train_quality_preset",
    "train_batch_size",
    "train_min_quality_score",
    "train_preferred_chars_per_sec",
    "train_max_chars_per_sec",
    "train_max_clip_sec",
    "train_min_snr_db",
    "train_max_background_bleed_db",
    "train_max_side_to_mid_db",
    "train_target_clean_sec",
    "train_auto_epoch_min",
    "train_auto_epoch_max",
    "train_min_clean_sec",
    "train_min_clean_segments",
    "train_augment_enabled",
    "train_augment_min_real_sec",
    "train_augment_max_multiplier",
    "train_preprocess_processes",
    "train_f0_workers",
    "train_feature_workers",
    "train_save_every_epoch",
    "train_reuse_intermediate_cache",
    "train_output_model_path",
    "train_output_index_path",
    "command",
    "batch_infer",
    "batch_command",
    "batch_size",
    "batch_concurrency",
    "working_dir",
    "timeout_sec",
    "concurrency",
    "model_path",
    "index_path",
    "device",
    "f0_up_key",
    "f0_method",
    "index_rate",
    "filter_radius",
    "resample_sr",
    "rms_mix_rate",
    "protect",
    "failure_policy",
    "allow_pre_rvc_fallback",
    "duration_tolerance",
    "auto_profiles",
    "speaker_models",
]
_MIX_FLAT_FIELDS = [
    "profile",
    "background_bed",
    "dialogue_gain_db",
    "dialogue_fade_ms",
    "loudness_strategy",
    "peak_limit_dbfs",
    "allow_korean_timing_draft",
]
_BACKGROUND_FLAT_FIELDS = [
    "gain_db",
    "speech_suppression",
    "speech_suppression_db",
    "speech_suppression_pad_sec",
    "speech_suppression_fade_ms",
]
_VOICE_BANK_DIARIZATION_FLAT_FIELDS = [
    "model_id",
    "embedding_model_id",
    "auto_download",
    "min_speakers",
    "max_speakers",
    "embedding_match_threshold",
]
_QWEN_TTS_FLAT_FIELDS = [
    "model_id",
    "candidate_count",
    "device_map",
    "dtype",
    "attn_implementation",
    "local_files_only",
    "candidate_batch_size",
    "segment_batch_size",
    "target_vram_gb",
    "temperature",
    "top_p",
    "max_new_tokens",
    "x_vector_only_mode",
]
_FISH_TTS_FLAT_FIELDS = [
    "repo_dir",
    "base_url",
    "candidate_count",
    "timeout_sec",
    "chunk_length",
    "temperature",
    "top_p",
    "repetition_penalty",
    "max_new_tokens",
    "normalize",
    "latency",
]
_COSYVOICE_FLAT_FIELDS = [
    "repo_dir",
    "model_dir",
    "base_url",
    "candidate_count",
    "timeout_sec",
    "mode",
    "sample_rate",
    "instruct_text",
]

_PROJECT_CONFIG_FLAT_ALIASES: dict[str, FlatConfigPath] = {
    "mix_sample_rate": ("mix", "sample_rate"),
    "gemma_sample_rate": ("gemma", "sample_rate"),
    "default_gemma_backend": ("gemma", "default_backend"),
    "hf_local_files_only": ("safety", "hf_local_files_only"),
    "duration_tolerance": ("mix", "duration_tolerance"),
    "voice_bank_path": ("voice_bank", "path"),
    "speaker_assignment_backend": ("voice_bank", "speaker_assignment_backend"),
    "source_speaker_training_min_dominant_overlap_ratio": (
        "voice_bank",
        "source_speaker_training_min_dominant_overlap_ratio",
    ),
    "voice_bank_source_speaker_training_min_dominant_overlap_ratio": (
        "voice_bank",
        "source_speaker_training_min_dominant_overlap_ratio",
    ),
    "source_separation_backend": ("asr", "source_separation_backend"),
    "source_separation_model": ("asr", "source_separation_model"),
    "source_separation_device": ("asr", "source_separation_device"),
    "segmentation_min_segment_sec": ("asr", "segmentation_min_segment_sec"),
    "segmentation_max_segment_sec": ("asr", "segmentation_max_segment_sec"),
    "segmentation_silence_db": ("asr", "segmentation_silence_db"),
    "segmentation_min_silence_sec": ("asr", "segmentation_min_silence_sec"),
    "asr_hotwords": ("asr", "correction_profile", "hotwords"),
    "asr_repair_suspicious_text_patterns": (
        "asr",
        "correction_profile",
        "repair_suspicious_text_patterns",
    ),
    "asr_text_replacements": ("asr", "correction_profile", "text_replacements"),
    "asr_review_suspicious_text_patterns": (
        "asr",
        "correction_profile",
        "review_suspicious_text_patterns",
    ),
    "asr_review_candidate_replacements": (
        "asr",
        "correction_profile",
        "review_candidate_replacements",
    ),
    "asr_review_auto_resolution_rules": (
        "asr",
        "correction_profile",
        "review_auto_resolution_rules",
    ),
    "asr_translation_backcheck_source_patterns": (
        "asr",
        "correction_profile",
        "translation_backcheck_source_patterns",
    ),
    "asr_translation_backcheck_ko_patterns": (
        "asr",
        "correction_profile",
        "translation_backcheck_ko_patterns",
    ),
}
_PROJECT_CONFIG_FLAT_ALIASES.update(
    {f"asr_{field}": ("asr", field) for field in _ASR_DIRECT_FLAT_FIELDS}
)
_PROJECT_CONFIG_FLAT_ALIASES.update(
    {f"qwen_asr_{field}": ("asr", f"qwen_{field}") for field in _QWEN_ASR_FLAT_FIELDS}
)
_PROJECT_CONFIG_FLAT_ALIASES.update(
    {f"gemma_{field}": ("gemma", field) for field in _GEMMA_SIMPLE_FLAT_FIELDS}
)
_PROJECT_CONFIG_FLAT_ALIASES.update(
    {
        f"gemma_llama_cpp_{field}": ("gemma", f"llama_cpp_{field}")
        for field in _GEMMA_LLAMA_CPP_FLAT_FIELDS
    }
)
_PROJECT_CONFIG_FLAT_ALIASES.update(
    {f"gemma_text_{field}": ("gemma", f"text_{field}") for field in _GEMMA_TEXT_FLAT_FIELDS}
)
_PROJECT_CONFIG_FLAT_ALIASES["gemma_audio_style_concurrency"] = (
    "gemma",
    "audio_style_concurrency",
)
_PROJECT_CONFIG_FLAT_ALIASES["gemma_audio_style_scope"] = (
    "gemma",
    "audio_style_scope",
)
_PROJECT_CONFIG_FLAT_ALIASES.update(
    {f"gsv_{field}": ("gsv", field) for field in _GSV_FLAT_FIELDS}
)
_PROJECT_CONFIG_FLAT_ALIASES.update(
    {f"rvc_{field}": ("rvc", field) for field in _RVC_FLAT_FIELDS}
)
_PROJECT_CONFIG_FLAT_ALIASES.update(
    {f"mix_{field}": ("mix", field) for field in _MIX_FLAT_FIELDS}
)
_PROJECT_CONFIG_FLAT_ALIASES.update(
    {f"background_{field}": ("mix", f"background_{field}") for field in _BACKGROUND_FLAT_FIELDS}
)
_PROJECT_CONFIG_FLAT_ALIASES.update(
    {
        f"diarization_{field}": ("voice_bank", f"diarization_{field}")
        for field in _VOICE_BANK_DIARIZATION_FLAT_FIELDS
    }
)
_PROJECT_CONFIG_FLAT_ALIASES.update(
    {f"qwen_tts_{field}": ("tts", f"qwen_{field}") for field in _QWEN_TTS_FLAT_FIELDS}
)
_PROJECT_CONFIG_FLAT_ALIASES.update(
    {f"fish_tts_{field}": ("tts", f"fish_{field}") for field in _FISH_TTS_FLAT_FIELDS}
)
_PROJECT_CONFIG_FLAT_ALIASES.update(
    {f"cosyvoice_{field}": ("tts", f"cosyvoice_{field}") for field in _COSYVOICE_FLAT_FIELDS}
)



class RightsAudit(StrictBaseModel):
    confirmed: bool = False
    confirmed_at: datetime | None = None
    command: str | None = None
    source_path: str | None = None
    source_sha256: str | None = None
    voice_reference_notice: str = (
        "User confirmed they own or have permission/consent for all voice references."
    )
    distribution_notice: str = (
        "User confirmed they own or have permission for source content and distribution."
    )
    local_processing_notice: str = (
        "Processing is local-first unless the user explicitly configures a remote endpoint."
    )
    history: list[dict[str, Any]] = Field(default_factory=list)


class SourceInfo(StrictBaseModel):
    path: str
    duration_sec: float = Field(ge=0)
    sample_rate: int | None = Field(default=None, ge=1)
    channels: int | None = Field(default=None, ge=1)
    codec: str | None = None
    format_name: str | None = None
    has_video: bool = False
    bit_rate: int | None = Field(default=None, ge=0)
    raw: dict[str, Any] = Field(default_factory=dict)


class NonverbalCue(StrictBaseModel):
    kind: str
    source_text: str = ""
    normalized_text: str = ""
    position: int = Field(default=0, ge=0)
    intensity: float = Field(default=0.5, ge=0.0, le=1.0)
    pause_sec: float | None = Field(default=None, ge=0.0)
    notes: str = ""


class StyleMetadata(StrictBaseModel):
    ref_style: str = "whisper_close"
    emotion: Emotion = "gentle"
    pace: Pace = "slow"
    volume: Volume = "soft"
    spatial_style: SpatialStyle = "center"
    style_tags: list[str] = Field(default_factory=list)


class ScriptRetryPolicy(StrictBaseModel):
    duration_too_long: DurationRetryAction = "request_shorter_script"
    duration_too_short: DurationRetryAction = "request_longer_script"
    repetition_detected: VariationRetryAction = "regenerate_with_variation_and_new_seed"
    omission_detected: VariationRetryAction = "regenerate_with_variation_and_new_seed"
    max_script_rewrites: int = Field(default=2, ge=0, le=5)
    max_tts_regenerations: int = Field(default=2, ge=0, le=8)
    seed_strategy: str = "increment_candidate_seed"
    shortening_prompt: str = (
        "Shorten tts_text while preserving the source meaning, ASMR tone, and nonverbal metadata."
    )
    variation_prompt: str = (
        "Generate a fresh wording with the same meaning and style metadata; avoid repeated or omitted phrases."
    )


class JapaneseScript(StrictBaseModel):
    literal_ja: str = ""
    ja_text: str
    tts_text: str
    tts_language: str = "ja"
    source_language: str = "ja"
    target_language: str = "ja"
    ref_style: str = "whisper_close"
    emotion: Emotion = "gentle"
    pace: Pace = "slow"
    volume: Volume = "soft"
    nonverbal_cues: list[NonverbalCue] = Field(default_factory=list)
    spatial_style: SpatialStyle = "center"
    expected_tts_duration_sec: float = Field(default=1.0, ge=0)
    style_tags: list[str] = Field(default_factory=list)
    style: StyleMetadata | None = None
    retry_policy: ScriptRetryPolicy = Field(default_factory=ScriptRetryPolicy)
    rewrite_count: int = Field(default=0, ge=0)
    risk_flags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _sync_style(self) -> JapaneseScript:
        if self.style is None:
            self.style = StyleMetadata(
                ref_style=self.ref_style,
                emotion=self.emotion,
                pace=self.pace,
                volume=self.volume,
                spatial_style=self.spatial_style,
                style_tags=self.style_tags,
            )
        return self

    @field_validator("ja_text", "tts_text")
    @classmethod
    def _text_must_not_be_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("script text must not be empty")
        return value


class SourceScript(StrictBaseModel):
    text: str = ""
    language: str = "ja"
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    backend: str
    start: float = Field(ge=0)
    end: float = Field(gt=0)


class TranscriptWord(StrictBaseModel):
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    text: str
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class SourceLaneTranscript(StrictBaseModel):
    lane_id: str
    channel: Literal["left", "right", "mid", "side", "mono"]
    text: str = ""
    language: str = "ja"
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    pan: float = Field(default=0.0, ge=-1.0, le=1.0)
    spatial_style: SpatialStyle = "center"
    backend: str
    candidate_id: str
    clip_start: float = Field(ge=0)
    clip_end: float = Field(gt=0)
    words: list[TranscriptWord] = Field(default_factory=list)
    boundary_clipped: bool = False
    review_reasons: list[str] = Field(default_factory=list)

    @field_validator("lane_id")
    @classmethod
    def _lane_id_safe(cls, value: str) -> str:
        return validate_file_safe_id(value, "lane_id")

    @model_validator(mode="after")
    def _validate_times(self) -> SourceLaneTranscript:
        if self.end <= self.start:
            raise ValueError("lane transcript end must be greater than start")
        if self.clip_end <= self.clip_start:
            raise ValueError("lane transcript clip_end must be greater than clip_start")
        return self


class KoreanTranslation(StrictBaseModel):
    ko_literal: str
    ko_natural: str
    notes: list[str] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    model: str
    batch_id: str


class TTSCandidate(StrictBaseModel):
    candidate_index: int = Field(ge=0)
    seed: int
    payload: dict[str, Any] = Field(default_factory=dict)
    output_path: str
    duration_sec: float | None = Field(default=None, ge=0)
    backend: Literal[
        "mock",
        "gpt-sovits",
        "gpt-sovits-countdown-renderer",
        "qwen-tts",
        "fish-tts",
        "cosyvoice",
    ] = "mock"
    selected: bool = False
    error: str | None = None
    duration_ratio: float | None = Field(default=None, ge=0)
    duration_gate: Literal["pass", "too_short", "too_long", "unknown"] = "unknown"
    timing_quality_gate: GSVTimingQualityGate = "unknown"
    acceptable_for_mix: bool = False
    selection_score: float | None = None
    selection_reason: str = ""
    retry_summary: dict[str, Any] = Field(default_factory=dict)


class TTSMetadata(StrictBaseModel):
    backend: Literal[
        "mock",
        "gpt-sovits",
        "gpt-sovits-countdown-renderer",
        "qwen-tts",
        "fish-tts",
        "cosyvoice",
    ] = "mock"
    ref_style: str = "whisper_close"
    speed_factor: float = Field(default=1.0, gt=0)
    candidate_count: int = Field(default=1, ge=1)
    selected_candidate_path: str | None = None
    candidates: list[TTSCandidate] = Field(default_factory=list)
    source_language: str = "ja"
    target_language: str = "ja"
    cross_lingual_voice_transfer: bool = False
    retry_summary: dict[str, Any] = Field(default_factory=dict)


class RVCMetadata(StrictBaseModel):
    backend: Literal["command", "mock"]
    input_path: str
    output_path: str | None = None
    selected_profile_name: str | None = None
    candidate_paths: list[str] = Field(default_factory=list)
    model_path: str | None = None
    index_path: str | None = None
    settings: dict[str, Any] = Field(default_factory=dict)
    pre_duration_sec: float | None = Field(default=None, ge=0)
    post_duration_sec: float | None = Field(default=None, ge=0)
    duration_ratio: float | None = Field(default=None, ge=0)
    accepted: bool = False
    fallback_used: bool = False
    fallback_reason: str | None = None
    error: str | None = None
    command: list[str] | None = None
    attempts: list[dict[str, Any]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)


class QCMetadata(StrictBaseModel):
    duration_ratio: float | None = Field(default=None, ge=0)
    peak_dbfs: float | None = None
    rms_dbfs: float | None = None
    clipping_ratio: float | None = Field(default=None, ge=0)
    leading_silence_sec: float | None = Field(default=None, ge=0)
    trailing_silence_sec: float | None = Field(default=None, ge=0)
    text_match_score: float = Field(default=1.0, ge=0, le=1)
    pronunciation_score: float = Field(default=1.0, ge=0, le=1)
    asmr_style_score: float = Field(default=1.0, ge=0, le=1)
    timing_score: float = Field(default=1.0, ge=0, le=1)
    repetition_detected: bool = False
    omission_detected: bool = False
    unsafe_or_rights_issue: bool = False
    recommendation: QCRecommendation = "pass"
    issues: list[str] = Field(default_factory=list)
    score: float = Field(default=1.0, ge=0, le=1)
    status: SegmentStatus = "ok"


class Segment(StrictBaseModel):
    id: str
    speaker_id: str | None = None
    parent_segment_id: str | None = None
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    duration: float = Field(gt=0)
    audio_for_gemma: str
    audio_for_mix: str
    estimated_pan: float = Field(default=0.0, ge=-1.0, le=1.0)
    keep_original_texture: bool = True
    status: SegmentStatus = "raw"
    analysis: dict[str, Any] = Field(default_factory=dict)
    source_script: SourceScript | None = None
    source_lane: SourceLaneTranscript | None = None
    source_lanes: list[SourceLaneTranscript] = Field(default_factory=list)
    script: JapaneseScript | None = None
    translation_ko: KoreanTranslation | None = None
    tts: TTSMetadata | None = None
    rvc: RVCMetadata | None = None
    qc: QCMetadata | None = None
    mix: dict[str, Any] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def _id_safe(cls, value: str) -> str:
        return validate_file_safe_id(value, "segment id")

    @field_validator("speaker_id")
    @classmethod
    def _speaker_id_safe(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return validate_file_safe_id(value, "speaker_id")

    @field_validator("parent_segment_id")
    @classmethod
    def _parent_segment_id_safe(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return validate_file_safe_id(value, "parent_segment_id")

    @model_validator(mode="after")
    def _validate_times(self) -> Segment:
        if self.end <= self.start:
            raise ValueError("segment end must be greater than start")
        expected = self.end - self.start
        if abs(expected - self.duration) > 0.01:
            raise ValueError("segment duration must match end-start")
        return self


class PipelineManifest(StrictBaseModel):
    schema_version: str = SCHEMA_VERSION
    project_config: ProjectConfig = Field(default_factory=ProjectConfig)
    source_info: SourceInfo | None = None
    rights_audit: RightsAudit = Field(default_factory=RightsAudit)
    segments: list[Segment] = Field(default_factory=list)
    stage_state: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    def mark_updated(self) -> None:
        self.updated_at = utc_now()
