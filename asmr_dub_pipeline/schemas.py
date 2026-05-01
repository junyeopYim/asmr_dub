from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = "1.0"

SegmentStatus = Literal[
    "raw",
    "analyzed",
    "scripted",
    "synthesized",
    "rvc_trained",
    "rvc_converted",
    "ok",
    "needs_regeneration",
    "needs_manual_review",
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
SpeakerAssignmentBackend = Literal["none", "mock", "pyannote"]


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


class ProjectConfig(StrictBaseModel):
    project_name: str = "asmr-dub-project"
    source_language: SourceLanguage = "ja"
    target_language: TargetLanguage = "ko"
    candidate_count: int = Field(default=1, ge=1, le=8)
    base_seed: int = 12345
    mix_sample_rate: int = 48_000
    gemma_sample_rate: int = 16_000
    default_gemma_backend: Literal["mock", "hf", "http", "llama_cpp"] = "mock"
    gemma_model_id: str = "google/gemma-4-E4B-it"
    gemma_http_url: str | None = None
    gemma_http_send_audio: bool = False
    hf_local_files_only: bool = True
    gemma_llama_cpp_cli_path: str = (
        ".cache/llama_cpp/src/llama.cpp/build/bin/llama-mtmd-cli"
    )
    gemma_llama_cpp_model_path: str = (
        ".cache/llama_cpp/models/HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive/"
        "Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf"
    )
    gemma_llama_cpp_mmproj_path: str = (
        ".cache/llama_cpp/models/HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive/"
        "mmproj-Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-f16.gguf"
    )
    gemma_llama_cpp_timeout_sec: float = Field(default=600.0, gt=0)
    gemma_llama_cpp_ctx_size: int = Field(default=4096, ge=512)
    gemma_llama_cpp_n_predict: int = Field(default=1024, ge=64)
    gemma_llama_cpp_gpu_layers: int = Field(default=999, ge=0)
    gemma_llama_cpp_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    gemma_llama_cpp_seed: int = 12345
    gemma_llama_cpp_extra_args: list[str] = Field(default_factory=list)
    asr_backend: Literal["mock", "faster_whisper"] = "faster_whisper"
    asr_model_id: str = "mobiuslabsgmbh/faster-whisper-large-v3-turbo"
    asr_language: str = "ja"
    asr_local_files_only: bool = True
    asr_resegment_from_chunks: bool = True
    asr_resegment_min_sec: float = Field(default=0.8, gt=0)
    asr_resegment_merge_gap_sec: float = Field(default=0.45, ge=0)
    source_separation_backend: Literal["auto", "none", "demucs", "mock"] = "auto"
    source_separation_model: str = "htdemucs"
    source_separation_device: str | None = None
    gemma_text_server_url: str = "http://127.0.0.1:8080"
    gemma_text_server_auto_start: bool = True
    gemma_text_server_command: list[str] = Field(default_factory=list)
    gemma_text_batch_size: int = Field(default=1, ge=1, le=200)
    gemma_text_concurrency: int = Field(default=4, ge=1, le=8)
    gemma_text_n_predict: int = Field(default=2048, ge=64)
    gemma_text_timeout_sec: float = Field(default=180.0, gt=0)
    gemma_text_retries: int = Field(default=1, ge=0, le=5)
    gemma_text_server_startup_timeout_sec: float = Field(default=120.0, gt=0)
    gemma_text_server_shutdown_timeout_sec: float = Field(default=10.0, gt=0)
    gsv_url: str = "http://127.0.0.1:9880"
    gsv_gpt_weights_path: str | None = None
    gsv_sovits_weights_path: str | None = None
    gsv_gpt_weights_policy: GSVGPTWeightsPolicy = "auto"
    gsv_sovits_weights_policy: GSVSoVITSWeightsPolicy = "auto"
    gsv_speaker_models: dict[str, GSVSpeakerConfig] = Field(default_factory=dict)
    gsv_timeout_sec: float = Field(default=120.0, gt=0)
    gsv_retries: int = Field(default=2, ge=0)
    gsv_concurrency: int = Field(default=3, ge=1, le=8)
    gsv_auto_start: bool = False
    gsv_server_command: list[str] = Field(default_factory=list)
    gsv_server_cwd: str | None = None
    gsv_server_startup_timeout_sec: float = Field(default=120.0, gt=0)
    gsv_server_shutdown_timeout_sec: float = Field(default=10.0, gt=0)
    gsv_trim_edge_silence: bool = True
    gsv_trim_silence_threshold_db: float = -50.0
    gsv_trim_silence_keep_sec: float = Field(default=0.08, ge=0.0, le=1.0)
    gsv_few_shot_enabled: bool = True
    gsv_few_shot_target_sec: float = Field(default=60.0, gt=0)
    gsv_few_shot_min_clip_sec: float = Field(default=1.0, gt=0)
    gsv_few_shot_max_clip_sec: float = Field(default=10.0, gt=0)
    gsv_few_shot_min_quality_score: float = Field(default=0.20, ge=0.0, le=1.0)
    gsv_ref_min_sec: float = Field(default=3.0, gt=0)
    gsv_ref_max_sec: float = Field(default=10.0, gt=0)
    gsv_ref_min_quality_score: float = Field(default=0.25, ge=0.0, le=1.0)
    gsv_ko_text_min_hangul_ratio: float = Field(default=0.20, ge=0.0, le=1.0)
    gsv_tts_min_speed_factor: float = Field(default=0.85, gt=0.0, le=1.0)
    gsv_tts_max_speed_factor: float = Field(default=1.12, ge=1.0, le=1.35)
    gsv_few_shot_force: bool = False
    gsv_few_shot_version: Literal["auto", "v1", "v2", "v3", "v4", "v2Pro", "v2ProPlus"] = "auto"
    voice_bank_path: str = "voice_bank/voice_bank_manifest.json"
    speaker_assignment_backend: SpeakerAssignmentBackend = "none"
    diarization_model_id: str = "pyannote/speaker-diarization-community-1"
    diarization_embedding_model_id: str = "pyannote/wespeaker-voxceleb-resnet34-LM"
    diarization_auto_download: bool = True
    diarization_min_speakers: int | None = Field(default=None, ge=1)
    diarization_max_speakers: int | None = Field(default=None, ge=1)
    diarization_embedding_match_threshold: float = Field(default=0.78, ge=0.0, le=1.0)
    segmentation_min_segment_sec: float = Field(default=0.25, gt=0)
    segmentation_max_segment_sec: float = Field(default=20.0, gt=0)
    segmentation_silence_db: float = -45.0
    segmentation_min_silence_sec: float = Field(default=0.30, ge=0)
    duration_tolerance: float = Field(default=0.20, gt=0, lt=1)
    mix_profile: MixProfile = "asmr_stereo"
    mix_background_bed: MixBackgroundBed = "preserve_original"
    background_gain_db: float = Field(default=-18.0, ge=-60.0, le=6.0)
    background_speech_suppression: bool = True
    background_speech_suppression_db: float = Field(default=-42.0, ge=-80.0, le=0.0)
    background_speech_suppression_pad_sec: float = Field(default=0.06, ge=0.0, le=1.0)
    background_speech_suppression_fade_ms: float = Field(default=30.0, ge=0.0, le=500.0)
    mix_dialogue_gain_db: float = Field(default=0.0, ge=-60.0, le=12.0)
    mix_dialogue_fade_ms: float | None = Field(default=None, ge=0.0, le=250.0)
    mix_loudness_strategy: MixLoudnessStrategy = "peak_guard_only"
    mix_peak_limit_dbfs: float = Field(default=-1.0, ge=-24.0, le=0.0)
    mix_allow_korean_timing_draft: bool = False
    rvc_required: bool = True
    rvc_backend: Literal["command", "mock"] = "command"
    rvc_train_required: bool = True
    rvc_train_backend: Literal["command", "mock"] = "command"
    rvc_train_command: list[str] = Field(default_factory=list)
    rvc_train_working_dir: str | None = None
    rvc_train_timeout_sec: float = Field(default=14400.0, gt=0)
    rvc_train_experiment_name: str = "asmr-rvc-speaker-1"
    rvc_train_sample_rate: int = Field(default=48_000, ge=0)
    rvc_train_batch_size: int = Field(default=0, ge=0, le=64)
    rvc_train_preprocess_processes: int = Field(default=0, ge=0)
    rvc_train_f0_workers: int = Field(default=0, ge=0)
    rvc_train_feature_workers: int = Field(default=0, ge=0)
    rvc_train_save_every_epoch: int = Field(default=50, ge=1)
    rvc_train_reuse_intermediate_cache: bool = True
    rvc_train_output_model_path: str | None = None
    rvc_train_output_index_path: str | None = None
    rvc_command: list[str] = Field(default_factory=list)
    rvc_working_dir: str | None = None
    rvc_timeout_sec: float = Field(default=180.0, gt=0)
    rvc_concurrency: int = Field(default=1, ge=1, le=8)
    rvc_model_path: str | None = None
    rvc_index_path: str | None = None
    rvc_device: str = "cuda:0"
    rvc_f0_up_key: int = 0
    rvc_f0_method: str = "rmvpe"
    rvc_index_rate: float = Field(default=0.45, ge=0.0, le=1.0)
    rvc_filter_radius: int = Field(default=3, ge=0)
    rvc_resample_sr: int = Field(default=48_000, ge=0)
    rvc_rms_mix_rate: float = Field(default=0.25, ge=0.0, le=1.0)
    rvc_protect: float = Field(default=0.33, ge=0.0, le=0.5)
    rvc_failure_policy: Literal["retry_then_error", "error"] = "retry_then_error"
    rvc_allow_pre_rvc_fallback: bool = False
    rvc_duration_tolerance: float | None = Field(default=None, gt=0, lt=1)
    rvc_auto_profiles: list[RVCProfile] = Field(default_factory=default_rvc_profiles)
    rvc_speaker_models: dict[str, RVCSpeakerConfig] = Field(default_factory=dict)

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
        if self.gsv_ref_max_sec <= self.gsv_ref_min_sec:
            raise ValueError("gsv_ref_max_sec must be greater than gsv_ref_min_sec")
        if self.gsv_tts_max_speed_factor < self.gsv_tts_min_speed_factor:
            raise ValueError("gsv_tts_max_speed_factor must be >= gsv_tts_min_speed_factor")
        if self.rvc_required and not self.rvc_auto_profiles:
            raise ValueError("rvc_auto_profiles must contain at least one profile when RVC is required")
        if self.rvc_required and not self.rvc_train_required and not self.rvc_speaker_models:
            raise ValueError(
                "rvc_train_required must be true when RVC is required unless rvc_speaker_models are configured"
            )
        profile_names = [profile.name for profile in self.rvc_auto_profiles]
        if len(profile_names) != len(set(profile_names)):
            raise ValueError("rvc_auto_profiles names must be unique")
        if (
            self.diarization_min_speakers is not None
            and self.diarization_max_speakers is not None
            and self.diarization_min_speakers > self.diarization_max_speakers
        ):
            raise ValueError("diarization_min_speakers must be <= diarization_max_speakers")
        return self


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
    backend: Literal["mock", "gpt-sovits"] = "mock"
    selected: bool = False
    error: str | None = None
    duration_ratio: float | None = Field(default=None, ge=0)
    duration_gate: Literal["pass", "too_short", "too_long", "unknown"] = "unknown"
    acceptable_for_mix: bool = False
    selection_score: float | None = None
    selection_reason: str = ""
    retry_summary: dict[str, Any] = Field(default_factory=dict)


class TTSMetadata(StrictBaseModel):
    backend: Literal["mock", "gpt-sovits"] = "mock"
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
