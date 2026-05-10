from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, StrictBool, ValidationError, field_validator

from asmr_dub_pipeline.schemas import (
    Emotion,
    Pace,
    QCRecommendation,
    ScriptRetryPolicy,
    SpatialStyle,
    Volume,
)

from .json_repair import JSONRepairError, loads_json_dict


class GemmaTaskSchema(BaseModel):
    model_config = ConfigDict(extra="allow")


class VoiceTrainingDecision(BaseModel):
    clean_voice: StrictBool = True
    eligible: StrictBool = True
    reason: str = ""
    effect_tags: list[str] = Field(default_factory=list)
    same_speaker_under_effect: StrictBool = False


class AudioEffectEvent(BaseModel):
    model_config = ConfigDict(extra="allow")

    tag: str
    target: str = "voice"
    start_sec: float = Field(default=0.0, ge=0.0)
    end_sec: float | None = Field(default=None, ge=0.0)
    intensity: float = Field(default=1.0, ge=0.0, le=1.0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    params: dict[str, Any] = Field(default_factory=dict)


_SPATIAL_STYLE_ALIASES = {
    "centered": "center",
    "centre": "center",
    "middle": "center",
    "here": "center",
    "front": "center",
    "close": "center_close",
    "near": "center_close",
    "center_near": "center_close",
    "centre_near": "center_close",
    "left": "left_close",
    "left_near": "left_close",
    "left_center": "left_close",
    "center_left": "left_close",
    "close_left": "left_close",
    "right": "right_close",
    "right_near": "right_close",
    "right_center": "right_close",
    "center_right": "right_close",
    "close_right": "right_close",
    "far": "center_far",
    "distant": "center_far",
    "center_distant": "center_far",
    "sleepy": "sleepy_center",
    "binaural": "binaural_sweep",
    "sweep": "binaural_sweep",
}
_SPATIAL_STYLE_VALUES = set(get_args(SpatialStyle))
_AUDIO_STYLE_EFFECT_TAGS = {"telephone", "radio", "robot", "distortion", "reverb", "echo"}
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


def _coerce_audio_style_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null", "n/a"}:
        return []
    return [value]


def _coerce_audio_style_float(value: Any, default: float, low: float, high: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(low, min(high, number))


def _coerce_audio_style_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"true", "yes", "y", "1"}:
            return True
        if token in {"false", "no", "n", "0"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def _coerce_audio_style_confidence(value: Any, default: float = 0.0) -> float:
    if isinstance(value, str):
        label = value.strip().lower()
        labels = {"high": 0.9, "medium": 0.6, "med": 0.6, "low": 0.3}
        if label in labels:
            return labels[label]
    return _coerce_audio_style_float(value, default, 0.0, 1.0)


def _coerce_audio_style_intensity(value: Any, default: float = 1.0) -> float:
    if isinstance(value, str):
        label = value.strip().lower()
        labels = {"high": 1.0, "medium": 0.6, "med": 0.6, "low": 0.3}
        if label in labels:
            return labels[label]
    return _coerce_audio_style_float(value, default, 0.0, 1.0)


def _normalize_audio_style_token(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _normalize_audio_style_effect_tag(value: Any) -> str:
    tag = _normalize_audio_style_token(value)
    return _AUDIO_STYLE_EFFECT_ALIASES.get(tag, tag)


def _audio_style_first_present(mapping: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _normalize_audio_style_effect_event(raw_event: Any) -> dict[str, Any] | None:
    if not isinstance(raw_event, Mapping):
        return None
    tag = _normalize_audio_style_effect_tag(
        _audio_style_first_present(raw_event, ("tag", "name", "effect", "type"))
    )
    if tag == "none" or tag not in _AUDIO_STYLE_EFFECT_TAGS:
        return None
    target = _normalize_audio_style_token(raw_event.get("target") or "voice")
    if target in {"voice_effect", "speech", "speaker"}:
        target = "voice"
    if target in {"bg", "background_noise"}:
        target = "background"
    if target not in _AUDIO_STYLE_EVENT_TARGETS:
        target = "voice"
    params = raw_event.get("params")
    return {
        "tag": tag,
        "target": target,
        "start_sec": _coerce_audio_style_float(
            _audio_style_first_present(raw_event, ("start_sec", "start", "begin", "from")),
            0.0,
            0.0,
            86400.0,
        ),
        "end_sec": _coerce_audio_style_float(
            _audio_style_first_present(raw_event, ("end_sec", "end", "stop", "to")),
            0.0,
            0.0,
            86400.0,
        )
        if any(key in raw_event for key in ("end_sec", "end", "stop", "to"))
        else None,
        "intensity": _coerce_audio_style_intensity(raw_event.get("intensity")),
        "confidence": _coerce_audio_style_confidence(raw_event.get("confidence")),
        "params": dict(params) if isinstance(params, Mapping) else {},
    }


def _normalize_audio_style_effect_events(value: Any) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for raw_event in _coerce_audio_style_list(value):
        event = _normalize_audio_style_effect_event(raw_event)
        if event is not None:
            events.append(event)
    return events


def _normalize_audio_style_response(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    style = _normalize_spatial_style_alias(normalized.get("spatial_style") or "center")
    normalized["spatial_style"] = style if style in _SPATIAL_STYLE_VALUES else "center"
    normalized["style_tags"] = _coerce_audio_style_list(normalized.get("style_tags"))
    normalized["nonverbal_cues"] = _coerce_audio_style_list(normalized.get("nonverbal_cues"))
    normalized["risk_flags"] = _coerce_audio_style_list(normalized.get("risk_flags"))
    normalized["effect_events"] = _normalize_audio_style_effect_events(normalized.get("effect_events"))
    normalized["estimated_pan"] = _coerce_audio_style_float(
        normalized.get("estimated_pan"),
        0.0,
        -1.0,
        1.0,
    )
    normalized["keep_original_texture"] = _coerce_audio_style_bool(
        normalized.get("keep_original_texture"),
        True,
    )
    normalized["confidence"] = _coerce_audio_style_confidence(normalized.get("confidence"))
    raw_voice_training = normalized.get("voice_training")
    voice_training = dict(raw_voice_training) if isinstance(raw_voice_training, Mapping) else {}
    voice_training["clean_voice"] = _coerce_audio_style_bool(
        voice_training.get("clean_voice"),
        True,
    )
    voice_training["eligible"] = _coerce_audio_style_bool(
        voice_training.get("eligible"),
        True,
    )
    voice_training["reason"] = str(voice_training.get("reason") or "")
    voice_training["effect_tags"] = _coerce_audio_style_list(
        voice_training.get("effect_tags", normalized.get("effect_tags"))
    )
    voice_training["same_speaker_under_effect"] = _coerce_audio_style_bool(
        voice_training.get("same_speaker_under_effect"),
        False,
    )
    normalized["voice_training"] = voice_training
    return normalized


def _normalize_spatial_style_alias(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    token = value.strip().lower().replace("-", "_").replace(" ", "_")
    return _SPATIAL_STYLE_ALIASES.get(token, token)


class GemmaAnalysisResult(GemmaTaskSchema):
    source_language: str
    transcript_original: str
    literal_ja: str
    speech_style: str
    speaker_count: int = Field(ge=0)
    emotion: Emotion
    pace: Pace
    volume: Volume
    nonverbal_cues: list[dict[str, Any]] = Field(default_factory=list)
    spatial_style: SpatialStyle
    style_tags: list[str] = Field(default_factory=list)
    estimated_pan: float = Field(ge=-1.0, le=1.0)
    keep_original_texture: StrictBool
    risk_flags: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    voice_training: VoiceTrainingDecision = Field(default_factory=VoiceTrainingDecision)


class GemmaAudioStyleResult(GemmaTaskSchema):
    style_tags: list[str] = Field(default_factory=list)
    nonverbal_cues: list[dict[str, Any]] = Field(default_factory=list)
    spatial_style: SpatialStyle
    estimated_pan: float = Field(ge=-1.0, le=1.0)
    keep_original_texture: StrictBool
    risk_flags: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    effect_events: list[AudioEffectEvent] = Field(default_factory=list)
    voice_training: VoiceTrainingDecision = Field(default_factory=VoiceTrainingDecision)

    @field_validator("spatial_style", mode="before")
    @classmethod
    def _normalize_spatial_style(cls, value: Any) -> Any:
        return _normalize_spatial_style_alias(value)


class GemmaScriptResult(GemmaTaskSchema):
    literal_ja: str
    ja_text: str
    tts_text: str
    ref_style: str
    emotion: Emotion
    pace: Pace
    volume: Volume
    nonverbal_cues: list[dict[str, Any]] = Field(default_factory=list)
    spatial_style: SpatialStyle
    expected_tts_duration_sec: float = Field(ge=0.0)
    style_tags: list[str] = Field(default_factory=list)
    retry_policy: ScriptRetryPolicy = Field(default_factory=ScriptRetryPolicy)
    risk_flags: list[str] = Field(default_factory=list)


class GemmaQCResult(GemmaTaskSchema):
    text_match_score: float = Field(default=1.0, ge=0.0, le=1.0)
    pronunciation_score: float = Field(default=1.0, ge=0.0, le=1.0)
    asmr_style_score: float = Field(default=1.0, ge=0.0, le=1.0)
    timing_score: float = Field(default=1.0, ge=0.0, le=1.0)
    repetition_detected: StrictBool = False
    omission_detected: StrictBool = False
    unsafe_or_rights_issue: StrictBool = False
    recommendation: QCRecommendation
    issues: list[str] = Field(default_factory=list)


TaskName = Literal["analyze", "audio_style", "script", "qc"]

TASK_SCHEMAS: dict[TaskName, type[GemmaTaskSchema]] = {
    "analyze": GemmaAnalysisResult,
    "audio_style": GemmaAudioStyleResult,
    "script": GemmaScriptResult,
    "qc": GemmaQCResult,
}


TASK_REQUIRED_KEYS: dict[TaskName, set[str]] = {
    task: {name for name, field in model.model_fields.items() if field.is_required()}
    for task, model in TASK_SCHEMAS.items()
}


def validate_gemma_task_response(task: TaskName, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        if task == "audio_style":
            payload = _normalize_audio_style_response(payload)
        return TASK_SCHEMAS[task].model_validate(payload).model_dump(mode="json")
    except ValidationError:
        raise


def parse_gemma_task_response(task: TaskName, payload: Mapping[str, Any] | str) -> dict[str, Any]:
    """Parse raw Gemma output and validate it against the task's Pydantic schema."""
    if isinstance(payload, str):
        required_keys = None if task == "audio_style" else TASK_REQUIRED_KEYS[task]
        data = loads_json_dict(payload, required_keys=required_keys)
    elif isinstance(payload, Mapping):
        data = dict(payload)
        if task != "audio_style" and not TASK_REQUIRED_KEYS[task].issubset(data):
            for key in ("json", "response", "output", "text", "content"):
                nested = data.get(key)
                if isinstance(nested, str):
                    try:
                        data = loads_json_dict(nested, required_keys=TASK_REQUIRED_KEYS[task])
                        break
                    except JSONRepairError:
                        continue
    else:
        raise JSONRepairError("Gemma response must be a JSON object or text containing one.")
    return validate_gemma_task_response(task, data)
