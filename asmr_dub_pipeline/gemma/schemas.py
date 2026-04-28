from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, ValidationError

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


TaskName = Literal["analyze", "script", "qc"]

TASK_SCHEMAS: dict[TaskName, type[GemmaTaskSchema]] = {
    "analyze": GemmaAnalysisResult,
    "script": GemmaScriptResult,
    "qc": GemmaQCResult,
}


TASK_REQUIRED_KEYS: dict[TaskName, set[str]] = {
    task: {name for name, field in model.model_fields.items() if field.is_required()}
    for task, model in TASK_SCHEMAS.items()
}


def validate_gemma_task_response(task: TaskName, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        return TASK_SCHEMAS[task].model_validate(payload).model_dump(mode="json")
    except ValidationError:
        raise


def parse_gemma_task_response(task: TaskName, payload: Mapping[str, Any] | str) -> dict[str, Any]:
    """Parse raw Gemma output and validate it against the task's Pydantic schema."""
    if isinstance(payload, str):
        data = loads_json_dict(payload, required_keys=TASK_REQUIRED_KEYS[task])
    elif isinstance(payload, Mapping):
        data = dict(payload)
        if not TASK_REQUIRED_KEYS[task].issubset(data):
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
