from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import Field, field_validator

from asmr_dub_pipeline.schemas import StrictBaseModel, utc_now, validate_file_safe_id

TTSBackendName = Literal[
    "mock",
    "gpt_sovits",
    "qwen_tts",
    "fish_tts",
    "cosyvoice",
]


class TTSRoute(StrictBaseModel):
    """Per-segment routing decision for TTS candidate generation."""

    segment_id: str
    backends: list[TTSBackendName]
    reason_codes: list[str] = Field(default_factory=list)
    candidate_budget: int = Field(default=1, ge=1)
    route_id: str | None = None

    @field_validator("segment_id")
    @classmethod
    def _segment_id_safe(cls, value: str) -> str:
        return validate_file_safe_id(value, "segment_id")


class TTSCandidate(StrictBaseModel):
    """Common on-disk metadata for one synthesized TTS candidate."""

    segment_id: str
    candidate_id: str
    backend: TTSBackendName
    wav_path: str
    metadata_path: str = ""
    duration_sec: float | None = Field(default=None, ge=0)
    input_hash: str
    backend_config_hash: str
    attempt: int = Field(default=0, ge=0)
    payload: dict[str, Any] = Field(default_factory=dict)
    generation_id: str
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("segment_id", "candidate_id")
    @classmethod
    def _ids_safe(cls, value: str, info) -> str:
        return validate_file_safe_id(value, info.field_name)


class CandidateScore(StrictBaseModel):
    """Selector score and hard-fail reasons for one candidate."""

    candidate_id: str
    backend: TTSBackendName
    blocked: bool = False
    hard_fail_reasons: list[str] = Field(default_factory=list)
    score: float = 0.0
    score_parts: dict[str, float] = Field(default_factory=dict)

    @field_validator("candidate_id")
    @classmethod
    def _candidate_id_safe(cls, value: str) -> str:
        return validate_file_safe_id(value, "candidate_id")


class SelectionResult(StrictBaseModel):
    """Result of selecting one candidate from a candidate pool."""

    segment_id: str
    selected: TTSCandidate | None = None
    scores: list[CandidateScore] = Field(default_factory=list)
    route_reason_codes: list[str] = Field(default_factory=list)
    terminal_reason: str | None = None

    @field_validator("segment_id")
    @classmethod
    def _segment_id_safe(cls, value: str) -> str:
        return validate_file_safe_id(value, "segment_id")
