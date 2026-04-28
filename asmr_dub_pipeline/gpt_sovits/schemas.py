from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class GPTSoVITSBase(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GPTSoVITSRef(GPTSoVITSBase):
    ref_audio_path: str
    prompt_text: str
    prompt_lang: str = "ja"
    aux_ref_audio_paths: list[str] = Field(default_factory=list)
    source_language: str = "ja"
    target_language: str = "ko"
    cross_lingual_role: str = ""


class GPTSoVITSTTSOptions(GPTSoVITSBase):
    text_lang: str = "ja"
    top_k: int = 15
    top_p: float = 1.0
    temperature: float = 1.0
    text_split_method: str = "cut5"
    batch_size: int = 1
    batch_threshold: float = 0.75
    split_bucket: bool = True
    speed_factor: float = 1.0
    fragment_interval: float = 0.3
    seed: int = -1
    media_type: str = "wav"
    streaming_mode: bool | int = False
    parallel_infer: bool = True
    repetition_penalty: float = 1.35
    sample_steps: int = 32
    super_sampling: bool = False
    overlap_length: int = 2
    min_chunk_length: int = 16


class GPTSoVITSTTSRequest(GPTSoVITSBase):
    text: str
    text_lang: str = "ja"
    ref_audio_path: str
    prompt_text: str
    prompt_lang: str = "ja"
    aux_ref_audio_paths: list[str] = Field(default_factory=list)
    top_k: int = 15
    top_p: float = 1.0
    temperature: float = 1.0
    text_split_method: str = "cut5"
    batch_size: int = 1
    batch_threshold: float = 0.75
    split_bucket: bool = True
    speed_factor: float = 1.0
    fragment_interval: float = 0.3
    seed: int = -1
    media_type: str = "wav"
    streaming_mode: bool | int = False
    parallel_infer: bool = True
    repetition_penalty: float = 1.35
    sample_steps: int = 32
    super_sampling: bool = False
    overlap_length: int = 2
    min_chunk_length: int = 16

    def as_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)
