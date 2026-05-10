from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import Field

from asmr_dub_pipeline.schemas import Segment, SourceScript, StrictBaseModel


class ASRUnavailableError(RuntimeError):
    pass


class ASRWord(StrictBaseModel):
    start: float
    end: float
    text: str
    confidence: float | None = None


class ASRChunk(StrictBaseModel):
    start: float
    end: float
    text: str
    language: str = "ja"
    confidence: float | None = None
    words: list[ASRWord] = Field(default_factory=list)


class ASRBackend(ABC):
    name: str

    @abstractmethod
    def transcribe(self, audio_path: Path, segments: Sequence[Segment]) -> list[ASRChunk]:
        raise NotImplementedError


def _overlap_sec(segment: Segment, chunk: ASRChunk) -> float:
    return max(0.0, min(segment.end, chunk.end) - max(segment.start, chunk.start))


def _weighted_confidence(items: list[tuple[ASRChunk, float]]) -> float | None:
    weighted = [(chunk.confidence, overlap) for chunk, overlap in items if chunk.confidence is not None]
    total = sum(overlap for _, overlap in weighted)
    if total <= 0:
        return None
    return sum(float(confidence) * overlap for confidence, overlap in weighted) / total


def map_chunks_to_segments(
    segments: Sequence[Segment],
    chunks: Sequence[ASRChunk],
    *,
    backend: str,
    fallback_language: str = "ja",
) -> dict[str, SourceScript | None]:
    assigned: dict[str, list[tuple[ASRChunk, float]]] = {segment.id: [] for segment in segments}
    for chunk in chunks:
        best_segment: Segment | None = None
        best_overlap = 0.0
        for segment in segments:
            overlap = _overlap_sec(segment, chunk)
            if overlap > best_overlap:
                best_segment = segment
                best_overlap = overlap
        if best_segment is not None and best_overlap > 0:
            assigned[best_segment.id].append((chunk, best_overlap))

    mapped: dict[str, SourceScript | None] = {}
    for segment in segments:
        overlaps = assigned[segment.id]
        if not overlaps:
            mapped[segment.id] = None
            continue
        overlaps.sort(key=lambda item: item[0].start)
        text = " ".join(chunk.text.strip() for chunk, _ in overlaps if chunk.text.strip()).strip()
        if not text:
            mapped[segment.id] = None
            continue
        language = next((chunk.language for chunk, _ in overlaps if chunk.language), fallback_language)
        mapped[segment.id] = SourceScript(
            text=text,
            language=language,
            confidence=_weighted_confidence(overlaps),
            backend=backend,
            start=segment.start,
            end=segment.end,
        )
    return mapped


def create_asr_backend(kind: str, config: Mapping[str, Any] | None = None) -> ASRBackend:
    config = config or {}
    normalized = kind.replace("-", "_")
    if normalized == "mock":
        from .mock import MockASRBackend

        return MockASRBackend(language=str(config.get("language", "ja")))
    if normalized == "faster_whisper":
        from .faster_whisper import FasterWhisperASRBackend

        return FasterWhisperASRBackend(
            model_id=str(config.get("model_id", "Systran/faster-whisper-large-v3")),
            language=str(config.get("language", "ja")),
            local_files_only=bool(config.get("local_files_only", True)),
            device=str(config.get("device", "auto")),
            compute_type=str(config.get("compute_type", "default")),
            batched_inference=bool(config.get("batched_inference", False)),
            batch_size=int(config.get("batch_size", 8)),
            beam_size=int(config.get("beam_size", 5)),
            best_of=int(config.get("best_of", 5)),
            condition_on_previous_text=bool(config.get("condition_on_previous_text", False)),
            vad_filter=bool(config.get("vad_filter", True)),
            vad_parameters=dict(config.get("vad_parameters") or {}),
            word_timestamps=bool(config.get("word_timestamps", False)),
            hallucination_silence_threshold=config.get("hallucination_silence_threshold"),
            initial_prompt=str(config.get("initial_prompt") or "") or None,
            hotwords=str(config.get("hotwords") or "") or None,
        )
    if normalized == "qwen_asr":
        from .qwen_asr import QwenASRBackend

        return QwenASRBackend(
            model_id=str(config.get("qwen_model_id") or "Qwen/Qwen3-ASR-1.7B"),
            language=str(config.get("language", "ja")),
            local_files_only=bool(config.get("local_files_only", True)),
            forced_aligner_model_id=(
                str(config["qwen_forced_aligner_model_id"])
                if config.get("qwen_forced_aligner_model_id")
                else None
            ),
            device_map=str(config.get("qwen_device_map", "cuda:0")),
            dtype=str(config.get("qwen_dtype", "bfloat16")),
            return_timestamps=bool(config.get("qwen_return_timestamps", True)),
            context=str(config.get("qwen_context", "")),
            max_inference_batch_size=int(config.get("qwen_max_inference_batch_size", 8)),
            max_new_tokens=int(config.get("qwen_max_new_tokens", 4096)),
        )
    raise ASRUnavailableError(f"Unsupported ASR backend: {kind}")
