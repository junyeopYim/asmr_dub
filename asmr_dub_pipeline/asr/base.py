from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from asmr_dub_pipeline.schemas import Segment, SourceScript, StrictBaseModel


class ASRUnavailableError(RuntimeError):
    pass


class ASRChunk(StrictBaseModel):
    start: float
    end: float
    text: str
    language: str = "ja"
    confidence: float | None = None


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
            model_id=str(config.get("model_id", "mobiuslabsgmbh/faster-whisper-large-v3-turbo")),
            language=str(config.get("language", "ja")),
            local_files_only=bool(config.get("local_files_only", True)),
        )
    raise ASRUnavailableError(f"Unsupported ASR backend: {kind}")
