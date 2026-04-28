from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from asmr_dub_pipeline.schemas import Segment

from .base import ASRBackend, ASRChunk


class MockASRBackend(ASRBackend):
    name = "mock"

    def __init__(self, language: str = "ja") -> None:
        self.language = language

    def transcribe(self, audio_path: Path, segments: Sequence[Segment]) -> list[ASRChunk]:
        _ = audio_path
        return [
            ASRChunk(
                start=segment.start,
                end=segment.end,
                text=f"mock source script for {segment.id}",
                language=self.language,
                confidence=0.99,
            )
            for segment in segments
        ]
