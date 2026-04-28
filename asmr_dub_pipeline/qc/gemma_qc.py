from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from asmr_dub_pipeline.gemma.base import GemmaBackend
from asmr_dub_pipeline.schemas import Segment


def run_gemma_qc(
    backend: GemmaBackend,
    audio_path: Path,
    target_text: str,
    segment: Segment,
    context: Mapping[str, Any],
) -> dict[str, Any]:
    return backend.qc_audio(audio_path, target_text, segment, context)
