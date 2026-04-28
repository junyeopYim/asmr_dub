from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from asmr_dub_pipeline.audio.features import load_audio
from asmr_dub_pipeline.schemas import Segment

from .base import GemmaUnavailableError
from .hf_client import HFGemmaBackend

RUN_GEMMA_SMOKE_ENV = "RUN_GEMMA_SMOKE"
GEMMA_AUDIO_SAMPLE_RATE = 16_000
GEMMA_AUDIO_CHANNELS = 1
GEMMA_MAX_SEGMENT_SECONDS = 30.0


def gemma_hf_smoke_enabled(env: Mapping[str, str] | None = None) -> bool:
    return (env or os.environ).get(RUN_GEMMA_SMOKE_ENV) == "1"


def run_hf_smoke(
    audio_path: Path,
    *,
    model_id: str = "google/gemma-4-E4B-it",
    local_files_only: bool = True,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Run a real HF Gemma analysis smoke only when explicitly enabled."""
    if not gemma_hf_smoke_enabled(env):
        raise GemmaUnavailableError(f"Set {RUN_GEMMA_SMOKE_ENV}=1 to run the HF Gemma smoke.")

    data, sr = load_audio(audio_path)
    duration = len(data) / sr
    if duration > GEMMA_MAX_SEGMENT_SECONDS:
        raise GemmaUnavailableError(
            f"HF Gemma smoke clips must be <= {GEMMA_MAX_SEGMENT_SECONDS:g}s; got {duration:.3f}s."
        )

    segment = Segment(
        id="gemma_hf_smoke",
        start=0.0,
        end=round(duration, 3),
        duration=round(duration, 3),
        audio_for_gemma=str(audio_path),
        audio_for_mix=str(audio_path),
    )
    backend = HFGemmaBackend(model_id=model_id, local_files_only=local_files_only)
    return backend.analyze_segment(audio_path, segment, {"smoke": True})
