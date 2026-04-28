from __future__ import annotations

from typing import Any

from asmr_dub_pipeline.schemas import PipelineManifest, utc_now


def mark_stage(
    manifest: PipelineManifest,
    stage: str,
    status: str,
    **metadata: Any,
) -> None:
    manifest.stage_state[stage] = {
        "status": status,
        "updated_at": utc_now().isoformat(),
        **metadata,
    }
