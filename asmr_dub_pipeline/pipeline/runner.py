from __future__ import annotations

from pathlib import Path
from typing import Any

from asmr_dub_pipeline.pipeline.artifacts import verify_segment_generation_chain
from asmr_dub_pipeline.pipeline.context import PipelineContext
from asmr_dub_pipeline.pipeline.stages.qc import run_qc_stage
from asmr_dub_pipeline.pipeline.stages.rvc import run_rvc_stage
from asmr_dub_pipeline.pipeline.stages.tts_candidates import (
    run_tts_candidate_pool_stage,
    run_tts_select_stage,
)


def run_closure(
    ctx: PipelineContext,
    target_nodes: list[str],
    segment_ids: set[str],
    *,
    refs_path: Path = Path("refs/refs.json"),
    confirm_rights: bool = False,
    gemma_backend: str = "mock",
    tts_backend: str = "auto",
    force: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run the minimal downstream closure for segment-scoped repair/regenerate work."""

    executed_nodes: list[str] = []
    normalized_targets = list(dict.fromkeys(target_nodes))
    if "tts.candidate_pool" in normalized_targets:
        run_tts_candidate_pool_stage(
            ctx,
            refs_path=refs_path,
            confirm_rights=confirm_rights,
            requested_backend=tts_backend,
            only_segment_ids=segment_ids,
            mock=tts_backend == "mock",
            **{
                key: value
                for key, value in kwargs.items()
                if key
                in {
                    "gsv_url",
                    "gpt_weights_path",
                    "sovits_weights_path",
                    "use_trained_gpt",
                    "auto_gsv_server",
                    "gsv_server_command",
                    "qwen_model_id",
                    "qwen_candidate_count",
                    "qwen_local_files_only",
                }
            },
        )
        executed_nodes.append("tts.candidate_pool")
    if "tts.select" in normalized_targets:
        run_tts_select_stage(ctx, only_segment_ids=segment_ids, force=force)
        executed_nodes.append("tts.select")
    if "rvc" in normalized_targets:
        run_rvc_stage(
            ctx,
            confirm_rights=confirm_rights,
            force=force,
            only_segment_ids=segment_ids,
        )
        executed_nodes.append("rvc")
    if "qc" in normalized_targets:
        run_qc_stage(ctx, gemma_backend, confirm_rights=confirm_rights, only_segment_ids=segment_ids)
        executed_nodes.append("qc")
    manifest = ctx.reload_manifest()
    verification = {
        segment.id: verify_segment_generation_chain(
            segment,
            strict=True,
            mutate=False,
            target_nodes=normalized_targets,
        )
        for segment in manifest.segments
        if segment.id in segment_ids
    }
    return {
        "executed_nodes": executed_nodes,
        "segment_ids": sorted(segment_ids),
        "verified": all(item["ok"] for item in verification.values()),
        "verification": verification,
    }
