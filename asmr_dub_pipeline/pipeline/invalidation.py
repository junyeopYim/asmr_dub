from __future__ import annotations

from typing import Any

from asmr_dub_pipeline.schemas import PipelineManifest, Segment, utc_now

NODE_ORDER = [
    "translate_ko",
    "korean_script",
    "tts.candidate_pool",
    "tts.select",
    "rvc",
    "qc",
    "mix",
    "export",
]

NODE_ALIASES = {
    "script": "korean_script",
    "korean-script": "korean_script",
    "translate-ko": "translate_ko",
    "synth": "tts.candidate_pool",
    "synth-qwen": "tts.candidate_pool",
    "tts": "tts.candidate_pool",
    "tts_candidate_pool": "tts.candidate_pool",
    "tts_select": "tts.select",
    "selected_tts": "tts.select",
}

STAGE_BY_NODE = {
    "translate_ko": "translate-ko",
    "korean_script": "korean-script",
    "tts.candidate_pool": "tts.candidate_pool",
    "tts.select": "tts.select",
    "rvc": "rvc",
    "qc": "qc",
    "mix": "mix",
    "export": "export",
}

ARTIFACTS_BY_NODE = {
    "tts.candidate_pool": ("tts_candidate_pool", "qwen_tts", "fish_tts", "cosyvoice_tts"),
    "tts.select": ("tts_selected",),
    "rvc": ("rvc_manifest", "rvc"),
    "qc": ("qc",),
    "mix": ("mix", "mix_manifest", "final_audio", "dialogue_stem", "source_suppressed_background"),
    "export": ("export", "export_manifest", "final_video"),
}


def normalize_node(node: str) -> str:
    return NODE_ALIASES.get(node.strip().lower().replace("-", "_"), node.strip())


def downstream_nodes(from_node: str) -> list[str]:
    node = normalize_node(from_node)
    if node not in NODE_ORDER:
        raise ValueError(f"Unsupported invalidation node: {from_node}")
    return NODE_ORDER[NODE_ORDER.index(node) :]


def _segment_by_id(manifest: PipelineManifest, segment_id: str) -> Segment:
    for segment in manifest.segments:
        if segment.id == segment_id:
            return segment
    raise ValueError(f"Unknown segment id for invalidation: {segment_id}")


def invalidate_segment(
    manifest: PipelineManifest,
    segment_id: str,
    from_node: str,
    reason: str,
) -> dict[str, Any]:
    """Invalidate segment-scoped state from a node through downstream nodes."""

    segment = _segment_by_id(manifest, segment_id)
    nodes = downstream_nodes(from_node)
    if "korean_script" in nodes:
        segment.script = None
    if (
        ("tts.candidate_pool" in nodes or "tts.select" in nodes)
        and normalize_node(from_node) in {"korean_script", "tts.candidate_pool"}
    ):
        segment.tts = None
    if "rvc" in nodes:
        segment.rvc = None
    if "qc" in nodes:
        segment.qc = None
    if "mix" in nodes or "export" in nodes:
        segment.mix = {}
    if "tts.candidate_pool" in nodes:
        segment.analysis.pop("tts_route", None)
        segment.analysis.pop("tts_candidate_pool", None)
        segment.analysis.pop("tts_candidate_pool_clear", None)
    if "tts.select" in nodes:
        segment.analysis.pop("tts_selection", None)
    if segment.status in {"ok", "synthesized", "rvc_converted", "needs_regeneration", "failed", "needs_manual_review"}:
        if normalize_node(from_node) in {"korean_script", "tts.candidate_pool"}:
            segment.status = "scripted" if segment.script is not None else "transcribed"
        elif normalize_node(from_node) == "tts.select" or normalize_node(from_node) == "rvc":
            segment.status = "synthesized" if segment.tts is not None else "scripted"

    record = {
        "segment_id": segment_id,
        "from_node": normalize_node(from_node),
        "reason": reason,
        "invalidated_nodes": [node for node in nodes if node in {"korean_script", "tts.candidate_pool", "tts.select", "rvc", "qc"}],
        "updated_at": utc_now().isoformat(),
    }
    history = segment.analysis.setdefault("invalidation", [])
    if not isinstance(history, list):
        history = []
        segment.analysis["invalidation"] = history
    history.append(record)
    segment.analysis.setdefault("_state", {})["last_invalidation"] = record
    manifest.stage_state.setdefault("segment_invalidation", {"segments": {}})
    manifest.stage_state["segment_invalidation"]["segments"][segment_id] = record
    for node in nodes:
        stage = STAGE_BY_NODE.get(node)
        if stage:
            manifest.stage_state.pop(stage, None)
        for artifact in ARTIFACTS_BY_NODE.get(node, ()):
            manifest.artifacts.pop(artifact, None)
    return record
