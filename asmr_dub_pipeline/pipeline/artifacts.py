from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from asmr_dub_pipeline.schemas import JapaneseScript, Segment


def stable_hash(payload: Any) -> str:
    """Return a deterministic short hash for JSON-compatible payloads."""

    def normalize(value: Any) -> Any:
        if isinstance(value, BaseModel):
            return value.model_dump(mode="json")
        if isinstance(value, dict):
            return {str(key): normalize(val) for key, val in sorted(value.items())}
        if isinstance(value, (list, tuple)):
            return [normalize(item) for item in value]
        return value

    data = json.dumps(normalize(payload), ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:20]


def make_generation_id(prefix: str, payload: Any) -> str:
    return f"{prefix}:{stable_hash(payload)}"


def file_sha256(path: str | Path | None) -> str | None:
    """Return a streaming sha256 for an existing file, or None if unavailable."""

    if path is None:
        return None
    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        return None
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_fingerprint(path: str | Path | None) -> dict[str, Any]:
    """Return deterministic file content and fallback metadata for manifests."""

    if path is None:
        return {"sha256": None, "size_bytes": None, "mtime_ns": None}
    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        return {"sha256": None, "size_bytes": None, "mtime_ns": None}
    stat = file_path.stat()
    return {
        "sha256": file_sha256(file_path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def make_script_generation_id(script: JapaneseScript | None) -> str:
    return make_generation_id("script", script.model_dump(mode="json") if script else {})


def make_selected_tts_generation_id(
    *,
    segment_id: str,
    candidate_id: str,
    wav_path: str,
    input_script_generation_id: str | None,
    candidate_generation_id: str | None = None,
    backend: str | None = None,
    source_wav_path: str | None = None,
    source_wav_sha256: str | None = None,
    final_wav_path: str | None = None,
    final_wav_sha256: str | None = None,
    input_script_hash: str | None = None,
) -> str:
    return make_generation_id(
        "tts",
        {
            "segment_id": segment_id,
            "candidate_id": candidate_id,
            "candidate_generation_id": candidate_generation_id,
            "backend": backend,
            "source_wav_path": source_wav_path,
            "source_wav_sha256": source_wav_sha256,
            "final_wav_path": final_wav_path or wav_path,
            "final_wav_sha256": final_wav_sha256,
            "wav_path": wav_path,
            "input_script_generation_id": input_script_generation_id,
            "input_script_hash": input_script_hash,
        },
    )


def make_rvc_generation_id(
    *,
    segment_id: str,
    output_path: str | None,
    input_selected_tts_generation_id: str | None,
    settings: dict[str, Any] | None = None,
) -> str:
    return make_generation_id(
        "rvc",
        {
            "segment_id": segment_id,
            "output_path": output_path,
            "input_selected_tts_generation_id": input_selected_tts_generation_id,
            "settings": settings or {},
        },
    )


def make_qc_generation_id(
    *,
    segment_id: str,
    input_rvc_generation_id: str | None,
    recommendation: str,
    issues: list[str],
) -> str:
    return make_generation_id(
        "qc",
        {
            "segment_id": segment_id,
            "input_rvc_generation_id": input_rvc_generation_id,
            "recommendation": recommendation,
            "issues": issues,
        },
    )


def ensure_segment_generation_ids(segment: Segment) -> None:
    """Fill missing generation ids for existing segment metadata."""

    script_generation_id = make_script_generation_id(segment.script)
    if segment.tts is not None:
        if not segment.tts.input_script_generation_id:
            segment.tts.input_script_generation_id = script_generation_id
        if not segment.tts.input_script_hash:
            segment.tts.input_script_hash = stable_hash(segment.script or {})
        selected_candidate_id = segment.tts.selected_candidate_id or "legacy_selected_tts"
        if not segment.tts.selected_tts_generation_id and segment.tts.selected_candidate_path:
            segment.tts.selected_tts_generation_id = make_selected_tts_generation_id(
                segment_id=segment.id,
                candidate_id=selected_candidate_id,
                wav_path=segment.tts.selected_candidate_path,
                input_script_generation_id=segment.tts.input_script_generation_id,
            )
        if not segment.tts.generation_id:
            segment.tts.generation_id = segment.tts.selected_tts_generation_id
    if segment.rvc is not None:
        if not segment.rvc.input_selected_tts_generation_id and segment.tts is not None:
            segment.rvc.input_selected_tts_generation_id = segment.tts.selected_tts_generation_id
        if not segment.rvc.generation_id:
            segment.rvc.generation_id = make_rvc_generation_id(
                segment_id=segment.id,
                output_path=segment.rvc.output_path,
                input_selected_tts_generation_id=segment.rvc.input_selected_tts_generation_id,
                settings=segment.rvc.settings,
            )
    if segment.qc is not None:
        if not segment.qc.input_rvc_generation_id and segment.rvc is not None:
            segment.qc.input_rvc_generation_id = segment.rvc.generation_id
        if not segment.qc.generation_id:
            segment.qc.generation_id = make_qc_generation_id(
                segment_id=segment.id,
                input_rvc_generation_id=segment.qc.input_rvc_generation_id,
                recommendation=segment.qc.recommendation,
                issues=segment.qc.issues,
            )


def verify_segment_generation_chain(
    segment: Segment,
    *,
    strict: bool = False,
    mutate: bool = True,
    target_nodes: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Check whether selected TTS, RVC, and QC generation ids match."""

    if mutate and not strict:
        ensure_segment_generation_ids(segment)
    issues: list[str] = []
    required_nodes = {str(node) for node in target_nodes or ()}
    selected_tts_generation_id = segment.tts.selected_tts_generation_id if segment.tts else None
    rvc_generation_id = segment.rvc.generation_id if segment.rvc else None
    qc_generation_id = segment.qc.generation_id if segment.qc else None
    if strict:
        if "tts.select" in required_nodes and segment.tts is None:
            issues.append("missing_tts_metadata")
        if "rvc" in required_nodes and segment.rvc is None:
            issues.append("missing_rvc_metadata")
        if "qc" in required_nodes and segment.qc is None:
            issues.append("missing_qc_metadata")
        if segment.tts is not None:
            if not segment.tts.selected_tts_generation_id:
                issues.append("missing_selected_tts_generation_id")
            if not segment.tts.generation_id:
                issues.append("missing_tts_generation_id")
        if segment.rvc is not None:
            if not segment.rvc.generation_id:
                issues.append("missing_rvc_generation_id")
            if not segment.rvc.input_selected_tts_generation_id:
                issues.append("missing_rvc_input_selected_tts_generation_id")
        if segment.qc is not None:
            if not segment.qc.generation_id:
                issues.append("missing_qc_generation_id")
            if not segment.qc.input_rvc_generation_id:
                issues.append("missing_qc_input_rvc_generation_id")
    if (
        segment.tts is not None
        and segment.tts.generation_id
        and selected_tts_generation_id
        and segment.tts.generation_id != selected_tts_generation_id
    ):
        issues.append("tts_generation_selected_tts_generation_mismatch")
    if (
        segment.rvc is not None
        and segment.rvc.input_selected_tts_generation_id
        and selected_tts_generation_id
        and segment.rvc.input_selected_tts_generation_id != selected_tts_generation_id
    ):
        issues.append("rvc_input_selected_tts_generation_mismatch")
    if (
        segment.qc is not None
        and segment.qc.input_rvc_generation_id
        and rvc_generation_id
        and segment.qc.input_rvc_generation_id != rvc_generation_id
    ):
        issues.append("qc_input_rvc_generation_mismatch")
    return {
        "ok": not issues,
        "issues": issues,
        "selected_tts_generation_id": selected_tts_generation_id,
        "rvc_generation_id": rvc_generation_id,
        "qc_generation_id": qc_generation_id,
        "strict": strict,
        "mutated": bool(mutate and not strict),
    }
