from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from asmr_dub_pipeline.pipeline.manifest_io import write_json_atomic
from asmr_dub_pipeline.schemas import validate_file_safe_id
from asmr_dub_pipeline.tts.types import TTSCandidate


class CandidateStore:
    """Small deterministic file store for segment-scoped TTS candidates."""

    def __init__(self, project_dir: Path | str) -> None:
        self.project_dir = Path(project_dir).expanduser().resolve()
        self.root = self.project_dir / "work" / "tts"
        self.candidates_root = self.root / "candidates"
        self.selected_root = self.root / "selected"

    def candidate_metadata_path(self, segment_id: str, backend: str, candidate_id: str) -> Path:
        safe_segment_id = validate_file_safe_id(segment_id, "segment_id")
        safe_backend = validate_file_safe_id(backend, "backend")
        safe_candidate_id = validate_file_safe_id(candidate_id, "candidate_id")
        return self.candidates_root / safe_segment_id / safe_backend / f"{safe_candidate_id}.json"

    def save_candidate(self, candidate: TTSCandidate) -> Path:
        path = self.candidate_metadata_path(candidate.segment_id, candidate.backend, candidate.candidate_id)
        candidate = candidate.model_copy(update={"metadata_path": str(path)})
        write_json_atomic(path, candidate.model_dump(mode="json"))
        return path

    def load_segment_candidates(self, segment_id: str) -> list[TTSCandidate]:
        safe_segment_id = validate_file_safe_id(segment_id, "segment_id")
        segment_root = self.candidates_root / safe_segment_id
        if not segment_root.exists():
            return []
        candidates: list[TTSCandidate] = []
        for path in sorted(segment_root.glob("*/*.json")):
            payload = json.loads(path.read_text("utf-8"))
            candidates.append(TTSCandidate.model_validate(payload))
        return candidates

    def selected_path(self, segment_id: str) -> Path:
        safe_segment_id = validate_file_safe_id(segment_id, "segment_id")
        return self.selected_root / f"{safe_segment_id}.json"

    def save_selected(self, segment_id: str, payload: dict[str, Any]) -> Path:
        path = self.selected_path(segment_id)
        write_json_atomic(path, payload)
        return path

    def load_selected(self, segment_id: str) -> dict[str, Any]:
        path = self.selected_path(segment_id)
        if not path.exists():
            return {}
        payload = json.loads(path.read_text("utf-8"))
        return payload if isinstance(payload, dict) else {}
