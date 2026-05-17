from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from asmr_dub_pipeline.pipeline.artifacts import file_fingerprint
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
        fingerprint = file_fingerprint(candidate.wav_path)
        update: dict[str, Any] = {"metadata_path": str(path)}
        if fingerprint["sha256"] is not None:
            update.setdefault("source_wav_sha256", fingerprint["sha256"])
            update.setdefault("wav_sha256", fingerprint["sha256"])
        if fingerprint["size_bytes"] is not None:
            update.setdefault("source_wav_size_bytes", fingerprint["size_bytes"])
            update.setdefault("wav_size_bytes", fingerprint["size_bytes"])
        if fingerprint["mtime_ns"] is not None:
            update.setdefault("source_wav_mtime_ns", fingerprint["mtime_ns"])
            update.setdefault("wav_mtime_ns", fingerprint["mtime_ns"])
        update = {key: value for key, value in update.items() if getattr(candidate, key, None) in {None, ""}}
        update["metadata_path"] = str(path)
        candidate = candidate.model_copy(update=update)
        write_json_atomic(path, candidate.model_dump(mode="json"))
        return path

    def clear_selected(self, segment_id: str) -> bool:
        path = self.selected_path(segment_id)
        if not path.exists():
            return False
        path.unlink()
        return True

    def clear_segment(
        self,
        segment_id: str,
        *,
        clear_selected: bool = True,
        clear_final: bool = False,
    ) -> dict[str, Any]:
        safe_segment_id = validate_file_safe_id(segment_id, "segment_id")
        segment_root = self.candidates_root / safe_segment_id
        cleared_candidate_metadata = segment_root.exists()
        if segment_root.exists():
            shutil.rmtree(segment_root)
        cleared_selected_metadata = self.clear_selected(segment_id) if clear_selected else False
        final_path = self.root / f"{safe_segment_id}_final.wav"
        cleared_final_wav = False
        if clear_final and final_path.exists():
            final_path.unlink()
            cleared_final_wav = True
        return {
            "segment_id": safe_segment_id,
            "cleared_candidate_metadata": cleared_candidate_metadata,
            "cleared_selected_metadata": cleared_selected_metadata,
            "cleared_final_wav": cleared_final_wav,
            "clear_final": clear_final,
        }

    @staticmethod
    def _stale_reasons(
        candidate: TTSCandidate,
        *,
        expected_script_generation_id: str | None,
        expected_script_hash: str | None,
        expected_route_id: str | None,
        expected_pool_generation_id: str | None,
    ) -> list[str]:
        reasons: list[str] = []
        if (
            expected_script_generation_id is not None
            and candidate.input_script_generation_id is not None
            and candidate.input_script_generation_id != expected_script_generation_id
        ):
            reasons.append("input_script_generation_id_mismatch")
        if (
            expected_script_generation_id is not None
            and candidate.input_script_generation_id is None
        ):
            reasons.append("missing_input_script_generation_id")
        if (
            expected_script_hash is not None
            and candidate.input_script_hash is not None
            and candidate.input_script_hash != expected_script_hash
        ):
            reasons.append("input_script_hash_mismatch")
        if expected_script_hash is not None and candidate.input_script_hash is None:
            reasons.append("missing_input_script_hash")
        if (
            expected_route_id is not None
            and candidate.route_id is not None
            and candidate.route_id != expected_route_id
        ):
            reasons.append("route_id_mismatch")
        if expected_route_id is not None and candidate.route_id is None:
            reasons.append("missing_route_id")
        if (
            expected_pool_generation_id is not None
            and candidate.pool_generation_id is not None
            and candidate.pool_generation_id != expected_pool_generation_id
        ):
            reasons.append("pool_generation_id_mismatch")
        if expected_pool_generation_id is not None and candidate.pool_generation_id is None:
            reasons.append("missing_pool_generation_id")
        return reasons

    def load_segment_candidates(
        self,
        segment_id: str,
        *,
        expected_script_generation_id: str | None = None,
        expected_script_hash: str | None = None,
        expected_route_id: str | None = None,
        expected_pool_generation_id: str | None = None,
        discard_stale: bool = True,
    ) -> list[TTSCandidate]:
        safe_segment_id = validate_file_safe_id(segment_id, "segment_id")
        segment_root = self.candidates_root / safe_segment_id
        if not segment_root.exists():
            return []
        candidates: list[TTSCandidate] = []
        for path in sorted(segment_root.glob("*/*.json")):
            payload = json.loads(path.read_text("utf-8"))
            candidate = TTSCandidate.model_validate(payload)
            reasons = self._stale_reasons(
                candidate,
                expected_script_generation_id=expected_script_generation_id,
                expected_script_hash=expected_script_hash,
                expected_route_id=expected_route_id,
                expected_pool_generation_id=expected_pool_generation_id,
            )
            if reasons and discard_stale:
                continue
            if reasons:
                stale_payload = dict(candidate.payload)
                stale_payload["stale_filter"] = {
                    "is_stale": True,
                    "reasons": reasons,
                    "expected_script_generation_id": expected_script_generation_id,
                    "expected_script_hash": expected_script_hash,
                    "expected_route_id": expected_route_id,
                    "expected_pool_generation_id": expected_pool_generation_id,
                }
                candidate = candidate.model_copy(update={"payload": stale_payload})
            candidates.append(candidate)
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
