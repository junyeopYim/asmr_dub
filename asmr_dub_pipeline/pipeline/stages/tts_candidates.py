from __future__ import annotations

import copy
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from asmr_dub_pipeline.audio.features import load_audio, write_audio
from asmr_dub_pipeline.pipeline.artifacts import (
    file_fingerprint,
    make_generation_id,
    make_script_generation_id,
    make_selected_tts_generation_id,
    stable_hash,
)
from asmr_dub_pipeline.pipeline.context import PipelineContext
from asmr_dub_pipeline.pipeline.manifest_io import save_manifest, write_json_atomic
from asmr_dub_pipeline.pipeline.stage_readiness import (
    NON_BLOCKING_SYNTH_SEGMENT_STATUSES,
    synth_ready_for_downstream,
)
from asmr_dub_pipeline.pipeline.stages.common import (
    _load_config_into_manifest,
    _log_stage_complete,
    _log_stage_start,
    _segment_counts,
)
from asmr_dub_pipeline.pipeline.stages.synth_gpt_sovits import run_synth_stage
from asmr_dub_pipeline.pipeline.stages.synth_qwen import run_synth_qwen_stage
from asmr_dub_pipeline.pipeline.state import mark_stage
from asmr_dub_pipeline.schemas import PipelineManifest, Segment, TTSMetadata
from asmr_dub_pipeline.schemas import TTSCandidate as LegacyTTSCandidate
from asmr_dub_pipeline.tts.candidate_store import CandidateStore
from asmr_dub_pipeline.tts.failure_taxonomy import classify_tts_hard_failed_segment
from asmr_dub_pipeline.tts.router import route_segment_tts
from asmr_dub_pipeline.tts.scoring import segment_has_numeric_sequence
from asmr_dub_pipeline.tts.selector import select_tts_candidate
from asmr_dub_pipeline.tts.types import CandidateScore, TTSCandidate, TTSRoute


def _common_backend_name(backend: str) -> str:
    normalized = backend.strip().lower().replace("-", "_")
    if normalized == "gpt_sovits_countdown_renderer":
        return "gpt_sovits"
    return normalized


def _legacy_backend_name(backend: str) -> str:
    return backend.replace("_", "-")


def _candidate_generation_id(
    *,
    segment: Segment,
    backend: str,
    candidate_id: str,
    wav_path: str,
    payload: dict[str, Any],
) -> str:
    return make_generation_id(
        "tts-candidate",
        {
            "segment_id": segment.id,
            "backend": backend,
            "candidate_id": candidate_id,
            "wav_path": wav_path,
            "script": segment.script.model_dump(mode="json") if segment.script else None,
            "payload": payload,
        },
    )


def _from_legacy_candidate(
    segment: Segment,
    candidate: LegacyTTSCandidate,
    *,
    backend_override: str | None = None,
) -> TTSCandidate:
    backend = _common_backend_name(backend_override or candidate.backend)
    candidate_id = candidate.candidate_id or (
        f"{backend}_cand_{candidate.candidate_index:02d}_attempt_{candidate.attempt:02d}"
    )
    payload = copy.deepcopy(candidate.payload)
    payload.setdefault("legacy_candidate", candidate.model_dump(mode="json"))
    generation_id = candidate.generation_id or _candidate_generation_id(
        segment=segment,
        backend=backend,
        candidate_id=candidate_id,
        wav_path=candidate.output_path,
        payload=payload,
    )
    return TTSCandidate(
        segment_id=segment.id,
        candidate_id=candidate_id,
        backend=backend,  # type: ignore[arg-type]
        wav_path=candidate.output_path,
        metadata_path=candidate.metadata_path or "",
        duration_sec=candidate.duration_sec,
        input_hash=candidate.input_hash or stable_hash(segment.script or {}),
        backend_config_hash=candidate.backend_config_hash or stable_hash({"backend": backend}),
        attempt=candidate.attempt,
        payload=payload,
        generation_id=generation_id,
        input_script_generation_id=make_script_generation_id(segment.script),
        input_script_hash=stable_hash(segment.script or {}),
        route_id=candidate.route_id,
        pool_generation_id=candidate.pool_generation_id,
        source_wav_sha256=candidate.source_wav_sha256 or candidate.wav_sha256,
        wav_sha256=candidate.wav_sha256 or candidate.source_wav_sha256,
        source_wav_size_bytes=candidate.source_wav_size_bytes or candidate.wav_size_bytes,
        wav_size_bytes=candidate.wav_size_bytes or candidate.source_wav_size_bytes,
        source_wav_mtime_ns=candidate.source_wav_mtime_ns or candidate.wav_mtime_ns,
        wav_mtime_ns=candidate.wav_mtime_ns or candidate.source_wav_mtime_ns,
    )


def _qwen_candidates_from_analysis(segment: Segment) -> list[TTSCandidate]:
    summary = segment.analysis.get("qwen_tts")
    if not isinstance(summary, dict):
        return []
    raw_candidates = summary.get("candidates")
    if not isinstance(raw_candidates, list):
        return []
    candidates: list[TTSCandidate] = []
    for index, raw in enumerate(raw_candidates):
        if not isinstance(raw, dict):
            continue
        legacy = LegacyTTSCandidate.model_validate(raw)
        if legacy.error:
            continue
        common = _from_legacy_candidate(segment, legacy, backend_override="qwen-tts")
        if common.candidate_id.startswith("qwen_tts_"):
            common = common.model_copy(update={"candidate_id": f"qwen_tts_cand_{index:02d}"})
        candidates.append(common)
    return candidates


def _gsv_candidates_from_tts(segment: Segment) -> list[TTSCandidate]:
    if segment.tts is None or segment.tts.backend not in {"gpt-sovits", "mock", "gpt-sovits-countdown-renderer"}:
        return []
    return [
        _from_legacy_candidate(
            segment,
            candidate,
            backend_override="mock" if segment.tts.backend == "mock" else "gpt-sovits",
        )
        for candidate in segment.tts.candidates
        if not candidate.error
    ]


def collect_segment_candidates(segment: Segment) -> list[TTSCandidate]:
    candidates = [*_gsv_candidates_from_tts(segment), *_qwen_candidates_from_analysis(segment)]
    deduped: dict[tuple[str, str], TTSCandidate] = {}
    for candidate in candidates:
        deduped[(candidate.backend, candidate.candidate_id)] = candidate
    return list(deduped.values())


def _annotate_candidate_identity(
    candidate: TTSCandidate,
    *,
    segment: Segment,
    route: TTSRoute | None,
    pool_generation_id: str | None,
) -> TTSCandidate:
    script_generation_id = make_script_generation_id(segment.script)
    script_hash = stable_hash(segment.script or {})
    fingerprint = file_fingerprint(candidate.wav_path)
    return candidate.model_copy(
        update={
            "input_script_generation_id": script_generation_id,
            "input_script_hash": script_hash,
            "route_id": route.route_id if route else candidate.route_id,
            "pool_generation_id": pool_generation_id or candidate.pool_generation_id,
            "source_wav_sha256": fingerprint["sha256"] or candidate.source_wav_sha256,
            "wav_sha256": fingerprint["sha256"] or candidate.wav_sha256,
            "source_wav_size_bytes": fingerprint["size_bytes"] or candidate.source_wav_size_bytes,
            "wav_size_bytes": fingerprint["size_bytes"] or candidate.wav_size_bytes,
            "source_wav_mtime_ns": fingerprint["mtime_ns"] or candidate.source_wav_mtime_ns,
            "wav_mtime_ns": fingerprint["mtime_ns"] or candidate.wav_mtime_ns,
        }
    )


def _route_targets(
    manifest: PipelineManifest,
    segment_ids: set[str] | None,
    requested_backend: str,
) -> dict[str, TTSRoute]:
    routes: dict[str, TTSRoute] = {}
    for segment in manifest.segments:
        if segment_ids is not None and segment.id not in segment_ids:
            continue
        if not segment.script:
            continue
        route = route_segment_tts(segment, manifest.project_config, requested_backend=requested_backend)
        routes[segment.id] = route
        segment.analysis["tts_route"] = route.model_dump(mode="json")
    return routes


def run_tts_candidate_pool_stage(
    ctx: PipelineContext,
    *,
    refs_path: Path = Path("refs/refs.json"),
    confirm_rights: bool = False,
    requested_backend: str = "auto",
    gsv_url: str | None = None,
    gpt_weights_path: str | None = None,
    sovits_weights_path: str | None = None,
    use_trained_gpt: bool = False,
    auto_gsv_server: bool | None = None,
    gsv_server_command: list[str] | str | None = None,
    qwen_model_id: str | None = None,
    qwen_candidate_count: int | None = None,
    qwen_local_files_only: bool | None = None,
    only_segment_ids: set[str] | None = None,
    mock: bool = False,
) -> PipelineManifest:
    project_dir = ctx.project_dir
    manifest = ctx.reload_manifest()
    _load_config_into_manifest(project_dir, manifest)
    routes = _route_targets(manifest, only_segment_ids, requested_backend)
    _log_stage_start("tts.candidate_pool", f"segments={len(routes)} backend={requested_backend}")
    if not routes:
        mark_stage(manifest, "tts.candidate_pool", "skipped", target_segments=[])
        save_manifest(project_dir, manifest)
        return ctx.update_manifest(manifest)
    store = CandidateStore(project_dir)
    clear_results: dict[str, dict[str, Any]] = {}
    for segment_id in sorted(routes):
        clear_result = store.clear_segment(segment_id)
        clear_result["clear_reason"] = "new_candidate_pool_run"
        clear_results[segment_id] = clear_result
    for segment in manifest.segments:
        clear_result = clear_results.get(segment.id)
        if clear_result is not None:
            segment.analysis["tts_candidate_pool_clear"] = clear_result
    save_manifest(project_dir, manifest)
    gsv_ids = {segment_id for segment_id, route in routes.items() if "gpt_sovits" in route.backends or "mock" in route.backends}
    qwen_ids = {segment_id for segment_id, route in routes.items() if "qwen_tts" in route.backends}
    if gsv_ids:
        run_synth_stage(
            ctx,
            gsv_url,
            refs_path,
            mock=mock or requested_backend == "mock",
            confirm_rights=confirm_rights,
            gpt_weights_path=gpt_weights_path,
            sovits_weights_path=sovits_weights_path,
            auto_gsv_server=auto_gsv_server,
            gsv_server_command=gsv_server_command,
            use_trained_gpt=use_trained_gpt,
            only_segment_ids=gsv_ids,
            render_countdowns=False,
        )
    if qwen_ids:
        run_synth_qwen_stage(
            ctx,
            refs_path,
            confirm_rights=confirm_rights,
            model_id=qwen_model_id,
            candidate_count=qwen_candidate_count,
            promote=False,
            local_files_only=qwen_local_files_only,
            only_segment_ids=qwen_ids,
        )
    manifest = ctx.reload_manifest()
    _load_config_into_manifest(project_dir, manifest)
    pool_manifest: dict[str, Any] = {"segments": [], "requested_backend": requested_backend}
    saved_count = 0
    for segment in manifest.segments:
        route = routes.get(segment.id)
        if route is None:
            continue
        candidates = collect_segment_candidates(segment)
        pool_generation_id = make_generation_id(
            "tts-pool",
            {
                "segment_id": segment.id,
                "route": route.model_dump(mode="json"),
                "candidate_generation_ids": [candidate.generation_id for candidate in candidates],
            },
        )
        candidates = [
            _annotate_candidate_identity(
                candidate,
                segment=segment,
                route=route,
                pool_generation_id=pool_generation_id,
            )
            for candidate in candidates
        ]
        for candidate in candidates:
            store.save_candidate(candidate)
        saved_count += len(candidates)
        clear_result = clear_results.get(segment.id, {})
        segment.analysis["tts_candidate_pool"] = {
            "status": "completed" if candidates else "empty",
            "route": route.model_dump(mode="json"),
            "candidate_count": len(candidates),
            "candidate_ids": [candidate.candidate_id for candidate in candidates],
            "generation_id": pool_generation_id,
            "cleared_candidate_metadata": bool(clear_result.get("cleared_candidate_metadata")),
            "cleared_selected_metadata": bool(clear_result.get("cleared_selected_metadata")),
            "clear_reason": clear_result.get("clear_reason"),
        }
        pool_manifest["segments"].append(segment.analysis["tts_candidate_pool"])
    out_path = project_dir / "work" / "tts" / "candidate_pool_manifest.json"
    write_json_atomic(out_path, pool_manifest)
    manifest.artifacts["tts_candidate_pool"] = str(out_path)
    mark_stage(
        manifest,
        "tts.candidate_pool",
        "completed",
        requested_backend=requested_backend,
        target_segments=sorted(routes),
        candidate_count=saved_count,
        candidate_pool_manifest=str(out_path),
        segment_counts=_segment_counts(manifest),
    )
    save_manifest(project_dir, manifest)
    _log_stage_complete("tts.candidate_pool", manifest, f"candidates={saved_count}")
    return ctx.update_manifest(manifest)


def _legacy_candidates_from_common(
    candidates: list[TTSCandidate],
    selected_candidate_id: str,
) -> list[LegacyTTSCandidate]:
    legacy: list[LegacyTTSCandidate] = []
    for index, candidate in enumerate(candidates):
        legacy_payload = copy.deepcopy(candidate.payload)
        legacy_payload.setdefault("candidate_pool", candidate.model_dump(mode="json"))
        legacy.append(
            LegacyTTSCandidate(
                candidate_index=index,
                seed=int(legacy_payload.get("seed") or index),
                payload=legacy_payload,
                output_path=candidate.wav_path,
                duration_sec=candidate.duration_sec,
                backend=_legacy_backend_name(candidate.backend),  # type: ignore[arg-type]
                selected=candidate.candidate_id == selected_candidate_id,
                duration_ratio=(
                    legacy_payload.get("duration_ratio")
                    if isinstance(legacy_payload.get("duration_ratio"), (int, float))
                    else None
                ),
                duration_gate=str(legacy_payload.get("duration_gate") or "unknown"),  # type: ignore[arg-type]
                acceptable_for_mix=candidate.candidate_id == selected_candidate_id,
                selection_score=None,
                selection_reason="candidate_pool_selector",
                candidate_id=candidate.candidate_id,
                metadata_path=candidate.metadata_path,
                input_hash=candidate.input_hash,
                backend_config_hash=candidate.backend_config_hash,
                attempt=candidate.attempt,
                generation_id=candidate.generation_id,
                input_script_generation_id=candidate.input_script_generation_id,
                input_script_hash=candidate.input_script_hash,
                route_id=candidate.route_id,
                pool_generation_id=candidate.pool_generation_id,
                source_wav_sha256=candidate.source_wav_sha256,
                wav_sha256=candidate.wav_sha256,
                source_wav_size_bytes=candidate.source_wav_size_bytes,
                wav_size_bytes=candidate.wav_size_bytes,
                source_wav_mtime_ns=candidate.source_wav_mtime_ns,
                wav_mtime_ns=candidate.wav_mtime_ns,
            )
        )
    return legacy


def _dedupe_reason_codes(*groups: list[str]) -> list[str]:
    values: list[str] = []
    for group in groups:
        values.extend(str(item) for item in group if item)
    return list(dict.fromkeys(values))


def _apply_texture_or_micro_resolution(
    segment: Segment,
    taxonomy: dict[str, Any],
) -> dict[str, Any] | None:
    class_name = str(taxonomy.get("class") or "")
    if class_name == "micro_absorb":
        segment.status = "absorbed"
        segment.keep_original_texture = True
        action = "absorbed"
    elif class_name in {"missing_script_texture", "non_speech_texture"}:
        segment.status = "non_speech_texture"
        segment.keep_original_texture = True
        action = "non_speech_texture"
    else:
        return None
    segment.tts = None
    segment.rvc = None
    segment.qc = None
    segment.mix = {}
    payload = {
        "status": "bypassed",
        "action": action,
        "terminal_reason": class_name,
        "failure_taxonomy": taxonomy,
    }
    segment.analysis["tts_texture_or_micro_resolution"] = payload
    segment.analysis["tts_failure_taxonomy"] = taxonomy
    segment.analysis["tts_selection"] = payload
    return payload


def _score_map(scores: list[CandidateScore]) -> dict[str, CandidateScore]:
    return {score.candidate_id: score for score in scores}


def _candidate_content_qc_allows_duration_rescue(candidate: TTSCandidate) -> bool:
    payload = candidate.payload or {}
    preflight = payload.get("preflight") or payload.get("text_preflight")
    if isinstance(preflight, dict) and preflight.get("blocked") is True:
        return False
    for key in ("pronunciation_qc", "semantic_qc", "numeric_sequence_qc", "numeric_qc"):
        qc = payload.get(key)
        if isinstance(qc, dict) and str(qc.get("gate") or "").strip().lower() == "fail":
            return False
    asr_backcheck = payload.get("ko_asr_backcheck")
    if isinstance(asr_backcheck, dict) and str(asr_backcheck.get("severity") or "").strip().lower() == "severe":
        return False
    return True


def _short_reaction_segment(segment: Segment) -> bool:
    text = segment.script.tts_text if segment.script else ""
    hangul_count = sum(1 for char in text if "\uac00" <= char <= "\ud7a3")
    return segment.duration <= 1.6 and hangul_count <= 8


def _duration_rescue_max_stretch_ratio(segment: Segment) -> float:
    return 1.35 if _short_reaction_segment(segment) else 1.25


def _resample_audio_to_frames(data: np.ndarray, target_frames: int) -> np.ndarray:
    if target_frames <= 0:
        return data
    if len(data) == target_frames:
        return data.astype(np.float32, copy=True)
    if len(data) <= 1:
        return np.repeat(data[:1], target_frames, axis=0).astype(np.float32)
    source_x = np.linspace(0.0, 1.0, num=len(data), dtype=np.float64)
    target_x = np.linspace(0.0, 1.0, num=target_frames, dtype=np.float64)
    channels = [np.interp(target_x, source_x, data[:, channel]) for channel in range(data.shape[1])]
    return np.stack(channels, axis=1).astype(np.float32)


def _write_duration_rescue_audio(
    source_path: Path,
    output_path: Path,
    *,
    source_duration_sec: float,
    target_duration_sec: float,
    max_stretch_ratio: float,
) -> tuple[str, float]:
    data, sample_rate = load_audio(source_path)
    target_frames = max(1, int(round(target_duration_sec * sample_rate)))
    stretch_ratio = target_duration_sec / max(source_duration_sec, 1e-6)
    if stretch_ratio <= max_stretch_ratio:
        rescued = _resample_audio_to_frames(data, target_frames)
        method = "time_stretch"
    elif source_duration_sec < target_duration_sec:
        pad_frames = max(0, target_frames - len(data))
        rescued = np.concatenate([data, np.zeros((pad_frames, data.shape[1]), dtype=np.float32)], axis=0)
        method = "tail_pad"
    else:
        rescued = _resample_audio_to_frames(data, target_frames)
        method = "time_compress"
    write_audio(output_path, rescued, sample_rate)
    return method, len(rescued) / float(sample_rate)


def _create_qwen_duration_rescue_candidates(
    *,
    project_dir: Path,
    store: CandidateStore,
    segment: Segment,
    candidates: list[TTSCandidate],
    scores: list[CandidateScore],
    route: TTSRoute | None,
    pool_generation_id: str | None,
) -> tuple[list[TTSCandidate], dict[str, Any] | None]:
    if segment_has_numeric_sequence(segment):
        return [], None
    scores_by_id = _score_map(scores)
    qwen_candidates = [candidate for candidate in candidates if candidate.backend == "qwen_tts"]
    if not qwen_candidates:
        return [], None
    duration_only_candidates = [
        candidate
        for candidate in qwen_candidates
        if scores_by_id.get(candidate.candidate_id)
        and set(scores_by_id[candidate.candidate_id].hard_fail_reasons) == {"duration_tolerance_exceeded"}
        and _candidate_content_qc_allows_duration_rescue(candidate)
    ]
    if len(duration_only_candidates) != len(qwen_candidates):
        return [], None
    source = min(
        duration_only_candidates,
        key=lambda candidate: abs((candidate.duration_sec or 0.0) - segment.duration),
    )
    if not source.duration_sec or source.duration_sec <= 0:
        return [], None
    source_path = Path(source.wav_path)
    if not source_path.exists():
        return [], None
    rescue_dir = project_dir / "work" / "tts" / "duration_rescue"
    candidate_id = f"{source.candidate_id}_duration_rescue"
    output_path = rescue_dir / f"{segment.id}_{candidate_id}.wav"
    max_stretch_ratio = _duration_rescue_max_stretch_ratio(segment)
    method, rescued_duration = _write_duration_rescue_audio(
        source_path,
        output_path,
        source_duration_sec=float(source.duration_sec),
        target_duration_sec=float(segment.duration),
        max_stretch_ratio=max_stretch_ratio,
    )
    payload = copy.deepcopy(source.payload)
    rescue_payload = {
        "kind": "qwen_duration_rescue",
        "method": method,
        "source_candidate_id": source.candidate_id,
        "source_candidate_generation_id": source.generation_id,
        "source_wav_path": source.wav_path,
        "source_duration_sec": round(float(source.duration_sec), 6),
        "target_duration_sec": round(float(segment.duration), 6),
        "rescued_duration_sec": round(float(rescued_duration), 6),
        "max_stretch_ratio": max_stretch_ratio,
        "route_reason_codes": ["duration_rescue_after_qwen_duration_mismatch"],
    }
    payload["duration_rescue"] = rescue_payload
    payload["derived_from"] = {
        "candidate_id": source.candidate_id,
        "candidate_generation_id": source.generation_id,
        "backend": source.backend,
        "wav_path": source.wav_path,
    }
    generation_id = make_generation_id(
        "tts-duration-rescue",
        {
            "segment_id": segment.id,
            "source_candidate_generation_id": source.generation_id,
            "method": method,
            "target_duration_sec": round(float(segment.duration), 6),
        },
    )
    rescued = source.model_copy(
        update={
            "candidate_id": candidate_id,
            "wav_path": str(output_path),
            "metadata_path": "",
            "duration_sec": round(float(rescued_duration), 6),
            "payload": payload,
            "generation_id": generation_id,
            "route_id": route.route_id if route else source.route_id,
            "pool_generation_id": pool_generation_id or source.pool_generation_id,
            "source_wav_sha256": None,
            "wav_sha256": None,
            "source_wav_size_bytes": None,
            "wav_size_bytes": None,
            "source_wav_mtime_ns": None,
            "wav_mtime_ns": None,
        }
    )
    store.save_candidate(rescued)
    summary = {
        "created": True,
        "candidate_id": candidate_id,
        "source_candidate_id": source.candidate_id,
        "method": method,
        "target_duration_sec": round(float(segment.duration), 6),
        "rescued_duration_sec": round(float(rescued_duration), 6),
        "route_reason_codes": ["duration_rescue_after_qwen_duration_mismatch"],
    }
    return [rescued], summary


def run_tts_select_stage(
    ctx: PipelineContext,
    *,
    only_segment_ids: set[str] | None = None,
    force: bool = False,
) -> PipelineManifest:
    project_dir = ctx.project_dir
    manifest = ctx.reload_manifest()
    _load_config_into_manifest(project_dir, manifest)
    store = CandidateStore(project_dir)
    selected_segments: list[str] = []
    hard_failed_segments: list[str] = []
    texture_bypassed_segments: list[str] = []
    absorbed_segments: list[str] = []
    late_qwen_scheduled_segments: list[str] = []
    duration_rescue_scheduled_segments: list[str] = []
    hard_fail_reason_counts: Counter[str] = Counter()
    failure_taxonomy_counts: Counter[str] = Counter()
    scoped_segment_count = sum(
        1 for segment in manifest.segments if only_segment_ids is None or segment.id in only_segment_ids
    )
    for segment in manifest.segments:
        if only_segment_ids is not None and segment.id not in only_segment_ids:
            continue
        initial_taxonomy = classify_tts_hard_failed_segment(segment)
        initial_resolution = _apply_texture_or_micro_resolution(segment, initial_taxonomy)
        if initial_resolution is not None:
            failure_taxonomy_counts.update([str(initial_taxonomy["class"])])
            if initial_resolution["action"] == "absorbed":
                absorbed_segments.append(segment.id)
            else:
                texture_bypassed_segments.append(segment.id)
            store.clear_selected(segment.id)
            continue
        route_payload = segment.analysis.get("tts_route")
        route = TTSRoute.model_validate(route_payload) if isinstance(route_payload, dict) else None
        script_generation_id = make_script_generation_id(segment.script)
        script_hash = stable_hash(segment.script or {})
        pool_payload = segment.analysis.get("tts_candidate_pool")
        pool_generation_id = None
        if isinstance(pool_payload, dict):
            pool_generation_id = str(pool_payload.get("generation_id") or "") or None
        all_stored_candidates = store.load_segment_candidates(
            segment.id,
            expected_script_generation_id=script_generation_id,
            expected_script_hash=script_hash,
            expected_route_id=route.route_id if route else None,
            expected_pool_generation_id=pool_generation_id,
            discard_stale=False,
        )
        stale_candidates = [
            candidate
            for candidate in all_stored_candidates
            if isinstance(candidate.payload.get("stale_filter"), dict)
            and candidate.payload["stale_filter"].get("is_stale")
        ]
        stale_reasons: dict[str, list[str]] = {
            candidate.candidate_id: list(candidate.payload["stale_filter"].get("reasons") or [])
            for candidate in stale_candidates
            if isinstance(candidate.payload.get("stale_filter"), dict)
        }
        candidates = store.load_segment_candidates(
            segment.id,
            expected_script_generation_id=script_generation_id,
            expected_script_hash=script_hash,
            expected_route_id=route.route_id if route else None,
            expected_pool_generation_id=pool_generation_id,
            discard_stale=True,
        )
        legacy_fallback_used = False
        if not candidates:
            candidates = collect_segment_candidates(segment)
            if candidates:
                legacy_fallback_used = True
                candidates = [
                    _annotate_candidate_identity(
                        candidate,
                        segment=segment,
                        route=route,
                        pool_generation_id=pool_generation_id,
                    )
                    for candidate in candidates
                ]
            for candidate in candidates:
                store.save_candidate(candidate)
        if not candidates and stale_candidates:
            store.clear_selected(segment.id)
            segment.tts = None
            segment.rvc = None
            segment.qc = None
            segment.mix = {}
            segment.status = "needs_manual_review"
            hard_fail_reason_counts.update(["all_candidates_stale"])
            segment.analysis["tts_selection"] = {
                "status": "manual_review",
                "terminal_reason": "all_candidates_stale",
                "scores": [],
                "hard_fail_reasons": ["all_candidates_stale"],
                "stale_candidate_filtered_count": len(stale_candidates),
                "stale_candidate_filter_reasons": stale_reasons,
                "expected_script_generation_id": script_generation_id,
                "expected_script_hash": script_hash,
                "expected_route_id": route.route_id if route else None,
                "expected_pool_generation_id": pool_generation_id,
            }
            taxonomy = classify_tts_hard_failed_segment(segment)
            segment.analysis["tts_failure_taxonomy"] = taxonomy
            segment.analysis["tts_selection"]["failure_taxonomy"] = taxonomy
            failure_taxonomy_counts.update([str(taxonomy["class"])])
            hard_failed_segments.append(segment.id)
            continue
        result = select_tts_candidate(segment, candidates, manifest.project_config, route=route)
        selected = result.selected
        duration_rescue_summary: dict[str, Any] | None = None
        extra_route_reason_codes: list[str] = []
        if selected is None:
            segment_hard_fail_reasons = sorted(
                {
                    reason
                    for score in result.scores
                    for reason in score.hard_fail_reasons
                }
            )
            selection_failure_payload = {
                "status": "unresolved",
                "terminal_reason": result.terminal_reason,
                "scores": [score.model_dump(mode="json") for score in result.scores],
                "hard_fail_reasons": segment_hard_fail_reasons,
                "stale_candidate_filtered_count": len(stale_candidates),
                "stale_candidate_filter_reasons": stale_reasons,
                "expected_script_generation_id": script_generation_id,
                "expected_script_hash": script_hash,
                "expected_route_id": route.route_id if route else None,
                "expected_pool_generation_id": pool_generation_id,
            }
            segment.analysis["tts_selection"] = selection_failure_payload
            taxonomy = classify_tts_hard_failed_segment(segment)
            segment.analysis["tts_failure_taxonomy"] = taxonomy
            segment.analysis["tts_selection"]["failure_taxonomy"] = taxonomy
            failure_taxonomy_counts.update([str(taxonomy["class"])])
            if taxonomy["class"] == "qwen_duration_mismatch":
                rescue_candidates, duration_rescue_summary = _create_qwen_duration_rescue_candidates(
                    project_dir=project_dir,
                    store=store,
                    segment=segment,
                    candidates=candidates,
                    scores=result.scores,
                    route=route,
                    pool_generation_id=pool_generation_id,
                )
                if rescue_candidates:
                    duration_rescue_scheduled_segments.append(segment.id)
                    extra_route_reason_codes.append("duration_rescue_after_qwen_duration_mismatch")
                    candidates = [*candidates, *rescue_candidates]
                    result = select_tts_candidate(segment, candidates, manifest.project_config, route=route)
                    selected = result.selected
            if selected is None:
                resolution = _apply_texture_or_micro_resolution(segment, taxonomy)
                if resolution is not None:
                    if resolution["action"] == "absorbed":
                        absorbed_segments.append(segment.id)
                    else:
                        texture_bypassed_segments.append(segment.id)
                    store.clear_selected(segment.id)
                    continue
                store.clear_selected(segment.id)
                segment.tts = None
                segment.rvc = None
                segment.qc = None
                segment.mix = {}
                if taxonomy["class"] == "gsv_only_needs_late_qwen":
                    segment.status = "needs_regeneration"
                    extra_route_reason_codes.append("late_qwen_after_gsv_hard_fail")
                    late_qwen_scheduled_segments.append(segment.id)
                    segment.analysis["ko_qc_repair_plan"] = {
                        "action": "fallback_tts_qwen",
                        "root_cause": "gsv_only_all_candidates_hard_failed",
                        "route": "late_qwen_after_gsv_hard_fail",
                        "terminal_manual": False,
                        "issues": segment_hard_fail_reasons or ["all_candidates_hard_failed"],
                        "source": "tts_failure_taxonomy",
                    }
                    segment.analysis["tts_selection"] = {
                        **selection_failure_payload,
                        "status": "scheduled_repair",
                        "suggested_action": "late_qwen",
                        "failure_taxonomy": taxonomy,
                        "route_reason_codes": _dedupe_reason_codes(result.route_reason_codes, extra_route_reason_codes),
                    }
                    continue
                segment.status = "needs_manual_review"
                hard_fail_reason_counts.update(segment_hard_fail_reasons)
                segment.analysis["tts_selection"] = {
                    **selection_failure_payload,
                    "status": "manual_review",
                    "failure_taxonomy": taxonomy,
                    "route_reason_codes": _dedupe_reason_codes(result.route_reason_codes, extra_route_reason_codes),
                }
                hard_failed_segments.append(segment.id)
                continue
        final_path = project_dir / "work" / "tts" / f"{segment.id}_final.wav"
        source_path = Path(selected.wav_path)
        if force or not final_path.exists() or source_path.resolve() != final_path.resolve():
            final_path.parent.mkdir(parents=True, exist_ok=True)
            if source_path.resolve() != final_path.resolve():
                shutil.copy2(source_path, final_path)
        source_fingerprint = file_fingerprint(source_path)
        final_fingerprint = file_fingerprint(final_path)
        selected_generation_id = make_selected_tts_generation_id(
            segment_id=segment.id,
            candidate_id=selected.candidate_id,
            wav_path=str(final_path),
            input_script_generation_id=script_generation_id,
            candidate_generation_id=selected.generation_id,
            backend=selected.backend,
            source_wav_path=selected.wav_path,
            source_wav_sha256=source_fingerprint["sha256"],
            final_wav_path=str(final_path),
            final_wav_sha256=final_fingerprint["sha256"],
            input_script_hash=script_hash,
        )
        scores_by_id = {score.candidate_id: score for score in result.scores}
        selected_score = scores_by_id.get(selected.candidate_id)
        selection_payload = {
            "segment_id": segment.id,
            "selected_candidate_id": selected.candidate_id,
            "backend": selected.backend,
            "wav_path": str(final_path),
            "source_wav_path": selected.wav_path,
            "selected_candidate_generation_id": selected.generation_id,
            "selected_tts_generation_id": selected_generation_id,
            "input_script_generation_id": script_generation_id,
            "input_script_hash": script_hash,
            "script_hash": script_hash,
            "route_id": route.route_id if route else None,
            "pool_generation_id": pool_generation_id,
            "source_wav_sha256": source_fingerprint["sha256"],
            "source_wav_size_bytes": source_fingerprint["size_bytes"],
            "source_wav_mtime_ns": source_fingerprint["mtime_ns"],
            "final_wav_sha256": final_fingerprint["sha256"],
            "final_wav_size_bytes": final_fingerprint["size_bytes"],
            "final_wav_mtime_ns": final_fingerprint["mtime_ns"],
            "score": selected_score.score if selected_score else None,
            "score_parts": selected_score.score_parts if selected_score else {},
            "scores": [score.model_dump(mode="json") for score in result.scores],
            "hard_fail_reasons": {
                score.candidate_id: score.hard_fail_reasons
                for score in result.scores
                if score.hard_fail_reasons
            },
            "route_reason_codes": _dedupe_reason_codes(result.route_reason_codes, extra_route_reason_codes),
            "legacy_candidate_fallback": legacy_fallback_used,
            "stale_candidate_filtered_count": len(stale_candidates),
            "stale_candidate_filter_reasons": stale_reasons,
        }
        if duration_rescue_summary is not None:
            selection_payload["duration_rescue"] = duration_rescue_summary
        selected_path = store.save_selected(segment.id, selection_payload)
        previous_generation_id = segment.tts.selected_tts_generation_id if segment.tts else None
        segment.tts = TTSMetadata(
            backend=_legacy_backend_name(selected.backend),  # type: ignore[arg-type]
            selected_candidate_path=str(final_path),
            selected_candidate_id=selected.candidate_id,
            selected_metadata_path=str(selected_path),
            candidate_pool_manifest_path=manifest.artifacts.get("tts_candidate_pool"),
            candidates=_legacy_candidates_from_common(candidates, selected.candidate_id),
            source_language=segment.script.source_language if segment.script else manifest.project_config.source_language,
            target_language=segment.script.target_language if segment.script else manifest.project_config.target_language,
            cross_lingual_voice_transfer=(
                (segment.script.source_language if segment.script else manifest.project_config.source_language)
                != (segment.script.target_language if segment.script else manifest.project_config.target_language)
            ),
            generation_id=selected_generation_id,
            tts_pool_generation_id=pool_generation_id,
            selected_tts_generation_id=selected_generation_id,
            input_script_generation_id=script_generation_id,
            input_script_hash=script_hash,
            route_reason_codes=selection_payload["route_reason_codes"],
            selected_candidate_generation_id=selected.generation_id,
            source_wav_sha256=source_fingerprint["sha256"],
            final_wav_sha256=final_fingerprint["sha256"],
            retry_summary={
                "candidate_pool_selector": selection_payload,
                "selected_generation_changed": previous_generation_id != selected_generation_id,
            },
        )
        segment.rvc = None
        segment.qc = None
        segment.mix = {}
        segment.status = "synthesized"
        segment.analysis["tts_selection"] = {"status": "selected", **selection_payload}
        selected_segments.append(segment.id)
    out_path = project_dir / "work" / "tts" / "selected" / "selection_manifest.json"
    non_blocking_hard_failed_segments = [
        segment_id
        for segment_id in hard_failed_segments
        if (segment := next((item for item in manifest.segments if item.id == segment_id), None)) is not None
        and segment.status in NON_BLOCKING_SYNTH_SEGMENT_STATUSES
    ]
    downstream_blocking_segments = [
        segment.id
        for segment in manifest.segments
        if segment.status not in NON_BLOCKING_SYNTH_SEGMENT_STATUSES
        and not (segment.tts and segment.tts.selected_candidate_path)
    ]
    downstream_ready = not downstream_blocking_segments
    hard_fail_reason_counts_payload = dict(sorted(hard_fail_reason_counts.items()))
    failure_taxonomy_counts_payload = dict(sorted(failure_taxonomy_counts.items()))
    raw_hard_fail_segment_count = (
        len(hard_failed_segments)
        + len(texture_bypassed_segments)
        + len(absorbed_segments)
        + len(late_qwen_scheduled_segments)
        + len(duration_rescue_scheduled_segments)
    )
    true_manual_review_count = len(hard_failed_segments)
    texture_or_absorbed_count = len(texture_bypassed_segments) + len(absorbed_segments)
    denominator = max(scoped_segment_count, 1)
    raw_hard_fail_rate = round(raw_hard_fail_segment_count / denominator, 6)
    actionable_manual_review_rate = round(true_manual_review_count / denominator, 6)
    texture_or_absorbed_rate = round(texture_or_absorbed_count / denominator, 6)
    downstream_blocking_rate = round(len(downstream_blocking_segments) / denominator, 6)
    partial_status = bool(
        hard_failed_segments
        or texture_bypassed_segments
        or absorbed_segments
        or late_qwen_scheduled_segments
        or duration_rescue_scheduled_segments
    )
    write_json_atomic(
        out_path,
        {
            "selected_segments": selected_segments,
            "hard_failed_segments": hard_failed_segments,
            "texture_bypassed_segments": texture_bypassed_segments,
            "absorbed_segments": absorbed_segments,
            "late_qwen_scheduled_segments": late_qwen_scheduled_segments,
            "duration_rescue_scheduled_segments": duration_rescue_scheduled_segments,
            "downstream_ready": downstream_ready,
            "downstream_blocking_segments": downstream_blocking_segments,
            "non_blocking_hard_failed_segments": non_blocking_hard_failed_segments,
            "selected_segment_count": len(selected_segments),
            "hard_failed_segment_count": len(hard_failed_segments),
            "true_manual_review_count": true_manual_review_count,
            "texture_bypassed_segment_count": len(texture_bypassed_segments),
            "absorbed_segment_count": len(absorbed_segments),
            "late_qwen_scheduled_count": len(late_qwen_scheduled_segments),
            "duration_rescue_scheduled_count": len(duration_rescue_scheduled_segments),
            "raw_hard_fail_segment_count": raw_hard_fail_segment_count,
            "failure_taxonomy_counts": failure_taxonomy_counts_payload,
            "hard_fail_reason_counts": hard_fail_reason_counts_payload,
            "raw_hard_fail_rate": raw_hard_fail_rate,
            "actionable_manual_review_rate": actionable_manual_review_rate,
            "texture_or_absorbed_rate": texture_or_absorbed_rate,
            "downstream_blocking_rate": downstream_blocking_rate,
            "segments": [
                {"id": segment.id, "tts_selection": segment.analysis.get("tts_selection")}
                for segment in manifest.segments
                if only_segment_ids is None or segment.id in only_segment_ids
            ],
        },
    )
    manifest.artifacts["tts_selected"] = str(out_path)
    mark_stage(
        manifest,
        "tts.select",
        "completed_with_hard_failed_candidates" if partial_status else "completed",
        selected_segments=selected_segments,
        hard_failed_segments=hard_failed_segments,
        texture_bypassed_segments=texture_bypassed_segments,
        absorbed_segments=absorbed_segments,
        late_qwen_scheduled_segments=late_qwen_scheduled_segments,
        duration_rescue_scheduled_segments=duration_rescue_scheduled_segments,
        downstream_ready=downstream_ready,
        downstream_blocking_segments=downstream_blocking_segments,
        non_blocking_hard_failed_segments=non_blocking_hard_failed_segments,
        selected_segment_count=len(selected_segments),
        hard_failed_segment_count=len(hard_failed_segments),
        true_manual_review_count=true_manual_review_count,
        texture_bypassed_segment_count=len(texture_bypassed_segments),
        absorbed_segment_count=len(absorbed_segments),
        late_qwen_scheduled_count=len(late_qwen_scheduled_segments),
        duration_rescue_scheduled_count=len(duration_rescue_scheduled_segments),
        raw_hard_fail_segment_count=raw_hard_fail_segment_count,
        failure_taxonomy_counts=failure_taxonomy_counts_payload,
        hard_fail_reason_counts=hard_fail_reason_counts_payload,
        raw_hard_fail_rate=raw_hard_fail_rate,
        actionable_manual_review_rate=actionable_manual_review_rate,
        texture_or_absorbed_rate=texture_or_absorbed_rate,
        downstream_blocking_rate=downstream_blocking_rate,
        selection_manifest=str(out_path),
        segment_counts=_segment_counts(manifest),
    )
    mark_stage(
        manifest,
        "synth",
        "completed_with_hard_failed_candidates" if partial_status else "completed",
        backend="candidate-pool",
        selected_segments=selected_segments,
        hard_failed_segments=hard_failed_segments,
        texture_bypassed_segments=texture_bypassed_segments,
        absorbed_segments=absorbed_segments,
        late_qwen_scheduled_segments=late_qwen_scheduled_segments,
        duration_rescue_scheduled_segments=duration_rescue_scheduled_segments,
        downstream_ready=downstream_ready,
        downstream_blocking_segments=downstream_blocking_segments,
        non_blocking_hard_failed_segments=non_blocking_hard_failed_segments,
        selected_segment_count=len(selected_segments),
        hard_failed_segment_count=len(hard_failed_segments),
        true_manual_review_count=true_manual_review_count,
        texture_bypassed_segment_count=len(texture_bypassed_segments),
        absorbed_segment_count=len(absorbed_segments),
        late_qwen_scheduled_count=len(late_qwen_scheduled_segments),
        duration_rescue_scheduled_count=len(duration_rescue_scheduled_segments),
        raw_hard_fail_segment_count=raw_hard_fail_segment_count,
        failure_taxonomy_counts=failure_taxonomy_counts_payload,
        hard_fail_reason_counts=hard_fail_reason_counts_payload,
        raw_hard_fail_rate=raw_hard_fail_rate,
        actionable_manual_review_rate=actionable_manual_review_rate,
        texture_or_absorbed_rate=texture_or_absorbed_rate,
        downstream_blocking_rate=downstream_blocking_rate,
        segment_counts=_segment_counts(manifest),
    )
    manifest.stage_state["synth"]["downstream_readiness"] = synth_ready_for_downstream(manifest)
    save_manifest(project_dir, manifest)
    return ctx.update_manifest(manifest)
