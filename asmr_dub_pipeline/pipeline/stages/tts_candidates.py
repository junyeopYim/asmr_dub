from __future__ import annotations

import copy
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

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
from asmr_dub_pipeline.tts.router import route_segment_tts
from asmr_dub_pipeline.tts.selector import select_tts_candidate
from asmr_dub_pipeline.tts.types import TTSCandidate, TTSRoute


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
    hard_fail_reason_counts: Counter[str] = Counter()
    for segment in manifest.segments:
        if only_segment_ids is not None and segment.id not in only_segment_ids:
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
            hard_failed_segments.append(segment.id)
            continue
        result = select_tts_candidate(segment, candidates, manifest.project_config, route=route)
        selected = result.selected
        if selected is None:
            store.clear_selected(segment.id)
            segment.tts = None
            segment.rvc = None
            segment.qc = None
            segment.mix = {}
            segment.status = "needs_manual_review"
            segment_hard_fail_reasons = sorted(
                {
                    reason
                    for score in result.scores
                    for reason in score.hard_fail_reasons
                }
            )
            hard_fail_reason_counts.update(segment_hard_fail_reasons)
            segment.analysis["tts_selection"] = {
                "status": "manual_review",
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
            "route_reason_codes": result.route_reason_codes,
            "legacy_candidate_fallback": legacy_fallback_used,
            "stale_candidate_filtered_count": len(stale_candidates),
            "stale_candidate_filter_reasons": stale_reasons,
        }
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
            route_reason_codes=result.route_reason_codes,
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
    write_json_atomic(
        out_path,
        {
            "selected_segments": selected_segments,
            "hard_failed_segments": hard_failed_segments,
            "downstream_ready": downstream_ready,
            "downstream_blocking_segments": downstream_blocking_segments,
            "non_blocking_hard_failed_segments": non_blocking_hard_failed_segments,
            "selected_segment_count": len(selected_segments),
            "hard_failed_segment_count": len(hard_failed_segments),
            "hard_fail_reason_counts": hard_fail_reason_counts_payload,
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
        "completed" if not hard_failed_segments else "completed_with_hard_failed_candidates",
        selected_segments=selected_segments,
        hard_failed_segments=hard_failed_segments,
        downstream_ready=downstream_ready,
        downstream_blocking_segments=downstream_blocking_segments,
        non_blocking_hard_failed_segments=non_blocking_hard_failed_segments,
        selected_segment_count=len(selected_segments),
        hard_failed_segment_count=len(hard_failed_segments),
        hard_fail_reason_counts=hard_fail_reason_counts_payload,
        selection_manifest=str(out_path),
        segment_counts=_segment_counts(manifest),
    )
    mark_stage(
        manifest,
        "synth",
        "completed" if not hard_failed_segments else "completed_with_hard_failed_candidates",
        backend="candidate-pool",
        selected_segments=selected_segments,
        hard_failed_segments=hard_failed_segments,
        downstream_ready=downstream_ready,
        downstream_blocking_segments=downstream_blocking_segments,
        non_blocking_hard_failed_segments=non_blocking_hard_failed_segments,
        selected_segment_count=len(selected_segments),
        hard_failed_segment_count=len(hard_failed_segments),
        hard_fail_reason_counts=hard_fail_reason_counts_payload,
        segment_counts=_segment_counts(manifest),
    )
    manifest.stage_state["synth"]["downstream_readiness"] = synth_ready_for_downstream(manifest)
    save_manifest(project_dir, manifest)
    return ctx.update_manifest(manifest)
