from __future__ import annotations

# ruff: noqa: F403,F405,I001

from asmr_dub_pipeline.pipeline.context import PipelineContext
from asmr_dub_pipeline.pipeline.stages.common import *
from asmr_dub_pipeline.gemma.text_translate import _parse_literal_response, build_literal_translate_prompt
from asmr_dub_pipeline.script.text_qc import has_minor_sexualized_content

_TRANSLATION_REFUSAL_MARKERS = (
    "refusal",
    "cannot comply",
    "can't comply",
    "content_filter",
    "content filter",
    "safety",
    "blocked",
    "policy",
)


def _translation_refusal_reason(error: object) -> str | None:
    text = str(error).strip().lower()
    if not text:
        return None
    if any(marker in text for marker in _TRANSLATION_REFUSAL_MARKERS):
        if "minor" in text or "underage" in text:
            return "safety_critical_source_policy"
        if "content_filter" in text or "content filter" in text:
            return "provider_safety_block"
        return "model_refusal"
    return None


def run_translate_ko_stage(
    ctx: PipelineContext,
    gemma_text_backend: str | None = None,
    confirm_rights: bool = False,
    force_retranslate: bool = False,
    retry_failed: bool = False,
    repair_only: bool = False,
    force_retranslate_failed: bool = False,
    *,
    only_segment_ids: set[str] | None = None,
) -> PipelineManifest:
    project_dir = ctx.project_dir
    manifest = ctx.reload_manifest()
    _load_config_into_manifest(project_dir, manifest)
    cfg = manifest.project_config
    backend_kind = (gemma_text_backend or "llama_server").replace("-", "_")
    total = len(manifest.segments)
    target_id_set = set(only_segment_ids) if only_segment_ids is not None else None
    target_segments = [
        segment for segment in manifest.segments if target_id_set is None or segment.id in target_id_set
    ]
    target_segment_ids = [segment.id for segment in target_segments]
    skipped_non_target_segments = [
        segment.id for segment in manifest.segments if target_id_set is not None and segment.id not in target_id_set
    ]
    processed_by_only_segment_ids = target_id_set is not None
    _log_stage_start(
        "translate-ko",
        f"backend={backend_kind}, segments={total}, targets={len(target_segments)}",
    )
    _require_audio_stage_rights(
        manifest,
        "translate-ko",
        confirm_rights,
        metadata={
            "backend": backend_kind,
            "processed_by_only_segment_ids": processed_by_only_segment_ids,
            "target_segment_ids": target_segment_ids,
        },
    )
    if manifest.stage_state.get("transcribe", {}).get("status") != "completed":
        raise ValueError("translate-ko requires a completed transcribe stage.")
    if backend_kind not in {"llama_server", "mock"}:
        raise ValueError(f"Unsupported Gemma text backend: {gemma_text_backend}")

    jsonl_path = project_dir / "work" / "translate_ko" / "translation_bundles.jsonl"
    summary_path = project_dir / "work" / "translate_ko" / "summary.json"
    diagnostics_path = project_dir / "work" / "translate_ko" / "diagnostics.json"
    rows: list[dict[str, Any]] = []
    raw_translation_bundles: list[dict[str, Any]] = []
    repaired_translation_bundles: list[dict[str, Any]] = []
    retry_attempts: list[dict[str, Any]] = []
    diagnostics_lock = Lock()
    quality_counters: Counter[str] = Counter()
    translated = 0
    needs_manual_review = 0
    no_speech_detected = 0
    colloquialized = 0
    digit_pronunciation_postprocessed = 0
    ordinal_postprocessed = 0
    asr_homophone_postprocessed = 0
    numeric_counting_postprocessed = 0
    embedded_countdown_translation_repaired = 0
    asr_backcheck_count = 0
    safety_blocked = 0
    quarantined = 0
    quarantine_rows: list[dict[str, Any]] = []
    model_name = cfg.gemma_llama_cpp_model_path if backend_kind == "llama_server" else "mock"
    translatable: list[Segment] = []

    def reset_downstream_state_for_retranslation(segment: Segment) -> None:
        if segment.status in SKIP_STATUSES:
            segment.status = "transcribed"
        segment.script = None
        segment.tts = None
        segment.rvc = None
        segment.qc = None
        segment.mix = {}
        segment.errors = [
            error
            for error in segment.errors
            if error
            not in {
                "No acceptable TTS candidates for mix.",
                "All TTS candidates failed.",
            }
            and not error.startswith("GPT-SoVITS synthesis failed")
            and not error.startswith("Korean TTS preflight blocked synthesis")
            and not error.startswith("korean-script skipped segment status")
        ]

    def embedded_countdown_event_values(segment: Segment) -> list[int] | None:
        event = segment.analysis.get(COUNTDOWN_EVENT_KEY)
        if not isinstance(event, dict):
            return None
        raw_values = event.get("values")
        if not isinstance(raw_values, list) or not all(isinstance(value, int) for value in raw_values):
            return None
        values = [int(value) for value in raw_values]
        source_text = _translation_source_text(segment)
        if not source_text:
            return None
        if not values or not is_descending_countdown(values):
            return None
        if source_countdown_values(source_text) == values:
            return None
        return values

    def deterministic_countdown_translation_needs_repair(segment: Segment) -> list[int] | None:
        values = embedded_countdown_event_values(segment)
        if values is None or segment.translation_ko is None:
            return None
        translation = segment.translation_ko
        if translation.model == "deterministic:countdown-event":
            return values
        if COUNTDOWN_EVENT_NOTE in translation.notes:
            return values
        return None

    def reset_embedded_countdown_translation(segment: Segment, values: list[int]) -> None:
        reset_downstream_state_for_retranslation(segment)
        segment.status = "transcribed"
        segment.translation_ko = None
        segment.script = None
        segment.tts = None
        segment.rvc = None
        segment.qc = None
        segment.mix = {}
        segment.errors = [
            error
            for error in segment.errors
            if error != "RVC requires segment.tts.selected_candidate_path from synth."
            and not error.startswith("Countdown ")
        ]
        event = segment.analysis.get(COUNTDOWN_EVENT_KEY)
        if isinstance(event, dict):
            event["kind"] = "embedded_countdown"
            event["synth_eligible"] = False
            event["deterministic_translation_eligible"] = False
        segment.analysis["embedded_countdown_translation_repair"] = {
            "reason": "deterministic_countdown_translation_on_embedded_countdown",
            "action": "clear_translation_and_downstream_state",
            "values": values,
        }

    for segment in target_segments:
        if segment.status in NO_SPEECH_STATUSES:
            no_speech_detected += 1
            rows.append(
                {
                    "segment_id": segment.id,
                    "status": segment.status,
                    "reason": f"segment status is {segment.status}",
                    "source_text": "",
                    "translation_ko": None,
                }
            )
            continue
        source_text = segment.source_script.text if segment.source_script else ""
        if source_text.strip() and has_minor_sexualized_content(source_text):
            safety_blocked += 1
            quality_counters["source_minor_sexualized_content"] += 1
            if cfg.translate_ko_safety_block_policy == "quarantine_segment":
                quarantined += 1
                segment.translation_ko = None
                segment.script = None
                segment.tts = None
                segment.rvc = None
                segment.qc = None
                segment.mix = {}
                segment.status = "quarantined"
                message = "translate-ko quarantined segment: safety_critical_source_policy"
                if message not in segment.errors:
                    segment.errors.append(message)
                quarantine = {
                    "segment_id": segment.id,
                    "kind": "safety_critical_source_policy",
                    "reason_code": "safety_critical_source_policy",
                    "recoverable": False,
                    "terminal": True,
                    "reason": "tts_safety_minor_sexualized_content",
                    "batch_id": "source_safety_preflight",
                    "next_action": "manual_review",
                    "attempt_count": 0,
                    "attempt_mode": "source_safety_preflight",
                    "processed_by_only_segment_ids": processed_by_only_segment_ids,
                    "target_segment_ids": target_segment_ids,
                    "final_status": "quarantined",
                    "downstream_invalidated_from": "translate_ko",
                }
                segment.analysis["translate_ko_quarantine"] = quarantine
                quarantine_rows.append(quarantine)
                rows.append(
                    {
                        "segment_id": segment.id,
                        "status": "quarantined",
                        "reason": "safety_critical_source_policy",
                        "source_text": source_text,
                        "translation_ko": None,
                        "quarantine": quarantine,
                    }
                )
            else:
                needs_manual_review += 1
                segment.status = "needs_manual_review"
                message = "translate-ko safety blocked source: tts_safety_minor_sexualized_content"
                if message not in segment.errors:
                    segment.errors.append(message)
                rows.append(
                    {
                        "segment_id": segment.id,
                        "status": "needs_manual_review",
                        "reason": message,
                        "source_text": source_text,
                        "translation_ko": None,
                    }
                )
            continue
        repair_values = deterministic_countdown_translation_needs_repair(segment)
        if repair_values is not None:
            reset_embedded_countdown_translation(segment, repair_values)
            embedded_countdown_translation_repaired += 1
            quality_counters["embedded_countdown_translation_repair"] += 1
        translation = segment.translation_ko
        failed_status = segment.status in SKIP_STATUSES
        retry_this_failed = retry_failed and failed_status
        legacy_numeric_translation = bool(
            translation
            and LEGACY_DETERMINISTIC_NUMERIC_TRANSLATION_NOTE in translation.notes
        )
        countdown_values = _countdown_values_for_segment(segment)
        countdown_translation = (
            _countdown_translation_for_segment(segment, countdown_values)
            if countdown_values is not None
            else None
        )
        if countdown_translation is not None:
            if force_retranslate or retry_this_failed or failed_status:
                reset_downstream_state_for_retranslation(segment)
            segment.translation_ko = countdown_translation
            segment.status = "transcribed"
            segment.errors = []
            translated += 1
            quality_counters[COUNTDOWN_EVENT_NOTE] += 1
            row = {
                "batch_id": countdown_translation.batch_id,
                "segment_id": segment.id,
                "status": "translated",
                "reason": COUNTDOWN_EVENT_NOTE,
                "source_text": source_text,
                "translation_ko": countdown_translation.model_dump(mode="json"),
                "deterministic": True,
            }
            rows.append(row)
            raw_translation_bundles.append(json.loads(json.dumps(row, ensure_ascii=False)))
            continue
        counting_values = _counting_values_for_segment(segment)
        counting_translation = (
            _counting_translation_for_segment(segment, counting_values)
            if counting_values is not None
            else None
        )
        if counting_translation is not None:
            if force_retranslate or retry_this_failed or failed_status:
                reset_downstream_state_for_retranslation(segment)
            segment.translation_ko = counting_translation
            segment.status = "transcribed"
            segment.errors = []
            translated += 1
            numeric_counting_postprocessed += 1
            quality_counters[NUMERIC_COUNTING_POSTPROCESS_NOTE] += 1
            row = {
                "batch_id": counting_translation.batch_id,
                "segment_id": segment.id,
                "status": "translated",
                "reason": NUMERIC_COUNTING_POSTPROCESS_NOTE,
                "source_text": source_text,
                "translation_ko": counting_translation.model_dump(mode="json"),
                "deterministic": True,
            }
            rows.append(row)
            raw_translation_bundles.append(json.loads(json.dumps(row, ensure_ascii=False)))
            continue
        if (
            translation
            and translation.ko_natural.strip()
            and not force_retranslate
            and not (retry_this_failed and force_retranslate_failed)
            and not legacy_numeric_translation
        ):
            row_status = "translated" if not (failed_status and not retry_failed) else "needs_manual_review"
            if row_status == "translated":
                translated += 1
            else:
                needs_manual_review += 1
            row = {
                "batch_id": translation.batch_id,
                "segment_id": segment.id,
                "status": row_status,
                "reason": f"segment status is {segment.status}" if row_status != "translated" else None,
                "source_text": segment.source_script.text if segment.source_script else "",
                "translation_ko": translation.model_dump(mode="json"),
                "resumed": True,
            }
            rows.append(row)
            if row_status == "translated":
                raw_translation_bundles.append(json.loads(json.dumps(row, ensure_ascii=False)))
        elif repair_only:
            needs_manual_review += 1
            rows.append(
                {
                    "segment_id": segment.id,
                    "status": "needs_manual_review",
                    "reason": "repair_only has no existing translation",
                    "source_text": segment.source_script.text if segment.source_script else "",
                    "translation_ko": None,
                }
            )
        elif failed_status and not retry_this_failed and not force_retranslate:
            needs_manual_review += 1
            rows.append(
                {
                    "segment_id": segment.id,
                    "status": "needs_manual_review",
                    "reason": f"segment status is {segment.status}",
                    "source_text": segment.source_script.text if segment.source_script else "",
                    "translation_ko": None,
                }
            )
        elif segment.source_script and segment.source_script.text.strip():
            if force_retranslate or legacy_numeric_translation or (retry_this_failed and force_retranslate_failed):
                segment.translation_ko = None
            if force_retranslate or retry_this_failed:
                reset_downstream_state_for_retranslation(segment)
            translatable.append(segment)
        else:
            needs_manual_review += 1
            rows.append(
                {
                    "segment_id": segment.id,
                    "status": "needs_manual_review",
                    "reason": "missing source_script text",
                    "source_text": segment.source_script.text if segment.source_script else "",
                    "translation_ko": None,
                }
            )

    translation_batches = (
        _translation_span_batches(
            translatable,
            max_segments=cfg.gemma_text_span_size,
            max_duration_sec=cfg.gemma_text_span_max_sec,
            max_gap_sec=cfg.gemma_text_span_max_gap_sec,
        )
        if backend_kind == "llama_server"
        else _chunked(translatable, cfg.gemma_text_batch_size)
    )
    translation_worker_count = (
        _effective_lane_count(cfg.gemma_text_concurrency, len(translation_batches))
        if backend_kind == "llama_server" and translatable
        else 1
    )
    translation_base_urls = [cfg.gemma_text_server_url.rstrip("/")]
    translation_auto_start = bool(cfg.gemma_text_server_auto_start)

    def create_translation_client(_worker_index: int = 0) -> Any:
        if backend_kind == "llama_server":
            return LlamaServerTranslationClient(
                translation_base_urls[0],
                timeout_sec=cfg.gemma_text_timeout_sec,
                retries=cfg.gemma_text_retries,
                n_predict=cfg.gemma_text_n_predict,
                model=model_name,
                two_pass=cfg.gemma_text_two_pass,
                auto_salvage_enabled=cfg.gemma_text_auto_salvage_enabled,
            )
        return MockTranslationClient(model=model_name)

    server_managers: list[ManagedGemmaTextServer] = []
    if backend_kind == "llama_server" and translatable:
        for lane_index, base_url in enumerate(translation_base_urls):
            command = (
                _gemma_text_server_command(cfg, base_url=base_url, lane_index=lane_index)
                if translation_auto_start
                else []
            )
            log_name = "llama_server.log"
            server_managers.append(
                ManagedGemmaTextServer(
                    enabled=translation_auto_start,
                    base_url=base_url,
                    command=command,
                    log_path=project_dir / "work" / "translate_ko" / log_name,
                    startup_timeout_sec=cfg.gemma_text_server_startup_timeout_sec,
                    shutdown_timeout_sec=cfg.gemma_text_server_shutdown_timeout_sec,
                )
            )

    def build_server_metadata() -> dict[str, Any] | None:
        if backend_kind != "llama_server":
            return None
        instances: list[dict[str, Any]] = []
        for index, manager in enumerate(server_managers):
            base_url = getattr(
                manager,
                "base_url",
                translation_base_urls[index] if index < len(translation_base_urls) else "",
            )
            command = [str(part) for part in list(getattr(manager, "command", []) or [])]
            log_path = getattr(manager, "log_path", None)
            instance = {
                "base_url": base_url,
                "enabled": bool(getattr(manager, "enabled", translation_auto_start)),
                "started": bool(getattr(manager, "started", False)),
                "reused_existing": bool(getattr(manager, "reused_existing", False)),
                "log_path": str(log_path) if log_path else None,
                "command_preview": _format_command_preview(command) if command else None,
            }
            instances.append(instance)
        metadata: dict[str, Any] = {
            "auto_start": translation_auto_start,
            "concurrency": translation_worker_count,
            "server_count": len(server_managers),
            "mode": "single_server_slots",
            "base_urls": translation_base_urls,
            "instances": instances,
        }
        if len(instances) == 1:
            metadata.update(
                started=instances[0]["started"],
                reused_existing=instances[0]["reused_existing"],
                log_path=instances[0]["log_path"],
                command_preview=instances[0]["command_preview"],
            )
        return metadata

    if server_managers:
        server_metadata = build_server_metadata() or {}
        first_instance = (server_metadata.get("instances") or [{}])[0]
        _log_stage_checkpoint(
            "translate-ko",
            "Gemma text server configured",
            "auto_start={auto_start} base_urls={base_urls} log_path={log_path} command={command}".format(
                auto_start=translation_auto_start,
                base_urls=",".join(translation_base_urls),
                log_path=first_instance.get("log_path"),
                command=first_instance.get("command_preview"),
            ),
        )

    started_at = monotonic()
    last_logged_at = started_at
    processed = translated + needs_manual_review + quarantined

    def record_raw_translation_row(row: dict[str, Any]) -> None:
        if row.get("status") == "translated":
            raw_translation_bundles.append(json.loads(json.dumps(row, ensure_ascii=False)))

    def record_retry_attempt(
        *,
        attempt_type: str,
        batch_id: str,
        segments: list[Segment],
        accepted: bool,
        reason: str | None = None,
        returned_segment_ids: list[str] | None = None,
    ) -> None:
        payload = {
            "attempt_type": attempt_type,
            "attempt_mode": attempt_type,
            "batch_id": batch_id,
            "segment_ids": [segment.id for segment in segments],
            "accepted": accepted,
            "reason": reason,
            "reason_code": reason,
            "recoverable": accepted,
            "terminal": not accepted,
            "final_status": "accepted" if accepted else "failed",
            "processed_by_only_segment_ids": processed_by_only_segment_ids,
            "target_segment_ids": target_segment_ids,
            "returned_segment_ids": returned_segment_ids or [],
        }
        with diagnostics_lock:
            retry_attempts.append(payload)

    def record_translation_safety_attempt(
        segment: Segment,
        *,
        attempt: int,
        mode: str,
        status: str,
        reason_code: str,
        batch_id: str,
        error: str | None = None,
    ) -> None:
        payload = segment.analysis.setdefault("translation_safety", {})
        if not isinstance(payload, dict):
            payload = {}
            segment.analysis["translation_safety"] = payload
        attempts = payload.setdefault("attempts", [])
        if not isinstance(attempts, list):
            attempts = []
            payload["attempts"] = attempts
        record = {
            "attempt": attempt,
            "mode": mode,
            "attempt_mode": mode,
            "status": status,
            "reason_code": reason_code,
            "recoverable": status == "failed" and reason_code != "safety_critical_source_policy",
            "terminal": status != "success" and reason_code == "safety_critical_source_policy",
            "final_status": status,
            "batch_id": batch_id,
            "parse_status": "parsed" if status == "success" else "failed",
            "processed_by_only_segment_ids": processed_by_only_segment_ids,
            "target_segment_ids": target_segment_ids,
        }
        if error:
            record["error"] = error
        attempts.append(record)
        payload.update(
            {
                "attempt_count": len(attempts),
                "last_status": status,
                "last_reason_code": reason_code,
                "max_attempts": cfg.translate_ko_safety_retry_max_attempts,
            }
        )

    def quarantine_translation_segment(
        segment: Segment,
        *,
        reason_code: str,
        error: str,
        recoverable: bool,
        batch_id: str,
    ) -> dict[str, Any]:
        segment.translation_ko = None
        segment.script = None
        segment.tts = None
        segment.rvc = None
        segment.qc = None
        segment.mix = {}
        segment.status = "quarantined"
        message = f"translate-ko quarantined segment: {reason_code}"
        if message not in segment.errors:
            segment.errors.append(message)
        quarantine = {
            "segment_id": segment.id,
            "kind": reason_code,
            "reason_code": reason_code,
            "recoverable": recoverable,
            "terminal": not recoverable,
            "reason": error,
            "batch_id": batch_id,
            "next_action": "retry_translate_ko" if recoverable else "manual_review",
            "attempt_count": len(
                segment.analysis.get("translation_safety", {}).get("attempts", [])
                if isinstance(segment.analysis.get("translation_safety"), dict)
                else []
            ),
            "attempt_mode": "translation_safety_retry_exhausted",
            "processed_by_only_segment_ids": processed_by_only_segment_ids,
            "target_segment_ids": target_segment_ids,
            "final_status": "quarantined",
            "downstream_invalidated_from": "translate_ko",
        }
        segment.analysis["translate_ko_quarantine"] = quarantine
        quarantine_rows.append(quarantine)
        return quarantine

    def translate_safety_retry_mode(
        client: Any,
        segment: Segment,
        retry_batch_id: str,
        context_segments: list[Segment],
        mode: str,
    ) -> dict[str, Any]:
        if mode == "literal_only_no_expansion" and hasattr(client, "_translate_once"):
            return client._translate_once(
                segments=[segment],
                batch_id=retry_batch_id,
                prompt=build_literal_translate_prompt([segment], retry_batch_id, context_segments),
                parser=_parse_literal_response,
                output_field="ko_literal",
                context_segments=context_segments,
            )
        return _translate_with_optional_context(
            client,
            [segment],
            retry_batch_id,
            context_segments,
        )

    def final_translation_diagnostics_rows() -> list[dict[str, Any]]:
        final_rows = json.loads(json.dumps(rows, ensure_ascii=False))
        for row in final_rows:
            if row.get("status") == "translated":
                row.setdefault("accepted", True)
                row.setdefault("rejected_reasons", [])
                continue
            row.setdefault("accepted", False)
            rejected_reason = str(row.get("error") or row.get("reason") or "needs_manual_review")
            row.setdefault("rejected_reasons", [rejected_reason])
        return final_rows

    def build_diagnostics_payload(partial: bool) -> dict[str, Any]:
        payload = {
            "backend": backend_kind,
            "model": model_name,
            "partial": partial,
            "segments": total,
            "raw_translation_bundles": raw_translation_bundles,
            "repaired_translation_bundles": repaired_translation_bundles,
            "final_translation_bundles": final_translation_diagnostics_rows(),
            "retry_attempts": retry_attempts,
            "quality_counters": dict(sorted(quality_counters.items())),
            "policy": {
                "retry_failed": retry_failed,
                "repair_only": repair_only,
                "force_retranslate": force_retranslate,
                "force_retranslate_failed": force_retranslate_failed,
                "severe_backcheck_promotes_manual_review": True,
                "processed_by_only_segment_ids": processed_by_only_segment_ids,
                "target_segment_ids": target_segment_ids,
                "skipped_non_target_segments": skipped_non_target_segments,
            },
        }
        server_metadata = build_server_metadata()
        if server_metadata is not None:
            payload["server"] = server_metadata
        return payload

    def persist_partial() -> None:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        _write_jsonl_atomic(jsonl_path, rows)
        summary_payload: dict[str, Any] = {
            "backend": backend_kind,
            "model": model_name,
            "segments": total,
            "translated": translated,
            "needs_manual_review": needs_manual_review,
            "colloquialized": colloquialized,
            "digit_pronunciation_postprocessed": digit_pronunciation_postprocessed,
            "ordinal_postprocessed": ordinal_postprocessed,
            "asr_homophone_postprocessed": asr_homophone_postprocessed,
            "numeric_counting_postprocessed": numeric_counting_postprocessed,
            "embedded_countdown_translation_repaired": embedded_countdown_translation_repaired,
            "asr_backcheck_count": asr_backcheck_count,
            "concurrency": translation_worker_count,
            "context_radius": cfg.gemma_text_context_radius,
            "span_size": cfg.gemma_text_span_size if backend_kind == "llama_server" else cfg.gemma_text_batch_size,
            "span_max_sec": cfg.gemma_text_span_max_sec if backend_kind == "llama_server" else None,
            "span_max_gap_sec": cfg.gemma_text_span_max_gap_sec if backend_kind == "llama_server" else None,
            "span_count": len(translation_batches),
            "two_pass": cfg.gemma_text_two_pass if backend_kind == "llama_server" else False,
            "force_retranslate": force_retranslate,
            "retry_failed": retry_failed,
            "repair_only": repair_only,
            "force_retranslate_failed": force_retranslate_failed,
            "base_urls": translation_base_urls if backend_kind == "llama_server" else [],
            "partial": True,
            "processed_by_only_segment_ids": processed_by_only_segment_ids,
            "target_segment_ids": target_segment_ids,
            "processed_target_segments": target_segment_ids,
            "skipped_non_target_segments": skipped_non_target_segments,
        }
        server_metadata = build_server_metadata()
        if server_metadata is not None:
            summary_payload["server"] = server_metadata
        write_json_atomic(summary_path, summary_payload)
        write_json_atomic(diagnostics_path, build_diagnostics_payload(partial=True))
        manifest.artifacts["translation_bundles"] = str(jsonl_path)
        manifest.artifacts["translation_summary"] = str(summary_path)
        manifest.artifacts["translation_diagnostics"] = str(diagnostics_path)
        save_manifest(project_dir, manifest)

    def persist_failed(exc: Exception) -> None:
        persist_partial()
        server_metadata = build_server_metadata()
        _log_stage_checkpoint(
            "translate-ko",
            "failed",
            f"error={exc} server={server_metadata}",
        )
        mark_stage(
            manifest,
            "translate-ko",
            "failed",
            backend=backend_kind,
            model=model_name,
            translated=translated,
            needs_manual_review=needs_manual_review,
            no_speech_detected=no_speech_detected,
            quality_counters=dict(sorted(quality_counters.items())),
            concurrency=translation_worker_count,
            context_radius=cfg.gemma_text_context_radius,
            span_size=cfg.gemma_text_span_size if backend_kind == "llama_server" else cfg.gemma_text_batch_size,
            span_max_sec=cfg.gemma_text_span_max_sec if backend_kind == "llama_server" else None,
            span_max_gap_sec=cfg.gemma_text_span_max_gap_sec if backend_kind == "llama_server" else None,
            span_count=len(translation_batches),
            two_pass=cfg.gemma_text_two_pass if backend_kind == "llama_server" else False,
            force_retranslate=force_retranslate,
            retry_failed=retry_failed,
            repair_only=repair_only,
            force_retranslate_failed=force_retranslate_failed,
            base_urls=translation_base_urls if backend_kind == "llama_server" else [],
            partial=True,
            processed_by_only_segment_ids=processed_by_only_segment_ids,
            target_segment_ids=target_segment_ids,
            processed_target_segments=target_segment_ids,
            skipped_non_target_segments=skipped_non_target_segments,
            error=str(exc),
            server=server_metadata,
        )
        save_manifest(project_dir, manifest)

    if rows:
        persist_partial()
    if safety_blocked and cfg.translate_ko_safety_block_policy == "fail_stage":
        mark_stage(
            manifest,
            "translate-ko",
            "failed",
            backend=backend_kind,
            model=model_name,
            translated=translated,
            needs_manual_review=needs_manual_review,
            no_speech_detected=no_speech_detected,
            safety_blocked=safety_blocked,
            quality_counters=dict(sorted(quality_counters.items())),
        )
        save_manifest(project_dir, manifest)
        raise ValueError(
            "translate-ko blocked minor sexualized source content before translation "
            f"({safety_blocked} segment(s))."
        )

    def translate_batch_with_retries(
        batch: list[Segment],
        batch_id: str,
        worker_index: int,
    ) -> tuple[list[Segment], str, dict[str, Any], dict[str, list[str]]]:
        client = create_translation_client(worker_index)
        translation_failures: dict[str, list[str]] = {}
        translations: dict[str, Any] = {}

        def record_failure(segment: Segment, message: str) -> None:
            translation_failures.setdefault(segment.id, []).append(message)

        def retry_single(missing_segment: Segment, retry_batch_id: str) -> None:
            context_segments = _translation_context_segments(
                manifest.segments,
                [missing_segment],
                cfg.gemma_text_context_radius,
            )
            try:
                translations.update(
                    _translate_with_optional_context(
                        client,
                        [missing_segment],
                        retry_batch_id,
                        context_segments,
                    )
                )
                record_retry_attempt(
                    attempt_type="single",
                    batch_id=retry_batch_id,
                    segments=[missing_segment],
                    accepted=missing_segment.id in translations,
                    reason=None if missing_segment.id in translations else "missing model response",
                    returned_segment_ids=[segment_id for segment_id in translations if segment_id == missing_segment.id],
                )
            except Exception as exc:
                message = f"Korean translation retry failed for {retry_batch_id}: {exc}"
                record_retry_attempt(
                    attempt_type="single",
                    batch_id=retry_batch_id,
                    segments=[missing_segment],
                    accepted=False,
                    reason=str(exc),
                )
                record_failure(missing_segment, message)
                return
            if missing_segment.id not in translations:
                record_failure(
                    missing_segment,
                    f"Korean translation retry failed for {retry_batch_id}: missing model response",
                )

        def translate_group(group: list[Segment], group_batch_id: str) -> None:
            if not group:
                return
            context_segments = _translation_context_segments(
                manifest.segments,
                group,
                cfg.gemma_text_context_radius,
            )
            try:
                group_translations = _translate_with_optional_context(
                    client,
                    group,
                    group_batch_id,
                    context_segments,
                )
                record_retry_attempt(
                    attempt_type=(
                        "single"
                        if len(group) == 1
                        else "split"
                        if "_split_" in group_batch_id or "_missing_" in group_batch_id
                        else "batch"
                    ),
                    batch_id=group_batch_id,
                    segments=group,
                    accepted=True,
                    returned_segment_ids=list(group_translations),
                )
            except Exception as exc:
                if backend_kind == "llama_server" and _is_gemma_text_server_unavailable(exc):
                    raise
                refusal_reason = _translation_refusal_reason(exc)
                if (
                    refusal_reason is not None
                    and len(group) == 1
                    and cfg.translate_ko_safety_retry_enabled
                    and cfg.translate_ko_safety_block_policy == "quarantine_segment"
                ):
                    segment = group[0]
                    retry_modes = list(cfg.translate_ko_safety_retry_modes)[
                        : cfg.translate_ko_safety_retry_max_attempts
                    ]
                    for attempt, mode in enumerate(retry_modes, start=1):
                        retry_batch_id = f"{group_batch_id}_safety_{attempt:02d}_{mode}"
                        try:
                            context_segments = _translation_context_segments(
                                manifest.segments,
                                [segment],
                                0 if mode == "context_trimmed" else cfg.gemma_text_context_radius,
                            )
                            retry_translations = translate_safety_retry_mode(
                                client,
                                segment,
                                retry_batch_id,
                                context_segments,
                                mode,
                            )
                        except Exception as retry_exc:
                            retry_reason = _translation_refusal_reason(retry_exc) or refusal_reason
                            record_translation_safety_attempt(
                                segment,
                                attempt=attempt,
                                mode=mode,
                                status="failed",
                                reason_code=retry_reason,
                                batch_id=retry_batch_id,
                                error=str(retry_exc),
                            )
                            continue
                        if segment.id in retry_translations:
                            translations.update({segment.id: retry_translations[segment.id]})
                            record_translation_safety_attempt(
                                segment,
                                attempt=attempt,
                                mode=mode,
                                status="success",
                                reason_code=refusal_reason,
                                batch_id=retry_batch_id,
                            )
                            record_retry_attempt(
                                attempt_type="safety_retry",
                                batch_id=retry_batch_id,
                                segments=[segment],
                                accepted=True,
                                reason=refusal_reason,
                                returned_segment_ids=[segment.id],
                            )
                            return
                    record_failure(group[0], f"translation_safety_refusal:{refusal_reason}:{exc}")
                    record_retry_attempt(
                        attempt_type="safety_retry",
                        batch_id=group_batch_id,
                        segments=group,
                        accepted=False,
                        reason=refusal_reason,
                    )
                    return
                message = f"Korean translation batch failed for {group_batch_id}: {exc}"
                record_retry_attempt(
                    attempt_type=(
                        "single"
                        if len(group) == 1
                        else "split"
                        if "_split_" in group_batch_id or "_missing_" in group_batch_id
                        else "batch"
                    ),
                    batch_id=group_batch_id,
                    segments=group,
                    accepted=False,
                    reason=str(exc),
                )
                if len(group) == 1:
                    record_failure(group[0], message)
                    return
                for segment in group:
                    record_failure(segment, message)
                midpoint = max(1, len(group) // 2)
                translate_group(group[:midpoint], f"{group_batch_id}_split_01")
                translate_group(group[midpoint:], f"{group_batch_id}_split_02")
                return
            translations.update(group_translations)
            missing_after_group = [segment for segment in group if segment.id not in translations]
            if not missing_after_group:
                return
            if len(group) == 1:
                retry_single(group[0], f"{group_batch_id}_single_01")
                return
            if len(missing_after_group) == 1:
                retry_single(missing_after_group[0], f"{group_batch_id}_single_01")
                return
            midpoint = max(1, len(missing_after_group) // 2)
            translate_group(missing_after_group[:midpoint], f"{group_batch_id}_missing_01")
            translate_group(missing_after_group[midpoint:], f"{group_batch_id}_missing_02")

        translate_group(batch, batch_id)
        return batch, batch_id, translations, translation_failures

    def apply_batch_result(
        batch: list[Segment],
        batch_id: str,
        translations: dict[str, Any],
        translation_failures: dict[str, list[str]],
    ) -> None:
        nonlocal last_logged_at, needs_manual_review, numeric_counting_postprocessed, processed, quarantined, translated
        for segment in batch:
            translation = translations.get(segment.id)
            if translation is None:
                failure_text = "; ".join(translation_failures.get(segment.id, []))
                refusal_reason = _translation_refusal_reason(failure_text)
                if refusal_reason is not None and cfg.translate_ko_safety_block_policy == "quarantine_segment":
                    quarantine = quarantine_translation_segment(
                        segment,
                        reason_code=refusal_reason,
                        error=failure_text,
                        recoverable=refusal_reason != "safety_critical_source_policy",
                        batch_id=batch_id,
                    )
                    quarantined += 1
                    processed += 1
                    source_text = segment.source_script.text if segment.source_script else ""
                    rows.append(
                        {
                            "batch_id": batch_id,
                            "segment_id": segment.id,
                            "status": "quarantined",
                            "reason": refusal_reason,
                            "source_text": source_text,
                            "translation_ko": None,
                            "quarantine": quarantine,
                            "error": failure_text,
                        }
                    )
                    last_logged_at = _log_translate_progress(
                        processed,
                        total,
                        segment,
                        "quarantined",
                        source_text,
                        None,
                        started_at,
                        last_logged_at,
                    )
                    continue
                segment.status = "needs_manual_review"
                for message in translation_failures.get(segment.id, []):
                    if message not in segment.errors:
                        segment.errors.append(message)
                needs_manual_review += 1
                processed += 1
                source_text = segment.source_script.text if segment.source_script else ""
                rows.append(
                    {
                        "batch_id": batch_id,
                        "segment_id": segment.id,
                        "status": "needs_manual_review",
                        "reason": "missing translation in model response",
                        "source_text": source_text,
                        "translation_ko": None,
                        "error": "; ".join(translation_failures.get(segment.id, [])) or None,
                    }
                )
                last_logged_at = _log_translate_progress(
                    processed,
                    total,
                    segment,
                    "needs_manual_review",
                    source_text,
                    None,
                    started_at,
                    last_logged_at,
                )
                continue
            segment.translation_ko = translation
            source_text = segment.source_script.text if segment.source_script else ""
            raw_row = {
                "batch_id": batch_id,
                "segment_id": segment.id,
                "status": "translated",
                "source_text": source_text,
                "translation_ko": translation.model_dump(mode="json"),
            }
            record_raw_translation_row(raw_row)
            numeric_counting_postprocessed += _apply_korean_numeric_counting_postprocess([segment])
            translation = segment.translation_ko or translation
            _clear_korean_translation_errors(segment)
            translated += 1
            processed += 1
            rows.append(
                {
                    "batch_id": batch_id,
                    "segment_id": segment.id,
                    "status": "translated",
                    "source_text": source_text,
                    "translation_ko": translation.model_dump(mode="json"),
                }
            )
            last_logged_at = _log_translate_progress(
                processed,
                total,
                segment,
                "translated",
                source_text,
                translation.ko_natural,
                started_at,
                last_logged_at,
            )

    try:
        for server_manager in server_managers:
            _log_stage_checkpoint(
                "translate-ko",
                "starting Gemma text server",
                "base_url={base_url} auto_start={auto_start} log_path={log_path}".format(
                    base_url=getattr(server_manager, "base_url", ""),
                    auto_start=translation_auto_start,
                    log_path=getattr(server_manager, "log_path", None),
                ),
            )
            server_manager.start()
            _log_stage_checkpoint(
                "translate-ko",
                "Gemma text server ready",
                "base_url={base_url} started={started} reused_existing={reused_existing}".format(
                    base_url=getattr(server_manager, "base_url", ""),
                    started=getattr(server_manager, "started", False),
                    reused_existing=getattr(server_manager, "reused_existing", False),
                ),
            )
        batch_jobs = []
        for job_index, batch in enumerate(translation_batches):
            batch_id = f"batch_{job_index + 1:04d}"
            worker_index = job_index % translation_worker_count
            batch_jobs.append((job_index, batch, batch_id, worker_index))
        if translation_worker_count > 1 and len(batch_jobs) > 1:
            pending: dict[int, tuple[list[Segment], str, dict[str, Any], dict[str, list[str]]]] = {}
            next_to_apply = 0
            with ThreadPoolExecutor(max_workers=translation_worker_count) as executor:
                futures = {
                    executor.submit(
                        translate_batch_with_retries,
                        batch,
                        batch_id,
                        worker_index,
                    ): job_index
                    for job_index, batch, batch_id, worker_index in batch_jobs
                }
                for future in as_completed(futures):
                    job_index = futures[future]
                    pending[job_index] = future.result()
                    while next_to_apply in pending:
                        apply_batch_result(*pending.pop(next_to_apply))
                        persist_partial()
                        next_to_apply += 1
        else:
            for _, batch, batch_id, worker_index in batch_jobs:
                apply_batch_result(*translate_batch_with_retries(batch, batch_id, worker_index))
                persist_partial()
    except Exception as exc:
        persist_failed(exc)
        raise
    finally:
        for server_manager in reversed(server_managers):
            server_manager.stop()

    digit_pronunciation_postprocessed = _apply_korean_digit_pronunciation_postprocess(
        target_segments,
        repaired_translation_bundles,
        quality_counters,
    )
    ordinal_postprocessed = _apply_korean_ordinal_postprocess(
        target_segments,
        repaired_translation_bundles,
        quality_counters,
    )
    asr_homophone_postprocessed = _apply_korean_asr_homophone_postprocess(
        target_segments,
        repaired_translation_bundles,
        quality_counters,
    )
    _apply_korean_onomatopoeia_postprocess(
        target_segments,
        repaired_translation_bundles,
        quality_counters,
    )
    _apply_korean_fluency_postprocess(
        target_segments,
        repaired_translation_bundles,
        quality_counters,
    )
    colloquialized = _apply_korean_colloquial_postprocess(target_segments)
    numeric_counting_postprocessed += _apply_korean_numeric_counting_postprocess(target_segments)
    asr_backcheck_items = _apply_translation_asr_backcheck(target_segments, cfg)
    asr_backcheck_count = len(asr_backcheck_items)
    _refresh_translation_rows(rows, manifest.segments)
    _attach_asr_backcheck_to_translation_rows(rows, asr_backcheck_items)
    _finalize_translation_acceptance(
        rows,
        manifest.segments,
        asr_backcheck_items,
        quality_counters,
        cfg=cfg,
    )
    translated = sum(1 for row in rows if row.get("status") == "translated")
    needs_manual_review = sum(1 for row in rows if row.get("status") == "needs_manual_review")
    no_speech_detected = sum(1 for row in rows if row.get("status") in NO_SPEECH_STATUSES)
    non_speech_texture = sum(1 for row in rows if row.get("status") == "non_speech_texture")
    quarantined = sum(1 for row in rows if row.get("status") == "quarantined")
    quarantine_path = project_dir / "work" / "translate_ko" / "quarantine.json"
    write_json_atomic(
        quarantine_path,
        {
            "segments": quarantine_rows,
            "quarantine_count": quarantined,
            "policy": cfg.translate_ko_safety_block_policy,
            "processed_by_only_segment_ids": processed_by_only_segment_ids,
            "target_segment_ids": target_segment_ids,
        },
    )
    _write_jsonl_atomic(jsonl_path, rows)
    asr_backcheck_summary_path = project_dir / "work" / "translate_ko" / "asr_backcheck_summary.json"
    write_json_atomic(
        asr_backcheck_summary_path,
        {
            "enabled": cfg.asr_translation_backcheck_enabled,
            "flagged": asr_backcheck_count,
            "mark_manual_review": cfg.asr_translation_backcheck_mark_manual_review,
            "items": asr_backcheck_items,
        },
    )
    summary = {
        "backend": backend_kind,
        "model": model_name,
        "segments": total,
        "translated": translated,
        "needs_manual_review": needs_manual_review,
        "no_speech_detected": no_speech_detected,
        "non_speech_texture": non_speech_texture,
        "colloquialized": colloquialized,
        "digit_pronunciation_postprocessed": digit_pronunciation_postprocessed,
        "ordinal_postprocessed": ordinal_postprocessed,
        "asr_homophone_postprocessed": asr_homophone_postprocessed,
        "numeric_counting_postprocessed": numeric_counting_postprocessed,
        "embedded_countdown_translation_repaired": embedded_countdown_translation_repaired,
        "asr_backcheck_count": asr_backcheck_count,
        "concurrency": translation_worker_count,
        "context_radius": cfg.gemma_text_context_radius,
        "span_size": cfg.gemma_text_span_size if backend_kind == "llama_server" else cfg.gemma_text_batch_size,
        "span_max_sec": cfg.gemma_text_span_max_sec if backend_kind == "llama_server" else None,
        "span_max_gap_sec": cfg.gemma_text_span_max_gap_sec if backend_kind == "llama_server" else None,
        "span_count": len(translation_batches),
        "two_pass": cfg.gemma_text_two_pass if backend_kind == "llama_server" else False,
        "force_retranslate": force_retranslate,
        "retry_failed": retry_failed,
        "repair_only": repair_only,
        "force_retranslate_failed": force_retranslate_failed,
        "base_urls": translation_base_urls if backend_kind == "llama_server" else [],
        "quality_counters": dict(sorted(quality_counters.items())),
        "processed_by_only_segment_ids": processed_by_only_segment_ids,
        "target_segment_ids": target_segment_ids,
        "processed_target_segments": target_segment_ids,
        "skipped_non_target_segments": skipped_non_target_segments,
    }
    write_json_atomic(summary_path, summary)
    write_json_atomic(diagnostics_path, build_diagnostics_payload(partial=False))
    manifest.artifacts["translation_bundles"] = str(jsonl_path)
    manifest.artifacts["translation_summary"] = str(summary_path)
    manifest.artifacts["translation_diagnostics"] = str(diagnostics_path)
    manifest.artifacts["translation_asr_backcheck_summary"] = str(asr_backcheck_summary_path)
    if quarantined:
        manifest.artifacts["translation_quarantine"] = str(quarantine_path)
    server_metadata = build_server_metadata()
    stage_status = "completed_with_quarantined_segments" if quarantined else "completed"
    mark_stage(
        manifest,
        "translate-ko",
        stage_status,
        backend=backend_kind,
        model=model_name,
        translated=translated,
        needs_manual_review=needs_manual_review,
        no_speech_detected=no_speech_detected,
        non_speech_texture=non_speech_texture,
        quarantined_segments=[row["segment_id"] for row in quarantine_rows],
        quarantine_count=quarantined,
        recoverable_quarantine_count=sum(1 for row in quarantine_rows if row.get("recoverable")),
        colloquialized=colloquialized,
        digit_pronunciation_postprocessed=digit_pronunciation_postprocessed,
        ordinal_postprocessed=ordinal_postprocessed,
        asr_homophone_postprocessed=asr_homophone_postprocessed,
        numeric_counting_postprocessed=numeric_counting_postprocessed,
        embedded_countdown_translation_repaired=embedded_countdown_translation_repaired,
        asr_backcheck_count=asr_backcheck_count,
        quality_counters=dict(sorted(quality_counters.items())),
        concurrency=translation_worker_count,
        context_radius=cfg.gemma_text_context_radius,
        span_size=cfg.gemma_text_span_size if backend_kind == "llama_server" else cfg.gemma_text_batch_size,
        span_max_sec=cfg.gemma_text_span_max_sec if backend_kind == "llama_server" else None,
        span_max_gap_sec=cfg.gemma_text_span_max_gap_sec if backend_kind == "llama_server" else None,
        span_count=len(translation_batches),
        two_pass=cfg.gemma_text_two_pass if backend_kind == "llama_server" else False,
        force_retranslate=force_retranslate,
        retry_failed=retry_failed,
        repair_only=repair_only,
        force_retranslate_failed=force_retranslate_failed,
        processed_by_only_segment_ids=processed_by_only_segment_ids,
        target_segment_ids=target_segment_ids,
        processed_target_segments=target_segment_ids,
        skipped_non_target_segments=skipped_non_target_segments,
        server=server_metadata,
    )
    save_manifest(project_dir, manifest)
    _log_stage_complete("translate-ko", manifest, f"backend={backend_kind}")
    return ctx.update_manifest(manifest)
