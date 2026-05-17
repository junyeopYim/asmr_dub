from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from .config import load_project_config, save_project_config
from .gpu_memory import clear_gpu_vram
from .pipeline.manifest_io import load_manifest, save_manifest
from .pipeline.steps import (
    analyze_step,
    assign_speakers_step,
    audio_style_step,
    auto_repair_step,
    countdown_synth_step,
    export_step,
    extract_step,
    gsv_few_shot_step,
    import_voice_bank_source_separation_cache_step,
    init_project,
    korean_script_step,
    mix_step,
    prepare_source_voice_refs_step,
    qc_step,
    regenerate_needs_step,
    rvc_step,
    rvc_train_step,
    script_step,
    segment_step,
    skip_rvc_train_for_voice_bank_step,
    source_separation_step,
    source_speakers_step,
    synth_step,
    transcribe_step,
    translate_ko_step,
    tts_candidate_pool_step,
    tts_select_step,
)
from .qc.repair_plan import plan_is_repairable
from .rvc import validate_rvc_config, validate_rvc_training_config
from .schemas import PipelineManifest

T = TypeVar("T")


def _run_with_optional_gpu_cleanup(
    stage_name: str,
    cleanup_gpu: bool,
    func: Callable[..., T],
    *args: Any,
    **kwargs: Any,
) -> T:
    try:
        return func(*args, **kwargs)
    finally:
        if cleanup_gpu:
            clear_gpu_vram(stage_name)


def _asr_source_separation_fallback_recommendation(
    manifest: PipelineManifest,
) -> dict[str, Any] | None:
    summary_path = manifest.artifacts.get("asr_diagnostics_summary")
    if not summary_path:
        return None
    try:
        summary = json.loads(Path(summary_path).read_text("utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    if not isinstance(summary, dict):
        return None
    raw_recommendation = summary.get("source_separation_fallback")
    if isinstance(raw_recommendation, dict):
        recommendation = raw_recommendation
    elif summary.get("recommend_source_separation_fallback"):
        recommendation = {
            "recommended": True,
            "recommended_backend": summary.get("recommended_source_separation_backend"),
        }
    else:
        return None
    if not recommendation.get("recommended"):
        return None
    if recommendation.get("recommended_backend") != "demucs":
        return None
    return recommendation


def _set_source_separation_backend(project_dir: Path, backend: str) -> str:
    cfg = load_project_config(project_dir)
    previous_backend = cfg.source_separation_backend
    if cfg.source_separation_backend == backend:
        return previous_backend
    cfg.source_separation_backend = backend
    save_project_config(cfg, project_dir / "pipeline.yaml")
    return previous_backend


def _normalize_backend_name(value: str | None) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def _resolve_korean_qc_backend(
    cfg: Any,
    *,
    mock: bool,
    gemma_backend: str,
    ko_qc_backend: str | None = None,
) -> str:
    if mock:
        return "mock"
    configured = _normalize_backend_name(ko_qc_backend or cfg.gemma_qc_backend)
    if configured in {"", "auto"}:
        configured = _normalize_backend_name(gemma_backend)
        if configured in {"", "mock"}:
            configured = _normalize_backend_name(cfg.default_gemma_backend)
        if configured in {"", "mock"}:
            configured = "llama_cpp"
    if configured == "mock" and not cfg.gemma_allow_mock_qc_for_real_korean_lane:
        raise ValueError(
            "Real Korean lane QC cannot use mock unless "
            "gemma_allow_mock_qc_for_real_korean_lane is enabled."
        )
    return configured


def _auto_repair_targets_exist(project_dir: Path) -> bool:
    try:
        manifest = load_manifest(project_dir)
    except Exception:
        return False
    for segment in manifest.segments:
        if segment.status == "needs_regeneration":
            return True
        if segment.status in {"needs_manual_review", "failed"} and plan_is_repairable(
            segment.analysis.get("ko_qc_repair_plan")
        ):
            return True
    return False


def _transcribe_with_optional_source_separation_fallback(
    project_dir: Path,
    *,
    confirm_rights: bool,
    cleanup_gpu: bool = False,
    asr_backend: str | None,
    asr_preset: str | None,
    asr_vad_off: bool | None,
    asr_diagnostics: bool | None,
    asr_device: str | None,
    asr_compute_type: str | None,
    asr_batched_inference: bool | None,
    asr_batch_size: int | None,
) -> PipelineManifest:
    transcribe_kwargs = {
        "asr_backend": asr_backend,
        "asr_preset": asr_preset,
        "asr_vad_off": asr_vad_off,
        "asr_diagnostics": asr_diagnostics,
        "asr_device": asr_device,
        "asr_compute_type": asr_compute_type,
        "asr_batched_inference": asr_batched_inference,
        "asr_batch_size": asr_batch_size,
    }
    manifest = _run_with_optional_gpu_cleanup(
        "transcribe",
        cleanup_gpu,
        transcribe_step,
        project_dir,
        **transcribe_kwargs,
    )
    recommendation = _asr_source_separation_fallback_recommendation(manifest)
    if recommendation is None:
        return manifest

    previous_source_separation_backend = _set_source_separation_backend(project_dir, "demucs")
    try:
        _run_with_optional_gpu_cleanup(
            "source-separation",
            cleanup_gpu,
            source_separation_step,
            project_dir,
            confirm_rights=confirm_rights,
            force=True,
        )
    except Exception:
        if previous_source_separation_backend != "demucs":
            _set_source_separation_backend(project_dir, previous_source_separation_backend)
        raise
    rerun_manifest = _run_with_optional_gpu_cleanup(
        "transcribe",
        cleanup_gpu,
        transcribe_step,
        project_dir,
        **transcribe_kwargs,
    )
    rerun_manifest.stage_state["asr-source-separation-fallback"] = {
        "status": "completed",
        "backend": "demucs",
        "reasons": list(recommendation.get("reasons", [])),
        "initial_metrics": dict(recommendation.get("metrics", {}) or {}),
    }
    save_manifest(project_dir, rerun_manifest)
    return rerun_manifest


def run_pipeline(
    input_path: Path,
    project_dir: Path,
    confirm_rights: bool,
    mock: bool = True,
    gemma_backend: str = "mock",
    gsv_url: str | None = None,
    refs_path: Path | None = None,
    gpt_weights_path: str | None = None,
    sovits_weights_path: str | None = None,
    auto_gsv_server: bool | None = None,
    gsv_server_command: str | list[str] | None = None,
    few_shot: bool | None = None,
    gsv_few_shot_force: bool | None = None,
    rvc_train_force: bool = False,
    use_trained_gpt: bool = False,
    target_language: str | None = None,
    asr_backend: str | None = None,
    asr_preset: str | None = None,
    asr_vad_off: bool | None = None,
    asr_diagnostics: bool | None = None,
    asr_device: str | None = None,
    asr_compute_type: str | None = None,
    asr_batched_inference: bool | None = None,
    asr_batch_size: int | None = None,
    voice_bank_path: Path | None = None,
    require_voice_bank: bool = False,
    source_separation_cache_project: Path | None = None,
    regenerate_before_mix: bool = False,
    merge_input_parts: bool = False,
    auto_repair: bool | None = None,
    auto_repair_max_rounds: int | None = None,
    ko_qc_backend: str | None = None,
    micro_segments: bool | None = None,
) -> PipelineManifest:
    if mock:
        gemma_backend = "mock"
    normalized_gemma_backend = gemma_backend.replace("-", "_")
    normalized_asr_backend = asr_backend.replace("-", "_") if asr_backend else None
    if mock:
        normalized_asr_backend = "mock"
    use_korean_text_lane = not mock and normalized_gemma_backend == "llama_cpp"
    init_project(project_dir)
    cfg = load_project_config(project_dir)
    if target_language is not None:
        cfg = type(cfg).model_validate({**cfg.model_dump(mode="json"), "target_language": target_language})
        save_project_config(cfg, project_dir / "pipeline.yaml")
    if normalized_asr_backend is not None:
        cfg = type(cfg).model_validate({**cfg.model_dump(mode="json"), "asr_backend": normalized_asr_backend})
        save_project_config(cfg, project_dir / "pipeline.yaml")
    if ko_qc_backend is not None:
        cfg = type(cfg).model_validate({**cfg.model_dump(mode="json"), "gemma_qc_backend": ko_qc_backend})
        save_project_config(cfg, project_dir / "pipeline.yaml")
    if auto_repair_max_rounds is not None:
        cfg = type(cfg).model_validate(
            {**cfg.model_dump(mode="json"), "auto_repair_max_rounds": auto_repair_max_rounds}
        )
        save_project_config(cfg, project_dir / "pipeline.yaml")
    if auto_repair is not None:
        cfg = type(cfg).model_validate(
            {**cfg.model_dump(mode="json"), "auto_repair_enabled": auto_repair}
        )
        save_project_config(cfg, project_dir / "pipeline.yaml")
    if micro_segments is not None:
        cfg = type(cfg).model_validate(
            {**cfg.model_dump(mode="json"), "gsv_micro_segment_enabled": micro_segments}
        )
        save_project_config(cfg, project_dir / "pipeline.yaml")
    if asr_preset is not None:
        cfg = type(cfg).model_validate({**cfg.model_dump(mode="json"), "asr_preset": asr_preset.replace("-", "_")})
        save_project_config(cfg, project_dir / "pipeline.yaml")
    if mock and (
        cfg.rvc_backend != "mock"
        or cfg.rvc_train_backend != "mock"
        or cfg.source_separation_backend != "mock"
    ):
        cfg = type(cfg).model_validate(
            {
                **cfg.model_dump(mode="json"),
                "rvc_backend": "mock",
                "rvc_train_backend": "mock",
                "source_separation_backend": "mock",
            }
        )
        save_project_config(cfg, project_dir / "pipeline.yaml")
    use_voice_bank = require_voice_bank or voice_bank_path is not None
    if not mock and not use_voice_bank:
        validate_rvc_training_config(project_dir, cfg, real=True)
        validate_rvc_config(project_dir, cfg, real=True, allow_trained_artifact=True)
    use_korean_text_lane = cfg.target_language == "ko"
    use_few_shot = False if mock or use_voice_bank else cfg.gsv_few_shot_enabled if few_shot is None else few_shot
    cleanup_gpu_after_stage = not mock

    def run_stage(stage_name: str, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        return _run_with_optional_gpu_cleanup(
            stage_name,
            cleanup_gpu_after_stage,
            func,
            *args,
            **kwargs,
        )

    run_stage("extract", extract_step, input_path, project_dir, confirm_rights, merge_parts=merge_input_parts)
    if source_separation_cache_project is not None and not merge_input_parts:
        run_stage(
            "source-separation-cache",
            import_voice_bank_source_separation_cache_step,
            project_dir,
            input_path,
            source_separation_cache_project,
        )
    run_stage("source-separation", source_separation_step, project_dir, confirm_rights)
    if use_korean_text_lane:
        _transcribe_with_optional_source_separation_fallback(
            project_dir,
            confirm_rights=confirm_rights,
            cleanup_gpu=cleanup_gpu_after_stage,
            asr_backend=normalized_asr_backend,
            asr_preset=asr_preset,
            asr_vad_off=asr_vad_off,
            asr_diagnostics=asr_diagnostics,
            asr_device=asr_device,
            asr_compute_type=asr_compute_type,
            asr_batched_inference=asr_batched_inference,
            asr_batch_size=asr_batch_size,
        )
        run_stage("segment", segment_step, project_dir)
        if use_voice_bank:
            run_stage(
                "speaker-assign",
                assign_speakers_step,
                project_dir,
                voice_bank_path=voice_bank_path,
                backend_kind=None,
                require_all=True,
            )
        elif not mock:
            run_stage(
                "source-speakers",
                source_speakers_step,
                project_dir,
                backend_kind="pyannote",
                confirm_rights=confirm_rights,
            )
        audio_style_backend = (
            "mock"
            if mock
            else "llama_server_audio"
            if normalized_gemma_backend == "llama_cpp"
            else gemma_backend
        )
        run_stage(
            "audio-style",
            audio_style_step,
            project_dir,
            audio_style_backend,
            confirm_rights=confirm_rights,
            scope=cfg.gemma_audio_style_scope,
        )
        if not mock and not use_voice_bank:
            run_stage(
                "prepare-refs",
                prepare_source_voice_refs_step,
                project_dir,
                refs_path or Path("refs/refs.json"),
            )
        run_stage("translate-ko", translate_ko_step, project_dir, "mock" if mock else "llama_server")
        run_stage("korean-script", korean_script_step, project_dir)
    else:
        run_stage("segment", segment_step, project_dir)
        if use_voice_bank:
            run_stage(
                "speaker-assign",
                assign_speakers_step,
                project_dir,
                voice_bank_path=voice_bank_path,
                backend_kind=None,
                require_all=True,
            )
        if use_few_shot:
            _transcribe_with_optional_source_separation_fallback(
                project_dir,
                confirm_rights=confirm_rights,
                cleanup_gpu=cleanup_gpu_after_stage,
                asr_backend=normalized_asr_backend,
                asr_preset=asr_preset,
                asr_vad_off=asr_vad_off,
                asr_diagnostics=asr_diagnostics,
                asr_device=asr_device,
                asr_compute_type=asr_compute_type,
                asr_batched_inference=asr_batched_inference,
                asr_batch_size=asr_batch_size,
            )
        if not use_voice_bank and not mock:
            run_stage(
                "source-speakers",
                source_speakers_step,
                project_dir,
                backend_kind="pyannote",
                confirm_rights=confirm_rights,
            )
        run_stage("analyze", analyze_step, project_dir, gemma_backend)
        run_stage("script", script_step, project_dir, gemma_backend)
        if use_few_shot:
            run_stage(
                "prepare-refs",
                prepare_source_voice_refs_step,
                project_dir,
                refs_path or Path("refs/refs.json"),
            )
    if use_few_shot and not (gpt_weights_path and sovits_weights_path):
        run_stage(
            "gsv-few-shot",
            gsv_few_shot_step,
            project_dir,
            confirm_rights=confirm_rights,
            force=gsv_few_shot_force,
            gsv_url=gsv_url,
            gsv_server_command=gsv_server_command,
        )
    tts_pool_enabled = bool(cfg.tts.candidate_pool_enabled)
    if tts_pool_enabled:
        run_stage(
            "tts.candidate_pool",
            tts_candidate_pool_step,
            project_dir,
            refs_path=refs_path or Path("refs/refs.json"),
            confirm_rights=confirm_rights,
            requested_backend="mock" if mock else "auto",
            gsv_url=gsv_url,
            gpt_weights_path=gpt_weights_path,
            sovits_weights_path=sovits_weights_path,
            auto_gsv_server=auto_gsv_server,
            gsv_server_command=gsv_server_command,
            use_trained_gpt=use_trained_gpt,
            mock=mock,
        )
        run_stage("tts.select", tts_select_step, project_dir)
    else:
        run_stage(
            "synth",
            synth_step,
            project_dir,
            gsv_url=gsv_url,
            refs_path=refs_path or Path("refs/refs.json"),
            mock=mock,
            confirm_rights=confirm_rights,
            gpt_weights_path=gpt_weights_path,
            sovits_weights_path=sovits_weights_path,
            auto_gsv_server=auto_gsv_server,
            gsv_server_command=gsv_server_command,
            use_trained_gpt=use_trained_gpt,
            render_countdowns=False,
        )
        run_stage(
            "countdown-synth",
            countdown_synth_step,
            project_dir,
            gsv_url=gsv_url,
            refs_path=refs_path or Path("refs/refs.json"),
            mock=mock,
            confirm_rights=confirm_rights,
            gpt_weights_path=gpt_weights_path,
            sovits_weights_path=sovits_weights_path,
            auto_gsv_server=auto_gsv_server,
            gsv_server_command=gsv_server_command,
            use_trained_gpt=use_trained_gpt,
        )
    if use_voice_bank:
        run_stage("train-rvc", skip_rvc_train_for_voice_bank_step, project_dir)
    else:
        run_stage(
            "train-rvc",
            rvc_train_step,
            project_dir,
            confirm_rights=confirm_rights,
            force=rvc_train_force,
            mock=mock,
        )
    run_stage("rvc", rvc_step, project_dir, confirm_rights=confirm_rights, mock=mock)
    qc_backend = (
        _resolve_korean_qc_backend(
            cfg,
            mock=mock,
            gemma_backend=gemma_backend,
            ko_qc_backend=ko_qc_backend,
        )
        if use_korean_text_lane
        else gemma_backend
    )
    run_stage("qc", qc_step, project_dir, qc_backend)
    if regenerate_before_mix:
        run_stage(
            "regenerate",
            regenerate_needs_step,
            project_dir,
            refs_path=refs_path or Path("refs/refs.json"),
            confirm_rights=confirm_rights,
            gemma_backend=qc_backend,
            tts_backend="auto" if tts_pool_enabled else "gpt-sovits",
            gsv_url=gsv_url,
            gpt_weights_path=gpt_weights_path,
            sovits_weights_path=sovits_weights_path,
            use_trained_gpt=use_trained_gpt,
            auto_gsv_server=auto_gsv_server,
            gsv_server_command=gsv_server_command,
        )
    effective_auto_repair_enabled = cfg.auto_repair_enabled if auto_repair is None else auto_repair
    effective_auto_repair_rounds = (
        cfg.auto_repair_max_rounds
        if auto_repair_max_rounds is None
        else auto_repair_max_rounds
    )
    if (
        not mock
        and use_korean_text_lane
        and effective_auto_repair_enabled
        and cfg.auto_repair_run_after_qc
        and effective_auto_repair_rounds > 0
    ):
        for _round_index in range(effective_auto_repair_rounds):
            if not _auto_repair_targets_exist(project_dir):
                break
            run_stage(
                "auto-repair",
                auto_repair_step,
                project_dir,
                refs_path=refs_path or Path("refs/refs.json"),
                confirm_rights=confirm_rights,
                max_attempts=cfg.auto_repair_max_attempts,
                gemma_backend=qc_backend,
                tts_backend="auto" if tts_pool_enabled else "gpt-sovits",
                gsv_url=gsv_url,
                gpt_weights_path=gpt_weights_path,
                sovits_weights_path=sovits_weights_path,
                use_trained_gpt=use_trained_gpt,
                auto_gsv_server=auto_gsv_server,
                gsv_server_command=gsv_server_command,
            )
    run_stage("mix", mix_step, project_dir, confirm_rights)
    return run_stage("export", export_step, input_path, project_dir, confirm_rights)
