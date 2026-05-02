from __future__ import annotations

from pathlib import Path

from .config import load_project_config, save_project_config
from .pipeline.steps import (
    analyze_step,
    assign_speakers_step,
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
    synth_step,
    transcribe_step,
    translate_ko_step,
)
from .rvc import validate_rvc_config, validate_rvc_training_config
from .schemas import PipelineManifest


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
    use_trained_gpt: bool = False,
    target_language: str | None = None,
    asr_backend: str | None = None,
    voice_bank_path: Path | None = None,
    require_voice_bank: bool = False,
    source_separation_cache_project: Path | None = None,
    regenerate_before_mix: bool = False,
) -> PipelineManifest:
    if mock:
        gemma_backend = "mock"
    normalized_gemma_backend = gemma_backend.replace("-", "_")
    normalized_asr_backend = asr_backend.replace("-", "_") if asr_backend else None
    use_korean_text_lane = not mock and normalized_gemma_backend == "llama_cpp"
    init_project(project_dir)
    cfg = load_project_config(project_dir)
    if target_language is not None:
        cfg = type(cfg).model_validate({**cfg.model_dump(mode="json"), "target_language": target_language})
        save_project_config(cfg, project_dir / "pipeline.yaml")
    if normalized_asr_backend is not None:
        cfg = type(cfg).model_validate({**cfg.model_dump(mode="json"), "asr_backend": normalized_asr_backend})
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
    extract_step(input_path, project_dir, confirm_rights)
    if source_separation_cache_project is not None:
        import_voice_bank_source_separation_cache_step(
            project_dir,
            input_path,
            source_separation_cache_project,
        )
    source_separation_step(project_dir, confirm_rights)
    segment_step(project_dir)
    if use_voice_bank:
        assign_speakers_step(
            project_dir,
            voice_bank_path=voice_bank_path,
            backend_kind=None,
            require_all=True,
        )
    if use_korean_text_lane:
        transcribe_step(project_dir, asr_backend=normalized_asr_backend)
        translate_ko_step(project_dir, "mock" if mock else "llama_server")
        korean_script_step(project_dir)
        if not mock and not use_voice_bank:
            prepare_source_voice_refs_step(project_dir, refs_path or Path("refs/refs.json"))
    else:
        if use_few_shot:
            transcribe_step(project_dir, asr_backend=normalized_asr_backend)
        analyze_step(project_dir, gemma_backend)
        script_step(project_dir, gemma_backend)
        if use_few_shot:
            prepare_source_voice_refs_step(project_dir, refs_path or Path("refs/refs.json"))
    if use_few_shot and not (gpt_weights_path and sovits_weights_path):
        gsv_few_shot_step(
            project_dir,
            confirm_rights=confirm_rights,
            force=gsv_few_shot_force,
            gsv_url=gsv_url,
            gsv_server_command=gsv_server_command,
        )
    synth_step(
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
        skip_rvc_train_for_voice_bank_step(project_dir)
    else:
        rvc_train_step(project_dir, confirm_rights=confirm_rights, mock=mock)
    rvc_step(project_dir, confirm_rights=confirm_rights, mock=mock)
    qc_backend = "mock" if use_korean_text_lane else gemma_backend
    qc_step(project_dir, qc_backend)
    if regenerate_before_mix:
        regenerate_needs_step(
            project_dir,
            refs_path=refs_path or Path("refs/refs.json"),
            confirm_rights=confirm_rights,
            gemma_backend=qc_backend,
            tts_backend="gpt-sovits",
            gsv_url=gsv_url,
            gpt_weights_path=gpt_weights_path,
            sovits_weights_path=sovits_weights_path,
            use_trained_gpt=use_trained_gpt,
            auto_gsv_server=auto_gsv_server,
            gsv_server_command=gsv_server_command,
        )
    mix_step(project_dir, confirm_rights)
    return export_step(input_path, project_dir, confirm_rights)
