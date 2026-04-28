from __future__ import annotations

from pathlib import Path

from .config import load_project_config, save_project_config
from .pipeline.steps import (
    analyze_step,
    export_step,
    extract_step,
    gsv_few_shot_step,
    init_project,
    korean_script_step,
    mix_step,
    prepare_source_voice_refs_step,
    qc_step,
    script_step,
    segment_step,
    source_separation_step,
    synth_step,
    transcribe_step,
    translate_ko_step,
)
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
    target_language: str | None = None,
) -> PipelineManifest:
    if mock:
        gemma_backend = "mock"
    normalized_gemma_backend = gemma_backend.replace("-", "_")
    use_korean_text_lane = not mock and normalized_gemma_backend == "llama_cpp"
    init_project(project_dir)
    cfg = load_project_config(project_dir)
    if target_language is not None:
        cfg = type(cfg).model_validate({**cfg.model_dump(mode="json"), "target_language": target_language})
        save_project_config(cfg, project_dir / "pipeline.yaml")
    use_korean_text_lane = cfg.target_language == "ko"
    use_few_shot = False if mock else cfg.gsv_few_shot_enabled if few_shot is None else few_shot
    extract_step(input_path, project_dir, confirm_rights)
    source_separation_step(project_dir, confirm_rights)
    segment_step(project_dir)
    if use_korean_text_lane:
        transcribe_step(project_dir)
        translate_ko_step(project_dir, "mock" if mock else "llama_server")
        korean_script_step(project_dir)
        prepare_source_voice_refs_step(project_dir, refs_path or Path("refs/refs.json"))
    else:
        if use_few_shot:
            transcribe_step(project_dir)
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
    )
    qc_step(project_dir, "mock" if use_korean_text_lane else gemma_backend)
    mix_step(project_dir, confirm_rights)
    return export_step(input_path, project_dir, confirm_rights)
