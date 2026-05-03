from __future__ import annotations

# ruff: noqa: F403,F405,I001

from asmr_dub_pipeline.pipeline.context import PipelineContext
from asmr_dub_pipeline.pipeline.stages.common import *


def run_export_stage(ctx: PipelineContext, input_path: Path, confirm_rights: bool) -> PipelineManifest:
    project_dir = ctx.project_dir
    _log_stage_start("export", f"input={input_path}")
    manifest = ctx.reload_manifest()
    _load_config_into_manifest(project_dir, manifest)
    audit = require_confirmed_rights(confirm_rights, "export", input_path)
    manifest.rights_audit = merge_rights_audit(manifest.rights_audit, audit)
    _require_rvc_ready_for_downstream(project_dir, manifest)
    final_audio = Path(manifest.artifacts.get("final_audio", project_dir / "work/mix/final_audio.wav"))
    suffix = ".mp4" if manifest.source_info and manifest.source_info.has_video else ".wav"
    output = ensure_inside_project(project_dir, project_dir / "output" / f"{input_path.stem}_dub{suffix}")
    ensure_not_same_path(input_path, output)
    try:
        console.print(f"[cyan]export[/cyan] muxing output: {output}")
        ffmpeg.mux_audio(input_path, final_audio, output)
    except ffmpeg.FFmpegError:
        if suffix == ".wav":
            shutil.copy2(final_audio, output)
        else:
            raise
    manifest.artifacts["export"] = str(output)
    export_manifest = project_dir / "work" / "export" / "export_manifest.json"
    write_json_atomic(
        export_manifest,
        {
            "input": str(input_path),
            "final_audio": str(final_audio),
            "output": str(output),
            "has_video": bool(manifest.source_info and manifest.source_info.has_video),
        },
    )
    manifest.artifacts["export_manifest"] = str(export_manifest)
    mark_stage(manifest, "export", "completed", output=str(output), export_manifest=str(export_manifest))
    save_manifest(project_dir, manifest)
    _log_stage_complete("export", manifest, f"output={output}")
    return ctx.update_manifest(manifest)
