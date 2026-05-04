from __future__ import annotations

# ruff: noqa: F403,F405,I001

from asmr_dub_pipeline.pipeline.context import PipelineContext
from asmr_dub_pipeline.pipeline.stages.common import *


def _slice_audio_for_folder_export(
    input_path: Path,
    start_sec: float,
    end_sec: float,
    output_path: Path,
) -> Path:
    try:
        return ffmpeg.slice_audio(input_path, start_sec, end_sec, output_path)
    except ffmpeg.FFmpegError:
        data, sample_rate = load_audio(input_path)
        start_frame = max(0, int(round(start_sec * sample_rate)))
        end_frame = max(start_frame + 1, int(round(end_sec * sample_rate)))
        clip = data[start_frame:min(end_frame, len(data))]
        if len(clip) == 0:
            channels = data.shape[1] if data.ndim == 2 else 1
            clip = np.zeros((max(1, int(sample_rate * 0.05)), channels), dtype=np.float32)
        write_audio(output_path, clip, sample_rate)
        return output_path


def _folder_part_has_video(part: dict[str, Any]) -> bool:
    if isinstance(part.get("has_video"), bool):
        return bool(part["has_video"])
    try:
        return ffmpeg.probe_media(Path(part["path"])).has_video
    except ffmpeg.FFmpegError:
        return False


def _unique_output_path(output_dir: Path, stem: str, suffix: str, used: set[Path]) -> Path:
    base = translate_media_stem_to_korean(stem)
    candidate = output_dir / f"{base}_dub{suffix}"
    counter = 2
    while candidate in used or candidate.exists():
        candidate = output_dir / f"{base}_{counter}_dub{suffix}"
        counter += 1
    used.add(candidate)
    return candidate


def _export_folder_outputs(
    *,
    project_dir: Path,
    input_path: Path,
    manifest: PipelineManifest,
    final_audio: Path,
) -> tuple[Path, Path]:
    folder_manifest_path = Path(manifest.artifacts["folder_input_manifest"])
    folder_metadata = json.loads(folder_manifest_path.read_text("utf-8"))
    mix_parts = folder_metadata.get("mix_parts") or []
    if not mix_parts:
        raise ValueError(f"Folder input manifest has no mix parts: {folder_manifest_path}")
    output_dir = ensure_inside_project(project_dir, project_dir / "output")
    output_dir.mkdir(parents=True, exist_ok=True)
    slice_dir = ensure_inside_project(project_dir, project_dir / "work" / "export" / "slices")
    slice_dir.mkdir(parents=True, exist_ok=True)
    try:
        final_duration = duration_sec(final_audio)
    except Exception:
        final_duration = 0.0
    used_outputs: set[Path] = set()
    outputs: list[dict[str, Any]] = []
    for index, part in enumerate(mix_parts, start=1):
        part_path = Path(part["path"])
        start_sec = float(part.get("start_sec", 0.0))
        end_sec = float(part.get("end_sec", start_sec))
        if final_duration > 0:
            start_sec = min(start_sec, final_duration)
            end_sec = min(end_sec, final_duration)
        if end_sec <= start_sec:
            end_sec = start_sec + 0.05
        slice_path = slice_dir / f"{index:03d}_{translate_media_stem_to_korean(part_path.stem)}.wav"
        _slice_audio_for_folder_export(final_audio, start_sec, end_sec, slice_path)
        suffix = part_path.suffix if _folder_part_has_video(part) else ".wav"
        output = ensure_inside_project(
            project_dir,
            _unique_output_path(
                output_dir,
                str(part.get("translated_stem_ko") or part_path.stem),
                suffix,
                used_outputs,
            ),
        )
        ensure_not_same_path(part_path, output)
        try:
            console.print(f"[cyan]export[/cyan] muxing folder part {index}: {output}")
            ffmpeg.mux_audio(part_path, slice_path, output)
        except ffmpeg.FFmpegError:
            if suffix == ".wav":
                shutil.copy2(slice_path, output)
            else:
                raise
        outputs.append(
            {
                "part_index": index,
                "input": str(part_path),
                "start_sec": start_sec,
                "end_sec": end_sec,
                "audio_slice": str(slice_path),
                "output": str(output),
                "translated_stem_ko": translate_media_stem_to_korean(
                    str(part.get("translated_stem_ko") or part_path.stem)
                ),
                "has_video": suffix != ".wav",
            }
        )
    export_manifest = project_dir / "work" / "export" / "export_manifest.json"
    write_json_atomic(
        export_manifest,
        {
            "folder_input": True,
            "input": str(input_path),
            "folder_input_manifest": str(folder_manifest_path),
            "final_audio": str(final_audio),
            "output_dir": str(output_dir),
            "outputs": outputs,
        },
    )
    return output_dir, export_manifest


def run_export_stage(ctx: PipelineContext, input_path: Path, confirm_rights: bool) -> PipelineManifest:
    project_dir = ctx.project_dir
    _log_stage_start("export", f"input={input_path}")
    manifest = ctx.reload_manifest()
    _load_config_into_manifest(project_dir, manifest)
    audit = require_confirmed_rights(confirm_rights, "export", input_path)
    manifest.rights_audit = merge_rights_audit(manifest.rights_audit, audit)
    _require_rvc_ready_for_downstream(project_dir, manifest)
    final_audio = Path(manifest.artifacts.get("final_audio", project_dir / "work/mix/final_audio.wav"))
    if "folder_input_manifest" in manifest.artifacts:
        output_dir, export_manifest = _export_folder_outputs(
            project_dir=project_dir,
            input_path=input_path,
            manifest=manifest,
            final_audio=final_audio,
        )
        manifest.artifacts["export"] = str(output_dir)
        manifest.artifacts["export_manifest"] = str(export_manifest)
        mark_stage(
            manifest,
            "export",
            "completed",
            output=str(output_dir),
            export_manifest=str(export_manifest),
            input_kind="folder",
        )
        save_manifest(project_dir, manifest)
        _log_stage_complete("export", manifest, f"folder outputs={output_dir}")
        return ctx.update_manifest(manifest)
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
