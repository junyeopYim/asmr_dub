from __future__ import annotations

# ruff: noqa: F403,F405,I001

from asmr_dub_pipeline.pipeline.context import PipelineContext
from asmr_dub_pipeline.pipeline.stages.common import *
from asmr_dub_pipeline.script.text_qc import has_minor_sexualized_content


def run_extract_stage(ctx: PipelineContext, input_path: Path, confirm_rights: bool, merge_parts: bool = False) -> PipelineManifest:
    project_dir = ctx.project_dir
    detail = f"input={input_path}"
    if merge_parts:
        detail += " merge_parts=on"
    _log_stage_start("extract", detail)
    create_project_structure(project_dir)
    manifest = ctx.reload_manifest()
    _load_config_into_manifest(project_dir, manifest)
    audit = require_confirmed_rights(
        confirm_rights,
        "extract",
        input_path,
        metadata={"merge_parts_requested": merge_parts, "folder_input_requested": input_path.is_dir()},
    )
    if input_path.is_dir():
        folder_plan = plan_folder_input(input_path)
        if not folder_plan.should_prepare:
            raise ValueError(folder_plan.reason)
        _block_minor_sexualized_input_names(
            project_dir,
            manifest,
            [folder_plan.requested_path, *folder_plan.mix_parts, *folder_plan.asr_parts],
        )
        stereo, mono = prepare_folder_input_audio(folder_plan, project_dir)
        folder_metadata = folder_input_metadata(
            folder_plan,
            mix_audio_path=stereo,
            asr_audio_path=mono,
        )
        manifest.source_info = probe_with_fallback(stereo)
        manifest.source_info.raw["folder_input"] = folder_metadata
        folder_manifest_path = project_dir / "work" / "input" / "folder_input_manifest.json"
        write_json_atomic(folder_manifest_path, folder_metadata)
        manifest.artifacts["folder_input_manifest"] = str(folder_manifest_path)
        manifest.artifacts["folder_mix_source_audio"] = str(stereo)
        manifest.artifacts["folder_asr_source_audio"] = str(mono)
        if audit.history:
            history = [*audit.history]
            history[-1] = {**history[-1], "folder_input": folder_metadata}
            audit = audit.model_copy(update={"history": history})
        manifest.rights_audit = merge_rights_audit(manifest.rights_audit, audit)
        manifest.artifacts["original_stereo_48k"] = str(stereo)
        manifest.artifacts["gemma_mono_16k"] = str(mono)
        mark_stage(
            manifest,
            "extract",
            "completed",
            input_kind="folder",
            part_count=len(folder_plan.mix_parts),
            asr_source_status=folder_plan.asr_source_status,
        )
        save_manifest(project_dir, manifest)
        _log_stage_complete(
            "extract",
            manifest,
            f"folder audio prepared ({len(folder_plan.mix_parts)} part(s), ASR={folder_plan.asr_source_status})",
        )
        return ctx.update_manifest(manifest)
    prepared_input_path = input_path
    input_merge_metadata: dict[str, Any] | None = None
    if merge_parts:
        merge_plan = plan_numbered_part_merge(input_path)
        if merge_plan.status in {"missing_first_part", "missing_numbered_part"}:
            raise ValueError(merge_plan.reason)
        _block_minor_sexualized_input_names(
            project_dir,
            manifest,
            [merge_plan.requested_path, *merge_plan.parts],
        )
        if merge_plan.should_merge:
            prepared_input_path = merge_numbered_parts_to_audio(merge_plan, project_dir)
            input_merge_metadata = _input_merge_metadata(
                requested_path=input_path,
                selected_path=prepared_input_path,
                parts=merge_plan.parts,
                status="merged",
                reason=merge_plan.reason,
                merged_path=prepared_input_path,
            )
        else:
            warning = f"input merge skipped: {merge_plan.reason}"
            if warning not in manifest.warnings:
                manifest.warnings.append(warning)
            input_merge_metadata = _input_merge_metadata(
                requested_path=input_path,
                selected_path=prepared_input_path,
                parts=merge_plan.parts,
                status="skipped",
                reason=merge_plan.reason,
                merged_path=None,
            )
    _block_minor_sexualized_input_names(project_dir, manifest, [prepared_input_path])
    stereo, mono = extract_project_audio(prepared_input_path, project_dir)
    manifest.source_info = probe_with_fallback(prepared_input_path)
    if input_merge_metadata:
        input_merge_metadata["selected_duration_sec"] = manifest.source_info.duration_sec
        manifest.source_info.raw["input_merge"] = input_merge_metadata
        parts_manifest_path = project_dir / "work" / "input" / "input_parts_manifest.json"
        write_json_atomic(parts_manifest_path, input_merge_metadata)
        manifest.artifacts["input_parts_manifest"] = str(parts_manifest_path)
        if input_merge_metadata["status"] == "merged":
            manifest.artifacts["merged_source_audio"] = str(prepared_input_path)
    audit = _attach_input_merge_to_audit(audit, input_merge_metadata)
    manifest.rights_audit = merge_rights_audit(manifest.rights_audit, audit)
    manifest.artifacts["original_stereo_48k"] = str(stereo)
    manifest.artifacts["gemma_mono_16k"] = str(mono)
    mark_stage(manifest, "extract", "completed")
    save_manifest(project_dir, manifest)
    _log_stage_complete("extract", manifest, "audio prepared")
    return ctx.update_manifest(manifest)


def _block_minor_sexualized_input_names(
    project_dir: Path,
    manifest: PipelineManifest,
    paths: list[Path],
) -> None:
    blocked_by_key = {
        str(path.resolve()): path
        for path in paths
        if has_minor_sexualized_content(" ".join(path.parts[-4:]))
    }
    blocked = list(blocked_by_key.values())
    if not blocked:
        return
    mark_stage(
        manifest,
        "extract",
        "failed",
        safety_blocked=len(blocked),
        safety_blocked_paths=[str(path) for path in blocked[:20]],
    )
    save_manifest(project_dir, manifest)
    raise ValueError(
        "extract blocked minor sexualized input names before audio processing "
        f"({len(blocked)} path(s))."
    )
