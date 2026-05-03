from __future__ import annotations

# ruff: noqa: F403,F405,I001

from asmr_dub_pipeline.pipeline.context import PipelineContext
from asmr_dub_pipeline.pipeline.stages.common import *


def run_import_voice_bank_source_separation_cache_stage(ctx: PipelineContext, input_path: Path, cache_project_dir: Path) -> PipelineManifest:
    project_dir = ctx.project_dir
    manifest = ctx.reload_manifest()
    _load_config_into_manifest(project_dir, manifest)
    original_audio = Path(
        manifest.artifacts.get("original_stereo_48k", project_dir / "work/audio/original_stereo_48k.wav")
    )
    if not original_audio.exists():
        return ctx.update_manifest(manifest)
    cache_project_dir = cache_project_dir.expanduser().resolve()
    candidates = _voice_bank_source_separation_candidates(
        cache_project_dir,
        input_path.expanduser().resolve(),
        original_audio,
        manifest.project_config,
    )
    if not candidates:
        return ctx.update_manifest(manifest)

    candidate = candidates[0]
    audio_dir = ensure_inside_project(project_dir, project_dir / "work" / "audio")
    separation_dir = ensure_inside_project(project_dir, project_dir / "work" / "source_separation")
    audio_dir.mkdir(parents=True, exist_ok=True)
    separation_dir.mkdir(parents=True, exist_ok=True)
    destination_paths = {
        "source_vocals_48k": audio_dir / "source_vocals_48k.wav",
        "source_vocals_mono_16k": audio_dir / "source_vocals_mono_16k.wav",
        "background_only_48k": audio_dir / "background_only_48k.wav",
    }
    for key, source_path in candidate.paths.items():
        destination_path = ensure_inside_project(project_dir, destination_paths[key])
        if source_path.resolve() != destination_path.resolve():
            shutil.copy2(source_path, destination_path)

    import_manifest_path = separation_dir / "source_separation_cache_import.json"
    separation_manifest_path = separation_dir / "source_separation_manifest.json"
    import_metadata = {
        "cache_project_dir": str(cache_project_dir),
        "source_dir": str(candidate.source_dir.resolve()),
        "input_path": str(input_path.expanduser().resolve()),
        "matched_by": candidate.matched_by,
        "source_paths": {key: str(path.resolve()) for key, path in candidate.paths.items()},
        "destination_paths": {key: str(path.resolve()) for key, path in destination_paths.items()},
    }
    write_json_atomic(import_manifest_path, import_metadata)
    write_json_atomic(
        separation_manifest_path,
        {
            "backend": "cached",
            "model": "voice_bank_cache",
            "input_audio_path": str(original_audio),
            "vocals_path": str(destination_paths["source_vocals_48k"]),
            "vocals_mono_path": str(destination_paths["source_vocals_mono_16k"]),
            "background_path": str(destination_paths["background_only_48k"]),
            "reused_existing": True,
            "command": [],
            "cache_import_manifest": str(import_manifest_path),
        },
    )
    manifest.artifacts["source_separation_cache_import"] = str(import_manifest_path)
    manifest.artifacts["source_separation_manifest"] = str(separation_manifest_path)
    save_manifest(project_dir, manifest)
    console.print(f"[cyan]source-separation[/cyan] imported cached voice-bank stems: {candidate.source_dir}")
    return ctx.update_manifest(manifest)
