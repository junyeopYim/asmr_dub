from __future__ import annotations

import os
import re
import shutil
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

import typer
from rich.table import Table

from asmr_dub_pipeline.audio.preprocess import numbered_part_base_stem
from asmr_dub_pipeline.gemma.llama_cpp_client import (
    DEFAULT_LLAMA_CPP_CLI,
    DEFAULT_LLAMA_CPP_MMPROJ,
    DEFAULT_LLAMA_CPP_MODEL,
)

from .config import load_project_config, save_project_config
from .gpu_memory import clear_gpu_vram
from .logging import console
from .orchestrator import run_pipeline
from .pipeline.manifest_io import manifest_path, write_json_atomic
from .pipeline.steps import (
    analyze_step,
    audio_style_step,
    auto_repair_step,
    countdown_synth_step,
    export_step,
    extract_step,
    gsv_few_shot_step,
    init_project,
    inspect_input,
    korean_script_step,
    mix_step,
    prepare_source_voice_refs_step,
    qc_step,
    regenerate_needs_step,
    rvc_step,
    rvc_train_step,
    script_step,
    segment_step,
    source_separation_step,
    source_speakers_step,
    synth_experimental_tts_step,
    synth_qwen_step,
    synth_step,
    transcribe_step,
    translate_ko_step,
)
from .rights import RIGHTS_MESSAGE, RightsError, require_confirmed_rights
from .rvc import validate_rvc_config, validate_rvc_training_config
from .voice_bank import build_voice_bank

RIGHTS_HELP = (
    "Confirm you own or have permission/consent for the source content, voice "
    "references, and distribution."
)
TRAINED_GPT_HELP = (
    "Use the few-shot trained GPT .ckpt during synthesis instead of the auto "
    "base-GPT fallback; explicit --gpt-weights still wins."
)
REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".mp4", ".mkv", ".mov"}
DEFAULT_VOICE_BANK_PROJECT_NAME = "voice_bank_all"
T = TypeVar("T")
FULL_REAL_QUALITY_PRESET = {
    "source_language": "ja",
    "asr_preset": "whisper",
    "asr_batched_inference": True,
    "asr_batch_size": 16,
    "asr_word_timestamps": True,
    "asr_diagnostics_enabled": True,
    "asr_resegment_max_sec": 10.0,
    "asr_qwen_repair_fallback_enabled": True,
    "asr_review_enabled": True,
    "asr_review_generate_candidates": True,
    "source_separation_backend": "auto",
    "asr_translation_backcheck_enabled": True,
    "candidate_count": 5,
    "duration_tolerance": 0.25,
    "auto_repair_enabled": True,
    "auto_repair_max_rounds": 3,
    "auto_repair_run_after_qc": True,
    "auto_repair_max_attempts": 3,
    "gemma_qc_backend": "auto",
    "gemma_allow_mock_qc_for_real_korean_lane": False,
    "gemma_llama_cpp_ctx_size": 16384,
    "gemma_llama_cpp_n_predict": 2048,
    "gemma_text_batch_size": 1,
    "gemma_text_context_radius": 10,
    "gemma_text_concurrency": 1,
    "gemma_text_n_predict": 1536,
    "gemma_text_retries": 2,
    "gemma_text_span_size": 10,
    "gemma_text_span_max_sec": 75.0,
    "gemma_text_span_max_gap_sec": 5.0,
    "gemma_text_timeout_sec": 900.0,
    "gemma_text_server_startup_timeout_sec": 900.0,
    "gemma_audio_style_scope": "speaker_suspicious",
    "gsv_timeout_sec": 240.0,
    "gsv_retries": 3,
    "gsv_concurrency": 2,
    "gsv_pronunciation_qc_workers": 2,
    "gsv_few_shot_min_total_sec": 60.0,
    "gsv_few_shot_min_clip_sec": 2.0,
    "gsv_few_shot_max_clip_sec": 8.0,
    "gsv_few_shot_min_quality_score": 0.30,
    "gsv_few_shot_clean_source_filter": True,
    "gsv_few_shot_max_background_bleed_db": -21.0,
    "gsv_few_shot_max_side_to_mid_db": -6.0,
    "gsv_few_shot_preferred_chars_per_sec": 4.5,
    "gsv_few_shot_max_chars_per_sec": 5.2,
    "gsv_few_shot_min_selection_score": 0.50,
    "gsv_few_shot_max_total_sec": 240.0,
    "gsv_few_shot_asr_risk_filter": True,
    "gsv_few_shot_prefer_plain_text": False,
    "gsv_few_shot_enabled": True,
    "gsv_ref_mode": "segment",
    "gsv_ref_min_sec": 3.0,
    "gsv_ref_max_sec": 10.0,
    "gsv_ref_min_quality_score": 0.40,
    "gsv_segment_ref_min_overlap_ratio": 0.75,
    "gsv_segment_ref_relaxed_training_reasons": [
        "low_dominant_source_speaker_overlap",
        "borderline_single_speaker_overlap",
        "neighbor_confirmed_single_speaker_overlap",
        "single_speaker_overlap_tts_routing",
        "merged_overlap_candidates_tts_routing",
    ],
    "gsv_tts_max_speed_factor": 1.20,
    "gsv_tts_min_speed_factor": 0.90,
    "gsv_max_attempts_per_candidate": 3,
    "gsv_duration_rewrite_backend": "none",
    "gsv_duration_rewrite_max_attempts": 1,
    "gsv_duration_rewrite_pre_candidate_count": None,
    "gsv_timefit_max_tempo": 1.18,
    "gsv_timefit_max_stretch": 1.08,
    "gsv_timefit_micro_max_sec": 2.0,
    "gsv_timefit_micro_max_tempo": 1.30,
    "gsv_timefit_long_min_sec": 7.0,
    "gsv_timefit_long_max_stretch": 1.15,
    "gsv_micro_segment_enabled": True,
    "gsv_micro_segment_max_sec": 1.2,
    "gsv_micro_segment_texture_max_sec": 0.7,
    "gsv_micro_segment_max_hangul_syllables": 4,
    "gsv_micro_segment_fallback_backend": "qwen",
    "gsv_micro_segment_keep_original_texture_enabled": True,
    "gsv_micro_segment_carrier_slice_enabled": True,
    "gsv_top_k": 8,
    "gsv_top_p": 0.9,
    "gsv_temperature": 0.7,
    "gsv_text_split_method": "cut0",
    "gsv_parallel_infer": False,
    "gsv_repetition_penalty": 1.25,
    "gsv_sample_steps": 32,
    "gsv_super_sampling": True,
    "gsv_min_chunk_length": 8,
    "gsv_numeric_cadence_periods_enabled": True,
    "gsv_numeric_cadence_min_values": 3,
    "gsv_numeric_sequence_qc_enabled": True,
    "gsv_numeric_sequence_qc_require_contiguous": True,
    "gsv_numeric_sequence_qc_failure_blocks_mix": True,
    "gsv_gpt_weights_policy": "base_for_korean",
    "gsv_sovits_weights_policy": "unchanged",
    "mix_allow_korean_timing_draft": False,
    "rvc_required": True,
    "rvc_backend": "command",
    "rvc_train_required": True,
    "rvc_train_backend": "command",
    "rvc_train_sample_rate": 48_000,
    "rvc_train_epochs": 100,
    "rvc_train_batch_size": 0,
    "rvc_train_preferred_chars_per_sec": 4.5,
    "rvc_train_max_chars_per_sec": 5.2,
    "rvc_train_min_clean_sec": 600.0,
    "rvc_train_augment_enabled": False,
    "rvc_train_augment_min_real_sec": 300.0,
    "rvc_train_timeout_sec": 43200.0,
    "rvc_train_preprocess_processes": 0,
    "rvc_train_f0_workers": 0,
    "rvc_train_feature_workers": 0,
    "rvc_train_save_every_epoch": 10,
    "rvc_train_reuse_intermediate_cache": True,
    "rvc_concurrency": 4,
    "rvc_batch_infer": True,
    "rvc_batch_size": 200,
    "rvc_batch_concurrency": 2,
    "rvc_failure_policy": "retry_then_error",
    "rvc_allow_pre_rvc_fallback": False,
}

app = typer.Typer(
    help=(
        "Local-first Japanese ASMR dubbing pipeline. You must own or have permission "
        "for source content, voice references, and distribution. This tool does not "
        "scrape, download from streaming sites, or bypass DRM."
    )
)


def _handle_error(exc: Exception) -> None:
    console.print(f"[red]{exc}[/red]")
    raise typer.Exit(code=1) from exc


def _backend_may_use_gpu(backend: str | None) -> bool:
    if backend is None:
        return False
    return backend.replace("-", "_").strip().lower() not in {"", "mock", "none"}


def _run_cli_stage(
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


def _parse_only_segment_ids(value: str | None) -> set[str] | None:
    if value is None:
        return None
    segment_ids = {part for part in re.split(r"[\s,]+", value.strip()) if part}
    if not segment_ids:
        raise ValueError("--only-segments must include at least one segment id.")
    return segment_ids


def _safe_run_name(input_path: Path, *, merge_parts: bool = False) -> str:
    stem = numbered_part_base_stem(input_path) if merge_parts else None
    stem = stem or input_path.stem
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "-", stem).strip("-._")
    return stem or "input"


def _default_full_project_dir(input_path: Path, *, merge_parts: bool = False) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return REPO_ROOT / "runs" / f"{timestamp}_{_safe_run_name(input_path, merge_parts=merge_parts)}"


def _default_voice_bank_project_dir() -> Path:
    return REPO_ROOT / "runs" / DEFAULT_VOICE_BANK_PROJECT_NAME


def _configure_local_model_cache() -> list[str]:
    """Point HF libraries at the repo-local cache when the user has not set one."""
    lines: list[str] = []
    hf_cache = REPO_ROOT / ".cache" / "huggingface"
    hf_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(hf_cache))
    os.environ.setdefault("HF_HUB_CACHE", str(hf_cache / "hub"))
    lines.append(f"HF cache: {hf_cache}")
    gemma_cache = hf_cache / "hub" / "models--google--gemma-4-E4B-it"
    lines.append(f"Gemma cache: {'found' if gemma_cache.exists() else 'missing'}")
    gsv_cache = REPO_ROOT / ".cache" / "gpt_sovits" / "GPT_SoVITS" / "pretrained_models"
    lines.append(f"GPT-SoVITS local weights: {'found' if gsv_cache.exists() else 'missing'}")
    gsv_api_candidates = [
        REPO_ROOT / ".cache/third_party/GPT-SoVITS/api_v2.py",
        REPO_ROOT / ".cache/third_party/GPT_SoVITS/api_v2.py",
        REPO_ROOT / ".cache/gpt_sovits/GPT_SoVITS/api_v2.py",
        REPO_ROOT / ".cache/gpt_sovits/GPT-SoVITS/api_v2.py",
    ]
    lines.append(
        "GPT-SoVITS api_v2: "
        f"{'found' if any(path.exists() for path in gsv_api_candidates) else 'missing'}"
    )
    rvc_assets = REPO_ROOT / ".cache" / "rvc" / "assets"
    lines.append(f"RVC local assets: {'found' if rvc_assets.exists() else 'missing'}")
    fish_root = REPO_ROOT / ".cache" / "tts_backends" / "fish-speech"
    fish_weights = fish_root / "checkpoints" / "s2-pro"
    lines.append(f"Fish Speech repo: {'found' if fish_root.exists() else 'missing'}")
    lines.append(f"Fish Speech s2-pro weights: {'found' if fish_weights.exists() else 'missing'}")
    cosy_root = REPO_ROOT / ".cache" / "tts_backends" / "CosyVoice"
    cosy_weights = cosy_root / "pretrained_models" / "CosyVoice2-0.5B"
    lines.append(f"CosyVoice repo: {'found' if cosy_root.exists() else 'missing'}")
    lines.append(f"CosyVoice2 weights: {'found' if cosy_weights.exists() else 'missing'}")
    llama_model = REPO_ROOT / DEFAULT_LLAMA_CPP_MODEL
    llama_mmproj = REPO_ROOT / DEFAULT_LLAMA_CPP_MMPROJ
    llama_cli = REPO_ROOT / DEFAULT_LLAMA_CPP_CLI
    lines.append(f"llama.cpp Gemma Q4: {'found' if llama_model.exists() else 'missing'}")
    lines.append(f"llama.cpp mmproj: {'found' if llama_mmproj.exists() else 'missing'}")
    lines.append(f"llama.cpp CLI: {'found' if llama_cli.exists() else 'missing'}")
    return lines


def _hf_repo_cache_name(model_id: str) -> str:
    return "models--" + model_id.replace("/", "--")


def _cached_hf_snapshot(model_id: str) -> Path | None:
    snapshot_root = REPO_ROOT / ".cache" / "huggingface" / "hub" / _hf_repo_cache_name(model_id) / "snapshots"
    if not snapshot_root.exists():
        return None
    snapshots = [path for path in snapshot_root.iterdir() if path.is_dir()]
    if not snapshots:
        return None
    with_config = [path for path in snapshots if (path / "config.yaml").exists()]
    return max(with_config or snapshots, key=lambda path: path.stat().st_mtime_ns).resolve()


def _discover_audio_inputs(audio_dir: Path) -> list[Path]:
    root = audio_dir.expanduser().resolve()
    if not root.exists():
        raise ValueError(f"Audio directory does not exist: {root}")
    if not root.is_dir():
        raise ValueError(f"Audio path is not a directory: {root}")
    return sorted(
        path.resolve()
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    )


def _folder_contains_supported_media(folder: Path) -> bool:
    return any(
        path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS for path in folder.rglob("*")
    )


def _discover_audio_work_dirs(audio_dir: Path) -> list[Path]:
    root = audio_dir.expanduser().resolve()
    if not root.exists():
        raise ValueError(f"Audio directory does not exist: {root}")
    if not root.is_dir():
        raise ValueError(f"Audio path is not a directory: {root}")
    return sorted(
        path.resolve()
        for path in root.iterdir()
        if path.is_dir() and not path.name.startswith(".") and _folder_contains_supported_media(path)
    )


def _default_full_audio_batch_dir() -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return REPO_ROOT / "runs" / f"{timestamp}_audio_full_real_batch"


def _remove_tree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _copy_export_result(export_path: str | None, result_dir: Path) -> list[str]:
    if not export_path:
        raise ValueError("Pipeline completed without an export artifact.")
    source = Path(export_path).expanduser().resolve()
    if not source.exists():
        raise ValueError(f"Pipeline export artifact is missing: {source}")
    result_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    if source.is_dir():
        for child in sorted(source.iterdir()):
            target = result_dir / child.name
            if child.is_dir():
                shutil.copytree(child, target, dirs_exist_ok=True)
            else:
                shutil.copy2(child, target)
            copied.append(str(target))
        return copied
    target = result_dir / source.name
    shutil.copy2(source, target)
    copied.append(str(target))
    return copied


def _resolve_existing_project_file(project_dir: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_dir / path
    path = path.resolve()
    if not path.is_file():
        return None
    return path


def _manual_review_file_refs(segment) -> list[tuple[str, str | None]]:
    refs: list[tuple[str, str | None]] = [
        ("audio_for_gemma", segment.audio_for_gemma),
        ("audio_for_mix", segment.audio_for_mix),
    ]
    if segment.tts is not None:
        refs.append(("tts_selected", segment.tts.selected_candidate_path))
        refs.extend(
            (f"tts_candidate_{candidate.candidate_index:02d}", candidate.output_path)
            for candidate in segment.tts.candidates
        )
    if segment.rvc is not None:
        refs.append(("rvc_input", segment.rvc.input_path))
        refs.append(("rvc_output", segment.rvc.output_path))
        refs.extend(
            (f"rvc_candidate_{index:02d}", path)
            for index, path in enumerate(segment.rvc.candidate_paths, start=1)
        )
    return refs


def _copy_manual_review_bundle(
    *,
    input_path: Path,
    project_dir: Path,
    review_dir: Path,
    manifest,
) -> list[str]:
    manual_segments = [
        segment for segment in manifest.segments if segment.status == "needs_manual_review"
    ]
    review_dir.mkdir(parents=True, exist_ok=True)
    copied_files: list[dict[str, str]] = []
    for segment in manual_segments:
        segment_dir = review_dir / "segments" / segment.id
        segment_dir.mkdir(parents=True, exist_ok=True)
        for label, source_value in _manual_review_file_refs(segment):
            source = _resolve_existing_project_file(project_dir, source_value)
            if source is None:
                continue
            suffix = source.suffix or ".bin"
            target = segment_dir / f"{label}{suffix}"
            counter = 2
            while target.exists():
                target = segment_dir / f"{label}_{counter}{suffix}"
                counter += 1
            shutil.copy2(source, target)
            copied_files.append(
                {
                    "segment_id": segment.id,
                    "label": label,
                    "source": str(source),
                    "path": str(target),
                }
            )
    write_json_atomic(
        review_dir / "manual_review_segments.json",
        {
            "input": str(input_path),
            "project": str(project_dir),
            "manual_review_count": len(manual_segments),
            "segments": [segment.model_dump(mode="json") for segment in manual_segments],
            "copied_files": copied_files,
            "stage_state": manifest.stage_state,
            "warnings": manifest.warnings,
        },
    )
    return [item["path"] for item in copied_files]


def _apply_personal_voice_bank_defaults(project_dir: Path) -> None:
    init_project(project_dir)
    cfg = load_project_config(project_dir)
    payload = cfg.model_dump(mode="json")
    payload.update(
        {
            "speaker_assignment_backend": "pyannote",
            "diarization_auto_download": True,
            "diarization_embedding_match_threshold": 0.75,
        }
    )
    for field_name, candidates in {
        "diarization_model_id": (
            "pyannote/speaker-diarization-3.1",
            "pyannote/speaker-diarization-community-1",
        ),
        "diarization_embedding_model_id": (
            "pyannote/wespeaker-voxceleb-resnet34-LM",
            "pyannote/embedding",
        ),
    }.items():
        for model_id in candidates:
            snapshot = _cached_hf_snapshot(model_id)
            if snapshot is not None:
                payload[field_name] = str(snapshot)
                break
    save_project_config(type(cfg).model_validate(payload), project_dir / "pipeline.yaml")


def _rvc_python() -> str:
    env_python = os.environ.get("ASMR_DUB_RVC_PYTHON")
    if env_python:
        return env_python
    for candidate in (
        REPO_ROOT / ".cache" / "rvc_venv" / "bin" / "python",
        REPO_ROOT / ".cache" / "rvc" / ".venv" / "bin" / "python",
        REPO_ROOT / ".cache" / "third_party" / "Retrieval-based-Voice-Conversion-WebUI" / ".venv" / "bin" / "python",
    ):
        if candidate.exists():
            return str(candidate)
    return sys.executable


def _rvc_experiment_name(source_path: Path) -> str:
    return f"asmr-{_safe_run_name(source_path).lower()}-speaker-1"


def _local_rvc_webui_defaults(project_dir: Path, source_path: Path | None = None) -> dict[str, object]:
    rvc_root = REPO_ROOT / ".cache" / "third_party" / "Retrieval-based-Voice-Conversion-WebUI"
    if not rvc_root.exists():
        return {}
    source_for_name = source_path or project_dir
    python = _rvc_python()
    train_wrapper = REPO_ROOT / "asmr_dub_pipeline" / "rvc" / "webui_train.py"
    infer_wrapper = REPO_ROOT / "asmr_dub_pipeline" / "rvc" / "webui_infer.py"
    batch_infer_wrapper = REPO_ROOT / "asmr_dub_pipeline" / "rvc" / "webui_batch_infer.py"
    return {
        "rvc_train_experiment_name": _rvc_experiment_name(source_for_name),
        "rvc_train_command": [
            python,
            str(train_wrapper),
            "--rvc-root",
            str(rvc_root),
            "--dataset",
            "{dataset}",
            "--experiment-name",
            "{experiment_name}",
            "--output-model",
            "{output_model}",
            "--output-index",
            "{output_index}",
            "--sample-rate",
            "{sample_rate}",
            "--device",
            "{device}",
            "--epochs",
            "{epochs}",
            "--save-every-epoch",
            "{save_every_epoch}",
            "--batch-size",
            "{batch_size}",
            "--processes",
            "{preprocess_processes}",
            "--f0-workers",
            "{f0_workers}",
            "--feature-workers",
            "{feature_workers}",
            "--reuse-intermediate-cache",
            "{reuse_intermediate_cache}",
        ],
        "rvc_command": [
            python,
            str(infer_wrapper),
            "--rvc-root",
            str(rvc_root),
            "--input",
            "{input}",
            "--output",
            "{output}",
            "--model",
            "{model}",
            "--index",
            "{index}",
            "--device",
            "{device}",
            "--f0-up-key",
            "{f0_up_key}",
            "--f0-method",
            "{f0_method}",
            "--index-rate",
            "{index_rate}",
            "--filter-radius",
            "{filter_radius}",
            "--resample-sr",
            "{resample_sr}",
            "--rms-mix-rate",
            "{rms_mix_rate}",
            "--protect",
            "{protect}",
        ],
        "rvc_batch_command": [
            python,
            str(batch_infer_wrapper),
            "--rvc-root",
            str(rvc_root),
            "--jobs",
            "{jobs}",
            "--results",
            "{results}",
            "--model",
            "{model}",
            "--index",
            "{index}",
            "--device",
            "{device}",
        ],
    }


def _apply_full_real_quality_preset(
    project_dir: Path,
    target_language: str,
    source_path: Path | None = None,
    *,
    few_shot: bool = True,
) -> None:
    init_project(project_dir)
    cfg = load_project_config(project_dir)
    payload = cfg.model_dump(mode="json")
    payload.update(FULL_REAL_QUALITY_PRESET)
    if few_shot:
        payload["gsv_few_shot_enabled"] = True
        payload["gsv_gpt_weights_policy"] = "auto"
        payload["gsv_sovits_weights_policy"] = "auto"
    else:
        payload["gsv_few_shot_enabled"] = False
        payload["gsv_gpt_weights_policy"] = "base_for_korean"
        payload["gsv_sovits_weights_policy"] = "unchanged"
    for key, value in _local_rvc_webui_defaults(project_dir, source_path).items():
        if key in {"rvc_train_command", "rvc_command", "rvc_batch_command"} and payload.get(key):
            continue
        payload[key] = value
    payload["target_language"] = target_language
    save_project_config(type(cfg).model_validate(payload), project_dir / "pipeline.yaml")


@app.command()
def init(project_dir: Path = typer.Argument(..., help="Project directory to initialize.")) -> None:
    """Create project folders and default config files."""
    try:
        init_project(project_dir.expanduser().resolve())
    except Exception as exc:
        _handle_error(exc)
    console.print(f"Initialized project at {project_dir}")


@app.command()
def inspect(
    input: Path = typer.Argument(..., help="Input media."),
    project: Path = typer.Option(..., "--project", "-p"),
    confirm_rights: bool = typer.Option(False, "--confirm-rights", help=RIGHTS_HELP),
) -> None:
    """Print input metadata using ffprobe when available."""
    try:
        require_confirmed_rights(confirm_rights, "inspect", input.expanduser().resolve())
        info = inspect_input(input.expanduser().resolve())
    except Exception as exc:
        _handle_error(exc)
    table = Table(title="Input metadata")
    table.add_column("Field")
    table.add_column("Value")
    for key, value in info.model_dump(mode="json", exclude={"raw"}).items():
        table.add_row(key, str(value))
    console.print(table)
    _ = project


@app.command()
def extract(
    input: Path = typer.Argument(..., help="Input media."),
    project: Path = typer.Option(..., "--project", "-p"),
    confirm_rights: bool = typer.Option(False, "--confirm-rights", help=RIGHTS_HELP),
    merge_parts: bool = typer.Option(
        False,
        "--merge-parts",
        help="Merge consecutive sibling files named <base>_1, <base>_2, ... into an audio-only source.",
    ),
) -> None:
    """Extract stereo 48 kHz and Gemma mono 16 kHz audio."""
    try:
        extract_step(
            input.expanduser().resolve(),
            project.expanduser().resolve(),
            confirm_rights,
            merge_parts=merge_parts,
        )
    except RightsError as exc:
        _handle_error(exc)
    except Exception as exc:
        _handle_error(exc)
    console.print(f"Extracted audio. Manifest: {manifest_path(project)}")


@app.command(name="separate-background")
def separate_background(
    project: Path = typer.Option(..., "--project", "-p"),
    confirm_rights: bool = typer.Option(False, "--confirm-rights", help=RIGHTS_HELP),
    force: bool = typer.Option(False, "--force", help="Re-run source separation even if stems exist."),
) -> None:
    """Separate source voice from background before segmentation, ASR, few-shot, and mix."""
    try:
        manifest = _run_cli_stage(
            "source-separation",
            True,
            source_separation_step,
            project.expanduser().resolve(),
            confirm_rights=confirm_rights,
            force=force,
        )
    except Exception as exc:
        _handle_error(exc)
    console.print(
        "Source separation: "
        f"{manifest.stage_state.get('source-separation', {}).get('status', 'unknown')}"
    )


@app.command()
def segment(
    project: Path = typer.Option(..., "--project", "-p"),
    confirm_rights: bool = typer.Option(False, "--confirm-rights", help=RIGHTS_HELP),
) -> None:
    """Finalize ASR-derived segments or create preliminary segment manifests."""
    try:
        manifest = segment_step(project.expanduser().resolve(), confirm_rights=confirm_rights)
    except Exception as exc:
        _handle_error(exc)
    console.print(f"Created {len(manifest.segments)} segment(s).")


@app.command()
def analyze(
    project: Path = typer.Option(..., "--project", "-p"),
    gemma_backend: str = typer.Option("mock", "--gemma-backend", help="mock|hf|http|llama_cpp"),
    model_id: str = typer.Option("google/gemma-4-E4B-it", "--model-id"),
    confirm_rights: bool = typer.Option(False, "--confirm-rights", help=RIGHTS_HELP),
) -> None:
    """Run Gemma-style segment analysis."""
    try:
        _run_cli_stage(
            "analyze",
            _backend_may_use_gpu(gemma_backend),
            analyze_step,
            project.expanduser().resolve(),
            gemma_backend,
            model_id,
            confirm_rights=confirm_rights,
        )
    except Exception as exc:
        _handle_error(exc)
    console.print("Analysis complete.")


@app.command(name="audio-style")
def audio_style_cmd(
    project: Path = typer.Option(..., "--project", "-p"),
    gemma_backend: str = typer.Option("mock", "--gemma-backend", help="mock|hf|http|llama_cpp|llama_server_audio"),
    model_id: str = typer.Option("google/gemma-4-E4B-it", "--model-id"),
    confirm_rights: bool = typer.Option(False, "--confirm-rights", help=RIGHTS_HELP),
    force: bool = typer.Option(False, "--force", help="Re-run audio style analysis even when segment results already exist."),
    scope: str = typer.Option("all", "--scope", help="all|speaker-suspicious"),
) -> None:
    """Analyze source audio style metadata for translated lanes."""
    try:
        _run_cli_stage(
            "audio-style",
            _backend_may_use_gpu(gemma_backend),
            audio_style_step,
            project.expanduser().resolve(),
            gemma_backend,
            model_id,
            confirm_rights=confirm_rights,
            force=force,
            scope=scope,
        )
    except Exception as exc:
        _handle_error(exc)
    console.print("Audio style analysis complete.")


@app.command()
def transcribe(
    project: Path = typer.Option(..., "--project", "-p"),
    asr_backend: str = typer.Option("faster_whisper", "--asr-backend", help="faster_whisper|qwen_asr|mock"),
    asr_preset: str | None = typer.Option(
        None,
        "--asr-preset",
        help="Runtime ASR preset: default|conservative|whisper|no_vad_repair.",
    ),
    asr_vad_off: bool = typer.Option(
        False,
        "--asr-vad-off",
        help="Disable VAD for the main ASR pass without editing pipeline.yaml.",
    ),
    asr_diagnostics: bool | None = typer.Option(
        None,
        "--asr-diagnostics/--no-asr-diagnostics",
        help="Write unified ASR diagnostics artifacts for this run.",
    ),
    asr_device: str | None = typer.Option(
        None,
        "--asr-device",
        help="Override faster-whisper device, e.g. auto|cuda|cpu.",
    ),
    asr_compute_type: str | None = typer.Option(
        None,
        "--asr-compute-type",
        help="Override faster-whisper compute type, e.g. default|float16|int8_float16.",
    ),
    asr_batched: bool | None = typer.Option(
        None,
        "--asr-batched/--no-asr-batched",
        help="Use faster-whisper BatchedInferencePipeline for higher GPU throughput.",
    ),
    asr_batch_size: int | None = typer.Option(
        None,
        "--asr-batch-size",
        help="Batch size for faster-whisper batched inference.",
    ),
    asr_repair: bool | None = typer.Option(
        None,
        "--asr-repair/--no-asr-repair",
        help="Enable or disable ASR repair retranscription passes for suspicious chunks.",
    ),
    asr_review: bool = typer.Option(
        False,
        "--asr-review",
        help="Use the configured Gemma audio+text model to review suspicious ASR candidates.",
    ),
    confirm_rights: bool = typer.Option(False, "--confirm-rights", help=RIGHTS_HELP),
) -> None:
    """Create segment-level source scripts with local ASR."""
    try:
        _run_cli_stage(
            "transcribe",
            _backend_may_use_gpu(asr_backend),
            transcribe_step,
            project.expanduser().resolve(),
            asr_backend,
            confirm_rights=confirm_rights,
            asr_review=True if asr_review else None,
            asr_preset=asr_preset,
            asr_vad_off=True if asr_vad_off else None,
            asr_diagnostics=asr_diagnostics,
            asr_device=asr_device,
            asr_compute_type=asr_compute_type,
            asr_batched_inference=asr_batched,
            asr_batch_size=asr_batch_size,
            asr_repair_enabled=asr_repair,
        )
    except Exception as exc:
        _handle_error(exc)
    console.print("Transcription complete.")


@app.command(name="translate-ko")
def translate_ko_cmd(
    project: Path = typer.Option(..., "--project", "-p"),
    gemma_text_backend: str = typer.Option(
        "llama_server",
        "--gemma-text-backend",
        help="llama_server|mock",
    ),
    confirm_rights: bool = typer.Option(False, "--confirm-rights", help=RIGHTS_HELP),
    force_retranslate: bool = typer.Option(
        False,
        "--force-retranslate",
        help="Ignore existing Korean translations and regenerate translate-ko artifacts.",
    ),
    retry_failed: bool = typer.Option(
        False,
        "--retry-failed",
        help="Retry segments currently marked needs_manual_review or failed when source text exists.",
    ),
    repair_only: bool = typer.Option(
        False,
        "--repair-only",
        help="Only run deterministic translate-ko repairs and diagnostics on existing translations.",
    ),
    force_retranslate_failed: bool = typer.Option(
        False,
        "--force-retranslate-failed",
        help="When used with --retry-failed, discard existing failed translations before retrying.",
    ),
) -> None:
    """Translate source scripts to Korean with text-only Gemma."""
    try:
        _run_cli_stage(
            "translate-ko",
            _backend_may_use_gpu(gemma_text_backend),
            translate_ko_step,
            project.expanduser().resolve(),
            gemma_text_backend,
            confirm_rights=confirm_rights,
            force_retranslate=force_retranslate,
            retry_failed=retry_failed,
            repair_only=repair_only,
            force_retranslate_failed=force_retranslate_failed,
        )
    except Exception as exc:
        _handle_error(exc)
    console.print("Korean translation complete.")


@app.command(name="korean-script")
def korean_script_cmd(
    project: Path = typer.Option(..., "--project", "-p"),
    confirm_rights: bool = typer.Option(False, "--confirm-rights", help=RIGHTS_HELP),
    only_segments: str | None = typer.Option(
        None,
        "--only-segments",
        help="Comma- or whitespace-separated segment IDs to script, e.g. seg_0001,seg_0020.",
    ),
) -> None:
    """Build Korean TTS script metadata from completed translate-ko output."""
    try:
        korean_script_step(
            project.expanduser().resolve(),
            confirm_rights=confirm_rights,
            only_segment_ids=_parse_only_segment_ids(only_segments),
        )
    except Exception as exc:
        _handle_error(exc)
    console.print("Korean script complete.")


@app.command(name="script")
def script_cmd(
    project: Path = typer.Option(..., "--project", "-p"),
    gemma_backend: str = typer.Option("mock", "--gemma-backend", help="mock|hf|http|llama_cpp"),
    confirm_rights: bool = typer.Option(False, "--confirm-rights", help=RIGHTS_HELP),
) -> None:
    """Generate Japanese ASMR script metadata and normalize TTS text."""
    try:
        _run_cli_stage(
            "script",
            _backend_may_use_gpu(gemma_backend),
            script_step,
            project.expanduser().resolve(),
            gemma_backend,
            confirm_rights=confirm_rights,
        )
    except Exception as exc:
        _handle_error(exc)
    console.print("Script generation complete.")


@app.command()
def synth(
    project: Path = typer.Option(..., "--project", "-p"),
    gsv_url: str | None = typer.Option(None, "--gsv-url"),
    refs: Path = typer.Option(Path("refs/refs.json"), "--refs"),
    mock: bool = typer.Option(False, "--mock", help="Generate deterministic synthetic WAV files."),
    confirm_rights: bool = typer.Option(False, "--confirm-rights", help=RIGHTS_HELP),
    gpt_weights: str | None = typer.Option(None, "--gpt-weights", help="Optional GPT weights path for api_v2."),
    sovits_weights: str | None = typer.Option(None, "--sovits-weights", help="Optional SoVITS weights path for api_v2."),
    use_trained_gpt: bool = typer.Option(False, "--use-trained-gpt", help=TRAINED_GPT_HELP),
    auto_gsv_server: bool = typer.Option(
        False,
        "--auto-gsv-server/--no-auto-gsv-server",
        help="Start a local GPT-SoVITS api_v2 server if gsv_url is not already HTTP-ready.",
    ),
    gsv_server_command: str | None = typer.Option(
        None,
        "--gsv-server-command",
        help="Shell-style command used when --auto-gsv-server needs to start api_v2.",
    ),
    retry_failed: bool = typer.Option(
        False,
        "--retry-failed",
        help="Retry segments already marked failed when script metadata exists.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Discard existing synthesized TTS metadata and regenerate selected segments.",
    ),
    only_segments: str | None = typer.Option(
        None,
        "--only-segments",
        help="Comma- or whitespace-separated segment IDs to synthesize, e.g. seg_0001,seg_0020.",
    ),
) -> None:
    """Generate TTS candidates per segment."""
    try:
        _run_cli_stage(
            "synth",
            not mock,
            synth_step,
            project.expanduser().resolve(),
            gsv_url,
            refs,
            mock=mock,
            confirm_rights=confirm_rights,
            gpt_weights_path=gpt_weights,
            sovits_weights_path=sovits_weights,
            use_trained_gpt=use_trained_gpt,
            auto_gsv_server=auto_gsv_server,
            gsv_server_command=gsv_server_command,
            retry_failed=retry_failed,
            force=force,
            render_countdowns=False,
            only_segment_ids=_parse_only_segment_ids(only_segments),
        )
    except RightsError as exc:
        _handle_error(exc)
    except Exception as exc:
        _handle_error(exc)
    console.print("Synthesis complete.")


@app.command(name="countdown-synth")
def countdown_synth(
    project: Path = typer.Option(..., "--project", "-p"),
    gsv_url: str | None = typer.Option(None, "--gsv-url"),
    refs: Path = typer.Option(Path("refs/refs.json"), "--refs"),
    mock: bool = typer.Option(False, "--mock", help="Generate deterministic synthetic countdown WAV files."),
    confirm_rights: bool = typer.Option(False, "--confirm-rights", help=RIGHTS_HELP),
    gpt_weights: str | None = typer.Option(None, "--gpt-weights", help="Optional GPT weights path for api_v2."),
    sovits_weights: str | None = typer.Option(None, "--sovits-weights", help="Optional SoVITS weights path for api_v2."),
    use_trained_gpt: bool = typer.Option(False, "--use-trained-gpt", help=TRAINED_GPT_HELP),
    auto_gsv_server: bool = typer.Option(
        False,
        "--auto-gsv-server/--no-auto-gsv-server",
        help="Start a local GPT-SoVITS api_v2 server if gsv_url is not already HTTP-ready.",
    ),
    gsv_server_command: str | None = typer.Option(
        None,
        "--gsv-server-command",
        help="Shell-style command used when --auto-gsv-server needs to start api_v2.",
    ),
    retry_failed: bool = typer.Option(
        False,
        "--retry-failed",
        help="Retry countdown segments already marked failed when script metadata exists.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Discard existing countdown TTS metadata and regenerate selected countdown segments.",
    ),
    only_segments: str | None = typer.Option(
        None,
        "--only-segments",
        help="Comma- or whitespace-separated segment IDs to synthesize, e.g. seg_0001,seg_0020.",
    ),
) -> None:
    """Generate only countdown TTS tokens and assemble countdown segment audio."""
    try:
        _run_cli_stage(
            "countdown-synth",
            not mock,
            countdown_synth_step,
            project.expanduser().resolve(),
            gsv_url,
            refs,
            mock=mock,
            confirm_rights=confirm_rights,
            gpt_weights_path=gpt_weights,
            sovits_weights_path=sovits_weights,
            use_trained_gpt=use_trained_gpt,
            auto_gsv_server=auto_gsv_server,
            gsv_server_command=gsv_server_command,
            retry_failed=retry_failed,
            force=force,
            only_segment_ids=_parse_only_segment_ids(only_segments),
        )
    except RightsError as exc:
        _handle_error(exc)
    except Exception as exc:
        _handle_error(exc)
    console.print("Countdown synthesis complete.")


@app.command(name="synth-qwen")
def synth_qwen(
    project: Path = typer.Option(..., "--project", "-p"),
    refs: Path = typer.Option(Path("refs/refs.json"), "--refs"),
    confirm_rights: bool = typer.Option(False, "--confirm-rights", help=RIGHTS_HELP),
    model_id: str | None = typer.Option(None, "--model-id", help="Qwen3-TTS model id or local path."),
    candidate_count: int | None = typer.Option(
        None,
        "--candidate-count",
        min=1,
        max=8,
        help="Number of Qwen candidates per segment. Defaults to qwen_tts_candidate_count.",
    ),
    candidate_batch_size: int | None = typer.Option(
        None,
        "--candidate-batch-size",
        min=1,
        max=8,
        help="Number of Qwen candidates to synthesize in one GPU batch.",
    ),
    segment_batch_size: int | None = typer.Option(
        None,
        "--segment-batch-size",
        min=1,
        max=16,
        help="Number of scripted segments to synthesize in one Qwen GPU batch.",
    ),
    target_vram_gb: float | None = typer.Option(
        None,
        "--target-vram-gb",
        min=0.1,
        help="Approximate CUDA VRAM budget for Qwen TTS. Defaults to qwen_tts_target_vram_gb.",
    ),
    promote: bool = typer.Option(
        False,
        "--promote/--compare-only",
        help="Replace selected TTS with the best Qwen candidate and invalidate downstream RVC/QC/mix.",
    ),
    allow_download: bool = typer.Option(
        False,
        "--allow-download/--local-files-only",
        help="Allow qwen-tts to resolve a remote Hugging Face model id instead of requiring repo-local cache.",
    ),
) -> None:
    """Generate Qwen TTS candidates from an existing scripted manifest."""
    try:
        manifest = _run_cli_stage(
            "synth-qwen",
            True,
            synth_qwen_step,
            project.expanduser().resolve(),
            refs,
            confirm_rights=confirm_rights,
            model_id=model_id,
            candidate_count=candidate_count,
            candidate_batch_size=candidate_batch_size,
            segment_batch_size=segment_batch_size,
            target_vram_gb=target_vram_gb,
            promote=promote,
            local_files_only=False if allow_download else None,
        )
    except RightsError as exc:
        _handle_error(exc)
    except Exception as exc:
        _handle_error(exc)
    console.print(f"Qwen synthesis complete: {manifest.artifacts.get('qwen_tts')}")


@app.command(name="synth-fish")
def synth_fish(
    project: Path = typer.Option(..., "--project", "-p"),
    refs: Path = typer.Option(Path("refs/refs.json"), "--refs"),
    confirm_rights: bool = typer.Option(False, "--confirm-rights", help=RIGHTS_HELP),
    base_url: str | None = typer.Option(None, "--base-url", help="Fish Speech API server URL."),
    candidate_count: int | None = typer.Option(
        None,
        "--candidate-count",
        min=1,
        max=8,
        help="Number of Fish Speech candidates per segment. Defaults to fish_tts_candidate_count.",
    ),
    promote: bool = typer.Option(
        False,
        "--promote/--compare-only",
        help="Replace selected TTS with the best Fish Speech candidate and invalidate downstream RVC/QC/mix.",
    ),
) -> None:
    """Generate experimental Fish Speech candidates from an existing scripted manifest."""
    try:
        manifest = synth_experimental_tts_step(
            project.expanduser().resolve(),
            refs,
            backend="fish",
            confirm_rights=confirm_rights,
            base_url=base_url,
            candidate_count=candidate_count,
            promote=promote,
        )
    except RightsError as exc:
        _handle_error(exc)
    except Exception as exc:
        _handle_error(exc)
    console.print(f"Fish synthesis complete: {manifest.artifacts.get('fish_tts')}")


@app.command(name="synth-cosyvoice")
def synth_cosyvoice(
    project: Path = typer.Option(..., "--project", "-p"),
    refs: Path = typer.Option(Path("refs/refs.json"), "--refs"),
    confirm_rights: bool = typer.Option(False, "--confirm-rights", help=RIGHTS_HELP),
    base_url: str | None = typer.Option(None, "--base-url", help="CosyVoice FastAPI server URL."),
    candidate_count: int | None = typer.Option(
        None,
        "--candidate-count",
        min=1,
        max=8,
        help="Number of CosyVoice candidates per segment. Defaults to cosyvoice_candidate_count.",
    ),
    promote: bool = typer.Option(
        False,
        "--promote/--compare-only",
        help="Replace selected TTS with the best CosyVoice candidate and invalidate downstream RVC/QC/mix.",
    ),
) -> None:
    """Generate experimental CosyVoice candidates from an existing scripted manifest."""
    try:
        manifest = synth_experimental_tts_step(
            project.expanduser().resolve(),
            refs,
            backend="cosyvoice",
            confirm_rights=confirm_rights,
            base_url=base_url,
            candidate_count=candidate_count,
            promote=promote,
        )
    except RightsError as exc:
        _handle_error(exc)
    except Exception as exc:
        _handle_error(exc)
    console.print(f"CosyVoice synthesis complete: {manifest.artifacts.get('cosyvoice_tts')}")


@app.command(name="regenerate")
def regenerate(
    project: Path = typer.Option(..., "--project", "-p"),
    refs: Path = typer.Option(Path("refs/refs.json"), "--refs"),
    tts_backend: str = typer.Option("gpt-sovits", "--tts-backend", help="gpt-sovits|qwen|fish|cosyvoice|mock"),
    gemma_backend: str = typer.Option("mock", "--gemma-backend", help="mock|hf|http|llama_cpp"),
    confirm_rights: bool = typer.Option(False, "--confirm-rights", help=RIGHTS_HELP),
    gsv_url: str | None = typer.Option(None, "--gsv-url"),
    gpt_weights: str | None = typer.Option(None, "--gpt-weights", help="Optional GPT weights path for api_v2."),
    sovits_weights: str | None = typer.Option(None, "--sovits-weights", help="Optional SoVITS weights path for api_v2."),
    use_trained_gpt: bool = typer.Option(False, "--use-trained-gpt", help=TRAINED_GPT_HELP),
    auto_gsv_server: bool = typer.Option(
        False,
        "--auto-gsv-server/--no-auto-gsv-server",
        help="Start a local GPT-SoVITS api_v2 server if gsv_url is not already HTTP-ready.",
    ),
    gsv_server_command: str | None = typer.Option(
        None,
        "--gsv-server-command",
        help="Shell-style command used when --auto-gsv-server needs to start api_v2.",
    ),
    qwen_model_id: str | None = typer.Option(None, "--qwen-model-id", help="Qwen3-TTS model id or local path."),
    qwen_candidate_count: int | None = typer.Option(
        None,
        "--qwen-candidate-count",
        min=1,
        max=8,
        help="Number of Qwen candidates per regenerated segment.",
    ),
    qwen_allow_download: bool = typer.Option(
        False,
        "--qwen-allow-download/--qwen-local-files-only",
        help="Allow qwen-tts to resolve a remote Hugging Face model id.",
    ),
    experimental_tts_base_url: str | None = typer.Option(
        None,
        "--experimental-tts-base-url",
        help="Override Fish Speech or CosyVoice local server URL during regeneration.",
    ),
    experimental_tts_candidate_count: int | None = typer.Option(
        None,
        "--experimental-tts-candidate-count",
        min=1,
        max=8,
        help="Override Fish Speech or CosyVoice candidate count during regeneration.",
    ),
) -> None:
    """Regenerate QC-flagged segments, then rerun RVC and QC before mix."""
    try:
        manifest = _run_cli_stage(
            "regenerate",
            _backend_may_use_gpu(tts_backend) or _backend_may_use_gpu(gemma_backend),
            regenerate_needs_step,
            project.expanduser().resolve(),
            refs_path=refs,
            confirm_rights=confirm_rights,
            gemma_backend=gemma_backend,
            tts_backend=tts_backend,
            gsv_url=gsv_url,
            gpt_weights_path=gpt_weights,
            sovits_weights_path=sovits_weights,
            use_trained_gpt=use_trained_gpt,
            auto_gsv_server=auto_gsv_server,
            gsv_server_command=gsv_server_command,
            qwen_model_id=qwen_model_id,
            qwen_candidate_count=qwen_candidate_count,
            qwen_local_files_only=False if qwen_allow_download else None,
            experimental_tts_base_url=experimental_tts_base_url,
            experimental_tts_candidate_count=experimental_tts_candidate_count,
        )
    except RightsError as exc:
        _handle_error(exc)
    except Exception as exc:
        _handle_error(exc)
    console.print(f"Regeneration complete: {manifest.stage_state.get('regenerate')}")


@app.command(name="auto-repair")
def auto_repair(
    project: Path = typer.Option(..., "--project", "-p"),
    refs: Path = typer.Option(Path("refs/refs.json"), "--refs"),
    confirm_rights: bool = typer.Option(False, "--confirm-rights", help=RIGHTS_HELP),
    gemma_backend: str = typer.Option("mock", "--gemma-backend", help="mock|hf|http|llama_cpp"),
    tts_backend: str = typer.Option("gpt-sovits", "--tts-backend", help="gpt-sovits|qwen|mock"),
    max_attempts: int | None = typer.Option(
        None,
        "--max-attempts",
        min=1,
        max=12,
        help="Override auto_repair_max_attempts for this run.",
    ),
    only_segments: str | None = typer.Option(
        None,
        "--only-segments",
        help="Comma or whitespace separated segment ids to consider.",
    ),
    plan_only: bool = typer.Option(
        False,
        "--plan-only/--apply",
        help="Write the auto-repair summary without invoking downstream repair stages.",
    ),
    gsv_url: str | None = typer.Option(None, "--gsv-url"),
    gpt_weights: str | None = typer.Option(None, "--gpt-weights", help="Optional GPT weights path for api_v2."),
    sovits_weights: str | None = typer.Option(None, "--sovits-weights", help="Optional SoVITS weights path for api_v2."),
    use_trained_gpt: bool = typer.Option(False, "--use-trained-gpt", help=TRAINED_GPT_HELP),
    auto_gsv_server: bool = typer.Option(
        False,
        "--auto-gsv-server/--no-auto-gsv-server",
        help="Start a local GPT-SoVITS api_v2 server if needed.",
    ),
    gsv_server_command: str | None = typer.Option(
        None,
        "--gsv-server-command",
        help="Shell-style command used when --auto-gsv-server needs to start api_v2.",
    ),
) -> None:
    """Repair Korean QC failures that have deterministic recovery routes."""
    try:
        segment_ids = _parse_only_segment_ids(only_segments)
        manifest = _run_cli_stage(
            "auto-repair",
            _backend_may_use_gpu(gemma_backend) or _backend_may_use_gpu(tts_backend),
            auto_repair_step,
            project.expanduser().resolve(),
            refs_path=refs,
            confirm_rights=confirm_rights,
            max_attempts=max_attempts,
            plan_only=plan_only,
            only_segment_ids=segment_ids,
            gemma_backend=gemma_backend,
            tts_backend=tts_backend,
            gsv_url=gsv_url,
            gpt_weights_path=gpt_weights,
            sovits_weights_path=sovits_weights,
            use_trained_gpt=use_trained_gpt,
            auto_gsv_server=auto_gsv_server,
            gsv_server_command=gsv_server_command,
        )
    except RightsError as exc:
        _handle_error(exc)
    except Exception as exc:
        _handle_error(exc)
    console.print(f"Auto-repair complete: {manifest.stage_state.get('auto-repair')}")


@app.command(name="train-rvc")
def train_rvc(
    project: Path = typer.Option(..., "--project", "-p"),
    confirm_rights: bool = typer.Option(False, "--confirm-rights", help=RIGHTS_HELP),
    force: bool = typer.Option(False, "--force", help="Re-run RVC training even if artifacts exist."),
) -> None:
    """Train the required RVC voice model from source-derived segment audio."""
    try:
        manifest = _run_cli_stage(
            "train-rvc",
            True,
            rvc_train_step,
            project.expanduser().resolve(),
            confirm_rights=confirm_rights,
            force=force,
        )
    except RightsError as exc:
        _handle_error(exc)
    except Exception as exc:
        _handle_error(exc)
    console.print(f"RVC training complete: {manifest.artifacts.get('rvc_train_manifest')}")


@app.command(name="prepare-refs")
def prepare_refs(
    project: Path = typer.Option(..., "--project", "-p"),
    refs: Path = typer.Option(Path("refs/refs.json"), "--refs"),
    confirm_rights: bool = typer.Option(False, "--confirm-rights", help=RIGHTS_HELP),
) -> None:
    """Prepare source-derived GPT-SoVITS reference clips."""
    try:
        manifest = prepare_source_voice_refs_step(
            project.expanduser().resolve(),
            refs_path=refs,
            confirm_rights=confirm_rights,
        )
    except RightsError as exc:
        _handle_error(exc)
    except Exception as exc:
        _handle_error(exc)
    console.print(f"Source voice refs complete: {manifest.artifacts.get('source_voice_refs')}")


@app.command()
def rvc(
    project: Path = typer.Option(..., "--project", "-p"),
    confirm_rights: bool = typer.Option(False, "--confirm-rights", help=RIGHTS_HELP),
    force: bool = typer.Option(False, "--force", help="Re-run RVC even if candidate outputs exist."),
    retry_failed: bool = typer.Option(
        False,
        "--retry-failed",
        help="Retry only segments currently marked failed when selected TTS audio exists.",
    ),
    retry: str | None = typer.Option(
        None,
        "--retry",
        help="Retry mode alias. Currently supports: failed.",
    ),
) -> None:
    """Run mandatory RVC timbre correction on selected TTS outputs."""
    try:
        retry_failed_mode = retry_failed
        if retry is not None:
            retry_mode = retry.strip().lower().replace("_", "-")
            if retry_mode != "failed":
                raise ValueError("--retry currently supports only 'failed'.")
            retry_failed_mode = True
        manifest = _run_cli_stage(
            "rvc",
            True,
            rvc_step,
            project.expanduser().resolve(),
            confirm_rights=confirm_rights,
            force=force,
            retry_failed=retry_failed_mode,
        )
    except RightsError as exc:
        _handle_error(exc)
    except Exception as exc:
        _handle_error(exc)
    console.print(f"RVC complete: {manifest.artifacts.get('rvc_manifest')}")


@app.command(name="rvc-validate")
def rvc_validate(project: Path = typer.Option(..., "--project", "-p")) -> None:
    """Validate the configured RVC command backend without running conversion."""
    project_dir = project.expanduser().resolve()
    try:
        cfg = load_project_config(project_dir)
        validate_rvc_training_config(project_dir, cfg, real=cfg.rvc_train_backend == "command")
        validate_rvc_config(
            project_dir,
            cfg,
            real=cfg.rvc_backend == "command",
            allow_trained_artifact=cfg.rvc_train_required,
        )
    except Exception as exc:
        _handle_error(exc)
    console.print("RVC configuration is valid.")


@app.command(name="voice-bank-build")
def voice_bank_build(
    inputs: list[Path] = typer.Argument(..., help="Input media files used to build speaker voice models."),
    project: Path = typer.Option(..., "--project", "-p"),
    confirm_rights: bool = typer.Option(False, "--confirm-rights", help=RIGHTS_HELP),
    mock: bool = typer.Option(False, "--mock", help="Use mock diarization and placeholder voice model artifacts."),
    diarization_backend: str | None = typer.Option(
        None,
        "--diarization-backend",
        help="pyannote|mock. Defaults to pyannote unless --mock is set.",
    ),
    force: bool = typer.Option(False, "--force", help="Rebuild the project voice_bank directory."),
) -> None:
    """Build a project-local voice bank before dubbing runs."""
    try:
        for line in _configure_local_model_cache():
            console.print(f"[dim]{line}[/dim]")
        project_dir = project.expanduser().resolve()
        _apply_personal_voice_bank_defaults(project_dir)
        backend = "mock" if mock else diarization_backend or "pyannote"
        bank = _run_cli_stage(
            "voice-bank-build",
            not mock,
            build_voice_bank,
            [path.expanduser().resolve() for path in inputs],
            project_dir,
            confirm_rights=confirm_rights,
            backend_kind=backend,
            mock_training=mock,
            force=force,
        )
    except RightsError as exc:
        _handle_error(exc)
    except Exception as exc:
        _handle_error(exc)
    console.print(f"Voice bank complete: {len(bank.speakers)} speaker(s).")


@app.command(name="voice-bank-build-audio")
def voice_bank_build_audio(
    audio_dir: Path = typer.Option(Path("audio"), "--audio-dir", help="Directory of local media files."),
    project: Path | None = typer.Option(
        None,
        "--project",
        "-p",
        help="Project directory. Defaults to runs/voice_bank_all.",
    ),
    confirm_rights: bool = typer.Option(False, "--confirm-rights", help=RIGHTS_HELP),
    mock: bool = typer.Option(False, "--mock", help="Use mock diarization and placeholder voice model artifacts."),
    diarization_backend: str | None = typer.Option(
        None,
        "--diarization-backend",
        help="pyannote|mock. Defaults to pyannote unless --mock is set.",
    ),
    force: bool = typer.Option(False, "--force", help="Rebuild the project voice_bank directory."),
) -> None:
    """Build a voice bank from every supported media file in ./audio."""
    try:
        for line in _configure_local_model_cache():
            console.print(f"[dim]{line}[/dim]")
        project_dir = project.expanduser().resolve() if project else _default_voice_bank_project_dir()
        _apply_personal_voice_bank_defaults(project_dir)
        inputs = _discover_audio_inputs(audio_dir)
        if not inputs:
            raise ValueError(f"No supported audio/video files found in {audio_dir}.")
        backend = "mock" if mock else diarization_backend or "pyannote"
        console.print(f"[cyan]voice-bank[/cyan] discovered {len(inputs)} file(s) in {audio_dir}")
        bank = _run_cli_stage(
            "voice-bank-build-audio",
            not mock,
            build_voice_bank,
            inputs,
            project_dir,
            confirm_rights=confirm_rights,
            backend_kind=backend,
            mock_training=mock,
            force=force,
        )
    except RightsError as exc:
        _handle_error(exc)
    except Exception as exc:
        _handle_error(exc)
    console.print(f"Voice bank complete: {len(bank.speakers)} speaker(s). Project: {project_dir}")


@app.command(name="train-gsv")
def train_gsv(
    project: Path = typer.Option(..., "--project", "-p"),
    confirm_rights: bool = typer.Option(False, "--confirm-rights", help=RIGHTS_HELP),
    force: bool = typer.Option(False, "--force", help="Re-run GPT-SoVITS few-shot training."),
    gsv_url: str | None = typer.Option(None, "--gsv-url"),
    gsv_server_command: str | None = typer.Option(
        None,
        "--gsv-server-command",
        help="Shell-style command pointing at the local GPT-SoVITS api_v2.py checkout.",
    ),
) -> None:
    """Prepare source-derived data and fine-tune GPT-SoVITS weights."""
    try:
        manifest = _run_cli_stage(
            "train-gsv",
            True,
            gsv_few_shot_step,
            project.expanduser().resolve(),
            confirm_rights=confirm_rights,
            force=force,
            gsv_url=gsv_url,
            gsv_server_command=gsv_server_command,
        )
    except RightsError as exc:
        _handle_error(exc)
    except Exception as exc:
        _handle_error(exc)
    console.print(f"GPT-SoVITS few-shot complete: {manifest.artifacts.get('gsv_few_shot_gpt_weights')}")


@app.command(name="source-speakers")
def source_speakers(
    project: Path = typer.Option(..., "--project", "-p"),
    backend: str = typer.Option("pyannote", "--backend", help="pyannote|mock"),
    jobs: int = typer.Option(
        4,
        "--jobs",
        min=1,
        help="Parallel track diarization workers for folder inputs. Use 1 to disable parallel part diarization.",
    ),
    confirm_rights: bool = typer.Option(False, "--confirm-rights", help=RIGHTS_HELP),
) -> None:
    """Assign project-local speaker IDs from source diarization."""
    try:
        manifest = _run_cli_stage(
            "source-speakers",
            _backend_may_use_gpu(backend),
            source_speakers_step,
            project.expanduser().resolve(),
            backend_kind=backend,
            confirm_rights=confirm_rights,
            jobs=jobs,
        )
    except RightsError as exc:
        _handle_error(exc)
    except Exception as exc:
        _handle_error(exc)
    console.print(f"Source speaker assignment complete: {manifest.stage_state.get('source-speakers')}")


@app.command()
def qc(
    project: Path = typer.Option(..., "--project", "-p"),
    gemma_backend: str = typer.Option("mock", "--gemma-backend", help="mock|hf|http|llama_cpp"),
    confirm_rights: bool = typer.Option(False, "--confirm-rights", help=RIGHTS_HELP),
) -> None:
    """Run audio and Gemma-style QC."""
    try:
        _run_cli_stage(
            "qc",
            _backend_may_use_gpu(gemma_backend),
            qc_step,
            project.expanduser().resolve(),
            gemma_backend,
            confirm_rights=confirm_rights,
        )
    except Exception as exc:
        _handle_error(exc)
    console.print("QC complete.")


@app.command()
def mix(
    project: Path = typer.Option(..., "--project", "-p"),
    confirm_rights: bool = typer.Option(False, "--confirm-rights", help=RIGHTS_HELP),
) -> None:
    """Create dialogue stem and final mixed audio."""
    try:
        mix_step(project.expanduser().resolve(), confirm_rights)
    except RightsError as exc:
        _handle_error(exc)
    except Exception as exc:
        _handle_error(exc)
    console.print("Mix complete.")


@app.command()
def export(
    input: Path = typer.Argument(..., help="Original input media."),
    project: Path = typer.Option(..., "--project", "-p"),
    confirm_rights: bool = typer.Option(False, "--confirm-rights", help=RIGHTS_HELP),
) -> None:
    """Mux final audio back into video or export final WAV."""
    try:
        manifest = export_step(input.expanduser().resolve(), project.expanduser().resolve(), confirm_rights)
    except RightsError as exc:
        _handle_error(exc)
    except Exception as exc:
        _handle_error(exc)
    console.print(f"Exported: {manifest.artifacts.get('export')}")


@app.command()
def run(
    input: Path = typer.Argument(..., help="Input media."),
    project: Path = typer.Option(..., "--project", "-p"),
    confirm_rights: bool = typer.Option(False, "--confirm-rights", help=RIGHTS_HELP),
    merge_parts: bool = typer.Option(
        False,
        "--merge-parts",
        help="Merge consecutive sibling files named <base>_1, <base>_2, ... into an audio-only source.",
    ),
    mock: bool = typer.Option(False, "--mock", help="Use mock Gemma and mock TTS."),
    gemma_backend: str = typer.Option("mock", "--gemma-backend", help="mock|hf|http|llama_cpp"),
    ko_qc_backend: str | None = typer.Option(
        None,
        "--ko-qc-backend",
        help="Korean QC backend override: auto|hf|http|llama_cpp|mock.",
    ),
    target_language: str = typer.Option("ko", "--target-language", help="Output TTS language. Currently supports ko/kr."),
    auto_repair: bool = typer.Option(
        True,
        "--auto-repair/--no-auto-repair",
        help="Run the Korean QC auto-repair loop after QC.",
    ),
    auto_repair_max_rounds: int | None = typer.Option(
        None,
        "--auto-repair-max-rounds",
        min=0,
        max=12,
        help="Override the auto-repair round budget.",
    ),
    micro_segments: bool = typer.Option(
        True,
        "--micro-segments/--no-micro-segments",
        help="Use dedicated GPT-SoVITS micro-segment routing.",
    ),
    gsv_url: str | None = typer.Option(None, "--gsv-url"),
    refs: Path = typer.Option(Path("refs/refs.json"), "--refs"),
    gpt_weights: str | None = typer.Option(None, "--gpt-weights", help="Optional GPT weights path for api_v2."),
    sovits_weights: str | None = typer.Option(None, "--sovits-weights", help="Optional SoVITS weights path for api_v2."),
    use_trained_gpt: bool = typer.Option(False, "--use-trained-gpt", help=TRAINED_GPT_HELP),
    auto_gsv_server: bool = typer.Option(
        False,
        "--auto-gsv-server/--no-auto-gsv-server",
        help="Start a local GPT-SoVITS api_v2 server if gsv_url is not already HTTP-ready.",
    ),
    gsv_server_command: str | None = typer.Option(
        None,
        "--gsv-server-command",
        help="Shell-style command used when --auto-gsv-server needs to start api_v2.",
    ),
    few_shot: bool = typer.Option(
        False,
        "--few-shot/--zero-shot",
        help="Fine-tune GPT-SoVITS from source segments before synthesis. Default is zero-shot.",
    ),
    gsv_few_shot_force: bool = typer.Option(
        False,
        "--force-few-shot",
        help="Re-run GPT-SoVITS few-shot training even when cached weights match.",
    ),
    rvc_train_force: bool = typer.Option(
        False,
        "--force-rvc-train",
        help="Re-run RVC training even when model artifacts exist.",
    ),
    voice_bank: Path | None = typer.Option(
        None,
        "--voice-bank",
        help="Voice bank manifest path. Defaults to project voice_bank/voice_bank_manifest.json when required.",
    ),
    require_voice_bank: bool = typer.Option(
        False,
        "--require-voice-bank",
        help="Require speaker assignment and per-speaker SoVITS/RVC models before synthesis.",
    ),
) -> None:
    """Run the configured end-to-end lane, then synth, RVC, QC, mix, and export."""
    if not confirm_rights:
        _handle_error(RightsError(RIGHTS_MESSAGE))
    try:
        if mock:
            gemma_backend = "mock"
        manifest = run_pipeline(
            input.expanduser().resolve(),
            project.expanduser().resolve(),
            confirm_rights=confirm_rights,
            mock=mock,
            gemma_backend=gemma_backend,
            ko_qc_backend=ko_qc_backend,
            target_language=target_language,
            auto_repair=auto_repair,
            auto_repair_max_rounds=auto_repair_max_rounds,
            micro_segments=micro_segments,
            gsv_url=gsv_url,
            refs_path=refs,
            gpt_weights_path=gpt_weights,
            sovits_weights_path=sovits_weights,
            auto_gsv_server=auto_gsv_server,
            gsv_server_command=gsv_server_command,
            use_trained_gpt=use_trained_gpt if not mock else False,
            few_shot=few_shot if not mock else False,
            gsv_few_shot_force=gsv_few_shot_force,
            rvc_train_force=rvc_train_force,
            voice_bank_path=voice_bank,
            require_voice_bank=require_voice_bank,
            merge_input_parts=merge_parts,
        )
    except Exception as exc:
        _handle_error(exc)
    console.print(f"Pipeline complete: {manifest.artifacts.get('export')}")


@app.command()
def full(
    input: Path = typer.Argument(..., help="Input media from ./audio or any local path."),
    project: Path | None = typer.Option(
        None,
        "--project",
        "-p",
        help="Project directory. Defaults to runs/<timestamp>_<input-stem>.",
    ),
    confirm_rights: bool = typer.Option(False, "--confirm-rights", help=RIGHTS_HELP),
    merge_parts: bool = typer.Option(
        False,
        "--merge-parts",
        help="Merge consecutive sibling files named <base>_1, <base>_2, ... into an audio-only source.",
    ),
    real: bool = typer.Option(
        False,
        "--real",
        help="Use real Gemma/GPT-SoVITS backends. Default uses deterministic mock backends.",
    ),
    gemma_backend: str = typer.Option(
        "llama_cpp",
        "--gemma-backend",
        help="hf|http|llama_cpp when --real is set.",
    ),
    ko_qc_backend: str | None = typer.Option(
        None,
        "--ko-qc-backend",
        help="Korean QC backend override for --real: auto|hf|http|llama_cpp|mock.",
    ),
    asr_backend: str | None = typer.Option(
        None,
        "--asr-backend",
        help="Override ASR backend for the transcribe stage: faster_whisper|qwen_asr|mock.",
    ),
    asr_preset: str | None = typer.Option(
        None,
        "--asr-preset",
        help="Runtime ASR preset for the transcribe stage: default|conservative|whisper|no_vad_repair.",
    ),
    asr_vad_off: bool = typer.Option(
        False,
        "--asr-vad-off",
        help="Disable VAD for the main ASR pass during full runs.",
    ),
    asr_diagnostics: bool | None = typer.Option(
        None,
        "--asr-diagnostics/--no-asr-diagnostics",
        help="Write unified ASR diagnostics artifacts during full runs.",
    ),
    asr_device: str | None = typer.Option(
        None,
        "--asr-device",
        help="Override faster-whisper device during full runs, e.g. auto|cuda|cpu.",
    ),
    asr_compute_type: str | None = typer.Option(
        None,
        "--asr-compute-type",
        help="Override faster-whisper compute type during full runs.",
    ),
    asr_batched: bool | None = typer.Option(
        None,
        "--asr-batched/--no-asr-batched",
        help="Use faster-whisper BatchedInferencePipeline during full runs.",
    ),
    asr_batch_size: int | None = typer.Option(
        None,
        "--asr-batch-size",
        help="Batch size for faster-whisper batched inference during full runs.",
    ),
    target_language: str = typer.Option("ko", "--target-language", help="Output TTS language. Currently supports ko/kr."),
    auto_repair: bool = typer.Option(
        True,
        "--auto-repair/--no-auto-repair",
        help="Run the Korean QC auto-repair loop after QC.",
    ),
    auto_repair_max_rounds: int | None = typer.Option(
        None,
        "--auto-repair-max-rounds",
        min=0,
        max=12,
        help="Override the auto-repair round budget.",
    ),
    micro_segments: bool = typer.Option(
        True,
        "--micro-segments/--no-micro-segments",
        help="Use dedicated GPT-SoVITS micro-segment routing.",
    ),
    gsv_url: str | None = typer.Option(None, "--gsv-url"),
    refs: Path = typer.Option(Path("refs/refs.json"), "--refs"),
    gpt_weights: str | None = typer.Option(None, "--gpt-weights", help="Optional GPT weights path for api_v2."),
    sovits_weights: str | None = typer.Option(None, "--sovits-weights", help="Optional SoVITS weights path for api_v2."),
    use_trained_gpt: bool = typer.Option(False, "--use-trained-gpt", help=TRAINED_GPT_HELP),
    auto_gsv_server: bool = typer.Option(
        True,
        "--auto-gsv-server/--no-auto-gsv-server",
        help="Start a local GPT-SoVITS api_v2 server if --real needs it and gsv_url is not already HTTP-ready.",
    ),
    gsv_server_command: str | None = typer.Option(
        None,
        "--gsv-server-command",
        help="Shell-style command used when --auto-gsv-server needs to start api_v2.",
    ),
    few_shot: bool = typer.Option(
        True,
        "--few-shot/--zero-shot",
        help="Fine-tune GPT-SoVITS from source segments when --real is set. Use --zero-shot to opt out.",
    ),
    gsv_few_shot_force: bool = typer.Option(
        True,
        "--force-few-shot/--reuse-few-shot",
        help="Re-run GPT-SoVITS few-shot training for maximum quality, or reuse cached weights.",
    ),
    rvc_train_force: bool = typer.Option(
        True,
        "--force-rvc-train/--reuse-rvc-train",
        help="Re-run RVC training for maximum quality, or reuse existing model artifacts.",
    ),
    cache_status: bool = typer.Option(True, "--cache-status/--no-cache-status"),
    source_separation_cache: Path | None = typer.Option(
        None,
        "--source-separation-cache",
        help=(
            "Voice-bank project to reuse cached source-separated stems from. "
            "Defaults to runs/voice_bank_all for --real when it exists."
        ),
    ),
    reuse_source_separation_cache: bool = typer.Option(
        True,
        "--reuse-source-separation-cache/--no-source-separation-cache",
        help="Import matching voice-bank source separation stems before running Demucs.",
    ),
    voice_bank: Path | None = typer.Option(
        None,
        "--voice-bank",
        help="Voice bank manifest path. Defaults to project voice_bank/voice_bank_manifest.json when required.",
    ),
    require_voice_bank: bool = typer.Option(
        False,
        "--require-voice-bank",
        help="Require speaker assignment and per-speaker SoVITS/RVC models before synthesis.",
    ),
) -> None:
    """Run the full end-to-end pipeline with sensible one-command defaults."""
    if not confirm_rights:
        _handle_error(RightsError(RIGHTS_MESSAGE))
    input_path = input.expanduser().resolve()
    project_dir = project.expanduser().resolve() if project else _default_full_project_dir(input_path, merge_parts=merge_parts)
    cache_lines = _configure_local_model_cache()
    if cache_status:
        for line in cache_lines:
            console.print(f"[dim]{line}[/dim]")
    if real:
        _apply_full_real_quality_preset(project_dir, target_language, input_path, few_shot=few_shot)
        console.print("[dim]Applied full --real high-quality preset to pipeline.yaml[/dim]")
    source_separation_cache_project = None
    if reuse_source_separation_cache:
        if source_separation_cache is not None:
            source_separation_cache_project = source_separation_cache.expanduser().resolve()
        elif real:
            default_cache_project = _default_voice_bank_project_dir()
            if default_cache_project.exists():
                source_separation_cache_project = default_cache_project.resolve()
    try:
        manifest = run_pipeline(
            input_path,
            project_dir,
            confirm_rights=confirm_rights,
            mock=not real,
            gemma_backend=gemma_backend if real else "mock",
            ko_qc_backend=ko_qc_backend,
            asr_backend=asr_backend,
            asr_preset=asr_preset,
            asr_vad_off=True if asr_vad_off else None,
            asr_diagnostics=asr_diagnostics,
            asr_device=asr_device,
            asr_compute_type=asr_compute_type,
            asr_batched_inference=asr_batched,
            asr_batch_size=asr_batch_size,
            target_language=target_language,
            auto_repair=auto_repair,
            auto_repair_max_rounds=auto_repair_max_rounds,
            micro_segments=micro_segments,
            gsv_url=gsv_url,
            refs_path=refs,
            gpt_weights_path=gpt_weights,
            sovits_weights_path=sovits_weights,
            auto_gsv_server=auto_gsv_server if real else False,
            gsv_server_command=gsv_server_command,
            use_trained_gpt=use_trained_gpt if real else False,
            few_shot=few_shot if real else False,
            gsv_few_shot_force=gsv_few_shot_force,
            rvc_train_force=rvc_train_force if real else False,
            voice_bank_path=voice_bank,
            require_voice_bank=require_voice_bank,
            source_separation_cache_project=source_separation_cache_project,
            regenerate_before_mix=real,
            merge_input_parts=merge_parts,
        )
    except Exception as exc:
        _handle_error(exc)
    console.print(f"Pipeline complete: {manifest.artifacts.get('export')}")
    console.print(f"Project: {project_dir}")


@app.command(name="full-audio-batch")
def full_audio_batch(
    audio_dir: Path = typer.Option(
        Path("audio"),
        "--audio-dir",
        help="Directory whose immediate subfolders are treated as works.",
    ),
    batch_dir: Path | None = typer.Option(
        None,
        "--batch-dir",
        "-o",
        help="Batch result directory. Defaults to runs/<timestamp>_audio_full_real_batch.",
    ),
    confirm_rights: bool = typer.Option(False, "--confirm-rights", help=RIGHTS_HELP),
    gemma_backend: str = typer.Option(
        "llama_cpp",
        "--gemma-backend",
        help="hf|http|llama_cpp.",
    ),
    ko_qc_backend: str | None = typer.Option(
        None,
        "--ko-qc-backend",
        help="Korean QC backend override: auto|hf|http|llama_cpp|mock.",
    ),
    asr_backend: str | None = typer.Option(
        None,
        "--asr-backend",
        help="Override ASR backend for the transcribe stage: faster_whisper|qwen_asr|mock.",
    ),
    asr_preset: str | None = typer.Option(
        None,
        "--asr-preset",
        help="Runtime ASR preset for the transcribe stage: default|conservative|whisper|no_vad_repair.",
    ),
    asr_vad_off: bool = typer.Option(
        False,
        "--asr-vad-off",
        help="Disable VAD for the main ASR pass.",
    ),
    asr_diagnostics: bool | None = typer.Option(
        None,
        "--asr-diagnostics/--no-asr-diagnostics",
        help="Write unified ASR diagnostics artifacts during each full run.",
    ),
    asr_device: str | None = typer.Option(
        None,
        "--asr-device",
        help="Override faster-whisper device, e.g. auto|cuda|cpu.",
    ),
    asr_compute_type: str | None = typer.Option(
        None,
        "--asr-compute-type",
        help="Override faster-whisper compute type.",
    ),
    asr_batched: bool | None = typer.Option(
        None,
        "--asr-batched/--no-asr-batched",
        help="Use faster-whisper BatchedInferencePipeline.",
    ),
    asr_batch_size: int | None = typer.Option(
        None,
        "--asr-batch-size",
        help="Batch size for faster-whisper batched inference.",
    ),
    target_language: str = typer.Option("ko", "--target-language", help="Output TTS language. Currently supports ko/kr."),
    auto_repair: bool = typer.Option(
        True,
        "--auto-repair/--no-auto-repair",
        help="Run the Korean QC auto-repair loop after QC.",
    ),
    auto_repair_max_rounds: int | None = typer.Option(
        None,
        "--auto-repair-max-rounds",
        min=0,
        max=12,
        help="Override the auto-repair round budget.",
    ),
    micro_segments: bool = typer.Option(
        True,
        "--micro-segments/--no-micro-segments",
        help="Use dedicated GPT-SoVITS micro-segment routing.",
    ),
    gsv_url: str | None = typer.Option(None, "--gsv-url"),
    refs: Path = typer.Option(Path("refs/refs.json"), "--refs"),
    gpt_weights: str | None = typer.Option(None, "--gpt-weights", help="Optional GPT weights path for api_v2."),
    sovits_weights: str | None = typer.Option(None, "--sovits-weights", help="Optional SoVITS weights path for api_v2."),
    use_trained_gpt: bool = typer.Option(False, "--use-trained-gpt", help=TRAINED_GPT_HELP),
    auto_gsv_server: bool = typer.Option(
        True,
        "--auto-gsv-server/--no-auto-gsv-server",
        help="Start a local GPT-SoVITS api_v2 server if gsv_url is not already HTTP-ready.",
    ),
    gsv_server_command: str | None = typer.Option(
        None,
        "--gsv-server-command",
        help="Shell-style command used when --auto-gsv-server needs to start api_v2.",
    ),
    few_shot: bool = typer.Option(
        True,
        "--few-shot/--zero-shot",
        help="Fine-tune GPT-SoVITS from source segments. Use --zero-shot to opt out.",
    ),
    gsv_few_shot_force: bool = typer.Option(
        True,
        "--force-few-shot/--reuse-few-shot",
        help="Re-run GPT-SoVITS few-shot training for maximum quality, or reuse cached weights.",
    ),
    rvc_train_force: bool = typer.Option(
        True,
        "--force-rvc-train/--reuse-rvc-train",
        help="Re-run RVC training for maximum quality, or reuse existing model artifacts.",
    ),
    cache_status: bool = typer.Option(True, "--cache-status/--no-cache-status"),
    source_separation_cache: Path | None = typer.Option(
        None,
        "--source-separation-cache",
        help=(
            "Voice-bank project to reuse cached source-separated stems from. "
            "Defaults to runs/voice_bank_all when it exists."
        ),
    ),
    reuse_source_separation_cache: bool = typer.Option(
        True,
        "--reuse-source-separation-cache/--no-source-separation-cache",
        help="Import matching voice-bank source separation stems before running Demucs.",
    ),
    voice_bank: Path | None = typer.Option(
        None,
        "--voice-bank",
        help="Voice bank manifest path. Defaults to project voice_bank/voice_bank_manifest.json when required.",
    ),
    require_voice_bank: bool = typer.Option(
        False,
        "--require-voice-bank",
        help="Require speaker assignment and per-speaker SoVITS/RVC models before synthesis.",
    ),
) -> None:
    """Run full --real for every work folder under ./audio and keep only final/review outputs."""
    if not confirm_rights:
        _handle_error(RightsError(RIGHTS_MESSAGE))
    try:
        inputs = _discover_audio_work_dirs(audio_dir)
        if not inputs:
            raise ValueError(f"No supported work folders found in {audio_dir}.")
    except Exception as exc:
        _handle_error(exc)

    batch_root = batch_dir.expanduser().resolve() if batch_dir else _default_full_audio_batch_dir()
    batch_root.mkdir(parents=True, exist_ok=True)
    cache_lines = _configure_local_model_cache()
    if cache_status:
        for line in cache_lines:
            console.print(f"[dim]{line}[/dim]")

    source_separation_cache_project = None
    if reuse_source_separation_cache:
        if source_separation_cache is not None:
            source_separation_cache_project = source_separation_cache.expanduser().resolve()
        else:
            default_cache_project = _default_voice_bank_project_dir()
            if default_cache_project.exists():
                source_separation_cache_project = default_cache_project.resolve()

    items: list[dict[str, object]] = []
    summary: dict[str, object] = {
        "audio_dir": str(audio_dir.expanduser().resolve()),
        "batch_dir": str(batch_root),
        "started_at": datetime.now(UTC).isoformat(),
        "items": items,
    }

    def write_summary() -> None:
        summary["updated_at"] = datetime.now(UTC).isoformat()
        write_json_atomic(batch_root / "batch_summary.json", summary)

    console.print(f"[cyan]full-audio-batch[/cyan] discovered {len(inputs)} work folder(s).")
    failed_count = 0
    for index, input_path in enumerate(inputs, start=1):
        run_name = f"{index:03d}_{_safe_run_name(input_path)}"
        project_dir = batch_root / "_projects" / run_name
        item: dict[str, object] = {
            "index": index,
            "input": str(input_path),
            "run_name": run_name,
            "project": str(project_dir),
            "started_at": datetime.now(UTC).isoformat(),
        }
        items.append(item)
        console.print(f"[cyan]full-audio-batch[/cyan] {index}/{len(inputs)} {input_path.name}")
        try:
            _apply_full_real_quality_preset(project_dir, target_language, input_path, few_shot=few_shot)
            manifest = run_pipeline(
                input_path,
                project_dir,
                confirm_rights=confirm_rights,
                mock=False,
                gemma_backend=gemma_backend,
                ko_qc_backend=ko_qc_backend,
                asr_backend=asr_backend,
                asr_preset=asr_preset,
                asr_vad_off=True if asr_vad_off else None,
                asr_diagnostics=asr_diagnostics,
                asr_device=asr_device,
                asr_compute_type=asr_compute_type,
                asr_batched_inference=asr_batched,
                asr_batch_size=asr_batch_size,
                target_language=target_language,
                auto_repair=auto_repair,
                auto_repair_max_rounds=auto_repair_max_rounds,
                micro_segments=micro_segments,
                gsv_url=gsv_url,
                refs_path=refs,
                gpt_weights_path=gpt_weights,
                sovits_weights_path=sovits_weights,
                auto_gsv_server=auto_gsv_server,
                gsv_server_command=gsv_server_command,
                use_trained_gpt=use_trained_gpt,
                few_shot=few_shot,
                gsv_few_shot_force=gsv_few_shot_force,
                rvc_train_force=rvc_train_force,
                voice_bank_path=voice_bank,
                require_voice_bank=require_voice_bank,
                source_separation_cache_project=source_separation_cache_project,
                regenerate_before_mix=True,
                merge_input_parts=False,
            )
            manual_review_count = sum(
                1 for segment in manifest.segments if segment.status == "needs_manual_review"
            )
            if manual_review_count:
                review_dir = batch_root / "needs_manual_review" / run_name
                copied = _copy_manual_review_bundle(
                    input_path=input_path,
                    project_dir=project_dir,
                    review_dir=review_dir,
                    manifest=manifest,
                )
                item.update(
                    {
                        "status": "needs_manual_review",
                        "manual_review_count": manual_review_count,
                        "result_dir": str(review_dir),
                        "copied_files": copied,
                        "completed_at": datetime.now(UTC).isoformat(),
                    }
                )
                console.print(
                    f"[yellow]needs_manual_review[/yellow] {input_path.name}: "
                    f"{manual_review_count} segment(s). Kept {review_dir}"
                )
            else:
                result_dir = batch_root / "completed" / run_name
                copied = _copy_export_result(manifest.artifacts.get("export"), result_dir)
                item.update(
                    {
                        "status": "completed",
                        "result_dir": str(result_dir),
                        "copied_files": copied,
                        "completed_at": datetime.now(UTC).isoformat(),
                    }
                )
                console.print(f"[green]completed[/green] {input_path.name}: kept {result_dir}")
        except Exception as exc:
            failed_count += 1
            failure_dir = batch_root / "failed" / run_name
            failure_dir.mkdir(parents=True, exist_ok=True)
            write_json_atomic(
                failure_dir / "error.json",
                {
                    "input": str(input_path),
                    "project": str(project_dir),
                    "error": str(exc),
                    "failed_at": datetime.now(UTC).isoformat(),
                },
            )
            item.update(
                {
                    "status": "failed",
                    "result_dir": str(failure_dir),
                    "error": str(exc),
                    "completed_at": datetime.now(UTC).isoformat(),
                }
            )
            console.print(f"[red]failed[/red] {input_path.name}: {exc}")
        finally:
            _remove_tree(project_dir)
            write_summary()

    console.print(f"Batch summary: {batch_root / 'batch_summary.json'}")
    if failed_count:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
