from __future__ import annotations

import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import typer
from rich.table import Table

from asmr_dub_pipeline.audio.preprocess import numbered_part_base_stem
from asmr_dub_pipeline.gemma.llama_cpp_client import (
    DEFAULT_LLAMA_CPP_CLI,
    DEFAULT_LLAMA_CPP_MMPROJ,
    DEFAULT_LLAMA_CPP_MODEL,
)

from .config import load_project_config, save_project_config
from .logging import console
from .orchestrator import run_pipeline
from .pipeline.manifest_io import manifest_path
from .pipeline.steps import (
    analyze_step,
    export_step,
    extract_step,
    gsv_few_shot_step,
    init_project,
    inspect_input,
    mix_step,
    qc_step,
    regenerate_needs_step,
    rvc_step,
    rvc_train_step,
    script_step,
    segment_step,
    source_separation_step,
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
FULL_REAL_QUALITY_PRESET = {
    "source_language": "ja",
    "asr_preset": "whisper",
    "asr_batched_inference": True,
    "asr_batch_size": 16,
    "asr_diagnostics_enabled": True,
    "asr_resegment_max_sec": 14.0,
    "asr_review_enabled": True,
    "asr_review_generate_candidates": True,
    "asr_translation_backcheck_enabled": True,
    "candidate_count": 3,
    "duration_tolerance": 0.25,
    "gemma_llama_cpp_ctx_size": 16384,
    "gemma_llama_cpp_n_predict": 2048,
    "gemma_text_batch_size": 1,
    "gemma_text_context_radius": 8,
    "gemma_text_concurrency": 4,
    "gemma_text_n_predict": 2048,
    "gemma_text_retries": 2,
    "gemma_text_span_size": 12,
    "gemma_text_span_max_sec": 90.0,
    "gemma_text_span_max_gap_sec": 3.0,
    "gemma_text_timeout_sec": 900.0,
    "gemma_text_server_startup_timeout_sec": 900.0,
    "gsv_timeout_sec": 240.0,
    "gsv_retries": 3,
    "gsv_concurrency": 3,
    "gsv_few_shot_target_sec": 180.0,
    "gsv_few_shot_min_clip_sec": 2.0,
    "gsv_few_shot_max_clip_sec": 8.0,
    "gsv_few_shot_min_quality_score": 0.35,
    "gsv_ref_min_quality_score": 0.40,
    "gsv_tts_max_speed_factor": 1.0,
    "gsv_tts_min_speed_factor": 0.92,
    "gsv_top_k": 8,
    "gsv_top_p": 0.9,
    "gsv_temperature": 0.7,
    "gsv_text_split_method": "cut0",
    "gsv_parallel_infer": False,
    "gsv_repetition_penalty": 1.25,
    "gsv_sample_steps": 32,
    "gsv_super_sampling": True,
    "gsv_min_chunk_length": 8,
    "gsv_gpt_weights_policy": "auto",
    "gsv_sovits_weights_policy": "auto",
    "mix_allow_korean_timing_draft": False,
    "rvc_required": True,
    "rvc_backend": "command",
    "rvc_train_required": True,
    "rvc_train_backend": "command",
    "rvc_train_epochs": 20,
    "rvc_train_batch_size": 0,
    "rvc_train_timeout_sec": 43200.0,
    "rvc_train_preprocess_processes": 0,
    "rvc_train_f0_workers": 0,
    "rvc_train_feature_workers": 0,
    "rvc_train_save_every_epoch": 50,
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
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_cache / "transformers"))
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


def _apply_personal_voice_bank_defaults(project_dir: Path) -> None:
    init_project(project_dir)
    cfg = load_project_config(project_dir)
    payload = cfg.model_dump(mode="json")
    payload.update(
        {
            "speaker_assignment_backend": "pyannote",
            "diarization_auto_download": True,
            "diarization_embedding_match_threshold": 0.78,
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
) -> None:
    init_project(project_dir)
    cfg = load_project_config(project_dir)
    payload = cfg.model_dump(mode="json")
    payload.update(FULL_REAL_QUALITY_PRESET)
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
        manifest = source_separation_step(
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
        analyze_step(project.expanduser().resolve(), gemma_backend, model_id, confirm_rights=confirm_rights)
    except Exception as exc:
        _handle_error(exc)
    console.print("Analysis complete.")


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
    asr_review: bool = typer.Option(
        False,
        "--asr-review",
        help="Use the configured Gemma audio+text model to review suspicious ASR candidates.",
    ),
    confirm_rights: bool = typer.Option(False, "--confirm-rights", help=RIGHTS_HELP),
) -> None:
    """Create segment-level source scripts with local ASR."""
    try:
        transcribe_step(
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
        translate_ko_step(
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


@app.command(name="script")
def script_cmd(
    project: Path = typer.Option(..., "--project", "-p"),
    gemma_backend: str = typer.Option("mock", "--gemma-backend", help="mock|hf|http|llama_cpp"),
    confirm_rights: bool = typer.Option(False, "--confirm-rights", help=RIGHTS_HELP),
) -> None:
    """Generate Japanese ASMR script metadata and normalize TTS text."""
    try:
        script_step(project.expanduser().resolve(), gemma_backend, confirm_rights=confirm_rights)
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
) -> None:
    """Generate TTS candidates per segment."""
    try:
        synth_step(
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
        )
    except RightsError as exc:
        _handle_error(exc)
    except Exception as exc:
        _handle_error(exc)
    console.print("Synthesis complete.")


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
        manifest = synth_qwen_step(
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
        manifest = regenerate_needs_step(
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


@app.command(name="train-rvc")
def train_rvc(
    project: Path = typer.Option(..., "--project", "-p"),
    confirm_rights: bool = typer.Option(False, "--confirm-rights", help=RIGHTS_HELP),
    force: bool = typer.Option(False, "--force", help="Re-run RVC training even if artifacts exist."),
) -> None:
    """Train the required RVC voice model from source-derived segment audio."""
    try:
        manifest = rvc_train_step(project.expanduser().resolve(), confirm_rights=confirm_rights, force=force)
    except RightsError as exc:
        _handle_error(exc)
    except Exception as exc:
        _handle_error(exc)
    console.print(f"RVC training complete: {manifest.artifacts.get('rvc_train_manifest')}")


@app.command()
def rvc(
    project: Path = typer.Option(..., "--project", "-p"),
    confirm_rights: bool = typer.Option(False, "--confirm-rights", help=RIGHTS_HELP),
    force: bool = typer.Option(False, "--force", help="Re-run RVC even if candidate outputs exist."),
) -> None:
    """Run mandatory RVC timbre correction on selected TTS outputs."""
    try:
        manifest = rvc_step(project.expanduser().resolve(), confirm_rights=confirm_rights, force=force)
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
        bank = build_voice_bank(
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
        bank = build_voice_bank(
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
        manifest = gsv_few_shot_step(
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


@app.command()
def qc(
    project: Path = typer.Option(..., "--project", "-p"),
    gemma_backend: str = typer.Option("mock", "--gemma-backend", help="mock|hf|http|llama_cpp"),
    confirm_rights: bool = typer.Option(False, "--confirm-rights", help=RIGHTS_HELP),
) -> None:
    """Run audio and Gemma-style QC."""
    try:
        qc_step(project.expanduser().resolve(), gemma_backend, confirm_rights=confirm_rights)
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
    target_language: str = typer.Option("ko", "--target-language", help="Output TTS language. Currently supports ko/kr."),
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
        True,
        "--few-shot/--zero-shot",
        help="Fine-tune GPT-SoVITS from source segments before real synthesis.",
    ),
    gsv_few_shot_force: bool = typer.Option(
        False,
        "--force-few-shot",
        help="Re-run GPT-SoVITS few-shot training even when cached weights match.",
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
            target_language=target_language,
            gsv_url=gsv_url,
            refs_path=refs,
            gpt_weights_path=gpt_weights,
            sovits_weights_path=sovits_weights,
            auto_gsv_server=auto_gsv_server,
            gsv_server_command=gsv_server_command,
            use_trained_gpt=use_trained_gpt if not mock else False,
            few_shot=few_shot if not mock else False,
            gsv_few_shot_force=gsv_few_shot_force,
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
        help="Fine-tune GPT-SoVITS from source segments when --real is set.",
    ),
    gsv_few_shot_force: bool = typer.Option(
        True,
        "--force-few-shot/--reuse-few-shot",
        help="Re-run GPT-SoVITS few-shot training for maximum quality, or reuse cached weights.",
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
        _apply_full_real_quality_preset(project_dir, target_language, input_path)
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
            asr_backend=asr_backend,
            asr_preset=asr_preset,
            asr_vad_off=True if asr_vad_off else None,
            asr_diagnostics=asr_diagnostics,
            asr_device=asr_device,
            asr_compute_type=asr_compute_type,
            asr_batched_inference=asr_batched,
            asr_batch_size=asr_batch_size,
            target_language=target_language,
            gsv_url=gsv_url,
            refs_path=refs,
            gpt_weights_path=gpt_weights,
            sovits_weights_path=sovits_weights,
            auto_gsv_server=auto_gsv_server if real else False,
            gsv_server_command=gsv_server_command,
            use_trained_gpt=use_trained_gpt if real else False,
            few_shot=few_shot if real else False,
            gsv_few_shot_force=gsv_few_shot_force,
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


if __name__ == "__main__":
    app()
