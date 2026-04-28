from __future__ import annotations

import os
import re
from datetime import UTC, datetime
from pathlib import Path

import typer
from rich.table import Table

from asmr_dub_pipeline.gemma.llama_cpp_client import (
    DEFAULT_LLAMA_CPP_CLI,
    DEFAULT_LLAMA_CPP_MMPROJ,
    DEFAULT_LLAMA_CPP_MODEL,
)

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
    script_step,
    segment_step,
    source_separation_step,
    synth_step,
    transcribe_step,
    translate_ko_step,
)
from .rights import RIGHTS_MESSAGE, RightsError, require_confirmed_rights

RIGHTS_HELP = (
    "Confirm you own or have permission/consent for the source content, voice "
    "references, and distribution."
)
REPO_ROOT = Path(__file__).resolve().parents[1]

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


def _safe_run_name(input_path: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "-", input_path.stem).strip("-._")
    return stem or "input"


def _default_full_project_dir(input_path: Path) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return REPO_ROOT / "runs" / f"{timestamp}_{_safe_run_name(input_path)}"


def _configure_local_model_cache() -> list[str]:
    """Point HF libraries at the repo-local cache when the user has not set one."""
    lines: list[str] = []
    hf_cache = REPO_ROOT / ".cache" / "huggingface"
    if hf_cache.exists():
        os.environ.setdefault("HF_HOME", str(hf_cache))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_cache / "transformers"))
        lines.append(f"HF cache: {hf_cache}")
        gemma_cache = hf_cache / "hub" / "models--google--gemma-4-E4B-it"
        lines.append(f"Gemma cache: {'found' if gemma_cache.exists() else 'missing'}")
    else:
        lines.append(f"HF cache: missing ({hf_cache})")
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
    llama_model = REPO_ROOT / DEFAULT_LLAMA_CPP_MODEL
    llama_mmproj = REPO_ROOT / DEFAULT_LLAMA_CPP_MMPROJ
    llama_cli = REPO_ROOT / DEFAULT_LLAMA_CPP_CLI
    lines.append(f"llama.cpp Gemma Q4: {'found' if llama_model.exists() else 'missing'}")
    lines.append(f"llama.cpp mmproj: {'found' if llama_mmproj.exists() else 'missing'}")
    lines.append(f"llama.cpp CLI: {'found' if llama_cli.exists() else 'missing'}")
    return lines


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
) -> None:
    """Extract stereo 48 kHz and Gemma mono 16 kHz audio."""
    try:
        extract_step(input.expanduser().resolve(), project.expanduser().resolve(), confirm_rights)
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
    """Create preliminary segment manifests."""
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
    asr_backend: str = typer.Option("faster_whisper", "--asr-backend", help="faster_whisper|mock"),
    confirm_rights: bool = typer.Option(False, "--confirm-rights", help=RIGHTS_HELP),
) -> None:
    """Create segment-level source scripts with local ASR."""
    try:
        transcribe_step(project.expanduser().resolve(), asr_backend, confirm_rights=confirm_rights)
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
) -> None:
    """Translate source scripts to Korean with text-only Gemma."""
    try:
        translate_ko_step(
            project.expanduser().resolve(),
            gemma_text_backend,
            confirm_rights=confirm_rights,
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
    auto_gsv_server: bool = typer.Option(
        False,
        "--auto-gsv-server/--no-auto-gsv-server",
        help="Start a local GPT-SoVITS api_v2 server if gsv_url is not already listening.",
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
            auto_gsv_server=auto_gsv_server,
            gsv_server_command=gsv_server_command,
        )
    except RightsError as exc:
        _handle_error(exc)
    except Exception as exc:
        _handle_error(exc)
    console.print("Synthesis complete.")


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
    mock: bool = typer.Option(False, "--mock", help="Use mock Gemma and mock TTS."),
    gemma_backend: str = typer.Option("mock", "--gemma-backend", help="mock|hf|http|llama_cpp"),
    target_language: str = typer.Option("ko", "--target-language", help="Output TTS language. Currently supports ko/kr."),
    gsv_url: str | None = typer.Option(None, "--gsv-url"),
    refs: Path = typer.Option(Path("refs/refs.json"), "--refs"),
    gpt_weights: str | None = typer.Option(None, "--gpt-weights", help="Optional GPT weights path for api_v2."),
    sovits_weights: str | None = typer.Option(None, "--sovits-weights", help="Optional SoVITS weights path for api_v2."),
    auto_gsv_server: bool = typer.Option(
        False,
        "--auto-gsv-server/--no-auto-gsv-server",
        help="Start a local GPT-SoVITS api_v2 server if gsv_url is not already listening.",
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
) -> None:
    """Run extract, segment, analyze, script, synth, QC, mix, and export."""
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
            few_shot=few_shot if not mock else False,
            gsv_few_shot_force=gsv_few_shot_force,
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
    real: bool = typer.Option(
        False,
        "--real",
        help="Use real Gemma/GPT-SoVITS backends. Default uses deterministic mock backends.",
    ),
    gemma_backend: str = typer.Option(
        "hf",
        "--gemma-backend",
        help="hf|http|llama_cpp when --real is set.",
    ),
    target_language: str = typer.Option("ko", "--target-language", help="Output TTS language. Currently supports ko/kr."),
    gsv_url: str | None = typer.Option(None, "--gsv-url"),
    refs: Path = typer.Option(Path("refs/refs.json"), "--refs"),
    gpt_weights: str | None = typer.Option(None, "--gpt-weights", help="Optional GPT weights path for api_v2."),
    sovits_weights: str | None = typer.Option(None, "--sovits-weights", help="Optional SoVITS weights path for api_v2."),
    auto_gsv_server: bool = typer.Option(
        True,
        "--auto-gsv-server/--no-auto-gsv-server",
        help="Start a local GPT-SoVITS api_v2 server if --real needs it and gsv_url is not already listening.",
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
        False,
        "--force-few-shot",
        help="Re-run GPT-SoVITS few-shot training even when cached weights match.",
    ),
    cache_status: bool = typer.Option(True, "--cache-status/--no-cache-status"),
) -> None:
    """Run the full end-to-end pipeline with sensible one-command defaults."""
    if not confirm_rights:
        _handle_error(RightsError(RIGHTS_MESSAGE))
    input_path = input.expanduser().resolve()
    project_dir = project.expanduser().resolve() if project else _default_full_project_dir(input_path)
    cache_lines = _configure_local_model_cache()
    if cache_status:
        for line in cache_lines:
            console.print(f"[dim]{line}[/dim]")
    try:
        manifest = run_pipeline(
            input_path,
            project_dir,
            confirm_rights=confirm_rights,
            mock=not real,
            gemma_backend=gemma_backend if real else "mock",
            target_language=target_language,
            gsv_url=gsv_url,
            refs_path=refs,
            gpt_weights_path=gpt_weights,
            sovits_weights_path=sovits_weights,
            auto_gsv_server=auto_gsv_server if real else False,
            gsv_server_command=gsv_server_command,
            few_shot=few_shot if real else False,
            gsv_few_shot_force=gsv_few_shot_force,
        )
    except Exception as exc:
        _handle_error(exc)
    console.print(f"Pipeline complete: {manifest.artifacts.get('export')}")
    console.print(f"Project: {project_dir}")


if __name__ == "__main__":
    app()
