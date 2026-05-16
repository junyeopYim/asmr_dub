from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from asmr_dub_pipeline import cli as cli_module
from asmr_dub_pipeline.cli import app
from asmr_dub_pipeline.schemas import PipelineManifest


def _manifest() -> PipelineManifest:
    return PipelineManifest(artifacts={"export": "out.wav"})


def test_countdown_synth_cli_clears_gpu_vram_after_real_stage(
    cli_runner,
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cleanup_calls: list[str] = []
    monkeypatch.setattr(cli_module, "clear_gpu_vram", cleanup_calls.append, raising=False)
    monkeypatch.setattr(cli_module, "countdown_synth_step", lambda *args, **kwargs: _manifest())

    result = cli_runner.invoke(
        app,
        ["countdown-synth", "-p", str(tmp_project_dir), "--confirm-rights"],
    )

    assert result.exit_code == 0, result.output
    assert cleanup_calls == ["countdown-synth"]


def test_synth_cli_clears_gpu_vram_when_stage_fails(
    cli_runner,
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cleanup_calls: list[str] = []

    def fail_synth_step(*args: object, **kwargs: object) -> PipelineManifest:
        _ = args, kwargs
        raise RuntimeError("synth exploded")

    monkeypatch.setattr(cli_module, "clear_gpu_vram", cleanup_calls.append, raising=False)
    monkeypatch.setattr(cli_module, "synth_step", fail_synth_step)

    result = cli_runner.invoke(
        app,
        ["synth", "-p", str(tmp_project_dir), "--confirm-rights"],
    )

    assert result.exit_code == 1
    assert "synth exploded" in result.output
    assert cleanup_calls == ["synth"]


@pytest.mark.parametrize(
    ("step_attr", "argv", "stage_name"),
    [
        ("source_separation_step", ["separate-background", "--confirm-rights"], "source-separation"),
        ("transcribe_step", ["transcribe", "--asr-backend", "qwen_asr", "--confirm-rights"], "transcribe"),
        ("analyze_step", ["analyze", "--gemma-backend", "hf", "--confirm-rights"], "analyze"),
        (
            "audio_style_step",
            ["audio-style", "--gemma-backend", "llama_server_audio", "--confirm-rights"],
            "audio-style",
        ),
        ("translate_ko_step", ["translate-ko", "--gemma-text-backend", "llama_server", "--confirm-rights"], "translate-ko"),
        ("script_step", ["script", "--gemma-backend", "hf", "--confirm-rights"], "script"),
        ("synth_qwen_step", ["synth-qwen", "--confirm-rights"], "synth-qwen"),
        ("regenerate_needs_step", ["regenerate", "--tts-backend", "gpt-sovits", "--confirm-rights"], "regenerate"),
        ("rvc_train_step", ["train-rvc", "--confirm-rights"], "train-rvc"),
        ("rvc_step", ["rvc", "--confirm-rights"], "rvc"),
        ("gsv_few_shot_step", ["train-gsv", "--confirm-rights"], "train-gsv"),
        ("source_speakers_step", ["source-speakers", "--backend", "pyannote", "--confirm-rights"], "source-speakers"),
        ("qc_step", ["qc", "--gemma-backend", "hf", "--confirm-rights"], "qc"),
    ],
)
def test_gpu_capable_standalone_cli_clears_gpu_vram_after_stage(
    cli_runner,
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    step_attr: str,
    argv: list[str],
    stage_name: str,
) -> None:
    cleanup_calls: list[str] = []
    monkeypatch.setattr(cli_module, "clear_gpu_vram", cleanup_calls.append, raising=False)
    monkeypatch.setattr(cli_module, step_attr, lambda *args, **kwargs: _manifest())

    result = cli_runner.invoke(app, [*argv, "-p", str(tmp_project_dir)])

    assert result.exit_code == 0, result.output
    assert cleanup_calls == [stage_name]


def test_voice_bank_build_cli_clears_gpu_vram_after_real_stage(
    cli_runner,
    tmp_path: Path,
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cleanup_calls: list[str] = []
    monkeypatch.setattr(cli_module, "clear_gpu_vram", cleanup_calls.append, raising=False)
    monkeypatch.setattr(cli_module, "_configure_local_model_cache", lambda: [])
    monkeypatch.setattr(cli_module, "_apply_personal_voice_bank_defaults", lambda _project_dir: None)
    monkeypatch.setattr(
        cli_module,
        "build_voice_bank",
        lambda *args, **kwargs: SimpleNamespace(speakers={"speaker_01": object()}),
    )

    result = cli_runner.invoke(
        app,
        [
            "voice-bank-build",
            str(tmp_path / "input.wav"),
            "-p",
            str(tmp_project_dir),
            "--confirm-rights",
        ],
    )

    assert result.exit_code == 0, result.output
    assert cleanup_calls == ["voice-bank-build"]


def test_mock_countdown_synth_cli_skips_gpu_vram_cleanup(
    cli_runner,
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cleanup_calls: list[str] = []
    monkeypatch.setattr(cli_module, "clear_gpu_vram", cleanup_calls.append, raising=False)
    monkeypatch.setattr(cli_module, "countdown_synth_step", lambda *args, **kwargs: _manifest())

    result = cli_runner.invoke(
        app,
        ["countdown-synth", "-p", str(tmp_project_dir), "--mock", "--confirm-rights"],
    )

    assert result.exit_code == 0, result.output
    assert cleanup_calls == []
