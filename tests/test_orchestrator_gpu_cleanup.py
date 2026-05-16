from __future__ import annotations

from pathlib import Path

import pytest

import asmr_dub_pipeline.orchestrator as orchestrator
from asmr_dub_pipeline.schemas import PipelineManifest


def _install_pipeline_step_stubs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    fail_stage: str | None = None,
) -> list[str]:
    calls: list[str] = []

    def step(name: str):
        def inner(*args: object, **kwargs: object) -> PipelineManifest:
            _ = args, kwargs
            calls.append(name)
            if name == fail_stage:
                raise RuntimeError(f"{name} failed")
            return PipelineManifest(artifacts={"export": "out.wav"} if name == "export" else {})

        return inner

    def fake_init(path: Path) -> PipelineManifest:
        Path(path).mkdir(parents=True, exist_ok=True)
        calls.append("init")
        return PipelineManifest()

    monkeypatch.setattr(orchestrator, "init_project", fake_init)
    monkeypatch.setattr(orchestrator, "extract_step", step("extract"))
    monkeypatch.setattr(orchestrator, "source_separation_step", step("source-separation"))
    monkeypatch.setattr(orchestrator, "segment_step", step("segment"))
    monkeypatch.setattr(orchestrator, "transcribe_step", step("transcribe"))
    monkeypatch.setattr(orchestrator, "source_speakers_step", step("source-speakers"))
    monkeypatch.setattr(orchestrator, "audio_style_step", step("audio-style"))
    monkeypatch.setattr(orchestrator, "translate_ko_step", step("translate-ko"))
    monkeypatch.setattr(orchestrator, "korean_script_step", step("korean-script"))
    monkeypatch.setattr(orchestrator, "prepare_source_voice_refs_step", step("prepare-refs"))
    monkeypatch.setattr(orchestrator, "synth_step", step("synth"))
    monkeypatch.setattr(orchestrator, "countdown_synth_step", step("countdown-synth"))
    monkeypatch.setattr(orchestrator, "rvc_train_step", step("train-rvc"))
    monkeypatch.setattr(orchestrator, "rvc_step", step("rvc"))
    monkeypatch.setattr(orchestrator, "qc_step", step("qc"))
    monkeypatch.setattr(orchestrator, "regenerate_needs_step", step("regenerate"))
    monkeypatch.setattr(orchestrator, "mix_step", step("mix"))
    monkeypatch.setattr(orchestrator, "export_step", step("export"))
    monkeypatch.setattr(orchestrator, "validate_rvc_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(orchestrator, "validate_rvc_training_config", lambda *args, **kwargs: None)
    return calls


def test_real_pipeline_clears_gpu_vram_after_each_stage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = _install_pipeline_step_stubs(monkeypatch)
    cleanup_calls: list[str] = []
    monkeypatch.setattr(orchestrator, "clear_gpu_vram", cleanup_calls.append, raising=False)

    orchestrator.run_pipeline(
        tmp_path / "input.wav",
        tmp_path / "project",
        confirm_rights=True,
        mock=False,
        gemma_backend="llama_cpp",
        few_shot=False,
        regenerate_before_mix=True,
    )

    assert calls == [
        "init",
        "extract",
        "source-separation",
        "transcribe",
        "segment",
        "source-speakers",
        "audio-style",
        "translate-ko",
        "korean-script",
        "prepare-refs",
        "synth",
        "countdown-synth",
        "train-rvc",
        "rvc",
        "qc",
        "regenerate",
        "mix",
        "export",
    ]
    assert cleanup_calls == calls[1:]


def test_mock_pipeline_does_not_clear_gpu_vram(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_pipeline_step_stubs(monkeypatch)
    cleanup_calls: list[str] = []
    monkeypatch.setattr(orchestrator, "clear_gpu_vram", cleanup_calls.append, raising=False)

    orchestrator.run_pipeline(
        tmp_path / "input.wav",
        tmp_path / "project",
        confirm_rights=True,
        mock=True,
    )

    assert cleanup_calls == []


def test_real_pipeline_clears_gpu_vram_when_stage_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_pipeline_step_stubs(monkeypatch, fail_stage="source-separation")
    cleanup_calls: list[str] = []
    monkeypatch.setattr(orchestrator, "clear_gpu_vram", cleanup_calls.append, raising=False)

    with pytest.raises(RuntimeError, match="source-separation failed"):
        orchestrator.run_pipeline(
            tmp_path / "input.wav",
            tmp_path / "project",
            confirm_rights=True,
            mock=False,
            gemma_backend="llama_cpp",
            few_shot=False,
        )

    assert cleanup_calls == ["extract", "source-separation"]
