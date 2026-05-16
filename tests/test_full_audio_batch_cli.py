from __future__ import annotations

import json
from pathlib import Path

import asmr_dub_pipeline.cli as cli_module
from asmr_dub_pipeline.cli import app
from asmr_dub_pipeline.schemas import PipelineManifest, Segment


def test_full_audio_batch_requires_confirm_rights(cli_runner, tmp_path: Path) -> None:
    audio_dir = tmp_path / "audio"
    (audio_dir / "RJ001").mkdir(parents=True)

    result = cli_runner.invoke(app, ["full-audio-batch", "--audio-dir", str(audio_dir)])

    assert result.exit_code != 0
    assert "permission/consent" in result.output


def test_full_audio_batch_runs_real_full_for_each_audio_folder_and_keeps_only_outputs(
    cli_runner,
    tmp_path: Path,
    monkeypatch,
) -> None:
    audio_dir = tmp_path / "audio"
    first = audio_dir / "RJ001"
    second = audio_dir / "RJ002"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "track.wav").write_bytes(b"audio")
    (second / "track.wav").write_bytes(b"audio")
    batch_dir = tmp_path / "batch"
    captured: list[tuple[Path, Path, dict[str, object]]] = []

    def fake_run_pipeline(input_path: Path, project_dir: Path, **kwargs: object) -> PipelineManifest:
        captured.append((input_path, project_dir, kwargs))
        output = project_dir / "output" / f"{input_path.name}_dub.wav"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(input_path.name.encode("utf-8"))
        (project_dir / "work" / "large_intermediate.bin").parent.mkdir(parents=True, exist_ok=True)
        (project_dir / "work" / "large_intermediate.bin").write_bytes(b"delete me")
        return PipelineManifest(artifacts={"export": str(output)})

    monkeypatch.setattr(cli_module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(cli_module, "run_pipeline", fake_run_pipeline)

    result = cli_runner.invoke(
        app,
        [
            "full-audio-batch",
            "--audio-dir",
            str(audio_dir),
            "--batch-dir",
            str(batch_dir),
            "--confirm-rights",
            "--no-cache-status",
        ],
    )

    assert result.exit_code == 0, result.output
    assert [item[0] for item in captured] == [first.resolve(), second.resolve()]
    assert all(item[2]["mock"] is False for item in captured)
    assert all(item[2]["regenerate_before_mix"] is True for item in captured)
    assert (batch_dir / "completed" / "001_RJ001" / "RJ001_dub.wav").read_bytes() == b"RJ001"
    assert (batch_dir / "completed" / "002_RJ002" / "RJ002_dub.wav").read_bytes() == b"RJ002"
    assert not (batch_dir / "_projects" / "001_RJ001").exists()
    assert not (batch_dir / "_projects" / "002_RJ002").exists()
    summary = json.loads((batch_dir / "batch_summary.json").read_text("utf-8"))
    assert [item["status"] for item in summary["items"]] == ["completed", "completed"]


def test_full_audio_batch_manual_review_keeps_only_manual_review_bundle(
    cli_runner,
    tmp_path: Path,
    monkeypatch,
) -> None:
    audio_dir = tmp_path / "audio"
    work = audio_dir / "RJMANUAL"
    work.mkdir(parents=True)
    (work / "track.wav").write_bytes(b"audio")
    batch_dir = tmp_path / "batch"

    def fake_run_pipeline(input_path: Path, project_dir: Path, **kwargs: object) -> PipelineManifest:
        _ = input_path, kwargs
        manual_audio = project_dir / "work" / "segments" / "audio" / "manual.wav"
        ok_audio = project_dir / "work" / "segments" / "audio" / "ok.wav"
        manual_audio.parent.mkdir(parents=True, exist_ok=True)
        manual_audio.write_bytes(b"manual")
        ok_audio.write_bytes(b"ok")
        output = project_dir / "output" / "should_not_survive.wav"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"not final")
        return PipelineManifest(
            artifacts={"export": str(output)},
            segments=[
                Segment(
                    id="seg_0001",
                    start=0.0,
                    end=1.0,
                    duration=1.0,
                    audio_for_gemma=str(manual_audio),
                    audio_for_mix=str(manual_audio),
                    status="needs_manual_review",
                    errors=["review this"],
                ),
                Segment(
                    id="seg_0002",
                    start=1.0,
                    end=2.0,
                    duration=1.0,
                    audio_for_gemma=str(ok_audio),
                    audio_for_mix=str(ok_audio),
                    status="ok",
                ),
            ],
        )

    monkeypatch.setattr(cli_module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(cli_module, "run_pipeline", fake_run_pipeline)

    result = cli_runner.invoke(
        app,
        [
            "full-audio-batch",
            "--audio-dir",
            str(audio_dir),
            "--batch-dir",
            str(batch_dir),
            "--confirm-rights",
            "--no-cache-status",
        ],
    )

    assert result.exit_code == 0, result.output
    review_dir = batch_dir / "needs_manual_review" / "001_RJMANUAL"
    review_manifest = json.loads((review_dir / "manual_review_segments.json").read_text("utf-8"))
    assert [segment["id"] for segment in review_manifest["segments"]] == ["seg_0001"]
    assert (review_dir / "segments" / "seg_0001" / "audio_for_gemma.wav").read_bytes() == b"manual"
    assert not (review_dir / "segments" / "seg_0002").exists()
    assert not (batch_dir / "completed").exists()
    assert not (batch_dir / "_projects" / "001_RJMANUAL").exists()
