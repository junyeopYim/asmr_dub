from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

from asmr_dub_pipeline.audio import ffmpeg


def _load_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "asr_audio_folders.py"
    spec = importlib.util.spec_from_file_location("asr_audio_folders_script", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_audit_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "asr_audit_reports.py"
    spec = importlib.util.spec_from_file_location("asr_audit_reports_script", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_asr_audio_folders_defaults_to_system_memory_lean_batched_transcribe_command() -> None:
    module = _load_script_module()
    args = module.parse_args(["audio", "--confirm-rights"])
    item = module.WorkItem(
        source_dir=Path("/input/audio/work"),
        project_dir=Path("/runs/asr_only/batch/work"),
        reason="test",
    )

    command = module.command_for_transcribe(args, item)

    assert "--asr-batched" in command
    assert "--no-asr-batched" not in command
    assert command[command.index("--asr-batch-size") + 1] == "16"
    assert "--no-asr-diagnostics" in command
    assert "--asr-diagnostics" not in command
    assert "--no-asr-repair" in command
    assert args.source_separation_backend == "none"
    assert args.disable_asr_repair is True


def test_run_batch_applies_low_resource_project_config_between_extract_and_transcribe(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_script_module()
    source_dir = tmp_path / "audio" / "work"
    source_dir.mkdir(parents=True)
    project_dir = tmp_path / "runs" / "batch" / "work"
    item = module.WorkItem(source_dir=source_dir, project_dir=project_dir, reason="test")
    events: list[str] = []

    args = module.parse_args(
        [
            str(tmp_path / "audio"),
            "--confirm-rights",
            "--batch-id",
            "batch",
            "--runs-root",
            str(tmp_path / "runs"),
        ]
    )

    args.in_process = False
    monkeypatch.setattr(module, "discover_audio_folders", lambda *args, **kwargs: [item])
    monkeypatch.setattr(module, "_manifest_transcribe_completed", lambda project: False)
    monkeypatch.setattr(module, "command_for_extract", lambda work_item: ["extract"])
    monkeypatch.setattr(module, "command_for_transcribe", lambda parsed, work_item: ["transcribe"])

    def fake_run(command, *, cwd, dry_run):
        events.append(command[0])
        return 0

    def fake_apply(project, parsed):
        events.append(
            "config:"
            f"{project == project_dir}:"
            f"{parsed.source_separation_backend}:"
            f"{parsed.asr_diagnostics}:"
            f"{parsed.disable_asr_repair}"
        )

    monkeypatch.setattr(module, "_run", fake_run)
    monkeypatch.setattr(module, "_apply_asr_only_resource_config", fake_apply, raising=False)

    assert module.run_batch(args) == 0
    assert events == ["extract", "config:True:none:False:True", "transcribe"]


def test_run_batch_defaults_to_in_process_and_reuses_asr_backend_factory(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_script_module()
    source_root = tmp_path / "audio"
    first_source = source_root / "first"
    second_source = source_root / "second"
    first_source.mkdir(parents=True)
    second_source.mkdir()
    first = module.WorkItem(
        source_dir=first_source,
        project_dir=tmp_path / "runs" / "batch" / "first",
        reason="test",
    )
    second = module.WorkItem(
        source_dir=second_source,
        project_dir=tmp_path / "runs" / "batch" / "second",
        reason="test",
    )
    args = module.parse_args(
        [
            str(source_root),
            "--confirm-rights",
            "--batch-id",
            "batch",
            "--runs-root",
            str(tmp_path / "runs"),
        ]
    )
    events: list[tuple[str, object]] = []
    factories: list[object] = []

    assert args.in_process is True
    monkeypatch.setattr(module, "discover_audio_folders", lambda *args, **kwargs: [first, second])
    monkeypatch.setattr(module, "_manifest_transcribe_completed", lambda project: False)
    monkeypatch.setattr(module, "_run", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("_run called")))
    monkeypatch.setattr(
        module,
        "extract_step",
        lambda input_path, project_dir, confirm_rights: events.append(("extract", project_dir)),
        raising=False,
    )
    monkeypatch.setattr(
        module,
        "_apply_asr_only_resource_config",
        lambda project_dir, parsed: events.append(("config", project_dir)),
        raising=False,
    )

    def fake_transcribe_step(project_dir: Path, *args: object, **kwargs: object) -> None:
        factory = kwargs.get("asr_backend_factory")
        factories.append(factory)
        events.append(("transcribe", project_dir))

    monkeypatch.setattr(module, "transcribe_step", fake_transcribe_step, raising=False)

    assert module.run_batch(args) == 0
    assert events == [
        ("extract", first.project_dir),
        ("config", first.project_dir),
        ("transcribe", first.project_dir),
        ("extract", second.project_dir),
        ("config", second.project_dir),
        ("transcribe", second.project_dir),
    ]
    assert factories[0] is not None
    assert factories[0] is factories[1]


def test_ffmpeg_concat_audio_to_wav_with_silence_uses_lavfi_without_python_concat(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls: list[list[str]] = []

    monkeypatch.setattr(
        ffmpeg,
        "probe_media",
        lambda path: SimpleNamespace(duration_sec=2.5),
    )
    monkeypatch.setattr(ffmpeg, "run_ffmpeg", lambda args: calls.append(args))

    first = tmp_path / "first.wav"
    silent = tmp_path / "silent.wav"
    output = tmp_path / "out.wav"
    ffmpeg.concat_audio_to_wav_with_silence(
        [first, silent],
        output,
        silent_paths=[silent],
        sample_rate=16_000,
        channels=1,
    )

    assert calls
    args = calls[0]
    assert "lavfi" in args
    assert any(value == "anullsrc=r=16000:cl=mono" for value in args)
    assert any("concat=n=2:v=0:a=1[a]" in value for value in args)
    assert str(output) == args[-1]


def test_asr_audit_reports_aggregates_warning_and_blocking_patterns(tmp_path: Path) -> None:
    module = _load_audit_module()
    first_report = tmp_path / "run_a" / "work" / "transcribe" / "asr_high_risk_report.json"
    second_report = tmp_path / "run_b" / "work" / "transcribe" / "asr_high_risk_report.json"
    first_report.parent.mkdir(parents=True)
    second_report.parent.mkdir(parents=True)
    first_report.write_text(
        """
        {
          "summary": {"severe": 1, "warning": 1},
          "items": [
            {
              "segment_id": "seg_0001",
              "decision": "needs_review",
              "severity": "severe",
              "reasons": ["asr_suspicious_pattern:悪夢し"],
              "replacement_hits": [{"source": "悪夢し", "target": "アクメし", "count": 1}],
              "source_text": "悪夢していいですよ"
            },
            {
              "segment_id": "seg_0002",
              "decision": "sparse_speech_unverified",
              "severity": "warning",
              "reasons": ["asr_sparse_speech_unverified"],
              "replacement_hits": [],
              "source_text": "今の姿勢のまま"
            }
          ]
        }
        """,
        "utf-8",
    )
    second_report.write_text(
        """
        {
          "summary": {"severe": 1, "warning": 0},
          "items": [
            {
              "segment_id": "seg_0003",
              "decision": "needs_review",
              "severity": "severe",
              "reasons": ["asr_suspicious_pattern:悪夢し"],
              "replacement_hits": [],
              "source_text": "悪夢し続けます"
            }
          ]
        }
        """,
        "utf-8",
    )

    report = module.build_audit_report([tmp_path])

    assert report["summary"]["report_count"] == 2
    assert report["summary"]["item_count"] == 3
    assert report["summary"]["severe"] == 2
    assert report["summary"]["warning"] == 1
    assert report["reason_counts"]["asr_suspicious_pattern:悪夢し"] == 2
    assert report["decision_counts"]["needs_review"] == 2
    assert report["replacement_source_counts"]["悪夢し"] == 1
    assert report["top_review_candidates"][0]["reason"] == "asr_suspicious_pattern:悪夢し"
